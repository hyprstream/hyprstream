//! Iroh RPC plane — server-side protocol handler for ALPN `hyprstream-rpc/1`.
//!
//! Part of Epic #131 Phase 2 (#133). This module ships:
//!
//! - [`IrohRpcProtocolHandler`] — iroh [`ProtocolHandler`] that plugs into
//!   [`crate::transport::iroh_substrate`] under the `hyprstream-rpc/1` ALPN.
//!   Enforces a server-wide cap on concurrent streams (DoS bound) and
//!   drains in-flight requests on `ProtocolHandler::shutdown`.
//! - [`IrohRequestProcessor`] — sealed request processing trait; production
//!   network sessions admit only the canonical envelope/service bridge.
//! - [`LocalServiceBridge`] — adapts a [`crate::service::RequestService`]
//!   (potentially `!Send`) to the Send-bounded [`IrohRequestProcessor`].
//!
//! **Trust model**: The protocol handler does not parse `SignedEnvelope` —
//! it forwards opaque bytes to the [`IrohRequestProcessor`]. Envelope
//! verification, JWT/DPoP, and `authorize_signer` enforcement happen inside
//! the processor (which is `LocalServiceBridge` in production, delegating
//! to [`crate::service::dispatch::process_request`]). The handler does
//! hold a signing key for trusted-local transport-only processor failures.
//! Admission failures on the untrusted iroh carrier are reset/dropped without
//! a signed response, preventing an unauthenticated signing oracle.
//!
//! **Wire framing**: each request is the opaque bytes of a Cap'n Proto-encoded
//! [`crate::envelope::SignedEnvelope`] written to a freshly-opened iroh bidi
//! stream. The response is symmetric. Both endpoints are our code; iroh's
//! QUIC TLS authenticates the addressed NodeId only at the carrier layer; the
//! request envelope independently supplies application identity proof.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

// Transport-generic core lives in `super::rpc_session`. Re-export the shared
// surface from here so existing `iroh_rpc::{...}` imports keep working.
pub use super::rpc_session::{
    build_error_envelope, from_fn, read_to_cap, serve_rpc_connection, IrohRequestProcessor,
    DEFAULT_STREAM_LIMIT, MAX_FRAME_BYTES,
};

// ============================================================================
// IrohRpcProtocolHandler — iroh ProtocolHandler with concurrency cap + drain
// ============================================================================

/// Iroh protocol handler that terminates `hyprstream-rpc/1` bidi streams,
/// dispatches each request through the wrapped [`IrohRequestProcessor`],
/// caps concurrent streams server-wide (shared semaphore), and drains
/// in-flight requests
/// on `ProtocolHandler::shutdown` (called by `Router::shutdown`).
#[derive(Clone)]
pub struct IrohRpcProtocolHandler {
    inner: Arc<HandlerInner>,
}

#[derive(Clone)]
struct HandlerInner {
    processor: Arc<dyn IrohRequestProcessor>,
    /// Used for signed stub errors only below a trusted-local transport test
    /// boundary. Production iroh admission failures are silently dropped.
    /// Authenticated application errors come back from the processor already
    /// wrapped as signed `ResponseEnvelope` bytes.
    signing_key: SigningKey,
    /// Caps concurrent bidi streams in flight. Hold one permit per stream.
    stream_limit: Arc<Semaphore>,
    stream_limit_capacity: u32,
    /// Caps concurrent accepted connections (#165). iroh's Router spawns
    /// each accepted connection into an unbounded JoinSet — connections beyond
    /// this cap are rejected (dropped) rather than queued so a peer opening
    /// many idle connections can't exhaust fd/memory.
    connection_limit: Arc<Semaphore>,
    /// Per-stream request-read timeout (#159 slowloris bound).
    read_timeout: std::time::Duration,
    /// Accept-boundary carrier classification. Production is always iroh;
    /// unit tests may substitute inproc to exercise raw transport mechanics
    /// below the network envelope-policy boundary.
    carrier: crate::transport::carrier::CarrierContext,
    /// Level-triggered shutdown signal: once `cancel()` is called, every
    /// future and current `cancelled().await` resolves immediately. Used
    /// instead of `tokio::sync::Notify` because `Notify::notify_waiters`
    /// is edge-triggered and a shutdown signal landing between accept-loop
    /// iterations would be lost.
    shutdown: CancellationToken,
}

impl std::fmt::Debug for IrohRpcProtocolHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohRpcProtocolHandler")
            .field("stream_limit", &self.inner.stream_limit_capacity)
            .finish_non_exhaustive()
    }
}

impl IrohRpcProtocolHandler {
    /// Build a handler with the default server-wide stream limit
    /// ([`DEFAULT_STREAM_LIMIT`]).
    pub fn new<P: IrohRequestProcessor>(processor: P, signing_key: SigningKey) -> Self {
        Self::with_stream_limit(Arc::new(processor), signing_key, DEFAULT_STREAM_LIMIT)
    }

    /// Build a handler with an explicit server-wide stream limit.
    pub fn with_stream_limit(
        processor: Arc<dyn IrohRequestProcessor>,
        signing_key: SigningKey,
        stream_limit: usize,
    ) -> Self {
        let stream_limit_capacity = u32::try_from(stream_limit).unwrap_or(u32::MAX);
        Self {
            inner: Arc::new(HandlerInner {
                processor,
                signing_key,
                stream_limit: Arc::new(Semaphore::new(stream_limit)),
                stream_limit_capacity,
                connection_limit: Arc::new(Semaphore::new(
                    super::rpc_session::DEFAULT_CONNECTION_LIMIT,
                )),
                read_timeout: super::rpc_session::REQUEST_READ_TIMEOUT,
                carrier: crate::transport::carrier::CarrierContext::iroh(),
                shutdown: CancellationToken::new(),
            }),
        }
    }

    /// Override the per-stream request-read timeout (#159). Primarily for
    /// tests that need a short slowloris bound; production uses the default
    /// [`super::rpc_session::REQUEST_READ_TIMEOUT`].
    pub fn with_read_timeout(mut self, read_timeout: std::time::Duration) -> Self {
        self.mutate_inner(|i| i.read_timeout = read_timeout);
        self
    }

    #[cfg(test)]
    fn with_test_trusted_carrier(mut self) -> Self {
        self.mutate_inner(|i| i.carrier = crate::transport::carrier::CarrierContext::inproc());
        self
    }

    /// Override the concurrent-connection cap (builder style, mirrors
    /// [`super::quinn_transport::QuinnRpcServer::with_connection_limit`]).
    pub fn with_connection_limit(mut self, connection_limit: usize) -> Self {
        self.mutate_inner(|i| i.connection_limit = Arc::new(Semaphore::new(connection_limit)));
        self
    }

    /// Mutate the inner config, cloning `Arc<HandlerInner>` only if it is shared.
    fn mutate_inner(&mut self, f: impl FnOnce(&mut HandlerInner)) {
        match Arc::get_mut(&mut self.inner) {
            Some(inner) => f(inner),
            None => {
                let mut cloned = (*self.inner).clone();
                f(&mut cloned);
                self.inner = Arc::new(cloned);
            }
        }
    }

    /// Apply all tunables from an [`super::rpc_session::RpcConfig`] in one call (#197).
    pub fn with_rpc_config(self, cfg: &super::rpc_session::RpcConfig) -> Self {
        let processor = Arc::clone(&self.inner.processor);
        let signing_key = self.inner.signing_key.clone();
        Self::with_stream_limit(processor, signing_key, cfg.stream_limit)
            .with_read_timeout(cfg.request_read_timeout)
            .with_connection_limit(cfg.connection_limit)
    }
}

impl ProtocolHandler for IrohRpcProtocolHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Connection cap (#165): reject (drop) at cap rather than queue.
        // iroh's Router spawns each accepted connection into an unbounded
        // JoinSet — this try_acquire enforces the same bound QuinnRpcServer
        // and UdsRpcServer apply. The permit is held for the connection's
        // entire lifetime (dropped when this fn returns).
        let _conn_permit = match Arc::clone(&self.inner.connection_limit).try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                tracing::warn!("iroh-rpc: connection cap reached, rejecting connection");
                return Ok(());
            }
        };

        // Adapt the raw iroh `Connection` to the transport-generic `Session`
        // abstraction and delegate to the shared accept loop. Drain semantics
        // (`shutdown` below draining the same `Semaphore`) are unchanged.
        let session = web_transport_iroh::Session::raw(conn);
        // INV-2 (#1042): this accept boundary terminates an iroh connection —
        // an untrusted carrier regardless of direct/relay path or NodeId.
        serve_rpc_connection(
            session,
            Arc::clone(&self.inner.processor),
            self.inner.signing_key.clone(),
            Arc::clone(&self.inner.stream_limit),
            self.inner.read_timeout,
            self.inner.shutdown.clone(),
            self.inner.carrier,
        )
        .await
        .map_err(|e| AcceptError::from_err(std::io::Error::other(e.to_string())))
    }

    async fn shutdown(&self) {
        // Stop the accept loop from taking new streams. Level-triggered:
        // even if cancel() lands between iterations, the next time the
        // loop hits `cancelled().await` it returns immediately.
        self.inner.shutdown.cancel();
        // Wait for all in-flight streams to release their permits.
        // `acquire_many` succeeds only once every permit is returned. Bounded
        // (#159) so a wedged processor/transport can't hang shutdown forever;
        // on timeout we close() and proceed (remaining tasks die with the conn).
        let cap = self.inner.stream_limit_capacity;
        match tokio::time::timeout(
            super::rpc_session::DRAIN_TIMEOUT,
            self.inner.stream_limit.acquire_many(cap),
        )
        .await
        {
            Ok(Ok(permits)) => {
                // Keep permits drained so any post-shutdown accept also
                // sees a closed semaphore.
                permits.forget();
                self.inner.stream_limit.close();
            }
            Ok(Err(_)) => {
                // Already closed; nothing to drain.
            }
            Err(_) => {
                tracing::warn!(
                    timeout = ?super::rpc_session::DRAIN_TIMEOUT,
                    "iroh-rpc: drain timed out, forcing teardown"
                );
                self.inner.stream_limit.close();
            }
        }
    }
}

/// Client-side helper: open a bidi stream on `hyprstream-rpc/1` against an
/// already-connected iroh [`Connection`], write the request, read the response.
///
/// Primitive used by tests and internally by
/// [`super::iroh_transport::IrohTransport`] — production callers should
/// construct an `IrohTransport` + [`crate::rpc_client::RpcClientImpl`] instead.
pub async fn client_request(conn: &Connection, request: &[u8]) -> Result<Bytes> {
    let (mut send, mut recv) = conn.open_bi().await.context("open_bi")?;
    send.write_all(request).await.context("write request")?;
    send.finish().context("finish send")?;
    let buf = recv
        .read_to_end(MAX_FRAME_BYTES)
        .await
        .context("read response")?;
    Ok(Bytes::from(buf))
}

// ============================================================================
// LocalServiceBridge — adapt a (possibly `!Send`) RequestService to the Send-bound
// IrohRequestProcessor trait by running the service on a dedicated LocalSet
// thread and forwarding requests over an mpsc channel.
// ============================================================================

/// Per-request payload + response slot exchanged with the bridge thread.
struct BridgeMessage {
    request: Bytes,
    /// Accept-boundary carrier classification (INV-2 #1042), forwarded
    /// verbatim to the dispatch pipeline's cleartext policy.
    carrier: crate::transport::carrier::CarrierContext,
    respond: tokio::sync::oneshot::Sender<Result<Bytes>>,
}

/// Shared dispatch loop body: relay `BridgeMessage`s from `rx` to `service`,
/// each in its own `spawn_local` task so a slow handler never head-of-lines
/// the queue.
async fn run_bridge_dispatch_loop<S>(
    service: std::rc::Rc<S>,
    mut rx: tokio::sync::mpsc::Receiver<BridgeMessage>,
    nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
    shutdown: CancellationToken,
) -> BridgeShutdownResult
where
    S: crate::service::RequestService + 'static,
{
    let mut requests = tokio::task::JoinSet::new();
    let mut result = Ok(());
    // Closing the receiver rejects new sends but preserves the accepted queue.
    // The dispatch owner joins every spawned request before returning.
    loop {
        let msg = tokio::select! {
            biased;
            _ = shutdown.cancelled(), if !rx.is_closed() => {
                rx.close();
                continue;
            }
            joined = requests.join_next(), if !requests.is_empty() => {
                if matches!(joined, Some(Err(_))) {
                    result = Err(BridgeShutdownError::ThreadPanicked);
                }
                continue;
            }
            msg = rx.recv() => match msg {
                Some(msg) => msg,
                None => break,
            },
        };
        let service = std::rc::Rc::clone(&service);
        let nonce_cache = Arc::clone(&nonce_cache);
        requests.spawn_local(async move {
            let signing_key = service.signing_key();
            let result = crate::service::dispatch::process_request(
                msg.request.as_ref(),
                &*service,
                crate::envelope::EnvelopeVerification::AnySigner,
                &signing_key,
                &nonce_cache,
                msg.carrier,
            )
            .await
            .map(Bytes::from);
            let _ = msg.respond.send(result);
        });
    }
    while let Some(joined) = requests.join_next().await {
        if joined.is_err() { result = Err(BridgeShutdownError::ThreadPanicked); }
    }
    result
}

/// Adapt a [`crate::service::RequestService`] to [`IrohRequestProcessor`].
///
/// Spins up a dedicated thread running a single-threaded tokio runtime with
/// a `LocalSet`. The service runs on that thread (compatible with both `Send`
/// and `!Send` services like inference); requests arriving on the iroh accept
/// loop are forwarded via an `mpsc` channel and awaited via `oneshot`.
///
/// **Lifecycle**: shutdown closes admission, drains queued and running requests
/// within the same grace period as network RPC, then destroys residual LocalSet
/// tasks and joins the thread. A retained client cannot pin the service. Final
/// handle drop provides synchronous cleanup; explicit owners use async shutdown.
pub struct LocalServiceBridge {
    tx: tokio::sync::mpsc::Sender<BridgeMessage>,
    shutdown: CancellationToken,
    thread: Arc<BridgeThread>,
    admission: parking_lot::Mutex<()>,
}

/// Stable terminal outcome shared by all explicit shutdown waiters.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BridgeShutdownError {
    #[error("bridge thread panicked")]
    ThreadPanicked,
    #[error("bridge request drain timed out")]
    DrainTimedOut,
    #[error("bridge runtime initialization failed: {0}")]
    Runtime(String),
    #[error("cannot join bridge from its own thread")]
    SelfJoin,
    #[error("bridge join task failed: {0}")]
    JoinTask(String),
}

type BridgeShutdownResult = std::result::Result<(), BridgeShutdownError>;

/// Join ownership survives cancellation of an async shutdown waiter.
struct BridgeThread {
    id: std::thread::ThreadId,
    handle: parking_lot::Mutex<Option<std::thread::JoinHandle<BridgeShutdownResult>>>,
    completed: parking_lot::Mutex<Option<BridgeShutdownResult>>,
    completion: parking_lot::Condvar,
}

impl BridgeThread {
    fn new(handle: std::thread::JoinHandle<BridgeShutdownResult>) -> Self {
        Self {
            id: handle.thread().id(),
            handle: parking_lot::Mutex::new(Some(handle)),
            completed: parking_lot::Mutex::new(None),
            completion: parking_lot::Condvar::new(),
        }
    }

    fn join(&self) -> BridgeShutdownResult {
        if self.id == std::thread::current().id() {
            return Err(BridgeShutdownError::SelfJoin);
        }
        let handle = self.handle.lock().take();
        if let Some(handle) = handle {
            // No lock needed by any caller or service destructor is held here.
            let result = handle.join().unwrap_or(Err(BridgeShutdownError::ThreadPanicked));
            *self.completed.lock() = Some(result.clone());
            self.completion.notify_all();
            result
        } else {
            let mut completed = self.completed.lock();
            loop {
                if let Some(result) = &*completed {
                    return result.clone();
                }
                self.completion.wait(&mut completed);
            }
        }
    }
}

async fn finish_bridge_local_set(
    local: tokio::task::LocalSet,
    dispatch: tokio::task::JoinHandle<BridgeShutdownResult>,
    shutdown: CancellationToken,
    grace: std::time::Duration,
) -> BridgeShutdownResult {
    let result = {
        // Await the dispatch task (which owns every accepted RPC), not the
        // whole LocalSet: streaming continuations are residual background work.
        let run = local.run_until(async move {
            dispatch.await.unwrap_or(Err(BridgeShutdownError::ThreadPanicked))
        });
        tokio::pin!(run);
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                tokio::time::timeout(grace, &mut run).await
                    .unwrap_or(Err(BridgeShutdownError::DrainTimedOut))
            }
            result = &mut run => result,
        }
    };
    // Entered runtime remains alive while residual tasks/services are destroyed.
    drop(local);
    result
}

impl crate::transport::rpc_session::sealed::Sealed for LocalServiceBridge {
    fn admits_untrusted_carrier(&self) -> bool {
        true
    }
}

impl LocalServiceBridge {
    /// Spawn a dedicated bridge thread that owns `service` and forwards
    /// requests through [`crate::service::dispatch::process_request`].
    ///
    /// The response signing key is taken from `service.signing_key()` — there
    /// is no separate parameter to avoid drift between the service's identity
    /// and the key used to sign responses.
    ///
    /// `queue_depth` is the mpsc channel capacity — backpressure point when
    /// the bridge falls behind. Pass `0` for default (128).
    pub fn spawn<S>(
        service: S,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
        queue_depth: usize,
    ) -> Result<Self>
    where
        S: crate::service::RequestService + Send + 'static,
    {
        let cap = if queue_depth == 0 { 128 } else { queue_depth };
        let (tx, rx) = tokio::sync::mpsc::channel::<BridgeMessage>(cap);
        let service_name = service.name().to_owned();

        let shutdown = CancellationToken::new();
        let stopped = shutdown.clone();
        let thread = std::thread::Builder::new()
            .name(format!("iroh-rpc-bridge:{service_name}"))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        return Err(BridgeShutdownError::Runtime(e.to_string()));
                    }
                };
                let local = tokio::task::LocalSet::new();
                let dispatch = local.spawn_local(run_bridge_dispatch_loop(
                    std::rc::Rc::new(service),
                    rx,
                    nonce_cache,
                    stopped.clone(),
                ));
                let result = rt.block_on(finish_bridge_local_set(
                    local, dispatch, stopped, super::rpc_session::DRAIN_TIMEOUT,
                ));
                drop(rt);
                result
            })
            .map_err(|e| anyhow::anyhow!("spawn iroh-rpc bridge thread: {e}"))?;

        Ok(Self { tx, shutdown, thread: Arc::new(BridgeThread::new(thread)), admission: parking_lot::Mutex::new(()) })
    }

    /// Like [`LocalServiceBridge::spawn`], but constructs the service ON the
    /// bridge thread via the async `build` closure. For services whose
    /// construction is itself `!Send` or async (e.g. GPU init that must happen on
    /// the serve thread, like `InferenceService`): the built service value never
    /// has to be `Send`-moved across threads — only the builder's captured inputs
    /// do (`F: Send`), and the `!Send` result lives only on the bridge thread.
    ///
    /// Returns the bridge plus a readiness receiver that resolves once `build`
    /// completes: `Ok(())` when the service is built and serving, or the build
    /// error. Callers MUST await it before advertising the service (a build
    /// failure otherwise surfaces only as "channel closed" on the first request).
    pub fn spawn_with<F, Fut, S>(
        thread_name: impl Into<String>,
        build: F,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
        queue_depth: usize,
    ) -> Result<(Self, tokio::sync::oneshot::Receiver<Result<()>>)>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<S>>,
        S: crate::service::RequestService + 'static,
    {
        let cap = if queue_depth == 0 { 128 } else { queue_depth };
        let (tx, rx) = tokio::sync::mpsc::channel::<BridgeMessage>(cap);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<()>>();
        let thread_name = thread_name.into();

        let shutdown = CancellationToken::new();
        let stopped = shutdown.clone();
        let thread = std::thread::Builder::new()
            .name(format!("rpc-bridge:{thread_name}"))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(anyhow::anyhow!("bridge runtime: {e}")));
                        return Err(BridgeShutdownError::Runtime(e.to_string()));
                    }
                };
                let local = tokio::task::LocalSet::new();
                let dispatch_shutdown = stopped.clone();
                let dispatch = local.spawn_local(async move {
                    // Build on-thread; a failure (e.g. GPU init) is reported via
                    // the readiness channel and the bridge thread exits.
                    let built = tokio::select! {
                        biased;
                        _ = dispatch_shutdown.cancelled() => return Ok(()),
                        built = build() => built,
                    };
                    let service = match built {
                        Ok(s) => std::rc::Rc::new(s),
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return Ok(());
                        }
                    };
                    if ready_tx.send(Ok(())).is_err() {
                        // Caller gave up waiting; no point serving.
                        return Ok(());
                    }
                    run_bridge_dispatch_loop(service, rx, nonce_cache, dispatch_shutdown).await
                });
                let result = rt.block_on(finish_bridge_local_set(
                    local, dispatch, stopped, super::rpc_session::DRAIN_TIMEOUT,
                ));
                drop(rt);
                result
            })
            .map_err(|e| anyhow::anyhow!("spawn rpc bridge thread: {e}"))?;

        Ok((Self { tx, shutdown, thread: Arc::new(BridgeThread::new(thread)), admission: parking_lot::Mutex::new(()) }, ready_rx))
    }
}

impl LocalServiceBridge {
    fn close_admission(&self) {
        let _admission = self.admission.lock();
        self.shutdown.cancel();
    }

    /// Close admission, drain accepted work, and wait for service/runtime destruction.
    /// Joining runs off the caller's runtime; no thread lock is held while waiting.
    /// Concurrent close calls share the same terminal result.
    pub async fn shutdown(&self) -> BridgeShutdownResult {
        self.close_admission();
        if self.thread.id == std::thread::current().id() {
            return Err(BridgeShutdownError::SelfJoin);
        }
        let thread = Arc::clone(&self.thread);
        tokio::task::spawn_blocking(move || thread.join()).await
            .map_err(|error| BridgeShutdownError::JoinTask(error.to_string()))?
    }
}

impl Drop for LocalServiceBridge {
    fn drop(&mut self) {
        self.close_admission();
        // Explicit async shutdown is the service-owner path. Final-handle drop
        // also owns teardown, including failed or cancelled spawn_with builders.
        if let Err(error) = self.thread.join() {
            tracing::error!(%error, "bridge drop teardown failed");
        }
    }
}

impl IrohRequestProcessor for LocalServiceBridge {
    fn close_admission(&self) { LocalServiceBridge::close_admission(self); }

    fn process(
        &self,
        request: Bytes,
        carrier: crate::transport::carrier::CarrierContext,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>> {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (respond_tx, respond_rx) = tokio::sync::oneshot::channel();
            let permit = tokio::select! {
                biased;
                _ = self.shutdown.cancelled() => anyhow::bail!("iroh-rpc bridge: admission closed"),
                reserved = tx.reserve() => {
                    reserved.map_err(|_| anyhow::anyhow!("iroh-rpc bridge: channel closed"))?
                }
            };
            {
                // Linearize successful enqueue against shutdown, without holding
                // any lock across queue backpressure, dispatch, or thread join.
                let _admission = self.admission.lock();
                anyhow::ensure!(!self.shutdown.is_cancelled(), "iroh-rpc bridge: admission closed");
                permit.send(BridgeMessage { request, carrier, respond: respond_tx });
            }
            respond_rx
                .await
                .map_err(|_| anyhow::anyhow!("iroh-rpc bridge: response dropped"))?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::iroh_substrate::{
        IrohSubstrate, NoopHandler, ALPN_HYPRSTREAM_RPC, ALPN_MOQ_LITE,
    };
    use iroh::{EndpointAddr, TransportAddr};
    use rand::RngCore;
    use std::time::Duration;

    fn fresh_key() -> [u8; 32] {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        k
    }

    fn fresh_signing_key() -> SigningKey {
        SigningKey::from_bytes(&fresh_key())
    }

    fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
        EndpointAddr::from_parts(
            substrate.endpoint_id(),
            substrate
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(TransportAddr::Ip),
        )
    }

    /// Minimal `RequestService` for `spawn_with` tests.
    struct BridgeEcho {
        name: String,
        transport: crate::transport::TransportConfig,
        signing_key: SigningKey,
        drop_probe: Option<BridgeDropProbe>,
        request_gate: Option<(Arc<tokio::sync::Semaphore>, Arc<tokio::sync::Semaphore>)>,
    }

    struct BridgeDropProbe {
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::mpsc::Receiver<()>,
        completed: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for BridgeDropProbe {
        fn drop(&mut self) {
            let _ = self.entered.send(());
            let _ = self.release.recv_timeout(Duration::from_secs(5));
            self.completed.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl BridgeEcho {
        fn new(signing_key: SigningKey) -> Self {
            Self {
                drop_probe: None,
                request_gate: None,
                name: "bridge-echo".to_owned(),
                transport: crate::transport::TransportConfig::inproc("bridge-echo-unused"),
                signing_key,
            }
        }
    }
    #[async_trait::async_trait(?Send)]
    impl crate::service::RequestService for BridgeEcho {
        async fn handle_request(
            &self,
            _ctx: &crate::service::EnvelopeContext,
            payload: &[u8],
        ) -> Result<(Vec<u8>, Option<crate::service::Continuation>)> {
            let mut out = vec![0xBE];
            out.extend_from_slice(payload);
            Ok((out, None))
        }
        async fn verify_claims(&self, _ctx: &mut crate::service::EnvelopeContext) -> Result<()> {
            if let Some((entered, release)) = &self.request_gate {
                entered.add_permits(1);
                release.acquire().await?.forget();
            }
            anyhow::bail!("gated claims denial")
        }
        fn build_error_payload(&self, _request_id: u64, error: &str) -> Vec<u8> {
            error.as_bytes().to_vec()
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn transport(&self) -> &crate::transport::TransportConfig {
            &self.transport
        }
        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }
    }

    #[test]
    fn bridge_drop_waits_for_service_destructor() -> Result<()> {
        assert_bridge_destructor_completion(false)
    }

    #[test]
    fn bridge_spawn_with_drop_waits_for_service_destructor() -> Result<()> {
        assert_bridge_destructor_completion(true)
    }

    fn assert_bridge_destructor_completion(build_on_thread: bool) -> Result<()> {
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut service = BridgeEcho::new(fresh_signing_key());
        service.drop_probe = Some(BridgeDropProbe {
            entered: entered_tx,
            release: release_rx,
            completed: Arc::clone(&completed),
        });
        let nonce = Arc::new(crate::envelope::InMemoryNonceCache::new());
        let bridge = Arc::new(if build_on_thread {
            let (bridge, ready) = LocalServiceBridge::spawn_with(
                "destructor-gate", move || async move { Ok(service) }, nonce, 0,
            )?;
            ready.blocking_recv()??;
            bridge
        } else {
            LocalServiceBridge::spawn(service, nonce, 0)?
        });
        let retained = Arc::clone(&bridge);
        drop(retained);
        assert!(!completed.load(std::sync::atomic::Ordering::SeqCst));
        let (returned_tx, returned_rx) = std::sync::mpsc::channel();
        let owner = std::thread::spawn(move || {
            drop(bridge);
            let _ = returned_tx.send(completed.load(std::sync::atomic::Ordering::SeqCst));
        });
        entered_rx.recv_timeout(Duration::from_secs(5))?;
        // The destructor is held at a causal gate, not slowed by a sleep.
        // Teardown must not return while that gate is held.
        let premature = returned_rx.recv_timeout(Duration::from_millis(100)).ok();
        release_tx.send(())?;
        owner.join().map_err(|_| anyhow::anyhow!("bridge owner panicked"))?;
        assert!(premature.is_none(), "bridge drop returned before service destruction");
        assert!(returned_rx.recv_timeout(Duration::from_secs(5))?);
        Ok(())
    }

    #[test]
    fn bridge_thread_join_never_joins_itself() -> Result<()> {
        let (owner_tx, owner_rx) = std::sync::mpsc::channel::<Arc<BridgeThread>>();
        let (returned_tx, returned_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            if let Ok(owner) = owner_rx.recv() {
                assert_eq!(owner.join(), Err(BridgeShutdownError::SelfJoin));
                let _ = returned_tx.send(());
            }
            Ok(())
        });
        let owner = Arc::new(BridgeThread::new(worker));
        owner_tx.send(Arc::clone(&owner))?;
        returned_rx.recv_timeout(Duration::from_secs(5))?;
        owner.join()?;
        assert_eq!(*owner.completed.lock(), Some(Ok(())));
        Ok(())
    }

    #[tokio::test]
    async fn bridge_cancelled_shutdown_waiter_keeps_completion_barrier() -> Result<()> {
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut service = BridgeEcho::new(fresh_signing_key());
        service.drop_probe = Some(BridgeDropProbe {
            entered: entered_tx, release: release_rx, completed: Arc::clone(&completed),
        });
        let bridge = Arc::new(LocalServiceBridge::spawn(
            service, Arc::new(crate::envelope::InMemoryNonceCache::new()), 0,
        )?);
        let owner = Arc::clone(&bridge);
        let first = tokio::spawn(async move { owner.shutdown().await });
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(5))).await??;
        first.abort();
        let _ = first.await;
        let premature = tokio::time::timeout(Duration::from_millis(100), bridge.shutdown()).await.is_ok();
        release_tx.send(())?;
        tokio::time::timeout(Duration::from_secs(5), bridge.shutdown()).await??;
        assert!(!premature, "cancelled waiter lost the in-progress thread join");
        assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
        Ok(())
    }

    #[tokio::test]
    async fn bridge_shutdown_cancels_builder_with_retained_handle() -> Result<()> {
        struct Dropped(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = Dropped(Arc::clone(&completed));
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (bridge, ready) = LocalServiceBridge::spawn_with::<_, _, BridgeEcho>(
            "pending-builder",
            move || async move {
                let _guard = guard;
                let _ = entered_tx.send(());
                std::future::pending().await
            },
            Arc::new(crate::envelope::InMemoryNonceCache::new()), 0,
        )?;
        let bridge = Arc::new(bridge);
        let retained = Arc::clone(&bridge);
        tokio::time::timeout(Duration::from_secs(5), entered_rx).await??;
        tokio::time::timeout(Duration::from_secs(5), bridge.shutdown()).await??;
        assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
        assert!(ready.await.is_err());
        assert!(retained.process(Bytes::new(),
            crate::transport::carrier::CarrierContext::iroh()).await.is_err());
        // Completion is idempotent while another owner still retains the bridge.
        bridge.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn bridge_shutdown_drains_accepted_queue_before_residual_cancellation() -> Result<()> {
        use crate::ToCapnp;
        let key = fresh_signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&key);
        let mut store = crate::envelope::KeyedPqTrustStore::new();
        store.bind(key.verifying_key().to_bytes(), &crate::crypto::pq::ml_dsa_sk_to_vk(&pq));
        let store = Arc::new(store);
        crate::envelope::install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid, pq_store: Some(store.clone()),
        })?;
        crate::envelope::install_response_verify_config(crate::envelope::ResponseVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid, pq_store: Some(store),
        })?;
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let mut service = BridgeEcho::new(key.clone());
        service.request_gate = Some((entered.clone(), release.clone()));
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let mut responses = Vec::new();
        for payload in [b"one", b"two"] {
            let envelope = crate::envelope::SignedEnvelope::new_signed_hybrid(
                crate::envelope::RequestEnvelope::new(payload.to_vec()), &key, &pq,
            );
            let mut message = capnp::message::Builder::new_default();
            envelope.write_to(&mut message.init_root::<crate::common_capnp::signed_envelope::Builder>());
            let mut wire = Vec::new();
            capnp::serialize::write_message(&mut wire, &message)?;
            let (respond, response) = tokio::sync::oneshot::channel();
            tx.try_send(BridgeMessage {
                request: Bytes::from(wire), carrier: crate::transport::carrier::CarrierContext::inproc(), respond,
            }).map_err(|_| anyhow::anyhow!("queue admission failed"))?;
            responses.push(response);
        }
        // Both requests are queued BEFORE shutdown, before dispatch is polled.
        let stopped = CancellationToken::new();
        stopped.cancel();
        let local = tokio::task::LocalSet::new();
        let dispatch = local.spawn_local(run_bridge_dispatch_loop(
            std::rc::Rc::new(service), rx, Arc::new(crate::envelope::InMemoryNonceCache::new()), stopped.clone(),
        ));
        let drain = finish_bridge_local_set(local, dispatch, stopped, super::super::rpc_session::DRAIN_TIMEOUT);
        let coordinate = async {
            tokio::time::timeout(Duration::from_secs(5), entered.acquire_many(2)).await??.forget();
            assert!(tx.is_closed(), "new admissions remained open during grace");
            release.add_permits(2);
            for response in responses {
                let wire = response.await??;
                let (_, payload) = crate::envelope::unwrap_response(&wire, Some(&key.verifying_key()))?;
                assert_eq!(payload, b"gated claims denial");
            }
            anyhow::Ok(())
        };
        let (drained, coordinated) = tokio::join!(drain, coordinate);
        drained?;
        coordinated?;
        Ok(())
    }

    #[tokio::test]
    async fn bridge_panicked_thread_result_is_shared_by_shutdown_waiters() -> Result<()> {
        let worker = std::thread::spawn(|| -> BridgeShutdownResult {
            panic!("causal bridge thread panic");
        });
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let bridge = Arc::new(LocalServiceBridge {
            tx,
            shutdown: CancellationToken::new(),
            thread: Arc::new(BridgeThread::new(worker)),
            admission: parking_lot::Mutex::new(()),
        });
        let (first, second) = tokio::join!(bridge.shutdown(), bridge.shutdown());
        assert_eq!(first, Err(BridgeShutdownError::ThreadPanicked));
        assert_eq!(second, first);
        assert_eq!(bridge.shutdown().await, first);
        Ok(())
    }

    #[tokio::test]
    async fn bridge_drain_timeout_cancels_accepted_request_and_shares_error() -> Result<()> {
        use crate::ToCapnp;
        let key = fresh_signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&key);
        let mut store = crate::envelope::KeyedPqTrustStore::new();
        store.bind(key.verifying_key().to_bytes(), &crate::crypto::pq::ml_dsa_sk_to_vk(&pq));
        crate::envelope::install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid, pq_store: Some(Arc::new(store)),
        })?;
        let envelope = crate::envelope::SignedEnvelope::new_signed_hybrid(
            crate::envelope::RequestEnvelope::new(b"held".to_vec()), &key, &pq,
        );
        let mut message = capnp::message::Builder::new_default();
        envelope.write_to(&mut message.init_root::<crate::common_capnp::signed_envelope::Builder>());
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message)?;
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let mut service = BridgeEcho::new(key);
        service.request_gate = Some((entered.clone(), Arc::new(tokio::sync::Semaphore::new(0))));
        let (drop_entered, _) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        release.send(())?;
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        service.drop_probe = Some(BridgeDropProbe {
            entered: drop_entered, release: released, completed: dropped.clone(),
        });
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (respond, response) = tokio::sync::oneshot::channel();
        tx.try_send(BridgeMessage {
            request: Bytes::from(wire), carrier: crate::transport::carrier::CarrierContext::inproc(), respond,
        }).map_err(|_| anyhow::anyhow!("queue admission failed"))?;
        let stopped = CancellationToken::new();
        let thread_stopped = stopped.clone();
        let thread = std::thread::spawn(move || {
            // Real dispatch/queue/destructor, with only the runtime clock made
            // virtual: exercise the unchanged production 40-second bound.
            let rt = tokio::runtime::Builder::new_current_thread().enable_all()
                .start_paused(true).build().map_err(|e| BridgeShutdownError::Runtime(e.to_string()))?;
            let local = tokio::task::LocalSet::new();
            let dispatch = local.spawn_local(run_bridge_dispatch_loop(
                std::rc::Rc::new(service), rx, Arc::new(crate::envelope::InMemoryNonceCache::new()), thread_stopped.clone(),
            ));
            rt.block_on(finish_bridge_local_set(local, dispatch, thread_stopped, super::super::rpc_session::DRAIN_TIMEOUT))
        });
        let bridge = LocalServiceBridge {
            tx, shutdown: stopped, thread: Arc::new(BridgeThread::new(thread)), admission: parking_lot::Mutex::new(()),
        };
        tokio::time::timeout(Duration::from_secs(5), entered.acquire()).await??.forget();
        let (first, second) = tokio::join!(bridge.shutdown(), bridge.shutdown());
        assert_eq!(first, Err(BridgeShutdownError::DrainTimedOut));
        assert_eq!(second, first);
        assert_eq!(bridge.shutdown().await, first);
        assert!(response.await.is_err(), "forced deadline retained the accepted response sender");
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_with_readiness_ok_on_success_err_on_failure() -> Result<()> {
        let nonce = Arc::new(crate::envelope::InMemoryNonceCache::new());

        // Builder succeeds → readiness resolves Ok and the bridge serves.
        let sk = fresh_signing_key();
        let (_bridge, ready) = LocalServiceBridge::spawn_with(
            "ok",
            move || async move { Ok(BridgeEcho::new(sk)) },
            Arc::clone(&nonce),
            0,
        )?;
        ready
            .await
            .map_err(|_| anyhow::anyhow!("readiness dropped"))??;

        // Builder fails (e.g. GPU init error) → the error surfaces on readiness,
        // not silently as a later channel-closed.
        let (_bridge2, ready2) = LocalServiceBridge::spawn_with::<_, _, BridgeEcho>(
            "fail",
            move || async move { Err(anyhow::anyhow!("boom")) },
            nonce,
            0,
        )?;
        let build_result = ready2
            .await
            .map_err(|_| anyhow::anyhow!("readiness dropped"))?;
        assert!(
            build_result.is_err(),
            "a build failure must surface on the readiness channel"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_request_response_round_trip() -> Result<()> {
        let processor = from_fn(|req: Bytes| async move {
            let mut out = Vec::with_capacity(1 + req.len());
            out.push(0xAB);
            out.extend_from_slice(&req);
            Ok(Bytes::from(out))
        });
        let rpc_handler =
            IrohRpcProtocolHandler::new(processor, fresh_signing_key()).with_test_trusted_carrier();

        let server =
            IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq-not-wired"), rpc_handler)
                .await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("client moq"),
            NoopHandler::new("client rpc"),
        )
        .await?;

        let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;
        let resp = client_request(&conn, b"ping").await?;
        assert_eq!(&resp[..], b"\xABping");

        let conn2 = client.connect(direct_addr(&server), ALPN_MOQ_LITE).await?;
        drop(conn2);

        // Release every caller-owned connection before draining the routers.
        // Keeping this RPC handle alive races noq's driver teardown and can
        // intermittently trip its GSO-batch alignment assertion.
        drop(conn);
        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_concurrent_requests() -> Result<()> {
        let processor = from_fn(|req: Bytes| async move {
            let mut out = req.to_vec();
            out.push(b'!');
            Ok(Bytes::from(out))
        });
        let server = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("moq"),
            IrohRpcProtocolHandler::new(processor, fresh_signing_key()).with_test_trusted_carrier(),
        )
        .await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = Arc::new(client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?);

        let mut handles = Vec::new();
        for i in 0..8u8 {
            let conn = Arc::clone(&conn);
            handles.push(tokio::spawn(async move {
                let resp = client_request(&conn, &[i]).await?;
                assert_eq!(&resp[..], &[i, b'!']);
                anyhow::Ok(())
            }));
        }
        for h in handles {
            h.await??;
        }

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// Processor returns `Err` → client gets a parseable signed
    /// `ResponseEnvelope` carrying the error message (request_id = 0), not
    /// an opaque EOF / Cap'n Proto parse failure. (Fix for review finding #4.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_processor_error_yields_parseable_envelope() -> Result<()> {
        let server_signing = fresh_signing_key();
        let server_vk = server_signing.verifying_key();

        let processor =
            from_fn(|_req: Bytes| async move { Err(anyhow::anyhow!("boom from processor")) });
        let handler =
            IrohRpcProtocolHandler::new(processor, server_signing).with_test_trusted_carrier();

        let server = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;

        // Send any bytes; the processor will Err regardless. The interesting
        // assertion is what comes back: a verifiable ResponseEnvelope.
        let resp_bytes = client_request(&conn, b"anything").await?;
        let (request_id, payload) =
            crate::envelope::unwrap_response(&resp_bytes, Some(&server_vk))?;
        assert_eq!(request_id, 0, "error envelope uses request_id=0");
        let body = std::str::from_utf8(&payload)?;
        assert!(body.contains("boom from processor"), "got: {body}");

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// Concurrency cap is enforced: with a stream_limit of 2, a third
    /// concurrent slow request must wait for one of the first two to finish.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_stream_limit_enforced() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let in_flight_c = Arc::clone(&in_flight);
        let peak_c = Arc::clone(&peak);

        let processor = from_fn(move |_req: Bytes| {
            let in_flight = Arc::clone(&in_flight_c);
            let peak = Arc::clone(&peak_c);
            async move {
                let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                // Update peak.
                let mut p = peak.load(Ordering::SeqCst);
                while cur > p {
                    match peak.compare_exchange_weak(p, cur, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(_) => break,
                        Err(prev) => p = prev,
                    }
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(Bytes::from_static(b"ok"))
            }
        });
        let handler =
            IrohRpcProtocolHandler::with_stream_limit(Arc::new(processor), fresh_signing_key(), 2)
                .with_test_trusted_carrier();

        let server = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = Arc::new(client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?);

        let mut handles = Vec::new();
        for _ in 0..6 {
            let conn = Arc::clone(&conn);
            handles.push(tokio::spawn(async move {
                let _ = client_request(&conn, b"x").await?;
                anyhow::Ok(())
            }));
        }
        for h in handles {
            h.await??;
        }

        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "stream_limit=2 violated, observed peak={}",
            peak.load(Ordering::SeqCst)
        );

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// #159: a stalled (slowloris) stream that holds a permit without sending
    /// FIN must be abandoned after the read timeout, releasing its permit so a
    /// well-behaved request can still be served. With `stream_limit=1`, the
    /// stall grabs the only permit; without the read-timeout fix the normal
    /// request would block forever and this test would hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_slowloris_stream_is_timed_out_and_permit_released() -> Result<()> {
        let processor = from_fn(move |_req: Bytes| async move { Ok(Bytes::from_static(b"ok")) });
        let handler =
            IrohRpcProtocolHandler::with_stream_limit(Arc::new(processor), fresh_signing_key(), 1)
                .with_test_trusted_carrier()
                .with_read_timeout(Duration::from_millis(300));

        let server = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;

        // Slowloris: open a bidi stream, dribble one byte, never finish.
        let (mut stall_send, _stall_recv) = conn.open_bi().await.context("open_bi stall")?;
        stall_send
            .write_all(b"x")
            .await
            .context("write stall byte")?;
        // Deliberately DO NOT call finish(); keep the stream (and its server-side
        // permit) alive. Hold the handle so it isn't dropped/reset early.

        // A normal request must still succeed once the stalled stream's read
        // times out (~300ms) and its permit is released. Bound the wait well
        // above the read timeout but far below "forever".
        let resp = tokio::time::timeout(Duration::from_secs(5), client_request(&conn, b"req"))
            .await
            .context("normal request hung — permit was not released (slowloris not bounded)")??;
        assert_eq!(&resp[..], b"ok");

        drop(stall_send);
        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// `Router::shutdown` (via `IrohSubstrate::shutdown`) drains in-flight
    /// requests before returning — a request in-progress when shutdown
    /// starts still gets a clean response. (Fix for review round-1 #1.)
    ///
    /// Uses an explicit `Notify` to synchronise "processor has entered the
    /// long sleep" with "now safe to shut down" — replaces the previous
    /// 50ms `tokio::sleep` that was flaky on loaded CI runners.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_shutdown_drains_in_flight() -> Result<()> {
        let entered = Arc::new(tokio::sync::Notify::new());
        let entered_c = Arc::clone(&entered);
        let processor = from_fn(move |_req: Bytes| {
            let entered = Arc::clone(&entered_c);
            async move {
                entered.notify_one();
                tokio::time::sleep(Duration::from_millis(300)).await;
                Ok(Bytes::from_static(b"drained-ok"))
            }
        });
        let handler =
            IrohRpcProtocolHandler::new(processor, fresh_signing_key()).with_test_trusted_carrier();

        let server = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = client.connect(server_addr, ALPN_HYPRSTREAM_RPC).await?;

        let req_task = {
            let conn = conn.clone();
            tokio::spawn(async move { client_request(&conn, b"slow").await })
        };

        // Synchronise: wait until the processor signals it has entered the
        // sleep, *then* shut down. No wall-clock guesses.
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .context("processor was never entered before shutdown")?;
        server.shutdown().await?;

        let resp = req_task.await??;
        assert_eq!(&resp[..], b"drained-ok");

        client.shutdown().await?;
        Ok(())
    }
}
