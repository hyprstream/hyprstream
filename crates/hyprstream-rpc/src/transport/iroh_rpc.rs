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
        let deadline = self
            .inner
            .processor
            .begin_shutdown(tokio::time::Instant::now() + super::rpc_session::DRAIN_TIMEOUT);
        // Wait for all in-flight streams to release their permits.
        // `acquire_many` succeeds only once every permit is returned. Bounded
        // (#159) so a wedged processor/transport can't hang shutdown forever;
        // on timeout we close() and proceed (remaining tasks die with the conn).
        let cap = self.inner.stream_limit_capacity;
        match tokio::time::timeout_at(deadline, self.inner.stream_limit.acquire_many(cap)).await {
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
    /// Server-owned lifetime signal associated with this accepted stream.
    request_cancel: CancellationToken,
    respond: tokio::sync::oneshot::Sender<Result<Bytes>>,
}

fn bridge_message_has_no_live_caller(msg: &BridgeMessage) -> bool {
    msg.request_cancel.is_cancelled() || msg.respond.is_closed()
}

/// Shared dispatch loop body: relay `BridgeMessage`s from `rx` to `service`,
/// each in its own `spawn_local` task so a slow handler never head-of-lines
/// the queue.
async fn run_bridge_dispatch_loop<S>(
    service: std::rc::Rc<S>,
    mut rx: tokio::sync::mpsc::Receiver<BridgeMessage>,
    nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
    shutdown: CancellationToken,
    force_abort: CancellationToken,
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
            _ = force_abort.cancelled() => {
                // The bounded drain expired. Abort every accepted handler,
                // then join them before this dispatcher completes so service
                // lifecycle cleanup cannot race handler destruction.
                requests.abort_all();
                while let Some(joined) = requests.join_next().await {
                    if let Err(error) = joined {
                        if !error.is_cancelled() {
                            result = Err(BridgeShutdownError::ThreadPanicked);
                        }
                    }
                }
                return result;
            }
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
        // A request can expire or disconnect while it waits in the accepted
        // queue. Do not start authentication/dispatch for work whose caller
        // can no longer receive a response. The task repeats the cancellation
        // check to cover the race between this dequeue and its first poll.
        if bridge_message_has_no_live_caller(&msg) {
            continue;
        }
        requests.spawn_local(async move {
            if bridge_message_has_no_live_caller(&msg) {
                return;
            }
            let signing_key = service.signing_key();
            let result = crate::service::dispatch::process_request_with_cancel(
                msg.request.as_ref(),
                &*service,
                crate::envelope::EnvelopeVerification::AnySigner,
                &signing_key,
                &nonce_cache,
                msg.carrier,
                msg.request_cancel,
            )
            .await
            .map(Bytes::from);
            let _ = msg.respond.send(result);
        });
    }
    while !requests.is_empty() {
        tokio::select! {
            biased;
            _ = force_abort.cancelled() => {
                requests.abort_all();
                while let Some(joined) = requests.join_next().await {
                    if let Err(error) = joined {
                        if !error.is_cancelled() {
                            result = Err(BridgeShutdownError::ThreadPanicked);
                        }
                    }
                }
                break;
            }
            joined = requests.join_next() => {
                if matches!(joined, Some(Err(_))) {
                    result = Err(BridgeShutdownError::ThreadPanicked);
                }
            }
        }
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
/// **Lifecycle**: shutdown closes admission and drains queued/running requests
/// within the network RPC grace period. After the drain (or forced abort and
/// join), `RequestService::on_drained` runs in a separate unbounded phase while
/// the LocalSet/runtime remain alive; residual LocalSet tasks, service, and
/// runtime are destroyed afterward. A retained client cannot pin the service.
/// Final handle drop provides synchronous cleanup; explicit owners use async shutdown.
pub struct LocalServiceBridge {
    tx: tokio::sync::mpsc::Sender<BridgeMessage>,
    shutdown: CancellationToken,
    thread: Arc<BridgeThread>,
    admission: Arc<parking_lot::Mutex<Option<tokio::time::Instant>>>,
}

/// Stable terminal outcome shared by all explicit shutdown waiters.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BridgeShutdownError {
    #[error("bridge thread panicked")]
    ThreadPanicked,
    #[error("bridge request drain timed out")]
    DrainTimedOut,
    #[error("service lifecycle drain hook failed: {0}")]
    ServiceLifecycle(String),
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
            let result = handle
                .join()
                .unwrap_or(Err(BridgeShutdownError::ThreadPanicked));
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
    mut dispatch: tokio::task::JoinHandle<BridgeShutdownResult>,
    shutdown: CancellationToken,
    force_abort: CancellationToken,
    grace: std::time::Duration,
    admission: Arc<parking_lot::Mutex<Option<tokio::time::Instant>>>,
    service: std::rc::Rc<
        std::cell::RefCell<Option<std::rc::Rc<dyn crate::service::RequestService>>>,
    >,
) -> BridgeShutdownResult {
    let (result, timed_out) = {
        // Await the dispatch task (which owns every accepted RPC), not the
        // whole LocalSet: streaming continuations are residual background work.
        let run = local.run_until(&mut dispatch);
        tokio::pin!(run);
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                let deadline = {
                    let admission = admission.lock();
                    admission.unwrap_or_else(|| tokio::time::Instant::now() + grace)
                };
                match tokio::time::timeout_at(deadline, &mut run).await {
                    Ok(joined) => (joined.unwrap_or(Err(BridgeShutdownError::ThreadPanicked)), false),
                    Err(_) => (Err(BridgeShutdownError::DrainTimedOut), true),
                }
            }
            joined = &mut run => (joined.unwrap_or(Err(BridgeShutdownError::ThreadPanicked)), false),
        }
    };
    if timed_out {
        // Ask the dispatch loop to abort and join all accepted handler tasks.
        // Keep the LocalSet/runtime alive and join the dispatcher before
        // calling service lifecycle code.
        force_abort.cancel();
        let _ = local.run_until(&mut dispatch).await;
    }

    // Lifecycle cleanup has its own unbounded phase after the bounded request
    // drain, but before LocalSet/runtime teardown. Keep an owned service
    // reference through LocalSet destruction so service destructors also run
    // while the runtime is still alive.
    let service = service.borrow_mut().take();
    let lifecycle = match service.as_ref() {
        Some(service) => local
            .run_until(service.on_drained())
            .await
            .map_err(|error| BridgeShutdownError::ServiceLifecycle(error.to_string())),
        None => Ok(()), // A builder that failed before producing a service.
    };
    drop(local);
    drop(service);
    match (result, lifecycle) {
        (Err(result), Err(lifecycle)) => {
            tracing::error!(%lifecycle, "service lifecycle drain hook failed after bridge shutdown error");
            Err(result)
        }
        (Err(result), _) => Err(result),
        (Ok(()), lifecycle) => lifecycle,
    }
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
        let force_abort = CancellationToken::new();
        let thread_force_abort = force_abort.clone();
        let admission = Arc::new(parking_lot::Mutex::new(None));
        let thread_admission = Arc::clone(&admission);
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
                let service = std::rc::Rc::new(service);
                let lifecycle_service =
                    std::rc::Rc::new(std::cell::RefCell::new(Some(std::rc::Rc::clone(&service)
                        as std::rc::Rc<dyn crate::service::RequestService>)));
                let dispatch = local.spawn_local(run_bridge_dispatch_loop(
                    service,
                    rx,
                    nonce_cache,
                    stopped.clone(),
                    thread_force_abort.clone(),
                ));
                let result = rt.block_on(finish_bridge_local_set(
                    local,
                    dispatch,
                    stopped,
                    thread_force_abort,
                    super::rpc_session::DRAIN_TIMEOUT,
                    thread_admission,
                    lifecycle_service,
                ));
                drop(rt);
                result
            })
            .map_err(|e| anyhow::anyhow!("spawn iroh-rpc bridge thread: {e}"))?;

        Ok(Self {
            tx,
            shutdown,
            thread: Arc::new(BridgeThread::new(thread)),
            admission,
        })
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
        let force_abort = CancellationToken::new();
        let thread_force_abort = force_abort.clone();
        let admission = Arc::new(parking_lot::Mutex::new(None));
        let thread_admission = Arc::clone(&admission);
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
                let dispatch_force_abort = thread_force_abort.clone();
                let lifecycle_service = std::rc::Rc::new(std::cell::RefCell::new(None));
                let dispatch_service = std::rc::Rc::clone(&lifecycle_service);
                let dispatch = local.spawn_local(async move {
                    // Build on-thread; a failure (e.g. GPU init) is reported via
                    // the readiness channel and the bridge thread exits.
                    let built = tokio::select! {
                        biased;
                        _ = dispatch_shutdown.cancelled() => return Ok(()),
                        built = build() => built,
                    };
                    let service = match built {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return Ok(());
                        }
                    };
                    let service = std::rc::Rc::new(service);
                    *dispatch_service.borrow_mut() = Some(std::rc::Rc::clone(&service)
                        as std::rc::Rc<dyn crate::service::RequestService>);
                    if ready_tx.send(Ok(())).is_err() {
                        // Caller gave up waiting; no point serving.
                        return Ok(());
                    }
                    run_bridge_dispatch_loop(
                        service,
                        rx,
                        nonce_cache,
                        dispatch_shutdown,
                        dispatch_force_abort,
                    )
                    .await
                });
                let result = rt.block_on(finish_bridge_local_set(
                    local,
                    dispatch,
                    stopped,
                    thread_force_abort,
                    super::rpc_session::DRAIN_TIMEOUT,
                    thread_admission,
                    lifecycle_service,
                ));
                drop(rt);
                result
            })
            .map_err(|e| anyhow::anyhow!("spawn rpc bridge thread: {e}"))?;

        Ok((
            Self {
                tx,
                shutdown,
                thread: Arc::new(BridgeThread::new(thread)),
                admission,
            },
            ready_rx,
        ))
    }
}

impl LocalServiceBridge {
    fn close_admission(&self) {
        self.begin_shutdown(tokio::time::Instant::now() + super::rpc_session::DRAIN_TIMEOUT);
    }

    /// Begin draining immediately; subsequent callers reuse the first deadline.
    pub fn begin_shutdown(&self, deadline: tokio::time::Instant) -> tokio::time::Instant {
        let mut admission = self.admission.lock();
        let deadline = *admission.get_or_insert(deadline);
        self.shutdown.cancel();
        deadline
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
        tokio::task::spawn_blocking(move || thread.join())
            .await
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
    fn close_admission(&self) {
        LocalServiceBridge::close_admission(self);
    }

    fn begin_shutdown(&self, deadline: tokio::time::Instant) -> tokio::time::Instant {
        LocalServiceBridge::begin_shutdown(self, deadline)
    }

    fn process(
        &self,
        request: Bytes,
        carrier: crate::transport::carrier::CarrierContext,
        request_cancel: CancellationToken,
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
                anyhow::ensure!(
                    !self.shutdown.is_cancelled(),
                    "iroh-rpc bridge: admission closed"
                );
                permit.send(BridgeMessage {
                    request,
                    carrier,
                    request_cancel,
                    respond: respond_tx,
                });
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

    #[test]
    fn bridge_message_is_unavailable_after_cancel_or_response_drop() {
        let (respond, response) = tokio::sync::oneshot::channel();
        let mut msg = BridgeMessage {
            request: Bytes::new(),
            carrier: crate::transport::carrier::CarrierContext::inproc(),
            request_cancel: CancellationToken::new(),
            respond,
        };
        assert!(!bridge_message_has_no_live_caller(&msg));

        msg.request_cancel.cancel();
        assert!(bridge_message_has_no_live_caller(&msg));
        drop(response);
        msg.request_cancel = CancellationToken::new();
        assert!(bridge_message_has_no_live_caller(&msg));
    }

    #[tokio::test]
    async fn bridge_skips_cancelled_accepted_request_before_processing() -> Result<()> {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let (cancelled_respond, cancelled_response) = tokio::sync::oneshot::channel();
        tx.send(BridgeMessage {
            request: Bytes::from_static(b"must not be processed"),
            carrier: crate::transport::carrier::CarrierContext::inproc(),
            request_cancel: cancelled,
            respond: cancelled_respond,
        })
        .await?;

        let (active_respond, active_response) = tokio::sync::oneshot::channel();
        tx.send(BridgeMessage {
            request: Bytes::from_static(b"invalid envelope, but still dispatched"),
            carrier: crate::transport::carrier::CarrierContext::inproc(),
            request_cancel: CancellationToken::new(),
            respond: active_respond,
        })
        .await?;
        drop(tx);

        let service = std::rc::Rc::new(BridgeEcho::new(fresh_signing_key()));
        let local = tokio::task::LocalSet::new();
        let dispatch = local.spawn_local(run_bridge_dispatch_loop(
            service,
            rx,
            Arc::new(crate::envelope::InMemoryNonceCache::new()),
            CancellationToken::new(),
            CancellationToken::new(),
        ));
        local.run_until(dispatch).await??;

        assert!(
            cancelled_response.await.is_err(),
            "cancelled accepted request reached response dispatch"
        );
        assert!(
            active_response.await?.is_err(),
            "active control request did not reach normal request processing"
        );
        Ok(())
    }

    #[tokio::test]
    async fn process_request_with_cancel_rejects_expired_work_before_envelope_decode() -> Result<()>
    {
        let service = BridgeEcho::new(fresh_signing_key());
        let key = service.signing_key.clone();
        let nonce_cache = crate::envelope::InMemoryNonceCache::new();
        let cancelled = CancellationToken::new();
        cancelled.cancel();

        let result = crate::service::dispatch::process_request_with_cancel(
            b"not an envelope",
            &service,
            crate::envelope::EnvelopeVerification::AnySigner,
            &key,
            &nonce_cache,
            crate::transport::carrier::CarrierContext::inproc(),
            cancelled,
        )
        .await;
        let error = match result {
            Ok(_) => anyhow::bail!("cancelled request entered envelope verification"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("request cancelled before envelope verification"));
        Ok(())
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
        request_active: Option<Arc<std::sync::atomic::AtomicBool>>,
        lifecycle_gate: Option<(
            Arc<tokio::sync::Semaphore>,
            Arc<tokio::sync::Semaphore>,
            Arc<std::sync::atomic::AtomicBool>,
            Arc<std::sync::atomic::AtomicBool>,
            Arc<std::sync::atomic::AtomicBool>,
        )>,
        lifecycle_events: Option<Arc<parking_lot::Mutex<Vec<&'static str>>>>,
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
            self.completed
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl BridgeEcho {
        fn new(signing_key: SigningKey) -> Self {
            Self {
                drop_probe: None,
                request_gate: None,
                request_active: None,
                lifecycle_gate: None,
                lifecycle_events: None,
                name: "bridge-echo".to_owned(),
                transport: crate::transport::TransportConfig::inproc("bridge-echo-unused"),
                signing_key,
            }
        }
    }
    impl Drop for BridgeEcho {
        fn drop(&mut self) {
            if let Some(events) = &self.lifecycle_events {
                events.lock().push("service_drop");
            }
        }
    }
    #[async_trait::async_trait(?Send)]
    impl crate::service::RequestService for BridgeEcho {
        async fn on_drained(&self) -> Result<()> {
            if let Some((entered, release, completed, runtime_alive, request_inactive)) =
                &self.lifecycle_gate
            {
                if let Some(events) = &self.lifecycle_events {
                    events.lock().push("hook_start");
                }
                runtime_alive.store(
                    tokio::runtime::Handle::try_current().is_ok(),
                    std::sync::atomic::Ordering::SeqCst,
                );
                let inactive = match &self.request_active {
                    Some(active) => !active.load(std::sync::atomic::Ordering::SeqCst),
                    None => true,
                };
                request_inactive.store(inactive, std::sync::atomic::Ordering::SeqCst);
                entered.add_permits(1);
                release.acquire().await?.forget();
                if let Some(events) = &self.lifecycle_events {
                    events.lock().push("hook_end");
                }
                completed.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(())
        }
        async fn handle_request(
            &self,
            _ctx: &crate::service::EnvelopeContext,
            body: &crate::service::DecodedRequestBody,
        ) -> Result<(Vec<u8>, Option<crate::service::Continuation>)> {
            let mut out = vec![0xBE];
            out.extend_from_slice(body.bytes());
            Ok((out, None))
        }
        async fn verify_claims(&self, _ctx: &mut crate::service::EnvelopeContext) -> Result<()> {
            struct ResetRequestActive(Arc<std::sync::atomic::AtomicBool>);
            impl Drop for ResetRequestActive {
                fn drop(&mut self) {
                    self.0.store(false, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let _active_guard = self.request_active.as_ref().map(|active| {
                active.store(true, std::sync::atomic::Ordering::SeqCst);
                ResetRequestActive(active.clone())
            });
            if let Some((entered, release)) = &self.request_gate {
                entered.add_permits(1);
                release.acquire().await?.forget();
            }
            anyhow::bail!("gated claims denial")
        }
        fn build_error_payload(&self, _request_id: u64, error: &str) -> Vec<u8> {
            error.as_bytes().to_vec()
        }
        fn decode_request_body(
            &self,
            signed_body: &[u8],
        ) -> Result<crate::service::DecodedRequestBody> {
            Ok(crate::service::DecodedRequestBody::opaque(
                signed_body.to_vec(),
            ))
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
                "destructor-gate",
                move || async move { Ok(service) },
                nonce,
                0,
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
        owner
            .join()
            .map_err(|_| anyhow::anyhow!("bridge owner panicked"))?;
        assert!(
            premature.is_none(),
            "bridge drop returned before service destruction"
        );
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
            entered: entered_tx,
            release: release_rx,
            completed: Arc::clone(&completed),
        });
        let bridge = Arc::new(LocalServiceBridge::spawn(
            service,
            Arc::new(crate::envelope::InMemoryNonceCache::new()),
            0,
        )?);
        let owner = Arc::clone(&bridge);
        let first = tokio::spawn(async move { owner.shutdown().await });
        tokio::task::spawn_blocking(move || entered_rx.recv_timeout(Duration::from_secs(5)))
            .await??;
        first.abort();
        let _ = first.await;
        let premature = tokio::time::timeout(Duration::from_millis(100), bridge.shutdown())
            .await
            .is_ok();
        release_tx.send(())?;
        tokio::time::timeout(Duration::from_secs(5), bridge.shutdown()).await??;
        assert!(
            !premature,
            "cancelled waiter lost the in-progress thread join"
        );
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
            Arc::new(crate::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let bridge = Arc::new(bridge);
        let retained = Arc::clone(&bridge);
        tokio::time::timeout(Duration::from_secs(5), entered_rx).await??;
        tokio::time::timeout(Duration::from_secs(5), bridge.shutdown()).await??;
        assert!(completed.load(std::sync::atomic::Ordering::SeqCst));
        assert!(ready.await.is_err());
        assert!(retained
            .process(
                Bytes::new(),
                crate::transport::carrier::CarrierContext::iroh(),
                CancellationToken::new(),
            )
            .await
            .is_err());
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
        store.bind(
            key.verifying_key().to_bytes(),
            &crate::crypto::pq::ml_dsa_sk_to_vk(&pq),
        );
        let store = Arc::new(store);
        crate::envelope::install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(store.clone()),
        })?;
        crate::envelope::install_response_verify_config(crate::envelope::ResponseVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(store),
        })?;
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let mut service = BridgeEcho::new(key.clone());
        service.request_gate = Some((entered.clone(), release.clone()));
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let mut responses = Vec::new();
        for payload in [b"one", b"two"] {
            let envelope = crate::envelope::SignedEnvelope::new_signed_hybrid(
                crate::envelope::RequestEnvelope::new(payload.to_vec()),
                &key,
                &pq,
            );
            let mut message = capnp::message::Builder::new_default();
            envelope.write_to(
                &mut message.init_root::<crate::common_capnp::signed_envelope::Builder>(),
            );
            let mut wire = Vec::new();
            capnp::serialize::write_message(&mut wire, &message)?;
            let (respond, response) = tokio::sync::oneshot::channel();
            tx.try_send(BridgeMessage {
                request: Bytes::from(wire),
                carrier: crate::transport::carrier::CarrierContext::inproc(),
                request_cancel: CancellationToken::new(),
                respond,
            })
            .map_err(|_| anyhow::anyhow!("queue admission failed"))?;
            responses.push(response);
        }
        // Both requests are queued BEFORE shutdown, before dispatch is polled.
        let stopped = CancellationToken::new();
        let force_abort = CancellationToken::new();
        stopped.cancel();
        let local = tokio::task::LocalSet::new();
        let service = std::rc::Rc::new(service);
        let lifecycle_service = std::rc::Rc::new(std::cell::RefCell::new(Some(
            std::rc::Rc::clone(&service) as std::rc::Rc<dyn crate::service::RequestService>,
        )));
        let dispatch = local.spawn_local(run_bridge_dispatch_loop(
            service,
            rx,
            Arc::new(crate::envelope::InMemoryNonceCache::new()),
            stopped.clone(),
            force_abort.clone(),
        ));
        let drain = finish_bridge_local_set(
            local,
            dispatch,
            stopped,
            force_abort,
            super::super::rpc_session::DRAIN_TIMEOUT,
            Arc::new(parking_lot::Mutex::new(None)),
            lifecycle_service,
        );
        let coordinate = async {
            tokio::time::timeout(Duration::from_secs(5), entered.acquire_many(2))
                .await??
                .forget();
            assert!(tx.is_closed(), "new admissions remained open during grace");
            release.add_permits(2);
            for response in responses {
                let wire = response.await??;
                let (_, payload) =
                    crate::envelope::unwrap_response(&wire, Some(&key.verifying_key()))?;
                assert_eq!(
                    payload,
                    crate::service::dispatch::DISPATCH_DENIED.as_bytes()
                );
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
            admission: Arc::new(parking_lot::Mutex::new(None)),
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
        store.bind(
            key.verifying_key().to_bytes(),
            &crate::crypto::pq::ml_dsa_sk_to_vk(&pq),
        );
        crate::envelope::install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        })?;
        let envelope = crate::envelope::SignedEnvelope::new_signed_hybrid(
            crate::envelope::RequestEnvelope::new(b"held".to_vec()),
            &key,
            &pq,
        );
        let mut message = capnp::message::Builder::new_default();
        envelope
            .write_to(&mut message.init_root::<crate::common_capnp::signed_envelope::Builder>());
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message)?;
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let mut service = BridgeEcho::new(key);
        service.request_gate = Some((entered.clone(), Arc::new(tokio::sync::Semaphore::new(0))));
        let request_active = Arc::new(std::sync::atomic::AtomicBool::new(false));
        service.request_active = Some(request_active);
        let hook_entered = Arc::new(tokio::sync::Semaphore::new(0));
        let hook_release = Arc::new(tokio::sync::Semaphore::new(0));
        let hook_completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_runtime_alive = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook_request_inactive = Arc::new(std::sync::atomic::AtomicBool::new(false));
        struct ReleaseLifecycleOnDrop(Arc<tokio::sync::Semaphore>);
        impl Drop for ReleaseLifecycleOnDrop {
            fn drop(&mut self) {
                self.0.add_permits(1);
            }
        }
        let _release_lifecycle_on_failure = ReleaseLifecycleOnDrop(hook_release.clone());
        let lifecycle_events = Arc::new(parking_lot::Mutex::new(Vec::new()));
        service.lifecycle_gate = Some((
            hook_entered.clone(),
            hook_release.clone(),
            hook_completed.clone(),
            hook_runtime_alive.clone(),
            hook_request_inactive.clone(),
        ));
        service.lifecycle_events = Some(lifecycle_events.clone());
        let (drop_entered, _) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        release.send(())?;
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        service.drop_probe = Some(BridgeDropProbe {
            entered: drop_entered,
            release: released,
            completed: dropped.clone(),
        });
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let (respond, response) = tokio::sync::oneshot::channel();
        tx.try_send(BridgeMessage {
            request: Bytes::from(wire),
            carrier: crate::transport::carrier::CarrierContext::inproc(),
            request_cancel: CancellationToken::new(),
            respond,
        })
        .map_err(|_| anyhow::anyhow!("queue admission failed"))?;
        let stopped = CancellationToken::new();
        let thread_stopped = stopped.clone();
        let force_abort = CancellationToken::new();
        let thread_force_abort = force_abort.clone();
        let thread = std::thread::spawn(move || {
            // Real dispatch/queue/destructor, with only the runtime clock made
            // virtual: exercise the unchanged production 40-second bound.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .start_paused(true)
                .build()
                .map_err(|e| BridgeShutdownError::Runtime(e.to_string()))?;
            let local = tokio::task::LocalSet::new();
            let service = std::rc::Rc::new(service);
            let lifecycle_service = std::rc::Rc::new(std::cell::RefCell::new(Some(
                std::rc::Rc::clone(&service) as std::rc::Rc<dyn crate::service::RequestService>,
            )));
            let dispatch = local.spawn_local(run_bridge_dispatch_loop(
                service,
                rx,
                Arc::new(crate::envelope::InMemoryNonceCache::new()),
                thread_stopped.clone(),
                thread_force_abort.clone(),
            ));
            rt.block_on(finish_bridge_local_set(
                local,
                dispatch,
                thread_stopped,
                thread_force_abort,
                super::super::rpc_session::DRAIN_TIMEOUT,
                Arc::new(parking_lot::Mutex::new(None)),
                lifecycle_service,
            ))
        });
        let bridge = Arc::new(LocalServiceBridge {
            tx,
            shutdown: stopped,
            thread: Arc::new(BridgeThread::new(thread)),
            admission: Arc::new(parking_lot::Mutex::new(None)),
        });
        tokio::time::timeout(Duration::from_secs(5), entered.acquire())
            .await??
            .forget();
        let first_bridge = bridge.clone();
        let second_bridge = bridge.clone();
        let first_shutdown = tokio::spawn(async move { first_bridge.shutdown().await });
        let second_shutdown = tokio::spawn(async move { second_bridge.shutdown().await });
        tokio::time::timeout(Duration::from_secs(5), hook_entered.acquire())
            .await??
            .forget();
        assert!(!hook_completed.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!dropped.load(std::sync::atomic::Ordering::SeqCst));
        hook_release.add_permits(1);
        let (first, second) = tokio::join!(first_shutdown, second_shutdown);
        let first = first?;
        let second = second?;
        assert_eq!(first, Err(BridgeShutdownError::DrainTimedOut));
        assert_eq!(second, first);
        assert_eq!(bridge.shutdown().await, first);
        assert!(
            response.await.is_err(),
            "forced deadline retained the accepted response sender"
        );
        assert!(hook_completed.load(std::sync::atomic::Ordering::SeqCst));
        assert!(hook_runtime_alive.load(std::sync::atomic::Ordering::SeqCst));
        assert!(hook_request_inactive.load(std::sync::atomic::Ordering::SeqCst));
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            &*lifecycle_events.lock(),
            &["hook_start", "hook_end", "service_drop"]
        );
        Ok(())
    }

    fn deadline_test_identity() -> Result<SigningKey> {
        let key = fresh_signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&key);
        let mut store = crate::envelope::KeyedPqTrustStore::new();
        store.bind(
            key.verifying_key().to_bytes(),
            &crate::crypto::pq::ml_dsa_sk_to_vk(&pq),
        );
        crate::envelope::install_verify_config(crate::envelope::EnvelopeVerifyConfig {
            policy: crate::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        })?;
        Ok(key)
    }

    fn deadline_test_bridge(
        key: &SigningKey,
    ) -> Result<(
        Arc<LocalServiceBridge>,
        Arc<tokio::sync::Semaphore>,
        Arc<tokio::sync::Semaphore>,
        Arc<std::sync::atomic::AtomicBool>,
    )> {
        let entered = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut service = BridgeEcho::new(key.clone());
        service.request_gate = Some((entered.clone(), release.clone()));
        let (drop_entered, _) = std::sync::mpsc::channel();
        let (drop_release, released) = std::sync::mpsc::channel();
        drop_release.send(())?;
        service.drop_probe = Some(BridgeDropProbe {
            entered: drop_entered,
            release: released,
            completed: dropped.clone(),
        });
        let bridge = Arc::new(LocalServiceBridge::spawn(
            service,
            Arc::new(crate::envelope::InMemoryNonceCache::new()),
            0,
        )?);
        Ok((bridge, entered, release, dropped))
    }

    async fn exercise_transport_bridge_deadline<T, F, D>(
        transport: T,
        key: &SigningKey,
        bridge: Arc<LocalServiceBridge>,
        entered: Arc<tokio::sync::Semaphore>,
        release: Arc<tokio::sync::Semaphore>,
        dropped: Arc<std::sync::atomic::AtomicBool>,
        forced: bool,
        drain: D,
    ) -> Result<()>
    where
        T: crate::transport::Transport + 'static,
        F: Future<Output = Result<()>> + Send + 'static,
        D: FnOnce(tokio::time::Instant) -> F,
    {
        let mut kem = crate::crypto::hybrid_kem::KeyedKemTrustStore::new();
        kem.bind(
            key.verifying_key().to_bytes(),
            crate::node_identity::derive_mesh_kem_recipient(key)?.public(),
        );
        let pq = crate::node_identity::derive_mesh_mldsa_key(key);
        let mut response_trust = crate::envelope::KeyedPqTrustStore::new();
        response_trust.bind(
            key.verifying_key().to_bytes(),
            &crate::crypto::pq::ml_dsa_sk_to_vk(&pq),
        );
        let client = Arc::new(
            crate::rpc_client::RpcClientImpl::new(
                crate::signer::LocalSigner::new(key.clone()),
                transport,
                Some(key.verifying_key()),
            )
            .with_request_kem_store(Arc::new(kem))
            .with_response_pq_store(Arc::new(response_trust)),
        );
        // Release on assertion/error too: a regression must fail, not strand
        // the real worker behind its fixture gate during test unwinding.
        struct ReleaseOnDrop(Arc<tokio::sync::Semaphore>);
        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                self.0.add_permits(1);
            }
        }
        let _cleanup = ReleaseOnDrop(release.clone());
        let caller = client.clone();
        let response = tokio::spawn(async move {
            caller
                .call_for_service("bridge-echo", b"held".to_vec())
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), entered.acquire())
            .await??
            .forget();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        assert_eq!(bridge.begin_shutdown(deadline), deadline);
        assert_eq!(
            bridge.begin_shutdown(deadline + Duration::from_secs(40)),
            deadline,
            "later carrier shutdown must not extend the first grace"
        );
        assert!(bridge
            .process(
                Bytes::new(),
                crate::transport::carrier::CarrierContext::inproc(),
                CancellationToken::new(),
            )
            .await
            .is_err());
        assert!(!dropped.load(std::sync::atomic::Ordering::SeqCst));
        let carrier = tokio::spawn(drain(deadline));
        if !forced {
            release.add_permits(1);
            let payload = tokio::time::timeout(Duration::from_secs(2), response).await???;
            assert_eq!(
                payload,
                crate::service::dispatch::DISPATCH_DENIED.as_bytes()
            );
        } else {
            // The accepted RPC is still held: neither a caller cancellation
            // nor an already-finished request can make the deadline test pass.
            assert!(!response.is_finished());
            assert!(
                tokio::time::timeout(Duration::from_secs(3), response)
                    .await??
                    .is_err(),
                "held request must be cancelled at deadline"
            );
        }
        tokio::time::timeout(Duration::from_secs(3), carrier).await???;
        // Deliberately join AFTER the transport drain too: this must reuse the
        // established deadline/result, never grant the bridge another 40s.
        let result = tokio::time::timeout(Duration::from_secs(1), bridge.shutdown()).await?;
        if forced {
            assert_eq!(result, Err(BridgeShutdownError::DrainTimedOut));
        } else {
            result?;
            assert!(
                tokio::time::Instant::now() < deadline,
                "released request did not drain during grace"
            );
        }
        assert!(
            dropped.load(std::sync::atomic::Ordering::SeqCst),
            "join returned before destructor despite retained client"
        );
        assert_eq!(
            bridge.begin_shutdown(tokio::time::Instant::now() + Duration::from_secs(40)),
            deadline
        );
        // Keep the actual network client alive through the completed join.
        drop(client);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn irohrpc_bridge_shared_deadline_replies_or_forces_join() -> Result<()> {
        let key = deadline_test_identity()?;
        for forced in [false, true] {
            let (bridge, entered, release, dropped) = deadline_test_bridge(&key)?;
            let server = IrohSubstrate::new_test(
                fresh_key(),
                NoopHandler::new("moq"),
                IrohRpcProtocolHandler::with_stream_limit(
                    bridge.clone(),
                    key.clone(),
                    DEFAULT_STREAM_LIMIT,
                ),
            )
            .await?;
            let client = IrohSubstrate::new_test(
                fresh_key(),
                NoopHandler::new("client moq"),
                NoopHandler::new("client rpc"),
            )
            .await?;
            let connection = client
                .connect(direct_addr(&server), ALPN_HYPRSTREAM_RPC)
                .await?;
            exercise_transport_bridge_deadline(
                crate::transport::iroh_transport::IrohTransport::new(connection),
                &key,
                bridge,
                entered,
                release,
                dropped,
                forced,
                move |_| async move { server.shutdown().await },
            )
            .await?;
            client.shutdown().await?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quinnrpc_bridge_shared_deadline_replies_or_forces_join() -> Result<()> {
        use crate::transport::quinn_transport::{connect_pinned, QuinnRpcServer, QuinnTransport};
        crate::transport::pq_provider::install_pq_crypto_provider()?;
        let key = deadline_test_identity()?;
        for forced in [false, true] {
            let (bridge, entered, release, dropped) = deadline_test_bridge(&key)?;
            let certificate = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
            let cert_der = certificate.cert.der().to_vec();
            let private = rustls::pki_types::PrivateKeyDer::Pkcs8(
                rustls::pki_types::PrivatePkcs8KeyDer::from(certificate.key_pair.serialize_der()),
            );
            let server = web_transport_quinn::ServerBuilder::new()
                .with_addr("127.0.0.1:0".parse()?)
                .with_certificate(
                    vec![rustls::pki_types::CertificateDer::from(cert_der.clone())],
                    private,
                )?;
            let address = server.local_addr()?;
            let rpc = QuinnRpcServer::with_capacity(
                server,
                bridge.clone(),
                key.clone(),
                DEFAULT_STREAM_LIMIT,
            );
            let limit = rpc.stream_limit();
            let capacity = rpc.capacity();
            let token = rpc.shutdown_token();
            let task = tokio::spawn(rpc.run());
            let session = connect_pinned(address, &cert_der).await?;
            exercise_transport_bridge_deadline(
                QuinnTransport::new(session),
                &key,
                bridge,
                entered,
                release,
                dropped,
                forced,
                move |deadline| async move {
                    QuinnRpcServer::shutdown_until(&limit, capacity, &token, deadline).await;
                    task.await??;
                    Ok(())
                },
            )
            .await?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn udsrpc_bridge_shared_deadline_replies_or_forces_join() -> Result<()> {
        let key = deadline_test_identity()?;
        for forced in [false, true] {
            let (bridge, entered, release, dropped) = deadline_test_bridge(&key)?;
            let directory = std::env::temp_dir()
                .join(format!("hyprstream-r3-{}", hex::encode(&fresh_key()[..8])));
            std::fs::create_dir(&directory)?;
            let path = directory.join("rpc.sock");
            let config = crate::transport::TransportConfig::ipc(path.clone());
            let shutdown = Arc::new(tokio::sync::Notify::new());
            struct StopOnDrop(Arc<tokio::sync::Notify>);
            impl Drop for StopOnDrop {
                fn drop(&mut self) {
                    self.0.notify_one();
                }
            }
            let _stop = StopOnDrop(shutdown.clone());
            let (ready, readiness) = tokio::sync::oneshot::channel();
            let processor = bridge.clone();
            let signing_key = key.clone();
            let stopped = shutdown.clone();
            let task = tokio::spawn(async move {
                crate::service::serve::serve_bridged(
                    &config,
                    processor,
                    signing_key,
                    stopped,
                    Some(ready),
                )
                .await
            });
            tokio::time::timeout(Duration::from_secs(5), readiness).await??;
            exercise_transport_bridge_deadline(
                crate::transport::lazy_uds::LazyUdsTransport::new(path),
                &key,
                bridge,
                entered,
                release,
                dropped,
                forced,
                move |_| async move {
                    shutdown.notify_one();
                    task.await??;
                    Ok(())
                },
            )
            .await?;
        }
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn transport_shared_deadline_does_not_reset_for_held_response_permits() -> Result<()> {
        // Hold real drain permits independently of bridge completion, as a
        // stuck response write can do. A later carrier cannot get a new 40s.
        let quinn_limit = Arc::new(Semaphore::new(1));
        let uds_limit = Arc::new(Semaphore::new(1));
        let _quinn_permit = quinn_limit.clone().acquire_owned().await?;
        let _uds_permit = uds_limit.clone().acquire_owned().await?;
        let deadline = tokio::time::Instant::now() + super::super::rpc_session::DRAIN_TIMEOUT;
        tokio::time::advance(Duration::from_secs(30)).await;
        let quinn_token = CancellationToken::new();
        let uds_token = CancellationToken::new();
        let quinn_stopped = quinn_token.clone();
        let uds_stopped = uds_token.clone();
        let task = tokio::spawn(async move {
            tokio::join!(
                crate::transport::quinn_transport::QuinnRpcServer::shutdown_until(
                    &quinn_limit,
                    1,
                    &quinn_stopped,
                    deadline
                ),
                crate::transport::uds_server::UdsRpcServer::shutdown_until(
                    &uds_limit,
                    1,
                    &uds_stopped,
                    deadline
                ),
            );
            assert!(quinn_limit.is_closed() && uds_limit.is_closed());
        });
        quinn_token.cancelled().await;
        uds_token.cancelled().await;
        tokio::time::advance(Duration::from_secs(9)).await;
        assert!(
            !task.is_finished(),
            "drain expired before the shared deadline"
        );
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        assert!(
            task.is_finished(),
            "carrier started a fresh grace instead of using the owner's remaining budget"
        );
        task.await?;
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

    struct DisconnectProbeProcessor {
        started: tokio::sync::mpsc::UnboundedSender<(Bytes, CancellationToken)>,
        finished: tokio::sync::mpsc::UnboundedSender<Bytes>,
        release: Arc<Semaphore>,
    }

    impl crate::transport::rpc_session::sealed::Sealed for DisconnectProbeProcessor {
        fn admits_untrusted_carrier(&self) -> bool {
            // This processor exists only inside the transport unit test and
            // deliberately accepts opaque test bytes so the real iroh stream
            // closure behavior can be exercised without fabricating an
            // application envelope. Production processors remain sealed to
            // the canonical authenticated dispatch bridge.
            true
        }
    }

    impl IrohRequestProcessor for DisconnectProbeProcessor {
        fn process(
            &self,
            request: Bytes,
            _carrier: crate::transport::carrier::CarrierContext,
            request_cancel: CancellationToken,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>> {
            let started = self.started.clone();
            let finished = self.finished.clone();
            let release = Arc::clone(&self.release);
            Box::pin(async move {
                started.send((request.clone(), request_cancel)).ok();
                release.acquire().await?.forget();
                finished.send(request.clone()).ok();
                Ok(request)
            })
        }
    }

    /// A complete request FIN is not cancellation. STOP_SENDING on only one
    /// response stream trips only that server-owned request token, and the
    /// accepted processor still reaches its terminal path before the stream
    /// handler returns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rpc_stop_sending_cancels_only_its_request_after_full_request_fin() -> Result<()> {
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (finished_tx, mut finished_rx) = tokio::sync::mpsc::unbounded_channel();
        let release = Arc::new(Semaphore::new(0));
        let processor = DisconnectProbeProcessor {
            started: started_tx,
            finished: finished_tx,
            release: Arc::clone(&release),
        };
        let handler = IrohRpcProtocolHandler::new(processor, fresh_signing_key());
        let server = IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let client = IrohSubstrate::new_test(
            fresh_key(),
            NoopHandler::new("c-moq"),
            NoopHandler::new("c-rpc"),
        )
        .await?;
        let conn = client
            .connect(direct_addr(&server), ALPN_HYPRSTREAM_RPC)
            .await?;

        let (mut abandoned_send, mut abandoned_recv) = conn.open_bi().await?;
        abandoned_send.write_all(b"abandoned").await?;
        abandoned_send.finish()?;
        let (mut retained_send, mut retained_recv) = conn.open_bi().await?;
        retained_send.write_all(b"retained").await?;
        retained_send.finish()?;

        let mut signals = Vec::new();
        for _ in 0..2 {
            signals.push(
                tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
                    .await?
                    .context("processor did not accept complete request")?,
            );
        }
        let abandoned_token = signals
            .iter()
            .find(|(request, _)| request.as_ref() == b"abandoned")
            .context("abandoned request missing")?
            .1
            .clone();
        let retained_token = signals
            .iter()
            .find(|(request, _)| request.as_ref() == b"retained")
            .context("retained request missing")?
            .1
            .clone();
        assert!(!abandoned_token.is_cancelled());
        assert!(!retained_token.is_cancelled());

        // Iroh's normal RecvStream drop path sends STOP_SENDING(0). The request
        // side already completed with FIN; reproduce that exact raw QUIC signal
        // because the WebTransport adapter maps code 0 to `closed() == Ok(())`.
        abandoned_recv.stop(iroh::endpoint::VarInt::from_u32(0))?;
        tokio::time::timeout(Duration::from_secs(5), abandoned_token.cancelled())
            .await
            .context("response STOP_SENDING did not cancel its accepted request")?;
        assert!(abandoned_token.is_cancelled());
        assert!(
            !retained_token.is_cancelled(),
            "disconnect signal escaped its stream/request"
        );

        release.add_permits(2);
        let retained_response = retained_recv.read_to_end(MAX_FRAME_BYTES).await?;
        assert_eq!(retained_response, b"retained");
        assert!(
            !retained_token.is_cancelled(),
            "ordinary request FIN was misclassified as caller disconnect"
        );
        let mut terminal_requests = Vec::new();
        for _ in 0..2 {
            terminal_requests.push(
                tokio::time::timeout(Duration::from_secs(5), finished_rx.recv())
                    .await?
                    .context("accepted processor did not reach terminal completion")?,
            );
        }
        assert!(terminal_requests
            .iter()
            .any(|request| request.as_ref() == b"abandoned"));
        assert!(terminal_requests
            .iter()
            .any(|request| request.as_ref() == b"retained"));

        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }
}
