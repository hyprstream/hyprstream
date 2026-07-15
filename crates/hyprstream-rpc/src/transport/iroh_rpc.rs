//! Iroh RPC plane — server-side protocol handler for ALPN `hyprstream-rpc/1`.
//!
//! Part of Epic #131 Phase 2 (#133). This module ships:
//!
//! - [`IrohRpcProtocolHandler`] — iroh [`ProtocolHandler`] that plugs into
//!   [`crate::transport::iroh_substrate`] under the `hyprstream-rpc/1` ALPN.
//!   Enforces a server-wide cap on concurrent streams (DoS bound) and
//!   drains in-flight requests on `ProtocolHandler::shutdown`.
//! - [`IrohRequestProcessor`] — trait callers implement to wire actual
//!   request processing (envelope verification + service dispatch).
//! - [`LocalServiceBridge`] — adapts a [`crate::service::RequestService`]
//!   (potentially `!Send`) to the Send-bounded [`IrohRequestProcessor`].
//!
//! **Trust model**: The protocol handler does not parse `SignedEnvelope` —
//! it forwards opaque bytes to the [`IrohRequestProcessor`]. Envelope
//! verification, JWT/DPoP, and `authorize_signer` enforcement happen inside
//! the processor (which is `LocalServiceBridge` in production, delegating
//! to [`crate::service::dispatch::process_request`]). The handler does
//! hold a signing key, but uses it only to produce signed error envelopes
//! when the processor itself returns `Err` (a wire-level failure), so the
//! client always sees a parseable `ResponseEnvelope` rather than an opaque
//! EOF + Cap'n Proto parse error.
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
    DEFAULT_STREAM_LIMIT, IrohRequestProcessor, MAX_FRAME_BYTES, build_error_envelope, from_fn,
    read_to_cap, serve_rpc_connection,
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
    /// Used to produce signed error envelopes when the processor itself
    /// returns `Err` (wire-level fatal). Application errors come back from
    /// the processor pre-wrapped as signed `ResponseEnvelope` bytes.
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
            crate::transport::carrier::CarrierContext::iroh(),
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
) where
    S: crate::service::RequestService + 'static,
{
    while let Some(msg) = rx.recv().await {
        let service = std::rc::Rc::clone(&service);
        let nonce_cache = Arc::clone(&nonce_cache);
        tokio::task::spawn_local(async move {
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
}

/// Adapt a [`crate::service::RequestService`] to [`IrohRequestProcessor`].
///
/// Spins up a dedicated thread running a single-threaded tokio runtime with
/// a `LocalSet`. The service runs on that thread (compatible with both `Send`
/// and `!Send` services like inference); requests arriving on the iroh accept
/// loop are forwarded via an `mpsc` channel and awaited via `oneshot`.
///
/// **Lifecycle**: in-flight requests are drained when the
/// [`IrohRpcProtocolHandler`] holding this bridge is shut down via
/// `Router::shutdown` (each in-flight handler task holds a semaphore permit;
/// the handler's drain waits for all permits to return, which only happens
/// once the bridge has produced each response). After that, dropping the
/// last `Arc<LocalServiceBridge>` closes the mpsc Sender, the bridge thread
/// exits its receive loop, and `LocalSet::block_on` returns.
pub struct LocalServiceBridge {
    tx: tokio::sync::mpsc::Sender<BridgeMessage>,
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

        std::thread::Builder::new()
            .name(format!("iroh-rpc-bridge:{service_name}"))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!(error = ?e, "bridge: failed to build runtime");
                        return;
                    }
                };
                let local = tokio::task::LocalSet::new();
                local.spawn_local(run_bridge_dispatch_loop(
                    std::rc::Rc::new(service),
                    rx,
                    nonce_cache,
                ));
                rt.block_on(local);
            })
            .map_err(|e| anyhow::anyhow!("spawn iroh-rpc bridge thread: {e}"))?;

        Ok(Self { tx })
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

        std::thread::Builder::new()
            .name(format!("rpc-bridge:{thread_name}"))
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(anyhow::anyhow!("bridge runtime: {e}")));
                        return;
                    }
                };
                let local = tokio::task::LocalSet::new();
                local.spawn_local(async move {
                    // Build on-thread; a failure (e.g. GPU init) is reported via
                    // the readiness channel and the bridge thread exits.
                    let service = match build().await {
                        Ok(s) => std::rc::Rc::new(s),
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    if ready_tx.send(Ok(())).is_err() {
                        // Caller gave up waiting; no point serving.
                        return;
                    }
                    run_bridge_dispatch_loop(service, rx, nonce_cache).await;
                });
                rt.block_on(local);
            })
            .map_err(|e| anyhow::anyhow!("spawn rpc bridge thread: {e}"))?;

        Ok((Self { tx }, ready_rx))
    }
}

impl IrohRequestProcessor for LocalServiceBridge {
    fn process(
        &self,
        request: Bytes,
        carrier: crate::transport::carrier::CarrierContext,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>> {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (respond_tx, respond_rx) = tokio::sync::oneshot::channel();
            tx.send(BridgeMessage {
                request,
                carrier,
                respond: respond_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("iroh-rpc bridge: channel closed"))?;
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
        ALPN_HYPRSTREAM_RPC, ALPN_MOQ_LITE, IrohSubstrate, NoopHandler,
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
    }
    impl BridgeEcho {
        fn new(signing_key: SigningKey) -> Self {
            Self {
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
        ready.await.map_err(|_| anyhow::anyhow!("readiness dropped"))??;

        // Builder fails (e.g. GPU init error) → the error surfaces on readiness,
        // not silently as a later channel-closed.
        let (_bridge2, ready2) = LocalServiceBridge::spawn_with::<_, _, BridgeEcho>(
            "fail",
            move || async move { Err(anyhow::anyhow!("boom")) },
            nonce,
            0,
        )?;
        let build_result = ready2.await.map_err(|_| anyhow::anyhow!("readiness dropped"))?;
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
        let rpc_handler = IrohRpcProtocolHandler::new(processor, fresh_signing_key());

        let server =
            IrohSubstrate::new_test(fresh_key(), NoopHandler::new("moq-not-wired"), rpc_handler).await?;
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
            IrohRpcProtocolHandler::new(processor, fresh_signing_key()),
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
        let handler = IrohRpcProtocolHandler::new(processor, server_signing);

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
        let (request_id, payload) = crate::envelope::unwrap_response(&resp_bytes, Some(&server_vk))?;
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
        let handler = IrohRpcProtocolHandler::with_stream_limit(
            Arc::new(processor),
            fresh_signing_key(),
            2,
        );

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
        let handler = IrohRpcProtocolHandler::with_stream_limit(
            Arc::new(processor),
            fresh_signing_key(),
            1,
        )
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
        stall_send.write_all(b"x").await.context("write stall byte")?;
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
        let handler = IrohRpcProtocolHandler::new(processor, fresh_signing_key());

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
        entered.notified().await;
        server.shutdown().await?;

        let resp = req_task.await??;
        assert_eq!(&resp[..], b"drained-ok");

        client.shutdown().await?;
        Ok(())
    }

}
