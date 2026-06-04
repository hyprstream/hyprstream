//! Iroh RPC plane — server-side protocol handler for ALPN `hyprstream-rpc/1`.
//!
//! Part of Epic #131 Phase 2 (#133). This module ships:
//!
//! - [`IrohRpcProtocolHandler`] — iroh [`ProtocolHandler`] that plugs into
//!   [`crate::transport::iroh_substrate`] under the `hyprstream-rpc/1` ALPN.
//!   Enforces a per-connection cap on concurrent streams (DoS bound) and
//!   drains in-flight requests on `ProtocolHandler::shutdown`.
//! - [`IrohRequestProcessor`] — trait callers implement to wire actual
//!   request processing (envelope verification + service dispatch).
//! - [`LocalServiceBridge`] — adapts a [`crate::service::ZmqService`]
//!   (potentially `!Send`) to the Send-bounded [`IrohRequestProcessor`].
//!
//! **Trust model**: The protocol handler does not parse `SignedEnvelope` —
//! it forwards opaque bytes to the [`IrohRequestProcessor`]. Envelope
//! verification, JWT/DPoP, and `authorize_signer` enforcement happen inside
//! the processor (which is `LocalServiceBridge` in production, delegating
//! to [`crate::transport::zmtp_quic::process_request`]). The handler does
//! hold a signing key, but uses it only to produce signed error envelopes
//! when the processor itself returns `Err` (a wire-level failure), so the
//! client always sees a parseable `ResponseEnvelope` rather than an opaque
//! EOF + Cap'n Proto parse error.
//!
//! **Wire framing**: each request is the opaque bytes of a Cap'n Proto-encoded
//! [`crate::envelope::SignedEnvelope`] written to a freshly-opened iroh bidi
//! stream. The response is symmetric. Both endpoints are our code; iroh's
//! QUIC TLS already authenticates the connection peer's Ed25519 NodeId at
//! the transport layer.

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
/// caps concurrent streams per connection, and drains in-flight requests
/// on `ProtocolHandler::shutdown` (called by `Router::shutdown`).
#[derive(Clone)]
pub struct IrohRpcProtocolHandler {
    inner: Arc<HandlerInner>,
}

struct HandlerInner {
    processor: Arc<dyn IrohRequestProcessor>,
    /// Used to produce signed error envelopes when the processor itself
    /// returns `Err` (wire-level fatal). Application errors come back from
    /// the processor pre-wrapped as signed `ResponseEnvelope` bytes.
    signing_key: SigningKey,
    /// Caps concurrent bidi streams in flight. Hold one permit per stream.
    stream_limit: Arc<Semaphore>,
    stream_limit_capacity: u32,
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
    /// Build a handler with default per-connection stream limit
    /// ([`DEFAULT_STREAM_LIMIT`]).
    pub fn new<P: IrohRequestProcessor>(processor: P, signing_key: SigningKey) -> Self {
        Self::with_stream_limit(Arc::new(processor), signing_key, DEFAULT_STREAM_LIMIT)
    }

    /// Build a handler with an explicit per-connection stream limit.
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
                shutdown: CancellationToken::new(),
            }),
        }
    }
}

impl ProtocolHandler for IrohRpcProtocolHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Adapt the raw iroh `Connection` to the transport-generic `Session`
        // abstraction and delegate to the shared accept loop. Drain semantics
        // (`shutdown` below draining the same `Semaphore`) are unchanged.
        let session = web_transport_iroh::Session::raw(conn);
        serve_rpc_connection(
            session,
            Arc::clone(&self.inner.processor),
            self.inner.signing_key.clone(),
            Arc::clone(&self.inner.stream_limit),
            self.inner.shutdown.clone(),
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
        // `acquire_many` succeeds only once every permit is returned.
        let cap = self.inner.stream_limit_capacity;
        match self.inner.stream_limit.acquire_many(cap).await {
            Ok(permits) => {
                // Keep permits drained so any post-shutdown accept also
                // sees a closed semaphore.
                permits.forget();
                self.inner.stream_limit.close();
            }
            Err(_) => {
                // Already closed; nothing to drain.
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
// LocalServiceBridge — adapt a (possibly `!Send`) ZmqService to the Send-bound
// IrohRequestProcessor trait by running the service on a dedicated LocalSet
// thread and forwarding requests over an mpsc channel.
// ============================================================================

/// Per-request payload + response slot exchanged with the bridge thread.
struct BridgeMessage {
    request: Bytes,
    respond: tokio::sync::oneshot::Sender<Result<Bytes>>,
}

/// Adapt a [`crate::service::ZmqService`] to [`IrohRequestProcessor`].
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
    /// requests through [`crate::transport::zmtp_quic::process_request`].
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
        S: crate::service::ZmqService + Send + 'static,
    {
        let cap = if queue_depth == 0 { 128 } else { queue_depth };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BridgeMessage>(cap);
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
                local.spawn_local(async move {
                    let service = std::rc::Rc::new(service);
                    while let Some(msg) = rx.recv().await {
                        let service = std::rc::Rc::clone(&service);
                        let nonce_cache = Arc::clone(&nonce_cache);
                        // Each request gets its own local task so a slow
                        // handler doesn't head-of-line the queue.
                        tokio::task::spawn_local(async move {
                            // Re-derive the signing key per call: cheap clone,
                            // tracks any future rotation in the service.
                            let signing_key = service.signing_key();
                            let result = crate::transport::zmtp_quic::process_request(
                                msg.request.as_ref(),
                                &*service,
                                crate::envelope::EnvelopeVerification::AnySigner,
                                &signing_key,
                                &nonce_cache,
                            )
                            .await
                            .and_then(|(bytes, cont)| {
                                // Streaming continuations are not wired on
                                // the iroh RPC plane — streaming moves to
                                // moq-net (`moql` ALPN, Phase 3, #134). In
                                // debug, panic loudly; in release, return an
                                // error so the wire emits a signed error
                                // envelope (handler's build_error_envelope)
                                // rather than silently dropping the stream.
                                if cont.is_some() {
                                    debug_assert!(
                                        false,
                                        "iroh-rpc bridge: streaming continuation not supported \
                                         — wait for Phase 3 (#134)"
                                    );
                                    return Err(anyhow::anyhow!(
                                        "iroh-rpc bridge: service returned a streaming \
                                         continuation, which is not supported on the iroh \
                                         RPC plane (Phase 3 / #134 moves streaming to moq-net)"
                                    ));
                                }
                                Ok(Bytes::from(bytes))
                            });
                            let _ = msg.respond.send(result);
                        });
                    }
                });
                rt.block_on(local);
            })
            .map_err(|e| anyhow::anyhow!("spawn iroh-rpc bridge thread: {e}"))?;

        Ok(Self { tx })
    }
}

impl IrohRequestProcessor for LocalServiceBridge {
    fn process(
        &self,
        request: Bytes,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>> {
        let tx = self.tx.clone();
        Box::pin(async move {
            let (respond_tx, respond_rx) = tokio::sync::oneshot::channel();
            tx.send(BridgeMessage {
                request,
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
            IrohSubstrate::new(fresh_key(), NoopHandler::new("moq-not-wired"), rpc_handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
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
        let server = IrohSubstrate::new(
            fresh_key(),
            NoopHandler::new("moq"),
            IrohRpcProtocolHandler::new(processor, fresh_signing_key()),
        )
        .await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
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

        let server = IrohSubstrate::new(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
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

        let server = IrohSubstrate::new(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
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

        let server = IrohSubstrate::new(fresh_key(), NoopHandler::new("moq"), handler).await?;
        let server_addr = direct_addr(&server);

        let client = IrohSubstrate::new(
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
