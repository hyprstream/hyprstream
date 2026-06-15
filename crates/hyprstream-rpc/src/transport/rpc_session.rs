//! Transport-generic RPC plane core — Epic #131 / moq M1 (#151).
//!
//! This module generalises the iroh-specific RPC server/client (originally in
//! [`super::iroh_rpc`] and [`super::iroh_transport`]) over the
//! [`web_transport_trait::Session`] abstraction, so the RPC plane is
//! transport-pluggable: any `Session` impl (iroh's `web-transport-iroh`,
//! quinn's `web-transport-quinn`, or a future WASM `web-transport-web`) can
//! carry the exact same Cap'n Proto bidi wire protocol with identical DoS
//! bounds, graceful-drain, and error-envelope semantics.
//!
//! - [`serve_rpc_connection`] — the generic server accept loop. Terminates
//!   `SignedEnvelope` bidi streams on a [`Session`], dispatches each through an
//!   [`IrohRequestProcessor`], caps concurrent streams per connection (DoS
//!   bound) and drains in-flight requests on `shutdown`.
//! - [`SessionRpcTransport`] — the generic client transport implementing
//!   [`crate::transport_traits::Transport`] over a [`Session`]. Each `send()`
//!   opens a fresh bidi stream.
//! - [`IrohRequestProcessor`] — the pluggable request-processing trait
//!   (retained under its original name to avoid churn in callers).
//! - [`build_error_envelope`] — signed error-envelope synthesis.
//!
//! **Wire framing**: each request/response is the opaque bytes of a
//! Cap'n Proto-encoded envelope written to a freshly-opened bidi stream. The
//! length is delimited by stream FIN (no explicit length prefix); reads are
//! bounded by [`MAX_FRAME_BYTES`] via [`read_to_cap`].

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use bytes::Bytes;
use ed25519_dalek::SigningKey;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use web_transport_trait::{RecvStream, SendStream, Session};

use crate::transport_traits::{PublishSink, Transport};

/// Hard cap on per-frame size (request or response) for the RPC control plane.
///
/// SECURITY (#162): the original 64 MiB cap would let an attacker buffer up to
/// `64 MiB × DEFAULT_STREAM_LIMIT` (= 4 GiB) across concurrent streams before
/// any application-layer rejection. The RPC plane carries control messages only
/// (SQL queries, metric batches, Cap'n Proto envelopes) — no ingest blobs,
/// no streaming chunks. Those belong on the streaming ALPN (moq-net). 4 MiB is
/// generous for any realistic control-plane payload.
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024; // 4 MiB

/// Default concurrent-stream cap. NOTE: the `Arc<Semaphore>` this sizes is
/// shared **server-wide** across all connections (one `Arc<HandlerInner>`),
/// not per-connection — so this bounds total in-flight streams for the whole
/// server, capping memory to `MAX_FRAME_BYTES × DEFAULT_STREAM_LIMIT` = 256 MiB.
pub const DEFAULT_STREAM_LIMIT: usize = 64;

/// Grace period for [`SendStream::closed`] after writing the response — if
/// the peer crashes without acking the FIN, we don't leak a task forever.
pub const STOPPED_GRACE: Duration = Duration::from_secs(5);

/// Maximum wall-clock time the server will spend reading a single request
/// frame before abandoning the stream and releasing its semaphore permit.
///
/// SECURITY (#159): without this bound an UNAUTHENTICATED peer can open
/// `DEFAULT_STREAM_LIMIT` bidi streams, send one byte on each, and stall —
/// each stalled read holds a server-wide permit forever, wedging the whole
/// server before any envelope is verified (a pre-auth slowloris) and hanging
/// the shutdown drain. Bounding the *total* read (not just per-`read()` idle)
/// also defeats a trickle attacker who dribbles one byte per idle window.
pub const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Default per-call timeout on the client side when the caller passes `None`.
/// Distinct from [`REQUEST_READ_TIMEOUT`] (server-side per-request read budget).
const DEFAULT_CLIENT_TIMEOUT: Duration = Duration::from_secs(30);

/// Upper bound on how long graceful shutdown waits for in-flight streams to
/// drain before forcing teardown. With [`REQUEST_READ_TIMEOUT`] bounding each
/// read, well-behaved drains finish quickly; this is a backstop so a wedged
/// processor or transport can't hang shutdown forever (#159).
///
/// Must exceed [`REQUEST_READ_TIMEOUT`] (30 s) so in-flight requests can finish
/// before the connection tears down. The 10 s margin is intentional.
pub const DRAIN_TIMEOUT: Duration = Duration::from_secs(40);

/// Default cap on concurrent accepted connections per server (#162). Bounds
/// fd/memory from a peer that opens many connections that each sit idle (no
/// streams) — the per-stream [`DEFAULT_STREAM_LIMIT`] does not cover that.
/// Connections beyond the cap are rejected (dropped) rather than queued, so a
/// flood can't build unbounded backpressure.
pub const DEFAULT_CONNECTION_LIMIT: usize = 256;

/// Maximum time the server waits for a peer's WebTransport/QUIC handshake to
/// complete before abandoning it (#162). Bounds a peer that completes the QUIC
/// handshake then stalls the WebTransport CONNECT. Resolved inside the
/// per-connection task so a slow handshake never blocks the accept loop.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

// ============================================================================
// RpcConfig — unified server-side tunables (#197)
// ============================================================================

/// Unified RPC transport tunables for server builders.
///
/// All values default to the process-global constants; override via
/// `with_rpc_config()` on any server builder. This consolidates the
/// previously scattered `with_stream_limit` / `with_connection_limit` /
/// `with_read_timeout` builder methods behind one struct so a single
/// config source (e.g. a loaded `HyprConfig`) can tune all planes at once.
#[derive(Clone, Debug)]
pub struct RpcConfig {
    /// Max concurrent in-flight bidi streams (server-wide semaphore).
    pub stream_limit: usize,
    /// Max concurrent accepted connections per server.
    pub connection_limit: usize,
    /// Max wall-clock time to read a single request frame.
    pub request_read_timeout: Duration,
    /// Max time for a peer's QUIC/WebTransport handshake to complete.
    pub handshake_timeout: Duration,
    /// Grace period after writing a response for the peer to ack FIN.
    pub stopped_grace: Duration,
    /// Max wall-clock time for graceful drain on shutdown.
    pub drain_timeout: Duration,
}

impl Default for RpcConfig {
    fn default() -> Self {
        Self {
            stream_limit: DEFAULT_STREAM_LIMIT,
            connection_limit: DEFAULT_CONNECTION_LIMIT,
            request_read_timeout: REQUEST_READ_TIMEOUT,
            handshake_timeout: HANDSHAKE_TIMEOUT,
            stopped_grace: STOPPED_GRACE,
            drain_timeout: DRAIN_TIMEOUT,
        }
    }
}

// ============================================================================
// IrohRequestProcessor — pluggable request handling trait
// ============================================================================

/// Trait implemented by callers to wire actual request processing.
///
/// The processor receives the raw request bytes (a Cap'n Proto-encoded
/// `SignedEnvelope`) and returns the raw response bytes.
///
/// **Contract for `process`**:
/// - **`Ok(bytes)`** — `bytes` MUST be a valid Cap'n Proto-encoded
///   `ResponseEnvelope` and is written verbatim to the bidi stream.
///   Application-layer errors (verification failure, handler error, etc.)
///   MUST be returned this way as signed error envelopes — the server
///   forwards them unchanged.
/// - **`Err(_)`** — wire-level fatal: the processor cannot produce *any*
///   response. The server responds with its own signed error envelope
///   (`request_id = 0`) so the client sees a parseable error rather than a
///   Cap'n Proto parse failure.
///
/// Implementations MUST be `Send + Sync + 'static` because accept loops run on
/// a multi-threaded tokio runtime. For services that are `!Send` (e.g. those
/// holding `tch-rs` tensors), use [`super::iroh_rpc::LocalServiceBridge`].
///
/// (Name retained from the iroh-only era to avoid churn in callers.)
pub trait IrohRequestProcessor: Send + Sync + 'static {
    fn process(
        &self,
        request: Bytes,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>>;
}

/// Convenience: build an [`IrohRequestProcessor`] from a `Send + Sync`
/// async closure. Useful for tests.
pub fn from_fn<F, Fut>(f: F) -> impl IrohRequestProcessor
where
    F: Fn(Bytes) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Bytes>> + Send + 'static,
{
    struct FnProcessor<F>(F);
    impl<F, Fut> IrohRequestProcessor for FnProcessor<F>
    where
        F: Fn(Bytes) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Bytes>> + Send + 'static,
    {
        fn process(
            &self,
            request: Bytes,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Bytes>> + Send + '_>> {
            Box::pin((self.0)(request))
        }
    }
    FnProcessor(f)
}

// ============================================================================
// Bounded read helper
// ============================================================================

/// Read an entire stream into [`Bytes`], capping the total at `cap` bytes.
///
/// [`RecvStream::read_all`] is unbounded and a hostile peer could exhaust
/// memory with it. This loops over [`RecvStream::read`] into a growing buffer,
/// erroring out if the peer sends more than `cap` bytes before FIN.
pub async fn read_to_cap<R: RecvStream>(recv: &mut R, cap: usize) -> Result<Bytes> {
    // Chunk size for each read. Bounded so a single read can't over-allocate.
    const CHUNK: usize = 64 * 1024;
    let mut out: Vec<u8> = Vec::new();
    let mut scratch = vec![0u8; CHUNK];
    loop {
        match recv
            .read(&mut scratch)
            .await
            .map_err(|e| anyhow!("recv read: {e}"))?
        {
            None => break, // FIN
            Some(0) => continue,
            Some(n) => {
                if out.len() + n > cap {
                    bail!("frame exceeds {cap} byte cap");
                }
                out.extend_from_slice(&scratch[..n]);
            }
        }
    }
    Ok(Bytes::from(out))
}

// ============================================================================
// serve_rpc_connection — transport-generic server accept loop
// ============================================================================

/// Serve one accepted [`Session`] until the peer closes it or `shutdown` is
/// cancelled. Terminates `hyprstream-rpc/1`-style bidi streams, caps concurrent
/// streams via the shared `stream_limit`, and (in concert with the caller's
/// drain on that same `stream_limit`) drains in-flight requests on shutdown.
///
/// The caller owns the drain: it holds the same `Arc<Semaphore>` and, on
/// shutdown, cancels `shutdown` then `acquire_many(capacity)`s every permit so
/// detached `handle_stream` tasks finish before teardown. This fn only acquires
/// one per-stream permit per accepted bidi stream.
pub async fn serve_rpc_connection<S>(
    session: S,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    stream_limit: Arc<Semaphore>,
    read_timeout: Duration,
    shutdown: CancellationToken,
) -> Result<()>
where
    S: Session,
{
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                tracing::debug!("rpc-session: accept loop cancelled");
                return Ok(());
            }
            streams = session.accept_bi() => {
                let (send, recv) = match streams {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::debug!(error = ?e, "rpc-session: connection closed");
                        return Ok(());
                    }
                };

                // Acquire a permit to cap concurrent streams. If the semaphore
                // is closed (shutdown drained it), exit cleanly.
                let permit = match Arc::clone(&stream_limit).acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return Ok(()),
                };

                let processor = Arc::clone(&processor);
                let signing_key = signing_key.clone();
                // Hold a clone of the session in the per-stream task so the
                // underlying QUIC connection stays alive until the response is
                // fully written, even if the accept loop exits first on
                // shutdown (some `Session` impls — e.g. web-transport-quinn —
                // close the connection when the last `Session` handle drops).
                let session_keepalive = session.clone();
                tokio::spawn(async move {
                    let _permit = permit; // released on task end
                    let _keepalive = session_keepalive;
                    handle_stream::<S>(send, recv, processor, signing_key, read_timeout).await;
                });
            }
        }
    }
}

/// Handle one bidi stream end-to-end: read request, dispatch to processor,
/// write response (or signed error envelope on processor failure).
async fn handle_stream<S>(
    mut send: S::SendStream,
    mut recv: S::RecvStream,
    processor: Arc<dyn IrohRequestProcessor>,
    signing_key: SigningKey,
    read_timeout: Duration,
) where
    S: Session,
{
    // SECURITY (#159): bound the request read in wall-clock time. On timeout
    // we return, dropping `recv`/`send` (which resets the stream) and releasing
    // the semaphore permit held by the caller — so a stalled/trickling peer
    // cannot pin a server-wide permit indefinitely.
    let request = match tokio::time::timeout(read_timeout, read_to_cap(&mut recv, MAX_FRAME_BYTES)).await {
        Ok(Ok(buf)) => buf,
        Ok(Err(e)) => {
            tracing::warn!(error = ?e, "rpc-session: failed reading request");
            return;
        }
        Err(_) => {
            tracing::warn!(
                timeout = ?read_timeout,
                "rpc-session: request read timed out, abandoning stream"
            );
            return;
        }
    };

    let response = match processor.process(request).await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::warn!(error = ?e, "rpc-session: processor error, sending stub error envelope");
            build_error_envelope(&signing_key, &format!("processor error: {e}"))
        }
    };

    if let Err(e) = send.write_all(&response).await {
        tracing::warn!(error = ?e, "rpc-session: failed writing response");
        return;
    }
    if let Err(e) = send.finish() {
        tracing::warn!(error = ?e, "rpc-session: failed finishing send");
        return;
    }
    // Wait for the peer to ack the FIN, but cap it so a dead peer can't leak
    // this task forever.
    let _ = tokio::time::timeout(STOPPED_GRACE, send.closed()).await;
}

/// Build a signed `ResponseEnvelope` (request_id = 0) carrying `message` as
/// the payload. Used when the processor itself fails — guarantees the client
/// sees a parseable envelope rather than EOF.
pub fn build_error_envelope(signing_key: &SigningKey, message: &str) -> Bytes {
    use crate::ToCapnp;
    use capnp::{message::Builder, serialize};
    let envelope =
        crate::envelope::ResponseEnvelope::new_signed(0, message.as_bytes().to_vec(), signing_key);
    let mut msg = Builder::new_default();
    {
        let mut builder = msg.init_root::<crate::common_capnp::response_envelope::Builder>();
        envelope.write_to(&mut builder);
    }
    let mut bytes = Vec::new();
    if let Err(e) = serialize::write_message(&mut bytes, &msg) {
        // Last-ditch: if even this fails, return empty bytes — client will see
        // a Cap'n Proto parse error, which is no worse than the original
        // behaviour. This should be unreachable in practice.
        tracing::error!(error = ?e, "rpc-session: failed serializing error envelope");
        return Bytes::new();
    }
    Bytes::from(bytes)
}

// ============================================================================
// SessionRpcTransport — transport-generic client
// ============================================================================

/// Wraps a [`web_transport_trait::Session`] as a [`Transport`] for RPC plane
/// traffic. Generic over the session impl, so the same client logic rides iroh,
/// quinn, or any future `Session` backend.
///
/// Clone-cheap — `Session` is `Clone` (reference-counted internally by the
/// underlying QUIC connection).
#[derive(Clone)]
pub struct SessionRpcTransport<S: Session> {
    session: S,
}

impl<S: Session> SessionRpcTransport<S> {
    /// Build from an already-established session on the RPC ALPN.
    pub fn new(session: S) -> Self {
        Self { session }
    }

    /// Borrow the underlying session (e.g. for inspection).
    pub fn session(&self) -> &S {
        &self.session
    }
}

/// Stub stream type for [`Transport::Sub`]. Always pending — the RPC plane does
/// not carry SUB-style topic streams; those live on the streaming plane.
pub type RpcPendingStream = futures::stream::Pending<Result<Vec<Vec<u8>>>>;

/// Stub publish sink for [`Transport::Pub`]. Errors on any `send_frames` call;
/// the RPC plane is request-response only.
pub struct RpcPublishStub;

#[async_trait]
impl PublishSink for RpcPublishStub {
    async fn send_frames(&self, _frames: &[&[u8]]) -> Result<()> {
        bail!("RPC plane does not support PUB/PUSH — use moq-net on the streaming ALPN")
    }
}

#[async_trait]
impl<S: Session> Transport for SessionRpcTransport<S> {
    type Sub = RpcPendingStream;
    type Pub = RpcPublishStub;

    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        let timeout = timeout_ms
            .map(|ms| Duration::from_millis(ms.max(0) as u64))
            .unwrap_or(DEFAULT_CLIENT_TIMEOUT);
        let session = self.session.clone();
        let fut = async move {
            let (mut send, mut recv) = session
                .open_bi()
                .await
                .map_err(|e| anyhow!("session open_bi: {e}"))?;
            send.write_all(&payload)
                .await
                .map_err(|e| anyhow!("session write_all: {e}"))?;
            send.finish().map_err(|e| anyhow!("session finish: {e}"))?;
            let buf = read_to_cap(&mut recv, MAX_FRAME_BYTES).await?;
            Ok::<Vec<u8>, anyhow::Error>(buf.to_vec())
        };
        tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_| anyhow!("RPC timeout after {timeout:?}"))?
    }

    async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
        bail!("RPC plane does not support SUB — use moq-net on the streaming ALPN")
    }

    async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
        bail!("RPC plane does not support PUB — use moq-net on the streaming ALPN")
    }
}
