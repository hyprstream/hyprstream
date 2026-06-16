//! moq-lite streaming plane (M2a of epic #134).
//!
//! This is the moq-native replacement for the ZMQ PULL→XPUB queuing proxy in
//! [`crate::service::streaming`]. Instead of a forwarding proxy with a custom
//! per-topic queue + late-join rejoin buffer, in-process publishers append
//! directly to a shared `moq_net::Origin`, and external subscribers consume the
//! *same* origin over the `moql` ALPN via
//! [`crate::transport::iroh_moq::IrohMoqProtocolHandler`].
//!
//! # Topic → Track mapping
//!
//! The existing call-site contract is a topic string (64-hex DH-derived, or a
//! `notify-*` / control topic) plus opaque payload bytes. We map:
//!
//! ```text
//!   topic-string  ->  moq Broadcast path  {tenant}/{service}/{topic}/{instance}
//!   StreamBlock   ->  one moq Group (sequence = block index)
//!                       containing one Frame = capnp_bytes || mac[16]
//! ```
//!
//! For the in-process / direct path (M2a), the broadcast path is built from a
//! caller-supplied `{tenant}/{service}/.../{instance}` prefix joined with the
//! opaque topic. The topic itself stays opaque (and, for the relay'd path in
//! M2b, DH-derived/unguessable). A single track name (`STREAM_TRACK`) carries
//! the block sequence.
//!
//! # §7.5 chained-HMAC tokenstream
//!
//! The chained-HMAC envelope is preserved 1:1: each Group's Frame payload is
//! the same `[capnp StreamBlock || 16-byte truncated MAC]` that the ZMQ wire
//! format carried in frames 1+2 (frame 0, the topic, is now the Track/Broadcast
//! path and is therefore dropped from the payload). [`StreamHmacState`] /
//! [`StreamVerifier`] are reused unchanged, so the MAC chain — which binds each
//! block to its predecessor — is byte-identical to the ZMQ path.
//!
//! Late-join is now moq's job: a subscriber that joins late is served the
//! latest Group natively by the moq Track cache (respecting upstream Group
//! cache consts), so the custom `StreamResume` / rejoin-buffer code is gone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use bytes::Bytes;
use moq_net::{BroadcastProducer, Group, OriginConsumer, OriginProducer, Track, TrackProducer};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use crate::crypto::StreamHmacState;
use crate::streaming::{StreamContext, StreamPayloadData, StreamVerifier};

// ============================================================================
// Process-global moq streaming origin — set once at startup, read everywhere.
// Allows StreamChannel (publisher side) to use the same origin as StreamService
// (server side) without threading it through every service factory.
// ============================================================================

static GLOBAL_MOQ_ORIGIN: OnceLock<MoqStreamOrigin> = OnceLock::new();
/// The UDS socket path serving the moq plane (set by `serve_moq_uds_background`).
static GLOBAL_MOQ_UDS_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Register the process-global moq streaming origin.
///
/// Must be called once at startup (from the streams service factory) before any
/// `StreamChannel::publisher()` call. Returns `true` on first call, `false` if
/// already set (idempotent — a second set is silently ignored).
pub fn init_global_moq_origin(origin: MoqStreamOrigin) -> bool {
    GLOBAL_MOQ_ORIGIN.set(origin).is_ok()
}

/// Borrow the process-global moq streaming origin, if initialized.
///
/// `None` when moq is not yet wired (unit tests, ZMQ-only deployments).
pub fn global_moq_origin() -> Option<&'static MoqStreamOrigin> {
    GLOBAL_MOQ_ORIGIN.get()
}

/// Path of the UDS socket that serves the moq streaming plane to local clients.
///
/// Set by [`serve_moq_uds_background`] once the listener is ready.
/// `None` in ZMQ-only or unit-test deployments.
pub fn global_moq_uds_path() -> Option<&'static Path> {
    GLOBAL_MOQ_UDS_PATH.get().map(PathBuf::as_path)
}

/// Start a UDS moq server in the background, serving the moq origin's consumer
/// to local cross-process subscribers (e.g. `hyprstream tui attach`).
///
/// Each accepted connection that presents [`crate::transport::uds_session::PLANE_MOQ`]
/// gets a dedicated moq server session via `moq_net::Server::with_publish`.
/// The origin consumer is cloned per session so every subscriber sees the
/// same live broadcast tree.
///
/// Idempotent: a second call is a no-op (the first path wins).
pub fn serve_moq_uds_background(origin: MoqStreamOrigin, path: PathBuf) {
    use crate::transport::uds_session::{accept_uds, PLANE_MOQ};
    use moq_net::Server as MoqServer;

    // Remove stale socket from a previous run (best-effort).
    let _ = std::fs::remove_file(&path);

    // Bind synchronously so the socket exists before we advertise the path.
    // Any caller reading global_moq_uds_path() is guaranteed the socket is ready.
    let listener = match std::os::unix::net::UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(path = %path.display(), "moq UDS bind failed: {e}");
            return;
        }
    };

    // 0o600: owner read/write only; SO_PEERCRED enforces uid match on Linux (see uds_server.rs).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }

    // Convert to async and publish the path only after the socket is bound.
    let listener = match tokio::net::UnixListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("moq UDS listener conversion failed: {e}");
            return;
        }
    };

    if GLOBAL_MOQ_UDS_PATH.set(path.clone()).is_err() {
        return; // already started (concurrent call)
    }

    tracing::info!(path = %path.display(), "moq UDS listener ready");

    tokio::spawn(async move {
        loop {
            let stream = match listener.accept().await {
                Ok((s, _)) => s,
                Err(e) => {
                    tracing::warn!("moq UDS accept error: {e}");
                    continue;
                }
            };
            let consumer = origin.consumer().clone();
            tokio::spawn(async move {
                let (plane, session) = match accept_uds(stream).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::debug!("moq UDS handshake error: {e}");
                        return;
                    }
                };
                if plane != PLANE_MOQ {
                    tracing::debug!("moq UDS: unexpected plane 0x{plane:02x} — dropping");
                    return;
                }
                if let Err(e) = MoqServer::new().with_publish(consumer).accept(session).await {
                    tracing::debug!("moq UDS session ended: {e}");
                }
            });
        }
    });
}

/// Timeout waiting for the moq origin to announce a broadcast at startup.
pub const BROADCAST_ANNOUNCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Timeout between consecutive moq Groups on a subscribed track.
/// A timeout here signals the publisher is gone; the subscriber breaks out.
pub const GROUP_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Timeout reading a single Frame from an already-opened Group.
pub const FRAME_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The single track name that carries StreamBlock groups for a broadcast.
pub const STREAM_TRACK: &str = "stream";

/// Default `{tenant}/{service}` broadcast-path prefix for the in-process plane.
///
/// The instance segment is appended per-publisher (the opaque topic). Callers
/// that want tenant isolation pass their own prefix to
/// [`MoqStreamOrigin::with_prefix`].
pub const DEFAULT_PREFIX: &str = "local/streams";

/// Shared moq origin for the streaming plane, plus the in-process publish gate.
///
/// Holds the `OriginProducer` (what in-process publishers append into) and the
/// `OriginConsumer` (handed to `moq_net::Server` to serve external subscribers).
/// This is the moq replacement for [`crate::service::StreamService`]'s ZMQ
/// proxy: there is no forwarding loop — producers and consumers share one tree.
#[derive(Clone)]
pub struct MoqStreamOrigin {
    inner: Arc<OriginInner>,
}

/// Builder for [`MoqStreamOrigin`] — set the prefix and publish gate before the
/// shared `Arc` is constructed (avoids clone-on-write of the origin tree).
pub struct MoqStreamOriginBuilder {
    producer: OriginProducer,
    consumer: OriginConsumer,
    prefix: String,
    authorize_signer: Option<Arc<dyn Fn(&[u8; 32]) -> bool + Send + Sync>>,
}

impl MoqStreamOriginBuilder {
    /// Set the `{tenant}/{service}` broadcast-path prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Install the in-process publish gate (mirrors
    /// `StreamService::with_authorize_signer`).
    pub fn with_authorize_signer(
        mut self,
        f: impl Fn(&[u8; 32]) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.authorize_signer = Some(Arc::new(f));
        self
    }

    /// Finish building.
    pub fn build(self) -> MoqStreamOrigin {
        MoqStreamOrigin {
            inner: Arc::new(OriginInner {
                producer: self.producer,
                consumer: self.consumer,
                prefix: self.prefix,
                authorize_signer: self.authorize_signer,
                broadcasts: Mutex::new(HashMap::new()),
            }),
        }
    }
}

struct OriginInner {
    producer: OriginProducer,
    consumer: OriginConsumer,
    /// `{tenant}/{service}` prefix for broadcast paths.
    prefix: String,
    /// Optional in-process publish gate (mirrors `StreamService::authorize_signer`).
    ///
    /// When set, [`MoqStreamOrigin::authorize_signer`] must return `true` before
    /// a publisher is created for a topic. `None` accepts any caller
    /// (testing/bootstrap), matching the ZMQ default.
    authorize_signer: Option<Arc<dyn Fn(&[u8; 32]) -> bool + Send + Sync>>,
    /// Keep broadcast producers alive while their publishers are active.
    ///
    /// Keyed by broadcast path (replace semantics) so a re-announced topic
    /// drops the old `BroadcastProducer` (unannouncing it) rather than
    /// accumulating unboundedly (#164).
    broadcasts: Mutex<HashMap<String, BroadcastProducer>>,
}

impl MoqStreamOrigin {
    /// Begin building from an existing producer/consumer pair (e.g. the one
    /// held by [`crate::transport::iroh_moq::IrohMoqProtocolHandler`]).
    pub fn builder(producer: OriginProducer, consumer: OriginConsumer) -> MoqStreamOriginBuilder {
        MoqStreamOriginBuilder {
            producer,
            consumer,
            prefix: DEFAULT_PREFIX.to_owned(),
            authorize_signer: None,
        }
    }

    /// Build directly from a producer/consumer pair with defaults.
    pub fn from_pair(producer: OriginProducer, consumer: OriginConsumer) -> Self {
        Self::builder(producer, consumer).build()
    }

    /// Begin building a standalone origin with a fresh random id.
    ///
    /// Used when no shared substrate origin is available yet (M2a bootstrap).
    /// Callers that have the substrate's `IrohMoqProtocolHandler` should use
    /// [`Self::builder`] with its `origin_producer()` / `origin_consumer()`
    /// instead, so external subscribers see the same tree.
    pub fn standalone() -> MoqStreamOriginBuilder {
        let producer = moq_net::Origin::random().produce();
        let consumer = producer.consume();
        Self::builder(producer, consumer)
    }

    /// Borrow the consumer (hand this to `moq_net::Server::with_publish`).
    pub fn consumer(&self) -> &OriginConsumer {
        &self.inner.consumer
    }

    /// Borrow the producer.
    pub fn producer(&self) -> &OriginProducer {
        &self.inner.producer
    }

    /// Check the in-process publish gate for a signer pubkey.
    ///
    /// Returns `true` when no gate is installed (bootstrap/testing).
    pub fn authorize_signer(&self, signer: &[u8; 32]) -> bool {
        match &self.inner.authorize_signer {
            Some(f) => f(signer),
            None => true,
        }
    }

    /// Build the broadcast path for an opaque topic: `{prefix}/{topic}`.
    pub fn broadcast_path(&self, topic: &str) -> String {
        format!("{}/{}", self.inner.prefix, topic)
    }

    /// Create an in-process publisher for `ctx`'s topic.
    ///
    /// Creates (or replaces) the broadcast at `{prefix}/{topic}` with a single
    /// [`STREAM_TRACK`] track, and returns a [`MoqStreamPublisher`] whose
    /// chained-HMAC state is seeded from `ctx.mac_key()` / `ctx.topic()` — i.e.
    /// byte-identical to the ZMQ `StreamBuilder`.
    pub fn publisher(&self, ctx: &StreamContext) -> Result<MoqStreamPublisher> {
        let path = self.broadcast_path(ctx.topic());
        let mut broadcast = self
            .inner
            .producer
            .create_broadcast(path.as_str())
            .ok_or_else(|| anyhow!("create_broadcast denied for {path}"))?;
        let track = broadcast.create_track(Track::new(STREAM_TRACK))?;

        // Retain the broadcast producer so it stays announced for the
        // publisher's lifetime (dropping it would unannounce the broadcast).
        // Replace-semantics: inserting the same path twice drops the old
        // BroadcastProducer rather than accumulating indefinitely (#164).
        self.inner.broadcasts.lock().insert(path, broadcast);

        Ok(MoqStreamPublisher {
            hmac_state: StreamHmacState::new(*ctx.mac_key(), ctx.topic().to_owned()),
            track,
            next_group: 0,
            cancel_token: ctx.cancel_token().clone(),
            terminated: false,
            topic: ctx.topic().to_owned(),
        })
    }
}

/// In-process moq publisher with the §7.5 chained-HMAC tokenstream.
///
/// API mirrors [`crate::streaming::StreamPublisher`] (`publish_data`,
/// `publish_error`, `complete`, ...) but appends to a moq Track instead of a
/// ZMQ PUSH socket. Each call produces one StreamBlock = one moq Group whose
/// single Frame payload is `capnp_bytes || mac[16]`.
/// NOTE (M2a vs M2b): the StreamBlock *encoding* + HMAC chain are byte-identical
/// to the ZMQ path (shared `encode_stream_block`), but the *batching policy*
/// differs — this emits one payload per block/Group, whereas the ZMQ
/// `StreamBuilder` adaptively batches multiple payloads per block. The MAC
/// chain is valid either way (the verifier is batch-agnostic). Port
/// `BatchingConfig` in M2b for granularity parity.
pub struct MoqStreamPublisher {
    hmac_state: StreamHmacState,
    track: TrackProducer,
    next_group: u64,
    cancel_token: CancellationToken,
    terminated: bool,
    topic: String,
}

impl MoqStreamPublisher {
    /// Publish one binary payload as a StreamBlock group.
    pub async fn publish_data(&mut self, data: &[u8]) -> Result<()> {
        if self.cancel_token.is_cancelled() {
            anyhow::bail!("stream cancelled");
        }
        self.write_block(&[StreamPayloadData::Data(data.to_vec())])
    }

    /// Publish an error payload (terminal).
    pub async fn publish_error(&mut self, message: &str) -> Result<()> {
        self.terminated = true;
        self.write_block(&[StreamPayloadData::Error(message.to_owned())])
    }

    /// Complete the stream with metadata (terminal).
    pub async fn complete(mut self, metadata: &[u8]) -> Result<()> {
        self.complete_ref(metadata).await
    }

    /// Complete the stream without consuming `self`.
    pub async fn complete_ref(&mut self, metadata: &[u8]) -> Result<()> {
        self.terminated = true;
        self.write_block(&[StreamPayloadData::Complete(metadata.to_vec())])
    }

    /// The opaque topic this publisher serves.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Whether a terminal frame (Error/Complete) was sent.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Whether the stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Serialize payloads into a StreamBlock, chain the MAC, and append as a
    /// moq Group (one Frame = `capnp || mac`).
    fn write_block(&mut self, payloads: &[StreamPayloadData]) -> Result<()> {
        // The StreamBlock sequenceNumber (#219) IS the moq Group id — unified so the
        // in-block sequenceNumber the consumer authenticates matches the transport Group.
        // epoch is 0 until the #223 key-epoch lifecycle lands.
        let sequence_number = self.next_group;
        self.next_group += 1;
        let capnp_bytes = crate::streaming::encode_stream_block(
            self.hmac_state.prev_mac_bytes(),
            sequence_number,
            0,
            payloads,
        )?;
        let mac = self.hmac_state.compute_next(&capnp_bytes);

        let mut frame = Vec::with_capacity(capnp_bytes.len() + 16);
        frame.extend_from_slice(&capnp_bytes);
        frame.extend_from_slice(&mac);

        let mut group = self.track.create_group(Group::from(sequence_number))?;
        group.write_frame(Bytes::from(frame))?;
        group.finish()?;
        Ok(())
    }
}

// ============================================================================
// AnyStreamPublisher — moq-lite publish API
// ============================================================================

/// Publisher for the moq-lite streaming plane.
///
/// Type alias for [`MoqStreamPublisher`]; the ZMQ variant was removed in the
/// N4 hard cutover (#138/#213). Retained as an alias so existing call sites
/// compile unchanged.
pub type AnyStreamPublisher = MoqStreamPublisher;

impl MoqStreamPublisher {
    /// Publish binary data with a rate hint (ignored — each call maps 1:1 to one moq Group).
    pub async fn publish_data_with_rate(&mut self, data: &[u8], rate: f32) -> Result<()> {
        let _ = rate;
        self.publish_data(data).await
    }

    /// Publish a progress update (`stage:current:total`).
    pub async fn publish_progress(&mut self, stage: &str, current: usize, total: usize) -> Result<()> {
        let data = format!("{}:{}:{}", stage, current, total);
        self.publish_data(data.as_bytes()).await
    }

    /// Flush — no-op (each block is published immediately).
    pub async fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Non-blocking publish — always succeeds immediately (no HWM concept).
    /// The `rate` hint is ignored.
    pub async fn try_publish_data(&mut self, data: &[u8], rate: f32) -> Result<bool> {
        let _ = rate;
        self.publish_data(data).await.map(|_| true)
    }
}

/// moq stream consumer handle for MCP tool calls and other async consumers.
///
/// Connects to the moq UDS plane, subscribes to the broadcast, verifies the
/// chained-HMAC frames, and delivers [`crate::streaming::StreamPayload`]s via
/// an internal channel. Mirrors the `StreamHandle` interface (same `recv_next()`,
/// `stream_id()`, `cancel_token()`, and `futures::Stream` impl) so callers need
/// no code change beyond the constructor.
pub struct MoqStreamHandle {
    rx: tokio::sync::mpsc::Receiver<anyhow::Result<crate::streaming::StreamPayload>>,
    broadcast_path: String,
    cancel: tokio_util::sync::CancellationToken,
}

impl MoqStreamHandle {
    /// Construct a handle and immediately spawn the background receive task.
    ///
    /// The task connects to the UDS moq server, subscribes to `broadcast_path`,
    /// verifies frames with the chained HMAC (derived from `mac_key` + `topic`),
    /// and forwards payloads to the channel. Errors and end-of-stream close the
    /// channel so `recv_next()` returns `Err` or `Ok(None)` respectively.
    pub fn new(
        uds_path: String,
        broadcast_path: String,
        mac_key: [u8; 32],
        topic: String,
    ) -> Self {
        let cancel = tokio_util::sync::CancellationToken::new();
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<crate::streaming::StreamPayload>>(64);
        tokio::spawn(moq_stream_handle_task(uds_path, broadcast_path.clone(), mac_key, topic, tx, cancel.clone()));
        Self { rx, broadcast_path, cancel }
    }

    /// Receive the next stream payload.
    ///
    /// Returns `Ok(None)` when the stream ends cleanly, `Err` on a stream error.
    pub async fn recv_next(&mut self) -> anyhow::Result<Option<crate::streaming::StreamPayload>> {
        match self.rx.recv().await {
            Some(Ok(p)) => Ok(Some(p)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }

    /// Returns the moq broadcast path used as stream identifier.
    pub fn stream_id(&self) -> &str {
        &self.broadcast_path
    }

    /// Returns a cancellation token that aborts the background receive task.
    pub fn cancel_token(&self) -> &tokio_util::sync::CancellationToken {
        &self.cancel
    }

    /// Cancel the background receive task immediately.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl futures::Stream for MoqStreamHandle {
    type Item = anyhow::Result<crate::streaming::StreamPayload>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

// TODO: add reconnect backoff on UDS disconnect (ZMQ auto-reconnect parity)
async fn moq_stream_handle_task(
    uds_path: String,
    broadcast_path: String,
    mac_key: [u8; 32],
    topic: String,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<crate::streaming::StreamPayload>>,
    cancel: tokio_util::sync::CancellationToken,
) {
    use crate::streaming::StreamVerifier;
    use crate::transport::uds_session::{connect_uds, PLANE_MOQ};
    use moq_net::{Client as MoqClient, Origin, Track};

    let session = match connect_uds(&uds_path, PLANE_MOQ).await {
        Ok(s) => s,
        Err(e) => { let _ = tx.send(Err(anyhow!("moq UDS connect {uds_path}: {e}"))).await; return; }
    };
    let client_origin = Origin::random().produce();
    let client_consumer = client_origin.consume();
    let moq_client = MoqClient::new().with_consume(client_origin);
    let _session = match moq_client.connect(session).await {
        Ok(s) => s,
        Err(e) => { let _ = tx.send(Err(anyhow!("moq handshake: {e}"))).await; return; }
    };
    let bc = match tokio::time::timeout(
        BROADCAST_ANNOUNCE_TIMEOUT,
        client_consumer.announced_broadcast(&broadcast_path),
    ).await {
        Ok(Some(bc)) => bc,
        Ok(None) => {
            let _ = tx.send(Err(anyhow!("broadcast {broadcast_path} not announced"))).await;
            return;
        }
        Err(_) => {
            let _ = tx.send(Err(anyhow!("timeout waiting for broadcast {broadcast_path}"))).await;
            return;
        }
    };
    let mut track = match bc.subscribe_track(&Track::new(STREAM_TRACK)) {
        Ok(t) => t,
        Err(e) => { let _ = tx.send(Err(anyhow!("subscribe_track: {e}"))).await; return; }
    };
    let mut verifier = StreamVerifier::new(mac_key, topic.clone());
    loop {
        if cancel.is_cancelled() {
            break;
        }
        let mut group = match tokio::time::timeout(GROUP_IDLE_TIMEOUT, track.next_group()).await {
            Ok(Ok(Some(g))) => g,
            Ok(Ok(None)) => break, // track ended cleanly
            Err(_elapsed) => {
                let _ = tx.send(Err(anyhow!(
                    "stream idle: no group for {}s",
                    GROUP_IDLE_TIMEOUT.as_secs()
                ))).await;
                break;
            }
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("moq next_group: {e}"))).await;
                break;
            }
        };
        let frame: bytes::Bytes = match tokio::time::timeout(FRAME_READ_TIMEOUT, group.read_frame()).await {
            Ok(Ok(Some(f))) => f,
            Ok(Ok(None)) => break, // group ended without a frame
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("frame read error: {e}"))).await;
                break;
            }
            Err(_elapsed) => {
                let _ = tx.send(Err(anyhow!(
                    "frame read timeout after {}s",
                    FRAME_READ_TIMEOUT.as_secs()
                ))).await;
                break;
            }
        };
        match verify_moq_frame(&mut verifier, &topic, &frame) {
            Ok(payloads) => {
                for p in payloads {
                    let is_terminal = matches!(
                        p,
                        crate::streaming::StreamPayload::Complete(_)
                            | crate::streaming::StreamPayload::Error(_)
                    );
                    if tx.send(Ok(p)).await.is_err() {
                        return;
                    }
                    if is_terminal {
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }
}

/// Consumer-side helper: split a moq Frame payload back into the ZMQ-style
/// `[topic, capnp, mac]` frames expected by [`StreamVerifier::verify`], then
/// verify and parse it.
///
/// `topic` is the opaque topic (not the broadcast path) and must match the
/// verifier's topic, exactly as the ZMQ frame-0 contract required.
pub fn verify_moq_frame(
    verifier: &mut StreamVerifier,
    topic: &str,
    frame: &[u8],
) -> Result<Vec<crate::streaming::StreamPayload>> {
    if frame.len() < 16 {
        anyhow::bail!("moq frame too short: {} bytes", frame.len());
    }
    let split = frame.len() - 16;
    let capnp = frame[..split].to_vec();
    let mac = frame[split..].to_vec();
    verifier.verify(&[topic.as_bytes().to_vec(), capnp, mac])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::StreamPayload;
    use moq_net::Origin;

    fn origin() -> MoqStreamOrigin {
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        MoqStreamOrigin::from_pair(producer, consumer)
    }

    /// In-process publish → in-process consume over the *same* origin (the data
    /// an external moq subscriber sees on the wire), verifying the chained-HMAC.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moq_stream_round_trip() -> Result<()> {
        let origin = origin();
        let (_client_secret, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();

        let mut pub_ = origin.publisher(&ctx)?;
        pub_.publish_data(b"hello").await?;
        pub_.publish_data(b"world").await?;
        pub_.complete_ref(b"{}").await?;

        // Consume from the shared origin (same bytes a wire subscriber reads).
        let path = origin.broadcast_path(&topic);
        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut track = bc.subscribe_track(&Track::new(STREAM_TRACK))?;

        let mut verifier = StreamVerifier::new(mac_key, topic.clone());
        let mut got: Vec<StreamPayload> = Vec::new();
        for _ in 0..3 {
            let mut group = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                track.next_group(),
            )
            .await??
            .ok_or_else(|| anyhow!("next_group None"))?;
            let frame = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                group.read_frame(),
            )
            .await??
            .ok_or_else(|| anyhow!("read_frame None"))?;
            got.extend(verify_moq_frame(&mut verifier, &topic, &frame)?);
        }

        assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"hello"));
        assert!(matches!(&got[1], StreamPayload::Data(d) if d == b"world"));
        assert!(matches!(&got[2], StreamPayload::Complete(_)));
        Ok(())
    }

    /// `AnyStreamPublisher` round-trip: publish via the type alias and verify
    /// the same bytes arrive on the moq consumer side.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn any_stream_publisher_moq_round_trip() -> Result<()> {
        let origin = origin();
        let (_client_secret, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();

        let mut any_pub: AnyStreamPublisher = origin.publisher(&ctx)?;
        any_pub.publish_data(b"ping").await?;
        any_pub.complete_ref(b"{}").await?;

        // Consume and verify
        let path = origin.broadcast_path(&topic);
        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut track = bc.subscribe_track(&Track::new(STREAM_TRACK))?;
        let mut verifier = StreamVerifier::new(mac_key, topic.clone());
        let mut got: Vec<StreamPayload> = Vec::new();
        for _ in 0..2 {
            let mut group = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                track.next_group(),
            )
            .await??
            .ok_or_else(|| anyhow!("next_group None"))?;
            let frame = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                group.read_frame(),
            )
            .await??
            .ok_or_else(|| anyhow!("read_frame None"))?;
            got.extend(verify_moq_frame(&mut verifier, &topic, &frame)?);
        }
        assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"ping"));
        assert!(matches!(&got[1], StreamPayload::Complete(_)));
        Ok(())
    }

    #[test]
    fn authorize_signer_gate() {
        let producer = Origin::random().produce();
        let consumer = producer.consume();
        let gated = MoqStreamOrigin::builder(producer, consumer)
            .with_authorize_signer(|pk| pk[0] == 1)
            .build();
        assert!(gated.authorize_signer(&[1u8; 32]));
        assert!(!gated.authorize_signer(&[2u8; 32]));
        // No gate -> accept all.
        assert!(origin().authorize_signer(&[9u8; 32]));
    }
}
