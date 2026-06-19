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

// ============================================================================
// Process-global producer reach (#274) — the network-routable way external
// subscribers reach this node's moq plane. Set once when the QUIC /
// web_transport_quinn server binds; read by every StreamInfo producer site so
// the `reach` field is built from ONE source (no per-site assembly, no drift),
// the same Rust value `root_did_document()` uses for its QuicTransport entry.
// ============================================================================

/// The node's network-routable moq reach: the bound QUIC address, its TLS
/// server name, and the leaf-cert SHA-256 pins. `None` until the daemon binds
/// its `web_transport_quinn` server (UDS-only / unit-test deployments).
static GLOBAL_PRODUCER_REACH: OnceLock<NodeStreamReach> = OnceLock::new();

/// The node's own moq reach parameters — the single source for the `reach`
/// list every StreamInfo producer publishes.
#[derive(Clone, Debug)]
pub struct NodeStreamReach {
    /// Bound socket address external subscribers dial (`/moq` over WebTransport).
    pub addr: std::net::SocketAddr,
    /// TLS SNI / WebPKI validation name advertised for the endpoint.
    pub server_name: String,
    /// Acceptable leaf-cert SHA-256 pins (self-signed mesh; rotation = multiple).
    pub cert_hashes: Vec<[u8; 32]>,
}

/// Register the node's network-routable moq reach (idempotent — first wins).
///
/// Called once when the daemon binds its `web_transport_quinn` server, with the
/// same `(addr, server_name, cert_hash)` the RPC endpoint registers and the
/// DID-doc `#quic` entry advertises.
pub fn init_global_producer_reach(reach: NodeStreamReach) -> bool {
    GLOBAL_PRODUCER_REACH.set(reach).is_ok()
}

/// Borrow the node's registered network reach, if the QUIC server is bound.
pub fn global_producer_reach() -> Option<&'static NodeStreamReach> {
    GLOBAL_PRODUCER_REACH.get()
}

/// Build the `StreamInfo.reach` list for a producer on this node (#274).
///
/// Centralised so every producer site emits an identical reach, sourced from the
/// one [`NodeStreamReach`] the QUIC bind registered. Returns an empty list when
/// the node has no networked reach (UDS-only); co-located clients fall back to
/// the same-host UDS fast path and ignore `reach` entirely.
pub fn producer_reach() -> Vec<crate::stream_info::Destination> {
    use crate::stream_info::{QuicReach, Destination, Role, TransportConfig};
    match global_producer_reach() {
        Some(r) => vec![Destination {
            role: Role::Direct,
            transport: TransportConfig::Quic(QuicReach {
                addr: r.addr.to_string(),
                server_name: r.server_name.clone(),
                cert_hashes: r.cert_hashes.iter().map(|h| h.to_vec()).collect(),
            }),
        }],
        None => Vec::new(),
    }
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

    // tokio requires the listener be non-blocking before adoption; otherwise
    // `from_std` panics ("Registering a blocking socket with the tokio runtime
    // is unsupported"). Mirrors the event-bus plane (moq_event.rs).
    if let Err(e) = listener.set_nonblocking(true) {
        tracing::error!("moq UDS set_nonblocking failed: {e}");
        return;
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
        self.publisher_with_provenance(ctx, None)
    }

    /// Create an in-process publisher that additionally signs each StreamBlock
    /// with the node's per-host hybrid COSE identity (#321 provenance / C-PROV).
    ///
    /// `provenance = Some(signer)` attaches a `StreamBlock.provenance` signature
    /// so consumers can attribute each block to the producing host (threat T3);
    /// `None` keeps the legacy chained-HMAC-only block.
    pub fn publisher_with_provenance(
        &self,
        ctx: &StreamContext,
        provenance: Option<crate::stream_provenance::ProvenanceSigner>,
    ) -> Result<MoqStreamPublisher> {
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
            // #321: AEAD enc_key — `Some` only on the DH (mesh) path. `None` on the
            // keyless `StreamContext::new` path (NotificationService topics, whose
            // payloads are already E2E-encrypted), where transport AEAD is skipped.
            enc_key: ctx.enc_key().copied(),
            provenance,
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
    /// Transport-level AEAD key (#321). `Some` ⇒ each Data/Complete payload is
    /// sealed with AES-256-GCM into a `Tagged` payload before the HMAC chain runs;
    /// `None` ⇒ cleartext (keyless notification path).
    enc_key: Option<[u8; 32]>,
    /// Per-host provenance signer (#321). `Some` ⇒ each StreamBlock carries a
    /// hybrid COSE signature over its canonical signed region.
    provenance: Option<crate::stream_provenance::ProvenanceSigner>,
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
        let epoch = 0u64;

        // #321: on the mesh/DH path (`enc_key = Some`), seal each Data/Complete
        // payload with AES-256-GCM into a `Tagged` payload BEFORE the HMAC chain
        // runs (so the chain authenticates the ciphertext — no double-encryption,
        // ordering/anti-replay unchanged). Error frames stay cleartext (operational
        // status, terminal). The AEAD AAD/key-commitment are bound to the block's
        // `epoch` (and topic), so a rekey can't replay a block across epochs.
        let sealed: Vec<StreamPayloadData>;
        let payloads: &[StreamPayloadData] = match self.enc_key {
            Some(ref enc_key) => {
                sealed = payloads
                    .iter()
                    .map(|p| seal_payload(enc_key, &self.topic, epoch, p))
                    .collect::<Result<Vec<_>>>()?;
                &sealed
            }
            None => payloads,
        };

        // Canonical signed region (#321): the StreamBlock with provenance EMPTY.
        // This is what the provenance signature covers and what the consumer
        // reconstructs; it also equals the legacy block when provenance is off.
        let signed_region = crate::streaming::encode_stream_block(
            self.hmac_state.prev_mac_bytes(),
            sequence_number,
            epoch,
            payloads,
        )?;

        // #321 provenance: sign the signed region with the host's hybrid identity,
        // then emit the block WITH the provenance field. The HMAC below covers the
        // full wire bytes (incl. provenance); the sig covers the signed region, so
        // verification is a layer on top of HMAC. No provenance ⇒ wire == signed_region.
        let capnp_bytes = match self.provenance {
            Some(ref signer) => {
                let (signer_kid, sig) = signer.sign(&signed_region)?;
                crate::streaming::encode_stream_block_with_provenance(
                    self.hmac_state.prev_mac_bytes(),
                    sequence_number,
                    epoch,
                    payloads,
                    Some((&signer_kid, &sig)),
                )?
            }
            None => signed_region,
        };
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
        enc_key: [u8; 32],
        topic: String,
    ) -> Self {
        let cancel = tokio_util::sync::CancellationToken::new();
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<crate::streaming::StreamPayload>>(64);
        tokio::spawn(moq_stream_handle_task(uds_path, broadcast_path.clone(), mac_key, enc_key, topic, tx, cancel.clone()));
        Self { rx, broadcast_path, cancel }
    }

    /// Construct a handle that subscribes over the **network** (#274).
    ///
    /// Resolves the signed `StreamInfo`'s `reach` list and dials the first
    /// dialable network reach via [`crate::dial::dial_stream`] over
    /// `web_transport_quinn` — exactly as the working CLI `quick infer` consumer
    /// does. After connecting, it subscribes to `broadcast_path` and verifies the
    /// chained-HMAC frames exactly as [`Self::new`] does. Errors/end-of-stream
    /// close the channel.
    ///
    /// ## Transport selection (#275 TUI streaming fix)
    ///
    /// The producer's `reach` is the source of truth for where the stream lives.
    /// A stream is a networked address; the subscriber dials the **producer's**
    /// reach. Loopback works same-host (the bound QUIC `/moq` endpoint), proven by
    /// the CLI, so the networked reach is used uniformly whenever the StreamInfo
    /// carries one.
    ///
    /// The local moq UDS plane ([`global_moq_uds_path`]) is used **only as a
    /// fallback** when the StreamInfo carries no dialable reach (UDS-only /
    /// unit-test deployments). It must NOT be preferred when a reach is present:
    /// the local UDS is *this process's own* moq plane, which only carries the
    /// producer's broadcast if the producer is co-located in this very process.
    /// A consumer process that runs its own moq plane for unrelated streams (e.g.
    /// the TUI daemon serving its PTY/shell stdout stream) would otherwise connect
    /// to its own empty plane and time out waiting for the model service's
    /// broadcast that was published on a *different* process's plane (#275).
    pub fn networked(
        reach: Vec<crate::stream_info::Destination>,
        broadcast_path: String,
        mac_key: [u8; 32],
        enc_key: [u8; 32],
        topic: String,
    ) -> Self {
        let cancel = tokio_util::sync::CancellationToken::new();
        let (tx, rx) =
            tokio::sync::mpsc::channel::<anyhow::Result<crate::streaming::StreamPayload>>(64);
        // Prefer the producer's networked reach (the StreamInfo source of truth):
        // dial the producer directly, mirroring the CLI. Only fall back to the
        // local moq UDS plane when the StreamInfo carries no dialable reach —
        // never the other way around (see method docs; #275).
        let has_dialable_reach = reach.iter().any(|d| reach_to_transport_config(d).is_some());
        if has_dialable_reach {
            tokio::spawn(moq_stream_handle_task_networked(
                reach,
                broadcast_path.clone(),
                mac_key,
                enc_key,
                topic,
                tx,
                cancel.clone(),
            ));
        } else if let Some(uds) = global_moq_uds_path() {
            let uds_path = uds.to_string_lossy().into_owned();
            tokio::spawn(moq_stream_handle_task(
                uds_path,
                broadcast_path.clone(),
                mac_key,
                enc_key,
                topic,
                tx,
                cancel.clone(),
            ));
        } else {
            // No dialable reach and no local UDS plane: surface a clear error
            // rather than spawning a task that cannot connect.
            tokio::spawn(async move {
                let _ = tx
                    .send(Err(anyhow!(
                        "no dialable reach in StreamInfo and no local moq UDS plane — \
                         cannot subscribe to broadcast"
                    )))
                    .await;
            });
        }
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
#[allow(clippy::too_many_arguments)]
async fn moq_stream_handle_task(
    uds_path: String,
    broadcast_path: String,
    mac_key: [u8; 32],
    enc_key: [u8; 32],
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
    let track = match bc.subscribe_track(&Track::new(STREAM_TRACK)) {
        Ok(t) => t,
        Err(e) => { let _ = tx.send(Err(anyhow!("subscribe_track: {e}"))).await; return; }
    };
    // #321: AEAD ON for this DH-keyed mesh stream — open sealed Tagged blocks.
    let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
    // #145: read groups by EXACT sequence (get_group), not arrival-order next_group.
    // Each Group is served on its own QUIC uni-stream, so Groups can arrive out of
    // order; next_group's monotonic cursor returns the first Group with sequence >=
    // its cursor in *arrival* order, so a lower-seq Group that arrives after a higher
    // one (e.g. the small terminal / always-retained max_sequence Group) is skipped —
    // fatally breaking the ordered, gap-fatal chained HMAC. get_group(seq) waits for
    // each exact Group (even out-of-order), guaranteeing in-order, gap-free delivery.
    let mut expected_seq = 0u64;
    loop {
        if cancel.is_cancelled() {
            break;
        }
        let mut group = match tokio::time::timeout(GROUP_IDLE_TIMEOUT, track.get_group(expected_seq)).await {
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
        expected_seq += 1;
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

/// Convert a wire-published [`crate::stream_info::Destination`] into the local
/// [`crate::transport::TransportConfig`] [`crate::dial::dial_stream`] dials (#274).
///
/// Only the network-routable transports map; same-host endpoints are never
/// carried in `reach` (co-located clients use the UDS fast path).
fn reach_to_transport_config(
    reach: &crate::stream_info::Destination,
) -> Option<crate::transport::TransportConfig> {
    use crate::stream_info::TransportConfig as ReachTransport;
    use crate::transport::{QuicServerAuth, TransportConfig};
    match &reach.transport {
        ReachTransport::Quic(q) => {
            let addr: std::net::SocketAddr = q.addr.parse().ok()?;
            // Fixed-size cert pins (SHA-256 = 32 bytes); skip any malformed entry.
            let hashes: Vec<[u8; 32]> = q
                .cert_hashes
                .iter()
                .filter_map(|h| <[u8; 32]>::try_from(h.as_slice()).ok())
                .collect();
            let auth = if hashes.is_empty() {
                QuicServerAuth::web_pki()
            } else {
                // Pinned self-signed mesh (matches the DID-doc #quic entry).
                QuicServerAuth::pinned(hashes).ok()?
            };
            Some(TransportConfig::quic_with_auth(addr, q.server_name.clone(), auth))
        }
        // #282 SEAM: `dial_stream` now dials an iroh `moql` reach
        // (`EndpointType::Iroh { node_id, .. }`), but the **wire** reach enum's
        // `Iroh` variant is a unit variant carrying no `node_id`/relay payload
        // (it is code-generated from `stream.capnp`). Until that schema grows a
        // `nodeId`/`relays` field, a StreamInfo cannot publish a dialable iroh
        // reach, so this maps to `None` (publishers carry the Quic reach). The
        // local `dial_stream` iroh arm is covered by the dial.rs loopback test.
        ReachTransport::Iroh => None,
    }
}

/// Networked variant of [`moq_stream_handle_task`]: dial a `/moq`
/// `web_transport_quinn` session from the `reach` list instead of the UDS
/// plane, then run the identical subscribe + chained-HMAC verify loop (#274).
#[allow(clippy::too_many_arguments)]
async fn moq_stream_handle_task_networked(
    reach: Vec<crate::stream_info::Destination>,
    broadcast_path: String,
    mac_key: [u8; 32],
    enc_key: [u8; 32],
    topic: String,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<crate::streaming::StreamPayload>>,
    cancel: tokio_util::sync::CancellationToken,
) {
    use crate::streaming::StreamVerifier;
    use moq_net::{Client as MoqClient, Origin, Track};

    // Dial the first reach we can resolve + connect.
    let mut session = None;
    let mut last_err: Option<String> = None;
    for opt in &reach {
        let Some(cfg) = reach_to_transport_config(opt) else {
            continue;
        };
        match crate::dial::dial_stream(&cfg).await {
            Ok(s) => {
                session = Some(s);
                break;
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }
    let Some(session) = session else {
        let msg = last_err.unwrap_or_else(|| "no dialable reach in StreamInfo".to_owned());
        let _ = tx.send(Err(anyhow!("moq networked dial failed: {msg}"))).await;
        return;
    };

    let client_origin = Origin::random().produce();
    let client_consumer = client_origin.consume();
    let moq_client = MoqClient::new().with_consume(client_origin);
    // `session` is a `MoqStreamSession` (quinn or iroh, #282) — dispatch the moq
    // handshake to the concrete transport.
    let _session = match session.connect_moq(&moq_client).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(Err(anyhow!("moq handshake: {e}"))).await;
            return;
        }
    };
    let bc = match tokio::time::timeout(
        BROADCAST_ANNOUNCE_TIMEOUT,
        client_consumer.announced_broadcast(&broadcast_path),
    )
    .await
    {
        Ok(Some(bc)) => bc,
        Ok(None) => {
            let _ = tx.send(Err(anyhow!("broadcast {broadcast_path} not announced"))).await;
            return;
        }
        Err(_) => {
            let _ = tx
                .send(Err(anyhow!("timeout waiting for broadcast {broadcast_path}")))
                .await;
            return;
        }
    };
    let track = match bc.subscribe_track(&Track::new(STREAM_TRACK)) {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.send(Err(anyhow!("subscribe_track: {e}"))).await;
            return;
        }
    };
    // #321: AEAD ON for this DH-keyed mesh stream — open sealed Tagged blocks.
    let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
    // #145: read groups by EXACT sequence (get_group), not arrival-order next_group.
    // Each Group is served on its own QUIC uni-stream, so Groups can arrive out of
    // order; next_group's monotonic cursor returns the first Group with sequence >=
    // its cursor in *arrival* order, so a lower-seq Group that arrives after a higher
    // one (e.g. the small terminal / always-retained max_sequence Group) is skipped —
    // fatally breaking the ordered, gap-fatal chained HMAC. get_group(seq) waits for
    // each exact Group (even out-of-order), guaranteeing in-order, gap-free delivery.
    let mut expected_seq = 0u64;
    loop {
        if cancel.is_cancelled() {
            break;
        }
        let mut group = match tokio::time::timeout(GROUP_IDLE_TIMEOUT, track.get_group(expected_seq)).await {
            Ok(Ok(Some(g))) => g,
            Ok(Ok(None)) => break,
            Err(_elapsed) => {
                let _ = tx
                    .send(Err(anyhow!("stream idle: no group for {}s", GROUP_IDLE_TIMEOUT.as_secs())))
                    .await;
                break;
            }
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("moq next_group: {e}"))).await;
                break;
            }
        };
        expected_seq += 1;
        let frame: bytes::Bytes = match tokio::time::timeout(FRAME_READ_TIMEOUT, group.read_frame()).await {
            Ok(Ok(Some(f))) => f,
            Ok(Ok(None)) => break,
            Ok(Err(e)) => {
                let _ = tx.send(Err(anyhow!("frame read error: {e}"))).await;
                break;
            }
            Err(_elapsed) => {
                let _ = tx
                    .send(Err(anyhow!("frame read timeout after {}s", FRAME_READ_TIMEOUT.as_secs())))
                    .await;
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
    // Zero-copy: pass slices into the received frame (`Bytes` from moq-net) straight
    // to the verifier — no per-frame `Vec` allocation of the (potentially large) payload.
    verifier.verify_parts(topic.as_bytes(), &frame[..split], &frame[split..])
}

/// Mesh consumer entry point (#321): verify a moq frame's chained HMAC + AEAD
/// (via the `verifier`) AND its per-host provenance signature against a roster.
///
/// `roster` resolves a signer's anchored ML-DSA-65 key; `is_enrolled` confirms the
/// signer is a known mesh peer. Provenance is REQUIRED (fail-closed): a block with
/// no/invalid/unknown signer is rejected. Use [`verify_moq_frame`] for the
/// non-mesh client path that has no roster.
pub fn verify_moq_frame_with_provenance(
    verifier: &mut StreamVerifier,
    topic: &str,
    frame: &[u8],
    roster: &dyn crate::envelope::PqTrustStore,
    is_enrolled: &dyn Fn(&[u8; 32]) -> bool,
) -> Result<Vec<crate::streaming::StreamPayload>> {
    if frame.len() < 16 {
        anyhow::bail!("moq frame too short: {} bytes", frame.len());
    }
    let split = frame.len() - 16;
    let capnp_data = &frame[..split];

    // 1) Chained HMAC + AEAD open (also rejects topic/order/MAC failures).
    let payloads = verifier.verify_parts(topic.as_bytes(), capnp_data, &frame[split..])?;

    // 2) Per-host provenance, layered on top: parse the block, reconstruct the
    //    provenance-cleared signed region, and verify the signature + roster.
    let mut slice: &[u8] = capnp_data;
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut slice,
        capnp::message::ReaderOptions::default(),
    )?;
    let block = reader.get_root::<crate::streaming_capnp::stream_block::Reader>()?;
    let prov = block.get_provenance()?;
    let signer_kid = prov.get_signer_kid()?;
    let sig = prov.get_sig()?;
    let signed_region = crate::stream_provenance::signed_region_from_block(&block)?;
    crate::stream_provenance::verify_provenance(
        signer_kid,
        sig,
        &signed_region,
        roster,
        is_enrolled,
    )?;

    Ok(payloads)
}

/// Seal a single Data/Complete payload into an AES-256-GCM `Tagged` payload
/// (#321). The 1-byte kind tag is prepended to the plaintext so the consumer can
/// restore the original variant. Error/Tagged/other variants pass through
/// unchanged (Error is operational, already-Tagged is the E2E notification path).
///
/// Shares its AAD + kind-tag framing with the cross-target open path
/// ([`crate::stream_consumer::open_sealed_payload`]).
fn seal_payload(
    enc_key: &[u8; 32],
    topic: &str,
    epoch: u64,
    payload: &StreamPayloadData,
) -> Result<StreamPayloadData> {
    use crate::crypto::event_crypto::{encrypt_event, EventPrivacy};
    use crate::stream_consumer::{stream_aead_aad, SEALED_KIND_COMPLETE, SEALED_KIND_DATA};

    let (kind, body): (u8, &[u8]) = match payload {
        StreamPayloadData::Data(d) => (SEALED_KIND_DATA, d),
        StreamPayloadData::Complete(d) => (SEALED_KIND_COMPLETE, d),
        // Leave non-sealed variants untouched.
        other => return Ok(other.clone()),
    };

    let mut plaintext = Vec::with_capacity(1 + body.len());
    plaintext.push(kind);
    plaintext.extend_from_slice(body);

    let aad = stream_aead_aad(topic, epoch);
    let (tag, ciphertext, nonce, key_commitment) =
        encrypt_event(enc_key, &aad, &plaintext, EventPrivacy::ZeroKnowledge)
            .map_err(|e| anyhow!("stream AEAD seal failed: {e}"))?;

    Ok(StreamPayloadData::Tagged {
        tag,
        payload: ciphertext,
        nonce: nonce.to_vec(),
        key_commitment: key_commitment.to_vec(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
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
        // #321: DH path is AEAD-on; the verifier shares the same enc_key.
        let enc_key = *ctx.enc_key().expect("DH ctx has enc_key");

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

        let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
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
        // #321: DH path is AEAD-on; the verifier shares the same enc_key.
        let enc_key = *ctx.enc_key().expect("DH ctx has enc_key");

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
        let mut verifier = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
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

    /// #275: a StreamInfo carrying a dialable Quic reach must be classified as
    /// dialable, so `networked()` routes to the producer's networked reach
    /// (`dial_stream`) rather than this process's local moq UDS plane.
    #[test]
    fn quic_reach_is_dialable() {
        use crate::stream_info::{Destination, QuicReach, Role, TransportConfig as ReachTransport};
        let reach = Destination {
            role: Role::Direct,
            transport: ReachTransport::Quic(QuicReach {
                addr: "127.0.0.1:4433".to_owned(),
                server_name: "localhost".to_owned(),
                cert_hashes: vec![vec![0u8; 32]],
            }),
        };
        assert!(
            reach_to_transport_config(&reach).is_some(),
            "a Quic reach must resolve to a dialable TransportConfig (selects the \
             networked dial_stream path, not the local UDS)"
        );
    }

    /// #275: the consumer dials the producer's networked reach even when this
    /// process also serves its own local moq UDS plane. Regression for the TUI
    /// timeout: the local UDS must NOT shadow a present reach (it is this
    /// process's plane, not the producer's).
    ///
    /// We assert the branch selection deterministically. With a dialable reach
    /// present we set the local UDS path to a socket that does NOT exist: the
    /// **UDS branch** would fail in milliseconds with a connect error naming that
    /// path, whereas the **networked branch** dials QUIC (which, to a closed
    /// loopback port, keeps retrying past a short deadline). So: a UDS-path error
    /// before the deadline ⇒ the pre-fix bug (UDS preferred — FAIL); reaching the
    /// deadline still trying ⇒ the networked dial was taken (PASS). A networked
    /// dial error is also an acceptable PASS (reach was dialed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn networked_prefers_reach_over_local_uds() -> Result<()> {
        use crate::stream_info::{Destination, QuicReach, Role, TransportConfig as ReachTransport};

        // Pretend this process has a local moq UDS plane (the TUI daemon does, to
        // serve its PTY stdout). The pre-fix code short-circuited onto this path
        // and ignored the reach; the fix must dial the reach instead. The path is
        // distinctive so a UDS connect error is unambiguously identifiable.
        let uds_marker = "/nonexistent/hyprstream-275-uds-marker.sock";
        let _ = GLOBAL_MOQ_UDS_PATH.set(PathBuf::from(uds_marker));

        // A dialable Quic reach to a closed loopback port.
        let reach = vec![Destination {
            role: Role::Direct,
            transport: ReachTransport::Quic(QuicReach {
                addr: "127.0.0.1:1".to_owned(),
                server_name: "localhost".to_owned(),
                cert_hashes: vec![[0xABu8; 32].to_vec()],
            }),
        }];

        let mut handle = MoqStreamHandle::networked(
            reach,
            "local/streams/test/deadbeef".to_owned(),
            [0u8; 32],
            [0u8; 32],
            "deadbeef".repeat(8),
        );

        match tokio::time::timeout(std::time::Duration::from_secs(3), handle.recv_next()).await {
            // Still dialing QUIC at the deadline — the networked branch was taken.
            Err(_elapsed) => Ok(()),
            // An early error: it must be a *networked* dial failure, NOT a UDS
            // connect to our marker path (which would prove the bug).
            Ok(res) => {
                let err = res.expect_err("dial to a closed port must not yield a payload");
                let msg = err.to_string();
                assert!(
                    !msg.contains(uds_marker),
                    "consumer connected to the local UDS plane instead of dialing the \
                     producer's reach (#275 regression): {msg}"
                );
                assert!(
                    msg.contains("networked dial"),
                    "expected a networked dial failure (reach was dialed), got: {msg}"
                );
                Ok(())
            }
        }
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

    // ── #321 AEAD seal/open ────────────────────────────────────────────────

    #[test]
    fn aead_seal_open_roundtrip() {
        let enc_key = [0x42u8; 32];
        let topic = "deadbeef";
        let epoch = 7u64;
        for payload in [
            StreamPayloadData::Data(b"hello tokens".to_vec()),
            StreamPayloadData::Complete(b"{\"done\":true}".to_vec()),
        ] {
            let sealed = seal_payload(&enc_key, topic, epoch, &payload).unwrap();
            let StreamPayloadData::Tagged { tag, payload: ct, nonce, key_commitment } = &sealed
            else {
                panic!("seal must produce a Tagged payload");
            };
            let opened = crate::stream_consumer::open_sealed_payload(
                &enc_key, topic, epoch, tag, ct, nonce, key_commitment,
            )
            .unwrap();
            match (&payload, &opened) {
                (StreamPayloadData::Data(a), StreamPayload::Data(b))
                | (StreamPayloadData::Complete(a), StreamPayload::Complete(b)) => {
                    assert_eq!(a, b);
                }
                _ => panic!("opened variant must match sealed variant"),
            }
        }
    }

    #[test]
    fn aead_open_rejects_wrong_key_and_tamper() {
        let enc_key = [0x42u8; 32];
        let topic = "t";
        let epoch = 1u64;
        let sealed =
            seal_payload(&enc_key, topic, epoch, &StreamPayloadData::Data(b"secret".to_vec()))
                .unwrap();
        let StreamPayloadData::Tagged { tag, payload: ct, nonce, key_commitment } = &sealed else {
            panic!("expected Tagged");
        };

        // Wrong key.
        let wrong = [0x99u8; 32];
        assert!(crate::stream_consumer::open_sealed_payload(
            &wrong, topic, epoch, tag, ct, nonce, key_commitment
        )
        .is_err());

        // Wrong epoch (AAD mismatch) — anti-replay across epochs.
        assert!(crate::stream_consumer::open_sealed_payload(
            &enc_key, topic, 2, tag, ct, nonce, key_commitment
        )
        .is_err());

        // Tampered ciphertext.
        let mut bad_ct = ct.clone();
        if let Some(b) = bad_ct.first_mut() {
            *b ^= 0xFF;
        }
        assert!(crate::stream_consumer::open_sealed_payload(
            &enc_key, topic, epoch, tag, &bad_ct, nonce, key_commitment
        )
        .is_err());
    }

    // ── #321 provenance over a published block ─────────────────────────────

    /// Publish a block with provenance ON, then verify HMAC + AEAD + provenance
    /// against a roster; assert wrong/unknown signers are rejected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provenance_publish_verify_and_reject() -> Result<()> {
        use crate::envelope::KeyedPqTrustStore;
        use crate::stream_provenance::ProvenanceSigner;
        use ed25519_dalek::SigningKey;

        let origin = origin();
        let (_cs, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();
        let enc_key = *ctx.enc_key().expect("DH ctx has enc_key");

        let host_ed = SigningKey::from_bytes(&[5u8; 32]);
        let signer = ProvenanceSigner::from_ed25519(host_ed.clone());
        let kid = signer.signer_kid();

        let mut pub_ = origin.publisher_with_provenance(&ctx, Some(signer))?;
        pub_.publish_data(b"hi").await?;
        pub_.complete_ref(b"{}").await?;

        // Drain the two frames from the shared origin.
        let path = origin.broadcast_path(&topic);
        let bc = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await?
        .ok_or_else(|| anyhow!("broadcast not announced"))?;
        let mut track = bc.subscribe_track(&Track::new(STREAM_TRACK))?;
        let mut frames: Vec<bytes::Bytes> = Vec::new();
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
            frames.push(frame);
        }

        // Roster anchoring the host's mesh ML-DSA key + enrolled-set closure.
        use ml_dsa::Keypair;
        let mut roster = KeyedPqTrustStore::new();
        roster.bind(kid, &crate::node_identity::derive_mesh_mldsa_key(&host_ed).verifying_key());
        let enrolled = |k: &[u8; 32]| *k == kid;

        // Valid: verify HMAC + AEAD + provenance.
        let mut v = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        let got = verify_moq_frame_with_provenance(&mut v, &topic, &frames[0], &roster, &enrolled)?;
        assert!(matches!(&got[0], StreamPayload::Data(d) if d == b"hi"));

        // Unknown signer: empty roster / not enrolled → reject (re-verify frame 0
        // with a fresh verifier so the HMAC chain restarts at the same block).
        let empty = KeyedPqTrustStore::new();
        let none_enrolled = |_: &[u8; 32]| false;
        let mut v2 = StreamVerifier::new(mac_key, topic.clone()).with_enc_key(enc_key);
        assert!(verify_moq_frame_with_provenance(
            &mut v2, &topic, &frames[0], &empty, &none_enrolled
        )
        .is_err());
        Ok(())
    }
}
