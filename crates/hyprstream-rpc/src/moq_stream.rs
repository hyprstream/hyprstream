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
                broadcasts: Mutex::new(Vec::new()),
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
    /// Keep broadcast producers alive for the lifetime of their publishers.
    broadcasts: Mutex<Vec<BroadcastProducer>>,
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
        self.inner.broadcasts.lock().push(broadcast);

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
// AnyStreamPublisher — unified publish API over ZMQ or moq-lite paths (#220)
// ============================================================================

/// Unified publisher that dispatches to either the ZMQ or moq-lite streaming
/// plane. The API surface matches `StreamPublisher` exactly so all call sites
/// that receive a publisher via closure (e.g. `StreamChannel::run_stream`) work
/// unchanged regardless of which transport is active.
pub enum AnyStreamPublisher {
    /// ZMQ PUSH path (legacy, kept behind `StreamService::with_zmq_fallback`).
    Zmq(crate::streaming::StreamPublisher),
    /// moq-lite in-process path (M2a primary).
    Moq(MoqStreamPublisher),
}

impl AnyStreamPublisher {
    /// Publish binary data. For the moq path the rate hint is ignored (each
    /// call maps 1:1 to one moq Group; batching is a ZMQ-path concern).
    pub async fn publish_data(&mut self, data: &[u8]) -> Result<()> {
        match self {
            Self::Zmq(p) => p.publish_data(data).await,
            Self::Moq(p) => p.publish_data(data).await,
        }
    }

    /// Publish binary data with adaptive-batching rate hint (ZMQ path only).
    /// On the moq path the rate is ignored and data is published immediately.
    pub async fn publish_data_with_rate(&mut self, data: &[u8], rate: f32) -> Result<()> {
        match self {
            Self::Zmq(p) => p.publish_data_with_rate(data, rate).await,
            Self::Moq(p) => {
                let _ = rate;
                p.publish_data(data).await
            }
        }
    }

    /// Publish a progress update (`stage:current:total`).
    pub async fn publish_progress(&mut self, stage: &str, current: usize, total: usize) -> Result<()> {
        let data = format!("{}:{}:{}", stage, current, total);
        self.publish_data(data.as_bytes()).await
    }

    /// Publish an error payload (terminal).
    pub async fn publish_error(&mut self, message: &str) -> Result<()> {
        match self {
            Self::Zmq(p) => p.publish_error(message).await,
            Self::Moq(p) => p.publish_error(message).await,
        }
    }

    /// Complete the stream with app-specific metadata (consumes self).
    pub async fn complete(self, metadata: &[u8]) -> Result<()> {
        match self {
            Self::Zmq(p) => p.complete(metadata).await,
            Self::Moq(p) => p.complete(metadata).await,
        }
    }

    /// Complete the stream without consuming self (for use inside `run_stream`).
    pub async fn complete_ref(&mut self, metadata: &[u8]) -> Result<()> {
        match self {
            Self::Zmq(p) => p.complete_ref(metadata).await,
            Self::Moq(p) => p.complete_ref(metadata).await,
        }
    }

    /// Flush any pending batched data (ZMQ path). No-op on the moq path.
    pub async fn flush(&mut self) -> Result<()> {
        match self {
            Self::Zmq(p) => p.flush().await,
            Self::Moq(_) => Ok(()), // moq publishes each block immediately
        }
    }

    /// The opaque topic this publisher serves.
    pub fn topic(&self) -> &str {
        match self {
            Self::Zmq(p) => p.topic(),
            Self::Moq(p) => p.topic(),
        }
    }

    /// Whether a terminal frame (Error/Complete) has been sent.
    pub fn is_terminated(&self) -> bool {
        match self {
            Self::Zmq(p) => p.is_terminated(),
            Self::Moq(p) => p.is_terminated(),
        }
    }

    /// Whether the stream has been cancelled via the `CancellationToken`.
    pub fn is_cancelled(&self) -> bool {
        match self {
            Self::Zmq(p) => p.is_cancelled(),
            Self::Moq(p) => p.is_cancelled(),
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

    /// `AnyStreamPublisher::Moq` round-trip: publish via the enum wrapper and
    /// verify the same bytes arrive on the moq consumer side.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn any_stream_publisher_moq_round_trip() -> Result<()> {
        let origin = origin();
        let (_client_secret, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let ctx = StreamContext::from_dh(&client_pub.to_bytes())?;
        let topic = ctx.topic().to_owned();
        let mac_key = *ctx.mac_key();

        // Publish via AnyStreamPublisher::Moq (the primary M2a path)
        let mut any_pub = AnyStreamPublisher::Moq(origin.publisher(&ctx)?);
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
