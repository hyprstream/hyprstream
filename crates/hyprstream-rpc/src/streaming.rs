//! Generic streaming infrastructure for authenticated PUB/SUB communication.
//!
//! This module provides rate-controlled, MAC-authenticated streaming that works
//! for both inference (UTF-8 tokens) and worker I/O (arbitrary binary data).
//!
//! # Architecture
//!
//! ```text
//! Producer                              Consumer
//! ────────                              ────────
//! StreamBuilder::with_dh() ◄── DH ──► StreamHandle::new()
//!       │                                    │
//!       │ add_data(rate)                     │ next()
//!       ▼                                    ▼
//! [topic, capnp, mac] ────────────► StreamVerifier
//! ```
//!
//! # Wire Format
//!
//! ```text
//! ZMQ Multipart:
//!   Frame 0: topic (64 hex chars, DH-derived)
//!   Frame 1: capnp StreamBlock
//!   Frame 2: mac (16 bytes truncated MAC)
//! ```
//!
//! # Security
//!
//! - DH key exchange: Ristretto255 ECDH derives topic + mac_key
//! - MAC chain: Each block's MAC depends on previous, enforces ordering
//! - E2E authentication: StreamService is blind forwarder
//!
//! # Backend
//!
//! - Default: Blake3 `keyed_hash()` (~10+ GB/s with SIMD)
//! - FIPS mode: HMAC-SHA256 (FIPS 198-1)

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use anyhow::Result;
use capnp::message::Builder;
use capnp::serialize;
use futures::{SinkExt, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::auth::Claims;
use crate::prelude::SigningKey;
use crate::registry::{global as endpoint_registry, SocketKind};

use tmq::SocketExt;

use crate::crypto::{derive_stream_keys, keyed_mac_truncated};

// DH key types - Ristretto255 (default) or P-256 (FIPS)
#[cfg(not(feature = "fips"))]
use crate::crypto::{ristretto_dh as dh_compute, RistrettoPublic as DhPublic, RistrettoSecret as DhSecret};

#[cfg(feature = "fips")]
use crate::crypto::{p256_dh as dh_compute, P256PublicKey as DhPublic, P256SecretKey as DhSecret};
use crate::streaming_capnp;

// ============================================================================
// StreamInfo — re-exported from target-independent module
// ============================================================================

pub use crate::stream_info::StreamInfo;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a StreamPublisher socket.
///
/// Controls high-water mark and whether the publisher gets a dedicated socket
/// (bypasses the shared StreamChannel socket for high-frequency publishers like TUI frames).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPublisherConfig {
    /// ZMQ send high-water mark. 0 = unlimited. Default: 1000.
    #[serde(default = "default_sndhwm")]
    pub sndhwm: i32,
    /// If true, the publisher creates its own PUSH socket instead of sharing.
    #[serde(default)]
    pub dedicated: bool,
}

fn default_sndhwm() -> i32 { 1000 }

impl Default for StreamPublisherConfig {
    fn default() -> Self {
        Self { sndhwm: default_sndhwm(), dedicated: false }
    }
}

/// Socket variant for StreamPublisher — shared (default) or dedicated.
///
/// `Shared` reuses the StreamChannel's single PUSH socket (suitable for most streams).
/// `Dedicated` owns its own PUSH socket (for high-frequency publishers like TUI frame loops
/// where HWM saturation on the shared socket would block other streams).
pub enum PublisherSocket {
    /// Shared socket from StreamChannel (existing behavior).
    Shared(Arc<tokio::sync::Mutex<tmq::push::Push>>),
    /// Dedicated socket owned by this publisher. Wrapped in `Option` for `Drop` take.
    Dedicated(Option<tmq::push::Push>),
}

/// Configuration for adaptive batching (rate control).
///
/// Controls how payloads are batched based on throughput rate.
/// Higher rates → larger batches (reduced overhead).
/// Lower rates → smaller batches (reduced latency).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Minimum payloads per block (1 = immediate send for slow streams)
    #[serde(default = "default_min_batch_size")]
    pub min_batch_size: usize,

    /// Maximum payloads per block
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Maximum block size in bytes
    #[serde(default = "default_max_block_bytes")]
    pub max_block_bytes: usize,

    /// Minimum rate for logarithmic scaling (payloads/sec)
    #[serde(default = "default_min_rate")]
    pub min_rate: f32,

    /// Maximum rate for logarithmic scaling (payloads/sec)
    #[serde(default = "default_max_rate")]
    pub max_rate: f32,
}

fn default_min_batch_size() -> usize { 1 }
fn default_max_batch_size() -> usize { 16 }
fn default_max_block_bytes() -> usize { 65536 }
fn default_min_rate() -> f32 { 1.0 }
fn default_max_rate() -> f32 { 100.0 }

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            min_batch_size: default_min_batch_size(),
            max_batch_size: default_max_batch_size(),
            max_block_bytes: default_max_block_bytes(),
            min_rate: default_min_rate(),
            max_rate: default_max_rate(),
        }
    }
}

// ============================================================================
// Payload Types
// ============================================================================

/// Input payload for StreamBuilder (what gets serialized).
///
/// Stream identity comes from the DH-derived topic, not from payload fields.
#[derive(Clone, Debug)]
pub enum StreamPayloadData {
    /// Generic binary data (tokens, I/O, etc.)
    Data(Vec<u8>),
    /// Error during streaming
    Error(String),
    /// Completion with app-specific metadata
    Complete(Vec<u8>),
    /// Encrypted tagged payload with key commitment
    Tagged {
        tag: Vec<u8>,
        payload: Vec<u8>,
        nonce: Vec<u8>,
        key_commitment: Vec<u8>,
    },
}

/// Output payload from StreamVerifier (what gets parsed).
///
/// Re-exported from `stream_consumer` for backwards compatibility.
pub use crate::stream_consumer::StreamPayload;

// ============================================================================
// HMAC Chain State
// ============================================================================

/// HMAC chain state for StreamBlock with 16-byte truncated MACs.
///
/// MAC chain:
// StreamHmacState was moved to crypto::hmac for cross-platform availability.
// Re-exported here for backward compatibility.
pub use crate::crypto::StreamHmacState;

// ============================================================================
// Stream Frames
// ============================================================================

/// ZMQ multipart frames for a StreamBlock.
pub struct StreamFrames {
    /// Frame 0: topic (64 hex chars)
    pub topic: Vec<u8>,
    /// Frame 1: capnp StreamBlock
    pub capnp: Vec<u8>,
    /// Frame 2: 16-byte truncated HMAC
    pub mac: [u8; 16],
}

impl StreamFrames {
    /// Send frames via raw ZMQ socket (sync).
    ///
    /// Use this for low-level streaming code that manages its own zmq sockets.
    /// For service-level code, prefer `StreamChannel::run_stream()`.
    pub fn send(&self, socket: &zmq::Socket) -> Result<()> {
        socket.send(&self.topic, zmq::SNDMORE)?;
        socket.send(&self.capnp, zmq::SNDMORE)?;
        socket.send(self.mac.as_slice(), 0)?;
        Ok(())
    }

    /// Send frames via async tmq Push socket.
    pub async fn send_async(&self, socket: &mut tmq::push::Push) -> Result<()> {
        let multipart = tmq::Multipart::from(vec![
            self.topic.clone(),
            self.capnp.clone(),
            self.mac.to_vec(),
        ]);
        socket.send(multipart).await
            .map_err(|e| anyhow::anyhow!("Failed to send stream frames: {}", e))?;
        Ok(())
    }

    /// Try to send frames via async tmq Push socket with zero timeout.
    ///
    /// Returns `Ok(())` if sent, `Err` if the send would block (HWM full).
    /// Used by `try_publish_data()` for non-blocking frame delivery.
    pub async fn try_send_async(&self, socket: &mut tmq::push::Push) -> Result<()> {
        let multipart = tmq::Multipart::from(vec![
            self.topic.clone(),
            self.capnp.clone(),
            self.mac.to_vec(),
        ]);
        // tmq::push::Push wraps ZMQ PUSH — if HWM is hit and SNDTIMEO is 0,
        // send returns EAGAIN. We rely on the socket having SNDTIMEO=0 for
        // dedicated sockets used in frame loops.
        socket.send(multipart).await
            .map_err(|e| anyhow::anyhow!("try_send failed (HWM full?): {}", e))?;
        Ok(())
    }

    /// Convert to Vec of frames for buffering.
    pub fn to_vec(self) -> Vec<Vec<u8>> {
        vec![self.topic, self.capnp, self.mac.to_vec()]
    }
}

// ============================================================================
// Stream Builder (Producer)
// ============================================================================

/// Stream producer with adaptive batching and DH encapsulation.
///
/// # Example
///
/// ```ignore
/// // Create with DH key exchange
/// let mut builder = StreamBuilder::with_dh(
///     config,
///     &server_secret,
///     &client_pubkey,
///     &server_pubkey,
/// )?;
///
/// // Add data with rate-based batching
/// while let Some(data) = source.next().await {
///     if let Some(frames) = builder.add_data(&stream_id, &data, rate)? {
///         frames.send(&socket)?;
///     }
/// }
///
/// // Final flush
/// if let Some(frames) = builder.flush()? {
///     frames.send(&socket)?;
/// }
/// ```
pub struct StreamBuilder {
    config: BatchingConfig,
    hmac_state: StreamHmacState,
    pending: Vec<StreamPayloadData>,
    pending_bytes: usize,
    /// Producer-assigned monotonic counter per epoch (#219). Named `sequenceNumber`
    /// per IETF convention (DTLS RFC 9147, RTP RFC 3550). Incremented once per
    /// emitted block; stamped into the block (MAC-covered) for resume/dedup/anti-replay.
    /// Epoch is 0 until the #223 epoch lifecycle lands.
    sequence_number: u64,
}

impl StreamBuilder {
    /// Create a new StreamBuilder with raw keys.
    pub fn new(config: BatchingConfig, mac_key: [u8; 32], topic: String) -> Self {
        Self {
            config,
            hmac_state: StreamHmacState::new(mac_key, topic),
            pending: Vec::new(),
            pending_bytes: 0,
            sequence_number: 0,
        }
    }

    /// Create with DH key derivation (encapsulated).
    ///
    /// Performs DH (Ristretto255 or P-256 in FIPS mode) and derives topic + mac_key internally.
    ///
    /// # Note
    ///
    /// FIPS mode uses P-256 which requires 33-byte compressed public keys.
    /// Default mode uses Ristretto255 with 32-byte keys.
    pub fn with_dh(
        config: BatchingConfig,
        server_secret: &DhSecret,
        client_pubkey: &[u8],
        server_pubkey: &[u8],
    ) -> Result<Self> {
        // Perform DH
        let client_pub = DhPublic::from_slice(client_pubkey)
            .ok_or_else(|| anyhow::anyhow!("Invalid client public key"))?;
        let shared_secret = dh_compute(server_secret, &client_pub);

        // Derive stream keys (needs 32-byte arrays for salt computation)
        // For non-32-byte keys, hash them to 32 bytes
        let client_pub_32 = pubkey_to_32(client_pubkey);
        let server_pub_32 = pubkey_to_32(server_pubkey);
        let keys = derive_stream_keys(&shared_secret, &client_pub_32, &server_pub_32)?;

        Ok(Self::new(config, *keys.mac_key, keys.topic))
    }

    /// Compute adaptive batch size based on rate (logarithmic scaling).
    fn adaptive_batch_size(&self, rate: f32) -> usize {
        let min = self.config.min_batch_size;
        let max = self.config.max_batch_size;

        let log_min_rate = self.config.min_rate.max(1.0).ln();
        let log_max_rate = self.config.max_rate.max(1.0).ln();

        let log_rate = rate.max(1.0).ln();
        let t = ((log_rate - log_min_rate) / (log_max_rate - log_min_rate)).clamp(0.0, 1.0);

        // Interpolate: result is in [min, max]
        let batch_f32 = (min as f32 + t * (max - min) as f32).round();

        // Return early for boundary/edge cases (no cast needed)
        if !batch_f32.is_finite() || batch_f32 <= min as f32 {
            return min;
        }
        if batch_f32 >= max as f32 {
            return max;
        }

        // SAFETY: batch_f32 is in (min, max), both small positive integers (typically 1-64).
        // Clippy can't verify bounds, but we've proven batch_f32 > min >= 1 and < max.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        { batch_f32 as usize }
    }

    /// Add data payload with adaptive batching.
    ///
    /// Returns frames if batch is ready to send.
    pub fn add_data(&mut self, data: &[u8], rate: f32) -> Result<Option<StreamFrames>> {
        self.pending_bytes += data.len() + 8; // Estimate overhead
        self.pending.push(StreamPayloadData::Data(data.to_vec()));

        let batch_size = self.adaptive_batch_size(rate);
        if self.pending.len() >= batch_size || self.pending_bytes >= self.config.max_block_bytes {
            self.flush()
        } else {
            Ok(None)
        }
    }

    /// Add error payload (flushes immediately).
    pub fn add_error(&mut self, message: &str) -> Result<Option<StreamFrames>> {
        self.pending.push(StreamPayloadData::Error(message.to_owned()));
        self.flush()
    }

    /// Add completion payload (flushes immediately).
    pub fn add_complete(&mut self, metadata: &[u8]) -> Result<Option<StreamFrames>> {
        self.pending.push(StreamPayloadData::Complete(metadata.to_vec()));
        self.flush()
    }

    /// Flush pending payloads to StreamFrames.
    pub fn flush(&mut self) -> Result<Option<StreamFrames>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let payloads = std::mem::take(&mut self.pending);
        self.pending_bytes = 0;

        // Build StreamBlock capnp message (shared encoder; also used by the
        // moq plane in `moq_stream.rs`).
        let sequence_number = self.sequence_number;
        self.sequence_number += 1;
        // epoch is 0 until the #223 key-epoch lifecycle lands.
        let capnp_bytes = encode_stream_block(self.hmac_state.prev_mac_bytes(), sequence_number, 0, &payloads)?;

        let mac = self.hmac_state.compute_next(&capnp_bytes);

        Ok(Some(StreamFrames {
            topic: self.hmac_state.topic().as_bytes().to_vec(),
            capnp: capnp_bytes,
            mac,
        }))
    }

    /// Consume builder and flush remaining payloads.
    pub fn finish(mut self) -> Result<Option<StreamFrames>> {
        self.flush()
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        self.hmac_state.topic()
    }
}

// ============================================================================
// Stream Context (Encapsulates stream state for producers)
// ============================================================================

/// Encapsulates all state needed to produce a stream.
///
/// Created by `StreamingService::prepare_stream()` after DH key exchange.
/// Contains everything needed to:
/// - Return stream info to client (stream_id, server_pubkey)
/// - Create a `StreamPublisher` for sending data
///
/// # Example
///
/// ```ignore
/// // In service handler:
/// let ctx = self.prepare_stream(envelope_ctx)?;
///
/// // Return to client:
/// let response = Response::stream_started(ctx.stream_id(), ctx.server_pubkey());
///
/// // After response sent, create publisher:
/// let mut publisher = StreamPublisher::new(&push_socket, &ctx)?;
/// publisher.publish_data(b"hello")?;
/// publisher.complete(b"{}")?;
/// ```
#[derive(Clone)]
pub struct StreamContext {
    /// Human-readable stream ID (for logging/display)
    stream_id: String,
    /// DH-derived topic (64 hex chars) - used for ZMQ routing
    topic: String,
    /// DH-derived HMAC key for authenticated stream blocks
    mac_key: [u8; 32],
    /// Server's ephemeral public key - client needs this for DH
    server_pubkey: [u8; 32],
    /// DH-derived control channel topic (64 hex chars)
    ctrl_topic: String,
    /// DH-derived control channel HMAC key
    ctrl_mac_key: [u8; 32],
    /// Cancellation token — fired by control listener or JWT expiry
    cancel_token: CancellationToken,
    /// QoS options advertised in StreamInfo and honoured by MoqStreamPublisher (#169).
    qos: crate::stream_info::StreamOpt,
}

impl StreamContext {
    /// Create a new stream context with pre-computed DH values.
    pub fn new(
        stream_id: String,
        topic: String,
        mac_key: [u8; 32],
        server_pubkey: [u8; 32],
    ) -> Self {
        Self {
            stream_id,
            topic,
            mac_key,
            server_pubkey,
            ctrl_topic: String::new(),
            ctrl_mac_key: [0u8; 32],
            cancel_token: CancellationToken::new(),
            qos: crate::stream_info::StreamOpt::default(),
        }
    }

    /// Create stream context by performing DH key exchange.
    ///
    /// # Arguments
    /// * `client_ephemeral_pubkey` - Client's ephemeral public key from request
    ///
    /// # Returns
    /// Stream context with DH-derived topic and mac_key
    #[cfg(not(feature = "fips"))]
    pub fn from_dh(client_ephemeral_pubkey: &[u8]) -> Result<Self> {
        use crate::crypto::generate_ephemeral_keypair;

        let (server_secret, server_pubkey) = generate_ephemeral_keypair();
        let server_pubkey_bytes = server_pubkey.to_bytes();

        let client_pub = DhPublic::from_slice(client_ephemeral_pubkey)
            .ok_or_else(|| anyhow::anyhow!("Invalid client ephemeral public key"))?;
        let shared_secret = dh_compute(&server_secret, &client_pub);

        let client_pub_32 = pubkey_to_32(client_ephemeral_pubkey);
        let server_pub_32 = pubkey_to_32(&server_pubkey_bytes);
        let keys = derive_stream_keys(&shared_secret, &client_pub_32, &server_pub_32)?;

        let stream_id = format!("stream-{}", uuid::Uuid::new_v4());

        Ok(Self {
            stream_id,
            topic: keys.topic,
            mac_key: *keys.mac_key,
            server_pubkey: server_pubkey_bytes,
            ctrl_topic: keys.ctrl_topic,
            ctrl_mac_key: *keys.ctrl_mac_key,
            cancel_token: CancellationToken::new(),
            qos: crate::stream_info::StreamOpt::default(),
        })
    }

    /// Get the stream ID (for logging/display).
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// Get the DH-derived topic (for ZMQ routing).
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Get the MAC key (for HMAC chain).
    pub fn mac_key(&self) -> &[u8; 32] {
        &self.mac_key
    }

    /// Get the server's ephemeral public key (client needs this for DH).
    pub fn server_pubkey(&self) -> &[u8; 32] {
        &self.server_pubkey
    }

    /// Get the control channel topic.
    pub fn ctrl_topic(&self) -> &str {
        &self.ctrl_topic
    }

    /// Get the control channel MAC key.
    pub fn ctrl_mac_key(&self) -> &[u8; 32] {
        &self.ctrl_mac_key
    }

    /// Get the cancellation token.
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Set the QoS options for this stream using a typed preset (#169).
    ///
    /// Call with a preset ZST: `.with_qos_preset::<Job>()`, `.with_qos_preset::<Pipe>()`.
    pub fn with_qos_preset<Q: crate::stream_info::StreamOptPreset>(mut self) -> Self {
        self.qos = Q::stream_opt();
        self
    }

    /// Set the QoS options for this stream from a runtime value (#169).
    pub fn with_qos(mut self, qos: crate::stream_info::StreamOpt) -> Self {
        self.qos = qos;
        self
    }

    /// Get the QoS options for this stream (#169).
    pub fn qos(&self) -> &crate::stream_info::StreamOpt {
        &self.qos
    }
}

// ============================================================================
// Stream Publisher (High-level producer API)
// ============================================================================

/// High-level async stream publisher with automatic batching and MAC chain.
///
/// Wraps `StreamBuilder` and an async tmq Push socket to provide a simple API for
/// publishing stream data. Handles all the complexity of batching,
/// MAC computation, and frame sending.
///
/// # Example
///
/// ```ignore
/// let publisher = StreamPublisher::new(socket_arc.clone(), &stream_ctx);
///
/// // Publish data (batched automatically)
/// for chunk in data_source {
///     publisher.publish_data(&chunk).await?;
/// }
///
/// // Or publish progress updates
/// publisher.publish_progress("Downloading", 50, 100).await?;
///
/// // Complete the stream
/// publisher.complete(b"{\"status\":\"done\"}").await?;
/// ```
pub struct StreamPublisher {
    builder: StreamBuilder,
    socket: PublisherSocket,
    cancel_token: CancellationToken,
    terminated: bool,
}

impl StreamPublisher {
    /// Create a new publisher from a stream context (shared socket).
    pub fn new(socket: Arc<tokio::sync::Mutex<tmq::push::Push>>, ctx: &StreamContext) -> Self {
        Self::with_config(socket, ctx, BatchingConfig::default())
    }

    /// Create a new publisher with custom batching config (shared socket).
    pub fn with_config(
        socket: Arc<tokio::sync::Mutex<tmq::push::Push>>,
        ctx: &StreamContext,
        config: BatchingConfig,
    ) -> Self {
        let builder = StreamBuilder::new(config, ctx.mac_key, ctx.topic.clone());
        Self {
            builder,
            socket: PublisherSocket::Shared(socket),
            cancel_token: ctx.cancel_token.clone(),
            terminated: false,
        }
    }

    /// Create a publisher with a dedicated PUSH socket.
    ///
    /// The dedicated socket is owned exclusively by this publisher, avoiding
    /// contention with other streams on the shared StreamChannel socket.
    /// Used for high-frequency publishers (e.g., TUI frame loops at 30fps).
    pub fn with_dedicated_socket(
        socket: tmq::push::Push,
        ctx: &StreamContext,
        config: BatchingConfig,
    ) -> Self {
        let builder = StreamBuilder::new(config, ctx.mac_key, ctx.topic.clone());
        Self {
            builder,
            socket: PublisherSocket::Dedicated(Some(socket)),
            cancel_token: ctx.cancel_token.clone(),
            terminated: false,
        }
    }

    /// Publish binary data with automatic batching.
    ///
    /// Data is batched based on the configured batch size and rate.
    /// Use `flush()` to force immediate send.
    pub async fn publish_data(&mut self, data: &[u8]) -> Result<()> {
        self.publish_data_with_rate(data, 10.0).await
    }

    /// Publish binary data with explicit rate for adaptive batching.
    ///
    /// Higher rates result in larger batches (more efficient).
    /// Lower rates result in smaller batches (lower latency).
    pub async fn publish_data_with_rate(&mut self, data: &[u8], rate: f32) -> Result<()> {
        if self.cancel_token.is_cancelled() {
            anyhow::bail!("stream cancelled");
        }
        if let Some(frames) = self.builder.add_data(data, rate)? {
            self.send_frames(frames).await?;
        }
        Ok(())
    }

    /// Try to publish data without blocking on a full HWM.
    ///
    /// Returns `Ok(true)` if the data was sent, `Ok(false)` if the socket's
    /// high-water mark is full (zero-timeout send). Useful for frame loops
    /// where a slow viewer should be skipped rather than blocking the loop.
    ///
    /// Only meaningful with `Dedicated` sockets; `Shared` sockets always await.
    pub async fn try_publish_data(&mut self, data: &[u8], rate: f32) -> Result<bool> {
        if self.cancel_token.is_cancelled() {
            anyhow::bail!("stream cancelled");
        }
        if let Some(frames) = self.builder.add_data(data, rate)? {
            match &mut self.socket {
                PublisherSocket::Dedicated(Some(sock)) => {
                    match frames.try_send_async(sock).await {
                        Ok(()) => return Ok(true),
                        Err(_) => return Ok(false),
                    }
                }
                _ => {
                    self.send_frames(frames).await?;
                }
            }
        }
        Ok(true)
    }

    /// Send frames via the appropriate socket variant.
    async fn send_frames(&mut self, frames: StreamFrames) -> Result<()> {
        match &mut self.socket {
            PublisherSocket::Shared(arc) => {
                let mut socket = arc.lock().await;
                frames.send_async(&mut socket).await?;
            }
            PublisherSocket::Dedicated(Some(sock)) => {
                frames.send_async(sock).await?;
            }
            PublisherSocket::Dedicated(None) => {
                anyhow::bail!("dedicated socket already taken (publisher dropped?)");
            }
        }
        Ok(())
    }

    /// Publish a progress update.
    ///
    /// Convenience method that formats progress as `stage:current:total`.
    /// Client should parse this format or use a structured payload.
    pub async fn publish_progress(&mut self, stage: &str, current: usize, total: usize) -> Result<()> {
        let data = format!("{}:{}:{}", stage, current, total);
        self.publish_data(data.as_bytes()).await
    }

    /// Publish an error and flush immediately.
    pub async fn publish_error(&mut self, message: &str) -> Result<()> {
        self.terminated = true; // Before await — cancellation-safe
        if let Some(frames) = self.builder.add_error(message)? {
            self.send_frames(frames).await?;
        }
        Ok(())
    }

    /// Complete the stream with metadata and flush.
    ///
    /// This consumes the publisher - no more data can be sent after completion.
    pub async fn complete(mut self, metadata: &[u8]) -> Result<()> {
        self.complete_ref(metadata).await
    }

    /// Complete the stream with metadata and flush (by reference).
    ///
    /// Same as `complete()` but doesn't consume self, allowing use in
    /// callback-based APIs like `StreamChannel::run_stream()`.
    pub async fn complete_ref(&mut self, metadata: &[u8]) -> Result<()> {
        self.terminated = true; // Before await — cancellation-safe
        if let Some(frames) = self.builder.add_complete(metadata)? {
            self.send_frames(frames).await?;
        }
        if let Some(frames) = self.builder.flush()? {
            self.send_frames(frames).await?;
        }
        Ok(())
    }

    /// Flush any pending batched data immediately.
    pub async fn flush(&mut self) -> Result<()> {
        if let Some(frames) = self.builder.flush()? {
            self.send_frames(frames).await?;
        }
        Ok(())
    }

    /// Get the topic being published to.
    pub fn topic(&self) -> &str {
        self.builder.topic()
    }

    /// Check if this stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Whether a terminal frame (Error or Complete) has been sent.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }
}

impl Drop for StreamPublisher {
    fn drop(&mut self) {
        if self.terminated || self.cancel_token.is_cancelled() {
            return;
        }

        // Take the original builder to preserve HMAC chain state.
        // The replacement is a dummy that will never be used (we're being dropped).
        let mut builder = std::mem::replace(
            &mut self.builder,
            StreamBuilder::new(BatchingConfig::default(), [0u8; 32], String::new()),
        );

        let frames = match builder.add_error("publisher dropped without terminal frame") {
            Ok(Some(f)) => f,
            _ => return,
        };

        // Take the socket out of self for the spawned flush task.
        let socket = std::mem::replace(
            &mut self.socket,
            PublisherSocket::Dedicated(None),
        );

        // Spawn async send — the spawned task yields to the runtime, so Drop
        // itself returns immediately. Bounded 200ms wait covers brief lock
        // contention from concurrent publishers or register_topic() on the
        // same StreamChannel. If the lock cannot be acquired, log an error
        // so the failure is observable rather than silent.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    match socket {
                        PublisherSocket::Shared(arc) => {
                            match tokio::time::timeout(
                                std::time::Duration::from_millis(200),
                                arc.lock(),
                            )
                            .await
                            {
                                Ok(mut sock) => {
                                    let _ = frames.send_async(&mut sock).await;
                                }
                                Err(_) => {
                                    tracing::error!(
                                        "StreamPublisher::drop: could not acquire socket lock \
                                         within 200ms; terminal error frame not sent — \
                                         client will hang until JWT expiry"
                                    );
                                }
                            }
                        }
                        PublisherSocket::Dedicated(Some(mut sock)) => {
                            let _ = frames.send_async(&mut sock).await;
                        }
                        PublisherSocket::Dedicated(None) => {
                            tracing::error!(
                                "StreamPublisher::drop: dedicated socket already taken"
                            );
                        }
                    }
                });
            }
            Err(_) => {
                tracing::error!(
                    "StreamPublisher::drop: no tokio runtime; \
                     terminal error frame not sent"
                );
            }
        }
    }
}

// ============================================================================
// Progress Streaming (Bridges progress callbacks to streams)
// ============================================================================

/// Progress update message for channel-based streaming.
#[derive(Debug, Clone)]
pub enum ProgressUpdate {
    /// Progress update: stage, current, total
    Progress { stage: String, current: usize, total: usize },
    /// Operation completed successfully
    Complete(Vec<u8>),
    /// Operation failed with error
    Error(String),
}

/// Channel-based progress sender for use with git2db or other progress-reporting operations.
///
/// This can be passed to operations that accept a progress callback (like git clone).
/// Progress updates are sent through a channel and can be forwarded to a stream.
///
/// # Example
///
/// ```ignore
/// // Create progress channel
/// let (sender, receiver) = progress_channel();
///
/// // In spawn_blocking (runs the git operation):
/// let reporter = ChannelProgressReporter::new(sender);
/// git2db_clone_with_progress(..., reporter);
/// reporter.complete(b"{}")?;
///
/// // In async context (publishes to stream):
/// let mut publisher = StreamPublisher::new(&socket, &ctx);
/// while let Ok(update) = receiver.recv() {
///     match update {
///         ProgressUpdate::Progress { stage, current, total } => {
///             publisher.publish_progress(&stage, current, total)?;
///         }
///         ProgressUpdate::Complete(meta) => {
///             publisher.complete(&meta)?;
///             break;
///         }
///         ProgressUpdate::Error(msg) => {
///             publisher.publish_error(&msg)?;
///             break;
///         }
///     }
/// }
/// ```
#[derive(Clone)]
pub struct ChannelProgressReporter {
    sender: tokio::sync::mpsc::Sender<ProgressUpdate>,
}

impl ChannelProgressReporter {
    /// Create a new channel-based progress reporter.
    pub fn new(sender: tokio::sync::mpsc::Sender<ProgressUpdate>) -> Self {
        Self { sender }
    }

    /// Report progress (implements the same interface as git2db::ProgressReporter).
    ///
    /// Uses `blocking_send` since this is typically called from sync contexts.
    pub fn report(&self, stage: &str, current: usize, total: usize) {
        let _ = self.sender.blocking_send(ProgressUpdate::Progress {
            stage: stage.to_owned(),
            current,
            total,
        });
    }

    /// Signal successful completion with metadata.
    pub fn complete(&self, metadata: Vec<u8>) -> Result<()> {
        self.sender.blocking_send(ProgressUpdate::Complete(metadata))
            .map_err(|_| anyhow::anyhow!("Progress channel closed"))
    }

    /// Signal an error occurred.
    pub fn error(&self, message: &str) -> Result<()> {
        self.sender.blocking_send(ProgressUpdate::Error(message.to_owned()))
            .map_err(|_| anyhow::anyhow!("Progress channel closed"))
    }
}

/// Create a progress channel for streaming operations.
///
/// Returns (sender, receiver) where:
/// - sender: Pass to `ChannelProgressReporter::new()` for use in blocking operations
/// - receiver: Poll in async context to forward updates to `StreamPublisher`
pub fn progress_channel(buffer_size: usize) -> (
    tokio::sync::mpsc::Sender<ProgressUpdate>,
    tokio::sync::mpsc::Receiver<ProgressUpdate>,
) {
    tokio::sync::mpsc::channel(buffer_size)
}

/// Helper to forward progress updates from a tokio channel to a StreamPublisher.
///
/// This is a convenience function for the common pattern of receiving progress
/// updates from a blocking operation and forwarding them to a stream.
///
/// # Arguments
/// * `receiver` - Tokio channel receiving progress updates
/// * `publisher` - StreamPublisher to forward updates to
///
/// # Returns
/// Ok(()) on successful completion, Err on any error
pub async fn forward_progress_to_stream(
    mut receiver: tokio::sync::mpsc::Receiver<ProgressUpdate>,
    publisher: &mut StreamPublisher,
) -> Result<()> {
    while let Some(update) = receiver.recv().await {
        match update {
            ProgressUpdate::Progress { stage, current, total } => {
                publisher.publish_progress(&stage, current, total).await?;
            }
            ProgressUpdate::Complete(metadata) => {
                publisher.complete_ref(&metadata).await?;
                return Ok(());
            }
            ProgressUpdate::Error(msg) => {
                publisher.publish_error(&msg).await?;
                return Err(anyhow::anyhow!("Operation failed: {}", msg));
            }
        }
    }
    // Channel closed without completion - treat as error
    publisher.publish_error("Progress channel closed unexpectedly").await?;
    Err(anyhow::anyhow!("Progress channel closed unexpectedly"))
}

// ============================================================================
// Stream Verifier (Consumer helper)
// ============================================================================

// #224: the canonical StreamVerifier lives in `stream_consumer` (cross-target — compiles
// on native + wasm). The native-only duplicate the cross-target extraction left here is
// removed; re-export the shared one (mirrors the `StreamPayload` re-export above) so
// `crate::StreamVerifier`, the moq path, and the wasm/browser consumer all use a single
// verifier — the one place #163's policy-selected enforcement will land.
pub use crate::stream_consumer::StreamVerifier;

// ============================================================================
// Stream Handle (Consumer)
// ============================================================================

/// E2E authenticated stream consumer with DH encapsulation.
///
/// # Example
///
/// ```ignore
/// let mut handle = StreamHandle::new(
///     &context,
///     stream_id,
///     endpoint,
///     &server_pubkey,
///     &client_secret,
///     &client_pubkey,
/// )?;
///
/// use futures::StreamExt;
/// while let Some(payload) = handle.next().await {
///     match payload? {
///         StreamPayload::Data(data) => process(data),
///         StreamPayload::Complete(_) => break,
///         StreamPayload::Error(message) => return Err(message.into()),
///     }
/// }
/// ```
pub struct StreamHandle {
    subscriber: tmq::subscribe::Subscribe,
    stream_id: String,
    topic: String,
    verifier: StreamVerifier,
    pending: VecDeque<StreamPayload>,
    completed: bool,
    /// PUSH socket for sending control messages (lazy, consumer → StreamService)
    /// Kept as sync zmq::Socket — cancel is best-effort fire-and-forget with DONTWAIT.
    ctrl_push: Option<zmq::Socket>,
    /// Control channel topic (DH-derived)
    ctrl_topic: String,
    /// Control channel MAC key
    ctrl_mac_key: [u8; 32],
    /// Cancellation token — consumer-side; fired by cancel() to unblock poll_next
    cancel_token: CancellationToken,
}

// StreamHandle must be Send for ToolResult::Stream(Box<StreamHandle>)
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _check() {
        _assert_send::<StreamHandle>();
    }
};

impl StreamHandle {
    /// Create with DH key derivation (encapsulated).
    ///
    /// # Note
    ///
    /// FIPS mode uses P-256 which requires 33-byte compressed public keys.
    /// Default mode uses Ristretto255 with 32-byte keys.
    pub fn new(
        context: &zmq::Context,
        stream_id: String,
        endpoint: &str,
        server_pubkey: &[u8],
        client_secret: &DhSecret,
        client_pubkey: &[u8],
        qos: crate::stream_info::StreamOpt,
    ) -> Result<Self> {
        // Perform DH
        let server_pub = DhPublic::from_slice(server_pubkey)
            .ok_or_else(|| anyhow::anyhow!("Invalid server public key"))?;
        let shared_secret = dh_compute(client_secret, &server_pub);

        // Derive stream keys (needs 32-byte arrays for salt computation)
        let client_pub_32 = pubkey_to_32(client_pubkey);
        let server_pub_32 = pubkey_to_32(server_pubkey);
        let keys = derive_stream_keys(&shared_secret, &client_pub_32, &server_pub_32)?;

        // Create async TMQ subscriber
        let subscriber = tmq::subscribe::subscribe(context)
            .connect(endpoint)
            .map_err(|e| anyhow::anyhow!("SUB connect to {}: {}", endpoint, e))?
            .subscribe(keys.topic.as_bytes())
            .map_err(|e| anyhow::anyhow!("SUB subscribe to topic: {}", e))?;

        subscriber.set_linger(0)
            .map_err(|e| anyhow::anyhow!("Failed to set linger on SUB: {}", e))?;

        tracing::debug!(
            stream_id = %stream_id,
            topic = %keys.topic,
            endpoint = %endpoint,
            "Subscribed to E2E authenticated stream"
        );

        let verifier = StreamVerifier::with_policy(*keys.mac_key, keys.topic.clone(), qos.into());

        // Set up control channel PUSH socket (consumer → StreamService → producer)
        let ctrl_push = context.socket(zmq::PUSH).ok();
        if let Some(ref sock) = ctrl_push {
            let push_endpoint = endpoint_registry()
                .endpoint("streams", SocketKind::Push)
                .to_zmq_string();
            // Best-effort connect — cancel is not critical path
            let _ = sock.connect(&push_endpoint);
        }

        Ok(Self {
            subscriber,
            stream_id,
            topic: keys.topic,
            verifier,
            pending: VecDeque::new(),
            completed: false,
            ctrl_push,
            ctrl_topic: keys.ctrl_topic,
            ctrl_mac_key: *keys.ctrl_mac_key,
            cancel_token: CancellationToken::new(),
        })
    }

    /// Receive next payload (async).
    ///
    /// Returns `None` when stream is complete or socket closed.
    /// Prefer using `StreamExt::next()` directly for composability.
    pub async fn recv_next(&mut self) -> Result<Option<StreamPayload>> {
        self.next().await.transpose()
    }

    /// Get the stream ID.
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Check if stream is complete.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Get the cancellation token (clone to trigger from another context).
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Send a cancel control message to the producer via the control channel.
    ///
    /// Also fires the cancellation token to unblock `poll_next` immediately.
    /// Best-effort: if the PUSH socket is unavailable or send fails, the
    /// JWT expiry timeout will still stop the stream.
    pub fn cancel(&self) {
        self.cancel_token.cancel();  // Unblock poll_next (waker fires)

        let Some(ref sock) = self.ctrl_push else { return };

        // Build StreamControl::Cancel capnp message
        let mut builder = Builder::new_default();
        {
            let mut ctrl = builder.init_root::<crate::streaming_capnp::stream_control::Builder>();
            ctrl.set_cancel(());
        }
        let mut buf = Vec::new();
        if serialize::write_message(&mut buf, &builder).is_err() {
            return;
        }

        // Compute HMAC over the capnp payload using ctrl_mac_key
        let mac = keyed_mac_truncated(&self.ctrl_mac_key, &buf);

        // Send [ctrl_topic, capnp, mac] — best effort, non-blocking
        let _ = sock.send(self.ctrl_topic.as_bytes(), zmq::SNDMORE | zmq::DONTWAIT);
        let _ = sock.send(&buf, zmq::SNDMORE | zmq::DONTWAIT);
        let _ = sock.send(mac.as_slice(), zmq::DONTWAIT);

        tracing::debug!(stream_id = %self.stream_id, "sent cancel on control channel");
    }
}

impl Stream for StreamHandle {
    type Item = Result<StreamPayload>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            // Fast path: drain buffered payloads
            if let Some(payload) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(payload)));
            }
            if self.completed {
                return Poll::Ready(None);
            }

            // Poll cancellation token — registers cx.waker() with the token.
            // When cancel() fires, the waker is invoked and poll_next re-runs.
            let is_cancelled = {
                let cancel_fut = std::pin::pin!(self.cancel_token.cancelled());
                cancel_fut.poll(cx).is_ready()
            };
            if is_cancelled {
                self.completed = true;
                return Poll::Ready(None);
            }

            // Poll TMQ subscriber — registers cx.waker() with epoll.
            // Now the SAME waker is registered with BOTH token AND epoll.
            match Pin::new(&mut self.subscriber).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.completed = true;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e.into()))),
                Poll::Ready(Some(Ok(multipart))) => {
                    let frames: Vec<Vec<u8>> =
                        multipart.into_iter().map(|m| m.to_vec()).collect();

                    if frames.len() != 3 || frames[2].len() != 16 {
                        return Poll::Ready(Some(Err(anyhow::anyhow!(
                            "Invalid StreamBlock format: expected 3 frames with 16-byte MAC, got {} frames",
                            frames.len()
                        ))));
                    }

                    match self.verifier.verify(&frames) {
                        Err(e) => return Poll::Ready(Some(Err(e))),
                        Ok(payloads) => {
                            for p in payloads {
                                if matches!(
                                    p,
                                    StreamPayload::Complete(..) | StreamPayload::Error(..)
                                ) {
                                    self.completed = true;
                                }
                                self.pending.push_back(p);
                            }
                            // Loop back: if payloads is non-empty, pop_front
                            // returns the first one. If empty (heartbeat-only),
                            // we re-poll the TMQ subscriber which will return
                            // Poll::Pending and register the waker via epoll.
                            continue;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// StreamChannel - Service-Level Streaming Infrastructure
// ============================================================================

/// Service-level async streaming infrastructure that encapsulates socket management
/// and stream pre-authorization.
///
/// `StreamChannel` handles the complexity of:
/// - Async PUSH socket creation and lazy initialization (via tmq)
/// - DH key exchange via `StreamContext::from_dh()`
/// - Stream pre-authorization with StreamService
/// - Endpoint discovery via the registry
///
/// # Example
///
/// ```ignore
/// // In service initialization
/// let stream_channel = StreamChannel::new(signing_key.clone());
///
/// // In request handler
/// let stream_ctx = stream_channel.prepare_stream(&client_pubkey, 600).await?;
///
/// // In async context
/// let mut publisher = stream_channel.publisher(&stream_ctx).await?;
/// publisher.publish_progress("processing", 0, 100).await?;
/// // ... do work ...
/// publisher.complete(&metadata).await?;
/// ```
pub struct StreamChannel {
    /// Ed25519 signing key for stream authorization (wired into moq publish claims, N7).
    #[allow(dead_code)]
    signing_key: SigningKey,
    /// ML-DSA-65 signing key for the post-quantum half of the StreamRegister
    /// composite signature. Mirrors `LocalSigner` on the RPC plane so the
    /// streaming control plane signs under the same policy (#161). Auto-
    /// generated; the node's persistent ML-DSA identity is wired in via
    /// [`Self::with_pq_key`] once peer attestation lands (#157).
    pq_signing_key: Option<crate::crypto::pq::MlDsaSigningKey>,
}

impl StreamChannel {
    /// Create a new stream channel.
    ///
    /// Auto-generates an ML-DSA-65 key for the post-quantum half of the
    /// StreamRegister composite, mirroring `LocalSigner::new` on the RPC
    /// plane. Use [`Self::with_pq_key`] to install the node's persistent
    /// ML-DSA identity (#157).
    pub fn new(signing_key: SigningKey) -> Self {
        let (pq_sk, _) = crate::crypto::pq::ml_dsa_generate_keypair();
        Self {
            signing_key,
            pq_signing_key: Some(pq_sk),
        }
    }

    /// Install the node's persistent ML-DSA-65 signing key, replacing the
    /// auto-generated one. The matching public key must be anchored in the
    /// StreamService verifier's PQ trust store for Hybrid verification to
    /// succeed (peer attestation, #157).
    pub fn with_pq_key(mut self, key: crate::crypto::pq::MlDsaSigningKey) -> Self {
        self.pq_signing_key = Some(key);
        self
    }

    /// Prepare a stream with DH key exchange and pre-authorization.
    ///
    /// This method:
    /// 1. Performs DH key exchange to derive topic and MAC key
    /// 2. Pre-authorizes the stream with StreamService
    ///
    /// # Arguments
    /// * `client_ephemeral_pubkey` - Client's ephemeral public key (32 bytes)
    /// * `expiry_secs` - Stream expiration in seconds from now
    ///
    /// # Returns
    /// `StreamContext` ready for publishing
    pub async fn prepare_stream(
        &self,
        client_ephemeral_pubkey: &[u8],
        expiry_secs: i64,
    ) -> Result<StreamContext> {
        self.prepare_stream_with_claims(client_ephemeral_pubkey, expiry_secs, None).await
    }

    /// Prepare a stream with DH key exchange, pre-authorization, and claims.
    ///
    /// Same as `prepare_stream()` but allows passing user claims for authorization.
    /// Claims are forwarded to StreamService for subscription-time validation.
    ///
    /// # Arguments
    /// * `client_ephemeral_pubkey` - Client's ephemeral public key (32 bytes)
    /// * `expiry_secs` - Stream expiration in seconds from now
    /// * `claims` - Optional user claims for authorization
    ///
    /// # Returns
    /// `StreamContext` ready for publishing
    pub async fn prepare_stream_with_claims(
        &self,
        client_ephemeral_pubkey: &[u8],
        expiry_secs: i64,
        _claims: Option<Claims>,
    ) -> Result<StreamContext> {
        // 1. DH key exchange
        let stream_ctx = StreamContext::from_dh(client_ephemeral_pubkey)?;

        // Spawn JWT expiry timeout as universal backstop.
        let token = stream_ctx.cancel_token().clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(expiry_secs.max(0) as u64)).await;
            token.cancel();
        });

        Ok(stream_ctx)
    }

    /// Register a topic with the stream plane.
    ///
    /// No-op on the moq path — topics are published lazily by
    /// `MoqStreamPublisher` on first frame; no pre-registration is needed.
    /// Kept for API compatibility with callers such as `NotificationService`.
    pub async fn register_topic(&self, _topic: &str, _expiry: i64, _claims: Option<Claims>) -> Result<()> {
        Ok(())
    }

    /// Create a publisher for a stream context.
    ///
    /// Fails loudly if the process-global moq stream origin has not been
    /// initialized (server not started). In production the `streams` factory
    /// always calls `init_global_moq_origin` before any service handles requests.
    pub async fn publisher(&self, ctx: &StreamContext) -> Result<crate::moq_stream::AnyStreamPublisher> {
        let origin = crate::moq_stream::global_moq_origin()
            .ok_or_else(|| anyhow::anyhow!("no moq stream origin — server not initialized"))?;
        Ok(crate::moq_stream::AnyStreamPublisher::Moq(origin.publisher(ctx)?))
    }

    /// Run a streaming operation with framework-guaranteed terminal frame.
    ///
    /// The closure receives an [`AnyStreamPublisher`] by value and **must return
    /// it** alongside its result. The framework sends a terminal frame if the
    /// closure didn't. Transport (moq-lite or ZMQ) is selected automatically
    /// based on whether a global moq origin is registered.
    ///
    /// After the closure returns:
    /// - If `Ok` and not terminated → framework sends `Complete` (empty metadata)
    /// - If `Err` and not terminated → framework sends `Error` with the error message
    /// - If already terminated → no-op (closure handled it)
    pub async fn run_stream<F, Fut, R>(&self, ctx: &StreamContext, f: F) -> Result<R>
    where
        F: FnOnce(crate::moq_stream::AnyStreamPublisher) -> Fut,
        Fut: Future<Output = (crate::moq_stream::AnyStreamPublisher, Result<R>)>,
    {
        let publisher = self.publisher(ctx).await?;
        let (mut publisher, result) = f(publisher).await;

        if !publisher.is_terminated() && !publisher.is_cancelled() {
            match &result {
                Ok(_) => {
                    if let Err(e) = publisher.complete_ref(b"").await {
                        tracing::error!("run_stream: failed to send Complete: {}", e);
                    }
                }
                Err(e) => {
                    if let Err(send_err) = publisher.publish_error(&e.to_string()).await {
                        tracing::error!("run_stream: failed to send Error: {}", send_err);
                    }
                }
            }
        }

        result
    }

    /// Create a publisher for a pre-registered topic (no DH).
    ///
    /// Used by NotificationService where topics are registered via `register_topic()`
    /// and don't use DH-based key exchange. The transport MAC key is randomly generated
    /// (separate from notification E2E MAC which is embedded in the payload).
    pub async fn publisher_for_topic(&self, topic: &str) -> Result<crate::moq_stream::AnyStreamPublisher> {
        // Generate a random transport-level MAC key for the HMAC chain.
        // (Not the notification's E2E MAC — that's embedded in the payload.)
        let mut mac_key = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut mac_key);

        let ctx = StreamContext::new(
            format!("notify-{}", uuid::Uuid::new_v4()),
            topic.to_owned(),
            mac_key,
            [0u8; 32], // No server pubkey needed for notification delivery
        );
        self.publisher(&ctx).await
    }

    /// Get the stream endpoint for clients to subscribe to.
    ///
    /// Returns the SUB endpoint URL from the registry.
    pub fn stream_endpoint(&self) -> String {
        endpoint_registry()
            .endpoint("streams", SocketKind::Sub)
            .to_zmq_string()
    }
}

// ============================================================================
// ResponseStream - Unified Response + Stream Coordination
// ============================================================================

/// Wrapper type for streaming responses that coordinates the REP-before-PUB pattern.
///
/// When a service handler needs to return a response AND start streaming,
/// `ResponseStream` bundles:
/// - The REP response bytes to send immediately
/// - The `StreamContext` for later publishing
/// - Application-specific pending work
///
/// This ensures consistent handling across all streaming services.
///
/// # Example
///
/// ```ignore
/// fn handle_clone_stream(&self, ...) -> Result<ResponseStream<CloneTask>> {
///     let stream_ctx = self.stream_channel.prepare_stream(&pubkey, 600)?;
///
///     let response = build_response(request_id, &stream_ctx);
///
///     Ok(ResponseStream::new(response, stream_ctx, CloneTask { url, name }))
/// }
/// ```
pub struct ResponseStream<T> {
    /// The REP response to send immediately
    pub response: Vec<u8>,
    /// Stream context for later publishing
    pub stream_ctx: StreamContext,
    /// Application-specific pending work
    pub pending: T,
}

impl<T> ResponseStream<T> {
    /// Create a new response stream.
    pub fn new(response: Vec<u8>, stream_ctx: StreamContext, pending: T) -> Self {
        Self {
            response,
            stream_ctx,
            pending,
        }
    }

    /// Get the stream ID.
    pub fn stream_id(&self) -> &str {
        self.stream_ctx.stream_id()
    }

    /// Get the server's public key for client DH.
    pub fn server_pubkey(&self) -> &[u8; 32] {
        self.stream_ctx.server_pubkey()
    }
}

// ============================================================================
// StreamGuard (Codegen hook for dispatch-level terminal frame enforcement)
// ============================================================================

/// Codegen hook point for dispatch-level streaming continuation wrapping.
///
/// Currently a **no-op pass-through**. The real terminal-frame guarantee
/// lives in [`StreamChannel::run_stream()`], which service handlers call
/// directly. This struct exists so generated dispatch code has a single
/// place to wrap continuations — a future iteration can extend `wrap()`
/// to inject `run_stream` automatically (removing the need for handlers
/// to call it themselves).
pub struct StreamGuard;

impl StreamGuard {
    /// Wrap a continuation (currently no-op pass-through).
    ///
    /// Terminal frame enforcement is provided by `StreamChannel::run_stream()`,
    /// not by this wrapper. See struct-level docs for roadmap.
    pub fn wrap(continuation: crate::service::Continuation) -> crate::service::Continuation {
        continuation
    }
}

// ============================================================================
// Streaming-response concurrency (#186)
// ============================================================================

/// Default ceiling on **concurrent server-side streaming responses per
/// service** — i.e. how many streaming RPCs one service may be actively pushing
/// data for at once. Overridable via [`install_max_concurrent_streams_per_service`]
/// (wired from `ServerConfig::max_concurrent_streams_per_service`).
///
/// This is the former per-`RequestLoop` `MAX_INFLIGHT_CONTINUATIONS` (16). When
/// the spawn moved out of the transport front-ends into the dispatch core
/// (#186) the bound is keyed by service name rather than living on each loop —
/// the *same* granularity, since each service has exactly one `RequestLoop`.
/// Keeping it per-service (not process-wide) preserves the original isolation:
/// one service's stuck or long-lived streams cannot starve another's.
pub const DEFAULT_MAX_CONCURRENT_STREAMS_PER_SERVICE: usize = 16;

/// Process-global override for the per-service concurrent-streams cap.
/// First-write-wins, mirroring `install_verify_config`; installed once at
/// startup from the loaded `ServerConfig`. Read when a service's semaphore is
/// first created, so it must be installed before serving (it always is — the
/// daemon installs it right after loading config).
static MAX_CONCURRENT_STREAMS_PER_SERVICE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Install the per-service concurrent-streams cap. Call once at startup with
/// `ServerConfig::max_concurrent_streams_per_service`. Values are clamped to a
/// minimum of 1. Returns `Err(existing)` if already installed (first-write-wins).
pub fn install_max_concurrent_streams_per_service(n: usize) -> Result<(), usize> {
    MAX_CONCURRENT_STREAMS_PER_SERVICE.set(n.max(1))
}

/// The effective cap: the installed value, or [`DEFAULT_MAX_CONCURRENT_STREAMS_PER_SERVICE`].
fn max_concurrent_streams_per_service() -> usize {
    MAX_CONCURRENT_STREAMS_PER_SERVICE
        .get()
        .copied()
        .unwrap_or(DEFAULT_MAX_CONCURRENT_STREAMS_PER_SERVICE)
}

/// Per-service admission semaphores, created on first use at the effective cap.
/// A permit is held for the full lifetime of a streaming response (which is
/// itself bounded by the stream's JWT/TTL cancel token in `StreamChannel`, so
/// permits are always eventually released — there is no unbounded hold). The map
/// only ever grows by the number of distinct services (small, bounded), so it is
/// never pruned.
fn stream_admission_semaphore(service_name: &str) -> std::sync::Arc<tokio::sync::Semaphore> {
    use parking_lot::RwLock;
    use std::collections::HashMap;
    static MAP: std::sync::OnceLock<RwLock<HashMap<String, std::sync::Arc<tokio::sync::Semaphore>>>> =
        std::sync::OnceLock::new();
    let map = MAP.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(sem) = map.read().get(service_name) {
        return std::sync::Arc::clone(sem);
    }
    std::sync::Arc::clone(
        map.write()
            .entry(service_name.to_owned())
            .or_insert_with(|| {
                std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent_streams_per_service()))
            }),
    )
}

/// Spawn the server-side half of a streaming RPC — the task that keeps pushing
/// data *after* the `StreamInfo` reply has been sent — onto the current
/// `LocalSet`, bounded by a per-service admission permit.
///
/// Replaces the former "transport front-end spawns the continuation it got back
/// from `process_request`" model (#186): the dispatch core now spawns this task
/// itself and returns only the reply bytes, so the streaming lifecycle is no
/// longer threaded through every transport's request/response path. This is the
/// M1 shape; in M2 the streaming transport (`StreamChannel`) moves to a
/// service-owned moq broadcast and this coarse per-service cap is replaced by
/// the StreamOpt backpressure axes (#134).
///
/// `service_name` keys the per-service permit pool (see
/// [`DEFAULT_MAX_CONCURRENT_STREAMS_PER_SERVICE`] and
/// [`install_max_concurrent_streams_per_service`]). When the pool is saturated
/// the task waits for a permit rather than being dropped — and the wait is
/// logged so saturation is observable rather than a silent stall.
///
/// # Invariant
///
/// MUST be called from within a `tokio::task::LocalSet`: the task is `?Send`
/// (like every [`RequestService`](crate::service::RequestService)), and
/// `spawn_local` panics outside a `LocalSet`. Every
/// [`process_request`](crate::service::dispatch::process_request) caller already
/// runs on a `LocalSet` (ZMQ `RequestLoop`, the WebTransport server, and the
/// generic plane's `LocalServiceBridge`), which is the only place this is
/// invoked.
pub fn spawn_streaming_response(service_name: &str, continuation: crate::service::Continuation) {
    let sem = stream_admission_semaphore(service_name);
    let service_name = service_name.to_owned();
    tokio::task::spawn_local(async move {
        // Observe saturation: a permit that isn't immediately available means
        // this service is at its concurrent-streams cap and new streams are
        // queueing behind running ones.
        let permit = match sem.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                tracing::warn!(
                    service = %service_name,
                    cap = max_concurrent_streams_per_service(),
                    "concurrent-streams cap reached; new stream waiting for a slot"
                );
                // The semaphore is a static that is never closed, so
                // acquire_owned cannot fail; if it somehow did, drop the stream.
                let Ok(p) = sem.acquire_owned().await else {
                    tracing::error!(service = %service_name, "stream admission semaphore closed; dropping stream");
                    return;
                };
                p
            }
        };
        let _permit = permit; // held for the streaming response's lifetime
        continuation.await;
    });
}

// ============================================================================
// Helpers
// ============================================================================

/// Encode a `StreamBlock` capnp message from a `prev_mac` + payload list.
///
/// Factored out of [`StreamBuilder::flush`] so the moq streaming plane
/// (`crate::moq_stream`) can produce byte-identical StreamBlocks for the §7.5
/// chained-HMAC tokenstream.
pub fn encode_stream_block(
    prev_mac: &[u8],
    sequence_number: u64,
    epoch: u64,
    payloads: &[StreamPayloadData],
) -> Result<Vec<u8>> {
    let mut msg = Builder::new_default();
    {
        let mut block = msg.init_root::<streaming_capnp::stream_block::Builder>();
        block.set_prev_mac(prev_mac);
        // Producer-assigned counter (#219); MAC-covered implicitly (whole block is signed).
        // epoch is 0 until the #223 key-epoch lifecycle lands.
        block.set_sequence_number(sequence_number);
        block.set_epoch(epoch);

        // Cap'n Proto uses u32 for list lengths
        let payloads_len = u32::try_from(payloads.len()).unwrap_or(u32::MAX);
        let mut list = block.init_payloads(payloads_len);
        for (i, payload) in payloads.iter().enumerate() {
            let idx = u32::try_from(i).unwrap_or(u32::MAX);
            let mut p = list.reborrow().get(idx);
            match payload {
                StreamPayloadData::Data(data) => {
                    p.set_data(data);
                }
                StreamPayloadData::Error(message) => {
                    let mut err = p.init_error();
                    err.set_message(message);
                    err.set_code("");
                    err.set_details("");
                }
                StreamPayloadData::Complete(data) => {
                    p.set_complete(data);
                }
                StreamPayloadData::Tagged { tag, payload, nonce, key_commitment } => {
                    let mut tagged = p.init_tagged();
                    tagged.set_tag(tag);
                    tagged.set_payload(payload);
                    tagged.set_nonce(nonce);
                    tagged.set_key_commitment(key_commitment);
                }
            }
        }
    }

    let mut capnp_bytes = Vec::new();
    serialize::write_message(&mut capnp_bytes, &msg)?;
    Ok(capnp_bytes)
}

/// Convert a public key to 32 bytes for derive_stream_keys.
///
/// - If 32 bytes: use as-is (Ristretto255)
/// - If different length: hash with Blake3/SHA-256 to get 32 bytes (P-256)
fn pubkey_to_32(pubkey: &[u8]) -> [u8; 32] {
    if pubkey.len() == 32 {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(pubkey);
        arr
    } else {
        // Hash to 32 bytes for non-32-byte keys (P-256 is 33 bytes compressed)
        crate::crypto::keyed_mac(&[0u8; 32], pubkey)
    }
}

// `constant_time_eq` removed with the duplicate StreamVerifier (#224) — the canonical
// verifier in `stream_consumer` has its own.

#[cfg(test)]
mod tests {
    use super::*;

    fn build_stream_register_envelope(
        topic: &str,
        expiry: i64,
        signing_key: &SigningKey,
        pq_signing_key: Option<&crate::crypto::pq::MlDsaSigningKey>,
        _claims: Option<Claims>,
    ) -> Vec<u8> {
        use crate::common_capnp;
        use crate::envelope::{RequestEnvelope, SignedEnvelope};
        use crate::ToCapnp;

        let mut inner_msg = Builder::new_default();
        {
            let mut register = inner_msg.init_root::<crate::streaming_capnp::stream_register::Builder>();
            register.set_topic(topic);
            register.set_exp(expiry);
        }
        let mut inner_bytes = Vec::new();
        if let Err(e) = serialize::write_message(&mut inner_bytes, &inner_msg) {
            tracing::error!("Failed to serialize StreamRegister: {e}");
            return Vec::new();
        }
        let envelope = RequestEnvelope::new(inner_bytes);
        let policy = if pq_signing_key.is_some() {
            crate::crypto::CryptoPolicy::Hybrid
        } else {
            crate::crypto::CryptoPolicy::Classical
        };
        let signed =
            SignedEnvelope::new_signed_with_policy(envelope, signing_key, pq_signing_key, policy);
        let mut msg = Builder::new_default();
        {
            let mut builder = msg.init_root::<common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        let mut bytes = Vec::new();
        if let Err(e) = serialize::write_message(&mut bytes, &msg) {
            tracing::error!("Failed to serialize SignedEnvelope: {e}");
            return Vec::new();
        }
        bytes
    }

    /// #161: a StreamRegister built with a PQ key is a Hybrid (SNS) composite
    /// that verifies under the global Hybrid policy when the signer's ML-DSA
    /// key is anchored in the PQ trust store — and fail-closes when it isn't,
    /// exactly mirroring the RPC plane (no silent Classical downgrade).
    #[test]
    fn stream_register_hybrid_verifies_only_when_pq_anchored() -> anyhow::Result<()> {
        use crate::crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_from_bytes};
        use crate::crypto::CryptoPolicy;
        use crate::envelope::{
            InMemoryNonceCache, KeyedPqTrustStore, SignedEnvelope,
        };
        use crate::common_capnp;
        use crate::FromCapnp;

        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();

        let bytes = build_stream_register_envelope(
            "deadbeef".repeat(8).as_str(),
            i64::MAX,
            &signing_key,
            Some(&pq_sk),
            None,
        );
        assert!(!bytes.is_empty(), "register envelope must serialize");

        let reader = serialize::read_message(
            &mut std::io::Cursor::new(&bytes[..]),
            capnp::message::ReaderOptions::default(),
        )?;
        let signed = SignedEnvelope::read_from(
            reader.get_root::<common_capnp::signed_envelope::Reader>()?,
        )?;

        assert_eq!(signed.policy, CryptoPolicy::Hybrid, "must sign Hybrid with PQ key");

        // Anchored: the signer's ML-DSA vk is bound to its Ed25519 identity.
        let mut store = KeyedPqTrustStore::new();
        let vk = ml_dsa_vk_from_bytes(&crate::crypto::pq::ml_dsa_vk_bytes(&pq_vk))?;
        store.bind(signing_key.verifying_key().to_bytes(), &vk);
        let nonce_ok = InMemoryNonceCache::new();
        assert!(
            signed
                .verify_any_signer_with(&nonce_ok, Some(&store), CryptoPolicy::Hybrid)
                .is_ok(),
            "anchored Hybrid register must verify"
        );

        // Not anchored: empty store under Hybrid fails closed.
        let empty = KeyedPqTrustStore::new();
        let nonce_empty = InMemoryNonceCache::new();
        assert!(
            signed
                .verify_any_signer_with(&nonce_empty, Some(&empty), CryptoPolicy::Hybrid)
                .is_err(),
            "unanchored Hybrid register must fail closed (no Classical downgrade)"
        );
        Ok(())
    }

    #[test]
    fn test_batching_config_default() {
        let config = BatchingConfig::default();
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.max_block_bytes, 65536);
    }

    #[test]
    fn test_adaptive_batch_size() {
        let config = BatchingConfig::default();
        let builder = StreamBuilder::new(config, [0u8; 32], "topic".to_owned());

        // Low rate → small batch
        assert_eq!(builder.adaptive_batch_size(1.0), 1);

        // High rate → large batch
        assert_eq!(builder.adaptive_batch_size(100.0), 16);

        // Mid rate → mid batch
        let mid = builder.adaptive_batch_size(10.0);
        assert!(mid > 1 && mid < 16);
    }

    #[test]
    fn test_hmac_chain() {
        let key = [0x42u8; 32];
        let topic = "test_topic".to_owned();

        let mut state = StreamHmacState::new(key, topic);

        let mac1 = state.compute_next(b"data1");
        let mac2 = state.compute_next(b"data2");

        // MACs should be different
        assert_ne!(mac1, mac2);

        // Chain state should advance to mac2
        assert_eq!(state.prev_mac_bytes(), &mac2[..]);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn stream_block_carries_sequence_number_and_epoch() {
        // #219: the producer stamps sequenceNumber/epoch into the StreamBlock; since the
        // MAC covers the whole serialized block, they're authenticated implicitly. Confirm
        // the wire round-trips them (consumer enforcement is policy-selected, #163).
        let payloads = vec![StreamPayloadData::Data(b"hello".to_vec())];
        let bytes = encode_stream_block(&[0u8; 16], 42, 7, &payloads).unwrap();
        let reader = capnp::serialize::read_message(
            &mut std::io::Cursor::new(&bytes),
            capnp::message::ReaderOptions::default(),
        )
        .unwrap();
        let block = reader
            .get_root::<crate::streaming_capnp::stream_block::Reader>()
            .unwrap();
        assert_eq!(block.get_sequence_number(), 42);
        assert_eq!(block.get_epoch(), 7);
    }
}
