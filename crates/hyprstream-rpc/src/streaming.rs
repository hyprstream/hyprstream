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
use std::sync::Arc;

use anyhow::Result;
use capnp::message::Builder;
use capnp::serialize;
use dashmap::DashMap;
use futures::SinkExt;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::auth::Claims;
use crate::prelude::SigningKey;
use crate::registry::{global as endpoint_registry, SocketKind};

use crate::crypto::{derive_stream_keys, keyed_mac_truncated};

// DH key types - Ristretto255 (default) or P-256 (FIPS)
#[cfg(not(feature = "fips"))]
use crate::crypto::{ristretto_dh as dh_compute, RistrettoPublic as DhPublic, RistrettoSecret as DhSecret};

#[cfg(feature = "fips")]
use crate::crypto::{p256_dh as dh_compute, P256PublicKey as DhPublic, P256SecretKey as DhSecret};
use crate::streaming_capnp;

// ============================================================================
// Configuration
// ============================================================================

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
}

/// Output payload from StreamVerifier (what gets parsed).
///
/// Stream identity comes from the DH-derived topic, not from payload fields.
#[derive(Clone, Debug)]
pub enum StreamPayload {
    /// Generic binary data (tokens, I/O, etc.)
    Data(Vec<u8>),
    /// Error during streaming
    Error(String),
    /// Completion with app-specific metadata
    Complete(Vec<u8>),
}

// ============================================================================
// HMAC Chain State
// ============================================================================

/// HMAC chain state for StreamBlock with 16-byte truncated MACs.
///
/// MAC chain:
/// - Block 0: HMAC(key, topic || capnp)[..16]
/// - Block N: HMAC(key, prev_mac || capnp)[..16]
#[derive(Clone)]
pub struct StreamHmacState {
    key: [u8; 32],
    prev_mac: Option<[u8; 16]>,
    topic: String,
}

impl StreamHmacState {
    /// Create new HMAC chain state.
    pub fn new(key: [u8; 32], topic: String) -> Self {
        Self {
            key,
            prev_mac: None,
            topic,
        }
    }

    /// Compute 16-byte truncated MAC for next block.
    pub fn compute_next(&mut self, capnp_data: &[u8]) -> [u8; 16] {
        // Build input: (prev_mac or topic) || capnp_data
        let mut input = Vec::with_capacity(64 + capnp_data.len());
        match &self.prev_mac {
            None => input.extend_from_slice(self.topic.as_bytes()),
            Some(prev) => input.extend_from_slice(prev),
        }
        input.extend_from_slice(capnp_data);

        // Compute 16-byte truncated MAC using backend
        let truncated = keyed_mac_truncated(&self.key, &input);
        self.prev_mac = Some(truncated);
        truncated
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Get previous MAC bytes (for prevMac field in StreamBlock).
    pub fn prev_mac_bytes(&self) -> &[u8] {
        match &self.prev_mac {
            Some(mac) => mac,
            None => &self.topic.as_bytes()[..16.min(self.topic.len())],
        }
    }
}

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
    /// For service-level code, prefer `StreamChannel::with_publisher()`.
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
}

impl StreamBuilder {
    /// Create a new StreamBuilder with raw keys.
    pub fn new(config: BatchingConfig, mac_key: [u8; 32], topic: String) -> Self {
        Self {
            config,
            hmac_state: StreamHmacState::new(mac_key, topic),
            pending: Vec::new(),
            pending_bytes: 0,
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

        // Build StreamBlock capnp message
        let mut msg = Builder::new_default();
        {
            let mut block = msg.init_root::<streaming_capnp::stream_block::Builder>();
            block.set_prev_mac(self.hmac_state.prev_mac_bytes());

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
                }
            }
        }

        let mut capnp_bytes = Vec::new();
        serialize::write_message(&mut capnp_bytes, &msg)?;

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
    socket: Arc<tokio::sync::Mutex<tmq::push::Push>>,
    cancel_token: CancellationToken,
}

impl StreamPublisher {
    /// Create a new publisher from a stream context.
    pub fn new(socket: Arc<tokio::sync::Mutex<tmq::push::Push>>, ctx: &StreamContext) -> Self {
        Self::with_config(socket, ctx, BatchingConfig::default())
    }

    /// Create a new publisher with custom batching config.
    pub fn with_config(
        socket: Arc<tokio::sync::Mutex<tmq::push::Push>>,
        ctx: &StreamContext,
        config: BatchingConfig,
    ) -> Self {
        let builder = StreamBuilder::new(config, ctx.mac_key, ctx.topic.clone());
        Self {
            builder,
            socket,
            cancel_token: ctx.cancel_token.clone(),
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
            self.publish_error("cancelled").await?;
            anyhow::bail!("stream cancelled");
        }
        if let Some(frames) = self.builder.add_data(data, rate)? {
            let mut socket = self.socket.lock().await;
            frames.send_async(&mut socket).await?;
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
        if let Some(frames) = self.builder.add_error(message)? {
            let mut socket = self.socket.lock().await;
            frames.send_async(&mut socket).await?;
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
    /// callback-based APIs like `StreamChannel::with_publisher()`.
    pub async fn complete_ref(&mut self, metadata: &[u8]) -> Result<()> {
        let mut socket = self.socket.lock().await;
        if let Some(frames) = self.builder.add_complete(metadata)? {
            frames.send_async(&mut socket).await?;
        }
        if let Some(frames) = self.builder.flush()? {
            frames.send_async(&mut socket).await?;
        }
        Ok(())
    }

    /// Flush any pending batched data immediately.
    pub async fn flush(&mut self) -> Result<()> {
        if let Some(frames) = self.builder.flush()? {
            let mut socket = self.socket.lock().await;
            frames.send_async(&mut socket).await?;
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

/// HMAC chain verifier for StreamBlock.
pub struct StreamVerifier {
    key: [u8; 32],
    prev_mac: Option<[u8; 16]>,
    topic: String,
}

impl StreamVerifier {
    /// Create a new verifier.
    pub fn new(key: [u8; 32], topic: String) -> Self {
        Self {
            key,
            prev_mac: None,
            topic,
        }
    }

    /// Verify frames and return parsed payloads.
    ///
    /// Expected frames: [topic, capnp StreamBlock, 16-byte MAC]
    pub fn verify(&mut self, frames: &[Vec<u8>]) -> Result<Vec<StreamPayload>> {
        if frames.len() != 3 {
            anyhow::bail!("Expected 3 frames, got {}", frames.len());
        }

        let received_topic = &frames[0];
        let capnp_data = &frames[1];
        let received_mac = &frames[2];

        if received_mac.len() != 16 {
            anyhow::bail!("Expected 16-byte MAC, got {}", received_mac.len());
        }

        if received_topic != self.topic.as_bytes() {
            anyhow::bail!("Topic mismatch");
        }

        // Compute expected MAC
        let mut input = Vec::with_capacity(64 + capnp_data.len());
        match &self.prev_mac {
            None => input.extend_from_slice(self.topic.as_bytes()),
            Some(prev) => input.extend_from_slice(prev),
        }
        input.extend_from_slice(capnp_data);

        let expected_mac = keyed_mac_truncated(&self.key, &input);

        if !constant_time_eq(received_mac, &expected_mac) {
            anyhow::bail!("MAC verification failed");
        }

        // Update chain state
        let mut new_prev = [0u8; 16];
        new_prev.copy_from_slice(received_mac);
        self.prev_mac = Some(new_prev);

        // Parse StreamBlock
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(capnp_data),
            capnp::message::ReaderOptions::default(),
        )?;
        let block = reader.get_root::<streaming_capnp::stream_block::Reader>()?;

        let payloads_reader = block.get_payloads()?;
        let mut payloads = Vec::with_capacity(payloads_reader.len() as usize);

        for i in 0..payloads_reader.len() {
            let p = payloads_reader.get(i);

            use streaming_capnp::stream_payload::Which;
            let payload = match p.which()? {
                Which::Data(data_result) => {
                    StreamPayload::Data(data_result?.to_vec())
                }
                Which::Error(err_result) => {
                    let err = err_result?;
                    StreamPayload::Error(err.get_message()?.to_string()?)
                }
                Which::Complete(complete_result) => {
                    StreamPayload::Complete(complete_result?.to_vec())
                }
                Which::Heartbeat(()) => {
                    continue;
                }
            };

            payloads.push(payload);
        }

        Ok(payloads)
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

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
/// while let Some(payload) = handle.recv_next()? {
///     match payload {
///         StreamPayload::Data { data, .. } => process(data),
///         StreamPayload::Complete { .. } => break,
///         StreamPayload::Error { message, .. } => return Err(message.into()),
///     }
/// }
/// ```
pub struct StreamHandle {
    subscriber: zmq::Socket,
    stream_id: String,
    topic: String,
    verifier: StreamVerifier,
    pending: VecDeque<StreamPayload>,
    completed: bool,
    /// PUSH socket for sending control messages (lazy, consumer → StreamService)
    ctrl_push: Option<zmq::Socket>,
    /// Control channel topic (DH-derived)
    ctrl_topic: String,
    /// Control channel MAC key
    ctrl_mac_key: [u8; 32],
}

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
    ) -> Result<Self> {
        // Perform DH
        let server_pub = DhPublic::from_slice(server_pubkey)
            .ok_or_else(|| anyhow::anyhow!("Invalid server public key"))?;
        let shared_secret = dh_compute(client_secret, &server_pub);

        // Derive stream keys (needs 32-byte arrays for salt computation)
        let client_pub_32 = pubkey_to_32(client_pubkey);
        let server_pub_32 = pubkey_to_32(server_pubkey);
        let keys = derive_stream_keys(&shared_secret, &client_pub_32, &server_pub_32)?;

        // Create subscriber
        let subscriber = context.socket(zmq::SUB)?;
        subscriber.connect(endpoint)?;
        subscriber.set_subscribe(keys.topic.as_bytes())?;

        tracing::debug!(
            stream_id = %stream_id,
            topic = %keys.topic,
            endpoint = %endpoint,
            "Subscribed to E2E authenticated stream"
        );

        let verifier = StreamVerifier::new(*keys.mac_key, keys.topic.clone());

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
        })
    }

    /// Receive next payload (blocking).
    ///
    /// Returns `None` when stream is complete.
    pub fn recv_next(&mut self) -> Result<Option<StreamPayload>> {
        // Return buffered payloads first
        if let Some(payload) = self.pending.pop_front() {
            return Ok(Some(payload));
        }

        if self.completed {
            return Ok(None);
        }

        // Receive and verify
        let frames = self.subscriber.recv_multipart(0)?;

        if frames.len() != 3 || frames[2].len() != 16 {
            anyhow::bail!(
                "Invalid StreamBlock format: expected 3 frames with 16-byte MAC, got {} frames",
                frames.len()
            );
        }

        let payloads = self.verifier.verify(&frames)?;

        // Buffer payloads
        for payload in payloads {
            if matches!(payload, StreamPayload::Complete { .. } | StreamPayload::Error { .. }) {
                self.completed = true;
            }
            self.pending.push_back(payload);
        }

        // Return first
        Ok(self.pending.pop_front())
    }

    /// Try to receive next payload (non-blocking).
    pub fn try_next(&mut self) -> Result<Option<StreamPayload>> {
        // Return buffered payloads first
        if let Some(payload) = self.pending.pop_front() {
            return Ok(Some(payload));
        }

        if self.completed {
            return Ok(None);
        }

        // Non-blocking receive
        match self.subscriber.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => {
                if frames.len() != 3 || frames[2].len() != 16 {
                    anyhow::bail!("Invalid StreamBlock format");
                }

                let payloads = self.verifier.verify(&frames)?;

                for payload in payloads {
                    if matches!(payload, StreamPayload::Complete { .. } | StreamPayload::Error { .. }) {
                        self.completed = true;
                    }
                    self.pending.push_back(payload);
                }

                Ok(self.pending.pop_front())
            }
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(e) => Err(e.into()),
        }
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

    /// Send a cancel control message to the producer via the control channel.
    ///
    /// Best-effort: if the PUSH socket is unavailable or send fails, the
    /// JWT expiry timeout will still stop the stream.
    pub fn cancel(&self) {
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
/// let stream_channel = StreamChannel::new(
///     zmq_context.clone(),
///     signing_key.clone(),
/// );
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
    /// ZMQ context (shared)
    context: Arc<zmq::Context>,
    /// Signing key for stream registration
    signing_key: SigningKey,
    /// Lazy-initialized async PUSH socket to StreamService (wrapped in Arc for sharing)
    push_socket: OnceCell<Arc<tokio::sync::Mutex<tmq::push::Push>>>,
    /// Shared SUB socket for control channel messages (one FD for all streams)
    ctrl_sub: OnceCell<Arc<tokio::sync::Mutex<tmq::subscribe::Subscribe>>>,
    /// Active cancel tokens keyed by ctrl_topic
    cancel_tokens: Arc<DashMap<String, CancellationToken>>,
}

impl StreamChannel {
    /// Create a new stream channel.
    pub fn new(context: Arc<zmq::Context>, signing_key: SigningKey) -> Self {
        Self {
            context,
            signing_key,
            push_socket: OnceCell::new(),
            ctrl_sub: OnceCell::new(),
            cancel_tokens: Arc::new(DashMap::new()),
        }
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
        claims: Option<Claims>,
    ) -> Result<StreamContext> {
        // 1. DH key exchange
        let stream_ctx = StreamContext::from_dh(client_ephemeral_pubkey)?;

        // 2. Pre-authorize data + control topics with StreamService
        let exp = chrono::Utc::now().timestamp() + expiry_secs;
        self.pre_authorize(&stream_ctx, exp, claims.clone()).await?;
        self.register_topic(stream_ctx.ctrl_topic(), exp, claims).await?;

        // 3. Subscribe control channel and register cancel token
        let ctrl_sub = self.get_or_init_ctrl_sub().await?;
        {
            let mut sub = ctrl_sub.lock().await;
            sub.subscribe(stream_ctx.ctrl_topic().as_bytes())
                .map_err(|e| anyhow::anyhow!("Failed to subscribe ctrl topic: {}", e))?;
        }
        self.cancel_tokens.insert(
            stream_ctx.ctrl_topic().to_owned(),
            stream_ctx.cancel_token().clone(),
        );

        // 4. Spawn JWT expiry timeout as universal backstop
        let token = stream_ctx.cancel_token().clone();
        let ctrl_topic = stream_ctx.ctrl_topic().to_owned();
        let tokens_map = Arc::clone(&self.cancel_tokens);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(expiry_secs.max(0) as u64)).await;
            token.cancel();
            tokens_map.remove(&ctrl_topic);
        });

        Ok(stream_ctx)
    }

    /// Pre-authorize a stream with StreamService.
    ///
    /// Sends a signed StreamRegister message to StreamService so the topic
    /// is ready for subscriptions before the client tries to subscribe.
    async fn pre_authorize(&self, ctx: &StreamContext, expiry: i64, claims: Option<Claims>) -> Result<()> {
        self.register_topic(ctx.topic(), expiry, claims).await?;
        tracing::debug!(
            stream_id = %ctx.stream_id(),
            topic = %ctx.topic(),
            expiry = expiry,
            "Stream pre-authorized with StreamService"
        );
        Ok(())
    }

    /// Register a topic with StreamService.
    ///
    /// This is a low-level method that sends a StreamRegister message.
    /// For DH-based streams, use `prepare_stream()` instead.
    ///
    /// Use cases:
    /// - Re-authorizing an existing stream by ID
    /// - Legacy stream authorization
    pub async fn register_topic(&self, topic: &str, expiry: i64, claims: Option<Claims>) -> Result<()> {
        let register_msg = build_stream_register_envelope(
            topic,
            expiry,
            &self.signing_key,
            claims,
        );

        let socket_arc = self.get_or_init_socket().await?;
        let mut socket = socket_arc.lock().await;

        let multipart = tmq::Multipart::from(vec![register_msg]);
        socket.send(multipart).await
            .map_err(|e| anyhow::anyhow!("Failed to send stream registration: {}", e))?;

        Ok(())
    }

    /// Get or initialize the async PUSH socket to StreamService.
    ///
    /// Returns an Arc to the socket mutex, initializing it on first call.
    async fn get_or_init_socket(&self) -> Result<Arc<tokio::sync::Mutex<tmq::push::Push>>> {
        let socket_arc = self.push_socket.get_or_try_init(|| async {
            // Connect to StreamService's PUSH endpoint
            let endpoint = endpoint_registry()
                .endpoint("streams", SocketKind::Push)
                .to_zmq_string();

            let socket = tmq::push::push(&self.context)
                .set_sndtimeo(1000)
                .connect(&endpoint)
                .map_err(|e| anyhow::anyhow!("Failed to connect to StreamService: {}", e))?;

            tracing::debug!(endpoint = %endpoint, "StreamChannel connected to StreamService (async)");

            Ok::<_, anyhow::Error>(Arc::new(tokio::sync::Mutex::new(socket)))
        }).await?;
        Ok(Arc::clone(socket_arc))
    }

    /// Get or initialize the shared control channel SUB socket.
    async fn get_or_init_ctrl_sub(&self) -> Result<Arc<tokio::sync::Mutex<tmq::subscribe::Subscribe>>> {
        let sub_arc = self.ctrl_sub.get_or_try_init(|| async {
            let endpoint = endpoint_registry()
                .endpoint("streams", SocketKind::Sub)
                .to_zmq_string();

            let without_topic = tmq::subscribe::subscribe(&self.context)
                .connect(&endpoint)
                .map_err(|e| anyhow::anyhow!("Failed to connect ctrl SUB: {}", e))?;

            // Subscribe to a NUL-prefixed topic that will never match real hex topics
            let sub = without_topic.subscribe(b"\x00__ctrl_init__")
                .map_err(|e| anyhow::anyhow!("Failed to init ctrl SUB: {}", e))?;

            let sub = Arc::new(tokio::sync::Mutex::new(sub));

            // Spawn the control listener task
            self.spawn_ctrl_listener(Arc::clone(&sub));

            Ok::<_, anyhow::Error>(sub)
        }).await?;
        Ok(Arc::clone(sub_arc))
    }

    /// Spawn a background task that listens for control messages and fires cancel tokens.
    fn spawn_ctrl_listener(&self, sub: Arc<tokio::sync::Mutex<tmq::subscribe::Subscribe>>) {
        use futures::StreamExt;

        let tokens = Arc::clone(&self.cancel_tokens);
        tokio::spawn(async move {
            loop {
                // Poll for the next message, releasing the mutex between
                // iterations so other tasks (e.g. prepare_stream_with_claims)
                // can subscribe new topics.  We use poll_next via a short
                // select! with a sleep to avoid holding the lock across a
                // long-lived .next().await which would deadlock any caller
                // of ctrl_sub.lock().
                let msg = {
                    let mut guard = sub.lock().await;
                    // Try to get one message with a brief timeout.
                    // If nothing arrives, release the lock and sleep
                    // before retrying so other tasks can acquire it.
                    let result = tokio::time::timeout(
                        std::time::Duration::from_millis(50),
                        guard.next(),
                    ).await;
                    match result {
                        Ok(msg) => msg,
                        Err(_) => {
                            // Timeout — no message, release lock and retry
                            drop(guard);
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            continue;
                        }
                    }
                };
                let multipart = match msg {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => {
                        tracing::warn!("ctrl SUB error: {}", e);
                        continue;
                    }
                    None => break, // socket closed
                };

                // Wire format: [ctrl_topic, capnp, mac]
                if multipart.is_empty() {
                    continue;
                }
                let topic = String::from_utf8_lossy(&multipart[0]);

                // Fire the cancel token if we have one for this topic
                if let Some((_, token)) = tokens.remove(topic.as_ref()) {
                    tracing::debug!(ctrl_topic = %topic, "control cancel received");
                    token.cancel();
                }
            }
        });
    }

    /// Create a publisher for the given stream context.
    ///
    /// The publisher owns a reference to the socket and can be used in async contexts
    /// without lifetime concerns.
    ///
    /// # Arguments
    /// * `ctx` - Stream context from `prepare_stream()`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut publisher = stream_channel.publisher(&stream_ctx).await?;
    /// publisher.publish_progress("cloning", 0, 1).await?;
    /// // ... perform operation ...
    /// publisher.complete(&result_bytes).await?;
    /// ```
    pub async fn publisher(&self, ctx: &StreamContext) -> Result<StreamPublisher> {
        let socket_arc = self.get_or_init_socket().await?;
        Ok(StreamPublisher::new(socket_arc, ctx))
    }

    /// Execute streaming work with a publisher (convenience wrapper).
    ///
    /// This is a convenience method that creates a publisher and passes it to the callback.
    /// The callback receives the publisher by value, so it can be used in async blocks.
    ///
    /// # Arguments
    /// * `ctx` - Stream context from `prepare_stream()`
    /// * `f` - Async callback that receives the publisher
    ///
    /// # Example
    ///
    /// ```ignore
    /// stream_channel.with_publisher(&stream_ctx, |mut publisher| async move {
    ///     publisher.publish_progress("cloning", 0, 1).await?;
    ///     // ... perform operation ...
    ///     publisher.complete(&result_bytes).await?;
    ///     Ok(())
    /// }).await?;
    /// ```
    pub async fn with_publisher<F, Fut, R>(&self, ctx: &StreamContext, f: F) -> Result<R>
    where
        F: FnOnce(StreamPublisher) -> Fut,
        Fut: Future<Output = Result<R>>,
    {
        let publisher = self.publisher(ctx).await?;
        f(publisher).await
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
// Helpers
// ============================================================================

/// Build a StreamRegister message wrapped in SignedEnvelope.
///
/// Used by `StreamChannel::pre_authorize()` to register streams with StreamService.
fn build_stream_register_envelope(
    topic: &str,
    expiry: i64,
    signing_key: &SigningKey,
    claims: Option<Claims>,
) -> Vec<u8> {
    use crate::common_capnp;
    use crate::envelope::{RequestEnvelope, RequestIdentity, SignedEnvelope};
    use crate::ToCapnp;

    // Build StreamRegister message
    let mut inner_msg = Builder::new_default();
    {
        let mut register = inner_msg.init_root::<crate::streaming_capnp::stream_register::Builder>();
        register.set_topic(topic);
        register.set_exp(expiry);
    }

    let mut inner_bytes = Vec::new();
    // Vec write cannot fail for memory, and the capnp message is well-formed
    if let Err(e) = serialize::write_message(&mut inner_bytes, &inner_msg) {
        tracing::error!("Failed to serialize StreamRegister: {e}");
        return Vec::new();
    }

    // Wrap in SignedEnvelope
    let mut envelope = RequestEnvelope::new(RequestIdentity::local(), inner_bytes);
    if let Some(c) = claims {
        envelope = envelope.with_claims(c);
    }

    let signed = SignedEnvelope::new_signed(envelope, signing_key);

    let mut msg = Builder::new_default();
    {
        let mut builder = msg.init_root::<common_capnp::signed_envelope::Builder>();
        signed.write_to(&mut builder);
    }

    let mut bytes = Vec::new();
    // Vec write cannot fail for memory, and the capnp message is well-formed
    if let Err(e) = serialize::write_message(&mut bytes, &msg) {
        tracing::error!("Failed to serialize SignedEnvelope: {e}");
        return Vec::new();
    }
    bytes
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

/// Constant-time byte slice comparison.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Chain state should update
        assert_eq!(state.prev_mac, Some(mac2));
    }
}
