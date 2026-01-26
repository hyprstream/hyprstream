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

use anyhow::Result;
use capnp::message::Builder;
use capnp::serialize;
use serde::{Deserialize, Serialize};

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
    /// Send frames via ZMQ socket.
    pub fn send(&self, socket: &zmq::Socket) -> Result<()> {
        socket.send(&self.topic, zmq::SNDMORE)?;
        socket.send(&self.capnp, zmq::SNDMORE)?;
        socket.send(self.mac.as_slice(), 0)?;
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
        let min = self.config.min_batch_size as f32;
        let max = self.config.max_batch_size as f32;

        let log_min_rate = self.config.min_rate.max(1.0).ln();
        let log_max_rate = self.config.max_rate.max(1.0).ln();

        let log_rate = rate.max(1.0).ln();
        let t = ((log_rate - log_min_rate) / (log_max_rate - log_min_rate)).clamp(0.0, 1.0);

        (min + t * (max - min)).round() as usize
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
        self.pending.push(StreamPayloadData::Error(message.to_string()));
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

            let mut list = block.init_payloads(payloads.len() as u32);
            for (i, payload) in payloads.iter().enumerate() {
                let mut p = list.reborrow().get(i as u32);
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
/// while let Some(payload) = handle.next()? {
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

        Ok(Self {
            subscriber,
            stream_id,
            topic: keys.topic,
            verifier,
            pending: VecDeque::new(),
            completed: false,
        })
    }

    /// Receive next payload (blocking).
    ///
    /// Returns `None` when stream is complete.
    pub fn next(&mut self) -> Result<Option<StreamPayload>> {
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
}

// ============================================================================
// Helpers
// ============================================================================

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
        let builder = StreamBuilder::new(config, [0u8; 32], "topic".to_string());

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
        let topic = "test_topic".to_string();

        let mut state = StreamHmacState::new(key, topic);

        let mac1 = state.compute_next(b"data1");
        let mac2 = state.compute_next(b"data2");

        // MACs should be different
        assert_ne!(mac1, mac2);

        // Chain state should update
        assert_eq!(state.prev_mac, Some(mac2));
    }
}
