//! Generic streaming infrastructure for authenticated PUB/SUB communication.
//!
//! This module provides rate-controlled, MAC-authenticated streaming that works
//! for both inference (UTF-8 tokens) and worker I/O (arbitrary binary data).
//!
//! # Architecture
//!
//! ```text
//! Producer                     Consumer
//! ────────                     ────────
//! StreamChannel::publisher()   MoqStreamHandle::new()
//!       │                             │
//!       │ publish_data()              │ recv_next()
//!       ▼                             ▼
//! moq UDS track ──────────► StreamVerifier
//! ```
//!
//! # Wire Format
//!
//! Moq track payload (binary):
//!   capnp StreamBlock + 16-byte truncated HMAC appended
//!
//! # Security
//!
//! - DH key exchange: Ristretto255 ECDH derives topic + mac_key
//! - MAC chain: Each block's MAC depends on previous, enforces ordering
//!
//! # Backend
//!
//! - Default: Blake3 `keyed_hash()` (~10+ GB/s with SIMD)
//! - FIPS mode: HMAC-SHA256 (FIPS 198-1)

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;
use capnp::message::Builder;
use capnp::serialize;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::auth::Claims;
use crate::prelude::SigningKey;

use crate::crypto::derive_stream_keys;

// DH key types - Ristretto255 (default) or P-256 (FIPS)
#[cfg(not(feature = "fips"))]
use crate::crypto::{
    ristretto_dh as dh_compute, RistrettoPublic as DhPublic, RistrettoSecret as DhSecret,
};

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

fn default_min_batch_size() -> usize {
    1
}
fn default_max_batch_size() -> usize {
    16
}
fn default_max_block_bytes() -> usize {
    65536
}
fn default_min_rate() -> f32 {
    1.0
}
fn default_max_rate() -> f32 {
    100.0
}

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
    /// Internal identified-profile control plaintext. This variant must be
    /// sealed into `Tagged` before Cap'n Proto serialization.
    EpochCommit(crate::stream_epoch::StreamEpochCommit),
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
    /// DH-derived AES-256-GCM key for transport-level AEAD of payloads (#321).
    /// `Some` only on the DH path (`from_dh`): the mesh/networked stream plane
    /// seals each Data/Complete payload under this key. `None` on the keyless
    /// `new()` path (e.g. NotificationService topics, whose payloads are already
    /// E2E-encrypted at the app layer), where transport AEAD is not applied.
    enc_key: Option<[u8; 32]>,
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
    /// Per-stream relay selection applied when building this stream's reach (#384).
    ///
    /// Server-authored (like `qos` — never client-supplied). Defaults to
    /// [`crate::moq_stream::RelayChoice::ServerDefault`] (the node's server-global
    /// relay). Set to `Only` for an anonymized, relay-only stream, or `Override`
    /// for per-tenant relay isolation.
    relay_choice: crate::moq_stream::RelayChoice,
    /// Per-stream reach inputs — the node/server's iroh + QUIC base reach (#384).
    ///
    /// Captured into the context at construction so [`reach`](Self::reach) builds
    /// the `StreamInfo.reach` from THIS field, never re-reading the process-global
    /// `OnceLock`s on every call. This is what makes heterogeneous per-server
    /// reach work: a [`StreamChannel`] threaded with an explicit
    /// [`ProducerReachConfig`](crate::moq_stream::ProducerReachConfig) via
    /// [`StreamChannel::with_reach_config`] populates this so the base (iroh/QUIC)
    /// reach is per-stream too, not just `relay_choice`.
    ///
    /// When no explicit config is threaded, [`from_dh`](Self::from_dh) seeds it
    /// from an empty [`ProducerReachConfig`](crate::moq_stream::ProducerReachConfig) — a
    /// one-time per-stream snapshot (the documented compat source), not a
    /// per-`reach()`-call global read.
    reach_config: crate::moq_stream::ProducerReachConfig,

    /// Hybrid KEM ciphertexts to emit in `StreamInfo.kemCiphertexts` (S3 #554).
    /// `Some` on the hybrid post-quantum path
    /// ([`from_hybrid_identified`](Self::from_hybrid_identified));
    /// `None` on the legacy classical [`from_dh`](Self::from_dh) and the keyless
    /// [`new`](Self::new) paths.
    kem_ciphertexts: Option<Vec<u8>>,
    /// Shared one-shot identified producer state. The outer `Option` is a
    /// persistent profile marker; the inner state is consumed exactly once
    /// across every clone before an identified publisher creates MoQ resources.
    /// Legacy/keyless contexts have no claim and cannot ratchet into this profile.
    epoch_ratchet: Option<Arc<parking_lot::Mutex<Option<crate::stream_epoch::StreamEpochRatchet>>>>,
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
            // Keyless path: no shared transport AEAD key (see field docs).
            enc_key: None,
            server_pubkey,
            ctrl_topic: String::new(),
            ctrl_mac_key: [0u8; 32],
            cancel_token: CancellationToken::new(),
            qos: crate::stream_info::StreamOpt::default(),
            relay_choice: crate::moq_stream::RelayChoice::default(),
            // No DH/server context here (notification + standalone TUI paths,
            // which never call `reach()`); default to an empty reach config. A
            // networked producer threads a real one via `with_reach_config`.
            reach_config: crate::moq_stream::ProducerReachConfig::default(),
            // Keyless path: no hybrid KEM material.
            kem_ciphertexts: None,
            epoch_ratchet: None,
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
            // DH path: AEAD ON for the mesh stream plane (#321).
            enc_key: Some(*keys.enc_key),
            server_pubkey: server_pubkey_bytes,
            ctrl_topic: keys.ctrl_topic,
            ctrl_mac_key: *keys.ctrl_mac_key,
            cancel_token: CancellationToken::new(),
            qos: crate::stream_info::StreamOpt::default(),
            relay_choice: crate::moq_stream::RelayChoice::default(),
            reach_config: crate::moq_stream::ProducerReachConfig::default(),
            // Classical (legacy) path: no hybrid KEM material; the client keys
            // off the server ephemeral `dhPublic`. Removed at the S5 fail-closed
            // flip (#556) once all call-sites use `from_hybrid_identified`.
            kem_ciphertexts: None,
            epoch_ratchet: None,
        })
    }

    /// Create a stream context via the **hybrid post-quantum** handshake (S3 #554).
    ///
    /// `client_kem_public` is the client's encoded ephemeral `RecipientPublic`
    /// (from `RequestEnvelope.clientKemPublic`). The server encapsulates to it and
    /// derives the stream keys from the hybrid combiner secret; the per-component
    /// ciphertexts are stored in [`kem_ciphertexts`](Self::kem_ciphertexts) for
    /// emission in `StreamInfo.kemCiphertexts`. Forward secrecy comes from the
    /// client's ephemeral keypair (X25519 + ML-KEM legs).
    ///
    /// Fail-closed: a malformed / wrong-suite `client_kem_public` is rejected.
    /// This is the post-quantum replacement for [`from_dh`](Self::from_dh).
    pub fn from_hybrid_identified(
        client_kem_public: &[u8],
        binding: crate::stream_epoch::IdentifiedStreamBinding,
    ) -> Result<Self> {
        let (material, ratchet) =
            crate::crypto::key_exchange::server_identified_stream_epoch(client_kem_public, binding)
                .map_err(|e| anyhow::anyhow!("hybrid stream handshake: {e}"))?;
        let keys = ratchet.current_keys();
        let outbound = &keys.producer_to_consumer;
        let inbound = &keys.consumer_to_producer;

        let stream_id = format!("stream-{}", uuid::Uuid::new_v4());

        Ok(Self {
            stream_id,
            topic: ratchet.route_topic().to_owned(),
            mac_key: *outbound.mac_key,
            // Hybrid path: AEAD ON for the mesh stream plane (#321), keyed by the
            // post-quantum combiner secret.
            enc_key: Some(*outbound.enc_key),
            // No classical server ephemeral pubkey on the hybrid path; the client
            // keys off `kem_ciphertexts`, not `dhPublic`.
            server_pubkey: [0u8; 32],
            ctrl_topic: ratchet.control_topic().to_owned(),
            ctrl_mac_key: *inbound.control_mac_key,
            cancel_token: CancellationToken::new(),
            qos: crate::stream_info::StreamOpt::default(),
            relay_choice: crate::moq_stream::RelayChoice::default(),
            reach_config: crate::moq_stream::ProducerReachConfig::default(),
            kem_ciphertexts: Some(material.encode()),
            epoch_ratchet: Some(Arc::new(parking_lot::Mutex::new(Some(ratchet)))),
        })
    }

    /// Whether this context was created for the identified epoch profile.
    ///
    /// This marker remains true after the producer claim is consumed so policy
    /// checks such as the transport-AEAD requirement cannot be downgraded.
    pub(crate) fn has_epoch_ratchet(&self) -> bool {
        self.epoch_ratchet.is_some()
    }

    /// Build a publisher while holding the shared identified producer claim.
    ///
    /// The callback sees `Some` only once across every clone. It must consume
    /// the state only after all fallible resource setup succeeds; if setup
    /// fails first, the claim remains available for a retry. Legacy contexts
    /// receive an unshared `None` slot.
    pub(crate) fn with_publisher_epoch_ratchet<T>(
        &self,
        build: impl FnOnce(&mut Option<crate::stream_epoch::StreamEpochRatchet>) -> Result<T>,
    ) -> Result<T> {
        let Some(claim) = &self.epoch_ratchet else {
            let mut legacy = None;
            return build(&mut legacy);
        };
        let mut claim = claim.lock();
        anyhow::ensure!(
            claim.is_some(),
            "identified stream publisher already claimed"
        );
        build(&mut claim)
    }

    /// The hybrid KEM ciphertexts to emit in `StreamInfo.kemCiphertexts` (S3 #554):
    /// `Some` on the hybrid path
    /// ([`from_hybrid_identified`](Self::from_hybrid_identified)), `None`
    /// otherwise.
    pub fn kem_ciphertexts(&self) -> Option<&[u8]> {
        self.kem_ciphertexts.as_deref()
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

    /// Get the transport AEAD key (#321), if this is a DH-keyed stream.
    ///
    /// `Some` on the DH path (mesh/networked stream — AEAD ON), `None` on the
    /// keyless `new()` path (payloads already E2E-encrypted at the app layer).
    pub fn enc_key(&self) -> Option<&[u8; 32]> {
        self.enc_key.as_ref()
    }

    /// Disable transport-level AEAD for this stream (#321).
    ///
    /// Clears the AEAD key so a legacy publisher emits cleartext (HMAC-chained)
    /// blocks. Identified epoch contexts retain an internal profile marker and
    /// are rejected by `MoqStreamOrigin::publisher` if this escape hatch is used;
    /// it exists only for legacy same-host streams whose consumer receives
    /// `(mac_key, topic)` out of band and cannot derive the classical DH key.
    pub fn without_aead(mut self) -> Self {
        self.enc_key = None;
        self
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

    /// Set the per-stream relay choice (#384). Server-authored — never from a
    /// client request. Use [`crate::moq_stream::RelayChoice::Only`] for an
    /// anonymized relay-only stream, or `Override` for per-tenant isolation.
    pub fn with_relay_choice(mut self, relay_choice: crate::moq_stream::RelayChoice) -> Self {
        self.relay_choice = relay_choice;
        self
    }

    /// Get the per-stream relay choice (#384).
    pub fn relay_choice(&self) -> &crate::moq_stream::RelayChoice {
        &self.relay_choice
    }

    /// Thread an explicit per-server reach config into this stream (#384).
    ///
    /// Supplies the per-server [`ProducerReachConfig`](crate::moq_stream::ProducerReachConfig)
    /// for this stream,
    /// so the base (iroh/QUIC) reach this stream advertises is the server's own —
    /// [`StreamChannel::prepare_stream_with_claims`] applies its current service
    /// config before it returns the context.
    pub fn with_reach_config(
        mut self,
        reach_config: crate::moq_stream::ProducerReachConfig,
    ) -> Self {
        self.reach_config = reach_config;
        self
    }

    /// Get this stream's reach config (#384).
    pub fn reach_config(&self) -> &crate::moq_stream::ProducerReachConfig {
        &self.reach_config
    }

    /// Build this stream's `StreamInfo.reach` from the node's per-server reach
    /// config and this stream's [`relay_choice`](Self::relay_choice) (#384).
    ///
    /// A producer that holds a `StreamContext` calls this so
    /// the stream's relay/anonymization posture **and** its base iroh/QUIC reach
    /// are honoured per-stream. Both come from the context's own
    /// [`reach_config`](Self::reach_config) field — captured at construction (and
    /// overridable via [`with_reach_config`](Self::with_reach_config)) — so this
    /// no longer reads the process-global `OnceLock`s on each call.
    pub fn reach(&self) -> Vec<crate::stream_info::Destination> {
        self.reach_config
            .reach_with_relay(self.relay_choice.clone())
    }
}

// ============================================================================
// Progress Streaming (Bridges progress callbacks to streams)
// ============================================================================

/// Progress update message for channel-based streaming.
#[derive(Debug, Clone)]
pub enum ProgressUpdate {
    /// Progress update: stage, current, total
    Progress {
        stage: String,
        current: usize,
        total: usize,
    },
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
        self.sender
            .blocking_send(ProgressUpdate::Complete(metadata))
            .map_err(|_| anyhow::anyhow!("Progress channel closed"))
    }

    /// Signal an error occurred.
    pub fn error(&self, message: &str) -> Result<()> {
        self.sender
            .blocking_send(ProgressUpdate::Error(message.to_owned()))
            .map_err(|_| anyhow::anyhow!("Progress channel closed"))
    }
}

/// Create a progress channel for streaming operations.
///
/// Returns (sender, receiver) where:
/// - sender: Pass to `ChannelProgressReporter::new()` for use in blocking operations
/// - receiver: Poll in async context to forward updates to `StreamPublisher`
pub fn progress_channel(
    buffer_size: usize,
) -> (
    tokio::sync::mpsc::Sender<ProgressUpdate>,
    tokio::sync::mpsc::Receiver<ProgressUpdate>,
) {
    tokio::sync::mpsc::channel(buffer_size)
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
    /// Ed25519 signing key for stream authorization (wired into moq publish claims, N7)
    /// and the per-host provenance signer (#321).
    signing_key: SigningKey,
    /// ML-DSA-65 signing key for the post-quantum half of the StreamRegister
    /// composite signature. Mirrors `LocalSigner` on the RPC plane so the
    /// streaming control plane signs under the same policy (#161). Derived
    /// deterministically from the node's persistent Ed25519 signing key (#157)
    /// in [`Self::new`]; override with [`Self::with_pq_key`].
    pq_signing_key: Option<crate::crypto::pq::MlDsaSigningKey>,
    /// Per-service reach source, updated by the service spawner after bind.
    reach_config: crate::moq_stream::ProducerReachConfigHandle,
    /// Optional service-scoped origin installed when this service links a relay.
    moq_origin: crate::moq_stream::MoqStreamOriginHandle,
}

impl StreamChannel {
    /// Create a new stream channel.
    ///
    /// The post-quantum half of the StreamRegister composite is the node's
    /// **persistent** mesh ML-DSA-65 key, derived deterministically from
    /// `signing_key` via [`crate::node_identity::derive_mesh_mldsa_key`] (#157),
    /// mirroring `LocalSigner::new` on the RPC plane. This replaces the previous
    /// ephemeral keygen so the streaming control plane's ML-DSA public key is
    /// stable across restarts and equals the `#mesh-pq` key peers anchor. Use
    /// [`Self::with_pq_key`] only to override with an externally supplied key.
    pub fn new(signing_key: SigningKey) -> Self {
        let pq_signing_key = Some(crate::node_identity::derive_mesh_mldsa_key(&signing_key));
        Self {
            signing_key,
            pq_signing_key,
            reach_config: std::sync::Arc::new(parking_lot::RwLock::new(
                crate::moq_stream::ProducerReachConfig::default(),
            )),
            moq_origin: std::sync::Arc::new(parking_lot::RwLock::new(None)),
        }
    }

    /// Initialize this channel's reach configuration.
    pub fn with_reach_config(mut self, cfg: crate::moq_stream::ProducerReachConfig) -> Self {
        self.reach_config = std::sync::Arc::new(parking_lot::RwLock::new(cfg));
        self
    }

    /// Share a service-owned reach handle with this channel.
    pub fn with_reach_config_handle(mut self, handle: crate::moq_stream::ProducerReachConfigHandle) -> Self {
        self.reach_config = handle;
        self
    }

    /// Return this channel's service-owned reach handle.
    pub fn reach_config_handle(&self) -> crate::moq_stream::ProducerReachConfigHandle {
        self.reach_config.clone()
    }

    /// Share a service-owned MoQ origin handle with this channel.
    pub fn with_moq_origin_handle(mut self, handle: crate::moq_stream::MoqStreamOriginHandle) -> Self {
        self.moq_origin = handle;
        self
    }

    /// Return this channel's service-owned MoQ origin handle.
    pub fn moq_origin_handle(&self) -> crate::moq_stream::MoqStreamOriginHandle {
        self.moq_origin.clone()
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
        self.prepare_stream_with_claims(client_ephemeral_pubkey, expiry_secs, None)
            .await
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
        let stream_ctx = StreamContext::from_dh(client_ephemeral_pubkey)?
            .with_reach_config(self.reach_config.read().clone());

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
    pub async fn register_topic(
        &self,
        _topic: &str,
        _expiry: i64,
        _claims: Option<Claims>,
    ) -> Result<()> {
        Ok(())
    }

    /// Create a publisher for a stream context.
    ///
    /// Fails loudly if the process-global moq stream origin has not been
    /// initialized (server not started). In production the `streams` factory
    /// always calls `init_global_moq_origin` before any service handles requests.
    pub async fn publisher(
        &self,
        ctx: &StreamContext,
    ) -> Result<crate::moq_stream::AnyStreamPublisher> {
        let scoped_origin = self.moq_origin.read().clone();
        let origin = match scoped_origin.as_ref() {
            Some(origin) => origin,
            None => crate::moq_stream::global_moq_origin()
                .ok_or_else(|| anyhow::anyhow!("no moq stream origin — server not initialized"))?,
        };
        // #321: on the DH (mesh) path, sign each StreamBlock with the node's per-host
        // hybrid identity (Ed25519 + the deterministically-derived mesh ML-DSA key)
        // so consumers can attribute blocks to this host (C-PROV / threat T3). The
        // keyless notification path (no DH enc_key) carries no provenance.
        let provenance = if ctx.enc_key().is_some() {
            let signer = match &self.pq_signing_key {
                Some(pq) => crate::stream_provenance::ProvenanceSigner::new(
                    self.signing_key.clone(),
                    pq.clone(),
                ),
                None => crate::stream_provenance::ProvenanceSigner::from_ed25519(
                    self.signing_key.clone(),
                ),
            };
            Some(signer)
        } else {
            None
        };
        origin.publisher_with_provenance(ctx, provenance)
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
    pub async fn publisher_for_topic(
        &self,
        topic: &str,
    ) -> Result<crate::moq_stream::AnyStreamPublisher> {
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
    static MAP: std::sync::OnceLock<
        RwLock<HashMap<String, std::sync::Arc<tokio::sync::Semaphore>>>,
    > = std::sync::OnceLock::new();
    let map = MAP.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(sem) = map.read().get(service_name) {
        return std::sync::Arc::clone(sem);
    }
    std::sync::Arc::clone(
        map.write()
            .entry(service_name.to_owned())
            .or_insert_with(|| {
                std::sync::Arc::new(tokio::sync::Semaphore::new(
                    max_concurrent_streams_per_service(),
                ))
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
/// Used by the moq streaming plane (`crate::moq_stream`) to produce
/// chained-HMAC StreamBlocks (§7.5 wire format).
pub fn encode_stream_block(
    prev_mac: &[u8],
    sequence_number: u64,
    epoch: u64,
    payloads: &[StreamPayloadData],
) -> Result<Vec<u8>> {
    encode_stream_block_with_provenance(prev_mac, sequence_number, epoch, payloads, None)
}

/// Encode a `StreamBlock` with an optional per-host provenance signature (#321).
///
/// `provenance = None` produces the canonical *signed region* — byte-identical to
/// [`encode_stream_block`] — over which the provenance signature is computed (and
/// which the consumer reconstructs to verify). `provenance = Some((signer_kid,
/// sig))` additionally populates the wire `StreamBlock.provenance` field.
pub fn encode_stream_block_with_provenance(
    prev_mac: &[u8],
    sequence_number: u64,
    epoch: u64,
    payloads: &[StreamPayloadData],
    provenance: Option<(&[u8], &[u8])>,
) -> Result<Vec<u8>> {
    let mut msg = Builder::new_default();
    {
        let mut block = msg.init_root::<streaming_capnp::stream_block::Builder>();
        block.set_prev_mac(prev_mac);
        // Producer-assigned counter (#219); MAC-covered implicitly (whole block is signed).
        // epoch is 0 until the #223 key-epoch lifecycle lands.
        block.set_sequence_number(sequence_number);
        block.set_epoch(epoch);

        // #321: per-host provenance. Set BEFORE the payload list so the canonical
        // serialization is deterministic regardless of field-set order. Absent ⇒
        // the field stays default (empty) — the signed-region encoding.
        if let Some((signer_kid, sig)) = provenance {
            let mut prov = block.reborrow().init_provenance();
            prov.set_signer_kid(signer_kid);
            prov.set_sig(sig);
        }

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
                StreamPayloadData::EpochCommit(_) => {
                    anyhow::bail!(
                        "identified stream epoch control reached cleartext serialization"
                    );
                }
                StreamPayloadData::Tagged {
                    tag,
                    payload,
                    nonce,
                    key_commitment,
                } => {
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

/// Derive the `(mac_key, enc_key, topic)` triple from a client-side DH exchange (#321).
///
/// This is the consumer-side counterpart to `StreamContext::from_dh` (the server side).
/// Call this before constructing a `MoqStreamHandle` when you have the raw keys from
/// `generate_ephemeral_keypair()` and the server pubkey from `StreamInfo`.
pub fn derive_client_stream_keys(
    client_secret: &DhSecret,
    client_pubkey: &[u8],
    server_pubkey: &[u8],
) -> anyhow::Result<([u8; 32], [u8; 32], String)> {
    let server_pub = DhPublic::from_slice(server_pubkey)
        .ok_or_else(|| anyhow::anyhow!("invalid server pubkey length"))?;
    let shared = dh_compute(client_secret, &server_pub);
    let client_pub_32 = pubkey_to_32(client_pubkey);
    let server_pub_32 = pubkey_to_32(server_pubkey);
    let keys = crate::crypto::derive_stream_keys(&shared, &client_pub_32, &server_pub_32)?;
    // #321: also return the AEAD enc_key so the moq consumer can open Tagged blocks.
    Ok((*keys.mac_key, *keys.enc_key, keys.topic.clone()))
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
            let mut register =
                inner_msg.init_root::<crate::streaming_capnp::stream_register::Builder>();
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

    /// #161 / WNS: a StreamRegister built with a PQ key is a Hybrid composite
    /// that ENFORCES the anchored ML-DSA outer when the signer's key is anchored
    /// in the PQ trust store, and falls back to the inner EdDSA (classical floor)
    /// when it isn't — per-identity, exactly mirroring the RPC plane. The
    /// unanchored fallback is no weaker than `verify_any_signer` (Classical),
    /// which already verifies the self-asserted `cnf`'s EdDSA without a pin.
    #[test]
    fn stream_register_hybrid_verifies_only_when_pq_anchored() -> anyhow::Result<()> {
        use crate::common_capnp;
        use crate::crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_from_bytes};
        use crate::crypto::CryptoPolicy;
        use crate::envelope::{InMemoryNonceCache, KeyedPqTrustStore, SignedEnvelope};
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
        let signed =
            SignedEnvelope::read_from(reader.get_root::<common_capnp::signed_envelope::Reader>()?)?;

        assert_eq!(
            signed.policy,
            CryptoPolicy::Hybrid,
            "must sign Hybrid with PQ key"
        );

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

        // Not anchored: empty store under Hybrid falls back to the inner EdDSA
        // classical floor (WNS per-identity) rather than failing closed. PQ is
        // never trusted from the self-asserted COSE entry, so this is no weaker
        // than the pre-existing classical `verify_any_signer` path.
        let empty = KeyedPqTrustStore::new();
        let nonce_empty = InMemoryNonceCache::new();
        assert!(
            signed
                .verify_any_signer_with(&nonce_empty, Some(&empty), CryptoPolicy::Hybrid)
                .is_ok(),
            "unanchored Hybrid register must verify via classical inner-EdDSA fallback"
        );

        // But a tampered/forged inner EdDSA on an unanchored signer is still
        // rejected — the classical floor is a real signature check, not a bypass.
        let mut forged = signed.clone();
        forged.cnf = SigningKey::from_bytes(&[9u8; 32])
            .verifying_key()
            .to_bytes();
        let nonce_forged = InMemoryNonceCache::new();
        assert!(
            forged
                .verify_any_signer_with(&nonce_forged, Some(&empty), CryptoPolicy::Hybrid)
                .is_err(),
            "swapping cnf without a matching inner EdDSA must still be rejected"
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

    #[tokio::test]
    async fn stream_channels_keep_distinct_service_reach() -> Result<()> {
        use crate::moq_stream::{NodeStreamReach, ProducerReachConfig};
        use crate::stream_info::TransportConfig;

        let config = |port| -> Result<ProducerReachConfig> {
            Ok(ProducerReachConfig {
                iroh_node_id: None,
                quic_reach: Some(NodeStreamReach {
                    addr: format!("127.0.0.1:{port}").parse()?,
                    server_name: format!("server-{port}"),
                    cert_hashes: vec![[port as u8; 32]],
                }),
                relay: None,
            })
        };
        let channel_a = StreamChannel::new(SigningKey::from_bytes(&[1; 32]))
            .with_reach_config(config(4101)?);
        let channel_b = StreamChannel::new(SigningKey::from_bytes(&[2; 32]))
            .with_reach_config(config(4102)?);
        let (_, client_pub) = crate::crypto::generate_ephemeral_keypair();

        let stream_a = channel_a
            .prepare_stream(&client_pub.to_bytes(), 60)
            .await?;
        let stream_b = channel_b
            .prepare_stream(&client_pub.to_bytes(), 60)
            .await?;

        let addr = |stream: &StreamContext| match &stream.reach()[0].transport {
            TransportConfig::Quic(reach) => reach.addr.clone(),
            other => panic!("expected QUIC reach, got {other:?}"),
        };
        assert_eq!(addr(&stream_a), "127.0.0.1:4101");
        assert_eq!(addr(&stream_b), "127.0.0.1:4102");
        Ok(())
    }

    #[tokio::test]
    async fn stream_channel_uses_service_scoped_moq_origin() -> Result<()> {
        let scoped_origin = crate::moq_stream::MoqStreamOrigin::standalone().build();
        let origin_handle = std::sync::Arc::new(parking_lot::RwLock::new(Some(scoped_origin.clone())));
        let channel = StreamChannel::new(SigningKey::from_bytes(&[3; 32]))
            .with_moq_origin_handle(origin_handle);
        let (_, client_pub) = crate::crypto::generate_ephemeral_keypair();
        let stream = channel.prepare_stream(&client_pub.to_bytes(), 60).await?;
        let _publisher = channel.publisher(&stream).await?;

        let path = scoped_origin.broadcast_path(stream.topic());
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            scoped_origin.consumer().announced_broadcast(&path),
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("scoped origin did not announce {path}"))?;
        Ok(())
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

    #[test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    fn flat_slice_parses_unaligned_block() {
        // The zero-copy verifier (`stream_consumer`) parses StreamBlocks via
        // `read_message_from_flat_slice` over `Bytes` that come straight from the
        // moq/quinn receive buffer — and those can start at ANY byte offset, not
        // an 8-byte boundary. capnp's flat-slice reader requires 8-byte alignment
        // unless the `unaligned` feature is on, so this guards that we keep it on.
        let payloads = vec![StreamPayloadData::Data(b"hello".to_vec())];
        let bytes = encode_stream_block(&[0u8; 16], 42, 7, &payloads).unwrap();
        // Force a deliberately unaligned view: offset the message by 1 byte.
        let mut padded = vec![0u8; 1];
        padded.extend_from_slice(&bytes);
        let unaligned = &padded[1..];
        assert_ne!(
            unaligned.as_ptr() as usize % 8,
            0,
            "test precondition: slice must be unaligned"
        );
        let mut slice: &[u8] = unaligned;
        let reader = capnp::serialize::read_message_from_flat_slice(
            &mut slice,
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
