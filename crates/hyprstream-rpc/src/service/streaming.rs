//! StreamService - PULL/XPUB queuing proxy with signed registration (E2E blind forwarder)
//!
//! # Architecture
//!
//! ```text
//! InferenceService                 StreamService                    Client
//!    │                                  │                              │
//!    │─ SignedEnvelope(StreamRegister) ►│                              │
//!    │   [verify sig, check claims]     │                              │
//!    │                                  │                              │
//!    │─ StreamBlock [topic,capnp,mac] ─►│                              │
//!    │   [extract topic from frame 0]   │                              │
//!    │   [NO HMAC verification]         │                              │
//!    │                                  │─ [topic,capnp,mac] ─────────►│
//!    │                                  │   [XPUB prefix routing]      │[verify HMAC chain]
//!    │                                  │                              │
//!    │                                  │◄─ StreamResume(topic,hmac) ──│
//!    │                                  │   [find hmac in buffer]      │
//!    │                                  │─ {buffered blocks...} ──────►│
//! ```
//!
//! # Why PUSH/PULL instead of PUB/XSUB
//!
//! PUB/SUB drops messages when no subscriber exists. This causes a race condition:
//! - Publisher starts immediately after returning stream_id
//! - Client needs time to subscribe
//! - Early messages are dropped before client subscribes
//!
//! PUSH/PULL solves this:
//! - PUSH buffers at HWM (never drops)
//! - StreamService queues per-topic until subscriber arrives
//! - On subscribe, queued messages are flushed to client
//!
//! # Security Model (E2E Authentication)
//!
//! StreamService is a **blind forwarder** - it does NOT verify HMACs.
//! HMAC verification is done **end-to-end** by the client.
//!
//! - **Topic derivation**: DH-derived topic (InferenceService ↔ Client) - unpredictable
//! - **StreamRegister**: Signed capnp wrapped in `SignedEnvelope`, verified before accepting
//! - **StreamBlock**: 3-frame multipart [topic, capnp, 16-byte MAC] with batched payloads
//! - **Client verifies**: Client derives same keys from DH and verifies HMAC chain
//! - **StreamResume**: Client provides last valid HMAC, service resends subsequent blocks
//!
//! # Memory Management
//!
//! - **Unified StreamState**: Single HashMap tracks auth, subscription, messages, and HMAC state
//! - **Claims-based expiry**: Entries removed when claims.exp timestamp passes
//! - **Unsubscribe cleanup**: Entry removed entirely on 0x00 (prevents leaks)
//! - **Message TTL**: Individual messages expire after 30s if not delivered
//! - **Per-topic limit**: Max 1000 messages per topic (oldest dropped on overflow)
//! - **Retransmit buffer**: Chunks retained for resume requests (HMAC-indexed)

use crate::auth::Scope;
use crate::capnp::FromCapnp;
use crate::common_capnp;
use crate::streaming_capnp;
use crate::crypto::VerifyingKey;
use crate::envelope::{InMemoryNonceCache, SignedEnvelope};
use crate::transport::TransportConfig;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use subtle::ConstantTimeEq;
use tokio::sync::Notify;
use tracing::{debug, error, info, trace, warn};

/// A pending message waiting for a subscriber
struct PendingMessage {
    /// Message frames (multipart: [topic, capnp, mac])
    frames: Vec<Vec<u8>>,
    /// When this message was received
    received_at: Instant,
    /// The MAC from this message (for retransmit buffer indexing, 16 bytes)
    mac: Vec<u8>,
}

/// State for an authorized stream
///
/// Unified structure that tracks authorization, subscription status, and pending messages.
/// Note: StreamService is a blind forwarder - it does NOT verify HMACs.
/// HMAC verification is done end-to-end by the client.
struct StreamState {
    /// Expiration timestamp (Unix millis) from the authorizing request's claims
    /// Stream is removed during compact() when `now > exp`
    exp: i64,

    /// Whether a client has subscribed to this stream
    subscribed: bool,

    /// Messages queued before subscriber arrived (flushed on subscribe)
    /// Also serves as retransmit buffer - indexed by HMAC for StreamResume
    messages: VecDeque<PendingMessage>,
}

impl StreamState {
    /// Find position in buffer matching the given MAC
    /// Supports both 32-byte legacy HMAC and 16-byte StreamBlock MAC
    fn find_mac_position(&self, mac: &[u8]) -> Option<usize> {
        self.messages.iter().position(|msg| {
            if msg.mac.len() == mac.len() {
                msg.mac.ct_eq(mac).into()
            } else {
                false
            }
        })
    }
}

/// StreamService - PULL/XPUB queuing proxy with signed registration (E2E blind forwarder)
///
/// Receives messages via PULL (from publishers using PUSH), queues them
/// per-topic until a subscriber arrives, then delivers via XPUB.
///
/// Security (E2E Authentication):
/// - StreamService is a **blind forwarder** - does NOT verify HMACs
/// - Registration via `SignedEnvelope(StreamRegister)` - signature verified
/// - Client verifies HMAC chain end-to-end using DH-derived keys
/// - Resume via `StreamResume` - client provides last valid HMAC
pub struct StreamService {
    /// Service name (for logging and registry)
    name: String,

    /// ZMQ context
    context: Arc<zmq::Context>,

    /// XPUB frontend transport (client-facing)
    pub_transport: TransportConfig,

    /// PULL backend transport (receives from publishers)
    pull_transport: TransportConfig,

    /// Message TTL - messages older than this are dropped (default: 30s)
    message_ttl: Duration,

    /// Maximum pending messages per topic (default: 1000)
    max_pending_per_topic: usize,

    /// Interval between compaction runs (default: 5s)
    compact_interval: Duration,

    /// Nonce cache for replay protection on SignedEnvelope
    nonce_cache: Arc<InMemoryNonceCache>,

    /// Verifying key for signature verification on StreamRegister messages
    verifying_key: VerifyingKey,
}

impl StreamService {
    /// Create a new StreamService
    ///
    /// # Arguments
    /// * `name` - Service name for logging (matches ProxyService pattern)
    /// * `context` - ZMQ context
    /// * `pub_transport` - XPUB frontend config (client-facing, with optional CurveZMQ)
    /// * `pull_transport` - PULL backend config (receives from publishers)
    /// * `verifying_key` - Ed25519 public key for verifying StreamRegister signatures
    ///
    /// # Security (E2E Authentication)
    ///
    /// StreamService is a **blind forwarder**:
    /// - Does NOT verify HMACs (client verifies end-to-end)
    /// - Topics are DH-derived (InferenceService ↔ Client), unpredictable
    /// - Stream registration via `SignedEnvelope(StreamRegister)` - signature verified
    /// - CurveZMQ can be enabled via transport configs for transport-layer encryption
    pub fn new(
        name: impl Into<String>,
        context: Arc<zmq::Context>,
        pub_transport: TransportConfig,
        pull_transport: TransportConfig,
        verifying_key: VerifyingKey,
    ) -> Self {
        let name = name.into();

        // Log transport security configuration
        let curve_enabled = pub_transport.curve.is_some() || pull_transport.curve.is_some();
        if curve_enabled {
            info!(
                service=%name,
                pub_encrypted=pub_transport.curve.is_some(),
                pull_encrypted=pull_transport.curve.is_some(),
                "StreamService initialized with CurveZMQ encryption (E2E blind forwarder)"
            );
        } else {
            info!(
                service=%name,
                "StreamService initialized (E2E blind forwarder, plaintext transport)"
            );
        }

        Self {
            name,
            context,
            pub_transport,
            pull_transport,
            message_ttl: Duration::from_secs(30),
            max_pending_per_topic: 1000,
            compact_interval: Duration::from_secs(5),
            nonce_cache: Arc::new(InMemoryNonceCache::new()),
            verifying_key,
        }
    }

    /// Configure buffer sizes and TTL
    ///
    /// # Arguments
    /// * `max_pending_per_topic` - Maximum messages to buffer per topic (for queue and retransmit)
    /// * `message_ttl` - How long to keep messages before expiry
    /// * `compact_interval` - How often to run compaction
    pub fn with_buffer_config(
        mut self,
        max_pending_per_topic: usize,
        message_ttl: Duration,
        compact_interval: Duration,
    ) -> Self {
        self.max_pending_per_topic = max_pending_per_topic;
        self.message_ttl = message_ttl;
        self.compact_interval = compact_interval;
        self
    }

    /// Handle a SignedEnvelope(StreamRegister) message
    ///
    /// 1. Parse SignedEnvelope from capnp
    /// 2. Verify signature against expected service pubkey
    /// 3. Parse StreamRegister from payload
    /// 4. Check claims allow publishing to this topic
    /// 5. Register the stream (topic is DH-derived, unpredictable)
    ///
    /// Note: StreamService is a blind forwarder - no HMAC key derivation.
    /// Topic is DH-derived by InferenceService and Client.
    fn handle_register(
        &self,
        msg: &[u8],
        streams: &mut HashMap<String, StreamState>,
    ) -> Result<String> {
        use capnp::serialize;

        // Parse SignedEnvelope
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(msg),
            capnp::message::ReaderOptions::default(),
        )?;
        let signed_reader = reader.get_root::<common_capnp::signed_envelope::Reader>()?;
        let signed = SignedEnvelope::read_from(signed_reader)?;

        // Verify signature against expected service pubkey (mandatory)
        // Full verification with nonce cache for replay protection
        signed.verify(&self.verifying_key, &*self.nonce_cache)?;

        // Parse StreamRegister from payload
        // Topic is DH-derived (64 hex chars), unpredictable
        let (topic, exp) = parse_stream_register(&signed.envelope.payload)
            .ok_or_else(|| anyhow!("Invalid StreamRegister payload"))?;

        // Check claims allow publishing to this topic
        if let Some(claims) = &signed.envelope.claims {
            let required = Scope::new(
                "publish".to_owned(),
                "stream".to_owned(),
                topic.clone(),
            );
            let has_scope = claims.admin || claims.scopes.iter().any(|s| s.grants(&required));

            if !has_scope {
                return Err(anyhow!(
                    "Claims do not authorize publishing to stream: {}",
                    topic
                ));
            }
        }

        // Register the stream (blind forwarder - no HMAC key needed)
        streams.insert(topic.clone(), StreamState {
            exp,
            subscribed: false,
            messages: VecDeque::new(),
        });

        Ok(topic)
    }

    /// Setup PULL socket with transport-layer security (CurveZMQ + permissions)
    ///
    /// PULL socket receives messages from publishers (using PUSH).
    /// Unlike XSUB, PULL receives ALL messages - no subscription matching.
    /// This prevents message drops before subscribers arrive.
    ///
    /// Uses the unified transport layer which handles:
    /// - CurveZMQ encryption (if configured)
    /// - Socket binding (inproc/IPC/systemd)
    /// - Filesystem permissions (IPC sockets)
    fn setup_pull(&self) -> Result<zmq::Socket> {
        let mut pull = self.context.socket(zmq::PULL)
            .map_err(|e| anyhow!("Failed to create PULL socket: {}", e))?;

        // Set high water mark for buffering (100K messages ~ 10 seconds at 10K msg/s)
        pull.set_rcvhwm(100_000)
            .map_err(|e| anyhow!("Failed to set PULL HWM: {}", e))?;

        // Bind using transport layer (handles CurveZMQ automatically)
        self.pull_transport.bind(&mut pull)
            .map_err(|e| anyhow!("Failed to bind PULL: {}", e))?;

        let endpoint = self.pull_transport.to_zmq_string();
        if self.pull_transport.curve.is_some() {
            info!(
                service = %self.name,
                endpoint = %endpoint,
                "PULL backend bound with CurveZMQ encryption"
            );
        } else {
            info!(
                service = %self.name,
                endpoint = %endpoint,
                "PULL backend bound (plaintext)"
            );
        }

        // Set restrictive filesystem permissions for IPC sockets (Layer 3 defense)
        #[cfg(unix)]
        if let crate::transport::EndpointType::Ipc { ref path } = self.pull_transport.endpoint {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| anyhow!("Failed to set IPC socket permissions: {}", e))?;

            debug!(
                service = %self.name,
                path = %path.display(),
                "IPC socket permissions set to 0o600 (owner-only)"
            );
        }

        Ok(pull)
    }

    /// Setup XPUB socket (client-facing frontend) with transport-layer security
    ///
    /// Uses the unified transport layer which handles CurveZMQ automatically.
    fn setup_xpub(&self) -> Result<zmq::Socket> {
        let mut xpub = self.context.socket(zmq::XPUB)
            .map_err(|e| anyhow!("Failed to create XPUB socket: {}", e))?;

        // Set high water mark for buffering
        xpub.set_sndhwm(100_000)
            .map_err(|e| anyhow!("Failed to set XPUB HWM: {}", e))?;

        // Enable verbose mode to receive all subscribe/unsubscribe notifications
        xpub.set_xpub_verbose(true)
            .map_err(|e| anyhow!("Failed to set XPUB verbose: {}", e))?;

        // Bind using transport layer (handles CurveZMQ automatically)
        self.pub_transport.bind(&mut xpub)
            .map_err(|e| anyhow!("Failed to bind XPUB: {}", e))?;

        let endpoint = self.pub_transport.to_zmq_string();
        if self.pub_transport.curve.is_some() {
            info!(
                service = %self.name,
                endpoint = %endpoint,
                "XPUB frontend bound with CurveZMQ encryption"
            );
        } else {
            info!(
                service = %self.name,
                endpoint = %endpoint,
                "XPUB frontend bound (plaintext)"
            );
        }

        Ok(xpub)
    }

    /// Run the streaming proxy loop with pre-bound sockets (blocking)
    ///
    /// This is the main event loop that:
    /// 1. Receives messages from PULL (publishers)
    /// 2. Queues messages per-topic until subscriber arrives
    /// 3. Receives subscriptions from XPUB (clients)
    /// 4. Validates JWT, adds subscriber, flushes queued messages
    ///
    /// # Arguments
    /// * `xpub` - Pre-bound XPUB socket (client-facing)
    /// * `pull` - Pre-bound PULL socket (receives from publishers)
    /// * `shutdown` - Notification signal to stop the service
    fn run_loop_with_sockets(
        &self,
        xpub: zmq::Socket,
        pull: zmq::Socket,
        shutdown: Arc<Notify>,
    ) -> Result<(), crate::error::RpcError> {
        // Create CTRL socket for shutdown (PAIR pattern)
        let ctrl = self.context.socket(zmq::PAIR)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL socket: {e}")))?;
        let ctrl_endpoint = format!("inproc://stream-ctrl-{}", self.name);
        ctrl.bind(&ctrl_endpoint)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL bind: {e}")))?;

        // Spawn shutdown listener thread
        let ctrl_sender = self.context.socket(zmq::PAIR)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL sender: {e}")))?;
        ctrl_sender.connect(&ctrl_endpoint)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL connect: {e}")))?;

        let name_clone = self.name.clone();
        std::thread::spawn(move || {
            // Block until shutdown is signaled
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::error!("Failed to create shutdown listener runtime: {e}");
                    return;
                }
            };
            rt.block_on(shutdown.notified());

            // Send termination message
            debug!("Sending TERMINATE to StreamService {}", name_clone);
            let _ = ctrl_sender.send("TERMINATE", 0);
        });

        // Unified stream state (local to this loop)
        // Replaces separate pending, subscribers, and authorized_streams collections
        let mut streams: HashMap<String, StreamState> = HashMap::new();
        let mut last_compact = Instant::now();

        // Create poll items (PULL, XPUB, CTRL)
        let mut items = [
            pull.as_poll_item(zmq::POLLIN),
            xpub.as_poll_item(zmq::POLLIN),
            ctrl.as_poll_item(zmq::POLLIN),
        ];

        info!(service = %self.name, "StreamService queuing proxy loop started");

        loop {
            // Periodic compaction (every compact_interval)
            // Removes expired streams and expired messages within active streams
            if last_compact.elapsed() >= self.compact_interval {
                last_compact = Instant::now();
                let now_unix = chrono::Utc::now().timestamp();
                let now_instant = Instant::now();

                streams.retain(|topic, state| {
                    // Remove if claims expired
                    if state.exp <= now_unix {
                        debug!(
                            service = %self.name,
                            topic = %topic,
                            "Removing expired stream (claims expired)"
                        );
                        return false;
                    }

                    // For non-subscribed streams, also expire old messages
                    if !state.subscribed {
                        while let Some(msg) = state.messages.front() {
                            if now_instant.duration_since(msg.received_at) > self.message_ttl {
                                trace!(
                                    service = %self.name,
                                    topic = %topic,
                                    "Expired pending message"
                                );
                                state.messages.pop_front();
                            } else {
                                break;
                            }
                        }
                    }

                    // Keep if: subscribed OR has pending messages
                    state.subscribed || !state.messages.is_empty()
                });
            }

            // Poll with 1 second timeout (allows periodic compaction)
            zmq::poll(&mut items, 1000)
                .map_err(|e| crate::error::RpcError::SpawnFailed(format!("Poll failed: {e}")))?;

            // Check CTRL socket for shutdown
            if items[2].is_readable() {
                if let Ok(msg) = ctrl.recv_string(0) {
                    if msg.as_ref().map(std::string::String::as_str) == Ok("TERMINATE") {
                        info!(service = %self.name, "Received TERMINATE signal, stopping proxy");
                        break;
                    }
                }
            }

            // PULL → Handle SignedEnvelope(StreamRegister), StreamBlock, or StreamResume
            if items[0].is_readable() {
                match pull.recv_multipart(0) {
                    Ok(frames) => {
                        // Check for 3-frame StreamBlock multipart: [topic, capnp, 16-byte mac]
                        if frames.len() == 3 && frames[2].len() == 16 {
                            // StreamBlock multipart format
                            let topic = String::from_utf8_lossy(&frames[0]).to_string();
                            let mac = frames[2].clone();

                            if let Some(state) = streams.get_mut(&topic) {
                                if state.subscribed {
                                    // Forward multipart message (topic is already first frame)
                                    trace!(
                                        service = %self.name,
                                        topic = %topic,
                                        frames = frames.len(),
                                        "Forwarding StreamBlock to subscriber (blind)"
                                    );
                                    if let Err(e) = send_multipart(&xpub, &frames) {
                                        error!(
                                            service = %self.name,
                                            topic = %topic,
                                            error = %e,
                                            "Failed to forward StreamBlock"
                                        );
                                    }
                                }

                                // Buffer for retransmission
                                while state.messages.len() >= self.max_pending_per_topic {
                                    state.messages.pop_front();
                                }

                                state.messages.push_back(PendingMessage {
                                    frames,
                                    received_at: Instant::now(),
                                    mac,
                                });

                                if !state.subscribed {
                                    trace!(
                                        service = %self.name,
                                        topic = %topic,
                                        queue_len = state.messages.len(),
                                        "Queued StreamBlock for later delivery"
                                    );
                                }
                            } else {
                                trace!(
                                    service = %self.name,
                                    topic = %topic,
                                    "Dropping StreamBlock for unregistered topic"
                                );
                            }
                            continue;
                        }

                        // Single-frame message: SignedEnvelope or StreamResume
                        if frames.len() == 1 {
                            let message = &frames[0];

                            // Try SignedEnvelope (StreamRegister)
                            if is_signed_envelope(message) {
                                match self.handle_register(message, &mut streams) {
                                    Ok(topic) => {
                                        info!(
                                            service = %self.name,
                                            topic = %topic,
                                            "Stream registered via SignedEnvelope"
                                        );
                                    }
                                    Err(e) => {
                                        warn!(
                                            service = %self.name,
                                            error = %e,
                                            "Failed to process StreamRegister"
                                        );
                                    }
                                }
                                continue;
                            }

                            // Try StreamResume
                            if let Some((topic, resume_mac)) = parse_stream_resume(message) {
                                if let Some(state) = streams.get(&topic) {
                                    if let Some(start_idx) = state.find_mac_position(&resume_mac) {
                                        let mut resent = 0;
                                        for msg in state.messages.iter().skip(start_idx + 1) {
                                            if let Err(e) = send_multipart(&xpub, &msg.frames) {
                                                error!(
                                                    service = %self.name,
                                                    topic = %topic,
                                                    error = %e,
                                                    "Failed to resend on resume"
                                                );
                                                break;
                                            }
                                            resent += 1;
                                        }
                                        info!(
                                            service = %self.name,
                                            topic = %topic,
                                            resent = resent,
                                            "Processed StreamResume request"
                                        );
                                    } else {
                                        warn!(
                                            service = %self.name,
                                            topic = %topic,
                                            "StreamResume MAC not found in buffer"
                                        );
                                    }
                                } else {
                                    warn!(
                                        service = %self.name,
                                        topic = %topic,
                                        "StreamResume for unknown topic"
                                    );
                                }
                                continue;
                            }

                            // Unknown single-frame message (legacy StreamChunk no longer supported)
                            warn!(
                                service = %self.name,
                                len = message.len(),
                                "Unknown single-frame message format"
                            );
                        } else {
                            warn!(
                                service = %self.name,
                                frames = frames.len(),
                                "Unexpected multipart frame count"
                            );
                        }
                    }
                    Err(e) => {
                        error!(
                            service = %self.name,
                            error = %e,
                            "Failed to receive from PULL"
                        );
                    }
                }
            }

            // XPUB subscriptions from clients
            if items[1].is_readable() {
                match xpub.recv_bytes(0) {
                    Ok(subscription) => {
                        match subscription.first() {
                            Some(0x01) => {
                                // Subscribe request
                                trace!(
                                    service = %self.name,
                                    len = subscription.len(),
                                    "Received subscription from XPUB"
                                );

                                // Extract topic from subscription (skip 0x01 control byte)
                                // Client subscribes with just "stream-{uuid}" after pre-authorization
                                let topic = String::from_utf8_lossy(&subscription[1..]).to_string();

                                // Check if stream was authorized
                                if let Some(state) = streams.get_mut(&topic) {
                                    // Mark as subscribed
                                    state.subscribed = true;
                                    info!(
                                        service = %self.name,
                                        topic = %topic,
                                        "Stream subscription accepted"
                                    );

                                    // Flush any pending messages (keep in buffer for retransmit)
                                    let now = Instant::now();
                                    let mut delivered = 0;
                                    let mut expired = 0;

                                    for msg in state.messages.iter() {
                                        if now.duration_since(msg.received_at) <= self.message_ttl {
                                            // Send multipart (handles both single and multi-frame)
                                            if let Err(e) = send_multipart(&xpub, &msg.frames) {
                                                error!(
                                                    service = %self.name,
                                                    topic = %topic,
                                                    error = %e,
                                                    "Failed to flush pending message"
                                                );
                                            } else {
                                                delivered += 1;
                                            }
                                        } else {
                                            expired += 1;
                                        }
                                    }

                                    if delivered > 0 || expired > 0 {
                                        info!(
                                            service = %self.name,
                                            topic = %topic,
                                            delivered = delivered,
                                            expired = expired,
                                            "Flushed pending messages on subscribe"
                                        );
                                    }
                                } else {
                                    warn!(
                                        service = %self.name,
                                        topic = %topic,
                                        "Subscription rejected - stream not authorized"
                                    );
                                }
                            }
                            Some(0x00) => {
                                // Unsubscribe request - extract topic (skip control byte)
                                // Remove entry ENTIRELY to prevent memory leak
                                let topic = String::from_utf8_lossy(&subscription[1..]).to_string();
                                if streams.remove(&topic).is_some() {
                                    debug!(
                                        service = %self.name,
                                        topic = %topic,
                                        "Removed stream on unsubscribe"
                                    );
                                }
                            }
                            _ => {
                                trace!(
                                    service = %self.name,
                                    "Received unknown XPUB control message"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        error!(
                            service = %self.name,
                            error = %e,
                            "Failed to receive from XPUB"
                        );
                    }
                }
            }
        }

        info!(
            service = %self.name,
            streams = streams.len(),
            "StreamService proxy stopped"
        );
        Ok(())
    }
}

/// Parse a StreamRegister message from SignedEnvelope payload
fn parse_stream_register(payload: &[u8]) -> Option<(String, i64)> {
    use capnp::serialize;

    let reader = serialize::read_message(
        &mut std::io::Cursor::new(payload),
        capnp::message::ReaderOptions::default(),
    ).ok()?;

    let register = reader.get_root::<streaming_capnp::stream_register::Reader>().ok()?;
    let topic = register.get_topic().ok()?.to_str().ok()?.to_owned();
    let exp = register.get_exp();

    Some((topic, exp))
}

/// Parse a StreamResume message
fn parse_stream_resume(msg: &[u8]) -> Option<(String, [u8; 32])> {
    use capnp::serialize;

    let reader = serialize::read_message(
        &mut std::io::Cursor::new(msg),
        capnp::message::ReaderOptions::default(),
    ).ok()?;

    let resume = reader.get_root::<streaming_capnp::stream_resume::Reader>().ok()?;
    let topic = resume.get_topic().ok()?.to_str().ok()?.to_owned();
    let hmac_data = resume.get_resume_from_hmac().ok()?;

    if hmac_data.len() != 32 {
        return None;
    }
    let mut hmac = [0u8; 32];
    hmac.copy_from_slice(hmac_data);

    Some((topic, hmac))
}

/// Check if a message is a SignedEnvelope (for registration)
fn is_signed_envelope(msg: &[u8]) -> bool {
    use capnp::serialize;

    let Ok(reader) = serialize::read_message(
        &mut std::io::Cursor::new(msg),
        capnp::message::ReaderOptions::default(),
    ) else {
        return false;
    };

    let Ok(envelope) = reader.get_root::<common_capnp::signed_envelope::Reader>() else {
        return false;
    };

    // Also check that signature is exactly 64 bytes (Ed25519 signature)
    let Ok(signature) = envelope.get_signature() else {
        return false;
    };
    signature.len() == 64
}

/// Send a multipart message over a ZMQ socket
///
/// Sends all but last frame with SNDMORE, then last frame without.
fn send_multipart(socket: &zmq::Socket, frames: &[Vec<u8>]) -> Result<()> {
    if frames.is_empty() {
        return Err(anyhow!("Cannot send empty multipart message"));
    }

    let last_idx = frames.len() - 1;
    for (i, frame) in frames.iter().enumerate() {
        let flags = if i < last_idx { zmq::SNDMORE } else { 0 };
        socket.send(frame.as_slice(), flags)
            .map_err(|e| anyhow!("Failed to send frame {}: {}", i, e))?;
    }

    Ok(())
}

/// Implement Spawnable trait for StreamService
///
/// This allows StreamService to be spawned by ServiceSpawner in the same way
/// as other proxy services (ProxyService, etc.)
impl crate::service::spawner::Spawnable for StreamService {
    fn name(&self) -> &str {
        &self.name
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn registrations(&self) -> Vec<(crate::registry::SocketKind, TransportConfig)> {
        vec![
            // Note: Registrations show the client-side view
            // XPUB frontend is registered as "Sub" (clients subscribe via SUB)
            // PULL backend is registered as "Push" (publishers connect via PUSH)
            (crate::registry::SocketKind::Sub, self.pub_transport.clone()),
            (crate::registry::SocketKind::Push, self.pull_transport.clone()),
        ]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), crate::error::RpcError> {
        info!(
            service = %self.name,
            pub_endpoint = %self.pub_transport.to_zmq_string(),
            pull_endpoint = %self.pull_transport.to_zmq_string(),
            "Starting StreamService queuing proxy"
        );

        // Setup sockets BEFORE signaling ready
        let xpub = self.setup_xpub()
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("XPUB setup: {e}")))?;
        let pull = self.setup_pull()
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("PULL setup: {e}")))?;

        // Signal ready AFTER sockets are bound
        if let Some(tx) = on_ready {
            let _ = tx.send(());
        }

        // Notify systemd that service is ready (for Type=notify services)
        let _ = crate::notify::ready();

        // Run the queuing proxy loop with pre-bound sockets
        self.run_loop_with_sockets(xpub, pull, shutdown)
    }
}
