//! StreamService - PULL/XPUB queuing proxy with claims-based expiry
//!
//! # Architecture
//!
//! ```text
//! InferenceService                StreamService                     Client
//!       │                              │                              │
//!       │──AUTHORIZE|stream-uuid|exp──►│ (pre-authorize with expiry)  │
//!       │                              │                              │
//!       │──stream-uuid.chunk──────────►│ (queue if no subscriber)     │
//!       │                              │                              │
//!       │                              │◄───────\x01stream-uuid───────│
//!       │                              │ (subscription, authorize OK)  │
//!       │                              │                              │
//!       │                              │────stream-uuid.chunk────────►│
//!       │                              │ (flush queued + deliver)      │
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
//! # Authorization Flow
//!
//! 1. InferenceService validates JWT claims at request time
//! 2. InferenceService sends AUTHORIZE message with stream_id and claims expiry
//! 3. Client subscribes with just `stream-{uuid}` (no JWT needed)
//! 4. StreamService checks stream was authorized, allows subscription
//! 5. On unsubscribe (0x00), entry is removed entirely
//! 6. Periodic compact() removes expired entries (claims.exp)
//!
//! # Memory Management
//!
//! - **Unified StreamState**: Single HashMap tracks auth, subscription, and messages
//! - **Claims-based expiry**: Entries removed when claims.exp timestamp passes
//! - **Unsubscribe cleanup**: Entry removed entirely on 0x00 (prevents leaks)
//! - **Message TTL**: Individual messages expire after 30s if not delivered
//! - **Per-topic limit**: Max 1000 messages per topic (oldest dropped on overflow)

use crate::transport::TransportConfig;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tracing::{debug, error, info, trace, warn};

/// A pending message waiting for a subscriber
struct PendingMessage {
    data: Vec<u8>,
    received_at: Instant,
}

/// State for an authorized stream
///
/// Unified structure that tracks authorization, subscription status, and pending messages.
/// Replaces separate `authorized_streams`, `subscribers`, and `pending` collections.
struct StreamState {
    /// Expiration timestamp (Unix seconds) from the authorizing request's claims
    /// Stream is removed during compact() when `now > exp`
    exp: i64,
    /// Whether a client has subscribed to this stream
    subscribed: bool,
    /// Messages queued before subscriber arrived (flushed on subscribe)
    messages: VecDeque<PendingMessage>,
}

/// StreamService - PULL/XPUB queuing proxy with claims-based expiry
///
/// Receives messages via PULL (from publishers using PUSH), queues them
/// per-topic until a subscriber arrives, then delivers via XPUB.
/// Authorization is handled via AUTHORIZE messages with expiry timestamps.
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
}

impl StreamService {
    /// Create a new StreamService
    ///
    /// # Arguments
    /// * `name` - Service name for logging (matches ProxyService pattern)
    /// * `context` - ZMQ context
    /// * `pub_transport` - XPUB frontend config (client-facing, with optional CurveZMQ)
    /// * `pull_transport` - PULL backend config (receives from publishers)
    ///
    /// # Security
    ///
    /// CurveZMQ can be enabled via `pub_transport.with_curve()` and `pull_transport.with_curve()`.
    /// The transport layer handles encryption automatically.
    /// Authorization is handled via AUTHORIZE messages with claims-based expiry.
    pub fn new(
        name: impl Into<String>,
        context: Arc<zmq::Context>,
        pub_transport: TransportConfig,
        pull_transport: TransportConfig,
    ) -> Self {
        let name = name.into();

        // Log transport security configuration
        let curve_enabled = pub_transport.curve.is_some() || pull_transport.curve.is_some();
        if curve_enabled {
            info!(
                service=%name,
                pub_encrypted=pub_transport.curve.is_some(),
                pull_encrypted=pull_transport.curve.is_some(),
                "StreamService initialized with CurveZMQ encryption"
            );
        } else {
            info!(
                service=%name,
                "StreamService initialized (CurveZMQ disabled - plaintext transport)"
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
        }
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
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL socket: {}", e)))?;
        let ctrl_endpoint = format!("inproc://stream-ctrl-{}", self.name);
        ctrl.bind(&ctrl_endpoint)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL bind: {}", e)))?;

        // Spawn shutdown listener thread
        let ctrl_sender = self.context.socket(zmq::PAIR)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL sender: {}", e)))?;
        ctrl_sender.connect(&ctrl_endpoint)
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("CTRL connect: {}", e)))?;

        let name_clone = self.name.clone();
        std::thread::spawn(move || {
            // Block until shutdown is signaled
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("shutdown listener runtime");
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
                .map_err(|e| crate::error::RpcError::SpawnFailed(format!("Poll failed: {}", e)))?;

            // Check CTRL socket for shutdown
            if items[2].is_readable() {
                if let Ok(msg) = ctrl.recv_string(0) {
                    if msg.as_ref().map(|s| s.as_str()) == Ok("TERMINATE") {
                        info!(service = %self.name, "Received TERMINATE signal, stopping proxy");
                        break;
                    }
                }
            }

            // PULL → Handle authorize messages OR queue/deliver chunks
            if items[0].is_readable() {
                match pull.recv_bytes(0) {
                    Ok(message) => {
                        // Check for AUTHORIZE message from InferenceService
                        // Format: "AUTHORIZE|stream-{uuid}|{exp_timestamp}"
                        if message.starts_with(b"AUTHORIZE|") {
                            let auth_msg = String::from_utf8_lossy(&message);
                            let parts: Vec<&str> = auth_msg.splitn(3, '|').collect();
                            if parts.len() >= 3 {
                                let stream_id = parts[1].to_string();
                                let exp = parts[2].parse::<i64>().unwrap_or_else(|_| {
                                    // Default: 5 minutes from now if parsing fails
                                    chrono::Utc::now().timestamp() + 300
                                });

                                // Create stream state with expiry
                                streams.insert(stream_id.clone(), StreamState {
                                    exp,
                                    subscribed: false,
                                    messages: VecDeque::new(),
                                });
                                info!(
                                    service = %self.name,
                                    stream_id = %stream_id,
                                    exp = %exp,
                                    "Stream authorized with expiry"
                                );
                            } else if parts.len() == 2 {
                                // Legacy format without expiry (shouldn't happen in production)
                                let stream_id = parts[1].to_string();
                                let exp = chrono::Utc::now().timestamp() + 300; // 5 min default
                                streams.insert(stream_id.clone(), StreamState {
                                    exp,
                                    subscribed: false,
                                    messages: VecDeque::new(),
                                });
                                warn!(
                                    service = %self.name,
                                    stream_id = %stream_id,
                                    "Stream authorized with legacy format (no expiry provided)"
                                );
                            } else {
                                warn!(
                                    service = %self.name,
                                    "Invalid AUTHORIZE message format"
                                );
                            }
                            continue;
                        }

                        // Regular chunk message - extract topic (format: "stream-{uuid}...")
                        if let Some(topic) = extract_topic(&message) {
                            if let Some(state) = streams.get_mut(&topic) {
                                if state.subscribed {
                                    // Subscriber exists → deliver immediately
                                    trace!(
                                        service = %self.name,
                                        topic = %topic,
                                        len = message.len(),
                                        "Delivering message to subscriber"
                                    );
                                    if let Err(e) = xpub.send(&message, 0) {
                                        error!(
                                            service = %self.name,
                                            topic = %topic,
                                            error = %e,
                                            "Failed to deliver message to XPUB"
                                        );
                                    }
                                } else {
                                    // Authorized but no subscriber yet → queue for later
                                    // Enforce per-topic limit (drop oldest)
                                    while state.messages.len() >= self.max_pending_per_topic {
                                        state.messages.pop_front();
                                        trace!(
                                            service = %self.name,
                                            topic = %topic,
                                            "Dropped oldest pending message (queue full)"
                                        );
                                    }

                                    state.messages.push_back(PendingMessage {
                                        data: message,
                                        received_at: Instant::now(),
                                    });
                                    trace!(
                                        service = %self.name,
                                        topic = %topic,
                                        queue_len = state.messages.len(),
                                        "Queued message for later delivery"
                                    );
                                }
                            } else {
                                // Not authorized - drop message
                                trace!(
                                    service = %self.name,
                                    topic = %topic,
                                    "Dropping message for unauthorized stream"
                                );
                            }
                        } else {
                            warn!(
                                service = %self.name,
                                len = message.len(),
                                "Received message with invalid topic format"
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

                                    // Flush any pending messages
                                    let now = Instant::now();
                                    let mut delivered = 0;
                                    let mut expired = 0;

                                    while let Some(msg) = state.messages.pop_front() {
                                        if now.duration_since(msg.received_at) <= self.message_ttl {
                                            if let Err(e) = xpub.send(&msg.data, 0) {
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

/// Extract topic from a message
///
/// Topic format: "stream-{uuid}" = 43 bytes (7 + 36)
fn extract_topic(msg: &[u8]) -> Option<String> {
    // Topic format: "stream-{uuid}" = 43 bytes (7 + 36)
    if msg.len() >= 43 && msg.starts_with(b"stream-") {
        String::from_utf8(msg[..43].to_vec()).ok()
    } else {
        None
    }
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
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("XPUB setup: {}", e)))?;
        let pull = self.setup_pull()
            .map_err(|e| crate::error::RpcError::SpawnFailed(format!("PULL setup: {}", e)))?;

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
