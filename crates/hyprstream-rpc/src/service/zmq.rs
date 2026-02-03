//! ZMQ service infrastructure
//!
//! Provides the foundation for REQ/REP services and clients using ZMQ.
//! Uses TMQ for async I/O - services run as async tasks with proper epoll integration.
//!
//! # Envelope-Based Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `RequestLoop` unwraps and verifies signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.casbin_subject()` for policy checks

use crate::prelude::*;
use crate::transport::TransportConfig;
use anyhow::{anyhow, Result};
use chrono::Utc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tmq::{FromZmqSocket, Multipart, RequestReceiver, RequestSender};
use tokio::sync::Notify;
use tracing::{debug, error, info, trace, warn};

/// Context extracted from a verified SignedEnvelope.
///
/// This struct is passed to service handlers after the `RequestLoop`
/// has verified the envelope signature. Handlers use this for:
/// - Authorization checks via `casbin_subject()`
/// - Correlation via `request_id`
/// - Stream HMAC key derivation via `ephemeral_pubkey`
///
/// # Example
///
/// ```ignore
/// fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
///     // Check authorization
///     if !policy_manager.check(&ctx.casbin_subject(), "Model", "infer") {
///         return Err(anyhow!("unauthorized: {}", ctx.casbin_subject()));
///     }
///     // Process request...
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EnvelopeContext {
    /// Unique request ID for correlation and logging
    pub request_id: u64,

    /// Verified identity of the requester (after signature verification)
    pub identity: RequestIdentity,

    /// Ephemeral public key for stream HMAC derivation (if streaming)
    pub ephemeral_pubkey: Option<[u8; 32]>,

    /// User claims from envelope (protected by envelope signature)
    /// Already verified by envelope signature verification
    claims: Option<crate::auth::Claims>,
}

impl EnvelopeContext {
    /// Create context from a verified SignedEnvelope.
    ///
    /// This should only be called after signature verification succeeds.
    pub fn from_verified(envelope: &SignedEnvelope) -> Self {
        Self {
            request_id: envelope.request_id(),
            identity: envelope.identity().clone(),
            ephemeral_pubkey: envelope.ephemeral_pubkey().copied(),
            claims: envelope.envelope.claims.clone(),
        }
    }

    /// Get the namespaced Casbin subject for policy checks.
    ///
    /// Returns prefixed identities like `"local:alice"`, `"token:bob"`, etc.
    pub fn casbin_subject(&self) -> String {
        self.identity.casbin_subject()
    }

    /// Get the raw user string (without namespace prefix).
    pub fn user(&self) -> &str {
        self.identity.user()
    }

    /// Check if the identity is authenticated (not anonymous).
    pub fn is_authenticated(&self) -> bool {
        self.identity.is_authenticated()
    }

    /// Get user claims (if present in envelope)
    ///
    /// Claims are already verified by the envelope signature.
    pub fn claims(&self) -> Option<&crate::auth::Claims> {
        self.claims.as_ref()
    }

    /// Get user subject for Casbin checks (if claims present)
    pub fn user_subject(&self) -> Option<String> {
        self.claims.as_ref().map(super::super::auth::claims::Claims::casbin_subject)
    }

    /// Get effective subject (user if present, otherwise service identity)
    pub fn effective_subject(&self) -> String {
        self.user_subject().unwrap_or_else(|| self.casbin_subject())
    }

    /// Check if request has user context
    pub fn has_user_context(&self) -> bool {
        self.claims.is_some()
    }

    /// Get user scopes (if claims present)
    pub fn user_scopes(&self) -> Option<Vec<String>> {
        self.claims.as_ref().map(|c| c.scopes.iter().map(super::super::auth::scope::Scope::to_string).collect())
    }

    /// Check if user has specific scope (if claims present)
    pub fn has_scope(&self, required_scope: &crate::auth::Scope) -> bool {
        self.claims
            .as_ref()
            .map(|c| c.has_scope(required_scope))
            .unwrap_or(false)
    }

    /// Get the client's ephemeral public key for DH key exchange (if provided).
    ///
    /// This is used for deriving stream keys in E2E authenticated streaming.
    /// Returns None if the client didn't include an ephemeral pubkey.
    pub fn ephemeral_pubkey(&self) -> Option<&[u8]> {
        self.ephemeral_pubkey.as_ref().map(<[u8; 32]>::as_slice)
    }
}

/// Trait for ZMQ-based REQ/REP services.
///
/// This is the unified trait for services that:
/// 1. Handle requests via `handle_request()`
/// 2. Are automatically spawnable (blanket `impl Spawnable for S: ZmqService`)
///
/// Services include their infrastructure (context, transport, verifying key) so they
/// can be spawned directly without wrapping.
///
/// # Security Model
///
/// The `RequestLoop` unwraps and verifies `SignedEnvelope` before calling handlers.
/// Handlers receive `EnvelopeContext` with verified identity - no need to re-verify.
///
/// # Example
///
/// ```ignore
/// pub struct MyService {
///     // Business logic
///     data: MyData,
///     // Infrastructure (required for spawning)
///     context: Arc<zmq::Context>,
///     transport: TransportConfig,
///     verifying_key: VerifyingKey,
/// }
///
/// impl ZmqService for MyService {
///     fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
///         // ctx.identity is already verified
///         info!("Request from {} (id={})", ctx.casbin_subject(), ctx.request_id);
///         Ok(vec![])
///     }
///
///     fn name(&self) -> &str { "my-service" }
///     fn context(&self) -> &Arc<zmq::Context> { &self.context }
///     fn transport(&self) -> &TransportConfig { &self.transport }
///     fn verifying_key(&self) -> VerifyingKey { self.verifying_key }
/// }
///
/// // MyService is now automatically Spawnable!
/// manager.spawn(Box::new(my_service)).await?;
/// ```
pub trait ZmqService: Send + Sync + 'static {
    /// Process a request and return a response.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Verified envelope context with identity
    /// * `payload` - Raw inner request bytes (Cap'n Proto encoded)
    ///
    /// Returns raw bytes for the response (Cap'n Proto encoded).
    ///
    /// This runs in a blocking task to avoid blocking the async runtime.
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>>;

    /// Service name (for logging and registry).
    fn name(&self) -> &str;

    /// ZMQ context for socket creation.
    fn context(&self) -> &Arc<zmq::Context>;

    /// Transport configuration (endpoint binding).
    fn transport(&self) -> &TransportConfig;

    /// Ed25519 signing key for signing responses.
    fn signing_key(&self) -> SigningKey;

    /// Ed25519 verifying key for envelope signature verification.
    fn verifying_key(&self) -> VerifyingKey {
        self.signing_key().verifying_key()
    }
}

/// REQ/REP message loop for ZmqService.
///
/// Runs the REQ/REP message loop as an async task with TMQ for I/O.
/// Uses proper async/await with epoll integration instead of blocking threads.
///
/// # Envelope Verification
///
/// The loop unwraps and verifies `SignedEnvelope` for every request:
/// 1. Deserialize `SignedEnvelope` from wire bytes
/// 2. Verify Ed25519 signature against server's public key
/// 3. Check nonce not replayed (replay protection)
/// 4. Extract `EnvelopeContext` and dispatch to handler
///
/// Invalid/unsigned requests are rejected with an error response.
///
/// # Usage
///
/// Typically used internally by the blanket `Spawnable` impl for `ZmqService`.
/// Direct usage is for advanced cases only.
pub struct RequestLoop {
    /// Transport configuration (supports SystemdFd for socket activation)
    transport: TransportConfig,
    /// ZMQ context for socket creation (required for inproc:// connectivity)
    context: Arc<zmq::Context>,
    /// Server's Ed25519 verifying key for signature verification
    server_pubkey: VerifyingKey,
    /// Server's Ed25519 signing key for signing responses
    signing_key: SigningKey,
    /// Nonce cache for replay protection
    nonce_cache: Arc<InMemoryNonceCache>,
}

impl RequestLoop {
    /// Create a new request loop bound to the given transport.
    ///
    /// # Arguments
    ///
    /// * `transport` - Transport configuration (supports SystemdFd for socket activation)
    /// * `context` - ZMQ context (must be shared for inproc:// to work)
    /// * `signing_key` - Server's Ed25519 signing key for signing responses
    pub fn new(transport: TransportConfig, context: Arc<zmq::Context>, signing_key: SigningKey) -> Self {
        Self {
            transport,
            context,
            server_pubkey: signing_key.verifying_key(),
            signing_key,
            nonce_cache: Arc::new(InMemoryNonceCache::new()),
        }
    }

    /// Create a service runner with a shared nonce cache.
    ///
    /// Use this when multiple services should share replay protection state.
    pub fn with_nonce_cache(
        transport: TransportConfig,
        context: Arc<zmq::Context>,
        signing_key: SigningKey,
        nonce_cache: Arc<InMemoryNonceCache>,
    ) -> Self {
        Self {
            transport,
            context,
            server_pubkey: signing_key.verifying_key(),
            signing_key,
            nonce_cache,
        }
    }

    /// Run the service as an async task
    ///
    /// This spawns an async task that:
    /// 1. Creates a REP socket and binds using TransportConfig (supports SystemdFd)
    /// 2. Converts to TMQ for async I/O via epoll
    /// 3. Loops receiving and verifying `SignedEnvelope` requests
    /// 4. Dispatches to handler with verified `EnvelopeContext`
    ///
    /// Handler execution runs in spawn_blocking to avoid blocking the runtime.
    /// Waits for socket binding to complete before returning.
    /// Returns a handle that can be used to stop the service.
    pub async fn run<S: ZmqService>(self, service: S) -> Result<ServiceHandle> {
        let transport = self.transport.clone();
        let context = Arc::clone(&self.context);
        let server_pubkey = self.server_pubkey;
        let signing_key = self.signing_key.clone();
        let nonce_cache = Arc::clone(&self.nonce_cache);
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = Arc::clone(&shutdown);

        // Create oneshot channel for ready signal
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

        // Wrap service in Arc for sharing with spawn_blocking handlers
        let service = Arc::new(service);

        // Spawn async task (not spawn_blocking - TMQ provides async I/O)
        let handle = tokio::spawn(async move {
            if let Err(e) = Self::service_loop_async(
                transport,
                context,
                service,
                server_pubkey,
                signing_key,
                nonce_cache,
                shutdown_clone,
                Some(ready_tx),
            )
            .await
            {
                error!("Service loop error: {}", e);
            }
        });

        // Wait for the socket to bind (or get bind error)
        ready_rx
            .await
            .map_err(|_| anyhow!("Service task exited before signaling ready"))??;

        Ok(ServiceHandle {
            task: Some(handle),
            shutdown: Some(shutdown),
        })
    }

    /// Async service loop using TMQ
    ///
    /// The `ready_tx` channel signals when the socket is bound and ready.
    /// Uses raw ZMQ socket creation + TransportConfig::bind() for SystemdFd support,
    /// then converts to TMQ via from_zmq_socket for async I/O.
    async fn service_loop_async<S: ZmqService>(
        transport: TransportConfig,
        context: Arc<zmq::Context>,
        service: Arc<S>,
        server_pubkey: VerifyingKey,
        signing_key: SigningKey,
        nonce_cache: Arc<InMemoryNonceCache>,
        shutdown: Arc<Notify>,
        ready_tx: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    ) -> Result<()> {
        let endpoint = transport.zmq_endpoint();

        // Create raw ZMQ REP socket
        let mut socket = match context.socket(zmq::REP) {
            Ok(s) => s,
            Err(e) => {
                let err = anyhow!("failed to create REP socket: {}", e);
                if let Some(tx) = ready_tx {
                    let _ = tx.send(Err(anyhow!("{}", err)));
                }
                return Err(err);
            }
        };
        socket.set_linger(0).ok();

        // Bind using TransportConfig - handles SystemdFd via set_use_fd()
        if let Err(e) = transport.bind(&mut socket) {
            let err = anyhow!("failed to bind to {}: {}", endpoint, e);
            if let Some(tx) = ready_tx {
                let _ = tx.send(Err(anyhow!("{}", err)));
            }
            return Err(err);
        }

        // Convert to TMQ for async I/O (from_zmq_socket makes no assumptions about state)
        let mut receiver: RequestReceiver = match RequestReceiver::from_zmq_socket(socket) {
            Ok(r) => r,
            Err(e) => {
                let err = anyhow!("failed to wrap socket in TMQ: {}", e);
                if let Some(tx) = ready_tx {
                    let _ = tx.send(Err(anyhow!("{}", err)));
                }
                return Err(err);
            }
        };

        info!("{} service bound to {}", service.name(), endpoint);

        // Signal ready AFTER socket is bound
        if let Some(tx) = ready_tx {
            if tx.send(Ok(())).is_err() {
                warn!(
                    "{} service ready signal dropped - receiver gone",
                    service.name()
                );
            }
        }

        loop {
            // Use select! for clean shutdown without polling timeouts
            tokio::select! {
                biased;

                // Check for shutdown signal
                _ = shutdown.notified() => {
                    debug!("{} service received shutdown signal", service.name());
                    break;
                }

                // Receive request via TMQ async I/O
                result = receiver.recv() => {
                    let (request_msg, sender) = result
                        .map_err(|e| anyhow!("recv error: {}", e))?;

                    // Extract bytes from multipart message
                    let request: Vec<u8> = request_msg
                        .into_iter()
                        .flat_map(|frame| frame.to_vec())
                        .collect();

                    trace!(
                        "{} received request ({} bytes)",
                        service.name(),
                        request.len()
                    );

                    // Unwrap and verify SignedEnvelope
                    let (ctx, payload) = match crate::envelope::unwrap_envelope(&request, &server_pubkey, &*nonce_cache) {
                        Ok((ctx, payload)) => (ctx, payload),
                        Err(e) => {
                            warn!(
                                "{} envelope verification failed: {}",
                                service.name(),
                                e
                            );
                            // Send error response
                            let msg: Multipart = vec![vec![]].into();
                            receiver = sender
                                .send(msg)
                                .await
                                .map_err(|e| anyhow!("send error: {}", e))?;
                            continue;
                        }
                    };

                    let request_id = ctx.request_id;
                    debug!(
                        "{} verified request from {} (id={})",
                        service.name(),
                        ctx.casbin_subject(),
                        request_id
                    );

                    // Process request in spawn_blocking (handler may do blocking work)
                    let service_clone = Arc::clone(&service);
                    let response_payload = tokio::task::spawn_blocking(move || {
                        service_clone.handle_request(&ctx, &payload)
                    })
                    .await
                    .map_err(|e| anyhow!("spawn_blocking join error: {}", e))?;

                    let response_payload = match response_payload {
                        Ok(resp) => resp,
                        Err(e) => {
                            error!("{} request handling error: {}", service.name(), e);
                            vec![] // Error response
                        }
                    };

                    // Wrap response in signed envelope and serialize (scoped for Send)
                    let response_bytes = {
                        use crate::envelope::ResponseEnvelope;
                        use capnp::message::Builder;
                        use capnp::serialize;

                        let signed_response = ResponseEnvelope::new_signed(
                            request_id,
                            response_payload,
                            &signing_key,
                        );

                        let mut message = Builder::new_default();
                        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
                        signed_response.write_to(&mut builder);

                        let mut bytes = Vec::new();
                        serialize::write_message(&mut bytes, &message)
                            .map_err(|e| anyhow!("Failed to serialize signed response: {}", e))?;
                        bytes
                    };

                    // Send signed response via TMQ async I/O
                    let msg: Multipart = vec![response_bytes].into();
                    receiver = sender
                        .send(msg)
                        .await
                        .map_err(|e| anyhow!("send error: {}", e))?;
                }
            }
        }

        info!("{} service stopped", service.name());
        Ok(())
    }
}

/// Handle for a running service
pub struct ServiceHandle {
    task: Option<tokio::task::JoinHandle<()>>,
    /// Shutdown signal using Notify (clean async shutdown)
    shutdown: Option<Arc<Notify>>,
}

impl ServiceHandle {
    /// Create a dummy handle for services that manage their own lifecycle
    pub fn dummy() -> Self {
        Self {
            task: None,
            shutdown: None,
        }
    }

    /// Create a handle from an existing task and shutdown signal.
    ///
    /// Used by ServiceSpawner when spawning Spawnable services.
    pub fn from_task(task: tokio::task::JoinHandle<()>, shutdown: Arc<Notify>) -> Self {
        Self {
            task: Some(task),
            shutdown: Some(shutdown),
        }
    }

    /// Stop the service gracefully
    ///
    /// Idempotent: subsequent calls are no-ops if already stopped.
    pub async fn stop(&mut self) {
        // Signal shutdown via Notify
        if let Some(shutdown) = &self.shutdown {
            shutdown.notify_one();
        }
        // Wait for task to complete
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }

    /// Check if the service is still running
    pub fn is_running(&self) -> bool {
        self.task.as_ref().map(|t| !t.is_finished()).unwrap_or(true)
    }
}

/// Authenticated ZMQ client with automatic request signing and response verification.
///
/// This is the unified client for all ZMQ-based services. All requests are
/// automatically wrapped in `SignedEnvelope` for authentication, and all
/// responses are cryptographically verified.
///
/// # E2E Authentication
///
/// - **Requests**: Automatically wrapped in `SignedEnvelope` (Ed25519 signed)
/// - **Responses**: Automatically verified against server's public key (Ed25519)
///
/// There is NO way to receive unverified response data from this client.
/// This prevents MITM attacks on response data (e.g., DH public keys).
///
/// # Resilience
///
/// Uses a persistent socket with ZMQ's built-in reliability features:
/// - `ZMQ_REQ_RELAXED`: Allows sending new requests after timeout (no stuck state)
/// - `ZMQ_REQ_CORRELATE`: Matches replies to requests automatically
/// - `ZMQ_RECONNECT_IVL/MAX`: Automatic reconnection with exponential backoff
///
/// # Usage
///
/// Use extension traits (`RegistryOps`, `InferenceOps`) to add service-specific
/// methods to this client:
///
/// ```ignore
/// use hyprstream_rpc::service::{ZmqClient};
///
/// let client = ZmqClient::new("inproc://hyprstream/registry", context, signing_key, server_verifying_key, identity);
/// let repos = client.list().await?;  // Extension trait method
/// ```
pub struct ZmqClient {
    /// ZMQ endpoint
    endpoint: String,
    /// ZMQ context for socket creation (required for inproc:// connectivity)
    context: Arc<zmq::Context>,
    /// Ed25519 signing key for request authentication
    signing_key: SigningKey,
    /// Server's Ed25519 verifying key for response verification (mandatory)
    server_verifying_key: VerifyingKey,
    /// Identity included in requests for authorization
    identity: RequestIdentity,
    /// Monotonic request ID counter
    request_id: AtomicU64,
    /// Persistent socket wrapped in TMQ RequestSender (protected by async mutex)
    /// Using Option to allow re-initialization on connection errors
    sender: tokio::sync::Mutex<Option<RequestSender>>,
}

/// Options for ZMQ RPC calls.
///
/// Consolidates all optional parameters for `call()` into a single struct.
/// Use `Default::default()` for basic calls, or builder-style methods for options.
///
/// # Example
/// ```ignore
/// // Basic call (no options)
/// client.call(payload, CallOptions::default()).await?;
///
/// // With timeout
/// client.call(payload, CallOptions::default().timeout(5000)).await?;
///
/// // With user claims
/// client.call(payload, CallOptions::default().claims(user_claims)).await?;
///
/// // With ephemeral pubkey for E2E authenticated streaming
/// client.call(payload, CallOptions::default().ephemeral_pubkey(pubkey)).await?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    /// Explicit timeout in milliseconds (defaults to 30s)
    pub timeout_ms: Option<i32>,
    /// Requested lifetime for this request
    pub requested_lifetime_ms: Option<i32>,
    /// User authorization claims (for authenticated calls)
    pub claims: Option<crate::auth::Claims>,
    /// Ephemeral Ristretto255 public key for stream HMAC key derivation
    pub ephemeral_pubkey: Option<[u8; 32]>,
}

impl CallOptions {
    /// Set explicit timeout in milliseconds.
    pub fn timeout(mut self, ms: i32) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// Set requested lifetime in milliseconds.
    pub fn lifetime(mut self, ms: i32) -> Self {
        self.requested_lifetime_ms = Some(ms);
        self
    }

    /// Set user authorization claims.
    pub fn claims(mut self, claims: crate::auth::Claims) -> Self {
        self.claims = Some(claims);
        self
    }

    /// Set ephemeral public key for E2E authenticated streaming.
    pub fn ephemeral_pubkey(mut self, pubkey: [u8; 32]) -> Self {
        self.ephemeral_pubkey = Some(pubkey);
        self
    }

    /// Create options from EnvelopeContext (preserves user claims).
    pub fn from_context(ctx: &EnvelopeContext) -> Self {
        Self {
            claims: ctx.claims().cloned(),
            ..Default::default()
        }
    }
}

/// Calculate ZMQ timeout using min() of multiple constraints.
///
/// Returns the smaller of:
/// - Explicit timeout parameter (if provided)
/// - Requested lifetime (if provided)
/// - JWT remaining time (with 1s safety buffer)
/// - Default timeout (30 seconds)
///
/// This ensures:
/// - We don't wait past JWT expiration (security)
/// - We respect service-specific timeout requirements (practicality)
/// - We respect user-requested lifetime (quality of service)
/// - We don't have excessively long timeouts (practicality)
fn calculate_timeout(
    explicit_timeout: Option<i32>,
    requested_lifetime_ms: Option<i32>,
    claims: Option<&crate::auth::Claims>,
) -> i32 {
    const DEFAULT_TIMEOUT_MS: i32 = 30000; // 30 seconds
    const SAFETY_BUFFER_MS: i64 = 1000; // 1 second buffer

    // Start with explicit timeout or default
    let mut timeout = explicit_timeout.unwrap_or(DEFAULT_TIMEOUT_MS);

    // Apply requested lifetime constraint (if provided)
    if let Some(lifetime) = requested_lifetime_ms {
        timeout = timeout.min(lifetime);
    }

    // Apply Claims expiration constraint (if provided)
    if let Some(claims) = claims {
        let now = Utc::now().timestamp();
        let remaining_ms = (claims.exp - now) * 1000 - SAFETY_BUFFER_MS;

        if remaining_ms > 0 {
            timeout = timeout.min(remaining_ms as i32);
        } else {
            warn!("Claims have expired or will expire immediately, using minimal timeout");
            return 100; // 100ms minimal timeout
        }
    }

    timeout
}

impl ZmqClient {
    /// Create a new client with signing credentials and server verification key.
    ///
    /// Creates a persistent socket with ZMQ's reliability options enabled:
    /// - `ZMQ_REQ_RELAXED`: Don't get stuck waiting for replies
    /// - `ZMQ_REQ_CORRELATE`: Match replies to requests
    /// - Auto-reconnect with exponential backoff (100ms to 5s)
    ///
    /// # Arguments
    /// * `endpoint` - ZMQ endpoint (e.g., `inproc://hyprstream/registry`)
    /// * `context` - ZMQ context (must be shared for inproc:// to work)
    /// * `signing_key` - Ed25519 signing key for request authentication
    /// * `server_verifying_key` - Server's Ed25519 public key for response verification
    /// * `identity` - Identity to include in requests (for authorization)
    ///
    /// # Security
    ///
    /// The `server_verifying_key` is MANDATORY. All responses are verified against
    /// this key before being returned. There is no way to bypass verification.
    pub fn new(
        endpoint: &str,
        context: Arc<zmq::Context>,
        signing_key: SigningKey,
        server_verifying_key: VerifyingKey,
        identity: RequestIdentity,
    ) -> Self {
        // Socket is lazily initialized on first call to allow construction without errors
        Self {
            endpoint: endpoint.to_owned(),
            context,
            signing_key,
            server_verifying_key,
            identity,
            request_id: AtomicU64::new(1),
            sender: tokio::sync::Mutex::new(None),
        }
    }

    /// Create and configure a new REQ socket with resilience options.
    fn create_socket(&self) -> Result<zmq::Socket> {
        let socket = self.context.socket(zmq::REQ)
            .map_err(|e| anyhow!("Failed to create REQ socket: {}", e))?;

        // Enable relaxed REQ mode - allows sending new request after timeout
        // without getting stuck in send/recv state machine
        socket.set_req_relaxed(true)
            .map_err(|e| anyhow!("Failed to set REQ_RELAXED: {}", e))?;

        // Enable request correlation - ZMQ automatically matches replies to requests
        // Required when using REQ_RELAXED to handle out-of-order/late replies
        socket.set_req_correlate(true)
            .map_err(|e| anyhow!("Failed to set REQ_CORRELATE: {}", e))?;

        // Auto-reconnect settings for resilience
        socket.set_reconnect_ivl(100)  // Start with 100ms
            .map_err(|e| anyhow!("Failed to set reconnect interval: {}", e))?;
        socket.set_reconnect_ivl_max(5000)  // Max 5s with exponential backoff
            .map_err(|e| anyhow!("Failed to set max reconnect interval: {}", e))?;

        // Don't block on close
        socket.set_linger(0)
            .map_err(|e| anyhow!("Failed to set linger: {}", e))?;

        // Connect to endpoint
        socket.connect(&self.endpoint)
            .map_err(|e| anyhow!("Failed to connect to {}: {}", self.endpoint, e))?;

        debug!("ZmqClient connected to {} with REQ_RELAXED+REQ_CORRELATE", self.endpoint);

        Ok(socket)
    }

    /// Get or create the RequestSender, reconnecting if necessary.
    async fn get_or_create_sender(&self) -> Result<RequestSender> {
        let mut guard = self.sender.lock().await;

        if let Some(sender) = guard.take() {
            return Ok(sender);
        }

        // Create new socket and wrap with TMQ
        let socket = self.create_socket()?;
        let sender = RequestSender::from_zmq_socket(socket)
            .map_err(|e| anyhow!("Failed to wrap socket in TMQ: {}", e))?;

        Ok(sender)
    }

    /// Store the sender for reuse after a successful call.
    async fn store_sender(&self, sender: RequestSender) {
        let mut guard = self.sender.lock().await;
        *guard = Some(sender);
    }

    /// Get the next request ID (monotonically increasing).
    pub fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the endpoint this client is connected to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Get the identity used for requests.
    pub fn identity(&self) -> &RequestIdentity {
        &self.identity
    }

    /// Get the signing key.
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    /// Get the server's verifying key for response verification.
    pub fn server_verifying_key(&self) -> &VerifyingKey {
        &self.server_verifying_key
    }

    /// Get the ZMQ context.
    pub fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    /// Sign and send a request, verify and return the response.
    ///
    /// # E2E Authentication
    ///
    /// - **Requests**: Automatically wrapped in `SignedEnvelope` (Ed25519 signed)
    /// - **Responses**: Automatically verified against server's public key (Ed25519)
    ///
    /// There is NO way to receive unverified response data. If signature
    /// verification fails, an error is returned. This prevents MITM attacks.
    ///
    /// Uses a persistent socket with automatic reconnection. If the service
    /// is temporarily unavailable, ZMQ will automatically reconnect when it
    /// becomes available.
    ///
    /// # Arguments
    /// * `payload` - Request payload bytes
    /// * `opts` - Call options (timeout, claims, ephemeral_pubkey)
    ///
    /// # Returns
    /// The verified response payload bytes. The outer envelope signature has
    /// been verified and stripped - only the inner payload is returned.
    ///
    /// # Resilience
    /// - Socket auto-reconnects if connection is lost
    /// - REQ_RELAXED prevents getting stuck if reply is lost
    /// - On timeout, socket is reset and will be recreated on next call
    ///
    /// # Example
    /// ```ignore
    /// // Basic call
    /// client.call(payload, CallOptions::default()).await?;
    ///
    /// // With timeout
    /// client.call(payload, CallOptions::default().timeout(5000)).await?;
    ///
    /// // With user claims
    /// client.call(payload, CallOptions::default().claims(user_claims)).await?;
    ///
    /// // With ephemeral pubkey for E2E streaming
    /// client.call(payload, CallOptions::default().ephemeral_pubkey(pubkey)).await?;
    /// ```
    pub async fn call(&self, payload: Vec<u8>, opts: CallOptions) -> Result<Vec<u8>> {
        let signed = self.sign_request(payload, &opts)?;
        let timeout = calculate_timeout(opts.timeout_ms, opts.requested_lifetime_ms, opts.claims.as_ref());

        trace!(
            "ZmqClient sending {} bytes to {} (timeout: {}ms)",
            signed.len(),
            self.endpoint,
            timeout
        );

        // Get or create the persistent sender
        let sender = self.get_or_create_sender().await?;

        // TMQ REQ pattern: send returns RequestReceiver
        let receiver = sender
            .send(tmq::Multipart::from(vec![signed]))
            .await
            .map_err(|e| anyhow!("Failed to send request to {}: {}", self.endpoint, e))?;

        // Receive response with timeout
        // Use tokio::time::timeout since ZMQ socket timeout may not work with TMQ async
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(timeout as u64),
            receiver.recv()
        ).await;

        match result {
            Ok(Ok((response, sender))) => {
                // Success - store sender for reuse
                self.store_sender(sender).await;

                // Extract bytes from multipart message
                let wire_bytes: Vec<u8> = response
                    .into_iter()
                    .flat_map(|frame| frame.to_vec())
                    .collect();

                trace!("ZmqClient received {} bytes from {}", wire_bytes.len(), self.endpoint);

                // MANDATORY: Verify response signature before returning
                // There is no way to bypass this - all responses must be verified
                let (_request_id, payload) = crate::envelope::unwrap_response(
                    &wire_bytes,
                    Some(&self.server_verifying_key),
                )?;

                trace!("ZmqClient verified response ({} bytes payload)", payload.len());
                Ok(payload)
            }
            Ok(Err(e)) => {
                // Recv error - don't store sender, will recreate on next call
                warn!("ZmqClient recv error from {}: {}", self.endpoint, e);
                Err(anyhow!("Failed to receive response from {}: {}", self.endpoint, e))
            }
            Err(_) => {
                // Timeout - don't store sender (it's stuck with the receiver)
                // REQ_RELAXED allows the next call to work with a fresh socket
                warn!("ZmqClient timeout after {}ms waiting for {}", timeout, self.endpoint);
                Err(anyhow!("Request to {} timed out after {}ms", self.endpoint, timeout))
            }
        }
    }

    /// Wrap a payload in a SignedEnvelope and serialize to bytes.
    fn sign_request(&self, payload: Vec<u8>, opts: &CallOptions) -> Result<Vec<u8>> {
        use capnp::message::Builder;
        use capnp::serialize;

        let mut envelope = RequestEnvelope::new(self.identity.clone(), payload);

        // Apply optional claims
        if let Some(ref claims) = opts.claims {
            envelope = envelope.with_claims(claims.clone());
        }

        // Apply optional ephemeral pubkey for E2E streaming
        if let Some(pubkey) = opts.ephemeral_pubkey {
            envelope = envelope.with_ephemeral_pubkey(pubkey);
        }

        let signed = SignedEnvelope::new_signed(envelope, &self.signing_key);

        let mut message = Builder::new_default();
        {
            let mut builder =
                message.init_root::<crate::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::generate_signing_keypair;

    /// Test service with infrastructure (new pattern)
    struct EchoService {
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    }

    impl EchoService {
        fn new(context: Arc<zmq::Context>, transport: TransportConfig, signing_key: SigningKey) -> Self {
            Self { context, transport, signing_key }
        }
    }

    impl ZmqService for EchoService {
        fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
            // Echo back the payload, but prepend the user
            let user = ctx.user();
            let mut response = format!("from {user}:").into_bytes();
            response.extend_from_slice(payload);
            Ok(response)
        }

        fn name(&self) -> &str {
            "echo"
        }

        fn context(&self) -> &Arc<zmq::Context> {
            &self.context
        }

        fn transport(&self) -> &TransportConfig {
            &self.transport
        }

        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }
    }

    #[tokio::test]
    async fn test_request_loop() -> Result<(), Box<dyn std::error::Error>> {
        let context = Arc::new(zmq::Context::new());
        let transport = TransportConfig::inproc("test-echo-service-rpc");
        let endpoint = transport.zmq_endpoint();

        // Generate keypair for this test (same key for client and server)
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Create service with infrastructure
        let service = EchoService::new(Arc::clone(&context), transport.clone(), signing_key.clone());

        // Start the service (waits for socket binding)
        let runner = RequestLoop::new(transport, Arc::clone(&context), signing_key.clone());
        let mut handle = runner.run(service).await?;

        // Use ZmqClient with server's verifying key for response verification
        let client = ZmqClient::new(&endpoint, context, signing_key, verifying_key, RequestIdentity::local());
        let response = client.call(b"hello".to_vec(), CallOptions::default()).await?;

        // Response should start with "from <user>:"
        let response_str = String::from_utf8_lossy(&response);
        assert!(
            response_str.contains("hello"),
            "Response should contain 'hello': {response_str}"
        );

        // Stop the service
        handle.stop().await;
        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_request_signature_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let context = Arc::new(zmq::Context::new());
        let transport = TransportConfig::inproc("test-invalid-req-sig-rpc");
        let endpoint = transport.zmq_endpoint();

        // Generate two keypairs - service uses one, client uses other
        let (server_signing_key, server_verifying_key) = generate_signing_keypair();
        let (client_signing_key, _client_verifying_key) = generate_signing_keypair();

        // Create service with server's key
        let service = EchoService::new(Arc::clone(&context), transport.clone(), server_signing_key.clone());

        // Start the service (waits for socket binding)
        let runner = RequestLoop::new(transport, Arc::clone(&context), server_signing_key);
        let mut handle = runner.run(service).await?;

        // Sign request with different key than service expects
        // But verify responses with server's key
        let client = ZmqClient::new(&endpoint, context, client_signing_key, server_verifying_key, RequestIdentity::local());
        let result = client
            .call(b"should fail".to_vec(), CallOptions::default())
            .await;

        // Request should be rejected by server (empty response or error)
        // The response is still signed, but empty
        match result {
            Ok(response) => {
                // Empty response from server means request was rejected
                assert!(
                    response.is_empty(),
                    "Invalid request signature should return empty response"
                );
            }
            Err(_) => {
                // Error is also acceptable (deserialization of empty response may fail)
            }
        }

        handle.stop().await;
        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_response_signature_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let context = Arc::new(zmq::Context::new());
        let transport = TransportConfig::inproc("test-invalid-resp-sig-rpc");
        let endpoint = transport.zmq_endpoint();

        // Generate two keypairs - server signs with one, client expects other
        let (server_signing_key, _server_verifying_key) = generate_signing_keypair();
        let (_different_signing_key, different_verifying_key) = generate_signing_keypair();

        // Create service with server's signing key
        let service = EchoService::new(Arc::clone(&context), transport.clone(), server_signing_key.clone());

        // Start the service (waits for socket binding)
        let runner = RequestLoop::new(transport, Arc::clone(&context), server_signing_key.clone());
        let mut handle = runner.run(service).await?;

        // Client expects responses signed by a DIFFERENT key than server uses
        // This simulates a MITM attack or misconfigured client
        let client = ZmqClient::new(&endpoint, context, server_signing_key, different_verifying_key, RequestIdentity::local());
        let result = client
            .call(b"should fail verification".to_vec(), CallOptions::default())
            .await;

        // Response signature verification should fail
        assert!(
            result.is_err(),
            "Response with wrong signature should be rejected: {result:?}"
        );

        handle.stop().await;
        Ok(())
    }
}
