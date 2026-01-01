//! Core service infrastructure for ZMQ-based services
//!
//! Provides the foundation for REQ/REP services and clients using ZMQ.
//! Uses TMQ for async I/O - services run as async tasks with proper epoll integration.
//!
//! # Envelope-Based Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - `ServiceRunner` unwraps and verifies signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.casbin_subject()` for policy checks

use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use hyprstream_rpc::prelude::*;
use std::sync::Arc;
use tmq::{reply, Multipart, RequestReceiver};
use tokio::sync::Notify;
use tracing::{debug, error, info, trace, warn};

/// Context extracted from a verified SignedEnvelope.
///
/// This struct is passed to service handlers after the `ServiceRunner`
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
}

/// Trait for ZMQ-based services
///
/// Implement this trait to create a service that handles REQ/REP requests.
/// Services run as async tasks with TMQ for I/O, handlers execute in spawn_blocking.
///
/// Requires `Sync` because the service is shared via `Arc` for concurrent handler execution.
///
/// # Security Model
///
/// The `ServiceRunner` unwraps and verifies `SignedEnvelope` before calling handlers.
/// Handlers receive `EnvelopeContext` with verified identity - no need to re-verify.
///
/// # Example
///
/// ```ignore
/// impl ZmqService for MyService {
///     fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
///         // ctx.identity is already verified
///         info!("Request from {} (id={})", ctx.casbin_subject(), ctx.request_id);
///         // ...
///     }
/// }
/// ```
pub trait ZmqService: Send + Sync + 'static {
    /// Process a request and return a response
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

    /// Get the name of this service (for logging)
    fn name(&self) -> &str;
}

/// REQ/REP service runner
///
/// Runs a ZmqService as an async task with TMQ for I/O.
/// Uses proper async/await with epoll integration instead of blocking threads.
///
/// # Envelope Verification
///
/// The runner unwraps and verifies `SignedEnvelope` for every request:
/// 1. Deserialize `SignedEnvelope` from wire bytes
/// 2. Verify Ed25519 signature against server's public key
/// 3. Check nonce not replayed (replay protection)
/// 4. Extract `EnvelopeContext` and dispatch to handler
///
/// Invalid/unsigned requests are rejected with an error response.
pub struct ServiceRunner {
    endpoint: String,
    /// Server's Ed25519 verifying key for signature verification
    server_pubkey: VerifyingKey,
    /// Nonce cache for replay protection
    nonce_cache: Arc<InMemoryNonceCache>,
}

impl ServiceRunner {
    /// Create a new service runner bound to the given endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - ZMQ endpoint (e.g., `inproc://hyprstream/registry`)
    /// * `server_pubkey` - Server's Ed25519 public key for verifying request signatures
    pub fn new(endpoint: &str, server_pubkey: VerifyingKey) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            server_pubkey,
            nonce_cache: Arc::new(InMemoryNonceCache::new()),
        }
    }

    /// Create a service runner with a shared nonce cache.
    ///
    /// Use this when multiple services should share replay protection state.
    pub fn with_nonce_cache(
        endpoint: &str,
        server_pubkey: VerifyingKey,
        nonce_cache: Arc<InMemoryNonceCache>,
    ) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            server_pubkey,
            nonce_cache,
        }
    }

    /// Run the service as an async task
    ///
    /// This spawns an async task that:
    /// 1. Creates a REP socket with TMQ (async I/O via epoll)
    /// 2. Binds to the endpoint
    /// 3. Loops receiving and verifying `SignedEnvelope` requests
    /// 4. Dispatches to handler with verified `EnvelopeContext`
    ///
    /// Handler execution runs in spawn_blocking to avoid blocking the runtime.
    /// Waits for socket binding to complete before returning.
    /// Returns a handle that can be used to stop the service.
    pub async fn run<S: ZmqService>(self, service: S) -> Result<ServiceHandle> {
        let endpoint = self.endpoint.clone();
        let server_pubkey = self.server_pubkey;
        let nonce_cache = self.nonce_cache.clone();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();

        // Create oneshot channel for ready signal
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

        // Wrap service in Arc for sharing with spawn_blocking handlers
        let service = Arc::new(service);

        // Spawn async task (not spawn_blocking - TMQ provides async I/O)
        let handle = tokio::spawn(async move {
            if let Err(e) =
                Self::service_loop_async(endpoint, service, server_pubkey, nonce_cache, shutdown_clone, Some(ready_tx)).await
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
    async fn service_loop_async<S: ZmqService>(
        endpoint: String,
        service: Arc<S>,
        server_pubkey: VerifyingKey,
        nonce_cache: Arc<InMemoryNonceCache>,
        shutdown: Arc<Notify>,
        ready_tx: Option<tokio::sync::oneshot::Sender<Result<()>>>,
    ) -> Result<()> {
        // Use the global context for inproc:// connectivity
        // All ZMQ sockets must use the same context for inproc to work
        let context = global_context();

        // Create REP socket with TMQ
        let mut receiver: RequestReceiver = match reply(&*context)
            .set_linger(0)
            .bind(&endpoint)
        {
            Ok(r) => r,
            Err(e) => {
                let err = anyhow!("failed to bind to {}: {}", endpoint, e);
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
                warn!("{} service ready signal dropped - receiver gone", service.name());
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
                    let (ctx, payload) = match Self::unwrap_envelope(&request, &server_pubkey, &*nonce_cache) {
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

                    debug!(
                        "{} verified request from {} (id={})",
                        service.name(),
                        ctx.casbin_subject(),
                        ctx.request_id
                    );

                    // Process request in spawn_blocking (handler may do blocking work)
                    let service_clone = service.clone();
                    let response = tokio::task::spawn_blocking(move || {
                        service_clone.handle_request(&ctx, &payload)
                    })
                    .await
                    .map_err(|e| anyhow!("spawn_blocking join error: {}", e))?;

                    let response = match response {
                        Ok(resp) => resp,
                        Err(e) => {
                            error!("{} request handling error: {}", service.name(), e);
                            vec![] // Error response
                        }
                    };

                    // Send response via TMQ async I/O
                    let msg: Multipart = vec![response].into();
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

    /// Unwrap and verify a SignedEnvelope from wire bytes.
    ///
    /// Returns the verified context and inner payload on success.
    fn unwrap_envelope(
        request: &[u8],
        server_pubkey: &VerifyingKey,
        nonce_cache: &dyn NonceCache,
    ) -> Result<(EnvelopeContext, Vec<u8>)> {
        use capnp::serialize;

        // Deserialize SignedEnvelope from Cap'n Proto
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(request),
            capnp::message::ReaderOptions::default(),
        )?;
        let signed_reader = reader.get_root::<hyprstream_rpc::common_capnp::signed_envelope::Reader>()?;
        let signed = SignedEnvelope::read_from(signed_reader)?;

        // Verify signature and replay protection
        signed.verify(server_pubkey, nonce_cache)?;

        // Extract context and payload
        let ctx = EnvelopeContext::from_verified(&signed);
        let payload = signed.payload().to_vec();

        Ok((ctx, payload))
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

    /// Stop the service gracefully
    pub async fn stop(self) {
        // Signal shutdown via Notify
        if let Some(shutdown) = &self.shutdown {
            shutdown.notify_one();
        }
        // Wait for task to complete
        if let Some(task) = self.task {
            let _ = task.await;
        }
    }

    /// Check if the service is still running
    pub fn is_running(&self) -> bool {
        self.task.as_ref().map(|t| !t.is_finished()).unwrap_or(true)
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Authenticated ZMQ client with automatic request signing.
///
/// This is the unified client for all ZMQ-based services. All requests are
/// automatically wrapped in `SignedEnvelope` for authentication.
///
/// Uses the global ZMQ context for `inproc://` compatibility with services.
///
/// # Usage
///
/// Use extension traits (`RegistryOps`, `InferenceOps`) to add service-specific
/// methods to this client:
///
/// ```ignore
/// use crate::services::{ZmqClient, RegistryOps};
///
/// let client = ZmqClient::new("inproc://hyprstream/registry", signing_key, identity);
/// let repos = client.list().await?;  // RegistryOps method
/// ```
pub struct ZmqClient {
    /// ZMQ endpoint
    endpoint: String,
    /// Ed25519 signing key for request authentication
    signing_key: SigningKey,
    /// Identity included in requests for authorization
    identity: RequestIdentity,
    /// Monotonic request ID counter
    request_id: AtomicU64,
}

impl ZmqClient {
    /// Create a new client with signing credentials.
    ///
    /// # Arguments
    /// * `endpoint` - ZMQ endpoint (e.g., `inproc://hyprstream/registry`)
    /// * `signing_key` - Ed25519 signing key for request authentication
    /// * `identity` - Identity to include in requests (for authorization)
    pub fn new(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            signing_key,
            identity,
            request_id: AtomicU64::new(1),
        }
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

    /// Sign and send a request.
    ///
    /// All requests are automatically wrapped in `SignedEnvelope`.
    /// This ensures every call is authenticated - no bypass possible.
    ///
    /// Uses TMQ with the global context for proper `inproc://` support.
    pub async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        let signed = self.sign_request(payload)?;

        // Use global context for inproc:// connectivity
        let context = global_context();

        // Create REQ socket using TMQ for proper async I/O
        let socket = tmq::request(&*context)
            .connect(&self.endpoint)
            .map_err(|e| anyhow!("Failed to connect to {}: {}", self.endpoint, e))?;

        trace!("ZmqClient sending {} bytes to {}", signed.len(), self.endpoint);

        // TMQ REQ pattern: send returns RequestReceiver
        let receiver = socket
            .send(tmq::Multipart::from(vec![signed]))
            .await
            .map_err(|e| anyhow!("Failed to send request: {}", e))?;

        // Receive response
        let (response, _sender) = receiver
            .recv()
            .await
            .map_err(|e| anyhow!("Failed to receive response: {}", e))?;

        // Extract bytes from multipart message
        let bytes: Vec<u8> = response
            .into_iter()
            .flat_map(|frame| frame.to_vec())
            .collect();

        trace!("ZmqClient received {} bytes", bytes.len());

        Ok(bytes)
    }

    /// Wrap a payload in a SignedEnvelope and serialize to bytes.
    fn sign_request(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        use capnp::message::Builder;
        use capnp::serialize;

        let envelope = RequestEnvelope::new(self.identity.clone(), payload);
        let signed = SignedEnvelope::new_signed(envelope, &self.signing_key);

        let mut message = Builder::new_default();
        {
            let mut builder =
                message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }
}

/// Client for REQ/REP services
///
/// Note: ServiceClient is not thread-safe because ZMQ sockets are not thread-safe.
/// Create a new client for each thread/task that needs to communicate with the service.
pub struct ServiceClient {
    socket: zmq::Socket,
    endpoint: String,
}

impl ServiceClient {
    /// Connect to a service at the given endpoint
    pub fn connect(endpoint: &str) -> Result<Self> {
        let ctx = global_context();

        let socket = ctx
            .socket(zmq::REQ)
            .map_err(|e| anyhow!("failed to create REQ socket: {}", e))?;

        socket
            .connect(endpoint)
            .map_err(|e| anyhow!("failed to connect to {}: {}", endpoint, e))?;

        // Set LINGER to 0 for immediate close without blocking
        socket
            .set_linger(0)
            .map_err(|e| anyhow!("failed to set socket linger: {}", e))?;

        debug!("ServiceClient connected to {}", endpoint);

        Ok(Self {
            socket,
            endpoint: endpoint.to_string(),
        })
    }

    /// Send a request and wait for a response
    ///
    /// This is a blocking call.
    pub fn call(&self, request: &[u8]) -> Result<Vec<u8>> {
        self.socket
            .send(request, 0)
            .map_err(|e| anyhow!("failed to send request: {}", e))?;

        let response = self
            .socket
            .recv_bytes(0)
            .map_err(|e| anyhow!("failed to receive response: {}", e))?;

        trace!(
            "ServiceClient {} call completed ({} bytes response)",
            self.endpoint,
            response.len()
        );

        Ok(response)
    }

    /// Send a request and wait for a response with timeout
    pub fn call_with_timeout(&self, request: &[u8], timeout_ms: i32) -> Result<Vec<u8>> {
        // Set receive timeout
        self.socket
            .set_rcvtimeo(timeout_ms)
            .map_err(|e| anyhow!("failed to set timeout: {}", e))?;

        self.socket
            .send(request, 0)
            .map_err(|e| anyhow!("failed to send request: {}", e))?;

        let response = self
            .socket
            .recv_bytes(0)
            .map_err(|e| anyhow!("failed to receive response (timeout={}ms): {}", timeout_ms, e))?;

        // Reset timeout
        let _ = self.socket.set_rcvtimeo(-1);

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::transport::AsyncServiceClient;

    struct EchoService;

    impl ZmqService for EchoService {
        fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
            // Echo back the payload, but prepend the user
            let user = ctx.user();
            let mut response = format!("from {}:", user).into_bytes();
            response.extend_from_slice(payload);
            Ok(response)
        }

        fn name(&self) -> &str {
            "echo"
        }
    }

    /// Create a signed envelope with the given payload
    fn create_signed_request(signing_key: &SigningKey, payload: &[u8]) -> Vec<u8> {
        use capnp::message::Builder;
        use capnp::serialize;

        let envelope = RequestEnvelope::local(payload.to_vec());
        let signed = SignedEnvelope::new_signed(envelope, signing_key);

        let mut message = Builder::new_default();
        {
            let mut builder = message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).expect("test: serialize message");
        bytes
    }

    #[tokio::test]
    async fn test_service_runner() {
        // Use a unique endpoint for this test
        let endpoint = "inproc://test-echo-service";

        // Generate keypair for this test
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Start the service (waits for socket binding)
        let runner = ServiceRunner::new(endpoint, verifying_key);
        let handle = runner.run(EchoService).await.expect("test: start service");

        // Create signed request
        let request = create_signed_request(&signing_key, b"hello");

        // Test the service
        let response = tokio::task::spawn_blocking(move || {
            let client = ServiceClient::connect(endpoint).expect("test: connect client");
            client.call(&request)
        })
        .await
        .expect("test: spawn blocking");

        let response = response.expect("test: service call");
        // Response should start with "from <user>:"
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("hello"), "Response should contain 'hello': {}", response_str);

        // Stop the service
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_async_client() {
        let endpoint = "inproc://test-async-echo";

        // Generate keypair for this test
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Start the service (waits for socket binding)
        let runner = ServiceRunner::new(endpoint, verifying_key);
        let handle = runner.run(EchoService).await.expect("test: start service");

        // Create signed request
        let request = create_signed_request(&signing_key, b"async hello");

        // Test async client
        let client = AsyncServiceClient::new(endpoint);
        let response = client.call(request).await.expect("test: async call");
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("async hello"), "Response should contain 'async hello': {}", response_str);

        // Stop the service
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_invalid_signature_rejected() {
        let endpoint = "inproc://test-invalid-sig";

        // Generate two keypairs - use one to sign, verify with other
        let (signing_key, _) = generate_signing_keypair();
        let (_, wrong_verifying_key) = generate_signing_keypair();

        // Start the service with wrong key (waits for socket binding)
        let runner = ServiceRunner::new(endpoint, wrong_verifying_key);
        let handle = runner.run(EchoService).await.expect("test: start service");

        // Create signed request with different key
        let request = create_signed_request(&signing_key, b"should fail");

        // Test the service - should get empty response (error)
        let response = tokio::task::spawn_blocking(move || {
            let client = ServiceClient::connect(endpoint).expect("test: connect client");
            client.call(&request)
        })
        .await
        .expect("test: spawn blocking");

        // Response should be empty (verification failure)
        assert!(response.expect("test: get response").is_empty(), "Invalid signature should return empty response");

        handle.stop().await;
    }
}
