//! Core service infrastructure for ZMQ-based services
//!
//! Provides the foundation for REQ/REP services and clients using ZMQ.
//! Uses TMQ for async I/O - services run as async tasks with proper epoll integration.
//!
//! # Envelope-Based Security
//!
//! All requests are wrapped in `SignedEnvelope` for authentication:
//! - Service infrastructure unwraps and verifies signatures before dispatching
//! - Handlers receive `EnvelopeContext` with verified identity
//! - Services use `ctx.casbin_subject()` for policy checks
//!
//! # Note
//!
//! The core types (`EnvelopeContext`, `ZmqService`, `ZmqClient`)
//! are now defined in `hyprstream-rpc` and re-exported here for backward compatibility.
//!
//! This module provides convenience constructors that use the global ZMQ context.

use crate::zmq::global_context;
use anyhow::Result;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::transport::TransportConfig;
use std::sync::Arc;

// Re-export core types from hyprstream-rpc
pub use hyprstream_rpc::service::{
    EnvelopeContext, ServiceRunner as ServiceRunnerBase, ZmqClient as ZmqClientBase,
    ZmqService,
};

/// REQ/REP service runner with global ZMQ context.
///
/// This is a convenience wrapper around `hyprstream_rpc::ServiceRunner` that
/// automatically uses the global ZMQ context for `inproc://` connectivity.
///
/// For direct context control, use `ServiceRunnerBase` from hyprstream-rpc.
pub struct ServiceRunner {
    inner: ServiceRunnerBase,
}

impl ServiceRunner {
    /// Create a new service runner bound to the given endpoint.
    ///
    /// Uses the global ZMQ context for `inproc://` connectivity.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - ZMQ endpoint (e.g., `inproc://hyprstream/registry`)
    /// * `server_pubkey` - Server's Ed25519 public key for verifying request signatures
    pub fn new(endpoint: &str, server_pubkey: VerifyingKey) -> Self {
        Self {
            inner: ServiceRunnerBase::new(
                TransportConfig::from_endpoint(endpoint),
                global_context(),
                server_pubkey,
            ),
        }
    }

    /// Create a new service runner bound to the given transport configuration.
    ///
    /// Uses the global ZMQ context for `inproc://` connectivity.
    /// Supports SystemdFd for socket activation.
    ///
    /// # Arguments
    ///
    /// * `transport` - Transport configuration (supports SystemdFd)
    /// * `server_pubkey` - Server's Ed25519 public key for verifying request signatures
    pub fn with_transport(transport: TransportConfig, server_pubkey: VerifyingKey) -> Self {
        Self {
            inner: ServiceRunnerBase::new(transport, global_context(), server_pubkey),
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
            inner: ServiceRunnerBase::with_nonce_cache(
                TransportConfig::from_endpoint(endpoint),
                global_context(),
                server_pubkey,
                nonce_cache,
            ),
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
        self.inner.run(service).await
    }
}

/// Authenticated ZMQ client with automatic request signing.
///
/// This is a convenience wrapper around `hyprstream_rpc::ZmqClient` that
/// automatically uses the global ZMQ context for `inproc://` connectivity.
///
/// For direct context control, use `ZmqClientBase` from hyprstream-rpc.
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
    inner: ZmqClientBase,
}

impl ZmqClient {
    /// Create a new client with signing credentials.
    ///
    /// Uses the global ZMQ context for `inproc://` connectivity.
    ///
    /// # Arguments
    /// * `endpoint` - ZMQ endpoint (e.g., `inproc://hyprstream/registry`)
    /// * `signing_key` - Ed25519 signing key for request authentication
    /// * `identity` - Identity to include in requests (for authorization)
    pub fn new(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            inner: ZmqClientBase::new(endpoint, global_context(), signing_key, identity),
        }
    }

    /// Get the next request ID (monotonically increasing).
    pub fn next_id(&self) -> u64 {
        self.inner.next_id()
    }

    /// Get the endpoint this client is connected to.
    pub fn endpoint(&self) -> &str {
        self.inner.endpoint()
    }

    /// Get the identity used for requests.
    pub fn identity(&self) -> &RequestIdentity {
        self.inner.identity()
    }

    /// Get the signing key.
    pub fn signing_key(&self) -> &SigningKey {
        self.inner.signing_key()
    }

    /// Sign and send a request.
    ///
    /// All requests are automatically wrapped in `SignedEnvelope`.
    /// This ensures every call is authenticated - no bypass possible.
    ///
    /// # Arguments
    /// * `payload` - Request payload bytes
    /// * `timeout_ms` - Optional explicit timeout in milliseconds (defaults to 30s)
    ///
    /// Uses TMQ with the global context for proper `inproc://` support.
    pub async fn call(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>> {
        self.inner.call(payload, timeout_ms).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::generate_signing_keypair;

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

    #[tokio::test]
    async fn test_service_runner() {
        let endpoint = "inproc://test-echo-service";

        // Generate keypair for this test
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Start the service (waits for socket binding)
        let runner = ServiceRunner::new(endpoint, verifying_key);
        let mut handle = runner.run(EchoService).await.expect("test: start service");

        // Use ZmqClient directly (handles signing internally)
        let client = ZmqClient::new(endpoint, signing_key, RequestIdentity::local());
        let response = client.call(b"hello".to_vec()).await.expect("test: call");

        // Response should start with "from <user>:"
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("hello"), "Response should contain 'hello': {}", response_str);

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
        let mut handle = runner.run(EchoService).await.expect("test: start service");

        // Sign with different key than service expects
        let client = ZmqClient::new(endpoint, signing_key, RequestIdentity::local());
        let response = client.call(b"should fail".to_vec()).await.expect("test: call");

        // Response should be empty (verification failure)
        assert!(response.is_empty(), "Invalid signature should return empty response");

        handle.stop().await;
    }
}
