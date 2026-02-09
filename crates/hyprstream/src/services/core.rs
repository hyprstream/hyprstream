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
//! - Services use `ctx.subject()` for policy checks and resource isolation
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

// Re-export core types from hyprstream-rpc
pub use hyprstream_rpc::service::{CallOptions, EnvelopeContext, ZmqClient as ZmqClientBase, ZmqService};

/// Authenticated ZMQ client with automatic request signing and response verification.
///
/// This is a convenience wrapper around `hyprstream_rpc::ZmqClient` that
/// automatically uses the global ZMQ context for `inproc://` connectivity.
///
/// For direct context control, use `ZmqClientBase` from hyprstream-rpc.
///
/// # E2E Authentication
///
/// - **Requests**: Automatically signed with Ed25519
/// - **Responses**: Automatically verified against server's public key
///
/// There is NO way to receive unverified response data.
///
/// # Usage
///
/// Use extension traits (`RegistryOps`, `InferenceOps`) to add service-specific
/// methods to this client:
///
/// ```ignore
/// use crate::services::{ZmqClient, RegistryOps};
///
/// let client = ZmqClient::new("inproc://hyprstream/registry", signing_key, server_verifying_key, identity);
/// let repos = client.list().await?;  // RegistryOps method
/// ```
pub struct ZmqClient {
    inner: ZmqClientBase,
}

impl ZmqClient {
    /// Create a new client with signing credentials and server verification key.
    ///
    /// Uses the global ZMQ context for `inproc://` connectivity.
    ///
    /// # Arguments
    /// * `endpoint` - ZMQ endpoint (e.g., `inproc://hyprstream/registry`)
    /// * `signing_key` - Ed25519 signing key for request authentication
    /// * `server_verifying_key` - Server's Ed25519 public key for response verification
    /// * `identity` - Identity to include in requests (for authorization)
    ///
    /// # Security
    ///
    /// The `server_verifying_key` is MANDATORY. All responses are verified
    /// against this key before being returned. There is no way to bypass verification.
    pub fn new(endpoint: &str, signing_key: SigningKey, server_verifying_key: VerifyingKey, identity: RequestIdentity) -> Self {
        Self {
            inner: ZmqClientBase::new(endpoint, global_context(), signing_key, server_verifying_key, identity),
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

    /// Get the server's verifying key for response verification.
    pub fn server_verifying_key(&self) -> &VerifyingKey {
        self.inner.server_verifying_key()
    }

    /// Sign and send a request.
    ///
    /// All requests are automatically wrapped in `SignedEnvelope`.
    /// This ensures every call is authenticated - no bypass possible.
    ///
    /// # Arguments
    /// * `payload` - Request payload bytes
    /// * `opts` - Call options (timeout, claims, ephemeral_pubkey)
    ///
    /// Uses TMQ with the global context for proper `inproc://` support.
    pub async fn call(&self, payload: Vec<u8>, opts: hyprstream_rpc::service::CallOptions) -> Result<Vec<u8>> {
        self.inner.call(payload, opts).await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::service::RequestLoop;
    use hyprstream_rpc::transport::TransportConfig;

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
            let mut response = format!("from {}:", user).into_bytes();
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
    async fn test_request_loop() {
        let transport = TransportConfig::inproc("test-echo-service-core");
        let endpoint = transport.zmq_endpoint();

        // Generate keypair for this test
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Create service with infrastructure
        let service = EchoService::new(global_context(), transport.clone(), signing_key.clone());

        // Start the service using RequestLoop from hyprstream-rpc
        let runner = RequestLoop::new(transport, global_context(), signing_key.clone());
        let mut handle = runner.run(service).await.expect("test: start service");

        // Use ZmqClient with server's verifying key for response verification
        let client = ZmqClient::new(&endpoint, signing_key, verifying_key, RequestIdentity::local());
        let response = client.call(b"hello".to_vec(), CallOptions::default()).await.expect("test: call");

        // Response should start with "from <user>:"
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("hello"), "Response should contain 'hello': {}", response_str);

        // Stop the service
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_invalid_request_signature_rejected() {
        let transport = TransportConfig::inproc("test-invalid-req-sig-core");
        let endpoint = transport.zmq_endpoint();

        // Generate two keypairs - service uses one, client uses other
        let (server_signing_key, server_verifying_key) = generate_signing_keypair();
        let (client_signing_key, _client_verifying_key) = generate_signing_keypair();

        // Create service with server's key
        let service = EchoService::new(global_context(), transport.clone(), server_signing_key.clone());

        // Start the service
        let runner = RequestLoop::new(transport, global_context(), server_signing_key);
        let mut handle = runner.run(service).await.expect("test: start service");

        // Sign request with different key than service expects
        // But verify responses with server's key
        let client = ZmqClient::new(&endpoint, client_signing_key, server_verifying_key, RequestIdentity::local());
        let result = client.call(b"should fail".to_vec(), CallOptions::default()).await;

        // Request should be rejected by server
        match result {
            Ok(response) => {
                assert!(response.is_empty(), "Invalid request signature should return empty response");
            }
            Err(_) => {
                // Error is also acceptable
            }
        }

        handle.stop().await;
    }
}
