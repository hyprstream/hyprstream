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

// Re-export core types from hyprstream-rpc
pub use hyprstream_rpc::service::{CallOptions, EnvelopeContext, ZmqClient as ZmqClientBase, ZmqService};

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
    /// * `opts` - Call options (timeout, claims, ephemeral_pubkey)
    ///
    /// Uses TMQ with the global context for proper `inproc://` support.
    pub async fn call(&self, payload: Vec<u8>, opts: hyprstream_rpc::service::CallOptions) -> Result<Vec<u8>> {
        self.inner.call(payload, opts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::generate_signing_keypair;

    /// Test service with infrastructure (new pattern)
    struct EchoService {
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        verifying_key: VerifyingKey,
    }

    impl EchoService {
        fn new(context: Arc<zmq::Context>, transport: TransportConfig, verifying_key: VerifyingKey) -> Self {
            Self { context, transport, verifying_key }
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

        fn verifying_key(&self) -> VerifyingKey {
            self.verifying_key
        }
    }

    #[tokio::test]
    async fn test_request_loop() {
        let transport = TransportConfig::inproc("test-echo-service-core");
        let endpoint = transport.zmq_endpoint();

        // Generate keypair for this test
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Create service with infrastructure
        let service = EchoService::new(global_context(), transport.clone(), verifying_key);

        // Start the service using RequestLoop from hyprstream-rpc
        let runner = RequestLoopBase::new(transport, global_context(), verifying_key);
        let mut handle = runner.run(service).await.expect("test: start service");

        // Use ZmqClient directly (handles signing internally)
        let client = ZmqClient::new(&endpoint, signing_key, RequestIdentity::local());
        let response = client.call(b"hello".to_vec(), None).await.expect("test: call");

        // Response should start with "from <user>:"
        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains("hello"), "Response should contain 'hello': {}", response_str);

        // Stop the service
        handle.stop().await;
    }

    #[tokio::test]
    async fn test_invalid_signature_rejected() {
        let transport = TransportConfig::inproc("test-invalid-sig-core");
        let endpoint = transport.zmq_endpoint();

        // Generate two keypairs - use one to sign, verify with other
        let (signing_key, _) = generate_signing_keypair();
        let (_, wrong_verifying_key) = generate_signing_keypair();

        // Create service (verifying_key here doesn't matter - RequestLoop uses its own)
        let service = EchoService::new(global_context(), transport.clone(), wrong_verifying_key);

        // Start the service with wrong key
        let runner = RequestLoopBase::new(transport, global_context(), wrong_verifying_key);
        let mut handle = runner.run(service).await.expect("test: start service");

        // Sign with different key than service expects
        let client = ZmqClient::new(&endpoint, signing_key, RequestIdentity::local());
        let response = client.call(b"should fail".to_vec(), None).await.expect("test: call");

        // Response should be empty (verification failure)
        assert!(response.is_empty(), "Invalid signature should return empty response");

        handle.stop().await;
    }
}
