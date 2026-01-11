//! Policy service for authorization checks over ZMQ
//!
//! Wraps PolicyManager and exposes it as a ZmqService.
//! Runs on multi-threaded runtime where block_in_place works.
//!
//! # Architecture
//!
//! ```text
//! InferenceService (single-threaded)
//!       │
//!       │ PolicyZmqClient.check() [async ZMQ I/O]
//!       ▼
//! PolicyService (multi-threaded)
//!       │
//!       │ block_in_place + PolicyManager.check()
//!       ▼
//! Casbin enforcer
//! ```
//!
//! This architecture solves the threading issue where InferenceService
//! uses a single-threaded runtime (due to tch-rs raw pointers) but
//! PolicyManager needs block_in_place() which requires multi-threaded.

use crate::auth::{Operation, PolicyManager};
use crate::policy_capnp;
use crate::services::{EnvelopeContext, ServiceRunner, ZmqClient, ZmqService};
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use std::sync::Arc;
use tracing::{debug, trace};

/// Service name for endpoint registry
const SERVICE_NAME: &str = "policy";

// ============================================================================
// PolicyService (server-side)
// ============================================================================

/// Policy service that wraps PolicyManager
///
/// Runs on multi-threaded runtime where block_in_place works safely.
/// Receives policy check requests over ZMQ and delegates to PolicyManager.
pub struct PolicyService {
    policy_manager: Arc<PolicyManager>,
}

impl PolicyService {
    /// Create a new policy service
    pub fn new(policy_manager: Arc<PolicyManager>) -> Self {
        Self { policy_manager }
    }

    /// Start the policy service at the default endpoint (from registry)
    pub async fn start(
        policy_manager: Arc<PolicyManager>,
        server_pubkey: VerifyingKey,
    ) -> Result<crate::services::ServiceHandle> {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::start_at(policy_manager, server_pubkey, &endpoint).await
    }

    /// Start the policy service at a specific endpoint
    pub async fn start_at(
        policy_manager: Arc<PolicyManager>,
        server_pubkey: VerifyingKey,
        endpoint: &str,
    ) -> Result<crate::services::ServiceHandle> {
        let service = Self::new(policy_manager);
        let runner = ServiceRunner::new(endpoint, server_pubkey);
        let handle = runner.run(service).await?;
        tracing::info!("PolicyService started at {}", endpoint);
        Ok(handle)
    }

    /// Parse operation from string
    fn parse_operation(op_str: &str) -> Result<Operation> {
        match op_str {
            "infer" => Ok(Operation::Infer),
            "train" => Ok(Operation::Train),
            "query" => Ok(Operation::Query),
            "write" => Ok(Operation::Write),
            "serve" => Ok(Operation::Serve),
            "manage" => Ok(Operation::Manage),
            "context" => Ok(Operation::Context),
            _ => Err(anyhow!("Unknown operation: {}", op_str)),
        }
    }

    /// Build an allowed response
    fn build_response(request_id: u64, allowed: bool) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<policy_capnp::policy_response::Builder>();
            response.set_request_id(request_id);
            response.set_allowed(allowed);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Build an error response
    fn build_error_response(request_id: u64, message_text: &str, code: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut response = message.init_root::<policy_capnp::policy_response::Builder>();
            response.set_request_id(request_id);
            let mut error = response.init_error();
            error.set_message(message_text);
            error.set_code(code);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }
}

impl ZmqService for PolicyService {
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
        trace!(
            "Policy request from {} (id={})",
            ctx.casbin_subject(),
            ctx.request_id
        );

        // Parse request
        let reader = serialize::read_message(&mut std::io::Cursor::new(payload), ReaderOptions::new())?;
        let req = reader.get_root::<policy_capnp::policy_request::Reader>()?;

        let request_id = req.get_id();
        let subject = req.get_subject()?.to_str()?;
        let domain = req.get_domain()?.to_str()?;
        let resource = req.get_resource()?.to_str()?;
        let operation_str = req.get_operation()?.to_str()?;

        // Parse operation
        let operation = match Self::parse_operation(operation_str) {
            Ok(op) => op,
            Err(e) => {
                return Self::build_error_response(
                    request_id,
                    &format!("Invalid operation: {}", e),
                    "INVALID_OPERATION",
                );
            }
        };

        debug!(
            "Policy check: subject={}, domain={}, resource={}, op={}",
            subject, domain, resource, operation_str
        );

        // PolicyManager.check() is async, but we're on multi-threaded runtime
        // so block_in_place works here
        let allowed = tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(self.policy_manager.check_with_domain(
                subject,
                domain,
                resource,
                operation,
            ))
        });

        debug!("Policy check result: allowed={}", allowed);

        Self::build_response(request_id, allowed)
    }

    fn name(&self) -> &str {
        "policy"
    }
}

// ============================================================================
// PolicyZmqClient (client-side)
// ============================================================================

/// Client for policy checks over ZMQ
///
/// This client uses async ZMQ I/O which does NOT require block_in_place.
/// Safe to call from single-threaded runtimes like InferenceService.
#[derive(Clone)]
pub struct PolicyZmqClient {
    /// Underlying ZMQ client
    client: Arc<ZmqClient>,
}

impl PolicyZmqClient {
    /// Create a new policy client (endpoint from registry)
    ///
    /// # Arguments
    /// * `signing_key` - Ed25519 signing key for request authentication
    /// * `identity` - Identity to include in requests (for authorization)
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self {
            client: Arc::new(ZmqClient::new(&endpoint, signing_key, identity)),
        }
    }

    /// Create a new policy client at a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            client: Arc::new(ZmqClient::new(endpoint, signing_key, identity)),
        }
    }

    /// Check if subject is allowed to perform operation on resource
    ///
    /// This is async and does NOT use block_in_place.
    /// Safe to call from single-threaded runtime.
    pub async fn check(
        &self,
        subject: &str,
        resource: &str,
        operation: Operation,
    ) -> Result<bool> {
        self.check_with_domain(subject, "*", resource, operation).await
    }

    /// Check with explicit domain
    pub async fn check_with_domain(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: Operation,
    ) -> Result<bool> {
        let request_id = self.client.next_id();

        // Build request
        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<policy_capnp::policy_request::Builder>();
            req.set_id(request_id);
            req.set_subject(subject);
            req.set_domain(domain);
            req.set_resource(resource);
            req.set_operation(operation.as_str());
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        // Send request via ZMQ (async I/O - doesn't block runtime)
        let response_bytes = self.client.call(request_bytes).await?;

        // Parse response
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(&response_bytes),
            ReaderOptions::new(),
        )?;
        let response = reader.get_root::<policy_capnp::policy_response::Reader>()?;

        // Use which() to match the union variant
        use crate::policy_capnp::policy_response::Which;
        match response.which()? {
            Which::Allowed(allowed) => Ok(allowed),
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "Policy check failed: {} ({})",
                    error.get_message()?.to_str()?,
                    error.get_code()?.to_str()?
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_operation() {
        assert!(matches!(PolicyService::parse_operation("infer"), Ok(Operation::Infer)));
        assert!(matches!(PolicyService::parse_operation("train"), Ok(Operation::Train)));
        assert!(matches!(PolicyService::parse_operation("query"), Ok(Operation::Query)));
        assert!(matches!(PolicyService::parse_operation("write"), Ok(Operation::Write)));
        assert!(matches!(PolicyService::parse_operation("serve"), Ok(Operation::Serve)));
        assert!(matches!(PolicyService::parse_operation("manage"), Ok(Operation::Manage)));
        assert!(matches!(PolicyService::parse_operation("context"), Ok(Operation::Context)));
        assert!(PolicyService::parse_operation("unknown").is_err());
    }
}
