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
use crate::services::{CallOptions, EnvelopeContext, ZmqClient, ZmqService};
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
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
    // Business logic
    policy_manager: Arc<PolicyManager>,
    signing_key: Arc<SigningKey>,
    token_config: crate::config::TokenConfig,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    verifying_key: VerifyingKey,
}

impl PolicyService {
    /// Create a new policy service with infrastructure
    pub fn new(
        policy_manager: Arc<PolicyManager>,
        signing_key: Arc<SigningKey>,
        token_config: crate::config::TokenConfig,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        verifying_key: VerifyingKey,
    ) -> Self {
        Self {
            policy_manager,
            signing_key,
            token_config,
            context,
            transport,
            verifying_key,
        }
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

    /// Handle JWT token issuance request
    fn handle_issue_token(
        &self,
        ctx: &EnvelopeContext,
        request_id: u64,
        request: policy_capnp::issue_token::Reader,
    ) -> Result<Vec<u8>> {
        use crate::services::rpc_types::PolicyResponse;
        use hyprstream_rpc::auth::Scope;

        let subject = ctx.casbin_subject();

        // Parse scopes
        let requested_scopes: Result<Vec<String>> = request
            .get_requested_scopes()?
            .iter()
            .map(|s| s?.to_str().map(String::from).map_err(Into::into))
            .collect();
        let requested_scopes = requested_scopes?;

        // Authorize each scope via Casbin
        // Scopes are used directly as resources (no domain mapping needed)
        for scope_str in &requested_scopes {
            // Parse scope to validate format
            let scope = match Scope::parse(scope_str) {
                Ok(s) => s,
                Err(_) => {
                    return Ok(PolicyResponse::error(
                        request_id,
                        &format!("Invalid scope format: {}", scope_str),
                        "INVALID_SCOPE"
                    ));
                }
            };

            let allowed = tokio::task::block_in_place(|| {
                let handle = tokio::runtime::Handle::current();
                handle.block_on(self.policy_manager.check(
                    &subject,
                    &scope.to_string(),  // Use scope directly as resource
                    Operation::Infer,
                ))
            });

            if !allowed {
                // Use PolicyResponse helper (eliminates boilerplate)
                return Ok(PolicyResponse::unauthorized(request_id, &scope.to_string()));
            }
        }

        // Validate TTL
        let requested_ttl = if request.get_ttl() == 0 {
            self.token_config.default_ttl_seconds
        } else {
            request.get_ttl()
        };

        if requested_ttl > self.token_config.max_ttl_seconds {
            // Use PolicyResponse helper
            return Ok(PolicyResponse::ttl_exceeded(
                request_id,
                requested_ttl,
                self.token_config.max_ttl_seconds
            ));
        }

        // Parse scopes into Scope objects
        let parsed_scopes: Result<Vec<Scope>> = requested_scopes
            .iter()
            .map(|s| Scope::parse(s))
            .collect();
        let parsed_scopes = parsed_scopes?;

        // Create and sign JWT
        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new(
            subject,
            now,
            now + requested_ttl as i64,
            parsed_scopes,
            ctx.identity.is_local() || ctx.user().contains("admin"),
        );

        let token = crate::auth::jwt::encode(&claims, &self.signing_key);

        // Use PolicyResponse helper (eliminates boilerplate)
        Ok(PolicyResponse::token_success(request_id, &token, claims.exp))
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

        // Deserialize request
        let reader = serialize::read_message(&mut std::io::Cursor::new(payload), ReaderOptions::new())?;
        let req = reader.get_root::<policy_capnp::policy_request::Reader>()?;

        let request_id = req.get_id();

        // Handle based on request type using union discriminator
        use policy_capnp::policy_request::Which;

        match req.which()? {
            Which::Check(check) => {
                let check = check?;
                let subject = check.get_subject()?.to_str()?;
                let domain = check.get_domain()?.to_str()?;
                let resource = check.get_resource()?.to_str()?;
                let operation_str = check.get_operation()?.to_str()?;

                trace!(
                    "Policy check: subject={}, domain={}, resource={}, operation={}",
                    subject, domain, resource, operation_str
                );

                // Parse operation
                let operation = match Self::parse_operation(operation_str) {
                    Ok(op) => op,
                    Err(_) => {
                        return Self::build_error_response(
                            request_id,
                            &format!("Invalid operation: {}", operation_str),
                            "INVALID_OPERATION"
                        );
                    }
                };

                // Check authorization
                let allowed = tokio::task::block_in_place(|| {
                    let handle = tokio::runtime::Handle::current();
                    handle.block_on(self.policy_manager.check_with_domain(
                        &subject,
                        domain,
                        resource,
                        operation
                    ))
                });

                debug!("Policy check result: allowed={}", allowed);

                // Build response
                Self::build_response(request_id, allowed)
            }

            Which::IssueToken(token_req) => {
                let token_req = token_req?;
                trace!("Issuing JWT token");
                self.handle_issue_token(ctx, request_id, token_req)
            }
        }
    }

    fn name(&self) -> &str {
        "policy"
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

        // Build request with union discriminator
        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<policy_capnp::policy_request::Builder>();
            req.set_id(request_id);

            // Initialize check union variant
            let mut check = req.init_check();
            check.set_subject(subject);
            check.set_domain(domain);
            check.set_resource(resource);
            check.set_operation(operation.as_str());
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        // Send request via ZMQ (async I/O - doesn't block runtime)
        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;

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

    /// Issue a JWT token with requested scopes
    ///
    /// # Arguments
    /// * `scopes` - Structured scopes in "action:resource:identifier" format
    /// * `ttl` - Optional TTL in seconds (None = use server default)
    ///
    /// # Returns
    /// * `(token, expires_at)` - JWT token string and expiration timestamp
    ///
    /// # Example
    /// ```ignore
    /// let scopes = vec![
    ///     "infer:model:qwen-7b".to_string(),
    ///     "subscribe:stream:abc-123".to_string(),
    /// ];
    /// let (token, expires_at) = client.issue_token(scopes, Some(300)).await?;
    /// ```
    pub async fn issue_token(
        &self,
        scopes: Vec<String>,
        ttl: Option<u32>,
    ) -> Result<(String, i64)> {
        let request_id = self.client.next_id();

        // Build request with union discriminator
        let mut message = Builder::new_default();
        {
            let mut req = message.init_root::<policy_capnp::policy_request::Builder>();
            req.set_id(request_id);

            // Initialize issueToken union variant
            let mut issue_token = req.init_issue_token();
            issue_token.set_ttl(ttl.unwrap_or(0));

            // Set scopes list
            let mut scopes_list = issue_token.init_requested_scopes(scopes.len() as u32);
            for (i, scope) in scopes.iter().enumerate() {
                scopes_list.set(i as u32, scope);
            }
        }

        let mut request_bytes = Vec::new();
        serialize::write_message(&mut request_bytes, &message)?;

        // Send request via ZMQ (async I/O)
        let response_bytes = self.client.call(request_bytes, CallOptions::default()).await?;

        // Parse response
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(&response_bytes),
            ReaderOptions::new(),
        )?;
        let response = reader.get_root::<policy_capnp::issue_token_response::Reader>()?;

        // Match union variant
        use crate::policy_capnp::issue_token_response::Which;
        match response.which()? {
            Which::Success(token_info_reader) => {
                let token_info = token_info_reader?;
                Ok((
                    token_info.get_token()?.to_str()?.to_string(),
                    token_info.get_expires_at(),
                ))
            }
            Which::Error(error_reader) => {
                let error = error_reader?;
                Err(anyhow!(
                    "Token issuance failed: {} ({})",
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
