//! Policy service for authorization checks over ZMQ
//!
//! Wraps PolicyManager and exposes it as a ZmqService.
//! Handlers are async and use `.await` directly (compatible with single-threaded runtime).

use async_trait::async_trait;
use crate::auth::{Operation, PolicyManager};
use crate::services::{EnvelopeContext, ZmqService};
use crate::services::generated::policy_client::{
    ErrorInfo, PolicyClient, PolicyHandler, PolicyResponseVariant, TokenInfo,
    PolicyCheck, IssueToken,
    dispatch_policy,
};
use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::service::ZmqClient as ZmqClientBase;
use hyprstream_rpc::service::factory::ServiceClient;
use hyprstream_rpc::transport::TransportConfig;
use std::sync::Arc;
use tracing::{debug, trace};

/// Service name for endpoint registry
const SERVICE_NAME: &str = "policy";

// ============================================================================
// PolicyService (server-side)
// ============================================================================

/// Policy service that wraps PolicyManager.
/// Receives policy check requests over ZMQ and delegates to PolicyManager.
pub struct PolicyService {
    // Business logic
    policy_manager: Arc<PolicyManager>,
    signing_key: Arc<SigningKey>,
    token_config: crate::config::TokenConfig,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
}

impl PolicyService {
    /// Create a new policy service with infrastructure
    pub fn new(
        policy_manager: Arc<PolicyManager>,
        signing_key: Arc<SigningKey>,
        token_config: crate::config::TokenConfig,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        Self {
            policy_manager,
            signing_key,
            token_config,
            context,
            transport,
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

}

// ============================================================================
// PolicyHandler implementation (generated trait)
// ============================================================================

#[async_trait::async_trait(?Send)]
impl PolicyHandler for PolicyService {
    async fn handle_check(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &PolicyCheck,
    ) -> Result<PolicyResponseVariant> {
        trace!(
            "Policy check: subject={}, domain={}, resource={}, operation={}",
            data.subject, data.domain, data.resource, data.operation
        );

        // Parse operation
        let operation = match Self::parse_operation(&data.operation) {
            Ok(op) => op,
            Err(_) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Invalid operation: {}", data.operation),
                    code: "INVALID_OPERATION".to_string(),
                    details: String::new(),
                }));
            }
        };

        // Check authorization
        let allowed = self.policy_manager.check_with_domain(
            &data.subject,
            &data.domain,
            &data.resource,
            operation,
        ).await;

        debug!("Policy check result: allowed={}", allowed);
        Ok(PolicyResponseVariant::Allowed(allowed))
    }

    async fn handle_issue_token(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &IssueToken,
    ) -> Result<PolicyResponseVariant> {
        trace!("Issuing JWT token");

        // Determine subject: explicit subject (if provided and authorized) or envelope identity
        let subject = if !data.subject.is_empty() {
            // Explicit subject requires `manage` permission on `policy:issue-token`
            let caller = ctx.subject().to_string();
            let allowed = self.policy_manager.check_with_domain(
                &caller,
                "*",
                "policy:issue-token",
                Operation::Manage,
            ).await;
            if !allowed {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!(
                        "Subject '{}' is not authorized to issue tokens on behalf of '{}'",
                        caller, data.subject
                    ),
                    code: "UNAUTHORIZED_SUBJECT".to_string(),
                    details: "Requires 'manage' permission on 'policy:issue-token'".to_string(),
                }));
            }
            data.subject.clone()
        } else {
            ctx.subject().to_string()
        };

        // Validate TTL
        let requested_ttl = if data.ttl == 0 {
            self.token_config.default_ttl_seconds
        } else {
            data.ttl
        };

        const MIN_TTL_SECONDS: u32 = 60;
        if requested_ttl < MIN_TTL_SECONDS {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("TTL too short: {} < {} seconds minimum", requested_ttl, MIN_TTL_SECONDS),
                code: "TTL_TOO_SHORT".to_string(),
                details: String::new(),
            }));
        }

        if requested_ttl > self.token_config.max_ttl_seconds {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("TTL exceeds maximum: {} > {}", requested_ttl, self.token_config.max_ttl_seconds),
                code: "TTL_EXCEEDED".to_string(),
                details: String::new(),
            }));
        }

        // Create and sign JWT with optional audience binding (RFC 8707)
        // Scopes are not embedded in JWT - Casbin enforces authorization server-side
        let now = chrono::Utc::now().timestamp();
        let audience = if data.audience.is_empty() {
            None
        } else {
            Some(data.audience.clone())
        };

        let claims = hyprstream_rpc::auth::Claims::new(
            subject,
            now,
            now + requested_ttl as i64,
        ).with_audience(audience);

        let token = crate::auth::jwt::encode(&claims, &self.signing_key);

        Ok(PolicyResponseVariant::TokenSuccess(TokenInfo {
            token,
            expires_at: claims.exp,
        }))
    }
}

#[async_trait(?Send)]
impl ZmqService for PolicyService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
        trace!(
            "Policy request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_policy(self, ctx, payload).await
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

    fn signing_key(&self) -> SigningKey {
        (*self.signing_key).clone()
    }
}

// ============================================================================
// PolicyClient convenience methods (extends generated client)
// ============================================================================

impl PolicyClient {
    /// Create a new policy client (endpoint from registry)
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a new policy client at a specific endpoint
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let server_verifying_key = signing_key.verifying_key();
        let zmq_client = ZmqClientBase::new(endpoint, global_context(), signing_key, server_verifying_key, identity);
        Self::from_zmq(zmq_client)
    }

    /// Check if subject is allowed to perform operation on resource.
    ///
    /// Accepts a typed `Subject` for type safety. The string conversion
    /// for Casbin happens internally.
    pub async fn check_policy(
        &self,
        subject: &hyprstream_rpc::Subject,
        resource: &str,
        operation: Operation,
    ) -> Result<bool> {
        self.check_with_domain_policy(subject, "*", resource, operation).await
    }

    /// Check policy with a string subject (for HTTP route callers).
    ///
    /// Parses the string as a Subject, falling back to anonymous on parse failure.
    pub async fn check_policy_str(
        &self,
        subject: &str,
        resource: &str,
        operation: Operation,
    ) -> Result<bool> {
        self.check_with_domain_policy_str(subject, "*", resource, operation).await
    }

    /// Check with explicit domain
    pub async fn check_with_domain_policy(
        &self,
        subject: &hyprstream_rpc::Subject,
        domain: &str,
        resource: &str,
        operation: Operation,
    ) -> Result<bool> {
        let subject_str = subject.to_string();
        match self.check(&subject_str, domain, resource, operation.as_str()).await? {
            PolicyResponseVariant::Allowed(allowed) => Ok(allowed),
            PolicyResponseVariant::Error(ref e) => {
                Err(anyhow!("Policy check failed: {} ({})", e.message, e.code))
            }
            _ => Err(anyhow!("Unexpected response type for policy check")),
        }
    }

    /// Check with explicit domain (string subject variant for HTTP routes)
    pub async fn check_with_domain_policy_str(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: Operation,
    ) -> Result<bool> {
        match self.check(subject, domain, resource, operation.as_str()).await? {
            PolicyResponseVariant::Allowed(allowed) => Ok(allowed),
            PolicyResponseVariant::Error(ref e) => {
                Err(anyhow!("Policy check failed: {} ({})", e.message, e.code))
            }
            _ => Err(anyhow!("Unexpected response type for policy check")),
        }
    }

    /// Issue a JWT token with requested scopes
    pub async fn issue_jwt_token(
        &self,
        scopes: Vec<String>,
        ttl: Option<u32>,
    ) -> Result<(String, i64)> {
        self.issue_jwt_token_with_audience(scopes, ttl, None).await
    }

    /// Issue a JWT token with requested scopes and optional audience (RFC 8707)
    pub async fn issue_jwt_token_with_audience(
        &self,
        scopes: Vec<String>,
        ttl: Option<u32>,
        audience: Option<String>,
    ) -> Result<(String, i64)> {
        self.issue_jwt_token_full(scopes, ttl, audience, None).await
    }

    /// Issue a JWT token with all options including explicit subject.
    ///
    /// When `subject` is `Some(...)`, the token is issued on behalf of that subject.
    /// Requires caller to have `manage` permission on `policy:issue-token`.
    pub async fn issue_jwt_token_full(
        &self,
        scopes: Vec<String>,
        ttl: Option<u32>,
        audience: Option<String>,
        subject: Option<String>,
    ) -> Result<(String, i64)> {
        match self.issue_token(
            &scopes,
            ttl.unwrap_or(0),
            &audience.unwrap_or_default(),
            &subject.unwrap_or_default(),
        ).await? {
            PolicyResponseVariant::TokenSuccess(ref data) => {
                Ok((data.token.clone(), data.expires_at))
            }
            PolicyResponseVariant::Error(ref e) => {
                Err(anyhow!("Token issuance failed: {} ({})", e.message, e.code))
            }
            _ => Err(anyhow!("Unexpected response type for issue_token")),
        }
    }
}

// ============================================================================
// Policy file watcher (hot-reload)
// ============================================================================

/// Watch policy.csv for changes and reload PolicyManager automatically.
///
/// Watches the parent directory (not the file directly) to handle atomic
/// rename patterns used by editors like vim and emacs.
pub(crate) async fn watch_policy_file(
    policy_manager: Arc<PolicyManager>,
    policy_csv: std::path::PathBuf,
) {
    use notify::{Event, EventKind, RecursiveMode, Watcher};
    use tracing::{info, warn};

    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(16);

    let csv_path = policy_csv.clone();
    let mut watcher = match notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            match event.kind {
                EventKind::Modify(_) | EventKind::Create(_) => {
                    // Only trigger for events involving our policy.csv
                    if event.paths.iter().any(|p| p.ends_with("policy.csv")) {
                        let _ = tx.blocking_send(());
                    }
                }
                _ => {}
            }
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            warn!("Failed to create policy file watcher: {}", e);
            return;
        }
    };

    // Watch parent directory to catch atomic renames
    let watch_dir = match policy_csv.parent() {
        Some(dir) => dir,
        None => {
            warn!("policy.csv has no parent directory, cannot watch");
            return;
        }
    };

    if let Err(e) = watcher.watch(watch_dir, RecursiveMode::NonRecursive) {
        warn!("Failed to watch {}: {}", watch_dir.display(), e);
        return;
    }

    info!("Watching {} for policy changes", csv_path.display());

    loop {
        // Wait for first event
        if rx.recv().await.is_none() {
            break; // Channel closed
        }

        // Debounce: wait 500ms then drain remaining events
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        while rx.try_recv().is_ok() {}

        // Reload policy
        match policy_manager.reload().await {
            Ok(()) => info!("Policy reloaded from disk"),
            Err(e) => warn!("Failed to reload policy: {}", e),
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
