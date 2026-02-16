//! Policy service for authorization checks over ZMQ
//!
//! Wraps PolicyManager and exposes it as a ZmqService.
//! Handlers are async and use `.await` directly (compatible with single-threaded runtime).

use async_trait::async_trait;
use crate::auth::{Operation, PolicyManager};
use crate::services::{EnvelopeContext, ZmqService};
use crate::services::generated::policy_client::{
    ErrorInfo, PolicyClient, PolicyHandler, PolicyResponseVariant, TokenInfo, ScopeList,
    PolicyCheck, IssueToken,
    dispatch_policy, serialize_response,
};
use anyhow::{anyhow, Result};
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

/// Policy service that wraps PolicyManager.
/// Receives policy check requests over ZMQ and delegates to PolicyManager.
pub struct PolicyService {
    // Business logic
    policy_manager: Arc<PolicyManager>,
    signing_key: Arc<SigningKey>,
    token_config: crate::config::TokenConfig,
    /// Supported scopes computed once at construction from ServiceFactory inventory
    supported_scopes: Vec<String>,
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
            supported_scopes: compute_supported_scopes(),
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

/// Collect all supported scopes from compile-time schema metadata
/// via the ServiceFactory inventory. No hardcoded service imports needed.
///
/// Scopes use flat format `action:service:*` — coarse-grained per OAuth convention.
/// Fine-grained authorization is handled by Casbin resource patterns.
fn compute_supported_scopes() -> Vec<String> {
    use hyprstream_rpc::service::factory::list_factories;

    let mut scopes = std::collections::BTreeSet::new();

    for factory in list_factories() {
        if let Some(metadata_fn) = factory.metadata {
            let (service_name, methods) = metadata_fn();
            for method in methods {
                // Fallback is "query" (ScopeAction::query @0), NOT "read"
                // which doesn't exist in Operation or ScopeAction enums.
                let action = if method.scope.is_empty() { "query" } else { method.scope };
                scopes.insert(format!("{}:{}:*", action, service_name));
            }
        }
    }

    scopes.into_iter().collect()
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
        Ok(PolicyResponseVariant::CheckResult(allowed))
    }

    async fn handle_issue_token(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &IssueToken,
    ) -> Result<PolicyResponseVariant> {
        trace!("Issuing JWT token");

        // Determine subject: explicit subject (if provided and authorized) or envelope identity.
        // JWT sub must contain a bare username (e.g. "randy", "birdetta") — the identity
        // system adds the namespace prefix ("token:randy") when the JWT is decoded.
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
            // Use bare username from the envelope identity.
            ctx.user().to_string()
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

        Ok(PolicyResponseVariant::IssueTokenResult(TokenInfo {
            token,
            expires_at: claims.exp,
        }))
    }

    async fn handle_list_scopes(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        Ok(PolicyResponseVariant::ListScopesResult(ScopeList {
            scopes: self.supported_scopes.clone(),
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

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = PolicyResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_string(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// PolicyClient construction (uses create_service_client pattern)
// ============================================================================

impl PolicyClient {
    /// Create a new policy client (endpoint from registry)
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        crate::services::core::create_service_client(&endpoint, signing_key, identity)
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
