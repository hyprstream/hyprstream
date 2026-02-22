//! Policy service for authorization checks over ZMQ
//!
//! Wraps PolicyManager and exposes it as a ZmqService.
//! Handlers are async and use `.await` directly (compatible with single-threaded runtime).

use async_trait::async_trait;
use crate::auth::{Operation, PolicyManager};
use crate::auth::policy_templates;
use crate::services::{EnvelopeContext, ZmqService};
use crate::services::generated::policy_client::{
    ErrorInfo, PolicyClient, PolicyHandler, PolicyResponseVariant, TokenInfo, ScopeList,
    PolicyCheck, IssueToken,
    ApplyTemplate, ApplyDraft, RollbackPolicy, GetHistory, GetDiff,
    PolicyInfo, PolicyRule, Grouping,
    PolicyHistory, PolicyHistoryEntry, DraftStatus,
    dispatch_policy, serialize_response,
};
use anyhow::{anyhow, Result};
use git2db::{Git2DB, RepoId};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

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
    /// Shared git2db registry for git operations on .registry repo
    git2db: Arc<RwLock<Git2DB>>,
    /// RepoId of the .registry self-tracked entry
    registry_repo_id: RepoId,
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
        git2db: Arc<RwLock<Git2DB>>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        let registry_repo_id = RepoId::from_uuid(git2db::registry::registry_self_uuid());
        Self {
            policy_manager,
            signing_key,
            token_config,
            supported_scopes: compute_supported_scopes(),
            git2db,
            registry_repo_id,
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

    /// Stage policies/ and commit with the given message via git2db.
    /// Returns the commit OID as a hex string.
    async fn stage_and_commit_policies(&self, message: &str) -> Result<String> {
        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;

        // Stage the policies/ directory
        handle.staging().add_all().await
            .map_err(|e| anyhow!("Failed to stage policy files: {}", e))?;

        // Commit
        let oid = handle.commit(message).await
            .map_err(|e| anyhow!("Failed to commit policy: {}", e))?;

        Ok(oid.to_string())
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
                    code: "INVALID_OPERATION".to_owned(),
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
                    code: "UNAUTHORIZED_SUBJECT".to_owned(),
                    details: "Requires 'manage' permission on 'policy:issue-token'".to_owned(),
                }));
            }
            data.subject.clone()
        } else {
            // Use bare username from the envelope identity.
            ctx.user().to_owned()
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
                code: "TTL_TOO_SHORT".to_owned(),
                details: String::new(),
            }));
        }

        if requested_ttl > self.token_config.max_ttl_seconds {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("TTL exceeds maximum: {} > {}", requested_ttl, self.token_config.max_ttl_seconds),
                code: "TTL_EXCEEDED".to_owned(),
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

    async fn handle_get_policy(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        trace!("Getting current policy");

        let policies = self.policy_manager.get_policy().await;
        let groupings = self.policy_manager.get_grouping_policy().await;

        let rules: Vec<PolicyRule> = policies
            .into_iter()
            .map(|p| PolicyRule {
                subject: p.first().cloned().unwrap_or_default(),
                domain: p.get(1).cloned().unwrap_or_default(),
                resource: p.get(2).cloned().unwrap_or_default(),
                action: p.get(3).cloned().unwrap_or_default(),
                effect: p.get(4).cloned().unwrap_or_default(),
            })
            .collect();

        let grouping_list: Vec<Grouping> = groupings
            .into_iter()
            .map(|g| Grouping {
                user: g.first().cloned().unwrap_or_default(),
                role: g.get(1).cloned().unwrap_or_default(),
            })
            .collect();

        Ok(PolicyResponseVariant::GetPolicyResult(PolicyInfo {
            rules,
            groupings: grouping_list,
        }))
    }

    async fn handle_apply_template(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ApplyTemplate,
    ) -> Result<PolicyResponseVariant> {
        info!("Applying policy template: {}", data.name);

        // Validate template exists
        let template = match policy_templates::get_template(&data.name) {
            Some(t) => t,
            None => {
                let available: Vec<&str> = policy_templates::get_templates()
                    .iter()
                    .map(|t| t.name)
                    .collect();
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!(
                        "Unknown template: '{}'. Available: {}",
                        data.name,
                        available.join(", ")
                    ),
                    code: "UNKNOWN_TEMPLATE".to_owned(),
                    details: String::new(),
                }));
            }
        };

        // Get the expanded rules
        let new_content = template.expanded_rules();

        // Read existing content for rollback on validation failure
        let policy_path = self.policy_manager.policy_csv_path();
        let existing_content = tokio::fs::read_to_string(&policy_path).await
            .unwrap_or_default();

        // Write the new policy with restrictive permissions
        crate::auth::write_policy_file(&policy_path, &new_content).await
            .map_err(|e| anyhow!("Failed to write policy file: {}", e))?;

        // Validate by reloading
        if let Err(e) = self.policy_manager.reload().await {
            // Rollback on validation failure
            warn!("Template validation failed, rolling back: {}", e);
            let _ = crate::auth::write_policy_file(&policy_path, &existing_content).await;
            let _ = self.policy_manager.reload().await;
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Policy validation failed: {}", e),
                code: "VALIDATION_FAILED".to_owned(),
                details: String::new(),
            }));
        }

        // Stage and commit
        let commit_msg = format!("policy: apply {} template", data.name);
        match self.stage_and_commit_policies(&commit_msg).await {
            Ok(_) => {
                info!("Template '{}' applied and committed", data.name);
                Ok(PolicyResponseVariant::ApplyTemplateResult(commit_msg))
            }
            Err(e) => {
                // Policy is already reloaded and valid, just commit failed
                warn!("Template applied but commit failed: {}", e);
                Ok(PolicyResponseVariant::ApplyTemplateResult(
                    format!("policy: apply {} template (commit failed: {})", data.name, e)
                ))
            }
        }
    }

    async fn handle_apply_draft(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ApplyDraft,
    ) -> Result<PolicyResponseVariant> {
        info!("Applying draft policy changes");

        // Validate current disk state
        if let Err(e) = self.policy_manager.reload().await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Policy validation failed: {}", e),
                code: "VALIDATION_FAILED".to_owned(),
                details: "Fix errors in policy.csv before applying.".to_owned(),
            }));
        }

        // Generate commit message
        let commit_msg = if data.message.is_empty() {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            format!("policy: update access control rules ({timestamp})")
        } else {
            data.message.clone()
        };

        // Stage and commit
        match self.stage_and_commit_policies(&commit_msg).await {
            Ok(_) => {
                info!("Draft policy applied: {}", commit_msg);
                Ok(PolicyResponseVariant::ApplyDraftResult(commit_msg))
            }
            Err(e) => {
                Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Commit failed: {}", e),
                    code: "COMMIT_FAILED".to_owned(),
                    details: String::new(),
                }))
            }
        }
    }

    async fn handle_rollback(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RollbackPolicy,
    ) -> Result<PolicyResponseVariant> {
        info!("Rolling back policy to: {}", data.git_ref);

        let git_ref = data.git_ref.clone();

        // Use git2 escape hatch to checkout policies/ from the target ref
        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        // Resolve ref and checkout policies/ from it
        let git_ref_clone = git_ref.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let obj = repo.revparse_single(&git_ref_clone)
                .map_err(|e| anyhow!("Invalid git ref '{}': {}", git_ref_clone, e))?;
            let commit = obj.peel_to_commit()
                .map_err(|e| anyhow!("Ref '{}' does not point to a commit: {}", git_ref_clone, e))?;
            let tree = commit.tree()
                .map_err(|e| anyhow!("Failed to get tree: {}", e))?;

            // Find the policies/ subtree
            let policies_entry = tree.get_path(std::path::Path::new("policies"))
                .map_err(|e| anyhow!("No policies/ directory in {}: {}", git_ref_clone, e))?;
            let policies_tree = repo.find_tree(policies_entry.id())
                .map_err(|e| anyhow!("Failed to read policies tree: {}", e))?;

            // Checkout the policies tree to the workdir
            let mut checkout_opts = git2::build::CheckoutBuilder::new();
            checkout_opts.force();
            checkout_opts.path("policies");
            repo.checkout_tree(policies_tree.as_object(), Some(&mut checkout_opts))
                .map_err(|e| anyhow!("Failed to checkout policies/ from {}: {}", git_ref_clone, e))?;

            Ok(())
        }).await
            .map_err(|e| anyhow!("Checkout task failed: {}", e))??;

        // Validate the restored policy
        if let Err(e) = self.policy_manager.reload().await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Rollback validation failed: {}", e),
                code: "VALIDATION_FAILED".to_owned(),
                details: "The target version contains invalid policy.".to_owned(),
            }));
        }

        // Stage and commit the rollback
        let commit_msg = format!("policy: rollback to {}", git_ref);
        match self.stage_and_commit_policies(&commit_msg).await {
            Ok(_) => {
                info!("Policy rolled back to {}", git_ref);
                Ok(PolicyResponseVariant::RollbackResult(commit_msg))
            }
            Err(e) => {
                Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Rollback commit failed: {}", e),
                    code: "COMMIT_FAILED".to_owned(),
                    details: String::new(),
                }))
            }
        }
    }

    async fn handle_get_history(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetHistory,
    ) -> Result<PolicyResponseVariant> {
        let count = if data.count == 0 { 10 } else { data.count as usize };

        trace!("Getting policy history (count={})", count);

        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        let entries = tokio::task::spawn_blocking(move || -> Result<Vec<PolicyHistoryEntry>> {
            let mut entries = Vec::new();
            let mut revwalk = repo.revwalk()
                .map_err(|e| anyhow!("Failed to create revwalk: {}", e))?;
            revwalk.push_head()
                .map_err(|e| anyhow!("Failed to push HEAD: {}", e))?;

            for oid_result in revwalk {
                if entries.len() >= count {
                    break;
                }

                let oid = oid_result.map_err(|e| anyhow!("Revwalk error: {}", e))?;
                let commit = repo.find_commit(oid)
                    .map_err(|e| anyhow!("Failed to find commit: {}", e))?;

                // Check if this commit touches policies/
                let dominated = if let Ok(parent) = commit.parent(0) {
                    let commit_tree = commit.tree().ok();
                    let parent_tree = parent.tree().ok();
                    if let (Some(ct), Some(pt)) = (commit_tree, parent_tree) {
                        let diff = repo.diff_tree_to_tree(Some(&pt), Some(&ct), None).ok();
                        diff.is_some_and(|d| {
                            d.deltas().any(|delta| {
                                let path = delta.new_file().path()
                                    .or_else(|| delta.old_file().path());
                                path.is_some_and(|p| p.starts_with("policies"))
                            })
                        })
                    } else {
                        false
                    }
                } else {
                    // Root commit — check if it has policies/
                    commit.tree().ok()
                        .and_then(|t| t.get_path(std::path::Path::new("policies")).ok())
                        .is_some()
                };

                if dominated {
                    let time = commit.time();
                    let date = chrono::DateTime::from_timestamp(time.seconds(), 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "unknown".to_owned());

                    entries.push(PolicyHistoryEntry {
                        hash: oid.to_string()[..8].to_owned(),
                        message: commit.message().unwrap_or("").trim().to_owned(),
                        date,
                    });
                }
            }

            Ok(entries)
        }).await
            .map_err(|e| anyhow!("History task failed: {}", e))??;

        Ok(PolicyResponseVariant::GetHistoryResult(PolicyHistory { entries }))
    }

    async fn handle_get_diff(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetDiff,
    ) -> Result<PolicyResponseVariant> {
        let git_ref = if data.git_ref.is_empty() { "HEAD".to_owned() } else { data.git_ref.clone() };

        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        let output = tokio::task::spawn_blocking(move || -> Result<String> {
            let obj = repo.revparse_single(&git_ref)
                .map_err(|e| anyhow!("Invalid git ref '{}': {}", git_ref, e))?;
            let tree = obj.peel_to_tree()
                .map_err(|e| anyhow!("Could not peel {} to tree: {}", git_ref, e))?;

            let mut diff_opts = git2::DiffOptions::new();
            diff_opts.pathspec("policies/");

            let diff = repo.diff_tree_to_workdir_with_index(
                Some(&tree),
                Some(&mut diff_opts),
            ).map_err(|e| anyhow!("Failed to compute diff: {}", e))?;

            let mut result = String::new();
            diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
                let origin = line.origin();
                if origin == '+' || origin == '-' || origin == ' ' {
                    result.push(origin);
                }
                if let Ok(s) = std::str::from_utf8(line.content()) {
                    result.push_str(s);
                }
                true
            })?;

            Ok(result)
        }).await
            .map_err(|e| anyhow!("Diff task failed: {}", e))??;

        Ok(PolicyResponseVariant::GetDiffResult(output))
    }

    async fn handle_get_draft_status(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        let (has_changes, summary) = tokio::task::spawn_blocking(move || -> Result<(bool, String)> {
            let mut opts = git2::StatusOptions::new();
            opts.pathspec("policies/");
            opts.include_untracked(true);

            let statuses = repo.statuses(Some(&mut opts))
                .map_err(|e| anyhow!("Failed to get status: {}", e))?;

            let count = statuses.len();
            let summary = if count == 0 {
                "no changes".to_owned()
            } else {
                format!("{} file(s) changed", count)
            };

            Ok((count > 0, summary))
        }).await
            .map_err(|e| anyhow!("Status task failed: {}", e))??;

        Ok(PolicyResponseVariant::GetDraftStatusResult(DraftStatus {
            has_changes,
            summary,
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
            code: "INTERNAL".to_owned(),
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
