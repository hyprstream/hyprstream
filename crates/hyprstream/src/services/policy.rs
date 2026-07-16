//! Policy service for authorization checks over ZMQ
//!
//! Wraps PolicyManager and exposes it as a RequestService.
//! Handlers are async and use `.await` directly (compatible with single-threaded runtime).

use async_trait::async_trait;
use crate::auth::PolicyManager;
use crate::auth::policy_templates;
use crate::services::{EnvelopeContext, RequestService};
use crate::services::generated::policy_client::{
    ErrorInfo, PolicyHandler, PolicyResponseVariant, TokenInfo, ScopeList,
    PolicyCheck, IssueToken,
    ApplyTemplate, ApplyDraft, RollbackPolicy, GetHistory, GetDiff,
    PolicyInfo, PolicyRule, Grouping,
    PolicyHistory, PolicyHistoryEntry, DraftStatus,
    AddGrouping, RemoveGrouping, SetBranchVisibility,
    RegisterEventPrefix, SubscribeEventPrefix, GetPendingSubscribers, DepositWrappedKeys,
    EventPrefixAccess, PendingSubscribers,
    ResolveServiceKey, RegisterServiceKey, ServiceKeyResponse,
    RefreshServiceTokenRequest, ExchangeWit,
    dispatch_policy, serialize_response,
};
use anyhow::{anyhow, Result};
use git2db::{Git2DB, RepoId};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::transport::TransportConfig;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

// ============================================================================
// PolicyService (server-side)
// ============================================================================

/// Policy service that wraps PolicyManager.
/// Receives policy check requests over ZMQ and delegates to PolicyManager.
/// Per-prefix event state (blind key relay).
///
/// PolicyService stores opaque wrapped key blobs — it never sees plaintext
/// group keys or wrap keys. The publisher wraps directly against subscriber
/// pubkeys via DH.
struct EventPrefixState {
    publisher_pubkey: [u8; 32],
    schema: String,
    /// Subscriber ephemeral pubkeys, keyed by Blake3 hash of pubkey.
    subscriber_pubkeys: HashMap<[u8; 32], [u8; 32]>,
    /// Opaque wrapped key blobs deposited by the publisher, keyed by subscriber pubkey hash.
    wrapped_keys: HashMap<[u8; 32], Vec<u8>>,
}

pub struct PolicyService {
    // Business logic
    policy_manager: Arc<PolicyManager>,
    signing_key: Arc<SigningKey>,
    /// Purpose-derived key for JWT token signing (isolated from envelope signing)
    jwt_signing_key: SigningKey,
    token_config: crate::config::TokenConfig,
    /// Supported scopes computed once at construction from ServiceFactory inventory
    supported_scopes: Vec<String>,
    /// Shared git2db registry for git operations on .registry repo
    git2db: Arc<RwLock<Git2DB>>,
    /// RepoId of the .registry self-tracked entry
    registry_repo_id: RepoId,
    /// Default audience for issued tokens (OAuth issuer URL, shared instance identifier).
    /// Used when IssueToken.audience is empty, ensuring all tokens get an `aud` claim.
    default_audience: Option<String>,
    /// JWT key source for verifying JWTs (local and federated).
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    // Infrastructure (for Spawnable)
    transport: TransportConfig,
    /// Event prefix state for secure event transport (Phase 7).
    /// PolicyService is a blind relay — stores opaque wrapped blobs, never plaintext keys.
    event_prefixes: RwLock<HashMap<String, EventPrefixState>>,
    /// Shared JWT ID blocklist for access token revocation.
    jti_blocklist: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>,
    /// ES256 (P-256) key rotation store for DPoP/atproto interop.
    es256_key_store: Option<Arc<crate::auth::Es256SigningKeyStore>>,
    /// ML-DSA-65 key rotation store for PQ-hybrid composite token issuance.
    ml_dsa_key_store: Option<Arc<crate::auth::MlDsaSigningKeyStore>>,
}

impl PolicyService {
    /// Create a new policy service with infrastructure
    pub fn new(
        policy_manager: Arc<PolicyManager>,
        signing_key: Arc<SigningKey>,
        token_config: crate::config::TokenConfig,
        git2db: Arc<RwLock<Git2DB>>,
        transport: TransportConfig,
    ) -> Self {
        let registry_repo_id = RepoId::from_uuid(git2db::registry::registry_self_uuid());
        let jwt_signing_key = hyprstream_rpc::node_identity::derive_purpose_key(&signing_key, "hyprstream-jwt-v1");
        Self {
            policy_manager,
            signing_key,
            jwt_signing_key,
            token_config,
            supported_scopes: compute_supported_scopes(),
            git2db,
            registry_repo_id,
            default_audience: None,
            jwt_key_source: None,
            transport,
            event_prefixes: RwLock::new(HashMap::new()),
            jti_blocklist: Arc::new(hyprstream_rpc::auth::InMemoryJtiBlocklist::new()),
            es256_key_store: None,
            ml_dsa_key_store: None,
        }
    }

    /// Get a shared reference to the JWT ID blocklist (for wiring into OAuthState).
    pub fn jti_blocklist_arc(&self) -> Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist> {
        Arc::clone(&self.jti_blocklist)
    }

    /// Set the default audience for issued tokens (typically the OAuth issuer URL).
    pub fn with_default_audience(mut self, audience: String) -> Self {
        self.default_audience = Some(audience);
        self
    }

    /// Sign a token, selecting the suite from [`CryptoPolicy`] (Fu3/#677).
    ///
    /// The composite PQ signature (EdDSA + ML-DSA-65) is used under a Hybrid
    /// policy; under Classical the Ed25519-only suite is used. **Hybrid fails
    /// closed:** if the node is Hybrid but no ML-DSA-65 signing key is
    /// provisioned, this returns `Err` rather than silently minting a
    /// classical-only token — mirroring [`crate::mac::audit::CoseAuditSigner`],
    /// which the S7 audit path already gates the same way. Previously this seam
    /// picked composite-vs-classical by *keystore state*, so a Hybrid node with
    /// an empty/rotating ML-DSA store quietly downgraded minted tokens.
    async fn sign_token(
        &self,
        claims: &hyprstream_rpc::auth::Claims,
        is_service: bool,
    ) -> Result<String> {
        let policy = hyprstream_rpc::envelope::envelope_policy_from_env();
        if policy.uses_pq() {
            let snapshot = hyprstream_rpc::auth::global_composite_key_set()
                .mint_snapshot()
                .map_err(|error| anyhow!("composite authority unavailable: {error}"))?;
            let signing = snapshot
                .active_signing_pair(hyprstream_rpc::auth::CompositePairRole::Policy)
                .and_then(hyprstream_rpc::auth::CompositeKeyPair::signing_keys);
            let Some((ml_key, ed_key)) = signing else {
                warn!("no authorized active Policy composite pair; refusing to mint");
                return Err(anyhow!("hybrid token signing pair not provisioned"));
            };
            Ok(if is_service {
                crate::auth::jwt::encode_composite_service_jwt(
                    claims, &ml_key, &ed_key,
                )
            } else {
                crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
                    claims, &ml_key, &ed_key,
                )
            })
        } else {
            Ok(if is_service {
                crate::auth::jwt::encode_service_jwt(claims, &self.jwt_signing_key)
            } else {
                crate::auth::jwt::encode(claims, &self.jwt_signing_key)
            })
        }
    }

    #[cfg(test)]
    pub(crate) async fn sign_token_through_production_path(
        base_dir: &std::path::Path,
        claims: &hyprstream_rpc::auth::Claims,
    ) -> Result<String> {
        let policy_manager = Arc::new(PolicyManager::permissive().await?);
        let git2db = Arc::new(RwLock::new(Git2DB::open(base_dir).await?));
        let service = Self::new(
            policy_manager,
            Arc::new(SigningKey::from_bytes(&[0x73; 32])),
            crate::config::TokenConfig::default(),
            git2db,
            TransportConfig::inproc("multiprocess-policy-sign"),
        );
        service.sign_token(claims, false).await
    }

    /// Set the JWT key source for verifying JWTs (local and federated).
    pub fn with_jwt_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    ) -> Self {
        self.jwt_key_source = Some(src);
        self
    }

    /// Attach the ES256 (P-256) key rotation store.
    pub fn with_es256_key_store(mut self, store: Arc<crate::auth::Es256SigningKeyStore>) -> Self {
        self.es256_key_store = Some(store);
        self
    }

    /// Attach the ML-DSA-65 key rotation store for composite token issuance.
    pub fn with_ml_dsa_key_store(mut self, store: Arc<crate::auth::MlDsaSigningKeyStore>) -> Self {
        self.ml_dsa_key_store = Some(store);
        self
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
    use hyprstream_service::list_factories;

    let mut scopes = std::collections::BTreeSet::new();

    for factory in list_factories() {
        if let Some(metadata_fn) = factory.metadata {
            let (service_name, methods) = metadata_fn();
            for method in methods {
                // S3 (#547): scope is mandatory at build time, so `method.scope` is
                // non-empty for every enforced method. An empty scope here can only be a
                // `$scopeExempt` method (e.g. the authz check itself) — it requires no
                // grant, so it contributes no advertised scope. No silent "query" default.
                if method.scope.is_empty() {
                    continue;
                }
                scopes.insert(format!("{}:{}:*", method.scope, service_name));
            }
        }
    }

    scopes.into_iter().collect()
}

// ============================================================================
// PolicyHandler implementation (generated trait)
// ============================================================================

/// Validate an event prefix string.
///
/// Rejects empty strings, path traversal (`..`), and Casbin metacharacters (`*`, `#`).
/// Only allows alphanumeric, `.`, `-`, and `_`.
fn validate_event_prefix(prefix: &str) -> Result<(), String> {
    if prefix.is_empty() {
        return Err("prefix must not be empty".to_owned());
    }
    if prefix.len() > 128 {
        return Err("prefix exceeds 128 characters".to_owned());
    }
    if prefix.contains("..") {
        return Err("prefix must not contain '..'".to_owned());
    }
    if !prefix
        .chars()
        .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_')
    {
        return Err(
            "prefix may only contain alphanumeric, '.', '-', '_' characters".to_owned(),
        );
    }
    Ok(())
}

#[async_trait::async_trait(?Send)]
impl PolicyHandler for PolicyService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject();
        let allowed = self.policy_manager.check_with_domain(
            &subject.to_string(),
            "*",
            resource,
            operation,
        ).await;
        if allowed {
            Ok(())
        } else {
            anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
        }
    }

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

        // Check authorization — pass the operation string directly so dot-namespaced
        // actions (e.g. "ttt.writeback") are forwarded verbatim to the Casbin enforcer.
        let allowed = self.policy_manager.check_with_domain(
            &data.subject,
            &data.domain,
            &data.resource,
            &data.operation,
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

        let is_service_token = data.subject.as_ref().is_some_and(|s| s.starts_with("service:"));

        // Determine subject: explicit subject (if provided and authorized) or envelope identity.
        // JWT sub must contain a bare username (e.g. "randy", "birdetta") — the identity
        // system adds the namespace prefix ("token:randy") when the JWT is decoded.
        // For service tokens: sub = "service:{name}", e.g. "service:model".
        let subject = if let Some(ref subj) = data.subject.as_ref().filter(|s| !s.is_empty()) {
            // Explicit subject requires `manage` permission on `policy:IssueToken`
            // (matches the capnp type name used by the transport-level Casbin check).
            let caller = ctx.subject().to_string();
            let allowed = self.policy_manager.check_with_domain(
                &caller,
                "*",
                "policy:IssueToken",
                "manage",
            ).await;
            if !allowed {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!(
                        "Subject '{}' is not authorized to issue tokens on behalf of '{}'",
                        caller, subj
                    ),
                    code: "UNAUTHORIZED_SUBJECT".to_owned(),
                    details: "Requires 'manage' permission on 'policy:IssueToken'".to_owned(),
                }));
            }
            (*subj).clone()
        } else {
            // Use bare username from the envelope identity.
            ctx.user().to_owned()
        };

        // Validate TTL — service tokens get a longer default (7 days)
        let default_ttl = if is_service_token {
            data.ttl.filter(|&t| t != 0).unwrap_or(604800) // 7 days for service tokens
        } else {
            data.ttl.filter(|&t| t != 0).unwrap_or(self.token_config.default_ttl_seconds)
        };
        let requested_ttl = default_ttl;

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

        // Create and sign JWT with audience binding (RFC 8707)
        // Scopes are not embedded in JWT - Casbin enforces authorization server-side
        let now = chrono::Utc::now().timestamp();
        let audience = data.audience.as_ref().filter(|s| !s.is_empty()).cloned()
            .or_else(|| self.default_audience.clone());

        // Service tokens: cnf.jwk must be the service's REGISTERED key, never a
        // CA-derived guess (#441/#806) — bootstrap generates independent random
        // per-service keys (`load_or_generate_service_signing_key`), so a derived
        // pubkey here would not match the key the service actually holds. If no
        // registered key exists yet, error rather than sign a wrong key binding.
        // User tokens: decode the caller-provided pubkey (from OAuth consent page).
        let service_key_bytes: Option<[u8; 32]> = if is_service_token {
            let svc_name = &subject["service:".len()..];
            let trust = hyprstream_service::global_trust_store();
            let vk = trust.resolve_one(svc_name).ok_or_else(|| {
                anyhow!("service key '{svc_name}' not registered; refusing to issue a service token with a fabricated cnf.jwk")
            })?;
            Some(*vk.as_bytes())
        } else {
            use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
            data.user_pub_key.as_deref().and_then(|s| {
                URL_SAFE_NO_PAD.decode(s).ok()?.try_into().ok()
            })
        };

        // Populate iss with the OAuth issuer URL so federation peers can fetch JWKS.
        // default_audience is the OAuth issuer URL (set via with_default_audience).
        let issuer = self.default_audience.clone().unwrap_or_default();
        let mut claims = hyprstream_rpc::auth::Claims::new(
            subject,
            now,
            now + requested_ttl as i64,
        ).with_issuer(issuer)
         .with_audience(audience);

        // DPoP jkt takes priority over userPubKey (RFC 9449 § 6).
        if let Some(ref jkt) = data.dpop_jkt {
            claims.cnf = Some(hyprstream_rpc::auth::Cnf {
                jwk: None,
                jkt: Some(jkt.clone()),
            });
        } else if let Some(key_bytes) = service_key_bytes {
            claims = claims.with_cnf_jwk(&key_bytes);
        }

        let token = match self.sign_token(&claims, is_service_token).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Failed to issue token".to_owned(),
                    code: "SIGNING_NOT_CONFIGURED".to_owned(),
                    details: e.to_string(),
                }));
            }
        };

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
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ApplyTemplate,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let allowed = self.policy_manager.check_with_domain(
            &caller, "*", "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

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

        // Apply template rules via the Casbin enforcer.
        // Base rules are always present (injected at init/reload), so templates
        // only add their own rules on top. The enforcer's save_policy() persists
        // everything (base + template) to disk via the FileAdapter.
        if let Err(e) = self.policy_manager.apply_template(template).await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Failed to apply template: {}", e),
                code: "TEMPLATE_APPLY_FAILED".to_owned(),
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
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ApplyDraft,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let allowed = self.policy_manager.check_with_domain(
            &caller, "*", "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

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
        let commit_msg = data.message.as_ref().filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            format!("policy: update access control rules ({timestamp})")
        });

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
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RollbackPolicy,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let allowed = self.policy_manager.check_with_domain(
            &caller, "*", "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        info!("Rolling back policy to: {}", data.git_ref);

        // Validate git_ref to prevent shell injection or path traversal.
        // Accept: 40-hex SHA, short SHA (7+ hex chars), or simple branch/tag names.
        let git_ref = data.git_ref.trim().to_owned();
        {
            let valid = !git_ref.is_empty()
                && git_ref.len() <= 256
                && git_ref.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.'));
            if !valid {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Invalid git ref '{}': only [a-zA-Z0-9._/-] allowed", git_ref),
                    code: "INVALID_GIT_REF".into(),
                    details: String::new(),
                }));
            }
        }

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
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetHistory,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let allowed = self.policy_manager.check_with_domain(
            &caller, "*", "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

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
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetDiff,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let allowed = self.policy_manager.check_with_domain(
            &caller, "*", "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        let git_ref = data.git_ref.as_ref().filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| "HEAD".to_owned());

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
        ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let allowed = self.policy_manager.check_with_domain(
            &caller, "*", "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

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

    async fn handle_add_grouping(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &AddGrouping,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();

        // Fine-grained permission check: caller must have ttt.writeback on policy:roles
        let allowed = self.policy_manager.check_with_domain(
            &caller,
            "*",
            "policy:roles",
            "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "Subject '{}' is not authorized to manage role assignments",
                    caller
                ),
                code: "UNAUTHORIZED".to_owned(),
                details: "Requires 'ttt.writeback' permission on 'policy:roles'".to_owned(),
            }));
        }

        // Validate inputs
        if data.user.is_empty() || data.role.is_empty() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "user and role must be non-empty".to_owned(),
                code: "INVALID_INPUT".to_owned(),
                details: String::new(),
            }));
        }

        // Elevated roles can only be assigned by policy service.
        const ELEVATED_ROLES: &[&str] = &["ttt.privileged", "operator"];
        let caller_subject = ctx.subject().to_string();
        if ELEVATED_ROLES.contains(&data.role.as_str())
            && caller_subject != "service:policy"
        {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "Role '{}' is an elevated role and can only be assigned by policy service",
                    data.role
                ),
                code: "UNAUTHORIZED".to_owned(),
                details: "Elevated roles require service:policy identity".to_owned(),
            }));
        }

        // Callers cannot assign roles to themselves
        if data.user == caller {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "Cannot assign roles to yourself".to_owned(),
                code: "SELF_ASSIGNMENT".to_owned(),
                details: "Callers may not assign roles to themselves".to_owned(),
            }));
        }

        // Apply the role assignment
        self.policy_manager.add_role_for_user(&data.user, &data.role).await
            .map_err(|e| anyhow!("Failed to add role: {}", e))?;

        // Persist in-memory Casbin state to disk before staging
        self.policy_manager.save().await
            .map_err(|e| anyhow!("Failed to save policy after role grant: {}", e))?;

        // Commit to git
        let commit_msg = format!(
            "policy: grant role {} to {} [by {}]",
            data.role, data.user, caller
        );
        let sha = match self.stage_and_commit_policies(&commit_msg).await {
            Ok(sha) => sha,
            Err(e) => {
                warn!("Role granted but commit failed: {}", e);
                format!("(commit failed: {})", e)
            }
        };

        info!("Granted role '{}' to '{}' (caller={})", data.role, data.user, caller);
        Ok(PolicyResponseVariant::AddGroupingResult(sha))
    }

    async fn handle_remove_grouping(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RemoveGrouping,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();

        // Fine-grained permission check: caller must have ttt.writeback on policy:roles
        let allowed = self.policy_manager.check_with_domain(
            &caller,
            "*",
            "policy:roles",
            "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "Subject '{}' is not authorized to manage role assignments",
                    caller
                ),
                code: "UNAUTHORIZED".to_owned(),
                details: "Requires 'ttt.writeback' permission on 'policy:roles'".to_owned(),
            }));
        }

        // Validate inputs
        if data.user.is_empty() || data.role.is_empty() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "user and role must be non-empty".to_owned(),
                code: "INVALID_INPUT".to_owned(),
                details: String::new(),
            }));
        }

        // Remove the role assignment
        self.policy_manager.remove_role_for_user(&data.user, &data.role).await
            .map_err(|e| anyhow!("Failed to remove role: {}", e))?;

        // Persist in-memory Casbin state to disk before staging
        self.policy_manager.save().await
            .map_err(|e| anyhow!("Failed to save policy after role revoke: {}", e))?;

        // Commit to git
        let commit_msg = format!(
            "policy: revoke role {} from {} [by {}]",
            data.role, data.user, caller
        );
        let sha = match self.stage_and_commit_policies(&commit_msg).await {
            Ok(sha) => sha,
            Err(e) => {
                warn!("Role revoked but commit failed: {}", e);
                format!("(commit failed: {})", e)
            }
        };

        info!("Revoked role '{}' from '{}' (caller={})", data.role, data.user, caller);
        Ok(PolicyResponseVariant::RemoveGroupingResult(sha))
    }

    async fn handle_set_branch_visibility(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &SetBranchVisibility,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let resource = format!("model:{}:{}", data.model_name, data.branch_name);

        // Require manage (ttt.writeback) on the model resource
        let allowed = self.policy_manager.check_with_domain(
            &caller,
            "*",
            &resource,
            "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage {}", caller, resource),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        if data.public {
            // Make public: add wildcard infer+query rules
            let _ = self.policy_manager.add_policy_with_domain(
                "*", "*", &resource, "infer.generate", "allow").await;
            let _ = self.policy_manager.add_policy_with_domain(
                "*", "*", &resource, "query.status", "allow").await;
        } else {
            // Make private: remove wildcard rules
            let _ = self.policy_manager.remove_policy_with_domain(
                "*", "*", &resource, "infer.generate", "allow").await;
            let _ = self.policy_manager.remove_policy_with_domain(
                "*", "*", &resource, "query.status", "allow").await;
        }

        // Persist in-memory Casbin state to disk before staging
        self.policy_manager.save().await
            .map_err(|e| anyhow!("Failed to save policy after visibility change: {}", e))?;

        let vis_str = if data.public { "public" } else { "private" };
        let msg = format!(
            "policy: set {}/{} visibility={} [by {}]",
            data.model_name, data.branch_name, vis_str, caller
        );
        let sha = match self.stage_and_commit_policies(&msg).await {
            Ok(sha) => sha,
            Err(e) => {
                warn!("Visibility set but commit failed: {}", e);
                format!("(commit failed: {})", e)
            }
        };

        info!(
            "Set branch {}/{} to {} (caller={})",
            data.model_name, data.branch_name, vis_str, caller
        );
        Ok(PolicyResponseVariant::SetBranchVisibilityResult(sha))
    }

    /// Publisher registers a topic prefix. No group key stored here — publisher holds it.
    async fn handle_register_event_prefix(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterEventPrefix,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope (prevents Casbin metacharacter injection)
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: publish:events:{prefix}.*
        let scope = format!("publish:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "register").await?;

        let mut pubkey = [0u8; 32];
        if data.publisher_ephemeral_pubkey.len() != 32 {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "publisher pubkey must be 32 bytes".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        pubkey.copy_from_slice(&data.publisher_ephemeral_pubkey);

        let mut prefixes = self.event_prefixes.write().await;
        prefixes.insert(data.prefix.clone(), EventPrefixState {
            publisher_pubkey: pubkey,
            schema: data.schema.clone(),
            subscriber_pubkeys: HashMap::new(),
            wrapped_keys: HashMap::new(),
        });

        tracing::info!(prefix = %data.prefix, "Registered event prefix");
        Ok(PolicyResponseVariant::RegisterEventPrefixResult)
    }

    /// Subscriber requests access. Checks scope, stores subscriber pubkey,
    /// returns publisher pubkey + any pre-wrapped key blob.
    async fn handle_subscribe_event_prefix(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &SubscribeEventPrefix,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: subscribe:events:{prefix}.*
        let scope = format!("subscribe:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "subscribe").await?;

        let mut sub_pubkey = [0u8; 32];
        if data.subscriber_ephemeral_pubkey.len() != 32 {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "subscriber pubkey must be 32 bytes".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        sub_pubkey.copy_from_slice(&data.subscriber_ephemeral_pubkey);

        let sub_hash = blake3::hash(&sub_pubkey);
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(sub_hash.as_bytes());

        let mut prefixes = self.event_prefixes.write().await;
        let state = match prefixes.get_mut(&data.prefix) {
            Some(s) => s,
            None => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("prefix '{}' not registered", data.prefix),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        };

        state.subscriber_pubkeys.insert(hash_bytes, sub_pubkey);

        // If publisher has already wrapped a key for this subscriber, return it.
        let wrapped = state.wrapped_keys.get(&hash_bytes).cloned().unwrap_or_default();

        Ok(PolicyResponseVariant::SubscribeEventPrefixResult(EventPrefixAccess {
            publisher_ephemeral_pubkey: state.publisher_pubkey.to_vec(),
            wrapped_group_key: wrapped,
            schema: state.schema.clone(),
        }))
    }

    /// Publisher fetches new subscriber pubkeys that need wrapping.
    async fn handle_get_pending_subscribers(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetPendingSubscribers,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: publish:events:{prefix}.*
        let scope = format!("publish:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "get_subscribers").await?;

        let prefixes = self.event_prefixes.read().await;
        let state = match prefixes.get(&data.prefix) {
            Some(s) => s,
            None => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("prefix '{}' not registered", data.prefix),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        };

        // Return pubkeys that don't have wrapped keys yet.
        let pending: Vec<Vec<u8>> = state.subscriber_pubkeys.iter()
            .filter(|(hash, _)| !state.wrapped_keys.contains_key(*hash))
            .map(|(_, pubkey)| pubkey.to_vec())
            .collect();

        Ok(PolicyResponseVariant::GetPendingSubscribersResult(PendingSubscribers {
            pubkeys: pending,
        }))
    }

    /// Publisher deposits wrapped group key blobs (opaque to PolicyService).
    async fn handle_deposit_wrapped_keys(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &DepositWrappedKeys,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: publish:events:{prefix}.*
        let scope = format!("publish:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "deposit_keys").await?;

        let mut prefixes = self.event_prefixes.write().await;
        let state = match prefixes.get_mut(&data.prefix) {
            Some(s) => s,
            None => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("prefix '{}' not registered", data.prefix),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        };

        let mut deposited = 0u32;
        for entry in &data.entries {
            if entry.sub_pubkey_hash.len() != 32 {
                warn!(
                    prefix = %data.prefix,
                    hash_len = entry.sub_pubkey_hash.len(),
                    "Rejecting malformed wrapped key entry: sub_pubkey_hash must be 32 bytes"
                );
                continue;
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&entry.sub_pubkey_hash);
            state.wrapped_keys.insert(hash, entry.wrapped_blob.clone());
            deposited += 1;
        }

        tracing::debug!(
            prefix = %data.prefix,
            deposited,
            submitted = data.entries.len(),
            "Deposited wrapped keys"
        );
        Ok(PolicyResponseVariant::DepositWrappedKeysResult)
    }

    async fn handle_resolve_service_key(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ResolveServiceKey,
    ) -> Result<PolicyResponseVariant> {
        // #441: authoritative resolution — return the REGISTERED key or ERROR.
        // We never derive an "expected" key from the root CA as a fallback: a
        // consumer must never receive a key the signer didn't actually register,
        // because a guessed key produces a silent mis-verify ("Response signed by
        // unexpected key") three layers away at the envelope check. Registered-or-
        // error converts that into a clear, early failure.
        let trust = hyprstream_service::global_trust_store();
        let vk = trust.resolve_one(&data.service_name)
            .ok_or_else(|| anyhow!("service key '{}' not registered", data.service_name))?;
        let att = trust.get(&vk)
            .ok_or_else(|| anyhow!("service key '{}' not registered (no attestation)", data.service_name))?;
        debug!("Resolved service key for '{}'", data.service_name);
        Ok(PolicyResponseVariant::ResolveServiceKeyResult(
            ServiceKeyResponse {
                verifying_key: vk.to_bytes().to_vec(),
                service_jwt: att.jwt.clone(),
            }
        ))
    }

    async fn handle_register_service_key(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterServiceKey,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();

        // Verify the caller is who they claim to be.
        // The service JWT must be signed by the CA (our jwt_signing_key) and
        // its subject must match "service:{serviceName}".
        let claims = hyprstream_rpc::auth::jwt::decode_with_key(
            &data.service_jwt,
            &self.jwt_signing_key.verifying_key(),
            None,
        ).map_err(|e| anyhow!("Invalid service JWT: {e}"))?;

        let expected_sub = format!("service:{}", data.service_name);
        if claims.sub != expected_sub {
            anyhow::bail!(
                "JWT subject '{}' does not match service name '{}'",
                claims.sub, data.service_name
            );
        }

        // Verify the provided verifying key matches the JWT's cnf.jwk claim.
        let vk_bytes: [u8; 32] = data.verifying_key.as_slice().try_into()
            .map_err(|_| anyhow!("verifying_key must be 32 bytes"))?;
        let vk = VerifyingKey::from_bytes(&vk_bytes)
            .map_err(|e| anyhow!("Invalid Ed25519 verifying key: {e}"))?;

        if let Some(cnf_bytes) = claims.cnf_key_bytes() {
            if cnf_bytes != vk_bytes {
                anyhow::bail!("JWT cnf.jwk does not match provided verifying key");
            }
        }

        // Store in trust store (key-centric: the key IS the identity)
        {
            let trust = hyprstream_service::global_trust_store();
            trust.insert(vk, hyprstream_service::Attestation {
                scopes: std::iter::once(data.service_name.clone()).collect(),
                subject: None,
                jwt: Some(data.service_jwt.clone()),
                expires_at: claims.exp,
                attested_by: Some(self.signing_key.verifying_key().to_bytes()),
            });
        }

        info!(service = %data.service_name, caller = %caller, "Registered service verifying key");

        Ok(PolicyResponseVariant::RegisterServiceKeyResult)
    }

    async fn handle_refresh_service_token(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RefreshServiceTokenRequest,
    ) -> Result<PolicyResponseVariant> {
        const MAX_TTL: i64 = 2_592_000; // 30 days
        const MIN_TTL: i64 = 3_600;     // 1 hour

        let subject = match ctx.subject().name() {
            Some(s) if s.starts_with("service:") => s.to_owned(),
            Some(s) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Subject '{s}' is not a service identity; only services may self-renew"),
                    code: "NOT_A_SERVICE".to_owned(),
                    details: String::new(),
                }));
            }
            None => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Anonymous callers cannot refresh service tokens".to_owned(),
                    code: "ANONYMOUS".to_owned(),
                    details: String::new(),
                }));
            }
        };

        let ttl = data.ttl_seconds.clamp(MIN_TTL, MAX_TTL);
        let svc_name = &subject["service:".len()..];

        let now = chrono::Utc::now().timestamp();
        let expires_at = now + ttl;

        // cnf.jwk must be the service's REGISTERED key, never a CA-derived guess
        // (#441/#806) — see handle_issue_token for the rationale. Registered-or-
        // error: refuse to refresh a service token we can't bind to a real key.
        let trust = hyprstream_service::global_trust_store();
        let vk = trust.resolve_one(svc_name).ok_or_else(|| {
            anyhow!("service key '{svc_name}' not registered; refusing to refresh a service token with a fabricated cnf.jwk")
        })?;

        let issuer = self.default_audience.clone().unwrap_or_default();
        let claims = hyprstream_rpc::auth::Claims::new(subject.clone(), now, expires_at)
            .with_issuer(issuer)
            .with_cnf_jwk(vk.as_bytes());

        let token = match self.sign_token(&claims, true).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Failed to refresh service token".to_owned(),
                    code: "SIGNING_NOT_CONFIGURED".to_owned(),
                    details: e.to_string(),
                }));
            }
        };

        // Persist renewed JWT to disk so it survives a server restart
        let credentials_dir = crate::auth::identity_store::credentials_dir()?;
        if let Err(e) = crate::auth::identity_store::write_service_jwt(&credentials_dir, svc_name, &token) {
            warn!(service = svc_name, "Failed to persist renewed JWT to disk: {e}");
        }

        info!(service = svc_name, expires_at, "Renewed service JWT");
        Ok(PolicyResponseVariant::RefreshServiceTokenResult(TokenInfo {
            token,
            expires_at,
        }))
    }
    async fn handle_exchange_wit(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ExchangeWit,
    ) -> Result<PolicyResponseVariant> {
        // Identity is read from the already-verified envelope WIT — no credential submission.
        let sub = match ctx.subject().name() {
            Some(s) => s.to_owned(),
            None => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Anonymous callers cannot exchange WIT for access token".to_owned(),
                    code: "ANONYMOUS".to_owned(),
                    details: String::new(),
                }));
            }
        };

        // cnf.jwk from the verified WIT — carried through into the issued at+jwt.
        let cnf_key_bytes = ctx.claims().and_then(hyprstream_rpc::auth::Claims::cnf_key_bytes);
        if cnf_key_bytes.is_none() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "Caller WIT missing cnf.jwk — key binding required for ExchangeWit".to_owned(),
                code: "NO_CNF_JWK".to_owned(),
                details: String::new(),
            }));
        }

        // Casbin: caller must have 'exchange' on 'policy:exchange-wit'.
        let allowed = self.policy_manager.check_with_domain(
            &sub,
            "*",
            "policy:exchange-wit",
            "exchange",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Subject '{sub}' is not authorized to exchange WIT for access token"),
                code: "UNAUTHORIZED".to_owned(),
                details: "Requires 'exchange' permission on 'policy:exchange-wit'".to_owned(),
            }));
        }

        let ttl = data.ttl.unwrap_or(self.token_config.default_ttl_seconds);
        let ttl = ttl.clamp(60, self.token_config.max_ttl_seconds);

        let now = chrono::Utc::now().timestamp();
        let expires_at = now + ttl as i64;

        let audience = data.audience.as_ref().filter(|s| !s.is_empty()).cloned()
            .or_else(|| self.default_audience.clone());

        let issuer = self.default_audience.clone().unwrap_or_default();
        let mut claims = hyprstream_rpc::auth::Claims::new(sub.clone(), now, expires_at)
            .with_issuer(issuer)
            .with_audience(audience);

        // Key binding: carry cnf.jwk from WIT into the at+jwt.
        if let Some(key_bytes) = cnf_key_bytes {
            claims = claims.with_cnf_jwk(&key_bytes);
        }

        let token = match self.sign_token(&claims, false).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Failed to issue WIT".to_owned(),
                    code: "SIGNING_NOT_CONFIGURED".to_owned(),
                    details: e.to_string(),
                }));
            }
        };

        info!(sub = %sub, expires_at, "ExchangeWit: issued at+jwt");
        Ok(PolicyResponseVariant::ExchangeWitResult(TokenInfo {
            token,
            expires_at,
        }))
    }
}

#[async_trait(?Send)]
impl RequestService for PolicyService {
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

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        (*self.signing_key).clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.default_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.jwt_key_source.clone()
    }

    fn resolve_key_subject(&self, signer_pubkey: &[u8; 32]) -> Option<hyprstream_rpc::envelope::Subject> {
        hyprstream_service::global_trust_store().resolve_subject(signer_pubkey)
    }

    fn jti_blocklist(&self) -> Option<&dyn hyprstream_rpc::auth::JtiBlocklist> {
        Some(self.jti_blocklist.as_ref())
    }

    fn cache_key_binding(
        &self,
        verifying_key: ed25519_dalek::VerifyingKey,
        subject: &str,
        jwt: &str,
        expires_at: i64,
    ) {
        hyprstream_service::global_trust_store().insert(verifying_key, hyprstream_service::Attestation {
            scopes: std::collections::HashSet::new(),
            subject: Some(subject.to_owned()),
            jwt: Some(jwt.to_owned()),
            expires_at,
            attested_by: Some(self.signing_key.verifying_key().to_bytes()),
        });
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
                EventKind::Modify(_) | EventKind::Create(_)
                    if event.paths.iter().any(|p| p.ends_with("policy.csv")) =>
                {
                    let _ = tx.blocking_send(());
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
