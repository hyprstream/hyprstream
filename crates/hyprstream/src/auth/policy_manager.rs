//! Casbin-powered policy manager for RBAC/ABAC access control
//!
//! Policies are stored in the `.registry/policies/` directory:
//! - `model.conf` - Casbin access control model definition
//! - `policy.csv` - Authorization rules
//!
//! Policy format follows Casbin conventions:
//! ```text
//! # model.conf defines the policy structure
//! [request_definition]
//! r = sub, obj, act
//!
//! [policy_definition]
//! p = sub, obj, act
//!
//! [role_definition]
//! g = _, _
//!
//! [policy_effect]
//! e = some(where (p.eft == allow))
//!
//! [matchers]
//! m = g(r.sub, p.sub) && keyMatch(r.obj, p.obj) && r.act == p.act
//! ```
//!
//! ```text
//! # policy.csv defines the actual rules
//! p, operator, *, *, *, allow
//! p, trainer, *, model:*, infer, allow
//! p, trainer, *, model:*, train, allow
//! g, alice, trainer
//! ```

use crate::auth::policy_templates::{base_policies_to_csv, base_policies_to_vec, ServiceGrouping, ServicePolicyRule};
use crate::auth::Operation;
use casbin::{CoreApi, DefaultModel, Enforcer, FileAdapter, MemoryAdapter, MgmtApi, RbacApi};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

static GLOBAL_POLICY_MANAGER: OnceLock<Arc<PolicyManager>> = OnceLock::new();

/// Set the process-global PolicyManager. Called once by the PolicyService factory.
pub fn set_global_policy_manager(pm: Arc<PolicyManager>) {
    let _ = GLOBAL_POLICY_MANAGER.set(pm);
}

/// Access the process-global PolicyManager. Returns `None` before PolicyService starts.
pub fn global_policy_manager() -> Option<Arc<PolicyManager>> {
    GLOBAL_POLICY_MANAGER.get().cloned()
}

/// Errors from policy operations
#[derive(Error, Debug)]
pub enum PolicyError {
    #[error("Failed to load policy model: {0}")]
    ModelLoadError(String),

    #[error("Failed to load policy rules: {0}")]
    PolicyLoadError(String),

    #[error("Failed to save policy: {0}")]
    PolicySaveError(String),

    #[error("Casbin error: {0}")]
    CasbinError(#[from] casbin::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Policy directory not found: {0}")]
    PolicyDirNotFound(PathBuf),

    #[error("Invalid policy component: {0}")]
    ValidationError(String),
}

/// Maximum length for policy components (user, resource, operation)
const MAX_POLICY_COMPONENT_LEN: usize = 256;

/// Validate a policy component for security
///
/// Rejects:
/// - Components longer than 256 characters
/// - Null bytes
/// - Line breaks (CSV injection prevention)
fn validate_policy_component(component: &str, name: &str) -> Result<(), PolicyError> {
    if component.len() > MAX_POLICY_COMPONENT_LEN {
        return Err(PolicyError::ValidationError(format!(
            "{name} exceeds maximum length of {MAX_POLICY_COMPONENT_LEN} characters"
        )));
    }

    if component.contains('\0') {
        return Err(PolicyError::ValidationError(format!(
            "{name} contains null byte"
        )));
    }

    if component.contains('\n') || component.contains('\r') {
        return Err(PolicyError::ValidationError(format!(
            "{name} contains line break (potential CSV injection)"
        )));
    }

    if component.contains(',') {
        return Err(PolicyError::ValidationError(format!(
            "{name} contains comma (potential CSV injection)"
        )));
    }

    // NOTE: wildcards ('*') are intentionally NOT rejected here. The Casbin model
    // (DEFAULT_MODEL_CONF) is wildcard-based: every rule carries domain="*", and
    // legitimate features rely on wildcard subjects/resources/actions — public
    // models ("*" subject, see PolicyService::handle_set_branch_visibility),
    // role-scoped resource grants ("model:*"), action namespaces ("ttt.*"), and
    // the OIDC self-ownership grant. A blanket '*' ban here broke all of those
    // (and made add_policy() — which injects domain="*" — always fail) while
    // adding no real protection: who may mutate policy is enforced at the
    // PolicyService RPC authz layer (the correct place for anti-escalation), not
    // by banning '*' in policy *content*. The null/newline/comma/length checks
    // above remain — those are genuine CSV-injection / DoS guards.

    Ok(())
}

/// Default Casbin model configuration for hyprstream RBAC with domain support
///
/// Request format: (subject, domain, object, action)
/// Policy format: (subject, domain, object, action, effect)
///
/// Supports:
/// - Domain-scoped access control (e.g., HfModel, HfDataset)
/// - Deny rules via explicit effect column
/// - Wildcards for domain and action
/// - Role-based access via g()
const DEFAULT_MODEL_CONF: &str = r#"[request_definition]
r = sub, dom, obj, act

[policy_definition]
p = sub, dom, obj, act, eft

[role_definition]
g = _, _
g2 = _, _, _

[policy_effect]
e = some(where (p.eft == allow)) && !some(where (p.eft == deny))

[matchers]
m = (g(r.sub, p.sub) || g2(r.sub, p.sub, r.dom) || keyMatch(r.sub, p.sub)) && \
    (p.dom == "*" || r.dom == p.dom) && \
    keyMatch(r.obj, p.obj) && \
    (p.act == "*" || keyMatch(r.act, p.act))
"#;

/// Default policy rules: deny-by-default with service grants
///
/// All access is denied until the operator explicitly configures policy via:
///   hyprstream quick policy apply-template public-inference  # open inference API
///
/// Service-to-service rules are defined in [`policy_templates::SERVICE_BASE_POLICIES`].
///
/// Policy format: p, subject, domain, resource, action, effect
fn default_policy_csv() -> String {
    format!(
        "# Hyprstream Access Control Policy — deny-by-default\n\
        # Service-to-service base rules\n\
        {}\n\
        # The TUI display server is a local-only service; anonymous browser clients\n\
        # (connecting via WebTransport from localhost) may access TUI resources.\n\
        p, anonymous, *, tui:*, *, allow\n\
        #\n\
        # No other access is granted until you apply a policy template:\n\
        #\n\
        #   hyprstream quick policy apply-template public-inference  # anonymous inference\n\
        #   hyprstream quick policy apply-template public-read       # anonymous registry browse\n\
        #\n\
        # Or add rules manually:\n\
        #   p, alice, *, *, *, allow                           # Alice full access\n\
        #   p, alice, *, model:*, infer, allow                 # Alice can infer\n\
        #   p, anonymous, *, inference:*, infer, allow         # public inference\n\
        #\n",
        base_policies_to_csv(),
    )
}

/// Validate policy.csv format before loading
///
/// Ensures all policy lines have the correct number of fields:
/// - Policy lines (p, ...): 6 fields (p, subject, domain, resource, action, effect)
/// - Global role lines (g, ...): 3 fields (g, user, role)
/// - Domain role lines (g2, ...): 4 fields (g2, user, role, domain)
fn validate_policy_csv(policy_path: &Path) -> Result<(), PolicyError> {
    let content = std::fs::read_to_string(policy_path)
        .map_err(PolicyError::IoError)?;

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Check policy lines (p, ...)
        if line.starts_with("p,") || line.starts_with("p ") {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 6 {  // "p" + 5 fields
                return Err(PolicyError::ValidationError(format!(
                    "Invalid policy format at line {}:\n\
                    Found:    {}\n\
                    Expected: p, subject, domain, resource, action, effect\n\
                    Example:  p, alice, *, *, *, allow\n\n\
                    Your policy has {} fields but 6 are required.\n\
                    Edit {} to fix the format.",
                    line_num + 1, line, fields.len(), policy_path.display()
                )));
            }
        }

        // Check role lines (g, ...)
        if line.starts_with("g,") || line.starts_with("g ") {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 3 {  // "g" + 2 fields
                return Err(PolicyError::ValidationError(format!(
                    "Invalid role format at line {}:\n\
                    Found:    {}\n\
                    Expected: g, user, role\n\
                    Example:  g, alice, trainer",
                    line_num + 1, line
                )));
            }
        }

        // Domain-scoped role membership (g2, user, role, domain)
        if line.starts_with("g2,") || line.starts_with("g2 ") {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 4 {
                return Err(PolicyError::ValidationError(format!(
                    "Invalid domain role format at line {}:\n\
                    Found:    {}\n\
                    Expected: g2, user, role, domain\n\
                    Example:  g2, alice, trainer, tenant-a",
                    line_num + 1,
                    line
                )));
            }
        }
    }
    Ok(())
}

/// Write a policy file with restrictive permissions (0o640 on Unix).
///
/// Policy files contain authorization rules and should not be world-readable.
/// Uses tokio::fs::write then sets permissions afterward. The policies directory
/// itself should already be 0o750, limiting the window where the file is exposed.
pub async fn write_policy_file(path: &Path, content: impl AsRef<[u8]>) -> Result<(), PolicyError> {
    tokio::fs::write(path, content).await?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).await?;
    }
    Ok(())
}

async fn ensure_domain_role_model(model_path: &Path) -> Result<(), PolicyError> {
    let content = tokio::fs::read_to_string(model_path).await?;
    if content.contains("g2 = _, _, _")
        && content.contains("g2(r.sub, p.sub, r.dom)")
    {
        return Ok(());
    }

    let migrated = content
        .replacen(
            "[role_definition]\ng = _, _",
            "[role_definition]\ng = _, _\ng2 = _, _, _",
            1,
        )
        .replacen(
            "m = (g(r.sub, p.sub) || keyMatch(r.sub, p.sub))",
            "m = (g(r.sub, p.sub) || g2(r.sub, p.sub, r.dom) || keyMatch(r.sub, p.sub))",
            1,
        );
    if !migrated.contains("g2 = _, _, _")
        || !migrated.contains("g2(r.sub, p.sub, r.dom)")
    {
        return Err(PolicyError::ValidationError(format!(
            "{} must define domain-scoped g2(user, role, domain) membership",
            model_path.display()
        )));
    }

    write_policy_file(model_path, migrated).await?;
    Ok(())
}

/// Policy manager wrapping Casbin enforcer
pub struct PolicyManager {
    /// Casbin enforcer
    enforcer: Arc<RwLock<Enforcer>>,
    /// Path to policies directory (.registry/policies/)
    policies_dir: PathBuf,
}

impl PolicyManager {
    /// Create a new PolicyManager loading policies from the given directory
    ///
    /// If the policies directory doesn't exist, creates it with default policies.
    pub async fn new(policies_dir: impl AsRef<Path>) -> Result<Self, PolicyError> {
        let policies_dir = policies_dir.as_ref().to_path_buf();

        // Ensure policies directory exists with restricted permissions
        if !policies_dir.exists() {
            info!("Creating policies directory at {:?}", policies_dir);
            tokio::fs::create_dir_all(&policies_dir).await?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                tokio::fs::set_permissions(&policies_dir, std::fs::Permissions::from_mode(0o750)).await?;
            }
        }

        let model_path = policies_dir.join("model.conf");
        let policy_path = policies_dir.join("policy.csv");

        // Create default model.conf if not exists
        if !model_path.exists() {
            info!("Creating default model.conf");
            write_policy_file(&model_path, DEFAULT_MODEL_CONF).await?;
        }
        ensure_domain_role_model(&model_path).await?;

        // Create default policy.csv if not exists
        if !policy_path.exists() {
            info!("Creating default policy.csv (deny-by-default — run `hyprstream quick policy apply-template <name>` to grant access; see `policy list-templates`)");
            write_policy_file(&policy_path, default_policy_csv()).await?;
        }

        // Validate policy.csv format before loading
        // This catches common errors like missing domain or effect columns
        validate_policy_csv(&policy_path)?;

        // Load the model
        let model = DefaultModel::from_file(&model_path)
            .await
            .map_err(|e| PolicyError::ModelLoadError(e.to_string()))?;

        // Create enforcer with file adapter
        let adapter = FileAdapter::new(policy_path.to_string_lossy().to_string());
        let mut enforcer = Enforcer::new(model, adapter)
            .await
            .map_err(|e| PolicyError::PolicyLoadError(e.to_string()))?;

        // Inject SERVICE_BASE_POLICIES into the in-memory enforcer.
        // These are code-defined infrastructure rules (service key resolution,
        // policy checks, registry access) that must always be present regardless
        // of what's in the on-disk policy.csv.
        //
        // Casbin's add_policies() aborts the ENTIRE batch if ANY rule already
        // exists (early return on first duplicate), so we must filter out
        // rules that were already loaded from the CSV.
        let base = base_policies_to_vec();
        let existing = enforcer.get_policy();
        let new_rules: Vec<Vec<String>> = base
            .into_iter()
            .filter(|rule| !existing.contains(rule))
            .collect();
        if !new_rules.is_empty() {
            enforcer
                .add_policies(new_rules)
                .await
                .map_err(PolicyError::CasbinError)?;
        }

        info!(
            "PolicyManager initialized with {} policies",
            enforcer.get_policy().len()
        );

        Ok(Self {
            enforcer: Arc::new(RwLock::new(enforcer)),
            policies_dir,
        })
    }

    /// Create a PolicyManager with allow-all policies (for tests only).
    ///
    /// WARNING: Not for production use. Production uses `new()` which loads
    /// deny-by-default policies from disk.
    pub async fn permissive() -> Result<Self, PolicyError> {
        // Use the same model as DEFAULT_MODEL_CONF for consistency
        let model = DefaultModel::from_str(DEFAULT_MODEL_CONF)
            .await
            .map_err(|e| PolicyError::ModelLoadError(e.to_string()))?;

        let adapter = MemoryAdapter::default();
        let mut enforcer = Enforcer::new(model, adapter)
            .await
            .map_err(|e| PolicyError::PolicyLoadError(e.to_string()))?;

        // Add permissive default rule: anyone can do anything in any domain
        enforcer
            .add_policy(vec![
                "*".to_owned(),      // subject
                "*".to_owned(),      // domain
                "*".to_owned(),      // resource
                "*".to_owned(),      // action
                "allow".to_owned(),  // effect
            ])
            .await
            .map_err(PolicyError::CasbinError)?;

        Ok(Self {
            enforcer: Arc::new(RwLock::new(enforcer)),
            policies_dir: PathBuf::new(),
        })
    }

    /// Create a PolicyManager with a clean in-memory adapter and no pre-seeded rules.
    ///
    /// Intended for unit tests that need a blank-slate enforcer (deny-by-default)
    /// without touching the filesystem or inheriting the permissive `*` rule from
    /// [`permissive()`].
    ///
    /// WARNING: Not for production use.
    #[cfg(test)]
    pub async fn new_in_memory() -> Result<Self, PolicyError> {
        let model = DefaultModel::from_str(DEFAULT_MODEL_CONF)
            .await
            .map_err(|e| PolicyError::ModelLoadError(e.to_string()))?;

        let adapter = MemoryAdapter::default();
        let enforcer = Enforcer::new(model, adapter)
            .await
            .map_err(|e| PolicyError::PolicyLoadError(e.to_string()))?;

        Ok(Self {
            enforcer: Arc::new(RwLock::new(enforcer)),
            policies_dir: PathBuf::new(),
        })
    }

    /// Check if a user is allowed to perform an operation on a resource
    ///
    /// Uses wildcard "*" for domain. For domain-specific checks, use `check_with_domain()`.
    ///
    /// # Arguments
    /// * `user` - The user or role requesting access
    /// * `resource` - The resource being accessed (e.g., "model:qwen3-small")
    /// * `operation` - The operation being requested
    ///
    /// # Returns
    /// `true` if access is allowed, `false` otherwise
    pub async fn check(
        &self,
        user: &str,
        resource: &str,
        operation: Operation,
    ) -> bool {
        self.check_with_domain(user, "*", resource, operation.as_str()).await
    }

    /// Check if a user is allowed to perform an operation on a resource in a specific domain
    ///
    /// Accepts any `&str` for `operation`, supporting both classic enum-backed actions
    /// (e.g., `"infer"`, `"train"`) and dot-namespaced actions (e.g., `"ttt.writeback"`).
    ///
    /// # Arguments
    /// * `user` - The user or role requesting access
    /// * `domain` - The domain (e.g., "HfModel", "HfDataset")
    /// * `resource` - The resource being accessed (e.g., "model:qwen3-small")
    /// * `operation` - The operation string (e.g., "infer", "ttt.writeback", "*")
    ///
    /// # Returns
    /// `true` if access is allowed, `false` otherwise
    pub async fn check_with_domain(
        &self,
        user: &str,
        domain: &str,
        resource: &str,
        operation: &str,
    ) -> bool {
        let enforcer = self.enforcer.read().await;
        match enforcer.enforce((user, domain, resource, operation)) {
            Ok(allowed) => {
                debug!(
                    "Policy check: user={}, domain={}, resource={}, op={} -> {}",
                    user,
                    domain,
                    resource,
                    operation,
                    if allowed { "ALLOW" } else { "DENY" }
                );
                allowed
            }
            Err(e) => {
                warn!("Policy enforcement error: {}", e);
                false // Deny on error
            }
        }
    }

    /// Synchronous policy check (blocks on the async check)
    ///
    /// Use this in synchronous contexts. For async code, prefer `check()`.
    pub fn check_sync(&self, user: &str, resource: &str, operation: Operation) -> bool {
        // Try to use existing runtime, or create a temporary one
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // We're in an async context, use block_in_place
            tokio::task::block_in_place(|| {
                handle.block_on(self.check(user, resource, operation))
            })
        } else {
            // No runtime, create a temporary one (less efficient)
            let rt = tokio::runtime::Runtime::new().ok();
            rt.map(|rt| rt.block_on(self.check(user, resource, operation)))
                .unwrap_or(false)
        }
    }

    /// Check multiple operations at once, returning a bitmask
    pub async fn check_all(
        &self,
        user: &str,
        resource: &str,
    ) -> u32 {
        let mut mask = 0u32;
        for op in Operation::all() {
            if self.check(user, resource, *op).await {
                mask |= 1 << (op.code() as u32 - 'a' as u32);
            }
        }
        mask
    }

    /// Add a policy rule (uses wildcard "*" for domain, "allow" for effect)
    ///
    /// For domain-specific policies, use `add_policy_with_domain()`.
    ///
    /// All inputs are validated for:
    /// - Maximum length (256 characters)
    /// - No null bytes
    /// - No line breaks (CSV injection prevention)
    pub async fn add_policy(
        &self,
        user: &str,
        resource: &str,
        operation: &str,
    ) -> Result<bool, PolicyError> {
        self.add_policy_with_domain(user, "*", resource, operation, "allow")
            .await
    }

    /// Add a policy rule with domain and effect
    ///
    /// # Arguments
    /// * `user` - The user or role this policy applies to
    /// * `domain` - The domain (e.g., "HfModel", "HfDataset", "*" for all)
    /// * `resource` - The resource pattern (e.g., "model:*", "data:sales-*")
    /// * `operation` - The operation (e.g., "infer", "train", "*" for all)
    /// * `effect` - "allow" or "deny"
    pub async fn add_policy_with_domain(
        &self,
        user: &str,
        domain: &str,
        resource: &str,
        operation: &str,
        effect: &str,
    ) -> Result<bool, PolicyError> {
        // Validate all inputs
        validate_policy_component(user, "user")?;
        validate_policy_component(domain, "domain")?;
        validate_policy_component(resource, "resource")?;
        validate_policy_component(operation, "operation")?;
        validate_policy_component(effect, "effect")?;

        // Validate effect is "allow" or "deny"
        if effect != "allow" && effect != "deny" {
            return Err(PolicyError::ValidationError(
                "effect must be 'allow' or 'deny'".to_owned(),
            ));
        }

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .add_policy(vec![
                user.to_owned(),
                domain.to_owned(),
                resource.to_owned(),
                operation.to_owned(),
                effect.to_owned(),
            ])
            .await
            .map_err(PolicyError::CasbinError)
    }

    /// Remove a policy rule (uses wildcard "*" for domain, "allow" for effect)
    ///
    /// For domain-specific policies, use `remove_policy_with_domain()`.
    pub async fn remove_policy(
        &self,
        user: &str,
        resource: &str,
        operation: &str,
    ) -> Result<bool, PolicyError> {
        self.remove_policy_with_domain(user, "*", resource, operation, "allow")
            .await
    }

    /// Remove a policy rule with domain and effect
    ///
    /// All inputs are validated for security.
    pub async fn remove_policy_with_domain(
        &self,
        user: &str,
        domain: &str,
        resource: &str,
        operation: &str,
        effect: &str,
    ) -> Result<bool, PolicyError> {
        // Validate all inputs
        validate_policy_component(user, "user")?;
        validate_policy_component(domain, "domain")?;
        validate_policy_component(resource, "resource")?;
        validate_policy_component(operation, "operation")?;
        validate_policy_component(effect, "effect")?;

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .remove_policy(vec![
                user.to_owned(),
                domain.to_owned(),
                resource.to_owned(),
                operation.to_owned(),
                effect.to_owned(),
            ])
            .await
            .map_err(PolicyError::CasbinError)
    }

    /// Add a role for a user
    ///
    /// All inputs are validated for security.
    pub async fn add_role_for_user(
        &self,
        user: &str,
        role: &str,
    ) -> Result<bool, PolicyError> {
        // Validate all inputs
        validate_policy_component(user, "user")?;
        validate_policy_component(role, "role")?;

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .add_grouping_policy(vec![user.to_owned(), role.to_owned()])
            .await
            .map_err(PolicyError::CasbinError)
    }

    /// Remove a role from a user
    ///
    /// All inputs are validated for security.
    pub async fn remove_role_for_user(
        &self,
        user: &str,
        role: &str,
    ) -> Result<bool, PolicyError> {
        // Validate all inputs
        validate_policy_component(user, "user")?;
        validate_policy_component(role, "role")?;

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .remove_grouping_policy(vec![user.to_owned(), role.to_owned()])
            .await
            .map_err(PolicyError::CasbinError)
    }

    /// Add a role membership scoped to one verified tenant/domain.
    pub async fn add_role_for_user_in_domain(
        &self,
        user: &str,
        role: &str,
        domain: &str,
    ) -> Result<bool, PolicyError> {
        validate_policy_component(user, "user")?;
        validate_policy_component(role, "role")?;
        validate_policy_component(domain, "domain")?;

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .add_named_grouping_policy(
                "g2",
                vec![user.to_owned(), role.to_owned(), domain.to_owned()],
            )
            .await
            .map_err(PolicyError::CasbinError)
    }

    /// Remove only the role membership belonging to one tenant/domain.
    pub async fn remove_role_for_user_in_domain(
        &self,
        user: &str,
        role: &str,
        domain: &str,
    ) -> Result<bool, PolicyError> {
        validate_policy_component(user, "user")?;
        validate_policy_component(role, "role")?;
        validate_policy_component(domain, "domain")?;

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .remove_named_grouping_policy(
                "g2",
                vec![user.to_owned(), role.to_owned(), domain.to_owned()],
            )
            .await
            .map_err(PolicyError::CasbinError)
    }

    /// Get domain-scoped role membership records (`user`, `role`, `domain`).
    pub async fn get_domain_grouping_policy(&self) -> Vec<Vec<String>> {
        let enforcer = self.enforcer.read().await;
        enforcer.get_named_grouping_policy("g2")
    }

    /// Get all roles for a user
    pub async fn get_roles_for_user(&self, user: &str) -> Vec<String> {
        let enforcer = self.enforcer.read().await;
        enforcer.get_roles_for_user(user, None)
    }

    /// Get all users with a role
    pub async fn get_users_for_role(&self, role: &str) -> Vec<String> {
        let enforcer = self.enforcer.read().await;
        enforcer.get_users_for_role(role, None)
    }

    /// Get all policy rules
    pub async fn get_policy(&self) -> Vec<Vec<String>> {
        let enforcer = self.enforcer.read().await;
        enforcer.get_policy()
    }

    /// Get all grouping rules (role assignments)
    pub async fn get_grouping_policy(&self) -> Vec<Vec<String>> {
        let enforcer = self.enforcer.read().await;
        enforcer.get_grouping_policy()
    }

    /// Save policies to disk
    pub async fn save(&self) -> Result<(), PolicyError> {
        let mut enforcer = self.enforcer.write().await;
        enforcer
            .save_policy()
            .await
            .map_err(|e| PolicyError::PolicySaveError(e.to_string()))?;
        Ok(())
    }

    /// Reload policies from disk and re-inject base rules.
    ///
    /// `load_policy()` wipes in-memory additions, so base rules must be
    /// re-injected after every reload to guarantee they're always present.
    pub async fn reload(&self) -> Result<(), PolicyError> {
        let mut enforcer = self.enforcer.write().await;
        enforcer
            .load_policy()
            .await
            .map_err(|e| PolicyError::PolicyLoadError(e.to_string()))?;
        // Filter out rules already loaded from CSV (Casbin add_policies
        // aborts the entire batch on first duplicate).
        let base = base_policies_to_vec();
        let existing = enforcer.get_policy();
        let new_rules: Vec<Vec<String>> = base
            .into_iter()
            .filter(|rule| !existing.contains(rule))
            .collect();
        if !new_rules.is_empty() {
            enforcer
                .add_policies(new_rules)
                .await
                .map_err(PolicyError::CasbinError)?;
        }
        Ok(())
    }

    /// Apply a template's policies and groupings to the enforcer, then persist to disk.
    ///
    /// Uses the Casbin enforcer's `add_policies()` + `add_grouping_policies()` APIs,
    /// then calls `save_policy()` to write everything (including base rules) to disk.
    pub async fn apply_template(
        &self,
        template: &crate::auth::policy_templates::PolicyTemplate,
    ) -> Result<(), PolicyError> {
        let mut enforcer = self.enforcer.write().await;

        // Add policy rules
        let policies = template.expanded_policies();
        let policy_vecs: Vec<Vec<String>> = policies.iter().map(ServicePolicyRule::to_vec).collect();
        if !policy_vecs.is_empty() {
            enforcer
                .add_policies(policy_vecs)
                .await
                .map_err(PolicyError::CasbinError)?;
        }

        // Add grouping rules
        if let Some(groupings) = template.groupings {
            let grouping_vecs: Vec<Vec<String>> = groupings.iter().map(ServiceGrouping::to_vec).collect();
            if !grouping_vecs.is_empty() {
                enforcer
                    .add_grouping_policies(grouping_vecs)
                    .await
                    .map_err(PolicyError::CasbinError)?;
            }
        }

        // Persist full enforcer state (base + template) to disk via FileAdapter
        enforcer
            .save_policy()
            .await
            .map_err(|e| PolicyError::PolicySaveError(e.to_string()))?;

        Ok(())
    }

    /// Get the policies directory path
    pub fn policies_dir(&self) -> &Path {
        &self.policies_dir
    }

    /// Get the model.conf path
    pub fn model_conf_path(&self) -> PathBuf {
        self.policies_dir.join("model.conf")
    }

    /// Get the policy.csv path
    pub fn policy_csv_path(&self) -> PathBuf {
        self.policies_dir.join("policy.csv")
    }

    /// Format policies as a human-readable string for display
    pub async fn format_policy(&self) -> String {
        let policies = self.get_policy().await;
        let groupings = self.get_grouping_policy().await;
        let domain_groupings = self.get_domain_grouping_policy().await;

        let mut output = String::new();

        if !policies.is_empty() {
            output.push_str("# Policy Rules\n");
            for p in &policies {
                output.push_str(&format!("p, {}\n", p.join(", ")));
            }
        }

        if !groupings.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str("# Role Assignments\n");
            for g in &groupings {
                output.push_str(&format!("g, {}\n", g.join(", ")));
            }
        }

        if !domain_groupings.is_empty() {
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str("# Domain Role Assignments\n");
            for grouping in &domain_groupings {
                output.push_str(&format!("g2, {}\n", grouping.join(", ")));
            }
        }

        if output.is_empty() {
            output.push_str("# No policies defined\n");
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_permissive_policy_manager() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Permissive policy allows everything
        assert!(pm.check("anyone", "anything", Operation::Infer).await);
        assert!(pm.check("user", "model:test", Operation::Train).await);
        Ok(())
    }

    #[tokio::test]
    async fn test_policy_manager_from_dir() -> Result<(), PolicyError> {
        let temp = TempDir::new().map_err(PolicyError::IoError)?;
        let policies_dir = temp.path().join("policies");

        // Create policy manager (should create default files)
        let pm = PolicyManager::new(&policies_dir).await?;

        // Default policy is deny-all — no access until explicitly configured
        assert!(!pm.check("user", "model:test", Operation::Infer).await);
        assert!(!pm.check("local:alice", "model:test", Operation::Infer).await);

        // Verify files were created
        assert!(policies_dir.join("model.conf").exists());
        assert!(policies_dir.join("policy.csv").exists());

        // After adding explicit policy, access should work
        pm.add_policy("user", "model:test", "infer.generate").await?;
        assert!(pm.check("user", "model:test", Operation::Infer).await);
        Ok(())
    }

    #[tokio::test]
    async fn test_add_and_check_policy() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Remove the default permissive rule
        pm.remove_policy("*", "*", "*").await?;

        // Now access should be denied
        assert!(!pm.check("user", "model:test", Operation::Infer).await);

        // Add specific policy
        pm.add_policy("user", "model:test", "infer.generate").await?;

        // Now this specific access should work
        assert!(pm.check("user", "model:test", Operation::Infer).await);
        // But not other operations
        assert!(!pm.check("user", "model:test", Operation::Train).await);
        Ok(())
    }

    #[tokio::test]
    async fn test_role_based_access() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Remove permissive rule
        pm.remove_policy("*", "*", "*").await?;

        // Add role-based policies
        pm.add_policy("trainer", "model:*", "infer.generate").await?;
        pm.add_policy("trainer", "model:*", "ttt.train").await?;

        // Assign role to user
        pm.add_role_for_user("alice", "trainer").await?;

        // Alice should have trainer permissions
        assert!(pm.check("alice", "model:test", Operation::Infer).await);
        assert!(pm.check("alice", "model:test", Operation::Train).await);

        // Bob without role should not
        assert!(!pm.check("bob", "model:test", Operation::Infer).await);
        Ok(())
    }

    #[tokio::test]
    async fn cross_tenant_global_role_mutation_is_denied() -> Result<(), PolicyError> {
        let pm = PolicyManager::new_in_memory().await?;
        for tenant in ["tenant-a", "tenant-b"] {
            pm.add_policy_with_domain(
                "trainer",
                tenant,
                "model:*",
                "infer.generate",
                "allow",
            )
            .await?;
        }

        pm.add_role_for_user_in_domain("alice", "trainer", "tenant-a")
            .await?;

        assert!(
            pm.check_with_domain("alice", "tenant-a", "model:test", "infer.generate")
                .await
        );
        assert!(
            !pm.check_with_domain("alice", "tenant-b", "model:test", "infer.generate")
                .await,
            "tenant-a role membership must not authorize tenant-b"
        );
        assert!(
            pm.get_grouping_policy().await.is_empty(),
            "mutable assignment must not create a global g(user, role) entry"
        );
        assert_eq!(
            pm.get_domain_grouping_policy().await,
            vec![vec![
                "alice".to_owned(),
                "trainer".to_owned(),
                "tenant-a".to_owned()
            ]]
        );

        assert!(
            !pm.remove_role_for_user_in_domain("alice", "trainer", "tenant-b")
                .await?,
            "tenant-b removal must not delete tenant-a membership"
        );
        assert!(
            pm.check_with_domain("alice", "tenant-a", "model:test", "infer.generate")
                .await
        );
        Ok(())
    }

    #[tokio::test]
    async fn multiple_identities_share_only_their_tenant_grant() -> Result<(), PolicyError> {
        let pm = PolicyManager::new_in_memory().await?;
        pm.add_policy_with_domain("*", "tenant-a", "event:orders", "subscribe", "allow")
            .await?;

        for identity in ["alice", "bob"] {
            assert!(
                pm.check_with_domain(identity, "tenant-a", "event:orders", "subscribe")
                    .await,
                "{identity} should receive the shared tenant-a grant"
            );
            assert!(
                !pm.check_with_domain(identity, "tenant-b", "event:orders", "subscribe")
                    .await,
                "{identity} must not receive the tenant-a grant in tenant-b"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_format_policy() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        pm.add_policy("operator", "*", "*").await?;
        pm.add_role_for_user("alice", "operator").await?;

        let formatted = pm.format_policy().await;
        assert!(formatted.contains("Policy Rules"));
        assert!(formatted.contains("operator"));
        assert!(formatted.contains("Role Assignments"));
        assert!(formatted.contains("alice"));
        Ok(())
    }

    #[tokio::test]
    async fn test_input_validation_rejects_null_bytes() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Null bytes should be rejected
        let result = pm.add_policy("user\0evil", "model:test", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));

        let result = pm.add_policy("user", "model:test\0evil", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));
        Ok(())
    }

    #[tokio::test]
    async fn test_input_validation_rejects_line_breaks() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Line breaks should be rejected (CSV injection)
        let result = pm.add_policy("user\np, evil, *, *", "model:test", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));

        let result = pm.add_role_for_user("user\revil", "operator").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));
        Ok(())
    }

    #[tokio::test]
    async fn test_input_validation_rejects_long_strings() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Strings over 256 chars should be rejected
        let long_string = "a".repeat(257);
        let result = pm.add_policy(&long_string, "model:test", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));
        Ok(())
    }

    #[tokio::test]
    async fn test_input_validation_accepts_valid_input() -> Result<(), PolicyError> {
        let pm = PolicyManager::permissive().await?;

        // Valid inputs should work
        assert!(pm.add_policy("user-123", "model:qwen3-*", "infer").await.is_ok());
        assert!(pm.add_role_for_user("alice_test", "trainer").await.is_ok());
        Ok(())
    }

    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_act_wildcard_keymatch() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        pm.add_policy_with_domain("alice", "*", "model:qwen3-small:main", "ttt.*", "allow").await.unwrap();
        // Specific action should match the wildcard
        assert!(pm.check_with_domain("alice", "*", "model:qwen3-small:main", "ttt.writeback").await);
        assert!(pm.check_with_domain("alice", "*", "model:qwen3-small:main", "ttt.evict").await);
        // Different namespace should NOT match
        assert!(!pm.check_with_domain("alice", "*", "model:qwen3-small:main", "infer.generate").await);
    }

    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_set_branch_visibility_adds_removes_rules() {
        let pm = PolicyManager::new_in_memory().await.unwrap();

        // Grant alice manage (ttt.writeback) on the model resource
        pm.add_policy_with_domain("alice", "*", "model:qwen3:main", "ttt.writeback", "allow").await.unwrap();

        // Initially bob cannot infer
        assert!(!pm.check_with_domain("bob", "*", "model:qwen3:main", "infer.generate").await);

        // Make public: add wildcard infer + query rules
        pm.add_policy_with_domain("*", "*", "model:qwen3:main", "infer.generate", "allow").await.unwrap();
        pm.add_policy_with_domain("*", "*", "model:qwen3:main", "query.status", "allow").await.unwrap();

        // After making public, any user can infer and query
        assert!(pm.check_with_domain("bob", "*", "model:qwen3:main", "infer.generate").await);
        assert!(pm.check_with_domain("carol", "*", "model:qwen3:main", "query.status").await);

        // Make private: remove wildcard rules
        pm.remove_policy_with_domain("*", "*", "model:qwen3:main", "infer.generate", "allow").await.unwrap();
        pm.remove_policy_with_domain("*", "*", "model:qwen3:main", "query.status", "allow").await.unwrap();

        // After making private, unauthenticated users are denied again
        assert!(!pm.check_with_domain("bob", "*", "model:qwen3:main", "infer.generate").await);
        assert!(!pm.check_with_domain("carol", "*", "model:qwen3:main", "query.status").await);

        // Alice still has manage access (her rule was not removed)
        assert!(pm.check_with_domain("alice", "*", "model:qwen3:main", "ttt.writeback").await);
    }

    /// Verify that SERVICE_BASE_POLICIES are injected on init and allow
    /// service-to-service authorization (the exact scenario that broke OAuth startup).
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_base_rules_injected_on_init() {
        let temp = TempDir::new().map_err(PolicyError::IoError).unwrap();
        let policies_dir = temp.path().join("policies");

        let pm = PolicyManager::new(&policies_dir).await.unwrap();

        // The OAuth startup failure: service:oauth cannot query on policy.
        // The dispatch codegen now passes resource="policy:ResolveServiceKey"
        // for struct-type RPCs. Base rules use "policy:*" which keyMatch matches.
        assert!(
            pm.check_with_domain("service:oauth", "*", "policy:ResolveServiceKey", "query").await,
            "service:oauth should be able to query policy:ResolveServiceKey via base rules"
        );

        // Other service grants from SERVICE_BASE_POLICIES
        assert!(pm.check_with_domain("service:policy", "*", "policy:IssueToken", "manage").await);
        // The OAuth token issuance path: service:oauth must be able to issue tokens via
        // the wildcard rule (service:*,*,policy:IssueToken,manage,allow). Action must be
        // "manage" (not Operation::Manage.as_str() which returns "ttt.writeback").
        assert!(pm.check_with_domain("service:oauth", "*", "policy:IssueToken", "manage").await,
            "service:oauth must be allowed to issue tokens (service:* wildcard rule)");
        assert!(pm.check_with_domain("service:model", "*", "policy:Check", "check").await);
        assert!(pm.check_with_domain("service:model", "*", "registry:ListModels", "infer").await);
        assert!(pm.check_with_domain("service:oai", "*", "model:Generate", "infer.generate").await);

        // Non-granted service should be denied
        assert!(!pm.check_with_domain("service:unknown", "*", "policy:ResolveServiceKey", "query").await);
    }

    /// Reproduce the exact server failure: load from the production policy.csv
    /// and verify service:oauth can query policy:ResolveServiceKey.
    #[tokio::test]
    #[allow(clippy::unwrap_used, clippy::print_stderr)]
    async fn test_production_policy_csv_oauth_query() {
        let prod_path = std::path::Path::new("/home/birdetta/.local/share/hyprstream/models/.registry/policies");
        if !prod_path.exists() {
            eprintln!("Skipping: production policy dir not found");
            return;
        }

        // Copy production files to temp dir to avoid mutating real data
        let temp = TempDir::new().map_err(PolicyError::IoError).unwrap();
        let temp_policies = temp.path().join("policies");
        tokio::fs::create_dir_all(&temp_policies).await.unwrap();
        tokio::fs::copy(prod_path.join("model.conf"), temp_policies.join("model.conf")).await.unwrap();
        tokio::fs::copy(prod_path.join("policy.csv"), temp_policies.join("policy.csv")).await.unwrap();

        let pm = PolicyManager::new(&temp_policies).await.unwrap();

        let all_policies = pm.get_policy().await;
        eprintln!("Loaded {} policies from production CSV:", all_policies.len());
        for p in &all_policies {
            eprintln!("  p, {}", p.join(", "));
        }

        // The exact failing check from the server
        let allowed = pm.check_with_domain("service:oauth", "*", "policy:ResolveServiceKey", "query").await;
        eprintln!("service:oauth query policy:ResolveServiceKey = {}", allowed);

        // List all rules that match oauth
        for p in &all_policies {
            if p[0].contains("oauth") || p[0] == "service:*" {
                eprintln!("  oauth-relevant: p, {}", p.join(", "));
            }
        }

        assert!(allowed, "service:oauth should be able to query policy:ResolveServiceKey");
    }

    // ========================================================================
    // atproto-scoped authorization invariants
    //
    // These assert *behavioral* contracts that any authorization model must
    // hold (deny-by-default, per-identity isolation, public/private visibility,
    // stream isolation) — NOT the current Casbin mechanism/vocabulary. They are
    // written to survive the future tiles.run / atproto authz alignment: the
    // new model has to satisfy the same invariants, so these double as a
    // conformance harness. Resource identifiers go through the helpers below so
    // that when model resources move toward AT-URI form the naming changes
    // in one place, not every assertion. (OAuth scope-vocabulary tests are
    // intentionally deferred — hyprstream's `action:service:*` scopes are not
    // the access gate and will be replaced by atproto scopes.)
    // ========================================================================

    /// Resource string for a model branch. Centralized for atproto forward-compat.
    fn model_resource(name: &str, branch: &str) -> String {
        format!("model:{name}:{branch}")
    }
    /// Resource pattern for an identity's own namespace (self-ownership).
    fn user_ns(subject: &str) -> String {
        format!("user:{subject}:*")
    }
    /// Resource string for a moq event-prefix subscribe scope.
    fn subscribe_scope(prefix: &str) -> String {
        format!("subscribe:events:{prefix}.*")
    }
    /// Resource string for an inference-mesh job (#319). The tenant is the first
    /// path segment so the caveat survives translation to a UCAN delegation.
    fn mesh_job(tenant: &str, job: &str) -> String {
        format!("mesh://{tenant}/job/{job}")
    }
    /// Per-host mesh authorization subject (#328).
    fn mesh_host(id: &str) -> String {
        format!("service:inference:host-{id}")
    }

    /// Regression guard for the wildcard-validation bug: `add_policy` injects
    /// domain="*", so a blanket '*' ban made it always fail (even on
    /// wildcard-free input). It must succeed and take effect.
    #[tokio::test]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    async fn test_add_policy_succeeds_with_default_domain() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        // Wildcard-free 3-arg add_policy (internally domain="*") must succeed.
        pm.add_policy("alice", &model_resource("qwen3", "main"), "infer.generate")
            .await
            .expect("add_policy must succeed (domain=* is the legitimate global domain)");
        assert!(
            pm.check("alice", &model_resource("qwen3", "main"), Operation::Infer).await,
            "the rule add_policy created must take effect"
        );
    }

    /// P0a — self-namespace isolation: an identity may reach its own namespace,
    /// and MUST NOT reach another identity's namespace (cross-identity denial).
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_self_namespace_isolation() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        // The self-ownership grant the OIDC callback issues on first login.
        pm.add_policy_with_domain("alice", "*", &user_ns("alice"), "*", "allow")
            .await
            .unwrap();
        pm.add_policy_with_domain("bob", "*", &user_ns("bob"), "*", "allow")
            .await
            .unwrap();

        // Each identity reaches its own namespace.
        assert!(pm.check_with_domain("alice", "*", "user:alice:model1", "ttt.writeback").await);
        assert!(pm.check_with_domain("bob", "*", "user:bob:model1", "ttt.writeback").await);

        // Neither identity may reach the other's namespace.
        assert!(!pm.check_with_domain("alice", "*", "user:bob:model1", "infer.generate").await,
            "alice must not access bob's namespace");
        assert!(!pm.check_with_domain("bob", "*", "user:alice:model1", "ttt.writeback").await,
            "bob must not access alice's namespace");
    }

    /// P1a — deny-by-default and least-privilege: an unknown subject is denied,
    /// a scoped grant authorizes only that exact (resource, action) pair, and
    /// does not leak to other resources or actions.
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_deny_by_default_and_least_privilege() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        let resource = model_resource("qwen3", "main");

        // Deny-by-default: nobody is authorized until granted.
        assert!(!pm.check_with_domain("mallory", "*", &resource, "infer.generate").await);

        // Grant alice exactly infer on this model.
        pm.add_policy_with_domain("alice", "*", &resource, "infer.generate", "allow")
            .await
            .unwrap();

        assert!(pm.check_with_domain("alice", "*", &resource, "infer.generate").await);
        // Grant does not leak to other actions...
        assert!(!pm.check_with_domain("alice", "*", &resource, "ttt.writeback").await,
            "infer grant must not confer manage");
        // ...nor to other resources...
        assert!(!pm.check_with_domain("alice", "*", &model_resource("llama", "main"), "infer.generate").await,
            "grant on one model must not confer another");
        // ...nor to other subjects.
        assert!(!pm.check_with_domain("mallory", "*", &resource, "infer.generate").await);
    }

    /// P2a — moq subscribe-prefix isolation: a subscribe grant on one event
    /// prefix MUST NOT authorize a sibling/private prefix (no cross-tenant leak
    /// via prefix matching).
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_subscribe_prefix_isolation() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        // alice may subscribe under the "public" prefix only.
        pm.add_policy_with_domain("alice", "*", &subscribe_scope("public"), "subscribe", "allow")
            .await
            .unwrap();

        assert!(pm.check_with_domain("alice", "*", "subscribe:events:public.metrics", "subscribe").await,
            "alice may subscribe within her granted prefix");
        // A private prefix is NOT covered by the public-prefix grant.
        assert!(!pm.check_with_domain("alice", "*", "subscribe:events:private.secrets", "subscribe").await,
            "public-prefix grant must not leak to a private prefix");
        // An ungranted subject is denied entirely.
        assert!(!pm.check_with_domain("bob", "*", "subscribe:events:public.metrics", "subscribe").await);
    }

    // ========================================================================
    // Inference-mesh authorization invariants (#319)
    //
    // The mesh authority actions (`infer.stage`, `delta.submit`, `mesh.rpc`)
    // must be deny-by-default, granted ONLY to non-wildcard
    // `service:inference:host-<id>` subjects, scoped to a single tenant. The
    // read ability (`query.status`) may be granted to a host group. These
    // mirror the atproto-scoped invariant style above and double as the #319
    // security harness.
    // ========================================================================

    /// Apply the worked-example `mesh-host` template rules to an enforcer.
    /// Mirrors what `policy apply-template mesh-host` does at runtime.
    #[allow(clippy::unwrap_used)]
    async fn apply_template(pm: &PolicyManager, name: &str) {
        let tmpl = crate::auth::policy_templates::get_template(name)
            .unwrap_or_else(|| panic!("template {name} must exist"));
        for r in tmpl.expanded_policies() {
            pm.add_policy_with_domain(r.subject, r.domain, r.resource, r.action, r.effect)
                .await
                .unwrap();
        }
        if let Some(groupings) = tmpl.groupings {
            for g in groupings {
                pm.add_role_for_user(g.user, g.role).await.unwrap();
            }
        }
    }

    /// #319 invariant — the mesh authority actions are NEVER reachable by a
    /// wildcard (`*`) or `anonymous` subject, even when a `federation-open`-style
    /// wildcard rule AND the public-inference template are both loaded. Only a
    /// specific enrolled host subject may stage/submit.
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_mesh_authority_never_matches_wildcard_or_anonymous() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        // Load a broad wildcard rule (federation-open) and the public-inference
        // template — neither must leak into mesh authority.
        apply_template(&pm, "federation-open").await;
        apply_template(&pm, "public-inference").await;
        // And the real per-host mesh grants for tenant acme.
        apply_template(&pm, "mesh-host").await;

        let job = mesh_job("acme", "j7");
        for action in ["infer.stage", "delta.submit", "mesh.rpc"] {
            assert!(
                !pm.check_with_domain("*", "acme", &job, action).await,
                "wildcard subject must NEVER match mesh authority action {action}"
            );
            assert!(
                !pm.check_with_domain("anonymous", "acme", &job, action).await,
                "anonymous must NEVER match mesh authority action {action}"
            );
            // A non-enrolled host subject is likewise denied.
            assert!(
                !pm.check_with_domain(&mesh_host("99"), "acme", &job, action).await,
                "un-enrolled host must be denied mesh authority action {action}"
            );
        }
        // Positive control: an enrolled host CAN stage within its tenant.
        assert!(
            pm.check_with_domain(&mesh_host("1"), "acme", &job, "infer.stage").await,
            "enrolled host-1 must be able to infer.stage in its tenant"
        );
    }

    /// #319 invariant — tenant isolation: a host enrolled for tenant `acme`
    /// cannot exercise ANY mesh action against tenant `beta`'s resources, even
    /// for the same job name. The Casbin domain exact-match is the gate.
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_mesh_tenant_isolation() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        // host-1 is granted on tenant acme via the worked-example template.
        apply_template(&pm, "mesh-host").await;

        // In its own tenant: allowed.
        assert!(
            pm.check_with_domain(&mesh_host("1"), "acme", &mesh_job("acme", "j7"), "infer.stage").await
        );
        // Cross-tenant (domain beta): denied for every action, even though the
        // host holds the same grant in acme.
        for action in ["infer.stage", "delta.submit", "mesh.rpc", "query.status"] {
            assert!(
                !pm.check_with_domain(&mesh_host("1"), "beta", &mesh_job("beta", "j7"), action).await,
                "acme-scoped host-1 must NOT reach tenant beta for {action}"
            );
        }
        // Even pointing the resource path at beta while claiming the acme domain
        // must not match the acme rule (resource prefix differs).
        assert!(
            !pm.check_with_domain(&mesh_host("1"), "acme", &mesh_job("beta", "j7"), "infer.stage").await,
            "acme-domain rule must not match a beta resource path"
        );
    }

    /// #319 invariant — group read, per-host authority: a host group may be
    /// granted the read ability (`query.status`) via Casbin grouping, but the
    /// group MUST NOT confer any authority action.
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_mesh_group_read_not_authority() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        apply_template(&pm, "mesh-host-group").await;

        let job = mesh_job("acme", "j7");
        // host-1 inherits the group read grant.
        assert!(
            pm.check_with_domain(&mesh_host("1"), "acme", &job, "query.status").await,
            "grouped host must inherit the read ability"
        );
        // But the group does NOT confer authority.
        for action in ["infer.stage", "delta.submit", "mesh.rpc"] {
            assert!(
                !pm.check_with_domain(&mesh_host("1"), "acme", &job, action).await,
                "read group must NOT confer authority action {action}"
            );
        }
        // A host NOT in the group gets nothing.
        assert!(
            !pm.check_with_domain(&mesh_host("3"), "acme", &job, "query.status").await,
            "un-grouped host must not inherit the read ability"
        );
    }

    /// #319 invariant — deny-by-default: with no mesh template applied, an
    /// enrolled-looking host subject is denied every mesh action.
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn test_mesh_deny_by_default() {
        let pm = PolicyManager::new_in_memory().await.unwrap();
        let job = mesh_job("acme", "j7");
        for action in ["infer.stage", "delta.submit", "mesh.rpc", "query.status"] {
            assert!(
                !pm.check_with_domain(&mesh_host("1"), "acme", &job, action).await,
                "un-granted host must be denied {action} (deny-by-default)"
            );
        }
    }
}
