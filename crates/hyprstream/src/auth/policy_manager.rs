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
//! p, admin, *, *
//! p, trainer, model:*, infer
//! p, trainer, model:*, train
//! g, alice, trainer
//! ```

use crate::auth::Operation;
use casbin::{CoreApi, DefaultModel, Enforcer, FileAdapter, MemoryAdapter, MgmtApi, RbacApi};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

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
            "{} exceeds maximum length of {} characters",
            name, MAX_POLICY_COMPONENT_LEN
        )));
    }

    if component.contains('\0') {
        return Err(PolicyError::ValidationError(format!(
            "{} contains null byte",
            name
        )));
    }

    if component.contains('\n') || component.contains('\r') {
        return Err(PolicyError::ValidationError(format!(
            "{} contains line break (potential CSV injection)",
            name
        )));
    }

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

[policy_effect]
e = some(where (p.eft == allow)) && !some(where (p.eft == deny))

[matchers]
m = (g(r.sub, p.sub) || keyMatch(r.sub, p.sub)) && \
    (p.dom == "*" || r.dom == p.dom) && \
    keyMatch(r.obj, p.obj) && \
    (p.act == "*" || r.act == p.act)
"#;

/// Default policy rules (deny-all by default for security)
/// Add explicit policies for access grants
///
/// Policy format: p, subject, domain, resource, action, effect
const DEFAULT_POLICY_CSV: &str = r#"# Default policy (secure default with local user access)
# Add explicit policies below to grant additional access.
#
# Policy format: p, subject, domain, resource, action, effect
#
# Default policies for local users (CLI operations):
p, local:*, *, registry, query, allow
p, local:*, *, registry:*, query, allow
p, local:*, *, registry:*, write, allow
p, local:*, *, model:*, infer, allow
#
# Examples for additional policies:
#   p, admin, *, *, *, allow             # Admin has full access to all domains
#   p, trainer, HfModel, model:*, train, allow  # Trainer can train HfModel domain
#   p, user, HfModel, model:*, infer, allow     # User can infer in HfModel domain
#   p, analyst, HfDataset, data:*, query, allow # Analyst can query datasets
#   p, *, *, *, manage, deny             # Nobody can manage (explicit deny)
#
# Role assignments:
#   g, alice, trainer        # Assign alice to trainer role
#
# IMPORTANT: Add policies above to grant access to specific users/roles.
"#;

/// Validate policy.csv format before loading
///
/// Ensures all policy lines have the correct number of fields:
/// - Policy lines (p, ...): 6 fields (p, subject, domain, resource, action, effect)
/// - Role lines (g, ...): 3 fields (g, user, role)
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
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() != 6 {  // "p" + 5 fields
                return Err(PolicyError::ValidationError(format!(
                    "Invalid policy format at line {}:\n\
                    Found:    {}\n\
                    Expected: p, subject, domain, resource, action, effect\n\
                    Example:  p, admin, *, *, *, allow\n\n\
                    Your policy has {} fields but 6 are required.\n\
                    Edit {} to fix the format.",
                    line_num + 1, line, fields.len(), policy_path.display()
                )));
            }
        }

        // Check role lines (g, ...)
        if line.starts_with("g,") || line.starts_with("g ") {
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if fields.len() != 3 {  // "g" + 2 fields
                return Err(PolicyError::ValidationError(format!(
                    "Invalid role format at line {}:\n\
                    Found:    {}\n\
                    Expected: g, user, role\n\
                    Example:  g, alice, admin",
                    line_num + 1, line
                )));
            }
        }
    }
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

        // Ensure policies directory exists
        if !policies_dir.exists() {
            info!("Creating policies directory at {:?}", policies_dir);
            tokio::fs::create_dir_all(&policies_dir).await?;
        }

        let model_path = policies_dir.join("model.conf");
        let policy_path = policies_dir.join("policy.csv");

        // Create default model.conf if not exists
        if !model_path.exists() {
            info!("Creating default model.conf");
            tokio::fs::write(&model_path, DEFAULT_MODEL_CONF).await?;
        }

        // Create default policy.csv if not exists
        if !policy_path.exists() {
            info!("Creating default policy.csv (deny-all - add policies to grant access)");
            tokio::fs::write(&policy_path, DEFAULT_POLICY_CSV).await?;
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
        let enforcer = Enforcer::new(model, adapter)
            .await
            .map_err(|e| PolicyError::PolicyLoadError(e.to_string()))?;

        info!(
            "PolicyManager initialized with {} policies",
            enforcer.get_policy().len()
        );

        Ok(Self {
            enforcer: Arc::new(RwLock::new(enforcer)),
            policies_dir,
        })
    }

    /// Create a PolicyManager with default permissive policies (no file storage)
    ///
    /// Useful for testing or when policies aren't configured.
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
                "*".to_string(),      // subject
                "*".to_string(),      // domain
                "*".to_string(),      // resource
                "*".to_string(),      // action
                "allow".to_string(),  // effect
            ])
            .await
            .map_err(PolicyError::CasbinError)?;

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
        self.check_with_domain(user, "*", resource, operation).await
    }

    /// Check if a user is allowed to perform an operation on a resource in a specific domain
    ///
    /// # Arguments
    /// * `user` - The user or role requesting access
    /// * `domain` - The domain (e.g., "HfModel", "HfDataset")
    /// * `resource` - The resource being accessed (e.g., "model:qwen3-small")
    /// * `operation` - The operation being requested
    ///
    /// # Returns
    /// `true` if access is allowed, `false` otherwise
    pub async fn check_with_domain(
        &self,
        user: &str,
        domain: &str,
        resource: &str,
        operation: Operation,
    ) -> bool {
        let enforcer = self.enforcer.read().await;
        match enforcer.enforce((user, domain, resource, operation.as_str())) {
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
                "effect must be 'allow' or 'deny'".to_string(),
            ));
        }

        let mut enforcer = self.enforcer.write().await;
        enforcer
            .add_policy(vec![
                user.to_string(),
                domain.to_string(),
                resource.to_string(),
                operation.to_string(),
                effect.to_string(),
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
                user.to_string(),
                domain.to_string(),
                resource.to_string(),
                operation.to_string(),
                effect.to_string(),
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
            .add_grouping_policy(vec![user.to_string(), role.to_string()])
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
            .remove_grouping_policy(vec![user.to_string(), role.to_string()])
            .await
            .map_err(PolicyError::CasbinError)
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

    /// Reload policies from disk
    pub async fn reload(&self) -> Result<(), PolicyError> {
        let mut enforcer = self.enforcer.write().await;
        enforcer
            .load_policy()
            .await
            .map_err(|e| PolicyError::PolicyLoadError(e.to_string()))?;
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

    // ========================================================================
    // Convenience Methods for Common Policy Configurations
    // ========================================================================

    /// Enable full access for local users (local:*)
    ///
    /// This is the default secure configuration that allows local users
    /// to access all resources while still requiring authentication.
    pub async fn enable_local_full_access(&self) -> Result<(), PolicyError> {
        let policies = [
            // Registry access
            ("local:*", "*", "registry", "query", "allow"),
            ("local:*", "*", "registry", "write", "allow"),
            ("local:*", "*", "registry:*", "query", "allow"),
            ("local:*", "*", "registry:*", "write", "allow"),
            ("local:*", "*", "registry:*", "manage", "allow"),
            // Inference access
            ("local:*", "*", "inference", "manage", "allow"),
            ("local:*", "*", "inference:*", "infer", "allow"),
            ("local:*", "*", "inference:*", "query", "allow"),
            ("local:*", "*", "inference:*", "write", "allow"),
        ];

        for (sub, dom, obj, act, eft) in policies {
            self.add_policy_with_domain(sub, dom, obj, act, eft).await?;
        }

        info!("Enabled full access for local users");
        Ok(())
    }

    /// Enable public inference access for anonymous users
    ///
    /// Use this when exposing an OpenAI-compatible API publicly.
    /// Only allows inference and query operations, not write or manage.
    pub async fn enable_public_inference(&self) -> Result<(), PolicyError> {
        let policies = [
            ("anonymous", "*", "inference:*", "infer", "allow"),
            ("anonymous", "*", "inference:*", "query", "allow"),
        ];

        for (sub, dom, obj, act, eft) in policies {
            self.add_policy_with_domain(sub, dom, obj, act, eft).await?;
        }

        info!("Enabled public inference access for anonymous users");
        Ok(())
    }

    /// Enable public read-only registry access for anonymous users
    ///
    /// Use this when you want to allow anonymous users to browse
    /// available models without authentication.
    pub async fn enable_public_read(&self) -> Result<(), PolicyError> {
        let policies = [
            ("anonymous", "*", "registry", "query", "allow"),
            ("anonymous", "*", "registry:*", "query", "allow"),
        ];

        for (sub, dom, obj, act, eft) in policies {
            self.add_policy_with_domain(sub, dom, obj, act, eft).await?;
        }

        info!("Enabled public read access for anonymous users");
        Ok(())
    }

    /// Format policies as a human-readable string for display
    pub async fn format_policy(&self) -> String {
        let policies = self.get_policy().await;
        let groupings = self.get_grouping_policy().await;

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
    async fn test_permissive_policy_manager() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Permissive policy allows everything
        assert!(pm.check("anyone", "anything", Operation::Infer).await);
        assert!(pm.check("user", "model:test", Operation::Train).await);
    }

    #[tokio::test]
    async fn test_policy_manager_from_dir() {
        let temp = TempDir::new().expect("test: create temp dir");
        let policies_dir = temp.path().join("policies");

        // Create policy manager (should create default files)
        let pm = PolicyManager::new(&policies_dir).await.expect("test: create policy manager");

        // Default policy is deny-all (secure default)
        assert!(!pm.check("user", "model:test", Operation::Infer).await);

        // Verify files were created
        assert!(policies_dir.join("model.conf").exists());
        assert!(policies_dir.join("policy.csv").exists());

        // After adding explicit policy, access should work
        pm.add_policy("user", "model:test", "infer").await.expect("test: add policy");
        assert!(pm.check("user", "model:test", Operation::Infer).await);
    }

    #[tokio::test]
    async fn test_add_and_check_policy() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Remove the default permissive rule
        pm.remove_policy("*", "*", "*").await.expect("test: remove policy");

        // Now access should be denied
        assert!(!pm.check("user", "model:test", Operation::Infer).await);

        // Add specific policy
        pm.add_policy("user", "model:test", "infer").await.expect("test: add policy");

        // Now this specific access should work
        assert!(pm.check("user", "model:test", Operation::Infer).await);
        // But not other operations
        assert!(!pm.check("user", "model:test", Operation::Train).await);
    }

    #[tokio::test]
    async fn test_role_based_access() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Remove permissive rule
        pm.remove_policy("*", "*", "*").await.expect("test: remove policy");

        // Add role-based policies
        pm.add_policy("trainer", "model:*", "infer").await.expect("test: add policy");
        pm.add_policy("trainer", "model:*", "train").await.expect("test: add policy");

        // Assign role to user
        pm.add_role_for_user("alice", "trainer").await.expect("test: add role");

        // Alice should have trainer permissions
        assert!(pm.check("alice", "model:test", Operation::Infer).await);
        assert!(pm.check("alice", "model:test", Operation::Train).await);

        // Bob without role should not
        assert!(!pm.check("bob", "model:test", Operation::Infer).await);
    }

    #[tokio::test]
    async fn test_format_policy() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        pm.add_policy("admin", "*", "*").await.expect("test: add policy");
        pm.add_role_for_user("alice", "admin").await.expect("test: add role");

        let formatted = pm.format_policy().await;
        assert!(formatted.contains("Policy Rules"));
        assert!(formatted.contains("admin"));
        assert!(formatted.contains("Role Assignments"));
        assert!(formatted.contains("alice"));
    }

    #[tokio::test]
    async fn test_input_validation_rejects_null_bytes() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Null bytes should be rejected
        let result = pm.add_policy("user\0evil", "model:test", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));

        let result = pm.add_policy("user", "model:test\0evil", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));
    }

    #[tokio::test]
    async fn test_input_validation_rejects_line_breaks() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Line breaks should be rejected (CSV injection)
        let result = pm.add_policy("user\np, evil, *, *", "model:test", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));

        let result = pm.add_role_for_user("user\revil", "admin").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));
    }

    #[tokio::test]
    async fn test_input_validation_rejects_long_strings() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Strings over 256 chars should be rejected
        let long_string = "a".repeat(257);
        let result = pm.add_policy(&long_string, "model:test", "infer").await;
        assert!(matches!(result, Err(PolicyError::ValidationError(_))));
    }

    #[tokio::test]
    async fn test_input_validation_accepts_valid_input() {
        let pm = PolicyManager::permissive().await.expect("test: create policy manager");

        // Valid inputs should work
        assert!(pm.add_policy("user-123", "model:qwen3-*", "infer").await.is_ok());
        assert!(pm.add_role_for_user("alice_test", "trainer").await.is_ok());
    }
}
