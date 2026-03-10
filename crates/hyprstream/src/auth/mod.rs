//! Authorization module for RBAC/ABAC access control
//!
//! Uses Casbin for policy-based access control with policies
//! stored in the `.registry/policies/` directory.
//!
//! Also provides JWT token authentication with Ed25519 signatures.

pub mod federation;
pub mod jwt;
mod policy_manager;
pub mod policy_migration;
pub mod policy_templates;
pub mod user_store;

pub use federation::FederationKeyResolver;
pub use jwt::{Claims, JwtError};
pub use policy_manager::{PolicyManager, PolicyError, write_policy_file};
pub use policy_migration::migrate_policy_csv;
pub use policy_templates::{PolicyTemplate, get_template, get_templates};
pub use user_store::{LocalKeyStore, UserStore};

/// Operation types that can be controlled via policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Run model inference (i)
    Infer,
    /// Train/fine-tune model (t)
    Train,
    /// Query data (q)
    Query,
    /// Write data (w)
    Write,
    /// Serve via API (s)
    Serve,
    /// Admin operations (m)
    Manage,
    /// Context-augmented generation (c)
    Context,
}

impl Operation {
    /// Get the short code for this operation
    pub fn code(&self) -> char {
        match self {
            Operation::Infer => 'i',
            Operation::Train => 't',
            Operation::Query => 'q',
            Operation::Write => 'w',
            Operation::Serve => 's',
            Operation::Manage => 'm',
            Operation::Context => 'c',
        }
    }

    /// Get the dot-namespaced operation name for policy matching.
    ///
    /// Returns the canonical dot-namespaced string for this operation variant.
    ///
    /// Forwarded verbatim to the Casbin enforcer, which uses `keyMatch` to accept
    /// both this format and legacy flat strings stored in policy files.
    ///
    /// **Note on `Manage`:** This enum variant is a coarse gate over the `ttt.*`
    /// namespace. `as_str()` returns `"ttt.writeback"` as a representative string,
    /// but sub-actions (`ttt.evict`, `ttt.zero`) are all gated by the same `Manage`
    /// variant. When callers need to enforce at sub-action granularity (e.g. in an
    /// RPC handler), they should pass the specific dot-namespaced string directly to
    /// `check_with_domain` rather than going through this method.
    ///
    /// `Operation::Manage.as_str()` (i.e. `"ttt.writeback"`) is intentionally used
    /// only for gates that specifically require the *highest* privilege in the `ttt.*`
    /// namespace, such as issuing tokens on behalf of another subject. Handlers that
    /// only need any `ttt.*` permission (e.g. `ttt.evict`) must pass the specific
    /// action string to avoid incorrectly denying callers who hold a lesser `ttt.*`
    /// permission but not `ttt.writeback`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Operation::Infer   => "infer.generate",
            Operation::Train   => "ttt.train",
            Operation::Query   => "query.status",
            Operation::Write   => "persist.save",
            Operation::Serve   => "serve.api",
            Operation::Manage  => "ttt.writeback",
            Operation::Context => "context.augment",
        }
    }

    /// Parse an operation from a dot-namespaced or legacy flat string.
    ///
    /// Returns `None` for unrecognized strings.  Handles both the new
    /// dot-namespaced format (`"infer.generate"`) and the old flat format
    /// (`"infer"`) for migration compatibility.
    pub fn from_dot_str(s: &str) -> Option<Self> {
        match s {
            // New dot-namespaced and legacy flat strings merged per variant
            "infer.generate" | "infer" => Some(Self::Infer),
            "ttt.train" | "train" => Some(Self::Train),
            "ttt.writeback" | "ttt.evict" | "ttt.zero" | "manage" => Some(Self::Manage),
            "query.status" | "query.delta" | "query" => Some(Self::Query),
            "persist.save" | "persist.export" | "persist.snapshot" | "write" => Some(Self::Write),
            "context.augment" | "context" => Some(Self::Context),
            "serve.api" | "serve" => Some(Self::Serve),
            _ => None,
        }
    }

    /// Parse from short code
    pub fn from_code(c: char) -> Option<Self> {
        match c {
            'i' => Some(Operation::Infer),
            't' => Some(Operation::Train),
            'q' => Some(Operation::Query),
            'w' => Some(Operation::Write),
            's' => Some(Operation::Serve),
            'm' => Some(Operation::Manage),
            'c' => Some(Operation::Context),
            _ => None,
        }
    }

    /// All operations
    pub fn all() -> &'static [Operation] {
        &[
            Operation::Infer,
            Operation::Train,
            Operation::Query,
            Operation::Write,
            Operation::Serve,
            Operation::Manage,
            Operation::Context,
        ]
    }
}

impl std::str::FromStr for Operation {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_dot_str(s)
            .ok_or_else(|| anyhow::anyhow!("Unknown operation: {}", s))
    }
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Convert capabilities to an access string based on policy
///
/// Returns a comma-separated list of permitted operations (e.g., "infer,train,serve").
/// Only includes operations that both the resource supports AND the user is permitted.
pub fn capabilities_to_access_string(
    policy_manager: &PolicyManager,
    user: &str,
    resource: &str,
    capabilities: &crate::archetypes::capabilities::CapabilitySet,
) -> String {
    use crate::archetypes::capabilities::{Context, Infer, Manage, Query, Serve, Train, Write};

    let mut permitted = Vec::new();

    // Only show access for capabilities the resource actually has AND user is permitted
    if capabilities.has::<Infer>() && policy_manager.check_sync(user, resource, Operation::Infer) {
        permitted.push("infer");
    }
    if capabilities.has::<Train>() && policy_manager.check_sync(user, resource, Operation::Train) {
        permitted.push("train");
    }
    if capabilities.has::<Query>() && policy_manager.check_sync(user, resource, Operation::Query) {
        permitted.push("query");
    }
    if capabilities.has::<Write>() && policy_manager.check_sync(user, resource, Operation::Write) {
        permitted.push("write");
    }
    if capabilities.has::<Serve>() && policy_manager.check_sync(user, resource, Operation::Serve) {
        permitted.push("serve");
    }
    if capabilities.has::<Manage>()
        && policy_manager.check_sync(user, resource, Operation::Manage)
    {
        permitted.push("manage");
    }
    if capabilities.has::<Context>()
        && policy_manager.check_sync(user, resource, Operation::Context)
    {
        permitted.push("context");
    }

    permitted.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_operation_codes() {
        assert_eq!(Operation::Infer.code(), 'i');
        assert_eq!(Operation::Train.code(), 't');
        assert_eq!(Operation::Query.code(), 'q');
        assert_eq!(Operation::Write.code(), 'w');
        assert_eq!(Operation::Serve.code(), 's');
        assert_eq!(Operation::Manage.code(), 'm');
        assert_eq!(Operation::Context.code(), 'c');
    }

    #[test]
    fn test_operation_from_code() {
        assert_eq!(Operation::from_code('i'), Some(Operation::Infer));
        assert_eq!(Operation::from_code('t'), Some(Operation::Train));
        assert_eq!(Operation::from_code('c'), Some(Operation::Context));
        assert_eq!(Operation::from_code('x'), None);
    }

    #[test]
    fn test_operation_dot_namespaced_strings() {
        assert_eq!(Operation::Infer.as_str(), "infer.generate");
        assert_eq!(Operation::Train.as_str(), "ttt.train");
        assert_eq!(Operation::Query.as_str(), "query.status");
        assert_eq!(Operation::Write.as_str(), "persist.save");
        assert_eq!(Operation::Manage.as_str(), "ttt.writeback");
        assert_eq!(Operation::Context.as_str(), "context.augment");
        assert_eq!(Operation::Serve.as_str(), "serve.api");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_operation_from_str() {
        // Legacy flat strings still parse correctly
        assert!(matches!(Operation::from_str("infer"), Ok(Operation::Infer)));
        assert!(matches!(Operation::from_str("train"), Ok(Operation::Train)));
        assert!(matches!(Operation::from_str("query"), Ok(Operation::Query)));
        assert!(matches!(Operation::from_str("write"), Ok(Operation::Write)));
        assert!(matches!(Operation::from_str("serve"), Ok(Operation::Serve)));
        assert!(matches!(Operation::from_str("manage"), Ok(Operation::Manage)));
        assert!(matches!(Operation::from_str("context"), Ok(Operation::Context)));
        // Dot-namespaced strings (emitted by Display) must also round-trip
        assert!(matches!("infer.generate".parse::<Operation>(), Ok(Operation::Infer)));
        assert!(matches!("ttt.train".parse::<Operation>(), Ok(Operation::Train)));
        assert!(matches!("ttt.writeback".parse::<Operation>(), Ok(Operation::Manage)));
        assert!(matches!("query.status".parse::<Operation>(), Ok(Operation::Query)));
        assert!(matches!("persist.save".parse::<Operation>(), Ok(Operation::Write)));
        assert!(matches!("serve.api".parse::<Operation>(), Ok(Operation::Serve)));
        assert!(matches!("context.augment".parse::<Operation>(), Ok(Operation::Context)));
        // Verify format!("{}", op).parse() round-trip for every variant
        for op in Operation::all() {
            let displayed = format!("{}", op);
            let parsed: Operation = displayed.parse().expect("round-trip parse failed");
            assert_eq!(&parsed, op);
        }
        assert!(Operation::from_str("foo").is_err());
    }

    #[test]
    fn test_operation_all() {
        let all = Operation::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&Operation::Context));
    }

    #[test]
    fn test_operation_from_dot_str() {
        // New dot-namespaced strings
        assert_eq!(Operation::from_dot_str("infer.generate"), Some(Operation::Infer));
        assert_eq!(Operation::from_dot_str("ttt.train"), Some(Operation::Train));
        assert_eq!(Operation::from_dot_str("ttt.writeback"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("ttt.evict"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("ttt.zero"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("query.status"), Some(Operation::Query));
        assert_eq!(Operation::from_dot_str("query.delta"), Some(Operation::Query));
        assert_eq!(Operation::from_dot_str("persist.save"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("persist.export"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("persist.snapshot"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("context.augment"), Some(Operation::Context));
        assert_eq!(Operation::from_dot_str("serve.api"), Some(Operation::Serve));
        // Legacy flat strings
        assert_eq!(Operation::from_dot_str("infer"), Some(Operation::Infer));
        assert_eq!(Operation::from_dot_str("train"), Some(Operation::Train));
        assert_eq!(Operation::from_dot_str("query"), Some(Operation::Query));
        assert_eq!(Operation::from_dot_str("write"), Some(Operation::Write));
        assert_eq!(Operation::from_dot_str("serve"), Some(Operation::Serve));
        assert_eq!(Operation::from_dot_str("manage"), Some(Operation::Manage));
        assert_eq!(Operation::from_dot_str("context"), Some(Operation::Context));
        // Unknown
        assert_eq!(Operation::from_dot_str("unknown"), None);
        assert_eq!(Operation::from_dot_str(""), None);
    }
}
