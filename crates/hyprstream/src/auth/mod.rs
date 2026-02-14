//! Authorization module for RBAC/ABAC access control
//!
//! Uses Casbin for policy-based access control with policies
//! stored in the `.registry/policies/` directory.
//!
//! Also provides JWT token authentication with Ed25519 signatures.

pub mod jwt;
mod policy_manager;

pub use jwt::{Claims, JwtError, TOKEN_PREFIX, ADMIN_TOKEN_PREFIX};
pub use policy_manager::{PolicyManager, PolicyError};

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

    /// Get the operation name for policy matching
    pub fn as_str(&self) -> &'static str {
        match self {
            Operation::Infer => "infer",
            Operation::Train => "train",
            Operation::Query => "query",
            Operation::Write => "write",
            Operation::Serve => "serve",
            Operation::Manage => "manage",
            Operation::Context => "context",
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

    /// Parse from operation name string (e.g., "write", "query").
    ///
    /// Used by generated `authorize()` overrides to convert the string action
    /// from `$mcpScope` annotations into an `Operation` enum value.
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "infer" => Ok(Operation::Infer),
            "train" => Ok(Operation::Train),
            "query" => Ok(Operation::Query),
            "write" => Ok(Operation::Write),
            "serve" => Ok(Operation::Serve),
            "manage" => Ok(Operation::Manage),
            "context" => Ok(Operation::Context),
            other => anyhow::bail!("Unknown operation: {}", other),
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
    fn test_operation_as_str() {
        assert_eq!(Operation::Infer.as_str(), "infer");
        assert_eq!(Operation::Train.as_str(), "train");
        assert_eq!(Operation::Manage.as_str(), "manage");
        assert_eq!(Operation::Context.as_str(), "context");
    }

    #[test]
    fn test_operation_from_str() {
        assert!(matches!(Operation::from_str("infer"), Ok(Operation::Infer)));
        assert!(matches!(Operation::from_str("train"), Ok(Operation::Train)));
        assert!(matches!(Operation::from_str("query"), Ok(Operation::Query)));
        assert!(matches!(Operation::from_str("write"), Ok(Operation::Write)));
        assert!(matches!(Operation::from_str("serve"), Ok(Operation::Serve)));
        assert!(matches!(Operation::from_str("manage"), Ok(Operation::Manage)));
        assert!(matches!(Operation::from_str("context"), Ok(Operation::Context)));
        assert!(Operation::from_str("foo").is_err());
    }

    #[test]
    fn test_operation_all() {
        let all = Operation::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&Operation::Context));
    }
}
