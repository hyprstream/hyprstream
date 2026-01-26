//! Structured scope for fine-grained authorization.
//!
//! Format: action:resource:identifier
//! Examples:
//!   infer:model:qwen-7b     - Specific model inference
//!   subscribe:stream:abc    - Specific stream subscription
//!   read:model:*            - Read any model (explicit wildcard)
//!   admin:*:*               - Admin wildcard

use crate::common_capnp;
use crate::capnp::{ToCapnp, FromCapnp};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Structured scope for fine-grained authorization.
///
/// Format: action:resource:identifier
/// Examples:
///   infer:model:qwen-7b     - Specific model inference
///   subscribe:stream:abc    - Specific stream subscription
///   read:model:*            - Read any model (explicit wildcard)
///   admin:*:*               - Admin wildcard
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub action: String,
    pub resource: String,
    pub identifier: String,
}

impl ToCapnp for Scope {
    type Builder<'a> = common_capnp::scope::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_action(&self.action);
        builder.set_resource(&self.resource);
        builder.set_identifier(&self.identifier);
    }
}

impl FromCapnp for Scope {
    type Reader<'a> = common_capnp::scope::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        Ok(Self {
            action: reader.get_action()?.to_str()?.to_string(),
            resource: reader.get_resource()?.to_str()?.to_string(),
            identifier: reader.get_identifier()?.to_str()?.to_string(),
        })
    }
}

impl Scope {
    /// Create a new scope.
    pub fn new(action: String, resource: String, identifier: String) -> Self {
        Self {
            action,
            resource,
            identifier,
        }
    }

    /// Parse from string format "action:resource:identifier"
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(anyhow!("Invalid scope format: {}", s));
        }
        Ok(Self::new(
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
        ))
    }

    /// Format as "action:resource:identifier"
    pub fn to_string(&self) -> String {
        format!("{}:{}:{}", self.action, self.resource, self.identifier)
    }

    /// Check if this scope grants permission for the required scope.
    ///
    /// Safe wildcard matching with action/resource isolation:
    /// - Actions must match exactly (no wildcards)
    /// - Resources must match exactly (no wildcards)
    /// - Identifier: "*" grants all, otherwise exact match
    pub fn grants(&self, required: &Scope) -> bool {
        // Action must match exactly (no wildcards for actions)
        if self.action != required.action {
            return false;
        }

        // Resource must match exactly (no wildcards for resource types)
        if self.resource != required.resource {
            return false;
        }

        // Identifier matching: "*" grants all, otherwise exact match
        match self.identifier.as_str() {
            "*" => true,
            id => id == required.identifier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_parse() {
        let scope = Scope::parse("infer:model:qwen-7b").unwrap();
        assert_eq!(scope.action, "infer");
        assert_eq!(scope.resource, "model");
        assert_eq!(scope.identifier, "qwen-7b");
    }

    #[test]
    fn test_scope_to_string() {
        let scope = Scope::new("infer".to_string(), "model".to_string(), "qwen-7b".to_string());
        assert_eq!(scope.to_string(), "infer:model:qwen-7b");
    }

    #[test]
    fn test_scope_grants_exact_match() {
        let granted = Scope::parse("infer:model:qwen-7b").unwrap();
        let required = Scope::parse("infer:model:qwen-7b").unwrap();
        assert!(granted.grants(&required));
    }

    #[test]
    fn test_scope_grants_wildcard() {
        let granted = Scope::parse("infer:model:*").unwrap();
        let required = Scope::parse("infer:model:qwen-7b").unwrap();
        assert!(granted.grants(&required));
    }

    #[test]
    fn test_scope_action_isolation() {
        let granted = Scope::parse("read:model:*").unwrap();
        let required = Scope::parse("write:model:qwen-7b").unwrap();
        assert!(!granted.grants(&required));
    }

    #[test]
    fn test_scope_resource_isolation() {
        let granted = Scope::parse("infer:model:*").unwrap();
        let required = Scope::parse("infer:stream:abc").unwrap();
        assert!(!granted.grants(&required));
    }
}
