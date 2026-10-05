//! Structured scope for fine-grained authorization.
//!
//! Format: action:resource:identifier
//! The identifier may contain one `:<ref>` suffix for model resources; the
//! parser preserves the full identifier so `model:<name>:<ref>` stays exact.
//! Examples:
//!   infer:model:qwen-7b     - Specific model inference
//!   subscribe:stream:abc    - Specific stream subscription
//!   read:model:*            - Read any model (explicit wildcard)
//!   manage:*:*              - Manage all resources

use crate::capnp::{FromCapnp, ToCapnp};
use crate::common_capnp;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Structured scope for fine-grained authorization.
///
/// Format: action:resource:identifier
/// A model identifier may carry one ref suffix (`<name>:<ref>`). The first
/// two separators are structural; the complete remainder is the identifier.
/// Examples:
///   infer:model:qwen-7b     - Specific model inference
///   subscribe:stream:abc    - Specific stream subscription
///   read:model:*            - Read any model (explicit wildcard)
///   manage:*:*              - Manage all resources
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
            action: reader.get_action()?.to_str()?.to_owned(),
            resource: reader.get_resource()?.to_str()?.to_owned(),
            identifier: reader.get_identifier()?.to_str()?.to_owned(),
        })
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.action, self.resource, self.identifier)
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

    /// Parse the canonical `action:resource:identifier` form.
    ///
    /// The first two separators delimit action and resource type. A model
    /// identifier may contain exactly one additional separator to encode its
    /// ref, as in `infer:model:qwen2.5-0.5b-instruct:main`. No components are
    /// discarded, and other resource identifiers cannot contain separators.
    pub fn parse(s: &str) -> Result<Self> {
        let (action, tail) = s
            .split_once(':')
            .ok_or_else(|| anyhow!("Invalid scope format: {}", s))?;
        let (resource, identifier) = tail
            .split_once(':')
            .ok_or_else(|| anyhow!("Invalid scope format: {}", s))?;
        if action.is_empty() || resource.is_empty() || identifier.is_empty() {
            return Err(anyhow!("Invalid scope format: {}", s));
        }
        if identifier.contains(':') {
            let (model, reference) = identifier
                .split_once(':')
                .ok_or_else(|| anyhow!("Invalid scope format: {}", s))?;
            if resource != "model"
                || model.is_empty()
                || reference.is_empty()
                || reference.contains(':')
            {
                return Err(anyhow!("Invalid scope format: {}", s));
            }
        }
        Ok(Self::new(
            action.to_owned(),
            resource.to_owned(),
            identifier.to_owned(),
        ))
    }

    /// Return the exact Policy resource represented by this scope.
    pub fn policy_resource(&self) -> String {
        format!("{}:{}", self.resource, self.identifier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_parse() -> Result<()> {
        let scope = Scope::parse("infer:model:qwen-7b")?;
        assert_eq!(scope.action, "infer");
        assert_eq!(scope.resource, "model");
        assert_eq!(scope.identifier, "qwen-7b");
        Ok(())
    }

    #[test]
    fn model_ref_scope_preserves_the_entire_dispatch_resource() -> Result<()> {
        const MODEL_REF: &str = "qwen2.5-0.5b-instruct:main";
        let scope = Scope::parse("infer:model:qwen2.5-0.5b-instruct:main")?;
        assert_eq!(scope.action, "infer");
        assert_eq!(scope.resource, "model");
        assert_eq!(scope.identifier, MODEL_REF);
        assert_eq!(scope.to_string(), "infer:model:qwen2.5-0.5b-instruct:main");
        assert_eq!(scope.policy_resource(), format!("model:{MODEL_REF}"));
        Ok(())
    }

    #[test]
    fn malformed_or_ambiguous_scope_separators_are_rejected() {
        for scope in [
            "infer:model",
            "infer::qwen:main",
            "infer:model:",
            "infer:model:qwen::main",
            "infer:model:qwen:main:other",
            "infer:registry:repo:main",
            ":model:qwen:main",
        ] {
            assert!(
                Scope::parse(scope).is_err(),
                "accepted malformed scope {scope}"
            );
        }
    }

    #[test]
    fn test_scope_to_string() {
        let scope = Scope::new("infer".to_owned(), "model".to_owned(), "qwen-7b".to_owned());
        assert_eq!(scope.to_string(), "infer:model:qwen-7b");
    }

}
