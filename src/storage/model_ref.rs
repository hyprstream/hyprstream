//! Simple model reference type for git-native model management

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Model reference in format: model:ref
/// Examples:
///   "llama3"          -> model "llama3" at registry's pinned commit
///   "llama3:main"     -> model "llama3" at branch "main"
///   "llama3:v2.0"     -> model "llama3" at tag "v2.0"
///   "llama3:abc123"   -> model "llama3" at commit "abc123"
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    pub model: String,
    pub git_ref: Option<String>,
}

impl ModelRef {
    /// Parse a model reference string
    pub fn parse(s: &str) -> Result<Self> {
        // Still support UUID for backwards compatibility
        if let Ok(uuid) = Uuid::parse_str(s) {
            return Ok(ModelRef {
                model: uuid.to_string(),
                git_ref: None,
            });
        }

        // Parse model:ref format
        let (model, git_ref) = match s.split_once(':') {
            Some((m, r)) => (m.to_string(), Some(r.to_string())),
            None => (s.to_string(), None),
        };

        // Validate model name
        validate_model_name(&model)?;
        if let Some(ref r) = git_ref {
            validate_git_ref(r)?;
        }

        Ok(ModelRef { model, git_ref })
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match &self.git_ref {
            Some(r) => format!("{}:{}", self.model, r),
            None => self.model.clone(),
        }
    }
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl std::str::FromStr for ModelRef {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

/// Validate model name
pub fn validate_model_name(name: &str) -> Result<()> {
    // Don't validate UUIDs (backwards compatibility)
    if Uuid::parse_str(name).is_ok() {
        return Ok(());
    }

    const MAX_LEN: usize = 50;
    const RESERVED: &[&str] = &["api", "system", "admin", "registry", "test"];

    if name.is_empty() || name.len() > MAX_LEN {
        bail!("Model name must be 1-{} characters", MAX_LEN);
    }

    // Must be alphanumeric with hyphens only (allow mixed case for HuggingFace compatibility)
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        bail!("Model name must contain only a-z, A-Z, 0-9, and hyphens");
    }

    // Can't start/end with hyphen
    if name.starts_with('-') || name.ends_with('-') {
        bail!("Model name cannot start or end with hyphen");
    }

    // No consecutive hyphens
    if name.contains("--") {
        bail!("Model name cannot contain consecutive hyphens");
    }

    // Check reserved names
    if RESERVED.contains(&name) {
        bail!("Model name '{}' is reserved", name);
    }

    Ok(())
}

/// Validate git reference
pub fn validate_git_ref(ref_str: &str) -> Result<()> {
    // Security: prevent injection attacks
    if ref_str.contains("..") || ref_str.contains("~") || ref_str.contains("^") {
        bail!("Git ref contains dangerous revision syntax");
    }

    if ref_str.len() > 128 {
        bail!("Git ref too long (max 128 chars)");
    }

    // Allow valid git ref characters
    if !ref_str.chars().all(|c|
        c.is_ascii_alphanumeric() ||
        c == '-' || c == '_' || c == '.' || c == '/'
    ) {
        bail!("Git ref contains invalid characters");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_ref() {
        // Model only
        let ref1 = ModelRef::parse("llama3").unwrap();
        assert_eq!(ref1.model, "llama3");
        assert_eq!(ref1.git_ref, None);

        // Model with branch
        let ref2 = ModelRef::parse("llama3:main").unwrap();
        assert_eq!(ref2.model, "llama3");
        assert_eq!(ref2.git_ref, Some("main".to_string()));

        // Model with tag
        let ref3 = ModelRef::parse("llama3:v2.0").unwrap();
        assert_eq!(ref3.model, "llama3");
        assert_eq!(ref3.git_ref, Some("v2.0".to_string()));

        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let ref4 = ModelRef::parse(uuid).unwrap();
        assert_eq!(ref4.model, uuid);
        assert_eq!(ref4.git_ref, None);
    }

    #[test]
    fn test_validate_model_name() {
        // Valid names
        assert!(validate_model_name("llama3").is_ok());
        assert!(validate_model_name("gpt-2").is_ok());
        assert!(validate_model_name("model-123").is_ok());
        assert!(validate_model_name("Qwen3-4B").is_ok()); // HuggingFace mixed case
        assert!(validate_model_name("GPT-4").is_ok()); // uppercase
        assert!(validate_model_name("Model").is_ok()); // mixed case now allowed

        // Invalid names
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("-model").is_err());
        assert!(validate_model_name("model-").is_err());
        assert!(validate_model_name("model--name").is_err());
        assert!(validate_model_name("api").is_err()); // reserved
        assert!(validate_model_name("model with spaces").is_err()); // spaces not allowed
    }

    #[test]
    fn test_validate_git_ref() {
        // Valid refs
        assert!(validate_git_ref("main").is_ok());
        assert!(validate_git_ref("v1.0.0").is_ok());
        assert!(validate_git_ref("feature/new-model").is_ok());
        assert!(validate_git_ref("abc123def").is_ok());

        // Invalid refs
        assert!(validate_git_ref("main~1").is_err());
        assert!(validate_git_ref("HEAD^").is_err());
        assert!(validate_git_ref("../etc/passwd").is_err());
    }
}