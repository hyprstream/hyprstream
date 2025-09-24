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
            if !git2::Reference::is_valid_name(r) {
                bail!("Invalid git reference name: '{}'", r);
            }
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

    // Follow Git reference naming rules (but adapted for model names)
    // Reject characters that Git doesn't allow
    if name.chars().any(|c|
        c.is_ascii_control() ||    // ASCII control characters
        c == ' ' ||                // Space
        c == '~' ||                // Tilde
        c == '^' ||                // Caret
        c == ':' ||                // Colon
        c == '?' ||                // Question mark
        c == '*' ||                // Asterisk
        c == '[' ||                // Open bracket
        c == '\\' ||               // Backslash
        c == '\t' ||               // Tab
        c == '\n' ||               // Newline
        c == '\r'                  // Carriage return
    ) {
        bail!("Model name contains invalid characters (spaces, control chars, or Git-forbidden chars)");
    }

    // No consecutive dots (Git rule)
    if name.contains("..") {
        bail!("Model name cannot contain consecutive dots (..)");
    }

    // Cannot end with .lock (Git rule)
    if name.ends_with(".lock") {
        bail!("Model name cannot end with '.lock'");
    }

    // Cannot be just "@" (Git rule)
    if name == "@" {
        bail!("Model name cannot be '@'");
    }

    // Cannot contain @{ sequence (Git rule)
    if name.contains("@{") {
        bail!("Model name cannot contain '@{{' sequence");
    }

    // Cannot start or end with dot (adapted Git rule - no component can start with dot)
    if name.starts_with('.') || name.ends_with('.') {
        bail!("Model name cannot start or end with dot");
    }

    // Check reserved names
    if RESERVED.contains(&name) {
        bail!("Model name '{}' is reserved", name);
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
        // Valid names (Git-compliant)
        assert!(validate_model_name("llama3").is_ok());
        assert!(validate_model_name("gpt-2").is_ok());
        assert!(validate_model_name("model-123").is_ok());
        assert!(validate_model_name("Qwen3-4B").is_ok()); // HuggingFace mixed case
        assert!(validate_model_name("GPT-4").is_ok()); // uppercase
        assert!(validate_model_name("Model").is_ok()); // mixed case
        assert!(validate_model_name("Qwen3-0.6B").is_ok()); // dots in version numbers
        assert!(validate_model_name("model-v1.2.3").is_ok()); // dots in versions
        assert!(validate_model_name("model_with_underscores").is_ok()); // underscores now allowed
        assert!(validate_model_name("model/submodel").is_ok()); // slashes now allowed
        assert!(validate_model_name("org-name_model.v1").is_ok()); // mixed separators
        assert!(validate_model_name("123model").is_ok()); // starting with numbers
        assert!(validate_model_name("model+special").is_ok()); // plus signs
        assert!(validate_model_name("model=equals").is_ok()); // equals signs

        // Invalid names (following Git rules)
        assert!(validate_model_name("").is_err()); // empty
        assert!(validate_model_name(".model").is_err()); // starts with dot
        assert!(validate_model_name("model.").is_err()); // ends with dot
        assert!(validate_model_name("model..name").is_err()); // consecutive dots
        assert!(validate_model_name("model.lock").is_err()); // ends with .lock
        assert!(validate_model_name("@").is_err()); // just @
        assert!(validate_model_name("model@{ref}").is_err()); // contains @{
        assert!(validate_model_name("api").is_err()); // reserved
        assert!(validate_model_name("model with spaces").is_err()); // spaces not allowed
        assert!(validate_model_name("model~1").is_err()); // tilde not allowed
        assert!(validate_model_name("model^HEAD").is_err()); // caret not allowed
        assert!(validate_model_name("model:tag").is_err()); // colon not allowed
        assert!(validate_model_name("model?query").is_err()); // question mark not allowed
        assert!(validate_model_name("model*glob").is_err()); // asterisk not allowed
        assert!(validate_model_name("model[bracket]").is_err()); // bracket not allowed
    }

    #[test]
    fn test_git_ref_validation() {
        // Test using git2::Reference::is_valid_name directly
        // Valid refs (according to libgit2)
        assert!(git2::Reference::is_valid_name("refs/heads/main"));
        assert!(git2::Reference::is_valid_name("refs/tags/v1.0.0"));
        assert!(git2::Reference::is_valid_name("refs/heads/feature/new-model"));
        assert!(git2::Reference::is_valid_name("HEAD"));
        assert!(git2::Reference::is_valid_name("ORIG_HEAD"));

        // Invalid refs (according to libgit2)
        assert!(!git2::Reference::is_valid_name("main")); // one-level refs not allowed by default
        assert!(!git2::Reference::is_valid_name("main~1")); // tilde
        assert!(!git2::Reference::is_valid_name("HEAD^")); // caret
        assert!(!git2::Reference::is_valid_name("branch..other")); // consecutive dots
        assert!(!git2::Reference::is_valid_name("@")); // just @
        assert!(!git2::Reference::is_valid_name("branch@{HEAD}")); // @{ sequence
        assert!(!git2::Reference::is_valid_name("/invalid")); // starts with slash
        assert!(!git2::Reference::is_valid_name("invalid/")); // ends with slash
        assert!(!git2::Reference::is_valid_name("refs/heads//double")); // consecutive slashes
        assert!(!git2::Reference::is_valid_name("refs/.hidden")); // component starts with dot
        assert!(!git2::Reference::is_valid_name("branch:tag")); // colon not allowed
        assert!(!git2::Reference::is_valid_name("branch with spaces")); // spaces not allowed
    }
}