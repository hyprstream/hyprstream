//! Advanced model reference type for git-native model management

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// Re-export git2db's GitRef instead of duplicating
pub use git2db::GitRef;

/// Model reference with typed git references
/// Examples:
///   "llama3"          -> model "llama3" with GitRef::DefaultBranch
///   "llama3:main"     -> model "llama3" with GitRef::Branch("main")
///   "llama3:v2.0"     -> model "llama3" with GitRef::Tag("v2.0")
///   "llama3:abc123"   -> model "llama3" with GitRef::Commit(oid)
///   "llama3:HEAD~1"   -> model "llama3" with GitRef::Revspec("HEAD~1")
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    pub model: String,
    pub git_ref: GitRef,
}


impl ModelRef {
    /// Create a new ModelRef with default branch
    pub fn new(model: String) -> Self {
        Self {
            model,
            git_ref: GitRef::DefaultBranch,
        }
    }

    /// Create a new ModelRef with a specific git reference
    pub fn with_ref(model: String, git_ref: GitRef) -> Self {
        Self { model, git_ref }
    }

    /// Parse a model reference string
    pub fn parse(s: &str) -> Result<Self> {
        // Still support UUID for backwards compatibility
        if let Ok(uuid) = Uuid::parse_str(s) {
            return Ok(ModelRef {
                model: uuid.to_string(),
                git_ref: GitRef::DefaultBranch,
            });
        }

        // Parse model:ref format
        let (model, git_ref) = match s.split_once(':') {
            Some((m, r)) => {
                // Use git2db's GitRef::parse() and convert error
                let git_ref = GitRef::parse(r)
                    .map_err(|e| anyhow::anyhow!("Failed to parse git ref '{}': {}", r, e))?;
                (m.to_string(), git_ref)
            }
            None => (s.to_string(), GitRef::DefaultBranch),
        };

        // Validate model name
        validate_model_name(&model)?;

        Ok(ModelRef { model, git_ref })
    }


    /// Get the git reference as an option string (for compatibility)
    pub fn git_ref_str(&self) -> Option<String> {
        match &self.git_ref {
            GitRef::DefaultBranch => None,
            _ => Some(self.git_ref.display_name()),
        }
    }
}

// Note: impl Display for GitRef removed - git2db already provides it

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.git_ref {
            GitRef::DefaultBranch => write!(f, "{}", self.model),
            _ => write!(f, "{}:{}", self.model, self.git_ref.display_name()),
        }
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
    if name.chars().any(
        |c| {
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
        c == '\r'
        }, // Carriage return
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
        // Model only (default branch)
        let ref1 = ModelRef::parse("llama3").unwrap();
        assert_eq!(ref1.model, "llama3");
        assert_eq!(ref1.git_ref, GitRef::DefaultBranch);

        // Model with branch
        let ref2 = ModelRef::parse("llama3:main").unwrap();
        assert_eq!(ref2.model, "llama3");
        assert_eq!(ref2.git_ref, GitRef::Branch("main".to_string()));

        // Model with tag (must use explicit tag syntax)
        let ref3 = ModelRef::parse("llama3:tags/v2.0").unwrap();
        assert_eq!(ref3.model, "llama3");
        assert_eq!(ref3.git_ref, GitRef::Tag("v2.0".to_string()));

        // Model with commit SHA
        let ref4 = ModelRef::parse("llama3:1234567890abcdef1234567890abcdef12345678").unwrap();
        assert_eq!(ref4.model, "llama3");
        if let GitRef::Commit(oid) = ref4.git_ref {
            assert_eq!(oid.to_string(), "1234567890abcdef1234567890abcdef12345678");
        } else {
            panic!("Expected GitRef::Commit");
        }

        // Model with revspec
        let ref5 = ModelRef::parse("llama3:HEAD~1").unwrap();
        assert_eq!(ref5.model, "llama3");
        assert_eq!(ref5.git_ref, GitRef::Revspec("HEAD~1".to_string()));

        // UUID backwards compatibility
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let ref6 = ModelRef::parse(uuid).unwrap();
        assert_eq!(ref6.model, uuid);
        assert_eq!(ref6.git_ref, GitRef::DefaultBranch);
    }

    #[test]
    fn test_git_ref_parsing() {
        // Test GitRef parsing (now using git2db's GitRef::parse)
        assert_eq!(
            GitRef::parse("main").unwrap(),
            GitRef::Branch("main".to_string())
        );
        // Explicit tag syntax required
        assert!(matches!(
            GitRef::parse("tags/v1.0.0").unwrap(),
            GitRef::Tag(_)
        ));
        assert!(matches!(
            GitRef::parse("HEAD~1").unwrap(),
            GitRef::Revspec(_) | GitRef::DefaultBranch
        ));
        assert!(matches!(
            GitRef::parse("main^2").unwrap(),
            GitRef::Revspec(_)
        ));

        // Full SHA
        let sha = "1234567890abcdef1234567890abcdef12345678";
        if let GitRef::Commit(oid) = GitRef::parse(sha).unwrap() {
            assert_eq!(oid.to_string(), sha);
        } else {
            panic!("Expected GitRef::Commit for full SHA");
        }

        // Abbreviated SHA (should be treated as revspec)
        assert!(matches!(
            GitRef::parse("1234567").unwrap(),
            GitRef::Revspec(_)
        ));
    }

    #[test]
    fn test_git_ref_display() {
        // Test display_name (git2db's API)
        assert_eq!(GitRef::DefaultBranch.display_name(), "HEAD");
        assert_eq!(
            GitRef::Branch("main".to_string()).display_name(),
            "main"
        );
        // git2db's Tag display includes tags/ prefix
        assert!(GitRef::Tag("v1.0".to_string()).display_name().contains("v1.0"));
        assert_eq!(
            GitRef::Revspec("HEAD~1".to_string()).display_name(),
            "HEAD~1"
        );

        let oid = git2db::Oid::from_str("1234567890abcdef1234567890abcdef12345678").unwrap();
        // display_name shows abbreviated SHA
        assert!(GitRef::Commit(oid).display_name().contains("12345678"));
    }

    #[test]
    fn test_model_ref_compatibility() {
        // Test that the new API provides backward compatibility
        let model_ref = ModelRef::parse("llama3:main").unwrap();

        // Should have compatible git_ref_str method
        assert_eq!(model_ref.git_ref_str(), Some("main".to_string()));

        // Should display the same as before
        assert_eq!(model_ref.to_string(), "llama3:main");

        // Default branch should show just the model name
        let default_ref = ModelRef::parse("llama3").unwrap();
        assert_eq!(default_ref.to_string(), "llama3");
        assert_eq!(default_ref.git_ref_str(), None);
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
        // Test that git2db's GitRef parsing handles various git reference formats

        // Simple branch names
        assert!(matches!(
            GitRef::parse("main").unwrap(),
            GitRef::Branch(_)
        ));
        assert!(matches!(
            GitRef::parse("feature/new-model").unwrap(),
            GitRef::Branch(_)
        ));

        // Tags (explicit tag syntax required)
        assert!(matches!(
            GitRef::parse("tags/v1.0.0").unwrap(),
            GitRef::Tag(_)
        ));
        assert!(matches!(
            GitRef::parse("refs/tags/v2.1").unwrap(),
            GitRef::Tag(_)
        ));

        // Revspecs or DefaultBranch
        // Note: git2db treats "HEAD" as DefaultBranch, not Revspec
        assert!(matches!(
            GitRef::parse("HEAD").unwrap(),
            GitRef::DefaultBranch
        ));
        assert!(matches!(
            GitRef::parse("main~1").unwrap(),
            GitRef::Revspec(_)
        ));
        assert!(matches!(
            GitRef::parse("HEAD^").unwrap(),
            GitRef::Revspec(_)
        ));
        assert!(matches!(
            GitRef::parse("branch..other").unwrap(),
            GitRef::Revspec(_)
        ));
        assert!(matches!(
            GitRef::parse("branch@{HEAD}").unwrap(),
            GitRef::Revspec(_)
        ));

        // Commits
        assert!(matches!(
            GitRef::parse("1234567890abcdef1234567890abcdef12345678").unwrap(),
            GitRef::Commit(_)
        ));

        // Test libgit2 validation separately (for reference)
        assert!(git2::Reference::is_valid_name("refs/heads/main"));
        assert!(git2::Reference::is_valid_name("refs/tags/v1.0.0"));
        assert!(git2::Reference::is_valid_name(
            "refs/heads/feature/new-model"
        ));
        assert!(git2::Reference::is_valid_name("HEAD"));
    }
}
