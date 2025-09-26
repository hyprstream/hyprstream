//! Advanced model reference type for git-native model management

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use uuid::Uuid;
use git2::{Oid, Repository};
use std::fmt;

/// Serde helper for Oid serialization
mod oid_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(oid: &Oid, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&oid.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Oid, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Oid::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// Git reference types for precise model versioning
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GitRef {
    /// Use the default branch (usually main/master)
    DefaultBranch,
    /// Reference a specific branch by name
    Branch(String),
    /// Reference a specific tag by name
    Tag(String),
    /// Reference a specific commit by SHA
    #[serde(with = "oid_serde")]
    Commit(Oid),
    /// Complex revision specification (HEAD~1, main..develop, etc.)
    Revspec(String),
}

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

impl GitRef {
    /// Parse a git reference string into the appropriate GitRef type
    pub fn parse(git_ref_str: &str, repo: Option<&Repository>) -> Result<Self> {
        // Check if it's a commit SHA (40-character hex string)
        if git_ref_str.len() >= 7 && git_ref_str.chars().all(|c| c.is_ascii_hexdigit()) {
            if git_ref_str.len() == 40 {
                // Full SHA
                let oid = Oid::from_str(git_ref_str)
                    .map_err(|e| anyhow::anyhow!("Invalid commit SHA '{}': {}", git_ref_str, e))?;
                return Ok(GitRef::Commit(oid));
            } else if git_ref_str.len() >= 7 {
                // Abbreviated SHA - try to resolve it if we have a repo
                if let Some(repo) = repo {
                    if let Ok(oid) = repo.revparse_single(git_ref_str) {
                        return Ok(GitRef::Commit(oid.id()));
                    }
                }
                // If we can't resolve it, treat it as a revspec
                return Ok(GitRef::Revspec(git_ref_str.to_string()));
            }
        }

        // Check for revspec patterns (contains ~, ^, .., etc.)
        if git_ref_str.contains('~') || git_ref_str.contains('^') || git_ref_str.contains("..") ||
           git_ref_str.contains("@{") || git_ref_str == "HEAD" || git_ref_str == "ORIG_HEAD" {
            return Ok(GitRef::Revspec(git_ref_str.to_string()));
        }

        // Check if it looks like a tag (starts with 'v' followed by numbers/dots)
        if git_ref_str.starts_with('v') && git_ref_str[1..].chars().next().map_or(false, |c| c.is_ascii_digit()) {
            return Ok(GitRef::Tag(git_ref_str.to_string()));
        }

        // If we have a repository, try to determine the type more precisely
        if let Some(repo) = repo {
            // Check if it's a branch
            if repo.find_branch(git_ref_str, git2::BranchType::Local).is_ok() ||
               repo.find_branch(git_ref_str, git2::BranchType::Remote).is_ok() {
                return Ok(GitRef::Branch(git_ref_str.to_string()));
            }

            // Check if it's a tag
            if let Ok(_) = repo.find_reference(&format!("refs/tags/{}", git_ref_str)) {
                return Ok(GitRef::Tag(git_ref_str.to_string()));
            }
        }

        // Default to branch (most common case)
        Ok(GitRef::Branch(git_ref_str.to_string()))
    }

    /// Convert to string representation for display and storage
    pub fn to_string(&self) -> Option<String> {
        match self {
            GitRef::DefaultBranch => None,
            GitRef::Branch(name) => Some(name.clone()),
            GitRef::Tag(name) => Some(name.clone()),
            GitRef::Commit(oid) => Some(oid.to_string()),
            GitRef::Revspec(spec) => Some(spec.clone()),
        }
    }

    /// Get the reference as a string suitable for git operations
    pub fn as_ref_str(&self) -> Option<&str> {
        match self {
            GitRef::DefaultBranch => None,
            GitRef::Branch(name) => Some(name),
            GitRef::Tag(name) => Some(name),
            GitRef::Commit(_) => None, // Commits need to be handled specially
            GitRef::Revspec(spec) => Some(spec),
        }
    }

    /// Get the commit OID if this is a commit reference
    pub fn as_oid(&self) -> Option<&Oid> {
        match self {
            GitRef::Commit(oid) => Some(oid),
            _ => None,
        }
    }
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
        Self::parse_with_repo(s, None)
    }

    /// Parse a model reference string with repository context for better type detection
    pub fn parse_with_repo(s: &str, repo: Option<&Repository>) -> Result<Self> {
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
                let git_ref = GitRef::parse(r, repo)?;
                (m.to_string(), git_ref)
            },
            None => (s.to_string(), GitRef::DefaultBranch),
        };

        // Validate model name
        validate_model_name(&model)?;

        Ok(ModelRef { model, git_ref })
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self.git_ref.to_string() {
            Some(git_ref_str) => format!("{}:{}", self.model, git_ref_str),
            None => self.model.clone(),
        }
    }

    /// Get the git reference as an option string (for compatibility)
    pub fn git_ref_str(&self) -> Option<String> {
        self.git_ref.to_string()
    }
}

impl fmt::Display for GitRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_string() {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "<default>"),
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        // Model only (default branch)
        let ref1 = ModelRef::parse("llama3").unwrap();
        assert_eq!(ref1.model, "llama3");
        assert_eq!(ref1.git_ref, GitRef::DefaultBranch);

        // Model with branch
        let ref2 = ModelRef::parse("llama3:main").unwrap();
        assert_eq!(ref2.model, "llama3");
        assert_eq!(ref2.git_ref, GitRef::Branch("main".to_string()));

        // Model with tag
        let ref3 = ModelRef::parse("llama3:v2.0").unwrap();
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
        // Test GitRef parsing without repository context
        assert_eq!(GitRef::parse("main", None).unwrap(), GitRef::Branch("main".to_string()));
        assert_eq!(GitRef::parse("v1.0.0", None).unwrap(), GitRef::Tag("v1.0.0".to_string()));
        assert_eq!(GitRef::parse("HEAD~1", None).unwrap(), GitRef::Revspec("HEAD~1".to_string()));
        assert_eq!(GitRef::parse("main^2", None).unwrap(), GitRef::Revspec("main^2".to_string()));

        // Full SHA
        let sha = "1234567890abcdef1234567890abcdef12345678";
        if let GitRef::Commit(oid) = GitRef::parse(sha, None).unwrap() {
            assert_eq!(oid.to_string(), sha);
        } else {
            panic!("Expected GitRef::Commit for full SHA");
        }

        // Abbreviated SHA (should be treated as revspec without repo)
        assert_eq!(GitRef::parse("1234567", None).unwrap(), GitRef::Revspec("1234567".to_string()));
    }

    #[test]
    fn test_git_ref_to_string() {
        assert_eq!(GitRef::DefaultBranch.to_string(), None);
        assert_eq!(GitRef::Branch("main".to_string()).to_string(), Some("main".to_string()));
        assert_eq!(GitRef::Tag("v1.0".to_string()).to_string(), Some("v1.0".to_string()));
        assert_eq!(GitRef::Revspec("HEAD~1".to_string()).to_string(), Some("HEAD~1".to_string()));

        let oid = Oid::from_str("1234567890abcdef1234567890abcdef12345678").unwrap();
        assert_eq!(GitRef::Commit(oid).to_string(), Some("1234567890abcdef1234567890abcdef12345678".to_string()));
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
        // Test that our GitRef parsing handles various git reference formats
        // These should all parse successfully (even if git2::Reference::is_valid_name would reject them)
        // because our GitRef enum is more permissive and handles different contexts

        // Simple branch names
        assert!(matches!(GitRef::parse("main", None).unwrap(), GitRef::Branch(_)));
        assert!(matches!(GitRef::parse("feature/new-model", None).unwrap(), GitRef::Branch(_)));

        // Tags
        assert!(matches!(GitRef::parse("v1.0.0", None).unwrap(), GitRef::Tag(_)));
        assert!(matches!(GitRef::parse("v2.1", None).unwrap(), GitRef::Tag(_)));

        // Revspecs
        assert!(matches!(GitRef::parse("HEAD", None).unwrap(), GitRef::Revspec(_)));
        assert!(matches!(GitRef::parse("ORIG_HEAD", None).unwrap(), GitRef::Revspec(_)));
        assert!(matches!(GitRef::parse("main~1", None).unwrap(), GitRef::Revspec(_)));
        assert!(matches!(GitRef::parse("HEAD^", None).unwrap(), GitRef::Revspec(_)));
        assert!(matches!(GitRef::parse("branch..other", None).unwrap(), GitRef::Revspec(_)));
        assert!(matches!(GitRef::parse("branch@{HEAD}", None).unwrap(), GitRef::Revspec(_)));

        // Commits
        assert!(matches!(GitRef::parse("1234567890abcdef1234567890abcdef12345678", None).unwrap(), GitRef::Commit(_)));

        // Test libgit2 validation separately (for reference)
        assert!(git2::Reference::is_valid_name("refs/heads/main"));
        assert!(git2::Reference::is_valid_name("refs/tags/v1.0.0"));
        assert!(git2::Reference::is_valid_name("refs/heads/feature/new-model"));
        assert!(git2::Reference::is_valid_name("HEAD"));
        assert!(git2::Reference::is_valid_name("ORIG_HEAD"));
    }
}