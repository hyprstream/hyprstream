//! Domain-specific newtype wrappers for type safety
//!
//! This module provides newtype patterns to replace stringly-typed APIs
//! with type-safe domain types, improving compile-time guarantees.

use crate::storage::errors::{ModelRefError, ModelRefResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Type-safe wrapper for model names with validation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelName(String);

impl ModelName {
    /// Create a new validated model name
    pub fn new(name: impl Into<String>) -> ModelRefResult<Self> {
        let name = name.into();
        crate::storage::validate_model_name(&name)
            .map_err(|e| ModelRefError::GitRefParsing(e.to_string()))?;
        Ok(Self(name))
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to owned string
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ModelName {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl AsRef<str> for ModelName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Type-safe wrapper for Git branch names
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchName(String);

impl BranchName {
    /// Create a new branch name with Git validation
    pub fn new(name: impl Into<String>) -> ModelRefResult<Self> {
        let name = name.into();

        // Basic Git branch name validation
        if name.is_empty() {
            return Err(ModelRefError::GitRefParsing("Branch name cannot be empty".to_string()));
        }

        if name.starts_with('-') || name.ends_with('-') {
            return Err(ModelRefError::GitRefParsing("Branch name cannot start or end with hyphen".to_string()));
        }

        if name.contains("..") || name.contains("@{") {
            return Err(ModelRefError::GitRefParsing("Branch name contains invalid sequences".to_string()));
        }

        // Check if it's a valid Git reference name
        let ref_name = format!("refs/heads/{}", name);
        if !git2::Reference::is_valid_name(&ref_name) {
            return Err(ModelRefError::GitRefParsing(format!("Invalid branch name: {}", name)));
        }

        Ok(Self(name))
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Get as Git reference name
    pub fn as_ref_name(&self) -> String {
        format!("refs/heads/{}", self.0)
    }
}

impl fmt::Display for BranchName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for BranchName {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Type-safe wrapper for Git tag names
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TagName(String);

impl TagName {
    /// Create a new tag name with Git validation
    pub fn new(name: impl Into<String>) -> ModelRefResult<Self> {
        let name = name.into();

        if name.is_empty() {
            return Err(ModelRefError::GitRefParsing("Tag name cannot be empty".to_string()));
        }

        // Check if it's a valid Git reference name
        let ref_name = format!("refs/tags/{}", name);
        if !git2::Reference::is_valid_name(&ref_name) {
            return Err(ModelRefError::GitRefParsing(format!("Invalid tag name: {}", name)));
        }

        Ok(Self(name))
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Get as Git reference name
    pub fn as_ref_name(&self) -> String {
        format!("refs/tags/{}", self.0)
    }

    /// Check if this looks like a semantic version tag
    pub fn is_semver(&self) -> bool {
        self.0.starts_with('v') &&
        self.0[1..].chars().next().map_or(false, |c| c.is_ascii_digit())
    }
}

impl fmt::Display for TagName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for TagName {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Type-safe wrapper for Git revision specifications
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RevSpec(String);

impl RevSpec {
    /// Create a new revision specification
    pub fn new(spec: impl Into<String>) -> Self {
        Self(spec.into())
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if this is a simple commit-like revspec
    pub fn is_commit_like(&self) -> bool {
        self.0.chars().all(|c| c.is_ascii_hexdigit()) && self.0.len() >= 7
    }

    /// Check if this is a relative revspec (contains ~, ^, etc.)
    pub fn is_relative(&self) -> bool {
        self.0.contains('~') || self.0.contains('^')
    }

    /// Check if this is a range revspec (contains ..)
    pub fn is_range(&self) -> bool {
        self.0.contains("..")
    }
}

impl fmt::Display for RevSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for RevSpec {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

/// Type-safe wrapper for adapter names
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdapterName(String);

impl AdapterName {
    /// Create a new adapter name with validation
    pub fn new(name: impl Into<String>) -> ModelRefResult<Self> {
        let name = name.into();

        if name.is_empty() {
            return Err(ModelRefError::GitRefParsing("Adapter name cannot be empty".to_string()));
        }

        // Basic filename safety checks
        if name.contains('/') || name.contains('\\') || name.contains('\0') {
            return Err(ModelRefError::GitRefParsing("Adapter name contains invalid characters".to_string()));
        }

        if name.starts_with('.') || name.ends_with('.') {
            return Err(ModelRefError::GitRefParsing("Adapter name cannot start or end with dot".to_string()));
        }

        Ok(Self(name))
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Create indexed filename for this adapter
    pub fn to_indexed_filename(&self, index: u32) -> String {
        format!("{:02}_{}.safetensors", index, self.0)
    }

    /// Create config filename for this adapter
    pub fn to_config_filename(&self, index: u32) -> String {
        format!("{:02}_{}.config.json", index, self.0)
    }
}

impl fmt::Display for AdapterName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for AdapterName {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Type-safe wrapper for remote names
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RemoteName(String);

impl RemoteName {
    /// Create a new remote name
    pub fn new(name: impl Into<String>) -> ModelRefResult<Self> {
        let name = name.into();

        if name.is_empty() {
            return Err(ModelRefError::GitRefParsing("Remote name cannot be empty".to_string()));
        }

        // Git remote name validation
        if name.contains(' ') || name.contains('\t') || name.contains('\n') {
            return Err(ModelRefError::GitRefParsing("Remote name contains whitespace".to_string()));
        }

        Ok(Self(name))
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Common remote names
    pub fn origin() -> Self {
        Self("origin".to_string())
    }

    pub fn upstream() -> Self {
        Self("upstream".to_string())
    }
}

impl fmt::Display for RemoteName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for RemoteName {
    fn default() -> Self {
        Self::origin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_validation() {
        // Valid names
        assert!(ModelName::new("llama3").is_ok());
        assert!(ModelName::new("gpt-4").is_ok());
        assert!(ModelName::new("model_123").is_ok());

        // Invalid names would be caught by validate_model_name
        // The validation logic is already tested in model_ref.rs
    }

    #[test]
    fn test_branch_name_validation() {
        // Valid branch names
        assert!(BranchName::new("main").is_ok());
        assert!(BranchName::new("feature/new-model").is_ok());
        assert!(BranchName::new("v1.0.0").is_ok());

        // Invalid branch names
        assert!(BranchName::new("").is_err());
        assert!(BranchName::new("-invalid").is_err());
        assert!(BranchName::new("invalid-").is_err());
        assert!(BranchName::new("inv..alid").is_err());
        assert!(BranchName::new("inv@{alid").is_err());
    }

    #[test]
    fn test_tag_name_validation() {
        // Valid tag names
        assert!(TagName::new("v1.0.0").is_ok());
        assert!(TagName::new("release-1.0").is_ok());

        // Invalid tag names
        assert!(TagName::new("").is_err());
    }

    #[test]
    fn test_revspec_classification() {
        let commit_like = RevSpec::new("1234567890abcdef");
        assert!(commit_like.is_commit_like());
        assert!(!commit_like.is_relative());
        assert!(!commit_like.is_range());

        let relative = RevSpec::new("HEAD~1");
        assert!(!relative.is_commit_like());
        assert!(relative.is_relative());
        assert!(!relative.is_range());

        let range = RevSpec::new("main..develop");
        assert!(!range.is_commit_like());
        assert!(!range.is_relative());
        assert!(range.is_range());
    }

    #[test]
    fn test_adapter_name_validation() {
        // Valid adapter names
        assert!(AdapterName::new("base").is_ok());
        assert!(AdapterName::new("fine_tuned").is_ok());
        assert!(AdapterName::new("lora-adapter").is_ok());

        // Invalid adapter names
        assert!(AdapterName::new("").is_err());
        assert!(AdapterName::new("inv/alid").is_err());
        assert!(AdapterName::new("inv\\alid").is_err());
        assert!(AdapterName::new(".invalid").is_err());
        assert!(AdapterName::new("invalid.").is_err());
    }

    #[test]
    fn test_adapter_filename_generation() {
        let adapter = AdapterName::new("test_adapter").unwrap();
        assert_eq!(adapter.to_indexed_filename(5), "05_test_adapter.safetensors");
        assert_eq!(adapter.to_config_filename(5), "05_test_adapter.config.json");
    }

    #[test]
    fn test_semver_tag_detection() {
        let semver_tag = TagName::new("v1.2.3").unwrap();
        assert!(semver_tag.is_semver());

        let non_semver_tag = TagName::new("release-candidate").unwrap();
        assert!(!non_semver_tag.is_semver());
    }
}