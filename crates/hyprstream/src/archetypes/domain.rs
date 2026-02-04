//! Domain trait and detection result types
//!
//! This module defines the `Domain` trait that replaces `RepoArchetype`,
//! providing richer error handling and type-safe capability detection.

use super::capabilities::CapabilitySet;
use std::fmt;
use std::path::{Path, PathBuf};

/// Error that can occur during domain detection
#[derive(Debug, Clone)]
pub enum DetectionError {
    /// I/O error accessing repository path
    Io(String),
    /// Path does not exist
    PathNotFound(PathBuf),
    /// Invalid repository structure (e.g., malformed config file)
    InvalidStructure(String),
}

impl fmt::Display for DetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectionError::Io(msg) => write!(f, "I/O error: {msg}"),
            DetectionError::PathNotFound(path) => write!(f, "Path not found: {}", path.display()),
            DetectionError::InvalidStructure(msg) => write!(f, "Invalid structure: {msg}"),
        }
    }
}

impl std::error::Error for DetectionError {}

impl From<std::io::Error> for DetectionError {
    fn from(e: std::io::Error) -> Self {
        DetectionError::Io(e.to_string())
    }
}

/// Result of domain detection
///
/// Unlike the old boolean `detect()` method, this provides rich information
/// about detection success, failure, or errors.
#[derive(Debug, Clone)]
pub enum DetectionResult {
    /// Domain was detected with these capabilities
    Detected(CapabilitySet),
    /// Domain was not detected (normal case - not an error)
    NotDetected,
    /// Detection failed due to an error (should be logged)
    Error(DetectionError),
}

impl DetectionResult {
    /// Check if domain was detected
    pub fn is_detected(&self) -> bool {
        matches!(self, DetectionResult::Detected(_))
    }

    /// Check if detection resulted in error
    pub fn is_error(&self) -> bool {
        matches!(self, DetectionResult::Error(_))
    }

    /// Get capabilities if detected
    pub fn capabilities(&self) -> Option<&CapabilitySet> {
        match self {
            DetectionResult::Detected(caps) => Some(caps),
            _ => None,
        }
    }

    /// Convert to Option, ignoring errors
    pub fn into_option(self) -> Option<CapabilitySet> {
        match self {
            DetectionResult::Detected(caps) => Some(caps),
            _ => None,
        }
    }
}

/// A domain that provides capabilities for repositories
///
/// Domains replace archetypes in the new capability system. Each domain:
/// - Detects whether a repository belongs to it
/// - Provides a set of capabilities for that repository
/// - Can detect additional capabilities based on repository contents
///
/// # Example
///
/// ```ignore
/// use hyprstream::archetypes::domain::{Domain, DetectionResult};
/// use hyprstream::archetypes::capabilities::{CapabilitySet, Infer, Train, Serve};
///
/// struct HfModelDomain;
///
/// impl Domain for HfModelDomain {
///     fn name(&self) -> &'static str { "HfModel" }
///     fn description(&self) -> &'static str { "HuggingFace model" }
///
///     fn detect(&self, repo_path: &Path) -> DetectionResult {
///         let config_path = repo_path.join("config.json");
///         if !config_path.exists() {
///             return DetectionResult::NotDetected;
///         }
///
///         // Validate config and build capabilities
///         let mut caps = CapabilitySet::model();
///         if repo_path.join("adapters").is_dir() {
///             caps.insert::<LoraSupport>();
///         }
///
///         DetectionResult::Detected(caps)
///     }
/// }
/// ```
pub trait Domain: Send + Sync + 'static {
    /// Domain identifier for Casbin (e.g., "HfModel")
    ///
    /// This is used in policy rules: `p, user, HfModel, model:*, infer`
    fn name(&self) -> &'static str;

    /// Human-readable description
    fn description(&self) -> &'static str;

    /// Detect if a repository belongs to this domain
    ///
    /// Returns:
    /// - `Detected(caps)` if the repo matches with capabilities
    /// - `NotDetected` if the repo doesn't match (not an error)
    /// - `Error(e)` if detection failed (I/O error, invalid structure)
    fn detect(&self, repo_path: &Path) -> DetectionResult;

    /// Base capabilities this domain always provides when detected
    ///
    /// Override `detect()` for conditional capabilities.
    fn base_capabilities(&self) -> CapabilitySet {
        CapabilitySet::new()
    }
}

/// Extension trait for domains with simple boolean detection
///
/// Implement this trait for domains where detection is a simple boolean check.
/// The blanket implementation will automatically implement `Domain` for you,
/// converting `matches()` to `DetectionResult`.
pub trait SimpleDomain: Send + Sync + 'static {
    /// Domain identifier for Casbin (e.g., "HfModel")
    fn name(&self) -> &'static str;

    /// Human-readable description
    fn description(&self) -> &'static str;

    /// Check if repository matches this domain (simple boolean check)
    fn matches(&self, repo_path: &Path) -> bool;

    /// Base capabilities this domain provides when detected
    fn base_capabilities(&self) -> CapabilitySet {
        CapabilitySet::new()
    }
}

/// Blanket implementation that converts SimpleDomain::matches to DetectionResult
impl<T: SimpleDomain> Domain for T {
    fn name(&self) -> &'static str {
        SimpleDomain::name(self)
    }

    fn description(&self) -> &'static str {
        SimpleDomain::description(self)
    }

    fn detect(&self, repo_path: &Path) -> DetectionResult {
        if self.matches(repo_path) {
            DetectionResult::Detected(SimpleDomain::base_capabilities(self))
        } else {
            DetectionResult::NotDetected
        }
    }

    fn base_capabilities(&self) -> CapabilitySet {
        SimpleDomain::base_capabilities(self)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::archetypes::capabilities::{Infer, Serve, Train};

    #[test]
    fn test_detection_result_is_detected() {
        let detected = DetectionResult::Detected(CapabilitySet::model());
        assert!(detected.is_detected());

        let not_detected = DetectionResult::NotDetected;
        assert!(!not_detected.is_detected());

        let error = DetectionResult::Error(DetectionError::Io("test".into()));
        assert!(!error.is_detected());
    }

    #[test]
    fn test_detection_result_capabilities() {
        let caps = CapabilitySet::model();
        let detected = DetectionResult::Detected(caps.clone());

        let result_caps = detected.capabilities().expect("test: get capabilities");
        assert!(result_caps.has::<Infer>());
        assert!(result_caps.has::<Train>());
        assert!(result_caps.has::<Serve>());
    }

    #[test]
    fn test_detection_result_into_option() {
        let caps = CapabilitySet::model();
        let detected = DetectionResult::Detected(caps);
        assert!(detected.into_option().is_some());

        let not_detected = DetectionResult::NotDetected;
        assert!(not_detected.into_option().is_none());

        let error = DetectionResult::Error(DetectionError::Io("test".into()));
        assert!(error.into_option().is_none());
    }

    #[test]
    fn test_detection_error_display() {
        let io_err = DetectionError::Io("permission denied".into());
        assert!(io_err.to_string().contains("permission denied"));

        let path_err = DetectionError::PathNotFound(PathBuf::from("/missing"));
        assert!(path_err.to_string().contains("/missing"));

        let struct_err = DetectionError::InvalidStructure("missing field".into());
        assert!(struct_err.to_string().contains("missing field"));
    }
}
