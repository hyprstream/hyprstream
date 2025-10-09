//! Custom error types for the storage layer
//!
//! This module provides type-safe error handling patterns using thiserror,
//! improving error diagnostics and enabling better error recovery strategies.

use thiserror::Error;
use std::path::PathBuf;

/// Comprehensive error types for storage operations
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Model '{0}' not found")]
    ModelNotFound(String),

    #[error("Model '{name}' not found at path: {path:?}")]
    ModelNotFoundAtPath { name: String, path: PathBuf },

    #[error("Invalid model name: {0}")]
    InvalidModelName(String),

    #[error("Model name '{0}' is reserved")]
    ReservedModelName(String),

    #[error("Model '{0}' already exists")]
    ModelAlreadyExists(String),

    #[error("Git operation failed: {0}")]
    Git(#[from] git2::Error),

    #[error("Git reference '{ref_name}' is invalid: {reason}")]
    InvalidGitRef { ref_name: String, reason: String },

    #[error("Git reference '{0}' not found")]
    GitRefNotFound(String),

    #[error("Repository at '{path:?}' is not initialized")]
    RepositoryNotInitialized { path: PathBuf },

    #[error("Repository at '{path:?}' is in an inconsistent state: {reason}")]
    RepositoryInconsistent { path: PathBuf, reason: String },

    #[error("IO operation failed")]
    Io(#[from] std::io::Error),

    #[error("Failed to access file at '{path:?}': {source}")]
    FileAccess {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Serialization failed")]
    Serialization(#[from] serde_json::Error),

    #[error("Registry operation failed: {0}")]
    Registry(String),

    #[error("Registry is locked and cannot be accessed")]
    RegistryLocked,

    #[error("Registry at '{path:?}' is corrupted: {reason}")]
    RegistryCorrupted { path: PathBuf, reason: String },

    #[error("Adapter '{0}' not found")]
    AdapterNotFound(String),

    #[error("Adapter '{name}' already exists at index {index}")]
    AdapterAlreadyExists { name: String, index: u32 },

    #[error("Invalid adapter configuration: {0}")]
    InvalidAdapterConfig(String),

    #[error("XET storage operation failed: {0}")]
    XetStorage(String),

    #[error("Network operation failed: {0}")]
    Network(String),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Permission denied for operation: {0}")]
    PermissionDenied(String),

    #[error("Operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Concurrent modification detected for model '{0}'")]
    ConcurrentModification(String),

    #[error("Path validation failed: {path:?} - {reason}")]
    PathValidation { path: PathBuf, reason: String },
}

/// Error types specific to model references
#[derive(Debug, Error)]
pub enum ModelRefError {
    #[error("Invalid model reference format: '{0}'")]
    InvalidFormat(String),

    #[error("Model name is required")]
    MissingModelName,

    #[error("Git reference parsing failed: {0}")]
    GitRefParsing(String),

    #[error("UUID parsing failed: {0}")]
    UuidParsing(#[from] uuid::Error),
}

/// Error types specific to Git operations
#[derive(Debug, Error)]
pub enum GitOperationError {
    #[error("Checkout failed for ref '{ref_name}': {reason}")]
    CheckoutFailed { ref_name: String, reason: String },

    #[error("Merge conflicts detected in '{path:?}'")]
    MergeConflicts { path: PathBuf },

    #[error("Branch '{0}' does not exist")]
    BranchNotFound(String),

    #[error("Tag '{0}' does not exist")]
    TagNotFound(String),

    #[error("Commit '{0}' does not exist")]
    CommitNotFound(String),

    #[error("Repository has uncommitted changes")]
    UncommittedChanges,

    #[error("Remote '{0}' is not configured")]
    RemoteNotConfigured(String),

    #[error("Push rejected by remote: {0}")]
    PushRejected(String),

    #[error("Pull failed: {0}")]
    PullFailed(String),

    #[error("Submodule operation failed: {0}")]
    SubmoduleFailed(String),
}

/// Results using our custom error types
pub type StorageResult<T> = Result<T, StorageError>;
pub type ModelRefResult<T> = Result<T, ModelRefError>;
pub type GitOperationResult<T> = Result<T, GitOperationError>;

impl StorageError {
    /// Check if the error is recoverable (can be retried)
    pub fn is_recoverable(&self) -> bool {
        match self {
            StorageError::Network(_)
                | StorageError::Timeout { .. }
                | StorageError::RegistryLocked
                | StorageError::ConcurrentModification(_) => true,
            StorageError::Git(git_err) => is_recoverable_git_error(git_err),
            _ => false,
        }
    }

    /// Get suggested retry delay for recoverable errors
    pub fn retry_delay(&self) -> Option<std::time::Duration> {
        match self {
            StorageError::Network(_) => Some(std::time::Duration::from_secs(1)),
            StorageError::Timeout { .. } => Some(std::time::Duration::from_secs(2)),
            StorageError::RegistryLocked => Some(std::time::Duration::from_millis(100)),
            StorageError::ConcurrentModification(_) => Some(std::time::Duration::from_millis(500)),
            _ => None,
        }
    }

    /// Create a model not found error with path context
    pub fn model_not_found_at_path(name: String, path: PathBuf) -> Self {
        Self::ModelNotFoundAtPath { name, path }
    }

    /// Create a file access error with path context
    pub fn file_access(path: PathBuf, source: std::io::Error) -> Self {
        Self::FileAccess { path, source }
    }

    /// Create a registry corrupted error
    pub fn registry_corrupted(path: PathBuf, reason: String) -> Self {
        Self::RegistryCorrupted { path, reason }
    }
}

impl GitOperationError {
    /// Create a checkout failed error
    pub fn checkout_failed(ref_name: String, reason: String) -> Self {
        Self::CheckoutFailed { ref_name, reason }
    }

    /// Create a merge conflicts error
    pub fn merge_conflicts(path: PathBuf) -> Self {
        Self::MergeConflicts { path }
    }
}

/// Helper function to determine if a git2::Error is recoverable
fn is_recoverable_git_error(git_err: &git2::Error) -> bool {
    matches!(
        git_err.code(),
        git2::ErrorCode::GenericError
            | git2::ErrorCode::NotFound
            | git2::ErrorCode::Locked
    )
}

/// Convenience macro for creating storage errors
#[macro_export]
macro_rules! storage_error {
    (ModelNotFound, $name:expr) => {
        $crate::storage::errors::StorageError::ModelNotFound($name.to_string())
    };
    (InvalidModelName, $name:expr) => {
        $crate::storage::errors::StorageError::InvalidModelName($name.to_string())
    };
    (Registry, $msg:expr) => {
        $crate::storage::errors::StorageError::Registry($msg.to_string())
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recovery_classification() {
        let network_err = StorageError::Network("Connection timeout".to_string());
        assert!(network_err.is_recoverable());
        assert!(network_err.retry_delay().is_some());

        let not_found_err = StorageError::ModelNotFound("test".to_string());
        assert!(!not_found_err.is_recoverable());
        assert!(not_found_err.retry_delay().is_none());
    }

    #[test]
    fn test_error_context() {
        let path = PathBuf::from("/test/path");
        let err = StorageError::model_not_found_at_path("test_model".to_string(), path.clone());

        assert_eq!(err.to_string(), "Model 'test_model' not found at path: \"/test/path\"");
    }
}