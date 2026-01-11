//! Unified error handling for git2db operations
//!
//! This module consolidates error handling patterns from the original codebase
//! and provides enhanced error classification using libgit2's error types.

use git2::{ErrorClass, ErrorCode};
use std::path::PathBuf;
use thiserror::Error;

/// Result type for git2db operations
pub type Git2DBResult<T> = std::result::Result<T, Git2DBError>;

/// LFS-specific error kinds for detailed error classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfsErrorKind {
    /// LFS pointer content is invalid
    InvalidPointer,
    /// Failed to parse LFS pointer fields
    ParseError,
    /// Failed to convert hash formats
    HashConversion,
    /// File size doesn't match pointer metadata
    SizeMismatch,
    /// SHA256 hash doesn't match pointer metadata
    HashMismatch,
    /// Failed to smudge (download) content
    SmudgeFailed,
    /// I/O operation failed
    IoError,
    /// Directory is not part of a git repository
    NotInRepository,
}

/// Comprehensive error types for git2db operations
#[derive(Error, Debug)]
pub enum Git2DBError {
    /// Git operation errors with enhanced classification
    #[error("Git operation failed: {message} (class: {class:?}, code: {code:?})")]
    GitOperation {
        class: ErrorClass,
        code: ErrorCode,
        message: String,
        recoverable: bool,
    },

    /// Repository access errors
    #[error("Repository error at {path:?}: {message}")]
    Repository { path: PathBuf, message: String },

    /// Submodule operation errors
    #[error("Submodule operation failed for '{name}': {message}")]
    Submodule { name: String, message: String },

    /// Reference resolution errors
    #[error("Reference resolution failed for '{reference}': {message}")]
    Reference { reference: String, message: String },

    /// Authentication errors
    #[error("Authentication failed for {url}: {message}")]
    Authentication { url: String, message: String },

    /// Network operation errors
    #[error("Network operation failed: {message}")]
    Network { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Path validation errors
    #[error("Invalid path: {path:?} - {message}")]
    InvalidPath { path: PathBuf, message: String },

    /// Repository validation errors
    #[error("Invalid repository '{name}': {message}")]
    InvalidRepository { name: String, message: String },

    /// Worktree already exists at path
    #[error("Worktree already exists at {path:?}. Use 'remove' to delete it first.")]
    WorktreeExists { path: PathBuf },

    /// Operation cancelled by user
    #[error("Operation cancelled: {operation}")]
    Cancelled { operation: String },

    /// Invalid operation (e.g., nothing to commit, empty merge)
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    /// Internal library errors
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// LFS operation errors
    #[error("LFS error ({kind:?}): {message}")]
    Lfs { kind: LfsErrorKind, message: String },

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic errors from dependencies
    #[error("External error: {0}")]
    External(#[from] anyhow::Error),
}

impl Git2DBError {
    /// Get the error message regardless of variant
    pub fn message(&self) -> String {
        match self {
            Self::GitOperation { message, .. } => message.clone(),
            Self::Repository { message, .. } => message.clone(),
            Self::Submodule { message, .. } => message.clone(),
            Self::Reference { message, .. } => message.clone(),
            Self::Authentication { message, .. } => message.clone(),
            Self::Network { message, .. } => message.clone(),
            Self::Configuration { message, .. } => message.clone(),
            Self::InvalidPath { message, .. } => message.clone(),
            Self::InvalidRepository { message, .. } => message.clone(),
            Self::WorktreeExists { path } => {
                format!("Worktree already exists at {}", path.display())
            }
            Self::Cancelled { operation } => format!("Operation cancelled: {}", operation),
            Self::InvalidOperation { message } => message.clone(),
            Self::Internal { message } => message.clone(),
            Self::Lfs { message, .. } => message.clone(),
            Self::Io(e) => e.to_string(),
            Self::Json(e) => e.to_string(),
            Self::External(e) => e.to_string(),
        }
    }

    /// Get error class if this is a GitOperation error
    pub fn class(&self) -> Option<ErrorClass> {
        match self {
            Self::GitOperation { class, .. } => Some(*class),
            _ => None,
        }
    }

    /// Get error code if this is a GitOperation error
    pub fn code(&self) -> Option<ErrorCode> {
        match self {
            Self::GitOperation { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// Create a git operation error from git2::Error
    pub fn from_git_error(err: git2::Error) -> Self {
        let class = err.class();
        let code = err.code();
        let message = err.message().to_string();

        let recoverable = matches!(class, ErrorClass::Net | ErrorClass::Http | ErrorClass::Ssh);

        Self::GitOperation {
            class,
            code,
            message,
            recoverable,
        }
    }

    /// Create a repository error
    pub fn repository<P: Into<PathBuf>, S: Into<String>>(path: P, message: S) -> Self {
        Self::Repository {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Create a submodule error
    pub fn submodule<N: Into<String>, M: Into<String>>(name: N, message: M) -> Self {
        Self::Submodule {
            name: name.into(),
            message: message.into(),
        }
    }

    /// Create a reference error
    pub fn reference<R: Into<String>, M: Into<String>>(reference: R, message: M) -> Self {
        Self::Reference {
            reference: reference.into(),
            message: message.into(),
        }
    }

    /// Create an authentication error
    pub fn authentication<U: Into<String>, M: Into<String>>(url: U, message: M) -> Self {
        Self::Authentication {
            url: url.into(),
            message: message.into(),
        }
    }

    /// Create a network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create an invalid path error
    pub fn invalid_path<P: Into<PathBuf>, M: Into<String>>(path: P, message: M) -> Self {
        Self::InvalidPath {
            path: path.into(),
            message: message.into(),
        }
    }

    /// Create an invalid repository error
    pub fn invalid_repository<N: Into<String>, M: Into<String>>(name: N, message: M) -> Self {
        Self::InvalidRepository {
            name: name.into(),
            message: message.into(),
        }
    }

    /// Create a worktree exists error
    pub fn worktree_exists<P: Into<PathBuf>>(path: P) -> Self {
        Self::WorktreeExists { path: path.into() }
    }

    /// Check if this error is a worktree exists error
    pub fn is_worktree_exists(&self) -> bool {
        matches!(self, Self::WorktreeExists { .. })
    }

    /// Create a cancelled operation error
    pub fn cancelled<S: Into<String>>(operation: S) -> Self {
        Self::Cancelled {
            operation: operation.into(),
        }
    }

    /// Create an invalid operation error
    pub fn invalid_operation<S: Into<String>>(message: S) -> Self {
        Self::InvalidOperation {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create an LFS error with specific kind
    pub fn lfs<S: Into<String>>(kind: LfsErrorKind, message: S) -> Self {
        Self::Lfs {
            kind,
            message: message.into(),
        }
    }

    /// Get LFS error kind if this is an LFS error
    pub fn lfs_kind(&self) -> Option<LfsErrorKind> {
        match self {
            Self::Lfs { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    /// Create a merge conflict error
    pub fn merge_conflict<S: Into<String>>(message: S) -> Self {
        Self::GitOperation {
            class: ErrorClass::Merge,
            code: ErrorCode::Conflict,
            message: message.into(),
            recoverable: true,
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::GitOperation { recoverable, .. } => *recoverable,
            Self::Network { .. } => true,
            Self::Authentication { .. } => true,
            Self::Cancelled { .. } => false,
            Self::Configuration { .. } => false,
            Self::InvalidPath { .. } => false,
            Self::InvalidRepository { .. } => false,
            Self::WorktreeExists { .. } => false,
            Self::InvalidOperation { .. } => false,
            _ => false,
        }
    }

}

impl From<git2::Error> for Git2DBError {
    fn from(err: git2::Error) -> Self {
        Self::from_git_error(err)
    }
}

/// Helper macro for creating repository errors
#[macro_export]
macro_rules! repo_error {
    ($path:expr, $msg:expr) => {
        $crate::errors::Git2DBError::repository($path, $msg)
    };
    ($path:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::errors::Git2DBError::repository($path, format!($fmt, $($arg)*))
    };
}

/// Helper macro for creating submodule errors
#[macro_export]
macro_rules! submodule_error {
    ($name:expr, $msg:expr) => {
        $crate::errors::Git2DBError::submodule($name, $msg)
    };
    ($name:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::errors::Git2DBError::submodule($name, format!($fmt, $($arg)*))
    };
}

/// Helper macro for creating reference errors
#[macro_export]
macro_rules! ref_error {
    ($reference:expr, $msg:expr) => {
        $crate::errors::Git2DBError::reference($reference, $msg)
    };
    ($reference:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::errors::Git2DBError::reference($reference, format!($fmt, $($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_operation_error_classification() {
        // Test network error classification
        let net_error =
            git2::Error::new(ErrorCode::GenericError, ErrorClass::Net, "Network timeout");
        let err = Git2DBError::from_git_error(net_error);

        if let Git2DBError::GitOperation {
            class,
            code,
            recoverable,
            ..
        } = err
        {
            assert_eq!(class, ErrorClass::Net);
            assert_eq!(code, ErrorCode::GenericError);
            assert!(recoverable);
        } else {
            panic!("Expected GitOperation error");
        }
    }

    #[test]
    fn test_git_operation_non_recoverable_errors() {
        // Test repository error classification
        let repo_error = git2::Error::new(
            ErrorCode::NotFound,
            ErrorClass::Repository,
            "Repository not found",
        );
        let err = Git2DBError::from_git_error(repo_error);

        if let Git2DBError::GitOperation {
            class,
            code,
            recoverable,
            ..
        } = err
        {
            assert_eq!(class, ErrorClass::Repository);
            assert_eq!(code, ErrorCode::NotFound);
            assert!(!recoverable);
        } else {
            panic!("Expected GitOperation error");
        }
    }

    #[test]
    fn test_repository_error_creation() {
        let path = PathBuf::from("/test/path");
        let err = Git2DBError::repository(&path, "Test error message");

        if let Git2DBError::Repository {
            path: ref err_path,
            ref message,
        } = err
        {
            assert_eq!(err_path, &path);
            assert_eq!(message, "Test error message");
        } else {
            panic!("Expected Repository error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_submodule_error_creation() {
        let err = Git2DBError::submodule("test-module", "Failed to initialize");

        if let Git2DBError::Submodule {
            ref name,
            ref message,
        } = err
        {
            assert_eq!(name, "test-module");
            assert_eq!(message, "Failed to initialize");
        } else {
            panic!("Expected Submodule error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_reference_error_creation() {
        let err = Git2DBError::reference("refs/heads/main", "Reference not found");

        if let Git2DBError::Reference { reference, message } = err {
            assert_eq!(reference, "refs/heads/main");
            assert_eq!(message, "Reference not found");
        } else {
            panic!("Expected Reference error");
        }
    }

    #[test]
    fn test_authentication_error_creation() {
        let err =
            Git2DBError::authentication("https://github.com/test/repo.git", "Invalid credentials");

        if let Git2DBError::Authentication {
            ref url,
            ref message,
        } = err
        {
            assert_eq!(url, "https://github.com/test/repo.git");
            assert_eq!(message, "Invalid credentials");
        } else {
            panic!("Expected Authentication error");
        }

        assert!(err.is_recoverable());
    }

    #[test]
    fn test_network_error_creation() {
        let err = Git2DBError::network("Connection timeout");

        if let Git2DBError::Network { ref message } = err {
            assert_eq!(message, "Connection timeout");
        } else {
            panic!("Expected Network error");
        }

        assert!(err.is_recoverable());
    }

    #[test]
    fn test_configuration_error_creation() {
        let err = Git2DBError::configuration("Invalid signature configuration");

        if let Git2DBError::Configuration { ref message } = err {
            assert_eq!(message, "Invalid signature configuration");
        } else {
            panic!("Expected Configuration error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_invalid_path_error_creation() {
        let path = PathBuf::from("../../../etc/passwd");
        let err = Git2DBError::invalid_path(&path, "Path traversal detected");

        if let Git2DBError::InvalidPath {
            path: ref err_path,
            ref message,
        } = err
        {
            assert_eq!(err_path, &path);
            assert_eq!(message, "Path traversal detected");
        } else {
            panic!("Expected InvalidPath error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_invalid_repository_error_creation() {
        let err = Git2DBError::invalid_repository("name with spaces", "Invalid characters in name");

        if let Git2DBError::InvalidRepository {
            ref name,
            ref message,
        } = err
        {
            assert_eq!(name, "name with spaces");
            assert_eq!(message, "Invalid characters in name");
        } else {
            panic!("Expected InvalidRepository error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_cancelled_error_creation() {
        let err = Git2DBError::cancelled("clone operation");

        if let Git2DBError::Cancelled { ref operation } = err {
            assert_eq!(operation, "clone operation");
        } else {
            panic!("Expected Cancelled error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_internal_error_creation() {
        let err = Git2DBError::internal("Unexpected null pointer");

        if let Git2DBError::Internal { ref message } = err {
            assert_eq!(message, "Unexpected null pointer");
        } else {
            panic!("Expected Internal error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_display_formatting() {
        let err = Git2DBError::repository("/test/path", "Test message");
        let display_str = format!("{}", err);
        assert!(display_str.contains("/test/path"));
        assert!(display_str.contains("Test message"));

        let git_err = Git2DBError::from_git_error(git2::Error::new(
            ErrorCode::NotFound,
            ErrorClass::Repository,
            "Not found",
        ));
        let git_display = format!("{}", git_err);
        assert!(git_display.contains("Git operation failed"));
        assert!(git_display.contains("Not found"));
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let err: Git2DBError = io_err.into();

        if let Git2DBError::Io(_) = err {
            // Expected
        } else {
            panic!("Expected Io error");
        }

        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let err: Git2DBError = json_err.into();

        if let Git2DBError::Json(_) = err {
            // Expected
        } else {
            panic!("Expected Json error");
        }
    }

    #[test]
    fn test_error_from_anyhow_error() {
        let anyhow_err = anyhow::anyhow!("Some external error");
        let err: Git2DBError = anyhow_err.into();

        if let Git2DBError::External(_) = err {
            // Expected
        } else {
            panic!("Expected External error");
        }
    }

    #[test]
    fn test_error_macros() {
        let repo_err = repo_error!("/test/path", "Repository error");
        if let Git2DBError::Repository {
            ref path,
            ref message,
        } = repo_err
        {
            assert_eq!(path, &PathBuf::from("/test/path"));
            assert_eq!(message, "Repository error");
        } else {
            panic!("Expected Repository error from macro");
        }

        let formatted_repo_err = repo_error!("/test/path", "Error code: {}", 404);
        if let Git2DBError::Repository { ref message, .. } = formatted_repo_err {
            assert_eq!(message, "Error code: 404");
        } else {
            panic!("Expected Repository error from macro with formatting");
        }

        let submodule_err = submodule_error!("test-module", "Submodule error");
        if let Git2DBError::Submodule {
            ref name,
            ref message,
        } = submodule_err
        {
            assert_eq!(name, "test-module");
            assert_eq!(message, "Submodule error");
        } else {
            panic!("Expected Submodule error from macro");
        }

        let ref_err = ref_error!("refs/heads/main", "Reference error");
        if let Git2DBError::Reference {
            ref reference,
            ref message,
        } = ref_err
        {
            assert_eq!(reference, "refs/heads/main");
            assert_eq!(message, "Reference error");
        } else {
            panic!("Expected Reference error from macro");
        }
    }

    #[test]
    fn test_comprehensive_recoverable_classification() {
        let test_cases = vec![
            (Git2DBError::network("test"), true),
            (Git2DBError::authentication("url", "test"), true),
            (Git2DBError::repository("/test", "test"), false),
            (Git2DBError::configuration("test"), false),
            (Git2DBError::invalid_path("/test", "test"), false),
            (Git2DBError::invalid_repository("test", "test"), false),
            (Git2DBError::cancelled("test"), false),
            (Git2DBError::internal("test"), false),
        ];

        for (error, expected_recoverable) in test_cases {
            assert_eq!(
                error.is_recoverable(),
                expected_recoverable,
                "Recoverable classification failed for: {:?}",
                error
            );
        }
    }

}
