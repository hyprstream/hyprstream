//! Filesystem error types.

/// Filesystem containment error type.
///
/// Covers filesystem-level errors only. Protocol errors (`BadFd`, `Transport`,
/// `Unavailable`) belong in the service layer (e.g., `RegistryService`).
#[derive(Debug, thiserror::Error)]
pub enum FsError {
    /// Path or file not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Permission denied.
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Path escaped containment root (symlink or traversal attack).
    #[error("Path containment violation: {0}")]
    PathEscape(String),

    /// Target already exists (e.g., `create` with `OEXCL`, `mkdir` on existing).
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Resource limit exceeded.
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),
}

impl FsError {
    /// Create a `PathEscape` error from a message.
    pub fn path_escape(msg: impl Into<String>) -> Self {
        Self::PathEscape(msg.into())
    }

    /// Create a `NotFound` error from a message.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
}
