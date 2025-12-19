//! Registry client trait and error types.

use crate::{Git2DBError, RepoId, TrackedRepository};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// Worktree information returned by list_worktrees.
#[derive(Debug, Clone)]
pub struct WorktreeInfo {
    /// Path to the worktree
    pub path: PathBuf,
    /// Branch name (if associated with a branch)
    pub branch: Option<String>,
    /// Storage driver name
    pub driver: String,
}

/// Remote information returned by list_remotes.
#[derive(Debug, Clone)]
pub struct RemoteInfo {
    /// Remote name (e.g., "origin", "upstream")
    pub name: String,
    /// Fetch URL
    pub url: String,
    /// Push URL (if different from fetch URL)
    pub push_url: Option<String>,
}

/// Transport-agnostic registry client trait.
///
/// Implementations can be in-process (channels) or remote (gRPC, etc.).
/// All methods return owned data to avoid lifetime issues across transports.
#[async_trait]
pub trait RegistryClient: Send + Sync {
    // === Discovery (read-heavy, consider using cached_list for performance) ===

    /// List all tracked repositories.
    ///
    /// Goes through the service channel to get fresh data.
    /// For read-heavy workloads, prefer `cached_list()` when available.
    async fn list(&self) -> Result<Vec<TrackedRepository>, ServiceError>;

    /// Get repository by ID.
    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, ServiceError>;

    /// Get repository by name.
    async fn get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>, ServiceError>;

    /// Fast path: get cached list (bypasses channel if available).
    ///
    /// Returns `None` if caching is not supported by this client.
    /// Default implementation returns `None`.
    fn cached_list(&self) -> Option<Vec<TrackedRepository>> {
        None
    }

    // === Mutation (always through channel) ===

    /// Clone a repository from URL.
    ///
    /// # Arguments
    /// * `url` - Repository URL to clone
    /// * `name` - Optional name for the repository (defaults to URL-derived name)
    async fn clone_repo(&self, url: &str, name: Option<&str>) -> Result<RepoId, ServiceError>;

    /// Register an existing repository.
    ///
    /// # Arguments
    /// * `id` - Repository ID to assign
    /// * `name` - Optional name for the repository
    /// * `path` - Path to the existing repository
    async fn register(
        &self,
        id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), ServiceError>;

    /// Upsert: update if exists, create if not.
    ///
    /// This is useful for ensuring a repository exists without checking first.
    async fn upsert(&self, name: &str, url: &str) -> Result<RepoId, ServiceError>;

    /// Remove a repository from the registry.
    async fn remove(&self, id: &RepoId) -> Result<(), ServiceError>;

    // === Health ===

    /// Check service health (for testing/monitoring).
    ///
    /// Returns `Ok(())` if the service is healthy, error otherwise.
    async fn health_check(&self) -> Result<(), ServiceError>;

    // === Repository Client Access ===

    /// Get a scoped repository client by name.
    ///
    /// Returns a client that provides repository-level operations (worktrees, branches, etc.).
    async fn repo(&self, name: &str) -> Result<Arc<dyn RepositoryClient>, ServiceError>;

    /// Get a scoped repository client by ID.
    async fn repo_by_id(&self, id: &RepoId) -> Result<Arc<dyn RepositoryClient>, ServiceError>;
}

/// Scoped repository operations (mirrors RepositoryHandle).
///
/// This trait provides repository-level operations that work over the service layer.
/// Implementations send requests to the service, which uses RepositoryHandle internally.
#[async_trait]
pub trait RepositoryClient: Send + Sync {
    /// Repository name
    fn name(&self) -> &str;

    /// Repository ID
    fn id(&self) -> &RepoId;

    // === Worktree Operations ===

    /// Create a new worktree for a branch.
    ///
    /// This properly handles LFS/XET file smudging via the service.
    async fn create_worktree(&self, path: &Path, branch: &str) -> Result<PathBuf, ServiceError>;

    /// List all worktrees.
    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, ServiceError>;

    /// Get the path for a worktree.
    async fn worktree_path(&self, branch: &str) -> Result<Option<PathBuf>, ServiceError>;

    // === Branch Operations ===

    /// Create a new branch.
    async fn create_branch(&self, name: &str, from: Option<&str>) -> Result<(), ServiceError>;

    /// Checkout a branch or ref.
    async fn checkout(&self, ref_spec: &str) -> Result<(), ServiceError>;

    /// Get the default branch name.
    async fn default_branch(&self) -> Result<String, ServiceError>;

    /// List all branches.
    async fn list_branches(&self) -> Result<Vec<String>, ServiceError>;

    // === Remote Operations ===

    /// List all remotes.
    async fn list_remotes(&self) -> Result<Vec<RemoteInfo>, ServiceError>;

    /// Add a new remote.
    async fn add_remote(&self, name: &str, url: &str) -> Result<(), ServiceError>;

    /// Remove a remote.
    async fn remove_remote(&self, name: &str) -> Result<(), ServiceError>;

    /// Change a remote's URL.
    async fn set_remote_url(&self, name: &str, url: &str) -> Result<(), ServiceError>;

    /// Rename a remote.
    async fn rename_remote(&self, old_name: &str, new_name: &str) -> Result<(), ServiceError>;
}

/// Service error type wrapping Git2DBError with service-layer concerns.
#[derive(Debug, Error)]
pub enum ServiceError {
    /// Registry operation failed (wraps underlying Git2DBError).
    #[error("Registry operation failed: {0}")]
    Registry(#[from] Git2DBError),

    /// Service communication failed (channel closed, send failed, etc.).
    #[error("Service communication failed: {0}")]
    Channel(String),

    /// Service is unavailable (not started, shutdown, etc.).
    #[error("Service unavailable")]
    Unavailable,
}

impl ServiceError {
    /// Check if this error suggests retrying the operation.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Registry(e) => e.should_retry(),
            Self::Channel(_) => true,
            Self::Unavailable => true,
        }
    }

    /// Create a channel error.
    pub fn channel<S: Into<String>>(msg: S) -> Self {
        Self::Channel(msg.into())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_error_is_retryable() {
        // Channel errors should be retryable
        let err = ServiceError::channel("test");
        assert!(err.is_retryable());

        // Unavailable should be retryable
        let err = ServiceError::Unavailable;
        assert!(err.is_retryable());
    }
}
