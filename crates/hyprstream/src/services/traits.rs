//! Service trait definitions for registry operations.
//!
//! These traits define the contract for registry client implementations.
//! Implementations can use ZMQ/Cap'n Proto (RegistryZmqClient) or other transports.

use async_trait::async_trait;
use git2db::{Git2DBError, RepoId, RepositoryStatus, TrackedRepository};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

// Re-export RemoteInfo from rpc_types for public API
pub use super::rpc_types::RemoteInfo;

/// Worktree information returned by list_worktrees.
#[derive(Debug, Clone)]
pub struct WorktreeInfo {
    /// Path to the worktree
    pub path: PathBuf,
    /// Branch name (if associated with a branch)
    pub branch: Option<String>,
    /// Storage driver name
    pub driver: String,
    /// Whether the worktree has uncommitted changes
    pub is_dirty: bool,
}

/// Service error type for registry operations.
///
/// Split into:
/// - Registry: Underlying git2db errors
/// - Transport: Service communication errors (ZMQ, Cap'n Proto, etc.)
/// - Unavailable: Service not ready
#[derive(Debug, Error)]
pub enum RegistryServiceError {
    /// Registry operation failed (wraps underlying Git2DBError).
    #[error("Registry operation failed: {0}")]
    Registry(#[from] Git2DBError),

    /// Service communication failed (ZMQ, channel, network, etc.).
    #[error("Service communication failed: {0}")]
    Transport(String),

    /// Service is unavailable (not started, shutdown, etc.).
    #[error("Service unavailable")]
    Unavailable,
}

impl RegistryServiceError {
    /// Create a transport error.
    pub fn transport<S: Into<String>>(msg: S) -> Self {
        Self::Transport(msg.into())
    }
}

/// Transport-agnostic registry client trait.
///
/// Implementations can be:
/// - `RegistryZmqClient` (ZMQ + Cap'n Proto)
/// - Direct (in-process, for testing)
///
/// All methods return owned data to avoid lifetime issues across transports.
#[async_trait]
pub trait RegistryClient: Send + Sync {
    // === Discovery (read-heavy, consider using cached_list for performance) ===

    /// List all tracked repositories.
    ///
    /// Goes through the service channel to get fresh data.
    /// For read-heavy workloads, prefer `cached_list()` when available.
    async fn list(&self) -> Result<Vec<TrackedRepository>, RegistryServiceError>;

    /// Get repository by ID.
    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, RegistryServiceError>;

    /// Get repository by name.
    async fn get_by_name(&self, name: &str)
        -> Result<Option<TrackedRepository>, RegistryServiceError>;

    /// Fast path: get cached list (bypasses channel if available).
    ///
    /// Returns `None` if caching is not supported by this client.
    /// Default implementation returns `None`.
    fn cached_list(&self) -> Option<Vec<TrackedRepository>> {
        None
    }

    // === Convenience Helpers ===

    /// Check if a repository exists by name (sync, cache-only).
    ///
    /// Returns `true` if the repository exists in the cache, `false` otherwise.
    /// If no cache is available, returns `false` (use `exists_async` for reliable check).
    fn exists(&self, name: &str) -> bool {
        if let Some(repos) = self.cached_list() {
            repos.iter().any(|t| t.name.as_ref() == Some(&name.to_string()))
        } else {
            false
        }
    }

    /// Check if a repository exists by name (async, reliable).
    ///
    /// Queries the service if not in cache. Use this when you need a definitive answer.
    async fn exists_async(&self, name: &str) -> bool {
        self.get_by_name(name).await.ok().flatten().is_some()
    }

    // === Mutation (always through channel) ===

    /// Clone a repository from URL.
    ///
    /// # Arguments
    /// * `url` - Repository URL to clone
    /// * `name` - Optional name for the repository (defaults to URL-derived name)
    async fn clone_repo(
        &self,
        url: &str,
        name: Option<&str>,
    ) -> Result<RepoId, RegistryServiceError>;

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
    ) -> Result<(), RegistryServiceError>;

    /// Upsert: update if exists, create if not.
    ///
    /// This is useful for ensuring a repository exists without checking first.
    async fn upsert(&self, name: &str, url: &str) -> Result<RepoId, RegistryServiceError>;

    /// Remove a repository from the registry.
    async fn remove(&self, id: &RepoId) -> Result<(), RegistryServiceError>;

    // === Health ===

    /// Check service health (for testing/monitoring).
    ///
    /// Returns `Ok(())` if the service is healthy, error otherwise.
    async fn health_check(&self) -> Result<(), RegistryServiceError>;

    // === Repository Client Access ===

    /// Get a scoped repository client by name.
    ///
    /// Returns a client that provides repository-level operations (worktrees, branches, etc.).
    async fn repo(&self, name: &str) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError>;

    /// Get a scoped repository client by ID.
    async fn repo_by_id(
        &self,
        id: &RepoId,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError>;
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
    async fn create_worktree(
        &self,
        path: &Path,
        branch: &str,
    ) -> Result<PathBuf, RegistryServiceError>;

    /// List all worktrees.
    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, RegistryServiceError>;

    /// Get the path for a worktree.
    async fn worktree_path(&self, branch: &str) -> Result<Option<PathBuf>, RegistryServiceError>;

    // === Branch Operations ===

    /// Create a new branch.
    async fn create_branch(
        &self,
        name: &str,
        from: Option<&str>,
    ) -> Result<(), RegistryServiceError>;

    /// Checkout a branch or ref.
    async fn checkout(&self, ref_spec: &str) -> Result<(), RegistryServiceError>;

    /// Get the default branch name.
    async fn default_branch(&self) -> Result<String, RegistryServiceError>;

    /// List all branches.
    async fn list_branches(&self) -> Result<Vec<String>, RegistryServiceError>;

    /// Merge a branch or reference into the current HEAD.
    ///
    /// # Arguments
    /// * `source` - The branch or ref to merge (e.g., "feature-branch", "refs/tags/v1.0")
    /// * `message` - Optional commit message (defaults to "Merge branch 'source'")
    ///
    /// Returns the merge commit OID as a string.
    async fn merge(&self, source: &str, message: Option<&str>) -> Result<String, RegistryServiceError>;

    /// Remove a worktree.
    async fn remove_worktree(&self, path: &Path) -> Result<(), RegistryServiceError>;

    // === Staging/Commit Operations ===

    /// Stage all changes in the repository.
    async fn stage_all(&self) -> Result<(), RegistryServiceError>;

    /// Stage specific files.
    async fn stage_files(&self, files: &[&str]) -> Result<(), RegistryServiceError>;

    /// Commit staged changes with a message.
    ///
    /// Returns the commit OID as a string.
    async fn commit(&self, message: &str) -> Result<String, RegistryServiceError>;

    /// Get repository status (branch, dirty files, etc.)
    async fn status(&self) -> Result<RepositoryStatus, RegistryServiceError>;

    // === Reference Operations ===

    /// Get the HEAD commit OID.
    async fn get_head(&self) -> Result<String, RegistryServiceError>;

    /// Get the OID for a named reference.
    async fn get_ref(&self, ref_name: &str) -> Result<String, RegistryServiceError>;

    /// Update repository from remote (fetch).
    ///
    /// # Arguments
    /// * `refspec` - Optional refspec to fetch (e.g., "refs/heads/main")
    async fn update(&self, refspec: Option<&str>) -> Result<(), RegistryServiceError>;

    // === Remote Operations ===

    /// List all remotes.
    async fn list_remotes(&self) -> Result<Vec<RemoteInfo>, RegistryServiceError>;

    /// Add a new remote.
    async fn add_remote(&self, name: &str, url: &str) -> Result<(), RegistryServiceError>;

    /// Remove a remote.
    async fn remove_remote(&self, name: &str) -> Result<(), RegistryServiceError>;

    /// Change a remote's URL.
    async fn set_remote_url(&self, name: &str, url: &str) -> Result<(), RegistryServiceError>;

    /// Rename a remote.
    async fn rename_remote(
        &self,
        old_name: &str,
        new_name: &str,
    ) -> Result<(), RegistryServiceError>;
}

