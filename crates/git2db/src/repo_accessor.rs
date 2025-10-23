//! Shared repository access trait for managers
//!
//! Provides common functionality for managers that need to access repositories
//! tracked by the registry.

use crate::errors::{Git2DBError, Git2DBResult};
use crate::registry::{Git2DB, RepoId};
use git2::Repository;
use std::path::PathBuf;

/// Trait for types that need to access repositories through the registry
///
/// This trait provides default implementations for common operations like
/// getting the repository path and opening the repository.
pub trait RepositoryAccessor {
    /// Get the registry reference
    fn registry(&self) -> &Git2DB;

    /// Get the repository ID
    fn repo_id(&self) -> &RepoId;

    /// Get the repository worktree path
    fn repo_path(&self) -> Git2DBResult<PathBuf> {
        self.registry()
            .get_worktree_path(self.repo_id())
            .ok_or_else(|| {
                Git2DBError::invalid_repository(
                    self.repo_id().to_string(),
                    "Repository not found in registry",
                )
            })
    }

    /// Open the git repository (async, uses spawn_blocking)
    ///
    /// Opens a fresh Repository instance. Note that git2::Repository is not
    /// thread-safe, so each call creates a new instance.
    ///
    /// This method uses `spawn_blocking` to avoid blocking the async executor,
    /// following the pattern from CLAUDE.md for libgit2 operations.
    ///
    /// Returns an `impl Future + Send` to explicitly ensure Send-safety.
    fn open_repo(&self) -> impl std::future::Future<Output = Git2DBResult<Repository>> + Send + '_
    where
        Self: Sync,
    {
        async move {
            let path = self.repo_path()?;

            tokio::task::spawn_blocking(move || {
                Repository::open(&path).map_err(|e| {
                    Git2DBError::repository(&path, format!("Failed to open repository: {}", e))
                })
            })
            .await
            .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
        }
    }
}
