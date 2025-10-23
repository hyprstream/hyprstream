//! Git worktree strategy using libgit2
//!
//! This is the default, always-available strategy that uses Git's native
//! worktree feature for branch isolation.

use super::strategy::{StrategyCapabilities, WorktreeHandle, WorktreeMetadata, WorktreeStrategy};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use chrono::Utc;
use git2::{Repository, WorktreeAddOptions};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Git worktree strategy using libgit2
///
/// Creates isolated worktrees for different branches using Git's native
/// worktree feature. Each worktree is a full checkout but shares .git data.
pub struct GitWorktreeStrategy;

impl GitWorktreeStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GitWorktreeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorktreeStrategy for GitWorktreeStrategy {
    async fn create(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        branch: &str,
    ) -> Git2DBResult<Box<dyn WorktreeHandle>> {
        debug!(
            "Creating git worktree: base={}, path={}, branch={}",
            base_repo.display(),
            worktree_path.display(),
            branch
        );

        // Use blocking task for git operations
        let base_repo = base_repo.to_path_buf();
        let worktree_path = worktree_path.to_path_buf();
        let branch = branch.to_string();

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&base_repo)
                .map_err(|e| Git2DBError::internal(format!("Failed to open repository: {}", e)))?;

            // Create worktree
            let mut opts = WorktreeAddOptions::new();
            opts.lock(false); // Don't lock by default

            let _worktree = repo
                .worktree(&branch, &worktree_path, Some(&opts))
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to create git worktree: {}", e))
                })?;

            info!(
                "Created git worktree at {} for branch {}",
                worktree_path.display(),
                branch
            );

            Ok(Box::new(GitWorktreeHandle {
                path: worktree_path,
                branch,
                created_at: Utc::now(),
            }) as Box<dyn WorktreeHandle>)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Join error: {}", e)))?
    }

    fn name(&self) -> &'static str {
        "git-worktree"
    }

    fn is_available(&self) -> bool {
        true // Always available if libgit2 is available
    }

    fn capabilities(&self) -> StrategyCapabilities {
        StrategyCapabilities {
            requires_privileges: false,
            space_efficient: false,    // Full checkout per worktree
            relative_performance: 1.0, // Baseline
            platforms: vec!["linux", "macos", "windows"],
            requirements: Vec::new(),
        }
    }
}

/// Handle to a Git worktree
struct GitWorktreeHandle {
    path: PathBuf,
    branch: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl WorktreeHandle for GitWorktreeHandle {
    fn path(&self) -> &Path {
        &self.path
    }

    fn is_valid(&self) -> bool {
        self.path.exists() && self.path.join(".git").exists()
    }

    fn metadata(&self) -> WorktreeMetadata {
        WorktreeMetadata {
            strategy_name: "git-worktree".to_string(),
            created_at: self.created_at,
            space_saved_bytes: None, // Git worktrees don't save space
            backend_info: Some(format!("branch: {}", self.branch)),
            read_only: false,
        }
    }

    fn cleanup(&self) -> Git2DBResult<()> {
        // Git worktree cleanup is handled by libgit2
        // The worktree directory can be removed normally
        debug!("Git worktree cleanup at {}", self.path.display());
        Ok(())
    }
}

impl Drop for GitWorktreeHandle {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            tracing::warn!("Failed to cleanup git worktree: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_strategy_available() {
        let strategy = GitWorktreeStrategy::new();
        assert!(strategy.is_available());
        assert_eq!(strategy.name(), "git-worktree");
    }

    #[test]
    fn test_git_strategy_capabilities() {
        let strategy = GitWorktreeStrategy::new();
        let caps = strategy.capabilities();

        assert!(!caps.requires_privileges);
        assert!(!caps.space_efficient);
        assert_eq!(caps.relative_performance, 1.0);
        assert!(caps.platforms.contains(&"linux"));
    }
}
