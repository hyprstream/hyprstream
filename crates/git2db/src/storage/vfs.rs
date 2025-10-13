//! VFS driver (plain directories)
//!
//! The vfs driver is Docker's universal fallback - it uses plain directories
//! with no optimization. This driver always works on all platforms.
//!
//! In our case, it creates standard git worktrees using libgit2 without
//! any storage optimization layers underneath.

use super::driver::{Driver, DriverCapabilities, DriverOpts, WorktreeHandle};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::Path;
use tracing::info;

/// Configuration for vfs driver
#[derive(Debug, Clone, Default)]
pub struct VfsConfig {
    // VFS has no special configuration - it's just plain git worktrees
}

/// VFS storage driver (plain git worktree, no optimization)
///
/// This is the universal fallback that always works. It creates standard
/// git worktrees using libgit2 without any CoW or space-saving features.
pub struct VfsDriver;

#[async_trait]
impl Driver for VfsDriver {
    fn name(&self) -> &'static str {
        "vfs"
    }

    fn is_available(&self) -> bool {
        // VFS is always available (it's just plain directories)
        true
    }

    fn capabilities(&self) -> DriverCapabilities {
        DriverCapabilities {
            copy_on_write: false,
            space_savings_percent: 0,
            requires_privileges: false,
            platforms: vec!["linux", "macos", "windows"], // All platforms
            required_binaries: vec![],
            relative_performance: 1.0, // Baseline performance
        }
    }

    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        // Validate inputs
        if !opts.base_repo.exists() {
            return Err(Git2DBError::invalid_path(
                opts.base_repo.clone(),
                "Base repository does not exist",
            ));
        }

        // Ensure parent directory exists
        if let Some(parent) = opts.worktree_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to create parent directory: {}", e))
                })?;
        }

        info!(
            "Creating vfs worktree: base={}, path={}, ref={}",
            opts.base_repo.display(),
            opts.worktree_path.display(),
            opts.ref_spec
        );

        // Create git worktree using libgit2
        self.create_git_worktree(opts).await?;

        // Return handle (no special cleanup needed for plain directories)
        Ok(WorktreeHandle::new(
            opts.worktree_path.clone(),
            "vfs".to_string(),
        ))
    }
}

impl VfsDriver {
    /// Create git worktree using libgit2 with unified ref support
    ///
    /// Supports any git ref: branches, commits, tags, symbolic refs (HEAD~3), etc.
    async fn create_git_worktree(&self, opts: &DriverOpts) -> Git2DBResult<()> {
        // Open the base repository
        let repo = git2::Repository::open(&opts.base_repo).map_err(|e| {
            Git2DBError::internal(format!("Failed to open repository: {}", e))
        })?;

        // Resolve ref_spec to a commit using git_revparse_single
        let object = repo.revparse_single(&opts.ref_spec).map_err(|e| {
            Git2DBError::internal(format!("Failed to resolve ref '{}': {}", opts.ref_spec, e))
        })?;

        let commit = object.peel_to_commit().map_err(|e| {
            Git2DBError::internal(format!("Ref '{}' does not point to a commit: {}", opts.ref_spec, e))
        })?;

        // Check if this is a branch (for branch tracking)
        let branch_ref_name = format!("refs/heads/{}", opts.ref_spec);
        let is_branch = repo.find_reference(&branch_ref_name).is_ok();

        // Create worktree name
        let worktree_name = opts
            .worktree_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                Git2DBError::invalid_path(
                    opts.worktree_path.clone(),
                    "Invalid worktree path",
                )
            })?;

        // Create worktree appropriately based on ref type
        if is_branch {
            // Branch: Create with branch tracking
            let reference = repo.find_reference(&branch_ref_name)?;
            repo.worktree(
                worktree_name,
                &opts.worktree_path,
                Some(
                    git2::WorktreeAddOptions::new()
                        .reference(Some(&reference)),
                ),
            )
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to create worktree: {}", e))
            })?;

            info!(
                "Created git worktree at {} for branch '{}' (commit: {})",
                opts.worktree_path.display(),
                opts.ref_spec,
                commit.id()
            );
        } else {
            // Commit/Tag/Symbolic: Create with detached HEAD
            repo.worktree(worktree_name, &opts.worktree_path, None)
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to create worktree: {}", e))
                })?;

            // Set detached HEAD to the resolved commit
            let wt_repo = git2::Repository::open(&opts.worktree_path)?;
            wt_repo.set_head_detached(commit.id()).map_err(|e| {
                Git2DBError::internal(format!("Failed to set detached HEAD: {}", e))
            })?;

            // Checkout the commit
            wt_repo.checkout_head(Some(
                git2::build::CheckoutBuilder::default().force()
            )).map_err(|e| {
                Git2DBError::internal(format!("Failed to checkout HEAD: {}", e))
            })?;

            info!(
                "Created git worktree at {} for ref '{}' (detached HEAD at {})",
                opts.worktree_path.display(),
                opts.ref_spec,
                commit.id()
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_name() {
        let driver = VfsDriver;
        assert_eq!(driver.name(), "vfs");
    }

    #[test]
    fn test_always_available() {
        let driver = VfsDriver;
        assert!(driver.is_available());
    }

    #[test]
    fn test_capabilities() {
        let driver = VfsDriver;
        let caps = driver.capabilities();

        assert!(!caps.copy_on_write);
        assert_eq!(caps.space_savings_percent, 0);
        assert!(!caps.requires_privileges);
        assert!(caps.platforms.contains(&"linux"));
        assert!(caps.platforms.contains(&"macos"));
        assert!(caps.platforms.contains(&"windows"));
        assert_eq!(caps.relative_performance, 1.0);
    }
}
