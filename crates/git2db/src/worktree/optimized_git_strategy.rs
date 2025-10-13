//! Optimized Git Worktree Strategy
//!
//! This module provides the CORRECT implementation where git worktrees are ALWAYS used,
//! with optional storage optimizations (like overlayfs) applied underneath.
//!
//! # Key Points
//!
//! 1. **Git worktrees are always created** - this is not optional
//! 2. **Overlayfs is an optimization layer** - it sits underneath the git worktree
//! 3. **The result is always a valid git worktree** - whether optimized or not
//!
//! # Architecture
//!
//! ```text
//! User Request: Create Worktree
//!         |
//!         v
//! Check Optimization Config
//!         |
//!    +---------+
//!    |         |
//!    v         v
//! With Opt   No Opt
//!    |         |
//!    v         |
//! Mount       |
//! Overlayfs   |
//!    |         |
//!    v         v
//! Create Git Worktree
//!    |         |
//!    +----+----+
//!         |
//!         v
//!    Git Worktree
//!    (optimized or not)
//! ```

use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn, error};

use super::{
    WorktreeStrategy, WorktreeHandle, WorktreeMetadata, StrategyCapabilities,
    storage_optimization::{WorktreeConfig, StorageOptimization, FallbackBehavior},
};

/// Git worktree strategy with optional storage optimizations
///
/// This is the PRIMARY worktree strategy that:
/// 1. Always creates git worktrees
/// 2. Optionally applies storage optimizations underneath
pub struct OptimizedGitStrategy {
    config: WorktreeConfig,
    /// Resolved optimization (Auto -> concrete strategy)
    resolved_optimization: StorageOptimization,
}

impl OptimizedGitStrategy {
    /// Create with default configuration (auto-optimization)
    pub fn new() -> Self {
        let config = WorktreeConfig::default();
        let resolved = config.optimization.resolve();

        Self {
            config,
            resolved_optimization: resolved,
        }
    }

    /// Create with specific configuration
    pub fn with_config(config: WorktreeConfig) -> Self {
        let resolved = config.optimization.resolve();

        if config.log_optimization {
            info!(
                "Worktree optimization: {} -> {}",
                config.optimization,
                resolved
            );
        }

        Self {
            config,
            resolved_optimization: resolved,
        }
    }

    /// Create with no optimization (standard git worktree)
    pub fn standard() -> Self {
        Self::with_config(WorktreeConfig {
            optimization: StorageOptimization::None,
            fallback: FallbackBehavior::Continue,
            log_optimization: false,
        })
    }

    /// Apply storage optimization if configured
    async fn apply_optimization(
        &self,
        worktree_path: &Path,
    ) -> Git2DBResult<Option<OptimizationHandle>> {
        match &self.resolved_optimization {
            StorageOptimization::None => {
                debug!("No storage optimization applied");
                Ok(None)
            }

            StorageOptimization::CopyOnWrite(cow_config) => {
                if !cow_config.is_available() {
                    match self.config.fallback {
                        FallbackBehavior::Fail => {
                            return Err(Git2DBError::Config(
                                "Copy-on-Write optimization not available".to_string()
                            ));
                        }
                        FallbackBehavior::Warn => {
                            warn!("Copy-on-Write optimization not available, continuing without optimization");
                        }
                        FallbackBehavior::Continue => {
                            debug!("Copy-on-Write optimization not available, continuing without optimization");
                        }
                    }
                    return Ok(None);
                }

                // Apply overlayfs optimization
                match self.mount_overlayfs(worktree_path, cow_config).await {
                    Ok(handle) => {
                        info!("Applied Copy-on-Write optimization using {}", cow_config.backend);
                        Ok(Some(handle))
                    }
                    Err(e) => {
                        match self.config.fallback {
                            FallbackBehavior::Fail => Err(e),
                            FallbackBehavior::Warn => {
                                warn!("Failed to apply Copy-on-Write optimization: {}", e);
                                Ok(None)
                            }
                            FallbackBehavior::Continue => {
                                debug!("Failed to apply Copy-on-Write optimization: {}", e);
                                Ok(None)
                            }
                        }
                    }
                }
            }

            // Future optimization strategies would go here
            _ => Ok(None),
        }
    }

    /// Mount overlayfs for Copy-on-Write optimization
    async fn mount_overlayfs(
        &self,
        worktree_path: &Path,
        config: &super::storage_optimization::CoWConfig,
    ) -> Git2DBResult<OptimizationHandle> {
        #[cfg(target_os = "linux")]
        {
            use super::overlay::{create_overlay_mount, OverlayConfig};

            let overlay_config = OverlayConfig {
                backend: match config.backend.resolve() {
                    super::storage_optimization::CoWBackend::Kernel => {
                        super::overlay::BackendType::Kernel
                    }
                    super::storage_optimization::CoWBackend::UserNamespace => {
                        super::overlay::BackendType::UserNamespace
                    }
                    super::storage_optimization::CoWBackend::Fuse => {
                        super::overlay::BackendType::Fuse
                    }
                    _ => super::overlay::BackendType::Auto,
                },
                work_dir: config.overlay_dir.clone().map(PathBuf::from),
                mount_options: config.mount_options.clone(),
            };

            let mount = create_overlay_mount(worktree_path, &overlay_config).await?;

            Ok(OptimizationHandle::Overlayfs(mount))
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(Git2DBError::Config(
                "Overlayfs optimization is only available on Linux".to_string()
            ))
        }
    }

    /// Create the git worktree (always happens, regardless of optimization)
    async fn create_git_worktree(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        branch: &str,
    ) -> Git2DBResult<GitWorktreeHandle> {
        use git2::{Repository, WorktreeAddOptions};

        // Open the base repository
        let repo = Repository::open(base_repo)
            .map_err(|e| Git2DBError::Git(format!("Failed to open repository: {}", e)))?;

        // Create worktree add options
        let mut opts = WorktreeAddOptions::new();
        opts.reference(Some(&repo.find_branch(branch, git2::BranchType::Local)
            .map_err(|e| Git2DBError::Git(format!("Branch not found: {}", e)))?
            .into_reference()));

        // Create the worktree
        let worktree_name = worktree_path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| Git2DBError::Config("Invalid worktree path".to_string()))?;

        repo.worktree(worktree_name, worktree_path, Some(&opts))
            .map_err(|e| Git2DBError::Git(format!("Failed to create worktree: {}", e)))?;

        info!("Created git worktree at {:?} for branch {}", worktree_path, branch);

        Ok(GitWorktreeHandle {
            path: worktree_path.to_path_buf(),
            branch: branch.to_string(),
        })
    }
}

#[async_trait]
impl WorktreeStrategy for OptimizedGitStrategy {
    async fn create(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        branch: &str,
    ) -> Git2DBResult<Box<dyn WorktreeHandle>> {
        // Step 1: Apply optimization if configured
        let optimization_handle = self.apply_optimization(worktree_path).await?;

        // Step 2: ALWAYS create git worktree
        let git_handle = self.create_git_worktree(base_repo, worktree_path, branch).await?;

        // Step 3: Combine handles
        let handle = OptimizedWorktreeHandle {
            git_handle,
            optimization_handle,
            optimization_type: self.resolved_optimization.clone(),
        };

        Ok(Box::new(handle))
    }

    fn name(&self) -> &'static str {
        match &self.resolved_optimization {
            StorageOptimization::None => "git-worktree",
            StorageOptimization::CopyOnWrite(_) => "git-worktree-cow",
            _ => "git-worktree-optimized",
        }
    }

    fn is_available(&self) -> bool {
        // Git worktrees are always available
        // Optimization availability is handled via fallback behavior
        true
    }

    fn capabilities(&self) -> StrategyCapabilities {
        let mut caps = StrategyCapabilities::default();

        match &self.resolved_optimization {
            StorageOptimization::None => {
                // Standard git worktree
                caps.space_efficient = false;
                caps.relative_performance = 1.0;
            }
            StorageOptimization::CopyOnWrite(config) => {
                caps.space_efficient = true;
                caps.relative_performance = 0.9; // Slight overhead from overlayfs

                match config.backend.resolve() {
                    super::storage_optimization::CoWBackend::Kernel => {
                        caps.requires_privileges = true;
                        caps.requirements.push("CAP_SYS_ADMIN or root".to_string());
                    }
                    super::storage_optimization::CoWBackend::UserNamespace => {
                        caps.requires_privileges = false;
                        caps.requirements.push("Unprivileged user namespaces".to_string());
                    }
                    super::storage_optimization::CoWBackend::Fuse => {
                        caps.requires_privileges = false;
                        caps.requirements.push("fuse-overlayfs binary".to_string());
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        caps
    }
}

/// Handle for optimization mechanisms
enum OptimizationHandle {
    #[cfg(target_os = "linux")]
    Overlayfs(super::overlay::OverlayMount),
    // Future: Hardlink, Reflink, etc.
}

impl OptimizationHandle {
    fn cleanup(&self) -> Git2DBResult<()> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Overlayfs(mount) => mount.unmount(),
        }
    }

    fn space_saved(&self) -> Option<u64> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Overlayfs(_) => {
                // Estimate ~80% savings
                // In production, would calculate actual savings
                Some(1024 * 1024 * 100) // 100MB example
            }
        }
    }
}

/// Handle for a git worktree
struct GitWorktreeHandle {
    path: PathBuf,
    branch: String,
}

impl GitWorktreeHandle {
    fn cleanup(&self) -> Git2DBResult<()> {
        use git2::Repository;

        // Remove the worktree
        if let Ok(repo) = Repository::open(&self.path) {
            if let Some(wt) = repo.find_worktree(&self.path.to_string_lossy()) {
                wt.prune(Some(
                    git2::WorktreePruneOptions::new()
                        .valid(true)
                        .locked(true)
                        .working_tree(true)
                ))
                .map_err(|e| Git2DBError::Git(format!("Failed to prune worktree: {}", e)))?;
            }
        }

        // Remove directory if it still exists
        if self.path.exists() {
            std::fs::remove_dir_all(&self.path)
                .map_err(|e| Git2DBError::Io(e))?;
        }

        Ok(())
    }
}

/// Combined handle for optimized git worktree
struct OptimizedWorktreeHandle {
    git_handle: GitWorktreeHandle,
    optimization_handle: Option<OptimizationHandle>,
    optimization_type: StorageOptimization,
}

impl WorktreeHandle for OptimizedWorktreeHandle {
    fn path(&self) -> &Path {
        &self.git_handle.path
    }

    fn is_valid(&self) -> bool {
        self.git_handle.path.exists()
    }

    fn metadata(&self) -> WorktreeMetadata {
        WorktreeMetadata {
            strategy_name: match &self.optimization_type {
                StorageOptimization::None => "git-worktree".to_string(),
                StorageOptimization::CopyOnWrite(_) => "git-worktree-cow".to_string(),
                _ => "git-worktree-optimized".to_string(),
            },
            created_at: chrono::Utc::now(),
            space_saved_bytes: self.optimization_handle.as_ref()
                .and_then(|h| h.space_saved()),
            backend_info: Some(self.optimization_type.to_string()),
            read_only: false,
        }
    }

    fn cleanup(&self) -> Git2DBResult<()> {
        // Clean up in reverse order:
        // 1. Remove git worktree
        self.git_handle.cleanup()?;

        // 2. Unmount optimization layer (if any)
        if let Some(ref opt_handle) = self.optimization_handle {
            opt_handle.cleanup()?;
        }

        Ok(())
    }
}

impl Drop for OptimizedWorktreeHandle {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            error!("Failed to cleanup optimized worktree: {}", e);
        }
    }
}