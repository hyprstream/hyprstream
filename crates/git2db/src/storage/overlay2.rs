//! Overlay2 driver (Linux overlayfs)
//!
//! Implements the overlay2 storage driver using Linux overlayfs,
//! following Docker's overlay2 driver design.
//!
//! This driver provides:
//! - ~80% disk space savings via Copy-on-Write
//! - Multiple backend options (FUSE, user namespace, kernel)
//! - Full git worktree functionality with optimized storage

use super::driver::{Driver, DriverCapabilities, DriverOpts, WorktreeHandle};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

// Re-export overlay backends from worktree module
#[cfg(feature = "overlayfs")]
use crate::worktree::overlay::{
    select_best_backend, FuseBackend, KernelBackend, OverlayBackend, UserNamespaceBackend,
};

/// Configuration for overlay2 driver
#[derive(Debug, Clone)]
pub struct Overlay2Config {
    /// Force a specific backend (fuse, userns, kernel)
    pub force_backend: Option<String>,

    /// Custom overlay directory (default: temp directory)
    pub overlay_dir: Option<PathBuf>,
}

impl Default for Overlay2Config {
    fn default() -> Self {
        Self {
            force_backend: None,
            overlay_dir: None,
        }
    }
}

/// Overlay2 storage driver
///
/// Uses Linux overlayfs to create space-efficient git worktrees.
pub struct Overlay2Driver {
    #[cfg(feature = "overlayfs")]
    backend: std::sync::Arc<dyn OverlayBackend>,
    #[allow(dead_code)]
    config: Overlay2Config,
}

impl Overlay2Driver {
    /// Create with default configuration
    #[cfg(feature = "overlayfs")]
    pub fn new() -> Self {
        Self {
            backend: select_best_backend(),
            config: Overlay2Config::default(),
        }
    }

    /// Create without overlayfs feature (stub)
    #[cfg(not(feature = "overlayfs"))]
    pub fn new() -> Self {
        Self {
            config: Overlay2Config::default(),
        }
    }

    /// Create with custom configuration
    #[cfg(feature = "overlayfs")]
    pub fn with_config(config: Overlay2Config) -> Self {
        let backend = if let Some(ref name) = config.force_backend {
            match name.as_str() {
                "fuse" => std::sync::Arc::new(FuseBackend::new()) as std::sync::Arc<dyn OverlayBackend>,
                "userns" => std::sync::Arc::new(UserNamespaceBackend::new()) as std::sync::Arc<dyn OverlayBackend>,
                "kernel" => std::sync::Arc::new(KernelBackend::new()) as std::sync::Arc<dyn OverlayBackend>,
                _ => select_best_backend(),
            }
        } else {
            select_best_backend()
        };

        Self { backend, config }
    }

    /// Create with custom configuration (stub without overlayfs)
    #[cfg(not(feature = "overlayfs"))]
    pub fn with_config(config: Overlay2Config) -> Self {
        Self { config }
    }
}

#[cfg(feature = "overlayfs")]
impl Default for Overlay2Driver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Driver for Overlay2Driver {
    fn name(&self) -> &'static str {
        "overlay2"
    }

    #[cfg(feature = "overlayfs")]
    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.backend.is_available()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    #[cfg(not(feature = "overlayfs"))]
    fn is_available(&self) -> bool {
        false
    }

    fn capabilities(&self) -> DriverCapabilities {
        #[cfg(feature = "overlayfs")]
        {
            let backend_caps = self.backend.capabilities();
            DriverCapabilities {
                copy_on_write: true,
                space_savings_percent: backend_caps.space_savings_percent,
                requires_privileges: backend_caps.requires_privileges,
                platforms: vec!["linux"],
                required_binaries: backend_caps
                    .requires_binary
                    .map(|b| vec![b])
                    .unwrap_or_default(),
                relative_performance: backend_caps.relative_performance,
            }
        }

        #[cfg(not(feature = "overlayfs"))]
        {
            DriverCapabilities {
                copy_on_write: false,
                space_savings_percent: 0,
                requires_privileges: false,
                platforms: vec![],
                required_binaries: vec![],
                relative_performance: 0.0,
            }
        }
    }

    #[cfg(feature = "overlayfs")]
    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        // Validate inputs
        if !opts.base_repo.exists() {
            return Err(Git2DBError::invalid_path(
                opts.base_repo.clone(),
                "Base repository does not exist",
            ));
        }

        // Generate unique ID
        let id = format!("git2db-{}", uuid::Uuid::new_v4());

        // Create overlay directories
        let overlay_base = opts
            .worktree_path
            .parent()
            .ok_or_else(|| {
                Git2DBError::invalid_path(
                    opts.worktree_path.clone(),
                    "Worktree path has no parent directory",
                )
            })?
            .join(".git2db-overlay")
            .join(&id);

        let upper_dir = overlay_base.join("upper");
        let work_dir = overlay_base.join("work");

        tokio::fs::create_dir_all(&upper_dir)
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to create upper directory: {}", e)))?;
        tokio::fs::create_dir_all(&work_dir)
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to create work directory: {}", e)))?;
        tokio::fs::create_dir_all(&opts.worktree_path)
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to create mount point: {}", e)))?;

        info!(
            "Creating overlay2 worktree: id={}, backend={}, lower={}, mount={}",
            id,
            self.backend.name(),
            opts.base_repo.display(),
            opts.worktree_path.display()
        );

        // Mount overlayfs
        self.backend
            .mount(&opts.base_repo, &upper_dir, &work_dir, &opts.worktree_path)
            .await?;

        // Create git worktree on the overlay mount (supports any ref: branch, commit, tag, etc.)
        self.create_git_worktree(&opts.base_repo, &opts.worktree_path, &opts.ref_spec).await?;

        // Create handle with cleanup
        let mount_point = opts.worktree_path.clone();
        let backend = self.backend.clone();
        let upper = upper_dir.clone();
        let work = work_dir.clone();
        let id_clone = id.clone();

        let cleanup = Box::new(move || {
            let mount_point = mount_point.clone();
            let backend = backend.clone();
            let upper = upper.clone();
            let work = work.clone();
            let id = id_clone.clone();

            tokio::spawn(async move {
                info!("Cleaning up overlay2 worktree {}", id);

                // Unmount
                if let Err(e) = backend.unmount(&mount_point).await {
                    warn!("Failed to unmount {} in cleanup: {}", mount_point.display(), e);
                }

                // Remove overlay directories
                let _ = tokio::fs::remove_dir_all(&upper).await;
                let _ = tokio::fs::remove_dir_all(&work).await;
                let _ = tokio::fs::remove_dir_all(&mount_point).await;
            });
        });

        Ok(WorktreeHandle::with_cleanup(
            opts.worktree_path.clone(),
            format!("overlay2-{}", self.backend.name()),
            cleanup,
        ))
    }

    #[cfg(not(feature = "overlayfs"))]
    async fn create_worktree(&self, _opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        Err(Git2DBError::internal(
            "overlay2 driver requires 'overlayfs' feature to be enabled",
        ))
    }
}

#[cfg(feature = "overlayfs")]
impl Overlay2Driver {
    /// Create git worktree using libgit2 with unified ref support
    ///
    /// Note: The overlay is already mounted at worktree_path by the driver.
    /// We just need to create the git worktree structure on top of it.
    async fn create_git_worktree(&self, base_repo: &Path, worktree_path: &Path, ref_spec: &str) -> Git2DBResult<()> {
        // Open the base repository
        let repo = git2::Repository::open(base_repo).map_err(|e| {
            Git2DBError::internal(format!("Failed to open repository: {}", e))
        })?;

        // Resolve ref_spec to a commit
        let object = repo.revparse_single(ref_spec).map_err(|e| {
            Git2DBError::internal(format!("Failed to resolve ref '{}': {}", ref_spec, e))
        })?;

        let commit = object.peel_to_commit().map_err(|e| {
            Git2DBError::internal(format!("Ref '{}' does not point to a commit: {}", ref_spec, e))
        })?;

        // Check if this is a branch
        let branch_ref_name = format!("refs/heads/{}", ref_spec);
        let is_branch = repo.find_reference(&branch_ref_name).is_ok();

        // Create worktree name
        let worktree_name = worktree_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                Git2DBError::invalid_path(
                    worktree_path.to_path_buf(),
                    "Invalid worktree path",
                )
            })?;

        // Create worktree on the overlay mount
        if is_branch {
            let reference = repo.find_reference(&branch_ref_name)?;
            repo.worktree(
                worktree_name,
                worktree_path,
                Some(
                    git2::WorktreeAddOptions::new()
                        .reference(Some(&reference)),
                ),
            )
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to create worktree: {}", e))
            })?;

            info!(
                "Created overlay2 worktree at {} for branch '{}' (commit: {})",
                worktree_path.display(),
                ref_spec,
                commit.id()
            );
        } else {
            repo.worktree(worktree_name, worktree_path, None)
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to create worktree: {}", e))
                })?;

            let wt_repo = git2::Repository::open(worktree_path)?;
            wt_repo.set_head_detached(commit.id())?;
            wt_repo.checkout_head(Some(
                git2::build::CheckoutBuilder::default().force()
            ))?;

            info!(
                "Created overlay2 worktree at {} for ref '{}' (detached HEAD at {})",
                worktree_path.display(),
                ref_spec,
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
        let driver = Overlay2Driver::new();
        assert_eq!(driver.name(), "overlay2");
    }

    #[test]
    #[cfg(all(feature = "overlayfs", target_os = "linux"))]
    fn test_availability() {
        let driver = Overlay2Driver::new();
        // May or may not be available depending on system
        println!("Overlay2 available: {}", driver.is_available());
    }

    #[test]
    fn test_capabilities() {
        let driver = Overlay2Driver::new();
        let caps = driver.capabilities();

        #[cfg(feature = "overlayfs")]
        {
            assert!(caps.copy_on_write);
            assert!(caps.space_savings_percent > 0);
        }

        #[cfg(not(feature = "overlayfs"))]
        {
            assert!(!caps.copy_on_write);
            assert_eq!(caps.space_savings_percent, 0);
        }
    }
}
