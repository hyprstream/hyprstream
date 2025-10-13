//! CoW-based git worktree implementation
//!
//! This module provides the main API for creating git worktrees
//! with automatic CoW mechanism selection and fallback handling.

use super::cow_mechanism::{
    CoWMechanism, CoWResolution, WorktreeConfig, UnavailableAction,
    OverlayfsConfig, ReflinkConfig, HardlinkConfig,
};
use crate::errors::{Git2DBError, Git2DBResult};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{info, warn, error, debug};

/// Main struct for creating CoW-based git worktrees
pub struct CoWWorktree {
    config: WorktreeConfig,
    repo_path: PathBuf,
}

impl CoWWorktree {
    /// Create a new CoW worktree manager
    pub fn new(repo_path: impl AsRef<Path>, config: WorktreeConfig) -> Self {
        Self {
            config,
            repo_path: repo_path.as_ref().to_path_buf(),
        }
    }

    /// Create a worktree with automatic CoW mechanism selection
    pub fn create_worktree(
        &self,
        branch: &str,
        target_path: impl AsRef<Path>,
    ) -> Git2DBResult<WorktreeHandle> {
        let target = target_path.as_ref();

        // Resolve the CoW mechanism
        let resolution = self.config.mechanism.resolve(&self.config.performance_hints);

        match resolution {
            CoWResolution::Available(mechanism) => {
                self.create_with_mechanism(branch, target, mechanism)
            }
            CoWResolution::Unavailable { requested, reason } => {
                self.handle_unavailable(branch, target, requested, reason)
            }
            CoWResolution::NoneAvailable => {
                self.handle_none_available(branch, target)
            }
        }
    }

    /// Create worktree with specific CoW mechanism
    fn create_with_mechanism(
        &self,
        branch: &str,
        target: &Path,
        mechanism: CoWMechanism,
    ) -> Git2DBResult<WorktreeHandle> {
        if self.config.log_mechanism_selection {
            info!(
                "Creating git worktree with {} CoW mechanism (~{}% space savings)",
                mechanism.name(),
                mechanism.space_savings_estimate()
            );
        }

        match mechanism {
            CoWMechanism::Overlayfs(config) => {
                self.create_overlayfs_worktree(branch, target, config)
            }
            CoWMechanism::Reflink(config) => {
                self.create_reflink_worktree(branch, target, config)
            }
            CoWMechanism::Hardlink(config) => {
                self.create_hardlink_worktree(branch, target, config)
            }
            CoWMechanism::Auto => {
                unreachable!("Auto should be resolved before this point")
            }
            CoWMechanism::Custom(backend) => {
                self.create_custom_worktree(branch, target, backend)
            }
        }
    }

    /// Handle unavailable CoW mechanism
    fn handle_unavailable(
        &self,
        branch: &str,
        target: &Path,
        requested: CoWMechanism,
        reason: String,
    ) -> Git2DBResult<WorktreeHandle> {
        match self.config.unavailable_action {
            UnavailableAction::Fail => {
                error!("Requested CoW mechanism '{}' unavailable: {}", requested.name(), reason);
                Err(Git2DBError::Other(format!(
                    "CoW mechanism '{}' unavailable: {}",
                    requested.name(),
                    reason
                )))
            }
            UnavailableAction::Fallback => {
                debug!("CoW mechanism '{}' unavailable ({}), using plain worktree",
                    requested.name(), reason);
                self.create_plain_worktree(branch, target)
            }
            UnavailableAction::WarnAndFallback => {
                warn!(
                    "CoW mechanism '{}' unavailable: {}. Falling back to plain worktree (no space savings)",
                    requested.name(),
                    reason
                );
                self.create_plain_worktree(branch, target)
            }
            UnavailableAction::TryNext => {
                // Try next available mechanism
                self.try_alternative_mechanisms(branch, target, Some(requested))
            }
        }
    }

    /// Handle case where no CoW mechanisms are available
    fn handle_none_available(
        &self,
        branch: &str,
        target: &Path,
    ) -> Git2DBResult<WorktreeHandle> {
        match self.config.unavailable_action {
            UnavailableAction::Fail => {
                error!("No CoW mechanisms available on this system");
                Err(Git2DBError::Other(
                    "No CoW mechanisms available, and fallback disabled".into()
                ))
            }
            UnavailableAction::Fallback => {
                debug!("No CoW mechanisms available, using plain worktree");
                self.create_plain_worktree(branch, target)
            }
            UnavailableAction::WarnAndFallback => {
                warn!(
                    "WARNING: No CoW mechanisms available on this system. \
                     Creating plain worktree without space savings."
                );
                self.create_plain_worktree(branch, target)
            }
            UnavailableAction::TryNext => {
                // No alternatives to try, must fallback or fail
                warn!("No CoW mechanisms to try, falling back to plain worktree");
                self.create_plain_worktree(branch, target)
            }
        }
    }

    /// Try alternative CoW mechanisms
    fn try_alternative_mechanisms(
        &self,
        branch: &str,
        target: &Path,
        skip: Option<CoWMechanism>,
    ) -> Git2DBResult<WorktreeHandle> {
        let mechanisms = vec![
            CoWMechanism::Overlayfs(OverlayfsConfig::default()),
            CoWMechanism::Reflink(ReflinkConfig::default()),
            CoWMechanism::Hardlink(HardlinkConfig::default()),
        ];

        for mechanism in mechanisms {
            // Skip if this was the originally requested one
            if let Some(ref skip_mech) = skip {
                if std::mem::discriminant(&mechanism) == std::mem::discriminant(skip_mech) {
                    continue;
                }
            }

            if mechanism.is_available() {
                info!("Trying alternative CoW mechanism: {}", mechanism.name());
                return self.create_with_mechanism(branch, target, mechanism);
            }
        }

        // No alternatives worked
        self.handle_none_available(branch, target)
    }

    /// Create overlayfs-based worktree
    fn create_overlayfs_worktree(
        &self,
        branch: &str,
        target: &Path,
        config: OverlayfsConfig,
    ) -> Git2DBResult<WorktreeHandle> {
        info!("Creating overlayfs-based worktree for branch '{}'", branch);

        // Implementation would:
        // 1. Create base git worktree in temp location
        // 2. Set up overlayfs mount with base as lower
        // 3. Mount at target location
        // 4. Return handle with cleanup logic

        // For now, placeholder:
        todo!("Implement overlayfs worktree creation")
    }

    /// Create reflink-based worktree
    fn create_reflink_worktree(
        &self,
        branch: &str,
        target: &Path,
        config: ReflinkConfig,
    ) -> Git2DBResult<WorktreeHandle> {
        info!("Creating reflink-based worktree for branch '{}'", branch);

        // Implementation would:
        // 1. Create git worktree
        // 2. Use cp --reflink or filesystem-specific APIs
        // 3. Return handle

        todo!("Implement reflink worktree creation")
    }

    /// Create hardlink-based worktree
    fn create_hardlink_worktree(
        &self,
        branch: &str,
        target: &Path,
        config: HardlinkConfig,
    ) -> Git2DBResult<WorktreeHandle> {
        info!("Creating hardlink-based worktree for branch '{}'", branch);

        // Implementation would:
        // 1. Create git worktree
        // 2. Replace eligible files with hardlinks
        // 3. Track which files are hardlinked
        // 4. Return handle with special cleanup

        todo!("Implement hardlink worktree creation")
    }

    /// Create custom backend worktree
    fn create_custom_worktree(
        &self,
        branch: &str,
        target: &Path,
        backend: Box<dyn super::cow_mechanism::CoWBackend>,
    ) -> Git2DBResult<WorktreeHandle> {
        info!("Creating worktree with custom backend: {}", backend.name());
        backend.create_worktree(&self.repo_path, target)?;

        Ok(WorktreeHandle {
            path: target.to_path_buf(),
            mechanism: CoWMechanism::Custom(backend),
            cleanup_on_drop: true,
        })
    }

    /// Create plain worktree (fallback when no CoW available)
    fn create_plain_worktree(
        &self,
        branch: &str,
        target: &Path,
    ) -> Git2DBResult<WorktreeHandle> {
        warn!(
            "Creating plain git worktree without CoW optimization. \
             This will use full disk space with no deduplication."
        );

        // Standard git worktree command
        let output = Command::new("git")
            .args(&["worktree", "add", target.to_str().unwrap(), branch])
            .current_dir(&self.repo_path)
            .output()
            .map_err(|e| Git2DBError::Other(format!("Failed to create worktree: {}", e)))?;

        if !output.status.success() {
            return Err(Git2DBError::Other(format!(
                "Git worktree creation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(WorktreeHandle {
            path: target.to_path_buf(),
            mechanism: CoWMechanism::Auto, // Indicates fallback was used
            cleanup_on_drop: true,
        })
    }
}

/// Handle to a created worktree with automatic cleanup
pub struct WorktreeHandle {
    pub path: PathBuf,
    pub mechanism: CoWMechanism,
    pub cleanup_on_drop: bool,
}

impl WorktreeHandle {
    /// Get the path to the worktree
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the CoW mechanism used
    pub fn mechanism(&self) -> &CoWMechanism {
        &self.mechanism
    }

    /// Get space savings estimate
    pub fn space_savings_estimate(&self) -> u8 {
        self.mechanism.space_savings_estimate()
    }

    /// Disable automatic cleanup on drop
    pub fn persist(mut self) -> Self {
        self.cleanup_on_drop = false;
        self
    }
}

impl Drop for WorktreeHandle {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            if let Err(e) = cleanup_worktree(&self.path, &self.mechanism) {
                error!("Failed to cleanup worktree: {}", e);
            }
        }
    }
}

/// Cleanup a worktree based on its creation mechanism
fn cleanup_worktree(path: &Path, mechanism: &CoWMechanism) -> Git2DBResult<()> {
    match mechanism {
        CoWMechanism::Overlayfs(_) => {
            // Unmount overlayfs, cleanup temps
            todo!("Implement overlayfs cleanup")
        }
        CoWMechanism::Reflink(_) | CoWMechanism::Hardlink(_) | CoWMechanism::Auto => {
            // Standard git worktree removal
            Command::new("git")
                .args(&["worktree", "remove", "--force", path.to_str().unwrap()])
                .output()
                .map_err(|e| Git2DBError::Other(format!("Cleanup failed: {}", e)))?;
            Ok(())
        }
        CoWMechanism::Custom(backend) => {
            backend.cleanup_worktree(path)
        }
    }
}

/// Convenience builder for common use cases
pub struct QuickWorktree;

impl QuickWorktree {
    /// Create worktree with best available CoW
    pub fn auto(repo: impl AsRef<Path>) -> CoWWorktree {
        CoWWorktree::new(repo, WorktreeConfig::default())
    }

    /// Create worktree, fail if no CoW available
    pub fn require_cow(repo: impl AsRef<Path>) -> CoWWorktree {
        let mut config = WorktreeConfig::default();
        config.unavailable_action = UnavailableAction::Fail;
        CoWWorktree::new(repo, config)
    }

    /// Create worktree, prefer space savings
    pub fn minimize_space(repo: impl AsRef<Path>) -> CoWWorktree {
        let mut config = WorktreeConfig::default();
        config.performance_hints.priority = super::cow_mechanism::SelectionPriority::SpaceSaving;
        config.unavailable_action = UnavailableAction::Fail;
        CoWWorktree::new(repo, config)
    }

    /// Create worktree with maximum compatibility
    pub fn compatible(repo: impl AsRef<Path>) -> CoWWorktree {
        let mut config = WorktreeConfig::default();
        config.mechanism = CoWMechanism::Hardlink(HardlinkConfig::default());
        config.unavailable_action = UnavailableAction::Fallback;
        CoWWorktree::new(repo, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_builders() {
        let repo = PathBuf::from("/tmp/test-repo");

        // Auto selection
        let auto = QuickWorktree::auto(&repo);
        assert_eq!(auto.config.mechanism, CoWMechanism::Auto);

        // Require CoW
        let required = QuickWorktree::require_cow(&repo);
        assert_eq!(required.config.unavailable_action, UnavailableAction::Fail);

        // Space optimized
        let space = QuickWorktree::minimize_space(&repo);
        assert_eq!(
            space.config.performance_hints.priority,
            super::cow_mechanism::SelectionPriority::SpaceSaving
        );
    }
}