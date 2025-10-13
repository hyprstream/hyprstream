//! Kernel overlayfs backend using direct mount syscalls

use super::backend::{OverlayBackend, BackendCapabilities};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::Path;
use tracing::{debug, warn};

#[cfg(all(target_os = "linux", feature = "overlayfs"))]
use nix::mount::{mount, umount, MsFlags};

/// Kernel overlayfs backend
///
/// Uses direct mount(2) syscall via nix crate.
/// Requires CAP_SYS_ADMIN or user namespace.
pub struct KernelBackend;

impl KernelBackend {
    pub fn new() -> Self {
        Self
    }

    /// Check if we can actually mount (have privileges)
    fn can_mount(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "overlayfs"))]
        {
            // Try to detect capabilities if caps crate is available
            #[cfg(feature = "overlayfs")]
            {
                use caps::{Capability, CapSet};

                // Check for CAP_SYS_ADMIN
                if let Ok(true) = caps::has_cap(None, CapSet::Effective, Capability::CAP_SYS_ADMIN) {
                    return true;
                }
            }

            // Could be in user namespace - try a test mount
            // (We'll detect this properly in the auto-selection logic)
            false
        }

        #[cfg(not(all(target_os = "linux", feature = "overlayfs")))]
        false
    }
}

#[async_trait]
impl OverlayBackend for KernelBackend {
    #[cfg(all(target_os = "linux", feature = "overlayfs"))]
    async fn mount(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        target: &Path,
    ) -> Git2DBResult<()> {
        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            lower.display(),
            upper.display(),
            work.display()
        );

        debug!("Mounting kernel overlayfs at {}", target.display());

        mount(
            Some("overlay"),
            target,
            Some("overlay"),
            MsFlags::empty(),
            Some(options.as_str()),
        ).map_err(|e| Git2DBError::internal(format!("Failed to mount kernel overlayfs: {}", e)))?;

        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "overlayfs")))]
    async fn mount(
        &self,
        _lower: &Path,
        _upper: &Path,
        _work: &Path,
        _target: &Path,
    ) -> Git2DBResult<()> {
        Err(Git2DBError::internal("Kernel overlayfs not available on this platform"))
    }

    #[cfg(all(target_os = "linux", feature = "overlayfs"))]
    async fn unmount(&self, target: &Path) -> Git2DBResult<()> {
        debug!("Unmounting kernel overlayfs at {}", target.display());

        if let Err(e) = umount(target) {
            warn!("Failed to unmount {}: {}", target.display(), e);

            // Try lazy unmount
            let _ = tokio::process::Command::new("umount")
                .arg("-l")
                .arg(target)
                .status()
                .await;
        }

        Ok(())
    }

    #[cfg(not(all(target_os = "linux", feature = "overlayfs")))]
    async fn unmount(&self, _target: &Path) -> Git2DBResult<()> {
        Ok(())
    }

    fn is_available(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "overlayfs"))]
        {
            // Check if overlay filesystem is supported
            if let Ok(filesystems) = std::fs::read_to_string("/proc/filesystems") {
                return filesystems.contains("overlay") && self.can_mount();
            }
            false
        }

        #[cfg(not(all(target_os = "linux", feature = "overlayfs")))]
        false
    }

    fn name(&self) -> &'static str {
        "kernel"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            requires_privileges: true,
            requires_binary: None,
            relative_performance: 1.0,
            user_namespace_compatible: true,
            space_savings_percent: 80,
        }
    }
}
