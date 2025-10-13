//! FUSE overlayfs backend using fuse-overlayfs

use super::backend::{OverlayBackend, BackendCapabilities};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::Path;
use tokio::process::Command;
use tracing::{debug, info};

/// FUSE overlayfs backend
///
/// Uses fuse-overlayfs userspace implementation.
/// No privileges required, but needs fuse-overlayfs installed.
pub struct FuseBackend {
    binary_path: String,
}

impl FuseBackend {
    pub fn new() -> Self {
        Self {
            binary_path: "fuse-overlayfs".to_string(),
        }
    }

    /// Check if fuse-overlayfs binary is available
    fn check_binary(&self) -> bool {
        std::process::Command::new(&self.binary_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

#[async_trait]
impl OverlayBackend for FuseBackend {
    async fn mount(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        target: &Path,
    ) -> Git2DBResult<()> {
        debug!("Mounting fuse-overlayfs at {}", target.display());

        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            lower.display(),
            upper.display(),
            work.display()
        );

        let output = Command::new(&self.binary_path)
            .arg("-o")
            .arg(&options)
            .arg(target)
            .output()
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to execute fuse-overlayfs: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Git2DBError::internal(format!("fuse-overlayfs failed: {}", stderr)));
        }

        info!("Mounted fuse-overlayfs at {}", target.display());

        Ok(())
    }

    async fn unmount(&self, target: &Path) -> Git2DBResult<()> {
        debug!("Unmounting fuse-overlayfs at {}", target.display());

        // Use fusermount to unmount
        let output = Command::new("fusermount")
            .arg("-u")
            .arg(target)
            .output()
            .await;

        if let Err(e) = output {
            // Fallback to regular umount
            let _ = Command::new("umount")
                .arg(target)
                .status()
                .await;

            return Err(Git2DBError::internal(format!("Failed to unmount: {}", e)));
        }

        Ok(())
    }

    fn is_available(&self) -> bool {
        self.check_binary()
    }

    fn name(&self) -> &'static str {
        "fuse"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            requires_privileges: false,
            requires_binary: Some("fuse-overlayfs"),
            relative_performance: 0.95, // ~5% slower than kernel
            user_namespace_compatible: true,
            space_savings_percent: 80,
        }
    }
}
