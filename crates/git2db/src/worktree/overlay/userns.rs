//! User namespace utilities for unprivileged overlayfs

use crate::errors::{Git2DBError, Git2DBResult};
use std::path::Path;
use tracing::{debug, info};

/// Check if unprivileged user namespaces are enabled
pub fn are_user_namespaces_enabled() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check sysctl setting
        if let Ok(content) = std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone") {
            if content.trim() == "1" {
                return true;
            }
        }

        // Some kernels don't have this sysctl but still support user namespaces
        // Try to create a user namespace to test
        can_create_user_namespace()
    }

    #[cfg(not(target_os = "linux"))]
    false
}

/// Test if we can create a user namespace
#[cfg(target_os = "linux")]
fn can_create_user_namespace() -> bool {
    use nix::sched::{unshare, CloneFlags};

    match unshare(CloneFlags::CLONE_NEWUSER) {
        Ok(_) => {
            debug!("Successfully created user namespace (test)");
            true
        }
        Err(e) => {
            debug!("Cannot create user namespace: {}", e);
            false
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn can_create_user_namespace() -> bool {
    false
}

/// Mount overlayfs in a user namespace
///
/// This creates a user namespace, maps root, and mounts overlayfs inside.
#[cfg(all(target_os = "linux", feature = "overlayfs"))]
pub async fn mount_in_userns(
    lower: &Path,
    upper: &Path,
    work: &Path,
    target: &Path,
) -> Git2DBResult<()> {
    use tokio::process::Command;

    info!("Mounting overlayfs in user namespace");

    let mount_opts = format!(
        "lowerdir={},upperdir={},workdir={}",
        lower.display(),
        upper.display(),
        work.display()
    );

    // Use unshare to create user namespace and mount
    let output = Command::new("unshare")
        .arg("--user")
        .arg("--map-root-user")
        .arg("--mount")
        .arg("mount")
        .arg("-t")
        .arg("overlay")
        .arg("overlay")
        .arg("-o")
        .arg(&mount_opts)
        .arg(target)
        .output()
        .await
        .map_err(|e| Git2DBError::internal(format!("Failed to mount in user namespace: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Git2DBError::internal(format!(
            "User namespace mount failed: {}",
            stderr
        )));
    }

    info!("Successfully mounted overlayfs in user namespace");

    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "overlayfs")))]
pub async fn mount_in_userns(
    _lower: &Path,
    _upper: &Path,
    _work: &Path,
    _target: &Path,
) -> Git2DBResult<()> {
    Err(Git2DBError::internal(
        "User namespace mounting not supported",
    ))
}

/// Backend that wraps kernel overlayfs with user namespace support
pub struct UserNamespaceBackend;

impl UserNamespaceBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn is_available() -> bool {
        are_user_namespaces_enabled()
    }
}

use super::backend::{BackendCapabilities, OverlayBackend};
use async_trait::async_trait;

#[async_trait]
impl OverlayBackend for UserNamespaceBackend {
    async fn mount(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        target: &Path,
    ) -> Git2DBResult<()> {
        mount_in_userns(lower, upper, work, target).await
    }

    async fn unmount(&self, target: &Path) -> Git2DBResult<()> {
        use tokio::process::Command;

        // Unmount can be done from outside the namespace
        let _ = Command::new("umount").arg(target).status().await;

        Ok(())
    }

    fn is_available(&self) -> bool {
        Self::is_available()
    }

    fn name(&self) -> &'static str {
        "userns"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            requires_privileges: false,
            requires_binary: Some("unshare"),
            relative_performance: 1.0,
            user_namespace_compatible: true,
            space_savings_percent: 80,
        }
    }
}
