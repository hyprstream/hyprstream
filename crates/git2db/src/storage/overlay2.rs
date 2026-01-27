//! Overlay2 driver (Linux overlayfs)
use super::driver::{Driver, DriverOpts, WorktreeHandle};
#[cfg(all(target_os = "linux", feature = "overlayfs"))]
use super::driver::DriverFactory;
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::Path;
#[cfg(all(target_os = "linux", feature = "overlayfs"))]
use tokio::process::Command;
#[cfg(all(target_os = "linux", feature = "overlayfs"))]
use tracing::{info, debug, warn};

#[cfg(all(target_os = "linux", feature = "overlayfs"))]
inventory::submit!(DriverFactory::new(
    "overlay2",
    || Box::new(Overlay2Driver)
));

/// Overlay2 storage driver (Linux overlayfs)
///
/// Automatically tries mount methods in order: kernel → userns → fuse
#[allow(dead_code)] // Constructed via inventory::submit! when overlayfs feature enabled
pub struct Overlay2Driver;

#[async_trait]
impl Driver for Overlay2Driver {
    fn name(&self) -> &'static str {
        "overlay2"
    }

    fn is_available(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "overlayfs"))]
        {
            // Check if overlay filesystem is supported by kernel
            if let Ok(filesystems) = std::fs::read_to_string("/proc/filesystems") {
                filesystems.contains("overlay")
            } else {
                false
            }
        }

        #[cfg(not(all(target_os = "linux", feature = "overlayfs")))]
        false
    }

  
  #[cfg(feature = "overlayfs")]
    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        // Strict: worktree path must not exist
        if opts.worktree_path.exists() {
            return Err(Git2DBError::worktree_exists(&opts.worktree_path));
        }

        // Validate inputs
        if !opts.base_repo.exists() {
            return Err(Git2DBError::invalid_path(
                opts.base_repo.clone(),
                "Base repository does not exist",
            ));
        }

        // Generate unique ID
        let id = format!("git2db-{}", uuid::Uuid::new_v4());

        // Create overlay directories in a separate location
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
        let mount_dir = overlay_base.join("mount");

        tokio::fs::create_dir_all(&upper_dir).await.map_err(|e| {
            Git2DBError::internal(format!("Failed to create upper directory: {e}"))
        })?;
        tokio::fs::create_dir_all(&work_dir).await.map_err(|e| {
            Git2DBError::internal(format!("Failed to create work directory: {e}"))
        })?;
        tokio::fs::create_dir_all(&mount_dir)
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to create mount point: {e}")))?;

        info!(
            "Creating overlay2 worktree: id={}, lower={}, mount={}",
            id,
            opts.base_repo.display(),
            mount_dir.display()
        );

        // Try mounting with fallback strategy
        let mount_method = self.try_mount_overlayfs(
            &opts.base_repo,
            &upper_dir,
            &work_dir,
            &mount_dir,
        )
        .await?;

        info!("Successfully mounted using {}", mount_method);

        // Create git worktree at the desired path
        self.create_git_worktree(&opts.base_repo, &opts.worktree_path, &opts.ref_spec)
            .await?;

        // Create handle with cleanup
        let mount_point = mount_dir.clone();
        let upper = upper_dir.clone();
        let work = work_dir.clone();
        let id_clone = id.clone();

        let cleanup = move || async move {
            let mount_point = mount_point.clone();
            let upper = upper.clone();
            let work = work.clone();
            let id = id_clone.clone();

            info!("Cleaning up overlay2 worktree {}", id);

            // Unmount (try fusermount first, then umount)
            if let Err(e) = Self::unmount_overlayfs(&mount_point).await {
                warn!(
                    "Failed to unmount {} in cleanup: {}",
                    mount_point.display(),
                    e
                );
            }

            // Remove overlay directories
            if let Err(e) = tokio::fs::remove_dir_all(&upper).await {
                warn!("Failed to remove upper dir {}: {}", upper.display(), e);
            }
            if let Err(e) = tokio::fs::remove_dir_all(&work).await {
                warn!("Failed to remove work dir {}: {}", work.display(), e);
            }
            if let Err(e) = tokio::fs::remove_dir_all(&mount_point).await {
                warn!("Failed to remove mount point {}: {}", mount_point.display(), e);
            }

            Ok(())
        };

        Ok(WorktreeHandle::with_cleanup(
            opts.worktree_path.clone(),
            format!("overlay2-{mount_method}"),
            cleanup,
        ))
    }

    #[cfg(not(feature = "overlayfs"))]
    async fn create_worktree(&self, _opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        Err(Git2DBError::internal(
            "overlay2 driver requires 'overlayfs' feature to be enabled",
        ))
    }

    #[cfg(feature = "overlayfs")]
    async fn get_worktrees(&self, base_repo: &Path) -> Git2DBResult<Vec<WorktreeHandle>> {
        let worktrees_dir = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees");

        let mut worktrees = Vec::new();

        if !worktrees_dir.exists() {
            return Ok(worktrees);
        }

        // Read worktrees directory
        for entry in std::fs::read_dir(&worktrees_dir)? {
            let entry = entry?;
            let worktree_path = entry.path();

            // Skip non-directories and git's internal directories
            if !worktree_path.is_dir() || worktree_path.file_name().is_some_and(|name| {
                name.to_string_lossy().starts_with(".git")
            }) {
                continue;
            }

            // Check if this is an overlay2 worktree by looking for overlay signatures
            if self.is_overlay2_worktree(&worktree_path) {
                worktrees.push(WorktreeHandle::new(worktree_path, "overlay2".to_owned()));
            }
        }

        Ok(worktrees)
    }

    #[cfg(feature = "overlayfs")]
    async fn get_worktree(&self, base_repo: &Path, branch: &str) -> Git2DBResult<Option<WorktreeHandle>> {
        let worktree_path = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees")
            .join(branch);

        if worktree_path.exists() && self.is_overlay2_worktree(&worktree_path) {
            Ok(Some(WorktreeHandle::new(worktree_path, "overlay2".to_owned())))
        } else {
            Ok(None)
        }
    }

    #[cfg(not(feature = "overlayfs"))]
    async fn get_worktrees(&self, _base_repo: &Path) -> Git2DBResult<Vec<WorktreeHandle>> {
        Err(Git2DBError::internal(
            "overlay2 driver requires 'overlayfs' feature to be enabled",
        ))
    }

    #[cfg(not(feature = "overlayfs"))]
    async fn get_worktree(&self, _base_repo: &Path, _branch: &str) -> Git2DBResult<Option<WorktreeHandle>> {
        Err(Git2DBError::internal(
            "overlay2 driver requires 'overlayfs' feature to be enabled",
        ))
    }
}

#[cfg(all(target_os = "linux", feature = "overlayfs"))]
impl Overlay2Driver {
    /// Try mounting overlayfs with automatic fallback strategy
    ///
    /// Attempts methods in order of preference:
    /// 1. Kernel overlayfs (fastest, requires CAP_SYS_ADMIN or user namespace)
    /// 2. User namespace overlayfs (good perf, unprivileged, requires kernel support)
    /// 3. FUSE overlayfs (slow but compatible, requires fuse-overlayfs binary)
    async fn try_mount_overlayfs(
        &self,
        lower: &Path,
        upper: &Path,
        work: &Path,
        target: &Path,
    ) -> Git2DBResult<String> {
        // Build common mount options
        let mount_opts = format!(
            "lowerdir={},upperdir={},workdir={}",
            lower.display(),
            upper.display(),
            work.display()
        );

        // Try 1: Kernel overlayfs (fastest)
        if let Ok(filesystems) = std::fs::read_to_string("/proc/filesystems") {
            if filesystems.contains("overlay") {
                debug!("Attempting kernel overlayfs mount");
                match Self::mount_kernel(target, &mount_opts).await {
                    Ok(()) => {
                        return Ok("kernel".to_owned());
                    }
                    Err(e) => {
                        debug!("Kernel overlayfs failed, trying next method: {}", e);
                    }
                }
            }
        }

        // Try 2: User namespace overlayfs (unprivileged)
        if Self::can_use_userns().await {
            debug!("Attempting user namespace overlayfs mount");
            match Self::mount_userns(lower, upper, work, target).await {
                Ok(()) => {
                    return Ok("userns".to_owned());
                }
                Err(e) => {
                    debug!("User namespace mount failed, trying next method: {}", e);
                }
            }
        }

        // Try 3: FUSE overlayfs (most compatible)
        debug!("Attempting FUSE overlayfs mount");
        match Self::mount_fuse(target, &mount_opts).await {
            Ok(()) => {
                return Ok("fuse".to_owned());
            }
            Err(e) => {
                debug!("FUSE overlayfs failed: {}", e);
            }
        }

        // All methods failed
        Err(Git2DBError::internal(
            "All overlayfs mount methods failed. Ensure one of: \
             1) Kernel overlayfs + CAP_SYS_ADMIN capability, \
             2) User namespace support (unprivileged_userns_clone=1), \
             3) fuse-overlayfs binary installed and available".to_owned()
        ))
    }

    /// Mount using kernel overlayfs syscall
    #[cfg(all(target_os = "linux", feature = "overlayfs"))]
    async fn mount_kernel(target: &Path, mount_opts: &str) -> Git2DBResult<()> {
        use nix::mount::{mount, MsFlags};

        mount(
            Some("overlay"),
            target,
            Some("overlay"),
            MsFlags::empty(),
            Some(mount_opts),
        )
        .map_err(|e| {
            Git2DBError::internal(format!("Failed to mount kernel overlayfs: {e}"))
        })
    }

    /// Check if user namespaces are available
    async fn can_use_userns() -> bool {
        // Check sysctl setting
        if let Ok(content) = std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone") {
            return content.trim() == "1";
        }

        // Try to create a test namespace
        use nix::sched::{unshare, CloneFlags};
        unshare(CloneFlags::CLONE_NEWUSER).is_ok()
    }

    /// Mount using user namespace
    async fn mount_userns(
        lower: &Path,
        upper: &Path,
        work: &Path,
        target: &Path,
    ) -> Git2DBResult<()> {
        let mount_opts = format!(
            "lowerdir={},upperdir={},workdir={},userxattr",
            lower.display(),
            upper.display(),
            work.display()
        );

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
            .map_err(|e| Git2DBError::internal(format!("Failed to execute unshare: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Git2DBError::internal(format!(
                "User namespace mount failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Mount using FUSE overlayfs
    async fn mount_fuse(target: &Path, mount_opts: &str) -> Git2DBResult<()> {
        let output = Command::new("fuse-overlayfs")
            .arg("-o")
            .arg(mount_opts)
            .arg(target)
            .output()
            .await
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to execute fuse-overlayfs: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Git2DBError::internal(format!(
                "FUSE overlayfs failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Unmount overlayfs (try fusermount first, then umount)
    async fn unmount_overlayfs(target: &Path) -> Git2DBResult<()> {
        // Try fusermount first (for FUSE mounts)
        let fusermount_output = Command::new("fusermount")
            .arg("-u")
            .arg(target)
            .output()
            .await;

        if let Ok(output) = fusermount_output {
            if output.status.success() {
                return Ok(());
            }
        }

        // Fallback to umount
        let output = Command::new("umount")
            .arg(target)
            .output()
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to execute umount: {e}")))?;

        if !output.status.success() {
            // Try lazy unmount as last resort
            let _ = Command::new("umount")
                .arg("-l")
                .arg(target)
                .output()
                .await;
        }

        Ok(())
    }
}

#[cfg(feature = "overlayfs")]
impl Overlay2Driver {
    /// Create git worktree using libgit2 with unified ref support
    ///
    /// Note: The overlay is already mounted by the driver.
    /// We just need to create the git worktree structure on top of it.
    async fn create_git_worktree(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        ref_spec: &str,
    ) -> Git2DBResult<()> {
        // Open the base repository
        let repo = git2::Repository::open(base_repo)
            .map_err(|e| Git2DBError::internal(format!("Failed to open repository: {e}")))?;

        // Resolve ref_spec to a commit
        let object = repo.revparse_single(ref_spec).map_err(|e| {
            Git2DBError::internal(format!("Failed to resolve ref '{ref_spec}': {e}"))
        })?;

        let commit = object.peel_to_commit().map_err(|e| {
            Git2DBError::internal(format!(
                "Ref '{ref_spec}' does not point to a commit: {e}"
            ))
        })?;

        // Check if this is a branch
        let branch_ref_name = format!("refs/heads/{ref_spec}");
        let is_branch = repo.find_reference(&branch_ref_name).is_ok();

        // Create worktree name
        let worktree_name = worktree_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                Git2DBError::invalid_path(worktree_path.to_path_buf(), "Invalid worktree path")
            })?;

        // Create worktree at the desired path (no conflict now since mount is elsewhere)
        if is_branch {
            let reference = repo.find_reference(&branch_ref_name)?;
            repo.worktree(
                worktree_name,
                worktree_path,
                Some(git2::WorktreeAddOptions::new().reference(Some(&reference))),
            )
            .map_err(|e| Git2DBError::internal(format!("Failed to create worktree: {e}")))?;

            info!(
                "Created overlay2 worktree at {} for branch '{}' (commit: {})",
                worktree_path.display(),
                ref_spec,
                commit.id()
            );
        } else {
            repo.worktree(worktree_name, worktree_path, None)
                .map_err(|e| Git2DBError::internal(format!("Failed to create worktree: {e}")))?;

            let wt_repo = git2::Repository::open(worktree_path)?;
            wt_repo.set_head_detached(commit.id())?;
            wt_repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;

            info!(
                "Created overlay2 worktree at {} for ref '{}' (detached HEAD at {})",
                worktree_path.display(),
                ref_spec,
                commit.id()
            );
        }

        Ok(())
    }

    /// Check if a worktree is managed by overlay2 driver
    fn is_overlay2_worktree(&self, worktree_path: &Path) -> bool {
        // Check for overlay2 signatures
        let overlay_path = worktree_path.join(".overlay");
        let git_overlay_path = worktree_path.join(".git").join("overlay");

        // Also check for overlayfs mount signatures
        let has_overlay_sig = overlay_path.exists() || git_overlay_path.exists();

        // Additional check: look for overlay mount points in /proc/mounts
        if has_overlay_sig {
            if let Ok(mounts) = std::fs::read_to_string("/proc/mounts") {
                let worktree_str = worktree_path.to_string_lossy();
                if mounts.contains(&*worktree_str) {
                    return true;
                }
            }
        }

        has_overlay_sig
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_name() {
        assert_eq!(Overlay2Driver.name(), "overlay2");
    }

    #[test]
    #[cfg(all(feature = "overlayfs", target_os = "linux"))]
    fn test_availability() {
        // May or may not be available depending on system
        println!("Overlay2 available: {}", Overlay2Driver.is_available());
    }
}