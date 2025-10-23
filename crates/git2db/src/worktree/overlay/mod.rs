//! Overlayfs-backed worktrees with composable backends
//!
//! Supports multiple overlayfs mounting strategies:
//! - FUSE overlayfs (compatible, no privileges, requires fuse-overlayfs package)
//! - User namespace overlayfs (fast, no privileges, requires kernel support)
//! - Kernel overlayfs (fastest, requires CAP_SYS_ADMIN)

mod backend;
mod fuse;
mod kernel;
mod userns;

pub use backend::{BackendCapabilities, OverlayBackend};
pub use fuse::FuseBackend;
pub use kernel::KernelBackend;
pub use userns::{are_user_namespaces_enabled, UserNamespaceBackend};

use super::strategy::{StrategyCapabilities, WorktreeHandle, WorktreeMetadata, WorktreeStrategy};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// Overlayfs worktree strategy
///
/// Implements WorktreeStrategy using Linux overlayfs for space-efficient CoW.
pub struct OverlayWorktreeStrategy {
    backend: Arc<dyn OverlayBackend>,
}

impl OverlayWorktreeStrategy {
    /// Create with automatic backend selection
    pub fn new() -> Self {
        Self {
            backend: select_best_backend(),
        }
    }

    /// Create with specific backend
    pub fn with_backend(backend: Arc<dyn OverlayBackend>) -> Self {
        Self { backend }
    }
}

impl Default for OverlayWorktreeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorktreeStrategy for OverlayWorktreeStrategy {
    async fn create(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        _branch: &str,
    ) -> Git2DBResult<Box<dyn WorktreeHandle>> {
        // Validate inputs
        if !base_repo.exists() {
            return Err(Git2DBError::invalid_path(
                base_repo.to_path_buf(),
                "Base repository does not exist",
            ));
        }

        // Generate unique ID
        let id = format!("git2db-{}", uuid::Uuid::new_v4());

        // Create overlay directories
        let overlay_base = worktree_path
            .parent()
            .ok_or_else(|| {
                Git2DBError::invalid_path(
                    worktree_path.to_path_buf(),
                    "Mount point path has no parent directory",
                )
            })?
            .join(".git2db-overlay")
            .join(&id);

        let upper_dir = overlay_base.join("upper");
        let work_dir = overlay_base.join("work");

        tokio::fs::create_dir_all(&upper_dir).await.map_err(|e| {
            Git2DBError::internal(format!("Failed to create upper directory: {}", e))
        })?;
        tokio::fs::create_dir_all(&work_dir).await.map_err(|e| {
            Git2DBError::internal(format!("Failed to create work directory: {}", e))
        })?;
        tokio::fs::create_dir_all(worktree_path)
            .await
            .map_err(|e| Git2DBError::internal(format!("Failed to create mount point: {}", e)))?;

        info!(
            "Creating overlay worktree using {} backend: id={}, lower={}, mount={}",
            self.backend.name(),
            id,
            base_repo.display(),
            worktree_path.display()
        );

        // Mount using selected backend
        self.backend
            .mount(base_repo, &upper_dir, &work_dir, worktree_path)
            .await?;

        Ok(Box::new(OverlayWorktreeHandle {
            id,
            mount_point: worktree_path.to_path_buf(),
            upper_dir,
            work_dir,
            lower_dir: base_repo.to_path_buf(),
            backend: self.backend.clone(),
            created_at: Utc::now(),
        }))
    }

    fn name(&self) -> &'static str {
        self.backend.name()
    }

    fn is_available(&self) -> bool {
        self.backend.is_available()
    }

    fn capabilities(&self) -> StrategyCapabilities {
        let backend_caps = self.backend.capabilities();
        StrategyCapabilities {
            requires_privileges: backend_caps.requires_privileges,
            space_efficient: true, // Overlayfs is always space-efficient
            relative_performance: backend_caps.relative_performance,
            platforms: vec!["linux"],
            requirements: backend_caps
                .requires_binary
                .map(|b| vec![b.to_string()])
                .unwrap_or_default(),
        }
    }
}

/// Handle to an overlayfs worktree
struct OverlayWorktreeHandle {
    id: String,
    mount_point: PathBuf,
    upper_dir: PathBuf,
    work_dir: PathBuf,
    lower_dir: PathBuf,
    backend: Arc<dyn OverlayBackend>,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl WorktreeHandle for OverlayWorktreeHandle {
    fn path(&self) -> &Path {
        &self.mount_point
    }

    fn is_valid(&self) -> bool {
        self.mount_point.exists()
    }

    fn metadata(&self) -> WorktreeMetadata {
        // Calculate space saved (estimate based on upper layer size)
        let space_saved = self.calculate_space_saved();

        WorktreeMetadata {
            strategy_name: format!("overlayfs-{}", self.backend.name()),
            created_at: self.created_at,
            space_saved_bytes: Some(space_saved),
            backend_info: Some(format!("backend: {}, id: {}", self.backend.name(), self.id)),
            read_only: false,
        }
    }

    fn cleanup(&self) -> Git2DBResult<()> {
        // Cleanup is handled in Drop
        Ok(())
    }
}

impl OverlayWorktreeHandle {
    /// Calculate approximate space saved by using overlayfs
    fn calculate_space_saved(&self) -> u64 {
        // Estimate: base model size - upper layer size
        let lower_size = Self::dir_size(&self.lower_dir).unwrap_or(0);
        let upper_size = Self::dir_size(&self.upper_dir).unwrap_or(0);

        lower_size.saturating_sub(upper_size)
    }

    /// Calculate directory size (simple estimation)
    fn dir_size(path: &Path) -> Option<u64> {
        std::process::Command::new("du")
            .arg("-sb")
            .arg(path)
            .output()
            .ok()
            .and_then(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
            })
    }
}

impl Drop for OverlayWorktreeHandle {
    fn drop(&mut self) {
        // Best-effort cleanup
        let mount_point = self.mount_point.clone();
        let backend = self.backend.clone();
        let upper_dir = self.upper_dir.clone();
        let work_dir = self.work_dir.clone();
        let id = self.id.clone();

        tokio::spawn(async move {
            info!("Cleaning up overlay worktree {}", id);

            // Unmount
            if let Err(e) = backend.unmount(&mount_point).await {
                warn!("Failed to unmount {} in drop: {}", mount_point.display(), e);
            }

            // Remove overlay directories
            let _ = tokio::fs::remove_dir_all(&upper_dir).await;
            let _ = tokio::fs::remove_dir_all(&work_dir).await;
            let _ = tokio::fs::remove_dir_all(&mount_point).await;
        });
    }
}

/// Select the best available backend
///
/// Priority order:
/// 1. FUSE overlayfs - best compatibility, no privileges needed
/// 2. User namespace overlayfs - good performance, no privileges
/// 3. Kernel overlayfs - best performance, requires privileges
pub fn select_best_backend() -> Arc<dyn OverlayBackend> {
    // Try FUSE first - best compatibility
    let fuse = FuseBackend::new();
    if fuse.is_available() {
        info!("Selected fuse-overlayfs backend");
        return Arc::new(fuse);
    }

    // Try user namespace - good performance, no privileges
    let userns = UserNamespaceBackend::new();
    if userns.is_available() {
        info!("Selected user namespace backend");
        return Arc::new(userns);
    }

    // Try kernel overlayfs - best performance
    let kernel = KernelBackend::new();
    if kernel.is_available() {
        warn!("Selected kernel overlayfs backend (requires privileges)");
        return Arc::new(kernel);
    }

    // No backend available - return FUSE (will fail at mount time with helpful message)
    Arc::new(FuseBackend::new())
}

/// Check if any overlayfs backend is available
pub fn is_available() -> bool {
    FuseBackend::new().is_available()
        || UserNamespaceBackend::new().is_available()
        || KernelBackend::new().is_available()
}

/// Get available backends
pub fn available_backends() -> Vec<(&'static str, BackendCapabilities)> {
    let backends: Vec<Box<dyn OverlayBackend>> = vec![
        Box::new(FuseBackend::new()),
        Box::new(UserNamespaceBackend::new()),
        Box::new(KernelBackend::new()),
    ];

    backends
        .into_iter()
        .filter(|b| b.is_available())
        .map(|b| (b.name(), b.capabilities()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backends = available_backends();
        println!("Available backends:");
        for (name, caps) in backends {
            println!(
                "  {} - privileges: {}, binary: {:?}, perf: {}",
                name, caps.requires_privileges, caps.requires_binary, caps.relative_performance
            );
        }
    }

    #[test]
    fn test_strategy_available() {
        let strategy = OverlayWorktreeStrategy::new();
        println!("Overlayfs available: {}", strategy.is_available());
        println!("Using backend: {}", strategy.name());
    }
}
