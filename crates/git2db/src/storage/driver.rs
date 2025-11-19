//! Core driver trait and types

use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

/// Storage driver selection (Docker's graphdriver pattern)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageDriver {
    Auto,
    Overlay2,
    Reflink,
    Vfs,
}

impl fmt::Display for StorageDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Overlay2 => write!(f, "overlay2"),
            Self::Btrfs => write!(f, "btrfs"),
            Self::Reflink => write!(f, "reflink"),
            Self::Hardlink => write!(f, "hardlink"),
            Self::Vfs => write!(f, "vfs"),
        }
    }
}

impl std::str::FromStr for StorageDriver {
    type Err = DriverError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" | "automatic" => Ok(Self::Auto),
            "overlay2" | "overlayfs" => Ok(Self::Overlay2),
            "btrfs" => Ok(Self::Btrfs),
            "reflink" => Ok(Self::Reflink),
            "hardlink" => Ok(Self::Hardlink),
            "vfs" | "none" => Ok(Self::Vfs),
            _ => Err(DriverError::UnknownDriver(s.to_string())),
        }
    }
}

/// Options for creating a worktree with a driver
#[derive(Debug, Clone)]
pub struct DriverOpts {
    /// Base repository path (lower layer for CoW drivers)
    pub base_repo: PathBuf,

    /// Target worktree path (where to mount/create)
    pub worktree_path: PathBuf,

    /// Git ref specification (branch, commit SHA, tag, HEAD~3, etc.)
    /// Examples: "main", "a1b2c3d4", "v1.0.0", "HEAD~3", "origin/main"
    pub ref_spec: String,

    /// Optional: Force specific backend (for drivers with multiple backends)
    pub force_backend: Option<String>,
}

/// Driver capabilities
#[derive(Debug, Clone)]
pub struct DriverCapabilities {
    /// Uses Copy-on-Write
    pub copy_on_write: bool,

    /// Estimated space savings percentage (0-100)
    pub space_savings_percent: u8,

    /// Requires elevated privileges
    pub requires_privileges: bool,

    /// Platform support
    pub platforms: Vec<&'static str>,

    /// Required external binaries
    pub required_binaries: Vec<&'static str>,

    /// Performance relative to vfs (1.0 = same, >1.0 = faster)
    pub relative_performance: f32,
}

/// Driver-specific errors
#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    #[error("Unknown driver: {0}")]
    UnknownDriver(String),

    #[error("Driver not available: {0}")]
    NotAvailable(String),

    #[error("Driver operation failed: {0}")]
    OperationFailed(String),

    #[error("Git2DB error: {0}")]
    Git2DB(#[from] Git2DBError),
}

impl From<DriverError> for Git2DBError {
    fn from(e: DriverError) -> Self {
        Git2DBError::internal(e.to_string())
    }
}

/// Storage driver trait (Docker's graphdriver interface)
///
/// All storage drivers must implement this trait to provide worktree
/// creation with optional storage optimization.
#[async_trait]
pub trait Driver: Send + Sync {
    /// Get driver name (e.g., "overlay2", "vfs")
    fn name(&self) -> &'static str;

    /// Check if this driver is available on the current system
    fn is_available(&self) -> bool;

    /// Get driver capabilities
    fn capabilities(&self) -> DriverCapabilities;

    /// Create a worktree using this driver
    ///
    /// This creates a git worktree, potentially with storage optimization
    /// layers underneath (overlayfs, reflinks, etc.).
    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle>;
}

/// Async cleanup function type
pub type AsyncCleanupFn = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Git2DBResult<()>> + Send>> + Send>;

/// Handle to a created worktree
///
/// Provides access to the worktree and handles cleanup on drop.
/// Implements the WorktreeHandle trait from worktree module.
pub struct WorktreeHandle {
    /// Path to the worktree
    pub path: PathBuf,

    /// Driver that created this worktree
    pub driver_name: String,

    /// Async cleanup function, wrapped in Arc<Mutex> for Sync
    cleanup: Option<Arc<Mutex<Option<AsyncCleanupFn>>>>,
}

impl WorktreeHandle {
    /// Create a new worktree handle
    pub fn new(path: PathBuf, driver_name: String) -> Self {
        Self {
            path,
            driver_name,
            cleanup: None,
        }
    }

    /// Create with async cleanup function
    pub fn with_cleanup<F, Fut>(path: PathBuf, driver_name: String, cleanup: F) -> Self
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Git2DBResult<()>> + Send + 'static,
    {
        let cleanup_fn: AsyncCleanupFn = Box::new(move || Box::pin(cleanup()));
        Self {
            path,
            driver_name,
            cleanup: Some(Arc::new(Mutex::new(Some(cleanup_fn)))),
        }
    }

    /// Get the driver name
    pub fn driver_name(&self) -> &str {
        &self.driver_name
    }
}

// Implement the WorktreeHandle trait from worktree module
#[async_trait]
impl crate::worktree::WorktreeHandle for WorktreeHandle {
    fn path(&self) -> &Path {
        &self.path
    }

    fn is_valid(&self) -> bool {
        self.path.exists()
    }

    fn metadata(&self) -> crate::worktree::WorktreeMetadata {
        crate::worktree::WorktreeMetadata {
            strategy_name: format!("driver-{}", self.driver_name),
            created_at: chrono::Utc::now(),
            space_saved_bytes: None, // Drivers don't track this yet
            backend_info: Some(format!("driver: {}", self.driver_name)),
            read_only: false,
        }
    }

    async fn cleanup(&mut self) -> Git2DBResult<()> {
        if let Some(cleanup_arc) = self.cleanup.take() {
            let cleanup_fn = {
                if let Ok(mut cleanup_guard) = cleanup_arc.lock() {
                    cleanup_guard.take()
                } else {
                    None
                }
            };

            if let Some(cleanup_fn) = cleanup_fn {
                cleanup_fn().await?;
            }
        }
        Ok(())
    }
}

impl Drop for WorktreeHandle {
    fn drop(&mut self) {
        if self.cleanup.is_some() {
            tracing::warn!(
                "WorktreeHandle at {} dropped without calling cleanup() - \
                 resources may leak (overlayfs mounts, temporary directories)",
                self.path.display()
            );
        }
    }
}

impl fmt::Debug for WorktreeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorktreeHandle")
            .field("path", &self.path)
            .field("driver_name", &self.driver_name)
            .finish()
    }
}
