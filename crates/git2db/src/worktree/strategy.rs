//! Worktree strategy trait and types
//!
//! Defines the interface for pluggable worktree creation strategies.

use crate::errors::Git2DBResult;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::path::Path;

/// Strategy for creating isolated worktrees
///
/// Implementations provide different approaches to worktree isolation:
/// - Git worktrees: Branch isolation via libgit2
/// - Overlayfs: Space-efficient copy-on-write
/// - Future: Container-based, reflink-based, etc.
#[async_trait]
pub trait WorktreeStrategy: Send + Sync {
    /// Create a worktree at the specified path
    ///
    /// # Arguments
    /// * `base_repo` - Path to the base repository
    /// * `worktree_path` - Path where worktree should be created
    /// * `branch` - Branch name for the worktree
    ///
    /// # Returns
    /// A handle to the created worktree that will be automatically cleaned up on drop
    async fn create(
        &self,
        base_repo: &Path,
        worktree_path: &Path,
        branch: &str,
    ) -> Git2DBResult<Box<dyn WorktreeHandle>>;

    /// Name of this strategy (for logging/diagnostics)
    fn name(&self) -> &'static str;

    /// Check if this strategy is available on the current system
    fn is_available(&self) -> bool;

    /// Get strategy capabilities
    fn capabilities(&self) -> StrategyCapabilities;
}

/// Handle to a created worktree
///
/// Provides access to the worktree and ensures cleanup on drop.
/// All implementations should clean up resources (unmount, remove files, etc.)
/// in their Drop implementation.
pub trait WorktreeHandle: Send + Sync {
    /// Get the worktree path
    fn path(&self) -> &Path;

    /// Check if worktree is still valid
    fn is_valid(&self) -> bool;

    /// Get strategy-specific metadata
    fn metadata(&self) -> WorktreeMetadata;

    /// Cleanup the worktree
    ///
    /// This is called automatically by Drop, but can be called explicitly
    /// for error handling.
    fn cleanup(&self) -> Git2DBResult<()>;
}

/// Strategy capabilities for auto-selection
#[derive(Debug, Clone)]
pub struct StrategyCapabilities {
    /// Requires elevated privileges (CAP_SYS_ADMIN, root, etc.)
    pub requires_privileges: bool,

    /// Space-efficient (CoW, deduplication, etc.)
    pub space_efficient: bool,

    /// Relative performance compared to baseline (1.0 = same as git worktree)
    pub relative_performance: f32,

    /// Supported platforms
    pub platforms: Vec<&'static str>,

    /// Additional requirements (binaries, kernel features, etc.)
    pub requirements: Vec<String>,
}

impl Default for StrategyCapabilities {
    fn default() -> Self {
        Self {
            requires_privileges: false,
            space_efficient: false,
            relative_performance: 1.0,
            platforms: vec!["linux", "macos", "windows"],
            requirements: Vec::new(),
        }
    }
}

/// Metadata about a created worktree
#[derive(Debug, Clone)]
pub struct WorktreeMetadata {
    /// Strategy that created this worktree
    pub strategy_name: String,

    /// When the worktree was created
    pub created_at: DateTime<Utc>,

    /// Estimated space saved compared to full copy (for CoW strategies)
    pub space_saved_bytes: Option<u64>,

    /// Backend-specific information (e.g., "fuse-overlayfs", "git-worktree")
    pub backend_info: Option<String>,

    /// Whether this worktree is read-only
    pub read_only: bool,
}

impl Default for WorktreeMetadata {
    fn default() -> Self {
        Self {
            strategy_name: "unknown".to_string(),
            created_at: Utc::now(),
            space_saved_bytes: None,
            backend_info: None,
            read_only: false,
        }
    }
}
