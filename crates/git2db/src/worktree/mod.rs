//! Worktree management with pluggable strategies
//!
//! This module provides worktree creation with multiple strategies:
//! - Git worktree (always available, branch isolation via libgit2)
//! - Overlayfs worktree (Linux-only, space-efficient CoW)
//!
//! Strategies are automatically selected based on platform and availability,
//! with graceful fallback to Git worktrees.

mod git_strategy;
mod strategy;

#[cfg(feature = "overlayfs")]
pub mod overlay;

pub use git_strategy::GitWorktreeStrategy;
pub use strategy::{StrategyCapabilities, WorktreeHandle, WorktreeMetadata, WorktreeStrategy};

#[cfg(feature = "overlayfs")]
pub use overlay::{
    available_backends, is_available as overlayfs_available, select_best_backend,
    BackendCapabilities, FuseBackend, KernelBackend, OverlayBackend, OverlayWorktreeStrategy,
    UserNamespaceBackend,
};

use std::sync::Arc;
use tracing::info;

/// Select the best available worktree strategy for the current platform
///
/// Priority order:
/// 1. Overlayfs (if feature enabled and available)
/// 2. Git worktree (always available fallback)
pub fn select_best_strategy() -> Arc<dyn WorktreeStrategy> {
    #[cfg(feature = "overlayfs")]
    {
        // Try overlayfs first on Linux
        let overlay = OverlayWorktreeStrategy::new();
        if overlay.is_available() {
            info!("Selected overlayfs worktree strategy ({})", overlay.name());
            return Arc::new(overlay);
        }
    }

    // Fall back to Git worktree
    info!("Selected git worktree strategy");
    Arc::new(GitWorktreeStrategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_selection() {
        let strategy = select_best_strategy();
        assert!(strategy.is_available());
        println!("Selected strategy: {}", strategy.name());
    }

    #[test]
    fn test_git_strategy_always_available() {
        let strategy = GitWorktreeStrategy;
        assert!(strategy.is_available());
        assert_eq!(strategy.name(), "git-worktree");
    }

    #[cfg(all(feature = "overlayfs", target_os = "linux"))]
    #[test]
    fn test_overlayfs_detection() {
        if overlayfs_available() {
            println!("Overlayfs is available on this system");
            let strategy = OverlayWorktreeStrategy::new();
            println!("Selected backend: {}", strategy.name());
        } else {
            println!("Overlayfs not available, will use git worktree");
        }
    }
}
