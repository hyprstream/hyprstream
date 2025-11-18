//! Worktree management with pluggable strategies
//!
//! This module provides worktree creation with Git worktree strategy.
//! Overlay2-based worktrees are now handled at the storage driver level
//! in storage/overlay2.rs for better separation of concerns.

mod git_strategy;
mod strategy;

pub use git_strategy::GitWorktreeStrategy;
pub use strategy::{StrategyCapabilities, WorktreeHandle, WorktreeMetadata, WorktreeStrategy};

use std::sync::Arc;
use tracing::info;

/// Select the best available worktree strategy for the current platform
///
/// Currently returns Git worktree strategy. Overlay2 driver is selected
/// at the storage driver level for better architectural separation.
pub fn select_best_strategy() -> Arc<dyn WorktreeStrategy> {
    // Git worktree is always available
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
}
