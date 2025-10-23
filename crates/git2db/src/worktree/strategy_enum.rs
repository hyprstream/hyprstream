//! Explicit worktree strategy enumeration
//!
//! This module provides an explicit, user-friendly enum for selecting worktree strategies.
//! Each variant clearly indicates the technology being used, avoiding abstract concepts.

use crate::errors::{Git2DBError, Git2DBResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

use super::{GitWorktreeStrategy, WorktreeStrategy};

#[cfg(feature = "overlayfs")]
use super::overlay::{OverlayWorktreeStrategy, select_best_backend, BackendType};

/// Explicit worktree strategy selection
///
/// Each variant clearly identifies the technology being used.
/// No abstract concepts like "OptimizeSpace" or "Compatible".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorktreeStrategyType {
    /// Automatically select the best available strategy
    /// Priority: Overlayfs (if available) -> Git
    Automatic,

    /// Use native git worktrees (always available)
    /// - Works on all platforms
    /// - Standard disk usage
    /// - No special requirements
    Git,

    /// Use Linux overlayfs for copy-on-write
    /// - Linux-only
    /// - ~80% disk space savings
    /// - Automatically selects best backend (kernel/userns/FUSE)
    #[cfg(feature = "overlayfs")]
    Overlayfs,

    /// Use overlayfs with explicit kernel backend
    /// - Requires CAP_SYS_ADMIN or root
    /// - Best performance
    #[cfg(feature = "overlayfs")]
    OverlayfsKernel,

    /// Use overlayfs with user namespace backend
    /// - No special privileges required
    /// - Good performance
    /// - Requires unprivileged userns support
    #[cfg(feature = "overlayfs")]
    OverlayfsUserns,

    /// Use overlayfs with FUSE backend (fuse-overlayfs)
    /// - No special privileges required
    /// - Moderate performance
    /// - Requires fuse-overlayfs binary
    #[cfg(feature = "overlayfs")]
    OverlayfsFuse,

    /// Try overlayfs, fallback to git if unavailable
    /// - Attempts overlayfs first for space savings
    /// - Falls back to git worktrees if overlayfs fails
    #[cfg(feature = "overlayfs")]
    PreferOverlayfs,

    /// Future: Use hardlinks for space efficiency
    /// - Not yet implemented
    /// - Would provide space savings on all platforms
    #[cfg(feature = "hardlink")]
    Hardlink,

    /// Future: Try hardlinks, fallback to git if unavailable
    #[cfg(feature = "hardlink")]
    PreferHardlink,

    /// Future: Use reflinks (btrfs/XFS CoW)
    /// - Not yet implemented
    /// - Requires btrfs or XFS filesystem
    #[cfg(feature = "reflink")]
    Reflink,

    /// Future: Try reflinks, fallback to git if unavailable
    #[cfg(feature = "reflink")]
    PreferReflink,
}

impl Default for WorktreeStrategyType {
    fn default() -> Self {
        Self::Automatic
    }
}

impl fmt::Display for WorktreeStrategyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Automatic => write!(f, "automatic"),
            Self::Git => write!(f, "git"),

            #[cfg(feature = "overlayfs")]
            Self::Overlayfs => write!(f, "overlayfs"),
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsKernel => write!(f, "overlayfs-kernel"),
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsUserns => write!(f, "overlayfs-userns"),
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsFuse => write!(f, "overlayfs-fuse"),
            #[cfg(feature = "overlayfs")]
            Self::PreferOverlayfs => write!(f, "prefer-overlayfs"),

            #[cfg(feature = "hardlink")]
            Self::Hardlink => write!(f, "hardlink"),
            #[cfg(feature = "hardlink")]
            Self::PreferHardlink => write!(f, "prefer-hardlink"),

            #[cfg(feature = "reflink")]
            Self::Reflink => write!(f, "reflink"),
            #[cfg(feature = "reflink")]
            Self::PreferReflink => write!(f, "prefer-reflink"),
        }
    }
}

impl WorktreeStrategyType {
    /// Parse from string (case-insensitive, accepts various formats)
    pub fn from_str_lenient(s: &str) -> Option<Self> {
        let s = s.to_lowercase().replace('-', "_");
        match s.as_str() {
            "auto" | "automatic" => Some(Self::Automatic),
            "git" | "git_worktree" | "worktree" => Some(Self::Git),

            #[cfg(feature = "overlayfs")]
            "overlayfs" | "overlay" => Some(Self::Overlayfs),
            #[cfg(feature = "overlayfs")]
            "overlayfs_kernel" | "overlay_kernel" | "kernel" => Some(Self::OverlayfsKernel),
            #[cfg(feature = "overlayfs")]
            "overlayfs_userns" | "overlay_userns" | "userns" | "user_namespace" => Some(Self::OverlayfsUserns),
            #[cfg(feature = "overlayfs")]
            "overlayfs_fuse" | "overlay_fuse" | "fuse" | "fuse_overlayfs" => Some(Self::OverlayfsFuse),
            #[cfg(feature = "overlayfs")]
            "prefer_overlayfs" | "prefer_overlay" => Some(Self::PreferOverlayfs),

            #[cfg(feature = "hardlink")]
            "hardlink" | "hard_link" => Some(Self::Hardlink),
            #[cfg(feature = "hardlink")]
            "prefer_hardlink" | "prefer_hard_link" => Some(Self::PreferHardlink),

            #[cfg(feature = "reflink")]
            "reflink" | "ref_link" => Some(Self::Reflink),
            #[cfg(feature = "reflink")]
            "prefer_reflink" | "prefer_ref_link" => Some(Self::PreferReflink),

            _ => None,
        }
    }

    /// Check if this strategy is available on the current platform
    pub fn is_available(&self) -> bool {
        match self {
            Self::Automatic | Self::Git => true,

            #[cfg(feature = "overlayfs")]
            Self::Overlayfs | Self::PreferOverlayfs => {
                super::overlay::is_available()
            }

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsKernel => {
                super::overlay::KernelBackend.is_available()
            }

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsUserns => {
                super::overlay::UserNamespaceBackend.is_available()
            }

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsFuse => {
                super::overlay::FuseBackend.is_available()
            }

            #[cfg(feature = "hardlink")]
            Self::Hardlink | Self::PreferHardlink => false, // Not implemented yet

            #[cfg(feature = "reflink")]
            Self::Reflink | Self::PreferReflink => false, // Not implemented yet
        }
    }

    /// Get availability error message if strategy is not available
    pub fn availability_error(&self) -> Option<String> {
        if self.is_available() {
            return None;
        }

        Some(match self {
            #[cfg(feature = "overlayfs")]
            Self::Overlayfs => "Overlayfs is not available on this system".to_string(),

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsKernel => "Overlayfs kernel backend requires CAP_SYS_ADMIN or root".to_string(),

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsUserns => "Overlayfs user namespace backend requires unprivileged userns support".to_string(),

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsFuse => "Overlayfs FUSE backend requires fuse-overlayfs binary".to_string(),

            #[cfg(feature = "hardlink")]
            Self::Hardlink => "Hardlink strategy is not yet implemented".to_string(),

            #[cfg(feature = "reflink")]
            Self::Reflink => "Reflink strategy is not yet implemented".to_string(),

            _ => "Strategy is not available".to_string(),
        })
    }

    /// Create the actual WorktreeStrategy implementation
    pub fn create_strategy(&self) -> Git2DBResult<Arc<dyn WorktreeStrategy>> {
        match self {
            Self::Automatic => {
                // Try overlayfs first if available
                #[cfg(feature = "overlayfs")]
                {
                    if super::overlay::is_available() {
                        let strategy = OverlayWorktreeStrategy::new();
                        return Ok(Arc::new(strategy));
                    }
                }

                // Fallback to git
                Ok(Arc::new(GitWorktreeStrategy::new()))
            }

            Self::Git => Ok(Arc::new(GitWorktreeStrategy::new())),

            #[cfg(feature = "overlayfs")]
            Self::Overlayfs => {
                if !super::overlay::is_available() {
                    return Err(Git2DBError::Config(
                        "Overlayfs is not available on this system".to_string()
                    ));
                }
                Ok(Arc::new(OverlayWorktreeStrategy::new()))
            }

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsKernel => {
                let backend = super::overlay::KernelBackend;
                if !backend.is_available() {
                    return Err(Git2DBError::Config(
                        "Overlayfs kernel backend is not available (requires CAP_SYS_ADMIN or root)".to_string()
                    ));
                }
                Ok(Arc::new(OverlayWorktreeStrategy::with_backend(BackendType::Kernel)))
            }

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsUserns => {
                let backend = super::overlay::UserNamespaceBackend;
                if !backend.is_available() {
                    return Err(Git2DBError::Config(
                        "Overlayfs user namespace backend is not available".to_string()
                    ));
                }
                Ok(Arc::new(OverlayWorktreeStrategy::with_backend(BackendType::UserNamespace)))
            }

            #[cfg(feature = "overlayfs")]
            Self::OverlayfsFuse => {
                let backend = super::overlay::FuseBackend;
                if !backend.is_available() {
                    return Err(Git2DBError::Config(
                        "Overlayfs FUSE backend is not available (fuse-overlayfs not found)".to_string()
                    ));
                }
                Ok(Arc::new(OverlayWorktreeStrategy::with_backend(BackendType::Fuse)))
            }

            #[cfg(feature = "overlayfs")]
            Self::PreferOverlayfs => {
                // Try overlayfs, but fallback to git if not available
                if super::overlay::is_available() {
                    Ok(Arc::new(OverlayWorktreeStrategy::new()))
                } else {
                    Ok(Arc::new(GitWorktreeStrategy::new()))
                }
            }

            #[cfg(feature = "hardlink")]
            Self::Hardlink => {
                Err(Git2DBError::Config(
                    "Hardlink strategy is not yet implemented".to_string()
                ))
            }

            #[cfg(feature = "hardlink")]
            Self::PreferHardlink => {
                // When implemented, try hardlink first
                // For now, fallback to git
                Ok(Arc::new(GitWorktreeStrategy::new()))
            }

            #[cfg(feature = "reflink")]
            Self::Reflink => {
                Err(Git2DBError::Config(
                    "Reflink strategy is not yet implemented".to_string()
                ))
            }

            #[cfg(feature = "reflink")]
            Self::PreferReflink => {
                // When implemented, try reflink first
                // For now, fallback to git
                Ok(Arc::new(GitWorktreeStrategy::new()))
            }
        }
    }

    /// Get a human-readable description of this strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::Automatic => "Automatically select best available strategy",
            Self::Git => "Native git worktrees (standard disk usage)",

            #[cfg(feature = "overlayfs")]
            Self::Overlayfs => "Linux overlayfs copy-on-write (~80% space savings)",
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsKernel => "Overlayfs with kernel backend (requires root/CAP_SYS_ADMIN)",
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsUserns => "Overlayfs with user namespace backend (no privileges required)",
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsFuse => "Overlayfs with FUSE backend (requires fuse-overlayfs)",
            #[cfg(feature = "overlayfs")]
            Self::PreferOverlayfs => "Try overlayfs for space savings, fallback to git if unavailable",

            #[cfg(feature = "hardlink")]
            Self::Hardlink => "Hardlinks for space efficiency (not yet implemented)",
            #[cfg(feature = "hardlink")]
            Self::PreferHardlink => "Try hardlinks, fallback to git if unavailable",

            #[cfg(feature = "reflink")]
            Self::Reflink => "Copy-on-write reflinks (btrfs/XFS, not yet implemented)",
            #[cfg(feature = "reflink")]
            Self::PreferReflink => "Try reflinks, fallback to git if unavailable",
        }
    }

    /// List all available strategies on the current platform
    pub fn available_strategies() -> Vec<Self> {
        let all = vec![
            Self::Automatic,
            Self::Git,
            #[cfg(feature = "overlayfs")]
            Self::Overlayfs,
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsKernel,
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsUserns,
            #[cfg(feature = "overlayfs")]
            Self::OverlayfsFuse,
            #[cfg(feature = "overlayfs")]
            Self::PreferOverlayfs,
        ];

        all.into_iter().filter(|s| s.is_available()).collect()
    }
}

/// Builder for creating strategies with validation
pub struct WorktreeStrategyBuilder {
    strategy_type: WorktreeStrategyType,
    require_available: bool,
    log_selection: bool,
}

impl WorktreeStrategyBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            strategy_type: WorktreeStrategyType::Automatic,
            require_available: false,
            log_selection: true,
        }
    }

    /// Set the strategy type
    pub fn strategy(mut self, strategy: WorktreeStrategyType) -> Self {
        self.strategy_type = strategy;
        self
    }

    /// Require the strategy to be available (fail if not)
    pub fn require_available(mut self, require: bool) -> Self {
        self.require_available = require;
        self
    }

    /// Enable/disable logging of strategy selection
    pub fn log_selection(mut self, log: bool) -> Self {
        self.log_selection = log;
        self
    }

    /// Build the strategy
    pub fn build(self) -> Git2DBResult<Arc<dyn WorktreeStrategy>> {
        use tracing::{info, warn};

        // Check availability if required
        if self.require_available && !self.strategy_type.is_available() {
            if let Some(error) = self.strategy_type.availability_error() {
                return Err(Git2DBError::Config(error));
            }
        }

        // Create the strategy
        let result = self.strategy_type.create_strategy();

        // Log the selection
        if self.log_selection {
            match &result {
                Ok(strategy) => {
                    info!(
                        "Selected worktree strategy: {} ({})",
                        strategy.name(),
                        self.strategy_type.description()
                    );

                    let caps = strategy.capabilities();
                    if caps.space_efficient {
                        info!("Strategy provides ~80% disk space savings");
                    }
                    if caps.requires_privileges {
                        warn!("Strategy requires elevated privileges");
                    }
                }
                Err(e) => {
                    warn!("Failed to create strategy {}: {}", self.strategy_type, e);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_parsing() {
        assert_eq!(
            WorktreeStrategyType::from_str_lenient("auto"),
            Some(WorktreeStrategyType::Automatic)
        );
        assert_eq!(
            WorktreeStrategyType::from_str_lenient("git-worktree"),
            Some(WorktreeStrategyType::Git)
        );

        #[cfg(feature = "overlayfs")]
        {
            assert_eq!(
                WorktreeStrategyType::from_str_lenient("overlayfs"),
                Some(WorktreeStrategyType::Overlayfs)
            );
            assert_eq!(
                WorktreeStrategyType::from_str_lenient("overlay-kernel"),
                Some(WorktreeStrategyType::OverlayfsKernel)
            );
            assert_eq!(
                WorktreeStrategyType::from_str_lenient("prefer-overlay"),
                Some(WorktreeStrategyType::PreferOverlayfs)
            );
        }
    }

    #[test]
    fn test_strategy_availability() {
        // Git and Automatic should always be available
        assert!(WorktreeStrategyType::Git.is_available());
        assert!(WorktreeStrategyType::Automatic.is_available());
    }

    #[test]
    fn test_strategy_creation() {
        // Git strategy should always be creatable
        let git_strategy = WorktreeStrategyType::Git.create_strategy();
        assert!(git_strategy.is_ok());

        // Automatic should always succeed (falls back to git)
        let auto_strategy = WorktreeStrategyType::Automatic.create_strategy();
        assert!(auto_strategy.is_ok());
    }

    #[test]
    fn test_builder() {
        // Builder with git strategy should always work
        let strategy = WorktreeStrategyBuilder::new()
            .strategy(WorktreeStrategyType::Git)
            .require_available(true)
            .build();
        assert!(strategy.is_ok());
    }
}