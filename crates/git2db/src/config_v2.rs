//! Enhanced configuration with explicit worktree strategies
//!
//! This module provides the new configuration structure that uses
//! explicit worktree strategy names instead of abstract concepts.

use crate::worktree::strategy_enum::WorktreeStrategyType;
use serde::{Deserialize, Serialize};

/// Enhanced worktree configuration with explicit strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeConfigV2 {
    /// Explicit strategy selection
    ///
    /// Possible values:
    /// - `automatic`: Select best available (default)
    /// - `git`: Native git worktrees
    /// - `overlayfs`: Linux overlayfs (auto-select backend)
    /// - `overlayfs-kernel`: Overlayfs with kernel backend
    /// - `overlayfs-userns`: Overlayfs with user namespace
    /// - `overlayfs-fuse`: Overlayfs with FUSE backend
    /// - `prefer-overlayfs`: Try overlayfs, fallback to git
    /// - `hardlink`: Hardlinks (future)
    /// - `prefer-hardlink`: Try hardlinks, fallback to git (future)
    /// - `reflink`: Reflinks/CoW (future)
    /// - `prefer-reflink`: Try reflinks, fallback to git (future)
    #[serde(default)]
    pub strategy: WorktreeStrategyType,

    /// Fail if selected strategy is not available
    ///
    /// When true, operations will fail if the selected strategy cannot be used.
    /// When false, will attempt fallback strategies (for "prefer-*" variants).
    #[serde(default)]
    pub require_strategy: bool,

    /// Enable detailed logging of strategy selection
    #[serde(default = "default_true")]
    pub log_selection: bool,

    /// Platform-specific overrides
    ///
    /// Allow different strategies per platform
    #[serde(default)]
    pub platform_overrides: PlatformOverrides,

    /// Advanced tuning options
    #[serde(default)]
    pub advanced: AdvancedOptions,
}

/// Platform-specific strategy overrides
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlatformOverrides {
    /// Strategy to use on Linux
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linux: Option<WorktreeStrategyType>,

    /// Strategy to use on macOS
    #[serde(skip_serializing_if = "Option::is_none")]
    pub macos: Option<WorktreeStrategyType>,

    /// Strategy to use on Windows
    #[serde(skip_serializing_if = "Option::is_none")]
    pub windows: Option<WorktreeStrategyType>,
}

/// Advanced tuning options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOptions {
    /// Maximum time to wait for overlayfs mount (milliseconds)
    #[serde(default = "default_mount_timeout")]
    pub mount_timeout_ms: u64,

    /// Retry overlayfs mount on failure
    #[serde(default = "default_true")]
    pub retry_mount: bool,

    /// Number of mount retry attempts
    #[serde(default = "default_mount_retries")]
    pub mount_retries: u32,

    /// Enable experimental features (future strategies)
    #[serde(default)]
    pub experimental: bool,

    /// Custom mount options for overlayfs (advanced users only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlayfs_options: Option<String>,
}

impl Default for WorktreeConfigV2 {
    fn default() -> Self {
        Self {
            strategy: WorktreeStrategyType::Automatic,
            require_strategy: false,
            log_selection: true,
            platform_overrides: PlatformOverrides::default(),
            advanced: AdvancedOptions::default(),
        }
    }
}

impl Default for AdvancedOptions {
    fn default() -> Self {
        Self {
            mount_timeout_ms: 5000,
            retry_mount: true,
            mount_retries: 3,
            experimental: false,
            overlayfs_options: None,
        }
    }
}

impl WorktreeConfigV2 {
    /// Create a configuration that requires overlayfs
    pub fn overlayfs_required() -> Self {
        Self {
            strategy: WorktreeStrategyType::Overlayfs,
            require_strategy: true,
            ..Default::default()
        }
    }

    /// Create a configuration that prefers overlayfs but falls back to git
    pub fn overlayfs_preferred() -> Self {
        Self {
            strategy: WorktreeStrategyType::PreferOverlayfs,
            require_strategy: false,
            ..Default::default()
        }
    }

    /// Create a configuration that always uses git worktrees
    pub fn git_only() -> Self {
        Self {
            strategy: WorktreeStrategyType::Git,
            require_strategy: true,
            ..Default::default()
        }
    }

    /// Get the effective strategy for the current platform
    pub fn effective_strategy(&self) -> WorktreeStrategyType {
        // Check platform overrides first
        #[cfg(target_os = "linux")]
        if let Some(strategy) = &self.platform_overrides.linux {
            return strategy.clone();
        }

        #[cfg(target_os = "macos")]
        if let Some(strategy) = &self.platform_overrides.macos {
            return strategy.clone();
        }

        #[cfg(target_os = "windows")]
        if let Some(strategy) = &self.platform_overrides.windows {
            return strategy.clone();
        }

        // Use the default strategy
        self.strategy.clone()
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        let effective = self.effective_strategy();

        // Check if required strategy is available
        if self.require_strategy && !effective.is_available() {
            if let Some(error) = effective.availability_error() {
                return Err(format!("Required strategy not available: {}", error));
            }
        }

        // Validate mount timeout
        if self.advanced.mount_timeout_ms == 0 {
            return Err("Mount timeout must be greater than 0".to_string());
        }

        // Validate mount retries
        if self.advanced.mount_retries > 10 {
            return Err("Mount retries should not exceed 10".to_string());
        }

        Ok(())
    }
}

// Helper functions for serde defaults
fn default_true() -> bool {
    true
}

fn default_mount_timeout() -> u64 {
    5000
}

fn default_mount_retries() -> u32 {
    3
}

/// Migration helper to convert from old WorktreeConfig to new WorktreeConfigV2
pub fn migrate_config(old: &crate::config::WorktreeConfig) -> WorktreeConfigV2 {
    let strategy = if let Some(backend) = &old.backend {
        // Handle forced backend
        match backend.as_str() {
            "overlayfs" | "overlay" => WorktreeStrategyType::Overlayfs,
            "overlayfs-kernel" | "kernel" => WorktreeStrategyType::OverlayfsKernel,
            "overlayfs-userns" | "userns" => WorktreeStrategyType::OverlayfsUserns,
            "overlayfs-fuse" | "fuse" => WorktreeStrategyType::OverlayfsFuse,
            "git" | "git-worktree" => WorktreeStrategyType::Git,
            _ => WorktreeStrategyType::Automatic,
        }
    } else if old.use_overlayfs {
        if old.fallback {
            WorktreeStrategyType::PreferOverlayfs
        } else {
            WorktreeStrategyType::Overlayfs
        }
    } else {
        WorktreeStrategyType::Git
    };

    WorktreeConfigV2 {
        strategy,
        require_strategy: !old.fallback,
        log_selection: old.log,
        platform_overrides: PlatformOverrides::default(),
        advanced: AdvancedOptions::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorktreeConfigV2::default();
        assert_eq!(config.strategy, WorktreeStrategyType::Automatic);
        assert!(!config.require_strategy);
        assert!(config.log_selection);
    }

    #[test]
    fn test_overlayfs_required() {
        let config = WorktreeConfigV2::overlayfs_required();
        assert_eq!(config.strategy, WorktreeStrategyType::Overlayfs);
        assert!(config.require_strategy);
    }

    #[test]
    fn test_config_validation() {
        let mut config = WorktreeConfigV2::default();
        assert!(config.validate().is_ok());

        // Test invalid mount timeout
        config.advanced.mount_timeout_ms = 0;
        assert!(config.validate().is_err());
        config.advanced.mount_timeout_ms = 5000;

        // Test excessive retries
        config.advanced.mount_retries = 20;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_platform_overrides() {
        let mut config = WorktreeConfigV2::default();
        config.platform_overrides.linux = Some(WorktreeStrategyType::Overlayfs);
        config.platform_overrides.macos = Some(WorktreeStrategyType::Git);

        #[cfg(target_os = "linux")]
        assert_eq!(config.effective_strategy(), WorktreeStrategyType::Overlayfs);

        #[cfg(target_os = "macos")]
        assert_eq!(config.effective_strategy(), WorktreeStrategyType::Git);
    }

    #[test]
    fn test_migration() {
        use crate::config::WorktreeConfig;

        // Test overlayfs with fallback
        let old = WorktreeConfig {
            use_overlayfs: true,
            fallback: true,
            backend: None,
            log: true,
        };
        let new = migrate_config(&old);
        assert_eq!(new.strategy, WorktreeStrategyType::PreferOverlayfs);
        assert!(!new.require_strategy);

        // Test forced backend
        let old = WorktreeConfig {
            use_overlayfs: false,
            fallback: false,
            backend: Some("overlayfs-kernel".to_string()),
            log: false,
        };
        let new = migrate_config(&old);
        assert_eq!(new.strategy, WorktreeStrategyType::OverlayfsKernel);
        assert!(new.require_strategy);
    }
}