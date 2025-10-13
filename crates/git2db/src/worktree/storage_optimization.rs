//! Storage optimization strategies for git worktrees
//!
//! This module provides configuration for storage optimizations that can be applied
//! to git worktrees to reduce disk usage and improve performance.
//!
//! # Architecture
//!
//! Git worktrees are ALWAYS used as the base mechanism for creating isolated working
//! directories. Storage optimizations like overlayfs are optional layers that sit
//! UNDERNEATH the git worktree to provide benefits like:
//!
//! - **Copy-on-Write (CoW)**: ~80% disk space savings
//! - **Isolation**: Changes are isolated to an upper layer
//! - **Performance**: Faster creation and cleanup
//!
//! # Important
//!
//! These optimizations do NOT replace git worktrees - they enhance them.
//! A git worktree with overlayfs optimization is still a fully functional git worktree,
//! it just happens to be mounted on a CoW filesystem layer for efficiency.

use crate::errors::{Git2DBError, Git2DBResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Configuration for worktree storage optimization
///
/// Determines how git worktrees should be optimized for disk usage and performance.
/// The default is to enable optimizations when available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorktreeConfig {
    /// Storage optimization strategy
    #[serde(default)]
    pub optimization: StorageOptimization,

    /// Fallback behavior when optimization isn't available
    #[serde(default)]
    pub fallback: FallbackBehavior,

    /// Log optimization decisions
    #[serde(default = "default_log_enabled")]
    pub log_optimization: bool,
}

impl Default for WorktreeConfig {
    fn default() -> Self {
        Self {
            optimization: StorageOptimization::Auto,
            fallback: FallbackBehavior::Continue,
            log_optimization: true,
        }
    }
}

fn default_log_enabled() -> bool {
    true
}

/// Storage optimization strategies for git worktrees
///
/// These optimizations are applied UNDERNEATH git worktrees to reduce
/// disk usage while maintaining full git functionality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageOptimization {
    /// No optimization - standard git worktree with full disk usage
    ///
    /// Use when:
    /// - You need maximum compatibility
    /// - Performance is more important than disk space
    /// - Working with very small repositories
    None,

    /// Automatically enable best available optimization
    ///
    /// Tries optimizations in order of preference:
    /// 1. Overlayfs (CoW) on Linux - ~80% space savings
    /// 2. Future: Reflinks on btrfs/XFS
    /// 3. Future: Hardlinks where supported
    /// 4. Falls back to None if nothing available
    #[serde(alias = "automatic")]
    Auto,

    /// Copy-on-Write optimization via overlayfs (Linux only)
    ///
    /// Creates git worktree on an overlayfs mount for:
    /// - ~80% disk space savings
    /// - Fast worktree creation
    /// - Automatic cleanup
    /// - Full git functionality
    #[serde(alias = "overlayfs")]
    CopyOnWrite(CoWConfig),

    /// Future: Hardlink optimization
    ///
    /// Would provide space savings on all platforms
    /// by hardlinking unchanged files
    #[cfg(feature = "hardlink")]
    Hardlink,

    /// Future: Reflink optimization (btrfs/XFS)
    ///
    /// Would use filesystem-level CoW for space savings
    #[cfg(feature = "reflink")]
    Reflink,
}

impl StorageOptimization {
    /// Check if this optimization is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            Self::None => true, // Always available (no optimization)
            Self::Auto => true, // Always "available" (will find best option)
            Self::CopyOnWrite(config) => config.is_available(),
            #[cfg(feature = "hardlink")]
            Self::Hardlink => false, // Not yet implemented
            #[cfg(feature = "reflink")]
            Self::Reflink => false, // Not yet implemented
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "No optimization (standard git worktree)",
            Self::Auto => "Automatic optimization selection",
            Self::CopyOnWrite(_) => "Copy-on-Write via overlayfs (~80% space savings)",
            #[cfg(feature = "hardlink")]
            Self::Hardlink => "Hardlink optimization (not yet implemented)",
            #[cfg(feature = "reflink")]
            Self::Reflink => "Reflink CoW optimization (not yet implemented)",
        }
    }

    /// Get the effective optimization (resolving Auto)
    pub fn resolve(&self) -> Self {
        match self {
            Self::Auto => {
                // Try optimizations in order of preference
                let cow = CoWConfig::default();
                if cow.is_available() {
                    return Self::CopyOnWrite(cow);
                }

                // Future: Try reflinks, hardlinks, etc.

                // No optimization available
                Self::None
            }
            other => other.clone(),
        }
    }
}

impl fmt::Display for StorageOptimization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Auto => write!(f, "auto"),
            Self::CopyOnWrite(config) => write!(f, "cow:{}", config),
            #[cfg(feature = "hardlink")]
            Self::Hardlink => write!(f, "hardlink"),
            #[cfg(feature = "reflink")]
            Self::Reflink => write!(f, "reflink"),
        }
    }
}

/// Configuration for Copy-on-Write optimization
///
/// Controls how overlayfs is used to optimize git worktrees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CoWConfig {
    /// Which overlayfs backend to use
    #[serde(default)]
    pub backend: CoWBackend,

    /// Mount options for overlayfs
    #[serde(default)]
    pub mount_options: Vec<String>,

    /// Directory for overlayfs work/upper layers
    /// (defaults to system temp directory)
    pub overlay_dir: Option<String>,
}

impl Default for CoWConfig {
    fn default() -> Self {
        Self {
            backend: CoWBackend::Auto,
            mount_options: Vec::new(),
            overlay_dir: None,
        }
    }
}

impl CoWConfig {
    /// Check if CoW optimization is available
    pub fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.backend.is_available()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl fmt::Display for CoWConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.backend)
    }
}

/// Copy-on-Write backend selection
///
/// Controls which overlayfs implementation to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoWBackend {
    /// Automatically select best available backend
    ///
    /// Priority order:
    /// 1. Kernel (if privileged)
    /// 2. UserNamespace (if supported)
    /// 3. FUSE (if fuse-overlayfs available)
    Auto,

    /// Kernel overlayfs (requires CAP_SYS_ADMIN or root)
    ///
    /// - Best performance
    /// - Most stable
    /// - Requires privileges
    Kernel,

    /// User namespace overlayfs (unprivileged)
    ///
    /// - Good performance
    /// - No privileges required
    /// - Requires kernel 5.11+ with unprivileged userns
    UserNamespace,

    /// FUSE-based overlayfs
    ///
    /// - Moderate performance
    /// - No privileges required
    /// - Requires fuse-overlayfs binary
    Fuse,
}

impl CoWBackend {
    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            use super::overlay;
            match self {
                Self::Auto => {
                    overlay::KernelBackend.is_available() ||
                    overlay::UserNamespaceBackend.is_available() ||
                    overlay::FuseBackend.is_available()
                }
                Self::Kernel => overlay::KernelBackend.is_available(),
                Self::UserNamespace => overlay::UserNamespaceBackend.is_available(),
                Self::Fuse => overlay::FuseBackend.is_available(),
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Get the actual backend (resolving Auto)
    pub fn resolve(&self) -> Self {
        match self {
            Self::Auto => {
                #[cfg(target_os = "linux")]
                {
                    use super::overlay;
                    if overlay::KernelBackend.is_available() {
                        return Self::Kernel;
                    }
                    if overlay::UserNamespaceBackend.is_available() {
                        return Self::UserNamespace;
                    }
                    if overlay::FuseBackend.is_available() {
                        return Self::Fuse;
                    }
                }
                // No backend available, but this shouldn't happen
                // if is_available() was checked first
                Self::Kernel
            }
            other => other.clone(),
        }
    }
}

impl fmt::Display for CoWBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Kernel => write!(f, "kernel"),
            Self::UserNamespace => write!(f, "userns"),
            Self::Fuse => write!(f, "fuse"),
        }
    }
}

/// Fallback behavior when optimization fails
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackBehavior {
    /// Continue without optimization (use standard git worktree)
    ///
    /// This is the default - if optimization fails, we still
    /// create a functional git worktree, just without space savings.
    Continue,

    /// Fail if optimization is not available
    ///
    /// Use when space savings are critical and you'd rather
    /// fail fast than use full disk space.
    Fail,

    /// Warn and continue
    ///
    /// Like Continue, but logs a warning to make it clear
    /// that optimization was requested but not available.
    Warn,
}

impl Default for FallbackBehavior {
    fn default() -> Self {
        Self::Continue
    }
}

/// Builder for creating optimized worktrees
///
/// Provides a fluent API for configuring and creating git worktrees
/// with optional storage optimizations.
pub struct OptimizedWorktreeBuilder {
    config: WorktreeConfig,
}

impl OptimizedWorktreeBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: WorktreeConfig::default(),
        }
    }

    /// Set the optimization strategy
    pub fn optimization(mut self, optimization: StorageOptimization) -> Self {
        self.config.optimization = optimization;
        self
    }

    /// Enable Copy-on-Write optimization with default settings
    pub fn with_cow(mut self) -> Self {
        self.config.optimization = StorageOptimization::CopyOnWrite(CoWConfig::default());
        self
    }

    /// Enable Copy-on-Write with specific backend
    pub fn with_cow_backend(mut self, backend: CoWBackend) -> Self {
        self.config.optimization = StorageOptimization::CopyOnWrite(CoWConfig {
            backend,
            ..Default::default()
        });
        self
    }

    /// Disable all optimizations (standard git worktree)
    pub fn no_optimization(mut self) -> Self {
        self.config.optimization = StorageOptimization::None;
        self
    }

    /// Set fallback behavior
    pub fn fallback(mut self, fallback: FallbackBehavior) -> Self {
        self.config.fallback = fallback;
        self
    }

    /// Set whether to log optimization decisions
    pub fn log_optimization(mut self, enable: bool) -> Self {
        self.config.log_optimization = enable;
        self
    }

    /// Build the configuration
    pub fn build(self) -> WorktreeConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorktreeConfig::default();
        assert_eq!(config.optimization, StorageOptimization::Auto);
        assert_eq!(config.fallback, FallbackBehavior::Continue);
        assert!(config.log_optimization);
    }

    #[test]
    fn test_builder() {
        let config = OptimizedWorktreeBuilder::new()
            .with_cow()
            .fallback(FallbackBehavior::Warn)
            .log_optimization(false)
            .build();

        assert!(matches!(config.optimization, StorageOptimization::CopyOnWrite(_)));
        assert_eq!(config.fallback, FallbackBehavior::Warn);
        assert!(!config.log_optimization);
    }

    #[test]
    fn test_no_optimization() {
        let config = OptimizedWorktreeBuilder::new()
            .no_optimization()
            .build();

        assert_eq!(config.optimization, StorageOptimization::None);
    }

    #[test]
    fn test_serialization() {
        let config = WorktreeConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WorktreeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}