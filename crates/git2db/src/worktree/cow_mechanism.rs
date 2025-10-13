//! Copy-on-Write mechanisms for git worktrees
//!
//! This module provides the core abstraction for CoW-based git worktrees.
//! ALL worktree creation uses some form of CoW or overlay mechanism -
//! the only question is WHICH mechanism to use.
//!
//! # Core Insight
//!
//! Git worktrees are ALWAYS created with some form of deduplication/overlay:
//! - **Overlayfs**: Linux kernel CoW filesystem (~80% space savings)
//! - **Reflink**: Filesystem-level CoW (btrfs/XFS/APFS)
//! - **Hardlink**: Poor man's CoW via hardlinks (cross-platform)
//! - **Direct**: Fallback only when no CoW mechanism works
//!
//! The "Direct" option is NOT a first-class choice - it's what happens
//! when all CoW mechanisms fail and we must fall back to a plain worktree.

use crate::errors::{Git2DBError, Git2DBResult};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// Worktree configuration focused on CoW mechanism selection
///
/// This configuration emphasizes that CoW is the default expectation.
/// We're not choosing WHETHER to use CoW, but WHICH CoW mechanism to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WorktreeConfig {
    /// Which CoW mechanism to use for the worktree
    #[serde(default)]
    pub mechanism: CoWMechanism,

    /// What to do when the selected mechanism isn't available
    #[serde(default)]
    pub unavailable_action: UnavailableAction,

    /// Whether to log CoW mechanism selection decisions
    #[serde(default = "default_true")]
    pub log_mechanism_selection: bool,

    /// Performance hints for mechanism selection
    #[serde(default)]
    pub performance_hints: PerformanceHints,
}

impl Default for WorktreeConfig {
    fn default() -> Self {
        Self {
            mechanism: CoWMechanism::Auto,
            unavailable_action: UnavailableAction::Fallback,
            log_mechanism_selection: true,
            performance_hints: PerformanceHints::default(),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Copy-on-Write mechanisms for git worktrees
///
/// These are different ways to achieve deduplication and isolation
/// for git worktrees. All options (except the fallback) provide CoW semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoWMechanism {
    /// Automatically select the best available CoW mechanism
    ///
    /// Selection priority (configurable via performance hints):
    /// 1. Overlayfs (best performance, Linux only)
    /// 2. Reflink (excellent performance, filesystem-specific)
    /// 3. Hardlink (good compatibility, moderate performance)
    ///
    /// Falls back to Direct only if no CoW mechanism works
    Auto,

    /// Linux overlayfs - kernel-level CoW filesystem
    ///
    /// - Best performance and space savings (~80%)
    /// - Requires Linux with overlayfs support
    /// - May need privileges depending on configuration
    Overlayfs(OverlayfsConfig),

    /// Filesystem-level Copy-on-Write via reflinks
    ///
    /// - Excellent performance, native filesystem CoW
    /// - Requires btrfs, XFS with reflink, or APFS
    /// - No special privileges needed
    Reflink(ReflinkConfig),

    /// Hardlink-based deduplication
    ///
    /// - Good cross-platform compatibility
    /// - Moderate space savings
    /// - Some git operations may break hardlinks
    Hardlink(HardlinkConfig),

    /// Custom CoW mechanism (for extensibility)
    ///
    /// Allows users to plug in their own CoW implementations
    #[serde(skip)]
    Custom(Box<dyn CoWBackend>),
}

impl CoWMechanism {
    /// Check if this mechanism is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            Self::Auto => true, // Always "available" - will find something
            Self::Overlayfs(config) => config.is_available(),
            Self::Reflink(config) => config.is_available(),
            Self::Hardlink(config) => config.is_available(),
            Self::Custom(backend) => backend.is_available(),
        }
    }

    /// Resolve Auto to a specific mechanism
    pub fn resolve(&self, hints: &PerformanceHints) -> CoWResolution {
        match self {
            Self::Auto => self.auto_select(hints),
            other => {
                if other.is_available() {
                    CoWResolution::Available(other.clone())
                } else {
                    CoWResolution::Unavailable {
                        requested: other.clone(),
                        reason: other.unavailable_reason(),
                    }
                }
            }
        }
    }

    /// Auto-select the best available mechanism
    fn auto_select(&self, hints: &PerformanceHints) -> CoWResolution {
        // Build priority list based on hints
        let mechanisms = match hints.priority {
            SelectionPriority::Performance => vec![
                Self::Overlayfs(OverlayfsConfig::default()),
                Self::Reflink(ReflinkConfig::default()),
                Self::Hardlink(HardlinkConfig::default()),
            ],
            SelectionPriority::Compatibility => vec![
                Self::Hardlink(HardlinkConfig::default()),
                Self::Reflink(ReflinkConfig::default()),
                Self::Overlayfs(OverlayfsConfig::default()),
            ],
            SelectionPriority::SpaceSaving => vec![
                Self::Overlayfs(OverlayfsConfig::default()),
                Self::Reflink(ReflinkConfig::default()),
                Self::Hardlink(HardlinkConfig::default()),
            ],
        };

        // Try each mechanism in priority order
        for mechanism in mechanisms {
            if mechanism.is_available() {
                return CoWResolution::Available(mechanism);
            }
        }

        // No CoW mechanism available - must use fallback
        CoWResolution::NoneAvailable
    }

    /// Get reason why this mechanism is unavailable
    fn unavailable_reason(&self) -> String {
        match self {
            Self::Auto => unreachable!("Auto is always available"),
            Self::Overlayfs(_) => "Overlayfs not available (Linux-only, may need privileges)".into(),
            Self::Reflink(_) => "Filesystem doesn't support reflinks".into(),
            Self::Hardlink(_) => "Hardlinks not supported on this filesystem".into(),
            Self::Custom(_) => "Custom backend unavailable".into(),
        }
    }

    /// Get a human-readable name for this mechanism
    pub fn name(&self) -> &str {
        match self {
            Self::Auto => "auto-selection",
            Self::Overlayfs(_) => "overlayfs",
            Self::Reflink(_) => "reflink",
            Self::Hardlink(_) => "hardlink",
            Self::Custom(_) => "custom",
        }
    }

    /// Get expected space savings percentage
    pub fn space_savings_estimate(&self) -> u8 {
        match self {
            Self::Auto => 0, // Unknown until resolved
            Self::Overlayfs(_) => 80,
            Self::Reflink(_) => 75,
            Self::Hardlink(_) => 60,
            Self::Custom(_) => 50, // Conservative estimate
        }
    }
}

/// Result of CoW mechanism resolution
#[derive(Debug, Clone)]
pub enum CoWResolution {
    /// Mechanism is available and ready to use
    Available(CoWMechanism),

    /// Requested mechanism is not available
    Unavailable {
        requested: CoWMechanism,
        reason: String,
    },

    /// No CoW mechanisms are available on this system
    NoneAvailable,
}

/// Configuration for overlayfs CoW mechanism
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OverlayfsConfig {
    /// Which overlayfs backend to use
    #[serde(default)]
    pub backend: OverlayfsBackend,

    /// Custom mount options
    #[serde(default)]
    pub mount_options: Vec<String>,

    /// Directory for upper/work layers (default: temp)
    pub overlay_dir: Option<String>,
}

impl Default for OverlayfsConfig {
    fn default() -> Self {
        Self {
            backend: OverlayfsBackend::Auto,
            mount_options: vec![],
            overlay_dir: None,
        }
    }
}

impl OverlayfsConfig {
    pub fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check for overlayfs support
            std::path::Path::new("/sys/fs/overlayfs").exists() ||
            std::path::Path::new("/sys/module/overlay").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

/// Overlayfs backend selection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayfsBackend {
    /// Auto-select best backend
    Auto,
    /// Kernel overlayfs (may need root)
    Kernel,
    /// User namespace (unprivileged)
    UserNamespace,
    /// FUSE-based (fuse-overlayfs)
    Fuse,
}

/// Configuration for reflink CoW mechanism
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ReflinkConfig {
    /// Which filesystems to consider
    #[serde(default)]
    pub filesystems: Vec<ReflinkFilesystem>,

    /// Whether to verify reflink support before use
    #[serde(default = "default_true")]
    pub verify_support: bool,
}

impl Default for ReflinkConfig {
    fn default() -> Self {
        Self {
            filesystems: vec![
                ReflinkFilesystem::Btrfs,
                ReflinkFilesystem::Xfs,
                ReflinkFilesystem::Apfs,
            ],
            verify_support: true,
        }
    }
}

impl ReflinkConfig {
    pub fn is_available(&self) -> bool {
        // Check filesystem type and reflink support
        // This is a simplified check - real implementation would be more thorough
        #[cfg(target_os = "linux")]
        {
            // Check for btrfs or XFS with reflink
            false // TODO: Implement actual filesystem detection
        }
        #[cfg(target_os = "macos")]
        {
            // Check for APFS
            false // TODO: Implement APFS detection
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }
}

/// Filesystems that support reflinks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReflinkFilesystem {
    Btrfs,
    Xfs,
    Apfs,
    Bcachefs,
}

/// Configuration for hardlink CoW mechanism
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct HardlinkConfig {
    /// Whether to use hardlinks for directories (if supported)
    #[serde(default)]
    pub link_directories: bool,

    /// Minimum file size to hardlink (bytes)
    #[serde(default = "default_min_size")]
    pub min_file_size: u64,

    /// Whether to verify hardlink support
    #[serde(default = "default_true")]
    pub verify_support: bool,
}

impl Default for HardlinkConfig {
    fn default() -> Self {
        Self {
            link_directories: false,
            min_file_size: 1024, // 1KB minimum
            verify_support: true,
        }
    }
}

impl HardlinkConfig {
    pub fn is_available(&self) -> bool {
        // Hardlinks are generally available on most filesystems
        // Real implementation would check specific filesystem support
        true
    }
}

fn default_min_size() -> u64 {
    1024
}

/// What to do when the requested CoW mechanism is unavailable
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableAction {
    /// Automatically fall back to a plain directory (no CoW)
    ///
    /// This is the default - we always want a working worktree,
    /// even if we can't optimize it with CoW.
    Fallback,

    /// Try the next best CoW mechanism
    ///
    /// Instead of falling back to plain, try other CoW mechanisms
    /// in order of preference.
    TryNext,

    /// Fail with an error
    ///
    /// Use when CoW is mandatory (e.g., space-constrained environments)
    Fail,

    /// Warn and fall back
    ///
    /// Like Fallback, but logs a warning to make it clear
    /// that CoW was expected but unavailable.
    WarnAndFallback,
}

impl Default for UnavailableAction {
    fn default() -> Self {
        Self::Fallback
    }
}

/// Performance hints for mechanism selection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct PerformanceHints {
    /// Priority for auto-selection
    #[serde(default)]
    pub priority: SelectionPriority,

    /// Expected number of worktrees
    #[serde(default)]
    pub expected_worktrees: Option<u32>,

    /// Expected worktree lifetime (seconds)
    #[serde(default)]
    pub expected_lifetime_seconds: Option<u64>,

    /// Whether worktrees will be modified heavily
    #[serde(default)]
    pub heavy_writes_expected: bool,
}

/// Priority for mechanism selection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPriority {
    /// Prioritize performance (overlayfs > reflink > hardlink)
    Performance,

    /// Prioritize compatibility (hardlink > reflink > overlayfs)
    Compatibility,

    /// Prioritize space saving (overlayfs > reflink > hardlink)
    SpaceSaving,
}

impl Default for SelectionPriority {
    fn default() -> Self {
        Self::Performance
    }
}

/// Trait for custom CoW backend implementations
pub trait CoWBackend: std::fmt::Debug + Send + Sync {
    /// Check if this backend is available
    fn is_available(&self) -> bool;

    /// Create a CoW worktree
    fn create_worktree(&self, source: &Path, target: &Path) -> Git2DBResult<()>;

    /// Clean up a CoW worktree
    fn cleanup_worktree(&self, target: &Path) -> Git2DBResult<()>;

    /// Get backend name
    fn name(&self) -> &str;

    /// Get estimated space savings
    fn space_savings_estimate(&self) -> u8 {
        50 // Conservative default
    }
}

/// Builder for fluent worktree configuration
pub struct WorktreeBuilder {
    config: WorktreeConfig,
}

impl WorktreeBuilder {
    /// Create a new builder with auto-selection
    pub fn new() -> Self {
        Self {
            config: WorktreeConfig::default(),
        }
    }

    /// Use a specific CoW mechanism
    pub fn mechanism(mut self, mechanism: CoWMechanism) -> Self {
        self.config.mechanism = mechanism;
        self
    }

    /// Use overlayfs with default config
    pub fn overlayfs(mut self) -> Self {
        self.config.mechanism = CoWMechanism::Overlayfs(OverlayfsConfig::default());
        self
    }

    /// Use reflinks with default config
    pub fn reflink(mut self) -> Self {
        self.config.mechanism = CoWMechanism::Reflink(ReflinkConfig::default());
        self
    }

    /// Use hardlinks with default config
    pub fn hardlink(mut self) -> Self {
        self.config.mechanism = CoWMechanism::Hardlink(HardlinkConfig::default());
        self
    }

    /// Set action when mechanism unavailable
    pub fn on_unavailable(mut self, action: UnavailableAction) -> Self {
        self.config.unavailable_action = action;
        self
    }

    /// Set performance hints
    pub fn performance_hints(mut self, hints: PerformanceHints) -> Self {
        self.config.performance_hints = hints;
        self
    }

    /// Prioritize performance in auto-selection
    pub fn prioritize_performance(mut self) -> Self {
        self.config.performance_hints.priority = SelectionPriority::Performance;
        self
    }

    /// Prioritize space saving in auto-selection
    pub fn prioritize_space(mut self) -> Self {
        self.config.performance_hints.priority = SelectionPriority::SpaceSaving;
        self
    }

    /// Build the configuration
    pub fn build(self) -> WorktreeConfig {
        self.config
    }
}

impl fmt::Display for CoWMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_emphasizes_cow() {
        let config = WorktreeConfig::default();
        assert_eq!(config.mechanism, CoWMechanism::Auto);
        assert_eq!(config.unavailable_action, UnavailableAction::Fallback);

        // Auto means "find me a CoW mechanism"
        // Fallback means "only use plain if nothing else works"
    }

    #[test]
    fn test_builder_fluent_api() {
        let config = WorktreeBuilder::new()
            .overlayfs()
            .on_unavailable(UnavailableAction::TryNext)
            .prioritize_space()
            .build();

        assert!(matches!(config.mechanism, CoWMechanism::Overlayfs(_)));
        assert_eq!(config.unavailable_action, UnavailableAction::TryNext);
        assert_eq!(config.performance_hints.priority, SelectionPriority::SpaceSaving);
    }

    #[test]
    fn test_space_savings_estimates() {
        assert_eq!(CoWMechanism::Overlayfs(Default::default()).space_savings_estimate(), 80);
        assert_eq!(CoWMechanism::Reflink(Default::default()).space_savings_estimate(), 75);
        assert_eq!(CoWMechanism::Hardlink(Default::default()).space_savings_estimate(), 60);
    }
}