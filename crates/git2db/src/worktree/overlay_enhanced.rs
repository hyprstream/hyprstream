//! Enhanced overlayfs implementation with explicit backend selection
//!
//! This module extends the existing overlayfs implementation to support
//! explicit backend selection through the new strategy enum system.

use crate::errors::{Git2DBError, Git2DBResult};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Explicit overlayfs backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Automatically select best available backend
    Auto,
    /// Kernel overlayfs (requires CAP_SYS_ADMIN or root)
    Kernel,
    /// User namespace overlayfs (no privileges required)
    UserNamespace,
    /// FUSE overlayfs (requires fuse-overlayfs binary)
    Fuse,
}

impl BackendType {
    /// Check if this specific backend is available
    pub fn is_available(&self) -> bool {
        match self {
            Self::Auto => {
                // At least one backend must be available
                Self::Kernel.is_available() ||
                Self::UserNamespace.is_available() ||
                Self::Fuse.is_available()
            }
            Self::Kernel => {
                #[cfg(target_os = "linux")]
                {
                    super::overlay::KernelBackend.is_available()
                }
                #[cfg(not(target_os = "linux"))]
                false
            }
            Self::UserNamespace => {
                #[cfg(target_os = "linux")]
                {
                    super::overlay::UserNamespaceBackend.is_available()
                }
                #[cfg(not(target_os = "linux"))]
                false
            }
            Self::Fuse => {
                #[cfg(target_os = "linux")]
                {
                    super::overlay::FuseBackend.is_available()
                }
                #[cfg(not(target_os = "linux"))]
                false
            }
        }
    }

    /// Get human-readable name for this backend
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Kernel => "kernel",
            Self::UserNamespace => "userns",
            Self::Fuse => "fuse",
        }
    }

    /// Get detailed description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Auto => "Automatically select best available overlayfs backend",
            Self::Kernel => "Kernel overlayfs (requires root or CAP_SYS_ADMIN)",
            Self::UserNamespace => "User namespace overlayfs (no privileges required)",
            Self::Fuse => "FUSE-based overlayfs (requires fuse-overlayfs)",
        }
    }

    /// Get availability requirements
    pub fn requirements(&self) -> Vec<String> {
        match self {
            Self::Auto => vec!["Linux kernel with overlayfs support".to_string()],
            Self::Kernel => vec![
                "Linux kernel with overlayfs".to_string(),
                "CAP_SYS_ADMIN capability or root".to_string(),
            ],
            Self::UserNamespace => vec![
                "Linux kernel with overlayfs".to_string(),
                "Unprivileged user namespaces enabled".to_string(),
                "Kernel 5.11+ recommended".to_string(),
            ],
            Self::Fuse => vec![
                "fuse-overlayfs binary in PATH".to_string(),
                "FUSE kernel module".to_string(),
                "No special privileges".to_string(),
            ],
        }
    }
}

/// Enhanced overlayfs strategy with explicit backend selection
pub struct OverlayWorktreeStrategyEnhanced {
    backend_type: BackendType,
    #[cfg(feature = "overlayfs")]
    inner: super::overlay::OverlayWorktreeStrategy,
}

impl OverlayWorktreeStrategyEnhanced {
    /// Create with automatic backend selection
    pub fn new() -> Self {
        Self::with_backend(BackendType::Auto)
    }

    /// Create with explicit backend selection
    pub fn with_backend(backend_type: BackendType) -> Self {
        #[cfg(feature = "overlayfs")]
        {
            let inner = match backend_type {
                BackendType::Auto => {
                    super::overlay::OverlayWorktreeStrategy::new()
                }
                BackendType::Kernel => {
                    super::overlay::OverlayWorktreeStrategy::with_backend(
                        super::overlay::BackendType::Kernel
                    )
                }
                BackendType::UserNamespace => {
                    super::overlay::OverlayWorktreeStrategy::with_backend(
                        super::overlay::BackendType::UserNamespace
                    )
                }
                BackendType::Fuse => {
                    super::overlay::OverlayWorktreeStrategy::with_backend(
                        super::overlay::BackendType::Fuse
                    )
                }
            };

            Self {
                backend_type,
                inner,
            }
        }

        #[cfg(not(feature = "overlayfs"))]
        {
            Self { backend_type }
        }
    }

    /// Get the selected backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }

    /// Check if the selected backend is available
    pub fn is_available(&self) -> bool {
        self.backend_type.is_available()
    }

    /// Get detailed availability status
    pub fn availability_status(&self) -> AvailabilityStatus {
        if !cfg!(target_os = "linux") {
            return AvailabilityStatus::UnsupportedPlatform {
                platform: std::env::consts::OS.to_string(),
                reason: "Overlayfs is Linux-only".to_string(),
            };
        }

        if !cfg!(feature = "overlayfs") {
            return AvailabilityStatus::FeatureDisabled {
                feature: "overlayfs".to_string(),
                reason: "Compile with --features overlayfs to enable".to_string(),
            };
        }

        if self.backend_type.is_available() {
            AvailabilityStatus::Available {
                backend: self.backend_type.name().to_string(),
                capabilities: self.get_capabilities(),
            }
        } else {
            AvailabilityStatus::Unavailable {
                backend: self.backend_type.name().to_string(),
                requirements: self.backend_type.requirements(),
                alternatives: self.get_alternative_backends(),
            }
        }
    }

    /// Get alternative backends if the selected one is unavailable
    fn get_alternative_backends(&self) -> Vec<String> {
        let mut alternatives = Vec::new();

        if self.backend_type != BackendType::Kernel && BackendType::Kernel.is_available() {
            alternatives.push("overlayfs-kernel".to_string());
        }
        if self.backend_type != BackendType::UserNamespace && BackendType::UserNamespace.is_available() {
            alternatives.push("overlayfs-userns".to_string());
        }
        if self.backend_type != BackendType::Fuse && BackendType::Fuse.is_available() {
            alternatives.push("overlayfs-fuse".to_string());
        }

        // Always suggest git as a fallback
        alternatives.push("git".to_string());

        alternatives
    }

    /// Get capabilities of the selected backend
    fn get_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            space_efficient: true,
            requires_privileges: matches!(self.backend_type, BackendType::Kernel),
            performance_rating: match self.backend_type {
                BackendType::Kernel => PerformanceRating::Excellent,
                BackendType::UserNamespace => PerformanceRating::Good,
                BackendType::Fuse => PerformanceRating::Moderate,
                BackendType::Auto => PerformanceRating::Good, // Average
            },
            estimated_space_savings: 80, // ~80% savings with overlayfs
        }
    }
}

/// Detailed availability status
#[derive(Debug, Clone)]
pub enum AvailabilityStatus {
    /// Backend is available and ready to use
    Available {
        backend: String,
        capabilities: BackendCapabilities,
    },
    /// Backend is not available on this system
    Unavailable {
        backend: String,
        requirements: Vec<String>,
        alternatives: Vec<String>,
    },
    /// Platform doesn't support this backend
    UnsupportedPlatform {
        platform: String,
        reason: String,
    },
    /// Feature is disabled at compile time
    FeatureDisabled {
        feature: String,
        reason: String,
    },
}

impl AvailabilityStatus {
    /// Get a user-friendly message about availability
    pub fn message(&self) -> String {
        match self {
            Self::Available { backend, capabilities } => {
                format!(
                    "{} backend is available ({}% space savings, {} performance)",
                    backend,
                    capabilities.estimated_space_savings,
                    capabilities.performance_rating,
                )
            }
            Self::Unavailable { backend, requirements, alternatives } => {
                format!(
                    "{} backend is not available. Requirements: {}. Try: {}",
                    backend,
                    requirements.join(", "),
                    alternatives.join(", ")
                )
            }
            Self::UnsupportedPlatform { platform, reason } => {
                format!("Not supported on {}: {}", platform, reason)
            }
            Self::FeatureDisabled { feature, reason } => {
                format!("{} feature disabled: {}", feature, reason)
            }
        }
    }

    /// Check if the status indicates availability
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

/// Backend capabilities information
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub space_efficient: bool,
    pub requires_privileges: bool,
    pub performance_rating: PerformanceRating,
    pub estimated_space_savings: u32, // Percentage
}

/// Performance rating for backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceRating {
    Excellent,
    Good,
    Moderate,
    Poor,
}

impl std::fmt::Display for PerformanceRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "excellent"),
            Self::Good => write!(f, "good"),
            Self::Moderate => write!(f, "moderate"),
            Self::Poor => write!(f, "poor"),
        }
    }
}

/// Helper to select best backend based on requirements
pub struct BackendSelector {
    require_no_privileges: bool,
    prefer_performance: bool,
    allow_experimental: bool,
}

impl BackendSelector {
    pub fn new() -> Self {
        Self {
            require_no_privileges: false,
            prefer_performance: true,
            allow_experimental: false,
        }
    }

    /// Require backends that don't need special privileges
    pub fn require_no_privileges(mut self, require: bool) -> Self {
        self.require_no_privileges = require;
        self
    }

    /// Prefer higher performance backends
    pub fn prefer_performance(mut self, prefer: bool) -> Self {
        self.prefer_performance = prefer;
        self
    }

    /// Allow experimental backends
    pub fn allow_experimental(mut self, allow: bool) -> Self {
        self.allow_experimental = allow;
        self
    }

    /// Select the best available backend based on criteria
    pub fn select(&self) -> Option<BackendType> {
        let mut candidates = vec![
            (BackendType::Kernel, 3),      // Best performance
            (BackendType::UserNamespace, 2), // Good, no privileges
            (BackendType::Fuse, 1),         // Moderate, no privileges
        ];

        // Filter by availability
        candidates.retain(|(backend, _)| backend.is_available());

        // Filter by privilege requirements
        if self.require_no_privileges {
            candidates.retain(|(backend, _)| {
                !matches!(backend, BackendType::Kernel)
            });
        }

        // Sort by score (higher is better)
        if self.prefer_performance {
            candidates.sort_by_key(|(_, score)| std::cmp::Reverse(*score));
        }

        candidates.first().map(|(backend, _)| *backend)
    }

    /// Get all available backends with their scores
    pub fn list_available(&self) -> Vec<(BackendType, String)> {
        vec![
            BackendType::Kernel,
            BackendType::UserNamespace,
            BackendType::Fuse,
        ]
        .into_iter()
        .filter(|b| b.is_available())
        .map(|b| (b, b.description().to_string()))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_names() {
        assert_eq!(BackendType::Auto.name(), "auto");
        assert_eq!(BackendType::Kernel.name(), "kernel");
        assert_eq!(BackendType::UserNamespace.name(), "userns");
        assert_eq!(BackendType::Fuse.name(), "fuse");
    }

    #[test]
    fn test_backend_requirements() {
        let kernel_reqs = BackendType::Kernel.requirements();
        assert!(kernel_reqs.iter().any(|r| r.contains("CAP_SYS_ADMIN")));

        let fuse_reqs = BackendType::Fuse.requirements();
        assert!(fuse_reqs.iter().any(|r| r.contains("fuse-overlayfs")));
    }

    #[test]
    fn test_backend_selector() {
        let selector = BackendSelector::new()
            .require_no_privileges(true)
            .prefer_performance(true);

        // Should not select kernel backend when no privileges required
        if let Some(selected) = selector.select() {
            assert_ne!(selected, BackendType::Kernel);
        }
    }

    #[test]
    fn test_availability_status() {
        let strategy = OverlayWorktreeStrategyEnhanced::new();
        let status = strategy.availability_status();

        // Status should have a message
        assert!(!status.message().is_empty());

        #[cfg(all(target_os = "linux", feature = "overlayfs"))]
        {
            // On Linux with overlayfs feature, something should be available
            // or we should get meaningful unavailable status
            match status {
                AvailabilityStatus::Available { .. } => {
                    assert!(status.is_available());
                }
                AvailabilityStatus::Unavailable { alternatives, .. } => {
                    assert!(!alternatives.is_empty());
                    assert!(alternatives.contains(&"git".to_string()));
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_performance_rating_display() {
        assert_eq!(format!("{}", PerformanceRating::Excellent), "excellent");
        assert_eq!(format!("{}", PerformanceRating::Good), "good");
        assert_eq!(format!("{}", PerformanceRating::Moderate), "moderate");
        assert_eq!(format!("{}", PerformanceRating::Poor), "poor");
    }
}