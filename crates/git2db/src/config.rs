//! Configuration for git2db operations
//!
//! This module uses the `config` crate directly for maximum flexibility.
//! Configuration sources are loaded in priority order:
//! 1. Programmatic overrides (`.set_override()`)
//! 2. Environment variables (`GIT2DB__*`)
//! 3. Config file (`~/.config/git2db/config.toml`)
//! 4. Default values
//!
//! # Basic Usage
//!
//! ```rust,ignore
//! use git2db::config::Git2DBConfig;
//!
//! // Simple case - use defaults
//! let config = Git2DBConfig::load()?;
//! ```
//!
//! # Programmatic Overrides
//!
//! ```rust,ignore
//! use git2db::config::Git2DBConfig;
//!
//! let config: Git2DBConfig = Git2DBConfig::builder()?
//!     .set_override("network.timeout_secs", 60)?
//!     .set_override("performance.max_repo_cache", 200)?
//!     .build()?
//!     .try_deserialize()?;
//! ```
//!
//! # Custom Config Files
//!
//! ```rust,ignore
//! use git2db::config::Git2DBConfig;
//! use config::File;
//!
//! let config: Git2DBConfig = Git2DBConfig::builder()?
//!     .add_source(File::with_name("/custom/config"))?
//!     .build()?
//!     .try_deserialize()?;
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Main configuration for git2db operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Git2DBConfig {
    /// Repository management configuration
    pub repository: RepositoryConfig,
    /// Network and authentication configuration
    pub network: NetworkConfig,
    /// Performance and caching configuration
    pub performance: PerformanceConfig,
    /// Default signature for commits
    pub signature: GitSignature,
    /// Worktree strategy configuration
    #[serde(default)]
    pub worktree: WorktreeConfig,
    /// GitTorrent P2P transport configuration (optional, requires gittorrent-transport feature)
    #[cfg(feature = "gittorrent-transport")]
    #[serde(default)]
    pub gittorrent: gittorrent::service::GitTorrentConfig,
    /// XET large file storage configuration (optional, requires xet-storage feature)
    #[cfg(feature = "xet-storage")]
    #[serde(default)]
    pub xet: XetConfig,
}


impl Git2DBConfig {
    /// Create a configuration builder with git2db defaults
    ///
    /// Returns the native `config::ConfigBuilder` for maximum flexibility.
    /// This follows the same pattern as gittorrent for consistency.
    ///
    /// **Default Priority Order (highest to lowest):**
    /// 1. Programmatic overrides (`.set_override()`)
    /// 2. Environment variables (`GIT2DB__*`)
    /// 3. Config file (`~/.config/git2db/config.toml`)
    /// 4. Default values
    ///
    /// # Basic Usage
    ///
    /// ```rust,ignore
    /// use git2db::config::Git2DBConfig;
    ///
    /// // Simple case - use defaults
    /// let config: Git2DBConfig = Git2DBConfig::builder()?
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Programmatic Overrides
    ///
    /// ```rust,ignore
    /// use git2db::config::Git2DBConfig;
    ///
    /// let config: Git2DBConfig = Git2DBConfig::builder()?
    ///     .set_override("network.timeout_secs", 60)?
    ///     .set_override("performance.max_repo_cache", 200)?
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Custom Config Sources
    ///
    /// ```rust,ignore
    /// use git2db::config::Git2DBConfig;
    /// use config::File;
    ///
    /// let config: Git2DBConfig = Git2DBConfig::builder()?
    ///     .add_source(File::with_name("/custom/config"))  // Highest priority
    ///     .build()?
    ///     .try_deserialize()?;
    /// ```
    ///
    /// # Supported Environment Variables
    ///
    /// - `GIT2DB_REPOSITORY__PREFER_SHALLOW`: Enable shallow clones (bool)
    /// - `GIT2DB_REPOSITORY__AUTO_INIT_SUBMODULES`: Auto-init submodules on open (bool)
    /// - `GIT2DB_NETWORK__TIMEOUT_SECS`: Network timeout in seconds (u64)
    /// - `GIT2DB_PERFORMANCE__MAX_REPO_CACHE`: Max cached repos (usize)
    /// - `GIT2DB_SIGNATURE__NAME`: Git commit author name
    /// - `GIT2DB_SIGNATURE__EMAIL`: Git commit author email
    /// - `GIT2DB_GITTORRENT__P2P_PORT`: GitTorrent P2P port (requires feature)
    pub fn builder() -> Result<config::ConfigBuilder<config::builder::DefaultState>> {
        Ok(config::Config::builder()
            // Set defaults (lowest priority)
            .set_default("repository.prefer_shallow", true)?
            .set_default("repository.shallow_depth", 1)?
            .set_default("repository.auto_init", true)?
            .set_default("repository.auto_init_submodules", true)?
            .set_default("network.timeout_secs", 30u64)?
            .set_default("network.max_retries", 3)?
            .set_default("network.retry_base_delay_ms", 500u64)?
            .set_default("network.retry_max_delay_secs", 10u64)?
            .set_default("network.user_agent", format!("git2db/{}", crate::VERSION))?
            .set_default("network.use_credential_helper", true)?
            .set_default("performance.max_repo_cache", 100)?
            .set_default("performance.repo_cache_ttl_secs", 300u64)?
            .set_default("performance.auto_cleanup", true)?
            .set_default("performance.cleanup_interval_secs", 60u64)?
            .set_default("performance.max_concurrent_ops", 10)?
            .set_default("signature.name", "git2db")?
            .set_default("signature.email", "git2db@local")?
            // Add config file (optional, medium priority)
            .add_source(
                config::File::from(Self::default_config_path().unwrap_or_default()).required(false),
            )
            // Add environment variables (high priority)
            .add_source(
                config::Environment::with_prefix("GIT2DB")
                    .separator("__")
                    .try_parsing(true),
            ))
    }

    /// Convenience method to load configuration with defaults
    ///
    /// This is equivalent to `Git2DBConfig::builder()?.build()?.try_deserialize()?`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use git2db::config::Git2DBConfig;
    ///
    /// let config = Git2DBConfig::load()?;
    /// ```
    pub fn load() -> Result<Self> {
        let config = Self::builder()?
            .build()
            .context("Failed to build configuration")?;

        let deserialized: Self = config
            .try_deserialize()
            .context("Failed to deserialize configuration")?;

        // Debug log
        tracing::info!(
            "Loaded git2db config: token present = {}, use_cred_helper = {}",
            deserialized.network.access_token.is_some(),
            deserialized.network.use_credential_helper
        );

        Ok(deserialized)
    }

    /// Get the default config file path (~/.config/git2db/config.toml)
    fn default_config_path() -> Option<PathBuf> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(xdg_config) = std::env::var("XDG_CONFIG_HOME") {
                return Some(PathBuf::from(xdg_config).join("git2db").join("config.toml"));
            }
        }

        if let Ok(home) = std::env::var("HOME") {
            return Some(
                PathBuf::from(home)
                    .join(".config")
                    .join("git2db")
                    .join("config.toml"),
            );
        }

        None
    }
}

/// Repository-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    /// Enable shallow clones by default
    pub prefer_shallow: bool,
    /// Default clone depth for shallow clones
    pub shallow_depth: Option<u32>,
    /// Automatically create initial commit for new registries
    pub auto_init: bool,
    /// Automatically initialize and update submodules when opening registry
    ///
    /// When true (default), opening a registry will call `git submodule init`
    /// and `git submodule update` to ensure all tracked repositories are
    /// available. This is equivalent to `git clone --recurse-submodules`.
    ///
    /// Set to false if you want manual control over submodule initialization.
    pub auto_init_submodules: bool,
}

impl Default for RepositoryConfig {
    fn default() -> Self {
        Self {
            prefer_shallow: true,
            shallow_depth: Some(1),
            auto_init: true,
            auto_init_submodules: true,
        }
    }
}

/// Network and authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network timeout for operations
    pub timeout: Duration,
    /// Maximum retry attempts for network operations
    pub max_retries: usize,
    /// Base delay for exponential backoff
    pub retry_base_delay: Duration,
    /// Maximum delay between retries
    pub retry_max_delay: Duration,
    /// Proxy configuration
    pub proxy_url: Option<String>,
    /// User agent for HTTP requests
    pub user_agent: String,
    /// Enable git credential helper (default credentials)
    /// This allows using system git credentials, SSH agents, etc.
    #[serde(default = "default_use_credential_helper")]
    pub use_credential_helper: bool,
    /// Personal access token (e.g., for GitHub, GitLab, Hugging Face)
    /// Can also be set via GIT2DB_NETWORK__ACCESS_TOKEN env var
    pub access_token: Option<String>,
}

fn default_use_credential_helper() -> bool {
    true
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_base_delay: Duration::from_millis(500),
            retry_max_delay: Duration::from_secs(10),
            proxy_url: None,
            user_agent: format!("git2db/{}", crate::VERSION),
            use_credential_helper: default_use_credential_helper(),
            access_token: None,
        }
    }
}

/// Performance and caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum number of cached repository handles
    pub max_repo_cache: usize,
    /// Repository cache TTL
    pub repo_cache_ttl: Duration,
    /// Enable automatic cache cleanup
    pub auto_cleanup: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Maximum concurrent git operations
    pub max_concurrent_ops: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_repo_cache: 100,
            repo_cache_ttl: Duration::from_secs(300), // 5 minutes
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60), // 1 minute
            max_concurrent_ops: 10,
        }
    }
}

/// Git signature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitSignature {
    /// Name for git commits
    pub name: String,
    /// Email for git commits
    pub email: String,
}

impl Default for GitSignature {
    fn default() -> Self {
        Self {
            name: "git2db".to_string(),
            email: "git2db@local".to_string(),
        }
    }
}

impl GitSignature {
    /// Create a new signature
    pub fn new<N: Into<String>, E: Into<String>>(name: N, email: E) -> Self {
        Self {
            name: name.into(),
            email: email.into(),
        }
    }

    /// Create a git2::Signature from this configuration
    pub fn to_git2_signature(&self) -> Result<git2::Signature<'static>, git2::Error> {
        git2::Signature::now(&self.name, &self.email)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Git2DBConfig::default();
        assert!(config.repository.prefer_shallow);
        assert_eq!(config.repository.shallow_depth, Some(1));
        assert_eq!(config.network.timeout, Duration::from_secs(30));
        assert_eq!(config.performance.max_repo_cache, 100);
    }

    #[test]
    fn test_git_signature() {
        let sig = GitSignature::new("Test User", "test@example.com");
        assert_eq!(sig.name, "Test User");
        assert_eq!(sig.email, "test@example.com");

        let git2_sig = sig.to_git2_signature().unwrap();
        assert_eq!(git2_sig.name().unwrap(), "Test User");
        assert_eq!(git2_sig.email().unwrap(), "test@example.com");
    }

    #[test]
    fn test_repository_config_defaults() {
        let config = RepositoryConfig::default();
        assert!(config.prefer_shallow);
        assert_eq!(config.shallow_depth, Some(1));
        assert!(config.auto_init);
    }

    #[test]
    fn test_network_config_defaults() {
        let config = NetworkConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay, Duration::from_millis(500));
        assert_eq!(config.retry_max_delay, Duration::from_secs(10));
        assert!(config.proxy_url.is_none());
        assert!(config.user_agent.contains("git2db"));
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();
        assert_eq!(config.max_repo_cache, 100);
        assert_eq!(config.repo_cache_ttl, Duration::from_secs(300));
        assert!(config.auto_cleanup);
        assert_eq!(config.cleanup_interval, Duration::from_secs(60));
        assert_eq!(config.max_concurrent_ops, 10);
    }

    #[test]
    fn test_git_signature_defaults() {
        let sig = GitSignature::default();
        assert_eq!(sig.name, "git2db");
        assert_eq!(sig.email, "git2db@local");
    }

    #[test]
    fn test_config_serialization() {
        let config = Git2DBConfig::default();

        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"prefer_shallow\":true"));
        assert!(json.contains("\"max_repo_cache\":100"));

        // Test deserialization
        let deserialized: Git2DBConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            config.repository.prefer_shallow,
            deserialized.repository.prefer_shallow
        );
        assert_eq!(
            config.performance.max_repo_cache,
            deserialized.performance.max_repo_cache
        );
    }

    #[test]
    fn test_git_signature_with_special_characters() {
        let sig = GitSignature::new("Test User with Ünicöde", "test+tag@example.com");
        assert_eq!(sig.name, "Test User with Ünicöde");
        assert_eq!(sig.email, "test+tag@example.com");

        // Should be able to create git2 signature
        let git2_sig = sig.to_git2_signature().unwrap();
        assert_eq!(git2_sig.name().unwrap(), "Test User with Ünicöde");
        assert_eq!(git2_sig.email().unwrap(), "test+tag@example.com");
    }

    #[test]
    fn test_config_builder_returns_config_builder() {
        // Test that builder() returns a ConfigBuilder
        let builder = Git2DBConfig::builder();
        assert!(builder.is_ok());
    }

    #[test]
    fn test_config_components_independently() {
        // Test each config component can be created with custom values
        let repo_config = RepositoryConfig {
            prefer_shallow: false,
            shallow_depth: Some(10),
            auto_init: false,
            auto_init_submodules: false,
        };
        assert!(!repo_config.prefer_shallow);
        assert_eq!(repo_config.shallow_depth, Some(10));

        let network_config = NetworkConfig {
            timeout: Duration::from_secs(120),
            max_retries: 5,
            retry_base_delay: Duration::from_millis(1000),
            retry_max_delay: Duration::from_secs(30),
            proxy_url: Some("http://proxy.example.com:8080".to_string()),
            user_agent: "test-agent".to_string(),
            use_credential_helper: true,
            access_token: Some("test-token".to_string()),
        };
        assert_eq!(network_config.timeout, Duration::from_secs(120));
        assert_eq!(network_config.max_retries, 5);

        let perf_config = PerformanceConfig {
            max_repo_cache: 500,
            repo_cache_ttl: Duration::from_secs(600),
            auto_cleanup: false,
            cleanup_interval: Duration::from_secs(120),
            max_concurrent_ops: 50,
        };
        assert_eq!(perf_config.max_repo_cache, 500);
        assert!(!perf_config.auto_cleanup);
    }

    #[test]
    fn test_full_config_construction() {
        // Test building a complete config manually
        let config = Git2DBConfig {
            repository: RepositoryConfig {
                prefer_shallow: false,
                shallow_depth: None,
                auto_init: false,
                auto_init_submodules: false,
            },
            network: NetworkConfig::default(),
            performance: PerformanceConfig::default(),
            signature: GitSignature::new("Test", "test@example.com"),
            worktree: WorktreeConfig::default(),
            #[cfg(feature = "gittorrent-transport")]
            gittorrent: gittorrent::service::GitTorrentConfig::default(),
            #[cfg(feature = "xet-storage")]
            xet: git_xet_filter::XetConfig::default(),
        };

        assert!(!config.repository.prefer_shallow);
        assert_eq!(config.signature.name, "Test");
    }

    #[test]
    fn test_git_signature_creation_variants() {
        // Test with owned strings
        let sig1 = GitSignature::new("User1".to_string(), "user1@example.com".to_string());
        assert_eq!(sig1.name, "User1");

        // Test with &str
        let sig2 = GitSignature::new("User2", "user2@example.com");
        assert_eq!(sig2.name, "User2");

        // Test with String references
        let name = "User3".to_string();
        let email = "user3@example.com".to_string();
        let sig3 = GitSignature::new(&name, &email);
        assert_eq!(sig3.name, "User3");
    }
}

/// Worktree storage driver configuration (Docker's graphdriver pattern)
///
/// Default behavior:
/// - Uses "auto" driver selection (overlay2 on Linux, vfs otherwise)
/// - Falls back to vfs if selected driver unavailable
/// - Logs driver selection
///
/// # Example
///
/// ```toml
/// [worktree]
/// driver = "overlay2"  # or "auto", "vfs", etc.
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeConfig {
    /// Storage driver selection (Docker-style)
    ///
    /// Options:
    /// - "auto" (default): Auto-select best available driver
    /// - "overlay2": Linux overlayfs (~80% space savings)
    /// - "vfs": Plain directories (always works, no optimization)
    /// - Future: "btrfs", "reflink", "hardlink"
    ///
    /// Default: "auto"
    #[serde(default = "default_driver")]
    pub driver: String,

    /// Fallback to vfs if selected driver unavailable
    ///
    /// Default: true (graceful degradation)
    ///
    /// When true, falls back to vfs if selected driver not available.
    /// When false, create_worktree() fails if driver unavailable.
    #[serde(default = "default_true")]
    pub fallback: bool,

    /// Force specific backend for overlay2 driver (advanced)
    ///
    /// Options: "fuse", "userns", "kernel"
    /// Default: None (automatic backend selection)
    #[serde(default)]
    pub force_backend: Option<String>,

    /// Log driver selection decisions
    ///
    /// Default: true
    #[serde(default = "default_true")]
    pub log_driver: bool,
}

fn default_driver() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for WorktreeConfig {
    fn default() -> Self {
        Self {
            driver: "auto".to_string(),
            fallback: true,
            force_backend: None,
            log_driver: true,
        }
    }
}

impl WorktreeConfig {
    /// Require overlay2 driver (fail if unavailable)
    pub fn overlay2_only() -> Self {
        Self {
            driver: "overlay2".to_string(),
            fallback: false,
            force_backend: None,
            log_driver: true,
        }
    }

    /// Use only vfs driver (plain git worktrees, no optimization)
    pub fn vfs_only() -> Self {
        Self {
            driver: "vfs".to_string(),
            fallback: true,
            force_backend: None,
            log_driver: true,
        }
    }

    /// Force specific driver
    pub fn with_driver(driver: impl Into<String>) -> Self {
        Self {
            driver: driver.into(),
            fallback: true,
            force_backend: None,
            log_driver: true,
        }
    }

    /// Force specific overlay2 backend
    pub fn with_overlay2_backend(backend: impl Into<String>) -> Self {
        Self {
            driver: "overlay2".to_string(),
            fallback: true,
            force_backend: Some(backend.into()),
            log_driver: true,
        }
    }
}

/// XET large file storage configuration
///
/// Re-exported from git-xet-filter for convenience.
/// The xet-filter crate can be used standalone.
#[cfg(feature = "xet-storage")]
pub use git_xet_filter::XetConfig;
