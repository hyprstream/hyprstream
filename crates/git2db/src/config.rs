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
    /// - `GIT2DB_WORKTREE__DRIVER`: Storage driver selection (overlay2, vfs, reflink)
    /// - `GIT2DB_WORKTREE__LOG_DRIVER`: Log driver selection decisions (bool)
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
            .set_default("worktree.driver", "vfs")?
            .set_default("worktree.log_driver", true)?
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
                    .try_parsing(true)
                    .ignore_empty(true),
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
            .context("Failed to deserialize configuration. This might be due to:\n  - Invalid config file format\n  - Invalid environment variable values\n  - Type mismatches in config")?;

        // Debug log
        tracing::info!(
            "Loaded git2db config: token present = {}, use_cred_helper = {}, worktree.driver = {}",
            deserialized.network.access_token.is_some(),
            deserialized.network.use_credential_helper,
            deserialized.worktree.driver
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
///
/// Field naming convention: All timeout/duration fields use unit suffixes:
/// - `*_secs` = seconds (u64)
/// - `*_ms` = milliseconds (u64)
///
/// This ensures clear time unit specification in configuration files and environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network timeout for operations (in seconds)
    #[serde(default = "default_network_timeout_secs")]
    pub timeout_secs: u64,
    /// Maximum retry attempts for network operations
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    /// Base delay for exponential backoff (in milliseconds)
    #[serde(default = "default_retry_base_delay_ms")]
    pub retry_base_delay_ms: u64,
    /// Maximum delay between retries (in seconds)
    #[serde(default = "default_retry_max_delay_secs")]
    pub retry_max_delay_secs: u64,
    /// Proxy configuration
    pub proxy_url: Option<String>,
    /// User agent for HTTP requests
    #[serde(default = "default_user_agent")]
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

fn default_network_timeout_secs() -> u64 {
    30
}

fn default_max_retries() -> usize {
    3
}

fn default_retry_base_delay_ms() -> u64 {
    500
}

fn default_retry_max_delay_secs() -> u64 {
    10
}

fn default_user_agent() -> String {
    format!("git2db/{}", crate::VERSION)
}

fn default_max_repo_cache() -> usize {
    100
}

fn default_repo_cache_ttl_secs() -> u64 {
    300
}

fn default_auto_cleanup() -> bool {
    true
}

fn default_cleanup_interval_secs() -> u64 {
    60
}

fn default_max_concurrent_ops() -> usize {
    10
}


impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_network_timeout_secs(),
            max_retries: default_max_retries(),
            retry_base_delay_ms: default_retry_base_delay_ms(),
            retry_max_delay_secs: default_retry_max_delay_secs(),
            proxy_url: None,
            user_agent: default_user_agent(),
            use_credential_helper: default_use_credential_helper(),
            access_token: None,
        }
    }
}

/// Performance and caching configuration
///
/// Field naming convention: All timeout/duration fields use unit suffixes:
/// - `*_secs` = seconds (u64)
/// - Cache sizes use `*_cache` suffix (usize)
///
/// This ensures clear time unit and size specification in configuration files and environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum number of cached repository handles
    #[serde(default = "default_max_repo_cache")]
    pub max_repo_cache: usize,
    /// Repository cache TTL (in seconds)
    #[serde(default = "default_repo_cache_ttl_secs")]
    pub repo_cache_ttl_secs: u64,
    /// Enable automatic cache cleanup
    #[serde(default = "default_auto_cleanup")]
    pub auto_cleanup: bool,
    /// Cleanup interval (in seconds)
    #[serde(default = "default_cleanup_interval_secs")]
    pub cleanup_interval_secs: u64,
    /// Maximum concurrent git operations
    #[serde(default = "default_max_concurrent_ops")]
    pub max_concurrent_ops: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_repo_cache: default_max_repo_cache(),
            repo_cache_ttl_secs: default_repo_cache_ttl_secs(),
            auto_cleanup: default_auto_cleanup(),
            cleanup_interval_secs: default_cleanup_interval_secs(),
            max_concurrent_ops: default_max_concurrent_ops(),
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
            name: "git2db".to_owned(),
            email: "git2db@local".to_owned(),
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
        assert_eq!(config.network.timeout_secs, 30);
        assert_eq!(config.performance.max_repo_cache, 100);
    }

    #[test]
    fn test_git_signature() -> Result<(), git2::Error> {
        let sig = GitSignature::new("Test User", "test@example.com");
        assert_eq!(sig.name, "Test User");
        assert_eq!(sig.email, "test@example.com");

        let git2_sig = sig.to_git2_signature()?;
        assert_eq!(git2_sig.name(), Some("Test User"));
        assert_eq!(git2_sig.email(), Some("test@example.com"));
        Ok(())
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
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay_ms, 500);
        assert_eq!(config.retry_max_delay_secs, 10);
        assert!(config.proxy_url.is_none());
        assert!(config.user_agent.contains("git2db"));
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();
        assert_eq!(config.max_repo_cache, 100);
        assert_eq!(config.repo_cache_ttl_secs, 300);
        assert!(config.auto_cleanup);
        assert_eq!(config.cleanup_interval_secs, 60);
        assert_eq!(config.max_concurrent_ops, 10);
    }

    #[test]
    fn test_git_signature_defaults() {
        let sig = GitSignature::default();
        assert_eq!(sig.name, "git2db");
        assert_eq!(sig.email, "git2db@local");
    }

    #[test]
    fn test_config_serialization() -> Result<(), serde_json::Error> {
        let config = Git2DBConfig::default();

        // Test JSON serialization
        let json = serde_json::to_string(&config)?;
        assert!(json.contains("\"prefer_shallow\":true"));
        assert!(json.contains("\"max_repo_cache\":100"));

        // Test deserialization
        let deserialized: Git2DBConfig = serde_json::from_str(&json)?;
        assert_eq!(
            config.repository.prefer_shallow,
            deserialized.repository.prefer_shallow
        );
        assert_eq!(
            config.performance.max_repo_cache,
            deserialized.performance.max_repo_cache
        );
        Ok(())
    }

    #[test]
    fn test_git_signature_with_special_characters() -> Result<(), git2::Error> {
        let sig = GitSignature::new("Test User with Ünicöde", "test+tag@example.com");
        assert_eq!(sig.name, "Test User with Ünicöde");
        assert_eq!(sig.email, "test+tag@example.com");

        // Should be able to create git2 signature
        let git2_sig = sig.to_git2_signature()?;
        assert_eq!(git2_sig.name(), Some("Test User with Ünicöde"));
        assert_eq!(git2_sig.email(), Some("test+tag@example.com"));
        Ok(())
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
            timeout_secs: 120,
            max_retries: 5,
            retry_base_delay_ms: 1000,
            retry_max_delay_secs: 30,
            proxy_url: Some("http://proxy.example.com:8080".to_owned()),
            user_agent: "test-agent".to_owned(),
            use_credential_helper: true,
            access_token: Some("test-token".to_owned()),
        };
        assert_eq!(network_config.timeout_secs, 120);
        assert_eq!(network_config.max_retries, 5);

        let perf_config = PerformanceConfig {
            max_repo_cache: 500,
            repo_cache_ttl_secs: 600,
            auto_cleanup: false,
            cleanup_interval_secs: 120,
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
        let sig1 = GitSignature::new("User1".to_owned(), "user1@example.com".to_owned());
        assert_eq!(sig1.name, "User1");

        // Test with &str
        let sig2 = GitSignature::new("User2", "user2@example.com");
        assert_eq!(sig2.name, "User2");

        // Test with String references
        let name = "User3".to_owned();
        let email = "user3@example.com".to_owned();
        let sig3 = GitSignature::new(&name, &email);
        assert_eq!(sig3.name, "User3");
    }
}

/// Worktree storage driver configuration
///
/// Configuration now requires explicit driver specification.
/// Auto-detection and fallback behavior have been removed for simplicity.
///
/// # Configuration Sources
///
/// Use environment variables (highest priority):
/// - `GIT2DB__WORKTREE__DRIVER=overlay2`
/// - `GIT2DB__WORKTREE__LOG_DRIVER=true`
///
/// Or config file:
/// ```toml
/// [worktree]
/// driver = "overlay2"  # or "vfs", "reflink"
/// ```
///
/// # Example
///
/// ```toml
/// [worktree]
/// driver = "overlay2"  # or "vfs", "reflink"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeConfig {
    /// Storage driver selection
    ///
    /// Options:
    /// - "overlay2": Linux overlayfs (~80% space savings)
    /// - "vfs": Plain directories (always works, no optimization)
    /// - "reflink": Cross-platform copy-on-write support
    ///
    /// Default: "vfs" (safe fallback)
    #[serde(default = "default_driver")]
    pub driver: String,
}

fn default_driver() -> String {
    "vfs".to_owned()
}

impl Default for WorktreeConfig {
    fn default() -> Self {
        Self {
            driver: "vfs".to_owned(),
        }
    }
}


/// XET large file storage configuration
///
/// Re-exported from git-xet-filter for convenience.
/// The xet-filter crate can be used standalone.
#[cfg(feature = "xet-storage")]
pub use git_xet_filter::XetConfig;
