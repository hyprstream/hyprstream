//! Configuration for XET filter

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// HuggingFace XET CAS endpoint
pub const HUGGINGFACE_XET_ENDPOINT: &str = "https://transfer.xethub.hf.co";

/// Read token from HuggingFace CLI config file
///
/// Token path resolution (matching huggingface_hub/constants.py):
/// 1. `$HF_TOKEN_PATH` if set (explicit override)
/// 2. `$HF_HOME/token` if `HF_HOME` is set
/// 3. `~/.cache/huggingface/token` (default)
fn read_huggingface_token_file() -> Option<String> {
    let token_path = if let Ok(path) = std::env::var("HF_TOKEN_PATH") {
        PathBuf::from(path)
    } else if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("token")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("token")
    } else {
        return None;
    };

    std::fs::read_to_string(&token_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Configuration for XET large file storage
#[derive(Clone, Serialize, Deserialize)]
pub struct XetConfig {
    /// CAS endpoint URL
    ///
    /// Can be set via:
    /// - `GIT2DB_XET_ENDPOINT` environment variable
    /// - `XETHUB_ENDPOINT` environment variable
    /// - Programmatic configuration
    ///
    /// If empty, XET is disabled.
    pub endpoint: String,

    /// Authentication token
    ///
    /// Resolved in priority order:
    /// 1. `XETHUB_TOKEN` environment variable
    /// 2. `HF_TOKEN` environment variable
    /// 3. HuggingFace CLI token file (`~/.cache/huggingface/token`)
    pub token: Option<String>,

    /// Compression scheme
    #[cfg(feature = "xet-storage")]
    #[serde(skip)]
    pub compression: Option<cas_object::CompressionScheme>,

    #[cfg(not(feature = "xet-storage"))]
    #[serde(skip)]
    pub compression: Option<()>,
}

impl std::fmt::Debug for XetConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XetConfig")
            .field("endpoint", &self.endpoint)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("compression", &self.compression)
            .finish()
    }
}

impl Default for XetConfig {
    fn default() -> Self {
        // Check environment variables for endpoint (in priority order)
        let endpoint = std::env::var("GIT2DB_XET_ENDPOINT")
            .or_else(|_| std::env::var("XETHUB_ENDPOINT"))
            .unwrap_or_default();

        // Token priority (highest to lowest):
        // 1. XETHUB_TOKEN env var (explicit XET token)
        // 2. HF_TOKEN env var (explicit HuggingFace token)
        // 3. HuggingFace CLI token file (~/.cache/huggingface/token)
        let token = std::env::var("XETHUB_TOKEN")
            .or_else(|_| std::env::var("HF_TOKEN"))
            .ok()
            .or_else(read_huggingface_token_file);

        Self {
            endpoint,
            token,
            compression: None,
        }
    }
}

impl XetConfig {
    /// Create a new XET configuration with the given endpoint
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Default::default()
        }
    }

    /// Create a configuration for HuggingFace repositories
    ///
    /// Uses the HuggingFace XET endpoint and automatically picks up
    /// authentication from `HF_TOKEN` or `XETHUB_TOKEN` environment variables.
    pub fn huggingface() -> Self {
        Self {
            endpoint: HUGGINGFACE_XET_ENDPOINT.to_string(),
            ..Default::default()
        }
    }

    /// Check if this configuration is for HuggingFace
    pub fn is_huggingface(&self) -> bool {
        self.endpoint.contains("xethub.hf.co")
    }

    /// Check if XET is enabled (endpoint is not empty)
    pub fn is_enabled(&self) -> bool {
        !self.endpoint.is_empty()
    }

    /// Set the authentication token
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Set the compression scheme
    #[cfg(feature = "xet-storage")]
    pub fn with_compression(mut self, compression: cas_object::CompressionScheme) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Create XET config appropriate for a git URL
    ///
    /// Resolution order:
    /// 1. Environment variable (XETHUB_ENDPOINT/GIT2DB_XET_ENDPOINT) - always takes priority
    /// 2. URL pattern matching - convenience for known providers
    /// 3. None - XET disabled for this URL
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // With env var set, always uses that endpoint
    /// std::env::set_var("XETHUB_ENDPOINT", "https://my.xet.server");
    /// let config = XetConfig::for_url("https://github.com/any/repo");
    /// assert!(config.is_some());
    ///
    /// // Without env var, uses URL pattern matching
    /// std::env::remove_var("XETHUB_ENDPOINT");
    /// let config = XetConfig::for_url("https://hf.co/Qwen/Qwen3-0.6B");
    /// assert!(config.unwrap().is_huggingface());
    ///
    /// // Unknown URL with no env var = None
    /// let config = XetConfig::for_url("https://github.com/unknown/repo");
    /// assert!(config.is_none());
    /// ```
    pub fn for_url(url: &str) -> Option<Self> {
        // Priority 1: Environment variable always wins
        let default = Self::default();
        if default.is_enabled() {
            return Some(default);
        }

        // Priority 2: Known provider patterns (convenience only)
        if url.contains("huggingface.co") || url.contains("hf.co") {
            return Some(Self::huggingface());
        }
        // Future providers can be added here as simple pattern matches

        // No XET endpoint configured for this URL
        None
    }

    /// Check if the endpoint uses SSH transport
    pub fn is_ssh_transport(&self) -> bool {
        self.endpoint.starts_with("ssh://")
    }

    /// Check if the endpoint uses HTTPS transport
    pub fn is_https_transport(&self) -> bool {
        self.endpoint.starts_with("https://") || self.endpoint.starts_with("http://")
    }

    /// Create the appropriate storage backend based on the endpoint URL.
    ///
    /// - `ssh://` endpoints use [`SshStorage`] (requires `ssh-transport` feature)
    /// - `https://` endpoints use [`XetStorage`]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = XetConfig::new("ssh://user@host/path/to/cas");
    /// let storage = config.create_storage().await?;
    /// ```
    #[cfg(feature = "xet-storage")]
    pub async fn create_storage(&self) -> crate::error::Result<Box<dyn crate::storage::StorageBackend>> {
        #[cfg(feature = "ssh-transport")]
        if self.is_ssh_transport() {
            let storage = crate::ssh_client::SshStorage::connect(&self.endpoint).await?;
            return Ok(Box::new(storage));
        }

        #[cfg(not(feature = "ssh-transport"))]
        if self.is_ssh_transport() {
            return Err(crate::error::XetError::new(
                crate::error::XetErrorKind::InvalidConfig,
                "SSH transport requires the 'ssh-transport' feature",
            ));
        }

        // Default to HTTPS transport
        let storage = crate::storage::XetStorage::new(self).await?;
        Ok(Box::new(storage))
    }
}

// NOTE: Tests that modify environment variables use ENV_MUTEX to avoid race conditions.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to serialize tests that modify environment variables
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    /// Helper to clear XET-related env vars
    fn clear_xet_env_vars() {
        std::env::remove_var("GIT2DB_XET_ENDPOINT");
        std::env::remove_var("XETHUB_ENDPOINT");
    }

    #[test]
    fn test_default_config_disabled_without_env() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_xet_env_vars();

        let config = XetConfig::default();
        // XET is disabled by default when no env vars set
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_config_builder() {
        // No env var access, no lock needed
        let config = XetConfig::new("https://custom.endpoint.dev")
            .with_token("test_token");

        assert_eq!(config.endpoint, "https://custom.endpoint.dev");
        assert_eq!(config.token, Some("test_token".to_string()));
        assert!(config.is_enabled());
    }

    #[test]
    fn test_huggingface_config() {
        // No env var access, no lock needed
        let config = XetConfig::huggingface();

        assert_eq!(config.endpoint, HUGGINGFACE_XET_ENDPOINT);
        assert!(config.is_huggingface());
        assert!(config.is_enabled());
    }

    #[test]
    fn test_for_url_huggingface() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_xet_env_vars();

        // HuggingFace URLs should return huggingface config
        let config = XetConfig::for_url("https://huggingface.co/Qwen/Qwen3-0.6B");
        assert!(config.is_some());
        assert!(config.unwrap().is_huggingface());

        // Short hf.co URLs should also work
        let config = XetConfig::for_url("https://hf.co/Qwen/Qwen3-0.6B");
        assert!(config.is_some());
        assert!(config.unwrap().is_huggingface());
    }

    #[test]
    fn test_for_url_unknown() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_xet_env_vars();

        // Unknown URLs should return None
        let config = XetConfig::for_url("https://github.com/some/repo");
        assert!(config.is_none());
    }

    #[test]
    fn test_for_url_env_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_xet_env_vars();

        // Set env var - should take priority over URL pattern
        std::env::set_var("XETHUB_ENDPOINT", "https://custom.xet.server");

        let config = XetConfig::for_url("https://github.com/any/repo");
        assert!(config.is_some());
        assert_eq!(config.unwrap().endpoint, "https://custom.xet.server");

        // Clean up
        std::env::remove_var("XETHUB_ENDPOINT");
    }

    #[test]
    fn test_for_url_with_trailing_slash() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_xet_env_vars();

        // URLs with trailing slash should still match
        let config = XetConfig::for_url("https://huggingface.co/Qwen/Model/");
        assert!(config.is_some());
        assert!(config.unwrap().is_huggingface());
    }

    #[test]
    fn test_debug_redacts_token() {
        // No env var access, no lock needed
        let config = XetConfig::new("https://example.com")
            .with_token("secret_token_12345");

        let debug_str = format!("{:?}", config);

        // Token should be redacted
        assert!(debug_str.contains("[REDACTED]"));
        assert!(!debug_str.contains("secret_token_12345"));
        // Endpoint should still be visible
        assert!(debug_str.contains("https://example.com"));
    }

    #[test]
    fn test_debug_without_token() {
        // No env var access, no lock needed
        let config = XetConfig::new("https://example.com");

        let debug_str = format!("{:?}", config);

        // Should show None for token (no redaction needed)
        assert!(debug_str.contains("None"));
        assert!(debug_str.contains("https://example.com"));
    }
}
