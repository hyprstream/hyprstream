//! Configuration for XET filter

use serde::{Deserialize, Serialize};

/// Configuration for XET large file storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XetConfig {
    /// CAS endpoint URL (default: "https://cas.xet.dev")
    pub endpoint: String,

    /// Authentication token (from env XETHUB_TOKEN or config)
    pub token: Option<String>,

    /// Compression scheme
    #[cfg(feature = "xet-storage")]
    #[serde(skip)]
    pub compression: Option<cas_object::CompressionScheme>,

    #[cfg(not(feature = "xet-storage"))]
    #[serde(skip)]
    pub compression: Option<()>,
}

impl Default for XetConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://cas.xet.dev".to_string(),
            token: std::env::var("XETHUB_TOKEN").ok(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = XetConfig::default();
        assert_eq!(config.endpoint, "https://cas.xet.dev");
    }

    #[test]
    fn test_config_builder() {
        let config = XetConfig::new("https://custom.endpoint.dev")
            .with_token("test_token");

        assert_eq!(config.endpoint, "https://custom.endpoint.dev");
        assert_eq!(config.token, Some("test_token".to_string()));
    }
}
