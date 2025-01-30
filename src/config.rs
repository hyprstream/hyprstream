//! Configuration management for Hyprstream service.
//!
//! This module provides configuration handling through multiple sources:
//! 1. Default configuration (embedded in binary)
//! 2. System-wide configuration file (`/etc/hyprstream/config.toml`)
//! 3. User-specified configuration file
//! 4. Environment variables (prefixed with `HYPRSTREAM_`)
//! 5. Command-line arguments
//!
//! Configuration options are loaded in order of precedence, with later sources
//! overriding earlier ones.

use crate::cli::commands::{
    config::LoggingConfig,
    server::{CacheConfig, EngineConfig, ServerConfig},
};
use config::{Config, ConfigError};
use config::builder::{ConfigBuilder, DefaultState};
use config::File;
use serde::Deserialize;
use std::path::PathBuf;
use tonic::transport::{Identity, Certificate};

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");
const DEFAULT_CONFIG_PATH: &str = "/etc/hyprstream/config.toml";

/// Set TLS configuration using certificate data directly
pub fn set_tls_data(
    builder: ConfigBuilder<DefaultState>,
    cert: &[u8],
    key: &[u8],
    ca: Option<&[u8]>,
) -> Result<ConfigBuilder<DefaultState>, ConfigError> {
    // Store the TLS configuration
    let mut builder = builder
        .set_default("tls.enabled", true)?
        .set_default("tls.cert_data", cert.to_vec())?
        .set_default("tls.key_data", key.to_vec())?;

    if let Some(ca_data) = ca {
        builder = builder.set_default("tls.ca_data", ca_data.to_vec())?;
    }

    Ok(builder)
}

/// Get TLS configuration from Config
pub fn get_tls_config(config: &Config) -> Option<(Identity, Option<Certificate>)> {
    if !config.get_bool("tls.enabled").unwrap_or(false) {
        return None;
    }

    // Try to get certificate data first
    let cert_result = config.get::<Vec<u8>>("tls.cert_data");
    let key_result = config.get::<Vec<u8>>("tls.key_data");
    let ca_result = config.get::<Vec<u8>>("tls.ca_data");

    if let (Ok(cert), Ok(key)) = (cert_result, key_result) {
        let identity = Identity::from_pem(&cert, &key);
        let ca_cert = ca_result.ok().map(|ca| Certificate::from_pem(&ca));
        return Some((identity, ca_cert));
    }

    // Fall back to loading from files
    let cert_path = config.get_string("tls.cert_path").ok()?;
    let key_path = config.get_string("tls.key_path").ok()?;
    if cert_path.is_empty() || key_path.is_empty() {
        return None;
    }

    let cert = std::fs::read(cert_path).ok()?;
    let key = std::fs::read(key_path).ok()?;
    let identity = Identity::from_pem(&cert, &key);

    let ca_cert = config
        .get_string("tls.ca_path")
        .ok()
        .filter(|p| !p.is_empty())
        .and_then(|p| std::fs::read(p).ok())
        .map(|ca| Certificate::from_pem(&ca));

    Some((identity, ca_cert))
}

/// Complete service configuration.
///
/// This structure holds all configuration options for the service,
/// including server settings, storage backend configuration, and
/// cache settings.
#[derive(Debug, Deserialize)]
pub struct Settings {
    /// Server configuration
    #[serde(default)]
    pub server: ServerConfig,
    /// Engine configuration
    #[serde(default)]
    pub engine: EngineConfig,
    /// Cache configuration
    #[serde(default)]
    pub cache: CacheConfig,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Settings {
    /// Loads configuration from all available sources.
    pub fn new(
        server: ServerConfig,
        engine: EngineConfig,
        cache: CacheConfig,
        logging: LoggingConfig,
        config_path: Option<PathBuf>,
    ) -> Result<Self, ConfigError> {
        let mut builder = ::config::Config::builder();

        // Load default configuration
        builder = builder.add_source(File::from_str(
            DEFAULT_CONFIG,
            config::FileFormat::Toml,
        ));

        // Load system configuration if it exists
        if let Ok(metadata) = std::fs::metadata(DEFAULT_CONFIG_PATH) {
            if metadata.is_file() {
                builder =
                    builder.add_source(File::from(PathBuf::from(DEFAULT_CONFIG_PATH)));
            }
        }

        // Load user configuration if specified
        if let Some(ref config_path) = config_path {
            builder = builder.add_source(File::from(config_path.clone()));
        }

        // Add environment variables (prefixed with HYPRSTREAM_)
        builder = builder.add_source(config::Environment::with_prefix("HYPRSTREAM"));

        // Build initial settings
        let mut settings: Settings = builder.build()?.try_deserialize()?;

        // Override with command line arguments
        settings.server = server;
        settings.engine = engine;
        settings.cache = cache;
        settings.logging = logging;

        Ok(settings)
    }
}
