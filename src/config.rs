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

use crate::cli::commands::server::{CacheConfig, EngineConfig, ServerConfig};
use config::{Config, ConfigError};
use serde::Deserialize;
use std::path::PathBuf;

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");
const DEFAULT_CONFIG_PATH: &str = "/etc/hyprstream/config.toml";

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
}

impl Settings {
    /// Loads configuration from all available sources.
    pub fn new(
        server: ServerConfig,
        engine: EngineConfig,
        cache: CacheConfig,
        config_path: Option<PathBuf>,
    ) -> Result<Self, ConfigError> {
        let mut builder = Config::builder();

        // Load default configuration
        builder = builder.add_source(config::File::from_str(
            DEFAULT_CONFIG,
            config::FileFormat::Toml,
        ));

        // Load system configuration if it exists
        if let Ok(metadata) = std::fs::metadata(DEFAULT_CONFIG_PATH) {
            if metadata.is_file() {
                builder =
                    builder.add_source(config::File::from(PathBuf::from(DEFAULT_CONFIG_PATH)));
            }
        }

        // Load user configuration if specified
        if let Some(ref config_path) = config_path {
            builder = builder.add_source(config::File::from(config_path.clone()));
        }

        // Add environment variables (prefixed with HYPRSTREAM_)
        builder = builder.add_source(config::Environment::with_prefix("HYPRSTREAM"));

        // Build initial settings
        let mut settings: Settings = builder.build()?.try_deserialize()?;

        // Override with command line arguments
        settings.server = server;
        settings.engine = engine;
        settings.cache = cache;

        Ok(settings)
    }
}
