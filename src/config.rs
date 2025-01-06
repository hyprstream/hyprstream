//! Configuration management for Hyprstream service.
//!
//! This module provides configuration handling through multiple sources:
//! 1. Default configuration (embedded in binary)
//! 2. System-wide configuration file (`/etc/hyprstream/config.toml`)
//! 3. User-specified configuration file
//! 4. Environment variables
//! 5. Command-line arguments
//!
//! Configuration options are loaded in order of precedence, with later sources
//! overriding earlier ones.

use clap::Parser;
use config::{Config, ConfigError, File};
use serde::Deserialize;
use std::path::PathBuf;

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");
const DEFAULT_CONFIG_PATH: &str = "/etc/hyprstream/config.toml";

/// Command-line arguments parser.
///
/// This structure defines all available command-line options and their
/// corresponding environment variables. It uses clap for parsing and
/// supports both short and long option forms.
#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct CliArgs {
    /// Path to the configuration file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Server host address
    #[arg(long, env = "HYPRSTREAM_SERVER_HOST")]
    host: Option<String>,

    /// Server port
    #[arg(long, env = "HYPRSTREAM_SERVER_PORT")]
    port: Option<u16>,

    /// Primary storage engine type
    #[arg(long, env = "HYPRSTREAM_ENGINE")]
    engine: Option<String>,

    /// Primary storage engine connection string
    #[arg(long, env = "HYPRSTREAM_ENGINE_CONNECTION")]
    engine_connection: Option<String>,

    /// Primary storage engine options (key=value pairs)
    #[arg(long, env = "HYPRSTREAM_ENGINE_OPTIONS")]
    engine_options: Option<Vec<String>>,

    /// Enable caching
    #[arg(long, env = "HYPRSTREAM_ENABLE_CACHE")]
    enable_cache: Option<bool>,

    /// Cache engine type
    #[arg(long, env = "HYPRSTREAM_CACHE_ENGINE")]
    cache_engine: Option<String>,

    /// Cache engine connection string
    #[arg(long, env = "HYPRSTREAM_CACHE_CONNECTION")]
    cache_connection: Option<String>,

    /// Cache engine options (key=value pairs)
    #[arg(long, env = "HYPRSTREAM_CACHE_OPTIONS")]
    cache_options: Option<Vec<String>>,

    /// Cache maximum duration in seconds
    #[arg(long, env = "HYPRSTREAM_CACHE_MAX_DURATION")]
    cache_max_duration: Option<u64>,
}

/// Complete service configuration.
///
/// This structure holds all configuration options for the service,
/// including server settings, storage backend configuration, and
/// cache settings.
#[derive(Debug, Deserialize)]
pub struct Settings {
    /// Server configuration
    pub server: ServerConfig,
    /// Engine configuration
    pub engine: EngineConfig,
    /// Cache configuration
    pub cache: CacheConfig,
}

/// Server configuration options.
///
/// Defines the network interface and port for the Flight SQL service.
#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to
    pub host: String,
    /// Port number to listen on
    pub port: u16,
}

/// Engine configuration.
///
/// Specifies the primary storage engine to use for metric data.
#[derive(Debug, Deserialize)]
pub struct EngineConfig {
    /// Engine type ("duckdb" or "adbc")
    pub engine: String,
    /// Connection string for the engine
    pub connection: String,
    /// Engine-specific options
    #[serde(default)]
    pub options: std::collections::HashMap<String, String>,
}

/// Cache configuration options.
///
/// Defines caching behavior and expiry policies.
#[derive(Debug, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Cache engine type ("duckdb" or "adbc")
    #[serde(default = "default_cache_engine")]
    pub engine: String,
    /// Cache connection string
    #[serde(default = "default_cache_connection")]
    pub connection: String,
    /// Cache engine-specific options
    #[serde(default)]
    pub options: std::collections::HashMap<String, String>,
    /// Maximum cache duration in seconds
    #[serde(default = "default_cache_duration")]
    pub max_duration_secs: u64,
}

fn default_cache_engine() -> String {
    "duckdb".to_string()
}

fn default_cache_connection() -> String {
    ":memory:".to_string()
}

fn default_cache_duration() -> u64 {
    3600
}

impl Settings {
    /// Creates a new Settings instance from all configuration sources.
    ///
    /// This method loads and merges configuration from:
    /// 1. Default configuration
    /// 2. System configuration file
    /// 3. User configuration file
    /// 4. Environment variables
    /// 5. Command-line arguments
    ///
    /// Later sources override earlier ones, allowing for flexible configuration
    /// management.
    ///
    /// # Returns
    ///
    /// * `Result<Settings, ConfigError>` - Parsed settings or error
    pub fn new() -> Result<Self, ConfigError> {
        let cli = CliArgs::parse();
        let mut builder = Config::builder();

        // Start with default configuration
        builder = builder.add_source(config::File::from_str(
            DEFAULT_CONFIG,
            config::FileFormat::Toml,
        ));

        // Add system-wide configuration
        builder = builder.add_source(File::with_name(DEFAULT_CONFIG_PATH).required(false));

        // Add user configuration file
        if let Some(config_path) = cli.config {
            builder = builder.add_source(File::from(config_path).required(true));
        }

        // Add CLI overrides
        if let Some(host) = cli.host {
            builder = builder.set_override("server.host", host)?;
        }
        if let Some(port) = cli.port {
            builder = builder.set_override("server.port", port)?;
        }
        if let Some(engine) = cli.engine {
            builder = builder.set_override("engine.engine", engine)?;
        }
        if let Some(connection) = cli.engine_connection {
            builder = builder.set_override("engine.connection", connection)?;
        }
        if let Some(options) = cli.engine_options {
            let options: std::collections::HashMap<String, String> = options
                .into_iter()
                .filter_map(|opt| {
                    let parts: Vec<&str> = opt.split('=').collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
                .collect();
            builder = builder.set_override("engine.options", options)?;
        }

        // Cache settings
        if let Some(enabled) = cli.enable_cache {
            builder = builder.set_override("cache.enabled", enabled)?;
        }
        if let Some(engine) = cli.cache_engine {
            builder = builder.set_override("cache.engine", engine)?;
        }
        if let Some(connection) = cli.cache_connection {
            builder = builder.set_override("cache.connection", connection)?;
        }
        if let Some(options) = cli.cache_options {
            let options: std::collections::HashMap<String, String> = options
                .into_iter()
                .filter_map(|opt| {
                    let parts: Vec<&str> = opt.split('=').collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
                .collect();
            builder = builder.set_override("cache.options", options)?;
        }
        if let Some(duration) = cli.cache_max_duration {
            builder = builder.set_override("cache.max_duration_secs", duration)?;
        }

        // Build and deserialize
        builder.build()?.try_deserialize()
    }
}
