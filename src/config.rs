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

    /// Storage backend type
    #[arg(long, env = "HYPRSTREAM_STORAGE_BACKEND")]
    storage_backend: Option<String>,

    /// Cache backend type
    #[arg(long, env = "HYPRSTREAM_CACHE_BACKEND")]
    cache_backend: Option<String>,

    /// Cache duration in seconds
    #[arg(long, env = "HYPRSTREAM_CACHE_DURATION")]
    cache_duration: Option<i64>,

    /// DuckDB connection string
    #[arg(long, env = "HYPRSTREAM_DUCKDB_CONNECTION")]
    duckdb_connection: Option<String>,

    /// ADBC driver path
    #[arg(long, env = "HYPRSTREAM_ADBC_DRIVER_PATH")]
    driver_path: Option<String>,

    /// Database URL
    #[arg(long, env = "HYPRSTREAM_ADBC_URL")]
    db_url: Option<String>,

    /// Database username
    #[arg(long, env = "HYPRSTREAM_ADBC_USERNAME")]
    db_user: Option<String>,

    /// Database name
    #[arg(long, env = "HYPRSTREAM_ADBC_DATABASE")]
    db_name: Option<String>,
}

/// Complete service configuration.
///
/// This structure holds all configuration options for the service,
/// including server settings, storage backend configuration, and
/// cache settings.
#[derive(Debug, Deserialize)]
pub struct Settings {
    /// Server configuration options
    pub server: ServerConfig,
    /// Storage backend configuration
    pub storage: StorageConfig,
    /// Cache configuration
    pub cache: CacheConfig,
    /// ADBC backend configuration
    pub adbc: AdbcConfig,
    /// DuckDB backend configuration
    pub duckdb: DuckDbConfig,
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

/// Storage backend configuration.
///
/// Specifies which storage backend to use for metric data.
#[derive(Debug, Deserialize)]
pub struct StorageConfig {
    /// Backend type ("duckdb", "adbc", or "cached")
    pub backend: String,
}

/// Cache configuration options.
///
/// Defines caching behavior and expiry policies.
#[derive(Debug, Deserialize)]
pub struct CacheConfig {
    /// Cache backend type ("duckdb" or "adbc")
    pub backend: String,
    /// Cache entry lifetime in seconds
    pub duration_secs: i64,
}

/// DuckDB configuration options.
///
/// Specifies connection settings for DuckDB backend.
#[derive(Debug, Deserialize)]
pub struct DuckDbConfig {
    /// DuckDB connection string (defaults to ":memory:")
    #[serde(default = "default_duckdb_connection")]
    pub connection_string: String,
}

/// ADBC configuration options.
///
/// Defines connection settings for ADBC-compliant databases.
#[derive(Debug, Deserialize)]
pub struct AdbcConfig {
    /// Path to ADBC driver library
    pub driver_path: String,
    /// Database connection URL
    pub url: String,
    /// Database username
    pub username: String,
    /// Database password (optional)
    #[serde(default)]
    pub password: String,
    /// Database name
    pub database: String,
    /// Connection pool settings
    #[serde(default)]
    pub pool: PoolConfig,
}

/// Database connection pool configuration.
///
/// Controls the behavior of the connection pool for ADBC backends.
#[derive(Debug, Default, Deserialize)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    /// Minimum number of connections to maintain
    #[serde(default = "default_min_connections")]
    pub min_connections: u32,
    /// Timeout when acquiring connections (seconds)
    #[serde(default = "default_acquire_timeout")]
    pub acquire_timeout_secs: u32,
}

/// Default DuckDB connection string (in-memory database)
fn default_duckdb_connection() -> String {
    ":memory:".to_string()
}

/// Default maximum connections for the pool
fn default_max_connections() -> u32 {
    10
}

/// Default minimum connections for the pool
fn default_min_connections() -> u32 {
    1
}

/// Default connection acquisition timeout
fn default_acquire_timeout() -> u32 {
    30
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

        // Start with default configuration embedded in the binary
        builder = builder.add_source(config::File::from_str(
            DEFAULT_CONFIG,
            config::FileFormat::Toml,
        ));

        // Add system-wide configuration file if it exists
        builder = builder.add_source(File::with_name(DEFAULT_CONFIG_PATH).required(false));

        // Add user-specified configuration file if provided
        if let Some(config_path) = cli.config {
            builder = builder.add_source(File::from(config_path).required(true));
        }

        // Add command line arguments and environment variables (handled by clap)
        if let Some(host) = cli.host {
            builder = builder.set_override("server.host", host)?;
        }
        if let Some(port) = cli.port {
            builder = builder.set_override("server.port", port)?;
        }
        if let Some(storage_backend) = cli.storage_backend {
            builder = builder.set_override("storage.backend", storage_backend)?;
        }
        if let Some(cache_backend) = cli.cache_backend {
            builder = builder.set_override("cache.backend", cache_backend)?;
        }
        if let Some(duration) = cli.cache_duration {
            builder = builder.set_override("cache.duration_secs", duration)?;
        }
        if let Some(connection) = cli.duckdb_connection {
            builder = builder.set_override("duckdb.connection_string", connection)?;
        }
        if let Some(driver_path) = cli.driver_path {
            builder = builder.set_override("adbc.driver_path", driver_path)?;
        }
        if let Some(url) = cli.db_url {
            builder = builder.set_override("adbc.url", url)?;
        }
        if let Some(username) = cli.db_user {
            builder = builder.set_override("adbc.username", username)?;
        }
        if let Some(database) = cli.db_name {
            builder = builder.set_override("adbc.database", database)?;
        }

        // Build and deserialize
        builder.build()?.try_deserialize()
    }
}
