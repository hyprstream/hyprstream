use clap::Parser;
use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");
const DEFAULT_CONFIG_PATH: &str = "/etc/hyprstream/config.toml";

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

    /// Cache duration in seconds
    #[arg(long, env = "HYPRSTREAM_CACHE_DURATION")]
    cache_duration: Option<i64>,

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

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub server: ServerConfig,
    pub cache: CacheConfig,
    pub adbc: AdbcConfig,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct CacheConfig {
    pub duration_secs: i64,
}

#[derive(Debug, Deserialize)]
pub struct AdbcConfig {
    pub driver_path: String,
    pub url: String,
    pub username: String,
    #[serde(default)]
    pub password: String,
    pub database: String,
    #[serde(default)]
    pub options: HashMap<String, String>,
    #[serde(default)]
    pub pool: PoolConfig,
}

#[derive(Debug, Default, Deserialize)]
pub struct PoolConfig {
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    #[serde(default = "default_min_connections")]
    pub min_connections: u32,
    #[serde(default = "default_acquire_timeout")]
    pub acquire_timeout_secs: u32,
}

fn default_max_connections() -> u32 { 10 }
fn default_min_connections() -> u32 { 1 }
fn default_acquire_timeout() -> u32 { 30 }

impl Settings {
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
        if let Some(duration) = cli.cache_duration {
            builder = builder.set_override("cache.duration_secs", duration)?;
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

impl AdbcConfig {
    pub fn into_driver_options(&self) -> Vec<(&str, String)> {
        let mut options = vec![
            ("driver", self.driver_path.clone()),
            ("url", self.url.clone()),
            ("username", self.username.clone()),
            ("password", self.password.clone()),
            ("database", self.database.clone()),
        ];

        // Add any additional database-specific options
        for (key, value) in &self.options {
            options.push((key.as_str(), value.clone()));
        }

        // Add pool configuration
        options.push(("pool.max_connections", self.pool.max_connections.to_string()));
        options.push(("pool.min_connections", self.pool.min_connections.to_string()));
        options.push(("pool.acquire_timeout", self.pool.acquire_timeout_secs.to_string()));

        options
    }
}
