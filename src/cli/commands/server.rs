use super::config::{ConfigOptionDef, ConfigSection};
use clap::Args;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Default, Deserialize, Args)]
pub struct ServerConfig {
    /// Server host address
    #[arg(long, env = "HYPRSTREAM_SERVER_HOST")]
    pub host: Option<String>,

    /// Server port
    #[arg(long, env = "HYPRSTREAM_SERVER_PORT")]
    pub port: Option<u16>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "HYPRSTREAM_LOG_LEVEL")]
    pub log_level: Option<String>,

    /// Working directory for the server when running in detached mode
    #[arg(long, env = "HYPRSTREAM_WORKING_DIR")]
    pub working_dir: Option<String>,

    /// PID file location when running in detached mode
    #[arg(long, env = "HYPRSTREAM_PID_FILE")]
    pub pid_file: Option<String>,
}

impl ConfigSection for ServerConfig {
    fn options() -> Vec<ConfigOptionDef<String>> {
        vec![
            ConfigOptionDef::new("server.host", "Server host address")
                .with_env("HYPRSTREAM_SERVER_HOST")
                .with_cli("host"),
            ConfigOptionDef::new("server.port", "Server port")
                .with_env("HYPRSTREAM_SERVER_PORT")
                .with_cli("port"),
            ConfigOptionDef::new(
                "server.log_level",
                "Log level (trace, debug, info, warn, error)",
            )
            .with_env("HYPRSTREAM_LOG_LEVEL")
            .with_cli("log-level"),
            ConfigOptionDef::new(
                "server.working_dir",
                "Working directory for the server when running in detached mode",
            )
            .with_env("HYPRSTREAM_WORKING_DIR")
            .with_cli("working-dir"),
            ConfigOptionDef::new(
                "server.pid_file",
                "PID file location when running in detached mode",
            )
            .with_env("HYPRSTREAM_PID_FILE")
            .with_cli("pid-file"),
        ]
    }

    fn from_config<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ServerConfigFile {
            host: Option<String>,
            port: Option<u16>,
            log_level: Option<String>,
            working_dir: Option<String>,
            pid_file: Option<String>,
        }

        let config = ServerConfigFile::deserialize(deserializer)?;
        Ok(ServerConfig {
            host: config.host,
            port: config.port,
            log_level: config.log_level,
            working_dir: config.working_dir,
            pid_file: config.pid_file,
        })
    }
}

#[derive(Debug, Default, Deserialize, Args)]
pub struct EngineConfig {
    /// Primary storage engine type
    #[arg(long = "engine-type", env = "HYPRSTREAM_ENGINE")]
    pub engine_type: Option<String>,

    /// Primary storage engine connection string
    #[arg(long = "engine-connection", env = "HYPRSTREAM_ENGINE_CONNECTION")]
    pub engine_connection: Option<String>,

    /// Primary storage engine options (key=value pairs)
    #[arg(long = "engine-options", env = "HYPRSTREAM_ENGINE_OPTIONS")]
    pub engine_options: Option<Vec<String>>,

    /// Primary storage engine username
    #[arg(long = "engine-username", env = "HYPRSTREAM_ENGINE_USERNAME")]
    pub engine_username: Option<String>,

    /// Primary storage engine password
    #[arg(long = "engine-password", env = "HYPRSTREAM_ENGINE_PASSWORD")]
    pub engine_password: Option<String>,
}

impl ConfigSection for EngineConfig {
    fn options() -> Vec<ConfigOptionDef<String>> {
        vec![
            ConfigOptionDef::new("engine.type", "Primary storage engine type")
                .with_env("HYPRSTREAM_ENGINE")
                .with_cli("engine-type"),
            ConfigOptionDef::new(
                "engine.connection",
                "Primary storage engine connection string",
            )
            .with_env("HYPRSTREAM_ENGINE_CONNECTION")
            .with_cli("engine-connection"),
            ConfigOptionDef::new(
                "engine.options",
                "Primary storage engine options (key=value pairs)",
            )
            .with_env("HYPRSTREAM_ENGINE_OPTIONS")
            .with_cli("engine-options"),
            ConfigOptionDef::new("engine.username", "Primary storage engine username")
                .with_env("HYPRSTREAM_ENGINE_USERNAME")
                .with_cli("engine-username"),
            ConfigOptionDef::new("engine.password", "Primary storage engine password")
                .with_env("HYPRSTREAM_ENGINE_PASSWORD")
                .with_cli("engine-password"),
        ]
    }

    fn from_config<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct EngineConfigFile {
            engine_type: Option<String>,
            engine_connection: Option<String>,
            engine_options: Option<Vec<String>>,
            engine_username: Option<String>,
            engine_password: Option<String>,
        }

        let config = EngineConfigFile::deserialize(deserializer)?;
        Ok(EngineConfig {
            engine_type: config.engine_type,
            engine_connection: config.engine_connection,
            engine_options: config.engine_options,
            engine_username: config.engine_username,
            engine_password: config.engine_password,
        })
    }
}

#[derive(Debug, Default, Deserialize, Args)]
pub struct CacheConfig {
    /// Enable caching
    #[arg(long, env = "HYPRSTREAM_ENABLE_CACHE")]
    pub enabled: Option<bool>,

    /// Cache engine type
    #[arg(long = "cache-type", env = "HYPRSTREAM_CACHE_ENGINE")]
    pub cache_type: Option<String>,

    /// Cache engine connection string
    #[arg(long = "cache-connection", env = "HYPRSTREAM_CACHE_CONNECTION")]
    pub cache_connection: Option<String>,

    /// Cache engine options (key=value pairs)
    #[arg(long = "cache-options", env = "HYPRSTREAM_CACHE_OPTIONS")]
    pub cache_options: Option<Vec<String>>,

    /// Cache maximum duration in seconds
    #[arg(long = "cache-max-duration", env = "HYPRSTREAM_CACHE_MAX_DURATION")]
    pub max_duration: Option<u64>,

    /// Cache engine username
    #[arg(long = "cache-username", env = "HYPRSTREAM_CACHE_USERNAME")]
    pub cache_username: Option<String>,

    /// Cache engine password
    #[arg(long = "cache-password", env = "HYPRSTREAM_CACHE_PASSWORD")]
    pub cache_password: Option<String>,
}

impl ConfigSection for CacheConfig {
    fn options() -> Vec<ConfigOptionDef<String>> {
        vec![
            ConfigOptionDef::new("cache.enabled", "Enable caching")
                .with_env("HYPRSTREAM_ENABLE_CACHE")
                .with_cli("enable-cache"),
            ConfigOptionDef::new("cache.type", "Cache engine type")
                .with_env("HYPRSTREAM_CACHE_ENGINE")
                .with_cli("cache-type"),
            ConfigOptionDef::new("cache.connection", "Cache engine connection string")
                .with_env("HYPRSTREAM_CACHE_CONNECTION")
                .with_cli("cache-connection"),
            ConfigOptionDef::new("cache.options", "Cache engine options (key=value pairs)")
                .with_env("HYPRSTREAM_CACHE_OPTIONS")
                .with_cli("cache-options"),
            ConfigOptionDef::new("cache.max_duration", "Cache maximum duration in seconds")
                .with_env("HYPRSTREAM_CACHE_MAX_DURATION")
                .with_cli("cache-max-duration"),
            ConfigOptionDef::new("cache.username", "Cache engine username")
                .with_env("HYPRSTREAM_CACHE_USERNAME")
                .with_cli("cache-username"),
            ConfigOptionDef::new("cache.password", "Cache engine password")
                .with_env("HYPRSTREAM_CACHE_PASSWORD")
                .with_cli("cache-password"),
        ]
    }

    fn from_config<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CacheConfigFile {
            enabled: Option<bool>,
            cache_type: Option<String>,
            cache_connection: Option<String>,
            cache_options: Option<Vec<String>>,
            max_duration: Option<u64>,
            cache_username: Option<String>,
            cache_password: Option<String>,
        }

        let config = CacheConfigFile::deserialize(deserializer)?;
        Ok(CacheConfig {
            enabled: config.enabled,
            cache_type: config.cache_type,
            cache_connection: config.cache_connection,
            cache_options: config.cache_options,
            max_duration: config.max_duration,
            cache_username: config.cache_username,
            cache_password: config.cache_password,
        })
    }
}

#[derive(Debug, Default, Args)]
pub struct ServerCommand {
    /// Run server in detached mode
    #[arg(short = 'd', long = "detach")]
    pub detach: bool,

    /// Path to the configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    #[command(flatten)]
    pub server: ServerConfig,

    #[command(flatten)]
    pub engine: EngineConfig,

    #[command(flatten)]
    pub cache: CacheConfig,
}
