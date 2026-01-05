use super::config::{ConfigOptionDef, ConfigSection, LoggingConfig};
use clap::Args;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Args)]
pub struct ServerConfig {
    /// Server host address
    #[arg(long, env = "HYPRSTREAM_SERVER_HOST")]
    pub host: Option<String>,

    /// Server port
    #[arg(long, env = "HYPRSTREAM_SERVER_PORT")]
    pub port: Option<u16>,

    /// Working directory for the server when running in detached mode
    #[arg(long, env = "HYPRSTREAM_WORKING_DIR")]
    pub working_dir: Option<String>,

    /// PID file location when running in detached mode
    #[arg(long, env = "HYPRSTREAM_PID_FILE")]
    pub pid_file: Option<String>,

    /// Path to TLS certificate file
    #[arg(long = "tls-cert", env = "HYPRSTREAM_TLS_CERT")]
    pub tls_cert: Option<PathBuf>,

    /// Path to TLS private key file
    #[arg(long = "tls-key", env = "HYPRSTREAM_TLS_KEY")]
    pub tls_key: Option<PathBuf>,

    /// Path to CA certificate for client authentication (enables mTLS)
    #[arg(long = "tls-client-ca", env = "HYPRSTREAM_TLS_CLIENT_CA")]
    pub tls_client_ca: Option<PathBuf>,

    /// Minimum TLS version (1.2|1.3)
    #[arg(long, env = "HYPRSTREAM_TLS_MIN_VERSION")]
    pub tls_min_version: Option<String>,

    /// Allowed TLS cipher suites
    #[arg(long, env = "HYPRSTREAM_TLS_CIPHER_LIST")]
    pub tls_cipher_list: Option<String>,

    /// Prefer server cipher order
    #[arg(long, env = "HYPRSTREAM_TLS_PREFER_SERVER_CIPHERS")]
    pub tls_prefer_server_ciphers: Option<bool>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: None,
            port: None,
            working_dir: None,
            pid_file: None,
            tls_cert: None,
            tls_key: None,
            tls_client_ca: None,
            tls_min_version: Some("1.2".to_string()),
            tls_cipher_list: None,
            tls_prefer_server_ciphers: Some(true),
        }
    }
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
            ConfigOptionDef::new("server.tls_cert", "Path to TLS certificate file")
                .with_env("HYPRSTREAM_TLS_CERT")
                .with_cli("tls-cert"),
            ConfigOptionDef::new("server.tls_key", "Path to TLS private key file")
                .with_env("HYPRSTREAM_TLS_KEY")
                .with_cli("tls-key"),
            ConfigOptionDef::new(
                "server.tls_client_ca",
                "Path to CA certificate for client authentication (enables mTLS)",
            )
            .with_env("HYPRSTREAM_TLS_CLIENT_CA")
            .with_cli("tls-client-ca"),
            ConfigOptionDef::new(
                "server.tls_min_version",
                "Minimum TLS version (1.2|1.3)",
            )
            .with_env("HYPRSTREAM_TLS_MIN_VERSION")
            .with_cli("tls-min-version"),
            ConfigOptionDef::new(
                "server.tls_cipher_list",
                "Allowed TLS cipher suites",
            )
            .with_env("HYPRSTREAM_TLS_CIPHER_LIST")
            .with_cli("tls-cipher-list"),
            ConfigOptionDef::new(
                "server.tls_prefer_server_ciphers",
                "Prefer server cipher order",
            )
            .with_env("HYPRSTREAM_TLS_PREFER_SERVER_CIPHERS")
            .with_cli("tls-prefer-server-ciphers"),
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
            working_dir: Option<String>,
            pid_file: Option<String>,
            tls_cert: Option<PathBuf>,
            tls_key: Option<PathBuf>,
            tls_client_ca: Option<PathBuf>,
            tls_min_version: Option<String>,
            tls_cipher_list: Option<String>,
            tls_prefer_server_ciphers: Option<bool>,
        }

        let config = ServerConfigFile::deserialize(deserializer)?;
        Ok(ServerConfig {
            host: config.host,
            port: config.port,
            working_dir: config.working_dir,
            pid_file: config.pid_file,
            tls_cert: config.tls_cert,
            tls_key: config.tls_key,
            tls_client_ca: config.tls_client_ca,
            tls_min_version: config.tls_min_version.or(Some("1.2".to_string())),
            tls_cipher_list: config.tls_cipher_list,
            tls_prefer_server_ciphers: config.tls_prefer_server_ciphers.or(Some(true)),
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

#[derive(Debug, Args)]
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

    #[command(flatten)]
    pub logging: LoggingConfig,
}
