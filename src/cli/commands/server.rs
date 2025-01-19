use clap::Args;
use std::path::PathBuf;

#[derive(Args)]
pub struct ServerCommand {
    /// Run server in detached mode
    #[arg(short = 'd', long)]
    pub detach: bool,

    /// Path to the configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

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

    /// Primary storage engine type
    #[arg(long, env = "HYPRSTREAM_ENGINE")]
    pub engine: Option<String>,

    /// Primary storage engine connection string
    #[arg(long, env = "HYPRSTREAM_ENGINE_CONNECTION")]
    pub engine_connection: Option<String>,

    /// Primary storage engine options (key=value pairs)
    #[arg(long, env = "HYPRSTREAM_ENGINE_OPTIONS")]
    pub engine_options: Option<Vec<String>>,

    /// Enable caching
    #[arg(long, env = "HYPRSTREAM_ENABLE_CACHE")]
    pub enable_cache: Option<bool>,

    /// Cache engine type
    #[arg(long, env = "HYPRSTREAM_CACHE_ENGINE")]
    pub cache_engine: Option<String>,

    /// Cache engine connection string
    #[arg(long, env = "HYPRSTREAM_CACHE_CONNECTION")]
    pub cache_connection: Option<String>,

    /// Cache engine options (key=value pairs)
    #[arg(long, env = "HYPRSTREAM_CACHE_OPTIONS")]
    pub cache_options: Option<Vec<String>>,

    /// Cache maximum duration in seconds
    #[arg(long, env = "HYPRSTREAM_CACHE_MAX_DURATION")]
    pub cache_max_duration: Option<u64>,

    /// Primary storage engine username
    #[arg(long, env = "HYPRSTREAM_ENGINE_USERNAME")]
    pub engine_username: Option<String>,

    /// Primary storage engine password
    #[arg(long, env = "HYPRSTREAM_ENGINE_PASSWORD")]
    pub engine_password: Option<String>,

    /// Cache engine username
    #[arg(long, env = "HYPRSTREAM_CACHE_USERNAME")]
    pub cache_username: Option<String>,

    /// Cache engine password
    #[arg(long, env = "HYPRSTREAM_CACHE_PASSWORD")]
    pub cache_password: Option<String>,
}

impl From<&ServerCommand> for crate::config::CliArgs {
    fn from(cmd: &ServerCommand) -> Self {
        Self {
            config: cmd.config.clone(),
            host: cmd.host.clone(),
            port: cmd.port,
            log_level: cmd.log_level.clone(),
            working_dir: cmd.working_dir.clone(),
            pid_file: cmd.pid_file.clone(),
            engine: cmd.engine.clone(),
            engine_connection: cmd.engine_connection.clone(),
            engine_options: cmd.engine_options.clone(),
            enable_cache: cmd.enable_cache,
            cache_engine: cmd.cache_engine.clone(),
            cache_connection: cmd.cache_connection.clone(),
            cache_options: cmd.cache_options.clone(),
            cache_max_duration: cmd.cache_max_duration,
            engine_username: cmd.engine_username.clone(),
            engine_password: cmd.engine_password.clone(),
            cache_username: cmd.cache_username.clone(),
            cache_password: cmd.cache_password.clone(),
        }
    }
}