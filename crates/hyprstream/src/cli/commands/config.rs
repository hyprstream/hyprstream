use clap::Args;
use serde::Deserialize;

/// Trait for configuration options that can be set via CLI, env vars, or config file
pub trait ConfigOption {
    /// Get the environment variable name for this option
    fn env_var(&self) -> Option<&'static str>;

    /// Get the long CLI flag for this option
    fn cli_flag(&self) -> Option<&'static str>;

    /// Get the short CLI flag for this option
    fn cli_short(&self) -> Option<char>;

    /// Get the config file path for this option (e.g., "server.host")
    fn config_path(&self) -> &'static str;

    /// Get the help text for this option
    fn help_text(&self) -> &'static str;
}

/// A configuration option with its metadata
#[derive(Debug, Clone)]
pub struct ConfigOptionDef<T> {
    /// The current value of the option
    pub value: Option<T>,
    /// The environment variable name
    pub env_var: Option<&'static str>,
    /// The long CLI flag
    pub cli_flag: Option<&'static str>,
    /// The short CLI flag
    pub cli_short: Option<char>,
    /// The config file path
    pub config_path: &'static str,
    /// The help text
    pub help_text: &'static str,
}

impl<T> ConfigOptionDef<T> {
    pub fn new(config_path: &'static str, help_text: &'static str) -> Self {
        Self {
            value: None,
            env_var: None,
            cli_flag: None,
            cli_short: None,
            config_path,
            help_text,
        }
    }

    pub fn with_env(mut self, env_var: &'static str) -> Self {
        self.env_var = Some(env_var);
        self
    }

    pub fn with_cli(mut self, flag: &'static str) -> Self {
        self.cli_flag = Some(flag);
        self
    }

    pub fn with_short(mut self, short: char) -> Self {
        self.cli_short = Some(short);
        self
    }
}

impl<T> ConfigOption for ConfigOptionDef<T> {
    fn env_var(&self) -> Option<&'static str> {
        self.env_var
    }

    fn cli_flag(&self) -> Option<&'static str> {
        self.cli_flag
    }

    fn cli_short(&self) -> Option<char> {
        self.cli_short
    }

    fn config_path(&self) -> &'static str {
        self.config_path
    }

    fn help_text(&self) -> &'static str {
        self.help_text
    }
}

/// Trait for configuration sections that can be loaded from multiple sources
pub trait ConfigSection: Sized + Default {
    /// Get all configuration options for this section
    fn options() -> Vec<ConfigOptionDef<String>>;

    /// Create a new instance from a config file section
    fn from_config<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>;
}

/// Logging configuration that can be set via CLI, env vars, or config file
#[derive(Debug, Clone, Default, Args, Deserialize)]
pub struct LoggingConfig {
    /// Enable verbose logging (-v for debug, -vv for trace)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    #[serde(skip)]
    pub verbose: u8,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long = "log-level", env = "HYPRSTREAM_LOG_LEVEL")]
    pub log_level: Option<String>,

    /// Log filter directives
    #[arg(long = "log-filter", env = "HYPRSTREAM_LOG_FILTER")]
    pub log_filter: Option<String>,
}

impl LoggingConfig {
    pub fn get_effective_level(&self) -> &str {
        match (self.verbose, self.log_level.as_deref()) {
            (2, _) => "trace",         // -vv flag
            (1, _) => "debug",         // -v flag
            (0, Some(level)) => level, // Configured level
            _ => "info",               // Default
        }
    }
}

impl ConfigSection for LoggingConfig {
    fn options() -> Vec<ConfigOptionDef<String>> {
        vec![
            ConfigOptionDef::new(
                "logging.level",
                "Log level (trace, debug, info, warn, error)",
            )
            .with_env("HYPRSTREAM_LOG_LEVEL")
            .with_cli("log-level"),
            ConfigOptionDef::new("logging.filter", "Log filter directives")
                .with_env("HYPRSTREAM_LOG_FILTER")
                .with_cli("log-filter"),
        ]
    }

    fn from_config<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LoggingConfigFile {
            level: Option<String>,
            filter: Option<String>,
        }

        let config = LoggingConfigFile::deserialize(deserializer)?;
        Ok(LoggingConfig {
            verbose: 0,
            log_level: config.level,
            log_filter: config.filter,
        })
    }
}

/// Service management configuration
#[derive(Debug, Clone, Default, Args, Deserialize)]
pub struct ServicesConfigCli {
    /// Comma-separated list of services to start at startup
    ///
    /// Example: --services-startup registry,policy,worker,event
    /// Environment: HYPRSTREAM_SERVICES_STARTUP=registry,policy,worker,event
    #[arg(long = "services-startup", env = "HYPRSTREAM_SERVICES_STARTUP", value_delimiter = ',')]
    #[serde(skip)]
    pub startup: Option<Vec<String>>,
}

impl ConfigSection for ServicesConfigCli {
    fn options() -> Vec<ConfigOptionDef<String>> {
        vec![
            ConfigOptionDef::new(
                "services.startup",
                "Comma-separated list of services to start at startup (registry, policy, worker, event)",
            )
            .with_env("HYPRSTREAM_SERVICES_STARTUP")
            .with_cli("services-startup"),
        ]
    }

    fn from_config<'de, D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ServicesConfigFile {
            startup: Option<Vec<String>>,
        }

        let config = ServicesConfigFile::deserialize(deserializer)?;
        Ok(ServicesConfigCli {
            startup: config.startup,
        })
    }
}

/// Credentials for authentication
#[derive(Debug, Clone, Deserialize)]
pub struct Credentials {
    /// Username for authentication
    pub username: String,
    /// Password for authentication
    pub password: String,
}
