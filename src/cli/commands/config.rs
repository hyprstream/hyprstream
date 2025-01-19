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

/// Credentials for authentication
#[derive(Debug, Clone, Deserialize)]
pub struct Credentials {
    /// Username for authentication
    pub username: String,
    /// Password for authentication
    pub password: String,
}
