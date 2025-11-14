//! Server configuration with builder pattern
//!
//! This module provides a unified server configuration that combines:
//! - Network settings (host, port)
//! - Runtime settings (caching, preloading, logging, metrics)
//! - CORS configuration
//! - TLS settings
//! - Process management (working directory, PID file)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Enable CORS middleware
    #[serde(default = "default_cors_enabled")]
    pub enabled: bool,

    /// Allowed origins (use ["*"] for all origins - NOT recommended for production)
    #[serde(default = "default_cors_origins")]
    pub allowed_origins: Vec<String>,

    /// Allow credentials in CORS requests
    #[serde(default = "default_cors_credentials")]
    pub allow_credentials: bool,

    /// Max age for preflight cache (in seconds)
    #[serde(default = "default_cors_max_age")]
    pub max_age: u64,

    /// Allow all headers (permissive mode for development - NOT recommended for production)
    #[serde(default)]
    pub permissive_headers: bool,
}

fn default_cors_enabled() -> bool {
    true
}
fn default_cors_origins() -> Vec<String> {
    vec![
        "http://localhost:3000".to_string(),
        "http://localhost:3001".to_string(),
        "http://127.0.0.1:3000".to_string(),
        "http://127.0.0.1:3001".to_string(),
    ]
}
fn default_cors_credentials() -> bool {
    true
}
fn default_cors_max_age() -> u64 {
    3600
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: default_cors_enabled(),
            allowed_origins: default_cors_origins(),
            allow_credentials: default_cors_credentials(),
            max_age: default_cors_max_age(),
            permissive_headers: false,
        }
    }
}

impl CorsConfig {
    /// Create CORS config with dynamic port-based origins
    pub fn with_port(port: u16) -> Self {
        let mut config = Self::default();
        config.allowed_origins.extend(vec![
            format!("http://localhost:{}", port),
            format!("http://127.0.0.1:{}", port),
        ]);
        config
    }
}

/// Default sampling parameters for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParamDefaults {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_top_p")]
    pub top_p: f32,

    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,

    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_repeat_penalty() -> f32 {
    1.1
}
fn default_stream_timeout_secs() -> u64 {
    300
}

impl Default for SamplingParamDefaults {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            repeat_penalty: default_repeat_penalty(),
            stream_timeout_secs: default_stream_timeout_secs(),
        }
    }
}

/// Unified server configuration combining network, runtime, and TLS settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    // Network settings
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    // Runtime settings
    #[serde(default = "default_max_cached_models")]
    pub max_cached_models: usize,

    #[serde(default)]
    pub preload_models: Vec<String>,

    #[serde(default = "default_true")]
    pub enable_logging: bool,

    #[serde(default = "default_true")]
    pub enable_metrics: bool,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default = "default_max_tokens_limit")]
    pub max_tokens_limit: usize,

    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    // Concurrency settings
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    #[serde(default = "default_generation_timeout_secs")]
    pub generation_timeout_secs: u64,

    #[serde(default = "default_cancellation_check_interval")]
    pub cancellation_check_interval: u64,

    // CORS configuration
    #[serde(default)]
    pub cors: CorsConfig,

    // Default sampling parameters
    #[serde(default)]
    pub sampling_defaults: SamplingParamDefaults,

    // TLS settings
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    #[serde(default)]
    pub tls_key: Option<PathBuf>,

    #[serde(default)]
    pub tls_client_ca: Option<PathBuf>,

    #[serde(default = "default_tls_min_version")]
    pub tls_min_version: String,

    #[serde(default)]
    pub tls_cipher_list: Option<String>,

    #[serde(default = "default_true")]
    pub tls_prefer_server_ciphers: bool,

    // Process management
    #[serde(default)]
    pub working_dir: Option<PathBuf>,

    #[serde(default)]
    pub pid_file: Option<PathBuf>,
}

// Default value functions for serde
fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    50051
}
fn default_max_cached_models() -> usize {
    5
}
fn default_true() -> bool {
    true
}
fn default_max_tokens_limit() -> usize {
    4096
}
fn default_request_timeout_secs() -> u64 {
    300
}
fn default_max_concurrent_requests() -> usize {
    100
}
fn default_generation_timeout_secs() -> u64 {
    600
}
fn default_cancellation_check_interval() -> u64 {
    100
}
fn default_tls_min_version() -> String {
    "1.2".to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            max_cached_models: default_max_cached_models(),
            preload_models: Vec::new(),
            enable_logging: true,
            enable_metrics: true,
            api_key: None,
            max_tokens_limit: default_max_tokens_limit(),
            request_timeout_secs: default_request_timeout_secs(),
            max_concurrent_requests: default_max_concurrent_requests(),
            generation_timeout_secs: default_generation_timeout_secs(),
            cancellation_check_interval: default_cancellation_check_interval(),
            cors: CorsConfig::default(),
            sampling_defaults: SamplingParamDefaults::default(),
            tls_cert: None,
            tls_key: None,
            tls_client_ca: None,
            tls_min_version: default_tls_min_version(),
            tls_cipher_list: None,
            tls_prefer_server_ciphers: true,
            working_dir: None,
            pid_file: None,
        }
    }
}

/// Builder for ServerConfig with chainable methods
#[derive(Debug, Default)]
pub struct ServerConfigBuilder {
    config: ServerConfig,
}

impl ServerConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: ServerConfig::default(),
        }
    }

    /// Start from an existing config
    pub fn from_config(config: ServerConfig) -> Self {
        Self { config }
    }

    // Network settings
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    // Runtime settings
    pub fn max_cached_models(mut self, max: usize) -> Self {
        self.config.max_cached_models = max;
        self
    }

    pub fn preload_models(mut self, models: Vec<String>) -> Self {
        self.config.preload_models = models;
        self
    }

    pub fn add_preload_model(mut self, model: impl Into<String>) -> Self {
        self.config.preload_models.push(model.into());
        self
    }

    pub fn enable_logging(mut self, enabled: bool) -> Self {
        self.config.enable_logging = enabled;
        self
    }

    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    pub fn max_tokens_limit(mut self, limit: usize) -> Self {
        self.config.max_tokens_limit = limit;
        self
    }

    pub fn request_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.request_timeout_secs = timeout;
        self
    }

    // Concurrency settings
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    pub fn generation_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.generation_timeout_secs = timeout;
        self
    }

    pub fn generation_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.generation_timeout_secs = timeout.as_secs();
        self
    }

    pub fn cancellation_check_interval(mut self, interval: u64) -> Self {
        self.config.cancellation_check_interval = interval;
        self
    }

    pub fn cancellation_check_interval_duration(mut self, interval: std::time::Duration) -> Self {
        self.config.cancellation_check_interval = interval.as_millis() as u64;
        self
    }

    // CORS settings
    pub fn cors(mut self, cors: CorsConfig) -> Self {
        self.config.cors = cors;
        self
    }

    pub fn cors_enabled(mut self, enabled: bool) -> Self {
        self.config.cors.enabled = enabled;
        self
    }

    pub fn cors_origins(mut self, origins: Vec<String>) -> Self {
        self.config.cors.allowed_origins = origins;
        self
    }

    pub fn cors_allow_credentials(mut self, allow: bool) -> Self {
        self.config.cors.allow_credentials = allow;
        self
    }

    pub fn cors_permissive_headers(mut self, permissive: bool) -> Self {
        self.config.cors.permissive_headers = permissive;
        self
    }

    // Sampling defaults
    pub fn sampling_defaults(mut self, defaults: SamplingParamDefaults) -> Self {
        self.config.sampling_defaults = defaults;
        self
    }

    // TLS settings
    pub fn tls_cert(mut self, cert: PathBuf) -> Self {
        self.config.tls_cert = Some(cert);
        self
    }

    pub fn tls_key(mut self, key: PathBuf) -> Self {
        self.config.tls_key = Some(key);
        self
    }

    pub fn tls_client_ca(mut self, ca: PathBuf) -> Self {
        self.config.tls_client_ca = Some(ca);
        self
    }

    pub fn tls_min_version(mut self, version: impl Into<String>) -> Self {
        self.config.tls_min_version = version.into();
        self
    }

    pub fn tls_cipher_list(mut self, ciphers: impl Into<String>) -> Self {
        self.config.tls_cipher_list = Some(ciphers.into());
        self
    }

    pub fn tls_prefer_server_ciphers(mut self, prefer: bool) -> Self {
        self.config.tls_prefer_server_ciphers = prefer;
        self
    }

    // Process management
    pub fn working_dir(mut self, dir: PathBuf) -> Self {
        self.config.working_dir = Some(dir);
        self
    }

    pub fn pid_file(mut self, file: PathBuf) -> Self {
        self.config.pid_file = Some(file);
        self
    }

    /// Load values from environment variables (merges with current config)
    pub fn from_env(mut self) -> Self {
        // Network
        if let Ok(host) = std::env::var("HYPRSTREAM_SERVER_HOST") {
            self.config.host = host;
        }
        if let Ok(port) = std::env::var("HYPRSTREAM_SERVER_PORT") {
            if let Ok(p) = port.parse() {
                self.config.port = p;
            }
        }

        // Concurrency settings
        if let Ok(max_concurrent) = std::env::var("HYPRSTREAM_MAX_CONCURRENT_REQUESTS") {
            if let Ok(n) = max_concurrent.parse() {
                self.config.max_concurrent_requests = n;
            }
        }

        if let Ok(timeout) = std::env::var("HYPRSTREAM_GENERATION_TIMEOUT_SECS") {
            if let Ok(t) = timeout.parse() {
                self.config.generation_timeout_secs = t;
            }
        }

        if let Ok(interval) = std::env::var("HYPRSTREAM_CANCELLATION_CHECK_INTERVAL") {
            if let Ok(i) = interval.parse() {
                self.config.cancellation_check_interval = i;
            }
        }

        // Runtime settings
        if let Ok(max_cached) = std::env::var("HYPRSTREAM_MAX_CACHED_MODELS") {
            if let Ok(n) = max_cached.parse() {
                self.config.max_cached_models = n;
            }
        }

        if let Ok(models) = std::env::var("HYPRSTREAM_PRELOAD_MODELS") {
            self.config.preload_models = models
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        if let Ok(api_key) = std::env::var("HYPRSTREAM_API_KEY") {
            self.config.api_key = Some(api_key);
        }

        // CORS
        if let Ok(cors_enabled) = std::env::var("HYPRSTREAM_CORS_ENABLED") {
            self.config.cors.enabled = cors_enabled.to_lowercase() != "false";
        }

        if let Ok(cors_origins) = std::env::var("HYPRSTREAM_CORS_ORIGINS") {
            self.config.cors.allowed_origins = cors_origins
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if self.config.cors.allowed_origins.contains(&"*".to_string()) {
                self.config.cors.allow_credentials = false;
            }
        }

        if let Ok(cors_credentials) = std::env::var("HYPRSTREAM_CORS_CREDENTIALS") {
            if !self.config.cors.allowed_origins.contains(&"*".to_string()) {
                self.config.cors.allow_credentials = cors_credentials.to_lowercase() == "true";
            }
        }

        if let Ok(permissive) = std::env::var("HYPRSTREAM_CORS_PERMISSIVE_HEADERS") {
            self.config.cors.permissive_headers = permissive.to_lowercase() == "true";
        }

        // TLS
        if let Ok(cert) = std::env::var("HYPRSTREAM_TLS_CERT") {
            self.config.tls_cert = Some(PathBuf::from(cert));
        }
        if let Ok(key) = std::env::var("HYPRSTREAM_TLS_KEY") {
            self.config.tls_key = Some(PathBuf::from(key));
        }
        if let Ok(ca) = std::env::var("HYPRSTREAM_TLS_CLIENT_CA") {
            self.config.tls_client_ca = Some(PathBuf::from(ca));
        }
        if let Ok(version) = std::env::var("HYPRSTREAM_TLS_MIN_VERSION") {
            self.config.tls_min_version = version;
        }

        // Process management
        if let Ok(dir) = std::env::var("HYPRSTREAM_WORKING_DIR") {
            self.config.working_dir = Some(PathBuf::from(dir));
        }
        if let Ok(file) = std::env::var("HYPRSTREAM_PID_FILE") {
            self.config.pid_file = Some(PathBuf::from(file));
        }

        self
    }

    /// Finalize CORS configuration based on the server port
    pub fn finalize_cors(mut self) -> Self {
        // Update CORS to use port-aware defaults if still using defaults
        if self.config.cors.allowed_origins == default_cors_origins() {
            self.config.cors = CorsConfig::with_port(self.config.port);
        }
        self
    }

    /// Build the final configuration
    pub fn build(self) -> ServerConfig {
        self.config
    }
}

impl ServerConfig {
    /// Create a builder for this config
    pub fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder::new()
    }

    /// Create a builder from existing config
    pub fn to_builder(self) -> ServerConfigBuilder {
        ServerConfigBuilder::from_config(self)
    }
}
