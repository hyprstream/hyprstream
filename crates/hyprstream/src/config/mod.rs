//! Unified configuration system for Hyprstream
//!
//! This module provides a layered configuration architecture:
//! - `HyprConfig`: Root configuration combining all subsystems
//! - `ServerConfig`: HTTP server configuration (network, CORS, TLS)
//! - Model and runtime configs for ML inference

pub mod server;

// Re-export main configuration types
pub use server::{CorsConfig, SamplingParamDefaults, ServerConfig, ServerConfigBuilder};

// Export root configuration and builder (defined below in this module)
// Note: HyprConfig and HyprConfigBuilder are exported automatically as pub structs

use crate::runtime::generation_metrics::GenerationQualityMetrics;
use crate::storage::paths::StoragePaths;
use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Unified configuration for the Hyprstream system
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct HyprConfig {
    /// HTTP server configuration
    #[serde(default)]
    pub server: ServerConfig,

    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Runtime execution settings
    #[serde(default)]
    pub runtime: RuntimeConfig,

    /// Text generation parameters
    #[serde(default)]
    pub generation: GenerationConfig,

    /// LoRA adapter settings
    #[serde(default)]
    pub lora: LoRAConfig,

    /// Storage paths configuration
    #[serde(default)]
    pub storage: StorageConfig,

    /// Git storage and P2P transport configuration
    #[serde(default)]
    pub git2db: git2db::config::Git2DBConfig,

    /// Worker service configuration (optional)
    ///
    /// When present, the WorkerService will be started for container/sandbox management.
    /// This enables Kata-based isolated workload execution.
    #[serde(default)]
    pub worker: Option<hyprstream_workers::config::WorkerConfig>,

    /// Service management configuration
    ///
    /// Controls which services are started at startup in ipc-systemd mode.
    #[serde(default)]
    pub services: ServicesConfig,

    /// JWT token configuration
    #[serde(default)]
    pub token: TokenConfig,

    /// OpenAI-compatible HTTP API configuration
    #[serde(default)]
    pub oai: OAIConfig,

    /// Arrow Flight SQL server configuration
    #[serde(default)]
    pub flight: FlightConfig,

    /// StreamService configuration (buffer sizes, TTL, etc.)
    #[serde(default)]
    pub streaming: StreamingConfig,
}

/// JWT token issuance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    #[serde(default = "default_token_ttl")]
    pub default_ttl_seconds: u32,

    #[serde(default = "default_max_token_ttl")]
    pub max_ttl_seconds: u32,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            default_ttl_seconds: 300,   // 5 minutes
            max_ttl_seconds: 3600,      // 1 hour
        }
    }
}

fn default_token_ttl() -> u32 { 300 }
fn default_max_token_ttl() -> u32 { 3600 }

/// OpenAI-compatible HTTP API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAIConfig {
    /// Host address for HTTP server
    #[serde(default = "default_oai_host")]
    pub host: String,

    /// Port for HTTP server
    #[serde(default = "default_oai_port")]
    pub port: u16,

    /// TLS certificate path (optional)
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional)
    #[serde(default)]
    pub tls_key: Option<PathBuf>,

    /// Request timeout in seconds
    #[serde(default = "default_oai_timeout")]
    pub request_timeout_secs: u64,

    /// CORS configuration
    #[serde(default)]
    pub cors: server::CorsConfig,
}

impl Default for OAIConfig {
    fn default() -> Self {
        Self {
            host: default_oai_host(),
            port: default_oai_port(),
            tls_cert: None,
            tls_key: None,
            request_timeout_secs: default_oai_timeout(),
            cors: server::CorsConfig::default(),
        }
    }
}

fn default_oai_host() -> String { "0.0.0.0".to_string() }
fn default_oai_port() -> u16 { 8080 }
fn default_oai_timeout() -> u64 { 300 }

/// Arrow Flight SQL server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightConfig {
    /// Host address for Flight SQL server
    #[serde(default = "default_flight_host")]
    pub host: String,

    /// Port for Flight SQL server
    #[serde(default = "default_flight_port")]
    pub port: u16,

    /// Default dataset to serve (optional)
    #[serde(default)]
    pub default_dataset: Option<String>,

    /// TLS certificate path (optional)
    #[serde(default)]
    pub tls_cert: Option<PathBuf>,

    /// TLS private key path (optional)
    #[serde(default)]
    pub tls_key: Option<PathBuf>,
}

impl Default for FlightConfig {
    fn default() -> Self {
        Self {
            host: default_flight_host(),
            port: default_flight_port(),
            default_dataset: None,
            tls_cert: None,
            tls_key: None,
        }
    }
}

fn default_flight_host() -> String { "0.0.0.0".to_string() }
fn default_flight_port() -> u16 { 50051 }

/// StreamService configuration
///
/// Controls the PULL/XPUB streaming proxy behavior including buffer sizes,
/// message TTL, retransmission settings, and StreamBlock batching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Maximum pending messages per topic (for pre-subscribe queue and retransmit buffer)
    /// Default: 1000
    #[serde(default = "default_max_pending_per_topic")]
    pub max_pending_per_topic: usize,

    /// Message TTL in seconds - messages older than this are dropped
    /// Default: 30
    #[serde(default = "default_message_ttl_secs")]
    pub message_ttl_secs: u64,

    /// Interval between compaction runs in seconds
    /// Default: 5
    #[serde(default = "default_compact_interval_secs")]
    pub compact_interval_secs: u64,

    /// StreamBlock batching configuration (rate control)
    ///
    /// Controls adaptive batching based on throughput rate.
    /// Higher rates → larger batches (reduced overhead).
    /// Lower rates → smaller batches (reduced latency).
    #[serde(flatten, default)]
    pub batching: hyprstream_rpc::streaming::BatchingConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_pending_per_topic: default_max_pending_per_topic(),
            message_ttl_secs: default_message_ttl_secs(),
            compact_interval_secs: default_compact_interval_secs(),
            batching: hyprstream_rpc::streaming::BatchingConfig::default(),
        }
    }
}

fn default_max_pending_per_topic() -> usize { 1000 }
fn default_message_ttl_secs() -> u64 { 30 }
fn default_compact_interval_secs() -> u64 { 5 }

/// Storage paths and directories configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Models directory path
    pub models_dir: PathBuf,
    /// LoRAs directory path
    pub loras_dir: PathBuf,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Config directory path
    pub config_dir: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        // Try to get XDG-compliant paths, fall back to current directory
        let (models_dir, loras_dir, cache_dir, config_dir) = match StoragePaths::new() {
            Ok(storage_paths) => (
                storage_paths.models_dir().unwrap_or_else(|_| PathBuf::from("./models")),
                storage_paths.loras_dir().unwrap_or_else(|_| PathBuf::from("./loras")),
                storage_paths.cache_dir().unwrap_or_else(|_| PathBuf::from("./cache")),
                storage_paths.config_dir().unwrap_or_else(|_| PathBuf::from("./config")),
            ),
            Err(e) => {
                tracing::warn!("XDG paths unavailable: {}, using local directories", e);
                (
                    PathBuf::from("./models"),
                    PathBuf::from("./loras"),
                    PathBuf::from("./cache"),
                    PathBuf::from("./config"),
                )
            }
        };

        Self {
            models_dir,
            loras_dir,
            cache_dir,
            config_dir,
        }
    }
}

/// Service management configuration
///
/// Controls which services are started at startup in ipc-systemd mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicesConfig {
    /// Services to start automatically at startup (ipc-systemd mode)
    ///
    /// Default: ["registry", "policy", "worker", "event"]
    #[serde(default = "default_startup_services")]
    pub startup: Vec<String>,
}

impl Default for ServicesConfig {
    fn default() -> Self {
        Self {
            startup: default_startup_services(),
        }
    }
}

/// Default list of services to start at startup
fn default_startup_services() -> Vec<String> {
    vec![
        "event".to_string(),     // Must start first (message bus)
        "registry".to_string(),  // Model registry
        "policy".to_string(),    // Authorization
        "streams".to_string(),   // Streaming proxy with JWT validation
        "worker".to_string(),    // Container workloads
        "model".to_string(),     // Model management
        "oai".to_string(),       // OpenAI-compatible HTTP API
        "flight".to_string(),    // Arrow Flight SQL server
    ]
}

/// Model loading and identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model file
    pub path: PathBuf,
    /// Model identifier (e.g., "qwen2-1.5b")
    pub name: String,
    /// Architecture type ("llama", "qwen", etc.)
    pub architecture: String,
    /// Expected parameter count
    pub parameters: Option<u64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            name: String::new(),
            architecture: String::new(),
            parameters: None,
        }
    }
}

/// Runtime execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Context window size
    pub context_length: usize,
    /// Maximum context length override for KV cache allocation.
    /// None = use model's max_position_embeddings (can be very large, e.g., 40K tokens)
    /// Some(n) = cap KV cache at n tokens (significantly reduces GPU memory)
    pub max_context: Option<usize>,
    /// KV cache quantization type (None, INT8, NF4, FP4).
    /// Reduces GPU memory by 50-75% at slight quality cost.
    #[serde(default)]
    pub kv_quant_type: crate::runtime::kv_quant::KVQuantType,
    /// Batch processing size
    pub batch_size: usize,
    /// CPU threads (None = auto-detect)
    pub cpu_threads: Option<usize>,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// GPU device ID (None = auto-detect, typically device 0)
    pub gpu_device_id: Option<usize>,
    /// GPU layers to offload (None = auto)
    pub gpu_layers: Option<usize>,
    /// Use memory mapping for model files
    pub mmap: bool,
    /// KV cache size in MB
    pub kv_cache_size_mb: usize,
    /// Precision mode (BF16/FP16/FP32/FP8)
    pub precision_mode: Option<String>,
    // NEW: Concurrency and timeout settings
    pub max_concurrent_loads: usize,
    pub max_concurrent_generations: usize,
    pub default_generation_timeout_ms: u64,
    pub default_model_load_timeout_ms: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        // Check environment variables for runtime configuration
        // Precedence: CLI args > env vars (read here) > hardcoded defaults.
        // Environment variables set initial defaults; CLI args may override them later.
        let gpu_device_id = std::env::var("HYPRSTREAM_GPU_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let max_context = std::env::var("HYPRSTREAM_MAX_CONTEXT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());

        let kv_quant_type = std::env::var("HYPRSTREAM_KV_QUANT")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "int8" => Some(crate::runtime::kv_quant::KVQuantType::Int8),
                "nf4" => Some(crate::runtime::kv_quant::KVQuantType::Nf4),
                "fp4" => Some(crate::runtime::kv_quant::KVQuantType::Fp4),
                "none" | "" => Some(crate::runtime::kv_quant::KVQuantType::None),
                _ => None,
            })
            .unwrap_or(crate::runtime::kv_quant::KVQuantType::None);

        Self {
            context_length: 4096,
            max_context,
            kv_quant_type,
            batch_size: 512,
            cpu_threads: None,
            use_gpu: true,
            gpu_device_id, // From env or None (auto-detect device 0)
            gpu_layers: None,
            mmap: true,
            kv_cache_size_mb: 2048,
            precision_mode: Some("auto".to_string()),
            max_concurrent_loads: 2,
            max_concurrent_generations: 10,
            default_generation_timeout_ms: 120000, // 2 minutes
            default_model_load_timeout_ms: 300000, // 5 minutes
        }
    }
}

/// Text generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0-2.0)
    pub temperature: f32,
    /// Nucleus sampling threshold
    pub top_p: f32,
    /// Top-k sampling limit
    pub top_k: Option<usize>,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Stop sequences
    pub stop_tokens: Vec<String>,
    /// Random seed for reproducible generation
    pub seed: Option<u32>,
    /// Enable streaming output
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
            stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
            seed: None,
            stream: false,
        }
    }
}

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Enable LoRA adapters
    pub enabled: bool,
    /// Maximum number of active adapters
    pub max_adapters: usize,
    /// LoRA scaling factor (alpha)
    pub alpha: f32,
    /// Target sparsity ratio (0.0-1.0)
    pub sparsity: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_adapters: 4,
            alpha: 32.0,
            sparsity: 0.99,
        }
    }
}


/// Builder for Hyprstream configuration
pub struct HyprConfigBuilder {
    server_builder: ServerConfigBuilder,
    model: ModelConfig,
    runtime: RuntimeConfig,
    generation: GenerationConfig,
    lora: LoRAConfig,
    storage: StorageConfig,
    git2db: git2db::config::Git2DBConfig,
    worker: Option<hyprstream_workers::config::WorkerConfig>,
    services: ServicesConfig,
    token: TokenConfig,
    oai: OAIConfig,
    flight: FlightConfig,
    streaming: StreamingConfig,
}

impl HyprConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            server_builder: ServerConfigBuilder::new(),
            model: ModelConfig::default(),
            runtime: RuntimeConfig::default(),
            generation: GenerationConfig::default(),
            lora: LoRAConfig::default(),
            storage: StorageConfig::default(),
            git2db: git2db::config::Git2DBConfig::default(),
            worker: None,
            services: ServicesConfig::default(),
            token: TokenConfig::default(),
            oai: OAIConfig::default(),
            flight: FlightConfig::default(),
            streaming: StreamingConfig::default(),
        }
    }

    /// Start from an existing config
    pub fn from_config(config: HyprConfig) -> Self {
        Self {
            server_builder: config.server.to_builder(),
            model: config.model,
            runtime: config.runtime,
            generation: config.generation,
            lora: config.lora,
            storage: config.storage,
            git2db: config.git2db,
            worker: config.worker,
            services: config.services,
            token: config.token,
            oai: config.oai,
            flight: config.flight,
            streaming: config.streaming,
        }
    }

    /// Access server builder for chaining
    pub fn server(mut self, f: impl FnOnce(ServerConfigBuilder) -> ServerConfigBuilder) -> Self {
        self.server_builder = f(self.server_builder);
        self
    }

    /// Load all configurations from environment variables
    pub fn from_env(mut self) -> Self {
        self.server_builder = self.server_builder.from_env();
        self
    }

    /// Build the final configuration
    pub fn build(self) -> HyprConfig {
        HyprConfig {
            server: self.server_builder.build(),
            model: self.model,
            runtime: self.runtime,
            generation: self.generation,
            lora: self.lora,
            storage: self.storage,
            git2db: self.git2db,
            worker: self.worker,
            services: self.services,
            token: self.token,
            oai: self.oai,
            flight: self.flight,
            streaming: self.streaming,
        }
    }

    /// Set the worker service configuration
    pub fn worker(mut self, config: hyprstream_workers::config::WorkerConfig) -> Self {
        self.worker = Some(config);
        self
    }
}

impl Default for HyprConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HyprConfig {
    /// Create a builder for the application configuration
    pub fn builder() -> HyprConfigBuilder {
        HyprConfigBuilder::new()
    }

    /// Load configuration using the config crate with XDG directories and environment variables
    pub fn load() -> Result<Self, ConfigError> {
        let storage = StoragePaths::new().map_err(|e| {
            ConfigError::Message(format!("Failed to initialize storage paths: {}", e))
        })?;

        let config_dir = storage
            .config_dir()
            .map_err(|e| ConfigError::Message(format!("Failed to get config directory: {}", e)))?;

        let settings = Config::builder()
            // Load from default configuration structure
            .add_source(Config::try_from(&HyprConfig::default())?)
            // Load from config file if it exists
            .add_source(File::from(config_dir.join("config")).required(false))
            .add_source(File::from(config_dir.join("config.toml")).required(false))
            .add_source(File::from(config_dir.join("config.json")).required(false))
            .add_source(File::from(config_dir.join("config.yaml")).required(false))
            // Load from environment variables with HYPRSTREAM__ prefix (double underscore for nesting)
            .add_source(Environment::with_prefix("HYPRSTREAM").separator("__").try_parsing(true));

        // Build and deserialize configuration
        let mut hypr_config: HyprConfig = settings.build()?.try_deserialize()?;

        // Load git2db config from environment/file (it has its own env handling)
        // This ensures GIT2DB__* environment variables are properly loaded
        match git2db::config::Git2DBConfig::load() {
            Ok(git2db_config) => {
                tracing::info!(
                    "Loaded git2db config, token present: {}",
                    git2db_config.network.access_token.is_some()
                );
                hypr_config.git2db = git2db_config;
            }
            Err(e) => {
                tracing::warn!("Failed to load git2db config: {}, using default", e);
                hypr_config.git2db = git2db::config::Git2DBConfig::default();
            }
        }

        Ok(hypr_config)
    }

    /// Load configuration from file
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("toml");

        let config = match extension {
            "json" => serde_json::from_str(&contents)?,
            "yaml" | "yml" => serde_yaml::from_str(&contents)?,
            _ => toml::from_str(&contents)?,
        };

        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: &Path) -> anyhow::Result<()> {
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("toml");

        let contents = match extension {
            "json" => serde_json::to_string_pretty(self)?,
            "yaml" | "yml" => serde_yaml::to_string(self)?,
            _ => toml::to_string_pretty(self)?,
        };

        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Validate the entire configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        // Validate model config
        if !self.model.path.as_os_str().is_empty() && !self.model.path.exists() {
            anyhow::bail!(
                "Configured model path does not exist: {}",
                self.model.path.display()
            );
        }

        Ok(())
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &PathBuf {
        &self.storage.models_dir
    }

    /// Get the LoRAs directory path
    pub fn loras_dir(&self) -> &PathBuf {
        &self.storage.loras_dir
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &PathBuf {
        &self.storage.cache_dir
    }

    /// Get the config directory path
    pub fn config_dir(&self) -> &PathBuf {
        &self.storage.config_dir
    }

    /// Ensure all configured directories exist
    pub fn ensure_directories(&self) -> Result<(), std::io::Error> {
        std::fs::create_dir_all(&self.storage.models_dir)?;
        std::fs::create_dir_all(&self.storage.loras_dir)?;
        std::fs::create_dir_all(&self.storage.cache_dir)?;
        std::fs::create_dir_all(&self.storage.config_dir)?;
        Ok(())
    }

    /// Update model configuration after downloading
    pub fn set_model(&mut self, model_path: PathBuf, model_name: String, architecture: String) {
        self.model.path = model_path;
        self.model.name = model_name;
        self.model.architecture = architecture;
    }

    /// Create generation request from config + prompt
    ///
    /// Note: The prompt should already be templated (via apply_chat_template).
    pub fn create_request(&self, prompt: TemplatedPrompt) -> GenerationRequest {
        let mut request = GenerationRequest::from(&self.generation);
        request.prompt = prompt;
        request
    }

    /// Save configuration to default location
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let storage = StoragePaths::new()?;
        let config_dir = storage.config_dir()?;
        let config_path = config_dir.join("config.toml");

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;

        tracing::info!("✅ Configuration saved to: {}", config_path.display());
        Ok(())
    }

    /// Create a default configuration for a specific model path
    pub fn default_for_model(model_path: &Path) -> anyhow::Result<Self> {
        let storage_paths = StoragePaths::new()?;
        let mut config = Self::default();

        config.model.path = model_path.to_path_buf();
        config.model.name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        config.model.architecture = "auto".to_string(); // Auto-detect from model

        // Update storage paths to use XDG directories
        config.storage = StorageConfig {
            models_dir: storage_paths.models_dir()?,
            loras_dir: storage_paths.loras_dir()?,
            cache_dir: storage_paths.cache_dir()?,
            config_dir: storage_paths.config_dir()?,
        };

        Ok(config)
    }
}

/// Model information returned after loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: u64,
    pub context_length: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub architecture: String,
    pub quantization: Option<String>,
}

/// A prompt string that has been processed through the chat template engine.
/// This newtype prevents accidentally passing untemplated strings to generation.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(transparent)]
pub struct TemplatedPrompt(String);

impl TemplatedPrompt {
    /// Create from a templated string. Only call after template application.
    pub fn new(s: String) -> Self {
        Self(s)
    }

    /// Get the prompt as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the inner string.
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Get the length of the prompt in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the prompt is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for TemplatedPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Generation request with all parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationRequest {
    pub prompt: TemplatedPrompt,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    #[serde(default)]
    pub repeat_last_n: usize,
    pub stop_tokens: Vec<String>,
    pub seed: Option<u32>,
    /// Optional image paths for multimodal models
    #[serde(default)]
    pub images: Vec<String>,
    // Async configuration fields
    #[serde(default)]
    pub timeout: Option<u64>, // Duration in milliseconds
    /// Enable quality metrics collection for self-supervised training.
    /// Default: false (disabled for performance - metrics add ~10x overhead)
    #[serde(default)]
    pub collect_metrics: bool,
}

/// Unified sampling parameters with Option fields for clean precedence merging.
///
/// All fields are Option<T> to represent "not specified", enabling clear
/// precedence: Server defaults → Model defaults → User overrides
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub stop_tokens: Option<Vec<String>>,
    pub seed: Option<u64>,

    // Advanced parameters (HuggingFace transformers compatibility)
    #[serde(default)]
    pub length_penalty: Option<f32>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub epsilon_cutoff: Option<f32>,
    #[serde(default)]
    pub eta_cutoff: Option<f32>,
    #[serde(default)]
    pub do_sample: Option<bool>,

    // NEW: Async parameters
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

impl SamplingParams {
    /// Load model-specific config from a model directory
    pub async fn from_model_path(model_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let gen_config_path = model_path.join("generation_config.json");
        if gen_config_path.exists() {
            let content = tokio::fs::read_to_string(&gen_config_path).await?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            return Ok(Self::from_generation_config(&config));
        }

        let config_path = model_path.join("config.json");
        if config_path.exists() {
            let content = tokio::fs::read_to_string(&config_path).await?;
            let config: serde_json::Value = serde_json::from_str(&content)?;
            if let Some(gen_config) = config.get("generation_config") {
                return Ok(Self::from_generation_config(gen_config));
            }
        }

        Ok(Self::default())
    }

    /// Parse HuggingFace generation_config.json format
    fn from_generation_config(config: &serde_json::Value) -> Self {
        Self {
            temperature: config.get("temperature").and_then(|v| v.as_f64()).map(|v| v as f32),
            top_k: config.get("top_k").and_then(|v| v.as_u64()).map(|v| v as usize),
            top_p: config.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
            repeat_penalty: config.get("repetition_penalty").and_then(|v| v.as_f64()).map(|v| v as f32),
            max_tokens: config.get("max_new_tokens").and_then(|v| v.as_u64()).map(|v| v as usize)
                .or_else(|| config.get("max_length").and_then(|v| v.as_u64()).map(|v| v as usize)),
            length_penalty: config.get("length_penalty").and_then(|v| v.as_f64()).map(|v| v as f32),
            typical_p: config.get("typical_p").and_then(|v| v.as_f64()).map(|v| v as f32),
            epsilon_cutoff: config.get("epsilon_cutoff").and_then(|v| v.as_f64()).map(|v| v as f32),
            eta_cutoff: config.get("eta_cutoff").and_then(|v| v.as_f64()).map(|v| v as f32),
            do_sample: config.get("do_sample").and_then(|v| v.as_bool()),
            stop_tokens: config.get("eos_token_id").and_then(|v| {
                if let Some(arr) = v.as_array() {
                    let tokens: Vec<String> = arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    if tokens.is_empty() { None } else { Some(tokens) }
                } else {
                    None
                }
            }),
            seed: config.get("seed").and_then(|v| v.as_u64()),
            repeat_last_n: None,
            timeout_ms: None,
        }
    }

    /// Merge with another config. The other config takes precedence for any Some values.
    /// This enables clear precedence: `base.merge(override)` where override wins.
    pub fn merge(self, other: Self) -> Self {
        Self {
            max_tokens: other.max_tokens.or(self.max_tokens),
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            repeat_penalty: other.repeat_penalty.or(self.repeat_penalty),
            repeat_last_n: other.repeat_last_n.or(self.repeat_last_n),
            stop_tokens: other.stop_tokens.or(self.stop_tokens),
            seed: other.seed.or(self.seed),
            length_penalty: other.length_penalty.or(self.length_penalty),
            typical_p: other.typical_p.or(self.typical_p),
            epsilon_cutoff: other.epsilon_cutoff.or(self.epsilon_cutoff),
            eta_cutoff: other.eta_cutoff.or(self.eta_cutoff),
            do_sample: other.do_sample.or(self.do_sample),
            timeout_ms: other.timeout_ms.or(self.timeout_ms),
        }
    }

    /// Resolve to concrete values with defaults
    pub fn resolve(self) -> ResolvedSamplingParams {
        ResolvedSamplingParams {
            max_tokens: self.max_tokens.unwrap_or(2048),
            temperature: self.temperature.unwrap_or(0.7),
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty.unwrap_or(1.0),
            repeat_last_n: self.repeat_last_n.unwrap_or(64),
            stop_tokens: self.stop_tokens.unwrap_or_default(),
            seed: self.seed,
            length_penalty: self.length_penalty.unwrap_or(1.0),
            typical_p: self.typical_p,
            epsilon_cutoff: self.epsilon_cutoff,
            eta_cutoff: self.eta_cutoff,
            do_sample: self.do_sample.unwrap_or(true),
            timeout_ms: self.timeout_ms.unwrap_or(120000), // Use RuntimeConfig default (2 minutes)
        }
    }
}

/// Resolved sampling parameters with concrete values (no Options for required fields)
#[derive(Debug, Clone)]
pub struct ResolvedSamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub stop_tokens: Vec<String>,
    pub seed: Option<u64>,
    pub length_penalty: f32,
    pub typical_p: Option<f32>,
    pub epsilon_cutoff: Option<f32>,
    pub eta_cutoff: Option<f32>,
    pub do_sample: bool,
    // NEW: Async parameters
    pub timeout_ms: u64,
}

/// Builder for generation requests using the unified SamplingParams precedence system
pub struct GenerationRequestBuilder {
    prompt: String,
    params: SamplingParams,
    images: Vec<String>,
    collect_metrics: bool,
}

impl GenerationRequestBuilder {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            params: SamplingParams::default(),
            images: vec![],
            collect_metrics: false, // Default: off for performance
        }
    }

    /// Apply a config layer (server defaults, model defaults, or user overrides).
    /// Later calls take precedence over earlier calls.
    pub fn apply_config(mut self, config: &SamplingParams) -> Self {
        self.params = self.params.merge(config.clone());
        self
    }
    pub fn temperature(mut self, value: f32) -> Self {
        self.params.temperature = Some(value);
        self
    }

    pub fn max_tokens(mut self, value: usize) -> Self {
        self.params.max_tokens = Some(value);
        self
    }

    pub fn top_p(mut self, value: f32) -> Self {
        self.params.top_p = Some(value);
        self
    }

    pub fn top_k(mut self, value: Option<usize>) -> Self {
        self.params.top_k = value;
        self
    }

    pub fn repeat_penalty(mut self, value: f32) -> Self {
        self.params.repeat_penalty = Some(value);
        self
    }

    pub fn repeat_last_n(mut self, value: usize) -> Self {
        self.params.repeat_last_n = Some(value);
        self
    }

    pub fn stop_tokens(mut self, value: Vec<String>) -> Self {
        self.params.stop_tokens = Some(value);
        self
    }

    pub fn seed(mut self, value: Option<u64>) -> Self {
        self.params.seed = value;
        self
    }

    pub fn image_path(mut self, value: std::path::PathBuf) -> Self {
        self.images.push(value.to_string_lossy().to_string());
        self
    }

    pub fn images(mut self, value: Vec<String>) -> Self {
        self.images = value;
        self
    }

    // Async configuration methods
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.params.timeout_ms = Some(timeout.as_millis() as u64);
        self
    }

    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.params.timeout_ms = Some(timeout_ms);
        self
    }

    pub fn build_v2(self) -> GenerationRequestV2 {
        GenerationRequestV2 {
            prompt: self.prompt,
            params: self.params.resolve(),
        }
    }

    /// Enable quality metrics collection (expensive - ~10x slowdown)
    pub fn collect_metrics(mut self, value: bool) -> Self {
        self.collect_metrics = value;
        self
    }

    pub fn build(self) -> GenerationRequest {
        let resolved = self.params.resolve();
        GenerationRequest {
            prompt: TemplatedPrompt::new(self.prompt),
            max_tokens: resolved.max_tokens,
            temperature: resolved.temperature,
            top_p: resolved.top_p,
            top_k: resolved.top_k,
            repeat_penalty: resolved.repeat_penalty,
            repeat_last_n: resolved.repeat_last_n,
            stop_tokens: resolved.stop_tokens,
            seed: resolved.seed.map(|s| s as u32),
            images: self.images,
            timeout: Some(resolved.timeout_ms),
            collect_metrics: self.collect_metrics,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationRequestV2 {
    pub prompt: String,
    pub params: ResolvedSamplingParams,
}

impl GenerationRequest {
    pub fn builder(prompt: impl Into<String>) -> GenerationRequestBuilder {
        GenerationRequestBuilder::new(prompt)
    }
}

impl From<&crate::config::server::SamplingParamDefaults> for SamplingParams {
    fn from(defaults: &crate::config::server::SamplingParamDefaults) -> Self {
        Self {
            max_tokens: Some(defaults.max_tokens),
            temperature: Some(defaults.temperature),
            top_p: Some(defaults.top_p),
            repeat_penalty: Some(defaults.repeat_penalty),
            top_k: None,
            repeat_last_n: None,
            stop_tokens: None,
            seed: None,
            length_penalty: None,
            typical_p: None,
            epsilon_cutoff: None,
            eta_cutoff: None,
            do_sample: None,
            timeout_ms: None, // Don't set timeout here - let engine handle it
        }
    }
}

impl From<&GenerationConfig> for GenerationRequest {
    fn from(config: &GenerationConfig) -> Self {
        Self {
            prompt: TemplatedPrompt::new(String::new()),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: 64, // Default repeat_last_n
            stop_tokens: config.stop_tokens.clone(),
            images: vec![],  // No images in conversion
            seed: config.seed,
            timeout: None, // Not in GenerationConfig
            collect_metrics: false, // Default: off for performance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();

        // Verify max_tokens is set to 2048 (not 100)
        assert_eq!(config.max_tokens, 2048, "Default max_tokens should be 2048 for thinking mode support");

        // Verify other reasonable defaults
        assert!(config.temperature > 0.0, "Temperature should be non-zero");
        assert!(config.top_p > 0.0 && config.top_p <= 1.0, "top_p should be in valid range");
    }

    #[test]
    fn test_generation_request_builder() {
        let request = GenerationRequest::builder("test prompt")
            .temperature(0.8)
            .top_k(Some(30))
            .max_tokens(1000)
            .build();

        assert_eq!(request.prompt, TemplatedPrompt::new("test prompt".to_string()));
        assert_eq!(request.temperature, 0.8);
        assert_eq!(request.top_k, Some(30));
        assert_eq!(request.max_tokens, 1000);
    }

    #[test]
    fn test_clean_config_precedence() {
        let server_defaults = SamplingParams {
            max_tokens: Some(1024),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(50),
            repeat_penalty: Some(1.1),
            ..Default::default()
        };

        let model_defaults = SamplingParams {
            temperature: Some(0.7),
            top_k: Some(40),
            typical_p: Some(0.95),
            ..Default::default()
        };

        let user_overrides = SamplingParams {
            temperature: Some(0.9),
            max_tokens: Some(512),
            ..Default::default()
        };

        let final_config = server_defaults
            .merge(model_defaults)
            .merge(user_overrides);

        assert_eq!(final_config.temperature, Some(0.9));
        assert_eq!(final_config.max_tokens, Some(512));
        assert_eq!(final_config.top_k, Some(40));
        assert_eq!(final_config.top_p, Some(0.9));
        assert_eq!(final_config.repeat_penalty, Some(1.1));
        assert_eq!(final_config.typical_p, Some(0.95));
    }

    #[test]
    fn test_builder_flow() {
        let server_params = SamplingParams {
            max_tokens: Some(2048),
            temperature: Some(0.7),
            top_p: Some(0.95),
            ..Default::default()
        };

        let model_params = SamplingParams {
            temperature: Some(0.6),
            repeat_penalty: Some(1.2),
            ..Default::default()
        };

        let request = GenerationRequest::builder("test prompt")
            .apply_config(&server_params)
            .apply_config(&model_params)
            .temperature(0.8)
            .max_tokens(512)
            .build();

        assert_eq!(request.prompt, TemplatedPrompt::new("test prompt".to_string()));
        assert_eq!(request.temperature, 0.8);
        assert_eq!(request.max_tokens, 512);
        assert_eq!(request.top_p, 0.95);
        assert_eq!(request.repeat_penalty, 1.2);
    }

    #[test]
    fn test_resolved_params() {
        let params = SamplingParams {
            temperature: Some(0.5),
            max_tokens: None,
            ..Default::default()
        };

        let resolved = params.resolve();

        assert_eq!(resolved.temperature, 0.5);
        assert_eq!(resolved.max_tokens, 2048);
        assert_eq!(resolved.top_p, 0.95);
        assert_eq!(resolved.repeat_penalty, 1.0);
        assert_eq!(resolved.do_sample, true);
    }
}

/// Generation result with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: FinishReason,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    /// Quality metrics for self-supervised training
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_metrics: Option<GenerationQualityMetrics>,

    // Prefill metrics (processing the prompt)
    #[serde(default)]
    pub prefill_tokens: usize,
    #[serde(default)]
    pub prefill_time_ms: u64,
    #[serde(default)]
    pub prefill_tokens_per_sec: f32,

    // Inference metrics (generating new tokens, excluding prefill)
    #[serde(default)]
    pub inference_tokens: usize,
    #[serde(default)]
    pub inference_time_ms: u64,
    #[serde(default)]
    pub inference_tokens_per_sec: f32,
}

/// Why generation stopped
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokens,
    StopToken(String),
    EndOfSequence,
    Error(String),
    Stop,
}

// =============================================================================
// Training Mode Configuration (Phase D)
// =============================================================================

/// Model-level training mode configuration (embedded in config.json under "hyprstream_training")
///
/// This allows inference to automatically adapt models when enabled.
/// The training mode is set via `hyprstream training set test_time_training`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyprstreamTrainingConfig {
    /// Training mode: disabled, test_time_training, supervised
    #[serde(default)]
    pub mode: TrainingMode,

    /// Target adapter to train (e.g., "01_coding")
    pub target_adapter: Option<String>,

    /// Learning rate for training
    #[serde(default = "default_training_learning_rate")]
    pub learning_rate: f64,

    /// Batch size for training (used by supervised mode)
    #[serde(default = "default_training_batch_size")]
    pub batch_size: usize,

    /// Training steps per cycle (used by supervised mode)
    #[serde(default = "default_training_steps_per_cycle")]
    pub steps_per_cycle: usize,

    /// Minimum quality score to keep examples (0.0-1.0)
    #[serde(default = "default_training_min_quality")]
    pub min_quality_threshold: f32,

    /// Enable training on base model weights (vs LoRA only)
    #[serde(default)]
    pub train_base_model: bool,

    /// TTT-specific configuration (for TestTimeTraining mode)
    #[serde(default)]
    pub ttt: TTTTrainingConfig,
}

/// TTT-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTTTrainingConfig {
    /// Learning rate for TTT adaptation (higher than fine-tuning)
    #[serde(default = "default_ttt_learning_rate")]
    pub learning_rate: f64,

    /// Number of gradient steps per input
    #[serde(default = "default_ttt_gradient_steps")]
    pub gradient_steps: usize,

    /// Maximum gradient norm for clipping
    #[serde(default = "default_ttt_max_grad_norm")]
    pub max_grad_norm: f64,

    /// Minimum input length (tokens) to trigger TTT
    #[serde(default = "default_ttt_min_input_length")]
    pub min_input_length: usize,

    /// Maximum input length to process for TTT
    #[serde(default = "default_ttt_max_context")]
    pub max_ttt_context: usize,
}

fn default_ttt_learning_rate() -> f64 {
    3e-4
}
fn default_ttt_gradient_steps() -> usize {
    3
}
fn default_ttt_max_grad_norm() -> f64 {
    1.0
}
fn default_ttt_min_input_length() -> usize {
    32
}
fn default_ttt_max_context() -> usize {
    512
}

impl Default for TTTTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_ttt_learning_rate(),
            gradient_steps: default_ttt_gradient_steps(),
            max_grad_norm: default_ttt_max_grad_norm(),
            min_input_length: default_ttt_min_input_length(),
            max_ttt_context: default_ttt_max_context(),
        }
    }
}

impl HyprstreamTrainingConfig {
    /// Check if training is enabled (mode != Disabled)
    pub fn is_enabled(&self) -> bool {
        self.mode != TrainingMode::Disabled
    }
}

impl Default for HyprstreamTrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::default(),
            target_adapter: None,
            learning_rate: default_training_learning_rate(),
            batch_size: default_training_batch_size(),
            steps_per_cycle: default_training_steps_per_cycle(),
            min_quality_threshold: default_training_min_quality(),
            train_base_model: false,
            ttt: TTTTrainingConfig::default(),
        }
    }
}

/// Training mode configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    /// Training disabled (default)
    #[default]
    Disabled,
    /// Test-Time Training: adapts to input context before generation
    /// Research-valid approach based on TTT-E2E
    TestTimeTraining,
    /// Supervised training with explicit training data
    Supervised,
}

// Default functions for HyprstreamTrainingConfig
fn default_training_learning_rate() -> f64 {
    1e-5
}
fn default_training_batch_size() -> usize {
    4
}
fn default_training_steps_per_cycle() -> usize {
    10
}
fn default_training_min_quality() -> f32 {
    0.3
}
