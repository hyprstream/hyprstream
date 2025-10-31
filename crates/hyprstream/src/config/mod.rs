//! Unified configuration system for Hyprstream
//!
//! This module provides a layered configuration architecture:
//! - `HyprConfig`: Root configuration combining all subsystems
//! - `ServerConfig`: HTTP server configuration (network, CORS, TLS)
//! - Model and runtime configs for ML inference

pub mod server;

// Re-export main configuration types
pub use server::{CorsConfig, GenerationDefaults, ServerConfig, ServerConfigBuilder};

// Export root configuration and builder (defined below in this module)
// Note: HyprConfig and HyprConfigBuilder are exported automatically as pub structs

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
}

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
        let storage_paths = StoragePaths::new().expect("Failed to initialize storage paths");

        Self {
            models_dir: storage_paths
                .models_dir()
                .unwrap_or_else(|_| PathBuf::from("./models")),
            loras_dir: storage_paths
                .loras_dir()
                .unwrap_or_else(|_| PathBuf::from("./loras")),
            cache_dir: storage_paths
                .cache_dir()
                .unwrap_or_else(|_| PathBuf::from("./cache")),
            config_dir: storage_paths
                .config_dir()
                .unwrap_or_else(|_| PathBuf::from("./config")),
        }
    }
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
    /// Batch processing size
    pub batch_size: usize,
    /// CPU threads (None = auto-detect)
    pub cpu_threads: Option<usize>,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// GPU layers to offload (None = auto)
    pub gpu_layers: Option<usize>,
    /// Use memory mapping for model files
    pub mmap: bool,
    /// KV cache size in MB
    pub kv_cache_size_mb: usize,
    /// Precision mode (BF16/FP16/FP32/FP8)
    pub precision_mode: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            context_length: 4096,
            batch_size: 512,
            cpu_threads: None,
            use_gpu: true,
            gpu_layers: None,
            mmap: true,
            kv_cache_size_mb: 2048,
            precision_mode: Some("auto".to_string()),
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
        }
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
            // Load from environment variables with HYPRSTREAM_ prefix
            .add_source(Environment::with_prefix("HYPRSTREAM").separator("__"));

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
    pub fn create_request(&self, prompt: String) -> GenerationRequest {
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

/// Generation request with all parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub stop_tokens: Vec<String>,
    pub seed: Option<u32>,
}

impl GenerationRequest {
    /// Create a builder for ergonomic construction
    pub fn builder(prompt: impl Into<String>) -> GenerationRequestBuilder {
        GenerationRequestBuilder::new(prompt)
    }
}

/// Builder for GenerationRequest with parameter cascading
pub struct GenerationRequestBuilder {
    prompt: String,
    sampling: crate::runtime::sampling::SamplingConfig,
}

impl GenerationRequestBuilder {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            sampling: crate::runtime::sampling::SamplingConfig::default(),
        }
    }

    /// Load model-specific configuration from model path
    pub async fn with_model_config(
        mut self,
        model_path: &std::path::Path,
    ) -> anyhow::Result<Self> {
        self.sampling = crate::runtime::sampling::load_sampling_config(model_path).await?;
        Ok(self)
    }

    /// Merge with server or system defaults (self takes priority)
    pub fn with_defaults(mut self, defaults: &crate::runtime::sampling::SamplingConfig) -> Self {
        self.sampling = self.sampling.merge_with_defaults(defaults);
        self
    }

    /// Convenience method to apply server defaults from GenerationDefaults config
    pub fn with_server_defaults(self, defaults: &GenerationDefaults) -> Self {
        let server_sampling = crate::runtime::sampling::SamplingConfig {
            max_tokens: defaults.max_tokens,
            temperature: defaults.temperature,
            top_p: Some(defaults.top_p),
            repeat_penalty: defaults.repeat_penalty,
            ..Default::default()
        };
        self.with_defaults(&server_sampling)
    }

    /// Apply user overrides
    pub fn temperature(mut self, temp: Option<f32>) -> Self {
        self.sampling = self.sampling.apply_temperature(temp);
        self
    }

    pub fn top_p(mut self, top_p: Option<f32>) -> Self {
        self.sampling = self.sampling.apply_top_p(top_p);
        self
    }

    pub fn top_k(mut self, top_k: Option<usize>) -> Self {
        self.sampling = self.sampling.apply_top_k(top_k);
        self
    }

    pub fn repeat_penalty(mut self, repeat_penalty: Option<f32>) -> Self {
        self.sampling = self.sampling.apply_repeat_penalty(repeat_penalty);
        self
    }

    pub fn max_tokens(mut self, max_tokens: Option<usize>) -> Self {
        self.sampling = self.sampling.apply_max_tokens(max_tokens);
        self
    }

    pub fn stop_tokens(mut self, stop_tokens: Option<Vec<String>>) -> Self {
        self.sampling = self.sampling.apply_stop_tokens(stop_tokens);
        self
    }

    pub fn build(self) -> GenerationRequest {
        GenerationRequest {
            prompt: self.prompt,
            max_tokens: self.sampling.max_tokens,
            temperature: self.sampling.temperature,
            top_p: self.sampling.top_p.unwrap_or(1.0),
            top_k: self.sampling.top_k,
            repeat_penalty: self.sampling.repeat_penalty,
            stop_tokens: self.sampling.stop_tokens,
            seed: self.sampling.seed.map(|s| s as u32),
        }
    }
}

impl From<&GenerationConfig> for GenerationRequest {
    fn from(config: &GenerationConfig) -> Self {
        Self {
            prompt: String::new(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            stop_tokens: config.stop_tokens.clone(),
            seed: config.seed,
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
            .temperature(Some(0.8))
            .top_k(Some(30))
            .max_tokens(Some(1000))
            .build();

        assert_eq!(request.prompt, "test prompt");
        assert_eq!(request.temperature, 0.8);
        assert_eq!(request.top_k, Some(30));
        assert_eq!(request.max_tokens, 1000);
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
