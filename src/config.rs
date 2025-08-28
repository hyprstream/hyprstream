//! Unified configuration system for LLaMA.cpp-based inference

use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::storage::paths::StoragePaths;

/// Single unified configuration for the entire system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyprConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Runtime execution settings
    pub runtime: RuntimeConfig,
    /// Text generation parameters
    pub generation: GenerationConfig,
    /// LoRA adapter settings
    pub lora: LoRAConfig,
    /// Storage paths configuration
    pub storage: StorageConfig,
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
    /// VDB storage directory path
    pub vdb_storage_dir: PathBuf,
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

/// LLaMA.cpp runtime configuration
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
    pub precision_mode: Option<String>, // "bf16", "fp16", "fp32", "fp8-mixed"
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

impl Default for StorageConfig {
    fn default() -> Self {
        let storage_paths = StoragePaths::new().expect("Failed to initialize storage paths");
        
        Self {
            models_dir: storage_paths.models_dir().unwrap_or_else(|_| PathBuf::from("./models")),
            loras_dir: storage_paths.loras_dir().unwrap_or_else(|_| PathBuf::from("./loras")),
            cache_dir: storage_paths.cache_dir().unwrap_or_else(|_| PathBuf::from("./cache")),
            config_dir: storage_paths.config_dir().unwrap_or_else(|_| PathBuf::from("./config")),
            vdb_storage_dir: storage_paths.cache_dir()
                .unwrap_or_else(|_| PathBuf::from("./cache"))
                .join("vdb_storage"),
        }
    }
}

impl Default for HyprConfig {
    fn default() -> Self {
        Self {
            // No default model - users must explicitly download and configure
            model: ModelConfig {
                path: PathBuf::new(), // Empty path - no default model
                name: String::new(),   // No default name
                architecture: String::new(), // No default architecture
                parameters: None,
            },
            runtime: RuntimeConfig {
                context_length: 4096,
                batch_size: 512,
                cpu_threads: None,
                use_gpu: true,
                gpu_layers: None,
                mmap: true,
                kv_cache_size_mb: 2048,
                precision_mode: Some("auto".to_string()),
            },
            generation: GenerationConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                top_k: Some(40),
                repeat_penalty: 1.1,
                stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
                seed: None,
                stream: false,
            },
            lora: LoRAConfig {
                enabled: true,
                max_adapters: 4,
                alpha: 32.0,
                sparsity: 0.99,
            },
            storage: StorageConfig::default(),
        }
    }
}

/// Model information returned after loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: u64,
    pub context_length: usize,
    pub vocab_size: usize,
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
    pub stream: bool,
    
    // NEW: X-LoRA and real-time adaptation fields
    /// Active adapter IDs for this generation (X-LoRA feature)
    pub active_adapters: Option<Vec<String>>,
    
    /// Real-time adaptation request
    pub realtime_adaptation: Option<RealtimeAdaptationRequest>,
    
    /// User feedback for learning (optional)
    pub user_feedback: Option<UserFeedbackRequest>,
}

/// Real-time adaptation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAdaptationRequest {
    /// Target adapter ID for updates
    pub adapter_id: String,
    
    /// Enable feedback integration for this generation
    pub feedback_integration: bool,
    
    /// Learning rate override (optional)
    pub learning_rate_override: Option<f32>,
}

/// User feedback request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedbackRequest {
    /// Quality score from 0.0 to 1.0
    pub quality_score: f32,
    
    /// Whether the response was helpful
    pub helpful: bool,
    
    /// Optional text corrections or suggestions
    pub corrections: Option<String>,
    
    /// Additional context about the feedback
    pub context: Option<String>,
}

impl From<&GenerationConfig> for GenerationRequest {
    fn from(config: &GenerationConfig) -> Self {
        Self {
            prompt: String::new(), // Will be set by caller
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            stop_tokens: config.stop_tokens.clone(),
            seed: config.seed,
            stream: config.stream,
            active_adapters: None,
            realtime_adaptation: None,
            user_feedback: None,
        }
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
}

impl HyprConfig {
    /// Load configuration using the config crate with XDG directories and environment variables
    pub fn load() -> Result<Self, ConfigError> {
        let storage = StoragePaths::new()
            .map_err(|e| ConfigError::Message(format!("Failed to initialize storage paths: {}", e)))?;
        
        let config_dir = storage.config_dir()
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
        settings.build()?.try_deserialize()
    }
    
    /// Save configuration to TOML file in XDG config directory
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let storage = StoragePaths::new()?;
        let config_dir = storage.config_dir()?;
        let config_path = config_dir.join("config.toml");
        
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;
        
        println!("✅ Configuration saved to: {}", config_path.display());
        Ok(())
    }
    
    /// Load configuration from a specific file path  
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let settings = Config::builder()
            .add_source(Config::try_from(&HyprConfig::default())?)
            .add_source(File::from(path))
            .build()?;
            
        Ok(settings.try_deserialize()?)
    }
    
    /// Save configuration to a specific file path
    pub fn to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("toml");
        let contents = match extension {
            "json" => serde_json::to_string_pretty(self)?,
            "yaml" | "yml" => serde_yaml::to_string(self)?,
            _ => toml::to_string_pretty(self)?,
        };
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Create generation request from config + prompt
    pub fn create_request(&self, prompt: String) -> GenerationRequest {
        let mut request = GenerationRequest::from(&self.generation);
        request.prompt = prompt;
        request
    }
    
    /// Validate that the configuration has a model specified
    pub fn validate(&self) -> Result<(), String> {
        if self.model.path.as_os_str().is_empty() {
            return Err("No model configured. Use 'hyprstream model download <model>' to download a model first.".to_string());
        }
        
        if !self.model.path.exists() {
            return Err(format!(
                "Configured model path does not exist: {}. Use 'hyprstream model list' to see available models.",
                self.model.path.display()
            ));
        }
        
        if self.model.name.is_empty() {
            return Err("Model name is empty. Please configure a proper model.".to_string());
        }
        
        Ok(())
    }
    
    /// Update model configuration after downloading
    pub fn set_model(&mut self, model_path: PathBuf, model_name: String, architecture: String) {
        self.model.path = model_path;
        self.model.name = model_name;
        self.model.architecture = architecture;
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
    
    /// Get the VDB storage directory path
    pub fn vdb_storage_dir(&self) -> &PathBuf {
        &self.storage.vdb_storage_dir
    }
    
    /// Get a specific model path by name
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        use crate::utils::sanitize_filename;
        let sanitized_name = sanitize_filename(model_name);
        self.storage.models_dir.join(sanitized_name)
    }
    
    /// Get a specific LoRA path by name
    pub fn lora_path(&self, lora_name: &str) -> PathBuf {
        use crate::utils::sanitize_filename;
        let sanitized_name = sanitize_filename(lora_name);
        self.storage.loras_dir.join(sanitized_name)
    }
    
    /// Ensure all configured directories exist
    pub fn ensure_directories(&self) -> Result<(), std::io::Error> {
        std::fs::create_dir_all(&self.storage.models_dir)?;
        std::fs::create_dir_all(&self.storage.loras_dir)?;
        std::fs::create_dir_all(&self.storage.cache_dir)?;
        std::fs::create_dir_all(&self.storage.config_dir)?;
        std::fs::create_dir_all(&self.storage.vdb_storage_dir)?;
        Ok(())
    }
    
    /// Create a default configuration for a specific model path
    pub fn default_for_model(model_path: &Path) -> anyhow::Result<Self> {
        let storage_paths = StoragePaths::new()?;
        let mut config = Self::default();
        
        config.model.path = model_path.to_path_buf();
        config.model.name = model_path.file_stem()
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
            vdb_storage_dir: storage_paths.cache_dir()?.join("vdb_storage"),
        };
        
        Ok(config)
    }
}