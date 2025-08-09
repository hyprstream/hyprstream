//! Unified configuration system for LLaMA.cpp-based inference

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
}

/// Model loading and identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to GGUF model file
    pub path: PathBuf,
    /// Model identifier (e.g., "qwen2-1.5b")
    pub name: String,
    /// Architecture type ("llama", "qwen", etc.)
    pub architecture: String,
    /// Expected parameter count
    pub parameters: Option<u64>,
}

/// LLaMA.cpp runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

impl Default for HyprConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                path: PathBuf::from("./models/default.gguf"),
                name: "default".to_string(),
                architecture: "llama".to_string(),
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
    /// Load configuration from JSON file
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }
    
    /// Save configuration to JSON file
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }
    
    /// Create generation request from config + prompt
    pub fn create_request(&self, prompt: String) -> GenerationRequest {
        let mut request = GenerationRequest::from(&self.generation);
        request.prompt = prompt;
        request
    }
}