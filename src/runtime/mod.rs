//! Runtime abstraction layer for different ML inference engines

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;
use serde::{Deserialize, Serialize};

pub mod llamacpp_engine;
pub mod lora_wrapper;

pub use llamacpp_engine::LlamaCppEngine;
pub use lora_wrapper::{LoRAEngineWrapper, LoRAConfig, RuntimeLoRAAdapter};
pub use crate::adapters::sparse_lora::SparseLoRAAdapter;

/// Abstract runtime engine trait for different inference backends
#[async_trait]
pub trait RuntimeEngine: Send + Sync {
    /// Load a model from the given path
    async fn load_model(&mut self, path: &Path) -> Result<()>;
    
    /// Generate text from a prompt
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
    
    /// Generate text with additional parameters
    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult>;
    
    /// Get model information
    fn model_info(&self) -> ModelInfo;
    
    /// Check if model is loaded
    fn is_loaded(&self) -> bool;
}

/// Model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub parameters: u64,
    pub context_length: usize,
    pub vocab_size: usize,
    pub architecture: String,
    pub quantization: Option<String>,
}

/// Generation request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub stop_tokens: Vec<String>,
    pub seed: Option<u32>,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 50,
            temperature: 0.7,
            top_p: 1.0,
            top_k: None,
            stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
            seed: None,
        }
    }
}

/// Generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: FinishReason,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
}

/// Reason why generation finished
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokens,
    StopToken,
    EndOfSequence,
    Error(String),
}

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub context_length: usize,
    pub batch_size: usize,
    pub num_threads: Option<usize>,
    pub use_gpu: bool,
    pub gpu_layers: Option<usize>,
    pub mmap: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            context_length: 2048,
            batch_size: 512,
            num_threads: None,
            use_gpu: false,
            gpu_layers: None,
            mmap: true,
        }
    }
}