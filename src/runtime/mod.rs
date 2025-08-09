//! Runtime abstraction layer for LLaMA.cpp inference

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

// Re-export everything from the unified config
pub use crate::config::{
    HyprConfig, ModelConfig, RuntimeConfig, GenerationConfig, LoRAConfig,
    ModelInfo, GenerationRequest, GenerationResult, FinishReason
};

pub mod llamacpp_engine;
pub mod lora_wrapper;

pub use llamacpp_engine::LlamaCppEngine;
pub use lora_wrapper::{LoRAEngineWrapper, RuntimeLoRAAdapter};
pub use crate::adapters::sparse_lora::SparseLoRAAdapter;

/// Core runtime engine trait - all engines implement this
#[async_trait]
pub trait RuntimeEngine: Send + Sync {
    /// Load a GGUF model from the given path
    async fn load_model(&mut self, path: &Path) -> Result<()>;
    
    /// Generate text from a prompt (convenience method)
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
    
    /// Generate text with full parameters (main method)
    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult>;
    
    /// Get loaded model information
    fn model_info(&self) -> ModelInfo;
    
    /// Check if model is loaded and ready
    fn is_loaded(&self) -> bool;
}

/// Create the default LLaMA.cpp runtime engine
pub fn create_engine(config: &RuntimeConfig) -> Result<LlamaCppEngine> {
    LlamaCppEngine::new(config.clone())
}

/// Create engine with LoRA wrapper
pub fn create_lora_engine(
    base_engine: Box<dyn RuntimeEngine>, 
    _lora_config: &LoRAConfig
) -> Result<LoRAEngineWrapper> {
    use std::sync::Arc;
    LoRAEngineWrapper::new(Arc::from(base_engine))
}