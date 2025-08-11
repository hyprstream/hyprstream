//! Runtime abstraction layer for inference engines
//! 
//! This module provides a unified interface for different inference engines:
//! - MistralEngine: Primary engine with X-LoRA and real-time adaptation (NEW)
//! - LlamaCppEngine: Legacy engine for reference during migration (DEPRECATED)

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

// Re-export everything from the unified config
pub use crate::config::{
    HyprConfig, ModelConfig, RuntimeConfig, GenerationConfig, LoRAConfig,
    ModelInfo, GenerationRequest, GenerationResult, FinishReason
};

pub mod mistral_engine;      // NEW: Primary inference engine
pub mod llamacpp_engine;     // TEMPORARY: Keep during migration
pub mod lora_wrapper;

// Primary exports - use MistralEngine as default
pub use mistral_engine::{
    MistralEngine, XLoRAAdapter, AdaptationMode, UserFeedback, 
    XLoRARoutingStrategy, AdapterMetrics, ModelBuilderConfig
};

// Temporary exports for migration period
pub use llamacpp_engine::LlamaCppEngine;
pub use lora_wrapper::{LoRAEngineWrapper, RuntimeLoRAAdapter};
pub use crate::adapters::sparse_lora::SparseLoRAAdapter;

/// Core runtime engine trait - all engines implement this
#[async_trait]
pub trait RuntimeEngine: Send + Sync {
    /// Load a model from the given path
    async fn load_model(&mut self, path: &Path) -> Result<()>;
    
    /// Generate text from a prompt (convenience method)
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
    
    /// Generate text with full parameters (main method)
    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult>;
    
    /// Get loaded model information
    fn model_info(&self) -> ModelInfo;
    
    /// Check if model is loaded and ready
    fn is_loaded(&self) -> bool;
    
    // NEW: X-LoRA and real-time adaptation capabilities (default implementations for backward compatibility)
    
    /// Update adapter weights in real-time (< 5ms target)
    async fn update_adapter_realtime(&mut self, _adapter_id: &str, _weights: &crate::adapters::lora_checkpoints::LoRAWeightsData) -> Result<()> {
        Err(anyhow::anyhow!("Real-time adapter updates not supported by this engine"))
    }
    
    /// Switch active adapters instantly (< 1ms target)
    async fn switch_active_adapters(&mut self, _adapter_ids: &[String]) -> Result<()> {
        Err(anyhow::anyhow!("Adapter switching not supported by this engine"))
    }
    
    /// Configure X-LoRA multi-adapter routing
    async fn configure_xlora(&mut self, _max_adapters: usize, _routing_strategy: XLoRARoutingStrategy) -> Result<()> {
        Err(anyhow::anyhow!("X-LoRA not supported by this engine"))
    }
    
    /// Enable real-time adaptation mode
    async fn enable_realtime_adaptation(&mut self, _mode: AdaptationMode) -> Result<()> {
        Err(anyhow::anyhow!("Real-time adaptation not supported by this engine"))
    }
    
    /// Process generation feedback for learning
    async fn process_generation_feedback(&mut self, 
                                       _request: &GenerationRequest, 
                                       _result: &GenerationResult,
                                       _feedback: Option<UserFeedback>) -> Result<()> {
        // Default: no-op for engines that don't support learning
        Ok(())
    }
    
    /// Get adapter performance metrics
    async fn get_adapter_metrics(&self) -> Result<std::collections::HashMap<String, AdapterMetrics>> {
        Ok(std::collections::HashMap::new())
    }
    
    /// Load LoRA checkpoint as adapter
    async fn load_lora_checkpoint(&mut self, _checkpoint: &crate::adapters::lora_checkpoints::LoRACheckpoint) -> Result<String> {
        Err(anyhow::anyhow!("LoRA checkpoint loading not supported by this engine"))
    }
}

/// Create the default runtime engine (now uses mistral.rs)
pub fn create_engine(config: &RuntimeConfig) -> Result<MistralEngine> {
    MistralEngine::new(config.clone())
}

/// Create the legacy LLaMA.cpp engine (deprecated)
#[deprecated(note = "Use create_engine() which returns MistralEngine instead")]
pub fn create_llamacpp_engine(config: &RuntimeConfig) -> Result<LlamaCppEngine> {
    LlamaCppEngine::new(config.clone())
}

/// Create engine with LoRA wrapper (deprecated - use X-LoRA instead)
#[deprecated(note = "Use MistralEngine with X-LoRA for better performance")]
pub fn create_lora_engine(
    base_engine: Box<dyn RuntimeEngine>, 
    _lora_config: &LoRAConfig
) -> Result<LoRAEngineWrapper> {
    use std::sync::Arc;
    LoRAEngineWrapper::new(Arc::from(base_engine))
}