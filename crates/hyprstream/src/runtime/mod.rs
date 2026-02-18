//! Runtime abstraction layer for inference engines
//!
//! This module provides a unified interface for different inference engines:
//! - TorchEngine: Primary PyTorch-based engine with tch-rs
//! - CandleEngine: Legacy Candle engine (DEPRECATED)
//! - LlamaCppEngine: Legacy engine for reference (DEPRECATED)

use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

// Re-export everything from the unified config
pub use crate::config::{
    FinishReason, GenerationConfig, GenerationRequest, GenerationResult, HyprConfig,
    ModelConfig, ModelInfo, RuntimeConfig,
};

pub mod architectures; // Architecture-specific model implementations (includes Janus placeholder utils)
pub mod batched_lora; // Batched multi-tenant LoRA forward pass
// REMOVED: pub mod conversation_router; // Dead code - VDB TemporalStreamingLayer removed
pub mod generation_metrics; // Quality metrics for self-supervised training
pub mod kv_quant; // KV cache quantization types
pub mod tensor_sampling; // Device-agnostic tensor-based sampling
pub mod image_utils; // Image loading and preprocessing for multimodal models
pub mod kv_cache; // Key-Value caching for efficient autoregressive generation
pub mod model_config; // Unified model configuration management
pub mod model_factory; // Single factory for model creation
pub mod rope; // Rotary Position Embedding (RoPE) implementation
pub mod template_engine; // Jinja2 template engine for chat templates
pub mod tensor_helpers; // Helper functions for Tch tensor operations
pub mod tokenizer_config; // Trait-based tokenizer configuration for models
pub mod torch_engine; // PyTorch-based engine with tch-rs
pub mod torch_utils; // Utilities for safe PyTorch operations with OOM handling
pub mod weight_provider; // Weight provider for streaming large models

// Primary exports - use TorchEngine as default
pub use torch_engine::{TorchEngine, TextStream, GenerationStats};

// KV cache exports for multi-session support
pub use kv_cache::{CacheConfig, CacheOwner, KVCacheManager, KVCacheRegistry};

// Generation metrics exports for self-supervised training
pub use generation_metrics::{GenerationMetricsAccumulator, GenerationQualityMetrics, SessionMetrics};

#[derive(Debug, Clone)]
pub struct MistralEngine;

#[derive(Debug, Clone)]
pub enum ModelBuilderConfig {
    Default,
}

// REMOVED: Conversation routing exports - dead code
// pub use conversation_router::{...};

// LoRA and adapter exports
// LoRA wrapper removed - using direct PyTorch implementation

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

    /// Apply chat template to messages (for template support).
    ///
    /// `tools` is an optional JSON array of tool definitions. When provided, it is
    /// passed to the HF chat template as the `tools` variable so that models with
    /// native tool-calling support can format them correctly.
    fn apply_chat_template(
        &self,
        messages: &[template_engine::ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let _ = tools; // default impl ignores tools
        // Default implementation with simple formatting
        let mut formatted = String::new();
        for msg in messages {
            formatted.push_str(&format!("{}: {}\n", msg.role, msg.content.as_deref().unwrap_or("")));
        }
        if add_generation_prompt {
            formatted.push_str("assistant: ");
        }
        Ok(formatted)
    }

}

/// Create the default runtime engine (now uses PyTorch)
pub fn create_engine(config: &RuntimeConfig) -> Result<TorchEngine> {
    TorchEngine::new(config.clone())
}

/* Commented out - VDB TemporalStreamingLayer removed
/// Create conversation router with model pool and temporal streaming
pub async fn create_conversation_router(
    model_pool: std::sync::Arc<ModelPool>,
    temporal_streaming: std::sync::Arc<crate::storage::vdb::TemporalStreamingLayer>,
    config: Option<RoutingConfig>,
) -> Result<ConversationRouter> {
    ConversationRouter::new(
        model_pool,
        temporal_streaming,
        config.unwrap_or_default(),
    ).await
}
*/
