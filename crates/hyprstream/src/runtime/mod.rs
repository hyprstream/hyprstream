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
    FinishReason, GenerationConfig, GenerationRequest, GenerationResult, HyprConfig, LoRAConfig,
    ModelConfig, ModelInfo, RuntimeConfig,
};

pub mod architectures; // Architecture-specific model implementations (includes Janus placeholder utils)
// REMOVED: pub mod conversation_router; // Dead code - VDB TemporalStreamingLayer removed
pub mod generation_metrics; // Quality metrics for self-supervised training
pub mod kv_quant; // KV cache quantization types
pub mod tensor_sampling; // Device-agnostic tensor-based sampling
pub mod image_utils; // Image loading and preprocessing for multimodal models
pub mod kv_cache; // Key-Value caching for efficient autoregressive generation
pub mod lora_integration; // LoRA integration with gradient bridge
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
pub struct XLoRAAdapter {
    pub id: String,
}

#[derive(Debug, Clone)]
pub enum AdaptationMode {
    Disabled,
}

#[derive(Debug, Clone)]
pub struct UserFeedback;

#[derive(Debug, Clone)]
pub enum XLoRARoutingStrategy {
    Default,
}

#[derive(Debug, Clone, Default)]
pub struct AdapterMetrics;

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

    /// Apply chat template to messages (for template support)
    fn apply_chat_template(
        &self,
        messages: &[template_engine::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        // Default implementation with simple formatting
        let mut formatted = String::new();
        for msg in messages {
            formatted.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        if add_generation_prompt {
            formatted.push_str("assistant: ");
        }
        Ok(formatted)
    }

    // NEW: X-LoRA and real-time adaptation capabilities (default implementations for backward compatibility)

    /// Update adapter weights in real-time (< 5ms target)
    async fn update_adapter_realtime(
        &mut self,
        _adapter_id: &str,
        _weights: &crate::adapters::lora_checkpoints::LoRAWeightsData,
    ) -> Result<()> {
        Err(anyhow::anyhow!(
            "Real-time adapter updates not supported by this engine"
        ))
    }

    /// Switch active adapters instantly (< 1ms target)
    async fn switch_active_adapters(&mut self, _adapter_ids: &[String]) -> Result<()> {
        Err(anyhow::anyhow!(
            "Adapter switching not supported by this engine"
        ))
    }

    /// Configure X-LoRA multi-adapter routing
    async fn configure_xlora(
        &mut self,
        _max_adapters: usize,
        _routing_strategy: XLoRARoutingStrategy,
    ) -> Result<()> {
        Err(anyhow::anyhow!("X-LoRA not supported by this engine"))
    }

    /// Enable real-time adaptation mode
    async fn enable_realtime_adaptation(&mut self, _mode: AdaptationMode) -> Result<()> {
        Err(anyhow::anyhow!(
            "Real-time adaptation not supported by this engine"
        ))
    }

    /// Process generation feedback for learning
    async fn process_generation_feedback(
        &mut self,
        _request: &GenerationRequest,
        _result: &GenerationResult,
        _feedback: Option<UserFeedback>,
    ) -> Result<()> {
        // Default: no-op for engines that don't support learning
        Ok(())
    }

    /// Get adapter performance metrics
    async fn get_adapter_metrics(
        &self,
    ) -> Result<std::collections::HashMap<String, AdapterMetrics>> {
        Ok(std::collections::HashMap::new())
    }

    /// Load LoRA checkpoint as adapter
    async fn load_lora_checkpoint(
        &mut self,
        _checkpoint: &crate::adapters::lora_checkpoints::LoRACheckpoint,
    ) -> Result<String> {
        Err(anyhow::anyhow!(
            "LoRA checkpoint loading not supported by this engine"
        ))
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
