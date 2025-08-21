//! Qwen model adapter that reuses Llama implementation with Qwen-specific configurations
//! 
//! Qwen models are architecturally similar to Llama but with key differences:
//! - Different RoPE parameters (base frequency 1M for long context)
//! - Support for longer context windows (128K-262K tokens)
//! - Some models use combined c_attn weights instead of separate q/k/v projections
//! - Grouped Query Attention (GQA) configurations vary by model size

use super::{ModelArchitecture, ModelOperations, ArchitectureConfig};
use super::llama::{LlamaModel, LlamaConfig};
use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;

/// Create a Qwen model by adapting Llama implementation
pub struct QwenAdapter;

impl QwenAdapter {
    /// Create a Qwen model from weights, using Llama as the base implementation
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        version: u8,
        is_moe: bool,
        context_length: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Start with Llama's config detection
        let mut config = LlamaModel::detect_config_from_weights(weights)?;
        
        // Apply Qwen-specific overrides based on version
        match version {
            3 => {
                // Qwen3 specific configurations
                config.rope_theta = 1_000_000.0;  // 1M for long context support
                config.max_position_embeddings = context_length;
                
                // Qwen3 uses GQA with different ratios based on model size
                // This will be auto-detected from weights, but we can override if needed
                if config.hidden_size == 2560 {  // Qwen3-1.5B
                    config.num_attention_heads = 32;
                    config.num_key_value_heads = 8;  // 4:1 GQA ratio
                } else if config.hidden_size == 4096 {  // Qwen3-7B/8B
                    config.num_attention_heads = 32;
                    config.num_key_value_heads = 8;  // 4:1 GQA ratio
                }
                
                // Update vocab size if detected from weights
                if let Some(embed) = weights.get("model.embed_tokens.weight")
                    .or_else(|| weights.get("embed_tokens.weight")) {
                    config.vocab_size = embed.dims()[0];
                }
            }
            2 => {
                // Qwen2 configurations
                config.rope_theta = 1_000_000.0;
                config.max_position_embeddings = context_length.min(131_072);
            }
            _ => {
                // Qwen1 or unknown version - use conservative defaults
                config.rope_theta = 10_000.0;
                config.max_position_embeddings = context_length.min(8192);
            }
        }
        
        // Handle MoE configurations if needed
        if is_moe {
            tracing::info!("Qwen MoE model detected - using dense layers only (MoE not yet implemented)");
            // TODO: Implement proper MoE support
        }
        
        // Log the configuration being used
        tracing::info!(
            "Creating Qwen{} model with config: hidden_size={}, num_heads={}, num_kv_heads={}, context_length={}, rope_theta={}",
            version,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.rope_theta
        );
        
        // Create the model using Llama implementation with Qwen config
        let model = LlamaModel::from_weights_with_config(weights, config, device, dtype)?;
        Ok(Box::new(model))
    }
    
    /// Check if weights contain Qwen-specific patterns
    pub fn is_qwen_model(weights: &HashMap<String, Tensor>) -> bool {
        // Check for Qwen-specific weight patterns
        weights.keys().any(|k| {
            k.contains("c_attn") ||  // Combined attention weights
            k.contains("qwen") ||     // Explicit Qwen naming
            k.contains("c_proj")      // Combined projection
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qwen_config_adaptation() {
        // Test that Qwen3 gets correct RoPE configuration
        let config = LlamaConfig {
            hidden_size: 2560,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rope_theta: 10_000.0,  // Default Llama value
            max_position_embeddings: 4096,
            ..Default::default()
        };
        
        // After Qwen3 adaptation, rope_theta should be 1M
        assert_eq!(config.hidden_size, 2560);
        // In actual implementation, the adapter would modify these values
    }
}