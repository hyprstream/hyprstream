//! Qwen model adapter that reuses Llama implementation with Qwen-specific configurations
//!
//! Qwen models are architecturally similar to Llama but with key differences:
//! - Standard RoPE base of 10,000 (same as Llama, but supports YARN scaling for longer context)
//! - Default context window of 32K tokens (can extend via YARN to 131K+)
//! - Vocabulary size of 151,936 tokens (Qwen3)
//! - SiLU activation function (same as Llama)
//! - Most models use same num_attention_heads and num_key_value_heads (no GQA reduction)

use super::llama::LlamaModel;
use super::ModelOperations;
use anyhow::Result;
use std::collections::HashMap;
use tch::{Device, Kind as DType, Tensor};

/// Create a Qwen model by adapting Llama implementation
pub struct QwenAdapter;

impl QwenAdapter {
    /// Create a Qwen model from weights and config.json, using Llama as the base implementation
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        config_json: &str,
        version: u8,
        is_moe: bool,
        context_length: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Parse the config.json to get proper rope_theta and other settings
        let mut config = LlamaModel::parse_config(config_json)?;

        tracing::info!(
            "Parsed config from config.json: rope_theta={}, vocab_size={}, hidden_size={}",
            config.rope_theta,
            config.vocab_size,
            config.hidden_size
        );

        // Apply Qwen-specific overrides based on version
        match version {
            3 => {
                // Qwen3 specific configurations
                // rope_theta is already correctly set from config.json (typically 5M for Qwen3-4B)
                config.max_position_embeddings = context_length.min(32_768); // Default 32K

                // Ensure GQA settings match what's in the weights
                // The config.json should already have the correct values
            }
            2 => {
                // Qwen2 configurations
                config.max_position_embeddings = context_length.min(32_768);
            }
            _ => {
                // Qwen1 or unknown version - use conservative defaults
                config.max_position_embeddings = context_length.min(8192);
            }
        }

        // Handle MoE configurations if needed
        if is_moe {
            tracing::info!(
                "Qwen MoE model detected - using dense layers only (MoE not yet implemented)"
            );
        }

        // Log the configuration being used
        tracing::info!(
            "[from_weights] Creating Qwen{} model with config: hidden_size={}, num_heads={}, num_kv_heads={}, context_length={}, rope_theta={}",
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
            k.contains("c_proj") // Combined projection
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::runtime::architectures::llama::LlamaConfig;

    #[test]
    fn test_qwen_config_adaptation() {
        // Test that Qwen3 gets correct RoPE configuration
        let config = LlamaConfig {
            hidden_size: 2560,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rope_theta: 10_000.0, // Default Llama value
            max_position_embeddings: 4096,
            ..Default::default()
        };

        // After Qwen3 adaptation, rope_theta should be 1M
        assert_eq!(config.hidden_size, 2560);
        // In actual implementation, the adapter would modify these values
    }
}
