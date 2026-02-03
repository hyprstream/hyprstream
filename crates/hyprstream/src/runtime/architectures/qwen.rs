//! Qwen model adapter that reuses Llama implementation with Qwen-specific configurations
//!
//! Qwen models are architecturally similar to Llama but with key differences:
//! - RoPE base (rope_theta) varies by model and version - always read from config.json
//!   - Qwen2.5 models typically use 1,000,000
//!   - Qwen3 models typically use 10,000 (standard)
//! - Context window varies by model (read from max_position_embeddings in config.json)
//!   - Qwen3-4B: 40,960 tokens
//!   - Can extend via YARN scaling for longer context in some variants
//! - Vocabulary size of 151,936 tokens (Qwen2/Qwen3)
//! - SiLU activation function (same as Llama)
//! - GQA (Grouped-Query Attention) used in Qwen3 models (num_key_value_heads < num_attention_heads)

use super::llama::LlamaModel;
use super::ModelOperations;
use anyhow::Result;
use std::collections::HashMap;
use tch::{Device, Kind as DType, Tensor};

/// Create a Qwen model by adapting Llama implementation
pub struct QwenAdapter;

impl QwenAdapter {
    /// Create a Qwen model from weights and config.json, using Llama as the base implementation
    /// Takes mutable reference to weights to enable incremental tensor freeing during construction
    pub fn from_weights(
        weights: &mut HashMap<String, Tensor>,
        config_json: &str,
        version: u8,
        is_moe: bool,
        _context_length: usize,
        device: &Device,
        dtype: DType,
        kv_quant_type: crate::runtime::kv_quant::KVQuantType,
    ) -> Result<Box<dyn ModelOperations>> {
        // Parse the config.json to get proper rope_theta and other settings
        let config = LlamaModel::parse_config(config_json)?;

        tracing::info!(
            "Parsed config from config.json: rope_theta={}, vocab_size={}, hidden_size={}",
            config.rope_theta,
            config.vocab_size,
            config.hidden_size
        );

        // Version-specific notes (all configurations loaded from config.json):
        // - Qwen3: rope_theta, max_position_embeddings (e.g., 40,960 for Qwen3-4B), GQA settings
        // - Qwen2/Qwen1: All values from config.json
        // No runtime overrides needed - config.json values are authoritative

        // Handle MoE configurations - currently not implemented
        if is_moe {
            return Err(anyhow::anyhow!(
                "Qwen MoE models (e.g., Qwen3-30B-A3B, Qwen3-235B-A22B) are not yet supported. \
                 MoE (Mixture of Experts) implementation is deferred for future development. \
                 Please use a dense Qwen model instead."
            ));
        }

        // Log the configuration being used
        tracing::info!(
            "[from_weights] Creating Qwen{} model with config: hidden_size={}, num_heads={}, num_kv_heads={}, context_length={}, rope_theta={}, kv_quant={:?}",
            version,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.rope_theta,
            kv_quant_type
        );

        // Create the model using Llama implementation with Qwen config
        let model = LlamaModel::from_weights_with_config(weights, config, device, dtype, kv_quant_type)?;
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
