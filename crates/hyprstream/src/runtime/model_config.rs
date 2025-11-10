//! Unified model configuration management
//!
//! This module provides a single source of truth for model configurations,
//! resolving the current chaos of multiple override points.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tch::Tensor;
use tracing::info;

/// Unified model configuration that combines all sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Architecture identification
    pub architecture: ModelArchitecture,
    pub model_type: String,
    pub version: u32,

    // Core transformer parameters
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,

    // Vocabulary and embeddings
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,

    // Normalization
    pub rms_norm_eps: f32,
    pub layer_norm_eps: Option<f32>,

    // Activation
    pub hidden_activation: String,

    // Special configurations
    pub use_qk_norm: bool,
    pub scale_embeddings: bool,
    pub query_pre_attn_scalar: Option<f32>,

    // Precision and optimization
    pub dtype: String,
    pub use_flash_attention: bool,
    pub use_kv_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelArchitecture {
    Llama,
    Qwen,
    Gemma,
    Mistral,
    Unknown(String),
}

impl ModelConfig {
    /// Load configuration with clear priority:
    /// 1. config.json (if exists)
    /// 2. Weight detection (fill missing values)
    /// 3. Architecture defaults (last resort)
    pub fn load(model_path: &Path, weights: &HashMap<String, Tensor>) -> Result<Self> {
        // Step 1: Try to load config.json
        let config_path = model_path.join("config.json");
        let mut config = if config_path.exists() {
            info!("Loading model configuration");
            Self::from_json_file(&config_path)?
        } else {
            info!("⚠️ No config.json found, detecting from weights");
            Self::detect_from_weights(weights)?
        };

        // Step 2: Validate against weights
        config.validate_with_weights(weights)?;

        // Step 3: Log final configuration
        info!("✅ Final model configuration:");
        info!("   Architecture: {:?}", config.architecture);
        info!("   Hidden size: {}", config.hidden_size);
        info!("   Layers: {}", config.num_hidden_layers);
        info!("   Attention heads: {}", config.num_attention_heads);
        info!("   KV heads: {}", config.num_key_value_heads);
        info!("   Vocab size: {}", config.vocab_size);
        info!("   RoPE theta: {}", config.rope_theta);
        info!("   Max position: {}", config.max_position_embeddings);
        info!("   RMSNorm eps: {}", config.rms_norm_eps);

        Ok(config)
    }

    /// Load from config.json
    fn from_json_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;

        // Detect architecture from model_type or architectures field
        let architecture = Self::detect_architecture_from_json(&json);

        // Extract all configuration values
        let config = Self {
            architecture: architecture.clone(),
            model_type: json["model_type"].as_str().unwrap_or("unknown").to_string(),
            version: Self::detect_version(&json, &architecture),

            hidden_size: json["hidden_size"].as_u64().unwrap_or(4096) as usize,
            num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: json["num_key_value_heads"]
                .as_u64()
                .or_else(|| json["num_attention_heads"].as_u64())
                .unwrap_or(32) as usize,
            head_dim: json["head_dim"].as_u64().unwrap_or(128) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(11008) as usize,

            vocab_size: json["vocab_size"].as_u64().unwrap_or(32000) as usize,
            max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(4096)
                as usize,
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            rope_scaling: Self::parse_rope_scaling(&json),

            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            layer_norm_eps: json["layer_norm_eps"].as_f64().map(|v| v as f32),

            hidden_activation: json["hidden_activation"]
                .as_str()
                .unwrap_or("silu")
                .to_string(),

            use_qk_norm: json["use_qk_norm"].as_bool().unwrap_or(false),
            scale_embeddings: json["scale_embeddings"].as_bool().unwrap_or(false),
            query_pre_attn_scalar: json["query_pre_attn_scalar"].as_f64().map(|v| v as f32),

            dtype: "bfloat16".to_string(),
            use_flash_attention: true,
            use_kv_cache: true,
        };

        Ok(config)
    }

    /// Detect configuration from weights only (fallback)
    fn detect_from_weights(weights: &HashMap<String, Tensor>) -> Result<Self> {
        // Start with defaults
        let mut config = Self::default();

        // Detect architecture from weight names
        config.architecture = Self::detect_architecture_from_weights(weights);

        // Extract dimensions from embeddings
        if let Some(embed) = weights
            .get("model.embed_tokens.weight")
            .or_else(|| weights.get("embed_tokens.weight"))
        {
            let shape = embed.size();
            config.vocab_size = shape[0] as usize;
            config.hidden_size = shape[1] as usize;
        }

        // Count layers
        config.num_hidden_layers = Self::count_layers(weights);

        // Detect attention configuration from projections
        Self::detect_attention_config(weights, &mut config)?;

        // Apply architecture-specific defaults
        config.apply_architecture_defaults();

        Ok(config)
    }

    fn detect_architecture_from_json(json: &serde_json::Value) -> ModelArchitecture {
        // Check model_type field
        if let Some(model_type) = json["model_type"].as_str() {
            return match model_type.to_lowercase().as_str() {
                "llama" => ModelArchitecture::Llama,
                "qwen" | "qwen2" | "qwen3" => ModelArchitecture::Qwen,
                "gemma" => ModelArchitecture::Gemma,
                "mistral" => ModelArchitecture::Mistral,
                _ => ModelArchitecture::Unknown(model_type.to_string()),
            };
        }

        // Check architectures field
        if let Some(architectures) = json["architectures"].as_array() {
            if let Some(first) = architectures.first() {
                if let Some(arch_str) = first.as_str() {
                    return match arch_str.to_lowercase().as_str() {
                        s if s.contains("llama") => ModelArchitecture::Llama,
                        s if s.contains("qwen") => ModelArchitecture::Qwen,
                        s if s.contains("gemma") => ModelArchitecture::Gemma,
                        s if s.contains("mistral") => ModelArchitecture::Mistral,
                        _ => ModelArchitecture::Unknown(arch_str.to_string()),
                    };
                }
            }
        }

        ModelArchitecture::Unknown("unknown".to_string())
    }

    fn detect_architecture_from_weights(weights: &HashMap<String, Tensor>) -> ModelArchitecture {
        // Check for architecture-specific weight patterns
        for key in weights.keys() {
            if key.contains("q_norm") || key.contains("k_norm") {
                return ModelArchitecture::Qwen;
            }
            if key.contains("gated_gemm") {
                return ModelArchitecture::Gemma;
            }
        }

        // Default to Llama as most common
        ModelArchitecture::Llama
    }

    fn detect_version(json: &serde_json::Value, architecture: &ModelArchitecture) -> u32 {
        // Try to extract version from model_type field (e.g., "qwen2", "llama3")
        if let Some(model_type) = json["model_type"].as_str() {
            let lower = model_type.to_lowercase();

            // Look for explicit version numbers in model_type
            if lower.contains("qwen3") || lower.contains("qwen-3") {
                return 3;
            } else if lower.contains("qwen2") || lower.contains("qwen-2") {
                return 2;
            } else if lower.contains("llama3") || lower.contains("llama-3") {
                return 3;
            } else if lower.contains("llama2") || lower.contains("llama-2") {
                return 2;
            }

            // Try generic number extraction as fallback
            if let Some(captures) = regex::Regex::new(r"(\d+)").unwrap().captures(model_type) {
                if let Ok(version) = captures[1].parse::<u32>() {
                    return version;
                }
            }
        }

        // Try architectures field (e.g., ["Qwen3ForCausalLM"])
        if let Some(architectures) = json["architectures"].as_array() {
            if let Some(first_arch) = architectures.first().and_then(|v| v.as_str()) {
                let lower = first_arch.to_lowercase();

                if lower.contains("qwen3") {
                    return 3;
                } else if lower.contains("qwen2") {
                    return 2;
                } else if lower.contains("llama3") {
                    return 3;
                } else if lower.contains("llama2") {
                    return 2;
                }
            }
        }

        // Architecture-specific defaults (conservative)
        // Note: DO NOT use rope_theta or other config values to detect version
        // as these are not reliable indicators
        match architecture {
            ModelArchitecture::Qwen => 2,  // Default to Qwen2 if unknown
            ModelArchitecture::Llama => 2, // Default to Llama2 if unknown
            _ => 1,
        }
    }

    fn parse_rope_scaling(json: &serde_json::Value) -> Option<RopeScaling> {
        json["rope_scaling"].as_object().map(|obj| RopeScaling {
            rope_type: obj["type"].as_str().unwrap_or("linear").to_string(),
            factor: obj["factor"].as_f64().unwrap_or(1.0) as f32,
        })
    }

    fn count_layers(weights: &HashMap<String, Tensor>) -> usize {
        let mut max_layer = 0;
        for key in weights.keys() {
            if let Some(captures) = regex::Regex::new(r"layers\.(\d+)").unwrap().captures(key) {
                if let Ok(layer_idx) = captures[1].parse::<usize>() {
                    max_layer = max_layer.max(layer_idx + 1);
                }
            }
        }
        max_layer
    }

    fn detect_attention_config(weights: &HashMap<String, Tensor>, config: &mut Self) -> Result<()> {
        // Find Q and K projection shapes
        let q_proj = weights
            .keys()
            .find(|k| k.contains("q_proj.weight"))
            .and_then(|k| weights.get(k));
        let k_proj = weights
            .keys()
            .find(|k| k.contains("k_proj.weight"))
            .and_then(|k| weights.get(k));

        if let (Some(q), Some(k)) = (q_proj, k_proj) {
            let q_out = q.size()[0] as usize;
            let k_out = k.size()[0] as usize;

            // Try different head dimensions
            for head_dim in &[256, 128, 64, 32] {
                if q_out.is_multiple_of(*head_dim) && k_out.is_multiple_of(*head_dim) {
                    config.num_attention_heads = q_out / head_dim;
                    config.num_key_value_heads = k_out / head_dim;
                    config.head_dim = *head_dim;
                    break;
                }
            }
        }

        Ok(())
    }

    fn apply_architecture_defaults(&mut self) {
        match &self.architecture {
            ModelArchitecture::Qwen => {
                // Qwen3 specific defaults
                if self.vocab_size == 151936 {
                    self.rope_theta = 5_000_000.0; // Qwen3-4B uses 5M
                    self.version = 3;
                }
            }
            ModelArchitecture::Gemma => {
                if self.vocab_size == 262144 {
                    self.rope_theta = 1_000_000.0;
                    self.use_qk_norm = true;
                    self.scale_embeddings = true;
                    self.query_pre_attn_scalar = Some(256.0);
                }
            }
            _ => {}
        }
    }

    /// Validate configuration against actual weights
    fn validate_with_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // Check if detected values match weights
        if let Some(embed) = weights.get("model.embed_tokens.weight") {
            let actual_vocab = embed.size()[0] as usize;
            let actual_hidden = embed.size()[1] as usize;

            if self.vocab_size != actual_vocab {
                info!(
                    "⚠️ Config vocab_size {} doesn't match weights {}, using weights",
                    self.vocab_size, actual_vocab
                );
                self.vocab_size = actual_vocab;
            }

            if self.hidden_size != actual_hidden {
                info!(
                    "⚠️ Config hidden_size {} doesn't match weights {}, using weights",
                    self.hidden_size, actual_hidden
                );
                self.hidden_size = actual_hidden;
            }
        }

        Ok(())
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            model_type: "llama".to_string(),
            version: 2,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rms_norm_eps: 1e-5,
            layer_norm_eps: None,
            hidden_activation: "silu".to_string(),
            use_qk_norm: false,
            scale_embeddings: false,
            query_pre_attn_scalar: None,
            dtype: "bfloat16".to_string(),
            use_flash_attention: true,
            use_kv_cache: true,
        }
    }
}
