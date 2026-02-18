//! Unified model configuration management
//!
//! This module provides a single source of truth for model configurations,
//! resolving the current chaos of multiple override points.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;
use tch::Tensor;
use tracing::info;

/// Regex for extracting numeric version from model type strings
/// SAFETY: These patterns are compile-time constants that are guaranteed valid
static VERSION_REGEX: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"(\d+)").ok()
});

/// Regex for extracting layer indices from weight key names
static LAYER_REGEX: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"layers\.(\d+)").ok()
});

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
    Janus,
    Unknown(String),
}

/// Configuration source for different model architectures
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// Flat config (Llama, Qwen, Gemma, etc.)
    Flat(Box<ModelConfig>),

    /// Nested config (Janus, future multimodal models)
    Nested(Box<NestedModelConfig>),
}

/// Nested configuration for multimodal models like Janus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedModelConfig {
    /// Top-level architecture
    pub architecture: ModelArchitecture,

    /// Component configs (raw JSON for flexibility)
    pub language_config: Option<serde_json::Value>,
    pub vision_config: Option<serde_json::Value>,
    pub aligner_config: Option<serde_json::Value>,

    // Optional generation components
    pub gen_aligner_config: Option<serde_json::Value>,
    pub gen_vision_config: Option<serde_json::Value>,
    pub gen_head_config: Option<serde_json::Value>,

    /// Top-level metadata (not in sub-configs)
    pub num_hidden_layers: Option<usize>,
    pub image_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub begin_image_token_id: Option<u32>,
    pub torch_dtype: Option<String>,
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

        // For Janus models, extract language_config and use that as the source
        let lang_json_opt = if architecture == ModelArchitecture::Janus {
            info!("Detected Janus multimodal model - loading from language_config");

            let lang_config = json["language_config"].as_object()
                .ok_or_else(|| anyhow::anyhow!("Janus config missing required 'language_config'"))?;

            // Convert to serde_json::Value for uniform access
            let lang_json = serde_json::Value::Object(lang_config.clone());

            info!("Language config hidden_size: {}", lang_json["hidden_size"].as_u64().unwrap_or(0));

            Some(lang_json)
        } else {
            None
        };

        // Choose config source based on architecture
        let config_source = lang_json_opt.as_ref().unwrap_or(&json);

        // Extract all configuration values from the appropriate source
        let config = Self {
            architecture: architecture.clone(),
            model_type: json["model_type"].as_str().unwrap_or("unknown").to_owned(),
            version: Self::detect_version(&json, &architecture),

            hidden_size: config_source["hidden_size"].as_u64().unwrap_or(4096) as usize,
            num_hidden_layers: config_source["num_hidden_layers"].as_u64()
                .or_else(|| json["num_hidden_layers"].as_u64())
                .unwrap_or(32) as usize,
            num_attention_heads: config_source["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: config_source["num_key_value_heads"]
                .as_u64()
                .or_else(|| config_source["num_attention_heads"].as_u64())
                .unwrap_or(32) as usize,
            head_dim: config_source["head_dim"].as_u64()
                .or_else(|| {
                    // Calculate from hidden_size / num_attention_heads
                    let hidden = config_source["hidden_size"].as_u64()?;
                    let heads = config_source["num_attention_heads"].as_u64()?;
                    Some(hidden / heads)
                })
                .unwrap_or(128) as usize,
            intermediate_size: config_source["intermediate_size"].as_u64().unwrap_or(11008) as usize,

            vocab_size: config_source["vocab_size"].as_u64().unwrap_or(32000) as usize,
            max_position_embeddings: config_source["max_position_embeddings"].as_u64().unwrap_or(4096)
                as usize,
            rope_theta: config_source["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            rope_scaling: Self::parse_rope_scaling(config_source),

            rms_norm_eps: config_source["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            layer_norm_eps: config_source["layer_norm_eps"].as_f64().map(|v| v as f32),

            hidden_activation: config_source["hidden_activation"]
                .as_str()
                .unwrap_or("silu").to_owned(),

            use_qk_norm: config_source["use_qk_norm"].as_bool().unwrap_or(false),
            scale_embeddings: config_source["scale_embeddings"].as_bool().unwrap_or(false),
            query_pre_attn_scalar: config_source["query_pre_attn_scalar"].as_f64().map(|v| v as f32),

            dtype: "bfloat16".to_owned(),
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

    /// Detect the model architecture from a model directory's `config.json`.
    ///
    /// Returns `ModelArchitecture::Unknown` if the file is missing or unparseable.
    /// This is a lightweight operation (reads only config.json, not weights).
    pub fn detect_architecture(model_path: &Path) -> ModelArchitecture {
        let config_path = model_path.join("config.json");
        let content = match std::fs::read_to_string(&config_path) {
            Ok(c) => c,
            Err(_) => return ModelArchitecture::Unknown("unknown".to_owned()),
        };
        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return ModelArchitecture::Unknown("unknown".to_owned()),
        };
        Self::detect_architecture_from_json(&json)
    }

    fn detect_architecture_from_json(json: &serde_json::Value) -> ModelArchitecture {
        // Check model_type field
        if let Some(model_type) = json["model_type"].as_str() {
            return match model_type.to_lowercase().as_str() {
                "janus" => ModelArchitecture::Janus,  // Explicit Janus type
                "llama" => ModelArchitecture::Llama,
                "qwen" | "qwen2" | "qwen3" => ModelArchitecture::Qwen,
                "gemma" => ModelArchitecture::Gemma,
                "mistral" => ModelArchitecture::Mistral,
                _ => ModelArchitecture::Unknown(model_type.to_owned()),
            };
        }

        // Check architectures field
        if let Some(architectures) = json["architectures"].as_array() {
            if let Some(first) = architectures.first() {
                if let Some(arch_str) = first.as_str() {
                    return match arch_str.to_lowercase().as_str() {
                        s if s.contains("janus") => ModelArchitecture::Janus,  // Janus in architecture name
                        s if s.contains("llama") => ModelArchitecture::Llama,
                        s if s.contains("qwen") => ModelArchitecture::Qwen,
                        s if s.contains("gemma") => ModelArchitecture::Gemma,
                        s if s.contains("mistral") => ModelArchitecture::Mistral,
                        _ => ModelArchitecture::Unknown(arch_str.to_owned()),
                    };
                }
            }
        }

        ModelArchitecture::Unknown("unknown".to_owned())
    }

    fn detect_architecture_from_weights(weights: &HashMap<String, Tensor>) -> ModelArchitecture {
        // Check for architecture-specific weight patterns

        // Check for Janus multimodal components first
        let has_vision_model = weights.keys().any(|k| k.starts_with("vision_model.") || k.starts_with("vision_encoder."));
        let has_aligner = weights.keys().any(|k| k.starts_with("aligner.") || k.starts_with("vision_aligner."));
        let has_language_model = weights.keys().any(|k| k.starts_with("language_model."));

        if has_vision_model || has_aligner || has_language_model {
            info!("Detected Janus multimodal model from weight patterns");
            info!("  has_vision_model: {}", has_vision_model);
            info!("  has_aligner: {}", has_aligner);
            info!("  has_language_model: {}", has_language_model);
            return ModelArchitecture::Janus;
        }

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
            if let Some(ref regex) = *VERSION_REGEX {
                if let Some(captures) = regex.captures(model_type) {
                    if let Ok(version) = captures[1].parse::<u32>() {
                        return version;
                    }
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
            // Default to version 2 for Qwen/Llama if unknown
            ModelArchitecture::Qwen | ModelArchitecture::Llama => 2,
            _ => 1,
        }
    }

    fn parse_rope_scaling(json: &serde_json::Value) -> Option<RopeScaling> {
        json["rope_scaling"].as_object().map(|obj| RopeScaling {
            rope_type: obj["type"].as_str().unwrap_or("linear").to_owned(),
            factor: obj["factor"].as_f64().unwrap_or(1.0) as f32,
        })
    }

    fn count_layers(weights: &HashMap<String, Tensor>) -> usize {
        let mut max_layer = 0;
        if let Some(ref regex) = *LAYER_REGEX {
            for key in weights.keys() {
                if let Some(captures) = regex.captures(key) {
                    if let Ok(layer_idx) = captures[1].parse::<usize>() {
                        max_layer = max_layer.max(layer_idx + 1);
                    }
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
            model_type: "llama".to_owned(),
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
            hidden_activation: "silu".to_owned(),
            use_qk_norm: false,
            scale_embeddings: false,
            query_pre_attn_scalar: None,
            dtype: "bfloat16".to_owned(),
            use_flash_attention: true,
            use_kv_cache: true,
        }
    }
}

// =============================================================================
// Training Configuration Load/Save (Phase D)
// =============================================================================

impl ModelConfig {
    /// Load hyprstream training config from the model's config.json
    ///
    /// Extracts the `hyprstream_training` section if present, otherwise returns None.
    pub fn load_training_config(
        model_path: &Path,
    ) -> Option<crate::config::HyprstreamTrainingConfig> {
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&config_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Extract hyprstream_training section if present
        json.get("hyprstream_training")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Save hyprstream training config to the model's config.json
    ///
    /// Preserves all other fields in config.json, only updates/inserts `hyprstream_training`.
    pub fn save_training_config(
        model_path: &Path,
        training_config: &crate::config::HyprstreamTrainingConfig,
    ) -> Result<()> {
        let config_path = model_path.join("config.json");

        // Read existing config
        let content = std::fs::read_to_string(&config_path)?;
        let mut json: serde_json::Value = serde_json::from_str(&content)?;

        // Update or insert hyprstream_training section
        json["hyprstream_training"] = serde_json::to_value(training_config)?;

        // Write back with pretty formatting
        let output = serde_json::to_string_pretty(&json)?;
        std::fs::write(&config_path, output)?;

        info!(
            "✅ Training config saved to {}: mode={:?}, target_adapter={:?}",
            config_path.display(),
            training_config.mode,
            training_config.target_adapter
        );

        Ok(())
    }
}
