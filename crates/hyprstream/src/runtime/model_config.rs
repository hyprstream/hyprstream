//! Unified model configuration management
//!
//! This module provides a single source of truth for model configurations,
//! resolving the current chaos of multiple override points.

use anyhow::{bail, Context, Result};
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

    // Qwen3.5 hybrid SSM/attention fields
    pub partial_rotary_factor: Option<f32>,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,

    // Qwen3.5 MoE fields
    pub is_moe: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    // Vision fields (Qwen3.5 multimodal)
    pub has_vision: bool,
    pub vision_out_hidden_size: usize,
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
    /// Qwen3.5 hybrid GatedDeltaNet/full-attention model (dense and MoE variants)
    Qwen3_5,
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

/// Opt-in escape hatch: when set to a truthy value (`1`/`true`), a missing
/// `config.json` falls back to the fragile weight-name-scan heuristic instead of
/// hard-failing. The model's own `config.json` is authoritative; this exists only
/// for legacy/dev checkpoints that ship raw weights with no metadata.
const ALLOW_WEIGHT_DETECT_ENV: &str = "HYPRSTREAM_ALLOW_WEIGHT_DETECT";

/// Returns true when an env var is set to a recognized truthy value.
fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Extract a mandatory unsigned integer field from a config JSON object, failing
/// fast with a clear error (including the config path) when it is absent or not a
/// non-negative integer. Used for the architectural dimensions that must never be
/// silently defaulted (#315).
fn require_u64(source: &serde_json::Value, field: &str, config_path: &Path) -> Result<u64> {
    source[field].as_u64().with_context(|| {
        format!(
            "config.json ({path}) is missing required field `{field}` (or it is not a non-negative integer)",
            path = config_path.display()
        )
    })
}

impl ModelConfig {
    /// Load configuration, treating the model's own `config.json` as authoritative.
    ///
    /// No-fragile-fallbacks (#315): `config.json` is **required** by default. If it
    /// is absent, loading is a hard error — a silently-guessed config can corrupt a
    /// pipeline split (e.g. a wrong layer count). The legacy weight-name-scan
    /// heuristic remains available only behind the explicit
    /// `HYPRSTREAM_ALLOW_WEIGHT_DETECT=1` opt-in.
    ///
    /// When `model.safetensors.index.json` is present, `num_hidden_layers` is
    /// cross-checked against the shard manifest's `weight_map`; a mismatch is a
    /// hard error rather than a silent corruption.
    pub fn load(model_path: &Path, weights: &HashMap<String, Tensor>) -> Result<Self> {
        // Step 1: config.json is authoritative and required.
        let config_path = model_path.join("config.json");
        let mut config = if config_path.exists() {
            info!("Loading model configuration");
            Self::from_json_file(&config_path)?
        } else if env_flag(ALLOW_WEIGHT_DETECT_ENV) {
            info!(
                "⚠️ No config.json in {}; {ALLOW_WEIGHT_DETECT_ENV} is set, detecting from weights (fragile)",
                model_path.display()
            );
            Self::detect_from_weights(weights)?
        } else {
            bail!(
                "config.json is required but was not found in {}. \
                 A model's config.json is authoritative for its architecture; refusing to guess. \
                 Set {ALLOW_WEIGHT_DETECT_ENV}=1 to opt into the legacy weight-name heuristic.",
                model_path.display()
            );
        };

        // Step 1b: cross-check num_hidden_layers against the shard manifest.
        config.validate_against_index(model_path)?;

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

        // For Janus models, extract language_config and use that as the source.
        // For Qwen3.5 models, extract text_config.
        let nested_json_opt = if architecture == ModelArchitecture::Janus {
            info!("Detected Janus multimodal model - loading from language_config");

            let lang_config = json["language_config"].as_object()
                .ok_or_else(|| anyhow::anyhow!("Janus config missing required 'language_config'"))?;

            let lang_json = serde_json::Value::Object(lang_config.clone());
            info!("Language config hidden_size: {}", lang_json["hidden_size"].as_u64().unwrap_or(0));
            Some(lang_json)
        } else if architecture == ModelArchitecture::Qwen3_5 {
            info!("Detected Qwen3.5 model - loading from text_config");
            if let Some(text_cfg) = json["text_config"].as_object() {
                let text_json = serde_json::Value::Object(text_cfg.clone());
                info!("text_config hidden_size: {}", text_json["hidden_size"].as_u64().unwrap_or(0));
                Some(text_json)
            } else {
                None // flat config (text-only checkpoint)
            }
        } else {
            None
        };

        // Choose config source based on architecture
        let config_source = nested_json_opt.as_ref().unwrap_or(&json);

        // No-fragile-fallbacks (#315): the core architectural dimensions are
        // *mandatory* when config.json is present. A wrong layer count or hidden
        // size silently corrupts inference (and a pipeline split), so a
        // missing/unparseable value is a hard error rather than a magic default.
        // `num_hidden_layers` falls back to the top-level config for nested
        // (Janus/Qwen3.5) layouts that carry it outside the sub-config.
        let num_hidden_layers = config_source["num_hidden_layers"]
            .as_u64()
            .or_else(|| json["num_hidden_layers"].as_u64())
            .map(|v| v as usize)
            .with_context(|| {
                format!(
                    "config.json ({path}) is missing required field `num_hidden_layers`",
                    path = path.display()
                )
            })?;
        let hidden_size = require_u64(config_source, "hidden_size", path)? as usize;
        let num_attention_heads = require_u64(config_source, "num_attention_heads", path)? as usize;
        if num_attention_heads == 0 || num_hidden_layers == 0 || hidden_size == 0 {
            bail!(
                "config.json ({}) has invalid zero dimension(s): \
                 hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}, \
                 num_attention_heads={num_attention_heads}",
                path.display()
            );
        }

        // Extract all configuration values from the appropriate source
        let config = Self {
            architecture: architecture.clone(),
            model_type: json["model_type"].as_str().unwrap_or("unknown").to_owned(),
            version: Self::detect_version(&json, &architecture),

            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            // KV heads legitimately default to attention heads (MHA, not GQA).
            num_key_value_heads: config_source["num_key_value_heads"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(num_attention_heads),
            // head_dim is derivable from hidden_size / heads when omitted.
            head_dim: config_source["head_dim"].as_u64()
                .map(|v| v as usize)
                .unwrap_or(hidden_size / num_attention_heads),
            intermediate_size: config_source["intermediate_size"].as_u64().unwrap_or(11008) as usize,

            vocab_size: config_source["vocab_size"].as_u64().unwrap_or(32000) as usize,
            max_position_embeddings: config_source["max_position_embeddings"].as_u64().unwrap_or(4096)
                as usize,
            rope_theta: config_source["rope_parameters"]["rope_theta"].as_f64()
                .or_else(|| config_source["rope_theta"].as_f64())
                .unwrap_or(10_000.0) as f32,
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

            // Qwen3.5 hybrid SSM/attention fields
            partial_rotary_factor: if architecture == ModelArchitecture::Qwen3_5 {
                Some(
                    config_source["rope_parameters"]["partial_rotary_factor"].as_f64()
                        .or_else(|| config_source["partial_rotary_factor"].as_f64())
                        .unwrap_or(0.25) as f32,
                )
            } else {
                None
            },
            layer_types: if architecture == ModelArchitecture::Qwen3_5 {
                let num_layers = num_hidden_layers;
                config_source["layer_types"].as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_owned))
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        let interval = config_source["full_attention_interval"]
                            .as_u64().unwrap_or(4) as usize;
                        (0..num_layers)
                            .map(|i| {
                                if (i + 1) % interval == 0 { "full_attention".to_owned() }
                                else { "linear_attention".to_owned() }
                            })
                            .collect()
                    })
            } else {
                vec![]
            },
            linear_conv_kernel_dim: config_source["linear_conv_kernel_dim"].as_u64().unwrap_or(0) as usize,
            linear_key_head_dim: config_source["linear_key_head_dim"].as_u64().unwrap_or(0) as usize,
            linear_value_head_dim: config_source["linear_value_head_dim"].as_u64().unwrap_or(0) as usize,
            linear_num_key_heads: config_source["linear_num_key_heads"].as_u64().unwrap_or(0) as usize,
            linear_num_value_heads: config_source["linear_num_value_heads"].as_u64().unwrap_or(0) as usize,

            // Qwen3.5 MoE fields
            is_moe: json["model_type"].as_str().map(|s| s.contains("moe")).unwrap_or(false),
            num_experts: config_source["num_experts"].as_u64().unwrap_or(0) as usize,
            num_experts_per_tok: config_source["num_experts_per_tok"].as_u64().unwrap_or(0) as usize,
            moe_intermediate_size: config_source["moe_intermediate_size"].as_u64().unwrap_or(0) as usize,
            shared_expert_intermediate_size: config_source["shared_expert_intermediate_size"]
                .as_u64().unwrap_or(0) as usize,

            // Vision fields
            has_vision: json["vision_config"].is_object(),
            vision_out_hidden_size: json["vision_config"]["out_hidden_size"]
                .as_u64().unwrap_or(3584) as usize,
        };

        Ok(config)
    }

    /// Detect configuration from weights only (fallback)
    #[allow(clippy::field_reassign_with_default)]
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
                "janus" => ModelArchitecture::Janus,
                "llama" => ModelArchitecture::Llama,
                "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" => ModelArchitecture::Qwen3_5,
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
                        s if s.contains("janus") => ModelArchitecture::Janus,
                        s if s.contains("llama") => ModelArchitecture::Llama,
                        s if s.contains("qwen3_5") || s.contains("Qwen3_5") => ModelArchitecture::Qwen3_5,
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

        // Qwen3.5 has linear_attn layers (GatedDeltaNet)
        if weights.keys().any(|k| k.contains("linear_attn.")) {
            return ModelArchitecture::Qwen3_5;
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
            ModelArchitecture::Qwen3_5 => {
                // Qwen3.5 defaults — nested config parsing handles most fields
                if self.rope_theta == 10_000.0 {
                    self.rope_theta = 10_000_000.0; // Qwen3.5 uses 10M
                }
            }
            ModelArchitecture::Gemma if self.vocab_size == 262144 => {
                self.rope_theta = 1_000_000.0;
                self.use_qk_norm = true;
                self.scale_embeddings = true;
                self.query_pre_attn_scalar = Some(256.0);
            }
            _ => {}
        }
    }

    /// Cross-check `num_hidden_layers` against the shard manifest's `weight_map`.
    ///
    /// When `model.safetensors.index.json` is present, the number of distinct
    /// `model.layers.<i>.` (or `*.layers.<i>.` for nested layouts) indices listed
    /// in its `weight_map` must equal the config's `num_hidden_layers`. A mismatch
    /// means the config and the actual checkpoint disagree about depth, which would
    /// silently truncate or over-allocate a pipeline split — so it is a hard error.
    ///
    /// If the index file is absent this is a no-op: per #315 the index is required
    /// only where the *loader* consumes it (multi-shard models, validated in
    /// `model_factory`); a single-file model legitimately has no manifest.
    fn validate_against_index(&self, model_path: &Path) -> Result<()> {
        let index_path = model_path.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&index_path).with_context(|| {
            format!("failed to read shard manifest {}", index_path.display())
        })?;
        let index: serde_json::Value = serde_json::from_str(&content).with_context(|| {
            format!("failed to parse shard manifest {}", index_path.display())
        })?;
        let weight_map = index["weight_map"].as_object().ok_or_else(|| {
            anyhow::anyhow!(
                "shard manifest {} is missing required `weight_map` object",
                index_path.display()
            )
        })?;

        let mut layer_indices = std::collections::HashSet::new();
        if let Some(ref regex) = *LAYER_REGEX {
            for key in weight_map.keys() {
                if let Some(captures) = regex.captures(key) {
                    if let Ok(idx) = captures[1].parse::<usize>() {
                        layer_indices.insert(idx);
                    }
                }
            }
        }

        // Only enforce when the manifest actually encodes per-layer weights.
        // (Some manifests for non-transformer components may carry none.)
        if layer_indices.is_empty() {
            return Ok(());
        }

        let manifest_layers = layer_indices.len();
        if manifest_layers != self.num_hidden_layers {
            bail!(
                "num_hidden_layers mismatch: config.json declares {} but \
                 {} lists {} distinct transformer layers in its weight_map. \
                 Refusing to load with a layer count that disagrees with the checkpoint.",
                self.num_hidden_layers,
                index_path.display(),
                manifest_layers
            );
        }

        info!(
            "✅ num_hidden_layers ({}) matches shard manifest weight_map",
            self.num_hidden_layers
        );
        Ok(())
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

            partial_rotary_factor: None,
            layer_types: vec![],
            linear_conv_kernel_dim: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,

            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,

            has_vision: false,
            vision_out_hidden_size: 0,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tch::Tensor;

    fn empty_weights() -> HashMap<String, Tensor> {
        HashMap::new()
    }

    fn write(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).unwrap();
    }

    /// A minimal-but-valid Llama-style config.json.
    fn valid_config_json(num_layers: usize) -> String {
        format!(
            r#"{{
                "model_type": "llama",
                "hidden_size": 4096,
                "num_hidden_layers": {num_layers},
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5
            }}"#
        )
    }

    /// Happy path: a model that ships a proper config.json loads with the
    /// declared dimensions (no magic-number substitution).
    #[test]
    fn valid_config_loads_with_declared_dims() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "config.json", &valid_config_json(40));

        let cfg = ModelConfig::load(dir.path(), &empty_weights()).expect("valid config must load");
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        // head_dim derived from hidden_size / heads.
        assert_eq!(cfg.head_dim, 128);
        // KV heads honored (GQA), not silently set to attention-head count.
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    /// #315: a missing config.json is a hard error by default (no silent guess).
    #[test]
    fn missing_config_is_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = ModelConfig::load(dir.path(), &empty_weights())
            .expect_err("missing config.json must fail");
        assert!(
            err.to_string().contains("config.json is required"),
            "unexpected error: {err}"
        );
    }

    /// #315: the silent magic-number default for num_hidden_layers is gone — a
    /// config.json present but missing the field is a hard error (not 32).
    #[test]
    fn missing_num_hidden_layers_is_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "config.json",
            r#"{"model_type":"llama","hidden_size":4096,"num_attention_heads":32}"#,
        );
        let err = ModelConfig::load(dir.path(), &empty_weights())
            .expect_err("missing num_hidden_layers must fail");
        assert!(
            err.to_string().contains("num_hidden_layers"),
            "unexpected error: {err}"
        );
    }

    /// #315: the silent magic-number default for hidden_size is gone.
    #[test]
    fn missing_hidden_size_is_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "config.json",
            r#"{"model_type":"llama","num_hidden_layers":32,"num_attention_heads":32}"#,
        );
        let err = ModelConfig::load(dir.path(), &empty_weights())
            .expect_err("missing hidden_size must fail");
        assert!(err.to_string().contains("hidden_size"), "unexpected error: {err}");
    }

    /// #315: num_hidden_layers must match the shard manifest's weight_map.
    #[test]
    fn index_layer_count_mismatch_is_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        // config says 4 layers...
        write(dir.path(), "config.json", &valid_config_json(4));
        // ...but the manifest only has weights for layers 0,1,2 (3 layers).
        write(
            dir.path(),
            "model.safetensors.index.json",
            r#"{"weight_map":{
                "model.layers.0.self_attn.q_proj.weight":"model-00001-of-00002.safetensors",
                "model.layers.1.self_attn.q_proj.weight":"model-00001-of-00002.safetensors",
                "model.layers.2.self_attn.q_proj.weight":"model-00002-of-00002.safetensors",
                "model.embed_tokens.weight":"model-00001-of-00002.safetensors"
            }}"#,
        );
        let err = ModelConfig::load(dir.path(), &empty_weights())
            .expect_err("layer-count mismatch must fail");
        assert!(
            err.to_string().contains("num_hidden_layers mismatch"),
            "unexpected error: {err}"
        );
    }

    /// Happy path: config and manifest agree on layer count → loads cleanly.
    #[test]
    fn index_layer_count_match_loads() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "config.json", &valid_config_json(3));
        write(
            dir.path(),
            "model.safetensors.index.json",
            r#"{"weight_map":{
                "model.layers.0.self_attn.q_proj.weight":"model-00001-of-00002.safetensors",
                "model.layers.1.self_attn.q_proj.weight":"model-00001-of-00002.safetensors",
                "model.layers.2.self_attn.q_proj.weight":"model-00002-of-00002.safetensors"
            }}"#,
        );
        let cfg = ModelConfig::load(dir.path(), &empty_weights())
            .expect("matching layer counts must load");
        assert_eq!(cfg.num_hidden_layers, 3);
    }

    /// A zero architectural dimension is rejected (avoids div-by-zero / corruption).
    #[test]
    fn zero_dimension_is_hard_error() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "config.json",
            r#"{"model_type":"llama","hidden_size":4096,"num_hidden_layers":0,"num_attention_heads":32}"#,
        );
        let err =
            ModelConfig::load(dir.path(), &empty_weights()).expect_err("zero dimension must fail");
        assert!(err.to_string().contains("zero dimension"), "unexpected error: {err}");
    }
}
