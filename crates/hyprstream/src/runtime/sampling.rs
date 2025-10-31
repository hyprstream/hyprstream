//! Token sampling strategies for language model generation
//!
//! Implements various sampling methods including temperature, top-k, and top-p (nucleus) sampling
//! with model-specific configurations loaded from HuggingFace model cards.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Sampling configuration for text generation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Temperature for sampling (higher = more random)
    pub temperature: f32,

    /// Top-k sampling: only sample from top k tokens
    pub top_k: Option<usize>,

    /// Top-p (nucleus) sampling: sample from tokens with cumulative probability <= p
    pub top_p: Option<f32>,

    /// Repetition penalty (1.0 = no penalty)
    pub repeat_penalty: f32,

    /// Length penalty (1.0 = no penalty)
    pub length_penalty: f32,

    /// Seed for reproducible sampling
    pub seed: Option<u64>,

    /// Whether to use sampling (false = greedy/argmax)
    pub do_sample: bool,

    /// Typical p sampling threshold
    pub typical_p: Option<f32>,

    /// Epsilon cutoff for truncation sampling
    pub epsilon_cutoff: Option<f32>,

    /// Eta cutoff for truncation sampling
    pub eta_cutoff: Option<f32>,

    /// Stop sequences (generation stops if any of these appear)
    #[serde(default)]
    pub stop_tokens: Vec<String>,

    /// Maximum tokens to generate (0 = no limit)
    #[serde(default)]
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.95),
            repeat_penalty: 1.0,
            length_penalty: 1.0,
            seed: None,
            do_sample: true,
            typical_p: None,
            epsilon_cutoff: None,
            eta_cutoff: None,
            stop_tokens: vec![],
            max_tokens: 2048,
        }
    }
}

impl SamplingConfig {
    /// Create sampling config for a specific model based on its architecture
    pub fn from_model_card(model_id: &str, config_json: &serde_json::Value) -> Self {
        // Extract generation config if present
        if let Some(gen_config) = config_json.get("generation_config") {
            return Self::from_generation_config(gen_config);
        }

        // Otherwise use model-specific defaults
        Self::for_model(model_id)
    }

    /// Parse HuggingFace generation_config
    fn from_generation_config(config: &serde_json::Value) -> Self {
        Self {
            temperature: config
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0),

            top_k: config
                .get("top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),

            top_p: config
                .get("top_p")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),

            repeat_penalty: config
                .get("repetition_penalty")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0),

            length_penalty: config
                .get("length_penalty")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0),

            do_sample: config
                .get("do_sample")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),

            typical_p: config
                .get("typical_p")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),

            epsilon_cutoff: config
                .get("epsilon_cutoff")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),

            eta_cutoff: config
                .get("eta_cutoff")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),

            seed: None,

            stop_tokens: vec![],
            max_tokens: config
                .get("max_length")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(2048),
        }
    }

    /// Get model-specific default configurations
    pub fn for_model(model_id: &str) -> Self {
        let model_lower = model_id.to_lowercase();

        if model_lower.contains("qwen") {
            Self::qwen_defaults()
        } else if model_lower.contains("llama") {
            Self::llama_defaults()
        } else if model_lower.contains("mistral") {
            Self::mistral_defaults()
        } else if model_lower.contains("gemma") {
            Self::gemma_defaults()
        } else if model_lower.contains("phi") {
            Self::phi_defaults()
        } else if model_lower.contains("gpt") {
            Self::gpt_defaults()
        } else {
            Self::default()
        }
    }

    /// Qwen model defaults
    fn qwen_defaults() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(20),
            top_p: Some(0.8),
            repeat_penalty: 1.05,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Llama model defaults
    fn llama_defaults() -> Self {
        Self {
            temperature: 0.6,
            top_k: Some(40),
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Mistral model defaults
    fn mistral_defaults() -> Self {
        Self {
            temperature: 0.7,
            top_k: None, // Mistral typically doesn't use top-k
            top_p: Some(0.95),
            repeat_penalty: 1.0,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Gemma model defaults
    fn gemma_defaults() -> Self {
        Self {
            temperature: 0.8,
            top_k: Some(40),
            top_p: Some(0.95),
            repeat_penalty: 1.0,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Phi model defaults
    fn phi_defaults() -> Self {
        Self {
            temperature: 0.75,
            top_k: Some(50),
            top_p: Some(0.95),
            repeat_penalty: 1.0,
            do_sample: true,
            ..Default::default()
        }
    }

    /// GPT model defaults
    fn gpt_defaults() -> Self {
        Self {
            temperature: 0.8,
            top_k: None,
            top_p: Some(0.95),
            repeat_penalty: 1.0,
            do_sample: true,
            ..Default::default()
        }
    }

    /// Merge with defaults, keeping existing values where present
    /// Self takes priority over defaults
    pub fn merge_with_defaults(self, defaults: &SamplingConfig) -> Self {
        Self {
            temperature: self.temperature,
            top_k: self.top_k.or(defaults.top_k),
            top_p: self.top_p.or(defaults.top_p),
            repeat_penalty: self.repeat_penalty,
            length_penalty: self.length_penalty,
            seed: self.seed.or(defaults.seed),
            do_sample: self.do_sample,
            typical_p: self.typical_p.or(defaults.typical_p),
            epsilon_cutoff: self.epsilon_cutoff.or(defaults.epsilon_cutoff),
            eta_cutoff: self.eta_cutoff.or(defaults.eta_cutoff),
            stop_tokens: if self.stop_tokens.is_empty() {
                defaults.stop_tokens.clone()
            } else {
                self.stop_tokens
            },
            max_tokens: if self.max_tokens == 0 {
                defaults.max_tokens
            } else {
                self.max_tokens
            },
        }
    }

    /// Apply user overrides from request
    pub fn apply_temperature(mut self, temp: Option<f32>) -> Self {
        if let Some(t) = temp {
            self.temperature = t;
            self.do_sample = t > 0.0;
        }
        self
    }

    pub fn apply_top_p(mut self, top_p: Option<f32>) -> Self {
        if let Some(p) = top_p {
            self.top_p = Some(p);
        }
        self
    }

    pub fn apply_top_k(mut self, top_k: Option<usize>) -> Self {
        if let Some(k) = top_k {
            self.top_k = Some(k);
        }
        self
    }

    pub fn apply_repeat_penalty(mut self, repeat_penalty: Option<f32>) -> Self {
        if let Some(rp) = repeat_penalty {
            self.repeat_penalty = rp;
        }
        self
    }

    pub fn apply_max_tokens(mut self, max_tokens: Option<usize>) -> Self {
        if let Some(m) = max_tokens {
            self.max_tokens = m;
        }
        self
    }

    pub fn apply_stop_tokens(mut self, stop_tokens: Option<Vec<String>>) -> Self {
        if let Some(st) = stop_tokens {
            self.stop_tokens = st;
        }
        self
    }
}


/// Load sampling configuration from model's config.json
pub async fn load_sampling_config(model_path: &std::path::Path) -> Result<SamplingConfig> {
    let config_path = model_path.join("config.json");

    if config_path.exists() {
        let config_str = tokio::fs::read_to_string(&config_path).await?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

        // Extract model name from path or config
        let model_id = config_json
            .get("_name_or_path")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                model_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
            });

        Ok(SamplingConfig::from_model_card(model_id, &config_json))
    } else {
        // Try to infer from model path
        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        Ok(SamplingConfig::for_model(model_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_specific_configs() {
        let qwen_config = SamplingConfig::for_model("Qwen/Qwen2-1.5B-Instruct");
        assert_eq!(qwen_config.temperature, 0.7);
        assert_eq!(qwen_config.top_k, Some(20));

        let llama_config = SamplingConfig::for_model("meta-llama/Llama-2-7b");
        assert_eq!(llama_config.temperature, 0.6);
        assert_eq!(llama_config.top_k, Some(40));

        let mistral_config = SamplingConfig::for_model("mistralai/Mistral-7B-v0.1");
        assert_eq!(mistral_config.top_k, None);
        assert_eq!(mistral_config.top_p, Some(0.95));
    }

    #[test]
    fn test_sampling_config_from_json() {
        let config_json = serde_json::json!({
            "generation_config": {
                "temperature": 0.5,
                "top_k": 30,
                "top_p": 0.85,
                "repetition_penalty": 1.2,
                "do_sample": true
            }
        });

        let config = SamplingConfig::from_model_card("test", &config_json);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_k, Some(30));
        assert_eq!(config.top_p, Some(0.85));
        assert_eq!(config.repeat_penalty, 1.2);
    }
}
