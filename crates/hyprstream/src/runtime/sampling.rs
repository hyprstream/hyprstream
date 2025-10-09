//! Token sampling strategies for language model generation
//! 
//! Implements various sampling methods including temperature, top-k, and top-p (nucleus) sampling
//! with model-specific configurations loaded from HuggingFace model cards.

use tracing::warn;
use anyhow::{Result, anyhow};
use tch::Tensor;
use rand::prelude::*;
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
    pub repetition_penalty: f32,
    
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
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            seed: None,
            do_sample: true,
            typical_p: None,
            epsilon_cutoff: None,
            eta_cutoff: None,
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
            temperature: config.get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0),
            
            top_k: config.get("top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            
            top_p: config.get("top_p")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            
            repetition_penalty: config.get("repetition_penalty")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0),
            
            length_penalty: config.get("length_penalty")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(1.0),
            
            do_sample: config.get("do_sample")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            
            typical_p: config.get("typical_p")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            
            epsilon_cutoff: config.get("epsilon_cutoff")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            
            eta_cutoff: config.get("eta_cutoff")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            
            seed: None,
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
            repetition_penalty: 1.05,
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
            repetition_penalty: 1.1,
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
            repetition_penalty: 1.0,
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
            repetition_penalty: 1.0,
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
            repetition_penalty: 1.0,
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
            repetition_penalty: 1.0,
            do_sample: true,
            ..Default::default()
        }
    }
}

/// Token sampler with various strategies
pub struct TokenSampler {
    config: SamplingConfig,
    rng: StdRng,
    token_history: Vec<u32>,
}

impl TokenSampler {
    /// Create a new sampler with the given configuration
    pub fn new(config: SamplingConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        Self {
            config,
            rng,
            token_history: Vec::new(),
        }
    }
    
    /// Sample next token from logits
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        // Debug: Check logits statistics
        if let Ok(logits_vec) = Vec::<f32>::try_from(logits.shallow_clone()) {
            let max = logits_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| anyhow!("Empty logits tensor"))?;
            let min = logits_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| anyhow!("Empty logits tensor"))?;
            warn!("DEBUG: Logits shape: {:?}, min={:.3}, max={:.3}", logits.size(), min, max);
        }
        // Apply repetition penalty if needed
        let logits = if self.config.repetition_penalty != 1.0 && !self.token_history.is_empty() {
            self.apply_repetition_penalty(logits)?
        } else {
            logits.shallow_clone()
        };
        
        // If not sampling, use greedy decoding
        if !self.config.do_sample {
            let token = logits.argmax(0, false).int64_value(&[]) as u32;
            warn!("DEBUG: Using greedy decoding, selected token {}", token);
            self.token_history.push(token);
            return Ok(token);
        }
        
        // Apply temperature
        let logits = if self.config.temperature != 1.0 {
            logits / self.config.temperature as f64
        } else {
            logits
        };
        
        // Apply sampling strategy
        let token = if let Some(p) = self.config.typical_p {
            self.typical_p_sampling(&logits, p)?
        } else if self.config.top_k.is_some() || self.config.top_p.is_some() {
            self.top_k_top_p_sampling(&logits)?
        } else {
            self.multinomial_sampling(&logits)?
        };
        
        self.token_history.push(token);
        Ok(token)
    }
    
    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &Tensor) -> Result<Tensor> {
        let mut logits_vec: Vec<f32> = Vec::try_from(logits.shallow_clone()).map_err(|e| anyhow::anyhow!("Tensor conversion failed: {:?}", e))?;
        
        for &token_id in &self.token_history {
            if (token_id as usize) < logits_vec.len() {
                let score = logits_vec[token_id as usize];
                logits_vec[token_id as usize] = if score < 0.0 {
                    score * self.config.repetition_penalty
                } else {
                    score / self.config.repetition_penalty
                };
            }
        }
        
        Ok(Tensor::from_slice(&logits_vec).to_device(logits.device()).view_as(logits))
    }
    
    /// Top-k and top-p sampling
    fn top_k_top_p_sampling(&mut self, logits: &Tensor) -> Result<u32> {
        let probs = logits.softmax(0, tch::Kind::Float);
        let mut probs_vec: Vec<f32> = Vec::try_from(probs).map_err(|e| anyhow::anyhow!("Tensor conversion failed: {:?}", e))?;
        
        // Get indices sorted by probability
        let mut indices: Vec<usize> = (0..probs_vec.len()).collect();
        indices.sort_by(|&a, &b| probs_vec[b].partial_cmp(&probs_vec[a]).unwrap());
        
        // Apply top-k filtering
        if let Some(k) = self.config.top_k {
            if k < indices.len() {
                // Zero out probabilities outside top-k
                for &idx in &indices[k..] {
                    probs_vec[idx] = 0.0;
                }
            }
        }
        
        // Apply top-p (nucleus) filtering
        if let Some(p) = self.config.top_p {
            let mut cumsum = 0.0;
            let mut cutoff_idx = indices.len();
            
            for (i, &idx) in indices.iter().enumerate() {
                cumsum += probs_vec[idx];
                if cumsum > p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            
            // Zero out probabilities outside nucleus
            for &idx in &indices[cutoff_idx..] {
                probs_vec[idx] = 0.0;
            }
        }
        
        // Renormalize
        let sum: f32 = probs_vec.iter().sum();
        if sum <= 0.0 {
            // Fallback to uniform distribution over top-k tokens
            let k = self.config.top_k.unwrap_or(50).min(indices.len());
            return Ok(indices[self.rng.gen_range(0..k)] as u32);
        }
        
        for p in &mut probs_vec {
            *p /= sum;
        }
        
        // Sample from the filtered distribution
        self.sample_from_probs(&probs_vec)
    }
    
    /// Typical-p sampling (locally typical sampling)
    fn typical_p_sampling(&mut self, logits: &Tensor, typical_p: f32) -> Result<u32> {
        let probs = logits.softmax(0, tch::Kind::Float);
        let probs_vec: Vec<f32> = Vec::try_from(probs).map_err(|e| anyhow::anyhow!("Tensor conversion failed: {:?}", e))?;
        
        // Calculate entropy
        let entropy: f32 = probs_vec.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        
        // Calculate absolute difference from entropy for each token
        let mut diffs: Vec<(usize, f32, f32)> = probs_vec.iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| {
                let neg_log_p = -p.ln();
                let diff = (neg_log_p - entropy).abs();
                (i, p, diff)
            })
            .collect();
        
        // Sort by difference (ascending)
        diffs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        
        // Find cutoff where cumulative probability exceeds typical_p
        let mut cumsum = 0.0;
        let mut cutoff_idx = 0;
        
        for (i, (_, p, _)) in diffs.iter().enumerate() {
            cumsum += p;
            if cumsum >= typical_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        if cutoff_idx == 0 {
            cutoff_idx = 1; // At least one token
        }
        
        // Create probability distribution from locally typical tokens
        let mut filtered_probs = vec![0.0; probs_vec.len()];
        let mut sum = 0.0;
        
        for i in 0..cutoff_idx.min(diffs.len()) {
            let (idx, p, _) = diffs[i];
            filtered_probs[idx] = p;
            sum += p;
        }
        
        // Renormalize
        if sum > 0.0 {
            for p in &mut filtered_probs {
                *p /= sum;
            }
        }
        
        self.sample_from_probs(&filtered_probs)
    }
    
    /// Simple multinomial sampling from logits
    fn multinomial_sampling(&mut self, logits: &Tensor) -> Result<u32> {
        let probs = logits.softmax(0, tch::Kind::Float);
        let probs_vec: Vec<f32> = Vec::try_from(probs).map_err(|e| anyhow::anyhow!("Tensor conversion failed: {:?}", e))?;
        self.sample_from_probs(&probs_vec)
    }
    
    /// Sample from a probability distribution
    fn sample_from_probs(&mut self, probs: &[f32]) -> Result<u32> {
        let mut cumsum = 0.0;
        let threshold: f32 = self.rng.gen();
        
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= threshold {
                return Ok(i as u32);
            }
        }
        
        // Fallback to last token if we didn't sample anything
        Ok((probs.len() - 1) as u32)
    }
    
    /// Clear token history (useful for new generation)
    pub fn clear_history(&mut self) {
        self.token_history.clear();
    }
    
    /// Add tokens to history (useful when continuing from a prompt)
    pub fn add_to_history(&mut self, tokens: &[u32]) {
        self.token_history.extend_from_slice(tokens);
    }
}

/// Load sampling configuration from model's config.json
pub async fn load_sampling_config(model_path: &std::path::Path) -> Result<SamplingConfig> {
    let config_path = model_path.join("config.json");
    
    if config_path.exists() {
        let config_str = tokio::fs::read_to_string(&config_path).await?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        
        // Extract model name from path or config
        let model_id = config_json.get("_name_or_path")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                model_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
            });
        
        Ok(SamplingConfig::from_model_card(model_id, &config_json))
    } else {
        // Try to infer from model path
        let model_name = model_path.file_name()
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
        assert_eq!(config.repetition_penalty, 1.2);
    }
}