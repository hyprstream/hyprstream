//! Base model traits and utilities for ML inference

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

/// Configuration for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier
    pub name: String,
    
    /// Model architecture (e.g., "qwen3", "llama", "gpt")
    pub architecture: String,
    
    /// Hidden dimension size
    pub hidden_size: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    
    /// Device to run on ("cpu", "cuda", "mps")
    pub device: String,
    
    /// Data type ("f16", "f32", "bf16")
    pub dtype: String,
    
    /// Model path or HuggingFace ID
    pub model_path: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "qwen3-1.7b".to_string(),
            architecture: "qwen3".to_string(),
            hidden_size: 1536,
            num_layers: 28,
            num_attention_heads: 12,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            device: "cuda".to_string(),
            dtype: "f16".to_string(),
            model_path: "Qwen/Qwen3-1.7B".to_string(),
        }
    }
}

/// Token data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub logprob: Option<f32>,
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.0,
            do_sample: true,
            stop_sequences: vec!["</s>".to_string()],
        }
    }
}

/// Generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub tokens: Vec<Token>,
    pub finish_reason: FinishReason,
    pub generation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    MaxTokens,
    StopSequence,
    EndOfSequence,
    Error(String),
}

/// Base trait for language models
#[async_trait::async_trait]
pub trait LanguageModel: Send + Sync {
    /// Get model configuration
    fn config(&self) -> &ModelConfig;
    
    /// Tokenize input text
    async fn tokenize(&self, text: &str) -> Result<Vec<u32>, ModelError>;
    
    /// Detokenize token IDs to text
    async fn detokenize(&self, tokens: &[u32]) -> Result<String, ModelError>;
    
    /// Generate text from prompt
    async fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResult, ModelError>;
    
    /// Generate streaming tokens
    async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<TokenStream, ModelError>;
    
    /// Forward pass (for training/fine-tuning)
    async fn forward(&self, input_ids: &[u32]) -> Result<ModelOutput, ModelError>;
    
    /// Get model memory usage in bytes
    fn memory_usage(&self) -> usize;
    
    /// Check if model is ready for inference
    fn is_ready(&self) -> bool;
}

/// Streaming token generator
pub struct TokenStream {
    // Implementation details would go here
    // For now, simplified placeholder
    pub tokens: Vec<Token>,
}

impl TokenStream {
    pub async fn next_token(&mut self) -> Option<Token> {
        // Placeholder implementation
        self.tokens.pop()
    }
}

/// Model output for forward passes
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub logits: Vec<Vec<f32>>, // [batch_size, seq_len, vocab_size]
    pub hidden_states: Option<Vec<Vec<Vec<f32>>>>, // [batch_size, seq_len, hidden_size]
    pub attention_weights: Option<Vec<Vec<Vec<Vec<f32>>>>>, // Attention matrices
}

/// Model loading and management
pub struct ModelLoader;

impl ModelLoader {
    /// Load model from configuration
    pub async fn load_model(config: ModelConfig) -> Result<Box<dyn LanguageModel>, ModelError> {
        match config.architecture.as_str() {
            "qwen3" => {
                // Will implement when we add Qwen3Model
                Err(ModelError::UnsupportedArchitecture(config.architecture))
            }
            _ => Err(ModelError::UnsupportedArchitecture(config.architecture)),
        }
    }
    
    /// Auto-detect model architecture from path
    pub async fn auto_detect_config(model_path: &str) -> Result<ModelConfig, ModelError> {
        // Placeholder implementation
        // In reality, would read config.json from model directory
        if model_path.contains("qwen") || model_path.contains("Qwen") {
            Ok(ModelConfig {
                architecture: "qwen3".to_string(),
                model_path: model_path.to_string(),
                ..Default::default()
            })
        } else {
            Err(ModelError::ConfigNotFound(model_path.to_string()))
        }
    }
}

/// Model pool for managing multiple instances
pub struct ModelPool {
    models: Arc<RwLock<HashMap<String, Arc<dyn LanguageModel>>>>,
    max_instances: usize,
}

impl ModelPool {
    pub fn new(max_instances: usize) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            max_instances,
        }
    }
    
    /// Get or load model instance
    pub async fn get_model(&self, model_id: &str) -> Result<Arc<dyn LanguageModel>, ModelError> {
        let models = self.models.read().await;
        if let Some(model) = models.get(model_id) {
            return Ok(Arc::clone(model));
        }
        drop(models);
        
        // Model not loaded, need to load it
        self.load_model(model_id).await
    }
    
    async fn load_model(&self, model_id: &str) -> Result<Arc<dyn LanguageModel>, ModelError> {
        // Check if we have room
        {
            let models = self.models.read().await;
            if models.len() >= self.max_instances {
                return Err(ModelError::PoolFull);
            }
        }
        
        // Auto-detect config and load
        let config = ModelLoader::auto_detect_config(model_id).await?;
        let model = ModelLoader::load_model(config).await?;
        let arc_model = Arc::from(model);
        
        // Store in pool
        {
            let mut models = self.models.write().await;
            models.insert(model_id.to_string(), Arc::clone(&arc_model));
        }
        
        Ok(arc_model)
    }
    
    /// List loaded models
    pub async fn list_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }
    
    /// Unload model
    pub async fn unload_model(&self, model_id: &str) -> bool {
        let mut models = self.models.write().await;
        models.remove(model_id).is_some()
    }
    
    /// Get memory usage of all models
    pub async fn total_memory_usage(&self) -> usize {
        let models = self.models.read().await;
        models.values().map(|m| m.memory_usage()).sum()
    }
}

/// Model-related errors
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),
    
    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    
    #[error("Config not found for model: {0}")]
    ConfigNotFound(String),
    
    #[error("Model pool is full")]
    PoolFull,
    
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    
    #[error("Generation error: {0}")]
    GenerationError(String),
    
    #[error("Device error: {0}")]
    DeviceError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.architecture, "qwen3");
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_layers, 28);
    }
    
    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.temperature, 0.8);
        assert!(config.do_sample);
    }
    
    #[tokio::test]
    async fn test_model_pool() {
        let pool = ModelPool::new(2);
        assert_eq!(pool.max_instances, 2);
        
        let models = pool.list_models().await;
        assert!(models.is_empty());
    }
    
    #[tokio::test]
    async fn test_auto_detect_config() {
        let config = ModelLoader::auto_detect_config("Qwen/Qwen3-1.7B").await.unwrap();
        assert_eq!(config.architecture, "qwen3");
    }
}