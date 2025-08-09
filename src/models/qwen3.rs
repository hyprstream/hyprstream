//! Qwen3 model wrapper using unified runtime system

use crate::config::{HyprConfig, GenerationRequest, GenerationResult, ModelInfo};
use crate::runtime::{RuntimeEngine, create_engine};
use anyhow::Result;
use std::path::Path;

/// Qwen3 model wrapper that uses the unified runtime engine
pub struct Qwen3Wrapper {
    /// Underlying runtime engine
    engine: Box<dyn RuntimeEngine>,
    /// Model configuration
    config: HyprConfig,
}

impl Qwen3Wrapper {
    /// Create new Qwen3 wrapper with configuration
    pub fn new(config: HyprConfig) -> Result<Self> {
        let engine = create_engine(&config.runtime)?;
        
        Ok(Self {
            engine: Box::new(engine),
            config,
        })
    }
    
    /// Create Qwen3 wrapper with default configuration for the model
    pub fn new_default(model_path: &Path) -> Result<Self> {
        let mut config = HyprConfig::default();
        config.model.path = model_path.to_path_buf();
        config.model.name = "qwen3-1.5b".to_string();
        config.model.architecture = "qwen".to_string();
        
        // Qwen3-specific optimizations
        config.runtime.context_length = 32768; // Qwen3 supports long context
        config.generation.temperature = 0.7;   // Good default for Qwen3
        
        Self::new(config)
    }
    
    /// Load the Qwen3 model
    pub async fn load_model(&mut self) -> Result<()> {
        println!("🔄 Loading Qwen3 model: {}", self.config.model.path.display());
        self.engine.load_model(&self.config.model.path).await?;
        println!("✅ Qwen3 model loaded successfully");
        Ok(())
    }
    
    /// Generate text with Qwen3
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let request = self.config.create_request(prompt.to_string());
        let result = self.engine.generate_with_params(request).await?;
        Ok(result.text)
    }
    
    /// Generate with custom parameters
    pub async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult> {
        self.engine.generate_with_params(request).await
    }
    
    /// Check if model is ready
    pub fn is_ready(&self) -> bool {
        self.engine.is_loaded()
    }
    
    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        self.engine.model_info()
    }
    
    /// Get configuration
    pub fn config(&self) -> &HyprConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_qwen3_creation() {
        let model_path = PathBuf::from("test_model.gguf");
        let wrapper = Qwen3Wrapper::new_default(&model_path);
        assert!(wrapper.is_ok());
        
        let wrapper = wrapper.unwrap();
        assert_eq!(wrapper.config().model.architecture, "qwen");
        assert_eq!(wrapper.config().runtime.context_length, 32768);
    }
}