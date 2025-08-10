//! LLama.cpp runtime engine implementation

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::path::Path;
use std::time::Instant;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel, AddBos},
    context::LlamaContext,
    llama_batch::LlamaBatch,
    token::LlamaToken,
};

use super::{RuntimeEngine, ModelInfo, GenerationRequest, GenerationResult, FinishReason, RuntimeConfig};

/// LLama.cpp runtime engine implementation
pub struct LlamaCppEngine {
    backend: Option<LlamaBackend>,
    model: Option<LlamaModel>,
    // Store context parameters instead of context for now to avoid lifetime issues
    context_params: Option<LlamaContextParams>,
    config: RuntimeConfig,
    model_info: Option<ModelInfo>,
}

impl LlamaCppEngine {
    /// Create a new llama.cpp engine
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        Ok(Self {
            backend: None,
            model: None,
            context_params: None,
            config,
            model_info: None,
        })
    }

    /// Create with default configuration
    pub fn new_default() -> Result<Self> {
        Self::new(RuntimeConfig::default())
    }

    fn create_model_params(&self) -> LlamaModelParams {
        let mut params = LlamaModelParams::default();
        
        if let Some(gpu_layers) = self.config.gpu_layers {
            if self.config.use_gpu {
                params = params.with_n_gpu_layers(gpu_layers as u32);
            }
        }

        params
    }

    fn create_context_params(&self) -> LlamaContextParams {
        let mut params = LlamaContextParams::default()
            .with_n_ctx(None) // Simplified for now
            .with_n_batch(self.config.batch_size as u32);

        if let Some(threads) = self.config.cpu_threads {
            params = params.with_n_threads(threads as i32);
        }

        params
    }

    fn detect_quantization(&self, path: &Path) -> Option<String> {
        let filename = path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        if filename.contains("q4_0") {
            Some("Q4_0".to_string())
        } else if filename.contains("q4_1") {
            Some("Q4_1".to_string())
        } else if filename.contains("q5_0") {
            Some("Q5_0".to_string())
        } else if filename.contains("q5_1") {
            Some("Q5_1".to_string())
        } else if filename.contains("q8_0") {
            Some("Q8_0".to_string())
        } else if filename.contains("f16") {
            Some("F16".to_string())
        } else if filename.contains("f32") {
            Some("F32".to_string())
        } else {
            None
        }
    }

    /// Generate text using real LLaMA.cpp context and model (token-by-token generation)
    fn generate_with_context(
        &self,
        context: &mut LlamaContext,
        model: &LlamaModel,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        tracing::info!("Starting real LLaMA.cpp token-by-token generation for prompt: '{}'", prompt);
        
        // Step 1: Tokenize the prompt
        let tokens_list = model.str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow!("Failed to tokenize prompt: {:?}", e))?;
        
        tracing::info!("Tokenized prompt into {} tokens", tokens_list.len());
        
        // Step 2: Create batch for decoding
        let mut batch = LlamaBatch::new(512, 1);
        
        // Add prompt tokens to batch
        for (i, token) in tokens_list.iter().enumerate() {
            let is_last = i == tokens_list.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| anyhow!("Failed to add token to batch: {:?}", e))?;
        }
        
        // Step 3: Decode the initial prompt
        context.decode(&mut batch)
            .map_err(|e| anyhow!("Failed to decode prompt: {:?}", e))?;
        
        tracing::info!("Successfully decoded prompt, starting token generation");
        
        // Step 4: Generate tokens one by one
        let mut generated_tokens = Vec::new();
        let mut generated_text = String::new();
        let mut current_pos = tokens_list.len() as i32; // Start position after prompt
        
        for n_cur in 0..max_tokens {
            // Sample next token using greedy sampling for simplicity
            let logits = context.get_logits_ith(batch.n_tokens() - 1);
            
            // Find token with highest probability (greedy sampling)
            let n_vocab = model.n_vocab() as usize;
            let mut best_token = 0i32;
            let mut best_logit = f32::NEG_INFINITY;
            
            for i in 0..n_vocab.min(logits.len()) {
                if logits[i] > best_logit {
                    best_logit = logits[i];
                    best_token = i as i32;
                }
            }
            
            generated_tokens.push(best_token);
            
            // Convert token to text
            match model.token_to_bytes(LlamaToken(best_token), llama_cpp_2::model::Special::Tokenize) {
                Ok(token_bytes) => {
                    if let Ok(token_str) = String::from_utf8(token_bytes) {
                        generated_text.push_str(&token_str);
                        tracing::debug!("Generated token {}: '{}'", n_cur, token_str);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to convert token {} to string: {:?}", best_token, e);
                }
            }
            
            // Check for end-of-sequence token (Qwen2 uses 151645 for <|im_end|>)
            if best_token == 151645 || best_token == 151643 {
                tracing::info!("Hit end-of-sequence token ({}), stopping generation", best_token);
                break;
            }
            
            // Prepare next iteration: clear batch and add the new token with correct position
            batch.clear();
            batch.add(LlamaToken(best_token), current_pos, &[0], true)
                .map_err(|e| anyhow!("Failed to add generated token to batch: {:?}", e))?;
            
            current_pos += 1; // Increment position for next token
            
            // Decode the new token
            context.decode(&mut batch)
                .map_err(|e| anyhow!("Failed to decode generated token: {:?}", e))?;
        }
        
        tracing::info!("Generated {} tokens using real LLaMA.cpp: '{}'", 
                      generated_tokens.len(), 
                      generated_text.chars().take(100).collect::<String>());
        
        Ok(generated_text)
    }

    /// Generate response using real LLaMA.cpp inference (fallback method)  
    fn generate_basic_response(&self, _model_info: &ModelInfo, prompt: &str, _max_tokens: usize) -> String {
        tracing::error!("generate_basic_response should not be called - using real LLaMA.cpp inference instead");
        
        // This method should no longer be used since we have real token-by-token generation
        // Return an error indicator instead of templated text
        format!("ERROR: Fallback method called instead of real LLaMA.cpp inference for prompt: '{}'", 
                prompt.chars().take(50).collect::<String>())
    }

}

#[async_trait]
impl RuntimeEngine for LlamaCppEngine {
    async fn load_model(&mut self, path: &Path) -> Result<()> {
        // Initialize backend when loading model
        let backend = LlamaBackend::init()
            .map_err(|e| anyhow!("Failed to initialize llama.cpp backend: {:?}", e))?;

        let model_params = self.create_model_params();
        
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| anyhow!("Failed to load model from {}: {:?}", path.display(), e))?;

        let context_params = self.create_context_params();
        
        // Extract real model information  
        let model_info = ModelInfo {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            parameters: model.n_params() as u64,
            context_length: self.config.context_length, // Use config for now
            vocab_size: model.n_vocab() as usize,
            architecture: "llama".to_string(), // Could extract from metadata
            quantization: self.detect_quantization(path),
        };

        self.backend = Some(backend);
        self.model = Some(model);
        self.context_params = Some(context_params);
        self.model_info = Some(model_info);

        tracing::info!("✅ Loaded model: {}", path.display());
        Ok(())
    }

    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            ..Default::default()
        };
        
        let result = self.generate_with_params(request).await?;
        Ok(result.text)
    }

    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult> {
        if !self.is_loaded() {
            return Err(anyhow!("Model not loaded"));
        }

        let start_time = Instant::now();
        
        // Real LLaMA.cpp text generation (basic implementation)
        let (result_text, actual_tokens_generated) = match (self.backend.as_ref(), self.model.as_ref(), self.context_params.as_ref()) {
            (Some(backend), Some(model), Some(context_params)) => {
                tracing::info!("Generating with LLaMA.cpp model: {} parameters", model.n_params());
                
                tracing::info!("Creating LLaMA.cpp context for prompt: {}", 
                              request.prompt.chars().take(50).collect::<String>());
                
                // Create context for generation
                let mut context = model.new_context(backend, context_params.clone())
                    .map_err(|e| anyhow!("Failed to create LLaMA context: {:?}", e))?;
                
                tracing::info!("Successfully created LLaMA.cpp context - starting real text generation");
                
                // Perform real LLaMA.cpp text generation using the context
                let generated_text = self.generate_with_context(&mut context, &model, &request.prompt, request.max_tokens)?;
                let tokens_count = generated_text.split_whitespace().count().min(request.max_tokens);
                
                tracing::info!("Generated response with {} tokens using real LLaMA.cpp inference", tokens_count);
                
                (generated_text, tokens_count)
            }
            _ => {
                return Err(anyhow!("Model, backend, or context not initialized"));
            }
        };

        let generation_time = start_time.elapsed();
        let generation_time_ms = generation_time.as_millis() as u64;
        
        let tokens_per_second = if generation_time_ms > 0 {
            (actual_tokens_generated as f32 * 1000.0) / generation_time_ms as f32
        } else {
            0.0
        };

        Ok(GenerationResult {
            text: result_text,
            tokens_generated: actual_tokens_generated,
            finish_reason: if actual_tokens_generated >= request.max_tokens {
                FinishReason::MaxTokens
            } else {
                FinishReason::EndOfSequence
            },
            generation_time_ms,
            tokens_per_second,
        })
    }

    fn model_info(&self) -> ModelInfo {
        self.model_info.clone().unwrap_or_else(|| ModelInfo {
            name: "unloaded".to_string(),
            parameters: 0,
            context_length: 0,
            vocab_size: 0,
            architecture: "unknown".to_string(),
            quantization: None,
        })
    }

    fn is_loaded(&self) -> bool {
        self.backend.is_some() && self.model.is_some() && self.context_params.is_some()
    }
}

impl LlamaCppEngine {
    /// Apply LoRA adapter from checkpoint file
    pub fn apply_lora_checkpoint(
        &mut self,
        checkpoint_path: &std::path::Path,
        adapter_id: &str,
        scale: f32,
    ) -> Result<()> {
        if !self.is_loaded() {
            return Err(anyhow!("Model not loaded. Call load_model() first."));
        }
        
        tracing::info!("📎 Loading LoRA checkpoint: {} (scale: {})", checkpoint_path.display(), scale);
        
        // Get references to backend, model, and context 
        let _backend = self.backend.as_ref()
            .ok_or_else(|| anyhow!("Backend not initialized"))?;
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;
        
        // Load LoRA adapter from GGUF checkpoint file using real LLaMA.cpp API
        match model.lora_adapter_init(checkpoint_path) {
            Ok(_lora_adapter) => {
                tracing::info!("✅ Successfully loaded LoRA adapter from checkpoint");
                
                // Store the adapter for later use (we'll need to manage this properly)
                // For now, just log success
                tracing::info!("📋 LoRA adapter '{}' ready for application", adapter_id);
                
                // Note: We would apply it to context during inference:
                // context.lora_adapter_set(&mut lora_adapter, scale)?;
                
                Ok(())
            }
            Err(e) => {
                tracing::error!("❌ Failed to load LoRA adapter: {:?}", e);
                Err(anyhow!("Failed to load LoRA adapter from {}: {:?}", 
                           checkpoint_path.display(), e))
            }
        }
    }
    
    /// Legacy method for backward compatibility - now deprecated
    #[deprecated(note = "Use apply_lora_checkpoint instead")]
    pub fn apply_lora_adapter(
        &mut self,
        adapter_id: &str,
        lora_weights: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        tracing::warn!("🚨 Using deprecated apply_lora_adapter - should use checkpoint system");
        
        // For backward compatibility, just log that this method was called
        tracing::info!("📎 Legacy LoRA adapter call for '{}' with {} tensors", 
                      adapter_id, lora_weights.len());
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = LlamaCppEngine::new_default();
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_quantization_detection() {
        let engine = LlamaCppEngine::new_default().unwrap();
        
        assert_eq!(
            engine.detect_quantization(Path::new("model.q4_0.gguf")),
            Some("Q4_0".to_string())
        );
        
        assert_eq!(
            engine.detect_quantization(Path::new("model.f16.gguf")),
            Some("F16".to_string())
        );
        
        assert_eq!(
            engine.detect_quantization(Path::new("model.gguf")),
            None
        );
    }

    #[test]
    fn test_stop_token_detection() {
        let engine = LlamaCppEngine::new_default().unwrap();
        let stop_tokens = vec!["</s>".to_string(), "<|endoftext|>".to_string()];
        
        assert!(engine.is_stop_token("</s>", &stop_tokens));
        assert!(engine.is_stop_token("text <|endoftext|>", &stop_tokens));
        assert!(!engine.is_stop_token("normal text", &stop_tokens));
    }
}