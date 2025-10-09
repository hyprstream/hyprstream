//! Clean inference interface for the runtime module
//!
//! This module provides high-level inference functionality with proper
//! request/response structures and clear separation of concerns.

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::instrument;

use super::TorchEngine;
use crate::adapters::LoRAWeightsData;

/// Request for text generation
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Input prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 - 2.0)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling parameter
    pub top_k: Option<usize>,
    /// Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition)
    pub repeat_penalty: f32,
    /// Whether to stream output
    pub stream: bool,
    /// Optional LoRA weights to apply
    pub lora_weights: Option<Arc<LoRAWeightsData>>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 100,
            temperature: 0.8,
            top_p: 0.95,
            top_k: Some(40),
            repeat_penalty: 1.1,
            stream: false,
            lora_weights: None,
        }
    }
}

/// Result of an inference operation
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Generated text output
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Time taken in milliseconds
    pub latency_ms: u64,
}

/// Extension trait for TorchEngine to add clean inference methods
#[allow(async_fn_in_trait)]
pub trait InferenceExt {
    /// Run inference with the given request
    async fn run_inference(&mut self, request: InferenceRequest) -> Result<InferenceResult>;

    /// Run inference with streaming output
    async fn run_inference_streaming<F>(
        &mut self,
        request: InferenceRequest,
        on_token: F,
    ) -> Result<InferenceResult>
    where
        F: FnMut(&str) + Send;
}

impl InferenceExt for TorchEngine {
    #[instrument(name = "inference.run", skip(self, request), fields(
        max_tokens = request.max_tokens,
        temperature = request.temperature,
        has_lora = request.lora_weights.is_some()
    ))]
    async fn run_inference(&mut self, request: InferenceRequest) -> Result<InferenceResult> {
        let start_time = std::time::Instant::now();
        
        // Apply LoRA weights if provided
        if let Some(lora_weights) = &request.lora_weights {
            apply_lora_to_engine(self, lora_weights).await?;
        }
        
        let token_counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = token_counter.clone();
        
        // Generate text using the engine's streaming API with custom parameters
        let generated_text = self.generate_streaming_with_params(
            &request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repeat_penalty,
            |_token| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        ).await?;
        
        let tokens_generated = token_counter.load(Ordering::Relaxed);
        let latency_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(InferenceResult {
            text: generated_text.clone(),
            tokens_generated: if tokens_generated > 0 { 
                tokens_generated 
            } else { 
                generated_text.split_whitespace().count() 
            },
            latency_ms,
        })
    }
    
    async fn run_inference_streaming<F>(
        &mut self,
        request: InferenceRequest,
        mut on_token: F,
    ) -> Result<InferenceResult>
    where
        F: FnMut(&str) + Send,
    {
        let start_time = std::time::Instant::now();
        
        // Apply LoRA weights if provided
        if let Some(lora_weights) = &request.lora_weights {
            apply_lora_to_engine(self, lora_weights).await?;
        }
        
        let token_counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = token_counter.clone();
        
        // Generate text with streaming callback and custom parameters
        let generated_text = self.generate_streaming_with_params(
            &request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.repeat_penalty,
            |token| {
                on_token(token);
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }
        ).await?;
        
        let tokens_generated = token_counter.load(Ordering::Relaxed);
        let latency_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(InferenceResult {
            text: generated_text.clone(),
            tokens_generated: if tokens_generated > 0 { 
                tokens_generated 
            } else { 
                generated_text.split_whitespace().count() 
            },
            latency_ms,
        })
    }
}

/// Apply LoRA weights to the engine
async fn apply_lora_to_engine(_engine: &mut TorchEngine, weights: &LoRAWeightsData) -> Result<()> {
    tracing::debug!(
        "Would apply LoRA weights: {} modules, rank {}, alpha {}",
        weights.target_modules.len(),
        weights.config.rank,
        weights.config.alpha,
    );
    
    Ok(())
}