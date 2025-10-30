//! Unified generation core to eliminate code duplication
//!
//! This module provides a single implementation of the generation loop
//! that's used by all three generation methods (blocking, sync streaming, async streaming),
//! eliminating ~240 lines of duplicate code.

use anyhow::Result;
use tokio_util::sync::CancellationToken;
use std::time::Duration;

use super::{
    torch_engine::TorchEngine,
    GenerationRequest, GenerationResult, FinishReason,
    streaming::{StreamingCallback, ContinueGeneration},
};

/// Bundled sampling parameters to reduce parameter passing overhead
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
}

impl SamplingParams {
    /// Create from individual parameters (for backward compatibility)
    pub fn new(
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repeat_penalty,
        }
    }

    /// Create from GenerationRequest
    pub fn from_request(req: &GenerationRequest) -> Self {
        Self {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repeat_penalty: req.repeat_penalty,
        }
    }
}

/// Unified generation core that eliminates duplicate loops
pub struct GenerationCore<'a> {
    engine: &'a TorchEngine,
    generated_tokens: Vec<i64>, // Store only generated tokens
}

impl<'a> GenerationCore<'a> {
    /// Create a new generation core
    pub fn new(engine: &'a TorchEngine) -> Result<Self> {
        Ok(Self {
            engine,
            generated_tokens: Vec::new(),
        })
    }

    /// THE unified generation loop - single source of truth
    ///
    /// This is the ONLY place where the generation loop exists.
    /// All other generation methods eventually delegate to this method.
    ///
    /// Supports:
    /// - Async callbacks (Box<dyn StreamingCallback>)
    /// - Cancellation via CancellationToken
    /// - Timeout handling
    /// - Backpressure/pause support
    pub async fn generate_tokens_async(
        &mut self,
        mut input_ids: Vec<i64>,
        params: &SamplingParams,
        request: &GenerationRequest,
        mut callback: Box<dyn StreamingCallback>,
        cancel_token: CancellationToken,
        timeout: Duration,
    ) -> Result<GenerationResult> {
        use tokio::time::timeout as tokio_timeout;

        let start_time = std::time::Instant::now();
        let prompt_len = input_ids.len();
        let mut tokens_generated = 0;

        // Clear and prepare our generated tokens buffer
        self.generated_tokens.clear();
        self.generated_tokens.reserve(request.max_tokens);

        // Create DecodeStream for incremental decoding (O(1) per token!)
        let tokenizer = self.engine.get_tokenizer()?;
        let mut decode_stream = tokenizer.decode_stream(false); // Don't skip special tokens

        // Clear KV cache before generation to prevent context pollution from previous runs
        self.engine.clear_kv_cache();

        // Notify callback that generation is starting
        callback.on_start().await;

        // Create the generation future
        let generation_future = async {
            for i in 0..request.max_tokens {
                // Check cancellation
                if cancel_token.is_cancelled() {
                    tracing::info!("Generation cancelled by client");
                    break;
                }

                // Step 1: Forward pass with KV caching
                let logits = if i == 0 {
                    self.engine.forward(&input_ids)?
                } else {
                    self.engine.forward_cached(&input_ids, prompt_len + i - 1, true)?
                };

                // Step 2: Sample next token with new params interface
                // IMPORTANT: Only apply repeat penalty to generated tokens, not the prompt!
                // Using full input_ids would penalize words from conversation history

                // DIAGNOSTIC: Log what we're passing to repeat penalty
                if i == 0 {
                    tracing::debug!("🔍 REPEAT PENALTY DIAGNOSTIC:");
                    tracing::debug!("   Prompt length: {} tokens", prompt_len);
                    tracing::debug!("   Generated tokens so far: {} tokens", self.generated_tokens.len());
                    tracing::debug!("   Full input_ids: {} tokens", input_ids.len());
                    tracing::debug!("   Passing to repeat penalty: {} tokens (generated_tokens)", self.generated_tokens.len());
                }

                let next_token = self.engine.sample_token_with_params(
                    &logits,
                    params,
                    &self.generated_tokens,
                )?;

                // Log token generation progress every 50 tokens
                if tokens_generated % 50 == 0 {
                    tracing::info!("🔄 Generated {} tokens (repeat penalty applied to {} tokens)",
                        tokens_generated, self.generated_tokens.len());
                }

                // Step 3: Validate token BEFORE adding to context

                // Check if token is beyond the tokenizer's vocabulary
                let vocab_size = self.engine.get_vocab_size();
                if vocab_size > 0 && next_token >= vocab_size {
                    tracing::error!(
                        "Generated token {} exceeds vocabulary size {}. This indicates a bug in model initialization or sampling.",
                        next_token, vocab_size
                    );
                    return Err(anyhow::anyhow!(
                        "Generated out-of-vocabulary token {}: exceeds vocab size {}",
                        next_token,
                        vocab_size
                    ));
                }

                // Check for EOS token from model config
                if self.engine.is_eos_token(next_token) {
                    tracing::debug!("EOS token detected: {}", next_token);
                    break;
                }

                // Check if token should be blocked from text-only generation
                if let Some(false) = self.engine.check_special_token(next_token) {
                    tracing::debug!("Blocked special token: {}", next_token);
                    continue;
                }

                // Add token to sequence and our generated tokens buffer
                input_ids.push(next_token as i64);
                self.generated_tokens.push(next_token as i64);
                tokens_generated += 1;

                // Step 5: Decode incrementally using DecodeStream (O(1) per token!)
                let new_text = match decode_stream.step(next_token as u32) {
                    Ok(Some(text)) => {
                        // Log any unusual characters
                        if text.chars().any(|c| (c as u32) > 0x4E00 && (c as u32) < 0x9FFF) {
                            tracing::info!("⚠️  Token {} decoded to text with CJK character: {:?}", next_token, text);
                        }
                        if text.contains('�') {
                            tracing::info!("⚠️  Token {} decoded to replacement character: {:?}", next_token, text);
                        }
                        text
                    }
                    Ok(None) => {
                        tracing::debug!("Token {} produced no text (incomplete UTF-8)", next_token);
                        String::new()
                    }
                    Err(e) => {
                        tracing::info!("❌ Failed to decode token {}: {}", next_token, e);
                        String::new()
                    }
                };

                // Step 6: Call async callback with new text (if any)
                if !new_text.is_empty() {
                    match callback.on_token(&new_text).await? {
                        ContinueGeneration::Continue => {},
                        ContinueGeneration::Stop => {
                            tracing::info!("Generation stopped by callback");
                            break;
                        },
                        ContinueGeneration::Pause(duration) => {
                            tokio::time::sleep(duration).await;
                        }
                    }
                }

            // Step 7: Check for stop tokens at the end of generated text
            // We check if any stop token appears at the end to avoid false positives
            if request
                .stop_tokens
                .iter()
                .any(|stop| !stop.is_empty() && self.decoder.get_text().contains(stop))
            {
                tracing::debug!("Stop token detected in generated text");
                break;
            }

            Ok::<_, anyhow::Error>(())
        };

        // Apply timeout
        let result = match tokio_timeout(timeout, generation_future).await {
            Ok(Ok(())) => {
                let generation_time = start_time.elapsed();

                // Determine finish reason
                let finish_reason = if tokens_generated >= request.max_tokens {
                    FinishReason::MaxTokens
                } else {
                    FinishReason::Stop
                };

                // Notify completion
                callback.on_complete(finish_reason.clone()).await;

                // Get the final text by decoding all generated tokens
                let final_text = if tokens_generated > 0 {
                    self.engine.detokenize(&self.generated_tokens).unwrap_or_default()
                } else {
                    String::new()
                };

                Ok(GenerationResult {
                    text: final_text,
                    tokens_generated,
                    finish_reason,
                    generation_time_ms: generation_time.as_millis() as u64,
                    tokens_per_second: if generation_time.as_secs_f32() > 0.0 {
                        tokens_generated as f32 / generation_time.as_secs_f32()
                    } else {
                        0.0
                    },
                })
            }
            Ok(Err(e)) => {
                callback.on_error(anyhow::anyhow!("{}", e)).await;
                Err(e)
            }
            Err(_) => {
                let err = anyhow::anyhow!("Generation timeout after {:?}", timeout);
                callback.on_error(anyhow::anyhow!("{}", err)).await;
                Err(err)
            }
        };

        result
    }
}