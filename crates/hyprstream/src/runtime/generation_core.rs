//! Unified generation core to eliminate code duplication
//!
//! This module provides a single implementation of the generation loop
//! that's used by all three generation methods (blocking, sync streaming, async streaming),
//! eliminating ~240 lines of duplicate code.

use anyhow::Result;

use super::{
    torch_engine::TorchEngine,
    utf8_decoder::IncrementalUtf8Decoder,
    GenerationRequest, GenerationResult, FinishReason,
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

/// Control flow for generation callbacks
#[derive(Debug, Clone, Copy)]
pub enum CallbackControl {
    Continue,
    Stop,
}

/// Unified generation core that eliminates duplicate loops
pub struct GenerationCore<'a> {
    engine: &'a TorchEngine,
    decoder: IncrementalUtf8Decoder,
}

impl<'a> GenerationCore<'a> {
    /// Create a new generation core
    pub fn new(engine: &'a TorchEngine) -> Self {
        Self {
            engine,
            decoder: IncrementalUtf8Decoder::new(),
        }
    }

    /// Core generation loop - single implementation used by all generation methods
    ///
    /// This eliminates the duplicate loops in:
    /// - generate_with_params (blocking)
    /// - generate_streaming_internal (sync streaming)
    /// - generate_streaming_async (async streaming)
    pub fn generate_tokens<C>(
        &mut self,
        mut input_ids: Vec<i64>,
        params: &SamplingParams,
        request: &GenerationRequest,
        mut callback: C,
    ) -> Result<GenerationResult>
    where
        C: FnMut(&str) -> Result<CallbackControl>,
    {
        let start_time = std::time::Instant::now();
        let prompt_len = input_ids.len();
        let mut tokens_generated = 0;

        for i in 0..request.max_tokens {
            // Step 1: Forward pass with KV caching
            let logits = if i == 0 {
                self.engine.forward(&input_ids)?
            } else {
                self.engine.forward_cached(&input_ids, prompt_len + i - 1, true)?
            };

            // Step 2: Sample next token with new params interface
            let next_token = self.engine.sample_token_with_params(
                &logits,
                params,
                &input_ids,
            )?;

            // Step 3: Validate token BEFORE adding to context

            // Check if token is beyond the tokenizer's vocabulary
            // Tokens beyond vocab size indicate a bug in sampling or model configuration
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

            // Step 4: Add token to sequence ONLY after validation
            input_ids.push(next_token as i64);
            tokens_generated += 1;

            // Step 5: Decode new text incrementally using custom decoder
            // Note: Using IncrementalUtf8Decoder due to tokenizers 0.20.4 DecodeStream bugs
            // TODO: Upgrade to tokenizers 0.22.1+ to use native DecodeStream API for O(1) decoding
            let new_text = self.decoder.push_token_simple(next_token as i64, |tokens| {
                let mut full_ids = Vec::with_capacity(prompt_len + tokens.len());
                full_ids.extend(&input_ids[..prompt_len]);
                full_ids.extend(tokens);
                self.engine.detokenize(&full_ids[prompt_len..])
            })?;

            // Step 6: Call callback with new text (if any)
            if !new_text.is_empty() {
                match callback(&new_text)? {
                    CallbackControl::Continue => {},
                    CallbackControl::Stop => break,
                }
            }

            // Step 7: Check for stop tokens at the end of generated text
            // We check if any stop token appears at the end to avoid false positives
            if request
                .stop_tokens
                .iter()
                .any(|stop| !stop.is_empty() && self.decoder.get_text().ends_with(stop))
            {
                tracing::debug!("Stop token detected at end of text");
                break;
            }
        }

        let generation_time = start_time.elapsed();

        Ok(GenerationResult {
            text: self.decoder.get_text().to_string(),
            tokens_generated,
            finish_reason: if tokens_generated >= request.max_tokens {
                FinishReason::MaxTokens
            } else {
                FinishReason::Stop
            },
            generation_time_ms: generation_time.as_millis() as u64,
            tokens_per_second: if generation_time.as_secs_f32() > 0.0 {
                tokens_generated as f32 / generation_time.as_secs_f32()
            } else {
                0.0
            },
        })
    }

    /// Get the current decoded text
    pub fn get_text(&self) -> &str {
        self.decoder.get_text()
    }

    /// Get the number of tokens in the decoder
    pub fn get_token_count(&self) -> usize {
        self.decoder.get_tokens().len()
    }

    /// Reset the generation state for reuse
    pub fn reset(&mut self) {
        self.decoder = IncrementalUtf8Decoder::new();
    }
}