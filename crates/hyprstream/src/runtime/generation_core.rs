//! Unified generation core to eliminate code duplication
//!
//! This module provides a single implementation of the generation loop
//! that's used by all three generation methods (blocking, sync streaming, async streaming),
//! eliminating ~240 lines of duplicate code.

use anyhow::Result;

use super::{
    torch_engine::TorchEngine,
    utf8_decoder::{IncrementalUtf8Decoder, InvalidTokenCircuitBreaker},
    GenerationRequest, GenerationResult, FinishReason,
};

/// Bundled sampling parameters to reduce parameter passing overhead
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub vocab_size: Option<usize>,
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
            vocab_size: None,
        }
    }

    /// Create from GenerationRequest with vocab size
    pub fn from_request(req: &GenerationRequest, vocab_size: usize) -> Self {
        Self {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repeat_penalty: req.repeat_penalty,
            vocab_size: Some(vocab_size),
        }
    }

    /// Set the vocabulary size
    pub fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
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
    circuit_breaker: InvalidTokenCircuitBreaker,
}

impl<'a> GenerationCore<'a> {
    /// Create a new generation core
    pub fn new(engine: &'a TorchEngine) -> Self {
        Self {
            engine,
            decoder: IncrementalUtf8Decoder::new(),
            circuit_breaker: InvalidTokenCircuitBreaker::new(10),
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

            // Step 2: Sample next token using GPU sampling with new params interface
            let next_token = self.engine.sample_token_with_params(
                &logits,
                params,
                &input_ids,
            )?;

            // Step 3: Validate token is within vocabulary bounds
            if let Some(vocab_size) = params.vocab_size {
                if next_token >= vocab_size {
                    self.circuit_breaker.record_invalid(next_token, vocab_size)?;
                    continue;
                }
            }
            self.circuit_breaker.record_valid();

            // Step 4: Check for EOS token
            if self.engine.is_eos_token(next_token) {
                tracing::debug!("EOS token detected: {}", next_token);
                break;
            }

            // Step 5: Add token to sequence
            input_ids.push(next_token as i64);
            tokens_generated += 1;

            // Step 6: Decode new text incrementally
            let new_text = self.decoder.push_token_simple(next_token as i64, |tokens| {
                let mut full_ids = Vec::with_capacity(prompt_len + tokens.len());
                full_ids.extend(&input_ids[..prompt_len]);
                full_ids.extend(tokens);
                self.engine.detokenize(&full_ids[prompt_len..])
            })?;

            // Step 7: Call callback with new text (if any)
            if !new_text.is_empty() {
                match callback(&new_text)? {
                    CallbackControl::Continue => {},
                    CallbackControl::Stop => break,
                }
            }

            // Step 8: Check for stop tokens
            if request.stop_tokens.iter().any(|stop| self.decoder.get_text().contains(stop)) {
                tracing::debug!("Stop token detected");
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
        self.circuit_breaker = InvalidTokenCircuitBreaker::new(10);
    }
}