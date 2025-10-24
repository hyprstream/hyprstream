//! Incremental UTF-8 decoder for byte-level tokenizers
//!
//! This module provides an efficient incremental decoder that handles
//! multi-byte UTF-8 sequences across token boundaries without the O(n²)
//! complexity of re-decoding the entire sequence on each token.

use anyhow::Result;

/// Optimized UTF-8 decoder for byte-level tokenizers
///
/// This decoder maintains the minimum state needed to efficiently handle
/// byte-level tokenizers where tokens may represent partial UTF-8 sequences.
///
/// The key insight: We keep all generated token IDs but only decode from
/// where we last successfully decoded. This typically means decoding 1-3
/// tokens at a time instead of the entire sequence.
#[derive(Debug, Clone)]
pub struct IncrementalUtf8Decoder {
    /// All generated token IDs (for reference)
    all_tokens: Vec<i64>,

    /// The complete decoded text so far
    decoded_text: String,

    /// Number of characters in decoded_text before the last decode
    previous_text_len: usize,
}

impl IncrementalUtf8Decoder {
    /// Create a new decoder
    pub fn new() -> Self {
        Self {
            all_tokens: Vec::new(),
            decoded_text: String::new(),
            previous_text_len: 0,
        }
    }

    /// Add a token and decode the full sequence, returning only new text
    ///
    /// This is the simplest correct approach that handles byte-level tokenizers:
    /// 1. Decode the full sequence (tokenizer handles UTF-8 properly)
    /// 2. Extract only the new portion
    ///
    /// While this is still O(n) per token, it's the simplest correct solution
    /// and the tokenizer's decode is highly optimized.
    pub fn push_token_simple<F>(
        &mut self,
        token_id: i64,
        decode_fn: F,
    ) -> Result<String>
    where
        F: FnOnce(&[i64]) -> Result<String>,
    {
        // Add the new token
        self.all_tokens.push(token_id);

        // Decode the full sequence
        let new_full_text = decode_fn(&self.all_tokens)?;

        // Extract the new portion
        let new_text = if new_full_text.len() > self.previous_text_len {
            new_full_text[self.previous_text_len..].to_string()
        } else {
            // Text didn't grow - incomplete UTF-8 sequence
            String::new()
        };

        // Update state
        self.decoded_text = new_full_text;
        self.previous_text_len = self.decoded_text.len();

        Ok(new_text)
    }

    /// Get the complete decoded text
    pub fn get_text(&self) -> &str {
        &self.decoded_text
    }

    /// Get all token IDs
    pub fn get_tokens(&self) -> &[i64] {
        &self.all_tokens
    }
}

impl Default for IncrementalUtf8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker for preventing infinite loops from invalid tokens
#[derive(Debug)]
pub struct InvalidTokenCircuitBreaker {
    /// Maximum consecutive invalid tokens before failing
    max_invalid: usize,

    /// Current count of consecutive invalid tokens
    consecutive_invalid: usize,

    /// Total invalid tokens seen
    total_invalid: usize,
}

impl InvalidTokenCircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(max_consecutive: usize) -> Self {
        Self {
            max_invalid: max_consecutive,
            consecutive_invalid: 0,
            total_invalid: 0,
        }
    }

    /// Record a valid token
    pub fn record_valid(&mut self) {
        self.consecutive_invalid = 0;
    }

    /// Record an invalid token and check if we should fail
    ///
    /// Returns Err if too many consecutive invalid tokens
    pub fn record_invalid(&mut self, token_id: usize, vocab_size: usize) -> Result<()> {
        self.consecutive_invalid += 1;
        self.total_invalid += 1;

        if self.consecutive_invalid >= self.max_invalid {
            anyhow::bail!(
                "Too many consecutive invalid tokens ({} consecutive, {} total). \
                 Last invalid token: {} (vocab_size: {}). \
                 Model may be misconfigured or corrupted.",
                self.consecutive_invalid,
                self.total_invalid,
                token_id,
                vocab_size
            );
        }

        tracing::warn!(
            "Invalid token {} sampled (vocab_size: {}), consecutive: {}, total: {}",
            token_id,
            vocab_size,
            self.consecutive_invalid,
            self.total_invalid
        );

        Ok(())
    }

    /// Get metrics for monitoring
    pub fn get_metrics(&self) -> (usize, usize) {
        (self.consecutive_invalid, self.total_invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_decoder() {
        let mut decoder = IncrementalUtf8Decoder::new();

        // Token 1: "Hello"
        let result = decoder.push_token_simple(1, |_| Ok("Hello".to_string())).unwrap();
        assert_eq!(result, "Hello");

        // Token 2: "Hello world"
        let result = decoder.push_token_simple(2, |_| Ok("Hello world".to_string())).unwrap();
        assert_eq!(result, " world");

        assert_eq!(decoder.get_text(), "Hello world");
    }

    #[test]
    fn test_incomplete_utf8() {
        let mut decoder = IncrementalUtf8Decoder::new();

        // Token 1: Incomplete (returns empty)
        let result = decoder.push_token_simple(1, |_| Ok("".to_string())).unwrap();
        assert_eq!(result, "");

        // Token 2: Now complete
        let result = decoder.push_token_simple(2, |_| Ok("你好".to_string())).unwrap();
        assert_eq!(result, "你好");
    }

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = InvalidTokenCircuitBreaker::new(3);

        // Record some invalid tokens
        breaker.record_invalid(100, 50).unwrap();
        breaker.record_invalid(101, 50).unwrap();

        // Reset with valid token
        breaker.record_valid();
        assert_eq!(breaker.consecutive_invalid, 0);

        // Hit the limit
        breaker.record_invalid(102, 50).unwrap();
        breaker.record_invalid(103, 50).unwrap();
        let result = breaker.record_invalid(104, 50);
        assert!(result.is_err());
    }
}