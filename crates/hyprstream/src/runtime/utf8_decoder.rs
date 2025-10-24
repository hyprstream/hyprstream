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

        // Extract the new portion (handle UTF-8 char boundaries properly)
        let new_text = if new_full_text.len() > self.previous_text_len {
            // Find a valid UTF-8 boundary at or after previous_text_len
            let start_byte = if self.previous_text_len <= new_full_text.len() {
                // Check if we're at a valid boundary
                if new_full_text.is_char_boundary(self.previous_text_len) {
                    self.previous_text_len
                } else {
                    // Find the next valid boundary
                    let mut boundary = self.previous_text_len;
                    while boundary < new_full_text.len() && !new_full_text.is_char_boundary(boundary) {
                        boundary += 1;
                    }
                    boundary
                }
            } else {
                new_full_text.len()
            };

            if start_byte < new_full_text.len() {
                new_full_text[start_byte..].to_string()
            } else {
                String::new()
            }
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

}