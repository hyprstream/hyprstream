//! Trait-based tokenizer configuration for different model architectures
//!
//! This module provides a clean abstraction for model-specific tokenizer configuration,
//! allowing each model architecture to customize tokenizer behavior without polluting
//! the global runtime code.

use anyhow::Result;
use tokenizers::{AddedToken, Tokenizer};

/// Trait for model-specific tokenizer configuration
///
/// Each model architecture can implement this trait to provide custom
/// tokenizer configuration, including adding special tokens, configuring
/// post-processors, and handling vocabulary mismatches.
pub trait TokenizerConfig: Send + Sync {
    /// Configure the tokenizer for this specific model architecture
    ///
    /// This method is called after loading the base tokenizer from file,
    /// allowing the model to add special tokens, configure post-processing,
    /// or make other model-specific adjustments.
    ///
    /// # Arguments
    /// * `tokenizer` - The tokenizer to configure (mutable reference)
    /// * `model_vocab_size` - The vocabulary size from the model's config
    ///
    /// # Returns
    /// * `Result<()>` - Success or error during configuration
    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, model_vocab_size: usize) -> Result<()>;

    /// Get special tokens that should be added for this model
    ///
    /// This is called by `configure_tokenizer` but can also be used
    /// independently if needed.
    ///
    /// # Arguments
    /// * `model_vocab_size` - Total vocabulary size expected by the model
    /// * `tokenizer_vocab_size` - Current size of the tokenizer vocabulary
    ///
    /// # Returns
    /// * Vector of `AddedToken` objects to add to the tokenizer
    fn get_special_tokens(&self, _model_vocab_size: usize, _tokenizer_vocab_size: usize) -> Vec<AddedToken> {
        Vec::new() // Default: no special tokens
    }

    /// Check if a token ID represents a special token for this model
    ///
    /// # Arguments
    /// * `token_id` - The token ID to check
    /// * `vocab_size` - The base vocabulary size (without special tokens)
    ///
    /// # Returns
    /// * `true` if this is a special token, `false` otherwise
    fn is_special_token(&self, _token_id: usize, _vocab_size: usize) -> bool {
        false // Default: no special tokens
    }

    /// Get display representation for out-of-vocabulary or special tokens
    ///
    /// This is useful for debugging or when special tokens need custom display.
    ///
    /// # Arguments
    /// * `token_id` - The token ID to format
    /// * `vocab_size` - The base vocabulary size
    ///
    /// # Returns
    /// * `Some(String)` with the formatted token, or `None` if no special formatting
    fn format_special_token(&self, _token_id: usize, _vocab_size: usize) -> Option<String> {
        None // Default: no special formatting
    }

    /// Decode a token that might be out of vocabulary
    ///
    /// This method is called when a token needs to be decoded. It can handle
    /// tokens that are beyond the tokenizer's vocabulary by returning a special
    /// representation.
    ///
    /// # Arguments
    /// * `tokenizer` - The tokenizer to use for decoding
    /// * `token_id` - The token ID to decode
    ///
    /// # Returns
    /// * `Ok(Some(String))` - Successfully decoded token or special representation
    /// * `Ok(None)` - Token should be skipped
    /// * `Err` - Decoding error
    fn decode_token(&self, tokenizer: &Tokenizer, token_id: u32) -> Result<Option<String>> {
        // Default: try normal decoding
        match tokenizer.decode(&[token_id], false) {
            Ok(text) => Ok(Some(text)),
            Err(_) => {
                // If decoding fails, check if it's a special token
                let vocab_size = tokenizer.get_vocab_size(true);
                if let Some(special) = self.format_special_token(token_id as usize, vocab_size) {
                    Ok(Some(special))
                } else {
                    Ok(None) // Skip unknown tokens
                }
            }
        }
    }
}

// ============================================================================
// Model-specific implementations
// ============================================================================

/// Qwen model tokenizer configuration
///
/// Qwen models have a vocabulary mismatch between the model (151,936 tokens)
/// and the distributed tokenizer (151,643 tokens). The missing 293 tokens are
/// special tokens following the <|extra_N|> convention.
pub struct QwenTokenizerConfig;

impl TokenizerConfig for QwenTokenizerConfig {
    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, model_vocab_size: usize) -> Result<()> {
        let current_vocab_size = tokenizer.get_vocab_size(true);

        // Qwen models have special tokens beyond the base vocabulary
        if model_vocab_size > current_vocab_size {
            tracing::info!(
                "Qwen model vocabulary mismatch detected: model has {} tokens, tokenizer has {} tokens",
                model_vocab_size, current_vocab_size
            );

            let missing_count = model_vocab_size - current_vocab_size;

            // Create special tokens for the missing vocabulary entries
            // Qwen uses <|extra_N|> convention for additional tokens
            let mut tokens_to_add = Vec::with_capacity(missing_count);
            for i in 0..missing_count {
                // Create as special tokens (special=true)
                // They ARE special tokens in the model's vocabulary
                let token = AddedToken::from(format!("<|extra_{i}|>"), true)
                    .single_word(true)  // Treat as single word
                    .lstrip(false)
                    .rstrip(false)
                    .normalized(false);
                tokens_to_add.push(token);
            }

            // Add as special tokens to the tokenizer
            // They will be properly tracked as special tokens with correct IDs
            let added = tokenizer.add_special_tokens(&tokens_to_add);

            tracing::info!(
                "Added {} special tokens (<|extra_0|> through <|extra_{}|>) to Qwen tokenizer",
                added, missing_count - 1
            );

            // Verify they were actually added
            let new_size = tokenizer.get_vocab_size(true);
            if new_size == model_vocab_size {
                tracing::info!("✅ Vocabulary successfully expanded from {} to {}", current_vocab_size, new_size);
            } else if new_size > current_vocab_size {
                tracing::info!(
                    "✅ Vocabulary expanded from {} to {} (added {} tokens)",
                    current_vocab_size, new_size, new_size - current_vocab_size
                );
            } else {
                tracing::warn!(
                    "⚠️ Failed to expand vocabulary: size remains at {} (expected {})",
                    new_size, model_vocab_size
                );
            }
        }

        Ok(())
    }

    fn get_special_tokens(&self, model_vocab_size: usize, tokenizer_vocab_size: usize) -> Vec<AddedToken> {
        if model_vocab_size <= tokenizer_vocab_size {
            return Vec::new();
        }

        let missing_count = model_vocab_size - tokenizer_vocab_size;
        let mut tokens = Vec::with_capacity(missing_count);

        // Qwen uses <|extra_N|> convention for additional tokens
        // These include both reserved tokens and control tokens
        for i in 0..missing_count {
            let token = AddedToken::from(format!("<|extra_{i}|>"), true)
                .single_word(false)
                .lstrip(false)
                .rstrip(false)
                .normalized(false);
            tokens.push(token);
        }

        tokens
    }

    fn is_special_token(&self, token_id: usize, vocab_size: usize) -> bool {
        // Tokens beyond the base vocabulary are special tokens
        token_id >= vocab_size
    }

    fn format_special_token(&self, token_id: usize, vocab_size: usize) -> Option<String> {
        if token_id >= vocab_size {
            // Format as <|extra_N|> for display/debugging
            Some(format!("<|extra_{}|>", token_id - vocab_size))
        } else {
            None
        }
    }

    fn decode_token(&self, tokenizer: &Tokenizer, token_id: u32) -> Result<Option<String>> {
        // First try normal decoding (including special tokens)
        // Since we added <|extra_N|> as special tokens, they should decode properly
        match tokenizer.decode(&[token_id], false) {
            Ok(text) => Ok(Some(text)),
            Err(_) => {
                // If decoding fails, check if it's an out-of-bounds token
                let vocab_size = tokenizer.get_vocab_size(true);
                if (token_id as usize) >= vocab_size {
                    // Token is beyond even the expanded vocabulary
                    // This shouldn't happen if we configured the tokenizer correctly
                    tracing::warn!(
                        "Token {} is beyond expanded vocabulary size {}, formatting as <|extra_N|>",
                        token_id, vocab_size
                    );
                    let extra_idx = (token_id as usize) - vocab_size;
                    Ok(Some(format!("<|extra_{extra_idx}|>")))
                } else {
                    tracing::warn!("Failed to decode token {} within vocab", token_id);
                    Ok(None) // Skip tokens that can't be decoded
                }
            }
        }
    }
}

/// Llama model tokenizer configuration
///
/// Llama models typically have matching vocabulary sizes between
/// the model and tokenizer, requiring no special configuration.
pub struct LlamaTokenizerConfig;

impl TokenizerConfig for LlamaTokenizerConfig {
    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, model_vocab_size: usize) -> Result<()> {
        let current_vocab_size = tokenizer.get_vocab_size(true);

        if model_vocab_size != current_vocab_size {
            tracing::debug!(
                "Llama model vocab size ({}) differs from tokenizer vocab size ({}), but this is typically OK",
                model_vocab_size, current_vocab_size
            );
        }

        // Llama tokenizers typically don't need special configuration
        // The tokenizer.json should already be complete
        Ok(())
    }
}

/// Gemma model tokenizer configuration
pub struct GemmaTokenizerConfig;

impl TokenizerConfig for GemmaTokenizerConfig {
    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, model_vocab_size: usize) -> Result<()> {
        let current_vocab_size = tokenizer.get_vocab_size(true);

        if model_vocab_size != current_vocab_size {
            tracing::debug!(
                "Gemma model vocab size ({}) differs from tokenizer vocab size ({})",
                model_vocab_size, current_vocab_size
            );
        }

        // Gemma models may have their own special tokens or configuration
        // Add Gemma-specific setup here if needed
        Ok(())
    }
}

/// Default tokenizer configuration for unknown model types
///
/// This provides a fallback that doesn't modify the tokenizer,
/// suitable for models that don't require special handling.
pub struct DefaultTokenizerConfig;

impl TokenizerConfig for DefaultTokenizerConfig {
    fn configure_tokenizer(&self, _tokenizer: &mut Tokenizer, _model_vocab_size: usize) -> Result<()> {
        // No special configuration for unknown models
        tracing::debug!("Using default tokenizer configuration (no modifications)");
        Ok(())
    }
}