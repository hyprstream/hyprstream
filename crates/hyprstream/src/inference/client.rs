//! Inference client trait and error types.
//!
//! This module defines the transport-agnostic interface for inference services.
//! Implementations can be in-process (channels) or remote (gRPC, etc.).

use crate::config::{GenerationRequest, GenerationResult, ModelInfo};
use crate::lora::LoRAConfig;
use async_trait::async_trait;
use std::path::Path;
use thiserror::Error;

use super::response::StreamHandle;

/// Transport-agnostic inference client trait.
///
/// Implementations can be in-process (channels) or remote (gRPC, etc.).
/// All methods return owned data to avoid lifetime issues across transports.
#[async_trait]
pub trait InferenceClient: Send + Sync {
    // === Generation ===

    /// Generate text (non-streaming, blocking until complete).
    async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult, InferenceError>;

    /// Start streaming generation, returns handle to receive text chunks.
    async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<StreamHandle, InferenceError>;

    // === Model Info ===

    /// Get information about the loaded model.
    async fn model_info(&self) -> Result<ModelInfo, InferenceError>;

    /// Check if model is loaded and ready.
    async fn is_ready(&self) -> Result<bool, InferenceError>;

    /// Apply chat template to format messages for generation.
    async fn apply_chat_template(
        &self,
        messages: &[crate::runtime::template_engine::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, InferenceError>;

    // === LoRA Operations ===

    /// Initialize LoRA adapter structure for training.
    async fn create_lora(&self, config: LoRAConfig) -> Result<(), InferenceError>;

    /// Load LoRA adapter weights from file.
    async fn load_lora(&self, path: &Path) -> Result<(), InferenceError>;

    /// Save LoRA adapter weights to file.
    async fn save_lora(&self, path: &Path) -> Result<(), InferenceError>;

    /// Unload current LoRA adapter.
    async fn unload_lora(&self) -> Result<(), InferenceError>;

    /// Check if LoRA is loaded.
    async fn has_lora(&self) -> Result<bool, InferenceError>;

    // === Session Management (KV Cache) ===

    /// Set active session for KV cache isolation.
    async fn set_session(&self, session_id: String) -> Result<(), InferenceError>;

    /// Clear current session's KV cache.
    async fn clear_session(&self) -> Result<(), InferenceError>;

    /// Release a session's KV cache.
    async fn release_session(&self, session_id: &str) -> Result<(), InferenceError>;

    // === Health ===

    /// Check service health.
    async fn health_check(&self) -> Result<(), InferenceError>;

    /// Graceful shutdown signal.
    async fn shutdown(&self) -> Result<(), InferenceError>;
}

/// Inference service error type.
#[derive(Debug, Error, PartialEq)]
pub enum InferenceError {
    /// Model loading failed.
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    /// Generation failed.
    #[error("Generation failed: {0}")]
    Generation(String),

    /// LoRA operation failed.
    #[error("LoRA operation failed: {0}")]
    LoRA(String),

    /// Session management failed.
    #[error("Session operation failed: {0}")]
    Session(String),

    /// Service communication failed (channel closed, send failed, etc.).
    #[error("Service communication failed: {0}")]
    Channel(String),

    /// Service is unavailable (not started, shutdown, etc.).
    #[error("Service unavailable")]
    Unavailable,

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl InferenceError {
    /// Check if this error suggests retrying the operation.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Channel(_) | Self::Unavailable)
    }

    /// Create a channel error.
    pub fn channel<S: Into<String>>(msg: S) -> Self {
        Self::Channel(msg.into())
    }

    /// Create an internal error.
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        Self::Internal(msg.into())
    }
}

impl From<anyhow::Error> for InferenceError {
    fn from(e: anyhow::Error) -> Self {
        Self::Internal(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_error_is_retryable() {
        // Channel errors should be retryable
        let err = InferenceError::channel("test");
        assert!(err.is_retryable());

        // Unavailable should be retryable
        let err = InferenceError::Unavailable;
        assert!(err.is_retryable());

        // Generation errors should not be retryable
        let err = InferenceError::Generation("failed".into());
        assert!(!err.is_retryable());
    }
}
