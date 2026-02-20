//! RPC types: re-exports and inference-specific stream payload types.
//!
//! Generated types are used directly (including `StreamInfo` with
//! `[u8; 32]` for `server_pubkey` via `$fixedSize(32)` annotation).

use anyhow::Result;

// ============================================================================
// StreamInfo — re-exported from generated inference_client
// ============================================================================

/// Re-export generated StreamInfo as the canonical StreamInfo type.
///
/// The `$fixedSize(32)` annotation on `serverPubkey` in streaming.capnp
/// generates `server_pubkey: [u8; 32]` with validated FromCapnp deserialization.
pub use crate::services::generated::inference_client::StreamInfo;

// ============================================================================
// StreamBlock Types - Re-exported from hyprstream-rpc
// ============================================================================

// Re-export generic streaming types from hyprstream-rpc
pub use hyprstream_rpc::streaming::{
    BatchingConfig,
    StreamBuilder,
    StreamFrames,
    StreamHandle,
    StreamHmacState,
    StreamPayload,
    StreamPayloadData,
    StreamVerifier,
};

// ============================================================================
// Inference-Specific Stream Types
// ============================================================================

/// Inference-specific completion metadata (serialized into StreamPayload.complete).
///
/// This is the application-layer completion data for inference streams.
/// Serialized as JSON for simplicity and debuggability.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceComplete {
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
    pub finish_reason: String,
    // Prefill metrics
    pub prefill_tokens: usize,
    pub prefill_time_ms: u64,
    pub prefill_tokens_per_sec: f32,
    // Inference metrics
    pub inference_tokens: usize,
    pub inference_time_ms: u64,
    pub inference_tokens_per_sec: f32,
    pub inference_tokens_per_sec_ema: f32,
    // Optional quality metrics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avg_entropy: Option<f32>,
    // Online training (TTT) metrics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttt_metrics: Option<crate::config::TTTMetrics>,
}

impl InferenceComplete {
    /// Create an empty/default instance for fallback when deserialization fails.
    pub fn empty() -> Self {
        Self {
            tokens_generated: 0,
            generation_time_ms: 0,
            tokens_per_second: 0.0,
            finish_reason: "unknown".to_owned(),
            prefill_tokens: 0,
            prefill_time_ms: 0,
            prefill_tokens_per_sec: 0.0,
            inference_tokens: 0,
            inference_time_ms: 0,
            inference_tokens_per_sec: 0.0,
            inference_tokens_per_sec_ema: 0.0,
            perplexity: None,
            avg_entropy: None,
            ttt_metrics: None,
        }
    }

    /// Serialize to bytes for StreamPayload.complete.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from StreamPayload.complete bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| anyhow::anyhow!("Failed to parse InferenceComplete: {}", e))
    }
}

impl From<&crate::runtime::GenerationStats> for InferenceComplete {
    fn from(stats: &crate::runtime::GenerationStats) -> Self {
        let finish_reason = match &stats.finish_reason {
            Some(crate::config::FinishReason::MaxTokens) => "length",
            Some(crate::config::FinishReason::EndOfSequence) => "eos",
            Some(crate::config::FinishReason::Stop | crate::config::FinishReason::StopToken(_)) => "stop",
            Some(crate::config::FinishReason::Error(_)) => "error",
            None => "unknown",
        };
        Self {
            tokens_generated: stats.tokens_generated,
            generation_time_ms: stats.generation_time_ms,
            tokens_per_second: stats.tokens_per_second,
            finish_reason: finish_reason.to_owned(),
            prefill_tokens: stats.prefill_tokens,
            prefill_time_ms: stats.prefill_time_ms,
            prefill_tokens_per_sec: stats.prefill_tokens_per_sec,
            inference_tokens: stats.inference_tokens,
            inference_time_ms: stats.inference_time_ms,
            inference_tokens_per_sec: stats.inference_tokens_per_sec,
            inference_tokens_per_sec_ema: stats.inference_tokens_per_sec_ema,
            perplexity: stats.quality_metrics.as_ref().map(|m| m.perplexity),
            avg_entropy: stats.quality_metrics.as_ref().map(|m| m.avg_entropy),
            ttt_metrics: None,  // Attached by execute_stream in InferenceService
        }
    }
}

/// Inference-specific stream payload (parsed from generic StreamPayload).
///
/// Provides typed access to inference stream data:
/// - Token: UTF-8 text token
/// - Error: Error message
/// - Complete: Generation statistics
///
/// Note: Stream identity comes from the DH-derived topic, not from payload fields.
/// The topic cryptographically binds the stream to the DH key exchange.
#[derive(Debug, Clone)]
pub enum InferenceStreamPayload {
    /// UTF-8 text token
    Token(String),
    /// Error during streaming
    Error(String),
    /// Completion with generation statistics
    Complete(InferenceComplete),
}

/// Extension trait to convert generic StreamPayload to inference-specific payload.
pub trait StreamPayloadExt {
    /// Convert generic payload to inference-specific payload.
    ///
    /// Interprets Data as UTF-8 text tokens and Complete as InferenceComplete.
    fn to_inference(self) -> Result<InferenceStreamPayload>;
}

impl StreamPayloadExt for StreamPayload {
    fn to_inference(self) -> Result<InferenceStreamPayload> {
        match self {
            StreamPayload::Data(data) => {
                let text = String::from_utf8(data)
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in token: {}", e))?;
                Ok(InferenceStreamPayload::Token(text))
            }
            StreamPayload::Error(message) => {
                Ok(InferenceStreamPayload::Error(message))
            }
            StreamPayload::Complete(data) => {
                let stats = InferenceComplete::from_bytes(&data)?;
                Ok(InferenceStreamPayload::Complete(stats))
            }
        }
    }
}
