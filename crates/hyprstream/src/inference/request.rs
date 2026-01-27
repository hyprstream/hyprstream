//! Request types for the inference service.
//!
//! These types define the message format for communication between
//! LocalInferenceClient and LocalInferenceService via channels.

use crate::config::{GenerationRequest, GenerationResult, ModelInfo};
use crate::lora::LoRAConfig;
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};

use super::client::InferenceError;
use super::response::StreamStats;

/// Request types for the inference service.
///
/// All variants include a oneshot reply channel for returning results.
/// For streaming generation, we use mpsc channels for chunks.
pub enum InferenceRequest {
    // === Generation ===

    /// Non-streaming generation request.
    Generate {
        request: GenerationRequest,
        reply: oneshot::Sender<Result<GenerationResult, InferenceError>>,
    },

    /// Streaming generation request.
    ///
    /// Unlike other requests, this uses an mpsc channel for streaming chunks
    /// and a separate oneshot for final stats.
    GenerateStream {
        request: GenerationRequest,
        /// Channel to send text chunks as they're generated.
        chunk_sender: mpsc::Sender<Result<String, InferenceError>>,
        /// Channel to send final statistics after generation completes.
        stats_sender: oneshot::Sender<StreamStats>,
        /// Reply channel for acknowledging request receipt or early errors.
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    // === Model Info ===

    /// Get model information.
    ModelInfo {
        reply: oneshot::Sender<Result<ModelInfo, InferenceError>>,
    },

    /// Check if model is ready.
    IsReady {
        reply: oneshot::Sender<Result<bool, InferenceError>>,
    },

    /// Apply chat template to messages.
    ApplyChatTemplate {
        messages: Vec<crate::runtime::template_engine::ChatMessage>,
        add_generation_prompt: bool,
        reply: oneshot::Sender<Result<String, InferenceError>>,
    },

    // === LoRA Operations ===

    /// Create LoRA adapter structure.
    CreateLora {
        config: LoRAConfig,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Load LoRA adapter from file.
    LoadLora {
        path: PathBuf,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Save LoRA adapter to file.
    SaveLora {
        path: PathBuf,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Unload current LoRA adapter.
    UnloadLora {
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Check if LoRA is loaded.
    HasLora {
        reply: oneshot::Sender<Result<bool, InferenceError>>,
    },

    // === Session Management ===

    /// Set active session for KV cache isolation.
    SetSession {
        session_id: String,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Clear current session's KV cache.
    ClearSession {
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Release a session's KV cache.
    ReleaseSession {
        session_id: String,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    // === Health ===

    /// Health check request.
    HealthCheck {
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },

    /// Shutdown request.
    Shutdown {
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },
}

impl std::fmt::Debug for InferenceRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate { request, .. } => {
                write!(f, "InferenceRequest::Generate {{ prompt_len: {} }}",
                       request.prompt.len())
            }
            Self::GenerateStream { request, .. } => {
                write!(f, "InferenceRequest::GenerateStream {{ prompt_len: {} }}",
                       request.prompt.len())
            }
            Self::ModelInfo { .. } => write!(f, "InferenceRequest::ModelInfo"),
            Self::IsReady { .. } => write!(f, "InferenceRequest::IsReady"),
            Self::ApplyChatTemplate { messages, .. } => {
                write!(f, "InferenceRequest::ApplyChatTemplate {{ messages: {} }}", messages.len())
            }
            Self::CreateLora { config, .. } => {
                write!(f, "InferenceRequest::CreateLora {{ rank: {} }}", config.rank)
            }
            Self::LoadLora { path, .. } => {
                write!(f, "InferenceRequest::LoadLora {{ path: {path:?} }}")
            }
            Self::SaveLora { path, .. } => {
                write!(f, "InferenceRequest::SaveLora {{ path: {path:?} }}")
            }
            Self::UnloadLora { .. } => write!(f, "InferenceRequest::UnloadLora"),
            Self::HasLora { .. } => write!(f, "InferenceRequest::HasLora"),
            Self::SetSession { session_id, .. } => {
                write!(f, "InferenceRequest::SetSession {{ session_id: {session_id:?} }}")
            }
            Self::ClearSession { .. } => write!(f, "InferenceRequest::ClearSession"),
            Self::ReleaseSession { session_id, .. } => {
                write!(f, "InferenceRequest::ReleaseSession {{ session_id: {session_id:?} }}")
            }
            Self::HealthCheck { .. } => write!(f, "InferenceRequest::HealthCheck"),
            Self::Shutdown { .. } => write!(f, "InferenceRequest::Shutdown"),
        }
    }
}
