//! Inference API for applying LoRA adapters to base models


use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::Result;

pub mod model_loader;
pub mod inference_engine;
// LoRA fusion moved to lora module
pub mod block_engine;
pub mod scheduler;

// Re-export unified config system
pub use crate::config::{HyprConfig, GenerationRequest, GenerationResult, ModelInfo};

pub use model_loader::{ModelLoader, BaseModelHandle};
pub use inference_engine::{InferenceEngine, InferenceEngineStats};
// pub use lora_fusion::{LoRAFusion, FusionStrategy}; // Module removed
pub use block_engine::{BlockEngine, BlockEngineStats, AllocStatus};
pub use scheduler::{HyprScheduler, SchedulerConfig, SchedulerOutput, SchedulerStats};

/// Inference session with active LoRA adapters
#[derive(Debug, Clone)]
pub struct InferenceSession {
    pub session_id: String,
    pub model_name: String,
    pub active_adapters: Vec<String>,
    pub adapter_weights: HashMap<String, f32>, // Mixing weights for multiple adapters
    pub created_at: i64,
    pub last_used: i64,
}

/// Inference statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_latency_ms: f64,
    pub active_sessions: u64,
    pub adapter_cache_hits: u64,
    pub adapter_cache_misses: u64,
    pub gpu_utilization: f32,
    pub memory_usage_mb: u64,
}



/// Input for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceInput {
    pub prompt: Option<String>,
    pub input_ids: Option<Vec<i64>>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
}

/// Output from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub text: String,
    pub tokens: Vec<i64>,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub adapter_contribution: HashMap<String, f32>,
}

/// Single token for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceToken {
    pub token: String,
    pub token_id: i64,
    pub logprob: f32,
    pub timestamp_ms: i64,
}

pub struct FusedAdapterWeights {
    pub weights: std::collections::HashMap<String, ()>, // Placeholder
    pub fusion_metadata: FusionMetadata,
}

/// Metadata about adapter fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    pub num_adapters: usize,
    pub total_sparse_weights: usize,
    pub fusion_strategy: String,
    pub timestamp: i64,
}

/// Sparse weight update for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseWeightUpdate {
    pub adapter_id: String,
    pub layer_name: String,
    pub coordinates: Vec<(i32, i32, i32)>,
    pub values: Vec<f32>,
    pub timestamp: i64,
}

fn parse_weight_key(_key: &str) -> Option<(usize, (i32, i32))> {
    // Parse keys like "layer.0.weight[100,200]"
    // This is a simplified parser - implement based on actual key format
    None
}

// Re-export uuid if not already available
use uuid;