//! REST API for creating and managing sparse auto-regressive LoRA training layers

use axum::Router;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

pub mod model_registry;
pub mod model_storage;
pub mod model_downloader;
pub mod git_downloader;
pub mod adapter_storage;
pub mod model_sharing;
pub mod openai_compat;
pub mod training_service;

use adapter_storage::{AdapterStorage, AdapterId, AdapterConfig};
use training_service::{TrainingService, TrainingConfig};

/// Main API server state
#[derive(Clone)]
pub struct ApiState {
    /// Adapter storage manager
    adapter_storage: Arc<AdapterStorage>,

    /// Training service for auto-regressive learning
    training_service: Arc<TrainingService>,

    /// Active endpoints mapping
    endpoints: Arc<RwLock<HashMap<String, LoRAEndpoint>>>,
}

impl ApiState {
    /// Get model information with architecture configuration
    /// This should eventually integrate with the model registry to get actual model configs
    async fn get_model_info(&self, model_name: &str) -> Result<ModelArchitectureInfo, String> {
        // For now, return an error to force proper implementation
        // This ensures we don't create LoRA adapters without proper model configuration
        Err(format!(
            "Model configuration lookup not implemented. Please ensure model '{}' is registered in the model registry with proper architecture configuration.",
            model_name
        ))
    }
}

#[derive(Debug, Clone)]
struct ModelArchitectureInfo {
    hidden_size: usize,
    vocab_size: usize,
    architecture: crate::runtime::architectures::ModelArchitecture,
}

/// LoRA endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAEndpoint {
    pub adapter_id: AdapterId,
    pub base_path: String,
    pub created_at: i64,
    pub config: LoRAConfig,
    pub training_enabled: bool,
    pub auto_regressive: bool,
}

/// Configuration for creating a new LoRA layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateLoRARequest {
    /// Optional name for the LoRA layer
    pub name: Option<String>,
    
    /// Model to base this on (e.g., "qwen3-1.7b")
    pub base_model: String,
    
    /// LoRA configuration
    pub config: LoRAConfig,
    
    /// Enable auto-regressive training
    pub auto_regressive: bool,
    
    /// Training configuration
    pub training_config: Option<TrainingConfig>,
}

/// LoRA layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the LoRA adaptation
    pub rank: usize,
    
    /// Alpha scaling parameter
    pub alpha: f32,
    
    /// Dropout rate for training
    pub dropout: f32,
    
    /// Target modules (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
    
    /// Sparsity ratio (0.0 to 1.0)
    pub sparsity_ratio: f32,
    
    /// Use neural compression
    pub use_neural_compression: bool,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            sparsity_ratio: 0.99,
            use_neural_compression: true,
        }
    }
}

/// Response when creating a new LoRA layer
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateLoRAResponse {
    pub lora_id: String,
    pub endpoint: String,
    pub openai_base_url: String,
    pub status: String,
}

/// Create the main API router
pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .with_state(state)
}














/// Training sample for auto-regressive learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub input: String,
    pub output: String,
    pub timestamp: i64,
}

/// Training status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub is_training: bool,
    pub total_samples_processed: u64,
    pub current_loss: f32,
    pub learning_rate: f32,
    pub last_update: i64,
}

/// LoRA statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAStats {
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_latency_ms: f64,
    pub sparsity_ratio: f32,
    pub memory_usage_mb: u64,
    pub compression_ratio: f32,
}



