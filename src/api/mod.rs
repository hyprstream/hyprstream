//! REST API for creating and managing sparse auto-regressive LoRA training layers

use axum::{
    Router,
    extract::{Path, State, Json},
    response::Json as JsonResponse,
    http::StatusCode,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

pub mod lora_registry;
pub mod model_registry;
pub mod model_storage;
pub mod model_downloader;
pub mod huggingface;
pub mod openai_compat;
pub mod training_service;

use lora_registry::{LoRARegistry, LoRALayer, LoRAId};
use training_service::{TrainingService, TrainingConfig};

/// Main API server state
#[derive(Clone)]
pub struct ApiState {
    /// Registry of all LoRA layers
    lora_registry: Arc<LoRARegistry>,
    
    /// Training service for auto-regressive learning
    training_service: Arc<TrainingService>,
    
    /// VDB storage backend (only available with VDB feature)
    
    vdb_storage: Arc<crate::storage::vdb::hardware_accelerated::HardwareVDBStorage>,
    
    
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
    pub lora_id: LoRAId,
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
    
    /// Sparsity ratio (0.99 for 99% sparse)
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

/// Create a new LoRA layer
#[allow(dead_code)]
async fn create_lora_layer(
    State(state): State<ApiState>,
    Json(request): Json<CreateLoRARequest>,
) -> Result<JsonResponse<CreateLoRAResponse>, StatusCode> {
    let lora_id = LoRAId::new();
    
    // Get model info from model registry first
    let model_info = state.get_model_info(&request.base_model).await
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    // Create the LoRA layer in the registry
    let layer = LoRALayer {
        id: lora_id.clone(),
        name: request.name.unwrap_or_else(|| format!("lora_{}", lora_id.to_string()[..8].to_string())),
        base_model: request.base_model,
        config: request.config.clone(),
        created_at: chrono::Utc::now().timestamp(),
        updated_at: chrono::Utc::now().timestamp(),
        training_enabled: request.auto_regressive,
        total_tokens_trained: 0,
        sparsity_ratio: request.config.sparsity_ratio,
    };
    
    // Register in the registry
    state.lora_registry.register(layer.clone()).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Create sparse adapter configuration using model registry information
    let adapter_config = crate::adapters::sparse_lora::SparseLoRAConfig {
        in_features: model_info.hidden_size,
        out_features: model_info.hidden_size,
        rank: request.config.rank,
        sparsity: request.config.sparsity_ratio,
        learning_rate: request.training_config.as_ref()
            .map(|tc| tc.learning_rate)
            .unwrap_or(1e-4),
        dropout: request.config.dropout,
        alpha: request.config.alpha,
        bias: false,
        target_modules: request.config.target_modules.clone(),
        init_method: crate::adapters::sparse_lora::InitMethod::Random,
        sparsity_threshold: 1.0 - request.config.sparsity_ratio,
        enable_gradient_checkpointing: true,
        mixed_precision: true,
    };
    
    let adapter = crate::adapters::sparse_lora::SparseLoRAAdapter::new(adapter_config);
    adapter.initialize_random().await;
    
    // Store in VDB with neural compression if enabled
    if request.config.use_neural_compression {
        
        { 
            state.vdb_storage.store_adapter_neural_compressed(&lora_id.to_string(), &adapter).await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        
        { 
            // VDB feature not enabled, skip storage
        }
    } else {
        
        { 
            state.vdb_storage.store_adapter_accelerated(&lora_id.to_string(), &adapter).await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        
        { 
            // VDB feature not enabled, skip storage
        }
    }
    
    // Register endpoint
    let endpoint = LoRAEndpoint {
        lora_id: lora_id.clone(),
        base_path: format!("/v1/inference/{}", lora_id),
        created_at: chrono::Utc::now().timestamp(),
        config: request.config,
        training_enabled: request.auto_regressive,
        auto_regressive: request.auto_regressive,
    };
    
    let mut endpoints = state.endpoints.write().await;
    endpoints.insert(lora_id.to_string(), endpoint);
    
    // Start auto-regressive training if enabled
    if request.auto_regressive {
        if let Some(training_config) = request.training_config {
            state.training_service.start_auto_training(
                &lora_id.to_string(),
                training_config,
            ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
    }
    
    Ok(JsonResponse(CreateLoRAResponse {
        lora_id: lora_id.to_string(),
        endpoint: format!("/v1/inference/{}", lora_id),
        openai_base_url: format!("/v1/inference/{}", lora_id),
        status: "created".to_string(),
    }))
}

/// List all LoRA layers
#[allow(dead_code)]
async fn list_lora_layers(
    State(state): State<ApiState>,
) -> Result<JsonResponse<Vec<LoRALayer>>, StatusCode> {
    let layers = state.lora_registry.list_all().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(JsonResponse(layers))
}

/// Get information about a specific LoRA layer
#[allow(dead_code)]
async fn get_lora_info(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<LoRALayer>, StatusCode> {
    let layer = state.lora_registry.get_by_id_or_name(&lora_id).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(JsonResponse(layer))
}

/// Delete a LoRA layer
#[allow(dead_code)]
async fn delete_lora_layer(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    // Stop training if active
    let _ = state.training_service.stop_auto_training(&lora_id).await;
    
    // Remove from registry
    state.lora_registry.unregister_by_name(&lora_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Remove from VDB storage
    
    { 
        state.vdb_storage.remove_adapter(&lora_id).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    
    { 
        // VDB feature not enabled, nothing to remove
    }
    
    // Remove endpoint
    let mut endpoints = state.endpoints.write().await;
    endpoints.remove(&lora_id);
    
    Ok(JsonResponse(serde_json::json!({
        "status": "deleted",
        "lora_id": lora_id
    })))
}

/// OpenAI-compatible chat completions endpoint
#[allow(dead_code)]
async fn openai_chat_completions(
    State(_state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(_request): Json<openai_compat::ChatCompletionRequest>,
) -> Result<JsonResponse<openai_compat::ChatCompletionResponse>, StatusCode> {
    tracing::error!("Chat completions not implemented for LoRA {}", lora_id);
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// OpenAI-compatible completions endpoint
#[allow(dead_code)]
async fn openai_completions(
    State(_state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(_request): Json<openai_compat::CompletionRequest>,
) -> Result<JsonResponse<openai_compat::CompletionResponse>, StatusCode> {
    tracing::error!("Text completions not implemented for LoRA {}", lora_id);
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// OpenAI-compatible embeddings endpoint
#[allow(dead_code)]
async fn openai_embeddings(
    State(_state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(_request): Json<openai_compat::EmbeddingRequest>,
) -> Result<JsonResponse<openai_compat::EmbeddingResponse>, StatusCode> {
    tracing::error!("Embeddings not implemented for LoRA {}", lora_id);
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// List models (OpenAI-compatible)
#[allow(dead_code)]
async fn openai_list_models(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<openai_compat::ListModelsResponse>, StatusCode> {
    let lora_id = lora_id.parse::<LoRAId>().map_err(|_| StatusCode::BAD_REQUEST)?;
    let layer = state.lora_registry.get(&lora_id).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    
    let model = openai_compat::Model {
        id: format!("lora-{}", lora_id),
        object: "model".to_string(),
        created: layer.created_at,
        owned_by: "user".to_string(),
    };
    
    Ok(JsonResponse(openai_compat::ListModelsResponse {
        object: "list".to_string(),
        data: vec![model],
    }))
}

/// Start auto-regressive training
#[allow(dead_code)]
async fn start_auto_training(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(config): Json<TrainingConfig>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    state.training_service.start_auto_training(&lora_id, config).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(JsonResponse(serde_json::json!({
        "status": "started",
        "lora_id": lora_id
    })))
}

/// Stop auto-regressive training
#[allow(dead_code)]
async fn stop_auto_training(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    state.training_service.stop_auto_training(&lora_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(JsonResponse(serde_json::json!({
        "status": "stopped",
        "lora_id": lora_id
    })))
}

/// Get training status
#[allow(dead_code)]
async fn training_status(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<TrainingStatus>, StatusCode> {
    let status = state.training_service.get_training_status(&lora_id).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(JsonResponse(status))
}

/// Trigger manual training
#[allow(dead_code)]
async fn trigger_training(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(samples): Json<Vec<TrainingSample>>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    for sample in samples {
        state.training_service.queue_training_sample(&lora_id, sample).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    }
    
    Ok(JsonResponse(serde_json::json!({
        "status": "queued",
        "lora_id": lora_id
    })))
}

/// Get LoRA statistics
#[allow(dead_code)]
async fn get_lora_stats(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<LoRAStats>, StatusCode> {
    let lora_id = lora_id.parse::<LoRAId>().map_err(|_| StatusCode::BAD_REQUEST)?;
    let stats = state.lora_registry.get_stats(&lora_id).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(JsonResponse(stats))
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


#[allow(dead_code)]
fn format_chat_messages(messages: &[openai_compat::ChatMessage]) -> String {
    messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content.as_ref().unwrap_or(&String::new())))
        .collect::<Vec<_>>()
        .join("\n")
}

// Re-exports
use chrono;
