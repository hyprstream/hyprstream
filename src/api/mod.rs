//! REST API for creating and managing sparse auto-regressive LoRA training layers

use axum::{
    Router,
    routing::{get, post},
    extract::{Path, State, Json},
    response::Json as JsonResponse,
    http::StatusCode,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod lora_registry;
pub mod model_management;
pub mod model_registry;
pub mod model_storage;
pub mod huggingface;
pub mod inference_service;
pub mod openai_compat;
pub mod training_service;

use lora_registry::{LoRARegistry, LoRALayer};
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

/// LoRA endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAEndpoint {
    pub lora_id: String,
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
        // LoRA management endpoints
        .route("/v1/lora/create", post(create_lora_layer))
        .route("/v1/lora/list", get(list_lora_layers))
        .route("/v1/lora/:lora_id/info", get(get_lora_info))
        .route("/v1/lora/:lora_id/delete", post(delete_lora_layer))
        .route("/v1/lora/:lora_id/train", post(trigger_training))
        .route("/v1/lora/:lora_id/stats", get(get_lora_stats))
        
        // OpenAI-compatible endpoints (dynamically registered per LoRA)
        .route("/v1/inference/:lora_id/chat/completions", post(openai_chat_completions))
        .route("/v1/inference/:lora_id/completions", post(openai_completions))
        .route("/v1/inference/:lora_id/embeddings", post(openai_embeddings))
        .route("/v1/inference/:lora_id/models", get(openai_list_models))
        
        // Training endpoints
        .route("/v1/training/:lora_id/start", post(start_auto_training))
        .route("/v1/training/:lora_id/stop", post(stop_auto_training))
        .route("/v1/training/:lora_id/status", get(training_status))
        
        .with_state(state)
}

/// Create a new LoRA layer
async fn create_lora_layer(
    State(state): State<ApiState>,
    Json(request): Json<CreateLoRARequest>,
) -> Result<JsonResponse<CreateLoRAResponse>, StatusCode> {
    let lora_id = Uuid::new_v4().to_string();
    
    // Create the LoRA layer in the registry
    let layer = LoRALayer {
        id: lora_id.clone(),
        name: request.name.unwrap_or_else(|| format!("lora_{}", &lora_id[..8])),
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
    
    // Create sparse adapter in VDB storage
    let adapter_config = crate::adapters::sparse_lora::SparseLoRAConfig {
        in_features: 1536, // TODO: get from model config
        out_features: 1536, // TODO: get from model config
        rank: request.config.rank,
        sparsity: request.config.sparsity_ratio,
        learning_rate: 1e-4, // TODO: get from request
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
            state.vdb_storage.store_adapter_neural_compressed(&lora_id, &adapter).await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
        
        { 
            // VDB feature not enabled, skip storage
        }
    } else {
        
        { 
            state.vdb_storage.store_adapter_accelerated(&lora_id, &adapter).await
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
    endpoints.insert(lora_id.clone(), endpoint);
    
    // Start auto-regressive training if enabled
    if request.auto_regressive {
        if let Some(training_config) = request.training_config {
            state.training_service.start_auto_training(
                &lora_id,
                training_config,
            ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
    }
    
    Ok(JsonResponse(CreateLoRAResponse {
        lora_id: lora_id.clone(),
        endpoint: format!("/v1/inference/{}", lora_id),
        openai_base_url: format!("/v1/inference/{}", lora_id),
        status: "created".to_string(),
    }))
}

/// List all LoRA layers
async fn list_lora_layers(
    State(state): State<ApiState>,
) -> Result<JsonResponse<Vec<LoRALayer>>, StatusCode> {
    let layers = state.lora_registry.list_all().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(JsonResponse(layers))
}

/// Get information about a specific LoRA layer
async fn get_lora_info(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<LoRALayer>, StatusCode> {
    let layer = state.lora_registry.get(&lora_id).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(JsonResponse(layer))
}

/// Delete a LoRA layer
async fn delete_lora_layer(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<serde_json::Value>, StatusCode> {
    // Stop training if active
    let _ = state.training_service.stop_auto_training(&lora_id).await;
    
    // Remove from registry
    state.lora_registry.unregister(&lora_id).await
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
async fn openai_chat_completions(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(request): Json<openai_compat::ChatCompletionRequest>,
) -> Result<JsonResponse<openai_compat::ChatCompletionResponse>, StatusCode> {
    // Load the LoRA adapter
    
    let _adapter = state.vdb_storage.load_adapter_neural_compressed(
        &lora_id,
        Default::default(),
    ).await.map_err(|_| StatusCode::NOT_FOUND)?;
    
    
    return Err(StatusCode::SERVICE_UNAVAILABLE);
    
    // Create inference session
    let session_id = state.training_service.create_inference_session(
        &lora_id,
        vec![lora_id.clone()],
    ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Convert OpenAI request to internal format
    let input = crate::inference::InferenceInput {
        prompt: Some(format_chat_messages(&request.messages)),
        input_ids: None,
        max_tokens: request.max_tokens.unwrap_or(2048),
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        stream: request.stream.unwrap_or(false),
    };
    
    // Run inference
    let output = state.training_service.infer(&session_id, input).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // If auto-regressive training is enabled, learn from this interaction
    if let Some(endpoint) = state.endpoints.read().await.get(&lora_id) {
        if endpoint.auto_regressive {
            // Queue for training
            state.training_service.queue_training_sample(
                &lora_id,
                TrainingSample {
                    input: format_chat_messages(&request.messages),
                    output: output.text.clone(),
                    timestamp: chrono::Utc::now().timestamp(),
                },
            ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
    }
    
    // Convert to OpenAI response format
    let response = openai_compat::ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: format!("lora-{}", lora_id),
        choices: vec![
            openai_compat::ChatChoice {
                index: 0,
                message: openai_compat::ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(output.text),
                    function_call: None,
                },
                finish_reason: Some("stop".to_string()),
            }
        ],
        usage: Some(openai_compat::Usage {
            prompt_tokens: request.messages.len() * 10, // Estimate
            completion_tokens: output.tokens_generated,
            total_tokens: request.messages.len() * 10 + output.tokens_generated,
        }),
    };
    
    // Close session
    let _ = state.training_service.close_inference_session(&session_id).await;
    
    Ok(JsonResponse(response))
}

/// OpenAI-compatible completions endpoint
async fn openai_completions(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(request): Json<openai_compat::CompletionRequest>,
) -> Result<JsonResponse<openai_compat::CompletionResponse>, StatusCode> {
    // Similar to chat completions but for raw completions
    
    let _adapter = state.vdb_storage.load_adapter_neural_compressed(
        &lora_id,
        Default::default(),
    ).await.map_err(|_| StatusCode::NOT_FOUND)?;
    
    
    return Err(StatusCode::SERVICE_UNAVAILABLE);
    
    let session_id = state.training_service.create_inference_session(
        &lora_id,
        vec![lora_id.clone()],
    ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let input = crate::inference::InferenceInput {
        prompt: Some(request.prompt.clone()),
        input_ids: None,
        max_tokens: request.max_tokens.unwrap_or(2048),
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        stream: request.stream.unwrap_or(false),
    };
    
    let output = state.training_service.infer(&session_id, input).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Auto-regressive training
    if let Some(endpoint) = state.endpoints.read().await.get(&lora_id) {
        if endpoint.auto_regressive {
            state.training_service.queue_training_sample(
                &lora_id,
                TrainingSample {
                    input: request.prompt.clone(),
                    output: output.text.clone(),
                    timestamp: chrono::Utc::now().timestamp(),
                },
            ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }
    }
    
    let response = openai_compat::CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: format!("lora-{}", lora_id),
        choices: vec![
            openai_compat::CompletionChoice {
                text: output.text,
                index: 0,
                logprobs: None,
                finish_reason: Some("stop".to_string()),
            }
        ],
        usage: Some(openai_compat::Usage {
            prompt_tokens: request.prompt.len() / 4, // Rough estimate
            completion_tokens: output.tokens_generated,
            total_tokens: request.prompt.len() / 4 + output.tokens_generated,
        }),
    };
    
    let _ = state.training_service.close_inference_session(&session_id).await;
    
    Ok(JsonResponse(response))
}

/// OpenAI-compatible embeddings endpoint
async fn openai_embeddings(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
    Json(request): Json<openai_compat::EmbeddingRequest>,
) -> Result<JsonResponse<openai_compat::EmbeddingResponse>, StatusCode> {
    // Generate embeddings using the LoRA-adapted model
    
    let _adapter = state.vdb_storage.load_adapter_neural_compressed(
        &lora_id,
        Default::default(),
    ).await.map_err(|_| StatusCode::NOT_FOUND)?;
    
    
    return Err(StatusCode::SERVICE_UNAVAILABLE);
    
    let mut embeddings = Vec::new();
    
    for (idx, input) in request.input.iter().enumerate() {
        // Generate embedding (simplified - would use actual model)
        let embedding = state.training_service.generate_embedding(
            &lora_id,
            input,
        ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        embeddings.push(openai_compat::EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: idx,
        });
    }
    
    let response = openai_compat::EmbeddingResponse {
        object: "list".to_string(),
        data: embeddings,
        model: format!("lora-{}", lora_id),
        usage: openai_compat::Usage {
            prompt_tokens: request.input.len() * 10,
            completion_tokens: 0,
            total_tokens: request.input.len() * 10,
        },
    };
    
    Ok(JsonResponse(response))
}

/// List models (OpenAI-compatible)
async fn openai_list_models(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<openai_compat::ListModelsResponse>, StatusCode> {
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
async fn training_status(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<TrainingStatus>, StatusCode> {
    let status = state.training_service.get_training_status(&lora_id).await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(JsonResponse(status))
}

/// Trigger manual training
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
async fn get_lora_stats(
    State(state): State<ApiState>,
    Path(lora_id): Path<String>,
) -> Result<JsonResponse<LoRAStats>, StatusCode> {
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

fn format_chat_messages(messages: &[openai_compat::ChatMessage]) -> String {
    messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content.as_ref().unwrap_or(&String::new())))
        .collect::<Vec<_>>()
        .join("\n")
}

// Re-exports
use chrono;
use uuid;