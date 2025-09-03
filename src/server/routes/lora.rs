//! LoRA adapter management endpoints

use axum::{
    Router,
    routing::{get, post, delete},
    extract::{State, Path, Json},
    response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use crate::{
    api::{
        lora_registry::{LoRARegistry, LoRALayer, LoRAId},
        LoRAConfig,
    },
    server::state::ServerState,
};

/// Create LoRA management router
pub fn create_router() -> Router<ServerState> {
    Router::new()
        .route("/create", post(create_lora))
        .route("/list", get(list_loras))
        .route("/:id/info", get(get_lora_info))
        .route("/:id/delete", delete(delete_lora))
        .route("/:id/update", post(update_lora))
}

/// Request to create a new LoRA adapter
#[derive(Debug, Deserialize)]
struct CreateLoRARequest {
    name: Option<String>,
    base_model: String,
    config: LoRAConfig,
    auto_regressive: bool,
}

/// Response when creating a LoRA
#[derive(Debug, Serialize)]
struct CreateLoRAResponse {
    id: String,
    name: String,
    status: String,
}

/// Create a new LoRA adapter
async fn create_lora(
    State(state): State<ServerState>,
    Json(request): Json<CreateLoRARequest>,
) -> impl IntoResponse {
    let lora_id = LoRAId::new();
    let name = request.name.unwrap_or_else(|| format!("lora_{}", &lora_id.to_string()[..8]));
    
    let layer = LoRALayer {
        id: lora_id.clone(),
        name: name.clone(),
        base_model: request.base_model,
        config: request.config.clone(),
        created_at: chrono::Utc::now().timestamp(),
        updated_at: chrono::Utc::now().timestamp(),
        training_enabled: request.auto_regressive,
        total_tokens_trained: 0,
        sparsity_ratio: request.config.sparsity_ratio,
    };
    
    match state.lora_registry.register(layer).await {
        Ok(_) => {
            Json(CreateLoRAResponse {
                id: lora_id.to_string(),
                name,
                status: "created".to_string(),
            }).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to create LoRA: {}", e)
            }))).into_response()
        }
    }
}

/// List all LoRA adapters
async fn list_loras(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    match state.lora_registry.list_all().await {
        Ok(loras) => Json(loras).into_response(),
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to list LoRAs: {}", e)
            }))).into_response()
        }
    }
}

/// Get information about a specific LoRA
async fn get_lora_info(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.lora_registry.get_by_id_or_name(&id).await {
        Ok(lora) => Json(lora).into_response(),
        Err(_) => {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": "LoRA not found"
            }))).into_response()
        }
    }
}

/// Delete a LoRA adapter
async fn delete_lora(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Stop any active training
    let _ = state.training_service.stop_auto_training(&id).await;
    
    match state.lora_registry.unregister_by_name(&id).await {
        Ok(_) => {
            Json(serde_json::json!({
                "status": "deleted",
                "id": id
            })).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to delete LoRA: {}", e)
            }))).into_response()
        }
    }
}

/// Update LoRA configuration
async fn update_lora(
    State(state): State<ServerState>,
    Path(id): Path<String>,
    Json(config): Json<LoRAConfig>,
) -> impl IntoResponse {
    match state.lora_registry.get_by_id_or_name(&id).await {
        Ok(mut lora) => {
            lora.config = config;
            lora.updated_at = chrono::Utc::now().timestamp();
            
            // Re-register with updated config
            match state.lora_registry.register(lora).await {
                Ok(_) => {
                    Json(serde_json::json!({
                        "status": "updated",
                        "id": id
                    })).into_response()
                }
                Err(e) => {
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                        "error": format!("Failed to update LoRA: {}", e)
                    }))).into_response()
                }
            }
        }
        Err(_) => {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": "LoRA not found"
            }))).into_response()
        }
    }
}