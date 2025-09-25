//! LoRA adapter management endpoints

use axum::{
    Router,
    routing::{get, post, delete},
    extract::{State, Path, Json},
    response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use crate::server::state::ServerState;

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
    // config: LoRAConfig,
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
    State(_state): State<ServerState>,
    Json(_request): Json<CreateLoRARequest>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": "LoRA registry has been removed"
    }))).into_response()
}

/// List all LoRA adapters
async fn list_loras(
    State(_state): State<ServerState>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": "LoRA registry has been removed"
    }))).into_response()
}

/// Get information about a specific LoRA
async fn get_lora_info(
    State(_state): State<ServerState>,
    Path(_id): Path<String>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": "LoRA registry has been removed"
    }))).into_response()
}

/// Delete a LoRA adapter
async fn delete_lora(
    State(_state): State<ServerState>,
    Path(_id): Path<String>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": "LoRA registry has been removed"
    }))).into_response()
}

/// Update LoRA configuration
async fn update_lora(
    State(_state): State<ServerState>,
    Path(_id): Path<String>,
    Json(_config): Json<serde_json::Value>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": "LoRA registry has been removed"
    }))).into_response()
}