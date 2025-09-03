//! Model management endpoints

use axum::{
    Router,
    routing::{get, post},
    extract::{State, Path, Json},
    response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use crate::{
    api::{
        model_management::ModelUri,
        model_storage::{ModelStorage, ModelMetadata},
    },
    server::state::ServerState,
};

/// Create model management router
pub fn create_router() -> Router<ServerState> {
    Router::new()
        .route("/list", get(list_models))
        .route("/:id/info", get(get_model_info))
        .route("/download", post(download_model))
        .route("/:id/load", post(load_model))
        .route("/:id/unload", post(unload_model))
}

/// Request to download a model
#[derive(Debug, Deserialize)]
struct DownloadModelRequest {
    uri: String,
    name: Option<String>,
}

/// Response for model download
#[derive(Debug, Serialize)]
struct DownloadModelResponse {
    id: String,
    name: String,
    status: String,
    path: String,
}

/// List all available models
async fn list_models(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    match state.model_storage.list_models().await {
        Ok(models) => Json(models).into_response(),
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to list models: {}", e)
            }))).into_response()
        }
    }
}

/// Get information about a specific model
async fn get_model_info(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Try to parse as UUID
    if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        let model_id = crate::api::model_storage::ModelId(uuid);
        match state.model_storage.get_metadata_by_id(&model_id).await {
            Ok(metadata) => return Json(metadata).into_response(),
            Err(_) => {}
        }
    }
    
    // Try as URI
    if let Ok(uri) = ModelUri::parse(&id) {
        match state.model_storage.get_metadata(&uri).await {
            Ok(metadata) => return Json(metadata).into_response(),
            Err(_) => {}
        }
    }
    
    (StatusCode::NOT_FOUND, Json(serde_json::json!({
        "error": "Model not found"
    }))).into_response()
}

/// Download a model from a registry
async fn download_model(
    State(state): State<ServerState>,
    Json(request): Json<DownloadModelRequest>,
) -> impl IntoResponse {
    let uri = match ModelUri::parse(&request.uri) {
        Ok(uri) => uri,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": format!("Invalid model URI: {}", e)
            }))).into_response();
        }
    };
    
    // Start download (this would be async in production)
    let name = request.name.clone();
    match state.model_storage.download_model(&uri, name).await {
        Ok(model_id) => {
            // Get metadata for response
            if let Ok(metadata) = state.model_storage.get_metadata_by_id(&model_id).await {
                Json(DownloadModelResponse {
                    id: model_id.to_string(),
                    name: metadata.name,
                    status: "downloaded".to_string(),
                    path: metadata.local_path.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_default(),
                }).into_response()
            } else {
                Json(DownloadModelResponse {
                    id: model_id.to_string(),
                    name: request.name.unwrap_or_else(|| uri.uri.clone()),
                    status: "downloaded".to_string(),
                    path: format!("models/{}", model_id),
                }).into_response()
            }
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to download model: {}", e)
            }))).into_response()
        }
    }
}

/// Load a model into the engine pool
async fn load_model(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Get model path
    let model_path = if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
        let model_id = crate::api::model_storage::ModelId(uuid);
        match state.model_storage.get_metadata_by_id(&model_id).await {
            Ok(metadata) => metadata.local_path.unwrap_or_else(|| std::path::PathBuf::from(".")),
            Err(e) => {
                return (StatusCode::NOT_FOUND, Json(serde_json::json!({
                    "error": format!("Model not found: {}", e)
                }))).into_response();
            }
        }
    } else {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "Invalid model ID"
        }))).into_response();
    };
    
    // Load into engine pool
    // Note: This is simplified - actual implementation would manage engine pool state
    Json(serde_json::json!({
        "status": "loaded",
        "id": id,
        "path": model_path.to_string_lossy()
    })).into_response()
}

/// Unload a model from the engine pool
async fn unload_model(
    State(_state): State<ServerState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Note: This is simplified - actual implementation would manage engine pool state
    Json(serde_json::json!({
        "status": "unloaded",
        "id": id
    })).into_response()
}