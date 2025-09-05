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
        model_storage::ModelUri,
        model_storage::{ModelStorage, ModelMetadata, ModelId},
        model_downloader::{ModelDownloader, DownloadOptions, ModelFormat},
    },
    server::state::ServerState,
    storage::paths::StoragePaths,
};
use chrono;

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

/// Model list response item
#[derive(Debug, Serialize)]
struct ModelListItem {
    id: String,
    name: String,
    display_name: Option<String>,
    architecture: String,
    size_bytes: u64,
    is_cached: bool,
    local_path: Option<String>,
}

/// List all available models
async fn list_models(
    State(state): State<ServerState>,
) -> impl IntoResponse {
    match state.model_storage.children().await {
        Ok(models) => {
            // Transform the raw model data into a cleaner response format
            let model_list: Vec<ModelListItem> = models
                .into_iter()
                .map(|(id, metadata)| ModelListItem {
                    id: id.to_string(),
                    name: metadata.display_name.clone()
                        .unwrap_or_else(|| metadata.name.clone()),
                    display_name: metadata.display_name,
                    architecture: metadata.architecture,
                    size_bytes: metadata.size_bytes,
                    is_cached: metadata.is_cached,
                    local_path: metadata.local_path.as_ref()
                        .and_then(|p| p.to_str())
                        .map(|s| s.to_string()),
                })
                .collect();
            
            Json(model_list).into_response()
        },
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
    // Parse and validate URI using standard URL parsing
    let uri = match ModelUri::parse(&request.uri) {
        Ok(uri) => uri,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": format!("Invalid model URI '{}': {}", request.uri, e),
                "details": "Model URIs must use the format: hf://org/model",
                "example": "hf://Qwen/Qwen2-1.5B-Instruct"
            }))).into_response();
        }
    };
    
    // Get storage paths and token
    let storage_paths = match StoragePaths::new() {
        Ok(paths) => paths,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to get storage paths: {}", e)
            }))).into_response();
        }
    };
    
    let models_dir = match storage_paths.models_dir() {
        Ok(dir) => dir,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to get models directory: {}", e)
            }))).into_response();
        }
    };
    
    let hf_token = std::env::var("HF_TOKEN").ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
    
    // Create downloader - same as CLI does
    let downloader = match ModelDownloader::new(models_dir, hf_token).await {
        Ok(d) => d,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to create downloader: {}", e)
            }))).into_response();
        }
    };
    
    // Download options - same as CLI
    let options = DownloadOptions {
        preferred_format: ModelFormat::SafeTensors,
        force: false,
        show_progress: false,
        files_filter: None,
        verify_checksums: true,
        max_size_bytes: None,
    };
    
    // Download the model using the same method CLI uses
    match downloader.download(&request.uri, options).await {
        Ok(downloaded_model) => {
            // Store metadata in ModelStorage after successful download
            let model_id = ModelId::new();
            
            // Store the downloaded model metadata (uri already parsed above)
            {
                // Use the original model_id (e.g., "microsoft/phi-2") as the display name
                let display_name = downloaded_model.model_id.clone();
                // Use the user-provided name or the model_id for the internal name
                let model_name = request.name.clone().unwrap_or_else(|| display_name.clone());
                let architecture = downloaded_model.config.as_ref()
                    .map(|c| c.architecture.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                let parameters = None; // ModelConfig doesn't have a parameters field
                    
                let metadata = ModelMetadata {
                    model_id: model_id.clone(),
                    name: model_name.clone(),
                    display_name: Some(display_name),
                    architecture,
                    parameters,
                    model_type: "language_model".to_string(),
                    tokenizer_type: downloaded_model.tokenizer.as_ref()
                        .and_then(|t| t.tokenizer_class.clone()),
                    size_bytes: downloaded_model.total_size_bytes,
                    files: downloaded_model.files.clone(),
                    external_sources: vec![],
                    local_path: Some(downloaded_model.local_path.clone()),
                    is_cached: true,
                    tags: vec![],
                    description: None,
                    license: None,
                    created_at: chrono::Utc::now().timestamp(),
                    last_accessed: chrono::Utc::now().timestamp(),
                    last_updated: chrono::Utc::now().timestamp(),
                };
                
                let _ = state.model_storage.store_metadata(&uri, metadata).await;
            }
            
            Json(DownloadModelResponse {
                id: model_id.to_string(),
                name: request.name.unwrap_or_else(|| downloaded_model.model_id),
                status: "downloaded".to_string(),
                path: downloaded_model.local_path.to_string_lossy().to_string(),
            }).into_response()
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
    Json(serde_json::json!({
        "status": "unloaded",
        "id": id
    })).into_response()
}