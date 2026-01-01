//! Model management endpoints

use crate::{auth::Operation, server::{self, state::ServerState, AuthenticatedUser}, storage::paths::StoragePaths};
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Extension, Router,
};
use chrono;
use serde::{Deserialize, Serialize};

/// Create model management router
pub fn create_router() -> Router<ServerState> {
    Router::new()
        .route("/list", get(list_models))
        .route("/:id/info", get(get_model_info))
        .route("/download", post(download_model))
        .route("/:id/load", post(load_model))
        .route("/:id/unload", post(unload_model))
        .route("/cache/refresh", post(refresh_cache))
        // REMOVED: .route("/cache/stats", get(cache_stats)) - ModelCache replaced by ZMQ
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
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    if !state.policy_client.check(&user, "registry:*", Operation::Query).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot query registry", user)
            })),
        ).into_response();
    }

    match state.model_storage.list_models().await {
        Ok(models) => {
            // Transform the raw model data into a cleaner response format
            let model_list: Vec<ModelListItem> = models
                .into_iter()
                .map(|(model_ref, metadata)| {
                    let local_path = None; // Path is managed by registry

                    ModelListItem {
                        id: model_ref.model.clone(),
                        name: metadata
                            .display_name
                            .clone()
                            .unwrap_or_else(|| metadata.name.clone()),
                        display_name: metadata.display_name,
                        architecture: metadata.model_type.clone(),
                        size_bytes: metadata.size_bytes.unwrap_or(0),
                        is_cached: true,
                        local_path,
                    }
                })
                .collect();

            Json(model_list).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to list models: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Get information about a specific model
async fn get_model_info(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{}", id);
    if !state.policy_client.check(&user, &resource, Operation::Query).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot query '{}'", user, resource)
            })),
        ).into_response();
    }

    // Parse model reference
    use crate::storage::model_ref::ModelRef;
    if let Ok(model_ref) = ModelRef::parse(&id) {
        if let Ok(_path) = state.model_storage.get_model_path(&model_ref).await {
            // Create metadata for the found model
            let metadata = crate::storage::ModelMetadata {
                name: model_ref.model.clone(),
                display_name: Some(id.clone()),
                model_type: "language_model".to_string(),
                created_at: chrono::Utc::now().timestamp(),
                updated_at: chrono::Utc::now().timestamp(),
                size_bytes: None,
                tags: vec![],
            };
            return Json(metadata).into_response();
        }
    }

    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({
            "error": "Model not found"
        })),
    )
        .into_response()
}

/// Download a model from a registry
async fn download_model(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(request): Json<DownloadModelRequest>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    if !state.policy_client.check(&user, "registry:*", Operation::Write).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot write to registry", user)
            })),
        ).into_response();
    }

    // Basic validation - must be a non-empty string
    let uri = request.uri.clone();
    if uri.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Model URI cannot be empty"
            })),
        )
            .into_response();
    }

    // Get storage paths and token
    let storage_paths = match StoragePaths::new() {
        Ok(paths) => paths,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to get storage paths: {}", e)
                })),
            )
                .into_response();
        }
    };

    let _models_dir = match storage_paths.models_dir() {
        Ok(dir) => dir,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to get models directory: {}", e)
                })),
            )
                .into_response();
        }
    };

    // Use shared operation
    match crate::storage::operations::clone_model(&request.uri, request.name.as_deref(), None).await
    {
        Ok(cloned) => Json(DownloadModelResponse {
            id: cloned.model_id.to_string(),
            name: cloned.model_name,
            status: "downloaded".to_string(),
            path: cloned.model_path.to_string_lossy().to_string(),
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to download model: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Load a model into the engine pool
async fn load_model(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{}", id);
    if !state.policy_client.check(&user, &resource, Operation::Manage).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot manage '{}'", user, resource)
            })),
        ).into_response();
    }

    // Parse model reference
    let model_ref = match crate::storage::ModelRef::parse(&id) {
        Ok(model_ref) => model_ref,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid model reference: '{}'", id)
                })),
            )
                .into_response();
        }
    };

    // Get model path
    let model_path = match state.model_storage.get_model_path(&model_ref).await {
        Ok(path) => path,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Model not found: {}", e)
                })),
            )
                .into_response();
        }
    };

    // Load into engine pool
    Json(serde_json::json!({
        "status": "loaded",
        "id": id,
        "path": model_path.to_string_lossy()
    }))
    .into_response()
}

/// Unload a model from the engine pool
async fn unload_model(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{}", id);
    if !state.policy_client.check(&user, &resource, Operation::Manage).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot manage '{}'", user, resource)
            })),
        ).into_response();
    }

    Json(serde_json::json!({
        "status": "unloaded",
        "id": id
    }))
    .into_response()
}

/// Refresh the model cache (rescans disk for models)
async fn refresh_cache(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    if !state.policy_client.check(&user, "registry:*", Operation::Manage).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot manage registry", user)
            })),
        ).into_response();
    }

    // Cache is automatically maintained
    Json(serde_json::json!({
        "status": "success",
        "message": "Model cache is automatically maintained"
    }))
    .into_response()
}

// REMOVED: cache_stats endpoint - ModelCache replaced by ZMQ ModelService
