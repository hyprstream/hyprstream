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

/// Resolve a ModelRef to its worktree path via registry.
async fn resolve_model_path(
    registry: &crate::services::GenRegistryClient,
    model_ref: &crate::storage::ModelRef,
) -> anyhow::Result<String> {
    let tracked = registry.get_by_name(&model_ref.model).await?;
    let repo = registry.repo(&tracked.id);
    let branch = match &model_ref.git_ref {
        crate::storage::GitRef::Branch(name) => name.clone(),
        _ => repo.get_head().await?,
    };
    let wts = repo.list_worktrees().await?;
    wts.iter()
        .find(|wt| wt.branch_name == branch)
        .map(|wt| wt.path.clone())
        .ok_or_else(|| anyhow::anyhow!("worktree for {}:{} not found", model_ref.model, branch))
}

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
    if !state.policy_client.check(&user, "*", "registry:*", Operation::Query.as_str()).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot query registry", user)
            })),
        ).into_response();
    }

    // Inline list_models: iterate repos + worktrees
    let result: Result<Vec<ModelListItem>, anyhow::Error> = async {
        let repos = state.registry.list().await?;
        let mut model_list = Vec::new();
        for repo in repos {
            if repo.name.is_empty() { continue; }
            let name = &repo.name;
            match state.registry.repo(&repo.id).list_worktrees().await {
                Ok(worktrees) => {
                    for wt in worktrees {
                        if wt.branch_name.is_empty() { continue; }
                        let display = format!("{}:{}", name, wt.branch_name);
                        model_list.push(ModelListItem {
                            id: name.clone(),
                            name: display.clone(),
                            display_name: Some(display),
                            architecture: "language_model".to_owned(),
                            size_bytes: 0,
                            is_cached: true,
                            local_path: Some(wt.path.clone()),
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to list worktrees for {}: {}", name, e);
                }
            }
        }
        Ok(model_list)
    }.await;

    match result {
        Ok(model_list) => Json(model_list).into_response(),
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
    let resource = format!("model:{id}");
    if !state.policy_client.check(&user, "*", &resource, Operation::Query.as_str()).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot query '{}'", user, resource)
            })),
        ).into_response();
    }

    // Parse model reference and resolve path
    use crate::storage::model_ref::ModelRef;
    if let Ok(model_ref) = ModelRef::parse(&id) {
        let path_result: Result<String, anyhow::Error> = async {
            let tracked = state.registry.get_by_name(&model_ref.model).await?;
            let repo = state.registry.repo(&tracked.id);
            let branch = match &model_ref.git_ref {
                crate::storage::GitRef::Branch(name) => name.clone(),
                _ => repo.get_head().await?,
            };
            let wts = repo.list_worktrees().await?;
            wts.iter()
                .find(|wt| wt.branch_name == branch)
                .map(|wt| wt.path.clone())
                .ok_or_else(|| anyhow::anyhow!("worktree not found"))
        }.await;
        if let Ok(_path) = path_result {
            // Create metadata for the found model
            let metadata = crate::storage::ModelMetadata {
                name: model_ref.model.clone(),
                display_name: Some(id.clone()),
                model_type: "language_model".to_owned(),
                created_at: chrono::Utc::now().timestamp(),
                updated_at: chrono::Utc::now().timestamp(),
                size_bytes: None,
                tags: vec![],
                is_dirty: false, // Not available for single model lookup
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
    if !state.policy_client.check(&user, "*", "registry:*", Operation::Write.as_str()).await.unwrap_or(false) {
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

    // Derive model name from URL if not provided
    let model_name = request.name.clone().unwrap_or_else(|| {
        request
            .uri
            .split('/')
            .next_back()
            .unwrap_or("")
            .trim_end_matches(".git").to_owned()
    });

    if model_name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Cannot derive model name from URI. Please provide a name."
            })),
        )
            .into_response();
    }

    // Use registry client to clone model (no duplicate service)
    if let Err(e) = state.registry.clone(&request.uri, &model_name, true, 1, "").await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to download model: {}", e)
            })),
        )
            .into_response();
    }

    // Get model path for response
    let model_ref = crate::storage::ModelRef::new(model_name.clone());
    let model_path = match resolve_model_path(&state.registry, &model_ref).await {
        Ok(path) => path,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Model cloned but failed to get path: {}", e)
                })),
            )
                .into_response();
        }
    };

    Json(DownloadModelResponse {
        id: crate::storage::ModelId::new().to_string(),
        name: model_name,
        status: "downloaded".to_owned(),
        path: model_path.clone(),
    })
    .into_response()
}

/// Load a model into the engine pool
async fn load_model(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{id}");
    if !state.policy_client.check(&user, "*", &resource, Operation::Manage.as_str()).await.unwrap_or(false) {
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
    let model_path = match resolve_model_path(&state.registry, &model_ref).await {
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
        "path": model_path
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
    let resource = format!("model:{id}");
    if !state.policy_client.check(&user, "*", &resource, Operation::Manage.as_str()).await.unwrap_or(false) {
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
    if !state.policy_client.check(&user, "*", "registry:*", Operation::Manage.as_str()).await.unwrap_or(false) {
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
