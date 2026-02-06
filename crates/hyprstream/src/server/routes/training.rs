//! Training service endpoints

use crate::{
    api::{TrainingSample, TrainingStatus},
    api::training_service::TrainingConfig,
    auth::Operation,
    server::{self, state::ServerState, AuthenticatedUser},
    training::{CheckpointManager, WeightFormat, WeightSnapshot},
};
use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Extension, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Create training service router
pub fn create_router() -> Router<ServerState> {
    Router::new()
        .route("/:lora_id/start", post(start_training))
        .route("/:lora_id/stop", post(stop_training))
        .route("/:lora_id/status", get(get_status))
        .route("/:lora_id/submit", post(submit_samples))
        .route("/checkpoint", post(write_checkpoint))
        .route("/commit", post(commit_checkpoint))
        .route("/pretrain", post(start_pretraining))
}

/// Batch training submission
#[derive(Debug, Deserialize)]
struct SubmitSamplesRequest {
    samples: Vec<TrainingSample>,
}

/// Checkpoint write request
#[derive(Debug, Deserialize)]
struct CheckpointRequest {
    model_id: String,
    #[allow(dead_code)]
    name: Option<String>,
    step: Option<usize>,
}

/// Checkpoint write response
#[derive(Debug, Serialize)]
struct CheckpointResponse {
    path: String,
    step: usize,
}

/// Checkpoint commit request
#[derive(Debug, Deserialize)]
struct CommitRequest {
    checkpoint_path: String,
    message: Option<String>,
    branch: Option<String>,
}

/// Checkpoint commit response
#[derive(Debug, Serialize)]
struct CommitResponse {
    commit_id: String,
}

/// Pre-training request
#[derive(Debug, Deserialize)]
struct PreTrainingRequest {
    model_id: String,
    learning_rate: f32,
    batch_size: usize,
    warmup_steps: usize,
    total_steps: usize,
}

/// Start training for a LoRA adapter
async fn start_training(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(lora_id): Path<String>,
    Json(config): Json<TrainingConfig>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{lora_id}");
    if !state.policy_client.check_policy(&user, &resource, Operation::Train).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot train '{}'", user, resource)
            })),
        ).into_response();
    }

    match state
        .training_service
        .start_auto_training(&lora_id, config)
        .await
    {
        Ok(_) => Json(serde_json::json!({
            "status": "started",
            "lora_id": lora_id
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to start training: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Stop training for a LoRA adapter
async fn stop_training(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(lora_id): Path<String>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{lora_id}");
    if !state.policy_client.check_policy(&user, &resource, Operation::Train).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot train '{}'", user, resource)
            })),
        ).into_response();
    }

    match state.training_service.stop_auto_training(&lora_id).await {
        Ok(_) => Json(serde_json::json!({
            "status": "stopped",
            "lora_id": lora_id
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to stop training: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Get training status
async fn get_status(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(lora_id): Path<String>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{lora_id}");
    if !state.policy_client.check_policy(&user, &resource, Operation::Query).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot query '{}'", user, resource)
            })),
        ).into_response();
    }

    match state.training_service.get_training_status(&lora_id).await {
        Ok(status) => Json(status).into_response(),
        Err(_) => Json(TrainingStatus {
            is_training: false,
            total_samples_processed: 0,
            current_loss: 0.0,
            learning_rate: 0.0,
            last_update: 0,
        })
        .into_response(),
    }
}

/// Submit training samples
async fn submit_samples(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Path(lora_id): Path<String>,
    Json(request): Json<SubmitSamplesRequest>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{lora_id}");
    if !state.policy_client.check_policy(&user, &resource, Operation::Train).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot train '{}'", user, resource)
            })),
        ).into_response();
    }

    let mut submitted = 0;

    for sample in request.samples {
        let training_sample = crate::api::TrainingSample {
            input: sample.input,
            output: sample.output,
        };

        if state
            .training_service
            .queue_training_sample(&lora_id, training_sample)
            .await
            .is_ok()
        {
            submitted += 1;
        }
    }

    Json(serde_json::json!({
        "status": "submitted",
        "lora_id": lora_id,
        "samples_submitted": submitted
    }))
    .into_response()
}

/// Write checkpoint to filesystem (no Git)
async fn write_checkpoint(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<CheckpointRequest>,
) -> impl IntoResponse {
    use crate::storage::ModelRef;

    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{}", req.model_id);
    if !state.policy_client.check_policy(&user, &resource, Operation::Write).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot write '{}'", user, resource)
            })),
        ).into_response();
    }

    // Resolve model path via registry
    let model_ref = match ModelRef::parse(&req.model_id) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid model ID: {}", e)
                })),
            )
                .into_response();
        }
    };

    let model_path = match state.registry.get_model_path(&model_ref).await {
        Ok(p) => p,
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

    // Create checkpoint manager
    let checkpoint_mgr = match CheckpointManager::new(model_path) {
        Ok(mgr) => mgr,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to create checkpoint manager: {}", e)
                })),
            )
                .into_response();
        }
    };

    // Get step number
    let step = req.step.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        // SAFETY: Only fails if system time is before Unix epoch (1970)
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as usize)
            .unwrap_or(0)
    });

    // Create dummy weight snapshot (would need actual implementation)
    let weights = WeightSnapshot::Memory {
        data: vec![], // Placeholder - would need actual weight data
        format: WeightFormat::SafeTensors,
    };

    // Write checkpoint
    match checkpoint_mgr.write_checkpoint(weights, step, None).await {
        Ok(path) => Json(CheckpointResponse {
            path: path.to_string_lossy().to_string(),
            step,
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to write checkpoint: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Commit checkpoint to Git
async fn commit_checkpoint(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<CommitRequest>,
) -> impl IntoResponse {
    // Extract model_id from checkpoint path for auth check
    let checkpoint_path = PathBuf::from(&req.checkpoint_path);
    let model_id = checkpoint_path
        .parent()
        .and_then(|p| {
            if p.file_name() == Some(std::ffi::OsStr::new(".checkpoints")) {
                p.parent().and_then(|mp| mp.file_name())
            } else {
                p.file_name()
            }
        })
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_owned());

    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{model_id}");
    if !state.policy_client.check_policy(&user, &resource, Operation::Write).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot write '{}'", user, resource)
            })),
        ).into_response();
    }

    let checkpoint_path = PathBuf::from(&req.checkpoint_path);

    // Get model path (parent of .checkpoints directory)
    let model_path = checkpoint_path
        .parent()
        .and_then(|p| {
            if p.file_name() == Some(std::ffi::OsStr::new(".checkpoints")) {
                p.parent()
            } else {
                Some(p)
            }
        })
        .ok_or("Invalid checkpoint path");

    let model_path = match model_path {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": e
                })),
            )
                .into_response();
        }
    };

    // Create checkpoint manager
    let checkpoint_mgr = match CheckpointManager::new(model_path.to_path_buf()) {
        Ok(mgr) => mgr,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to create checkpoint manager: {}", e)
                })),
            )
                .into_response();
        }
    };

    // Commit checkpoint
    match checkpoint_mgr
        .commit_checkpoint(&checkpoint_path, req.message, req.branch)
        .await
    {
        Ok(commit_id) => Json(CommitResponse { commit_id }).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to commit checkpoint: {}", e)
            })),
        )
            .into_response(),
    }
}

/// Start pre-training
async fn start_pretraining(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(req): Json<PreTrainingRequest>,
) -> impl IntoResponse {
    let user = server::extract_user(auth_user.as_ref());
    let resource = format!("model:{}", req.model_id);
    if !state.policy_client.check_policy(&user, &resource, Operation::Train).await.unwrap_or(false) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": format!("Permission denied: user '{}' cannot train '{}'", user, resource)
            })),
        ).into_response();
    }

    // For now, return a session ID
    // Full implementation would spawn a training task
    let session_id = uuid::Uuid::new_v4().to_string();

    Json(serde_json::json!({
        "status": "started",
        "session_id": session_id,
        "model_id": req.model_id,
        "config": {
            "learning_rate": req.learning_rate,
            "batch_size": req.batch_size,
            "warmup_steps": req.warmup_steps,
            "total_steps": req.total_steps,
        }
    }))
    .into_response()
}
