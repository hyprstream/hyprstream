//! Training service endpoints

use axum::{
    Router,
    routing::{get, post},
    extract::{State, Path, Json},
    response::IntoResponse,
    http::StatusCode,
};
use serde::Deserialize;
use crate::{
    api::training_service::TrainingConfig,
    api::TrainingStatus,
    server::state::ServerState,
};

/// Create training service router
pub fn create_router() -> Router<ServerState> {
    Router::new()
        .route("/:lora_id/start", post(start_training))
        .route("/:lora_id/stop", post(stop_training))
        .route("/:lora_id/status", get(get_status))
        .route("/:lora_id/submit", post(submit_samples))
}

/// Training sample submission
#[derive(Debug, Deserialize)]
struct TrainingSample {
    input: String,
    output: String,
}

/// Batch training submission
#[derive(Debug, Deserialize)]
struct SubmitSamplesRequest {
    samples: Vec<TrainingSample>,
}

/// Start training for a LoRA adapter
async fn start_training(
    State(state): State<ServerState>,
    Path(lora_id): Path<String>,
    Json(config): Json<TrainingConfig>,
) -> impl IntoResponse {
    match state.training_service.start_auto_training(&lora_id, config).await {
        Ok(_) => {
            Json(serde_json::json!({
                "status": "started",
                "lora_id": lora_id
            })).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to start training: {}", e)
            }))).into_response()
        }
    }
}

/// Stop training for a LoRA adapter
async fn stop_training(
    State(state): State<ServerState>,
    Path(lora_id): Path<String>,
) -> impl IntoResponse {
    match state.training_service.stop_auto_training(&lora_id).await {
        Ok(_) => {
            Json(serde_json::json!({
                "status": "stopped",
                "lora_id": lora_id
            })).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to stop training: {}", e)
            }))).into_response()
        }
    }
}

/// Get training status
async fn get_status(
    State(state): State<ServerState>,
    Path(lora_id): Path<String>,
) -> impl IntoResponse {
    match state.training_service.get_training_status(&lora_id).await {
        Ok(status) => Json(status).into_response(),
        Err(_) => {
            Json(TrainingStatus {
                is_training: false,
                total_samples_processed: 0,
                current_loss: 0.0,
                learning_rate: 0.0,
                last_update: 0,
            }).into_response()
        }
    }
}

/// Submit training samples
async fn submit_samples(
    State(state): State<ServerState>,
    Path(lora_id): Path<String>,
    Json(request): Json<SubmitSamplesRequest>,
) -> impl IntoResponse {
    let mut submitted = 0;
    
    for sample in request.samples {
        let training_sample = crate::api::TrainingSample {
            input: sample.input,
            output: sample.output,
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        if state.training_service.queue_training_sample(&lora_id, training_sample).await.is_ok() {
            submitted += 1;
        }
    }
    
    Json(serde_json::json!({
        "status": "submitted",
        "lora_id": lora_id,
        "samples_submitted": submitted
    })).into_response()
}