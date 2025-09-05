//! OpenAI-compatible API endpoints

use axum::{
    Router,
    routing::{get, post},
    extract::{State, Json},
    response::{IntoResponse, Response, Sse},
    http::{StatusCode, header},
};
use futures::stream::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, error};


use crate::{
    api::openai_compat::{
        ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage,
        CompletionRequest, CompletionResponse, CompletionChoice,
        EmbeddingRequest, EmbeddingResponse, EmbeddingData,
        ListModelsResponse, Model, Usage,
        OpenAIStreamResponse, StreamChoice, Delta,
    },
    runtime::{RuntimeEngine, GenerationRequest, FinishReason},
    server::{
        state::{ServerState, ServerConfig},
        engine_pool::EnginePoolError,
    },
};

/// RAII guard for metrics cleanup
struct MetricsGuard<'a> {
    metrics: &'a crate::server::state::Metrics,
    decremented: bool,
}

impl<'a> MetricsGuard<'a> {
    fn new(metrics: &'a crate::server::state::Metrics) -> Self {
        Self { metrics, decremented: false }
    }
    
    fn mark_completed(&mut self) {
        if !self.decremented {
            self.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            self.decremented = true;
        }
    }
}

impl<'a> Drop for MetricsGuard<'a> {
    fn drop(&mut self) {
        if !self.decremented {
            self.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Standard error response format
#[derive(serde::Serialize)]
struct ErrorResponse {
    error: ErrorDetails,
}

#[derive(serde::Serialize)]
struct ErrorDetails {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: String,
}

impl ErrorResponse {
    fn new(message: impl Into<String>, error_type: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: ErrorDetails {
                message: message.into(),
                error_type: error_type.into(),
                code: code.into(),
            }
        }
    }
    
    fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(message, "service_unavailable", "engine_pool_exhausted")
    }
    
    #[allow(dead_code)]
    fn internal_error(message: impl Into<String>) -> Self {
        Self::new(message, "internal_error", "generation_failed")
    }
}

/// Validate chat completion request
fn validate_chat_request(request: &ChatCompletionRequest, config: &ServerConfig) -> Result<(), String> {
    // Validate messages exist and are not empty
    if request.messages.is_empty() {
        return Err("Messages array cannot be empty".to_string());
    }
    
    // Validate max_tokens limit
    if let Some(max_tokens) = request.max_tokens {
        if max_tokens > config.max_tokens_limit {
            return Err(format!("max_tokens ({}) exceeds limit ({})", max_tokens, config.max_tokens_limit));
        }
        if max_tokens < 1 {
            return Err("max_tokens must be at least 1".to_string());
        }
    }
    
    // Validate temperature
    if let Some(temp) = request.temperature {
        if !(0.0..=2.0).contains(&temp) {
            return Err(format!("temperature must be between 0.0 and 2.0, got {}", temp));
        }
    }
    
    // Validate top_p
    if let Some(top_p) = request.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(format!("top_p must be between 0.0 and 1.0, got {}", top_p));
        }
    }
    
    // Validate n (number of completions)
    if let Some(n) = request.n {
        if n < 1 || n > 10 {
            return Err(format!("n must be between 1 and 10, got {}", n));
        }
    }
    
    Ok(())
}

/// Create OpenAI API router
pub fn create_router() -> Router<ServerState> {
    Router::new()
        .route("/chat/completions", post(chat_completions))
        .route("/completions", post(completions))
        .route("/embeddings", post(embeddings))
        .route("/models", get(list_models))
}

/// Handle chat completion requests
async fn chat_completions(
    State(state): State<ServerState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    // Validate request
    if let Err(e) = validate_chat_request(&request, &state.config) {
        return (StatusCode::BAD_REQUEST, Json(ErrorResponse::new(
            e,
            "invalid_request",
            "validation_failed"
        ))).into_response();
    }
    
    // Update metrics
    state.metrics.active_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state.metrics.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    let start_time = std::time::Instant::now();
    
    // Check for streaming
    if request.stream.unwrap_or(false) {
        return stream_chat(state, request).await;
    }
    
    // Acquire engine from pool - let the pool handle model loading
    // The API layer should not be concerned with HOW models are loaded
    let engine_guard = match state.engine_pool.read().await.acquire_model(&request.model).await {
        Ok(guard) => guard,
        Err(e) => {
            error!("Failed to acquire engine for model '{}': {}", request.model, e);
            state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            
            // Determine appropriate error response based on error type
            let (status, error_response) = match e {
                EnginePoolError::ModelNotFound(model) => (
                    StatusCode::NOT_FOUND,
                    ErrorResponse::new(
                        format!("Model '{}' not found", model),
                        "model_not_found",
                        "invalid_model"
                    )
                ),
                EnginePoolError::ModelLoadFailed(model, err) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::new(
                        format!("Model '{}' failed to load: {}", model, err),
                        "model_load_error",
                        "internal_error"
                    )
                ),
                EnginePoolError::NoEnginesAvailable | EnginePoolError::PoolExhausted => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    ErrorResponse::service_unavailable("No engines available")
                ),
                EnginePoolError::Internal(err) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::internal_error(&err)
                ),
            };
            
            return (status, Json(error_response)).into_response();
        }
    };
    
    let engine = engine_guard.get();
    
    // Create generation request - let the backend handle message formatting
    let defaults = &state.config.generation_defaults;
    let gen_request = GenerationRequest {
        prompt: format_messages(&request.messages),
        max_tokens: request.max_tokens.unwrap_or(defaults.max_tokens),
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        top_k: None,
        repeat_penalty: defaults.repeat_penalty,
        stop_tokens: request.stop.clone().unwrap_or_default(),
        seed: None,
        stream: false,
        active_adapters: None,
        realtime_adaptation: None,
        user_feedback: None,
    };
    
    // Generate response
    let result = {
        let engine = engine.lock().await;
        engine.generate_with_params(gen_request).await
    };
    
    state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    
    match result {
        Ok(generation) => {
            // Update metrics
            state.metrics.total_tokens.fetch_add(
                generation.tokens_generated as u64,
                std::sync::atomic::Ordering::Relaxed
            );
            
            let latency_ms = start_time.elapsed().as_millis() as f64;
            let mut avg_latency = state.metrics.avg_latency_ms.write().await;
            *avg_latency = (*avg_latency * 0.9) + (latency_ms * 0.1); // Exponential moving average
            
            // Create response
            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: request.model.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(generation.text),
                        function_call: None,
                    },
                    finish_reason: Some(match generation.finish_reason {
                        FinishReason::MaxTokens => "length",
                        FinishReason::StopToken(_) => "stop",
                        FinishReason::EndOfSequence => "stop",
                        FinishReason::Stop => "stop",
                        FinishReason::Error(_) => "stop",
                    }.to_string()),
                }],
                usage: Some(Usage {
                    prompt_tokens: 0,
                    completion_tokens: generation.tokens_generated,
                    total_tokens: generation.tokens_generated,
                }),
            };
            
            // Add no-cache headers to prevent client caching
            let mut response = Json(response).into_response();
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                "no-cache, no-store, must-revalidate".parse().unwrap()
            );
            response.headers_mut().insert(
                header::PRAGMA,
                "no-cache".parse().unwrap()
            );
            response
        }
        Err(e) => {
            error!("Generation failed: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": {
                    "message": format!("Generation failed: {}", e),
                    "type": "generation_error",
                    "code": "internal_error"
                }
            }))).into_response()
        }
    }
}

/// Handle streaming chat completions with real token-by-token generation
async fn stream_chat(
    state: ServerState,
    request: ChatCompletionRequest,
) -> Response {
    // Create channel for SSE events
    let (tx, rx) = mpsc::channel::<Result<serde_json::Value, anyhow::Error>>(100);
    let cancel_token = CancellationToken::new();
    let cancel_token_clone = cancel_token.clone();
    
    // Clone state for metrics cleanup
    let state_clone = state.clone();
    
    // Spawn generation task with configured defaults
    let defaults = state.config.generation_defaults.clone();
    let model_name = request.model.clone();
    let messages = request.messages.clone();
    let max_tokens = request.max_tokens.unwrap_or(defaults.max_tokens);
    let temperature = request.temperature.unwrap_or(defaults.temperature);
    let top_p = request.top_p.unwrap_or(defaults.top_p);
    let stop_sequences = request.stop.clone().unwrap_or_default();
    
    // Track active request for proper cleanup
    state_clone.metrics.active_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    tokio::spawn(async move {
        // Ensure metrics are decremented on all exit paths
        let _metrics_guard = MetricsGuard::new(&state.metrics);
        // Acquire engine with the requested model
        let engine_guard = match state.engine_pool.read().await.acquire_model(&model_name).await {
            Ok(guard) => guard,
            Err(e) => {
                let error_msg = match e {
                    EnginePoolError::ModelNotFound(m) => {
                        format!("Model '{}' not found", m)
                    }
                    _ => e.to_string()
                };
                let _ = tx.send(Err(anyhow::anyhow!(error_msg))).await;
                return;
            }
        };
        
        let engine_arc = engine_guard.get();
        let engine = engine_arc.lock().await;
        
        // Create SSE streaming callback (clone tx for callback)
        let tx_callback = tx.clone();
        let callback = Box::new(
            crate::runtime::streaming::SseStreamingCallback::new(tx_callback, model_name)
        );
        
        // Create generation request
        let prompt = format_messages(&messages);
        let gen_request = GenerationRequest {
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k: None,
            repeat_penalty: defaults.repeat_penalty,
            stop_tokens: stop_sequences,
            stream: true,
            seed: None,
            active_adapters: None,
            realtime_adaptation: None,
            user_feedback: None,
        };
        
        // Create generation context
        let context = crate::runtime::streaming::GenerationContext {
            cancel_token: cancel_token_clone,
            timeout: std::time::Duration::from_secs(defaults.stream_timeout_secs),
        };
        
        // Generate with async streaming
        match engine.generate_streaming_async(gen_request, callback, context).await {
            Ok(result) => {
                // Update metrics
                state.metrics.total_tokens.fetch_add(
                    result.tokens_generated as u64,
                    std::sync::atomic::Ordering::Relaxed
                );
                
                // Send [DONE] message
                let _ = tx.send(Ok(serde_json::json!({
                    "done": true
                }))).await;
            }
            Err(e) => {
                error!("Streaming generation failed: {}", e);
                let _ = tx.send(Err(e)).await;
            }
        }
        // Metrics are automatically decremented by MetricsGuard drop
    });
    
    // Convert channel to SSE stream
    let stream = ReceiverStream::new(rx).map(|result| {
        match result {
            Ok(json) => {
                if json.get("done").is_some() {
                    // Send [DONE] message
                    Ok::<_, Infallible>(axum::response::sse::Event::default().data("[DONE]"))
                } else {
                    // Send data chunk
                    Ok(axum::response::sse::Event::default()
                        .data(format!("{}", serde_json::to_string(&json).unwrap())))
                }
            }
            Err(e) => {
                // Send error
                Ok(axum::response::sse::Event::default()
                    .data(format!("{{\"error\": \"{}\"}}", e)))
            }
        }
    });
    
    // Set up SSE response with keep-alive and no-cache headers
    let mut response = Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new())
        .into_response();
    
    // Add cache control headers to prevent caching
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        "no-cache, no-store, must-revalidate".parse().unwrap()
    );
    response.headers_mut().insert(
        header::PRAGMA,
        "no-cache".parse().unwrap()
    );
    response.headers_mut().insert(
        header::EXPIRES,
        "0".parse().unwrap()
    );
    
    response
}

/// Handle text completion requests
async fn completions(
    State(state): State<ServerState>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    // Similar to chat completions but with different format
    state.metrics.active_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state.metrics.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    // Acquire engine with the requested model - let the pool handle everything
    let engine_guard = match state.engine_pool.read().await.acquire_model(&request.model).await {
        Ok(guard) => guard,
        Err(e) => {
            state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            
            // Use proper error matching
            let (status, error_json) = match e {
                EnginePoolError::ModelNotFound(model) => (
                    StatusCode::NOT_FOUND,
                    serde_json::json!({
                        "error": {
                            "message": format!("Model '{}' not found", model),
                            "type": "model_not_found",
                            "code": "invalid_model"
                        }
                    })
                ),
                EnginePoolError::NoEnginesAvailable | EnginePoolError::PoolExhausted => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    serde_json::json!({
                        "error": {
                            "message": "No engines available",
                            "type": "service_unavailable",
                            "code": "engine_pool_exhausted"
                        }
                    })
                ),
                _ => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    serde_json::json!({
                        "error": {
                            "message": format!("Failed to acquire engine: {}", e),
                            "type": "internal_error",
                            "code": "engine_error"
                        }
                    })
                )
            };
            
            return (status, Json(error_json)).into_response();
        }
    };
    
    let engine = engine_guard.get();
    
    let gen_request = GenerationRequest {
        prompt: request.prompt.clone(),
        max_tokens: request.max_tokens.unwrap_or(2048),
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: None,
        repeat_penalty: 1.1,
        stop_tokens: request.stop.clone().unwrap_or_default(),
        seed: None,
        stream: request.stream.unwrap_or(false),
        active_adapters: None,
        realtime_adaptation: None,
        user_feedback: None,
    };
    
    let result = {
        let engine = engine.lock().await;
        engine.generate_with_params(gen_request).await
    };
    
    state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    
    match result {
        Ok(generation) => {
            let response = CompletionResponse {
                id: format!("cmpl-{}", uuid::Uuid::new_v4()),
                object: "text_completion".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: request.model.clone(),
                choices: vec![CompletionChoice {
                    text: generation.text,
                    index: 0,
                    logprobs: None,
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Some(Usage {
                    prompt_tokens: request.prompt.len() / 4, // Rough estimate: 4 chars per token
                    completion_tokens: generation.tokens_generated,
                    total_tokens: request.prompt.len() / 4 + generation.tokens_generated,
                }),
            };
            
            // Add no-cache headers to prevent client caching
            let mut response = Json(response).into_response();
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                "no-cache, no-store, must-revalidate".parse().unwrap()
            );
            response.headers_mut().insert(
                header::PRAGMA,
                "no-cache".parse().unwrap()
            );
            response
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": {
                    "message": format!("Generation failed: {}", e),
                    "type": "generation_error",
                    "code": "internal_error"
                }
            }))).into_response()
        }
    }
}

/// Handle embedding requests
async fn embeddings(
    State(_state): State<ServerState>,
    Json(_request): Json<EmbeddingRequest>,
) -> Response {
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": {
            "message": "Embeddings not yet implemented",
            "type": "not_implemented",
            "code": "feature_not_available"
        }
    }))).into_response()
}

/// List available models
async fn list_models(
    State(state): State<ServerState>,
) -> Response {
    // Get models from storage and LoRA registry
    let mut models = vec![];
    
    // Get models from storage - the backend should provide properly formatted names
    match state.model_storage.children().await {
        Ok(model_list) => {
            for (_model_id, metadata) in model_list {
                // The backend should provide the correct display name
                // API layer should not be determining how to name models
                let model_name = metadata.display_name
                    .or_else(|| {
                        metadata.local_path.as_ref()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or(metadata.name);
                
                models.push(Model {
                    id: model_name,
                    object: "model".to_string(),
                    created: metadata.created_at,
                    owned_by: "system".to_string(),
                });
            }
        }
        Err(e) => {
            error!("Failed to list models from storage: {}", e);
        }
    }
    
    // Add currently loaded model if available and not in list
    if let Some(ref default_model) = state.config.default_model {
        // Check if this model is already in the list
        let model_name = std::path::Path::new(default_model)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(default_model);
        
        if !models.iter().any(|m| m.id == model_name) {
            models.push(Model {
                id: model_name.to_string(),
                object: "model".to_string(),
                created: chrono::Utc::now().timestamp(),
                owned_by: "system".to_string(),
            });
        }
    }
    
    // Add LoRA adapters as fine-tuned models
    if let Ok(loras) = state.lora_registry.list_all().await {
        for lora in loras {
            models.push(Model {
                id: format!("lora-{}", lora.id),
                object: "model".to_string(),
                created: lora.created_at,
                owned_by: "user".to_string(),
            });
        }
    }
    
    // Add no-cache headers to prevent client caching
    let mut response = Json(ListModelsResponse {
        object: "list".to_string(),
        data: models,
    }).into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        "no-cache, no-store, must-revalidate".parse().unwrap()
    );
    response.headers_mut().insert(
        header::PRAGMA,
        "no-cache".parse().unwrap()
    );
    response
}

/// Format chat messages into a prompt
fn format_messages(messages: &[ChatMessage]) -> String {
    messages.iter()
        .map(|msg| {
            format!("{}: {}", 
                msg.role, 
                msg.content.as_ref().unwrap_or(&String::new())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

