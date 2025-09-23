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
    runtime::{RuntimeEngine, GenerationRequest, FinishReason, TorchEngine},
    server::{
        state::{ServerState, ServerConfig},
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
        Self::new(message, "service_unavailable", "model_cache_exhausted")
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
) -> impl IntoResponse {
    // Debug log the incoming request
    info!("Chat completion request - model: {}, stream: {:?}, messages: {} msgs", 
        request.model, request.stream, request.messages.len());
    
    // Log if streaming is defaulting
    let is_streaming = request.stream.unwrap_or(false);
    info!("Streaming mode: {} (explicit: {:?})", is_streaming, request.stream);
    
    // Validate request
    if let Err(e) = validate_chat_request(&request, &state.config) {
        error!("Request validation failed: {}", e);
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
        info!("Handling streaming request");
        return stream_chat(state, request).await.into_response();
    }
    info!("Handling non-streaming request");
    
    // Get engine from model cache
    let engine = match state.model_cache.get_or_load(&request.model).await {
        Ok(engine) => engine,
        Err(e) => {
            error!("Failed to load model '{}': {}", request.model, e);
            state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            
            // Determine appropriate error response
            let (status, error_response) = if e.to_string().contains("not found") {
                (
                    StatusCode::NOT_FOUND,
                    ErrorResponse::new(
                        format!("Model '{}' not found", request.model),
                        "model_not_found",
                        "invalid_model"
                    )
                )
            } else if e.to_string().contains("failed to load") {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::new(
                        format!("Model '{}' failed to load: {}", request.model, e),
                        "model_load_error",
                        "internal_error"
                    )
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::internal_error(&e.to_string())
                )
            };
            
            return (status, Json(error_response)).into_response();
        }
    };
    
    // Create generation request - use template-aware formatting
    let defaults = &state.config.generation_defaults;
    info!("Formatting messages with template...");
    let formatted_prompt = match format_messages_with_template(&request.messages, &engine).await {
        Ok(prompt) => {
            info!("Template formatting successful, prompt preview: {}", 
                &prompt.chars().take(200).collect::<String>());
            prompt
        },
        Err(e) => {
            error!("Template formatting failed: {}", e);
            state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse::new(
                e,
                "template_error",
                "template_formatting_failed"
            ))).into_response();
        }
    };
    let gen_request = GenerationRequest {
        prompt: formatted_prompt,
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
    info!("Starting generation with prompt length: {}", gen_request.prompt.len());
    let result = {
        let engine_guard = engine.lock().await;
        engine_guard.generate_with_params(gen_request).await
    };
    
    info!("Generation completed - success: {}", result.is_ok());
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
) -> impl IntoResponse {
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
        // Get engine from model cache
        let engine_arc = match state.model_cache.get_or_load(&model_name).await {
            Ok(engine) => engine,
            Err(e) => {
                let error_msg = if e.to_string().contains("not found") {
                    format!("Model '{}' not found", model_name)
                } else {
                    e.to_string()
                };
                let _ = tx.send(Err(anyhow::anyhow!(error_msg))).await;
                return;
            }
        };
        
        // Format messages BEFORE locking the engine to avoid deadlock
        info!("Formatting messages for streaming...");
        let prompt = match format_messages_with_template(&messages, &engine_arc).await {
            Ok(p) => {
                info!("Streaming template formatting successful, prompt length: {}", p.len());
                p
            },
            Err(e) => {
                error!("Template formatting failed in streaming: {}", e);
                let _ = tx.send(Err(anyhow::anyhow!("Template formatting failed: {}", e))).await;
                return;
            }
        };
        
        // NOW lock the engine for generation
        let engine = engine_arc.lock().await;
        
        // Create SSE streaming callback (clone tx for callback)
        let tx_callback = tx.clone();
        let callback = Box::new(
            crate::runtime::streaming::SseStreamingCallback::new(tx_callback, model_name.clone())
        );
        
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
        info!("Starting streaming generation...");
        match engine.generate_streaming_async(gen_request, callback, context).await {
            Ok(result) => {
                info!("Streaming generation completed with {} tokens", result.tokens_generated);
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
) -> impl IntoResponse {
    // Similar to chat completions but with different format
    state.metrics.active_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state.metrics.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    // Get engine from model cache
    let engine = match state.model_cache.get_or_load(&request.model).await {
        Ok(engine) => engine,
        Err(e) => {
            state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            
            // Determine error type
            let (status, error_json) = if e.to_string().contains("not found") {
                (
                    StatusCode::NOT_FOUND,
                    serde_json::json!({
                        "error": {
                            "message": format!("Model '{}' not found", request.model),
                            "type": "model_not_found",
                            "code": "invalid_model"
                        }
                    })
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    serde_json::json!({
                        "error": {
                            "message": format!("Failed to load model: {}", e),
                            "type": "internal_error",
                            "code": "engine_error"
                        }
                    })
                )
            };
            
            return (status, Json(error_json)).into_response();
        }
    };
    
    // Convert raw prompt to chat format and apply template
    // The completions endpoint expects raw text, but modern models need templated input
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(request.prompt.clone()),
        function_call: None,
    }];
    
    let formatted_prompt = match format_messages_with_template(&messages, &engine).await {
        Ok(prompt) => prompt,
        Err(e) => {
            error!("Template formatting failed for completions endpoint: {}", e);
            state.metrics.active_requests.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": {
                    "message": format!("Template formatting failed: {}", e),
                    "type": "template_error",
                    "code": "template_formatting_failed"
                }
            }))).into_response();
        }
    };
    
    let gen_request = GenerationRequest {
        prompt: formatted_prompt,
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
) -> impl IntoResponse {
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
) -> impl IntoResponse {
    // Get models from storage and LoRA registry
    let mut models = vec![];
    
    // Get models from storage - the backend should provide properly formatted names
    match state.model_storage.list_models().await {
        Ok(model_list) => {
            for (model_ref, metadata) in model_list {
                // The backend should provide the correct display name
                // API layer should not be determining how to name models
                let model_name = metadata.display_name
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
    
    // Add cached models with a special indicator
    // This helps users know which models are already loaded and will be fast
    let cache_stats = state.model_cache.stats().await;
    if cache_stats.cached_models > 0 {
        // Mark cached models in the list
        for model in models.iter_mut() {
            if state.model_cache.is_cached_by_name(&model.id).await {
                // Append indicator that model is cached
                model.owned_by = format!("{} (cached)", model.owned_by);
            }
        }
    }
    
    // Add LoRA adapters as fine-tuned models
    if let Ok(loras) = state.adapter_storage.list_adapters().await {
        for (adapter_id, adapter_config) in loras {
            models.push(Model {
                id: format!("lora-{}", adapter_id),
                object: "model".to_string(),
                created: adapter_config.created_at.timestamp(),
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

/// Format chat messages into a prompt using template engine
async fn format_messages_with_template(
    messages: &[ChatMessage],
    engine: &Arc<Mutex<TorchEngine>>,
) -> Result<String, String> {
    // Log messages for debugging
    for (i, msg) in messages.iter().enumerate() {
        info!("Message {}: role={}, content={:?}", i, msg.role, 
            msg.content.as_ref().map(|c| {
                if c.len() <= 100 {
                    c.as_str()
                } else {
                    // Find the last character boundary before or at position 100
                    match c.char_indices().nth(99) {
                        Some((idx, _)) => &c[..idx],
                        None => c.as_str(), // Less than 100 characters
                    }
                }
            }));
    }
    
    // Convert OpenAI messages to template engine format
    let template_messages: Vec<crate::runtime::template_engine::ChatMessage> = messages
        .iter()
        .map(|msg| crate::runtime::template_engine::ChatMessage {
            role: msg.role.clone(),
            content: msg.content.as_ref().unwrap_or(&String::new()).clone(),
        })
        .collect();
    
    info!("Acquiring engine lock for template formatting...");
    // Apply the engine's template formatting
    let engine_guard = engine.lock().await;
    info!("Engine lock acquired, applying template...");
    let result = engine_guard.apply_chat_template(&template_messages, true)
        .map_err(|e| format!("Failed to apply chat template: {}", e));
    info!("Template application complete: {:?}", result.is_ok());
    result
}

