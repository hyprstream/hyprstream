//! OpenAI-compatible API endpoints

use axum::{
    extract::{Json, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Sse},
    routing::{get, post},
    Router,
};
use futures::stream::StreamExt;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info};

use crate::{
    api::openai_compat::{
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
        CompletionRequest, CompletionResponse, EmbeddingRequest, ListModelsResponse, Model, Usage,
    },
    config::TrainingMode,
    runtime::{model_config::ModelConfig, CacheOwner, FinishReason, GenerationRequest, RuntimeEngine, TorchEngine},
    server::state::ServerState,
    training::{ReplayBufferConfig, SelfSupervisedConfig, SelfSupervisedTrainer},
};

/// RAII guard for metrics cleanup
struct MetricsGuard<'a> {
    metrics: &'a crate::server::state::Metrics,
    decremented: bool,
}

impl<'a> MetricsGuard<'a> {
    fn new(metrics: &'a crate::server::state::Metrics) -> Self {
        Self {
            metrics,
            decremented: false,
        }
    }
}

impl<'a> Drop for MetricsGuard<'a> {
    fn drop(&mut self) {
        if !self.decremented {
            self.metrics
                .active_requests
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
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
    fn new(
        message: impl Into<String>,
        error_type: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            error: ErrorDetails {
                message: message.into(),
                error_type: error_type.into(),
                code: code.into(),
            },
        }
    }

    #[allow(dead_code)]
    fn internal_error(message: impl Into<String>) -> Self {
        Self::new(message, "internal_error", "generation_failed")
    }
}

// validate_chat_request removed - validation now handled by streaming pipeline
// The TextStream and sampling code handle all parameter edge cases safely:
// - Empty messages → empty response (safe)
// - Invalid temperature/top_p → clamped or falls back to greedy (safe)
// - max_tokens → enforced by generation loop (safe)
// Rate limiting should be handled at middleware layer, not per-endpoint

/// Extract cache owner from request headers.
///
/// Session management:
/// - `x-session-id` header: Uses session-based caching (context preserved)
/// - No header: Generates a stateless request ID (context not preserved)
///
/// The session ID is used to route the request to the appropriate KV cache,
/// allowing multiple concurrent sessions to have isolated context.
fn extract_cache_owner(headers: &HeaderMap) -> CacheOwner {
    if let Some(session_id) = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
    {
        debug!("Using session-based caching for session: {}", session_id);
        CacheOwner::Session(session_id.to_string())
    } else {
        // Stateless request - generate unique ID
        let request_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        debug!("Using stateless caching for request: {}", request_id);
        CacheOwner::Stateless(request_id)
    }
}

/// Helper: Load model from cache with proper error handling
async fn load_model_or_error(
    state: &ServerState,
    model_name: &str,
) -> Result<Arc<Mutex<TorchEngine>>, impl IntoResponse> {
    match state.model_cache.get_or_load(model_name).await {
        Ok(engine) => Ok(engine),
        Err(e) => {
            error!("Failed to load model '{}': {}", model_name, e);

            let (status, error_response) = if e.to_string().contains("not found") {
                (
                    StatusCode::NOT_FOUND,
                    ErrorResponse::new(
                        format!("Model '{}' not found", model_name),
                        "model_not_found",
                        "invalid_model",
                    ),
                )
            } else if e.to_string().contains("failed to load") {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::new(
                        format!("Model '{}' failed to load: {}", model_name, e),
                        "model_load_error",
                        "internal_error",
                    ),
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorResponse::internal_error(e.to_string()),
                )
            };

            Err((status, Json(error_response)).into_response())
        }
    }
}

/// Helper: Resolve model name to filesystem path
async fn resolve_model_path(
    state: &ServerState,
    model_name: &str,
) -> Result<std::path::PathBuf, impl IntoResponse> {
    let model_ref = match crate::storage::ModelRef::parse(model_name) {
        Ok(r) => r,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    format!("Invalid model reference: {}", e),
                    "invalid_model_ref",
                    "parse_error",
                )),
            )
                .into_response());
        }
    };

    match state.model_storage.get_model_path(&model_ref).await {
        Ok(path) => Ok(path),
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                format!("Model path not found: {}", e),
                "model_not_found",
                "path_error",
            )),
        )
            .into_response()),
    }
}

/// Helper: Add no-cache headers to response
fn add_no_cache_headers(response: &mut axum::response::Response) {
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        "no-cache, no-store, must-revalidate".parse().unwrap(),
    );
    headers.insert(header::PRAGMA, "no-cache".parse().unwrap());
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
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Extract session/request ID for KV cache routing
    let cache_owner = extract_cache_owner(&headers);

    // Debug log the incoming request
    info!(
        "Chat completion request - model: {}, stream: {:?}, messages: {} msgs, cache: {:?}",
        request.model,
        request.stream,
        request.messages.len(),
        cache_owner
    );

    // Log if streaming is defaulting
    let is_streaming = request.stream.unwrap_or(false);
    info!(
        "Streaming mode: {} (explicit: {:?})",
        is_streaming, request.stream
    );

    if request.stream.unwrap_or(false) {
        info!("Handling streaming request");
        return stream_chat(state, headers, request).await.into_response();
    }
    info!("Handling non-streaming request");

    state
        .metrics
        .active_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state
        .metrics
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _metrics_guard = MetricsGuard::new(&state.metrics);

    let start_time = std::time::Instant::now();

    let engine = match load_model_or_error(&state, &request.model).await {
        Ok(engine) => engine,
        Err(response) => return response.into_response(),
    };

    info!("Formatting messages with template...");
    let formatted_prompt = match format_messages_with_template(&request.messages, &engine).await {
        Ok(prompt) => {
            info!(
                "Template formatting successful, prompt preview: {}",
                &prompt.chars().take(200).collect::<String>()
            );
            prompt
        }
        Err(e) => {
            error!("Template formatting failed: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    e,
                    "template_error",
                    "template_formatting_failed",
                )),
            )
                .into_response();
        }
    };

    let model_path = match resolve_model_path(&state, &request.model).await {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };

    let gen_request = GenerationRequest::builder(formatted_prompt)
        .apply_config(&(&state.config.sampling_defaults).into())
        .apply_config(&crate::config::SamplingParams::from_model_path(&model_path).await.unwrap_or_default())
        .apply_config(&(&request).into())
        .build();

    info!(
        "Using generation config: max_tokens={}, temp={}, top_p={}, top_k={:?}, repeat_penalty={}",
        gen_request.max_tokens, gen_request.temperature, gen_request.top_p, gen_request.top_k, gen_request.repeat_penalty
    );

    info!(
        "Starting generation with prompt length: {}",
        gen_request.prompt.len()
    );

    // Determine if this is a stateless request (for cleanup after generation)
    let is_stateless = matches!(cache_owner, CacheOwner::Stateless(_));

    let result = {
        let engine_guard = engine.lock().await;

        // Set session for KV cache routing (if registry is initialized)
        if let Err(e) = engine_guard.set_session(cache_owner.clone()) {
            debug!("Could not set session (registry may not be initialized): {}", e);
        }

        engine_guard.generate_with_params(gen_request).await
    };

    // Release stateless caches to free memory (session caches are preserved)
    if is_stateless {
        let engine_guard = engine.lock().await;
        if let Err(e) = engine_guard.release_session(&cache_owner) {
            debug!("Could not release session cache: {}", e);
        }
    }

    info!("Generation completed - success: {}", result.is_ok());

    match result {
        Ok(generation) => {
            state.metrics.total_tokens.fetch_add(
                generation.tokens_generated as u64,
                std::sync::atomic::Ordering::Relaxed,
            );

            let latency_ms = start_time.elapsed().as_millis() as f64;
            let mut avg_latency = state.metrics.avg_latency_ms.write().await;
            *avg_latency = (*avg_latency * 0.9) + (latency_ms * 0.1);

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
                    finish_reason: Some(
                        match generation.finish_reason {
                            FinishReason::MaxTokens => "length",
                            FinishReason::StopToken(_) => "stop",
                            FinishReason::EndOfSequence => "stop",
                            FinishReason::Stop => "stop",
                            FinishReason::Error(_) => "stop",
                        }
                        .to_string(),
                    ),
                }],
                usage: Some(Usage {
                    prompt_tokens: 0,
                    completion_tokens: generation.tokens_generated,
                    total_tokens: generation.tokens_generated,
                }),
            };

            let mut response = Json(response).into_response();
            add_no_cache_headers(&mut response);
            response
        }
        Err(e) => {
            error!("Generation failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Generation failed: {}", e),
                        "type": "generation_error",
                        "code": "internal_error"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Handle streaming chat completions using TextStream
async fn stream_chat(state: ServerState, headers: HeaderMap, request: ChatCompletionRequest) -> impl IntoResponse {
    // Extract session/request ID for KV cache routing
    let cache_owner = extract_cache_owner(&headers);
    let is_stateless = matches!(cache_owner, CacheOwner::Stateless(_));

    // Create channel for SSE events
    let (tx, rx) = mpsc::channel::<Result<serde_json::Value, anyhow::Error>>(100);

    // Clone state for metrics cleanup
    let state_clone = state.clone();

    // Spawn generation task with configured defaults
    let defaults = state.config.sampling_defaults.clone();
    let model_name = request.model.clone();
    let messages = request.messages.clone();
    let stop_sequences = request.stop.clone().unwrap_or_default();

    // Track active request for proper cleanup
    state_clone
        .metrics
        .active_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    tokio::spawn(async move {
        use futures::StreamExt;

        // Ensure metrics are decremented on all exit paths
        let _metrics_guard = MetricsGuard::new(&state.metrics);

        // Get model path for config loading
        let model_ref = match crate::storage::ModelRef::parse(&model_name) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("Invalid model reference: {}", e))).await;
                return;
            }
        };

        let model_path = match state.model_storage.get_model_path(&model_ref).await {
            Ok(path) => path,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("Could not get model path: {}", e))).await;
                return;
            }
        };

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
                info!(
                    "Streaming template formatting successful, prompt length: {}",
                    p.len()
                );
                p
            }
            Err(e) => {
                error!("Template formatting failed in streaming: {}", e);
                let _ = tx
                    .send(Err(anyhow::anyhow!("Template formatting failed: {}", e)))
                    .await;
                return;
            }
        };

        // NOW lock the engine for generation
        let engine = engine_arc.lock().await;

        // Set session for KV cache routing (if registry is initialized)
        if let Err(e) = engine.set_session(cache_owner.clone()) {
            debug!("Could not set session (registry may not be initialized): {}", e);
        }

        // Keep a copy for training (prompt is moved into request builder)
        let prompt_for_training = prompt.clone();

        let gen_request = GenerationRequest::builder(prompt)
            .apply_config(&(&defaults).into())
            .apply_config(&crate::config::SamplingParams::from_model_path(&model_path).await.unwrap_or_default())
            .apply_config(&(&request).into())
            .stop_tokens(stop_sequences)
            .build();

        info!(
            "Streaming: max_tokens={}, temp={}, top_p={}, top_k={:?}, repeat_penalty={}",
            gen_request.max_tokens, gen_request.temperature, gen_request.top_p, gen_request.top_k, gen_request.repeat_penalty
        );

        // Create text stream - handles all decoding internally
        let mut text_stream = match engine.generate(gen_request) {
            Ok(stream) => stream,
            Err(e) => {
                error!("Failed to create text stream: {}", e);
                let _ = tx.send(Err(e)).await;
                return;
            }
        };

        // Send initial role message
        let stream_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let initial_msg = serde_json::json!({
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": null
            }]
        });
        let _ = tx.send(Ok(initial_msg)).await;

        // Consume text chunks from stream (collect for training)
        info!("Starting streaming generation...");
        let mut full_response = String::new();
        while let Some(text_result) = text_stream.next().await {
            match text_result {
                Ok(text) => {
                    // Collect for training
                    full_response.push_str(&text);

                    // Send text chunk
                    let chunk = serde_json::json!({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": chrono::Utc::now().timestamp(),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": text
                            },
                            "finish_reason": null
                        }]
                    });

                    if tx.send(Ok(chunk)).await.is_err() {
                        // Client disconnected - drop the stream to stop generation
                        info!("Client disconnected during streaming");
                        break;
                    }
                }
                Err(e) => {
                    // Generation error
                    error!("Generation error: {}", e);
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
        }

        // Get final statistics
        let stats = text_stream.stats();
        info!(
            "Streaming generation completed with {} tokens in {}ms ({:.2} tokens/sec)",
            stats.tokens_generated, stats.generation_time_ms, stats.tokens_per_second
        );

        // Update metrics
        state.metrics.total_tokens.fetch_add(
            stats.tokens_generated as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Collect training example if self-supervised training is enabled
        if let Some(ref qm) = stats.quality_metrics {
            // Check if training is enabled for this model
            if let Some(tc) = ModelConfig::load_training_config(&model_path) {
                if tc.is_enabled() && tc.mode == TrainingMode::SelfSupervised {
                    // Get or create trainer for this model
                    let trainer = state.trainers
                        .entry(model_name.clone())
                        .or_insert_with(|| {
                            let ss_config = SelfSupervisedConfig {
                                learning_rate: tc.learning_rate,
                                batch_size: tc.batch_size,
                                min_buffer_size: tc.min_buffer_size,
                                steps_per_cycle: tc.steps_per_cycle,
                                ..Default::default()
                            };
                            let buffer_config = ReplayBufferConfig {
                                min_quality_threshold: tc.min_quality_threshold,
                                ..Default::default()
                            };
                            Arc::new(SelfSupervisedTrainer::new(ss_config, buffer_config))
                        })
                        .clone();

                    // Tokenize for training using the engine
                    let tokenize = |text: &str| -> Result<Vec<i64>, anyhow::Error> {
                        let engine_guard = engine_arc.blocking_lock();
                        let tokenizer = engine_guard.get_tokenizer()?;
                        let encoding = tokenizer
                            .encode(text, false)
                            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
                    };

                    match (tokenize(&prompt_for_training), tokenize(&full_response)) {
                        (Ok(prompt_tokens), Ok(response_tokens)) => {
                            // Extract session ID if present for session-based filtering
                            let session_id = match &cache_owner {
                                CacheOwner::Session(id) => Some(id.clone()),
                                _ => None,
                            };

                            trainer
                                .add_example(prompt_tokens, response_tokens, qm.clone(), session_id)
                                .await;

                            debug!(
                                "Training example collected for model {}: quality={:.3}, buffer_size={}",
                                model_name,
                                qm.quality_score(),
                                trainer.replay_buffer.len().await
                            );

                            // Note: Training cycles are triggered asynchronously
                            // We don't block the response - training happens in background
                            if trainer.ready_to_train().await {
                                let trainer_clone = trainer.clone();
                                let engine_clone = engine_arc.clone();
                                tokio::spawn(async move {
                                    let mut engine = engine_clone.lock().await;
                                    match trainer_clone.train_cycle(&mut engine).await {
                                        Ok(result) => {
                                            info!(
                                                "Training cycle complete: {} steps, loss={:.4}",
                                                result.steps, result.total_loss
                                            );
                                        }
                                        Err(e) => {
                                            error!("Training cycle failed: {}", e);
                                        }
                                    }
                                });
                            }
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            debug!("Failed to tokenize for training: {}", e);
                        }
                    }
                }
            }
        }

        // Send completion message with finish reason
        let finish_reason = match stats.finish_reason {
            Some(FinishReason::MaxTokens) => "length",
            Some(FinishReason::StopToken(_)) => "stop",
            Some(FinishReason::EndOfSequence) => "stop",
            Some(FinishReason::Stop) => "stop",
            Some(FinishReason::Error(_)) => "stop",
            None => "stop",
        };

        let completion_msg = serde_json::json!({
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        let _ = tx.send(Ok(completion_msg)).await;

        // Send [DONE] message
        let _ = tx.send(Ok(serde_json::json!({"done": true}))).await;

        // Release stateless caches to free memory (session caches are preserved)
        // Need to drop engine lock first, then reacquire for release
        drop(engine);
        if is_stateless {
            let engine_guard = engine_arc.lock().await;
            if let Err(e) = engine_guard.release_session(&cache_owner) {
                debug!("Could not release session cache: {}", e);
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
                        .data(serde_json::to_string(&json).unwrap()))
                }
            }
            Err(e) => {
                // Send error
                Ok(axum::response::sse::Event::default().data(format!("{{\"error\": \"{}\"}}", e)))
            }
        }
    });

    // Set up SSE response with keep-alive and no-cache headers
    let mut response = Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::new())
        .into_response();

    // Add cache control headers (using helper)
    add_no_cache_headers(&mut response);
    response
        .headers_mut()
        .insert(header::EXPIRES, "0".parse().unwrap());

    response
}

/// Handle text completion requests
async fn completions(
    State(state): State<ServerState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Update metrics (use RAII guard for automatic cleanup)
    state
        .metrics
        .active_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    state
        .metrics
        .total_requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _metrics_guard = MetricsGuard::new(&state.metrics);

    // Get engine from model cache (using helper)
    let engine = match load_model_or_error(&state, &request.model).await {
        Ok(engine) => engine,
        Err(response) => return response.into_response(),
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
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Template formatting failed: {}", e),
                        "type": "template_error",
                        "code": "template_formatting_failed"
                    }
                })),
            )
                .into_response();
        }
    };

    // Get model path (using helper)
    let model_path = match resolve_model_path(&state, &request.model).await {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };

    let gen_request = GenerationRequest::builder(formatted_prompt)
        .apply_config(&(&state.config.sampling_defaults).into())
        .apply_config(&crate::config::SamplingParams::from_model_path(&model_path).await.unwrap_or_default())
        .apply_config(&(&request).into())
        .build();

    info!(
        "Completions: max_tokens={}, temp={}, top_p={}, top_k={:?}, repeat_penalty={}",
        gen_request.max_tokens, gen_request.temperature, gen_request.top_p, gen_request.top_k, gen_request.repeat_penalty
    );

    let result = {
        let engine = engine.lock().await;
        engine.generate_with_params(gen_request).await
    };

    // Metrics automatically decremented by MetricsGuard on drop

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

            let mut response = Json(response).into_response();
            add_no_cache_headers(&mut response);
            response
        }
        Err(e) => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Generation failed: {}", e),
                        "type": "generation_error",
                        "code": "internal_error"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Handle embedding requests
async fn embeddings(
    State(_state): State<ServerState>,
    Json(_request): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "Embeddings not yet implemented",
                "type": "not_implemented",
                "code": "feature_not_available"
            }
        })),
    )
        .into_response()
}

/// List available models
///
/// Returns all worktrees as model:branch references.
/// Models are only accessible via their worktree branches.
async fn list_models(State(state): State<ServerState>) -> impl IntoResponse {
    let mut models = vec![];

    // Get all worktrees from storage (formatted as model:branch)
    match state.model_storage.list_models().await {
        Ok(model_list) => {
            for (model_ref, metadata) in model_list {
                // Use ModelRef's Display impl for consistent model:branch format
                let model_id = model_ref.to_string();

                // Build owned_by field with worktree metadata
                let mut owned_by_parts = vec!["system".to_string()];

                // Add worktree metadata tags (driver, space saved, age)
                if !metadata.tags.is_empty() {
                    owned_by_parts.push(metadata.tags.join(", "));
                }

                // Check if this model is cached (by model:branch name)
                if state.model_cache.is_cached_by_name(&model_id).await {
                    owned_by_parts.push("cached".to_string());
                }

                let owned_by = owned_by_parts.join(" ");

                models.push(Model {
                    id: model_id,
                    object: "model".to_string(),
                    created: metadata.created_at,
                    owned_by,
                });
            }
        }
        Err(e) => {
            error!("Failed to list models from storage: {}", e);
        }
    }

    // Add no-cache headers
    let mut response = Json(ListModelsResponse {
        object: "list".to_string(),
        data: models,
    })
    .into_response();
    add_no_cache_headers(&mut response);
    response
}

/// Format chat messages into a prompt using template engine
async fn format_messages_with_template(
    messages: &[ChatMessage],
    engine: &Arc<Mutex<TorchEngine>>,
) -> Result<String, String> {
    // Log messages for debugging
    for (i, msg) in messages.iter().enumerate() {
        info!(
            "Message {}: role={}, content={:?}",
            i,
            msg.role,
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
            })
        );
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
    let result = engine_guard
        .apply_chat_template(&template_messages, true)
        .map_err(|e| format!("Failed to apply chat template: {}", e));
    info!("Template application complete: {:?}", result.is_ok());
    result
}
