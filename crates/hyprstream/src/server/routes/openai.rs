//! OpenAI-compatible API endpoints

use axum::{
    extract::{Json, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Sse},
    routing::{get, post},
    Extension, Router,
};
use futures::stream::StreamExt;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, trace};

use crate::{
    api::openai_compat::{
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CompletionChoice,
        CompletionRequest, CompletionResponse, EmbeddingRequest, ListModelsResponse, Model, Usage,
    },
    archetypes::capabilities::Infer,
    auth::Operation,
    config::GenerationRequest,
    runtime::{CacheOwner, FinishReason},
    server::{state::ServerState, AuthenticatedUser},
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
        trace!("Using session-based caching for session: {}", session_id);
        CacheOwner::Session(session_id.to_string())
    } else {
        // Stateless request - generate unique ID
        let request_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        trace!("Using stateless caching for request: {}", request_id);
        CacheOwner::Stateless(request_id)
    }
}

/// Extract user identity from authenticated user.
///
/// JWT `sub` claim should already contain prefixed subject (e.g., "token:alice").
/// Returns "anonymous" if no authentication provided.
fn extract_user_from_auth(auth_user: Option<&AuthenticatedUser>) -> String {
    if let Some(user) = auth_user {
        trace!("Using authenticated user: {}", user.user);
        return user.user.clone();
    }

    trace!("No authentication provided, using anonymous identity");
    "anonymous".to_string()
}

/// Helper: Resolve model name to filesystem path (also validates inference capability)
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
        Ok(path) => {
            // Check if model has INFERENCE capability
            let archetype_registry = crate::archetypes::global_registry();
            let detected = archetype_registry.detect(&path);

            let domains = detected.to_detected_domains();
            if !domains.has::<Infer>() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse::new(
                        format!(
                            "Model '{}' does not support inference. Detected domains: {:?}",
                            model_name, domains.domains
                        ),
                        "capability_error",
                        "inference_not_supported",
                    )),
                )
                    .into_response());
            }

            Ok(path)
        }
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
    use axum::http::HeaderValue;
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-store, must-revalidate"),
    );
    headers.insert(
        header::PRAGMA,
        HeaderValue::from_static("no-cache"),
    );
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
    auth_user: Option<Extension<AuthenticatedUser>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Extract user identity from JWT (via middleware)
    let user = extract_user_from_auth(auth_user.as_ref().map(|Extension(u)| u));

    // Check permission for inference on this model via ZMQ
    let resource = format!("model:{}", request.model);
    match state
        .policy_client
        .check(&user, &resource, Operation::Infer)
        .await
    {
        Ok(allowed) if !allowed => {
            return (
                StatusCode::FORBIDDEN,
                Json(ErrorResponse::new(
                    format!("Permission denied: user '{}' cannot infer on '{}'", user, request.model),
                    "permission_denied",
                    "insufficient_permissions",
                )),
            )
                .into_response();
        }
        Err(e) => {
            error!("Policy check failed: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    format!("Policy check failed: {}", e),
                    "policy_error",
                    "internal_error",
                )),
            )
                .into_response();
        }
        _ => {} // allowed
    }

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

    if is_streaming {
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

    // Resolve model path (also validates inference capability)
    let model_path = match resolve_model_path(&state, &request.model).await {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };

    // Apply chat template via ZMQ ModelService
    info!("Applying chat template via ModelService...");
    let templated_prompt = match state
        .model_client
        .apply_chat_template(&request.model, &request.messages, true)
        .await
    {
        Ok(prompt) => prompt,
        Err(e) => {
            error!("Failed to apply chat template: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    format!("Failed to apply chat template: {}", e),
                    "template_error",
                    "internal_error",
                )),
            )
                .into_response();
        }
    };
    info!(
        "Template applied, prompt length: {}, preview: {}",
        templated_prompt.len(),
        &templated_prompt.as_str().chars().take(200).collect::<String>()
    );

    // Build generation request with templated prompt
    let gen_request = GenerationRequest::builder(templated_prompt.as_str())
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

    // Call inference via ZMQ ModelService
    let result = state.model_client.infer(&request.model, &gen_request).await;

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

/// Handle streaming chat completions via ZMQ PUB/SUB
async fn stream_chat(state: ServerState, _headers: HeaderMap, request: ChatCompletionRequest) -> impl IntoResponse {
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

        // Apply chat template via ZMQ ModelService
        info!("Applying chat template for streaming...");
        let templated_prompt = match state.model_client
            .apply_chat_template(&model_name, &messages, true)
            .await
        {
            Ok(prompt) => {
                info!("Streaming template applied, prompt length: {}", prompt.len());
                prompt
            }
            Err(e) => {
                error!("Template formatting failed in streaming: {}", e);
                let _ = tx.send(Err(anyhow::anyhow!("Template formatting failed: {}", e))).await;
                return;
            }
        };

        // Build generation request
        let gen_request = GenerationRequest::builder(templated_prompt.as_str())
            .apply_config(&(&defaults).into())
            .apply_config(&crate::config::SamplingParams::from_model_path(&model_path).await.unwrap_or_default())
            .apply_config(&(&request).into())
            .stop_tokens(stop_sequences)
            .build();

        info!(
            "Streaming: max_tokens={}, temp={}, top_p={}, top_k={:?}, repeat_penalty={}",
            gen_request.max_tokens, gen_request.temperature, gen_request.top_p, gen_request.top_k, gen_request.repeat_penalty
        );

        // Start ZMQ stream - returns (zmq_stream_id, endpoint)
        let (zmq_stream_id, endpoint) = match state.model_client.infer_stream(&model_name, &gen_request).await {
            Ok((id, ep)) => {
                info!("ZMQ stream started: id={}, endpoint={}", id, ep);
                (id, ep)
            }
            Err(e) => {
                error!("Failed to start ZMQ stream: {}", e);
                let _ = tx.send(Err(anyhow::anyhow!("Generation failed: {}", e))).await;
                return;
            }
        };

        // Create ZMQ SUB socket and subscribe to stream topic
        let ctx = crate::zmq::global_context();
        let sub_socket = match ctx.socket(zmq::SUB) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to create SUB socket: {}", e);
                let _ = tx.send(Err(anyhow::anyhow!("Failed to create stream socket: {}", e))).await;
                return;
            }
        };

        if let Err(e) = sub_socket.connect(&endpoint) {
            error!("Failed to connect to stream endpoint: {}", e);
            let _ = tx.send(Err(anyhow::anyhow!("Failed to connect to stream: {}", e))).await;
            return;
        }

        if let Err(e) = sub_socket.set_subscribe(zmq_stream_id.as_bytes()) {
            error!("Failed to subscribe to stream: {}", e);
            let _ = tx.send(Err(anyhow::anyhow!("Failed to subscribe to stream: {}", e))).await;
            return;
        }

        // Set receive timeout to allow periodic cancellation checks (100ms)
        if let Err(e) = sub_socket.set_rcvtimeo(100) {
            error!("Failed to set socket timeout: {}", e);
            // Continue without timeout - less responsive to cancellation but still works
        }

        // OpenAI-style stream ID for SSE responses
        let sse_stream_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        // Send initial role message
        let initial_msg = serde_json::json!({
            "id": sse_stream_id,
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

        // ZMQ receive loop - forward tokens to SSE
        info!("Starting ZMQ streaming receive loop...");
        let topic_len = zmq_stream_id.len();

        loop {
            // Check if client disconnected (channel closed)
            if tx.is_closed() {
                info!("Client disconnected, stopping stream");
                break;
            }

            match sub_socket.recv_bytes(0) {
                Ok(msg) => {
                    // Strip topic prefix to get Cap'n Proto data
                    if msg.len() <= topic_len {
                        continue; // Empty message after topic
                    }
                    let data = &msg[topic_len..];

                    // Parse Cap'n Proto StreamChunk
                    use capnp::message::ReaderOptions;
                    use capnp::serialize;
                    use crate::inference_capnp;

                    let reader = match serialize::read_message(
                        &mut std::io::Cursor::new(data),
                        ReaderOptions::default(),
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Failed to parse stream chunk: {}", e);
                            continue;
                        }
                    };

                    let chunk = match reader.get_root::<inference_capnp::stream_chunk::Reader>() {
                        Ok(c) => c,
                        Err(e) => {
                            error!("Failed to get stream chunk root: {}", e);
                            continue;
                        }
                    };

                    // Extract chunk data into owned values BEFORE any await
                    // (capnp readers contain raw pointers and cannot be held across await points)
                    use inference_capnp::stream_chunk::Which;
                    enum ChunkAction {
                        SendText(String),
                        Complete { finish_reason: String, gen_time_ms: u64, toks_per_sec: f32, toks_gen: u32 },
                        Error(String),
                        Continue,
                    }

                    let action = match chunk.which() {
                        Ok(Which::Text(text_result)) => {
                            match text_result {
                                Ok(t) => ChunkAction::SendText(t.to_string().unwrap_or_default()),
                                Err(_) => ChunkAction::Continue,
                            }
                        }
                        Ok(Which::Complete(stats_result)) => {
                            if let Ok(stats) = stats_result {
                                let toks = stats.get_tokens_generated();
                                let fr = match stats.get_finish_reason() {
                                    Ok(inference_capnp::FinishReason::MaxTokens) => "length",
                                    _ => "stop",
                                };
                                ChunkAction::Complete {
                                    finish_reason: fr.to_string(),
                                    gen_time_ms: stats.get_generation_time_ms(),
                                    toks_per_sec: stats.get_tokens_per_second(),
                                    toks_gen: toks,
                                }
                            } else {
                                ChunkAction::Complete {
                                    finish_reason: "stop".to_string(),
                                    gen_time_ms: 0,
                                    toks_per_sec: 0.0,
                                    toks_gen: 0,
                                }
                            }
                        }
                        Ok(Which::Error(error_result)) => {
                            let msg = if let Ok(err) = error_result {
                                err.get_message()
                                    .ok()
                                    .and_then(|r| r.to_str().ok())
                                    .unwrap_or("Unknown error")
                                    .to_string()
                            } else {
                                "Stream error".to_string()
                            };
                            ChunkAction::Error(msg)
                        }
                        Err(e) => {
                            error!("Failed to parse stream chunk variant: {:?}", e);
                            ChunkAction::Continue
                        }
                    };
                    // chunk is now dropped, safe to use awaits

                    match action {
                        ChunkAction::SendText(text) => {
                            let sse_chunk = serde_json::json!({
                                "id": sse_stream_id,
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

                            if tx.send(Ok(sse_chunk)).await.is_err() {
                                info!("Client disconnected during streaming");
                                break;
                            }
                        }
                        ChunkAction::Complete { finish_reason, gen_time_ms, toks_per_sec, toks_gen } => {
                            info!(
                                "Streaming complete: {} tokens in {}ms ({:.2} tok/s)",
                                toks_gen,
                                gen_time_ms,
                                toks_per_sec
                            );

                            state.metrics.total_tokens.fetch_add(
                                toks_gen as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );

                            let completion_msg = serde_json::json!({
                                "id": sse_stream_id,
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

                            let _ = tx.send(Ok(serde_json::json!({"done": true}))).await;
                            break;
                        }
                        ChunkAction::Error(error_msg) => {
                            error!("Stream error: {}", error_msg);
                            let _ = tx.send(Err(anyhow::anyhow!("Generation error: {}", error_msg))).await;
                            break;
                        }
                        ChunkAction::Continue => {
                            continue;
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout - yield and check for cancellation
                    tokio::task::yield_now().await;
                    continue;
                }
                Err(e) => {
                    error!("ZMQ recv error: {}", e);
                    let _ = tx.send(Err(anyhow::anyhow!("Stream receive error: {}", e))).await;
                    break;
                }
            }
        }

        // Explicit socket cleanup
        let _ = sub_socket.disconnect(&endpoint);
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
                        .data(serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_string())))
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
        .insert(
            header::EXPIRES,
            axum::http::HeaderValue::from_static("0"),
        );

    response
}

/// Handle text completion requests
async fn completions(
    State(state): State<ServerState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    _headers: HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Extract user identity from JWT (via middleware)
    let user = extract_user_from_auth(auth_user.as_ref().map(|Extension(u)| u));

    // Check permission for inference on this model
    let resource = format!("model:{}", request.model);
    match state
        .policy_client
        .check(&user, &resource, Operation::Infer)
        .await
    {
        Ok(allowed) if !allowed => {
            return (
                StatusCode::FORBIDDEN,
                Json(ErrorResponse::new(
                    format!("Permission denied: user '{}' cannot infer on '{}'", user, request.model),
                    "permission_denied",
                    "insufficient_permissions",
                )),
            )
                .into_response();
        }
        Err(e) => {
            error!("Policy check failed: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    format!("Policy check failed: {}", e),
                    "policy_error",
                    "internal_error",
                )),
            )
                .into_response();
        }
        _ => {} // allowed
    }

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

    // Resolve model path (also validates inference capability)
    let model_path = match resolve_model_path(&state, &request.model).await {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };

    // Convert raw prompt to chat format and apply template via ZMQ ModelService
    // The completions endpoint expects raw text, but modern models need templated input
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(request.prompt.clone()),
        function_call: None,
    }];

    let templated_prompt = match state
        .model_client
        .apply_chat_template(&request.model, &messages, true)
        .await
    {
        Ok(prompt) => prompt,
        Err(e) => {
            error!("Template formatting failed for completions endpoint: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    format!("Template formatting failed: {}", e),
                    "template_error",
                    "template_formatting_failed",
                )),
            )
                .into_response();
        }
    };

    let gen_request = GenerationRequest::builder(templated_prompt.as_str())
        .apply_config(&(&state.config.sampling_defaults).into())
        .apply_config(&crate::config::SamplingParams::from_model_path(&model_path).await.unwrap_or_default())
        .apply_config(&(&request).into())
        .build();

    info!(
        "Completions: max_tokens={}, temp={}, top_p={}, top_k={:?}, repeat_penalty={}",
        gen_request.max_tokens, gen_request.temperature, gen_request.top_p, gen_request.top_k, gen_request.repeat_penalty
    );

    // Call inference via ZMQ ModelService
    let result = state.model_client.infer(&request.model, &gen_request).await;

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

                // Note: Model caching is now handled by ModelService internally
                // The "cached" status is no longer exposed at the HTTP layer

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

// REMOVED: format_messages_with_template - replaced by model_client.apply_chat_template()
