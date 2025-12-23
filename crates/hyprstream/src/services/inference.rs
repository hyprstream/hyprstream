//! ZMQ-based inference service for text generation
//!
//! This service wraps TorchEngine and provides a ZMQ interface for inference operations.
//! It uses:
//! - REQ/REP for standard requests (generate, model_info, lora operations, etc.)
//! - PUB/SUB for streaming generation (via stream IDs)
//!
//! # Thread Model
//!
//! The service runs on a dedicated thread with its own TorchEngine because:
//! - tch-rs types contain raw pointers (not Send)
//! - GPU operations benefit from thread isolation
//!
//! # Streaming Architecture
//!
//! ```text
//! Client                          Service
//!   │                               │
//!   │──── REQ: GenerateStream ─────►│
//!   │◄─── REP: { stream_id } ───────│
//!   │                               │
//!   │  SUB: inproc://inference/stream/{id}
//!   │◄──── PUB: chunk "Hello" ──────│
//!   │◄──── PUB: chunk " world" ─────│
//!   │◄──── PUB: StreamComplete ─────│
//! ```

use crate::config::{FinishReason, GenerationRequest, GenerationResult, ModelInfo};
use crate::events::{EventBus, EventEnvelope, EventPayload, EventSource, GenerationMetrics};
use crate::inference_capnp;
use crate::lora::LoRAConfig;
use crate::runtime::kv_cache::CacheOwner;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};
use crate::zmq::global_context;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{error, info, trace, warn};

/// Default endpoint for the inference service
pub const INFERENCE_ENDPOINT: &str = "inproc://hyprstream/inference";

/// Endpoint for streaming chunks
pub const INFERENCE_STREAM_ENDPOINT: &str = "inproc://hyprstream/inference/stream";

/// ZMQ-based inference service
///
/// Wraps TorchEngine and provides a Cap'n Proto interface over ZMQ.
/// Runs on a dedicated thread for thread safety with tch-rs types.
/// Uses RefCell for interior mutability since it runs on a single thread.
pub struct InferenceService {
    engine: RefCell<TorchEngine>,
    stream_id_counter: AtomicU64,
    /// Optional event bus for publishing generation events
    event_bus: Option<Arc<EventBus>>,
    /// Model identifier for events
    model_id: String,
    /// Current session ID for events
    session_id: RefCell<Option<String>>,
}

impl InferenceService {
    /// Start the inference service at the default endpoint
    pub async fn start(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
        event_bus: Option<Arc<EventBus>>,
    ) -> Result<crate::services::ServiceHandle> {
        Self::start_at(model_path, config, INFERENCE_ENDPOINT, event_bus).await
    }

    /// Start the inference service at a specific endpoint
    pub async fn start_at(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
        endpoint: &str,
        event_bus: Option<Arc<EventBus>>,
    ) -> Result<crate::services::ServiceHandle> {
        let model_path = model_path.as_ref().to_path_buf();
        let endpoint_owned = endpoint.to_string();
        let model_id = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Use oneshot to get initialization result
        let (init_tx, init_rx) = tokio::sync::oneshot::channel();

        // Spawn service on dedicated thread
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create service runtime");

            rt.block_on(async move {
                match Self::initialize(model_path, config, event_bus, model_id).await {
                    Ok(service) => {
                        let _ = init_tx.send(Ok(()));
                        // Run the service loop (this blocks)
                        Self::run_service_loop(service, &endpoint_owned);
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                    }
                }
            });
        });

        // Wait for initialization
        init_rx
            .await
            .map_err(|_| anyhow!("Service init channel closed"))??;

        info!("Inference service started at {}", endpoint);

        // Return a dummy handle (the service manages its own lifecycle)
        Ok(crate::services::ServiceHandle::dummy())
    }

    /// Initialize the service
    async fn initialize(
        model_path: PathBuf,
        config: RuntimeConfig,
        event_bus: Option<Arc<EventBus>>,
        model_id: String,
    ) -> Result<Self> {
        let mut engine = TorchEngine::new(config.clone())?;
        RuntimeEngine::load_model(&mut engine, &model_path).await?;

        // Initialize KV cache registry
        let model_info = RuntimeEngine::model_info(&engine);
        let num_layers = model_info.num_hidden_layers.unwrap_or(32);
        let max_seq_len = config.max_context.unwrap_or(model_info.context_length);
        engine.initialize_kv_registry(num_layers, max_seq_len, config.kv_quant_type, None);

        info!(
            "KV cache registry initialized: {} layers, max_seq_len={}",
            num_layers, max_seq_len
        );

        Ok(Self {
            engine: RefCell::new(engine),
            stream_id_counter: AtomicU64::new(1),
            event_bus,
            model_id,
            session_id: RefCell::new(None),
        })
    }

    /// Run the service loop (blocking)
    fn run_service_loop(service: Self, endpoint: &str) {
        let ctx = global_context();

        // Create REP socket
        let socket = match ctx.socket(zmq::REP) {
            Ok(s) => s,
            Err(e) => {
                error!("failed to create REP socket: {}", e);
                return;
            }
        };

        // Set socket options
        if let Err(e) = socket.set_rcvtimeo(100) {
            warn!("failed to set receive timeout: {}", e);
        }

        // Bind to endpoint
        if let Err(e) = socket.bind(endpoint) {
            error!("failed to bind to {}: {}", endpoint, e);
            return;
        }

        info!("inference service bound to {}", endpoint);

        // Main service loop
        loop {
            match socket.recv_bytes(0) {
                Ok(request) => {
                    trace!("inference received request ({} bytes)", request.len());

                    let response = match service.handle_request(&request) {
                        Ok(resp) => resp,
                        Err(e) => {
                            error!("inference request handling error: {}", e);
                            service.build_error_response(0, &e.to_string())
                        }
                    };

                    if let Err(e) = socket.send(&response, 0) {
                        error!("failed to send response: {}", e);
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, continue
                    continue;
                }
                Err(e) => {
                    warn!("inference recv error: {}", e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }

    /// Generate next stream ID
    fn next_stream_id(&self) -> String {
        let id = self.stream_id_counter.fetch_add(1, Ordering::Relaxed);
        format!("stream-{}", id)
    }

    /// Publish a generation complete event
    fn publish_generation_complete(&self, result: &GenerationResult) {
        if let Some(bus) = &self.event_bus {
            let metrics = GenerationMetrics {
                perplexity: 0.0,        // Not computed in basic generation
                avg_entropy: 0.0,       // Not computed in basic generation
                entropy_variance: 0.0,  // Not computed in basic generation
                repetition_ratio: 0.0,  // Could be computed but isn't yet
                token_count: result.tokens_generated as u32,
                tokens_per_second: result.tokens_per_second,
                generation_time_ms: result.generation_time_ms,
            };

            let event = EventEnvelope::new(
                EventSource::Inference,
                "inference.generation_complete",
                EventPayload::GenerationComplete {
                    model_id: self.model_id.clone(),
                    session_id: self.session_id.borrow().clone(),
                    metrics,
                },
            );

            // Publish in background (fire and forget)
            let bus_clone = bus.clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            if let Ok(rt) = rt {
                let _ = rt.block_on(async { bus_clone.publish(&event).await });
            }
        }
    }

    /// Publish a generation failed event
    fn publish_generation_failed(&self, error: &str) {
        if let Some(bus) = &self.event_bus {
            let event = EventEnvelope::new(
                EventSource::Inference,
                "inference.generation_failed",
                EventPayload::GenerationFailed {
                    model_id: self.model_id.clone(),
                    session_id: self.session_id.borrow().clone(),
                    error: error.to_string(),
                    error_code: None,
                },
            );

            let bus_clone = bus.clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            if let Ok(rt) = rt {
                let _ = rt.block_on(async { bus_clone.publish(&event).await });
            }
        }
    }

    /// Handle non-streaming generation
    fn handle_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        // Create a runtime for the async operation
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let engine = self.engine.borrow();
        let result = rt.block_on(async { RuntimeEngine::generate_with_params(&*engine, request).await });

        // Publish event based on result
        match &result {
            Ok(gen_result) => self.publish_generation_complete(gen_result),
            Err(e) => self.publish_generation_failed(&e.to_string()),
        }

        result
    }

    /// Handle streaming generation
    fn handle_generate_stream(&self, request: GenerationRequest) -> Result<String> {
        use futures::StreamExt;

        let stream_id = self.next_stream_id();
        let stream_endpoint = format!("{}/{}", INFERENCE_STREAM_ENDPOINT, stream_id);

        // Create PUB socket for this stream
        let ctx = global_context();
        let pub_socket = ctx.socket(zmq::PUB)?;
        pub_socket.bind(&stream_endpoint)?;

        // Small delay to allow subscriber to connect
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Run the stream
        let engine = self.engine.borrow();
        let stream_result = engine.generate(request);

        match stream_result {
            Ok(mut stream) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;

                let mut seq_num: u32 = 0;

                rt.block_on(async {
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(text) => {
                                // Build and send chunk message
                                let chunk_bytes =
                                    self.build_stream_chunk(&stream_id, seq_num, &text);
                                if let Err(e) = pub_socket.send(&chunk_bytes, 0) {
                                    warn!("failed to send stream chunk: {}", e);
                                    break;
                                }
                                seq_num += 1;
                            }
                            Err(e) => {
                                let error_bytes =
                                    self.build_stream_error(&stream_id, seq_num, &e.to_string());
                                let _ = pub_socket.send(&error_bytes, 0);
                                break;
                            }
                        }
                    }

                    // Send completion
                    let stats = stream.stats();
                    let complete_bytes = self.build_stream_complete(&stream_id, seq_num, &stats);
                    let _ = pub_socket.send(&complete_bytes, 0);
                });
            }
            Err(e) => {
                let error_bytes = self.build_stream_error(&stream_id, 0, &e.to_string());
                let _ = pub_socket.send(&error_bytes, 0);
            }
        }

        Ok(stream_id)
    }

    /// Handle model info request
    fn handle_model_info(&self) -> ModelInfo {
        RuntimeEngine::model_info(&*self.engine.borrow())
    }

    /// Handle is ready request
    fn handle_is_ready(&self) -> bool {
        self.engine.borrow().is_loaded()
    }

    /// Handle apply chat template
    fn handle_apply_chat_template(
        &self,
        messages: Vec<crate::runtime::template_engine::ChatMessage>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.engine
            .borrow()
            .apply_chat_template(&messages, add_generation_prompt)
    }

    /// Handle create LoRA
    fn handle_create_lora(&self, config: LoRAConfig) -> Result<()> {
        self.engine.borrow_mut().create_lora(config)
    }

    /// Handle load LoRA
    fn handle_load_lora(&self, path: &Path) -> Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let mut engine = self.engine.borrow_mut();
        rt.block_on(async { engine.load_lora_from_file(path).await })
    }

    /// Handle save LoRA
    fn handle_save_lora(&self, path: &str) -> Result<()> {
        self.engine.borrow().save_lora(path)
    }

    /// Handle unload LoRA
    fn handle_unload_lora(&self) -> Result<()> {
        self.engine.borrow_mut().unload_lora()
    }

    /// Handle has LoRA
    fn handle_has_lora(&self) -> bool {
        self.engine.borrow().has_lora_model()
    }

    /// Handle set session
    fn handle_set_session(&self, session_id: String) -> Result<()> {
        // Track session ID for events
        *self.session_id.borrow_mut() = Some(session_id.clone());
        self.engine
            .borrow_mut()
            .set_session(CacheOwner::Session(session_id))
    }

    /// Handle clear session
    fn handle_clear_session(&self) {
        *self.session_id.borrow_mut() = None;
        self.engine.borrow_mut().clear_kv_cache();
    }

    /// Handle release session
    fn handle_release_session(&self, session_id: &str) -> Result<()> {
        self.engine
            .borrow_mut()
            .release_session(&CacheOwner::Session(session_id.to_string()))
    }

    /// Convert FinishReason enum to capnp
    fn finish_reason_to_capnp(&self, reason: &FinishReason) -> inference_capnp::FinishReason {
        match reason {
            FinishReason::MaxTokens => inference_capnp::FinishReason::MaxTokens,
            FinishReason::StopToken(_) => inference_capnp::FinishReason::StopToken,
            FinishReason::EndOfSequence => inference_capnp::FinishReason::EndOfSequence,
            FinishReason::Error(_) => inference_capnp::FinishReason::Error,
            FinishReason::Stop => inference_capnp::FinishReason::Stop,
        }
    }

    /// Build an error response
    fn build_error_response(&self, request_id: u64, error: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(error);
        error_info.set_code("ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a success response
    fn build_success_response(&self, request_id: u64) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_success(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a generation result response
    fn build_generation_result_response(
        &self,
        request_id: u64,
        result: &GenerationResult,
    ) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut gen_result = response.init_generation_result();
        gen_result.set_text(&result.text);
        gen_result.set_tokens_generated(result.tokens_generated as u32);
        gen_result.set_finish_reason(self.finish_reason_to_capnp(&result.finish_reason));
        gen_result.set_generation_time_ms(result.generation_time_ms);
        gen_result.set_tokens_per_second(result.tokens_per_second);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a stream started response
    fn build_stream_started_response(&self, request_id: u64, stream_id: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut stream_info = response.init_stream_started();
        stream_info.set_stream_id(stream_id);
        stream_info.set_endpoint(&format!("{}/{}", INFERENCE_STREAM_ENDPOINT, stream_id));

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a model info response
    fn build_model_info_response(&self, request_id: u64, info: &ModelInfo) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut model_info = response.init_model_info();
        model_info.set_model_id(&info.name);
        model_info.set_architecture(&info.architecture);
        model_info.set_vocab_size(info.vocab_size as u32);
        model_info.set_hidden_size(info.hidden_size as u32);
        model_info.set_num_layers(info.num_hidden_layers.unwrap_or(0) as u32);
        model_info.set_num_heads(info.num_attention_heads.unwrap_or(0) as u32);
        model_info.set_max_sequence_length(info.context_length as u32);
        model_info.set_quantization(info.quantization.as_deref().unwrap_or("none"));
        model_info.set_has_vision(false);
        model_info.set_lora_loaded(self.engine.borrow().has_lora_model());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a ready response
    fn build_ready_response(&self, request_id: u64, ready: bool) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_ready(ready);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a template result response
    fn build_template_result_response(&self, request_id: u64, result: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_template_result(result);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a has LoRA result response
    fn build_has_lora_response(&self, request_id: u64, has_lora: bool) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);
        response.set_has_lora_result(has_lora);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a health status response
    fn build_health_response(&self, request_id: u64) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<inference_capnp::inference_response::Builder>();
        response.set_request_id(request_id);

        let mut health = response.init_health();
        health.set_status("healthy");
        health.set_model_loaded(self.engine.borrow().is_loaded());
        health.set_kv_cache_usage_percent(0.0);
        health.set_gpu_memory_used_mb(0);
        health.set_gpu_memory_total_mb(0);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a stream chunk message
    fn build_stream_chunk(&self, stream_id: &str, seq_num: u32, text: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut chunk = message.init_root::<inference_capnp::stream_chunk::Builder>();
        chunk.set_stream_id(stream_id);
        chunk.set_sequence_num(seq_num);
        chunk.set_text(text);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a stream error message
    fn build_stream_error(&self, stream_id: &str, seq_num: u32, error: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut chunk = message.init_root::<inference_capnp::stream_chunk::Builder>();
        chunk.set_stream_id(stream_id);
        chunk.set_sequence_num(seq_num);

        let mut error_info = chunk.init_error();
        error_info.set_message(error);
        error_info.set_code("GENERATION_ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a stream complete message
    fn build_stream_complete(
        &self,
        stream_id: &str,
        seq_num: u32,
        stats: &crate::runtime::GenerationStats,
    ) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut chunk = message.init_root::<inference_capnp::stream_chunk::Builder>();
        chunk.set_stream_id(stream_id);
        chunk.set_sequence_num(seq_num);

        let mut complete = chunk.init_complete();
        complete.set_tokens_generated(stats.tokens_generated as u32);
        let finish_reason = stats
            .finish_reason
            .as_ref()
            .map(|r| self.finish_reason_to_capnp(r))
            .unwrap_or(inference_capnp::FinishReason::Stop);
        complete.set_finish_reason(finish_reason);
        complete.set_generation_time_ms(stats.generation_time_ms);
        complete.set_tokens_per_second(stats.tokens_per_second);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Parse a generation request from capnp
    fn parse_generation_request(
        &self,
        reader: inference_capnp::generation_request::Reader,
    ) -> Result<GenerationRequest> {
        let prompt = reader.get_prompt()?.to_str()?.to_string();
        let max_tokens = reader.get_max_tokens() as usize;
        let temperature = reader.get_temperature();
        let top_p = reader.get_top_p();
        let top_k = if reader.get_top_k() == 0 {
            None
        } else {
            Some(reader.get_top_k() as usize)
        };
        let repeat_penalty = reader.get_repeat_penalty();
        let repeat_last_n = reader.get_repeat_last_n() as usize;

        let stop_tokens: Vec<String> = reader
            .get_stop_tokens()?
            .iter()
            .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(|s| s.to_string())))
            .collect();

        let seed = if reader.get_seed() == 0 {
            None
        } else {
            Some(reader.get_seed())
        };

        // Parse images if present - convert from bytes to paths
        let images: Vec<String> = Vec::new(); // Images are bytes in capnp, but strings in GenerationRequest

        let timeout = if reader.get_timeout_ms() == 0 {
            None
        } else {
            Some(reader.get_timeout_ms())
        };

        Ok(GenerationRequest {
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            repeat_last_n,
            stop_tokens,
            seed,
            images,
            timeout,
        })
    }
}

impl InferenceService {
    /// Handle a capnp request and return a capnp response
    fn handle_request(&self, request: &[u8]) -> Result<Vec<u8>> {
        // Deserialize request
        let reader = serialize::read_message(request, ReaderOptions::new())?;
        let req = reader.get_root::<inference_capnp::inference_request::Reader>()?;

        let request_id = req.get_id();

        use inference_capnp::inference_request::Which;

        match req.which()? {
            Which::Generate(gen_req) => {
                let gen_req = gen_req?;
                let request = self.parse_generation_request(gen_req)?;

                match self.handle_generate(request) {
                    Ok(result) => Ok(self.build_generation_result_response(request_id, &result)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::GenerateStream(gen_req) => {
                let gen_req = gen_req?;
                let request = self.parse_generation_request(gen_req)?;

                match self.handle_generate_stream(request) {
                    Ok(stream_id) => {
                        Ok(self.build_stream_started_response(request_id, &stream_id))
                    }
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ModelInfo(()) => {
                let info = self.handle_model_info();
                Ok(self.build_model_info_response(request_id, &info))
            }

            Which::IsReady(()) => {
                let ready = self.handle_is_ready();
                Ok(self.build_ready_response(request_id, ready))
            }

            Which::ApplyChatTemplate(template_req) => {
                let template_req = template_req?;
                let messages: Vec<crate::runtime::template_engine::ChatMessage> = template_req
                    .get_messages()?
                    .iter()
                    .filter_map(|m| {
                        Some(crate::runtime::template_engine::ChatMessage {
                            role: m.get_role().ok()?.to_str().ok()?.to_string(),
                            content: m.get_content().ok()?.to_str().ok()?.to_string(),
                        })
                    })
                    .collect();

                let add_generation_prompt = template_req.get_add_generation_prompt();

                match self.handle_apply_chat_template(messages, add_generation_prompt) {
                    Ok(result) => Ok(self.build_template_result_response(request_id, &result)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::CreateLora(lora_config) => {
                let lora_config = lora_config?;
                let config = LoRAConfig {
                    rank: lora_config.get_rank() as usize,
                    alpha: lora_config.get_alpha(),
                    dropout: lora_config.get_dropout(),
                    target_modules: lora_config
                        .get_target_modules()?
                        .iter()
                        .filter_map(|s| s.ok().and_then(|t| t.to_str().ok().map(|s| s.to_string())))
                        .collect(),
                    learning_rate: 0.0001, // Default learning rate
                };

                match self.handle_create_lora(config) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::LoadLora(path) => {
                let path = path?.to_str()?;
                match self.handle_load_lora(Path::new(path)) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::SaveLora(path) => {
                let path = path?.to_str()?;
                match self.handle_save_lora(path) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::UnloadLora(()) => match self.handle_unload_lora() {
                Ok(()) => Ok(self.build_success_response(request_id)),
                Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
            },

            Which::HasLora(()) => {
                let has_lora = self.handle_has_lora();
                Ok(self.build_has_lora_response(request_id, has_lora))
            }

            Which::SetSession(session_id) => {
                let session_id = session_id?.to_str()?.to_string();
                match self.handle_set_session(session_id) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ClearSession(()) => {
                self.handle_clear_session();
                Ok(self.build_success_response(request_id))
            }

            Which::ReleaseSession(session_id) => {
                let session_id = session_id?.to_str()?;
                match self.handle_release_session(session_id) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::HealthCheck(()) => Ok(self.build_health_response(request_id)),

            Which::Shutdown(()) => {
                info!("Inference service shutdown requested");
                Ok(self.build_success_response(request_id))
            }
        }
    }
}

/// Client for the inference service
pub struct InferenceZmqClient {
    client: crate::services::AsyncServiceClient,
    request_id: AtomicU64,
}

impl InferenceZmqClient {
    /// Create a new inference client
    pub fn new() -> Self {
        Self::with_endpoint(INFERENCE_ENDPOINT)
    }

    /// Create an inference client connected to a specific endpoint
    pub fn with_endpoint(endpoint: &str) -> Self {
        Self {
            client: crate::services::AsyncServiceClient::new(endpoint),
            request_id: AtomicU64::new(1),
        }
    }

    /// Get the next request ID
    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Convert capnp FinishReason to Rust enum
    fn parse_finish_reason(
        &self,
        reason: inference_capnp::FinishReason,
    ) -> FinishReason {
        match reason {
            inference_capnp::FinishReason::MaxTokens => FinishReason::MaxTokens,
            inference_capnp::FinishReason::StopToken => FinishReason::StopToken(String::new()),
            inference_capnp::FinishReason::EndOfSequence => FinishReason::EndOfSequence,
            inference_capnp::FinishReason::Error => FinishReason::Error(String::new()),
            inference_capnp::FinishReason::Stop => FinishReason::Stop,
        }
    }

    /// Generate text (non-streaming)
    pub async fn generate(&self, request: &GenerationRequest) -> Result<GenerationResult> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);

        let mut gen_req = req.init_generate();
        gen_req.set_prompt(&request.prompt);
        gen_req.set_max_tokens(request.max_tokens as u32);
        gen_req.set_temperature(request.temperature);
        gen_req.set_top_p(request.top_p);
        gen_req.set_top_k(request.top_k.unwrap_or(0) as u32);
        gen_req.set_repeat_penalty(request.repeat_penalty);
        gen_req.set_repeat_last_n(request.repeat_last_n as u32);
        gen_req.set_seed(request.seed.unwrap_or(0));
        gen_req.set_timeout_ms(request.timeout.unwrap_or(0));

        // init_stop_tokens must be called last as it consumes the builder
        if !request.stop_tokens.is_empty() {
            let mut stop_list = gen_req.init_stop_tokens(request.stop_tokens.len() as u32);
            for (i, token) in request.stop_tokens.iter().enumerate() {
                stop_list.set(i as u32, token);
            }
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_generation_result(&response)
    }

    /// Check if model is ready
    pub async fn is_ready(&self) -> Result<bool> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);
        req.set_is_ready(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_ready_response(&response)
    }

    /// Get model info
    pub async fn model_info(&self) -> Result<ModelInfo> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);
        req.set_model_info(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_model_info_response(&response)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<()> {
        let id = self.next_id();

        let mut message = Builder::new_default();
        let mut req = message.init_root::<inference_capnp::inference_request::Builder>();
        req.set_id(id);
        req.set_health_check(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;

        let response = self.client.call(bytes).await?;
        self.parse_health_response(&response)
    }

    /// Parse generation result
    fn parse_generation_result(&self, response: &[u8]) -> Result<GenerationResult> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::GenerationResult(result) => {
                let result = result?;
                Ok(GenerationResult {
                    text: result.get_text()?.to_str()?.to_string(),
                    tokens_generated: result.get_tokens_generated() as usize,
                    finish_reason: self.parse_finish_reason(result.get_finish_reason()?),
                    generation_time_ms: result.get_generation_time_ms(),
                    tokens_per_second: result.get_tokens_per_second(),
                })
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse ready response
    fn parse_ready_response(&self, response: &[u8]) -> Result<bool> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::Ready(ready) => Ok(ready),
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse model info response
    fn parse_model_info_response(&self, response: &[u8]) -> Result<ModelInfo> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::ModelInfo(info) => {
                let info = info?;
                Ok(ModelInfo {
                    name: info.get_model_id()?.to_str()?.to_string(),
                    architecture: info.get_architecture()?.to_str()?.to_string(),
                    vocab_size: info.get_vocab_size() as usize,
                    hidden_size: info.get_hidden_size() as usize,
                    num_hidden_layers: Some(info.get_num_layers() as usize),
                    num_attention_heads: Some(info.get_num_heads() as usize),
                    context_length: info.get_max_sequence_length() as usize,
                    quantization: Some(info.get_quantization()?.to_str()?.to_string()),
                    parameters: 0,
                    intermediate_size: None,
                })
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse health response
    fn parse_health_response(&self, response: &[u8]) -> Result<()> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<inference_capnp::inference_response::Reader>()?;

        use inference_capnp::inference_response::Which;
        match resp.which()? {
            Which::Health(_) => Ok(()),
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }
}

impl Default for InferenceZmqClient {
    fn default() -> Self {
        Self::new()
    }
}
