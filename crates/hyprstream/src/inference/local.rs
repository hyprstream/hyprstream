//! In-process inference service using channels.
//!
//! This module provides a local service implementation that runs on a dedicated
//! thread and communicates with clients via unbounded mpsc channels.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │         LocalInferenceClient (cloneable)         │
//! │  - sender: mpsc::UnboundedSender<Request>       │
//! └───────────────────────┬─────────────────────────┘
//!                         │ sends request
//!                         ▼
//! ┌─────────────────────────────────────────────────┐
//! │            LocalInferenceService                 │
//! │  - engine: TorchEngine                          │
//! │  - runs on dedicated thread                     │
//! └─────────────────────────────────────────────────┘
//! ```

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, instrument, warn};

use crate::config::{GenerationRequest, GenerationResult, ModelInfo};
use crate::lora::LoRAConfig;
use crate::runtime::kv_cache::CacheOwner;
use crate::runtime::{RuntimeConfig, RuntimeEngine, TorchEngine};

use super::client::{InferenceClient, InferenceError};
use super::request::InferenceRequest;
use super::response::{StreamHandle, StreamStats};

/// In-process inference service that runs on a dedicated thread.
///
/// The service owns a TorchEngine and processes requests from clients
/// via an unbounded mpsc channel. It runs on its own thread with a
/// single-threaded tokio runtime to isolate GPU operations.
pub struct LocalInferenceService {
    engine: TorchEngine,
    requests: mpsc::UnboundedReceiver<InferenceRequest>,
    shutdown_requested: bool,
}

impl LocalInferenceService {
    /// Start the service and return a client handle.
    ///
    /// The service runs on a dedicated thread with its own tokio runtime.
    /// This is necessary because tch-rs types contain raw pointers that
    /// are not Send, and GPU operations should be isolated.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model directory
    /// * `config` - Runtime configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// let client = LocalInferenceService::start("/path/to/model", config).await?;
    /// let result = client.generate(request).await?;
    /// ```
    #[instrument(skip_all, fields(model_path = %model_path.as_ref().display()))]
    pub async fn start(
        model_path: impl AsRef<Path>,
        config: RuntimeConfig,
    ) -> Result<LocalInferenceClient, InferenceError> {
        let model_path = model_path.as_ref().to_path_buf();
        info!("Starting inference service");

        // Create channel for communication
        let (tx, rx) = mpsc::unbounded_channel();

        // Use oneshot to get result from spawned thread
        let (init_tx, init_rx) = oneshot::channel();

        // Spawn service on dedicated thread with its own runtime
        // This is necessary because:
        // 1. tch-rs types contain raw pointers (not Send)
        // 2. GPU operations benefit from thread isolation
        // 3. Matches git2db's LocalService pattern
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create service runtime");

            rt.block_on(async move {
                // Load model
                match Self::initialize(model_path, config, rx).await {
                    Ok(service) => {
                        let _ = init_tx.send(Ok(()));
                        service.run().await;
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
            .map_err(|_| InferenceError::channel("Service init channel closed"))?
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;

        info!("Inference service started");
        Ok(LocalInferenceClient { sender: tx })
    }

    /// Initialize the service (called on service thread).
    async fn initialize(
        model_path: PathBuf,
        config: RuntimeConfig,
        requests: mpsc::UnboundedReceiver<InferenceRequest>,
    ) -> Result<Self, anyhow::Error> {
        let mut engine = TorchEngine::new(config.clone())?;
        engine.load_model(&model_path).await?;

        // Initialize KV cache registry for session-based cache isolation
        // This enables concurrent inference with isolated context per session
        let model_info = engine.model_info();
        let num_layers = model_info.num_hidden_layers.unwrap_or(32);
        let max_seq_len = config.max_context.unwrap_or(model_info.context_length);
        engine.initialize_kv_registry(
            num_layers,
            max_seq_len,
            config.kv_quant_type,
            None, // No memory budget limit for now
        );
        info!(
            "KV cache registry initialized: {} layers, max_seq_len={}",
            num_layers, max_seq_len
        );

        Ok(Self {
            engine,
            requests,
            shutdown_requested: false,
        })
    }

    /// Main service loop - processes requests until shutdown or channel closes.
    async fn run(mut self) {
        debug!("Service loop started");
        while let Some(request) = self.requests.recv().await {
            if self.shutdown_requested {
                debug!("Shutdown requested, draining remaining requests");
                // Continue processing to drain queue gracefully
            }
            debug!(?request, "Processing request");
            self.handle_request(request).await;

            if self.shutdown_requested {
                break;
            }
        }
        info!("Service loop ended");
    }

    /// Handle a single request.
    async fn handle_request(&mut self, request: InferenceRequest) {
        match request {
            InferenceRequest::Generate { request, reply } => {
                let result = self.handle_generate(request).await;
                let _ = reply.send(result);
            }

            InferenceRequest::GenerateStream {
                request,
                chunk_sender,
                stats_sender,
                reply,
            } => {
                // Acknowledge request receipt
                let _ = reply.send(Ok(()));
                // Run streaming generation
                self.handle_generate_stream(request, chunk_sender, stats_sender)
                    .await;
            }

            InferenceRequest::ModelInfo { reply } => {
                let info = self.engine.model_info();
                let _ = reply.send(Ok(info));
            }

            InferenceRequest::IsReady { reply } => {
                // Engine is ready if model is loaded
                let ready = !self.engine.model_info().name.is_empty();
                let _ = reply.send(Ok(ready));
            }

            InferenceRequest::ApplyChatTemplate {
                messages,
                add_generation_prompt,
                reply,
            } => {
                let result = self
                    .engine
                    .apply_chat_template(&messages, add_generation_prompt)
                    .map_err(|e| InferenceError::Internal(e.to_string()));
                let _ = reply.send(result);
            }

            InferenceRequest::CreateLora { config, reply } => {
                let result = self
                    .engine
                    .create_lora(config)
                    .map_err(|e| InferenceError::LoRA(e.to_string()));
                let _ = reply.send(result);
            }

            InferenceRequest::LoadLora { path, reply } => {
                let result = self
                    .engine
                    .load_lora_from_file(&path)
                    .await
                    .map_err(|e| InferenceError::LoRA(e.to_string()));
                let _ = reply.send(result);
            }

            InferenceRequest::SaveLora { path, reply } => {
                let result = path.to_str()
                    .ok_or_else(|| InferenceError::LoRA("Invalid path".to_string()))
                    .and_then(|p| {
                        self.engine.save_lora(p)
                            .map_err(|e| InferenceError::LoRA(e.to_string()))
                    });
                let _ = reply.send(result);
            }

            InferenceRequest::UnloadLora { reply } => {
                let result = self.engine.unload_lora()
                    .map_err(|e| InferenceError::LoRA(e.to_string()));
                let _ = reply.send(result);
            }

            InferenceRequest::HasLora { reply } => {
                let has_lora = self.engine.has_lora_model();
                let _ = reply.send(Ok(has_lora));
            }

            InferenceRequest::SetSession { session_id, reply } => {
                let result = self.engine.set_session(CacheOwner::Session(session_id))
                    .map_err(|e| InferenceError::Session(e.to_string()));
                let _ = reply.send(result);
            }

            InferenceRequest::ClearSession { reply } => {
                self.engine.clear_kv_cache();
                let _ = reply.send(Ok(()));
            }

            InferenceRequest::ReleaseSession { session_id, reply } => {
                let result = self.engine.release_session(&CacheOwner::Session(session_id.clone()))
                    .map_err(|e| InferenceError::Session(e.to_string()));
                let _ = reply.send(result);
            }

            InferenceRequest::HealthCheck { reply } => {
                // Service is healthy if we can respond
                let _ = reply.send(Ok(()));
            }

            InferenceRequest::Shutdown { reply } => {
                info!("Shutdown requested");
                self.shutdown_requested = true;
                let _ = reply.send(Ok(()));
            }
        }
    }

    /// Handle non-streaming generation.
    async fn handle_generate(
        &mut self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, InferenceError> {
        self.engine
            .generate_with_params(request)
            .await
            .map_err(|e| InferenceError::Generation(e.to_string()))
    }

    /// Handle streaming generation.
    ///
    /// This method runs the TextStream internally and pumps chunks
    /// to the client via the provided channel.
    async fn handle_generate_stream(
        &mut self,
        request: GenerationRequest,
        chunk_sender: mpsc::Sender<Result<String, InferenceError>>,
        stats_sender: oneshot::Sender<StreamStats>,
    ) {
        use futures::StreamExt;

        let stream_result = self.engine.generate(request);

        match stream_result {
            Ok(mut stream) => {
                // Pump stream to channel
                while let Some(chunk_result) = stream.next().await {
                    let mapped =
                        chunk_result.map_err(|e| InferenceError::Generation(e.to_string()));
                    if chunk_sender.send(mapped).await.is_err() {
                        // Client dropped, stop generating
                        warn!("Client dropped, stopping generation");
                        break;
                    }
                }

                // Send final stats
                let stats = StreamStats::from_generation_stats(&stream.stats());
                let _ = stats_sender.send(stats);
            }
            Err(e) => {
                error!("Generation failed: {}", e);
                let _ = chunk_sender
                    .send(Err(InferenceError::Generation(e.to_string())))
                    .await;
                let _ = stats_sender.send(StreamStats::default());
            }
        }
    }
}

/// Lightweight, cloneable client handle for the inference service.
///
/// Clients can be cloned and shared across tasks. All clients communicate
/// with the same underlying service instance.
#[derive(Clone)]
pub struct LocalInferenceClient {
    sender: mpsc::UnboundedSender<InferenceRequest>,
}

#[async_trait]
impl InferenceClient for LocalInferenceClient {
    async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::Generate { request, reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<StreamHandle, InferenceError> {
        let (chunk_tx, chunk_rx) = mpsc::channel(32); // Bounded for backpressure
        let (stats_tx, stats_rx) = oneshot::channel();
        let (reply_tx, reply_rx) = oneshot::channel();

        self.sender
            .send(InferenceRequest::GenerateStream {
                request,
                chunk_sender: chunk_tx,
                stats_sender: stats_tx,
                reply: reply_tx,
            })
            .map_err(|_| InferenceError::Unavailable)?;

        // Wait for acknowledgment
        reply_rx
            .await
            .map_err(|_| InferenceError::channel("No response"))??;

        Ok(StreamHandle::new(chunk_rx, stats_rx))
    }

    async fn model_info(&self) -> Result<ModelInfo, InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::ModelInfo { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn is_ready(&self) -> Result<bool, InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::IsReady { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn apply_chat_template(
        &self,
        messages: &[crate::runtime::template_engine::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::ApplyChatTemplate {
                messages: messages.to_vec(),
                add_generation_prompt,
                reply: tx,
            })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn create_lora(&self, config: LoRAConfig) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::CreateLora { config, reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn load_lora(&self, path: &Path) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::LoadLora {
                path: path.to_path_buf(),
                reply: tx,
            })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn save_lora(&self, path: &Path) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::SaveLora {
                path: path.to_path_buf(),
                reply: tx,
            })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn unload_lora(&self) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::UnloadLora { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn has_lora(&self) -> Result<bool, InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::HasLora { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn set_session(&self, session_id: String) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::SetSession {
                session_id,
                reply: tx,
            })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn clear_session(&self) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::ClearSession { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn release_session(&self, session_id: &str) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::ReleaseSession {
                session_id: session_id.to_string(),
                reply: tx,
            })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn health_check(&self) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::HealthCheck { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }

    async fn shutdown(&self) -> Result<(), InferenceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(InferenceRequest::Shutdown { reply: tx })
            .map_err(|_| InferenceError::Unavailable)?;
        rx.await.map_err(|_| InferenceError::channel("No response"))?
    }
}
