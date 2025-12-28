//! Auto-regressive training service for LoRA adapters

use crate::config::{GenerationRequest, GenerationResult};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{info, warn, trace};

/// Training sample for LoRA fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub input: String,
    pub output: String,
}

/// Training status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub is_training: bool,
    pub total_samples_processed: usize,
    pub current_loss: f32,
    pub learning_rate: f32,
    pub last_update: i64,
}

/// Configuration for auto-regressive training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for training
    pub learning_rate: f32,

    /// Batch size for gradient updates
    pub batch_size: usize,

    /// Minimum samples before training starts
    pub min_samples_before_training: usize,

    /// Maximum training queue size
    pub max_queue_size: usize,

    /// Training frequency (train every N samples)
    pub training_frequency: usize,

    /// Enable gradient accumulation
    pub gradient_accumulation: bool,

    /// L2 regularization strength
    pub weight_decay: f32,

    /// Sparsity target during training
    pub sparsity_target: f32,

    /// Use mixed precision training
    pub mixed_precision: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 8,
            min_samples_before_training: 10,
            max_queue_size: 1000,
            training_frequency: 5,
            gradient_accumulation: true,
            weight_decay: 0.01,
            sparsity_target: 0.99,
            mixed_precision: true,
        }
    }
}

/// Training session state
#[derive(Debug, Clone)]
pub struct TrainingSession {
    pub lora_id: String,
    pub config: TrainingConfig,
    pub samples_processed: u64,
    pub current_loss: f32,
    pub is_active: bool,
    pub started_at: i64,
    pub last_update: i64,
}

/// Training service for auto-regressive LoRA learning
pub struct TrainingService {
    /// Active training sessions
    sessions: Arc<RwLock<HashMap<String, TrainingSession>>>,

    /// Training sample queues per LoRA - bounded to prevent memory pressure
    sample_queues: Arc<RwLock<HashMap<String, mpsc::Sender<TrainingSample>>>>,

    /// Training statistics
    stats: Arc<RwLock<TrainingStats>>,

    /// Task handles for background training
    task_handles: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

/// Training statistics
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingStats {
    pub total_samples_processed: u64,
    pub total_gradient_updates: u64,
    pub avg_training_loss: f32,
    pub active_training_sessions: u64,
    pub total_training_time_ms: u64,
}

impl Default for TrainingService {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingService {
    /// Create new training service
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            sample_queues: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(TrainingStats::default())),
            task_handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start auto-regressive training for a LoRA adapter
    pub async fn start_auto_training(&self, lora_id: &str, config: TrainingConfig) -> Result<()> {
        let now = chrono::Utc::now().timestamp();

        // Create training session
        let session = TrainingSession {
            lora_id: lora_id.to_string(),
            config: config.clone(),
            samples_processed: 0,
            current_loss: 0.0,
            is_active: true,
            started_at: now,
            last_update: now,
        };

        // Register session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(lora_id.to_string(), session);
        }

        // Create bounded sample queue to prevent memory pressure
        let queue_size = config.max_queue_size;
        let (tx, mut rx) = mpsc::channel::<TrainingSample>(queue_size);
        {
            let mut queues = self.sample_queues.write().await;
            queues.insert(lora_id.to_string(), tx);
        }

        // Start background training task
        let lora_id_clone = lora_id.to_string();

        let sessions = self.sessions.clone();
        let stats = self.stats.clone();

        let handle = tokio::spawn(async move {
            let mut sample_batch = Vec::new();

            while let Some(sample) = rx.recv().await {
                sample_batch.push(sample);

                // Train when we have enough samples
                if sample_batch.len() >= config.batch_size {
                    if let Err(e) = Self::train_batch(
                        &lora_id_clone,
                        &sample_batch,
                        &config,
                        &sessions,
                        &stats
                    ).await {
                        warn!("Training error for {}: {}", lora_id_clone, e);
                    }

                    sample_batch.clear();
                }
            }

            // Process remaining samples
            if !sample_batch.is_empty() {
                let _ = Self::train_batch(
                    &lora_id_clone,
                    &sample_batch,
                    &config,
                    &sessions,
                    &stats
                ).await;
            }

            info!("Training task ended for {}", lora_id_clone);
        });

        // Store task handle
        {
            let mut handles = self.task_handles.lock().await;
            handles.insert(lora_id.to_string(), handle);
        }

        Ok(())
    }

    /// Stop auto-regressive training
    pub async fn stop_auto_training(&self, lora_id: &str) -> Result<()> {
        // Mark session as inactive
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(lora_id) {
                session.is_active = false;
            }
        }

        // Remove sample queue (will cause training loop to exit)
        {
            let mut queues = self.sample_queues.write().await;
            queues.remove(lora_id);
        }

        // Cancel background task
        {
            let mut handles = self.task_handles.lock().await;
            if let Some(handle) = handles.remove(lora_id) {
                handle.abort();
                info!("Aborted training task for {}", lora_id);
            }
        }

        Ok(())
    }

    /// Queue a training sample for auto-regressive learning with backpressure handling
    pub async fn queue_training_sample(&self, lora_id: &str, sample: TrainingSample) -> Result<()> {
        // Check session is active before attempting to queue
        let session_active = {
            let sessions = self.sessions.read().await;
            sessions.get(lora_id)
                .map(|s| s.is_active)
                .unwrap_or(false)
        };

        if !session_active {
            return Err(anyhow::anyhow!("No active training session for {}", lora_id));
        }

        // Get queue sender (short-lived lock)
        let sender = {
            let queues = self.sample_queues.read().await;
            queues.get(lora_id).cloned()
        };

        if let Some(sender) = sender {
            // Try to send with backpressure handling
            match sender.try_send(sample) {
                Ok(()) => {
                    trace!("Sample queued for training: {}", lora_id);
                    Ok(())
                }
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    warn!("Training queue full for {}, dropping sample", lora_id);
                    Err(anyhow::anyhow!("Training queue full for {}", lora_id))
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    warn!("Training queue closed for {}", lora_id);
                    Err(anyhow::anyhow!("Training queue closed for {}", lora_id))
                }
            }
        } else {
            // Queue disappeared between session check and now
            warn!("Training queue disappeared for {}", lora_id);
            Err(anyhow::anyhow!("Training session for {} is shutting down", lora_id))
        }
    }

    
    
    /// Train a batch of samples
    async fn train_batch(
        lora_id: &str,
        samples: &[TrainingSample],
        _config: &TrainingConfig,
        _sessions: &Arc<RwLock<HashMap<String, TrainingSession>>>,
        _stats: &Arc<RwLock<TrainingStats>>,
    ) -> Result<()> {
        info!(
            "🎯 Training LoRA {} with {} samples",
            lora_id,
            samples.len()
        );

        // TODO: Implement actual training logic
        Err(anyhow::anyhow!("Training not available"))
    }

    /// Get training status for a LoRA
    pub async fn get_training_status(&self, lora_id: &str) -> Result<TrainingStatus> {
        let sessions = self.sessions.read().await;
        let session = sessions
            .get(lora_id)
            .ok_or_else(|| anyhow::anyhow!("Training session not found"))?;

        Ok(TrainingStatus {
            is_training: session.is_active,
            total_samples_processed: session.samples_processed as usize,
            current_loss: session.current_loss,
            learning_rate: session.config.learning_rate,
            last_update: session.last_update,
        })
    }

    /// Create inference session (delegated to inference API)
    pub async fn create_inference_session(
        &self,
        lora_id: &str,
        _adapter_ids: Vec<String>,
    ) -> Result<String> {
        Ok(format!("lora-session-{}", lora_id))
    }

    /// Run inference (delegated to inference API)
    pub async fn infer(
        &self,
        _session_id: &str,
        _input: GenerationRequest,
    ) -> Result<GenerationResult> {
        Ok(GenerationResult {
            text: "Inference not yet implemented".to_string(),
            tokens_generated: 0,
            finish_reason: crate::config::FinishReason::Stop,
            generation_time_ms: 0,
            tokens_per_second: 0.0,
            quality_metrics: None,
            // Prefill/inference metrics not tracked in training service
            prefill_tokens: 0,
            prefill_time_ms: 0,
            prefill_tokens_per_sec: 0.0,
            inference_tokens: 0,
            inference_time_ms: 0,
            inference_tokens_per_sec: 0.0,
        })
    }

    /// Close inference session (delegated to inference API)
    pub async fn close_inference_session(&self, _session_id: &str) -> Result<()> {
        Ok(())
    }

    /// Generate embedding (simplified implementation)
    pub async fn generate_embedding(&self, lora_id: &str, _input: &str) -> Result<Vec<f32>> {
        // Create temporary inference session
        let session_id = self
            .create_inference_session(lora_id, vec![lora_id.to_string()])
            .await?;

        // Generate embedding (simplified - would use actual model)
        let embedding = vec![0.1; 768]; // Placeholder embedding

        // Close session
        let _ = self.close_inference_session(&session_id).await;

        Ok(embedding)
    }

    /// Get overall training statistics
    pub async fn get_stats(&self) -> TrainingStats {
        let stats = self.stats.read().await;
        let sessions = self.sessions.read().await;

        let mut stats_clone = stats.clone();
        stats_clone.active_training_sessions =
            sessions.values().filter(|s| s.is_active).count() as u64;

        stats_clone
    }
}
