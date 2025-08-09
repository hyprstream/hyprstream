//! Inference API for applying LoRA adapters to base models

use crate::adapters::sparse_lora::SparseLoRAAdapter;
#[cfg(feature = "vdb")]
use crate::storage::vdb::hardware_accelerated::HardwareVDBStorage;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::Result;

pub mod model_loader;
pub mod inference_engine;
pub mod lora_fusion;

// Re-export unified config system
pub use crate::config::{HyprConfig, GenerationRequest, GenerationResult, ModelInfo};

pub use model_loader::{ModelLoader, BaseModelHandle};
pub use inference_engine::{InferenceEngine, InferenceEngineStats};
pub use lora_fusion::{LoRAFusion, FusionStrategy};

/// Inference session with active LoRA adapters
#[derive(Debug, Clone)]
pub struct InferenceSession {
    pub session_id: String,
    pub model_name: String,
    pub active_adapters: Vec<String>,
    pub adapter_weights: HashMap<String, f32>, // Mixing weights for multiple adapters
    pub created_at: i64,
    pub last_used: i64,
}

/// Inference statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    pub total_requests: u64,
    pub total_tokens_generated: u64,
    pub avg_latency_ms: f64,
    pub active_sessions: u64,
    pub adapter_cache_hits: u64,
    pub adapter_cache_misses: u64,
    pub gpu_utilization: f32,
    pub memory_usage_mb: u64,
}

/// Main inference API
pub struct InferenceAPI {
    /// Base model storage (mmap'd weights)
    model_loader: Arc<ModelLoader>,
    
    /// VDB storage for LoRA adapters (only available with VDB feature)
    #[cfg(feature = "vdb")]
    vdb_storage: Arc<HardwareVDBStorage>,
    
    /// Inference engine
    engine: Arc<InferenceEngine>,
    
    /// Active inference sessions
    sessions: Arc<RwLock<HashMap<String, InferenceSession>>>,
    
    /// Statistics
    stats: Arc<RwLock<InferenceStats>>,
}

impl InferenceAPI {
    /// Create new inference API
    pub async fn new(
        model_path: &Path,
        #[cfg(feature = "vdb")] vdb_storage: Arc<HardwareVDBStorage>,
        config: HyprConfig,
    ) -> Result<Self> {
        // Load base model with mmap
        let model_loader = Arc::new(ModelLoader::new(model_path).await?);
        
        // Create inference engine
        let engine = Arc::new(InferenceEngine::new(config)?);
        
        Ok(Self {
            model_loader,
            #[cfg(feature = "vdb")]
            vdb_storage,
            engine,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(InferenceStats {
                total_requests: 0,
                total_tokens_generated: 0,
                avg_latency_ms: 0.0,
                active_sessions: 0,
                adapter_cache_hits: 0,
                adapter_cache_misses: 0,
                gpu_utilization: 0.0,
                memory_usage_mb: 0,
            })),
        })
    }
    
    /// Create inference session with specified adapters
    pub async fn create_session(
        &self,
        model_name: String,
        adapter_ids: Vec<String>,
        adapter_weights: Option<HashMap<String, f32>>,
    ) -> Result<String> {
        let session_id = format!("session_{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;
        
        // Default equal weights if not specified
        let weights = adapter_weights.unwrap_or_else(|| {
            let weight = 1.0 / adapter_ids.len() as f32;
            adapter_ids.iter()
                .map(|id| (id.clone(), weight))
                .collect()
        });
        
        let session = InferenceSession {
            session_id: session_id.clone(),
            model_name,
            active_adapters: adapter_ids,
            adapter_weights: weights,
            created_at: now,
            last_used: now,
        };
        
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);
        
        let mut stats = self.stats.write().await;
        stats.active_sessions = sessions.len() as u64;
        
        Ok(session_id)
    }
    
    /// Run inference with LoRA adapters
    pub async fn infer(
        &self,
        session_id: &str,
        input: InferenceInput,
    ) -> Result<InferenceOutput> {
        let start = std::time::Instant::now();
        
        // Get session
        let session = {
            let mut sessions = self.sessions.write().await;
            let session = sessions.get_mut(session_id)
                .ok_or_else(|| anyhow::anyhow!("Session not found"))?;
            session.last_used = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs() as i64;
            session.clone()
        };
        
        // Load and fuse adapters
        let fused_weights = self.load_and_fuse_adapters(&session).await?;
        
        // Run inference with fused weights
        let output = self.engine.infer(
            &self.model_loader,
            &fused_weights,
            input,
        ).await?;
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
            stats.total_tokens_generated += output.tokens_generated as u64;
            let latency = start.elapsed().as_millis() as f64;
            stats.avg_latency_ms = (stats.avg_latency_ms * (stats.total_requests - 1) as f64 
                + latency) / stats.total_requests as f64;
        }
        
        Ok(output)
    }
    
    /// Stream inference with dynamic weight updates
    pub async fn stream_infer(
        &self,
        session_id: &str,
        input: InferenceInput,
        update_channel: tokio::sync::mpsc::Receiver<SparseWeightUpdate>,
    ) -> Result<impl futures::Stream<Item = Result<InferenceToken>>> {
        let session = {
            let sessions = self.sessions.read().await;
            sessions.get(session_id)
                .ok_or_else(|| anyhow::anyhow!("Session not found"))?
                .clone()
        };
        
        // Create streaming inference
        #[cfg(feature = "vdb")]
        let stream = self.engine.stream_infer_with_updates(
            &self.model_loader,
            &self.vdb_storage,
            session,
            input,
            update_channel,
        ).await?;
        
        #[cfg(not(feature = "vdb"))]
        let stream = self.engine.stream_infer_with_updates(
            &self.model_loader,
            session,
            input,
            update_channel,
        ).await?;
        
        Ok(stream)
    }
    
    /// Load and fuse multiple LoRA adapters
    async fn load_and_fuse_adapters(
        &self,
        session: &InferenceSession,
    ) -> Result<FusedAdapterWeights> {
        let mut adapters = Vec::new();
        let mut stats = self.stats.write().await;
        
        #[cfg(feature = "vdb")]
        {
            for adapter_id in &session.active_adapters {
                // Load from VDB storage
                match self.vdb_storage.load_adapter_neural_compressed(
                    adapter_id,
                    Default::default(),
                ).await {
                    Ok(adapter) => {
                        stats.adapter_cache_hits += 1;
                        adapters.push((adapter_id.clone(), adapter));
                    }
                    Err(_) => {
                        stats.adapter_cache_misses += 1;
                        return Err(anyhow::anyhow!("Failed to load adapter: {}", adapter_id));
                    }
                }
            }
        }
        
        #[cfg(not(feature = "vdb"))]
        {
            // Without VDB feature, cannot load adapters
            return Err(anyhow::anyhow!("VDB feature not enabled - cannot load adapters"));
        }
        
        // Fuse adapters with specified weights
        let mut fusion = LoRAFusion::new(FusionStrategy::WeightedAverage);
        let fused = fusion.fuse_adapters(adapters, &session.adapter_weights)?;
        
        Ok(fused)
    }
    
    /// Update adapter dynamically during inference
    pub async fn update_adapter(
        &self,
        adapter_id: &str,
        updates: HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        // Convert to sparse updates
        let sparse_updates = self.convert_to_sparse_updates(updates)?;
        
        // Apply updates through VDB storage
        #[cfg(feature = "vdb")]
        {
            self.vdb_storage.gpu_sparse_update(adapter_id, &sparse_updates).await?;
        }
        #[cfg(not(feature = "vdb"))]
        {
            // Placeholder - VDB not available, skip sparse updates
            tracing::warn!("VDB storage not available - skipping sparse updates for adapter: {}", adapter_id);
        }
        
        Ok(())
    }
    
    /// Get inference statistics
    pub async fn get_stats(&self) -> InferenceStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// List active sessions
    pub async fn list_sessions(&self) -> Vec<InferenceSession> {
        let sessions = self.sessions.read().await;
        sessions.values().cloned().collect()
    }
    
    /// Close session
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found"))?;
        
        let mut stats = self.stats.write().await;
        stats.active_sessions = sessions.len() as u64;
        
        Ok(())
    }
    
    fn convert_to_sparse_updates(
        &self,
        updates: HashMap<String, Vec<f32>>,
    ) -> Result<HashMap<crate::storage::vdb::grid::Coordinate3D, f32>> {
        // Convert dense updates to sparse format
        let mut sparse = HashMap::new();
        
        for (key, values) in updates {
            // Parse layer and position from key
            // Example: "layer.0.self_attn.q_proj.weight[100,200]"
            if let Some((layer, coords)) = parse_weight_key(&key) {
                for (idx, value) in values.iter().enumerate() {
                    if value.abs() > 1e-6 { // Only non-zero values
                        let coord = crate::storage::vdb::grid::Coordinate3D::new(
                            coords.0 + idx as i32,
                            coords.1,
                            layer as i32,
                        );
                        sparse.insert(coord, *value);
                    }
                }
            }
        }
        
        Ok(sparse)
    }
}

/// Input for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceInput {
    pub prompt: Option<String>,
    pub input_ids: Option<Vec<i64>>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
}

/// Output from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub text: String,
    pub tokens: Vec<i64>,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub adapter_contribution: HashMap<String, f32>,
}

/// Single token for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceToken {
    pub token: String,
    pub token_id: i64,
    pub logprob: f32,
    pub timestamp_ms: i64,
}

/// Fused adapter weights
pub struct FusedAdapterWeights {
    pub weights: HashMap<String, SparseLoRAAdapter>,
    pub fusion_metadata: FusionMetadata,
}

/// Metadata about adapter fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    pub num_adapters: usize,
    pub total_sparse_weights: usize,
    pub fusion_strategy: String,
    pub timestamp: i64,
}

/// Sparse weight update for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseWeightUpdate {
    pub adapter_id: String,
    pub layer_name: String,
    pub coordinates: Vec<(i32, i32, i32)>,
    pub values: Vec<f32>,
    pub timestamp: i64,
}

fn parse_weight_key(_key: &str) -> Option<(usize, (i32, i32))> {
    // Parse keys like "layer.0.weight[100,200]"
    // This is a simplified parser - implement based on actual key format
    None
}

// Re-export uuid if not already available
use uuid;