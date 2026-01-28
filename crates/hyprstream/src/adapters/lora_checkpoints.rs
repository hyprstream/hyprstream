//! LoRA checkpoint management system
//!
//! Handles conversion from VDB sparse storage to checkpoints for inference

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

use crate::storage::StoragePaths;

/// Metadata for a LoRA checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRACheckpoint {
    /// Unique checkpoint ID
    pub checkpoint_id: String,
    /// Parent LoRA adapter UUID
    pub lora_uuid: Uuid,
    /// Human-readable tag (e.g., "v1.0", "epoch_100", "best")
    pub tag: String,
    /// Timestamp when checkpoint was created
    pub created_at: i64,
    /// Path to weights data file
    pub weights_path: PathBuf,
    /// Training metrics at checkpoint time
    pub metrics: CheckpointMetrics,
    /// File size in bytes
    pub file_size: u64,
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Serializable LoRA weights data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAWeightsData {
    /// LoRA configuration
    pub config: LoRAConfig,
    /// A matrix weights (input → low-rank)
    pub a_weights: HashMap<String, Vec<Vec<f32>>>,
    /// B matrix weights (low-rank → output)
    pub b_weights: HashMap<String, Vec<Vec<f32>>>,
    /// Target module names
    pub target_modules: Vec<String>,
    /// Scaling factor (alpha/rank)
    pub scaling: f32,
}

/// LoRA configuration for checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub sparsity: f32,
}

/// Training metrics captured at checkpoint time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetrics {
    /// Training loss at checkpoint
    pub loss: Option<f32>,
    /// Number of training steps
    pub steps: u64,
    /// Sparsity ratio (percentage of zero weights)
    pub sparsity: f32,
    /// Total number of active parameters
    pub active_params: u64,
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha parameter
    pub alpha: f32,
}

/// Manages LoRA checkpoints with UUID-based tagging
pub struct LoRACheckpointManager {
    /// Storage paths configuration
    #[allow(dead_code)]
    storage_paths: StoragePaths,
    /// In-memory cache of checkpoint metadata
    checkpoint_cache: HashMap<String, LoRACheckpoint>,
    /// Base directory for checkpoint storage
    checkpoint_dir: PathBuf,
}

impl LoRACheckpointManager {
    /// Create new checkpoint manager
    pub async fn new() -> Result<Self> {
        let storage_paths = StoragePaths::new()?;
        let checkpoint_dir = storage_paths.cache_dir()?.join("lora_checkpoints");

        // Ensure checkpoint directory exists
        fs::create_dir_all(&checkpoint_dir).await?;

        let mut manager = Self {
            storage_paths,
            checkpoint_cache: HashMap::new(),
            checkpoint_dir,
        };

        // Load existing checkpoints
        manager.load_checkpoint_metadata().await?;

        Ok(manager)
    }

    /// List all checkpoints for a LoRA UUID
    pub fn list_checkpoints(&self, lora_uuid: Uuid) -> Vec<&LoRACheckpoint> {
        self.checkpoint_cache
            .values()
            .filter(|cp| cp.lora_uuid == lora_uuid)
            .collect()
    }

    /// Get checkpoint by tag (returns latest if multiple)
    pub fn get_checkpoint_by_tag(&self, lora_uuid: Uuid, tag: &str) -> Option<&LoRACheckpoint> {
        self.checkpoint_cache
            .values()
            .filter(|cp| cp.lora_uuid == lora_uuid && cp.tag == tag)
            .max_by_key(|cp| cp.created_at)
    }

    /// Get latest checkpoint for a LoRA UUID
    pub fn get_latest_checkpoint(&self, lora_uuid: Uuid) -> Option<&LoRACheckpoint> {
        self.checkpoint_cache
            .values()
            .filter(|cp| cp.lora_uuid == lora_uuid)
            .max_by_key(|cp| cp.created_at)
    }

    /// Delete a checkpoint
    pub async fn delete_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        if let Some(checkpoint) = self.checkpoint_cache.remove(checkpoint_id) {
            // Delete weights file
            if checkpoint.weights_path.exists() {
                fs::remove_file(&checkpoint.weights_path).await?;
            }

            // Delete metadata file
            let metadata_path = self.get_metadata_path(&checkpoint.checkpoint_id);
            if metadata_path.exists() {
                fs::remove_file(&metadata_path).await?;
            }

            tracing::info!("🗑️ Deleted checkpoint: {}", checkpoint_id);
        }

        Ok(())
    }

    /// Load all checkpoint metadata from disk
    async fn load_checkpoint_metadata(&mut self) -> Result<()> {
        let mut entries = fs::read_dir(&self.checkpoint_dir).await?;
        let mut loaded_count = 0;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(data) = fs::read_to_string(&path).await {
                    if let Ok(checkpoint) = serde_json::from_str::<LoRACheckpoint>(&data) {
                        self.checkpoint_cache
                            .insert(checkpoint.checkpoint_id.clone(), checkpoint);
                        loaded_count += 1;
                    }
                }
            }
        }

        if loaded_count > 0 {
            tracing::info!("📚 Loaded {} LoRA checkpoints from disk", loaded_count);
        }

        Ok(())
    }

    /// Get metadata file path for checkpoint
    fn get_metadata_path(&self, checkpoint_id: &str) -> PathBuf {
        self.checkpoint_dir.join(format!("{checkpoint_id}.json"))
    }

    /// Get checkpoint directory
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Get checkpoint statistics
    pub fn get_stats(&self) -> CheckpointManagerStats {
        let total_checkpoints = self.checkpoint_cache.len();
        let unique_lora_count = self
            .checkpoint_cache
            .values()
            .map(|cp| cp.lora_uuid)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let total_size_bytes = self.checkpoint_cache.values().map(|cp| cp.file_size).sum();

        CheckpointManagerStats {
            total_checkpoints,
            unique_lora_count,
            total_size_bytes,
        }
    }
}

/// Statistics about the checkpoint manager
#[derive(Debug, Clone)]
pub struct CheckpointManagerStats {
    pub total_checkpoints: usize,
    pub unique_lora_count: usize,
    pub total_size_bytes: u64,
}

impl CheckpointManagerStats {
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes as f64 / 1024.0 / 1024.0
    }
}
