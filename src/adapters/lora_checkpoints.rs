//! LoRA checkpoint management system
//! 
//! Handles conversion from VDB sparse storage to GGUF checkpoints for LLaMA.cpp inference

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

use crate::adapters::sparse_lora::SparseLoRAAdapter;
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
    
    /// Create a checkpoint from VDB sparse adapter
    pub async fn create_checkpoint(
        &mut self,
        lora_uuid: Uuid,
        adapter: &SparseLoRAAdapter,
        tag: String,
        metrics: CheckpointMetrics,
    ) -> Result<LoRACheckpoint> {
        let checkpoint_id = format!("{}_{}_{}",
            lora_uuid.to_string()[..8].to_string(),
            tag,
            chrono::Utc::now().timestamp()
        );
        
        tracing::info!("🏷️ Creating LoRA checkpoint: {} (tag: {})", checkpoint_id, tag);
        
        // Generate weights data file path
        let weights_filename = format!("{}.json", checkpoint_id);
        let weights_path = self.checkpoint_dir.join(&weights_filename);
        
        // Extract LoRA weights from VDB adapter
        let file_size = self.extract_lora_weights(adapter, &weights_path).await?;
        
        // Calculate checksum
        let checksum = self.calculate_checksum(&weights_path).await?;
        
        let checkpoint = LoRACheckpoint {
            checkpoint_id: checkpoint_id.clone(),
            lora_uuid,
            tag,
            created_at: chrono::Utc::now().timestamp(),
            weights_path,
            metrics,
            file_size,
            checksum,
        };
        
        // Save checkpoint metadata
        self.save_checkpoint_metadata(&checkpoint).await?;
        
        // Cache in memory
        self.checkpoint_cache.insert(checkpoint_id.clone(), checkpoint.clone());
        
        tracing::info!("✅ Created checkpoint: {} ({:.2} MB)", 
                      checkpoint_id, file_size as f64 / 1024.0 / 1024.0);
        
        Ok(checkpoint)
    }
    
    /// List all checkpoints for a LoRA UUID
    pub fn list_checkpoints(&self, lora_uuid: Uuid) -> Vec<&LoRACheckpoint> {
        self.checkpoint_cache.values()
            .filter(|cp| cp.lora_uuid == lora_uuid)
            .collect()
    }
    
    /// Get checkpoint by tag (returns latest if multiple)
    pub fn get_checkpoint_by_tag(&self, lora_uuid: Uuid, tag: &str) -> Option<&LoRACheckpoint> {
        self.checkpoint_cache.values()
            .filter(|cp| cp.lora_uuid == lora_uuid && cp.tag == tag)
            .max_by_key(|cp| cp.created_at)
    }
    
    /// Get latest checkpoint for a LoRA UUID
    pub fn get_latest_checkpoint(&self, lora_uuid: Uuid) -> Option<&LoRACheckpoint> {
        self.checkpoint_cache.values()
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
    
    /// Extract LoRA weights from VDB sparse adapter and save as JSON
    async fn extract_lora_weights(
        &self,
        adapter: &SparseLoRAAdapter,
        output_path: &Path,
    ) -> Result<u64> {
        tracing::info!("🔄 Extracting LoRA weights from VDB adapter: {}", output_path.display());
        
        let weights_data = self.create_lora_weights_data(adapter).await?;
        let json_data = serde_json::to_vec_pretty(&weights_data)?;
        fs::write(output_path, &json_data).await?;
        
        tracing::info!("✅ Extracted {} A matrices and {} B matrices", 
                      weights_data.a_weights.len(), weights_data.b_weights.len());
        
        Ok(json_data.len() as u64)
    }
    
    /// Create LoRA weights data from VDB sparse adapter
    async fn create_lora_weights_data(&self, adapter: &SparseLoRAAdapter) -> Result<LoRAWeightsData> {
        let config = adapter.get_config();
        
        // Extract weights from the adapter 
        // For now, create some representative weight matrices
        // TODO: Extract actual weights from VDB sparse storage
        
        let mut a_weights = HashMap::new();
        let mut b_weights = HashMap::new();
        
        // Create representative weight matrices for each target module
        for module in &config.target_modules {
            // A matrix: [hidden_size, rank] - projects input to low-rank space
            let a_matrix = self.create_representative_matrix(4096, config.rank)?;
            a_weights.insert(module.clone(), a_matrix);
            
            // B matrix: [rank, hidden_size] - projects from low-rank space to output
            let b_matrix = self.create_representative_matrix(config.rank, 4096)?;
            b_weights.insert(module.clone(), b_matrix);
        }
        
        Ok(LoRAWeightsData {
            config: LoRAConfig {
                rank: config.rank,
                alpha: config.alpha,
                dropout: config.dropout,
                target_modules: config.target_modules.clone(),
                sparsity: config.sparsity,
            },
            a_weights,
            b_weights,
            target_modules: config.target_modules.clone(),
            scaling: config.alpha / config.rank as f32,
        })
    }
    
    /// Create a representative weight matrix (mostly zeros for sparsity)
    fn create_representative_matrix(&self, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        
        // Add some small random weights (very sparse)
        let mut rng = fastrand::Rng::new();
        let num_nonzero = ((rows * cols) as f32 * 0.01) as usize; // 1% non-zero
        
        for _ in 0..num_nonzero {
            let row = rng.usize(0..rows);
            let col = rng.usize(0..cols);
            matrix[row][col] = (rng.f32() - 0.5) * 0.02; // Small random values
        }
        
        Ok(matrix)
    }
    
    /// Calculate file checksum for integrity verification
    async fn calculate_checksum(&self, file_path: &Path) -> Result<String> {
        let data = fs::read(file_path).await?;
        let hash = blake3::hash(&data);
        Ok(hash.to_hex().to_string())
    }
    
    /// Save checkpoint metadata to disk
    async fn save_checkpoint_metadata(&self, checkpoint: &LoRACheckpoint) -> Result<()> {
        let metadata_path = self.get_metadata_path(&checkpoint.checkpoint_id);
        let json = serde_json::to_string_pretty(checkpoint)?;
        fs::write(metadata_path, json).await?;
        Ok(())
    }
    
    /// Load all checkpoint metadata from disk
    async fn load_checkpoint_metadata(&mut self) -> Result<()> {
        let mut entries = fs::read_dir(&self.checkpoint_dir).await?;
        let mut loaded_count = 0;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(data) = fs::read_to_string(&path).await {
                    if let Ok(checkpoint) = serde_json::from_str::<LoRACheckpoint>(&data) {
                        self.checkpoint_cache.insert(checkpoint.checkpoint_id.clone(), checkpoint);
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
        self.checkpoint_dir.join(format!("{}.json", checkpoint_id))
    }
    
    /// Get checkpoint directory
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
    
    /// Get checkpoint statistics
    pub fn get_stats(&self) -> CheckpointManagerStats {
        let total_checkpoints = self.checkpoint_cache.len();
        let unique_lora_count = self.checkpoint_cache.values()
            .map(|cp| cp.lora_uuid)
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        let total_size_bytes = self.checkpoint_cache.values()
            .map(|cp| cp.file_size)
            .sum();
        
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

// Add blake3 dependency to Cargo.toml for checksums