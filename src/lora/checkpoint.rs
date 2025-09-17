//! Simple LoRA checkpoint management
//! 
//! Provides basic save/load functionality for LoRA adapters using SafeTensors.

use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::fs;

use crate::lora::{
    config::LoRAConfig,
    torch_adapter::PyTorchLoRA,
};

/// Simple checkpoint metadata
#[derive(Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    /// Training step when checkpoint was saved
    pub step: usize,
    
    /// Training loss at checkpoint
    pub loss: f64,
    
    /// Timestamp (seconds since epoch)
    pub timestamp: i64,
    
    /// LoRA configuration
    pub config: LoRAConfig,
}

/// Manages checkpoint saving and loading
pub struct CheckpointManager {
    /// Directory to save checkpoints
    checkpoint_dir: PathBuf,
    
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(checkpoint_dir: impl AsRef<Path>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            max_checkpoints: 5, // Default to keeping last 5 checkpoints
        }
    }
    
    /// Set maximum number of checkpoints to keep
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }
    
    /// Save a checkpoint
    pub fn save_checkpoint(
        &self,
        adapter: &PyTorchLoRA,
        step: usize,
        loss: f64,
    ) -> Result<PathBuf> {
        // Create checkpoint directory if needed
        fs::create_dir_all(&self.checkpoint_dir)?;
        
        // Create checkpoint filename
        let checkpoint_name = format!("checkpoint_step_{:06}.safetensors", step);
        let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);
        
        // Save the adapter weights
        adapter.save_safetensors(&checkpoint_path)?;
        
        // Save metadata
        let info = CheckpointInfo {
            step,
            loss,
            timestamp: chrono::Utc::now().timestamp(),
            config: adapter.config().clone(),
        };
        
        let metadata_path = checkpoint_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&info)?;
        fs::write(&metadata_path, metadata_json)?;
        
        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;
        
        println!("💾 Saved checkpoint at step {} (loss: {:.4})", step, loss);
        
        Ok(checkpoint_path)
    }
    
    /// Load a checkpoint
    pub fn load_checkpoint(
        &self,
        checkpoint_path: &Path,
    ) -> Result<(PyTorchLoRA, CheckpointInfo)> {
        // Load metadata
        let metadata_path = checkpoint_path.with_extension("json");
        if !metadata_path.exists() {
            return Err(anyhow!("Checkpoint metadata not found: {:?}", metadata_path));
        }
        
        let metadata_json = fs::read_to_string(&metadata_path)?;
        let info: CheckpointInfo = serde_json::from_str(&metadata_json)?;
        
        // Load the adapter
        let device = tch::Device::cuda_if_available();
        let mut adapter = PyTorchLoRA::new(info.config.clone(), device)?;
        adapter.load_safetensors(checkpoint_path)?;
        
        println!("✅ Loaded checkpoint from step {} (loss: {:.4})", info.step, info.loss);
        
        Ok((adapter, info))
    }
    
    /// Load the latest checkpoint
    pub fn load_latest(&self) -> Result<(PyTorchLoRA, CheckpointInfo)> {
        let checkpoints = self.list_checkpoints()?;
        
        if checkpoints.is_empty() {
            return Err(anyhow!("No checkpoints found in {:?}", self.checkpoint_dir));
        }
        
        // Get the most recent checkpoint
        let latest = &checkpoints[checkpoints.len() - 1];
        self.load_checkpoint(&latest.0)
    }
    
    /// List all checkpoints sorted by step
    pub fn list_checkpoints(&self) -> Result<Vec<(PathBuf, usize)>> {
        if !self.checkpoint_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut checkpoints = Vec::new();
        
        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension() == Some(std::ffi::OsStr::new("safetensors")) {
                // Extract step number from filename
                if let Some(stem) = path.file_stem() {
                    let stem_str = stem.to_string_lossy();
                    if let Some(step_str) = stem_str.strip_prefix("checkpoint_step_") {
                        if let Ok(step) = step_str.parse::<usize>() {
                            checkpoints.push((path, step));
                        }
                    }
                }
            }
        }
        
        // Sort by step number
        checkpoints.sort_by_key(|c| c.1);
        
        Ok(checkpoints)
    }
    
    /// Remove old checkpoints beyond max_checkpoints limit
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let checkpoints = self.list_checkpoints()?;
        
        if checkpoints.len() > self.max_checkpoints {
            let to_remove = checkpoints.len() - self.max_checkpoints;
            
            for (checkpoint_path, _) in checkpoints.iter().take(to_remove) {
                // Remove both the checkpoint and its metadata
                fs::remove_file(checkpoint_path)?;
                let metadata_path = checkpoint_path.with_extension("json");
                if metadata_path.exists() {
                    fs::remove_file(metadata_path)?;
                }
                
                println!("🗑️  Removed old checkpoint: {:?}", checkpoint_path.file_name());
            }
        }
        
        Ok(())
    }
}

/// Helper to find the best checkpoint by loss
pub fn find_best_checkpoint(checkpoint_dir: &Path) -> Result<Option<(PathBuf, f64)>> {
    let manager = CheckpointManager::new(checkpoint_dir);
    let checkpoints = manager.list_checkpoints()?;
    
    let mut best: Option<(PathBuf, f64)> = None;
    
    for (path, _) in checkpoints {
        let metadata_path = path.with_extension("json");
        if let Ok(metadata_json) = fs::read_to_string(&metadata_path) {
            if let Ok(info) = serde_json::from_str::<CheckpointInfo>(&metadata_json) {
                if best.is_none() || info.loss < best.as_ref().unwrap().1 {
                    best = Some((path, info.loss));
                }
            }
        }
    }
    
    Ok(best)
}