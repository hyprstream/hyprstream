//! Async checkpoint management for training
//! 
//! Provides non-blocking checkpoint saving during training,
//! with commits to model branches for version control.

use anyhow::{Result, Context, anyhow};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::fs;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::git::{BranchManager, get_repository, create_hyprstream_signature};

/// Checkpoint request sent to background worker
#[derive(Debug, Clone)]
pub struct CheckpointRequest {
    pub step: usize,
    pub epoch: Option<usize>,
    pub weights: WeightSnapshot,
    pub metrics: Option<TrainingMetrics>,
    pub timestamp: DateTime<Utc>,
    pub commit_to_git: bool,
}

/// Snapshot of model weights for checkpointing
#[derive(Debug, Clone)]
pub enum WeightSnapshot {
    /// Weights copied to memory (for small models/adapters)
    Memory {
        data: Vec<u8>,
        format: WeightFormat,
    },
    
    /// Path to file that should be copied (for large models)
    FilePath {
        source: PathBuf,
        format: WeightFormat,
    },
    
    /// Incremental diff from previous checkpoint
    Diff {
        base_step: usize,
        changes: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightFormat {
    SafeTensors,
    PyTorch,
    AdapterBin,
}

/// Training metrics to save with checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub learning_rate: f32,
    pub gradient_norm: Option<f32>,
    pub validation_loss: Option<f32>,
    pub validation_accuracy: Option<f32>,
    pub duration_seconds: f64,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step: usize,
    pub epoch: Option<usize>,
    pub timestamp: DateTime<Utc>,
    pub metrics: Option<TrainingMetrics>,
    pub format: WeightFormat,
    pub parent_checkpoint: Option<String>,
}

/// Manages asynchronous checkpointing during training
pub struct CheckpointManager {
    model_path: PathBuf,
    checkpoint_dir: PathBuf,
    checkpoint_tx: mpsc::Sender<CheckpointRequest>,
    worker_handle: Option<JoinHandle<()>>,
    #[allow(dead_code)]
    max_checkpoints: usize,
    git_enabled: bool,
    #[allow(dead_code)]
    branch_name: Option<String>,  // Track which branch we're on
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(model_path: PathBuf) -> Result<Self> {
        Self::with_config(model_path, CheckpointConfig::default(), None)
    }
    
    /// Create checkpoint manager for an adapter branch
    pub fn for_adapter(model_path: PathBuf, branch_name: String) -> Result<Self> {
        Self::with_config(model_path, CheckpointConfig::default(), Some(branch_name))
    }
    
    /// Create with custom configuration
    pub fn with_config(model_path: PathBuf, config: CheckpointConfig, branch_name: Option<String>) -> Result<Self> {
        let checkpoint_dir = model_path.join(".checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        let (tx, rx) = mpsc::channel(config.queue_size);
        
        // Check if model path is a Git repository
        let git_enabled = model_path.join(".git").exists();
        
        // Spawn checkpoint worker thread
        let worker_handle = {
            let model_path = model_path.clone();
            let checkpoint_dir = checkpoint_dir.clone();
            let max_checkpoints = config.max_checkpoints;
            let git_interval = config.git_commit_interval;
            let branch = branch_name.clone();
            
            tokio::spawn(async move {
                Self::checkpoint_worker(
                    rx,
                    model_path,
                    checkpoint_dir,
                    max_checkpoints,
                    git_interval,
                    git_enabled,
                    branch,
                ).await;
            })
        };
        
        Ok(Self {
            model_path,
            checkpoint_dir,
            checkpoint_tx: tx,
            worker_handle: Some(worker_handle),
            max_checkpoints: config.max_checkpoints,
            git_enabled,
            branch_name,
        })
    }
    
    /// Request asynchronous checkpoint
    pub async fn checkpoint(&self, request: CheckpointRequest) -> Result<()> {
        self.checkpoint_tx.send(request).await
            .context("Failed to send checkpoint request")?;
        Ok(())
    }
    
    /// Create checkpoint request for current training state
    pub fn create_request(
        step: usize,
        weights: WeightSnapshot,
        metrics: Option<TrainingMetrics>,
    ) -> CheckpointRequest {
        CheckpointRequest {
            step,
            epoch: None,
            weights,
            metrics,
            timestamp: Utc::now(),
            commit_to_git: step % 10000 == 0, // Commit every 10k steps by default
        }
    }
    
    /// Write checkpoint to filesystem without Git commit
    pub async fn write_checkpoint(
        &self,
        weights: WeightSnapshot,
        step: usize,
        metadata: Option<TrainingMetrics>,
    ) -> Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join(format!("step-{}.safetensors", step));

        // Write weights directly to filesystem
        match weights {
            WeightSnapshot::Memory { ref data, format: _ } => {
                // Direct write from memory
                fs::write(&checkpoint_path, data).await?;
            }
            WeightSnapshot::FilePath { ref source, format: _ } => {
                // Use copy-on-write if available
                #[cfg(target_os = "linux")]
                {
                    use std::process::Command;
                    let result = Command::new("cp")
                        .args(&["--reflink=auto",
                               source.to_str().unwrap(),
                               checkpoint_path.to_str().unwrap()])
                        .output();

                    if result.is_err() || !result.unwrap().status.success() {
                        fs::copy(source, &checkpoint_path).await?;
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    fs::copy(source, &checkpoint_path).await?;
                }
            }
            WeightSnapshot::Diff { base_step, ref changes } => {
                // Save diff file
                let diff_path = self.checkpoint_dir.join(format!("diff_from_{}_to_{}.bin", base_step, step));
                fs::write(&diff_path, changes).await?;
                return Ok(diff_path);
            }
        }

        // Save metadata if provided
        if let Some(metrics) = metadata {
            let metadata = CheckpointMetadata {
                step,
                epoch: None,
                timestamp: Utc::now(),
                metrics: Some(metrics),
                format: WeightFormat::SafeTensors,
                parent_checkpoint: None,
            };

            let metadata_path = checkpoint_path.with_extension("json");
            let metadata_json = serde_json::to_string_pretty(&metadata)?;
            fs::write(&metadata_path, metadata_json).await?;
        }

        tracing::info!("Checkpoint written to filesystem: {}", checkpoint_path.display());
        Ok(checkpoint_path)
    }

    /// Commit existing checkpoint to Git (separate from write)
    pub async fn commit_checkpoint(
        &self,
        checkpoint_path: &Path,
        message: Option<String>,
        branch: Option<String>,
    ) -> Result<String> {
        if !self.git_enabled {
            return Err(anyhow!("Git is not enabled for this model path"));
        }

        // Open repository
        let repo = get_repository(&self.model_path)?;

        // Switch branch if specified
        if let Some(branch_name) = branch {
            let branch_ref = format!("refs/heads/{}", branch_name);
            if repo.find_reference(&branch_ref).is_ok() {
                repo.set_head(&branch_ref)?;
                repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            } else {
                // Create new branch
                let head = repo.head()?.peel_to_commit()?;
                repo.branch(&branch_name, &head, false)?;
                repo.set_head(&branch_ref)?;
            }
        }

        // Add checkpoint to index
        let mut index = repo.index()?;
        let relative_path = checkpoint_path.strip_prefix(&self.model_path)
            .unwrap_or(checkpoint_path);
        index.add_path(relative_path)?;

        // Also add metadata if it exists
        let metadata_path = checkpoint_path.with_extension("json");
        if metadata_path.exists() {
            let relative_metadata = metadata_path.strip_prefix(&self.model_path)
                .unwrap_or(&metadata_path);
            index.add_path(relative_metadata)?;
        }

        index.write()?;

        // Create commit
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = create_hyprstream_signature()?;
        let parent = repo.head()?.peel_to_commit()?;

        let commit_message = message.unwrap_or_else(|| {
            format!("Training checkpoint: {}", checkpoint_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"))
        });

        let oid = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            &commit_message,
            &tree,
            &[&parent],
        )?;

        tracing::info!("Checkpoint committed to Git: {}", oid);
        Ok(oid.to_string())
    }

    /// Wait for all pending checkpoints to complete
    pub async fn flush(&self) -> Result<()> {
        // Send a sentinel value to ensure all previous requests are processed
        let sentinel = CheckpointRequest {
            step: usize::MAX,
            epoch: None,
            weights: WeightSnapshot::Memory {
                data: vec![],
                format: WeightFormat::SafeTensors,
            },
            metrics: None,
            timestamp: Utc::now(),
            commit_to_git: false,
        };

        self.checkpoint(sentinel).await?;

        // Give worker time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(())
    }
    
    /// List available checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<CheckpointInfo>> {
        let mut checkpoints = Vec::new();
        
        let mut entries = fs::read_dir(&self.checkpoint_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                
                // Parse step from directory name (e.g., "step-1000")
                if let Some(step) = name.strip_prefix("step-").and_then(|s| s.parse().ok()) {
                    // Load metadata if available
                    let metadata_path = path.join("metadata.json");
                    let metadata = if metadata_path.exists() {
                        let json = fs::read_to_string(&metadata_path).await?;
                        serde_json::from_str(&json).ok()
                    } else {
                        None
                    };
                    
                    checkpoints.push(CheckpointInfo {
                        step,
                        path,
                        metadata,
                    });
                }
            }
        }
        
        checkpoints.sort_by_key(|c| c.step);
        Ok(checkpoints)
    }
    
    /// Load specific checkpoint
    pub async fn load_checkpoint(&self, step: usize) -> Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join(format!("step-{}", step));
        if !checkpoint_path.exists() {
            anyhow::bail!("Checkpoint for step {} not found", step);
        }
        Ok(checkpoint_path)
    }
    
    /// Worker that processes checkpoint requests in background
    async fn checkpoint_worker(
        mut rx: mpsc::Receiver<CheckpointRequest>,
        model_path: PathBuf,
        checkpoint_dir: PathBuf,
        max_checkpoints: usize,
        _git_interval: usize,
        git_enabled: bool,
        branch_name: Option<String>,
    ) {
        while let Some(request) = rx.recv().await {
            // Skip sentinel values
            if request.step == usize::MAX {
                continue;
            }
            
            if let Err(e) = Self::process_checkpoint(
                &model_path,
                &checkpoint_dir,
                request,
                git_enabled,
                &branch_name,
            ).await {
                tracing::error!("Failed to process checkpoint: {}", e);
            }
            
            // Clean up old checkpoints
            if let Err(e) = Self::cleanup_old_checkpoints(&checkpoint_dir, max_checkpoints).await {
                tracing::warn!("Failed to clean up old checkpoints: {}", e);
            }
        }
    }
    
    /// Process a single checkpoint request
    async fn process_checkpoint(
        model_path: &Path,
        checkpoint_dir: &Path,
        request: CheckpointRequest,
        git_enabled: bool,
        branch_name: &Option<String>,
    ) -> Result<()> {
        let step_dir = checkpoint_dir.join(format!("step-{}", request.step));
        fs::create_dir_all(&step_dir).await?;
        
        // Save weights
        let _weights_file = match request.weights {
            WeightSnapshot::Memory { ref data, format } => {
                let filename = match format {
                    WeightFormat::SafeTensors => "model.safetensors",
                    WeightFormat::PyTorch => "pytorch_model.bin",
                    WeightFormat::AdapterBin => "adapter_model.bin",
                };
                let path = step_dir.join(filename);
                fs::write(&path, data).await?;
                path
            }
            
            WeightSnapshot::FilePath { ref source, format } => {
                let filename = match format {
                    WeightFormat::SafeTensors => "model.safetensors",
                    WeightFormat::PyTorch => "pytorch_model.bin",
                    WeightFormat::AdapterBin => "adapter_model.bin",
                };
                let dest = step_dir.join(filename);
                
                // Try copy-on-write first (Linux only)
                #[cfg(target_os = "linux")]
                {
                    use std::process::Command;
                    let result = Command::new("cp")
                        .args(&["--reflink=auto", source.to_str().unwrap(), dest.to_str().unwrap()])
                        .output();
                    
                    if result.is_err() || !result.unwrap().status.success() {
                        fs::copy(source, &dest).await?;
                    }
                }
                
                #[cfg(not(target_os = "linux"))]
                {
                    fs::copy(source, &dest).await?;
                }
                
                dest
            }
            
            WeightSnapshot::Diff { base_step, ref changes } => {
                // Save diff file
                let diff_path = step_dir.join(format!("diff_from_{}.bin", base_step));
                fs::write(&diff_path, changes).await?;
                diff_path
            }
        };
        
        // Save metadata
        let metadata = CheckpointMetadata {
            step: request.step,
            epoch: request.epoch,
            timestamp: request.timestamp,
            metrics: request.metrics,
            format: match &request.weights {
                WeightSnapshot::Memory { format, .. } => *format,
                WeightSnapshot::FilePath { format, .. } => *format,
                WeightSnapshot::Diff { .. } => WeightFormat::SafeTensors,
            },
            parent_checkpoint: None,
        };
        
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(step_dir.join("metadata.json"), metadata_json).await?;
        
        // Optionally commit to Git branch
        if git_enabled && request.commit_to_git {
            if let Err(e) = Self::git_commit_checkpoint(model_path, checkpoint_dir, request.step, branch_name).await {
                tracing::warn!("Failed to commit checkpoint to Git: {}", e);
            }
        }
        
        tracing::info!(
            "Checkpoint saved: step {} at {}",
            request.step,
            step_dir.display()
        );
        
        Ok(())
    }
    
    /// Commit checkpoint to Git branch
    async fn git_commit_checkpoint(
        model_path: &Path,
        _checkpoint_dir: &Path,
        step: usize,
        branch_name: &Option<String>,
    ) -> Result<()> {
        // Open repository
        let repo = get_repository(model_path)?;
        
        // If we have a specific branch, ensure we're on it
        if let Some(branch) = branch_name {
            // Check if we're in a worktree (adapter training)
            let head = repo.head()?;
            let current_branch = head.shorthand().unwrap_or("");
            
            if !current_branch.contains(branch) {
                // Switch to the branch if not already on it
                let branch_ref = format!("refs/heads/{}", branch);
                repo.set_head(&branch_ref)?;
                repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            }
        }
        
        // Add checkpoint files to index
        let checkpoint_path = format!(".checkpoints/step-{}", step);
        let mut index = repo.index()?;
        index.add_path(Path::new(&checkpoint_path))?;
        index.write()?;
        
        // Create commit
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = create_hyprstream_signature()?;
        
        let parent_commit = repo.head()?.peel_to_commit()?;
        let message = if branch_name.is_some() {
            format!("Checkpoint at step {} (branch: {})", step, branch_name.as_ref().unwrap())
        } else {
            format!("Checkpoint at step {}", step)
        };
        
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            &message,
            &tree,
            &[&parent_commit],
        )?;
        
        // For major checkpoints, create human-friendly tags
        if step % 50000 == 0 {
            let tag_name = if let Some(branch) = branch_name {
                format!("checkpoint-{}-step-{}", branch, step)
            } else {
                format!("checkpoint-step-{}", step)
            };
            
            // Use BranchManager to create the tag if available
            if let Ok(branch_mgr) = BranchManager::new(model_path) {
                let _ = branch_mgr.create_tag(&tag_name, "HEAD");
            } else {
                // Fallback to direct tag creation
                repo.tag_lightweight(
                    &tag_name,
                    &repo.head()?.peel_to_commit()?.into_object(),
                    false,
                )?;
            }
        }
        
        Ok(())
    }
    
    /// Clean up old checkpoints, keeping only the most recent N
    async fn cleanup_old_checkpoints(checkpoint_dir: &Path, max_checkpoints: usize) -> Result<()> {
        let checkpoints = Self::list_checkpoint_dirs(checkpoint_dir).await?;
        
        if checkpoints.len() <= max_checkpoints {
            return Ok(());
        }
        
        // Sort by step number and remove oldest
        let mut checkpoints = checkpoints;
        checkpoints.sort_by_key(|(step, _)| *step);
        
        let to_remove = checkpoints.len() - max_checkpoints;
        for (_, path) in checkpoints.into_iter().take(to_remove) {
            if let Err(e) = fs::remove_dir_all(path).await {
                tracing::warn!("Failed to remove old checkpoint: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// List checkpoint directories
    async fn list_checkpoint_dirs(checkpoint_dir: &Path) -> Result<Vec<(usize, PathBuf)>> {
        let mut dirs = Vec::new();
        
        let mut entries = fs::read_dir(checkpoint_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Some(step) = name.strip_prefix("step-").and_then(|s| s.parse().ok()) {
                        dirs.push((step, path));
                    }
                }
            }
        }
        
        Ok(dirs)
    }
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub max_checkpoints: usize,
    pub git_commit_interval: usize,
    pub queue_size: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints: 5,
            git_commit_interval: 10000,
            queue_size: 10,
        }
    }
}

/// Information about a checkpoint
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub step: usize,
    pub path: PathBuf,
    pub metadata: Option<CheckpointMetadata>,
}

impl Drop for CheckpointManager {
    fn drop(&mut self) {
        // Abort the worker thread when manager is dropped
        if let Some(handle) = self.worker_handle.take() {
            handle.abort();
        }
    }
}