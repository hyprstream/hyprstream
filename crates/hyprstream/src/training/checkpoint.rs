//! Async checkpoint management for training
//!
//! Provides non-blocking checkpoint saving during training,
//! with commits to model branches for version control.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use git2db::GitManager;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

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
    Memory { data: Vec<u8>, format: WeightFormat },

    /// Path to file that should be copied (for large models)
    FilePath {
        source: PathBuf,
        format: WeightFormat,
    },

    /// Incremental diff from previous checkpoint
    Diff { base_step: usize, changes: Vec<u8> },
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
    branch_name: Option<String>, // Track which branch we're on
    /// Target adapter to update on each checkpoint (e.g., "01_coding")
    /// When set, checkpoints will also update adapters/{target_adapter}.safetensors
    target_adapter: Option<String>,
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
    pub fn with_config(
        model_path: PathBuf,
        config: CheckpointConfig,
        branch_name: Option<String>,
    ) -> Result<Self> {
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
                )
                .await;
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
            target_adapter: None,
        })
    }

    /// Set target adapter to update on each checkpoint
    ///
    /// When set, each checkpoint will also copy weights to
    /// `adapters/{target_adapter}.safetensors` for live inference.
    ///
    /// # Example
    /// ```ignore
    /// let manager = CheckpointManager::new(model_path)?
    ///     .with_target_adapter("01_coding".to_owned());
    /// ```
    pub fn with_target_adapter(mut self, adapter_name: String) -> Self {
        self.target_adapter = Some(adapter_name);
        self
    }

    /// Get the target adapter name if set
    pub fn target_adapter(&self) -> Option<&str> {
        self.target_adapter.as_deref()
    }

    /// Request asynchronous checkpoint
    pub async fn checkpoint(&self, request: CheckpointRequest) -> Result<()> {
        self.checkpoint_tx
            .send(request)
            .await
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
            commit_to_git: step.is_multiple_of(10000), // Commit every 10k steps by default
        }
    }

    /// Write checkpoint to filesystem without Git commit
    /// Note: Uses single checkpoint.safetensors file - git handles versioning
    pub async fn write_checkpoint(
        &self,
        weights: WeightSnapshot,
        step: usize,
        metadata: Option<TrainingMetrics>,
    ) -> Result<PathBuf> {
        // Single checkpoint file - git tracks version history
        let checkpoint_path = self.checkpoint_dir.join("checkpoint.safetensors");

        // Write weights directly to filesystem
        match weights {
            WeightSnapshot::Memory {
                ref data,
                format: _,
            } => {
                // Direct write from memory
                fs::write(&checkpoint_path, data).await?;
            }
            WeightSnapshot::FilePath {
                ref source,
                format: _,
            } => {
                // Use copy-on-write if available
                #[cfg(target_os = "linux")]
                {
                    use std::process::Command;
                    let source_str = source.to_str()
                        .ok_or_else(|| anyhow!("source path is not valid UTF-8"))?;
                    let checkpoint_str = checkpoint_path.to_str()
                        .ok_or_else(|| anyhow!("checkpoint path is not valid UTF-8"))?;
                    let result = Command::new("cp")
                        .args(["--reflink=auto", source_str, checkpoint_str])
                        .output();

                    match result {
                        Ok(output) if output.status.success() => {}
                        _ => {
                            fs::copy(source, &checkpoint_path).await?;
                        }
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    fs::copy(source, &checkpoint_path).await?;
                }
            }
            WeightSnapshot::Diff {
                base_step,
                ref changes,
            } => {
                // Save diff file
                let diff_path = self
                    .checkpoint_dir
                    .join(format!("diff_from_{base_step}_to_{step}.bin"));
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

        // Update target adapter if configured
        if let Some(ref adapter_name) = self.target_adapter {
            self.update_target_adapter(&checkpoint_path, adapter_name)
                .await?;
        }

        tracing::info!(
            "Checkpoint written to filesystem: {}",
            checkpoint_path.display()
        );
        Ok(checkpoint_path)
    }

    /// Update the target adapter file with checkpoint weights
    async fn update_target_adapter(
        &self,
        checkpoint_path: &Path,
        adapter_name: &str,
    ) -> Result<()> {
        // Ensure adapters directory exists
        let adapters_dir = self.model_path.join("adapters");
        fs::create_dir_all(&adapters_dir).await?;

        let adapter_path = adapters_dir.join(format!("{adapter_name}.safetensors"));

        // Copy checkpoint to adapter path
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            let checkpoint_str = checkpoint_path.to_str()
                .ok_or_else(|| anyhow!("checkpoint path is not valid UTF-8"))?;
            let adapter_str = adapter_path.to_str()
                .ok_or_else(|| anyhow!("adapter path is not valid UTF-8"))?;
            let result = Command::new("cp")
                .args(["--reflink=auto", checkpoint_str, adapter_str])
                .output();

            match result {
                Ok(output) if output.status.success() => {}
                _ => {
                    fs::copy(checkpoint_path, &adapter_path).await?;
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            fs::copy(checkpoint_path, &adapter_path).await?;
        }

        tracing::info!(
            "Updated target adapter: {}",
            adapter_path.display()
        );
        Ok(())
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
        let repo = GitManager::global()
            .get_repository(&self.model_path)
            .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?
            .open()
            .map_err(|e| anyhow::anyhow!("Failed to open repository: {}", e))?;

        // Switch branch if specified
        if let Some(branch_name) = branch {
            let branch_ref = format!("refs/heads/{branch_name}");
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
        let relative_path = checkpoint_path
            .strip_prefix(&self.model_path)
            .unwrap_or(checkpoint_path);
        index.add_path(relative_path)?;

        // Also add metadata if it exists
        let metadata_path = checkpoint_path.with_extension("json");
        if metadata_path.exists() {
            let relative_metadata = metadata_path
                .strip_prefix(&self.model_path)
                .unwrap_or(&metadata_path);
            index.add_path(relative_metadata)?;
        }

        index.write()?;

        // Create commit
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = GitManager::global()
            .create_signature(None, None)
            .map_err(|e| anyhow::anyhow!("Failed to create signature: {}", e))?;
        let parent = repo.head()?.peel_to_commit()?;

        let commit_message = message.unwrap_or_else(|| {
            format!(
                "Training checkpoint: {}",
                checkpoint_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
            )
        });

        let oid = repo.commit(Some("HEAD"), &sig, &sig, &commit_message, &tree, &[&parent])?;

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

    /// Gracefully shutdown the checkpoint manager
    ///
    /// This method:
    /// 1. Flushes all pending checkpoints
    /// 2. Closes the channel to signal shutdown
    /// 3. Waits for the worker to complete
    ///
    /// **IMPORTANT**: Always call this method before dropping the manager
    /// to prevent checkpoint corruption. The Drop implementation will warn
    /// if shutdown() wasn't called.
    ///
    /// # Example
    /// ```ignore
    /// let manager = CheckpointManager::new(path)?;
    /// // ... use manager ...
    /// manager.shutdown().await?;  // Clean shutdown
    /// ```
    pub async fn shutdown(mut self) -> Result<()> {
        // Flush pending checkpoints first
        if let Err(e) = self.flush().await {
            tracing::warn!("Error flushing checkpoints during shutdown: {}", e);
        }

        // Drop the sender to signal the worker to stop
        // (The worker will exit when the channel closes after processing all items)
        drop(std::mem::replace(
            &mut self.checkpoint_tx,
            mpsc::channel(1).0, // Replace with a dummy sender
        ));

        // Wait for worker to complete
        if let Some(handle) = self.worker_handle.take() {
            // Give a reasonable timeout for the worker to finish
            match tokio::time::timeout(tokio::time::Duration::from_secs(30), handle).await {
                Ok(Ok(())) => {
                    tracing::debug!("CheckpointManager worker shut down cleanly");
                }
                Ok(Err(e)) => {
                    tracing::warn!("CheckpointManager worker panicked during shutdown: {}", e);
                }
                Err(_) => {
                    tracing::error!(
                        "CheckpointManager worker timed out during shutdown - \
                         checkpoint may be incomplete"
                    );
                }
            }
        }

        Ok(())
    }

    /// Check if there are pending checkpoints in the queue
    ///
    /// Note: This is an approximation since we can't peek into mpsc channels.
    /// Returns true if the channel has capacity remaining (meaning items might be queued).
    pub fn has_pending(&self) -> bool {
        // We can check if the channel is closed as a proxy
        // If it's closed, there are definitely no pending items
        !self.checkpoint_tx.is_closed()
    }

    /// Get current checkpoint info (single file, git handles history)
    pub async fn get_checkpoint(&self) -> Result<Option<CheckpointInfo>> {
        let metadata_path = self.checkpoint_dir.join("checkpoint.json");

        if !metadata_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&metadata_path).await?;
        let metadata: CheckpointMetadata = serde_json::from_str(&json)?;

        Ok(Some(CheckpointInfo {
            step: metadata.step,
            path: self.checkpoint_dir.clone(),
            metadata: Some(metadata),
        }))
    }

    /// Load current checkpoint weights path
    pub async fn load_checkpoint(&self) -> Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join("checkpoint.safetensors");
        if !checkpoint_path.exists() {
            anyhow::bail!("No checkpoint found at {:?}", checkpoint_path);
        }
        Ok(checkpoint_path)
    }

    /// Worker that processes checkpoint requests in background
    async fn checkpoint_worker(
        mut rx: mpsc::Receiver<CheckpointRequest>,
        model_path: PathBuf,
        checkpoint_dir: PathBuf,
        _max_checkpoints: usize, // No longer used - git handles versioning
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
            )
            .await
            {
                tracing::error!("Failed to process checkpoint: {}", e);
            }

            // Note: No cleanup needed - single file, git handles versioning
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
        // Use single checkpoint file - git handles versioning (no step-N dirs)
        fs::create_dir_all(checkpoint_dir).await?;

        // Save weights - single checkpoint file
        let _weights_file = match request.weights {
            WeightSnapshot::Memory { ref data, format } => {
                let filename = match format {
                    WeightFormat::SafeTensors => "checkpoint.safetensors",
                    WeightFormat::PyTorch => "checkpoint.bin",
                    WeightFormat::AdapterBin => "checkpoint_adapter.bin",
                };
                let path = checkpoint_dir.join(filename);
                fs::write(&path, data).await?;
                path
            }

            WeightSnapshot::FilePath { ref source, format } => {
                let filename = match format {
                    WeightFormat::SafeTensors => "checkpoint.safetensors",
                    WeightFormat::PyTorch => "checkpoint.bin",
                    WeightFormat::AdapterBin => "checkpoint_adapter.bin",
                };
                let dest = checkpoint_dir.join(filename);

                // Try copy-on-write first (Linux only)
                #[cfg(target_os = "linux")]
                {
                    use std::process::Command;
                    let source_str = source.to_str()
                        .ok_or_else(|| anyhow!("source path is not valid UTF-8"))?;
                    let dest_str = dest.to_str()
                        .ok_or_else(|| anyhow!("dest path is not valid UTF-8"))?;
                    let result = Command::new("cp")
                        .args(["--reflink=auto", source_str, dest_str])
                        .output();

                    match result {
                        Ok(output) if output.status.success() => {}
                        _ => {
                            fs::copy(source, &dest).await?;
                        }
                    }
                }

                #[cfg(not(target_os = "linux"))]
                {
                    fs::copy(source, &dest).await?;
                }

                dest
            }

            WeightSnapshot::Diff {
                base_step: _,
                ref changes,
            } => {
                // Diff files still use step since they need to reference base
                let diff_path = checkpoint_dir.join("checkpoint_diff.bin");
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
                WeightSnapshot::Memory { format, .. } | WeightSnapshot::FilePath { format, .. } => *format,
                WeightSnapshot::Diff { .. } => WeightFormat::SafeTensors,
            },
            parent_checkpoint: None,
        };

        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(checkpoint_dir.join("checkpoint.json"), metadata_json).await?;

        // Optionally commit to Git branch
        if git_enabled && request.commit_to_git {
            if let Err(e) =
                Self::git_commit_checkpoint(model_path, checkpoint_dir, request.step, branch_name)
                    .await
            {
                tracing::warn!("Failed to commit checkpoint to Git: {}", e);
            }
        }

        tracing::info!(
            "Checkpoint saved: step {} at {}",
            request.step,
            checkpoint_dir.display()
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
        let repo = GitManager::global()
            .get_repository(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to get repository: {}", e))?
            .open()
            .map_err(|e| anyhow::anyhow!("Failed to open repository: {}", e))?;

        // If we have a specific branch, ensure we're on it
        if let Some(branch) = branch_name {
            // Check if we're in a worktree (adapter training)
            let head = repo.head()?;
            let current_branch = head.shorthand().unwrap_or("");

            if !current_branch.contains(branch) {
                // Switch to the branch if not already on it
                let branch_ref = format!("refs/heads/{branch}");
                repo.set_head(&branch_ref)?;
                repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
            }
        }

        // Add checkpoint files to index (single checkpoint file, git handles versioning)
        let mut index = repo.index()?;
        index.add_path(Path::new(".checkpoints/checkpoint.safetensors"))?;
        index.add_path(Path::new(".checkpoints/checkpoint.json"))?;
        index.write()?;

        // Create commit
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = GitManager::global()
            .create_signature(None, None)
            .map_err(|e| anyhow::anyhow!("Failed to create signature: {}", e))?;

        let parent_commit = repo.head()?.peel_to_commit()?;
        let message = if let Some(name) = branch_name.as_ref() {
            format!("Training checkpoint step {step} (branch: {name})")
        } else {
            format!("Training checkpoint step {step}")
        };

        repo.commit(Some("HEAD"), &sig, &sig, &message, &tree, &[&parent_commit])?;

        // For major checkpoints, create human-friendly tags
        if step.is_multiple_of(50000) {
            let tag_name = if let Some(branch) = branch_name {
                format!("checkpoint-{branch}-step-{step}")
            } else {
                format!("checkpoint-step-{step}")
            };

            // Create checkpoint tag
            let _ = crate::git::helpers::create_tag(model_path, &tag_name);
        }

        Ok(())
    }

    // Note: cleanup_old_checkpoints and list_checkpoint_dirs removed
    // Git handles versioning - single checkpoint file, git tracks history
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
        // Check if shutdown was called properly
        if let Some(handle) = self.worker_handle.take() {
            // Worker handle still exists - shutdown() wasn't called!
            tracing::warn!(
                "CheckpointManager dropped without calling shutdown()! \
                 This may cause checkpoint corruption if writes were in progress. \
                 Always call `manager.shutdown().await` before dropping."
            );

            // Best-effort cleanup: abort the worker
            // This is not ideal but prevents resource leaks
            handle.abort();
        }
        // If worker_handle is None, shutdown() was called properly
    }
}
