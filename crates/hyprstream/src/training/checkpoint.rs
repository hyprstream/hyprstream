//! Async checkpoint management for training
//!
//! Provides non-blocking checkpoint saving during training,
//! with commits to model branches for version control.
//!
//! Two modes of operation:
//! - **Direct**: Uses `PathBuf` and `tokio::fs` (when `fs` field is None).
//! - **FsOps**: Uses `WorktreeClient` for worktree-scoped, path-contained access.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::services::{WorktreeClient, RepositoryClient};

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
    /// Repository client for git operations (None = git disabled)
    repo_client: Option<RepositoryClient>,
    #[allow(dead_code)]
    branch_name: Option<String>, // Track which branch we're on
    /// Target adapter to update on each checkpoint (e.g., "01_coding")
    /// When set, checkpoints will also update adapters/{target_adapter}.safetensors
    target_adapter: Option<String>,
    /// WorktreeClient for worktree-scoped file operations (None = direct access)
    fs: Option<WorktreeClient>,
}

impl CheckpointManager {
    /// Create new checkpoint manager (no git operations)
    pub fn new(model_path: PathBuf) -> Result<Self> {
        Self::with_config(model_path, CheckpointConfig::default(), None, None)
    }

    /// Create checkpoint manager with a repository client for git operations
    pub fn with_repo_client(
        model_path: PathBuf,
        repo_client: RepositoryClient,
    ) -> Result<Self> {
        Self::with_config(
            model_path,
            CheckpointConfig::default(),
            None,
            Some(repo_client),
        )
    }

    /// Create checkpoint manager for an adapter branch
    pub fn for_adapter(
        model_path: PathBuf,
        branch_name: String,
        repo_client: Option<RepositoryClient>,
    ) -> Result<Self> {
        Self::with_config(
            model_path,
            CheckpointConfig::default(),
            Some(branch_name),
            repo_client,
        )
    }

    /// Create with custom configuration
    pub fn with_config(
        model_path: PathBuf,
        config: CheckpointConfig,
        branch_name: Option<String>,
        repo_client: Option<RepositoryClient>,
    ) -> Result<Self> {
        let checkpoint_dir = model_path.join(".checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;

        // Derive FsOps from repo_client if available
        let fs: Option<WorktreeClient> = repo_client.as_ref().and_then(|client| {
            let branch = branch_name.as_deref().unwrap_or("main");
            Some(client.worktree(branch))
        });

        let (tx, rx) = mpsc::channel(config.queue_size);

        // Spawn checkpoint worker thread
        let worker_handle = {
            let max_checkpoints = config.max_checkpoints;
            let git_interval = config.git_commit_interval;
            let branch = branch_name.clone();
            let client = repo_client.clone();
            let worker_fs = fs.clone();
            let worker_checkpoint_dir = checkpoint_dir.clone();

            tokio::spawn(async move {
                Self::checkpoint_worker(
                    rx,
                    worker_checkpoint_dir,
                    max_checkpoints,
                    git_interval,
                    client,
                    branch,
                    worker_fs,
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
            repo_client,
            branch_name,
            target_adapter: None,
            fs,
        })
    }

    /// Set target adapter to update on each checkpoint
    ///
    /// When set, each checkpoint will also copy weights to
    /// `adapters/{target_adapter}.safetensors` for live inference.
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

        if let Some(fs) = &self.fs {
            // FsOps path: use relative paths within worktree
            match weights {
                WeightSnapshot::Memory {
                    ref data,
                    format: _,
                } => {
                    fs.write_file(".checkpoints/checkpoint.safetensors", data).await?;
                }
                WeightSnapshot::FilePath {
                    ref source,
                    format: _,
                } => {
                    // source is an absolute path, need to handle this:
                    // If source is within worktree, compute relative path for copy.
                    // Otherwise fall back to reading the source and writing via FsOps.
                    if let Ok(rel) = source.strip_prefix(&self.model_path) {
                        let rel_str = rel.to_string_lossy();
                        fs.copy(&rel_str, ".checkpoints/checkpoint.safetensors").await?;
                    } else {
                        // Source outside worktree - read it directly and write via FsOps
                        let data = tokio::fs::read(source).await?;
                        fs.write_file(".checkpoints/checkpoint.safetensors", &data).await?;
                    }
                }
                WeightSnapshot::Diff {
                    base_step,
                    ref changes,
                } => {
                    let rel_diff = format!(".checkpoints/diff_from_{base_step}_to_{step}.bin");
                    fs.write_file(&rel_diff, changes).await?;
                    return Ok(self.checkpoint_dir.join(format!("diff_from_{base_step}_to_{step}.bin")));
                }
            }

            // Save metadata if provided
            if let Some(metrics) = metadata {
                let ckpt_meta = CheckpointMetadata {
                    step,
                    epoch: None,
                    timestamp: Utc::now(),
                    metrics: Some(metrics),
                    format: WeightFormat::SafeTensors,
                    parent_checkpoint: None,
                };
                let metadata_json = serde_json::to_string_pretty(&ckpt_meta)?;
                fs.write_file(".checkpoints/checkpoint.json", metadata_json.as_bytes()).await?;
            }

            // Update target adapter if configured
            if let Some(ref adapter_name) = self.target_adapter {
                fs.mkdir("adapters", true).await?;
                let rel_adapter = format!("adapters/{adapter_name}.safetensors");
                fs.copy(".checkpoints/checkpoint.safetensors", &rel_adapter).await?;
                tracing::info!("Updated target adapter via FsOps: {}", rel_adapter);
            }
        } else {
            // Direct path (original behavior)
            match weights {
                WeightSnapshot::Memory {
                    ref data,
                    format: _,
                } => {
                    fs::write(&checkpoint_path, data).await?;
                }
                WeightSnapshot::FilePath {
                    ref source,
                    format: _,
                } => {
                    // Validate source is a regular file (not a symlink to an external location)
                    let metadata = std::fs::symlink_metadata(source)
                        .map_err(|e| anyhow!("Cannot stat source file {:?}: {}", source, e))?;
                    if metadata.file_type().is_symlink() {
                        return Err(anyhow!("Source path {:?} is a symlink; refusing to copy", source));
                    }
                    crate::git::ops::cow_copy(source, &checkpoint_path)?;
                }
                WeightSnapshot::Diff {
                    base_step,
                    ref changes,
                } => {
                    let diff_path = self
                        .checkpoint_dir
                        .join(format!("diff_from_{base_step}_to_{step}.bin"));
                    fs::write(&diff_path, changes).await?;
                    return Ok(diff_path);
                }
            }

            // Save metadata if provided
            if let Some(metrics) = metadata {
                let ckpt_meta = CheckpointMetadata {
                    step,
                    epoch: None,
                    timestamp: Utc::now(),
                    metrics: Some(metrics),
                    format: WeightFormat::SafeTensors,
                    parent_checkpoint: None,
                };
                let metadata_path = checkpoint_path.with_extension("json");
                let metadata_json = serde_json::to_string_pretty(&ckpt_meta)?;
                fs::write(&metadata_path, metadata_json).await?;
            }

            // Update target adapter if configured
            if let Some(ref adapter_name) = self.target_adapter {
                self.update_target_adapter(&checkpoint_path, adapter_name)
                    .await?;
            }
        }

        tracing::info!(
            "Checkpoint written to filesystem: {}",
            checkpoint_path.display()
        );
        Ok(checkpoint_path)
    }

    /// Update the target adapter file with checkpoint weights (direct path only)
    async fn update_target_adapter(
        &self,
        checkpoint_path: &Path,
        adapter_name: &str,
    ) -> Result<()> {
        let adapters_dir = self.model_path.join("adapters");
        fs::create_dir_all(&adapters_dir).await?;

        let adapter_path = adapters_dir.join(format!("{adapter_name}.safetensors"));
        crate::git::ops::cow_copy(checkpoint_path, &adapter_path)?;

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
        let repo_client = self
            .repo_client
            .as_ref()
            .ok_or_else(|| anyhow!("Git is not enabled (no RepositoryClient configured)"))?;

        // Switch branch if specified
        if let Some(ref branch_name) = branch {
            // create_branch may fail if it already exists — that's fine
            let _ = repo_client.create_branch(branch_name, "").await;
            repo_client
                .checkout(branch_name, false)
                .await
                .map_err(|e| anyhow!("Failed to checkout branch '{}': {}", branch_name, e))?;
        }

        // Stage checkpoint files
        let relative_path = checkpoint_path
            .strip_prefix(&self.model_path)
            .unwrap_or(checkpoint_path);
        let mut files_to_stage: Vec<&str> = vec![relative_path.to_str().unwrap_or("")];

        let metadata_path = checkpoint_path.with_extension("json");
        let relative_metadata;

        let meta_exists = if let Some(fs) = &self.fs {
            let rel = metadata_path
                .strip_prefix(&self.model_path)
                .unwrap_or(&metadata_path)
                .to_string_lossy()
                .to_string();
            fs.stat(&rel).await.map(|s| s.exists).unwrap_or(false)
        } else {
            metadata_path.exists()
        };

        if meta_exists {
            relative_metadata = metadata_path
                .strip_prefix(&self.model_path)
                .unwrap_or(&metadata_path)
                .to_string_lossy()
                .to_string();
            files_to_stage.push(&relative_metadata);
        }

        let files_owned: Vec<String> = files_to_stage.iter().map(|s| s.to_string()).collect();
        repo_client
            .stage_files(&files_owned)
            .await
            .map_err(|e| anyhow!("Failed to stage files: {}", e))?;

        let commit_message = message.unwrap_or_else(|| {
            format!(
                "Training checkpoint: {}",
                checkpoint_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
            )
        });

        let oid = repo_client
            .commit(&commit_message, "", "")
            .await
            .map_err(|e| anyhow!("Failed to commit: {}", e))?;

        tracing::info!("Checkpoint committed to Git: {}", oid);
        Ok(oid)
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
    pub async fn shutdown(mut self) -> Result<()> {
        // Drop the sender first to prevent new requests, then let worker drain
        drop(std::mem::replace(
            &mut self.checkpoint_tx,
            mpsc::channel(1).0,
        ));

        // Wait for worker to complete
        if let Some(handle) = self.worker_handle.take() {
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
    pub fn has_pending(&self) -> bool {
        !self.checkpoint_tx.is_closed()
    }

    /// Get current checkpoint info (single file, git handles history)
    pub async fn get_checkpoint(&self) -> Result<Option<CheckpointInfo>> {
        if let Some(fs) = &self.fs {
            if !fs.stat(".checkpoints/checkpoint.json").await.map(|s| s.exists).unwrap_or(false) {
                return Ok(None);
            }
            let json = String::from_utf8(fs.read_file(".checkpoints/checkpoint.json").await?.data)?;
            let metadata: CheckpointMetadata = serde_json::from_str(&json)?;
            Ok(Some(CheckpointInfo {
                step: metadata.step,
                path: self.checkpoint_dir.clone(),
                metadata: Some(metadata),
            }))
        } else {
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
    }

    /// Load current checkpoint weights path
    pub async fn load_checkpoint(&self) -> Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join("checkpoint.safetensors");

        if let Some(fs) = &self.fs {
            if !fs.stat(".checkpoints/checkpoint.safetensors").await.map(|s| s.exists).unwrap_or(false) {
                anyhow::bail!("No checkpoint found at {:?}", checkpoint_path);
            }
        } else if !checkpoint_path.exists() {
            anyhow::bail!("No checkpoint found at {:?}", checkpoint_path);
        }

        Ok(checkpoint_path)
    }

    /// Worker that processes checkpoint requests in background
    async fn checkpoint_worker(
        mut rx: mpsc::Receiver<CheckpointRequest>,
        checkpoint_dir: PathBuf,
        _max_checkpoints: usize,
        _git_interval: usize,
        repo_client: Option<RepositoryClient>,
        branch_name: Option<String>,
        fs: Option<WorktreeClient>,
    ) {
        while let Some(request) = rx.recv().await {
            // Skip sentinel values
            if request.step == usize::MAX {
                continue;
            }

            if let Err(e) = Self::process_checkpoint(
                &checkpoint_dir,
                request,
                repo_client.as_ref(),
                &branch_name,
                fs.as_ref(),
            )
            .await
            {
                tracing::error!("Failed to process checkpoint: {}", e);
            }
        }
    }

    /// Process a single checkpoint request
    async fn process_checkpoint(
        checkpoint_dir: &Path,
        request: CheckpointRequest,
        repo_client: Option<&RepositoryClient>,
        branch_name: &Option<String>,
        fs: Option<&WorktreeClient>,
    ) -> Result<()> {
        if let Some(fs) = fs {
            // FsOps path
            fs.mkdir(".checkpoints", true).await?;

            // Save weights
            match request.weights {
                WeightSnapshot::Memory { ref data, format } => {
                    let filename = match format {
                        WeightFormat::SafeTensors => "checkpoint.safetensors",
                        WeightFormat::PyTorch => "checkpoint.bin",
                        WeightFormat::AdapterBin => "checkpoint_adapter.bin",
                    };
                    let rel_path = format!(".checkpoints/{filename}");
                    fs.write_file(&rel_path, data).await?;
                }
                WeightSnapshot::FilePath { ref source, format } => {
                    let filename = match format {
                        WeightFormat::SafeTensors => "checkpoint.safetensors",
                        WeightFormat::PyTorch => "checkpoint.bin",
                        WeightFormat::AdapterBin => "checkpoint_adapter.bin",
                    };
                    let rel_dest = format!(".checkpoints/{filename}");
                    // Source may be outside worktree; read and write if so
                    let data = tokio::fs::read(source).await?;
                    fs.write_file(&rel_dest, &data).await?;
                }
                WeightSnapshot::Diff {
                    base_step: _,
                    ref changes,
                } => {
                    fs.write_file(".checkpoints/checkpoint_diff.bin", changes).await?;
                }
            }

            // Save metadata
            let metadata = CheckpointMetadata {
                step: request.step,
                epoch: request.epoch,
                timestamp: request.timestamp,
                metrics: request.metrics,
                format: match &request.weights {
                    WeightSnapshot::Memory { format, .. }
                    | WeightSnapshot::FilePath { format, .. } => *format,
                    WeightSnapshot::Diff { .. } => WeightFormat::SafeTensors,
                },
                parent_checkpoint: None,
            };
            let metadata_json = serde_json::to_string_pretty(&metadata)?;
            fs.write_file(".checkpoints/checkpoint.json", metadata_json.as_bytes()).await?;
        } else {
            // Direct path (original behavior)
            tokio::fs::create_dir_all(checkpoint_dir).await?;

            let _weights_file = match request.weights {
                WeightSnapshot::Memory { ref data, format } => {
                    let filename = match format {
                        WeightFormat::SafeTensors => "checkpoint.safetensors",
                        WeightFormat::PyTorch => "checkpoint.bin",
                        WeightFormat::AdapterBin => "checkpoint_adapter.bin",
                    };
                    let path = checkpoint_dir.join(filename);
                    tokio::fs::write(&path, data).await?;
                    path
                }
                WeightSnapshot::FilePath { ref source, format } => {
                    let filename = match format {
                        WeightFormat::SafeTensors => "checkpoint.safetensors",
                        WeightFormat::PyTorch => "checkpoint.bin",
                        WeightFormat::AdapterBin => "checkpoint_adapter.bin",
                    };
                    let dest = checkpoint_dir.join(filename);
                    crate::git::ops::cow_copy(source, &dest)?;
                    dest
                }
                WeightSnapshot::Diff {
                    base_step: _,
                    ref changes,
                } => {
                    let diff_path = checkpoint_dir.join("checkpoint_diff.bin");
                    tokio::fs::write(&diff_path, changes).await?;
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
                    WeightSnapshot::Memory { format, .. }
                    | WeightSnapshot::FilePath { format, .. } => *format,
                    WeightSnapshot::Diff { .. } => WeightFormat::SafeTensors,
                },
                parent_checkpoint: None,
            };
            let metadata_json = serde_json::to_string_pretty(&metadata)?;
            tokio::fs::write(checkpoint_dir.join("checkpoint.json"), metadata_json).await?;
        }

        // Optionally commit to Git via RepositoryClient
        if let Some(client) = repo_client {
            if request.commit_to_git {
                if let Err(e) =
                    Self::git_commit_checkpoint(client, request.step, branch_name).await
                {
                    tracing::warn!("Failed to commit checkpoint to Git: {}", e);
                }
            }
        }

        tracing::info!(
            "Checkpoint saved: step {} at {}",
            request.step,
            checkpoint_dir.display()
        );

        Ok(())
    }

    /// Commit checkpoint to Git branch via RepositoryClient
    async fn git_commit_checkpoint(
        repo_client: &RepositoryClient,
        step: usize,
        branch_name: &Option<String>,
    ) -> Result<()> {
        // Ensure we're on the right branch
        if let Some(branch) = branch_name {
            // create_branch may fail if it already exists — that's fine
            let _ = repo_client.create_branch(branch, "").await;
            repo_client
                .checkout(branch, false)
                .await
                .map_err(|e| anyhow!("Failed to checkout branch '{}': {}", branch, e))?;
        }

        // Stage checkpoint files
        repo_client
            .stage_files(&[
                ".checkpoints/checkpoint.safetensors".to_string(),
                ".checkpoints/checkpoint.json".to_string(),
            ])
            .await
            .map_err(|e| anyhow!("Failed to stage checkpoint files: {}", e))?;

        // Commit
        let message = if let Some(name) = branch_name.as_ref() {
            format!("Training checkpoint step {step} (branch: {name})")
        } else {
            format!("Training checkpoint step {step}")
        };

        repo_client
            .commit(&message, "", "")
            .await
            .map_err(|e| anyhow!("Failed to commit: {}", e))?;

        // For major checkpoints, create human-friendly tags
        if step.is_multiple_of(50000) {
            let tag_name = if let Some(branch) = branch_name {
                format!("checkpoint-{branch}-step-{step}")
            } else {
                format!("checkpoint-step-{step}")
            };

            let _ = repo_client.create_tag(&tag_name, "").await;
        }

        Ok(())
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
        // Check if shutdown was called properly
        if let Some(handle) = self.worker_handle.take() {
            tracing::warn!(
                "CheckpointManager dropped without calling shutdown()! \
                 This may cause checkpoint corruption if writes were in progress. \
                 Always call `manager.shutdown().await` before dropping."
            );
            handle.abort();
        }
    }
}
