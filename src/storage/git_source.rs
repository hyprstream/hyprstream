//! Git-based model source for cloning and managing models

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, debug, warn};
use uuid::Uuid;

use super::ModelId;
use super::xet_native::XetNativeStorage;
use crate::git::{GitManager, GitConfig, GitProgress, GitProgressInfo, GitError, CloneOptions};

/// Progress reporter implementation for git operations
pub struct ModelCloneProgress {
    model_name: String,
    show_progress: bool,
}

impl GitProgress for ModelCloneProgress {
    fn on_progress(&self, progress: &GitProgressInfo) {
        if !self.show_progress {
            return;
        }

        if progress.total > 0 {
            let percentage = (progress.current * 100) / progress.total;
            debug!("Cloning {}: {}% ({}/{}) - {} bytes received",
                   self.model_name, percentage, progress.current, progress.total, progress.bytes_received);
        }
    }

    fn on_error(&self, error: &GitError) {
        warn!("Git operation failed for {}: {} (class: {:?}, code: {:?})",
              self.model_name, error.message, error.class, error.code);

        if error.retry_suggested {
            info!("Error may be retryable for {}", self.model_name);
        }
    }

    fn is_cancelled(&self) -> bool {
        false // For now, no cancellation support
    }
}

/// Git-based model source for cloning models
pub struct GitModelSource {
    cache_dir: PathBuf,
    /// Optional XET storage for LFS processing
    xet_storage: Option<Arc<XetNativeStorage>>,
    /// Git manager for advanced operations
    git_manager: Arc<GitManager>,
}

impl GitModelSource {
    /// Create a new Git model source
    pub fn new(cache_dir: PathBuf) -> Self {
        let git_config = GitConfig {
            prefer_shallow: true,
            shallow_depth: Some(1),
            ..Default::default()
        };

        Self {
            cache_dir,
            xet_storage: None,
            git_manager: Arc::new(GitManager::new(git_config)),
        }
    }

    /// Create a new Git model source with custom Git configuration
    pub fn new_with_config(cache_dir: PathBuf, git_config: GitConfig) -> Self {
        Self {
            cache_dir,
            xet_storage: None,
            git_manager: Arc::new(GitManager::new(git_config)),
        }
    }

    /// Create a new Git model source with XET storage for LFS processing
    pub fn new_with_xet(cache_dir: PathBuf, xet_storage: Arc<XetNativeStorage>) -> Self {
        let git_config = GitConfig {
            prefer_shallow: true,
            shallow_depth: Some(1),
            ..Default::default()
        };

        Self {
            cache_dir,
            xet_storage: Some(xet_storage),
            git_manager: Arc::new(GitManager::new(git_config)),
        }
    }

    /// Create with both XET storage and custom git config
    pub fn new_with_xet_and_config(
        cache_dir: PathBuf,
        xet_storage: Arc<XetNativeStorage>,
        git_config: GitConfig,
    ) -> Self {
        Self {
            cache_dir,
            xet_storage: Some(xet_storage),
            git_manager: Arc::new(GitManager::new(git_config)),
        }
    }

    /// Clone a model from a Git repository
    pub async fn clone_model(&self, repo_url: &str) -> Result<(ModelId, PathBuf)> {
        self.clone_model_with_progress(repo_url, true).await
    }

    /// Clone a model with optional progress reporting
    pub async fn clone_model_with_progress(
        &self,
        repo_url: &str,
        show_progress: bool,
    ) -> Result<(ModelId, PathBuf)> {
        // Extract model name from URL
        let model_name = repo_url.split('/').last()
            .unwrap_or("unknown")
            .trim_end_matches(".git");

        // Generate a new UUID for this model
        let model_id = ModelId::new();

        // Use model name as directory instead of UUID
        let model_path = self.cache_dir.join(model_name);

        // Check if model already exists
        if model_path.exists() {
            info!("Model {} already exists at {}", model_name, model_path.display());
            return Ok((model_id, model_path));
        }

        if show_progress {
            info!("Cloning model from {} to {}", repo_url, model_path.display());
        }

        // Create progress reporter
        let progress: Option<Arc<dyn GitProgress>> = if show_progress {
            Some(Arc::new(ModelCloneProgress {
                model_name: model_name.to_string(),
                show_progress,
            }))
        } else {
            None
        };

        // Configure clone options for shallow clone
        let clone_options = CloneOptions {
            shallow: true,
            depth: Some(1),
            ..Default::default()
        };

        // Clone repository directly (retry logic is handled internally by GitManager)
        let repository = self.git_manager.clone_repository(
            repo_url,
            &model_path,
            clone_options,
            progress
        ).await.map_err(|e| {
            warn!("Failed to clone {} after retries: {}", repo_url, e);
            e
        })?;

        // Process LFS files if XET storage is available
        if let Some(xet_storage) = &self.xet_storage {
            if show_progress {
                info!("Processing LFS files in parallel after clone: {}", model_path.display());
            }

            // Use the enhanced batch processing method for better performance
            match xet_storage.batch_process_lfs_files(&model_path).await {
                Ok(processed_files) => {
                    if !processed_files.is_empty() && show_progress {
                        info!("Successfully processed {} LFS files in parallel", processed_files.len());
                    }
                }
                Err(e) => {
                    // Log warning but don't fail the clone operation
                    warn!("Failed to batch process LFS files after clone: {}", e);

                    // Fallback to sequential processing if parallel fails
                    if show_progress {
                        info!("Falling back to sequential LFS processing");
                    }
                    match xet_storage.process_worktree_lfs(&model_path).await {
                        Ok(processed_files) => {
                            if !processed_files.is_empty() && show_progress {
                                info!("Successfully processed {} LFS files sequentially", processed_files.len());
                            }
                        }
                        Err(fallback_e) => {
                            warn!("Both parallel and sequential LFS processing failed: {}", fallback_e);
                        }
                    }
                }
            }
        }

        Ok((model_id, model_path))
    }

    /// Clone a model at a specific ref
    pub async fn clone_ref(
        &self,
        repo_url: &str,
        ref_spec: &str,
    ) -> Result<(ModelId, PathBuf)> {
        // First clone the repository
        let (model_id, model_path) = self.clone_model(repo_url).await?;

        // If the model already existed, we still need to checkout the ref
        let ref_spec = ref_spec.to_string();
        let model_path_clone = model_path.clone();
        let git_manager = Arc::clone(&self.git_manager);

        tokio::task::spawn_blocking(move || {
            let repo = git_manager.get_repository(&model_path_clone)?;

            // Fetch to ensure we have the latest refs
            let mut remote = repo.find_remote("origin")?;
            remote.fetch(&[&ref_spec], None, None)?;

            // Then checkout the specific ref
            let obj = repo.revparse_single(&ref_spec)?;
            repo.checkout_tree(&obj, None)?;
            repo.set_head_detached(obj.id())?;
            Ok::<(), anyhow::Error>(())
        }).await??;

        Ok((model_id, model_path))
    }
}