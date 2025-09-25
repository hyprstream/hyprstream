//! Git-based model source for cloning and managing models

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, debug, warn};

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

        // Process LFS pointer files after clone
        // XET is a special case of LFS - try XET first for all LFS pointers,
        // then use Git LFS for any that XET couldn't handle

        // Step 1: Try XET processing if available (handles XET-enabled LFS pointers)
        let mut xet_processed = 0;
        if let Some(xet_storage) = &self.xet_storage {
            if show_progress {
                info!("Attempting to process LFS pointers with XET");
            }

            // Try batch processing first for performance
            match xet_storage.batch_process_lfs_files(&model_path).await {
                Ok(processed_files) => {
                    xet_processed = processed_files.len();
                    if xet_processed > 0 && show_progress {
                        info!("Processed {} LFS files via XET", xet_processed);
                    }
                }
                Err(e) => {
                    // Log but don't fail - we'll try Git LFS next
                    debug!("XET processing not applicable or failed: {}", e);
                }
            }
        }

        // Step 2: Process remaining LFS pointers with Git LFS
        // This handles all LFS pointers that XET didn't process
        // (either because XET isn't available or they're standard Git LFS pointers)

        match self.process_git_lfs_files(&model_path).await {
            Ok(lfs_count) => {
                if lfs_count > 0 && show_progress {
                    info!("Processed {} LFS files", lfs_count);
                }

                let total_processed = xet_processed + lfs_count;
                if show_progress && total_processed > 0 {
                    info!("Total LFS pointers processed: {} (XET: {}, Git LFS: {})",
                          total_processed, xet_processed, lfs_count);
                }
            }
            Err(e) => {
                // Don't fail the clone, but warn about incomplete files
                warn!("LFS processing failed: {} - model files may be incomplete", e);

                // Check if we have unprocessed pointers
                if xet_processed == 0 {
                    if let Ok(has_lfs) = self.check_for_lfs_pointers(&model_path).await {
                        if has_lfs {
                            warn!("Repository contains LFS pointer files that couldn't be processed. Model files will be incomplete.");
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

    /// Process Git LFS files in a repository
    async fn process_git_lfs_files(&self, repo_path: &Path) -> Result<usize> {
        use tokio::process::Command;

        // First, check which LFS files are still pointers (not yet smudged)
        // git lfs ls-files shows all tracked files, but we need to check which are still pointers
        let ls_files_output = Command::new("git")
            .arg("lfs")
            .arg("ls-files")
            .arg("-n")  // Show just names
            .current_dir(repo_path)
            .output()
            .await?;

        if !ls_files_output.status.success() {
            // git lfs might not be initialized for this repo, which is ok
            return Ok(0);
        }

        let ls_files_str = String::from_utf8_lossy(&ls_files_output.stdout);
        let lfs_files: Vec<_> = ls_files_str.lines().collect();

        if lfs_files.is_empty() {
            return Ok(0);
        }

        // Count how many are still pointers (not yet processed)
        let mut unprocessed_count = 0;
        for file in &lfs_files {
            let file_path = repo_path.join(file);
            if file_path.exists() {
                // Check if it's still a pointer
                if let Ok(contents) = tokio::fs::read_to_string(&file_path).await {
                    if contents.starts_with("version https://git-lfs.github.com/spec/v1") {
                        unprocessed_count += 1;
                    }
                }
            }
        }

        if unprocessed_count == 0 {
            // All LFS files already processed (probably by XET)
            return Ok(0);
        }

        debug!("Found {} unprocessed LFS pointers, fetching via Git LFS", unprocessed_count);

        // Fetch LFS objects
        let fetch_result = Command::new("git")
            .arg("lfs")
            .arg("fetch")
            .arg("--all")
            .current_dir(repo_path)
            .status()
            .await?;

        if !fetch_result.success() {
            bail!("git lfs fetch failed with exit code {:?}", fetch_result.code());
        }

        // Checkout LFS objects
        let checkout_result = Command::new("git")
            .arg("lfs")
            .arg("checkout")
            .current_dir(repo_path)
            .status()
            .await?;

        if !checkout_result.success() {
            bail!("git lfs checkout failed with exit code {:?}", checkout_result.code());
        }

        Ok(unprocessed_count)
    }

    /// Check if a repository has LFS pointer files
    async fn check_for_lfs_pointers(&self, repo_path: &Path) -> Result<bool> {
        use tokio::process::Command;

        // Use git lfs ls-files to check for LFS tracked files
        let output = Command::new("git")
            .arg("lfs")
            .arg("ls-files")
            .current_dir(repo_path)
            .output()
            .await?;

        if output.status.success() {
            let ls_files_str = String::from_utf8_lossy(&output.stdout);
            Ok(!ls_files_str.trim().is_empty())
        } else {
            // If git lfs ls-files fails, check for .gitattributes with LFS filters
            let gitattributes = repo_path.join(".gitattributes");
            if gitattributes.exists() {
                let content = tokio::fs::read_to_string(&gitattributes).await?;
                Ok(content.contains("filter=lfs"))
            } else {
                Ok(false)
            }
        }
    }
}