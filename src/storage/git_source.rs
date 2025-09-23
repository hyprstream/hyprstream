//! Git-based model source for cloning and managing models

use anyhow::{Result, bail};
use git2::{Repository, build::RepoBuilder, FetchOptions};
use std::path::{Path, PathBuf};
use tracing::{info, debug};
use uuid::Uuid;

use super::ModelId;

/// Git-based model source for cloning models
pub struct GitModelSource {
    cache_dir: PathBuf,
}

impl GitModelSource {
    /// Create a new Git model source
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
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

        // Clone the repository
        let repo_url_clone = repo_url.to_string();
        let model_path_clone = model_path.clone();

        // Run git clone in blocking task
        tokio::task::spawn_blocking(move || {
            let mut builder = RepoBuilder::new();
            let mut fetch_opts = FetchOptions::new();

            if show_progress {
                // Set up progress callback
                let mut callbacks = git2::RemoteCallbacks::new();
                callbacks.transfer_progress(|stats| {
                    if stats.received_objects() == stats.total_objects() {
                        debug!("Resolving deltas {}/{}",
                              stats.indexed_deltas(),
                              stats.total_deltas());
                    } else if stats.total_objects() > 0 {
                        debug!("Receiving objects: {}% ({}/{})",
                              100 * stats.received_objects() / stats.total_objects(),
                              stats.received_objects(),
                              stats.total_objects());
                    }
                    true
                });
                fetch_opts.remote_callbacks(callbacks);
            }

            builder.fetch_options(fetch_opts);
            builder.clone(&repo_url_clone, &model_path_clone)
        }).await??;

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

        tokio::task::spawn_blocking(move || {
            let repo = Repository::open(&model_path_clone)?;

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