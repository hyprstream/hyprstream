//! High-level model operations shared between CLI and server
//!
//! This module provides common operations to reduce duplication

use anyhow::Result;
use git2::Repository;
use std::path::PathBuf;
use std::sync::Arc;
use super::{GitModelSource, ModelStorage, ModelId, paths::StoragePaths};
use super::xet_native::{XetNativeStorage, XetConfig};

/// Result of cloning a model
pub struct ClonedModel {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub model_name: String,
}

/// Clone a model from a git repository using proper git submodule workflow
///
/// This is the shared implementation used by both CLI and server
pub async fn clone_model(
    repo_url: &str,
    name: Option<&str>,
    git_ref: Option<&str>,
) -> Result<ClonedModel> {
    // Get storage paths
    let storage_paths = StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    // Determine model name: use provided name or extract from URL
    let model_name = if let Some(n) = name {
        n.to_string()
    } else {
        // Extract model name from URL
        let extracted = repo_url.split('/').last()
            .unwrap_or("")
            .trim_end_matches(".git");

        // If extraction yields empty string (e.g., gittorrent URLs), require explicit name
        if extracted.is_empty() {
            anyhow::bail!("Cannot derive model name from URL '{}'. Please provide a name using --name flag.", repo_url);
        }

        extracted.to_string()
    };

    // Load config to get git2db settings (includes auth tokens from env)
    let hypr_config = crate::config::HyprConfig::load().unwrap_or_default();

    // Create model storage with registry and config
    let model_storage = ModelStorage::create_with_config(
        models_dir.clone(),
        hypr_config.git2db.clone()
    ).await?;
    let registry = model_storage.registry();

    // Use proper git submodule workflow
    tracing::info!("Adding model {} as git submodule from {}", model_name, repo_url);
    let model_ref = registry.add_model(&model_name, repo_url).await?;

    // Get the model path (should exist after submodule add)
    let model_path = registry.get_model_path(&model_ref).await?;

    // If a specific git ref was requested, checkout that ref
    if let Some(ref_spec) = git_ref {
        tracing::info!("Checking out git ref '{}' for model {}", ref_spec, model_name);

        // Open the repository and checkout the ref
        let repo = Repository::open(&model_path)?;

        // Fetch the ref if needed
        let mut remote = repo.find_remote("origin")?;
        remote.fetch(&[ref_spec], None, None)?;

        // Parse and checkout the ref
        let obj = repo.revparse_single(ref_spec)?;
        repo.checkout_tree(&obj, None)?;
        repo.set_head_detached(obj.id())?;

        tracing::info!("Successfully checked out {} at {}", model_name, ref_spec);
    }

    // Generate a model ID for compatibility
    let model_id = ModelId::new();

    tracing::info!("Successfully cloned model {} to {:?}", model_name, model_path);

    Ok(ClonedModel {
        model_id,
        model_path,
        model_name,
    })
}

/// List all available models
pub async fn list_models() -> Result<Vec<(super::ModelRef, super::ModelMetadata)>> {
    let storage_paths = StoragePaths::new()?;
    let hypr_config = crate::config::HyprConfig::load().unwrap_or_default();
    let model_storage = ModelStorage::create_with_config(
        storage_paths.models_dir()?,
        hypr_config.git2db.clone()
    ).await?;
    model_storage.list_models().await
}