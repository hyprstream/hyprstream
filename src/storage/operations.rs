//! High-level model operations shared between CLI and server
//!
//! This module provides common operations to reduce duplication

use anyhow::Result;
use std::path::PathBuf;
use super::{GitModelSource, ModelStorage, ModelId, paths::StoragePaths};

/// Result of cloning a model
pub struct ClonedModel {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub model_name: String,
}

/// Clone a model from a git repository
///
/// This is the shared implementation used by both CLI and server
pub async fn clone_model(
    repo_url: &str,
    git_ref: Option<&str>,
) -> Result<ClonedModel> {
    // Get storage paths
    let storage_paths = StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    // Initialize GitModelSource
    let git_source = GitModelSource::new(models_dir.clone());

    // Clone the model
    let (model_id, model_path) = if let Some(ref_str) = git_ref {
        git_source.clone_ref(repo_url, ref_str).await?
    } else {
        git_source.clone_model(repo_url).await?
    };

    // Extract model name from URL
    let model_name = repo_url.split('/').last()
        .unwrap_or("unknown")
        .trim_end_matches(".git")
        .to_string();

    // Register with model storage
    let model_storage = ModelStorage::create(models_dir).await?;
    if let Err(e) = model_storage.register_with_git_registry(
        &model_id,
        &model_name,
        Some(repo_url.to_string())
    ).await {
        tracing::warn!("Failed to register with Git registry: {}", e);
    }

    Ok(ClonedModel {
        model_id,
        model_path,
        model_name,
    })
}

/// List all available models
pub async fn list_models() -> Result<Vec<(super::ModelRef, super::ModelMetadata)>> {
    let storage_paths = StoragePaths::new()?;
    let model_storage = ModelStorage::create(storage_paths.models_dir()?).await?;
    model_storage.list_models().await
}