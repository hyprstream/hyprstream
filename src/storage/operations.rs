//! High-level model operations shared between CLI and server
//!
//! This module provides common operations to reduce duplication

use anyhow::Result;
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
    git_ref: Option<&str>,
) -> Result<ClonedModel> {
    // TODO: git_ref support for submodules (checkout specific branch/tag after clone)
    if git_ref.is_some() {
        tracing::warn!("git_ref not yet supported with submodule workflow, using default branch");
    }

    // Get storage paths
    let storage_paths = StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    // Extract model name from URL
    let model_name = repo_url.split('/').last()
        .unwrap_or("unknown")
        .trim_end_matches(".git")
        .to_string();

    // Create model storage with registry
    let model_storage = ModelStorage::create(models_dir.clone()).await?;
    let registry = model_storage.registry();

    // Use proper git submodule workflow
    tracing::info!("Adding model {} as git submodule from {}", model_name, repo_url);
    let model_ref = registry.add_model(&model_name, repo_url).await?;

    // Get the model path (should exist after submodule add)
    let model_path = registry.get_model_path(&model_ref).await?;

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
    let model_storage = ModelStorage::create(storage_paths.models_dir()?).await?;
    model_storage.list_models().await
}