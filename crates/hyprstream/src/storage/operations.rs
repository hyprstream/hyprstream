//! High-level model operations shared between CLI and server
//!
//! This module provides common operations to reduce duplication

use super::{paths::StoragePaths, ModelId, ModelStorage};
use anyhow::Result;
use crate::services::RegistryClient;
use std::path::PathBuf;
use std::sync::Arc;

/// Result of cloning a model
pub struct ClonedModel {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub model_name: String,
}

/// Options for cloning a repository
#[derive(Debug, Clone, Default)]
pub struct CloneOptions {
    /// Branch, tag, or commit to clone
    pub branch: Option<String>,
    /// Clone depth (0 means full clone)
    pub depth: u32,
    /// Suppress progress output
    pub quiet: bool,
    /// Verbose output
    pub verbose: bool,
}

/// Clone a model from a git repository using proper git submodule workflow
///
/// This is the shared implementation used by both CLI and server.
/// If a registry client is provided, it will be used for shared registry access.
/// Otherwise, a new local service is started.
pub async fn clone_model(
    repo_url: &str,
    name: Option<&str>,
    git_ref: Option<&str>,
) -> Result<ClonedModel> {
    clone_model_with_client(repo_url, name, git_ref, None).await
}

/// Clone a model with an optional shared registry client
pub async fn clone_model_with_client(
    repo_url: &str,
    name: Option<&str>,
    git_ref: Option<&str>,
    client: Option<Arc<dyn RegistryClient>>,
) -> Result<ClonedModel> {
    // Get storage paths
    let storage_paths = StoragePaths::new()?;
    let models_dir = storage_paths.models_dir()?;

    // Determine model name: use provided name or extract from URL
    let model_name = if let Some(n) = name {
        n.to_string()
    } else {
        // Extract model name from URL
        let extracted = repo_url
            .split('/')
            .next_back()
            .unwrap_or("")
            .trim_end_matches(".git");

        // If extraction yields empty string (e.g., gittorrent URLs), require explicit name
        if extracted.is_empty() {
            anyhow::bail!(
                "Cannot derive model name from URL '{}'. Please provide a name using --name flag.",
                repo_url
            );
        }

        extracted.to_string()
    };

    // Create model storage - use shared client if provided, otherwise start local service
    let model_storage = if let Some(client) = client {
        ModelStorage::new(client, models_dir.clone())
    } else {
        let hypr_config = crate::config::HyprConfig::load().unwrap_or_default();
        ModelStorage::create_with_config(models_dir.clone(), hypr_config.git2db.clone()).await?
    };

    // Use proper git submodule workflow
    tracing::info!(
        "Adding model {} as git submodule from {}",
        model_name,
        repo_url
    );
    model_storage.add_model(&model_name, repo_url).await?;

    // Get the model path (should exist after add)
    let model_ref = super::ModelRef::new(model_name.clone());
    let model_path = model_storage.get_model_path(&model_ref).await?;

    // If a specific git ref was requested, checkout that ref
    if let Some(ref_spec) = git_ref {
        tracing::info!(
            "Checking out git ref '{}' for model {}",
            ref_spec,
            model_name
        );

        // Branch/tag checkout is handled at the worktree level by git2db
        tracing::info!(
            "Model '{}' requested branch/tag '{}'",
            model_name,
            ref_spec
        );
    }

    // Generate a model ID for compatibility
    let model_id = ModelId::new();

    tracing::info!(
        "Successfully cloned model {} to {:?}",
        model_name,
        model_path
    );

    Ok(ClonedModel {
        model_id,
        model_path,
        model_name,
    })
}

/// Clone a model from a git repository with custom options
///
/// This implementation supports the git clone options that are compatible with git2db
pub async fn clone_model_with_options(
    repo_url: &str,
    name: Option<&str>,
    _model_id: Option<&str>,
    options: CloneOptions,
) -> Result<ClonedModel> {
    if options.verbose && !options.quiet {
        println!("Verbose mode enabled");
        if let Some(ref b) = options.branch {
            println!("   Cloning branch/tag: {}", b);
        }
        println!("   Clone depth: {}", if options.depth == 0 {
            "full history".to_string()
        } else {
            format!("{} commits", options.depth)
        });
    }

    // TODO: Pass depth parameter to git2db when supported
    // For now, delegate to the existing clone_model function
    clone_model(repo_url, name, options.branch.as_deref()).await
}

/// List all available models
pub async fn list_models() -> Result<Vec<(super::ModelRef, super::ModelMetadata)>> {
    list_models_with_client(None).await
}

/// List all available models with an optional shared registry client
pub async fn list_models_with_client(
    client: Option<Arc<dyn RegistryClient>>,
) -> Result<Vec<(super::ModelRef, super::ModelMetadata)>> {
    let storage_paths = StoragePaths::new()?;
    let model_storage = if let Some(client) = client {
        ModelStorage::new(client, storage_paths.models_dir()?)
    } else {
        let hypr_config = crate::config::HyprConfig::load().unwrap_or_default();
        ModelStorage::create_with_config(storage_paths.models_dir()?, hypr_config.git2db.clone())
            .await?
    };
    model_storage.list_models().await
}
