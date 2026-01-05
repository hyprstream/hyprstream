//! Handlers for remote management commands

use crate::storage::{ModelRef, ModelStorage};
use anyhow::Result;
use tracing::info;

/// Handle remote add command - add a new remote to a model
pub async fn handle_remote_add(
    storage: &ModelStorage,
    model: &str,
    name: &str,
    url: &str,
) -> Result<()> {
    info!("Adding remote '{}' to model {}", name, model);

    let model_ref = ModelRef::new(model.to_string());
    let repo_client = storage.get_repo_client(&model_ref).await?;

    repo_client
        .add_remote(name, url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to add remote '{}': {}", name, e))?;

    println!("Added remote '{}': {}", name, url);
    Ok(())
}

/// Handle remote list command - list all remotes for a model
pub async fn handle_remote_list(
    storage: &ModelStorage,
    model: &str,
    verbose: bool,
) -> Result<()> {
    info!("Listing remotes for model {}", model);

    let model_ref = ModelRef::new(model.to_string());
    let repo_client = storage.get_repo_client(&model_ref).await?;

    let remotes = repo_client
        .list_remotes()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to list remotes: {}", e))?;

    if remotes.is_empty() {
        println!("No remotes configured for {}", model);
        println!("\nAdd a remote with:");
        println!("  hyprstream remote add {} <name> <url>", model);
        return Ok(());
    }

    println!("Remotes for {}:\n", model);
    for remote in remotes {
        if verbose {
            println!("  {} (fetch): {}", remote.name, remote.url);
            if let Some(push_url) = &remote.push_url {
                println!("  {} (push):  {}", remote.name, push_url);
            }
        } else {
            println!("  {}\t{}", remote.name, remote.url);
        }
    }

    Ok(())
}

/// Handle remote remove command - remove a remote from a model
pub async fn handle_remote_remove(storage: &ModelStorage, model: &str, name: &str) -> Result<()> {
    info!("Removing remote '{}' from model {}", name, model);

    let model_ref = ModelRef::new(model.to_string());
    let repo_client = storage.get_repo_client(&model_ref).await?;

    repo_client
        .remove_remote(name)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to remove remote '{}': {}", name, e))?;

    println!("Removed remote '{}'", name);
    Ok(())
}

/// Handle remote set-url command - change a remote's URL
pub async fn handle_remote_set_url(
    storage: &ModelStorage,
    model: &str,
    name: &str,
    url: &str,
) -> Result<()> {
    info!("Setting URL for remote '{}' in model {}", name, model);

    let model_ref = ModelRef::new(model.to_string());
    let repo_client = storage.get_repo_client(&model_ref).await?;

    repo_client
        .set_remote_url(name, url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to set URL for remote '{}': {}", name, e))?;

    println!("Updated remote '{}': {}", name, url);
    Ok(())
}

/// Handle remote rename command - rename a remote
pub async fn handle_remote_rename(
    storage: &ModelStorage,
    model: &str,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    info!(
        "Renaming remote '{}' to '{}' in model {}",
        old_name, new_name, model
    );

    let model_ref = ModelRef::new(model.to_string());
    let repo_client = storage.get_repo_client(&model_ref).await?;

    repo_client
        .rename_remote(old_name, new_name)
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to rename remote '{}' to '{}': {}",
                old_name,
                new_name,
                e
            )
        })?;

    println!("Renamed remote '{}' to '{}'", old_name, new_name);
    Ok(())
}
