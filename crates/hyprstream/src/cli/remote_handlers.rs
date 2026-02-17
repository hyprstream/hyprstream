//! Handlers for remote management commands
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use crate::services::GenRegistryClient;
use anyhow::Result;
use tracing::info;

/// Handle remote add command - add a new remote to a model
pub async fn handle_remote_add(
    registry: &GenRegistryClient,
    model: &str,
    name: &str,
    url: &str,
) -> Result<()> {
    info!("Adding remote '{}' to model {}", name, model);

    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    repo_client
        .add_remote(name, url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to add remote '{}': {}", name, e))?;

    println!("Added remote '{name}': {url}");
    Ok(())
}

/// Handle remote list command - list all remotes for a model
pub async fn handle_remote_list(
    registry: &GenRegistryClient,
    model: &str,
    verbose: bool,
) -> Result<()> {
    info!("Listing remotes for model {}", model);

    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    let remotes = repo_client
        .list_remotes()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to list remotes: {}", e))?;

    if remotes.is_empty() {
        println!("No remotes configured for {model}");
        println!("\nAdd a remote with:");
        println!("  hyprstream remote add {model} <name> <url>");
        return Ok(());
    }

    println!("Remotes for {model}:\n");
    for remote in remotes {
        if verbose {
            println!("  {} (fetch): {}", remote.name, remote.url);
            if !remote.push_url.is_empty() {
                println!("  {} (push):  {}", remote.name, remote.push_url);
            }
        } else {
            println!("  {}\t{}", remote.name, remote.url);
        }
    }

    Ok(())
}

/// Handle remote remove command - remove a remote from a model
pub async fn handle_remote_remove(
    registry: &GenRegistryClient,
    model: &str,
    name: &str,
) -> Result<()> {
    info!("Removing remote '{}' from model {}", name, model);

    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    repo_client
        .remove_remote(name)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to remove remote '{}': {}", name, e))?;

    println!("Removed remote '{name}'");
    Ok(())
}

/// Handle remote set-url command - change a remote's URL
pub async fn handle_remote_set_url(
    registry: &GenRegistryClient,
    model: &str,
    name: &str,
    url: &str,
) -> Result<()> {
    info!("Setting URL for remote '{}' in model {}", name, model);

    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

    repo_client
        .set_remote_url(name, url)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to set URL for remote '{}': {}", name, e))?;

    println!("Updated remote '{name}': {url}");
    Ok(())
}

/// Handle remote rename command - rename a remote
pub async fn handle_remote_rename(
    registry: &GenRegistryClient,
    model: &str,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    info!(
        "Renaming remote '{}' to '{}' in model {}",
        old_name, new_name, model
    );

    let tracked = registry.get_by_name(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let repo_client = registry.repo(&tracked.id);

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

    println!("Renamed remote '{old_name}' to '{new_name}'");
    Ok(())
}
