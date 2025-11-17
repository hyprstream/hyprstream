//! Handlers for worktree management commands

use crate::storage::{ModelRef, ModelStorage};
use anyhow::Result;
use std::io::{self, Write};
use tracing::info;

/// Handle worktree list command
pub async fn handle_worktree_list(storage: &ModelStorage, model: &str) -> Result<()> {
    info!("Listing worktrees for model {}", model);

    let model_ref = ModelRef::new(model.to_string());
    let worktrees = storage.list_worktrees(&model_ref).await?;

    if worktrees.is_empty() {
        println!("No worktrees found for model {}", model);
        println!("\nCreate a worktree with:");
        println!("  hyprstream branch {} <branch-name>", model);
        return Ok(());
    }

    println!("Worktrees for {}:\n", model);

    for branch_name in worktrees {
        println!("  {} ({})", branch_name, model);

        // Get worktree path
        if let Ok(path) = storage.get_worktree_path(&model_ref, &branch_name).await {
            println!("    Path: {}", path.display());
        }

        println!();
    }

    Ok(())
}

/// Handle worktree info command
pub async fn handle_worktree_info(
    storage: &ModelStorage,
    model: &str,
    branch: &str,
) -> Result<()> {
    info!("Getting info for worktree {}/{}", model, branch);

    let model_ref = ModelRef::new(model.to_string());
    let worktree_path = storage.get_worktree_path(&model_ref, branch).await?;

    if !worktree_path.exists() {
        anyhow::bail!("Worktree {} does not exist for model {}", branch, model);
    }

    println!("Worktree: {}/{}\n", model, branch);
    println!("Status: Active");
    println!("Path: {}", worktree_path.display());
    println!("Branch: {}", branch);

    Ok(())
}

/// Handle worktree remove command
pub async fn handle_worktree_remove(
    storage: &ModelStorage,
    model: &str,
    branch: &str,
    force: bool,
) -> Result<()> {
    info!("Removing worktree {}/{}", model, branch);

    let model_ref = ModelRef::new(model.to_string());
    let worktree_path = storage.get_worktree_path(&model_ref, branch).await?;

    if !worktree_path.exists() {
        anyhow::bail!("Worktree {} does not exist for model {}", branch, model);
    }

    // Confirm removal unless forced
    if !force {
        println!("⚠️  Warning: This will remove the worktree at:");
        println!("    {}", worktree_path.display());
        println!("\n  Any uncommitted changes will be lost!");
        print!("\n  Continue? [y/N] ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    // Remove the worktree
    storage.remove_worktree(&model_ref, branch).await?;

    println!("✓ Removed worktree {}/{}", model, branch);

    Ok(())
}
