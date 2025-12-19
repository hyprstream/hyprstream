//! Handlers for worktree management commands

use crate::storage::{ModelRef, ModelStorage};
use anyhow::Result;
use std::io::{self, Write};
use tracing::info;

/// Handle worktree add command - create worktree from existing branch
pub async fn handle_worktree_add(
    storage: &ModelStorage,
    model: &str,
    branch: &str,
) -> Result<()> {
    info!("Creating worktree {}/{}", model, branch);

    let model_ref = ModelRef::new(model.to_string());

    // Check if worktree already exists
    let existing_path = storage.get_worktree_path(&model_ref, branch).await?;
    if existing_path.exists() {
        anyhow::bail!(
            "Worktree {} already exists for model {} at:\n  {}",
            branch,
            model,
            existing_path.display()
        );
    }

    // Get repo client and verify branch exists
    let repo_client = storage.get_repo_client(&model_ref).await?;
    let branches = repo_client.list_branches().await
        .map_err(|e| anyhow::anyhow!("Failed to list branches: {}", e))?;

    if !branches.contains(&branch.to_string()) {
        let branch_list = if branches.is_empty() {
            "  (no branches found)".to_string()
        } else {
            branches.iter().map(|b| format!("  - {}", b)).collect::<Vec<_>>().join("\n")
        };
        anyhow::bail!(
            "Branch '{}' does not exist in model {}.\n\nAvailable branches:\n{}",
            branch,
            model,
            branch_list
        );
    }

    // Create worktree using storage layer
    let worktree_path = storage.create_worktree(&model_ref, branch).await?;

    println!("Created worktree {}/{}:", model, branch);
    println!("  Path: {}", worktree_path.display());

    Ok(())
}

/// Handle worktree list command
pub async fn handle_worktree_list(
    storage: &ModelStorage,
    model: &str,
    show_all: bool,
) -> Result<()> {
    info!("Listing worktrees for model {}", model);

    let model_ref = ModelRef::new(model.to_string());
    let worktrees = storage.list_worktrees(&model_ref).await?;

    if show_all {
        // Show all branches including those without worktrees
        let repo_client = storage.get_repo_client(&model_ref).await?;
        let all_branches = repo_client.list_branches().await
            .map_err(|e| anyhow::anyhow!("Failed to list branches: {}", e))?;

        if all_branches.is_empty() {
            println!("No branches found for model {}", model);
            return Ok(());
        }

        println!("Branches for {}:\n", model);
        for branch in &all_branches {
            let has_worktree = worktrees.contains(branch);
            let status = if has_worktree { "[active]" } else { "[no worktree]" };
            println!("  {} {}", branch, status);
        }

        let inactive_count = all_branches.iter().filter(|b| !worktrees.contains(*b)).count();
        if inactive_count > 0 {
            println!("\nCreate a worktree with:");
            println!("  hyprstream worktree add {} <branch>", model);
        }
    } else {
        // Existing behavior - only show active worktrees
        if worktrees.is_empty() {
            println!("No worktrees found for model {}", model);
            println!("\nCreate a worktree with:");
            println!("  hyprstream worktree add {} <branch>", model);
            println!("\nList all branches with:");
            println!("  hyprstream worktree list {} --all", model);
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
