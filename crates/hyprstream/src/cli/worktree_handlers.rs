//! Handlers for worktree management commands
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use crate::cli::git_handlers::apply_policy_template_to_model;
use crate::services::RegistryClient;
use anyhow::Result;
use std::io::{self, Write};
use std::path::PathBuf;
use tracing::info;

/// Handle worktree add command - create worktree from existing branch
pub async fn handle_worktree_add(
    registry: &dyn RegistryClient,
    model: &str,
    branch: &str,
    policy_template: Option<String>,
) -> Result<()> {
    info!("Creating worktree {}/{}", model, branch);

    let repo_client = registry.repo(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

    // Check if worktree already exists via service
    let worktrees = repo_client.list_worktrees().await
        .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

    if let Some(existing) = worktrees.iter().find(|wt| wt.branch.as_deref() == Some(branch)) {
        anyhow::bail!(
            "Worktree {} already exists for model {} at:\n  {}",
            branch,
            model,
            existing.path.display()
        );
    }

    // Verify branch exists
    let branches = repo_client.list_branches().await
        .map_err(|e| anyhow::anyhow!("Failed to list branches: {}", e))?;

    if !branches.contains(&branch.to_owned()) {
        let branch_list = if branches.is_empty() {
            "  (no branches found)".to_owned()
        } else {
            branches.iter().map(|b| format!("  - {b}")).collect::<Vec<_>>().join("\n")
        };
        anyhow::bail!(
            "Branch '{}' does not exist in model {}.\n\nAvailable branches:\n{}",
            branch,
            model,
            branch_list
        );
    }

    // Create worktree via service - service determines the path
    let worktree_path = repo_client.create_worktree(&PathBuf::new(), branch).await
        .map_err(|e| anyhow::anyhow!("Failed to create worktree: {}", e))?;

    println!("Created worktree {model}/{branch}:");
    println!("  Path: {}", worktree_path.display());

    // Apply policy template if specified
    if let Some(ref template_name) = policy_template {
        apply_policy_template_to_model(registry, model, template_name).await?;
    }

    Ok(())
}

/// Handle worktree list command
pub async fn handle_worktree_list(
    registry: &dyn RegistryClient,
    model: &str,
    show_all: bool,
) -> Result<()> {
    info!("Listing worktrees for model {}", model);

    let repo_client = registry.repo(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;
    let worktrees = repo_client.list_worktrees().await
        .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

    // Helper to check if a branch has a worktree
    let has_worktree = |branch: &str| -> bool {
        worktrees.iter().any(|wt| wt.branch.as_deref() == Some(branch))
    };

    if show_all {
        // Show all branches including those without worktrees
        let all_branches = repo_client.list_branches().await
            .map_err(|e| anyhow::anyhow!("Failed to list branches: {}", e))?;

        if all_branches.is_empty() {
            println!("No branches found for model {model}");
            return Ok(());
        }

        println!("Branches for {model}:\n");
        for branch in &all_branches {
            let status = if has_worktree(branch) { "[active]" } else { "[no worktree]" };
            println!("  {branch} {status}");
        }

        let inactive_count = all_branches.iter().filter(|b| !has_worktree(b)).count();
        if inactive_count > 0 {
            println!("\nCreate a worktree with:");
            println!("  hyprstream worktree add {model} <branch>");
        }
    } else {
        // Existing behavior - only show active worktrees
        if worktrees.is_empty() {
            println!("No worktrees found for model {model}");
            println!("\nCreate a worktree with:");
            println!("  hyprstream worktree add {model} <branch>");
            println!("\nList all branches with:");
            println!("  hyprstream worktree list {model} --all");
            return Ok(());
        }

        println!("Worktrees for {model}:\n");

        for wt in &worktrees {
            let branch_name = wt.branch.as_deref().unwrap_or("<unnamed>");
            let dirty_marker = if wt.is_dirty { " [dirty]" } else { "" };
            println!("  {branch_name} ({model}){dirty_marker}");
            println!("    Path: {}", wt.path.display());
            println!();
        }
    }

    Ok(())
}

/// Handle worktree info command
pub async fn handle_worktree_info(
    registry: &dyn RegistryClient,
    model: &str,
    branch: &str,
) -> Result<()> {
    info!("Getting info for worktree {}/{}", model, branch);

    let repo_client = registry.repo(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

    // Get worktree info from service
    let worktrees = repo_client.list_worktrees().await
        .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

    let wt = worktrees.iter()
        .find(|wt| wt.branch.as_deref() == Some(branch))
        .ok_or_else(|| anyhow::anyhow!("Worktree {} does not exist for model {}", branch, model))?;

    println!("Worktree: {model}/{branch}\n");
    println!("Status: Active");
    println!("Path: {}", wt.path.display());
    println!("Branch: {branch}");
    if wt.is_dirty {
        println!("Dirty: yes (uncommitted changes)");
    }

    Ok(())
}

/// Handle worktree remove command
pub async fn handle_worktree_remove(
    registry: &dyn RegistryClient,
    model: &str,
    branch: &str,
    force: bool,
) -> Result<()> {
    info!("Removing worktree {}/{}", model, branch);

    let repo_client = registry.repo(model).await
        .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

    // Get worktree info from service
    let worktrees = repo_client.list_worktrees().await
        .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

    let wt = worktrees.iter()
        .find(|wt| wt.branch.as_deref() == Some(branch))
        .ok_or_else(|| anyhow::anyhow!("Worktree {} does not exist for model {}", branch, model))?;

    let worktree_path = &wt.path;

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

    // Remove the worktree via service
    repo_client.remove_worktree(worktree_path).await
        .map_err(|e| anyhow::anyhow!("Failed to remove worktree: {}", e))?;

    println!("✓ Removed worktree {model}/{branch}");

    Ok(())
}
