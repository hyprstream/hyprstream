//! Handlers for worktree management commands

use crate::storage::{format_duration, ModelRef, ModelStorage, WorktreeMetadata};
use anyhow::Result;
use std::io::{self, Write};
use tracing::info;

/// Handle worktree list command
pub async fn handle_worktree_list(storage: &ModelStorage, model: &str) -> Result<()> {
    info!("Listing worktrees for model {}", model);

    let model_ref = ModelRef::new(model.to_string());
    let worktrees = storage.list_worktrees_with_metadata(&model_ref).await?;

    if worktrees.is_empty() {
        println!("No worktrees found for model {}", model);
        println!("\nCreate a worktree with:");
        println!("  hyprstream branch {} <branch-name>", model);
        return Ok(());
    }

    println!("Worktrees for {}:\n", model);

    for (branch_name, meta_opt) in worktrees {
        if let Some(meta) = meta_opt {
            let age_str = format_duration(meta.age());
            let saved_str = meta.space_saved_human();
            let last_access = meta
                .time_since_last_access()
                .map(|d| format_duration(d))
                .unwrap_or_else(|| "never".to_string());

            println!("  Branch: {}", branch_name);
            println!("    Created: {} ago", age_str);
            if let Some(from) = &meta.created_from {
                println!("    From: {}", from);
            }
            println!("    Driver: {}", meta.storage_driver);
            if let Some(backend) = &meta.backend {
                println!("    Backend: {}", backend);
            }
            println!("    Space saved: {}", saved_str);
            println!("    Last accessed: {} ago", last_access);

            // Get worktree path
            if let Ok(path) = storage.get_worktree_path(&model_ref, &branch_name).await {
                println!("    Path: {}", path.display());
            }

            println!();
        } else {
            println!("  Branch: {} (no metadata)", branch_name);
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

    if let Some(meta) = WorktreeMetadata::try_load(&worktree_path) {
        println!("Status: Active");
        println!("Path: {}", worktree_path.display());
        println!();

        println!("Created: {} ({} ago)", meta.created_at, format_duration(meta.age()));
        if let Some(from) = &meta.created_from {
            println!("Created from: {}", from);
        }
        println!();

        println!("Storage driver: {}", meta.storage_driver);
        if let Some(backend) = &meta.backend {
            println!("Backend: {}", backend);
        }
        println!();

        if meta.space_saved_bytes.is_some() {
            println!("Space saved: {}", meta.space_saved_human());
            if let Some(efficiency) = meta.space_efficiency() {
                println!("Efficiency: {:.1}%", efficiency);
            }
        } else {
            println!("Space saved: Not available");
        }
        println!();

        if let Some(last_access) = meta.time_since_last_access() {
            println!("Last accessed: {} ago", format_duration(last_access));
        }

        if !meta.tags.is_empty() {
            println!("\nTags: {}", meta.tags.join(", "));
        }
    } else {
        println!("Status: Active (no metadata)");
        println!("Path: {}", worktree_path.display());
        println!("\nNote: This worktree was created without metadata tracking.");
    }

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

    // Get space saved info before removal
    let space_info = WorktreeMetadata::try_load(&worktree_path)
        .map(|meta| format!(" (freed {})", meta.space_saved_human()));

    // Remove the worktree
    storage.remove_worktree(&model_ref, branch).await?;

    println!("✓ Removed worktree {}/{}", model, branch);
    if let Some(info) = space_info {
        println!("  {}", info);
    }

    Ok(())
}

/// Handle worktree prune command
pub async fn handle_worktree_prune(
    storage: &ModelStorage,
    model: &str,
    days: u32,
    dry_run: bool,
) -> Result<()> {
    info!("Pruning worktrees for model {} (inactive for {} days)", model, days);

    let model_ref = ModelRef::new(model.to_string());
    let worktrees = storage.list_worktrees_with_metadata(&model_ref).await?;

    let threshold = chrono::Duration::days(days as i64);
    let mut to_prune = Vec::new();

    for (branch_name, meta_opt) in worktrees {
        if let Some(meta) = meta_opt {
            if let Some(last_access) = meta.time_since_last_access() {
                if last_access > threshold {
                    to_prune.push((branch_name, meta));
                }
            }
        }
    }

    if to_prune.is_empty() {
        println!("No stale worktrees found for model {}", model);
        println!("(worktrees inactive for more than {} days)", days);
        return Ok(());
    }

    println!("Found {} stale worktrees:\n", to_prune.len());

    for (branch_name, meta) in &to_prune {
        let last_access = meta
            .time_since_last_access()
            .map(|d| format_duration(d))
            .unwrap_or_else(|| "unknown".to_string());
        println!("  {} (last accessed {} ago)", branch_name, last_access);
    }

    if dry_run {
        println!("\n(Dry run - no changes made)");
        println!("Run without --dry-run to actually remove these worktrees.");
        return Ok(());
    }

    println!("\n⚠️  Remove these worktrees?");
    print!("  Continue? [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Aborted.");
        return Ok(());
    }

    // Remove the worktrees
    for (branch_name, _meta) in to_prune {
        match storage.remove_worktree(&model_ref, &branch_name).await {
            Ok(_) => println!("✓ Removed {}", branch_name),
            Err(e) => eprintln!("✗ Failed to remove {}: {}", branch_name, e),
        }
    }

    Ok(())
}
