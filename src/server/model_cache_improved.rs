//! Improved model cache using Git worktrees instead of copying files

use anyhow::{Result, anyhow};
use git2::{Repository, Oid};
use std::path::{Path, PathBuf};
use tracing::{info, debug};

/// Create or get a worktree for a specific commit
///
/// This is what get_or_create_checkout SHOULD be doing
pub async fn get_or_create_checkout_improved(
    model_path: &Path,
    commit_id: Oid,
    checkout_base: &Path,
) -> Result<PathBuf> {
    let checkout_dir = checkout_base.join(format!("{}", commit_id));

    // Fast path: already exists
    if checkout_dir.exists() {
        debug!("Reusing existing worktree at {:?}", checkout_dir);
        return Ok(checkout_dir);
    }

    // Create worktree for this specific commit
    info!("Creating worktree for commit {} at {:?}", commit_id, checkout_dir);

    // Run in blocking task since git2 is not async
    let model_path = model_path.to_path_buf();
    let checkout_dir_clone = checkout_dir.clone();

    tokio::task::spawn_blocking(move || -> Result<PathBuf> {
        let repo = Repository::open(&model_path)?;

        // Create worktree name based on commit ID
        let worktree_name = format!("cache-{}", &commit_id.to_string()[..8]);

        // Check if worktree already exists
        if repo.find_worktree(&worktree_name).is_ok() {
            // Worktree exists but directory was removed, prune it
            if let Ok(wt) = repo.find_worktree(&worktree_name) {
                wt.prune(Some(git2::WorktreePruneOptions::new().working_tree(true)))?;
            }
        }

        // Create parent directory
        if let Some(parent) = checkout_dir_clone.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Add worktree at specific commit
        let mut opts = git2::WorktreeAddOptions::new();
        opts.reference(Some(&repo.find_commit(commit_id)?));

        repo.worktree(
            &worktree_name,
            &checkout_dir_clone,
            Some(&opts),
        )?;

        // The worktree is already at the right commit due to the reference option

        Ok(checkout_dir_clone)
    }).await?
}

/// Alternative implementation using detached HEAD
///
/// This works when WorktreeAddOptions::reference is not available
pub async fn get_or_create_checkout_detached(
    model_path: &Path,
    commit_id: Oid,
    checkout_base: &Path,
) -> Result<PathBuf> {
    let checkout_dir = checkout_base.join(format!("{}", commit_id));

    if checkout_dir.exists() {
        return Ok(checkout_dir);
    }

    let model_path = model_path.to_path_buf();
    let checkout_dir_clone = checkout_dir.clone();

    tokio::task::spawn_blocking(move || -> Result<PathBuf> {
        let repo = Repository::open(&model_path)?;

        let worktree_name = format!("cache-{}", &commit_id.to_string()[..8]);

        // Clean up if needed
        if let Ok(wt) = repo.find_worktree(&worktree_name) {
            wt.prune(Some(git2::WorktreePruneOptions::new().working_tree(true)))?;
        }

        // Create directory
        std::fs::create_dir_all(&checkout_dir_clone)?;

        // Create worktree (will be at HEAD initially)
        let opts = git2::WorktreeAddOptions::new();
        repo.worktree(
            &worktree_name,
            &checkout_dir_clone,
            Some(&opts),
        )?;

        // Open the worktree repo and set it to the specific commit
        let wt_repo = Repository::open(&checkout_dir_clone)?;
        wt_repo.set_head_detached(commit_id)?;
        wt_repo.checkout_head(Some(
            git2::build::CheckoutBuilder::default()
                .force()
        ))?;

        Ok(checkout_dir_clone)
    }).await?
}

/// Clean up old worktrees
pub async fn cleanup_old_worktrees(
    model_path: &Path,
    keep_count: usize,
) -> Result<()> {
    let model_path = model_path.to_path_buf();

    tokio::task::spawn_blocking(move || -> Result<()> {
        let repo = Repository::open(&model_path)?;

        // List all worktrees that start with "cache-"
        let mut cache_worktrees = Vec::new();

        repo.worktree_foreach(|name, _path| {
            if name.starts_with("cache-") {
                cache_worktrees.push(name.to_string());
            }
            true
        })?;

        // Sort by name (which includes timestamp or commit ID)
        cache_worktrees.sort();

        // Remove old ones if we have too many
        if cache_worktrees.len() > keep_count {
            let to_remove = cache_worktrees.len() - keep_count;
            for name in cache_worktrees.iter().take(to_remove) {
                if let Ok(wt) = repo.find_worktree(name) {
                    info!("Pruning old worktree: {}", name);
                    wt.prune(Some(
                        git2::WorktreePruneOptions::new()
                            .working_tree(true)
                    ))?;
                }
            }
        }

        Ok(())
    }).await?
}