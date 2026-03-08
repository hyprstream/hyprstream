//! VFS driver (plain directories)
//!
//! The vfs driver is Docker's universal fallback - it uses plain directories
//! with no optimization. This driver always works on all platforms.
//!
//! In our case, it creates standard git worktrees using libgit2 without
//! any storage optimization layers underneath.

use super::driver::{Driver, DriverOpts, WorktreeHandle, DriverFactory};
use crate::errors::{Git2DBError, Git2DBResult};
use async_trait::async_trait;
use std::path::Path;
use tracing::{info, warn};

inventory::submit!(DriverFactory::new(
    "vfs",
    || Box::new(VfsDriver)
));

/// VFS storage driver (plain git worktree, no optimization)
///
/// This is the universal fallback that always works. It creates standard
/// git worktrees using libgit2 without any CoW or space-saving features.
pub struct VfsDriver;

#[async_trait]
impl Driver for VfsDriver {
    fn name(&self) -> &'static str {
        "vfs"
    }

    fn is_available(&self) -> bool {
        // VFS is always available (it's just plain directories)
        true
    }

    async fn create_worktree(&self, opts: &DriverOpts) -> Git2DBResult<WorktreeHandle> {
        // Strict: worktree path must not exist
        if opts.worktree_path.exists() {
            return Err(Git2DBError::worktree_exists(&opts.worktree_path));
        }

        // Validate inputs
        if !opts.base_repo.exists() {
            return Err(Git2DBError::invalid_path(
                opts.base_repo.clone(),
                "Base repository does not exist",
            ));
        }

        // Ensure parent directory exists
        if let Some(parent) = opts.worktree_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                Git2DBError::internal(format!("Failed to create parent directory: {e}"))
            })?;
        }

        info!(
            "Creating vfs worktree: base={}, path={}, ref={}",
            opts.base_repo.display(),
            opts.worktree_path.display(),
            opts.ref_spec
        );

        // Create git worktree using libgit2
        self.create_git_worktree(opts).await?;

        // Return handle (no special cleanup needed for plain directories)
        Ok(WorktreeHandle::new(
            opts.worktree_path.clone(),
            "vfs".to_owned(),
        ))
    }

    async fn get_worktrees(&self, base_repo: &Path) -> Git2DBResult<Vec<WorktreeHandle>> {
        let worktrees_dir = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees");

        let mut worktrees = Vec::new();

        if !worktrees_dir.exists() {
            return Ok(worktrees);
        }

        // Read worktrees directory
        for entry in std::fs::read_dir(&worktrees_dir)? {
            let entry = entry?;
            let worktree_path = entry.path();

            // Skip non-directories and git's internal directories
            if !worktree_path.is_dir() || worktree_path.file_name().is_some_and(|name| {
                name.to_string_lossy().starts_with(".git")
            }) {
                continue;
            }

            // For VFS driver, any directory with a .git inside is a valid worktree
            if worktree_path.join(".git").exists() {
                worktrees.push(WorktreeHandle::new(worktree_path, "vfs".to_owned()));
            }
        }

        Ok(worktrees)
    }

    async fn get_worktree(&self, base_repo: &Path, branch: &str) -> Git2DBResult<Option<WorktreeHandle>> {
        let worktrees_dir = base_repo.parent()
            .ok_or_else(|| Git2DBError::invalid_path(base_repo.to_path_buf(), "Invalid base repository path"))?
            .join("worktrees");
        let worktree_path = hyprstream_containedfs::contained_join(&worktrees_dir, branch)
            .map_err(|e| Git2DBError::invalid_path(worktrees_dir, format!("Path containment: {e}")))?;

        if worktree_path.exists() && worktree_path.join(".git").exists() {
            Ok(Some(WorktreeHandle::new(worktree_path, "vfs".to_owned())))
        } else {
            Ok(None)
        }
    }
}

/// Skip-worktree flag in git index extended flags.
///
/// When set on an index entry, `checkout_head()` will not overwrite the
/// (absent) working-tree file, preventing re-materialization of filtered-out
/// paths on subsequent checkouts.
pub(crate) const GIT_INDEX_ENTRY_SKIP_WORKTREE: u16 = 0x4000;

/// Check whether `path` matches any of the `keep_paths` with proper path-boundary semantics.
///
/// A keep_path ending with `/` is a directory prefix: `backends/cuda130/` matches
/// `backends/cuda130/lib.so` but NOT `backends/cuda130_old/lib.so`.
/// A keep_path without `/` suffix requires exact match or a `/`-separated prefix:
/// `manifest.toml` matches `manifest.toml` but NOT `manifest.toml.bak`.
fn path_matches_keep(path: &str, keep: &str) -> bool {
    if keep.ends_with('/') {
        path.starts_with(keep)
    } else {
        path == keep || path.starts_with(&format!("{keep}/"))
    }
}

/// Apply pathspec filtering to a worktree after creation.
///
/// 1. Re-checkout with only the matching paths materialized
/// 2. Remove working-tree files that don't match
/// 3. Set skip-worktree bits on excluded index entries
///
/// Shared by VFS and overlay2 drivers.
pub(crate) fn apply_pathspec_filter(
    worktree_path: &std::path::Path,
    keep_paths: &[String],
) -> Git2DBResult<()> {
    let wt_repo = git2::Repository::open(worktree_path)
        .map_err(|e| Git2DBError::internal(format!("Failed to open worktree: {e}")))?;

    info!(
        "Applying pathspec filter to worktree at {}: {:?}",
        worktree_path.display(),
        keep_paths
    );

    // Step 1: Remove working-tree files that don't match keep_paths
    {
        let index = wt_repo.index().map_err(|e| {
            Git2DBError::internal(format!("Failed to open index: {e}"))
        })?;
        for i in 0..index.len() {
            let entry = match index.get(i) {
                Some(e) => e,
                None => continue,
            };
            let path = String::from_utf8_lossy(&entry.path).to_string();
            let keep = keep_paths.iter().any(|p| path_matches_keep(&path, p));
            if !keep {
                let full_path = worktree_path.join(&path);
                if full_path.exists() {
                    if let Err(e) = std::fs::remove_file(&full_path) {
                        warn!("Could not remove filtered path {}: {e}", full_path.display());
                    }
                }
            }
        }
        // Clean up empty directories left behind
        remove_empty_dirs(worktree_path);
    }

    // Step 2: Set skip-worktree bits on excluded index entries
    mark_skip_worktree(&wt_repo, keep_paths)?;

    let index = wt_repo.index().map_err(|e| {
        Git2DBError::internal(format!("Failed to open index: {e}"))
    })?;
    let total = index.len();
    let skipped = (0..total)
        .filter(|i| {
            index
                .get(*i)
                .map(|e| e.flags_extended & GIT_INDEX_ENTRY_SKIP_WORKTREE != 0)
                .unwrap_or(false)
        })
        .count();

    info!(
        "Pathspec filter applied: {}/{} entries materialized, {} skipped",
        total - skipped,
        total,
        skipped
    );

    Ok(())
}

/// Set `GIT_INDEX_ENTRY_SKIP_WORKTREE` on index entries that don't match
/// the keep_paths prefixes. This prevents `checkout_head()` from
/// re-materializing them on subsequent operations.
pub(crate) fn mark_skip_worktree(
    repo: &git2::Repository,
    keep_paths: &[String],
) -> Git2DBResult<()> {
    let mut index = repo.index().map_err(|e| {
        Git2DBError::internal(format!("Failed to open index: {e}"))
    })?;

    for i in 0..index.len() {
        let entry = match index.get(i) {
            Some(e) => e,
            None => continue,
        };
        let path = String::from_utf8_lossy(&entry.path).to_string();
        let dominated = keep_paths.iter().any(|p| path_matches_keep(&path, p));
        if !dominated {
            let mut modified = entry;
            modified.flags_extended |= GIT_INDEX_ENTRY_SKIP_WORKTREE;
            index.add(&modified).map_err(|e| {
                Git2DBError::internal(format!(
                    "Failed to set skip-worktree on '{}': {e}",
                    path
                ))
            })?;
        }
    }

    index.write().map_err(|e| {
        Git2DBError::internal(format!("Failed to write index: {e}"))
    })?;

    Ok(())
}

/// Recursively remove empty directories under `root`, ignoring `.git`.
fn remove_empty_dirs(root: &std::path::Path) {
    fn walk(dir: &std::path::Path) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        let mut has_files = false;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            if name == ".git" {
                has_files = true;
                continue;
            }
            if path.is_dir() {
                if walk(&path) {
                    has_files = true;
                } else {
                    let _ = std::fs::remove_dir(&path);
                }
            } else {
                has_files = true;
            }
        }
        has_files
    }
    walk(root);
}

impl VfsDriver {
    /// Create git worktree using libgit2 with unified ref support
    ///
    /// Supports any git ref: branches, commits, tags, symbolic refs (HEAD~3), etc.
    /// When `opts.checkout_paths` is set, only matching files are materialized
    /// and excluded entries get skip-worktree bits.
    async fn create_git_worktree(&self, opts: &DriverOpts) -> Git2DBResult<()> {
        let base_repo = opts.base_repo.clone();
        let worktree_path = opts.worktree_path.clone();
        let ref_spec = opts.ref_spec.clone();
        let progress = opts.progress.clone();

        let _ = tokio::task::spawn_blocking(move || {
            // Set up smudge progress hook if progress reporter is available
            if let Some(ref reporter) = progress {
                let r = std::sync::Arc::clone(reporter);
                git_xet_filter::set_smudge_progress(std::sync::Arc::new(move |count, _path| {
                    r.report("smudge", count, 0);
                }));
            }

            let result = (|| -> Git2DBResult<()> {
                let repo = git2::Repository::open(&base_repo)
                    .map_err(|e| Git2DBError::internal(format!("Failed to open repository: {e}")))?;

                let object = repo.revparse_single(&ref_spec).map_err(|e| {
                    Git2DBError::internal(format!("Failed to resolve ref '{}': {}", ref_spec, e))
                })?;

                let commit = object.peel_to_commit().map_err(|e| {
                    Git2DBError::internal(format!(
                        "Ref '{}' does not point to a commit: {}",
                        ref_spec, e
                    ))
                })?;

                let branch_ref_name = format!("refs/heads/{}", ref_spec);
                let is_branch = repo.find_reference(&branch_ref_name).is_ok();

                let worktree_name = worktree_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| {
                        Git2DBError::invalid_path(worktree_path.clone(), "Invalid worktree path")
                    })?;

                if is_branch {
                    let reference = repo.find_reference(&branch_ref_name)?;
                    repo.worktree(
                        worktree_name,
                        &worktree_path,
                        Some(git2::WorktreeAddOptions::new().reference(Some(&reference))),
                    )
                    .map_err(|e| Git2DBError::internal(format!("Failed to create worktree: {e}")))?;

                    info!(
                        "Created git worktree at {} for branch '{}' (commit: {})",
                        worktree_path.display(),
                        ref_spec,
                        commit.id()
                    );
                } else {
                    repo.worktree(worktree_name, &worktree_path, None)
                        .map_err(|e| Git2DBError::internal(format!("Failed to create worktree: {e}")))?;

                    let wt_repo = git2::Repository::open(&worktree_path)?;
                    wt_repo.set_head_detached(commit.id()).map_err(|e| {
                        Git2DBError::internal(format!("Failed to set detached HEAD: {e}"))
                    })?;

                    wt_repo
                        .checkout_head(Some(git2::build::CheckoutBuilder::default().force()))
                        .map_err(|e| Git2DBError::internal(format!("Failed to checkout HEAD: {e}")))?;

                    info!(
                        "Created git worktree at {} for ref '{}' (detached HEAD at {})",
                        worktree_path.display(),
                        ref_spec,
                        commit.id()
                    );
                }

                Ok(())
            })();

            // Always clear smudge progress hook
            git_xet_filter::clear_smudge_progress();

            result
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?;

        // Apply pathspec filter if requested
        if let Some(ref paths) = opts.checkout_paths {
            apply_pathspec_filter(&opts.worktree_path, paths)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_name() {
        let driver = VfsDriver;
        assert_eq!(driver.name(), "vfs");
    }

    #[test]
    fn test_always_available() {
        let driver = VfsDriver;
        assert!(driver.is_available());
    }
}