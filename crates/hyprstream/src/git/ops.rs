//! Low-level git operations on &git2::Repository
//!
//! These are the shared implementations behind both RegistryService handlers
//! and CheckpointManager. Each function takes &Repository and parameters,
//! performing a single git operation.

use anyhow::{anyhow, Result};
use git2::Repository;
use std::path::Path;

use crate::services::rpc_types::{DetailedStatusData, FileStatusData};

// === Commit Operations ===

/// Commit staged changes with default signature
pub fn commit_index(repo: &Repository, message: &str) -> Result<git2::Oid> {
    let mut index = repo.index()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;

    let sig = repo.signature()
        .map_err(|e| anyhow!("Failed to create signature: {}", e))?;
    let parent = repo.head()?.peel_to_commit()?;

    let oid = repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[&parent])?;
    Ok(oid)
}

/// Commit with explicit author
pub fn commit_with_author(
    repo: &Repository,
    message: &str,
    name: &str,
    email: &str,
) -> Result<git2::Oid> {
    let mut index = repo.index()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;

    let head = repo.head()?;
    let parent_commit = head.peel_to_commit()?;

    let author = git2::Signature::now(name, email)?;
    let committer = git2::Signature::now(name, email)?;

    let oid = repo.commit(
        Some("HEAD"),
        &author,
        &committer,
        message,
        &tree,
        &[&parent_commit],
    )?;

    Ok(oid)
}

/// Amend HEAD commit with staged changes
pub fn amend_head(repo: &Repository, message: &str) -> Result<git2::Oid> {
    let mut index = repo.index()?;
    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;

    let head = repo.head()?;
    let commit_to_amend = head.peel_to_commit()?;

    let new_oid = commit_to_amend.amend(
        Some("HEAD"),
        None,  // Keep author
        None,  // Keep committer
        None,  // Keep encoding
        Some(message),
        Some(&tree),
    )?;

    Ok(new_oid)
}

// === Staging Operations ===

/// Stage all files including untracked (git add -A)
pub fn stage_all_with_untracked(repo: &Repository) -> Result<()> {
    let mut index = repo.index()?;
    index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
    index.write()?;
    Ok(())
}

/// Stage specific files by relative path
pub fn stage_files(repo: &Repository, files: &[&str]) -> Result<()> {
    let mut index = repo.index()?;
    for file in files {
        index.add_path(Path::new(file))?;
    }
    index.write()?;
    Ok(())
}

// === Tag Operations ===

/// List all tags
pub fn list_tags(repo: &Repository) -> Result<Vec<String>> {
    let tags = repo.tag_names(None)?;
    let result: Vec<String> = tags
        .iter()
        .filter_map(|t| t.map(std::borrow::ToOwned::to_owned))
        .collect();
    Ok(result)
}

/// Create lightweight tag
///
/// If `target` is None, tags HEAD. If `force` is true, overwrites existing tags.
pub fn create_tag(
    repo: &Repository,
    name: &str,
    target: Option<&str>,
    force: bool,
) -> Result<()> {
    let target_oid = if let Some(target_spec) = target {
        repo.revparse_single(target_spec)?.id()
    } else {
        repo.head()?
            .target()
            .ok_or_else(|| anyhow!("HEAD has no target"))?
    };

    let commit = repo.find_commit(target_oid)?;
    repo.tag_lightweight(name, commit.as_object(), force)?;

    Ok(())
}

/// Delete a tag
pub fn delete_tag(repo: &Repository, name: &str) -> Result<()> {
    repo.tag_delete(name)
        .map_err(|e| anyhow!("Failed to delete tag '{}': {}", name, e))?;
    Ok(())
}

// === Status ===

/// Detailed status with per-file change info, ahead/behind, merge state
pub fn detailed_status(repo: &Repository) -> Result<DetailedStatusData> {
    let repo_path = repo
        .workdir()
        .unwrap_or_else(|| repo.path())
        .to_path_buf();

    // Get branch name
    let branch = repo
        .head()
        .ok()
        .and_then(|h| h.shorthand().map(std::borrow::ToOwned::to_owned));

    // Get HEAD OID
    let head = repo
        .head()
        .ok()
        .and_then(|h| h.target().map(|o| o.to_string()));

    // Check for merge/rebase in progress
    let merge_in_progress = repo.find_reference("MERGE_HEAD").is_ok();
    let rebase_in_progress = repo_path.join(".git/rebase-merge").exists()
        || repo_path.join(".git/rebase-apply").exists();

    // Get statuses
    let statuses = repo.statuses(None)?;

    let mut files = Vec::new();
    for entry in statuses.iter() {
        if let Some(path) = entry.path() {
            let status = entry.status();

            // Index status
            let index_status = if status.contains(git2::Status::INDEX_NEW) {
                Some("A".to_owned())
            } else if status.contains(git2::Status::INDEX_MODIFIED) {
                Some("M".to_owned())
            } else if status.contains(git2::Status::INDEX_DELETED) {
                Some("D".to_owned())
            } else if status.contains(git2::Status::INDEX_RENAMED) {
                Some("R".to_owned())
            } else if status.contains(git2::Status::INDEX_TYPECHANGE) {
                Some("T".to_owned())
            } else {
                None
            };

            // Worktree status
            let worktree_status = if status.contains(git2::Status::WT_NEW) {
                Some("?".to_owned())
            } else if status.contains(git2::Status::WT_MODIFIED) {
                Some("M".to_owned())
            } else if status.contains(git2::Status::WT_DELETED) {
                Some("D".to_owned())
            } else if status.contains(git2::Status::WT_RENAMED) {
                Some("R".to_owned())
            } else if status.contains(git2::Status::WT_TYPECHANGE) {
                Some("T".to_owned())
            } else if status.contains(git2::Status::CONFLICTED) {
                Some("U".to_owned())
            } else {
                None
            };

            files.push(FileStatusData {
                path: path.to_owned(),
                index_status,
                worktree_status,
            });
        }
    }

    // Get ahead/behind (simplified - assume origin/{branch})
    let (ahead, behind) = if let Ok(head_ref) = repo.head() {
        if let Some(branch_name) = head_ref.shorthand() {
            let upstream_name = format!("origin/{}", branch_name);
            if let Ok(upstream) = repo.revparse_single(&upstream_name) {
                if let (Ok(local), Ok(remote)) = (
                    head_ref.peel_to_commit(),
                    upstream.peel_to_commit(),
                ) {
                    repo.graph_ahead_behind(local.id(), remote.id())
                        .unwrap_or((0, 0))
                } else {
                    (0, 0)
                }
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    Ok(DetailedStatusData {
        branch,
        head,
        merge_in_progress,
        rebase_in_progress,
        files,
        ahead: ahead as u32,
        behind: behind as u32,
    })
}

// === Merge Operations ===

/// Abort merge: reset to ORIG_HEAD, cleanup state
pub fn abort_merge(repo: &Repository) -> Result<()> {
    let orig_head = repo
        .refname_to_id("ORIG_HEAD")
        .map_err(|_| anyhow!("No merge in progress (ORIG_HEAD not found)"))?;

    let commit = repo.find_commit(orig_head)?;
    repo.reset(commit.as_object(), git2::ResetType::Hard, None)?;
    repo.cleanup_state()?;

    Ok(())
}

/// Continue merge: check no conflicts, create merge commit, cleanup
pub fn continue_merge(repo: &Repository, message: Option<&str>) -> Result<git2::Oid> {
    let mut index = repo.index()?;

    if index.has_conflicts() {
        return Err(anyhow!(
            "Conflicts still present. Resolve all conflicts before continuing."
        ));
    }

    let tree_id = index.write_tree()?;
    let tree = repo.find_tree(tree_id)?;
    let sig = repo.signature()?;

    let head = repo.head()?.peel_to_commit()?;
    let merge_head = repo
        .find_reference("MERGE_HEAD")
        .map_err(|_| anyhow!("No merge in progress (MERGE_HEAD not found)"))?
        .peel_to_commit()?;

    let commit_message = message.map(String::from).unwrap_or_else(|| {
        format!(
            "Merge branch '{}'",
            merge_head.summary().unwrap_or("unknown")
        )
    });

    let oid = repo.commit(
        Some("HEAD"),
        &sig,
        &sig,
        &commit_message,
        &tree,
        &[&head, &merge_head],
    )?;

    repo.cleanup_state()?;

    Ok(oid)
}

/// Quit merge: cleanup state only
pub fn quit_merge(repo: &Repository) -> Result<()> {
    repo.cleanup_state()?;
    Ok(())
}

// === Remote Operations ===

/// Fetch from remote with optional refspec
pub fn fetch(repo: &Repository, remote_name: &str, refspec: Option<&str>) -> Result<()> {
    let mut remote = repo
        .find_remote(remote_name)
        .map_err(|e| anyhow!("Failed to find remote '{}': {}", remote_name, e))?;

    let refspecs: Vec<&str> = match refspec {
        Some(r) => vec![r],
        None => vec![],
    };

    remote
        .fetch(&refspecs, None, None)
        .map_err(|e| anyhow!("Failed to fetch from {}: {}", remote_name, e))?;

    Ok(())
}

/// Push to remote
pub fn push(repo: &Repository, remote_name: &str, refspec: &str, force: bool) -> Result<()> {
    let mut remote = repo
        .find_remote(remote_name)
        .map_err(|e| anyhow!("Failed to find remote '{}': {}", remote_name, e))?;

    let mut push_options = git2::PushOptions::new();

    let push_refspec = if force {
        format!("+{}", refspec)
    } else {
        refspec.to_owned()
    };

    remote
        .push(&[&push_refspec], Some(&mut push_options))
        .map_err(|e| anyhow!("Failed to push to {}: {}", remote_name, e))?;

    Ok(())
}

// === Branch ===

/// Ensure branch exists, creating from HEAD if needed; switch to it
pub fn ensure_branch(repo: &Repository, name: &str) -> Result<()> {
    let branch_ref = format!("refs/heads/{}", name);
    if repo.find_reference(&branch_ref).is_ok() {
        repo.set_head(&branch_ref)?;
        repo.checkout_head(Some(git2::build::CheckoutBuilder::default().force()))?;
    } else {
        let head = repo.head()?.peel_to_commit()?;
        repo.branch(name, &head, false)?;
        repo.set_head(&branch_ref)?;
    }
    Ok(())
}

// === Utility ===

/// COW-aware file copy (uses reflink when filesystem supports it, falls back to regular copy).
pub fn cow_copy(src: &Path, dst: &Path) -> Result<()> {
    reflink_copy::reflink_or_copy(src, dst)
        .map(|_| ())
        .map_err(|e| anyhow!("cow_copy failed: {}", e))
}

/// Sanitize a string for use as a Git tag name
pub fn sanitize_tag_name(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            ' ' | '\t' | '\n' => '-',
            '/' | '\\' | ':' | '?' | '*' | '"' | '<' | '>' | '|' | '[' | ']' => '_',
            _ if c.is_control() => '_',
            _ => c,
        })
        .collect::<String>()
        .trim_matches(|c| c == '-' || c == '_' || c == '.')
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_tag_name() {
        assert_eq!(sanitize_tag_name("checkpoint-v1"), "checkpoint-v1");
        assert_eq!(sanitize_tag_name("My Adapter Name"), "my-adapter-name");
        assert_eq!(sanitize_tag_name("test/branch:name"), "test_branch_name");
        assert_eq!(sanitize_tag_name("  trim  "), "trim");
        assert_eq!(sanitize_tag_name("Upper CASE"), "upper-case");
    }
}
