//! Git helper functions for common operations
//!
//! Simple wrappers around git2 for tasks not yet in git2db API

use anyhow::{Context, Result};
use git2db::GitManager;
use std::path::Path;

/// Create a lightweight Git tag pointing to HEAD
///
/// This is a simple helper for checkpoint tagging and other marking operations.
/// For more complex tag management, use git2 directly or wait for git2db tag API.
///
/// # Example
///
/// ```rust,ignore
/// create_tag(model_path, "checkpoint-step-1000")?;
/// ```
pub fn create_tag(repo_path: impl AsRef<Path>, tag_name: &str) -> Result<()> {
    let repo_path = repo_path.as_ref();

    // Get repository handle via GitManager (uses caching)
    let repo = GitManager::global()
        .get_repository(repo_path)
        .context("Failed to get repository")?
        .open()
        .context("Failed to open repository")?;

    // Get current HEAD commit
    let commit = repo
        .head()
        .context("Failed to get HEAD")?
        .peel_to_commit()
        .context("HEAD is not a commit")?;

    // Save commit ID before moving
    let commit_id = commit.id();

    // Create lightweight tag (overwrites if exists)
    repo.tag_lightweight(tag_name, &commit.into_object(), true)
        .with_context(|| format!("Failed to create tag '{tag_name}'"))?;

    tracing::debug!("Created tag '{}' at {}", tag_name, commit_id);

    Ok(())
}

/// Sanitize a string for use as a Git tag name
///
/// Replaces invalid characters with safe alternatives:
/// - Whitespace → '-'
/// - Path separators and special chars → '_'
/// - Control characters → '_'
/// - Trims leading/trailing punctuation
/// - Converts to lowercase
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
