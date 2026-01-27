//! Staging area management for tracked repositories
//!
//! Provides git-native staging operations similar to `git add`, `git rm`, etc.

use crate::errors::{Git2DBError, Git2DBResult};
use crate::registry::{Git2DB, RepoId};
use crate::repo_accessor::RepositoryAccessor;
use git2::{IndexAddOption, Repository, Status, StatusOptions};
use std::path::{Path, PathBuf};

/// Manager for repository staging area
///
/// Provides operations similar to `git add` and `git rm` commands.
///
/// # Examples
///
/// ```rust,ignore
/// // Add file to staging area
/// repo.staging().add("src/main.rs")?;
///
/// // Add all changes
/// repo.staging().add_all()?;
///
/// // Remove file from staging area
/// repo.staging().remove("old.txt")?;
///
/// // Check what's staged
/// let staged = repo.staging().status()?;
/// for file in staged {
///     println!("{:?}: {}", file.status, file.path);
/// }
/// # Ok(())
/// # }
/// ```
pub struct StageManager<'a> {
    registry: &'a Git2DB,
    repo_id: RepoId,
}

/// Staged file information
#[derive(Debug, Clone)]
pub struct StagedFile {
    /// File path relative to repository root
    pub path: PathBuf,
    /// Git status flags
    pub status: Status,
    /// Whether file is staged for commit
    pub is_staged: bool,
}

impl<'a> RepositoryAccessor for StageManager<'a> {
    fn registry(&self) -> &Git2DB {
        self.registry
    }

    fn repo_id(&self) -> &RepoId {
        &self.repo_id
    }
}

impl<'a> StageManager<'a> {
    /// Create a new stage manager
    pub(crate) fn new(registry: &'a Git2DB, repo_id: RepoId) -> Self {
        Self { registry, repo_id }
    }

    /// Add a file or directory to the staging area
    ///
    /// Similar to `git add <path>`
    pub async fn add(&self, path: impl AsRef<Path>) -> Git2DBResult<()> {
        let repo_path = self.repo_path()?;
        let file_path = path.as_ref().to_path_buf();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            let mut index = repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {e}")))?;

            index.add_path(&file_path).map_err(|e| {
                Git2DBError::internal(format!(
                    "Failed to add path '{}': {}",
                    file_path.display(),
                    e
                ))
            })?;

            index
                .write()
                .map_err(|e| Git2DBError::internal(format!("Failed to write index: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Add all changes to the staging area
    ///
    /// Similar to `git add -A` or `git add .`
    pub async fn add_all(&self) -> Git2DBResult<()> {
        let repo_path = self.repo_path()?;

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            let mut index = repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {e}")))?;

            // Add all files (new, modified, deleted)
            index
                .add_all(["."].iter(), IndexAddOption::DEFAULT, None)
                .map_err(|e| Git2DBError::internal(format!("Failed to add all: {e}")))?;

            index
                .write()
                .map_err(|e| Git2DBError::internal(format!("Failed to write index: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Remove a file from the staging area and working directory
    ///
    /// Similar to `git rm <path>`
    pub async fn remove(&self, path: impl AsRef<Path>) -> Git2DBResult<()> {
        let repo_path = self.repo_path()?;
        let file_path = path.as_ref().to_path_buf();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            let mut index = repo
                .index()
                .map_err(|e| Git2DBError::internal(format!("Failed to get index: {e}")))?;

            index.remove_path(&file_path).map_err(|e| {
                Git2DBError::internal(format!(
                    "Failed to remove path '{}': {}",
                    file_path.display(),
                    e
                ))
            })?;

            index
                .write()
                .map_err(|e| Git2DBError::internal(format!("Failed to write index: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Unstage a file (remove from staging area but keep changes)
    ///
    /// Similar to `git reset HEAD <path>`
    pub async fn unstage(&self, path: impl AsRef<Path>) -> Git2DBResult<()> {
        let repo_path = self.repo_path()?;
        let file_path = path.as_ref().to_path_buf();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            // Get HEAD commit
            let head = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
            })?;

            let head_commit = head.peel_to_commit().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {e}"))
            })?;

            // Reset the path to HEAD
            repo.reset_default(Some(head_commit.as_object()), [&file_path].iter())
                .map_err(|e| {
                    Git2DBError::internal(format!(
                        "Failed to unstage '{}': {}",
                        file_path.display(),
                        e
                    ))
                })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Reset all staged changes
    ///
    /// Similar to `git reset HEAD`
    pub async fn reset(&self) -> Git2DBResult<()> {
        let repo_path = self.repo_path()?;

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            let head = repo.head().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD: {e}"))
            })?;

            let head_commit = head.peel_to_commit().map_err(|e| {
                Git2DBError::reference("HEAD", format!("Failed to get HEAD commit: {e}"))
            })?;

            repo.reset(head_commit.as_object(), git2::ResetType::Mixed, None)
                .map_err(|e| {
                    Git2DBError::internal(format!("Failed to reset staging area: {e}"))
                })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Get status of all files in the repository
    ///
    /// Returns information about staged and unstaged changes.
    pub async fn status(&self) -> Git2DBResult<Vec<StagedFile>> {
        let repo_path = self.repo_path()?;

        tokio::task::spawn_blocking(move || -> Git2DBResult<Vec<StagedFile>> {
            let repo = Repository::open(&repo_path).map_err(|e| {
                Git2DBError::repository(&repo_path, format!("Failed to open repository: {e}"))
            })?;

            let mut opts = StatusOptions::new();
            opts.include_untracked(true);
            opts.recurse_untracked_dirs(true);

            let statuses = repo.statuses(Some(&mut opts)).map_err(|e| {
                Git2DBError::internal(format!("Failed to get repository status: {e}"))
            })?;

            let mut files = Vec::new();
            for entry in statuses.iter() {
                if let Some(path) = entry.path() {
                    let status = entry.status();
                    let is_staged = status.intersects(
                        Status::INDEX_NEW
                            | Status::INDEX_MODIFIED
                            | Status::INDEX_DELETED
                            | Status::INDEX_RENAMED
                            | Status::INDEX_TYPECHANGE,
                    );

                    files.push(StagedFile {
                        path: PathBuf::from(path),
                        status,
                        is_staged,
                    });
                }
            }

            Ok(files)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))?
    }

    /// Get only staged files
    ///
    /// Returns files that are ready to be committed.
    pub async fn staged_files(&self) -> Git2DBResult<Vec<StagedFile>> {
        let all_files = self.status().await?;
        Ok(all_files.into_iter().filter(|f| f.is_staged).collect())
    }

    /// Check if staging area is empty
    pub async fn is_empty(&self) -> Git2DBResult<bool> {
        Ok(self.staged_files().await?.is_empty())
    }
}

#[cfg(test)]
mod tests {
    

    // Tests will be added for staging operations
}
