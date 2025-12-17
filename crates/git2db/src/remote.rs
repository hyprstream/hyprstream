//! Remote management for tracked repositories
//!
//! Provides git-native remote operations similar to `git remote`.
//!
//! # Metadata Sync Pattern
//!
//! Remote operations (`add`, `remove`, `set_url`, `rename`) persist changes to
//! `.git/config` immediately, but do NOT automatically update registry metadata.
//! To sync remotes to registry metadata, call `registry.sync_repository_remotes()`
//! after making changes:
//!
//! ```rust,no_run
//! # async fn example(registry: &mut git2db::Git2DB, repo_id: &git2db::RepoId) -> Result<(), Box<dyn std::error::Error>> {
//! let repo = registry.repo(repo_id)?;
//! let remote_mgr = repo.remote();
//!
//! // Make changes (persisted to .git/config immediately)
//! remote_mgr.add("backup", "https://backup.com/repo.git").await?;
//! remote_mgr.set_url("origin", "https://new-url.com/repo.git").await?;
//!
//! // Sync all changes to registry metadata in one call
//! registry.sync_repository_remotes(repo_id).await?;
//! # Ok(())
//! # }
//! ```
//!
//! This "pull" model allows batching multiple remote changes with a single metadata
//! commit, rather than committing after each operation.

use crate::errors::{Git2DBError, Git2DBResult};
use crate::registry::{Git2DB, RepoId};
use crate::repo_accessor::RepositoryAccessor;
use git2::Repository;

/// Manager for repository remotes
///
/// Provides operations similar to `git remote` commands.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example(registry: &git2db::Git2DB, repo_id: &git2db::RepoId) -> Result<(), Box<dyn std::error::Error>> {
/// let repo = registry.repo(repo_id)?;
/// let remote_mgr = repo.remote();
///
/// // Add remotes
/// remote_mgr.add("origin", "https://github.com/user/repo.git").await?;
/// remote_mgr.add("p2p", "gittorrent://peer/repo").await?;
/// remote_mgr.add("backup", "https://backup.com/repo.git").await?;
///
/// // List remotes
/// for remote in remote_mgr.list().await? {
///     println!("{}: {}", remote.name, remote.url);
/// }
///
/// // Change URL
/// remote_mgr.set_url("origin", "https://github.com/newuser/repo.git").await?;
///
/// // Remove remote
/// remote_mgr.remove("backup").await?;
/// # Ok(())
/// # }
/// ```
pub struct RemoteManager<'a> {
    registry: &'a Git2DB,
    repo_id: RepoId,
}

impl<'a> RepositoryAccessor for RemoteManager<'a> {
    fn registry(&self) -> &Git2DB {
        self.registry
    }

    fn repo_id(&self) -> &RepoId {
        &self.repo_id
    }
}

impl<'a> RemoteManager<'a> {
    /// Create a new remote manager
    pub(crate) fn new(registry: &'a Git2DB, repo_id: RepoId) -> Self {
        Self { registry, repo_id }
    }

    /// Add a new remote
    ///
    /// Similar to `git remote add <name> <url>`
    ///
    /// Note: After adding a remote, call `registry.sync_repository_remotes(repo_id).await?`
    /// to sync the change to registry metadata.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # async fn example(registry: &mut git2db::Git2DB, repo_id: &git2db::RepoId) -> Result<(), Box<dyn std::error::Error>> {
    /// let repo = registry.repo(repo_id)?;
    /// let remote_mgr = repo.remote();
    /// remote_mgr.add("backup", "https://backup.com/repo.git").await?;
    /// // Sync to metadata
    /// registry.sync_repository_remotes(repo_id).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add(&self, name: &str, url: &str) -> Git2DBResult<()> {
        let path = self.repo_path()?;
        let name = name.to_string();
        let url = url.to_string();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {}", e))
            })?;

            // Check if remote already exists
            if repo.find_remote(&name).is_ok() {
                return Err(Git2DBError::configuration(format!(
                    "Remote '{}' already exists",
                    name
                )));
            }

            // Add remote to git config
            repo.remote(&name, &url).map_err(|e| {
                Git2DBError::configuration(format!("Failed to add remote '{}': {}", name, e))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))??;

        Ok(())
    }

    /// Remove a remote
    ///
    /// Similar to `git remote remove <name>`
    ///
    /// Note: After removing a remote, call `registry.sync_repository_remotes(repo_id).await?`
    /// to sync the change to registry metadata.
    pub async fn remove(&self, name: &str) -> Git2DBResult<()> {
        let path = self.repo_path()?;
        let name = name.to_string();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {}", e))
            })?;

            // Check if remote exists
            repo.find_remote(&name)
                .map_err(|_| Git2DBError::configuration(format!("Remote '{}' not found", name)))?;

            // Remove from git config
            repo.remote_delete(&name).map_err(|e| {
                Git2DBError::configuration(format!("Failed to remove remote '{}': {}", name, e))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))??;

        Ok(())
    }

    /// Change a remote's URL
    ///
    /// Similar to `git remote set-url <name> <url>`
    ///
    /// Note: After changing a remote URL, call `registry.sync_repository_remotes(repo_id).await?`
    /// to sync the change to registry metadata.
    pub async fn set_url(&self, name: &str, url: &str) -> Git2DBResult<()> {
        let path = self.repo_path()?;
        let name = name.to_string();
        let url = url.to_string();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {}", e))
            })?;

            // Verify remote exists
            repo.find_remote(&name)
                .map_err(|_| Git2DBError::configuration(format!("Remote '{}' not found", name)))?;

            // Update URL
            repo.remote_set_url(&name, &url).map_err(|e| {
                Git2DBError::configuration(format!(
                    "Failed to set URL for remote '{}': {}",
                    name, e
                ))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))??;

        Ok(())
    }

    /// Rename a remote
    ///
    /// Similar to `git remote rename <old> <new>`
    ///
    /// Note: After renaming a remote, call `registry.sync_repository_remotes(repo_id).await?`
    /// to sync the change to registry metadata.
    pub async fn rename(&self, old_name: &str, new_name: &str) -> Git2DBResult<()> {
        let path = self.repo_path()?;
        let old_name = old_name.to_string();
        let new_name = new_name.to_string();

        tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {}", e))
            })?;

            // Verify old remote exists
            repo.find_remote(&old_name).map_err(|_| {
                Git2DBError::configuration(format!("Remote '{}' not found", old_name))
            })?;

            // Check new name doesn't exist
            if repo.find_remote(&new_name).is_ok() {
                return Err(Git2DBError::configuration(format!(
                    "Remote '{}' already exists",
                    new_name
                )));
            }

            // Rename remote
            repo.remote_rename(&old_name, &new_name).map_err(|e| {
                Git2DBError::configuration(format!(
                    "Failed to rename remote '{}' to '{}': {}",
                    old_name, new_name, e
                ))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))??;

        Ok(())
    }

    /// List all remotes
    ///
    /// Similar to `git remote -v`
    pub async fn list(&self) -> Git2DBResult<Vec<Remote>> {
        let path = self.repo_path()?;

        tokio::task::spawn_blocking(move || -> Git2DBResult<Vec<Remote>> {
            let repo = Repository::open(&path).map_err(|e| {
                Git2DBError::repository(&path, format!("Failed to open repository: {}", e))
            })?;

            let remote_names = repo
                .remotes()
                .map_err(|e| Git2DBError::internal(format!("Failed to list remotes: {}", e)))?;

            let mut remotes = Vec::new();

            for name in remote_names.iter().flatten() {
                if let Ok(remote) = repo.find_remote(name) {
                    let url = remote.url().unwrap_or("").to_string();
                    let fetch_refspec = remote
                        .fetch_refspecs()
                        .map_err(|_| Git2DBError::internal("Failed to get fetch refspecs"))?;

                    let fetch_refs: Vec<String> = fetch_refspec
                        .iter()
                        .flatten()
                        .map(|s| s.to_string())
                        .collect();

                    remotes.push(Remote {
                        name: name.to_string(),
                        url,
                        fetch_refs,
                        push_url: remote.pushurl().map(|s| s.to_string()),
                    });
                }
            }

            Ok(remotes)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))?
    }

    /// Get default remote (usually "origin")
    pub async fn default(&self) -> Git2DBResult<Option<Remote>> {
        let remotes = self.list().await?;

        // Look for "origin" first
        if let Some(origin) = remotes.iter().find(|r| r.name == "origin") {
            return Ok(Some(origin.clone()));
        }

        // Return first remote if any
        Ok(remotes.into_iter().next())
    }
}

/// Remote information
#[derive(Debug, Clone)]
pub struct Remote {
    pub name: String,
    pub url: String,
    pub fetch_refs: Vec<String>,
    pub push_url: Option<String>,
}

impl Remote {
    /// Check if this is the default remote (origin)
    pub fn is_default(&self) -> bool {
        self.name == "origin"
    }

    /// Get display URL (prefer push URL if set)
    pub fn display_url(&self) -> &str {
        self.push_url.as_deref().unwrap_or(&self.url)
    }
}
