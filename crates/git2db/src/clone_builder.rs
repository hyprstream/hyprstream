//! Fluent builder API for cloning repositories
//!
//! Provides a git-native interface for cloning with multiple configuration options

use crate::clone_options::CloneOptions;
use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::GitRef;
use crate::registry::{Git2DB, RemoteConfig, RepoId};

/// Builder for cloning repositories with fluent configuration
///
/// Provides a chainable interface for configuring clone operations.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example(registry: &mut git2db::Git2DB) -> Result<(), Box<dyn std::error::Error>> {
/// // Basic clone
/// let id = registry.clone("https://github.com/user/repo.git")
///     .exec()
///     .await?;
///
/// // Clone with configuration
/// let id = registry.clone("https://github.com/user/repo.git")
///     .name("my-repo")
///     .branch("develop")
///     .remote("backup", "https://backup.com/repo.git")
///     .depth(1)
///     .exec()
///     .await?;
///
/// // Clone with multiple remotes (gittorrent support)
/// let id = registry.clone("https://github.com/user/repo.git")
///     .name("distributed-repo")
///     .remote("p2p", "gittorrent://peer/repo")
///     .remote("mirror", "https://mirror.com/repo.git")
///     .exec()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct CloneBuilder<'a> {
    registry: &'a mut Git2DB,
    url: String,
    name: Option<String>,
    reference: GitRef,
    depth: Option<u32>,
    remotes: Vec<(String, String)>,
}

impl<'a> CloneBuilder<'a> {
    /// Create a new clone builder
    pub(crate) fn new(registry: &'a mut Git2DB, url: String) -> Self {
        Self {
            registry,
            url,
            name: None,
            reference: GitRef::DefaultBranch,
            depth: None,
            remotes: Vec::new(),
        }
    }

    /// Set a custom name for the repository
    ///
    /// If not specified, a UUID will be used.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Checkout a specific branch
    ///
    /// Similar to `git clone --branch <name>`
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        self.reference = GitRef::Branch(branch.into());
        self
    }

    /// Checkout a specific tag
    ///
    /// Similar to `git clone --branch <tag>`
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.reference = GitRef::Tag(tag.into());
        self
    }

    /// Checkout a specific commit
    pub fn commit(mut self, oid: git2::Oid) -> Self {
        self.reference = GitRef::Commit(oid);
        self
    }

    /// Checkout a specific revspec
    ///
    /// Can be branch, tag, commit hash, or complex expressions like `HEAD~3`
    pub fn revspec(mut self, spec: impl Into<String>) -> Self {
        self.reference = GitRef::Revspec(spec.into());
        self
    }

    /// Add an additional remote
    ///
    /// The primary URL becomes "origin". Additional remotes are configured
    /// after cloning. Supports gittorrent:// URLs.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// registry.clone("https://github.com/user/repo.git")
    ///     .remote("backup", "https://backup.com/repo.git")
    ///     .remote("p2p", "gittorrent://peer/repo")
    ///     .exec()
    ///     .await?;
    /// ```
    pub fn remote(mut self, name: impl Into<String>, url: impl Into<String>) -> Self {
        self.remotes.push((name.into(), url.into()));
        self
    }

    /// Create a shallow clone with specific depth
    ///
    /// Similar to `git clone --depth <n>`
    pub fn depth(mut self, depth: u32) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Execute the clone operation
    ///
    /// Returns the RepoId of the newly cloned repository.
    pub async fn exec(self) -> Git2DBResult<RepoId> {
        // Generate repository ID
        let repo_id = RepoId::new();

        // Determine target path
        let target_path = if let Some(name) = &self.name {
            self.registry.base_dir().join(name)
        } else {
            self.registry.base_dir().join(repo_id.to_string())
        };

        // Validate target doesn't exist
        if target_path.exists() {
            return Err(Git2DBError::configuration(format!(
                "Target path already exists: {}",
                target_path.display()
            )));
        }

        // Perform the clone using GitManager
        let manager = crate::manager::GitManager::global();

        // Start with default options (includes auth callbacks)
        let mut options = manager.default_clone_options();

        // Override with builder-specific settings
        if let Some(depth) = self.depth {
            options.shallow = true;
            options.depth = Some(depth as i32);
        }

        // Set branch for initial checkout if specified
        if let GitRef::Branch(ref branch_name) = self.reference {
            options.branch = Some(branch_name.clone());
        }

        // Clone the repository
        let repo = manager
            .clone_repository(&self.url, &target_path, Some(options))
            .await
            .map_err(|e| {
                Git2DBError::repository(&target_path, format!("Failed to clone repository: {}", e))
            })?;

        // Checkout the specified reference if not default
        if !matches!(self.reference, GitRef::DefaultBranch) {
            let reference_str = match &self.reference {
                GitRef::Branch(b) => b.clone(),
                GitRef::Tag(t) => t.clone(),
                GitRef::Commit(oid) => oid.to_string(),
                GitRef::Revspec(spec) => spec.clone(),
                GitRef::DefaultBranch => unreachable!(),
            };

            // Use libgit2 to checkout
            let obj = repo.revparse_single(&reference_str).map_err(|e| {
                Git2DBError::reference(&reference_str, format!("Failed to resolve: {}", e))
            })?;

            repo.checkout_tree(&obj, None)
                .map_err(|e| Git2DBError::internal(format!("Failed to checkout tree: {}", e)))?;

            // Update HEAD
            match &self.reference {
                GitRef::Branch(branch) => {
                    repo.set_head(&format!("refs/heads/{}", branch))
                        .map_err(|e| {
                            Git2DBError::internal(format!("Failed to update HEAD: {}", e))
                        })?;
                }
                _ => {
                    repo.set_head_detached(obj.id()).map_err(|e| {
                        Git2DBError::internal(format!("Failed to detach HEAD: {}", e))
                    })?;
                }
            }
        }

        // Add additional remotes
        for (remote_name, remote_url) in &self.remotes {
            repo.remote(remote_name, remote_url).map_err(|e| {
                Git2DBError::configuration(format!("Failed to add remote '{}': {}", remote_name, e))
            })?;
        }

        // Fetch LFS files if the repository uses LFS
        // Since XET is disabled, fallback to standard Git LFS
        Self::fetch_lfs_files(&target_path).await?;

        // Build remote configs (origin + additional)
        let mut remote_configs = vec![RemoteConfig {
            name: "origin".to_string(),
            url: self.url.clone(),
            fetch_refs: vec!["+refs/heads/*:refs/remotes/origin/*".to_string()],
        }];

        for (name, url) in self.remotes {
            remote_configs.push(RemoteConfig {
                name: name.clone(),
                url,
                fetch_refs: vec![format!("+refs/heads/*:refs/remotes/{}/*", name)],
            });
        }

        // Register in Git2DB with full configuration
        self.registry
            .register_repository_full(
                repo_id.clone(),
                self.name.clone(),
                self.url,
                target_path,
                self.reference,
                remote_configs,
            )
            .await?;

        Ok(repo_id)
    }

    /// Fetch LFS files if the repository uses Git LFS
    ///
    /// This checks for .gitattributes with LFS configuration and runs `git lfs pull`
    async fn fetch_lfs_files(repo_path: &std::path::Path) -> Git2DBResult<()> {
        use tokio::process::Command;

        // Check if repository uses LFS by looking for .gitattributes with "filter=lfs"
        let gitattributes_path = repo_path.join(".gitattributes");

        let uses_lfs = if gitattributes_path.exists() {
            tokio::fs::read_to_string(&gitattributes_path)
                .await
                .map(|content| content.contains("filter=lfs"))
                .unwrap_or(false)
        } else {
            false
        };

        if !uses_lfs {
            tracing::debug!("Repository does not use Git LFS, skipping LFS fetch");
            return Ok(());
        }

        tracing::info!("Repository uses Git LFS, fetching LFS files...");

        // Run git lfs pull
        let output = Command::new("git")
            .args(&["lfs", "pull"])
            .current_dir(repo_path)
            .output()
            .await
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to run 'git lfs pull': {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!("git lfs pull failed: {}", stderr);
            return Err(Git2DBError::internal(format!(
                "git lfs pull failed: {}",
                stderr
            )));
        }

        tracing::info!("Successfully fetched LFS files");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_fluent_api() {
        // This test just validates the fluent API compiles correctly
        fn _example(registry: &mut Git2DB) {
            let _builder = registry
                .clone("https://github.com/user/repo.git")
                .name("my-repo")
                .branch("main")
                .remote("backup", "https://backup.com/repo.git")
                .depth(1);
        }
    }
}
