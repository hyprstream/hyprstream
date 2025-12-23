//! Fluent builder API for cloning repositories
//!
//! Provides a git-native interface for cloning with multiple configuration options

use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::GitRef;
use crate::registry::{Git2DB, RemoteConfig, RepoId};

/// Automatically initialize XET for URLs that have XET support
///
/// XET endpoint resolution:
/// 1. Environment variable (XETHUB_ENDPOINT/GIT2DB_XET_ENDPOINT) - always takes priority
/// 2. URL pattern matching for known providers (e.g., huggingface.co, hf.co)
/// 3. None - XET disabled, will use git-lfs
#[cfg(feature = "xet-storage")]
async fn maybe_init_xet_for_url(url: &str) -> Git2DBResult<()> {
    // Check if XET is already initialized
    if crate::xet_filter::is_initialized() {
        tracing::debug!(url = %url, "XET filter already initialized");
        return Ok(());
    }

    // Get XET config for this URL (if any)
    let Some(config) = crate::xet_filter::XetConfig::for_url(url) else {
        tracing::debug!(url = %url, "No XET endpoint configured for URL");
        return Ok(());
    };

    // Token is optional for public repos, but log a hint for private repos
    if config.token.is_none() {
        tracing::info!(
            url = %url,
            "No XET token found. Public repos will work, private repos require authentication. \
             Set XETHUB_TOKEN or HF_TOKEN environment variable."
        );
    }

    // Initialize XET - failures are non-fatal (fallback to Git LFS)
    tracing::info!(url = %url, "Auto-initializing XET filter");

    match crate::xet_filter::initialize(config).await {
        Ok(()) => {
            tracing::info!(url = %url, "XET filter initialized successfully");
            Ok(())
        }
        Err(e) => {
            // Non-fatal: fallback to Git LFS
            tracing::warn!(url = %url, error = %e, "XET init failed, falling back to Git LFS");
            Ok(())
        }
    }
}

#[cfg(not(feature = "xet-storage"))]
async fn maybe_init_xet_for_url(_url: &str) -> Git2DBResult<()> {
    Ok(())
}

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
        // Auto-initialize XET for URLs with XET support (e.g., HuggingFace)
        maybe_init_xet_for_url(&self.url).await?;

        // Generate repository ID
        let repo_id = RepoId::new();

        // Determine repository name
        let repo_name = self
            .name
            .clone()
            .unwrap_or_else(|| repo_id.to_string());

        // Validate repository name for security
        validate_repo_name(&repo_name)?;

        // Build paths using safe_path to prevent traversal
        let models_dir = self.registry.base_dir();

        // models/{name}/
        let repo_dir = safe_path::scoped_join(models_dir, &repo_name)
            .map_err(|e| Git2DBError::configuration(format!("Invalid repository name: {}", e)))?;

        // Check if already fully registered (idempotent return)
        if let Some(tracked) = self.registry.get_by_name(&repo_name) {
            tracing::info!("Repository '{}' already registered with ID {}", repo_name, tracked.id);
            return Ok(tracked.id.clone());
        }

        // Build paths (repo_name is already validated, so simple join is safe)
        let bare_repo_name = format!("{}.git", &repo_name);
        let bare_repo_path = repo_dir.join(&bare_repo_name);
        let worktrees_dir = repo_dir.join("worktrees");

        // Check for existing bare repo (resume case)
        let existing_bare_repo = if repo_dir.exists() {
            match git2::Repository::open_bare(&bare_repo_path) {
                Ok(repo) => {
                    tracing::info!("Resuming clone: valid bare repo found at {:?}", bare_repo_path);
                    Some(repo)
                }
                Err(_) => {
                    // Corrupted/incomplete - cleanup and restart
                    tracing::warn!("Removing incomplete/corrupted clone at {:?}", repo_dir);
                    std::fs::remove_dir_all(&repo_dir).map_err(|e| {
                        Git2DBError::repository(&repo_dir, format!("Failed to cleanup incomplete clone: {}", e))
                    })?;
                    None
                }
            }
        } else {
            None
        };

        // Create directory structure only if fresh clone
        if existing_bare_repo.is_none() {
            std::fs::create_dir_all(&repo_dir).map_err(|e| {
                Git2DBError::repository(&repo_dir, format!("Failed to create repo directory: {}", e))
            })?;
            std::fs::create_dir_all(&worktrees_dir).map_err(|e| {
                Git2DBError::repository(&worktrees_dir, format!("Failed to create worktrees directory: {}", e))
            })?;
        }

        // Clone or reuse existing bare repo
        let bare_repo = if let Some(repo) = existing_bare_repo {
            repo
        } else {
            // Perform the clone using git2 directly for bare repos
            tracing::info!("Cloning repository '{}' as bare to {:?}", repo_name, bare_repo_path);

            // Get default clone options from GitManager for authentication
            let clone_options = crate::manager::GitManager::global().default_clone_options();

            let url_clone = self.url.clone();
            let bare_path_clone = bare_repo_path.clone();

            tokio::task::spawn_blocking(move || -> Git2DBResult<git2::Repository> {
                // Convert to git2 options inside spawn_blocking (where non-Send is acceptable)
                let mut git2_options = clone_options.to_git2_options();

                let mut builder = git2::build::RepoBuilder::new();

                // Set up bare clone
                builder.bare(true);

                // Apply fetch options with authentication callbacks from GitManager
                builder.fetch_options(git2_options.create_fetch_options()
                    .map_err(|e| Git2DBError::internal(format!("Failed to create fetch options: {}", e)))?);

                // Perform the clone
                let repo = builder.clone(&url_clone, &bare_path_clone)
                    .map_err(|e| Git2DBError::repository(
                        &bare_path_clone,
                        format!("Failed to clone bare repository: {}", e)
                    ))?;

                Ok(repo)
            })
            .await
            .map_err(|e| Git2DBError::internal(format!("Task join error: {}", e)))??
        };

        // Add additional remotes to bare repo
        for (remote_name, remote_url) in &self.remotes {
            bare_repo.remote(remote_name, remote_url).map_err(|e| {
                Git2DBError::configuration(format!("Failed to add remote '{}': {}", remote_name, e))
            })?;
        }

        // Get the default branch from bare repo
        let default_branch = get_default_branch(&bare_repo)?;
        tracing::debug!("Default branch detected: {}", default_branch);

        // Create initial worktree for the default branch
        let initial_worktree = worktrees_dir.join(&default_branch);

        // Create parent directories if default branch has hierarchy
        if let Some(parent) = initial_worktree.parent() {
            if parent != worktrees_dir {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Git2DBError::configuration(format!("Failed to create worktree parent directories: {}", e))
                })?;
            }
        }

        tracing::info!("Creating default worktree '{}' at {:?}", default_branch, initial_worktree);

        // Get storage driver from registry for worktree creation
        let driver = self.registry.storage_driver().clone();

        // Create worktree from bare repo
        create_worktree(&driver, &bare_repo_path, &initial_worktree, &default_branch).await?;

        // If a specific reference was requested and it's different from default, create another worktree
        let checkout_ref = match &self.reference {
            GitRef::DefaultBranch => default_branch.clone(),
            GitRef::Branch(b) => b.clone(),
            GitRef::Tag(t) => t.clone(),
            GitRef::Commit(oid) => oid.to_string(),
            GitRef::Revspec(spec) => spec.clone(),
        };

        if checkout_ref != default_branch && !matches!(self.reference, GitRef::DefaultBranch) {
            // Create additional worktree for the requested reference
            let ref_worktree_path = worktrees_dir.join(&checkout_ref);

            // Create parent directories for hierarchical branches
            if let Some(parent) = ref_worktree_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Git2DBError::configuration(format!("Failed to create worktree parent directories: {}", e))
                })?;
            }

            tracing::info!("Creating worktree for ref '{}'", checkout_ref);
            create_worktree(&driver, &bare_repo_path, &ref_worktree_path, &checkout_ref).await?;
        }

        // Note: LFS files are automatically fetched by GitManager::create_worktree()
        // Fetch is always atomic - failures trigger automatic worktree rollback

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

        // Register in Git2DB with full configuration (registry tracks the bare repo)
        self.registry
            .register_repository_internal(
                repo_id.clone(),
                self.name.clone(),
                self.url,
                bare_repo_path, // Registry tracks bare repo, not worktree
                self.reference,
                remote_configs,
                std::collections::HashMap::new(), // No metadata for cloned repos by default
            )
            .await?;

        Ok(repo_id)
    }
}

/// Validate repository name to prevent path traversal
fn validate_repo_name(name: &str) -> Git2DBResult<()> {
    // Check for empty name
    if name.is_empty() {
        return Err(Git2DBError::configuration("Repository name cannot be empty"));
    }

    // Check for path separators and parent directory references
    if name.contains('/') || name.contains("..") {
        return Err(Git2DBError::configuration("Repository name cannot contain path separators or parent references"));
    }

    Ok(())
}

/// Get the default branch from a bare repository
fn get_default_branch(repo: &git2::Repository) -> Git2DBResult<String> {
    // Try to get from HEAD
    if let Ok(head) = repo.head() {
        if let Some(name) = head.shorthand() {
            return Ok(name.to_string());
        }
    }

    // Try to get from remote HEAD
    if let Ok(remote) = repo.find_remote("origin") {
        if let Ok(buf) = remote.default_branch() {
            if let Some(s) = buf.as_str() {
                // Remove refs/heads/ prefix
                let branch = s.strip_prefix("refs/heads/").unwrap_or(s);
                return Ok(branch.to_string());
            }
        }
    }

    // Fallback to common defaults
    for default in &["main", "master"] {
        if repo.find_branch(default, git2::BranchType::Local).is_ok() {
            return Ok(default.to_string());
        }
    }

    // Last resort
    Ok("main".to_string())
}

/// Create a worktree from a bare repository using the storage driver
async fn create_worktree(
    driver: &std::sync::Arc<dyn crate::storage::Driver>,
    bare_repo_path: &std::path::Path,
    worktree_path: &std::path::Path,
    branch: &str,
) -> Git2DBResult<()> {
    // Ensure parent exists
    if let Some(parent) = worktree_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Git2DBError::configuration(format!("Failed to create worktree parent directories: {}", e))
        })?;
    }

    // Use storage driver (strict: errors if path exists)
    let opts = crate::storage::DriverOpts {
        base_repo: bare_repo_path.to_path_buf(),
        worktree_path: worktree_path.to_path_buf(),
        ref_spec: branch.to_string(),
    };

    match driver.create_worktree(&opts).await {
        Ok(_) => {
            // LFS fetch (idempotent)
            crate::repository_handle::RepositoryHandle::fetch_lfs_files(worktree_path).await?;
            tracing::info!("Created worktree at {:?} for branch '{}'", worktree_path, branch);
            Ok(())
        }
        Err(e) if e.is_worktree_exists() => {
            // Clone resume: worktree already exists, just fetch LFS
            tracing::info!("Worktree already exists at {:?}, fetching LFS files", worktree_path);
            crate::repository_handle::RepositoryHandle::fetch_lfs_files(worktree_path).await?;
            Ok(())
        }
        Err(e) => Err(Git2DBError::repository(
            worktree_path,
            format!("Failed to create worktree for branch '{}': {}", branch, e)
        )),
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

    #[test]
    fn test_validate_repo_name() {
        // Valid names
        assert!(validate_repo_name("my-repo").is_ok());
        assert!(validate_repo_name("repo_123").is_ok());
        assert!(validate_repo_name("MyRepo").is_ok());
        assert!(validate_repo_name(".hidden").is_ok()); // Hidden files are fine on Linux

        // Invalid names - only path traversal concerns
        assert!(validate_repo_name("").is_err());
        assert!(validate_repo_name("my/repo").is_err());
        assert!(validate_repo_name("../etc").is_err());
        assert!(validate_repo_name("repo..name").is_err());
    }
}
