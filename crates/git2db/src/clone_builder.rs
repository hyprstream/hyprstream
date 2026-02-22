//! Fluent builder API for cloning repositories
//!
//! Provides a git-native interface for cloning with multiple configuration options.
//!
//! [`CloneBuilder`] takes ownership of an `Arc<RwLock<Git2DB>>` and manages locks
//! internally for optimal performance - releasing the lock during network I/O.

use crate::callback_config::CallbackConfig;
use crate::errors::{Git2DBError, Git2DBResult};
use crate::references::GitRef;
use crate::registry::{Git2DB, RemoteConfig, RepoId};
use crate::storage::Driver;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

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

/// Builder for cloning repositories with fluent configuration.
///
/// Manages locks internally for minimal lock duration:
/// - Brief read lock for configuration (base_dir, existing repo check)
/// - No lock during network clone (the slow part)
/// - Brief write lock for final registration
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use git2db::Git2DB;
/// use std::sync::Arc;
/// use tokio::sync::RwLock;
///
/// let registry = Arc::new(RwLock::new(Git2DB::open("/models").await?));
///
/// // Basic clone
/// let id = git2db::CloneBuilder::new(Arc::clone(&registry), "https://github.com/user/repo.git")
///     .exec()
///     .await?;
///
/// // Clone with configuration
/// let id = git2db::CloneBuilder::new(Arc::clone(&registry), "https://github.com/user/repo.git")
///     .name("my-repo")
///     .branch("develop")
///     .depth(1)
///     .exec()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct CloneBuilder {
    registry: Arc<RwLock<Git2DB>>,
    url: String,
    name: Option<String>,
    reference: GitRef,
    depth: Option<u32>,
    remotes: Vec<(String, String)>,
    callback_config: Option<CallbackConfig>,
}

impl CloneBuilder {
    /// Create a new clone builder.
    pub fn new(registry: Arc<RwLock<Git2DB>>, url: impl Into<String>) -> Self {
        Self {
            registry,
            url: url.into(),
            name: None,
            reference: GitRef::DefaultBranch,
            depth: None,
            remotes: Vec::new(),
            callback_config: None,
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
    /// CloneBuilder::new(Arc::clone(&registry), "https://github.com/user/repo.git")
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

    /// Set callback configuration for progress reporting and authentication
    ///
    /// This allows custom progress callbacks during clone operations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use git2db::callback_config::{CallbackConfig, ProgressConfig, ProgressReporter};
    /// use std::sync::Arc;
    ///
    /// struct MyReporter;
    /// impl ProgressReporter for MyReporter {
    ///     fn report(&self, stage: &str, current: usize, total: usize) {
    ///         println!("{}: {}/{}", stage, current, total);
    ///     }
    /// }
    ///
    /// let config = CallbackConfig::new()
    ///     .with_progress(ProgressConfig::Channel(Arc::new(MyReporter)));
    ///
    /// CloneBuilder::new(Arc::clone(&registry), "https://github.com/user/repo.git")
    ///     .callback_config(config)
    ///     .exec()
    ///     .await?;
    /// ```
    pub fn callback_config(mut self, config: CallbackConfig) -> Self {
        self.callback_config = Some(config);
        self
    }

    /// Execute the clone operation with minimal lock duration.
    ///
    /// Lock strategy:
    /// 1. Read lock: Get config (base_dir, check existing, storage driver)
    /// 2. No lock: Perform network clone (the slow part)
    /// 3. Write lock: Register repository
    pub async fn exec(mut self) -> Git2DBResult<RepoId> {
        // Auto-initialize XET for URLs with XET support
        maybe_init_xet_for_url(&self.url).await?;

        // Generate repository ID
        let repo_id = RepoId::new();

        // Determine repository name
        let repo_name = self.name.clone().unwrap_or_else(|| repo_id.to_string());
        validate_repo_name(&repo_name)?;

        // ===== Phase 1: Read lock for configuration =====
        let (models_dir, driver): (PathBuf, Arc<dyn Driver>) = {
            let registry = self.registry.read().await;

            // Check if already registered (idempotent return)
            if let Some(tracked) = registry.get_by_name(&repo_name) {
                tracing::info!("Repository '{}' already registered with ID {}", repo_name, tracked.id);
                return Ok(tracked.id.clone());
            }

            (
                registry.base_dir().to_path_buf(),
                registry.storage_driver().clone(),
            )
        }; // Read lock released

        // Build paths
        let repo_dir = hyprstream_containedfs::contained_join(&models_dir, &repo_name)
            .map_err(|e| Git2DBError::configuration(format!("Invalid repository name: {e}")))?;
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
                    tracing::warn!("Removing incomplete/corrupted clone at {:?}", repo_dir);
                    std::fs::remove_dir_all(&repo_dir).map_err(|e| {
                        Git2DBError::repository(&repo_dir, format!("Failed to cleanup incomplete clone: {e}"))
                    })?;
                    None
                }
            }
        } else {
            None
        };

        // Create directory structure if fresh clone
        if existing_bare_repo.is_none() {
            std::fs::create_dir_all(&repo_dir).map_err(|e| {
                Git2DBError::repository(&repo_dir, format!("Failed to create repo directory: {e}"))
            })?;
            std::fs::create_dir_all(&worktrees_dir).map_err(|e| {
                Git2DBError::repository(&worktrees_dir, format!("Failed to create worktrees directory: {e}"))
            })?;
        }

        // ===== Phase 2: No lock - perform network clone =====
        let bare_repo = if let Some(repo) = existing_bare_repo {
            repo
        } else {
            tracing::info!("Cloning repository '{}' as bare to {:?}", repo_name, bare_repo_path);

            let clone_options = if let Some(config) = self.callback_config.take() {
                crate::clone_options::CloneOptions::with_callback_config(config)
            } else {
                crate::manager::GitManager::global().default_clone_options()
            };

            let url_clone = self.url.clone();
            let bare_path_clone = bare_repo_path.clone();

            // This is the slow network operation - no lock held!
            tokio::task::spawn_blocking(move || -> Git2DBResult<git2::Repository> {
                let mut git2_options = clone_options.to_git2_options();
                let mut builder = git2::build::RepoBuilder::new();
                builder.bare(true);
                builder.fetch_options(git2_options.create_fetch_options()
                    .map_err(|e| Git2DBError::internal(format!("Failed to create fetch options: {e}")))?);

                builder.clone(&url_clone, &bare_path_clone)
                    .map_err(|e| Git2DBError::repository(
                        &bare_path_clone,
                        format!("Failed to clone bare repository: {e}")
                    ))
            })
            .await
            .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))??
        };

        // Add additional remotes
        for (remote_name, remote_url) in &self.remotes {
            bare_repo.remote(remote_name, remote_url).map_err(|e| {
                Git2DBError::configuration(format!("Failed to add remote '{remote_name}': {e}"))
            })?;
        }

        // Get default branch and create worktrees
        let default_branch = get_default_branch(&bare_repo)?;
        tracing::debug!("Default branch detected: {}", default_branch);

        // Validate branch name for path safety (C2 fix)
        hyprstream_containedfs::validate_ref_name(&default_branch)
            .map_err(|e| Git2DBError::configuration(format!("Unsafe default branch name: {e}")))?;

        let initial_worktree = hyprstream_containedfs::contained_join(&worktrees_dir, &default_branch)
            .map_err(|e| Git2DBError::configuration(format!("Invalid worktree path: {e}")))?;
        if let Some(parent) = initial_worktree.parent() {
            if parent != worktrees_dir {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Git2DBError::configuration(format!("Failed to create worktree parent directories: {e}"))
                })?;
            }
        }

        tracing::info!("Creating default worktree '{}' at {:?}", default_branch, initial_worktree);
        create_worktree(&driver, &bare_repo_path, &initial_worktree, &default_branch).await?;

        // Create additional worktree if requested ref differs from default
        let checkout_ref = match &self.reference {
            GitRef::DefaultBranch => default_branch.clone(),
            GitRef::Branch(b) => b.clone(),
            GitRef::Tag(t) => t.clone(),
            GitRef::Commit(oid) => oid.to_string(),
            GitRef::Revspec(spec) => spec.clone(),
        };

        if checkout_ref != default_branch && !matches!(self.reference, GitRef::DefaultBranch) {
            // Validate checkout ref for path safety (C2 fix)
            hyprstream_containedfs::validate_ref_name(&checkout_ref)
                .map_err(|e| Git2DBError::configuration(format!("Unsafe checkout ref name: {e}")))?;
            let ref_worktree_path = hyprstream_containedfs::contained_join(&worktrees_dir, &checkout_ref)
                .map_err(|e| Git2DBError::configuration(format!("Invalid worktree path: {e}")))?;
            if let Some(parent) = ref_worktree_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Git2DBError::configuration(format!("Failed to create worktree parent directories: {e}"))
                })?;
            }
            tracing::info!("Creating worktree for ref '{}'", checkout_ref);
            create_worktree(&driver, &bare_repo_path, &ref_worktree_path, &checkout_ref).await?;
        }

        // Build remote configs
        let mut remote_configs = vec![RemoteConfig {
            name: "origin".to_owned(),
            url: self.url.clone(),
            fetch_refs: vec!["+refs/heads/*:refs/remotes/origin/*".to_owned()],
        }];
        for (name, url) in self.remotes {
            remote_configs.push(RemoteConfig {
                name: name.clone(),
                url,
                fetch_refs: vec![format!("+refs/heads/*:refs/remotes/{}/*", name)],
            });
        }

        // ===== Phase 3: Write lock for registration =====
        {
            let mut registry = self.registry.write().await;
            registry
                .register_repository_internal(
                    repo_id.clone(),
                    self.name.clone(),
                    self.url,
                    bare_repo_path,
                    self.reference,
                    remote_configs,
                    std::collections::HashMap::new(),
                )
                .await?;
        } // Write lock released

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
            return Ok(name.to_owned());
        }
    }

    // Try to get from remote HEAD
    if let Ok(remote) = repo.find_remote("origin") {
        if let Ok(buf) = remote.default_branch() {
            if let Some(s) = buf.as_str() {
                // Remove refs/heads/ prefix
                let branch = s.strip_prefix("refs/heads/").unwrap_or(s);
                return Ok(branch.to_owned());
            }
        }
    }

    // Fallback to common defaults
    for default in &["main", "master"] {
        if repo.find_branch(default, git2::BranchType::Local).is_ok() {
            return Ok((*default).to_owned());
        }
    }

    // Last resort
    Ok("main".to_owned())
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
            Git2DBError::configuration(format!("Failed to create worktree parent directories: {e}"))
        })?;
    }

    // Use storage driver (strict: errors if path exists)
    let opts = crate::storage::DriverOpts {
        base_repo: bare_repo_path.to_path_buf(),
        worktree_path: worktree_path.to_path_buf(),
        ref_spec: branch.to_owned(),
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
            format!("Failed to create worktree for branch '{branch}': {e}")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_fluent_api() {
        // This test just validates the fluent API compiles correctly
        fn _example(registry: Arc<RwLock<Git2DB>>) {
            let _builder = CloneBuilder::new(Arc::clone(&registry), "https://github.com/user/repo.git")
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
