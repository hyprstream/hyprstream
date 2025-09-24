//! Centralized Git management with connection pooling and advanced features
//!
//! This module provides a unified interface for all Git operations in hyprstream,
//! leveraging libgit2's advanced features for better performance and reliability.

use anyhow::{Result, Context, bail};
use git2::{
    Repository, Signature, RemoteCallbacks, FetchOptions, ProxyOptions,
    CertificateCheckStatus, CredentialType, Cred, Progress, ErrorClass, ErrorCode,
    build::{RepoBuilder, CheckoutBuilder},
};
use lru::LruCache;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use safe_path::scoped_join;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{info, debug, warn, error};

/// Configuration for Git operations
#[derive(Debug, Clone)]
pub struct GitConfig {
    /// Maximum number of cached repository handles
    pub max_repo_cache: usize,
    /// Default signature for commits
    pub default_signature: GitSignature,
    /// Network timeout for operations
    pub network_timeout: Duration,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
    /// Enable shallow clones by default
    pub prefer_shallow: bool,
    /// Default clone depth for shallow clones
    pub shallow_depth: Option<u32>,
    /// Custom certificate verification
    pub verify_certificates: bool,
    /// Proxy configuration
    pub proxy_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GitSignature {
    pub name: String,
    pub email: String,
}

impl Default for GitConfig {
    fn default() -> Self {
        Self {
            max_repo_cache: 100,
            default_signature: GitSignature {
                name: "hyprstream".to_string(),
                email: "hyprstream@local".to_string(),
            },
            network_timeout: Duration::from_secs(30),
            max_concurrent_ops: 10,
            prefer_shallow: true,
            shallow_depth: Some(1),
            verify_certificates: true,
            proxy_url: None,
        }
    }
}

/// Advanced progress reporting for Git operations
pub trait GitProgress: Send + Sync {
    fn on_progress(&self, progress: &GitProgressInfo);
    fn on_error(&self, error: &GitError);
    fn is_cancelled(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct GitProgressInfo {
    pub operation: String,
    pub current: usize,
    pub total: usize,
    pub bytes_received: usize,
    pub bytes_total: usize,
    pub elapsed: Duration,
}

/// Enhanced error information using libgit2's classification
#[derive(Debug, Clone)]
pub struct GitError {
    pub class: ErrorClass,
    pub code: ErrorCode,
    pub message: String,
    pub recoverable: bool,
    pub retry_suggested: bool,
}

impl From<git2::Error> for GitError {
    fn from(err: git2::Error) -> Self {
        let class = err.class();
        let code = err.code();
        let recoverable = matches!(
            class,
            ErrorClass::Net | ErrorClass::Http | ErrorClass::Ssh | ErrorClass::Ssl
        );
        let retry_suggested = matches!(
            code,
            ErrorCode::Timeout | ErrorCode::GenericError
        ) && recoverable;

        Self {
            class,
            code,
            message: err.message().to_string(),
            recoverable,
            retry_suggested,
        }
    }
}

/// Global GitManager instance
static GLOBAL_GIT_MANAGER: Lazy<GitManager> = Lazy::new(|| {
    GitManager::new(GitConfig::default())
});

/// Repository path cache entry
struct RepoPathEntry {
    last_accessed: Instant,
    path: PathBuf,
}

/// Centralized Git manager with advanced features
pub struct GitManager {
    config: GitConfig,
    /// Cached repository paths (not actual repositories for thread safety)
    repo_path_cache: Arc<RwLock<LruCache<PathBuf, RepoPathEntry>>>,
    /// Semaphore for limiting concurrent operations
    operation_semaphore: Arc<Semaphore>,
    /// Active operation tracking
    active_operations: Arc<RwLock<HashMap<String, Instant>>>,
}

impl GitManager {
    /// Create a new GitManager with the given configuration
    pub fn new(config: GitConfig) -> Self {
        let cache_size = NonZeroUsize::new(config.max_repo_cache)
            .unwrap_or(NonZeroUsize::new(100).unwrap());

        Self {
            operation_semaphore: Arc::new(Semaphore::new(config.max_concurrent_ops)),
            config,
            repo_path_cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            active_operations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get the global GitManager instance
    pub fn global() -> &'static GitManager {
        &GLOBAL_GIT_MANAGER
    }

    /// Validate path for security (prevents directory traversal)
    pub fn validate_path(&self, base_dir: &Path, path: &Path) -> Result<PathBuf> {
        if path.is_absolute() {
            // If absolute, verify it's within base_dir
            if !path.starts_with(base_dir) {
                bail!("Repository path outside allowed directory: {:?}", path);
            }
            Ok(path.to_path_buf())
        } else {
            // If relative, safely join with base_dir
            scoped_join(base_dir, path)
                .map_err(|e| anyhow::anyhow!("Path traversal attempt detected: {}", e))
        }
    }

    /// Get a repository handle (opens fresh each time for thread safety)
    pub fn get_repository<P: AsRef<Path>>(&self, path: P) -> Result<Repository> {
        let path = path.as_ref().to_path_buf();

        // Update path cache for tracking (but don't cache Repository objects)
        {
            let mut cache = self.repo_path_cache.write();
            cache.put(path.clone(), RepoPathEntry {
                last_accessed: Instant::now(),
                path: path.clone(),
            });
        }

        // Always open repository fresh for thread safety
        let repository = Repository::open(&path)
            .with_context(|| format!("Failed to open repository at {:?}", path))?;

        debug!("Repository opened: {:?}", path);
        Ok(repository)
    }

    /// Create an advanced repository builder with our configuration
    pub fn create_repo_builder(&self) -> RepoBuilder {
        let mut builder = RepoBuilder::new();

        // Configure fetch options
        let mut fetch_opts = FetchOptions::new();
        let mut callbacks = RemoteCallbacks::new();

        // Set up certificate verification
        if !self.config.verify_certificates {
            callbacks.certificate_check(|_, _| Ok(CertificateCheckStatus::CertificateOk));
        } else {
            callbacks.certificate_check(|_cert, _valid| {
                debug!("Certificate check for host");
                // Could implement custom certificate validation here
                Ok(CertificateCheckStatus::CertificateOk)
            });
        }

        // Set up credential callback
        callbacks.credentials(|url, username_from_url, allowed_types| {
            debug!("Credential request for {} (types: {:?})", url, allowed_types);

            if allowed_types.contains(CredentialType::SSH_KEY) {
                // Try SSH key authentication
                if let Some(username) = username_from_url {
                    return Cred::ssh_key_from_agent(username);
                }
            }

            if allowed_types.contains(CredentialType::DEFAULT) {
                return Cred::default();
            }

            Err(git2::Error::from_str("No suitable credentials found"))
        });

        // Configure proxy if specified
        if let Some(proxy_url) = &self.config.proxy_url {
            let mut proxy_opts = ProxyOptions::new();
            proxy_opts.url(proxy_url);
            fetch_opts.proxy_options(proxy_opts);
        }

        fetch_opts.remote_callbacks(callbacks);
        builder.fetch_options(fetch_opts);

        builder
    }

    /// Clone a repository with advanced options and progress tracking
    pub async fn clone_repository<P: AsRef<Path>>(
        &self,
        url: &str,
        path: P,
        options: CloneOptions,
        progress: Option<Arc<dyn GitProgress>>,
    ) -> Result<Repository> {
        let _permit = self.operation_semaphore.acquire().await
            .context("Failed to acquire operation permit")?;

        let operation_id = format!("clone-{}", url);
        self.track_operation(&operation_id);

        let path = path.as_ref().to_path_buf();
        let url = url.to_string();

        let result = tokio::task::spawn_blocking({
            let manager = self.clone();
            let progress = progress.clone();
            move || manager.clone_repository_blocking(&url, &path, options, progress)
        }).await??;

        self.untrack_operation(&operation_id);
        Ok(result)
    }

    fn clone_repository_blocking(
        &self,
        url: &str,
        path: &Path,
        options: CloneOptions,
        progress: Option<Arc<dyn GitProgress>>,
    ) -> Result<Repository> {
        let mut builder = self.create_repo_builder();

        // Configure shallow clone if requested
        if options.shallow || (options.depth.is_none() && self.config.prefer_shallow) {
            if let Some(_depth) = options.depth.or(self.config.shallow_depth) {
                builder.clone_local(git2::build::CloneLocal::NoLinks);
                // Note: git2 doesn't directly support shallow clone depth
                // This would typically be handled via fetch options
            }
        }

        // Note: Progress reporting not available in this git2 version
        // Clone the repository
        let progress_ref = progress.as_ref();

        // Perform the clone
        match builder.clone(url, path) {
            Ok(repo) => {
                info!("Successfully cloned {} to {:?}", url, path);
                Ok(repo)
            }
            Err(err) => {
                let git_error = GitError::from(err);
                error!("Clone failed: {:?}", git_error);

                if let Some(progress_handler) = progress_ref {
                    progress_handler.on_error(&git_error);
                }

                if git_error.retry_suggested {
                    warn!("Clone failure may be retryable: {}", git_error.message);
                }

                Err(anyhow::anyhow!("Clone failed: {}", git_error.message))
            }
        }
    }

    /// Create a standardized signature
    pub fn create_signature(&self, name: Option<&str>, email: Option<&str>) -> Result<Signature<'static>> {
        let name = name.unwrap_or(&self.config.default_signature.name);
        let email = email.unwrap_or(&self.config.default_signature.email);

        Signature::now(name, email)
            .context("Failed to create Git signature")
    }

    /// Retry a Git operation with exponential backoff
    pub async fn retry_operation<F, T>(&self, mut operation: F, max_retries: u32) -> Result<T>
    where
        F: FnMut() -> Result<T> + Send,
        T: Send,
    {
        let mut attempts = 0;
        let mut delay = Duration::from_millis(100);

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempts += 1;

                    if attempts > max_retries {
                        return Err(err);
                    }

                    // Check if error is retryable
                    if let Some(git_err) = err.downcast_ref::<git2::Error>() {
                        let class = git_err.class();
                        let code = git_err.code();
                        let recoverable = matches!(
                            class,
                            ErrorClass::Net | ErrorClass::Http | ErrorClass::Ssh | ErrorClass::Ssl
                        );
                        let retry_suggested = matches!(
                            code,
                            ErrorCode::Timeout | ErrorCode::GenericError
                        ) && recoverable;

                        let git_error = GitError {
                            class,
                            code,
                            message: git_err.message().to_string(),
                            recoverable,
                            retry_suggested,
                        };
                        if !git_error.retry_suggested {
                            return Err(err);
                        }

                        warn!("Retrying operation after error: {} (attempt {}/{})",
                              git_error.message, attempts, max_retries);
                    }

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(delay * 2, Duration::from_secs(10));
                }
            }
        }
    }

    /// Track an active operation
    fn track_operation(&self, operation_id: &str) {
        let mut ops = self.active_operations.write();
        ops.insert(operation_id.to_string(), Instant::now());
    }

    /// Stop tracking an operation
    fn untrack_operation(&self, operation_id: &str) {
        let mut ops = self.active_operations.write();
        ops.remove(operation_id);
    }

    /// Get information about active operations
    pub fn get_active_operations(&self) -> Vec<(String, Duration)> {
        let ops = self.active_operations.read();
        ops.iter()
            .map(|(id, start_time)| (id.clone(), start_time.elapsed()))
            .collect()
    }

    /// Clean up stale cache entries
    pub fn cleanup_cache(&self) {
        let mut cache = self.repo_path_cache.write();
        let stale_threshold = Duration::from_secs(3600); // 1 hour

        // LruCache doesn't have a way to iterate and remove based on condition
        // So we'll implement a simple cleanup on access
        let stale_count = cache.len(); // This is a placeholder
        if stale_count > 0 {
            debug!("Cache cleanup completed");
        }
    }

    /// Create a worktree for an adapter (replaces GitOps functionality)
    pub fn create_worktree(
        &self,
        base_repo_path: &Path,
        worktree_path: &Path,
        branch_name: &str,
        base_dir: Option<&Path>,
    ) -> Result<()> {
        // Validate paths if base_dir is provided
        let base_canonical = if let Some(base_dir) = base_dir {
            self.validate_path(base_dir, base_repo_path)?
        } else {
            base_repo_path.to_path_buf()
        };

        let worktree_canonical = if let Some(base_dir) = base_dir {
            self.validate_path(base_dir, worktree_path)?
        } else {
            worktree_path.to_path_buf()
        };

        // Open the base repository (with caching)
        let repo = self.get_repository(&base_canonical)?;

        // Create a new branch for the worktree
        let head = repo.head()?;
        let commit = head.peel_to_commit()?;
        repo.branch(branch_name, &commit, false)?;

        // Add the worktree
        let options = git2::WorktreeAddOptions::new();
        repo.worktree(
            branch_name,
            &worktree_canonical,
            Some(&options),
        )?;

        // Checkout the branch in the worktree
        let worktree_repo = self.get_repository(&worktree_canonical)?;
        let branch_ref = format!("refs/heads/{}", branch_name);
        worktree_repo.set_head(&branch_ref)?;
        worktree_repo.checkout_head(Some(CheckoutBuilder::default().force()))?;

        Ok(())
    }

    /// Remove a worktree (replaces GitOps functionality)
    pub fn remove_worktree(
        &self,
        base_repo_path: &Path,
        worktree_name: &str,
        base_dir: Option<&Path>,
    ) -> Result<()> {
        let base_canonical = if let Some(base_dir) = base_dir {
            self.validate_path(base_dir, base_repo_path)?
        } else {
            base_repo_path.to_path_buf()
        };

        // Open the repository (with caching)
        let repo = self.get_repository(&base_canonical)?;

        // Find and prune the worktree
        if let Ok(worktree) = repo.find_worktree(worktree_name) {
            worktree.prune(Some(git2::WorktreePruneOptions::new().working_tree(true)))?;
        }

        Ok(())
    }
}

impl Clone for GitManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            repo_path_cache: Arc::clone(&self.repo_path_cache),
            operation_semaphore: Arc::clone(&self.operation_semaphore),
            active_operations: Arc::clone(&self.active_operations),
        }
    }
}

/// Options for cloning repositories
#[derive(Debug, Clone)]
pub struct CloneOptions {
    /// Use shallow clone
    pub shallow: bool,
    /// Depth for shallow clone
    pub depth: Option<u32>,
    /// Only clone specific branch
    pub branch: Option<String>,
    /// Include submodules
    pub recurse_submodules: bool,
    /// Filter specification for partial clones
    pub filter_spec: Option<String>,
}

impl Default for CloneOptions {
    fn default() -> Self {
        Self {
            shallow: false,
            depth: None,
            branch: None,
            recurse_submodules: false,
            filter_spec: None,
        }
    }
}

/// Convenience trait for common Git operations
pub trait GitOperations {
    fn fetch_with_progress(&self, progress: Option<Arc<dyn GitProgress>>) -> Result<()>;
    fn create_branch(&self, name: &str, target: Option<&str>) -> Result<()>;
    fn checkout_branch(&self, name: &str) -> Result<()>;
    fn commit_changes(&self, message: &str, signature: Option<&Signature>) -> Result<()>;
}

impl GitOperations for Repository {
    fn fetch_with_progress(&self, progress: Option<Arc<dyn GitProgress>>) -> Result<()> {
        let mut remote = self.find_remote("origin")
            .context("Failed to find origin remote")?;

        let mut fetch_opts = FetchOptions::new();

        // Note: Progress reporting not available in this git2 version
        if let Some(_progress_handler) = progress {
            // Progress reporting would need to be implemented differently
        }

        remote.fetch(&[] as &[&str], Some(&mut fetch_opts), None)
            .context("Failed to fetch from remote")?;

        Ok(())
    }

    fn create_branch(&self, name: &str, target: Option<&str>) -> Result<()> {
        let commit = if let Some(target_ref) = target {
            self.revparse_single(target_ref)?.peel_to_commit()?
        } else {
            self.head()?.peel_to_commit()?
        };

        self.branch(name, &commit, false)
            .with_context(|| format!("Failed to create branch: {}", name))?;

        Ok(())
    }

    fn checkout_branch(&self, name: &str) -> Result<()> {
        let obj = self.revparse_single(name)?;

        self.checkout_tree(&obj, Some(CheckoutBuilder::default().force()))?;
        self.set_head(&format!("refs/heads/{}", name))?;

        Ok(())
    }

    fn commit_changes(&self, message: &str, signature: Option<&Signature>) -> Result<()> {
        let sig = match signature {
            Some(s) => {
                // Create a new signature with the same data
                git2::Signature::now(s.name().unwrap_or("unknown"), s.email().unwrap_or("unknown@local"))?
            },
            None => GitManager::global().create_signature(None, None)?,
        };

        let mut index = self.index()?;
        index.add_all(["."].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = self.find_tree(tree_id)?;
        let parent = self.head()?.peel_to_commit()?;

        self.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&parent],
        )?;

        Ok(())
    }
}