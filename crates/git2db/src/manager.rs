//! Git repository manager with thread-safe operations
//!
//! This module provides centralized Git management following hyprstream's patterns
//! for thread safety and connection pooling.

use anyhow::{bail, Result};
use dashmap::DashMap;
use git2::{build::RepoBuilder, Repository, Signature};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use safe_path::scoped_join;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, info, trace};

use crate::clone_options::CloneOptions;
use crate::config::Git2DBConfig;
use crate::errors::{Git2DBError, Git2DBResult};
use crate::repository::{CacheStats, RepositoryCache};
use crate::transport_registry::TransportRegistry;

// Import storage driver system
use crate::storage::DriverRegistry;

/// Global GitManager instance - singleton pattern from hyprstream
static GLOBAL_GIT_MANAGER: OnceCell<GitManager> = OnceCell::new();

/// Cache entry for repository with TTL tracking
struct CacheEntry {
    cache: RepositoryCache,
    last_accessed: parking_lot::Mutex<Instant>,
}

impl CacheEntry {
    fn new(cache: RepositoryCache) -> Self {
        Self {
            cache,
            last_accessed: parking_lot::Mutex::new(Instant::now()),
        }
    }

    fn touch(&self) {
        *self.last_accessed.lock() = Instant::now();
    }

    fn last_accessed(&self) -> Instant {
        *self.last_accessed.lock()
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.last_accessed().elapsed() > ttl
    }
}

/// Centralized Git manager with thread-safe operations
pub struct GitManager {
    config: Git2DBConfig,
    /// Cached repositories with metadata (DashMap for lock-free concurrent access)
    repo_cache: Arc<DashMap<PathBuf, CacheEntry>>,
    /// Semaphore for limiting concurrent operations
    operation_semaphore: Arc<Semaphore>,
    /// Active operation tracking
    active_operations: Arc<RwLock<HashMap<String, Instant>>>,
    /// Thread-safe transport registry for URL schemes
    transport_registry: Arc<TransportRegistry>,
    /// Storage driver registry (Docker's graphdriver pattern)
    driver_registry: Arc<DriverRegistry>,
    /// Background cleanup task handle (kept alive for the lifetime of GitManager)
    #[allow(dead_code)]
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl GitManager {
    /// Create a new GitManager with the given configuration
    pub fn new(config: Git2DBConfig) -> Self {
        let repo_cache = Arc::new(DashMap::new());

        let cleanup_handle = if config.performance.auto_cleanup {
            Some(Self::start_cleanup_task(
                repo_cache.clone(),
                Duration::from_secs(config.performance.repo_cache_ttl_secs),
                config.performance.max_repo_cache,
            ))
        } else {
            None
        };

        // Create driver registry (Docker pattern)
        let driver_registry = Arc::new(DriverRegistry::new());

        Self {
            operation_semaphore: Arc::new(Semaphore::new(config.performance.max_concurrent_ops)),
            config,
            repo_cache,
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            transport_registry: Arc::new(TransportRegistry::new()),
            driver_registry,
            cleanup_handle,
        }
    }

    /// Start background cleanup task for expired cache entries
    fn start_cleanup_task(
        cache: Arc<DashMap<PathBuf, CacheEntry>>,
        ttl: Duration,
        max_size: usize,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;

                // Remove expired entries
                let expired: Vec<PathBuf> = cache
                    .iter()
                    .filter_map(|entry| {
                        if entry.is_expired(ttl) {
                            Some(entry.key().clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                for path in &expired {
                    cache.remove(path);
                }

                if !expired.is_empty() {
                    debug!(
                        "Cleaned up {} expired repository cache entries",
                        expired.len()
                    );
                }

                // Evict oldest if over max size
                while cache.len() > max_size {
                    if let Some(oldest) = cache
                        .iter()
                        .min_by_key(|entry| entry.last_accessed())
                        .map(|entry| entry.key().clone())
                    {
                        cache.remove(&oldest);
                        debug!("Evicted repository from cache (size limit): {:?}", oldest);
                    } else {
                        break;
                    }
                }
            }
        })
    }

    /// Initialize the global GitManager with a custom configuration
    ///
    /// This must be called before the first call to `global()`.
    /// Subsequent calls will return an error to avoid silent configuration changes.
    ///
    /// Returns Ok(()) if initialization succeeded, Err if it failed or already initialized.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use git2db::config::Git2DBConfig;
    /// use git2db::GitManager;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = Git2DBConfig::default();
    /// GitManager::init_with_config(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn init_with_config(config: Git2DBConfig) -> Result<(), crate::errors::Git2DBError> {
        if GLOBAL_GIT_MANAGER.get().is_some() {
            return Err(crate::errors::Git2DBError::configuration(
                "GitManager is already initialized. Cannot reinitialize with new configuration.",
            ));
        }

        GLOBAL_GIT_MANAGER
            .set(GitManager::new(config))
            .map_err(|_| {
                crate::errors::Git2DBError::internal("Failed to initialize global GitManager")
            })
            .map(|_| ())
    }

    /// Get the global GitManager instance
    ///
    /// If not already initialized via `init_with_config()`, this will initialize
    /// it with default configuration.
    pub fn global() -> &'static GitManager {
        GLOBAL_GIT_MANAGER.get_or_init(|| {
            // Load config from environment and config files
            tracing::info!("Initializing global GitManager...");
            let config = match Git2DBConfig::load() {
                Ok(cfg) => {
                    tracing::info!("Successfully loaded git2db config from environment/file");
                    cfg
                }
                Err(e) => {
                    tracing::warn!("Failed to load git2db config, using defaults: {}", e);
                    Git2DBConfig::default()
                }
            };
            GitManager::new(config)
        })
    }

    /// Get the configuration for this GitManager
    pub fn config(&self) -> &Git2DBConfig {
        &self.config
    }

    /// Create default clone options based on configuration
    pub fn default_clone_options(&self) -> CloneOptions {
        let mut builder = CloneOptions::builder();

        // Set up shallow clone if configured
        if self.config.repository.prefer_shallow {
            builder = builder.shallow(true);
            if let Some(depth) = self.config.repository.shallow_depth {
                builder = builder.depth(depth as i32);
            }
        }

        // Set up proxy if configured
        if let Some(proxy_url) = &self.config.network.proxy_url {
            builder = builder.proxy_url(proxy_url.clone());
        }

        // Set up timeout
        builder = builder.timeout(self.config.network.timeout_secs as u32);

        // Set up authentication
        use crate::auth::AuthStrategy;
        use crate::callback_config::{CallbackConfigBuilder, CertificateConfig};

        let mut callback_builder = CallbackConfigBuilder::new();

        // Add token authentication if configured
        let token_from_config = self.config.network.access_token.clone();
        let token_from_env = std::env::var("GIT2DB_NETWORK__ACCESS_TOKEN").ok();

        let token = token_from_config.or(token_from_env);

        if let Some(ref token_value) = token {
            tracing::info!(
                "Using token authentication (token starts with: {})",
                &token_value.chars().take(10).collect::<String>()
            );
            callback_builder = callback_builder.auth(AuthStrategy::Token {
                token: token_value.clone(),
            });
        } else {
            tracing::warn!("No access token configured");
        }

        // Add SSH agent authentication
        callback_builder = callback_builder.auth(AuthStrategy::SshAgent {
            username: Some("git".to_owned()),
        });

        // Add default credential helper if enabled (for git credentials, osxkeychain, etc.)
        if self.config.network.use_credential_helper {
            callback_builder = callback_builder.auth(AuthStrategy::Default);
        }

        // Use system certificate validation
        callback_builder = callback_builder.certificates(CertificateConfig::SystemDefault);

        builder = builder.callback_config(callback_builder.build());

        builder.build()
    }

    /// Check if the global GitManager has been initialized
    pub fn is_initialized() -> bool {
        GLOBAL_GIT_MANAGER.get().is_some()
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

    /// Get a repository cache handle with intelligent caching
    ///
    /// Returns a RepositoryCache that tracks metadata and provides fresh Repository
    /// instances on each .open() call (for thread safety with git2).
    pub fn get_repository<P: AsRef<Path>>(&self, path: P) -> Git2DBResult<RepositoryCache> {
        let path_ref = path.as_ref();
        let path = path_ref.canonicalize().map_err(|e| {
            Git2DBError::invalid_path(path_ref, format!("Failed to canonicalize: {e}"))
        })?;

        // Check cache first (lock-free read with DashMap)
        if let Some(entry) = self.repo_cache.get(&path) {
            if !entry.is_expired(Duration::from_secs(self.config.performance.repo_cache_ttl_secs)) {
                entry.touch();
                trace!("Repository cache hit for {:?}", path);
                return Ok(entry.cache.clone());
            } else {
                // Remove expired entry
                drop(entry); // Release the ref before removing
                self.repo_cache.remove(&path);
                debug!("Removed expired repository cache entry for {:?}", path);
            }
        }

        // Cache miss - open repository
        trace!("Repository cache miss for {:?}", path);
        let repository = Repository::open(&path).map_err(|e| {
            Git2DBError::repository(&path, format!("Failed to open repository: {e}"))
        })?;

        let cache = RepositoryCache::new(repository, path.clone());
        let entry = CacheEntry::new(cache.clone());

        // Evict oldest if cache is full
        if self.repo_cache.len() >= self.config.performance.max_repo_cache {
            self.evict_oldest();
        }

        self.repo_cache.insert(path.clone(), entry);
        debug!("Added repository to cache: {:?}", path);

        Ok(cache)
    }

    /// Evict the oldest cache entry
    fn evict_oldest(&self) {
        if let Some(oldest) = self
            .repo_cache
            .iter()
            .min_by_key(|entry| entry.last_accessed())
            .map(|entry| entry.key().clone())
        {
            self.repo_cache.remove(&oldest);
            debug!("Evicted oldest repository from cache: {:?}", oldest);
        }
    }

    /// Get repository cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let mut total_accesses = 0;
        let mut oldest_created = None;
        let mut newest_created = None;
        let mut oldest_accessed = None;
        let mut newest_accessed = None;

        for entry in self.repo_cache.iter() {
            total_accesses += entry.cache.access_count();
            let created = entry.cache.created_at();
            let accessed = entry.last_accessed();

            oldest_created = Some(oldest_created.map_or(created, |old: Instant| old.min(created)));
            newest_created = Some(newest_created.map_or(created, |new: Instant| new.max(created)));
            oldest_accessed =
                Some(oldest_accessed.map_or(accessed, |old: Instant| old.min(accessed)));
            newest_accessed =
                Some(newest_accessed.map_or(accessed, |new: Instant| new.max(accessed)));
        }

        CacheStats {
            total_entries: self.repo_cache.len(),
            total_accesses,
            oldest_created: oldest_created.unwrap_or_else(Instant::now),
            newest_created: newest_created.unwrap_or_else(Instant::now),
            oldest_accessed: oldest_accessed.unwrap_or_else(Instant::now),
            newest_accessed: newest_accessed.unwrap_or_else(Instant::now),
        }
    }

    /// Clear the entire repository cache
    pub fn clear_cache(&self) {
        let count = self.repo_cache.len();
        self.repo_cache.clear();
        debug!("Cleared {} repository cache entries", count);
    }

    /// Manually trigger cache cleanup (remove expired entries)
    pub fn cleanup_cache(&self) {
        let ttl = Duration::from_secs(self.config.performance.repo_cache_ttl_secs);
        let expired: Vec<PathBuf> = self
            .repo_cache
            .iter()
            .filter_map(|entry| {
                if entry.is_expired(ttl) {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();

        for path in &expired {
            self.repo_cache.remove(path);
        }

        debug!(
            "Manually cleaned up {} expired cache entries",
            expired.len()
        );
    }

    /// Clone a repository with Send-safe async support
    ///
    /// Uses spawn_blocking internally to handle git2 operations safely.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use git2db::clone_options::CloneOptions;
    /// use git2db::callback_config::{CallbackConfigBuilder, ProgressConfig};
    /// use git2db::auth::AuthStrategy;
    ///
    /// let options = CloneOptions::builder()
    ///     .callback_config(
    ///         CallbackConfigBuilder::new()
    ///             .auth(AuthStrategy::SshAgent { username: Some("git".to_owned()) })
    ///             .progress(ProgressConfig::Stdout)
    ///             .build()
    ///     )
    ///     .shallow(true)
    ///     .depth(1)
    ///     .build();
    ///
    /// let repo = manager.clone_repository(url, path, Some(options)).await?;
    /// ```
    pub async fn clone_repository(
        &self,
        url: &str,
        target_path: &Path,
        options: Option<CloneOptions>,
    ) -> Git2DBResult<Repository> {
        use tokio::task::spawn_blocking;

        // Convert inputs to owned types for spawn_blocking
        let url = url.to_owned();
        let target_path = target_path.to_path_buf();
        let options = options.unwrap_or_else(|| self.default_clone_options());

        // Acquire operation permit
        let _permit =
            self.operation_semaphore.acquire().await.map_err(|e| {
                Git2DBError::internal(format!("Failed to acquire semaphore: {e}"))
            })?;

        // Track operation
        let operation_id = format!("clone_{}", target_path.display());
        {
            let mut ops = self.active_operations.write();
            ops.insert(operation_id.clone(), Instant::now());
        }

        info!("Cloning repository from {} to {:?}", url, target_path);

        // Validate path - use absolute paths directly, validate relative paths
        let validated_path = if target_path.is_absolute() {
            // Already absolute - use directly after basic validation
            target_path.to_path_buf()
        } else {
            // Relative path - join with current dir and validate
            let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            scoped_join(&current_dir, &target_path).map_err(|e| {
                Git2DBError::invalid_path(&target_path, format!("Path validation failed: {e}"))
            })?
        };

        // Create parent directory if needed
        if let Some(parent) = validated_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                Git2DBError::repository(
                    &validated_path,
                    format!("Failed to create parent directory: {e}"),
                )
            })?;
        }

        // Perform the clone in a blocking task
        let repo = spawn_blocking(move || -> Git2DBResult<Repository> {
            // Convert to git2 options inside spawn_blocking
            let mut git2_options = options.to_git2_options();

            // Configure builder with options
            let mut builder = RepoBuilder::new();

            // Apply fetch options
            builder.fetch_options(git2_options.create_fetch_options()?);

            // Apply checkout options
            builder.with_checkout(git2_options.create_checkout_builder());

            // Apply branch if specified
            if let Some(branch) = &git2_options.branch {
                builder.branch(branch);
            }

            debug!("Cloning from URL: {}", url);

            // Perform clone
            let repo = builder.clone(&url, &validated_path).map_err(|e| {
                Git2DBError::repository(
                    &validated_path,
                    format!("Failed to clone repository: {e}"),
                )
            })?;

            info!("Successfully cloned repository to {:?}", validated_path);
            Ok(repo)
        })
        .await
        .map_err(|e| Git2DBError::internal(format!("Task join error: {e}")))??;

        // Clean up operation tracking
        {
            let mut ops = self.active_operations.write();
            ops.remove(&operation_id);
        }

        Ok(repo)
    }

    /// Create a signature for commits
    pub fn create_signature(
        &self,
        name: Option<&str>,
        email: Option<&str>,
    ) -> Git2DBResult<Signature<'static>> {
        let sig_name = name.unwrap_or(&self.config.signature.name);
        let sig_email = email.unwrap_or(&self.config.signature.email);

        Signature::now(sig_name, sig_email)
            .map_err(|e| Git2DBError::internal(format!("Failed to create signature: {e}")))
    }

    /// Remove a worktree with cleanup
    pub fn remove_worktree(
        &self,
        base_repo_path: &Path,
        worktree_name: &str,
        base_dir: Option<&Path>,
    ) -> Git2DBResult<()> {
        let base_repo_path = if let Some(base) = base_dir {
            self.validate_path(base, base_repo_path)
                .map_err(|e| Git2DBError::internal(format!("Invalid base repo path: {e}")))?
        } else {
            base_repo_path.to_path_buf()
        };

        debug!("Removing worktree '{}'", worktree_name);

        let repo_cache = self.get_repository(&base_repo_path)?;
        let repo = repo_cache.open()?;

        // Find and prune worktree
        let wt = repo.find_worktree(worktree_name).map_err(|e| {
            Git2DBError::repository(&base_repo_path, format!("Failed to find worktree: {e}"))
        })?;

        // Configure prune options:
        // - valid(true): Allow pruning even if worktree is valid (not orphaned)
        // - working_tree(true): Also remove the working directory contents
        let mut opts = git2::WorktreePruneOptions::new();
        opts.valid(true);
        opts.working_tree(true);

        wt.prune(Some(&mut opts))
            .map_err(|e| {
                Git2DBError::repository(&base_repo_path, format!("Failed to prune worktree: {e}"))
            })?;

        info!("Successfully removed worktree '{}'", worktree_name);
        Ok(())
    }

    /// List all worktrees for a repository
    pub fn list_worktrees(&self, repo_path: &Path) -> Git2DBResult<Vec<String>> {
        let repo_cache = self.get_repository(repo_path)?;
        let repo = repo_cache.open()?;

        let worktrees = repo.worktrees().map_err(|e| {
            Git2DBError::repository(repo_path, format!("Failed to list worktrees: {e}"))
        })?;

        Ok(worktrees
            .into_iter()
            .flatten()
            .map(std::borrow::ToOwned::to_owned)
            .collect())
    }

    /// Get statistics about cached paths
    /// Get active operations
    pub fn active_operations(&self) -> Vec<(String, Duration)> {
        let ops = self.active_operations.read();
        let now = Instant::now();

        ops.iter()
            .map(|(id, start)| (id.clone(), now.duration_since(*start)))
            .collect()
    }

    /// Register a custom transport for a URL scheme
    ///
    /// This allows integration with custom git transports like gittorrent://,
    /// ipfs://, s3://, or any other custom protocol using a thread-safe registry.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use git2db::GitManager;
    /// # use std::sync::Arc;
    /// # struct MyTransportFactory;
    /// # impl git2db::TransportFactory for MyTransportFactory {
    /// #     fn create_transport(&self, url: &str) -> anyhow::Result<Box<dyn git2::transport::SmartSubtransport>> {
    /// #         unimplemented!()
    /// #     }
    /// #     fn supports_url(&self, url: &str) -> bool { url.starts_with("custom://") }
    /// # }
    ///
    /// let manager = GitManager::new(Default::default());
    /// let factory = Arc::new(MyTransportFactory);
    /// manager.register_transport("custom", factory).unwrap();
    /// ```
    pub fn register_transport(
        &self,
        scheme: &str,
        factory: Arc<dyn crate::transport::TransportFactory>,
    ) -> Result<(), Git2DBError> {
        let scheme = scheme.to_owned();
        self.transport_registry
            .register_transport(scheme.clone(), factory.clone())?;
        info!(
            "Successfully registered custom transport for scheme: {}",
            scheme
        );
        Ok(())
    }

    /// Unregister a custom transport
    ///
    /// Returns the factory if it was registered, None otherwise.
    /// Uses reference counting for thread-safe cleanup.
    pub fn unregister_transport(
        &self,
        scheme: &str,
    ) -> Option<Arc<dyn crate::transport::TransportFactory>> {
        let result = self.transport_registry.unregister_transport(scheme);
        if result.is_some() {
            info!(
                "Successfully unregistered custom transport for scheme: {}",
                scheme
            );
        }
        result
    }

    /// Check if a custom transport is registered for a given scheme
    pub fn has_transport(&self, scheme: &str) -> bool {
        self.transport_registry.has_transport(scheme)
    }

    /// Get a list of all registered transport schemes
    pub fn registered_transports(&self) -> Vec<String> {
        self.transport_registry.registered_schemes()
    }

    /// Get transport registry statistics
    pub fn transport_stats(&self) -> crate::transport_registry::RegistryStats {
        self.transport_registry.stats()
    }

    // ===== Worktree Management =====

    /// Get the driver registry
    pub fn driver_registry(&self) -> &Arc<DriverRegistry> {
        &self.driver_registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper to create a test config without async cleanup
    fn test_config() -> Git2DBConfig {
        let mut config = Git2DBConfig::default();
        config.performance.auto_cleanup = false; // Disable async task spawning
        config
    }

    #[test]
    fn test_global_manager_singleton() {
        // Initialize with test config first to avoid Tokio requirement
        let _ = GitManager::init_with_config(test_config());

        let manager1 = GitManager::global();
        let manager2 = GitManager::global();

        // Should be the same instance
        assert!(std::ptr::eq(manager1, manager2));
    }

    #[test]
    fn test_cache_functionality() -> crate::Git2DBResult<()> {
        let config = test_config();
        let manager = GitManager::new(config);

        // Initially empty
        let stats = manager.cache_stats();
        assert_eq!(stats.total_entries, 0);

        // Create a temp dir with an actual git repository
        let temp_dir = TempDir::new()?;
        let test_path = temp_dir.path();
        git2::Repository::init(test_path)?;
        let cache1 = manager.get_repository(test_path)?;
        let cache2 = manager.get_repository(test_path)?;

        // Should have one entry
        let stats = manager.cache_stats();
        assert_eq!(stats.total_entries, 1);

        // Should be the same cache entry (compare by path)
        assert_eq!(cache1.path(), cache2.path());

        // Cleanup
        manager.clear_cache();
        Ok(())
    }

    #[test]
    fn test_path_validation() -> crate::Git2DBResult<()> {
        let config = test_config();
        let manager = GitManager::new(config);

        // Use an actual temp directory for base_dir
        let temp_dir = TempDir::new()?;
        let base_dir = temp_dir.path();

        // Valid relative path
        let result = manager.validate_path(base_dir, Path::new("valid/file.txt"));
        assert!(result.is_ok());

        // Absolute path within base
        let absolute_path = base_dir.join("file.txt");
        let result = manager.validate_path(base_dir, &absolute_path);
        assert!(result.is_ok());

        // Absolute path outside base should fail
        let result = manager.validate_path(base_dir, Path::new("/etc/passwd"));
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_operations() -> crate::Git2DBResult<()> {
        let config = test_config();
        let manager = Arc::new(GitManager::new(config));

        let mut handles = Vec::new();
        let mut temp_dirs = Vec::new();

        for _ in 0..5 {
            let dir = TempDir::new()?;
            git2::Repository::init(dir.path())?;
            temp_dirs.push(dir);
        }

        // Create concurrent repository operations
        // Iterate by reference to keep temp_dirs alive
        for temp_dir in &temp_dirs {
            let manager_clone = manager.clone();
            let path = temp_dir.path().to_path_buf();

            let handle = tokio::spawn(async move {
                let result = manager_clone.get_repository(&path);
                result.map(|_| ())
            });
            handles.push(handle);
        }

        // All should succeed without conflicts
        for handle in handles {
            let result = handle.await.map_err(|e| crate::Git2DBError::internal(format!("Task join error: {e}")))?;
            assert!(result.is_ok());
        }

        // temp_dirs dropped here, after all operations complete
        Ok(())
    }

    #[test]
    fn test_cache_operations() -> crate::Git2DBResult<()> {
        let manager = GitManager::new(test_config());

        // Initially empty
        let stats = manager.cache_stats();
        assert_eq!(stats.total_entries, 0);

        // Access some paths (this would happen through get_repository in practice)
        let temp_dir = TempDir::new()?;
        Repository::init(temp_dir.path())?;
        manager.get_repository(temp_dir.path())?;

        // Cache should have one entry
        let stats = manager.cache_stats();
        assert_eq!(stats.total_entries, 1);

        // Clear cache
        manager.clear_cache();
        let stats = manager.cache_stats();
        assert_eq!(stats.total_entries, 0);
        Ok(())
    }

    #[test]
    fn test_signature_creation() -> crate::Git2DBResult<()> {
        let manager = GitManager::new(test_config());

        // Default signature
        let sig = manager.create_signature(None, None)?;
        assert_eq!(sig.name().ok_or_else(|| crate::Git2DBError::internal("no name"))?, "git2db");
        assert_eq!(sig.email().ok_or_else(|| crate::Git2DBError::internal("no email"))?, "git2db@local");

        // Custom signature
        let custom_sig = manager
            .create_signature(Some("Test User"), Some("test@example.com"))?;
        assert_eq!(custom_sig.name().ok_or_else(|| crate::Git2DBError::internal("no name"))?, "Test User");
        assert_eq!(custom_sig.email().ok_or_else(|| crate::Git2DBError::internal("no email"))?, "test@example.com");
        Ok(())
    }
}