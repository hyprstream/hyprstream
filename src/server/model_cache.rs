//! Git-native model cache using commit SHA for consistency

use anyhow::{Result, anyhow};
use git2;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn, debug};

use crate::runtime::{TorchEngine, RuntimeConfig, RuntimeEngine};
use crate::storage::{ModelRef, SharedModelRegistry};

/// Cached model entry
#[derive(Clone)]
pub struct CachedModel {
    /// Commit SHA this model was loaded from
    pub commit_id: git2::Oid,
    /// Model reference string
    pub model_ref: String,
    /// Loaded engine
    pub engine: Arc<Mutex<TorchEngine>>,
    /// Last access time
    pub last_accessed: std::time::Instant,
}

/// Model cache that caches by commit SHA
pub struct ModelCache {
    /// LRU cache mapping commit SHA to loaded engine
    cache: Arc<Mutex<LruCache<git2::Oid, CachedModel>>>,
    /// Registry for resolving model references
    registry: Arc<SharedModelRegistry>,
    /// Sparse checkouts directory
    checkout_base: PathBuf,
    /// Reuse checkouts for same commit
    checkouts: Arc<RwLock<HashMap<git2::Oid, PathBuf>>>,
    /// Maximum cache size
    max_size: usize,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(
        max_size: usize,
        registry: Arc<SharedModelRegistry>,
        checkout_base: PathBuf,
    ) -> Result<Self> {
        let cache_size = NonZeroUsize::new(max_size)
            .unwrap_or_else(|| NonZeroUsize::new(5).unwrap());

        // Ensure checkout directory exists
        std::fs::create_dir_all(&checkout_base)?;

        Ok(Self {
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            registry,
            checkout_base,
            checkouts: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        })
    }

    /// Get or load a model by reference
    pub async fn get_or_load(&self, model_ref_str: &str) -> Result<Arc<Mutex<TorchEngine>>> {
        // Parse model reference
        let model_ref = ModelRef::parse(model_ref_str)?;

        // Resolve to exact commit SHA
        let commit_id = self.registry.resolve(&model_ref).await?;

        // Check cache by commit SHA
        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get_mut(&commit_id) {
                info!("Cache hit: {} @ {}", model_ref_str, format_oid(&commit_id));
                cached.last_accessed = std::time::Instant::now();
                return Ok(Arc::clone(&cached.engine));
            }
        }

        // Load model at specific commit
        info!("Cache miss: {} @ {}, loading...", model_ref_str, format_oid(&commit_id));
        let checkout_path = self.get_or_create_checkout(&model_ref, commit_id).await?;

        // Create engine and load model
        let config = RuntimeConfig::default();
        let mut engine = TorchEngine::new(config)?;

        info!("Loading model from: {:?}", checkout_path);
        engine.load_model(&checkout_path).await?;
        info!("Successfully loaded model {}", model_ref_str);

        let engine = Arc::new(Mutex::new(engine));

        // Create cached entry
        let cached_model = CachedModel {
            commit_id,
            model_ref: model_ref_str.to_string(),
            engine: Arc::clone(&engine),
            last_accessed: std::time::Instant::now(),
        };

        // Add to cache (may evict LRU)
        {
            let mut cache = self.cache.lock().await;
            if let Some((evicted_id, evicted)) = cache.push(commit_id, cached_model) {
                warn!("Evicted {} @ {} from cache",
                      evicted.model_ref, format_oid(&evicted_id));
            }
        }

        Ok(engine)
    }

    /// Get or load a model by name (for backwards compatibility)
    pub async fn get_or_load_by_name(&self, model_name: &str) -> Result<Arc<Mutex<TorchEngine>>> {
        self.get_or_load(model_name).await
    }

    /// Get or create a checkout for a specific commit
    async fn get_or_create_checkout(
        &self,
        model_ref: &ModelRef,
        commit_id: git2::Oid,
    ) -> Result<PathBuf> {
        // Check if we already have this checkout
        {
            let checkouts = self.checkouts.read().await;
            if let Some(path) = checkouts.get(&commit_id) {
                debug!("Reusing existing checkout at {:?}", path);
                return Ok(path.clone());
            }
        }

        // Create worktree checkout for this commit
        let checkout_dir = self.checkout_base.join(format_oid(&commit_id));

        if !checkout_dir.exists() {
            info!("Creating worktree checkout at {:?}", checkout_dir);

            // Get model path from registry
            let model_path = self.registry.get_model_path(&model_ref.model).await?;

            // Create worktree for specific commit
            let checkout_dir_clone = checkout_dir.clone();
            tokio::task::spawn_blocking(move || -> Result<()> {
                let repo = git2::Repository::open(&model_path)?;
                let worktree_name = format!("cache-{}", &commit_id.to_string()[..8]);

                // Clean up if worktree already exists
                if let Ok(wt) = repo.find_worktree(&worktree_name) {
                    wt.prune(Some(git2::WorktreePruneOptions::new().working_tree(true)))?;
                }

                // Create parent directory
                if let Some(parent) = checkout_dir_clone.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                // Create worktree
                let opts = git2::WorktreeAddOptions::new();
                repo.worktree(&worktree_name, &checkout_dir_clone, Some(&opts))?;

                // Checkout specific commit with detached HEAD
                let wt_repo = git2::Repository::open(&checkout_dir_clone)?;
                wt_repo.set_head_detached(commit_id)?;
                wt_repo.checkout_head(Some(
                    git2::build::CheckoutBuilder::default().force()
                ))?;

                debug!("Created worktree for commit {}", commit_id);
                Ok(())
            }).await??;
        }

        // Cache the checkout path
        {
            let mut checkouts = self.checkouts.write().await;
            checkouts.insert(commit_id, checkout_dir.clone());
        }

        Ok(checkout_dir)
    }

    /// Check if a model is cached by name
    pub async fn is_cached_by_name(&self, model_ref_str: &str) -> bool {
        if let Ok(model_ref) = ModelRef::parse(model_ref_str) {
            if let Ok(commit_id) = self.registry.resolve(&model_ref).await {
                let cache = self.cache.lock().await;
                return cache.peek(&commit_id).is_some();
            }
        }
        false
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().await;
        let checkouts = self.checkouts.read().await;

        CacheStats {
            cached_models: cache.len(),
            cached_checkouts: checkouts.len(),
            max_size: self.max_size,
        }
    }

    /// Clear the cache
    pub async fn clear(&self) {
        let mut cache = self.cache.lock().await;
        cache.clear();

        let mut checkouts = self.checkouts.write().await;
        checkouts.clear();

        // Clean up checkout directory
        if let Ok(entries) = std::fs::read_dir(&self.checkout_base) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }

        info!("Model cache cleared");
    }

    /// Clean up old worktrees when cache grows too large
    pub async fn cleanup_old_worktrees(&self, keep_count: usize) -> Result<()> {
        let checkouts = self.checkouts.read().await;

        // Only clean up if we have too many checkouts
        if checkouts.len() <= keep_count {
            return Ok(());
        }

        // Get all model paths that have worktrees to clean
        let mut model_paths = std::collections::HashSet::new();
        if let Ok(entries) = std::fs::read_dir(&self.checkout_base.parent().unwrap_or(&self.checkout_base)) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path.join(".git").exists() {
                    model_paths.insert(path);
                }
            }
        }

        drop(checkouts); // Release the read lock

        // Clean up old worktrees from each model repository
        for model_path in model_paths {
            self.cleanup_worktrees_for_model(&model_path, keep_count).await?;
        }

        Ok(())
    }

    /// Clean up old worktrees for a specific model repository
    async fn cleanup_worktrees_for_model(&self, model_path: &Path, keep_count: usize) -> Result<()> {
        let model_path = model_path.to_path_buf();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let repo = match git2::Repository::open(&model_path) {
                Ok(repo) => repo,
                Err(_) => return Ok(()), // Skip if not a git repo
            };

            // List all worktrees that start with "cache-"
            let mut cache_worktrees = Vec::new();

            // Get worktree list from git
            if let Ok(worktrees) = repo.worktrees() {
                for worktree_name in worktrees.iter().flatten() {
                    if worktree_name.starts_with("cache-") {
                        cache_worktrees.push(worktree_name.to_string());
                    }
                }
            }

            // Sort by name (which includes commit ID)
            cache_worktrees.sort();

            // Remove old ones if we have too many
            if cache_worktrees.len() > keep_count {
                let to_remove = cache_worktrees.len() - keep_count;
                for name in cache_worktrees.iter().take(to_remove) {
                    if let Ok(wt) = repo.find_worktree(name) {
                        info!("Pruning old worktree: {}", name);
                        let _ = wt.prune(Some(
                            git2::WorktreePruneOptions::new()
                                .working_tree(true)
                        ));
                    }
                }
            }

            Ok(())
        }).await?
    }

    /// Invalidate a specific model (no-op in commit-based cache)
    pub async fn invalidate_model(&self, _model_uuid: uuid::Uuid) {
        // No-op for backwards compatibility
        // Commit-based cache doesn't need UUID invalidation
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cached_models: usize,
    pub cached_checkouts: usize,
    pub max_size: usize,
}

/// Format git OID for display
fn format_oid(oid: &git2::Oid) -> String {
    let s = oid.to_string();
    if s.len() > 8 {
        s[..8].to_string()
    } else {
        s
    }
}