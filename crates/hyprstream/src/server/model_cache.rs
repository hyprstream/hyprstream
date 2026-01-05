//! Git-native model cache using commit SHA for consistency

use anyhow::Result;
use git2;
use git2db::{Git2DBConfig as GitConfig, GitManager};
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, instrument, warn};

use crate::cli::commands::KVQuantArg;
use crate::inference::{InferenceClient, LocalInferenceClient, LocalInferenceService};
use crate::runtime::kv_quant::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::storage::{ModelRef, ModelStorage};

/// Cached model entry
#[derive(Clone)]
pub struct CachedModel {
    /// Commit SHA this model was loaded from
    pub commit_id: git2db::Oid,
    /// Model reference string
    pub model_ref: String,
    /// Inference client handle (cloneable, communicates with service on dedicated thread)
    pub client: LocalInferenceClient,
    /// Last access time
    pub last_accessed: std::time::Instant,
}

/// Model cache that caches by commit SHA
pub struct ModelCache {
    /// LRU cache mapping commit SHA to loaded engine
    cache: Arc<Mutex<LruCache<git2db::Oid, CachedModel>>>,
    /// Registry for resolving model references
    registry: Arc<ModelStorage>,
    /// Cache of worktree paths for specific commits
    checkouts: Arc<RwLock<HashMap<git2db::Oid, PathBuf>>>,
    /// Maximum cache size
    max_size: usize,
    /// Git service for repository operations
    git_manager: Arc<GitManager>,
    /// Maximum context length for KV cache allocation (overrides model's max_position_embeddings)
    max_context: Option<usize>,
    /// KV cache quantization type
    kv_quant: KVQuantType,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(
        max_size: usize,
        registry: Arc<ModelStorage>,
        max_context: Option<usize>,
        kv_quant: KVQuantArg,
    ) -> Result<Self> {
        Self::new_with_config(max_size, registry, GitConfig::default(), max_context, kv_quant)
    }

    /// Create a new model cache with custom Git configuration
    pub fn new_with_config(
        max_size: usize,
        registry: Arc<ModelStorage>,
        git_config: GitConfig,
        max_context: Option<usize>,
        kv_quant: KVQuantArg,
    ) -> Result<Self> {
        // SAFETY: 5 is a valid non-zero value
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(max_size).unwrap_or(DEFAULT_CACHE_SIZE);

        let git_manager = Arc::new(GitManager::new(git_config));

        if let Some(mc) = max_context {
            info!("ModelCache using max_context override: {} tokens", mc);
        }

        let kv_quant_type: KVQuantType = kv_quant.into();
        if kv_quant_type != KVQuantType::None {
            info!("ModelCache using KV cache quantization: {:?}", kv_quant_type);
        }

        Ok(Self {
            cache: Arc::new(Mutex::new(LruCache::new(cache_size))),
            registry,
            checkouts: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            git_manager,
            max_context,
            kv_quant: kv_quant_type,
        })
    }

    /// Get or load a model by reference
    #[instrument(name = "model_cache.get_or_load", skip(self), fields(
        model_ref = %model_ref_str,
        cache_hit = tracing::field::Empty
    ))]
    pub async fn get_or_load(&self, model_ref_str: &str) -> Result<LocalInferenceClient> {
        // Parse model reference
        let model_ref = ModelRef::parse(model_ref_str)?;

        // Resolve to exact commit SHA
        let commit_id = self.registry.resolve_git_ref(&model_ref).await?;

        // Check cache by commit SHA
        {
            let mut cache = self.cache.lock().await;
            if let Some(cached) = cache.get_mut(&commit_id) {
                info!(
                    "Using cached model: {} @ {}",
                    model_ref_str,
                    format_oid(&commit_id)
                );
                tracing::Span::current().record("cache_hit", true);
                cached.last_accessed = std::time::Instant::now();
                return Ok(cached.client.clone());
            }
        }

        // Load model at specific commit
        info!(
            "Loading model: {} @ {}",
            model_ref_str,
            format_oid(&commit_id)
        );
        tracing::Span::current().record("cache_hit", false);
        let checkout_path = self.get_or_create_checkout(&model_ref, commit_id).await?;

        // Create runtime config with max_context and kv_quant overrides
        let mut config = RuntimeConfig::default();
        config.max_context = self.max_context;
        config.kv_quant_type = self.kv_quant;

        info!("Model path: {:?}", checkout_path);
        let load_start = std::time::Instant::now();

        // Start inference service (runs on dedicated thread, loads model, initializes KV cache)
        let client = LocalInferenceService::start(&checkout_path, config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start inference service: {}", e))?;

        let load_time = load_start.elapsed();
        info!(
            "Model ready: {} (loaded in {:.2}s)",
            model_ref_str,
            load_time.as_secs_f64()
        );

        // Create cached entry
        let cached_model = CachedModel {
            commit_id,
            model_ref: model_ref_str.to_string(),
            client: client.clone(),
            last_accessed: std::time::Instant::now(),
        };

        // Add to cache (may evict LRU)
        {
            let mut cache = self.cache.lock().await;
            if let Some((evicted_id, evicted)) = cache.push(commit_id, cached_model) {
                warn!(
                    "Cache full - removing {} @ {}",
                    evicted.model_ref,
                    format_oid(&evicted_id)
                );
                // Send shutdown signal to evicted service (fire-and-forget)
                tokio::spawn(async move {
                    if let Err(e) = evicted.client.shutdown().await {
                        warn!("Failed to shutdown evicted model service: {}", e);
                    }
                });
            }
        }

        Ok(client)
    }

    /// Get or load a model by name (for backwards compatibility)
    pub async fn get_or_load_by_name(&self, model_name: &str) -> Result<LocalInferenceClient> {
        self.get_or_load(model_name).await
    }

    /// Get or create a checkout for a specific commit
    ///
    /// Uses git2db's worktree management to find the appropriate worktree
    async fn get_or_create_checkout(
        &self,
        model_ref: &ModelRef,
        commit_id: git2db::Oid,
    ) -> Result<PathBuf> {
        // Check if we already have this path cached
        {
            let checkouts = self.checkouts.read().await;
            if let Some(path) = checkouts.get(&commit_id) {
                debug!("Reusing cached worktree path at {:?}", path);
                return Ok(path.clone());
            }
        }

        // Get the worktree path for the requested branch
        // get_model_path() handles branch resolution:
        // - If branch specified: uses that branch
        // - If default/unspecified: uses default branch
        let worktree_path = self.registry.get_model_path(model_ref).await?;

        if !worktree_path.exists() {
            warn!(
                "Worktree not found for model {} at {:?}",
                model_ref.model,
                worktree_path
            );
            return Err(anyhow::anyhow!(
                "Model worktree not found for {}. \
                Please ensure the model is properly cloned with 'hyprstream clone'.",
                model_ref.model
            ));
        }

        // Worktree exists, verify it's valid
        debug!("Using existing worktree at {:?}", worktree_path);

        // Verify the worktree contains expected model files
        let model_file_checks = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
        ];

        let has_model = model_file_checks.iter().any(|f| worktree_path.join(f).exists());

        if !has_model {
            return Err(anyhow::anyhow!(
                "Worktree at {:?} does not contain valid model files. \
                Please ensure the model was properly cloned.",
                worktree_path
            ));
        }

        // Cache the worktree path for this commit
        {
            let mut checkouts = self.checkouts.write().await;
            checkouts.insert(commit_id, worktree_path.clone());
        }

        Ok(worktree_path)
    }

    /// Check if a model is cached by name
    pub async fn is_cached_by_name(&self, model_ref_str: &str) -> bool {
        if let Ok(model_ref) = ModelRef::parse(model_ref_str) {
            if let Ok(commit_id) = self.registry.resolve_git_ref(&model_ref).await {
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

        // Note: Worktrees are now managed by git2db and cleaned up through
        // its mechanisms. We only clear the checkout tracking here.

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
        let models_dir = self.registry.get_models_dir();
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
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
            self.cleanup_worktrees_for_model(&model_path, keep_count)
                .await?;
        }

        Ok(())
    }

    /// Clean up old worktrees for a specific model repository
    async fn cleanup_worktrees_for_model(
        &self,
        model_path: &Path,
        keep_count: usize,
    ) -> Result<()> {
        let model_path = model_path.to_path_buf();
        let git_manager = Arc::clone(&self.git_manager);

        tokio::task::spawn_blocking(move || -> Result<()> {
            let repo_cache = match git_manager.get_repository(&model_path) {
                Ok(cache) => cache,
                Err(_) => return Ok(()), // Skip if not a git repo
            };
            let repo = repo_cache.open()?;

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
                        debug!("Pruning old worktree: {}", name);
                        let _ =
                            wt.prune(Some(git2::WorktreePruneOptions::new().working_tree(true)));
                    }
                }
            }

            Ok(())
        })
        .await?
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
fn format_oid(oid: &git2db::Oid) -> String {
    let s = oid.to_string();
    if s.len() > 8 {
        s[..8].to_string()
    } else {
        s
    }
}
