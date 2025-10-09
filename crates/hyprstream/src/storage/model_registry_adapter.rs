//! Adapter layer between ModelRegistry and Git2DB
//!
//! This adapter maps model names (used by ModelRegistry) to RepoIds (used by Git2DB),
//! enabling ModelRegistry to be a thin wrapper around Git2DB while maintaining
//! backward compatibility.

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use git2db::{Git2DB, RepoId};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::model_ref::ModelRef;

/// Internal adapter that maps ModelRef → RepoId
///
/// This adapter maintains the mapping between human-readable model names
/// (used by ModelRegistry's public API) and UUIDs (used internally by Git2DB).
pub struct RegistryAdapter {
    /// The underlying Git2DB registry
    git2db: Arc<RwLock<Git2DB>>,

    /// Fast lookup cache: model name → RepoId
    /// This avoids scanning git2db's list every time we need to resolve a name
    name_to_id: Arc<DashMap<String, RepoId>>,
}

impl RegistryAdapter {
    /// Create a new adapter wrapping a Git2DB registry
    pub fn new(git2db: Git2DB) -> Self {
        let adapter = Self {
            git2db: Arc::new(RwLock::new(git2db)),
            name_to_id: Arc::new(DashMap::new()),
        };

        // Populate cache from existing repositories
        adapter.rebuild_cache_sync();

        adapter
    }

    /// Rebuild the name→id cache from Git2DB's tracked repositories
    fn rebuild_cache_sync(&self) {
        // This needs to be sync because it's called from new()
        // We use try_read to avoid blocking if someone else has the lock
        if let Ok(registry) = self.git2db.try_read() {
            self.name_to_id.clear();

            for repo in registry.list() {
                if let Some(name) = repo.name.as_ref() {
                    self.name_to_id.insert(name.clone(), repo.id.clone());
                    debug!("Cached mapping: {} → {}", name, repo.id);
                }
            }

            info!("Rebuilt name cache with {} entries", self.name_to_id.len());
        }
    }

    /// Rebuild the cache asynchronously
    pub async fn rebuild_cache(&self) {
        let registry = self.git2db.read().await;
        self.name_to_id.clear();

        for repo in registry.list() {
            if let Some(name) = repo.name.as_ref() {
                self.name_to_id.insert(name.clone(), repo.id.clone());
            }
        }

        info!("Rebuilt name cache with {} entries", self.name_to_id.len());
    }

    /// Get or register a model in git2db
    ///
    /// If the model exists, returns its RepoId. Otherwise, creates a new RepoId
    /// and registers it (without cloning - caller should clone separately if needed).
    pub async fn ensure_registered(&self, name: &str, url: &str) -> Result<RepoId> {
        // Check cache first
        if let Some(id) = self.name_to_id.get(name) {
            debug!("Model '{}' found in cache: {}", name, id.value());
            return Ok(id.clone());
        }

        // Not in cache - check git2db directly (cache might be stale)
        {
            let registry = self.git2db.read().await;
            if let Some(tracked) = registry.get_by_name(name) {
                let id = tracked.id.clone();
                self.name_to_id.insert(name.to_string(), id.clone());
                debug!("Model '{}' found in git2db: {}", name, id);
                return Ok(id);
            }
        }

        // Not found - create a new RepoId and register it
        info!("Registering new model '{}' with URL: {}", name, url);
        let repo_id = git2db::RepoId::new();

        let mut registry = self.git2db.write().await;
        registry.register_repository(&repo_id, Some(name.to_string()), url.to_string()).await?;

        // Update cache
        self.name_to_id.insert(name.to_string(), repo_id.clone());

        info!("Registered model '{}' as {}", name, repo_id);
        Ok(repo_id)
    }

    /// Resolve ModelRef to RepoId
    ///
    /// Returns an error if the model is not found in the registry.
    pub fn resolve(&self, model_ref: &ModelRef) -> Result<RepoId> {
        self.name_to_id.get(&model_ref.model)
            .map(|id| id.clone())
            .ok_or_else(|| anyhow!(
                "Model '{}' not found in registry. Use add_model() to register it.",
                model_ref.model
            ))
    }

    /// Try to resolve ModelRef to RepoId, returning None if not found
    pub fn try_resolve(&self, model_ref: &ModelRef) -> Option<RepoId> {
        self.name_to_id.get(&model_ref.model).map(|id| id.clone())
    }

    /// Check if a model exists in the registry
    pub fn exists(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
    }

    /// Get the git2db registry (read-only)
    pub async fn registry(&self) -> tokio::sync::RwLockReadGuard<'_, Git2DB> {
        self.git2db.read().await
    }

    /// Get the git2db registry (writable)
    pub async fn registry_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, Git2DB> {
        self.git2db.write().await
    }

    /// Remove a model from the cache and registry
    pub async fn remove(&self, name: &str) -> Result<()> {
        let repo_id = self.name_to_id.get(name)
            .map(|id| id.clone())
            .ok_or_else(|| anyhow!("Model '{}' not found", name))?;

        // Remove from git2db
        let mut registry = self.git2db.write().await;
        registry.remove_repository(&repo_id).await?;

        // Remove from cache
        self.name_to_id.remove(name);

        info!("Removed model '{}' ({})", name, repo_id);
        Ok(())
    }

    /// List all models in the registry
    pub async fn list_models(&self) -> Vec<String> {
        let registry = self.git2db.read().await;
        registry.list()
            .filter_map(|r| r.name.clone())
            .collect()
    }

    /// Get the base directory of the registry
    pub async fn base_dir(&self) -> std::path::PathBuf {
        let registry = self.git2db.read().await;
        registry.base_dir().to_path_buf()
    }
}

impl Clone for RegistryAdapter {
    fn clone(&self) -> Self {
        Self {
            git2db: Arc::clone(&self.git2db),
            name_to_id: Arc::clone(&self.name_to_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_adapter_ensure_registered() {
        let temp = TempDir::new().unwrap();
        let git2db = Git2DB::open(temp.path()).await.unwrap();
        let adapter = RegistryAdapter::new(git2db);

        // Register a model
        let id1 = adapter.ensure_registered("test-model", "https://example.com/model.git")
            .await
            .unwrap();

        // Calling again should return the same ID
        let id2 = adapter.ensure_registered("test-model", "https://example.com/model.git")
            .await
            .unwrap();

        assert_eq!(id1, id2);
    }

    #[tokio::test]
    async fn test_adapter_resolve() {
        let temp = TempDir::new().unwrap();
        let git2db = Git2DB::open(temp.path()).await.unwrap();
        let adapter = RegistryAdapter::new(git2db);

        // Register a model
        let id = adapter.ensure_registered("test-model", "https://example.com/model.git")
            .await
            .unwrap();

        // Resolve by ModelRef
        let model_ref = ModelRef::new("test-model".to_string());
        let resolved_id = adapter.resolve(&model_ref).unwrap();

        assert_eq!(id, resolved_id);
    }

    #[tokio::test]
    async fn test_adapter_exists() {
        let temp = TempDir::new().unwrap();
        let git2db = Git2DB::open(temp.path()).await.unwrap();
        let adapter = RegistryAdapter::new(git2db);

        assert!(!adapter.exists("nonexistent"));

        adapter.ensure_registered("test-model", "https://example.com/model.git")
            .await
            .unwrap();

        assert!(adapter.exists("test-model"));
    }
}
