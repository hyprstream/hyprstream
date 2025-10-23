//! Simplified model storage working directly with git2db

use anyhow::Result;
use dashmap::DashMap;
use git2db::{Git2DB, RepoId};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::model_ref::{validate_model_name, ModelRef};

/// Model identifier (kept for backward compatibility)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId(pub Uuid);

impl ModelId {
    pub fn new() -> Self {
        ModelId(Uuid::new_v4())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub display_name: Option<String>,
    pub model_type: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub size_bytes: Option<u64>,
    pub tags: Vec<String>,
}

#[derive(Debug)]
pub struct CacheStats {
    pub total_size_bytes: u64,
}

/// Simplified model storage working directly with git2db
pub struct ModelStorage {
    base_dir: PathBuf,
    registry: Arc<RwLock<Git2DB>>,
    /// Fast name → RepoId lookup cache
    name_cache: Arc<DashMap<String, RepoId>>,
}

impl ModelStorage {
    /// Create with a new registry
    pub async fn create(base_dir: PathBuf) -> Result<Self> {
        Self::create_with_config(base_dir, git2db::config::Git2DBConfig::default()).await
    }

    /// Create with a new registry and custom git2db configuration
    pub async fn create_with_config(
        base_dir: PathBuf,
        _git2db_config: git2db::config::Git2DBConfig,
    ) -> Result<Self> {
        // GitManager::global() loads config from environment automatically
        let git2db = Git2DB::open(&base_dir).await?;

        // Build name cache
        let name_cache = Arc::new(DashMap::new());
        for tracked in git2db.list() {
            if let Some(name) = &tracked.name {
                name_cache.insert(name.clone(), tracked.id.clone());
            }
        }

        Ok(Self {
            base_dir,
            registry: Arc::new(RwLock::new(git2db)),
            name_cache,
        })
    }

    /// Get models directory
    pub fn get_models_dir(&self) -> PathBuf {
        self.base_dir.clone()
    }

    /// Resolve model name to RepoId
    fn resolve_name(&self, name: &str) -> Result<RepoId> {
        self.name_cache
            .get(name)
            .map(|id| id.clone())
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", name))
    }

    /// Get model path by reference
    pub async fn get_model_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        let repo_id = self.resolve_name(&model_ref.model)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;
        Ok(handle.worktree()?.to_path_buf())
    }

    /// List all models
    pub async fn list_models(&self) -> Result<Vec<(ModelRef, ModelMetadata)>> {
        let mut result = Vec::new();
        let registry = self.registry.read().await;

        for tracked in registry.list() {
            if let Some(name) = &tracked.name {
                let model_ref = ModelRef::new(name.clone());

                // Calculate size if model exists
                let size_bytes = if let Ok(handle) = registry.repo(&tracked.id) {
                    if let Ok(model_path) = handle.worktree() {
                        if model_path.exists() {
                            Self::calculate_dir_size(model_path).ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let metadata = ModelMetadata {
                    name: name.clone(),
                    display_name: Some(name.clone()),
                    model_type: "base".to_string(),
                    created_at: chrono::Utc::now().timestamp(),
                    updated_at: chrono::Utc::now().timestamp(),
                    size_bytes,
                    tags: vec![],
                };

                result.push((model_ref, metadata));
            }
        }

        Ok(result)
    }

    /// Calculate directory size
    fn calculate_dir_size(path: &Path) -> Result<u64> {
        let mut total_size = 0u64;
        for entry in walkdir::WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                if let Ok(metadata) = entry.metadata() {
                    total_size += metadata.len();
                }
            }
        }
        Ok(total_size)
    }

    /// Check if a model exists
    pub async fn model_exists(&self, model_name: &str) -> bool {
        self.name_cache.contains_key(model_name)
    }

    /// Add a new model
    pub async fn add_model(&self, name: &str, source: &str) -> Result<()> {
        validate_model_name(name)?;

        let mut registry = self.registry.write().await;
        let repo_id = registry
            .clone(source)
            .name(name)
            .depth(1)
            .exec()
            .await?;

        // Update cache
        self.name_cache.insert(name.to_string(), repo_id);

        Ok(())
    }

    /// Update a model to a different version
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        let repo_id = self.resolve_name(name)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;

        handle.fetch(None).await?;
        handle.checkout(ref_spec).await?;

        Ok(())
    }

    /// Get read access to the git2db registry
    ///
    /// Use this with `resolve_repo_id()` to perform git operations:
    /// ```ignore
    /// let repo_id = storage.resolve_repo_id(&model_ref)?;
    /// let registry = storage.registry().await;
    /// let handle = registry.repo(&repo_id)?;
    /// handle.branch().create("new", None).await?;
    /// ```
    pub async fn registry(&self) -> tokio::sync::RwLockReadGuard<'_, Git2DB> {
        self.registry.read().await
    }

    /// Resolve model name to RepoId for use with registry()
    pub fn resolve_repo_id(&self, model_ref: &ModelRef) -> Result<RepoId> {
        self.resolve_name(&model_ref.model)
    }

    // ========== Compatibility Methods ==========

    /// CLI compatibility method
    pub async fn list_local_models(&self) -> Result<Vec<(ModelRef, ModelMetadata)>> {
        self.list_models().await
    }

    /// Get cache stats
    pub async fn get_cache_stats(&self) -> Result<CacheStats> {
        let models = self.list_models().await?;
        let total_size: u64 = models.iter()
            .filter_map(|(_, metadata)| metadata.size_bytes)
            .sum();

        Ok(CacheStats {
            total_size_bytes: total_size,
        })
    }


    /// Get repository status
    pub async fn status(&self, model_ref: &ModelRef) -> Result<git2db::RepositoryStatus> {
        let repo_id = self.resolve_name(&model_ref.model)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;
        Ok(handle.status().await?)
    }

    /// Checkout a git reference
    pub async fn checkout(
        &self,
        model_ref: &ModelRef,
        _options: super::CheckoutOptions,
    ) -> Result<super::CheckoutResult> {
        let repo_id = self.resolve_name(&model_ref.model)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;

        let previous_oid = handle.current_oid()?.unwrap_or(git2db::Oid::zero());
        handle.checkout(model_ref.git_ref.clone()).await?;
        let new_oid = handle.current_oid()?.unwrap_or(git2db::Oid::zero());

        Ok(super::CheckoutResult {
            previous_oid,
            new_oid,
            previous_ref_name: None,
            new_ref_name: None,
            was_forced: false,
            files_changed: 0,
            has_submodule: true,
        })
    }

    /// Get default branch
    pub async fn get_default_branch(&self, model_ref: &ModelRef) -> Result<String> {
        let repo_id = self.resolve_name(&model_ref.model)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;
        Ok(handle.default_branch()?)
    }

    /// Remove model
    pub async fn remove_model(&self, model_ref: &ModelRef) -> Result<()> {
        let repo_id = self.resolve_name(&model_ref.model)?;
        let mut registry = self.registry.write().await;
        registry.remove_repository(&repo_id).await?;
        self.name_cache.remove(&model_ref.model);
        Ok(())
    }
}
