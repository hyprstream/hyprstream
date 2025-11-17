//! Simplified model storage working directly with git2db

use anyhow::Result;
use git2db::{Git2DB, GitRef, RepoId};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::model_ref::{validate_model_name, ModelRef};

/// Model identifier (kept for backward compatibility)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId(pub Uuid);

impl Default for ModelId {
    fn default() -> Self {
        Self::new()
    }
}

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

        Ok(Self {
            base_dir,
            registry: Arc::new(RwLock::new(git2db)),
        })
    }

    /// Get models directory
    pub fn get_models_dir(&self) -> PathBuf {
        self.base_dir.clone()
    }

    /// Resolve model name to RepoId
    async fn resolve_name(&self, name: &str) -> Result<RepoId> {
        // Query git2db registry directly (O(n) but always accurate, no sync issues)
        // For ~100 models, this is negligible overhead
        let registry = self.registry.read().await;
        let repo_id = registry
            .list()
            .find(|t| t.name.as_ref() == Some(&name.to_string()))
            .map(|t| t.id.clone())
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", name))?;
        drop(registry);  // Explicitly drop the read guard before returning
        Ok(repo_id)
    }

    /// Get bare repository path by model reference
    pub async fn get_bare_repo_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        let repo_id = self.resolve_name(&model_ref.model).await?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;

        // The registry tracks the bare repo path (models/{name}/{name}.git/)
        Ok(handle.worktree()?.to_path_buf())
    }

    /// Get worktree path for a specific branch
    pub async fn get_worktree_path(&self, model_ref: &ModelRef, branch: &str) -> Result<PathBuf> {
        let repo_id = self.resolve_name(&model_ref.model).await?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;

        // Get bare repo path from registry
        let bare_repo_path = handle.worktree()?;

        // Navigate from bare repo to worktrees directory
        // From models/{name}/{name}.git/ to models/{name}/worktrees/{branch}
        let repo_dir = bare_repo_path.parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid bare repo path"))?;

        let worktrees_dir = repo_dir.join("worktrees");
        // Use canonical branch path conversion
        let worktree_path = worktrees_dir.join(branch);

        Ok(worktree_path)
    }

    /// Get worktree path for the model reference
    /// If the model_ref specifies a branch, returns that branch's worktree path.
    /// Otherwise, returns the default branch worktree path.
    pub async fn get_model_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        // Resolve to branch name (avoiding unnecessary clones)
        let branch = match &model_ref.git_ref {
            GitRef::Branch(ref name) => name.as_str(),
            GitRef::DefaultBranch | _ => {
                if !matches!(model_ref.git_ref, GitRef::DefaultBranch) {
                    tracing::warn!(
                        "Model reference specifies non-branch git ref {:?}, using default branch",
                        model_ref.git_ref
                    );
                }
                // Need to get default branch (returns owned String)
                return self.get_worktree_path(
                    model_ref,
                    &self.get_default_branch(model_ref).await?
                ).await;
            }
        };

        self.get_worktree_path(model_ref, branch).await
    }

    /// List all models as worktree references (model:branch format)
    ///
    /// This returns all available worktrees across all models, formatted as
    /// "model:branch" references. Base models without explicit branches are not included.
    pub async fn list_models(&self) -> Result<Vec<(ModelRef, ModelMetadata)>> {
        let mut result = Vec::new();
        let registry = self.registry.read().await;

        for tracked in registry.list() {
            if let Some(name) = &tracked.name {
                let base_ref = ModelRef::new(name.clone());

                // Enumerate all worktrees for this model
                match self.list_worktrees(&base_ref).await {
                    Ok(worktrees) => {
                        for branch_name in worktrees {
                            // Create model:branch reference
                            let model_ref = ModelRef::with_ref(
                                name.clone(),
                                git2db::GitRef::Branch(branch_name.clone())
                            );

                            // Calculate size from worktree path
                            let worktree_path = self.get_worktree_path(&base_ref, &branch_name).await.ok();
                            let size_bytes = if let Some(path) = &worktree_path {
                                if path.exists() {
                                    Self::calculate_dir_size(path).ok()
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            // Build metadata
                            let metadata = ModelMetadata {
                                name: name.clone(),
                                display_name: Some(format!("{}:{}", name, branch_name)),
                                model_type: "worktree".to_string(),
                                created_at: chrono::Utc::now().timestamp(),
                                updated_at: chrono::Utc::now().timestamp(),
                                size_bytes,
                                tags: vec![],
                            };

                            result.push((model_ref, metadata));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to list worktrees for model {}: {}", name, e);
                        // Continue with other models
                    }
                }
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
        let registry = self.registry.read().await;
        let exists = registry
            .list()
            .any(|t| t.name.as_ref() == Some(&model_name.to_string()));
        drop(registry);  // Explicitly drop the read guard
        exists
    }

    /// Add a new model
    pub async fn add_model(&self, name: &str, source: &str) -> Result<()> {
        validate_model_name(name)?;

        let mut registry = self.registry.write().await;
        let _repo_id = registry
            .clone(source)
            .name(name)
            .depth(1)
            .exec()
            .await?;

        Ok(())
    }

    /// Update a model to a different version
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        let repo_id = self.resolve_name(name).await?;
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
    pub async fn resolve_repo_id(&self, model_ref: &ModelRef) -> Result<RepoId> {
        self.resolve_name(&model_ref.model).await
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
        let repo_id = self.resolve_name(&model_ref.model).await?;
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
        let repo_id = self.resolve_name(&model_ref.model).await?;
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
        let repo_id = self.resolve_name(&model_ref.model).await?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;
        Ok(handle.default_branch()?)
    }

    /// Remove model
    pub async fn remove_model(&self, model_ref: &ModelRef) -> Result<()> {
        let repo_id = self.resolve_name(&model_ref.model).await?;
        let mut registry = self.registry.write().await;
        registry.remove_repository(&repo_id).await?;
        Ok(())
    }

    /// Create a new worktree for a model
    pub async fn create_worktree(&self, model_ref: &ModelRef, branch: &str) -> Result<PathBuf> {
        let bare_repo_path = self.get_bare_repo_path(model_ref).await?;
        let worktree_path = self.get_worktree_path(model_ref, branch).await?;

        if worktree_path.exists() {
            return Err(anyhow::anyhow!("Worktree already exists at {:?}", worktree_path));
        }

        // Use git2db's high-level worktree creation with automatic directory handling
        tracing::info!(
            "Creating worktree for {} at {} (branch: {})",
            model_ref.model,
            worktree_path.display(),
            branch
        );

        // Create the worktree using git2db's automatic directory management
        let _worktree_handle = git2db::GitManager::global()
            .create_worktree(&bare_repo_path, &worktree_path, branch)
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to create worktree at {}: {}", worktree_path.display(), e)
            })?;

        tracing::info!("Successfully created worktree at {}", worktree_path.display());
        Ok(worktree_path)
    }

    /// List all worktrees for a model
    pub async fn list_worktrees(&self, model_ref: &ModelRef) -> Result<Vec<String>> {
        let bare_repo_path = self.get_bare_repo_path(model_ref).await?;

        // Use git2db's high-level worktree listing with automatic filtering
        let worktrees = git2db::GitManager::global()
            .list_worktrees(&bare_repo_path)
            .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))?;

        Ok(worktrees)
    }


    /// Remove a worktree for a model
    pub async fn remove_worktree(&self, model_ref: &ModelRef, branch: &str) -> Result<()> {
        let bare_repo_path = self.get_bare_repo_path(model_ref).await?;

        // Use git2db's high-level worktree removal with automatic cleanup
        git2db::GitManager::global()
            .remove_worktree(&bare_repo_path, branch, Some(&self.base_dir))
            .map_err(|e| anyhow::anyhow!("Failed to remove worktree: {}", e))?;

        Ok(())
    }
}
