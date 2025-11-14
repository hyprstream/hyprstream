//! Simplified model storage working directly with git2db

use anyhow::Result;
use dashmap::DashMap;
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

    /// Get bare repository path by model reference
    pub async fn get_bare_repo_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        let repo_id = self.resolve_name(&model_ref.model)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&repo_id)?;

        // The registry tracks the bare repo path (models/{name}/{name}.git/)
        Ok(handle.worktree()?.to_path_buf())
    }

    /// Get worktree path for a specific branch
    pub async fn get_worktree_path(&self, model_ref: &ModelRef, branch: &str) -> Result<PathBuf> {
        let repo_id = self.resolve_name(&model_ref.model)?;
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
                match self.list_worktrees_with_metadata(&base_ref).await {
                    Ok(worktrees) => {
                        for (branch_name, wt_meta_opt) in worktrees {
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

                            // Build metadata with worktree info
                            let mut metadata = ModelMetadata {
                                name: name.clone(),
                                display_name: Some(format!("{}:{}", name, branch_name)),
                                model_type: "worktree".to_string(),
                                created_at: wt_meta_opt.as_ref()
                                    .map(|m| m.created_at.timestamp())
                                    .unwrap_or_else(|| chrono::Utc::now().timestamp()),
                                updated_at: chrono::Utc::now().timestamp(),
                                size_bytes,
                                tags: vec![],
                            };

                            // Enrich with worktree metadata
                            if let Some(wt_meta) = wt_meta_opt {
                                // Add storage driver info
                                metadata.tags.push(format!("driver:{}", wt_meta.storage_driver));
                                if let Some(backend) = &wt_meta.backend {
                                    metadata.tags.push(format!("backend:{}", backend));
                                }

                                // Add space savings info
                                if let Some(saved) = wt_meta.space_saved_bytes {
                                    metadata.tags.push(format!("saved:{}", Self::format_bytes(saved)));
                                }

                                // Add age info
                                let age = super::format_duration(wt_meta.age());
                                metadata.tags.push(format!("age:{}", age));
                            }

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

    /// Helper function to format bytes as human-readable string
    fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        if unit_idx == 0 {
            format!("{} {}", bytes, UNITS[unit_idx])
        } else {
            format!("{:.1} {}", size, UNITS[unit_idx])
        }
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

    /// Create a new worktree for a model
    pub async fn create_worktree(&self, model_ref: &ModelRef, branch: &str) -> Result<PathBuf> {
        let bare_repo_path = self.get_bare_repo_path(model_ref).await?;
        let worktree_path = self.get_worktree_path(model_ref, branch).await?;

        if worktree_path.exists() {
            return Err(anyhow::anyhow!("Worktree already exists at {:?}", worktree_path));
        }

        // Create parent directories for hierarchical branches (e.g., feature/ui)
        if let Some(parent) = worktree_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // ATOMIC OPERATION: Create worktree with metadata - rollback on any failure
        tracing::info!(
            "Creating worktree for {} at {} (branch: {})",
            model_ref.model,
            worktree_path.display(),
            branch
        );

        // Create the worktree (this already has rollback for LFS failures)
        let handle = git2db::GitManager::global()
            .create_worktree(&bare_repo_path, &worktree_path, branch)
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to create worktree at {}: {}", worktree_path.display(), e)
            })?;

        // Get worktree metadata from handle
        let wt_metadata = handle.metadata();

        // Create and save worktree metadata
        let mut metadata = super::WorktreeMetadata::new(
            branch.to_string(),
            model_ref.git_ref.to_ref_string(),
            wt_metadata.strategy_name.clone(),
        );

        // Add backend info and space savings from git2db handle
        metadata.backend = wt_metadata.backend_info;
        metadata.space_saved_bytes = wt_metadata.space_saved_bytes;

        // ATOMIC: Save metadata - rollback worktree on failure
        if let Err(e) = metadata.save(&worktree_path) {
            tracing::error!("Failed to save worktree metadata: {}", e);
            tracing::info!("Rolling back worktree creation due to metadata save failure");

            // ROLLBACK: Clean up the worktree
            handle.cleanup().unwrap_or_else(|cleanup_err| {
                tracing::error!("Failed to cleanup worktree during rollback: {}", cleanup_err);
            });

            return Err(anyhow::anyhow!(
                "Failed to save worktree metadata: {}. Worktree has been rolled back.",
                e
            ));
        }

        tracing::info!("Successfully created worktree at {}", worktree_path.display());
        Ok(worktree_path)
    }

    /// List all worktrees for a model
    pub async fn list_worktrees(&self, model_ref: &ModelRef) -> Result<Vec<String>> {
        let bare_repo_path = self.get_bare_repo_path(model_ref).await?;

        // Navigate to worktrees directory
        let repo_dir = bare_repo_path.parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid bare repo path"))?;
        let worktrees_dir = repo_dir.join("worktrees");

        if !worktrees_dir.exists() {
            return Ok(Vec::new());
        }

        let mut worktrees = Vec::new();
        for entry in std::fs::read_dir(worktrees_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    // Skip hidden directories (starting with .)
                    if name.starts_with('.') {
                        continue;
                    }
                    worktrees.push(name.to_string());
                }
            }
        }

        Ok(worktrees)
    }

    /// List all worktrees for a model with metadata
    pub async fn list_worktrees_with_metadata(
        &self,
        model_ref: &ModelRef,
    ) -> Result<Vec<(String, Option<super::WorktreeMetadata>)>> {
        let bare_repo_path = self.get_bare_repo_path(model_ref).await?;

        // Navigate to worktrees directory
        let repo_dir = bare_repo_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid bare repo path"))?;
        let worktrees_dir = repo_dir.join("worktrees");

        if !worktrees_dir.exists() {
            return Ok(Vec::new());
        }

        let mut worktrees = Vec::new();
        for entry in std::fs::read_dir(worktrees_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    // Skip hidden directories (starting with .)
                    if name.starts_with('.') {
                        continue;
                    }
                    let worktree_path = entry.path();
                    let metadata = super::WorktreeMetadata::try_load(&worktree_path);
                    worktrees.push((name.to_string(), metadata));
                }
            }
        }

        Ok(worktrees)
    }

    /// Remove a worktree for a model
    pub async fn remove_worktree(&self, model_ref: &ModelRef, branch: &str) -> Result<()> {
        let worktree_path = self.get_worktree_path(model_ref, branch).await?;

        if !worktree_path.exists() {
            return Err(anyhow::anyhow!("Worktree does not exist at {:?}", worktree_path));
        }

        // Remove the worktree directory
        std::fs::remove_dir_all(&worktree_path)?;

        // TODO: Also prune from git worktree list if needed

        Ok(())
    }
}
