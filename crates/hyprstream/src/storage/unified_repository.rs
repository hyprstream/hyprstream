//! Unified repository pattern implementation for model storage
//!
//! This module implements the Repository pattern to provide a single source of truth
//! for model management, fixing the dual authority anti-pattern in the current architecture.

use anyhow::{Result, Context, bail};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn, error};

use super::{
    ModelRef, ModelId, ModelMetadata, ModelStatus, CheckoutOptions, CheckoutResult,
    SharedModelRegistry, ModelStorage, GitModelSource, StorageError, StorageResult,
};

/// Unified model information combining registry and file system data
#[derive(Debug, Clone)]
pub struct UnifiedModelInfo {
    pub model_ref: ModelRef,
    pub model_id: Option<ModelId>,
    pub metadata: ModelMetadata,
    pub path: PathBuf,
    pub is_registered: bool,
    pub has_local_changes: bool,
}

/// Model source for cloning operations
#[derive(Debug, Clone)]
pub enum ModelSource {
    Git { url: String, branch: Option<String> },
    Local { path: PathBuf },
    Registry { name: String },
}

/// Repository trait defining the contract for model management
#[async_trait]
pub trait ModelRepository: Send + Sync {
    /// List all available models with unified information
    async fn list(&self) -> Result<Vec<UnifiedModelInfo>>;

    /// Get a specific model
    async fn get(&self, model_ref: &ModelRef) -> Result<Option<UnifiedModelInfo>>;

    /// Add a new model (handles both cloning and registration)
    async fn add(&self, source: ModelSource) -> Result<ModelId>;

    /// Remove a model (from both file system and registry)
    async fn remove(&self, model_ref: &ModelRef) -> Result<()>;

    /// Get model status
    async fn status(&self, model_ref: &ModelRef) -> Result<ModelStatus>;

    /// Checkout a specific version
    async fn checkout(&self, model_ref: &ModelRef, options: CheckoutOptions) -> Result<CheckoutResult>;

    /// Commit changes
    async fn commit(&self, model_ref: &ModelRef, message: &str, stage_all: bool) -> Result<git2::Oid>;

    /// Synchronize registry with file system
    async fn synchronize(&self) -> Result<SyncReport>;
}

/// Report from synchronization operations
#[derive(Debug, Default)]
pub struct SyncReport {
    pub models_registered: Vec<String>,
    pub models_removed: Vec<String>,
    pub errors: Vec<(String, String)>,
}

/// Unified repository implementation
pub struct UnifiedModelRepository {
    base_dir: PathBuf,
    models_dir: PathBuf,
    registry: Arc<SharedModelRegistry>,
    git_source: Arc<GitModelSource>,
}

impl UnifiedModelRepository {
    /// Create a new unified repository
    pub fn new(
        base_dir: PathBuf,
        registry: Arc<SharedModelRegistry>,
    ) -> Result<Self> {
        let models_dir = base_dir.join("models");
        std::fs::create_dir_all(&models_dir)?;

        let git_source = Arc::new(GitModelSource::new(models_dir.clone()));

        Ok(Self {
            base_dir,
            models_dir,
            registry,
            git_source,
        })
    }

    /// Discover models from the file system
    async fn discover_filesystem_models(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut models = Vec::new();

        let entries = std::fs::read_dir(&self.models_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Check if it's a git repository
                if path.join(".git").exists() {
                    if let Some(name) = path.file_name() {
                        let name = name.to_string_lossy().to_string();
                        models.push((name, path));
                    }
                }
            }
        }

        Ok(models)
    }

    /// Check if a model is registered in the git registry
    async fn is_registered(&self, model_name: &str) -> bool {
        let model_ref = ModelRef::new(model_name.to_string());
        self.registry.get_model_info(&model_ref).await.is_ok()
    }

    /// Register a model with the git registry
    async fn register_model(&self, name: &str, source: Option<&str>) -> Result<()> {
        info!("Registering model {} with git registry", name);

        let model_ref = ModelRef::new(name.to_string());
        let model_path = self.models_dir.join(name);

        // Verify the model exists locally
        if !model_path.exists() {
            return Err(StorageError::ModelNotFoundAtPath {
                name: name.to_string(),
                path: model_path,
            }.into());
        }

        // Add to registry as a submodule
        // Note: add_model takes name as a string, not ModelRef
        self.registry
            .add_model(name, source.unwrap_or(""))
            .await
            .context(format!("Failed to register model {}", name))?;

        info!("Successfully registered model {} in git registry", name);
        Ok(())
    }
}

#[async_trait]
impl ModelRepository for UnifiedModelRepository {
    async fn list(&self) -> Result<Vec<UnifiedModelInfo>> {
        let mut unified_models = Vec::new();

        // 1. Get all models from the file system
        let fs_models = self.discover_filesystem_models().await?;

        // 2. Get all models from the registry
        let registry_models = self.registry.list_models().await?;

        // 3. Create a unified view
        for (name, path) in fs_models {
            let model_ref = ModelRef::new(name.clone());
            let is_registered = self.is_registered(&name).await;

            // Try to get status if registered
            let has_local_changes = if is_registered {
                match self.registry.status(&model_ref).await {
                    Ok(status) => status.is_dirty,
                    Err(_) => false,
                }
            } else {
                false
            };

            // Create metadata
            let metadata = ModelMetadata {
                name: name.clone(),
                display_name: Some(name.clone()),
                model_type: "git".to_string(),
                created_at: 0, // Would need to read from git
                updated_at: 0,
                size_bytes: None,
                tags: vec![],
            };

            unified_models.push(UnifiedModelInfo {
                model_ref,
                model_id: None, // Legacy ID support
                metadata,
                path,
                is_registered,
                has_local_changes,
            });
        }

        // 4. Check for models in registry but not in file system (inconsistent state)
        // Note: registry_models returns (String, Oid) not (ModelRef, _)
        for (model_name, _commit_id) in registry_models {
            let model_path = self.models_dir.join(&model_name);
            if !model_path.exists() {
                warn!(
                    "Model {} is in registry but not in file system - inconsistent state",
                    model_name
                );
            }
        }

        Ok(unified_models)
    }

    async fn get(&self, model_ref: &ModelRef) -> Result<Option<UnifiedModelInfo>> {
        let model_path = self.models_dir.join(&model_ref.model);

        if !model_path.exists() {
            return Ok(None);
        }

        let is_registered = self.is_registered(&model_ref.model).await;
        let has_local_changes = if is_registered {
            match self.registry.status(model_ref).await {
                Ok(status) => status.is_dirty,
                Err(_) => false,
            }
        } else {
            false
        };

        let metadata = ModelMetadata {
            name: model_ref.model.clone(),
            display_name: Some(model_ref.model.clone()),
            model_type: "git".to_string(),
            created_at: 0,
            updated_at: 0,
            size_bytes: None,
            tags: vec![],
        };

        Ok(Some(UnifiedModelInfo {
            model_ref: model_ref.clone(),
            model_id: None,
            metadata,
            path: model_path,
            is_registered,
            has_local_changes,
        }))
    }

    async fn add(&self, source: ModelSource) -> Result<ModelId> {
        match source {
            ModelSource::Git { url, branch } => {
                info!("Cloning model from {}", url);

                // 1. Clone the model
                let (model_id, model_path) = if let Some(branch) = branch {
                    self.git_source.clone_ref(&url, &branch).await?
                } else {
                    self.git_source.clone_model(&url).await?
                };

                // 2. Extract model name
                let model_name = model_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;

                // 3. CRITICAL: Register with git registry
                match self.register_model(model_name, Some(&url)).await {
                    Ok(_) => info!("Model {} successfully registered", model_name),
                    Err(e) => {
                        // Registration failed - we should clean up
                        error!("Failed to register model {}: {}", model_name, e);

                        // Attempt cleanup
                        if let Err(cleanup_err) = std::fs::remove_dir_all(&model_path) {
                            error!("Failed to clean up after registration failure: {}", cleanup_err);
                        }

                        return Err(e);
                    }
                }

                Ok(model_id)
            }
            ModelSource::Local { path } => {
                bail!("Local model import not yet implemented")
            }
            ModelSource::Registry { name } => {
                bail!("Registry model cloning not yet implemented")
            }
        }
    }

    async fn remove(&self, model_ref: &ModelRef) -> Result<()> {
        info!("Removing model {}", model_ref.model);

        // 1. Remove from registry if registered
        // TODO: The registry doesn't currently have a remove_model method
        // This would need to be implemented to remove submodules
        if self.is_registered(&model_ref.model).await {
            warn!("Registry removal not yet implemented - only removing from file system");
            // self.registry.remove_model(model_ref).await?;
        }

        // 2. Remove from file system
        let model_path = self.models_dir.join(&model_ref.model);
        if model_path.exists() {
            std::fs::remove_dir_all(model_path)?;
        }

        Ok(())
    }

    async fn status(&self, model_ref: &ModelRef) -> Result<ModelStatus> {
        if !self.is_registered(&model_ref.model).await {
            bail!("Model {} is not registered in git registry", model_ref.model);
        }

        self.registry.status(model_ref).await
    }

    async fn checkout(&self, model_ref: &ModelRef, options: CheckoutOptions) -> Result<CheckoutResult> {
        if !self.is_registered(&model_ref.model).await {
            bail!("Model {} is not registered in git registry", model_ref.model);
        }

        self.registry.checkout(model_ref, options).await
    }

    async fn commit(&self, model_ref: &ModelRef, message: &str, stage_all: bool) -> Result<git2::Oid> {
        if !self.is_registered(&model_ref.model).await {
            bail!("Model {} is not registered in git registry", model_ref.model);
        }

        self.registry.commit_model(model_ref, message, stage_all).await
    }

    async fn synchronize(&self) -> Result<SyncReport> {
        let mut report = SyncReport::default();

        info!("Synchronizing file system with git registry");

        // 1. Find all models in file system
        let fs_models = self.discover_filesystem_models().await?;

        // 2. Register any unregistered models
        for (name, _path) in fs_models {
            if !self.is_registered(&name).await {
                info!("Found unregistered model: {}", name);

                match self.register_model(&name, None).await {
                    Ok(_) => {
                        info!("Successfully registered model {}", name);
                        report.models_registered.push(name);
                    }
                    Err(e) => {
                        error!("Failed to register model {}: {}", name, e);
                        report.errors.push((name, e.to_string()));
                    }
                }
            }
        }

        // 3. Check registry for models without file system presence
        let registry_models = self.registry.list_models().await?;
        for (model_name, _commit_id) in registry_models {
            let model_path = self.models_dir.join(&model_name);
            if !model_path.exists() {
                warn!("Model {} in registry but not in file system - needs manual cleanup", model_name);
                report.models_removed.push(model_name.clone());
                // TODO: Implement remove_model in registry to clean up orphaned submodules
                // match self.registry.remove_model(&model_ref).await {
                //     Ok(_) => {
                //         report.models_removed.push(model_name);
                //     }
                //     Err(e) => {
                //         report.errors.push((model_name, e.to_string()));
                //     }
                // }
            }
        }

        info!("Synchronization complete: {} registered, {} removed, {} errors",
              report.models_registered.len(),
              report.models_removed.len(),
              report.errors.len());

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_repository() -> Result<(UnifiedModelRepository, TempDir)> {
        let temp_dir = TempDir::new()?;
        let base_dir = temp_dir.path().to_path_buf();

        let registry_dir = base_dir.join("registry");
        let registry = SharedModelRegistry::open(registry_dir, None).await?;
        let repo = UnifiedModelRepository::new(base_dir, Arc::new(registry))?;

        Ok((repo, temp_dir))
    }

    #[tokio::test]
    async fn test_empty_repository() {
        let (repo, _temp) = create_test_repository().await.unwrap();
        let models = repo.list().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_synchronization() {
        let (repo, _temp) = create_test_repository().await.unwrap();

        // Create a model directory without registering it
        let unregistered_model = repo.models_dir.join("unregistered-model");
        std::fs::create_dir_all(&unregistered_model).unwrap();

        // Initialize it as a git repo
        git2::Repository::init(&unregistered_model).unwrap();

        // Run synchronization
        let report = repo.synchronize().await.unwrap();

        assert_eq!(report.models_registered.len(), 1);
        assert_eq!(report.models_registered[0], "unregistered-model");
    }
}