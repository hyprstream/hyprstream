//! Simplified model storage that works with git-native registry

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

use super::model_ref::ModelRef;
use super::model_registry::SharedModelRegistry;

/// Model identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId(pub Uuid);

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

/// Model metadata file for storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataFile {
    pub model_id: ModelId,
    pub name: String,
    pub display_name: String,
    pub source_uri: String,
    pub architecture: Option<String>,
    pub parameters: Option<u64>,
    pub created_at: i64,
    pub last_accessed: i64,
}

impl ModelId {
    /// Create a new random model ID
    pub fn new() -> Self {
        ModelId(Uuid::new_v4())
    }
}

/// Model storage that works with the git-native registry
pub struct ModelStorage {
    base_dir: PathBuf,
    registry: Arc<SharedModelRegistry>,
}

impl ModelStorage {
    /// Create new model storage with registry
    pub fn new(base_dir: PathBuf, registry: Arc<SharedModelRegistry>) -> Self {
        Self {
            base_dir,
            registry,
        }
    }

    /// Get the registry
    pub fn registry(&self) -> Arc<SharedModelRegistry> {
        self.registry.clone()
    }

    /// Create with a new registry
    pub async fn create(base_dir: PathBuf) -> Result<Self> {
        Self::create_with_config(base_dir, git2db::config::Git2DBConfig::default()).await
    }

    /// Create with a new registry and custom git2db configuration
    pub async fn create_with_config(
        base_dir: PathBuf,
        _git2db_config: git2db::config::Git2DBConfig,
    ) -> Result<Self> {
        // Note: GitManager::global() now loads config from environment automatically
        // We don't need to explicitly initialize it here anymore

        // Use the models directory itself as the registry, not a subdirectory
        let registry = SharedModelRegistry::open(base_dir.clone(), None).await?;
        Ok(Self {
            base_dir,
            registry: Arc::new(registry),
        })
    }

    /// Get base directory
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get models directory
    pub fn get_models_dir(&self) -> PathBuf {
        // base_dir is already the models directory
        self.base_dir.clone()
    }

    /// Get model path by reference
    pub async fn get_model_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        self.registry.get_model_path(&model_ref).await
    }

    /// List all models
    pub async fn list_models(&self) -> Result<Vec<(ModelRef, ModelMetadata)>> {
        let mut result = Vec::new();

        // Look for models in the models directory
        let models_dir = self.get_models_dir();
        if models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    if let Ok(file_type) = entry.file_type() {
                        if file_type.is_dir() {
                            if let Some(name) = entry.file_name().to_str() {
                                // Skip hidden directories
                                if name.starts_with('.') {
                                    continue;
                                }
                                // Check if it's a git repository
                                let git_dir = entry.path().join(".git");
                                if git_dir.exists() {
                                    let model_ref = ModelRef::new(name.to_string());

                                    // Calculate size if possible
                                    let size_bytes = Self::calculate_dir_size(&entry.path()).ok();

                                    let metadata = ModelMetadata {
                                        name: name.to_string(),
                                        display_name: Some(name.to_string()),
                                        model_type: "base".to_string(),
                                        created_at: chrono::Utc::now().timestamp(),
                                        updated_at: chrono::Utc::now().timestamp(),
                                        size_bytes,
                                        tags: vec![],
                                    };
                                    result.push((model_ref, metadata));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Also check registry for submodules
        if let Ok(registry_models) = self.registry.list_models().await {
            for (name, _commit_id) in registry_models {
                // Don't duplicate if already found in directory scan
                if !result.iter().any(|(ref r, _)| r.model == name) {
                    let model_ref = ModelRef::new(name.clone());
                    let metadata = ModelMetadata {
                        name: name.clone(),
                        display_name: Some(name.clone()),
                        model_type: "base".to_string(),
                        created_at: chrono::Utc::now().timestamp(),
                        updated_at: chrono::Utc::now().timestamp(),
                        size_bytes: None,
                        tags: vec![],
                    };
                    result.push((model_ref, metadata));
                }
            }
        }

        Ok(result)
    }

    /// Calculate directory size
    fn calculate_dir_size(path: &Path) -> Result<u64> {
        let mut total_size = 0u64;
        for entry in walkdir::WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
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
        let model_ref = ModelRef::new(model_name.to_string());
        self.registry.get_model_path(&model_ref).await.is_ok()
    }

    /// Add a new model
    pub async fn add_model(&self, name: &str, source: &str) -> Result<()> {
        self.registry.add_model(name, source).await?;
        Ok(())
    }

    /// Update a model to a different version
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        self.registry.update_model(name, ref_spec).await
    }

    /// Get metadata by ID
    pub async fn get_metadata_by_id(&self, _id: &ModelId) -> Result<ModelMetadata> {
        bail!("UUID-based model lookup is deprecated. Use model names instead.")
    }

    /// Get model path by ID (deprecated)
    pub fn get_model_path_by_id(&self, _id: &ModelId) -> Result<PathBuf> {
        bail!("UUID-based model lookup is deprecated. Use model names instead.")
    }

    /// Get UUID model path
    pub fn get_uuid_model_path(&self, _id: &ModelId) -> PathBuf {
        self.base_dir.join("deprecated").join(_id.to_string())
    }

    /// List children
    pub async fn children(&self) -> Result<Vec<(ModelId, ModelMetadata)>> {
        // Return empty for backwards compatibility
        Ok(vec![])
    }

    /// CLI compatibility methods
    pub async fn list_local_models(&self) -> Result<Vec<(ModelRef, ModelMetadata)>> {
        self.list_models().await
    }

    pub async fn get_cache_stats(&self) -> Result<CacheStats> {
        Ok(CacheStats {
            total_size_bytes: 0,
        })
    }

    pub async fn register_with_git_registry(
        &self,
        model_id: &ModelId,
        name: &str,
        source: Option<String>,
    ) -> Result<()> {
        // CRITICAL FIX: Actually register the model with the git registry
        // This fixes the registry bypass bug where models were cloned but never registered

        tracing::info!("Registering model {} (ID: {}) with git registry", name, model_id);

        // Create a ModelRef for the registry operations
        let _model_ref = ModelRef::new(name.to_string());

        // Verify the model exists in the file system
        let model_path = self.get_models_dir().join(name);
        if !model_path.exists() {
            bail!("Model {} not found at expected path {:?} - cannot register", name, model_path);
        }

        // Check if it's actually a git repository
        if !model_path.join(".git").exists() {
            bail!("Model {} at {:?} is not a git repository", name, model_path);
        }

        // Register with the git registry as a submodule
        // The registry will handle adding it as a git submodule
        // Note: add_model takes name as a string, not ModelRef
        match self.registry.add_model(name, source.as_deref().unwrap_or("")).await {
            Ok(_) => {
                tracing::info!("Successfully registered model {} in git registry", name);
                Ok(())
            }
            Err(e) => {
                // This is a critical error - the model exists but isn't properly tracked
                tracing::error!("Failed to register model {} with git registry: {}", name, e);

                // Re-throw as a more specific error
                bail!("Failed to register model {} with git registry: {}. The model exists at {:?} but could not be added as a submodule. This may require manual intervention.", name, e, model_path)
            }
        }
    }

    pub async fn remove_metadata_by_id(&self, _id: &ModelId) -> Result<()> {
        Ok(())
    }

    pub async fn repair_metadata(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub total_size_bytes: u64,
}