//! Git-based model registry using submodules
//! 
//! The .registry Git repository tracks all local models as submodules,
//! providing a Git-native way to manage model relationships and share
//! specific models with peers in a heterogeneous network.

use anyhow::{Result, Context, bail};
use git2::{Repository, Signature, SubmoduleUpdateOptions, IndexAddOption};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;
use crate::api::model_storage::ModelId;

/// Registry metadata stored in registry.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    pub version: String,
    pub models: HashMap<Uuid, RegisteredModel>,
}

/// Information about a registered model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub uuid: Uuid,
    pub name: String,
    pub model_type: ModelType,
    pub source: Option<String>,
    pub registered_at: i64,
    pub base_model: Option<String>, // For adapters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_state: Option<ModelErrorState>, // Track if model is broken
}

/// Error states for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelErrorState {
    /// Submodule exists but target directory is missing
    DereferencedSubmodule,
    /// Submodule points to invalid location
    InvalidSubmodule,
    /// Model directory exists but not properly initialized
    UninitializedModel,
    /// Other error with description
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Base,
    Adapter,
}

/// Git-based model registry using submodules
pub struct GitModelRegistry {
    /// Path to the .registry Git repository
    registry_path: PathBuf,

    /// Base models directory
    base_dir: PathBuf,

    /// Registry metadata
    metadata: RegistryMetadata,
}

impl GitModelRegistry {
    /// Initialize or open the registry
    pub async fn init(base_dir: PathBuf) -> Result<Self> {
        let registry_path = base_dir.join(".registry");
        
        // Create or open registry repo (temporarily for init)
        if registry_path.exists() {
            // Verify we can open it
            let _repo = Repository::open(&registry_path)
                .context("Failed to open registry repository")?;
        } else {
            // Initialize new registry
            tracing::info!("Initializing new model registry at {:?}", registry_path);
            
            fs::create_dir_all(&registry_path).await?;
            let repo = Repository::init(&registry_path)?;
            
            // Create directory structure
            fs::create_dir_all(registry_path.join("bases")).await?;
            fs::create_dir_all(registry_path.join("adapters")).await?;
            
            // Create initial registry.json
            let metadata = RegistryMetadata {
                version: "1.0.0".to_string(),
                models: HashMap::new(),
            };
            
            let json = serde_json::to_string_pretty(&metadata)?;
            fs::write(registry_path.join("registry.json"), json).await?;
            
            // Initial commit
            let sig = Signature::now("hyprstream", "hyprstream@local")?;
            let tree_id = {
                let mut index = repo.index()?;
                index.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
                index.write()?;
                index.write_tree()?
            };
            
            let tree = repo.find_tree(tree_id)?;
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                "Initialize model registry",
                &tree,
                &[],
            )?;

            // Repository created successfully, we'll open it on-demand now
        };
        
        // Load metadata
        let metadata_path = registry_path.join("registry.json");
        let metadata = if metadata_path.exists() {
            let content = fs::read_to_string(&metadata_path).await?;
            serde_json::from_str(&content)?
        } else {
            RegistryMetadata {
                version: "1.0.0".to_string(),
                models: HashMap::new(),
            }
        };
        
        Ok(Self {
            registry_path,
            base_dir,
            metadata,
        })
    }

    /// Open the repository on-demand
    fn open_repo(&self) -> Result<Repository> {
        Repository::open(&self.registry_path)
            .with_context(|| format!("Failed to open registry repository at {:?}", self.registry_path))
    }

    /// Register a model that was cloned to a UUID directory
    pub async fn register_model(
        &mut self,
        model_id: &ModelId,
        name: &str,
        source: Option<String>,
    ) -> Result<()> {
        // Check if UUID is already registered
        if self.metadata.models.contains_key(&model_id.0) {
            tracing::warn!("Model with UUID {} is already registered, skipping", model_id.0);
            return Ok(());
        }

        // Path to the actual model
        let model_path = self.base_dir.join(model_id.0.to_string());
        if !model_path.exists() {
            bail!("Model directory does not exist: {:?}", model_path);
        }

        // Add as submodule (relative path from .registry)
        let relative_path = format!("../{}", model_id.0);
        // Use UUID for submodule path to avoid name conflicts
        let submodule_path = format!("bases/{}", model_id.0);

        // Check if submodule already exists
        let repo = self.open_repo()?;
        if repo.find_submodule(&submodule_path).is_ok() {
            tracing::warn!("Submodule {} already exists, skipping", submodule_path);
        } else {
            // Add submodule
            repo.submodule(
                &relative_path,
                Path::new(&submodule_path),
                true, // use gitlink
            )?;
        }

        // Update metadata
        self.metadata.models.insert(model_id.0, RegisteredModel {
            uuid: model_id.0,
            name: name.to_string(),
            model_type: ModelType::Base,
            source,
            registered_at: chrono::Utc::now().timestamp(),
            base_model: None,
            error_state: None,
        });

        // Save metadata
        self.save_metadata().await?;

        // Commit changes
        self.commit_changes(&format!("Register model: {} ({})", name, model_id.0))?;

        tracing::info!("Registered model '{}' with UUID {}", name, model_id.0);

        Ok(())
    }
    
    /// Register an adapter branch in the registry
    pub async fn register_adapter(
        &mut self,
        base_model_uuid: &Uuid,
        adapter_name: &str,
        branch_uuid: uuid::Uuid,
    ) -> Result<()> {
        // Find base model by UUID
        let base_model = self.metadata.models.get(base_model_uuid)
            .ok_or_else(|| anyhow::anyhow!("Base model with UUID '{}' not found", base_model_uuid))?
            .clone();

        // Check if adapter UUID is already registered
        if self.metadata.models.contains_key(&branch_uuid) {
            tracing::warn!("Adapter with UUID {} is already registered, skipping", branch_uuid);
            return Ok(());
        }

        // The adapter is a branch in the base model, not a separate directory
        // We track it in metadata but don't create a submodule
        self.metadata.models.insert(branch_uuid, RegisteredModel {
            uuid: branch_uuid,  // Store the branch UUID
            name: adapter_name.to_string(),
            model_type: ModelType::Adapter,
            source: None,
            registered_at: chrono::Utc::now().timestamp(),
            base_model: Some(base_model.name.clone()),
            error_state: None,
        });

        // Save metadata
        self.save_metadata().await?;

        // Commit
        self.commit_changes(&format!("Register adapter branch: {} (branch {}) from {}",
            adapter_name, branch_uuid, base_model.name))?;

        tracing::info!("Registered adapter '{}' as branch {} of base model '{}'",
            adapter_name, branch_uuid, base_model.name);

        Ok(())
    }
    
    /// List all registered models
    pub fn list_models(&self) -> Result<Vec<(Uuid, RegisteredModel)>> {
        Ok(self.metadata.models.iter()
            .map(|(uuid, model)| (*uuid, model.clone()))
            .collect())
    }

    /// Get model info by UUID
    pub fn get_model(&self, uuid: &Uuid) -> Option<&RegisteredModel> {
        self.metadata.models.get(uuid)
    }

    /// Get model info by name (searches all models)
    pub fn get_model_by_name(&self, name: &str) -> Option<&RegisteredModel> {
        self.metadata.models.values()
            .find(|model| model.name == name)
    }
    
    /// Get shareable reference for a model (for peer sharing)
    pub fn get_shareable_ref(&self, name: &str) -> Result<ShareableModelRef> {
        let model = self.get_model_by_name(name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", name))?;

        // Get submodule info (now using UUID for submodule path)
        let submodule_path = match model.model_type {
            ModelType::Base => format!("bases/{}", model.uuid),
            ModelType::Adapter => format!("adapters/{}", model.uuid),
        };

        let repo = self.open_repo()?;
        let submodule = repo.find_submodule(&submodule_path)?;

        // Get the HEAD commit of the model
        let model_path = self.base_dir.join(model.uuid.to_string());
        let model_repo = Repository::open(&model_path)?;
        let head = model_repo.head()?;
        let commit = head.peel_to_commit()?;

        Ok(ShareableModelRef {
            name: model.name.clone(),
            model_type: model.model_type.clone(),
            uuid: model.uuid,
            git_commit: commit.id().to_string(),
            source: model.source.clone(),
        })
    }
    
    /// Import a shared model reference from a peer
    pub async fn import_shared_model(
        &mut self,
        share_ref: ShareableModelRef,
        git_url: &str,
    ) -> Result<ModelId> {
        // Generate our own UUID for this model
        let our_id = ModelId::new();
        let our_path = self.base_dir.join(our_id.0.to_string());
        
        // Clone the shared model
        tracing::info!("Importing shared model '{}' from {}", share_ref.name, git_url);
        Repository::clone(git_url, &our_path)?;
        
        // Register in our local registry with a peer prefix
        let our_name = format!("{}-peer", share_ref.name);
        self.register_model(&our_id, &our_name, Some(git_url.to_string())).await?;
        
        Ok(our_id)
    }
    
    /// Save metadata to disk
    async fn save_metadata(&self) -> Result<()> {
        let registry_path = self.base_dir.join(".registry");
        let metadata_path = registry_path.join("registry.json");
        
        let json = serde_json::to_string_pretty(&self.metadata)?;
        fs::write(metadata_path, json).await?;
        
        Ok(())
    }
    
    /// Commit changes to registry
    fn commit_changes(&self, message: &str) -> Result<()> {
        let sig = Signature::now("hyprstream", "hyprstream@local")?;
        
        // Stage all changes
        let repo = self.open_repo()?;
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;

        // Get HEAD as parent
        let parent = repo.head()?.peel_to_commit()?;
        
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&parent],
        )?;
        
        Ok(())
    }

    /// Validate registry and detect broken submodules
    pub async fn validate_registry(&mut self) -> Result<Vec<(Uuid, ModelErrorState)>> {
        let mut errors = Vec::new();
        let repo = self.open_repo()?;

        // Check all registered models in metadata
        for (uuid, model) in self.metadata.models.clone() {
            let mut error_state = None;

            // Check if model directory exists
            let model_path = self.base_dir.join(uuid.to_string());
            if !model_path.exists() {
                error_state = Some(ModelErrorState::DereferencedSubmodule);
            } else if !model_path.join(".git").exists() {
                error_state = Some(ModelErrorState::UninitializedModel);
            }

            // Check submodule status
            let submodule_path = match model.model_type {
                ModelType::Base => format!("bases/{}", uuid),
                ModelType::Adapter => format!("adapters/{}", uuid),
            };

            match repo.find_submodule(&submodule_path) {
                Ok(submodule) => {
                    // Check if submodule URL points to valid location
                    if let Some(url) = submodule.url() {
                        if !url.starts_with("../") && !url.starts_with("file://") {
                            error_state = Some(ModelErrorState::InvalidSubmodule);
                        }
                    }
                }
                Err(_) => {
                    // Submodule not found but model is registered
                    if error_state.is_none() {
                        error_state = Some(ModelErrorState::Other("Submodule missing".to_string()));
                    }
                }
            }

            // Update model's error state
            if let Some(state) = error_state.clone() {
                errors.push((uuid, state.clone()));
                if let Some(model_mut) = self.metadata.models.get_mut(&uuid) {
                    model_mut.error_state = Some(state);
                }
            } else {
                // Clear any previous error state
                if let Some(model_mut) = self.metadata.models.get_mut(&uuid) {
                    model_mut.error_state = None;
                }
            }
        }

        // Check for orphaned submodules (in Git but not in metadata)
        let mut submodule_iter = repo.submodules()?;
        for submodule in submodule_iter.iter() {
            let name = submodule.name().unwrap_or("");

            // Extract UUID from path (bases/UUID or adapters/UUID)
            if let Some(uuid_str) = name.split('/').nth(1) {
                if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                    if !self.metadata.models.contains_key(&uuid) {
                        // Found orphaned submodule - add it to metadata with error state
                        tracing::warn!("Found orphaned submodule: {}", uuid);

                        // Try to determine model info from submodule
                        let model_type = if name.starts_with("bases/") {
                            ModelType::Base
                        } else {
                            ModelType::Adapter
                        };

                        self.metadata.models.insert(uuid, RegisteredModel {
                            uuid,
                            name: format!("orphaned-{}", uuid),
                            model_type,
                            source: submodule.url().map(String::from),
                            registered_at: chrono::Utc::now().timestamp(),
                            base_model: None,
                            error_state: Some(ModelErrorState::Other("Orphaned submodule".to_string())),
                        });
                    }
                }
            }
        }

        // Save updated metadata
        self.save_metadata().await?;

        Ok(errors)
    }

    /// Rebuild registry from Git submodules (use when corrupted)
    pub async fn rebuild_from_git(&mut self) -> Result<()> {
        tracing::info!("Rebuilding registry from Git submodules...");

        let repo = self.open_repo()?;
        let mut new_models = HashMap::new();

        // Scan all submodules
        let submodules = repo.submodules()?;
        for submodule in submodules.iter() {
            let name = submodule.name().unwrap_or("");
            let url = submodule.url();

            // Extract UUID from path
            if let Some(uuid_str) = name.split('/').nth(1) {
                if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                    let model_type = if name.starts_with("bases/") {
                        ModelType::Base
                    } else if name.starts_with("adapters/") {
                        ModelType::Adapter
                    } else {
                        continue; // Skip unknown submodule types
                    };

                    // Check if model directory exists
                    let model_path = self.base_dir.join(uuid.to_string());
                    let error_state = if !model_path.exists() {
                        Some(ModelErrorState::DereferencedSubmodule)
                    } else if !model_path.join(".git").exists() {
                        Some(ModelErrorState::UninitializedModel)
                    } else {
                        None
                    };

                    // Try to get name from existing metadata or use UUID
                    let model_name = self.metadata.models.get(&uuid)
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| {
                            // Try to read model name from model directory if possible
                            if model_path.exists() {
                                // Could read from config.json or other metadata files
                                uuid.to_string()
                            } else {
                                uuid.to_string()
                            }
                        });

                    new_models.insert(uuid, RegisteredModel {
                        uuid,
                        name: model_name,
                        model_type,
                        source: url.map(String::from),
                        registered_at: self.metadata.models.get(&uuid)
                            .map(|m| m.registered_at)
                            .unwrap_or_else(|| chrono::Utc::now().timestamp()),
                        base_model: self.metadata.models.get(&uuid)
                            .and_then(|m| m.base_model.clone()),
                        error_state,
                    });
                }
            }
        }

        // Also check for models in directory but not in submodules
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Some(name) = entry.file_name().to_str() {
                        if let Ok(uuid) = Uuid::parse_str(name) {
                            if !new_models.contains_key(&uuid) {
                                // Model directory exists but no submodule
                                let model_path = entry.path();
                                if model_path.is_dir() && model_path.join(".git").exists() {
                                    tracing::warn!("Found unregistered model directory: {}", uuid);

                                    // Add with error state
                                    new_models.insert(uuid, RegisteredModel {
                                        uuid,
                                        name: uuid.to_string(),
                                        model_type: ModelType::Base,
                                        source: None,
                                        registered_at: chrono::Utc::now().timestamp(),
                                        base_model: None,
                                        error_state: Some(ModelErrorState::Other("Not in submodules".to_string())),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update metadata
        self.metadata.models = new_models;
        self.save_metadata().await?;

        // Commit the rebuilt registry
        if let Err(e) = self.commit_changes("Rebuild registry from Git submodules") {
            tracing::warn!("Failed to commit rebuilt registry: {}", e);
        }

        tracing::info!("Registry rebuilt with {} models", self.metadata.models.len());
        Ok(())
    }

    /// Auto-repair registry issues
    pub async fn auto_repair(&mut self) -> Result<()> {
        let errors = self.validate_registry().await?;

        if errors.is_empty() {
            tracing::info!("Registry validation passed, no repairs needed");
            return Ok(());
        }

        tracing::warn!("Found {} registry errors, attempting repairs...", errors.len());

        for (uuid, error_state) in errors {
            match error_state {
                ModelErrorState::DereferencedSubmodule => {
                    // Submodule exists but target missing - try to re-clone
                    if let Some(model) = self.metadata.models.get(&uuid) {
                        if let Some(source) = &model.source {
                            tracing::info!("Attempting to re-clone dereferenced model {} from {}", uuid, source);
                            // Note: Actual re-cloning would require access to git_downloader
                            // For now, just mark it as needing repair
                        }
                    }
                }
                ModelErrorState::InvalidSubmodule => {
                    // Remove and re-add submodule with correct path
                    tracing::info!("Fixing invalid submodule for {}", uuid);
                    // Implementation would go here
                }
                _ => {
                    tracing::warn!("Cannot auto-repair error for {}: {:?}", uuid, error_state);
                }
            }
        }

        Ok(())
    }
}

/// Reference to share a model with peers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareableModelRef {
    pub name: String,
    pub model_type: ModelType,
    pub uuid: Uuid,
    pub git_commit: String,
    pub source: Option<String>,
}