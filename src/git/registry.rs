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
    pub models: HashMap<String, RegisteredModel>,
}

/// Information about a registered model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub uuid: Uuid,
    pub model_type: ModelType,
    pub source: Option<String>,
    pub registered_at: i64,
    pub base_model: Option<String>, // For adapters
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
        // Check if already registered
        if self.metadata.models.contains_key(name) {
            bail!("Model '{}' is already registered", name);
        }
        
        // Path to the actual model
        let model_path = self.base_dir.join(model_id.0.to_string());
        if !model_path.exists() {
            bail!("Model directory does not exist: {:?}", model_path);
        }
        
        // Add as submodule (relative path from .registry)
        let relative_path = format!("../{}", model_id.0);
        let submodule_path = format!("bases/{}", name);
        
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
        self.metadata.models.insert(name.to_string(), RegisteredModel {
            uuid: model_id.0,
            model_type: ModelType::Base,
            source,
            registered_at: chrono::Utc::now().timestamp(),
            base_model: None,
        });
        
        // Save metadata
        self.save_metadata().await?;
        
        // Commit changes
        self.commit_changes(&format!("Register model: {}", name))?;
        
        tracing::info!("Registered model '{}' with UUID {}", name, model_id.0);
        
        Ok(())
    }
    
    /// Register an adapter branch in the registry
    pub async fn register_adapter(
        &mut self,
        base_model_name: &str,
        adapter_name: &str,
        branch_uuid: uuid::Uuid,
    ) -> Result<()> {
        // Find base model
        let base_model = self.metadata.models.get(base_model_name)
            .ok_or_else(|| anyhow::anyhow!("Base model '{}' not found", base_model_name))?
            .clone();
        
        // The adapter is a branch in the base model, not a separate directory
        // We track it in metadata but don't create a submodule
        self.metadata.models.insert(adapter_name.to_string(), RegisteredModel {
            uuid: branch_uuid,  // Store the branch UUID
            model_type: ModelType::Adapter,
            source: None,
            registered_at: chrono::Utc::now().timestamp(),
            base_model: Some(base_model_name.to_string()),
        });
        
        // Save metadata
        self.save_metadata().await?;
        
        // Commit
        self.commit_changes(&format!("Register adapter branch: {} (branch {}) from {}", 
            adapter_name, branch_uuid, base_model_name))?;
        
        tracing::info!("Registered adapter '{}' as branch {} of base model '{}'", 
            adapter_name, branch_uuid, base_model_name);
        
        Ok(())
    }
    
    /// List all registered models
    pub fn list_models(&self) -> Result<Vec<(String, RegisteredModel)>> {
        Ok(self.metadata.models.iter()
            .map(|(name, model)| (name.clone(), model.clone()))
            .collect())
    }
    
    /// Get model info by name
    pub fn get_model(&self, name: &str) -> Option<&RegisteredModel> {
        self.metadata.models.get(name)
    }
    
    /// Get shareable reference for a model (for peer sharing)
    pub fn get_shareable_ref(&self, name: &str) -> Result<ShareableModelRef> {
        let model = self.metadata.models.get(name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", name))?;
        
        // Get submodule info
        let submodule_path = match model.model_type {
            ModelType::Base => format!("bases/{}", name),
            ModelType::Adapter => format!("adapters/{}", name),
        };
        
        let repo = self.open_repo()?;
        let submodule = repo.find_submodule(&submodule_path)?;
        
        // Get the HEAD commit of the model
        let model_path = self.base_dir.join(model.uuid.to_string());
        let model_repo = Repository::open(&model_path)?;
        let head = model_repo.head()?;
        let commit = head.peel_to_commit()?;
        
        Ok(ShareableModelRef {
            name: name.to_string(),
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