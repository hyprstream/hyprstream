//! Model sharing protocol for distributed network
//!
//! Enables selective sharing of models and adapters between
//! nodes in a heterogeneous network via Git.

use anyhow::{Result, Context, bail};
use tracing::info;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::storage::ModelId;
use git2db::{Git2DB as GitModelRegistry, GitManager, Git2DBConfig as GitConfig};
use git2db::clone_options::CloneOptions;

/// Shareable reference to a model or adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareableModelRef {
    /// Friendly name of the model
    pub name: String,
    
    /// Type of model (base or adapter)
    pub model_type: ModelType,
    
    /// Git URL for cloning (may be local path or remote)
    pub git_url: Option<String>,
    
    /// Current commit hash
    pub commit: String,
    
    /// Size in bytes (for network planning)
    pub size_bytes: u64,
    
    /// Performance metrics (optional)
    pub metrics: Option<ModelMetrics>,
    
    /// Signature for authenticity (optional)
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Base,
    Adapter { base_model: String },
}

/// Performance metrics for shared models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub loss: f32,
    pub accuracy: Option<f32>,
    pub training_steps: usize,
    pub evaluation_dataset: Option<String>,
}

/// Model sharing manager
pub struct ModelSharing {
    base_dir: PathBuf,
    registry: Arc<RwLock<GitModelRegistry>>,
    git_manager: Arc<GitManager>,
}

impl ModelSharing {
    /// Create new model sharing manager
    pub async fn new(base_dir: PathBuf) -> Result<Self> {
        Self::new_with_config(base_dir, GitConfig::default()).await
    }

    /// Create with custom Git configuration
    pub async fn new_with_config(base_dir: PathBuf, git_config: GitConfig) -> Result<Self> {
        let git_manager = Arc::new(GitManager::new(git_config));

        // Initialize Git registry (fail fast if unable to create)
        let registry = GitModelRegistry::open(base_dir.clone()).await
            .context("Failed to initialize Git registry for model sharing")?;

        Ok(Self {
            base_dir,
            registry: Arc::new(RwLock::new(registry)),
            git_manager,
        })
    }
    
    /// Create a shareable reference for a model using git2db APIs
    pub async fn create_share_ref(
        &self,
        model_name: &str,
        include_metrics: bool,
    ) -> Result<ShareableModelRef> {
        // ✅ Try to find model in registry first (git2db v2 API)
        let registry = self.registry.read().await;
        if let Ok(handle) = registry.repo_by_name(model_name) {
            // ✅ Use RepositoryHandle to get metadata (no git operations)
            if let Ok(model) = handle.metadata() {
                return self.create_ref_from_registry(model_name, model, include_metrics).await;
            }
        }

        // Fallback to direct path lookup for untracked models
        self.create_ref_from_path(model_name, include_metrics).await
    }
    
    /// Create reference from registry entry using git2db RepositoryHandle
    async fn create_ref_from_registry(
        &self,
        name: &str,
        model: &crate::git::RegisteredModel,
        include_metrics: bool,
    ) -> Result<ShareableModelRef> {
        // ✅ Use git2db metadata directly (no git operations needed)
        let model_path = &model.worktree_path;
        let git_url = Some(model.url.clone());

        // ✅ Get HEAD commit from metadata if available, else query git
        let commit = if let Some(ref oid_str) = model.current_oid {
            oid_str.clone()
        } else {
            // Fallback: query git if metadata doesn't have current_oid
            let repo = self.git_manager.get_repository(&model_path)
                .context("Failed to open model repository")?;
            let head = repo.head()?.peel_to_commit()?;
            head.id().to_string()
        };
        
        // Calculate size
        let size_bytes = self.calculate_model_size(&model_path).await?;
        
        // Determine model type
        let model_type = if model_path.join("adapter_config.json").exists() {
            // Read adapter config to get base model
            let config = tokio::fs::read_to_string(model_path.join("adapter_config.json")).await?;
            let config: serde_json::Value = serde_json::from_str(&config)?;
            let base_model = config.get("base_model")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            ModelType::Adapter { base_model }
        } else {
            ModelType::Base
        };
        
        // Load metrics if requested
        let metrics = if include_metrics {
            self.load_model_metrics(&model_path).await.ok()
        } else {
            None
        };
        
        Ok(ShareableModelRef {
            name: name.to_string(),
            model_type,
            git_url,
            commit,
            size_bytes,
            metrics,
            signature: None, // TODO: Implement signing
        })
    }
    
    /// Create reference from direct path
    async fn create_ref_from_path(
        &self,
        model_name: &str,
        include_metrics: bool,
    ) -> Result<ShareableModelRef> {
        // Try standard locations
        let possible_paths = vec![
            self.base_dir.join("base").join(model_name),
            self.base_dir.join("adapters").join(model_name),
            self.base_dir.join(model_name),
        ];
        
        for path in possible_paths {
            if path.exists() && path.join(".git").exists() {
                let repo = self.git_manager.get_repository(&path)?;
                let head = repo.head()?.peel_to_commit()?;
                
                let model_type = if path.join("adapter_config.json").exists() {
                    let config = tokio::fs::read_to_string(path.join("adapter_config.json")).await?;
                    let config: serde_json::Value = serde_json::from_str(&config)?;
                    let base_model = config.get("base_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    ModelType::Adapter { base_model }
                } else {
                    ModelType::Base
                };
                
                let git_url = repo.find_remote("origin")
                    .ok()
                    .and_then(|remote| remote.url().map(String::from));
                
                let metrics = if include_metrics {
                    self.load_model_metrics(&path).await.ok()
                } else {
                    None
                };
                
                return Ok(ShareableModelRef {
                    name: model_name.to_string(),
                    model_type,
                    git_url,
                    commit: head.id().to_string(),
                    size_bytes: self.calculate_model_size(&path).await?,
                    metrics,
                    signature: None,
                });
            }
        }
        
        bail!("Model '{}' not found", model_name)
    }
    
    /// Import a shared model from a peer
    pub async fn import_shared_model(
        &mut self,
        share_ref: ShareableModelRef,
        git_url: &str,
        local_name: Option<String>,
    ) -> Result<ModelId> {
        let name = local_name.unwrap_or_else(|| format!("{}-import", share_ref.name));
        
        // Determine target directory based on type
        let target_dir = match &share_ref.model_type {
            ModelType::Base => self.base_dir.join("base").join(&name),
            ModelType::Adapter { .. } => self.base_dir.join("adapters").join(&name),
        };
        
        // Check if already exists
        if target_dir.exists() {
            bail!("Model '{}' already exists locally", name);
        }
        
        info!("📥 Importing model '{}' from {}", share_ref.name, git_url);

        // Clone the repository with advanced options (shallow for efficiency)
        let clone_options = CloneOptions::builder()
            .shallow(true)
            .depth(1)
            .build();

        let repo = self.git_manager
            .clone_repository(git_url, &target_dir, Some(clone_options))
            .await
            .context("Failed to clone shared model")?;

        // Checkout specific commit
        let oid = git2::Oid::from_str(&share_ref.commit)?;
        let commit = repo.find_commit(oid)?;
        repo.checkout_tree(commit.as_object(), None)?;
        repo.set_head_detached(oid)?;
        
        // Generate local model ID
        let model_id = ModelId::new();

        // Register with local registry
        let mut registry = self.registry.write().await;
        let git2db_repo_id = git2db::RepoId::from_uuid(model_id.0);
        registry.register_repository(
            &git2db_repo_id,
            Some(name.to_string()),
            git_url.to_string(),
        ).await?;
        
        // Verify model integrity if signature provided
        if let Some(signature) = &share_ref.signature {
            self.verify_model_signature(&target_dir, signature).await?;
        }
        
        info!("✅ Imported model '{}' as '{}'", share_ref.name, name);
        info!("   Type: {:?}", share_ref.model_type);
        info!("   Commit: {}", share_ref.commit);
        info!("   Size: {:.2} GB", share_ref.size_bytes as f64 / 1_073_741_824.0);
        
        if let Some(metrics) = &share_ref.metrics {
            info!("   Performance: loss={:.4}, steps={}", metrics.loss, metrics.training_steps);
        }
        
        Ok(model_id)
    }
    
    /// Push local model to remote for sharing
    pub async fn push_to_remote(
        &self,
        model_name: &str,
        remote_url: &str,
        remote_name: Option<&str>,
    ) -> Result<()> {
        let model_path = self.find_model_path(model_name).await?;
        let repo = self.git_manager.get_repository(&model_path)?;
        
        let remote_name = remote_name.unwrap_or("origin");
        
        // Add or update remote
        match repo.find_remote(remote_name) {
            Ok(remote) => {
                repo.remote_set_url(remote_name, remote_url)?;
            }
            Err(_) => {
                repo.remote(remote_name, remote_url)?;
            }
        }
        
        // Push to remote
        let mut remote = repo.find_remote(remote_name)?;
        let refspecs: Vec<String> = vec!["refs/heads/main:refs/heads/main".to_string()];
        
        remote.push(&refspecs, None)?;
        
        info!("✅ Pushed model '{}' to {}", model_name, remote_url);
        
        Ok(())
    }
    
    /// List models available for sharing
    pub async fn list_shareable_models(&self) -> Result<Vec<(String, ModelType, u64)>> {
        let mut models = Vec::new();
        
        // Check base models
        let base_dir = self.base_dir.join("base");
        if base_dir.exists() {
            let mut entries = tokio::fs::read_dir(base_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.join(".git").exists() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let size = self.calculate_model_size(&path).await?;
                    models.push((name, ModelType::Base, size));
                }
            }
        }
        
        // Check adapters
        let adapters_dir = self.base_dir.join("adapters");
        if adapters_dir.exists() {
            let mut entries = tokio::fs::read_dir(adapters_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                if path.join(".git").exists() && path.join("adapter_config.json").exists() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    
                    let config = tokio::fs::read_to_string(path.join("adapter_config.json")).await?;
                    let config: serde_json::Value = serde_json::from_str(&config)?;
                    let base_model = config.get("base_model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    
                    let size = self.calculate_model_size(&path).await?;
                    models.push((name, ModelType::Adapter { base_model }, size));
                }
            }
        }
        
        Ok(models)
    }
    
    /// Calculate model size in bytes
    async fn calculate_model_size(&self, path: &Path) -> Result<u64> {
        let mut total_size = 0u64;
        
        let mut entries = tokio::fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                total_size += metadata.len();
            }
        }
        
        Ok(total_size)
    }
    
    /// Load model metrics from checkpoint
    async fn load_model_metrics(&self, model_path: &Path) -> Result<ModelMetrics> {
        let metrics_path = model_path.join(".checkpoints/latest/metadata.json");
        
        if metrics_path.exists() {
            let json = tokio::fs::read_to_string(&metrics_path).await?;
            let metadata: serde_json::Value = serde_json::from_str(&json)?;
            
            if let Some(metrics) = metadata.get("metrics") {
                Ok(ModelMetrics {
                    loss: metrics.get("loss")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32,
                    accuracy: metrics.get("validation_accuracy")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32),
                    training_steps: metrics.get("step")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    evaluation_dataset: metrics.get("evaluation_dataset")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                })
            } else {
                bail!("No metrics found in checkpoint metadata")
            }
        } else {
            bail!("No checkpoint metadata found")
        }
    }
    
    /// Verify model signature (stub for future implementation)
    async fn verify_model_signature(&self, _model_path: &Path, _signature: &str) -> Result<()> {
        // TODO: Implement cryptographic signature verification
        // For now, just log
        tracing::info!("Signature verification not yet implemented");
        Ok(())
    }
    
    /// Find model path by name
    async fn find_model_path(&self, model_name: &str) -> Result<PathBuf> {
        let possible_paths = vec![
            self.base_dir.join("base").join(model_name),
            self.base_dir.join("adapters").join(model_name),
            self.base_dir.join(model_name),
        ];
        
        for path in possible_paths {
            if path.exists() {
                return Ok(path);
            }
        }
        
        bail!("Model '{}' not found", model_name)
    }
}