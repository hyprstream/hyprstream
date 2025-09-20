//! Adapter-specific storage management
//! 
//! Handles LoRA adapters as branches of base models,
//! providing optimized workflows for adapter training.

use anyhow::{Result, Context, bail};
use git2::{Repository, Signature, IndexAddOption};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;
use chrono::Utc;
use crate::api::model_storage::ModelId;
use crate::git::{BranchManager, BranchInfo};

/// Adapter identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AdapterId(pub Uuid);

impl AdapterId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl std::fmt::Display for AdapterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Adapter configuration stored in adapter_config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub adapter_id: AdapterId,
    pub adapter_name: String,
    pub base_model: String,
    pub base_model_path: String,
    pub adapter_type: String,
    pub lora_r: Option<u32>,
    pub lora_alpha: Option<f32>,
    pub target_modules: Vec<String>,
    pub created_at: chrono::DateTime<Utc>,
    pub training_config: Option<serde_json::Value>,
}

/// Active adapter training session
pub struct AdapterSession {
    pub adapter_id: AdapterId,
    pub adapter_path: PathBuf,
    pub base_model_path: PathBuf,
    pub config: AdapterConfig,
    pub is_training: bool,
}

/// Adapter storage manager using branch-based storage
pub struct AdapterStorage {
    base_dir: PathBuf,
    working_dir: PathBuf,  // Directory for adapter worktrees
    registry: std::sync::Arc<tokio::sync::RwLock<Option<crate::git::GitModelRegistry>>>,
}

impl AdapterStorage {
    /// Create new adapter storage manager
    pub async fn new(base_dir: PathBuf) -> Result<Self> {
        let working_dir = base_dir.join("working");
        fs::create_dir_all(&working_dir).await?;
        
        // Try to initialize Git registry
        let registry = match crate::git::GitModelRegistry::init(base_dir.clone()).await {
            Ok(reg) => Some(reg),
            Err(e) => {
                tracing::warn!("Failed to initialize Git registry for adapters: {}", e);
                None
            }
        };
        
        Ok(Self {
            base_dir,
            working_dir,
            registry: std::sync::Arc::new(tokio::sync::RwLock::new(registry)),
        })
    }
    
    /// Create a new adapter as a branch of the base model
    pub async fn create_adapter(
        &self,
        base_model: &str,
        adapter_name: &str,
        lora_config: Option<serde_json::Value>,
    ) -> Result<AdapterId> {
        // Find base model path (either UUID or name)
        let base_model_path = self.resolve_base_model_path(base_model).await?;
        
        // Create branch manager for the base model
        let branch_mgr = BranchManager::new(&base_model_path)?;
        
        // Create adapter branch with human-friendly name
        let branch_info = branch_mgr.create_branch(
            "adapter",
            Some(adapter_name),
            None,  // Use current HEAD
        )?;
        
        // Generate adapter ID from branch UUID
        let adapter_id = AdapterId::from_uuid(branch_info.uuid);
        
        // Create worktree for the adapter branch
        let worktree_path = self.working_dir.join(adapter_id.0.to_string());
        let worktree_path = branch_mgr.create_worktree(
            &branch_info.branch_name,
            Some(worktree_path),
        )?;
        
        // Create adapter configuration
        let config = AdapterConfig {
            adapter_id: adapter_id.clone(),
            adapter_name: adapter_name.to_string(),
            base_model: base_model.to_string(),
            base_model_path: base_model_path.to_string_lossy().to_string(),
            adapter_type: "lora".to_string(),
            lora_r: lora_config.as_ref().and_then(|c| c.get("r").and_then(|v| v.as_u64()).map(|v| v as u32)),
            lora_alpha: lora_config.as_ref().and_then(|c| c.get("alpha").and_then(|v| v.as_f64()).map(|v| v as f32)),
            target_modules: lora_config.as_ref()
                .and_then(|c| c.get("target_modules"))
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_else(|| vec!["q_proj".to_string(), "v_proj".to_string()]),
            created_at: Utc::now(),
            training_config: lora_config,
        };
        
        // Save adapter configuration in worktree
        let config_path = worktree_path.join("adapter_config.json");
        let config_json = serde_json::to_string_pretty(&config)?;
        fs::write(&config_path, config_json).await?;
        
        
        // Commit adapter files in the worktree
        self.commit_adapter(&worktree_path, "Initialize adapter").await?;
        
        // Register with Git registry if available
        if let Some(registry) = self.registry.write().await.as_mut() {
            if let Err(e) = registry.register_adapter(
                base_model,
                adapter_name,
                branch_info.uuid,
            ).await {
                tracing::warn!("Failed to register adapter with Git registry: {}", e);
            }
        }
        
        tracing::info!("Created adapter '{}' with ID {}", adapter_name, adapter_id);
        
        Ok(adapter_id)
    }
    
    /// Load an adapter session
    pub async fn load_adapter(&self, adapter_name: &str) -> Result<AdapterSession> {
        // Try to find adapter worktree by name
        let adapter_path = self.find_adapter_worktree(adapter_name).await?;
        
        // Load adapter configuration
        let config_path = adapter_path.join("adapter_config.json");
        let config_json = fs::read_to_string(&config_path).await?;
        let config: AdapterConfig = serde_json::from_str(&config_json)?;
        
        // Resolve base model path
        let base_model_path = if config.base_model_path.starts_with("/") {
            PathBuf::from(&config.base_model_path)
        } else {
            self.base_dir.join(&config.base_model_path)
        };
        
        // Verify base model exists
        if !base_model_path.exists() {
            bail!(
                "Base model '{}' not found at {}",
                config.base_model,
                base_model_path.display()
            );
        }
        
        Ok(AdapterSession {
            adapter_id: config.adapter_id.clone(),
            adapter_path,
            base_model_path,
            config,
            is_training: false,
        })
    }
    
    /// List all adapters
    pub async fn list_adapters(&self) -> Result<Vec<(String, AdapterConfig)>> {
        let mut adapters = Vec::new();
        
        // List all worktrees in the working directory
        let mut entries = fs::read_dir(&self.working_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                let config_path = path.join("adapter_config.json");
                if config_path.exists() {
                    match fs::read_to_string(&config_path).await {
                        Ok(json) => {
                            match serde_json::from_str::<AdapterConfig>(&json) {
                                Ok(config) => {
                                    let name = path.file_name()
                                        .and_then(|n| n.to_str())
                                        .unwrap_or("unknown")
                                        .to_string();
                                    adapters.push((name, config));
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to parse adapter config at {}: {}", config_path.display(), e);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to read adapter config at {}: {}", config_path.display(), e);
                        }
                    }
                }
            }
        }
        
        Ok(adapters)
    }
    
    /// Delete an adapter branch and its worktree
    pub async fn delete_adapter(&self, adapter_name: &str) -> Result<()> {
        // Find adapter worktree
        let adapter_path = self.find_adapter_worktree(adapter_name).await?;
        
        // Load config to get base model
        let config_path = adapter_path.join("adapter_config.json");
        let config_json = fs::read_to_string(&config_path).await?;
        let config: AdapterConfig = serde_json::from_str(&config_json)?;
        
        // Find base model and delete branch
        let base_model_path = self.resolve_base_model_path(&config.base_model).await?;
        let mut branch_mgr = BranchManager::new(&base_model_path)?;
        
        // Delete branch (will also remove worktree)
        branch_mgr.delete_branch(adapter_name)?;
        
        // Remove worktree directory
        if adapter_path.exists() {
            fs::remove_dir_all(&adapter_path).await?;
        }
        
        tracing::info!("Deleted adapter '{}'", adapter_name);
        
        Ok(())
    }
    
    /// Commit changes to adapter repository
    pub async fn commit_adapter(&self, adapter_path: &Path, message: &str) -> Result<()> {
        let repo = Repository::open(adapter_path)?;
        
        // Stage all changes
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
        index.write()?;
        
        // Check if there are changes to commit
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        
        let sig = Signature::now("hyprstream", "hyprstream@local")?;
        
        // Get HEAD if it exists (for non-initial commits)
        if let Ok(head) = repo.head() {
            let parent = head.peel_to_commit()?;
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                message,
                &tree,
                &[&parent],
            )?;
        } else {
            // Initial commit
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                message,
                &tree,
                &[],
            )?;
        }
        
        Ok(())
    }
    
    /// Start training session for an adapter
    pub async fn start_training(&self, adapter_name: &str) -> Result<AdapterSession> {
        let mut session = self.load_adapter(adapter_name).await?;
        session.is_training = true;
        
        // Create sessions symlink for easy access
        let sessions_dir = self.base_dir.join("sessions");
        fs::create_dir_all(&sessions_dir).await?;
        
        let session_link = sessions_dir.join(format!("training-{}", adapter_name));
        if session_link.exists() {
            fs::remove_file(&session_link).await?;
        }
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&session.adapter_path, &session_link)?;
        }
        
        tracing::info!("Started training session for adapter '{}'", adapter_name);
        
        Ok(session)
    }
    
    /// Find adapter worktree by name
    async fn find_adapter_worktree(&self, adapter_name: &str) -> Result<PathBuf> {
        // First check if it's a UUID
        if let Ok(uuid) = Uuid::parse_str(adapter_name) {
            let path = self.working_dir.join(uuid.to_string());
            if path.exists() {
                return Ok(path);
            }
        }
        
        // Search worktrees for matching adapter name
        let mut entries = fs::read_dir(&self.working_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                let config_path = path.join("adapter_config.json");
                if config_path.exists() {
                    if let Ok(json) = fs::read_to_string(&config_path).await {
                        if let Ok(config) = serde_json::from_str::<AdapterConfig>(&json) {
                            if config.adapter_name == adapter_name {
                                return Ok(path);
                            }
                        }
                    }
                }
            }
        }
        
        bail!("Adapter '{}' not found", adapter_name)
    }
    
    /// Resolve base model name to path
    async fn resolve_base_model_path(&self, base_model: &str) -> Result<PathBuf> {
        // Check if it's an absolute path
        if base_model.starts_with("/") {
            let path = PathBuf::from(base_model);
            if path.exists() {
                return Ok(path);
            }
        }
        
        // Check if it's a UUID
        if let Ok(uuid) = Uuid::parse_str(base_model) {
            let path = self.base_dir.join(uuid.to_string());
            if path.exists() {
                return Ok(path);
            }
        }
        
        // Try to resolve via registry
        if let Some(registry) = self.registry.read().await.as_ref() {
            if let Some(model) = registry.get_model(base_model) {
                let path = self.base_dir.join(model.uuid.to_string());
                if path.exists() {
                    return Ok(path);
                }
            }
        }
        
        bail!("Base model '{}' not found", base_model)
    }
}

impl AdapterSession {
    /// Save adapter weights
    pub async fn save_weights(&self, weights_data: &[u8]) -> Result<()> {
        let weights_path = self.adapter_path.join("adapter_model.bin");
        fs::write(&weights_path, weights_data).await?;
        Ok(())
    }
    
    /// Load adapter weights
    pub async fn load_weights(&self) -> Result<Vec<u8>> {
        let weights_path = self.adapter_path.join("adapter_model.bin");
        if !weights_path.exists() {
            // Return empty weights if file doesn't exist yet
            return Ok(Vec::new());
        }
        Ok(fs::read(&weights_path).await?)
    }
    
    /// Get base model weights path
    pub fn base_model_weights_path(&self) -> PathBuf {
        self.base_model_path.join("model.safetensors")
    }
}