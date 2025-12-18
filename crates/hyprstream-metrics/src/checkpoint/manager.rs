//! Checkpoint manager for git2db-based versioned storage.

use crate::checkpoint::state::{Checkpoint, CheckpointMetadata, TableCheckpoint};
use crate::storage::duckdb::DuckDbBackend;
use crate::storage::StorageBackend;
use chrono::Utc;
use git2db::service::RegistryClient;
use git2db::{GitManager, RepoId};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tonic::Status;
use tracing::{debug, error, info, instrument};

/// Configuration for checkpoint behavior
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Interval between automatic checkpoints
    pub interval: Duration,
    /// Number of checkpoints to retain (0 = unlimited)
    pub retention_count: usize,
    /// Whether to create a checkpoint on shutdown
    pub checkpoint_on_shutdown: bool,
    /// Base directory for checkpoint storage
    pub base_dir: PathBuf,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5 * 60), // 5 minutes
            retention_count: 10,
            checkpoint_on_shutdown: true,
            base_dir: PathBuf::from(".checkpoints"),
        }
    }
}

impl CheckpointConfig {
    /// Create a new config with specified interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Set retention count
    pub fn with_retention(mut self, count: usize) -> Self {
        self.retention_count = count;
        self
    }

    /// Set base directory
    pub fn with_base_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.base_dir = dir.into();
        self
    }
}

/// Manages checkpoints for metrics storage
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Shared registry client (if using shared registry)
    client: Option<Arc<dyn RegistryClient>>,
    /// Repository name in registry (if using shared registry)
    repo_name: Option<String>,
    /// Path to the checkpoint repository
    repo_path: PathBuf,
    /// List of recent checkpoints (cached)
    checkpoints: RwLock<Vec<Checkpoint>>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager (standalone mode)
    ///
    /// This creates a local git repository that is not tracked in any registry.
    /// For shared registry integration, use `new_with_client()` instead.
    #[instrument(skip(config), fields(base_dir = ?config.base_dir))]
    pub async fn new(config: CheckpointConfig) -> Result<Self, Status> {
        Self::create_internal(config, None, None).await
    }

    /// Create a new checkpoint manager with shared registry client
    ///
    /// The checkpoint repository will be registered in the shared registry,
    /// allowing it to be discovered and managed alongside other repositories.
    ///
    /// # Arguments
    /// - `client` - Shared registry client
    /// - `repo_name` - Name for the checkpoint repository (e.g., "metrics-checkpoints")
    /// - `config` - Checkpoint configuration
    #[instrument(skip(client, config), fields(repo_name = %repo_name, base_dir = ?config.base_dir))]
    pub async fn new_with_client(
        client: Arc<dyn RegistryClient>,
        repo_name: &str,
        config: CheckpointConfig,
    ) -> Result<Self, Status> {
        info!("Initializing CheckpointManager with shared registry");

        // Check if repo already exists in registry
        let existing = client
            .get_by_name(repo_name)
            .await
            .map_err(|e| Status::internal(format!("Failed to query registry: {}", e)))?;

        if let Some(tracked) = existing {
            // Repository already registered - use its path
            let repo_path = PathBuf::from(&tracked.worktree_path);
            info!("Using existing registered checkpoint repository at {:?}", repo_path);

            return Ok(Self {
                config,
                client: Some(client),
                repo_name: Some(repo_name.to_string()),
                repo_path,
                checkpoints: RwLock::new(Vec::new()),
            });
        }

        // Create the repository and register it
        let manager = Self::create_internal(
            config,
            Some(client.clone()),
            Some(repo_name.to_string()),
        )
        .await?;

        // Register with the registry
        let repo_id = RepoId::new();
        client
            .register(&repo_id, Some(repo_name), &manager.repo_path)
            .await
            .map_err(|e| Status::internal(format!("Failed to register checkpoint repo: {}", e)))?;

        info!("Registered checkpoint repository '{}' in registry", repo_name);

        Ok(manager)
    }

    /// Internal helper to create the checkpoint manager
    async fn create_internal(
        config: CheckpointConfig,
        client: Option<Arc<dyn RegistryClient>>,
        repo_name: Option<String>,
    ) -> Result<Self, Status> {
        info!("Initializing CheckpointManager");

        // Ensure base directory exists
        std::fs::create_dir_all(&config.base_dir).map_err(|e| {
            Status::internal(format!(
                "Failed to create checkpoint directory: {}",
                e
            ))
        })?;

        // Create or get the checkpoint repository
        let repo_path = config.base_dir.join("repo");

        // Initialize git repository if it doesn't exist
        if !repo_path.join(".git").exists() {
            std::fs::create_dir_all(&repo_path).map_err(|e| {
                Status::internal(format!("Failed to create repository directory: {}", e))
            })?;

            git2::Repository::init(&repo_path).map_err(|e| {
                Status::internal(format!("Failed to initialize git repository: {}", e))
            })?;

            info!("Created new checkpoint repository at {:?}", repo_path);
        } else {
            info!("Using existing checkpoint repository at {:?}", repo_path);
        }

        Ok(Self {
            config,
            client,
            repo_name,
            repo_path,
            checkpoints: RwLock::new(Vec::new()),
        })
    }

    /// Create a checkpoint from the current storage state
    #[instrument(skip(self, storage), fields(checkpoint_name = ?name))]
    pub async fn create_checkpoint(
        &self,
        storage: &DuckDbBackend,
        name: Option<String>,
    ) -> Result<Checkpoint, Status> {
        info!("Creating checkpoint");

        // Get list of tables
        let tables = storage.list_tables().await?;
        debug!(table_count = tables.len(), "Found tables to checkpoint");

        // Ensure tables directory exists
        let tables_dir = self.repo_path.join("tables");
        std::fs::create_dir_all(&tables_dir).map_err(|e| {
            Status::internal(format!("Failed to create tables directory: {}", e))
        })?;

        // Export each table to Parquet
        let mut table_checkpoints = Vec::new();
        for table_name in &tables {
            // Skip internal tables
            if table_name.starts_with("sqlite_") || table_name == "view_metadata" {
                continue;
            }

            let parquet_path = tables_dir.join(format!("{}.parquet", table_name));
            debug!(table = %table_name, path = ?parquet_path, "Exporting table");

            storage
                .export_to_parquet(table_name, &parquet_path)
                .await?;

            // Get file size
            let file_size = std::fs::metadata(&parquet_path)
                .map(|m| m.len())
                .unwrap_or(0);

            // Get row count (estimate from file size for now)
            // TODO: Query actual row count from DuckDB
            let row_count = 0u64; // Placeholder

            table_checkpoints.push(TableCheckpoint::new(
                table_name.clone(),
                row_count,
                file_size,
                format!("tables/{}.parquet", table_name),
            ));
        }

        // Create checkpoint metadata
        let mut metadata = CheckpointMetadata::new(table_checkpoints);
        if let Some(n) = name.clone() {
            metadata = metadata.with_name(n);
        }
        metadata = metadata.with_metadata("created_by", "checkpoint_manager");

        // Save metadata file
        let metadata_path = self.repo_path.join("checkpoint.json");
        let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            Status::internal(format!("Failed to serialize checkpoint metadata: {}", e))
        })?;
        std::fs::write(&metadata_path, metadata_json).map_err(|e| {
            Status::internal(format!("Failed to write checkpoint metadata: {}", e))
        })?;

        // Commit the checkpoint
        let commit_id = self.commit_checkpoint(&metadata, name.as_deref()).await?;

        // Update metadata with commit ID
        let mut final_metadata = metadata;
        final_metadata.id = commit_id.clone();

        let checkpoint = Checkpoint::new(commit_id, final_metadata);

        // Update cache
        {
            let mut checkpoints = self.checkpoints.write().await;
            checkpoints.push(checkpoint.clone());

            // Enforce retention policy
            if self.config.retention_count > 0 && checkpoints.len() > self.config.retention_count {
                checkpoints.remove(0);
            }
        }

        info!(checkpoint_id = %checkpoint.id(), "Checkpoint created successfully");
        Ok(checkpoint)
    }

    /// Commit the checkpoint to git
    async fn commit_checkpoint(
        &self,
        metadata: &CheckpointMetadata,
        name: Option<&str>,
    ) -> Result<String, Status> {
        // Use GitManager for repository access
        let repo_cache = GitManager::global()
            .get_repository(&self.repo_path)
            .map_err(|e| Status::internal(format!("Failed to get repository: {}", e)))?;

        let repo = repo_cache
            .open()
            .map_err(|e| Status::internal(format!("Failed to open repository: {}", e)))?;

        // Stage all files using git2 directly
        let mut index = repo.index().map_err(|e| {
            Status::internal(format!("Failed to get index: {}", e))
        })?;
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None).map_err(|e| {
            Status::internal(format!("Failed to stage files: {}", e))
        })?;
        index.write().map_err(|e| {
            Status::internal(format!("Failed to write index: {}", e))
        })?;

        let sig = GitManager::global()
            .create_signature(None, None)
            .map_err(|e| Status::internal(format!("Failed to create signature: {}", e)))?;

        let message = format!(
            "Checkpoint: {}\n\nTables: {}\nTotal size: {} bytes",
            name.unwrap_or(&metadata.created_at.to_rfc3339()),
            metadata.tables.len(),
            metadata.total_size
        );

        // Get tree from index
        let mut index = repo.index().map_err(|e| {
            Status::internal(format!("Failed to get index: {}", e))
        })?;
        let tree_id = index.write_tree().map_err(|e| {
            Status::internal(format!("Failed to write tree: {}", e))
        })?;
        let tree = repo.find_tree(tree_id).map_err(|e| {
            Status::internal(format!("Failed to find tree: {}", e))
        })?;

        // Get parent commit if exists
        let parent = repo.head().ok().and_then(|h| h.peel_to_commit().ok());

        // Create commit
        let commit_id = if let Some(parent) = parent {
            repo.commit(Some("HEAD"), &sig, &sig, &message, &tree, &[&parent])
        } else {
            repo.commit(Some("HEAD"), &sig, &sig, &message, &tree, &[])
        }
        .map_err(|e| Status::internal(format!("Failed to create commit: {}", e)))?;

        // Create tag if name provided
        if let Some(tag_name) = name {
            if let Ok(obj) = repo.find_object(commit_id, None) {
                let _ = repo.tag_lightweight(tag_name, &obj, true);
            }
        }

        Ok(commit_id.to_string())
    }

    /// Restore from a checkpoint
    #[instrument(skip(self, storage), fields(checkpoint_id = %checkpoint_id))]
    pub async fn restore(
        &self,
        checkpoint_id: &str,
        storage: &DuckDbBackend,
    ) -> Result<(), Status> {
        info!("Restoring from checkpoint");

        // Use GitManager for repository access
        let repo_cache = GitManager::global()
            .get_repository(&self.repo_path)
            .map_err(|e| Status::internal(format!("Failed to get repository: {}", e)))?;

        let repo = repo_cache
            .open()
            .map_err(|e| Status::internal(format!("Failed to open repository: {}", e)))?;

        let oid = git2::Oid::from_str(checkpoint_id).map_err(|e| {
            Status::invalid_argument(format!("Invalid checkpoint ID: {}", e))
        })?;

        let commit = repo.find_commit(oid).map_err(|e| {
            Status::not_found(format!("Checkpoint not found: {}", e))
        })?;

        // Reset to the checkpoint
        repo.reset(commit.as_object(), git2::ResetType::Hard, None)
            .map_err(|e| Status::internal(format!("Failed to reset to checkpoint: {}", e)))?;

        // Read checkpoint metadata
        let metadata_path = self.repo_path.join("checkpoint.json");
        let metadata_json = std::fs::read_to_string(&metadata_path).map_err(|e| {
            Status::internal(format!("Failed to read checkpoint metadata: {}", e))
        })?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json).map_err(|e| {
            Status::internal(format!("Failed to parse checkpoint metadata: {}", e))
        })?;

        // Import each table from Parquet
        for table in &metadata.tables {
            let parquet_path = self.repo_path.join(&table.file_path);
            debug!(table = %table.name, path = ?parquet_path, "Importing table");

            // Drop existing table if it exists
            let _ = storage.drop_table(&table.name).await;

            // Import from Parquet
            storage
                .import_from_parquet(&table.name, &parquet_path)
                .await?;
        }

        info!("Checkpoint restored successfully");
        Ok(())
    }

    /// Get the latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> Option<Checkpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.last().cloned()
    }

    /// List all checkpoints
    pub async fn list_checkpoints(&self) -> Vec<Checkpoint> {
        self.checkpoints.read().await.clone()
    }

    /// Start background checkpoint task
    pub fn start_background(
        self: Arc<Self>,
        storage: Arc<DuckDbBackend>,
    ) -> JoinHandle<()> {
        let interval = self.config.interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            loop {
                interval_timer.tick().await;

                let checkpoint_name = format!("auto-{}", Utc::now().format("%Y%m%d-%H%M%S"));
                match self.create_checkpoint(&storage, Some(checkpoint_name)).await {
                    Ok(checkpoint) => {
                        info!(
                            checkpoint_id = %checkpoint.id(),
                            "Background checkpoint created"
                        );
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to create background checkpoint");
                    }
                }
            }
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Get the registry client if using shared registry
    pub fn registry_client(&self) -> Option<&Arc<dyn RegistryClient>> {
        self.client.as_ref()
    }

    /// Get the repository name in registry if using shared registry
    pub fn repo_name(&self) -> Option<&str> {
        self.repo_name.as_deref()
    }

    /// Get the repository path
    pub fn repo_path(&self) -> &PathBuf {
        &self.repo_path
    }
}

impl std::fmt::Debug for CheckpointManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointManager")
            .field("config", &self.config)
            .field("repo_path", &self.repo_path)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config_defaults() {
        let config = CheckpointConfig::default();
        assert_eq!(config.interval, Duration::from_secs(300));
        assert_eq!(config.retention_count, 10);
        assert!(config.checkpoint_on_shutdown);
    }

    #[test]
    fn test_checkpoint_config_builder() {
        let config = CheckpointConfig::default()
            .with_interval(Duration::from_secs(60))
            .with_retention(5)
            .with_base_dir("/tmp/checkpoints");

        assert_eq!(config.interval, Duration::from_secs(60));
        assert_eq!(config.retention_count, 5);
        assert_eq!(config.base_dir, PathBuf::from("/tmp/checkpoints"));
    }
}
