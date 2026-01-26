//! Model storage using ZMQ-based registry service
//!
//! This module provides model storage using a shared registry service.
//! Registry operations (list, clone, register) go through the service,
//! while repository operations (worktrees, branches) use local git access.

use anyhow::Result;
use crate::services::{PolicyZmqClient, RegistryClient, RepositoryClient, WorktreeInfo};
use git2db::{GitManager, GitRef, RepoId, TrackedRepository};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;
use hyprstream_rpc::service::ServiceManager;

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

/// Result of cloning a model repository
#[derive(Debug, Clone)]
pub struct ClonedModel {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub model_name: String,
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
    /// Whether the worktree has uncommitted changes
    pub is_dirty: bool,
}

#[derive(Debug)]
pub struct CacheStats {
    pub total_size_bytes: u64,
}

/// Model storage using a shared registry service.
///
/// Registry operations (list, clone, register) go through the service client.
/// Repository operations (worktrees, branches) use the service layer.
pub struct ModelStorage {
    base_dir: PathBuf,
    client: Arc<dyn RegistryClient>,
}

impl ModelStorage {
    /// Create with a registry client.
    ///
    /// The client is typically obtained from `LocalService::start()` or passed
    /// from a parent component that manages the shared registry.
    pub fn new(client: Arc<dyn RegistryClient>, base_dir: PathBuf) -> Self {
        Self { client, base_dir }
    }

    /// Get the registry client (for accessing repository operations)
    pub fn registry(&self) -> &Arc<dyn RegistryClient> {
        &self.client
    }

    /// Create ModelStorage with default configuration.
    ///
    /// This is a convenience method for CLI and standalone usage where there's
    /// no shared registry service. For server usage, prefer passing a shared
    /// RegistryClient via `new()`.
    pub async fn create(base_dir: PathBuf) -> Result<Self> {
        Self::create_with_config(base_dir, git2db::Git2DBConfig::default()).await
    }

    /// Create ModelStorage with configuration, starting a ZMQ registry service.
    ///
    /// This is a convenience method for CLI and standalone usage where there's
    /// no shared registry service. For server usage, prefer passing a shared
    /// RegistryClient via `new()`.
    ///
    /// Note: This generates a transient keypair for the service. For production
    /// usage, the signing key should be loaded from configuration.
    pub async fn create_with_config(
        base_dir: PathBuf,
        _config: git2db::Git2DBConfig,
    ) -> Result<Self> {
        use crate::services::{RegistryService, RegistryZmqClient};
        use hyprstream_rpc::{generate_signing_keypair, RequestIdentity};

        // Generate keypair for this service instance
        // TODO: Load from configuration for production
        let (signing_key, _verifying_key) = generate_signing_keypair();

        // Create policy client that connects to the already-running PolicyService
        // (PolicyService should be started by main.rs before any ModelStorage is created)
        // If PolicyService isn't running, policy checks will fail gracefully
        let policy_client = PolicyZmqClient::new(signing_key.clone(), RequestIdentity::local());

        // Start ZMQ-based registry service using unified pattern (service includes infrastructure)
        let registry_transport = hyprstream_rpc::transport::TransportConfig::inproc("hyprstream/registry");
        let registry_service = RegistryService::new(
            &base_dir,
            policy_client.clone(),
            crate::zmq::global_context().clone(),
            registry_transport,
            signing_key.clone(),
        ).await
            .map_err(|e| anyhow::anyhow!("Failed to create registry service: {}", e))?;
        let manager = hyprstream_rpc::service::InprocManager::new();
        let _service_handle = manager.spawn(Box::new(registry_service))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start registry service: {}", e))?;

        // Create ZMQ client with signing credentials
        // Use local identity (current OS user) for internal model storage operations
        let client: Arc<dyn RegistryClient> = Arc::new(RegistryZmqClient::new(
            signing_key,
            RequestIdentity::local(),
        ));

        Ok(Self {
            client,
            base_dir,
        })
    }

    /// Get models directory
    pub fn get_models_dir(&self) -> PathBuf {
        self.base_dir.clone()
    }

    /// Resolve model name to TrackedRepository
    async fn resolve_repo(&self, name: &str) -> Result<TrackedRepository> {
        // Use cached_list for fast reads, fall back to list() if cache unavailable
        let repos = self
            .client
            .cached_list()
            .unwrap_or(self.client.list().await.map_err(|e| anyhow::anyhow!("{}", e))?);

        repos
            .into_iter()
            .find(|t| t.name.as_ref() == Some(&name.to_string()))
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in registry", name))
    }

    /// Resolve model name to RepoId
    async fn resolve_name(&self, name: &str) -> Result<RepoId> {
        Ok(self.resolve_repo(name).await?.id)
    }

    /// Get bare repository path by model reference
    pub async fn get_bare_repo_path(&self, model_ref: &ModelRef) -> Result<PathBuf> {
        let repo = self.resolve_repo(&model_ref.model).await?;
        // TrackedRepository stores the worktree path
        Ok(PathBuf::from(&repo.worktree_path))
    }

    /// Get worktree path for a specific branch
    pub async fn get_worktree_path(&self, model_ref: &ModelRef, branch: &str) -> Result<PathBuf> {
        let repo = self.resolve_repo(&model_ref.model).await?;

        // Get bare repo path from tracked repository
        let bare_repo_path = PathBuf::from(&repo.worktree_path);

        // Navigate from bare repo to worktrees directory
        // From models/{name}/{name}.git/ to models/{name}/worktrees/{branch}
        let repo_dir = bare_repo_path
            .parent()
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

        // Use cached_list for fast reads, fall back to list() if cache unavailable
        let repos = self
            .client
            .cached_list()
            .unwrap_or(self.client.list().await.map_err(|e| anyhow::anyhow!("{}", e))?);

        for tracked in repos {
            if let Some(name) = &tracked.name {
                let base_ref = ModelRef::new(name.clone());

                // Enumerate all worktrees for this model
                match self.list_worktrees(&base_ref).await {
                    Ok(worktrees) => {
                        for wt in worktrees {
                            // Skip worktrees without branch names
                            let branch_name = match wt.branch {
                                Some(ref b) => b.clone(),
                                None => continue,
                            };

                            // Create model:branch reference
                            let model_ref = ModelRef::with_ref(
                                name.clone(),
                                git2db::GitRef::Branch(branch_name.clone())
                            );

                            // Build metadata (size calculation removed for performance)
                            let metadata = ModelMetadata {
                                name: name.clone(),
                                display_name: Some(format!("{}:{}", name, branch_name)),
                                model_type: "worktree".to_string(),
                                created_at: chrono::Utc::now().timestamp(),
                                updated_at: chrono::Utc::now().timestamp(),
                                size_bytes: None,  // Removed expensive directory size calculation
                                tags: vec![],
                                is_dirty: wt.is_dirty,
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

    /// Check if a model exists
    pub async fn model_exists(&self, model_name: &str) -> bool {
        // Use cached_list for fast reads
        let repos = self
            .client
            .cached_list()
            .unwrap_or_else(|| {
                // Fallback to blocking call - this shouldn't happen often
                vec![]
            });

        repos
            .iter()
            .any(|t| t.name.as_ref() == Some(&model_name.to_string()))
    }

    /// Add a new model
    pub async fn add_model(&self, name: &str, source: &str) -> Result<()> {
        validate_model_name(name)?;

        // Clone via registry service
        let _repo_id = self
            .client
            .clone_repo(source, Some(name))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to clone model: {}", e))?;

        Ok(())
    }

    /// Update a model to a different version
    pub async fn update_model(&self, name: &str, ref_spec: &str) -> Result<()> {
        let tracked = self.resolve_repo(name).await?;
        let repo_path = PathBuf::from(&tracked.worktree_path);

        // Use GitManager for local git operations
        let repo_cache = GitManager::global().get_repository(&repo_path)?;
        let repo = repo_cache.open()?;

        // Fetch from remote
        let mut remote = repo.find_remote("origin")?;
        remote.fetch(&["refs/heads/*:refs/remotes/origin/*"], None, None)?;

        // Checkout the ref spec
        let (object, reference) = repo.revparse_ext(ref_spec)?;
        repo.checkout_tree(&object, None)?;

        if let Some(ref_name) = reference {
            repo.set_head(ref_name.name().unwrap_or(ref_spec))?;
        } else {
            repo.set_head_detached(object.id())?;
        }

        Ok(())
    }

    /// Open repository for direct git operations
    ///
    /// This provides access to the underlying git2::Repository for advanced
    /// operations not covered by the high-level ModelStorage API.
    ///
    /// # Example
    /// ```ignore
    /// let repo = storage.open_repo(&model_ref).await?;
    /// let branches = repo.branches(None)?;
    /// ```
    pub async fn open_repo(&self, model_ref: &ModelRef) -> Result<git2::Repository> {
        let tracked = self.resolve_repo(&model_ref.model).await?;
        let repo_path = PathBuf::from(&tracked.worktree_path);

        let repo_cache = GitManager::global().get_repository(&repo_path)?;
        Ok(repo_cache.open()?)
    }

    /// Create a branch for a model
    pub async fn create_branch(
        &self,
        model_ref: &ModelRef,
        branch_name: &str,
        from_ref: Option<&str>,
    ) -> Result<()> {
        let repo = self.open_repo(model_ref).await?;

        // Find the commit to branch from
        let commit = if let Some(ref_name) = from_ref {
            let (obj, _) = repo.revparse_ext(ref_name)?;
            repo.find_commit(obj.id())?
        } else {
            repo.head()?.peel_to_commit()?
        };

        // Create the branch
        repo.branch(branch_name, &commit, false)?;
        Ok(())
    }

    /// Resolve model name to RepoId for use with client operations
    pub async fn resolve_repo_id(&self, model_ref: &ModelRef) -> Result<RepoId> {
        self.resolve_name(&model_ref.model).await
    }

    /// Resolve a git reference to a commit OID
    pub async fn resolve_git_ref(&self, model_ref: &ModelRef) -> Result<git2db::Oid> {
        let repo = self.open_repo(model_ref).await?;

        let ref_spec = match &model_ref.git_ref {
            GitRef::Branch(name) => name.clone(),
            GitRef::Tag(name) => format!("refs/tags/{}", name),
            GitRef::Commit(oid) => return Ok(*oid),
            GitRef::DefaultBranch => "HEAD".to_string(),
            GitRef::Revspec(spec) => spec.clone(),
        };

        let (obj, _) = repo.revparse_ext(&ref_spec)?;
        Ok(obj.id())
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
    ///
    /// Status is obtained from the worktree for the specified branch, not the bare repo.
    pub async fn status(&self, model_ref: &ModelRef) -> Result<git2db::RepositoryStatus> {
        // Get the worktree path for this model:branch reference
        let worktree_path = self.get_model_path(model_ref).await?;

        // Check if worktree exists
        if !worktree_path.exists() {
            return Err(anyhow::anyhow!(
                "Worktree does not exist at {}. Create it with: hyprstream branch {} {}",
                worktree_path.display(),
                model_ref.model,
                model_ref.git_ref.display_name()
            ));
        }

        // Use GitManager to open the worktree repository
        let repo_cache = GitManager::global().get_repository(&worktree_path)?;
        let repo = repo_cache.open()?;

        // Get status from repository
        let statuses = repo.statuses(None)?;

        // Build RepositoryStatus
        let is_clean = statuses.is_empty();
        let branch = repo.head().ok().and_then(|h| h.shorthand().map(String::from));
        let head = repo.head().ok().and_then(|h| h.target());

        // Collect modified file paths
        let modified_files: Vec<PathBuf> = statuses
            .iter()
            .filter_map(|entry| entry.path().map(|p| PathBuf::from(p)))
            .collect();

        Ok(git2db::RepositoryStatus {
            is_clean,
            branch,
            head,
            ahead: 0,
            behind: 0,
            modified_files,
        })
    }

    /// Checkout a git reference
    pub async fn checkout(
        &self,
        model_ref: &ModelRef,
        _options: super::CheckoutOptions,
    ) -> Result<super::CheckoutResult> {
        let tracked = self.resolve_repo(&model_ref.model).await?;
        let repo_path = PathBuf::from(&tracked.worktree_path);

        // Use GitManager for local git operations
        let repo_cache = GitManager::global().get_repository(&repo_path)?;
        let repo = repo_cache.open()?;

        let previous_oid = repo.head().ok().and_then(|h| h.target()).unwrap_or(git2db::Oid::zero());

        // Convert GitRef to string for revparse
        let ref_spec = match &model_ref.git_ref {
            GitRef::Branch(name) => name.clone(),
            GitRef::Tag(name) => format!("refs/tags/{}", name),
            GitRef::Commit(oid) => oid.to_string(),
            GitRef::DefaultBranch => "HEAD".to_string(),
            GitRef::Revspec(spec) => spec.clone(),
        };

        // Checkout the ref spec
        let (object, reference) = repo.revparse_ext(&ref_spec)?;
        repo.checkout_tree(&object, None)?;

        if let Some(ref_name) = reference {
            repo.set_head(ref_name.name().unwrap_or(&ref_spec))?;
        } else {
            repo.set_head_detached(object.id())?;
        }

        let new_oid = repo.head().ok().and_then(|h| h.target()).unwrap_or(git2db::Oid::zero());

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
        // Use RepositoryClient to get default branch via service layer
        let repo_client = self.client.repo(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

        repo_client.default_branch().await
            .map_err(|e| anyhow::anyhow!("Failed to get default branch: {}", e))
    }

    /// Remove model
    pub async fn remove_model(&self, model_ref: &ModelRef) -> Result<()> {
        let repo_id = self.resolve_name(&model_ref.model).await?;

        // Remove via registry service
        self.client
            .remove(&repo_id)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to remove model: {}", e))?;

        Ok(())
    }

    /// Create a new worktree for a model
    ///
    /// Uses RepositoryClient to route through the service layer, which properly
    /// handles LFS file smudging via RepositoryHandle.
    pub async fn create_worktree(&self, model_ref: &ModelRef, branch: &str) -> Result<PathBuf> {
        let worktree_path = self.get_worktree_path(model_ref, branch).await?;

        if worktree_path.exists() {
            return Err(anyhow::anyhow!("Worktree already exists at {:?}", worktree_path));
        }

        tracing::info!(
            "Creating worktree for {} at {} (branch: {})",
            model_ref.model,
            worktree_path.display(),
            branch
        );

        // Use RepositoryClient to create worktree - routes through service layer
        // which uses RepositoryHandle.create_worktree() with proper LFS handling
        let repo_client = self.client.repo(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

        let result_path = repo_client.create_worktree(&worktree_path, branch).await
            .map_err(|e| anyhow::anyhow!("Failed to create worktree: {}", e))?;

        tracing::info!("Successfully created worktree at {}", result_path.display());
        Ok(result_path)
    }

    /// Get a repository client for a model
    ///
    /// Returns a client that provides repository-level operations (worktrees, branches, etc.)
    /// through the service layer.
    pub async fn get_repo_client(
        &self,
        model_ref: &ModelRef,
    ) -> Result<Arc<dyn RepositoryClient>> {
        self.client
            .repo(&model_ref.model)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client for '{}': {}", model_ref.model, e))
    }

    /// List all worktrees for a model
    ///
    /// Returns the full WorktreeInfo for each worktree, including dirty status.
    /// If no worktrees exist, returns an empty list.
    pub async fn list_worktrees(&self, model_ref: &ModelRef) -> Result<Vec<WorktreeInfo>> {
        // Use RepositoryClient to get worktrees via service layer
        let repo_client = self.client.repo(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

        repo_client.list_worktrees().await
            .map_err(|e| anyhow::anyhow!("Failed to list worktrees: {}", e))
    }


    /// Remove a worktree for a model
    pub async fn remove_worktree(&self, model_ref: &ModelRef, branch: &str) -> Result<()> {
        // Get the worktree path for this branch
        let worktree_path = self.get_worktree_path(model_ref, branch).await?;

        // Use RepositoryClient to remove worktree via service layer
        // This routes through the service which handles storage driver cleanup properly
        let repo_client = self.client.repo(&model_ref.model).await
            .map_err(|e| anyhow::anyhow!("Failed to get repository client: {}", e))?;

        repo_client.remove_worktree(&worktree_path).await
            .map_err(|e| anyhow::anyhow!("Failed to remove worktree: {}", e))?;

        Ok(())
    }
}
