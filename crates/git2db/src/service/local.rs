//! In-process registry service using channels.
//!
//! This module provides a local service implementation that runs as a tokio task
//! and communicates with clients via unbounded mpsc channels.

use arc_swap::ArcSwap;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, instrument};

use crate::{Git2DB, RepoId, TrackedRepository};

use super::client::{RegistryClient, RepositoryClient, ServiceError, WorktreeInfo};
use super::request::RegistryRequest;

/// In-process registry service that runs as a tokio task.
///
/// The service owns a Git2DB registry and processes requests from clients
/// via an unbounded mpsc channel. It also maintains a cached snapshot of
/// repositories for fast reads.
pub struct LocalService {
    registry: Git2DB,
    requests: mpsc::UnboundedReceiver<RegistryRequest>,
    cache: Arc<ArcSwap<Vec<TrackedRepository>>>,
}

impl LocalService {
    /// Start the service and return a client handle.
    ///
    /// The service runs as a background tokio task. When all client handles
    /// are dropped, the service will shut down.
    ///
    /// # Arguments
    /// * `base_dir` - Base directory for the Git2DB registry
    ///
    /// # Example
    /// ```rust,ignore
    /// let client = LocalService::start("/path/to/registry").await?;
    /// let repos = client.list().await?;
    /// ```
    #[instrument(skip_all, fields(base_dir = %base_dir.as_ref().display()))]
    pub async fn start(base_dir: impl AsRef<Path>) -> Result<LocalClient, ServiceError> {
        let base_dir = base_dir.as_ref();
        info!("Starting registry service");

        let registry = Git2DB::open(base_dir).await?;

        // Initial cache snapshot (clone to get owned data)
        let repos: Vec<TrackedRepository> = registry.list().cloned().collect();
        debug!(repo_count = repos.len(), "Initial cache populated");
        let cache = Arc::new(ArcSwap::from_pointee(repos));

        let (tx, rx) = mpsc::unbounded_channel();

        let service = Self {
            registry,
            requests: rx,
            cache: cache.clone(),
        };

        // Spawn service on a dedicated thread with its own runtime
        // This is necessary because git2 types contain raw pointers that are not Send
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create service runtime");
            rt.block_on(service.run());
        });

        info!("Registry service started");
        Ok(LocalClient { sender: tx, cache })
    }

    /// Main service loop - processes requests until channel closes.
    async fn run(mut self) {
        debug!("Service loop started");
        while let Some(request) = self.requests.recv().await {
            debug!(?request, "Processing request");
            self.handle_request(request).await;
        }
        info!("Service loop ended (all clients dropped)");
    }

    /// Handle a single request.
    async fn handle_request(&mut self, request: RegistryRequest) {
        match request {
            RegistryRequest::List { reply } => {
                let repos: Vec<TrackedRepository> = self.registry.list().cloned().collect();
                let _ = reply.send(Ok(repos));
            }

            RegistryRequest::Get { id, reply } => {
                let repo = self.registry.list().find(|r| r.id == id).cloned();
                let _ = reply.send(Ok(repo));
            }

            RegistryRequest::GetByName { name, reply } => {
                let repo = self
                    .registry
                    .list()
                    .find(|r| r.name.as_ref() == Some(&name))
                    .cloned();
                let _ = reply.send(Ok(repo));
            }

            RegistryRequest::Clone { url, name, reply } => {
                let result = self.do_clone(&url, name.as_deref()).await;

                // Update cache after mutation
                if result.is_ok() {
                    self.refresh_cache();
                }
                let _ = reply.send(result);
            }

            RegistryRequest::Register {
                id,
                name,
                path,
                reply,
            } => {
                let result = self.do_register(id, name, path).await;

                // Update cache after mutation
                if result.is_ok() {
                    self.refresh_cache();
                }
                let _ = reply.send(result);
            }

            RegistryRequest::Upsert { name, url, reply } => {
                let result = self
                    .registry
                    .upsert_repository(&name, &url)
                    .await
                    .map_err(ServiceError::from);

                // Update cache after mutation
                if result.is_ok() {
                    self.refresh_cache();
                }
                let _ = reply.send(result);
            }

            RegistryRequest::Remove { id, reply } => {
                let result = self
                    .registry
                    .remove_repository(&id)
                    .await
                    .map_err(ServiceError::from);

                // Update cache after mutation
                if result.is_ok() {
                    self.refresh_cache();
                }
                let _ = reply.send(result);
            }

            RegistryRequest::HealthCheck { reply } => {
                // Service is healthy if we can process requests
                let _ = reply.send(Ok(()));
            }

            // === Repository Operations ===

            RegistryRequest::CreateWorktree {
                repo_id,
                worktree_path,
                branch,
                reply,
            } => {
                let result = self.do_create_worktree(&repo_id, &worktree_path, &branch).await;
                let _ = reply.send(result);
            }

            RegistryRequest::ListWorktrees { repo_id, reply } => {
                let result = self.do_list_worktrees(&repo_id).await;
                let _ = reply.send(result);
            }

            RegistryRequest::WorktreePath {
                repo_id,
                branch,
                reply,
            } => {
                let result = self.do_worktree_path(&repo_id, &branch).await;
                let _ = reply.send(result);
            }

            RegistryRequest::CreateBranch {
                repo_id,
                name,
                from,
                reply,
            } => {
                let result = self.do_create_branch(&repo_id, &name, from.as_deref()).await;
                let _ = reply.send(result);
            }

            RegistryRequest::Checkout {
                repo_id,
                ref_spec,
                reply,
            } => {
                let result = self.do_checkout(&repo_id, &ref_spec).await;
                let _ = reply.send(result);
            }

            RegistryRequest::DefaultBranch { repo_id, reply } => {
                let result = self.do_default_branch(&repo_id).await;
                let _ = reply.send(result);
            }

            RegistryRequest::ListBranches { repo_id, reply } => {
                let result = self.do_list_branches(&repo_id).await;
                let _ = reply.send(result);
            }
        }
    }

    /// Clone a repository.
    async fn do_clone(&mut self, url: &str, name: Option<&str>) -> Result<RepoId, ServiceError> {
        let mut builder = self.registry.clone(url);
        if let Some(n) = name {
            builder = builder.name(n);
        }
        builder.exec().await.map_err(ServiceError::from)
    }

    /// Register an existing repository.
    async fn do_register(
        &mut self,
        id: RepoId,
        name: Option<String>,
        path: std::path::PathBuf,
    ) -> Result<(), ServiceError> {
        let mut builder = self.registry.register(id).worktree_path(&path);
        if let Some(n) = name {
            builder = builder.name(n);
        }
        builder.exec().await.map_err(ServiceError::from)
    }

    /// Refresh the cached repository list.
    fn refresh_cache(&self) {
        let repos: Vec<TrackedRepository> = self.registry.list().cloned().collect();
        debug!(repo_count = repos.len(), "Cache refreshed");
        self.cache.store(Arc::new(repos));
    }

    // === Repository Operation Implementations ===

    /// Create a worktree - uses RepositoryHandle which handles LFS properly.
    async fn do_create_worktree(
        &self,
        repo_id: &RepoId,
        worktree_path: &Path,
        branch: &str,
    ) -> Result<PathBuf, ServiceError> {
        let handle = self.registry.repo(repo_id)?;
        let worktree_handle = handle.create_worktree(worktree_path, branch).await?;
        Ok(worktree_handle.path().to_path_buf())
    }

    /// List worktrees.
    async fn do_list_worktrees(&self, repo_id: &RepoId) -> Result<Vec<WorktreeInfo>, ServiceError> {
        let handle = self.registry.repo(repo_id)?;
        let worktree_handles = handle.get_worktrees().await?;

        // Convert WorktreeHandle to WorktreeInfo
        let mut result = Vec::new();
        for wt in worktree_handles {
            // Extract branch name from path (last component: worktrees/{branch})
            let branch = wt.path()
                .file_name()
                .map(|s| s.to_string_lossy().into_owned());

            result.push(WorktreeInfo {
                path: wt.path().to_path_buf(),
                branch,
                driver: wt.driver_name.clone(),
            });
        }
        Ok(result)
    }

    /// Get worktree path for a branch.
    async fn do_worktree_path(
        &self,
        repo_id: &RepoId,
        branch: &str,
    ) -> Result<Option<PathBuf>, ServiceError> {
        let handle = self.registry.repo(repo_id)?;

        // Use get_worktree which looks up by branch name
        if let Some(wt) = handle.get_worktree(branch).await? {
            return Ok(Some(wt.path().to_path_buf()));
        }
        Ok(None)
    }

    /// Create a branch.
    async fn do_create_branch(
        &self,
        repo_id: &RepoId,
        name: &str,
        from: Option<&str>,
    ) -> Result<(), ServiceError> {
        let handle = self.registry.repo(repo_id)?;
        handle.branch().create(name, from).await?;
        Ok(())
    }

    /// Checkout a branch or ref.
    async fn do_checkout(&self, repo_id: &RepoId, ref_spec: &str) -> Result<(), ServiceError> {
        let handle = self.registry.repo(repo_id)?;
        handle.branch().checkout(ref_spec).await?;
        Ok(())
    }

    /// Get the default branch.
    async fn do_default_branch(&self, repo_id: &RepoId) -> Result<String, ServiceError> {
        let handle = self.registry.repo(repo_id)?;
        // default_branch is synchronous
        let default = handle.default_branch()?;
        Ok(default)
    }

    /// List all branches.
    async fn do_list_branches(&self, repo_id: &RepoId) -> Result<Vec<String>, ServiceError> {
        let handle = self.registry.repo(repo_id)?;
        let branches = handle.branch().list().await?;
        // Convert Branch structs to their names
        Ok(branches.into_iter().map(|b| b.name).collect())
    }
}

/// Client handle for the in-process registry service.
///
/// This is a lightweight, cloneable handle that can be shared across
/// threads and async tasks. All operations are sent to the background
/// service via an unbounded channel.
#[derive(Clone)]
pub struct LocalClient {
    sender: mpsc::UnboundedSender<RegistryRequest>,
    cache: Arc<ArcSwap<Vec<TrackedRepository>>>,
}

#[async_trait]
impl RegistryClient for LocalClient {
    /// Fast path: return cached snapshot (no channel round-trip).
    fn cached_list(&self) -> Option<Vec<TrackedRepository>> {
        Some(self.cache.load().as_ref().clone())
    }

    /// List all repositories (through channel, gets fresh data).
    async fn list(&self) -> Result<Vec<TrackedRepository>, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::List { reply: tx })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Get repository by ID.
    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::Get {
                id: id.clone(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Get repository by name.
    async fn get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::GetByName {
                name: name.to_string(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Clone a repository from URL.
    async fn clone_repo(&self, url: &str, name: Option<&str>) -> Result<RepoId, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::Clone {
                url: url.to_string(),
                name: name.map(String::from),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Register an existing repository.
    async fn register(
        &self,
        id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::Register {
                id: id.clone(),
                name: name.map(String::from),
                path: path.to_path_buf(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Upsert: update if exists, create if not.
    async fn upsert(&self, name: &str, url: &str) -> Result<RepoId, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::Upsert {
                name: name.to_string(),
                url: url.to_string(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Remove a repository from the registry.
    async fn remove(&self, id: &RepoId) -> Result<(), ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::Remove {
                id: id.clone(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Health check.
    async fn health_check(&self) -> Result<(), ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::HealthCheck { reply: tx })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    /// Get a scoped repository client by name.
    async fn repo(&self, name: &str) -> Result<Arc<dyn RepositoryClient>, ServiceError> {
        // Find repository by name
        let tracked = self.get_by_name(name).await?;
        let tracked = tracked.ok_or_else(|| {
            ServiceError::channel(format!("Repository '{}' not found", name))
        })?;

        let repo_name = tracked.name.unwrap_or_else(|| name.to_string());
        Ok(Arc::new(LocalRepositoryClient {
            sender: self.sender.clone(),
            repo_id: tracked.id,
            name: repo_name,
        }))
    }

    /// Get a scoped repository client by ID.
    async fn repo_by_id(&self, id: &RepoId) -> Result<Arc<dyn RepositoryClient>, ServiceError> {
        // Find repository by ID
        let tracked = self.get(id).await?;
        let tracked = tracked.ok_or_else(|| {
            ServiceError::channel(format!("Repository with ID {:?} not found", id))
        })?;

        let name = tracked.name.unwrap_or_else(|| id.to_string());
        Ok(Arc::new(LocalRepositoryClient {
            sender: self.sender.clone(),
            repo_id: id.clone(),
            name,
        }))
    }
}

/// Scoped repository client for a specific repository.
///
/// This client sends requests to the LocalService for a specific repository.
/// All operations use RepositoryHandle internally, ensuring proper LFS handling.
#[derive(Clone)]
pub struct LocalRepositoryClient {
    sender: mpsc::UnboundedSender<RegistryRequest>,
    repo_id: RepoId,
    name: String,
}

#[async_trait]
impl RepositoryClient for LocalRepositoryClient {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> &RepoId {
        &self.repo_id
    }

    async fn create_worktree(&self, path: &Path, branch: &str) -> Result<PathBuf, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::CreateWorktree {
                repo_id: self.repo_id.clone(),
                worktree_path: path.to_path_buf(),
                branch: branch.to_string(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::ListWorktrees {
                repo_id: self.repo_id.clone(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    async fn worktree_path(&self, branch: &str) -> Result<Option<PathBuf>, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::WorktreePath {
                repo_id: self.repo_id.clone(),
                branch: branch.to_string(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    async fn create_branch(&self, name: &str, from: Option<&str>) -> Result<(), ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::CreateBranch {
                repo_id: self.repo_id.clone(),
                name: name.to_string(),
                from: from.map(String::from),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    async fn checkout(&self, ref_spec: &str) -> Result<(), ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::Checkout {
                repo_id: self.repo_id.clone(),
                ref_spec: ref_spec.to_string(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    async fn default_branch(&self) -> Result<String, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::DefaultBranch {
                repo_id: self.repo_id.clone(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }

    async fn list_branches(&self) -> Result<Vec<String>, ServiceError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RegistryRequest::ListBranches {
                repo_id: self.repo_id.clone(),
                reply: tx,
            })
            .map_err(|_| ServiceError::Unavailable)?;
        rx.await.map_err(|_| ServiceError::channel("No response"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_service_start_and_health_check() {
        let temp_dir = TempDir::new().unwrap();
        let client = LocalService::start(temp_dir.path()).await.unwrap();

        // Health check should succeed
        client.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_service_list_empty() {
        let temp_dir = TempDir::new().unwrap();
        let client = LocalService::start(temp_dir.path()).await.unwrap();

        let repos = client.list().await.unwrap();
        assert!(repos.is_empty());
    }

    #[tokio::test]
    async fn test_cached_list() {
        let temp_dir = TempDir::new().unwrap();
        let client = LocalService::start(temp_dir.path()).await.unwrap();

        // Cached list should work
        let cached = client.cached_list();
        assert!(cached.is_some());
        assert!(cached.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_clone_shares_cache() {
        let temp_dir = TempDir::new().unwrap();
        let client1 = LocalService::start(temp_dir.path()).await.unwrap();
        let client2 = client1.clone();

        // Both clients should see same data
        let repos1 = client1.list().await.unwrap();
        let repos2 = client2.list().await.unwrap();
        assert_eq!(repos1.len(), repos2.len());
    }
}
