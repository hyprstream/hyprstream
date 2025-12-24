//! ZMQ-based registry service for repository management
//!
//! This service wraps git2db and provides a ZMQ REQ/REP interface for
//! repository operations. It uses Cap'n Proto for serialization.

use crate::registry_capnp;
use crate::services::{ServiceRunner, ZmqService};
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use git2db::{Git2DB, RepoId, TrackedRepository, GitRef};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Default endpoint for the registry service
pub const REGISTRY_ENDPOINT: &str = "inproc://hyprstream/registry";

/// ZMQ-based registry service
///
/// Wraps git2db::Git2DB and provides a Cap'n Proto interface over ZMQ.
///
/// Note: The runtime handle is captured on creation to avoid the nested runtime
/// anti-pattern. Handler methods use this handle for async operations instead
/// of creating new runtimes.
///
/// The registry is wrapped in RwLock for interior mutability since some operations
/// (like clone) require mutable access but ZmqService::handle_request takes &self.
pub struct RegistryService {
    registry: RwLock<Git2DB>,
    base_dir: PathBuf,
    /// Captured runtime handle for async operations in sync handlers
    runtime_handle: tokio::runtime::Handle,
}

impl RegistryService {
    /// Create a new registry service
    ///
    /// Must be called from within a tokio runtime context.
    pub async fn new(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let registry = Git2DB::open(&base_dir).await?;
        // Capture the runtime handle to avoid nested runtime anti-pattern
        let runtime_handle = tokio::runtime::Handle::current();

        Ok(Self {
            registry: RwLock::new(registry),
            base_dir,
            runtime_handle,
        })
    }

    /// Start the registry service on the default endpoint
    pub async fn start(base_dir: impl AsRef<Path>) -> Result<crate::services::ServiceHandle> {
        Self::start_at(base_dir, REGISTRY_ENDPOINT).await
    }

    /// Start the registry service at a specific endpoint
    pub async fn start_at(
        base_dir: impl AsRef<Path>,
        endpoint: &str,
    ) -> Result<crate::services::ServiceHandle> {
        let service = Self::new(base_dir).await?;
        let runner = ServiceRunner::new(endpoint);
        Ok(runner.run(service))
    }

    /// Parse a RepoId from string
    fn parse_repo_id(id_str: &str) -> Result<RepoId> {
        let uuid = Uuid::parse_str(id_str)
            .map_err(|e| anyhow!("invalid repo id '{}': {}", id_str, e))?;
        Ok(RepoId::from_uuid(uuid))
    }

    /// Handle a list repositories request
    fn handle_list(&self) -> Result<Vec<TrackedRepository>> {
        let registry = self.registry.read()
            .map_err(|_| anyhow!("registry lock poisoned"))?;
        Ok(registry.list().cloned().collect())
    }

    /// Handle get repository by ID
    fn handle_get(&self, repo_id: &str) -> Result<Option<TrackedRepository>> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read()
            .map_err(|_| anyhow!("registry lock poisoned"))?;
        let result = registry.list().find(|r| r.id == id).cloned();
        Ok(result)
    }

    /// Handle get repository by name
    fn handle_get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>> {
        let registry = self.registry.read()
            .map_err(|_| anyhow!("registry lock poisoned"))?;
        let result = registry
            .list()
            .find(|r| r.name.as_ref() == Some(&name.to_string()))
            .cloned();
        Ok(result)
    }

    /// Handle list branches
    fn handle_list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            let branches = handle.branch().list().await?;
            Ok(branches.into_iter().map(|b| b.name).collect())
        })
    }

    /// Handle create branch
    fn handle_create_branch(
        &self,
        repo_id: &str,
        branch_name: &str,
        start_point: Option<&str>,
    ) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.branch().create(branch_name, start_point).await?;
            Ok(())
        })
    }

    /// Handle checkout
    fn handle_checkout(&self, repo_id: &str, ref_name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.branch().checkout(ref_name).await?;
            Ok(())
        })
    }

    /// Handle stage all
    fn handle_stage_all(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.staging().add_all().await?;
            Ok(())
        })
    }

    /// Handle commit
    fn handle_commit(&self, repo_id: &str, message: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            let oid = handle.commit(message).await?;
            Ok(oid.to_string())
        })
    }

    /// Handle clone operation
    fn handle_clone(
        &self,
        url: &str,
        name: Option<&str>,
        shallow: bool,
        depth: Option<u32>,
        branch: Option<&str>,
    ) -> Result<TrackedRepository> {
        // Clone operation needs write lock since it mutates registry
        let url = url.to_string();
        let name = name.map(|s| s.to_string());
        let branch = branch.map(|s| s.to_string());

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let mut registry = self.registry.write()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let mut clone_builder = registry.clone(&url);

            if let Some(ref n) = name {
                clone_builder = clone_builder.name(n);
            }

            // depth > 0 implies shallow clone
            // if shallow is explicitly set but depth is 0, use depth=1
            if let Some(d) = depth {
                if d > 0 {
                    clone_builder = clone_builder.depth(d);
                }
            } else if shallow {
                clone_builder = clone_builder.depth(1);
            }

            if let Some(ref b) = branch {
                if !b.is_empty() {
                    clone_builder = clone_builder.branch(b);
                }
            }

            let repo_id = clone_builder.exec().await?;

            // Get the tracked repository to return
            let repo = registry
                .list()
                .find(|r| r.id == repo_id)
                .cloned()
                .ok_or_else(|| anyhow!("Failed to find cloned repository"))?;

            Ok(repo)
        })
    }

    /// Handle list worktrees
    fn handle_list_worktrees(&self, repo_id: &str) -> Result<Vec<WorktreeData>> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            let worktrees = handle.get_worktrees().await?;
            Ok(worktrees
                .into_iter()
                .map(|wt| {
                    // Extract branch name from worktree path (last component)
                    let branch_name = wt
                        .path()
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string());

                    WorktreeData {
                        path: wt.path().to_path_buf(),
                        branch_name,
                        head_oid: String::new(),
                        is_locked: false,
                    }
                })
                .collect())
        })
    }

    /// Handle create worktree
    fn handle_create_worktree(
        &self,
        repo_id: &str,
        path: &str,
        branch: &str,
    ) -> Result<PathBuf> {
        let id = Self::parse_repo_id(repo_id)?;
        let path = PathBuf::from(path);
        let branch = branch.to_string();

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            let worktree = handle.create_worktree(&path, &branch).await?;
            Ok(worktree.path().to_path_buf())
        })
    }

    /// Handle register operation
    #[allow(deprecated)]
    fn handle_register(
        &self,
        path: &str,
        name: Option<&str>,
        _tracking_ref: Option<&str>,
    ) -> Result<TrackedRepository> {
        let path = PathBuf::from(path);
        let name = name.map(|s| s.to_string());
        // Note: tracking_ref is not yet used by register_repository

        // Register requires write lock
        self.runtime_handle.block_on(async {
            let mut registry = self.registry.write()
                .map_err(|_| anyhow!("registry lock poisoned"))?;

            // Generate a new repo ID
            let repo_id = RepoId::new();

            // Derive URL from path (local file URL)
            let url = format!("file://{}", path.display());

            // Use deprecated method for now (builder pattern would require more refactoring)
            registry.register_repository(&repo_id, name, url).await?;

            // Get the tracked repository to return
            let repo = registry
                .list()
                .find(|r| r.id == repo_id)
                .cloned()
                .ok_or_else(|| anyhow!("Failed to find registered repository"))?;

            Ok(repo)
        })
    }

    /// Handle remove operation
    fn handle_remove(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        // Remove requires write lock
        self.runtime_handle.block_on(async {
            let mut registry = self.registry.write()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            registry.remove_repository(&id).await?;
            Ok(())
        })
    }

    /// Handle remove worktree operation
    fn handle_remove_worktree(&self, repo_id: &str, worktree_path: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let worktree_path = PathBuf::from(worktree_path);

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;

            // Get base repo path
            let repo_path = handle.worktree()?;

            // Extract worktree name from path
            let worktree_name = worktree_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("Invalid worktree path"))?;

            // Use GitManager to remove worktree
            git2db::GitManager::global().remove_worktree(repo_path, worktree_name, None)?;
            Ok(())
        })
    }

    /// Handle stage files operation
    fn handle_stage_files(&self, repo_id: &str, files: Vec<String>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;

            for file in files {
                handle.staging().add(&file).await?;
            }
            Ok(())
        })
    }

    /// Handle get HEAD operation
    fn handle_get_head(&self, repo_id: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;

            // Get HEAD ref (DefaultBranch resolves to HEAD)
            let oid = handle.resolve_git_ref(&GitRef::DefaultBranch).await?;
            Ok(oid.to_string())
        })
    }

    /// Handle get ref operation
    fn handle_get_ref(&self, repo_id: &str, ref_name: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let ref_name = ref_name.to_string();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;

            // Try to resolve as a revspec
            let oid = handle.resolve_revspec(&ref_name).await?;
            Ok(oid.to_string())
        })
    }

    /// Handle update operation (fetch from remote)
    ///
    /// Note: Full fetch support requires storage driver access.
    /// For now, this is a placeholder that returns success.
    fn handle_update(&self, repo_id: &str, _refspec: Option<&str>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        // Verify the repository exists
        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let _handle = registry.repo(&id)?;

            // TODO: Implement full fetch via storage driver
            // For now, just verify the repo exists and return success
            Ok(())
        })
    }

    /// Handle health check
    fn handle_health_check(&self) -> Result<HealthStatus> {
        let registry = self.registry.read()
            .map_err(|_| anyhow!("registry lock poisoned"))?;
        let repo_count = registry.list().count() as u32;
        Ok(HealthStatus {
            status: "healthy".to_string(),
            repository_count: repo_count,
            worktree_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Handle list remotes operation
    fn handle_list_remotes(&self, repo_id: &str) -> Result<Vec<RemoteInfo>> {
        let id = Self::parse_repo_id(repo_id)?;

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;

            let remotes = handle.remote().list().await?;
            let result = remotes
                .into_iter()
                .map(|remote| RemoteInfo {
                    name: remote.name,
                    url: remote.url,
                    push_url: remote.push_url,
                })
                .collect();
            Ok(result)
        })
    }

    /// Handle add remote operation
    fn handle_add_remote(&self, repo_id: &str, name: &str, url: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_string();
        let url = url.to_string();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.remote().add(&name, &url).await?;
            Ok(())
        })
    }

    /// Handle remove remote operation
    fn handle_remove_remote(&self, repo_id: &str, name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_string();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.remote().remove(&name).await?;
            Ok(())
        })
    }

    /// Handle set remote URL operation
    fn handle_set_remote_url(&self, repo_id: &str, name: &str, url: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_string();
        let url = url.to_string();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.remote().set_url(&name, &url).await?;
            Ok(())
        })
    }

    /// Handle rename remote operation
    fn handle_rename_remote(&self, repo_id: &str, old_name: &str, new_name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let old_name = old_name.to_string();
        let new_name = new_name.to_string();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;
            handle.remote().rename(&old_name, &new_name).await?;
            Ok(())
        })
    }

    /// Build an error response
    fn build_error_response(&self, request_id: u64, error: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut error_info = response.init_error();
        error_info.set_message(error);
        error_info.set_code("ERROR");
        error_info.set_details("");

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a success response
    fn build_success_response(&self, request_id: u64) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_success(());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a path response (for worktree creation, etc.)
    fn build_path_response(&self, request_id: u64, path: &Path) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_path(&path.to_string_lossy());

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a repository response
    fn build_repository_response(
        &self,
        request_id: u64,
        repo: &TrackedRepository,
    ) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut repo_builder = response.init_repository();
        repo_builder.set_id(&repo.id.to_string());
        if let Some(ref name) = repo.name {
            repo_builder.set_name(name);
        }
        repo_builder.set_url(&repo.url);
        repo_builder.set_worktree_path(&repo.worktree_path.to_string_lossy());
        repo_builder.set_tracking_ref(&repo.tracking_ref.to_string());
        if let Some(ref oid) = repo.current_oid {
            repo_builder.set_current_oid(oid);
        }
        repo_builder.set_registered_at(repo.registered_at);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a repositories list response
    fn build_repositories_response(
        &self,
        request_id: u64,
        repos: &[TrackedRepository],
    ) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut repos_builder = response.init_repositories(repos.len() as u32);
        for (i, repo) in repos.iter().enumerate() {
            let mut repo_builder = repos_builder.reborrow().get(i as u32);
            repo_builder.set_id(&repo.id.to_string());
            if let Some(ref name) = repo.name {
                repo_builder.set_name(name);
            }
            repo_builder.set_url(&repo.url);
            repo_builder.set_worktree_path(&repo.worktree_path.to_string_lossy());
            repo_builder.set_tracking_ref(&repo.tracking_ref.to_string());
            if let Some(ref oid) = repo.current_oid {
                repo_builder.set_current_oid(oid);
            }
            repo_builder.set_registered_at(repo.registered_at);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a branches list response
    fn build_branches_response(&self, request_id: u64, branches: &[String]) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut branches_builder = response.init_branches(branches.len() as u32);
        for (i, branch) in branches.iter().enumerate() {
            branches_builder.set(i as u32, branch);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a commit OID response
    fn build_commit_oid_response(&self, request_id: u64, oid: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_commit_oid(oid);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a ref OID response (for getHead, getRef)
    fn build_ref_oid_response(&self, request_id: u64, oid: &str) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);
        response.set_ref_oid(oid);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a remotes list response
    fn build_remotes_response(&self, request_id: u64, remotes: &[RemoteInfo]) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut remotes_list = response.init_remotes(remotes.len() as u32);
        for (i, remote) in remotes.iter().enumerate() {
            let mut r = remotes_list.reborrow().get(i as u32);
            r.set_name(&remote.name);
            r.set_url(&remote.url);
            if let Some(push_url) = &remote.push_url {
                r.set_push_url(push_url);
            }
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a health status response
    fn build_health_response(&self, request_id: u64, status: &HealthStatus) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut health = response.init_health();
        health.set_status(&status.status);
        health.set_repository_count(status.repository_count);
        health.set_worktree_count(status.worktree_count);
        health.set_cache_hits(status.cache_hits);
        health.set_cache_misses(status.cache_misses);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }

    /// Build a worktrees list response
    fn build_worktrees_response(&self, request_id: u64, worktrees: &[WorktreeData]) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut response = message.init_root::<registry_capnp::registry_response::Builder>();
        response.set_request_id(request_id);

        let mut wt_builder = response.init_worktrees(worktrees.len() as u32);
        for (i, wt) in worktrees.iter().enumerate() {
            let mut wt_entry = wt_builder.reborrow().get(i as u32);
            wt_entry.set_path(&wt.path.to_string_lossy());
            if let Some(ref branch) = wt.branch_name {
                wt_entry.set_branch_name(branch);
            }
            wt_entry.set_head_oid(&wt.head_oid);
            wt_entry.set_is_locked(wt.is_locked);
        }

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).unwrap_or_default();
        bytes
    }
}

impl ZmqService for RegistryService {
    fn handle_request(&self, request: &[u8]) -> Result<Vec<u8>> {
        // Deserialize request
        let reader = serialize::read_message(request, ReaderOptions::new())?;
        let req = reader.get_root::<registry_capnp::registry_request::Reader>()?;

        let request_id = req.get_id();

        // Handle based on request type
        use registry_capnp::registry_request::Which;

        match req.which()? {
            Which::List(()) => {
                match self.handle_list() {
                    Ok(repos) => Ok(self.build_repositories_response(request_id, &repos)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Get(id) => {
                let id = id?;
                match self.handle_get(id.to_str()?) {
                    Ok(Some(repo)) => Ok(self.build_repository_response(request_id, &repo)),
                    Ok(None) => Ok(self.build_error_response(request_id, "Repository not found")),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::GetByName(name) => {
                let name = name?;
                match self.handle_get_by_name(name.to_str()?) {
                    Ok(Some(repo)) => Ok(self.build_repository_response(request_id, &repo)),
                    Ok(None) => Ok(self.build_error_response(request_id, "Repository not found")),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Clone(clone_req) => {
                let clone_req = clone_req?;
                let url = clone_req.get_url()?.to_str()?;
                let name = if clone_req.has_name() {
                    let n = clone_req.get_name()?.to_str()?;
                    if n.is_empty() { None } else { Some(n) }
                } else {
                    None
                };
                let shallow = clone_req.get_shallow();
                let depth = clone_req.get_depth();
                let branch = if clone_req.has_branch() {
                    let b = clone_req.get_branch()?.to_str()?;
                    if b.is_empty() { None } else { Some(b) }
                } else {
                    None
                };

                match self.handle_clone(url, name, shallow, Some(depth), branch) {
                    Ok(repo) => Ok(self.build_repository_response(request_id, &repo)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Register(reg_req) => {
                let reg_req = reg_req?;
                let path = reg_req.get_path()?.to_str()?;
                let name = if reg_req.has_name() {
                    let n = reg_req.get_name()?.to_str()?;
                    if n.is_empty() { None } else { Some(n) }
                } else {
                    None
                };
                let tracking_ref = if reg_req.has_tracking_ref() {
                    let r = reg_req.get_tracking_ref()?.to_str()?;
                    if r.is_empty() { None } else { Some(r) }
                } else {
                    None
                };

                match self.handle_register(path, name, tracking_ref) {
                    Ok(repo) => Ok(self.build_repository_response(request_id, &repo)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Remove(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                match self.handle_remove(repo_id) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListBranches(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                match self.handle_list_branches(repo_id) {
                    Ok(branches) => Ok(self.build_branches_response(request_id, &branches)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::CreateBranch(branch_req) => {
                let branch_req = branch_req?;
                let repo_id = branch_req.get_repo_id()?.to_str()?;
                let branch_name = branch_req.get_branch_name()?.to_str()?;
                let start_point = if branch_req.has_start_point() {
                    Some(branch_req.get_start_point()?.to_str()?)
                } else {
                    None
                };

                match self.handle_create_branch(repo_id, branch_name, start_point) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Checkout(checkout_req) => {
                let checkout_req = checkout_req?;
                let repo_id = checkout_req.get_repo_id()?.to_str()?;
                let ref_name = checkout_req.get_ref_name()?.to_str()?;

                match self.handle_checkout(repo_id, ref_name) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::StageAll(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                match self.handle_stage_all(repo_id) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Commit(commit_req) => {
                let commit_req = commit_req?;
                let repo_id = commit_req.get_repo_id()?.to_str()?;
                let message = commit_req.get_message()?.to_str()?;

                match self.handle_commit(repo_id, message) {
                    Ok(oid) => Ok(self.build_commit_oid_response(request_id, &oid)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::HealthCheck(()) => {
                match self.handle_health_check() {
                    Ok(status) => Ok(self.build_health_response(request_id, &status)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListWorktrees(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                match self.handle_list_worktrees(repo_id) {
                    Ok(worktrees) => Ok(self.build_worktrees_response(request_id, &worktrees)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::CreateWorktree(wt_req) => {
                let wt_req = wt_req?;
                let repo_id = wt_req.get_repo_id()?.to_str()?;
                let path = wt_req.get_path()?.to_str()?;
                let branch = wt_req.get_branch_name()?.to_str()?;

                match self.handle_create_worktree(repo_id, path, branch) {
                    Ok(wt_path) => Ok(self.build_path_response(request_id, &wt_path)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::RemoveWorktree(wt_req) => {
                let wt_req = wt_req?;
                let repo_id = wt_req.get_repo_id()?.to_str()?;
                let worktree_path = wt_req.get_worktree_path()?.to_str()?;

                match self.handle_remove_worktree(repo_id, worktree_path) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::Update(update_req) => {
                let update_req = update_req?;
                let repo_id = update_req.get_repo_id()?.to_str()?;
                let refspec = if update_req.has_refspec() {
                    let r = update_req.get_refspec()?.to_str()?;
                    if r.is_empty() { None } else { Some(r) }
                } else {
                    None
                };

                match self.handle_update(repo_id, refspec) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::StageFiles(stage_req) => {
                let stage_req = stage_req?;
                let repo_id = stage_req.get_repo_id()?.to_str()?;
                let files_reader = stage_req.get_files()?;
                let mut files = Vec::new();
                for file in files_reader.iter() {
                    files.push(file?.to_str()?.to_string());
                }

                match self.handle_stage_files(repo_id, files) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::GetHead(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                match self.handle_get_head(repo_id) {
                    Ok(oid) => Ok(self.build_ref_oid_response(request_id, &oid)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::GetRef(ref_req) => {
                let ref_req = ref_req?;
                let repo_id = ref_req.get_repo_id()?.to_str()?;
                let ref_name = ref_req.get_ref_name()?.to_str()?;

                match self.handle_get_ref(repo_id, ref_name) {
                    Ok(oid) => Ok(self.build_ref_oid_response(request_id, &oid)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::ListRemotes(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                match self.handle_list_remotes(repo_id) {
                    Ok(remotes) => Ok(self.build_remotes_response(request_id, &remotes)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::AddRemote(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                let name = remote_req.get_name()?.to_str()?;
                let url = remote_req.get_url()?.to_str()?;

                match self.handle_add_remote(repo_id, name, url) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::RemoveRemote(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                let name = remote_req.get_name()?.to_str()?;

                match self.handle_remove_remote(repo_id, name) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::SetRemoteUrl(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                let name = remote_req.get_name()?.to_str()?;
                let url = remote_req.get_url()?.to_str()?;

                match self.handle_set_remote_url(repo_id, name, url) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }

            Which::RenameRemote(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                let old_name = remote_req.get_old_name()?.to_str()?;
                let new_name = remote_req.get_new_name()?.to_str()?;

                match self.handle_rename_remote(repo_id, old_name, new_name) {
                    Ok(()) => Ok(self.build_success_response(request_id)),
                    Err(e) => Ok(self.build_error_response(request_id, &e.to_string())),
                }
            }
        }
    }

    fn name(&self) -> &str {
        "registry"
    }
}

/// Health status for the registry service
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub status: String,
    pub repository_count: u32,
    pub worktree_count: u32,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Worktree data for internal use (matches Cap'n Proto schema)
#[derive(Debug, Clone)]
pub struct WorktreeData {
    pub path: PathBuf,
    pub branch_name: Option<String>,
    pub head_oid: String,
    pub is_locked: bool,
}

/// ZMQ client for the registry service
///
/// Provides a convenient async interface for communicating with the registry service.
/// Implements the `RegistryClient` trait from `crate::services::traits`.
pub struct RegistryZmqClient {
    client: crate::services::AsyncServiceClient,
    request_id: std::sync::atomic::AtomicU64,
}

impl RegistryZmqClient {
    /// Create a new registry client
    pub fn new() -> Self {
        Self::with_endpoint(REGISTRY_ENDPOINT)
    }

    /// Create a registry client connected to a specific endpoint
    pub fn with_endpoint(endpoint: &str) -> Self {
        Self {
            client: crate::services::AsyncServiceClient::new(endpoint),
            request_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Get the next request ID
    fn next_id(&self) -> u64 {
        self.request_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// List all repositories
    pub async fn list(&self) -> Result<Vec<TrackedRepository>> {
        let id = self.next_id();

        // Scope the Cap'n Proto builder so it drops before the await
        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_list(());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            bytes
        };

        let response = self.client.call(bytes).await?;
        self.parse_repositories_response(&response)
    }

    /// Get repository by ID
    pub async fn get(&self, repo_id: &str) -> Result<Option<TrackedRepository>> {
        let id = self.next_id();

        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_get(repo_id);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            bytes
        };

        let response = self.client.call(bytes).await?;
        self.parse_optional_repository_response(&response)
    }

    /// Get repository by name
    pub async fn get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>> {
        let id = self.next_id();

        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_get_by_name(name);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            bytes
        };

        let response = self.client.call(bytes).await?;
        self.parse_optional_repository_response(&response)
    }

    /// List branches for a repository
    pub async fn list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = self.next_id();

        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_list_branches(repo_id);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            bytes
        };

        let response = self.client.call(bytes).await?;
        self.parse_branches_response(&response)
    }

    /// Health check
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let id = self.next_id();

        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_health_check(());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            bytes
        };

        let response = self.client.call(bytes).await?;
        self.parse_health_response(&response)
    }

    /// Parse a repositories list response
    fn parse_repositories_response(&self, response: &[u8]) -> Result<Vec<TrackedRepository>> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()?;

        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Repositories(repos) => {
                let repos = repos?;
                let mut result = Vec::new();
                for repo in repos.iter() {
                    result.push(self.parse_tracked_repository(repo)?);
                }
                Ok(result)
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse an optional repository response
    fn parse_optional_repository_response(
        &self,
        response: &[u8],
    ) -> Result<Option<TrackedRepository>> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()?;

        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Repository(repo) => {
                let repo = repo?;
                Ok(Some(self.parse_tracked_repository(repo)?))
            }
            Which::Error(_) => Ok(None),
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse a branches list response
    fn parse_branches_response(&self, response: &[u8]) -> Result<Vec<String>> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()?;

        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Branches(branches) => {
                let branches = branches?;
                let mut result = Vec::new();
                for branch in branches.iter() {
                    result.push(branch?.to_str()?.to_string());
                }
                Ok(result)
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse a health response
    fn parse_health_response(&self, response: &[u8]) -> Result<HealthStatus> {
        let reader = serialize::read_message(response, ReaderOptions::new())?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()?;

        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Health(health) => {
                let health = health?;
                Ok(HealthStatus {
                    status: health.get_status()?.to_str()?.to_string(),
                    repository_count: health.get_repository_count(),
                    worktree_count: health.get_worktree_count(),
                    cache_hits: health.get_cache_hits(),
                    cache_misses: health.get_cache_misses(),
                })
            }
            Which::Error(err) => {
                let err = err?;
                Err(anyhow!("{}", err.get_message()?.to_str()?))
            }
            _ => Err(anyhow!("Unexpected response type")),
        }
    }

    /// Parse a TrackedRepository from capnp reader
    fn parse_tracked_repository(
        &self,
        repo: registry_capnp::tracked_repository::Reader,
    ) -> Result<TrackedRepository> {
        let id_str = repo.get_id()?.to_str()?;
        let id = RepoId::from_uuid(
            Uuid::parse_str(id_str).map_err(|e| anyhow!("invalid repo id: {}", e))?,
        );

        let name = if repo.has_name() {
            Some(repo.get_name()?.to_str()?.to_string())
        } else {
            None
        };

        let url = repo.get_url()?.to_str()?.to_string();
        let worktree_path = PathBuf::from(repo.get_worktree_path()?.to_str()?);

        let tracking_ref_str = repo.get_tracking_ref()?.to_str()?;
        let tracking_ref = if tracking_ref_str.is_empty() {
            GitRef::Branch("main".to_string())
        } else {
            GitRef::Branch(tracking_ref_str.to_string())
        };

        let current_oid = if repo.has_current_oid() {
            Some(repo.get_current_oid()?.to_str()?.to_string())
        } else {
            None
        };

        let registered_at = repo.get_registered_at();

        Ok(TrackedRepository {
            id,
            name,
            url,
            worktree_path,
            tracking_ref,
            remotes: Vec::new(),
            registered_at,
            current_oid,
            metadata: HashMap::new(),
        })
    }
}

impl Default for RegistryZmqClient {
    fn default() -> Self {
        Self::new()
    }
}

// === RegistryClient Trait Implementation ===

use crate::services::traits::{RegistryClient, RegistryServiceError, RemoteInfo, RepositoryClient, WorktreeInfo};
use async_trait::async_trait;

#[async_trait]
impl RegistryClient for RegistryZmqClient {
    async fn list(&self) -> Result<Vec<TrackedRepository>, RegistryServiceError> {
        self.list()
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, RegistryServiceError> {
        self.get(&id.to_string())
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn get_by_name(
        &self,
        name: &str,
    ) -> Result<Option<TrackedRepository>, RegistryServiceError> {
        self.get_by_name(name)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn clone_repo(
        &self,
        url: &str,
        name: Option<&str>,
    ) -> Result<RepoId, RegistryServiceError> {
        let id = self.next_id();

        // Build clone request - scope builder to drop before await
        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut clone_req = req.init_clone();
            clone_req.set_url(url);
            if let Some(n) = name {
                clone_req.set_name(n);
            }
            clone_req.set_shallow(false);
            clone_req.set_depth(0);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = self.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Repository(repo)) => {
                let repo = repo.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let id_str = repo.get_id()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let uuid = Uuid::parse_str(id_str)
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Ok(RepoId::from_uuid(uuid))
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn register(
        &self,
        _id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), RegistryServiceError> {
        let id = self.next_id();

        // Build register request - scope builder to drop before await
        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut reg_req = req.init_register();
            reg_req.set_path(
                path.to_str()
                    .ok_or_else(|| RegistryServiceError::transport("Invalid path encoding"))?,
            );
            if let Some(n) = name {
                reg_req.set_name(n);
            }

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = self
            .client
            .call(bytes)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader
            .get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err
                    .get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn upsert(&self, _name: &str, _url: &str) -> Result<RepoId, RegistryServiceError> {
        // Upsert operation not yet supported via ZMQ
        Err(RegistryServiceError::transport(
            "Upsert operation not yet supported via ZMQ",
        ))
    }

    async fn remove(&self, id: &RepoId) -> Result<(), RegistryServiceError> {
        let req_id = self.next_id();

        // Build remove request - scope builder to drop before await
        let bytes = {
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(req_id);
            req.set_remove(&id.to_string());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = self
            .client
            .call(bytes)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader
            .get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err
                    .get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn health_check(&self) -> Result<(), RegistryServiceError> {
        self.health_check()
            .await
            .map(|_| ())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn repo(
        &self,
        name: &str,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError> {
        // Get repository by name first
        let repo = self
            .get_by_name(name)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
            .ok_or_else(|| RegistryServiceError::transport(format!("Repository '{}' not found", name)))?;

        Ok(Arc::new(RepositoryZmqClient::new(
            self.client.endpoint().to_string(),
            repo.id,
            name.to_string(),
        )))
    }

    async fn repo_by_id(
        &self,
        id: &RepoId,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError> {
        // Get repository by ID first
        let repo = self
            .get(&id.to_string())
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
            .ok_or_else(|| RegistryServiceError::transport(format!("Repository '{}' not found", id)))?;

        let name = repo.name.unwrap_or_else(|| id.to_string());

        Ok(Arc::new(RepositoryZmqClient::new(
            self.client.endpoint().to_string(),
            id.clone(),
            name,
        )))
    }
}

// === hyprstream_metrics::RegistryClient Implementation ===
//
// This allows RegistryZmqClient to be used with hyprstream_flight and other
// crates that depend on the minimal hyprstream_metrics::RegistryClient trait.

use hyprstream_metrics::checkpoint::manager::{
    RegistryClient as MetricsRegistryClient, RegistryError as MetricsRegistryError,
};

#[async_trait]
impl MetricsRegistryClient for RegistryZmqClient {
    async fn get_by_name(
        &self,
        name: &str,
    ) -> Result<Option<TrackedRepository>, MetricsRegistryError> {
        self.get_by_name(name)
            .await
            .map_err(|e| MetricsRegistryError::Operation(e.to_string()))
    }

    async fn register(
        &self,
        id: &RepoId,
        name: Option<&str>,
        path: &std::path::Path,
    ) -> Result<(), MetricsRegistryError> {
        // Delegate to the RegistryClient implementation
        RegistryClient::register(self, id, name, path)
            .await
            .map_err(|e| MetricsRegistryError::Operation(e.to_string()))
    }
}

/// ZMQ client for repository-level operations
///
/// Provides operations on a specific repository through the registry service.
pub struct RepositoryZmqClient {
    endpoint: String,
    repo_id: RepoId,
    repo_name: String,
}

impl RepositoryZmqClient {
    /// Create a new repository client
    pub fn new(endpoint: String, repo_id: RepoId, repo_name: String) -> Self {
        Self {
            endpoint,
            repo_id,
            repo_name,
        }
    }

    /// Parse a worktrees list response
    fn parse_worktrees_response(
        &self,
        response: &[u8],
    ) -> Result<Vec<WorktreeInfo>, RegistryServiceError> {
        let reader = serialize::read_message(response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader
            .get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Worktrees(wts)) => {
                let wts = wts.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let mut result = Vec::new();
                for wt in wts.iter() {
                    let path_text = wt
                        .get_path()
                        .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                    let path_str = path_text
                        .to_str()
                        .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

                    let branch = if wt.has_branch_name() {
                        let branch_text = wt
                            .get_branch_name()
                            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                        Some(
                            branch_text
                                .to_str()
                                .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                                .to_string(),
                        )
                    } else {
                        None
                    };

                    result.push(WorktreeInfo {
                        path: PathBuf::from(path_str),
                        branch,
                        driver: "zmq".to_string(), // Default driver name for ZMQ-fetched worktrees
                    });
                }
                Ok(result)
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err
                    .get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    /// Parse a remotes list response
    fn parse_remotes_response(
        &self,
        response: &[u8],
    ) -> Result<Vec<RemoteInfo>, RegistryServiceError> {
        let reader = serialize::read_message(response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader
            .get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Remotes(remotes)) => {
                let remotes = remotes.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let mut result = Vec::new();
                for remote in remotes.iter() {
                    let name = remote
                        .get_name()
                        .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                        .to_str()
                        .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                        .to_string();

                    let url = remote
                        .get_url()
                        .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                        .to_str()
                        .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                        .to_string();

                    let push_url = if remote.has_push_url() {
                        let pu = remote
                            .get_push_url()
                            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                            .to_str()
                            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                        if pu.is_empty() {
                            None
                        } else {
                            Some(pu.to_string())
                        }
                    } else {
                        None
                    };

                    result.push(RemoteInfo { name, url, push_url });
                }
                Ok(result)
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err
                    .get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }
}

#[async_trait]
impl RepositoryClient for RepositoryZmqClient {
    fn name(&self) -> &str {
        &self.repo_name
    }

    fn id(&self) -> &RepoId {
        &self.repo_id
    }

    async fn create_worktree(
        &self,
        path: &Path,
        branch: &str,
    ) -> Result<PathBuf, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build createWorktree request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut wt_req = req.init_create_worktree();
            wt_req.set_repo_id(&self.repo_id.to_string());
            wt_req.set_path(&path.to_string_lossy());
            wt_req.set_branch_name(branch);
            wt_req.set_create_branch(false);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Path(path_text)) => {
                let path_text = path_text.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let path_str = path_text.to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Ok(PathBuf::from(path_str))
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build listWorktrees request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_list_worktrees(&self.repo_id.to_string());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        self.parse_worktrees_response(&response)
    }

    async fn worktree_path(&self, _branch: &str) -> Result<Option<PathBuf>, RegistryServiceError> {
        // Worktree path lookup not yet supported via ZMQ
        Err(RegistryServiceError::transport(
            "Worktree path lookup not yet supported via ZMQ",
        ))
    }

    async fn create_branch(
        &self,
        name: &str,
        from: Option<&str>,
    ) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build create branch request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut branch_req = req.init_create_branch();
            branch_req.set_repo_id(&self.repo_id.to_string());
            branch_req.set_branch_name(name);
            if let Some(start_point) = from {
                branch_req.set_start_point(start_point);
            }

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn checkout(&self, ref_spec: &str) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut checkout_req = req.init_checkout();
            checkout_req.set_repo_id(&self.repo_id.to_string());
            checkout_req.set_ref_name(ref_spec);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn default_branch(&self) -> Result<String, RegistryServiceError> {
        // Default branch lookup not yet supported via ZMQ
        // Return "main" as default
        Ok("main".to_string())
    }

    async fn list_branches(&self) -> Result<Vec<String>, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);
        client
            .list_branches(&self.repo_id.to_string())
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn stage_all(&self) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build stageAll request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_stage_all(&self.repo_id.to_string());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn commit(&self, message: &str) -> Result<String, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build commit request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut msg_builder = Builder::new_default();
            let mut req = msg_builder.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut commit_req = req.init_commit();
            commit_req.set_repo_id(&self.repo_id.to_string());
            commit_req.set_message(message);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &msg_builder)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::CommitOid(oid)) => {
                let oid = oid.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Ok(oid.to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_string())
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn remove_worktree(&self, path: &Path) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build removeWorktree request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut wt_req = req.init_remove_worktree();
            wt_req.set_repo_id(&self.repo_id.to_string());
            wt_req.set_worktree_path(&path.to_string_lossy());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn stage_files(&self, files: &[&str]) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build stageFiles request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut stage_req = req.init_stage_files();
            stage_req.set_repo_id(&self.repo_id.to_string());
            let mut files_list = stage_req.init_files(files.len() as u32);
            for (i, file) in files.iter().enumerate() {
                files_list.set(i as u32, file);
            }

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn get_head(&self) -> Result<String, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build getHead request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_get_head(&self.repo_id.to_string());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::RefOid(oid)) => {
                let oid = oid.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Ok(oid.to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_string())
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn get_ref(&self, ref_name: &str) -> Result<String, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build getRef request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut ref_req = req.init_get_ref();
            ref_req.set_repo_id(&self.repo_id.to_string());
            ref_req.set_ref_name(ref_name);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::RefOid(oid)) => {
                let oid = oid.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Ok(oid.to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_string())
            }
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn update(&self, refspec: Option<&str>) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build update request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut update_req = req.init_update();
            update_req.set_repo_id(&self.repo_id.to_string());
            if let Some(r) = refspec {
                update_req.set_refspec(r);
            }

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn list_remotes(&self) -> Result<Vec<RemoteInfo>, RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build listRemotes request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_list_remotes(&self.repo_id.to_string());

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        self.parse_remotes_response(&response)
    }

    async fn add_remote(&self, name: &str, url: &str) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build addRemote request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut remote_req = req.init_add_remote();
            remote_req.set_repo_id(&self.repo_id.to_string());
            remote_req.set_name(name);
            remote_req.set_url(url);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn remove_remote(&self, name: &str) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build removeRemote request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut remote_req = req.init_remove_remote();
            remote_req.set_repo_id(&self.repo_id.to_string());
            remote_req.set_name(name);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn set_remote_url(&self, name: &str, url: &str) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build setRemoteUrl request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut remote_req = req.init_set_remote_url();
            remote_req.set_repo_id(&self.repo_id.to_string());
            remote_req.set_name(name);
            remote_req.set_url(url);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }

    async fn rename_remote(
        &self,
        old_name: &str,
        new_name: &str,
    ) -> Result<(), RegistryServiceError> {
        let client = RegistryZmqClient::with_endpoint(&self.endpoint);

        // Build renameRemote request - scope builder to drop before await
        let bytes = {
            let id = client.next_id();
            let mut message = Builder::new_default();
            let mut req = message.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);

            let mut remote_req = req.init_rename_remote();
            remote_req.set_repo_id(&self.repo_id.to_string());
            remote_req.set_old_name(old_name);
            remote_req.set_new_name(new_name);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)
                .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
            bytes
        };

        let response = client.client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        // Parse response
        let reader = serialize::read_message(&*response, ReaderOptions::new())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let resp = reader.get_root::<registry_capnp::registry_response::Reader>()
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        use registry_capnp::registry_response::Which;
        match resp.which() {
            Ok(Which::Success(())) => Ok(()),
            Ok(Which::Error(err)) => {
                let err = err.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                let msg = err.get_message()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    .to_str()
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
                Err(RegistryServiceError::transport(msg))
            }
            _ => Err(RegistryServiceError::transport("Unexpected response type")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_registry_service_health_check() {
        let temp_dir = TempDir::new().unwrap();

        // Start the service
        let handle = RegistryService::start_at(temp_dir.path(), "inproc://test-registry")
            .await
            .unwrap();

        // Give it time to bind
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Test health check
        let client = RegistryZmqClient::with_endpoint("inproc://test-registry");
        let health = client.health_check().await.unwrap();

        assert_eq!(health.status, "healthy");

        // Stop the service
        handle.stop().await;
    }
}
