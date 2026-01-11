//! ZMQ-based registry service for repository management
//!
//! This service wraps git2db and provides a ZMQ REQ/REP interface for
//! repository operations. It uses Cap'n Proto for serialization.

use crate::auth::Operation;
use crate::services::PolicyZmqClient;
use crate::registry_capnp;
use crate::services::rpc_types::{HealthStatus, RemoteInfo, RegistryResponse, WorktreeData};
use crate::services::{EnvelopeContext, ServiceRunner, ZmqClient, ZmqService};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::{serialize_message, FromCapnp};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use git2db::{Git2DB, RepoId, TrackedRepository, GitRef};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::debug;
use uuid::Uuid;

/// Service name for endpoint registry
const SERVICE_NAME: &str = "registry";

// ============================================================================
// Extension Trait for Registry Operations on ZmqClient
// ============================================================================

/// Registry operations extension trait for `ZmqClient`.
///
/// Provides registry-specific methods when `ZmqClient` is connected to a registry endpoint.
/// All requests are automatically signed via `ZmqClient::call()`.
///
/// # Example
///
/// ```ignore
/// use crate::services::{ZmqClient, RegistryOps};
///
/// let client = ZmqClient::new(REGISTRY_ENDPOINT, signing_key, identity);
/// let repos = client.list().await?;
/// ```
#[async_trait]
pub trait RegistryOps {
    /// List all repositories.
    async fn list(&self) -> Result<Vec<TrackedRepository>>;

    /// Get repository by ID.
    async fn get(&self, repo_id: &str) -> Result<Option<TrackedRepository>>;

    /// Get repository by name.
    async fn get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>>;

    /// List branches for a repository.
    async fn list_branches(&self, repo_id: &str) -> Result<Vec<String>>;

    /// Health check.
    async fn health_check(&self) -> Result<HealthStatus>;

    /// Clone a repository.
    async fn clone_repo(&self, url: &str, name: Option<&str>) -> Result<RepoId>;

    /// Register an existing repository.
    async fn register_repo(&self, name: Option<&str>, path: &Path) -> Result<()>;

    /// Remove a repository.
    async fn remove(&self, id: &RepoId) -> Result<()>;
}

#[async_trait]
impl RegistryOps for ZmqClient {
    async fn list(&self) -> Result<Vec<TrackedRepository>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_list(());
        })?;
        let response = self.call(payload).await?;
        parse_repositories_response(&response)
    }

    async fn get(&self, repo_id: &str) -> Result<Option<TrackedRepository>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_get(repo_id);
        })?;
        let response = self.call(payload).await?;
        parse_optional_repository_response(&response)
    }

    async fn get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_get_by_name(name);
        })?;
        let response = self.call(payload).await?;
        parse_optional_repository_response(&response)
    }

    async fn list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_list_branches(repo_id);
        })?;
        let response = self.call(payload).await?;
        parse_branches_response(&response)
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            req.set_health_check(());
        })?;
        let response = self.call(payload).await?;
        parse_health_response(&response)
    }

    async fn clone_repo(&self, url: &str, name: Option<&str>) -> Result<RepoId> {
        let id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            let mut clone_req = req.init_clone();
            clone_req.set_url(url);
            if let Some(n) = name {
                clone_req.set_name(n);
            }
            clone_req.set_shallow(false);
            clone_req.set_depth(0);
        })?;
        let response = self.call(payload).await?;
        parse_repo_id_response(&response)
    }

    async fn register_repo(&self, name: Option<&str>, path: &Path) -> Result<()> {
        let id = self.next_id();
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid path encoding"))?;

        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            let mut reg_req = req.init_register();
            reg_req.set_path(path_str);
            if let Some(n) = name {
                reg_req.set_name(n);
            }
        })?;
        let response = self.call(payload).await?;
        parse_success_response(&response)
    }

    async fn remove(&self, id: &RepoId) -> Result<()> {
        let req_id = self.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(req_id);
            req.set_remove(&id.to_string());
        })?;
        let response = self.call(payload).await?;
        parse_success_response(&response)
    }
}

// ============================================================================
// Parsing Helper Functions
// ============================================================================

/// Generic response parser that handles deserialization and error extraction.
///
/// This reduces boilerplate by handling the common pattern:
/// 1. Deserialize Cap'n Proto message
/// 2. Get root RegistryResponse
/// 3. Handle Error variant uniformly
/// 4. Call extractor for specific variant
fn parse_response<T, F>(response: &[u8], extractor: F) -> Result<T>
where
    F: FnOnce(registry_capnp::registry_response::Reader) -> Result<T>,
{
    let reader = serialize::read_message(response, ReaderOptions::new())?;
    let resp = reader.get_root::<registry_capnp::registry_response::Reader>()?;

    use registry_capnp::registry_response::Which;
    if let Which::Error(err) = resp.which()? {
        let err = err?;
        return Err(anyhow!("{}", err.get_message()?.to_str()?));
    }

    extractor(resp)
}

/// Parse a repositories list response.
fn parse_repositories_response(response: &[u8]) -> Result<Vec<TrackedRepository>> {
    parse_response(response, |resp| {
        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Repositories(repos) => {
                let repos = repos?;
                repos.iter().map(|r| parse_tracked_repository(r)).collect()
            }
            _ => Err(anyhow!("Expected repositories response")),
        }
    })
}

/// Parse an optional repository response.
fn parse_optional_repository_response(response: &[u8]) -> Result<Option<TrackedRepository>> {
    parse_response(response, |resp| {
        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Repository(repo) => Ok(Some(parse_tracked_repository(repo?)?)),
            _ => Ok(None),
        }
    })
}

/// Parse a branches list response.
fn parse_branches_response(response: &[u8]) -> Result<Vec<String>> {
    parse_response(response, |resp| {
        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Branches(branches) => {
                let branches = branches?;
                branches.iter().map(|b| Ok(b?.to_str()?.to_string())).collect()
            }
            _ => Err(anyhow!("Expected branches response")),
        }
    })
}

/// Parse a health response.
fn parse_health_response(response: &[u8]) -> Result<HealthStatus> {
    parse_response(response, |resp| {
        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Health(health) => HealthStatus::read_from(health?),
            _ => Err(anyhow!("Expected health response")),
        }
    })
}

/// Parse a repo ID response.
fn parse_repo_id_response(response: &[u8]) -> Result<RepoId> {
    parse_response(response, |resp| {
        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Repository(repo) => {
                let repo = repo?;
                let id_str = repo.get_id()?.to_str()?;
                let uuid = Uuid::parse_str(id_str)?;
                Ok(RepoId::from_uuid(uuid))
            }
            _ => Err(anyhow!("Expected repository response")),
        }
    })
}

/// Parse a success response.
fn parse_success_response(response: &[u8]) -> Result<()> {
    parse_response(response, |resp| {
        use registry_capnp::registry_response::Which;
        match resp.which()? {
            Which::Success(()) => Ok(()),
            _ => Err(anyhow!("Expected success response")),
        }
    })
}

/// Parse a TrackedRepository from capnp reader.
fn parse_tracked_repository(
    repo: registry_capnp::tracked_repository::Reader,
) -> Result<TrackedRepository> {
    let id_str = repo.get_id()?.to_str()?;
    let id = RepoId::from_uuid(Uuid::parse_str(id_str)?);

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

// ============================================================================
// RegistryZmq - Wrapper implementing RegistryClient trait
// ============================================================================

/// Wrapper type that implements `RegistryClient` trait for `ZmqClient`.
///
/// This provides the bridge between the extension trait approach (`RegistryOps`)
/// and the trait-object compatible `RegistryClient` trait.
pub struct RegistryZmq {
    client: ZmqClient,
}

impl RegistryZmq {
    /// Create a new registry client with signing credentials.
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a registry client connected to a specific endpoint.
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        Self {
            client: ZmqClient::new(endpoint, signing_key, identity),
        }
    }

    /// Get the underlying client (for repository client creation).
    pub fn inner(&self) -> &ZmqClient {
        &self.client
    }
}

use crate::services::traits::{RegistryClient, RegistryServiceError, RepositoryClient, WorktreeInfo};

#[async_trait]
impl RegistryClient for RegistryZmq {
    async fn list(&self) -> Result<Vec<TrackedRepository>, RegistryServiceError> {
        RegistryOps::list(&self.client)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, RegistryServiceError> {
        RegistryOps::get(&self.client, &id.to_string())
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn get_by_name(
        &self,
        name: &str,
    ) -> Result<Option<TrackedRepository>, RegistryServiceError> {
        RegistryOps::get_by_name(&self.client, name)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn clone_repo(
        &self,
        url: &str,
        name: Option<&str>,
    ) -> Result<RepoId, RegistryServiceError> {
        RegistryOps::clone_repo(&self.client, url, name)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn register(
        &self,
        _id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), RegistryServiceError> {
        RegistryOps::register_repo(&self.client, name, path)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn upsert(&self, _name: &str, _url: &str) -> Result<RepoId, RegistryServiceError> {
        // Upsert operation not yet supported via ZMQ
        Err(RegistryServiceError::transport(
            "Upsert operation not yet supported via ZMQ",
        ))
    }

    async fn remove(&self, id: &RepoId) -> Result<(), RegistryServiceError> {
        RegistryOps::remove(&self.client, id)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn health_check(&self) -> Result<(), RegistryServiceError> {
        RegistryOps::health_check(&self.client)
            .await
            .map(|_| ())
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn repo(
        &self,
        name: &str,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError> {
        // Get repository by name first
        let repo = RegistryOps::get_by_name(&self.client, name)
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
            .ok_or_else(|| RegistryServiceError::transport(format!("Repository '{}' not found", name)))?;

        Ok(Arc::new(RepositoryZmqClient::new(
            self.client.endpoint().to_string(),
            repo.id,
            name.to_string(),
            self.client.signing_key().clone(),
            self.client.identity().clone(),
        )))
    }

    async fn repo_by_id(
        &self,
        id: &RepoId,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError> {
        // Get repository by ID first
        let repo = RegistryOps::get(&self.client, &id.to_string())
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
            .ok_or_else(|| RegistryServiceError::transport(format!("Repository '{}' not found", id)))?;

        let name = repo.name.unwrap_or_else(|| id.to_string());

        Ok(Arc::new(RepositoryZmqClient::new(
            self.client.endpoint().to_string(),
            id.clone(),
            name,
            self.client.signing_key().clone(),
            self.client.identity().clone(),
        )))
    }
}

// ============================================================================
// Registry Service (server-side)
// ============================================================================

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
    #[allow(dead_code)] // Future: base directory for relative path operations
    base_dir: PathBuf,
    /// Captured runtime handle for async operations in sync handlers
    runtime_handle: tokio::runtime::Handle,
    /// Policy client for authorization checks (uses ZMQ to PolicyService)
    policy_client: PolicyZmqClient,
}

impl RegistryService {
    /// Create a new registry service
    ///
    /// Must be called from within a tokio runtime context.
    pub async fn new(
        base_dir: impl AsRef<Path>,
        policy_client: PolicyZmqClient,
    ) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let registry = Git2DB::open(&base_dir).await?;
        // Capture the runtime handle to avoid nested runtime anti-pattern
        let runtime_handle = tokio::runtime::Handle::current();

        Ok(Self {
            registry: RwLock::new(registry),
            base_dir,
            runtime_handle,
            policy_client,
        })
    }

    /// Start the registry service on the default endpoint (from registry)
    pub async fn start(
        base_dir: impl AsRef<Path>,
        server_pubkey: hyprstream_rpc::VerifyingKey,
        policy_client: PolicyZmqClient,
    ) -> Result<crate::services::ServiceHandle> {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::start_at(base_dir, server_pubkey, policy_client, &endpoint).await
    }

    /// Start the registry service at a specific endpoint
    ///
    /// Waits for socket binding to complete before returning.
    pub async fn start_at(
        base_dir: impl AsRef<Path>,
        server_pubkey: hyprstream_rpc::VerifyingKey,
        policy_client: PolicyZmqClient,
        endpoint: &str,
    ) -> Result<crate::services::ServiceHandle> {
        let service = Self::new(base_dir, policy_client).await?;
        let runner = ServiceRunner::new(endpoint, server_pubkey);
        let handle = runner.run(service).await?;
        tracing::info!("RegistryService started at {}", endpoint);
        Ok(handle)
    }

    /// Check authorization for an operation.
    ///
    /// Returns the unauthorized response if the check fails, or None if authorized.
    /// Uses PolicyZmqClient for async policy checks over ZMQ.
    fn check_auth(
        &self,
        ctx: &EnvelopeContext,
        request_id: u64,
        resource: &str,
        operation: Operation,
    ) -> Option<Vec<u8>> {
        let subject = ctx.casbin_subject();
        // RegistryService runs on multi-threaded runtime, so block_in_place is safe
        let allowed = tokio::task::block_in_place(|| {
            self.runtime_handle.block_on(
                self.policy_client.check(&subject, resource, operation)
            )
        }).unwrap_or(false);

        if allowed {
            None // Authorized
        } else {
            debug!(
                "Authorization denied: {} cannot {} on {}",
                subject,
                operation.as_str(),
                resource
            );
            Some(RegistryResponse::unauthorized(
                request_id,
                &subject,
                resource,
                operation.as_str(),
            ))
        }
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
            let mut worktrees = handle.get_worktrees().await?;

            let mut result = Vec::with_capacity(worktrees.len());
            for wt in &mut worktrees {
                // Extract branch name from worktree path (last component)
                let branch_name = wt
                    .path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string());

                // Use WorktreeHandle::status() - single source of truth for dirty status
                let status = wt.status().await.ok();
                let is_dirty = status.as_ref().map(|s| !s.is_clean).unwrap_or(false);
                let head_oid = status
                    .and_then(|s| s.head.map(|h| h.to_string()))
                    .unwrap_or_default();

                result.push(WorktreeData {
                    path: wt.path().to_path_buf(),
                    branch_name,
                    head_oid,
                    is_locked: false,
                    is_dirty,
                });
            }
            Ok(result)
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
    /// Fetches updates from the remote repository. If a refspec is provided,
    /// only that refspec is fetched; otherwise, all refs are fetched from origin.
    fn handle_update(&self, repo_id: &str, refspec: Option<&str>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let refspec = refspec.map(|s| s.to_string());

        self.runtime_handle.block_on(async {
            let registry = self.registry.read()
                .map_err(|_| anyhow!("registry lock poisoned"))?;
            let handle = registry.repo(&id)?;

            // Get the repository path
            let repo_path = handle.worktree()?.to_path_buf();

            // Perform fetch in blocking task (git2 is not async)
            tokio::task::spawn_blocking(move || -> Result<()> {
                let repo = git2::Repository::open(&repo_path)
                    .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

                let mut remote = repo.find_remote("origin")
                    .map_err(|e| anyhow!("Failed to find remote 'origin': {}", e))?;

                // Fetch with optional refspec
                let refspecs: Vec<&str> = match &refspec {
                    Some(r) => vec![r.as_str()],
                    None => vec![],
                };

                remote.fetch(&refspecs, None, None)
                    .map_err(|e| anyhow!("Failed to fetch from origin: {}", e))?;

                Ok(())
            })
            .await
            .map_err(|e| anyhow!("Task join error: {}", e))??;

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

}

impl ZmqService for RegistryService {
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
        // Log identity for audit trail
        debug!(
            "Registry request from {} (envelope_id={}, authenticated={})",
            ctx.casbin_subject(),
            ctx.request_id,
            ctx.is_authenticated()
        );

        // Deserialize inner request from payload
        let reader = serialize::read_message(payload, ReaderOptions::new())?;
        let req = reader.get_root::<registry_capnp::registry_request::Reader>()?;

        let request_id = req.get_id();

        // Handle based on request type
        use registry_capnp::registry_request::Which;

        match req.which()? {
            Which::List(()) => {
                // Authorization: Query on registry
                if let Some(resp) = self.check_auth(ctx, request_id, "registry", Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_list() {
                    Ok(repos) => Ok(RegistryResponse::repositories(request_id, &repos)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Get(id) => {
                let id = id?;
                let id_str = id.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", id_str);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_get(id_str) {
                    Ok(Some(repo)) => Ok(RegistryResponse::repository(request_id, &repo)),
                    Ok(None) => Ok(RegistryResponse::error(request_id, "Repository not found")),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::GetByName(name) => {
                let name = name?;
                let name_str = name.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", name_str);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_get_by_name(name_str) {
                    Ok(Some(repo)) => Ok(RegistryResponse::repository(request_id, &repo)),
                    Ok(None) => Ok(RegistryResponse::error(request_id, "Repository not found")),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Clone(clone_req) => {
                // Authorization: Write on registry (creating new entry)
                if let Some(resp) = self.check_auth(ctx, request_id, "registry", Operation::Write) {
                    return Ok(resp);
                }
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
                    Ok(repo) => Ok(RegistryResponse::repository(request_id, &repo)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Register(reg_req) => {
                // Authorization: Write on registry (creating new entry)
                if let Some(resp) = self.check_auth(ctx, request_id, "registry", Operation::Write) {
                    return Ok(resp);
                }
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
                    Ok(repo) => Ok(RegistryResponse::repository(request_id, &repo)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Remove(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                // Authorization: Manage on specific registry entry (destructive)
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Manage) {
                    return Ok(resp);
                }
                match self.handle_remove(repo_id) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::ListBranches(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_list_branches(repo_id) {
                    Ok(branches) => Ok(RegistryResponse::branches(request_id, &branches)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::CreateBranch(branch_req) => {
                let branch_req = branch_req?;
                let repo_id = branch_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let branch_name = branch_req.get_branch_name()?.to_str()?;
                let start_point = if branch_req.has_start_point() {
                    Some(branch_req.get_start_point()?.to_str()?)
                } else {
                    None
                };

                match self.handle_create_branch(repo_id, branch_name, start_point) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Checkout(checkout_req) => {
                let checkout_req = checkout_req?;
                let repo_id = checkout_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let ref_name = checkout_req.get_ref_name()?.to_str()?;

                match self.handle_checkout(repo_id, ref_name) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::StageAll(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                match self.handle_stage_all(repo_id) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Commit(commit_req) => {
                let commit_req = commit_req?;
                let repo_id = commit_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let message = commit_req.get_message()?.to_str()?;

                match self.handle_commit(repo_id, message) {
                    Ok(oid) => Ok(RegistryResponse::commit_oid(request_id, &oid)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::HealthCheck(()) => {
                // Health check is public (no authorization required)
                match self.handle_health_check() {
                    Ok(status) => Ok(RegistryResponse::health(request_id, &status)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::ListWorktrees(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_list_worktrees(repo_id) {
                    Ok(worktrees) => Ok(RegistryResponse::worktrees(request_id, &worktrees)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::CreateWorktree(wt_req) => {
                let wt_req = wt_req?;
                let repo_id = wt_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let path = wt_req.get_path()?.to_str()?;
                let branch = wt_req.get_branch_name()?.to_str()?;

                match self.handle_create_worktree(repo_id, path, branch) {
                    Ok(wt_path) => Ok(RegistryResponse::path(request_id, &wt_path)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::RemoveWorktree(wt_req) => {
                let wt_req = wt_req?;
                let repo_id = wt_req.get_repo_id()?.to_str()?;
                // Authorization: Manage on specific registry entry (destructive)
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Manage) {
                    return Ok(resp);
                }
                let worktree_path = wt_req.get_worktree_path()?.to_str()?;

                match self.handle_remove_worktree(repo_id, worktree_path) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Update(update_req) => {
                let update_req = update_req?;
                let repo_id = update_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let refspec = if update_req.has_refspec() {
                    let r = update_req.get_refspec()?.to_str()?;
                    if r.is_empty() { None } else { Some(r) }
                } else {
                    None
                };

                match self.handle_update(repo_id, refspec) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::StageFiles(stage_req) => {
                let stage_req = stage_req?;
                let repo_id = stage_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let files_reader = stage_req.get_files()?;
                let mut files = Vec::new();
                for file in files_reader.iter() {
                    files.push(file?.to_str()?.to_string());
                }

                match self.handle_stage_files(repo_id, files) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::GetHead(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_get_head(repo_id) {
                    Ok(oid) => Ok(RegistryResponse::ref_oid(request_id, &oid)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::GetRef(ref_req) => {
                let ref_req = ref_req?;
                let repo_id = ref_req.get_repo_id()?.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                let ref_name = ref_req.get_ref_name()?.to_str()?;

                match self.handle_get_ref(repo_id, ref_name) {
                    Ok(oid) => Ok(RegistryResponse::ref_oid(request_id, &oid)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::ListRemotes(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                // Authorization: Query on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_list_remotes(repo_id) {
                    Ok(remotes) => Ok(RegistryResponse::remotes(request_id, &remotes)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::AddRemote(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let name = remote_req.get_name()?.to_str()?;
                let url = remote_req.get_url()?.to_str()?;

                match self.handle_add_remote(repo_id, name, url) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::RemoveRemote(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let name = remote_req.get_name()?.to_str()?;

                match self.handle_remove_remote(repo_id, name) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::SetRemoteUrl(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let name = remote_req.get_name()?.to_str()?;
                let url = remote_req.get_url()?.to_str()?;

                match self.handle_set_remote_url(repo_id, name, url) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::RenameRemote(remote_req) => {
                let remote_req = remote_req?;
                let repo_id = remote_req.get_repo_id()?.to_str()?;
                // Authorization: Write on specific registry entry
                let resource = format!("registry:{}", repo_id);
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                    return Ok(resp);
                }
                let old_name = remote_req.get_old_name()?.to_str()?;
                let new_name = remote_req.get_new_name()?.to_str()?;

                match self.handle_rename_remote(repo_id, old_name, new_name) {
                    Ok(()) => Ok(RegistryResponse::success(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }
        }
    }

    fn name(&self) -> &str {
        "registry"
    }
}

// ============================================================================
// MetricsRegistryClient Implementation
// ============================================================================

use hyprstream_metrics::checkpoint::manager::{
    RegistryClient as MetricsRegistryClient, RegistryError as MetricsRegistryError,
};

#[async_trait]
impl MetricsRegistryClient for RegistryZmq {
    async fn get_by_name(
        &self,
        name: &str,
    ) -> Result<Option<TrackedRepository>, MetricsRegistryError> {
        RegistryOps::get_by_name(&self.client, name)
            .await
            .map_err(|e| MetricsRegistryError::Operation(e.to_string()))
    }

    async fn register(
        &self,
        id: &RepoId,
        name: Option<&str>,
        path: &std::path::Path,
    ) -> Result<(), MetricsRegistryError> {
        RegistryClient::register(self, id, name, path)
            .await
            .map_err(|e| MetricsRegistryError::Operation(e.to_string()))
    }
}

// Type alias for backwards compatibility
pub type RegistryZmqClient = RegistryZmq;

// ============================================================================
// Repository-Level Client
// ============================================================================

/// ZMQ client for repository-level operations
///
/// Provides operations on a specific repository through the registry service.
pub struct RepositoryZmqClient {
    endpoint: String,
    repo_id: RepoId,
    repo_name: String,
    signing_key: SigningKey,
    identity: RequestIdentity,
}

impl RepositoryZmqClient {
    /// Create a new repository client with signing credentials
    pub fn new(
        endpoint: String,
        repo_id: RepoId,
        repo_name: String,
        signing_key: SigningKey,
        identity: RequestIdentity,
    ) -> Self {
        Self {
            endpoint,
            repo_id,
            repo_name,
            signing_key,
            identity,
        }
    }

    /// Create a ZmqClient with this client's credentials for raw RPC calls.
    fn create_inner_client(&self) -> ZmqClient {
        ZmqClient::new(&self.endpoint, self.signing_key.clone(), self.identity.clone())
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
                        is_dirty: wt.get_is_dirty(),
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
                    result.push(
                        RemoteInfo::read_from(remote)
                            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
                    );
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();
        client
            .list_branches(&self.repo_id.to_string())
            .await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn stage_all(&self) -> Result<(), RegistryServiceError> {
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        self.parse_remotes_response(&response)
    }

    async fn add_remote(&self, name: &str, url: &str) -> Result<(), RegistryServiceError> {
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
        let client = self.create_inner_client();

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

        let response = client.call(bytes).await
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
    use crate::auth::PolicyManager;
    use crate::services::{PolicyService, PolicyZmqClient};
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_registry_service_health_check() {
        let temp_dir = TempDir::new().expect("test: create temp dir");

        // Generate keypair for signing/verification
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Create a permissive policy manager and start PolicyService first
        let policy_manager = Arc::new(PolicyManager::permissive().await.expect("test: create policy manager"));
        let _policy_handle = PolicyService::start_at(
            policy_manager,
            verifying_key,
            "inproc://test-policy",
        )
        .await
        .expect("test: start policy service");

        // Create policy client for RegistryService
        let policy_client = PolicyZmqClient::with_endpoint(
            "inproc://test-policy",
            signing_key.clone(),
            RequestIdentity::local(),
        );

        // Start the registry service with policy client
        let mut handle = RegistryService::start_at(
            temp_dir.path(),
            verifying_key,
            policy_client,
            "inproc://test-registry",
        )
        .await
        .expect("test: start registry service");

        // Create signed client with matching key and local identity
        let client = RegistryZmqClient::with_endpoint(
            "inproc://test-registry",
            signing_key,
            RequestIdentity::local(),
        );
        // health_check returns () on success
        let result = client.health_check().await;
        assert!(result.is_ok(), "health_check should succeed: {:?}", result.err());

        // Stop the service
        handle.stop().await;
    }
}
