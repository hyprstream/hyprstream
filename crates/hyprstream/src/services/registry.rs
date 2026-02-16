//! ZMQ-based registry service for repository management
//!
//! This service wraps git2db and provides a ZMQ REQ/REP interface for
//! repository operations. It uses Cap'n Proto for serialization.

use crate::auth::Operation;
use crate::services::PolicyClient;
use crate::services::types::{MAX_FDS_GLOBAL, MAX_FDS_PER_CLIENT, MAX_FS_IO_SIZE};
use crate::services::contained_root::{self, ContainedRoot, FsServiceError};
use crate::services::{EnvelopeContext, ZmqService};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as endpoint_registry, SocketKind};
use hyprstream_rpc::{StreamChannel, StreamContext};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use git2db::{CloneBuilder, Git2DB, GitRef, RepoId, TrackedRepository};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{Read as _, Write as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, warn};
use uuid::Uuid;

/// Service name for endpoint registry
const SERVICE_NAME: &str = "registry";

// Generated client types
use crate::services::generated::registry_client::{
    RegistryClient as GenRegistryClient, RegistryResponseVariant,
    RegistryHandler, RepoHandler, WorktreeHandler, dispatch_registry,
    StreamInfo, ErrorInfo, HealthStatus, DetailedStatusInfo, RemoteInfo,
    CloneRequest, RegisterRequest,
    CreateWorktreeRequest, RemoveWorktreeRequest,
    BranchRequest, CheckoutRequest, StageFilesRequest,
    CommitRequest, MergeRequest, ContinueMergeRequest,
    GetRefRequest, AddRemoteRequest, RemoveRemoteRequest,
    SetRemoteUrlRequest, RenameRemoteRequest,
    PushRequest, AmendCommitRequest, CommitWithAuthorRequest,
    CreateTagRequest, DeleteTagRequest, UpdateRequest,
    FsOpenRequest, FsOpenResponse, FsCloseRequest,
    FsReadRequest, FsReadResponse,
    FsWriteRequest, FsWriteResponse,
    FsPreadRequest, FsPwriteRequest, FsSeekRequest, FsSeekResponse,
    FsTruncateRequest, FsSyncRequest, FsPathRequest, FsMkdirRequest,
    FsRenameRequest, FsCopyRequest, FsWriteFileRequest,
    FsStatResponse, FsDirEntryInfo, FsStreamInfoResponse,
    SeekWhenceEnum, EnsureWorktreeRequest,
};
// Conflicting names — use canonical path at usage sites:
//   registry_client::TrackedRepository, registry_client::RepositoryStatus, registry_client::WorktreeInfo

// ============================================================================
// Parsing Helper Functions
// ============================================================================

/// Parse a stream started response — uses generated response parser
fn parse_stream_started_response(response: &[u8]) -> Result<crate::services::rpc_types::StreamInfo> {
    match GenRegistryClient::parse_response(response)? {
        RegistryResponseVariant::CloneStreamResult(data) => Ok(crate::services::rpc_types::StreamInfo {
            stream_id: data.stream_id,
            endpoint: data.endpoint,
            server_pubkey: data.server_pubkey,
        }),
        RegistryResponseVariant::Error(ref e) => Err(anyhow!("{}", e.message)),
        _ => Err(anyhow!("Expected clone_stream_result response")),
    }
}

/// Convert generated variant fields into a TrackedRepository.
fn variant_to_tracked_repository(
    id: &str,
    name: &str,
    url: &str,
    worktree_path: &str,
    tracking_ref: &str,
    current_oid: &str,
    registered_at: i64,
) -> Result<TrackedRepository> {
    let uuid = Uuid::parse_str(id)?;
    let repo_id = RepoId::from_uuid(uuid);
    let name_opt = if name.is_empty() { None } else { Some(name.to_owned()) };
    let tracking = if tracking_ref.is_empty() {
        GitRef::Branch("main".to_owned())
    } else {
        GitRef::Branch(tracking_ref.to_owned())
    };
    let oid = if current_oid.is_empty() { None } else { Some(current_oid.to_owned()) };

    Ok(TrackedRepository {
        id: repo_id,
        name: name_opt,
        url: url.to_owned(),
        worktree_path: PathBuf::from(worktree_path),
        tracking_ref: tracking,
        remotes: Vec::new(),
        registered_at,
        current_oid: oid,
        metadata: HashMap::new(),
    })
}


// ============================================================================
// Registry Service (server-side)
// ============================================================================

// ============================================================================
// File Descriptor Table for Filesystem Operations
// ============================================================================

/// An open file in the FD table.
struct OpenFile {
    /// The underlying file handle, mutex-protected for concurrent access.
    file: Mutex<std::fs::File>,
    /// Identity of the client that opened this FD (for owner verification).
    owner_identity: String,
    /// Whether the file was opened for writing.
    writable: bool,
    /// Epoch seconds of last access (for idle timeout reaping).
    last_accessed: AtomicU64,
}

/// Process-global file descriptor table for filesystem operations.
///
/// Uses `DashMap` for lock-free concurrent access. FDs are allocated with
/// a simple atomic counter with collision retry.
struct FdTable {
    next_fd: AtomicU32,
    fds: DashMap<u32, OpenFile>,
    /// Per-client FD count for resource limiting.
    client_fd_counts: DashMap<String, AtomicU32>,
}

/// Idle timeout for file descriptors (5 minutes).
const FD_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl FdTable {
    fn new() -> Self {
        Self {
            next_fd: AtomicU32::new(3), // Skip stdin/stdout/stderr
            fds: DashMap::new(),
            client_fd_counts: DashMap::new(),
        }
    }

    /// Allocate a new FD for a client, checking per-client and global limits.
    fn alloc_fd(&self, client_id: &str) -> Result<u32, FsServiceError> {
        // Check per-client limit
        let count = self
            .client_fd_counts
            .entry(client_id.to_owned())
            .or_insert_with(|| AtomicU32::new(0));
        if count.load(Relaxed) >= MAX_FDS_PER_CLIENT {
            return Err(FsServiceError::ResourceLimit(
                "too many open files for client".into(),
            ));
        }
        // Check global limit
        if self.fds.len() >= MAX_FDS_GLOBAL as usize {
            return Err(FsServiceError::ResourceLimit(
                "too many open files globally".into(),
            ));
        }
        // Allocate with collision retry
        for _ in 0..1000 {
            let fd = self.next_fd.fetch_add(1, Relaxed);
            if fd >= 3 && !self.fds.contains_key(&fd) {
                count.fetch_add(1, Relaxed);
                return Ok(fd);
            }
        }
        Err(FsServiceError::ResourceLimit(
            "failed to allocate FD after retries".into(),
        ))
    }

    /// Insert an open file into the table.
    fn insert(&self, fd: u32, file: OpenFile) {
        self.fds.insert(fd, file);
    }

    /// Remove an FD and decrement the client's count.
    fn remove(&self, fd: u32, client_id: &str) -> Option<OpenFile> {
        let removed = self.fds.remove(&fd).map(|(_, v)| v);
        if removed.is_some() {
            if let Some(count) = self.client_fd_counts.get(client_id) {
                count.fetch_sub(1, Relaxed);
            }
        }
        removed
    }

    /// Get a reference to an open file, verifying ownership.
    fn get_verified(
        &self,
        fd: u32,
        client_id: &str,
    ) -> Result<dashmap::mapref::one::Ref<'_, u32, OpenFile>, FsServiceError> {
        let entry = self
            .fds
            .get(&fd)
            .ok_or(FsServiceError::BadFd(fd))?;
        if entry.owner_identity != client_id {
            return Err(FsServiceError::PermissionDenied(
                "FD not owned by caller".into(),
            ));
        }
        entry.last_accessed.store(now_epoch_secs(), Relaxed);
        Ok(entry)
    }
}

/// ZMQ-based registry service
///
/// Wraps git2db::Git2DB and provides a Cap'n Proto interface over ZMQ.
///
/// ## Streaming Support
///
/// Streaming clone uses the Continuation pipeline (same as inference streaming):
/// 1. Handler performs DH key exchange and returns (StreamInfo, Continuation)
/// 2. Dispatch serializes StreamInfo as the REP response
/// 3. RequestLoop spawns the Continuation after REP is sent
/// 4. Continuation publishes clone progress via PUB/SUB
///
/// The registry is wrapped in RwLock for interior mutability since some operations
/// (like clone) require mutable access but ZmqService::handle_request takes &self.
pub struct RegistryService {
    // Business logic
    registry: Arc<RwLock<Git2DB>>,
    #[allow(dead_code)] // Future: base directory for relative path operations
    base_dir: PathBuf,
    /// Policy client for authorization checks (uses ZMQ to PolicyService)
    policy_client: PolicyClient,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
    /// File descriptor table for FS operations.
    fd_table: Arc<FdTable>,
    /// Cached contained roots for worktrees: (repo_id, worktree_name) → ContainedRoot.
    contained_roots: DashMap<(String, String), Arc<dyn ContainedRoot>>,
}

/// Progress reporter that sends updates via a tokio mpsc channel.
///
/// Implements `git2db::callback_config::ProgressReporter` to bridge git2db's
/// progress callbacks to hyprstream's stream publishing system.
///
/// Uses `blocking_send` since this is called from a sync context (spawn_blocking).
struct CloneProgressReporter {
    sender: tokio::sync::mpsc::Sender<hyprstream_rpc::streaming::ProgressUpdate>,
}

impl CloneProgressReporter {
    fn new(sender: tokio::sync::mpsc::Sender<hyprstream_rpc::streaming::ProgressUpdate>) -> Self {
        Self { sender }
    }
}

impl git2db::callback_config::ProgressReporter for CloneProgressReporter {
    fn report(&self, stage: &str, current: usize, total: usize) {
        // Use blocking_send since we're in a sync context (spawn_blocking)
        // Log if channel is full instead of silently dropping
        if let Err(e) = self.sender.blocking_send(hyprstream_rpc::streaming::ProgressUpdate::Progress {
            stage: stage.to_owned(),
            current,
            total,
        }) {
            tracing::trace!("Progress channel full, dropping update: {}", e);
        }
    }
}

impl RegistryService {
    /// Create a new registry service with infrastructure
    ///
    /// Must be called from within a tokio runtime context.
    pub async fn new(
        base_dir: impl AsRef<Path>,
        policy_client: PolicyClient,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let registry = Git2DB::open(&base_dir).await?;

        let worker_registry = Arc::new(RwLock::new(registry));

        // Create FD table and spawn reaper
        let fd_table = Arc::new(FdTable::new());
        let reaper_fd_table = Arc::clone(&fd_table);
        tokio::spawn(async move {
            Self::fd_reaper(reaper_fd_table).await;
        });

        let service = Self {
            registry: worker_registry,
            base_dir,
            policy_client,
            context,
            transport,
            signing_key,
            fd_table,
            contained_roots: DashMap::new(),
        };

        Ok(service)
    }

    /// Background task that reaps idle file descriptors.
    async fn fd_reaper(fd_table: Arc<FdTable>) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let now = now_epoch_secs();
            fd_table.fds.retain(|_fd, entry| {
                let idle = now.saturating_sub(entry.last_accessed.load(Relaxed));
                if idle > FD_IDLE_TIMEOUT.as_secs() {
                    // Decrement client FD count
                    if let Some(count) = fd_table.client_fd_counts.get(&entry.owner_identity) {
                        count.fetch_sub(1, Relaxed);
                    }
                    debug!("Reaped idle FD (idle {}s)", idle);
                    false // remove
                } else {
                    true // keep
                }
            });
        }
    }

    /// Execute a streaming clone with real-time progress via Continuation.
    ///
    /// Uses git2db's callback_config to receive progress updates during clone,
    /// which are forwarded to the client via StreamChannel in real-time.
    /// Called as a Continuation after the REP response is sent.
    async fn execute_clone_stream(
        stream_channel: StreamChannel,
        registry: Arc<RwLock<Git2DB>>,
        stream_ctx: StreamContext,
        url: String,
        name: Option<String>,
        shallow: bool,
        depth: Option<u32>,
        branch: Option<String>,
    ) {
        use hyprstream_rpc::streaming::ProgressUpdate;

        debug!(
            stream_id = %stream_ctx.stream_id(),
            url = %url,
            "Starting streaming clone with progress reporting"
        );

        // Create tokio channel for receiving updates from git2db
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::channel::<ProgressUpdate>(100);

        // Create reporter that implements git2db::ProgressReporter
        let reporter = Arc::new(CloneProgressReporter::new(progress_tx.clone()));

        // Build callback config with progress reporter
        let callback_config = git2db::callback_config::CallbackConfig::new()
            .with_progress(git2db::callback_config::ProgressConfig::Channel(reporter));

        // Execute clone and stream progress concurrently
        let result = stream_channel.with_publisher(&stream_ctx, |mut publisher| async move {
            // Spawn clone task - runs concurrently with progress streaming
            let registry_clone = Arc::clone(&registry);

            let clone_handle = tokio::spawn(async move {
                // CloneBuilder manages locks internally for optimal performance
                // (read lock for config, no lock during network I/O, write lock for registration)
                Self::clone_repo_inner(
                    registry_clone,
                    &url,
                    name.as_deref(),
                    shallow,
                    depth,
                    branch.as_deref(),
                    Some(callback_config),
                ).await
            });

            // Drop sender after spawning so receiver knows when clone finishes
            drop(progress_tx);

            // Stream progress updates in real-time as they arrive
            // (Ignore Complete/Error from channel - we'll send our own based on clone_result)
            while let Some(update) = progress_rx.recv().await {
                if let ProgressUpdate::Progress { stage, current, total } = update {
                    publisher.publish_progress(&stage, current, total).await?;
                }
            }

            // Wait for clone to complete and send final status
            match clone_handle.await {
                Ok(Ok(repo)) => {
                    let metadata = serde_json::json!({
                        "repo_id": repo.id.to_string(),
                        "name": repo.name,
                        "url": repo.url,
                    });
                    publisher.complete_ref(metadata.to_string().as_bytes()).await?;
                    Ok(())
                }
                Ok(Err(e)) => {
                    publisher.publish_error(&e.to_string()).await?;
                    Err(e)
                }
                Err(e) => {
                    let err = anyhow!("Clone task panicked: {}", e);
                    publisher.publish_error(&err.to_string()).await?;
                    Err(err)
                }
            }
        }).await;

        if let Err(e) = result {
            error!(
                stream_id = %stream_ctx.stream_id(),
                error = %e,
                "Clone stream failed"
            );
        }
    }

    /// Check if a request is authorized (returns bool for generated handler methods).
    async fn is_authorized(&self, ctx: &EnvelopeContext, resource: &str, operation: Operation) -> bool {
        let subject = ctx.subject();
        self.policy_client.check(&subject.to_string(), "*", resource, operation.as_str())
            .await
            .unwrap_or_else(|e| {
                warn!("Policy check failed for {} on {}: {} - denying access", subject, resource, e);
                false
            })
    }

    /// Parse a RepoId from string
    fn parse_repo_id(id_str: &str) -> Result<RepoId> {
        let uuid = Uuid::parse_str(id_str)
            .map_err(|e| anyhow!("invalid repo id '{}': {}", id_str, e))?;
        Ok(RepoId::from_uuid(uuid))
    }

    /// Internal clone logic shared by sync and streaming clone operations.
    ///
    /// Builds clone request, executes it, and returns the cloned repository.
    /// CloneBuilder manages locks internally for optimal performance.
    ///
    /// # Arguments
    /// * `callback_config` - Optional callback configuration for progress reporting
    async fn clone_repo_inner(
        registry: Arc<RwLock<Git2DB>>,
        url: &str,
        name: Option<&str>,
        shallow: bool,
        depth: Option<u32>,
        branch: Option<&str>,
        callback_config: Option<git2db::callback_config::CallbackConfig>,
    ) -> Result<TrackedRepository> {
        let mut clone_builder = CloneBuilder::new(Arc::clone(&registry), url);

        if let Some(n) = name {
            clone_builder = clone_builder.name(n);
        }

        // depth > 0 implies shallow clone
        // if shallow is explicitly set but depth is 0, use depth=1
        if let Some(d) = depth.filter(|&d| d > 0) {
            clone_builder = clone_builder.depth(d);
        } else if shallow {
            clone_builder = clone_builder.depth(1);
        }

        if let Some(b) = branch.filter(|b| !b.is_empty()) {
            clone_builder = clone_builder.branch(b);
        }

        // Add callback config for progress reporting if provided
        if let Some(config) = callback_config {
            clone_builder = clone_builder.callback_config(config);
        }

        let repo_id = clone_builder.exec().await?;

        // Get the tracked repository to return
        let registry_guard = registry.read().await;
        let result = registry_guard
            .list()
            .find(|r| r.id == repo_id)
            .cloned()
            .ok_or_else(|| anyhow!("Failed to find cloned repository"));
        drop(registry_guard);
        result
    }

    /// Handle a list repositories request
    async fn handle_list(&self) -> Result<Vec<TrackedRepository>> {
        let registry = self.registry.read().await;
        Ok(registry.list().cloned().collect())
    }

    /// Handle get repository by ID
    async fn handle_get(&self, repo_id: &str) -> Result<Option<TrackedRepository>> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let result = registry.list().find(|r| r.id == id).cloned();
        Ok(result)
    }

    /// Handle get repository by name
    async fn handle_get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>> {
        let registry = self.registry.read().await;
        let result = registry
            .list()
            .find(|r| r.name.as_ref() == Some(&name.to_owned()))
            .cloned();
        Ok(result)
    }

    /// Handle list branches
    async fn handle_list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let branches = handle.branch().list().await?;
        Ok(branches.into_iter().map(|b| b.name).collect())
    }

    /// Handle create branch
    async fn handle_create_branch(
        &self,
        repo_id: &str,
        branch_name: &str,
        start_point: Option<&str>,
    ) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.branch().create(branch_name, start_point).await?;
        Ok(())
    }

    /// Handle checkout
    async fn handle_checkout(&self, repo_id: &str, ref_name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.branch().checkout(ref_name).await?;
        Ok(())
    }

    /// Handle stage all
    async fn handle_stage_all(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.staging().add_all().await?;
        Ok(())
    }

    /// Handle commit
    async fn handle_commit(&self, repo_id: &str, message: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let oid = handle.commit(message).await?;
        Ok(oid.to_string())
    }

    /// Handle merge
    async fn handle_merge(&self, repo_id: &str, source: &str, message: Option<&str>) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let source = source.to_owned();
        let message = message.map(std::borrow::ToOwned::to_owned);
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let oid = handle.merge(&source, message.as_deref()).await?;
        Ok(oid.to_string())
    }

    /// Handle status
    async fn handle_status(&self, repo_id: &str) -> Result<git2db::RepositoryStatus> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use with_repo_blocking to properly resolve worktree path for bare repos
        self.with_repo_blocking(&id, |repo| {
            let head = repo.head().ok();
            let branch = head.as_ref().and_then(|h| h.shorthand().map(String::from));
            let head_oid = head.as_ref().and_then(git2::Reference::target);

            let statuses = repo.statuses(None)
                .map_err(|e| anyhow!("Failed to get statuses: {}", e))?;
            let is_clean = statuses.is_empty();
            let modified_files: Vec<std::path::PathBuf> = statuses
                .iter()
                .filter_map(|e| e.path().map(std::path::PathBuf::from))
                .collect();

            // Compute ahead/behind
            let (ahead, behind) = if let Some(ref head_ref) = repo.head().ok() {
                if let Some(branch_name) = head_ref.shorthand() {
                    let upstream_name = format!("origin/{}", branch_name);
                    if let Ok(upstream) = repo.revparse_single(&upstream_name) {
                        if let (Ok(local), Ok(remote)) = (
                            head_ref.peel_to_commit(),
                            upstream.peel_to_commit(),
                        ) {
                            repo.graph_ahead_behind(local.id(), remote.id())
                                .unwrap_or((0, 0))
                        } else { (0, 0) }
                    } else { (0, 0) }
                } else { (0, 0) }
            } else { (0, 0) };

            Ok(git2db::RepositoryStatus {
                branch,
                head: head_oid,
                ahead,
                behind,
                is_clean,
                modified_files,
            })
        }).await
    }

    /// Handle clone operation
    async fn handle_clone(
        &self,
        url: &str,
        name: Option<&str>,
        shallow: bool,
        depth: Option<u32>,
        branch: Option<&str>,
    ) -> Result<TrackedRepository> {
        Self::clone_repo_inner(
            Arc::clone(&self.registry),
            url,
            name,
            shallow,
            depth,
            branch,
            None,
        ).await
    }

    // ========================================================================
    // Streaming Clone Support
    // ========================================================================

    /// Prepare a streaming clone operation and return (StreamInfo, Continuation).
    ///
    /// Creates a StreamChannel for DH key exchange and pre-authorization.
    /// The continuation contains the clone work, executed by RequestLoop after REP is sent.
    async fn prepare_clone_stream(
        &self,
        url: &str,
        name: Option<&str>,
        shallow: bool,
        depth: Option<u32>,
        branch: Option<&str>,
        client_ephemeral_pubkey: Option<&[u8]>,
    ) -> Result<(StreamInfo, hyprstream_rpc::service::Continuation)> {
        // DH key derivation is required
        let client_pub_bytes = client_ephemeral_pubkey
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;

        // Create StreamChannel for DH key exchange and publishing
        let stream_channel = StreamChannel::new(
            Arc::clone(&self.context),
            self.signing_key.clone(),
        );

        // 10 minutes expiry for clone operations
        let stream_ctx = stream_channel.prepare_stream(client_pub_bytes, 600).await?;

        debug!(
            stream_id = %stream_ctx.stream_id(),
            topic = %stream_ctx.topic(),
            "Clone stream prepared (DH + pre-authorization via StreamChannel)"
        );

        let stream_endpoint = endpoint_registry()
            .endpoint("streams", SocketKind::Sub)
            .to_zmq_string();

        let stream_info = StreamInfo {
            stream_id: stream_ctx.stream_id().to_owned(),
            endpoint: stream_endpoint,
            server_pubkey: *stream_ctx.server_pubkey(),
        };

        // Build continuation that executes the clone and streams progress
        let registry = Arc::clone(&self.registry);
        let url = url.to_owned();
        let name = name.map(std::borrow::ToOwned::to_owned);
        let branch = branch.map(std::borrow::ToOwned::to_owned);

        let continuation: hyprstream_rpc::service::Continuation = Box::pin(async move {
            Self::execute_clone_stream(
                stream_channel,
                registry,
                stream_ctx,
                url,
                name,
                shallow,
                depth,
                branch,
            ).await;
        });

        Ok((stream_info, continuation))
    }

    /// Handle list worktrees
    async fn handle_list_worktrees(&self, repo_id: &str) -> Result<Vec<crate::services::generated::registry_client::WorktreeInfo>> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let mut worktrees = handle.get_worktrees().await?;

        let mut result = Vec::with_capacity(worktrees.len());
        for wt in &mut worktrees {
            // Extract branch name from worktree path (last component)
            let branch_name = wt
                .path()
                .file_name()
                .and_then(|s| s.to_str())
                .map(std::borrow::ToOwned::to_owned)
                .unwrap_or_default();

            // Use WorktreeHandle::status() - single source of truth for dirty status
            let status = wt.status().await.ok();
            let is_dirty = status.as_ref().map(|s| !s.is_clean).unwrap_or(false);
            let head_oid = status
                .and_then(|s| s.head.map(|h| h.to_string()))
                .unwrap_or_default();

            result.push(crate::services::generated::registry_client::WorktreeInfo {
                path: wt.path().to_string_lossy().to_string(),
                branch_name,
                head_oid,
                is_locked: false,
                is_dirty,
            });
        }
        Ok(result)
    }

    /// Handle create worktree
    async fn handle_create_worktree(
        &self,
        repo_id: &str,
        path: &str,
        branch: &str,
    ) -> Result<PathBuf> {
        let id = Self::parse_repo_id(repo_id)?;
        let path = PathBuf::from(path);
        let branch = branch.to_owned();
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let worktree = handle.create_worktree(&path, &branch).await?;
        Ok(worktree.path().to_path_buf())
    }

    /// Handle register operation
    #[allow(deprecated)]
    async fn handle_register(
        &self,
        path: &str,
        name: Option<&str>,
        _tracking_ref: Option<&str>,
    ) -> Result<TrackedRepository> {
        let path = PathBuf::from(path);
        let name = name.map(std::borrow::ToOwned::to_owned);
        // Note: tracking_ref is not yet used by register_repository

        // Register requires write lock
        let mut registry = self.registry.write().await;

        // Generate a new repo ID
        let repo_id = RepoId::new();

        // Derive URL from path (local file URL)
        let url = format!("file://{}", path.display());

        // TODO: Migrate to registry.register(repo_id) builder API
        registry.register_repository(&repo_id, name, url).await?;

        // Get the tracked repository to return
        let repo = registry
            .list()
            .find(|r| r.id == repo_id)
            .cloned()
            .ok_or_else(|| anyhow!("Failed to find registered repository"))?;

        Ok(repo)
    }

    /// Handle remove operation
    async fn handle_remove(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        // Remove requires write lock
        let mut registry = self.registry.write().await;
        registry.remove_repository(&id).await?;
        Ok(())
    }

    /// Handle remove worktree operation
    async fn handle_remove_worktree(&self, repo_id: &str, worktree_path: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let worktree_path = PathBuf::from(worktree_path);
        let registry = self.registry.read().await;
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
    }

    /// Handle stage files operation
    async fn handle_stage_files(&self, repo_id: &str, files: Vec<String>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;

        for file in files {
            handle.staging().add(&file).await?;
        }
        Ok(())
    }

    /// Handle get HEAD operation
    async fn handle_get_head(&self, repo_id: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;

        // Get HEAD ref (DefaultBranch resolves to HEAD)
        let oid = handle.resolve_git_ref(&GitRef::DefaultBranch).await?;
        Ok(oid.to_string())
    }

    /// Handle get ref operation
    async fn handle_get_ref(&self, repo_id: &str, ref_name: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let ref_name = ref_name.to_owned();
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;

        // Try to resolve as a revspec
        let oid = handle.resolve_revspec(&ref_name).await?;
        Ok(oid.to_string())
    }

    /// Resolve a worktree path from a repository handle.
    ///
    /// For bare repositories (cloned models), `handle.worktree()` returns the bare
    /// repo path (e.g. `models/name/name.git`), which can't be used for operations
    /// that require a working tree (status, staging, etc.). This method finds the
    /// actual worktree for the tracking ref.
    async fn resolve_worktree_path(handle: &git2db::RepositoryHandle<'_>) -> Result<std::path::PathBuf> {
        let base_path = handle.worktree()?.to_path_buf();

        // Detect bare repository: path ends in .git or has HEAD but no .git subdir
        let is_bare = base_path.extension().map_or(false, |ext| ext == "git")
            || (base_path.join("HEAD").exists() && !base_path.join(".git").exists());

        if !is_bare {
            return Ok(base_path);
        }

        // Try the tracking ref's worktree first
        let tracked = handle.metadata()?;
        let branch = match &tracked.tracking_ref {
            git2db::GitRef::Branch(b) => Some(b.as_str()),
            _ => None,
        };

        if let Some(branch) = branch {
            if let Ok(Some(wt)) = handle.get_worktree(branch).await {
                return Ok(wt.path().to_path_buf());
            }
        }

        // Fallback: try any available worktree
        if let Ok(worktrees) = handle.get_worktrees().await {
            if let Some(wt) = worktrees.into_iter().next() {
                return Ok(wt.path().to_path_buf());
            }
        }

        Err(anyhow!("No worktrees found for bare repository: {:?}", base_path))
    }

    /// Execute a blocking git2 operation on a repository.
    ///
    /// Handles the boilerplate of: read registry → get handle → resolve worktree →
    /// spawn_blocking → Repository::open → operation.
    async fn with_repo_blocking<F, T>(&self, id: &git2db::RepoId, f: F) -> Result<T>
    where
        F: FnOnce(git2::Repository) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let registry = self.registry.read().await;
        let handle = registry.repo(id)?;
        let repo_path = Self::resolve_worktree_path(&handle).await?;
        tokio::task::spawn_blocking(move || {
            let repo = git2::Repository::open(&repo_path)
                .map_err(|e| anyhow!("Failed to open repository: {}", e))?;
            f(repo)
        })
        .await
        .map_err(|e| anyhow!("Task join error: {}", e))?
    }

    /// Handle update operation (fetch from remote)
    async fn handle_update(&self, repo_id: &str, refspec: Option<&str>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let refspec = refspec.map(std::borrow::ToOwned::to_owned);
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::fetch(&repo, "origin", refspec.as_deref())
        }).await
    }

    /// Handle health check
    async fn handle_health_check(&self) -> Result<HealthStatus> {
        let registry = self.registry.read().await;
        let repo_count = registry.list().count() as u32;
        Ok(HealthStatus {
            status: "healthy".to_owned(),
            repository_count: repo_count,
            worktree_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Handle list remotes operation
    async fn handle_list_remotes(&self, repo_id: &str) -> Result<Vec<RemoteInfo>> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;

        let remotes = handle.remote().list().await?;
        let result = remotes
            .into_iter()
            .map(|remote| RemoteInfo {
                name: remote.name,
                url: remote.url,
                push_url: remote.push_url.unwrap_or_default(),
            })
            .collect();
        Ok(result)
    }

    /// Handle add remote operation
    async fn handle_add_remote(&self, repo_id: &str, name: &str, url: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        let url = url.to_owned();
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.remote().add(&name, &url).await?;
        Ok(())
    }

    /// Handle remove remote operation
    async fn handle_remove_remote(&self, repo_id: &str, name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.remote().remove(&name).await?;
        Ok(())
    }

    /// Handle set remote URL operation
    async fn handle_set_remote_url(&self, repo_id: &str, name: &str, url: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        let url = url.to_owned();
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.remote().set_url(&name, &url).await?;
        Ok(())
    }

    /// Handle rename remote operation
    async fn handle_rename_remote(&self, repo_id: &str, old_name: &str, new_name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let old_name = old_name.to_owned();
        let new_name = new_name.to_owned();
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        handle.remote().rename(&old_name, &new_name).await?;
        Ok(())
    }

    // ========================================================================
    // Push Operations
    // ========================================================================

    /// Handle push operation
    async fn handle_push(&self, repo_id: &str, remote: &str, refspec: &str, force: bool) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let remote = remote.to_owned();
        let refspec = refspec.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::push(&repo, &remote, &refspec, force)
        }).await
    }

    // ========================================================================
    // Advanced Commit Operations
    // ========================================================================

    /// Handle amend commit operation
    async fn handle_amend_commit(&self, repo_id: &str, message: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let message = message.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            Ok(crate::git::ops::amend_head(&repo, &message)?.to_string())
        }).await
    }

    /// Handle commit with author operation
    async fn handle_commit_with_author(
        &self,
        repo_id: &str,
        message: &str,
        author_name: &str,
        author_email: &str,
    ) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let message = message.to_owned();
        let author_name = author_name.to_owned();
        let author_email = author_email.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            Ok(crate::git::ops::commit_with_author(&repo, &message, &author_name, &author_email)?.to_string())
        }).await
    }

    /// Handle stage all including untracked operation
    async fn handle_stage_all_including_untracked(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::stage_all_with_untracked(&repo)
        }).await
    }

    // ========================================================================
    // Merge Conflict Resolution
    // ========================================================================

    /// Handle abort merge operation
    async fn handle_abort_merge(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::abort_merge(&repo)
        }).await
    }

    /// Handle continue merge operation
    async fn handle_continue_merge(&self, repo_id: &str, message: Option<&str>) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let message = message.map(std::borrow::ToOwned::to_owned);
        self.with_repo_blocking(&id, move |repo| {
            Ok(crate::git::ops::continue_merge(&repo, message.as_deref())?.to_string())
        }).await
    }

    /// Handle quit merge operation
    async fn handle_quit_merge(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::quit_merge(&repo)
        }).await
    }

    // ========================================================================
    // Tag Operations
    // ========================================================================

    /// Handle list tags operation
    async fn handle_list_tags(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::list_tags(&repo)
        }).await
    }

    /// Handle create tag operation
    async fn handle_create_tag(&self, repo_id: &str, name: &str, target: Option<&str>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        let target = target.map(std::borrow::ToOwned::to_owned);
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::create_tag(&repo, &name, target.as_deref(), false)
        }).await
    }

    /// Handle delete tag operation
    async fn handle_delete_tag(&self, repo_id: &str, name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::delete_tag(&repo, &name)
        }).await
    }

    // ========================================================================
    // Detailed Status
    // ========================================================================

    /// Handle detailed status operation
    async fn handle_detailed_status(&self, repo_id: &str) -> Result<DetailedStatusInfo> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::detailed_status(&repo)
        }).await
    }

    // ========================================================================
    // Filesystem Operations
    // ========================================================================

    /// Get or create a ContainedRoot for a (repo_id, worktree) pair.
    async fn get_contained_root(
        &self,
        repo_id: &str,
        worktree: &str,
    ) -> Result<Arc<dyn ContainedRoot>, FsServiceError> {
        let key = (repo_id.to_owned(), worktree.to_owned());
        if let Some(root) = self.contained_roots.get(&key) {
            return Ok(Arc::clone(&root));
        }

        // Resolve worktree path from repo
        let id = Self::parse_repo_id(repo_id)
            .map_err(|e| FsServiceError::NotFound(e.to_string()))?;

        let registry = self.registry.read().await;
        let handle = registry.repo(&id)
            .map_err(|e| FsServiceError::NotFound(e.to_string()))?;
        let worktrees = handle.get_worktrees().await
            .map_err(|e| FsServiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

        // Find the worktree matching the requested name
        let mut worktree_path = None;
        for wt in &worktrees {
            let wt_name = wt.path()
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if wt_name == worktree {
                worktree_path = Some(wt.path().to_path_buf());
                break;
            }
        }

        // If worktree name is "." or empty, use the repo's base worktree path
        let worktree_path = if let Some(p) = worktree_path {
            p
        } else if worktree == "." || worktree.is_empty() {
            let repo = handle.open_repo()
                .map_err(|e| FsServiceError::NotFound(e.to_string()))?;
            repo.workdir()
                .map(|wdir| wdir.to_path_buf())
                .ok_or_else(|| FsServiceError::NotFound(
                    format!("worktree '{}' not found for repo '{}'", worktree, repo_id),
                ))?
        } else {
            return Err(FsServiceError::NotFound(
                format!("worktree '{}' not found for repo '{}'", worktree, repo_id),
            ));
        };

        let root = contained_root::open_contained_root(&worktree_path)?;
        let root: Arc<dyn ContainedRoot> = Arc::from(root);
        self.contained_roots.insert(key, Arc::clone(&root));
        Ok(root)
    }

    /// Handle FS open
    async fn handle_fs_open(
        &self,
        ctx: &EnvelopeContext,
        repo_id: &str,
        worktree: &str,
        path: &str,
        read: bool,
        write: bool,
        create: bool,
        truncate: bool,
        append: bool,
        exclusive: bool,
    ) -> Result<u32, FsServiceError> {
        let _ = read; // always open for reading
        let root = self.get_contained_root(repo_id, worktree).await?;
        let file = root.open_file(path, write, create, truncate, append, exclusive)?;

        let subject = ctx.subject().to_string();
        let fd = self.fd_table.alloc_fd(&subject)?;
        self.fd_table.insert(
            fd,
            OpenFile {
                file: Mutex::new(file),
                owner_identity: subject,
                writable: write,
                last_accessed: AtomicU64::new(now_epoch_secs()),
            },
        );
        Ok(fd)
    }

    /// Handle FS close
    async fn handle_fs_close(&self, ctx: &EnvelopeContext, fd: u32) -> Result<(), FsServiceError> {
        let client_id = ctx.subject().to_string();
        // Verify ownership before removing
        {
            let entry = self.fd_table.fds.get(&fd).ok_or(FsServiceError::BadFd(fd))?;
            if entry.owner_identity != client_id {
                return Err(FsServiceError::PermissionDenied("FD not owned by caller".into()));
            }
        }
        self.fd_table.remove(fd, &client_id);
        Ok(())
    }

    /// Handle FS read
    async fn handle_fs_read(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        length: u64,
    ) -> Result<Vec<u8>, FsServiceError> {
        if length > MAX_FS_IO_SIZE {
            return Err(FsServiceError::ResourceLimit(
                format!("read length {} exceeds max {}", length, MAX_FS_IO_SIZE),
            ));
        }
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        let mut file = entry.file.lock();
        let mut buf = vec![0u8; length as usize];
        let n = file.read(&mut buf).map_err(FsServiceError::Io)?;
        buf.truncate(n);
        Ok(buf)
    }

    /// Handle FS write
    async fn handle_fs_write(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        data: &[u8],
    ) -> Result<u64, FsServiceError> {
        if data.len() as u64 > MAX_FS_IO_SIZE {
            return Err(FsServiceError::ResourceLimit(
                format!("write length {} exceeds max {}", data.len(), MAX_FS_IO_SIZE),
            ));
        }
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        if !entry.writable {
            return Err(FsServiceError::PermissionDenied("FD not opened for writing".into()));
        }
        let mut file = entry.file.lock();
        let n = file.write(data).map_err(FsServiceError::Io)?;
        Ok(n as u64)
    }

    /// Handle FS pread (positional read)
    async fn handle_fs_pread(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, FsServiceError> {
        if length > MAX_FS_IO_SIZE {
            return Err(FsServiceError::ResourceLimit(
                format!("pread length {} exceeds max {}", length, MAX_FS_IO_SIZE),
            ));
        }
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        let mut file = entry.file.lock();
        file.seek(SeekFrom::Start(offset)).map_err(FsServiceError::Io)?;
        let mut buf = vec![0u8; length as usize];
        let n = file.read(&mut buf).map_err(FsServiceError::Io)?;
        buf.truncate(n);
        Ok(buf)
    }

    /// Handle FS pwrite (positional write)
    async fn handle_fs_pwrite(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        offset: u64,
        data: &[u8],
    ) -> Result<u64, FsServiceError> {
        if data.len() as u64 > MAX_FS_IO_SIZE {
            return Err(FsServiceError::ResourceLimit(
                format!("pwrite length {} exceeds max {}", data.len(), MAX_FS_IO_SIZE),
            ));
        }
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        if !entry.writable {
            return Err(FsServiceError::PermissionDenied("FD not opened for writing".into()));
        }
        let mut file = entry.file.lock();
        file.seek(SeekFrom::Start(offset)).map_err(FsServiceError::Io)?;
        let n = file.write(data).map_err(FsServiceError::Io)?;
        Ok(n as u64)
    }

    /// Handle FS seek
    async fn handle_fs_seek(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        offset: i64,
        whence: SeekWhenceEnum,
    ) -> Result<u64, FsServiceError> {
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        let mut file = entry.file.lock();
        let seek_from = match whence {
            SeekWhenceEnum::Set => SeekFrom::Start(offset as u64),
            SeekWhenceEnum::Cur => SeekFrom::Current(offset),
            SeekWhenceEnum::End => SeekFrom::End(offset),
        };
        let pos = file.seek(seek_from).map_err(FsServiceError::Io)?;
        Ok(pos)
    }

    /// Handle FS truncate
    async fn handle_fs_truncate(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        length: u64,
    ) -> Result<(), FsServiceError> {
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        if !entry.writable {
            return Err(FsServiceError::PermissionDenied("FD not opened for writing".into()));
        }
        let file = entry.file.lock();
        file.set_len(length).map_err(FsServiceError::Io)
    }

    /// Handle FS fsync
    async fn handle_fs_fsync(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        data_only: bool,
    ) -> Result<(), FsServiceError> {
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        let file = entry.file.lock();
        if data_only {
            file.sync_data().map_err(FsServiceError::Io)
        } else {
            file.sync_all().map_err(FsServiceError::Io)
        }
    }

    /// Handle FS stat
    async fn handle_fs_stat(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<FsStatResponse, FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        match root.stat(path) {
            Ok(meta) => {
                let modified_at = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                Ok(FsStatResponse {
                    exists: true,
                    is_dir: meta.is_dir(),
                    size: meta.len(),
                    modified_at,
                })
            }
            Err(FsServiceError::NotFound(_)) | Err(FsServiceError::Io(_)) => {
                // For stat, NotFound is not an error — it means the file doesn't exist
                Ok(FsStatResponse {
                    exists: false,
                    is_dir: false,
                    size: 0,
                    modified_at: 0,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Handle FS mkdir
    async fn handle_fs_mkdir(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
        recursive: bool,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        if recursive {
            root.mkdir_all(path)
        } else {
            root.mkdir(path)
        }
    }

    /// Handle FS remove file
    async fn handle_fs_remove(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        root.remove_file(path)
    }

    /// Handle FS rmdir
    async fn handle_fs_rmdir(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        root.remove_dir(path)
    }

    /// Handle FS rename
    async fn handle_fs_rename(
        &self,
        repo_id: &str,
        worktree: &str,
        src: &str,
        dst: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        root.rename(src, dst)
    }

    /// Handle FS copy
    async fn handle_fs_copy(
        &self,
        repo_id: &str,
        worktree: &str,
        src: &str,
        dst: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        root.copy_file(src, dst)
    }

    /// Handle FS list directory
    async fn handle_fs_list_dir(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<Vec<FsDirEntryInfo>, FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree).await?;
        root.list_dir(path)
    }

}

// ============================================================================
// Generated Handler Helpers
// ============================================================================

fn tracked_repo_to_data(repo: &TrackedRepository) -> crate::services::generated::registry_client::TrackedRepository {
    crate::services::generated::registry_client::TrackedRepository {
        id: repo.id.to_string(),
        name: repo.name.clone().unwrap_or_default(),
        url: repo.url.clone(),
        worktree_path: repo.worktree_path.to_string_lossy().to_string(),
        tracking_ref: match &repo.tracking_ref {
            GitRef::Branch(b) => b.clone(),
            _ => String::new(),
        },
        current_oid: repo.current_oid.clone().unwrap_or_default(),
        registered_at: repo.registered_at,
    }
}

macro_rules! tracked_variant {
    ($variant:ident, $repo:expr) => {{
        let d = tracked_repo_to_data($repo);
        RegistryResponseVariant::$variant(d)
    }};
}

fn reg_error(msg: &str) -> RegistryResponseVariant {
    RegistryResponseVariant::Error(ErrorInfo {
        message: msg.to_owned(),
        code: "INTERNAL".to_owned(),
        details: String::new(),
    })
}

// ============================================================================
// Generated RegistryHandler Implementation
// ============================================================================

#[async_trait::async_trait(?Send)]
impl RegistryHandler for RegistryService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let op = Operation::from_str(operation)?;
        if self.is_authorized(ctx, resource, op).await {
            Ok(())
        } else {
            anyhow::bail!("Unauthorized: {} cannot {} on {}", ctx.subject(), operation, resource)
        }
    }

    async fn handle_list(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<RegistryResponseVariant> {
        match self.handle_list().await {
            Ok(repos) => Ok(RegistryResponseVariant::ListResult(
                repos.iter().map(tracked_repo_to_data).collect()
            )),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

    async fn handle_get(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<RegistryResponseVariant> {
        match self.handle_get(value).await {
            Ok(Some(repo)) => Ok(tracked_variant!(GetResult, &repo)),
            Ok(None) => Ok(reg_error("Repository not found")),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

    async fn handle_get_by_name(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<RegistryResponseVariant> {
        match self.handle_get_by_name(value).await {
            Ok(Some(repo)) => Ok(tracked_variant!(GetByNameResult, &repo)),
            Ok(None) => Ok(reg_error("Repository not found")),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

    async fn handle_clone(&self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &CloneRequest,
    ) -> Result<RegistryResponseVariant> {
        let name_opt = if data.name.is_empty() { None } else { Some(data.name.as_str()) };
        let branch_opt = if data.branch.is_empty() { None } else { Some(data.branch.as_str()) };
        match self.handle_clone(&data.url, name_opt, data.shallow, Some(data.depth), branch_opt).await {
            Ok(repo) => Ok(tracked_variant!(CloneResult, &repo)),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

    async fn handle_clone_stream(&self, ctx: &EnvelopeContext, _request_id: u64,
        data: &CloneRequest,
    ) -> Result<(StreamInfo, hyprstream_rpc::service::Continuation)> {
        let name_opt = if data.name.is_empty() { None } else { Some(data.name.as_str()) };
        let branch_opt = if data.branch.is_empty() { None } else { Some(data.branch.as_str()) };
        let client_ephemeral_pubkey = ctx.ephemeral_pubkey();
        self.prepare_clone_stream(&data.url, name_opt, data.shallow, Some(data.depth), branch_opt, client_ephemeral_pubkey).await
    }

    async fn handle_register(&self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &RegisterRequest,
    ) -> Result<RegistryResponseVariant> {
        let name_opt = if data.name.is_empty() { None } else { Some(data.name.as_str()) };
        let tracking_ref_opt = if data.tracking_ref.is_empty() { None } else { Some(data.tracking_ref.as_str()) };
        match self.handle_register(&data.path, name_opt, tracking_ref_opt).await {
            Ok(repo) => Ok(tracked_variant!(RegisterResult, &repo)),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64, value: &str) -> Result<RegistryResponseVariant> {
        match self.handle_remove(value).await {
            Ok(()) => Ok(RegistryResponseVariant::RemoveResult),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

    async fn handle_health_check(&self, _ctx: &EnvelopeContext, _request_id: u64) -> Result<RegistryResponseVariant> {
        match self.handle_health_check().await {
            Ok(status) => Ok(RegistryResponseVariant::HealthCheckResult(status)),
            Err(e) => Ok(reg_error(&e.to_string())),
        }
    }

}

// ============================================================================
// Generated RepoHandler Implementation (repo-scoped operations)
// ============================================================================
//
// Auth is handled by the generated dispatch_repo() function via authorize().
// Each method delegates to the corresponding internal method on RegistryService.

#[async_trait::async_trait(?Send)]
impl RepoHandler for RegistryService {
    async fn handle_create_worktree(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &CreateWorktreeRequest,
    ) -> Result<String> {
        let path_buf = self.handle_create_worktree(repo_id, &data.path, &data.branch_name).await?;
        Ok(path_buf.to_string_lossy().to_string())
    }

    async fn handle_list_worktrees(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<Vec<crate::services::generated::registry_client::WorktreeInfo>> {
        self.handle_list_worktrees(repo_id).await
    }

    async fn handle_remove_worktree(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &RemoveWorktreeRequest,
    ) -> Result<()> {
        self.handle_remove_worktree(repo_id, &data.worktree_path).await
    }

    async fn handle_create_branch(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &BranchRequest,
    ) -> Result<()> {
        let sp = if data.start_point.is_empty() { None } else { Some(data.start_point.as_str()) };
        self.handle_create_branch(repo_id, &data.branch_name, sp).await
    }

    async fn handle_list_branches(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<Vec<String>> {
        self.handle_list_branches(repo_id).await
    }

    async fn handle_checkout(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &CheckoutRequest,
    ) -> Result<()> {
        self.handle_checkout(repo_id, &data.ref_name).await
    }

    async fn handle_stage_all(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<()> {
        self.handle_stage_all(repo_id).await
    }

    async fn handle_stage_files(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &StageFilesRequest,
    ) -> Result<()> {
        self.handle_stage_files(repo_id, data.files.clone()).await
    }

    async fn handle_commit(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &CommitRequest,
    ) -> Result<String> {
        self.handle_commit(repo_id, &data.message).await
    }

    async fn handle_merge(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &MergeRequest,
    ) -> Result<()> {
        let msg = if data.message.is_empty() { None } else { Some(data.message.as_str()) };
        self.handle_merge(repo_id, &data.source, msg).await?;
        Ok(())
    }

    async fn handle_abort_merge(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<()> {
        self.handle_abort_merge(repo_id).await
    }

    async fn handle_continue_merge(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &ContinueMergeRequest,
    ) -> Result<()> {
        let msg = if data.message.is_empty() { None } else { Some(data.message.as_str()) };
        self.handle_continue_merge(repo_id, msg).await?;
        Ok(())
    }

    async fn handle_quit_merge(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<()> {
        self.handle_quit_merge(repo_id).await
    }

    async fn handle_get_head(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<String> {
        self.handle_get_head(repo_id).await
    }

    async fn handle_get_ref(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &GetRefRequest,
    ) -> Result<String> {
        self.handle_get_ref(repo_id, &data.ref_name).await
    }

    async fn handle_status(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<crate::services::generated::registry_client::RepositoryStatus> {
        let status = self.handle_status(repo_id).await?;
        Ok(crate::services::generated::registry_client::RepositoryStatus {
            branch: status.branch.unwrap_or_default(),
            head_oid: status.head.map(|h| h.to_string()).unwrap_or_default(),
            ahead: status.ahead as u32,
            behind: status.behind as u32,
            is_clean: status.is_clean,
            modified_files: status.modified_files.iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
        })
    }

    async fn handle_detailed_status(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<DetailedStatusInfo> {
        self.handle_detailed_status(repo_id).await
    }

    async fn handle_list_remotes(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<Vec<RemoteInfo>> {
        self.handle_list_remotes(repo_id).await
    }

    async fn handle_add_remote(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &AddRemoteRequest,
    ) -> Result<()> {
        self.handle_add_remote(repo_id, &data.name, &data.url).await
    }

    async fn handle_remove_remote(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &RemoveRemoteRequest,
    ) -> Result<()> {
        self.handle_remove_remote(repo_id, &data.name).await
    }

    async fn handle_set_remote_url(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &SetRemoteUrlRequest,
    ) -> Result<()> {
        self.handle_set_remote_url(repo_id, &data.name, &data.url).await
    }

    async fn handle_rename_remote(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &RenameRemoteRequest,
    ) -> Result<()> {
        self.handle_rename_remote(repo_id, &data.old_name, &data.new_name).await
    }

    async fn handle_push(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &PushRequest,
    ) -> Result<()> {
        self.handle_push(repo_id, &data.remote, &data.refspec, data.force).await
    }

    async fn handle_amend_commit(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &AmendCommitRequest,
    ) -> Result<String> {
        self.handle_amend_commit(repo_id, &data.message).await
    }

    async fn handle_commit_with_author(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &CommitWithAuthorRequest,
    ) -> Result<String> {
        self.handle_commit_with_author(repo_id, &data.message, &data.author_name, &data.author_email).await
    }

    async fn handle_stage_all_including_untracked(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<()> {
        self.handle_stage_all_including_untracked(repo_id).await
    }

    async fn handle_list_tags(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str,
    ) -> Result<Vec<String>> {
        self.handle_list_tags(repo_id).await
    }

    async fn handle_create_tag(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &CreateTagRequest,
    ) -> Result<()> {
        let t = if data.target.is_empty() { None } else { Some(data.target.as_str()) };
        self.handle_create_tag(repo_id, &data.name, t).await
    }

    async fn handle_delete_tag(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &DeleteTagRequest,
    ) -> Result<()> {
        self.handle_delete_tag(repo_id, &data.name).await
    }

    async fn handle_update(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &UpdateRequest,
    ) -> Result<()> {
        let r = if data.refspec.is_empty() { None } else { Some(data.refspec.as_str()) };
        self.handle_update(repo_id, r).await
    }

    async fn handle_ensure_worktree(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, data: &EnsureWorktreeRequest,
    ) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;

        // Check if a worktree for this branch already exists
        let worktrees = handle.get_worktrees().await?;
        for wt in &worktrees {
            let wt_branch = wt
                .path()
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if wt_branch == data.branch {
                return Ok(wt.path().to_string_lossy().to_string());
            }
        }

        // Not found — create it
        let worktree = handle.create_worktree(
            &PathBuf::from(&data.branch),
            &data.branch,
        ).await?;
        Ok(worktree.path().to_string_lossy().to_string())
    }
}

// ============================================================================
// Generated WorktreeHandler Implementation (nested scope under RepoHandler)
// ============================================================================

#[async_trait::async_trait(?Send)]
impl WorktreeHandler for RegistryService {
    async fn handle_open(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsOpenRequest,
    ) -> Result<FsOpenResponse> {
        let fd = self.handle_fs_open(
            ctx, repo_id, name, &data.path,
            data.read, data.write, data.create,
            data.truncate, data.append, data.exclusive,
        ).await?;
        Ok(FsOpenResponse { fd })
    }

    async fn handle_close(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsCloseRequest,
    ) -> Result<()> {
        self.handle_fs_close(ctx, data.fd).await?;
        Ok(())
    }

    async fn handle_read(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsReadRequest,
    ) -> Result<FsReadResponse> {
        let read_data = self.handle_fs_read(ctx, data.fd, data.length).await?;
        Ok(FsReadResponse { data: read_data })
    }

    async fn handle_write(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsWriteRequest,
    ) -> Result<FsWriteResponse> {
        let bytes_written = self.handle_fs_write(ctx, data.fd, &data.data).await?;
        Ok(FsWriteResponse { bytes_written })
    }

    async fn handle_pread(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsPreadRequest,
    ) -> Result<FsReadResponse> {
        let read_data = self.handle_fs_pread(ctx, data.fd, data.offset, data.length).await?;
        Ok(FsReadResponse { data: read_data })
    }

    async fn handle_pwrite(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsPwriteRequest,
    ) -> Result<FsWriteResponse> {
        let bytes_written = self.handle_fs_pwrite(ctx, data.fd, data.offset, &data.data).await?;
        Ok(FsWriteResponse { bytes_written })
    }

    async fn handle_seek(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsSeekRequest,
    ) -> Result<FsSeekResponse> {
        let position = self.handle_fs_seek(ctx, data.fd, data.offset, data.whence).await?;
        Ok(FsSeekResponse { position })
    }

    async fn handle_truncate(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsTruncateRequest,
    ) -> Result<()> {
        self.handle_fs_truncate(ctx, data.fd, data.length).await?;
        Ok(())
    }

    async fn handle_fsync(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &FsSyncRequest,
    ) -> Result<()> {
        self.handle_fs_fsync(ctx, data.fd, data.data_only).await?;
        Ok(())
    }

    async fn handle_stat(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsPathRequest,
    ) -> Result<FsStatResponse> {
        Ok(self.handle_fs_stat(repo_id, name, &data.path).await?)
    }

    async fn handle_mkdir(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsMkdirRequest,
    ) -> Result<()> {
        self.handle_fs_mkdir(repo_id, name, &data.path, data.recursive).await?;
        Ok(())
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsPathRequest,
    ) -> Result<()> {
        self.handle_fs_remove(repo_id, name, &data.path).await?;
        Ok(())
    }

    async fn handle_rmdir(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsPathRequest,
    ) -> Result<()> {
        self.handle_fs_rmdir(repo_id, name, &data.path).await?;
        Ok(())
    }

    async fn handle_rename(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsRenameRequest,
    ) -> Result<()> {
        self.handle_fs_rename(repo_id, name, &data.src, &data.dst).await?;
        Ok(())
    }

    async fn handle_copy(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsCopyRequest,
    ) -> Result<()> {
        self.handle_fs_copy(repo_id, name, &data.src, &data.dst).await?;
        Ok(())
    }

    async fn handle_list_dir(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsPathRequest,
    ) -> Result<Vec<FsDirEntryInfo>> {
        Ok(self.handle_fs_list_dir(repo_id, name, &data.path).await?)
    }

    async fn handle_open_stream(&self, _ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, _data: &FsOpenRequest,
    ) -> Result<FsStreamInfoResponse> {
        anyhow::bail!("FS streaming not yet implemented")
    }

    async fn handle_read_file(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsPathRequest,
    ) -> Result<FsReadResponse> {
        use std::io::Read;
        let root = self.get_contained_root(repo_id, name).await
            .map_err(|e| anyhow!("{}", e))?;
        let mut file = root.open_file(&data.path, false, false, false, false, false)
            .map_err(|e| anyhow!("{}", e))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        Ok(FsReadResponse { data: buf })
    }

    async fn handle_write_file(&self, _ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &FsWriteFileRequest,
    ) -> Result<FsWriteResponse> {
        use std::io::Write;
        let root = self.get_contained_root(repo_id, name).await
            .map_err(|e| anyhow!("{}", e))?;
        // write=true, create=true, truncate=true
        let mut file = root.open_file(&data.path, true, true, true, false, false)
            .map_err(|e| anyhow!("{}", e))?;
        file.write_all(&data.data)?;
        file.flush()?;
        Ok(FsWriteResponse { bytes_written: data.data.len() as u64 })
    }
}

#[async_trait(?Send)]
impl ZmqService for RegistryService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
        dispatch_registry(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "registry"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

// ============================================================================
// MetricsRegistryClient Implementation (on generated RegistryClient)
// ============================================================================

use hyprstream_metrics::checkpoint::manager::{
    RegistryClient as MetricsRegistryClient, RegistryError as MetricsRegistryError,
};

#[async_trait]
impl MetricsRegistryClient for GenRegistryClient {
    async fn get_by_name(
        &self,
        name: &str,
    ) -> Result<Option<TrackedRepository>, MetricsRegistryError> {
        let r = self.get_by_name(name)
            .await
            .map_err(|e| MetricsRegistryError::Operation(e.to_string()))?;
        Ok(Some(variant_to_tracked_repository(
            &r.id, &r.name, &r.url, &r.worktree_path, &r.tracking_ref, &r.current_oid, r.registered_at,
        ).map_err(|e| MetricsRegistryError::Operation(e.to_string()))?))
    }

    async fn register(
        &self,
        _id: &RepoId,
        name: Option<&str>,
        path: &std::path::Path,
    ) -> Result<(), MetricsRegistryError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| MetricsRegistryError::Operation("Invalid path encoding".to_owned()))?;
        self.register(path_str, name.unwrap_or(""), "")
            .await
            .map(|_| ())
            .map_err(|e| MetricsRegistryError::Operation(e.to_string()))
    }
}







#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::auth::PolicyManager;
    use crate::services::{PolicyService, PolicyClient};
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_registry_service_health_check() {
        use hyprstream_rpc::service::InprocManager;
        use hyprstream_rpc::transport::TransportConfig;

        let temp_dir = TempDir::new().expect("test: create temp dir");
        let context = crate::zmq::global_context();

        // Generate keypair for signing/verification
        let (signing_key, _verifying_key) = generate_signing_keypair();

        // Create a permissive policy manager and start PolicyService first
        let policy_manager = Arc::new(PolicyManager::permissive().await.expect("test: create policy manager"));
        let policy_transport = TransportConfig::inproc("test-policy-health");
        let policy_service = PolicyService::new(
            policy_manager,
            Arc::new(signing_key.clone()),
            crate::config::TokenConfig::default(),
            context.clone(),
            policy_transport,
        );
        let manager = InprocManager::new();
        let _policy_handle = manager.spawn(Box::new(policy_service)).await.expect("test: start policy service");

        // Create policy client for RegistryService
        let policy_client: PolicyClient = crate::services::core::create_service_client(
            "inproc://test-policy-health",
            signing_key.clone(),
            RequestIdentity::local(),
        );

        // Start the registry service with policy client
        let registry_transport = TransportConfig::inproc("test-registry-health");
        let registry_service = RegistryService::new(
            temp_dir.path(),
            policy_client,
            context.clone(),
            registry_transport,
            signing_key.clone(),
        ).await.expect("test: create registry service");
        let mut handle = manager.spawn(Box::new(registry_service)).await.expect("test: start registry service");

        // Create signed client with matching key and local identity
        let client: GenRegistryClient = crate::services::core::create_service_client(
            "inproc://test-registry-health",
            signing_key,
            RequestIdentity::local(),
        );
        // health_check returns () on success
        let result = client.health_check().await;
        assert!(result.is_ok(), "health_check should succeed: {:?}", result.err());

        // Stop the service
        let _ = handle.stop().await;
    }
}
