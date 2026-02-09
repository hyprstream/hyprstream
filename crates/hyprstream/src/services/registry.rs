//! ZMQ-based registry service for repository management
//!
//! This service wraps git2db and provides a ZMQ REQ/REP interface for
//! repository operations. It uses Cap'n Proto for serialization.

use crate::auth::Operation;
use crate::services::PolicyClient;
use crate::registry_capnp;
use crate::services::rpc_types::{
    DetailedStatusData, FsVoidVariant, HealthStatus, RemoteInfo, RegistryResponse, WorktreeData,
};
use crate::services::traits::{CloneOptions, DetailedStatus, FileChangeType, FileStatus};
use crate::services::traits::{FsDirEntry, FsServiceError, FsStatInfo, SeekWhence, MAX_FDS_GLOBAL, MAX_FDS_PER_CLIENT, MAX_FS_IO_SIZE};
use crate::services::contained_root::{self, ContainedRoot};
use crate::services::{CallOptions, EnvelopeContext, ZmqService};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as endpoint_registry, SocketKind};
use hyprstream_rpc::serialize_message;
use hyprstream_rpc::{StreamChannel, StreamContext};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use capnp::message::ReaderOptions;
use capnp::serialize;
use dashmap::DashMap;
use git2db::{CloneBuilder, Git2DB, GitRef, RepoId, RepositoryStatus, TrackedRepository};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::io::{Read as _, Write as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Service name for endpoint registry
const SERVICE_NAME: &str = "registry";

// Generated client types
use crate::services::generated::registry_client::{
    RegistryClient as GenRegistryClient, RegistryResponseVariant,
};

// ============================================================================
// Parsing Helper Functions
// ============================================================================

/// Parse a stream started response — uses generated response parser
fn parse_stream_started_response(response: &[u8]) -> Result<crate::services::rpc_types::StreamStartedInfo> {
    match GenRegistryClient::parse_response(response)? {
        RegistryResponseVariant::CloneStreamResult { stream_id, stream_endpoint, server_pubkey } => {
            Ok(crate::services::rpc_types::StreamStartedInfo {
                stream_id,
                endpoint: stream_endpoint,
                server_pubkey: server_pubkey.try_into().unwrap_or([0u8; 32]),
            })
        }
        RegistryResponseVariant::Error { message, .. } => Err(anyhow!("{}", message)),
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
// RegistryZmq - Wrapper implementing RegistryClient trait
// ============================================================================

/// ZMQ-based registry client wrapping the generated `RegistryClient`.
///
/// Simple methods delegate to `gen`; `list()` and `clone_stream()` use manual
/// parsing (List(TrackedRepository) placeholder and custom CallOptions respectively).
pub struct RegistryZmq {
    /// Generated typed client (handles all transport including streaming via call_with_options)
    pub(crate) gen: GenRegistryClient,
}

impl RegistryZmq {
    /// Create a new registry client with signing credentials.
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = endpoint_registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        Self::with_endpoint(&endpoint, signing_key, identity)
    }

    /// Create a registry client connected to a specific endpoint.
    pub fn with_endpoint(endpoint: &str, signing_key: SigningKey, identity: RequestIdentity) -> Self {
        use crate::services::core::ZmqClientBase;
        use crate::zmq::global_context;
        use hyprstream_rpc::service::factory::ServiceClient;

        let server_verifying_key = signing_key.verifying_key();
        let base = ZmqClientBase::new(
            endpoint, global_context(), signing_key, server_verifying_key, identity,
        );
        Self {
            gen: GenRegistryClient::from_zmq(base),
        }
    }

    /// Get a `TrackedRepository` by name (internal helper for repo() and MetricsRegistryClient).
    async fn get_by_name_internal(&self, name: &str) -> Result<Option<TrackedRepository>> {
        let r = self.gen.get_by_name(name).await?;
        Ok(Some(variant_to_tracked_repository(
            &r.id, &r.name, &r.url, &r.worktree_path, &r.tracking_ref, &r.current_oid, r.registered_at,
        )?))
    }
}

use crate::services::traits::{RegistryClient, RegistryServiceError, RepositoryClient, WorktreeInfo};

#[async_trait]
impl RegistryClient for RegistryZmq {
    async fn list(&self) -> Result<Vec<TrackedRepository>, RegistryServiceError> {
        let repos = self.gen.list().await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        repos.into_iter().map(|r| {
            variant_to_tracked_repository(
                &r.id, &r.name, &r.url, &r.worktree_path,
                &r.tracking_ref, &r.current_oid, r.registered_at,
            ).map_err(|e| RegistryServiceError::transport(e.to_string()))
        }).collect()
    }

    async fn get(&self, id: &RepoId) -> Result<Option<TrackedRepository>, RegistryServiceError> {
        let r = self.gen.get(&id.to_string()).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        Ok(Some(variant_to_tracked_repository(
            &r.id, &r.name, &r.url, &r.worktree_path, &r.tracking_ref, &r.current_oid, r.registered_at,
        ).map_err(|e| RegistryServiceError::transport(e.to_string()))?))
    }

    async fn get_by_name(
        &self,
        name: &str,
    ) -> Result<Option<TrackedRepository>, RegistryServiceError> {
        self.get_by_name_internal(name).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn clone_repo(
        &self,
        url: &str,
        name: Option<&str>,
        options: &CloneOptions,
    ) -> Result<RepoId, RegistryServiceError> {
        let r = self.gen.clone(
            url,
            name.unwrap_or(""),
            options.shallow,
            options.depth,
            options.branch.as_deref().unwrap_or(""),
        ).await.map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        let uuid = Uuid::parse_str(&r.id)
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        Ok(RepoId::from_uuid(uuid))
    }

    async fn clone_stream(
        &self,
        url: &str,
        name: Option<&str>,
        options: &CloneOptions,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<crate::services::rpc_types::StreamStartedInfo, RegistryServiceError> {
        // Manual: needs custom CallOptions for ephemeral pubkey
        let id = self.gen.next_id();
        let payload = serialize_message(|msg| {
            let mut req = msg.init_root::<registry_capnp::registry_request::Builder>();
            req.set_id(id);
            let mut clone_req = req.init_clone_stream();
            clone_req.set_url(url);
            if let Some(n) = name {
                clone_req.set_name(n);
            }
            clone_req.set_shallow(options.shallow);
            clone_req.set_depth(options.depth);
            if let Some(ref branch) = options.branch {
                clone_req.set_branch(branch);
            }
        }).map_err(|e| RegistryServiceError::transport(e.to_string()))?;

        let opts = match ephemeral_pubkey {
            Some(pk) => CallOptions::default().ephemeral_pubkey(pk),
            None => CallOptions::default(),
        };

        let response = self.gen.call_with_options(payload, opts).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        parse_stream_started_response(&response)
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn register(
        &self,
        _id: &RepoId,
        name: Option<&str>,
        path: &Path,
    ) -> Result<(), RegistryServiceError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| RegistryServiceError::transport("Invalid path encoding"))?;
        let _r = self.gen.register(path_str, name.unwrap_or(""), "").await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        Ok(())
    }

    async fn upsert(&self, _name: &str, _url: &str) -> Result<RepoId, RegistryServiceError> {
        Err(RegistryServiceError::transport(
            "Upsert operation not yet supported via ZMQ",
        ))
    }

    async fn remove(&self, id: &RepoId) -> Result<(), RegistryServiceError> {
        self.gen.remove(&id.to_string()).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))
    }

    async fn health_check(&self) -> Result<(), RegistryServiceError> {
        let _r = self.gen.health_check().await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?;
        Ok(())
    }

    async fn repo(
        &self,
        name: &str,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError> {
        let repo = self.get_by_name_internal(name).await
            .map_err(|e| RegistryServiceError::transport(e.to_string()))?
            .ok_or_else(|| RegistryServiceError::transport(format!("Repository '{name}' not found")))?;

        Ok(Arc::new(RepositoryZmqClient::new(
            self.gen.endpoint().to_owned(),
            repo.id,
            name.to_owned(),
            self.gen.signing_key().clone(),
            self.gen.identity().clone(),
        )))
    }

    async fn repo_by_id(
        &self,
        id: &RepoId,
    ) -> Result<Arc<dyn RepositoryClient>, RegistryServiceError> {
        let repo = self.get(id).await?
            .ok_or_else(|| RegistryServiceError::transport(format!("Repository '{id}' not found")))?;

        let name = repo.name.unwrap_or_else(|| id.to_string());

        Ok(Arc::new(RepositoryZmqClient::new(
            self.gen.endpoint().to_owned(),
            id.clone(),
            name,
            self.gen.signing_key().clone(),
            self.gen.identity().clone(),
        )))
    }
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
/// Note: The runtime handle is captured on creation to avoid the nested runtime
/// anti-pattern. Handler methods use this handle for async operations instead
///
/// ## Streaming Support
///
/// For streaming clone operations, the service:
/// 1. Performs DH key exchange with the client
/// 2. Pre-authorizes the stream with StreamService (so client can subscribe)
/// 3. Queues the clone task to a background worker thread
/// 4. The worker publishes progress via PUSH to StreamService
///
/// The registry is wrapped in RwLock for interior mutability since some operations
/// (like clone) require mutable access but ZmqService::handle_request takes &self.
pub struct RegistryService {
    // Business logic
    registry: Arc<RwLock<Git2DB>>,
    #[allow(dead_code)] // Future: base directory for relative path operations
    base_dir: PathBuf,
    /// Captured runtime handle for async operations in sync handlers
    runtime_handle: tokio::runtime::Handle,
    /// Policy client for authorization checks (uses ZMQ to PolicyService)
    policy_client: PolicyClient,
    // Infrastructure (for Spawnable)
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
    /// Channel to send pending clone streams to the background worker
    pending_clone_tx: std::sync::mpsc::Sender<PendingCloneStreamTask>,
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

/// Task sent to the background streaming worker
struct PendingCloneStreamTask {
    /// Stream context with DH-derived keys
    stream_ctx: StreamContext,
    /// Clone URL
    url: String,
    /// Optional name for the repository
    name: Option<String>,
    /// Shallow clone flag
    shallow: bool,
    /// Clone depth
    depth: Option<u32>,
    /// Branch to clone
    branch: Option<String>,
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
        // Capture the runtime handle to avoid nested runtime anti-pattern
        let runtime_handle = tokio::runtime::Handle::current();

        // Create channel for pending clone streams
        let (pending_clone_tx, pending_clone_rx) = std::sync::mpsc::channel::<PendingCloneStreamTask>();

        // Spawn background worker for streaming clone operations
        // Worker creates its own StreamChannel and tokio runtime
        let worker_registry = Arc::new(RwLock::new(registry));
        let worker_registry_clone = Arc::clone(&worker_registry);
        let worker_context = Arc::clone(&context);
        let worker_signing_key = signing_key.clone();

        std::thread::spawn(move || {
            // Create worker's own StreamChannel
            let worker_stream_channel = StreamChannel::new(worker_context, worker_signing_key);
            Self::streaming_worker(
                worker_stream_channel,
                worker_registry_clone,
                pending_clone_rx,
            );
        });

        // Create FD table and spawn reaper
        let fd_table = Arc::new(FdTable::new());
        let reaper_fd_table = Arc::clone(&fd_table);
        tokio::spawn(async move {
            Self::fd_reaper(reaper_fd_table).await;
        });

        let service = Self {
            registry: worker_registry,
            base_dir,
            runtime_handle,
            policy_client,
            context,
            transport,
            signing_key,
            pending_clone_tx,
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

    /// Background worker thread for streaming clone operations.
    ///
    /// Creates its own tokio runtime for async operations and uses
    /// StreamChannel for publishing progress updates.
    fn streaming_worker(
        stream_channel: StreamChannel,
        registry: Arc<RwLock<Git2DB>>,
        receiver: std::sync::mpsc::Receiver<PendingCloneStreamTask>,
    ) {
        info!("Registry streaming worker started");

        // Create a dedicated tokio runtime for this worker thread
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                error!("Failed to create tokio runtime for streaming worker: {}", e);
                return;
            }
        };

        // Process pending streams
        loop {
            match receiver.recv() {
                Ok(task) => {
                    rt.block_on(Self::execute_clone_stream_task(
                        &stream_channel,
                        &registry,
                        task,
                    ));
                }
                Err(_) => {
                    // Channel closed, shutdown
                    debug!("Registry streaming worker shutting down");
                    break;
                }
            }
        }
    }

    /// Execute a single streaming clone task with real-time progress.
    ///
    /// Uses git2db's callback_config to receive progress updates during clone,
    /// which are forwarded to the client via StreamChannel in real-time.
    async fn execute_clone_stream_task(
        stream_channel: &StreamChannel,
        registry: &Arc<RwLock<Git2DB>>,
        task: PendingCloneStreamTask,
    ) {
        use hyprstream_rpc::streaming::ProgressUpdate;

        let stream_ctx = &task.stream_ctx;

        debug!(
            stream_id = %stream_ctx.stream_id(),
            url = %task.url,
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
        let result = stream_channel.with_publisher(stream_ctx, |mut publisher| async move {
            // Spawn clone task - runs concurrently with progress streaming
            let registry_clone = Arc::clone(registry);
            let url = task.url.clone();
            let name = task.name.clone();
            let shallow = task.shallow;
            let depth = task.depth;
            let branch = task.branch.clone();

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

    /// Check authorization for an operation.
    ///
    /// Returns the unauthorized response if the check fails, or None if authorized.
    /// Uses PolicyClient for async policy checks over ZMQ.
    fn check_auth(
        &self,
        ctx: &EnvelopeContext,
        request_id: u64,
        resource: &str,
        operation: Operation,
    ) -> Option<Vec<u8>> {
        let subject = ctx.subject();
        // RegistryService runs on multi-threaded runtime, so block_in_place is safe
        let allowed = tokio::task::block_in_place(|| {
            self.runtime_handle.block_on(
                self.policy_client.check_policy(&subject, resource, operation)
            )
        }).unwrap_or_else(|e| {
            // Log error before denying - prevents silent failures
            warn!(
                "Policy check failed for {} on {}: {} - denying access",
                subject, resource, e
            );
            false
        });

        if allowed {
            None // Authorized
        } else {
            debug!(
                "Authorization denied: {} cannot {} on {}",
                subject,
                operation.as_str(),
                resource
            );
            let subject_str = subject.to_string();
            Some(RegistryResponse::unauthorized(
                request_id,
                &subject_str,
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
    fn handle_list(&self) -> Result<Vec<TrackedRepository>> {
        let registry = self.registry.blocking_read();
        Ok(registry.list().cloned().collect())
    }

    /// Handle get repository by ID
    fn handle_get(&self, repo_id: &str) -> Result<Option<TrackedRepository>> {
        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.blocking_read();
        let result = registry.list().find(|r| r.id == id).cloned();
        Ok(result)
    }

    /// Handle get repository by name
    fn handle_get_by_name(&self, name: &str) -> Result<Option<TrackedRepository>> {
        let registry = self.registry.blocking_read();
        let result = registry
            .list()
            .find(|r| r.name.as_ref() == Some(&name.to_owned()))
            .cloned();
        Ok(result)
    }

    /// Handle list branches
    fn handle_list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
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
            let registry = self.registry.read().await;
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
            let registry = self.registry.read().await;
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
            let registry = self.registry.read().await;
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
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            let oid = handle.commit(message).await?;
            Ok(oid.to_string())
        })
    }

    /// Handle merge
    fn handle_merge(&self, repo_id: &str, source: &str, message: Option<&str>) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let source = source.to_owned();
        let message = message.map(std::borrow::ToOwned::to_owned);

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            let oid = handle.merge(&source, message.as_deref()).await?;
            Ok(oid.to_string())
        })
    }

    /// Handle status
    fn handle_status(&self, repo_id: &str) -> Result<git2db::RepositoryStatus> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            let status = handle.status().await?;
            Ok(status)
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
        // Use captured runtime handle instead of creating new runtime
        // CloneBuilder manages locks internally for optimal performance
        self.runtime_handle.block_on(async {
            Self::clone_repo_inner(
                Arc::clone(&self.registry),
                url,
                name,
                shallow,
                depth,
                branch,
                None,
            ).await
        })
    }

    // ========================================================================
    // Streaming Clone Support
    // ========================================================================

    /// Prepare a streaming clone operation and queue it for background execution.
    ///
    /// Creates a temporary StreamChannel to perform DH key exchange and pre-authorization,
    /// then queues the clone task. Returns stream info for the client.
    fn prepare_clone_stream(
        &self,
        url: &str,
        name: Option<&str>,
        shallow: bool,
        depth: Option<u32>,
        branch: Option<&str>,
        client_ephemeral_pubkey: Option<&[u8]>,
    ) -> Result<(String, [u8; 32])> {
        // DH key derivation is required
        let client_pub_bytes = client_ephemeral_pubkey
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;

        // Create temporary StreamChannel for DH key exchange and pre-authorization
        let stream_channel = StreamChannel::new(
            Arc::clone(&self.context),
            self.signing_key.clone(),
        );

        // 10 minutes expiry for clone operations
        // Use block_on since we're in a sync context
        let stream_ctx = self.runtime_handle.block_on(
            stream_channel.prepare_stream(client_pub_bytes, 600)
        )?;

        debug!(
            stream_id = %stream_ctx.stream_id(),
            topic = %stream_ctx.topic(),
            "Clone stream prepared (DH + pre-authorization via StreamChannel)"
        );

        let stream_id = stream_ctx.stream_id().to_owned();
        let server_pubkey = *stream_ctx.server_pubkey();

        // Create task and send to worker
        let task = PendingCloneStreamTask {
            stream_ctx,
            url: url.to_owned(),
            name: name.map(std::borrow::ToOwned::to_owned),
            shallow,
            depth,
            branch: branch.map(std::borrow::ToOwned::to_owned),
        };

        self.pending_clone_tx.send(task)
            .map_err(|_| anyhow!("Streaming worker channel closed"))?;

        Ok((stream_id, server_pubkey))
    }

    /// Handle list worktrees
    fn handle_list_worktrees(&self, repo_id: &str) -> Result<Vec<WorktreeData>> {
        let id = Self::parse_repo_id(repo_id)?;

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
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
                    .map(std::borrow::ToOwned::to_owned);

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
        let branch = branch.to_owned();

        // Use captured runtime handle instead of creating new runtime
        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
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
        let name = name.map(std::borrow::ToOwned::to_owned);
        // Note: tracking_ref is not yet used by register_repository

        // Register requires write lock
        self.runtime_handle.block_on(async {
            let mut registry = self.registry.write().await;

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
            let mut registry = self.registry.write().await;
            registry.remove_repository(&id).await?;
            Ok(())
        })
    }

    /// Handle remove worktree operation
    fn handle_remove_worktree(&self, repo_id: &str, worktree_path: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let worktree_path = PathBuf::from(worktree_path);

        self.runtime_handle.block_on(async {
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
        })
    }

    /// Handle stage files operation
    fn handle_stage_files(&self, repo_id: &str, files: Vec<String>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
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
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;

            // Get HEAD ref (DefaultBranch resolves to HEAD)
            let oid = handle.resolve_git_ref(&GitRef::DefaultBranch).await?;
            Ok(oid.to_string())
        })
    }

    /// Handle get ref operation
    fn handle_get_ref(&self, repo_id: &str, ref_name: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let ref_name = ref_name.to_owned();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;

            // Try to resolve as a revspec
            let oid = handle.resolve_revspec(&ref_name).await?;
            Ok(oid.to_string())
        })
    }

    /// Execute a blocking git2 operation on a repository.
    ///
    /// Handles the boilerplate of: parse_repo_id → block_on → read registry →
    /// get handle → worktree path → spawn_blocking → Repository::open → operation.
    fn with_repo_blocking<F, T>(&self, id: &git2db::RepoId, f: F) -> Result<T>
    where
        F: FnOnce(git2::Repository) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(id)?;
            let repo_path = handle.worktree()?.to_path_buf();
            tokio::task::spawn_blocking(move || {
                let repo = git2::Repository::open(&repo_path)
                    .map_err(|e| anyhow!("Failed to open repository: {}", e))?;
                f(repo)
            })
            .await
            .map_err(|e| anyhow!("Task join error: {}", e))?
        })
    }

    /// Handle update operation (fetch from remote)
    fn handle_update(&self, repo_id: &str, refspec: Option<&str>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let refspec = refspec.map(std::borrow::ToOwned::to_owned);
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::fetch(&repo, "origin", refspec.as_deref())
        })
    }

    /// Handle health check
    fn handle_health_check(&self) -> Result<HealthStatus> {
        let registry = self.registry.blocking_read();
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
    fn handle_list_remotes(&self, repo_id: &str) -> Result<Vec<RemoteInfo>> {
        let id = Self::parse_repo_id(repo_id)?;

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
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
        let name = name.to_owned();
        let url = url.to_owned();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            handle.remote().add(&name, &url).await?;
            Ok(())
        })
    }

    /// Handle remove remote operation
    fn handle_remove_remote(&self, repo_id: &str, name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            handle.remote().remove(&name).await?;
            Ok(())
        })
    }

    /// Handle set remote URL operation
    fn handle_set_remote_url(&self, repo_id: &str, name: &str, url: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        let url = url.to_owned();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            handle.remote().set_url(&name, &url).await?;
            Ok(())
        })
    }

    /// Handle rename remote operation
    fn handle_rename_remote(&self, repo_id: &str, old_name: &str, new_name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let old_name = old_name.to_owned();
        let new_name = new_name.to_owned();

        self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)?;
            handle.remote().rename(&old_name, &new_name).await?;
            Ok(())
        })
    }

    // ========================================================================
    // Push Operations
    // ========================================================================

    /// Handle push operation
    fn handle_push(&self, repo_id: &str, remote: &str, refspec: &str, force: bool) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let remote = remote.to_owned();
        let refspec = refspec.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::push(&repo, &remote, &refspec, force)
        })
    }

    // ========================================================================
    // Advanced Commit Operations
    // ========================================================================

    /// Handle amend commit operation
    fn handle_amend_commit(&self, repo_id: &str, message: &str) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let message = message.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            Ok(crate::git::ops::amend_head(&repo, &message)?.to_string())
        })
    }

    /// Handle commit with author operation
    fn handle_commit_with_author(
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
        })
    }

    /// Handle stage all including untracked operation
    fn handle_stage_all_including_untracked(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::stage_all_with_untracked(&repo)
        })
    }

    // ========================================================================
    // Merge Conflict Resolution
    // ========================================================================

    /// Handle abort merge operation
    fn handle_abort_merge(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::abort_merge(&repo)
        })
    }

    /// Handle continue merge operation
    fn handle_continue_merge(&self, repo_id: &str, message: Option<&str>) -> Result<String> {
        let id = Self::parse_repo_id(repo_id)?;
        let message = message.map(std::borrow::ToOwned::to_owned);
        self.with_repo_blocking(&id, move |repo| {
            Ok(crate::git::ops::continue_merge(&repo, message.as_deref())?.to_string())
        })
    }

    /// Handle quit merge operation
    fn handle_quit_merge(&self, repo_id: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::quit_merge(&repo)
        })
    }

    // ========================================================================
    // Tag Operations
    // ========================================================================

    /// Handle list tags operation
    fn handle_list_tags(&self, repo_id: &str) -> Result<Vec<String>> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::list_tags(&repo)
        })
    }

    /// Handle create tag operation
    fn handle_create_tag(&self, repo_id: &str, name: &str, target: Option<&str>) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        let target = target.map(std::borrow::ToOwned::to_owned);
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::create_tag(&repo, &name, target.as_deref(), false)
        })
    }

    /// Handle delete tag operation
    fn handle_delete_tag(&self, repo_id: &str, name: &str) -> Result<()> {
        let id = Self::parse_repo_id(repo_id)?;
        let name = name.to_owned();
        self.with_repo_blocking(&id, move |repo| {
            crate::git::ops::delete_tag(&repo, &name)
        })
    }

    // ========================================================================
    // Detailed Status
    // ========================================================================

    /// Handle detailed status operation
    fn handle_detailed_status(&self, repo_id: &str) -> Result<DetailedStatusData> {
        let id = Self::parse_repo_id(repo_id)?;
        self.with_repo_blocking(&id, |repo| {
            crate::git::ops::detailed_status(&repo)
        })
    }

    // ========================================================================
    // Filesystem Operations
    // ========================================================================

    /// Get or create a ContainedRoot for a (repo_id, worktree) pair.
    fn get_contained_root(
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

        let worktree_path = self.runtime_handle.block_on(async {
            let registry = self.registry.read().await;
            let handle = registry.repo(&id)
                .map_err(|e| FsServiceError::NotFound(e.to_string()))?;
            let worktrees = handle.get_worktrees().await
                .map_err(|e| FsServiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;

            // Find the worktree matching the requested name
            for wt in &worktrees {
                let wt_name = wt.path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                if wt_name == worktree {
                    return Ok(wt.path().to_path_buf());
                }
            }

            // If worktree name is "." or empty, use the repo's base worktree path
            if worktree == "." || worktree.is_empty() {
                let repo = handle.open_repo()
                    .map_err(|e| FsServiceError::NotFound(e.to_string()))?;
                if let Some(wdir) = repo.workdir() {
                    return Ok(wdir.to_path_buf());
                }
            }

            Err(FsServiceError::NotFound(
                format!("worktree '{}' not found for repo '{}'", worktree, repo_id),
            ))
        })?;

        let root = contained_root::open_contained_root(&worktree_path)?;
        let root: Arc<dyn ContainedRoot> = Arc::from(root);
        self.contained_roots.insert(key, Arc::clone(&root));
        Ok(root)
    }

    /// Handle FS open
    fn handle_fs_open(
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
        let root = self.get_contained_root(repo_id, worktree)?;
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
    fn handle_fs_close(&self, ctx: &EnvelopeContext, fd: u32) -> Result<(), FsServiceError> {
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
    fn handle_fs_read(
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
    fn handle_fs_write(
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
    fn handle_fs_pread(
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
    fn handle_fs_pwrite(
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
    fn handle_fs_seek(
        &self,
        ctx: &EnvelopeContext,
        fd: u32,
        offset: i64,
        whence: crate::services::traits::SeekWhence,
    ) -> Result<u64, FsServiceError> {
        let entry = self.fd_table.get_verified(fd, &ctx.subject().to_string())?;
        let mut file = entry.file.lock();
        let seek_from = match whence {
            crate::services::traits::SeekWhence::Set => SeekFrom::Start(offset as u64),
            crate::services::traits::SeekWhence::Cur => SeekFrom::Current(offset),
            crate::services::traits::SeekWhence::End => SeekFrom::End(offset),
        };
        let pos = file.seek(seek_from).map_err(FsServiceError::Io)?;
        Ok(pos)
    }

    /// Handle FS truncate
    fn handle_fs_truncate(
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
    fn handle_fs_fsync(
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
    fn handle_fs_stat(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<FsStatInfo, FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        match root.stat(path) {
            Ok(meta) => {
                let modified_at = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                Ok(FsStatInfo {
                    exists: true,
                    is_dir: meta.is_dir(),
                    size: meta.len(),
                    modified_at,
                })
            }
            Err(FsServiceError::NotFound(_)) | Err(FsServiceError::Io(_)) => {
                // For stat, NotFound is not an error — it means the file doesn't exist
                Ok(FsStatInfo {
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
    fn handle_fs_mkdir(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
        recursive: bool,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        if recursive {
            root.mkdir_all(path)
        } else {
            root.mkdir(path)
        }
    }

    /// Handle FS remove file
    fn handle_fs_remove(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        root.remove_file(path)
    }

    /// Handle FS rmdir
    fn handle_fs_rmdir(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        root.remove_dir(path)
    }

    /// Handle FS rename
    fn handle_fs_rename(
        &self,
        repo_id: &str,
        worktree: &str,
        src: &str,
        dst: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        root.rename(src, dst)
    }

    /// Handle FS copy
    fn handle_fs_copy(
        &self,
        repo_id: &str,
        worktree: &str,
        src: &str,
        dst: &str,
    ) -> Result<(), FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        root.copy_file(src, dst)
    }

    /// Handle FS list directory
    fn handle_fs_list_dir(
        &self,
        repo_id: &str,
        worktree: &str,
        path: &str,
    ) -> Result<Vec<FsDirEntry>, FsServiceError> {
        let root = self.get_contained_root(repo_id, worktree)?;
        root.list_dir(path)
    }

}

impl ZmqService for RegistryService {
    fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<Vec<u8>> {
        // Log identity for audit trail
        debug!(
            "Registry request from {} (envelope_id={}, authenticated={})",
            ctx.subject(),
            ctx.request_id,
            ctx.is_authenticated()
        );

        // Deserialize inner request from payload
        let reader = serialize::read_message(payload, ReaderOptions::new())?;
        let req = reader.get_root::<registry_capnp::registry_request::Reader>()?;

        let request_id = req.get_id();

        // Handle based on request type
        use registry_capnp::registry_request::Which;
        use crate::services::rpc_types::RepoVoidVariant;

        match req.which()? {
            Which::List(()) => {
                if let Some(resp) = self.check_auth(ctx, request_id, "registry", Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_list() {
                    Ok(repos) => Ok(RegistryResponse::list_result(request_id, &repos)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Get(id) => {
                let id_str = id?.to_str()?;
                let resource = format!("registry:{id_str}");
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_get(id_str) {
                    Ok(Some(repo)) => Ok(RegistryResponse::get_result(request_id, &repo)),
                    Ok(None) => Ok(RegistryResponse::error(request_id, "Repository not found")),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::GetByName(name) => {
                let name_str = name?.to_str()?;
                let resource = format!("registry:{name_str}");
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                    return Ok(resp);
                }
                match self.handle_get_by_name(name_str) {
                    Ok(Some(repo)) => Ok(RegistryResponse::get_by_name_result(request_id, &repo)),
                    Ok(None) => Ok(RegistryResponse::error(request_id, "Repository not found")),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Clone(clone_req) => {
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
                    Ok(repo) => Ok(RegistryResponse::clone_result(request_id, &repo)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::CloneStream(clone_req) => {
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

                let client_ephemeral_pubkey = ctx.ephemeral_pubkey();

                match self.prepare_clone_stream(url, name, shallow, Some(depth), branch, client_ephemeral_pubkey) {
                    Ok((stream_id, server_pubkey)) => {
                        let stream_endpoint = endpoint_registry()
                            .endpoint("streams", SocketKind::Sub)
                            .to_zmq_string();

                        Ok(RegistryResponse::clone_stream_result(
                            request_id,
                            &stream_id,
                            &stream_endpoint,
                            &server_pubkey,
                        ))
                    }
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Register(reg_req) => {
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
                    Ok(repo) => Ok(RegistryResponse::register_result(request_id, &repo)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::Remove(repo_id) => {
                let repo_id = repo_id?.to_str()?;
                let resource = format!("registry:{repo_id}");
                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Manage) {
                    return Ok(resp);
                }
                match self.handle_remove(repo_id) {
                    Ok(()) => Ok(RegistryResponse::remove_result(request_id)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            Which::HealthCheck(()) => {
                match self.handle_health_check() {
                    Ok(status) => Ok(RegistryResponse::health_check_result(request_id, &status)),
                    Err(e) => Ok(RegistryResponse::error(request_id, &e.to_string())),
                }
            }

            // Repository-scoped operations (nested under repo)
            Which::Repo(repo_req) => {
                let repo_req = repo_req?;
                let repo_id = repo_req.get_repo_id()?.to_str()?;
                let resource = format!("registry:{repo_id}");

                use registry_capnp::repository_request::Which as RepoWhich;
                match repo_req.which()? {
                    RepoWhich::CreateWorktree(wt_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let wt_req = wt_req?;
                        let path = wt_req.get_path()?.to_str()?;
                        let branch = wt_req.get_branch_name()?.to_str()?;
                        match self.handle_create_worktree(repo_id, path, branch) {
                            Ok(wt_path) => Ok(RegistryResponse::repo_create_worktree(request_id, &wt_path)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::ListWorktrees(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_list_worktrees(repo_id) {
                            Ok(worktrees) => Ok(RegistryResponse::repo_list_worktrees(request_id, &worktrees)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::RemoveWorktree(wt_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Manage) {
                            return Ok(resp);
                        }
                        let wt_req = wt_req?;
                        let worktree_path = wt_req.get_worktree_path()?.to_str()?;
                        match self.handle_remove_worktree(repo_id, worktree_path) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::RemoveWorktree)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::CreateBranch(branch_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let branch_req = branch_req?;
                        let branch_name = branch_req.get_branch_name()?.to_str()?;
                        let start_point = if branch_req.has_start_point() {
                            Some(branch_req.get_start_point()?.to_str()?)
                        } else {
                            None
                        };
                        match self.handle_create_branch(repo_id, branch_name, start_point) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::CreateBranch)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::ListBranches(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_list_branches(repo_id) {
                            Ok(branches) => Ok(RegistryResponse::repo_list_branches(request_id, &branches)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::Checkout(checkout_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let checkout_req = checkout_req?;
                        let ref_name = checkout_req.get_ref_name()?.to_str()?;
                        match self.handle_checkout(repo_id, ref_name) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::Checkout)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::StageAll(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        match self.handle_stage_all(repo_id) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::StageAll)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::StageFiles(stage_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let stage_req = stage_req?;
                        let files_reader = stage_req.get_files()?;
                        let mut files = Vec::new();
                        for file in files_reader.iter() {
                            files.push(file?.to_str()?.to_owned());
                        }
                        match self.handle_stage_files(repo_id, files) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::StageFiles)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::Commit(commit_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let commit_req = commit_req?;
                        let message = commit_req.get_message()?.to_str()?;
                        match self.handle_commit(repo_id, message) {
                            Ok(oid) => Ok(RegistryResponse::repo_commit(request_id, &oid)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::Merge(merge_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let merge_req = merge_req?;
                        let source = merge_req.get_source()?.to_str()?;
                        let message = if merge_req.has_message() {
                            Some(merge_req.get_message()?.to_str()?.to_owned())
                        } else {
                            None
                        };
                        match self.handle_merge(repo_id, source, message.as_deref()) {
                            Ok(_oid) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::Merge)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::AbortMerge(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        match self.handle_abort_merge(repo_id) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::AbortMerge)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::ContinueMerge(merge_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let merge_req = merge_req?;
                        let message = if merge_req.has_message() {
                            let m = merge_req.get_message()?.to_str()?;
                            if m.is_empty() { None } else { Some(m) }
                        } else {
                            None
                        };
                        match self.handle_continue_merge(repo_id, message) {
                            Ok(_oid) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::ContinueMerge)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::QuitMerge(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        match self.handle_quit_merge(repo_id) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::QuitMerge)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::GetHead(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_get_head(repo_id) {
                            Ok(oid) => Ok(RegistryResponse::repo_get_head(request_id, &oid)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::GetRef(ref_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        let ref_req = ref_req?;
                        let ref_name = ref_req.get_ref_name()?.to_str()?;
                        match self.handle_get_ref(repo_id, ref_name) {
                            Ok(oid) => Ok(RegistryResponse::repo_get_ref(request_id, &oid)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::Status(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_status(repo_id) {
                            Ok(status) => Ok(RegistryResponse::repo_status(request_id, &status)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::DetailedStatus(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_detailed_status(repo_id) {
                            Ok(status) => Ok(RegistryResponse::repo_detailed_status(request_id, &status)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::ListRemotes(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_list_remotes(repo_id) {
                            Ok(remotes) => Ok(RegistryResponse::repo_list_remotes(request_id, &remotes)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::AddRemote(remote_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let remote_req = remote_req?;
                        let name = remote_req.get_name()?.to_str()?;
                        let url = remote_req.get_url()?.to_str()?;
                        match self.handle_add_remote(repo_id, name, url) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::AddRemote)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::RemoveRemote(remote_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let remote_req = remote_req?;
                        let name = remote_req.get_name()?.to_str()?;
                        match self.handle_remove_remote(repo_id, name) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::RemoveRemote)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::SetRemoteUrl(remote_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let remote_req = remote_req?;
                        let name = remote_req.get_name()?.to_str()?;
                        let url = remote_req.get_url()?.to_str()?;
                        match self.handle_set_remote_url(repo_id, name, url) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::SetRemoteUrl)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::RenameRemote(remote_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let remote_req = remote_req?;
                        let old_name = remote_req.get_old_name()?.to_str()?;
                        let new_name = remote_req.get_new_name()?.to_str()?;
                        match self.handle_rename_remote(repo_id, old_name, new_name) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::RenameRemote)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::Push(push_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let push_req = push_req?;
                        let remote = push_req.get_remote()?.to_str()?;
                        let refspec = push_req.get_refspec()?.to_str()?;
                        let force = push_req.get_force();
                        match self.handle_push(repo_id, remote, refspec, force) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::Push)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::AmendCommit(amend_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let amend_req = amend_req?;
                        let message = amend_req.get_message()?.to_str()?;
                        match self.handle_amend_commit(repo_id, message) {
                            Ok(oid) => Ok(RegistryResponse::repo_amend_commit(request_id, &oid)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::CommitWithAuthor(commit_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let commit_req = commit_req?;
                        let message = commit_req.get_message()?.to_str()?;
                        let author_name = commit_req.get_author_name()?.to_str()?;
                        let author_email = commit_req.get_author_email()?.to_str()?;
                        match self.handle_commit_with_author(repo_id, message, author_name, author_email) {
                            Ok(oid) => Ok(RegistryResponse::repo_commit_with_author(request_id, &oid)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::StageAllIncludingUntracked(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        match self.handle_stage_all_including_untracked(repo_id) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::StageAllIncludingUntracked)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::ListTags(()) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                            return Ok(resp);
                        }
                        match self.handle_list_tags(repo_id) {
                            Ok(tags) => Ok(RegistryResponse::repo_list_tags(request_id, &tags)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::CreateTag(tag_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let tag_req = tag_req?;
                        let name = tag_req.get_name()?.to_str()?;
                        let target = if tag_req.has_target() {
                            let t = tag_req.get_target()?.to_str()?;
                            if t.is_empty() { None } else { Some(t) }
                        } else {
                            None
                        };
                        match self.handle_create_tag(repo_id, name, target) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::CreateTag)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::DeleteTag(tag_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let tag_req = tag_req?;
                        let name = tag_req.get_name()?.to_str()?;
                        match self.handle_delete_tag(repo_id, name) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::DeleteTag)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }
                    RepoWhich::Update(update_req) => {
                        if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                            return Ok(resp);
                        }
                        let update_req = update_req?;
                        let refspec = if update_req.has_refspec() {
                            let r = update_req.get_refspec()?.to_str()?;
                            if r.is_empty() { None } else { Some(r) }
                        } else {
                            None
                        };
                        match self.handle_update(repo_id, refspec) {
                            Ok(()) => Ok(RegistryResponse::repo_void(request_id, RepoVoidVariant::Update)),
                            Err(e) => Ok(RegistryResponse::repo_error(request_id, &e.to_string())),
                        }
                    }

                    // Worktree-scoped filesystem operations (nested under repo → worktree)
                    RepoWhich::Worktree(wt_req) => {
                        let wt_req = wt_req?;
                        let worktree = wt_req.get_name()?.to_str()?;

                        use registry_capnp::worktree_request::Which as WtWhich;
                        match wt_req.which()? {
                            WtWhich::Open(open_req) => {
                                let open_req = open_req?;
                                let path = open_req.get_path()?.to_str()?;
                                let write = open_req.get_write();
                                // Require Write auth if opening for write, else Query
                                let op = if write { Operation::Write } else { Operation::Query };
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, op) {
                                    return Ok(resp);
                                }
                                match self.handle_fs_open(
                                    ctx, repo_id, worktree, path,
                                    open_req.get_read(), write,
                                    open_req.get_create(), open_req.get_truncate(),
                                    open_req.get_append(), open_req.get_exclusive(),
                                ) {
                                    Ok(fd) => Ok(RegistryResponse::fs_open(request_id, fd)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Close(close_req) => {
                                let close_req = close_req?;
                                match self.handle_fs_close(ctx, close_req.get_fd()) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Close)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Read(read_req) => {
                                let read_req = read_req?;
                                match self.handle_fs_read(ctx, read_req.get_fd(), read_req.get_length()) {
                                    Ok(data) => Ok(RegistryResponse::fs_read(request_id, &data, false)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Write(write_req) => {
                                let write_req = write_req?;
                                let data = write_req.get_data()?;
                                match self.handle_fs_write(ctx, write_req.get_fd(), data) {
                                    Ok(n) => Ok(RegistryResponse::fs_write(request_id, n, false)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Pread(pread_req) => {
                                let pread_req = pread_req?;
                                match self.handle_fs_pread(
                                    ctx, pread_req.get_fd(),
                                    pread_req.get_offset(), pread_req.get_length(),
                                ) {
                                    Ok(data) => Ok(RegistryResponse::fs_read(request_id, &data, true)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Pwrite(pwrite_req) => {
                                let pwrite_req = pwrite_req?;
                                let data = pwrite_req.get_data()?;
                                match self.handle_fs_pwrite(
                                    ctx, pwrite_req.get_fd(),
                                    pwrite_req.get_offset(), data,
                                ) {
                                    Ok(n) => Ok(RegistryResponse::fs_write(request_id, n, true)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Seek(seek_req) => {
                                let seek_req = seek_req?;
                                use registry_capnp::SeekWhence as CapnpWhence;
                                let whence = match seek_req.get_whence()? {
                                    CapnpWhence::Set => crate::services::traits::SeekWhence::Set,
                                    CapnpWhence::Cur => crate::services::traits::SeekWhence::Cur,
                                    CapnpWhence::End => crate::services::traits::SeekWhence::End,
                                };
                                match self.handle_fs_seek(ctx, seek_req.get_fd(), seek_req.get_offset(), whence) {
                                    Ok(pos) => Ok(RegistryResponse::fs_seek(request_id, pos)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Truncate(trunc_req) => {
                                let trunc_req = trunc_req?;
                                match self.handle_fs_truncate(ctx, trunc_req.get_fd(), trunc_req.get_length()) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Truncate)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Fsync(fsync_req) => {
                                let fsync_req = fsync_req?;
                                match self.handle_fs_fsync(ctx, fsync_req.get_fd(), fsync_req.get_data_only()) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Fsync)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Stat(stat_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                                    return Ok(resp);
                                }
                                let stat_req = stat_req?;
                                let path = stat_req.get_path()?.to_str()?;
                                match self.handle_fs_stat(repo_id, worktree, path) {
                                    Ok(stat) => Ok(RegistryResponse::fs_stat(request_id, &stat)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Mkdir(mkdir_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                                    return Ok(resp);
                                }
                                let mkdir_req = mkdir_req?;
                                let path = mkdir_req.get_path()?.to_str()?;
                                let recursive = mkdir_req.get_recursive();
                                match self.handle_fs_mkdir(repo_id, worktree, path, recursive) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Mkdir)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Remove(remove_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                                    return Ok(resp);
                                }
                                let remove_req = remove_req?;
                                let path = remove_req.get_path()?.to_str()?;
                                match self.handle_fs_remove(repo_id, worktree, path) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Remove)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Rmdir(rmdir_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                                    return Ok(resp);
                                }
                                let rmdir_req = rmdir_req?;
                                let path = rmdir_req.get_path()?.to_str()?;
                                match self.handle_fs_rmdir(repo_id, worktree, path) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Rmdir)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Rename(rename_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                                    return Ok(resp);
                                }
                                let rename_req = rename_req?;
                                let src = rename_req.get_src()?.to_str()?;
                                let dst = rename_req.get_dst()?.to_str()?;
                                match self.handle_fs_rename(repo_id, worktree, src, dst) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Rename)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::Copy(copy_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Write) {
                                    return Ok(resp);
                                }
                                let copy_req = copy_req?;
                                let src = copy_req.get_src()?.to_str()?;
                                let dst = copy_req.get_dst()?.to_str()?;
                                match self.handle_fs_copy(repo_id, worktree, src, dst) {
                                    Ok(()) => Ok(RegistryResponse::fs_void(request_id, FsVoidVariant::Copy)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::ListDir(list_req) => {
                                if let Some(resp) = self.check_auth(ctx, request_id, &resource, Operation::Query) {
                                    return Ok(resp);
                                }
                                let list_req = list_req?;
                                let path = list_req.get_path()?.to_str()?;
                                match self.handle_fs_list_dir(repo_id, worktree, path) {
                                    Ok(entries) => Ok(RegistryResponse::fs_list_dir(request_id, &entries)),
                                    Err(e) => Ok(RegistryResponse::fs_error(request_id, &e.to_string())),
                                }
                            }
                            WtWhich::OpenStream(_) | WtWhich::StartStream(_) => {
                                // Streaming operations are not yet implemented
                                Ok(RegistryResponse::fs_error(request_id, "FS streaming not yet implemented"))
                            }
                        }
                    }
                }
            }
        }
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
        self.get_by_name_internal(name)
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
// Repository-Level Client (Adapter over Generated Client)
// ============================================================================

/// ZMQ client for repository-level operations.
///
/// Wraps the auto-generated `RepositoryClient` from build.rs and implements
/// the `RepositoryClient` trait with domain type conversions.
pub struct RepositoryZmqClient {
    gen: crate::services::generated::registry_client::RepositoryClient,
    repo_id: RepoId,
    repo_name: String,
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
        use crate::services::core::ZmqClientBase;
        use crate::zmq::global_context;
        use hyprstream_rpc::service::factory::ServiceClient;
        let server_verifying_key = signing_key.verifying_key();
        let zmq_client = ZmqClientBase::new(&endpoint, global_context(), signing_key, server_verifying_key, identity);
        let registry_client =
            crate::services::generated::registry_client::RegistryClient::from_zmq(zmq_client);
        let gen = registry_client.repo(&repo_id.to_string());
        Self {
            gen,
            repo_id,
            repo_name,
        }
    }

    /// Convert anyhow::Error to RegistryServiceError.
    fn transport_err(e: anyhow::Error) -> RegistryServiceError {
        RegistryServiceError::transport(e.to_string())
    }
}

/// Parse a single-character file change type.
fn parse_file_change_type(s: &str) -> Option<FileChangeType> {
    match s.chars().next()? {
        'A' => Some(FileChangeType::Added),
        'M' => Some(FileChangeType::Modified),
        'D' => Some(FileChangeType::Deleted),
        'R' => Some(FileChangeType::Renamed),
        '?' => Some(FileChangeType::Untracked),
        'T' => Some(FileChangeType::TypeChanged),
        'U' => Some(FileChangeType::Conflicted),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// WorktreeZmqClient — wraps generated WorktreeClient, implements FsOps trait
// ---------------------------------------------------------------------------

use crate::services::traits::FsOps;

/// ZMQ client for worktree-scoped filesystem operations.
///
/// Wraps the auto-generated `WorktreeClient` from the proc macro and implements
/// the `FsOps` trait with domain type conversions.
pub struct WorktreeZmqClient {
    gen: crate::services::generated::registry_client::WorktreeClient,
}

impl WorktreeZmqClient {
    fn transport_err(e: anyhow::Error) -> FsServiceError {
        FsServiceError::Transport(e.to_string())
    }
}

#[async_trait]
impl FsOps for WorktreeZmqClient {
    async fn open(
        &self,
        path: &str,
        write: bool,
        create: bool,
        truncate: bool,
    ) -> Result<u32, FsServiceError> {
        // Map simplified FsOps params to full schema fields:
        // read=true unless write-only, append=false, exclusive=false
        let r = self.gen.open(path, true, write, create, truncate, false, false)
            .await
            .map_err(Self::transport_err)?;
        Ok(r.fd)
    }

    async fn close(&self, fd: u32) -> Result<(), FsServiceError> {
        self.gen.close(fd).await.map_err(Self::transport_err)
    }

    async fn read(&self, fd: u32, len: u64) -> Result<Vec<u8>, FsServiceError> {
        let r = self.gen.read(fd, len).await.map_err(Self::transport_err)?;
        Ok(r.data)
    }

    async fn write(&self, fd: u32, data: &[u8]) -> Result<u64, FsServiceError> {
        let r = self.gen.write(fd, data).await.map_err(Self::transport_err)?;
        Ok(r.bytes_written)
    }

    async fn pread(&self, fd: u32, offset: u64, len: u64) -> Result<Vec<u8>, FsServiceError> {
        let r = self.gen.pread(fd, offset, len).await.map_err(Self::transport_err)?;
        Ok(r.data)
    }

    async fn pwrite(&self, fd: u32, offset: u64, data: &[u8]) -> Result<u64, FsServiceError> {
        let r = self.gen.pwrite(fd, offset, data).await.map_err(Self::transport_err)?;
        Ok(r.bytes_written)
    }

    async fn seek(&self, fd: u32, offset: i64, whence: SeekWhence) -> Result<u64, FsServiceError> {
        let whence_str = match whence {
            SeekWhence::Set => "set",
            SeekWhence::Cur => "cur",
            SeekWhence::End => "end",
        };
        let r = self.gen.seek(fd, offset, whence_str).await.map_err(Self::transport_err)?;
        Ok(r.position)
    }

    async fn truncate(&self, fd: u32, len: u64) -> Result<(), FsServiceError> {
        self.gen.truncate(fd, len).await.map_err(Self::transport_err)
    }

    async fn fsync(&self, fd: u32, data_only: bool) -> Result<(), FsServiceError> {
        self.gen.fsync(fd, data_only).await.map_err(Self::transport_err)
    }

    async fn stat(&self, path: &str) -> Result<FsStatInfo, FsServiceError> {
        let r = self.gen.stat(path).await.map_err(Self::transport_err)?;
        Ok(FsStatInfo {
            exists: r.exists,
            is_dir: r.is_dir,
            size: r.size,
            modified_at: r.modified_at,
        })
    }

    async fn mkdir(&self, path: &str, recursive: bool) -> Result<(), FsServiceError> {
        self.gen.mkdir(path, recursive).await.map_err(Self::transport_err)
    }

    async fn remove(&self, path: &str) -> Result<(), FsServiceError> {
        self.gen.remove(path).await.map_err(Self::transport_err)
    }

    async fn rmdir(&self, path: &str) -> Result<(), FsServiceError> {
        self.gen.rmdir(path).await.map_err(Self::transport_err)
    }

    async fn rename(&self, src: &str, dst: &str) -> Result<(), FsServiceError> {
        self.gen.rename(src, dst).await.map_err(Self::transport_err)
    }

    async fn copy(&self, src: &str, dst: &str) -> Result<(), FsServiceError> {
        self.gen.copy(src, dst).await.map_err(Self::transport_err)
    }

    async fn list_dir(&self, path: &str) -> Result<Vec<FsDirEntry>, FsServiceError> {
        let entries = self.gen.list_dir(path).await.map_err(Self::transport_err)?;
        Ok(entries
            .into_iter()
            .map(|e| FsDirEntry {
                name: e.name,
                is_dir: e.is_dir,
                size: e.size,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Trait implementation — delegates to generated client
// ---------------------------------------------------------------------------

/// Helper macro: delegate to generated typed client method.
/// The generated method already returns Result<T> with Error variant handling.
macro_rules! delegate_void {
    ($self:ident, $method:ident ( $($arg:expr),* ), $variant:ident) => {{
        $self.gen.$method( $($arg),* ).await.map_err(RepositoryZmqClient::transport_err)
    }};
}

/// Helper macro: delegate to generated typed client method returning String.
macro_rules! delegate_string {
    ($self:ident, $method:ident ( $($arg:expr),* ), $variant:ident) => {{
        $self.gen.$method( $($arg),* ).await.map_err(RepositoryZmqClient::transport_err)
    }};
}

/// Helper macro: delegate to generated typed client method returning Vec<String>.
macro_rules! delegate_string_list {
    ($self:ident, $method:ident ( $($arg:expr),* ), $variant:ident) => {{
        $self.gen.$method( $($arg),* ).await.map_err(RepositoryZmqClient::transport_err)
    }};
}

#[async_trait]
impl RepositoryClient for RepositoryZmqClient {
    fn name(&self) -> &str {
        &self.repo_name
    }

    fn id(&self) -> &RepoId {
        &self.repo_id
    }

    // ========================================================================
    // Worktree Operations
    // ========================================================================

    async fn create_worktree(
        &self,
        path: &Path,
        branch: &str,
    ) -> Result<PathBuf, RegistryServiceError> {
        let p = self
            .gen
            .create_worktree(path.to_string_lossy().as_ref(), branch, false)
            .await
            .map_err(Self::transport_err)?;
        Ok(PathBuf::from(p))
    }

    async fn list_worktrees(&self) -> Result<Vec<WorktreeInfo>, RegistryServiceError> {
        let wts = self.gen.list_worktrees().await.map_err(Self::transport_err)?;
        Ok(wts.into_iter().map(|wt| {
            let branch = if wt.branch_name.is_empty() { None } else { Some(wt.branch_name) };
            WorktreeInfo {
                path: PathBuf::from(wt.path),
                branch,
                driver: "zmq".to_owned(),
                is_dirty: wt.is_dirty,
            }
        }).collect())
    }

    async fn worktree_path(
        &self,
        branch: &str,
    ) -> Result<Option<PathBuf>, RegistryServiceError> {
        let worktrees = self.list_worktrees().await?;
        Ok(worktrees
            .into_iter()
            .find(|wt| wt.branch.as_deref() == Some(branch))
            .map(|wt| wt.path))
    }

    async fn remove_worktree(&self, path: &Path) -> Result<(), RegistryServiceError> {
        delegate_void!(
            self,
            remove_worktree(path.to_string_lossy().as_ref(), false),
            RemoveWorktree
        )
    }

    // ========================================================================
    // Branch Operations
    // ========================================================================

    async fn create_branch(
        &self,
        name: &str,
        from: Option<&str>,
    ) -> Result<(), RegistryServiceError> {
        delegate_void!(
            self,
            create_branch(name, from.unwrap_or("")),
            CreateBranch
        )
    }

    async fn checkout(&self, ref_spec: &str) -> Result<(), RegistryServiceError> {
        delegate_void!(self, checkout(ref_spec, false), Checkout)
    }

    async fn default_branch(&self) -> Result<String, RegistryServiceError> {
        // Not yet supported via ZMQ, return "main" as default
        Ok("main".to_owned())
    }

    async fn list_branches(&self) -> Result<Vec<String>, RegistryServiceError> {
        delegate_string_list!(self, list_branches(), ListBranches)
    }

    async fn merge(
        &self,
        source: &str,
        message: Option<&str>,
    ) -> Result<String, RegistryServiceError> {
        // Schema response is Void — return empty string
        delegate_void!(self, merge(source, message.unwrap_or("")), Merge)?;
        Ok(String::new())
    }

    // ========================================================================
    // Staging / Commit Operations
    // ========================================================================

    async fn stage_all(&self) -> Result<(), RegistryServiceError> {
        delegate_void!(self, stage_all(), StageAll)
    }

    async fn stage_files(&self, files: &[&str]) -> Result<(), RegistryServiceError> {
        let owned: Vec<String> = files.iter().map(|s| s.to_string()).collect();
        delegate_void!(self, stage_files(&owned), StageFiles)
    }

    async fn commit(&self, message: &str) -> Result<String, RegistryServiceError> {
        delegate_string!(self, commit(message, "", ""), Commit)
    }

    async fn status(&self) -> Result<RepositoryStatus, RegistryServiceError> {
        let r = self
            .gen
            .status()
            .await
            .map_err(Self::transport_err)?;
        let branch_opt = if r.branch.is_empty() {
            None
        } else {
            Some(r.branch)
        };
        let head = if r.head_oid.is_empty() {
            None
        } else {
            Some(
                git2db::Oid::from_str(&r.head_oid)
                    .map_err(|e| RegistryServiceError::transport(e.to_string()))?,
            )
        };
        Ok(RepositoryStatus {
            branch: branch_opt,
            head,
            ahead: r.ahead as usize,
            behind: r.behind as usize,
            is_clean: r.is_clean,
            modified_files: r.modified_files.into_iter().map(PathBuf::from).collect(),
        })
    }

    // ========================================================================
    // Reference Operations
    // ========================================================================

    async fn get_head(&self) -> Result<String, RegistryServiceError> {
        delegate_string!(self, get_head(), GetHead)
    }

    async fn get_ref(&self, ref_name: &str) -> Result<String, RegistryServiceError> {
        delegate_string!(self, get_ref(ref_name), GetRef)
    }

    async fn update(&self, refspec: Option<&str>) -> Result<(), RegistryServiceError> {
        delegate_void!(self, update(refspec.unwrap_or("")), Update)
    }

    // ========================================================================
    // Remote Operations
    // ========================================================================

    async fn list_remotes(&self) -> Result<Vec<RemoteInfo>, RegistryServiceError> {
        let remotes = self.gen.list_remotes().await.map_err(Self::transport_err)?;
        Ok(remotes.into_iter().map(|r| RemoteInfo {
            name: r.name,
            url: r.url,
            push_url: Some(r.push_url),
        }).collect())
    }

    async fn add_remote(&self, name: &str, url: &str) -> Result<(), RegistryServiceError> {
        delegate_void!(self, add_remote(name, url), AddRemote)
    }

    async fn remove_remote(&self, name: &str) -> Result<(), RegistryServiceError> {
        delegate_void!(self, remove_remote(name), RemoveRemote)
    }

    async fn set_remote_url(&self, name: &str, url: &str) -> Result<(), RegistryServiceError> {
        delegate_void!(self, set_remote_url(name, url), SetRemoteUrl)
    }

    async fn rename_remote(
        &self,
        old_name: &str,
        new_name: &str,
    ) -> Result<(), RegistryServiceError> {
        delegate_void!(self, rename_remote(old_name, new_name), RenameRemote)
    }

    // ========================================================================
    // Push Operations
    // ========================================================================

    async fn push(
        &self,
        remote: &str,
        refspec: &str,
        force: bool,
    ) -> Result<(), RegistryServiceError> {
        delegate_void!(self, push(remote, refspec, force), Push)
    }

    // ========================================================================
    // Advanced Commit Operations
    // ========================================================================

    async fn amend_commit(&self, message: &str) -> Result<String, RegistryServiceError> {
        delegate_string!(self, amend_commit(message), AmendCommit)
    }

    async fn commit_with_author(
        &self,
        message: &str,
        author_name: &str,
        author_email: &str,
    ) -> Result<String, RegistryServiceError> {
        delegate_string!(
            self,
            commit_with_author(message, author_name, author_email),
            CommitWithAuthor
        )
    }

    async fn stage_all_including_untracked(&self) -> Result<(), RegistryServiceError> {
        delegate_void!(
            self,
            stage_all_including_untracked(),
            StageAllIncludingUntracked
        )
    }

    // ========================================================================
    // Merge Conflict Resolution
    // ========================================================================

    async fn abort_merge(&self) -> Result<(), RegistryServiceError> {
        delegate_void!(self, abort_merge(), AbortMerge)
    }

    async fn continue_merge(
        &self,
        message: Option<&str>,
    ) -> Result<String, RegistryServiceError> {
        // Schema response is Void — return empty string
        delegate_void!(
            self,
            continue_merge(message.unwrap_or("")),
            ContinueMerge
        )?;
        Ok(String::new())
    }

    async fn quit_merge(&self) -> Result<(), RegistryServiceError> {
        delegate_void!(self, quit_merge(), QuitMerge)
    }

    // ========================================================================
    // Tag Operations
    // ========================================================================

    async fn list_tags(&self) -> Result<Vec<String>, RegistryServiceError> {
        delegate_string_list!(self, list_tags(), ListTags)
    }

    async fn create_tag(
        &self,
        name: &str,
        target: Option<&str>,
    ) -> Result<(), RegistryServiceError> {
        delegate_void!(self, create_tag(name, target.unwrap_or("")), CreateTag)
    }

    async fn delete_tag(&self, name: &str) -> Result<(), RegistryServiceError> {
        delegate_void!(self, delete_tag(name), DeleteTag)
    }

    // ========================================================================
    // Detailed Status
    // ========================================================================

    async fn detailed_status(&self) -> Result<DetailedStatus, RegistryServiceError> {
        let r = self.gen.detailed_status().await.map_err(Self::transport_err)?;
        let branch = if r.branch.is_empty() { None } else { Some(r.branch) };
        let head = if r.head_oid.is_empty() { None } else { Some(r.head_oid) };
        let files = r.files.into_iter().map(|f| {
            FileStatus {
                path: f.path,
                index_status: if f.index_status.is_empty() { None } else { parse_file_change_type(&f.index_status) },
                worktree_status: if f.worktree_status.is_empty() { None } else { parse_file_change_type(&f.worktree_status) },
            }
        }).collect();
        Ok(DetailedStatus {
            branch,
            head,
            merge_in_progress: r.merge_in_progress,
            rebase_in_progress: r.rebase_in_progress,
            files,
            ahead: r.ahead as usize,
            behind: r.behind as usize,
        })
    }

    // ========================================================================
    // Filesystem Operations
    // ========================================================================

    fn worktree(&self, name: &str) -> Arc<dyn FsOps> {
        Arc::new(WorktreeZmqClient {
            gen: self.gen.worktree(name),
        })
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
        let policy_client = PolicyClient::with_endpoint(
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
        let client = RegistryZmqClient::with_endpoint(
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
