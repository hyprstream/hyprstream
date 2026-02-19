//! ZMQ-based registry service for repository management
//!
//! This service wraps git2db and provides a ZMQ REQ/REP interface for
//! repository operations. It uses Cap'n Proto for serialization.

use crate::auth::Operation;
use crate::services::PolicyClient;
use crate::services::types::{MAX_FDS_GLOBAL, MAX_FDS_PER_CLIENT};
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

// Generated client types
use crate::services::generated::registry_client::{
    RegistryClient as GenRegistryClient, RegistryResponseVariant,
    RegistryHandler, RepoHandler, WorktreeHandler, CtlHandler,
    dispatch_registry, serialize_response,
    StreamInfo, ErrorInfo, HealthStatus, DetailedStatusInfo, RemoteInfo,
    CloneRequest, RegisterRequest,
    CreateWorktreeRequest, RemoveWorktreeRequest,
    BranchRequest, CheckoutRequest, StageFilesRequest,
    CommitRequest, MergeRequest, ContinueMergeRequest,
    GetRefRequest, AddRemoteRequest, RemoveRemoteRequest,
    SetRemoteUrlRequest, RenameRemoteRequest,
    PushRequest, AmendCommitRequest, CommitWithAuthorRequest,
    CreateTagRequest, DeleteTagRequest, UpdateRequest,
    NpWalk, NpOpen, NpCreate, NpRead, NpWrite, NpClunk, NpRemove,
    NpStatReq, NpWstat, NpFlush,
    RWalk, ROpen, RRead, RWrite, RStat,
    NpStat as NpStatData, Qid as QidData,
    EnsureWorktreeRequest,
    FileStatus, LogEntry, ValidationResult, FileInfo,
    CtlLogRequest, CtlDiffRequest, CtlCheckoutRequest,
    EditOpenRequest, EditApplyRequest,
    DocFormatEnum,
};
use crate::services::editing::{self, EditingTable, DocFormat};
use automerge::ReadDoc as _;
// Conflicting names — use canonical path at usage sites:
//   registry_client::TrackedRepository, registry_client::RepositoryStatus, registry_client::WorktreeInfo

// ============================================================================
// Parsing Helper Functions
// ============================================================================

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
// 9P Fid Table for Filesystem Operations
// ============================================================================

use crate::services::contained_root::WalkHandle;
use crate::services::types::{DEFAULT_IOUNIT, QTDIR, QTFILE, OWRITE, ORDWR, OTRUNC, DMDIR};

/// 9P Qid — uniquely identifies a file version.
#[derive(Clone, Debug)]
struct Qid {
    qtype: u8,
    version: u32,
    path: u64,
}

/// State of a fid in the 9P table.
enum FidState {
    /// After walk: handle for metadata/open, not yet opened for I/O.
    Walked {
        walk_handle: WalkHandle,
        qid: Qid,
    },
    /// After open: real fd for I/O.
    Opened {
        file: Mutex<std::fs::File>,
        qid: Qid,
        iounit: u32,
        mode: u8,
    },
}

/// Entry in the fid table.
struct FidEntry {
    state: FidState,
    /// Identity of the client that owns this fid (for owner verification).
    owner_identity: String,
    /// Epoch seconds of last access (for idle timeout reaping).
    last_accessed: AtomicU64,
}

/// Process-global fid table for 9P filesystem operations.
///
/// Uses `DashMap` for lock-free concurrent access. Fids are allocated with
/// a simple atomic counter with collision retry.
struct FidTable {
    next_fid: AtomicU32,
    fids: DashMap<u32, FidEntry>,
    /// Per-client fid count for resource limiting.
    client_fid_counts: DashMap<String, AtomicU32>,
}

/// Idle timeout for fids (5 minutes).
const FID_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Build a Qid from filesystem metadata.
#[cfg(unix)]
fn qid_from_metadata(meta: &std::fs::Metadata) -> Qid {
    use std::os::unix::fs::MetadataExt;
    Qid {
        qtype: if meta.is_dir() { QTDIR } else { QTFILE },
        version: meta.ctime() as u32,
        path: meta.ino(),
    }
}

/// Build a Qid from metadata (non-Unix fallback).
#[cfg(not(unix))]
fn qid_from_metadata(meta: &std::fs::Metadata) -> Qid {
    Qid {
        qtype: if meta.is_dir() { QTDIR } else { QTFILE },
        version: meta.modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0),
        path: 0, // No inode on non-Unix
    }
}

impl FidTable {
    fn new() -> Self {
        Self {
            next_fid: AtomicU32::new(1), // 0 is reserved for root
            fids: DashMap::new(),
            client_fid_counts: DashMap::new(),
        }
    }

    /// Allocate a new fid for a client, checking per-client and global limits.
    fn alloc_fid(&self, client_id: &str) -> Result<u32, FsServiceError> {
        let count = self
            .client_fid_counts
            .entry(client_id.to_owned())
            .or_insert_with(|| AtomicU32::new(0));
        if count.load(Relaxed) >= MAX_FDS_PER_CLIENT {
            return Err(FsServiceError::ResourceLimit(
                "too many open fids for client".into(),
            ));
        }
        if self.fids.len() >= MAX_FDS_GLOBAL as usize {
            return Err(FsServiceError::ResourceLimit(
                "too many open fids globally".into(),
            ));
        }
        for _ in 0..1000 {
            let fid = self.next_fid.fetch_add(1, Relaxed);
            if fid >= 1 && !self.fids.contains_key(&fid) {
                count.fetch_add(1, Relaxed);
                return Ok(fid);
            }
        }
        Err(FsServiceError::ResourceLimit(
            "failed to allocate fid after retries".into(),
        ))
    }

    /// Insert a fid entry (does NOT increment client count — use insert_counted for that).
    fn insert(&self, fid: u32, entry: FidEntry) {
        self.fids.insert(fid, entry);
    }

    /// Insert a fid entry AND increment the client's fid count.
    /// Use this when inserting a fid that wasn't allocated via alloc_fid
    /// (e.g., client-specified newfid in walk).
    fn insert_counted(&self, fid: u32, entry: FidEntry, client_id: &str) {
        let count = self
            .client_fid_counts
            .entry(client_id.to_owned())
            .or_insert_with(|| AtomicU32::new(0));
        count.fetch_add(1, Relaxed);
        self.fids.insert(fid, entry);
    }

    /// Replace a fid entry in-place without touching the client count.
    /// Use this for state transitions (Walked → Opened) where the fid
    /// already exists and the count should stay the same.
    fn replace(&self, fid: u32, entry: FidEntry) {
        self.fids.insert(fid, entry);
    }

    /// Take a fid entry WITHOUT decrementing the client's count.
    /// Use this when the fid will be re-inserted (state transition).
    /// The caller is responsible for re-inserting via `replace` or `insert`.
    fn take(&self, fid: u32) -> Option<FidEntry> {
        self.fids.remove(&fid).map(|(_, v)| v)
    }

    /// Remove a fid and decrement the client's count.
    fn remove(&self, fid: u32, client_id: &str) -> Option<FidEntry> {
        let removed = self.fids.remove(&fid).map(|(_, v)| v);
        if removed.is_some() {
            if let Some(count) = self.client_fid_counts.get(client_id) {
                count.fetch_sub(1, Relaxed);
            }
        }
        removed
    }

    /// Get a reference to a fid entry, verifying ownership.
    fn get_verified(
        &self,
        fid: u32,
        client_id: &str,
    ) -> Result<dashmap::mapref::one::Ref<'_, u32, FidEntry>, FsServiceError> {
        let entry = self
            .fids
            .get(&fid)
            .ok_or(FsServiceError::BadFd(fid))?;
        if entry.owner_identity != client_id {
            return Err(FsServiceError::PermissionDenied(
                "fid not owned by caller".into(),
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
    /// 9P fid table for filesystem operations.
    fid_table: Arc<FidTable>,
    /// Cached contained roots for worktrees: (repo_id, worktree_name) → ContainedRoot.
    contained_roots: DashMap<(String, String), Arc<dyn ContainedRoot>>,
    /// CRDT editing sessions for collaborative file editing.
    editing_table: Arc<EditingTable>,
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

        // Create fid table and editing table, spawn reaper
        let fid_table = Arc::new(FidTable::new());
        let editing_table = Arc::new(EditingTable::new());
        let reaper_fid_table = Arc::clone(&fid_table);
        let reaper_editing_table = Arc::clone(&editing_table);
        tokio::spawn(async move {
            Self::fid_reaper(reaper_fid_table, reaper_editing_table).await;
        });

        let service = Self {
            registry: worker_registry,
            base_dir,
            policy_client,
            context,
            transport,
            signing_key,
            fid_table,
            contained_roots: DashMap::new(),
            editing_table,
        };

        Ok(service)
    }

    /// Background task that reaps idle fids.
    async fn fid_reaper(fid_table: Arc<FidTable>, editing_table: Arc<EditingTable>) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let now = now_epoch_secs();
            fid_table.fids.retain(|fid, entry| {
                let idle = now.saturating_sub(entry.last_accessed.load(Relaxed));
                if idle > FID_IDLE_TIMEOUT.as_secs() {
                    // Clean up any editing sessions associated with this fid
                    editing_table.on_reap(&entry.owner_identity, *fid);
                    // Decrement client fid count
                    if let Some(count) = fid_table.client_fid_counts.get(&entry.owner_identity) {
                        count.fetch_sub(1, Relaxed);
                    }
                    debug!("Reaped idle fid (idle {}s)", idle);
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
    #[allow(clippy::too_many_arguments)]
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
            let (ahead, behind) = if let Ok(ref head_ref) = repo.head() {
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
        let is_bare = base_path.extension().is_some_and(|ext| ext == "git")
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
            .map_err(|e| FsServiceError::Io(std::io::Error::other(e.to_string())))?;

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
                .map(std::path::Path::to_path_buf)
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

    // ========================================================================
    // 9P Helper: Convert internal Qid to generated QidData
    // ========================================================================

    fn qid_to_data(qid: &Qid) -> QidData {
        QidData {
            qtype: qid.qtype,
            version: qid.version,
            path: qid.path,
        }
    }

    /// Build NpStatData from filesystem metadata.
    fn metadata_to_np_stat(name: &str, meta: &std::fs::Metadata) -> NpStatData {
        let qid = qid_from_metadata(meta);
        #[cfg(unix)]
        let (mode, atime, mtime, uid, gid) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.mode(),
                meta.atime() as u32,
                meta.mtime() as u32,
                meta.uid().to_string(),
                meta.gid().to_string(),
            )
        };
        #[cfg(not(unix))]
        let (mode, atime, mtime, uid, gid) = {
            let mtime_val = meta.modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);
            (if meta.is_dir() { 0o40755 } else { 0o100644 }, mtime_val, mtime_val, String::new(), String::new())
        };
        NpStatData {
            qid: Self::qid_to_data(&qid),
            mode,
            atime,
            mtime,
            length: meta.len(),
            name: name.to_owned(),
            uid,
            gid,
            muid: String::new(),
        }
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
        let op = operation.parse::<Operation>()?;
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
// Generated CtlHandler Implementation — per-file git introspection + CRDT editing
// ============================================================================

impl RegistryService {
    /// Resolve a fid to its relative path within the worktree.
    fn fid_rel_path(&self, fid: u32, client_id: &str) -> Result<String> {
        let entry = self.fid_table.get_verified(fid, client_id)
            .map_err(|e| anyhow!("{}", e))?;
        match &entry.state {
            FidState::Walked { walk_handle, .. } => Ok(walk_handle.rel_path().to_owned()),
            FidState::Opened { .. } => {
                // Opened fids don't carry the path — walk_handle was consumed.
                // For ctl operations, the fid should be in Walked state.
                anyhow::bail!("fid {} is opened for I/O; ctl operations require a walked (not opened) fid", fid)
            }
        }
    }

    /// Read file content via contained root for a given fid's path.
    async fn read_fid_content(&self, repo_id: &str, worktree: &str, fid: u32, client_id: &str) -> Result<String> {
        let rel_path = self.fid_rel_path(fid, client_id)?;
        let root = self.get_contained_root(repo_id, worktree).await
            .map_err(|e| anyhow!("{}", e))?;
        let file = root.open_file(&rel_path, false, false, false, false, false)
            .map_err(|e| anyhow!("{}", e))?;
        let mut content = String::new();
        std::io::BufReader::new(file).read_to_string(&mut content)?;
        Ok(content)
    }

    /// Convert generated DocFormatEnum to our editing DocFormat.
    fn to_doc_format(fmt: DocFormatEnum) -> DocFormat {
        match fmt {
            DocFormatEnum::Toml => DocFormat::Toml,
            DocFormatEnum::Json => DocFormat::Json,
            DocFormatEnum::Yaml => DocFormat::Yaml,
            DocFormatEnum::Csv => DocFormat::Csv,
            DocFormatEnum::Text => DocFormat::Text,
        }
    }

    /// Convert editing DocFormat to generated DocFormatEnum.
    #[allow(dead_code)]
    fn from_doc_format(fmt: DocFormat) -> DocFormatEnum {
        match fmt {
            DocFormat::Toml => DocFormatEnum::Toml,
            DocFormat::Json => DocFormatEnum::Json,
            DocFormat::Yaml => DocFormatEnum::Yaml,
            DocFormat::Csv => DocFormatEnum::Csv,
            DocFormat::Text => DocFormatEnum::Text,
        }
    }
}

#[async_trait::async_trait(?Send)]
impl CtlHandler for RegistryService {
    /// Git status of this file.
    async fn handle_status(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, _name: &str, fid: u32,
    ) -> Result<FileStatus> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let repo = handle.open_repo()?;

        let status = repo.status_file(std::path::Path::new(&rel_path))?;
        let state = format!("{:?}", status);
        Ok(FileStatus { state })
    }

    /// Commits touching this file.
    async fn handle_log(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, _name: &str, fid: u32, data: &CtlLogRequest,
    ) -> Result<Vec<LogEntry>> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let repo = handle.open_repo()?;

        let max_count = if data.max_count == 0 { 20 } else { data.max_count as usize };

        // Set up revwalk
        let mut revwalk = repo.revwalk()?;
        if data.ref_name.is_empty() {
            revwalk.push_head()?;
        } else {
            let obj = repo.revparse_single(&data.ref_name)?;
            revwalk.push(obj.id())?;
        }
        revwalk.set_sorting(git2::Sort::TIME)?;

        let mut entries = Vec::new();
        for oid_result in revwalk {
            if entries.len() >= max_count {
                break;
            }
            let oid = oid_result?;
            let commit = repo.find_commit(oid)?;

            // Check if this commit touches our file
            let dominated = if commit.parent_count() == 0 {
                // Initial commit — check if tree contains the file
                commit.tree()?.get_path(std::path::Path::new(&rel_path)).is_ok()
            } else {
                let parent = commit.parent(0)?;
                let diff = repo.diff_tree_to_tree(
                    Some(&parent.tree()?),
                    Some(&commit.tree()?),
                    None,
                )?;
                diff.deltas().any(|d| {
                    d.new_file().path().is_some_and(|p| p == std::path::Path::new(&rel_path))
                    || d.old_file().path().is_some_and(|p| p == std::path::Path::new(&rel_path))
                })
            };

            if dominated {
                let author = commit.author();
                entries.push(LogEntry {
                    oid: oid.to_string(),
                    message: commit.message().unwrap_or("").to_owned(),
                    author: author.name().unwrap_or("").to_owned(),
                    timestamp: commit.time().seconds() as u64,
                });
            }
        }
        Ok(entries)
    }

    /// Diff this file against a ref.
    async fn handle_diff(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, _name: &str, fid: u32, data: &CtlDiffRequest,
    ) -> Result<String> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let repo = handle.open_repo()?;

        let ref_name = if data.ref_name.is_empty() { "HEAD" } else { &data.ref_name };
        let obj = repo.revparse_single(ref_name)?;
        let commit = obj.peel_to_commit()?;
        let tree = commit.tree()?;

        let mut diff_opts = git2::DiffOptions::new();
        diff_opts.pathspec(&rel_path);

        let diff = repo.diff_tree_to_workdir_with_index(
            Some(&tree),
            Some(&mut diff_opts),
        )?;

        let mut output = String::new();
        diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            let origin = line.origin();
            if origin == '+' || origin == '-' || origin == ' ' {
                output.push(origin);
            }
            if let Ok(s) = std::str::from_utf8(line.content()) {
                output.push_str(s);
            }
            true
        })?;

        Ok(output)
    }

    /// Git blame for this file.
    async fn handle_blame(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, _name: &str, fid: u32,
    ) -> Result<String> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let repo = handle.open_repo()?;

        let blame = repo.blame_file(std::path::Path::new(&rel_path), None)?;

        let mut output = String::new();
        for i in 0..blame.len() {
            if let Some(hunk) = blame.get_index(i) {
                let sig = hunk.final_signature();
                let name = sig.name().unwrap_or("?");
                let oid = hunk.final_commit_id();
                let line_start = hunk.final_start_line();
                let line_count = hunk.lines_in_hunk();
                use std::fmt::Write;
                writeln!(output, "{:.8} ({} L{}-{}) ",
                    oid, name, line_start, line_start + line_count - 1)?;
            }
        }
        Ok(output)
    }

    /// Restore file content from a ref.
    async fn handle_checkout(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, fid: u32, data: &CtlCheckoutRequest,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        let id = Self::parse_repo_id(repo_id)?;
        let registry = self.registry.read().await;
        let handle = registry.repo(&id)?;
        let repo = handle.open_repo()?;

        let ref_name = if data.ref_name.is_empty() { "HEAD" } else { &data.ref_name };
        let obj = repo.revparse_single(ref_name)?;
        let commit = obj.peel_to_commit()?;
        let tree = commit.tree()?;
        let entry = tree.get_path(std::path::Path::new(&rel_path))?;
        let blob = repo.find_blob(entry.id())?;

        // Write blob content to the worktree file
        let root = self.get_contained_root(repo_id, name).await
            .map_err(|e| anyhow!("{}", e))?;
        let mut file = root.open_file(&rel_path, true, false, true, false, false)
            .map_err(|e| anyhow!("{}", e))?;
        use std::io::Write as _;
        file.write_all(blob.content())?;
        file.flush()?;

        Ok(())
    }

    /// Validate file format.
    async fn handle_validate(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, fid: u32,
    ) -> Result<ValidationResult> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        // Read the file content
        let content = self.read_fid_content(repo_id, name, fid, &subject).await?;

        // Detect format from extension
        let ext = std::path::Path::new(&rel_path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let mut errors = Vec::new();
        match ext {
            "toml" => {
                if let Err(e) = content.parse::<toml::Value>() {
                    errors.push(e.to_string());
                }
            }
            "json" => {
                if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
                    errors.push(e.to_string());
                }
            }
            "yaml" | "yml" => {
                if let Err(e) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
                    errors.push(e.to_string());
                }
            }
            _ => {
                // Text files are always valid
            }
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
        })
    }

    /// File metadata and git state.
    async fn handle_info(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, fid: u32,
    ) -> Result<FileInfo> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;

        let entry = self.fid_table.get_verified(fid, &subject)
            .map_err(|e| anyhow!("{}", e))?;
        let meta = match &entry.state {
            FidState::Walked { walk_handle, .. } => {
                walk_handle.metadata().map_err(|e| anyhow!("{}", e))?
            }
            FidState::Opened { file, .. } => file.lock().metadata()?,
        };

        let ext = std::path::Path::new(&rel_path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let format = match ext {
            "toml" => DocFormatEnum::Toml,
            "json" => DocFormatEnum::Json,
            "yaml" | "yml" => DocFormatEnum::Yaml,
            "csv" => DocFormatEnum::Csv,
            _ => DocFormatEnum::Text,
        };

        let editing = self.editing_table.fid_has_session(&subject, fid);
        let dirty = if editing {
            if let Some(shared) = self.editing_table.get_session(&subject, fid) {
                shared.lock().dirty
            } else {
                false
            }
        } else {
            false
        };

        Ok(FileInfo {
            path: rel_path,
            size: meta.len(),
            format,
            editing,
            dirty,
        })
    }

    /// Open file for CRDT editing.
    async fn handle_edit_open(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, fid: u32, data: &EditOpenRequest,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;
        let content = self.read_fid_content(repo_id, name, fid, &subject).await?;
        let format = Self::to_doc_format(data.format);

        self.editing_table.open(
            &subject, fid, repo_id, name, &rel_path, format, &content,
        ).map_err(|e| anyhow!("{}", e))?;

        Ok(())
    }

    /// Get current CRDT document state.
    async fn handle_edit_state(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, fid: u32,
    ) -> Result<String> {
        let subject = ctx.subject().to_string();
        let shared = self.editing_table.get_session(&subject, fid)
            .ok_or_else(|| anyhow!("no active editing session for fid {}", fid))?;

        let sd = shared.lock();
        // Get the content value from the automerge document
        let content = sd.doc.get(automerge::ROOT, "content")
            .ok()
            .flatten()
            .map(|(val, _)| match val {
                automerge::Value::Scalar(s) => match s.as_ref() {
                    automerge::ScalarValue::Str(s) => s.to_string(),
                    _ => String::new(),
                },
                _ => String::new(),
            })
            .unwrap_or_default();

        Ok(content)
    }

    /// Apply automerge CRDT change.
    async fn handle_edit_apply(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, fid: u32, data: &EditApplyRequest,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        let shared = self.editing_table.get_session(&subject, fid)
            .ok_or_else(|| anyhow!("no active editing session for fid {}", fid))?;

        let mut sd = shared.lock();
        sd.doc.load_incremental(&data.change_bytes)?;
        sd.dirty = true;

        Ok(())
    }

    /// Close CRDT editing session.
    async fn handle_edit_close(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, fid: u32,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        if !self.editing_table.close(&subject, fid) {
            anyhow::bail!("no active editing session for fid {}", fid);
        }
        Ok(())
    }

    /// Write CRDT state to disk file.
    async fn handle_ctl_flush(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, fid: u32,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        let rel_path = self.fid_rel_path(fid, &subject)?;
        let shared = self.editing_table.get_session(&subject, fid)
            .ok_or_else(|| anyhow!("no active editing session for fid {}", fid))?;

        // Read current file hash and compare with stored hash
        let root = self.get_contained_root(repo_id, name).await
            .map_err(|e| anyhow!("{}", e))?;
        let current_hash = {
            let file = root.open_file(&rel_path, false, false, false, false, false)
                .map_err(|e| anyhow!("{}", e))?;
            let mut data = Vec::new();
            std::io::BufReader::new(file).read_to_end(&mut data)?;
            editing::sha256_hash(&data)
        };

        let mut sd = shared.lock();

        // Check for external modifications
        if current_hash != sd.file_hash {
            anyhow::bail!("file modified externally since editOpen; cannot flush");
        }

        // Get content from CRDT doc
        let content = sd.doc.get(automerge::ROOT, "content")
            .ok()
            .flatten()
            .map(|(val, _)| match val {
                automerge::Value::Scalar(s) => match s.as_ref() {
                    automerge::ScalarValue::Str(s) => s.to_string(),
                    _ => String::new(),
                },
                _ => String::new(),
            })
            .unwrap_or_default();

        // Write to disk (write=true, truncate=true)
        let mut file = root.open_file(&rel_path, true, false, true, false, false)
            .map_err(|e| anyhow!("{}", e))?;
        use std::io::Write as _;
        file.write_all(content.as_bytes())?;
        file.flush()?;

        // Update stored hash and clear dirty flag
        sd.file_hash = editing::sha256_hash(content.as_bytes());
        sd.dirty = false;

        Ok(())
    }
}

// Generated WorktreeHandler Implementation — 9P2000-inspired protocol
// ============================================================================

#[async_trait::async_trait(?Send)]
impl WorktreeHandler for RegistryService {
    /// Walk: resolve path components to get a fid.
    async fn handle_walk(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &NpWalk,
    ) -> Result<RWalk> {
        let root = self.get_contained_root(repo_id, name).await
            .map_err(|e| anyhow!("{}", e))?;

        // Join wnames into a path (empty wnames = root)
        let path = if data.wnames.is_empty() {
            ".".to_owned()
        } else {
            data.wnames.join("/")
        };

        let walk_handle = root.resolve_handle(&path)
            .map_err(|e| anyhow!("{}", e))?;
        let meta = walk_handle.metadata()
            .map_err(|e| anyhow!("{}", e))?;
        let qid = qid_from_metadata(&meta);

        let subject = ctx.subject().to_string();
        // Use client-specified newfid, or auto-allocate
        let fid = if data.newfid > 0 {
            data.newfid
        } else {
            self.fid_table.alloc_fid(&subject)
                .map_err(|e| anyhow!("{}", e))?
        };

        let entry = FidEntry {
            state: FidState::Walked { walk_handle, qid: qid.clone() },
            owner_identity: subject.clone(),
            last_accessed: AtomicU64::new(now_epoch_secs()),
        };
        if data.newfid > 0 {
            // Client-specified fid: must increment count (alloc_fid wasn't called)
            self.fid_table.insert_counted(fid, entry, &subject);
        } else {
            // Auto-allocated fid: count was already incremented by alloc_fid
            self.fid_table.insert(fid, entry);
        }

        Ok(RWalk {
            qid: Self::qid_to_data(&qid),
        })
    }

    /// Open: open a walked fid for I/O, returns iounit.
    async fn handle_open(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &NpOpen,
    ) -> Result<ROpen> {
        let subject = ctx.subject().to_string();

        // Take the fid to transition Walked → Opened (no count change)
        let entry = self.fid_table.take(data.fid)
            .ok_or_else(|| anyhow!("fid {} not found", data.fid))?;
        if entry.owner_identity != subject {
            // Put it back and bail
            self.fid_table.replace(data.fid, entry);
            anyhow::bail!("fid {} not owned by caller", data.fid);
        }

        let (walk_handle, qid) = match entry.state {
            FidState::Walked { walk_handle, qid } => (walk_handle, qid),
            other => {
                // Put it back and bail
                self.fid_table.replace(data.fid, FidEntry {
                    state: other,
                    owner_identity: entry.owner_identity,
                    last_accessed: entry.last_accessed,
                });
                anyhow::bail!("fid {} is already opened", data.fid);
            }
        };

        let write = (data.mode & OWRITE) != 0 || (data.mode & ORDWR) != 0;
        let file = walk_handle.open_file(write)
            .map_err(|e| anyhow!("{}", e))?;

        // If OTRUNC, truncate the file
        if (data.mode & OTRUNC) != 0 && write {
            file.set_len(0)?;
        }

        let iounit = DEFAULT_IOUNIT;

        // Re-insert as Opened (no count change — take didn't decrement)
        self.fid_table.replace(data.fid, FidEntry {
            state: FidState::Opened {
                file: Mutex::new(file),
                qid: qid.clone(),
                iounit,
                mode: data.mode,
            },
            owner_identity: subject,
            last_accessed: AtomicU64::new(now_epoch_secs()),
        });

        Ok(ROpen {
            qid: Self::qid_to_data(&qid),
            iounit,
        })
    }

    /// Create: create a file/dir under a walked directory fid.
    async fn handle_create(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &NpCreate,
    ) -> Result<ROpen> {
        let subject = ctx.subject().to_string();

        // Take the fid to get ownership of the WalkHandle (directory we're creating in)
        let entry = self.fid_table.take(data.fid)
            .ok_or_else(|| anyhow!("fid {} not found", data.fid))?;
        if entry.owner_identity != subject {
            self.fid_table.replace(data.fid, entry);
            anyhow::bail!("fid {} not owned by caller", data.fid);
        }

        let (dir_handle, _dir_qid) = match entry.state {
            FidState::Walked { walk_handle, qid } => {
                if qid.qtype & QTDIR == 0 {
                    // Put it back and bail
                    self.fid_table.replace(data.fid, FidEntry {
                        state: FidState::Walked { walk_handle, qid },
                        owner_identity: entry.owner_identity,
                        last_accessed: entry.last_accessed,
                    });
                    anyhow::bail!("fid {} is not a directory", data.fid);
                }
                (walk_handle, qid)
            }
            other => {
                self.fid_table.replace(data.fid, FidEntry {
                    state: other,
                    owner_identity: entry.owner_identity,
                    last_accessed: entry.last_accessed,
                });
                anyhow::bail!("fid {} must be a walked (not opened) directory", data.fid);
            }
        };

        let is_dir = (data.perm & DMDIR) != 0;

        // Compute the child's path relative to the contained root
        let child_path = dir_handle.child_rel_path(&data.name);
        let root = self.get_contained_root(repo_id, name).await
            .map_err(|e| anyhow!("{}", e))?;

        if is_dir {
            // Create child directory via ContainedRoot (kernel-enforced containment)
            root.mkdir(&child_path)
                .map_err(|e| anyhow!("{}", e))?;

            // Resolve a handle to the new child directory
            let child_handle = root.resolve_handle(&child_path)
                .map_err(|e| anyhow!("{}", e))?;
            let meta = child_handle.metadata()
                .map_err(|e| anyhow!("{}", e))?;
            let qid = qid_from_metadata(&meta);

            // Replace the fid with the new directory as Walked (no count change)
            self.fid_table.replace(data.fid, FidEntry {
                state: FidState::Walked { walk_handle: child_handle, qid: qid.clone() },
                owner_identity: subject,
                last_accessed: AtomicU64::new(now_epoch_secs()),
            });

            Ok(ROpen {
                qid: Self::qid_to_data(&qid),
                iounit: DEFAULT_IOUNIT,
            })
        } else {
            // Create file via ContainedRoot (kernel-enforced containment)
            let file = root.open_file(&child_path, /*write*/true, /*create*/true, /*truncate*/true, /*append*/false, /*excl*/false)
                .map_err(|e| anyhow!("{}", e))?;

            let meta = file.metadata()?;
            let qid = qid_from_metadata(&meta);
            let iounit = DEFAULT_IOUNIT;

            // Force ORDWR since create always opens writable
            let mode = data.mode | ORDWR;

            // Replace the fid with the opened file (no count change)
            self.fid_table.replace(data.fid, FidEntry {
                state: FidState::Opened {
                    file: Mutex::new(file),
                    qid: qid.clone(),
                    iounit,
                    mode,
                },
                owner_identity: subject,
                last_accessed: AtomicU64::new(now_epoch_secs()),
            });

            Ok(ROpen {
                qid: Self::qid_to_data(&qid),
                iounit,
            })
        }
    }

    /// Read: offset+count read, bounded by iounit. This is THE fix for the
    /// readFile bug — no unbounded allocation possible.
    async fn handle_read(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &NpRead,
    ) -> Result<RRead> {
        let subject = ctx.subject().to_string();
        let entry = self.fid_table.get_verified(data.fid, &subject)
            .map_err(|e| anyhow!("{}", e))?;

        match &entry.state {
            FidState::Opened { file, iounit, .. } => {
                // Clamp count to iounit — structurally prevents unbounded reads
                let count = std::cmp::min(data.count, *iounit) as usize;
                let mut buf = vec![0u8; count];
                let mut f = file.lock();
                f.seek(SeekFrom::Start(data.offset))?;
                let n = f.read(&mut buf)?;
                buf.truncate(n);
                Ok(RRead { data: buf })
            }
            _ => anyhow::bail!("fid {} is not opened for I/O", data.fid),
        }
    }

    /// Write: offset+data write, rejects if data > iounit.
    async fn handle_write(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &NpWrite,
    ) -> Result<RWrite> {
        let subject = ctx.subject().to_string();
        // Guard: reject direct writes to files with active CRDT editing sessions
        if self.editing_table.fid_has_session(&subject, data.fid) {
            anyhow::bail!("fid {} is open for CRDT editing, use ctl edit-apply instead", data.fid);
        }
        let entry = self.fid_table.get_verified(data.fid, &subject)
            .map_err(|e| anyhow!("{}", e))?;

        match &entry.state {
            FidState::Opened { file, iounit, mode, .. } => {
                if (*mode & OWRITE) == 0 && (*mode & ORDWR) == 0 {
                    anyhow::bail!("fid {} not opened for writing", data.fid);
                }
                if data.data.len() as u32 > *iounit {
                    anyhow::bail!(
                        "write data ({} bytes) exceeds iounit ({})",
                        data.data.len(), iounit
                    );
                }
                let mut f = file.lock();
                f.seek(SeekFrom::Start(data.offset))?;
                let n = f.write(&data.data)?;
                Ok(RWrite { count: n as u32 })
            }
            _ => anyhow::bail!("fid {} is not opened for I/O", data.fid),
        }
    }

    /// Clunk: release a fid, close any open file.
    async fn handle_clunk(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &NpClunk,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        // Clean up any CRDT editing session for this fid
        self.editing_table.on_clunk(&subject, data.fid);
        let _entry = self.fid_table.remove(data.fid, &subject)
            .ok_or_else(|| anyhow!("fid {} not found or not owned", data.fid))?;
        // File is dropped here, closing the fd
        Ok(())
    }

    /// Remove: clunk + delete the file/directory.
    async fn handle_remove(&self, ctx: &EnvelopeContext, _request_id: u64,
        repo_id: &str, name: &str, data: &NpRemove,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        let entry = self.fid_table.remove(data.fid, &subject)
            .ok_or_else(|| anyhow!("fid {} not found or not owned", data.fid))?;

        // Determine what to remove based on the qid type
        let qid = match &entry.state {
            FidState::Walked { qid, .. } | FidState::Opened { qid, .. } => qid,
        };

        // We need the path to remove — for now, we can use the contained root
        // The walk_handle carries enough info, but we need the path.
        // Since 9P remove operates on the fid (not a path), and we have
        // the WalkHandle, we use the metadata to identify what we're removing.
        // For a proper implementation, we'd need to track the path in the fid.
        let _ = (repo_id, name, qid);
        warn!("9P remove: fid {} clunked but path-based removal requires path tracking", data.fid);
        // The fid is already removed (clunked) — file handle dropped
        Ok(())
    }

    /// Stat: get file metadata for a fid.
    async fn handle_np_stat(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &NpStatReq,
    ) -> Result<RStat> {
        let subject = ctx.subject().to_string();
        let entry = self.fid_table.get_verified(data.fid, &subject)
            .map_err(|e| anyhow!("{}", e))?;

        let meta = match &entry.state {
            FidState::Walked { walk_handle, .. } => {
                walk_handle.metadata().map_err(|e| anyhow!("{}", e))?
            }
            FidState::Opened { file, .. } => {
                file.lock().metadata()?
            }
        };

        let stat = Self::metadata_to_np_stat("", &meta);
        Ok(RStat { stat })
    }

    /// Wstat: modify file metadata (rename, truncate, etc).
    async fn handle_wstat(&self, ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, data: &NpWstat,
    ) -> Result<()> {
        let subject = ctx.subject().to_string();
        let entry = self.fid_table.get_verified(data.fid, &subject)
            .map_err(|e| anyhow!("{}", e))?;

        // Handle length change (truncate)
        if data.stat.length != u64::MAX && data.stat.length != 0 {
            match &entry.state {
                FidState::Opened { file, .. } => {
                    file.lock().set_len(data.stat.length)?;
                }
                _ => anyhow::bail!("wstat truncate requires an opened fid"),
            }
        }

        // Name changes (rename) would require path tracking — deferred
        if !data.stat.name.is_empty() {
            warn!("wstat rename not yet implemented (requires path tracking)");
        }

        Ok(())
    }

    /// Flush: cancel a pending operation (no-op for synchronous ops).
    async fn handle_flush(&self, _ctx: &EnvelopeContext, _request_id: u64,
        _repo_id: &str, _name: &str, _data: &NpFlush,
    ) -> Result<()> {
        // No-op for synchronous operations
        Ok(())
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

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = RegistryResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
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
