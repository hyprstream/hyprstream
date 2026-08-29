//! `/exec/instances/` VFS mount — projects `SandboxPool`'s active sandboxes
//! as a Plan9 `/proc`-style tree.
//!
//! This is the P2 slice of epic #608 ("P9 Task/Instance Projection"); see
//! epic #608 for the full `/exec` tree this is one branch of
//! (`/exec/instances/<id>/...`).
//!
//! Layout:
//! - `/exec/instances/`            — dynamic dir: active sandbox/instance ids
//! - `/exec/instances/<id>/ctl`    — ctl file: write a verb to drive the
//!   instance's lifecycle (see [`Verb`] for the grammar)
//! - `/exec/instances/<id>/status` — read-only: current `PodSandboxState`,
//!   live/non-blocking poll of pool state
//! - `/exec/instances/<id>/exit`   — read-only: blocks until the instance is
//!   terminal, then returns the terminal status. A read that starts after
//!   the instance has already terminated returns immediately with the
//!   retained status (read-then-subscribe, no missed completions).
//! - `/exec/instances/<id>/ns`     — read-only: best-effort textual listing
//!   of the sandbox's mount-prefixes/namespace, or an empty placeholder
//! - `/exec/instances/<id>/fd/1,2` — bounded terminal stdout/stderr streams
//!
//! `fd/` is a bounded task-local adapter over the existing MoQ stream
//! substrate. It deliberately exposes no socket or host-path escape hatch.
//! `fd/0` stays absent until a backend task interface can consume it: accepting
//! bytes into an unobserved buffer would violate the Plan 9 file contract.
//!
//! ## Ctl verb grammar
//!
//! Writes to `<id>/ctl` are UTF-8, one verb per write (Plan9-style short text
//! commands, not a structured protocol):
//!
//! | verb      | effect                                                          |
//! |-----------|------------------------------------------------------------------|
//! | `start`   | no-op today — sandboxes are started by `SandboxPool::acquire`;   |
//! |           | accepted for grammar completeness/forward-compat, returns ok     |
//! |           | if the instance exists and is not already terminal.              |
//! | `stop`    | calls `SandboxBackend::stop` (graceful stop) on the instance.    |
//! | `kill`    | alias for `stop` today (no separate SIGKILL-style path exposed   |
//! |           | by `SandboxBackend` yet); kept distinct in the grammar so a       |
//! |           | future hard-kill primitive can be wired without a ctl-format     |
//! |           | change.                                                           |
//! | `destroy` | calls `SandboxBackend::destroy`, then removes the instance from  |
//! |           | the pool's active map and marks it terminal for `exit`/`status`. |
//! | `exec`    | compatibility whitespace-split argv task through this mount's     |
//! |           | `SandboxBackend` seam, then latches stdout/stderr in `fd/1,2`.   |
//! | `exec-json` | lossless JSON string-array argv form; it is never shell-evaluated. |
//!
//! Unknown verbs return [`MountError::InvalidArgument`]. A verb may be
//! followed by trailing whitespace, which is trimmed.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rand::RngCore;
use tokio::sync::Notify;

use hyprstream_rpc::latch::{Terminal, TerminalStore};
use hyprstream_rpc::moq_stream::{MoqStreamOrigin, MoqStreamPublisher};
use hyprstream_rpc::stream_info::Job;
use hyprstream_rpc::streaming::StreamContext;
use hyprstream_rpc::{AttachmentId, AttachmentOperation, AuthorityGeneration};
use hyprstream_rpc_std::task::{
    ContentDigest, TaskAttachmentBinding, TaskContentRecord, TaskContentRole, TaskError,
    TaskHandle, TaskId, TaskPayload, TaskReaches, TaskResult, TaskRuntimeConstraints, TaskService,
    TaskSignal, TaskSignalRequest, TaskSnapshot, TaskSpawnRequest, TaskState,
};
use hyprstream_vfs::{DevFileState, DirEntry, Fid, Mount, MountError, Stat, Subject, VfsOpContext};
// `parking_lot::Mutex` remains for mount-level terminal and waiter maps.  Ctl
// response state itself uses `DevFileState`, so it is unambiguously per fid.
use parking_lot::Mutex as PmMutex;

use super::client::PodSandboxState;
use super::pool::SandboxPool;

// ─────────────────────────────────────────────────────────────────────────────
// Ctl verb grammar
// ─────────────────────────────────────────────────────────────────────────────

/// Verbs accepted by `/exec/instances/<id>/ctl`. See module docs for the
/// full grammar table.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Verb {
    Start,
    Stop,
    Kill,
    Destroy,
    /// Execute a task through this Plan 9 projection. Arguments are already
    /// tokenized; this is deliberately not a shell-string interface.
    Exec(Vec<String>),
    /// Lossless JSON argv form for adapters. Parsed as a JSON array only;
    /// it is never passed to a shell.
    ExecJson(Vec<String>),
}

impl Verb {
    fn parse(s: &str) -> Option<Self> {
        if let Some(argv) = s.trim().strip_prefix("exec-json ") {
            let argv: Vec<String> = serde_json::from_str(argv).ok()?;
            return (!argv.is_empty()).then_some(Self::ExecJson(argv));
        }
        let words: Vec<String> = s.split_whitespace().map(str::to_owned).collect();
        match words.as_slice() {
            [verb] if verb == "start" => Some(Self::Start),
            [verb] if verb == "stop" => Some(Self::Stop),
            [verb] if verb == "kill" => Some(Self::Kill),
            [verb] if verb == "destroy" => Some(Self::Destroy),
            [verb, command @ ..] if verb == "exec" && !command.is_empty() => {
                Some(Self::Exec(command.to_vec()))
            }
            _ => None,
        }
    }

    /// Whether this verb mutates the instance lifecycle destructively (stop/
    /// kill/destroy). These are the verbs the MAC PEP must mediate (#1272):
    /// `start` is constructive (it cannot remove or halt another subject's
    /// instance), so it is not in the mediated set.
    fn is_lifecycle_destructive(&self) -> bool {
        matches!(self, Verb::Stop | Verb::Kill | Verb::Destroy)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle authorization seam (#1272)
// ─────────────────────────────────────────────────────────────────────────────

/// The destructive lifecycle verbs a [`LifecyclePolicy`] mediates.
///
/// These are the `ctl` ops that halt or remove an instance — the
/// "lifecycle mutation" surface issue #1272 names. `Start` is deliberately
/// absent: it is constructive, not a mutation of another subject's running
/// instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleVerb {
    Stop,
    Kill,
    Destroy,
}

/// Fail-closed authorization seam for destructive `ctl` lifecycle verbs
/// (#1272, epic #1267 T3).
///
/// `ExecMount`'s `write` path previously ignored `_caller` entirely when
/// invoking `stop`/`kill`/`destroy`. This trait is the mediation point: when a
/// policy is installed on the mount ([`ExecMount::new_with_lifecycle`]), every
/// destructive verb must pass `authorize` before the backend is touched.
///
/// **Fail-closed contract:** returning `Err` ⇒ the verb is refused
/// (`MountError::PermissionDenied`); there is no permissive default inside a
/// policy. The mount's status-quo (no policy installed) is represented by
/// `lifecycle: None`, NOT by a permissive policy — mirroring the
/// [`hyprstream_vfs::NamespacePep`] activation posture (the separately-gated
/// B-lane flips construction sites to armed).
///
/// The clearance-provenance dependency (#698) is the same as the VFS PEP's: a
/// production policy resolves the caller's `SecurityContext` from verified
/// credential material and applies MAC `can_access` against the instance's
/// label. Until #698 wires that, the policy is a structural seam.
pub trait LifecyclePolicy: Send + Sync {
    /// Authorize `verb` on instance `id` by `caller`. `Ok(())` permits;
    /// `Err(detail)` denies with a human-readable reason.
    fn authorize(&self, caller: &Subject, id: &str, verb: LifecycleVerb) -> Result<(), String>;
}

/// Fail-closed policy: denies every destructive verb for every subject.
/// Installing this arms the mediation point to deny-by-default (e.g. during
/// the #698 dependency window).
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllLifecycle;

impl LifecyclePolicy for DenyAllLifecycle {
    fn authorize(&self, _caller: &Subject, _id: &str, _verb: LifecycleVerb) -> Result<(), String> {
        Err("lifecycle verb denied: no permissive policy".into())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Terminal tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Terminal payload latched into this mount's [`TerminalStore`] once an
/// instance exits (i.e. is destroyed/stopped-for-good). Populated by the
/// `ctl` `destroy` handler.
#[derive(Clone, Debug)]
struct TerminalStatus {
    /// Final `PodSandboxState` observed before removal.
    last_state: PodSandboxState,
}

impl TerminalStatus {
    fn render(&self) -> String {
        format!("exited state={:?}\n", self.last_state)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task-local fd streams (#1434)
// ─────────────────────────────────────────────────────────────────────────────

/// Hard per-fd retained-output ceiling for the shadow-CI projection.  The
/// backing MoQ stream keeps its `Job` profile (ordered, retained, terminal);
/// this cap bounds the Plan 9 reader's local materialization.
const MAX_RETAINED_FD_BYTES: usize = 1024 * 1024;
/// Explicit terminal indication for local 9P consumers that the bounded
/// retained fd projection omitted a suffix.
const FD_TRUNCATION_MARKER: &[u8] = b"\n[hyprstream: output truncated]\n";
/// Carrier terminal metadata for a complete local fd projection.
const FD_COMPLETE_METADATA: &[u8] = br#"{"exec_fd":"complete"}"#;
/// Carrier terminal metadata for a bounded projection that omitted a suffix.
const FD_TRUNCATED_METADATA: &[u8] = br#"{"exec_fd":"truncated"}"#;

const fn fd_complete_metadata(truncated: bool) -> &'static [u8] {
    if truncated {
        FD_TRUNCATED_METADATA
    } else {
        FD_COMPLETE_METADATA
    }
}

/// A bounded terminal byte stream projected by one Plan 9 fd.
struct TaskFdStream {
    state: tokio::sync::Mutex<TaskFdState>,
    wake: Arc<Notify>,
}

struct TaskFdState {
    bytes: Vec<u8>,
    closed: bool,
    truncated: bool,
    // Kept test-only solely to verify the actual carrier terminal payload;
    // production callers never receive this key.
    #[cfg(test)]
    mac_key: [u8; 32],
    publisher: MoqStreamPublisher,
}

impl TaskFdStream {
    fn new(origin: &MoqStreamOrigin, instance: &str, fd: u8) -> Result<Self, MountError> {
        let mut mac_key = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut mac_key);
        let context = StreamContext::new(
            format!("exec-{instance}-fd-{fd}"),
            format!("exec-{}-fd-{fd}", uuid::Uuid::new_v4()),
            mac_key,
            [0u8; 32],
        )
        .with_qos_preset::<Job>();
        let publisher = origin
            .publisher(&context)
            .map_err(|e| MountError::Io(format!("create fd/{fd} MoQ stream: {e}")))?;
        Ok(Self {
            state: tokio::sync::Mutex::new(TaskFdState {
                bytes: Vec::new(),
                closed: false,
                truncated: false,
                #[cfg(test)]
                mac_key,
                publisher,
            }),
            wake: Arc::new(Notify::new()),
        })
    }

    async fn append(&self, context: Option<&VfsOpContext>, data: &[u8]) -> Result<(), MountError> {
        // This is the stream-continuation boundary: do not publish the next
        // MoQ segment without a current explicit publish grant.
        if let Some(context) = context {
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        let mut state = self.state.lock().await;
        // The lock may have been contended while revocation raced this
        // continuation. Recheck the same publish scope at the precise
        // publication boundary rather than inheriting the earlier decision.
        if let Some(context) = context {
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        if state.closed {
            return Err(MountError::InvalidArgument(
                "fd stream is already closed".into(),
            ));
        }
        let available = MAX_RETAINED_FD_BYTES.saturating_sub(state.bytes.len());
        let retained = &data[..data.len().min(available)];
        if !retained.is_empty() {
            state
                .publisher
                .publish_data(retained)
                .await
                .map_err(|e| MountError::Io(format!("publish fd stream data: {e}")))?;
            state.bytes.extend_from_slice(retained);
        }
        if retained.len() != data.len() {
            state.truncated = true;
        }
        drop(state);
        self.wake.notify_waiters();
        Ok(())
    }

    async fn close(&self, context: Option<&VfsOpContext>) -> Result<(), MountError> {
        // Completion is also a continuation/external effect. Recheck at the
        // final publication boundary rather than inheriting a dispatch permit.
        if let Some(context) = context {
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        let mut state = self.state.lock().await;
        // Completion is a distinct carrier publication after an awaited lock.
        if let Some(context) = context {
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        if !state.closed {
            if state.truncated {
                // `append` normally fills the retained buffer exactly, so an
                // append-only marker would never fit. Reserve its space at
                // close by replacing the final retained bytes. This makes a
                // local fd reader observe truncation rather than silently
                // treating bounded output as the complete terminal stream.
                let retained_len = MAX_RETAINED_FD_BYTES.saturating_sub(FD_TRUNCATION_MARKER.len());
                state.bytes.truncate(retained_len);
                state.bytes.extend_from_slice(FD_TRUNCATION_MARKER);
            }
            let terminal_metadata = fd_complete_metadata(state.truncated);
            state
                .publisher
                .complete_ref(terminal_metadata)
                .await
                .map_err(|e| MountError::Io(format!("complete fd stream: {e}")))?;
            state.closed = true;
        }
        drop(state);
        self.wake.notify_waiters();
        Ok(())
    }

    /// Whether this stream has already latched EOF (a prior `exec`, or a
    /// `stop`/`destroy` through `apply_verb`). Checked by `exec_task` *before*
    /// invoking the backend so a second exec on one instance fails fast
    /// instead of running the command and then discarding its output.
    async fn is_closed(&self) -> bool {
        self.state.lock().await.closed
    }

    async fn read_until_closed(
        &self,
        context: Option<&VfsOpContext>,
        offset: u64,
        count: u32,
    ) -> Result<Vec<u8>, MountError> {
        loop {
            let notified = self.wake.notified();
            {
                let state = self.state.lock().await;
                // Every returned byte range (including EOF) is a read effect
                // after an awaited lock/wake boundary.
                if let Some(context) = context {
                    context.ensure_operation(AttachmentOperation::TaskRead)?;
                }
                let start = offset as usize;
                if start < state.bytes.len() {
                    let end = start.saturating_add(count as usize).min(state.bytes.len());
                    return Ok(state.bytes[start..end].to_vec());
                }
                if state.closed {
                    return Ok(Vec::new());
                }
            }
            notified.await;
        }
    }
}

/// The single per-task stdio adapter shared by the Plan 9 task operation and
/// fd readers. `fd/0` is intentionally absent: no backend task interface
/// consumes interactive stdin yet, so accepting input would be a false success.
struct TaskStdioAdapter {
    stdout: TaskFdStream,
    stderr: TaskFdStream,
}

/// One completed task execution as observed by both the production CRI caller
/// and the Plan 9 fd projection. Keeping this result adjacent to the adapter
/// makes the single-owner contract explicit: neither caller needs to invoke a
/// backend a second time or retain a competing output buffer.
#[derive(Debug)]
pub struct TaskExecResult {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Metadata retained for Tasks allocated through the attachment-bound standard
/// contract. Existing pre-spike `/exec/instances` entries remain compatible;
/// only a Task-contract allocation has this additional authority record.
#[derive(Clone, Debug)]
struct TaskRecord {
    /// Correlation snapshot only. A record never retains a grant: effects
    /// must be authorized by the caller's current context/binding.
    attachment_id: AttachmentId,
    authority_generation: AuthorityGeneration,
    subject: Subject,
    namespace_manifest: hyprstream_rpc_std::task::NamespaceManifestDigest,
    /// Timeout selected by the trusted clone policy (or the standard Task
    /// request) and applied when this instance executes its one argv task.
    timeout_secs: u64,
    state: TaskState,
    content_records: Vec<TaskContentRecord>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Fid types
// ─────────────────────────────────────────────────────────────────────────────

/// Which kind of file/dir a fid refers to.
#[derive(Clone, Debug)]
enum ExecFidKind {
    /// `/exec/instances/` (this mount's root).
    InstancesDir,
    /// `/exec/instances/<id>/` itself.
    InstanceDir(String),
    /// `/exec/instances/<id>/ctl`.
    Ctl(String),
    /// `/exec/instances/<id>/status`.
    Status(String),
    /// `/exec/instances/<id>/exit`.
    Exit(String),
    /// `/exec/instances/<id>/ns`.
    Ns(String),
    /// `/exec/instances/<id>/fd/`.
    FdDir(String),
    /// `/exec/instances/<id>/fd/{0,1,2}`.
    Fd(String, u8),
}

/// Fid state for the exec mount.
struct ExecFid {
    kind: ExecFidKind,
    /// A verified attachment context is retained on fids created through the
    /// narrow context-aware entry point.  A raw `Mount::walk` cannot fabricate
    /// it, so attachment-bound Tasks are hidden from the legacy Subject-only
    /// path until the whole Mount trait migrates.
    context: Option<VfsOpContext>,
    /// Latch for the ctl write→read pattern (verb result message).  This
    /// reusable per-fid primitive also backs other VFS device files, keeping
    /// a response local to the open fid that issued its lifecycle verb.
    write_buf: DevFileState,
}

// ─────────────────────────────────────────────────────────────────────────────
// ExecMount
// ─────────────────────────────────────────────────────────────────────────────

/// VFS mount that projects a [`SandboxPool`]'s active sandboxes as
/// `/exec/instances/<id>/{ctl,status,exit,ns}`.
///
/// Mount at `/exec/instances` in the namespace (the parent `/exec` tree —
/// `backends/`, `classes/`, `pool/`, `sched/` — is out of scope for this
/// issue; see the design doc §4).
pub struct ExecMount {
    pool: Arc<SandboxPool>,
    /// Task-local MoQ origin for per-instance stdio. It is intentionally not
    /// served as a new public network plane: `/exec` is the capability surface.
    fd_origin: MoqStreamOrigin,
    /// Per-instance fd state. Output streams use the existing MoQ publisher
    /// contract; the Plan 9 files retain their bounded local materialization.
    fd_streams: tokio::sync::Mutex<HashMap<String, Arc<TaskStdioAdapter>>>,
    /// Retained terminal state (EV7, `hyprstream_rpc::latch`) for instances
    /// that have been `destroy`ed through this mount. `latch()` is called
    /// exactly once per instance (write-once), so a late `exit` read is
    /// served from here immediately.
    terminal: TerminalStore<String, TerminalStatus>,
    /// Process completion is distinct from sandbox destruction: a completed
    /// command keeps its sandbox reachable for later explicit cleanup.
    task_exit: TerminalStore<String, i32>,
    /// Ids ever latched into `terminal`. `TerminalStore` doesn't expose key
    /// enumeration (by design — it's a per-key latch/read primitive, not a
    /// directory), so `readdir` tracks this separately to keep destroyed
    /// instances listed alongside `pool.list_active()`.
    terminal_ids: PmMutex<std::collections::HashSet<String>>,
    /// Per-instance wake for an early `exit` reader blocked on a not-yet-
    /// latched instance. `ExecMount`'s termination signal is entirely
    /// in-process (via `apply_verb`'s `Destroy` arm), so a lightweight
    /// `tokio::sync::Notify` per id is the read-then-subscribe "subscribe"
    /// half here, in place of `hyprstream_rpc::latch::read_then_subscribe`'s
    /// moq-backed `EventSubscriber` slow path (which is for cross-process
    /// fan-out this mount doesn't need). `TerminalStore` is still the single
    /// source of retained truth; this only wakes waiters once it's written.
    waiters: PmMutex<HashMap<String, Arc<Notify>>>,
    /// Optional MAC lifecycle policy over destructive `ctl` verbs (#1272).
    /// `None` ⇒ the un-enforced status quo (the mount's `write` ignores the
    /// caller, the gap #1272 describes); `Some` ⇒ `stop`/`kill`/`destroy`
    /// must pass [`LifecyclePolicy::authorize`] before the backend is touched.
    lifecycle: Option<Arc<dyn LifecyclePolicy>>,
    /// Authority-bound metadata for standard Task allocations. This is kept
    /// separate from the pool's compatibility projection so a raw instance id
    /// never becomes an authority bearer.
    task_records: PmMutex<HashMap<String, TaskRecord>>,
}

impl ExecMount {
    /// Create a new `ExecMount` over the given pool, **unenforced**.
    ///
    /// No lifecycle policy is installed: destructive `ctl` verbs behave as
    /// before (the dormant posture). To arm fail-closed mediation, use
    /// [`Self::new_with_lifecycle`].
    pub fn new(pool: Arc<SandboxPool>) -> Self {
        Self {
            pool,
            // This is a local carrier only: production has no server,
            // subscriber, MAC-key handoff, or reachable locator for this
            // origin. The 9P fd reads below serve `TaskFdStream`'s bounded
            // local buffer. Tests may consume the carrier to verify its frame
            // encoding; a client-reachable stream plane is deferred.
            fd_origin: MoqStreamOrigin::standalone()
                .with_prefix("local/exec")
                .build(),
            fd_streams: tokio::sync::Mutex::new(HashMap::new()),
            terminal: TerminalStore::new(),
            task_exit: TerminalStore::new(),
            terminal_ids: PmMutex::new(std::collections::HashSet::new()),
            waiters: PmMutex::new(HashMap::new()),
            lifecycle: None,
            task_records: PmMutex::new(HashMap::new()),
        }
    }

    /// Create a new `ExecMount` with a MAC lifecycle policy armed.
    ///
    /// Every destructive verb (`stop`/`kill`/`destroy`) written to
    /// `/exec/instances/<id>/ctl` must pass `policy.authorize(caller, id, verb)`
    /// before the backend is touched; a denial yields
    /// [`MountError::PermissionDenied`]. Pass [`DenyAllLifecycle`] for the
    /// fail-closed default during the #698 dependency window.
    pub fn new_with_lifecycle(pool: Arc<SandboxPool>, policy: Arc<dyn LifecyclePolicy>) -> Self {
        let mut mount = Self::new(pool);
        mount.lifecycle = Some(policy);
        mount
    }

    /// Get (or create) the `Notify` an `exit` reader waits on for `id`.
    fn waiter_for(&self, id: &str) -> Arc<Notify> {
        Arc::clone(
            self.waiters
                .lock()
                .entry(id.to_owned())
                .or_insert_with(|| Arc::new(Notify::new())),
        )
    }

    async fn fd_streams_for(&self, id: &str) -> Result<Arc<TaskStdioAdapter>, MountError> {
        let mut streams = self.fd_streams.lock().await;
        if let Some(existing) = streams.get(id) {
            return Ok(Arc::clone(existing));
        }
        let created = Arc::new(TaskStdioAdapter {
            stdout: TaskFdStream::new(&self.fd_origin, id, 1)?,
            stderr: TaskFdStream::new(&self.fd_origin, id, 2)?,
        });
        streams.insert(id.to_owned(), Arc::clone(&created));
        Ok(created)
    }

    /// Remove every local reach for a Task whose spawn failed after sandbox
    /// allocation, then return the sandbox to the pool. A failed spawn never
    /// leaves an attachment-bound record that the caller cannot observe.
    async fn rollback_task_allocation(&self, id: &str) {
        self.task_records.lock().remove(id);
        self.fd_streams.lock().await.remove(id);
        self.waiters.lock().remove(id);
        if let Err(error) = self.pool.release(id).await {
            tracing::warn!(
                sandbox_id = id,
                error = %error,
                "failed to release sandbox while rolling back task spawn"
            );
        }
    }

    /// Allocate an attachment-bound Task without executing it. This is the
    /// `/exec/clone` device seam: the caller supplies only a trusted-policy
    /// namespace commitment/constraints object; clone bytes never become
    /// policy input.
    pub async fn allocate_pending_task(
        &self,
        context: &VfsOpContext,
        namespace_manifest: hyprstream_rpc_std::task::NamespaceManifestDigest,
        constraints: TaskRuntimeConstraints,
    ) -> Result<String, MountError> {
        context.ensure_operation(AttachmentOperation::TaskSpawn)?;
        if let Some(requested) = constraints.backend_class.as_deref() {
            if requested != self.pool.backend().backend_type() {
                return Err(MountError::InvalidArgument(format!(
                    "clone policy requested backend {requested}, pool is {}",
                    self.pool.backend().backend_type()
                )));
            }
        }
        let id = self
            .pool
            .acquire(
                context.subject(),
                &super::client::PodSandboxConfig::default(),
            )
            .await
            .map_err(|error| MountError::Io(error.to_string()))?;
        if let Err(error) = context.ensure_operation(AttachmentOperation::TaskSpawn) {
            let _ = self.pool.release(&id).await;
            return Err(error);
        }
        self.task_records.lock().insert(
            id.clone(),
            TaskRecord {
                attachment_id: context.attachment_id().clone(),
                authority_generation: context.authority_generation(),
                subject: context.subject().clone(),
                namespace_manifest,
                timeout_secs: constraints.timeout_secs.unwrap_or(300),
                state: TaskState::Pending,
                content_records: Vec::new(),
            },
        );
        Ok(id)
    }

    /// Publish one completed task execution into the instance's terminal fd
    /// streams. The runtime calls this after `SandboxBackend::exec_sync`; the
    /// Plan 9 readers observe the same bounded output and then EOF.
    async fn publish_exec_result(
        &self,
        id: &str,
        caller: &Subject,
        context: Option<&VfsOpContext>,
        stdout: &[u8],
        stderr: &[u8],
    ) -> Result<(), MountError> {
        if self.pool.get(id).await.is_none() && !self.terminal.is_latched(&id.to_owned()) {
            return Err(MountError::NotFound(format!("instances/{id}")));
        }
        let context =
            self.task_context_for(id, caller, context, AttachmentOperation::TaskPublish)?;
        let streams = self.fd_streams_for(id).await?;
        if let Some(context) = context.as_ref() {
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        streams.stdout.append(context.as_ref(), stdout).await?;
        streams.stderr.append(context.as_ref(), stderr).await?;
        streams.stdout.close(context.as_ref()).await?;
        streams.stderr.close(context.as_ref()).await?;
        Ok(())
    }

    /// Validate the caller's *current* context against a contract Task's
    /// recorded correlation snapshot. The record intentionally contains no
    /// grant, so it cannot donate broader scopes from the original spawn.
    /// Compatibility instances have no record and retain their existing path.
    fn task_context_for(
        &self,
        id: &str,
        caller: &Subject,
        context: Option<&VfsOpContext>,
        operation: AttachmentOperation,
    ) -> Result<Option<VfsOpContext>, MountError> {
        let Some(record) = self.task_records.lock().get(id).cloned() else {
            return Ok(None);
        };
        let context = context.ok_or_else(|| {
            MountError::PermissionDenied("task requires a verified attachment context".into())
        })?;
        if context.subject() != caller
            || &record.subject != caller
            || &record.attachment_id != context.attachment_id()
            || record.authority_generation != context.authority_generation()
        {
            return Err(MountError::PermissionDenied(
                "attachment context does not authorize this task".into(),
            ));
        }
        context.ensure_operation(operation)?;
        Ok(Some(context.clone()))
    }

    /// Verify that a fid reaching an attachment-bound task was created from
    /// the exact attachment/generation that allocated it. This is the one-Mount
    /// prototype for `VfsOpContext`; the generic `Mount` entry point has only a
    /// Subject and therefore cannot be treated as an equivalent authority.
    fn verify_fid_context(
        &self,
        kind: &ExecFidKind,
        context: Option<&VfsOpContext>,
        caller: &Subject,
    ) -> Result<(), MountError> {
        let id = match kind {
            ExecFidKind::InstanceDir(id)
            | ExecFidKind::Ctl(id)
            | ExecFidKind::Status(id)
            | ExecFidKind::Exit(id)
            | ExecFidKind::Ns(id)
            | ExecFidKind::FdDir(id)
            | ExecFidKind::Fd(id, _) => id,
            ExecFidKind::InstancesDir => return Ok(()),
        };
        let Some(record) = self.task_records.lock().get(id).cloned() else {
            return Ok(());
        };
        let context = context.ok_or_else(|| {
            MountError::PermissionDenied("task requires a verified attachment context".into())
        })?;
        if context.subject() != caller
            || &record.subject != caller
            || &record.attachment_id != context.attachment_id()
            || record.authority_generation != context.authority_generation()
        {
            return Err(MountError::PermissionDenied(
                "attachment context does not authorize this task fid".into(),
            ));
        }
        context.ensure_operation(AttachmentOperation::TaskRead)?;
        Ok(())
    }

    /// Context-aware fid creation for the attachment-bound Task projection.
    /// The [`Mount`] override below routes trait-object callers here so a 9P
    /// attach cannot lose the opaque operation context while walking `/exec`.
    /// Accepting only a `Subject` would regress to ambient same-subject authority.
    pub async fn walk_with_context(
        &self,
        components: &[&str],
        context: &VfsOpContext,
    ) -> Result<Fid, MountError> {
        context.ensure_operation(AttachmentOperation::TaskRead)?;
        self.walk_inner(components, context.subject(), Some(context.clone()))
            .await
    }

    async fn walk_inner(
        &self,
        components: &[&str],
        caller: &Subject,
        context: Option<VfsOpContext>,
    ) -> Result<Fid, MountError> {
        let kind = match components {
            [] => ExecFidKind::InstancesDir,
            [id] => ExecFidKind::InstanceDir((*id).to_owned()),
            [id, "ctl"] => ExecFidKind::Ctl((*id).to_owned()),
            [id, "status"] => ExecFidKind::Status((*id).to_owned()),
            [id, "exit"] => ExecFidKind::Exit((*id).to_owned()),
            [id, "ns"] => ExecFidKind::Ns((*id).to_owned()),
            [id, "fd"] => ExecFidKind::FdDir((*id).to_owned()),
            [id, "fd", fd] => {
                let fd = fd
                    .parse::<u8>()
                    .map_err(|_| MountError::NotFound(components.join("/")))?;
                if !(1..=2).contains(&fd) {
                    return Err(MountError::NotFound(components.join("/")));
                }
                ExecFidKind::Fd((*id).to_owned(), fd)
            }
            _ => return Err(MountError::NotFound(components.join("/"))),
        };

        // Validate that referenced instances exist (directory walk into an
        // unknown id should 404, same as any 9P namespace).  Hide a
        // contract Task from an unbound/wrong attachment rather than exposing
        // whether its id exists.
        let id_to_check: Option<&str> = match &kind {
            ExecFidKind::InstanceDir(id)
            | ExecFidKind::Ctl(id)
            | ExecFidKind::Status(id)
            | ExecFidKind::Exit(id)
            | ExecFidKind::Ns(id)
            | ExecFidKind::FdDir(id)
            | ExecFidKind::Fd(id, _) => Some(id.as_str()),
            ExecFidKind::InstancesDir => None,
        };
        if let Some(id) = id_to_check {
            let exists =
                self.pool.get(id).await.is_some() || self.terminal.is_latched(&id.to_owned());
            if !exists {
                return Err(MountError::NotFound(components.join("/")));
            }
            if self
                .verify_fid_context(&kind, context.as_ref(), caller)
                .is_err()
            {
                return Err(MountError::NotFound(components.join("/")));
            }
        }

        Ok(Fid::new(ExecFid {
            kind,
            context,
            write_buf: DevFileState::new(),
        }))
    }

    fn task_error(error: MountError) -> TaskError {
        match error {
            MountError::NotFound(_) => {
                TaskError::InvalidRequest("task projection disappeared".into())
            }
            // A Mount permission error can originate in a lifecycle MAC/PEP
            // decision, a task-record correlation mismatch, or a stale
            // attachment. Callers that hold a Task binding must use
            // `task_error_for_binding` below to distinguish those cases.
            MountError::PermissionDenied(_) => TaskError::PermissionDenied,
            other => TaskError::Backend(other.to_string()),
        }
    }

    /// Translate a Mount failure at a Task-service boundary without relabeling
    /// every policy denial as a revoked attachment. A stale grant is reported
    /// as such; a still-current, correctly scoped grant means the denial came
    /// from another boundary (for example lifecycle MAC) and stays opaque as
    /// [`TaskError::PermissionDenied`].
    fn task_error_for_binding(
        error: MountError,
        attachment: &TaskAttachmentBinding,
        operations: &[AttachmentOperation],
    ) -> TaskError {
        if !matches!(&error, MountError::PermissionDenied(_)) {
            return Self::task_error(error);
        }

        for operation in operations {
            match attachment.ensure(*operation) {
                Ok(()) => {}
                Err(TaskError::StaleAttachment) => return TaskError::StaleAttachment,
                // A direct scope failure must not be hidden as revocation.
                Err(TaskError::PermissionDenied) => return TaskError::PermissionDenied,
                Err(other) => return other,
            }
        }
        TaskError::PermissionDenied
    }

    fn task_handle(id: &str, state: TaskState) -> TaskHandle {
        TaskHandle {
            task: TaskId::from_worker_instance(id),
            reaches: TaskReaches {
                vfs_path: format!("/exec/instances/{id}"),
                iroh_endpoint: None,
                // The local standalone publisher has no exported reach,
                // subscription, MAC-key handoff, or stable advertised topic.
                // Do not return fictional MoQ reaches from this prototype.
                moq_topics: Vec::new(),
            },
            state,
        }
    }

    fn bound_task_record(
        &self,
        task: &TaskId,
        attachment: &TaskAttachmentBinding,
        operation: AttachmentOperation,
    ) -> Result<TaskRecord, TaskError> {
        attachment.ensure(operation)?;
        let record = self
            .task_records
            .lock()
            .get(task.as_str())
            .cloned()
            .ok_or_else(|| TaskError::NotFound(task.clone()))?;
        if &record.attachment_id != attachment.attachment_id()
            || record.authority_generation != attachment.authority_generation()
            || &record.subject != attachment.subject()
        {
            return Err(TaskError::PermissionDenied);
        }
        Ok(record)
    }

    fn record_exec_result(&self, id: &str, result: &TaskExecResult) {
        if let Some(record) = self.task_records.lock().get_mut(id) {
            record.state = TaskState::Exited {
                code: result.exit_code,
            };
            let task = TaskId::from_worker_instance(id);
            record.content_records = vec![
                TaskContentRecord {
                    content: ContentDigest::from_bytes(&result.stdout),
                    task: task.clone(),
                    attachment_id: record.attachment_id.clone(),
                    authority_generation: record.authority_generation,
                    namespace_manifest: record.namespace_manifest,
                    role: TaskContentRole::Stdout,
                },
                TaskContentRecord {
                    content: ContentDigest::from_bytes(&result.stderr),
                    task,
                    attachment_id: record.attachment_id.clone(),
                    authority_generation: record.authority_generation,
                    namespace_manifest: record.namespace_manifest,
                    role: TaskContentRole::Stderr,
                },
            ];
        }
        // Process completion is separate from sandbox destruction.  Latch and
        // wake here (rather than in one ctl spelling) so every execution path,
        // including the standard `TaskService` path, makes `/exit` observable.
        self.task_exit.latch(
            id.to_owned(),
            Terminal {
                value: result.exit_code,
                latched_by: "exec-result".to_owned(),
            },
        );
        self.waiter_for(id).notify_waiters();
    }

    /// Execute through the selected sandbox backend and project the terminal
    /// stdout/stderr onto this instance's fd streams. This is the task-runtime
    /// bridge: callers do not receive a second host-path or socket interface.
    pub async fn exec_task(
        &self,
        id: &str,
        caller: &Subject,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<TaskExecResult, MountError> {
        self.exec_task_with_context(id, caller, None, command, timeout_secs)
            .await
    }

    /// Context-aware counterpart to [`Self::exec_task`] for the standard Task
    /// contract and a fid that already carries a verified operation grant.
    /// The public CRI compatibility API remains subject-only and therefore can
    /// execute only legacy instances without a Task record.
    async fn exec_task_with_context(
        &self,
        id: &str,
        caller: &Subject,
        context: Option<&VfsOpContext>,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<TaskExecResult, MountError> {
        // First lifecycle-effect gate: a contract Task must still hold its
        // verified attachment generation before a backend command can start.
        self.task_context_for(id, caller, context, AttachmentOperation::TaskSpawn)?;
        // `exec` necessarily emits terminal fd data, so refuse before the
        // backend starts unless the same grant also permits publication. A
        // spawn-only identity must not leave an unpublishable task result.
        self.task_context_for(id, caller, context, AttachmentOperation::TaskPublish)?;
        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;
        let task_subject = sandbox
            .annotations
            .iter()
            .find(|annotation| annotation.key == "hyprstream.io/subject")
            .map(|annotation| Subject::new(annotation.value.clone()))
            .ok_or_else(|| MountError::PermissionDenied("task has no verified Subject".into()))?;
        if &task_subject != caller {
            return Err(MountError::PermissionDenied(
                "caller does not own this task's Subject-scoped namespace".into(),
            ));
        }
        // Fail fast, before touching the backend (#1447 r2/k3 Finding B): the
        // per-instance fd streams are a one-shot latch, so a second exec on an
        // instance whose streams already closed (stopped, destroyed, or a
        // prior exec completed) must not run the command only to discard its
        // output afterward.
        let streams = self.fd_streams_for(id).await?;
        if streams.stdout.is_closed().await || streams.stderr.is_closed().await {
            return Err(MountError::InvalidArgument(
                "instance's exec stdio is already closed (stopped, destroyed, or a prior \
                 exec already ran); each instance supports at most one exec"
                    .into(),
            ));
        }
        // Pool lookup and local stream-state checks both awaited. Fence the
        // exact caller grant immediately before backend execution.
        self.task_context_for(id, caller, context, AttachmentOperation::TaskSpawn)?;
        self.task_context_for(id, caller, context, AttachmentOperation::TaskPublish)?;
        let (exit_code, stdout, stderr) = self
            .pool
            .backend()
            .exec_sync(&sandbox, command, timeout_secs)
            .await
            .map_err(|e| MountError::Io(e.to_string()))?;
        // Second gate: the backend may have run for a while. Do not publish a
        // now-revoked Task's next MoQ/VFS output continuation.
        self.publish_exec_result(id, caller, context, &stdout, &stderr)
            .await?;
        Ok(TaskExecResult {
            exit_code,
            stdout,
            stderr,
        })
    }

    async fn close_fd_streams(
        &self,
        id: &str,
        caller: &Subject,
        context: Option<&VfsOpContext>,
    ) -> Result<(), MountError> {
        let context =
            self.task_context_for(id, caller, context, AttachmentOperation::TaskSignal)?;
        if let Some(context) = context.as_ref() {
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        let streams = self.fd_streams_for(id).await?;
        if let Some(context) = context.as_ref() {
            context.ensure_operation(AttachmentOperation::TaskSignal)?;
            context.ensure_operation(AttachmentOperation::TaskPublish)?;
        }
        streams.stdout.close(context.as_ref()).await?;
        streams.stderr.close(context.as_ref()).await?;
        Ok(())
    }

    /// Resolve the verb against the pool/backend for instance `id`.
    ///
    /// `caller` is the [`Subject`] that wrote the `ctl` verb (#1272): when a
    /// [`LifecyclePolicy`] is armed, destructive verbs (stop/kill/destroy)
    /// must pass it before any backend state is touched.
    async fn apply_verb(
        &self,
        id: &str,
        verb: Verb,
        caller: &Subject,
        context: Option<&VfsOpContext>,
    ) -> Result<String, MountError> {
        // Instances already marked terminal (destroyed through this mount)
        // are not present in the pool's active map any more; ctl ops on them
        // are rejected rather than silently no-op'd.
        if self.terminal.is_latched(&id.to_owned()) {
            return Err(MountError::InvalidArgument(format!(
                "instance {id} is already terminal"
            )));
        }

        // Standard Task allocations always carry an attachment context. Check
        // it before every ctl lifecycle effect; raw legacy instances keep the
        // compatibility path until all callers migrate.
        let task_operation = match &verb {
            Verb::Exec(_) | Verb::ExecJson(_) => AttachmentOperation::TaskSpawn,
            Verb::Start | Verb::Stop | Verb::Kill | Verb::Destroy => {
                AttachmentOperation::TaskSignal
            }
        };
        self.task_context_for(id, caller, context, task_operation)?;
        if matches!(verb, Verb::Stop | Verb::Kill | Verb::Destroy) {
            // Stopping/destroying also completes the fd carrier. Require that
            // terminal publication before altering the backend lifecycle.
            self.task_context_for(id, caller, context, AttachmentOperation::TaskPublish)?;
        }

        // MAC lifecycle gate (#1272): a destructive verb must be authorized by
        // the armed policy before we touch the backend. No policy ⇒ the
        // un-enforced status quo (the documented gap); a policy ⇒ fail-closed
        // (its `authorize` returns Err ⇒ PermissionDenied).
        if verb.is_lifecycle_destructive() {
            if let Some(policy) = &self.lifecycle {
                let lv = match verb {
                    Verb::Stop => LifecycleVerb::Stop,
                    Verb::Kill => LifecycleVerb::Kill,
                    Verb::Destroy => LifecycleVerb::Destroy,
                    // is_lifecycle_destructive() is false for Start, so this
                    // arm is unreachable here; kept exhaustive without `expect`.
                    Verb::Start => {
                        return Err(MountError::InvalidArgument(
                            "start is not a lifecycle-destructive verb".into(),
                        ));
                    }
                    Verb::Exec(_) | Verb::ExecJson(_) => {
                        return Err(MountError::InvalidArgument(
                            "exec is not a lifecycle-destructive verb".into(),
                        ));
                    }
                };
                policy
                    .authorize(caller, id, lv)
                    .map_err(MountError::PermissionDenied)?;
            }
        }

        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;
        // `pool.get` is asynchronous; do not let a revocation during lookup
        // authorize the subsequent lifecycle backend operation.
        self.task_context_for(id, caller, context, task_operation)?;
        if matches!(verb, Verb::Stop | Verb::Kill | Verb::Destroy) {
            self.task_context_for(id, caller, context, AttachmentOperation::TaskPublish)?;
        }

        match verb {
            Verb::Start => {
                // Sandboxes are started by `SandboxPool::acquire`; there is no
                // separate "start an existing-but-stopped instance" backend
                // op today. Accepted for grammar completeness/forward-compat:
                // if the instance exists and isn't terminal, report ok.
                Ok("ok: already started\n".to_owned())
            }
            Verb::Stop | Verb::Kill => {
                self.pool
                    .backend()
                    .stop(&sandbox)
                    .await
                    .map_err(|e| MountError::Io(e.to_string()))?;
                // A stopped task can no longer produce output. Latch both
                // reader streams to EOF so pending fd/1 and fd/2 readers wake.
                self.close_fd_streams(id, caller, context).await?;
                Ok("ok: stopped\n".to_owned())
            }
            Verb::Destroy => {
                self.pool
                    .backend()
                    .destroy(&sandbox)
                    .await
                    .map_err(|e| MountError::Io(e.to_string()))?;
                // Best-effort removal from the pool's active map: `release`
                // also tries to warm-pool/reset, which we don't want after an
                // explicit destroy, so we go straight through the backend
                // above and just drop bookkeeping here. If the id is no
                // longer active (e.g. concurrently released) that's fine.
                let _ = self.pool.release(id).await;
                self.terminal.latch(
                    id.to_owned(),
                    Terminal {
                        value: TerminalStatus {
                            last_state: sandbox.state,
                        },
                        latched_by: "ctl:destroy".to_owned(),
                    },
                );
                self.terminal_ids.lock().insert(id.to_owned());
                self.close_fd_streams(id, caller, context).await?;
                // Wake any `exit` reader already blocked on this id.
                self.waiter_for(id).notify_waiters();
                Ok("ok: destroyed\n".to_owned())
            }
            Verb::Exec(command) | Verb::ExecJson(command) => {
                let timeout_secs = self
                    .task_records
                    .lock()
                    .get(id)
                    .map_or(300, |record| record.timeout_secs);
                let result = self
                    .exec_task_with_context(id, caller, context, &command, timeout_secs)
                    .await?;
                self.record_exec_result(id, &result);
                Ok(format!("ok: exited {}\n", result.exit_code))
            }
        }
    }

    /// Live, non-blocking poll of an instance's current state for `status`.
    async fn read_status(
        &self,
        id: &str,
        context: Option<&VfsOpContext>,
    ) -> Result<String, MountError> {
        if let Some(term) = self.terminal.get(&id.to_owned()) {
            if let Some(context) = context {
                context.ensure_operation(AttachmentOperation::TaskRead)?;
            }
            return Ok(format!("{:?}\n", term.value.last_state));
        }
        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;
        if let Some(context) = context {
            context.ensure_operation(AttachmentOperation::TaskRead)?;
        }
        Ok(format!("{:?}\n", sandbox.state))
    }

    /// `exit`: blocks until the instance is terminal, then returns the
    /// retained status. A read that starts after the instance already
    /// latched terminal (a "late" reader, in read-then-subscribe terms)
    /// returns immediately with the retained value — no missed completion.
    /// A read that starts before (an "early" reader) blocks until `apply_verb`
    /// latches the `Destroy` outcome and wakes this id's waiter.
    async fn read_exit(
        &self,
        id: &str,
        context: Option<&VfsOpContext>,
    ) -> Result<String, MountError> {
        let key = id.to_owned();
        if let Some(term) = self.task_exit.get(&key) {
            if let Some(context) = context {
                context.ensure_operation(AttachmentOperation::TaskRead)?;
            }
            return Ok(format!("{}\n", term.value));
        }
        // Fast path: already latched — serve the retained value immediately,
        // exactly the "late reader" half of read-then-subscribe.
        if let Some(term) = self.terminal.get(&key) {
            if let Some(context) = context {
                context.ensure_operation(AttachmentOperation::TaskRead)?;
            }
            return Ok(term.value.render());
        }
        // Confirm the instance actually exists (vs. a bogus id) before
        // blocking on it.
        if self.pool.get(id).await.is_none() {
            return Err(MountError::NotFound(format!("instances/{id}")));
        }
        // Slow path: subscribe to this id's wake, then re-check the store —
        // closes the race where `apply_verb` latches + notifies between the
        // fast-path check above and the `notified()` registration below.
        let notify = self.waiter_for(id);
        loop {
            let notified = notify.notified();
            if let Some(term) = self.task_exit.get(&key) {
                if let Some(context) = context {
                    context.ensure_operation(AttachmentOperation::TaskRead)?;
                }
                return Ok(format!("{}\n", term.value));
            }
            if let Some(term) = self.terminal.get(&key) {
                if let Some(context) = context {
                    context.ensure_operation(AttachmentOperation::TaskRead)?;
                }
                return Ok(term.value.render());
            }
            notified.await;
            if let Some(term) = self.task_exit.get(&key) {
                if let Some(context) = context {
                    context.ensure_operation(AttachmentOperation::TaskRead)?;
                }
                return Ok(format!("{}\n", term.value));
            }
            if let Some(term) = self.terminal.get(&key) {
                if let Some(context) = context {
                    context.ensure_operation(AttachmentOperation::TaskRead)?;
                }
                return Ok(term.value.render());
            }
        }
    }

    /// Best-effort textual listing of the sandbox's mount-prefixes/namespace.
    ///
    /// `PodSandbox` doesn't currently carry a generic namespace listing —
    /// only a `sandbox_path` and an optional `console_socket`. We surface
    /// what's available; backends with no accessible namespace info return
    /// an empty/placeholder listing rather than failing.
    async fn read_ns(
        &self,
        id: &str,
        context: Option<&VfsOpContext>,
    ) -> Result<String, MountError> {
        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;
        if let Some(context) = context {
            context.ensure_operation(AttachmentOperation::TaskRead)?;
        }

        let mut lines = vec![format!("runtime={}", sandbox.runtime_handler)];
        lines.push(format!("path={}", sandbox.sandbox_path().display()));
        if let Some(console) = sandbox.console_socket() {
            lines.push(format!("console={}", console.display()));
        }
        Ok(format!("{}\n", lines.join("\n")))
    }
}

#[async_trait]
impl TaskService for ExecMount {
    async fn spawn_task(&self, request: TaskSpawnRequest) -> Result<TaskHandle, TaskError> {
        // `/exec/clone` is modeled as this explicit allocation effect. It is
        // intentionally not a read: reaching this method allocates a sandbox
        // only after the attachment generation check below succeeds.
        request.validate_for_spawn()?;
        let is_argv = matches!(&request.payload, TaskPayload::Argv(_));
        if is_argv {
            // Argv execution necessarily emits terminal fd data. Fail before
            // allocation unless this current binding also grants publication.
            request.ensure_argv_spawn_permitted()?;
        }
        let TaskSpawnRequest {
            parent_task: _,
            attachment,
            namespace_manifest,
            payload,
            constraints,
        } = request;
        let command = match payload {
            TaskPayload::Argv(command) => command,
            TaskPayload::Content(_) => {
                return Err(TaskError::InvalidRequest(
                    "content payload materialization is not implemented by this Worker backend"
                        .into(),
                ));
            }
        };

        // `namespace_manifest` is retained as a task-contract provenance
        // commitment only. This spike does not derive a `Namespace` from it
        // or pass an effective mount description to `PodSandboxConfig`.

        if let Some(requested_backend) = constraints.backend_class.as_deref() {
            if requested_backend != self.pool.backend().backend_type() {
                return Err(TaskError::InvalidRequest(format!(
                    "requested backend {requested_backend} does not match this fail-closed pool's {} backend",
                    self.pool.backend().backend_type()
                )));
            }
        }

        let context = VfsOpContext::from_attachment_grant(attachment.operation_grant());
        let id = self
            .pool
            .acquire(
                context.subject(),
                &super::client::PodSandboxConfig::default(),
            )
            .await
            .map_err(|error| TaskError::Backend(error.to_string()))?;

        // The attachment may have been revoked while placement was in flight.
        // Do not leave a newly allocated sandbox reachable if that happened.
        if let Err(error) = attachment.ensure(AttachmentOperation::TaskSpawn) {
            let _ = self.pool.release(&id).await;
            return Err(error);
        }
        if let Err(error) = attachment.ensure(AttachmentOperation::TaskPublish) {
            let _ = self.pool.release(&id).await;
            return Err(error);
        }

        let timeout_secs = constraints.timeout_secs.unwrap_or(300);
        self.task_records.lock().insert(
            id.clone(),
            TaskRecord {
                attachment_id: context.attachment_id().clone(),
                authority_generation: context.authority_generation(),
                subject: context.subject().clone(),
                namespace_manifest,
                timeout_secs,
                state: TaskState::Running,
                content_records: Vec::new(),
            },
        );

        let result = match self
            .exec_task_with_context(
                &id,
                context.subject(),
                Some(&context),
                &command,
                timeout_secs,
            )
            .await
        {
            Ok(result) => result,
            Err(error) => {
                self.rollback_task_allocation(&id).await;
                return Err(Self::task_error_for_binding(
                    error,
                    &attachment,
                    &[
                        AttachmentOperation::TaskSpawn,
                        AttachmentOperation::TaskPublish,
                    ],
                ));
            }
        };
        self.record_exec_result(&id, &result);

        let state = self
            .task_records
            .lock()
            .get(&id)
            .map(|record| record.state)
            .ok_or_else(|| TaskError::NotFound(TaskId::from_worker_instance(&id)))?;
        Ok(Self::task_handle(&id, state))
    }

    async fn attach_task(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskHandle, TaskError> {
        let record = self.bound_task_record(task, &attachment, AttachmentOperation::TaskAttach)?;
        Ok(Self::task_handle(task.as_str(), record.state))
    }

    async fn signal_task(&self, request: TaskSignalRequest) -> Result<(), TaskError> {
        request.ensure_permitted()?;
        self.bound_task_record(
            &request.task,
            &request.attachment,
            AttachmentOperation::TaskSignal,
        )?;
        let verb = match request.signal {
            TaskSignal::Stop => Verb::Stop,
            TaskSignal::Kill => Verb::Kill,
            TaskSignal::Destroy => Verb::Destroy,
        };
        let context = VfsOpContext::from_attachment_grant(request.attachment.operation_grant());
        self.apply_verb(
            request.task.as_str(),
            verb,
            request.attachment.subject(),
            Some(&context),
        )
        .await
        .map_err(|error| {
            Self::task_error_for_binding(
                error,
                &request.attachment,
                &[
                    AttachmentOperation::TaskSignal,
                    AttachmentOperation::TaskPublish,
                ],
            )
        })?;
        if let Some(record) = self.task_records.lock().get_mut(request.task.as_str()) {
            record.state = TaskState::Cancelled;
        }
        Ok(())
    }

    async fn wait_task(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskResult, TaskError> {
        // This spike's deterministic fake backend completes argv tasks during
        // spawn. A non-terminal artifact task has no guest wait primitive yet,
        // so it fails explicitly rather than pretending an empty read is EOF.
        self.task_result(task, attachment).await
    }

    async fn snapshot_task(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskSnapshot, TaskError> {
        let record = self.bound_task_record(task, &attachment, AttachmentOperation::TaskRead)?;
        Ok(TaskSnapshot {
            task: task.clone(),
            namespace_manifest: record.namespace_manifest,
            state: record.state,
        })
    }

    async fn task_result(
        &self,
        task: &TaskId,
        attachment: TaskAttachmentBinding,
    ) -> Result<TaskResult, TaskError> {
        let record = self.bound_task_record(task, &attachment, AttachmentOperation::TaskRead)?;
        match record.state {
            TaskState::Exited { .. } | TaskState::Cancelled => Ok(TaskResult {
                task: task.clone(),
                state: record.state,
                content_records: record.content_records,
            }),
            TaskState::Pending | TaskState::Running => Err(TaskError::InvalidRequest(
                "task is not terminal; no guest wait bridge is installed yet".into(),
            )),
        }
    }

    async fn spawn_child_task(&self, request: TaskSpawnRequest) -> Result<TaskHandle, TaskError> {
        let _ = request;
        Err(TaskError::ChildDelegationRequired)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Mount for ExecMount {
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError> {
        // Compatibility entry point for legacy Subject-scoped instances. A
        // standard Task requires `walk_with_context`, so this method cannot
        // turn a same-Subject caller into an attachment authority.
        self.walk_inner(components, caller, None).await
    }

    async fn walk_with_context(
        &self,
        components: &[&str],
        context: &VfsOpContext,
    ) -> Result<Fid, MountError> {
        ExecMount::walk_with_context(self, components, context).await
    }

    async fn open(&self, fid: &mut Fid, _mode: u8, caller: &Subject) -> Result<(), MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.verify_fid_context(&inner.kind, inner.context.as_ref(), caller)?;
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.verify_fid_context(&inner.kind, inner.context.as_ref(), caller)?;

        let data = match &inner.kind {
            ExecFidKind::Ctl(_) => {
                // Ctl pattern: after a write, read returns the verb result.
                inner.write_buf.latched()
            }
            ExecFidKind::Status(id) => self
                .read_status(id, inner.context.as_ref())
                .await?
                .into_bytes(),
            ExecFidKind::Exit(id) => self
                .read_exit(id, inner.context.as_ref())
                .await?
                .into_bytes(),
            ExecFidKind::Ns(id) => self.read_ns(id, inner.context.as_ref()).await?.into_bytes(),
            // `read_until_closed` already applies `offset`/`count` (it slices
            // the retained buffer at `[offset, offset+count)`), so its result
            // must be returned directly here — falling through to the shared
            // offset tail below would re-apply `offset` a second time and
            // silently corrupt or premature-EOF any nonzero-offset fd read
            // (#1447 r2/k3 Finding A).
            ExecFidKind::Fd(id, 1) => {
                return Ok(self
                    .fd_streams_for(id)
                    .await?
                    .stdout
                    .read_until_closed(inner.context.as_ref(), offset, count)
                    .await?);
            }
            ExecFidKind::Fd(id, 2) => {
                return Ok(self
                    .fd_streams_for(id)
                    .await?
                    .stderr
                    .read_until_closed(inner.context.as_ref(), offset, count)
                    .await?);
            }
            ExecFidKind::Fd(_, _) => {
                return Err(MountError::NotFound("invalid exec fd".into()));
            }
            ExecFidKind::InstancesDir | ExecFidKind::InstanceDir(_) | ExecFidKind::FdDir(_) => {
                return Err(MountError::IsDirectory("use readdir".into()));
            }
        };

        let start = (offset as usize).min(data.len());
        let end = start.saturating_add(count as usize).min(data.len());
        Ok(data[start..end].to_vec())
    }

    async fn write(
        &self,
        fid: &Fid,
        _offset: u64,
        data: &[u8],
        caller: &Subject,
    ) -> Result<u32, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.verify_fid_context(&inner.kind, inner.context.as_ref(), caller)?;

        match &inner.kind {
            ExecFidKind::Ctl(id) => {
                let text = String::from_utf8_lossy(data);
                let verb = Verb::parse(&text).ok_or_else(|| {
                    MountError::InvalidArgument(format!("unknown ctl verb: {}", text.trim()))
                })?;
                // `caller` is now threaded into the lifecycle gate inside
                // `apply_verb` (#1272): a destructive verb is mediated by the
                // armed [`LifecyclePolicy`] before the backend is touched.
                let result = self
                    .apply_verb(id, verb, caller, inner.context.as_ref())
                    .await;
                let response_bytes = match result {
                    Ok(s) => s.into_bytes(),
                    Err(e) => format!("error: {e}\n").into_bytes(),
                };
                // Ctl write→read pattern: latch the result for a subsequent
                // read on the same fid. The `Mount::write(&self, &Fid, ..)`
                // signature needs interior mutability; `parking_lot::Mutex`
                // gives it safely through `&` (the same primitive
                // `hyprstream-workers-tcl`/`-python` and `devfile::DevFileState`
                // use on bases that contain #615), avoiding the `unsafe *mut`
                // cast-through-`&` the pre-review draft used — that would be a
                // data race if two futures ever held `&Fid` to one fid
                // concurrently (e.g. a 9P server multiplexing Twrite on one fid).
                inner.write_buf.latch(response_bytes);
                Ok(data.len() as u32)
            }
            ExecFidKind::Fd(_, 1 | 2) => Err(MountError::NotSupported(
                "fd/1 and fd/2 are read-only".into(),
            )),
            _ => Err(MountError::NotSupported("read-only".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        if let Some(context) = inner.context.as_ref() {
            context.ensure_operation(AttachmentOperation::TaskRead)?;
        }
        if !matches!(&inner.kind, ExecFidKind::InstancesDir) {
            self.verify_fid_context(&inner.kind, inner.context.as_ref(), caller)?;
        }

        match &inner.kind {
            ExecFidKind::InstancesDir => {
                let records = self.task_records.lock().clone();
                let active = self.pool.list_active().await;
                if let Some(context) = inner.context.as_ref() {
                    context.ensure_operation(AttachmentOperation::TaskRead)?;
                }
                let mut ids: Vec<String> = active
                    .into_iter()
                    .filter(|sandbox| {
                        records.get(&sandbox.id).is_none_or(|record| {
                            inner.context.as_ref().is_some_and(|context| {
                                context.subject() == caller
                                    && context.attachment_id() == &record.attachment_id
                                    && context.authority_generation() == record.authority_generation
                                    && context
                                        .ensure_operation(AttachmentOperation::TaskRead)
                                        .is_ok()
                            })
                        })
                    })
                    .map(|s| s.id)
                    .collect();
                // Also surface ids that have been destroyed-through-this-mount
                // (so `exit`/`status` remain reachable for stragglers that
                // already read "not yet terminal" and need to poll again).
                for id in self.terminal_ids.lock().iter() {
                    if records.get(id).is_some_and(|record| {
                        !inner.context.as_ref().is_some_and(|context| {
                            context.subject() == caller
                                && context.attachment_id() == &record.attachment_id
                                && context.authority_generation() == record.authority_generation
                                && context
                                    .ensure_operation(AttachmentOperation::TaskRead)
                                    .is_ok()
                        })
                    }) {
                        continue;
                    }
                    if !ids.contains(id) {
                        ids.push(id.clone());
                    }
                }
                Ok(ids
                    .into_iter()
                    .map(|name| DirEntry {
                        name,
                        is_dir: true,
                        size: 0,
                        stat: None,
                    })
                    .collect())
            }
            ExecFidKind::InstanceDir(_) => Ok(vec![
                DirEntry {
                    name: "ctl".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "status".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "exit".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "ns".into(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                },
                DirEntry {
                    name: "fd".into(),
                    is_dir: true,
                    size: 0,
                    stat: None,
                },
            ]),
            ExecFidKind::FdDir(_) => Ok((1..=2)
                .map(|fd| DirEntry {
                    name: fd.to_string(),
                    is_dir: false,
                    size: 0,
                    stat: None,
                })
                .collect()),
            _ => Err(MountError::NotDirectory(format!("{:?}", inner.kind))),
        }
    }

    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        self.verify_fid_context(&inner.kind, inner.context.as_ref(), caller)?;

        const QTDIR: u8 = 0x80;
        let (name, qtype) = match &inner.kind {
            ExecFidKind::InstancesDir => ("instances".to_owned(), QTDIR),
            ExecFidKind::InstanceDir(id) => (id.clone(), QTDIR),
            ExecFidKind::Ctl(_) => ("ctl".to_owned(), 0),
            ExecFidKind::Status(_) => ("status".to_owned(), 0),
            ExecFidKind::Exit(_) => ("exit".to_owned(), 0),
            ExecFidKind::Ns(_) => ("ns".to_owned(), 0),
            ExecFidKind::FdDir(_) => ("fd".to_owned(), QTDIR),
            ExecFidKind::Fd(_, fd) => (fd.to_string(), 0),
        };

        Ok(Stat {
            qtype,
            version: 0,
            path: 0,
            size: 0,
            name,
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::PoolConfig;
    use crate::error::Result as WorkerResult;
    use crate::runtime::backend::{SandboxBackend, SandboxHandle};
    use crate::runtime::client::{LinuxContainerResources, PodSandboxConfig};
    use crate::runtime::sandbox::PodSandbox;
    use crate::runtime::{CloneAdmission, CloneAdmissionSource, ExecRootMount};
    use hyprstream_9p::{
        AttachAuthorizer, Backend, MountBackend, SessionContext, VerifiedAttach,
        VerifiedAttachIdentity,
    };
    use hyprstream_rpc::auth::mac::{CompartmentSet, Level, SecurityContext, VerifiedKeyMaterial};
    use hyprstream_rpc::moq_stream::{STREAM_TRACK, verify_moq_frame};
    use hyprstream_rpc::streaming::{StreamPayload, StreamVerifier};
    use hyprstream_rpc::{AttachmentOperation, VerifiedAttachment};
    use hyprstream_rpc_std::task::NamespaceManifestDigest;
    use moq_net::Track;
    use std::any::Any;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Minimal in-memory fake `SandboxBackend` for exercising `ExecMount`
    /// without any real isolation runtime (no kata/nspawn/wasm dependency).
    #[derive(Debug)]
    struct FakeHandle;
    impl SandboxHandle for FakeHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Default)]
    struct FakeBackend {
        /// Tracks whether `destroy` was called, for assertions.
        destroyed: AtomicBool,
        stopped: AtomicBool,
        fail_exec: AtomicBool,
        revoke_during_exec: PmMutex<Option<VerifiedAttachment>>,
        commands: PmMutex<Vec<Vec<String>>>,
        timeouts: PmMutex<Vec<u64>>,
    }

    #[async_trait]
    impl SandboxBackend for FakeBackend {
        fn backend_type(&self) -> &'static str {
            "fake"
        }

        fn is_available(&self) -> bool {
            true
        }

        async fn initialize(&self, _config: &PoolConfig) -> WorkerResult<()> {
            Ok(())
        }

        async fn start(
            &self,
            _sandbox: &mut PodSandbox,
            _config: &PodSandboxConfig,
            _pool_config: &PoolConfig,
            _annotations: &HashMap<String, String>,
        ) -> WorkerResult<Arc<dyn SandboxHandle>> {
            Ok(Arc::new(FakeHandle))
        }

        async fn stop(&self, _sandbox: &PodSandbox) -> WorkerResult<()> {
            self.stopped.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn destroy(&self, _sandbox: &PodSandbox) -> WorkerResult<()> {
            self.destroyed.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn reset(&self, _sandbox: &mut PodSandbox) -> WorkerResult<bool> {
            // Ephemeral: never reusable, mirrors nspawn's shape.
            Ok(false)
        }

        async fn get_pids(&self, _sandbox: &PodSandbox) -> WorkerResult<Vec<u32>> {
            Ok(vec![])
        }

        fn supports_exec(&self) -> bool {
            true
        }

        async fn exec_sync(
            &self,
            _sandbox: &PodSandbox,
            command: &[String],
            timeout_secs: u64,
        ) -> WorkerResult<(i32, Vec<u8>, Vec<u8>)> {
            if let Some(attachment) = self.revoke_during_exec.lock().take() {
                attachment.revoke().map_err(|error| {
                    crate::error::WorkerError::ExecFailed(format!("test revoke failed: {error}"))
                })?;
            }
            if self.fail_exec.load(Ordering::SeqCst) {
                return Err(crate::error::WorkerError::ExecFailed(
                    "scripted fake exec failure".into(),
                ));
            }
            self.commands.lock().push(command.to_vec());
            self.timeouts.lock().push(timeout_secs);
            Ok((0, b"fake stdout\n".to_vec(), b"fake stderr\n".to_vec()))
        }

        async fn update_resources(
            &self,
            _sandbox: &PodSandbox,
            _resources: &LinuxContainerResources,
        ) -> WorkerResult<()> {
            Ok(())
        }
    }

    async fn make_pool_with(backend: Arc<dyn SandboxBackend>) -> Arc<SandboxPool> {
        let config = PoolConfig {
            max_sandboxes: 10,
            warm_pool_size: 0,
            ..Default::default()
        };
        Arc::new(SandboxPool::new(config, backend))
    }

    async fn make_pool() -> Arc<SandboxPool> {
        make_pool_with(Arc::new(FakeBackend::default())).await
    }

    fn subject() -> Subject {
        Subject::anonymous()
    }

    /// Distinct from `subject()` above: that one is the *VFS caller* passed
    /// to `ExecMount`'s `Mount` trait methods (anonymous is fine there — this
    /// projection doesn't itself enforce per-caller authorization). This one
    /// is the *admission* identity `SandboxPool::acquire` requires (#525 P2)
    /// — must be non-anonymous, or every `pool.acquire()` call below would
    /// fail closed.
    fn admission_subject() -> Subject {
        Subject::new("test-user")
    }

    #[tokio::test]
    async fn clone_device_adapts_pending_task_to_ctl_fd_and_numeric_exit() {
        struct FixedAdmission;
        impl CloneAdmissionSource for FixedAdmission {
            fn admit(&self, _context: &VfsOpContext) -> Result<CloneAdmission, MountError> {
                Ok(CloneAdmission::from_trusted_policy(
                    NamespaceManifestDigest::from_description_bytes(b"/work=fake\n"),
                    TaskRuntimeConstraints {
                        backend_class: Some("fake".into()),
                        timeout_secs: Some(30),
                    },
                ))
            }
        }

        struct StaticGrantAuthorizer(VerifiedAttach);

        #[async_trait]
        impl AttachAuthorizer for StaticGrantAuthorizer {
            async fn authenticate(
                &self,
                _uname: &str,
                _aname: &str,
            ) -> Result<VerifiedAttach, MountError> {
                Ok(self.0.clone())
            }
        }

        let fake = Arc::new(FakeBackend::default());
        let instances = Arc::new(ExecMount::new(make_pool_with(fake.clone()).await));
        let root = Arc::new(ExecRootMount::new(
            Arc::clone(&instances),
            Arc::new(FixedAdmission),
        ));
        let owner = Subject::new("clone-owner");
        let attachment = VerifiedAttachment::for_test_local_root(owner.clone()).unwrap();
        let grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSpawn,
            AttachmentOperation::TaskRead,
            AttachmentOperation::TaskPublish,
        ]);
        let identity =
            VerifiedAttachIdentity::from_verified_credential("clone-owner", "fixture-tenant");
        let verified = VerifiedAttach::try_new_with_operation_grant(
            identity.clone(),
            owner.clone(),
            SessionContext::from_verified_clearance(
                identity,
                SecurityContext::new(
                    Level::Secret,
                    CompartmentSet::EMPTY,
                    VerifiedKeyMaterial::Classical,
                ),
            ),
            &grant,
        )
        .unwrap();

        // A same-Subject connection with no independently-issued Task grant
        // can name `clone`, but cannot allocate from it.
        let legacy = MountBackend::new(root.clone(), owner.clone());
        legacy.walk(0, 10, &["clone".into()]).await.unwrap();
        assert!(legacy.open(10, 0).await.is_err());

        // This is the fake-backed Fersh/Wanix adapter fixture: the verified
        // attach supplies the opaque grant once, then the 9P backend carries
        // it over walk/open/read/write rather than deriving authority from a
        // subject, path, or generic 9P write.
        let backend =
            MountBackend::with_authorizer(root, Arc::new(StaticGrantAuthorizer(verified)));
        backend.attach("fixture-ticket", "", None).await.unwrap();
        // Discovery operations and a bare clone read are non-effects: they
        // cannot allocate a sandbox before the context-walked `open`.
        backend.walk(0, 9, &[]).await.unwrap();
        backend.stat(9).await.unwrap();
        backend.readdir(9, 0, 4096).await.unwrap();
        backend.walk(0, 1, &["clone".into()]).await.unwrap();
        assert!(backend.read(1, 0, 4096).await.is_err());
        assert!(instances.pool.list_active().await.is_empty());
        backend.open(1, 0).await.unwrap();
        backend.open(1, 0).await.unwrap();
        assert_eq!(instances.pool.list_active().await.len(), 1);
        let id = String::from_utf8(backend.read(1, 0, 4096).await.unwrap())
            .unwrap()
            .trim()
            .to_owned();
        assert!(!id.is_empty());

        backend
            .walk(0, 2, &["instances".into(), id.clone(), "ctl".into()])
            .await
            .unwrap();
        backend.open(2, 2).await.unwrap();
        backend
            .write(2, 0, br#"exec-json ["echo","arg with space"]"#)
            .await
            .unwrap();
        assert!(
            String::from_utf8(backend.read(2, 0, 4096).await.unwrap())
                .unwrap()
                .starts_with("ok: exited 0")
        );
        assert_eq!(
            fake.commands.lock().as_slice(),
            &[vec!["echo".to_owned(), "arg with space".to_owned()]],
            "exec-json must reach the backend as the exact JSON argv"
        );
        assert_eq!(fake.timeouts.lock().as_slice(), &[30]);

        backend
            .walk(
                0,
                3,
                &["instances".into(), id.clone(), "fd".into(), "1".into()],
            )
            .await
            .unwrap();
        assert_eq!(backend.read(3, 0, 5).await.unwrap(), b"fake ");
        backend
            .walk(0, 4, &["instances".into(), id, "exit".into()])
            .await
            .unwrap();
        assert_eq!(backend.read(4, 0, 64).await.unwrap(), b"0\n");

        // The fid retains the grant issued at attach, so revocation blocks
        // the next ctl effect before a second backend command can begin.
        attachment.revoke().unwrap();
        assert!(backend.read(3, 5, 4096).await.is_err());
        assert!(
            backend
                .write(2, 0, br#"exec-json ["echo","must not run"]"#)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn list_instances_empty() {
        let pool = make_pool().await;
        let mount = ExecMount::new(pool);
        let fid = mount.walk(&[], &subject()).await.unwrap();
        let entries = mount.readdir(&fid, &subject()).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn list_instances_after_acquire() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let fid = mount.walk(&[], &subject()).await.unwrap();
        let entries = mount.readdir(&fid, &subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&id.as_str()));
        assert!(entries.iter().all(|e| e.is_dir));
    }

    #[tokio::test]
    async fn walk_unknown_instance_not_found() {
        let pool = make_pool().await;
        let mount = ExecMount::new(pool);
        let result = mount.walk(&["nonexistent", "status"], &subject()).await;
        assert!(matches!(result, Err(MountError::NotFound(_))));
    }

    #[tokio::test]
    async fn read_status_reflects_pool_state() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "status"], &subject()).await.unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.contains("SandboxReady"), "got: {text}");
    }

    #[tokio::test]
    async fn ctl_stop_invokes_backend() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        let written = mount.write(&fid, 0, b"stop", &subject()).await.unwrap();
        assert_eq!(written, 4);

        let data = mount.read(&fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.starts_with("ok:"), "got: {text}");
    }

    /// `ctl` response state is a property of the open fid, not of the mount
    /// or task.  Writing two lifecycle verbs before either reply is read must
    /// preserve both results for their respective fids.
    #[tokio::test]
    async fn ctl_response_is_isolated_per_fid() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);
        let caller = subject();

        let mut start = mount.walk(&[&id, "ctl"], &caller).await.unwrap();
        let mut stop = mount.walk(&[&id, "ctl"], &caller).await.unwrap();
        mount.open(&mut start, 2, &caller).await.unwrap();
        mount.open(&mut stop, 2, &caller).await.unwrap();

        mount.write(&start, 0, b"start", &caller).await.unwrap();
        mount.write(&stop, 0, b"stop", &caller).await.unwrap();

        assert_eq!(mount.read(&start, 0, 4, &caller).await.unwrap(), b"ok: ");
        assert_eq!(
            mount.read(&start, 4, 4096, &caller).await.unwrap(),
            b"already started\n"
        );
        assert_eq!(
            mount.read(&stop, 0, 4096, &caller).await.unwrap(),
            b"ok: stopped\n"
        );
    }

    #[tokio::test]
    async fn ctl_unknown_verb_rejected() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        let result = mount.write(&fid, 0, b"frobnicate", &subject()).await;
        assert!(matches!(result, Err(MountError::InvalidArgument(_))));
    }

    /// Early waiter: a `read()` on `exit` that starts BEFORE the instance is
    /// terminal blocks, then unblocks with the correct status once
    /// `apply_verb`'s `Destroy` arm latches it from another task — the core
    /// EV7 read-then-subscribe guarantee (#668).
    #[tokio::test]
    async fn exit_blocks_then_unblocks_on_destroy() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = Arc::new(ExecMount::new(pool));

        let reader = {
            let mount = Arc::clone(&mount);
            let id = id.clone();
            tokio::spawn(async move {
                let mut fid = mount.walk(&[&id, "exit"], &subject()).await.unwrap();
                mount.open(&mut fid, 0, &subject()).await.unwrap();
                mount.read(&fid, 0, 4096, &subject()).await.unwrap()
            })
        };

        // The reader should be blocked (not yet resolved) — give it a beat to
        // reach the `.await` inside `read_exit`'s slow path, then confirm it
        // hasn't completed on its own.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !reader.is_finished(),
            "exit read must block before terminal"
        );

        // Destroy from a separate task — this is what latches + wakes the
        // blocked reader above.
        let mut ctl_fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut ctl_fid, 2, &subject()).await.unwrap();
        mount
            .write(&ctl_fid, 0, b"destroy", &subject())
            .await
            .unwrap();
        mount.read(&ctl_fid, 0, 4096, &subject()).await.unwrap();

        let data = tokio::time::timeout(std::time::Duration::from_secs(2), reader)
            .await
            .expect("exit read did not unblock after destroy")
            .unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.starts_with("exited"), "got: {text}");
    }

    /// Late waiter: a `read()` on `exit` that starts AFTER the instance is
    /// already terminal is served the retained status immediately, without
    /// blocking (#668).
    #[tokio::test]
    async fn exit_after_destroy_reports_terminal() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        // Drive destroy through ctl.
        let mut ctl_fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut ctl_fid, 2, &subject()).await.unwrap();
        mount
            .write(&ctl_fid, 0, b"destroy", &subject())
            .await
            .unwrap();
        let ctl_result = mount.read(&ctl_fid, 0, 4096, &subject()).await.unwrap();
        assert!(String::from_utf8(ctl_result).unwrap().starts_with("ok:"));

        // exit should now report terminal status immediately (late reader —
        // the retained value is served without blocking on the store).
        let mut exit_fid = mount.walk(&[&id, "exit"], &subject()).await.unwrap();
        mount.open(&mut exit_fid, 0, &subject()).await.unwrap();
        let data = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            mount.read(&exit_fid, 0, 4096, &subject()),
        )
        .await
        .expect("late exit read must not block")
        .unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.starts_with("exited"), "got: {text}");

        // status should also reflect terminal state via the retained store.
        let mut status_fid = mount.walk(&[&id, "status"], &subject()).await.unwrap();
        mount.open(&mut status_fid, 0, &subject()).await.unwrap();
        let data = mount.read(&status_fid, 0, 4096, &subject()).await.unwrap();
        assert!(!data.is_empty());
    }

    #[tokio::test]
    async fn ctl_on_terminal_instance_rejected() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let mut ctl_fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut ctl_fid, 2, &subject()).await.unwrap();
        mount
            .write(&ctl_fid, 0, b"destroy", &subject())
            .await
            .unwrap();

        let result = mount.write(&ctl_fid, 0, b"stop", &subject()).await;
        // apply_verb returns an error result captured into the ctl response
        // buffer rather than propagating through `write`'s Result, so check
        // the *next* destroy attempt is rejected via a fresh fid instead.
        assert!(result.is_ok()); // write() itself always "succeeds" (buffers the error text)
        let data = mount.read(&ctl_fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.starts_with("error:"), "got: {text}");
    }

    #[tokio::test]
    async fn read_ns_returns_placeholder_info() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "ns"], &subject()).await.unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.contains("runtime="), "got: {text}");
        assert!(text.contains("path="), "got: {text}");
    }

    #[tokio::test]
    async fn readdir_instance_dir_lists_files() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let fid = mount.walk(&[&id], &subject()).await.unwrap();
        let entries = mount.readdir(&fid, &subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"ctl"));
        assert!(names.contains(&"status"));
        assert!(names.contains(&"exit"));
        assert!(names.contains(&"ns"));
        assert!(names.contains(&"fd"));

        let fd_dir = mount.walk(&[&id, "fd"], &subject()).await.unwrap();
        let fd_entries = mount.readdir(&fd_dir, &subject()).await.unwrap();
        let fd_names: Vec<&str> = fd_entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(fd_names, ["1", "2"]);
    }

    #[tokio::test]
    async fn fd_output_blocks_until_runtime_publishes_then_latches_eof() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = Arc::new(ExecMount::new(pool));

        let reader = {
            let mount = Arc::clone(&mount);
            let id = id.clone();
            tokio::spawn(async move {
                let mut fid = mount.walk(&[&id, "fd", "1"], &subject()).await.unwrap();
                mount.open(&mut fid, 0, &subject()).await.unwrap();
                mount.read(&fid, 0, 4096, &subject()).await.unwrap()
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !reader.is_finished(),
            "fd/1 must wait for terminal task output"
        );

        mount
            .publish_exec_result(
                &id,
                &subject(),
                None,
                b"build output\n",
                b"warning output\n",
            )
            .await
            .unwrap();

        let stdout = tokio::time::timeout(std::time::Duration::from_secs(2), reader)
            .await
            .expect("fd/1 reader did not wake on task completion")
            .unwrap();
        assert_eq!(stdout, b"build output\n");

        let mut stderr_fid = mount.walk(&[&id, "fd", "2"], &subject()).await.unwrap();
        mount.open(&mut stderr_fid, 0, &subject()).await.unwrap();
        assert_eq!(
            mount.read(&stderr_fid, 0, 4096, &subject()).await.unwrap(),
            b"warning output\n"
        );
        assert!(
            mount
                .read(&stderr_fid, 4096, 4096, &subject())
                .await
                .unwrap()
                .is_empty(),
            "terminal stderr must report EOF after retained bytes"
        );
    }

    #[tokio::test]
    async fn fd_truncation_is_explicit_in_local_and_moq_terminal_views() {
        let origin = MoqStreamOrigin::standalone()
            .with_prefix("test/exec")
            .build();
        let fd = TaskFdStream::new(&origin, "truncated", 1).unwrap();
        let oversized = vec![b'x'; MAX_RETAINED_FD_BYTES + 1];
        fd.append(None, &oversized).await.unwrap();
        fd.close(None).await.unwrap();

        let (topic, mac_key, local) = {
            let state = fd.state.lock().await;
            (
                state.publisher.topic().to_owned(),
                state.mac_key,
                state.bytes.clone(),
            )
        };
        assert_eq!(local.len(), MAX_RETAINED_FD_BYTES);
        assert!(local.ends_with(FD_TRUNCATION_MARKER));

        // Consume the same carrier Objects that an Iroh/MoQ client would
        // verify. The local tail marker and authenticated Complete metadata
        // must agree on the bounded terminal result.
        let path = origin.broadcast_path(&topic);
        let broadcast = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            origin.consumer().announced_broadcast(path.as_str()),
        )
        .await
        .unwrap()
        .expect("fd broadcast must be announced");
        let mut track = broadcast
            .subscribe_track(&Track::new(STREAM_TRACK))
            .unwrap();
        let mut verifier = StreamVerifier::new(mac_key, topic.clone());
        let mut payloads = Vec::new();
        for _ in 0..2 {
            let mut group =
                tokio::time::timeout(std::time::Duration::from_secs(2), track.next_group())
                    .await
                    .unwrap()
                    .unwrap()
                    .expect("fd carrier group must exist");
            let frame = tokio::time::timeout(std::time::Duration::from_secs(2), group.read_frame())
                .await
                .unwrap()
                .unwrap()
                .expect("fd carrier frame must exist");
            payloads.extend(verify_moq_frame(&mut verifier, &topic, &frame).unwrap());
        }
        assert!(
            matches!(&payloads[0], StreamPayload::Data(bytes) if bytes.len() == MAX_RETAINED_FD_BYTES)
        );
        assert!(
            matches!(&payloads[1], StreamPayload::Complete(metadata) if metadata == FD_TRUNCATED_METADATA)
        );
    }

    /// Regression (#1447 r2/k3 Finding A): `read_until_closed` already slices
    /// the retained buffer at `[offset, offset+count)`; `Mount::read`'s shared
    /// offset tail must NOT re-apply `offset` to that already-offset result.
    /// Every other fd test in this suite reads at offset 0 or at an offset
    /// past end-of-buffer, both of which look identical under the buggy
    /// double-application and the correct behavior — only a nonzero,
    /// in-range offset (mid-buffer, on a buffer bigger than one read) can
    /// distinguish them. With a double application, offset=3/count=4 over a
    /// 10-byte buffer computes `bytes[3..7]` ("3456") in `read_until_closed`,
    /// then re-applies offset 3 to that 4-byte result (`"3456"[3..]` = "6"
    /// only) — silently wrong data, not even an empty read.
    #[tokio::test]
    async fn fd_read_at_nonzero_offset_is_not_double_applied() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        mount
            .publish_exec_result(&id, &subject(), None, b"0123456789", b"")
            .await
            .unwrap();

        let mut fid = mount.walk(&[&id, "fd", "1"], &subject()).await.unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();

        let data = mount.read(&fid, 3, 4, &subject()).await.unwrap();
        assert_eq!(
            data, b"3456",
            "offset=3,count=4 over a 10-byte retained buffer must return bytes[3..7] \
             verbatim from read_until_closed, not re-offset a second time"
        );
    }

    #[tokio::test]
    async fn ctl_exec_is_projected_to_terminal_fd_streams() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        let mut ctl = mount
            .walk(&[&id, "ctl"], &admission_subject())
            .await
            .unwrap();
        mount.open(&mut ctl, 2, &admission_subject()).await.unwrap();
        mount
            .write(&ctl, 0, b"exec build", &admission_subject())
            .await
            .unwrap();
        assert_eq!(
            mount
                .read(&ctl, 0, 4096, &admission_subject())
                .await
                .unwrap(),
            b"ok: exited 0\n"
        );
        let mut stdout = mount.walk(&[&id, "fd", "1"], &subject()).await.unwrap();
        mount.open(&mut stdout, 0, &subject()).await.unwrap();
        assert_eq!(
            mount.read(&stdout, 0, 4096, &subject()).await.unwrap(),
            b"fake stdout\n"
        );
        let mut stderr = mount.walk(&[&id, "fd", "2"], &subject()).await.unwrap();
        mount.open(&mut stderr, 0, &subject()).await.unwrap();
        assert_eq!(
            mount.read(&stderr, 0, 4096, &subject()).await.unwrap(),
            b"fake stderr\n"
        );
    }

    #[tokio::test]
    async fn fd_zero_is_omitted_until_the_task_runtime_consumes_stdin() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);
        assert!(matches!(
            mount.walk(&[&id, "fd", "0"], &subject()).await,
            Err(MountError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn ctl_exec_rejects_a_different_subject() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);
        let other = Subject::new("other-user");
        let mut ctl = mount.walk(&[&id, "ctl"], &other).await.unwrap();
        mount.open(&mut ctl, 2, &other).await.unwrap();
        mount.write(&ctl, 0, b"exec build", &other).await.unwrap();
        let reply = mount.read(&ctl, 0, 4096, &other).await.unwrap();
        assert!(
            String::from_utf8(reply)
                .unwrap()
                .contains("caller does not own this task's Subject-scoped namespace")
        );
    }

    #[tokio::test]
    async fn attachment_bound_task_uses_exec_projection_and_revocation_gates_effects() {
        let fake = Arc::new(FakeBackend::default());
        let backend: Arc<dyn SandboxBackend> = fake.clone();
        let pool = Arc::new(SandboxPool::new(
            PoolConfig {
                max_sandboxes: 2,
                warm_pool_size: 0,
                ..Default::default()
            },
            backend,
        ));
        let mount = ExecMount::new(Arc::clone(&pool));
        let owner = Subject::new("alice");
        let attachment = VerifiedAttachment::for_test_local_root(owner.clone()).unwrap();
        let grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSpawn,
            AttachmentOperation::TaskAttach,
            AttachmentOperation::TaskSignal,
            AttachmentOperation::TaskRead,
            AttachmentOperation::TaskPublish,
        ]);
        let vfs_context = VfsOpContext::from_attachment_grant(&grant);
        let binding = TaskAttachmentBinding::from_grant(&grant);
        let request = TaskSpawnRequest::new(
            binding.clone(),
            NamespaceManifestDigest::from_description_bytes(b"/work=cid-a\n"),
            TaskPayload::Argv(vec!["echo".into(), "hello".into()]),
            TaskRuntimeConstraints {
                backend_class: Some("fake".into()),
                timeout_secs: Some(10),
            },
        );

        // The explicit spawn is the `/exec/clone` allocation effect. It uses
        // the existing fake-backed pool and projects exactly one task tree.
        let handle = mount.spawn_task(request).await.unwrap();
        assert!(matches!(handle.state, TaskState::Exited { code: 0 }));
        assert!(handle.reaches.vfs_path.starts_with("/exec/instances/"));
        assert!(handle.reaches.iroh_endpoint.is_none());
        assert!(handle.reaches.moq_topics.is_empty());
        // The same Subject alone cannot replay the task through the legacy
        // Subject-only Mount API; the opaque attachment context is required.
        assert!(matches!(
            mount.walk(&[handle.task.as_str(), "fd", "1"], &owner).await,
            Err(MountError::NotFound(_))
        ));
        let unbound_root = mount.walk(&[], &owner).await.unwrap();
        assert!(
            !mount
                .readdir(&unbound_root, &owner)
                .await
                .unwrap()
                .iter()
                .any(|entry| entry.name == handle.task.as_str())
        );
        // Call through `dyn Mount`, as the 9P backend does. The override must
        // retain the opaque context on the returned fid rather than falling
        // back to the Subject-only compatibility walk.
        let dynamic_mount: &dyn Mount = &mount;
        let bound_root = dynamic_mount
            .walk_with_context(&[], &vfs_context)
            .await
            .unwrap();
        assert!(
            mount
                .readdir(&bound_root, &owner)
                .await
                .unwrap()
                .iter()
                .any(|entry| entry.name == handle.task.as_str())
        );

        let stdout = dynamic_mount
            .walk_with_context(&[handle.task.as_str(), "fd", "1"], &vfs_context)
            .await
            .unwrap();
        assert_eq!(
            mount.read(&stdout, 0, 4096, &owner).await.unwrap(),
            b"fake stdout\n"
        );
        let result = mount
            .task_result(&handle.task, binding.clone())
            .await
            .unwrap();
        assert_eq!(result.content_records.len(), 2);
        assert_eq!(
            result.content_records[0].content,
            ContentDigest::from_bytes(b"fake stdout\n")
        );
        assert_eq!(result.content_records[0].task, handle.task);
        assert_eq!(result.content_records[0].role, TaskContentRole::Stdout);
        let snapshot = mount
            .snapshot_task(&handle.task, binding.clone())
            .await
            .unwrap();
        assert_eq!(
            snapshot.namespace_manifest,
            NamespaceManifestDigest::from_description_bytes(b"/work=cid-a\n")
        );

        // Revocation fences both the next ctl lifecycle effect and the next
        // stream publication. The fake backend must not see a stop request.
        attachment.revoke().unwrap();
        let signal = mount
            .signal_task(TaskSignalRequest {
                task: handle.task.clone(),
                attachment: binding,
                signal: TaskSignal::Stop,
            })
            .await;
        assert_eq!(signal, Err(TaskError::StaleAttachment));
        assert!(!fake.stopped.load(Ordering::SeqCst));
        assert!(matches!(
            mount.read(&stdout, 0, 4096, &owner).await,
            Err(MountError::PermissionDenied(_))
        ));
        assert!(matches!(
            mount
                .publish_exec_result(
                    handle.task.as_str(),
                    &owner,
                    Some(&vfs_context),
                    b"late",
                    b""
                )
                .await,
            Err(MountError::PermissionDenied(_))
        ));
    }

    #[tokio::test]
    async fn current_read_grant_cannot_borrow_spawn_grant_for_task_control() {
        let fake = Arc::new(FakeBackend::default());
        let backend: Arc<dyn SandboxBackend> = fake.clone();
        let pool = Arc::new(SandboxPool::new(
            PoolConfig {
                max_sandboxes: 2,
                warm_pool_size: 0,
                ..Default::default()
            },
            backend,
        ));
        let mount = ExecMount::new(pool);
        let owner = Subject::new("scope-owner");
        let attachment = VerifiedAttachment::for_test_local_root(owner.clone()).unwrap();
        let spawn_grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSpawn,
            AttachmentOperation::TaskPublish,
        ]);
        let read_grant = attachment.for_test_operations(&[AttachmentOperation::TaskRead]);
        let signal_grant = attachment.for_test_operations(&[AttachmentOperation::TaskSignal]);
        let signal_publish_grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSignal,
            AttachmentOperation::TaskPublish,
        ]);

        let handle = mount
            .spawn_task(TaskSpawnRequest::new(
                TaskAttachmentBinding::from_grant(&spawn_grant),
                NamespaceManifestDigest::from_description_bytes(b"/work=scope-test\n"),
                TaskPayload::Argv(vec!["echo".into(), "scope".into()]),
                TaskRuntimeConstraints {
                    backend_class: Some("fake".into()),
                    timeout_secs: None,
                },
            ))
            .await
            .unwrap();
        let read_context = VfsOpContext::from_attachment_grant(&read_grant);

        // The same attachment may read its Task, but a read-only fid cannot
        // borrow the record's original Spawn scope to drive ctl lifecycle.
        let status = mount
            .walk_with_context(&[handle.task.as_str(), "status"], &read_context)
            .await
            .unwrap();
        assert!(
            !mount
                .read(&status, 0, 4096, &owner)
                .await
                .unwrap()
                .is_empty()
        );
        let ctl = mount
            .walk_with_context(&[handle.task.as_str(), "ctl"], &read_context)
            .await
            .unwrap();
        mount.write(&ctl, 0, b"stop", &owner).await.unwrap();
        assert!(
            String::from_utf8(mount.read(&ctl, 0, 4096, &owner).await.unwrap())
                .unwrap()
                .starts_with("error:")
        );
        assert!(!fake.stopped.load(Ordering::SeqCst));

        // The direct Task API has the same composite rule: terminal fd
        // publication means TaskSignal alone is insufficient.
        assert_eq!(
            mount
                .signal_task(TaskSignalRequest {
                    task: handle.task.clone(),
                    attachment: TaskAttachmentBinding::from_grant(&signal_grant),
                    signal: TaskSignal::Stop,
                })
                .await,
            Err(TaskError::PermissionDenied)
        );
        assert!(!fake.stopped.load(Ordering::SeqCst));
        mount
            .signal_task(TaskSignalRequest {
                task: handle.task,
                attachment: TaskAttachmentBinding::from_grant(&signal_publish_grant),
                signal: TaskSignal::Stop,
            })
            .await
            .unwrap();
        assert!(fake.stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn lifecycle_policy_denial_is_not_misreported_as_stale_attachment() {
        let fake = Arc::new(FakeBackend::default());
        let backend: Arc<dyn SandboxBackend> = fake.clone();
        let pool = Arc::new(SandboxPool::new(
            PoolConfig {
                max_sandboxes: 1,
                warm_pool_size: 0,
                ..Default::default()
            },
            backend,
        ));
        let mount = ExecMount::new_with_lifecycle(pool, Arc::new(DenyAllLifecycle));
        let attachment =
            VerifiedAttachment::for_test_local_root(Subject::new("mac-denied-owner")).unwrap();
        let grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSpawn,
            AttachmentOperation::TaskSignal,
            AttachmentOperation::TaskPublish,
        ]);
        let handle = mount
            .spawn_task(TaskSpawnRequest::new(
                TaskAttachmentBinding::from_grant(&grant),
                NamespaceManifestDigest::from_description_bytes(b"/work=mac-deny\n"),
                TaskPayload::Argv(vec!["echo".into(), "mac-deny".into()]),
                TaskRuntimeConstraints {
                    backend_class: Some("fake".into()),
                    timeout_secs: None,
                },
            ))
            .await
            .unwrap();

        let result = mount
            .signal_task(TaskSignalRequest {
                task: handle.task,
                attachment: TaskAttachmentBinding::from_grant(&grant),
                signal: TaskSignal::Stop,
            })
            .await;
        assert_eq!(result, Err(TaskError::PermissionDenied));
        assert!(grant.ensure(AttachmentOperation::TaskSignal).is_ok());
        assert!(grant.ensure(AttachmentOperation::TaskPublish).is_ok());
        assert!(!fake.stopped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn content_payload_is_rejected_before_sandbox_allocation() {
        let fake = Arc::new(FakeBackend::default());
        let backend: Arc<dyn SandboxBackend> = fake;
        let pool = Arc::new(SandboxPool::new(
            PoolConfig {
                max_sandboxes: 1,
                warm_pool_size: 0,
                ..Default::default()
            },
            backend,
        ));
        let mount = ExecMount::new(Arc::clone(&pool));
        let attachment =
            VerifiedAttachment::for_test_local_root(Subject::new("content-owner")).unwrap();
        let grant = attachment.for_test_operations(&[
            AttachmentOperation::TaskSpawn,
            AttachmentOperation::TaskPublish,
        ]);
        let result = mount
            .spawn_task(TaskSpawnRequest::new(
                TaskAttachmentBinding::from_grant(&grant),
                NamespaceManifestDigest::from_description_bytes(b"/work=content\n"),
                TaskPayload::Content(ContentDigest::from_bytes(b"not materialized")),
                TaskRuntimeConstraints {
                    backend_class: Some("fake".into()),
                    timeout_secs: None,
                },
            ))
            .await;
        assert!(matches!(result, Err(TaskError::InvalidRequest(_))));
        assert!(pool.list_active().await.is_empty());
        assert!(mount.task_records.lock().is_empty());
    }

    #[tokio::test]
    async fn fd_read_rechecks_current_read_grant_after_waiting() {
        let origin = MoqStreamOrigin::standalone()
            .with_prefix("test/revocation")
            .build();
        let fd = Arc::new(TaskFdStream::new(&origin, "revocation", 1).unwrap());
        let attachment = VerifiedAttachment::for_test_local_root(Subject::new("reader")).unwrap();
        let grant = attachment.for_test_operations(&[AttachmentOperation::TaskRead]);
        let context = VfsOpContext::from_attachment_grant(&grant);
        let reader = {
            let fd = Arc::clone(&fd);
            tokio::spawn(async move { fd.read_until_closed(Some(&context), 0, 4096).await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !reader.is_finished(),
            "reader must wait before data arrives"
        );
        attachment.revoke().unwrap();
        // Compatibility publication wakes the blocked local reader; its
        // post-wake operation fence must deny rather than return these bytes.
        fd.append(None, b"late data").await.unwrap();
        assert!(matches!(
            tokio::time::timeout(std::time::Duration::from_secs(2), reader)
                .await
                .unwrap()
                .unwrap(),
            Err(MountError::PermissionDenied(_))
        ));
    }

    #[tokio::test]
    async fn failed_or_revoked_spawn_rolls_back_every_local_task_reach() {
        for revoke_during_exec in [false, true] {
            let fake = Arc::new(FakeBackend::default());
            fake.fail_exec.store(!revoke_during_exec, Ordering::SeqCst);
            let backend: Arc<dyn SandboxBackend> = fake.clone();
            let pool = Arc::new(SandboxPool::new(
                PoolConfig {
                    max_sandboxes: 2,
                    warm_pool_size: 0,
                    ..Default::default()
                },
                backend,
            ));
            let mount = ExecMount::new(Arc::clone(&pool));
            let owner = Subject::new("rollback-owner");
            let attachment = VerifiedAttachment::for_test_local_root(owner).unwrap();
            let grant = attachment.for_test_operations(&[
                AttachmentOperation::TaskSpawn,
                AttachmentOperation::TaskPublish,
            ]);
            if revoke_during_exec {
                fake.revoke_during_exec.lock().replace(attachment.clone());
            }
            let request = TaskSpawnRequest::new(
                TaskAttachmentBinding::from_grant(&grant),
                NamespaceManifestDigest::from_description_bytes(b"/work=rollback\n"),
                TaskPayload::Argv(vec!["echo".into(), "rollback".into()]),
                TaskRuntimeConstraints {
                    backend_class: Some("fake".into()),
                    timeout_secs: Some(10),
                },
            );

            let error = mount.spawn_task(request).await.unwrap_err();
            if revoke_during_exec {
                assert_eq!(error, TaskError::StaleAttachment);
            } else {
                assert!(matches!(error, TaskError::Backend(_)));
            }
            assert!(pool.list_active().await.is_empty());
            assert!(mount.task_records.lock().is_empty());
            assert!(mount.fd_streams.lock().await.is_empty());
            assert!(fake.stopped.load(Ordering::SeqCst));
        }
    }

    #[tokio::test]
    async fn stopping_instance_latches_waiting_fd_reader_to_eof() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = Arc::new(ExecMount::new(pool));
        let reader = {
            let mount = Arc::clone(&mount);
            let id = id.clone();
            tokio::spawn(async move {
                let mut fid = mount.walk(&[&id, "fd", "2"], &subject()).await.unwrap();
                mount.open(&mut fid, 0, &subject()).await.unwrap();
                mount.read(&fid, 0, 4096, &subject()).await.unwrap()
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(
            !reader.is_finished(),
            "fd reader must wait before cancellation"
        );

        let mut ctl = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut ctl, 2, &subject()).await.unwrap();
        mount.write(&ctl, 0, b"stop", &subject()).await.unwrap();

        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(2), reader)
                .await
                .expect("cancellation did not wake fd reader")
                .unwrap()
                .is_empty(),
            "cancelled stream must latch terminal EOF"
        );
    }

    #[tokio::test]
    async fn stat_root_is_dir() {
        let pool = make_pool().await;
        let mount = ExecMount::new(pool);
        let fid = mount.walk(&[], &subject()).await.unwrap();
        let st = mount.stat(&fid, &subject()).await.unwrap();
        assert_eq!(st.qtype, 0x80);
        assert_eq!(st.name, "instances");
    }

    /// Regression: the ctl write→latch must be safe under concurrent access
    /// to the same fid. The `Mount::write`/`read` signatures take `&Fid`
    /// (shared), so the latch must use interior mutability. Before this fix
    /// the mount used an `unsafe *mut` cast through `&` to mutate `write_buf`,
    /// which was a data race / UB if two futures holding `&Fid` were polled
    /// concurrently (e.g. a 9P server multiplexing Twrite on one fid). Here a
    /// single ctl fid is shared across `join3`-concurrent writers: under the
    /// old `unsafe` cast Miri flags this as UB; under the `parking_lot::Mutex`
    /// latch it is sound, and exactly one writer's response wins the latch.
    #[tokio::test]
    async fn ctl_concurrent_writes_to_one_fid_are_safe() {
        let pool = make_pool().await;
        let id = pool
            .acquire(&admission_subject(), &PodSandboxConfig::default())
            .await
            .unwrap();
        let mount = ExecMount::new(pool);

        // One shared ctl fid; opened RDWR.
        let mut fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();

        // Drive several concurrent writers against the *same* `&Fid`. The
        // contract under test is *no UB / no panic* — the final latched value
        // just has to be a valid ctl response (the latch update must be
        // atomic, not a torn write).
        let caller = subject();
        let w1 = mount.write(&fid, 0, b"stop", &caller);
        let w2 = mount.write(&fid, 0, b"stop", &caller);
        let w3 = mount.write(&fid, 0, b"kill", &caller);
        let (r1, r2, r3) = futures::future::join3(w1, w2, w3).await;
        for r in [r1, r2, r3] {
            r.unwrap();
        }

        let out = mount.read(&fid, 0, 4096, &caller).await.unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("ok:"),
            "expected one of the stop/kill responses latched, got: {text}"
        );
    }
}
