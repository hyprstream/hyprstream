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
//! | `exec`    | runs an argv-only task through this mount's `SandboxBackend`      |
//! |           | seam, then latches stdout/stderr in `fd/1` and `fd/2`.            |
//!
//! Unknown verbs return [`MountError::InvalidArgument`]. A verb may be
//! followed by trailing whitespace, which is trimmed.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rand::RngCore;
use tokio::sync::Notify;

use hyprstream_rpc::AttachmentOperation;
use hyprstream_rpc::latch::{Terminal, TerminalStore};
use hyprstream_rpc::moq_stream::{MoqStreamOrigin, MoqStreamPublisher};
use hyprstream_rpc::stream_info::Job;
use hyprstream_rpc::streaming::StreamContext;
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
}

impl Verb {
    fn parse(s: &str) -> Option<Self> {
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

    async fn read_until_closed(&self, offset: u64, count: u32) -> Vec<u8> {
        loop {
            let notified = self.wake.notified();
            {
                let state = self.state.lock().await;
                let start = offset as usize;
                if start < state.bytes.len() {
                    let end = start.saturating_add(count as usize).min(state.bytes.len());
                    return state.bytes[start..end].to_vec();
                }
                if state.closed {
                    return Vec::new();
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
    context: VfsOpContext,
    namespace_manifest: hyprstream_rpc_std::task::NamespaceManifestDigest,
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
            // Unsubscribed shadow state today: nothing reads from this origin
            // (the 9P fd reads below serve the local `TaskFdStream` buffer
            // instead). Deferred, not wired, pending epic #608's stream-plane
            // mount — see Finding 3 of REVIEW-connie-1447-r2-k3-2026-07-30.md.
            fd_origin: MoqStreamOrigin::standalone()
                .with_prefix("local/exec")
                .build(),
            fd_streams: tokio::sync::Mutex::new(HashMap::new()),
            terminal: TerminalStore::new(),
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

    /// Publish one completed task execution into the instance's terminal fd
    /// streams. The runtime calls this after `SandboxBackend::exec_sync`; the
    /// Plan 9 readers observe the same bounded output and then EOF.
    async fn publish_exec_result(
        &self,
        id: &str,
        caller: &Subject,
        stdout: &[u8],
        stderr: &[u8],
    ) -> Result<(), MountError> {
        if self.pool.get(id).await.is_none() && !self.terminal.is_latched(&id.to_owned()) {
            return Err(MountError::NotFound(format!("instances/{id}")));
        }
        let context = self.task_context_for(id, caller, AttachmentOperation::TaskPublish)?;
        let streams = self.fd_streams_for(id).await?;
        streams.stdout.append(context.as_ref(), stdout).await?;
        streams.stderr.append(context.as_ref(), stderr).await?;
        streams.stdout.close(context.as_ref()).await?;
        streams.stderr.close(context.as_ref()).await?;
        Ok(())
    }

    /// Return a contract Task's context only when the caller is its verified
    /// attachment subject and the generation remains current. Compatibility
    /// instances have no record and retain their existing behavior.
    fn task_context_for(
        &self,
        id: &str,
        caller: &Subject,
        operation: AttachmentOperation,
    ) -> Result<Option<VfsOpContext>, MountError> {
        let Some(record) = self.task_records.lock().get(id).cloned() else {
            return Ok(None);
        };
        if record.context.subject() != caller {
            return Err(MountError::PermissionDenied(
                "caller does not own this task attachment".into(),
            ));
        }
        record.context.ensure_operation(operation)?;
        Ok(Some(record.context))
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
            || record.context.attachment_id() != context.attachment_id()
            || record.context.authority_generation() != context.authority_generation()
        {
            return Err(MountError::PermissionDenied(
                "attachment context does not authorize this task fid".into(),
            ));
        }
        context.ensure_operation(AttachmentOperation::TaskRead)?;
        record.context.ensure_current()?;
        Ok(())
    }

    /// Context-aware fid creation for the attachment-bound Task projection.
    /// This is intentionally separate from [`Mount::walk`]: accepting only a
    /// `Subject` there would regress to ambient same-subject authority.
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
            MountError::PermissionDenied(_) => TaskError::StaleAttachment,
            other => TaskError::Backend(other.to_string()),
        }
    }

    fn task_handle(id: &str, state: TaskState) -> TaskHandle {
        TaskHandle {
            task: TaskId::from_worker_instance(id),
            reaches: TaskReaches {
                vfs_path: format!("/exec/instances/{id}"),
                iroh_endpoint: None,
                moq_topics: vec![format!("exec-{id}-fd-1"), format!("exec-{id}-fd-2")],
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
        record
            .context
            .ensure_current()
            .map_err(|_| TaskError::StaleAttachment)?;
        if record.context.attachment_id() != attachment.attachment_id()
            || record.context.authority_generation() != attachment.authority_generation()
            || record.context.subject() != attachment.subject()
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
                    attachment_id: record.context.attachment_id().clone(),
                    authority_generation: record.context.authority_generation(),
                    namespace_manifest: record.namespace_manifest,
                    role: TaskContentRole::Stdout,
                },
                TaskContentRecord {
                    content: ContentDigest::from_bytes(&result.stderr),
                    task,
                    attachment_id: record.context.attachment_id().clone(),
                    authority_generation: record.context.authority_generation(),
                    namespace_manifest: record.namespace_manifest,
                    role: TaskContentRole::Stderr,
                },
            ];
        }
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
        // First lifecycle-effect gate: a contract Task must still hold its
        // verified attachment generation before a backend command can start.
        self.task_context_for(id, caller, AttachmentOperation::TaskSpawn)?;
        // `exec` necessarily emits terminal fd data, so refuse before the
        // backend starts unless the same grant also permits publication. A
        // spawn-only identity must not leave an unpublishable task result.
        self.task_context_for(id, caller, AttachmentOperation::TaskPublish)?;
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
        let (exit_code, stdout, stderr) = self
            .pool
            .backend()
            .exec_sync(&sandbox, command, timeout_secs)
            .await
            .map_err(|e| MountError::Io(e.to_string()))?;
        // Second gate: the backend may have run for a while. Do not publish a
        // now-revoked Task's next MoQ/VFS output continuation.
        self.publish_exec_result(id, caller, &stdout, &stderr)
            .await?;
        Ok(TaskExecResult {
            exit_code,
            stdout,
            stderr,
        })
    }

    async fn close_fd_streams(&self, id: &str, caller: &Subject) -> Result<(), MountError> {
        let context = self.task_context_for(id, caller, AttachmentOperation::TaskSignal)?;
        let streams = self.fd_streams_for(id).await?;
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
            Verb::Exec(_) => AttachmentOperation::TaskSpawn,
            Verb::Start | Verb::Stop | Verb::Kill | Verb::Destroy => {
                AttachmentOperation::TaskSignal
            }
        };
        self.task_context_for(id, caller, task_operation)?;
        if matches!(verb, Verb::Stop | Verb::Kill | Verb::Destroy) {
            // Stopping/destroying also completes the fd carrier. Require that
            // terminal publication before altering the backend lifecycle.
            self.task_context_for(id, caller, AttachmentOperation::TaskPublish)?;
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
                    Verb::Exec(_) => {
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
                self.close_fd_streams(id, caller).await?;
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
                self.close_fd_streams(id, caller).await?;
                // Wake any `exit` reader already blocked on this id.
                self.waiter_for(id).notify_waiters();
                Ok("ok: destroyed\n".to_owned())
            }
            Verb::Exec(command) => {
                let result = self.exec_task(id, caller, &command, 300).await?;
                self.record_exec_result(id, &result);
                Ok(format!("ok: exited {}\n", result.exit_code))
            }
        }
    }

    /// Live, non-blocking poll of an instance's current state for `status`.
    async fn read_status(&self, id: &str) -> Result<String, MountError> {
        if let Some(term) = self.terminal.get(&id.to_owned()) {
            return Ok(format!("{:?}\n", term.value.last_state));
        }
        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;
        Ok(format!("{:?}\n", sandbox.state))
    }

    /// `exit`: blocks until the instance is terminal, then returns the
    /// retained status. A read that starts after the instance already
    /// latched terminal (a "late" reader, in read-then-subscribe terms)
    /// returns immediately with the retained value — no missed completion.
    /// A read that starts before (an "early" reader) blocks until `apply_verb`
    /// latches the `Destroy` outcome and wakes this id's waiter.
    async fn read_exit(&self, id: &str) -> Result<String, MountError> {
        let key = id.to_owned();
        // Fast path: already latched — serve the retained value immediately,
        // exactly the "late reader" half of read-then-subscribe.
        if let Some(term) = self.terminal.get(&key) {
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
            if let Some(term) = self.terminal.get(&key) {
                return Ok(term.value.render());
            }
            notified.await;
            if let Some(term) = self.terminal.get(&key) {
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
    async fn read_ns(&self, id: &str) -> Result<String, MountError> {
        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;

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
        let TaskSpawnRequest {
            parent_task: _,
            attachment,
            namespace_manifest,
            payload,
            constraints,
        } = request;

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
        context
            .ensure_operation(AttachmentOperation::TaskSpawn)
            .map_err(|_| TaskError::StaleAttachment)?;
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
        if context.ensure_current().is_err() {
            let _ = self.pool.release(&id).await;
            return Err(TaskError::StaleAttachment);
        }

        self.task_records.lock().insert(
            id.clone(),
            TaskRecord {
                context: context.clone(),
                namespace_manifest,
                state: TaskState::Running,
                content_records: Vec::new(),
            },
        );

        match payload {
            TaskPayload::Argv(command) => {
                let timeout_secs = constraints.timeout_secs.unwrap_or(300);
                let result = match self
                    .exec_task(&id, context.subject(), &command, timeout_secs)
                    .await
                {
                    Ok(result) => result,
                    Err(error) => {
                        self.rollback_task_allocation(&id).await;
                        return Err(Self::task_error(error));
                    }
                };
                self.record_exec_result(&id, &result);
            }
            TaskPayload::Content(content) => {
                // This is an unmaterialized content digest, not an artifact
                // fetch. Record only its task association; placement,
                // execution, and durable materialization remain Worker work.
                if let Some(record) = self.task_records.lock().get_mut(&id) {
                    record.content_records.push(TaskContentRecord {
                        content,
                        task: TaskId::from_worker_instance(&id),
                        attachment_id: record.context.attachment_id().clone(),
                        authority_generation: record.context.authority_generation(),
                        namespace_manifest: record.namespace_manifest,
                        role: TaskContentRole::Input,
                    });
                }
            }
        }

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
        self.apply_verb(request.task.as_str(), verb, request.attachment.subject())
            .await
            .map_err(Self::task_error)?;
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
            ExecFidKind::Status(id) => self.read_status(id).await?.into_bytes(),
            ExecFidKind::Exit(id) => self.read_exit(id).await?.into_bytes(),
            ExecFidKind::Ns(id) => self.read_ns(id).await?.into_bytes(),
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
                    .read_until_closed(offset, count)
                    .await);
            }
            ExecFidKind::Fd(id, 2) => {
                return Ok(self
                    .fd_streams_for(id)
                    .await?
                    .stderr
                    .read_until_closed(offset, count)
                    .await);
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
                let result = self.apply_verb(id, verb, caller).await;
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

        if !matches!(&inner.kind, ExecFidKind::InstancesDir) {
            self.verify_fid_context(&inner.kind, inner.context.as_ref(), caller)?;
        }

        match &inner.kind {
            ExecFidKind::InstancesDir => {
                let records = self.task_records.lock().clone();
                let mut ids: Vec<String> = self
                    .pool
                    .list_active()
                    .await
                    .into_iter()
                    .filter(|sandbox| {
                        records.get(&sandbox.id).is_none_or(|record| {
                            inner.context.as_ref().is_some_and(|context| {
                                context.subject() == caller
                                    && context.attachment_id() == record.context.attachment_id()
                                    && context.authority_generation()
                                        == record.context.authority_generation()
                                    && context.ensure_current().is_ok()
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
                                && context.attachment_id() == record.context.attachment_id()
                                && context.authority_generation()
                                    == record.context.authority_generation()
                                && context.ensure_current().is_ok()
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
    use hyprstream_rpc::VerifiedAttachment;
    use hyprstream_rpc::moq_stream::{STREAM_TRACK, verify_moq_frame};
    use hyprstream_rpc::streaming::{StreamPayload, StreamVerifier};
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
            _command: &[String],
            _timeout_secs: u64,
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

    async fn make_pool() -> Arc<SandboxPool> {
        let backend: Arc<dyn SandboxBackend> = Arc::new(FakeBackend::default());
        let config = PoolConfig {
            max_sandboxes: 10,
            warm_pool_size: 0,
            ..Default::default()
        };
        Arc::new(SandboxPool::new(config, backend))
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
            .publish_exec_result(&id, &subject(), b"build output\n", b"warning output\n")
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
            .publish_exec_result(&id, &subject(), b"0123456789", b"")
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
        assert_eq!(handle.reaches.moq_topics.len(), 2);
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
        let bound_root = mount.walk_with_context(&[], &vfs_context).await.unwrap();
        assert!(
            mount
                .readdir(&bound_root, &owner)
                .await
                .unwrap()
                .iter()
                .any(|entry| entry.name == handle.task.as_str())
        );

        let stdout = mount
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
                .publish_exec_result(handle.task.as_str(), &owner, b"late", b"")
                .await,
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
