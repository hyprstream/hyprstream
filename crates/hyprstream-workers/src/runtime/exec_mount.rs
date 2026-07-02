//! `/exec/instances/` VFS mount — projects `SandboxPool`'s active sandboxes
//! as a Plan9 `/proc`-style tree.
//!
//! This is the P2 slice of epic #608 ("P9 Task/Instance Projection"); see
//! `docs/plans/2026-06-30-execution-control-plane-9p-composition.md` §4 for the
//! full `/exec` tree this is one branch of (`/exec/instances/<id>/...`).
//!
//! Layout:
//! - `/exec/instances/`            — dynamic dir: active sandbox/instance ids
//! - `/exec/instances/<id>/ctl`    — ctl file: write a verb to drive the
//!   instance's lifecycle (see [`Verb`] for the grammar)
//! - `/exec/instances/<id>/status` — read-only: current `PodSandboxState`,
//!   live/non-blocking poll of pool state
//! - `/exec/instances/<id>/exit`   — read-only: terminal status if already
//!   terminal, otherwise an empty "not yet terminal" marker. **STUB** — see
//!   the `TODO(#607)` below for the real blocking-await wiring.
//! - `/exec/instances/<id>/ns`     — read-only: best-effort textual listing
//!   of the sandbox's mount-prefixes/namespace, or an empty placeholder
//!
//! `fd/` (per-instance stdio/stream fds) is deliberately NOT implemented here:
//! it depends on the streaming plane / #170 (PUSH/PULL→XPUB response
//! streaming) landing first. Wiring it is a follow-up once that plane exists.
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
//!
//! Unknown verbs return [`MountError::InvalidArgument`]. A verb may be
//! followed by trailing whitespace, which is trimmed.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex as AsyncMutex;

use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};
// `parking_lot::Mutex` for the ctl write→read latch (interior mutability
// through the `&Fid` that `Mount::write` receives). When this branch is
// rebased onto a base containing #615, swap this hand-rolled latch for
// `hyprstream_vfs::devfile::DevFileState` (same parking_lot::Mutex shape).
use parking_lot::Mutex as PmMutex;

use super::client::PodSandboxState;
use super::pool::SandboxPool;

// ─────────────────────────────────────────────────────────────────────────────
// Ctl verb grammar
// ─────────────────────────────────────────────────────────────────────────────

/// Verbs accepted by `/exec/instances/<id>/ctl`. See module docs for the
/// full grammar table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verb {
    Start,
    Stop,
    Kill,
    Destroy,
}

impl Verb {
    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "start" => Some(Self::Start),
            "stop" => Some(Self::Stop),
            "kill" => Some(Self::Kill),
            "destroy" => Some(Self::Destroy),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Terminal tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Terminal status recorded for an instance once it has exited (i.e. been
/// destroyed/stopped-for-good). Populated by the `ctl` `destroy` handler.
///
/// This is intentionally tiny: it exists only so `exit` has *something*
/// non-empty to return once an instance leaves the pool's active map. The
/// real completion primitive (terminal-latch + read-then-subscribe) is #607
/// (EventService EV7), not yet implemented anywhere in this repo.
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
}

/// Fid state for the exec mount.
struct ExecFid {
    kind: ExecFidKind,
    /// Latch for the ctl write→read pattern (verb result message). A
    /// `parking_lot::Mutex<Vec<u8>>` provides safe interior mutability through
    /// the `&Fid` that `Mount::read`/`Mount::write` receive — so we avoid the
    /// `unsafe *mut` cast-through-`&` hazard. Unused for non-ctl fid kinds.
    write_buf: PmMutex<Vec<u8>>,
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
    /// Terminal status recorded for instances that have been `destroy`ed
    /// through this mount. Keyed by instance id.
    ///
    /// TODO(#607): this hand-rolled terminal map is the stub. Once EV7
    /// (terminal-latch + read-then-subscribe) lands in EventService, `exit`
    /// should subscribe to/await the instance's terminal event instead of
    /// polling this map, and a blocking `read()` on `exit` should suspend
    /// until the latch fires rather than returning immediately. The map
    /// itself (or something shaped like it) likely becomes the read-side of
    /// that latch rather than being deleted outright.
    terminal: AsyncMutex<HashMap<String, TerminalStatus>>,
}

impl ExecMount {
    /// Create a new `ExecMount` over the given pool.
    pub fn new(pool: Arc<SandboxPool>) -> Self {
        Self {
            pool,
            terminal: AsyncMutex::new(HashMap::new()),
        }
    }

    /// Resolve the verb against the pool/backend for instance `id`.
    async fn apply_verb(&self, id: &str, verb: Verb) -> Result<String, MountError> {
        // Instances already marked terminal (destroyed through this mount)
        // are not present in the pool's active map any more; ctl ops on them
        // are rejected rather than silently no-op'd.
        if self.terminal.lock().await.contains_key(id) {
            return Err(MountError::InvalidArgument(format!(
                "instance {id} is already terminal"
            )));
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
                self.terminal.lock().await.insert(
                    id.to_owned(),
                    TerminalStatus {
                        last_state: sandbox.state,
                    },
                );
                Ok("ok: destroyed\n".to_owned())
            }
        }
    }

    /// Live, non-blocking poll of an instance's current state for `status`.
    async fn read_status(&self, id: &str) -> Result<String, MountError> {
        if let Some(term) = self.terminal.lock().await.get(id) {
            return Ok(format!("{:?}\n", term.last_state));
        }
        let sandbox = self
            .pool
            .get(id)
            .await
            .ok_or_else(|| MountError::NotFound(format!("instances/{id}")))?;
        Ok(format!("{:?}\n", sandbox.state))
    }

    /// `exit` stub: returns the terminal status if already terminal, or an
    /// empty "not yet terminal" marker otherwise.
    ///
    /// TODO(#607): replace this synchronous poll with a real blocking-await
    /// on the instance's terminal latch (EventService EV7: terminal-latch +
    /// read-then-subscribe completion primitive). The semantics this issue
    /// (#610) intentionally does NOT implement: a `read()` on `exit` should
    /// block until the instance transitions to a terminal state (or the fid
    /// is clunked/cancelled), delivering the terminal status exactly once to
    /// each waiter that arrives before the latch fires, and immediately to
    /// any waiter that arrives after (read-then-subscribe — no missed
    /// wakeups). Until #607 lands, callers must poll this file themselves.
    async fn read_exit(&self, id: &str) -> Result<String, MountError> {
        if let Some(term) = self.terminal.lock().await.get(id) {
            return Ok(term.render());
        }
        // Confirm the instance actually exists (vs. a bogus id) before
        // reporting "not yet terminal".
        if self.pool.get(id).await.is_none() {
            return Err(MountError::NotFound(format!("instances/{id}")));
        }
        // Plan9-shaped "not done": empty read.
        Ok(String::new())
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

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Mount for ExecMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // NOTE on `Subject`: per the Mount contract this mount threads the
        // caller through every op but does not itself authorize — the
        // MAC reference-monitor (#547) is the eventual enforcement point.
        // `SandboxPool`/`SandboxBackend` don't take a `Subject` today, so
        // there is nothing further to pass it to here.
        let kind = match components {
            [] => ExecFidKind::InstancesDir,
            [id] => ExecFidKind::InstanceDir((*id).to_owned()),
            [id, "ctl"] => ExecFidKind::Ctl((*id).to_owned()),
            [id, "status"] => ExecFidKind::Status((*id).to_owned()),
            [id, "exit"] => ExecFidKind::Exit((*id).to_owned()),
            [id, "ns"] => ExecFidKind::Ns((*id).to_owned()),
            _ => return Err(MountError::NotFound(components.join("/"))),
        };

        // Validate that referenced instances exist (directory walk into an
        // unknown id should 404, same as any 9P namespace).
        let id_to_check: Option<&str> = match &kind {
            ExecFidKind::InstanceDir(id)
            | ExecFidKind::Ctl(id)
            | ExecFidKind::Status(id)
            | ExecFidKind::Exit(id)
            | ExecFidKind::Ns(id) => Some(id.as_str()),
            ExecFidKind::InstancesDir => None,
        };
        if let Some(id) = id_to_check {
            let exists =
                self.pool.get(id).await.is_some() || self.terminal.lock().await.contains_key(id);
            if !exists {
                return Err(MountError::NotFound(components.join("/")));
            }
        }

        Ok(Fid::new(ExecFid {
            kind,
            write_buf: PmMutex::new(Vec::new()),
        }))
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        _count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let data = match &inner.kind {
            ExecFidKind::Ctl(_) => {
                // Ctl pattern: after a write, read returns the verb result.
                inner.write_buf.lock().clone()
            }
            ExecFidKind::Status(id) => self.read_status(id).await?.into_bytes(),
            ExecFidKind::Exit(id) => self.read_exit(id).await?.into_bytes(),
            ExecFidKind::Ns(id) => self.read_ns(id).await?.into_bytes(),
            ExecFidKind::InstancesDir | ExecFidKind::InstanceDir(_) => {
                return Err(MountError::IsDirectory("use readdir".into()));
            }
        };

        let start = offset as usize;
        if start >= data.len() {
            return Ok(Vec::new());
        }
        Ok(data[start..].to_vec())
    }

    async fn write(
        &self,
        fid: &Fid,
        _offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match &inner.kind {
            ExecFidKind::Ctl(id) => {
                let text = String::from_utf8_lossy(data);
                let verb = Verb::parse(&text).ok_or_else(|| {
                    MountError::InvalidArgument(format!("unknown ctl verb: {}", text.trim()))
                })?;
                let result = self.apply_verb(id, verb).await;
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
                *inner.write_buf.lock() = response_bytes;
                Ok(data.len() as u32)
            }
            _ => Err(MountError::NotSupported("read-only".into())),
        }
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match &inner.kind {
            ExecFidKind::InstancesDir => {
                let mut ids: Vec<String> = self
                    .pool
                    .list_active()
                    .await
                    .into_iter()
                    .map(|s| s.id)
                    .collect();
                // Also surface ids that have been destroyed-through-this-mount
                // (so `exit`/`status` remain reachable for stragglers that
                // already read "not yet terminal" and need to poll again).
                for id in self.terminal.lock().await.keys() {
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
            ]),
            _ => Err(MountError::NotDirectory(format!("{:?}", inner.kind))),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<ExecFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        const QTDIR: u8 = 0x80;
        let (name, qtype) = match &inner.kind {
            ExecFidKind::InstancesDir => ("instances".to_owned(), QTDIR),
            ExecFidKind::InstanceDir(id) => (id.clone(), QTDIR),
            ExecFidKind::Ctl(_) => ("ctl".to_owned(), 0),
            ExecFidKind::Status(_) => ("status".to_owned(), 0),
            ExecFidKind::Exit(_) => ("exit".to_owned(), 0),
            ExecFidKind::Ns(_) => ("ns".to_owned(), 0),
        };

        Ok(Stat::unknown_qid(qtype, 0, name, 0))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::PoolConfig;
    use crate::error::Result as WorkerResult;
    use crate::runtime::backend::{SandboxBackend, SandboxHandle};
    use crate::runtime::client::{LinuxContainerResources, PodSandboxConfig};
    use crate::runtime::sandbox::PodSandbox;
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
            false
        }

        async fn exec_sync(
            &self,
            _sandbox: &PodSandbox,
            _command: &[String],
            _timeout_secs: u64,
        ) -> WorkerResult<(i32, Vec<u8>, Vec<u8>)> {
            Err(crate::error::WorkerError::ExecFailed(
                "not supported".into(),
            ))
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
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
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
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
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
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        let written = mount.write(&fid, 0, b"stop", &subject()).await.unwrap();
        assert_eq!(written, 4);

        let data = mount.read(&fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.starts_with("ok:"), "got: {text}");
    }

    #[tokio::test]
    async fn ctl_unknown_verb_rejected() {
        let pool = make_pool().await;
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "ctl"], &subject()).await.unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        let result = mount.write(&fid, 0, b"frobnicate", &subject()).await;
        assert!(matches!(result, Err(MountError::InvalidArgument(_))));
    }

    #[tokio::test]
    async fn exit_before_terminal_is_empty() {
        let pool = make_pool().await;
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
        let mount = ExecMount::new(pool);

        let mut fid = mount.walk(&[&id, "exit"], &subject()).await.unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject()).await.unwrap();
        assert!(data.is_empty(), "expected empty 'not yet terminal' marker");
    }

    #[tokio::test]
    async fn exit_after_destroy_reports_terminal() {
        let pool = make_pool().await;
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
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

        // exit should now report terminal status, non-blocking.
        let mut exit_fid = mount.walk(&[&id, "exit"], &subject()).await.unwrap();
        mount.open(&mut exit_fid, 0, &subject()).await.unwrap();
        let data = mount.read(&exit_fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.starts_with("exited"), "got: {text}");

        // status should also reflect terminal state via the recorded map.
        let mut status_fid = mount.walk(&[&id, "status"], &subject()).await.unwrap();
        mount.open(&mut status_fid, 0, &subject()).await.unwrap();
        let data = mount.read(&status_fid, 0, 4096, &subject()).await.unwrap();
        assert!(!data.is_empty());
    }

    #[tokio::test]
    async fn ctl_on_terminal_instance_rejected() {
        let pool = make_pool().await;
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
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
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
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
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
        let mount = ExecMount::new(pool);

        let fid = mount.walk(&[&id], &subject()).await.unwrap();
        let entries = mount.readdir(&fid, &subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"ctl"));
        assert!(names.contains(&"status"));
        assert!(names.contains(&"exit"));
        assert!(names.contains(&"ns"));
        // fd/ is deliberately not present — deferred to the streaming plane / #170.
        assert!(!names.contains(&"fd"));
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
        let id = pool.acquire(&PodSandboxConfig::default()).await.unwrap();
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
