//! Dedicated-thread async drivers for `!Send` language-shell interpreters (and,
//! via [`NamespaceDriver`], for direct `Namespace` access that doesn't need an
//! interpreter at all).
//!
//! The synchronous slash-command path (`ChatApp::handle_vfs_command`) needs to run
//! async VFS futures to completion from a key-event handler that must not spin. A
//! `Poll::Pending` on a VFS read that legitimately blocks on an external event
//! (e.g. an `/exec/instances/<id>/exit` read awaiting a worker-completion signal)
//! must park the caller, not hot-spin a core — which a noop-waker poll loop cannot
//! do, because a noop waker never delivers the wake-up.
//!
//! ## The driver
//!
//! [`ShellOwner`] spawns one dedicated OS thread per shell. That thread runs a
//! single-threaded `tokio::runtime::Runtime` (`Builder::new_current_thread()`)
//! driving a `tokio::task::LocalSet`, the same pattern used by
//! `hyprstream-workers::workflow::runner`'s per-job `TclShell` owner
//! (`crates/hyprstream-workers/src/workflow/runner.rs`, `run_job`). Because the
//! shell never leaves that thread, a `Pending` future registers a *real* waker
//! tied to the runtime's reactor/timer/notify machinery — the executor parks the
//! thread (zero CPU) until something actually wakes it, instead of spinning.
//!
//! `TclShell`'s `!Send` constraint (it wraps a molt `Interp`, which is `Rc`-based)
//! is load-bearing here: the shell is constructed on, and never leaves, its owner
//! thread — the whole point of a single-threaded `LocalSet` owner.
//!
//! ## The blocking `eval()` leg
//!
//! [`ShellOwner::eval`] round-trips a request to the owner thread and waits for the
//! reply over a plain `std::sync::mpsc` channel — deliberately NOT
//! `tokio::sync::oneshot::Receiver::blocking_recv()`, because `handle_input` runs
//! in two different contexts depending on how the TUI is hosted:
//!
//!   - a bare OS thread with no tokio runtime at all (`spawn_app_process` /
//!     `run_app_loop`, the TuiService-hosted path), and
//!   - directly inside an active tokio runtime (the CLI shell's event loop in
//!     `crates/hyprstream/src/cli/shell_handlers.rs`, which calls
//!     `app.handle_input()` from within a `tokio::select!` arm).
//!
//! `tokio::sync::oneshot::Receiver::blocking_recv()` panics ("Cannot block the
//! current thread from within a runtime") in the second context. `std::sync::mpsc`
//! has no opinion about tokio and parks the calling OS thread via a condvar in
//! both contexts, so it is safe everywhere `handle_vfs_command` is called from.
//! This blocks the calling (input-handling) thread for the duration of one `eval`
//! — a parked wait, not a spinning one. Making `handle_vfs_command` fully
//! non-blocking would mean enqueue-and-poll against `ShellOwner` rather than
//! waiting inline.

use std::sync::mpsc as std_mpsc;

// ─────────────────────────────────────────────────────────────────────────────
// TclShell owner
// ─────────────────────────────────────────────────────────────────────────────

/// One ad-hoc `eval()` request sent to a [`ShellOwner`]'s dedicated thread.
///
/// The job runs `shell.eval(&script)` and sends the result back over `reply` —
/// encapsulating the reply send keeps the channel a plain `FnOnce(&mut TclShell)`
/// regardless of the job's result type.
struct TclEvalJob {
    script: String,
    reply: std_mpsc::Sender<Result<String, String>>,
}

/// Dedicated-thread owner for a `TclShell`.
///
/// Spawns one OS thread that builds a current-thread tokio runtime + `LocalSet`,
/// constructs the `TclShell` on that thread (preserving its `!Send` requirement),
/// and services ad-hoc `eval()` calls (from slash-command routing) via
/// [`Self::eval`], which blocks the *calling* thread (parked, not spinning) until
/// the owner thread replies.
///
/// Dropping the `ShellOwner` drops `job_tx`, which ends the owner thread's job
/// loop.
pub struct ShellOwner {
    job_tx: std_mpsc::Sender<TclEvalJob>,
    _handle: std::thread::JoinHandle<()>,
}

impl ShellOwner {
    /// Spawn the owner thread, constructing a `TclShell` from `subject`/`namespace`.
    pub fn spawn(
        subject: hyprstream_vfs::Subject,
        namespace: std::sync::Arc<hyprstream_vfs::Namespace>,
    ) -> Self {
        let (job_tx, job_rx) = std_mpsc::channel::<TclEvalJob>();

        #[allow(clippy::expect_used)] // thread spawn failure is unrecoverable here
        let handle = std::thread::Builder::new()
            .name("tcl-shell-owner".into())
            .spawn(move || {
                // Single-threaded runtime + LocalSet: the same pattern used by
                // hyprstream-workers::workflow::runner's per-job TclShell owner
                // (crates/hyprstream-workers/src/workflow/runner.rs, run_job).
                // A Pending future here registers a real waker tied to this
                // runtime's reactor — `.await` parks the thread; it does not spin.
                #[allow(clippy::expect_used)] // runtime creation failure is unrecoverable
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tcl-shell-owner runtime");
                let local = tokio::task::LocalSet::new();

                local.block_on(&rt, async move {
                    let mut shell = hyprstream_workers_tcl::TclShell::new(subject, namespace);

                    // Bridge the plain std::sync::mpsc job queue into this
                    // LocalSet without ever blocking the runtime thread itself:
                    // a tiny forwarder thread does the (parked, not spun)
                    // std::sync::mpsc::Receiver::recv(), then hands jobs off
                    // through a tokio channel this task can `.await` on.
                    let (bridge_tx, mut bridge_rx) =
                        tokio::sync::mpsc::unbounded_channel::<TclEvalJob>();
                    std::thread::spawn(move || {
                        while let Ok(job) = job_rx.recv() {
                            if bridge_tx.send(job).is_err() {
                                break;
                            }
                        }
                    });

                    // job_tx (held by ShellOwner) dropped — owner is torn down.
                    while let Some(TclEvalJob { script, reply }) = bridge_rx.recv().await {
                        let result = shell.eval(&script).await;
                        let _ = reply.send(result);
                    }
                });
            })
            .expect("spawn tcl-shell-owner thread");

        Self { job_tx, _handle: handle }
    }

    /// Evaluate a Tcl script on the owner thread and block (parked, not spun)
    /// until the result is ready.
    ///
    /// Safe to call from any thread context, including from inside an active
    /// tokio runtime — uses `std::sync::mpsc`, not `tokio::sync::oneshot`, for
    /// the reply leg specifically so it never hits tokio's "cannot block the
    /// current thread from within a runtime" panic (see module docs).
    pub fn eval(&self, script: &str) -> Result<String, String> {
        let (reply_tx, reply_rx) = std_mpsc::channel();
        let sent = self.job_tx.send(TclEvalJob {
            script: script.to_owned(),
            reply: reply_tx,
        });
        if sent.is_err() {
            return Err("tcl shell owner thread is gone".to_owned());
        }
        reply_rx
            .recv()
            .unwrap_or_else(|_| Err("tcl shell owner thread is gone".to_owned()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Direct Namespace driver (Tcl-independent VFS access for bare cat/ls routing)
// ─────────────────────────────────────────────────────────────────────────────

/// One direct `Namespace` request — `cat` or `ls` — dispatched without going
/// through a `TclShell` at all.
enum NsOp {
    Cat(String),
    Ls(String),
}

struct NsJob {
    op: NsOp,
    reply: std_mpsc::Sender<Result<String, String>>,
}

/// Dedicated-thread driver for direct `Namespace` access (`cat`/`ls`), with no
/// Tcl interpreter involved at all.
///
/// This is what lets `ChatApp::handle_vfs_command` decouple simple path reads
/// from `TclShell`: a bare path like `/srv/model/status` no longer needs an
/// interpreter to exist, let alone to `eval("cat ...")` through one. Unlike
/// [`ShellOwner`], `Namespace::cat`/`Namespace::ls` futures are plain `Send`
/// (no `Rc`, no `!Send` constraint) — so this driver doesn't need a `LocalSet`,
/// just a parked single-threaded runtime to give synchronous callers (which may
/// or may not already be inside a tokio runtime — see module docs) a non-spinning
/// way to wait on an async VFS call.
pub struct NamespaceDriver {
    job_tx: std_mpsc::Sender<NsJob>,
    _handle: std::thread::JoinHandle<()>,
}

impl NamespaceDriver {
    /// Spawn the driver thread bound to `namespace`/`subject`.
    pub fn spawn(
        subject: hyprstream_vfs::Subject,
        namespace: std::sync::Arc<hyprstream_vfs::Namespace>,
    ) -> Self {
        let (job_tx, job_rx) = std_mpsc::channel::<NsJob>();

        #[allow(clippy::expect_used)] // thread spawn failure is unrecoverable here
        let handle = std::thread::Builder::new()
            .name("vfs-namespace-driver".into())
            .spawn(move || {
                #[allow(clippy::expect_used)] // runtime creation failure is unrecoverable
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("vfs-namespace-driver runtime");

                // job_rx.recv() blocks this dedicated OS thread (parked via
                // condvar, not spun) between requests. Each request is then
                // driven to completion with its own `rt.block_on(..)` — this
                // thread has no other work, so there is nothing to interleave
                // with and no benefit to keeping a task alive across requests.
                while let Ok(NsJob { op, reply }) = job_rx.recv() {
                    let result = rt.block_on(async {
                        match op {
                            NsOp::Cat(path) => namespace
                                .cat(&path, &subject)
                                .await
                                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                                .map_err(|e| e.to_string()),
                            NsOp::Ls(path) => namespace
                                .ls(&path, &subject)
                                .await
                                .map(|entries| {
                                    entries
                                        .iter()
                                        .map(|e| {
                                            if e.is_dir {
                                                format!("{}/", e.name)
                                            } else {
                                                e.name.clone()
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                })
                                .map_err(|e| e.to_string()),
                        }
                    });
                    let _ = reply.send(result);
                }
            })
            .expect("spawn vfs-namespace-driver thread");

        Self { job_tx, _handle: handle }
    }

    /// Read a file's contents as a UTF-8 (lossy) string. Blocks the calling
    /// thread (parked, not spun) until the result is ready; safe to call from
    /// any thread context, including from inside an active tokio runtime.
    pub fn cat(&self, path: &str) -> Result<String, String> {
        self.request(NsOp::Cat(path.to_owned()))
    }

    /// List a directory's entries (directories suffixed with `/`), newline-joined.
    pub fn ls(&self, path: &str) -> Result<String, String> {
        self.request(NsOp::Ls(path.to_owned()))
    }

    fn request(&self, op: NsOp) -> Result<String, String> {
        let (reply_tx, reply_rx) = std_mpsc::channel();
        if self.job_tx.send(NsJob { op, reply: reply_tx }).is_err() {
            return Err("vfs namespace driver thread is gone".to_owned());
        }
        reply_rx
            .recv()
            .unwrap_or_else(|_| Err("vfs namespace driver thread is gone".to_owned()))
    }
}
