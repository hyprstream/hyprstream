//! Standalone process spawner using tokio::process::Command.
//!
//! Spawns daemon processes directly without systemd integration. Notified
//! launches verify readiness through a per-child sd_notify endpoint and keep
//! ownership of every child until termination is confirmed; children are
//! spawned with `kill_on_drop(false)` ON PURPOSE — adopted daemons are
//! intended to outlive the launcher.
//!
//! Platform support: the credential-authenticated notification receiver
//! (`SO_PASSCRED`/`SCM_CREDENTIALS`, exact child-PID matching) exists on
//! Linux/Android only. Other targets refuse a supervised Notify request
//! before any launch side effect; `Immediate` launches are unchanged
//! everywhere.

// The credential-authenticated notification receiver is Linux/Android-only
// (see `ChildNotifySocket`); its imports are gated with it. Everything the
// Immediate path and the stop/cleanup helpers use stays portable.
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::os::fd::{AsRawFd, OwnedFd};
#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use super::{ProcessConfig, ProcessKind, ProcessReadiness, SpawnedProcess, SpawnerBackend};
use hyprstream_rpc::error::{Result, RpcError};

/// Give adopted daemons independent, pollable standard streams while draining
/// them in the launcher. PGlite registers stdout with epoll, so `/dev/null`
/// is not a safe sink on Linux; a pipe preserves pollability and prevents the
/// launcher's stdout/stderr or SSH channel from being held open by the child.
fn detached_stdio() -> std::io::Result<Stdio> {
    #[cfg(unix)]
    {
        let (reader, writer) = nix::unistd::pipe()?;
        let reader = unsafe { std::fs::File::from_raw_fd(reader) };
        let writer = unsafe { std::fs::File::from_raw_fd(writer) };
        std::thread::Builder::new()
            .name("hyprstream-daemon-stdio-drain".to_owned())
            .spawn(move || {
                let mut reader = reader;
                let mut sink = std::io::sink();
                let _ = std::io::copy(&mut reader, &mut sink);
            })?;
        Ok(Stdio::from(writer))
    }
    #[cfg(not(unix))]
    {
        Ok(Stdio::null())
    }
}

fn configure_detached_stdio(cmd: &mut Command) -> std::io::Result<()> {
    cmd.stdin(detached_stdio()?)
        .stdout(detached_stdio()?)
        .stderr(detached_stdio()?);
    Ok(())
}

/// One per-child sd_notify endpoint in the Linux/Android abstract namespace.
///
/// The launcher binds a compact random datagram address, hands it to the child as
/// `NOTIFY_SOCKET`, and accepts `READY=1` only from the spawned child's PID
/// (via `SO_PASSCRED` sender credentials). This is local lifecycle
/// supervision, not a service RPC endpoint — nothing dials it and no service
/// traffic flows through it.
///
/// Abstract addresses have no filesystem inode or mode and may be visible in
/// `/proc/net/unix`. The random name limits collisions; it is not an access
/// control or authentication mechanism. Kernel-attested sender credentials
/// and exact child-PID matching provide the authentication boundary.
///
/// Platform support: the credential combination this receiver is built on —
/// `SO_PASSCRED` plus `SCM_CREDENTIALS` for kernel-attested sender-PID
/// matching — is Linux/Android-specific in pinned `nix 0.27.1`, so the
/// authenticated receiver as implemented requires those targets. (The
/// datagram creation flags themselves are available on several other Unix
/// targets; the boundary is the credential contract, not the flags.) Other
/// platforms refuse supervised Notify launches pre-spawn (see
/// `spawn_notified`) instead of weakening the sender-PID check.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[derive(Debug)]
struct ChildNotifySocket {
    fd: OwnedFd,
    endpoint: String,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl ChildNotifySocket {
    fn bind(_name: &str) -> Result<Self> {
        Self::bind_with_name_source(|| format!("hypr-n-{}", uuid::Uuid::new_v4().simple()))
    }

    fn bind_with_name_source(mut next_name: impl FnMut() -> String) -> Result<Self> {
        use nix::sys::socket::{
            bind, setsockopt, sockopt::PassCred, socket, AddressFamily, SockFlag, SockType,
            UnixAddr,
        };

        // A new fd and random abstract address are created for each bounded
        // attempt. EADDRINUSE is the only retryable outcome; every other
        // setup error fails closed before a child is spawned.
        for _ in 0..8 {
            let name = next_name();
            let address = UnixAddr::new_abstract(name.as_bytes()).map_err(|e| {
                RpcError::SpawnFailed(format!("notification endpoint name invalid: {e}"))
            })?;
            let fd = socket(
                AddressFamily::Unix,
                SockType::Datagram,
                SockFlag::SOCK_NONBLOCK | SockFlag::SOCK_CLOEXEC,
                None,
            )
            .map_err(|e| RpcError::SpawnFailed(format!("notify socket creation failed: {e}")))?;
            setsockopt(&fd, PassCred, &true)
                .map_err(|e| RpcError::SpawnFailed(format!("SO_PASSCRED setup failed: {e}")))?;
            match bind(fd.as_raw_fd(), &address) {
                Ok(()) => {
                    return Ok(Self {
                        fd,
                        endpoint: format!("@{name}"),
                    })
                }
                Err(nix::errno::Errno::EADDRINUSE) => continue,
                Err(e) => {
                    return Err(RpcError::SpawnFailed(format!(
                        "notification endpoint bind failed: {e}"
                    )))
                }
            }
        }
        Err(RpcError::SpawnFailed(
            "could not bind a unique notification endpoint after 8 attempts".to_owned(),
        ))
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Non-blocking poll for a `READY=1` datagram from exactly `child_pid`.
    ///
    /// Returns `Ok(true)` when the child itself reported readiness,
    /// `Ok(false)` when nothing (or a non-child/spoofed datagram) arrived, and
    /// `Err` on a socket failure. Unrelated senders are ignored, not honored.
    fn try_recv_ready(&self, child_pid: i32) -> Result<bool> {
        use nix::cmsg_space;
        use nix::sys::socket::{
            recvmsg, ControlMessageOwned, MsgFlags, UnixAddr, UnixCredentials,
        };
        use std::io::IoSliceMut;

        let mut buffer = [0u8; 128];
        // Drain the complete nonblocking queue in one poll. A foreign sender
        // must never be able to keep the genuine child's READY=1 datagram
        // behind a sustained backlog; only the exact child PID can succeed.
        loop {
            // Scope the recvmsg borrows: `message` holds the iov/cmsg borrows,
            // so extract what we need before touching the payload buffer again.
            let (bytes, sender_pid) = {
                let mut iov = [IoSliceMut::new(&mut buffer)];
                let mut cmsg = cmsg_space!(UnixCredentials);
                let message = match recvmsg::<UnixAddr>(
                    self.fd.as_raw_fd(),
                    &mut iov,
                    Some(&mut cmsg),
                    MsgFlags::empty(),
                ) {
                    Ok(message) => message,
                    Err(nix::errno::Errno::EAGAIN) => return Ok(false),
                    Err(e) => {
                        return Err(RpcError::SpawnFailed(format!("notify recv failed: {e}")))
                    }
                };
                let sender_pid = message.cmsgs().find_map(|cmsg| match cmsg {
                    ControlMessageOwned::ScmCredentials(creds) => Some(creds.pid()),
                    _ => None,
                });
                (message.bytes, sender_pid)
            };

            if sender_pid != Some(child_pid) {
                tracing::warn!(
                    sender = ?sender_pid,
                    expected = child_pid,
                    "ignoring readiness datagram from a process other than the spawned child"
                );
                continue;
            }
            if bytes > 0 && buffer[..bytes] == *b"READY=1" {
                return Ok(true);
            }
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
/// What the synchronous startup backstop actually observed and did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackstopOutcome {
    /// An explicit terminal wait status was observed AND the guarded
    /// artifact was removed (or was already absent).
    TerminatedAndCleaned,
    /// A terminal status was observed, but the artifact removal FAILED:
    /// the artifact remains on disk as evidence and no cleanup success is
    /// claimed.
    TerminatedArtifactKept,
    /// Termination was NOT observed (kill failure, reap error, budget
    /// exhausted with a nonterminal status, or no usable handle): nothing
    /// is discarded and no cleanup is claimed.
    Unconfirmed,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
/// Terminal-status classification for backstop outcome purposes: only a
/// real exit or kill signal establishes disappearance. StillAlive,
/// stopped, continued, and ptrace states are nonterminal and never
/// release ownership.
fn wait_status_is_terminal(status: &nix::sys::wait::WaitStatus) -> bool {
    matches!(
        status,
        nix::sys::wait::WaitStatus::Exited(..) | nix::sys::wait::WaitStatus::Signaled(..)
    )
}

#[cfg(any(target_os = "linux", target_os = "android"))]
/// Startup ownership of a not-yet-adopted child (#1585).
///
/// Between `spawn` and readiness the launcher — not the process map — owns
/// the child. If anything returns early or the future is cancelled while the
/// guard is still armed, `Drop` performs a synchronous best-effort SIGKILL
/// with a bounded reap and removes any artifact the guard owns. This is best
/// effort within bounded budgets: if the OS kill/reap does not complete in
/// time, the timeout is logged — it is not silently discarded, and no
/// immortal guarantee is claimed.
struct StartupChildGuard {
    child: Option<Child>,
    name: String,
    pid_file: Option<std::path::PathBuf>,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl StartupChildGuard {
    /// Kill and reap the owned child, honestly reporting every failure.
    ///
    /// On partial failure the guard RETAINS ownership (`self.child` stays
    /// armed): the Drop backstop re-attempts SIGKILL and the bounded reap, so
    /// a failed cleanup does not silently disarm fallback ownership and
    /// orphan the child under `kill_on_drop(false)`. The guarded artifact is
    /// removed only after confirmed termination.
    async fn kill_and_reap(&mut self) -> std::result::Result<(), String> {
        let mut failures = Vec::new();
        if let Some(child) = self.child.as_mut() {
            if let Err(e) = child.start_kill() {
                failures.push(format!("kill failed: {e}"));
            }
            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(_)) => {
                    // Reaped: ownership transfer complete.
                    self.child = None;
                }
                Ok(Err(e)) => failures.push(format!("reap failed: {e}")),
                Err(_) => failures.push("reap timed out; ownership retained".to_owned()),
            }
        }
        if self.child.is_none() {
            if let Some(path) = self.pid_file.take() {
                if let Err(e) = std::fs::remove_file(&path) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        failures.push(format!("pid file removal failed: {e}"));
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    /// Synchronous cancellation backstop (the async reap is unavailable in
    /// `Drop`): SIGKILL the owned child and reap it within a bounded WNOHANG
    /// budget. Returns the observed [`BackstopOutcome`]; ownership is
    /// released only when an explicit TERMINAL wait status (real exit or
    /// kill signal — or the already-reaped `ECHILD`) is observed, and the
    /// guarded artifact is removed only when that removal actually
    /// succeeds. Every nonterminal wait state (still-alive, stopped,
    /// continued, ptrace) stays inside the same bounded budget and never
    /// establishes termination. Bounded best effort, not a guarantee.
    fn backstop_cleanup(&mut self) -> BackstopOutcome {
        let mut terminated = false;
        if let Some(child) = self.child.as_ref() {
            if let Some(pid) = child.id() {
                let raw = nix::unistd::Pid::from_raw(pid as i32);
                match nix::sys::signal::kill(raw, nix::sys::signal::Signal::SIGKILL) {
                    Err(nix::errno::Errno::ESRCH) => {
                        // The process is already gone; nothing to signal.
                        terminated = true;
                    }
                    Err(e) => {
                        tracing::warn!(
                            pid = %pid,
                            error = %e,
                            "startup backstop SIGKILL failed; termination not observed"
                        );
                    }
                    Ok(()) => {
                        let deadline = Instant::now() + Duration::from_secs(1);
                        loop {
                            match nix::sys::wait::waitpid(
                                raw,
                                Some(nix::sys::wait::WaitPidFlag::WNOHANG),
                            ) {
                                // Only an explicit terminal status — real
                                // exit or kill signal — establishes the reap.
                                Ok(status) if wait_status_is_terminal(&status) => {
                                    terminated = true;
                                    break;
                                }
                                // StillAlive AND nonterminal stopped/
                                // continued/ptrace states: NOT termination;
                                // keep waiting inside the same budget.
                                Ok(_) => {
                                    if Instant::now() >= deadline {
                                        tracing::warn!(
                                            pid = %pid,
                                            "reap backstop budget exhausted with a nonterminal status; child may linger"
                                        );
                                        break;
                                    }
                                    std::thread::sleep(Duration::from_millis(5));
                                }
                                Err(nix::errno::Errno::ECHILD) => {
                                    // No longer our child: already reaped.
                                    terminated = true;
                                    break;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        pid = %pid,
                                        error = %e,
                                        "startup backstop reap failed; termination not observed"
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !terminated {
            return BackstopOutcome::Unconfirmed;
        }
        self.child = None;
        let Some(path) = self.pid_file.take() else {
            return BackstopOutcome::TerminatedAndCleaned;
        };
        match std::fs::remove_file(&path) {
            // Removed — or already absent: the artifact is gone either way.
            Ok(()) => BackstopOutcome::TerminatedAndCleaned,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                BackstopOutcome::TerminatedAndCleaned
            }
            Err(e) => {
                tracing::warn!(
                    name = %self.name,
                    path = %path.display(),
                    error = %e,
                    "startup backstop could not remove the guarded artifact; retained on disk as evidence"
                );
                self.pid_file = Some(path);
                BackstopOutcome::TerminatedArtifactKept
            }
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl Drop for StartupChildGuard {
    fn drop(&mut self) {
        if self.child.is_some() {
            // Bounded best effort: each outcome is reported exactly as
            // observed. The fields drop with this struct — no handle or
            // cleanup survives an unsuccessful backstop.
            match self.backstop_cleanup() {
                BackstopOutcome::TerminatedAndCleaned => {
                    tracing::warn!(
                        name = %self.name,
                        "startup guard dropped with the child still owned; terminated, reaped, and cleaned up"
                    );
                }
                BackstopOutcome::TerminatedArtifactKept => {
                    tracing::warn!(
                        name = %self.name,
                        "startup guard dropped: child terminated but artifact removal failed; artifact retained on disk as evidence"
                    );
                }
                BackstopOutcome::Unconfirmed => {
                    tracing::warn!(
                        name = %self.name,
                        "startup guard dropped without confirmed termination; owned artifact retained on disk as evidence; no cleanup guarantee beyond Drop"
                    );
                }
            }
        }
    }
}

/// Standalone process spawner backend.
///
/// Spawns processes directly using `tokio::process::Command`.
/// Tracks spawned processes and provides cleanup on stop.
pub struct StandaloneBackend {
    /// Active processes (id -> Child handle).
    processes: DashMap<String, Arc<Mutex<Child>>>,
}

impl StandaloneBackend {
    /// Create a new standalone backend.
    pub fn new() -> Self {
        Self {
            processes: DashMap::new(),
        }
    }
}

impl Default for StandaloneBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StandaloneBackend {
    /// Legacy path: report success as soon as the child is spawned.
    ///
    /// Documented limitations (pre-existing at HEAD #1585; kept for
    /// compatibility, NOT the required-native contract): readiness is not
    /// verified in any way; PID publication is a direct, non-atomic write
    /// whose failure is logged rather than rolled back; and a pre-existing
    /// PID artifact is overwritten without a live-duplicate check, so a
    /// recycled PID in a stale artifact can misdirect later PID-based
    /// cleanup. The Required notified path (`spawn_notified`) is where the
    /// checked-publication/rollback contract is enforced.
    async fn spawn_immediate(&self, config: ProcessConfig) -> Result<SpawnedProcess> {
        tracing::debug!(
            name = %config.name,
            executable = %config.executable.display(),
            "Spawning daemon via direct process"
        );

        // Build the command
        let mut cmd = Command::new(&config.executable);

        // Add arguments
        cmd.args(&config.args);

        // Set working directory
        if let Some(ref dir) = config.working_dir {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        // Disable kill on drop - daemon processes should outlive the spawner
        configure_detached_stdio(&mut cmd).map_err(|e| {
            RpcError::SpawnFailed(format!("failed to detach {} standard streams: {e}", config.name))
        })?;
        cmd.kill_on_drop(false);

        // Spawn the process
        let child = cmd.spawn().map_err(|e| {
            RpcError::SpawnFailed(format!("failed to spawn {}: {}", config.name, e))
        })?;

        // Get the PID
        let pid = child.id().ok_or_else(|| {
            RpcError::SpawnFailed(format!("spawned {} but no PID available", config.name))
        })?;

        // Generate unique ID
        let id = format!("{}-{}", config.name, pid);

        // Store the child handle
        self.processes
            .insert(id.clone(), Arc::new(Mutex::new(child)));

        // Write PID file for daemon tracking
        let pid_file = hyprstream_rpc::paths::service_pid_file(&config.name);
        if let Some(parent) = pid_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(&pid_file, pid.to_string()) {
            tracing::warn!("Failed to write PID file {:?}: {}", pid_file, e);
        }

        tracing::info!(
            name = %config.name,
            pid = %pid,
            id = %id,
            "Daemon spawned successfully"
        );

        Ok(SpawnedProcess::new(id, ProcessKind::Direct(pid))
            .with_pid_file(pid_file))
    }

    /// Notify-supervised path (#1585): the spawn reports success only after
    /// the child's own `READY=1` datagram arrives from the child's PID; a
    /// child exit or the hard timeout fails the spawn, the child is
    /// terminated/reaped, and no PID artifact or live abstract notification
    /// endpoint survives.
    ///
    /// Platform support: the credential-authenticated receiver
    /// (`SO_PASSCRED`/`SCM_CREDENTIALS`, exact child-PID matching) exists on
    /// Linux/Android only. On any other target this request fails closed
    /// BEFORE spawning the child or binding a notification endpoint, creating
    /// a PID artifact, or causing any other launch side effect — there is no
    /// fallback to
    /// `Immediate` readiness and no unauthenticated receiver.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    async fn spawn_notified(
        &self,
        config: ProcessConfig,
        timeout: Duration,
    ) -> Result<SpawnedProcess> {
        tracing::debug!(
            name = %config.name,
            executable = %config.executable.display(),
            timeout_ms = %timeout.as_millis(),
            "Spawning daemon via direct process with notification readiness"
        );

        // Serialize same-service launches for the WHOLE launch — duplicate
        // precheck, publication, and any rollback run under this lock; it is
        // acquired first so it is dropped last, after every early-return
        // cleanup path. A held lock deterministically reports the pending
        // duplicate.
        let _launch_lock = ServiceLaunchLock::acquire(&config.name)?;
        let pid_file = hyprstream_rpc::paths::service_pid_file(&config.name);

        // Duplicate precheck BEFORE any side effect of this launch: no
        // notification endpoint is created and no child is spawned for a
        // refused launch. Only a confirmed-missing or confirmed-dead
        // predecessor may be replaced; ambiguous artifacts fail closed.
        match inspect_pid_file(&pid_file) {
            PidWitness::Absent => {}
            PidWitness::Dead(stale) => {
                tracing::info!(
                    name = %config.name,
                    stale_pid = %stale,
                    "replacing PID artifact of a confirmed-dead predecessor"
                );
            }
            PidWitness::Live(existing) => {
                return Err(RpcError::SpawnFailed(format!(
                    "service {} launch refused: service already appears live as pid {existing}",
                    config.name
                )));
            }
            PidWitness::Ambiguous(why) => {
                return Err(RpcError::SpawnFailed(format!(
                    "service {} launch refused: existing PID artifact is ambiguous ({why}); refusing to replace it",
                    config.name
                )));
            }
        }

        let notify = ChildNotifySocket::bind(&config.name)?;

        let mut cmd = Command::new(&config.executable);
        cmd.args(&config.args);
        if let Some(ref dir) = config.working_dir {
            cmd.current_dir(dir);
        }
        for (key, value) in &config.env {
            cmd.env(key, value);
        }
        // The child learns its notification endpoint only through its own
        // environment; the launcher's environment is untouched.
        cmd.env("NOTIFY_SOCKET", notify.endpoint());
        configure_detached_stdio(&mut cmd).map_err(|e| {
            RpcError::SpawnFailed(format!("failed to detach {} standard streams: {e}", config.name))
        })?;
        cmd.kill_on_drop(false);

        let child = cmd.spawn().map_err(|e| {
            RpcError::SpawnFailed(format!("failed to spawn {}: {}", config.name, e))
        })?;
        // Take ownership BEFORE any fallible branch: if the handle yields no
        // PID, the guard must still own and reap the child (`kill_on_drop`
        // is false by design — adopted daemons must outlive the launcher).
        let mut guard = StartupChildGuard {
            child: Some(child),
            name: config.name.clone(),
            pid_file: None,
        };
        // Capture the PID before any await: it must remain identifiable in
        // every error raised after this point.
        let Some(pid) = guard.child.as_ref().and_then(tokio::process::Child::id) else {
            let cleanup = guard.kill_and_reap().await;
            return Err(RpcError::SpawnFailed(format!(
                "service {} produced no PID{}",
                config.name,
                cleanup_suffix(cleanup),
            )));
        };

        // Readiness outcome; classified from the child's actual behavior, not
        // from whether its PID is still observable afterwards.
        let outcome = match guard.child.as_mut() {
            Some(owned) => self.await_child_readiness(owned, &notify, timeout).await,
            None => Ok(()),
        };

        match outcome {
            Ok(()) => {
                // READY arrived; reject a child that exited immediately after.
                let Some(owned) = guard.child.as_mut() else {
                    // Unreachable while the guard owns the child until
                    // adoption; if it ever occurs, fail closed with rollback.
                    let cleanup = guard.kill_and_reap().await;
                    return Err(RpcError::SpawnFailed(format!(
                        "service {} (pid {}) lost child ownership before the readiness recheck{}",
                        config.name,
                        pid,
                        cleanup_suffix(cleanup),
                    )));
                };
                match owned.try_wait() {
                    Ok(None) => {}
                    Ok(Some(status)) => {
                        let cleanup = guard.kill_and_reap().await;
                        return Err(RpcError::SpawnFailed(format!(
                            "service {} (pid {}) exited right after reporting readiness: {status}{}",
                            config.name,
                            pid,
                            cleanup_suffix(cleanup),
                        )));
                    }
                    Err(e) => {
                        let cleanup = guard.kill_and_reap().await;
                        return Err(RpcError::SpawnFailed(format!(
                            "service {} (pid {}) readiness recheck failed: {e}{}",
                            config.name,
                            pid,
                            cleanup_suffix(cleanup),
                        )));
                    }
                }
            }
            Err(reason) => {
                // Terminate and reap; the PID file is never published for a
                // child that never became ready.
                let cleanup = guard.kill_and_reap().await;
                return Err(RpcError::SpawnFailed(format!(
                    "service {} (pid {}) did not become ready: {reason}{}",
                    config.name,
                    pid,
                    cleanup_suffix(cleanup),
                )));
            }
        }

        // Atomic PID publication while the startup guard still owns the child:
        // any failure rolls the launch back (child killed/reaped, artifact
        // removed) instead of reporting success with untracked state. The
        // per-service launch lock is held for this whole section, so the
        // precheck and the publication are one deterministic unit.
        if let Some(parent) = pid_file.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                let cleanup = guard.kill_and_reap().await;
                return Err(RpcError::SpawnFailed(format!(
                    "service {} (pid {}) runtime dir creation failed: {e}{}",
                    config.name,
                    pid,
                    cleanup_suffix(cleanup),
                )));
            }
        }
        // Exclusive per-attempt temp artifact: `create_new` never clobbers a
        // pre-existing file, the PID bytes are written through the RETAINED
        // created handle (the exclusive inode ownership established at
        // creation is never traded for a reopened pathname), and the guard
        // is armed only for the file THIS attempt actually created.
        let temp_pid_file = {
            let mut created = None;
            for _ in 0..8 {
                let nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                let candidate = pid_file.with_extension(format!("pid.{nanos}"));
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&candidate)
                {
                    Ok(file) => {
                        created = Some((candidate, file));
                        break;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => {
                        let cleanup = guard.kill_and_reap().await;
                        return Err(RpcError::SpawnFailed(format!(
                            "service {} (pid {}) PID publication failed: {e}{}",
                            config.name,
                            pid,
                            cleanup_suffix(cleanup),
                        )));
                    }
                }
            }
            let Some((candidate, mut file)) = created else {
                let cleanup = guard.kill_and_reap().await;
                return Err(RpcError::SpawnFailed(format!(
                    "service {} (pid {}) PID publication failed: no unique temp artifact name{}",
                    config.name,
                    pid,
                    cleanup_suffix(cleanup),
                )));
            };
            // The guard owns this attempt's artifact from the moment of its
            // exclusive creation: any rollback removes it together with the
            // confirmed reap.
            guard.pid_file = Some(candidate.clone());
            use std::io::Write as _;
            if let Err(e) = file.write_all(pid.to_string().as_bytes()) {
                let cleanup = guard.kill_and_reap().await;
                return Err(RpcError::SpawnFailed(format!(
                    "service {} (pid {}) PID publication failed: {e}{}",
                    config.name,
                    pid,
                    cleanup_suffix(cleanup),
                )));
            }
            candidate
        };
        if let Err(e) = std::fs::rename(&temp_pid_file, &pid_file) {
            let cleanup = guard.kill_and_reap().await;
            return Err(RpcError::SpawnFailed(format!(
                "service {} (pid {}) PID publication failed: {e}{}",
                config.name,
                pid,
                cleanup_suffix(cleanup),
            )));
        }

        // Generate unique ID
        let id = format!("{}-{}", config.name, pid);

        // Transfer ownership: the process map adopts the live child and the
        // PID file becomes `stop`'s cleanup responsibility.
        let Some(owned) = guard.child.take() else {
            // Unreachable while the guard owns the child until adoption. The
            // handle is gone, so there is nothing to kill or reap; this
            // attempt's temp artifact is removed rather than leaked.
            if let Some(path) = guard.pid_file.take() {
                let _ = std::fs::remove_file(&path);
            }
            return Err(RpcError::SpawnFailed(format!(
                "service {} (pid {}) lost child ownership before adoption",
                config.name, pid,
            )));
        };
        guard.pid_file = None;
        self.processes
            .insert(id.clone(), Arc::new(Mutex::new(owned)));

        tracing::info!(
            name = %config.name,
            pid = %pid,
            id = %id,
            "Daemon spawned successfully (notification readiness satisfied)"
        );

        // ChildNotifySocket::drop closes the abstract endpoint.
        Ok(SpawnedProcess::new(id, ProcessKind::Direct(pid))
            .with_pid_file(pid_file))
    }

    /// Non-Linux arm of the Notify-supervised path: an explicit pre-spawn
    /// refusal (see the Linux/Android arm's contract above). Nothing is
    /// spawned and no notification endpoint, child guard, or PID artifact is
    /// created; there is deliberately no fallback to `Immediate` readiness,
    /// because a supervised launch whose sender cannot be kernel-attested
    /// must not report success from an unauthenticated payload.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    async fn spawn_notified(
        &self,
        _config: ProcessConfig,
        _timeout: Duration,
    ) -> Result<SpawnedProcess> {
        Err(RpcError::SpawnFailed(
            "authenticated notification readiness (SO_PASSCRED/SCM_CREDENTIALS) is \
             supported on Linux/Android only; a supervised Notify launch is refused \
             on this platform"
                .to_owned(),
        ))
    }

    /// Poll the notification endpoint against child exit and the hard timeout.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    async fn await_child_readiness(
        &self,
        child: &mut Child,
        notify: &ChildNotifySocket,
        timeout: Duration,
    ) -> std::result::Result<(), String> {
        let deadline = Instant::now() + timeout;
        let mut poll = tokio::time::interval(Duration::from_millis(20));
        poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick completes immediately; skip it so the first real
        // poll happens one interval in.
        poll.tick().await;

        loop {
            match notify.try_recv_ready(child.id().map(|p| p as i32).unwrap_or(-1)) {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(e) => return Err(e.to_string()),
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Err(format!("exited during startup with {status}"));
                }
                Ok(None) => {}
                Err(e) => return Err(format!("exit observation failed: {e}")),
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "no READY=1 within {}s (notification endpoint {})",
                    timeout.as_secs(),
                    notify.endpoint()
                ));
            }
            let wait = poll.tick();
            tokio::select! {
                _ = wait => {}
                status = child.wait() => {
                    // Exit while sleeping between polls is still a startup
                    // failure with the real exit status.
                    return match status {
                        Ok(status) => Err(format!("exited during startup with {status}")),
                        Err(e) => Err(format!("exit observation failed: {e}")),
                    };
                }
            }
        }
    }
}

/// A self-exited child stays visible to `kill(pid, 0)` as a zombie until its
/// supervisor reaps it; a zombie is not a running service. Best-effort procfs
/// inspection — if the state cannot be read the process is conservatively
/// treated as alive (previous `kill(pid, 0)` semantics).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn process_is_zombie(pid: i32) -> bool {
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    // The comm field is parenthesized and may itself contain parentheses or
    // spaces; the single-character state code follows the final ')'.
    let Some(after_comm) = stat
        .rsplit_once(')')
        .map(|(_, rest)| rest.trim_start())
    else {
        return false;
    };
    after_comm.starts_with('Z')
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn process_is_zombie(_pid: i32) -> bool {
    false
}

/// Format an optional cleanup failure onto a startup error: a cleanup that
/// itself failed is part of the failure report, never discarded.
/// Rollback-failure suffix for spawn error messages (used by the
/// Linux/Android supervised Notify arm only).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn cleanup_suffix(cleanup: std::result::Result<(), String>) -> String {
    match cleanup {
        Ok(()) => String::new(),
        Err(failure) => format!("; cleanup: {failure}"),
    }
}

/// Outcome of inspecting an existing PID artifact before a launch. Anything
/// other than a confirmed-missing or confirmed-dead predecessor is ambiguous
/// and must fail closed: only a verified missing/dead witness may be replaced.
#[cfg(any(target_os = "linux", target_os = "android"))]
enum PidWitness {
    /// No artifact: nothing to refuse, nothing to preserve.
    Absent,
    /// Parseable PID naming a non-live process: a stale witness, replaceable.
    Dead(u32),
    /// Parseable PID naming a live process: a duplicate launch must be
    /// refused and this witness preserved.
    Live(u32),
    /// Unreadable or unparseable: the artifact cannot vouch either way.
    Ambiguous(String),
}

/// Inspect an existing PID artifact into a [`PidWitness`] (Linux/Android;
/// used by the supervised Notify precheck).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn inspect_pid_file(path: &std::path::Path) -> PidWitness {
    match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => PidWitness::Absent,
        Err(e) => PidWitness::Ambiguous(format!("unreadable: {e}")),
        Ok(content) => match content.trim().parse::<u32>() {
            Err(_) => PidWitness::Ambiguous("unparseable pid".to_owned()),
            Ok(pid) if pid_is_live(pid) => PidWitness::Live(pid),
            Ok(pid) => PidWitness::Dead(pid),
        },
    }
}

/// Liveness of a PID witness: present (signalable — or present but
/// unsignalable, which is conservatively live), and NOT a zombie — a
/// self-exited process awaiting reap is a stale witness.
fn pid_is_live(pid: u32) -> bool {
    let Some(pid) = i32::try_from(pid).ok().filter(|&p| p > 0) else {
        return false;
    };
    matches!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
        Ok(()) | Err(nix::errno::Errno::EPERM)
    ) && !process_is_zombie(pid)
}

/// Terminate an untracked daemon handle by PID: SIGTERM with a bounded wait,
/// then SIGKILL with a bounded POST-SIGKILL confirmation. Returns whether
/// death was OBSERVED (ESRCH) — callers touch artifacts only on `true`.
///
/// Known pre-existing limitation (documented, not redesigned in #1585): on
/// an untracked handle the PID is a bare witness, so a recycled PID after
/// the target's exit can in principle be signaled; tracked children are
/// observed through their real handle and have no such exposure.
async fn stop_untracked_by_pid(pid: u32, failures: &mut Vec<String>) -> bool {
    let Some(pid_i32) = i32::try_from(pid).ok().filter(|&p| p > 0) else {
        failures.push("invalid PID".to_owned());
        return false;
    };
    let raw = nix::unistd::Pid::from_raw(pid_i32);
    match nix::sys::signal::kill(raw, nix::sys::signal::Signal::SIGTERM) {
        Err(nix::errno::Errno::ESRCH) => return true, // already gone
        Err(e) => {
            failures.push(format!("SIGTERM failed: {e}"));
            return false;
        }
        Ok(()) => {}
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match nix::sys::signal::kill(raw, None) {
            Err(nix::errno::Errno::ESRCH) => return true,
            Err(e) => {
                failures.push(format!("liveness check failed: {e}"));
                return false;
            }
            Ok(()) if Instant::now() >= deadline => break, // escalate
            Ok(()) => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
    match nix::sys::signal::kill(raw, nix::sys::signal::Signal::SIGKILL) {
        Err(nix::errno::Errno::ESRCH) => return true,
        Err(e) => {
            failures.push(format!("SIGKILL failed: {e}"));
            return false;
        }
        Ok(()) => {}
    }
    // SIGKILL delivery is fast but not instantaneous: death must be
    // observed, not assumed.
    let kill_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match nix::sys::signal::kill(raw, None) {
            Err(nix::errno::Errno::ESRCH) => return true,
            Err(e) => {
                failures.push(format!("post-SIGKILL liveness check failed: {e}"));
                return false;
            }
            Ok(()) if Instant::now() >= kill_deadline => {
                failures.push("termination not confirmed after SIGKILL".to_owned());
                return false;
            }
            Ok(()) => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
}

impl StandaloneBackend {
    /// Synchronous bounded stop for a tracked direct child, run from the
    /// launch-transaction guard's `Drop` when the orchestration future is
    /// cancelled (#1585).
    ///
    /// This preserves the backend's STRONGER tracked ownership instead of
    /// downgrading adopted children to a blind by-PID signal: when the child
    /// is still in the tracking map, termination goes through the retained
    /// `Child` handle itself (`start_kill` + bounded `try_wait` reaping — the
    /// sync counterpart of the tracked branch of [`Self::stop`], same 5s
    /// budget), and the map entry is surrendered only on a CONFIRMED reap. A
    /// child no longer in the map is either already stopped by an earlier
    /// confirmed stop or was never this backend's: for a live untracked PID
    /// this operation deliberately REFUSES to signal (a recycled PID could be
    /// an unrelated process) and fails honestly instead; an untracked PID
    /// already gone (`kill(pid, None)` → ESRCH — a liveness probe, never a
    /// signal) is cleaned up. PID-artifact removal keeps the established
    /// conditional semantics — the artifact is removed only while it still
    /// names this child, a newer child's artifact is preserved, and an actual
    /// IO failure is an error, not success. Best effort within bounded
    /// budgets, not a guarantee.
    pub fn stop_tracked_child_by_handle_sync(&self, process: &SpawnedProcess) -> Result<()> {
        if !process.is_direct() {
            return Err(RpcError::InvalidOperation(
                "synchronous cancellation stop requires a direct (PID-tracked) process"
                    .to_owned(),
            ));
        }
        let pid = process.pid().unwrap_or_default();

        // Tracked branch: the retained Child IS the ownership — no PID
        // signaling at all.
        let child_arc = self
            .processes
            .get(&process.id)
            .map(|entry| Arc::clone(entry.value()));
        if let Some(child_arc) = child_arc {
            let mut child = match child_arc.try_lock() {
                Ok(child) => child,
                Err(_) => {
                    return Err(RpcError::SpawnFailed(
                        "cancellation cleanup skipped: child handle lock contended; \
                         child remains tracked"
                            .to_owned(),
                    ));
                }
            };
            let mut failures: Vec<String> = Vec::new();
            if let Err(e) = child.start_kill() {
                failures.push(format!("kill failed: {e}"));
            }
            // Bounded synchronous reap (mirror of the tracked `stop` branch).
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut confirmed = false;
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        confirmed = true;
                        // Reaped: only now does the backend stop tracking.
                        self.processes.remove(&process.id);
                        break;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        failures.push(format!("reap failed: {e}"));
                        break;
                    }
                }
                if Instant::now() >= deadline {
                    failures.push("reap timed out; child remains tracked".to_owned());
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            if !confirmed {
                failures.push("termination unconfirmed".to_owned());
                // Unconfirmed: the PID artifact still names a possibly-live
                // process and cleanup must not claim success.
                return Err(RpcError::SpawnFailed(format!(
                    "synchronous cancellation stop of {} did not fully succeed: {}",
                    process.id,
                    failures.join("; ")
                )));
            }
            return finish_pid_artifact_sync(process, pid, &mut failures);
        }

        // Untracked branch: never signal a PID we do not track. Probe-only
        // liveness (kill(pid, None) is a probe, not a signal): gone → clean
        // the artifact; alive → fail honestly rather than risk a recycled
        // PID (the documented untracked exposure is not widened here).
        let raw = nix::unistd::Pid::from_raw(i32::try_from(pid).unwrap_or_default());
        match nix::sys::signal::kill(raw, None) {
            Err(nix::errno::Errno::ESRCH) => {
                let mut failures: Vec<String> = Vec::new();
                finish_pid_artifact_sync(process, pid, &mut failures)
            }
            Err(e) => Err(RpcError::SpawnFailed(format!(
                "cancellation liveness probe of untracked {} failed: {e}",
                process.id
            ))),
            Ok(()) => Err(RpcError::SpawnFailed(format!(
                "cancellation cleanup declined to signal untracked live PID {pid} ({})",
                process.id
            ))),
        }
    }
}

/// Conditional PID-artifact removal after a CONFIRMED synchronous stop: the
/// artifact is removed only while it still names `pid`; a file naming a
/// different (newer) child is intentionally preserved and is NOT a failure;
/// an actual IO failure IS a failure — cleanup never claims clean success it
/// did not achieve (#1585).
fn finish_pid_artifact_sync(
    process: &SpawnedProcess,
    pid: u32,
    failures: &mut Vec<String>,
) -> Result<()> {
    let Some(pid_file) = &process.pid_file else {
        return Ok(());
    };
    match std::fs::read_to_string(pid_file) {
        Ok(content) => {
            if content.trim() == pid.to_string() {
                match std::fs::remove_file(pid_file) {
                    Ok(()) => Ok(()),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    Err(e) => {
                        failures.push(format!("PID file removal failed: {e}"));
                        Err(RpcError::SpawnFailed(format!(
                            "synchronous stop of {} confirmed termination but the PID \
                             artifact cleanup failed: {}",
                            process.id,
                            failures.join("; ")
                        )))
                    }
                }
            } else {
                // The artifact names a different (newer) child: preserving it
                // is correct, not a failure.
                Ok(())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(RpcError::SpawnFailed(format!(
            "synchronous stop of {} confirmed termination but the PID artifact could \
             not be read for conditional removal: {e}",
            process.id
        ))),
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
/// Per-service nonblocking advisory lock for notified launches (#1585).
///
/// The lock file has a STABLE path (and thus stable inode) and is NEVER
/// unlinked — unlinking would break flock identity between concurrent
/// acquirers. Holding it makes the duplicate precheck, the PID publication,
/// and any rollback of one launch a deterministic unit against other launches
/// of the same service. Release is an EXPLICIT `FlockArg::Unlock` in `Drop`
/// followed by normal close: `flock` lives on the open file description, so
/// close alone can be prolonged by a concurrently forked process's transient
/// duplicate of that description (pre-exec), which would transiently and
/// falsely refuse the next same-service launch; explicit unlock releases the
/// lock immediately regardless of such copies. A held lock is reported as a
/// pending duplicate, deterministically. Scope is honest and narrow: it
/// coordinates only cooperating notified launches using this facility and
/// makes no claim of control over noncooperating processes; the artifact
/// lives in the protected runtime directory.
struct ServiceLaunchLock {
    _file: std::fs::File,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl Drop for ServiceLaunchLock {
    fn drop(&mut self) {
        use nix::fcntl::{flock, FlockArg};
        use std::os::fd::AsRawFd;

        // Explicit unlock BEFORE close (see the type docs): close alone
        // waits for every duplicate of this open file description to close,
        // which a concurrently forked child's pre-exec copy can delay
        // indefinitely. An unlock failure is logged honestly — the
        // subsequent close remains the fallback and the next acquirer may
        // then see a transient in-progress refusal (fail-closed), never a
        // duplicate acceptance.
        let fd = self._file.as_raw_fd();
        if let Err(e) = flock(fd, FlockArg::Unlock) {
            tracing::warn!(
                fd = %fd,
                error = %e,
                "launch lock explicit unlock failed; falling back to close semantics"
            );
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl ServiceLaunchLock {
    fn acquire(service: &str) -> Result<Self> {
        use nix::fcntl::{flock, FlockArg};
        use std::os::fd::AsRawFd;

        let path =
            hyprstream_rpc::paths::runtime_dir().join(format!("{service}.launch.lock"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                RpcError::SpawnFailed(format!(
                    "service {service} launch lock dir creation failed: {e}"
                ))
            })?;
        }
        // Create-or-open WITHOUT truncation: the file's only role is existing
        // as a stable inode for flock.
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| {
                RpcError::SpawnFailed(format!(
                    "service {service} launch lock open failed: {e}"
                ))
            })?;
        match flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock) {
            Ok(()) => Ok(Self { _file: file }),
            Err(nix::errno::Errno::EWOULDBLOCK) => Err(RpcError::SpawnFailed(format!(
                "service {service} launch refused: another launch is already in progress (launch lock held)"
            ))),
            Err(e) => Err(RpcError::SpawnFailed(format!(
                "service {service} launch lock failed: {e}"
            ))),
        }
    }
}

#[async_trait::async_trait]
impl SpawnerBackend for StandaloneBackend {
    async fn spawn(&self, config: ProcessConfig) -> Result<SpawnedProcess> {
        match config.readiness {
            ProcessReadiness::Immediate => self.spawn_immediate(config).await,
            ProcessReadiness::Notify { timeout } => self.spawn_notified(config, timeout).await,
        }
    }

    async fn stop(&self, process: &SpawnedProcess) -> Result<()> {
        let pid = match &process.kind {
            ProcessKind::Direct(pid) => *pid,
            ProcessKind::SystemdUnit(_) => {
                return Err(RpcError::InvalidOperation(
                    "StandaloneBackend cannot stop systemd units".to_owned(),
                ));
            }
        };

        tracing::debug!(
            id = %process.id,
            pid = %pid,
            "Stopping direct process"
        );

        // Clone the tracked handle out of the map before awaiting; the map
        // entry itself is removed ONLY after termination is CONFIRMED, so a
        // failed or timed-out cleanup leaves the child tracked for a
        // retrying stop instead of losing ownership mid-cleanup. That
        // retention holds only while THIS backend lives: backend drop or
        // process exit releases the map without killing anything, and the
        // bounded startup Drop cannot guarantee retained retry ownership
        // either — no reaper framework exists beyond this.
        let child_arc = self
            .processes
            .get(&process.id)
            .map(|entry| Arc::clone(entry.value()));

        let mut failures: Vec<String> = Vec::new();
        let mut confirmed = false;
        if let Some(child_arc) = child_arc {
            let mut child = child_arc.lock().await;

            // Forceful termination (`start_kill` sends SIGKILL — the comment
            // above used to promise SIGTERM; it does not).
            if let Err(e) = child.start_kill() {
                failures.push(format!("kill failed: {e}"));
            }
            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(_)) => {
                    confirmed = true;
                    // Reaped: only now does the backend stop tracking the
                    // child.
                    self.processes.remove(&process.id);
                }
                Ok(Err(e)) => failures.push(format!("reap failed: {e}")),
                Err(_) => failures.push("reap timed out; child remains tracked".to_owned()),
            }
        } else {
            // Not tracked (adopted handle, or a retry after a confirmed
            // earlier stop): terminate by PID, with termination OBSERVED
            // before any artifact is touched.
            tracing::debug!(
                id = %process.id,
                pid = %pid,
                "Process not in map, attempting direct kill by PID"
            );
            confirmed = stop_untracked_by_pid(pid, &mut failures).await;
        }

        if !confirmed {
            // Unconfirmed termination: the PID artifact still names a
            // possibly-live process and success must not be reported.
            failures.push("termination unconfirmed".to_owned());
            return Err(RpcError::SpawnFailed(format!(
                "stop of {} did not fully succeed: {}",
                process.id,
                failures.join("; ")
            )));
        }

        // Conditional PID artifact removal: delete only if the file still
        // names THIS child, so stopping an older duplicate cannot erase a
        // newer process's PID file.
        if let Some(pid_file) = &process.pid_file {
            match std::fs::read_to_string(pid_file) {
                Ok(content) => {
                    let same_child = content.trim() == pid.to_string();
                    if same_child {
                        if let Err(e) = std::fs::remove_file(pid_file) {
                            if e.kind() != std::io::ErrorKind::NotFound {
                                failures.push(format!(
                                    "PID file removal failed: {e}"
                                ));
                            }
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => failures.push(format!("PID file read failed: {e}")),
            }
        }

        if failures.is_empty() {
            tracing::info!(
                id = %process.id,
                "Process stopped"
            );
            Ok(())
        } else {
            Err(RpcError::SpawnFailed(format!(
                "stop of {} did not fully succeed: {}",
                process.id,
                failures.join("; ")
            )))
        }
    }

    async fn is_running(&self, process: &SpawnedProcess) -> Result<bool> {
        let pid = match &process.kind {
            ProcessKind::Direct(pid) => *pid,
            ProcessKind::SystemdUnit(_) => {
                return Err(RpcError::InvalidOperation(
                    "StandaloneBackend cannot check systemd units".to_owned(),
                ));
            }
        };

        let Some(pid_i32) = i32::try_from(pid).ok().filter(|&p| p > 0) else {
            return Ok(false); // Invalid PID means not running
        };

        // A TRACKED child is observed directly: `try_wait` observes — and
        // reaps — the actual child. That is portable and authoritative, and
        // it cannot mistake a self-exited-but-unreaped child (a zombie,
        // still visible to `kill(pid, 0)`) for a running service.
        if let Some(child_arc) = self
            .processes
            .get(&process.id)
            .map(|entry| Arc::clone(entry.value()))
        {
            let mut child = child_arc.lock().await;
            return match child.try_wait() {
                Ok(Some(_)) => Ok(false), // exited and reaped: not running
                Ok(None) => Ok(true),     // still running
                Err(e) => {
                    tracing::warn!(pid = %pid, error = %e, "Error checking child status");
                    Ok(false)
                }
            };
        }

        // Untracked daemon handle (adopted, or after a confirmed stop):
        // bounded legacy liveness fallback by PID signal 0, zombie-aware so
        // a self-exited process awaiting reap is not reported as running.
        let nix_pid = nix::unistd::Pid::from_raw(pid_i32);
        match nix::sys::signal::kill(nix_pid, None) {
            // Present (Ok) or present-but-unsignalable (EPERM) — but a
            // zombie is a self-exited child awaiting reap, not a running
            // service, so it must not be reported as running.
            Ok(()) | Err(nix::errno::Errno::EPERM) => Ok(!process_is_zombie(pid_i32)),
            Err(nix::errno::Errno::ESRCH) => Ok(false), // No such process
            Err(e) => {
                tracing::warn!(pid = %pid, error = %e, "Error checking process status");
                Ok(false)
            }
        }
    }

    fn stop_tracked_child_sync(&self, process: &SpawnedProcess) -> Result<()> {
        Self::stop_tracked_child_by_handle_sync(self, process)
    }

    fn backend_type(&self) -> &'static str {
        "standalone"
    }
}

impl Drop for StandaloneBackend {
    fn drop(&mut self) {
        // Children are spawned with kill_on_drop(false) ON PURPOSE: adopted
        // daemons are intended to outlive the launcher. Dropping the backend
        // therefore releases tracking only — it does NOT kill anything, and
        // no cleanup is performed here.
        let count = self.processes.len();
        if count > 0 {
            tracing::debug!(
                count = count,
                "StandaloneBackend dropped with tracked processes (NOT killed: daemons outlive the launcher)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_and_stop() -> hyprstream_rpc::Result<()> {
        let backend = StandaloneBackend::new();

        // Spawn a simple process (sleep)
        let config = ProcessConfig::new("test-sleep", "sleep").args(["100"]);

        let result = backend.spawn(config).await;

        match result {
            Ok(process) => {
                assert!(process.is_direct());
                assert!(process.pid().is_some());

                // Check it's running
                let running = backend.is_running(&process).await?;
                assert!(running, "Process should be running");

                // Stop it
                backend.stop(&process).await?;

                // Give it a moment to die
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;

                // Check it's stopped
                let running = backend.is_running(&process).await?;
                assert!(!running, "Process should be stopped");
            }
            #[allow(clippy::print_stderr)]
            Err(e) => {
                // sleep might not be available in some test environments
                eprintln!("Could not spawn test process: {e}");
            }
        }
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn standalone_spawn_preserves_non_utf8_argument_bytes() -> anyhow::Result<()> {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let root = tempfile::tempdir()?;
        let script = root.path().join("record-argument.sh");
        let observed = root.path().join("observed.bin");
        std::fs::write(
            &script,
            b"printf '%s' \"$1\" > \"$2\"\nexec sleep 30\n",
        )?;
        let expected = std::ffi::OsString::from_vec(b"config-\xff.toml".to_vec());
        let name = format!("non-utf8-argv-{}", uuid::Uuid::new_v4().simple());
        let backend = StandaloneBackend::new();
        let config = ProcessConfig::new(&name, "/bin/sh").args([
            script.as_os_str().to_owned(),
            expected.clone(),
            observed.as_os_str().to_owned(),
        ]);
        let process = backend.spawn(config).await?;

        let deadline = Instant::now() + Duration::from_secs(5);
        let observation = loop {
            match std::fs::read(&observed) {
                Ok(bytes) if bytes == expected.as_os_str().as_bytes() => break Ok(()),
                Ok(_) | Err(_) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Ok(bytes) => break Err(anyhow::anyhow!(
                    "child changed argument bytes: observed {bytes:?}"
                )),
                Err(error) => break Err(anyhow::anyhow!(
                    "child did not record its argument before the deadline: {error}"
                )),
            }
        };
        let stop = backend.stop(&process).await;
        observation?;
        stop?;
        assert!(!backend.is_running(&process).await?);
        assert!(
            !hyprstream_rpc::paths::service_pid_file(&name).exists(),
            "stopped child must leave no PID artifact"
        );
        Ok(())
    }

    #[test]
    fn test_backend_type() {
        let backend = StandaloneBackend::new();
        assert_eq!(backend.backend_type(), "standalone");
    }
}

/// Linux/Android-only coverage of the credential-authenticated receiver:
/// positive readiness, foreign-PID rejection, timeout/early-exit rollback,
/// spoofed-sender refusal, and the launcher-side cancellation/rollback
/// ownership regressions that exercise the supervised Notify path end to
/// end. These tests RUN on Linux/Android — the cfg only removes them from
/// non-Linux builds, where the supervised Notify path itself is refused
/// pre-spawn.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[cfg(test)]
mod notify_readiness_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use std::path::Path;

    const READY_HELPER: &str = "HYPRSTREAM_NOTIFY_READY_HELPER";
    const PENDING_HELPER: &str = "HYPRSTREAM_LAUNCHER_PENDING_HELPER";
    const PENDING_MARKER: &str = "HYPRSTREAM_LAUNCHER_PENDING_MARKER";
    const PENDING_RELEASE: &str = "HYPRSTREAM_LAUNCHER_PENDING_RELEASE";
    const OBSTACLE_HELPER: &str = "HYPRSTREAM_LAUNCHER_OBSTACLE_HELPER";
    const OBSTACLE_PATH: &str = "HYPRSTREAM_LAUNCHER_OBSTACLE_PATH";
    const LONG_RUNTIME_HELPER: &str = "HYPRSTREAM_LONG_NOTIFY_RUNTIME_HELPER";

    fn send_ready_to(endpoint: &str) -> anyhow::Result<()> {
        use nix::sys::socket::{
            sendto, socket, AddressFamily, MsgFlags, SockFlag, SockType, UnixAddr,
        };

        let name = endpoint
            .strip_prefix('@')
            .expect("launcher notification endpoint must be abstract");
        let fd = socket(
            AddressFamily::Unix,
            SockType::Datagram,
            SockFlag::SOCK_NONBLOCK | SockFlag::SOCK_CLOEXEC,
            None,
        )?;
        sendto(
            fd.as_raw_fd(),
            b"READY=1",
            &UnixAddr::new_abstract(name.as_bytes())?,
            MsgFlags::MSG_DONTWAIT,
        )?;
        Ok(())
    }

    /// Helper child: reports readiness through the real sd_notify send path,
    /// then stays alive so the supervisor observes a genuinely RUNNING ready
    /// child; the supervisor's stop is the controlled shutdown.
    #[test]
    fn notify_ready_helper_child() {
        if std::env::var_os(READY_HELPER).is_none() {
            return;
        }
        crate::notify::ready().expect("notify ready send");
        std::thread::sleep(Duration::from_secs(30));
    }

    /// Helper child: records its PID to a marker file, then stays PENDING
    /// (no READY, launch lock held) until a release file appears; only then
    /// does it report readiness and stay alive. Lets a test hold a REAL
    /// notified launch mid-flight across the readiness await.
    #[test]
    fn launch_pending_helper_child() {
        if std::env::var_os(PENDING_HELPER).is_none() {
            return;
        }
        let marker = std::path::PathBuf::from(std::env::var(PENDING_MARKER).expect("marker"));
        let release = std::path::PathBuf::from(std::env::var(PENDING_RELEASE).expect("release"));
        std::fs::write(&marker, std::process::id().to_string()).expect("marker write");
        let deadline = Instant::now() + Duration::from_secs(30);
        while !release.exists() {
            assert!(Instant::now() < deadline, "release never arrived");
            std::thread::sleep(Duration::from_millis(20));
        }
        crate::notify::ready().expect("notify ready send");
        std::thread::sleep(Duration::from_secs(30));
    }

    /// Helper child: creates a publication obstacle (a directory at the
    /// future PID artifact path) BEFORE reporting readiness, so a test can
    /// exercise a post-precheck publication failure deterministically.
    #[test]
    fn publication_obstacle_helper_child() {
        if std::env::var_os(OBSTACLE_HELPER).is_none() {
            return;
        }
        let obstacle = std::path::PathBuf::from(std::env::var(OBSTACLE_PATH).expect("obstacle"));
        std::fs::create_dir(&obstacle).expect("obstacle dir");
        crate::notify::ready().expect("notify ready send");
        std::thread::sleep(Duration::from_secs(30));
    }

    /// Assert a process is reaped (`kill(pid, 0)` says ESRCH).
    fn assert_pid_gone(pid: u32) {
        let gone = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            None,
        )
        .is_err_and(|e| e == nix::errno::Errno::ESRCH);
        assert!(gone, "child pid {pid} must be reaped after cleanup");
    }

    /// Assert that lifecycle supervision created no legacy pathname endpoint.
    /// Abstract endpoints have no filesystem inode and disappear when their
    /// owned descriptor closes.
    fn assert_no_notify_inode(name: &str) {
        let dir = hyprstream_rpc::paths::runtime_dir();
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|file| file.starts_with(&format!("notify-{name}-")))
            .collect();
        assert!(
            leftovers.is_empty(),
            "notification endpoints must be cleaned up, found {leftovers:?}"
        );
    }

    /// Cancellation: a pending notified spawn aborted after the child PID is
    /// learnable must leave the child reaped and no artifacts behind.
    #[tokio::test]
    async fn cancelled_pending_spawn_reaps_child_and_cleans_artifacts() -> Result<()> {
        let runtime = hyprstream_rpc::paths::runtime_dir();
        std::fs::create_dir_all(&runtime).ok();
        let observed_pid_file = runtime.join("notify-cancel-observed.pid");
        let _ = std::fs::remove_file(&observed_pid_file);

        let backend = Arc::new(StandaloneBackend::new());
        let config = ProcessConfig::new("notify-cancel", Path::new("/bin/sh"))
            .args([
                "-c",
                "echo $$ > \"$OBSERVED_PID_FILE\"; exec sleep 100",
            ])
            .env("OBSERVED_PID_FILE", observed_pid_file.display().to_string())
            .with_notify_ready(Duration::from_secs(30));

        // Abort the pending handshake once the child has recorded its PID.
        let task = tokio::spawn({
            let backend = Arc::clone(&backend);
            async move { SpawnerBackend::spawn(&*backend, config).await }
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        let observed: u32 = loop {
            if let Ok(content) = std::fs::read_to_string(&observed_pid_file) {
                if let Ok(pid) = content.trim().parse::<u32>() {
                    break pid;
                }
            }
            assert!(
                Instant::now() < deadline,
                "child never recorded its PID for cancellation"
            );
            // Yield to the runtime: the pending spawn runs as a task on this
            // same (current-thread) test runtime and must be polled.
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        task.abort();

        // The ownership backstop must SIGKILL + reap the child and remove the
        // notification endpoint; poll with a bounded budget.
        let gone_deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let gone = nix::sys::signal::kill(nix::unistd::Pid::from_raw(observed as i32), None)
                .is_err_and(|e| e == nix::errno::Errno::ESRCH);
            if gone {
                break;
            }
            assert!(
                Instant::now() < gone_deadline,
                "cancelled child {observed} must be reaped"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            !hyprstream_rpc::paths::service_pid_file("notify-cancel").exists(),
            "cancelled spawn must leave no PID file"
        );
        assert_no_notify_inode("notify-cancel");
        let _ = std::fs::remove_file(&observed_pid_file);
        Ok(())
    }

    /// Publication rollback: an unwritable PID artifact (directory squatting
    /// the publication path) after READY must fail the spawn, reap the ready
    /// child, and leave no notification endpoint.
    #[tokio::test]
    async fn ambiguous_artifact_refuses_launch_before_spawn() -> Result<()> {
        let unique = std::process::id();
        let name = format!("notify-amb-{unique}");
        let backend = StandaloneBackend::new();
        // A DIRECTORY at the PID artifact path is a read ERROR, not an
        // absence: the witness is ambiguous and the launch must fail closed
        // BEFORE any child runs — no endpoint, no child, no pid in the error.
        let squat = hyprstream_rpc::paths::service_pid_file(&name);
        std::fs::create_dir_all(&squat).expect("squat directory");
        let config = ProcessConfig::new(&name, Path::new("/bin/true"))
            .with_notify_ready(Duration::from_secs(5));

        let error = backend
            .spawn(config)
            .await
            .expect_err("an ambiguous PID artifact must fail the launch closed");
        assert!(
            error.to_string().contains("ambiguous"),
            "ambiguous refusal expected, got: {error}"
        );
        assert!(
            !error.to_string().contains("(pid "),
            "a refused precheck must not have spawned a child, got: {error}"
        );
        assert_no_notify_inode(&name);
        std::fs::remove_dir(&squat).ok();
        Ok(())
    }

    /// Post-precheck publication failure: the controlled child creates the
    /// publication obstacle (a directory at the future PID artifact path)
    /// only AFTER the duplicate precheck has passed, then reports READY — so
    /// the rename onto the obstacle fails deterministically after READY and
    /// the ready child is rolled back (reaped, no artifacts).
    #[tokio::test]
    async fn publication_failure_after_precheck_rolls_back_ready_child() -> Result<()> {
        let unique = std::process::id();
        let name = format!("notify-pubfail-{unique}");
        let pid_file = hyprstream_rpc::paths::service_pid_file(&name);
        if let Some(parent) = pid_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let backend = StandaloneBackend::new();
        let config = ProcessConfig::new(&name, test_binary())
            .args([
                "--exact",
                "service::spawner::standalone::notify_readiness_tests::publication_obstacle_helper_child",
                "--nocapture",
            ])
            .env(OBSTACLE_HELPER, "1")
            .env(OBSTACLE_PATH, pid_file.display().to_string())
            .with_notify_ready(Duration::from_secs(30));

        let error = backend
            .spawn(config)
            .await
            .expect_err("a post-precheck obstacle must fail publication");
        assert!(
            error.to_string().contains("PID publication failed"),
            "publication failure expected, got: {error}"
        );
        assert_pid_gone(failure_pid(&error, &name));
        assert_no_notify_inode(&name);
        std::fs::remove_dir(&pid_file).ok();
        Ok(())
    }

    /// Extract the observed child PID the backend embeds in spawn failures
    /// ("service <name> (pid <N>) ...").
    fn failure_pid(error: &RpcError, name: &str) -> u32 {
        let text = error.to_string();
        let marker = format!("service {name} (pid ");
        let start = text
            .find(&marker)
            .unwrap_or_else(|| panic!("failure must carry the observed pid: {text}"))
            + marker.len();
        let digits: String = text[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().unwrap_or_else(|_| panic!("pid digits: {text}"))
    }

    fn test_binary() -> std::path::PathBuf {
        std::env::current_exe().expect("test binary path")
    }

    #[tokio::test]
    async fn notified_spawn_succeeds_on_real_child_ready() -> Result<()> {
        if std::env::var_os(READY_HELPER).is_some() {
            return Ok(());
        }
        let backend = StandaloneBackend::new();
        let config = ProcessConfig::new("notify-ready-control", test_binary())
            .args([
                "--exact",
                "service::spawner::standalone::notify_readiness_tests::notify_ready_helper_child",
                "--nocapture",
            ])
            .env(READY_HELPER, "1")
            .with_notify_ready(Duration::from_secs(30));
        let process = backend.spawn(config).await?;
        assert!(process.is_direct());
        let pid = process.pid().expect("direct child pid");
        assert!(
            hyprstream_rpc::paths::service_pid_file("notify-ready-control").exists(),
            "a ready child publishes its PID file"
        );
        assert!(
            backend.is_running(&process).await?,
            "the ready child must still be alive after the spawn reports success"
        );
        // Supervised stop: reaps the child and removes its PID artifact.
        backend.stop(&process).await?;
        assert!(
            !backend.is_running(&process).await?,
            "stopped child must not be running"
        );
        assert!(
            !hyprstream_rpc::paths::service_pid_file("notify-ready-control").exists(),
            "stop must remove the published PID file"
        );
        assert_pid_gone(pid);
        assert_no_notify_inode("notify-ready-control");
        Ok(())
    }

    /// A valid long instance/runtime namespace must not influence the compact
    /// abstract notification address. Run this scenario in a subprocess so
    /// its process-global path environment cannot race other libtest cases.
    #[test]
    fn notified_spawn_with_long_runtime_reaches_ready_stops_and_reaps() -> Result<()> {
        if std::env::var_os(LONG_RUNTIME_HELPER).is_some() {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            return runtime.block_on(async {
                let name = "notify-long-runtime";
                let backend = StandaloneBackend::new();
                let config = ProcessConfig::new(name, test_binary())
                    .args([
                        "--exact",
                        "service::spawner::standalone::notify_readiness_tests::notify_ready_helper_child",
                        "--nocapture",
                    ])
                    .env(READY_HELPER, "1")
                    .with_notify_ready(Duration::from_secs(30));
                let process = backend.spawn(config).await?;
                let pid = process.pid().expect("direct child pid");
                assert!(backend.is_running(&process).await?);
                backend.stop(&process).await?;
                assert!(!backend.is_running(&process).await?);
                assert_pid_gone(pid);
                assert!(
                    !hyprstream_rpc::paths::service_pid_file(name).exists(),
                    "stopped child must leave no PID artifact"
                );
                assert_no_notify_inode(name);
                Ok(())
            });
        }

        let runtime = tempfile::tempdir()?;
        let status = std::process::Command::new(test_binary())
            .args([
                "--exact",
                "service::spawner::standalone::notify_readiness_tests::notified_spawn_with_long_runtime_reaches_ready_stops_and_reaps",
                "--nocapture",
            ])
            .env(LONG_RUNTIME_HELPER, "1")
            .env("XDG_RUNTIME_DIR", runtime.path())
            .env(
                "HYPRSTREAM_INSTANCE",
                "production-west-instance-0000000000000000",
            )
            .status()?;
        assert!(status.success(), "isolated long-runtime scenario failed: {status}");
        Ok(())
    }

    /// A child that exits on its own (no stop issued) must be reported as
    /// not-running: it is a zombie — still visible to `kill(pid, 0)` — until
    /// its supervisor reaps it, and "running" would be a lie. A subsequent
    /// supervised stop must still succeed (reap + conditional PID removal).
    #[tokio::test]
    async fn is_running_reports_self_exited_child_as_stopped() -> Result<()> {
        let backend = StandaloneBackend::new();
        let config = ProcessConfig::new("notify-self-exit", Path::new("/bin/sh"))
            .args(["-c", "exit 0"]);
        let process = SpawnerBackend::spawn(&backend, config).await?;
        let pid = process.pid().expect("direct child pid");

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if !backend.is_running(&process).await? {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "self-exited child must not be reported as running (zombie)"
            );
            // Yield: spawn runs as a task on this same current-thread runtime.
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // Stopping an already-exited child is still a clean, confirmed stop.
        backend.stop(&process).await?;
        assert!(!backend.is_running(&process).await?);
        assert_pid_gone(pid);

        // A SECOND stop after the confirmed first is a clean no-op through
        // the untracked-by-PID fallback: termination is observed (ESRCH),
        // not assumed.
        backend.stop(&process).await?;
        assert!(!backend.is_running(&process).await?);
        Ok(())
    }

    /// Cleanup failures are part of the error report, never swallowed
    /// (the discard-the-Err `unwrap_or_default` suffixes are gone).
    #[test]
    fn cleanup_failure_suffix_propagates_into_error() {
        assert_eq!(cleanup_suffix(Ok(())), "");
        assert_eq!(
            cleanup_suffix(Err("kill failed: boom".to_owned())),
            "; cleanup: kill failed: boom"
        );
    }

    /// Explicit-unlock release (P2): `flock` lives on the open file
    /// description, so a close-only release can be prolonged by any retained
    /// duplicate of that description — exactly what a concurrently forked
    /// child's pre-exec descriptor copy is. A safe `dup` of the locked
    /// descriptor stands in for such a duplicate; after the lock is dropped,
    /// a fresh same-service acquisition must succeed even though the
    /// duplicate is still open.
    #[test]
    fn launch_lock_releases_despite_retained_description_duplicate() {
        let unique = std::process::id();
        let name = format!("notify-unlock-{unique}");

        let held = ServiceLaunchLock::acquire(&name).expect("acquire");
        // Retained duplicate of the SAME open file description (RAII clone:
        // an assertion failure cannot leak the demonstration descriptor).
        let duplicate = held._file.try_clone().expect("try_clone");

        // Explicit-unlock drop: the lock must release despite the still-open
        // duplicate; close alone would not achieve that.
        drop(held);

        let reacquired = ServiceLaunchLock::acquire(&name).expect(
            "lock must release via explicit unlock despite the retained duplicate",
        );
        drop(reacquired);

        // The demonstration duplicate is closed by its own RAII drop here.
        drop(duplicate);
    }

    /// Startup-backstop decisions: OBSERVED termination cleans the owned
    /// artifact (regular file and already-absent paths); an unobservable
    /// outcome must RETAIN the artifact as evidence and claim no success.
    /// The unobserved branch is exercised through the deterministically
    /// inducible seam (a handle that yields no PID) — no kernel kill
    /// failure is forced or claimed.
    #[tokio::test]
    async fn startup_backstop_cleanup_tracks_observed_termination() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let artifact = dir.path().join("owned.pid");

        // Observed path: a live child is SIGKILLed and reaped by the
        // backstop; the owned artifact is removed.
        let mut cmd = tokio::process::Command::new("/bin/sleep");
        cmd.arg("30");
        let child = cmd.spawn().expect("sleep child");
        let pid = child.id().expect("live pid");
        std::fs::write(&artifact, pid.to_string()).expect("artifact");
        let mut guard = StartupChildGuard {
            child: Some(child),
            name: "backstop-observed".to_owned(),
            pid_file: Some(artifact.clone()),
        };
        assert_eq!(
            guard.backstop_cleanup(),
            BackstopOutcome::TerminatedAndCleaned,
            "live-child backstop must observe termination and clean its artifact"
        );
        assert!(
            !artifact.exists(),
            "observed cleanup must remove the owned artifact"
        );
        assert_pid_gone(pid);

        // Already-absent path: an armed artifact that does not exist is
        // gone either way — still a clean, observed outcome.
        let mut cmd = tokio::process::Command::new("/bin/sleep");
        cmd.arg("30");
        let child = cmd.spawn().expect("sleep child");
        let mut guard = StartupChildGuard {
            child: Some(child),
            name: "backstop-absent".to_owned(),
            pid_file: Some(dir.path().join("never-written.pid")),
        };
        assert_eq!(
            guard.backstop_cleanup(),
            BackstopOutcome::TerminatedAndCleaned,
            "a NotFound artifact must not downgrade the observed outcome"
        );
        assert!(!guard.pid_file.is_some(), "armed path must be cleared");

        // Unobserved path: the handle yields no PID, so the backstop cannot
        // observe anything and must not claim cleanup.
        let mut exiting = tokio::process::Command::new("/bin/sh");
        exiting.args(["-c", "exit 0"]);
        let mut child = exiting.spawn().expect("exiting child");
        child.wait().await.expect("reaped by this wait");
        std::fs::write(&artifact, "0").expect("artifact");
        let mut guard = StartupChildGuard {
            child: Some(child),
            name: "backstop-unobserved".to_owned(),
            pid_file: Some(artifact.clone()),
        };
        assert_eq!(
            guard.backstop_cleanup(),
            BackstopOutcome::Unconfirmed,
            "an unobservable outcome must not claim cleanup"
        );
        assert!(
            artifact.exists(),
            "unobserved backstop must retain the owned artifact as evidence"
        );
        Ok(())
    }

    /// Termination and artifact cleanup are separate observed facts: a
    /// live child is terminated and reaped, but an artifact that cannot be
    /// unlinked (a directory squats the armed path) must be reported as
    /// `TerminatedArtifactKept` and RETAINED on disk — never as a
    /// cleaned-up success.
    #[tokio::test]
    async fn startup_backstop_distinguishes_termination_from_failed_cleanup() -> Result<()> {
        let dir = tempfile::tempdir()?;
        // A DIRECTORY at the armed path: remove_file fails with EISDIR
        // (not NotFound) even after the child is genuinely reaped.
        let squat = dir.path().join("owned.pid");
        std::fs::create_dir(&squat)?;

        let mut cmd = tokio::process::Command::new("/bin/sleep");
        cmd.arg("30");
        let child = cmd.spawn().expect("sleep child");
        let pid = child.id().expect("live pid");
        let mut guard = StartupChildGuard {
            child: Some(child),
            name: "backstop-kept".to_owned(),
            pid_file: Some(squat.clone()),
        };
        assert_eq!(
            guard.backstop_cleanup(),
            BackstopOutcome::TerminatedArtifactKept,
            "failed artifact removal must be reported separately from termination"
        );
        // Termination was observed (reaped) AND the artifact is retained.
        assert_pid_gone(pid);
        assert!(
            squat.is_dir(),
            "the failed-removal artifact must be retained on disk as evidence"
        );
        std::fs::remove_dir(&squat)?;
        Ok(())
    }

    /// Narrow classification-helper coverage, honestly labelled: the real
    /// OS cannot deterministically present a nonterminal wait status under
    /// WNOHANG after SIGKILL, so this proves only that the classifier
    /// accepts exactly Exited/Signaled and rejects still-alive, stopped,
    /// and continued states.
    #[test]
    fn wait_status_classification_accepts_only_terminal_statuses() {
        use nix::sys::signal::Signal;
        use nix::sys::wait::WaitStatus;

        let pid = nix::unistd::Pid::from_raw(1);
        assert!(wait_status_is_terminal(&WaitStatus::Exited(pid, 0)));
        assert!(wait_status_is_terminal(&WaitStatus::Signaled(
            pid,
            Signal::SIGKILL,
            false
        )));
        assert!(!wait_status_is_terminal(&WaitStatus::StillAlive));
        assert!(!wait_status_is_terminal(&WaitStatus::Stopped(
            pid,
            Signal::SIGSTOP
        )));
        assert!(!wait_status_is_terminal(&WaitStatus::Continued(pid)));
    }

    /// UNIT-LEVEL duplicate controls (manual lock hold / manual witness; the
    /// END-TO-END pending→READY→live sequencing is proven by
    /// `duplicate_launch_sequencing_refuses_pending_then_live`): a held
    /// launch lock deterministically reports the pending duplicate, and a
    /// PID artifact naming a LIVE process refuses a new launch — never
    /// silently overwriting the sole witness of a live process.
    #[tokio::test]
    async fn duplicate_launch_controls_refuse_pending_and_live() -> Result<()> {
        let unique = std::process::id();
        let name = format!("notify-dup-{unique}");
        let backend = StandaloneBackend::new();

        // Pending duplicate: the launch lock itself is the deterministic
        // witness — a held lock refuses the attempt before anything runs.
        let held = ServiceLaunchLock::acquire(&name)?;
        let pending = ProcessConfig::new(&name, Path::new("/bin/true"))
            .with_notify_ready(Duration::from_secs(5));
        let pending_error = SpawnerBackend::spawn(&backend, pending)
            .await
            .expect_err("a held launch lock must refuse a pending duplicate");
        assert!(
            pending_error.to_string().contains("already in progress"),
            "pending-duplicate refusal expected, got: {pending_error}"
        );
        assert_no_notify_inode(&name);
        drop(held);

        // Live duplicate: a PID artifact naming a live process refuses a
        // second launch. The squat process is spawned (and later reaped) by
        // this test itself.
        let mut squat = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("squat process for live-duplicate witness");
        let squat_pid = squat.id();
        let pid_file = hyprstream_rpc::paths::service_pid_file(&name);
        if let Some(parent) = pid_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&pid_file, squat_pid.to_string()).expect("squat pid file");

        let ready_config = ProcessConfig::new(&name, test_binary())
            .args([
                "--exact",
                "service::spawner::standalone::notify_readiness_tests::notify_ready_helper_child",
                "--nocapture",
            ])
            .env(READY_HELPER, "1")
            .with_notify_ready(Duration::from_secs(30));
        let live_error = SpawnerBackend::spawn(&backend, ready_config)
            .await
            .expect_err("a live PID artifact must refuse a duplicate launch");
        assert!(
            live_error
                .to_string()
                .contains(&format!("already appears live as pid {squat_pid}")),
            "live-duplicate refusal expected, got: {live_error}"
        );
        // The refused attempt published nothing: the artifact still names
        // the live squat, and no notification endpoint survives.
        assert_eq!(
            std::fs::read_to_string(&pid_file)
                .expect("witness pid file")
                .trim(),
            squat_pid.to_string(),
            "live witness artifact must not be overwritten"
        );
        assert_no_notify_inode(&name);

        // Reap the squat process and leave no fixture artifact behind.
        squat.kill().expect("squat kill");
        squat.wait().expect("squat reap");
        let _ = std::fs::remove_file(&pid_file);
        Ok(())
    }

    /// REAL duplicate sequencing, end to end: a first notified launch held
    /// PENDING by its own child keeps the per-service launch lock held
    /// ACROSS the readiness await, so a second same-name launch is refused
    /// with no child of its own while the first survives; after release the
    /// first reports READY, publishes its PID artifact, and stays alive —
    /// and a further launch is refused against that LIVE published witness.
    #[tokio::test]
    async fn duplicate_launch_sequencing_refuses_pending_then_live() -> Result<()> {
        let backend = Arc::new(StandaloneBackend::new());
        let dir = tempfile::tempdir()?;
        let unique = std::process::id();
        let name = format!("notify-seq-{unique}");
        let marker = dir.path().join("pending.marker");
        let release = dir.path().join("pending.release");

        let pending_config = ProcessConfig::new(&name, test_binary())
            .args([
                "--exact",
                "service::spawner::standalone::notify_readiness_tests::launch_pending_helper_child",
                "--nocapture",
            ])
            .env(PENDING_HELPER, "1")
            .env(PENDING_MARKER, marker.display().to_string())
            .env(PENDING_RELEASE, release.display().to_string())
            .with_notify_ready(Duration::from_secs(30));

        // First launch: genuinely pending inside the readiness await, with
        // the launch lock held by this very launch.
        let first = tokio::spawn({
            let backend = Arc::clone(&backend);
            let config = pending_config.clone();
            async move { SpawnerBackend::spawn(&*backend, config).await }
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        let first_pid: u32 = loop {
            if let Ok(content) = std::fs::read_to_string(&marker) {
                if let Ok(pid) = content.trim().parse::<u32>() {
                    break pid;
                }
            }
            assert!(
                Instant::now() < deadline,
                "first launch never went pending (no marker)"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };

        // Second same-name launch while the first is mid-flight: refused on
        // the held lock, deterministically, before any child of its own.
        let second_error = SpawnerBackend::spawn(&*backend, pending_config.clone())
            .await
            .expect_err("a pending first launch must refuse a same-name launch");
        assert!(
            second_error.to_string().contains("already in progress"),
            "pending-duplicate refusal expected, got: {second_error}"
        );
        // The first process survived the rejected second call.
        assert!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(first_pid as i32), None).is_ok(),
            "first launch child must survive the refused duplicate"
        );

        // Release the first: it reports READY and publishes its PID artifact.
        std::fs::write(&release, b"release")?;
        let process = tokio::time::timeout(Duration::from_secs(30), first)
            .await
            .expect("first launch task must finish after release")
            .expect("first launch task must not panic")
            .expect("first launch spawn must succeed after release");
        assert_eq!(
            std::fs::read_to_string(hyprstream_rpc::paths::service_pid_file(&name))
                .expect("published pid file")
                .trim(),
            first_pid.to_string(),
            "the first launch must publish its own pid"
        );

        // A further launch is refused against the LIVE published witness.
        let third_error = SpawnerBackend::spawn(&*backend, pending_config.clone())
            .await
            .expect_err("a live published predecessor must refuse a duplicate");
        assert!(
            third_error
                .to_string()
                .contains(&format!("already appears live as pid {first_pid}")),
            "live-duplicate refusal expected, got: {third_error}"
        );

        // Supervised cleanup of the first; the persistent lock file survives
        // (stable inode, never unlinked) while the PID artifact is gone.
        backend.stop(&process).await?;
        assert_pid_gone(first_pid);
        assert!(
            !hyprstream_rpc::paths::service_pid_file(&name).exists(),
            "stopped first launch must leave no PID file"
        );
        assert!(
            hyprstream_rpc::paths::runtime_dir()
                .join(format!("{name}.launch.lock"))
                .exists(),
            "the launch lock file must persist (never unlinked)"
        );
        assert_no_notify_inode(&name);
        Ok(())
    }

    #[tokio::test]
    async fn notified_spawn_fails_on_early_child_exit() -> Result<()> {
        let backend = StandaloneBackend::new();
        let config = ProcessConfig::new("notify-early-exit", Path::new("/bin/sh"))
            .args(["-c", "exit 7"])
            .with_notify_ready(Duration::from_secs(30));
        let error = backend
            .spawn(config)
            .await
            .expect_err("early child exit must fail the spawn");
        assert!(
            error.to_string().contains("exited during startup"),
            "honest exit propagation expected, got: {error}"
        );
        assert!(
            !hyprstream_rpc::paths::service_pid_file("notify-early-exit").exists(),
            "a never-ready child must not publish a PID file"
        );
        assert_pid_gone(failure_pid(&error, "notify-early-exit"));
        assert_no_notify_inode("notify-early-exit");
        Ok(())
    }

    #[tokio::test]
    async fn notified_spawn_times_out_kills_and_cleans_up() -> Result<()> {
        let backend = StandaloneBackend::new();
        let config = ProcessConfig::new("notify-timeout", Path::new("/bin/sleep"))
            .args(["100"])
            .with_notify_ready(Duration::from_millis(300));
        let error = backend
            .spawn(config)
            .await
            .expect_err("a hung child must hit the hard readiness timeout");
        assert!(
            error.to_string().contains("no READY=1"),
            "timeout failure expected, got: {error}"
        );
        // Cleanup: the observed child was terminated and reaped, and no PID or
        // notification artifact survives.
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            !hyprstream_rpc::paths::service_pid_file("notify-timeout").exists(),
            "timed-out child must leave no PID artifact"
        );
        assert_pid_gone(failure_pid(&error, "notify-timeout"));
        assert_no_notify_inode("notify-timeout");
        Ok(())
    }

    #[test]
    fn notification_endpoint_is_compact_unique_and_abstract() -> Result<()> {
        let first = ChildNotifySocket::bind("a-service-name-that-is-deliberately-ignored")?;
        let second = ChildNotifySocket::bind("a-service-name-that-is-deliberately-ignored")?;
        assert!(first.endpoint().starts_with('@'));
        assert!(second.endpoint().starts_with('@'));
        assert_ne!(first.endpoint(), second.endpoint());
        assert!(first.endpoint().len() <= 48);
        Ok(())
    }

    #[test]
    fn notification_endpoint_collision_retries_are_bounded() -> anyhow::Result<()> {
        use nix::sys::socket::{
            bind, socket, AddressFamily, SockFlag, SockType, UnixAddr,
        };

        let occupied_name = format!("hypr-n-collision-{}", std::process::id());
        let occupied = socket(
            AddressFamily::Unix,
            SockType::Datagram,
            SockFlag::SOCK_CLOEXEC,
            None,
        )?;
        bind(
            occupied.as_raw_fd(),
            &UnixAddr::new_abstract(occupied_name.as_bytes())?,
        )?;

        let mut attempts = 0usize;
        let error = ChildNotifySocket::bind_with_name_source(|| {
            attempts += 1;
            occupied_name.clone()
        })
        .expect_err("eight collisions must fail closed");
        assert_eq!(attempts, 8);
        assert!(error.to_string().contains("after 8 attempts"));
        Ok(())
    }

    #[tokio::test]
    async fn foreign_datagram_flood_cannot_extend_readiness_deadline() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};

        let backend = StandaloneBackend::new();
        let notify = ChildNotifySocket::bind("flood-deadline")?;
        let mut child = tokio::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()?;
        let endpoint = notify.endpoint().to_owned();
        let flooding = Arc::new(AtomicBool::new(true));
        let sender = std::thread::spawn({
            let flooding = Arc::clone(&flooding);
            move || {
                while flooding.load(Ordering::Relaxed) {
                    let _ = send_ready_to(&endpoint);
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        });

        let started = Instant::now();
        let outcome = backend
            .await_child_readiness(&mut child, &notify, Duration::from_millis(200))
            .await;
        flooding.store(false, Ordering::Relaxed);
        sender.join().expect("flood sender thread");
        child.kill().await?;
        child.wait().await?;

        let error = outcome.expect_err("foreign datagrams must not satisfy readiness");
        assert!(error.contains("no READY=1"));
        assert!(started.elapsed() < Duration::from_secs(2));
        Ok(())
    }

    /// Sender-PID matching: a READY datagram from any process other than the
    /// expected child is ignored, not honored.
    #[tokio::test]
    async fn notify_socket_rejects_spoofed_sender() -> anyhow::Result<()> {
        let notify = ChildNotifySocket::bind("spoof-probe")?;
        let this_pid = std::process::id() as i32;
        // A sustained foreign queue must be drained in one poll so it cannot
        // hide the genuine sender's datagram behind the 20ms readiness tick.
        let endpoint = notify.endpoint().to_owned();
        let flooding = std::thread::spawn(move || {
            for _ in 0..128 {
                let _ = send_ready_to(&endpoint);
            }
        });
        flooding.join().expect("spoof flood sender thread");
        send_ready_to(notify.endpoint())?;
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            notify.try_recv_ready(this_pid)?,
            "the matching sender satisfies readiness"
        );
        Ok(())
    }
}

/// Non-Linux targets only: the supervised Notify request must be refused
/// before any launch side effect. The configured child is an executable
/// sentinel — it would write a marker file if it were ever spawned — so
/// marker, PID-artifact, and notify-runtime-directory absence at the checked
/// points evidence the boundary. This cfg cannot run on Linux lanes; it
/// exists for actual non-Linux targets and makes no claim about them from
/// Linux runs.
#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android")),
    test
))]
mod notify_refusal_non_linux_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crate::service::spawner::ProcessSpawner;

    const REFUSED_SENTINEL: &str = "HYPRSTREAM_NOTIFY_REFUSED_SENTINEL";

    /// A Notify request on an unsupported platform is refused pre-spawn by
    /// the explicit unsupported branch in `spawn_notified` — that branch
    /// returns before any launch operation, which is the production
    /// guarantee. This test observes that boundary: the executable sentinel
    /// was not observed running, no PID artifact was published, and no
    /// per-launch notify runtime directory appeared at the checked points.
    #[tokio::test]
    async fn notify_request_refused_before_any_launch_side_effect() -> anyhow::Result<()> {
        if let Some(path) = std::env::var_os(REFUSED_SENTINEL) {
            // Unreachable if the boundary holds: the refusal must fire before
            // any child exists.
            std::fs::write(&path, std::process::id().to_string())?;
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let dir = tempfile::tempdir()?;
        let marker = dir.path().join("sentinel.marker");
        // Unique per-run service name: unrelated runtime artifacts cannot
        // decide the result, and nothing existing is overwritten or removed.
        let name = format!("notify-refused-{}", nix::unistd::getpid());
        let pid_file = hyprstream_rpc::paths::service_pid_file(&name);
        let dir_prefix = format!("notify-{name}-");

        let spawner = ProcessSpawner::standalone();
        let error = match spawner
            .spawn(
                ProcessConfig::new(&name, &exe)
                    .args([
                        "--exact",
                        "service::spawner::standalone::notify_refusal_non_linux_tests::notify_request_refused_before_any_launch_side_effect",
                        "--nocapture",
                    ])
                    .env(REFUSED_SENTINEL, marker.display().to_string())
                    .with_notify_ready(std::time::Duration::from_secs(30)),
            )
            .await
        {
            Ok(process) => {
                panic!(
                    "Notify must be refused on non-Linux targets; spawned pid {:?}",
                    process.pid()
                );
            }
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("Linux/Android"),
            "refusal must name the platform boundary, got: {message}"
        );

        // Observed boundary: the sentinel never ran, no PID artifact was
        // published, and no per-launch notify runtime directory exists. An
        // ABSENT runtime directory is itself a valid refusal state (nothing
        // created it): only ErrorKind::NotFound is treated as absence; every
        // other read/entry error propagates.
        assert!(
            !marker.exists(),
            "the sentinel child must not have been spawned by the refused Notify request"
        );
        assert!(
            !pid_file.exists(),
            "a refused Notify request must not publish a PID artifact"
        );
        match std::fs::read_dir(hyprstream_rpc::paths::runtime_dir()) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
            Ok(entries) => {
                for entry in entries {
                    let entry = entry?;
                    let file_name = entry.file_name().to_string_lossy().to_string();
                    assert!(
                        !file_name.starts_with(&dir_prefix),
                        "a refused Notify request must not create a notify runtime \
                         directory, found {file_name}"
                    );
                }
            }
        }
        Ok(())
    }
}
