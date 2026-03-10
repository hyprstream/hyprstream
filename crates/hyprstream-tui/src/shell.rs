//! PTY-backed shell spawner.  Native only (not WASI).

#![cfg(not(target_os = "wasi"))]

use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};

// ============================================================================
// Public types
// ============================================================================

pub enum ShellInput {
    Bytes(Vec<u8>),
    Resize { cols: u16, rows: u16 },
    Kill,
}

pub struct ShellWindow {
    /// ANSI bytes from the PTY — poll with try_recv() in tick().
    pub stdout_rx: Receiver<Vec<u8>>,
    /// Send input bytes or control signals to the shell.
    pub input_tx:  SyncSender<ShellInput>,
    /// OS PID for SIGTERM on close.
    pub pid:       u32,
}

// ============================================================================
// Spawner
// ============================================================================

pub fn spawn_shell(cwd: Option<PathBuf>, cols: u16, rows: u16) -> io::Result<ShellWindow> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_owned());

    let (master_fd, slave_fd) = open_pty(cols, rows)?;

    // Create all slave dups BEFORE wrapping in Stdio so we know their exact FD numbers.
    // These three FDs will be dup2'd to stdin=0, stdout=1, stderr=2 by Rust's Command
    // after the pre_exec callback runs.
    let slave_dup1 = dup_fd(slave_fd)?;
    let slave_dup2 = dup_fd(slave_fd)?;

    let slave_in  = unsafe { Stdio::from_raw_fd(slave_fd) };
    let slave_out = unsafe { Stdio::from_raw_fd(slave_dup1) };
    let slave_err = unsafe { Stdio::from_raw_fd(slave_dup2) };

    let mut cmd = Command::new(&shell);
    cmd.stdin(slave_in).stdout(slave_out).stderr(slave_err);
    cmd.env("TERM", "xterm-256color");
    if let Some(ref dir) = cwd {
        cmd.current_dir(dir);
    }

    // Pre-compute sandbox paths in the parent (before fork) so no heap allocation
    // occurs post-fork in the child process.  Both are Option<PathBuf> moved into the
    // pre_exec closure.  sandbox_cwd = None means no sandboxing (plain shell tab).
    //
    // Sandbox support: Linux (Landlock) and OpenBSD (unveil).
    // Other platforms leave cwd unused here; it was already consumed by current_dir above.
    #[cfg(any(target_os = "linux", target_os = "openbsd"))]
    let sandbox_cwd: Option<PathBuf> = cwd;
    #[cfg(any(target_os = "linux", target_os = "openbsd"))]
    let sandbox_home: Option<PathBuf> = std::env::var_os("HOME").map(PathBuf::from);
    #[cfg(not(any(target_os = "linux", target_os = "openbsd")))]
    let _ = cwd;

    // FDs the child must keep open for Rust's stdio dup2 setup (happens after pre_exec).
    let keep_fds = [slave_fd, slave_dup1, slave_dup2];

    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(move || {
            libc::setsid();
            libc::ioctl(slave_fd, libc::TIOCSCTTY, 0_i32);
            // Close all inherited FDs > 2 except the three PTY slave FDs.
            // Without this, the exec'd shell inherits ZMQ's internal socketpair FDs.
            // When the shell eventually closes them, the parent's ZMQ signaler receives
            // an unexpected EOF that triggers zmq_assert(dummy == 0) in recv_failable().
            let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
            let soft_limit = if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) == 0 {
                rl.rlim_cur as i32
            } else {
                1024
            };
            let max_fd = soft_limit.min(65536);
            for fd in 3..max_fd {
                if !keep_fds.contains(&fd) {
                    libc::close(fd);
                }
            }
            // Apply filesystem sandboxing for scoped (worktree) terminals.
            // Paths were pre-allocated in the parent; no heap allocation here.
            // Linux: Landlock MAC.  OpenBSD: unveil(2).
            #[cfg(any(target_os = "linux", target_os = "openbsd"))]
            if let Some(ref wt) = sandbox_cwd {
                apply_worktree_sandbox(wt, sandbox_home.as_deref());
            }
            Ok(())
        });
    }

    // Allocate write_fd before spawn so it gets a fresh FD number above the slave FDs.
    // This prevents write_fd from colliding with slave_fd after cmd's Drop closes slave_in.
    let write_fd = dup_fd(master_fd)?;
    let child = cmd.spawn()?;
    let pid = child.id();
    // slave_fd is owned by slave_in (passed into cmd); cmd's Drop closes it — no manual close.

    let mut reader = unsafe { std::fs::File::from_raw_fd(master_fd) };
    let mut writer = unsafe { std::fs::File::from_raw_fd(write_fd) };

    let (stdout_tx, stdout_rx) = mpsc::sync_channel::<Vec<u8>>(64);
    let (input_tx, input_rx)   = mpsc::sync_channel::<ShellInput>(64);

    // Reader: PTY master → stdout_tx
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if stdout_tx.send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Writer: input_rx → PTY master
    std::thread::spawn(move || {
        while let Ok(msg) = input_rx.recv() {
            match msg {
                ShellInput::Bytes(data) => {
                    if writer.write_all(&data).is_err() {
                        break;
                    }
                }
                ShellInput::Resize { cols, rows } => {
                    let ws = libc::winsize {
                        ws_row: rows,
                        ws_col: cols,
                        ws_xpixel: 0,
                        ws_ypixel: 0,
                    };
                    unsafe { libc::ioctl(writer.as_raw_fd(), libc::TIOCSWINSZ, &ws) };
                }
                ShellInput::Kill => break,
            }
        }
    });

    Ok(ShellWindow { stdout_rx, input_tx, pid })
}

pub fn kill_shell(pid: u32) {
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
}

// ============================================================================
// Helpers
// ============================================================================

fn open_pty(cols: u16, rows: u16) -> io::Result<(i32, i32)> {
    let ws = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
    let mut master: libc::c_int = -1;
    let mut slave:  libc::c_int = -1;
    let ret = unsafe {
        libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null(), &ws)
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((master, slave))
}

fn dup_fd(fd: i32) -> io::Result<i32> {
    let new = unsafe { libc::dup(fd) };
    if new < 0 { Err(io::Error::last_os_error()) } else { Ok(new) }
}

// ============================================================================
// Landlock sandbox (Linux only)
// ============================================================================

/// Restrict the current process's filesystem access to a model worktree.
///
/// Policy:
/// - System paths (`/usr`, `/lib*`, `/bin`, `/etc`, `/proc`, `/dev`, …) — read + execute
/// - `/tmp`           — read + execute only (no writes; avoids temp-file leakage)
/// - `$HOME`          — read + execute only (shell init files; no history writes)
/// - `worktree`       — full access (the model directory the terminal is scoped to)
/// - everything else  — blocked (including other model worktrees)
///
/// Best-effort: if the running kernel does not support Landlock (< 5.13) or the
/// process lacks permission, sandboxing is silently skipped.
///
/// # Safety
/// Called from `pre_exec` (post-fork, pre-exec child).  All `PathBuf` arguments
/// were allocated in the parent before the fork; no heap allocation occurs here.
#[cfg(target_os = "linux")]
fn apply_worktree_sandbox(worktree: &Path, home: Option<&Path>) {
    use landlock::{ABI, Access, AccessFs, Compatible, CompatLevel, Ruleset, RulesetAttr, RulesetCreatedAttr, path_beneath_rules};

    // ABI::V3 = Linux 5.19 (Truncate right).  The crate degrades gracefully on older ABIs.
    let abi = ABI::V3;

    // Read-only rights: open files + list directories + execute binaries.
    let read_exec = AccessFs::from_read(abi);
    // Full rights for this ABI level.
    let full = AccessFs::from_all(abi);

    // Standard system paths that a usable shell needs to read/execute.
    // path_beneath_rules() silently skips paths that cannot be opened (e.g. /nix on non-NixOS).
    let sys_paths: &[&Path] = &[
        Path::new("/usr"),
        Path::new("/lib"),
        Path::new("/lib64"),
        Path::new("/lib32"),
        Path::new("/libx32"),
        Path::new("/bin"),
        Path::new("/sbin"),
        Path::new("/etc"),
        Path::new("/proc"),
        Path::new("/dev"),
        Path::new("/run"),
        Path::new("/sys"),
        Path::new("/nix"),
        Path::new("/opt"),
        Path::new("/snap"),
    ];

    // Best-effort: if any step fails (unsupported kernel, EPERM, …) just return.
    let result = (|| {
        let created = Ruleset::default()
            .set_compatibility(CompatLevel::BestEffort)
            .handle_access(full)?
            .create()?;

        // System paths: read + execute only.
        let created = created.add_rules(path_beneath_rules(sys_paths, read_exec))?;

        // /tmp: read + execute only (avoid leaking temp files across sessions).
        let created = created.add_rules(path_beneath_rules(&[Path::new("/tmp")], read_exec))?;

        // $HOME: read + execute only (shell init files; bash_history not written in sandboxed sessions).
        let created = match home {
            Some(h) => created.add_rules(path_beneath_rules(&[h], read_exec))?,
            None => created,
        };

        // Worktree: full access — the user is here to work on this model.
        let created = created.add_rules(path_beneath_rules(&[worktree], full))?;

        created.restrict_self()
    })();

    // Silently discard errors (best-effort sandbox).
    let _ = result;
}

/// Restrict the current process's filesystem access to a model worktree (OpenBSD).
///
/// Uses `unveil(2)` which is the direct OpenBSD equivalent of Linux Landlock:
/// applied once pre-exec and inherited by the shell and all its children.
///
/// Policy mirrors the Linux implementation:
/// - System paths — read + execute
/// - `$HOME`       — read only (shell init files)
/// - `worktree`    — full access
/// - everything else — blocked (including other model worktrees)
///
/// `pledge(2)` is intentionally NOT called here: the spawned shell will set its
/// own pledges, and pre-pledging would prevent the shell from expanding to the
/// syscall set it needs for interactive use.
///
/// # Safety
/// Called from `pre_exec` (post-fork, pre-exec child).  All `PathBuf` arguments
/// were allocated in the parent before the fork; no heap allocation occurs here.
#[cfg(target_os = "openbsd")]
fn apply_worktree_sandbox(worktree: &Path, home: Option<&Path>) {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    // Open a path with the given unveil permissions.  Errors are silently ignored.
    let do_unveil = |path: &Path, perms: &[u8]| {
        let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else { return };
        unsafe { libc::unveil(c_path.as_ptr(), perms.as_ptr() as *const libc::c_char) };
    };

    // Standard system paths the shell needs to read/execute.
    // /proc is not mounted by default on OpenBSD so is omitted.
    let sys_paths: &[&str] = &[
        "/usr",
        "/lib",
        "/libexec",     // OpenBSD: ld.so lives here
        "/bin",
        "/sbin",
        "/etc",
        "/dev",
    ];
    for &p in sys_paths {
        do_unveil(Path::new(p), b"rx\0");
    }

    // /tmp: read + write + create (for shell temp files; no execute needed).
    do_unveil(Path::new("/tmp"), b"rwc\0");

    // $HOME: read only (shell init files; history not written in sandboxed sessions).
    if let Some(h) = home {
        do_unveil(h, b"r\0");
    }

    // Worktree: full access — read, write, execute, create/remove.
    do_unveil(worktree, b"rwxc\0");

    // Lock: calling unveil(NULL, NULL) prevents any further unveil calls and
    // blocks access to all paths not already unveiled.
    unsafe { libc::unveil(std::ptr::null(), std::ptr::null()) };
}
