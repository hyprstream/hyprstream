//! PTY-backed shell spawner.  Native only (not WASI).

#![cfg(not(target_os = "wasi"))]

use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::path::PathBuf;
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
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }

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
