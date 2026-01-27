//! Daemonization support for running hyprstream in the background
//!
//! Provides helper functions for detaching the process from the terminal
//! and running as a background daemon.
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream::cli::daemon::maybe_daemonize;
//!
//! // In server startup, before creating tokio runtime:
//! maybe_daemonize(detach, working_dir, pid_file)?;
//! ```

use anyhow::{Context, Result};
use std::path::PathBuf;
use tracing::info;

/// Default PID file location
///
/// Delegates to hyprstream_workers::paths for consistent path resolution.
fn default_pid_file() -> PathBuf {
    hyprstream_workers::paths::pid_file()
}

/// Daemonize the process if requested
///
/// This function should be called early in main(), before creating the tokio runtime.
/// It uses the `daemonize` crate to properly fork and detach from the terminal.
///
/// # Arguments
///
/// * `detach` - Whether to detach from the terminal
/// * `working_dir` - Optional working directory to change to after daemonizing
/// * `pid_file` - Optional PID file path (defaults to runtime-appropriate location)
///
/// # Returns
///
/// Returns `Ok(())` if daemonization succeeded or was not requested.
/// Returns `Err` if daemonization failed.
///
/// # Example
///
/// ```ignore
/// // Call before creating tokio runtime
/// maybe_daemonize(cmd.detach, cmd.server.working_dir.clone(), cmd.server.pid_file.clone())?;
///
/// // Now create runtime and start server
/// let rt = tokio::runtime::Runtime::new()?;
/// rt.block_on(async { /* server code */ });
/// ```
pub fn maybe_daemonize(
    detach: bool,
    working_dir: Option<String>,
    pid_file: Option<String>,
) -> Result<()> {
    if !detach {
        return Ok(());
    }

    #[cfg(unix)]
    {
        use daemonize::Daemonize;

        let pid_path = pid_file
            .map(PathBuf::from)
            .unwrap_or_else(default_pid_file);

        // Ensure runtime directory exists with correct permissions
        hyprstream_workers::paths::ensure_runtime_dir()
            .with_context(|| "Failed to create runtime directory")?;

        let mut daemon = Daemonize::new()
            .pid_file(&pid_path)
            .chown_pid_file(true);

        // Set working directory if specified
        if let Some(ref dir) = working_dir {
            let path = PathBuf::from(dir);
            if !path.exists() {
                std::fs::create_dir_all(&path)
                    .with_context(|| format!("Failed to create working directory: {dir}"))?;
            }
            daemon = daemon.working_directory(path);
        }

        info!("Daemonizing with PID file: {}", pid_path.display());

        daemon
            .start()
            .with_context(|| format!("Failed to daemonize (PID file: {})", pid_path.display()))?;

        info!("Successfully daemonized");
        Ok(())
    }

    #[cfg(not(unix))]
    {
        anyhow::bail!("Daemonization is only supported on Unix systems. Use a service manager on Windows.");
    }
}

/// Check if a PID file exists and the process is still running
///
/// # Arguments
///
/// * `pid_file` - Optional PID file path (defaults to runtime-appropriate location)
///
/// # Returns
///
/// `Some(pid)` if the process is running, `None` if not running or PID file doesn't exist.
pub fn check_running(pid_file: Option<String>) -> Option<u32> {
    let pid_path = pid_file
        .map(PathBuf::from)
        .unwrap_or_else(default_pid_file);

    let pid_str = std::fs::read_to_string(&pid_path).ok()?;
    let pid: u32 = pid_str.trim().parse().ok()?;

    #[cfg(unix)]
    {
        // Check if process exists by sending signal 0 (null signal)
        use nix::sys::signal::kill;
        use nix::unistd::Pid;

        // Using None sends signal 0, which checks process existence without side effects
        if kill(Pid::from_raw(pid as i32), None).is_ok() {
            return Some(pid);
        }
    }

    #[cfg(not(unix))]
    {
        // On non-Unix, just check if PID file exists
        return Some(pid);
    }

    None
}

/// Stop a running daemon by sending SIGTERM
///
/// # Arguments
///
/// * `pid_file` - Optional PID file path (defaults to runtime-appropriate location)
///
/// # Returns
///
/// `Ok(())` if the daemon was stopped or wasn't running.
/// `Err` if stopping failed.
pub fn stop_daemon(pid_file: Option<String>) -> Result<()> {
    let pid_path = pid_file
        .map(PathBuf::from)
        .unwrap_or_else(default_pid_file);

    let pid_str = std::fs::read_to_string(&pid_path)
        .with_context(|| format!("No PID file found at: {}", pid_path.display()))?;

    let pid: u32 = pid_str
        .trim()
        .parse()
        .with_context(|| format!("Invalid PID in file: {pid_str}"))?;

    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        info!("Sending SIGTERM to process {}", pid);
        kill(Pid::from_raw(pid as i32), Signal::SIGTERM)
            .with_context(|| format!("Failed to send SIGTERM to PID {pid}"))?;

        // Remove PID file
        let _ = std::fs::remove_file(&pid_path);

        info!("Daemon stopped (PID {})", pid);
        Ok(())
    }

    #[cfg(not(unix))]
    {
        anyhow::bail!("Stopping daemons is only supported on Unix systems");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pid_file() {
        let pid_file = default_pid_file();
        assert!(pid_file.to_string_lossy().contains("hyprstream"));
    }

    #[test]
    fn test_check_running_no_file() {
        // Should return None if PID file doesn't exist
        let result = check_running(Some("/nonexistent/path/hyprstream.pid".to_owned()));
        assert!(result.is_none());
    }

    #[test]
    fn test_maybe_daemonize_disabled() {
        // Should succeed when detach is false
        let result = maybe_daemonize(false, None, None);
        assert!(result.is_ok());
    }
}
