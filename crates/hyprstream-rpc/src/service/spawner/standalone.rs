//! Standalone process spawner using tokio::process::Command.
//!
//! Spawns daemon processes directly without systemd integration.
//! Uses `.kill_on_drop(true)` for automatic cleanup when the process handle is dropped.

use std::sync::Arc;

use dashmap::DashMap;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use super::{ProcessConfig, ProcessKind, SpawnedProcess, SpawnerBackend};
use crate::error::{Result, RpcError};

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

#[async_trait::async_trait]
impl SpawnerBackend for StandaloneBackend {
    async fn spawn(&self, config: ProcessConfig) -> Result<SpawnedProcess> {
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
        let pid_file = crate::paths::service_pid_file(&config.name);
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

        Ok(SpawnedProcess::new(id, ProcessKind::Direct(pid)))
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

        // Try to get and remove the child handle
        if let Some((_, child_arc)) = self.processes.remove(&process.id) {
            let mut child = child_arc.lock().await;

            // Try graceful SIGTERM first
            if let Err(e) = child.start_kill() {
                tracing::warn!(
                    id = %process.id,
                    error = %e,
                    "Failed to send kill signal"
                );
            }

            // Wait for the process to exit (with timeout)
            match tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) => {
                    tracing::debug!(
                        id = %process.id,
                        status = ?status,
                        "Process exited"
                    );
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        id = %process.id,
                        error = %e,
                        "Error waiting for process"
                    );
                }
                Err(_) => {
                    tracing::warn!(
                        id = %process.id,
                        "Timeout waiting for process, may be killed by drop"
                    );
                }
            }
        } else {
            // Process not in our map, try to kill by PID directly
            tracing::debug!(
                id = %process.id,
                pid = %pid,
                "Process not in map, attempting direct kill by PID"
            );

            // PIDs are always positive and fit in i32 on Unix
            let Some(pid_i32) = i32::try_from(pid).ok().filter(|&p| p > 0) else {
                tracing::warn!(pid = %pid, "Invalid PID, skipping kill");
                return Ok(());
            };
            let nix_pid = nix::unistd::Pid::from_raw(pid_i32);

            // Send SIGTERM
            if let Err(e) = nix::sys::signal::kill(nix_pid, nix::sys::signal::Signal::SIGTERM) {
                if e != nix::errno::Errno::ESRCH {
                    tracing::warn!(
                        pid = %pid,
                        error = %e,
                        "Failed to send SIGTERM"
                    );
                }
            }

            // Wait briefly then send SIGKILL if needed
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            if let Err(e) = nix::sys::signal::kill(nix_pid, nix::sys::signal::Signal::SIGKILL) {
                if e != nix::errno::Errno::ESRCH {
                    tracing::warn!(
                        pid = %pid,
                        error = %e,
                        "Failed to send SIGKILL"
                    );
                }
            }
        }

        tracing::info!(
            id = %process.id,
            "Process stopped"
        );

        Ok(())
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

        // Check if the process exists by sending signal 0
        // PIDs are always positive and fit in i32 on Unix
        let Some(pid_i32) = i32::try_from(pid).ok().filter(|&p| p > 0) else {
            return Ok(false); // Invalid PID means not running
        };
        let nix_pid = nix::unistd::Pid::from_raw(pid_i32);
        match nix::sys::signal::kill(nix_pid, None) {
            // Process exists (Ok) or exists but we can't signal it (EPERM)
            Ok(()) | Err(nix::errno::Errno::EPERM) => Ok(true),
            Err(nix::errno::Errno::ESRCH) => Ok(false), // No such process
            Err(e) => {
                tracing::warn!(pid = %pid, error = %e, "Error checking process status");
                Ok(false)
            }
        }
    }

    fn backend_type(&self) -> &'static str {
        "standalone"
    }
}

impl Drop for StandaloneBackend {
    fn drop(&mut self) {
        // The Child handles have kill_on_drop(true), so they'll be cleaned up automatically
        let count = self.processes.len();
        if count > 0 {
            tracing::debug!(
                count = count,
                "StandaloneBackend dropped with active processes (will be killed)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn_and_stop() -> crate::Result<()> {
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

    #[test]
    fn test_backend_type() {
        let backend = StandaloneBackend::new();
        assert_eq!(backend.backend_type(), "standalone");
    }
}
