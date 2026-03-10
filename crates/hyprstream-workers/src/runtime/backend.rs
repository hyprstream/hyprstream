//! SandboxBackend trait — pluggable sandbox runtime abstraction
//!
//! Extracts the VM lifecycle out of `SandboxPool` so multiple backends
//! (Kata, systemd-nspawn, etc.) can provide sandbox isolation.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::PoolConfig;
use crate::error::Result;

use super::sandbox::PodSandbox;
use super::client::{LinuxContainerResources, PodSandboxConfig};

/// Opaque handle stored on each `PodSandbox`.
///
/// Backends stash runtime-specific state (hypervisor handles, PIDs, machine
/// names, …) here.  Callers can downcast via `as_any()` when they know the
/// concrete type.
pub trait SandboxHandle: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
}

/// Pluggable sandbox runtime backend.
///
/// Implementors manage the full sandbox lifecycle: create → start → stop →
/// destroy.  The pool calls these methods and stores the returned
/// `SandboxHandle` on the `PodSandbox`.
#[async_trait]
pub trait SandboxBackend: Send + Sync {
    /// Human-readable backend name (e.g. `"kata"`, `"nspawn"`).
    fn backend_type(&self) -> &'static str;

    /// Check whether the backend's runtime dependencies are available.
    fn is_available(&self) -> bool;

    /// One-time initialisation (verify paths, enable rootless mode, …).
    async fn initialize(&self, config: &PoolConfig) -> Result<()>;

    /// Start a sandbox and return the backend-specific handle.
    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        config: &PodSandboxConfig,
        pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>>;

    /// Gracefully stop a running sandbox.
    async fn stop(&self, sandbox: &PodSandbox) -> Result<()>;

    /// Destroy a sandbox and clean up all resources.
    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()>;

    /// Reset a sandbox for warm-pool reuse.
    ///
    /// Returns `true` if the sandbox can be reused in-place (e.g. Kata keeps
    /// the VM running).  Returns `false` if the sandbox is ephemeral and must
    /// be recreated (e.g. nspawn).
    async fn reset(&self, sandbox: &mut PodSandbox) -> Result<bool>;

    /// Get PIDs belonging to this sandbox (for monitoring / cgroup ops).
    async fn get_pids(&self, sandbox: &PodSandbox) -> Result<Vec<u32>>;

    /// Whether `exec_sync` is supported.
    fn supports_exec(&self) -> bool;

    /// Run a command synchronously inside the sandbox.
    async fn exec_sync(
        &self,
        sandbox: &PodSandbox,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)>;

    /// Apply resource limits to a running sandbox (optional).
    async fn update_resources(
        &self,
        _sandbox: &PodSandbox,
        _resources: &LinuxContainerResources,
    ) -> Result<()> {
        Ok(()) // default: no-op
    }
}
