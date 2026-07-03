//! SandboxBackend trait — pluggable sandbox runtime abstraction
//!
//! Extracts the VM lifecycle out of `SandboxPool` so multiple backends
//! (Kata, systemd-nspawn, etc.) can provide sandbox isolation.
//!
//! ## #508 / #635 — narrowing in progress
//!
//! Epic #508 converges worker sandboxing onto a "namespace IS the OS surface,
//! backends-as-transport" model: a `SandboxBackend` boots a guest OS under an
//! isolation boundary and delivers the composed [`Namespace`] to it — it does
//! NOT own image-pull or run commands. [`deliver_namespace`](SandboxBackend::deliver_namespace)
//! is the narrowed method this trait is converging on.
//!
//! The CRI-shaped methods below (`exec_sync`/`supports_exec`, the
//! `PodSandboxConfig`-driven image pull implicit in `start`, and
//! `update_resources`'s `LinuxContainerResources`) are kept on the trait for
//! now and marked **TRANSITIONAL**: they move off once the guest-OS
//! control-file service (kata-agent #344, Wanix `/task` #612) and the
//! universal filesystem-service Mount (#633/#641) are both wired end-to-end.
//! Until then, removing them here would break every existing caller
//! (`SandboxPool`, `ExecMount`, CRI-facing worker RPCs) for no working
//! replacement. See issue #635 for the full narrowing rationale.
use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::PoolConfig;
use crate::error::Result;

use super::sandbox::PodSandbox;
use super::client::{CpuUsage, LinuxContainerResources, MemoryUsage, PodSandboxConfig};
use hyprstream_vfs::{Namespace, Subject};

/// Opaque handle stored on each `PodSandbox`.
///
/// Backends stash runtime-specific state (hypervisor handles, PIDs, machine
/// names, …) here.  Callers can downcast via `as_any()` when they know the
/// concrete type.
pub trait SandboxHandle: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
}

/// How a composed [`Namespace`] should be attached to a booted guest.
///
/// Designed to fit Kata's existing virtio-fs/ShareFs delivery today, and to
/// be extensible to #653's local-FUSE-mount serve mode once it lands (a
/// `BindMount` target can point at a FUSE mountpoint just as well as a plain
/// directory — this enum does not need to change for that).
#[derive(Debug, Clone)]
pub enum NamespaceTransport {
    /// Serve the namespace over a vhost-user-fs (virtio-fs) socket. The guest
    /// mounts it via a ShareFs/virtio-fs device tagged `mount_tag`. This is
    /// Kata's transport.
    VirtioFs {
        /// Host-side socket path the backend should serve the namespace on
        /// (or has already served it on — backend-specific).
        socket_path: PathBuf,
        /// The virtio-fs mount tag the guest uses to identify the share.
        mount_tag: String,
    },
    /// The namespace is (or will be) materialized at a host directory the
    /// guest can see directly — a plain bind-mount for backends that don't
    /// need FUSE (e.g. nspawn), or a #653 local-FUSE mountpoint once that
    /// lands.
    BindMount {
        /// Host-side directory to bind/attach into the guest.
        target: PathBuf,
    },
    /// Pass mount references directly into the guest's in-process
    /// environment rather than over any wire transport — the wasm
    /// host-imports path (no separate guest OS to mount into).
    HostImports,
}

/// Result of [`SandboxBackend::deliver_namespace`]: what's now available for
/// the guest to consume, and any transitional bookkeeping the caller needs.
pub enum NamespaceDelivery {
    /// The namespace is being served over a vhost-user-fs socket.
    VirtioFs {
        socket_path: PathBuf,
        mount_tag: String,
        /// Backend-specific guard object that must be kept alive for as long
        /// as the guest needs this mount (e.g. the thread serving the
        /// socket). Opaque to callers that don't need the concrete type —
        /// dropping it early may tear down the serving thread/socket, just
        /// like dropping a `SandboxHandle` early would. `None` when the
        /// backend needs no such guard.
        guard: Option<Arc<dyn Any + Send + Sync>>,
    },
    /// The namespace is available at this host path.
    BindMount { target: PathBuf },
    /// The namespace was imported directly into the guest's environment.
    HostImports,
}

impl std::fmt::Debug for NamespaceDelivery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VirtioFs { socket_path, mount_tag, .. } => f
                .debug_struct("VirtioFs")
                .field("socket_path", socket_path)
                .field("mount_tag", mount_tag)
                .finish_non_exhaustive(),
            Self::BindMount { target } => {
                f.debug_struct("BindMount").field("target", target).finish()
            }
            Self::HostImports => write!(f, "HostImports"),
        }
    }
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
    // TRANSITIONAL (#635): `config: &PodSandboxConfig` is the CRI-shaped
    // image-pull/pod config. Once the universal filesystem-service Mount
    // (#633/#641) owns image resolution end-to-end, `start` should narrow to
    // "boot a guest OS", with image content arriving via `deliver_namespace`
    // instead of being resolved here. Not touched in #635 to avoid breaking
    // every existing caller before that replacement is wired.
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

    /// Deliver a composed [`Namespace`] to this sandbox's guest via
    /// `transport` (#635 — the narrowed `SandboxBackend` surface epic #508 is
    /// converging on).
    ///
    /// A backend's job here is exactly: attach `namespace` (already fully
    /// composed — rootfs + injected mounts — by the caller; composition
    /// itself is NOT this method's job) to the running/starting guest, using
    /// whichever `transport` variant it supports. `subject` is the policy/
    /// audit principal the namespace is served under (threaded into every VFS
    /// op at the Mount boundary).
    ///
    /// Default: unsupported. Not every backend has been re-grounded onto this
    /// method yet (see #617/#619/nspawn re-grounding, blocked on this issue);
    /// backends that haven't should fail closed rather than silently no-op.
    async fn deliver_namespace(
        &self,
        _sandbox: &PodSandbox,
        _namespace: Namespace,
        _subject: Subject,
        _transport: NamespaceTransport,
    ) -> Result<NamespaceDelivery> {
        Err(crate::error::WorkerError::Unsupported(format!(
            "{} backend does not implement deliver_namespace yet",
            self.backend_type()
        )))
    }

    /// Whether `exec_sync` is supported.
    // TRANSITIONAL (#635): `exec_sync`/`supports_exec` are CRI-shaped
    // ("run a command in the sandbox") and don't fit a guest with no
    // "command" absent a guest OS (wasm Profile A). They move off this trait
    // once the guest-OS control-file service (kata-agent #344, Wanix `/task`
    // #612) is wired — commands will run *in the guest OS*, driven through a
    // 9P control file, not through the backend directly. Kept here,
    // unchanged, until that replacement exists end-to-end.
    fn supports_exec(&self) -> bool;

    /// Run a command synchronously inside the sandbox.
    // TRANSITIONAL (#635): see `supports_exec` above — moves to the guest-OS
    // control-file service once #344/#612 land.
    async fn exec_sync(
        &self,
        sandbox: &PodSandbox,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)>;

    /// Apply resource limits to a running sandbox (optional).
    // TRANSITIONAL (#635): `LinuxContainerResources` is CRI-shaped. Moves to a
    // resource-control file service / namespace node once that lands; kept as
    // a trait method (with a no-op default) until then.
    async fn update_resources(
        &self,
        _sandbox: &PodSandbox,
        _resources: &LinuxContainerResources,
    ) -> Result<()> {
        Ok(()) // default: no-op
    }

    /// Fetch live per-container resource usage from the sandbox's guest.
    ///
    /// Returns `Some((cpu, memory))` sourced from inside the sandbox (e.g.
    /// the kata-agent stats RPC / guest cgroups), or `None` when the backend
    /// cannot observe the guest (default). Callers fall back to
    /// placeholder/zeroed usage when this is `None`, so a backend that has no
    /// in-guest visibility does not have to synthesise fake numbers.
    async fn container_stats(
        &self,
        _sandbox: &PodSandbox,
    ) -> Result<Option<(CpuUsage, MemoryUsage)>> {
        Ok(None) // default: no guest-level stats available
    }
}
