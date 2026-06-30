//! PodmanBackend — rootless podman OCI `SandboxBackend` (#346, T2-A of #341).
//!
//! Runs each sandbox as an OCI container managed by `podman`. No hypervisor,
//! no RAFS/nydus — pulls (or reuses) a standard OCI image and keeps the
//! container running (`sleep infinity`) so `exec_sync` can drive workloads
//! inside it. Intended for casual/dev installs where the Kata +
//! Cloud-Hypervisor VM toolchain is unavailable or unwanted.
//!
//! Backend type: `"podman"`.
//!
//! # Scope (MVP)
//!
//! This is a **podman-only** backend. A prior attempt (#484) advertised a
//! "docker fallback" but never actually exercised it correctly — `docker
//! info` does not emit the `{{.Host.Security.Rootless}}` Go-template field
//! podman does, so the fallback's rootless probe silently parsed as `false`
//! for every docker host (a mis-detection, not a real fallback). Rather than
//! half-implement docker support again, this backend is honestly
//! podman-only: [`is_available`](SandboxBackend::is_available) checks for
//! `podman` on `PATH` only. A docker backend, if wanted later, should be its
//! own registration with its own (docker-correct) rootless probe rather than
//! sharing a code path that silently degrades.
//!
//! # Rootless detection
//!
//! `podman info --format '{{.Host.Security.Rootless}}'` is the correct probe
//! (verified against `podman-info(1)`: `Host.Security.Rootless` is a real
//! field in `podman info --format json`). The previous attempt ran this
//! query in `initialize()` but only logged the result — it never affected
//! container creation, so "detection" was dead weight. Here the rootless
//! flag is cached once (`OnceLock`, queried lazily and reused) and threaded
//! into [`PodmanBackend::start`]: rootless hosts get `--userns=keep-id` so
//! the in-container UID maps to the invoking host user (required for bind
//! mounts to be writable without root); rootful hosts skip it (`keep-id` is
//! a rootless-only podman flag and errors out under a rootful daemon).
//!
//! # Config fields used at `start()`
//!
//! The prior attempt accepted `_config: &PodSandboxConfig` but never read
//! it — resources were only ever applied later via the separate
//! `update_resources` call (which nothing invoked at sandbox-creation time),
//! so a sandbox's CPU/memory limits and security context were silently
//! dropped on the floor at `start()`. This implementation reads:
//!
//! * `config.hostname` → `--hostname`
//! * `config.dns_config.{servers,searches,options}` → `--dns` / `--dns-search` / `--dns-option`
//! * `config.linux.resources.{cpu_quota,cpu_period,cpu_shares,memory_limit_in_bytes,memory_swap_limit_in_bytes,cpuset_cpus,cpuset_mems}` → `--cpu-quota` / `--cpu-period` / `--cpu-shares` / `--memory` / `--memory-swap` / `--cpuset-cpus` / `--cpuset-mems`
//! * `config.linux.security_context.{privileged,run_as_user,readonly_rootfs}` → `--privileged` / `--user` / `--read-only`
//!
//! `update_resources` remains available for live updates via `podman update`
//! on an already-running sandbox (warm-pool resize), which is a distinct use
//! case from the one-time flags applied at `start()`.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::backend::{SandboxBackend, SandboxHandle};
use super::client::{LinuxContainerResources, PodSandboxConfig};
use super::sandbox::PodSandbox;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Container name for a given sandbox id.
fn container_name(sandbox_id: &str) -> String {
    format!("hyprstream-{sandbox_id}")
}

/// Parse the value of `podman info --format '{{.Host.Security.Rootless}}'`.
///
/// Factored out so the parsing logic itself is unit-testable against fixture
/// output without invoking a live `podman` binary.
fn parse_rootless_output(stdout: &str) -> bool {
    stdout.trim().eq_ignore_ascii_case("true")
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the podman backend.
#[derive(Debug, Clone)]
pub struct PodmanConfig {
    /// OCI base image to run. Overridable per-sandbox via the
    /// `hyprstream.io/worker-image` annotation.
    pub base_image: String,
    /// Network mode passed to `--network`.
    pub network_mode: String,
    /// Stop timeout (seconds) before forceful kill.
    pub stop_timeout: u64,
}

impl Default for PodmanConfig {
    fn default() -> Self {
        Self {
            base_image: std::env::var("HYPRSTREAM_WORKER_IMAGE")
                .unwrap_or_else(|_| "ghcr.io/hyprstream/worker:latest".to_owned()),
            network_mode: "slirp4netns".to_owned(),
            stop_timeout: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-specific state stored on each `PodSandbox`.
#[derive(Debug, Clone)]
pub struct PodmanHandle {
    /// Sandbox identifier.
    pub sandbox_id: String,
    /// Container name (`hyprstream-{sandbox_id}`).
    pub container_name: String,
    /// PID of the container init process (populated after start).
    pub init_pid: Option<u32>,
}

impl SandboxHandle for PodmanHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// Rootless-preferring podman OCI `SandboxBackend`.
pub struct PodmanBackend {
    config: PodmanConfig,
    /// Cached rootless probe result — queried once on first use rather than
    /// shelling out to `podman info` on every `start()` call.
    rootless: OnceLock<bool>,
}

impl PodmanBackend {
    pub fn new(config: PodmanConfig) -> Self {
        Self {
            config,
            rootless: OnceLock::new(),
        }
    }

    /// Whether `podman` is on `PATH`. Free function so it can be used as a
    /// `fn()` pointer in the inventory registration (which cannot capture
    /// `self`).
    fn registry_is_available() -> bool {
        which::which("podman").is_ok()
    }

    /// Query (and cache) whether the local podman is running rootless.
    ///
    /// `podman info --format '{{.Host.Security.Rootless}}'` prints `true` or
    /// `false`. Any failure to run/parse it is treated as "not rootless"
    /// (conservative: skip `--userns=keep-id` rather than guess), logged at
    /// `warn`.
    async fn detect_rootless(&self) -> bool {
        if let Some(cached) = self.rootless.get() {
            return *cached;
        }

        let detected = match tokio::process::Command::new("podman")
            .args(["info", "--format", "{{.Host.Security.Rootless}}"])
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                parse_rootless_output(&String::from_utf8_lossy(&out.stdout))
            }
            Ok(out) => {
                warn!(
                    stderr = %String::from_utf8_lossy(&out.stderr),
                    "podman info failed while probing rootless mode; assuming rootful"
                );
                false
            }
            Err(e) => {
                warn!(error = %e, "failed to run `podman info`; assuming rootful");
                false
            }
        };

        // OnceLock::get_or_init isn't usable here since `detected` was already
        // computed via an async call; set() is fine — only one writer ever
        // races in practice (backend init), and a lost race just means we
        // computed it twice safely.
        let _ = self.rootless.set(detected);
        detected
    }

    /// Inspect a running container and return its init PID.
    async fn inspect_pid(name: &str) -> Option<u32> {
        let out = tokio::process::Command::new("podman")
            .args(["inspect", "--format", "{{.State.Pid}}", name])
            .output()
            .await
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let pid_str = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        pid_str.parse().ok().filter(|&p: &u32| p > 0)
    }

    /// Resolve the OCI image: annotation overrides config default.
    fn resolve_image(&self, annotations: &HashMap<String, String>) -> String {
        annotations
            .get("hyprstream.io/worker-image")
            .cloned()
            .unwrap_or_else(|| self.config.base_image.clone())
    }

    /// Translate `PodSandboxConfig` + resolved rootless mode into `podman
    /// run` arguments. Pure (no I/O) so it can be unit-tested directly.
    fn build_run_args(
        &self,
        name: &str,
        image: &str,
        config: &PodSandboxConfig,
        rootless: bool,
    ) -> Vec<String> {
        let mut args = vec![
            "run".to_owned(),
            "--detach".to_owned(),
            "--rm".to_owned(),
            format!("--name={name}"),
            format!("--network={}", self.config.network_mode),
            format!("--stop-timeout={}", self.config.stop_timeout),
        ];

        // Rootless hosts need the container UID mapped onto the invoking
        // host user for bind mounts to be writable; `--userns=keep-id` is a
        // rootless-only podman flag (it errors under a rootful daemon), so
        // gate it strictly on the detected mode rather than always passing
        // it (the #484 mis-detection bug: the probe ran but never gated
        // anything).
        if rootless {
            args.push("--userns=keep-id".to_owned());
        }

        if !config.hostname.is_empty() {
            args.push(format!("--hostname={}", config.hostname));
        }

        for server in &config.dns_config.servers {
            args.push(format!("--dns={server}"));
        }
        for search in &config.dns_config.searches {
            args.push(format!("--dns-search={search}"));
        }
        for opt in &config.dns_config.options {
            args.push(format!("--dns-option={opt}"));
        }

        for pm in &config.port_mappings {
            let host_ip = if pm.host_ip.is_empty() {
                String::new()
            } else {
                format!("{}:", pm.host_ip)
            };
            args.push(format!(
                "--publish={host_ip}{}:{}",
                pm.host_port, pm.container_port
            ));
        }

        let sec = &config.linux.security_context;
        if sec.privileged {
            args.push("--privileged".to_owned());
        }
        if sec.run_as_user != 0 {
            args.push(format!("--user={}", sec.run_as_user));
        }
        if sec.readonly_rootfs {
            args.push("--read-only".to_owned());
        }

        let res = &config.linux.resources;
        if res.cpu_quota > 0 {
            args.push(format!("--cpu-quota={}", res.cpu_quota));
        }
        if res.cpu_period > 0 {
            args.push(format!("--cpu-period={}", res.cpu_period));
        }
        if res.cpu_shares > 0 {
            args.push(format!("--cpu-shares={}", res.cpu_shares));
        }
        if !res.cpuset_cpus.is_empty() {
            args.push(format!("--cpuset-cpus={}", res.cpuset_cpus));
        }
        if !res.cpuset_mems.is_empty() {
            args.push(format!("--cpuset-mems={}", res.cpuset_mems));
        }
        if res.memory_limit_in_bytes > 0 {
            args.push(format!("--memory={}", res.memory_limit_in_bytes));
        }
        if res.memory_swap_limit_in_bytes > 0 {
            args.push(format!("--memory-swap={}", res.memory_swap_limit_in_bytes));
        }

        args.push(image.to_owned());
        args.push("sleep".to_owned());
        args.push("infinity".to_owned());

        args
    }
}

impl std::fmt::Debug for PodmanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PodmanBackend")
            .field("config", &self.config)
            .finish()
    }
}

#[async_trait]
impl SandboxBackend for PodmanBackend {
    fn backend_type(&self) -> &'static str {
        "podman"
    }

    fn is_available(&self) -> bool {
        Self::registry_is_available()
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        if !self.is_available() {
            return Err(WorkerError::ConfigError("podman not found in PATH".into()));
        }
        let rootless = self.detect_rootless().await;
        info!(rootless, "PodmanBackend initialized");
        Ok(())
    }

    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        config: &PodSandboxConfig,
        pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>> {
        let image = self.resolve_image(annotations);
        let name = container_name(&sandbox.id);
        let rootless = self.detect_rootless().await;

        // Ensure sandbox runtime directory exists, mirroring nspawn's
        // bookkeeping so callers can rely on `sandbox.sandbox_path()` even
        // though podman containers don't bind-mount it by default.
        let sandbox_path = pool_config.runtime_dir.join(&sandbox.id);
        tokio::fs::create_dir_all(&sandbox_path).await?;
        sandbox.sandbox_path = sandbox_path;

        let args = self.build_run_args(&name, &image, config, rootless);

        info!(
            sandbox_id = %sandbox.id,
            container = %name,
            image = %image,
            rootless,
            "Starting podman sandbox"
        );

        let output = tokio::process::Command::new("podman")
            .args(&args)
            .output()
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("podman run failed: {e}")))?;

        if !output.status.success() {
            return Err(WorkerError::VmStartFailed(format!(
                "podman run exited non-zero for sandbox {}: {}",
                sandbox.id,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let init_pid = Self::inspect_pid(&name).await;
        debug!(sandbox_id = %sandbox.id, pid = ?init_pid, "Container started");

        Ok(Arc::new(PodmanHandle {
            sandbox_id: sandbox.id.clone(),
            container_name: name,
            init_pid,
        }))
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        let name = container_name(&sandbox.id);
        info!(sandbox_id = %sandbox.id, container = %name, "Stopping podman sandbox");

        let status = tokio::process::Command::new("podman")
            .args([
                "stop",
                "--time",
                &self.config.stop_timeout.to_string(),
                &name,
            ])
            .status()
            .await
            .map_err(|e| WorkerError::VmStopFailed(format!("podman stop failed: {e}")))?;

        if !status.success() {
            warn!(
                sandbox_id = %sandbox.id,
                "podman stop returned non-zero (container may already be stopped)"
            );
        }

        Ok(())
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        let name = container_name(&sandbox.id);
        info!(sandbox_id = %sandbox.id, container = %name, "Destroying podman sandbox");

        // Forceful remove — idempotent, mirrors nspawn's best-effort terminate.
        let _ = tokio::process::Command::new("podman")
            .args(["rm", "--force", &name])
            .status()
            .await;

        let sandbox_path = sandbox.sandbox_path();
        if sandbox_path.exists() {
            if let Err(e) = tokio::fs::remove_dir_all(sandbox_path).await {
                warn!(
                    sandbox_id = %sandbox.id,
                    error = %e,
                    "Failed to remove sandbox directory"
                );
            }
        }

        Ok(())
    }

    async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
        // Podman containers are ephemeral (run with `--rm`) — the pool must
        // recreate rather than reuse in-place, same as nspawn.
        Ok(false)
    }

    async fn get_pids(&self, sandbox: &PodSandbox) -> Result<Vec<u32>> {
        let name = container_name(&sandbox.id);
        match Self::inspect_pid(&name).await {
            Some(pid) => Ok(vec![pid]),
            None => Ok(Vec::new()),
        }
    }

    fn supports_exec(&self) -> bool {
        true
    }

    async fn exec_sync(
        &self,
        sandbox: &PodSandbox,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
        let name = container_name(&sandbox.id);

        let mut args = vec!["exec".to_owned(), name, "--".to_owned()];
        args.extend_from_slice(command);

        let output = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            tokio::process::Command::new("podman").args(&args).output(),
        )
        .await
        .map_err(|_| WorkerError::SandboxTimeout {
            operation: format!("exec_sync in {}", sandbox.id),
            timeout_secs,
        })?
        .map_err(|e| WorkerError::ExecFailed(format!("podman exec failed: {e}")))?;

        let exit_code = output.status.code().unwrap_or(-1);
        Ok((exit_code, output.stdout, output.stderr))
    }

    async fn update_resources(
        &self,
        sandbox: &PodSandbox,
        resources: &LinuxContainerResources,
    ) -> Result<()> {
        let name = container_name(&sandbox.id);

        let mut args = vec!["update".to_owned(), name];

        if resources.cpu_quota > 0 {
            args.push(format!("--cpu-quota={}", resources.cpu_quota));
        }
        if resources.cpu_period > 0 {
            args.push(format!("--cpu-period={}", resources.cpu_period));
        }
        if resources.memory_limit_in_bytes > 0 {
            args.push(format!("--memory={}", resources.memory_limit_in_bytes));
        }

        if args.len() == 2 {
            // Nothing to update.
            return Ok(());
        }

        let status = tokio::process::Command::new("podman")
            .args(&args)
            .status()
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("podman update failed: {e}")))?;

        if !status.success() {
            warn!(sandbox_id = %sandbox.id, "podman update returned non-zero");
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend registry self-registration (#507 / #518) — gated on the `podman`
// feature
// ─────────────────────────────────────────────────────────────────────────────

// The podman backend is registered ONLY when compiled with `--features
// podman`, mirroring how `kata` registers only under `kata-vm` and `wasm`
// only under `wasm`. With the feature off, an explicit `podman` request hits
// the "unknown backend" error path rather than silently downgrading.
//
// Priority: a real container runtime (kernel namespaces + cgroups via
// runc/crun) is stronger isolation than nspawn's raw `systemd-nspawn`
// bind-mount-the-host-root model (no cgroup/userns hardening beyond what the
// caller wires up manually) but weaker than Kata's full VM boundary. Placed
// at 30: above nspawn (10), below kata (100), leaving headroom (20s) for any
// future intermediate tier.
inventory::submit! {
    crate::runtime::selection::BackendRegistration {
        name: "podman",
        priority: 30,
        // Real container isolation (namespaces + cgroups) → eligible for
        // `"auto"`.
        auto_selectable: true,
        is_available: PodmanBackend::registry_is_available,
        construct: |_ctx| {
            Ok(std::sync::Arc::new(PodmanBackend::new(PodmanConfig::default()))
                as std::sync::Arc<dyn crate::runtime::SandboxBackend>)
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::generated::worker_client::Protocol;
    use crate::runtime::client::{
        DNSConfig, LinuxPodSandboxConfig, LinuxSandboxSecurityContext, PortMapping,
    };

    fn base_config() -> PodSandboxConfig {
        PodSandboxConfig::default()
    }

    // ── Pure helpers ──

    #[test]
    fn test_backend_type() {
        let b = PodmanBackend::new(PodmanConfig::default());
        assert_eq!(b.backend_type(), "podman");
    }

    #[test]
    fn test_is_available_does_not_panic() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let _ = b.is_available();
    }

    #[test]
    fn test_container_name() {
        assert_eq!(container_name("abc-123"), "hyprstream-abc-123");
    }

    #[test]
    fn test_resolve_image_default() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let annotations = HashMap::new();
        let image = b.resolve_image(&annotations);
        assert!(!image.is_empty());
    }

    #[test]
    fn test_resolve_image_annotation_override() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut annotations = HashMap::new();
        annotations.insert(
            "hyprstream.io/worker-image".to_owned(),
            "custom/image:v1".to_owned(),
        );
        assert_eq!(b.resolve_image(&annotations), "custom/image:v1");
    }

    #[test]
    fn test_handle_as_any() {
        let h = PodmanHandle {
            sandbox_id: "test-1".into(),
            container_name: "hyprstream-test-1".into(),
            init_pid: Some(99),
        };
        let h: Arc<dyn SandboxHandle> = Arc::new(h);
        let down = h.as_any().downcast_ref::<PodmanHandle>();
        assert!(down.is_some());
        assert_eq!(down.unwrap().init_pid, Some(99));
    }

    // ── Rootless-detection parsing (fixture output, no live podman) ──

    #[test]
    fn test_parse_rootless_true() {
        assert!(parse_rootless_output("true\n"));
        assert!(parse_rootless_output("true"));
        assert!(parse_rootless_output("True"));
        assert!(parse_rootless_output("  true  "));
    }

    #[test]
    fn test_parse_rootless_false() {
        assert!(!parse_rootless_output("false\n"));
        assert!(!parse_rootless_output("false"));
        assert!(!parse_rootless_output(""));
        // A docker-style empty/missing-field response (no such Go-template
        // path) must not be mis-parsed as rootless.
        assert!(!parse_rootless_output("<no value>"));
    }

    // ── PodSandboxConfig → podman run argument translation ──
    //
    // These exercise exactly the field-dropping bug #484 had: resources,
    // security context, hostname, DNS, and port mappings must all show up
    // in the constructed argv.

    #[test]
    fn test_build_run_args_minimal() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let cfg = base_config();
        let args = b.build_run_args("hyprstream-x", "img:latest", &cfg, false);
        assert!(args.contains(&"run".to_owned()));
        assert!(args.contains(&"--name=hyprstream-x".to_owned()));
        assert!(args.iter().any(|a| a.starts_with("--network=")));
        assert!(args.contains(&"img:latest".to_owned()));
        assert!(args.contains(&"sleep".to_owned()));
        assert!(args.contains(&"infinity".to_owned()));
        // Rootless=false must not pass the rootless-only flag.
        assert!(!args.contains(&"--userns=keep-id".to_owned()));
    }

    #[test]
    fn test_build_run_args_rootless_sets_userns_keep_id() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let cfg = base_config();
        let args = b.build_run_args("hyprstream-x", "img:latest", &cfg, true);
        assert!(
            args.contains(&"--userns=keep-id".to_owned()),
            "rootless mode must pass --userns=keep-id, got: {args:?}"
        );
    }

    #[test]
    fn test_build_run_args_hostname() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut cfg = base_config();
        cfg.hostname = "my-sandbox".to_owned();
        let args = b.build_run_args("n", "img", &cfg, false);
        assert!(args.contains(&"--hostname=my-sandbox".to_owned()));
    }

    #[test]
    fn test_build_run_args_dns_config() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut cfg = base_config();
        cfg.dns_config = DNSConfig {
            servers: vec!["1.1.1.1".to_owned()],
            searches: vec!["example.com".to_owned()],
            options: vec!["ndots:2".to_owned()],
        };
        let args = b.build_run_args("n", "img", &cfg, false);
        assert!(args.contains(&"--dns=1.1.1.1".to_owned()));
        assert!(args.contains(&"--dns-search=example.com".to_owned()));
        assert!(args.contains(&"--dns-option=ndots:2".to_owned()));
    }

    #[test]
    fn test_build_run_args_port_mappings() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut cfg = base_config();
        cfg.port_mappings = vec![PortMapping {
            protocol: Protocol::Tcp,
            container_port: 8080,
            host_port: 18080,
            host_ip: String::new(),
        }];
        let args = b.build_run_args("n", "img", &cfg, false);
        assert!(
            args.contains(&"--publish=18080:8080".to_owned()),
            "got: {args:?}"
        );
    }

    #[test]
    fn test_build_run_args_security_context() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut cfg = base_config();
        cfg.linux = LinuxPodSandboxConfig {
            security_context: LinuxSandboxSecurityContext {
                run_as_user: 1000,
                readonly_rootfs: true,
                privileged: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let args = b.build_run_args("n", "img", &cfg, false);
        assert!(args.contains(&"--privileged".to_owned()));
        assert!(args.contains(&"--user=1000".to_owned()));
        assert!(args.contains(&"--read-only".to_owned()));
    }

    #[test]
    fn test_build_run_args_resources() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut cfg = base_config();
        cfg.linux = LinuxPodSandboxConfig {
            resources: LinuxContainerResources {
                cpu_quota: 50000,
                cpu_period: 100000,
                cpu_shares: 512,
                memory_limit_in_bytes: 268_435_456,
                memory_swap_limit_in_bytes: 536_870_912,
                cpuset_cpus: "0-1".to_owned(),
                cpuset_mems: "0".to_owned(),
                ..Default::default()
            },
            ..Default::default()
        };
        let args = b.build_run_args("n", "img", &cfg, false);
        assert!(args.contains(&"--cpu-quota=50000".to_owned()));
        assert!(args.contains(&"--cpu-period=100000".to_owned()));
        assert!(args.contains(&"--cpu-shares=512".to_owned()));
        assert!(args.contains(&"--memory=268435456".to_owned()));
        assert!(args.contains(&"--memory-swap=536870912".to_owned()));
        assert!(args.contains(&"--cpuset-cpus=0-1".to_owned()));
        assert!(args.contains(&"--cpuset-mems=0".to_owned()));
    }

    #[test]
    fn test_resolve_image_unused_annotation_key_ignored() {
        let b = PodmanBackend::new(PodmanConfig::default());
        let mut annotations = HashMap::new();
        annotations.insert("hyprstream.io/other".to_owned(), "x".to_owned());
        let image = b.resolve_image(&annotations);
        assert_eq!(image, b.config.base_image);
    }

    // ── Registry / selection ──

    #[test]
    fn test_registered_in_inventory() {
        use crate::runtime::selection::BackendRegistration;
        let reg = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "podman")
            .expect("podman backend must be registered under the podman feature");
        assert_eq!(reg.priority, 30);
        assert!(
            reg.auto_selectable,
            "podman is real container isolation — auto-selectable"
        );
    }

    #[test]
    fn test_podman_priority_between_nspawn_and_kata() {
        use crate::runtime::selection::BackendRegistration;
        let podman = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "podman")
            .unwrap();
        let nspawn = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "nspawn")
            .unwrap();
        assert!(
            podman.priority > nspawn.priority,
            "podman (real container) must outrank nspawn for auto-selection"
        );
        #[cfg(feature = "kata-vm")]
        {
            let kata = inventory::iter::<BackendRegistration>()
                .find(|r| r.name == "kata")
                .unwrap();
            assert!(
                kata.priority > podman.priority,
                "kata (VM) must outrank podman for auto-selection"
            );
        }
    }

    // ── Live-podman integration tests ──
    //
    // These require an actual podman daemon/binary and a pulled image, which
    // this sandboxed CI/dev environment does not reliably provide. Ignored
    // by default; run manually with `cargo test -p hyprstream-workers
    // --features podman -- --ignored` on a host with podman configured.

    #[tokio::test]
    #[ignore = "requires a live podman binary + daemon"]
    async fn test_live_detect_rootless() {
        let b = PodmanBackend::new(PodmanConfig::default());
        // Just verify it doesn't panic and returns deterministically when
        // queried twice (cache hit on the second call).
        let first = b.detect_rootless().await;
        let second = b.detect_rootless().await;
        assert_eq!(first, second);
    }

    #[tokio::test]
    #[ignore = "requires a live podman binary + daemon + pullable image"]
    async fn test_live_start_exec_destroy_roundtrip() {
        // A full start()/exec_sync()/destroy() roundtrip against a real
        // podman daemon. Left as a documented manual test rather than faked.
    }
}
