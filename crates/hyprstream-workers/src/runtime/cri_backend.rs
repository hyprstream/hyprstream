//! CriBackend — CRI-client sandbox isolation, driving an external containerd/CRI-O (#510)
//!
//! Unlike [`super::kata_backend::KataBackend`] (hyprstream embeds/drives the Kata
//! VM runtime directly) and [`super::oci_backend::OciBackend`] (hyprstream shells
//! out to podman/youki/crun itself, #694), `CriBackend` runs **no runtime at
//! all** — it is a thin `tonic` gRPC *client* of the CRI v1 (`runtime.v1`)
//! `RuntimeService`/`ImageService`, the exact protocol `kubelet` speaks against
//! an already-running CRI-compliant runtime service (containerd's built-in CRI
//! plugin, or CRI-O). This is the generic "any CRI-compliant runtime becomes
//! usable through one client" compatibility seam epic #508 calls for; getting
//! Kata isolation via Kubernetes `RuntimeClass` on a containerd host falls out
//! of this for free (`CriConfig::runtime_handler` / the
//! `hyprstream.io/runtime-class` annotation), but that is a side benefit, not
//! the point — **`KataBackend` stays the embedded/DAX-capable path and is not
//! touched or replaced here.**
//!
//! ## Wire mapping (near-isomorphic — our capnp schema already mirrors CRI v1)
//!
//! | `SandboxBackend`      | CRI v1 RPC(s)                                              |
//! |-----------------------|-------------------------------------------------------------|
//! | `start`                | `PullImage` (ImageService) → `RunPodSandbox` → `CreateContainer` → `StartContainer` |
//! | `stop`                 | `StopContainer` → `StopPodSandbox`                          |
//! | `destroy`              | `RemoveContainer` → `RemovePodSandbox`                       |
//! | `exec_sync`            | `ExecSync`                                                   |
//! | `update_resources`     | `UpdateContainerResources`                                   |
//! | `get_pids`             | not exposed generically by CRI v1 (see below) → `Ok(vec![])` |
//!
//! `get_pids`: CRI v1's `ContainerStatus` has no standardized PID field (some
//! runtimes surface it only in the free-form, verbose `info` map, which is not
//! part of the typed contract and not safe to parse runtime-agnostically). This
//! mirrors the trait's own `container_stats` default (`Ok(None)`) — a backend
//! that has no generic way to observe a value returns an empty/absent result
//! rather than fabricating one.
//!
//! ## `deliver_namespace` boundary (#635, from the CRI-client design spike)
//!
//! CRI's `Mount` type is host-path only (`container_path`/`host_path`), and it
//! is declared **once, at `CreateContainer` time** (threaded through `start`
//! via the same `hyprstream.io/mount.*` annotations `OciBackend` uses) — CRI v1
//! has no RPC that hot-attaches a new mount into an *already-running*
//! container, the way `nspawn`'s `machinectl bind` or Kata's virtio-fs
//! hot-plug do. The containerd-specific mechanism that could do this is an NRI
//! (Node Resource Interface) plugin hook — **explicitly deferred** ("(later)"
//! per issue #510) and not implemented here.
//!
//! So `deliver_namespace` here is verification-only: for `BindMount` it
//! confirms the source directory is already materialized on the host (the
//! same fail-fast check `NspawnBackend::deliver_namespace` does) and returns
//! `NamespaceDelivery::BindMount` — signalling "yes, this path exists and is
//! usable as a CRI `Mount.host_path`", not "this path has just been live-wired
//! into a running sandbox". Any content that must reach a *running* CRI
//! sandbox therefore has to have been declared as a mount before `start()`,
//! not delivered after. `VirtioFs`/`HostImports` are rejected outright: CRI has
//! no virtio-fs-equivalent primitive, and DAX zero-copy intentionally does not
//! traverse generic CRI (it stays the embedded-Kata/Wasm differentiator).

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;
use tracing::{debug, info, warn};

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::backend::{NamespaceDelivery, NamespaceTransport, SandboxBackend, SandboxHandle};
use super::client::{KeyValue, LinuxContainerResources, PodSandboxConfig};
use super::sandbox::PodSandbox;

use cri::image_service_client::ImageServiceClient;
use cri::runtime_service_client::RuntimeServiceClient;
use k8s_cri::v1 as cri;

/// Annotation key: image reference to run (same key `OciBackend` uses, so a
/// workload's annotations are portable across the oci/cri backends).
const ANN_IMAGE: &str = "hyprstream.io/oci-image";
/// Annotation key: workload command, appended as `ContainerConfig.command`
/// (same contract `OciBackend`/`wanix_workload` use).
const ANN_COMMAND: &str = "hyprstream.io/command";
/// Annotation key prefix: per-variable container environment.
const ANN_ENV_PREFIX: &str = "hyprstream.io/env.";
/// Annotation key prefix: bind mount spec (`hyprstream.io/mount.data=/host:/ctr[:ro]`).
const ANN_MOUNT_PREFIX: &str = "hyprstream.io/mount.";
/// Annotation key: CRI `RuntimeClass` name (empty = runtime default handler).
/// Setting this to a Kata-backed `RuntimeClass` on a containerd host is how
/// this backend gets Kata isolation "for free" without embedding kata-agent —
/// an optional side benefit of the generic CRI seam, not its point.
const ANN_RUNTIME_CLASS: &str = "hyprstream.io/runtime-class";

/// Default CRI v1 UDS endpoint (containerd's built-in CRI plugin).
const DEFAULT_CRI_ENDPOINT: &str = "/run/containerd/containerd.sock";
/// Default sandbox/infra image (CRI pause semantics), same default `OciBackend` uses.
const DEFAULT_PAUSE_IMAGE: &str = "registry.k8s.io/pause:3.10";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the CRI-client backend.
#[derive(Debug, Clone)]
pub struct CriConfig {
    /// Unix domain socket the external CRI runtime service listens on.
    /// Default: `/run/containerd/containerd.sock` (containerd's CRI plugin).
    /// CRI-O typically listens on `/run/crio/crio.sock` — override via
    /// `HYPRSTREAM_CRI_ENDPOINT` or this field.
    pub endpoint: PathBuf,
    /// Default `runtime_handler` (CRI `RuntimeClass`) used when a sandbox does
    /// not set [`ANN_RUNTIME_CLASS`]. Empty string selects the runtime's
    /// default handler.
    pub runtime_handler: String,
    /// Image used when no `hyprstream.io/oci-image` annotation is supplied.
    pub default_image: String,
    /// Timeout for exec/lifecycle gRPC calls that don't have their own
    /// CRI-level timeout parameter.
    pub call_timeout: Duration,
}

impl Default for CriConfig {
    fn default() -> Self {
        let endpoint = std::env::var("HYPRSTREAM_CRI_ENDPOINT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_CRI_ENDPOINT));
        let runtime_handler = std::env::var("HYPRSTREAM_CRI_RUNTIME_HANDLER").unwrap_or_default();
        let default_image = std::env::var("HYPRSTREAM_CRI_IMAGE")
            .unwrap_or_else(|_| DEFAULT_PAUSE_IMAGE.to_owned());

        Self {
            endpoint,
            runtime_handler,
            default_image,
            call_timeout: Duration::from_secs(30),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-specific state stored on each `PodSandbox`.
#[derive(Debug, Clone)]
pub struct CriHandle {
    /// Sandbox identifier (matches `PodSandbox::id`).
    pub sandbox_id: String,
    /// CRI-assigned pod sandbox id (`RunPodSandboxResponse.pod_sandbox_id`).
    pub pod_sandbox_id: String,
    /// CRI-assigned workload container id (`CreateContainerResponse.container_id`),
    /// once created.
    pub container_id: Option<String>,
    /// The resolved image reference this sandbox's workload container runs.
    pub image: String,
    /// The `runtime_handler` (CRI `RuntimeClass`) this sandbox was run under.
    pub runtime_handler: String,
}

impl SandboxHandle for CriHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// CRI-client sandbox backend: drives an external containerd/CRI-O over CRI v1 gRPC.
pub struct CriBackend {
    config: CriConfig,
    /// Lazily-connecting gRPC channel over the CRI UDS, built on first async
    /// use via [`Self::channel`]. Building a `tonic::transport::Channel` (even
    /// a "lazy" one) needs an active Tokio reactor, so it cannot happen in
    /// `new()` — the registry's `construct` fn is a plain sync `fn` pointer
    /// that may run outside any runtime context. Deferring to the first async
    /// trait method call (all of which run under tokio) keeps construction
    /// itself infallible and reactor-free.
    channel: tokio::sync::OnceCell<Channel>,
}

impl std::fmt::Debug for CriBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CriBackend")
            .field("config", &self.config)
            .finish()
    }
}

impl CriBackend {
    pub fn new(config: CriConfig) -> Self {
        Self {
            config,
            channel: tokio::sync::OnceCell::new(),
        }
    }

    async fn channel(&self) -> Channel {
        self.channel
            .get_or_init(|| async { uds_channel(&self.config.endpoint) })
            .await
            .clone()
    }

    async fn runtime_client(&self) -> RuntimeServiceClient<Channel> {
        RuntimeServiceClient::new(self.channel().await)
    }

    async fn image_client(&self) -> ImageServiceClient<Channel> {
        ImageServiceClient::new(self.channel().await)
    }

    /// Registry availability probe: does the configured (or default) CRI
    /// socket exist? A cheap, synchronous, fail-closed signal — actual
    /// reachability is confirmed by `initialize`'s `Version` RPC.
    fn registry_is_available() -> bool {
        Self::endpoint_available(&CriConfig::default().endpoint)
    }

    fn endpoint_available(endpoint: &Path) -> bool {
        endpoint.exists()
    }

    /// Resolve the image reference: annotation override → configured default.
    fn resolve_image(&self, annotations: &HashMap<String, String>) -> String {
        annotations
            .get(ANN_IMAGE)
            .cloned()
            .unwrap_or_else(|| self.config.default_image.clone())
    }

    /// Resolve the `runtime_handler` (CRI `RuntimeClass`): annotation override
    /// → configured default (empty = runtime's default handler).
    fn resolve_runtime_handler(&self, annotations: &HashMap<String, String>) -> String {
        annotations
            .get(ANN_RUNTIME_CLASS)
            .cloned()
            .unwrap_or_else(|| self.config.runtime_handler.clone())
    }
}

/// Build a `tonic` channel that dials `endpoint` as a Unix domain socket,
/// lazily (the connect only happens on first RPC use).
fn uds_channel(endpoint: &Path) -> Channel {
    let path = endpoint.to_owned();
    // The URI here is a placeholder — `connect_with_connector_lazy` ignores it
    // and always dials via the connector closure below (the same pattern
    // documented in `k8s_cri`'s own UDS example). `from_static` is infallible
    // (unlike `try_from`), so no fallible parse/unwrap is needed for a URI
    // that is a fixed literal.
    Endpoint::from_static("http://[::]").connect_with_connector_lazy(service_fn(move |_: Uri| {
        let path = path.clone();
        async move {
            let stream = tokio::net::UnixStream::connect(path).await?;
            Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(stream))
        }
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// capnp CRI-shaped types → CRI v1 wire types
// ─────────────────────────────────────────────────────────────────────────────

fn kv_to_map(kvs: &[KeyValue]) -> HashMap<String, String> {
    kvs.iter()
        .map(|kv| (kv.key.clone(), kv.value.clone()))
        .collect()
}

fn to_cri_resources(res: &LinuxContainerResources) -> cri::LinuxContainerResources {
    cri::LinuxContainerResources {
        cpu_period: res.cpu_period,
        cpu_quota: res.cpu_quota,
        cpu_shares: res.cpu_shares,
        memory_limit_in_bytes: res.memory_limit_in_bytes,
        oom_score_adj: res.oom_score_adj,
        cpuset_cpus: res.cpuset_cpus.clone(),
        cpuset_mems: res.cpuset_mems.clone(),
        memory_swap_limit_in_bytes: res.memory_swap_limit_in_bytes,
        ..Default::default()
    }
}

/// Build the CRI `PodSandboxConfig` for `RunPodSandbox`/`CreateContainer` from
/// our capnp-shaped [`PodSandboxConfig`] + the resolved annotation map. Fields
/// with no direct CRI-parity need in the #510 baseline (DNS, port mappings,
/// sysctls, SELinux/seccomp/apparmor, hugepage limits, cgroup-v2 `unified`)
/// are intentionally left at their CRI defaults — nothing here is silently
/// dropped that this backend claims to support; add a field here if/when a
/// workload actually needs it threaded.
fn to_cri_pod_sandbox_config(
    config: &PodSandboxConfig,
    sandbox_id: &str,
    annotations: &HashMap<String, String>,
) -> cri::PodSandboxConfig {
    let hostname = if config.metadata.name.is_empty() {
        sandbox_id.to_owned()
    } else {
        config.metadata.name.clone()
    };

    let sc = &config.linux.security_context;

    cri::PodSandboxConfig {
        metadata: Some(cri::PodSandboxMetadata {
            name: config.metadata.name.clone(),
            uid: config.metadata.uid.clone(),
            namespace: config.metadata.namespace.clone(),
            attempt: config.metadata.attempt,
        }),
        hostname,
        log_directory: config.log_directory.clone(),
        labels: kv_to_map(&config.labels),
        annotations: annotations.clone(),
        linux: Some(cri::LinuxPodSandboxConfig {
            cgroup_parent: config.linux.cgroup_parent.clone(),
            security_context: Some(cri::LinuxSandboxSecurityContext {
                run_as_user: Some(cri::Int64Value {
                    value: sc.run_as_user,
                }),
                run_as_group: Some(cri::Int64Value {
                    value: sc.run_as_group,
                }),
                readonly_rootfs: sc.readonly_rootfs,
                privileged: sc.privileged,
                ..Default::default()
            }),
            overhead: Some(to_cri_resources(&config.linux.overhead)),
            resources: Some(to_cri_resources(&config.linux.resources)),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build the CRI `ContainerConfig` for the sandbox's single workload container.
/// The `hyprstream.io/command` annotation supplies argv (same contract
/// `OciBackend` uses); absent that, the image's own entrypoint runs (CRI pause
/// semantics — a bare sandbox comes up idle).
fn to_cri_container_config(
    sandbox_id: &str,
    image: &str,
    annotations: &HashMap<String, String>,
) -> cri::ContainerConfig {
    let mut envs = vec![cri::KeyValue {
        key: "HYPRSTREAM_INSTANCE".to_owned(),
        value: sandbox_id.to_owned(),
    }];
    for (k, v) in annotations {
        if let Some(name) = k.strip_prefix(ANN_ENV_PREFIX) {
            if !name.is_empty() {
                envs.push(cri::KeyValue {
                    key: name.to_owned(),
                    value: v.clone(),
                });
            }
        }
    }

    let mut mounts = Vec::new();
    for (k, v) in annotations {
        if k.starts_with(ANN_MOUNT_PREFIX) {
            if let Some(m) = to_cri_mount(v) {
                mounts.push(m);
            } else {
                warn!(sandbox_id, spec = %v, "ignoring malformed mount annotation");
            }
        }
    }

    let command: Vec<String> = annotations
        .get(ANN_COMMAND)
        .map(|cmd| cmd.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();

    cri::ContainerConfig {
        metadata: Some(cri::ContainerMetadata {
            name: "workload".to_owned(),
            attempt: 0,
        }),
        image: Some(cri::ImageSpec {
            image: image.to_owned(),
            ..Default::default()
        }),
        command,
        envs,
        mounts,
        labels: HashMap::new(),
        annotations: annotations.clone(),
        ..Default::default()
    }
}

/// Translate a `host:container[:ro|:rw]` annotation spec (the same format
/// `OciBackend::normalize_mount_spec` accepts) into a CRI `Mount`.
fn to_cri_mount(spec: &str) -> Option<cri::Mount> {
    let mut parts = spec.splitn(3, ':');
    let host = parts.next().filter(|s| !s.is_empty())?;
    let ctr = parts.next().filter(|s| !s.is_empty())?;
    let readonly = match parts.next() {
        Some("ro") => true,
        Some("rw") | None => false,
        Some(_) => return None,
    };
    Some(cri::Mount {
        container_path: ctr.to_owned(),
        host_path: host.to_owned(),
        readonly,
        ..Default::default()
    })
}

#[async_trait]
impl SandboxBackend for CriBackend {
    fn backend_type(&self) -> &'static str {
        "cri"
    }

    fn is_available(&self) -> bool {
        Self::endpoint_available(&self.config.endpoint)
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        if !self.is_available() {
            return Err(WorkerError::ConfigError(format!(
                "CRI endpoint '{}' does not exist; is containerd (with the CRI \
                 plugin enabled) or CRI-O running?",
                self.config.endpoint.display()
            )));
        }

        // Confirm the socket actually speaks CRI v1 (not just "the file
        // exists") — fail-closed with a clear error rather than deferring the
        // first failure to some later, less obvious call site.
        let mut client = self.runtime_client().await;
        let resp = tokio::time::timeout(
            self.config.call_timeout,
            client.version(cri::VersionRequest {
                version: String::new(),
            }),
        )
        .await
        .map_err(|_| {
            WorkerError::ConfigError(format!(
                "CRI runtime at '{}' did not respond to Version within {:?}",
                self.config.endpoint.display(),
                self.config.call_timeout
            ))
        })?
        .map_err(|e| {
            WorkerError::ConfigError(format!(
                "CRI runtime at '{}' rejected Version: {e}",
                self.config.endpoint.display()
            ))
        })?
        .into_inner();

        info!(
            runtime_name = %resp.runtime_name,
            runtime_version = %resp.runtime_version,
            runtime_api_version = %resp.runtime_api_version,
            endpoint = %self.config.endpoint.display(),
            "CRI runtime connected"
        );
        Ok(())
    }

    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        config: &PodSandboxConfig,
        pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>> {
        let sandbox_path = pool_config.runtime_dir.join(&sandbox.id);
        tokio::fs::create_dir_all(&sandbox_path).await?;
        sandbox.sandbox_path = sandbox_path;

        let image = self.resolve_image(annotations);
        let runtime_handler = self.resolve_runtime_handler(annotations);
        let sandbox_config = to_cri_pod_sandbox_config(config, &sandbox.id, annotations);

        info!(
            sandbox_id = %sandbox.id,
            image = %image,
            runtime_handler = %runtime_handler,
            endpoint = %self.config.endpoint.display(),
            "Starting CRI sandbox"
        );

        let mut runtime = self.runtime_client().await;
        let mut images = self.image_client().await;

        let run_resp = runtime
            .run_pod_sandbox(cri::RunPodSandboxRequest {
                config: Some(sandbox_config.clone()),
                runtime_handler: runtime_handler.clone(),
            })
            .await
            .map_err(|e| WorkerError::SandboxCreationFailed(format!("RunPodSandbox failed: {e}")))?
            .into_inner();
        let pod_sandbox_id = run_resp.pod_sandbox_id;

        // Pull the workload image explicitly (kubelet-equivalent behaviour;
        // CreateContainer expects the image to already be present).
        images
            .pull_image(cri::PullImageRequest {
                image: Some(cri::ImageSpec {
                    image: image.clone(),
                    ..Default::default()
                }),
                auth: None,
                sandbox_config: Some(sandbox_config.clone()),
            })
            .await
            .map_err(|e| WorkerError::ImagePullFailed {
                image: image.clone(),
                reason: e.to_string(),
            })?;

        let container_config = to_cri_container_config(&sandbox.id, &image, annotations);
        let create_resp = runtime
            .create_container(cri::CreateContainerRequest {
                pod_sandbox_id: pod_sandbox_id.clone(),
                config: Some(container_config),
                sandbox_config: Some(sandbox_config),
            })
            .await
            .map_err(|e| {
                WorkerError::ContainerCreationFailed(format!("CreateContainer failed: {e}"))
            })?
            .into_inner();
        let container_id = create_resp.container_id;

        runtime
            .start_container(cri::StartContainerRequest {
                container_id: container_id.clone(),
            })
            .await
            .map_err(|e| {
                WorkerError::ContainerCreationFailed(format!("StartContainer failed: {e}"))
            })?;

        let handle = Arc::new(CriHandle {
            sandbox_id: sandbox.id.clone(),
            pod_sandbox_id: pod_sandbox_id.clone(),
            container_id: Some(container_id.clone()),
            image,
            runtime_handler,
        });

        sandbox.mark_ready();
        info!(
            sandbox_id = %sandbox.id,
            pod_sandbox_id = %pod_sandbox_id,
            container_id = %container_id,
            "CRI sandbox started"
        );

        Ok(handle)
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        let Some(handle) = sandbox
            .backend_handle
            .as_ref()
            .and_then(|h| h.as_any().downcast_ref::<CriHandle>())
        else {
            // Never started (or started by a different backend) — nothing to do.
            return Ok(());
        };

        let mut runtime = self.runtime_client().await;
        if let Some(container_id) = &handle.container_id {
            if let Err(e) = runtime
                .stop_container(cri::StopContainerRequest {
                    container_id: container_id.clone(),
                    timeout: 10,
                })
                .await
            {
                warn!(sandbox_id = %sandbox.id, error = %e, "StopContainer failed (continuing)");
            }
        }
        if let Err(e) = runtime
            .stop_pod_sandbox(cri::StopPodSandboxRequest {
                pod_sandbox_id: handle.pod_sandbox_id.clone(),
            })
            .await
        {
            warn!(sandbox_id = %sandbox.id, error = %e, "StopPodSandbox failed (continuing)");
        }
        Ok(())
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        let Some(handle) = sandbox
            .backend_handle
            .as_ref()
            .and_then(|h| h.as_any().downcast_ref::<CriHandle>())
        else {
            return Ok(());
        };

        let mut runtime = self.runtime_client().await;
        if let Some(container_id) = &handle.container_id {
            let _ = runtime
                .remove_container(cri::RemoveContainerRequest {
                    container_id: container_id.clone(),
                })
                .await;
        }
        let _ = runtime
            .remove_pod_sandbox(cri::RemovePodSandboxRequest {
                pod_sandbox_id: handle.pod_sandbox_id.clone(),
            })
            .await;
        Ok(())
    }

    async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
        // CRI sandboxes are ephemeral here too (like oci/nspawn) — recreate
        // rather than reuse in place.
        Ok(false)
    }

    async fn get_pids(&self, _sandbox: &PodSandbox) -> Result<Vec<u32>> {
        // CRI v1 has no standardized typed PID field on `ContainerStatus` (see
        // the module doc). Fail-safe empty result rather than parsing the
        // free-form `info` map, which is not part of the stable contract.
        Ok(Vec::new())
    }

    /// Deliver a composed namespace to a CRI-driven sandbox (#635). See the
    /// module doc for why this is verification-only, not a live hot-attach —
    /// CRI has no generic RPC for that (the containerd-specific NRI mechanism
    /// that could is explicitly deferred, #510).
    async fn deliver_namespace(
        &self,
        _sandbox: &PodSandbox,
        _namespace: hyprstream_vfs::Namespace,
        _subject: hyprstream_vfs::Subject,
        transport: NamespaceTransport,
    ) -> Result<NamespaceDelivery> {
        let NamespaceTransport::BindMount { target } = transport else {
            return Err(WorkerError::Unsupported(format!(
                "cri backend only supports the BindMount namespace transport (CRI's \
                 Mount type is host-path only, declared at CreateContainer time — \
                 there is no virtio-fs/HostImports equivalent), got {transport:?}"
            )));
        };

        if !target.exists() {
            return Err(WorkerError::SandboxCreationFailed(format!(
                "deliver_namespace: bind-mount source {} does not exist",
                target.display()
            )));
        }

        debug!(
            target = %target.display(),
            "cri deliver_namespace: source materialized; wire it into the workload \
             via a hyprstream.io/mount.* annotation before start() — CRI has no \
             hot-attach RPC for an already-running container (NRI, deferred)"
        );

        Ok(NamespaceDelivery::BindMount { target })
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
        if command.is_empty() {
            return Err(WorkerError::ExecFailed("exec_sync: empty command".into()));
        }
        let handle = sandbox
            .backend_handle
            .as_ref()
            .and_then(|h| h.as_any().downcast_ref::<CriHandle>())
            .ok_or_else(|| WorkerError::SandboxInvalidState {
                sandbox_id: sandbox.id.clone(),
                state: "no cri handle".into(),
                expected: "started".into(),
            })?;
        let container_id = handle.container_id.clone().ok_or_else(|| {
            WorkerError::ExecFailed(format!("sandbox {} has no workload container", sandbox.id))
        })?;

        let mut runtime = self.runtime_client().await;
        let resp = tokio::time::timeout(
            Duration::from_secs(timeout_secs),
            runtime.exec_sync(cri::ExecSyncRequest {
                container_id,
                cmd: command.to_vec(),
                timeout: timeout_secs as i64,
            }),
        )
        .await
        .map_err(|_| WorkerError::SandboxTimeout {
            operation: format!("exec_sync in {}", sandbox.id),
            timeout_secs,
        })?
        .map_err(|e| WorkerError::ExecFailed(format!("ExecSync failed: {e}")))?
        .into_inner();

        Ok((resp.exit_code, resp.stdout, resp.stderr))
    }

    async fn update_resources(
        &self,
        sandbox: &PodSandbox,
        resources: &LinuxContainerResources,
    ) -> Result<()> {
        let Some(handle) = sandbox
            .backend_handle
            .as_ref()
            .and_then(|h| h.as_any().downcast_ref::<CriHandle>())
        else {
            return Ok(());
        };
        let Some(container_id) = handle.container_id.clone() else {
            return Ok(());
        };

        let mut runtime = self.runtime_client().await;
        runtime
            .update_container_resources(cri::UpdateContainerResourcesRequest {
                container_id,
                linux: Some(to_cri_resources(resources)),
                windows: None,
                annotations: HashMap::new(),
            })
            .await
            .map_err(|e| {
                WorkerError::ConfigError(format!("UpdateContainerResources failed: {e}"))
            })?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend registry self-registration (#507 / #518) — gated on the `cri` feature
// ─────────────────────────────────────────────────────────────────────────────

// Registered only when compiled with `--features cri`. Mirrors `oci`/`wasm`:
// with the feature off, `cri` simply isn't in the registry, so an explicit
// request fails closed rather than silently downgrading.
//
// `auto_selectable: false` — deliberately explicit-name-only. Auto-selection
// must never silently hand a workload to *some other host's* runtime service
// just because a socket file happens to exist; driving an external runtime is
// an auth-surface decision (#510: "human sign-off"), not something "auto"
// should reach for on its own. It remains reachable via `worker.backend =
// "cri"`.
inventory::submit! {
    crate::runtime::selection::BackendRegistration {
        name: "cri",
        priority: 0,
        auto_selectable: false,
        // CRI containers bind-mount host paths (`Mount.host_path`) declared at
        // CreateContainer time (see `to_cri_container_config`), so a host 9P UDS
        // can be injected the same way oci/nspawn do it (#506) — just at
        // create-time via the `hyprstream.io/mount.*` annotation, not hot-attach.
        injects_9p_socket: true,
        // No FUSE-mount composition here: CRI already owns image pull / rootfs
        // via the external runtime, so there is no per-sandbox tenant-VFS FUSE
        // mount to advertise (unlike oci/nspawn's Model B, #715 — out of scope
        // for the #510 baseline; CRI's image/rootfs handling is intentionally
        // left to containerd/CRI-O, per the issue's design boundary).
        mounts_fuse_vfs: false,
        is_available: CriBackend::registry_is_available,
        construct: |_ctx| {
            Ok(std::sync::Arc::new(CriBackend::new(CriConfig::default()))
                as std::sync::Arc<dyn crate::runtime::SandboxBackend>)
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::client::{KeyValue, PodSandboxConfig};
    use super::*;
    use std::path::PathBuf;

    fn new_pod(id: &str, cfg: &PodSandboxConfig) -> PodSandbox {
        PodSandbox::new(id.to_owned(), cfg, PathBuf::from("/tmp/cri-test"))
    }

    #[test]
    fn backend_type_is_cri() {
        let backend = CriBackend::new(CriConfig::default());
        assert_eq!(backend.backend_type(), "cri");
    }

    #[test]
    fn default_endpoint_is_containerd_socket() {
        std::env::remove_var("HYPRSTREAM_CRI_ENDPOINT");
        let config = CriConfig::default();
        assert_eq!(config.endpoint, PathBuf::from(DEFAULT_CRI_ENDPOINT));
    }

    #[test]
    fn handle_downcasts() {
        let handle: Arc<dyn SandboxHandle> = Arc::new(CriHandle {
            sandbox_id: "abc".into(),
            pod_sandbox_id: "ps-abc".into(),
            container_id: Some("ctr-abc".into()),
            image: "img:latest".into(),
            runtime_handler: String::new(),
        });
        let down = handle.as_any().downcast_ref::<CriHandle>();
        assert!(down.is_some());
        assert_eq!(down.unwrap().sandbox_id, "abc");
        assert_eq!(down.unwrap().pod_sandbox_id, "ps-abc");
    }

    #[test]
    fn image_annotation_overrides_default() {
        let backend = CriBackend::new(CriConfig::default());
        let mut ann = HashMap::new();
        assert_eq!(backend.resolve_image(&ann), DEFAULT_PAUSE_IMAGE);
        ann.insert(ANN_IMAGE.into(), "alpine:3.20".into());
        assert_eq!(backend.resolve_image(&ann), "alpine:3.20");
    }

    #[test]
    fn runtime_class_annotation_overrides_default() {
        let backend = CriBackend::new(CriConfig::default());
        let mut ann = HashMap::new();
        assert_eq!(backend.resolve_runtime_handler(&ann), "");
        ann.insert(ANN_RUNTIME_CLASS.into(), "kata".into());
        assert_eq!(backend.resolve_runtime_handler(&ann), "kata");
    }

    #[test]
    fn mount_spec_conversion() {
        let m = to_cri_mount("/h:/c").unwrap();
        assert_eq!(m.host_path, "/h");
        assert_eq!(m.container_path, "/c");
        assert!(!m.readonly);

        let m = to_cri_mount("/h:/c:ro").unwrap();
        assert!(m.readonly);

        let m = to_cri_mount("/h:/c:rw").unwrap();
        assert!(!m.readonly);

        assert!(to_cri_mount("/h").is_none());
        assert!(to_cri_mount("/h:/c:bogus").is_none());
        assert!(to_cri_mount("").is_none());
    }

    #[test]
    fn resources_are_threaded_not_dropped() {
        let mut res = LinuxContainerResources::default();
        res.cpu_period = 100_000;
        res.cpu_quota = 50_000;
        res.cpu_shares = 512;
        res.memory_limit_in_bytes = 268_435_456;
        res.cpuset_cpus = "0-1".into();

        let cri_res = to_cri_resources(&res);
        assert_eq!(cri_res.cpu_period, 100_000);
        assert_eq!(cri_res.cpu_quota, 50_000);
        assert_eq!(cri_res.cpu_shares, 512);
        assert_eq!(cri_res.memory_limit_in_bytes, 268_435_456);
        assert_eq!(cri_res.cpuset_cpus, "0-1");
    }

    #[test]
    fn pod_sandbox_config_threads_cri_metadata() {
        let mut cfg = PodSandboxConfig::default();
        cfg.metadata.name = "my-pod".to_owned();
        cfg.metadata.namespace = "team-a".to_owned();
        cfg.metadata.uid = "uid-123".to_owned();
        cfg.labels = vec![KeyValue {
            key: "app".into(),
            value: "demo".into(),
        }];

        let mut ann = HashMap::new();
        ann.insert("custom".into(), "value".into());

        let cri_cfg = to_cri_pod_sandbox_config(&cfg, "sb-1", &ann);
        let meta = cri_cfg.metadata.unwrap();
        assert_eq!(meta.name, "my-pod");
        assert_eq!(meta.namespace, "team-a");
        assert_eq!(meta.uid, "uid-123");
        assert_eq!(cri_cfg.hostname, "my-pod");
        assert_eq!(cri_cfg.labels.get("app"), Some(&"demo".to_owned()));
        assert_eq!(cri_cfg.annotations.get("custom"), Some(&"value".to_owned()));
    }

    #[test]
    fn container_config_threads_command_env_and_mounts() {
        let mut ann = HashMap::new();
        ann.insert("hyprstream.io/env.FOO".into(), "bar".into());
        ann.insert("hyprstream.io/mount.data".into(), "/host:/data:ro".into());
        ann.insert(
            ANN_COMMAND.into(),
            "/usr/local/bin/wanix-guest --flag".into(),
        );

        let ctr_cfg = to_cri_container_config("sb-1", "alpine:latest", &ann);
        assert_eq!(ctr_cfg.image.unwrap().image, "alpine:latest");
        assert_eq!(
            ctr_cfg.command,
            vec!["/usr/local/bin/wanix-guest", "--flag"]
        );
        assert!(ctr_cfg
            .envs
            .iter()
            .any(|kv| kv.key == "FOO" && kv.value == "bar"));
        assert!(ctr_cfg
            .envs
            .iter()
            .any(|kv| kv.key == "HYPRSTREAM_INSTANCE" && kv.value == "sb-1"));
        assert_eq!(ctr_cfg.mounts.len(), 1);
        assert_eq!(ctr_cfg.mounts[0].host_path, "/host");
        assert_eq!(ctr_cfg.mounts[0].container_path, "/data");
        assert!(ctr_cfg.mounts[0].readonly);
    }

    #[test]
    fn command_annotation_key_matches_wanix_contract() {
        // Single source of truth: the CRI consumer key must equal the wanix
        // producer key (same wire annotation OciBackend uses).
        assert_eq!(ANN_COMMAND, super::super::wanix_workload::ANN_WANIX_COMMAND);
    }

    // ── deliver_namespace (#635) ──

    #[tokio::test]
    async fn deliver_namespace_rejects_non_bindmount_transport() {
        let backend = CriBackend::new(CriConfig::default());
        let cfg = PodSandboxConfig::default();
        let pod = new_pod("sb-ns", &cfg);
        let ns = hyprstream_vfs::Namespace::new();
        let subject = hyprstream_vfs::Subject::new("sb-ns".to_owned());

        let err = backend
            .deliver_namespace(&pod, ns, subject, NamespaceTransport::HostImports)
            .await
            .unwrap_err();
        assert!(matches!(err, WorkerError::Unsupported(_)));
    }

    #[tokio::test]
    async fn deliver_namespace_rejects_missing_bindmount_source() {
        let backend = CriBackend::new(CriConfig::default());
        let cfg = PodSandboxConfig::default();
        let pod = new_pod("sb-ns2", &cfg);
        let ns = hyprstream_vfs::Namespace::new();
        let subject = hyprstream_vfs::Subject::new("sb-ns2".to_owned());

        let missing = PathBuf::from("/nonexistent/path/for/cri-test-deliver-namespace");
        let err = backend
            .deliver_namespace(
                &pod,
                ns,
                subject,
                NamespaceTransport::BindMount { target: missing },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, WorkerError::SandboxCreationFailed(_)));
    }

    #[tokio::test]
    async fn deliver_namespace_accepts_existing_bindmount_source() {
        let backend = CriBackend::new(CriConfig::default());
        let cfg = PodSandboxConfig::default();
        let pod = new_pod("sb-ns3", &cfg);
        let ns = hyprstream_vfs::Namespace::new();
        let subject = hyprstream_vfs::Subject::new("sb-ns3".to_owned());

        let tmp = tempfile::tempdir().unwrap();
        let delivery = backend
            .deliver_namespace(
                &pod,
                ns,
                subject,
                NamespaceTransport::BindMount {
                    target: tmp.path().to_path_buf(),
                },
            )
            .await
            .unwrap();
        assert!(matches!(delivery, NamespaceDelivery::BindMount { .. }));
    }

    /// Model B (#715): cri does NOT advertise the FUSE tenant-VFS mount
    /// capability — CRI already owns image pull/rootfs via the external
    /// runtime (see the registration's doc comment); this is the negative
    /// sibling to `oci_advertises_fuse_mount_capability_in_real_inventory`.
    #[test]
    fn cri_does_not_advertise_fuse_mount_capability() {
        assert!(crate::runtime::selection::require_fuse_mount_capability("cri").is_err());
    }

    /// `cri` is explicit-name-only: `"auto"` must never pick it, even if it is
    /// the only registered backend, since driving an external runtime is an
    /// auth-surface decision (#510).
    #[test]
    fn cri_is_not_auto_selectable() {
        let regs: Vec<&crate::runtime::selection::BackendRegistration> =
            inventory::iter::<crate::runtime::selection::BackendRegistration>().collect();
        let cri_reg = regs.iter().find(|r| r.name == "cri");
        if let Some(reg) = cri_reg {
            assert!(!reg.auto_selectable);
        }
    }
}
