//! K8sBackend — Kubernetes **API-server** sandbox isolation (#781, epic #778)
//!
//! Unlike [`super::cri_backend::CriBackend`] (a client of a single node's
//! containerd/CRI-O **socket** — "shadow pods" that are same-node-only and
//! invisible to Kubernetes), `K8sBackend` is a kube-rs client of the cluster
//! **API server**: it creates real `Pod`/`Job` objects that are
//! scheduler-placed, quota-accounted, GPU-accounted, `kubectl`-visible and
//! cross-node. hyprstream stays the *client/orchestrator* here — it never
//! implements the kubelet-facing CRI contract (a #778 non-goal).
//!
//! ## Wire mapping
//!
//! | `SandboxBackend`  | Kubernetes API operation(s)                                   |
//! |-------------------|---------------------------------------------------------------|
//! | `start`           | `POST` a `Pod` (or `Job`) → watch until the Pod is `Running`   |
//! | `stop`            | `DELETE` the object with a grace period                        |
//! | `destroy`         | `DELETE` the object with grace period `0` (force)              |
//! | `exec_sync`       | `pods/exec` (websocket `AttachedProcess`, blocking capture)    |
//! | `get_pids`        | not exposed by the API server → `Ok(vec![])` (same as `cri`)   |
//! | `log_stream`      | `pods/log` (inherent helper; not a trait method)               |
//!
//! ## PodSpec mapping ([`to_k8s_podspec`])
//!
//! The capnp [`PodSandboxConfig`]/annotation surface (shared with the
//! `cri`/`oci` backends) lowers to a `PodSpec`: image + command + env from the
//! `hyprstream.io/*` annotations, `LinuxContainerResources` → container
//! `resources` (CPU millicores from `quota/period`, memory bytes), the sandbox
//! `securityContext` (run-as-user/group, privileged, read-only rootfs), and
//! `runtimeClassName` passthrough so a Kata-backed `RuntimeClass` still gives
//! VM isolation — now *scheduler-placed* rather than node-local.
//!
//! ## Origin markers / two-writer guard (#778 threat table)
//!
//! Every emitted object carries [`MANAGED_BY_LABEL`], the sandbox id
//! ([`SANDBOX_LABEL`]), and this backend instance's identity
//! ([`INSTANCE_LABEL`]) for audit/provenance. `stop`/`destroy` refuse to delete
//! an object that does not carry the expected managed-by + sandbox-id markers,
//! so a restarted backend can reap its own sandbox workload without adopting a
//! different sandbox's objects.
//!
//! ## Out of scope here
//!
//! - **Namespace delivery** (`deliver_namespace`) is K4b (#793, needs K3): the
//!   trait default (fail-closed "not implemented") is deliberately left in
//!   place.
//! - **Placement mapping** (node affinity / GPU / topology) is K4c.
//! - The "runs on a kind cluster" e2e acceptance is a follow-up; this change is
//!   the backend core + the offline-testable `to_k8s_podspec` mapping.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, info, warn};

use hyprstream_k8s::k8s_openapi::api::batch::v1::{Job, JobSpec};
use hyprstream_k8s::k8s_openapi::api::core::v1::{
    Container, EnvVar, Pod, PodSpec, PodTemplateSpec, ResourceRequirements, SecurityContext,
};
use hyprstream_k8s::k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use hyprstream_k8s::k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use hyprstream_k8s::kube::api::{Api, AttachParams, DeleteParams, LogParams, PostParams};
use hyprstream_k8s::kube::runtime::wait::{await_condition, conditions};
use hyprstream_k8s::kube::Client;
use hyprstream_k8s::MANAGED_BY_LABEL;

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::backend::{SandboxBackend, SandboxHandle};
use super::client::{KeyValue, PodSandboxConfig};
use super::sandbox::PodSandbox;

// ─────────────────────────────────────────────────────────────────────────────
// Annotation / label vocabulary
// ─────────────────────────────────────────────────────────────────────────────

/// Annotation key: image reference to run (same key `oci`/`cri` use, so a
/// workload's annotations are portable across all three container backends).
const ANN_IMAGE: &str = "hyprstream.io/oci-image";
/// Annotation key: workload command (whitespace-split into argv).
const ANN_COMMAND: &str = "hyprstream.io/command";
/// Annotation key prefix: per-variable container environment.
const ANN_ENV_PREFIX: &str = "hyprstream.io/env.";
/// Annotation key: `RuntimeClass` name (empty = cluster default). Passthrough so
/// a Kata-backed `RuntimeClass` still yields VM isolation — scheduler-placed.
const ANN_RUNTIME_CLASS: &str = "hyprstream.io/runtime-class";
/// Annotation key: target namespace override (else pod metadata → config default).
const ANN_NAMESPACE: &str = "hyprstream.io/k8s-namespace";
/// Annotation key: emit a `Job` (`"job"`) instead of a bare `Pod` (`"pod"`,
/// the default). A `Job` gets retry/backoff + completion tracking from the Job
/// controller; a bare `Pod` is the minimal, watch-until-`Running` shape.
const ANN_WORKLOAD_KIND: &str = "hyprstream.io/k8s-workload-kind";

/// Label value for [`MANAGED_BY_LABEL`] on objects this backend emits.
///
/// Deliberately distinct from `hyprstream-k8s`'s `MANAGED_BY_VALUE`
/// (`"hyprstream-operator"`): the operator (K5b) reconciles CRDs; this backend
/// is the worker data plane creating ephemeral Pods/Jobs. Keeping the values
/// distinct means the two writers never mistake each other's objects for their
/// own.
const MANAGED_BY_VALUE: &str = "hyprstream-workers";
/// Label key: the identity of the [`K8sBackend`] instance that created an
/// object (the two-writer guard anchor).
const INSTANCE_LABEL: &str = "hyprstream.io/instance";
/// Label key: the hyprstream sandbox id an object was created for.
const SANDBOX_LABEL: &str = "hyprstream.io/sandbox-id";

/// In-cluster ServiceAccount token path (presence ⇒ we're running in a Pod).
const IN_CLUSTER_TOKEN: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
/// Default namespace when none is resolvable from config/annotations.
const DEFAULT_NAMESPACE: &str = "default";
/// Default workload image when no `hyprstream.io/oci-image` annotation is set.
const DEFAULT_IMAGE: &str = "registry.k8s.io/pause:3.10";

/// Which Kubernetes object this backend emits for a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadKind {
    /// A bare `Pod` (default) — watched until `Running`.
    Pod,
    /// A `Job` wrapping the same PodSpec.
    Job,
}

impl WorkloadKind {
    fn from_annotations(annotations: &HashMap<String, String>) -> Self {
        match annotations.get(ANN_WORKLOAD_KIND).map(String::as_str) {
            Some("job") | Some("Job") => Self::Job,
            _ => Self::Pod,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Kubernetes API-server backend.
#[derive(Debug, Clone)]
pub struct K8sConfig {
    /// Namespace used when a workload does not set [`ANN_NAMESPACE`] and its
    /// pod metadata carries no namespace.
    pub default_namespace: String,
    /// `runtimeClassName` used when a workload sets no [`ANN_RUNTIME_CLASS`].
    /// Empty selects the cluster default (no `runtimeClassName` on the Pod).
    pub default_runtime_class: String,
    /// Image used when no `hyprstream.io/oci-image` annotation is supplied.
    pub default_image: String,
    /// Timeout for the start watch (Pod → `Running`) and lifecycle calls.
    pub call_timeout: Duration,
    /// Grace period (seconds) applied by `stop`. `destroy` always forces `0`.
    pub stop_grace_secs: i64,
}

impl Default for K8sConfig {
    fn default() -> Self {
        Self {
            default_namespace: std::env::var("HYPRSTREAM_K8S_NAMESPACE")
                .unwrap_or_else(|_| DEFAULT_NAMESPACE.to_owned()),
            default_runtime_class: std::env::var("HYPRSTREAM_K8S_RUNTIME_CLASS")
                .unwrap_or_default(),
            default_image: std::env::var("HYPRSTREAM_K8S_IMAGE")
                .unwrap_or_else(|_| DEFAULT_IMAGE.to_owned()),
            call_timeout: Duration::from_secs(120),
            stop_grace_secs: 30,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-specific state stored on each `PodSandbox`.
#[derive(Debug, Clone)]
pub struct K8sHandle {
    /// hyprstream sandbox id (matches `PodSandbox::id`).
    pub sandbox_id: String,
    /// Kubernetes namespace the object lives in.
    pub namespace: String,
    /// Object name (equals `sandbox_id`).
    pub name: String,
    /// Whether a `Pod` or a `Job` was created.
    kind: WorkloadKind,
}

impl SandboxHandle for K8sHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// Kubernetes API-server sandbox backend: emits real Pods/Jobs via kube-rs.
pub struct K8sBackend {
    config: K8sConfig,
    /// This backend instance's identity, stamped on every emitted object and
    /// retained as audit/provenance metadata.
    instance_id: String,
    /// Lazily-initialised kube client. Building a `Client` needs an async
    /// context (it reads in-cluster/kubeconfig and constructs an HTTPS stack),
    /// so — exactly like [`super::cri_backend::CriBackend`]'s channel — it is
    /// deferred out of the sync `construct` fn to the first async trait call.
    client: tokio::sync::OnceCell<Client>,
}

impl std::fmt::Debug for K8sBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("K8sBackend")
            .field("config", &self.config)
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

impl K8sBackend {
    pub fn new(config: K8sConfig) -> Self {
        Self {
            config,
            instance_id: uuid::Uuid::new_v4().to_string(),
            client: tokio::sync::OnceCell::new(),
        }
    }

    /// Get (or lazily build) the kube client, mapping failure to a clean
    /// fail-closed config error rather than deferring to a later call site.
    async fn client(&self) -> Result<&Client> {
        self.client
            .get_or_try_init(|| async {
                Client::try_default().await.map_err(|e| {
                    WorkerError::ConfigError(format!(
                        "could not construct a Kubernetes client (in-cluster \
                         ServiceAccount or kubeconfig): {e}"
                    ))
                })
            })
            .await
    }

    /// Registry availability probe: is there any usable cluster credential —
    /// an in-cluster ServiceAccount token, or a kubeconfig? Cheap and
    /// synchronous; actual reachability is confirmed by `initialize`.
    fn registry_is_available() -> bool {
        credentials_present()
    }

    /// Resolve the target namespace: annotation → pod metadata → config default.
    fn resolve_namespace(
        &self,
        config: &PodSandboxConfig,
        annotations: &HashMap<String, String>,
    ) -> String {
        annotations
            .get(ANN_NAMESPACE)
            .filter(|s| !s.is_empty())
            .cloned()
            .or_else(|| Some(config.metadata.namespace.clone()).filter(|s: &String| !s.is_empty()))
            .unwrap_or_else(|| self.config.default_namespace.clone())
    }

    /// Resolve the workload image: annotation override → configured default.
    fn resolve_image(&self, annotations: &HashMap<String, String>) -> String {
        annotations
            .get(ANN_IMAGE)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| self.config.default_image.clone())
    }

    /// Resolve the `runtimeClassName`: annotation override → configured default
    /// (empty ⇒ cluster default, i.e. no `runtimeClassName` on the Pod).
    fn resolve_runtime_class(&self, annotations: &HashMap<String, String>) -> Option<String> {
        annotations
            .get(ANN_RUNTIME_CLASS)
            .cloned()
            .unwrap_or_else(|| self.config.default_runtime_class.clone())
            .into_option_nonempty()
    }

    /// Two-writer guard: confirm the live object carries hyprstream's
    /// managed-by marker and the expected sandbox id before we delete it.
    /// [`INSTANCE_LABEL`] is audit metadata only, so a restarted backend can
    /// still reap objects for the same sandbox.
    fn labels_are_ours(&self, labels: Option<&BTreeMap<String, String>>, sandbox_id: &str) -> bool {
        let Some(labels) = labels else { return false };
        labels.get(MANAGED_BY_LABEL).map(String::as_str) == Some(MANAGED_BY_VALUE)
            && labels.get(SANDBOX_LABEL).map(String::as_str) == Some(sandbox_id)
    }

    /// Fetch `count` bytes of a running Pod's logs (inherent helper — the
    /// `SandboxBackend` trait has no logs method). Best-effort: returns the log
    /// text, or a clear error if the Pod is gone/unreadable.
    pub async fn log_stream(&self, sandbox: &PodSandbox) -> Result<String> {
        let handle = downcast(sandbox)?;
        let pods: Api<Pod> = Api::namespaced(self.client().await?.clone(), &handle.namespace);
        pods.logs(&handle.name, &LogParams::default())
            .await
            .map_err(|e| WorkerError::ExecFailed(format!("fetching pod logs failed: {e}")))
    }
}

/// Are any Kubernetes credentials present on this host (in-cluster token or a
/// kubeconfig)? Fail-closed: false ⇒ an explicit `k8s` request errors cleanly.
fn credentials_present() -> bool {
    if std::path::Path::new(IN_CLUSTER_TOKEN).exists() {
        return true;
    }
    if std::env::var_os("KUBECONFIG").is_some() {
        return true;
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".kube").join("config").exists();
    }
    false
}

/// Downcast a sandbox's opaque handle to our concrete [`K8sHandle`].
fn downcast(sandbox: &PodSandbox) -> Result<&K8sHandle> {
    sandbox
        .backend_handle()
        .and_then(|h| h.as_any().downcast_ref::<K8sHandle>())
        .ok_or_else(|| WorkerError::SandboxInvalidState {
            sandbox_id: sandbox.id.clone(),
            state: "no k8s handle".into(),
            expected: "started".into(),
        })
}

/// Small `String → Option<String>` (None when empty) helper for optional fields.
trait NonEmptyOption {
    fn into_option_nonempty(self) -> Option<String>;
}
impl NonEmptyOption for String {
    fn into_option_nonempty(self) -> Option<String> {
        if self.is_empty() {
            None
        } else {
            Some(self)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// capnp CRI-shaped types → Kubernetes PodSpec
// ─────────────────────────────────────────────────────────────────────────────

fn kv_to_btree(kvs: &[KeyValue]) -> BTreeMap<String, String> {
    kvs.iter()
        .map(|kv| (kv.key.clone(), kv.value.clone()))
        .collect()
}

/// Build the container `resources` from `LinuxContainerResources`. CPU is
/// derived from `quota/period` as millicores; memory from the byte limit. Both
/// `requests` and `limits` are set to the same values (Guaranteed QoS) so the
/// scheduler reserves exactly what the workload is capped at. Returns `None`
/// when no limits are expressed (leave it to namespace `LimitRange` defaults).
fn to_k8s_resources(config: &PodSandboxConfig) -> Option<ResourceRequirements> {
    let res = &config.linux.resources;
    let mut map: BTreeMap<String, Quantity> = BTreeMap::new();

    if res.cpu_period > 0 && res.cpu_quota > 0 {
        // millicores = (quota / period) * 1000, rounded up so we never
        // under-request a fractional core.
        let millicores =
            (res.cpu_quota as i128 * 1000 + res.cpu_period as i128 - 1) / res.cpu_period as i128;
        if millicores > 0 {
            map.insert("cpu".to_owned(), Quantity(format!("{millicores}m")));
        }
    }
    if res.memory_limit_in_bytes > 0 {
        map.insert(
            "memory".to_owned(),
            Quantity(res.memory_limit_in_bytes.to_string()),
        );
    }

    if map.is_empty() {
        return None;
    }
    Some(ResourceRequirements {
        limits: Some(map.clone()),
        requests: Some(map),
        ..Default::default()
    })
}

/// Build the container `securityContext` from the sandbox security context.
/// Only fields the workload actually asserts are set (root/`0` is treated as
/// "unset" so we never *force* a Pod to run as root, matching CRI's default-0
/// convention); returns `None` when nothing is asserted.
fn to_k8s_security_context(config: &PodSandboxConfig) -> Option<SecurityContext> {
    let sc = &config.linux.security_context;
    let run_as_user = (sc.run_as_user > 0).then_some(sc.run_as_user);
    let run_as_group = (sc.run_as_group > 0).then_some(sc.run_as_group);
    let privileged = sc.privileged.then_some(true);
    let read_only_root_filesystem = sc.readonly_rootfs.then_some(true);

    if run_as_user.is_none()
        && run_as_group.is_none()
        && privileged.is_none()
        && read_only_root_filesystem.is_none()
    {
        return None;
    }
    Some(SecurityContext {
        run_as_user,
        run_as_group,
        privileged,
        read_only_root_filesystem,
        ..Default::default()
    })
}

/// Container env: the injected instance marker plus every `hyprstream.io/env.*`
/// annotation, in a deterministic (sorted) order.
fn to_k8s_env(sandbox_id: &str, annotations: &HashMap<String, String>) -> Vec<EnvVar> {
    let mut env = vec![EnvVar {
        name: "HYPRSTREAM_INSTANCE".to_owned(),
        value: Some(sandbox_id.to_owned()),
        ..Default::default()
    }];
    let mut extra: Vec<(String, String)> = annotations
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(ANN_ENV_PREFIX)
                .filter(|name| !name.is_empty())
                .map(|name| (name.to_owned(), v.clone()))
        })
        .collect();
    extra.sort();
    env.extend(extra.into_iter().map(|(name, value)| EnvVar {
        name,
        value: Some(value),
        ..Default::default()
    }));
    env
}

/// Translate the capnp [`PodSandboxConfig`] + resolved annotations into a
/// Kubernetes [`PodSpec`] with a single `workload` container.
///
/// `restart_policy` is `Never`: hyprstream owns the sandbox lifecycle (a Job
/// wrapper, when requested, layers its own backoff on top). Fields with no
/// #781-baseline need (DNS/host aliases/volumes/probes/affinity — affinity is
/// K4c) are left at their defaults; nothing this backend claims to support is
/// silently dropped.
pub fn to_k8s_podspec(
    config: &PodSandboxConfig,
    sandbox_id: &str,
    annotations: &HashMap<String, String>,
    default_image: &str,
) -> PodSpec {
    let image = annotations
        .get(ANN_IMAGE)
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| default_image.to_owned());

    let command: Option<Vec<String>> = annotations.get(ANN_COMMAND).map(|cmd| {
        cmd.split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    });

    let runtime_class_name = annotations
        .get(ANN_RUNTIME_CLASS)
        .cloned()
        .and_then(NonEmptyOption::into_option_nonempty);

    let container = Container {
        name: "workload".to_owned(),
        image: Some(image),
        command,
        env: Some(to_k8s_env(sandbox_id, annotations)),
        resources: to_k8s_resources(config),
        security_context: to_k8s_security_context(config),
        ..Default::default()
    };

    PodSpec {
        containers: vec![container],
        restart_policy: Some("Never".to_owned()),
        runtime_class_name,
        ..Default::default()
    }
}

/// Origin-marker labels stamped on every emitted object: the config labels plus
/// the managed-by / instance / sandbox-id markers the two-writer guard checks.
fn object_labels(
    config: &PodSandboxConfig,
    sandbox_id: &str,
    instance_id: &str,
) -> BTreeMap<String, String> {
    let mut labels = kv_to_btree(&config.labels);
    labels.insert(MANAGED_BY_LABEL.to_owned(), MANAGED_BY_VALUE.to_owned());
    labels.insert(INSTANCE_LABEL.to_owned(), instance_id.to_owned());
    labels.insert(SANDBOX_LABEL.to_owned(), sandbox_id.to_owned());
    labels
}

fn object_meta(
    config: &PodSandboxConfig,
    sandbox_id: &str,
    namespace: &str,
    instance_id: &str,
) -> ObjectMeta {
    ObjectMeta {
        name: Some(sandbox_id.to_owned()),
        namespace: Some(namespace.to_owned()),
        labels: Some(object_labels(config, sandbox_id, instance_id)),
        ..Default::default()
    }
}

fn build_pod(
    config: &PodSandboxConfig,
    sandbox_id: &str,
    namespace: &str,
    instance_id: &str,
    default_image: &str,
    annotations: &HashMap<String, String>,
) -> Pod {
    Pod {
        metadata: object_meta(config, sandbox_id, namespace, instance_id),
        spec: Some(to_k8s_podspec(
            config,
            sandbox_id,
            annotations,
            default_image,
        )),
        ..Default::default()
    }
}

fn build_job(
    config: &PodSandboxConfig,
    sandbox_id: &str,
    namespace: &str,
    instance_id: &str,
    default_image: &str,
    annotations: &HashMap<String, String>,
) -> Job {
    let labels = object_labels(config, sandbox_id, instance_id);
    Job {
        metadata: object_meta(config, sandbox_id, namespace, instance_id),
        spec: Some(JobSpec {
            // Do not let the Job controller retry indefinitely; the sandbox
            // lifecycle is hyprstream's, one attempt is the sensible default.
            backoff_limit: Some(0),
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels),
                    ..Default::default()
                }),
                spec: Some(to_k8s_podspec(
                    config,
                    sandbox_id,
                    annotations,
                    default_image,
                )),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[async_trait]
impl SandboxBackend for K8sBackend {
    fn backend_type(&self) -> &'static str {
        "k8s"
    }

    fn is_available(&self) -> bool {
        credentials_present()
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        if !self.is_available() {
            return Err(WorkerError::ConfigError(
                "no Kubernetes credentials found (no in-cluster ServiceAccount \
                 token, no KUBECONFIG, no ~/.kube/config); the k8s backend needs \
                 API-server access"
                    .to_owned(),
            ));
        }
        // Confirm the API server is actually reachable (fail-closed) rather
        // than deferring the first failure to `start`.
        let client = self.client().await?;
        let version = tokio::time::timeout(self.config.call_timeout, client.apiserver_version())
            .await
            .map_err(|_| {
                WorkerError::ConfigError(format!(
                    "Kubernetes API server did not respond within {:?}",
                    self.config.call_timeout
                ))
            })?
            .map_err(|e| {
                WorkerError::ConfigError(format!("Kubernetes API server unreachable: {e}"))
            })?;
        info!(
            git_version = %version.git_version,
            platform = %version.platform,
            "connected to Kubernetes API server"
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

        let namespace = self.resolve_namespace(config, annotations);
        let image = self.resolve_image(annotations);
        let kind = WorkloadKind::from_annotations(annotations);
        let client = self.client().await?.clone();

        info!(
            sandbox_id = %sandbox.id,
            namespace = %namespace,
            image = %image,
            ?kind,
            runtime_class = ?self.resolve_runtime_class(annotations),
            "creating Kubernetes workload"
        );

        match kind {
            WorkloadKind::Pod => {
                let pod = build_pod(
                    config,
                    &sandbox.id,
                    &namespace,
                    &self.instance_id,
                    &image,
                    annotations,
                );
                let pods: Api<Pod> = Api::namespaced(client, &namespace);
                pods.create(&PostParams::default(), &pod)
                    .await
                    .map_err(|e| {
                        WorkerError::SandboxCreationFailed(format!("Pod create failed: {e}"))
                    })?;

                // Watch until Running (fail-closed on timeout).
                let running = await_condition(pods, &sandbox.id, conditions::is_pod_running());
                tokio::time::timeout(self.config.call_timeout, running)
                    .await
                    .map_err(|_| WorkerError::SandboxTimeout {
                        operation: format!("pod {} → Running", sandbox.id),
                        timeout_secs: self.config.call_timeout.as_secs(),
                    })?
                    .map_err(|e| {
                        WorkerError::SandboxCreationFailed(format!(
                            "waiting for pod {} to run failed: {e}",
                            sandbox.id
                        ))
                    })?;
            }
            WorkloadKind::Job => {
                let job = build_job(
                    config,
                    &sandbox.id,
                    &namespace,
                    &self.instance_id,
                    &image,
                    annotations,
                );
                let jobs: Api<Job> = Api::namespaced(client, &namespace);
                jobs.create(&PostParams::default(), &job)
                    .await
                    .map_err(|e| {
                        WorkerError::SandboxCreationFailed(format!("Job create failed: {e}"))
                    })?;
                // A Job's pod is scheduled asynchronously by the Job controller;
                // we do not block on Running here (the Job owns completion).
            }
        }

        let handle = Arc::new(K8sHandle {
            sandbox_id: sandbox.id.clone(),
            namespace: namespace.clone(),
            name: sandbox.id.clone(),
            kind,
        });
        sandbox.mark_ready();
        info!(sandbox_id = %sandbox.id, namespace = %namespace, "Kubernetes workload created");
        Ok(handle)
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        self.delete(sandbox, self.config.stop_grace_secs).await
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        // Force delete (grace 0).
        self.delete(sandbox, 0).await
    }

    async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
        // API-server workloads are ephemeral objects — recreate, never reuse
        // an old Pod/Job in place (same stance as oci/cri).
        Ok(false)
    }

    async fn get_pids(&self, _sandbox: &PodSandbox) -> Result<Vec<u32>> {
        // Host PIDs are a node-local concept the API server does not expose;
        // fail-safe empty result rather than fabricating one (same as `cri`).
        Ok(Vec::new())
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
        let handle = downcast(sandbox)?;
        if handle.kind != WorkloadKind::Pod {
            return Err(WorkerError::ExecFailed(
                "exec_sync is only supported against Pod workloads (a Job's pod \
                 name is controller-assigned, not the sandbox id)"
                    .into(),
            ));
        }
        let pods: Api<Pod> = Api::namespaced(self.client().await?.clone(), &handle.namespace);

        let ap = AttachParams::default()
            .container("workload")
            .stdout(true)
            .stderr(true)
            .stdin(false);

        let fut = async {
            let mut attached = pods
                .exec(&handle.name, command.iter().cloned(), &ap)
                .await
                .map_err(|e| WorkerError::ExecFailed(format!("pod exec failed: {e}")))?;

            let stdout = read_all(attached.stdout()).await;
            let stderr = read_all(attached.stderr()).await;

            // The exit status arrives on the process-status channel once the
            // command finishes; `take_status` yields the *future* that resolves
            // to it. Map a non-zero/absent status conservatively.
            let code = match attached.take_status() {
                Some(status_fut) => status_fut.await.map_or(0, |s| exit_code_from_status(&s)),
                None => 0,
            };
            let _ = attached.join().await;
            Ok::<(i32, Vec<u8>, Vec<u8>), WorkerError>((code, stdout, stderr))
        };

        tokio::time::timeout(Duration::from_secs(timeout_secs), fut)
            .await
            .map_err(|_| WorkerError::SandboxTimeout {
                operation: format!("exec_sync in {}", sandbox.id),
                timeout_secs,
            })?
    }
}

impl K8sBackend {
    /// Shared delete path for `stop`/`destroy`, guarded by the two-writer
    /// origin-marker check: never delete an object we did not create.
    async fn delete(&self, sandbox: &PodSandbox, grace_secs: i64) -> Result<()> {
        let Ok(handle) = downcast(sandbox) else {
            // Never started (or started by a different backend) — nothing to do.
            return Ok(());
        };
        let client = self.client().await?.clone();
        let dp = DeleteParams {
            grace_period_seconds: Some(grace_secs.max(0) as u32),
            ..Default::default()
        };

        match handle.kind {
            WorkloadKind::Pod => {
                let pods: Api<Pod> = Api::namespaced(client, &handle.namespace);
                match pods.get_opt(&handle.name).await {
                    Ok(Some(pod)) => {
                        if !self.labels_are_ours(pod.metadata.labels.as_ref(), &handle.sandbox_id) {
                            return Err(WorkerError::ConfigError(format!(
                                "refusing to delete pod {}/{}: it does not carry this \
                                 sandbox's origin markers (two-writer guard)",
                                handle.namespace, handle.name
                            )));
                        }
                        if let Err(e) = pods.delete(&handle.name, &dp).await {
                            warn!(sandbox_id = %sandbox.id, error = %e, "Pod delete failed (continuing)");
                        }
                    }
                    Ok(None) => debug!(sandbox_id = %sandbox.id, "pod already gone"),
                    Err(e) => {
                        warn!(sandbox_id = %sandbox.id, error = %e, "pod get before delete failed");
                    }
                }
            }
            WorkloadKind::Job => {
                let jobs: Api<Job> = Api::namespaced(client, &handle.namespace);
                match jobs.get_opt(&handle.name).await {
                    Ok(Some(job)) => {
                        if !self.labels_are_ours(job.metadata.labels.as_ref(), &handle.sandbox_id) {
                            return Err(WorkerError::ConfigError(format!(
                                "refusing to delete job {}/{}: it does not carry this \
                                 sandbox's origin markers (two-writer guard)",
                                handle.namespace, handle.name
                            )));
                        }
                        // Propagation=Background so the Job's pods are reaped too.
                        let dp = DeleteParams {
                            propagation_policy: Some(
                                hyprstream_k8s::kube::api::PropagationPolicy::Background,
                            ),
                            ..dp.clone()
                        };
                        if let Err(e) = jobs.delete(&handle.name, &dp).await {
                            warn!(sandbox_id = %sandbox.id, error = %e, "Job delete failed (continuing)");
                        }
                    }
                    Ok(None) => debug!(sandbox_id = %sandbox.id, "job already gone"),
                    Err(e) => {
                        warn!(sandbox_id = %sandbox.id, error = %e, "job get before delete failed");
                    }
                }
            }
        }
        Ok(())
    }
}

/// Drain an optional `AttachedProcess` byte stream to a `Vec<u8>`, tolerating
/// absence (stdout/stderr not requested or already closed). kube's attached
/// readers implement tokio's `AsyncRead`.
async fn read_all<S>(stream: Option<S>) -> Vec<u8>
where
    S: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let Some(mut stream) = stream else {
        return Vec::new();
    };
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf).await;
    buf
}

/// Extract a numeric exit code from the `pods/exec` process status object. The
/// non-zero exit surfaces as a `NonZeroExitCode` cause carrying the code; a
/// `Success` status (or an unparseable one) maps to `0`.
fn exit_code_from_status(
    status: &hyprstream_k8s::k8s_openapi::apimachinery::pkg::apis::meta::v1::Status,
) -> i32 {
    if status.status.as_deref() == Some("Success") {
        return 0;
    }
    status
        .details
        .as_ref()
        .and_then(|d| d.causes.as_ref())
        .and_then(|causes| {
            causes
                .iter()
                .find(|c| c.reason.as_deref() == Some("ExitCode"))
        })
        .and_then(|c| c.message.as_ref())
        .and_then(|m| m.parse::<i32>().ok())
        .unwrap_or(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend registry self-registration (#507 / #518) — gated on the `k8s` feature
// ─────────────────────────────────────────────────────────────────────────────

// Registered only when compiled with `--features k8s`. Mirrors `cri`/`oci`/
// `wasm`: with the feature off, `k8s` simply isn't in the registry, so an
// explicit request fails closed rather than silently downgrading.
//
// `auto_selectable: false` — deliberately explicit-name-only, the same
// auth-surface rationale as `cri` (#510/#778): handing a workload to a cluster
// (scheduler-placed, cross-node, quota-charged) must never be something `"auto"`
// reaches for just because a kubeconfig happens to exist. It stays reachable via
// `worker.backend = "k8s"`.
inventory::submit! {
    crate::runtime::selection::BackendRegistration {
        name: "k8s",
        priority: 0,
        auto_selectable: false,
        // The composed tenant namespace is delivered as Pod volumes/mounts at
        // create time (K4b, #793) — not by bind-mounting a host UDS into a
        // shared mount namespace (the API-server pod may land on another node).
        // So this backend does NOT advertise host-UDS 9P-socket injection.
        injects_9p_socket: false,
        // No host FUSE-mount composition: the cluster (kubelet + CSI/image) owns
        // the pod's rootfs, exactly as `cri` leaves rootfs to the external
        // runtime.
        mounts_fuse_vfs: false,
        is_available: K8sBackend::registry_is_available,
        construct: |_ctx| {
            Ok(std::sync::Arc::new(K8sBackend::new(K8sConfig::default()))
                as std::sync::Arc<dyn crate::runtime::SandboxBackend>)
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn cfg() -> PodSandboxConfig {
        PodSandboxConfig::default()
    }

    #[test]
    fn backend_type_is_k8s() {
        let b = K8sBackend::new(K8sConfig::default());
        assert_eq!(b.backend_type(), "k8s");
    }

    #[test]
    fn default_namespace_and_image() {
        std::env::remove_var("HYPRSTREAM_K8S_NAMESPACE");
        std::env::remove_var("HYPRSTREAM_K8S_IMAGE");
        let c = K8sConfig::default();
        assert_eq!(c.default_namespace, DEFAULT_NAMESPACE);
        assert_eq!(c.default_image, DEFAULT_IMAGE);
    }

    #[test]
    fn podspec_maps_image_command_and_env() {
        let mut ann = HashMap::new();
        ann.insert(ANN_IMAGE.into(), "alpine:3.20".into());
        ann.insert(ANN_COMMAND.into(), "/bin/sh -c echo".into());
        ann.insert("hyprstream.io/env.FOO".into(), "bar".into());

        let spec = to_k8s_podspec(&cfg(), "sb-1", &ann, DEFAULT_IMAGE);
        assert_eq!(spec.restart_policy.as_deref(), Some("Never"));
        let c = &spec.containers[0];
        assert_eq!(c.name, "workload");
        assert_eq!(c.image.as_deref(), Some("alpine:3.20"));
        assert_eq!(
            c.command.as_ref().unwrap(),
            &vec!["/bin/sh".to_owned(), "-c".to_owned(), "echo".to_owned()]
        );
        let env = c.env.as_ref().unwrap();
        assert!(env
            .iter()
            .any(|e| e.name == "HYPRSTREAM_INSTANCE" && e.value.as_deref() == Some("sb-1")));
        assert!(env
            .iter()
            .any(|e| e.name == "FOO" && e.value.as_deref() == Some("bar")));
    }

    #[test]
    fn podspec_defaults_image_when_no_annotation() {
        let spec = to_k8s_podspec(&cfg(), "sb-2", &HashMap::new(), DEFAULT_IMAGE);
        assert_eq!(spec.containers[0].image.as_deref(), Some(DEFAULT_IMAGE));
        // No command annotation ⇒ image entrypoint (None), not an empty argv.
        assert!(spec.containers[0].command.is_none());
    }

    #[test]
    fn podspec_passes_runtime_class_through() {
        let mut ann = HashMap::new();
        ann.insert(ANN_RUNTIME_CLASS.into(), "kata".into());
        let spec = to_k8s_podspec(&cfg(), "sb-3", &ann, DEFAULT_IMAGE);
        assert_eq!(spec.runtime_class_name.as_deref(), Some("kata"));

        // Empty runtime-class ⇒ cluster default (no runtimeClassName).
        let mut ann2 = HashMap::new();
        ann2.insert(ANN_RUNTIME_CLASS.into(), "".into());
        let spec2 = to_k8s_podspec(&cfg(), "sb-3b", &ann2, DEFAULT_IMAGE);
        assert!(spec2.runtime_class_name.is_none());
    }

    #[test]
    fn resources_map_cpu_millicores_and_memory() {
        let mut c = cfg();
        c.linux.resources.cpu_period = 100_000;
        c.linux.resources.cpu_quota = 50_000; // 0.5 core → 500m
        c.linux.resources.memory_limit_in_bytes = 268_435_456; // 256Mi bytes

        let spec = to_k8s_podspec(&c, "sb-4", &HashMap::new(), DEFAULT_IMAGE);
        let res = spec.containers[0].resources.as_ref().unwrap();
        let limits = res.limits.as_ref().unwrap();
        assert_eq!(limits.get("cpu").unwrap().0, "500m");
        assert_eq!(limits.get("memory").unwrap().0, "268435456");
        // Guaranteed QoS: requests mirror limits.
        assert_eq!(res.requests.as_ref().unwrap().get("cpu").unwrap().0, "500m");
    }

    #[test]
    fn no_resources_when_unspecified() {
        let spec = to_k8s_podspec(&cfg(), "sb-5", &HashMap::new(), DEFAULT_IMAGE);
        assert!(spec.containers[0].resources.is_none());
    }

    #[test]
    fn security_context_maps_only_asserted_fields() {
        let mut c = cfg();
        c.linux.security_context.run_as_user = 1000;
        c.linux.security_context.readonly_rootfs = true;

        let spec = to_k8s_podspec(&c, "sb-6", &HashMap::new(), DEFAULT_IMAGE);
        let sc = spec.containers[0].security_context.as_ref().unwrap();
        assert_eq!(sc.run_as_user, Some(1000));
        assert_eq!(sc.read_only_root_filesystem, Some(true));
        // Not asserted ⇒ left unset (never forced).
        assert!(sc.privileged.is_none());
        assert!(sc.run_as_group.is_none());
    }

    #[test]
    fn security_context_none_when_nothing_asserted() {
        // run_as_user == 0 is treated as unset (never force root).
        let spec = to_k8s_podspec(&cfg(), "sb-7", &HashMap::new(), DEFAULT_IMAGE);
        assert!(spec.containers[0].security_context.is_none());
    }

    #[test]
    fn object_carries_origin_markers() {
        let mut c = cfg();
        c.labels = vec![KeyValue {
            key: "app".into(),
            value: "demo".into(),
        }];
        let pod = build_pod(
            &c,
            "sb-8",
            "team-a",
            "inst-xyz",
            DEFAULT_IMAGE,
            &HashMap::new(),
        );
        assert_eq!(pod.metadata.name.as_deref(), Some("sb-8"));
        assert_eq!(pod.metadata.namespace.as_deref(), Some("team-a"));
        let labels = pod.metadata.labels.as_ref().unwrap();
        assert_eq!(
            labels.get(MANAGED_BY_LABEL).map(String::as_str),
            Some(MANAGED_BY_VALUE)
        );
        assert_eq!(
            labels.get(INSTANCE_LABEL).map(String::as_str),
            Some("inst-xyz")
        );
        assert_eq!(labels.get(SANDBOX_LABEL).map(String::as_str), Some("sb-8"));
        assert_eq!(labels.get("app").map(String::as_str), Some("demo"));
    }

    #[test]
    fn job_wraps_the_same_podspec_and_labels_template() {
        let job = build_job(
            &cfg(),
            "sb-9",
            "default",
            "inst-1",
            DEFAULT_IMAGE,
            &HashMap::new(),
        );
        let spec = job.spec.as_ref().unwrap();
        assert_eq!(spec.backoff_limit, Some(0));
        let tmpl = &spec.template;
        // The pod template carries the origin markers too, so the Job's pods
        // are recognizable as ours.
        let tmpl_labels = tmpl.metadata.as_ref().unwrap().labels.as_ref().unwrap();
        assert_eq!(
            tmpl_labels.get(INSTANCE_LABEL).map(String::as_str),
            Some("inst-1")
        );
        assert_eq!(
            tmpl.spec.as_ref().unwrap().restart_policy.as_deref(),
            Some("Never")
        );
    }

    #[test]
    fn workload_kind_from_annotations() {
        assert_eq!(
            WorkloadKind::from_annotations(&HashMap::new()),
            WorkloadKind::Pod
        );
        let mut ann = HashMap::new();
        ann.insert(ANN_WORKLOAD_KIND.into(), "job".into());
        assert_eq!(WorkloadKind::from_annotations(&ann), WorkloadKind::Job);
    }

    #[test]
    fn two_writer_guard_rejects_foreign_sandbox_labels() {
        let b = K8sBackend::new(K8sConfig::default());
        let sandbox_id = "sb-restart";
        // Our own labels pass.
        let mut ours = BTreeMap::new();
        ours.insert(MANAGED_BY_LABEL.to_owned(), MANAGED_BY_VALUE.to_owned());
        ours.insert(SANDBOX_LABEL.to_owned(), sandbox_id.to_owned());
        ours.insert(INSTANCE_LABEL.to_owned(), b.instance_id.clone());
        assert!(b.labels_are_ours(Some(&ours), sandbox_id));

        // A restarted backend has a different instance id, but the same
        // managed-by/sandbox-id ownership markers still pass.
        let mut restarted = ours.clone();
        restarted.insert(INSTANCE_LABEL.to_owned(), "someone-else".to_owned());
        assert!(b.labels_are_ours(Some(&restarted), sandbox_id));

        // A different sandbox id fails.
        let mut theirs = ours.clone();
        theirs.insert(SANDBOX_LABEL.to_owned(), "sb-other".to_owned());
        assert!(!b.labels_are_ours(Some(&theirs), sandbox_id));

        // Missing managed-by fails; no labels at all fails.
        let mut no_mgr = BTreeMap::new();
        no_mgr.insert(SANDBOX_LABEL.to_owned(), sandbox_id.to_owned());
        no_mgr.insert(INSTANCE_LABEL.to_owned(), b.instance_id.clone());
        assert!(!b.labels_are_ours(Some(&no_mgr), sandbox_id));
        assert!(!b.labels_are_ours(None, sandbox_id));
    }

    #[test]
    fn exit_code_parsing() {
        use hyprstream_k8s::k8s_openapi::apimachinery::pkg::apis::meta::v1::{
            Status, StatusCause, StatusDetails,
        };
        let success = Status {
            status: Some("Success".to_owned()),
            ..Default::default()
        };
        assert_eq!(exit_code_from_status(&success), 0);

        let failed = Status {
            status: Some("Failure".to_owned()),
            details: Some(StatusDetails {
                causes: Some(vec![StatusCause {
                    reason: Some("ExitCode".to_owned()),
                    message: Some("7".to_owned()),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(exit_code_from_status(&failed), 7);
    }

    /// `k8s` is explicit-name-only: `"auto"` must never pick it, even as the
    /// only registered backend — handing a workload to a cluster is an
    /// auth-surface decision (#781/#778), mirroring `cri`.
    #[test]
    fn k8s_is_not_auto_selectable() {
        let reg = inventory::iter::<crate::runtime::selection::BackendRegistration>()
            .find(|r| r.name == "k8s")
            .expect("k8s registered under the k8s feature");
        assert!(!reg.auto_selectable);
        assert!(!reg.injects_9p_socket);
        assert!(!reg.mounts_fuse_vfs);
    }
}
