//! Wanix-guest workload wiring (#506, deliverable 3).
//!
//! This is the OCI/nspawn analog of the kata/virtio-fs path in
//! [`super::sandbox_fs`] (FS-D, #365). Where `sandbox_fs` composes a per-sandbox
//! Subject-scoped VFS [`Namespace`](hyprstream_vfs::Namespace) and serves it over
//! a **vhost-user-fs** socket that a Cloud-Hypervisor guest attaches as a
//! virtio-fs device, this module serves a tenant's Subject-scoped [`Mount`] as a
//! **wire-faithful 9P2000.L server over a plain Unix socket** (PR-A's
//! [`hyprstream_9p::serve_mount_uds`]) and injects that socket into a
//! *bind-mount-capable* sandbox (OCI/nspawn) so the native Wanix guest (PR-B's
//! `workers/wanix-guest`) dials back and binds the export as its namespace root.
//!
//! ## Bidirectional 9P (#708 phase 2)
//!
//! The above is **Direction A** — the host exports its namespace, the guest
//! imports it. This module also wires **Direction B**, the reverse: the native
//! Wanix guest exports its OWN live namespace over a *second, independent* UDS,
//! and the host imports it via the phase-1 socket 9P client
//! [`Remote9pMount`](hyprstream_9p::Remote9pMount) and binds it into the host
//! VFS [`Namespace`] at `/workers/<tenant>/wanix` — giving the host visibility
//! into the guest's task tree.
//!
//! [`inject_9p_socket`] allocates the export path and injects it (a shared
//! export *directory* bind-mount + [`ENV_GUEST_EXPORT_SOCK`], mirroring the
//! Direction-A socket injection). After the sandbox starts,
//! [`import_guest_namespace`] dials the guest's export socket and binds the
//! imported tree. The two directions are **independent** — separate sockets,
//! separate 9P sessions, no shared coupling (#708).
//!
//! **Scope + payoff caveat.** This is wired for the shared-FS UDS backends
//! (OCI/nspawn); a kata guest would export over vsock (follow-up, not here).
//! And native Wanix currently has only [`native::ExecDriver`] — the exported
//! `#task` tree is host-exec processes + binds, not a wasi/VM task tree, until a
//! native wasi driver exists. The *mechanism* (host imports the guest's
//! namespace bidirectionally, over the phase-1 `Remote9pMount`) is the
//! deliverable; what is visible through it will grow as the guest gains drivers.
//!
//! [`native::ExecDriver`]: https://pkg.go.dev/tractor.dev/wanix/native
//!
//! ## The three injection steps (#506 deliverable 3)
//!
//! [`inject_9p_socket`] performs, for one workload:
//!
//! 1. **(a) allocate a per-workload UDS path** under the workload's runtime dir;
//! 2. **(b) spawn the 9P server** — [`hyprstream_9p::serve_mount_uds`] serving
//!    *that tenant's* Subject-scoped [`Mount`] on the socket, as a background
//!    task (the [`UnixListener`] is bound *synchronously* first, so the socket
//!    file exists the moment this returns — no race with sandbox start);
//! 3. **(c) produce the injection annotations** that the sandbox backend already
//!    consumes at its existing mount/env seam:
//!    * `hyprstream.io/mount.9p-sock=<host-sock>:<ctr-sock>:rw` — bind the host
//!      UDS into the container's mount namespace (OCI `--volume`, nspawn
//!      `--bind`);
//!    * `hyprstream.io/mount.wanix-guest=<host-bin>:<ctr-bin>:ro` — bind the
//!      (configurable, [`WanixGuestConfig::guest_bin`]) guest artifact in;
//!    * `hyprstream.io/env.HYPRSTREAM_9P_SOCK=<ctr-sock>` — tell the guest where
//!      to dial;
//!    * `hyprstream.io/command=<ctr-bin>` — run the guest as the workload command
//!      (consumed by [`OciBackend`](super::oci_backend)'s `build_run_args`).
//!
//! The returned [`WanixInjection`] carries both the annotations (a
//! `HashMap<String, String>`, exactly the shape a
//! [`SandboxBackend::start`](super::SandboxBackend::start) receives) and an
//! [`Injected9pServer`] RAII handle that owns the serve task and removes the
//! socket on drop.
//!
//! ## Where the tenant `Mount` + `Subject` come from
//!
//! The Subject-scoped [`Mount`] is a tenant's composed VFS namespace root — the
//! same thing [`SandboxFs::compose`](super::sandbox_fs::SandboxFs) builds for the
//! kata path (image rootfs + injected `/stream`, `/models`, `/deltas`, bound
//! under the sandbox's [`Subject`]). Because [`Namespace`](hyprstream_vfs::Namespace)
//! is not itself an `Arc<dyn Mount>` and the composition source differs per
//! caller, this module takes the `Mount` + `Subject` as **parameters** rather
//! than reaching into a specific composition site: the caller (a worker service
//! placing a tenant workload) passes the tenant's namespace root and principal.
//! This is the seam #506 leaves explicit — see the deliverable-3 note on the PR.
//!
//! ## Fail-closed
//!
//! Injection is gated on the **9P-socket-injection capability** (#506,
//! deliverable 4): [`require_9p_socket_capability`](super::require_9p_socket_capability)
//! must pass for the target backend before any socket is served. A backend that
//! cannot receive a host UDS (kata, wasm) is refused cleanly — never a silent
//! no-op where the guest comes up with no namespace. [`prepare_wanix_workload`]
//! bundles the gate + injection.

#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hyprstream_9p::Remote9pMount;
use hyprstream_vfs::{BindFlag, Mount, Namespace, Subject};
use tokio::net::UnixListener;
use tracing::{debug, info, warn};

use crate::error::{Result, WorkerError};
use crate::runtime::client::KeyValue;

// ─────────────────────────────────────────────────────────────────────────────
// Annotation / env contract (shared with the sandbox backends' existing seam)
// ─────────────────────────────────────────────────────────────────────────────

/// Annotation key prefix: per-variable container environment. Matches the OCI
/// backend's `ANN_ENV_PREFIX` (a wire contract between producer and consumer).
const ANN_ENV_PREFIX: &str = "hyprstream.io/env.";
/// Annotation key prefix: bind-mount spec `host:container[:ro|:rw]`. Matches the
/// OCI backend's `ANN_MOUNT_PREFIX`.
const ANN_MOUNT_PREFIX: &str = "hyprstream.io/mount.";
/// Annotation key: the workload command to run in the sandbox. Consumed by
/// [`OciBackend`](super::oci_backend)'s `build_run_args`. Must equal
/// `oci_backend::ANN_COMMAND` (asserted by a test under `--features oci`).
pub const ANN_WANIX_COMMAND: &str = "hyprstream.io/command";

/// Environment variable the Wanix guest reads for the 9P socket path (see
/// `workers/wanix-guest`). Kept in sync with the guest's `--sock` env fallback.
pub const ENV_9P_SOCK: &str = "HYPRSTREAM_9P_SOCK";

/// Environment variable the Wanix guest reads for the Direction-B **export**
/// socket path — the in-container UDS on which the guest serves its OWN live
/// namespace for the host to import (#708 phase 2). Kept in sync with the
/// guest's `--export-sock` env fallback (`workers/wanix-guest`).
pub const ENV_GUEST_EXPORT_SOCK: &str = "HYPRSTREAM_GUEST_EXPORT_SOCK";

/// File name of the per-workload 9P socket under the workload runtime dir.
const NINEP_SOCKET_NAME: &str = "9p.sock";
/// In-container path the host 9P socket is bind-mounted to.
const CTR_SOCK_PATH: &str = "/run/hyprstream/9p.sock";
/// In-container path the host `wanix-guest` binary is bind-mounted to.
const CTR_GUEST_BIN_PATH: &str = "/usr/local/bin/wanix-guest";

/// Sub-directory under the workload runtime dir shared with the container for
/// the Direction-B guest export. The guest CREATES its export listener inside
/// this dir (a bind-mount source must already exist, and the guest's socket
/// does not yet — so we share the *directory*, not a pre-made socket file, and
/// the socket the guest binds appears here on the host).
const GUEST_EXPORT_DIR_NAME: &str = "export";
/// File name the guest binds its export listener as, inside the shared dir.
const GUEST_EXPORT_SOCKET_NAME: &str = "wanix.sock";
/// In-container path the shared export dir is bind-mounted to.
const CTR_EXPORT_DIR: &str = "/run/hyprstream/export";
/// In-container path of the guest's export socket (inside [`CTR_EXPORT_DIR`]).
const CTR_EXPORT_SOCK_PATH: &str = "/run/hyprstream/export/wanix.sock";

/// Bounded attempts to dial the guest export socket after sandbox start (it
/// appears only once the guest boots and calls `listen(2)`).
const IMPORT_CONNECT_ATTEMPTS: u32 = 20;
/// Base backoff between import-connect attempts.
const IMPORT_CONNECT_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);

/// Env var overriding the host path to the `wanix-guest` build artifact.
const ENV_GUEST_BIN: &str = "HYPRSTREAM_WANIX_GUEST_BIN";
/// Default host path (relative to CWD) of the `wanix-guest` artifact produced by
/// `scripts/build-wanix-guest.sh`.
const DEFAULT_GUEST_BIN: &str = "target/wanix-guest";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Wanix-guest workload.
#[derive(Debug, Clone)]
pub struct WanixGuestConfig {
    /// Host path to the `wanix-guest` binary (a foreign-toolchain build artifact
    /// from `scripts/build-wanix-guest.sh`, *not* a cargo target). Configurable
    /// via `HYPRSTREAM_WANIX_GUEST_BIN`; **never** hardcoded to an absolute path.
    pub guest_bin: PathBuf,
}

impl Default for WanixGuestConfig {
    fn default() -> Self {
        let guest_bin = std::env::var(ENV_GUEST_BIN)
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_GUEST_BIN));
        Self { guest_bin }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Running 9P server handle
// ─────────────────────────────────────────────────────────────────────────────

/// RAII handle for a per-workload 9P server backing an injected socket.
///
/// Owns the background serve task (aborted on drop) and removes the socket file
/// on drop, so tearing a workload down does not leak a listener or a stale
/// socket. The task also exits on its own if the socket errors/closes.
pub struct Injected9pServer {
    /// Host-side path of the 9P Unix socket.
    socket_path: PathBuf,
    /// The background serve task ([`hyprstream_9p`]'s accept loop).
    task: tokio::task::JoinHandle<()>,
}

impl Injected9pServer {
    /// Host-side path of the 9P socket (the bind-mount source).
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Abort the serve task and remove the socket now (idempotent; `Drop` also
    /// does this).
    pub fn shutdown(&self) {
        self.task.abort();
        if let Err(e) = std::fs::remove_file(&self.socket_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                debug!(socket = %self.socket_path.display(), error = %e, "remove 9P socket");
            }
        }
    }
}

impl Drop for Injected9pServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for Injected9pServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Injected9pServer")
            .field("socket_path", &self.socket_path)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Injection result
// ─────────────────────────────────────────────────────────────────────────────

/// The product of [`inject_9p_socket`]: the annotations to place on the
/// workload's sandbox config, plus the running 9P server handle.
#[derive(Debug)]
pub struct WanixInjection {
    /// Annotations to merge into the sandbox's config. Shaped as the backend's
    /// `start` receives them (`HashMap<String, String>`); use
    /// [`WanixInjection::as_key_values`] to fold them into a
    /// [`PodSandboxConfig`](super::client::PodSandboxConfig)'s
    /// `annotations: Vec<KeyValue>` for the pool path.
    pub annotations: HashMap<String, String>,
    /// The live Direction-A 9P server (host exports its namespace to the guest).
    /// Keep it alive for the workload's lifetime; drop it to tear the export down.
    pub server: Injected9pServer,
    /// Host-side path where the guest's Direction-B export socket will appear
    /// once the guest boots and listens (inside the shared export dir bound into
    /// the container). Pass this to [`import_guest_namespace`] **after** the
    /// sandbox is started to import the guest's live namespace.
    pub export_socket_path: PathBuf,
}

impl WanixInjection {
    /// The injection annotations as `Vec<KeyValue>` for merging into a
    /// [`PodSandboxConfig`](super::client::PodSandboxConfig).
    pub fn as_key_values(&self) -> Vec<KeyValue> {
        self.annotations
            .iter()
            .map(|(k, v)| KeyValue {
                key: k.clone(),
                value: v.clone(),
            })
            .collect()
    }

    /// Host-side path of the guest's Direction-B export socket (see
    /// [`WanixInjection::export_socket_path`]).
    pub fn export_socket_path(&self) -> &Path {
        &self.export_socket_path
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Injection
// ─────────────────────────────────────────────────────────────────────────────

/// Allocate a per-workload UDS, serve `mount` (scoped to `subject`) as 9P over
/// it, and build the sandbox injection annotations (#506 deliverable 3, steps
/// a/b/c). See the module docs for the full contract.
///
/// The [`UnixListener`] is bound synchronously before the serve task is spawned,
/// so the socket file exists on return — the bind-mount at sandbox start never
/// races the server coming up.
///
/// This is the mechanical injection; the **capability gate** is
/// [`prepare_wanix_workload`] (or call
/// [`require_9p_socket_capability`](super::require_9p_socket_capability) first).
pub async fn inject_9p_socket(
    mount: Arc<dyn Mount>,
    subject: Subject,
    workload_dir: &Path,
    guest_cfg: &WanixGuestConfig,
) -> Result<WanixInjection> {
    // (a) Allocate the per-workload UDS path under the workload runtime dir.
    tokio::fs::create_dir_all(workload_dir).await.map_err(|e| {
        WorkerError::SandboxCreationFailed(format!(
            "create workload dir {}: {e}",
            workload_dir.display()
        ))
    })?;
    let socket_path = workload_dir.join(NINEP_SOCKET_NAME);
    // Remove any stale socket from a crashed predecessor (bind would EADDRINUSE).
    if let Err(e) = tokio::fs::remove_file(&socket_path).await {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!(socket = %socket_path.display(), error = %e, "remove stale 9P socket");
        }
    }

    // (b) Bind the listener *now* (socket file exists on return), then spawn the
    // 9P serve loop in the background over that tenant's Subject-scoped Mount.
    let listener = UnixListener::bind(&socket_path).map_err(|e| {
        WorkerError::SandboxCreationFailed(format!("bind 9P UDS at {}: {e}", socket_path.display()))
    })?;
    let sock_for_log = socket_path.clone();
    let task = tokio::spawn(async move {
        // `serve_uds` is PR-A's wire-faithful 9P2000.L server; it runs until the
        // socket errors/closes (or the task is aborted on teardown).
        if let Err(e) = hyprstream_9p::Translator::from_mount(
            mount,
            subject,
            Arc::new(hyprstream_9p::DenyAllDecider),
        )
        .serve_uds(listener)
        .await
        {
            warn!(socket = %sock_for_log.display(), error = %e, "9P workload server exited with error");
        }
    });

    // (c) Injection annotations, consumed at the backend's existing mount/env
    // seam (OCI `--volume`/`--env`/command; nspawn `--bind`/`--setenv`).
    let host_sock = socket_path.to_string_lossy().into_owned();
    let host_bin = guest_cfg.guest_bin.to_string_lossy().into_owned();

    let mut annotations = HashMap::new();
    // Bind the host 9P socket into the container's mount namespace (rw: the guest
    // connects and does full-duplex 9P over it).
    annotations.insert(
        format!("{ANN_MOUNT_PREFIX}9p-sock"),
        format!("{host_sock}:{CTR_SOCK_PATH}:rw"),
    );
    // Bind the configurable guest artifact in read-only.
    annotations.insert(
        format!("{ANN_MOUNT_PREFIX}wanix-guest"),
        format!("{host_bin}:{CTR_GUEST_BIN_PATH}:ro"),
    );
    // Point the guest at the in-container socket path.
    annotations.insert(
        format!("{ANN_ENV_PREFIX}{ENV_9P_SOCK}"),
        CTR_SOCK_PATH.to_owned(),
    );

    // ── Direction B (#708 phase 2): guest exports, host imports ──────────────
    //
    // Allocate a shared export DIRECTORY (not a pre-made socket file: the guest
    // is the server here, so it creates the listener — a bind-mount source must
    // already exist, but the guest's socket does not yet). The guest binds its
    // export listener inside this dir; because the dir is bind-mounted into the
    // container, that socket appears on the host at `export_socket_path`, which
    // the host later dials via [`import_guest_namespace`].
    let host_export_dir = workload_dir.join(GUEST_EXPORT_DIR_NAME);
    tokio::fs::create_dir_all(&host_export_dir)
        .await
        .map_err(|e| {
            WorkerError::SandboxCreationFailed(format!(
                "create guest export dir {}: {e}",
                host_export_dir.display()
            ))
        })?;
    let export_socket_path = host_export_dir.join(GUEST_EXPORT_SOCKET_NAME);
    // Remove any stale export socket left by a crashed predecessor guest.
    if let Err(e) = tokio::fs::remove_file(&export_socket_path).await {
        if e.kind() != std::io::ErrorKind::NotFound {
            warn!(socket = %export_socket_path.display(), error = %e, "remove stale guest export socket");
        }
    }
    let host_export_dir_str = host_export_dir.to_string_lossy().into_owned();
    // Bind the shared export dir into the container's mount namespace (rw: the
    // guest binds its listener inside it, the host reads the resulting socket).
    annotations.insert(
        format!("{ANN_MOUNT_PREFIX}wanix-export-dir"),
        format!("{host_export_dir_str}:{CTR_EXPORT_DIR}:rw"),
    );
    // Tell the guest where to export its namespace (in-container socket path).
    annotations.insert(
        format!("{ANN_ENV_PREFIX}{ENV_GUEST_EXPORT_SOCK}"),
        CTR_EXPORT_SOCK_PATH.to_owned(),
    );

    // Run the guest as the workload command (serve-namespace mode: no `--cmd`,
    // which is the reliable path today — see wanix-guest README's known gap).
    annotations.insert(ANN_WANIX_COMMAND.to_owned(), CTR_GUEST_BIN_PATH.to_owned());

    info!(
        socket = %socket_path.display(),
        export_socket = %export_socket_path.display(),
        guest_bin = %guest_cfg.guest_bin.display(),
        "prepared bidirectional Wanix-guest 9P workload injection"
    );

    Ok(WanixInjection {
        annotations,
        server: Injected9pServer { socket_path, task },
        export_socket_path,
    })
}

/// A guest namespace imported into the host VFS namespace (#708 phase 2,
/// Direction B). Holds the [`Remote9pMount`] backing the imported tree so the
/// remote 9P client connection lives as long as this handle; drop it (and
/// [`unmount`](ImportedGuestNamespace::unmount) the prefix) at workload teardown.
pub struct ImportedGuestNamespace {
    mount: Arc<dyn Mount>,
    prefix: String,
}

impl ImportedGuestNamespace {
    /// The host namespace prefix the guest tree is bound at
    /// (`/workers/<tenant>/wanix`).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The imported guest tree as an `Arc<dyn Mount>` (already bound into the
    /// host namespace by [`import_guest_namespace`]; exposed for rebind/teardown).
    pub fn mount(&self) -> Arc<dyn Mount> {
        Arc::clone(&self.mount)
    }

    /// Remove the imported tree from `host_ns` (the teardown counterpart of the
    /// bind [`import_guest_namespace`] performed).
    pub fn unmount(&self, host_ns: &mut Namespace) {
        host_ns.unmount(&self.prefix);
    }
}

impl std::fmt::Debug for ImportedGuestNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedGuestNamespace")
            .field("prefix", &self.prefix)
            .finish()
    }
}

/// Import the guest's live namespace (Direction B, #708 phase 2): dial the
/// guest's export socket with the phase-1 socket 9P client [`Remote9pMount`],
/// and bind the resulting [`Mount`] into `host_ns` at `/workers/<tenant>/wanix`,
/// giving the host visibility into the guest's task tree.
///
/// Call this **after** the sandbox has been started: the export socket appears
/// only once the guest boots and `listen(2)`s, so the dial is retried with
/// bounded backoff ([`IMPORT_CONNECT_ATTEMPTS`]). `export_sock` is the host-side
/// [`WanixInjection::export_socket_path`].
///
/// This is the reverse, independent 9P session of the Direction-A export the
/// host serves in [`inject_9p_socket`] — a separate socket, a separate client,
/// no shared coupling (#708).
pub async fn import_guest_namespace(
    host_ns: &mut Namespace,
    export_sock: &Path,
    tenant: &str,
) -> Result<ImportedGuestNamespace> {
    import_guest_namespace_with(
        host_ns,
        export_sock,
        tenant,
        IMPORT_CONNECT_ATTEMPTS,
        IMPORT_CONNECT_BACKOFF,
    )
    .await
}

/// [`import_guest_namespace`] with explicit retry bounds (the public wrapper
/// uses [`IMPORT_CONNECT_ATTEMPTS`]/[`IMPORT_CONNECT_BACKOFF`]). Kept separate so
/// tests can drive the retry loop without waiting out the production backoff.
async fn import_guest_namespace_with(
    host_ns: &mut Namespace,
    export_sock: &Path,
    tenant: &str,
    attempts: u32,
    backoff: std::time::Duration,
) -> Result<ImportedGuestNamespace> {
    let prefix = format!("/workers/{tenant}/wanix");

    // The guest's listener appears only after it boots — retry with backoff.
    let mut last_err: Option<anyhow::Error> = None;
    let mut mount = None;
    for attempt in 1..=attempts {
        match Remote9pMount::connect_uds(export_sock, "hyprstream", "").await {
            Ok(m) => {
                mount = Some(m);
                break;
            }
            Err(e) => {
                debug!(
                    socket = %export_sock.display(),
                    attempt,
                    error = %e,
                    "guest export socket not ready yet; retrying"
                );
                last_err = Some(e);
                tokio::time::sleep(backoff * attempt).await;
            }
        }
    }
    let mount = mount.ok_or_else(|| {
        WorkerError::SandboxCreationFailed(format!(
            "could not import guest namespace from {} after {attempts} attempts: {}",
            export_sock.display(),
            last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown error".into())
        ))
    })?;

    let mount: Arc<dyn Mount> = Arc::new(mount);
    host_ns
        .bind_mount(&prefix, Arc::clone(&mount), BindFlag::Replace)
        .map_err(|e| {
            WorkerError::SandboxCreationFailed(format!(
                "bind imported guest namespace at {prefix}: {e}"
            ))
        })?;

    info!(prefix = %prefix, socket = %export_sock.display(), "imported guest Wanix namespace into host VFS");

    Ok(ImportedGuestNamespace { mount, prefix })
}

/// Fail-closed convenience: verify `backend_type` can inject a 9P socket, then
/// perform the injection (#506, deliverables 3 + 4 together).
///
/// The capability gate runs **before** any socket is bound or served, so an
/// incapable backend (kata, wasm) is refused without side effects — never a
/// silent no-op where the guest boots with no namespace.
pub async fn prepare_wanix_workload(
    backend_type: &str,
    mount: Arc<dyn Mount>,
    subject: Subject,
    workload_dir: &Path,
    guest_cfg: &WanixGuestConfig,
) -> Result<WanixInjection> {
    super::require_9p_socket_capability(backend_type)?;
    inject_9p_socket(mount, subject, workload_dir, guest_cfg).await
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_vfs::injected::{SyntheticMount, SyntheticNode};
    use std::os::unix::net::UnixStream as StdUnixStream;

    fn tenant_mount() -> Arc<dyn Mount> {
        // A trivial Subject-scoped tree stands in for a composed tenant namespace.
        Arc::new(SyntheticMount::new(
            SyntheticNode::dir().with_child("hello", SyntheticNode::file(b"hi\n".to_vec())),
        ))
    }

    #[test]
    fn guest_bin_is_configurable_not_hardcoded() {
        // Default is the build-script artifact path (relative), never an absolute
        // hardcode.
        std::env::remove_var(ENV_GUEST_BIN);
        let cfg = WanixGuestConfig::default();
        assert_eq!(cfg.guest_bin, PathBuf::from(DEFAULT_GUEST_BIN));
        assert!(
            cfg.guest_bin.is_relative(),
            "default must not be an absolute hardcode"
        );

        std::env::set_var(ENV_GUEST_BIN, "/opt/custom/wanix-guest");
        let cfg = WanixGuestConfig::default();
        assert_eq!(cfg.guest_bin, PathBuf::from("/opt/custom/wanix-guest"));
        std::env::remove_var(ENV_GUEST_BIN);
    }

    #[tokio::test]
    async fn inject_creates_socket_and_expected_annotations() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = WanixGuestConfig {
            guest_bin: PathBuf::from("/host/build/wanix-guest"),
        };
        let inj = inject_9p_socket(tenant_mount(), Subject::anonymous(), dir.path(), &cfg)
            .await
            .unwrap();

        // (a) socket exists on return (bound synchronously) — the bind-mount
        // source is ready before any sandbox start.
        let sock = dir.path().join(NINEP_SOCKET_NAME);
        assert!(
            sock.exists(),
            "9P socket must exist immediately after inject"
        );
        assert_eq!(inj.server.socket_path(), sock.as_path());

        // (c) env points the guest at the in-container socket.
        assert_eq!(
            inj.annotations
                .get(&format!("{ANN_ENV_PREFIX}{ENV_9P_SOCK}"))
                .map(String::as_str),
            Some(CTR_SOCK_PATH)
        );
        // socket bind: host abs path → container path, rw.
        let sock_mount = inj
            .annotations
            .get(&format!("{ANN_MOUNT_PREFIX}9p-sock"))
            .expect("9p-sock mount annotation");
        assert_eq!(
            sock_mount,
            &format!("{}:{CTR_SOCK_PATH}:rw", sock.display())
        );
        // guest binary bind uses the CONFIGURED host path (read-only).
        assert_eq!(
            inj.annotations
                .get(&format!("{ANN_MOUNT_PREFIX}wanix-guest"))
                .map(String::as_str),
            Some(format!("/host/build/wanix-guest:{CTR_GUEST_BIN_PATH}:ro").as_str())
        );
        // command runs the guest.
        assert_eq!(
            inj.annotations.get(ANN_WANIX_COMMAND).map(String::as_str),
            Some(CTR_GUEST_BIN_PATH)
        );

        // Direction B (#708 phase 2): shared export dir bound rw, export env set,
        // and the host-side export socket path exposed under the shared dir.
        let export_dir = dir.path().join(GUEST_EXPORT_DIR_NAME);
        assert!(
            export_dir.is_dir(),
            "shared export dir must be created on inject"
        );
        assert_eq!(
            inj.annotations
                .get(&format!("{ANN_MOUNT_PREFIX}wanix-export-dir")),
            Some(&format!("{}:{CTR_EXPORT_DIR}:rw", export_dir.display()))
        );
        assert_eq!(
            inj.annotations
                .get(&format!("{ANN_ENV_PREFIX}{ENV_GUEST_EXPORT_SOCK}"))
                .map(String::as_str),
            Some(CTR_EXPORT_SOCK_PATH)
        );
        assert_eq!(
            inj.export_socket_path(),
            export_dir.join(GUEST_EXPORT_SOCKET_NAME).as_path()
        );
    }

    #[tokio::test]
    async fn import_guest_namespace_binds_and_reads() {
        // Stand up a local wire-faithful 9P server over a Mount — this stands in
        // for the guest's Direction-B export — then import it into a host
        // namespace and read a file back through the bound prefix. Exercises the
        // full host-import mechanism (phase-1 Remote9pMount → bind_mount) in
        // process, no sandbox boot.
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("guest-export.sock");
        let listener = UnixListener::bind(&sock).unwrap();
        let server = tokio::spawn(async move {
            let _ = hyprstream_9p::Translator::from_mount(
                tenant_mount(),
                Subject::anonymous(),
                Arc::new(hyprstream_9p::DenyAllDecider),
            )
            .serve_uds(listener)
            .await;
        });

        let mut host_ns = Namespace::new();
        let imported = import_guest_namespace(&mut host_ns, &sock, "t1")
            .await
            .unwrap();
        assert_eq!(imported.prefix(), "/workers/t1/wanix");

        let data = host_ns
            .cat("/workers/t1/wanix/hello", &Subject::anonymous())
            .await
            .unwrap();
        assert_eq!(
            data, b"hi\n",
            "host must read the guest's file over the imported 9P mount"
        );

        imported.unmount(&mut host_ns);
        assert!(
            !host_ns.mount_prefixes().contains(&"/workers/t1/wanix"),
            "unmount must remove the imported prefix"
        );
        server.abort();
    }

    #[tokio::test]
    async fn import_fails_when_no_guest_export() {
        // No server ever listens on the socket → bounded retries exhaust and the
        // import fails closed (never binds a dead mount).
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("never-appears.sock");
        let mut host_ns = Namespace::new();
        let err = import_guest_namespace_with(
            &mut host_ns,
            &sock,
            "t1",
            3,
            std::time::Duration::from_millis(1),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("could not import guest namespace"),
            "got: {err}"
        );
        assert!(
            host_ns.mount_prefixes().is_empty(),
            "nothing must be bound on failure"
        );
    }

    #[tokio::test]
    async fn injected_socket_is_a_live_9p_server() {
        // The spawned task actually serves 9P: a client that connects and sends
        // a Tversion gets a well-formed Rversion back. Proves (b) wired the real
        // PR-A server to the tenant Mount, not just created an empty socket.
        let dir = tempfile::tempdir().unwrap();
        let inj = inject_9p_socket(
            tenant_mount(),
            Subject::anonymous(),
            dir.path(),
            &WanixGuestConfig::default(),
        )
        .await
        .unwrap();

        let sock = inj.server.socket_path().to_path_buf();
        // Connect + do a minimal 9P2000.L version handshake.
        let reply = tokio::task::spawn_blocking(move || {
            use std::io::{Read, Write};
            let mut c = StdUnixStream::connect(&sock).unwrap();
            // Tversion: size[4] type[1]=100 tag[2]=0xFFFF msize[4] version[s]
            let version = b"9P2000.L";
            let mut msg = Vec::new();
            let body_len = 4 + 1 + 2 + 4 + 2 + version.len();
            msg.extend_from_slice(&(body_len as u32).to_le_bytes());
            msg.push(100); // Tversion
            msg.extend_from_slice(&0xFFFFu16.to_le_bytes());
            msg.extend_from_slice(&8192u32.to_le_bytes());
            msg.extend_from_slice(&(version.len() as u16).to_le_bytes());
            msg.extend_from_slice(version);
            c.write_all(&msg).unwrap();

            let mut hdr = [0u8; 5];
            c.read_exact(&mut hdr).unwrap();
            hdr[4] // message type byte
        })
        .await
        .unwrap();

        // Rversion == 101 (Tversion 100 + 1).
        assert_eq!(reply, 101, "expected Rversion from the injected 9P server");
    }

    #[tokio::test]
    async fn dropping_server_removes_socket() {
        let dir = tempfile::tempdir().unwrap();
        let inj = inject_9p_socket(
            tenant_mount(),
            Subject::anonymous(),
            dir.path(),
            &WanixGuestConfig::default(),
        )
        .await
        .unwrap();
        let sock = inj.server.socket_path().to_path_buf();
        assert!(sock.exists());
        drop(inj);
        assert!(
            !sock.exists(),
            "socket must be removed when the server is dropped"
        );
    }

    #[tokio::test]
    async fn prepare_fails_closed_on_incapable_backend() {
        // kata/wasm/unknown cannot inject a host UDS → refuse before serving.
        let dir = tempfile::tempdir().unwrap();
        let err = prepare_wanix_workload(
            "kata",
            tenant_mount(),
            Subject::anonymous(),
            dir.path(),
            &WanixGuestConfig::default(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("cannot inject a 9P socket"),
            "got: {err}"
        );
        // And no socket was created (fail-closed BEFORE any side effect).
        assert!(!dir.path().join(NINEP_SOCKET_NAME).exists());
    }

    #[cfg(feature = "nspawn")]
    #[tokio::test]
    async fn prepare_succeeds_on_capable_backend() {
        // nspawn is registered (under the `nspawn` feature) and capable → injection proceeds.
        let dir = tempfile::tempdir().unwrap();
        let inj = prepare_wanix_workload(
            "nspawn",
            tenant_mount(),
            Subject::anonymous(),
            dir.path(),
            &WanixGuestConfig::default(),
        )
        .await
        .unwrap();
        assert!(inj.server.socket_path().exists());
        assert!(inj.annotations.contains_key(ANN_WANIX_COMMAND));
    }
}
