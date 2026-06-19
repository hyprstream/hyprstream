//! KataBackend — Kata Containers VM-based sandbox isolation
//!
//! Implements `SandboxBackend` using Kata's `Hypervisor` trait for full VM
//! isolation.  Supports Cloud Hypervisor and Dragonball hypervisors.

use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use kata_hypervisor::ch::CloudHypervisor;
use kata_hypervisor::device::DeviceType;
#[cfg(feature = "dragonball")]
use kata_hypervisor::dragonball::Dragonball;
use kata_hypervisor::{Hypervisor, ShareFsConfig, ShareFsDevice};
use kata_types::config::hypervisor::{Hypervisor as HypervisorConfig, RootlessUser};
use kata_types::rootless;

use crate::config::{HypervisorType, ImageConfig, PoolConfig};
use crate::error::{Result, WorkerError};
use crate::image::RafsStore;

use super::backend::{SandboxBackend, SandboxHandle};
use super::client::PodSandboxConfig;
use super::sandbox::PodSandbox;
use super::sandbox_fs::{SandboxFs, SandboxFsServer, VFS_SOCKET_NAME};
use super::virtiofs::SandboxVirtiofs;
use hyprstream_vfs::{Subject, SyntheticNode};

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Kata-specific state stored on each `PodSandbox`.
pub struct KataHandle {
    /// The Kata hypervisor handle for VM lifecycle management.
    pub hypervisor: Arc<dyn Hypervisor>,
    /// Path to the VM API socket.
    pub api_socket: PathBuf,
    /// VirtioFS daemon serving RAFS to this VM (if an image was mounted).
    pub virtiofs_daemon: Option<Arc<SandboxVirtiofs>>,
    /// Path to the virtiofs socket.
    pub virtiofs_socket: Option<PathBuf>,
    /// Per-sandbox composed VFS server (FS-D, #365). When set, the guest's
    /// filesystem is the hyprstream VFS (rootfs + injected mounts) served by
    /// `hyprstream-vfs-server` rather than the external `nydusd` RAFS share.
    /// Held for the VM's lifetime so the serving thread + injected registries
    /// outlive the guest; dropped on sandbox teardown.
    pub vfs_server: Option<SandboxFsServer>,
}

impl std::fmt::Debug for KataHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KataHandle")
            .field("api_socket", &self.api_socket)
            .field("virtiofs_socket", &self.virtiofs_socket)
            .field("vfs_socket", &self.vfs_server.as_ref().map(SandboxFsServer::socket_path))
            .finish_non_exhaustive()
    }
}

impl SandboxHandle for KataHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// Kata Containers sandbox backend.
pub struct KataBackend {
    image_config: ImageConfig,
    rafs_store: Arc<RafsStore>,
}

impl KataBackend {
    pub fn new(image_config: ImageConfig, rafs_store: Arc<RafsStore>) -> Self {
        Self {
            image_config,
            rafs_store,
        }
    }

    /// Build a `HypervisorConfig` from `PoolConfig`.
    fn build_hypervisor_config(pool_config: &PoolConfig) -> HypervisorConfig {
        let mut config = HypervisorConfig::default();

        config.path = if pool_config.hypervisor_path.as_os_str().is_empty() {
            match pool_config.hypervisor {
                HypervisorType::CloudHypervisor => {
                    which::which("cloud-hypervisor")
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|e| {
                            tracing::warn!("Failed to find cloud-hypervisor in PATH: {}", e);
                            "cloud-hypervisor".to_owned()
                        })
                }
                #[cfg(feature = "dragonball")]
                HypervisorType::Dragonball => String::new(),
            }
        } else {
            pool_config.hypervisor_path.to_string_lossy().to_string()
        };

        config.boot_info.kernel = pool_config.kernel_path.to_string_lossy().to_string();
        config.boot_info.image = pool_config.vm_image.to_string_lossy().to_string();
        config.cpu_info.default_vcpus = pool_config.vm_cpus as f32;
        config.cpu_info.default_maxvcpus = pool_config.vm_cpus;
        config.memory_info.default_memory =
            u32::try_from(pool_config.vm_memory_mb).unwrap_or(u32::MAX);

        if rootless::is_rootless() {
            let uid = nix::unistd::getuid().as_raw();
            let gid = nix::unistd::getgid().as_raw();
            let username = std::env::var("USER").unwrap_or_else(|_| "user".to_owned());
            let groups = nix::unistd::getgroups()
                .map(|gs| gs.into_iter().map(nix::unistd::Gid::as_raw).collect())
                .unwrap_or_else(|_| vec![gid]);

            config.security_info.rootless = true;
            config.security_info.rootless_user = Some(RootlessUser {
                uid,
                gid,
                groups,
                user_name: username,
            });

            tracing::debug!(uid, gid, "Configured rootless user for hypervisor");
        }

        config
    }

    /// Create a hypervisor instance for the given sandbox.
    async fn create_hypervisor(
        pool_config: &PoolConfig,
        sandbox: &PodSandbox,
        api_socket: &Path,
        virtiofs_socket: &Path,
    ) -> Result<Arc<dyn Hypervisor>> {
        let config = Self::build_hypervisor_config(pool_config);

        let hypervisor: Arc<dyn Hypervisor> = match pool_config.hypervisor {
            HypervisorType::CloudHypervisor => {
                let ch = CloudHypervisor::new();
                ch.set_hypervisor_config(config).await;
                tracing::debug!(
                    sandbox_id = %sandbox.id,
                    api_socket = %api_socket.display(),
                    virtiofs_socket = %virtiofs_socket.display(),
                    "Created Cloud Hypervisor instance"
                );
                Arc::new(ch)
            }
            #[cfg(feature = "dragonball")]
            HypervisorType::Dragonball => {
                let db = Dragonball::new();
                db.set_hypervisor_config(config).await;
                tracing::debug!(
                    sandbox_id = %sandbox.id,
                    "Created Dragonball hypervisor instance"
                );
                Arc::new(db)
            }
        };

        Ok(hypervisor)
    }

    /// Attach a vhost-user-fs share to the hypervisor as a ShareFs device.
    ///
    /// This closes the gap noted in the FS spike (the virtiofs socket was created
    /// but never wired into the VM): it adds the existing vhost-user-fs socket as
    /// a Cloud Hypervisor `ShareFs` (virtio-fs) device via the **embedded**
    /// runtime-rs hypervisor crate (#343 = embed), so the guest can mount the
    /// share. The same path is used whether the socket is served by `nydusd`
    /// (RAFS image rootfs) or by hyprstream's own `hyprstream-vfs-server`
    /// (the VFS down-adapter).
    ///
    /// # Ordering
    ///
    /// Must be called **after `prepare_vm`** (which sets the hypervisor's
    /// `vm_path`) and **before `start_vm`**. When the VM is not yet running, the
    /// CH backend queues the device into its `pending_devices` and folds it into
    /// the `VmConfig.fs` at boot — no virtiofsd is spawned by the crate; CH
    /// connects to the socket we hand it. An **absolute** `sock_path` is used so
    /// CH does not re-root it under the sandbox dir.
    ///
    /// `fs_type` must be the literal `"virtio-fs"` or the CH backend rejects the
    /// device.
    async fn attach_share_fs(
        hypervisor: &Arc<dyn Hypervisor>,
        sandbox_id: &str,
        socket_path: &Path,
        mount_tag: &str,
    ) -> Result<()> {
        let sock_path = socket_path.to_str().ok_or_else(|| {
            WorkerError::SandboxCreationFailed("virtiofs socket path contains invalid UTF-8".into())
        })?;

        let config = ShareFsConfig {
            // CH never reads host_shared_path/options/mount_config — the daemon
            // that serves `sock_path` owns the exported tree. Only the socket,
            // tag, type, and queue geometry matter for the boot-time attach.
            fs_type: "virtio-fs".to_owned(),
            sock_path: sock_path.to_owned(),
            mount_tag: mount_tag.to_owned(),
            queue_num: 1,
            queue_size: 1024,
            ..Default::default()
        };
        let device = ShareFsDevice::new(&format!("share-fs-{sandbox_id}"), &config);

        hypervisor
            .add_device(DeviceType::ShareFs(device))
            .await
            .map_err(|e| {
                WorkerError::VmStartFailed(format!("failed to attach vhost-user-fs share: {e}"))
            })?;

        tracing::info!(
            sandbox_id = %sandbox_id,
            socket = %socket_path.display(),
            mount_tag = %mount_tag,
            "Attached vhost-user-fs ShareFs device to hypervisor"
        );
        Ok(())
    }

    /// Annotation key carrying the sandbox's policy principal (the [`Subject`]
    /// at the VFS Mount boundary, #353/#319/#328). When absent, the sandbox id is
    /// used as a stable per-sandbox principal so isolation still holds.
    const SUBJECT_ANNOTATION: &'static str = "hyprstream.io/subject";

    /// Resolve the [`Subject`] this sandbox's VFS is served under.
    fn sandbox_subject(sandbox: &PodSandbox, annotations: &HashMap<String, String>) -> Subject {
        annotations
            .get(Self::SUBJECT_ANNOTATION)
            .filter(|s| !s.is_empty())
            .map(Subject::new)
            .unwrap_or_else(|| Subject::new(sandbox.id.clone()))
    }

    /// Compose the per-sandbox VFS namespace (rootfs + injected mounts) and serve
    /// it over a per-sandbox Unix socket (FS-D, #365).
    ///
    /// Returns the running [`SandboxFsServer`] whose socket the caller attaches to
    /// CH. The namespace is forked per sandbox and bound to `subject`, so it
    /// exposes only this sandbox's rootfs + injected paths — never another
    /// sandbox's tree.
    ///
    /// `serve` blocks, so it runs on a dedicated thread inside `serve_on`; we
    /// pass the current tokio runtime handle for the down-adapter's async→sync
    /// bridge. Compose is CPU/IO-bound (RAFS load), so we run it on a blocking
    /// thread to avoid stalling the async runtime.
    async fn compose_and_serve_vfs(
        &self,
        sandbox: &PodSandbox,
        image_id: &str,
        subject: Subject,
    ) -> Result<SandboxFsServer> {
        let socket_path = sandbox.sandbox_path().join(VFS_SOCKET_NAME);
        let rt = tokio::runtime::Handle::current();
        let rafs_store = self.rafs_store.clone();
        let image_id = image_id.to_owned();
        let sandbox_dir = sandbox.sandbox_path().clone();

        // RAFS load + overlay setup is blocking; do it off the async executor.
        tokio::task::spawn_blocking(move || {
            let fs = SandboxFs::compose(
                &rafs_store,
                &image_id,
                &sandbox_dir,
                subject,
                // Models / deltas injected listings: empty mount points for now
                // (the worker runtime populates them per tenant). The mount
                // points exist so the guest sees /models and /deltas.
                SyntheticNode::dir(),
                SyntheticNode::dir(),
            )?;
            fs.serve_on(socket_path, rt)
        })
        .await
        .map_err(|e| WorkerError::SandboxCreationFailed(format!("VFS compose task join: {e}")))?
    }

    /// Generate cloud-init ISO for a sandbox.
    async fn generate_cloud_init_iso(sandbox: &PodSandbox, iso_path: &Path) -> Result<()> {
        let sandbox_runtime_dir = iso_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("ISO path has no parent directory"))?;

        let hostname = if sandbox.metadata.name.is_empty() {
            sandbox.id.clone()
        } else {
            sandbox.metadata.name.clone()
        };

        let user_data = format!(
            "#cloud-config\nhostname: {hostname}\nusers:\n  - name: root\n    lock_passwd: false\nwrite_files:\n  - path: /etc/sandbox-id\n    content: {id}\nruncmd:\n  - echo \"Sandbox {id} initialized\"\n",
            hostname = hostname,
            id = sandbox.id,
        );

        let user_data_path = sandbox_runtime_dir.join("user-data");
        tokio::fs::write(&user_data_path, user_data).await?;

        let meta_data = format!(
            "instance-id: {id}\nlocal-hostname: {hostname}\n",
            id = sandbox.id,
            hostname = hostname,
        );

        let meta_data_path = sandbox_runtime_dir.join("meta-data");
        tokio::fs::write(&meta_data_path, meta_data).await?;

        let iso_path_str = iso_path
            .to_str()
            .ok_or_else(|| WorkerError::CloudInitFailed("ISO path contains invalid UTF-8".into()))?;
        let user_data_str = user_data_path
            .to_str()
            .ok_or_else(|| WorkerError::CloudInitFailed("User data path contains invalid UTF-8".into()))?;
        let meta_data_str = meta_data_path
            .to_str()
            .ok_or_else(|| WorkerError::CloudInitFailed("Meta data path contains invalid UTF-8".into()))?;

        let status = tokio::process::Command::new("genisoimage")
            .args([
                "-output", iso_path_str, "-volid", "cidata", "-joliet", "-rock",
                user_data_str, meta_data_str,
            ])
            .status()
            .await
            .map_err(|e| WorkerError::CloudInitFailed(format!("failed to run genisoimage: {e}")))?;

        if !status.success() {
            return Err(WorkerError::CloudInitFailed("genisoimage failed".into()));
        }

        Ok(())
    }
}

impl std::fmt::Debug for KataBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KataBackend")
            .field("image_config", &self.image_config)
            .finish()
    }
}

#[async_trait]
impl SandboxBackend for KataBackend {
    fn backend_type(&self) -> &'static str {
        "kata"
    }

    fn is_available(&self) -> bool {
        which::which("cloud-hypervisor").is_ok()
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        let is_root = nix::unistd::geteuid().is_root();
        if !is_root {
            rootless::set_rootless(true);
            tracing::debug!("Enabled Kata rootless mode for non-root user");
        }
        Ok(())
    }

    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        _config: &PodSandboxConfig,
        pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>> {
        tracing::info!(sandbox_id = %sandbox.id, "Starting Kata VM for sandbox");

        let sandbox_path = sandbox.sandbox_path().clone();
        tokio::fs::create_dir_all(&sandbox_path).await?;

        let api_socket = sandbox_path.join("cloud-hypervisor.sock");
        let virtiofs_socket = sandbox_path.join("virtiofs.sock");
        let cloud_init_iso = sandbox_path.join("cloud-init.iso");

        if pool_config.cloud_init_dir.exists() {
            Self::generate_cloud_init_iso(sandbox, &cloud_init_iso).await?;
        }

        // FS-D (#365): compose a per-sandbox VFS (rootfs at `/` from FS-B's
        // OverlayFs(RAFS lower + per-sandbox writable upper) + the native
        // injected mounts `/stream`/`/models`/`/deltas`), forked into its own
        // Namespace bound to this sandbox's Subject, and serve it over a
        // per-sandbox Unix socket via FS-A's `hyprstream-vfs-server`. The guest
        // mounts this composed VFS — no external `nydusd`. Each sandbox's
        // namespace/socket/Subject/writable-upper is private, so sandbox A's
        // namespace cannot expose sandbox B's rootfs or injected paths.
        let virtiofs_daemon: Option<Arc<SandboxVirtiofs>> = None;
        let mut vfs_server: Option<SandboxFsServer> = None;
        let mut share_socket: Option<PathBuf> = None;
        if let Some(ref image_id) = sandbox.image_id {
            let subject = Self::sandbox_subject(sandbox, annotations);
            let server = self.compose_and_serve_vfs(sandbox, image_id, subject).await?;
            share_socket = Some(server.socket_path().to_path_buf());
            vfs_server = Some(server);
        }
        let virtiofs_sock = share_socket.clone();

        let hypervisor =
            Self::create_hypervisor(pool_config, sandbox, &api_socket, &virtiofs_socket).await?;

        hypervisor
            .prepare_vm(&sandbox.id, None, annotations, None)
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("failed to prepare VM: {e}")))?;

        // Attach the composed VFS socket as a CH ShareFs device, after prepare_vm
        // and before start_vm so it is folded into the boot VmConfig.
        if let Some(ref socket) = share_socket {
            Self::attach_share_fs(&hypervisor, &sandbox.id, socket, "hyprstream-vfs").await?;
        }

        let timeout_secs = i32::try_from(pool_config.create_timeout_secs).unwrap_or(i32::MAX);
        tracing::debug!(sandbox_id = %sandbox.id, timeout_secs, "Starting VM");
        hypervisor
            .start_vm(timeout_secs)
            .await
            .map_err(|e| WorkerError::VmStartFailed(format!("failed to start VM: {e}")))?;

        let pids = hypervisor.get_pids().await.ok();
        tracing::info!(
            sandbox_id = %sandbox.id,
            pids = ?pids,
            "VM started successfully via Kata Hypervisor trait"
        );

        let handle = Arc::new(KataHandle {
            hypervisor,
            api_socket,
            virtiofs_daemon,
            virtiofs_socket: virtiofs_sock,
            vfs_server,
        });

        Ok(handle)
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        tracing::info!(sandbox_id = %sandbox.id, "Stopping sandbox via Kata backend");

        if let Some(handle) = sandbox.backend_handle.as_ref() {
            if let Some(kata) = handle.as_any().downcast_ref::<KataHandle>() {
                kata.hypervisor
                    .stop_vm()
                    .await
                    .map_err(|e| WorkerError::VmStopFailed(format!("failed to stop VM: {e}")))?;
                tracing::debug!(sandbox_id = %sandbox.id, "VM stopped via Kata Hypervisor trait");
            }
        } else {
            tracing::warn!(
                sandbox_id = %sandbox.id,
                "No backend handle - sandbox may not have been started"
            );
        }

        tracing::info!(sandbox_id = %sandbox.id, "Sandbox stopped");
        Ok(())
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        tracing::info!(sandbox_id = %sandbox.id, "Destroying sandbox via Kata backend");

        if let Some(handle) = sandbox.backend_handle.as_ref() {
            if let Some(kata) = handle.as_any().downcast_ref::<KataHandle>() {
                if let Err(e) = kata.hypervisor.stop_vm().await {
                    tracing::warn!(sandbox_id = %sandbox.id, error = %e, "Error stopping VM during destroy");
                }
                if let Err(e) = kata.hypervisor.cleanup().await {
                    tracing::warn!(sandbox_id = %sandbox.id, error = %e, "Error cleaning up hypervisor resources");
                }

                if kata.api_socket.exists() {
                    let _ = tokio::fs::remove_file(&kata.api_socket).await;
                }
                if let Some(ref vs) = kata.virtiofs_socket {
                    if vs.exists() {
                        let _ = tokio::fs::remove_file(vs).await;
                    }
                }
            }
        }

        let sandbox_path = sandbox.sandbox_path();
        if sandbox_path.exists() {
            tracing::debug!(
                sandbox_id = %sandbox.id,
                dir = %sandbox_path.display(),
                "Removing sandbox runtime directory"
            );
            if let Err(e) = tokio::fs::remove_dir_all(sandbox_path).await {
                tracing::warn!(sandbox_id = %sandbox.id, error = %e, "Failed to remove sandbox runtime directory");
            }
        }

        tracing::info!(sandbox_id = %sandbox.id, "Sandbox destroyed");
        Ok(())
    }

    async fn reset(&self, sandbox: &mut PodSandbox) -> Result<bool> {
        tracing::debug!(sandbox_id = %sandbox.id, "Resetting Kata sandbox for warm pool");

        if let Some(handle) = sandbox.backend_handle.take() {
            if let Some(kata) = handle.as_any().downcast_ref::<KataHandle>() {
                let fresh = Arc::new(KataHandle {
                    hypervisor: Arc::clone(&kata.hypervisor),
                    api_socket: kata.api_socket.clone(),
                    virtiofs_daemon: None,
                    virtiofs_socket: None,
                    // The per-sandbox VFS server is not reusable across resets;
                    // a recycled sandbox composes a fresh one on next start.
                    vfs_server: None,
                });
                sandbox.backend_handle = Some(fresh);
            }
        }

        Ok(true)
    }

    async fn get_pids(&self, sandbox: &PodSandbox) -> Result<Vec<u32>> {
        if let Some(handle) = sandbox.backend_handle.as_ref() {
            if let Some(kata) = handle.as_any().downcast_ref::<KataHandle>() {
                return kata
                    .hypervisor
                    .get_pids()
                    .await
                    .map_err(|e| WorkerError::Internal(format!("get_pids failed: {e}")));
            }
        }
        Ok(Vec::new())
    }

    fn supports_exec(&self) -> bool {
        false
    }

    async fn exec_sync(
        &self,
        _sandbox: &PodSandbox,
        _command: &[String],
        _timeout_secs: u64,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
        Err(WorkerError::ExecFailed(
            "not supported by Kata backend (requires agent)".into(),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::{ImageConfig, PoolConfig};
    use crate::error::WorkerError;
    use crate::image::RafsStore;
    use tempfile::TempDir;

    /// Create a KataBackend with temporary directories.
    fn create_test_backend() -> (KataBackend, Arc<RafsStore>, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();

        let image_config = ImageConfig {
            blobs_dir: base.join("blobs"),
            bootstrap_dir: base.join("bootstrap"),
            refs_dir: base.join("refs"),
            cache_dir: base.join("cache"),
            runtime_dir: base.join("nydus-runtime"),
            ..Default::default()
        };

        std::fs::create_dir_all(&image_config.blobs_dir).unwrap();
        std::fs::create_dir_all(&image_config.bootstrap_dir).unwrap();
        std::fs::create_dir_all(&image_config.refs_dir).unwrap();
        std::fs::create_dir_all(&image_config.cache_dir).unwrap();

        let rafs_store = Arc::new(RafsStore::new(image_config.clone()).unwrap());
        let backend = KataBackend::new(image_config, Arc::clone(&rafs_store));
        (backend, rafs_store, temp_dir)
    }

    /// Create a PodSandbox with a temp directory as sandbox_path.
    fn create_test_sandbox(sandbox_path: PathBuf) -> PodSandbox {
        PodSandbox {
            id: "test-sandbox-001".to_owned(),
            metadata: crate::generated::worker_client::PodSandboxMetadata {
                name: String::new(),
                uid: String::new(),
                namespace: "default".to_owned(),
                attempt: 0,
            },
            state: crate::runtime::PodSandboxState::SandboxNotReady,
            created_at: chrono::Utc::now(),
            labels: vec![],
            annotations: vec![],
            runtime_handler: "kata".to_owned(),
            backend_handle: None,
            sandbox_path,
            image_id: None,
            console_socket: None,
        }
    }

    /// Create a KataHandle with a real CloudHypervisor instance (no VM started).
    fn create_test_handle(sandbox_path: &Path) -> Arc<KataHandle> {
        let ch = CloudHypervisor::new();
        Arc::new(KataHandle {
            hypervisor: Arc::new(ch),
            api_socket: sandbox_path.join("test.sock"),
            virtiofs_daemon: None,
            virtiofs_socket: Some(sandbox_path.join("virtiofs.sock")),
            vfs_server: None,
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // SandboxBackend trait method tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_backend_type() {
        let (backend, _, _temp) = create_test_backend();
        assert_eq!(backend.backend_type(), "kata");
    }

    #[test]
    fn test_supports_exec() {
        let (backend, _, _temp) = create_test_backend();
        assert!(!backend.supports_exec());
    }

    #[tokio::test]
    async fn test_exec_sync_returns_error() {
        let (backend, _, temp) = create_test_backend();
        let sandbox = create_test_sandbox(temp.path().join("sandbox"));

        let result = backend
            .exec_sync(&sandbox, &["echo".into(), "hello".into()], 5)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            WorkerError::ExecFailed(msg) => {
                assert!(msg.contains("not supported"), "unexpected message: {msg}");
            }
            other => panic!("Expected ExecFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_is_available() {
        let (backend, _, _temp) = create_test_backend();
        let expected = which::which("cloud-hypervisor").is_ok();
        assert_eq!(backend.is_available(), expected);
    }

    #[tokio::test]
    async fn test_initialize_sets_rootless() {
        let (backend, _, _temp) = create_test_backend();
        let pool_config = PoolConfig::default();

        backend.initialize(&pool_config).await.unwrap();

        // We're running as non-root in CI/dev, so rootless should be set.
        if !nix::unistd::geteuid().is_root() {
            assert!(rootless::is_rootless());
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // build_hypervisor_config tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_build_hypervisor_config_defaults() {
        let pool_config = PoolConfig {
            kernel_path: PathBuf::from("/boot/vmlinux"),
            vm_image: PathBuf::from("/images/rootfs.img"),
            vm_cpus: 4,
            vm_memory_mb: 2048,
            ..Default::default()
        };

        let config = KataBackend::build_hypervisor_config(&pool_config);

        assert_eq!(config.boot_info.kernel, "/boot/vmlinux");
        assert_eq!(config.boot_info.image, "/images/rootfs.img");
        assert_eq!(config.cpu_info.default_vcpus, 4.0);
        assert_eq!(config.cpu_info.default_maxvcpus, 4);
        assert_eq!(config.memory_info.default_memory, 2048);
    }

    #[test]
    fn test_build_hypervisor_config_explicit_path() {
        let pool_config = PoolConfig {
            hypervisor_path: PathBuf::from("/opt/custom/cloud-hypervisor"),
            ..Default::default()
        };

        let config = KataBackend::build_hypervisor_config(&pool_config);
        assert_eq!(config.path, "/opt/custom/cloud-hypervisor");
    }

    #[test]
    fn test_build_hypervisor_config_memory_overflow() {
        let pool_config = PoolConfig {
            vm_memory_mb: u64::MAX,
            ..Default::default()
        };

        let config = KataBackend::build_hypervisor_config(&pool_config);
        assert_eq!(config.memory_info.default_memory, u32::MAX);
    }

    // ─────────────────────────────────────────────────────────────────────
    // KataHandle tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_kata_handle_as_any_downcast() {
        let temp = TempDir::new().unwrap();
        let handle: Arc<dyn SandboxHandle> = create_test_handle(temp.path());

        let kata = handle.as_any().downcast_ref::<KataHandle>();
        assert!(kata.is_some(), "downcast to KataHandle should succeed");

        let kata = kata.unwrap();
        assert_eq!(kata.api_socket, temp.path().join("test.sock"));
        assert!(kata.virtiofs_daemon.is_none());
        assert_eq!(
            kata.virtiofs_socket.as_deref(),
            Some(temp.path().join("virtiofs.sock").as_path())
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // stop / get_pids / destroy with no handle
    // ─────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_stop_no_handle() {
        let (backend, _, temp) = create_test_backend();
        let sandbox = create_test_sandbox(temp.path().join("sandbox"));

        // Should succeed (logs warning, no error)
        backend.stop(&sandbox).await.unwrap();
    }

    #[tokio::test]
    async fn test_get_pids_no_handle() {
        let (backend, _, temp) = create_test_backend();
        let sandbox = create_test_sandbox(temp.path().join("sandbox"));

        let pids = backend.get_pids(&sandbox).await.unwrap();
        assert!(pids.is_empty());
    }

    #[tokio::test]
    async fn test_destroy_no_handle_cleans_dir() {
        let (backend, _, temp) = create_test_backend();
        let sandbox_path = temp.path().join("sandbox-to-destroy");
        std::fs::create_dir_all(&sandbox_path).unwrap();
        assert!(sandbox_path.exists());

        let sandbox = create_test_sandbox(sandbox_path.clone());
        backend.destroy(&sandbox).await.unwrap();

        assert!(!sandbox_path.exists(), "sandbox_path should be removed");
    }

    #[tokio::test]
    async fn test_destroy_no_handle_missing_dir() {
        let (backend, _, temp) = create_test_backend();
        let sandbox_path = temp.path().join("does-not-exist");
        assert!(!sandbox_path.exists());

        let sandbox = create_test_sandbox(sandbox_path);
        // Should succeed even though directory doesn't exist
        backend.destroy(&sandbox).await.unwrap();
    }

    // ─────────────────────────────────────────────────────────────────────
    // reset tests
    // ─────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_reset_no_handle() {
        let (backend, _, temp) = create_test_backend();
        let mut sandbox = create_test_sandbox(temp.path().join("sandbox"));

        let reusable = backend.reset(&mut sandbox).await.unwrap();
        assert!(reusable);
        assert!(sandbox.backend_handle.is_none());
    }

    #[tokio::test]
    async fn test_reset_preserves_hypervisor() {
        let (backend, _, temp) = create_test_backend();
        let sandbox_path = temp.path().join("sandbox");
        std::fs::create_dir_all(&sandbox_path).unwrap();
        let mut sandbox = create_test_sandbox(sandbox_path.clone());

        // Set a handle with virtiofs fields populated
        let original_handle = create_test_handle(&sandbox_path);
        let original_hypervisor_ptr =
            Arc::as_ptr(&original_handle.hypervisor) as *const () as usize;
        sandbox.backend_handle = Some(original_handle);

        let reusable = backend.reset(&mut sandbox).await.unwrap();
        assert!(reusable, "Kata reset should return true (reusable)");

        let new_handle = sandbox
            .backend_handle
            .as_ref()
            .expect("handle should still be set after reset");
        let kata = new_handle
            .as_any()
            .downcast_ref::<KataHandle>()
            .expect("should downcast to KataHandle");

        // Hypervisor Arc should point to the same allocation
        let new_hypervisor_ptr = Arc::as_ptr(&kata.hypervisor) as *const () as usize;
        assert_eq!(
            original_hypervisor_ptr, new_hypervisor_ptr,
            "reset should preserve the same hypervisor instance"
        );

        // virtiofs fields should be cleared
        assert!(kata.virtiofs_daemon.is_none(), "virtiofs_daemon should be cleared");
        assert!(kata.virtiofs_socket.is_none(), "virtiofs_socket should be cleared");

        // api_socket should be preserved
        assert_eq!(kata.api_socket, sandbox_path.join("test.sock"));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Debug impl
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_debug_impl() {
        let (backend, _, _temp) = create_test_backend();
        let debug = format!("{backend:?}");
        assert!(debug.contains("KataBackend"), "debug should contain struct name");
        assert!(debug.contains("image_config"), "debug should contain image_config field");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Cloud-init ISO generation (requires genisoimage)
    // ─────────────────────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore] // requires genisoimage binary
    async fn test_cloud_init_iso_generation() {
        let temp = TempDir::new().unwrap();
        let sandbox_dir = temp.path().join("sandbox-iso");
        std::fs::create_dir_all(&sandbox_dir).unwrap();

        let sandbox = create_test_sandbox(sandbox_dir.clone());
        let iso_path = sandbox_dir.join("cloud-init.iso");

        KataBackend::generate_cloud_init_iso(&sandbox, &iso_path)
            .await
            .unwrap();

        assert!(iso_path.exists(), "ISO file should be created");

        // Verify user-data was written with sandbox ID
        let user_data = tokio::fs::read_to_string(sandbox_dir.join("user-data"))
            .await
            .unwrap();
        assert!(user_data.contains("#cloud-config"));
        assert!(user_data.contains(&sandbox.id));

        // Verify meta-data was written with sandbox ID
        let meta_data = tokio::fs::read_to_string(sandbox_dir.join("meta-data"))
            .await
            .unwrap();
        assert!(meta_data.contains(&format!("instance-id: {}", sandbox.id)));
    }

    #[tokio::test]
    #[ignore] // requires genisoimage binary
    async fn test_cloud_init_uses_metadata_name() {
        let temp = TempDir::new().unwrap();
        let sandbox_dir = temp.path().join("sandbox-named");
        std::fs::create_dir_all(&sandbox_dir).unwrap();

        let mut sandbox = create_test_sandbox(sandbox_dir.clone());
        sandbox.metadata.name = "my-pod".to_owned();

        let iso_path = sandbox_dir.join("cloud-init.iso");
        KataBackend::generate_cloud_init_iso(&sandbox, &iso_path)
            .await
            .unwrap();

        let user_data = tokio::fs::read_to_string(sandbox_dir.join("user-data"))
            .await
            .unwrap();
        assert!(
            user_data.contains("hostname: my-pod"),
            "should use metadata name as hostname, got: {user_data}"
        );

        let meta_data = tokio::fs::read_to_string(sandbox_dir.join("meta-data"))
            .await
            .unwrap();
        assert!(
            meta_data.contains("local-hostname: my-pod"),
            "should use metadata name as local-hostname, got: {meta_data}"
        );
    }
}
