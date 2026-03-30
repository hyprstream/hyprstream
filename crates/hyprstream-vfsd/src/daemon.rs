//! VFS daemon — serves a VFS `Namespace` to guest VMs via vhost-user-fs.
//!
//! The `VfsDaemon` wraps a `VfsFuse` adapter around a `Mount` and exposes it
//! over a vhost-user-fs Unix socket. Guest VMs (e.g., Kata worker VMs) mount
//! the socket as a virtio-fs device to access the host VFS namespace.
//!
//! # Architecture
//!
//! ```text
//! Guest VM (mount -t virtiofs tag /mnt)
//!      |
//! vhost-user-fs socket (Unix domain)
//!      |
//! fuse-backend-rs Server<VfsFuse<M>>
//!      |
//! VfsFuse<M>  (this crate, FUSE <-> Mount adapter)
//!      |
//! hyprstream-vfs Mount trait
//! ```
//!
//! # KataBackend integration (documentation)
//!
//! To wire `VfsDaemon` into `KataBackend::start()`:
//!
//! 1. Build a `Namespace` with the authorized mounts for this sandbox:
//!    ```ignore
//!    let mut ns = Namespace::new();
//!    ns.mount("/srv/model", Arc::new(model_mount)).unwrap();
//!    ns.mount("/srv/tcl", Arc::new(tcl_mount)).unwrap();
//!    ```
//!
//! 2. Create and start a `VfsDaemon` on the sandbox's virtiofs socket:
//!    ```ignore
//!    let daemon = VfsDaemon::new(namespace_mount, socket_path);
//!    daemon.start()?;
//!    ```
//!
//! 3. Pass the socket to cloud-hypervisor via `--fs` flag:
//!    ```text
//!    cloud-hypervisor ... --fs socket=<socket_path>,tag=hyprfs,num_queues=1,queue_size=1024
//!    ```
//!
//! 4. In the guest VM, mount via:
//!    ```text
//!    mount -t virtiofs hyprfs /srv
//!    ```
//!
//! 5. Store the `VfsDaemon` in `KataHandle` alongside the existing `virtiofs_daemon`:
//!    ```ignore
//!    pub struct KataHandle {
//!        // ... existing fields ...
//!        pub vfs_daemon: Option<Arc<VfsDaemon<NamespaceMount>>>,
//!    }
//!    ```
//!
//! 6. On `stop()` / `destroy()`, call `daemon.stop()` to clean up.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use fuse_backend_rs::api::server::Server;

use hyprstream_vfs::Mount;

use crate::VfsFuse;

/// Daemon serving a VFS mount over a vhost-user-fs socket.
///
/// Lifecycle follows the `SandboxVirtiofs` pattern:
/// - `new()` configures the daemon (socket path, mount)
/// - `start()` spawns the vhost-user-fs listener thread
/// - `stop()` signals the listener to shut down
///
/// The daemon owns a `Server<VfsFuse<M>>` which translates between the
/// virtio-fs protocol and the VFS `Mount` trait.
pub struct VfsDaemon<M: Mount + 'static> {
    /// Socket path for vhost-user-fs connection.
    socket_path: PathBuf,
    /// The FUSE server wrapping our VFS mount.
    server: Arc<Server<VfsFuse<M>>>,
    /// Whether the daemon is running.
    running: Arc<AtomicBool>,
    /// Listener thread handle.
    thread_handle: Option<JoinHandle<()>>,
}

impl<M: Mount + 'static> VfsDaemon<M> {
    /// Create a new VFS daemon.
    ///
    /// # Arguments
    /// * `mount` - The VFS mount to serve (typically a `Namespace` adapter or a single service mount)
    /// * `socket_path` - Unix socket path for the vhost-user-fs connection
    pub fn new(mount: M, socket_path: impl Into<PathBuf>) -> Self {
        let rt = tokio::runtime::Handle::current();
        let vfs_fuse = VfsFuse::new(mount, rt);
        let server = Arc::new(Server::new(vfs_fuse));

        Self {
            socket_path: socket_path.into(),
            server,
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: None,
        }
    }

    /// Start the vhost-user-fs daemon.
    ///
    /// Spawns a dedicated thread that listens on the Unix socket for
    /// virtio-fs requests from the guest VM. The thread processes FUSE
    /// requests synchronously via `Server::handle_message()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket directory cannot be created or if
    /// the listener thread fails to start.
    pub fn start(&mut self) -> std::io::Result<()> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Ensure the socket parent directory exists.
        if let Some(parent) = self.socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Remove stale socket if present.
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)?;
        }

        self.running.store(true, Ordering::Release);

        let running = Arc::clone(&self.running);
        let _server = Arc::clone(&self.server);
        let socket_path = self.socket_path.clone();

        // NOTE: Full vhost-user-fs listener implementation requires:
        //
        // 1. Create a Unix listener socket at `socket_path`
        // 2. Accept a connection from the VMM (cloud-hypervisor)
        // 3. Negotiate the vhost-user protocol (feature flags, memory mapping)
        // 4. Process virtio-queue descriptors, translating to FUSE requests
        // 5. Dispatch FUSE requests to `server.handle_message()`
        //
        // The fuse-backend-rs crate provides the `Server` for FUSE request
        // handling and the `virtiofs` transport module for descriptor chain
        // I/O. The vhost-user protocol negotiation is provided by the
        // `vhost` crate (not currently a dependency).
        //
        // For now, we create the socket to signal readiness and log the
        // configuration. Full implementation requires adding:
        //   - `vhost` crate dependency (vhost-user slave)
        //   - Virtio queue setup and memory region mapping
        //   - Request processing loop
        //
        // See: https://github.com/rust-vmm/vhost for the vhost-user slave.

        let handle = std::thread::Builder::new()
            .name(format!("vfsd-{}", socket_path.display()))
            .spawn(move || {
                tracing::info!(
                    socket = %socket_path.display(),
                    "VFS daemon started (awaiting vhost-user-fs implementation)"
                );

                // Create the socket file to signal readiness to the VMM.
                // In production, this would be replaced by the actual
                // vhost-user listener socket.
                if let Err(e) = std::os::unix::net::UnixListener::bind(&socket_path) {
                    tracing::error!(
                        socket = %socket_path.display(),
                        error = %e,
                        "Failed to bind vhost-user-fs socket"
                    );
                    running.store(false, Ordering::Release);
                    return;
                }

                tracing::info!(
                    socket = %socket_path.display(),
                    "VFS daemon socket bound, waiting for VMM connection"
                );

                // Placeholder: wait until signaled to stop.
                // In production this would be the vhost-user request loop.
                while running.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }

                // Clean up socket.
                let _ = std::fs::remove_file(&socket_path);
                tracing::info!("VFS daemon stopped");
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Stop the daemon and clean up.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Release);

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }

        // Best-effort socket cleanup.
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
    }

    /// Check if the daemon is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get the socket path.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

impl<M: Mount + 'static> Drop for VfsDaemon<M> {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{DirEntry, Fid, MountError, Stat};

    /// Minimal mount for testing daemon lifecycle.
    struct StubMount;

    struct StubFid;

    #[async_trait]
    impl Mount for StubMount {
        async fn walk(&self, _c: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(StubFid))
        }
        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        async fn read(&self, _fid: &Fid, _offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            Ok(b"stub".to_vec())
        }
        async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(0)
        }
        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![])
        }
        async fn stat(&self, _fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            Ok(Stat { qtype: 0, size: 0, name: String::new(), mtime: 0 })
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    /// Wait for the socket to appear (thread needs time to bind).
    async fn wait_for_socket(path: &Path) {
        for _ in 0..20 {
            if path.exists() { return; }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    #[tokio::test]
    async fn daemon_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test-vfsd.sock");

        let mut daemon = VfsDaemon::new(StubMount, &sock);
        assert!(!daemon.is_running());

        daemon.start().unwrap();
        assert!(daemon.is_running());
        wait_for_socket(&sock).await;
        assert!(sock.exists(), "socket should be created");

        daemon.stop();
        assert!(!daemon.is_running());
        assert!(!sock.exists(), "socket should be cleaned up");
    }

    #[tokio::test]
    async fn daemon_double_start_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test-vfsd2.sock");

        let mut daemon = VfsDaemon::new(StubMount, &sock);
        daemon.start().unwrap();
        wait_for_socket(&sock).await;
        // Second start should be a no-op.
        daemon.start().unwrap();
        assert!(daemon.is_running());

        daemon.stop();
    }

    #[tokio::test]
    async fn daemon_drop_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test-vfsd3.sock");

        {
            let mut daemon = VfsDaemon::new(StubMount, &sock);
            daemon.start().unwrap();
            wait_for_socket(&sock).await;
            assert!(sock.exists());
        }
        // After drop, socket should be removed.
        // Give thread a moment to clean up.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(!sock.exists(), "socket should be cleaned up on drop");
    }

    #[tokio::test]
    async fn socket_path_accessor() {
        let daemon = VfsDaemon::new(StubMount, "/tmp/test-vfsd-accessor.sock");
        assert_eq!(daemon.socket_path(), Path::new("/tmp/test-vfsd-accessor.sock"));
    }
}
