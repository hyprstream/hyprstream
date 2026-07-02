//! Per-sandbox VFS composition + serve (FS-D, #365).
//!
//! This is the end-to-end integration that ties FS-A/FS-B/FS-C together for the
//! Worker GA filesystem: for **each** sandbox it builds a private VFS
//! [`Namespace`], composes the image filesystem + injected mounts into it, and
//! serves it over a per-sandbox Unix socket that the Cloud Hypervisor guest
//! attaches as a virtio-fs ShareFs device.
//!
//! ```text
//!   per-sandbox flow (one per sandbox, fully isolated)
//!
//!   RafsStore (shared, RO)         injected mounts (per-sandbox)
//!        │                          /stream  (StreamMount)
//!   image_fs_for(...)      ──┐      /models  (SyntheticMount)
//!   (ImageFs: RAFS lower +  │      /deltas  (SyntheticMount)
//!    per-sandbox writable   │          │
//!    upper, OverlayFs)      ▼          ▼
//!                       Namespace::fork() + bind_mount  (FS-C)
//!                              │  bound to this sandbox's Subject
//!                              ▼
//!                       VfsFileSystem (FS-A down-adapter, Subject-per-call)
//!                              ▼
//!                       serve() on  <sandbox>/vfs.sock   (FS-A vhost-user-fs)
//!                              ▼
//!                       attach_share_fs  →  CH ShareFs device  (kata_backend)
//! ```
//!
//! ## Tenant isolation (#365 / #353 / #319 / #328)
//!
//! Each sandbox gets its **own** forked `Namespace`, its **own** writable rootfs
//! upper, its **own** injected-mount registries, its **own** socket, and is
//! served under its **own** [`Subject`]. Sandbox A's namespace contains only
//! A's rootfs and injected paths; it has no mount for B's rootfs or B's
//! injected registries, so A cannot name — let alone read — B's tree. The
//! `Subject` is threaded into every VFS op as the policy/audit principal at the
//! Mount boundary.
//!
//! ## Fail-closed
//!
//! Composition hard-errors if the rootfs cannot be built (missing RAFS
//! bootstrap, etc.); we never serve a partially-composed namespace. The
//! injected mounts are read-only except `/stream/{topic}/ctl`.

#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hyprstream_vfs::injected::{StreamMount, StreamRegistry, SyntheticMount, SyntheticNode};
use hyprstream_vfs::{Mount, Namespace, Subject};
use hyprstream_vfs_server::{serve, VfsFileSystem};

use crate::error::{Result, WorkerError};
use crate::image::{image_fs_for, RafsStore};

/// Default guest mount points for the injected mounts.
const ROOTFS_PREFIX: &str = "/";
const STREAM_PREFIX: &str = "/stream";
const MODELS_PREFIX: &str = "/models";
const DELTAS_PREFIX: &str = "/deltas";

/// File name of the per-sandbox VFS socket under the sandbox dir.
pub const VFS_SOCKET_NAME: &str = "vfs.sock";

/// The injected mounts composed into a sandbox's namespace, plus the handles
/// the runtime keeps to feed them at runtime.
pub struct InjectedMounts {
    /// `/stream` registry — push blocks / mark complete here at runtime.
    pub stream_registry: Arc<StreamRegistry>,
}

/// A fully composed, per-sandbox VFS: the namespace, its Subject, and the
/// runtime handles for the injected mounts.
pub struct SandboxFs {
    /// The composed namespace (rootfs + injected mounts), private to one sandbox.
    namespace: Namespace,
    /// The policy/audit principal this namespace is served under.
    subject: Subject,
    /// Runtime handles for the injected mounts.
    injected: InjectedMounts,
}

impl SandboxFs {
    /// Compose a per-sandbox VFS namespace for `image_id`, bound to `subject`.
    ///
    /// Steps (the FS-D core):
    /// 1. Fork an empty base namespace ([`Namespace::fork`] of `new()` — every
    ///    sandbox starts from a clean, private mount table).
    /// 2. Mount the **image filesystem** ([`image_fs_for`]) at `/`. The RAFS image
    ///    is the shared RO lower; a per-sandbox writable upper under
    ///    `<sandbox_dir>/rootfs/` is the CoW layer. It is the root purely because
    ///    it is mounted at `/` — "root" is a mount position, not a type (#633).
    /// 3. Bind the **injected mounts** — `/stream` (native pipes), `/models`,
    ///    `/deltas` (synthetic, read-only) — into the namespace.
    ///
    /// `models` / `deltas` are passed as synthetic trees so the caller decides
    /// what to expose (e.g. available model ids / delta metadata for this
    /// tenant). Pass empty dirs to expose just the mount point.
    pub fn compose(
        rafs_store: &RafsStore,
        image_id: &str,
        sandbox_dir: &Path,
        subject: Subject,
        models: SyntheticNode,
        deltas: SyntheticNode,
    ) -> Result<Self> {
        // 1. Per-sandbox private namespace.
        let mut namespace = Namespace::new().fork();

        // 2. Image filesystem at "/" (ImageFs: RAFS lower + per-sandbox
        // writable upper). It is the root purely because the namespace recipe
        // mounts it here — "root" is a mount position, not a type (#633).
        let bootstrap = rafs_store.bootstrap_path(image_id);
        let image_fs = image_fs_for(&bootstrap, rafs_store.blobs_dir(), image_id, sandbox_dir)?;
        // `FsMount: Mount`, so the image filesystem coerces to the `Arc<dyn Mount>`
        // the namespace stores; the down-adapter recovers the `FsMount` vtable via
        // `Mount::as_fsmount` for writes/copy-up.
        let root_mount: Arc<dyn Mount> = Arc::new(image_fs);
        namespace
            .mount(ROOTFS_PREFIX, root_mount)
            .map_err(|e| WorkerError::SandboxCreationFailed(format!("mount root: {e}")))?;

        // 3. Injected mounts — per-sandbox, never shared with another sandbox.
        let stream_registry = Arc::new(StreamRegistry::new());
        let stream_mount: Arc<dyn Mount> = Arc::new(StreamMount::new(stream_registry.clone()));
        namespace
            .mount(STREAM_PREFIX, stream_mount)
            .map_err(|e| WorkerError::SandboxCreationFailed(format!("mount /stream: {e}")))?;

        let models_mount: Arc<dyn Mount> = Arc::new(SyntheticMount::new(models));
        namespace
            .mount(MODELS_PREFIX, models_mount)
            .map_err(|e| WorkerError::SandboxCreationFailed(format!("mount /models: {e}")))?;

        let deltas_mount: Arc<dyn Mount> = Arc::new(SyntheticMount::new(deltas));
        namespace
            .mount(DELTAS_PREFIX, deltas_mount)
            .map_err(|e| WorkerError::SandboxCreationFailed(format!("mount /deltas: {e}")))?;

        Ok(Self {
            namespace,
            subject,
            injected: InjectedMounts { stream_registry },
        })
    }

    /// The composed namespace (read-only borrow; e.g. for tests / inspection).
    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    /// The Subject this namespace is served under.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Runtime handles for the injected mounts.
    pub fn injected(&self) -> &InjectedMounts {
        &self.injected
    }

    /// Build the FS-A [`VfsFileSystem`] down-adapter from this composed namespace,
    /// consuming `self` (the namespace moves into the adapter).
    ///
    /// `rt` is a tokio runtime handle the adapter uses to drive the async VFS ops
    /// from the server's synchronous (OS-thread) vring loop.
    pub fn into_filesystem(self, rt: tokio::runtime::Handle) -> (VfsFileSystem, InjectedMounts) {
        let fs = VfsFileSystem::new(self.namespace, self.subject, rt);
        (fs, self.injected)
    }

    /// Compose, then **serve** the namespace over a per-sandbox Unix socket.
    ///
    /// Spawns a dedicated OS thread running FS-A's blocking [`serve`] daemon
    /// (which `VhostUserDaemon::wait()`s until CH disconnects). Returns a
    /// [`SandboxFsServer`] holding the socket path and the injected handles; the
    /// caller then attaches the socket to CH via
    /// [`attach_share_fs`](super::kata_backend::KataBackend) before `start_vm`.
    ///
    /// `rt` is captured for the down-adapter's async→sync bridge.
    pub fn serve_on(
        self,
        socket_path: PathBuf,
        rt: tokio::runtime::Handle,
    ) -> Result<SandboxFsServer> {
        let (fs, injected) = self.into_filesystem(rt);

        let sock_str = socket_path
            .to_str()
            .ok_or_else(|| {
                WorkerError::SandboxCreationFailed("VFS socket path contains invalid UTF-8".into())
            })?
            .to_owned();

        // FS-A's `serve` blocks until the VMM disconnects; run it on a dedicated
        // OS thread (the down-adapter's `block_on` uses the captured runtime
        // handle, not this thread's executor).
        let thread = std::thread::Builder::new()
            .name(format!("vfs-serve-{}", socket_filename(&socket_path)))
            .spawn(move || {
                if let Err(e) = serve(fs, &sock_str) {
                    tracing::error!(socket = %sock_str, error = %e, "VFS server exited with error");
                }
            })
            .map_err(|e| {
                WorkerError::SandboxCreationFailed(format!("spawn VFS server thread: {e}"))
            })?;

        Ok(SandboxFsServer {
            socket_path,
            injected,
            _thread: thread,
        })
    }
}

/// A running per-sandbox VFS server: owns the socket path and injected handles.
///
/// Dropping it does not forcibly stop the daemon (the daemon exits when CH
/// disconnects); the join handle is detached.
pub struct SandboxFsServer {
    socket_path: PathBuf,
    injected: InjectedMounts,
    _thread: std::thread::JoinHandle<()>,
}

impl SandboxFsServer {
    /// The Unix socket the namespace is served on (attach this to CH).
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Runtime handles for the injected mounts (feed `/stream` here).
    pub fn injected(&self) -> &InjectedMounts {
        &self.injected
    }
}

fn socket_filename(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "sock".to_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    use crate::config::ImageConfig;
    use crate::image::rafs_builder::build_rafs_bootstrap;

    /// Build a `RafsStore` whose image dirs are rooted under `root`.
    fn make_store(root: &Path) -> RafsStore {
        let config = ImageConfig {
            blobs_dir: root.join("blobs"),
            bootstrap_dir: root.join("bootstrap"),
            refs_dir: root.join("refs"),
            cache_dir: root.join("cache"),
            runtime_dir: root.join("runtime"),
            ..ImageConfig::default()
        };
        for d in [&config.blobs_dir, &config.bootstrap_dir, &config.cache_dir] {
            std::fs::create_dir_all(d).unwrap();
        }
        RafsStore::new(config).expect("create store")
    }

    /// Build a gzip tar layer of regular files.
    fn make_gzip_layer(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).unwrap();
        let gz = GzEncoder::new(file, Compression::default());
        let mut tar = tar::Builder::new(gz);
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_entry_type(tar::EntryType::Regular);
            tar.append_data(&mut header, name, *data).unwrap();
        }
        let gz = tar.into_inner().unwrap();
        gz.finish().unwrap();
    }

    /// Build image content (`image_id`) into an existing store's layout.
    fn add_image(store: &RafsStore, scratch: &Path, image_id: &str, entries: &[(&str, &[u8])]) {
        let bootstrap = store.bootstrap_path(image_id);
        if let Some(parent) = bootstrap.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let layer = scratch.join(format!("{}.tar.gz", image_id.replace(':', "_")));
        make_gzip_layer(&layer, entries);
        build_rafs_bootstrap(&[layer], store.blobs_dir(), &bootstrap).unwrap();
    }

    /// A `RafsStore` rooted under `root` carrying one image.
    fn store_with_image(root: &Path, image_id: &str, entries: &[(&str, &[u8])]) -> RafsStore {
        let store = make_store(root);
        add_image(&store, root, image_id, entries);
        store
    }

    fn models_tree() -> SyntheticNode {
        SyntheticNode::dir().with_child(
            "qwen3",
            SyntheticNode::dir().with_child("info", SyntheticNode::file(b"qwen3\n".to_vec())),
        )
    }

    /// Core: a composed namespace exposes the rootfs + injected mounts, all
    /// served under the sandbox's Subject.
    #[tokio::test]
    async fn composed_namespace_exposes_rootfs_and_injected() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_image(tmp.path(), "img:a", &[("etc/hostname", b"worker-a\n")]);
        let sandbox_dir = tmp.path().join("sandbox-a");
        let subj = Subject::new("tenant-a");

        let fs = SandboxFs::compose(
            &store,
            "img:a",
            &sandbox_dir,
            subj.clone(),
            models_tree(),
            SyntheticNode::dir(),
        )
        .unwrap();
        let ns = fs.namespace();

        // Rootfs file (from the RAFS lower) is readable via the namespace.
        let hostname = ns.cat("/etc/hostname", &subj).await.unwrap();
        assert_eq!(hostname, b"worker-a\n");

        // Injected /models synthetic file is readable.
        let info = ns.cat("/models/qwen3/info", &subj).await.unwrap();
        assert_eq!(info, b"qwen3\n");

        // Injected mount points are present at the namespace root.
        let prefixes = ns.mount_prefixes();
        assert!(prefixes.contains(&"/"));
        assert!(prefixes.contains(&"/stream"));
        assert!(prefixes.contains(&"/models"));
        assert!(prefixes.contains(&"/deltas"));

        // The /stream registry is reachable and per-sandbox.
        fs.injected().stream_registry.register("job-1", Some("tenant-a".into()));
        fs.injected().stream_registry.push("job-1", b"chunk\n".to_vec());
        let chunk = ns.read_one("/stream/job-1/data", &subj).await.unwrap();
        assert_eq!(chunk, b"chunk\n");
    }

    /// Isolation: two sandboxes get independent namespaces; A cannot read B's
    /// rootfs content nor B's injected stream topics, and vice versa.
    #[tokio::test]
    async fn two_sandboxes_are_isolated() {
        let tmp = tempfile::tempdir().unwrap();
        let store = make_store(tmp.path());

        // Two distinct images, each with a tenant-specific marker file.
        add_image(&store, tmp.path(), "img:a", &[("etc/secret", b"SECRET-A\n")]);
        add_image(&store, tmp.path(), "img:b", &[("etc/secret", b"SECRET-B\n")]);

        let subj_a = Subject::new("tenant-a");
        let subj_b = Subject::new("tenant-b");
        let fs_a = SandboxFs::compose(
            &store,
            "img:a",
            &tmp.path().join("sandbox-a"),
            subj_a.clone(),
            SyntheticNode::dir(),
            SyntheticNode::dir(),
        )
        .unwrap();
        let fs_b = SandboxFs::compose(
            &store,
            "img:b",
            &tmp.path().join("sandbox-b"),
            subj_b.clone(),
            SyntheticNode::dir(),
            SyntheticNode::dir(),
        )
        .unwrap();

        // Each sees only its own rootfs content.
        assert_eq!(fs_a.namespace().cat("/etc/secret", &subj_a).await.unwrap(), b"SECRET-A\n");
        assert_eq!(fs_b.namespace().cat("/etc/secret", &subj_b).await.unwrap(), b"SECRET-B\n");

        // Injected stream topics are per-sandbox: registering in A is invisible
        // in B's namespace (separate registries — A's path does not exist in B).
        fs_a.injected().stream_registry.register("a-only", Some("tenant-a".into()));
        fs_a.injected().stream_registry.push("a-only", b"a-data".to_vec());
        // A can read it.
        assert_eq!(
            fs_a.namespace().read_one("/stream/a-only/data", &subj_a).await.unwrap(),
            b"a-data"
        );
        // B's namespace has no such topic: walk fails.
        assert!(fs_b.namespace().read_one("/stream/a-only/data", &subj_b).await.is_err());

        // B's writable rootfs upper is a distinct directory from A's: a write in
        // A must not appear in B.
        let _ = std::fs::write(
            tmp.path().join("sandbox-a").join("rootfs/upper/etc/written-by-a"),
            b"x",
        );
        assert!(fs_b.namespace().cat("/etc/written-by-a", &subj_b).await.is_err());
    }

    /// Fail-closed: composing with a missing image bootstrap hard-errors rather
    /// than serving an empty / partial namespace.
    #[test]
    fn compose_missing_image_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let store = make_store(tmp.path());
        let res = SandboxFs::compose(
            &store,
            "img:does-not-exist",
            &tmp.path().join("sandbox"),
            Subject::new("t"),
            SyntheticNode::dir(),
            SyntheticNode::dir(),
        );
        assert!(res.is_err(), "missing image must fail-closed");
    }

    /// The composed namespace can be wrapped as the FS-A down-adapter
    /// (`VfsFileSystem`), proving the serve path's first step type-checks and the
    /// namespace is `Send` (moved into the adapter).
    #[tokio::test]
    async fn composed_namespace_builds_filesystem() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_image(tmp.path(), "img:fs", &[("etc/hostname", b"h\n")]);
        let fs = SandboxFs::compose(
            &store,
            "img:fs",
            &tmp.path().join("sandbox"),
            Subject::new("t"),
            SyntheticNode::dir(),
            SyntheticNode::dir(),
        )
        .unwrap();
        let (_vfs, injected) = fs.into_filesystem(tokio::runtime::Handle::current());
        // Injected handles survive the move into the adapter.
        injected.stream_registry.register("x", None);
        assert!(injected.stream_registry.exists("x"));
    }

    /// The FS-A down-adapter serves the composed namespace: driving the
    /// `FileSystem` trait (the surface the vhost-user-fs server dispatches to)
    /// resolves both the rootfs and the injected mount points. This is the
    /// adapter-level proof of "the server serves the composed Namespace"; the
    /// full vhost-user daemon over a live CH VM is the out-of-scope live
    /// validation.
    #[test]
    fn down_adapter_serves_composed_namespace() {
        use fuse_backend_rs::api::filesystem::{Context, FileSystem};
        use std::ffi::CString;

        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_image(tmp.path(), "img:s", &[("etc/hostname", b"served\n")]);
        let fs = SandboxFs::compose(
            &store,
            "img:s",
            &tmp.path().join("sandbox"),
            Subject::new("tenant"),
            models_tree(),
            SyntheticNode::dir(),
        )
        .unwrap();

        // Build the down-adapter on a dedicated multi-thread runtime (the
        // adapter's block_on must not run on the calling thread).
        // Held for the whole test: the adapter's `block_on` must run on a
        // runtime distinct from this (calling) thread, so a multi-thread handle.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let (vfs, _injected) = fs.into_filesystem(rt.handle().clone());

        let ctx = Context::default();
        let root = 1u64; // FUSE_ROOT_ID
        let cstr = |s: &str| CString::new(s).unwrap();

        // Rootfs: /etc/hostname resolves through the `/`-rooted overlay mount.
        let etc = vfs.lookup(&ctx, root, &cstr("etc")).unwrap();
        assert_eq!(etc.attr.st_mode & libc::S_IFMT, libc::S_IFDIR);
        let hostname = vfs.lookup(&ctx, etc.inode, &cstr("hostname")).unwrap();
        assert_eq!(hostname.attr.st_mode & libc::S_IFMT, libc::S_IFREG);

        // Injected mount points are visible as directories at the namespace root.
        for name in ["stream", "models", "deltas"] {
            let entry = vfs.lookup(&ctx, root, &cstr(name)).unwrap();
            assert_eq!(
                entry.attr.st_mode & libc::S_IFMT,
                libc::S_IFDIR,
                "{name} should be a directory"
            );
        }

        // Injected synthetic file resolves through the adapter.
        let models = vfs.lookup(&ctx, root, &cstr("models")).unwrap();
        let qwen = vfs.lookup(&ctx, models.inode, &cstr("qwen3")).unwrap();
        let info = vfs.lookup(&ctx, qwen.inode, &cstr("info")).unwrap();
        assert_eq!(info.attr.st_mode & libc::S_IFMT, libc::S_IFREG);
    }
}
