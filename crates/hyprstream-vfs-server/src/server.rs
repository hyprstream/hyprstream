//! vhost-user-fs server: serves a [`VfsFileSystem`] over a Unix socket.
//!
//! `fuse_backend_rs` provides the FUSE protocol engine
//! ([`Server`](fuse_backend_rs::api::server::Server) +
//! [`Reader`](fuse_backend_rs::transport::Reader) /
//! [`VirtioFsWriter`](fuse_backend_rs::transport::VirtioFsWriter)) but **not** a
//! vhost-user transport — we host it with the `vhost-user-backend` crate, the
//! same stacking production virtiofsd uses. Cloud Hypervisor connects to the
//! socket; for each request descriptor chain we build a `Reader`/`Writer` over
//! guest memory and call [`Server::handle_message`], which dispatches to our
//! [`VfsFileSystem`] (the `Namespace → FileSystem` down-adapter).
//!
//! This is the inverse direction of FS-C's adapter and the host side of the CH
//! ShareFs attach in `kata_backend.rs`.

use std::io;
use std::path::Path;
use std::sync::Arc;
// `vhost-user-backend` only implements `VhostUserBackend` for the std `Mutex`/
// `RwLock` (backend.rs:335/415), so we must use the std type here despite the
// project's parking_lot preference — there is no poison risk as the daemon owns
// the lock and our backend methods never panic while holding it.
#[allow(clippy::disallowed_types)]
use std::sync::Mutex;

use fuse_backend_rs::api::server::Server;
use fuse_backend_rs::transport::{FuseChannel, FuseSession, Reader, VirtioFsWriter, Writer};
use vhost::vhost_user::message::{VhostUserProtocolFeatures, VhostUserVirtioFeatures};
use vhost::vhost_user::Listener;
use vhost_user_backend::{VhostUserBackendMut, VhostUserDaemon, VringRwLock, VringT};
use virtio_bindings::bindings::virtio_config::VIRTIO_F_VERSION_1;
use virtio_bindings::bindings::virtio_ring::{
    VIRTIO_RING_F_EVENT_IDX, VIRTIO_RING_F_INDIRECT_DESC,
};
use virtio_queue::QueueOwnedT;
use vm_memory::{GuestAddressSpace, GuestMemoryAtomic, GuestMemoryMmap};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::eventfd::EventFd;

use crate::filesystem::VfsFileSystem;

/// Map any backend error into an `io::Error` for the vring loop.
fn into_io<E: std::fmt::Debug>(e: E) -> io::Error {
    io::Error::other(format!("{e:?}"))
}

/// Guest memory type the daemon manages.
type GuestMemory = GuestMemoryAtomic<GuestMemoryMmap>;

/// virtio-fs queues: 1 hiprio + 1 request queue (the minimal, single-thread set).
const NUM_QUEUES: usize = 2;
/// Max descriptors per queue (matches the CH `queue_size` default of 1024).
const QUEUE_SIZE: usize = 1024;

/// The vhost-user backend hosting a [`VfsFileSystem`] [`Server`].
pub struct VfsBackend {
    server: Arc<Server<VfsFileSystem>>,
    mem: Option<GuestMemory>,
    event_idx: bool,
    /// Exit eventfd so the daemon can be signalled to stop.
    exit: EventFd,
}

impl VfsBackend {
    fn new(fs: VfsFileSystem) -> io::Result<Self> {
        Ok(Self {
            server: Arc::new(Server::new(fs)),
            mem: None,
            event_idx: false,
            exit: EventFd::new(libc::EFD_NONBLOCK)?,
        })
    }

    /// Drain one request queue, dispatching each chain through the FUSE server.
    ///
    /// Iteration borrows the queue under the vring's write guard; `add_used`
    /// re-locks the vring, so we stage `(head, len)` completions and apply them
    /// after dropping the guard (holding both would self-deadlock the lock).
    fn process_queue(&self, vring: &VringRwLock) -> io::Result<bool> {
        let mem = self
            .mem
            .as_ref()
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EINVAL))?
            .memory();
        // `&*mem` is the guard's `Deref::Target` (`&GuestMemoryMmap`), which is
        // what `from_descriptor_chain`/`VirtioFsWriter::new` want; the chain's
        // own `M` is the cloned guard handed to `iter`.
        let desc_mem: &GuestMemoryMmap = &mem;

        let mut completed: Vec<(u16, u32)> = Vec::new();
        {
            let mut state = vring.get_mut();
            let queue = state.get_queue_mut();
            let avail = queue.iter(mem.clone()).map_err(into_io)?;
            for chain in avail {
                let head = chain.head_index();
                let reader =
                    Reader::from_descriptor_chain(desc_mem, chain.clone()).map_err(into_io)?;
                let writer: Writer = VirtioFsWriter::new(desc_mem, chain.clone())
                    .map_err(into_io)?
                    .into();
                let len = self
                    .server
                    .handle_message(reader, writer, None, None)
                    .map_err(into_io)?;
                completed.push((head, len as u32));
            }
        } // write guard dropped before add_used

        let used = !completed.is_empty();
        for (head, len) in completed {
            vring.add_used(head, len).map_err(into_io)?;
        }
        Ok(used)
    }
}

impl VhostUserBackendMut for VfsBackend {
    type Bitmap = ();
    type Vring = VringRwLock;

    fn num_queues(&self) -> usize {
        NUM_QUEUES
    }

    fn max_queue_size(&self) -> usize {
        QUEUE_SIZE
    }

    fn features(&self) -> u64 {
        (1 << VIRTIO_F_VERSION_1)
            | (1 << VIRTIO_RING_F_INDIRECT_DESC)
            | (1 << VIRTIO_RING_F_EVENT_IDX)
            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits()
    }

    fn protocol_features(&self) -> VhostUserProtocolFeatures {
        VhostUserProtocolFeatures::MQ | VhostUserProtocolFeatures::REPLY_ACK
    }

    fn set_event_idx(&mut self, enabled: bool) {
        self.event_idx = enabled;
    }

    fn update_memory(&mut self, mem: GuestMemory) -> io::Result<()> {
        self.mem = Some(mem);
        Ok(())
    }

    fn handle_event(
        &mut self,
        device_event: u16,
        _evset: EventSet,
        vrings: &[Self::Vring],
        _thread_id: usize,
    ) -> io::Result<()> {
        let vring = vrings
            .get(device_event as usize)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EINVAL))?;

        let mut used = false;
        if self.event_idx {
            // event_idx: disable notifications, drain, re-enable, retry if work
            // arrived in the window. Standard virtiofsd loop.
            loop {
                vring.disable_notification().map_err(into_io)?;
                used |= self.process_queue(vring)?;
                if !vring.enable_notification().map_err(into_io)? {
                    break;
                }
            }
        } else {
            used = self.process_queue(vring)?;
        }
        // Signal the guest that descriptors were consumed.
        if used {
            vring.signal_used_queue().map_err(into_io)?;
        }
        Ok(())
    }

    fn exit_event(&self, _thread_id: usize) -> Option<EventFd> {
        self.exit.try_clone().ok()
    }
}

/// Serve `fs` over the vhost-user-fs Unix socket at `socket_path`, blocking until
/// the daemon exits (the VMM disconnects or [`exit_event`](VfsBackend::exit_event)
/// is signalled).
///
/// The caller is responsible for the socket lifecycle and for attaching the same
/// path to Cloud Hypervisor as a ShareFs device (see `kata_backend.rs`). Run this
/// on a dedicated thread; it blocks.
#[allow(clippy::disallowed_types)] // std Mutex required by vhost-user-backend's blanket impl
pub fn serve(fs: VfsFileSystem, socket_path: &str) -> io::Result<()> {
    let backend = Arc::new(Mutex::new(VfsBackend::new(fs)?));
    let mem = GuestMemoryAtomic::new(GuestMemoryMmap::new());
    let mut daemon = VhostUserDaemon::new("hyprstream-vfs-fs".to_owned(), backend, mem)
        .map_err(|e| io::Error::other(format!("daemon init: {e}")))?;

    let listener = Listener::new(socket_path, true)
        .map_err(|e| io::Error::other(format!("listen {socket_path}: {e}")))?;
    daemon
        .start(listener)
        .map_err(|e| io::Error::other(format!("daemon start: {e}")))?;
    daemon
        .wait()
        .map_err(|e| io::Error::other(format!("daemon wait: {e}")))?;
    Ok(())
}

/// Mount `fs` at `mountpoint` as a real host directory via the kernel `fusedev`
/// transport (`/dev/fuse`), blocking until the mount is torn down (unmounted, or
/// the kernel closes the FUSE device).
///
/// This is the non-VM counterpart to [`serve`]: the same transport-agnostic
/// [`VfsFileSystem`] down-adapter (`SandboxFs::into_filesystem`), served as a
/// plain POSIX directory instead of a vhost-user-fs virtio-fs share. It exists
/// so backends that cannot attach a virtio-fs device — `podman run --rootfs
/// <dir>` (#617) and `systemd-nspawn --directory=<dir>` — can consume the same
/// composed per-sandbox namespace the kata/CH path uses, instead of
/// side-channeling their own image provisioning (spine violation, epic #508).
///
/// `mountpoint` must already exist as a directory; this function does not
/// create it. The mount is torn down when [`FuseSession`] is dropped (which
/// happens when this function returns, on any exit path) or when the kernel
/// unmounts it out from under us (e.g. `fusermount3 -u`, or the guest killing
/// the process).
///
/// Open question (tracked by #653, resolution deferred to #617): rootless FUSE
/// (unprivileged `/dev/fuse` access via `fusermount3`) combined with podman's
/// `--userns=keep-id` has not been validated here — this function mounts via
/// the same kernel `/dev/fuse` + `fusermount3` path `fuse-backend-rs` uses for
/// any fusedev consumer, but whether that composition works unprivileged under
/// `keep-id` user-namespace remapping is unresolved and out of scope for this
/// change.
///
/// Run this on a dedicated thread; like [`serve`], it blocks until the session
/// ends.
pub fn serve_local(fs: VfsFileSystem, mountpoint: &Path) -> io::Result<()> {
    let server = Arc::new(Server::new(fs));

    let mut session = FuseSession::new(mountpoint, "hyprstream-vfs", "", false)
        .map_err(|e| io::Error::other(format!("fuse session init {mountpoint:?}: {e}")))?;
    // `fuse-backend-rs` defaults `allow_other` to true, which requires
    // `user_allow_other` in `/etc/fuse.conf` (a host-wide opt-in most
    // deployments won't have set). We only need this mount visible to the
    // process serving it (and whatever shares its uid/gid — e.g. a podman
    // `--userns=keep-id` child), so disable it and mount unprivileged.
    session.set_allow_other(false);
    session
        .mount()
        .map_err(|e| io::Error::other(format!("fuse mount {mountpoint:?}: {e}")))?;

    let mut channel = session
        .new_channel()
        .map_err(|e| io::Error::other(format!("fuse channel {mountpoint:?}: {e}")))?;

    let result = drive_channel(&server, &mut channel);

    // Always attempt to tear down the mount, even if the request loop errored,
    // so a failure mid-mount doesn't leave a dangling `/dev/fuse` connection.
    if let Err(e) = session.umount() {
        tracing::warn!(mountpoint = %mountpoint.display(), error = %e, "fuse umount failed");
    }

    result
}

/// Drain `channel`, dispatching each request through the FUSE server, until the
/// kernel closes the session (`get_request` returns `Ok(None)`).
fn drive_channel(server: &Arc<Server<VfsFileSystem>>, channel: &mut FuseChannel) -> io::Result<()> {
    loop {
        let request = match channel.get_request() {
            Ok(r) => r,
            Err(e) => {
                // `FuseChannel::get_request` treats a clean `ENODEV` read as
                // `Ok(None)` (see its `linux_session.rs` docs), but a
                // concurrent unmount can also surface as an epoll-error event
                // on the `/dev/fuse` fd, which it reports as `Err` rather than
                // `Ok(None)`. Either shape means the same thing here: the
                // kernel end of the session is gone. Treat any channel error
                // as ordinary session teardown rather than a hard failure —
                // there is nothing left to serve once `/dev/fuse` is torn
                // down, and propagating it would turn a normal `fusermount -u`
                // into a spurious error from `serve_local`.
                tracing::debug!(error = %e, "fuse channel closed");
                return Ok(());
            }
        };
        let Some((reader, writer)) = request else {
            // Session closed (unmounted or kernel dropped /dev/fuse).
            return Ok(());
        };
        if let Err(e) = server.handle_message(reader, writer.into(), None, None) {
            // Mirrors fuse-backend-rs's own fusedev example: an encode failure
            // means the kernel end is gone, so stop; any other per-request
            // error is logged and the loop continues serving later requests.
            match e {
                fuse_backend_rs::Error::EncodeMessage(_) => return Ok(()),
                _ => tracing::error!(error = %e, "fuse request failed"),
            }
        }
    }
}
