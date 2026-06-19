//! `hyprstream-vfs-server` — serve a hyprstream VFS [`Namespace`] to a Cloud
//! Hypervisor guest over vhost-user-fs.
//!
//! This crate is the **down** half of the bidirectional `FileSystem ↔ Mount`
//! bridge from the converged FS/VFS design (FS-A, #362). FS-C built the *up*
//! adapter ([`hyprstream_vfs::FuseFileSystemMount`]) that exposes a
//! `fuse_backend_rs` `FileSystem` (RAFS / OverlayFs / Passthrough) as a VFS
//! [`Mount`](hyprstream_vfs::Mount). This crate builds the *down* adapter
//! ([`VfsFileSystem`]) that exposes a whole VFS [`Namespace`](hyprstream_vfs::Namespace)
//! as a `fuse_backend_rs` `FileSystem`, and wires `fuse_backend_rs`'s FUSE
//! protocol engine to a vhost-user-fs server ([`serve`]) so Cloud Hypervisor can
//! mount it as a virtio-fs share.
//!
//! ```text
//! Cloud Hypervisor  (virtio-fs / vhost-user-fs device)
//!   → vhost-user-backend       (Unix-socket transport + vring loop) — server.rs
//!     → fuse_backend_rs::Server (FUSE protocol engine)
//!       → VfsFileSystem         (Namespace → FileSystem down-adapter) — filesystem.rs
//!         → Namespace mounts    (Mount read/write + FsMount writes, Subject-per-call)
//! ```
//!
//! The crate is **native-only** (Linux): vhost-user-fs and `fuse_backend_rs`'s
//! virtio transport are not available on wasm, and the whole point is serving a
//! VMM. It is genuinely `Send + Sync` — no wasm `unsafe impl Send`.

#![cfg(not(target_arch = "wasm32"))]

mod filesystem;
mod inode;
mod server;

#[cfg(test)]
mod tests;

pub use filesystem::VfsFileSystem;
pub use server::{serve, VfsBackend};
