//! FUSE/virtio-fs adapter bridging VFS Mount trait to guest VMs.
//!
//! This crate implements a FUSE filesystem backed by [`hyprstream_vfs::Mount`],
//! allowing the VFS namespace to be exposed to guest VMs via virtio-fs.
//!
//! Linux-only: depends on fuse-backend-rs for the FUSE protocol implementation.
//!
//! # Architecture
//!
//! ```text
//!   Guest VM (virtio-fs mount)
//!        |
//!   vhost-user-fs socket
//!        |
//!   fuse-backend-rs (FUSE protocol)
//!        |
//!   VfsFuse<M: Mount> (this crate)
//!        |
//!   hyprstream-vfs Mount trait
//! ```

#[cfg(target_os = "linux")]
mod dax;
#[cfg(target_os = "linux")]
mod daemon;
#[cfg(target_os = "linux")]
mod fuse_fs;
#[cfg(target_os = "linux")]
mod inode_table;

#[cfg(target_os = "linux")]
pub use daemon::VfsDaemon;
#[cfg(target_os = "linux")]
pub use dax::DaxMount;
#[cfg(target_os = "linux")]
pub use fuse_fs::VfsFuse;
#[cfg(target_os = "linux")]
pub use inode_table::{InodeData, InodeTable};
