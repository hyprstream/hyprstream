//! DAX (Direct Access) mount trait for zero-copy file access.
//!
//! Mounts that can provide a backing file descriptor enable virtio-fs DAX
//! window mapping, allowing the guest to mmap model weights directly from
//! the host without copying through the FUSE data path.

use hyprstream_vfs::{Fid, Mount};

/// Extension trait for mounts that support DAX (direct memory-mapped access).
///
/// When a mount implements this trait, the VfsFuse adapter can use
/// `setupmapping` / `removemapping` to map file regions directly into
/// the guest's DAX window, bypassing FUSE read/write for large files.
///
/// Primary use case: model weight files (.safetensors) where zero-copy
/// access from the guest eliminates data path overhead.
#[cfg(unix)]
pub trait DaxMount: Mount {
    /// Return a file descriptor backing the given fid, if DAX-capable.
    ///
    /// Returns `None` if the fid does not support direct mapping (e.g.,
    /// synthetic ctl files, directories, or dynamic content).
    fn backing_fd(&self, fid: &Fid) -> Option<std::os::fd::OwnedFd>;
}
