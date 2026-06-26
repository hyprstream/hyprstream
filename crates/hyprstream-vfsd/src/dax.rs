//! DAX (Direct Access) mount trait for zero-copy file access.
//!
//! Mounts that can provide a backing file descriptor enable virtio-fs DAX
//! window mapping, allowing the guest to mmap model weights directly from
//! the host without copying through the FUSE data path.

use std::os::fd::OwnedFd;

use hyprstream_vfs::{Fid, Mount};

/// Extension trait for mounts that support DAX (direct memory-mapped access).
///
/// When a mount implements this trait, the [`crate::VfsFuse`] adapter can use
/// `setupmapping` / `removemapping` to map file regions directly into the
/// guest's DAX window, bypassing FUSE read/write for large files.
///
/// Primary use case: model weight files (`.safetensors`) where zero-copy
/// access from the guest eliminates data path overhead and double page-cache
/// buffering.
///
/// # Permissions — SECURITY
///
/// The returned descriptor's open mode IS the enforcement boundary for DAX
/// writes: a read-only file MUST return a descriptor opened `O_RDONLY` so a
/// guest cannot obtain a writable mapping of host memory it may not modify.
/// The adapter additionally rejects writable mappings against read-only opens
/// as defense-in-depth (see `VfsFuse::setupmapping`).
pub trait DaxMount: Mount {
    /// Return a file descriptor backing the given fid, if DAX-capable.
    ///
    /// Returns `None` if the fid does not support direct mapping (e.g.
    /// synthetic ctl files, directories, or dynamic content); the adapter
    /// then falls back to the normal FUSE read/write path.
    fn backing_fd(&self, fid: &Fid) -> Option<OwnedFd>;
}

/// Object-safe erasure of [`DaxMount`] so that [`crate::VfsFuse`], which is
/// generic over `M: Mount`, can hold an optional DAX capability without
/// constraining every `VfsFuse<M>` to `M: DaxMount`.
///
/// A `VfsFuse` built from a plain `Mount` carries `dax = None` and transparently
/// reports `ENOSYS` for mapping requests; one built via
/// [`crate::VfsFuse::with_dax`] (where `M: DaxMount`) stores a second `Arc`
/// to the same mount as `Arc<dyn ErasedDax>`. DAX is therefore an *optional
/// capability of the adapter*, not a special-cased second code path.
pub(crate) trait ErasedDax: Send + Sync {
    fn backing_fd(&self, fid: &Fid) -> Option<OwnedFd>;
}

impl<T: DaxMount + Send + Sync> ErasedDax for T {
    fn backing_fd(&self, fid: &Fid) -> Option<OwnedFd> {
        DaxMount::backing_fd(self, fid)
    }
}
