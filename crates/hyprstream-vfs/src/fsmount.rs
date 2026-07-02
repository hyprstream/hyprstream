//! `FsMount` — the writable / overlay filesystem supertrait.
//!
//! The base [`Mount`] trait (see `mount.rs`) is deliberately 9P-shaped and
//! read/write-only: it can `walk`/`open`/`read`/`write`/`readdir`/`stat`/`clunk`
//! a fid, and the only mutating semantics it carries are the open-mode bits
//! `OTRUNC` (truncate-on-open) and `ORCLOSE` (remove-on-clunk). That surface is
//! exactly right for synthetic, control, and stream mounts (`SyntheticTree`,
//! `EnvMount`, ctl files, `/stream`, …) and we keep it untouched.
//!
//! A *full* filesystem — a sandbox rootfs, a writable overlay, a passthrough of
//! a host directory — needs the rest of the POSIX namespace-mutation surface:
//! create files, unlink/rmdir, mkdir, rename, change attributes, symlinks and
//! hard links, plus the overlay-specific whiteout / opaque-dir primitives that
//! make copy-on-write layering observable. Rather than bloat `Mount` (and force
//! every existing read/write mount to grow `NotSupported` stubs), `FsMount` is a
//! **supertrait** that *extends* `Mount`. This is the hyprstream-owned overlay
//! interface: consumers code against `FsMount`, the v1 backend wraps
//! `fuse_backend_rs::OverlayFs` (see `overlay.rs`), and a future native
//! copy-on-write engine (backlog #370) can implement the very same trait with
//! zero consumer churn.
//!
//! ## Why a supertrait, not a `Mount` change
//!
//! - **Blast radius.** `SyntheticTree`, `EnvMount`, and friends stay on `Mount`
//!   verbatim. Only full-filesystem mounts (rootfs, overlay) opt in to
//!   `FsMount`. No existing impl needs to change.
//! - **Liskov.** Every `FsMount` *is* a `Mount`, so an `Arc<dyn FsMount>` can be
//!   coerced to the `Arc<dyn Mount>` a [`Namespace`](crate::Namespace) stores.
//!   The richer ops are reached by code that holds the concrete `FsMount`.
//! - **Capability typing.** "Is this mount writable as a real filesystem?" is
//!   answered by the type (`dyn FsMount`) rather than by probing for runtime
//!   `NotSupported` errors.
//!
//! ## Op semantics
//!
//! All ops are **path-addressed** (relative to the mount root, like
//! [`Mount::walk`]) and take the verified [`Subject`] of the caller for
//! per-tenant policy enforcement and fid isolation — the uniform
//! Subject-per-call boundary (#353/#319/#328) extended over the rootfs. Paths
//! are component slices, never raw strings, so a backend never has to re-parse
//! `/`-joined input. Mutating ops are **fail-closed**: a backend that cannot
//! satisfy an op safely must return an error, never silently succeed.

use async_trait::async_trait;
use hyprstream_rpc::Subject;

use crate::mount::{Mount, MountError, Stat};

// ─────────────────────────────────────────────────────────────────────────────
// Attribute change descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Attribute changes for [`FsMount::setattr`].
///
/// Each `Option` is "leave unchanged" when `None`. This mirrors the
/// `SetattrValid` bitmask used by FUSE/9P2000.L: only the populated fields are
/// applied, atomically where the backend supports it. Maps to chmod (`mode`),
/// chown (`uid`/`gid`), truncate (`size`), and utimes (`atime`/`mtime`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SetAttr {
    /// New permission bits (chmod). Lower 12 bits are the POSIX mode.
    pub mode: Option<u32>,
    /// New owner uid (chown).
    pub uid: Option<u32>,
    /// New owner gid (chown).
    pub gid: Option<u32>,
    /// New file length (truncate / extend).
    pub size: Option<u64>,
    /// New access time, seconds since the Unix epoch.
    pub atime: Option<u64>,
    /// New modification time, seconds since the Unix epoch.
    pub mtime: Option<u64>,
}

impl SetAttr {
    /// True when no field is set — the op would be a no-op.
    pub fn is_empty(&self) -> bool {
        self.mode.is_none()
            && self.uid.is_none()
            && self.gid.is_none()
            && self.size.is_none()
            && self.atime.is_none()
            && self.mtime.is_none()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FsMount supertrait
// ─────────────────────────────────────────────────────────────────────────────

/// A full, writable filesystem mount — the hyprstream overlay interface.
///
/// Extends [`Mount`] (walk/open/read/write/readdir/stat/clunk) with the
/// namespace-mutation surface a writable or overlay filesystem needs:
/// `create`, `unlink`/`rmdir`/`mkdir`, `rename`, `setattr`, `symlink`/`readlink`,
/// `link`, and the overlay whiteout / opaque-dir hooks.
///
/// The v1 backend ([`crate::overlay`]) wraps `fuse_backend_rs::OverlayFs`, which
/// implements copy-up and whiteouts internally — so an `FsMount` impl that
/// delegates to it does **not** hand-roll copy-on-write. A future native CoW
/// engine (backlog #370) implements this same trait directly.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait FsMount: Mount {
    /// Create (and leave on disk) a regular file at `path` with `mode`.
    ///
    /// Fails with [`MountError::AlreadyExists`] if the final component exists.
    /// On an overlay this triggers copy-up of the parent directory as needed and
    /// places the new file in the writable upper layer.
    async fn create(&self, path: &[&str], mode: u32, caller: &Subject) -> Result<(), MountError>;

    /// Remove a non-directory entry (file, symlink, fifo, …) at `path`.
    ///
    /// On an overlay, removing an entry that exists only in a lower (read-only)
    /// layer is realised as a **whiteout** in the upper layer — the lower entry
    /// is masked, never mutated.
    async fn unlink(&self, path: &[&str], caller: &Subject) -> Result<(), MountError>;

    /// Create a directory at `path` with `mode`.
    async fn mkdir(&self, path: &[&str], mode: u32, caller: &Subject) -> Result<(), MountError>;

    /// Remove an empty directory at `path`.
    ///
    /// Fails with [`MountError::NotDirectory`] if `path` is not a directory and
    /// (per POSIX) an error if it is non-empty. Overlay whiteout rules apply as
    /// for [`unlink`](Self::unlink).
    async fn rmdir(&self, path: &[&str], caller: &Subject) -> Result<(), MountError>;

    /// Rename `from` to `to`, atomically replacing `to` if it exists.
    ///
    /// `from` and `to` are both rooted at the mount root and may live in
    /// different directories. Across overlay layers this implies copy-up of the
    /// source plus a whiteout at the old location, all handled by the backend.
    async fn rename(&self, from: &[&str], to: &[&str], caller: &Subject) -> Result<(), MountError>;

    /// Change file attributes (chmod / chown / truncate / utimes).
    ///
    /// Only the populated fields of [`SetAttr`] are applied. A `size` change on
    /// a lower-layer file forces copy-up.
    async fn setattr(&self, path: &[&str], attr: &SetAttr, caller: &Subject) -> Result<(), MountError>;

    /// Create a symbolic link at `path` whose target is `target`.
    ///
    /// `target` is stored verbatim (it may be absolute or relative and is *not*
    /// resolved by this call).
    async fn symlink(&self, path: &[&str], target: &str, caller: &Subject) -> Result<(), MountError>;

    /// Read the target of the symbolic link at `path`.
    async fn readlink(&self, path: &[&str], caller: &Subject) -> Result<String, MountError>;

    /// Create a hard link at `new_path` referring to the same inode as
    /// `existing`.
    async fn link(&self, existing: &[&str], new_path: &[&str], caller: &Subject) -> Result<(), MountError>;

    // ── Overlay / copy-on-write hooks ───────────────────────────────────────
    //
    // Default implementations let non-overlay full filesystems (e.g. a plain
    // passthrough) ignore overlay semantics. The OverlayFs-backed v1 backend
    // applies real whiteouts/opaque-dirs internally; these hooks expose that
    // state to callers (e.g. FS-A's down-adapter) and let a native CoW engine
    // implement them explicitly.

    /// Report whether `path` resolves to a whiteout — an upper-layer marker that
    /// masks an entry present in a lower layer. Default: `false` (no overlay).
    async fn is_whiteout(&self, _path: &[&str], _caller: &Subject) -> Result<bool, MountError> {
        Ok(false)
    }

    /// Report whether the directory at `path` is *opaque* — i.e. it hides the
    /// contents of the same directory in all lower layers. Default: `false`.
    async fn is_opaque(&self, _path: &[&str], _caller: &Subject) -> Result<bool, MountError> {
        Ok(false)
    }

    /// Mark the directory at `path` opaque, hiding lower-layer contents.
    ///
    /// Default: [`MountError::NotSupported`] — a backend with no layering cannot
    /// honour opacity and must say so rather than silently no-op (fail-closed).
    async fn set_opaque(&self, _path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        Err(MountError::NotSupported("set_opaque: backend is not an overlay".into()))
    }

    /// Stat a `path` directly (without holding a fid), returning its metadata.
    ///
    /// Convenience over `walk` + `stat` + `clunk` for the mutating ops above,
    /// which routinely need to probe existence/type. Backends should implement
    /// it in terms of their own fid machinery.
    async fn stat_path(&self, path: &[&str], caller: &Subject) -> Result<Stat, MountError>;
}
