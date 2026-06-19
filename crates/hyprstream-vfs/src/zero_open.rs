//! `ZeroOpenLayer` — make a *handleless* `FileSystem`/`Layer` usable as an
//! `OverlayFs` lower (native only).
//!
//! ## Why
//!
//! `fuse_backend_rs::overlayfs::OverlayFs` is handle-based: when it serves
//! `open`/`read` it calls the underlying layer's `open` and **requires it to
//! return `Some(handle)`** — a `None` handle is treated as ENOENT (see
//! `OverlayFs::open`). Some filesystems are *handleless* ("zero-message-open"):
//! they return `(None, KEEP_CACHE, _)` from `open` and address reads by inode.
//! The motivating case is **RAFS** (`nydus_rafs::fs::Rafs`), the FS-B (#363)
//! rootfs lower: its `open` returns no handle, so used directly as an OverlayFs
//! lower every read of an image file fails with ENOENT.
//!
//! ## What
//!
//! [`ZeroOpenLayer`] wraps any [`Layer`] and substitutes a constant handle (`0`)
//! whenever the inner `open`/`opendir` returns `None`, leaving every other op a
//! straight delegation. The substituted handle is harmless: a handleless backend
//! ignores the handle on `read`/`readdir` (it resolves by inode), and on
//! `release`/`releasedir` the wrapper passes it straight back. The overlay stays
//! in its normal, production (Kata) handle mode — copy-up still routes writes to
//! the writable upper layer (which keeps real handles); only the read-only lower
//! is wrapped.
//!
//! This keeps the version-pinned overlay engine and the up-adapter unchanged: it
//! is purely a lower-layer shim.

use std::ffi::CStr;
use std::io;
use std::time::Duration;

use fuse_backend_rs::abi::fuse_abi::{stat64, statvfs64};
use fuse_backend_rs::api::filesystem::{
    Context, DirEntry, Entry, FileSystem, FsOptions, GetxattrReply, Layer, ListxattrReply,
    OpenOptions, SetattrValid, ZeroCopyReader, ZeroCopyWriter,
};

/// Wrap a handleless [`Layer`] so it presents a non-`None` handle to
/// `OverlayFs`. See the module docs.
pub struct ZeroOpenLayer<L>
where
    L: Layer<Inode = u64, Handle = u64> + Send + Sync,
{
    inner: L,
}

impl<L> ZeroOpenLayer<L>
where
    L: Layer<Inode = u64, Handle = u64> + Send + Sync,
{
    /// Wrap `inner`. The inner layer must already be ready to serve (e.g. a
    /// RAFS whose `import` has run).
    pub fn new(inner: L) -> Self {
        Self { inner }
    }
}

impl<L> FileSystem for ZeroOpenLayer<L>
where
    L: Layer<Inode = u64, Handle = u64> + Send + Sync,
{
    type Inode = u64;
    type Handle = u64;

    fn init(&self, capable: FsOptions) -> io::Result<FsOptions> {
        self.inner.init(capable)
    }

    fn destroy(&self) {
        self.inner.destroy()
    }

    fn lookup(&self, ctx: &Context, parent: u64, name: &CStr) -> io::Result<Entry> {
        self.inner.lookup(ctx, parent, name)
    }

    fn forget(&self, ctx: &Context, inode: u64, count: u64) {
        self.inner.forget(ctx, inode, count)
    }

    fn batch_forget(&self, ctx: &Context, requests: Vec<(u64, u64)>) {
        self.inner.batch_forget(ctx, requests)
    }

    fn getattr(
        &self,
        ctx: &Context,
        inode: u64,
        handle: Option<u64>,
    ) -> io::Result<(stat64, Duration)> {
        self.inner.getattr(ctx, inode, handle)
    }

    fn setattr(
        &self,
        ctx: &Context,
        inode: u64,
        attr: stat64,
        handle: Option<u64>,
        valid: SetattrValid,
    ) -> io::Result<(stat64, Duration)> {
        self.inner.setattr(ctx, inode, attr, handle, valid)
    }

    fn readlink(&self, ctx: &Context, inode: u64) -> io::Result<Vec<u8>> {
        self.inner.readlink(ctx, inode)
    }

    fn unlink(&self, ctx: &Context, parent: u64, name: &CStr) -> io::Result<()> {
        self.inner.unlink(ctx, parent, name)
    }

    fn rmdir(&self, ctx: &Context, parent: u64, name: &CStr) -> io::Result<()> {
        self.inner.rmdir(ctx, parent, name)
    }

    /// Substitute a constant handle when the inner layer is handleless.
    fn open(
        &self,
        ctx: &Context,
        inode: u64,
        flags: u32,
        fuse_flags: u32,
    ) -> io::Result<(Option<u64>, OpenOptions, Option<u32>)> {
        let (handle, opts, passthrough) = self.inner.open(ctx, inode, flags, fuse_flags)?;
        Ok((Some(handle.unwrap_or(0)), opts, passthrough))
    }

    #[allow(clippy::too_many_arguments)]
    fn read(
        &self,
        ctx: &Context,
        inode: u64,
        handle: u64,
        w: &mut dyn ZeroCopyWriter,
        size: u32,
        offset: u64,
        lock_owner: Option<u64>,
        flags: u32,
    ) -> io::Result<usize> {
        self.inner
            .read(ctx, inode, handle, w, size, offset, lock_owner, flags)
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        ctx: &Context,
        inode: u64,
        handle: u64,
        r: &mut dyn ZeroCopyReader,
        size: u32,
        offset: u64,
        lock_owner: Option<u64>,
        delayed_write: bool,
        flags: u32,
        fuse_flags: u32,
    ) -> io::Result<usize> {
        self.inner.write(
            ctx,
            inode,
            handle,
            r,
            size,
            offset,
            lock_owner,
            delayed_write,
            flags,
            fuse_flags,
        )
    }

    fn release(
        &self,
        ctx: &Context,
        inode: u64,
        flags: u32,
        handle: u64,
        flush: bool,
        flock_release: bool,
        lock_owner: Option<u64>,
    ) -> io::Result<()> {
        self.inner
            .release(ctx, inode, flags, handle, flush, flock_release, lock_owner)
    }

    fn statfs(&self, ctx: &Context, inode: u64) -> io::Result<statvfs64> {
        self.inner.statfs(ctx, inode)
    }

    fn getxattr(
        &self,
        ctx: &Context,
        inode: u64,
        name: &CStr,
        size: u32,
    ) -> io::Result<GetxattrReply> {
        self.inner.getxattr(ctx, inode, name, size)
    }

    fn listxattr(&self, ctx: &Context, inode: u64, size: u32) -> io::Result<ListxattrReply> {
        self.inner.listxattr(ctx, inode, size)
    }

    /// Substitute a constant handle when the inner layer is handleless.
    fn opendir(
        &self,
        ctx: &Context,
        inode: u64,
        flags: u32,
    ) -> io::Result<(Option<u64>, OpenOptions)> {
        let (handle, opts) = self.inner.opendir(ctx, inode, flags)?;
        Ok((Some(handle.unwrap_or(0)), opts))
    }

    fn readdir(
        &self,
        ctx: &Context,
        inode: u64,
        handle: u64,
        size: u32,
        offset: u64,
        add_entry: &mut dyn FnMut(DirEntry) -> io::Result<usize>,
    ) -> io::Result<()> {
        self.inner
            .readdir(ctx, inode, handle, size, offset, add_entry)
    }

    fn readdirplus(
        &self,
        ctx: &Context,
        inode: u64,
        handle: u64,
        size: u32,
        offset: u64,
        add_entry: &mut dyn FnMut(DirEntry, Entry) -> io::Result<usize>,
    ) -> io::Result<()> {
        self.inner
            .readdirplus(ctx, inode, handle, size, offset, add_entry)
    }

    fn releasedir(&self, ctx: &Context, inode: u64, flags: u32, handle: u64) -> io::Result<()> {
        self.inner.releasedir(ctx, inode, flags, handle)
    }

    fn access(&self, ctx: &Context, inode: u64, mask: u32) -> io::Result<()> {
        self.inner.access(ctx, inode, mask)
    }
}

impl<L> Layer for ZeroOpenLayer<L>
where
    L: Layer<Inode = u64, Handle = u64> + Send + Sync,
{
    fn root_inode(&self) -> u64 {
        self.inner.root_inode()
    }
}
