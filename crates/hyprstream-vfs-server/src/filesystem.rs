//! `Namespace → FileSystem` **down-adapter** (native only).
//!
//! This is the inverse of FS-C's [`FuseFileSystemMount`](hyprstream_vfs::FuseFileSystemMount)
//! up-adapter. Where the up-adapter exposes a `fuse_backend_rs` `FileSystem`
//! (RAFS / OverlayFs / Passthrough) *as* a VFS [`Mount`], this adapter exposes a
//! whole VFS [`Namespace`] *as* a `fuse_backend_rs`
//! [`FileSystem`](fuse_backend_rs::api::filesystem::FileSystem) — so the
//! namespace can be served over vhost-user-fs to a Cloud Hypervisor guest (see
//! [`crate::server`]).
//!
//! ## Path model
//!
//! `FileSystem` is inode/handle addressed; the VFS is path addressed. The
//! [`InodeTable`](crate::inode::InodeTable) owns the translation: inode `1`
//! (`FUSE_ROOT_ID`) is the namespace root, every other inode is interned lazily
//! on `lookup`/`create`/`mkdir` and reference counted so `forget` reclaims it.
//! [`HandleTable`](crate::inode::HandleTable) holds the VFS [`Fid`] for each open
//! file/dir.
//!
//! ## Routing
//!
//! A path resolves through [`Namespace::resolve_targets`] to the ordered mount
//! targets (bind order) plus the components relative to the mount root. Read
//! ops (`open`/`read`/`opendir`/`readdir`) drive the base [`Mount`] surface,
//! trying each target in bind order until one succeeds — the same union /
//! fallthrough policy the namespace's own `cat`/`ls` helpers use. Mutating ops
//! (`create`/`unlink`/`mkdir`/`rmdir`/`rename`/`setattr`/`symlink`/`link`)
//! require the writable [`FsMount`] capability, reached via
//! [`Mount::as_fsmount`]; a target that is a plain `Mount` (synthetic, ctl,
//! stream mounts) is **read-only** and such ops return `EROFS` — fail-closed,
//! never a silent no-op. Paths the namespace spans only implicitly (intermediate
//! directories, e.g. `/srv` when only `/srv/model` is mounted) are served as
//! synthetic read-only directories.
//!
//! ## Subject threading
//!
//! Every VFS call carries the verified [`Subject`] the server was constructed
//! with — the uniform Subject-per-call policy/audit boundary (#353/#319/#328)
//! extended across the guest filesystem. The FUSE [`Context`] (host uid/gid) is
//! *not* the identity; the `Subject` is.
//!
//! ## async → sync bridge
//!
//! `FileSystem` is synchronous; the VFS `Mount`/`FsMount` ops are `async`. We
//! bridge with a tokio runtime [`Handle`], running each op on it via
//! [`Handle::block_on`]. The server's vring threads are plain OS threads (not
//! tokio workers), so blocking on a multi-thread runtime handle is safe and does
//! not stall the async executor.
//!
//! ## Send + Sync
//!
//! Genuinely `Send + Sync`: all state is `Arc`/`Mutex`-guarded and the wrapped
//! `Namespace` is `Send + Sync`. No wasm `unsafe impl Send` — the crate is
//! native-only.

use std::ffi::CStr;
use std::io;
use std::time::Duration;

use fuse_backend_rs::abi::fuse_abi::CreateIn;
use fuse_backend_rs::api::filesystem::{
    Context, DirEntry as FuseDirEntry, Entry, FileSystem, FsOptions, GetxattrReply, ListxattrReply,
    OpenOptions, SetattrValid, ZeroCopyReader, ZeroCopyWriter,
};
use tokio::runtime::Handle;

use hyprstream_vfs::{
    DirEntry, FsMount, MountError, MountTarget, Namespace, SetAttr, Subject, ORDWR, OWRITE,
};

use crate::inode::{HandleTable, InodeData, InodeTable, Kind, OpenFid, ROOT_INODE};

/// Attribute / entry validity timeout handed to the guest kernel.
const TTL: Duration = Duration::from_secs(1);
/// Block size reported in `stat`.
const BLOCK_SIZE: u32 = 512;

/// Map a [`MountError`] to the `errno` the guest kernel expects.
fn errno(err: &MountError) -> i32 {
    match err {
        MountError::NotFound(_) => libc::ENOENT,
        MountError::PermissionDenied(_) => libc::EACCES,
        MountError::NotDirectory(_) => libc::ENOTDIR,
        MountError::IsDirectory(_) => libc::EISDIR,
        MountError::InvalidArgument(_) => libc::EINVAL,
        MountError::NotSupported(_) => libc::ENOSYS,
        MountError::AlreadyExists(_) => libc::EEXIST,
        MountError::Io(_) => libc::EIO,
    }
}

fn io_err(err: &MountError) -> io::Error {
    io::Error::from_raw_os_error(errno(err))
}

fn einval() -> io::Error {
    io::Error::from_raw_os_error(libc::EINVAL)
}

fn erofs() -> io::Error {
    io::Error::from_raw_os_error(libc::EROFS)
}

/// The down-adapter: a `fuse_backend_rs` [`FileSystem`] over a VFS [`Namespace`].
pub struct VfsFileSystem {
    ns: Namespace,
    subject: Subject,
    inodes: InodeTable,
    handles: HandleTable,
    rt: Handle,
}

impl VfsFileSystem {
    /// Wrap `ns`, serving every op as `subject`, bridging async ops onto `rt`.
    pub fn new(ns: Namespace, subject: Subject, rt: Handle) -> Self {
        Self {
            ns,
            subject,
            inodes: InodeTable::new(),
            handles: HandleTable::new(),
            rt,
        }
    }

    /// Borrow the wrapped namespace (e.g. for FS-D to inspect mounts).
    pub fn namespace(&self) -> &Namespace {
        &self.ns
    }

    /// Run an async VFS future to completion on the adapter's runtime.
    fn block<F: std::future::Future>(&self, fut: F) -> F::Output {
        self.rt.block_on(fut)
    }

    /// The `/`-joined absolute path for an inode (for routing through the
    /// namespace, which is string-prefix addressed at its public surface).
    fn abs_path(components: &[String]) -> String {
        if components.is_empty() {
            "/".to_owned()
        } else {
            format!("/{}", components.join("/"))
        }
    }

    /// Resolve `inode` → its [`InodeData`], or `ENOENT`.
    fn inode_data(&self, inode: u64) -> io::Result<InodeData> {
        self.inodes
            .get(inode)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))
    }

    /// Resolve the child path components of `parent` + `name`.
    fn child_components(&self, parent: u64, name: &CStr) -> io::Result<Vec<String>> {
        let parent = self.inode_data(parent)?;
        let name = name.to_str().map_err(|_| einval())?;
        if name.is_empty() || name == "." || name.contains('/') {
            return Err(einval());
        }
        let mut comps = parent.components.clone();
        comps.push(name.to_owned());
        Ok(comps)
    }

    /// Route a path to its mount targets (bind order) + mount-relative
    /// components. Maps a namespace miss to `ENOENT`.
    fn route(&self, components: &[String]) -> io::Result<(Vec<MountTarget>, Vec<String>)> {
        let path = Self::abs_path(components);
        self.ns
            .resolve_targets(&path)
            .map_err(|_| io::Error::from_raw_os_error(libc::ENOENT))
    }

    /// Stat a path through the namespace, trying each target in bind order.
    /// Returns the VFS [`hyprstream_vfs::Stat`] plus whether it is a directory.
    fn stat_path(&self, components: &[String]) -> io::Result<(hyprstream_vfs::Stat, bool)> {
        // Intermediate (implicitly-spanned) directories: no backing mount, but
        // the namespace knows they exist as ancestors of real mounts.
        let path = Self::abs_path(components);
        if self.ns.intermediate_children(&path).is_some() {
            return Ok((
                hyprstream_vfs::Stat {
                    qtype: 0x80,
                    version: 0,
                    path: 0,
                    size: 0,
                    name: components.last().cloned().unwrap_or_default(),
                    mtime: 0,
                },
                true,
            ));
        }

        let (targets, comps) = self.route(components)?;
        // A path that resolves to a mount root (no components left after the
        // prefix) is the mount point itself — always a directory. Don't require
        // the backing mount to stat its own root.
        if comps.is_empty() {
            return Ok((
                hyprstream_vfs::Stat {
                    qtype: 0x80,
                    version: 0,
                    path: 0,
                    size: 0,
                    name: components.last().cloned().unwrap_or_default(),
                    mtime: 0,
                },
                true,
            ));
        }
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let mut last = MountError::NotFound(path.clone());
        for mount in &targets {
            let res = self.block(async {
                let fid = mount.walk(&comp_refs, &self.subject).await?;
                let st = mount.stat(&fid, &self.subject).await;
                mount.clunk(fid, &self.subject).await;
                st
            });
            match res {
                Ok(st) => {
                    let is_dir = st.qtype & 0x80 != 0;
                    return Ok((st, is_dir));
                }
                Err(e) => last = e,
            }
        }
        Err(io_err(&last))
    }

    /// Synthesise a `libc::stat64` for an inode from a VFS stat + kind.
    fn fill_attr(inode: u64, st: &hyprstream_vfs::Stat, kind: Kind) -> libc::stat64 {
        // SAFETY: `stat64` is plain old data; zeroing is a valid initial value.
        let mut attr: libc::stat64 = unsafe { std::mem::zeroed() };
        attr.st_ino = inode;
        attr.st_size = st.size as i64;
        attr.st_blksize = BLOCK_SIZE as i64;
        attr.st_blocks = st.size.div_ceil(BLOCK_SIZE as u64) as i64;
        attr.st_nlink = 1;
        let (typ, perm) = match kind {
            Kind::Dir => (libc::S_IFDIR, 0o755),
            Kind::File => (libc::S_IFREG, 0o644),
            Kind::Symlink => (libc::S_IFLNK, 0o777),
        };
        attr.st_mode = typ | perm;
        attr.st_mtime = st.mtime as i64;
        attr.st_ctime = st.mtime as i64;
        attr.st_atime = st.mtime as i64;
        attr
    }

    /// Build a FUSE [`Entry`] for `inode`/`kind`/`stat`.
    fn make_entry(&self, inode: u64, kind: Kind, st: &hyprstream_vfs::Stat) -> Entry {
        Entry {
            inode,
            generation: 0,
            attr: Self::fill_attr(inode, st, kind),
            attr_flags: 0,
            attr_timeout: TTL,
            entry_timeout: TTL,
        }
    }

    /// Resolve the writable [`FsMount`] for a path, or `EROFS` if no target at
    /// the path is a full filesystem. Returns the mount plus mount-relative
    /// components. (Mutations don't union: the first FsMount target wins.)
    fn route_writable(
        &self,
        components: &[String],
    ) -> io::Result<(MountTarget, Vec<String>)> {
        let (targets, comps) = self.route(components)?;
        for mount in targets {
            if mount.as_fsmount().is_some() {
                return Ok((mount, comps));
            }
        }
        Err(erofs())
    }

}

impl FileSystem for VfsFileSystem {
    type Inode = u64;
    type Handle = u64;

    fn init(&self, _capable: FsOptions) -> io::Result<FsOptions> {
        // We negotiate no special capabilities; the default option set is fine.
        Ok(FsOptions::empty())
    }

    fn lookup(&self, _ctx: &Context, parent: Self::Inode, name: &CStr) -> io::Result<Entry> {
        let components = self.child_components(parent, name)?;
        let (st, is_dir) = self.stat_path(&components)?;
        let kind = if is_dir { Kind::Dir } else { Kind::File };
        let inode = self.inodes.intern(components, kind);
        Ok(self.make_entry(inode, kind, &st))
    }

    fn forget(&self, _ctx: &Context, inode: Self::Inode, count: u64) {
        self.inodes.forget(inode, count);
    }

    fn batch_forget(&self, _ctx: &Context, requests: Vec<(Self::Inode, u64)>) {
        for (inode, count) in requests {
            self.inodes.forget(inode, count);
        }
    }

    fn getattr(
        &self,
        _ctx: &Context,
        inode: Self::Inode,
        _handle: Option<Self::Handle>,
    ) -> io::Result<(libc::stat64, Duration)> {
        let data = self.inode_data(inode)?;
        if inode == ROOT_INODE {
            let st = hyprstream_vfs::Stat { qtype: 0x80, version: 0, path: 0, size: 0, name: String::new(), mtime: 0 };
            return Ok((Self::fill_attr(inode, &st, Kind::Dir), TTL));
        }
        let (st, is_dir) = self.stat_path(&data.components)?;
        let kind = if is_dir { Kind::Dir } else { data.kind };
        Ok((Self::fill_attr(inode, &st, kind), TTL))
    }

    fn opendir(
        &self,
        _ctx: &Context,
        inode: Self::Inode,
        _flags: u32,
    ) -> io::Result<(Option<Self::Handle>, OpenOptions)> {
        let data = self.inode_data(inode)?;
        // Root and intermediate directories have no backing mount fid; serve them
        // synthetically (handle 0 ⇒ no stored fid).
        let path = Self::abs_path(&data.components);
        if inode == ROOT_INODE || self.ns.intermediate_children(&path).is_some() {
            return Ok((Some(0), OpenOptions::empty()));
        }
        let (targets, comps) = self.route(&data.components)?;
        // Mount-root directories are served synthetically (readdir merges via the
        // namespace `ls`, which unions all targets); no backing dir fid needed.
        if comps.is_empty() {
            return Ok((Some(0), OpenOptions::empty()));
        }
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let mut last = MountError::NotFound(path);
        for mount in &targets {
            let res = self.block(async {
                let mut fid = mount.walk(&comp_refs, &self.subject).await?;
                mount.open(&mut fid, 0, &self.subject).await?;
                Ok::<_, MountError>(fid)
            });
            match res {
                Ok(fid) => {
                    let handle = self.handles.insert(OpenFid {
                        mount: mount.clone(),
                        fid,
                    });
                    return Ok((Some(handle), OpenOptions::empty()));
                }
                Err(e) => last = e,
            }
        }
        Err(io_err(&last))
    }

    fn readdir(
        &self,
        _ctx: &Context,
        inode: Self::Inode,
        _handle: Self::Handle,
        _size: u32,
        offset: u64,
        add_entry: &mut dyn FnMut(FuseDirEntry) -> io::Result<usize>,
    ) -> io::Result<()> {
        let data = self.inode_data(inode)?;
        let path = Self::abs_path(&data.components);
        // Merge real mount entries with any synthetic (intermediate) children,
        // matching the namespace's own union-readdir semantics.
        let entries: Vec<DirEntry> = self
            .block(self.ns.ls(&path, &self.subject))
            .map_err(|_| io::Error::from_raw_os_error(libc::ENOENT))?;

        // `.`/`..` first, then entries, honouring the offset cursor.
        let mut all: Vec<(String, bool, u64)> =
            vec![(".".to_owned(), true, inode), ("..".to_owned(), true, ROOT_INODE)];
        for e in entries {
            // Intern child inodes so a subsequent lookup is stable.
            let mut comps = data.components.clone();
            comps.push(e.name.clone());
            let kind = if e.is_dir { Kind::Dir } else { Kind::File };
            let child = self.inodes.intern(comps, kind);
            all.push((e.name, e.is_dir, child));
        }

        for (idx, (name, is_dir, ino)) in all.into_iter().enumerate().skip(offset as usize) {
            let typ = if is_dir { libc::DT_DIR } else { libc::DT_REG } as u32;
            let written = add_entry(FuseDirEntry {
                ino,
                offset: idx as u64 + 1,
                type_: typ,
                name: name.as_bytes(),
            })?;
            if written == 0 {
                break;
            }
        }
        Ok(())
    }

    fn releasedir(
        &self,
        _ctx: &Context,
        _inode: Self::Inode,
        _flags: u32,
        handle: Self::Handle,
    ) -> io::Result<()> {
        if handle != 0 {
            if let Some(open) = self.handles.remove(handle) {
                self.block(open.mount.clunk(open.fid, &self.subject));
            }
        }
        Ok(())
    }

    fn open(
        &self,
        _ctx: &Context,
        inode: Self::Inode,
        flags: u32,
        _fuse_flags: u32,
    ) -> io::Result<(Option<Self::Handle>, OpenOptions, Option<u32>)> {
        let data = self.inode_data(inode)?;
        let mode = match flags & libc::O_ACCMODE as u32 {
            x if x == libc::O_WRONLY as u32 => OWRITE,
            x if x == libc::O_RDWR as u32 => ORDWR,
            _ => 0, // OREAD
        };
        let (targets, comps) = self.route(&data.components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let mut last = MountError::NotFound(Self::abs_path(&data.components));
        for mount in &targets {
            // A write open against a read-only (non-FsMount) target fails closed.
            if mode != 0 && mount.as_fsmount().is_none() {
                last = MountError::PermissionDenied("read-only mount".into());
                continue;
            }
            let res = self.block(async {
                let mut fid = mount.walk(&comp_refs, &self.subject).await?;
                mount.open(&mut fid, mode, &self.subject).await?;
                Ok::<_, MountError>(fid)
            });
            match res {
                Ok(fid) => {
                    let handle = self.handles.insert(OpenFid {
                        mount: mount.clone(),
                        fid,
                    });
                    return Ok((Some(handle), OpenOptions::empty(), None));
                }
                Err(e) => last = e,
            }
        }
        if mode != 0 {
            Err(erofs())
        } else {
            Err(io_err(&last))
        }
    }

    fn read(
        &self,
        _ctx: &Context,
        _inode: Self::Inode,
        handle: Self::Handle,
        w: &mut dyn ZeroCopyWriter,
        size: u32,
        offset: u64,
        _lock_owner: Option<u64>,
        _flags: u32,
    ) -> io::Result<usize> {
        // Take the open fid out for the await, then return it.
        let open = self.handles.take(handle).ok_or_else(einval)?;
        let res = self.block(open.mount.read(&open.fid, offset, size, &self.subject));
        self.handles.put(handle, open);
        let data = res.map_err(|e| io_err(&e))?;
        w.write_all(&data)?;
        Ok(data.len())
    }

    fn write(
        &self,
        _ctx: &Context,
        _inode: Self::Inode,
        handle: Self::Handle,
        r: &mut dyn ZeroCopyReader,
        size: u32,
        offset: u64,
        _lock_owner: Option<u64>,
        _delayed_write: bool,
        _flags: u32,
        _fuse_flags: u32,
    ) -> io::Result<usize> {
        // Stage the incoming bytes into a heap buffer (the VFS write API is
        // buffer-based), then forward to the mount.
        let mut buf = vec![0u8; size as usize];
        let mut filled = 0;
        while filled < buf.len() {
            let n = r.read(&mut buf[filled..])?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        buf.truncate(filled);

        let open = self.handles.take(handle).ok_or_else(einval)?;
        let res = self.block(open.mount.write(&open.fid, offset, &buf, &self.subject));
        self.handles.put(handle, open);
        let n = res.map_err(|e| io_err(&e))?;
        Ok(n as usize)
    }

    fn release(
        &self,
        _ctx: &Context,
        _inode: Self::Inode,
        _flags: u32,
        handle: Self::Handle,
        _flush: bool,
        _flock_release: bool,
        _lock_owner: Option<u64>,
    ) -> io::Result<()> {
        if let Some(open) = self.handles.remove(handle) {
            self.block(open.mount.clunk(open.fid, &self.subject));
        }
        Ok(())
    }

    fn create(
        &self,
        _ctx: &Context,
        parent: Self::Inode,
        name: &CStr,
        args: CreateIn,
    ) -> io::Result<(Entry, Option<Self::Handle>, OpenOptions, Option<u32>)> {
        let components = self.child_components(parent, name)?;
        let (mount, comps) = self.route_writable(&components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let mode = args.mode & 0o7777;
        // Create, then re-open for writing on the same mount.
        let fid = self.block(async {
            let fs = mount.as_fsmount().ok_or(erofs())?;
            FsMount::create(fs, &comp_refs, mode, &self.subject)
                .await
                .map_err(|e| io_err(&e))?;
            let mut fid = mount.walk(&comp_refs, &self.subject).await.map_err(|e| io_err(&e))?;
            mount
                .open(&mut fid, OWRITE, &self.subject)
                .await
                .map_err(|e| io_err(&e))?;
            Ok::<_, io::Error>(fid)
        })?;
        let (st, _) = self.stat_path(&components)?;
        let inode = self.inodes.intern(components, Kind::File);
        let entry = self.make_entry(inode, Kind::File, &st);
        let handle = self.handles.insert(OpenFid { mount, fid });
        Ok((entry, Some(handle), OpenOptions::empty(), None))
    }

    fn unlink(&self, _ctx: &Context, parent: Self::Inode, name: &CStr) -> io::Result<()> {
        let components = self.child_components(parent, name)?;
        let (mount, comps) = self.route_writable(&components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let r = self.block(async {
            mount
                .as_fsmount()
                .ok_or(erofs())?
                .unlink(&comp_refs, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        });
        if r.is_ok() {
            self.inodes.invalidate_path(&components);
        }
        r
    }

    fn mkdir(
        &self,
        _ctx: &Context,
        parent: Self::Inode,
        name: &CStr,
        mode: u32,
        _umask: u32,
    ) -> io::Result<Entry> {
        let components = self.child_components(parent, name)?;
        let (mount, comps) = self.route_writable(&components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        self.block(async {
            mount
                .as_fsmount()
                .ok_or(erofs())?
                .mkdir(&comp_refs, mode & 0o7777, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        })?;
        let (st, _) = self.stat_path(&components)?;
        let inode = self.inodes.intern(components, Kind::Dir);
        Ok(self.make_entry(inode, Kind::Dir, &st))
    }

    fn rmdir(&self, _ctx: &Context, parent: Self::Inode, name: &CStr) -> io::Result<()> {
        let components = self.child_components(parent, name)?;
        let (mount, comps) = self.route_writable(&components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let r = self.block(async {
            mount
                .as_fsmount()
                .ok_or(erofs())?
                .rmdir(&comp_refs, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        });
        if r.is_ok() {
            self.inodes.invalidate_path(&components);
        }
        r
    }

    fn rename(
        &self,
        _ctx: &Context,
        olddir: Self::Inode,
        oldname: &CStr,
        newdir: Self::Inode,
        newname: &CStr,
        _flags: u32,
    ) -> io::Result<()> {
        let from = self.child_components(olddir, oldname)?;
        let to = self.child_components(newdir, newname)?;
        // Rename across distinct mounts is not supported (no cross-mount copy);
        // require both endpoints to resolve to the same writable mount.
        let (from_targets, from_comps) = self.route(&from)?;
        let (to_targets, to_comps) = self.route(&to)?;
        let from_fs = from_targets.iter().find(|m| m.as_fsmount().is_some());
        let to_fs = to_targets.iter().find(|m| m.as_fsmount().is_some());
        let (Some(from_fs), Some(to_fs)) = (from_fs, to_fs) else {
            return Err(erofs());
        };
        if !std::sync::Arc::ptr_eq(from_fs, to_fs) {
            return Err(io::Error::from_raw_os_error(libc::EXDEV));
        }
        let from_refs: Vec<&str> = from_comps.iter().map(String::as_str).collect();
        let to_refs: Vec<&str> = to_comps.iter().map(String::as_str).collect();
        let res = self.block(async {
            let fs = from_fs.as_fsmount().ok_or(erofs())?;
            fs.rename(&from_refs, &to_refs, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        });
        if res.is_ok() {
            self.inodes.invalidate_path(&from);
            self.inodes.invalidate_path(&to);
        }
        res
    }

    fn setattr(
        &self,
        _ctx: &Context,
        inode: Self::Inode,
        attr: libc::stat64,
        _handle: Option<Self::Handle>,
        valid: SetattrValid,
    ) -> io::Result<(libc::stat64, Duration)> {
        let data = self.inode_data(inode)?;
        let set = SetAttr {
            mode: valid.contains(SetattrValid::MODE).then_some(attr.st_mode & 0o7777),
            uid: valid.contains(SetattrValid::UID).then_some(attr.st_uid),
            gid: valid.contains(SetattrValid::GID).then_some(attr.st_gid),
            size: valid.contains(SetattrValid::SIZE).then_some(attr.st_size as u64),
            atime: valid.contains(SetattrValid::ATIME).then_some(attr.st_atime as u64),
            mtime: valid.contains(SetattrValid::MTIME).then_some(attr.st_mtime as u64),
        };
        let (mount, comps) = self.route_writable(&data.components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        self.block(async {
            mount
                .as_fsmount()
                .ok_or(erofs())?
                .setattr(&comp_refs, &set, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        })?;
        let (st, is_dir) = self.stat_path(&data.components)?;
        let kind = if is_dir { Kind::Dir } else { data.kind };
        Ok((Self::fill_attr(inode, &st, kind), TTL))
    }

    fn symlink(
        &self,
        _ctx: &Context,
        linkname: &CStr,
        parent: Self::Inode,
        name: &CStr,
    ) -> io::Result<Entry> {
        let components = self.child_components(parent, name)?;
        let target = linkname.to_str().map_err(|_| einval())?.to_owned();
        let (mount, comps) = self.route_writable(&components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        self.block(async {
            mount
                .as_fsmount()
                .ok_or(erofs())?
                .symlink(&comp_refs, &target, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        })?;
        let (st, _) = self.stat_path(&components)?;
        let inode = self.inodes.intern(components, Kind::Symlink);
        Ok(self.make_entry(inode, Kind::Symlink, &st))
    }

    fn readlink(&self, _ctx: &Context, inode: Self::Inode) -> io::Result<Vec<u8>> {
        let data = self.inode_data(inode)?;
        let (mount, comps) = self.route_writable(&data.components)?;
        let comp_refs: Vec<&str> = comps.iter().map(String::as_str).collect();
        let target = self.block(async {
            mount
                .as_fsmount()
                .ok_or(erofs())?
                .readlink(&comp_refs, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        })?;
        Ok(target.into_bytes())
    }

    fn link(
        &self,
        _ctx: &Context,
        inode: Self::Inode,
        newparent: Self::Inode,
        newname: &CStr,
    ) -> io::Result<Entry> {
        let existing = self.inode_data(inode)?.components;
        let new_path = self.child_components(newparent, newname)?;
        // Hard links must stay within one writable mount.
        let (ex_targets, ex_comps) = self.route(&existing)?;
        let (new_targets, new_comps) = self.route(&new_path)?;
        let ex_fs = ex_targets.iter().find(|m| m.as_fsmount().is_some());
        let new_fs = new_targets.iter().find(|m| m.as_fsmount().is_some());
        let (Some(ex_fs), Some(new_fs)) = (ex_fs, new_fs) else {
            return Err(erofs());
        };
        if !std::sync::Arc::ptr_eq(ex_fs, new_fs) {
            return Err(io::Error::from_raw_os_error(libc::EXDEV));
        }
        let ex_refs: Vec<&str> = ex_comps.iter().map(String::as_str).collect();
        let new_refs: Vec<&str> = new_comps.iter().map(String::as_str).collect();
        self.block(async {
            let fs = ex_fs.as_fsmount().ok_or(erofs())?;
            fs.link(&ex_refs, &new_refs, &self.subject)
                .await
                .map_err(|e| io_err(&e))
        })?;
        let (st, _) = self.stat_path(&new_path)?;
        let new_inode = self.inodes.intern(new_path, Kind::File);
        Ok(self.make_entry(new_inode, Kind::File, &st))
    }

    fn access(&self, _ctx: &Context, _inode: Self::Inode, _mask: u32) -> io::Result<()> {
        // Access control is enforced at the Subject/policy boundary, not via the
        // guest's POSIX mode bits; grant and let the mount op fail-close.
        Ok(())
    }

    fn getxattr(
        &self,
        _ctx: &Context,
        _inode: Self::Inode,
        _name: &CStr,
        _size: u32,
    ) -> io::Result<GetxattrReply> {
        Err(io::Error::from_raw_os_error(libc::ENODATA))
    }

    fn listxattr(&self, _ctx: &Context, _inode: Self::Inode, _size: u32) -> io::Result<ListxattrReply> {
        Ok(ListxattrReply::Names(Vec::new()))
    }
}

