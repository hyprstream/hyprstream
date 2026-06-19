//! `FileSystem` → `Mount`/`FsMount` **up-adapter** (native only).
//!
//! Anything that is already a [`fuse_backend_rs::api::filesystem::FileSystem`]
//! — RAFS (via `nydus-rafs`), `OverlayFs`, `PassthroughFs` — can be exposed to
//! the hyprstream VFS as an [`FsMount`] (and therefore a [`Mount`]) by this
//! wrapper. It is the "up" half of the bidirectional `FileSystem ↔ Mount`
//! bridge described in the converged FS/VFS design; the "down" half
//! (`Namespace → FileSystem`) is FS-A and is intentionally **not** built here.
//!
//! ## Path model
//!
//! `FileSystem` is inode/handle based and resolves one path component at a time
//! via `lookup`. The VFS, by contrast, hands a backend a *component slice*
//! rooted at the mount root. The adapter bridges the two by walking from the
//! filesystem root inode, calling `lookup` per component and `forget`-ing the
//! intermediates so the wrapped FS's lookup counts stay balanced.
//!
//! ## Subject threading
//!
//! Every op carries the verified [`Subject`]. There is no host uid/gid backing
//! a VFS caller, so the FUSE [`Context`] is constructed neutrally (root creds);
//! the `Subject` is carried alongside for the policy/audit boundary
//! (#353/#319/#328) and is where FS-A/FS-D will hook per-tenant enforcement.
//! Mutating ops are fail-closed: any error from the wrapped FS surfaces as a
//! [`MountError`].
//!
//! ## Send + Sync
//!
//! Natively this wrapper is *genuinely* `Send + Sync` (its `FileSystem` is, by
//! the trait bounds we require) — we do **not** use the wasm `unsafe impl Send`
//! escape hatch. The whole module is `cfg(not(target_arch = "wasm32"))`.

use std::ffi::CString;
use std::io;

use async_trait::async_trait;
use parking_lot::Mutex;
use fuse_backend_rs::abi::fuse_abi::CreateIn;
use fuse_backend_rs::api::filesystem::{Context, FileSystem, FsOptions, SetattrValid};

use hyprstream_rpc::Subject;

use crate::fsmount::{FsMount, SetAttr};
use crate::mount::{DirEntry, Fid, Mount, MountError, Stat, ORDWR, OTRUNC, OWRITE};

mod zerocopy;
use zerocopy::{MemReader, MemWriter};

/// FUSE root inode (`FUSE_ROOT_ID`).
const ROOT_INODE: u64 = 1;

/// Map a `std::io::Error` (the wrapped FS's error type) to a [`MountError`].
fn map_io(err: io::Error, ctx: &str) -> MountError {
    use io::ErrorKind::*;
    let msg = format!("{ctx}: {err}");
    match err.raw_os_error() {
        Some(libc::ENOENT) => MountError::NotFound(msg),
        Some(libc::EEXIST) => MountError::AlreadyExists(msg),
        Some(libc::EACCES) | Some(libc::EPERM) => MountError::PermissionDenied(msg),
        Some(libc::ENOTDIR) => MountError::NotDirectory(msg),
        Some(libc::EISDIR) => MountError::IsDirectory(msg),
        Some(libc::EINVAL) => MountError::InvalidArgument(msg),
        Some(libc::ENOSYS) | Some(libc::ENOTSUP) => MountError::NotSupported(msg),
        _ => match err.kind() {
            NotFound => MountError::NotFound(msg),
            AlreadyExists => MountError::AlreadyExists(msg),
            PermissionDenied => MountError::PermissionDenied(msg),
            InvalidInput => MountError::InvalidArgument(msg),
            Unsupported => MountError::NotSupported(msg),
            _ => MountError::Io(msg),
        },
    }
}

fn cstr(name: &str) -> Result<CString, MountError> {
    CString::new(name).map_err(|_| MountError::InvalidArgument(format!("interior NUL in '{name}'")))
}

/// A neutral FUSE context. VFS callers have no host uid/gid; the [`Subject`] is
/// the real identity and is threaded separately for policy/audit.
fn context() -> Context {
    Context::default()
}

/// Per-fid state held inside a [`Fid`]: the resolved inode, the open handle (if
/// the fid has been `open`ed), the open mode, and whether it is a directory.
struct FuseFid {
    inode: u64,
    handle: Option<u64>,
    mode: u8,
    is_dir: bool,
    /// Whether a remove-on-clunk (`ORCLOSE`) was requested. We need the parent
    /// inode + name to action it, captured at walk time.
    rclose: Option<(u64, String, bool)>,
}

/// Up-adapter: exposes a `fuse_backend_rs` [`FileSystem`] as an [`FsMount`].
///
/// `F::Inode` and `F::Handle` are required to be `u64` (which `OverlayFs`,
/// `PassthroughFs` and RAFS all use) so the adapter can store them in a generic
/// [`Fid`] without per-FS associated-type juggling.
pub struct FuseFileSystemMount<F>
where
    F: FileSystem<Inode = u64, Handle = u64> + Send + Sync,
{
    fs: F,
    /// Guards lookup-count bookkeeping. The wrapped `FileSystem` ops are `&self`
    /// and internally synchronised, but per-op walk/forget sequences are
    /// composite; the mutex keeps a mount's lookup accounting linearizable.
    lookup_guard: Mutex<()>,
}

impl<F> FuseFileSystemMount<F>
where
    F: FileSystem<Inode = u64, Handle = u64> + Send + Sync,
{
    /// Wrap `fs`, running its `init` so option negotiation (writeback, etc.)
    /// happens once. Use this for filesystems that are ready to serve (e.g. a
    /// `PassthroughFs` whose `import()` has been called, or an imported
    /// `OverlayFs`).
    pub fn new(fs: F) -> Result<Self, MountError> {
        fs.init(FsOptions::empty())
            .map_err(|e| map_io(e, "FileSystem::init"))?;
        Ok(Self {
            fs,
            lookup_guard: Mutex::new(()),
        })
    }

    /// Borrow the wrapped filesystem (e.g. to call backend-specific setup).
    pub fn inner(&self) -> &F {
        &self.fs
    }

    /// Resolve the parent directory inode plus the final component name.
    ///
    /// Empty `path` is the root itself and has no name — callers that need a
    /// final component (`create`, `unlink`, …) reject it.
    fn resolve_parent(&self, path: &[&str], ctx: &Context) -> Result<(u64, String), MountError> {
        let (last, parents) = path
            .split_last()
            .ok_or_else(|| MountError::InvalidArgument("operation requires a path".into()))?;
        let parent = self.resolve(parents, ctx)?;
        Ok((parent, (*last).to_owned()))
    }

    /// Walk `components` from the root inode, returning the resolved inode.
    ///
    /// Each `lookup` increments the wrapped FS's lookup count; we `forget` every
    /// intermediate so only the *returned* inode is left referenced. Callers that
    /// take ownership of the returned inode for a longer-lived fid must `forget`
    /// it on clunk; transient resolutions should pair with [`Self::forget`].
    fn resolve(&self, components: &[&str], ctx: &Context) -> Result<u64, MountError> {
        let mut inode = ROOT_INODE;
        for (i, comp) in components.iter().enumerate() {
            if comp.is_empty() || *comp == "." {
                continue;
            }
            let name = cstr(comp)?;
            let entry = self
                .fs
                .lookup(ctx, inode, &name)
                .map_err(|e| map_io(e, "lookup"))?;
            if entry.inode == 0 {
                // Negative entry — release any intermediates we hold.
                if inode != ROOT_INODE {
                    self.fs.forget(ctx, inode, 1);
                }
                return Err(MountError::NotFound(components[..=i].join("/")));
            }
            // Drop the parent's reference now that we've descended.
            if inode != ROOT_INODE {
                self.fs.forget(ctx, inode, 1);
            }
            inode = entry.inode;
        }
        Ok(inode)
    }

    /// Drop one lookup reference on `inode` (no-op for the root).
    fn forget(&self, inode: u64, ctx: &Context) {
        if inode != ROOT_INODE {
            self.fs.forget(ctx, inode, 1);
        }
    }

    /// Convert a FUSE [`Entry`] / `stat64` into the VFS [`Stat`] shape.
    fn entry_stat(name: &str, attr: &libc::stat64) -> Stat {
        let is_dir = attr.st_mode & libc::S_IFMT == libc::S_IFDIR;
        Stat {
            qtype: if is_dir { 0x80 } else { 0 },
            size: attr.st_size as u64,
            name: name.to_owned(),
            mtime: attr.st_mtime as u64,
        }
    }

    /// `getattr` for an inode.
    fn getattr(&self, inode: u64, ctx: &Context) -> Result<libc::stat64, MountError> {
        self.fs
            .getattr(ctx, inode, None)
            .map(|(st, _)| st)
            .map_err(|e| map_io(e, "getattr"))
    }
}

#[async_trait]
impl<F> Mount for FuseFileSystemMount<F>
where
    F: FileSystem<Inode = u64, Handle = u64> + Send + Sync,
{
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();

        // Capture the parent + name so ORCLOSE can action a remove on clunk.
        let rclose_target = match components.split_last() {
            Some((last, parents)) => {
                let parent = self.resolve(parents, &ctx)?;
                let name = cstr(last)?;
                let entry = self
                    .fs
                    .lookup(&ctx, parent, &name)
                    .map_err(|e| map_io(e, "walk lookup"))?;
                if entry.inode == 0 {
                    self.forget(parent, &ctx);
                    return Err(MountError::NotFound(components.join("/")));
                }
                let is_dir = entry.attr.st_mode & libc::S_IFMT == libc::S_IFDIR;
                // Hold the parent ref for the lifetime of the fid (ORCLOSE needs
                // it); the resolved inode ref is the fid's.
                Some((parent, last.to_string(), is_dir, entry.inode))
            }
            None => None, // root
        };

        match rclose_target {
            Some((parent, name, is_dir, inode)) => Ok(Fid::new(FuseFid {
                inode,
                handle: None,
                mode: 0,
                is_dir,
                rclose: Some((parent, name, is_dir)),
            })),
            None => {
                let attr = self.getattr(ROOT_INODE, &ctx)?;
                let is_dir = attr.st_mode & libc::S_IFMT == libc::S_IFDIR;
                Ok(Fid::new(FuseFid {
                    inode: ROOT_INODE,
                    handle: None,
                    mode: 0,
                    is_dir,
                    rclose: None,
                }))
            }
        }
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let inner = fid
            .downcast_mut::<FuseFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        inner.mode = mode;

        // Truncate-on-open is a setattr(size=0) on the writable layer.
        if mode & OTRUNC != 0 && !inner.is_dir {
            let mut attr: libc::stat64 = unsafe { std::mem::zeroed() };
            attr.st_size = 0;
            self.fs
                .setattr(&ctx, inner.inode, attr, None, SetattrValid::SIZE)
                .map_err(|e| map_io(e, "open OTRUNC"))?;
        }

        if inner.is_dir {
            let (handle, _) = self
                .fs
                .opendir(&ctx, inner.inode, libc::O_RDONLY as u32)
                .map_err(|e| map_io(e, "opendir"))?;
            inner.handle = handle;
        } else {
            let flags = match mode & 0x03 {
                OWRITE => libc::O_WRONLY,
                ORDWR => libc::O_RDWR,
                _ => libc::O_RDONLY,
            } as u32;
            let (handle, _, _) = self
                .fs
                .open(&ctx, inner.inode, flags, 0)
                .map_err(|e| map_io(e, "open"))?;
            inner.handle = handle;
        }
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let ctx = context();
        let inner = fid
            .downcast_ref::<FuseFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let handle = inner.handle.unwrap_or(0);
        let mut writer = MemWriter::with_capacity(count as usize);
        self.fs
            .read(&ctx, inner.inode, handle, &mut writer, count, offset, None, 0)
            .map_err(|e| map_io(e, "read"))?;
        Ok(writer.into_inner())
    }

    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let ctx = context();
        let inner = fid
            .downcast_ref::<FuseFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let handle = inner.handle.unwrap_or(0);
        let mut reader = MemReader::new(data);
        let n = self
            .fs
            .write(&ctx, inner.inode, handle, &mut reader, data.len() as u32, offset, None, false, 0, 0)
            .map_err(|e| map_io(e, "write"))?;
        Ok(n as u32)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let ctx = context();
        let inner = fid
            .downcast_ref::<FuseFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let handle = inner.handle.unwrap_or(0);

        let mut names: Vec<(String, bool)> = Vec::new();
        // Page through the directory until a pass yields nothing new.
        let mut offset = 0u64;
        loop {
            let before = names.len();
            self.fs
                .readdir(&ctx, inner.inode, handle, 1 << 16, offset, &mut |e| {
                    offset = e.offset;
                    let name = String::from_utf8_lossy(e.name).into_owned();
                    if name != "." && name != ".." {
                        let is_dir = e.type_ == libc::DT_DIR as u32;
                        names.push((name, is_dir));
                    }
                    Ok(1)
                })
                .map_err(|e| map_io(e, "readdir"))?;
            if names.len() == before {
                break;
            }
        }

        // Resolve sizes/attrs per entry via lookup (best-effort; skip on error).
        let mut entries = Vec::with_capacity(names.len());
        for (name, is_dir) in names {
            let c = match cstr(&name) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let (size, stat) = match self.fs.lookup(&ctx, inner.inode, &c) {
                Ok(entry) if entry.inode != 0 => {
                    let st = Self::entry_stat(&name, &entry.attr);
                    let size = entry.attr.st_size as u64;
                    self.forget(entry.inode, &ctx);
                    (size, Some(st))
                }
                _ => (0, None),
            };
            entries.push(DirEntry { name, is_dir, size, stat });
        }
        Ok(entries)
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let ctx = context();
        let inner = fid
            .downcast_ref::<FuseFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let attr = self.getattr(inner.inode, &ctx)?;
        Ok(Self::entry_stat("", &attr))
    }

    async fn clunk(&self, fid: Fid, _caller: &Subject) {
        let ctx = context();
        let Some(inner) = fid.downcast_ref::<FuseFid>() else {
            return;
        };
        // Release the open handle, if any.
        if let Some(handle) = inner.handle {
            if inner.is_dir {
                let _ = self.fs.releasedir(&ctx, inner.inode, libc::O_RDONLY as u32, handle);
            } else {
                let _ = self.fs.release(&ctx, inner.inode, 0, handle, true, false, None);
            }
        }
        // Action ORCLOSE (remove-on-clunk).
        if inner.mode & crate::mount::ORCLOSE != 0 {
            if let Some((parent, name, is_dir)) = &inner.rclose {
                if let Ok(c) = cstr(name) {
                    let _ = if *is_dir {
                        self.fs.rmdir(&ctx, *parent, &c)
                    } else {
                        self.fs.unlink(&ctx, *parent, &c)
                    };
                }
            }
        }
        // Balance lookup counts: the fid's own inode, and the held parent (if any).
        self.forget(inner.inode, &ctx);
        if let Some((parent, _, _)) = &inner.rclose {
            self.forget(*parent, &ctx);
        }
    }
}

#[async_trait]
impl<F> FsMount for FuseFileSystemMount<F>
where
    F: FileSystem<Inode = u64, Handle = u64> + Send + Sync,
{
    async fn create(&self, path: &[&str], mode: u32, caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let (parent, name) = self.resolve_parent(path, &ctx)?;
        let cname = cstr(&name)?;
        let args = CreateIn {
            flags: (libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY) as u32,
            mode,
            umask: 0,
            fuse_flags: 0,
        };
        let res = self.fs.create(&ctx, parent, &cname, args);
        self.forget(parent, &ctx);
        let (entry, handle, _, _) = res.map_err(|e| map_io(e, "create"))?;
        // We created and opened the file; close the handle immediately — the VFS
        // create op only guarantees existence, callers re-open to write.
        if let Some(h) = handle {
            let _ = self.fs.release(&ctx, entry.inode, args.flags, h, true, false, None);
        }
        self.forget(entry.inode, &ctx);
        let _ = caller; // carried for policy boundary; no host cred mapping yet
        Ok(())
    }

    async fn unlink(&self, path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let (parent, name) = self.resolve_parent(path, &ctx)?;
        let cname = cstr(&name)?;
        let res = self.fs.unlink(&ctx, parent, &cname);
        self.forget(parent, &ctx);
        res.map_err(|e| map_io(e, "unlink"))
    }

    async fn mkdir(&self, path: &[&str], mode: u32, _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let (parent, name) = self.resolve_parent(path, &ctx)?;
        let cname = cstr(&name)?;
        let res = self.fs.mkdir(&ctx, parent, &cname, mode, 0);
        self.forget(parent, &ctx);
        let entry = res.map_err(|e| map_io(e, "mkdir"))?;
        self.forget(entry.inode, &ctx);
        Ok(())
    }

    async fn rmdir(&self, path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let (parent, name) = self.resolve_parent(path, &ctx)?;
        let cname = cstr(&name)?;
        let res = self.fs.rmdir(&ctx, parent, &cname);
        self.forget(parent, &ctx);
        res.map_err(|e| map_io(e, "rmdir"))
    }

    async fn rename(&self, from: &[&str], to: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let (old_dir, old_name) = self.resolve_parent(from, &ctx)?;
        let (new_dir, new_name) = match self.resolve_parent(to, &ctx) {
            Ok(v) => v,
            Err(e) => {
                self.forget(old_dir, &ctx);
                return Err(e);
            }
        };
        let old_c = cstr(&old_name)?;
        let new_c = cstr(&new_name)?;
        let res = self.fs.rename(&ctx, old_dir, &old_c, new_dir, &new_c, 0);
        self.forget(old_dir, &ctx);
        self.forget(new_dir, &ctx);
        res.map_err(|e| map_io(e, "rename"))
    }

    async fn setattr(&self, path: &[&str], attr: &SetAttr, _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        if attr.is_empty() {
            return Ok(());
        }
        let _g = self.lookup_guard.lock();
        let inode = self.resolve(path, &ctx)?;
        let mut st: libc::stat64 = unsafe { std::mem::zeroed() };
        let mut valid = SetattrValid::empty();
        if let Some(mode) = attr.mode {
            st.st_mode = mode;
            valid |= SetattrValid::MODE;
        }
        if let Some(uid) = attr.uid {
            st.st_uid = uid;
            valid |= SetattrValid::UID;
        }
        if let Some(gid) = attr.gid {
            st.st_gid = gid;
            valid |= SetattrValid::GID;
        }
        if let Some(size) = attr.size {
            st.st_size = size as i64;
            valid |= SetattrValid::SIZE;
        }
        if let Some(atime) = attr.atime {
            st.st_atime = atime as i64;
            valid |= SetattrValid::ATIME;
        }
        if let Some(mtime) = attr.mtime {
            st.st_mtime = mtime as i64;
            valid |= SetattrValid::MTIME;
        }
        let res = self.fs.setattr(&ctx, inode, st, None, valid);
        self.forget(inode, &ctx);
        res.map(|_| ()).map_err(|e| map_io(e, "setattr"))
    }

    async fn symlink(&self, path: &[&str], target: &str, _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let (parent, name) = self.resolve_parent(path, &ctx)?;
        let cname = cstr(&name)?;
        let ctarget = cstr(target)?;
        let res = self.fs.symlink(&ctx, &ctarget, parent, &cname);
        self.forget(parent, &ctx);
        let entry = res.map_err(|e| map_io(e, "symlink"))?;
        self.forget(entry.inode, &ctx);
        Ok(())
    }

    async fn readlink(&self, path: &[&str], _caller: &Subject) -> Result<String, MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let inode = self.resolve(path, &ctx)?;
        let res = self.fs.readlink(&ctx, inode);
        self.forget(inode, &ctx);
        let bytes = res.map_err(|e| map_io(e, "readlink"))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn link(&self, existing: &[&str], new_path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let inode = self.resolve(existing, &ctx)?;
        let (new_parent, new_name) = match self.resolve_parent(new_path, &ctx) {
            Ok(v) => v,
            Err(e) => {
                self.forget(inode, &ctx);
                return Err(e);
            }
        };
        let cname = cstr(&new_name)?;
        let res = self.fs.link(&ctx, inode, new_parent, &cname);
        self.forget(inode, &ctx);
        self.forget(new_parent, &ctx);
        let entry = res.map_err(|e| map_io(e, "link"))?;
        self.forget(entry.inode, &ctx);
        Ok(())
    }

    async fn stat_path(&self, path: &[&str], _caller: &Subject) -> Result<Stat, MountError> {
        let ctx = context();
        let _g = self.lookup_guard.lock();
        let inode = self.resolve(path, &ctx)?;
        let attr = self.getattr(inode, &ctx);
        self.forget(inode, &ctx);
        let name = path.last().copied().unwrap_or("");
        Ok(Self::entry_stat(name, &attr?))
    }
}
