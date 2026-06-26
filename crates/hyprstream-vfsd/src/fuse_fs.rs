//! FUSE FileSystem implementation backed by the VFS Mount trait.
//!
//! Translates between the fuse-backend-rs `FileSystem` trait (Linux FUSE protocol)
//! and the hyprstream-vfs `Mount` trait (Plan9-inspired VFS).

use std::ffi::CStr;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fuse_backend_rs::abi::fuse_abi::{stat64, FsOptions};
use fuse_backend_rs::abi::virtio_fs::RemovemappingOne;
use fuse_backend_rs::api::filesystem::{
    Context, DirEntry as FuseDirEntry, Entry, FileSystem, ZeroCopyReader, ZeroCopyWriter,
};
use fuse_backend_rs::transport::FsCacheReqHandler;
use parking_lot::RwLock;

use hyprstream_rpc::Subject;
use hyprstream_vfs::{Fid, Mount, OREAD, OWRITE};

use crate::dax::{DaxMount, ErasedDax};
use crate::inode_table::{InodeTable, ROOT_INODE};

/// Default TTL for directory/attribute caches (1 second).
const DEFAULT_TTL: Duration = Duration::from_secs(1);

/// `SetupmappingFlags::WRITE` bit from the virtio-fs ABI (fuse.h
/// `FUSE_SETUPMAPPING_FLAG_WRITE`). fuse-backend-rs does not re-export the
/// flags type publicly, so the bit is mirrored here.
const SETUPMAPPING_FLAG_WRITE: u64 = 0x1;

/// Per-handle state tracking an open VFS fid.
struct HandleData {
    fid: RwLock<Fid>,
    #[allow(dead_code)] // Retained for diagnostics.
    inode: u64,
    /// Whether the handle was opened for writing — gates writable DAX mappings.
    write: bool,
}

/// FUSE filesystem adapter that bridges a VFS `Mount` to the FUSE protocol.
///
/// Maps FUSE inodes to VFS paths via an [`InodeTable`], and translates
/// FUSE operations (lookup, getattr, read, write, readdir) to Mount trait calls.
///
/// All operations use an anonymous `Subject` since FUSE context (uid/gid)
/// does not map directly to hyprstream's identity model. The virtio-fs
/// daemon should enforce access control at a higher layer.
pub struct VfsFuse<M: Mount> {
    mount: Arc<M>,
    /// Optional DAX capability, present iff the adapter was built via
    /// [`VfsFuse::with_dax`] from a mount implementing [`DaxMount`]. Points at
    /// the same instance as `mount`. `None` → DAX mapping requests return
    /// `ENOSYS` and the guest falls back to the FUSE read path.
    dax: Option<Arc<dyn ErasedDax>>,
    inodes: InodeTable,
    /// Monotonic handle allocator.
    next_handle: AtomicU64,
    /// Open handle table: handle_id -> HandleData.
    handles: dashmap::DashMap<u64, HandleData>,
    /// Tokio runtime handle for blocking on async Mount calls from sync FUSE callbacks.
    rt: tokio::runtime::Handle,
}

impl<M: Mount> VfsFuse<M> {
    /// Create a new FUSE adapter wrapping the given VFS mount, without DAX.
    ///
    /// Requires a tokio runtime handle because FUSE callbacks are synchronous
    /// but the Mount trait is async. Each callback uses `block_on()`.
    pub fn new(mount: M, rt: tokio::runtime::Handle) -> Self {
        Self {
            mount: Arc::new(mount),
            dax: None,
            inodes: InodeTable::new(),
            next_handle: AtomicU64::new(1),
            handles: dashmap::DashMap::new(),
            rt,
        }
    }
}

impl<M: DaxMount + 'static> VfsFuse<M> {
    /// Create a FUSE adapter with DAX (direct-access) support enabled.
    ///
    /// Available only when `M: DaxMount`. The guest can then request
    /// `setupmapping`/`removemapping` to map DAX-capable files directly into
    /// its DAX window; files for which [`DaxMount::backing_fd`] returns `None`
    /// transparently fall back to the FUSE read path.
    pub fn with_dax(mount: M, rt: tokio::runtime::Handle) -> Self {
        let mount = Arc::new(mount);
        let dax: Arc<dyn ErasedDax> = mount.clone();
        Self {
            mount,
            dax: Some(dax),
            inodes: InodeTable::new(),
            next_handle: AtomicU64::new(1),
            handles: dashmap::DashMap::new(),
            rt,
        }
    }
}

impl<M: Mount> VfsFuse<M> {
    /// Get the anonymous caller subject used for all VFS operations.
    fn caller() -> Subject {
        Subject::anonymous()
    }

    /// Build a stat64 for a directory inode.
    fn dir_stat(ino: u64) -> stat64 {
        let mut st: stat64 = unsafe { std::mem::zeroed() };
        st.st_ino = ino;
        st.st_mode = libc::S_IFDIR | 0o555;
        st.st_nlink = 2;
        st
    }

    /// Build a stat64 for a file inode with the given size.
    fn file_stat(ino: u64, size: u64) -> stat64 {
        let mut st: stat64 = unsafe { std::mem::zeroed() };
        st.st_ino = ino;
        st.st_mode = libc::S_IFREG | 0o444;
        st.st_nlink = 1;
        st.st_size = size as i64;
        st
    }

    /// Walk path components and stat the result to determine if dir or file.
    fn walk_and_stat(
        &self,
        components: &[String],
    ) -> io::Result<(bool, u64)> {
        let refs: Vec<&str> = components.iter().map(String::as_str).collect();
        let caller = Self::caller();
        let fid = self.rt
            .block_on(self.mount.walk(&refs, &caller))
            .map_err(mount_err_to_io)?;
        let stat = self.rt
            .block_on(self.mount.stat(&fid, &caller))
            .map_err(mount_err_to_io)?;
        let is_dir = stat.qtype & 0x80 != 0; // QTDIR
        let size = stat.size;
        self.rt.block_on(self.mount.clunk(fid, &caller));
        Ok((is_dir, size))
    }
}

impl<M: Mount + 'static> FileSystem for VfsFuse<M> {
    type Inode = u64;
    type Handle = u64;

    fn init(&self, _capable: FsOptions) -> io::Result<FsOptions> {
        Ok(FsOptions::empty())
    }

    fn destroy(&self) {}

    fn lookup(&self, _ctx: &Context, parent: u64, name: &CStr) -> io::Result<Entry> {
        let name_str = name
            .to_str()
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;

        // Build the full path from parent inode + child name.
        let parent_path = self
            .inodes
            .path_of(parent)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))?;

        let mut child_path = parent_path;
        child_path.push(name_str.to_owned());

        // Walk the VFS to verify existence and get metadata.
        let (is_dir, size) = self.walk_and_stat(&child_path)?;

        // Allocate or reuse an inode.
        let ino = self.inodes.lookup_or_insert(child_path, is_dir);

        let attr = if is_dir {
            Self::dir_stat(ino)
        } else {
            Self::file_stat(ino, size)
        };

        Ok(Entry {
            inode: ino,
            generation: 0,
            attr,
            attr_flags: 0,
            attr_timeout: DEFAULT_TTL,
            entry_timeout: DEFAULT_TTL,
        })
    }

    fn forget(&self, _ctx: &Context, inode: u64, count: u64) {
        self.inodes.forget(inode, count);
    }

    fn getattr(
        &self,
        _ctx: &Context,
        inode: u64,
        _handle: Option<u64>,
    ) -> io::Result<(stat64, Duration)> {
        if inode == ROOT_INODE {
            return Ok((Self::dir_stat(ROOT_INODE), DEFAULT_TTL));
        }

        let path = self
            .inodes
            .path_of(inode)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))?;

        let is_dir = self
            .inodes
            .get(inode)
            .map(|d| d.is_dir)
            .unwrap_or(false);

        let attr = if is_dir {
            Self::dir_stat(inode)
        } else {
            // Walk to get current size.
            let refs: Vec<&str> = path.iter().map(String::as_str).collect();
            let caller = Self::caller();
            let fid = self.rt
                .block_on(self.mount.walk(&refs, &caller))
                .map_err(mount_err_to_io)?;
            let stat = self.rt
                .block_on(self.mount.stat(&fid, &caller))
                .map_err(mount_err_to_io)?;
            self.rt.block_on(self.mount.clunk(fid, &caller));
            Self::file_stat(inode, stat.size)
        };

        Ok((attr, DEFAULT_TTL))
    }

    fn open(
        &self,
        _ctx: &Context,
        inode: u64,
        flags: u32,
        _fuse_flags: u32,
    ) -> io::Result<(Option<u64>, fuse_backend_rs::api::filesystem::OpenOptions, Option<u32>)> {
        let path = self
            .inodes
            .path_of(inode)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))?;

        let refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let caller = Self::caller();

        let mut fid = self.rt
            .block_on(self.mount.walk(&refs, &caller))
            .map_err(mount_err_to_io)?;

        // Map POSIX open flags to 9P mode. O_WRONLY and O_RDWR both imply write.
        let acc = flags & libc::O_ACCMODE as u32;
        let write = acc == libc::O_WRONLY as u32 || acc == libc::O_RDWR as u32;
        let mode = if write { OWRITE } else { OREAD };

        self.rt
            .block_on(self.mount.open(&mut fid, mode, &caller))
            .map_err(mount_err_to_io)?;

        let handle_id = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.handles.insert(
            handle_id,
            HandleData {
                fid: RwLock::new(fid),
                inode,
                write,
            },
        );

        Ok((
            Some(handle_id),
            fuse_backend_rs::api::filesystem::OpenOptions::empty(),
            None,
        ))
    }

    fn read(
        &self,
        _ctx: &Context,
        _inode: u64,
        handle: u64,
        w: &mut dyn ZeroCopyWriter,
        size: u32,
        offset: u64,
        _lock_owner: Option<u64>,
        _flags: u32,
    ) -> io::Result<usize> {
        let entry = self
            .handles
            .get(&handle)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EBADF))?;

        let fid = entry.fid.read();
        let caller = Self::caller();
        let data = self.rt
            .block_on(self.mount.read(&fid, offset, size, &caller))
            .map_err(mount_err_to_io)?;

        let len = data.len();
        w.write_all(&data)?;
        Ok(len)
    }

    fn write(
        &self,
        _ctx: &Context,
        _inode: u64,
        handle: u64,
        r: &mut dyn ZeroCopyReader,
        size: u32,
        offset: u64,
        _lock_owner: Option<u64>,
        _delayed_write: bool,
        _flags: u32,
        _fuse_flags: u32,
    ) -> io::Result<usize> {
        let entry = self
            .handles
            .get(&handle)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EBADF))?;

        let mut buf = vec![0u8; size as usize];
        r.read_exact(&mut buf)?;

        let fid = entry.fid.read();
        let caller = Self::caller();
        let written = self.rt
            .block_on(self.mount.write(&fid, offset, &buf, &caller))
            .map_err(mount_err_to_io)?;

        Ok(written as usize)
    }

    fn release(
        &self,
        _ctx: &Context,
        _inode: u64,
        _flags: u32,
        handle: u64,
        _flush: bool,
        _flock_release: bool,
        _lock_owner: Option<u64>,
    ) -> io::Result<()> {
        if let Some((_, data)) = self.handles.remove(&handle) {
            let caller = Self::caller();
            let fid = data.fid.into_inner();
            self.rt.block_on(self.mount.clunk(fid, &caller));
        }
        Ok(())
    }

    fn opendir(
        &self,
        _ctx: &Context,
        inode: u64,
        _flags: u32,
    ) -> io::Result<(Option<u64>, fuse_backend_rs::api::filesystem::OpenOptions)> {
        let path = self
            .inodes
            .path_of(inode)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))?;

        let refs: Vec<&str> = path.iter().map(String::as_str).collect();
        let caller = Self::caller();

        let mut fid = self.rt
            .block_on(self.mount.walk(&refs, &caller))
            .map_err(mount_err_to_io)?;

        self.rt
            .block_on(self.mount.open(&mut fid, OREAD, &caller))
            .map_err(mount_err_to_io)?;

        let handle_id = self.next_handle.fetch_add(1, Ordering::Relaxed);
        self.handles.insert(
            handle_id,
            HandleData {
                fid: RwLock::new(fid),
                inode,
                write: false,
            },
        );

        Ok((
            Some(handle_id),
            fuse_backend_rs::api::filesystem::OpenOptions::empty(),
        ))
    }

    fn readdir(
        &self,
        _ctx: &Context,
        inode: u64,
        handle: u64,
        _size: u32,
        offset: u64,
        add_entry: &mut dyn FnMut(FuseDirEntry<'_>) -> io::Result<usize>,
    ) -> io::Result<()> {
        let entry = self
            .handles
            .get(&handle)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EBADF))?;

        let fid = entry.fid.read();
        let caller = Self::caller();
        let entries = self.rt
            .block_on(self.mount.readdir(&fid, &caller))
            .map_err(mount_err_to_io)?;

        // Get parent path for building child paths.
        let parent_path = self
            .inodes
            .path_of(inode)
            .unwrap_or_default();

        for (i, vfs_entry) in entries.iter().enumerate() {
            let idx = i as u64 + 1; // 1-based offset
            if idx <= offset {
                continue;
            }

            // Build child path and allocate inode.
            let mut child_path = parent_path.clone();
            child_path.push(vfs_entry.name.clone());
            let child_ino = self
                .inodes
                .lookup_or_insert(child_path, vfs_entry.is_dir);

            let dtype = if vfs_entry.is_dir {
                libc::DT_DIR as u32
            } else {
                libc::DT_REG as u32
            };

            let fuse_entry = FuseDirEntry {
                ino: child_ino,
                offset: idx,
                type_: dtype,
                name: vfs_entry.name.as_bytes(),
            };

            match add_entry(fuse_entry) {
                Ok(0) => break, // Buffer full.
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    fn releasedir(
        &self,
        _ctx: &Context,
        _inode: u64,
        _flags: u32,
        handle: u64,
    ) -> io::Result<()> {
        if let Some((_, data)) = self.handles.remove(&handle) {
            let caller = Self::caller();
            let fid = data.fid.into_inner();
            self.rt.block_on(self.mount.clunk(fid, &caller));
        }
        Ok(())
    }

    /// DAX setup: map a region of the backing file directly into the guest's
    /// DAX window so the guest can `mmap` it without a FUSE read round-trip.
    ///
    /// Returns `ENOSYS` (so the guest falls back to the FUSE read path) when
    /// the adapter has no DAX capability, or when the open fid has no
    /// directly-mappable backing file (synthetic/ctl/dynamic content).
    /// Rejects a writable mapping against a read-only handle with `EACCES`.
    fn setupmapping(
        &self,
        _ctx: &Context,
        _inode: u64,
        handle: u64,
        foffset: u64,
        len: u64,
        flags: u64,
        moffset: u64,
        vu_req: &mut dyn FsCacheReqHandler,
    ) -> io::Result<()> {
        // No DAX capability → fall back to the FUSE read path.
        let dax = self
            .dax
            .as_ref()
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOSYS))?;

        let entry = self
            .handles
            .get(&handle)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EBADF))?;

        // Defense-in-depth: never grant a writable host mapping through a
        // read-only handle. (The backing fd's open mode is the primary guard.)
        if flags & SETUPMAPPING_FLAG_WRITE != 0 && !entry.write {
            return Err(io::Error::from_raw_os_error(libc::EACCES));
        }

        let fid = entry.fid.read();
        let fd = match dax.backing_fd(&fid) {
            Some(fd) => fd,
            // Not directly mappable → let the guest use the FUSE read path.
            None => return Err(io::Error::from_raw_os_error(libc::ENOSYS)),
        };

        // The VMM duplicates the fd over the vhost-user socket (SCM_RIGHTS)
        // during `map`, so the local `OwnedFd` may be dropped (closed) once
        // `map` returns — matching virtiofsd's fd lifetime.
        vu_req.map(foffset, moffset, len, flags, fd.as_raw_fd())
    }

    /// DAX teardown: remove previously-established DAX window mappings.
    fn removemapping(
        &self,
        _ctx: &Context,
        _inode: u64,
        requests: Vec<RemovemappingOne>,
        vu_req: &mut dyn FsCacheReqHandler,
    ) -> io::Result<()> {
        vu_req.unmap(requests)
    }
}

/// Convert a VFS MountError to a std::io::Error with appropriate errno.
fn mount_err_to_io(e: hyprstream_vfs::MountError) -> io::Error {
    use hyprstream_vfs::MountError;
    let errno = match &e {
        MountError::NotFound(_) => libc::ENOENT,
        MountError::PermissionDenied(_) => libc::EACCES,
        MountError::NotDirectory(_) => libc::ENOTDIR,
        MountError::IsDirectory(_) => libc::EISDIR,
        MountError::InvalidArgument(_) => libc::EINVAL,
        MountError::NotSupported(_) => libc::ENOSYS,
        MountError::AlreadyExists(_) => libc::EEXIST,
        MountError::Io(_) => libc::EIO,
    };
    io::Error::from_raw_os_error(errno)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::os::fd::OwnedFd;

    use async_trait::async_trait;
    use fuse_backend_rs::api::filesystem::Context;
    use hyprstream_vfs::{DirEntry, MountError, Stat};

    use crate::dax::DaxMount;
    use crate::inode_table::ROOT_INODE;

    /// Stub mount: every file is a 1 KiB regular file; `dax` controls whether
    /// it advertises a backing fd.
    struct StubMount {
        dax: bool,
    }

    #[async_trait]
    impl Mount for StubMount {
        async fn walk(&self, _c: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(()))
        }
        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        async fn read(&self, _fid: &Fid, _off: u64, _n: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            Ok(vec![0u8; 16])
        }
        async fn write(&self, _fid: &Fid, _off: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }
        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![])
        }
        async fn stat(&self, _fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            // qtype 0 == regular file (QTDIR bit 0x80 clear).
            Ok(Stat { qtype: 0, size: 1024, name: "weights".into(), mtime: 0, path: 0, version: 0 })
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    impl DaxMount for StubMount {
        fn backing_fd(&self, _fid: &Fid) -> Option<OwnedFd> {
            if !self.dax {
                return None;
            }
            // A real, mappable fd. /dev/null is always present and read-only here.
            std::fs::File::open("/dev/null").ok().map(OwnedFd::from)
        }
    }

    /// Records the DAX window operations the adapter forwards.
    #[derive(Default)]
    struct MockReq {
        mapped: u32,
        unmapped: u32,
    }
    impl FsCacheReqHandler for MockReq {
        fn map(&mut self, _fo: u64, _mo: u64, _len: u64, _flags: u64, _fd: std::os::fd::RawFd) -> io::Result<()> {
            self.mapped += 1;
            Ok(())
        }
        fn unmap(&mut self, requests: Vec<RemovemappingOne>) -> io::Result<()> {
            self.unmapped += requests.len() as u32;
            Ok(())
        }
    }

    /// Open a read- or write-mode handle on a freshly-looked-up file inode.
    fn open_handle<M: Mount + 'static>(fs: &VfsFuse<M>, write: bool) -> u64 {
        let ctx = Context::default();
        let name = CString::new("weights").unwrap();
        fs.lookup(&ctx, ROOT_INODE, &name).unwrap();
        let ino = fs.inodes.lookup_or_insert(vec!["weights".into()], false);
        let flags = if write { libc::O_WRONLY as u32 } else { libc::O_RDONLY as u32 };
        let (handle, _, _) = fs.open(&ctx, ino, flags, 0).unwrap();
        handle.unwrap()
    }

    #[test]
    fn non_dax_adapter_reports_enosys() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let fs = VfsFuse::new(StubMount { dax: false }, rt.handle().clone());
        let handle = open_handle(&fs, false);
        let mut req = MockReq::default();
        let err = fs
            .setupmapping(&Context::default(), 0, handle, 0, 4096, 0, 0, &mut req)
            .unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::ENOSYS));
        assert_eq!(req.mapped, 0);
    }

    #[test]
    fn dax_setupmapping_maps_then_removemapping_unmaps() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let fs = VfsFuse::with_dax(StubMount { dax: true }, rt.handle().clone());
        let handle = open_handle(&fs, false);
        let mut req = MockReq::default();

        fs.setupmapping(&Context::default(), 0, handle, 0, 4096, 0, 0, &mut req)
            .unwrap();
        assert_eq!(req.mapped, 1, "DAX-capable mount should program the window");

        let removes = vec![RemovemappingOne { moffset: 0, len: 4096 }];
        fs.removemapping(&Context::default(), 0, removes, &mut req)
            .unwrap();
        assert_eq!(req.unmapped, 1);
    }

    #[test]
    fn dax_falls_back_to_enosys_when_no_backing_fd() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        // DAX enabled on the adapter, but the mount returns no backing fd.
        let fs = VfsFuse::with_dax(StubMount { dax: false }, rt.handle().clone());
        let handle = open_handle(&fs, false);
        let mut req = MockReq::default();
        let err = fs
            .setupmapping(&Context::default(), 0, handle, 0, 4096, 0, 0, &mut req)
            .unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::ENOSYS));
        assert_eq!(req.mapped, 0);
    }

    #[test]
    fn writable_mapping_on_readonly_handle_is_denied() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let fs = VfsFuse::with_dax(StubMount { dax: true }, rt.handle().clone());
        let handle = open_handle(&fs, false); // read-only open
        let mut req = MockReq::default();
        let err = fs
            .setupmapping(
                &Context::default(),
                0,
                handle,
                0,
                4096,
                SETUPMAPPING_FLAG_WRITE,
                0,
                &mut req,
            )
            .unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::EACCES));
        assert_eq!(req.mapped, 0);
    }
}
