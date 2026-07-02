//! Unit tests for the `Namespace → FileSystem` down-adapter ([`VfsFileSystem`]).
//!
//! These exercise the adapter directly against a synthetic in-memory
//! [`FsMount`] — no VM, no vhost-user transport. (A full guest mount is live
//! validation, out of scope here.) They cover the core op surface the issue
//! calls out: lookup, read, write, create, readdir, rename, plus the
//! fail-closed `EROFS` behaviour over a read-only [`Mount`].

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::ffi::CString;
use std::io;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use fuse_backend_rs::abi::fuse_abi::CreateIn;
use fuse_backend_rs::api::filesystem::{
    Context, DirEntry as FuseDirEntry, FileSystem, ZeroCopyReader, ZeroCopyWriter,
};

use hyprstream_vfs::{
    DirEntry, Fid, FsMount, Mount, MountError, Namespace, SetAttr, Stat, Subject,
};

use crate::filesystem::VfsFileSystem;

// ── Synthetic in-memory writable filesystem ─────────────────────────────────

/// A trivial in-memory tree implementing both [`Mount`] and [`FsMount`]. Files
/// are byte vectors; directories are tracked as a set of paths ending in `/`.
#[derive(Default)]
struct MemFs {
    /// path (components joined by `/`) → contents. Directories map to `None`.
    nodes: Mutex<BTreeMap<String, Option<Vec<u8>>>>,
}

impl MemFs {
    fn new() -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert(String::new(), None); // root dir
        Self {
            nodes: Mutex::new(nodes),
        }
    }

    fn with_files(files: &[(&str, &[u8])]) -> Self {
        let fs = Self::new();
        {
            let mut nodes = fs.nodes.lock();
            for (path, data) in files {
                // Ensure parent dirs exist.
                let parts: Vec<&str> = path.split('/').collect();
                for i in 1..parts.len() {
                    let dir = parts[..i].join("/");
                    nodes.entry(dir).or_insert(None);
                }
                nodes.insert((*path).to_owned(), Some(data.to_vec()));
            }
        }
        fs
    }

    fn key(components: &[&str]) -> String {
        components.join("/")
    }
}

struct MemFid {
    path: String,
}

#[async_trait]
impl Mount for MemFs {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let key = Self::key(components);
        if self.nodes.lock().contains_key(&key) {
            Ok(Fid::new(MemFid { path: key }))
        } else {
            Err(MountError::NotFound(key))
        }
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        Ok(())
    }

    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let nodes = self.nodes.lock();
        match nodes.get(&inner.path) {
            Some(Some(data)) => {
                let start = offset as usize;
                if start >= data.len() {
                    Ok(vec![])
                } else {
                    let end = std::cmp::min(start + count as usize, data.len());
                    Ok(data[start..end].to_vec())
                }
            }
            _ => Err(MountError::NotFound(inner.path.clone())),
        }
    }

    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let mut nodes = self.nodes.lock();
        let entry = nodes.entry(inner.path.clone()).or_insert(Some(Vec::new()));
        let buf = entry.get_or_insert_with(Vec::new);
        let start = offset as usize;
        if buf.len() < start + data.len() {
            buf.resize(start + data.len(), 0);
        }
        buf[start..start + data.len()].copy_from_slice(data);
        Ok(data.len() as u32)
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let prefix = if inner.path.is_empty() { String::new() } else { format!("{}/", inner.path) };
        let nodes = self.nodes.lock();
        let mut entries = Vec::new();
        for (key, val) in nodes.iter() {
            if key.is_empty() {
                continue;
            }
            if let Some(rest) = key.strip_prefix(&prefix) {
                if !rest.is_empty() && !rest.contains('/') {
                    entries.push(DirEntry {
                        name: rest.to_owned(),
                        is_dir: val.is_none(),
                        size: val.as_ref().map(|d| d.len() as u64).unwrap_or(0),
                        stat: None,
                    });
                }
            }
        }
        Ok(entries)
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid.downcast_ref::<MemFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let nodes = self.nodes.lock();
        match nodes.get(&inner.path) {
            Some(val) => Ok(Stat {
                qtype: if val.is_none() { 0x80 } else { 0 },
                size: val.as_ref().map(|d| d.len() as u64).unwrap_or(0),
                name: inner.path.rsplit('/').next().unwrap_or("").to_owned(),
                mtime: 0,
            }),
            None => Err(MountError::NotFound(inner.path.clone())),
        }
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}

    fn as_fsmount(&self) -> Option<&dyn FsMount> {
        Some(self)
    }
}

#[async_trait]
impl FsMount for MemFs {
    async fn create(&self, path: &[&str], _mode: u32, _caller: &Subject) -> Result<(), MountError> {
        let key = Self::key(path);
        let mut nodes = self.nodes.lock();
        if nodes.contains_key(&key) {
            return Err(MountError::AlreadyExists(key));
        }
        nodes.insert(key, Some(Vec::new()));
        Ok(())
    }

    async fn unlink(&self, path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let key = Self::key(path);
        let mut nodes = self.nodes.lock();
        match nodes.get(&key) {
            Some(Some(_)) => {
                nodes.remove(&key);
                Ok(())
            }
            Some(None) => Err(MountError::IsDirectory(key)),
            None => Err(MountError::NotFound(key)),
        }
    }

    async fn mkdir(&self, path: &[&str], _mode: u32, _caller: &Subject) -> Result<(), MountError> {
        let key = Self::key(path);
        let mut nodes = self.nodes.lock();
        if nodes.contains_key(&key) {
            return Err(MountError::AlreadyExists(key));
        }
        nodes.insert(key, None);
        Ok(())
    }

    async fn rmdir(&self, path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let key = Self::key(path);
        let mut nodes = self.nodes.lock();
        match nodes.get(&key) {
            Some(None) => {
                nodes.remove(&key);
                Ok(())
            }
            Some(Some(_)) => Err(MountError::NotDirectory(key)),
            None => Err(MountError::NotFound(key)),
        }
    }

    async fn rename(&self, from: &[&str], to: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let from_key = Self::key(from);
        let to_key = Self::key(to);
        let mut nodes = self.nodes.lock();
        let val = nodes.remove(&from_key).ok_or(MountError::NotFound(from_key))?;
        nodes.insert(to_key, val);
        Ok(())
    }

    async fn setattr(&self, path: &[&str], attr: &SetAttr, _caller: &Subject) -> Result<(), MountError> {
        let key = Self::key(path);
        let mut nodes = self.nodes.lock();
        let entry = nodes.get_mut(&key).ok_or(MountError::NotFound(key))?;
        if let (Some(size), Some(data)) = (attr.size, entry.as_mut()) {
            data.resize(size as usize, 0);
        }
        Ok(())
    }

    async fn symlink(&self, path: &[&str], target: &str, _caller: &Subject) -> Result<(), MountError> {
        let key = Self::key(path);
        self.nodes.lock().insert(key, Some(target.as_bytes().to_vec()));
        Ok(())
    }

    async fn readlink(&self, path: &[&str], _caller: &Subject) -> Result<String, MountError> {
        let key = Self::key(path);
        match self.nodes.lock().get(&key) {
            Some(Some(data)) => Ok(String::from_utf8_lossy(data).into_owned()),
            _ => Err(MountError::NotFound(key)),
        }
    }

    async fn link(&self, existing: &[&str], new_path: &[&str], _caller: &Subject) -> Result<(), MountError> {
        let ex = Self::key(existing);
        let np = Self::key(new_path);
        let mut nodes = self.nodes.lock();
        let val = nodes.get(&ex).cloned().ok_or(MountError::NotFound(ex))?;
        nodes.insert(np, val);
        Ok(())
    }

    async fn stat_path(&self, path: &[&str], caller: &Subject) -> Result<Stat, MountError> {
        let fid = self.walk(path, caller).await?;
        let st = Mount::stat(self, &fid, caller).await;
        self.clunk(fid, caller).await;
        st
    }
}

// ── Read-only synthetic Mount (no FsMount) ──────────────────────────────────

/// A plain read-only [`Mount`] — *not* an [`FsMount`]. Used to assert mutations
/// fail closed with `EROFS`.
struct RoMount {
    files: BTreeMap<String, Vec<u8>>,
}

#[async_trait]
impl Mount for RoMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        Ok(Fid::new(MemFid { path: components.join("/") }))
    }
    async fn open(&self, _fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        if mode != 0 {
            Err(MountError::PermissionDenied("read-only".into()))
        } else {
            Ok(())
        }
    }
    async fn read(&self, fid: &Fid, offset: u64, count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let inner = fid.downcast_ref::<MemFid>().unwrap();
        match self.files.get(&inner.path) {
            Some(data) => {
                let s = offset as usize;
                if s >= data.len() { Ok(vec![]) } else { Ok(data[s..std::cmp::min(s + count as usize, data.len())].to_vec()) }
            }
            None => Err(MountError::NotFound(inner.path.clone())),
        }
    }
    async fn write(&self, _fid: &Fid, _o: u64, _d: &[u8], _c: &Subject) -> Result<u32, MountError> {
        Err(MountError::PermissionDenied("read-only".into()))
    }
    async fn readdir(&self, _fid: &Fid, _c: &Subject) -> Result<Vec<DirEntry>, MountError> {
        Ok(vec![])
    }
    async fn stat(&self, fid: &Fid, _c: &Subject) -> Result<Stat, MountError> {
        let inner = fid.downcast_ref::<MemFid>().unwrap();
        match self.files.get(&inner.path) {
            Some(d) => Ok(Stat { qtype: 0, size: d.len() as u64, name: inner.path.clone(), mtime: 0 }),
            None => Err(MountError::NotFound(inner.path.clone())),
        }
    }
    async fn clunk(&self, _fid: Fid, _c: &Subject) {}
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn subject() -> Subject {
    Subject::new("tester")
}

/// A ZeroCopyWriter into an owned Vec (mirrors the production MemWriter).
struct VecWriter(Vec<u8>);
impl io::Write for VecWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl ZeroCopyWriter for VecWriter {
    fn write_from(&mut self, _f: &mut dyn fuse_backend_rs::file_traits::FileReadWriteVolatile, _count: usize, _off: u64) -> io::Result<usize> {
        Ok(0)
    }
    fn available_bytes(&self) -> usize {
        usize::MAX
    }
}

/// A ZeroCopyReader over an owned slice (mirrors the production MemReader).
struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl io::Read for SliceReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = std::cmp::min(out.len(), self.data.len() - self.pos);
        out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}
impl ZeroCopyReader for SliceReader<'_> {
    fn read_to(&mut self, _f: &mut dyn fuse_backend_rs::file_traits::FileReadWriteVolatile, _count: usize, _off: u64) -> io::Result<usize> {
        Ok(0)
    }
}

/// Build a `VfsFileSystem` over a namespace with the given mounts, using a
/// dedicated current-thread runtime driven on a separate worker.
fn build(ns: Namespace) -> VfsFileSystem {
    // The adapter's `block_on` must run on a runtime distinct from the caller's
    // thread; tests use a multi-thread runtime handle.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    // Leak the runtime so its handle outlives the test fs (fine for tests).
    let handle = rt.handle().clone();
    std::mem::forget(rt);
    VfsFileSystem::new(ns, subject(), handle)
}

const ROOT: u64 = 1;

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn lookup_and_getattr() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::with_files(&[("hello.txt", b"hi")]))).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    // /data is an intermediate dir under root.
    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    assert!(data.inode != 0);
    assert_eq!(data.attr.st_mode & libc::S_IFMT, libc::S_IFDIR);

    // /data/hello.txt is a file.
    let file = fs.lookup(&ctx, data.inode, &cstr("hello.txt")).unwrap();
    assert_eq!(file.attr.st_mode & libc::S_IFMT, libc::S_IFREG);
    assert_eq!(file.attr.st_size, 2);

    let (attr, _) = fs.getattr(&ctx, file.inode, None).unwrap();
    assert_eq!(attr.st_size, 2);
}

#[test]
fn read_file() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::with_files(&[("hello.txt", b"hello world")]))).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    let file = fs.lookup(&ctx, data.inode, &cstr("hello.txt")).unwrap();
    let (handle, _, _) = fs.open(&ctx, file.inode, libc::O_RDONLY as u32, 0).unwrap();
    let handle = handle.unwrap();

    let mut w = VecWriter(Vec::new());
    let n = fs.read(&ctx, file.inode, handle, &mut w, 64, 0, None, 0).unwrap();
    assert_eq!(n, 11);
    assert_eq!(w.0, b"hello world");
    fs.release(&ctx, file.inode, 0, handle, false, false, None).unwrap();
}

#[test]
fn create_and_write_and_read() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::new())).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    let args = CreateIn { flags: libc::O_WRONLY as u32, mode: 0o644, umask: 0, fuse_flags: 0 };
    let (entry, handle, _, _) = fs.create(&ctx, data.inode, &cstr("new.txt"), args).unwrap();
    let handle = handle.unwrap();

    let payload = b"written bytes";
    let mut r = SliceReader { data: payload, pos: 0 };
    let n = fs.write(&ctx, entry.inode, handle, &mut r, payload.len() as u32, 0, None, false, 0, 0).unwrap();
    assert_eq!(n, payload.len());
    fs.release(&ctx, entry.inode, 0, handle, true, false, None).unwrap();

    // Re-open and read back.
    let file = fs.lookup(&ctx, data.inode, &cstr("new.txt")).unwrap();
    let (rh, _, _) = fs.open(&ctx, file.inode, libc::O_RDONLY as u32, 0).unwrap();
    let mut w = VecWriter(Vec::new());
    fs.read(&ctx, file.inode, rh.unwrap(), &mut w, 64, 0, None, 0).unwrap();
    assert_eq!(w.0, payload);
}

#[test]
fn readdir_lists_entries() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::with_files(&[("a.txt", b"a"), ("b.txt", b"b"), ("sub/c.txt", b"c")]))).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    let (handle, _) = fs.opendir(&ctx, data.inode, 0).unwrap();
    let handle = handle.unwrap();

    let mut names = Vec::new();
    {
        let mut add = |e: FuseDirEntry| -> io::Result<usize> {
            names.push(String::from_utf8_lossy(e.name).into_owned());
            Ok(e.name.len())
        };
        fs.readdir(&ctx, data.inode, handle, 4096, 0, &mut add).unwrap();
    }

    assert!(names.contains(&".".to_owned()));
    assert!(names.contains(&"..".to_owned()));
    assert!(names.contains(&"a.txt".to_owned()));
    assert!(names.contains(&"b.txt".to_owned()));
    assert!(names.contains(&"sub".to_owned()));
}

#[test]
fn rename_within_mount() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::with_files(&[("old.txt", b"payload")]))).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    fs.rename(&ctx, data.inode, &cstr("old.txt"), data.inode, &cstr("new.txt"), 0).unwrap();

    // old gone, new present.
    assert!(fs.lookup(&ctx, data.inode, &cstr("old.txt")).is_err());
    let new = fs.lookup(&ctx, data.inode, &cstr("new.txt")).unwrap();
    let (rh, _, _) = fs.open(&ctx, new.inode, libc::O_RDONLY as u32, 0).unwrap();
    let mut w = VecWriter(Vec::new());
    fs.read(&ctx, new.inode, rh.unwrap(), &mut w, 64, 0, None, 0).unwrap();
    assert_eq!(w.0, b"payload");
}

#[test]
fn mkdir_and_rmdir() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::new())).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    let dir = fs.mkdir(&ctx, data.inode, &cstr("sub"), 0o755, 0).unwrap();
    assert_eq!(dir.attr.st_mode & libc::S_IFMT, libc::S_IFDIR);

    fs.rmdir(&ctx, data.inode, &cstr("sub")).unwrap();
    assert!(fs.lookup(&ctx, data.inode, &cstr("sub")).is_err());
}

#[test]
fn unlink_removes_file() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::with_files(&[("gone.txt", b"x")]))).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    assert!(fs.lookup(&ctx, data.inode, &cstr("gone.txt")).is_ok());
    fs.unlink(&ctx, data.inode, &cstr("gone.txt")).unwrap();
    assert!(fs.lookup(&ctx, data.inode, &cstr("gone.txt")).is_err());
}

#[test]
fn readonly_mount_is_erofs() {
    let mut ns = Namespace::new();
    let mut files = BTreeMap::new();
    files.insert("ro.txt".to_owned(), b"data".to_vec());
    ns.mount("/ro", Arc::new(RoMount { files })).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let ro = fs.lookup(&ctx, ROOT, &cstr("ro")).unwrap();

    // create / mkdir / unlink on a read-only Mount fail closed with EROFS.
    let args = CreateIn { flags: libc::O_WRONLY as u32, mode: 0o644, umask: 0, fuse_flags: 0 };
    let err = fs.create(&ctx, ro.inode, &cstr("nope.txt"), args).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EROFS));

    let err = fs.mkdir(&ctx, ro.inode, &cstr("nope"), 0o755, 0).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EROFS));

    // Read of an existing file still works.
    let file = fs.lookup(&ctx, ro.inode, &cstr("ro.txt")).unwrap();
    let (h, _, _) = fs.open(&ctx, file.inode, libc::O_RDONLY as u32, 0).unwrap();
    let mut w = VecWriter(Vec::new());
    fs.read(&ctx, file.inode, h.unwrap(), &mut w, 64, 0, None, 0).unwrap();
    assert_eq!(w.0, b"data");

    // Write-open of a read-only mount fails (EROFS).
    let err = fs.open(&ctx, file.inode, libc::O_WRONLY as u32, 0).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EROFS));
}

#[test]
fn forget_reclaims_inode() {
    let mut ns = Namespace::new();
    ns.mount("/data", Arc::new(MemFs::with_files(&[("f.txt", b"x")]))).unwrap();
    let fs = build(ns);
    let ctx = Context::default();

    let data = fs.lookup(&ctx, ROOT, &cstr("data")).unwrap();
    let f = fs.lookup(&ctx, data.inode, &cstr("f.txt")).unwrap();
    // One lookup ref → forget(1) should make a subsequent getattr on the stale
    // inode fail (the inode is reclaimed).
    fs.forget(&ctx, f.inode, 1);
    assert!(fs.getattr(&ctx, f.inode, None).is_err());
}
