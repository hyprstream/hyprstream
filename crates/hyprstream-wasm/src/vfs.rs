//! VFS host-function seam (Profile-A capability surface) — SKETCH for #483.
//!
//! This module is a DESIGN SEAM, not a working implementation. It records the
//! shape of the VFS capability host functions as they will be wired into the
//! wasmtime [`Linker`](wasmtime::Linker) so that a future guest can do file I/O —
//! but ONLY against a per-[`Subject`](crate::Subject) VFS namespace, never the host
//! filesystem.
//!
//! ## How it will wire up (Profile A)
//!
//! Each host fn below maps 1:1 to a method on `hyprstream_vfs`'s `Mount`/`Namespace`
//! surface, every one of which already takes a `&Subject`:
//!
//! ```text
//!   vfs_walk   -> Namespace::walk     (resolve a path to a Fid)
//!   vfs_open   -> Mount::open
//!   vfs_read   -> Namespace::read_one / Mount::read
//!   vfs_write  -> Mount::write
//!   vfs_stat   -> Mount::stat
//!   vfs_ls     -> Namespace::ls / Mount::readdir
//!   vfs_create -> Namespace::create / Mount::create
//! ```
//!
//! The Store data ([`crate::SandboxState`]) already carries the bound `Subject`;
//! the wired host fns will read `caller.data().subject` and pass it straight to the
//! VFS, so file access is Subject-scoped BY CONSTRUCTION (same pattern as
//! `host_random`).
//!
// P1b/#483: the concrete backing will be a `hyprstream_vfs::spawn_vfs_proxy(ns, subject)
// -> tokio::sync::mpsc::Sender<VfsRequest>` held in the Store data. The host fns here
// become sync shims that submit a `VfsRequest { op, reply }` over that Sender and block
// on the reply channel (the proxy bridges sync wasm calls to the async Namespace, and
// its `VfsOp` enum DELIBERATELY excludes mount/bind/unmount — scripts cannot remount).
// We do NOT pull in `hyprstream-vfs` yet (it would drag the RPC/tokio stack into this
// otherwise-minimal host crate); this seam exists only so the design compiles and the
// #483 re-cut has a typed target.
//
//! Until #483, the trait below is backed by a trivial in-memory stub
//! ([`InMemoryVfs`]) used only by the seam test, so the surface compiles and is
//! legible.

use crate::Subject;

/// Errors a VFS capability host fn can return to the guest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfsError {
    /// The subject is not authorized to touch this path.
    Denied,
    /// No such path / fid.
    NotFound,
    /// Not implemented yet (the #483 default).
    Unimplemented,
    /// Any other backend error, stringified.
    Other(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VfsError::Denied => write!(f, "vfs: permission denied"),
            VfsError::NotFound => write!(f, "vfs: not found"),
            VfsError::Unimplemented => write!(f, "vfs: unimplemented (P1b/#483)"),
            VfsError::Other(s) => write!(f, "vfs: {s}"),
        }
    }
}

impl std::error::Error for VfsError {}

/// An opaque file handle (mirrors `hyprstream_vfs::mount::Fid`).
pub type Fid = u64;

/// A directory entry (the subset the guest needs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
}

/// File metadata (the subset the guest needs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stat {
    pub len: u64,
    pub is_dir: bool,
}

/// The Profile-A VFS capability surface, as it will be exposed to the guest.
///
/// EVERY method takes the call's [`Subject`] so the implementation can scope/authorize
/// per identity. This is the exact set of host fns the Linker will define under `env`
/// (`vfs_walk`, `vfs_open`, ...). The ABI those host fns present to wasm (ptr/len pairs
/// into guest linear memory, return-code conventions) is out of scope for this sketch;
/// this trait captures the typed Rust surface they delegate to.
pub trait VfsCapability {
    /// Resolve a path to a file handle. -> `Namespace::walk`.
    fn vfs_walk(&self, subject: &Subject, path: &str) -> Result<Fid, VfsError>;

    /// Open an existing handle with the given mode bits. -> `Mount::open`.
    fn vfs_open(&self, subject: &Subject, fid: Fid, mode: u8) -> Result<(), VfsError>;

    /// Read `count` bytes at `offset` from a handle. -> `Mount::read`.
    fn vfs_read(
        &self,
        subject: &Subject,
        fid: Fid,
        offset: u64,
        count: u32,
    ) -> Result<Vec<u8>, VfsError>;

    /// Write `data` at `offset` to a handle, returning bytes written. -> `Mount::write`.
    fn vfs_write(
        &self,
        subject: &Subject,
        fid: Fid,
        offset: u64,
        data: &[u8],
    ) -> Result<u32, VfsError>;

    /// Stat a handle. -> `Mount::stat`.
    fn vfs_stat(&self, subject: &Subject, fid: Fid) -> Result<Stat, VfsError>;

    /// List a directory handle. -> `Mount::readdir` / `Namespace::ls`.
    fn vfs_ls(&self, subject: &Subject, fid: Fid) -> Result<Vec<DirEntry>, VfsError>;

    /// Create a new file under a directory handle. -> `Mount::create`.
    fn vfs_create(&self, subject: &Subject, dir: Fid, name: &str) -> Result<Fid, VfsError>;
}

/// The #483 default: every VFS call is unimplemented until the real
/// `hyprstream_vfs` proxy is wired in.
///
/// Kept as a typed placeholder so the Linker-wiring code can be written against
/// [`VfsCapability`] today and only the backing object swapped later.
pub struct UnimplementedVfs;

impl VfsCapability for UnimplementedVfs {
    fn vfs_walk(&self, _subject: &Subject, _path: &str) -> Result<Fid, VfsError> {
        // P1b/#483: replace with a VfsRequest over spawn_vfs_proxy's Sender.
        Err(VfsError::Unimplemented)
    }
    fn vfs_open(&self, _subject: &Subject, _fid: Fid, _mode: u8) -> Result<(), VfsError> {
        Err(VfsError::Unimplemented)
    }
    fn vfs_read(
        &self,
        _subject: &Subject,
        _fid: Fid,
        _offset: u64,
        _count: u32,
    ) -> Result<Vec<u8>, VfsError> {
        Err(VfsError::Unimplemented)
    }
    fn vfs_write(
        &self,
        _subject: &Subject,
        _fid: Fid,
        _offset: u64,
        _data: &[u8],
    ) -> Result<u32, VfsError> {
        Err(VfsError::Unimplemented)
    }
    fn vfs_stat(&self, _subject: &Subject, _fid: Fid) -> Result<Stat, VfsError> {
        Err(VfsError::Unimplemented)
    }
    fn vfs_ls(&self, _subject: &Subject, _fid: Fid) -> Result<Vec<DirEntry>, VfsError> {
        Err(VfsError::Unimplemented)
    }
    fn vfs_create(&self, _subject: &Subject, _dir: Fid, _name: &str) -> Result<Fid, VfsError> {
        Err(VfsError::Unimplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// A trivial in-memory VFS used ONLY to prove the seam is legible and that the
    /// Subject is threaded through every call. Not a real namespace; #483 replaces
    /// it with the `hyprstream_vfs` proxy. Single-threaded test stub, so a
    /// `RefCell` is sufficient for interior mutability.
    struct InMemoryVfs {
        // fid -> (name, contents). Subject "denied" is refused, to show scoping.
        files: RefCell<HashMap<Fid, (String, Vec<u8>)>>,
    }

    impl InMemoryVfs {
        fn new() -> Self {
            let mut files = HashMap::new();
            files.insert(1u64, ("hello.txt".to_string(), b"hi".to_vec()));
            Self {
                files: RefCell::new(files),
            }
        }

        fn check(subject: &Subject) -> Result<(), VfsError> {
            if subject.id() == Some("denied") {
                return Err(VfsError::Denied);
            }
            Ok(())
        }
    }

    impl VfsCapability for InMemoryVfs {
        fn vfs_walk(&self, subject: &Subject, path: &str) -> Result<Fid, VfsError> {
            Self::check(subject)?;
            let files = self.files.borrow();
            files
                .iter()
                .find(|(_, (name, _))| name == path)
                .map(|(fid, _)| *fid)
                .ok_or(VfsError::NotFound)
        }
        fn vfs_open(&self, subject: &Subject, fid: Fid, _mode: u8) -> Result<(), VfsError> {
            Self::check(subject)?;
            if self.files.borrow().contains_key(&fid) {
                Ok(())
            } else {
                Err(VfsError::NotFound)
            }
        }
        fn vfs_read(
            &self,
            subject: &Subject,
            fid: Fid,
            offset: u64,
            count: u32,
        ) -> Result<Vec<u8>, VfsError> {
            Self::check(subject)?;
            let files = self.files.borrow();
            let (_, data) = files.get(&fid).ok_or(VfsError::NotFound)?;
            let start = (offset as usize).min(data.len());
            let end = (start + count as usize).min(data.len());
            Ok(data[start..end].to_vec())
        }
        fn vfs_write(
            &self,
            subject: &Subject,
            fid: Fid,
            _offset: u64,
            data: &[u8],
        ) -> Result<u32, VfsError> {
            Self::check(subject)?;
            let mut files = self.files.borrow_mut();
            let entry = files.get_mut(&fid).ok_or(VfsError::NotFound)?;
            entry.1 = data.to_vec();
            Ok(data.len() as u32)
        }
        fn vfs_stat(&self, subject: &Subject, fid: Fid) -> Result<Stat, VfsError> {
            Self::check(subject)?;
            let files = self.files.borrow();
            let (_, data) = files.get(&fid).ok_or(VfsError::NotFound)?;
            Ok(Stat {
                len: data.len() as u64,
                is_dir: false,
            })
        }
        fn vfs_ls(&self, subject: &Subject, _fid: Fid) -> Result<Vec<DirEntry>, VfsError> {
            Self::check(subject)?;
            Ok(self
                .files
                .borrow()
                .values()
                .map(|(name, _)| DirEntry {
                    name: name.clone(),
                    is_dir: false,
                })
                .collect())
        }
        fn vfs_create(&self, subject: &Subject, _dir: Fid, name: &str) -> Result<Fid, VfsError> {
            Self::check(subject)?;
            let mut files = self.files.borrow_mut();
            let fid = files.keys().max().copied().unwrap_or(0) + 1;
            files.insert(fid, (name.to_string(), Vec::new()));
            Ok(fid)
        }
    }

    #[test]
    fn unimplemented_seam_is_inert() {
        let vfs = UnimplementedVfs;
        let s = Subject::anonymous();
        assert_eq!(vfs.vfs_walk(&s, "x"), Err(VfsError::Unimplemented));
        assert_eq!(vfs.vfs_stat(&s, 0), Err(VfsError::Unimplemented));
    }

    #[test]
    fn seam_threads_subject_through_and_scopes() {
        let vfs = InMemoryVfs::new();
        let ok = Subject::named("alice");
        let denied = Subject::named("denied");

        // The Subject reaches the backend: "denied" is refused at the seam.
        assert_eq!(vfs.vfs_walk(&denied, "hello.txt"), Err(VfsError::Denied));

        // An allowed subject can resolve + read.
        let fid = vfs.vfs_walk(&ok, "hello.txt").expect("walk");
        vfs.vfs_open(&ok, fid, 0).expect("open");
        let data = vfs.vfs_read(&ok, fid, 0, 8).expect("read");
        assert_eq!(data, b"hi");
        assert_eq!(vfs.vfs_stat(&ok, fid).unwrap().len, 2);
    }
}
