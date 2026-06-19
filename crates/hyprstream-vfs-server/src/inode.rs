//! Inode and handle tables for the `Namespace → FileSystem` down-adapter.
//!
//! `fuse_backend_rs::FileSystem` is inode/handle addressed: the guest kernel
//! refers to files by an opaque `u64` inode (returned from `lookup`) and to open
//! files/dirs by an opaque `u64` handle (returned from `open`/`opendir`). The
//! VFS [`Namespace`](hyprstream_vfs::Namespace), by contrast, is **path
//! addressed** — every op takes a component slice rooted at `/`. The two tables
//! here own that translation:
//!
//! - [`InodeTable`] maps each live inode to its absolute VFS path (the component
//!   vector) plus a kind (file / dir / symlink). The FUSE root (`FUSE_ROOT_ID`,
//!   `1`) is the namespace root `/`. Inodes are allocated lazily on `lookup` and
//!   reference-counted so `forget` can reclaim them (matching the kernel's
//!   lookup-count contract).
//! - [`HandleTable`] maps each open handle to the VFS [`Fid`] obtained from
//!   `Mount::walk` + `Mount::open`, so subsequent `read`/`write`/`readdir`/
//!   `release` can reach the same open object.
//!
//! Both tables are plain `Mutex`-guarded maps — genuinely `Send + Sync`, no wasm
//! `unsafe impl`. The whole crate is native-only.

use std::collections::HashMap;

use parking_lot::Mutex;

/// The FUSE root inode (`FUSE_ROOT_ID`). Maps to the namespace root `/`.
pub const ROOT_INODE: u64 = 1;

/// The kind of object an inode refers to. Drives the `st_mode` we synthesise and
/// whether an open goes through the dir or file path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Dir,
    File,
    Symlink,
}

/// An inode's identity within the namespace: its absolute path components and
/// kind. The path is stored as owned components (never a `/`-joined string) so
/// it can be handed straight to the VFS `walk`/`FsMount` ops as a `&[&str]`.
#[derive(Clone, Debug)]
pub struct InodeData {
    pub components: Vec<String>,
    pub kind: Kind,
    /// Lookup reference count (the kernel's `nlookup`). When it hits zero on
    /// `forget`, the entry is evictable.
    pub refcount: u64,
}

/// Inode allocation + path translation table.
pub struct InodeTable {
    /// inode → data.
    by_inode: Mutex<HashMap<u64, InodeData>>,
    /// `/`-joined path → inode, so repeated `lookup`s of the same path reuse the
    /// inode (the kernel relies on a stable inode per path while referenced).
    by_path: Mutex<HashMap<String, u64>>,
    /// Next inode to hand out (root is 1, so we start at 2).
    next: Mutex<u64>,
}

impl InodeTable {
    /// Create a table pre-populated with the root inode mapped to `/`.
    pub fn new() -> Self {
        let mut by_inode = HashMap::new();
        by_inode.insert(
            ROOT_INODE,
            InodeData {
                components: Vec::new(),
                kind: Kind::Dir,
                // The root is never forgotten.
                refcount: u64::MAX,
            },
        );
        let mut by_path = HashMap::new();
        by_path.insert(String::new(), ROOT_INODE);
        Self {
            by_inode: Mutex::new(by_inode),
            by_path: Mutex::new(by_path),
            next: Mutex::new(ROOT_INODE + 1),
        }
    }

    fn key(components: &[String]) -> String {
        components.join("/")
    }

    /// Look up the data for an inode (a clone, to avoid holding the lock).
    pub fn get(&self, inode: u64) -> Option<InodeData> {
        self.by_inode.lock().get(&inode).cloned()
    }

    /// Get-or-allocate an inode for `components`/`kind`, bumping its lookup
    /// refcount. Returns the inode number. Used by `lookup`/`create`/`mkdir`.
    pub fn intern(&self, components: Vec<String>, kind: Kind) -> u64 {
        let key = Self::key(&components);
        let mut by_path = self.by_path.lock();
        let mut by_inode = self.by_inode.lock();
        if let Some(&inode) = by_path.get(&key) {
            if let Some(data) = by_inode.get_mut(&inode) {
                data.refcount = data.refcount.saturating_add(1);
                // A path can change kind (e.g. recreated as a different type);
                // keep the latest observed kind.
                data.kind = kind;
            }
            return inode;
        }
        let inode = {
            let mut n = self.next.lock();
            let i = *n;
            *n += 1;
            i
        };
        by_inode.insert(
            inode,
            InodeData {
                components,
                kind,
                refcount: 1,
            },
        );
        by_path.insert(key, inode);
        inode
    }

    /// Drop `count` lookup references from `inode`; evict when the count reaches
    /// zero (mirrors the FUSE `forget` contract). The root is never evicted.
    pub fn forget(&self, inode: u64, count: u64) {
        if inode == ROOT_INODE {
            return;
        }
        let mut by_inode = self.by_inode.lock();
        let evict = if let Some(data) = by_inode.get_mut(&inode) {
            data.refcount = data.refcount.saturating_sub(count);
            data.refcount == 0
        } else {
            false
        };
        if evict {
            if let Some(data) = by_inode.remove(&inode) {
                self.by_path.lock().remove(&Self::key(&data.components));
            }
        }
    }

    /// Forget a path's inode entirely (used after unlink/rmdir/rename so a stale
    /// path no longer resolves to a live inode). No-op if not present.
    pub fn invalidate_path(&self, components: &[String]) {
        let key = Self::key(components);
        if let Some(inode) = self.by_path.lock().remove(&key) {
            self.by_inode.lock().remove(&inode);
        }
    }
}

impl Default for InodeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// An open object: the VFS [`Fid`](hyprstream_vfs::Fid) plus the mount that
/// produced it. A `Fid` is opaque to the adapter and does not record its owning
/// mount, so we keep the `Arc<dyn Mount>` alongside it; `read`/`write`/`clunk`
/// route back to that exact mount.
pub struct OpenFid {
    pub mount: hyprstream_vfs::MountTarget,
    pub fid: hyprstream_vfs::Fid,
}

/// Open-handle table mapping `u64` handles to live [`OpenFid`]s.
///
/// [`hyprstream_vfs::Fid`] is `Send + Sync` but not `Clone`; we own it here for
/// the lifetime of the open file. Because the async VFS ops borrow the fid across
/// an await, the adapter [`take`](Self::take)s the `OpenFid` out, awaits, then
/// [`put`](Self::put)s it back under the same handle.
pub struct HandleTable {
    handles: Mutex<HashMap<u64, OpenFid>>,
    next: Mutex<u64>,
}

impl HandleTable {
    pub fn new() -> Self {
        Self {
            handles: Mutex::new(HashMap::new()),
            next: Mutex::new(1),
        }
    }

    /// Store an open fid + its mount, returning the handle id.
    pub fn insert(&self, open: OpenFid) -> u64 {
        let handle = {
            let mut n = self.next.lock();
            let h = *n;
            *n += 1;
            h
        };
        self.handles.lock().insert(handle, open);
        handle
    }

    /// Remove and return the open fid for `handle` (on `release`/`releasedir`).
    pub fn remove(&self, handle: u64) -> Option<OpenFid> {
        self.handles.lock().remove(&handle)
    }

    /// Temporarily take the open fid out so an `async` op can borrow it across an
    /// await without holding the table lock. Pair with [`Self::put`].
    pub fn take(&self, handle: u64) -> Option<OpenFid> {
        self.handles.lock().remove(&handle)
    }

    /// Return an open fid taken via [`Self::take`] under the same handle id.
    pub fn put(&self, handle: u64, open: OpenFid) {
        self.handles.lock().insert(handle, open);
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}
