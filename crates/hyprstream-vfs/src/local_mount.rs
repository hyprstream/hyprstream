//! Local mount trait — in-process 9P operations without network transport.

use hyprstream_rpc::Subject;

/// Opaque fid handle from a local mount.
pub struct LocalFid(Box<dyn std::any::Any + Send + Sync>);

impl LocalFid {
    /// Create a new LocalFid wrapping an arbitrary Send+Sync value.
    pub fn new<T: std::any::Any + Send + Sync>(val: T) -> Self {
        Self(Box::new(val))
    }

    /// Downcast to a concrete type (immutable).
    pub fn downcast_ref<T: std::any::Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }

    /// Downcast to a concrete type (mutable).
    pub fn downcast_mut<T: std::any::Any>(&mut self) -> Option<&mut T> {
        self.0.downcast_mut()
    }
}

/// Directory entry.
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// File metadata.
#[derive(Clone, Debug)]
pub struct Stat {
    pub qtype: u8,
    pub size: u64,
    pub name: String,
    pub mtime: u64,
}

/// In-process mount backend. Implements 9P semantics without ZMQ/IPC.
///
/// Used for `/config/`, `/private/`, `/cmd/`, `/tcl/`, and any other
/// client-local namespace entries. Every method receives the verified
/// `Subject` of the caller for per-tenant fid isolation and policy checks.
pub trait LocalMount: Send + Sync {
    /// Walk path components, returning an opaque fid.
    fn walk(&self, components: &[&str], caller: &Subject) -> Result<LocalFid, String>;

    /// Open a walked fid for I/O. `mode`: OREAD=0, OWRITE=1, ORDWR=2.
    fn open(&self, fid: &mut LocalFid, mode: u8, caller: &Subject) -> Result<(), String>;

    /// Read bytes from an open fid at offset.
    fn read(&self, fid: &LocalFid, offset: u64, count: u32, caller: &Subject) -> Result<Vec<u8>, String>;

    /// Write bytes to an open fid at offset. Returns bytes written.
    fn write(&self, fid: &LocalFid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, String>;

    /// Read directory entries from an open directory fid.
    fn readdir(&self, fid: &LocalFid, caller: &Subject) -> Result<Vec<DirEntry>, String>;

    /// Get file metadata.
    fn stat(&self, fid: &LocalFid, caller: &Subject) -> Result<Stat, String>;

    /// Release a fid.
    fn clunk(&self, fid: LocalFid, caller: &Subject);
}
