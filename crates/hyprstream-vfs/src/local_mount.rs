//! Local mount trait — in-process 9P operations without network transport.

/// Opaque fid handle from a local mount.
#[allow(dead_code)] // Field accessed via downcast in LocalMount impls.
pub struct LocalFid(pub(crate) Box<dyn std::any::Any + Send + Sync>);

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
/// client-local namespace entries.
pub trait LocalMount: Send + Sync {
    /// Walk path components, returning an opaque fid.
    fn walk(&self, components: &[&str]) -> Result<LocalFid, String>;

    /// Open a walked fid for I/O. `mode`: OREAD=0, OWRITE=1, ORDWR=2.
    fn open(&self, fid: &mut LocalFid, mode: u8) -> Result<(), String>;

    /// Read bytes from an open fid at offset.
    fn read(&self, fid: &LocalFid, offset: u64, count: u32) -> Result<Vec<u8>, String>;

    /// Write bytes to an open fid at offset. Returns bytes written.
    fn write(&self, fid: &LocalFid, offset: u64, data: &[u8]) -> Result<u32, String>;

    /// Read directory entries from an open directory fid.
    fn readdir(&self, fid: &LocalFid) -> Result<Vec<DirEntry>, String>;

    /// Get file metadata.
    fn stat(&self, fid: &LocalFid) -> Result<Stat, String>;

    /// Release a fid.
    fn clunk(&self, fid: LocalFid);
}
