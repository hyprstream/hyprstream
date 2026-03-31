//! Generic async interface for 9P-shaped filesystem operations.
//!
//! `FsClient` abstracts the RPC layer so that `RemoteMount<C>` in hyprstream-vfs
//! can work with any service's fs scope (model, mcp, etc.) without depending on
//! generated client types.

use async_trait::async_trait;

/// Result of a walk operation.
#[derive(Clone, Debug)]
pub struct FsWalkResult {
    /// Qid type of the walked-to entry (QTDIR=0x80 or QTFILE=0x00).
    pub qtype: u8,
}

/// Result of an open operation.
#[derive(Clone, Debug)]
pub struct FsOpenResult {
    /// Qid type of the opened entry.
    pub qtype: u8,
    /// I/O unit size (0 = no limit).
    pub iounit: u32,
}

/// Result of a stat operation.
#[derive(Clone, Debug)]
pub struct FsStatResult {
    /// Qid type.
    pub qtype: u8,
    /// File size in bytes.
    pub size: u64,
    /// File name.
    pub name: String,
    /// Modification time (seconds since epoch).
    pub mtime: u64,
}

/// Common async interface for any service's 9P-shaped filesystem scope.
///
/// Generated scoped clients (e.g., `ModelFsClient`) don't implement this
/// directly — an adapter struct bridges between the generated API and this
/// trait (see `ModelFsAdapter` in hyprstream).
///
/// The `fs_` prefix avoids name collisions when a type implements multiple
/// traits.
#[async_trait]
pub trait FsClient: Send + Sync {
    /// Walk path components, allocating `newfid` for the result.
    async fn fs_walk(&self, wnames: Vec<String>, newfid: u32) -> Result<FsWalkResult, String>;

    /// Open a fid for I/O with the given mode.
    async fn fs_open(&self, fid: u32, mode: u8) -> Result<FsOpenResult, String>;

    /// Read bytes from an open fid at the given offset.
    async fn fs_read(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>, String>;

    /// Write bytes to an open fid at the given offset. Returns count written.
    async fn fs_write(&self, fid: u32, offset: u64, data: Vec<u8>) -> Result<u32, String>;

    /// Release a fid. Best-effort; errors are ignored by callers.
    async fn fs_clunk(&self, fid: u32) -> Result<(), String>;

    /// Stat a fid.
    async fn fs_stat(&self, fid: u32) -> Result<FsStatResult, String>;

    /// Read raw directory data from an open directory fid.
    async fn fs_readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>, String>;
}
