//! Backend trait — the capnp-RPC-shaped seam the translator dispatches to.
//!
//! Each method mirrors a 9P2000.L filesystem operation as expressed in
//! `nine.capnp` (NpWalk / NpOpen / NpRead / NpWrite / NpClunk / NpStatReq).
//! The translator owns the server-side fid table and calls these methods
//! after resolving the incoming 9P fid.
//!
//! `MemoryBackend` ([`crate::memory::MemoryBackend`]) provides an
//! in-process implementation for standalone tests; the `hyprstream` binary
//! crate ships a `ModelBackend` that wraps the generated `ModelClient` and
//! turns each call into a capnp RPC against the model service's `fs` scope
//! (the inverse of `RemoteModelMount`, which goes VFS → RPC).

use async_trait::async_trait;

use crate::msg::Qid;

/// Result of a walk: the qid of the resolved path and, for directories, the
/// qids of each intermediate component. For a single-component walk (the
/// common case for this translator) `qids` holds exactly the leaf qid.
#[derive(Debug, Clone)]
pub struct WalkResult {
    /// Qids for each path component successfully walked (length matches the
    /// number of components traversed; the client treats a short list as
    /// "walk stopped early").
    pub qids: Vec<Qid>,
}

/// Result of an open/lopen.
#[derive(Debug, Clone)]
pub struct OpenResult {
    pub qid: Qid,
    /// Maximum read/write unit the backend will accept in a single call.
    pub iounit: u32,
}

/// Result of a stat/getattr.
#[derive(Debug, Clone)]
pub struct StatResult {
    pub qid: Qid,
    /// Unix mode bits (e.g. `0o040755` for a dir).
    pub mode: u32,
    pub size: u64,
    pub mtime_sec: u64,
}

/// 9P filesystem backend.
///
/// All methods are async and `Send`-safe so the translator can drive them on
/// a multi-threaded tokio runtime. Fid numbers are opaque to the backend —
/// the translator allocates and tracks them; the backend sees a stable fid
/// per walked path.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Walk `components` from `fid` (the previously-walked parent, or the
    /// attach root). Allocates internal state for `newfid`.
    async fn walk(
        &self,
        fid: u32,
        newfid: u32,
        components: &[String],
    ) -> anyhow::Result<WalkResult>;

    /// Open `fid` with Linux-style `flags` (Tlopen). Returns the leaf qid
    /// and the negotiated iounit.
    async fn open(&self, fid: u32, flags: u32) -> anyhow::Result<OpenResult>;

    /// Read up to `count` bytes from `fid` at `offset`.
    async fn read(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>>;

    /// Write `data` to `fid` at `offset`. Returns the number of bytes written.
    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> anyhow::Result<u32>;

    /// Stat `fid`.
    async fn stat(&self, fid: u32) -> anyhow::Result<StatResult>;

    /// List directory entries under `fid` (a directory). The returned bytes
    /// are 9P readdir records; the translator passes them through verbatim.
    async fn readdir(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>>;

    /// Release `fid`. Best-effort — errors are logged and ignored by the
    /// translator (the fid is dropped from its table regardless).
    async fn clunk(&self, fid: u32) -> anyhow::Result<()>;
}
