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
    /// Establish the session at `Tattach`, given the `uname` and `aname` the
    /// client presented. The default is a no-op: backends whose caller identity
    /// and export root are fixed at construction (the UDS/vsock listeners)
    /// ignore both fields.
    ///
    /// A backend that resolves its caller identity from the attach itself
    /// (the H1b `/9p` WebTransport plane carries a mount ticket in
    /// `Tattach.uname` — the browser `WebSocket` can't set headers and the
    /// cert-pinned WT session has no URL query) validates it here and binds
    /// the session `Subject`. `aname` is the 9P attach name/export selector;
    /// implementations may ignore it for a single-root export, or use it to
    /// select and authorize a narrower root. Returning `Err` fails the attach;
    /// the translator maps it to an `Rlerror` errno (a
    /// `hyprstream_vfs::MountError` in the cause chain picks the errno — e.g.
    /// `PermissionDenied → EACCES`).
    async fn attach(&self, _uname: &str, _aname: &str) -> anyhow::Result<()> {
        Ok(())
    }

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
