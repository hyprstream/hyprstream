//! Server-side 9P2000.L translator.
//!
//! Accepts 9P connections (TCP for now; virtio-9P later) and translates each
//! T-message into a [`Backend`] call. The backend is the capnp-RPC seam:
//! the `hyprstream` binary crate wraps `ModelClient` to turn each call into a
//! `nine.capnp` envelope against the model service's `fs` scope, mirroring
//! `RemoteModelMount` (which goes the other way: VFS → RPC).
//!
//! ## Server-side fid table
//!
//! The translator owns a [`FidTable`] mapping 9P fid → backend fid + metadata
//! (qtype, opened flag). This is the inverse of `RemoteModelMount`'s
//! client-side fid table.
//!
//! ## Transport
//!
//! [`Translator::serve`] runs a TCP accept loop; [`Translator::serve_uds`] runs
//! the same loop over a Unix domain socket ([`tokio::net::UnixListener`]). Both
//! delegate to one transport-agnostic accept loop (`serve_listener`) and the
//! same per-connection [`Translator::serve_connection`] core, which operates on
//! any `AsyncRead + AsyncWrite` stream. The UDS entry point is how a native
//! Wanix workload consumes a Subject-scoped export via its `p9kit.ClientFS` 9P
//! client (#506); [`Translator::from_mount`] / [`serve_mount_uds`] build the
//! translator directly from an `Arc<dyn Mount>` + `Subject`. virtio-9P will plug
//! in the same way with a third transport.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_rpc::Subject;
use hyprstream_vfs::Mount;
use tokio::io::{split, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

use crate::backend::Backend;
use crate::mount_backend::MountBackend;
use crate::msg::{
    self, encode_response, parse_request, rflush, Request, Response,
};

/// Negotiated maximum message size. Generous: Kata guest mounts may issue
/// large read-ahead. The backend's iounit still bounds individual reads.
const MSG_SIZE: u32 = 8 * 1024;

/// Per-fid state held by the translator (server side).
#[derive(Clone, Debug)]
struct FidState {
    /// Qtype of the walked file (QTDIR=0x80 / QTFILE=0x00).
    qtype: u8,
    /// Whether open() has succeeded on this fid.
    opened: bool,
}

/// Server-side fid table: 9P fid → state.
///
/// `DashMap` so the translator can be shared across per-connection tasks.
#[derive(Default)]
pub struct FidTable {
    fids: DashMap<u32, FidState>,
}

impl FidTable {
    /// Insert state for a fid (from walk/attach).
    pub fn insert(&self, fid: u32, qtype: u8) {
        self.fids.insert(fid, FidState { qtype, opened: false });
    }

    /// Mark a fid as opened.
    pub fn set_opened(&self, fid: u32) {
        if let Some(mut s) = self.fids.get_mut(&fid) {
            s.opened = true;
        }
    }

    /// Look up a fid's qtype.
    pub fn qtype(&self, fid: u32) -> Option<u8> {
        self.fids.get(&fid).map(|s| s.qtype)
    }

    /// Remove a fid (clunk).
    pub fn remove(&self, fid: u32) {
        self.fids.remove(&fid);
    }
}

/// The translator: a backend plus a shared fid table.
///
/// Cloning is cheap (Arc internals); one translator serves many connections.
pub struct Translator {
    backend: Arc<dyn Backend>,
    fids: Arc<FidTable>,
}

impl Translator {
    /// Build a translator over the given backend. The fid table starts empty.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend, fids: Arc::new(FidTable::default()) }
    }

    /// Build a translator that exports a single Subject-scoped VFS [`Mount`] as
    /// its 9P root.
    ///
    /// This is the entry point for #506: `mount` is the tenant's export root
    /// (already the single MAC policy-enforcement point) and `subject` is the
    /// verified caller identity threaded onto every backend op. It wraps the
    /// mount in a [`MountBackend`] — the same [`Backend`] seam the capnp-RPC
    /// `ModelBackend` uses on the TCP path — so no new attachment mechanism is
    /// introduced; only the export root differs.
    pub fn from_mount(mount: Arc<dyn Mount>, subject: Subject) -> Self {
        Self::new(Arc::new(MountBackend::new(mount, subject)))
    }

    /// Run a TCP accept loop until the listener errors or is closed.
    ///
    /// Each connection runs on its own task. Errors on one connection do not
    /// affect siblings.
    pub async fn serve(self, listener: TcpListener) -> Result<()> {
        self.serve_listener(listener).await
    }

    /// Run the accept loop over a Unix domain socket.
    ///
    /// Identical per-connection handling to [`Translator::serve`]; only the
    /// transport differs. This is how a native Wanix workload attaches to a
    /// Subject-scoped export over a UDS via its `p9kit.ClientFS` 9P client.
    pub async fn serve_uds(self, listener: UnixListener) -> Result<()> {
        self.serve_listener(listener).await
    }

    /// Transport-agnostic accept loop shared by [`serve`](Self::serve) and
    /// [`serve_uds`](Self::serve_uds). Each accepted connection is served on its
    /// own task via [`serve_connection`](Self::serve_connection); errors on one
    /// connection do not affect siblings.
    async fn serve_listener<L: Listen>(self, listener: L) -> Result<()> {
        let this = Arc::new(self);
        info!(endpoint = %listener.describe(), "9P translator: listening");
        loop {
            let (stream, peer) = match listener.accept_conn().await {
                Ok(v) => v,
                Err(e) => {
                    error!(error = %e, "9P translator: accept failed, shutting down");
                    return Err(e).context("9P accept loop terminated");
                }
            };
            let this = Arc::clone(&this);
            tokio::spawn(async move {
                debug!(%peer, "9P connection accepted");
                if let Err(e) = this.serve_connection(stream).await {
                    warn!(%peer, error = %e, "9P connection ended with error");
                }
            });
        }
    }

    /// Serve a single connection to completion (until EOF, parse error, or
    /// the peer resets).
    ///
    /// Generic over the byte stream so the same core serves TCP, UDS, and any
    /// future `AsyncRead + AsyncWrite` transport.
    pub async fn serve_connection<S>(self: &Arc<Self>, stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
    {
        let (mut rx, mut tx) = split(stream);
        // Reusable growable read buffer; capped per-message by msize.
        let mut len_buf = [0u8; 4];

        loop {
            // Read the 4-byte length prefix.
            let n = rx.read(&mut len_buf).await?;
            if n == 0 {
                debug!("9P peer closed connection");
                return Ok(());
            }
            if n < 4 {
                rx.read_exact(&mut len_buf[n..]).await?;
            }
            let total = u32::from_le_bytes(len_buf) as usize;
            if total < 7 || total > MSG_SIZE as usize {
                anyhow::bail!("invalid 9P message length: {total}");
            }

            // Read the rest of the message.
            let mut buf = vec![0u8; total];
            buf[..4].copy_from_slice(&len_buf);
            rx.read_exact(&mut buf[4..]).await?;

            // Tflush is handled inline: requests are processed serially per
            // connection, so there is never an outstanding request to cancel —
            // acknowledge with Rflush and continue. (Response has no Flush
            // variant because flush is a no-op at this layer.)
            let msg_type = buf.get(4).copied().unwrap_or(0);
            if msg_type == msg::TFLUSH {
                let tag = tag_or_notag(&buf);
                tx.write_all(&rflush(tag)).await?;
                continue;
            }

            let (tag, response) = match self.handle_message(&buf).await {
                Ok(r) => r,
                Err(e) => {
                    // Map a handler error to Rlerror (errno-style code).
                    // We use a synthetic code; real errno mapping is backend-
                    // specific and deferred to a later hardening pass.
                    let err_tag = tag_or_notag(&buf);
                    warn!(error = %e, tag = err_tag, "9P handler error");
                    (err_tag, Response::Error { ecode: libc_eio() })
                }
            };

            let out = encode_response(tag, &response);
            tx.write_all(&out).await?;
        }
    }

    /// Decode one T-message, dispatch it to the backend, and produce the
    /// R-message. Returns `(tag, response)`.
    async fn handle_message(&self, buf: &[u8]) -> Result<(u16, Response)> {
        let (tag, req) = parse_request(buf).context("decode 9P T-message")?;
        let resp = match req {
            Request::Version { msize, version } => self.handle_version(msize, version),
            Request::Attach { fid, .. } => self.handle_attach(fid).await?,
            // Flush is intercepted in serve_connection before reaching here;
            // this arm is unreachable but kept exhaustive.
            Request::Flush { .. } => Response::Clunk,
            Request::Walk { fid, newfid, wnames } => {
                self.handle_walk(fid, newfid, wnames).await?
            }
            Request::Lopen { fid, flags } => self.handle_lopen(fid, flags).await?,
            Request::Read { fid, offset, count } => {
                self.handle_read(fid, offset, count).await?
            }
            Request::Write { fid, offset, data } => {
                self.handle_write(fid, offset, &data).await?
            }
            Request::Clunk { fid } => self.handle_clunk(fid).await,
            Request::Getattr { fid, .. } => self.handle_getattr(fid).await?,
            Request::Readdir { fid, offset, count } => {
                self.handle_readdir(fid, offset, count).await?
            }
        };
        Ok((tag, resp))
    }

    // ────────────────────────────────────────────────────────────────────
    // Per-message handlers
    // ────────────────────────────────────────────────────────────────────

    fn handle_version(&self, requested: u32, version: String) -> Response {
        if version != "9P2000.L" && !version.starts_with("9P2000") {
            return Response::Error { ecode: libc_einval() };
        }
        let msize = requested.clamp(256, MSG_SIZE);
        Response::Version { msize, version: "9P2000.L".into() }
    }

    async fn handle_attach(&self, fid: u32) -> Result<Response> {
        // Attach establishes the root fid. Walk the empty path to materialize
        // a root qid from the backend, then record it.
        let walk = self.backend.walk(fid, fid, &[]).await?;
        let qtype = walk.qids.last().map(|q| q.qtype).unwrap_or(0);
        self.fids.insert(fid, qtype);
        let qid = walk.qids.into_iter().next().unwrap_or_default();
        Ok(Response::Attach { qid })
    }

    async fn handle_walk(
        &self,
        fid: u32,
        newfid: u32,
        wnames: Vec<String>,
    ) -> Result<Response> {
        // Mirror of RemoteModelMount::walk: call backend.walk with the source
        // fid, the new fid, and the path components.
        let result = self.backend.walk(fid, newfid, &wnames).await?;
        if let Some(q) = result.qids.last() {
            self.fids.insert(newfid, q.qtype);
        }
        Ok(Response::Walk { qids: result.qids })
    }

    async fn handle_lopen(&self, fid: u32, flags: u32) -> Result<Response> {
        let open = self.backend.open(fid, flags).await?;
        self.fids.set_opened(fid);
        Ok(Response::Lopen { qid: open.qid, iounit: open.iounit })
    }

    async fn handle_read(&self, fid: u32, offset: u64, count: u32) -> Result<Response> {
        let data = self.backend.read(fid, offset, count).await?;
        Ok(Response::Read { data })
    }

    async fn handle_write(&self, fid: u32, offset: u64, data: &[u8]) -> Result<Response> {
        let count = self.backend.write(fid, offset, data).await?;
        Ok(Response::Write { count })
    }

    async fn handle_clunk(&self, fid: u32) -> Response {
        // Best-effort clunk on the backend; the local fid is dropped either way.
        if let Err(e) = self.backend.clunk(fid).await {
            debug!(fid, error = %e, "backend clunk failed (ignored)");
        }
        self.fids.remove(fid);
        Response::Clunk
    }

    async fn handle_getattr(&self, fid: u32) -> Result<Response> {
        let st = self.backend.stat(fid).await?;
        Ok(Response::Getattr { qid: st.qid, mode: st.mode, size: st.size, mtime_sec: st.mtime_sec })
    }

    async fn handle_readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Response> {
        let data = self.backend.readdir(fid, offset, count).await?;
        Ok(Response::Readdir { data })
    }
}

/// Bind a Unix domain socket at `path` and serve `mount` (scoped to `subject`)
/// as its 9P2000.L root until the socket errors or is closed.
///
/// Convenience wrapper over [`Translator::from_mount`] + [`Translator::serve_uds`]
/// — the #506 one-call entry point a supervisor uses to expose a tenant's
/// Subject-scoped export to a native Wanix `p9kit.ClientFS` client.
pub async fn serve_mount_uds(
    mount: Arc<dyn Mount>,
    subject: Subject,
    path: impl AsRef<Path>,
) -> Result<()> {
    let listener = UnixListener::bind(path.as_ref())
        .with_context(|| format!("bind 9P UDS listener at {:?}", path.as_ref()))?;
    Translator::from_mount(mount, subject).serve_uds(listener).await
}

/// Transport-agnostic accept surface, implemented for [`TcpListener`] and
/// [`UnixListener`] so both share one accept loop (`serve_listener`).
///
/// Keeping this private and trait-bounded (rather than duplicating the loop)
/// means TCP and UDS run byte-identical connection handling; a new transport
/// only implements `accept_conn` + `describe`.
#[async_trait]
trait Listen: Send + Sync + 'static {
    /// The per-connection byte stream this listener produces.
    type Conn: AsyncRead + AsyncWrite + Send + 'static;

    /// Accept one connection, returning the stream and a display string for the
    /// peer (used only for tracing).
    async fn accept_conn(&self) -> std::io::Result<(Self::Conn, String)>;

    /// Human-readable endpoint description for the "listening" log line.
    fn describe(&self) -> String;
}

#[async_trait]
impl Listen for TcpListener {
    type Conn = TcpStream;

    async fn accept_conn(&self) -> std::io::Result<(TcpStream, String)> {
        let (stream, peer) = self.accept().await?;
        Ok((stream, peer.to_string()))
    }

    fn describe(&self) -> String {
        self.local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "tcp:?".to_owned())
    }
}

#[async_trait]
impl Listen for UnixListener {
    type Conn = UnixStream;

    async fn accept_conn(&self) -> std::io::Result<(UnixStream, String)> {
        let (stream, peer) = self.accept().await?;
        Ok((stream, format!("unix:{peer:?}")))
    }

    fn describe(&self) -> String {
        self.local_addr()
            .map(|a| format!("unix:{a:?}"))
            .unwrap_or_else(|_| "unix:?".to_owned())
    }
}

/// Extract the tag from a raw message for error-path Rlerror encoding.
fn tag_or_notag(buf: &[u8]) -> u16 {
    if buf.len() >= 6 {
        u16::from_le_bytes([buf[5], buf[6]])
    } else {
        0xFFFF // NOTAG
    }
}

// errno-style codes the translator emits on its own errors. We hardcode the
// integer values rather than depend on libc so the crate stays wasm-friendly.
const fn libc_eio() -> u32 {
    5 // EIO
}
const fn libc_einval() -> u32 {
    22 // EINVAL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryBackend;
    use crate::msg::Qid;

    fn sample_qid(qtype: u8, path: u64) -> Qid {
        Qid { qtype, version: 1, path }
    }

    #[tokio::test]
    async fn version_round_trips() {
        let t = Translator::new(Arc::new(MemoryBackend::default()));
        // Build a Tversion by hand via the codec and decode the response.
        let req = msg::tversion(1, 4096, "9P2000.L");
        let (tag, resp) = t.handle_message(&req).await.unwrap();
        assert_eq!(tag, 1);
        match resp {
            Response::Version { msize, version } => {
                assert_eq!(version, "9P2000.L");
                assert!(msize <= 4096);
            }
            other => panic!("expected Version, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn attach_then_walk_open_read_clunk() {
        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello world");
        let t = Translator::new(Arc::new(backend));

        // Attach: fid 0 becomes the root.
        let req = msg::tattach(1, 0, u32::MAX, "user", "/");
        let (_, resp) = t.handle_message(&req).await.unwrap();
        assert!(matches!(resp, Response::Attach { .. }));

        // Walk from root (fid 0) to a new fid (1) for the file.
        let req = msg::twalk(2, 0, 1, &["hello.txt"]);
        let (_, resp) = t.handle_message(&req).await.unwrap();
        match resp {
            Response::Walk { qids } => assert_eq!(qids.len(), 1),
            other => panic!("expected Walk, got {other:?}"),
        }

        // Open fid 1 for reading.
        let req = msg::tlopen(3, 1, 0); // OREAD
        let (_, resp) = t.handle_message(&req).await.unwrap();
        assert!(matches!(resp, Response::Lopen { .. }));

        // Read.
        let req = msg::tread(4, 1, 0, 64);
        let (_, resp) = t.handle_message(&req).await.unwrap();
        match resp {
            Response::Read { data } => assert_eq!(&data, b"hello world"),
            other => panic!("expected Read, got {other:?}"),
        }

        // Clunk fid 1.
        let req = msg::tclunk(5, 1);
        let (_, resp) = t.handle_message(&req).await.unwrap();
        assert!(matches!(resp, Response::Clunk));
    }

    #[tokio::test]
    async fn fid_table_tracks_open_and_clunk() {
        let table = FidTable::default();
        table.insert(7, 0x80);
        assert_eq!(table.qtype(7), Some(0x80));
        table.set_opened(7);
        table.remove(7);
        assert_eq!(table.qtype(7), None);
    }

    #[test]
    fn qid_helpers() {
        let q = sample_qid(0x80, 42);
        assert!(q.is_dir());
    }

    /// A Treaddir over the translator must produce a wire message typed RREADDIR
    /// (41), whose payload decodes as standard 9P2000.L dirent records — the
    /// exact contract a standard client (Wanix `p9kit`) enforces. Regression
    /// guard for the interop fix (previously framed as RREAD=117).
    #[tokio::test]
    async fn readdir_emits_standard_rreaddir_wire() {
        let backend = MemoryBackend::default();
        backend.add_file("/one.txt", b"111");
        backend.add_file("/two.txt", b"22");
        let t = Arc::new(Translator::new(Arc::new(backend)));

        // Attach fid 0 as the (directory) root, then open + readdir it.
        t.handle_message(&msg::tattach(1, 0, u32::MAX, "u", "/")).await.unwrap();
        t.handle_message(&msg::tlopen(2, 0, 0)).await.unwrap();
        let (tag, resp) = t.handle_message(&msg::treaddir(3, 0, 0, 8192)).await.unwrap();

        // Encode to the wire exactly as serve_connection does, and check the
        // message-type byte is RREADDIR, not RREAD.
        let wire = msg::encode_response(tag, &resp);
        assert_eq!(wire[4], msg::RREADDIR, "readdir must be framed as RREADDIR (41)");
        assert_ne!(wire[4], msg::RREAD, "readdir must NOT be framed as RREAD (117)");

        // The payload must decode as standard dirent records.
        let data = match resp {
            Response::Readdir { data } => data,
            other => panic!("expected Readdir, got {other:?}"),
        };
        let entries = msg::parse_readdir_entries(&data).unwrap();
        let mut names: Vec<_> = entries.iter().map(|e| e.name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["one.txt".to_string(), "two.txt".to_string()]);
        // Cookies are 1-based and monotonic.
        assert!(entries.iter().all(|e| e.offset >= 1));
    }
}
