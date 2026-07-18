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
//! the same loop over a Unix domain socket ([`tokio::net::UnixListener`]);
//! [`Translator::serve_vsock`] runs it over a Cloud-Hypervisor **hybrid-vsock**
//! host socket. All three delegate to one transport-agnostic accept loop
//! (`serve_listener`) and the same per-connection
//! [`Translator::serve_connection`] core, which operates on any
//! `AsyncRead + AsyncWrite` stream. The UDS entry point is how a native Wanix
//! workload consumes a Subject-scoped export via its `p9kit.ClientFS` 9P client
//! (#506); the vsock entry point is how a **kata guest** dials the host VFS over
//! native 9P (the CH `ch` path has no virtio-9p, so vsock is the 9P transport
//! into the guest — #730). [`Translator::from_mount`] / [`serve_mount_uds`] /
//! [`serve_mount_vsock`] build the translator directly from an
//! `Arc<dyn Mount>` + `Subject`.
//!
//! ### Hybrid-vsock
//!
//! Cloud Hypervisor exposes guest vsock as a host Unix socket that multiplexes
//! guest ports with a `connect <port>\n` text handshake — the same handshake
//! `hyprstream_workers::runtime::kata_agent::hybrid_vsock_connect` performs in
//! the opposite direction (host→guest agent). [`Translator::serve_vsock`]
//! plays the host multiplexer role: it strips and acknowledges that preamble
//! per connection, then hands the bare stream to the shared 9P core, so
//! hybrid-vsock reuses byte-identical connection handling with TCP and UDS and
//! adds no new dependency. Firecracker/Dragonball raw `AF_VSOCK`
//! (`vsock://<cid>`, mirroring `AgentAddress::Vsock`) is a follow-up: CH — the
//! actual target — uses hybrid-vsock, and the raw-vsock hypervisors are not
//! booted here.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use hyprstream_rpc::Subject;
use hyprstream_vfs::Mount;
use tokio::io::{split, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio::sync::OnceCell;
use tracing::{debug, error, info, warn};

use crate::backend::Backend;
use crate::mac_seam::{Action, ReferenceMonitor, SessionContext};
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
    /// Session derived once from the verified `Tattach` credential and
    /// inherited by every descendant fid. `None` when the translator runs
    /// without a [`ReferenceMonitor`] (the dormant pre-activation default).
    session: Option<SessionContext>,
    /// Absolute path of this walked fid, relative to the export root. The
    /// reference monitor resolves this path through the trusted
    /// manifest/genesis label resolver before every mediated operation.
    path: Vec<String>,
}

/// Server-side fid table: 9P fid → state.
///
/// `DashMap` so the translator can be shared across per-connection tasks.
#[derive(Default)]
pub struct FidTable {
    fids: DashMap<u32, FidState>,
}

impl FidTable {
    /// Insert state for a fid, inheriting the verified attach session (when a
    /// monitor is installed) and carrying the absolute walked path used for
    /// content-label resolution.
    pub fn insert(&self, fid: u32, qtype: u8, session: Option<SessionContext>, path: Vec<String>) {
        self.fids.insert(
            fid,
            FidState {
                qtype,
                opened: false,
                session,
                path,
            },
        );
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

    /// Look up the verified attach session inherited by a fid (`None` when
    /// the fid is unknown or the translator runs without a monitor).
    pub fn session(&self, fid: u32) -> Option<SessionContext> {
        self.fids.get(&fid).and_then(|s| s.session.clone())
    }

    /// Look up a fid's absolute export-relative path for trusted label lookup.
    pub fn path(&self, fid: u32) -> Option<Vec<String>> {
        self.fids.get(&fid).map(|s| s.path.clone())
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
    backend_factory: Arc<dyn Fn() -> Arc<dyn Backend> + Send + Sync>,
    fids: Arc<FidTable>,
    attach_session: OnceCell<SessionContext>,
    /// The S2 reference monitor (#568). `None` is the dormant default: the
    /// translator then performs no MAC enforcement at all, exactly matching
    /// pre-#568 behavior. Activation (installing a monitor on the production
    /// serve paths) is blocked on #698 — see the `mac_seam` module docs.
    monitor: Option<Arc<ReferenceMonitor>>,
}

impl Translator {
    /// Build a translator over the given backend. The fid table starts empty.
    /// No reference monitor is installed: per-op behavior is unchanged from
    /// before #568 until the application opts in via
    /// [`Translator::with_reference_monitor`].
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        let backend_factory_backend = Arc::clone(&backend);
        Self {
            backend,
            backend_factory: Arc::new(move || Arc::clone(&backend_factory_backend)),
            fids: Arc::new(FidTable::default()),
            attach_session: OnceCell::new(),
            monitor: None,
        }
    }

    /// Install the S2 reference monitor: every dispatched op on every fid is
    /// then mediated (attach-time token verification, trusted label
    /// resolution, token gate, independent IFC dominance, then the local
    /// AVC/PDP), failing closed at each step. Without this call the translator
    /// enforces nothing — the dormant pre-activation default.
    pub fn with_reference_monitor(mut self, monitor: Arc<ReferenceMonitor>) -> Self {
        self.monitor = Some(monitor);
        self
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
        let backend: Arc<dyn Backend> =
            Arc::new(MountBackend::new(Arc::clone(&mount), subject.clone()));
        Self {
            backend,
            backend_factory: Arc::new(move || {
                Arc::new(MountBackend::new(Arc::clone(&mount), subject.clone()))
                    as Arc<dyn Backend>
            }),
            fids: Arc::new(FidTable::default()),
            attach_session: OnceCell::new(),
            monitor: None,
        }
    }

    /// Build a translator that exports `mount`, resolving the session
    /// [`Subject`] from the mount ticket the client presents in `Tattach.uname`
    /// via `authorizer` (H1b `/9p` WebTransport plane, #765).
    ///
    /// Unlike [`from_mount`](Self::from_mount) — where the caller identity is
    /// fixed at construction (a UDS/vsock listener serves exactly one tenant) —
    /// the WebTransport plane serves a cert-pinned mesh session over which a
    /// tenant proves identity at attach. The `Subject` is bound once, at the
    /// first successful `Tattach`, and threaded onto every op by
    /// [`MountBackend`] thereafter; a denied ticket fails the attach with an
    /// `Rlerror` (see [`MountBackend::with_authorizer`]). Every other layer —
    /// the serve loop, fid table, errno mapping — is identical to `from_mount`.
    pub fn from_mount_authorized(
        mount: Arc<dyn Mount>,
        authorizer: Arc<dyn crate::mount_backend::AttachAuthorizer>,
    ) -> Self {
        let backend: Arc<dyn Backend> = Arc::new(MountBackend::with_authorizer(
            Arc::clone(&mount),
            Arc::clone(&authorizer),
        ));
        Self {
            backend,
            backend_factory: Arc::new(move || {
                Arc::new(MountBackend::with_authorizer(
                    Arc::clone(&mount),
                    Arc::clone(&authorizer),
                )) as Arc<dyn Backend>
            }),
            fids: Arc::new(FidTable::default()),
            attach_session: OnceCell::new(),
            monitor: None,
        }
    }

    /// Build a fresh per-connection translator state over the same wiring.
    ///
    /// The accept loop shares only immutable policy/auth seams. Fid state,
    /// attach context, and stateful mount backends are connection-local so one
    /// client cannot reuse another client's fid numbers or attach subject.
    fn connection_scoped(&self) -> Self {
        Self {
            backend: (self.backend_factory)(),
            backend_factory: Arc::clone(&self.backend_factory),
            fids: Arc::new(FidTable::default()),
            attach_session: OnceCell::new(),
            monitor: self.monitor.clone(),
        }
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

    /// Run the accept loop over a Cloud-Hypervisor **hybrid-vsock** host socket.
    ///
    /// The vsock sibling of [`serve_uds`](Self::serve_uds), and the entry point
    /// a kata guest dials to operate the host VFS over native 9P (#730). CH's
    /// `ch` path has no virtio-9p, so the guest reaches the host by opening a
    /// vsock connection; CH surfaces that as a host Unix socket whose accepted
    /// connections each begin with a `connect <port>\n` text preamble (the
    /// firecracker/CH multiplexer handshake — the reverse of
    /// `kata_agent::hybrid_vsock_connect`). This entry strips and acknowledges
    /// that preamble per connection, then runs the identical
    /// [`serve_connection`](Self::serve_connection) core as
    /// [`serve`](Self::serve) / [`serve_uds`](Self::serve_uds); the
    /// `Subject`-per-op enforcement is unchanged — only the transport differs.
    pub async fn serve_vsock(self, listener: UnixListener) -> Result<()> {
        self.serve_listener(HybridVsockListener { inner: listener }).await
    }

    /// Run the accept loop over a **raw** (no-handshake) hybrid-vsock host
    /// socket: a per-port host UDS the **guest dials**.
    ///
    /// The raw sibling of [`serve_vsock`](Self::serve_vsock). Where
    /// [`serve_vsock`] plays the CH host-multiplexer role for connections
    /// *hyprstream* initiates — and therefore strips a `connect <port>\n` text
    /// preamble off each accepted stream — this entry point serves a **per-port
    /// host listener** (`<vsock-base>_<port>`, e.g. `VFS_9P_VSOCK_PORT=564`) that
    /// the guest connects to. Per the Firecracker/CH hybrid-vsock spec,
    /// guest-initiated connections to a per-port host UDS arrive **raw**: there is
    /// no `connect <port>\n` preamble (the port is already encoded in the socket
    /// path, not in an in-band handshake). Stripping a preamble here would eat the
    /// guest's first 9P `Tversion` bytes and break the handshake (#741).
    ///
    /// So this hands each accepted [`UnixStream`] straight to the shared
    /// [`serve_connection`](Self::serve_connection) core with **no** preamble
    /// strip. The `Subject`-per-op enforcement and wire-faithful 9P2000.L framing
    /// are byte-identical to every other transport; only the accept surface
    /// differs (raw, via [`RawVsockListener`]).
    pub async fn serve_vsock_raw(self, listener: UnixListener) -> Result<()> {
        self.serve_listener(RawVsockListener { inner: listener }).await
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
            let this = Arc::new(this.connection_scoped());
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
                    // Map a typed backend error to a real 9P2000.L `Rlerror`
                    // errno (H1a / #764). `wanix serve` clients and shell UX
                    // depend on distinguishing ENOENT/EACCES/ENOTDIR from a
                    // blanket EIO; [`errno_from_error`] walks the anyhow cause
                    // chain for a [`hyprstream_vfs::MountError`] and maps it,
                    // falling back to EIO for untyped errors.
                    let err_tag = tag_or_notag(&buf);
                    let ecode = errno_from_error(&e);
                    warn!(error = %e, tag = err_tag, ecode, "9P handler error");
                    (err_tag, Response::Error { ecode })
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
            Request::Attach { fid, uname, aname, .. } => {
                self.handle_attach(fid, &uname, &aname).await?
            }
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

    async fn handle_attach(&self, fid: u32, uname: &str, aname: &str) -> Result<Response> {
        // Establish the session first: on the attach-time ticket path (H1b) this
        // validates `uname` and binds the session Subject; on the fixed-subject
        // path it is a no-op. A denied ticket returns here and is mapped to an
        // Rlerror by the serve loop — before any fid or mount handle exists.
        self.backend.attach(uname).await?;

        // #568: MAC mediation happens only when a reference monitor is
        // installed (never in production today — activation is blocked on
        // #698). The credential is then verified exactly once at attach, and
        // the derived session is cached on every fid walked from this one.
        let session = match &self.monitor {
            Some(monitor) => {
                let session = monitor.authenticate(uname, aname).await;
                self.bind_attach_session(&session)?;
                // The root fid itself is a mediated object. It has no
                // synthetic exemption: an unlabeled root or missing/expired
                // token denies at attach, before any backend object handle
                // is exposed.
                if !monitor.authorize(&session, &[], Action::Walk) {
                    return Ok(Response::Error { ecode: libc_eperm() });
                }
                Some(session)
            }
            None => None,
        };

        // Attach establishes the root fid. Walk the empty path to materialize
        // a root qid from the backend, then record it.
        let walk = self.backend.walk(fid, fid, &[]).await?;
        let qtype = walk.qids.last().map(|q| q.qtype).unwrap_or(0);
        self.fids.insert(fid, qtype, session, Vec::new());
        let qid = walk.qids.into_iter().next().unwrap_or_default();
        Ok(Response::Attach { qid })
    }

    fn bind_attach_session(&self, session: &SessionContext) -> Result<()> {
        if self.attach_session.set(session.clone()).is_ok() {
            return Ok(());
        }
        match self.attach_session.get() {
            Some(existing) if existing == session => Ok(()),
            Some(_) => Err(anyhow::Error::new(hyprstream_vfs::MountError::PermissionDenied(
                "conflicting attach session context".to_owned(),
            ))),
            None => Err(anyhow::anyhow!("attach session cell rejected without value")),
        }
    }

    async fn handle_walk(
        &self,
        fid: u32,
        newfid: u32,
        wnames: Vec<String>,
    ) -> Result<Response> {
        // The session is inherited from the fid being walked (#568 — the
        // credential is verified once at attach, never re-verified per op),
        // and the destination path extends the source fid's walked path.
        let session = self.fids.session(fid);
        let mut path = self.fids.path(fid).unwrap_or_default();
        path.extend(wnames.iter().cloned());

        if let Some(monitor) = &self.monitor {
            // A walk is checked against its destination, not merely its
            // source; otherwise a subject could traverse into a high-labeled
            // object and rely on a later operation to discover the denial.
            // A fid outside a verified session denies outright.
            let Some(session) = session.as_ref() else {
                return Ok(Response::Error { ecode: libc_eperm() });
            };
            if !monitor.authorize(session, &path, Action::Walk) {
                return Ok(Response::Error { ecode: libc_eperm() });
            }
        }

        // Mirror of RemoteModelMount::walk: call backend.walk with the source
        // fid, the new fid, and the path components.
        let result = self.backend.walk(fid, newfid, &wnames).await?;
        if wnames.is_empty() {
            // nwname=0 is a fid *clone*: newfid aliases fid (same file), and
            // the backend returns no qids. newfid must still inherit the source
            // fid's qtype, session, and path — otherwise the clone has no
            // cached session and a live monitor wrongly denies it for an
            // already-authenticated client (#568).
            let qtype = self.fids.qtype(fid).unwrap_or(0);
            self.fids.insert(newfid, qtype, session, path);
        } else if let Some(q) = result.qids.last() {
            self.fids.insert(newfid, q.qtype, session, path);
        }
        Ok(Response::Walk { qids: result.qids })
    }

    async fn handle_lopen(&self, fid: u32, flags: u32) -> Result<Response> {
        if let Some(err) = self.deny(fid, Action::Open) {
            return Ok(err);
        }
        let open = self.backend.open(fid, flags).await?;
        self.fids.set_opened(fid);
        Ok(Response::Lopen { qid: open.qid, iounit: open.iounit })
    }

    async fn handle_read(&self, fid: u32, offset: u64, count: u32) -> Result<Response> {
        if let Some(err) = self.deny(fid, Action::Read) {
            return Ok(err);
        }
        let data = self.backend.read(fid, offset, count).await?;
        Ok(Response::Read { data })
    }

    async fn handle_write(&self, fid: u32, offset: u64, data: &[u8]) -> Result<Response> {
        if let Some(err) = self.deny(fid, Action::Write) {
            return Ok(err);
        }
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
        // getattr leaks object metadata (qid/mode/size/mtime), so it is gated
        // like every other fid op — it is not a metadata-only exemption.
        if let Some(err) = self.deny(fid, Action::Getattr) {
            return Ok(err);
        }
        let st = self.backend.stat(fid).await?;
        Ok(Response::Getattr { qid: st.qid, mode: st.mode, size: st.size, mtime_sec: st.mtime_sec })
    }

    async fn handle_readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Response> {
        if let Some(err) = self.deny(fid, Action::Readdir) {
            return Ok(err);
        }
        let data = self.backend.readdir(fid, offset, count).await?;
        Ok(Response::Readdir { data })
    }

    /// Mediate one op on `fid` through the installed [`ReferenceMonitor`].
    /// Returns `Some(Response::Error)` (EPERM) when the op is denied, `None`
    /// when it may proceed to the backend.
    ///
    /// With no monitor installed (the dormant pre-activation default) every
    /// op proceeds — exactly the pre-#568 behavior. With a monitor, a fid
    /// outside a verified attach session (unknown fid, or a fid created
    /// before the monitor saw an attach) fails closed, and the session's
    /// cached path is mediated as `label → token → IFC → decider`.
    fn deny(&self, fid: u32, action: Action) -> Option<Response> {
        let monitor = self.monitor.as_ref()?;
        let (Some(session), Some(path)) = (self.fids.session(fid), self.fids.path(fid)) else {
            return Some(Response::Error { ecode: libc_eperm() });
        };
        if monitor.authorize(&session, &path, action) {
            None
        } else {
            Some(Response::Error { ecode: libc_eperm() })
        }
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

/// Bind a Cloud-Hypervisor **hybrid-vsock** host socket at `path` and serve
/// `mount` (scoped to `subject`) as its 9P2000.L root until the socket errors
/// or is closed.
///
/// The hybrid-vsock sibling of [`serve_mount_uds`] and the #730 one-call entry
/// a supervisor uses to expose a tenant's Subject-scoped export to a **kata
/// guest** dialing over vsock. Guest connections arrive on the host UDS `path`,
/// each prefixed with a `connect <port>\n` handshake that
/// [`Translator::serve_vsock`] strips before handing the stream to the shared
/// 9P core.
pub async fn serve_mount_vsock(
    mount: Arc<dyn Mount>,
    subject: Subject,
    path: impl AsRef<Path>,
) -> Result<()> {
    let listener = UnixListener::bind(path.as_ref())
        .with_context(|| format!("bind 9P hybrid-vsock listener at {:?}", path.as_ref()))?;
    Translator::from_mount(mount, subject).serve_vsock(listener).await
}

/// Bind a **raw** (no-handshake) hybrid-vsock **per-port** host socket at `path`
/// and serve `mount` (scoped to `subject`) as its 9P2000.L root until the socket
/// errors or is closed.
///
/// The raw sibling of [`serve_mount_vsock`], for the **guest-initiated**
/// direction (#741). `path` is a per-port host UDS (`<vsock-base>_<port>`) the
/// **guest dials**; per the Firecracker/CH hybrid-vsock spec such connections
/// arrive **raw** (no `connect <port>\n` preamble — the port is encoded in the
/// socket path). Accepted streams go straight to the shared 9P core via
/// [`Translator::serve_vsock_raw`] with no preamble strip, so the guest's first
/// `Tversion` bytes are treated as 9P and not consumed. This is the entry a kata
/// supervisor uses to expose a tenant's Subject-scoped export on
/// `VFS_9P_VSOCK_PORT`.
pub async fn serve_mount_vsock_raw(
    mount: Arc<dyn Mount>,
    subject: Subject,
    path: impl AsRef<Path>,
) -> Result<()> {
    let listener = UnixListener::bind(path.as_ref()).with_context(|| {
        format!("bind 9P raw hybrid-vsock per-port listener at {:?}", path.as_ref())
    })?;
    Translator::from_mount(mount, subject).serve_vsock_raw(listener).await
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

/// Maximum length (excluding the terminating newline) of the hybrid-vsock
/// `connect <port>` preamble line. The real handshake is well under 32 bytes;
/// the cap bounds a peer that never sends a newline.
const HYBRID_VSOCK_PREAMBLE_MAX: usize = 64;

/// A Cloud-Hypervisor **hybrid-vsock** host listener: a [`UnixListener`] whose
/// accepted connections each begin with a `connect <port>\n` text preamble (the
/// firecracker/CH multiplexer handshake). [`accept_conn`](Listen::accept_conn)
/// strips and acknowledges that preamble, then yields the bare [`UnixStream`]
/// to the shared 9P serve path — so hybrid-vsock reuses byte-identical
/// connection handling with TCP and UDS (`serve_listener` /
/// [`serve_connection`](Translator::serve_connection)) and adds no new
/// dependency.
struct HybridVsockListener {
    inner: UnixListener,
}

#[async_trait]
impl Listen for HybridVsockListener {
    type Conn = UnixStream;

    async fn accept_conn(&self) -> std::io::Result<(UnixStream, String)> {
        // A malformed/aborted handshake affects only that one peer: retry the
        // accept rather than propagating (which `serve_listener` treats as a
        // fatal listener error that tears down every sibling connection).
        loop {
            let (mut stream, _peer) = self.inner.accept().await?;
            match hybrid_vsock_handshake(&mut stream).await {
                Ok(port) => return Ok((stream, format!("hvsock:port{port}"))),
                Err(e) => {
                    warn!(error = %e, "hybrid-vsock handshake failed; dropping connection");
                    continue;
                }
            }
        }
    }

    fn describe(&self) -> String {
        self.inner
            .local_addr()
            .map(|a| format!("hvsock:{a:?}"))
            .unwrap_or_else(|_| "hvsock:?".to_owned())
    }
}

/// A **raw** (no-handshake) hybrid-vsock **per-port** host listener: a
/// [`UnixListener`] bound at a per-port host UDS (`<vsock-base>_<port>`) that the
/// **guest dials**. Unlike [`HybridVsockListener`], accepted connections carry
/// **no** `connect <port>\n` preamble — per the Firecracker/CH hybrid-vsock spec
/// the port is encoded in the socket path, and guest-initiated connections to a
/// per-port host UDS arrive raw (#741). So [`accept_conn`](Listen::accept_conn)
/// yields the accepted [`UnixStream`] verbatim to the shared 9P serve path; the
/// guest's first bytes are the 9P `Tversion`, never a preamble.
///
/// This is a thin, self-documenting wrapper (rather than serving the bare
/// [`UnixListener`]) so the "listening" log line and peer strings identify the
/// channel as raw hybrid-vsock; the byte handling is identical to the plain UDS
/// impl.
struct RawVsockListener {
    inner: UnixListener,
}

#[async_trait]
impl Listen for RawVsockListener {
    type Conn = UnixStream;

    async fn accept_conn(&self) -> std::io::Result<(UnixStream, String)> {
        // No preamble: hand the stream straight through. The first bytes the
        // guest sends are the 9P Tversion.
        let (stream, peer) = self.inner.accept().await?;
        Ok((stream, format!("rawvsock:{peer:?}")))
    }

    fn describe(&self) -> String {
        self.inner
            .local_addr()
            .map(|a| format!("rawvsock:{a:?}"))
            .unwrap_or_else(|_| "rawvsock:?".to_owned())
    }
}

/// Read, validate, and acknowledge the hybrid-vsock `connect <port>\n` preamble
/// on a freshly accepted host-UDS stream, returning the requested vsock port.
///
/// The preamble is read one byte at a time up to the terminating newline so no
/// following 9P payload is swallowed into a throwaway buffer. The acknowledgement
/// line contains `OK`, matching the firecracker/CH multiplexer and the reader in
/// `kata_agent::hybrid_vsock_connect` (which only requires the response to
/// contain `"OK"`).
async fn hybrid_vsock_handshake(stream: &mut UnixStream) -> Result<u32> {
    let mut line = Vec::with_capacity(HYBRID_VSOCK_PREAMBLE_MAX);
    let mut byte = [0u8; 1];
    loop {
        let n = stream
            .read(&mut byte)
            .await
            .context("read hybrid-vsock preamble")?;
        if n == 0 {
            anyhow::bail!("hybrid-vsock peer closed before sending `connect <port>` preamble");
        }
        if byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
        if line.len() > HYBRID_VSOCK_PREAMBLE_MAX {
            anyhow::bail!(
                "hybrid-vsock preamble exceeds {HYBRID_VSOCK_PREAMBLE_MAX} bytes without a newline"
            );
        }
    }

    let text = String::from_utf8_lossy(&line);
    let mut parts = text.split_whitespace();
    let verb = parts.next().unwrap_or_default();
    if !verb.eq_ignore_ascii_case("connect") {
        anyhow::bail!("hybrid-vsock preamble: expected `connect <port>`, got {text:?}");
    }
    let port: u32 = parts
        .next()
        .and_then(|p| p.parse().ok())
        .with_context(|| format!("hybrid-vsock preamble: missing/invalid port in {text:?}"))?;

    stream
        .write_all(format!("OK {port}\n").as_bytes())
        .await
        .context("write hybrid-vsock OK acknowledgement")?;
    Ok(port)
}

/// Extract the tag from a raw message for error-path Rlerror encoding.
fn tag_or_notag(buf: &[u8]) -> u16 {
    if buf.len() >= 6 {
        u16::from_le_bytes([buf[5], buf[6]])
    } else {
        0xFFFF // NOTAG
    }
}

// errno-style codes the translator emits. We hardcode the integer values
// (Linux asm-generic errno numbers) rather than depend on libc so the crate
// stays wasm-friendly and the Rlerror wire codes are stable across platforms.
const ENOENT: u32 = 2; // No such file or directory
const EIO: u32 = 5; // I/O error
const EACCES: u32 = 13; // Permission denied
const EEXIST: u32 = 17; // File exists
const ENOTDIR: u32 = 20; // Not a directory
const EISDIR: u32 = 21; // Is a directory
const EINVAL: u32 = 22; // Invalid argument
const ENOSYS: u32 = 38; // Function not implemented

const fn libc_einval() -> u32 {
    EINVAL
}

/// Map a backend handler error to a 9P2000.L `Rlerror` errno (H1a / #764).
///
/// The backend seam is stringly-typed at the `anyhow::Error` boundary, but the
/// VFS [`MountBackend`] preserves the underlying [`hyprstream_vfs::MountError`]
/// as an anyhow *source* (via `.context(...)`, not string interpolation), so
/// the concrete error survives in the cause chain. This walks that chain and
/// maps the first [`MountError`] it finds to its POSIX errno; an untyped error
/// (e.g. a fid-table bookkeeping failure) falls back to [`EIO`].
///
/// Mapping (min ENOENT/EACCES/ENOTDIR per the spike, plus the rest of the
/// [`MountError`] surface):
/// `NotFound→ENOENT`, `PermissionDenied→EACCES`, `NotDirectory→ENOTDIR`,
/// `IsDirectory→EISDIR`, `InvalidArgument→EINVAL`, `AlreadyExists→EEXIST`,
/// `NotSupported→ENOSYS`, `Io→EIO`.
pub(crate) fn errno_from_error(err: &anyhow::Error) -> u32 {
    for cause in err.chain() {
        if let Some(me) = cause.downcast_ref::<hyprstream_vfs::MountError>() {
            return errno_from_mount_error(me);
        }
    }
    EIO
}

/// Map a [`hyprstream_vfs::MountError`] variant to its POSIX errno.
fn errno_from_mount_error(e: &hyprstream_vfs::MountError) -> u32 {
    use hyprstream_vfs::MountError as M;
    match e {
        M::NotFound(_) => ENOENT,
        M::PermissionDenied(_) => EACCES,
        M::NotDirectory(_) => ENOTDIR,
        M::IsDirectory(_) => EISDIR,
        M::InvalidArgument(_) => EINVAL,
        M::AlreadyExists(_) => EEXIST,
        M::NotSupported(_) => ENOSYS,
        M::Io(_) => EIO,
    }
}
const fn libc_eperm() -> u32 {
    1 // EPERM
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::MemoryBackend;
    use crate::msg::Qid;

    // ── Typed errno mapping (H1a / #764) ────────────────────────────────

    /// Every [`hyprstream_vfs::MountError`] variant must map to its POSIX
    /// errno when wrapped exactly as [`MountBackend`] wraps it (via
    /// `.context(...)`, which preserves the concrete error as an anyhow
    /// source). An untyped error falls back to EIO. Regression guard for the
    /// blanket-EIO behaviour the spike flagged at `translator.rs:211`.
    #[test]
    fn mount_errors_map_to_errnos() {
        use hyprstream_vfs::MountError as M;
        let cases = [
            (M::NotFound("x".into()), ENOENT),
            (M::PermissionDenied("x".into()), EACCES),
            (M::NotDirectory("x".into()), ENOTDIR),
            (M::IsDirectory("x".into()), EISDIR),
            (M::InvalidArgument("x".into()), EINVAL),
            (M::AlreadyExists("x".into()), EEXIST),
            (M::NotSupported("x".into()), ENOSYS),
            (M::Io("x".into()), EIO),
        ];
        for (err, expected) in cases {
            // Mimic the MountBackend error path: attach context so the
            // MountError is an anyhow *source*, not string-flattened.
            let wrapped = anyhow::Error::new(err).context("mount op failed");
            assert_eq!(errno_from_error(&wrapped), expected);
        }
        // Untyped bookkeeping errors fall back to EIO.
        assert_eq!(errno_from_error(&anyhow::anyhow!("fid 3 not walked")), EIO);
    }

    /// End-to-end through [`MountBackend`]: a `Mount` that returns
    /// `NotFound` on a bad walk must surface as an anyhow chain the
    /// translator maps to `ENOENT` (not EIO). Proves the `.context(...)`
    /// wrapping in `MountBackend` keeps the `MountError` downcastable.
    #[tokio::test]
    async fn mount_backend_preserves_typed_errno_end_to_end() {
        use crate::mount_backend::MountBackend;
        use async_trait::async_trait;
        use hyprstream_rpc::Subject;
        use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};

        struct StubMount;
        #[async_trait]
        impl Mount for StubMount {
            async fn walk(&self, components: &[&str], _c: &Subject) -> Result<Fid, MountError> {
                if components.is_empty() {
                    // Attach walks the empty root path — resolve it.
                    Ok(Fid::new(()))
                } else {
                    Err(MountError::NotFound(components.join("/")))
                }
            }
            async fn open(&self, _f: &mut Fid, _m: u8, _c: &Subject) -> Result<(), MountError> {
                Err(MountError::PermissionDenied("open".into()))
            }
            async fn read(&self, _f: &Fid, _o: u64, _n: u32, _c: &Subject) -> Result<Vec<u8>, MountError> {
                Err(MountError::Io("read".into()))
            }
            async fn write(&self, _f: &Fid, _o: u64, _d: &[u8], _c: &Subject) -> Result<u32, MountError> {
                Err(MountError::NotSupported("write".into()))
            }
            async fn readdir(&self, _f: &Fid, _c: &Subject) -> Result<Vec<DirEntry>, MountError> {
                Err(MountError::NotDirectory("readdir".into()))
            }
            async fn stat(&self, _f: &Fid, _c: &Subject) -> Result<Stat, MountError> {
                Ok(Stat::unknown_qid(0x80, 0, "/".into(), 0))
            }
            async fn clunk(&self, _f: Fid, _c: &Subject) {}
        }

        let backend = MountBackend::new(Arc::new(StubMount), Subject::new("tenant"));
        let t = Arc::new(Translator::new(Arc::new(backend)));

        // Attach fid 0 as root (empty walk succeeds).
        t.handle_message(&msg::tattach(1, 0, u32::MAX, "u", "")).await.unwrap();

        // Walk to a missing child → Err whose errno must be ENOENT.
        let err = t
            .handle_message(&msg::twalk(2, 0, 1, &["nope"]))
            .await
            .unwrap_err();
        assert_eq!(errno_from_error(&err), ENOENT, "missing path must be ENOENT, not EIO");
    }

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

    // ── #568 reference-monitor test doubles ───────────────────────────
    //
    // The monitor seams are exercised with fakes that stand in for the
    // application's verified-token authenticator, genesis/manifest label
    // resolver, and AVC/PDP decider. No production path installs a monitor
    // yet (activation is blocked on #698); the unmonitored tests in this
    // module double as the dormant-default regression guard.

    use crate::mac_seam::{
        AccessDecider, AttachAuthenticator, ObjectLabelResolver, ObjectRef, VerifiedTokenScope,
    };
    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityContext, SecurityLabel, VerifiedKeyMaterial,
    };
    use std::time::{Duration, Instant};

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn ctx(level: Level) -> SecurityContext {
        // Classical key material so the derived assurance dominates the
        // Classical-assurance object labels used across these tests.
        SecurityContext::new(level, CompartmentSet::EMPTY, VerifiedKeyMaterial::Classical)
    }

    /// A session at `level` whose verified token permits exactly `ops` (with
    /// a Secret ceiling) for the next hour.
    fn permit_session(level: Level, ops: &[Action]) -> SessionContext {
        SessionContext::from_verified_token(
            ctx(level),
            VerifiedTokenScope::from_verified_token(
                label(Level::Secret),
                Arc::from(ops),
                Instant::now() + Duration::from_secs(3600),
            ),
        )
    }

    const ALL_OPS: &[Action] = &[
        Action::Walk,
        Action::Open,
        Action::Read,
        Action::Write,
        Action::Getattr,
        Action::Readdir,
    ];

    /// Authenticator returning a fixed session (the token is treated as
    /// already verified, as a real authenticator would after checking it).
    struct StaticAuth(SessionContext);
    #[async_trait]
    impl AttachAuthenticator for StaticAuth {
        async fn authenticate(&self, _u: &str, _a: &str) -> SessionContext {
            self.0.clone()
        }
    }

    /// Resolver returning a fixed label for every path (`None` = unlabeled).
    struct StaticLabels(Option<SecurityLabel>);
    impl ObjectLabelResolver for StaticLabels {
        fn resolve(&self, _object: ObjectRef<'_>) -> Option<SecurityLabel> {
            self.0
        }
    }

    struct AllowAll;
    impl AccessDecider for AllowAll {
        fn check(
            &self,
            _ctx: &SecurityContext,
            _object_label: &SecurityLabel,
            _action: Action,
        ) -> bool {
            true
        }
    }

    fn monitor(
        auth: SessionContext,
        resolved: Option<SecurityLabel>,
        decider: Arc<dyn AccessDecider>,
    ) -> Arc<ReferenceMonitor> {
        Arc::new(ReferenceMonitor::new(
            Arc::new(StaticAuth(auth)),
            Arc::new(StaticLabels(resolved)),
            decider,
        ))
    }

    /// With a monitor installed, an unlabeled root denies the attach itself:
    /// the root fid is a mediated object with no synthetic exemption.
    #[tokio::test]
    async fn monitor_denies_unlabeled_root_at_attach() {
        let t = Translator::new(Arc::new(MemoryBackend::default())).with_reference_monitor(
            monitor(permit_session(Level::Secret, ALL_OPS), None, Arc::new(AllowAll)),
        );
        let (_, resp) = t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM for unlabeled root, got {other:?}"),
        }
    }

    /// A session without a verified token denies at attach, before any
    /// backend object handle exists.
    #[tokio::test]
    async fn monitor_denies_missing_token_at_attach() {
        let t = Translator::new(Arc::new(MemoryBackend::default())).with_reference_monitor(
            monitor(SessionContext::deny(), Some(label(Level::Public)), Arc::new(AllowAll)),
        );
        let (_, resp) = t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM for missing token, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn monitor_denies_expired_token_at_attach() {
        let expired = SessionContext::from_verified_token(
            ctx(Level::Secret),
            VerifiedTokenScope::from_verified_token(
                label(Level::Secret),
                Arc::from(ALL_OPS),
                Instant::now() - Duration::from_secs(1),
            ),
        );
        let t = Translator::new(Arc::new(MemoryBackend::default())).with_reference_monitor(
            monitor(expired, Some(label(Level::Public)), Arc::new(AllowAll)),
        );
        let (_, resp) = t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM for expired token, got {other:?}"),
        }
    }

    /// When every gate passes — labeled object, unexpired token covering the
    /// op, dominating subject, permitting decider — the mediated session is
    /// served end to end.
    #[tokio::test]
    async fn monitor_allows_when_all_gates_pass() {
        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello world");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(monitor(
            permit_session(Level::Secret, ALL_OPS),
            Some(label(Level::Public)),
            Arc::new(AllowAll),
        ));

        let (_, resp) = t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "got {resp:?}");
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "got {resp:?}");
        let (_, resp) = t.handle_message(&msg::tlopen(3, 1, 0)).await.unwrap();
        assert!(matches!(resp, Response::Lopen { .. }), "got {resp:?}");
        let (_, resp) = t.handle_message(&msg::tread(4, 1, 0, 64)).await.unwrap();
        match resp {
            Response::Read { data } => assert_eq!(&data, b"hello world"),
            other => panic!("expected Read, got {other:?}"),
        }
    }

    /// #568: a translator with a live monitor actually blocks ops the decider
    /// rejects, proving the `deny` wiring is live — not just present but
    /// inert. Walk/open (permitted) still succeed; read is rejected with
    /// `Rlerror EPERM` rather than reaching the backend.
    #[tokio::test]
    async fn injected_decider_blocks_denied_action() {
        struct DenyReads;
        impl AccessDecider for DenyReads {
            fn check(
                &self,
                _ctx: &SecurityContext,
                _object_label: &SecurityLabel,
                action: Action,
            ) -> bool {
                !matches!(action, Action::Read)
            }
        }

        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello world");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(monitor(
            permit_session(Level::Secret, ALL_OPS),
            Some(label(Level::Public)),
            Arc::new(DenyReads),
        ));

        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "walk must still succeed");

        let (_, resp) = t.handle_message(&msg::tlopen(3, 1, 0)).await.unwrap();
        assert!(matches!(resp, Response::Lopen { .. }), "open must still succeed");

        let (_, resp) = t.handle_message(&msg::tread(4, 1, 0, 64)).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM, got {other:?}"),
        }
    }

    /// An op outside the verified token's operation set denies even when the
    /// decider would permit it — the token gate runs before policy.
    #[tokio::test]
    async fn monitor_denies_op_outside_token_scope() {
        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello world");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(monitor(
            permit_session(Level::Secret, &[Action::Walk, Action::Open, Action::Read]),
            Some(label(Level::Public)),
            Arc::new(AllowAll),
        ));

        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        t.handle_message(&msg::tlopen(3, 1, 1 /* OWRITE */)).await.unwrap();
        let (_, resp) = t.handle_message(&msg::twrite(4, 1, 0, b"nope")).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM for out-of-scope write, got {other:?}"),
        }
        let (_, resp) = t.handle_message(&msg::tread(5, 1, 0, 64)).await.unwrap();
        assert!(matches!(resp, Response::Read { .. }), "in-scope read must pass: {resp:?}");
    }

    /// A walk is mediated against its destination label: the token ceiling
    /// (Confidential) covers the public file but not the Secret one, so the
    /// second walk denies even though the source fid was authorized.
    #[tokio::test]
    async fn monitor_walk_checked_against_destination_label() {
        struct ByName;
        impl ObjectLabelResolver for ByName {
            fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
                match object {
                    ObjectRef::Path(parts) => Some(match parts.last().copied() {
                        Some("secret.txt") => label(Level::Secret),
                        _ => label(Level::Public),
                    }),
                    ObjectRef::Cid(_) => None,
                }
            }
        }
        let session = SessionContext::from_verified_token(
            ctx(Level::Secret),
            VerifiedTokenScope::from_verified_token(
                label(Level::Confidential),
                Arc::from(ALL_OPS),
                Instant::now() + Duration::from_secs(3600),
            ),
        );
        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hi");
        backend.add_file("/secret.txt", b"classified");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(Arc::new(
            ReferenceMonitor::new(
                Arc::new(StaticAuth(session)),
                Arc::new(ByName),
                Arc::new(AllowAll),
            ),
        ));

        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "public walk must pass: {resp:?}");
        let (_, resp) = t.handle_message(&msg::twalk(3, 0, 2, &["secret.txt"])).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM above the token ceiling, got {other:?}"),
        }
    }

    /// The IFC floor is independent of the token and the policy matrix: a
    /// Public subject holding a Secret-ceiling token with an allow-all
    /// decider still cannot walk into a Secret-labeled object.
    #[tokio::test]
    async fn monitor_ifc_floor_not_bypassed_by_permissive_decider() {
        struct AllSecret;
        impl ObjectLabelResolver for AllSecret {
            fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
                match object {
                    ObjectRef::Path([]) => Some(label(Level::Public)),
                    ObjectRef::Path(_) => Some(label(Level::Secret)),
                    ObjectRef::Cid(_) => None,
                }
            }
        }
        let backend = MemoryBackend::default();
        backend.add_file("/secret.txt", b"classified");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(Arc::new(
            ReferenceMonitor::new(
                Arc::new(StaticAuth(permit_session(Level::Public, ALL_OPS))),
                Arc::new(AllSecret),
                Arc::new(AllowAll),
            ),
        ));

        let (_, resp) = t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "public root must attach: {resp:?}");
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &["secret.txt"])).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM for IFC non-dominance, got {other:?}"),
        }
    }

    /// Clunk is fid disposal, not an object operation: it is never mediated,
    /// so a denied op can never wedge connection teardown.
    #[tokio::test]
    async fn clunk_is_not_mediated() {
        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hi");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(monitor(
            permit_session(Level::Secret, &[Action::Walk]),
            Some(label(Level::Public)),
            Arc::new(AllowAll),
        ));
        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        let (_, resp) = t.handle_message(&msg::tclunk(3, 1)).await.unwrap();
        assert!(matches!(resp, Response::Clunk), "clunk must not be mediated: {resp:?}");
    }

    /// #767 review: `getattr` leaks object metadata (qid/mode/size/mtime) and
    /// must be gated by the monitor like every other fid op — it was the one
    /// ungated op. A decider that denies `Action::Getattr` must produce EPERM.
    #[tokio::test]
    async fn injected_decider_blocks_getattr() {
        struct DenyGetattr;
        impl AccessDecider for DenyGetattr {
            fn check(
                &self,
                _ctx: &SecurityContext,
                _object_label: &SecurityLabel,
                action: Action,
            ) -> bool {
                !matches!(action, Action::Getattr)
            }
        }

        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hi");
        let t = Translator::new(Arc::new(backend)).with_reference_monitor(monitor(
            permit_session(Level::Secret, ALL_OPS),
            Some(label(Level::Public)),
            Arc::new(DenyGetattr),
        ));

        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "walk must still succeed");

        let (_, resp) = t.handle_message(&msg::tgetattr(3, 1, u64::MAX)).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM for gated getattr, got {other:?}"),
        }
    }

    /// #767 review: a zero-element walk (nwname=0) is a fid *clone*; the new
    /// fid must inherit the source fid's cached session rather than being left
    /// untracked (which would fail closed under a live monitor and wrongly
    /// deny an already-authenticated client).
    #[tokio::test]
    async fn walk_clone_inherits_source_context() {
        let t = Translator::new(Arc::new(MemoryBackend::default())).with_reference_monitor(
            monitor(
                permit_session(Level::Secret, ALL_OPS),
                Some(label(Level::Public)),
                Arc::new(AllowAll),
            ),
        );

        // Attach root fid 0 with the elevated (non-anonymous) session.
        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        // Clone fid 0 -> fid 1 via a zero-element walk.
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &[])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "clone walk must succeed");

        // The clone must carry the source's elevated session, not floor out.
        let session = t
            .fids
            .session(1)
            .expect("cloned fid must be tracked in the fid table");
        assert_eq!(
            session.security_context().level(),
            Level::Secret,
            "cloned fid must inherit the source's elevated session",
        );
    }

    #[tokio::test]
    async fn connection_scoped_translators_do_not_share_fid_context() {
        let root = Translator::new(Arc::new(MemoryBackend::default())).with_reference_monitor(
            monitor(
                permit_session(Level::Secret, ALL_OPS),
                Some(label(Level::Public)),
                Arc::new(AllowAll),
            ),
        );
        let conn_a = Arc::new(root.connection_scoped());
        let conn_b = Arc::new(root.connection_scoped());

        conn_a
            .handle_message(&msg::tattach(1, 1, u32::MAX, "alice", "/"))
            .await
            .unwrap();

        // conn_b never attached: fid 1 is unknown to its table, and a fid
        // outside a verified session fails closed under a live monitor.
        let (_, resp) = conn_b.handle_message(&msg::twalk(2, 1, 2, &[])).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected EPERM for unauthenticated fid reuse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn translator_rejects_conflicting_attach_security_context() {
        struct TicketAuth;
        #[async_trait]
        impl AttachAuthenticator for TicketAuth {
            async fn authenticate(&self, uname: &str, _aname: &str) -> SessionContext {
                let level = if uname == "alice-ticket" {
                    Level::Secret
                } else {
                    Level::Public
                };
                permit_session(level, ALL_OPS)
            }
        }

        let t = Translator::new(Arc::new(MemoryBackend::default())).with_reference_monitor(
            Arc::new(ReferenceMonitor::new(
                Arc::new(TicketAuth),
                Arc::new(StaticLabels(Some(label(Level::Public)))),
                Arc::new(AllowAll),
            )),
        );
        let (_, resp) = t
            .handle_message(&msg::tattach(1, 1, u32::MAX, "alice-ticket", "/"))
            .await
            .unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "got {resp:?}");

        let err = t
            .handle_message(&msg::tattach(2, 2, u32::MAX, "bob-ticket", "/"))
            .await
            .unwrap_err();
        assert_eq!(
            errno_from_error(&err),
            EACCES,
            "conflicting attach session must be EACCES",
        );
    }

    #[tokio::test]
    async fn fid_table_tracks_open_and_clunk() {
        let table = FidTable::default();
        table.insert(7, 0x80, None, Vec::new());
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

    /// End-to-end hybrid-vsock: a client that speaks the CH multiplexer
    /// handshake (`connect <port>\n` → line containing `OK`) over a temp host
    /// UDS must, after the preamble is stripped, drive a byte-identical 9P2000.L
    /// session (version → attach → walk → open → read) against the shared serve
    /// core. This exercises [`Translator::serve_vsock`] +
    /// [`hybrid_vsock_handshake`] without needing a real vsock device.
    #[tokio::test]
    async fn hybrid_vsock_handshake_then_9p_session() {
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::net::UnixStream;

        // Unique temp UDS path (no tempfile dep in this crate).
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let sock = std::env::temp_dir()
            .join(format!("hs9p-vsock-{}-{nanos}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);

        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello vsock");
        let listener = UnixListener::bind(&sock).expect("bind hybrid-vsock host UDS");
        let translator = Translator::new(Arc::new(backend));
        let server = tokio::spawn(translator.serve_vsock(listener));

        // ── Client side ─────────────────────────────────────────────────
        let mut client = UnixStream::connect(&sock).await.expect("dial host UDS");

        // Hybrid-vsock preamble, exactly as `kata_agent::hybrid_vsock_connect`
        // writes it, then read the acknowledgement line and require "OK".
        client.write_all(b"connect 1024\n").await.unwrap();
        let mut ack = Vec::new();
        let mut b = [0u8; 1];
        loop {
            let n = client.read(&mut b).await.unwrap();
            if n == 0 || b[0] == b'\n' {
                break;
            }
            ack.push(b[0]);
        }
        assert!(
            String::from_utf8_lossy(&ack).contains("OK"),
            "handshake ack must contain OK, got {ack:?}"
        );

        // Read one length-prefixed 9P frame off the wire.
        async fn read_frame(s: &mut UnixStream) -> Vec<u8> {
            let mut len = [0u8; 4];
            s.read_exact(&mut len).await.unwrap();
            let total = u32::from_le_bytes(len) as usize;
            let mut buf = vec![0u8; total];
            buf[..4].copy_from_slice(&len);
            s.read_exact(&mut buf[4..]).await.unwrap();
            buf
        }

        // Tversion → Rversion.
        client.write_all(&msg::tversion(1, MSG_SIZE, "9P2000.L")).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Version { .. }), "expected Rversion, got {resp:?}");

        // Tattach fid 0 as root.
        client.write_all(&msg::tattach(2, 0, u32::MAX, "u", "/")).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "expected Rattach, got {resp:?}");

        // Twalk to the file, Tlopen, Tread.
        client.write_all(&msg::twalk(3, 0, 1, &["hello.txt"])).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "expected Rwalk, got {resp:?}");

        client.write_all(&msg::tlopen(4, 1, 0)).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Lopen { .. }), "expected Rlopen, got {resp:?}");

        client.write_all(&msg::tread(5, 1, 0, 64)).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        match resp {
            Response::Read { data } => assert_eq!(&data, b"hello vsock"),
            other => panic!("expected Rread with payload, got {other:?}"),
        }

        drop(client);
        server.abort();
        let _ = std::fs::remove_file(&sock);
    }

    /// RAW (no-handshake) vsock: a guest-initiated client that connects to a
    /// per-port host UDS and sends its 9P `Tversion` **immediately** — with NO
    /// `connect <port>\n` preamble — must drive a byte-identical 9P2000.L session.
    /// This is the #741 guest→host direction: the raw serve path must treat the
    /// very first bytes as 9P and never consume a preamble. Exercises
    /// [`Translator::serve_vsock_raw`] / [`RawVsockListener`].
    #[tokio::test]
    async fn raw_vsock_no_preamble_9p_session() {
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::net::UnixStream;

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let sock = std::env::temp_dir()
            .join(format!("hs9p-rawvsock-{}-{nanos}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);

        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello raw");
        let listener = UnixListener::bind(&sock).expect("bind raw vsock per-port host UDS");
        let translator = Translator::new(Arc::new(backend));
        let server = tokio::spawn(translator.serve_vsock_raw(listener));

        // ── Client (guest) side ─────────────────────────────────────────
        let mut client = UnixStream::connect(&sock).await.expect("dial per-port host UDS");

        async fn read_frame(s: &mut UnixStream) -> Vec<u8> {
            let mut len = [0u8; 4];
            s.read_exact(&mut len).await.unwrap();
            let total = u32::from_le_bytes(len) as usize;
            let mut buf = vec![0u8; total];
            buf[..4].copy_from_slice(&len);
            s.read_exact(&mut buf[4..]).await.unwrap();
            buf
        }

        // FIRST bytes on the wire are 9P Tversion — NO `connect <port>\n`
        // preamble. If the raw path erroneously stripped a preamble it would
        // swallow these bytes and the session would hang / fail here.
        client.write_all(&msg::tversion(1, MSG_SIZE, "9P2000.L")).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(
            matches!(resp, Response::Version { .. }),
            "raw path must treat the first bytes as 9P Tversion, got {resp:?}"
        );

        // Rest of the session proves the stream was never desynchronized.
        client.write_all(&msg::tattach(2, 0, u32::MAX, "u", "/")).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "expected Rattach, got {resp:?}");

        client.write_all(&msg::twalk(3, 0, 1, &["hello.txt"])).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "expected Rwalk, got {resp:?}");

        client.write_all(&msg::tlopen(4, 1, 0)).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        assert!(matches!(resp, Response::Lopen { .. }), "expected Rlopen, got {resp:?}");

        client.write_all(&msg::tread(5, 1, 0, 64)).await.unwrap();
        let (_, resp) = msg::parse_response(&read_frame(&mut client).await).unwrap();
        match resp {
            Response::Read { data } => assert_eq!(&data, b"hello raw"),
            other => panic!("expected Rread with payload, got {other:?}"),
        }

        drop(client);
        server.abort();
        let _ = std::fs::remove_file(&sock);
    }

    /// A malformed preamble must drop only that connection: the accept loop
    /// stays live and a well-behaved client that connects afterward still gets
    /// served. Guards the "retry, don't propagate" contract in
    /// [`HybridVsockListener::accept_conn`].
    #[tokio::test]
    async fn hybrid_vsock_bad_preamble_does_not_kill_listener() {
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::net::UnixStream;

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let sock = std::env::temp_dir()
            .join(format!("hs9p-vsock-bad-{}-{nanos}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);

        let listener = UnixListener::bind(&sock).unwrap();
        let translator = Translator::new(Arc::new(MemoryBackend::default()));
        let server = tokio::spawn(translator.serve_vsock(listener));

        // First client sends garbage; server should drop it and keep listening.
        let mut bad = UnixStream::connect(&sock).await.unwrap();
        bad.write_all(b"garbage not a preamble\n").await.unwrap();
        // Server closes this connection without acknowledging; read returns EOF.
        let mut b = [0u8; 1];
        let _ = bad.read(&mut b).await; // 0 (EOF) or Err — either is fine.
        drop(bad);

        // Second client completes a valid handshake — proving the loop survived.
        let mut good = UnixStream::connect(&sock).await.unwrap();
        good.write_all(b"connect 1024\n").await.unwrap();
        let mut ack = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            let n = good.read(&mut byte).await.unwrap();
            if n == 0 || byte[0] == b'\n' {
                break;
            }
            ack.push(byte[0]);
        }
        assert!(
            String::from_utf8_lossy(&ack).contains("OK"),
            "listener must survive a bad handshake and serve the next client"
        );

        drop(good);
        server.abort();
        let _ = std::fs::remove_file(&sock);
    }

    // ── Attach-time mount ticket (H1b / #765) ───────────────────────────

    /// A fake [`AttachAuthorizer`] mirroring the H1b `Tattach.uname` ticket
    /// check: one fixed ticket string maps to a narrowed Subject; anything else
    /// is denied with `PermissionDenied` (→ `EACCES`).
    struct FakeTicketAuth;
    #[async_trait::async_trait]
    impl crate::mount_backend::AttachAuthorizer for FakeTicketAuth {
        async fn authorize(
            &self,
            uname: &str,
        ) -> std::result::Result<hyprstream_rpc::Subject, hyprstream_vfs::MountError> {
            match uname {
                "good-ticket" => Ok(hyprstream_rpc::Subject::new("alice")),
                "other-ticket" => Ok(hyprstream_rpc::Subject::new("bob")),
                _ => Err(hyprstream_vfs::MountError::PermissionDenied("bad ticket".into())),
            }
        }
    }

    fn authorized_translator() -> Arc<Translator> {
        use hyprstream_vfs::{SyntheticMount, SyntheticNode};
        let root = SyntheticNode::dir()
            .with_child("hello.txt", SyntheticNode::file(b"hi alice".to_vec()));
        let mount: Arc<dyn hyprstream_vfs::Mount> = Arc::new(SyntheticMount::new(root));
        Arc::new(Translator::from_mount_authorized(mount, Arc::new(FakeTicketAuth)))
    }

    /// A valid ticket in `Tattach.uname` binds the session Subject and the
    /// attach + subsequent ops succeed (the H1b happy path over the shared
    /// `handle_message` core).
    #[tokio::test]
    async fn attach_ticket_valid_binds_subject_and_serves() {
        let t = authorized_translator();
        let (_, resp) = t
            .handle_message(&msg::tattach(1, 0, u32::MAX, "good-ticket", ""))
            .await
            .unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "valid ticket must attach: {resp:?}");

        // The bound Subject now threads through walk/open/read.
        let (_, resp) = t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "got {resp:?}");
        let (_, resp) = t.handle_message(&msg::tlopen(3, 1, 0)).await.unwrap();
        assert!(matches!(resp, Response::Lopen { .. }), "got {resp:?}");
        let (_, resp) = t.handle_message(&msg::tread(4, 1, 0, 64)).await.unwrap();
        match resp {
            Response::Read { data } => assert_eq!(&data, b"hi alice"),
            other => panic!("expected Rread, got {other:?}"),
        }
    }

    /// An invalid ticket in `Tattach.uname` fails the attach; the translator
    /// maps the authorizer's `PermissionDenied` to an `Rlerror` `EACCES` — no
    /// fid or mount handle is ever created.
    #[tokio::test]
    async fn attach_ticket_denied_returns_eacces() {
        let t = authorized_translator();
        let err = t
            .handle_message(&msg::tattach(1, 0, u32::MAX, "forged", ""))
            .await
            .unwrap_err();
        assert_eq!(errno_from_error(&err), EACCES, "denied ticket must be EACCES");
    }

    /// An empty `uname` (no ticket presented) is likewise denied — the ticket is
    /// mandatory on the WebTransport plane.
    #[tokio::test]
    async fn attach_empty_ticket_denied() {
        let t = authorized_translator();
        let err = t
            .handle_message(&msg::tattach(1, 0, u32::MAX, "", ""))
            .await
            .unwrap_err();
        assert_eq!(errno_from_error(&err), EACCES, "missing ticket must be EACCES");
    }

    /// A second attach may repeat the same ticket, but it must not re-scope the
    /// session to a different valid Subject.
    #[tokio::test]
    async fn second_attach_conflicting_ticket_denied() {
        let t = authorized_translator();
        let (_, resp) = t
            .handle_message(&msg::tattach(1, 0, u32::MAX, "good-ticket", ""))
            .await
            .unwrap();
        assert!(matches!(resp, Response::Attach { .. }), "got {resp:?}");

        let err = t
            .handle_message(&msg::tattach(2, 0, u32::MAX, "other-ticket", ""))
            .await
            .unwrap_err();
        assert_eq!(errno_from_error(&err), EACCES, "conflicting attach must be EACCES");
    }

    /// Fail-closed: an op arriving before a successful attach binds the Subject
    /// must not reach the mount (the `OnceCell` is unset).
    #[tokio::test]
    async fn op_before_attach_fails_closed() {
        let t = authorized_translator();
        // No Tattach: a walk cannot resolve a caller and must error, not serve.
        let err = t.handle_message(&msg::twalk(1, 0, 1, &["hello.txt"])).await.unwrap_err();
        // Untyped bookkeeping error (no MountError) → EIO, never a success.
        assert_eq!(errno_from_error(&err), EIO);
    }
}
