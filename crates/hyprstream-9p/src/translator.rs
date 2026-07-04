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
use hyprstream_rpc::auth::mac::SecurityContext;
use hyprstream_rpc::Subject;
use hyprstream_vfs::Mount;
use tokio::io::{split, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

use crate::backend::Backend;
use crate::mac_seam::{
    anonymous_floor, AccessDecider, AllowAllDecider, Action, AnonymousAuthenticator,
    AttachAuthenticator,
};
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
    /// The caller's verified security context, derived once at `Tattach` by
    /// the translator's [`AttachAuthenticator`] and inherited by every fid
    /// walked from it (#568) — never re-derived per op.
    ctx: SecurityContext,
}

/// Server-side fid table: 9P fid → state.
///
/// `DashMap` so the translator can be shared across per-connection tasks.
#[derive(Default)]
pub struct FidTable {
    fids: DashMap<u32, FidState>,
}

impl FidTable {
    /// Insert state for a fid (from walk/attach), inheriting `ctx` from the
    /// attach or the source fid being walked.
    pub fn insert(&self, fid: u32, qtype: u8, ctx: SecurityContext) {
        self.fids.insert(fid, FidState { qtype, opened: false, ctx });
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

    /// Look up a fid's cached security context (from attach/inherited walk).
    pub fn security_context(&self, fid: u32) -> Option<SecurityContext> {
        self.fids.get(&fid).map(|s| s.ctx.clone())
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
    /// Verifies the `Tattach` credential and derives the fid's cached
    /// [`SecurityContext`] (#568). Defaults to [`AnonymousAuthenticator`]
    /// (always the anonymous floor), matching today's actual behavior — no
    /// identity verification at attach. Real verification is an explicit
    /// opt-in via [`Translator::with_authenticator`].
    authenticator: Arc<dyn AttachAuthenticator>,
    /// Authorizes each op against the fid's cached context (#568). Defaults
    /// to [`AllowAllDecider`] (always allow), matching today's actual
    /// behavior — no per-op authorization. See the module-level docs on
    /// `mac_seam` for why a real deny-on-unlabeled decider must NOT be the
    /// default: every object in the 9P namespace is unlabeled today, and the
    /// S1 invariant is unlabeled ⇒ deny, not ⇒ floor.
    decider: Arc<dyn AccessDecider>,
}

impl Translator {
    /// Build a translator over the given backend. The fid table starts empty.
    /// Attach/authorization use the inert defaults (no behavior change from
    /// before #568): see [`Translator::with_authenticator`] /
    /// [`Translator::with_decider`] to opt into real enforcement.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            fids: Arc::new(FidTable::default()),
            authenticator: Arc::new(AnonymousAuthenticator),
            decider: Arc::new(AllowAllDecider),
        }
    }

    /// Replace the attach-time credential verifier. The `hyprstream` crate is
    /// expected to call this with a real verifier (backed by the S6-minted
    /// access token + the service trust store) once a standalone "verify a
    /// presented token" entry point exists outside the OAuth HTTP handler —
    /// see the #568 PR discussion for why that plumbing is not included here.
    pub fn with_authenticator(mut self, authenticator: Arc<dyn AttachAuthenticator>) -> Self {
        self.authenticator = authenticator;
        self
    }

    /// Replace the per-op access decider. The `hyprstream` crate is expected
    /// to call this with a real decider (backed by `mac::avc::CachingAvc`,
    /// optionally wrapped in `mac_seam::AuditedDecider`) once object labels
    /// are actually available for the mount being served (genesis labeling,
    /// or #699 carrier-b for content-addressed data) — wiring a real decider
    /// before that point would deny every existing 9P client, since an
    /// unlabeled object must deny per the S1 invariant, not fall back to a
    /// permissive floor.
    pub fn with_decider(mut self, decider: Arc<dyn AccessDecider>) -> Self {
        self.decider = decider;
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
        Self::new(Arc::new(MountBackend::new(mount, subject)))
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
        Self::new(Arc::new(MountBackend::with_authorizer(mount, authorizer)))
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
        // #568: verify the attach credential exactly once, deriving the
        // context every fid walked from this attach will inherit. The
        // default AnonymousAuthenticator always floors — no behavior change
        // until a real authenticator is injected via `with_authenticator`.
        let ctx = self.authenticator.authenticate(uname, aname).await;
        // Attach establishes the root fid. Walk the empty path to materialize
        // a root qid from the backend, then record it.
        let walk = self.backend.walk(fid, fid, &[]).await?;
        let qtype = walk.qids.last().map(|q| q.qtype).unwrap_or(0);
        self.fids.insert(fid, qtype, ctx);
        let qid = walk.qids.into_iter().next().unwrap_or_default();
        Ok(Response::Attach { qid })
    }

    async fn handle_walk(
        &self,
        fid: u32,
        newfid: u32,
        wnames: Vec<String>,
    ) -> Result<Response> {
        if let Some(err) = self.deny(fid, Action::Walk) {
            return Ok(err);
        }
        // The context is inherited from the fid being walked (#568 — a
        // context is derived once at attach, never re-verified per op).
        let ctx = self.fids.security_context(fid).unwrap_or_else(anonymous_floor);
        // Mirror of RemoteModelMount::walk: call backend.walk with the source
        // fid, the new fid, and the path components.
        let result = self.backend.walk(fid, newfid, &wnames).await?;
        if let Some(q) = result.qids.last() {
            self.fids.insert(newfid, q.qtype, ctx);
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

    /// Consults the decider for `fid`+`action` against the fid's cached
    /// context. Returns `Some(Response::Error)` (EPERM) if denied, `None` if
    /// allowed. No object label is available at this layer (see the
    /// `mac_seam` module docs) — `object_label` is always `None` here; a real
    /// decider must treat that as "deny" per the S1 invariant, not as
    /// "unrestricted".
    ///
    /// A fid with no cached context (should not happen — every fid is
    /// populated at attach or inherits from its parent on walk) is treated
    /// as the anonymous floor rather than panicking.
    fn deny(&self, fid: u32, action: Action) -> Option<Response> {
        let ctx = self.fids.security_context(fid).unwrap_or_else(anonymous_floor);
        if self.decider.check(&ctx, None, action) {
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

    /// #568: a translator with a real (non-default) decider actually blocks
    /// ops, proving the `deny`/`FidTable::security_context` wiring is live —
    /// not just present but inert. Uses a decider that denies `Action::Read`
    /// unconditionally; walk/open (unblocked) still succeed, read is
    /// rejected with `Rlerror EPERM` rather than reaching the backend.
    #[tokio::test]
    async fn injected_decider_blocks_denied_action() {
        struct DenyReads;
        impl AccessDecider for DenyReads {
            fn check(
                &self,
                _ctx: &SecurityContext,
                _object_label: Option<&hyprstream_rpc::auth::mac::SecurityLabel>,
                action: Action,
            ) -> bool {
                !matches!(action, Action::Read)
            }
        }

        let backend = MemoryBackend::default();
        backend.add_file("/hello.txt", b"hello world");
        let t = Translator::new(Arc::new(backend)).with_decider(Arc::new(DenyReads));

        t.handle_message(&msg::tattach(1, 0, u32::MAX, "user", "/")).await.unwrap();
        let (_, resp) =
            t.handle_message(&msg::twalk(2, 0, 1, &["hello.txt"])).await.unwrap();
        assert!(matches!(resp, Response::Walk { .. }), "walk must still succeed");

        let (_, resp) = t.handle_message(&msg::tlopen(3, 1, 0)).await.unwrap();
        assert!(matches!(resp, Response::Lopen { .. }), "open must still succeed");

        let (_, resp) = t.handle_message(&msg::tread(4, 1, 0, 64)).await.unwrap();
        match resp {
            Response::Error { ecode } => assert_eq!(ecode, libc_eperm()),
            other => panic!("expected Rlerror EPERM, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fid_table_tracks_open_and_clunk() {
        let table = FidTable::default();
        table.insert(7, 0x80, anonymous_floor());
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
            if uname == "good-ticket" {
                Ok(hyprstream_rpc::Subject::new("alice"))
            } else {
                Err(hyprstream_vfs::MountError::PermissionDenied("bad ticket".into()))
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
