//! 9P-over-WebSocket export (H1a / #764).
//!
//! Serves hyprstream's Subject-scoped VFS as **9P2000.L over WebSocket** on the
//! existing axum HTTP server, wire-compatible with `wanix serve`, so stock
//! wanix 0.4 / apptron mount a hyprstream host with zero client glue.
//!
//! Two routes (registered in [`crate::server`]):
//!
//! - `GET /.well-known/export9p` — discovery. Advertises the `/9p` WebSocket
//!   endpoint, the mount-ticket query parameter, and the wire so a mounting
//!   client (S4 `WanixSessionContext`) can construct the `wss://…/9p?ticket=…`
//!   URL. (Upstream's `/.well-known/export9p` is a `mux`-session endpoint for
//!   the *reverse* direction — a browser exporting **its** namespace outward;
//!   H1a's interop path is the raw-frame `/9p` endpoint that stock wanix
//!   `import` binds connect to directly, so we serve a JSON discovery document
//!   here instead of the mux protocol. See the PR "WIRE CONTRACT" section.)
//! - `GET /9p` — WebSocket upgrade → a ws↔9p byte pump feeding the
//!   transport-agnostic [`Translator::serve_connection`] core over a
//!   [`MountBackend`] (the SAME core the UDS/vsock transports use — not a
//!   fork). Binary frames carry framed 9P2000.L messages; the pump tolerates
//!   arbitrary chunking (`P9PortReadWriter` on the wanix side buffers).
//!
//! ## Auth (reuses the existing chain)
//!
//! The mount ticket rides the URL query (`?ticket=<at+jwt>`) because a browser
//! `WebSocket` cannot set request headers. It is validated at the WS upgrade
//! via [`crate::server::middleware::verify_token_claims`] — the exact same
//! chain `auth_middleware` uses (issuer routing, Ed25519/ML-DSA signature +
//! audience validation, JTI revocation). The validated `sub` becomes the
//! session [`Subject`], threaded onto every 9P op by [`MountBackend`] (the
//! single MAC policy-enforcement point). Denials surface as `Rlerror` errnos.
//!
//! ## Server-owned liveness (H1a hard requirement)
//!
//! A dead browser peer is NOT observable to the far side (a closed port yields
//! no EOF), so a naive server would leak the 9P session forever. The pump
//! therefore OWNS liveness via **server-driven WS ping/pong keepalive**: it
//! sends a WS `Ping` every [`PING_INTERVAL`] (30s) and closes the session after
//! [`MAX_MISSED_PONGS`] consecutive unanswered pings (~90s). Closing drops the
//! pump's write half, which surfaces as EOF to `serve_connection` and tears the
//! 9P session down cleanly. A dead peer stops ponging and is reaped in ~90s; a
//! HEALTHY-but-idle mount survives **indefinitely** because pongs keep flowing
//! with zero 9P traffic (and any inbound frame — pong or data — counts as
//! liveness). The client's WS `close` event is the browser's unmount trigger,
//! so teardown stays 100% server-driven with no client keepalive. The server
//! never relies on the client noticing.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use hyprstream_9p::{AttachAuthorizer, Translator};
use hyprstream_rpc::transport::quinn_transport::{NinePWtHandler, NinePWtStream};
use hyprstream_rpc::Subject;
use hyprstream_vfs::{Mount, MountError, SyntheticMount, SyntheticNode};
use serde::{Deserialize, Serialize};
use tokio::io::{split, AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

use crate::server::state::ServerState;

/// Query-param name carrying the short-lived mount ticket. Kept in sync with
/// the discovery document and the design doc's `wss://…/9p?ticket=…` example.
const TICKET_PARAM: &str = "ticket";

/// Interval between server-sent WS keepalive `Ping`s. A HEALTHY-but-idle mount
/// stays alive indefinitely as long as the peer keeps ponging (see module docs).
const PING_INTERVAL: Duration = Duration::from_secs(30);

/// Consecutive unanswered pings tolerated before the server reaps the session.
/// With [`PING_INTERVAL`] = 30s this is ~90s of silence → dead-peer detection.
const MAX_MISSED_PONGS: u32 = 3;

/// Duplex buffer bridging the WS pump to the 9P serve core. Sized to comfortably
/// hold a full negotiated 9P msize plus headroom.
const DUPLEX_BUF: usize = 64 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Discovery: GET /.well-known/export9p
// ─────────────────────────────────────────────────────────────────────────────

/// Discovery document for the 9P-over-WebSocket export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Export9pDiscovery {
    /// 9P dialect served (`9P2000.L`).
    pub protocol: String,
    /// Transport (`websocket`).
    pub transport: String,
    /// WebSocket upgrade path (`/9p`).
    pub endpoint: String,
    /// Query-param name for the mount ticket (`ticket`).
    pub ticket_param: String,
    /// Required `Sec-WebSocket-Protocol` subprotocol, or `null` for none.
    /// H1a requires none — stock wanix opens `new WebSocket(src)` with no
    /// subprotocol.
    pub subprotocol: Option<String>,
    /// Wire framing: raw length-prefixed 9P2000.L messages, one per binary
    /// frame from the server (arbitrary chunking tolerated inbound).
    pub wire: String,
    /// The upstream tool this endpoint is wire-compatible with.
    pub compatible_with: String,
}

impl Default for Export9pDiscovery {
    fn default() -> Self {
        Self {
            protocol: "9P2000.L".to_owned(),
            transport: "websocket".to_owned(),
            endpoint: "/9p".to_owned(),
            ticket_param: TICKET_PARAM.to_owned(),
            subprotocol: None,
            wire: "raw-9p-binary-frames".to_owned(),
            compatible_with: "wanix serve".to_owned(),
        }
    }
}

/// `GET /.well-known/export9p` — advertise the 9P-over-WS export.
pub async fn export9p_metadata() -> impl IntoResponse {
    Json(Export9pDiscovery::default())
}

// ─────────────────────────────────────────────────────────────────────────────
// Mount: GET /9p (WebSocket upgrade)
// ─────────────────────────────────────────────────────────────────────────────

/// Query params on the `/9p` WebSocket upgrade.
#[derive(Debug, Deserialize)]
pub struct NinePQuery {
    /// Short-lived mount ticket (an `at+jwt`), minted via the existing token
    /// chain. Required — a missing/invalid ticket is rejected pre-upgrade.
    #[serde(default)]
    pub ticket: Option<String>,
}

/// `GET /9p` — validate the mount ticket, then upgrade to a 9P-over-WS session.
///
/// The ticket is validated BEFORE the upgrade so an unauthenticated client gets
/// a clean HTTP `401` rather than a dangling WebSocket. On success the session's
/// [`Subject`] is bound for the whole connection and enforced per-op by
/// [`MountBackend`].
pub async fn ninep_ws(
    State(state): State<ServerState>,
    Query(q): Query<NinePQuery>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let ticket = match q.ticket.as_deref() {
        Some(t) if !t.is_empty() => t,
        _ => {
            debug!("9P WS upgrade rejected: missing mount ticket");
            return (StatusCode::UNAUTHORIZED, "missing mount ticket").into_response();
        }
    };

    let subject = match validate_ticket(&state, ticket, &headers).await {
        Ok(s) => s,
        Err(reason) => {
            warn!(reason, "9P WS upgrade rejected: invalid mount ticket");
            return (StatusCode::UNAUTHORIZED, "invalid mount ticket").into_response();
        }
    };

    let mount = build_export_mount(&state, &subject);
    // `from_mount` wraps the Subject-scoped mount in a `MountBackend` and threads
    // `subject` onto every backend op — the same seam UDS/vsock use.
    let translator = Arc::new(Translator::from_mount(mount, subject));

    // No subprotocol negotiation: stock wanix connects with none.
    ws.on_upgrade(move |socket| serve_9p_websocket(socket, translator, PING_INTERVAL))
}

/// Validate a mount ticket to a session [`Subject`], reusing the server auth
/// chain ([`verify_token_claims`](crate::server::middleware::verify_token_claims)).
///
/// Real controls enforced here: Ed25519/ML-DSA signature, audience binding to
/// this resource, expiry, and JTI revocation (all inside `verify_token_claims`),
/// plus subject-name validation. The `sub` becomes the session Subject.
///
/// NOTE (flagged design fork — see PR body): one-shot replay prevention and
/// explicit browser-`Origin` binding beyond the JWT `aud` are NOT yet enforced;
/// the current controls are signature + short expiry + audience + revocation.
async fn validate_ticket(
    state: &ServerState,
    ticket: &str,
    _headers: &HeaderMap,
) -> Result<Subject, &'static str> {
    verify_mount_ticket(state, ticket).await
}

/// Transport-agnostic core of mount-ticket validation, shared by H1a's
/// URL-query path ([`validate_ticket`]) and H1b's `Tattach.uname` path
/// ([`TicketAttachAuthorizer`]). Both present the same `at+jwt` mount ticket;
/// only *where* it rides differs (WS query vs 9P attach), so the verification —
/// signature, audience, expiry, revocation (in `verify_token_claims`), then
/// subject-name validation — is identical.
async fn verify_mount_ticket(
    state: &ServerState,
    ticket: &str,
) -> Result<Subject, &'static str> {
    let claims = crate::server::middleware::verify_token_claims(state, ticket).await?;
    let local_issuers: &[&str] = &[&*state.oauth_issuer_url];
    let subject = claims.subject(local_issuers);
    subject.validate().map_err(|_| "subject validation failed")?;
    match subject.name() {
        Some(n) if !n.is_empty() => {}
        _ => return Err("empty subject"),
    }
    Ok(subject)
}

/// Build the Subject-scoped VFS [`Mount`] exported as the 9P root.
///
/// FLAGGED DESIGN FORK (see PR body): the real host VFS (`srv/{service}`,
/// `worktree/`, `stream/{topic}/{data,info,ctl}`) is composed on the #730 branch
/// and is not yet reachable from `ServerState` on `main`. Until that lands we
/// export a read-only synthetic skeleton mirroring that layout, so the endpoint
/// is live and interop-testable (stock wanix can attach + `ls`) and produces
/// real errnos. Swapping in the composed Subject-scoped namespace is a one-line
/// change here once #730 exposes it.
fn build_export_mount(_state: &ServerState, _subject: &Subject) -> Arc<dyn Mount> {
    let root = SyntheticNode::dir()
        .with_child("srv", SyntheticNode::dir())
        .with_child("worktree", SyntheticNode::dir())
        .with_child("stream", SyntheticNode::dir())
        .with_child(
            "README",
            SyntheticNode::file(
                b"hyprstream 9P-over-WebSocket export (H1a placeholder root).\n\
                  Real Subject-scoped host VFS lands with #730.\n"
                    .to_vec(),
            ),
        );
    Arc::new(SyntheticMount::new(root))
}

/// Drive one 9P-over-WebSocket session to completion.
///
/// Bridges the WebSocket message stream to the transport-agnostic
/// [`Translator::serve_connection`] core via an in-process duplex (mirroring
/// `wanix serve`'s ws↔9p pump), and OWNS session liveness via server-driven
/// ping/pong keepalive (see module docs): a `Ping` every `ping_interval`, close
/// after [`MAX_MISSED_PONGS`] consecutive unanswered pings; any inbound frame
/// resets the miss counter, so a healthy-idle mount lives indefinitely.
pub(crate) async fn serve_9p_websocket(
    socket: WebSocket,
    translator: Arc<Translator>,
    ping_interval: Duration,
) {
    // `core_side` feeds the 9P serve core; `pump_side` is driven by the WS pump.
    let (core_side, pump_side) = tokio::io::duplex(DUPLEX_BUF);
    let (mut pump_rd, mut pump_wr) = split(pump_side);
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Outbound framing is not cancel-safe (a full 9P message is two `read_exact`
    // reads: length prefix then payload), so it runs in its own task and feeds
    // whole frames to the select loop over a channel. Mirrors `serve.go`'s
    // 9p→ws goroutine.
    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(32);
    let frame_reader = tokio::spawn(async move {
        loop {
            let mut len = [0u8; 4];
            if pump_rd.read_exact(&mut len).await.is_err() {
                break; // core closed
            }
            let total = u32::from_le_bytes(len) as usize;
            if total < 4 {
                break;
            }
            let mut frame = vec![0u8; total];
            frame[..4].copy_from_slice(&len);
            if pump_rd.read_exact(&mut frame[4..]).await.is_err() {
                break;
            }
            if frame_tx.send(frame).await.is_err() {
                break; // select loop gone
            }
        }
        // `frame_tx` drops → `frame_rx` closes → select loop learns the core ended.
    });
    // `frame_reader` (a JoinHandle) owns `pump_rd`; aborted after the session.

    // The pump select loop owns the WS sink+stream, `pump_wr`, the keepalive
    // ping interval, and the missed-pong counter — the single point that both
    // forwards bytes and decides liveness.
    let pump = async move {
        let mut ping = tokio::time::interval(ping_interval);
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut missed_pongs: u32 = 0;
        loop {
            tokio::select! {
                // Outbound 9P frame → one WS binary frame.
                maybe = frame_rx.recv() => match maybe {
                    Some(frame) => {
                        if ws_tx.send(Message::Binary(frame)).await.is_err() {
                            break; // peer gone
                        }
                    }
                    None => break, // core closed (serve_connection ended)
                },
                // Inbound WS frame → core (binary), or liveness signal.
                msg = ws_rx.next() => match msg {
                    None => break, // stream ended
                    Some(Err(e)) => {
                        debug!(error = %e, "9P WS read error; closing");
                        break;
                    }
                    Some(Ok(m)) => {
                        // ANY inbound frame (pong OR data) proves the peer is alive.
                        missed_pongs = 0;
                        match m {
                            Message::Binary(bytes) => {
                                if pump_wr.write_all(&bytes).await.is_err() {
                                    break; // core gone
                                }
                            }
                            Message::Close(_) => break,
                            // Pong/Ping/Text carry no 9P payload; tungstenite
                            // auto-replies to inbound Pings at the protocol layer.
                            Message::Ping(_) | Message::Pong(_) | Message::Text(_) => {}
                        }
                    }
                },
                // Keepalive tick: send a Ping; reap the session after too many
                // consecutive unanswered pings (dead peer).
                _ = ping.tick() => {
                    missed_pongs += 1;
                    if missed_pongs >= MAX_MISSED_PONGS {
                        info!(missed_pongs, "9P WS keepalive: peer missed pongs; closing");
                        break;
                    }
                    if ws_tx.send(Message::Ping(Vec::new())).await.is_err() {
                        break; // peer gone
                    }
                }
            }
        }
        let _ = ws_tx.close().await;
        // `pump_wr` drops here → core read half sees EOF → serve_connection ends.
    };

    // Run the core and the pump concurrently. serve_connection returns on EOF
    // (pump drop) or a fatal 9P framing/write error; either way we then reap the
    // outbound frame reader.
    tokio::select! {
        r = translator.serve_connection(core_side) => {
            if let Err(e) = r {
                debug!(error = %e, "9P WS serve loop ended with error");
            }
        }
        _ = pump => {}
    }
    frame_reader.abort();
}

// ─────────────────────────────────────────────────────────────────────────────
// H1b: /9p over WebTransport (QUIC path-mux plane)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolves the mount ticket a client presents in `Tattach.uname` to a narrowed
/// session [`Subject`], reusing H1a's [`verify_mount_ticket`].
///
/// This is the attach-time credential seam for the H1b `/9p` WebTransport plane:
/// the cert-pinned mesh session carries no URL query (and a browser `WebSocket`
/// can't set headers), so the ticket rides `Tattach.uname` and is validated once
/// at attach — the per-session analogue of H1a's pre-upgrade check. On denial we
/// return [`MountError::PermissionDenied`], which the 9P translator maps to an
/// `Rlerror` (`EACCES`).
struct TicketAttachAuthorizer {
    state: ServerState,
}

#[async_trait]
impl AttachAuthorizer for TicketAttachAuthorizer {
    async fn authorize(&self, uname: &str) -> Result<Subject, MountError> {
        if uname.is_empty() {
            return Err(MountError::PermissionDenied("missing mount ticket".to_owned()));
        }
        match verify_mount_ticket(&self.state, uname).await {
            Ok(subject) => Ok(subject),
            Err(reason) => {
                warn!(reason, "9P WT attach rejected: invalid mount ticket");
                Err(MountError::PermissionDenied(reason.to_owned()))
            }
        }
    }
}

/// The injected 9P-over-WebTransport handler (H1b / #765).
///
/// Registered process-globally via [`register_ninep_wt_handler`] so the QUIC
/// path-mux `/9p` arm (`hyprstream-rpc`) can drive hyprstream's 9P core without
/// that transport crate depending on `hyprstream`. Each WT bidi stream on the
/// `/9p` path lands in [`NinePWtHandler::serve`], where it is served by the SAME
/// [`Translator::serve_connection`] core + [`build_export_mount`] as H1a —
/// only the ticket now rides `Tattach.uname` (validated by
/// [`TicketAttachAuthorizer`]) and the transport is a QUIC bidi stream.
pub struct NinePWtExport {
    state: ServerState,
}

impl NinePWtExport {
    /// Build the handler over the server's [`ServerState`] (the ticket-validation
    /// chain + export mount source).
    pub fn new(state: ServerState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl NinePWtHandler for NinePWtExport {
    async fn serve(&self, stream: Box<dyn NinePWtStream>) {
        // The export root is Subject-scoped per op by `MountBackend`; the Subject
        // is bound at `Tattach` from the validated ticket, so the mount is built
        // Subject-agnostic here (`build_export_mount` ignores the Subject in the
        // current placeholder root — see its docs; the real #730 namespace is
        // scoped by the bound Subject once it lands).
        let mount = build_export_mount(&self.state, &Subject::anonymous());
        let authorizer: Arc<dyn AttachAuthorizer> =
            Arc::new(TicketAttachAuthorizer { state: self.state.clone() });
        let translator = Arc::new(Translator::from_mount_authorized(mount, authorizer));
        if let Err(e) = translator.serve_connection(stream).await {
            debug!(error = %e, "9P WT serve loop ended with error");
        }
    }
}

/// Register the process-global 9P-over-WebTransport handler so the QUIC
/// path-mux `/9p` arm serves this node's export. Idempotent (first-wins);
/// call once at server startup where [`ServerState`] exists (see
/// [`crate::server::create_app`]).
pub fn register_ninep_wt_handler(state: ServerState) {
    let installed = hyprstream_rpc::transport::quinn_transport::set_global_ninep_handler(
        Arc::new(NinePWtExport::new(state)),
    );
    if installed {
        info!("registered 9P-over-WebTransport handler (/9p QUIC path-mux, H1b)");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::{routing::get, Router};
    use hyprstream_9p::msg::{self, Response as P9Response};
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    /// Discovery doc round-trips and advertises the H1a contract.
    #[test]
    fn discovery_defaults_match_wire_contract() {
        let d = Export9pDiscovery::default();
        assert_eq!(d.protocol, "9P2000.L");
        assert_eq!(d.endpoint, "/9p");
        assert_eq!(d.ticket_param, "ticket");
        assert_eq!(d.subprotocol, None); // stock wanix uses no subprotocol
    }

    /// End-to-end 9P-over-WebSocket attach: spin the `/9p` transport over a
    /// `MemoryBackend`, connect a real WebSocket client, and drive
    /// version → attach → walk → open → read. Exercises the full ws↔9p pump +
    /// the shared `serve_connection` core over genuine WS binary frames.
    #[tokio::test]
    async fn websocket_9p_attach_walk_read_roundtrip() {
        // A test route that serves a Subject-scoped SyntheticMount over WS via
        // `from_mount` — the SAME MountBackend path `/9p` uses in production, so
        // typed MountError→errno mapping is exercised end to end. No
        // ServerState/ticket, isolating the transport under test.
        async fn test_ws(ws: WebSocketUpgrade) -> Response {
            let root = SyntheticNode::dir()
                .with_child("hello.txt", SyntheticNode::file(b"hello ws".to_vec()));
            let mount: Arc<dyn Mount> = Arc::new(SyntheticMount::new(root));
            let translator = Arc::new(Translator::from_mount(mount, Subject::new("test")));
            ws.on_upgrade(move |s| serve_9p_websocket(s, translator, Duration::from_secs(10)))
        }

        let app = Router::new().route("/9p", get(test_ws));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let url = format!("ws://{addr}/9p");
        let (mut ws, _resp) = tokio_tungstenite::connect_async(&url).await.unwrap();

        // Helper: send a framed 9P T-message, await the next binary frame,
        // decode the R-message.
        async fn rpc(
            ws: &mut tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            tmsg: Vec<u8>,
        ) -> P9Response {
            ws.send(WsMessage::Binary(tmsg)).await.unwrap();
            loop {
                match ws.next().await.unwrap().unwrap() {
                    WsMessage::Binary(buf) => {
                        let (_tag, resp) = msg::parse_response(&buf).unwrap();
                        return resp;
                    }
                    _ => continue,
                }
            }
        }

        // Tversion → Rversion.
        let resp = rpc(&mut ws, msg::tversion(1, 8192, "9P2000.L")).await;
        assert!(matches!(resp, P9Response::Version { .. }), "got {resp:?}");

        // Tattach fid 0 as root.
        let resp = rpc(&mut ws, msg::tattach(2, 0, u32::MAX, "u", "")).await;
        assert!(matches!(resp, P9Response::Attach { .. }), "got {resp:?}");

        // Twalk to the file, Tlopen, Tread.
        let resp = rpc(&mut ws, msg::twalk(3, 0, 1, &["hello.txt"])).await;
        assert!(matches!(resp, P9Response::Walk { .. }), "got {resp:?}");

        let resp = rpc(&mut ws, msg::tlopen(4, 1, 0)).await;
        assert!(matches!(resp, P9Response::Lopen { .. }), "got {resp:?}");

        let resp = rpc(&mut ws, msg::tread(5, 1, 0, 64)).await;
        match resp {
            P9Response::Read { data } => assert_eq!(&data, b"hello ws"),
            other => panic!("expected Rread, got {other:?}"),
        }

        // A walk to a missing path returns a typed ENOENT Rlerror (H1a errnos).
        let resp = rpc(&mut ws, msg::twalk(6, 0, 2, &["nope"])).await;
        match resp {
            P9Response::Error { ecode } => assert_eq!(ecode, 2, "missing path must be ENOENT"),
            other => panic!("expected Rlerror ENOENT, got {other:?}"),
        }

        ws.close(None).await.unwrap();
        server.abort();
    }
}
