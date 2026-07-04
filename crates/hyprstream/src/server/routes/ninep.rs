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
//! no EOF), so a naive server would leak the 9P session forever. The inbound
//! pump therefore OWNS liveness: it wraps every WS read in an **idle timeout**
//! ([`IDLE_TIMEOUT`]); on idle, WS close, or WS error it drops the pump's write
//! half, which surfaces as EOF to `serve_connection` and tears the session
//! down cleanly. The server never relies on the client noticing.

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
use futures::{SinkExt, StreamExt};
use hyprstream_9p::Translator;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{Mount, SyntheticMount, SyntheticNode};
use serde::{Deserialize, Serialize};
use tokio::io::{split, AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

use crate::server::state::ServerState;

/// Query-param name carrying the short-lived mount ticket. Kept in sync with
/// the discovery document and the design doc's `wss://…/9p?ticket=…` example.
const TICKET_PARAM: &str = "ticket";

/// Idle timeout for a 9P-over-WS session. If no inbound WS message arrives
/// within this window the server closes the session (see module docs:
/// server-owned liveness — a dead browser peer is otherwise invisible).
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);

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
    ws.on_upgrade(move |socket| serve_9p_websocket(socket, translator, IDLE_TIMEOUT))
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
/// `wanix serve`'s ws↔9p pump), and OWNS session liveness (see module docs).
pub(crate) async fn serve_9p_websocket(
    socket: WebSocket,
    translator: Arc<Translator>,
    idle: Duration,
) {
    // `core_side` feeds the 9P serve core; `pump_side` is driven by the WS pump.
    let (core_side, pump_side) = tokio::io::duplex(DUPLEX_BUF);
    let (mut pump_rd, mut pump_wr) = split(pump_side);
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Outbound: framed 9P responses from the core → one WS binary frame each
    // (mirrors `serve.go`'s 9p→ws goroutine: read length prefix, then payload).
    let to_ws = tokio::spawn(async move {
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
            if ws_tx.send(Message::Binary(frame)).await.is_err() {
                break; // peer gone
            }
        }
        let _ = ws_tx.close().await;
    });

    // Inbound: WS binary frames → core (raw bytes; the core reframes). Owns
    // liveness — an idle/closed/errored peer drops `pump_wr`, EOF-ing the core.
    let inbound = async move {
        loop {
            match tokio::time::timeout(idle, ws_rx.next()).await {
                Err(_) => {
                    info!("9P WS session idle timeout; closing");
                    break;
                }
                Ok(None) => break, // stream ended
                Ok(Some(Err(e))) => {
                    debug!(error = %e, "9P WS read error; closing");
                    break;
                }
                Ok(Some(Ok(msg))) => match msg {
                    Message::Binary(bytes) => {
                        if pump_wr.write_all(&bytes).await.is_err() {
                            break; // core gone
                        }
                    }
                    Message::Close(_) => break,
                    // Text/Ping/Pong are not 9P payload; ignore (axum auto-Pongs Pings).
                    Message::Ping(_) | Message::Pong(_) | Message::Text(_) => {}
                },
            }
        }
        // `pump_wr` drops here → core read half sees EOF → serve_connection ends.
    };

    // Run the core and the inbound pump concurrently. serve_connection returns
    // on EOF (inbound drop) or a fatal 9P framing/write error; either way we
    // then abort the outbound task.
    tokio::select! {
        r = translator.serve_connection(core_side) => {
            if let Err(e) = r {
                debug!(error = %e, "9P WS serve loop ended with error");
            }
        }
        _ = inbound => {}
    }
    to_ws.abort();
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
