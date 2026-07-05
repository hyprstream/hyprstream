//! XetService — HuggingFace-XET CAS HTTP face (epic #654, task #5, FOUNDATION).
//!
//! Exposes the exact HuggingFace-XET CAS wire routes (as spoken by xet-core's
//! `cas_client/src/remote_client.rs`) so a standard xet-enabled git repo can use
//! hyprstream as an **alternative XET backend to HuggingFace** by pointing its
//! CAS endpoint at us:
//!
//! | Route                                | Status      | Notes |
//! |--------------------------------------|-------------|-------|
//! | `GET  /get_xorb/{hash}/`             | IMPLEMENTED | raw xorb bytes from `CasStore`, `Range`-aware |
//! | `GET  /v1/reconstructions/{hash}`    | 501         | needs verified HF term/fetch_info mapping |
//! | `GET  /v1/chunks/{key}`              | 501         | global dedup index is follow-up |
//! | `POST /v1/xorbs/{key}`               | 501         | xorb→`putBlob(grantRepo)` mapping unresolved |
//! | `POST /v1/shards`                    | 501         | shard ingest is follow-up |
//!
//! # Architecture (mirrors `OAIService`)
//!
//! A dual-stack service: an HTTP data plane (this file's axum `Router`) plus an
//! RPC control channel (health/shutdown) registered via `Spawnable`. It is a
//! **thin translator** over the authenticated core — it dials the `registry`
//! service (reusing the authenticated `putBlob`/`getBlob` RPCs) and holds no
//! standing CAS *write* authority of its own. Reads for `/get_xorb` come from the
//! shared `cas_serve::CasStore` (the same store the registry's `getBlob` uses).
//!
//! # Auth
//!
//! Every route is behind [`xet_auth_middleware`], which requires a valid
//! `Authorization: Bearer <jwt>` verified with the cluster JWT key
//! (`ctx.jwt_verifying_key()`) — the same Ed25519 verification path the OAI
//! service uses. The mapped identity (`sub`) is attached as
//! [`AuthenticatedUser`] for downstream authorization.
//!
//! ## Known authorization gap (foundation-level) — TODO(#654)
//!
//! The HF `/get_xorb/{hash}/` route carries a *xorb hash* and **no `grantRepo`**,
//! so it cannot be mapped onto the registry's `getBlob` (which is file-merkle +
//! `grantRepo` and enforces per-repo entitlement — "the hash is not a
//! capability"). This foundation therefore serves xorb bytes to any
//! *authenticated* caller but does **not** yet enforce the per-repo grant that
//! `getBlob` does. A follow-up must add a xorb→repo provenance reverse-index (or
//! an equivalent grant context) so xorb fetches are authorized, not merely
//! authenticated. Until then keep this surface disabled in untrusted multi-tenant
//! deployments (`[xet] enabled = false` by default).

use crate::config::{TlsConfig, XetConfig};
use crate::server::tls::{resolve_rustls_config, serve_app};
use crate::server::AuthenticatedUser;
use crate::services::RegistryClient;
use anyhow::Result;
use axum::{
    extract::{Path, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use crate::storage::cas::{CasError, CasSubstrate, DedupDomain};
use cas_serve::StoreError;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::Spawnable;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{info, warn};

/// Service name for registry and logging.
pub const SERVICE_NAME: &str = "xet";

/// Shared state for the axum HTTP handlers.
#[derive(Clone)]
pub struct XetState {
    /// L1 CAS substrate backing `/get_xorb` reads (shared with the registry's
    /// `getBlob` reconstruction path, #812). Reads use the default (untenanted,
    /// BLAKE3, local) dedup domain.
    pub store: CasSubstrate,
    /// Authenticated registry client — the write path (`/v1/xorbs`, `/v1/shards`)
    /// must translate through its `putBlob` RPC rather than writing directly.
    /// Dialed by the factory and held here so the 501 write stubs can be wired
    /// without a wiring change. `Option` because the write path is not yet
    /// implemented (the read/auth surface is exercisable without a live registry).
    #[allow(dead_code)]
    pub registry: Option<RegistryClient>,
    /// Cluster JWT verifying key (Ed25519) — same key path the OAI face uses.
    pub jwt_verifying_key: VerifyingKey,
    /// Expected JWT audience (this server's resource URL).
    pub audience: String,
}

/// XetService — HF-XET CAS HTTP face with an RPC control channel.
pub struct XetService {
    config: XetConfig,
    tls_config: TlsConfig,
    state: XetState,
    control_transport: TransportConfig,
    #[allow(dead_code)]
    verifying_key: VerifyingKey,
}

impl XetService {
    /// Create a new XetService.
    pub fn new(
        config: XetConfig,
        tls_config: TlsConfig,
        state: XetState,
        control_transport: TransportConfig,
        verifying_key: VerifyingKey,
    ) -> Self {
        Self {
            config,
            tls_config,
            state,
            control_transport,
            verifying_key,
        }
    }

    /// HTTP bind address from config.
    pub fn http_addr(&self) -> Result<SocketAddr> {
        let addr_str = format!("{}:{}", self.config.host, self.config.port);
        addr_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))
    }
}

/// Build the HF-XET CAS wire router. Route paths are the exact compatibility
/// contract with xet-core's `cas_client` — do not rename them.
pub fn create_xet_router(state: XetState) -> Router {
    Router::new()
        // IMPLEMENTED: raw xorb bytes, Range-aware.
        .route("/get_xorb/:hash/", get(get_xorb_handler))
        // 501 stubs (see module docs for the interop gap each carries).
        .route("/v1/reconstructions/:hash", get(reconstruction_stub))
        .route("/v1/chunks/:key", get(chunks_stub))
        .route("/v1/xorbs/:key", post(upload_xorb_stub))
        .route("/v1/shards", post(upload_shard_stub))
        // Auth applies to every route above.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            xet_auth_middleware,
        ))
        .with_state(state)
}

/// JWT auth middleware: require `Authorization: Bearer <jwt>`, verify it with the
/// cluster JWT key (same verification the OAI face performs), and attach the
/// mapped identity for downstream authorization. Rejects unauthenticated
/// requests with 401 (fail-closed).
async fn xet_auth_middleware(
    State(state): State<XetState>,
    mut request: Request,
    next: Next,
) -> Response {
    let token = match bearer_token(request.headers()) {
        Some(t) => t,
        None => return unauthorized("missing bearer token"),
    };

    // Verify with the cluster JWT key, binding the audience to this resource.
    match crate::auth::jwt::decode(&token, &state.jwt_verifying_key, Some(&state.audience)) {
        Ok(claims) => {
            request.extensions_mut().insert(AuthenticatedUser {
                user: claims.sub,
                token: Some(token),
                exp: Some(claims.exp),
            });
            next.run(request).await
        }
        Err(e) => {
            warn!(error = %e, "xet auth: JWT validation failed");
            unauthorized("invalid token")
        }
    }
}

/// Extract a bearer token from the `Authorization` header (RFC 6750).
fn bearer_token(headers: &HeaderMap) -> Option<String> {
    let h = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    if h.len() > 7 && h[..7].eq_ignore_ascii_case("bearer ") {
        Some(h[7..].trim().to_owned())
    } else {
        None
    }
}

/// Standard 401 with `WWW-Authenticate: Bearer`.
fn unauthorized(detail: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        detail.to_owned(),
    )
        .into_response()
}

/// `GET /get_xorb/{hash}/` — serve the raw bytes of a single xorb, honoring a
/// single-range `Range: bytes=start-end` header (inclusive end, RFC 7233).
///
/// The bytes are the store's on-disk xorb payload. NOTE(#654): byte-for-byte
/// compatibility with HuggingFace's xorb *serialization* (chunk framing /
/// compression) is a follow-up interop item — this returns exactly what the
/// hyprstream store holds.
async fn get_xorb_handler(
    State(state): State<XetState>,
    Path(hash): Path<String>,
    headers: HeaderMap,
) -> Response {
    let bytes = match state
        .store
        .read_xorb(&DedupDomain::local_default(), &hash)
        .await
    {
        Ok(b) => b,
        Err(CasError::Store(StoreError::NotFound(_))) => {
            return (StatusCode::NOT_FOUND, "xorb not found").into_response()
        }
        Err(CasError::Store(StoreError::InvalidHash(_))) => {
            return (StatusCode::BAD_REQUEST, "invalid xorb hash").into_response()
        }
        Err(e) => {
            warn!(error = %e, "get_xorb: store read failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "store error").into_response();
        }
    };

    match parse_range(headers.get(header::RANGE), bytes.len() as u64) {
        RangeOutcome::Full => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
                (header::ACCEPT_RANGES, "bytes".to_owned()),
            ],
            bytes,
        )
            .into_response(),
        RangeOutcome::Partial { start, end } => {
            // end is inclusive per RFC 7233.
            let slice = bytes[start as usize..=end as usize].to_vec();
            let content_range = format!("bytes {start}-{end}/{}", bytes.len());
            (
                StatusCode::PARTIAL_CONTENT,
                [
                    (header::CONTENT_TYPE, "application/octet-stream".to_owned()),
                    (header::ACCEPT_RANGES, "bytes".to_owned()),
                    (header::CONTENT_RANGE, content_range),
                ],
                slice,
            )
                .into_response()
        }
        RangeOutcome::Unsatisfiable => (
            StatusCode::RANGE_NOT_SATISFIABLE,
            [(header::CONTENT_RANGE, format!("bytes */{}", bytes.len()))],
            "range not satisfiable",
        )
            .into_response(),
    }
}

/// Result of interpreting a `Range` header against a known content length.
enum RangeOutcome {
    Full,
    Partial { start: u64, end: u64 },
    Unsatisfiable,
}

/// Parse a single `bytes=start-end` range (the only form the xet client emits).
/// Absent/unparseable headers fall back to a full-body response.
fn parse_range(header_val: Option<&axum::http::HeaderValue>, len: u64) -> RangeOutcome {
    let raw = match header_val.and_then(|v| v.to_str().ok()) {
        Some(s) => s,
        None => return RangeOutcome::Full,
    };
    let spec = match raw.strip_prefix("bytes=") {
        Some(s) => s.trim(),
        None => return RangeOutcome::Full, // not a byte range → serve full
    };
    // Only a single range is supported at foundation level.
    if spec.contains(',') {
        return RangeOutcome::Full;
    }
    let (start_s, end_s) = match spec.split_once('-') {
        Some(pair) => pair,
        None => return RangeOutcome::Full,
    };

    if len == 0 {
        return RangeOutcome::Unsatisfiable;
    }
    let last = len - 1;

    match (start_s.trim(), end_s.trim()) {
        // suffix range: "-N" → last N bytes
        ("", n) => match n.parse::<u64>() {
            Ok(0) => RangeOutcome::Unsatisfiable,
            Ok(n) => {
                let start = len.saturating_sub(n);
                RangeOutcome::Partial { start, end: last }
            }
            Err(_) => RangeOutcome::Full,
        },
        // open-ended: "S-"
        (s, "") => match s.parse::<u64>() {
            Ok(start) if start <= last => RangeOutcome::Partial { start, end: last },
            Ok(_) => RangeOutcome::Unsatisfiable,
            Err(_) => RangeOutcome::Full,
        },
        // closed: "S-E"
        (s, e) => match (s.parse::<u64>(), e.parse::<u64>()) {
            (Ok(start), Ok(end)) if start <= end && start <= last => {
                RangeOutcome::Partial {
                    start,
                    end: end.min(last),
                }
            }
            (Ok(_), Ok(_)) => RangeOutcome::Unsatisfiable,
            _ => RangeOutcome::Full,
        },
    }
}

// ── 501 stubs ────────────────────────────────────────────────────────────────
// Each returns a precise TODO(#654) rather than a guessed protocol response.

/// `GET /v1/reconstructions/{hash}` — reconstruction manifest.
///
/// TODO(#654): return a `cas_types::QueryReconstructionResponse` (public,
/// serde-serializable in the `cas_types` workspace dep). Building it correctly
/// requires mapping the hyprstream store's shard segments into HF-XET
/// `CASReconstructionTerm { hash, unpacked_length, range: ChunkRange }` +
/// per-xorb `CASReconstructionFetchInfo { range, url, url_range }`. The store's
/// xorbs are raw-concatenated chunks (whole-xorb segments), so the chunk-index
/// ranges and `url_range` byte offsets HF expects are NOT yet derivable without
/// interop verification against a real xet client. Stubbed instead of hand-
/// rolling a guessed schema.
async fn reconstruction_stub(Path(_hash): Path<String>) -> Response {
    not_implemented("reconstruction manifest: needs verified HF term/fetch_info mapping")
}

/// `GET /v1/chunks/{key}` — global chunk dedup query.
///
/// TODO(#654): the global chunk→shard dedup index does not exist yet; the store
/// is content-addressed but has no reverse chunk index. Follow-up.
async fn chunks_stub(Path(_key): Path<String>) -> Response {
    not_implemented("global chunk dedup query: dedup index is follow-up")
}

/// `POST /v1/xorbs/{key}` — upload a xorb.
///
/// TODO(#654): the authenticated write core is `registry.putBlob(bytes,
/// grantRepo)`, which ingests *file* bytes and server-computes the merkle bound
/// to a `grantRepo`. The HF xorb-upload request is xorb-granularity and carries
/// no grantRepo, so the mapping (grantRepo derivation, xorb-vs-file granularity,
/// HF xorb deserialization) is unresolved at foundation level. Returning 501
/// rather than opening an unauthenticated direct-to-store write path.
async fn upload_xorb_stub(Path(_key): Path<String>) -> Response {
    not_implemented("xorb upload: xorb→putBlob(grantRepo) mapping unresolved")
}

/// `POST /v1/shards` — upload a shard (dedup info).
///
/// TODO(#654): shard ingest must also route through the authenticated write
/// core; the grantRepo/provenance binding for a bare shard upload is unresolved.
async fn upload_shard_stub() -> Response {
    not_implemented("shard upload: authenticated ingest mapping is follow-up")
}

/// Uniform 501 with a `TODO(#654)`-tagged reason.
fn not_implemented(reason: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        format!("TODO(#654): {reason}"),
    )
        .into_response()
}

impl Spawnable for XetService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        vec![(SocketKind::Rep, self.control_transport.clone())]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("runtime: {e}")))?;

        rt.block_on(async move {
            let addr = self.http_addr().map_err(|e| {
                hyprstream_rpc::error::RpcError::SpawnFailed(format!("Invalid HTTP address: {e}"))
            })?;

            let rustls_config = resolve_rustls_config(
                &self.tls_config,
                self.config.tls_cert.as_ref(),
                self.config.tls_key.as_ref(),
            )
            .await
            .map_err(|e| hyprstream_rpc::error::RpcError::SpawnFailed(format!("TLS config: {e}")))?;

            let scheme = if rustls_config.is_some() { "https" } else { "http" };
            let app = create_xet_router(self.state.clone());

            info!("HF-XET CAS API available at {scheme}://{addr} (get_xorb implemented; other routes 501, see #654)");

            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }
            let _ = hyprstream_rpc::notify::ready();

            serve_app(addr, app, rustls_config, shutdown, "XetService").await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // panicking is correct in unit tests
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use tower::ServiceExt; // oneshot

    #[test]
    fn test_service_name() {
        assert_eq!(SERVICE_NAME, "xet");
    }

    /// Build an XetState over a temp CasStore holding one xorb, plus a random
    /// (never-signed-for) JWT key so auth rejects everything unauthenticated.
    fn test_state_with_xorb(dir: &std::path::Path, hash: &str, bytes: &[u8]) -> XetState {
        std::fs::create_dir_all(dir.join("xorbs")).unwrap();
        std::fs::write(dir.join("xorbs").join(format!("default.{hash}")), bytes).unwrap();

        let (_sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();

        XetState {
            store: CasSubstrate::new(dir),
            // Write path is 501; no live registry needed to exercise reads/auth.
            registry: None,
            jwt_verifying_key: vk,
            audience: "http://localhost:6792".to_owned(),
        }
    }

    // A valid 64-hex (sha256-shaped) content address the store's hash parser accepts.
    const HASH: &str = "aa00bb11cc22dd33ee44ff5566778899aabbccddeeff00112233445566778899";

    #[tokio::test]
    async fn get_xorb_happy_path_returns_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"xorb-payload-bytes";
        let state = test_state_with_xorb(dir.path(), HASH, payload);

        let resp = get_xorb_handler(
            State(state),
            Path(HASH.to_owned()),
            HeaderMap::new(),
        )
        .await;

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], payload);
    }

    #[tokio::test]
    async fn get_xorb_honors_range() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"0123456789";
        let state = test_state_with_xorb(dir.path(), HASH, payload);

        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, "bytes=2-5".parse().unwrap());
        let resp = get_xorb_handler(State(state), Path(HASH.to_owned()), headers).await;

        assert_eq!(resp.status(), StatusCode::PARTIAL_CONTENT);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"2345");
    }

    #[tokio::test]
    async fn unauthenticated_request_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_xorb(dir.path(), HASH, b"x");
        let app = create_xet_router(state);

        let resp = app
            .oneshot(
                HttpRequest::builder()
                    .uri(format!("/get_xorb/{HASH}/"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn stub_routes_return_501_when_authorized_shape() {
        // The 501 handlers are auth-gated too, so hit them directly to assert the
        // foundation contract (precise TODO, not a guessed body).
        let resp = reconstruction_stub(Path(HASH.to_owned())).await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let resp = upload_xorb_stub(Path("k".to_owned())).await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let resp = upload_shard_stub().await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
