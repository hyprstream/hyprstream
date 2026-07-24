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
//! # Architecture
//!
//! An HTTP data plane (this file's axum `Router`) implemented as a **thin
//! translator** over the authenticated core — it dials the `registry`
//! service (reusing the authenticated `putBlob`/`getBlob` RPCs) and holds no
//! standing CAS *write* authority of its own. Reads for `/get_xorb` go through
//! the Subject-threaded CAS mount, not a bare store read.
//!
//! # Auth
//!
//! Every data route uses the same shared resource authentication and rate-limit
//! middleware as OAI: DPoP sender binding/replay protection, federated issuer
//! resolution, rotation and composite keys, JTI revocation, strict audience
//! validation, and per-subject quotas. The mapped identity (`sub`) is attached
//! as [`AuthenticatedUser`] for downstream authorization.
//!
//! ## Known provenance gap (foundation-level)
//!
//! The HF `/get_xorb/{hash}/` route carries a *xorb hash* and **no `grantRepo`**,
//! so it cannot be mapped onto the registry's `getBlob` (which is file-merkle +
//! `grantRepo` and enforces per-repo entitlement — "the hash is not a
//! capability"). #813 moved this route behind the Subject-threaded CAS mount so
//! the policy boundary is the same `Mount` surface used by namespaces, and
//! #1094 made that boundary **enforcing**: the mount runs
//! [`BootstrapCasAuthorizer`], where the day-one grant "any authenticated
//! subject may read xorb X" is an explicit, audited policy object and
//! everything else is default-deny — plane #1 of #1091's R4b
//! compile-and-ratchet MAC rollout. The remaining work is attaching
//! repo/compartment provenance labels to xorb objects (#699) and flowing
//! subject clearances (#698) so grant breadth ratchets to 0 and the mount can
//! make a full AVC decision; the ratchet path is documented on
//! [`BootstrapCasAuthorizer`].

use crate::config::{TlsConfig, XetConfig};
use crate::server::state::ResourceAuthState;
use crate::server::tls::{resolve_rustls_config, serve_app};
use crate::server::{AuthenticatedUser, middleware as server_middleware};
use crate::services::RegistryClient;
use crate::storage::cas::{BootstrapCasAuthorizer, CasMount, CasSubstrate, DedupDomain};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::Spawnable;
use hyprstream_vfs::{Mount, MountError, OREAD};
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
    /// `getBlob` reconstruction path, #812). HTTP reads go through `CasMount`
    /// so the authenticated Subject reaches the VFS policy boundary (#813),
    /// enforced behind the #1094 bootstrap grant.
    pub store: CasSubstrate,
    /// Authenticated registry client — the write path (`/v1/xorbs`, `/v1/shards`)
    /// must translate through its `putBlob` RPC rather than writing directly.
    /// Dialed by the factory and held here so the 501 write stubs can be wired
    /// without a wiring change. `Option` because the write path is not yet
    /// implemented (the read/auth surface is exercisable without a live registry).
    #[allow(dead_code)]
    pub registry: Option<RegistryClient>,
    /// Shared OAI-equivalent authentication and rate-limit state.
    pub auth: ResourceAuthState,
}

/// XetService — HF-XET CAS HTTP face.
pub struct XetService {
    config: XetConfig,
    tls_config: TlsConfig,
    state: XetState,
}

impl XetService {
    /// Create a new XetService.
    pub fn new(config: XetConfig, tls_config: TlsConfig, state: XetState) -> Self {
        Self {
            config,
            tls_config,
            state,
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
    let protected_routes = Router::new()
        // IMPLEMENTED: raw xorb bytes, Range-aware.
        .route("/get_xorb/:hash/", get(get_xorb_handler))
        // 501 stubs (see module docs for the interop gap each carries).
        .route("/v1/reconstructions/:hash", get(reconstruction_stub))
        .route("/v1/chunks/:key", get(chunks_stub))
        .route("/v1/xorbs/:key", post(upload_xorb_stub))
        .route("/v1/shards", post(upload_shard_stub))
        // Rate limiting is inner, so it sees the identity inserted by auth.
        .layer(middleware::from_fn_with_state(
            state.auth.clone(),
            server_middleware::rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.auth.clone(),
            server_middleware::auth_middleware,
        ));

    Router::new()
        .route(
            "/.well-known/oauth-protected-resource",
            get(xet_protected_resource_metadata),
        )
        .merge(protected_routes)
        .with_state(state)
}

/// RFC 9728 metadata is Xet's public discovery surface. HTTP faces cannot be
/// represented by the RPC endpoint registry, whose socket kinds are RPC-only.
async fn xet_protected_resource_metadata(
    State(state): State<XetState>,
) -> Json<crate::services::oauth::ProtectedResourceMetadata> {
    let mut metadata = crate::services::oauth::protected_resource_metadata(
        &state.auth.resource_url,
        &state.auth.oauth_issuer_url,
    );
    metadata.resource_name = Some("HyprStream HuggingFace-XET CAS API".to_owned());
    Json(metadata)
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
    axum::extract::Extension(user): axum::extract::Extension<AuthenticatedUser>,
) -> Response {
    let subject = Subject::new(user.user);
    let mount = CasMount::with_authorizer(
        state.store.clone(),
        DedupDomain::local_default(),
        // #1094 plane #1: enforcing authorizer behind the explicit day-one
        // bootstrap grant ("any authenticated subject may read xorb X"),
        // default-deny for everything else. Never AllowAll on this route.
        BootstrapCasAuthorizer::new(),
    );
    let mut fid = match mount.walk(&["xorb", hash.as_str()], &subject).await {
        Ok(fid) => fid,
        Err(e) => return mount_error_response(e, "get_xorb: mount walk failed"),
    };
    if let Err(e) = mount.open(&mut fid, OREAD, &subject).await {
        return mount_error_response(e, "get_xorb: mount open failed");
    }
    let bytes = match mount.read(&fid, 0, u32::MAX, &subject).await {
        Ok(b) => b,
        Err(e) => return mount_error_response(e, "get_xorb: mount read failed"),
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

fn mount_error_response(err: MountError, log_message: &str) -> Response {
    match err {
        MountError::NotFound(_) => (StatusCode::NOT_FOUND, "xorb not found").into_response(),
        MountError::InvalidArgument(_) => {
            (StatusCode::BAD_REQUEST, "invalid xorb hash").into_response()
        }
        MountError::PermissionDenied(_) => (StatusCode::FORBIDDEN, "forbidden").into_response(),
        other => {
            warn!(error = %other, "{log_message}");
            (StatusCode::INTERNAL_SERVER_ERROR, "store error").into_response()
        }
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
            (Ok(start), Ok(end)) if start <= end && start <= last => RangeOutcome::Partial {
                start,
                end: end.min(last),
            },
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
    (StatusCode::NOT_IMPLEMENTED, format!("TODO(#654): {reason}")).into_response()
}

impl Spawnable for XetService {
    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        // Xet has no RequestService implementation or RPC control handler. An
        // advertised Rep endpoint would therefore be an unservable liveness lie.
        Vec::new()
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

            let audience = &self.state.auth.resource_url;
            let issuer = &self.state.auth.oauth_issuer_url;
            if !audience.starts_with(&format!("{scheme}://")) {
                warn!(
                    effective_audience = %audience,
                    served_scheme = scheme,
                    "Xet JWT audience scheme differs from the HTTP transport; tokens must use the logged effective audience"
                );
            }

            info!(
                effective_audience = %audience,
                oauth_issuer = %issuer,
                "HF-XET CAS API available at {scheme}://{addr} (get_xorb implemented; other routes 501, see #654)"
            );

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
    use hyprstream_rpc::auth::{
        CompositeKeyPair, CompositeKeySet, CompositePairRole, CompositePairState,
        InMemoryJtiBlocklist, JtiBlocklist as _,
    };
    use std::collections::HashMap;
    use tower::ServiceExt; // oneshot

    const AUDIENCE: &str = "http://localhost:6792";
    const ISSUER: &str = "http://localhost:6791";

    // A valid 64-hex (sha256-shaped) content address the store's hash parser accepts.
    const HASH: &str = "aa00bb11cc22dd33ee44ff5566778899aabbccddeeff00112233445566778899";

    struct TestIssuer {
        ml_dsa: hyprstream_rpc::crypto::pq::MlDsaSigningKey,
        ed25519: ed25519_dalek::SigningKey,
        key_set: Arc<CompositeKeySet>,
    }

    impl TestIssuer {
        fn new() -> Self {
            let (ml_dsa, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
            let ed25519 = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
            let kid = crate::auth::jwt::composite_kid(&ml_dsa_vk, &ed25519.verifying_key());
            let pair = CompositeKeyPair::verifying(
                kid,
                ml_dsa_vk,
                ed25519.verifying_key(),
                CompositePairRole::OAuth,
                CompositePairState::Active,
                0,
                i64::MAX,
            );
            let key_set = Arc::new(CompositeKeySet::default());
            key_set
                .publish(1, "xet-test-keys".to_owned(), vec![pair])
                .unwrap();
            Self {
                ml_dsa,
                ed25519,
                key_set,
            }
        }

        fn token(&self, audience: &str, jti: &str) -> String {
            let now = chrono::Utc::now().timestamp();
            let mut claims = crate::auth::jwt::Claims::new("alice".to_owned(), now, now + 3600)
                .with_issuer(ISSUER.to_owned())
                .with_audience(Some(audience.to_owned()));
            claims.jti = Some(jti.to_owned());
            crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
                &claims,
                &self.ml_dsa,
                &self.ed25519,
            )
        }
    }

    /// Build an XetState over a temp CasStore holding one xorb.
    fn test_state_with_xorb(dir: &std::path::Path, hash: &str, bytes: &[u8]) -> XetState {
        std::fs::create_dir_all(dir.join("xorbs")).unwrap();
        std::fs::write(dir.join("xorbs").join(format!("default.{hash}")), bytes).unwrap();

        let (_sk, vk) = hyprstream_rpc::crypto::generate_signing_keypair();
        let trusted_issuers = HashMap::new();
        let federation_resolver =
            Arc::new(crate::auth::FederationKeyResolver::new(&trusted_issuers));

        XetState {
            store: CasSubstrate::new(dir),
            // Write path is 501; no live registry needed to exercise reads/auth.
            registry: None,
            auth: ResourceAuthState::new(
                vk,
                AUDIENCE.to_owned(),
                ISSUER.to_owned(),
                federation_resolver,
                Arc::new(InMemoryJtiBlocklist::new()),
            ),
        }
    }

    fn with_test_issuer(mut state: XetState, issuer: &TestIssuer) -> XetState {
        state.auth.composite_key_set = Arc::clone(&issuer.key_set);
        state
    }

    fn authorized_request(uri: &str, token: &str) -> HttpRequest<Body> {
        HttpRequest::builder()
            .uri(uri)
            .header(header::AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn test_service_name() {
        assert_eq!(SERVICE_NAME, "xet");
    }

    #[test]
    fn does_not_advertise_an_unserved_rpc_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_xorb(dir.path(), HASH, b"x");
        let service = XetService::new(XetConfig::default(), TlsConfig::default(), state);
        assert!(service.registrations().is_empty());
    }

    #[test]
    fn config_has_no_second_enable_switch() {
        let serialized = serde_json::to_value(XetConfig::default()).unwrap();
        assert!(serialized.get("enabled").is_none());
    }

    #[test]
    fn configured_external_url_is_the_effective_audience() {
        let config = XetConfig {
            external_url: Some("https://xet.example.test/cas".to_owned()),
            ..XetConfig::default()
        };
        assert_eq!(config.resource_url(), "https://xet.example.test/cas");
    }

    #[tokio::test]
    async fn get_xorb_happy_path_returns_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let payload = b"xorb-payload-bytes";
        let state = test_state_with_xorb(dir.path(), HASH, payload);

        let resp = get_xorb_handler(
            State(state),
            Path(HASH.to_owned()),
            HeaderMap::new(),
            axum::extract::Extension(AuthenticatedUser {
                user: "alice".to_owned(),
                token: None,
                exp: None,
            }),
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
        let resp = get_xorb_handler(
            State(state),
            Path(HASH.to_owned()),
            headers,
            axum::extract::Extension(AuthenticatedUser {
                user: "alice".to_owned(),
                token: None,
                exp: None,
            }),
        )
        .await;

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
        assert_eq!(
            resp.headers()
                .get(header::WWW_AUTHENTICATE)
                .unwrap()
                .to_str()
                .unwrap(),
            format!("Bearer resource_metadata=\"{AUDIENCE}/.well-known/oauth-protected-resource\"")
        );
    }

    #[tokio::test]
    async fn wrong_audience_token_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let issuer = TestIssuer::new();
        let state = with_test_issuer(test_state_with_xorb(dir.path(), HASH, b"x"), &issuer);
        let token = issuer.token("https://wrong.example/resource", "wrong-aud");

        let resp = create_xet_router(state)
            .oneshot(authorized_request(&format!("/get_xorb/{HASH}/"), &token))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn revoked_jti_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let issuer = TestIssuer::new();
        let state = with_test_issuer(test_state_with_xorb(dir.path(), HASH, b"x"), &issuer);
        state.auth.jti_blocklist.revoke(
            "revoked-xet".to_owned(),
            chrono::Utc::now().timestamp() + 3600,
        );
        let token = issuer.token(AUDIENCE, "revoked-xet");

        let resp = create_xet_router(state)
            .oneshot(authorized_request(&format!("/get_xorb/{HASH}/"), &token))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn shared_subject_rate_limit_is_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let issuer = TestIssuer::new();
        let mut state = with_test_issuer(test_state_with_xorb(dir.path(), HASH, b"x"), &issuer);
        state.auth.rate_limiter = Arc::new(server_middleware::RateLimiter::new(1, 60));
        let app = create_xet_router(state);

        let first = app
            .clone()
            .oneshot(authorized_request(
                &format!("/v1/reconstructions/{HASH}"),
                &issuer.token(AUDIENCE, "rate-1"),
            ))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::NOT_IMPLEMENTED);

        let second = app
            .oneshot(authorized_request(
                &format!("/v1/reconstructions/{HASH}"),
                &issuer.token(AUDIENCE, "rate-2"),
            ))
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn protected_resource_metadata_is_public_and_xet_specific() {
        let dir = tempfile::tempdir().unwrap();
        let state = test_state_with_xorb(dir.path(), HASH, b"x");

        let resp = create_xet_router(state)
            .oneshot(
                HttpRequest::builder()
                    .uri("/.well-known/oauth-protected-resource")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let metadata: crate::services::oauth::ProtectedResourceMetadata =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(metadata.resource, AUDIENCE);
        assert_eq!(metadata.authorization_servers, vec![ISSUER]);
        assert_eq!(
            metadata.resource_name.as_deref(),
            Some("HyprStream HuggingFace-XET CAS API")
        );
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
