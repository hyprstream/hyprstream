//! `com.atproto.*` XRPC server surface — MVP public-read slice (#1112).
//!
//! HTTP XRPC endpoints served from the same Axum surface that hosts OAuth and
//! `/.well-known/atproto-did` (see `super::did_document`). `hyprstream-pds`
//! remains deliberately no-networking: this module is the serving layer that
//! calls its data functions (`Commit`/`MST`/`car::build_record_proof_car`).
//!
//! # Endpoints
//!
//! | Method | Path (under `/xrpc/`) | Notes |
//! |--------|-----------------------|-------|
//! | GET    | `com.atproto.identity.resolveHandle` | handle→DID |
//! | GET    | `com.atproto.repo.describeRepo`     | DID/handle + commit head + didDoc |
//! | GET    | `com.atproto.repo.getRecord`        | record JSON (optional `cid` pinning) |
//! | GET    | `com.atproto.sync.getRepo`          | full-repo CARv1 export (lazy stream) |
//! | GET    | `com.atproto.server.describeServer` | server DID and account-domain policy |
//! | GET    | `com.atproto.server.getServiceAuth` | protected hosted-account service JWT |
//!
//! **`createSession` remains out of scope.** Password / app-password
//! verification and credential minting require the account authority contract.
//! `getSession` is available only when an explicit native session resolver is
//! installed and is protected by the existing bearer/DPoP middleware.
//!
//! # Feature gate
//!
//! Routes are mounted only when `OAuthConfig::xrpc_read_slice` is `true`
//! (defaults to `false`). The in-process [`XrpcRepoStore`] starts empty; the
//! write path (#910) populates it with [`RepoSnapshot`]s whose [`public`]
//! flag is `true`. Only `public` snapshots are served by these endpoints.
//!
//! # Out of scope
//!
//! - `com.atproto.sync.subscribeRepos` (firehose) — issue #1112 defers it.
//! - Write path (`repo.createRecord` etc.) — sequenced with #910.
//! - `createSession` — sequenced with the account credential contract (#1113/#948).

mod durable_reads;
mod record_validation;

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    extract::{Extension, RawQuery, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use bytes::Bytes;
use futures::{stream, Stream};
use p256::ecdsa::VerifyingKey;
use rand::RngCore as _;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore};

use hyprstream_pds::atproto_cbor::AtprotoRecordKey;
use hyprstream_pds::car::{build_record_proof_car, car_block_bytes, car_header_bytes};
use hyprstream_pds::commit::Commit;
use hyprstream_pds::mst::{Node, NodeData};
use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
use hyprstream_pds::repo_authority::accept_repo_authority;
use hyprstream_pds::tid::Tid;
use hyprstream_pds::Cid;

use super::did_document::{build_did_document, AtprotoIdentity};
use super::state::OAuthState;
use super::{
    auth::{self, AuthenticatedUser},
    token_exchange::{ATPROTO_SESSION_EXCHANGE_NSID, MAX_ATPROTO_SERVICE_TOKEN_LIFETIME},
};

/// `ai.hyprstream.model` is the only collection this PDS hosts today.
pub const HOSTED_COLLECTION: &str = COLLECTION_NSID;

/// Protected hosted-PDS service-auth endpoint from the ATProto server lexicon.
pub const GET_SERVICE_AUTH_PATH: &str = "/xrpc/com.atproto.server.getServiceAuth";

const DEFAULT_SERVICE_AUTH_TTL_SECONDS: i64 = 60;
const MAX_SERVICE_AUTH_PARAMETER_BYTES: usize = 2_048;

/// Maximum number of concurrent `sync.getRepo` (full-CAR export) requests.
/// Each request streams the entire repo; bounding concurrency prevents
/// memory/CPU exhaustion from parallel full-repo exports.
pub const GET_REPO_CONCURRENCY: usize = 4;
/// Snapshot/encoding work has its own bound; slow CAR bodies cannot hold it.
const DURABLE_SNAPSHOT_CONCURRENCY: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// RepoSnapshot + XrpcRepoStore
// ─────────────────────────────────────────────────────────────────────────────

/// The five XRPC read-slice route declarations, as a sub-`Router` parameterised
/// over `Arc<OAuthState>`. This is the single source of truth for the XRPC route
/// table — `oauth::create_app` merges it conditionally on `xrpc_read_slice`,
/// and tests mount it directly. Changing the URI or handler here changes both.
pub fn xrpc_routes() -> axum::Router<Arc<OAuthState>> {
    use axum::routing::get;
    axum::Router::new()
        .route(
            "/xrpc/com.atproto.identity.resolveHandle",
            get(resolve_handle),
        )
        .route(
            "/xrpc/com.atproto.server.describeServer",
            get(describe_server),
        )
        .route("/xrpc/com.atproto.repo.describeRepo", get(describe_repo))
        .route("/xrpc/com.atproto.repo.getRecord", get(get_record))
        .route("/xrpc/com.atproto.sync.getRepo", get(get_repo))
}

/// Protected standard repository write routes. These are mounted only when a
/// public writer is explicitly installed in OAuthState; the default remains
/// read-only until account/session authorization is configured.
pub fn xrpc_write_routes() -> axum::Router<Arc<OAuthState>> {
    use axum::routing::post;
    axum::Router::new().route("/xrpc/com.atproto.repo.createRecord", post(create_record))
}

/// Native account authority's answer for an authenticated `getSession` query.
/// The resolver owns handle, DID-document and account lifecycle truth; this
/// adapter never derives a handle from an unverified DID string.
#[derive(Clone, Debug, Default)]
pub struct AtprotoSessionInfo {
    pub handle: String,
    pub did_doc: Option<Value>,
    pub email: Option<String>,
    pub email_confirmed: Option<bool>,
    pub email_auth_factor: Option<bool>,
    pub active: bool,
    pub status: Option<String>,
}

/// Explicit native authority seam for standard `getSession`.
///
/// Implementations must bind the returned handle and lifecycle state to the
/// requested DID using the authoritative account records. Returning `None`
/// means the DID is not a locally hosted account. No default implementation is
/// installed, so adding this interface cannot expose account state by itself.
#[async_trait::async_trait]
pub trait AtprotoSessionResolver: Send + Sync {
    async fn resolve_session(&self, did: &str) -> anyhow::Result<Option<AtprotoSessionInfo>>;
}

/// Protected standard session read route. It is mounted only when an
/// authority-provided [`AtprotoSessionResolver`] is installed.
pub fn xrpc_session_routes() -> axum::Router<Arc<OAuthState>> {
    use axum::routing::get;
    axum::Router::new().route(
        "/xrpc/com.atproto.server.getSession",
        get(get_session),
    )
}

/// An in-memory snapshot of one repo's signed state — enough to answer the
/// public read slice (`describeRepo` / `getRecord` / `sync.getRepo`).
#[derive(Clone, Debug)]
pub struct RepoSnapshot {
    pub did: String,
    pub handle: String,
    pub commit: Commit,
    pub node_blocks: Vec<(Cid, NodeData)>,
    pub records: BTreeMap<Tid, ModelRecord>,
    pub atproto_vk: VerifyingKey,
    /// **Publication boundary.** When `true`, this snapshot is anonymously
    /// readable via the public XRPC read endpoints. When `false`, the public
    /// read handlers act as though the repo does not exist.
    pub public: bool,
}

impl RepoSnapshot {
    pub fn root_cid(&self) -> Cid {
        self.commit.data
    }

    pub fn record_by_rkey(&self, rkey: &str) -> Option<(&ModelRecord, Cid)> {
        let tid = Tid::parse(rkey).ok()?;
        let rec = self.records.get(&tid)?;
        Some((rec, rec.cid()))
    }

    pub fn record_proof_car(&self, rkey: &str) -> Option<Vec<u8>> {
        let tid = Tid::parse(rkey).ok()?;
        let rec = self.records.get(&tid)?;
        let cids: BTreeMap<Tid, Cid> = self.records.iter().map(|(t, r)| (*t, r.cid())).collect();
        let tree = Node::from_records(HOSTED_COLLECTION, &cids);
        let proof = tree.proof(HOSTED_COLLECTION, &tid)?;
        Some(build_record_proof_car(
            &self.commit,
            &proof,
            &self.node_blocks,
            rec,
        ))
    }

    pub fn record_json(&self, rkey: &str, rec: &ModelRecord) -> Value {
        let uri = format!("at://{}/{HOSTED_COLLECTION}/{rkey}", self.did);
        json!({
            "uri": uri,
            "cid": rec.cid().to_string(),
            "value": {
                "$type": HOSTED_COLLECTION,
                "repo": rec.repo,
                "currentOid": rec.current_oid,
                "createdAt": rec.created_at,
            }
        })
    }

    fn did_doc(&self, issuer_url: &str) -> Value {
        let atproto = AtprotoIdentity {
            p256_vk: &self.atproto_vk,
            handle: &self.handle,
            drain: None,
            lead: None,
        };
        build_did_document(&self.did, issuer_url, &[], Some(&atproto), &[], None, None)
    }
}

/// In-process registry of hosted repos, keyed by DID.
#[derive(Debug)]
pub struct XrpcRepoStore {
    by_did: RwLock<BTreeMap<String, Arc<RepoSnapshot>>>,
    get_repo_sema: Arc<Semaphore>,
    snapshot_work_sema: Arc<Semaphore>,
}

impl Default for XrpcRepoStore {
    fn default() -> Self {
        Self {
            by_did: RwLock::new(BTreeMap::new()),
            get_repo_sema: Arc::new(Semaphore::new(GET_REPO_CONCURRENCY)),
            snapshot_work_sema: Arc::new(Semaphore::new(DURABLE_SNAPSHOT_CONCURRENCY)),
        }
    }
}

impl XrpcRepoStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a snapshot for an atproto-valid repo authority.
    pub async fn put(&self, snap: RepoSnapshot) -> anyhow::Result<()> {
        accept_repo_authority(&snap.did)?;
        self.by_did
            .write()
            .await
            .insert(snap.did.clone(), Arc::new(snap));
        Ok(())
    }

    /// Look up a snapshot by DID (any visibility — for internal/admin use).
    pub async fn get(&self, did: &str) -> Option<Arc<RepoSnapshot>> {
        self.by_did.read().await.get(did).cloned()
    }

    /// Look up a **public** snapshot by DID. Non-public repos are invisible
    /// to the public read endpoints (publication boundary).
    pub async fn get_public(&self, did: &str) -> Option<Arc<RepoSnapshot>> {
        self.get(did).await.filter(|s| s.public)
    }

    /// Resolve a handle (bare hostname) to a **public** snapshot.
    pub async fn by_handle_public(&self, handle: &str) -> Option<Arc<RepoSnapshot>> {
        let guard = self.by_did.read().await;
        let mut hits = guard.values().filter(|s| s.public && s.handle == handle);
        let first = hits.next()?;
        if hits.next().is_some() {
            return None; // ambiguous — refuse
        }
        Some(Arc::clone(first))
    }

    /// Bound durable snapshot/encoding work independently of response bodies.
    /// Move ownership into the blocking task so cancellation cannot release
    /// admission while that work is still running.
    async fn acquire_snapshot_work_owned(
        &self,
    ) -> Result<OwnedSemaphorePermit, tokio::sync::AcquireError> {
        self.snapshot_work_sema.clone().acquire_owned().await
    }

    /// Acquire an **owned** concurrency permit for full-CAR export. The permit
    /// lives until the body stream is consumed or dropped — not just until the
    /// handler returns. Bounded by [`GET_REPO_CONCURRENCY`].
    pub async fn acquire_get_repo_owned(
        &self,
    ) -> Result<OwnedSemaphorePermit, tokio::sync::AcquireError> {
        self.get_repo_sema.clone().acquire_owned().await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy CAR section stream (owned-permit held until EOF/drop)
// ─────────────────────────────────────────────────────────────────────────────

/// Phase of the lazy CAR section stream.
#[derive(Clone, Debug)]
enum CarPhase {
    Header,
    Commit,
    Nodes(usize),
    Records(Arc<Vec<Tid>>, usize),
    Done,
}

/// Produce a lazy CAR section stream. Each section is encoded on-demand when
/// polled — no full-repo materialization. The `OwnedSemaphorePermit` is held
/// inside the stream's state and released when the stream is dropped (body
/// consumed or client disconnected).
fn lazy_car_stream(
    snap: Arc<RepoSnapshot>,
    permit: OwnedSemaphorePermit,
) -> impl Stream<Item = std::io::Result<Bytes>> {
    let record_tids: Arc<Vec<Tid>> = Arc::new(snap.records.keys().copied().collect());

    struct StreamState {
        phase: CarPhase,
        _permit: OwnedSemaphorePermit,
    }

    let init = StreamState {
        phase: CarPhase::Header,
        _permit: permit,
    };

    stream::unfold(init, move |mut state| {
        let snap = Arc::clone(&snap);
        let record_tids = Arc::clone(&record_tids);
        async move {
            loop {
                match &mut state.phase {
                    CarPhase::Header => {
                        state.phase = CarPhase::Commit;
                        return Some((
                            Ok(Bytes::from(car_header_bytes(&[snap.commit.cid()]))),
                            state,
                        ));
                    }
                    CarPhase::Commit => {
                        state.phase = CarPhase::Nodes(0);
                        return Some((
                            Ok(Bytes::from(car_block_bytes(
                                snap.commit.cid(),
                                &snap.commit.to_dag_cbor(),
                            ))),
                            state,
                        ));
                    }
                    CarPhase::Nodes(idx) => {
                        if *idx < snap.node_blocks.len() {
                            let (cid, data) = &snap.node_blocks[*idx];
                            *idx += 1;
                            return Some((
                                Ok(Bytes::from(car_block_bytes(*cid, &data.encode()))),
                                state,
                            ));
                        }
                        state.phase = CarPhase::Records(Arc::clone(&record_tids), 0);
                        // continue loop → Records
                    }
                    CarPhase::Records(tids, ridx) => {
                        if *ridx < tids.len() {
                            let tid = tids[*ridx];
                            *ridx += 1;
                            let rec = &snap.records[&tid];
                            return Some((
                                Ok(Bytes::from(car_block_bytes(rec.cid(), &rec.to_dag_cbor()))),
                                state,
                            ));
                        }
                        state.phase = CarPhase::Done;
                        // continue loop → Done
                    }
                    CarPhase::Done => return None,
                }
            }
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// XRPC error helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn xrpc_error_body(error: &str, message: impl Into<String>) -> Value {
    json!({ "error": error, "message": message.into() })
}

pub fn xrpc_error(status: StatusCode, error: &str, message: impl Into<String>) -> Response {
    (status, axum::Json(xrpc_error_body(error, message))).into_response()
}

pub mod errors {
    pub const INVALID_REQUEST: &str = "InvalidRequest";
    pub const BAD_EXPIRATION: &str = "BadExpiration";
    pub const ACCOUNT_NOT_FOUND: &str = "AccountNotFound";
    pub const HANDLE_NOT_FOUND: &str = "HandleNotFound";
    pub const RECORD_NOT_FOUND: &str = "RecordNotFound";
    pub const REPO_NOT_FOUND: &str = "RepoNotFound";
    pub const INTERNAL_SERVER_ERROR: &str = "InternalServerError";
}

// ─────────────────────────────────────────────────────────────────────────────
// Protected hosted-account service auth
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq)]
struct GetServiceAuthParams {
    aud: String,
    exp: Option<i64>,
    lxm: String,
}

fn parse_get_service_auth_query(raw: Option<&str>) -> Result<GetServiceAuthParams, &'static str> {
    let mut aud = None;
    let mut exp = None;
    let mut lxm = None;
    for (key, value) in url::form_urlencoded::parse(raw.unwrap_or_default().as_bytes()) {
        if key.len() > MAX_SERVICE_AUTH_PARAMETER_BYTES
            || value.len() > MAX_SERVICE_AUTH_PARAMETER_BYTES
        {
            return Err("service-auth query parameter is too long");
        }
        match key.as_ref() {
            "aud" if aud.is_none() && !value.is_empty() => aud = Some(value.into_owned()),
            "exp" if exp.is_none() && !value.is_empty() => {
                exp = Some(
                    value
                        .parse()
                        .map_err(|_| "exp must be an integer Unix timestamp")?,
                );
            }
            "lxm" if lxm.is_none() && !value.is_empty() => lxm = Some(value.into_owned()),
            "aud" | "exp" | "lxm" => {
                return Err("service-auth query parameters must be non-empty and unique");
            }
            _ => return Err("unknown service-auth query parameter"),
        }
    }
    Ok(GetServiceAuthParams {
        aud: aud.ok_or("aud is required")?,
        exp,
        // The upstream lexicon permits an unbound token, but this hosted-PDS
        // surface only mints the one method-bound assertion consumed by #1354.
        lxm: lxm.ok_or("lxm is required")?,
    })
}

fn service_auth_response(status: StatusCode, body: Value) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        axum::Json(body),
    )
        .into_response()
}

fn service_auth_error(status: StatusCode, error: &str, message: impl Into<String>) -> Response {
    service_auth_response(status, xrpc_error_body(error, message))
}

/// Mint a short-lived, method-bound ATProto service JWT using the hosted
/// account's persisted `#atproto` key.
///
/// Authentication and DPoP proof verification happen in the protected-router
/// middleware. This handler independently re-verifies the signed OAuth grant
/// before deriving scope, subject, or tenant authority from it.
pub async fn get_service_auth(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    RawQuery(raw): RawQuery,
) -> Response {
    let params = match parse_get_service_auth_query(raw.as_deref()) {
        Ok(params) => params,
        Err(message) => {
            return service_auth_error(StatusCode::BAD_REQUEST, errors::INVALID_REQUEST, message);
        }
    };
    let expected_aud = match state.atproto_service_did() {
        Some(did) => did,
        None => {
            return service_auth_error(
                StatusCode::SERVICE_UNAVAILABLE,
                errors::INTERNAL_SERVER_ERROR,
                "host service DID is unavailable",
            );
        }
    };
    if params.aud != expected_aud {
        return service_auth_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "aud must equal this PDS host DID",
        );
    }
    if params.lxm != ATPROTO_SESSION_EXCHANGE_NSID {
        return service_auth_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            format!("lxm must equal {ATPROTO_SESSION_EXCHANGE_NSID}"),
        );
    }

    let now = chrono::Utc::now().timestamp();
    let exp = match params
        .exp
        .unwrap_or_else(|| now.saturating_add(DEFAULT_SERVICE_AUTH_TTL_SECONDS))
        .checked_sub(now)
    {
        Some(ttl) if ttl > 0 && ttl <= MAX_ATPROTO_SERVICE_TOKEN_LIFETIME => now + ttl,
        _ => {
            return service_auth_error(
                StatusCode::BAD_REQUEST,
                errors::BAD_EXPIRATION,
                "exp must be in the future and no more than one hour from now",
            );
        }
    };

    let token = match user.token.as_deref() {
        Some(token) => token,
        None => {
            return service_auth_error(
                StatusCode::UNAUTHORIZED,
                errors::INVALID_REQUEST,
                "verified OAuth access token is required",
            );
        }
    };
    let claims = match auth::validate_oauth_access_token(&state, token).await {
        Ok(claims) => claims,
        Err(_) => {
            return service_auth_error(
                StatusCode::UNAUTHORIZED,
                errors::INVALID_REQUEST,
                "OAuth access token is invalid or expired",
            );
        }
    };
    if claims.sub != user.user || claims.tenant != user.verified_tenant {
        tracing::error!("service-auth middleware and handler identity claims disagree");
        return service_auth_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "OAuth identity binding is invalid",
        );
    }
    if !claims.has_scope("atproto") {
        return service_auth_error(
            StatusCode::FORBIDDEN,
            "InsufficientScope",
            "the atproto scope is required",
        );
    }
    if claims.cnf_jkt().is_none() {
        return service_auth_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "a DPoP-bound OAuth access token is required",
        );
    }
    if super::token::hosted_account_tenant_from_did(&claims.sub, state.hosted_account_zone.as_ref())
        .is_err()
    {
        return service_auth_error(
            StatusCode::BAD_REQUEST,
            errors::ACCOUNT_NOT_FOUND,
            "OAuth subject is not a hosted account on this PDS",
        );
    }
    let record_tenant = match state.hosted_account_tenant(&claims.sub).await {
        Ok(Some(tenant)) => tenant,
        Ok(None) => {
            return service_auth_error(
                StatusCode::BAD_REQUEST,
                errors::ACCOUNT_NOT_FOUND,
                "OAuth subject has no hosted account record",
            );
        }
        Err(error) => {
            tracing::error!(
                subject = %claims.sub,
                error = %error,
                "hosted account tenant verification failed"
            );
            return service_auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "hosted account verification failed",
            );
        }
    };
    if claims.tenant.as_deref() != Some(record_tenant.as_str()) {
        tracing::error!(
            subject = %claims.sub,
            "OAuth tenant does not match the authority-owned hosted account record"
        );
        return service_auth_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "OAuth hosted-account tenant binding is invalid",
        );
    }

    let mut jti_bytes = [0_u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut jti_bytes);
    let header_segment = match serde_json::to_vec(&json!({
        "alg": "ES256",
        "typ": "JWT",
        "kid": "#atproto",
    })) {
        Ok(bytes) => URL_SAFE_NO_PAD.encode(bytes),
        Err(_) => {
            return service_auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "failed to encode service JWT",
            );
        }
    };
    let claims_segment = match serde_json::to_vec(&json!({
        "iss": claims.sub,
        "aud": expected_aud,
        "iat": now,
        "exp": exp,
        "lxm": ATPROTO_SESSION_EXCHANGE_NSID,
        "jti": URL_SAFE_NO_PAD.encode(jti_bytes),
    })) {
        Ok(bytes) => URL_SAFE_NO_PAD.encode(bytes),
        Err(_) => {
            return service_auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "failed to encode service JWT",
            );
        }
    };
    let signing_input = format!("{header_segment}.{claims_segment}");
    let Some(store) = state.hosted_account_store.as_ref() else {
        return service_auth_error(
            StatusCode::SERVICE_UNAVAILABLE,
            errors::INTERNAL_SERVER_ERROR,
            "hosted account signer is unavailable",
        );
    };
    let authority =
        hyprstream_rpc::Subject::new(hyprstream_pds_service::OAUTH_ACCOUNT_RESOLVER_SUBJECT);
    let signature = match store
        .sign_for_hosted_did(&authority, &user.user, signing_input.as_bytes())
        .await
    {
        Ok(Some(signature)) => signature,
        Ok(None) => {
            return service_auth_error(
                StatusCode::BAD_REQUEST,
                errors::ACCOUNT_NOT_FOUND,
                "OAuth subject has no hosted account signing record",
            );
        }
        Err(error) => {
            tracing::error!(
                subject = %user.user,
                error = %error,
                "hosted account service JWT signing failed"
            );
            return service_auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "hosted account signing failed",
            );
        }
    };

    service_auth_response(
        StatusCode::OK,
        json!({
            "token": format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(signature)),
        }),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Query parsing (manual — no axum Query rejection bypasses XRPC envelope)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a raw query string `key=value&key=value` into a map.
///
/// - **Bare keys** (no `=`) are inserted with an empty value — so `?since`
///   is treated as present, matching form/urlencoded semantics.
/// - **Percent-decoding**: `%XX` sequences are decoded and `+` becomes a
///   space (form semantics), so encoded DIDs/handles/rkeys resolve.
fn parse_query(raw: Option<&str>) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Some(q) = raw {
        for pair in q.split('&') {
            let (key, val) = match pair.split_once('=') {
                Some((k, v)) => (k, v),
                None => (pair, ""), // bare key → present with empty value
            };
            map.insert(percent_decode(key), percent_decode(val));
        }
    }
    map
}

/// Minimal percent-decoder: `%XX` → byte, `+` → space.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                    out.push(hi * 16 + lo);
                    i += 2;
                } else {
                    out.push(b'%');
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core handler logic (testable without OAuthState)
// ─────────────────────────────────────────────────────────────────────────────

/// Trusted public registry/issuer resolution, shared by reads and owned writes.
async fn resolve_handle_did(
    store: &XrpcRepoStore,
    issuer_url: &str,
    handle: &str,
) -> Option<String> {
    let handle = handle.to_ascii_lowercase();
    if !record_validation::valid_handle(&handle) {
        return None;
    }
    if let Some(snap) = store.by_handle_public(&handle).await {
        return Some(snap.did.clone());
    }
    if let Ok(origin) = url::Url::parse(issuer_url) {
        if origin.host_str() == Some(handle.as_str()) {
            return super::state::atproto_service_did_for_origin(issuer_url);
        }
    }
    None
}

async fn owned_public_writer(
    state: &OAuthState,
    repo: &str,
) -> Option<Arc<crate::services::public_repo::PublicRepoWriter>> {
    let writer = state.public_repo_writer.as_ref()?;
    let did = if repo.starts_with("did:") {
        repo.to_owned()
    } else {
        resolve_handle_did(&state.xrpc_repos, &state.issuer_url, repo).await?
    };
    (did == writer.did()).then(|| Arc::clone(writer))
}

async fn resolve_handle_core(store: &XrpcRepoStore, issuer_url: &str, handle: &str) -> Response {
    if let Some(did) = resolve_handle_did(store, issuer_url, handle).await {
        return axum::Json(json!({ "did": did })).into_response();
    }
    xrpc_error(
        StatusCode::BAD_REQUEST,
        errors::HANDLE_NOT_FOUND,
        format!("handle {handle:?} not found"),
    )
}

async fn describe_repo_core(store: &XrpcRepoStore, issuer_url: &str, repo: &str) -> Response {
    let snap = match lookup_public_snapshot(store, repo).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {repo:?} is not hosted by this PDS"),
            );
        }
    };
    let collections: Vec<&str> = if snap.records.is_empty() {
        Vec::new()
    } else {
        vec![HOSTED_COLLECTION]
    };
    let did_doc = snap.did_doc(issuer_url);
    axum::Json(json!({
        "handle": snap.handle,
        "did": snap.did,
        "didDoc": did_doc,
        "collections": collections,
        "handleIsCorrect": true,
    }))
    .into_response()
}

async fn get_record_core(
    store: &XrpcRepoStore,
    repo: &str,
    collection: &str,
    rkey: &str,
    cid: Option<&str>,
) -> Response {
    if collection != HOSTED_COLLECTION {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::RECORD_NOT_FOUND,
            format!("collection {collection:?} is not hosted"),
        );
    }
    if rkey.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "rkey is required",
        );
    }
    let snap = match lookup_public_snapshot(store, repo).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {repo:?} is not hosted by this PDS"),
            );
        }
    };
    let Some((rec, record_cid)) = snap.record_by_rkey(rkey) else {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::RECORD_NOT_FOUND,
            format!("record {collection}/{rkey} not found"),
        );
    };
    if let Some(requested_cid) = cid {
        if requested_cid != record_cid.to_string() {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::RECORD_NOT_FOUND,
                format!("record {collection}/{rkey} does not match cid {requested_cid:?}"),
            );
        }
    }
    axum::Json(snap.record_json(rkey, rec)).into_response()
}

async fn get_repo_core(store: &XrpcRepoStore, did: &str, since_present: bool) -> Response {
    if since_present {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "since (revision delta) is not yet supported; omit for full export",
        );
    }
    let snap = match store.get_public(did).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {did:?} is not hosted by this PDS"),
            );
        }
    };
    // Acquire owned permit — held inside the body stream until EOF/drop.
    let permit = match store.acquire_get_repo_owned().await {
        Ok(p) => p,
        Err(_) => {
            return xrpc_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "concurrency limiter closed",
            );
        }
    };
    let body_stream = lazy_car_stream(snap, permit);
    let body = axum::body::Body::from_stream(body_stream);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.ipld.car")],
        body,
    )
        .into_response()
}

async fn lookup_public_snapshot(store: &XrpcRepoStore, key: &str) -> Option<Arc<RepoSnapshot>> {
    if key.starts_with("did:") {
        store.get_public(key).await
    } else {
        store.by_handle_public(key).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Axum handler wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the standard server capability and account-domain description.
///
/// The service DID is derived from the configured issuer origin. Account
/// domains are advertised only when an authority-owned [`AccountZone`] is
/// installed; an unconfigured zone yields an empty list rather than a guessed
/// or operator-wide wildcard. Account creation remains unavailable until its
/// provisioning contract is installed, so invite-code requirement is kept
/// fail-closed.
pub async fn describe_server(State(state): State<Arc<OAuthState>>) -> Response {
    let Some(did) = state.atproto_service_did() else {
        return xrpc_error(
            StatusCode::SERVICE_UNAVAILABLE,
            errors::INTERNAL_SERVER_ERROR,
            "server DID is unavailable",
        );
    };
    let available_user_domains = state
        .hosted_account_zone
        .as_ref()
        .map(|zone| format!(".{}", zone.apex()))
        .into_iter()
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        axum::Json(json!({
            "did": did,
            "availableUserDomains": available_user_domains,
            "inviteCodeRequired": true,
            "phoneVerificationRequired": false,
        })),
    )
        .into_response()
}

/// Return the current authenticated ATProto session from the native account
/// authority. The bearer/DPoP middleware authenticates the request; this
/// handler revalidates the token and scope before consulting the resolver.
pub async fn get_session(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
) -> Response {
    let Some(token) = user.token.as_deref() else {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "verified OAuth access token is required",
        );
    };
    let claims = match auth::validate_oauth_access_token(&state, token).await {
        Ok(claims) => claims,
        Err(_) => {
            return xrpc_error(
                StatusCode::UNAUTHORIZED,
                errors::INVALID_REQUEST,
                "OAuth access token is invalid or expired",
            );
        }
    };
    if !claims.has_scope("atproto") {
        return xrpc_error(
            StatusCode::FORBIDDEN,
            "InsufficientScope",
            "the atproto scope is required",
        );
    }
    if claims.sub != user.user || claims.tenant != user.verified_tenant {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "OAuth identity binding is invalid",
        );
    }
    let Some(resolver) = state.atproto_session_resolver.as_ref() else {
        return xrpc_error(
            StatusCode::SERVICE_UNAVAILABLE,
            errors::INTERNAL_SERVER_ERROR,
            "native ATProto session resolver is not configured",
        );
    };
    let info = match resolver.resolve_session(&claims.sub).await {
        Ok(Some(info)) => info,
        Ok(None) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::ACCOUNT_NOT_FOUND,
                "account is not hosted by this PDS",
            );
        }
        Err(error) => {
            tracing::error!(%error, did = %claims.sub, "ATProto session resolver failed");
            return xrpc_error(
                StatusCode::SERVICE_UNAVAILABLE,
                errors::INTERNAL_SERVER_ERROR,
                "native account state is unavailable",
            );
        }
    };
    if info.handle.is_empty() || info.handle.chars().any(char::is_whitespace) {
        tracing::error!(did = %claims.sub, "ATProto session resolver returned invalid handle");
        return xrpc_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            errors::INTERNAL_SERVER_ERROR,
            "native account returned an invalid handle",
        );
    }
    if let Some(status) = info.status.as_deref() {
        if !matches!(status, "takendown" | "suspended" | "deactivated") {
            tracing::error!(did = %claims.sub, %status, "ATProto session resolver returned invalid status");
            return xrpc_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "native account returned an invalid status",
            );
        }
    }
    let mut body = json!({
        "handle": info.handle,
        "did": claims.sub,
        "active": info.active,
    });
    let Some(object) = body.as_object_mut() else {
        return xrpc_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            errors::INTERNAL_SERVER_ERROR,
            "session response construction failed",
        );
    };
    if let Some(value) = info.did_doc {
        object.insert("didDoc".to_owned(), value);
    }
    if let Some(value) = info.email {
        object.insert("email".to_owned(), Value::String(value));
    }
    if let Some(value) = info.email_confirmed {
        object.insert("emailConfirmed".to_owned(), Value::Bool(value));
    }
    if let Some(value) = info.email_auth_factor {
        object.insert("emailAuthFactor".to_owned(), Value::Bool(value));
    }
    if let Some(value) = info.status {
        object.insert("status".to_owned(), Value::String(value));
    }
    (StatusCode::OK, axum::Json(body)).into_response()
}

pub async fn resolve_handle(
    State(state): State<Arc<OAuthState>>,
    RawQuery(raw): RawQuery,
) -> Response {
    let params = parse_query(raw.as_deref());
    let handle = match params.get("handle") {
        Some(h) if !h.is_empty() => h.trim(),
        _ => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "handle query parameter is required",
            );
        }
    };
    resolve_handle_core(&state.xrpc_repos, &state.issuer_url, handle).await
}

pub async fn describe_repo(
    State(state): State<Arc<OAuthState>>,
    RawQuery(raw): RawQuery,
) -> Response {
    let params = parse_query(raw.as_deref());
    let key = match params.get("repo") {
        Some(r) if !r.is_empty() => r.trim(),
        _ => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "repo query parameter is required",
            );
        }
    };
    if let Some(writer) = owned_public_writer(&state, key).await {
        return durable_reads::describe_repo(&state, writer).await;
    }
    describe_repo_core(&state.xrpc_repos, &state.issuer_url, key).await
}

pub async fn get_record(State(state): State<Arc<OAuthState>>, RawQuery(raw): RawQuery) -> Response {
    let params = parse_query(raw.as_deref());
    let collection = params
        .get("collection")
        .map(std::string::String::as_str)
        .unwrap_or("");
    let rkey = params
        .get("rkey")
        .map(std::string::String::as_str)
        .unwrap_or("");
    let repo = match params.get("repo") {
        Some(r) if !r.is_empty() => r.trim(),
        _ => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "repo query parameter is required",
            );
        }
    };
    let cid = params.get("cid").map(|c| c.trim());
    if let Some(writer) = owned_public_writer(&state, repo).await {
        return durable_reads::get_record(&state.xrpc_repos, writer, collection, rkey, cid).await;
    }
    get_record_core(&state.xrpc_repos, repo, collection, rkey, cid).await
}

pub async fn get_repo(State(state): State<Arc<OAuthState>>, RawQuery(raw): RawQuery) -> Response {
    let params = parse_query(raw.as_deref());
    let did = match params.get("did") {
        Some(d) if !d.is_empty() => d.trim(),
        _ => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "did query parameter is required",
            );
        }
    };
    // Reject since by PRESENCE (not just non-empty) — ?since= and ?since=x both 400.
    let since_present = params.contains_key("since");
    if let Some(writer) = state
        .public_repo_writer
        .as_ref()
        .filter(|writer| writer.did() == did)
    {
        return durable_reads::get_repo(&state.xrpc_repos, Arc::clone(writer), since_present).await;
    }
    get_repo_core(&state.xrpc_repos, did, since_present).await
}

/// `com.atproto.repo.createRecord` for the first native-authorized public
/// posting slice. The route is deliberately opt-in: OAuthState must carry a
/// PublicRepoWriter, and the normal bearer/DPoP middleware must have inserted
/// AuthenticatedUser before this handler runs.
pub async fn create_record(
    State(state): State<Arc<OAuthState>>,
    Extension(user): Extension<AuthenticatedUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    const MAX_BODY_BYTES: usize = 1_048_576;
    if body.len() > MAX_BODY_BYTES {
        return xrpc_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            errors::INVALID_REQUEST,
            "record body exceeds 1 MiB",
        );
    }
    let Some(writer) = state.public_repo_writer.as_ref() else {
        return xrpc_error(
            StatusCode::SERVICE_UNAVAILABLE,
            errors::INTERNAL_SERVER_ERROR,
            "public repository writer is not configured",
        );
    };
    let Some(token) = user.token.as_deref() else {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "verified OAuth access token is required",
        );
    };
    let claims = match auth::validate_oauth_access_token(&state, token).await {
        Ok(claims) => claims,
        Err(_) => {
            return xrpc_error(
                StatusCode::UNAUTHORIZED,
                errors::INVALID_REQUEST,
                "OAuth access token is invalid or expired",
            )
        }
    };
    if !claims.has_scope("atproto") {
        return xrpc_error(
            StatusCode::FORBIDDEN,
            "InsufficientScope",
            "the atproto scope is required",
        );
    }
    if claims.sub != user.user || claims.tenant != user.verified_tenant {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "OAuth identity binding is invalid",
        );
    }
    // The protected router has verified the matching proof, ath, nonce, and
    // replay state for bound tokens. Unbound Bearer tokens remain valid for
    // other OAuth routes, but cannot authorize this atproto write endpoint.
    if claims.cnf_jkt().is_none() {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_REQUEST,
            "a DPoP-bound OAuth access token is required",
        );
    }
    let input: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "request body must be valid JSON",
            )
        }
    };
    let object = match input.as_object() {
        Some(object) => object,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "request body must be an object",
            )
        }
    };
    let expected_prev = match object.get("swapCommit") {
        None => None,
        Some(Value::String(value)) if !value.is_empty() => Some(value.as_str()),
        Some(_) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "swapCommit must be a non-empty CID string when present",
            )
        }
    };
    let validate = match object.get("validate") {
        None => true, // Both collections in this posting slice have known schemas.
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "validate must be a boolean when present",
            )
        }
    };
    let return_record = match object.get("returnRecord") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "returnRecord must be a boolean when present",
            )
        }
    };
    let mut idempotency_keys = headers.get_all("Idempotency-Key").iter();
    let request_id = match idempotency_keys.next() {
        None => None,
        Some(value) => {
            let value = match value.to_str() {
                Ok(value) if !value.is_empty() => value,
                _ => {
                    return xrpc_error(
                        StatusCode::BAD_REQUEST,
                        errors::INVALID_REQUEST,
                        "Idempotency-Key must be a non-empty ASCII string",
                    )
                }
            };
            if idempotency_keys.next().is_some() {
                return xrpc_error(
                    StatusCode::BAD_REQUEST,
                    errors::INVALID_REQUEST,
                    "only one Idempotency-Key may be supplied",
                );
            }
            // The native writer validates the publication request ID's length
            // and alphabet before authorization or any repository access.
            Some(value.to_owned())
        }
    };
    let repo = match object.get("repo").and_then(Value::as_str) {
        Some(repo) if !repo.is_empty() => repo,
        _ => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "repo is required",
            )
        }
    };
    let repo = if repo.starts_with("did:") {
        Some(repo.to_owned())
    } else {
        resolve_handle_did(&state.xrpc_repos, &state.issuer_url, repo).await
    };
    if repo.as_deref() != Some(writer.did()) {
        return xrpc_error(
            StatusCode::FORBIDDEN,
            "AuthRequired",
            "the request repo is not owned by this writer",
        );
    }
    let collection = match object.get("collection").and_then(Value::as_str) {
        Some(collection)
            if matches!(collection, "app.bsky.feed.post" | "app.bsky.actor.profile") =>
        {
            collection
        }
        Some(_) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "collection is outside the enabled posting slice",
            )
        }
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "collection is required",
            )
        }
    };
    let rkey = match (collection, object.get("rkey")) {
        ("app.bsky.actor.profile", None) => AtprotoRecordKey::new("self").map(Some),
        ("app.bsky.actor.profile", Some(Value::String(value))) if value == "self" => {
            AtprotoRecordKey::new(value.clone()).map(Some)
        }
        ("app.bsky.actor.profile", _) => Err(anyhow::anyhow!("profile rkey must be self")),
        (_, None) => Ok(None),
        (_, Some(Value::String(value)))
            if Tid::parse(value).is_ok_and(|tid| tid.encode() == *value) =>
        {
            // Reject alternate encodings instead of changing the record path.
            AtprotoRecordKey::new(value.clone()).map(Some)
        }
        _ => Err(anyhow::anyhow!("a valid TID rkey is required when present")),
    };
    let rkey = match rkey {
        Ok(rkey) => rkey,
        Err(error) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                error.to_string(),
            )
        }
    };
    let record_value = match object.get("record") {
        Some(record) => record,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "record is required",
            )
        }
    };
    if validate {
        if let Err(error) = record_validation::validate(collection, record_value) {
            let (status, code, message) = match error {
                record_validation::Error::Invalid => (
                    StatusCode::BAD_REQUEST,
                    errors::INVALID_REQUEST,
                    "record does not match its known schema",
                ),
                record_validation::Error::SchemaUnavailable => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    errors::INTERNAL_SERVER_ERROR,
                    "record schema validation is unavailable",
                ),
            };
            return xrpc_error(status, code, message);
        }
    }
    let record = match crate::services::public_repo::json_to_dag_cbor(record_value) {
        Ok(record) => record,
        Err(_) => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "record contains unsupported data",
            )
        }
    };
    if crate::services::public_repo::reject_unverified_blobs(&record).is_err() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "blob storage verification is unavailable",
        );
    }
    let request_id = request_id.unwrap_or_else(|| match rkey.as_ref() {
        Some(rkey) => format!("create-{}-{}", collection.replace('.', "_"), rkey.as_str()),
        // Without a client idempotency key, distinct omitted-key requests
        // create distinct records. Key allocation itself stays in the writer.
        None => format!("create-{}", uuid::Uuid::new_v4()),
    });
    let writer = Arc::clone(writer);
    let expected_prev = expected_prev.map(str::to_owned);
    let request = crate::services::public_repo::PublicCreateRequest {
        request_id,
        principal: user.user,
        did: writer.did().to_owned(),
        collection: collection.to_owned(),
        rkey,
        value: record,
        expected_prev: None,
    };
    // Authorization remains inside the transaction before any store access.
    // Native locks, RocksDB, signing and sync writes must not occupy Tokio workers.
    let result = tokio::task::spawn_blocking(move || {
        writer.create_record_with_expected_prev_text(request, expected_prev.as_deref())
    })
    .await
    .unwrap_or_else(|error| {
        Err(crate::services::public_repo::PublicRepoWriteError::Internal(error.into()))
    });
    use crate::services::public_repo::PublicRepoWriteError;
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            let (status, code, message) = match error {
                PublicRepoWriteError::InvalidRequest(_) => (
                    StatusCode::BAD_REQUEST,
                    errors::INVALID_REQUEST,
                    "record or request parameters are invalid",
                ),
                PublicRepoWriteError::Authorization(_) => (
                    StatusCode::FORBIDDEN,
                    "AuthRequired",
                    "public repository publication is not authorized",
                ),
                PublicRepoWriteError::RecordAlreadyExists => (
                    StatusCode::CONFLICT,
                    "RecordAlreadyExists",
                    "record key already exists",
                ),
                PublicRepoWriteError::InvalidSwap => (
                    StatusCode::CONFLICT,
                    "InvalidSwap",
                    "swapCommit does not match the repository head",
                ),
                PublicRepoWriteError::Internal(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "InternalServerError",
                    "public repository operation failed",
                ),
            };
            return xrpc_error(status, code, message);
        }
    };
    let mut response = json!({"uri": result.uri, "cid": result.cid.to_string()});
    if return_record {
        response["value"] = record_value.clone();
    }
    (StatusCode::OK, axum::Json(response)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use p256::ecdsa::SigningKey;
    use rand::rngs::OsRng;

    fn sample_snapshot(did: &str, handle: &str, public: bool) -> RepoSnapshot {
        let signing = SigningKey::random(&mut OsRng);
        let vk = VerifyingKey::from(&signing);
        let repo_at_uri = format!("at://{did}");
        let rec = ModelRecord::new(
            &repo_at_uri,
            "bafyreiexamplecurrentoid000000000000000000000000000a",
            "2026-07-19T00:00:00.000Z",
        )
        .expect("valid record");
        let tid = Tid::now();
        let mut record_cids: BTreeMap<Tid, Cid> = BTreeMap::new();
        record_cids.insert(tid, rec.cid());
        let mut records: BTreeMap<Tid, ModelRecord> = BTreeMap::new();
        records.insert(tid, rec);
        let tree = Node::from_records(HOSTED_COLLECTION, &record_cids);
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
        let root_cid = tree.root_cid();
        use hyprstream_pds::commit::UnsignedCommit;
        let unsigned = UnsignedCommit::new(did.to_owned(), root_cid, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing);
        commit.verify(&vk).expect("self-signed commit verifies");
        RepoSnapshot {
            did: did.to_owned(),
            handle: handle.to_owned(),
            commit,
            node_blocks,
            records,
            atproto_vk: vk,
            public,
        }
    }

    /// Read a response body as a JSON Value.
    async fn body_json(resp: Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ── Finding 1: lazy CAR + owned permit held until EOF ───────────────────

    #[tokio::test]
    async fn lazy_car_stream_produces_valid_car() {
        let snap = Arc::new(sample_snapshot(
            "did:web:h.example.com",
            "h.example.com",
            true,
        ));
        let store = XrpcRepoStore::new();
        let permit = store.acquire_get_repo_owned().await.unwrap();
        let strm = lazy_car_stream(Arc::clone(&snap), permit);
        // Collect all sections and concatenate.
        use futures::StreamExt;
        let mut collected: Vec<u8> = Vec::new();
        futures::pin_mut!(strm);
        while let Some(chunk) = strm.next().await {
            collected.extend_from_slice(&chunk.unwrap());
        }
        // Parse as a CARv1 and verify the commit root.
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1(&collected).unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], snap.commit.cid());
        assert!(blocks.len() >= 3); // commit + MST + record
    }

    #[tokio::test]
    async fn semaphore_owned_permit_held_until_stream_drop() {
        // With concurrency = 4, acquire 4 permits. The 5th must block.
        // Dropping one permit unblocks the 5th.
        let store = XrpcRepoStore::new();
        let mut permits: Vec<OwnedSemaphorePermit> = Vec::new();
        for _ in 0..GET_REPO_CONCURRENCY {
            permits.push(store.acquire_get_repo_owned().await.unwrap());
        }
        // The N+1th acquire must not complete immediately.
        let next = store.acquire_get_repo_owned();
        tokio::pin!(next);
        tokio::select! {
            _ = &mut next => panic!("N+1th permit acquired before capacity freed"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
        }
        // Drop one permit — the N+1th should now complete.
        permits.pop();
        let _extra = next.await.expect("N+1th permit resolves after drop");
    }

    #[tokio::test]
    async fn lazy_car_stream_releases_permit_on_drop() {
        // The permit held inside the lazy stream must be released when the
        // stream is dropped (e.g. client disconnects mid-body).
        let store = XrpcRepoStore::new();
        // Exhaust all permits.
        let mut held: Vec<OwnedSemaphorePermit> = Vec::new();
        for _ in 0..GET_REPO_CONCURRENCY {
            held.push(store.acquire_get_repo_owned().await.unwrap());
        }
        let snap = Arc::new(sample_snapshot(
            "did:web:h.example.com",
            "h.example.com",
            true,
        ));
        // Acquire one more for the stream.
        let stream_permit = held.pop().unwrap();
        let strm = lazy_car_stream(snap, stream_permit);
        // Drop the stream without consuming it — permit must be released.
        drop(strm);
        // Now we should be able to acquire a new permit.
        let _new = store
            .acquire_get_repo_owned()
            .await
            .expect("permit available after stream dropped");
    }

    // ── Finding 2: endpoint-level tests ──────────────────────────────────────

    const ISSUER: &str = "https://h.example.com";

    #[tokio::test]
    async fn endpoint_non_public_invisible_describe_repo() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:priv.example.com",
                "priv.example.com",
                false,
            ))
            .await
            .unwrap();
        let resp = describe_repo_core(&store, ISSUER, "did:web:priv.example.com").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_non_public_invisible_get_record() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:priv.example.com",
                "priv.example.com",
                false,
            ))
            .await
            .unwrap();
        let resp = get_record_core(
            &store,
            "did:web:priv.example.com",
            HOSTED_COLLECTION,
            "anything",
            None,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_non_public_invisible_get_repo() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:priv.example.com",
                "priv.example.com",
                false,
            ))
            .await
            .unwrap();
        let resp = get_repo_core(&store, "did:web:priv.example.com", false).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_non_public_invisible_resolve_handle() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:priv.example.com",
                "priv.example.com",
                false,
            ))
            .await
            .unwrap();
        let resp = resolve_handle_core(&store, ISSUER, "priv.example.com").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::HANDLE_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_public_snapshot_visible_describe_repo() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:pub.example.com",
                "pub.example.com",
                true,
            ))
            .await
            .unwrap();
        let resp = describe_repo_core(&store, ISSUER, "did:web:pub.example.com").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["did"], "did:web:pub.example.com");
        assert_eq!(body["handleIsCorrect"], true);
        assert!(!body["collections"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn store_rejects_path_form_did_web_snapshot() {
        let store = XrpcRepoStore::new();
        let did = "did:web:accounts.example:users:alice";
        let err = store
            .put(sample_snapshot(did, "alice.example", true))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("path-form did:web"));
        assert!(store.get(did).await.is_none());
    }

    #[tokio::test]
    async fn endpoint_get_record_cid_mismatch() {
        let store = XrpcRepoStore::new();
        let snap = sample_snapshot("did:web:pub.example.com", "pub.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        store.put(snap).await.unwrap();
        let resp = get_record_core(
            &store,
            "did:web:pub.example.com",
            HOSTED_COLLECTION,
            &rkey,
            Some("bafyreiwrongcid000000000000000000000000000000000000"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::RECORD_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_get_record_cid_match_succeeds() {
        let store = XrpcRepoStore::new();
        let snap = sample_snapshot("did:web:pub.example.com", "pub.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        let cid = snap.records.values().next().unwrap().cid().to_string();
        store.put(snap).await.unwrap();
        let resp = get_record_core(
            &store,
            "did:web:pub.example.com",
            HOSTED_COLLECTION,
            &rkey,
            Some(&cid),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── Finding 3: since rejected by presence ─────────────────────────────────

    #[tokio::test]
    async fn since_non_empty_rejected() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:pub.example.com",
                "pub.example.com",
                true,
            ))
            .await
            .unwrap();
        let resp = get_repo_core(&store, "did:web:pub.example.com", true).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn get_repo_without_since_succeeds() {
        let store = XrpcRepoStore::new();
        store
            .put(sample_snapshot(
                "did:web:pub.example.com",
                "pub.example.com",
                true,
            ))
            .await
            .unwrap();
        let resp = get_repo_core(&store, "did:web:pub.example.com", false).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/vnd.ipld.car",
        );
    }

    // ── Helper / unit tests ───────────────────────────────────────────────────

    #[test]
    fn error_body_shape_is_xrpc_convention() {
        let body = xrpc_error_body("RecordNotFound", "no such record");
        assert_eq!(body["error"], "RecordNotFound");
        assert_eq!(body["message"], "no such record");
        assert_eq!(body.as_object().unwrap().len(), 2);
    }

    #[test]
    fn parse_query_empty_value_since_is_present() {
        // ?since= → key "since" present with value "".
        let q = parse_query(Some("did=did:web:x&since="));
        assert!(q.contains_key("since"));
        assert_eq!(q.get("since").map(std::string::String::as_str), Some(""));
    }

    #[test]
    fn parse_query_bare_key_since_is_present() {
        // ?since (no =) → key "since" present with value "".
        let q = parse_query(Some("did=did:web:x&since"));
        assert!(q.contains_key("since"));
        assert_eq!(q.get("since").map(std::string::String::as_str), Some(""));
    }

    #[test]
    fn parse_query_percent_decodes_values() {
        // did%3Aweb%3Ax → did:web:x (colons percent-encoded).
        let q = parse_query(Some("repo=did%3Aweb%3Ax&collection=ai.hyprstream.model"));
        assert_eq!(
            q.get("repo").map(std::string::String::as_str),
            Some("did:web:x")
        );
        assert_eq!(
            q.get("collection").map(std::string::String::as_str),
            Some("ai.hyprstream.model")
        );
    }

    #[test]
    fn parse_query_plus_becomes_space() {
        let q = parse_query(Some("handle=foo+bar"));
        assert_eq!(
            q.get("handle").map(std::string::String::as_str),
            Some("foo bar")
        );
    }

    #[test]
    fn parse_query_extracts_key_value_pairs() {
        let q = parse_query(Some(
            "repo=did:web:x&collection=ai.hyprstream.model&rkey=abc",
        ));
        assert_eq!(
            q.get("repo").map(std::string::String::as_str),
            Some("did:web:x")
        );
        assert_eq!(
            q.get("collection").map(std::string::String::as_str),
            Some("ai.hyprstream.model")
        );
    }

    #[test]
    fn service_auth_query_is_strict_and_method_bound() {
        assert_eq!(
            parse_get_service_auth_query(Some(
                "aud=did%3Aweb%3Apds.example.test&lxm=ai.hyprstream.identity.exchangeSession"
            ))
            .unwrap(),
            GetServiceAuthParams {
                aud: "did:web:pds.example.test".to_owned(),
                exp: None,
                lxm: ATPROTO_SESSION_EXCHANGE_NSID.to_owned(),
            }
        );
        for malformed in [
            "",
            "aud=did:web:pds.example.test",
            "lxm=ai.hyprstream.identity.exchangeSession",
            "aud=did:web:pds.example.test&aud=did:web:other&lxm=ai.hyprstream.identity.exchangeSession",
            "aud=did:web:pds.example.test&lxm=ai.hyprstream.identity.exchangeSession&extra=1",
            "aud=did:web:pds.example.test&lxm=ai.hyprstream.identity.exchangeSession&exp=tomorrow",
        ] {
            assert!(
                parse_get_service_auth_query(Some(malformed)).is_err(),
                "query should fail closed: {malformed}"
            );
        }
    }

    // ── Router-mounted tests (findings 1 + 2) ─────────────────────────────────

    use crate::config::OAuthConfig;
    use crate::services::{DiscoveryClient, PolicyClient};
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::Router;
    use tower::ServiceExt; // oneshot

    /// Build a real OAuthState with dummy PolicyClient/DiscoveryClient (LazyUdsTransport
    /// pointing at /dev/null — never opened). Seeds a public + private snapshot.
    async fn build_test_state(xrpc_enabled: bool) -> Arc<OAuthState> {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let key = ed25519_dalek::SigningKey::from_bytes(&[0x76; 32]);
        let vk = ed25519_dalek::SigningKey::from_bytes(&[0x73; 32]).verifying_key();
        let dummy = std::path::PathBuf::from("/dev/null/xrpc-test.sock");

        let mk_client = || {
            let rpc = RpcClientImpl::new(
                LocalSigner::new(key.clone()),
                LazyUdsTransport::new(dummy.clone()),
                Some(vk),
            )
            .with_response_verify_policy(hyprstream_rpc::crypto::CryptoPolicy::Classical);
            Arc::new(rpc)
        };

        let mut config = OAuthConfig::default();
        config.xrpc_read_slice = xrpc_enabled;
        config.external_url = Some("https://h.example.com".to_owned());

        let state = Arc::new(OAuthState::new(
            &config,
            PolicyClient::new(mk_client()),
            DiscoveryClient::new(mk_client()),
            [0x76; 32],
        ));

        state
            .xrpc_repos
            .put(sample_snapshot(
                "did:web:pub.example.com",
                "pub.example.com",
                true,
            ))
            .await
            .unwrap();
        state
            .xrpc_repos
            .put(sample_snapshot(
                "did:web:priv.example.com",
                "priv.example.com",
                false,
            ))
            .await
            .unwrap();
        state
    }

    /// Build a router through the PRODUCTION `oauth::create_app` builder with
    /// `xrpc_read_slice=true` — exercises the real feature-gate conditional.
    async fn build_xrpc_router() -> Router {
        build_production_app(true).await
    }

    /// Build the production `create_app` with the given feature-gate value.
    async fn build_production_app(xrpc_enabled: bool) -> Router {
        use crate::config::server::CorsConfig;
        let state = build_test_state(xrpc_enabled).await;
        let cors = CorsConfig {
            enabled: false,
            ..Default::default()
        };
        crate::services::oauth::create_app(state, &cors)
    }

    /// Wrap a pre-built state in the production `create_app` (for tests that
    /// seed additional snapshots before constructing the router).
    async fn build_production_app_from_state(state: Arc<OAuthState>) -> Router {
        use crate::config::server::CorsConfig;
        let cors = CorsConfig {
            enabled: false,
            ..Default::default()
        };
        crate::services::oauth::create_app(state, &cors)
    }

    /// Like `sample_snapshot` but with a fixed TID so the rkey/CID are deterministic.
    fn sample_snapshot_fixed_tid(did: &str, handle: &str, public: bool) -> RepoSnapshot {
        let mut snap = sample_snapshot(did, handle, public);
        // Rebuild with a fixed TID for deterministic rkey.
        let signing = p256::ecdsa::SigningKey::random(&mut OsRng);
        let vk = VerifyingKey::from(&signing);
        let repo_at_uri = format!("at://{did}");
        let rec = ModelRecord::new(
            &repo_at_uri,
            "bafyreiexamplecurrentoid000000000000000000000000000a",
            "2026-07-19T00:00:00.000Z",
        )
        .expect("valid record");
        let fixed_tid = Tid::from_micros(1_700_000_000_000_000, 1);
        let mut record_cids: BTreeMap<Tid, Cid> = BTreeMap::new();
        record_cids.insert(fixed_tid, rec.cid());
        let mut records: BTreeMap<Tid, ModelRecord> = BTreeMap::new();
        records.insert(fixed_tid, rec);
        let tree = Node::from_records(HOSTED_COLLECTION, &record_cids);
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
        let root_cid = tree.root_cid();
        use hyprstream_pds::commit::UnsignedCommit;
        let unsigned = UnsignedCommit::new(did.to_owned(), root_cid, Tid::now(), None);
        let commit = Commit::sign(&unsigned, &signing);
        snap.commit = commit;
        snap.node_blocks = node_blocks;
        snap.records = records;
        snap.atproto_vk = vk;
        snap
    }

    async fn resp_json(resp: axum::response::Response) -> Value {
        body_json(resp).await
    }

    fn req(uri: &str) -> HttpRequest<Body> {
        HttpRequest::builder().uri(uri).body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn service_auth_is_mounted_and_protected_without_public_read_slice() {
        let app = build_production_app(false).await;
        let response = app
            .oneshot(req("/xrpc/com.atproto.server.getServiceAuth\
                 ?aud=did%3Aweb%3Ah.example.com\
                 &lxm=ai.hyprstream.identity.exchangeSession"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn router_create_record_is_absent_without_explicit_public_writer() {
        // The production builder must not expose a write endpoint until a
        // native-authorized PublicRepoWriter is explicitly installed.
        let app = build_production_app(true).await;
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/xrpc/com.atproto.repo.createRecord")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"repo":"did:web:pub.example.com","collection":"app.bsky.feed.post","rkey":"3jzfcijpj2z2a","record":{}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[derive(Default)]
    struct WriteInputGate(
        std::sync::atomic::AtomicUsize,
        parking_lot::Mutex<Option<String>>,
    );

    impl crate::services::public_repo::PublicPublicationAuthorizer for WriteInputGate {
        fn authorize(&self, _: &str, _: &str, _: &str) -> anyhow::Result<()> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(message) = self.1.lock().as_ref() {
                anyhow::bail!(message.clone());
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct WriteAccess {
        token: String,
        claims: hyprstream_rpc::auth::Claims,
        key: SigningKey,
        htu: String,
        nonce: String,
    }

    impl WriteAccess {
        fn proof_payload(&self) -> Value {
            use sha2::{Digest as _, Sha256};
            json!({
                "jti": uuid::Uuid::new_v4().to_string(),
                "htm": "POST",
                "htu": self.htu,
                "iat": chrono::Utc::now().timestamp(),
                "ath": URL_SAFE_NO_PAD.encode(Sha256::digest(self.token.as_bytes())),
                "nonce": self.nonce,
            })
        }
    }

    fn sign_write_proof(key: &SigningKey, payload: &Value) -> String {
        use p256::ecdsa::signature::Signer as _;
        let point = key.verifying_key().to_encoded_point(false);
        let header = json!({
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": {
                "kty": "EC", "crv": "P-256",
                "x": URL_SAFE_NO_PAD.encode(point.x().unwrap()),
                "y": URL_SAFE_NO_PAD.encode(point.y().unwrap()),
            }
        });
        let signing_input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap()),
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(payload).unwrap())
        );
        let signature: p256::ecdsa::Signature = key.sign(signing_input.as_bytes());
        format!(
            "{signing_input}.{}",
            URL_SAFE_NO_PAD.encode(signature.to_bytes())
        )
    }

    async fn build_write_input_fixture() -> (
        tempfile::TempDir,
        Arc<crate::services::public_repo::PublicRepoStore>,
        Arc<WriteInputGate>,
        Router,
        WriteAccess,
    ) {
        build_write_input_fixture_for("did:web:pub.example.com", "https://h.example.com").await
    }

    async fn build_write_input_fixture_for(
        did: &str,
        issuer_url: &str,
    ) -> (
        tempfile::TempDir,
        Arc<crate::services::public_repo::PublicRepoStore>,
        Arc<WriteInputGate>,
        Router,
        WriteAccess,
    ) {
        let (dir, store, gate, app, token, _) =
            build_write_input_fixture_with_admission(did, issuer_url).await;
        (dir, store, gate, app, token)
    }

    async fn build_write_input_fixture_with_admission(
        did: &str,
        issuer_url: &str,
    ) -> (
        tempfile::TempDir,
        Arc<crate::services::public_repo::PublicRepoStore>,
        Arc<WriteInputGate>,
        Router,
        WriteAccess,
        Arc<XrpcRepoStore>,
    ) {
        if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
            let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        let store =
            Arc::new(crate::services::public_repo::PublicRepoStore::open(dir.path()).unwrap());
        let gate = Arc::new(WriteInputGate::default());
        let writer = crate::services::public_repo::PublicRepoWriter::new(
            store.clone(),
            did,
            SigningKey::random(&mut OsRng),
            gate.clone(),
        )
        .unwrap();
        let mut state = build_test_state(true).await;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x19; 32]);
        let writable_state = Arc::get_mut(&mut state).unwrap();
        writable_state.issuer_url = issuer_url.to_owned();
        writable_state.verifying_key_bytes = signing_key.verifying_key().to_bytes();
        writable_state.public_repo_writer = Some(Arc::new(writer));
        let issuer = state.atproto_issuer_url();
        let now = chrono::Utc::now().timestamp();
        let key = SigningKey::random(&mut OsRng);
        let point = key.verifying_key().to_encoded_point(false);
        let jkt = super::super::dpop::DpopKey::Es256 {
            x: (*point.x().unwrap()).into(),
            y: (*point.y().unwrap()).into(),
        }
        .jkt();
        let nonce = state.issue_dpop_nonce().await;
        state.mark_dpop_client_nonced(&jkt).await;
        let htu = format!("{issuer}/xrpc/com.atproto.repo.createRecord");
        let claims = hyprstream_rpc::auth::Claims::new("xrpc-writer".to_owned(), now, now + 3600)
            .with_issuer(issuer.clone())
            .with_audience(Some(issuer))
            .with_tenant("xrpc-input-tests".to_owned())
            .with_client_id("xrpc-input-tests")
            .with_scope(Some("atproto".to_owned()))
            .with_cnf_jkt_thumbprint(jkt)
            .with_jti();
        let token = hyprstream_rpc::auth::jwt::encode(&claims, &signing_key);
        let token = WriteAccess {
            token,
            claims,
            key,
            htu,
            nonce,
        };
        let reads = state.xrpc_repos.clone();
        let app = build_production_app_from_state(state).await;
        (dir, store, gate, app, token, reads)
    }

    fn write_input(id: u64) -> Value {
        json!({
            "repo": "did:web:pub.example.com",
            "collection": "app.bsky.feed.post",
            "rkey": Tid::from_raw(id).encode(),
            "record": {
                "$type": "app.bsky.feed.post",
                "text": "input validation",
                "createdAt": "2026-09-10T00:00:00Z"
            }
        })
    }

    fn write_http_request(
        token: &WriteAccess,
        input: &Value,
        headers: HeaderMap,
    ) -> HttpRequest<Body> {
        let mut request = HttpRequest::builder()
            .method("POST")
            .uri("/xrpc/com.atproto.repo.createRecord")
            .header(header::AUTHORIZATION, format!("DPoP {}", token.token))
            .header("DPoP", sign_write_proof(&token.key, &token.proof_payload()))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(input).unwrap()))
            .unwrap();
        request.headers_mut().extend(headers);
        request
    }

    #[tokio::test]
    async fn router_create_record_requires_bound_token_and_proof_before_native_calls() {
        let (_dir, store, gate, app, access) = build_write_input_fixture().await;
        let mut unbound = access.clone();
        unbound.claims.cnf = None;
        unbound.token = hyprstream_rpc::auth::jwt::encode(
            &unbound.claims,
            &ed25519_dalek::SigningKey::from_bytes(&[0x19; 32]),
        );
        for (token, scheme, include_proof, error) in [
            (&unbound, "Bearer", false, errors::INVALID_REQUEST),
            (&unbound, "DPoP", true, errors::INVALID_REQUEST),
            (&access, "Bearer", false, "invalid_token"),
            (&access, "Bearer", true, "invalid_token"),
            (&access, "DPoP", false, "invalid_token"),
        ] {
            let mut request = write_http_request(token, &write_input(7), HeaderMap::new());
            request.headers_mut().insert(
                header::AUTHORIZATION,
                format!("{scheme} {}", token.token).parse().unwrap(),
            );
            if !include_proof {
                request.headers_mut().remove("DPoP");
            }
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            let body = body_json(response).await;
            assert_eq!(body["error"], error);
            if error == errors::INVALID_REQUEST {
                assert_eq!(
                    body["message"],
                    "a DPoP-bound OAuth access token is required"
                );
            }
            assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
            assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn router_create_record_rejects_wrong_dpop_proofs_before_native_calls() {
        let (_dir, store, gate, app, access) = build_write_input_fixture().await;
        for case in [
            "key",
            "method",
            "uri",
            "ath",
            "missing_ath",
            "nonce",
            "missing_nonce",
        ] {
            let mut payload = access.proof_payload();
            let wrong_key = SigningKey::random(&mut OsRng);
            let key = if case == "key" {
                &wrong_key
            } else {
                &access.key
            };
            let error = match case {
                "key" => "invalid_token",
                "method" => {
                    payload["htm"] = json!("GET");
                    "invalid_dpop_proof"
                }
                "uri" => {
                    payload["htu"] =
                        json!("https://other.example/xrpc/com.atproto.repo.createRecord");
                    "invalid_dpop_proof"
                }
                "ath" => {
                    payload["ath"] = json!("wrong-token-hash");
                    "invalid_dpop_proof"
                }
                "missing_ath" => {
                    payload.as_object_mut().unwrap().remove("ath");
                    "invalid_dpop_proof"
                }
                "nonce" => {
                    payload["nonce"] = json!("invalid-server-nonce");
                    "use_dpop_nonce"
                }
                "missing_nonce" => {
                    payload.as_object_mut().unwrap().remove("nonce");
                    "use_dpop_nonce"
                }
                _ => unreachable!(),
            };
            let mut headers = HeaderMap::new();
            headers.insert("DPoP", sign_write_proof(key, &payload).parse().unwrap());
            let response = app
                .clone()
                .oneshot(write_http_request(&access, &write_input(7), headers))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{case}");
            assert_eq!(body_json(response).await["error"], error, "{case}");
            assert_eq!(
                gate.0.load(std::sync::atomic::Ordering::Relaxed),
                0,
                "{case}"
            );
            assert!(
                store.snapshot("did:web:pub.example.com").unwrap().is_none(),
                "{case}"
            );
        }
    }

    #[tokio::test]
    async fn router_create_record_dpop_replay_rejected_but_fresh_retry_succeeds() {
        let (_dir, store, gate, app, access) = build_write_input_fixture().await;
        let mut headers = HeaderMap::new();
        headers.insert(
            "DPoP",
            sign_write_proof(&access.key, &access.proof_payload())
                .parse()
                .unwrap(),
        );
        headers.insert("Idempotency-Key", "dpop-retry".parse().unwrap());
        let response = app
            .clone()
            .oneshot(write_http_request(
                &access,
                &write_input(7),
                headers.clone(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let created = body_json(response).await;
        let head = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid();
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 1);

        // Replay cannot reach the authorizer, even when the body/idempotency
        // request changes. A fresh proof can retry the original operation.
        for input in [write_input(7), write_input(8)] {
            let response = app
                .clone()
                .oneshot(write_http_request(&access, &input, headers.clone()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(body_json(response).await["error"], "invalid_dpop_proof");
            assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 1);
            let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
            assert_eq!(snapshot.commit.cid(), head);
            assert_eq!(snapshot.records.len(), 1);
        }
        headers.remove("DPoP");
        let response = app
            .oneshot(write_http_request(&access, &write_input(7), headers))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await, created);
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 2);
        let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
        assert_eq!(snapshot.commit.cid(), head);
        assert_eq!(snapshot.records.len(), 1);
    }

    fn read_request(uri: &str) -> HttpRequest<Body> {
        HttpRequest::builder().uri(uri).body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn router_durable_reads_enforce_aggregate_snapshot_budget() {
        for (count, bytes) in [(257, 1), (18, 60_000)] {
            let (_dir, store, _gate, app, _token) = build_write_input_fixture().await;
            store
                .insert_snapshot_budget_fixture_for_test("did:web:pub.example.com", count, bytes)
                .unwrap();
            for uri in [
                "/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.feed.post&rkey=0000",
                "/xrpc/com.atproto.repo.describeRepo?repo=pub.example.com",
                "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
            ] {
                let response = app.clone().oneshot(read_request(uri)).await.unwrap();
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                assert_eq!(body_json(response).await, json!({"error":"InternalServerError","message":"public repository read failed"}));
            }
        }
    }

    #[tokio::test]
    async fn router_point_reads_remain_available_with_four_held_car_bodies() {
        let (_dir, _store, _gate, app, token) = build_write_input_fixture().await;
        let created = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let mut held = Vec::new();
        for _ in 0..GET_REPO_CONCURRENCY {
            held.push(
                app.clone()
                    .oneshot(read_request(
                        "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
                    ))
                    .await
                    .unwrap(),
            );
        }
        assert!(held
            .iter()
            .all(|response| response.status() == StatusCode::OK));
        // Waiting exports must acquire body admission before snapshot work.
        let mut queued_exports = Vec::new();
        for _ in 0..DURABLE_SNAPSHOT_CONCURRENCY {
            let app = app.clone();
            queued_exports.push(tokio::spawn(async move {
                app.oneshot(read_request(
                    "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
                ))
                .await
                .unwrap()
            }));
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut queued_exports[0])
                .await
                .is_err()
        );
        let mut pending = Vec::new();
        for uri in [
            format!("/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.feed.post&rkey={}", Tid::from_raw(7).encode()),
            "/xrpc/com.atproto.repo.describeRepo?repo=pub.example.com".to_owned(),
        ] {
            let app = app.clone();
            pending.push(tokio::spawn(async move { app.oneshot(read_request(&uri)).await.unwrap() }));
        }
        for request in pending {
            let response = tokio::time::timeout(std::time::Duration::from_secs(2), request)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        assert_eq!(held.len(), GET_REPO_CONCURRENCY);
        for export in queued_exports {
            export.abort();
        }
    }

    #[tokio::test]
    async fn router_snapshot_work_admission_bounds_every_durable_reader() {
        let (_dir, _store, _gate, app, token, reads) = build_write_input_fixture_with_admission(
            "did:web:pub.example.com",
            "https://h.example.com",
        )
        .await;
        let created = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let mut held = Vec::new();
        for _ in 0..DURABLE_SNAPSHOT_CONCURRENCY {
            held.push(reads.acquire_snapshot_work_owned().await.unwrap());
        }
        let mut pending = Vec::new();
        for uri in [
            format!("/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.feed.post&rkey={}", Tid::from_raw(7).encode()),
            "/xrpc/com.atproto.repo.describeRepo?repo=pub.example.com".to_owned(),
            "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com".to_owned(),
        ] {
            let app = app.clone();
            pending.push(tokio::spawn(async move { app.oneshot(read_request(&uri)).await.unwrap() }));
        }
        for request in &mut pending {
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(50), request)
                    .await
                    .is_err()
            );
        }
        drop(held.pop());
        let mut responses = Vec::new();
        for request in pending {
            let response = tokio::time::timeout(std::time::Duration::from_secs(2), request)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            responses.push(response);
        }
        // CAR body retains only export admission, not snapshot-work admission.
        assert_eq!(reads.snapshot_work_sema.available_permits(), 1);
        assert_eq!(
            reads.get_repo_sema.available_permits(),
            GET_REPO_CONCURRENCY - 1
        );
        drop(responses);
        assert_eq!(
            reads.get_repo_sema.available_permits(),
            GET_REPO_CONCURRENCY
        );
    }

    #[tokio::test]
    async fn router_describes_nonissuer_writers_without_legacy_handle_metadata() {
        for did in [
            "did:plc:abcdefghijklmnopqrstuvwx",
            "did:web:account.example.com",
        ] {
            let (_dir, _store, _gate, app, token) =
                build_write_input_fixture_for(did, "https://pds.example.com").await;
            let mut input = write_input(7);
            input["repo"] = json!(did);
            let created = app
                .clone()
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(created.status(), StatusCode::OK);
            let response = app
                .clone()
                .oneshot(read_request(&format!(
                    "/xrpc/com.atproto.repo.describeRepo?repo={did}"
                )))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response).await;
            assert_eq!(body["did"], did);
            assert_eq!(body["didDoc"]["id"], did);
            assert_eq!(body["handle"], "handle.invalid");
            assert_eq!(body["handleIsCorrect"], false);
            assert!(body["didDoc"].get("alsoKnownAs").is_none());
            assert_eq!(body["collections"], json!(["app.bsky.feed.post"]));
            // The unknown account handle must not claim the PDS issuer handle.
            input["repo"] = json!("pds.example.com");
            let foreign = app
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(foreign.status(), StatusCode::FORBIDDEN);
        }
    }

    #[tokio::test]
    async fn router_rejects_unverified_blobs_in_every_validation_mode_and_read_path() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        let blob = json!({"$type":"blob","ref":{"$link":Cid::from_raw(b"missing").to_string()},"mimeType":"image/png","size":7});
        for mode in [None, Some(true), Some(false)] {
            for extension in [
                blob.clone(),
                json!([{"nested":[blob.clone()]}]),
                json!({"$type":"com.example.future","media":blob}),
                json!({"cid":Cid::from_raw(b"missing").to_string(),"mimeType":"image/png"}),
            ] {
                let mut input = write_input(7);
                input["record"]["extension"] = extension;
                if let Some(mode) = mode {
                    input["validate"] = json!(mode);
                }
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                assert_eq!(
                    body_json(response).await,
                    json!({"error":"InvalidRequest","message":"blob storage verification is unavailable"})
                );
                assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
                assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
            }
        }
        // Model an otherwise correctly signed repository from before blob enforcement.
        store
            .insert_unverified_blob_for_test("did:web:pub.example.com")
            .unwrap();
        assert!(store
            .snapshot("did:web:pub.example.com")
            .unwrap_err()
            .to_string()
            .contains("blob storage verification"));
        for uri in [
            "/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.feed.post&rkey=legacy",
            "/xrpc/com.atproto.repo.describeRepo?repo=pub.example.com",
            "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
        ] {
            let response = app.clone().oneshot(read_request(uri)).await.unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(body_json(response).await, json!({"error":"InternalServerError","message":"public repository read failed"}));
        }
    }

    #[tokio::test]
    async fn router_retry_rejects_substituted_swap_without_rechecking_latest_head() {
        let (_dir, store, _gate, app, token) = build_write_input_fixture().await;
        let first = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let previous = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap()
            .to_string();
        let mut input = write_input(8);
        input["swapCommit"] = json!(previous);
        let second = app
            .clone()
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        let second = body_json(second).await;
        let third = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(9),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(third.status(), StatusCode::OK);
        let head = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap();
        let retry = app
            .clone()
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(body_json(retry).await, second);
        for changed in [
            Some(head.to_string()),
            Some(Cid::from_dag_cbor(b"substituted").to_string()),
            None,
        ] {
            let mut input = input.clone();
            match changed {
                Some(cid) => {
                    input["swapCommit"] = json!(cid);
                }
                None => {
                    input.as_object_mut().unwrap().remove("swapCommit");
                }
            }
            let response = app
                .clone()
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                body_json(response).await,
                json!({"error":"InvalidRequest","message":"record or request parameters are invalid"})
            );
            assert_eq!(
                store
                    .snapshot("did:web:pub.example.com")
                    .unwrap()
                    .unwrap()
                    .commit
                    .cid_atproto()
                    .unwrap(),
                head
            );
        }
    }

    #[tokio::test]
    async fn router_issuer_port_handle_uses_canonical_did_without_registry_entry() {
        let did = "did:web:pds.example.test%3A8443";
        let (_dir, store, gate, app, token) =
            build_write_input_fixture_for(did, "https://pds.example.test:8443").await;
        let resolve = app
            .clone()
            .oneshot(read_request(
                "/xrpc/com.atproto.identity.resolveHandle?handle=pds.example.test",
            ))
            .await
            .unwrap();
        assert_eq!(resolve.status(), StatusCode::OK);
        assert_eq!(body_json(resolve).await, json!({"did":did}));
        let mut input = write_input(7);
        input["repo"] = json!("pds.example.test");
        let created = app
            .clone()
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let created = body_json(created).await;
        input["repo"] = json!(did);
        let retry = app
            .clone()
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(body_json(retry).await, created);
        for repo in [
            "pds.example.test".to_owned(),
            urlencoding::encode(did).into_owned(),
        ] {
            let read = app.clone().oneshot(read_request(&format!("/xrpc/com.atproto.repo.getRecord?repo={repo}&collection=app.bsky.feed.post&rkey={}", input["rkey"].as_str().unwrap()))).await.unwrap();
            assert_eq!(read.status(), StatusCode::OK);
            assert_eq!(body_json(read).await["uri"], created["uri"]);
            let describe = app
                .clone()
                .oneshot(read_request(&format!(
                    "/xrpc/com.atproto.repo.describeRepo?repo={repo}"
                )))
                .await
                .unwrap();
            assert_eq!(describe.status(), StatusCode::OK);
            let describe = body_json(describe).await;
            assert_eq!(describe["did"], did);
            assert_eq!(describe["handle"], "pds.example.test");
            assert_eq!(describe["didDoc"]["id"], did);
        }
        assert_eq!(store.snapshot(did).unwrap().unwrap().records.len(), 1);
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn router_writes_release_tokio_worker_and_serialize_same_did() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        let response = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let head = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap()
            .to_string();
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let locked_store = store.clone();
        let owner = std::thread::spawn(move || {
            locked_store
                .with_account_lock_for_test("did:web:pub.example.com", || {
                    held_tx.send(()).unwrap();
                    let _ = release_rx.recv_timeout(std::time::Duration::from_secs(5));
                })
                .unwrap();
        });
        held_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let started = std::time::Instant::now();
        let mut writes = Vec::new();
        for id in [8, 9] {
            let mut input = write_input(id);
            input["swapCommit"] = json!(head);
            let request = write_http_request(&token, &input, HeaderMap::new());
            let app = app.clone();
            writes.push(tokio::spawn(
                async move { app.oneshot(request).await.unwrap() },
            ));
        }
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while gate.0.load(std::sync::atomic::Ordering::Relaxed) < 3 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let response = app
            .oneshot(read_request(
                "/xrpc/com.atproto.identity.resolveHandle?handle=pub.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        assert!(writes.iter().all(|write| !write.is_finished()));
        release_tx.send(()).unwrap();
        owner.join().unwrap();
        let mut statuses = Vec::new();
        for write in writes {
            statuses.push(write.await.unwrap().status().as_u16());
        }
        statuses.sort();
        assert_eq!(statuses, vec![200, 409]);
        assert_eq!(
            store
                .snapshot("did:web:pub.example.com")
                .unwrap()
                .unwrap()
                .records
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn router_create_accepts_owned_handle_and_rejects_foreign_handles() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        for repo in [
            "priv.example.com",
            "h.example.com",
            "unknown.example.com",
            "did:web:priv.example.com",
        ] {
            let mut input = write_input(7);
            input["repo"] = json!(repo);
            let response = app
                .clone()
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN, "{repo}");
            assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
            assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
        }
        let mut input = write_input(7);
        input["repo"] = json!("PUB.EXAMPLE.COM");
        let mut headers = HeaderMap::new();
        headers.insert("Idempotency-Key", "owned-handle-retry".parse().unwrap());
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &input, headers.clone()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let created = body_json(response).await;
        assert!(created["uri"]
            .as_str()
            .unwrap()
            .starts_with("at://did:web:pub.example.com/"));
        let retry = app
            .oneshot(write_http_request(&token, &write_input(7), headers))
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::OK);
        assert_eq!(body_json(retry).await, created);
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(
            store
                .snapshot("did:web:pub.example.com")
                .unwrap()
                .unwrap()
                .records
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn router_durable_creates_are_readable_without_stale_snapshot_fallback() {
        let (_dir, store, _gate, app, token) = build_write_input_fixture().await;
        let repo_uri = "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com";
        // The fixture already contains a legacy model snapshot for this DID.
        let missing = app.clone().oneshot(read_request(repo_uri)).await.unwrap();
        assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(missing).await["error"], "RepoNotFound");
        let mut input = write_input(7);
        input["record"]["extension"] = json!({"bytes":{"$bytes":"AQI="},"link":{"$link":Cid::from_dag_cbor(b"extension").to_string()}});
        let created = app
            .clone()
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let created = body_json(created).await;
        for repo in ["did:web:pub.example.com", "pub.example.com"] {
            let uri = format!("/xrpc/com.atproto.repo.getRecord?repo={repo}&collection=app.bsky.feed.post&rkey={}", input["rkey"].as_str().unwrap());
            let read = app.clone().oneshot(read_request(&uri)).await.unwrap();
            assert_eq!(read.status(), StatusCode::OK);
            let read = body_json(read).await;
            assert_eq!(read["uri"], created["uri"]);
            assert_eq!(read["cid"], created["cid"]);
            assert_eq!(read["value"], input["record"]);
        }
        let profile = json!({"repo":"pub.example.com","collection":"app.bsky.actor.profile","rkey":"self","record":{"$type":"app.bsky.actor.profile","displayName":"Durable"}});
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &profile, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let profile_read = app.clone().oneshot(read_request("/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.actor.profile&rkey=self")).await.unwrap();
        assert_eq!(profile_read.status(), StatusCode::OK);
        assert_eq!(body_json(profile_read).await["value"], profile["record"]);
        let describe_uri = "/xrpc/com.atproto.repo.describeRepo?repo=pub.example.com";
        let describe = app
            .clone()
            .oneshot(read_request(describe_uri))
            .await
            .unwrap();
        assert_eq!(describe.status(), StatusCode::OK);
        assert_eq!(
            body_json(describe).await["collections"],
            json!(["app.bsky.actor.profile", "app.bsky.feed.post"])
        );
        let exported = app.clone().oneshot(read_request(repo_uri)).await.unwrap();
        assert_eq!(exported.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(exported.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1_atproto(&bytes).unwrap();
        let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
        assert_eq!(roots, vec![snapshot.commit.cid_atproto().unwrap()]);
        for (cid, bytes) in &blocks {
            assert_eq!(*cid, Cid::from_dag_cbor(bytes));
        }
        for record in snapshot.records.values() {
            assert!(blocks
                .iter()
                .any(|(cid, bytes)| *cid == record.cid() && bytes == record.bytes()));
        }
        store
            .insert_malformed_record_for_test(
                "did:web:pub.example.com",
                "app.bsky.feed.post",
                "private-storage-context",
            )
            .unwrap();
        for uri in [repo_uri, describe_uri, "/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.actor.profile&rkey=self"] {
            let response = app.clone().oneshot(read_request(uri)).await.unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(body_json(response).await, json!({"error":"InternalServerError","message":"public repository read failed"}));
        }
    }

    #[tokio::test]
    async fn router_durable_export_holds_permit_until_body_drop() {
        let (_dir, _store, _gate, app, token) = build_write_input_fixture().await;
        let created = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::OK);
        let uri = "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com";
        let mut responses = Vec::new();
        for _ in 0..GET_REPO_CONCURRENCY {
            let response = app.clone().oneshot(read_request(uri)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            responses.push(response);
        }
        let mut pending =
            tokio::spawn(async move { app.oneshot(read_request(uri)).await.unwrap() });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut pending)
                .await
                .is_err()
        );
        drop(responses.pop());
        let response = tokio::time::timeout(std::time::Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn router_create_record_maps_input_and_conflicts_without_mutation() {
        let (_dir, store, _gate, app, token) = build_write_input_fixture().await;
        let mut invalid = write_input(7);
        invalid["record"]["$type"] = json!("private.authorization.denied");
        invalid["validate"] = json!(false); // Exercise the writer's typed input failure.
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &invalid, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            body_json(response).await,
            json!({
                "error": "InvalidRequest", "message": "record or request parameters are invalid"
            })
        );
        assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());

        let response = app
            .clone()
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let original = body_json(response).await;
        let head = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("Idempotency-Key", "another-request".parse().unwrap());
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &write_input(7), headers))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            body_json(response).await,
            json!({
                "error": "RecordAlreadyExists", "message": "record key already exists"
            })
        );
        let mut stale = write_input(8);
        stale["swapCommit"] = json!(hyprstream_pds::Cid::from_dag_cbor(b"stale").to_string());
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &stale, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            body_json(response).await,
            json!({
                "error": "InvalidSwap", "message": "swapCommit does not match the repository head"
            })
        );
        let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), head);
        assert_eq!(snapshot.records.len(), 1);
        let response = app
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await, original);
    }

    #[tokio::test]
    async fn router_create_record_authorization_errors_ignore_source_text() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        // Deliberately use words that formerly selected conflict or input status.
        for detail in [
            "private credential: CAS conflict",
            "private credential: already exists",
            "private credential",
        ] {
            *gate.1.lock() = Some(detail.to_owned());
            let response = app
                .clone()
                .oneshot(write_http_request(
                    &token,
                    &write_input(7),
                    HeaderMap::new(),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
            assert_eq!(
                body_json(response).await,
                json!({
                    "error": "AuthRequired", "message": "public repository publication is not authorized"
                })
            );
            assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
        }
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn router_create_record_internal_errors_are_sanitized() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        let private_key = "private-authorization-denied-CAS-conflict";
        store
            .insert_malformed_record_for_test(
                "did:web:pub.example.com",
                "app.bsky.feed.post",
                private_key,
            )
            .unwrap();
        let source = store
            .snapshot("did:web:pub.example.com")
            .unwrap_err()
            .to_string();
        assert!(
            source.contains(private_key),
            "fixture must exercise sensitive internal context"
        );
        let response = app
            .oneshot(write_http_request(
                &token,
                &write_input(7),
                HeaderMap::new(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body_json(response).await,
            json!({
                "error": "InternalServerError", "message": "public repository operation failed"
            })
        );
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn router_create_record_rejects_malformed_optional_fields_without_writes() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        for field in ["swapCommit", "validate", "returnRecord"] {
            let mut invalid = vec![Value::Null, json!(1), json!(""), json!([]), json!({})];
            if field == "swapCommit" {
                invalid.extend([json!(false), json!(true), json!("not-a-cid")]);
            }
            for value in invalid {
                let mut input = write_input(7);
                input[field] = value.clone();
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "{field}={value}"
                );
                assert_eq!(body_json(response).await["error"], errors::INVALID_REQUEST);
                assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
                assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
            }
        }
    }

    #[tokio::test]
    async fn router_create_record_rejects_invalid_idempotency_headers_without_writes() {
        use axum::http::HeaderValue;
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        for value in [
            HeaderValue::from_static(""),
            HeaderValue::from_static(" "),
            HeaderValue::from_static("bad/key"),
            HeaderValue::from_static("first,second"),
            HeaderValue::from_str(&"a".repeat(129)).unwrap(),
            HeaderValue::from_bytes(&[0xff]).unwrap(),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert("Idempotency-Key", value);
            let response = app
                .clone()
                .oneshot(write_http_request(&token, &write_input(7), headers))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(body_json(response).await["error"], errors::INVALID_REQUEST);
            assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
            assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
        }
        let mut headers = HeaderMap::new();
        headers.append("Idempotency-Key", HeaderValue::from_static("first"));
        headers.append("Idempotency-Key", HeaderValue::from_static("second"));
        let response = app
            .oneshot(write_http_request(&token, &write_input(7), headers))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["error"], errors::INVALID_REQUEST);
        assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
        assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn router_create_record_false_skips_schema_but_not_structural_checks() {
        let (_dir, store, _gate, app, token) = build_write_input_fixture().await;
        let mut input = write_input(7);
        input["validate"] = json!(false);
        input["returnRecord"] = json!(true);
        input["record"].as_object_mut().unwrap().remove("text");
        input["record"].as_object_mut().unwrap().remove("createdAt");
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await["value"], input["record"]);
        let head = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap();
        input["record"]["unsupported"] = json!(1.5);
        let response = app
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
        assert_eq!(snapshot.records.len(), 1);
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), head);
    }

    #[tokio::test]
    async fn router_create_record_known_schemas_validate_omitted_and_true() {
        for mode in [None, Some(true)] {
            let (_dir, store, gate, app, token) = build_write_input_fixture().await;
            let blob = json!({"$type":"blob", "ref":{"$link":hyprstream_pds::Cid::from_raw(b"image").to_string()}, "mimeType":"image/png", "size":5});
            let strong = json!({"uri":"at://did:plc:abc/app.bsky.feed.post/custom-key", "cid":hyprstream_pds::Cid::from_dag_cbor(b"record").to_string()});
            let mut post = write_input(7);
            post["record"]["text"] = json!("e\u{301}".repeat(300)); // 300 graphemes, 600 code points.
            post["record"]["langs"] = json!(["en", "i-klingon", "qaa-Zzzz-419"]);
            post["record"]["reply"] = json!({"root":strong.clone(), "parent":strong.clone()});
            post["record"]["facets"] = json!([{"index":{"byteStart":0,"byteEnd":1}, "features":[
                {"$type":"app.bsky.richtext.facet#mention", "did":"did:key:zExample"},
                {"$type":"app.bsky.richtext.facet#link", "uri":"https://example.com/path"}
            ]}]);
            post["record"]["embed"] = json!({"$type":"app.bsky.embed.images","images":[{"image":blob.clone(),"alt":"image","aspectRatio":{"width":1,"height":1}}]});
            let profile = json!({"repo":"did:web:pub.example.com", "collection":"app.bsky.actor.profile", "record":{
                "$type":"app.bsky.actor.profile", "displayName":"Profile", "pronouns":"they/them", "website":"https://example.com", "avatar":blob, "pinnedPost":strong,
                "createdAt":"1985-04-12T23:20:50.12345678912345Z"
            }});
            let mut invalid = Vec::new();
            for field in ["text", "createdAt"] {
                let mut missing = post.clone();
                missing["record"].as_object_mut().unwrap().remove(field);
                invalid.push(missing);
                for bad in [Value::Null, json!(5), json!([])] {
                    let mut bad_type = post.clone();
                    bad_type["record"][field] = bad;
                    invalid.push(bad_type);
                }
            }
            for (pointer, bad) in [
                ("/record/text", json!("x".repeat(301))),
                (
                    "/record/text",
                    json!(format!("x{}", "\u{301}".repeat(1600))),
                ),
                ("/record/createdAt", json!("2026-02-30T00:00:00Z")),
                ("/record/createdAt", json!("2026-01-01t00:00:00z")),
                ("/record/createdAt", json!("2026-01-01T00:00:00-00:00")),
                ("/record/langs", json!(["en_US"])),
                ("/record/langs", json!(["en", "en", "en", "en"])),
                ("/record/reply/root/cid", json!("private-not-a-cid")),
                (
                    "/record/reply/root/uri",
                    json!("at://invalid/app.bsky.feed.post/key"),
                ),
                ("/record/facets/0/index/byteStart", json!(-1)),
                ("/record/facets/0/features/0/did", json!("did::private")),
                (
                    "/record/facets/0/features/0/did",
                    json!("did:key:zExample%"),
                ),
                (
                    "/record/reply/root/uri",
                    json!("at://did:key:zExample%/app.bsky.feed.post/key"),
                ),
                (
                    "/record/facets/0/features/1/uri",
                    json!("https://example.com/invalid space"),
                ),
                ("/record/embed/images/0/image/mimeType", json!("text/plain")),
                ("/record/embed/images/0/image/size", json!(2000001)),
                ("/record/embed/images/0/aspectRatio/width", json!(0)),
            ] {
                let mut bad_input = post.clone();
                *bad_input.pointer_mut(pointer).unwrap() = bad;
                invalid.push(bad_input);
            }
            for (field, bad) in [
                ("displayName", json!("x".repeat(65))),
                ("description", json!("x".repeat(257))),
                ("pronouns", json!("x".repeat(21))),
                ("avatar", Value::Null),
                ("website", json!("relative/path")),
                ("displayName", json!(false)),
                ("pinnedPost", json!({})),
                (
                    "labels",
                    json!({"$type":"com.atproto.label.defs#selfLabels","values":[{}]}),
                ),
            ] {
                let mut bad_input = profile.clone();
                bad_input["record"][field] = bad;
                invalid.push(bad_input);
            }
            for mut input in invalid {
                if let Some(mode) = mode {
                    input["validate"] = json!(mode);
                }
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{input}");
                assert_eq!(
                    body_json(response).await,
                    json!({"error":"InvalidRequest","message":"record does not match its known schema"})
                );
                assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
                assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
            }
            for mut input in [post, profile] {
                record_validation::validate(
                    input["collection"].as_str().unwrap(),
                    &input["record"],
                )
                .unwrap();
                input["record"].as_object_mut().unwrap().remove("embed");
                input["record"].as_object_mut().unwrap().remove("avatar");
                if let Some(mode) = mode {
                    input["validate"] = json!(mode);
                }
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::OK,
                    "{}",
                    body_json(response).await
                );
            }
            assert_eq!(
                store
                    .snapshot("did:web:pub.example.com")
                    .unwrap()
                    .unwrap()
                    .records
                    .len(),
                2
            );
        }
    }

    #[tokio::test]
    async fn router_explicit_tids_preserve_odd_low_bits_in_all_validation_modes() {
        for mode in [None, Some(true), Some(false)] {
            let (_dir, store, gate, app, token) = build_write_input_fixture().await;
            for rkey in ["3jzfcijpj2z2b", "2222222222223"] {
                let mut input = write_input(7);
                input["rkey"] = json!(rkey);
                if let Some(mode) = mode {
                    input["validate"] = json!(mode);
                }
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let created = body_json(response).await;
                assert!(created["uri"]
                    .as_str()
                    .unwrap()
                    .ends_with(&format!("/{rkey}")));
                let read = app.clone().oneshot(read_request(&format!("/xrpc/com.atproto.repo.getRecord?repo=pub.example.com&collection=app.bsky.feed.post&rkey={rkey}"))).await.unwrap();
                assert_eq!(read.status(), StatusCode::OK);
                assert_eq!(body_json(read).await["uri"], created["uri"]);
            }
            assert_eq!(
                store
                    .snapshot("did:web:pub.example.com")
                    .unwrap()
                    .unwrap()
                    .records
                    .len(),
                2
            );
            assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 2);
        }
    }

    #[tokio::test]
    async fn router_create_record_canonical_keys_and_unknown_collections() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        let noncanonical = "kjzfcijpj2z2a";
        assert!(Tid::parse(noncanonical).is_err());
        for mode in [None, Some(true), Some(false)] {
            let mut input = write_input(7);
            if let Some(mode) = mode {
                input["validate"] = json!(mode);
            }
            let mut unknown = input.clone();
            unknown["collection"] = json!("com.example.unknown");
            unknown["record"]["$type"] = unknown["collection"].clone();
            input["rkey"] = json!(noncanonical);
            for invalid in [input, unknown] {
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &invalid, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                assert_eq!(body_json(response).await["error"], errors::INVALID_REQUEST);
                assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
                assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
            }
        }
        let mut input = write_input(7);
        input["record"]["embed"] = json!({"$type":"com.example.futureEmbed","extension":true});
        input["record"]["extension"] = json!({"value":true});
        let response = app
            .oneshot(write_http_request(&token, &input, HeaderMap::new()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK); // Open unions and extra properties stay extensible.
    }

    #[tokio::test]
    async fn router_create_record_profiles_use_self_for_explicit_and_omitted_keys() {
        for omit_key in [false, true] {
            let (_dir, store, gate, app, token) = build_write_input_fixture().await;
            let mut input = json!({
                "repo": "did:web:pub.example.com",
                "collection": "app.bsky.actor.profile",
                "record": {"$type": "app.bsky.actor.profile", "displayName": "Profile"},
                "validate": false
            });
            for invalid in [
                Value::Null,
                json!(false),
                json!(1),
                json!(""),
                json!("other"),
                json!(Tid::from_raw(7).encode()),
            ] {
                let mut malformed = input.clone();
                malformed["rkey"] = invalid;
                let response = app
                    .clone()
                    .oneshot(write_http_request(&token, &malformed, HeaderMap::new()))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
                assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
            }
            if !omit_key {
                input["rkey"] = json!("self");
            }
            let response = app
                .clone()
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let created = body_json(response).await;
            assert_eq!(
                created["uri"],
                "at://did:web:pub.example.com/app.bsky.actor.profile/self"
            );
            let response = app
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(body_json(response).await, created);
            let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
            assert_eq!(snapshot.records.len(), 1);
            assert!(snapshot.records.contains_key(&(
                "app.bsky.actor.profile".into(),
                AtprotoRecordKey::new("self").unwrap()
            )));
        }
    }

    #[tokio::test]
    async fn router_create_record_generates_post_keys_and_recovers_idempotent_results() {
        let (_dir, store, gate, app, token) = build_write_input_fixture().await;
        let mut input = write_input(7);
        input.as_object_mut().unwrap().remove("rkey");
        for invalid in [
            Value::Null,
            json!(false),
            json!(1),
            json!(""),
            json!("self"),
            json!("bad/key"),
        ] {
            let mut malformed = input.clone();
            malformed["rkey"] = invalid;
            let response = app
                .clone()
                .oneshot(write_http_request(&token, &malformed, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(store.snapshot("did:web:pub.example.com").unwrap().is_none());
            assert_eq!(gate.0.load(std::sync::atomic::Ordering::Relaxed), 0);
        }
        let mut headers = HeaderMap::new();
        headers.insert("Idempotency-Key", "generated-post".parse().unwrap());
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &input, headers.clone()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let created = body_json(response).await;
        let generated_key = created["uri"].as_str().unwrap().rsplit('/').next().unwrap();
        assert!(Tid::parse(generated_key).is_ok());
        let mut uris = std::collections::BTreeSet::new();
        uris.insert(created["uri"].as_str().unwrap().to_owned());
        for _ in 0..2 {
            let response = app
                .clone()
                .oneshot(write_http_request(&token, &input, HeaderMap::new()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let created = body_json(response).await;
            assert!(uris.insert(created["uri"].as_str().unwrap().to_owned()));
        }
        let latest = store
            .snapshot("did:web:pub.example.com")
            .unwrap()
            .unwrap()
            .commit
            .cid_atproto()
            .unwrap();
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &input, headers.clone()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await, created);
        let mut explicit_retry = input.clone();
        explicit_retry["rkey"] = json!(generated_key);
        let response = app
            .clone()
            .oneshot(write_http_request(&token, &explicit_retry, headers.clone()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        input["record"]["text"] = json!("changed content");
        let response = app
            .oneshot(write_http_request(&token, &input, headers))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let snapshot = store.snapshot("did:web:pub.example.com").unwrap().unwrap();
        assert_eq!(snapshot.records.len(), 3);
        assert_eq!(snapshot.commit.cid_atproto().unwrap(), latest);
    }

    // ── Finding 1: real capacity test through the mounted router ──────────────

    #[tokio::test]
    async fn router_capacity_n_plus_1_blocked_until_body_dropped() {
        let app = build_xrpc_router().await;
        let uri = "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com";

        // Issue N requests, RETAIN response bodies unconsumed.
        let mut held: Vec<axum::response::Response> = Vec::new();
        for _ in 0..GET_REPO_CONCURRENCY {
            let resp = app.clone().oneshot(req(uri)).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            held.push(resp);
        }

        // N+1th request must NOT complete while all permits are held in bodies.
        let n1 = app.clone().oneshot(req(uri));
        let mut n1 = Box::pin(n1);
        tokio::select! {
            _ = &mut n1 => panic!("N+1th getRepo completed before capacity freed"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
        }

        // Drop ONE held Response → its Body + embedded OwnedSemaphorePermit released.
        held.pop();

        // N+1th should now complete.
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), n1.as_mut()).await;
        assert!(
            result.is_ok(),
            "N+1th getRepo did not complete after dropping a body"
        );
        assert_eq!(result.unwrap().unwrap().status(), StatusCode::OK);
    }

    // ── Finding 2: router-mounted endpoint tests ──────────────────────────────

    #[tokio::test]
    async fn router_private_repo_invisible_describe_repo() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.repo.describeRepo?repo=did:web:priv.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn router_private_repo_invisible_get_record() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.repo.getRecord?repo=did:web:priv.example.com\
                 &collection=ai.hyprstream.model&rkey=abc",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn router_private_repo_invisible_get_repo() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.sync.getRepo?did=did:web:priv.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn router_private_repo_invisible_resolve_handle() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.identity.resolveHandle?handle=priv.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::HANDLE_NOT_FOUND);
    }

    #[tokio::test]
    async fn router_public_describe_repo_ok() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.repo.describeRepo?repo=did:web:pub.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp_json(resp).await;
        assert_eq!(body["did"], "did:web:pub.example.com");
        assert_eq!(body["handleIsCorrect"], true);
    }

    #[tokio::test]
    async fn router_describe_server_reports_service_did_and_safe_defaults() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req("/xrpc/com.atproto.server.describeServer"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp_json(resp).await;
        assert_eq!(body["did"], "did:web:h.example.com");
        assert_eq!(body["availableUserDomains"], json!([]));
        assert_eq!(body["inviteCodeRequired"], true);
        assert_eq!(body["phoneVerificationRequired"], false);
    }

    #[tokio::test]
    async fn describe_server_advertises_only_configured_account_zone() {
        let mut state = build_test_state(false).await;
        Arc::get_mut(&mut state)
            .unwrap()
            .hosted_account_zone = Some(crate::account::AccountZone::new("acct.example.com").unwrap());
        let body = resp_json(describe_server(State(state)).await).await;
        assert_eq!(body["availableUserDomains"], json!([".acct.example.com"]));
    }

    #[tokio::test]
    async fn router_get_record_nonexistent_rkey() {
        let app = build_xrpc_router().await;
        let resp = app
            .clone()
            .oneshot(req(
                "/xrpc/com.atproto.repo.getRecord?repo=did:web:pub.example.com\
                 &collection=ai.hyprstream.model&rkey=zzzzzzzzzzzz",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::RECORD_NOT_FOUND);
    }

    #[tokio::test]
    async fn router_missing_params_invalid_request() {
        let app = build_xrpc_router().await;
        // Missing repo param.
        let resp = app
            .clone()
            .oneshot(req(
                "/xrpc/com.atproto.repo.getRecord?collection=ai.hyprstream.model",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);

        // Missing handle param.
        let resp = app
            .oneshot(req("/xrpc/com.atproto.identity.resolveHandle"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn router_unknown_handle_handle_not_found() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.identity.resolveHandle?handle=does-not-exist.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::HANDLE_NOT_FOUND);
    }

    #[tokio::test]
    async fn router_since_empty_rejected() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com&since=",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn router_since_non_empty_rejected() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com\
                 &since=3whysti2w4mq2",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn router_get_repo_ok_streams_car() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req(
                "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/vnd.ipld.car",
        );
        // Consume the body fully so the permit is released.
        let bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        assert!(!bytes.is_empty());
    }

    // ── Finding 1: feature-gate matrix — all 5 routes, enabled AND disabled ────

    #[tokio::test]
    async fn router_feature_gate_disabled_all_five_routes_404() {
        let app = build_production_app(false).await;
        // All five XRPC routes must 404 when the gate is disabled.
        let routes = [
            "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
            "/xrpc/com.atproto.repo.describeRepo?repo=did:web:pub.example.com",
            "/xrpc/com.atproto.repo.getRecord?repo=did:web:pub.example.com&collection=ai.hyprstream.model&rkey=abc",
            "/xrpc/com.atproto.identity.resolveHandle?handle=pub.example.com",
            "/xrpc/com.atproto.server.describeServer",
        ];
        for uri in &routes {
            let resp = app.clone().oneshot(req(uri)).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "route {uri} should 404 when xrpc_read_slice is disabled"
            );
        }
    }

    #[tokio::test]
    async fn router_feature_gate_enabled_all_five_routes_reachable() {
        // Smoke-test: all five routes reach XRPC handlers (not 404) when enabled.
        // Detailed assertions are in the individual endpoint tests above.
        let app = build_production_app(true).await;
        let routes = [
            ("/xrpc/com.atproto.repo.describeRepo?repo=did:web:pub.example.com", StatusCode::OK),
            ("/xrpc/com.atproto.identity.resolveHandle?handle=pub.example.com", StatusCode::OK),
            ("/xrpc/com.atproto.repo.getRecord?repo=did:web:pub.example.com&collection=ai.hyprstream.model&rkey=abc", StatusCode::BAD_REQUEST), // RecordNotFound
            ("/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com", StatusCode::OK),
            ("/xrpc/com.atproto.server.describeServer", StatusCode::OK),
        ];
        for (uri, expected) in &routes {
            let resp = app.clone().oneshot(req(uri)).await.unwrap();
            assert_eq!(
                resp.status(),
                *expected,
                "route {uri} status mismatch when xrpc_read_slice is enabled"
            );
        }
    }

    #[tokio::test]
    async fn router_get_session_not_mounted_without_native_resolver() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req("/xrpc/com.atproto.server.getSession"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── Finding 2: routed CID match/mismatch + malformed query ─────────────────

    #[tokio::test]
    async fn router_get_record_cid_match_returns_record() {
        let state = build_test_state(true).await;
        let snap =
            sample_snapshot_fixed_tid("did:web:fixed.example.com", "fixed.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        let cid = snap.records.values().next().unwrap().cid().to_string();
        state.xrpc_repos.put(snap).await.unwrap();
        let app = build_production_app_from_state(state).await;
        let uri = format!(
            "/xrpc/com.atproto.repo.getRecord?repo=did:web:fixed.example.com\
             &collection={HOSTED_COLLECTION}&rkey={rkey}&cid={cid}"
        );
        let resp = app.oneshot(req(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp_json(resp).await;
        assert_eq!(body["cid"], cid);
        assert_eq!(body["value"]["$type"], HOSTED_COLLECTION);
    }

    #[tokio::test]
    async fn router_get_record_cid_mismatch_returns_record_not_found() {
        let state = build_test_state(true).await;
        let snap =
            sample_snapshot_fixed_tid("did:web:fixed.example.com", "fixed.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        state.xrpc_repos.put(snap).await.unwrap();
        let app = build_production_app_from_state(state).await;
        let uri = format!(
            "/xrpc/com.atproto.repo.getRecord?repo=did:web:fixed.example.com\
             &collection={HOSTED_COLLECTION}&rkey={rkey}\
             &cid=bafyreiwrongcid000000000000000000000000000000000000"
        );
        let resp = app.oneshot(req(&uri)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::RECORD_NOT_FOUND);
        assert!(
            body["message"]
                .as_str()
                .unwrap()
                .contains("does not match cid"),
            "message must explain the cid mismatch: {}",
            body["message"]
        );
    }

    #[tokio::test]
    async fn router_malformed_query_empty_repo() {
        let app = build_xrpc_router().await;
        // ?repo= — empty value at the wrapper boundary.
        let resp = app
            .clone()
            .oneshot(req("/xrpc/com.atproto.repo.getRecord?repo=\
                 &collection=ai.hyprstream.model&rkey=abc"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
        assert!(
            body["message"].as_str().unwrap().contains("repo"),
            "message must mention repo: {}",
            body["message"]
        );
    }

    #[tokio::test]
    async fn router_bare_since_rejected() {
        // ?since (bare key, no =) must still be caught by since-rejection.
        let app = build_xrpc_router().await;
        let resp = app
            .clone()
            .oneshot(req(
                "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com&since",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = resp_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn router_percent_encoded_did_resolves() {
        // did%3Aweb%3Apub.example.com → did:web:pub.example.com (percent-decoded).
        let app = build_xrpc_router().await;
        let resp = app
            .clone()
            .oneshot(req(
                "/xrpc/com.atproto.repo.describeRepo?repo=did%3Aweb%3Apub.example.com",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp_json(resp).await;
        assert_eq!(body["did"], "did:web:pub.example.com");
    }
}
