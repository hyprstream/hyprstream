//! `com.atproto.*` XRPC server surface — MVP public-read slice (#1112).
//!
//! HTTP XRPC endpoints served from the same Axum surface that hosts OAuth and
//! `/.well-known/atproto-did` (see `super::did_document`). `hyprstream-pds`
//! remains deliberately no-networking: this module is the serving layer that
//! calls its data functions (`Commit`/`MST`/`car::build_record_proof_car`).
//!
//! # Endpoints (public reads only)
//!
//! | Method | Path (under `/xrpc/`) | Notes |
//! |--------|-----------------------|-------|
//! | GET    | `com.atproto.identity.resolveHandle` | handle→DID |
//! | GET    | `com.atproto.repo.describeRepo`     | DID/handle + commit head + didDoc |
//! | GET    | `com.atproto.repo.getRecord`        | record JSON (optional `cid` pinning) |
//! | GET    | `com.atproto.sync.getRepo`          | full-repo CARv1 export (lazy stream) |
//!
//! **Session endpoints (`createSession`/`getSession`) are deliberately NOT in
//! this PR.** Credential verification (password / app-password) and the OAuth
//! JWT bridge belong with the #1113/#948 OAuth integration work.
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
//! - `createSession`/`getSession` — sequenced with #1113/#948.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    extract::{RawQuery, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::{stream, Stream};
use p256::ecdsa::VerifyingKey;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore};

use hyprstream_pds::car::{build_record_proof_car, car_block_bytes, car_header_bytes};
use hyprstream_pds::commit::Commit;
use hyprstream_pds::mst::{Node, NodeData};
use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
use hyprstream_pds::tid::Tid;
use hyprstream_pds::Cid;

use super::did_document::{build_did_document, issuer_authority, AtprotoIdentity};
use super::state::OAuthState;

/// `ai.hyprstream.model` is the only collection this PDS hosts today.
pub const HOSTED_COLLECTION: &str = COLLECTION_NSID;

/// Maximum number of concurrent `sync.getRepo` (full-CAR export) requests.
/// Each request streams the entire repo; bounding concurrency prevents
/// memory/CPU exhaustion from parallel full-repo exports.
pub const GET_REPO_CONCURRENCY: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// RepoSnapshot + XrpcRepoStore
// ─────────────────────────────────────────────────────────────────────────────

/// The four XRPC read-slice route declarations, as a sub-`Router` parameterised
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
        .route("/xrpc/com.atproto.repo.describeRepo", get(describe_repo))
        .route("/xrpc/com.atproto.repo.getRecord", get(get_record))
        .route("/xrpc/com.atproto.sync.getRepo", get(get_repo))
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
        let cids: BTreeMap<Tid, Cid> =
            self.records.iter().map(|(t, r)| (*t, r.cid())).collect();
        let tree = Node::from_records(HOSTED_COLLECTION, &cids);
        let proof = tree.proof(HOSTED_COLLECTION, &tid)?;
        Some(build_record_proof_car(&self.commit, &proof, &self.node_blocks, rec))
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
        };
        build_did_document(&self.did, issuer_url, &[], Some(&atproto), &[], None, None)
    }
}

/// In-process registry of hosted repos, keyed by DID.
#[derive(Debug)]
pub struct XrpcRepoStore {
    by_did: RwLock<BTreeMap<String, Arc<RepoSnapshot>>>,
    get_repo_sema: Arc<Semaphore>,
}

impl Default for XrpcRepoStore {
    fn default() -> Self {
        Self {
            by_did: RwLock::new(BTreeMap::new()),
            get_repo_sema: Arc::new(Semaphore::new(GET_REPO_CONCURRENCY)),
        }
    }
}

impl XrpcRepoStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a snapshot for a DID.
    pub async fn put(&self, snap: RepoSnapshot) {
        self.by_did.write().await.insert(snap.did.clone(), Arc::new(snap));
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

    /// Acquire an **owned** concurrency permit for full-CAR export. The permit
    /// lives until the body stream is consumed or dropped — not just until the
    /// handler returns. Bounded by [`GET_REPO_CONCURRENCY`].
    pub async fn acquire_get_repo_owned(&self) -> Result<OwnedSemaphorePermit, tokio::sync::AcquireError> {
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
                                Ok(Bytes::from(car_block_bytes(
                                    rec.cid(),
                                    &rec.to_dag_cbor(),
                                ))),
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
    pub const ACCOUNT_NOT_FOUND: &str = "AccountNotFound";
    pub const HANDLE_NOT_FOUND: &str = "HandleNotFound";
    pub const RECORD_NOT_FOUND: &str = "RecordNotFound";
    pub const REPO_NOT_FOUND: &str = "RepoNotFound";
    pub const INTERNAL_SERVER_ERROR: &str = "InternalServerError";
}

// ─────────────────────────────────────────────────────────────────────────────
// Query parsing (manual — no axum Query rejection bypasses XRPC envelope)
// ─────────────────────────────────────────────────────────────────────────────

fn parse_query(raw: Option<&str>) -> std::collections::HashMap<&str, &str> {
    let mut map = std::collections::HashMap::new();
    if let Some(q) = raw {
        for pair in q.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                map.insert(k, v);
            }
        }
    }
    map
}

// ─────────────────────────────────────────────────────────────────────────────
// Core handler logic (testable without OAuthState)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a handle to a DID via the store + issuer-derived self-handle.
async fn resolve_handle_core(
    store: &XrpcRepoStore,
    issuer_url: &str,
    handle: &str,
) -> Response {
    if let Some(snap) = store.by_handle_public(handle).await {
        return axum::Json(json!({ "did": snap.did })).into_response();
    }
    if let Some(authority) = issuer_authority(issuer_url) {
        let self_handle = authority.split(':').next().unwrap_or(&authority);
        if handle == self_handle {
            let did = format!("did:web:{authority}");
            return axum::Json(json!({ "did": did })).into_response();
        }
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

async fn lookup_public_snapshot(
    store: &XrpcRepoStore,
    key: &str,
) -> Option<Arc<RepoSnapshot>> {
    if key.starts_with("did:") {
        store.get_public(key).await
    } else {
        store.by_handle_public(key).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Axum handler wrappers
// ─────────────────────────────────────────────────────────────────────────────

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
    describe_repo_core(&state.xrpc_repos, &state.issuer_url, key).await
}

pub async fn get_record(
    State(state): State<Arc<OAuthState>>,
    RawQuery(raw): RawQuery,
) -> Response {
    let params = parse_query(raw.as_deref());
    let collection = params.get("collection").copied().unwrap_or("");
    let rkey = params.get("rkey").copied().unwrap_or("");
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
    get_record_core(&state.xrpc_repos, repo, collection, rkey, cid).await
}

pub async fn get_repo(
    State(state): State<Arc<OAuthState>>,
    RawQuery(raw): RawQuery,
) -> Response {
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
    get_repo_core(&state.xrpc_repos, did, since_present).await
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
        let snap = Arc::new(sample_snapshot("did:web:h.example.com", "h.example.com", true));
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
        let snap = Arc::new(sample_snapshot("did:web:h.example.com", "h.example.com", true));
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
        store.put(sample_snapshot("did:web:priv.example.com", "priv.example.com", false)).await;
        let resp = describe_repo_core(&store, ISSUER, "did:web:priv.example.com").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_non_public_invisible_get_record() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:priv.example.com", "priv.example.com", false)).await;
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
        store.put(sample_snapshot("did:web:priv.example.com", "priv.example.com", false)).await;
        let resp = get_repo_core(&store, "did:web:priv.example.com", false).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::REPO_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_non_public_invisible_resolve_handle() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:priv.example.com", "priv.example.com", false)).await;
        let resp = resolve_handle_core(&store, ISSUER, "priv.example.com").await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::HANDLE_NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_public_snapshot_visible_describe_repo() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:pub.example.com", "pub.example.com", true)).await;
        let resp = describe_repo_core(&store, ISSUER, "did:web:pub.example.com").await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["did"], "did:web:pub.example.com");
        assert_eq!(body["handleIsCorrect"], true);
        assert!(!body["collections"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn endpoint_get_record_cid_mismatch() {
        let store = XrpcRepoStore::new();
        let snap = sample_snapshot("did:web:pub.example.com", "pub.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        store.put(snap).await;
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
        store.put(snap).await;
        let resp =
            get_record_core(&store, "did:web:pub.example.com", HOSTED_COLLECTION, &rkey, Some(&cid))
                .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── Finding 3: since rejected by presence ─────────────────────────────────

    #[tokio::test]
    async fn since_non_empty_rejected() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:pub.example.com", "pub.example.com", true)).await;
        let resp = get_repo_core(&store, "did:web:pub.example.com", true).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], errors::INVALID_REQUEST);
    }

    #[tokio::test]
    async fn get_repo_without_since_succeeds() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:pub.example.com", "pub.example.com", true)).await;
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
    fn parse_query_rejects_empty_value_since() {
        // ?since= parses to key "since" with value "".
        let q = parse_query(Some("did=did:web:x&since="));
        assert!(q.contains_key("since"));
        assert_eq!(q.get("since").copied(), Some(""));
    }

    #[test]
    fn parse_query_extracts_key_value_pairs() {
        let q = parse_query(Some("repo=did:web:x&collection=ai.hyprstream.model&rkey=abc"));
        assert_eq!(q.get("repo").copied(), Some("did:web:x"));
        assert_eq!(q.get("collection").copied(), Some("ai.hyprstream.model"));
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
            .put(sample_snapshot("did:web:pub.example.com", "pub.example.com", true))
            .await;
        state
            .xrpc_repos
            .put(sample_snapshot("did:web:priv.example.com", "priv.example.com", false))
            .await;
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
        HttpRequest::builder()
            .uri(uri)
            .body(Body::empty())
            .unwrap()
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
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            n1.as_mut(),
        )
        .await;
        assert!(result.is_ok(), "N+1th getRepo did not complete after dropping a body");
        assert_eq!(result.unwrap().unwrap().status(), StatusCode::OK);
    }

    // ── Finding 2: router-mounted endpoint tests ──────────────────────────────

    #[tokio::test]
    async fn router_private_repo_invisible_describe_repo() {
        let app = build_xrpc_router().await;
        let resp = app
            .oneshot(req("/xrpc/com.atproto.repo.describeRepo?repo=did:web:priv.example.com"))
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
            .oneshot(req("/xrpc/com.atproto.sync.getRepo?did=did:web:priv.example.com"))
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
            .oneshot(req("/xrpc/com.atproto.identity.resolveHandle?handle=priv.example.com"))
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
            .oneshot(req("/xrpc/com.atproto.repo.describeRepo?repo=did:web:pub.example.com"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = resp_json(resp).await;
        assert_eq!(body["did"], "did:web:pub.example.com");
        assert_eq!(body["handleIsCorrect"], true);
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
            .oneshot(req("/xrpc/com.atproto.repo.getRecord?collection=ai.hyprstream.model"))
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
            .oneshot(req("/xrpc/com.atproto.identity.resolveHandle?handle=does-not-exist.com"))
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
            .oneshot(req("/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com&since="))
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
            .oneshot(req("/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com"))
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

    // ── Finding 1: feature-gate matrix — all 4 routes, enabled AND disabled ────

    #[tokio::test]
    async fn router_feature_gate_disabled_all_four_routes_404() {
        let app = build_production_app(false).await;
        // All four XRPC routes must 404 when the gate is disabled.
        let routes = [
            "/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com",
            "/xrpc/com.atproto.repo.describeRepo?repo=did:web:pub.example.com",
            "/xrpc/com.atproto.repo.getRecord?repo=did:web:pub.example.com&collection=ai.hyprstream.model&rkey=abc",
            "/xrpc/com.atproto.identity.resolveHandle?handle=pub.example.com",
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
    async fn router_feature_gate_enabled_all_four_routes_reachable() {
        // Smoke-test: all four routes reach XRPC handlers (not 404) when enabled.
        // Detailed assertions are in the individual endpoint tests above.
        let app = build_production_app(true).await;
        let routes = [
            ("/xrpc/com.atproto.repo.describeRepo?repo=did:web:pub.example.com", StatusCode::OK),
            ("/xrpc/com.atproto.identity.resolveHandle?handle=pub.example.com", StatusCode::OK),
            ("/xrpc/com.atproto.repo.getRecord?repo=did:web:pub.example.com&collection=ai.hyprstream.model&rkey=abc", StatusCode::BAD_REQUEST), // RecordNotFound
            ("/xrpc/com.atproto.sync.getRepo?did=did:web:pub.example.com", StatusCode::OK),
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

    // ── Finding 2: routed CID match/mismatch + malformed query ─────────────────

    #[tokio::test]
    async fn router_get_record_cid_match_returns_record() {
        let state = build_test_state(true).await;
        let snap = sample_snapshot_fixed_tid("did:web:fixed.example.com", "fixed.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        let cid = snap.records.values().next().unwrap().cid().to_string();
        state.xrpc_repos.put(snap).await;
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
        let snap = sample_snapshot_fixed_tid("did:web:fixed.example.com", "fixed.example.com", true);
        let rkey = snap.records.keys().next().unwrap().encode();
        state.xrpc_repos.put(snap).await;
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
            body["message"].as_str().unwrap().contains("does not match cid"),
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
            .oneshot(req(
                "/xrpc/com.atproto.repo.getRecord?repo=\
                 &collection=ai.hyprstream.model&rkey=abc",
            ))
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
}
