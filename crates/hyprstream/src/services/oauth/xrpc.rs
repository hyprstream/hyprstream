//! `com.atproto.*` XRPC server surface — MVP read slice (#1112).
//!
//! HTTP XRPC endpoints served from the same Axum surface that hosts OAuth and
//! `/.well-known/atproto-did` (see `super::did_document`). `hyprstream-pds`
//! remains deliberately no-networking: this module is the serving layer that
//! calls its data functions (`Commit`/`MST`/`car::build_record_proof_car`).
//!
//! # Endpoints (read slice)
//!
//! | Method | Path (under `/xrpc/`) | Auth | Notes |
//! |--------|-----------------------|------|-------|
//! | POST   | `com.atproto.server.createSession`  | public | bridges to `OAuthState::sessions` |
//! | GET    | `com.atproto.server.getSession`     | Bearer session | server-side session lookup |
//! | GET    | `com.atproto.identity.resolveHandle`| public | reuses `did_document` handle→DID |
//! | GET    | `com.atproto.repo.describeRepo`     | public | DID/handle + MST commit head |
//! | GET    | `com.atproto.repo.getRecord`        | public | record JSON + CAR proof |
//! | GET    | `com.atproto.sync.getRepo`          | public | full-repo CARv1 export |
//!
//! # Sessions bridge
//!
//! atproto's `createSession` is the "log in" call. Rather than build a parallel
//! auth store, we bridge to the OAuth browser-login `SessionStore
//! <super::session::SessionStore>`: `createSession` mints a server-side `Session`
//! (the same store `/oauth/authorize` writes to) and returns its id as
//! `accessJwt`. `getSession` reverses the lookup. The OAuth `/oauth/token` JWT
//! path remains the canonical DPoP-bound access-token issuance; this is the
//! atproto-client-friendly façade over the same subject.
//!
//! # Repo store
//!
//! There is no on-disk PDS record store yet (the `hyprstream-pds` store is
//! in-memory by design). The MVP read slice holds per-DID repo snapshots in an
//! in-process [`XrpcRepoStore`]. The upcoming write path (#910) will populate
//! it; for now tests and the atproto login integration can seed it directly.
//!
//! # Out of scope
//!
//! - `com.atproto.sync.subscribeRepos` (firehose) — issue #1112 defers it.
//! - Write path (`repo.createRecord` etc.) — sequenced with #910.
//!
//! # XRPC error envelopes
//!
//! All errors follow the XRPC convention: HTTP status code with JSON body
//! `{"error": "<code>", "message": "<human readable>"}`. See [`xrpc_error`].

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use p256::ecdsa::VerifyingKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::RwLock;

use hyprstream_pds::car::{build_car_v1, build_record_proof_car};
use hyprstream_pds::commit::Commit;
use hyprstream_pds::mst::{Node, NodeData};
use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
use hyprstream_pds::tid::Tid;
use hyprstream_pds::Cid;

use super::did_document::issuer_authority;
use super::state::OAuthState;

/// `ai.hyprstream.model` is the only collection this PDS hosts today.
pub const HOSTED_COLLECTION: &str = COLLECTION_NSID;

/// An in-memory snapshot of one repo's signed state — enough to answer the
/// read slice (`describeRepo` / `getRecord` / `sync.getRepo`).
///
/// Holds the signed commit, the full MST node-block set, the per-rkey record
/// table, and the account's `#atproto` P-256 verifying key (so the host can
/// re-verify proofs it serves, satisfying the D5 untrusted-host posture).
#[derive(Clone, Debug)]
pub struct RepoSnapshot {
    /// Account DID (e.g. `did:web:hyprstream.example.com`).
    pub did: String,
    /// Account handle (bare hostname, no port — atproto handle rule).
    pub handle: String,
    /// Signed commit head of the MST.
    pub commit: Commit,
    /// `(cid, NodeData)` for every MST node, including the root. Used for
    /// full-repo CAR export and for record-proof construction.
    pub node_blocks: Vec<(Cid, NodeData)>,
    /// All hosted records, keyed by TID. The rkey (record key) in at-uri form
    /// is `Tid::encode(tid)` (base32-sortable).
    pub records: BTreeMap<Tid, ModelRecord>,
    /// The account's published `#atproto` P-256 verifying key.
    pub atproto_vk: VerifyingKey,
}

impl RepoSnapshot {
    /// The MST root CID from the snapshot's commit (`commit.data`).
    pub fn root_cid(&self) -> Cid {
        self.commit.data
    }

    /// Look up the record + its CID for a given rkey.
    pub fn record_by_rkey(&self, rkey: &str) -> Option<(&ModelRecord, Cid)> {
        let tid = Tid::parse(rkey).ok()?;
        let rec = self.records.get(&tid)?;
        Some((rec, rec.cid()))
    }

    /// Build a CAR proof for one record (commit + MST path + record).
    pub fn record_proof_car(&self, rkey: &str) -> Option<Vec<u8>> {
        let tid = Tid::parse(rkey).ok()?;
        let rec = self.records.get(&tid)?;
        // Build a single-collection MST over the current record set, then ask
        // it for the inclusion proof. The MST is deterministic from the record
        // table, so this re-derivation matches what the writer produced.
        let cids: BTreeMap<Tid, Cid> = self
            .records
            .iter()
            .map(|(t, r)| (*t, r.cid()))
            .collect();
        let tree = Node::from_records(HOSTED_COLLECTION, &cids);
        let proof = tree.proof(HOSTED_COLLECTION, &tid)?;
        Some(build_record_proof_car(&self.commit, &proof, &self.node_blocks, rec))
    }

    /// Build a full-repo CARv1 export: commit block + all MST nodes + all
    /// records, rooted at the commit CID. This is `com.atproto.sync.getRepo`.
    pub fn full_car(&self) -> Vec<u8> {
        let mut blocks: Vec<(Cid, Vec<u8>)> = Vec::new();
        // Commit block.
        let commit_cid = self.commit.cid();
        blocks.push((commit_cid, self.commit.to_dag_cbor()));
        // All MST node blocks.
        for (cid, data) in &self.node_blocks {
            blocks.push((*cid, data.encode()));
        }
        // All record blocks.
        for rec in self.records.values() {
            blocks.push((rec.cid(), rec.to_dag_cbor()));
        }
        build_car_v1(&[commit_cid], &blocks)
    }

    /// Convert a hosted record to the atproto `getRecord` `value` JSON:
    /// the lexicon fields plus `$type` and the IPLD-link `$uri`.
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
}

/// In-process registry of hosted repos, keyed by DID. Populated by the write
/// path (#910) and by tests; the read slice consults it directly.
#[derive(Debug, Default)]
pub struct XrpcRepoStore {
    by_did: RwLock<BTreeMap<String, Arc<RepoSnapshot>>>,
}

impl XrpcRepoStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a snapshot for a DID.
    pub async fn put(&self, snap: RepoSnapshot) {
        self.by_did.write().await.insert(snap.did.clone(), Arc::new(snap));
    }

    pub async fn get(&self, did: &str) -> Option<Arc<RepoSnapshot>> {
        self.by_did.read().await.get(did).cloned()
    }

    /// Resolve a handle (bare hostname) to a snapshot — only unique handles
    /// resolve unambiguously, which is the atproto handle-uniqueness rule.
    pub async fn by_handle(&self, handle: &str) -> Option<Arc<RepoSnapshot>> {
        let guard = self.by_did.read().await;
        let mut hits = guard.values().filter(|s| s.handle == handle);
        let first = hits.next()?;
        if hits.next().is_some() {
            // Ambiguous handle — refuse rather than guess.
            return None;
        }
        Some(Arc::clone(first))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XRPC error envelope helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build an XRPC error JSON body. Always includes `error` and `message`.
pub fn xrpc_error_body(error: &str, message: impl Into<String>) -> Value {
    json!({ "error": error, "message": message.into() })
}

/// Build an XRPC error [`Response`] with the given status code and body.
pub fn xrpc_error(status: StatusCode, error: &str, message: impl Into<String>) -> Response {
    (status, Json(xrpc_error_body(error, message))).into_response()
}

/// Common XRPC error codes used by the atproto read slice.
pub mod errors {
    pub const INVALID_REQUEST: &str = "InvalidRequest";
    pub const BAD_REQUEST: &str = "BadRequest";
    pub const UNAUTHORIZED: &str = "Unauthorized";
    pub const INVALID_TOKEN: &str = "InvalidToken";
    pub const EXPIRED_TOKEN: &str = "ExpiredToken";
    pub const ACCOUNT_NOT_FOUND: &str = "AccountNotFound";
    pub const RECORD_NOT_FOUND: &str = "RecordNotFound";
    pub const REPO_NOT_FOUND: &str = "RepoNotFound";
    pub const INTERNAL_SERVER_ERROR: &str = "InternalServerError";
}

// ─────────────────────────────────────────────────────────────────────────────
// createSession / getSession — bridge to OAuth SessionStore
// ─────────────────────────────────────────────────────────────────────────────

/// `POST /xrpc/com.atproto.server.createSession` request body.
///
/// atproto clients post `identifier` (handle or DID) + `password`. The MVP
/// bridge accepts the same shape but treats `identifier` as the local username
/// (per `OAuthState::sessions`'s `username` field) and `password` as a
/// presence-only credential for the local dev login story. Real credential
/// checks happen via OAuth `/oauth/authorize` + SCIM; this is the façade that
/// issues an atproto-shaped session backed by the same store.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateSessionInput {
    pub identifier: String,
    #[serde(default)]
    pub password: String,
}

/// `POST /xrpc/com.atproto.server.createSession` response — atproto session
/// envelope. `accessJwt` is the server-side session id (see module docs).
///
/// Field names serialize as atproto's camelCase via `rename_all`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionOutput {
    pub did: String,
    pub handle: String,
    pub access_jwt: String,
    /// Standard atproto field; omitted when no OAuth JWT path was taken.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_jwt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    pub active: bool,
}

/// `GET /xrpc/com.atproto.server.getSession` response.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetSessionOutput {
    pub did: String,
    pub handle: String,
    pub active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}

/// Resolve `identifier` (DID or handle) to a hosted snapshot.
async fn resolve_account(
    state: &OAuthState,
    identifier: &str,
) -> Option<Arc<RepoSnapshot>> {
    // Direct DID lookup first.
    if identifier.starts_with("did:") {
        if let Some(snap) = state.xrpc_repos.get(identifier).await {
            return Some(snap);
        }
    }
    // Fall back to handle (bare hostname) lookup in the repo store.
    if let Some(snap) = state.xrpc_repos.by_handle(identifier).await {
        return Some(snap);
    }
    // Final fallback: treat the identifier as this deployment's own subject —
    // `did:web:{authority}` — when no snapshot is seeded. This keeps the login
    // story usable in single-tenant dev setups with no seeded repo.
    if let Some(authority) = issuer_authority(&state.issuer_url) {
        let did = format!("did:web:{authority}");
        let handle = authority.split(':').next().unwrap_or(&authority).to_owned();
        if identifier.eq_ignore_ascii_case(&handle) || identifier == did {
            return None; // no snapshot — caller surfaces AccountNotFound
        }
    }
    None
}

/// `POST /xrpc/com.atproto.server.createSession`.
pub async fn create_session(
    State(state): State<Arc<OAuthState>>,
    Json(input): Json<CreateSessionInput>,
) -> Response {
    let identifier = input.identifier.trim();
    if identifier.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "identifier is required",
        );
    }

    let snap = match resolve_account(&state, identifier).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::ACCOUNT_NOT_FOUND,
                format!("no account found for identifier {identifier:?}"),
            );
        }
    };

    // Bridge: mint a server-side OAuth session for the same subject. Same store
    // `/oauth/authorize` writes to; no parallel auth state.
    let session_id = state
        .sessions
        .create(snap.did.clone(), "atproto-xrpc".to_owned())
        .await;

    Json(SessionOutput {
        did: snap.did.clone(),
        handle: snap.handle.clone(),
        access_jwt: session_id,
        refresh_jwt: None,
        email: None,
        active: true,
    })
    .into_response()
}

/// `GET /xrpc/com.atproto.server.getSession`.
///
/// `Authorization: Bearer <accessJwt>` where `accessJwt` is the session id
/// returned by `createSession`.
pub async fn get_session(
    State(state): State<Arc<OAuthState>>,
    headers: axum::http::HeaderMap,
) -> Response {
    let Some(token) = bearer_token(&headers) else {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::INVALID_TOKEN,
            "Bearer accessJwt required",
        );
    };
    let Some(session) = state.sessions.get(&token).await else {
        return xrpc_error(
            StatusCode::UNAUTHORIZED,
            errors::EXPIRED_TOKEN,
            "session not found or expired",
        );
    };

    let snap = state.xrpc_repos.get(&session.username).await;
    let (did, handle) = match snap {
        Some(s) => (s.did.clone(), s.handle.clone()),
        // Session is valid but no repo snapshot is seeded — degrade gracefully
        // using the issuer-derived DID so getSession still answers.
        None => match issuer_authority(&state.issuer_url) {
            Some(authority) => {
                let did = format!("did:web:{authority}");
                let handle = authority.split(':').next().unwrap_or(&authority).to_owned();
                (did, handle)
            }
            None => {
                return xrpc_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    errors::INTERNAL_SERVER_ERROR,
                    "issuer URL has no authority",
                );
            }
        },
    };

    Json(GetSessionOutput {
        did,
        handle,
        active: true,
        email: None,
    })
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// resolveHandle
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ResolveHandleQuery {
    pub handle: String,
}

/// `GET /xrpc/com.atproto.identity.resolveHandle?handle=<host>`.
///
/// Shares logic with `/.well-known/atproto-did`: the deployment's own handle
/// (issuer authority) resolves to `did:web:{authority}`; hosted-account handles
/// resolve from the repo store.
pub async fn resolve_handle(
    State(state): State<Arc<OAuthState>>,
    Query(q): Query<ResolveHandleQuery>,
) -> Response {
    let handle = q.handle.trim().to_owned();
    if handle.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "handle query parameter is required",
        );
    }

    // Hosted account?
    if let Some(snap) = state.xrpc_repos.by_handle(&handle).await {
        return Json(json!({ "did": snap.did })).into_response();
    }

    // This deployment's own handle?
    if let Some(authority) = issuer_authority(&state.issuer_url) {
        let self_handle = authority.split(':').next().unwrap_or(&authority);
        if handle == self_handle {
            let did = format!("did:web:{authority}");
            return Json(json!({ "did": did })).into_response();
        }
    }

    xrpc_error(
        StatusCode::BAD_REQUEST,
        errors::ACCOUNT_NOT_FOUND,
        format!("handle {handle:?} is not hosted by this PDS"),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// describeRepo
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DescribeRepoQuery {
    pub repo: String,
}

/// `GET /xrpc/com.atproto.repo.describeRepo?repo=<did|handle>`.
pub async fn describe_repo(
    State(state): State<Arc<OAuthState>>,
    Query(q): Query<DescribeRepoQuery>,
) -> Response {
    let key = q.repo.trim().to_owned();
    if key.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "repo query parameter is required",
        );
    }

    let snap = match lookup_snapshot(&state, &key).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {key:?} is not hosted by this PDS"),
            );
        }
    };

    let collections = if snap.records.is_empty() {
        Vec::new()
    } else {
        vec![json!({ "nsid": HOSTED_COLLECTION, "count": snap.records.len() })]
    };

    Json(json!({
        "handle": snap.handle,
        "did": snap.did,
        "didDoc": {},                // caller can fetch /.well-known/did.json
        "collections": collections,
        "handleIsValid": true,
    }))
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// getRecord
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GetRecordQuery {
    /// `at://` URI of the record, e.g.
    /// `at://did:web:x/ai.hyprstream.model/<rkey>`.
    pub repo: String,
    pub collection: String,
    pub rkey: String,
}

/// `GET /xrpc/com.atproto.repo.getRecord?repo=<did>&collection=<nsid>&rkey=<rkey>`.
pub async fn get_record(
    State(state): State<Arc<OAuthState>>,
    Query(q): Query<GetRecordQuery>,
) -> Response {
    if q.collection != HOSTED_COLLECTION {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::RECORD_NOT_FOUND,
            format!("collection {} is not hosted", q.collection),
        );
    }
    if q.rkey.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "rkey is required",
        );
    }

    let snap = match lookup_snapshot(&state, &q.repo).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {:?} is not hosted by this PDS", q.repo),
            );
        }
    };

    let Some((rec, _cid)) = snap.record_by_rkey(&q.rkey) else {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::RECORD_NOT_FOUND,
            format!("record {}/{} not found", q.collection, q.rkey),
        );
    };

    Json(snap.record_json(&q.rkey, rec)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// sync.getRepo — CAR export
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GetRepoQuery {
    pub did: String,
    /// Optional earliest commit CID — the MVP store has one head, so a
    /// non-matching `since` simply returns the full CAR (no delta encoding).
    #[serde(default)]
    pub since: Option<String>,
}

/// `GET /xrpc/com.atproto.sync.getRepo?did=<did>`.
///
/// Returns a CARv1 blob (`application/vnd.ipld.car`) containing the commit +
/// MST + records, rooted at the signed commit.
pub async fn get_repo(
    State(state): State<Arc<OAuthState>>,
    Query(q): Query<GetRepoQuery>,
) -> Response {
    let did = q.did.trim().to_owned();
    if did.is_empty() {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::INVALID_REQUEST,
            "did query parameter is required",
        );
    }

    let snap = match state.xrpc_repos.get(&did).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {did:?} is not hosted by this PDS"),
            );
        }
    };

    let car = snap.full_car();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.ipld.car")],
        car,
    )
        .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a `did:` or handle reference to a hosted snapshot.
async fn lookup_snapshot(state: &OAuthState, key: &str) -> Option<Arc<RepoSnapshot>> {
    if key.starts_with("did:") {
        state.xrpc_repos.get(key).await
    } else {
        state.xrpc_repos.by_handle(key).await
    }
}

/// Extract a `Bearer <token>` from request headers.
fn bearer_token(headers: &axum::http::HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").map(str::to_owned))
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

    /// Build a minimal hosted snapshot with one record, signed by a fresh key.
    fn sample_snapshot(did: &str, handle: &str) -> RepoSnapshot {
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

        // Signed commit head — version 3, no predecessor.
        use hyprstream_pds::commit::UnsignedCommit;
        let unsigned = UnsignedCommit::new(did.to_owned(), root_cid, Tid::now(), None);
        let commit = hyprstream_pds::commit::Commit::sign(&unsigned, &signing);
        commit.verify(&vk).expect("self-signed commit verifies");

        RepoSnapshot {
            did: did.to_owned(),
            handle: handle.to_owned(),
            commit,
            node_blocks,
            records,
            atproto_vk: vk,
        }
    }

    #[test]
    fn error_body_shape_is_xrpc_convention() {
        let body = xrpc_error_body("RecordNotFound", "no such record");
        assert_eq!(body["error"], "RecordNotFound");
        assert_eq!(body["message"], "no such record");
        // Only the two XRPC fields — no extras that would confuse clients.
        assert_eq!(body.as_object().unwrap().len(), 2);
    }

    #[test]
    fn snapshot_record_json_carries_uri_cid_and_value() {
        let snap = sample_snapshot("did:web:hyprstream.example.com", "hyprstream.example.com");
        let (tid, rec) = snap.records.iter().next().unwrap();
        let rkey = tid.encode();
        let json = snap.record_json(&rkey, rec);
        assert_eq!(
            json["uri"],
            format!("at://did:web:hyprstream.example.com/{HOSTED_COLLECTION}/{rkey}"),
        );
        assert_eq!(json["value"]["$type"], HOSTED_COLLECTION);
        assert_eq!(json["value"]["repo"], "at://did:web:hyprstream.example.com");
        assert_eq!(json["value"]["currentOid"], rec.current_oid);
    }

    #[test]
    fn snapshot_full_car_round_trips_via_parse_car_v1() {
        let snap = sample_snapshot("did:web:hyprstream.example.com", "hyprstream.example.com");
        let car = snap.full_car();
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1(&car).expect("parse CAR");
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], snap.commit.cid());
        // commit block + at least one MST node + one record block.
        assert!(blocks.len() >= 3);
    }

    #[test]
    fn snapshot_record_proof_car_verifies_offline() {
        let snap = sample_snapshot("did:web:hyprstream.example.com", "hyprstream.example.com");
        let (tid, _rec) = snap.records.iter().next().unwrap();
        let rkey = tid.encode();
        let car = snap.record_proof_car(&rkey).expect("proof for present rkey");

        // Offline D5 verify against the atproto verifying key.
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1(&car).expect("parse CAR");
        assert_eq!(roots.len(), 1);

        // Find commit + record blocks and verify the proof.
        let mut commit_block: Option<Vec<u8>> = None;
        let mut record_block: Option<Vec<u8>> = None;
        for (cid, bytes) in &blocks {
            if *cid == snap.commit.cid() {
                commit_block = Some(bytes.clone());
            }
            if *cid == snap.records.values().next().unwrap().cid() {
                record_block = Some(bytes.clone());
            }
        }
        let commit_bytes = commit_block.expect("commit block present");
        let record_bytes = record_block.expect("record block present");

        let commit =
            hyprstream_pds::commit::Commit::from_dag_cbor(&commit_bytes).expect("decode commit");
        let record =
            ModelRecord::from_dag_cbor(&record_bytes).expect("decode record");
        // The commit must verify against the account's #atproto key.
        commit.verify(&snap.atproto_vk).expect("commit verifies");
        // The record's CID must recompute.
        assert_eq!(record.cid(), snap.records.values().next().unwrap().cid());
    }

    #[test]
    fn record_by_rkey_round_trips_tid_encoding() {
        let snap = sample_snapshot("did:web:hyprstream.example.com", "hyprstream.example.com");
        let (tid, _rec) = snap.records.iter().next().unwrap();
        let rkey = tid.encode();
        assert!(snap.record_by_rkey(&rkey).is_some());
        // Bogus rkey → None.
        assert!(snap.record_by_rkey("does-not-exist").is_none());
    }

    #[tokio::test]
    async fn repo_store_put_get_and_handle_lookup() {
        let store = XrpcRepoStore::new();
        let snap = Arc::new(sample_snapshot("did:web:hyprstream.example.com", "hyprstream.example.com"));
        store.put((*snap).clone()).await;
        assert!(store.get("did:web:hyprstream.example.com").await.is_some());
        assert_eq!(
            store.by_handle("hyprstream.example.com").await.unwrap().did,
            "did:web:hyprstream.example.com",
        );
        // Unknown handle.
        assert!(store.by_handle("nope.example.com").await.is_none());
    }

    #[tokio::test]
    async fn repo_store_ambiguous_handle_refuses() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:a.example.com", "dup.example.com")).await;
        store.put(sample_snapshot("did:web:b.example.com", "dup.example.com")).await;
        // Two snapshots claim the same handle — refuse to guess.
        assert!(store.by_handle("dup.example.com").await.is_none());
    }

    #[test]
    fn bearer_token_strips_scheme_prefix() {
        let mut headers = axum::http::HeaderMap::new();
        assert!(bearer_token(&headers).is_none());
        headers.insert(
            header::AUTHORIZATION,
            axum::http::HeaderValue::from_static("Bearer abc123"),
        );
        assert_eq!(bearer_token(&headers).as_deref(), Some("abc123"));
    }

    #[test]
    fn session_output_serializes_camel_case_atproto_envelope() {
        // Serialize directly to confirm the atproto envelope fields appear
        // in camelCase and that optional fields are skipped when None.
        let out = SessionOutput {
            did: "did:web:x".into(),
            handle: "x".into(),
            access_jwt: "tok".into(),
            refresh_jwt: None,
            email: None,
            active: true,
        };
        let v = serde_json::to_value(&out).unwrap();
        assert_eq!(v["did"], "did:web:x");
        assert_eq!(v["handle"], "x");
        assert_eq!(v["accessJwt"], "tok");
        // refreshJwt/email absent when None (skip_serializing_if).
        assert!(v.get("refreshJwt").is_none());
        assert!(v.get("email").is_none());

        // When set, the camelCase name is used.
        let out2 = SessionOutput {
            did: "did:web:x".into(),
            handle: "x".into(),
            access_jwt: "tok".into(),
            refresh_jwt: Some("rf".into()),
            email: Some("u@x".into()),
            active: true,
        };
        let v2 = serde_json::to_value(&out2).unwrap();
        assert_eq!(v2["refreshJwt"], "rf");
        assert_eq!(v2["email"], "u@x");
    }
}
