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
//! | GET    | `com.atproto.sync.getRepo`          | full-repo CARv1 export (streamed) |
//!
//! **Session endpoints (`createSession`/`getSession`) are deliberately NOT in
//! this PR.** Credential verification (password / app-password) and the OAuth
//! JWT bridge belong with the #1113/#948 OAuth integration work — a read-slice
//! PR is the wrong place to introduce an unauthenticated session-minting path.
//! The four endpoints above are public reads of explicitly-published repos.
//!
//! # Feature gate
//!
//! Routes are mounted only when `OAuthConfig::xrpc_read_slice` is `true`
//! (defaults to `false`). The in-process [`XrpcRepoStore`] starts empty; the
//! write path (#910) populates it with [`RepoSnapshot`]s whose [`public`]
//! flag is `true`. Only `public` snapshots are served by these endpoints —
//! the publication boundary is enforced in [`XrpcRepoStore::get_public`] and
//! tested in `publication_boundary_*`.
//!
//! # Out of scope
//!
//! - `com.atproto.sync.subscribeRepos` (firehose) — issue #1112 defers it.
//! - Write path (`repo.createRecord` etc.) — sequenced with #910.
//! - `createSession`/`getSession` — sequenced with #1113/#948.
//!
//! # XRPC error envelopes
//!
//! All errors follow the XRPC convention: HTTP status code with JSON body
//! `{"error": "<code>", "message": "<human readable>"}`. Extractor/query
//! parse failures are caught via manual query parsing (no axum `Query`
//! rejection bypasses).

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    extract::{RawQuery, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures::stream;
use p256::ecdsa::VerifyingKey;
use serde_json::{json, Value};
use tokio::sync::{RwLock, Semaphore};

use hyprstream_pds::car::{build_car_v1_sections, build_record_proof_car};
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
const GET_REPO_CONCURRENCY: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// RepoSnapshot + XrpcRepoStore
// ─────────────────────────────────────────────────────────────────────────────

/// An in-memory snapshot of one repo's signed state — enough to answer the
/// public read slice (`describeRepo` / `getRecord` / `sync.getRepo`).
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
    /// **Publication boundary.** When `true`, this snapshot is anonymously
    /// readable via the public XRPC read endpoints. When `false`, the public
    /// read handlers act as though the repo does not exist. The write path
    /// (#910) sets this from the per-DID/per-collection authz policy.
    pub public: bool,
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
    /// records, rooted at the commit CID. Returns length-framed sections
    /// suitable for incremental streaming (each `Vec<u8>` is one complete
    /// CAR section: header first, then one per block).
    pub fn full_car_sections(&self) -> Vec<Vec<u8>> {
        let mut blocks: Vec<(Cid, Vec<u8>)> = Vec::new();
        let commit_cid = self.commit.cid();
        blocks.push((commit_cid, self.commit.to_dag_cbor()));
        for (cid, data) in &self.node_blocks {
            blocks.push((*cid, data.encode()));
        }
        for rec in self.records.values() {
            blocks.push((rec.cid(), rec.to_dag_cbor()));
        }
        build_car_v1_sections(&[commit_cid], &blocks)
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

    /// Build a minimal atproto-compatible DID document for this snapshot's
    /// account (used as `describeRepo.didDoc`). Reuses [`build_did_document`]
    /// with only the `#atproto` verification method + `#atproto_pds` service.
    fn did_doc(&self, issuer_url: &str) -> Value {
        let atproto = AtprotoIdentity {
            p256_vk: &self.atproto_vk,
            handle: &self.handle,
        };
        build_did_document(&self.did, issuer_url, &[], Some(&atproto), &[], None, None)
    }
}

/// In-process registry of hosted repos, keyed by DID. Populated by the write
/// path (#910) and by tests; the read slice consults it directly.
#[derive(Debug)]
pub struct XrpcRepoStore {
    by_did: RwLock<BTreeMap<String, Arc<RepoSnapshot>>>,
    /// Concurrency limiter for full-CAR exports (`sync.getRepo`).
    get_repo_sema: Semaphore,
}

impl Default for XrpcRepoStore {
    fn default() -> Self {
        Self {
            by_did: RwLock::new(BTreeMap::new()),
            get_repo_sema: Semaphore::new(GET_REPO_CONCURRENCY),
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
    /// to the public read endpoints (publication boundary, finding #4).
    pub async fn get_public(&self, did: &str) -> Option<Arc<RepoSnapshot>> {
        self.get(did).await.filter(|s| s.public)
    }

    /// Resolve a handle (bare hostname) to a **public** snapshot — only
    /// unique handles resolve unambiguously (atproto handle-uniqueness rule).
    pub async fn by_handle_public(&self, handle: &str) -> Option<Arc<RepoSnapshot>> {
        let guard = self.by_did.read().await;
        let mut hits = guard.values().filter(|s| s.public && s.handle == handle);
        let first = hits.next()?;
        if hits.next().is_some() {
            return None; // ambiguous — refuse
        }
        Some(Arc::clone(first))
    }

    /// Acquire a concurrency permit for full-CAR export. Bounded by
    /// [`GET_REPO_CONCURRENCY`]. The permit is released when dropped.
    pub async fn acquire_get_repo(&self) -> Result<tokio::sync::SemaphorePermit<'_>, tokio::sync::AcquireError> {
        self.get_repo_sema.acquire().await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XRPC error helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build an XRPC error JSON body. Always includes `error` and `message`.
pub fn xrpc_error_body(error: &str, message: impl Into<String>) -> Value {
    json!({ "error": error, "message": message.into() })
}

/// Build an XRPC error [`Response`] with the given status code and body.
pub fn xrpc_error(status: StatusCode, error: &str, message: impl Into<String>) -> Response {
    (status, axum::Json(xrpc_error_body(error, message))).into_response()
}

/// XRPC error codes used by the atproto read slice.
pub mod errors {
    pub const INVALID_REQUEST: &str = "InvalidRequest";
    pub const ACCOUNT_NOT_FOUND: &str = "AccountNotFound";
    pub const HANDLE_NOT_FOUND: &str = "HandleNotFound";
    pub const RECORD_NOT_FOUND: &str = "RecordNotFound";
    pub const REPO_NOT_FOUND: &str = "RepoNotFound";
    pub const INTERNAL_SERVER_ERROR: &str = "InternalServerError";
}

// ─────────────────────────────────────────────────────────────────────────────
// Manual query parsing (avoids axum Query rejection bypassing XRPC envelope)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a raw query string `key=value&key=value` into a simple lookup.
///
/// XRPC query params (DIDs, handles, NSIDs, rkeys, CID strings) use only
/// URL-safe characters, so no percent-decoding is needed for the MVP.
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
// resolveHandle
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /xrpc/com.atproto.identity.resolveHandle?handle=<host>`.
///
/// Returns `HandleNotFound` (canonical lexicon) for unknown handles.
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

    // Hosted public account?
    if let Some(snap) = state.xrpc_repos.by_handle_public(handle).await {
        return axum::Json(json!({ "did": snap.did })).into_response();
    }

    // This deployment's own handle?
    if let Some(authority) = issuer_authority(&state.issuer_url) {
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

// ─────────────────────────────────────────────────────────────────────────────
// describeRepo — canonical lexicon shape
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /xrpc/com.atproto.repo.describeRepo?repo=<did|handle>`.
///
/// Canonical shape: `handleIsCorrect` (not `handleIsValid`), `collections`
/// is an array of NSID strings, `didDoc` is the populated DID document.
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

    let snap = match lookup_public_snapshot(&state, key).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {key:?} is not hosted by this PDS"),
            );
        }
    };

    // Canonical: collections = array of NSID strings (not objects).
    let collections: Vec<&str> = if snap.records.is_empty() {
        Vec::new()
    } else {
        vec![HOSTED_COLLECTION]
    };

    let did_doc = snap.did_doc(&state.issuer_url);

    axum::Json(json!({
        "handle": snap.handle,
        "did": snap.did,
        "didDoc": did_doc,
        "collections": collections,
        "handleIsCorrect": true,
    }))
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// getRecord — with optional cid pinning
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /xrpc/com.atproto.repo.getRecord?repo=<did>&collection=<nsid>&rkey=<rkey>[&cid=<cid>]`.
///
/// When `cid` is provided, returns `RecordNotFound` if the current record's CID
/// does not match (version mismatch).
pub async fn get_record(
    State(state): State<Arc<OAuthState>>,
    RawQuery(raw): RawQuery,
) -> Response {
    let params = parse_query(raw.as_deref());

    let collection = params.get("collection").copied().unwrap_or("");
    let rkey = params.get("rkey").copied().unwrap_or("");

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

    let snap = match lookup_public_snapshot(&state, repo).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {repo:?} is not hosted by this PDS"),
            );
        }
    };

    let Some((rec, cid)) = snap.record_by_rkey(rkey) else {
        return xrpc_error(
            StatusCode::BAD_REQUEST,
            errors::RECORD_NOT_FOUND,
            format!("record {collection}/{rkey} not found"),
        );
    };

    // Optional cid pinning: if the client requests a specific version and it
    // doesn't match, return RecordNotFound (the record at that version is gone).
    if let Some(requested_cid) = params.get("cid") {
        if *requested_cid != cid.to_string() {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::RECORD_NOT_FOUND,
                format!("record {collection}/{rkey} does not match cid {requested_cid:?}"),
            );
        }
    }

    axum::Json(snap.record_json(rkey, rec)).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// sync.getRepo — streamed CAR export with concurrency cap
// ─────────────────────────────────────────────────────────────────────────────

/// `GET /xrpc/com.atproto.sync.getRepo?did=<did>[&since=<commitCid>]`.
///
/// Streams a CARv1 blob (`application/vnd.ipld.car`) containing the commit +
/// MST + records, rooted at the signed commit. Sections are streamed
/// incrementally via `Body::from_stream` — no double materialization.
///
/// `since` (revision-delta) is not yet supported and returns `InvalidRequest`
/// rather than silently returning the full repo.
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

    // since: reject explicitly until delta export is implemented (#910).
    if let Some(since) = params.get("since") {
        if !since.is_empty() {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::INVALID_REQUEST,
                "since (revision delta) is not yet supported; omit for full export",
            );
        }
    }

    let snap = match state.xrpc_repos.get_public(did).await {
        Some(s) => s,
        None => {
            return xrpc_error(
                StatusCode::BAD_REQUEST,
                errors::REPO_NOT_FOUND,
                format!("repo {did:?} is not hosted by this PDS"),
            );
        }
    };

    // Acquire concurrency permit — bounded by GET_REPO_CONCURRENCY.
    let _permit = match state.xrpc_repos.acquire_get_repo().await {
        Ok(p) => p,
        Err(_) => {
            return xrpc_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                errors::INTERNAL_SERVER_ERROR,
                "concurrency limiter closed",
            );
        }
    };

    // Build length-framed sections and stream them incrementally.
    let sections = snap.full_car_sections();
    let body = axum::body::Body::from_stream(stream::iter(
        sections
            .into_iter()
            .map(|s| Ok::<Bytes, std::convert::Infallible>(Bytes::from(s))),
    ));

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/vnd.ipld.car")],
        body,
    )
        .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a `did:` or handle reference to a **public** snapshot.
/// Non-public repos are invisible (publication boundary).
async fn lookup_public_snapshot(
    state: &OAuthState,
    key: &str,
) -> Option<Arc<RepoSnapshot>> {
    if key.starts_with("did:") {
        state.xrpc_repos.get_public(key).await
    } else {
        state.xrpc_repos.by_handle_public(key).await
    }
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

    #[test]
    fn error_body_shape_is_xrpc_convention() {
        let body = xrpc_error_body("RecordNotFound", "no such record");
        assert_eq!(body["error"], "RecordNotFound");
        assert_eq!(body["message"], "no such record");
        assert_eq!(body.as_object().unwrap().len(), 2);
    }

    #[test]
    fn snapshot_record_json_carries_uri_cid_and_value() {
        let snap = sample_snapshot("did:web:h.example.com", "h.example.com", true);
        let (tid, rec) = snap.records.iter().next().unwrap();
        let rkey = tid.encode();
        let json = snap.record_json(&rkey, rec);
        assert_eq!(
            json["uri"],
            format!("at://did:web:h.example.com/{HOSTED_COLLECTION}/{rkey}"),
        );
        assert_eq!(json["value"]["$type"], HOSTED_COLLECTION);
    }

    #[test]
    fn snapshot_full_car_sections_round_trip_via_parse_car_v1() {
        let snap = sample_snapshot("did:web:h.example.com", "h.example.com", true);
        // Concatenate sections and parse as a single CARv1 blob.
        let sections = snap.full_car_sections();
        let car: Vec<u8> = sections.into_iter().flatten().collect();
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1(&car).expect("parse CAR");
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], snap.commit.cid());
        assert!(blocks.len() >= 3); // commit + MST node + record
    }

    #[test]
    fn snapshot_record_proof_car_verifies_offline() {
        let snap = sample_snapshot("did:web:h.example.com", "h.example.com", true);
        let (tid, _rec) = snap.records.iter().next().unwrap();
        let rkey = tid.encode();
        let car = snap.record_proof_car(&rkey).expect("proof for present rkey");
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1(&car).expect("parse CAR");
        assert_eq!(roots.len(), 1);

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
        let commit = Commit::from_dag_cbor(&commit_block.unwrap()).unwrap();
        let record = ModelRecord::from_dag_cbor(&record_block.unwrap()).unwrap();
        commit.verify(&snap.atproto_vk).expect("commit verifies");
        assert_eq!(record.cid(), snap.records.values().next().unwrap().cid());
    }

    #[tokio::test]
    async fn publication_boundary_public_snapshot_visible() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:pub.example.com", "pub.example.com", true)).await;
        assert!(store.get_public("did:web:pub.example.com").await.is_some());
        assert!(store.by_handle_public("pub.example.com").await.is_some());
    }

    #[tokio::test]
    async fn publication_boundary_non_public_snapshot_invisible() {
        let store = XrpcRepoStore::new();
        // Private snapshot — get() sees it but public lookups must not.
        store.put(sample_snapshot("did:web:priv.example.com", "priv.example.com", false)).await;
        assert!(store.get("did:web:priv.example.com").await.is_some());
        assert!(store.get_public("did:web:priv.example.com").await.is_none());
        assert!(store.by_handle_public("priv.example.com").await.is_none());
    }

    #[tokio::test]
    async fn repo_store_ambiguous_handle_refuses() {
        let store = XrpcRepoStore::new();
        store.put(sample_snapshot("did:web:a.example.com", "dup.example.com", true)).await;
        store.put(sample_snapshot("did:web:b.example.com", "dup.example.com", true)).await;
        assert!(store.by_handle_public("dup.example.com").await.is_none());
    }

    #[test]
    fn parse_query_extracts_key_value_pairs() {
        let q = parse_query(Some("repo=did:web:x&collection=ai.hyprstream.model&rkey=abc"));
        assert_eq!(q.get("repo").copied(), Some("did:web:x"));
        assert_eq!(q.get("collection").copied(), Some("ai.hyprstream.model"));
        assert_eq!(q.get("rkey").copied(), Some("abc"));
        assert_eq!(q.get("missing"), None);
    }

    #[test]
    fn parse_query_empty_and_malformed() {
        assert!(parse_query(None).is_empty());
        assert!(parse_query(Some("")).is_empty());
        // No `=` → skipped.
        assert!(parse_query(Some("garbage")).is_empty());
    }

    #[test]
    fn describe_repo_did_doc_carries_atproto_vm() {
        let snap = sample_snapshot("did:web:h.example.com", "h.example.com", true);
        let doc = snap.did_doc("https://h.example.com");
        assert_eq!(doc["id"], "did:web:h.example.com");
        // The atproto verification method must be present.
        let vms = doc["verificationMethod"].as_array().unwrap();
        assert!(vms.iter().any(|vm| vm["id"].as_str().unwrap_or("").ends_with("#atproto")));
        // #atproto_pds service present.
        let svcs = doc["service"].as_array().unwrap();
        assert!(svcs.iter().any(|s| s["id"].as_str().unwrap_or("").ends_with("#atproto_pds")));
    }
}
