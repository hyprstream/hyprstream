//! Discovery service — re-exports from `hyprstream-discovery`.
//!
//! The DiscoveryService implementation has moved to the `hyprstream-discovery` crate.
//! This module provides a `PolicyAuthProvider` that wraps `PolicyClient` to implement
//! the `AuthorizationProvider` trait expected by DiscoveryService.

pub use hyprstream_discovery::{AuthorizationProvider, DiscoveryService};

use crate::services::generated::policy_client::PolicyCheck;
use async_trait::async_trait;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context as _, Result as AnyResult};
use hyprstream_discovery::{RecordCarData, RecordResolver};
use hyprstream_pds::car::build_record_proof_car;
use hyprstream_pds::commit::Commit;
use hyprstream_pds::mst::Node;
use hyprstream_pds::record::ModelRecord;
use hyprstream_pds::tid::Tid;
use sha2::Digest as _;

// Re-export the generated discovery client types from our local generated module
// (hyprstream still needs its own discovery_client for the DiscoveryClient type)

/// Policy-based authorization provider wrapping PolicyClient.
///
/// Bridges the `AuthorizationProvider` trait from `hyprstream-discovery`
/// to the `PolicyClient` generated in this crate.
pub struct PolicyAuthProvider {
    client: crate::services::PolicyClient,
}

impl PolicyAuthProvider {
    /// Create a new policy-based authorization provider.
    pub fn new(client: crate::services::PolicyClient) -> Self {
        Self { client }
    }

}

#[async_trait(?Send)]
impl AuthorizationProvider for PolicyAuthProvider {
    async fn check(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: &str,
    ) -> anyhow::Result<bool> {
        self.client
            .check(&PolicyCheck {
                subject: subject.to_owned(),
                domain: domain.to_owned(),
                resource: resource.to_owned(),
                operation: operation.to_owned(),
            })
            .await
    }
}

// ============================================================================
// PdsRecordResolver — backs DiscoveryService getRecord/getRepo (#431)
// ============================================================================

/// One account's repo in the node's local PDS: its records (keyed by TID) and
/// the P-256 `#atproto` signing key that signs its commits.
///
/// The MST and signed commit are rebuilt deterministically from the record set
/// on demand (the MST shape is a pure function of the key set), so a stored repo
/// is just "the records + the signing key". This mirrors the e7 federation
/// publish path (`node_a_publish`) but as a queryable per-account store.
struct RepoState {
    /// P-256 `#atproto` signing key for this account's commits.
    signing_key: p256::ecdsa::SigningKey,
    /// `<collection>/<rkey>` is the MST key; here we key by `(collection, tid)`.
    records: BTreeMap<(String, Tid), ModelRecord>,
}

// ── RocksDB key scheme ───────────────────────────────────────────────────────
//
// DIDs and collection NSIDs never contain NUL bytes, so `\0` is a safe,
// unambiguous field separator for compound keys (same idea as the delimited
// keys `RocksDbUserStore` uses, just with a compound suffix here).
//
//   sk\0{did}                              -> 32-byte P-256 signing-key seed
//   rk\0{did}\0{collection}\0{tid}         -> canonical DAG-CBOR record bytes
//
// The record's rkey (a TID) is derived deterministically from the repo id
// (`PdsRecordStore::tid_for_repo`), so it is never minted or persisted. The
// `#atproto` signing key is held in memory (loaded from the 0600 secrets dir),
// never stored in this DB (#910a H1).

fn record_prefix(did: &str) -> Vec<u8> {
    format!("rk\0{did}\0").into_bytes()
}

fn record_key(did: &str, collection: &str, tid: Tid) -> Vec<u8> {
    format!("rk\0{did}\0{collection}\0{}", tid.encode()).into_bytes()
}

/// Split a full record key back into `(collection, tid)`, given the `did` it
/// was built for.
fn parse_record_key(key: &[u8], did: &str) -> AnyResult<(String, Tid)> {
    let prefix = record_prefix(did);
    let rest = key
        .strip_prefix(prefix.as_slice())
        .ok_or_else(|| anyhow!("record key missing expected prefix for {did:?}"))?;
    let rest = std::str::from_utf8(rest).context("record key suffix is not UTF-8")?;
    let (collection, tid_str) = rest
        .split_once('\0')
        .ok_or_else(|| anyhow!("record key missing collection/tid separator"))?;
    let tid = Tid::parse(tid_str)?;
    Ok((collection.to_owned(), tid))
}

/// Durable, RocksDB-backed PDS record store keyed by repo DID.
///
/// This is the per-account record store the DiscoveryService resolves against,
/// and that the registry service's register/commit paths publish into (#910a).
/// Persistence follows the same pattern as `RocksDbUserStore`
/// (`auth/rocksdb_store.rs`): a single `rocksdb::DB` at a well-known directory,
/// opened read-write by the *one* process that publishes, and read-only by
/// resolvers.
///
/// RocksDB allows exactly one read-write handle per directory; on a single
/// node the registry service (the publisher) is the sole writer, and the
/// discovery service (the resolver) opens the same directory read-only —
/// mirroring `RocksDbUserStore::open` / `open_readonly`. A read-only handle
/// snapshots the manifest at open time, so [`PdsRecordStore::open_readonly`]
/// callers that need a live view reopen per read (see [`RecordBacking::ReadOnly`]).
pub struct PdsRecordStore {
    backing: RecordBacking,
    /// This node's `#atproto` commit-signing key (P-256/ES256), held in memory
    /// and loaded from the 0600 secrets dir by both the publisher and the
    /// resolver. It is **never** persisted into the record DB (#910a H1) —
    /// putting a private key into 0644 RocksDB files would let any local reader
    /// forge signed PDS commits. Used to sign CAR proofs at read time.
    atproto_signing_key: p256::ecdsa::SigningKey,
}

enum RecordBacking {
    /// The sole writer: one long-lived, read-write RocksDB handle.
    ReadWrite(rocksdb::DB),
    /// A resolver: no handle is held between calls — each read opens a fresh
    /// `open_for_read_only` handle so writes from the publisher process become
    /// visible without a restart.
    ReadOnly(PathBuf),
}

impl PdsRecordStore {
    /// Open (or create) the durable store at `path` for read-write access.
    /// Exactly one process should hold this — the publisher (registry
    /// service). See the type-level docs for the single-writer rationale.
    pub fn open(path: &Path, atproto_signing_key: p256::ecdsa::SigningKey) -> AnyResult<Self> {
        std::fs::create_dir_all(path)
            .with_context(|| format!("failed to create PDS store dir {path:?}"))?;
        harden_store_dir(path)?;
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, path)
            .with_context(|| format!("failed to open PDS record store (rw) at {path:?}"))?;
        Ok(Self { backing: RecordBacking::ReadWrite(db), atproto_signing_key })
    }

    /// Open the store read-only at `path`, for resolvers that never write
    /// (the discovery service). Verifies the directory opens successfully now
    /// (surfacing a missing/corrupt store at startup) but does not hold the
    /// handle — see the type-level docs.
    pub fn open_readonly(path: &Path, atproto_signing_key: p256::ecdsa::SigningKey) -> AnyResult<Self> {
        let opts = readonly_opts();
        let _probe = rocksdb::DB::open_for_read_only(&opts, path, false)
            .with_context(|| format!("failed to open PDS record store (ro) at {path:?}"))?;
        Ok(Self { backing: RecordBacking::ReadOnly(path.to_path_buf()), atproto_signing_key })
    }

    /// Insert/replace a record for `did` under `collection` at `tid`. The
    /// account's `#atproto` commit key lives in memory (never in this DB —
    /// #910a H1). Used by the publishing path and by tests. Fails if this
    /// store was opened read-only.
    pub fn put_record(
        &self,
        did: &str,
        collection: &str,
        tid: Tid,
        record: ModelRecord,
    ) -> AnyResult<()> {
        let RecordBacking::ReadWrite(db) = &self.backing else {
            bail!("PdsRecordStore::put_record called on a read-only store");
        };
        db.put(record_key(did, collection, tid), record.to_dag_cbor())
            .context("PDS record store write failed")?;
        Ok(())
    }

    /// The stable record-key TID for `repo_id`, derived **deterministically**
    /// from the repo id (#910a M1).
    ///
    /// The rkey must be a pure function of `repo_id`: two distinct repos must
    /// never alias onto one `(collection, tid)` key (which would silently and
    /// permanently clobber one repo's record with the other's), and the same
    /// repo must always resolve to the same key so a re-publish is a pointer
    /// advance, not a new record. A wall-clock `Tid::now()` fails both (same
    /// microsecond → identical TID; and it needs a persisted mint with a
    /// get-then-put race). Instead derive the TID's 53-bit micros field and
    /// 10-bit clock id from `SHA-256(repo_id)` — a hash, not a timestamp:
    /// these records are not time-ordered, so 63 bits of collision resistance
    /// is exactly what we want, with no mint, no persistence, and no race.
    pub fn tid_for_repo(repo_id: &str) -> Tid {
        // SHA-256 output is always 32 bytes, so these indexes never panic and
        // need no fallible `try_into`.
        let d = sha2::Sha256::digest(repo_id.as_bytes());
        let micros = u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
        let clock_id = u16::from_be_bytes([d[8], d[9]]);
        Tid::from_micros(micros, clock_id)
    }

    /// Load the full record set + signing key for `did`, or `None` if the
    /// account has never published.
    fn load_repo(&self, did: &str) -> AnyResult<Option<RepoState>> {
        match &self.backing {
            RecordBacking::ReadWrite(db) => self.load_repo_from(db, did),
            RecordBacking::ReadOnly(path) => {
                let opts = readonly_opts();
                let db = rocksdb::DB::open_for_read_only(&opts, path, false)
                    .with_context(|| format!("failed to reopen PDS record store at {path:?}"))?;
                self.load_repo_from(&db, did)
            }
        }
    }

    fn load_repo_from(&self, db: &rocksdb::DB, did: &str) -> AnyResult<Option<RepoState>> {
        let prefix = record_prefix(did);
        let mut records = BTreeMap::new();
        // `prefix_iterator` without a configured `prefix_extractor` degrades to
        // a full forward scan from `prefix` — bound it manually.
        for item in db.prefix_iterator(&prefix) {
            let (key, value) = item.context("PDS store record scan failed")?;
            if !key.starts_with(prefix.as_slice()) {
                break;
            }
            let (collection, tid) = parse_record_key(&key, did)?;
            let record = ModelRecord::from_dag_cbor(&value)
                .with_context(|| format!("corrupt PDS record for {did}/{collection}/{tid}"))?;
            records.insert((collection, tid), record);
        }
        // "Never published" == no records for this DID. The `#atproto` key is
        // held in memory (never in the DB — #910a H1), so a repo's existence is
        // determined by its records, not by a stored key.
        if records.is_empty() {
            return Ok(None);
        }
        Ok(Some(RepoState { signing_key: self.atproto_signing_key.clone(), records }))
    }

    /// Build the MST over all of a repo's records and return it with the
    /// per-collection record-CID map (so callers can locate a target record).
    fn build_tree(repo: &RepoState) -> Node {
        // All records across all collections share one MST keyed by
        // `<collection>/<rkey>`. `Node::from_records` keys by Tid within one
        // collection NSID, so when a repo holds multiple collections we build a
        // per-collection subtree set is NOT how atproto works — atproto uses one
        // tree over `<collection>/<rkey>` keys. Our records today are a single
        // collection (`ai.hyprstream.model`), so build over that collection's
        // Tid→CID map. Mixed-collection repos are a follow-up (#910b).
        let mut by_tid: BTreeMap<Tid, hyprstream_pds::cid::Cid> = BTreeMap::new();
        for ((collection, tid), rec) in &repo.records {
            // Single-collection assumption (see note above): use the record's
            // own collection NSID as the MST collection.
            let _ = collection;
            by_tid.insert(*tid, rec.cid());
        }
        Node::from_records(hyprstream_pds::record::COLLECTION_NSID, &by_tid)
    }
}

fn readonly_opts() -> rocksdb::Options {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(false);
    opts
}

/// Restrict the record-store directory to owner-only (0700) on unix.
///
/// Defence in depth: even though the `#atproto` private key never lands in
/// this DB (#910a H1), the record *values* are what the node signs CAR proofs
/// over at read time — a store another local user could write would let them
/// inject records the node then serves as authentic. On non-unix this is a
/// no-op (the store path is expected to be an OS-managed private dir there).
fn harden_store_dir(path: &Path) -> AnyResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to set 0700 on PDS store dir {path:?}"))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

// ============================================================================
// PdsPublisher — the register/commit write side of #910a
// ============================================================================

/// Publishes/advances a repo's `ai.hyprstream.model` PDS record on this
/// node, over a read-write [`PdsRecordStore`]. Constructed once by the
/// registry service (the sole writer) and called from `handle_register` /
/// `handle_commit_with_author`.
pub struct PdsPublisher {
    store: Arc<PdsRecordStore>,
    /// This node's own `did:key` identity — the `repo` at-uri authority for
    /// every record this node publishes (single-node, self-hosted PDS; a
    /// per-account DID scheme is out of scope for #910a).
    did: String,
}

impl PdsPublisher {
    pub fn new(store: Arc<PdsRecordStore>, did: String) -> Self {
        Self { store, did }
    }

    /// Publish (or advance) the `ai.hyprstream.model` record for `repo_id`,
    /// pointing it at `current_oid_hex` (a git OID, hex-encoded).
    ///
    /// The record's rkey is derived deterministically from `repo_id`
    /// (`PdsRecordStore::tid_for_repo`) and reused on every call, so this
    /// always *replaces* the existing record's value rather than creating a
    /// new one — the atproto "pointer advance".
    ///
    /// Best-effort by design: the caller (registry service) should log and
    /// continue on error rather than fail the git operation that triggered
    /// this — the local git commit is the durable source of truth; a missed
    /// PDS publish is caught up by the next successful one (same
    /// eventual-consistency contract `git::promote` documents).
    pub fn publish(&self, repo_id: &str, current_oid_hex: &str) -> AnyResult<()> {
        let tid = PdsRecordStore::tid_for_repo(repo_id);
        let current_oid_cid = git_oid_to_cid_string(current_oid_hex)?;
        let record = ModelRecord::new(
            format!("at://{}", self.did),
            current_oid_cid,
            atproto_datetime_now(),
        )?;
        self.store
            .put_record(&self.did, hyprstream_pds::record::COLLECTION_NSID, tid, record)
    }
}

/// Encode a git OID (hex string, SHA-1 or SHA-256) as a `format: "cid"`
/// string for `ModelRecord::currentOid`.
///
/// TODO(#395): this is a placeholder encoding (a CIDv1 `raw` codec over the
/// OID's raw bytes, i.e. a *hash of the OID*, not an identity-preserving
/// encoding of it) — good enough to be deterministic and format-valid today,
/// but the real git-OID↔CID grammar belongs to the CID encoder work tracked
/// there. Once that lands, swap this call for the real encoder; nothing else
/// in the publish path needs to change.
fn git_oid_to_cid_string(oid_hex: &str) -> AnyResult<String> {
    let oid_bytes = hex::decode(oid_hex).with_context(|| format!("git OID {oid_hex:?} is not valid hex"))?;
    Ok(hyprstream_pds::cid::Cid::from_raw(&oid_bytes).encode())
}

/// The current UTC time in the atproto `datetime` lexicon format
/// (ISO-8601, millisecond precision, trailing `Z`).
fn atproto_datetime_now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
}

/// `RecordResolver` over a [`PdsRecordStore`]. Builds CAR proofs via the
/// reused `hyprstream_pds::car::build_record_proof_car` — no CAR/MST/commit
/// code is reinvented here.
pub struct PdsRecordResolver {
    store: Arc<PdsRecordStore>,
}

impl PdsRecordResolver {
    pub fn new(store: Arc<PdsRecordStore>) -> Self {
        Self { store }
    }
}

#[async_trait(?Send)]
impl RecordResolver for PdsRecordResolver {
    async fn resolve_record(
        &self,
        did: &str,
        collection: &str,
        rkey: &str,
    ) -> anyhow::Result<Option<RecordCarData>> {
        let tid = match Tid::parse(rkey) {
            Ok(t) => t,
            // A malformed rkey simply names no record — report NOT_FOUND, not an
            // error, so a bad address looks the same as an absent one.
            Err(_) => return Ok(None),
        };

        let Some(repo) = self.store.load_repo(did)? else {
            return Ok(None);
        };
        let key = (collection.to_owned(), tid);
        let Some(record) = repo.records.get(&key).cloned() else {
            return Ok(None);
        };

        // Rebuild the MST + signed commit, then build the CAR proof for the
        // target record (reusing the PDS CAR builder — #392).
        let tree = PdsRecordStore::build_tree(&repo);
        let root = tree.root_cid();
        let unsigned = hyprstream_pds::commit::UnsignedCommit::new(
            did.to_owned(),
            root,
            Tid::now(),
            None,
        );
        let commit = Commit::sign(&unsigned, &repo.signing_key);
        let proof = tree
            .proof(hyprstream_pds::record::COLLECTION_NSID, &tid)
            .ok_or_else(|| anyhow::anyhow!("record present but no MST proof — store inconsistent"))?;
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
        let car = build_record_proof_car(&commit, &proof, &node_blocks, &record);

        let uri = format!("at://{did}/{collection}/{rkey}");
        Ok(Some(RecordCarData { uri, car }))
    }

    async fn resolve_repo(&self, did: &str) -> anyhow::Result<Option<RecordCarData>> {
        let Some(repo) = self.store.load_repo(did)? else {
            return Ok(None);
        };
        if repo.records.is_empty() {
            return Ok(None);
        }

        // Full-repo CAR: a self-contained helper for the WHOLE repo (commit +
        // all MST nodes + all record blocks) does not exist in the PDS `car`
        // module yet (only `build_record_proof_car` for a single record). Rather
        // than reinvent the CAR/MST assembly here, return the proof CAR for ONE
        // record (the first, deterministically) as the repo-root proof. This is
        // documented as a deferral: getRepo currently returns a single-record
        // proof rooted at the same signed commit; a true all-records CAR is a
        // follow-up (needs a `build_repo_car` helper in hyprstream-pds).
        let Some(((collection, tid), record)) = repo.records.iter().next() else {
            return Ok(None);
        };
        let collection = collection.clone();
        let tid = *tid;
        let record = record.clone();

        let tree = PdsRecordStore::build_tree(&repo);
        let root = tree.root_cid();
        let unsigned = hyprstream_pds::commit::UnsignedCommit::new(
            did.to_owned(),
            root,
            Tid::now(),
            None,
        );
        let commit = Commit::sign(&unsigned, &repo.signing_key);
        let proof = tree
            .proof(hyprstream_pds::record::COLLECTION_NSID, &tid)
            .ok_or_else(|| anyhow::anyhow!("repo record has no MST proof — store inconsistent"))?;
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
        let car = build_record_proof_car(&commit, &proof, &node_blocks, &record);

        let uri = format!("at://{did}/{collection}/{}", tid.encode());
        Ok(Some(RecordCarData { uri, car }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod pds_store_tests {
    use super::*;

    fn sample_record(oid_suffix: &str) -> ModelRecord {
        ModelRecord::new(
            "at://did:key:zTestNode",
            format!("bafyreiexample{oid_suffix}0000000000000000000000000"),
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample record")
    }

    #[test]
    fn put_then_load_round_trips_through_rocksdb() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let store = PdsRecordStore::open(dir.path(), sk.clone()).expect("open");
        let tid = Tid::now();
        let did = "did:key:zTestNode";

        store
            .put_record(did, "ai.hyprstream.model", tid, sample_record("a"))
            .expect("put_record");

        let resolver = PdsRecordResolver::new(Arc::new(store));
        let got = tokio_test_block_on(resolver.resolve_record(did, "ai.hyprstream.model", &tid.encode()))
            .expect("resolve_record ok")
            .expect("record present");
        assert_eq!(got.uri, format!("at://{did}/ai.hyprstream.model/{}", tid.encode()));
    }

    #[test]
    fn readonly_store_survives_restart_and_sees_writer_updates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let did = "did:key:zTestNode2";
        let sk = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        let tid = Tid::now();

        {
            let store = PdsRecordStore::open(dir.path(), sk.clone()).expect("open rw");
            store
                .put_record(did, "ai.hyprstream.model", tid, sample_record("b"))
                .expect("put_record");
            // Writer handle dropped here — simulates a process restart.
        }

        let ro = PdsRecordStore::open_readonly(dir.path(), sk).expect("open ro");
        let resolver = PdsRecordResolver::new(Arc::new(ro));
        let got = tokio_test_block_on(resolver.resolve_repo(did))
            .expect("resolve_repo ok")
            .expect("repo present after reopen");
        assert!(got.uri.starts_with(&format!("at://{did}/")));
    }

    #[test]
    fn readonly_store_rejects_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sk = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
        // Create the directory with a real DB first so open_readonly succeeds.
        drop(PdsRecordStore::open(dir.path(), sk.clone()).expect("open rw"));

        let ro = PdsRecordStore::open_readonly(dir.path(), sk).expect("open ro");
        let err = ro
            .put_record("did:key:zX", "ai.hyprstream.model", Tid::now(), sample_record("c"))
            .unwrap_err();
        assert!(err.to_string().contains("read-only"));
    }

    #[test]
    fn tid_for_repo_is_deterministic_and_distinct() {
        let first = PdsRecordStore::tid_for_repo("repo-a");
        let second = PdsRecordStore::tid_for_repo("repo-a");
        assert_eq!(first, second, "the same repo must always get the same rkey");
        let other = PdsRecordStore::tid_for_repo("repo-b");
        assert_ne!(first, other, "distinct repos must not alias onto one rkey");
    }

    /// Minimal single-threaded block_on so these tests don't need a full
    /// `#[tokio::test]` runtime just to drive one `async_trait(?Send)` call.
    fn tokio_test_block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build current-thread runtime")
            .block_on(fut)
    }
}
