//! Discovery service — re-exports from `hyprstream-discovery`.
//!
//! The DiscoveryService implementation has moved to the `hyprstream-discovery` crate.
//! This module provides a `PolicyAuthProvider` that wraps `PolicyClient` to implement
//! the `AuthorizationProvider` trait expected by DiscoveryService.

pub use hyprstream_discovery::{AuthorizationProvider, DiscoveryService};

use crate::services::generated::policy_client::PolicyCheck;
use async_trait::async_trait;

use std::collections::BTreeMap;
use std::sync::Arc;

use hyprstream_discovery::{RecordCarData, RecordResolver};
use hyprstream_pds::car::build_record_proof_car;
use hyprstream_pds::commit::Commit;
use hyprstream_pds::mst::Node;
use hyprstream_pds::record::ModelRecord;
use hyprstream_pds::tid::Tid;
use parking_lot::RwLock;

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

/// In-memory PDS record store keyed by repo DID.
///
/// This is the per-account record store the DiscoveryService resolves against.
/// It is deliberately in-memory (the `hyprstream-pds` crate is storage-agnostic
/// by design); persistence is a follow-up. The publishing side (writing records
/// into this store when a model is registered + the atproto pointer advanced,
/// #394) is wired separately — this type is the read side #431 needs.
#[derive(Default)]
pub struct PdsRecordStore {
    repos: RwLock<BTreeMap<String, RepoState>>,
}

impl PdsRecordStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert/replace a record for `did` under `collection` at `tid`, using
    /// `signing_key` as the account's `#atproto` commit key. Used by the
    /// publishing path and by tests.
    pub fn put_record(
        &self,
        did: &str,
        collection: &str,
        tid: Tid,
        record: ModelRecord,
        signing_key: p256::ecdsa::SigningKey,
    ) {
        let mut repos = self.repos.write();
        let repo = repos.entry(did.to_owned()).or_insert_with(|| RepoState {
            signing_key: signing_key.clone(),
            records: BTreeMap::new(),
        });
        // Keep the most recently supplied signing key for the account.
        repo.signing_key = signing_key;
        repo.records.insert((collection.to_owned(), tid), record);
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
        // Tid→CID map. Mixed-collection repos are a follow-up.
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

        let repos = self.store.repos.read();
        let Some(repo) = repos.get(did) else {
            return Ok(None);
        };
        let key = (collection.to_owned(), tid);
        let Some(record) = repo.records.get(&key).cloned() else {
            return Ok(None);
        };

        // Rebuild the MST + signed commit, then build the CAR proof for the
        // target record (reusing the PDS CAR builder — #392).
        let tree = PdsRecordStore::build_tree(repo);
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
        let repos = self.store.repos.read();
        let Some(repo) = repos.get(did) else {
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

        let tree = PdsRecordStore::build_tree(repo);
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
