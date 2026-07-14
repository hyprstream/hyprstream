//! P1 placement directory — record ingestion + in-process index (#524).
//!
//! Day-1 ingestion is **polled `RecordResolver` reads**, not a firehose/moq_event
//! tail (that's explicitly out of scope — a day-2 ticket). Given a node DID, we
//! call [`crate::RecordResolver::resolve_repo`], parse the returned CARv1 blob
//! (`hyprstream_pds::car::parse_car_v1`), walk the repo's MST from the commit's
//! `data` root, and decode every `ai.hyprstream.placement.{node,workload,group,
//! groupItem}` record we find into an in-process index.
//!
//! # Trust posture — verified-by-construction ingest (#932)
//!
//! Membership is becoming a load-bearing input (scheduler capacity partitions
//! today; group ledger / spend authorization tomorrow — #924/#925), so the
//! index no longer trusts an ingested record graph merely because a resolver
//! handed it over. Ingest is **verified by construction**: when the resolver
//! can provide the DID's `#atproto` verifying key
//! ([`RecordResolver::resolve_verifying_key`]), [`PlacementIndex::ingest_did`]
//! verifies the repo CAR's commit signature against it. Regardless of key
//! availability, ingest **content-binds every CAR block to its declared CID**.
//! The commit block, each MST node, and each record block have their CID recomputed from their bytes and
//! checked against the CID the CAR / parent declared for them (mirroring
//! `hyprstream_pds::mst::Proof::verify`) — *before* any record enters the
//! index. The signature alone is not sufficient: `Commit::verify` only signs
//! the commit's own bytes (which name the MST root CID), so without per-block
//! CID binding an attacker could replay a genuine signed commit with
//! substituted MST-node/record blocks and forge membership; the content-bind
//! check closes that. A repo whose commit signature fails verification, whose
//! `commit.did` is not the requested DID, or whose any block's recomputed CID
//! ≠ its declared CID is refused — it cannot reach the derived indices, so it
//! cannot produce membership or any other fact derived from
//! [`PlacementIndex::is_member`]. This makes verification a property of the
//! index itself, not a caller obligation.
//!
//! When the resolver returns `Ok(None)` (it cannot provide a key for this DID
//! — e.g. a foreign DID whose DID-document `#atproto` method this crate can't
//! resolve yet), the index retains the day-1 trusted-resolver posture: the
//! injected `RecordResolver` is the same trusted, in-process resolver
//! `handle_get_repo` serves *from* (no untrusted network hop between the index
//! and its source) for authentication only; CID and structural validation are
//! still mandatory. DID-document `#atproto` key resolution for foreign repos is
//! the future federation-hardening follow-up that closes that gap; until then
//! the *per-candidate authz check* in `queryCandidates` remains the hard
//! security boundary, with the index as best-effort scheduling metadata.
//!
//! # Group membership — bidirectional consent
//!
//! A node is *effectively* a member of a group only when **both** sides agree:
//! the group owner published a `GroupItemRecord{group, subject}` naming the
//! node, **and** the node's own `NodeRecord.groups` lists that group's at-uri
//! (explicit node-side consent). Neither claim alone is membership — see
//! [`PlacementIndex::is_member`].
//!
//! Group membership is exposed to `queryCandidates`' label-selector matching as
//! a **synthetic label**, not a schema change: for every group a node is an
//! effective member of, the index adds a label `("group/<group-at-uri>", "true")`
//! to that node's effective label set (see [`PlacementIndex::effective_labels`]).
//! Each group gets its own unique label *key* (rather than several `("group",
//! X)` pairs under one shared key) because [`crate::scheduling::LabelSelector`]
//! looks up the *first* entry matching a key (the k8s label model: one value per
//! key) — multiple same-keyed pairs would only ever let the first one be
//! evaluated, silently breaking multi-group membership. A caller checks
//! membership in group `G` with `LabelSelector::new(format!("group/{G}"),
//! SelectorOp::Exists, vec![])`.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, ensure, Result};
use p256::ecdsa::VerifyingKey;
use parking_lot::RwLock;

use hyprstream_pds::car::parse_car_v1;
use hyprstream_pds::cid::Cid;
use hyprstream_pds::commit::Commit;
use hyprstream_pds::dag_cbor::DagCbor;
use hyprstream_pds::mst::NodeData;
use hyprstream_pds::placement::{group, group_item, node, workload};
use hyprstream_pds::placement::{GroupItemRecord, GroupRecord, NodeRecord, WorkloadRecord};

use crate::RecordResolver;

/// One node's durable (non-volatile) placement facts, decoded from its
/// `ai.hyprstream.placement.node` record.
#[derive(Debug, Clone, Default)]
pub struct NodeFacts {
    /// at-uri of the `NodeRecord` itself (`at://<did>/ai.hyprstream.placement.node/<rkey>`).
    pub record_uri: String,
    /// Declared `{key, value}` labels.
    pub labels: Vec<(String, String)>,
    /// Declared capacity (k8s-quantity strings) — NOT live/allocatable.
    /// `queryCandidates` uses the *live* liveness report for resource matching,
    /// never this durable declaration (see the module docs on trust posture).
    pub declared_resources: Vec<(String, String)>,
    /// at-uris of groups this node's own record *consents* to.
    pub consented_groups: Vec<String>,
}

/// One decoded `ai.hyprstream.placement.group` record's facts.
#[derive(Debug, Clone)]
struct GroupFacts {
    #[allow(dead_code)] // carried for future group-scoped listing; unused by queryCandidates today
    name: String,
    #[allow(dead_code)]
    owner_did: String,
}

/// Everything ingested from one DID's repo CAR in the most recent poll. Kept
/// so a re-poll can *replace* (not merge-forever-append) that DID's
/// contribution to the derived indices — a group/groupItem record deleted
/// upstream must be able to disappear from the index on the next poll.
#[derive(Debug, Clone, Default)]
struct DidSnapshot {
    node: Option<NodeFacts>,
    groups: Vec<(String, String, String)>, // (group at-uri, name, owner_did)
    group_items: Vec<GroupItemRecord>,
    /// Decoded but not indexed further (day-1 scope; see module docs).
    workload_count: usize,
}

/// In-process placement directory index. Rebuilt by polling
/// [`RecordResolver::resolve_repo`] for a bootstrap set of node DIDs (day-1
/// ingestion — see the module docs). Read from the `queryCandidates` request
/// path; written by the poller. `parking_lot::RwLock` mirrors the precedent
/// already used for `DiscoveryService`'s other in-memory maps.
#[derive(Default)]
pub struct PlacementIndex {
    /// Per-DID last-ingested snapshot (the source of truth for re-poll replace
    /// semantics).
    raw: RwLock<HashMap<String, DidSnapshot>>,
    /// Derived: node DID -> facts. Recomputed after every ingest.
    nodes: RwLock<HashMap<String, NodeFacts>>,
    /// Derived: group at-uri -> facts.
    groups: RwLock<HashMap<String, GroupFacts>>,
    /// Derived: group at-uri -> DIDs the group OWNER claims as members
    /// (`GroupItemRecord.subject`). Effective membership additionally requires
    /// the node's own consent — see [`Self::is_member`].
    group_claims: RwLock<HashMap<String, Vec<String>>>,
}

impl PlacementIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Poll `did`'s repo via `resolver` and (re-)ingest its placement records,
    /// replacing whatever this DID contributed on a previous poll.
    ///
    /// `Ok(())` with no visible effect when the DID has no stored repo (nothing
    /// to ingest) or its repo carries no placement records — this is not an
    /// error, just an empty directory entry.
    ///
    /// **Verified by construction (#932):** when the resolver provides a
    /// verifying key for `did` ([`RecordResolver::resolve_verifying_key`]), the
    /// repo CAR's commit signature is verified. Every CAR block is always
    /// content-bound to its declared CID before any record enters the index. A
    /// repo that fails either check — or whose
    /// `commit.did` is not `did` — is refused — its records never reach the
    /// derived indices — and this returns `Err` (the caller —
    /// `handle_report_node_liveness` — logs and continues, liveness still
    /// recorded). When the resolver returns `Ok(None)` the trusted-resolver
    /// day-1 posture applies (see the module trust-posture docs).
    pub async fn ingest_did(&self, resolver: &dyn RecordResolver, did: &str) -> Result<()> {
        let repo = match resolver.resolve_repo(did).await {
            Ok(repo) => repo,
            Err(e) => {
                self.clear_did(did);
                return Err(anyhow!("resolve_repo({did}) failed: {e}"));
            }
        };
        let Some(repo) = repo else {
            // No repo stored for this DID — clear any stale contribution and stop.
            self.clear_did(did);
            return Ok(());
        };
        let verify_key = match resolver.resolve_verifying_key(did).await {
            Ok(key) => key,
            Err(e) => {
                self.clear_did(did);
                return Err(anyhow!("resolve_verifying_key({did}) failed: {e}"));
            }
        };
        // Decode (and, when a key is available, signature-verify) the CAR
        // *before* touching the index. A verification failure leaves the index
        // untouched for this DID and surfaces as an error — the previous
        // contribution, if any, is cleared so stale facts can't survive a poll
        // that now fails to verify.
        let snapshot = match Self::decode_repo_car(did, &repo.car, verify_key.as_ref()) {
            Ok(snap) => snap,
            Err(e) => {
                self.clear_did(did);
                return Err(e);
            }
        };
        self.raw.write().insert(did.to_owned(), snapshot);
        self.recompute_derived();
        Ok(())
    }

    fn clear_did(&self, did: &str) {
        self.raw.write().remove(did);
        self.recompute_derived();
    }

    /// Parse a repo CARv1 blob and decode every `ai.hyprstream.placement.*`
    /// record reachable from the commit's MST root into a [`DidSnapshot`].
    ///
    /// Every CAR block is content-bound to its declared CID. When `verify_key`
    /// is `Some`, the CAR's commit signature is additionally verified against
    /// it (the verified-by-construction gate of #932). The commit
    /// block, each MST node, and each record block have their CID recomputed
    /// from their bytes and checked against the CID the CAR/parent declared for
    /// them before they are trusted (mirrors `hyprstream_pds::mst::Proof::verify`,
    /// which recomputes every node CID). A signature alone is not enough:
    /// `Commit::verify` only signs the commit's own bytes (which name the MST
    /// root CID), so without per-block binding an attacker could replay a
    /// genuine signed commit with substituted MST-node/record blocks and forge
    /// membership. A recomputed-CID ≠ declared-CID mismatch returns `Err` (the
    /// caller refuses to ingest). `None` skips signature authentication only
    /// (trusted-resolver posture); content and structural validation remain.
    fn decode_repo_car(
        did: &str,
        car: &[u8],
        verify_key: Option<&VerifyingKey>,
    ) -> Result<DidSnapshot> {
        let (roots, blocks) = parse_car_v1(car)?;
        ensure!(
            roots.len() == 1,
            "repo CAR for {did} must have exactly one commit root (got {})",
            roots.len()
        );
        let root_cid = *roots
            .first()
            .ok_or_else(|| anyhow!("repo CAR for {did} has no root"))?;
        let mut block_map = HashMap::with_capacity(blocks.len());
        for (cid, bytes) in blocks {
            verify_block_cid(did, cid, &bytes, "CAR")?;
            ensure!(
                block_map.insert(cid, bytes).is_none(),
                "repo CAR for {did} contains duplicate block CID {cid}"
            );
        }
        let commit_bytes = block_map
            .get(&root_cid)
            .ok_or_else(|| anyhow!("repo CAR for {did} missing its commit block"))?;
        let commit = Commit::from_dag_cbor(commit_bytes)?;
        // A validly-signed repo for DID A must not be replayable under DID B:
        // even a commit whose signature verifies under the presented key is
        // refused if its `did` is not the one we were asked to ingest.
        ensure!(
            commit.did == did,
            "repo CAR for {did} carries a commit for a different DID {}",
            commit.did
        );
        if let Some(vk) = verify_key {
            commit.verify(vk).map_err(|e| {
                anyhow!("repo CAR for {did} failed commit-signature verification: {e}")
            })?;
        }

        let mut entries = Vec::new();
        walk_mst(commit.data, &block_map, &mut entries, did)?;

        let mut snapshot = DidSnapshot::default();
        for (key, record_cid) in entries {
            let (collection, _rkey) = key
                .split_once('/')
                .ok_or_else(|| anyhow!("repo CAR for {did} has malformed MST key {key:?}"))?;
            let bytes = block_map.get(&record_cid).ok_or_else(|| {
                anyhow!("repo CAR for {did} missing record block {record_cid} referenced by MST key {key}")
            })?;

            if collection == node::COLLECTION_NSID {
                let rec = NodeRecord::from_dag_cbor(bytes)
                    .map_err(|e| anyhow!("invalid {collection} record at {key}: {e}"))?;
                snapshot.node = Some(NodeFacts {
                    record_uri: format!("at://{did}/{key}"),
                    labels: rec.labels.into_iter().map(|l| (l.key, l.value)).collect(),
                    declared_resources: rec
                        .resources
                        .into_iter()
                        .map(|r| (r.name, r.capacity))
                        .collect(),
                    consented_groups: rec.groups,
                });
            } else if collection == group::COLLECTION_NSID {
                let rec = GroupRecord::from_dag_cbor(bytes)
                    .map_err(|e| anyhow!("invalid {collection} record at {key}: {e}"))?;
                let uri = format!("at://{did}/{key}");
                snapshot.groups.push((uri, rec.name, rec.owner_did));
            } else if collection == group_item::COLLECTION_NSID {
                let rec = GroupItemRecord::from_dag_cbor(bytes)
                    .map_err(|e| anyhow!("invalid {collection} record at {key}: {e}"))?;
                snapshot.group_items.push(rec);
            } else if collection == workload::COLLECTION_NSID {
                // Decoded for completeness per #524 P1 scope; workload placement
                // decisions aren't consumed by queryCandidates (day-1 scope).
                WorkloadRecord::from_dag_cbor(bytes)
                    .map_err(|e| anyhow!("invalid {collection} record at {key}: {e}"))?;
                snapshot.workload_count += 1;
            }
        }
        Ok(snapshot)
    }

    /// Rebuild the derived (nodes / groups / group_claims) maps from `raw` in
    /// full. Simple full-rebuild rather than incremental patching — correct by
    /// construction (a deleted upstream record can't leak forever) and cheap at
    /// fleet scale; this is an index refresh, not a hot path.
    fn recompute_derived(&self) {
        let raw = self.raw.read();
        let mut nodes = HashMap::new();
        let mut groups = HashMap::new();
        let mut claims: HashMap<String, Vec<String>> = HashMap::new();
        for (did, snap) in raw.iter() {
            if let Some(n) = &snap.node {
                nodes.insert(did.clone(), n.clone());
            }
            for (uri, name, owner_did) in &snap.groups {
                groups.insert(
                    uri.clone(),
                    GroupFacts {
                        name: name.clone(),
                        owner_did: owner_did.clone(),
                    },
                );
            }
            for item in &snap.group_items {
                claims
                    .entry(item.group.clone())
                    .or_default()
                    .push(item.subject.clone());
            }
        }
        drop(raw);
        *self.nodes.write() = nodes;
        *self.groups.write() = groups;
        *self.group_claims.write() = claims;
    }

    /// All node DIDs currently known to the index (have an ingested
    /// `NodeRecord`).
    pub fn known_node_dids(&self) -> Vec<String> {
        self.nodes.read().keys().cloned().collect()
    }

    /// The at-uri of `did`'s `NodeRecord`, if known.
    pub fn record_uri(&self, did: &str) -> Option<String> {
        self.nodes.read().get(did).map(|f| f.record_uri.clone())
    }

    /// Whether `node_did` is an *effective* (bidirectional-consent) member of
    /// `group_uri`: the group owner must have published a `GroupItemRecord`
    /// naming this node, **and** the node's own `NodeRecord.groups` must list
    /// this group. Either claim alone is not membership.
    ///
    /// **Trust statement (#932):** membership facts returned here are always
    /// content-bound and are signature-verified when the resolver supplies a
    /// key. A `GroupItemRecord` / `NodeRecord` whose repo CAR commit failed
    /// signature verification against the resolver-provided `#atproto` key,
    /// whose `commit.did` is not the
    /// requested DID, **or whose any block's recomputed CID ≠ its declared CID**
    /// never entered the index, so it cannot back a `true` result. The per-block
    /// CID binding is what stops a replayed genuine signed commit carrying
    /// substituted blocks from forging membership. See the module trust-posture
    /// docs for the verified-by-construction ingest gate and its `Ok(None)`
    /// (trusted-resolver) fallback.
    pub fn is_member(&self, node_did: &str, group_uri: &str) -> bool {
        let node_consents = self
            .nodes
            .read()
            .get(node_did)
            .is_some_and(|f| f.consented_groups.iter().any(|g| g == group_uri));
        if !node_consents {
            return false;
        }
        self.group_claims
            .read()
            .get(group_uri)
            .is_some_and(|members| members.iter().any(|m| m == node_did))
    }

    /// A node's effective label set for selector matching: its declared
    /// `NodeRecord.labels` plus one synthetic `("group/<uri>", "true")` label
    /// per group it is an *effective* (bidirectional-consent) member of. See
    /// the module docs for why group membership uses a per-group label key.
    pub fn effective_labels(&self, node_did: &str) -> Vec<(String, String)> {
        let (mut labels, consented_groups) = match self.nodes.read().get(node_did) {
            Some(f) => (f.labels.clone(), f.consented_groups.clone()),
            None => return Vec::new(),
        };
        for group_uri in &consented_groups {
            if self.is_member(node_did, group_uri) {
                labels.push((format!("group/{group_uri}"), "true".to_owned()));
            }
        }
        labels
    }

    /// A node's declared (durable) capacity — NOT the live/allocatable figure
    /// `queryCandidates` matches resource requests against. Exposed mainly for
    /// tests / introspection.
    pub fn declared_resources(&self, node_did: &str) -> Vec<(String, String)> {
        self.nodes
            .read()
            .get(node_did)
            .map(|f| f.declared_resources.clone())
            .unwrap_or_default()
    }
}

/// Hard ceiling on the number of MST node visits per repo-CAR walk. An
/// untrusted CAR is not a well-formed tree by assumption: it can be
/// DAG-shaped (many `l`/`t` links pointing at the same child) or cyclic, so a
/// depth bound alone does not bound total work — a poisoned CAR could fan out
/// to visit the same subtrees exponentially, or loop forever. The `visited`
/// CID set makes the walk terminate (a node is expanded at most once); this
/// budget is the belt-and-suspenders ceiling that caps total node visits even
/// against a broad-but-acyclic adversarial CAR, and is chosen well above any
/// legitimate repo's node count.
const MST_MAX_NODE_VISITS: usize = 1 << 20; // 1,048,576 nodes

/// Recompute a block's CID from its bytes and confirm it matches the CID its
/// container (the CAR root list, a parent MST node's child pointer, or an MST
/// entry value) declared for it. A mismatch means the bytes were substituted
/// under a genuine CID — the whole ingest is rejected (fail closed). Mirrors
/// `hyprstream_pds::mst::Proof::verify`, which recomputes every node CID rather
/// than trusting the declared one.
fn verify_block_cid(did: &str, declared: Cid, bytes: &[u8], kind: &str) -> Result<()> {
    let actual = Cid::from_dag_cbor(bytes);
    ensure!(
        actual == declared,
        "repo CAR for {did}: {kind} block content/CID mismatch (declared {declared}, recomputed {actual})"
    );
    Ok(())
}

/// Maximum MST depth permitted during a walk. Real atproto MSTs are shallow
/// (a node splits every 2 leading key-characters, so a 256-char key caps depth
/// near 128); this bound just defends a CAR shaped to stack-overflow the
/// recursive walk (an attacker-supplied cyclic-ish pointer chain).
const MST_MAX_DEPTH: usize = 1024;

/// Enumerate every `(full_key, record_cid)` entry reachable from an MST node,
/// depth-first in key order: left subtree, then each entry (recursing into its
/// right subtree before the next entry). Mirrors the encode side
/// (`Node::to_node_data_rec` in `hyprstream_pds::mst`) exactly, including its
/// prefix-compression convention: `entry.p` compresses against the *previous
/// entry within this same node* only (resets to 0 at the first entry of every
/// node) — never against a key from a different node.
///
/// **Untrusted-ingest safety (#932):** `visited` records every node CID already
/// expanded so a DAG-shaped or cyclic CAR cannot revisit the same subtree, and
/// `budget` is a decrementing total-visit ceiling. Either guard trips a
/// fail-closed `Err` — the whole CAR is rejected rather than partially
/// ingested, so no derived membership fact is produced from an adversarial
/// graph.
///
fn walk_mst(
    node_cid: Cid,
    blocks: &HashMap<Cid, Vec<u8>>,
    out: &mut Vec<(String, Cid)>,
    did: &str,
) -> Result<()> {
    let mut visited = HashSet::new();
    let mut budget = MST_MAX_NODE_VISITS;
    walk_mst_inner(node_cid, blocks, out, did, 0, &mut visited, &mut budget)
}

fn walk_mst_inner(
    node_cid: Cid,
    blocks: &HashMap<Cid, Vec<u8>>,
    out: &mut Vec<(String, Cid)>,
    did: &str,
    depth: usize,
    visited: &mut HashSet<Cid>,
    budget: &mut usize,
) -> Result<()> {
    ensure!(
        depth <= MST_MAX_DEPTH,
        "MST walk for {did} exceeded max depth {MST_MAX_DEPTH} (possible CAR stack-overflow DoS)"
    );
    ensure!(
        *budget > 0,
        "MST walk exceeded the {MST_MAX_NODE_VISITS}-node visit budget (adversarial repo CAR?)"
    );
    *budget -= 1;
    ensure!(
        visited.insert(node_cid),
        "MST node {node_cid} is reachable more than once (non-tree repo CAR); refusing ingest"
    );
    let bytes = blocks
        .get(&node_cid)
        .ok_or_else(|| anyhow!("MST node {node_cid} missing from repo CAR blocks"))?;
    let value = DagCbor::decode(bytes)?;
    let data = NodeData::from_value(&value)?;

    if let Some(left) = data.l {
        walk_mst_inner(left, blocks, out, did, depth + 1, visited, budget)?;
    }
    let mut prev_key: Option<String> = None;
    for entry in &data.e {
        let key = match &prev_key {
            Some(prev) => {
                let prev_bytes = prev.as_bytes();
                let take = entry.p.min(prev_bytes.len());
                let mut full = prev_bytes[..take].to_vec();
                full.extend_from_slice(&entry.k);
                String::from_utf8(full).map_err(|e| anyhow!("MST entry key is not utf8: {e}"))?
            }
            None => String::from_utf8(entry.k.clone())
                .map_err(|e| anyhow!("MST entry key is not utf8: {e}"))?,
        };
        out.push((key.clone(), entry.v));
        prev_key = Some(key);
        if let Some(right) = entry.t {
            walk_mst_inner(right, blocks, out, did, depth + 1, visited, budget)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use hyprstream_pds::car::{build_car_v1, parse_car_v1};
    use hyprstream_pds::commit::UnsignedCommit;
    use hyprstream_pds::mst::Node;
    use hyprstream_pds::tid::Tid;
    use p256::ecdsa::{SigningKey as P256SigningKey, VerifyingKey as P256VerifyingKey};

    use crate::service::RecordCarData;

    const NODE_DID: &str = "did:web:node1.example.com";
    const OWNER_DID: &str = "did:web:owner.example.com";

    /// Build a repo CARv1 for `did` from a set of already-encoded `(full_key,
    /// dag_cbor_bytes)` records (mixed collections OK — the repo MST spans all
    /// of them, matching real atproto repos), returning the CAR bytes **and**
    /// the `#atproto` verifying key the commit was signed with (so the test
    /// resolver can hand it to the verified-by-construction ingest gate).
    fn repo_car(did: &str, records: &[(String, Vec<u8>)]) -> (Vec<u8>, P256VerifyingKey) {
        let signing_key = P256SigningKey::random(&mut rand::rngs::OsRng);
        let verifying_key = P256VerifyingKey::from(&signing_key);
        let mut keyed: BTreeMap<String, Cid> = BTreeMap::new();
        let mut record_blocks: Vec<(Cid, Vec<u8>)> = Vec::new();
        for (key, bytes) in records {
            let cid = Cid::from_dag_cbor(bytes);
            keyed.insert(key.clone(), cid);
            record_blocks.push((cid, bytes.clone()));
        }
        let tree = Node::from_keyed_records(&keyed);
        let root = tree.root_cid();
        let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
        let unsigned = UnsignedCommit::new(did.to_owned(), root, Tid::now(), None);
        let commit = hyprstream_pds::commit::Commit::sign(&unsigned, &signing_key);

        let mut blocks: Vec<(Cid, Vec<u8>)> = vec![(commit.cid(), commit.to_dag_cbor())];
        for (cid, data) in node_blocks {
            blocks.push((cid, data.encode()));
        }
        blocks.extend(record_blocks);
        (build_car_v1(&[commit.cid()], &blocks), verifying_key)
    }

    fn node_record_key(rkey: &str) -> String {
        format!("{}/{rkey}", node::COLLECTION_NSID)
    }
    fn group_record_key(rkey: &str) -> String {
        format!("{}/{rkey}", group::COLLECTION_NSID)
    }
    fn group_item_record_key(rkey: &str) -> String {
        format!("{}/{rkey}", group_item::COLLECTION_NSID)
    }

    /// A resolver serving fixed, per-DID repo CARs (built with real records)
    /// and the `#atproto` verifying key each was signed with — so the
    /// verified-by-construction ingest gate is armed in tests.
    struct FixedResolver {
        repos: HashMap<String, (Vec<u8>, P256VerifyingKey)>,
    }

    #[async_trait(?Send)]
    impl RecordResolver for FixedResolver {
        async fn resolve_record(
            &self,
            _did: &str,
            _collection: &str,
            _rkey: &str,
        ) -> Result<Option<RecordCarData>> {
            Ok(None)
        }
        async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>> {
            Ok(self.repos.get(did).map(|(car, _vk)| RecordCarData {
                uri: format!("at://{did}"),
                car: car.clone(),
            }))
        }
        async fn resolve_verifying_key(&self, did: &str) -> Result<Option<VerifyingKey>> {
            Ok(self.repos.get(did).map(|(_car, vk)| *vk))
        }
    }

    fn sample_node_record(groups: Vec<String>) -> NodeRecord {
        NodeRecord::new(
            format!("at://{NODE_DID}"),
            vec![node::Label {
                key: "zone".into(),
                value: "us-east".into(),
            }],
            vec![node::Resource {
                name: "nvidia.com/gpu".into(),
                capacity: "8".into(),
            }],
            groups,
            "2026-06-23T12:34:56.789Z",
        )
        .unwrap()
    }

    #[tokio::test]
    async fn ingest_decodes_node_record_labels_and_resources() {
        let rec = sample_node_record(vec![]);
        let car = repo_car(NODE_DID, &[(node_record_key("3a"), rec.to_dag_cbor())]);
        let resolver = FixedResolver {
            repos: HashMap::from([(NODE_DID.to_owned(), car)]),
        };
        let index = PlacementIndex::new();
        index.ingest_did(&resolver, NODE_DID).await.unwrap();

        assert_eq!(index.known_node_dids(), vec![NODE_DID.to_owned()]);
        assert_eq!(
            index.record_uri(NODE_DID).as_deref(),
            Some(format!("at://{NODE_DID}/{}", node_record_key("3a")).as_str())
        );
        let labels = index.effective_labels(NODE_DID);
        assert!(labels.contains(&("zone".to_owned(), "us-east".to_owned())));
        assert_eq!(
            index.declared_resources(NODE_DID),
            vec![("nvidia.com/gpu".to_owned(), "8".to_owned())]
        );
    }

    #[tokio::test]
    async fn ingest_missing_repo_is_not_an_error_and_yields_no_node() {
        let resolver = FixedResolver {
            repos: HashMap::new(),
        };
        let index = PlacementIndex::new();
        index.ingest_did(&resolver, NODE_DID).await.unwrap();
        assert!(index.known_node_dids().is_empty());
    }

    #[tokio::test]
    async fn group_membership_requires_both_owner_claim_and_node_consent() {
        let group_uri_key = group_record_key("3g");
        let group_rec = GroupRecord::new(
            "east-coast-gpus",
            OWNER_DID,
            None,
            "2026-06-23T12:34:56.789Z",
        )
        .unwrap();
        let group_uri = format!("at://{OWNER_DID}/{group_uri_key}");

        // Case 1: owner claims the node, but the node's own record does NOT
        // list the group (no node-side consent) -> not a member.
        {
            let owner_car = repo_car(
                OWNER_DID,
                &[
                    (group_uri_key.clone(), group_rec.to_dag_cbor()),
                    (
                        group_item_record_key("3i"),
                        GroupItemRecord::new(
                            group_uri.clone(),
                            NODE_DID,
                            "2026-06-23T12:34:56.789Z",
                        )
                        .unwrap()
                        .to_dag_cbor(),
                    ),
                ],
            );
            let node_rec = sample_node_record(vec![]); // no consent
            let node_car = repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);
            let resolver = FixedResolver {
                repos: HashMap::from([
                    (OWNER_DID.to_owned(), owner_car),
                    (NODE_DID.to_owned(), node_car),
                ]),
            };
            let index = PlacementIndex::new();
            index.ingest_did(&resolver, OWNER_DID).await.unwrap();
            index.ingest_did(&resolver, NODE_DID).await.unwrap();
            assert!(
                !index.is_member(NODE_DID, &group_uri),
                "owner-claim without node consent must NOT be membership"
            );
            assert!(!index
                .effective_labels(NODE_DID)
                .iter()
                .any(|(k, _)| k == &format!("group/{group_uri}")));
        }

        // Case 2: node consents (lists the group in its own record), but the
        // owner never published a GroupItemRecord naming it -> not a member.
        {
            let owner_car = repo_car(
                OWNER_DID,
                &[(group_uri_key.clone(), group_rec.to_dag_cbor())],
            );
            let node_rec = sample_node_record(vec![group_uri.clone()]);
            let node_car = repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);
            let resolver = FixedResolver {
                repos: HashMap::from([
                    (OWNER_DID.to_owned(), owner_car),
                    (NODE_DID.to_owned(), node_car),
                ]),
            };
            let index = PlacementIndex::new();
            index.ingest_did(&resolver, OWNER_DID).await.unwrap();
            index.ingest_did(&resolver, NODE_DID).await.unwrap();
            assert!(
                !index.is_member(NODE_DID, &group_uri),
                "node consent without an owner claim must NOT be membership"
            );
        }

        // Case 3: both sides agree -> effective member, surfaced as a synthetic label.
        {
            let owner_car = repo_car(
                OWNER_DID,
                &[
                    (group_uri_key.clone(), group_rec.to_dag_cbor()),
                    (
                        group_item_record_key("3i"),
                        GroupItemRecord::new(
                            group_uri.clone(),
                            NODE_DID,
                            "2026-06-23T12:34:56.789Z",
                        )
                        .unwrap()
                        .to_dag_cbor(),
                    ),
                ],
            );
            let node_rec = sample_node_record(vec![group_uri.clone()]);
            let node_car = repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);
            let resolver = FixedResolver {
                repos: HashMap::from([
                    (OWNER_DID.to_owned(), owner_car),
                    (NODE_DID.to_owned(), node_car),
                ]),
            };
            let index = PlacementIndex::new();
            index.ingest_did(&resolver, OWNER_DID).await.unwrap();
            index.ingest_did(&resolver, NODE_DID).await.unwrap();
            assert!(
                index.is_member(NODE_DID, &group_uri),
                "both claims present -> member"
            );
            assert!(index
                .effective_labels(NODE_DID)
                .iter()
                .any(|(k, v)| k == &format!("group/{group_uri}") && v == "true"));
        }
    }

    #[tokio::test]
    async fn reingesting_a_did_replaces_its_prior_contribution() {
        let rec_a = sample_node_record(vec![]);
        let car_a = repo_car(NODE_DID, &[(node_record_key("3a"), rec_a.to_dag_cbor())]);
        let resolver_a = FixedResolver {
            repos: HashMap::from([(NODE_DID.to_owned(), car_a)]),
        };
        let index = PlacementIndex::new();
        index.ingest_did(&resolver_a, NODE_DID).await.unwrap();
        assert_eq!(index.declared_resources(NODE_DID).len(), 1);

        // Re-poll with an empty repo (record deleted upstream) — must clear, not
        // leave the stale entry around forever.
        let resolver_b = FixedResolver {
            repos: HashMap::new(),
        };
        index.ingest_did(&resolver_b, NODE_DID).await.unwrap();
        assert!(
            index.known_node_dids().is_empty(),
            "stale contribution must be cleared on re-poll"
        );
    }

    #[tokio::test]
    async fn invalid_commit_signature_does_not_produce_membership() {
        // #932 — verified-by-construction ingest: a repo CAR whose commit
        // signature fails verification against the resolver-provided `#atproto`
        // key must be REFUSED at ingest, so its records cannot produce
        // membership (or any other derived fact). Mirror the "both sides agree
        // -> member" setup, but have the resolver report a *wrong* key for the
        // owner so the GroupItemRecord claiming NODE_DID never enters the index.
        let group_uri_key = group_record_key("3g");
        let group_rec = GroupRecord::new(
            "east-coast-gpus",
            OWNER_DID,
            None,
            "2026-06-23T12:34:56.789Z",
        )
        .unwrap();
        let group_uri = format!("at://{OWNER_DID}/{group_uri_key}");

        let (owner_car, _owner_vk) = repo_car(
            OWNER_DID,
            &[
                (group_uri_key.clone(), group_rec.to_dag_cbor()),
                (
                    group_item_record_key("3i"),
                    GroupItemRecord::new(group_uri.clone(), NODE_DID, "2026-06-23T12:34:56.789Z")
                        .unwrap()
                        .to_dag_cbor(),
                ),
            ],
        );
        // A key the owner's commit was NOT signed by -> verification must fail.
        let wrong_vk = P256VerifyingKey::from(&P256SigningKey::random(&mut rand::rngs::OsRng));

        // The node consents to the group (so membership would hold if the
        // owner's claim were ingested); its own repo verifies normally.
        let node_rec = sample_node_record(vec![group_uri.clone()]);
        let (node_car, node_vk) =
            repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);

        let resolver = FixedResolver {
            repos: HashMap::from([
                (OWNER_DID.to_owned(), (owner_car, wrong_vk)),
                (NODE_DID.to_owned(), (node_car, node_vk)),
            ]),
        };
        let index = PlacementIndex::new();
        // Owner ingest must fail closed — a commit that does not verify is
        // refused (the caller logs and continues; liveness is unaffected).
        assert!(
            index.ingest_did(&resolver, OWNER_DID).await.is_err(),
            "a repo whose commit signature fails verification must be refused at ingest"
        );
        index.ingest_did(&resolver, NODE_DID).await.unwrap();

        assert!(
            !index.is_member(NODE_DID, &group_uri),
            "an unverified GroupItemRecord must not produce membership even with node consent"
        );
        assert!(!index
            .effective_labels(NODE_DID)
            .iter()
            .any(|(k, _)| k == &format!("group/{group_uri}")));
    }

    #[tokio::test]
    async fn substituted_block_under_genuine_signed_commit_does_not_produce_membership() {
        // Security: the commit signature alone is not enough — `Commit::verify`
        // only signs the commit's own bytes (which name the MST root CID), so
        // an attacker who replays a genuine signed commit but SUBSTITUTES a
        // record block's bytes under its declared CID can otherwise forge a
        // GroupItemRecord claiming NODE_DID. The per-block CID content-binding
        // check at ingest must catch this: the recomputed CID ≠ declared CID,
        // so the whole ingest is refused (fail closed) and no membership is
        // produced — even though the commit signature still verifies.
        let group_uri_key = group_record_key("3g");
        let group_rec = GroupRecord::new(
            "east-coast-gpus",
            OWNER_DID,
            None,
            "2026-06-23T12:34:56.789Z",
        )
        .unwrap();
        let group_uri = format!("at://{OWNER_DID}/{group_uri_key}");

        // The genuine GroupItemRecord the owner *signed* into their MST.
        let genuine_item =
            GroupItemRecord::new(group_uri.clone(), NODE_DID, "2026-06-23T12:34:56.789Z").unwrap();
        let genuine_item_bytes = genuine_item.to_dag_cbor();
        // A *different* valid GroupItemRecord (forged bytes) the attacker swaps
        // in under the genuine one's CID. It still claims NODE_DID, so without
        // the CID check it would decode and produce membership — proving the
        // catch is the content-bind, not a decode failure.
        let forged_item_bytes =
            GroupItemRecord::new(group_uri.clone(), NODE_DID, "2026-07-09T00:00:00.000Z")
                .unwrap()
                .to_dag_cbor();
        assert_ne!(
            genuine_item_bytes, forged_item_bytes,
            "forged block must differ from the genuine one"
        );

        let (owner_car, owner_vk) = repo_car(
            OWNER_DID,
            &[
                (group_uri_key.clone(), group_rec.to_dag_cbor()),
                (group_item_record_key("3i"), genuine_item_bytes.clone()),
            ],
        );
        // The CID under which the genuine item lives in the signed MST.
        let item_cid = Cid::from_dag_cbor(&genuine_item_bytes);

        // Replay the owner's genuine signed commit, but swap the item block's
        // bytes for the forged record while keeping the declared CID unchanged.
        let (roots, blocks) = parse_car_v1(&owner_car).unwrap();
        let tampered_blocks: Vec<(Cid, Vec<u8>)> = blocks
            .into_iter()
            .map(|(cid, bytes)| {
                if cid == item_cid {
                    (cid, forged_item_bytes.clone())
                } else {
                    (cid, bytes)
                }
            })
            .collect();
        let tampered_owner_car = build_car_v1(&roots, &tampered_blocks);

        // Node consents (so membership WOULD hold if the forged claim ingested).
        let node_rec = sample_node_record(vec![group_uri.clone()]);
        let (node_car, node_vk) =
            repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);

        let resolver = FixedResolver {
            repos: HashMap::from([
                (OWNER_DID.to_owned(), (tampered_owner_car, owner_vk)),
                (NODE_DID.to_owned(), (node_car, node_vk)),
            ]),
        };
        let index = PlacementIndex::new();
        // The owner ingest MUST fail closed — the commit signature verifies, but
        // the substituted block's recomputed CID ≠ its declared CID.
        assert!(
            index.ingest_did(&resolver, OWNER_DID).await.is_err(),
            "a repo whose block bytes don't hash to their declared CID must be refused, \
             even with a valid commit signature"
        );
        index.ingest_did(&resolver, NODE_DID).await.unwrap();

        assert!(
            !index.is_member(NODE_DID, &group_uri),
            "a block substituted under a genuine signed commit must not forge membership"
        );
        assert!(!index
            .effective_labels(NODE_DID)
            .iter()
            .any(|(k, _)| k == &format!("group/{group_uri}")));
    }

    #[tokio::test]
    async fn missing_verifying_key_preserves_trusted_resolver_ingest() {
        // #932 — the `Ok(None)` fallback: a resolver that cannot provide a key
        // retains the day-1 trusted-resolver posture (records still ingest).
        // This is the seam full DID-document key resolution for foreign DIDs
        // will close later; until then it must not regress trusted ingest.
        let rec = sample_node_record(vec![]);
        let (car, _) = repo_car(NODE_DID, &[(node_record_key("3a"), rec.to_dag_cbor())]);

        // A resolver that declines to provide any verifying key.
        struct NoKeyResolver(HashMap<String, Vec<u8>>);
        #[async_trait(?Send)]
        impl RecordResolver for NoKeyResolver {
            async fn resolve_record(
                &self,
                _did: &str,
                _collection: &str,
                _rkey: &str,
            ) -> Result<Option<RecordCarData>> {
                Ok(None)
            }
            async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>> {
                Ok(self.0.get(did).map(|car| RecordCarData {
                    uri: format!("at://{did}"),
                    car: car.clone(),
                }))
            }
            // resolve_verifying_key defaults to Ok(None).
        }

        let resolver = NoKeyResolver(HashMap::from([(NODE_DID.to_owned(), car)]));
        let index = PlacementIndex::new();
        index.ingest_did(&resolver, NODE_DID).await.unwrap();
        assert_eq!(index.known_node_dids(), vec![NODE_DID.to_owned()]);
    }

    #[tokio::test]
    async fn did_confused_repo_is_rejected_and_clears_stale_state() {
        let rec = sample_node_record(vec![]);
        let valid = repo_car(NODE_DID, &[(node_record_key("3a"), rec.to_dag_cbor())]);
        let index = PlacementIndex::new();
        index
            .ingest_did(
                &FixedResolver {
                    repos: HashMap::from([(NODE_DID.to_owned(), valid)]),
                },
                NODE_DID,
            )
            .await
            .unwrap();

        let owner_repo = repo_car(
            OWNER_DID,
            &[(
                node_record_key("3a"),
                sample_node_record(vec![]).to_dag_cbor(),
            )],
        );
        let confused = FixedResolver {
            repos: HashMap::from([(NODE_DID.to_owned(), owner_repo)]),
        };
        assert!(index.ingest_did(&confused, NODE_DID).await.is_err());
        assert!(index.known_node_dids().is_empty());
    }

    #[tokio::test]
    async fn decode_failure_without_key_clears_stale_state() {
        struct NoKeyResolver(HashMap<String, Vec<u8>>);
        #[async_trait(?Send)]
        impl RecordResolver for NoKeyResolver {
            async fn resolve_record(
                &self,
                _did: &str,
                _collection: &str,
                _rkey: &str,
            ) -> Result<Option<RecordCarData>> {
                Ok(None)
            }
            async fn resolve_repo(&self, did: &str) -> Result<Option<RecordCarData>> {
                Ok(self.0.get(did).map(|car| RecordCarData {
                    uri: format!("at://{did}"),
                    car: car.clone(),
                }))
            }
        }

        let (car, _) = repo_car(
            NODE_DID,
            &[(
                node_record_key("3a"),
                sample_node_record(vec![]).to_dag_cbor(),
            )],
        );
        let index = PlacementIndex::new();
        index
            .ingest_did(
                &NoKeyResolver(HashMap::from([(NODE_DID.to_owned(), car)])),
                NODE_DID,
            )
            .await
            .unwrap();
        assert!(!index.known_node_dids().is_empty());

        let malformed = NoKeyResolver(HashMap::from([(NODE_DID.to_owned(), vec![0xff])]));
        assert!(index.ingest_did(&malformed, NODE_DID).await.is_err());
        assert!(index.known_node_dids().is_empty());
    }

    #[test]
    fn car_structure_bypasses_are_rejected() {
        let (car, vk) = repo_car(
            NODE_DID,
            &[(
                node_record_key("3a"),
                sample_node_record(vec![]).to_dag_cbor(),
            )],
        );
        let (roots, blocks) = parse_car_v1(&car).unwrap();

        let duplicate = build_car_v1(&roots, &[blocks.clone(), blocks.clone()].concat());
        assert!(PlacementIndex::decode_repo_car(NODE_DID, &duplicate, Some(&vk)).is_err());

        let multiple_roots = build_car_v1(&[roots[0], roots[0]], &blocks);
        assert!(PlacementIndex::decode_repo_car(NODE_DID, &multiple_roots, Some(&vk)).is_err());

        let missing_record = build_car_v1(&roots, &blocks[..blocks.len() - 1]);
        assert!(PlacementIndex::decode_repo_car(NODE_DID, &missing_record, Some(&vk)).is_err());
    }

    #[test]
    fn shared_mst_node_is_rejected_traversal_wide() {
        use hyprstream_pds::mst::TreeEntry;

        let leaf_bytes = NodeData { l: None, e: vec![] }.encode();
        let leaf_cid = Cid::from_dag_cbor(&leaf_bytes);
        let parent = NodeData {
            l: Some(leaf_cid),
            e: vec![TreeEntry {
                p: 0,
                k: b"x/y".to_vec(),
                v: Cid::from_dag_cbor(b"record"),
                t: Some(leaf_cid),
            }],
        };
        let parent_bytes = parent.encode();
        let parent_cid = Cid::from_dag_cbor(&parent_bytes);
        let blocks = HashMap::from([(leaf_cid, leaf_bytes), (parent_cid, parent_bytes)]);
        let mut out = Vec::new();

        assert!(walk_mst(parent_cid, &blocks, &mut out, NODE_DID).is_err());
    }
}
