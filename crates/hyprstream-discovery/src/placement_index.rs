//! P1 placement directory — record ingestion + in-process index (#524).
//!
//! Day-1 ingestion is **polled `RecordResolver` reads**, not a firehose/moq_event
//! tail (that's explicitly out of scope — a day-2 ticket). Given a node DID, we
//! call [`crate::RecordResolver::resolve_repo`], parse the returned CARv1 blob
//! (`hyprstream_pds::car::parse_car_v1`), walk the repo's MST from the commit's
//! `data` root, and decode every `ai.hyprstream.placement.{node,workload,group,
//! groupItem}` record we find into an in-process index.
//!
//! # Trust posture
//!
//! Unlike `getRecord`/`getRepo` (D5: untrusted relay, caller verifies the CAR
//! proof offline), this ingestion path does **not** verify the commit
//! signature. The injected `RecordResolver` is the same trusted, in-process
//! resolver `handle_get_repo` already serves *from* (see its trait docs: "the
//! node's local PDS") — there is no untrusted network hop between the index and
//! its source here. Verifying signatures would need a DID→verifying-key
//! resolution step this crate doesn't have; that's future federation-hardening
//! work, not a P1 blocker (the index is best-effort scheduling metadata, not a
//! security boundary — the *per-candidate authz check* in `queryCandidates` is
//! the security boundary).
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

use std::collections::HashMap;

use anyhow::{anyhow, Result};
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
    pub async fn ingest_did(&self, resolver: &dyn RecordResolver, did: &str) -> Result<()> {
        let repo = resolver
            .resolve_repo(did)
            .await
            .map_err(|e| anyhow!("resolve_repo({did}) failed: {e}"))?;
        let Some(repo) = repo else {
            // No repo stored for this DID — clear any stale contribution and stop.
            self.raw.write().remove(did);
            self.recompute_derived();
            return Ok(());
        };
        let snapshot = Self::decode_repo_car(did, &repo.car)?;
        self.raw.write().insert(did.to_owned(), snapshot);
        self.recompute_derived();
        Ok(())
    }

    /// Parse a repo CARv1 blob and decode every `ai.hyprstream.placement.*`
    /// record reachable from the commit's MST root into a [`DidSnapshot`].
    fn decode_repo_car(did: &str, car: &[u8]) -> Result<DidSnapshot> {
        let (roots, blocks) = parse_car_v1(car)?;
        let root_cid = *roots
            .first()
            .ok_or_else(|| anyhow!("repo CAR for {did} has no root"))?;
        let block_map: HashMap<Cid, Vec<u8>> = blocks.into_iter().collect();
        let commit_bytes = block_map
            .get(&root_cid)
            .ok_or_else(|| anyhow!("repo CAR for {did} missing its commit block"))?;
        let commit = Commit::from_dag_cbor(commit_bytes)?;

        let mut entries = Vec::new();
        walk_mst(commit.data, &block_map, &mut entries)?;

        let mut snapshot = DidSnapshot::default();
        for (key, record_cid) in entries {
            let Some((collection, _rkey)) = key.split_once('/') else {
                continue; // malformed key (no collection prefix) — skip, not fatal
            };
            let Some(bytes) = block_map.get(&record_cid) else {
                continue; // entry points at a block the CAR didn't include
            };

            if collection == node::COLLECTION_NSID {
                if let Ok(rec) = NodeRecord::from_dag_cbor(bytes) {
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
                }
            } else if collection == group::COLLECTION_NSID {
                if let Ok(rec) = GroupRecord::from_dag_cbor(bytes) {
                    let uri = format!("at://{did}/{key}");
                    snapshot.groups.push((uri, rec.name, rec.owner_did));
                }
            } else if collection == group_item::COLLECTION_NSID {
                if let Ok(rec) = GroupItemRecord::from_dag_cbor(bytes) {
                    snapshot.group_items.push(rec);
                }
            } else if collection == workload::COLLECTION_NSID {
                // Decoded for completeness per #524 P1 scope; workload placement
                // decisions aren't consumed by queryCandidates (day-1 scope).
                if WorkloadRecord::from_dag_cbor(bytes).is_ok() {
                    snapshot.workload_count += 1;
                }
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
                claims.entry(item.group.clone()).or_default().push(item.subject.clone());
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

/// Enumerate every `(full_key, record_cid)` entry reachable from an MST node,
/// depth-first in key order: left subtree, then each entry (recursing into its
/// right subtree before the next entry). Mirrors the encode side
/// (`Node::to_node_data_rec` in `hyprstream_pds::mst`) exactly, including its
/// prefix-compression convention: `entry.p` compresses against the *previous
/// entry within this same node* only (resets to 0 at the first entry of every
/// node) — never against a key from a different node.
fn walk_mst(node_cid: Cid, blocks: &HashMap<Cid, Vec<u8>>, out: &mut Vec<(String, Cid)>) -> Result<()> {
    let bytes = blocks
        .get(&node_cid)
        .ok_or_else(|| anyhow!("MST node {node_cid} missing from repo CAR blocks"))?;
    let value = DagCbor::decode(bytes)?;
    let data = NodeData::from_value(&value)?;

    if let Some(left) = data.l {
        walk_mst(left, blocks, out)?;
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
            walk_mst(right, blocks, out)?;
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
    use hyprstream_pds::car::build_car_v1;
    use hyprstream_pds::commit::UnsignedCommit;
    use hyprstream_pds::mst::Node;
    use hyprstream_pds::tid::Tid;
    use p256::ecdsa::SigningKey as P256SigningKey;

    use crate::service::RecordCarData;

    const NODE_DID: &str = "did:web:node1.example.com";
    const OWNER_DID: &str = "did:web:owner.example.com";

    /// Build a repo CARv1 for `did` from a set of already-encoded `(full_key,
    /// dag_cbor_bytes)` records (mixed collections OK — the repo MST spans all
    /// of them, matching real atproto repos).
    fn repo_car(did: &str, records: &[(String, Vec<u8>)]) -> Vec<u8> {
        let signing_key = P256SigningKey::random(&mut rand::rngs::OsRng);
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
        build_car_v1(&[commit.cid()], &blocks)
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

    /// A resolver serving fixed, per-DID repo CARs (built with real records).
    struct FixedResolver {
        repos: HashMap<String, Vec<u8>>,
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
            Ok(self.repos.get(did).map(|car| RecordCarData {
                uri: format!("at://{did}"),
                car: car.clone(),
            }))
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
        let resolver = FixedResolver { repos: HashMap::new() };
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
                        GroupItemRecord::new(group_uri.clone(), NODE_DID, "2026-06-23T12:34:56.789Z")
                            .unwrap()
                            .to_dag_cbor(),
                    ),
                ],
            );
            let node_rec = sample_node_record(vec![]); // no consent
            let node_car = repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);
            let resolver = FixedResolver {
                repos: HashMap::from([(OWNER_DID.to_owned(), owner_car), (NODE_DID.to_owned(), node_car)]),
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
            let owner_car = repo_car(OWNER_DID, &[(group_uri_key.clone(), group_rec.to_dag_cbor())]);
            let node_rec = sample_node_record(vec![group_uri.clone()]);
            let node_car = repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);
            let resolver = FixedResolver {
                repos: HashMap::from([(OWNER_DID.to_owned(), owner_car), (NODE_DID.to_owned(), node_car)]),
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
                        GroupItemRecord::new(group_uri.clone(), NODE_DID, "2026-06-23T12:34:56.789Z")
                            .unwrap()
                            .to_dag_cbor(),
                    ),
                ],
            );
            let node_rec = sample_node_record(vec![group_uri.clone()]);
            let node_car = repo_car(NODE_DID, &[(node_record_key("3a"), node_rec.to_dag_cbor())]);
            let resolver = FixedResolver {
                repos: HashMap::from([(OWNER_DID.to_owned(), owner_car), (NODE_DID.to_owned(), node_car)]),
            };
            let index = PlacementIndex::new();
            index.ingest_did(&resolver, OWNER_DID).await.unwrap();
            index.ingest_did(&resolver, NODE_DID).await.unwrap();
            assert!(index.is_member(NODE_DID, &group_uri), "both claims present -> member");
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
        let resolver_b = FixedResolver { repos: HashMap::new() };
        index.ingest_did(&resolver_b, NODE_DID).await.unwrap();
        assert!(index.known_node_dids().is_empty(), "stale contribution must be cleared on re-poll");
    }
}
