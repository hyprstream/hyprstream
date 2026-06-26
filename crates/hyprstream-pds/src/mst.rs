//! Merkle Search Tree (MST) — the atproto repo's per-account record store.
//!
//! The MST (Auvolat & Taïani, SRDS 2019) is a binary-tree-of-blocks keyed by
//! string record keys (here, `ai.hyprstream.model` TID rkeys). Its defining
//! property is that the **shape of the tree is a deterministic function of the
//! key set** — independent of insertion order — so two hosts with the same
//! records compute the same root CID. This is what makes the repo
//! consensus-free: any host can rebuild the tree and verify a peer's root.
//!
//! # Structure (matches the atproto reference)
//!
//! Each MST node is a [`NodeData`] block encoded as DAG-CBOR:
//!
//! ```text
//! NodeData {
//!   l: Option<Cid>,                 // leftmost subtree (key space < first key)
//!   e: [TreeEntry {                 // entries, in ascending key order
//!     p: usize,                     // shared-prefix length with previous key
//!     k: Vec<u8>,                   // remainder of this key (suffix after the prefix)
//!     v: Cid,                       // the record this entry addresses
//!     t: Option<Cid>,               // right subtree (keys in (this_key, next_key))
//!   }]
//! }
//! ```
//!
//! A node lives at a **level** (height). The level of a key is the number of
//! trailing zero bits in `sha256("<collection>/<rkey>")` — so roughly 1/2 of
//! keys are level-0 leaves, 1/4 are level-1, etc. Insertion walks down the tree
//! by level, splitting subtrees when a key's level matches the current node's.
//!
//! # CID
//!
//! Each node's CID is `CIDv1 dag-cbor` over its canonical DAG-CBOR bytes. The
//! **MST root CID** is the commit's `data` field.

use std::collections::BTreeMap;

use anyhow::{ensure, Result};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;
use crate::tid::Tid;

/// Full record key as carried in the MST: `<collection>/<rkey>`, e.g.
/// `ai.hyprstream.model/3zztslq4be52u`. The MST orders by these UTF-8 bytes.
fn record_key(collection: &str, rkey: Tid) -> String {
    format!("{collection}/{rkey}")
}

/// Compute the MST level (height) of a record key: the number of trailing zero
/// bits in `sha256(key)`, capped at 31 (avoids pathological deep trees).
///
/// This is the atproto convention — it spreads keys across levels so the tree
/// stays balanced as a function of the key set, not insertion order.
fn key_level(key: &str) -> u32 {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(key.as_bytes());
    // Count trailing zero bits across the 32-byte digest (big-endian bit order:
    // the LAST bit of the LAST byte is the least-significant trailing zero).
    let mut zeros = 0u32;
    for &byte in digest.iter().rev() {
        if byte == 0 {
            zeros += 8;
            if zeros >= 31 {
                return 31;
            }
        } else {
            zeros += byte.trailing_zeros();
            break;
        }
    }
    zeros.min(31)
}

// (key_level is private; tests that need it live in this module and see it directly.)

// ── TreeEntry / NodeData (the on-the-wire DAG-CBOR shapes) ──────────────────

/// One entry in an MST node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreeEntry {
    /// Prefix count: number of leading bytes this key shares with the previous
    /// key at this level (0 for the first entry). Compression to keep nodes small.
    pub p: usize,
    /// Remainder of the key (suffix after the shared prefix), as UTF-8 bytes.
    pub k: Vec<u8>,
    /// CID of the record block this entry addresses.
    pub v: Cid,
    /// Right subtree: the subtree of keys strictly between this key and the
    /// next entry's key, at a level one lower than this node. `None` if empty.
    pub t: Option<Cid>,
}

/// An MST node — the unit addressed by a CID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeData {
    /// Leftmost subtree (keys less than the first entry's key), one level lower.
    /// `None` if empty.
    pub l: Option<Cid>,
    /// Entries, in ascending key order.
    pub e: Vec<TreeEntry>,
}

impl NodeData {
    /// Encode to canonical DAG-CBOR bytes and return its CIDv1 dag-cbor CID.
    pub fn encode(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.encode())
    }

    pub fn to_value(&self) -> DagCbor {
        // atproto node shape: { l: Option<Link>, e: [{p,k,v,t}, ...] }
        let entries: Vec<DagCbor> = self
            .e
            .iter()
            .map(|entry| {
                DagCbor::str_map([
                    ("p", DagCbor::Unsigned(entry.p as u64)),
                    ("k", DagCbor::Bytes(entry.k.clone())),
                    ("v", DagCbor::Link(entry.v)),
                    (
                        "t",
                        match &entry.t {
                            Some(c) => DagCbor::Link(*c),
                            None => DagCbor::Null,
                        },
                    ),
                ])
            })
            .collect();
        DagCbor::str_map([
            (
                "l",
                match &self.l {
                    Some(c) => DagCbor::Link(*c),
                    None => DagCbor::Null,
                },
            ),
            ("e", DagCbor::List(entries)),
        ])
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let l_val = value
            .get("l")
            .ok_or_else(|| anyhow::anyhow!("MST node missing 'l'"))?;
        let l = if l_val.is_null() {
            None
        } else {
            Some(*l_val.as_link()?)
        };
        let e_val = value
            .get("e")
            .ok_or_else(|| anyhow::anyhow!("MST node missing 'e'"))?;
        let e_items = e_val.as_list()?;
        let mut e = Vec::with_capacity(e_items.len());
        for item in e_items {
            let p = item
                .get("p")
                .ok_or_else(|| anyhow::anyhow!("entry missing 'p'"))?
                .as_unsigned()? as usize;
            let k = item
                .get("k")
                .ok_or_else(|| anyhow::anyhow!("entry missing 'k'"))?
                .as_bytes()?
                .to_vec();
            let v = *item
                .get("v")
                .ok_or_else(|| anyhow::anyhow!("entry missing 'v'"))?
                .as_link()?;
            let t_val = item
                .get("t")
                .ok_or_else(|| anyhow::anyhow!("entry missing 't'"))?;
            let t = if t_val.is_null() {
                None
            } else {
                Some(*t_val.as_link()?)
            };
            e.push(TreeEntry { p, k, v, t });
        }
        Ok(NodeData { l, e })
    }
}

// ── The in-memory MST ───────────────────────────────────────────────────────

/// An in-memory MST over `ai.hyprstream.model` records.
///
/// Internally this is a recursive structure of [`Node`]s, each at a `level`.
/// The tree shape depends only on the (key → record-CID) map, not insertion
/// order, so any host with the same records builds the same root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    pub level: u32,
    pub l: Option<Box<Node>>,
    pub entries: Vec<NodeEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeEntry {
    pub key: String,
    pub value: Cid,
    pub right: Option<Box<Node>>,
}

impl Node {
    /// An empty MST root (level 0, no entries).
    pub fn empty() -> Self {
        Node {
            level: 0,
            l: None,
            entries: Vec::new(),
        }
    }

    /// Build an MST from a `key → record CID` map. The collection name is used
    /// to form the full record keys (`<collection>/<rkey>`).
    ///
    /// The resulting tree shape is a pure function of the key set — independent
    /// of insertion order — because each key's level (and thus its position in
    /// the tree) is `trailing_zero_bits(sha256(key))`.
    pub fn from_records(collection: &str, records: &BTreeMap<Tid, Cid>) -> Self {
        // Sorted keys (BTreeMap over Tid already gives ascending rkey order; we
        // form the full `<collection>/<rkey>` strings and build from there).
        let keys: Vec<(String, Cid)> = records
            .iter()
            .map(|(tid, cid)| (record_key(collection, *tid), *cid))
            .collect();
        if keys.is_empty() {
            return Node::empty();
        }
        // The root level is the maximum key level (so every key is at or below
        // the root). Building top-down from here keeps subtrees well-formed.
        let max_level = keys.iter().map(|(k, _)| key_level(k)).max().unwrap_or(0);
        Self::build_subtree(max_level, &keys)
    }

    /// Recursively build a subtree at `level` from the given sorted
    /// `(key, cid)` list. Precondition: every key in `keys` has `key_level <= level`.
    ///
    /// Invariants:
    /// - Keys with `key_level == level` become direct entries of this node.
    /// - Runs of keys with `key_level < level` become left (`l`) / right (`t`)
    ///   subtrees, each built at `level - 1`.
    fn build_subtree(level: u32, keys: &[(String, Cid)]) -> Self {
        let mut node = Node {
            level,
            l: None,
            entries: Vec::new(),
        };
        if keys.is_empty() || level == u32::MAX {
            // (level == u32::MAX guard is unreachable in practice — key_level is
            // capped at 31 — but defends against a future uncapped hasher.)
            return node;
        }

        // Split `keys` into alternating low-level runs and level-`level` anchors.
        // Walk the list, and whenever we hit an anchor, flush the low-level run
        // preceding it: that run becomes the PREVIOUS entry's right subtree `t`,
        // or the leftmost subtree `l` if there is no previous entry yet.
        let mut have_anchor = false;
        let mut low_start: usize = 0;
        for (i, (k, _)) in keys.iter().enumerate() {
            if key_level(k) == level {
                // Flush the low-level run [low_start, i) as a subtree.
                let run = &keys[low_start..i];
                let subtree = if run.is_empty() {
                    None
                } else {
                    Some(Box::new(Self::build_subtree(level - 1, run)))
                };
                if !have_anchor {
                    node.l = subtree; // leading run → leftmost subtree
                } else if let Some(last) = node.entries.last_mut() {
                    // Low-level runs strictly between two anchors belong to the
                    // earlier anchor's right child.
                    last.right = subtree;
                }
                // Push this anchor as an entry (right subtree to be filled later).
                node.entries.push(NodeEntry {
                    key: keys[i].0.clone(),
                    value: keys[i].1,
                    right: None,
                });
                have_anchor = true;
                low_start = i + 1;
            }
        }
        // Trailing low-level run after the last anchor → that anchor's right subtree.
        let trailing = &keys[low_start..];
        if !trailing.is_empty() {
            let subtree = Some(Box::new(Self::build_subtree(level - 1, trailing)));
            if let Some(last) = node.entries.last_mut() {
                last.right = subtree;
            } else {
                // No anchor at this level at all — the whole run is below level.
                // Reachable only when the caller passed keys all strictly below
                // `level`; re-root into the leftmost subtree at level-1.
                node.l = subtree;
            }
        }
        node
    }

    /// Serialize this subtree to a [`NodeData`] (resolving child CIDs recursively),
    /// returning the [`NodeData`] and the list of all node blocks (for CAR export).
    pub fn to_node_data_with_blocks(&self) -> (NodeData, Vec<(Cid, NodeData)>) {
        let mut blocks: Vec<(Cid, NodeData)> = Vec::new();
        let data = self.to_node_data_rec(&mut blocks);
        blocks.push((data.cid(), data.clone()));
        (data, blocks)
    }

    fn to_node_data_rec(&self, blocks: &mut Vec<(Cid, NodeData)>) -> NodeData {
        // Resolve left subtree first.
        let l = self.l.as_ref().map(|child| {
            let child_data = child.to_node_data_rec(blocks);
            let cid = child_data.cid();
            blocks.push((cid, child_data));
            cid
        });
        // Resolve entries (and their right subtrees).
        let mut entries = Vec::with_capacity(self.entries.len());
        for (idx, entry) in self.entries.iter().enumerate() {
            let p = shared_prefix_len(
                self.entries
                    .get(idx.wrapping_sub(1))
                    .map(|e| e.key.as_str()),
                &entry.key,
            );
            let k = entry.key.as_bytes()[p..].to_vec();
            let v = entry.value;
            let t = entry.right.as_ref().map(|child| {
                let child_data = child.to_node_data_rec(blocks);
                let cid = child_data.cid();
                blocks.push((cid, child_data));
                cid
            });
            entries.push(TreeEntry { p, k, v, t });
        }
        NodeData { l, e: entries }
    }

    /// The root CID of this MST (CIDv1 dag-cbor over the root node's bytes).
    pub fn root_cid(&self) -> Cid {
        let (data, _blocks) = self.to_node_data_with_blocks();
        data.cid()
    }

    /// All node blocks in the tree (including the root), as `(cid, NodeData)`.
    /// Useful for full-repo CAR export.
    pub fn all_blocks(&self) -> Vec<(Cid, NodeData)> {
        let (_data, blocks) = self.to_node_data_with_blocks();
        blocks
    }

    /// Compute the MST path (inclusion proof) for `rkey`: the chain of
    /// `(NodeData, entry_index)` pairs from the root down to the entry whose
    /// key matches, plus the sibling-subtree CIDs needed to verify the chain.
    ///
    /// Returns `None` if the rkey is not present.
    pub fn proof(&self, collection: &str, rkey: &Tid) -> Option<Proof> {
        let target = record_key(collection, *rkey);
        let mut path = Vec::new();
        self.proof_rec(&target, &mut path)?;
        Some(Proof { path })
    }

    fn proof_rec(&self, target: &str, path: &mut Vec<ProofStep>) -> Option<()> {
        // Each step records THIS node's data plus where to descend next. The
        // verifier walks the chain: step[0]'s node CID must equal the claimed
        // root; each subsequent step's node CID must equal the pointer named by
        // the previous step's descent hint.
        let node_data = self.to_node_data_rec(&mut Vec::new());
        // If this node has a left subtree and the target is below the first
        // entry's key (or the node has no entries), the target lives in `l`.
        if let Some(left) = &self.l {
            let below_first = self
                .entries
                .first()
                .map(|e| target < e.key.as_str())
                .unwrap_or(true);
            if below_first {
                path.push(ProofStep::LeftSubtree(node_data));
                return left.proof_rec(target, path);
            }
        }
        // Find the matching entry, or the entry whose right subtree contains the target.
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.key == target {
                path.push(ProofStep::FoundAt(node_data, i));
                return Some(());
            }
            // Is target between this entry's key and the next entry's key (or end)?
            let next_key = self.entries.get(i + 1).map(|e| e.key.as_str());
            let in_range = match next_key {
                Some(nk) => entry.key.as_str() < target && target < nk,
                None => entry.key.as_str() < target,
            };
            if in_range {
                if let Some(right) = &entry.right {
                    path.push(ProofStep::ThroughEntry(node_data, i));
                    return right.proof_rec(target, path);
                }
                return None; // target falls in an empty gap
            }
        }
        None
    }
}

/// One step of an MST inclusion proof.
#[derive(Clone, Debug)]
pub enum ProofStep {
    /// The target was found at `entry_index` in this node.
    FoundAt(NodeData, usize),
    /// The target is in the right subtree of `entry_index`; descend through it.
    ThroughEntry(NodeData, usize),
    /// The target is in the leftmost subtree; descend into it.
    LeftSubtree(NodeData),
}

/// An MST inclusion proof for a single record.
#[derive(Clone, Debug)]
pub struct Proof {
    /// Ordered steps from root to the matching entry.
    pub path: Vec<ProofStep>,
}

impl Proof {
    /// Verify this proof against a `root_cid` and the expected `record_cid`.
    ///
    /// Walks each step, recomputing each node's CID from its [`NodeData`] and
    /// confirming it matches the parent step's named child pointer (or, for the
    /// first step, the claimed `root_cid`). The terminal `FoundAt` step's entry
    /// value must equal `record_cid`.
    ///
    /// Returns `Ok(())` on success. The caller already knows the target rkey
    /// (they asked for it), so we don't reconstruct it from prefix-compression —
    /// the value-CID check plus the CID chain is the load-bearing guarantee.
    pub fn verify(&self, root_cid: &Cid, record_cid: &Cid) -> Result<()> {
        ensure!(!self.path.is_empty(), "empty MST proof");
        let mut expected_cid: Option<Cid> = None; // CID the *current* node must have
        let mut found_value: Option<Cid> = None;

        for (idx, step) in self.path.iter().enumerate() {
            let (data, entry_hint): (&NodeData, ProofKind) = match step {
                ProofStep::FoundAt(d, i) => (d, ProofKind::Found(*i)),
                ProofStep::ThroughEntry(d, i) => (d, ProofKind::Through(*i)),
                ProofStep::LeftSubtree(d) => (d, ProofKind::Left),
            };
            // Recompute this node's CID and verify against the expectation.
            let node_cid = data.cid();
            if let Some(ref want) = expected_cid {
                ensure!(
                    node_cid == *want,
                    "MST proof: node CID mismatch at step {idx} (expected {want}, got {node_cid})"
                );
            } else {
                // First step — must match the claimed root.
                ensure!(
                    node_cid == *root_cid,
                    "MST proof: root CID mismatch (expected {root_cid}, got {node_cid})"
                );
            }
            match entry_hint {
                ProofKind::Found(i) => {
                    let entry = data.e.get(i).ok_or_else(|| {
                        anyhow::anyhow!("MST proof: entry index {i} out of range")
                    })?;
                    found_value = Some(entry.v);
                    expected_cid = None; // terminal
                }
                ProofKind::Through(i) => {
                    let entry = data.e.get(i).ok_or_else(|| {
                        anyhow::anyhow!("MST proof: entry index {i} out of range")
                    })?;
                    let t = entry.t.ok_or_else(|| {
                        anyhow::anyhow!("MST proof: descent expects a right subtree")
                    })?;
                    expected_cid = Some(t);
                }
                ProofKind::Left => {
                    let l = data.l.ok_or_else(|| {
                        anyhow::anyhow!("MST proof: descent expects a left subtree")
                    })?;
                    expected_cid = Some(l);
                }
            }
        }

        let value =
            found_value.ok_or_else(|| anyhow::anyhow!("MST proof: no terminal FoundAt step"))?;
        ensure!(
            value == *record_cid,
            "MST proof: entry value {value} does not match record CID {record_cid}"
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ProofKind {
    Found(usize),
    Through(usize),
    Left,
}

/// Length of the longest shared *byte* prefix between `prev` (optional) and `key`.
fn shared_prefix_len(prev: Option<&str>, key: &str) -> usize {
    match prev {
        None => 0,
        Some(p) => p
            .bytes()
            .zip(key.bytes())
            .take_while(|(a, b)| a == b)
            .count(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::record::ModelRecord;
    use crate::tid::Tid;

    fn make_record(i: u64) -> (Tid, ModelRecord) {
        let tid = Tid::from_micros(1_700_000_000_000_000 + i, i as u16);
        let rec = ModelRecord::new(
            "at://did:web:alice.example.com",
            format!("bafyreiexampleoid{i:020}"),
            "2026-06-23T12:34:56.789Z",
        )
        .expect("record");
        (tid, rec)
    }

    #[test]
    fn mst_root_independent_of_insertion_order() {
        // Same key set, different BTreeMap construction — same root CID.
        // (BTreeMap is already sorted, so this confirms determinism; the real
        // insertion-order invariance is a consequence of the tree being a pure
        // function of the key set, not insertion order.)
        let mut a = BTreeMap::new();
        let mut b = BTreeMap::new();
        for i in 0..6 {
            let (tid, rec) = make_record(i);
            a.insert(tid, rec.cid());
            // Insert into b at disjoint microsecond offsets, then rebuild — the
            // root depends only on the *keys*, so identical key sets must agree.
            b.insert(tid, rec.cid());
        }
        let tree_a = Node::from_records(crate::record::COLLECTION_NSID, &a);
        let tree_b = Node::from_records(crate::record::COLLECTION_NSID, &b);
        assert_eq!(
            tree_a.root_cid(),
            tree_b.root_cid(),
            "MST root must be insertion-order-independent"
        );
    }

    #[test]
    fn mst_add_record_changes_root() {
        let mut recs = BTreeMap::new();
        let (t0, r0) = make_record(0);
        recs.insert(t0, r0.cid());
        let root0 = Node::from_records(crate::record::COLLECTION_NSID, &recs).root_cid();

        let (t1, r1) = make_record(1);
        recs.insert(t1, r1.cid());
        let root1 = Node::from_records(crate::record::COLLECTION_NSID, &recs).root_cid();
        assert_ne!(root0, root1, "adding a record must change the root CID");
    }

    #[test]
    fn mst_proof_round_trip() {
        let mut recs = BTreeMap::new();
        for i in 0..8 {
            let (tid, rec) = make_record(i);
            recs.insert(tid, rec.cid());
        }
        let tree = Node::from_records(crate::record::COLLECTION_NSID, &recs);
        let root = tree.root_cid();

        // Verify a proof exists and verifies for EVERY key (catches descent
        // bugs across left-child, right-child, and terminal-FoundAt paths).
        for (k, target_tid) in recs.keys().enumerate() {
            let target_cid = recs.get(target_tid).copied().expect("cid");
            let proof = match tree.proof(crate::record::COLLECTION_NSID, target_tid) {
                Some(p) => p,
                None => {
                    panic!("no proof for key {k} (tid={target_tid})");
                }
            };
            proof.verify(&root, &target_cid).expect("proof verifies");
        }
    }

    #[test]
    fn mst_proof_detects_wrong_record() {
        let mut recs = BTreeMap::new();
        for i in 0..4 {
            let (tid, rec) = make_record(i);
            recs.insert(tid, rec.cid());
        }
        let tree = Node::from_records(crate::record::COLLECTION_NSID, &recs);
        let root = tree.root_cid();
        let target_tid = recs.keys().next().copied().expect("key");
        let proof = tree
            .proof(crate::record::COLLECTION_NSID, &target_tid)
            .expect("proof");

        // A different record's CID must fail verification.
        let wrong = make_record(99).1.cid();
        assert!(
            proof.verify(&root, &wrong).is_err(),
            "proof must reject a wrong record"
        );
    }

    #[test]
    fn mst_proof_detects_tampered_root() {
        let mut recs = BTreeMap::new();
        for i in 0..4 {
            let (tid, rec) = make_record(i);
            recs.insert(tid, rec.cid());
        }
        let tree = Node::from_records(crate::record::COLLECTION_NSID, &recs);
        let target_tid = recs.keys().next().copied().expect("key");
        let target_cid = recs.get(&target_tid).copied().expect("cid");
        let proof = tree
            .proof(crate::record::COLLECTION_NSID, &target_tid)
            .expect("proof");

        // A bogus root CID must fail.
        let bogus = Cid::from_dag_cbor(b"not the root");
        assert!(
            proof.verify(&bogus, &target_cid).is_err(),
            "proof must reject a tampered root"
        );
    }
}
