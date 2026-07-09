//! `ai.hyprstream.ledger.checkpoint` — a signed ledger head (the anchorable
//! object).
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.ledger.checkpoint",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["seq", "headHash", "stateRoots"],
//!     "properties": {
//!       "seq":        { "type": "integer", "minimum": 0 },
//!       "headHash":   { "type": "string", "format": "cid" },
//!       "stateRoots": { "type": "array", "items": { "type": "object",
//!                         "required": ["name", "root"], "properties": {
//!                           "name": { "type": "string" },
//!                           "root": { "type": "string", "format": "cid" } } } },
//!       "anchor":     { "type": "string" }
//!     }
//! }}}
//! ```
//!
//! A checkpoint commits the ledger's head at a point in time. It is the object an
//! external timestamp/anchor (e.g. OpenTimestamps) attests to — so it **carries a
//! reference to** the anchor, it never *embeds* the anchor proof:
//!
//! - `seq` — the monotonically-advancing checkpoint sequence number (`u64`).
//! - `headHash` — the CID of the ledger head this checkpoint commits (`format:
//!   "cid"` string).
//! - `stateRoots` — the named state-tree roots at this head ([`StateRoot`]): each
//!   a `{name, root-cid}` pair (e.g. `{"accounts", <cid>}`, `{"receipts", <cid>}`).
//! - `anchor` — *optional* reference to the external anchor for this checkpoint
//!   (an opaque locator: an OTS receipt id, a txid, a CID …). Omitted until the
//!   checkpoint is anchored; the anchor proof itself lives outside this record.
//!   Per #928 rule 8, anchors carry digests only and are submitted on a fixed
//!   cadence — this record holds only the reference, never activity-correlated
//!   payload.

use anyhow::{bail, Result};

use super::{
    map_get_opt_str, map_get_str, map_get_uint, reject_unknown_fields, validate_cid_string,
    validate_token,
};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.ledger.checkpoint";

/// A named state-tree root at the checkpoint's head.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateRoot {
    /// The root's name (e.g. `"accounts"`, `"receipts"`). Non-empty.
    pub name: String,
    /// The root CID. `format: "cid"`.
    pub root: String,
}

/// An `ai.hyprstream.ledger.checkpoint` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointRecord {
    /// Monotonic checkpoint sequence number.
    pub seq: u64,
    /// CID of the ledger head this checkpoint commits. `format: "cid"`.
    pub head_hash: String,
    /// Named state-tree roots at this head.
    pub state_roots: Vec<StateRoot>,
    /// Optional reference to the external anchor (never the proof itself).
    pub anchor: Option<String>,
}

impl CheckpointRecord {
    /// Validate and construct a record.
    pub fn new(
        seq: u64,
        head_hash: impl Into<String>,
        state_roots: Vec<StateRoot>,
        anchor: Option<String>,
    ) -> Result<Self> {
        let head_hash = head_hash.into();
        validate_cid_string(&head_hash, "headHash")?;
        for sr in &state_roots {
            if sr.name.is_empty() {
                bail!("{COLLECTION_NSID}: stateRoots[].name must not be empty");
            }
            validate_cid_string(&sr.root, "stateRoots[].root")?;
        }
        if let Some(a) = &anchor {
            validate_token(a, "anchor")?;
        }
        Ok(CheckpointRecord {
            seq,
            head_hash,
            state_roots,
            anchor,
        })
    }

    /// Encode to canonical DAG-CBOR bytes (deterministic).
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    /// The CID of this record (CIDv1 dag-cbor over its canonical bytes).
    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    /// Build the typed [`DagCbor`] value form. The optional `anchor` field is
    /// omitted from the map when absent.
    pub fn to_value(&self) -> DagCbor {
        let state_roots = DagCbor::list(self.state_roots.iter().map(|sr| {
            DagCbor::str_map([
                ("name", DagCbor::Text(sr.name.clone())),
                ("root", DagCbor::Text(sr.root.clone())),
            ])
        }));
        let mut pairs: Vec<(&str, DagCbor)> = vec![
            ("seq", DagCbor::Unsigned(self.seq)),
            ("headHash", DagCbor::Text(self.head_hash.clone())),
            ("stateRoots", state_roots),
        ];
        if let Some(anchor) = &self.anchor {
            pairs.push(("anchor", DagCbor::Text(anchor.clone())));
        }
        DagCbor::str_map(pairs)
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown_fields(
            value,
            &["seq", "headHash", "stateRoots", "anchor"],
            COLLECTION_NSID,
        )?;
        let seq = map_get_uint(value, "seq", COLLECTION_NSID)?;
        let head_hash = map_get_str(value, "headHash", COLLECTION_NSID)?.to_owned();
        let state_roots = value
            .get("stateRoots")
            .ok_or_else(|| anyhow::anyhow!("{COLLECTION_NSID}: missing required field \"stateRoots\""))?
            .as_list()?
            .iter()
            .map(parse_state_root)
            .collect::<Result<Vec<_>>>()?;
        let anchor = map_get_opt_str(value, "anchor")?.map(str::to_owned);
        Self::new(seq, head_hash, state_roots, anchor)
    }
}

fn parse_state_root(value: &DagCbor) -> Result<StateRoot> {
    reject_unknown_fields(value, &["name", "root"], COLLECTION_NSID)?;
    let name = map_get_str(value, "name", COLLECTION_NSID)?.to_owned();
    let root = map_get_str(value, "root", COLLECTION_NSID)?.to_owned();
    Ok(StateRoot { name, root })
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

    fn roots() -> Vec<StateRoot> {
        vec![
            StateRoot {
                name: "accounts".into(),
                root: "bafyreiaccountsroot1234567890abcdef".into(),
            },
            StateRoot {
                name: "receipts".into(),
                root: "bafyreireceiptsroot1234567890abcdef".into(),
            },
        ]
    }

    fn sample() -> CheckpointRecord {
        CheckpointRecord::new(
            42,
            "bafyreiledgerhead1234567890abcdefghij",
            roots(),
            Some("ots:abcdef0123456789".into()),
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = CheckpointRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        assert_eq!(back.to_dag_cbor(), bytes);
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        assert_eq!(keys, vec!["anchor", "headHash", "seq", "stateRoots"]);
    }

    #[test]
    fn optional_anchor_omitted_when_absent() {
        let mut r = sample();
        r.anchor = None;
        let v = r.to_value();
        assert!(v.get("anchor").is_none(), "absent anchor must be omitted");
        let back = CheckpointRecord::from_dag_cbor(&r.to_dag_cbor()).expect("round-trip");
        assert_eq!(r, back);
    }

    #[test]
    fn no_legal_identity_field_is_representable() {
        let mut v = sample().to_value();
        if let DagCbor::Map(ref mut pairs) = v {
            pairs.push((DagCbor::Text("operatorName".into()), DagCbor::Text("Acme LLC".into())));
            pairs.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(CheckpointRecord::from_value(&v).is_err());
    }

    #[test]
    fn record_validates_formats() {
        // Bad head hash cid.
        assert!(CheckpointRecord::new(1, "x", roots(), None).is_err());
        // Bad state-root cid.
        assert!(CheckpointRecord::new(
            1,
            "bafyreiledgerhead1234567890abcdefghij",
            vec![StateRoot {
                name: "accounts".into(),
                root: "bad".into()
            }],
            None
        )
        .is_err());
        // Empty state-root name.
        assert!(CheckpointRecord::new(
            1,
            "bafyreiledgerhead1234567890abcdefghij",
            vec![StateRoot {
                name: "".into(),
                root: "bafyreiaccountsroot1234567890abcdef".into()
            }],
            None
        )
        .is_err());
        // Empty anchor when present.
        assert!(CheckpointRecord::new(
            1,
            "bafyreiledgerhead1234567890abcdefghij",
            roots(),
            Some("".into())
        )
        .is_err());
    }
}
