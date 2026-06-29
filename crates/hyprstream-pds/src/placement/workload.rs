//! `ai.hyprstream.placement.workload` — a workload→node placement decision.
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.placement.workload",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["repo", "workload", "node", "createdAt"],
//!     "properties": {
//!       "repo":      { "type": "string", "format": "at-uri" },
//!       "workload":  { "type": "string" },
//!       "node":      { "type": "string" },
//!       "group":     { "type": "string" },
//!       "createdAt": { "type": "string", "format": "datetime" }
//!     }
//! }}}
//! ```
//!
//! `workload` is a model at-uri or shard id (free-form string). `node` is the
//! DID of the placed node. `group` is the optional consent group the placement
//! was made under.

use anyhow::{bail, Result};

use super::{
    map_get_opt_str, map_get_str, validate_at_uri, validate_datetime, validate_did,
    validate_nonempty,
};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.placement.workload";

/// An `ai.hyprstream.placement.workload` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkloadRecord {
    /// at-uri of the owning repo. `format: "at-uri"`.
    pub repo: String,
    /// The workload identifier: a model at-uri or a shard id.
    pub workload: String,
    /// The DID of the node the workload is placed on.
    pub node: String,
    /// Optional consent group the placement was made under.
    pub group: Option<String>,
    /// ISO-8601 UTC datetime. `format: "datetime"`.
    pub created_at: String,
}

impl WorkloadRecord {
    /// Validate and construct a record.
    pub fn new(
        repo: impl Into<String>,
        workload: impl Into<String>,
        node: impl Into<String>,
        group: Option<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let repo = repo.into();
        let workload = workload.into();
        let node = node.into();
        let created_at = created_at.into();
        validate_at_uri(&repo)?;
        validate_nonempty(&workload, "workload")?;
        validate_did(&node)?;
        if let Some(g) = &group {
            validate_nonempty(g, "group")?;
        }
        validate_datetime(&created_at)?;
        Ok(WorkloadRecord {
            repo,
            workload,
            node,
            group,
            created_at,
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

    /// Build the typed [`DagCbor`] value form. The optional `group` field is
    /// omitted from the map when absent.
    pub fn to_value(&self) -> DagCbor {
        let mut pairs: Vec<(&str, DagCbor)> = vec![
            ("repo", DagCbor::Text(self.repo.clone())),
            ("workload", DagCbor::Text(self.workload.clone())),
            ("node", DagCbor::Text(self.node.clone())),
            ("createdAt", DagCbor::Text(self.created_at.clone())),
        ];
        if let Some(group) = &self.group {
            pairs.push(("group", DagCbor::Text(group.clone())));
        }
        DagCbor::str_map(pairs)
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let map = value.as_map()?;
        let repo = map_get_str(value, "repo", COLLECTION_NSID)?.to_owned();
        let workload = map_get_str(value, "workload", COLLECTION_NSID)?.to_owned();
        let node = map_get_str(value, "node", COLLECTION_NSID)?.to_owned();
        let group = map_get_opt_str(value, "group")?.map(str::to_owned);
        let created_at = map_get_str(value, "createdAt", COLLECTION_NSID)?.to_owned();
        // Reject unknown extra fields (frozen field set: 4 required + 1 optional).
        for (k, _v) in map {
            match k.as_str()? {
                "repo" | "workload" | "node" | "group" | "createdAt" => {}
                other => {
                    bail!("{COLLECTION_NSID}: unknown field {other:?} (lexicon is 5 fields)")
                }
            }
        }
        Self::new(repo, workload, node, group, created_at)
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

    fn sample() -> WorkloadRecord {
        WorkloadRecord::new(
            "at://did:web:alice.example.com",
            "at://did:web:alice.example.com/ai.hyprstream.model/3kxy",
            "did:web:node1.example.com",
            Some("at://did:web:alice.example.com/ai.hyprstream.placement.group/3kxz".into()),
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = WorkloadRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        assert_eq!(r.to_dag_cbor(), back.to_dag_cbor());
    }

    #[test]
    fn record_optional_group_omitted() {
        let mut r = sample();
        r.group = None;
        let v = r.to_value();
        assert!(v.get("group").is_none(), "absent group must be omitted");
        let back = WorkloadRecord::from_dag_cbor(&r.to_dag_cbor()).expect("round-trip");
        assert_eq!(r, back);
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // canonical (lexicographic byte) key order.
        assert_eq!(keys, vec!["createdAt", "group", "node", "repo", "workload"]);
    }

    #[test]
    fn record_rejects_extra_fields() {
        let mut extra = sample().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("extra".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(WorkloadRecord::from_value(&extra).is_err());
    }

    #[test]
    fn record_validates_formats() {
        // Bad repo at-uri.
        assert!(
            WorkloadRecord::new("nope", "w", "did:web:n", None, "2026-06-23T12:34:56.789Z")
                .is_err()
        );
        // Empty workload.
        assert!(WorkloadRecord::new(
            "at://did:web:x",
            "",
            "did:web:n",
            None,
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad node DID.
        assert!(WorkloadRecord::new(
            "at://did:web:x",
            "w",
            "not-a-did",
            None,
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Empty optional group.
        assert!(WorkloadRecord::new(
            "at://did:web:x",
            "w",
            "did:web:n",
            Some("".into()),
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad datetime.
        assert!(WorkloadRecord::new("at://did:web:x", "w", "did:web:n", None, "2026").is_err());
    }
}
