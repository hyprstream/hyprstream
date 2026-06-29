//! `ai.hyprstream.placement.group` — a consent group (the atproto *list* record).
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.placement.group",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["name", "ownerDid", "createdAt"],
//!     "properties": {
//!       "name":      { "type": "string" },
//!       "ownerDid":  { "type": "string" },
//!       "purpose":   { "type": "string" },
//!       "createdAt": { "type": "string", "format": "datetime" }
//!     }
//! }}}
//! ```
//!
//! Mirrors `app.bsky.graph.list`: this record *names* a group; membership is
//! carried by `ai.hyprstream.placement.groupItem` (the listitem). `ownerDid` is
//! the DID that owns the group.

use anyhow::{bail, Result};

use super::{map_get_opt_str, map_get_str, validate_datetime, validate_did, validate_nonempty};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.placement.group";

/// An `ai.hyprstream.placement.group` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupRecord {
    /// Human-readable group name.
    pub name: String,
    /// The DID that owns the group.
    pub owner_did: String,
    /// Optional free-form description of the group's purpose.
    pub purpose: Option<String>,
    /// ISO-8601 UTC datetime. `format: "datetime"`.
    pub created_at: String,
}

impl GroupRecord {
    /// Validate and construct a record.
    pub fn new(
        name: impl Into<String>,
        owner_did: impl Into<String>,
        purpose: Option<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let name = name.into();
        let owner_did = owner_did.into();
        let created_at = created_at.into();
        validate_nonempty(&name, "name")?;
        validate_did(&owner_did)?;
        if let Some(p) = &purpose {
            validate_nonempty(p, "purpose")?;
        }
        validate_datetime(&created_at)?;
        Ok(GroupRecord {
            name,
            owner_did,
            purpose,
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

    /// Build the typed [`DagCbor`] value form. The optional `purpose` field is
    /// omitted from the map when absent.
    pub fn to_value(&self) -> DagCbor {
        let mut pairs: Vec<(&str, DagCbor)> = vec![
            ("name", DagCbor::Text(self.name.clone())),
            ("ownerDid", DagCbor::Text(self.owner_did.clone())),
            ("createdAt", DagCbor::Text(self.created_at.clone())),
        ];
        if let Some(purpose) = &self.purpose {
            pairs.push(("purpose", DagCbor::Text(purpose.clone())));
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
        let name = map_get_str(value, "name", COLLECTION_NSID)?.to_owned();
        let owner_did = map_get_str(value, "ownerDid", COLLECTION_NSID)?.to_owned();
        let purpose = map_get_opt_str(value, "purpose")?.map(str::to_owned);
        let created_at = map_get_str(value, "createdAt", COLLECTION_NSID)?.to_owned();
        // Reject unknown extra fields (frozen field set: 3 required + 1 optional).
        for (k, _v) in map {
            match k.as_str()? {
                "name" | "ownerDid" | "purpose" | "createdAt" => {}
                other => {
                    bail!("{COLLECTION_NSID}: unknown field {other:?} (lexicon is 4 fields)")
                }
            }
        }
        Self::new(name, owner_did, purpose, created_at)
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

    fn sample() -> GroupRecord {
        GroupRecord::new(
            "east-coast-gpus",
            "did:web:alice.example.com",
            Some("GPU nodes in us-east".into()),
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = GroupRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        assert_eq!(r.to_dag_cbor(), back.to_dag_cbor());
    }

    #[test]
    fn record_optional_purpose_omitted() {
        let mut r = sample();
        r.purpose = None;
        let v = r.to_value();
        assert!(v.get("purpose").is_none(), "absent purpose must be omitted");
        let back = GroupRecord::from_dag_cbor(&r.to_dag_cbor()).expect("round-trip");
        assert_eq!(r, back);
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // canonical (lexicographic byte) key order.
        assert_eq!(keys, vec!["createdAt", "name", "ownerDid", "purpose"]);
    }

    #[test]
    fn record_rejects_extra_fields() {
        let mut extra = sample().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("extra".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(GroupRecord::from_value(&extra).is_err());
    }

    #[test]
    fn record_validates_formats() {
        // Empty name.
        assert!(GroupRecord::new("", "did:web:x", None, "2026-06-23T12:34:56.789Z").is_err());
        // Bad owner DID.
        assert!(GroupRecord::new("g", "not-a-did", None, "2026-06-23T12:34:56.789Z").is_err());
        // Empty optional purpose.
        assert!(GroupRecord::new(
            "g",
            "did:web:x",
            Some("".into()),
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad datetime.
        assert!(GroupRecord::new("g", "did:web:x", None, "2026").is_err());
    }
}
