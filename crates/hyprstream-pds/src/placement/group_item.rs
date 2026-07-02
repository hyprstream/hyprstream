//! `ai.hyprstream.placement.groupItem` — a group membership (atproto *listitem*).
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.placement.groupItem",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["group", "subject", "createdAt"],
//!     "properties": {
//!       "group":     { "type": "string", "format": "at-uri" },
//!       "subject":   { "type": "string" },
//!       "createdAt": { "type": "string", "format": "datetime" }
//!     }
//! }}}
//! ```
//!
//! Mirrors `app.bsky.graph.listitem`: `group` is the at-uri of the
//! `ai.hyprstream.placement.group` record, and `subject` is the DID of the node
//! being added to it.

use anyhow::{bail, Result};

use super::{map_get_str, validate_at_uri, validate_datetime, validate_did};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.placement.groupItem";

/// An `ai.hyprstream.placement.groupItem` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupItemRecord {
    /// at-uri of the owning group record. `format: "at-uri"`.
    pub group: String,
    /// The DID of the node being added to the group.
    pub subject: String,
    /// ISO-8601 UTC datetime. `format: "datetime"`.
    pub created_at: String,
}

impl GroupItemRecord {
    /// Validate and construct a record.
    pub fn new(
        group: impl Into<String>,
        subject: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let group = group.into();
        let subject = subject.into();
        let created_at = created_at.into();
        validate_at_uri(&group)?;
        validate_did(&subject)?;
        validate_datetime(&created_at)?;
        Ok(GroupItemRecord {
            group,
            subject,
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

    /// Build the typed [`DagCbor`] value form.
    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("group", DagCbor::Text(self.group.clone())),
            ("subject", DagCbor::Text(self.subject.clone())),
            ("createdAt", DagCbor::Text(self.created_at.clone())),
        ])
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let map = value.as_map()?;
        let group = map_get_str(value, "group", COLLECTION_NSID)?.to_owned();
        let subject = map_get_str(value, "subject", COLLECTION_NSID)?.to_owned();
        let created_at = map_get_str(value, "createdAt", COLLECTION_NSID)?.to_owned();
        // Reject unknown extra fields (the lexicon is frozen at 3 fields).
        for (k, _v) in map {
            match k.as_str()? {
                "group" | "subject" | "createdAt" => {}
                other => {
                    bail!("{COLLECTION_NSID}: unknown field {other:?} (lexicon is 3 fields)")
                }
            }
        }
        Self::new(group, subject, created_at)
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

    fn sample() -> GroupItemRecord {
        GroupItemRecord::new(
            "at://did:web:alice.example.com/ai.hyprstream.placement.group/3kxy",
            "did:web:node1.example.com",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = GroupItemRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        assert_eq!(r.to_dag_cbor(), back.to_dag_cbor());
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // canonical (lexicographic byte) key order.
        assert_eq!(keys, vec!["createdAt", "group", "subject"]);
    }

    #[test]
    fn record_rejects_extra_fields() {
        let mut extra = sample().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("extra".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(GroupItemRecord::from_value(&extra).is_err());
    }

    #[test]
    fn record_validates_formats() {
        // Bad group at-uri.
        assert!(GroupItemRecord::new("nope", "did:web:n", "2026-06-23T12:34:56.789Z").is_err());
        // Bad subject DID.
        assert!(GroupItemRecord::new(
            "at://did:web:x/ai.hyprstream.placement.group/3kxy",
            "not-a-did",
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad datetime.
        assert!(GroupItemRecord::new(
            "at://did:web:x/ai.hyprstream.placement.group/3kxy",
            "did:web:n",
            "2026"
        )
        .is_err());
    }
}
