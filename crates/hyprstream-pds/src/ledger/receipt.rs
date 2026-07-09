//! `ai.hyprstream.ledger.receipt` — a dual-signed usage record.
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.ledger.receipt",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["allocation", "spender", "host", "transferId", "quantum"],
//!     "properties": {
//!       "allocation": { "type": "string", "format": "cid" },
//!       "spender":    { "type": "string" },
//!       "host":       { "type": "string" },
//!       "transferId": { "type": "string" },
//!       "quantum":    { "type": "integer", "minimum": 0 }
//!     }
//! }}}
//! ```
//!
//! A receipt records one spend against a held [`crate::ledger::AllocationRecord`].
//! It binds **both** principals — the #681 two-principal vocabulary — so the
//! record attests who spent and who served:
//!
//! - `allocation` — CID of the allocation/grant the spend draws down (`format:
//!   "cid"` string).
//! - `spender` — the spending principal. For a **pseudonymous-tier** service this
//!   is the subject's **pairwise per-cell identifier** (#928 rule 7), never a
//!   root DID — a cell sees only the identifier the subject presents to it.
//! - `host` — the serving principal (the second of the two principals; the party
//!   on whose infrastructure the spend occurred).
//! - `transferId` — the spend's idempotency key. The write path derives the
//!   receipt's rkey from it, giving at-least-once (idempotent) receipt delivery:
//!   re-publishing the same `transferId` advances the same record.
//! - `quantum` — the amount spent, in the allocation unit's smallest quantum.

use anyhow::Result;

use super::{
    map_get_str, map_get_uint, reject_unknown_fields, validate_cid_string, validate_subject_id,
    validate_token,
};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.ledger.receipt";

/// An `ai.hyprstream.ledger.receipt` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptRecord {
    /// CID of the allocation/grant this spend draws down. `format: "cid"`.
    pub allocation: String,
    /// Spending principal — a pairwise per-cell id for pseudonymous-tier spends.
    pub spender: String,
    /// Serving principal (the second of the two #681 principals).
    pub host: String,
    /// Idempotency key; the receipt's rkey is derived from it.
    pub transfer_id: String,
    /// Amount spent, in the allocation unit's smallest quantum.
    pub quantum: u64,
}

impl ReceiptRecord {
    /// Validate and construct a record.
    pub fn new(
        allocation: impl Into<String>,
        spender: impl Into<String>,
        host: impl Into<String>,
        transfer_id: impl Into<String>,
        quantum: u64,
    ) -> Result<Self> {
        let allocation = allocation.into();
        let spender = spender.into();
        let host = host.into();
        let transfer_id = transfer_id.into();
        validate_cid_string(&allocation, "allocation")?;
        validate_subject_id(&spender, "spender")?;
        validate_subject_id(&host, "host")?;
        validate_token(&transfer_id, "transferId")?;
        Ok(ReceiptRecord {
            allocation,
            spender,
            host,
            transfer_id,
            quantum,
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
            ("allocation", DagCbor::Text(self.allocation.clone())),
            ("spender", DagCbor::Text(self.spender.clone())),
            ("host", DagCbor::Text(self.host.clone())),
            ("transferId", DagCbor::Text(self.transfer_id.clone())),
            ("quantum", DagCbor::Unsigned(self.quantum)),
        ])
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        let value = DagCbor::decode(bytes)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        reject_unknown_fields(
            value,
            &["allocation", "spender", "host", "transferId", "quantum"],
            COLLECTION_NSID,
        )?;
        let allocation = map_get_str(value, "allocation", COLLECTION_NSID)?.to_owned();
        let spender = map_get_str(value, "spender", COLLECTION_NSID)?.to_owned();
        let host = map_get_str(value, "host", COLLECTION_NSID)?.to_owned();
        let transfer_id = map_get_str(value, "transferId", COLLECTION_NSID)?.to_owned();
        let quantum = map_get_uint(value, "quantum", COLLECTION_NSID)?;
        Self::new(allocation, spender, host, transfer_id, quantum)
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

    fn sample() -> ReceiptRecord {
        ReceiptRecord::new(
            "bafyreiexamplealloccid1234567890abcdef",
            "pairwise:cell42:8f3ad9c0e1",
            "did:key:zHostNode",
            "transfer-01HXYZ",
            25,
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = ReceiptRecord::from_dag_cbor(&bytes).expect("round-trip");
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
        assert_eq!(
            keys,
            vec!["allocation", "host", "quantum", "spender", "transferId"]
        );
    }

    #[test]
    fn pairwise_spender_accepted() {
        // Receipts for pseudonymous-tier services bind the pairwise id, not a
        // root DID (#928 rule 7). A non-DID pairwise spender must be accepted.
        let r = sample();
        assert!(!r.spender.starts_with("did:"));
        assert_eq!(ReceiptRecord::from_dag_cbor(&r.to_dag_cbor()).expect("round-trip"), r);
    }

    #[test]
    fn no_legal_identity_field_is_representable() {
        let mut v = sample().to_value();
        if let DagCbor::Map(ref mut pairs) = v {
            pairs.push((DagCbor::Text("payerLegalId".into()), DagCbor::Text("SSN".into())));
            pairs.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(ReceiptRecord::from_value(&v).is_err());
    }

    #[test]
    fn record_validates_formats() {
        // Bad allocation cid.
        assert!(ReceiptRecord::new("x", "did:key:z", "did:key:h", "t", 1).is_err());
        // Empty spender.
        assert!(ReceiptRecord::new(
            "bafyreiexamplealloccid1234567890abcdef",
            "",
            "did:key:h",
            "t",
            1
        )
        .is_err());
        // Empty transferId.
        assert!(ReceiptRecord::new(
            "bafyreiexamplealloccid1234567890abcdef",
            "did:key:z",
            "did:key:h",
            "",
            1
        )
        .is_err());
    }
}
