//! `ai.hyprstream.ledger.allocation` — an issuer-signed grant record (the held
//! entitlement / inventory line).
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.ledger.allocation",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["grant", "unit", "amount", "epoch", "issuer", "holder", "class"],
//!     "properties": {
//!       "grant":  { "type": "string", "format": "cid" },
//!       "unit":   { "type": "object",
//!                   "required": ["code", "issuer"], "properties": {
//!                     "code":   { "type": "string" },
//!                     "issuer": { "type": "string", "format": "did" } } },
//!       "amount": { "type": "integer", "minimum": 0 },
//!       "epoch":  { "type": "integer", "minimum": 0 },
//!       "issuer": { "type": "string", "format": "did" },
//!       "holder": { "type": "string" },
//!       "class":  { "type": "string", "enum": ["prepaid", "underwritten"] }
//!     }
//! }}}
//! ```
//!
//! - `grant` — the CID of the UCAN grant this allocation realizes (`format: "cid"`
//!   string; the UCAN lives outside this repo's MST).
//! - `unit` — the credit unit, which **names its issuer** ([`Unit`]): credits are
//!   issuer *liabilities* (D8-1), never bearer tokens. There is no bare unit.
//! - `amount` — the granted amount, in the unit's smallest quantum (`u64`).
//! - `epoch` — the issuance epoch (`u64`).
//! - `issuer` — the issuing DID. Must equal `unit.issuer` (an allocation is only
//!   ever the unit-issuer's own liability).
//! - `holder` — the subject: a pseudonymous `did:key`/`did:at9p` **or** a pairwise
//!   per-cell identifier (#928 rule 3). Never forced to a root/`did:web` identity.
//! - `class` — the grant [`GrantClass`]: `prepaid` (bearer-like, lease-mode-only,
//!   issuable to a bare `did:key`, needs no identity) vs `underwritten`
//!   (issuer-relationship, the only detect-mode-eligible class). The class↔mode
//!   coupling is a protocol rule (#928 rule 4).

use anyhow::{bail, Result};

use super::{
    map_get_str, map_get_uint, reject_unknown_fields, validate_cid_string, validate_did,
    validate_subject_id,
};
use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type.
pub const COLLECTION_NSID: &str = "ai.hyprstream.ledger.allocation";

/// A credit unit that **names its issuer**.
///
/// The issuer is baked into the unit itself, so a credit is inseparable from the
/// party liable for it (D8-1). `code` is the issuer-scoped unit label (e.g.
/// `"compute-second"`, `"usd-cents"`) — it is only meaningful relative to
/// `issuer`, never a global bearer denomination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unit {
    /// Issuer-scoped unit label. Non-empty.
    pub code: String,
    /// The DID liable for credits denominated in this unit. `format: "did"`.
    pub issuer: String,
}

/// The grant class — couples anonymity to spend mode (#928 rule 4).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrantClass {
    /// Bearer-like: paid up front, so the issuer bears no credit risk and needs
    /// no recourse relationship — issuable to a bare `did:key` with no identity.
    /// **Restricted to lease/prevention mode**: overspend is made impossible, not
    /// detected after the fact.
    Prepaid,
    /// Postpaid: the issuer has a real bilateral relationship (employment,
    /// contract, billing) and recovers from cheaters itself. The only
    /// **detect-mode-eligible** class. The issuer's knowledge of its holder is a
    /// bilateral relationship, never protocol data.
    Underwritten,
}

impl GrantClass {
    /// The lexicon string form.
    pub fn as_str(self) -> &'static str {
        match self {
            GrantClass::Prepaid => "prepaid",
            GrantClass::Underwritten => "underwritten",
        }
    }

    /// Parse the lexicon string form.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "prepaid" => Ok(GrantClass::Prepaid),
            "underwritten" => Ok(GrantClass::Underwritten),
            other => bail!("{COLLECTION_NSID}: unknown grant class {other:?}"),
        }
    }

    /// Whether this class may be spent in **detect mode**.
    ///
    /// Detect mode observes overspend after it happens and relies on *recourse*
    /// to recover — which requires a relationship with the holder. A prepaid
    /// bearer grant has no such relationship, so detect mode on it is worthless
    /// (detection without recourse) *and* impossible to act on (recourse without
    /// identity). The coupling is therefore a protocol rule, not a policy knob:
    /// only [`GrantClass::Underwritten`] is detect-mode-eligible.
    pub fn detect_mode_eligible(self) -> bool {
        matches!(self, GrantClass::Underwritten)
    }
}

/// An `ai.hyprstream.ledger.allocation` record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllocationRecord {
    /// CID of the UCAN grant this allocation realizes. `format: "cid"`.
    pub grant: String,
    /// The credit unit (names its issuer).
    pub unit: Unit,
    /// Granted amount, in the unit's smallest quantum.
    pub amount: u64,
    /// Issuance epoch.
    pub epoch: u64,
    /// Issuing DID. Must equal `unit.issuer`. `format: "did"`.
    pub issuer: String,
    /// Subject: a pseudonymous DID or a pairwise per-cell identifier.
    pub holder: String,
    /// The grant class.
    pub class: GrantClass,
}

impl AllocationRecord {
    /// Validate and construct a record.
    pub fn new(
        grant: impl Into<String>,
        unit: Unit,
        amount: u64,
        epoch: u64,
        issuer: impl Into<String>,
        holder: impl Into<String>,
        class: GrantClass,
    ) -> Result<Self> {
        let grant = grant.into();
        let issuer = issuer.into();
        let holder = holder.into();
        validate_cid_string(&grant, "grant")?;
        validate_did(&issuer)?;
        validate_did(&unit.issuer)?;
        if unit.code.is_empty() {
            bail!("{COLLECTION_NSID}: unit.code must not be empty");
        }
        // D8-1: a credit is the *unit-issuer's* liability, so an allocation can
        // only be issued by that same issuer. This binds the record to its
        // liability holder structurally rather than by convention.
        if issuer != unit.issuer {
            bail!(
                "{COLLECTION_NSID}: issuer {issuer:?} must equal unit.issuer {:?} \
                 (credits are the unit-issuer's liability, D8-1)",
                unit.issuer
            );
        }
        validate_subject_id(&holder, "holder")?;
        Ok(AllocationRecord {
            grant,
            unit,
            amount,
            epoch,
            issuer,
            holder,
            class,
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
        let unit = DagCbor::str_map([
            ("code", DagCbor::Text(self.unit.code.clone())),
            ("issuer", DagCbor::Text(self.unit.issuer.clone())),
        ]);
        DagCbor::str_map([
            ("grant", DagCbor::Text(self.grant.clone())),
            ("unit", unit),
            ("amount", DagCbor::Unsigned(self.amount)),
            ("epoch", DagCbor::Unsigned(self.epoch)),
            ("issuer", DagCbor::Text(self.issuer.clone())),
            ("holder", DagCbor::Text(self.holder.clone())),
            ("class", DagCbor::Text(self.class.as_str().to_owned())),
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
            &["grant", "unit", "amount", "epoch", "issuer", "holder", "class"],
            COLLECTION_NSID,
        )?;
        let grant = map_get_str(value, "grant", COLLECTION_NSID)?.to_owned();
        let unit = parse_unit(
            value
                .get("unit")
                .ok_or_else(|| anyhow::anyhow!("{COLLECTION_NSID}: missing required field \"unit\""))?,
        )?;
        let amount = map_get_uint(value, "amount", COLLECTION_NSID)?;
        let epoch = map_get_uint(value, "epoch", COLLECTION_NSID)?;
        let issuer = map_get_str(value, "issuer", COLLECTION_NSID)?.to_owned();
        let holder = map_get_str(value, "holder", COLLECTION_NSID)?.to_owned();
        let class = GrantClass::parse(map_get_str(value, "class", COLLECTION_NSID)?)?;
        Self::new(grant, unit, amount, epoch, issuer, holder, class)
    }
}

fn parse_unit(value: &DagCbor) -> Result<Unit> {
    reject_unknown_fields(value, &["code", "issuer"], COLLECTION_NSID)?;
    let code = map_get_str(value, "code", COLLECTION_NSID)?.to_owned();
    let issuer = map_get_str(value, "issuer", COLLECTION_NSID)?.to_owned();
    Ok(Unit { code, issuer })
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

    const ISSUER: &str = "did:web:issuer.example.com";

    fn unit() -> Unit {
        Unit {
            code: "compute-second".into(),
            issuer: ISSUER.into(),
        }
    }

    fn sample() -> AllocationRecord {
        AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            unit(),
            1_000,
            7,
            ISSUER,
            "did:key:zHolderSubject",
            GrantClass::Underwritten,
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = AllocationRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        // Canonical: reserialize(decode(bytes)) == bytes.
        assert_eq!(back.to_dag_cbor(), bytes);
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // pure lexicographic byte order.
        assert_eq!(
            keys,
            vec!["amount", "class", "epoch", "grant", "holder", "issuer", "unit"]
        );
    }

    #[test]
    fn no_legal_identity_field_is_representable() {
        // A record carrying a legal-identity-looking field must not decode: the
        // schema has no such field, and unknown fields are rejected (#928 rule 1).
        let mut v = sample().to_value();
        if let DagCbor::Map(ref mut pairs) = v {
            pairs.push((DagCbor::Text("legalName".into()), DagCbor::Text("Jane Doe".into())));
            pairs.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        let err = AllocationRecord::from_value(&v).expect_err("legal-identity field must be rejected");
        assert!(err.to_string().contains("unknown field"), "unexpected: {err:#}");
    }

    #[test]
    fn pairwise_holder_accepted() {
        // A pairwise per-cell identifier (not a DID) is a valid holder (#928 rule 3).
        let r = AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            unit(),
            5,
            1,
            ISSUER,
            "pairwise:cell42:8f3ad9c0e1",
            GrantClass::Prepaid,
        )
        .expect("pairwise holder must be accepted");
        assert_eq!(
            AllocationRecord::from_dag_cbor(&r.to_dag_cbor()).expect("round-trip"),
            r
        );
    }

    #[test]
    fn prepaid_issuable_to_bare_did_key() {
        // Prepaid (bearer-like) is issuable to a bare did:key with no identity.
        AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            unit(),
            5,
            1,
            ISSUER,
            "did:key:zBareKeyNoIdentity",
            GrantClass::Prepaid,
        )
        .expect("prepaid to bare did:key must be accepted");
    }

    #[test]
    fn class_mode_coupling() {
        // Only underwritten is detect-mode eligible (#928 rule 4).
        assert!(!GrantClass::Prepaid.detect_mode_eligible());
        assert!(GrantClass::Underwritten.detect_mode_eligible());
    }

    #[test]
    fn unit_names_its_issuer_and_binds_liability() {
        // The unit carries its issuer, and the top-level issuer must match it
        // (D8-1: a credit is the unit-issuer's liability).
        let mismatched = AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            unit(),
            5,
            1,
            "did:web:other.example.com",
            "did:key:zHolder",
            GrantClass::Underwritten,
        );
        assert!(mismatched.is_err(), "issuer != unit.issuer must be rejected");
    }

    #[test]
    fn record_validates_formats() {
        // Bad grant cid.
        assert!(AllocationRecord::new("x", unit(), 1, 1, ISSUER, "did:key:z", GrantClass::Prepaid).is_err());
        // Bad issuer did.
        assert!(AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            unit(),
            1,
            1,
            "not-a-did",
            "did:key:z",
            GrantClass::Prepaid
        )
        .is_err());
        // Empty holder.
        assert!(AllocationRecord::new(
            "bafyreiexamplegrantcid1234567890abcdef",
            unit(),
            1,
            1,
            ISSUER,
            "",
            GrantClass::Prepaid
        )
        .is_err());
        // Unknown class string on decode.
        let mut v = sample().to_value();
        if let DagCbor::Map(ref mut pairs) = v {
            for (k, val) in pairs.iter_mut() {
                if k.as_str().ok() == Some("class") {
                    *val = DagCbor::Text("overdraft".into());
                }
            }
        }
        assert!(AllocationRecord::from_value(&v).is_err());
    }

    #[test]
    fn grant_and_cids_are_strings_not_links() {
        let v = sample().to_value();
        assert!(matches!(v.get("grant"), Some(DagCbor::Text(_))));
    }
}
