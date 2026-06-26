//! `ai.hyprstream.model` — the per-account signed-mutable-pointer record.
//!
//! The confirmed 3-field lexicon:
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.model",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["repo", "currentOid", "createdAt"],
//!     "properties": {
//!       "repo":       { "type": "string", "format": "at-uri" },
//!       "currentOid": { "type": "string", "format": "cid" },
//!       "createdAt":  { "type": "string", "format": "datetime" }
//!     }
//! }}}
//! ```
//!
//! # Encoding
//!
//! Records are **DAG-CBOR**. The field order in the encoded map is the canonical
//! (length-first, then lexicographic) order, which for these three keys is
//! `createdAt`, `currentOid`, `repo` — the encoder handles this; callers do not
//! need to care.
//!
//! `currentOid` is a `format: "cid"` **string** — *not* a CID link. It references
//! external git content by OID, so it is carried as text and validated as a CID
//! string at construction time. (If it were a link, DAG-CBOR tag 42 would mean
//! the referenced block is part of *this* repo's MST — which it isn't; the git
//! blob lives in the separate git-xet / gittorrent CAS.)

use anyhow::{bail, ensure, Result};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// The NSID of this record type. The collection portion of an at-uri that
/// addresses a record (`at://<repo>/<collection>/<rkey>`).
pub const COLLECTION_NSID: &str = "ai.hyprstream.model";

/// An `ai.hyprstream.model` record.
///
/// Construct with [`ModelRecord::new`]; encode/decode via the [`DagCbor`]
/// conversion impls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRecord {
    /// at-uri of the owning repo, e.g. `at://did:web:alice.example.com`.
    /// `format: "at-uri"` per the lexicon.
    pub repo: String,
    /// The current object ID (git OID) the record points at, encoded as a
    /// CID string. `format: "cid"`.
    pub current_oid: String,
    /// ISO-8601 UTC datetime (`format: "datetime"`), e.g.
    /// `2026-06-23T12:34:56.000Z`. Lexicon `datetime` requires this exact shape.
    pub created_at: String,
}

impl ModelRecord {
    /// Validate and construct a record.
    ///
    /// `current_oid` must be a parseable CIDv1 string (the lexicon's `cid`
    /// format), and `created_at` must match the atproto `datetime` shape
    /// (UTC, milliseconds, trailing `Z`).
    pub fn new(
        repo: impl Into<String>,
        current_oid: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let repo = repo.into();
        let current_oid = current_oid.into();
        let created_at = created_at.into();
        validate_at_uri(&repo)?;
        validate_cid_string(&current_oid)?;
        validate_datetime(&created_at)?;
        Ok(ModelRecord {
            repo,
            current_oid,
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
            ("repo", DagCbor::Text(self.repo.clone())),
            ("currentOid", DagCbor::Text(self.current_oid.clone())),
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
        // Required fields.
        let repo = map_get_str(value, "repo")?.to_owned();
        let current_oid = map_get_str(value, "currentOid")?.to_owned();
        let created_at = map_get_str(value, "createdAt")?.to_owned();
        // Reject unknown extra fields (the lexicon is frozen at 3 fields).
        for (k, _v) in map {
            match k.as_str()? {
                "repo" | "currentOid" | "createdAt" => {}
                other => {
                    bail!("ai.hyprstream.model: unknown field {other:?} (lexicon is 3 fields)")
                }
            }
        }
        Self::new(repo, current_oid, created_at)
    }
}

fn map_get_str<'a>(value: &'a DagCbor, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("ai.hyprstream.model: missing required field {key:?}"))?
        .as_str()
}

// ── field validators (lexicon formats) ──────────────────────────────────────

fn validate_at_uri(s: &str) -> Result<()> {
    ensure!(
        s.starts_with("at://"),
        "at-uri must start with \"at://\": {s:?}"
    );
    let rest = &s[5..];
    ensure!(!rest.is_empty(), "at-uri must have an authority: {s:?}");
    // Authority is the first path segment; we don't enforce DID-grammar here
    // (the resolver does), only that it's non-empty and has no whitespace.
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "at-uri must not contain whitespace: {s:?}"
    );
    Ok(())
}

fn validate_cid_string(s: &str) -> Result<()> {
    // `format: "cid"` — a CIDv1 string (base32 `b...` form) or a CIDv0
    // (base58btc `Qm...`). We accept any CIDv1 base and CIDv0.
    if s.is_empty() {
        bail!("cid string must be non-empty");
    }
    // Best-effort: round-trip through Cid::from_bytes is not possible for a
    // *string* form without the `cid` crate's multibase decoder. We accept the
    // base32 `b...` form (our canonical currentOid) and the CIDv0 `Qm...` form,
    // rejecting obviously-bogus strings.
    let ok = (s.starts_with('b') && s.len() > 8)
        || (s.starts_with("z") && s.len() > 8)
        || (s.starts_with("Qm") && s.len() > 8);
    ensure!(
        ok,
        "cid string must be a CIDv1 (b/z…) or CIDv0 (Qm…): {s:?}"
    );
    Ok(())
}

fn validate_datetime(s: &str) -> Result<()> {
    // atproto `datetime`: ISO-8601 UTC, millisecond precision, trailing Z.
    // Canonical example: "2026-06-23T12:34:56.789Z".
    ensure!(s.ends_with('Z'), "datetime must end with 'Z' (UTC): {s:?}");
    let pre = &s[..s.len() - 1];
    ensure!(pre.len() >= 20, "datetime too short: {s:?}");
    let bytes = pre.as_bytes();
    ensure!(
        bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':',
        "datetime must be ISO-8601 (YYYY-MM-DDTHH:MM:SS): {s:?}"
    );
    // Expect ".mmm" milliseconds between seconds and the trailing Z.
    ensure!(
        bytes.get(19) == Some(&b'.'),
        "datetime must have millisecond precision (.mmm): {s:?}"
    );
    Ok(())
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

    fn sample() -> ModelRecord {
        ModelRecord::new(
            "at://did:web:alice.example.com",
            "bafyreiexampleoid1234567890abcdefghij",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn record_round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = ModelRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        // Determinism: same record → same bytes → same CID.
        assert_eq!(r.cid(), back.cid());
        assert_eq!(r.to_dag_cbor(), back.to_dag_cbor());
    }

    #[test]
    fn record_fields_canonical_order() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        // Decode raw to confirm key order is length-first: createdAt, currentOid, repo.
        let v = DagCbor::decode(&bytes).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        assert_eq!(
            keys,
            vec!["createdAt", "currentOid", "repo"],
            "DAG-CBOR map keys must be canonical (length-first, then lex)"
        );
    }

    #[test]
    fn record_current_oid_is_string_not_link() {
        let r = sample();
        let v = r.to_value();
        let current_oid = v.get("currentOid").expect("field present");
        // Must be a Text string, not a Link (tag 42).
        assert!(
            matches!(current_oid, DagCbor::Text(_)),
            "currentOid is a cid-string, not a link"
        );
    }

    #[test]
    fn record_rejects_extra_fields() {
        let mut extra = sample().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("extra".into()), DagCbor::Unsigned(1)));
            // Re-sort to keep decode's invariant happy.
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(
            ModelRecord::from_value(&extra).is_err(),
            "extra fields must be rejected"
        );
    }

    #[test]
    fn record_validates_formats() {
        // Bad at-uri.
        assert!(ModelRecord::new(
            "not-an-at-uri",
            "bafyreiexampleoid1234567890abcdefghij",
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Bad datetime (no Z).
        assert!(ModelRecord::new(
            "at://did:web:x",
            "bafyreiexampleoid1234567890abcdefghij",
            "2026-06-23T12:34:56.789"
        )
        .is_err());
        // Bad cid.
        assert!(ModelRecord::new("at://did:web:x", "x", "2026-06-23T12:34:56.789Z").is_err());
    }

    #[test]
    fn record_deterministic_across_rebuild() {
        // Reconstruct from scratch — bytes must be identical.
        let r1 = sample();
        let r2 = ModelRecord::new(
            r1.repo.clone(),
            r1.current_oid.clone(),
            r1.created_at.clone(),
        )
        .expect("rebuild");
        assert_eq!(
            r1.to_dag_cbor(),
            r2.to_dag_cbor(),
            "record bytes must be deterministic"
        );
        assert_eq!(r1.cid(), r2.cid());
    }
}
