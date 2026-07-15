//! `ai.hyprstream.name` — the NameRecord: the trusted, PQ-signed **name → DID
//! URL** layer (#908, design #905 §1/§3/§6; #879 §Name-resolution).
//!
//! This is the thin naming layer that sits *on top of* `ai.hyprstream.model`
//! (#387 D1). A NameRecord binds a human-chosen `label` to a `subject` **DID
//! URL** and a content `pin`. It is the trusted complement the DHT must never
//! answer: the DHT is an untrusted *locator*; the authoritative name → DID-URL
//! mapping lives here, in a PQ-signed atproto record served from a PDS.
//!
//! The confirmed 4-field lexicon:
//!
//! ```json
//! {
//!   "lexicon": 1,
//!   "id": "ai.hyprstream.name",
//!   "defs": { "main": { "type": "record", "key": "tid", "record": {
//!     "type": "object",
//!     "required": ["subject", "pin", "label", "createdAt"],
//!     "properties": {
//!       "subject":   { "type": "string" },               // a DID URL, opaque
//!       "pin":       { "type": "string", "format": "cid" },
//!       "label":     { "type": "string" },               // human label
//!       "createdAt": { "type": "string", "format": "datetime" }
//!     }
//! }}}
//! ```
//!
//! # `subject` carries a DID URL, not an at-uri, and never a 9P path
//!
//! The `subject` field is a **DID URL** carried as an opaque string (a standard
//! lexicon extension: unmodified Bluesky/Tangled can store, sync, and serve
//! these records without understanding the field). The acceptance contract is
//! that this DID URL *dereferences* via the G1 (#906) resolution path.
//!
//! Crucially, the value is a `did:`-scheme DID URL — **never** a raw 9P path and
//! **never** an `at://` URI. `at://` addresses this record; the record's body
//! resolves onward to a DID URL. Keeping 9P paths out of the at:// / record
//! layer is the #879 name-resolution non-goal made structural: paths live below
//! the DID, reached only after the DID URL dereferences.
//!
//! # Encoding
//!
//! Records are canonical **DAG-CBOR**: map keys in pure lexicographic byte order
//! (RFC 7049 §4.2.1 "core determinism"), matching the rest of this crate. For
//! these four keys that order is `createdAt`, `label`, `pin`, `subject`. Same
//! record → same bytes → same CID.
//!
//! `pin` is a `format: "cid"` **string** (not a DAG-CBOR link): it references
//! external content addressed outside this repo's MST, so it is carried as text
//! and validated as a CID string — the same treatment `currentOid` gets in
//! [`crate::record`].

use anyhow::{ensure, Result};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;
use crate::list_record::{map_get_str, validate_datetime, validate_nonempty};

/// The NSID of this record type — the collection portion of an at-uri that
/// addresses a NameRecord (`at://<repo>/ai.hyprstream.name/<rkey>`).
pub const COLLECTION_NSID: &str = "ai.hyprstream.name";

/// An `ai.hyprstream.name` record: the trusted name → DID-URL binding.
///
/// Construct with [`NameRecord::new`]; encode/decode via the DAG-CBOR
/// conversion methods.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NameRecord {
    /// The **DID URL** this name resolves to, carried opaquely. Must be a
    /// `did:`-scheme DID URL (it may carry a path/query/fragment); it dereferences
    /// via G1. Never an `at://` URI and never a 9P path.
    subject: String,
    /// The content pin — a `format: "cid"` string addressing the pinned content.
    pin: String,
    /// A human-readable label for this name.
    label: String,
    /// ISO-8601 UTC datetime (`format: "datetime"`), millisecond precision, `Z`.
    created_at: String,
}

impl NameRecord {
    /// Validate and construct a NameRecord.
    ///
    /// `subject` must be a `did:`-scheme DID URL (a DID, optionally with a
    /// path/query/fragment), `pin` a parseable CID string, `label` non-empty, and
    /// `created_at` the atproto `datetime` shape (UTC, milliseconds, trailing `Z`).
    pub fn new(
        subject: impl Into<String>,
        pin: impl Into<String>,
        label: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let subject = subject.into();
        let pin = pin.into();
        let label = label.into();
        let created_at = created_at.into();
        validate_did_url(&subject)?;
        validate_pin_cid(&pin)?;
        validate_nonempty(&label, "label")?;
        validate_datetime(&created_at)?;
        Ok(Self {
            subject,
            pin,
            label,
            created_at,
        })
    }

    /// The DID URL this name resolves to.
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// The content CID string pinned by this name.
    pub fn pin(&self) -> &str {
        &self.pin
    }

    /// Human-readable label for this name.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Creation timestamp in canonical atproto datetime shape.
    pub fn created_at(&self) -> &str {
        &self.created_at
    }

    /// Encode to canonical DAG-CBOR bytes (deterministic).
    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    /// The CID of this record (CIDv1 dag-cbor over its canonical bytes).
    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    /// Build the typed [`DagCbor`] value form (fields in canonical order).
    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("createdAt", DagCbor::Text(self.created_at.clone())),
            ("label", DagCbor::Text(self.label.clone())),
            ("pin", DagCbor::Text(self.pin.clone())),
            ("subject", DagCbor::Text(self.subject.clone())),
        ])
    }

    /// Decode canonical DAG-CBOR bytes into a NameRecord, validating all fields.
    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&DagCbor::decode(bytes)?)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let map = value.as_map()?;
        let subject = map_get_str(value, "subject", COLLECTION_NSID)?.to_owned();
        let pin = map_get_str(value, "pin", COLLECTION_NSID)?.to_owned();
        let label = map_get_str(value, "label", COLLECTION_NSID)?.to_owned();
        let created_at = map_get_str(value, "createdAt", COLLECTION_NSID)?.to_owned();
        // The lexicon is frozen at 4 fields — reject anything else.
        for (k, _v) in map {
            match k.as_str()? {
                "subject" | "pin" | "label" | "createdAt" => {}
                other => anyhow::bail!(
                    "{COLLECTION_NSID}: unknown field {other:?} (lexicon is 4 fields)"
                ),
            }
        }
        // `DagCbor::decode` already rejects duplicate map keys on the wire path,
        // but `from_value` is public: reject hand-built values where a recognized
        // key appears twice (which would make `map_get_str` pick one of two
        // conflicting values).
        ensure!(
            map.len() == 4,
            "{COLLECTION_NSID}: duplicate field (lexicon is exactly 4 fields)"
        );
        Self::new(subject, pin, label, created_at)
    }
}

/// A **DID URL** carried as an opaque string: it must be a `did:`-scheme URL
/// with a non-empty `method` and `method-specific-id` (the id may carry a
/// `/path`, `?query`, or `#fragment`). Deliberately lenient — full DID grammar
/// and actual dereferencing are the resolver's job (G1) — but it firmly rejects
/// the two things a NameRecord subject must never be: an `at://` URI and a bare
/// 9P path.
fn validate_did_url(s: &str) -> Result<()> {
    ensure!(
        s.starts_with("did:"),
        "subject must be a did: DID URL (not an at-uri or 9P path): {s:?}"
    );
    let rest = &s[4..];
    ensure!(
        !rest.is_empty(),
        "subject DID URL must have a method: {s:?}"
    );
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "subject DID URL must not contain whitespace: {s:?}"
    );
    let mut parts = rest.splitn(2, ':');
    let method = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("");
    ensure!(
        !method.is_empty() && !id.is_empty(),
        "subject must be \"did:<method>:<id>[/path][?query][#fragment]\": {s:?}"
    );
    ensure!(
        !matches!(id.as_bytes().first(), Some(b'/' | b'?' | b'#')),
        "subject DID URL must include a method-specific id before any path/query/fragment: {s:?}"
    );
    Ok(())
}

fn validate_pin_cid(s: &str) -> Result<()> {
    hyprstream_rpc::cid::decode_cid(s)
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("pin must be a parseable CIDv1 base32 string: {s:?}: {err}"))
}

#[cfg(test)]
fn sample_pin() -> String {
    match hyprstream_rpc::cid::encode_cid(
        hyprstream_rpc::cid::Codec::GitRaw,
        hyprstream_rpc::cid::HashAlgo::Sha2_256,
        &[0x42; 32],
    ) {
        Ok(cid) => cid,
        Err(err) => panic!("valid sample CID: {err}"),
    }
}

#[cfg(test)]
fn sample_at9p_pin() -> String {
    match hyprstream_rpc::cid::encode_cid(
        hyprstream_rpc::cid::Codec::At9pCapsule,
        hyprstream_rpc::cid::HashAlgo::Blake3,
        &[0x24; 64],
    ) {
        Ok(cid) => cid,
        Err(err) => panic!("valid sample at9p CID: {err}"),
    }
}

#[cfg(test)]
fn truncated_pin() -> String {
    let cid = match hyprstream_rpc::cid::encode_cid(
        hyprstream_rpc::cid::Codec::GitRaw,
        hyprstream_rpc::cid::HashAlgo::Sha2_256,
        &[0x11; 32],
    ) {
        Ok(cid) => cid,
        Err(err) => panic!("valid CID: {err}"),
    };
    cid.chars().take(12).collect()
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

    fn sample() -> NameRecord {
        NameRecord::new(
            "did:web:alice.example.com/models/qwen3",
            sample_pin(),
            "qwen3-serving",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn round_trip_same_cid() {
        let r = sample();
        let bytes = r.to_dag_cbor();
        let back = NameRecord::from_dag_cbor(&bytes).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
        assert_eq!(r.to_dag_cbor(), back.to_dag_cbor());
    }

    #[test]
    fn fields_canonical_order() {
        let r = sample();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        assert_eq!(
            keys,
            vec!["createdAt", "label", "pin", "subject"],
            "DAG-CBOR map keys must be pure-lexicographic byte order"
        );
    }

    #[test]
    fn subject_is_a_did_url_not_a_link() {
        let r = sample();
        let v = r.to_value();
        // subject is carried as an opaque Text string, not a DAG-CBOR link.
        assert!(matches!(v.get("subject"), Some(DagCbor::Text(_))));
        // pin likewise is a cid-string, not a tag-42 link.
        assert!(matches!(v.get("pin"), Some(DagCbor::Text(_))));
    }

    #[test]
    fn accepts_did_at9p_subject() {
        // The acceptance case: a did:at9p DID URL as subject.
        let r = NameRecord::new(
            "did:at9p:bafkrei1234567890abcdefghijklmnop#service",
            sample_at9p_pin(),
            "my-node",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("did:at9p subject must be accepted");
        assert!(r.subject().starts_with("did:at9p:"));
    }

    #[test]
    fn rejects_at_uri_subject() {
        // A NameRecord subject must never be an at:// URI — at:// addresses the
        // record; the body resolves onward to a DID URL.
        assert!(NameRecord::new(
            "at://did:web:alice.example.com/ai.hyprstream.model/3kxy",
            sample_pin(),
            "bad",
            "2026-06-23T12:34:56.789Z",
        )
        .is_err());
    }

    #[test]
    fn rejects_9p_path_subject() {
        // A raw 9P path must never leak into the at:// / record layer (#879).
        assert!(NameRecord::new(
            "/models/qwen3/adapters",
            sample_pin(),
            "bad",
            "2026-06-23T12:34:56.789Z",
        )
        .is_err());
    }

    #[test]
    fn rejects_did_url_without_method_specific_id_before_suffix() {
        for subject in ["did:web:/models/qwen3", "did:web:?q", "did:web:#f"] {
            assert!(
                NameRecord::new(subject, sample_pin(), "bad", "2026-06-23T12:34:56.789Z",).is_err(),
                "{subject:?} must not treat a URL suffix delimiter as the DID id"
            );
        }

        NameRecord::new(
            "did:web:alice.example.com/models/qwen3?version=1#svc",
            sample_pin(),
            "ok",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid DID URL suffix after method-specific id");
    }

    #[test]
    fn validates_formats() {
        // Bad pin (not a cid).
        assert!(NameRecord::new("did:web:x", "x", "l", "2026-06-23T12:34:56.789Z").is_err());
        // Bad pin (malformed base32 CID body).
        assert!(
            NameRecord::new("did:web:x", "b!!!!!!!!", "l", "2026-06-23T12:34:56.789Z").is_err()
        );
        // Bad pin (valid base32 prefix but truncated multihash/CID body).
        assert!(NameRecord::new(
            "did:web:x",
            truncated_pin(),
            "l",
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
        // Empty label.
        assert!(
            NameRecord::new("did:web:x", sample_pin(), "", "2026-06-23T12:34:56.789Z").is_err()
        );
        // Bad datetime.
        assert!(
            NameRecord::new("did:web:x", sample_pin(), "l", "2026-06-23T12:34:56.789").is_err()
        );
    }

    #[test]
    fn rejects_extra_fields() {
        let mut extra = sample().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("bogus".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(NameRecord::from_value(&extra).is_err());
    }

    #[test]
    fn rejects_duplicate_recognized_field() {
        // A hand-built value with a second `subject` must not parse: the wire
        // decoder rejects duplicate keys, and the value-level path must too.
        let mut dup = sample().to_value();
        if let DagCbor::Map(ref mut v) = dup {
            v.push((
                DagCbor::Text("subject".into()),
                DagCbor::Text("did:web:evil.example.com".into()),
            ));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(NameRecord::from_value(&dup).is_err());
    }

    #[test]
    fn deterministic_across_rebuild() {
        let r1 = sample();
        let r2 =
            NameRecord::new(r1.subject(), r1.pin(), r1.label(), r1.created_at()).expect("rebuild");
        assert_eq!(r1.to_dag_cbor(), r2.to_dag_cbor());
        assert_eq!(r1.cid(), r2.cid());
    }
}
