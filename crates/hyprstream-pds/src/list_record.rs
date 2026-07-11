//! Generic atproto "list" + "listitem" record shapes, shared by every
//! list-style lexicon in this crate (`ai.hyprstream.placement.group`/
//! `groupItem`, `ai.hyprstream.event.group`/`groupItem`, and any future one).
//!
//! Mirrors `app.bsky.graph.list`/`listitem`: a *list* record names a group
//! (`name`/`ownerDid`/`purpose?`/`createdAt`); a *listitem* record (in the
//! owner's repo) claims a member of it (`group`/`subject`/`createdAt`). Two
//! independently-signed-off lexicons (`ai.hyprstream.placement.*` and
//! `ai.hyprstream.event.*`) need this exact shape but are NOT the same record
//! type — placement membership (who can be scheduled together) and event-group
//! membership (who shares a broadcast key) are distinct axes that happen to
//! share identical list/listitem mechanics. Keeping them as separate NSIDs
//! preserves cheap collection-based firehose/AppView filtering and avoids
//! reopening either lexicon's frozen sign-off; this module is where the
//! mechanics they share live, so neither duplicates it by hand.
//!
//! [`ListItemRecord`] is used directly (group/subject/createdAt is identical
//! across every list lexicon so far — no per-lexicon fields needed).
//! [`ListRecord<E>`] is generic over `E: ListExtra`, the lexicon-specific extra
//! fields beyond the common `name`/`ownerDid`/`purpose`/`createdAt` shape
//! (`()` for no extra fields; a lexicon-specific payload otherwise).

use anyhow::{bail, ensure, Result};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

// ── shared field accessors / validators ─────────────────────────────────────

/// Read a required string field, attributing errors to `nsid`.
pub fn map_get_str<'a>(value: &'a DagCbor, key: &str, nsid: &str) -> Result<&'a str> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("{nsid}: missing required field {key:?}"))?
        .as_str()
}

/// Read an optional string field. Returns `None` when the field is omitted; an
/// error if it is present but not a text string.
pub fn map_get_opt_str<'a>(value: &'a DagCbor, key: &str) -> Result<Option<&'a str>> {
    match value.get(key) {
        None => Ok(None),
        Some(v) => Ok(Some(v.as_str()?)),
    }
}

/// `format: "at-uri"` — must start with `at://` and carry a non-empty,
/// whitespace-free authority. (DID grammar is enforced by the resolver.)
pub fn validate_at_uri(s: &str) -> Result<()> {
    ensure!(
        s.starts_with("at://"),
        "at-uri must start with \"at://\": {s:?}"
    );
    let rest = &s[5..];
    ensure!(!rest.is_empty(), "at-uri must have an authority: {s:?}");
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "at-uri must not contain whitespace: {s:?}"
    );
    Ok(())
}

/// A DID string — must start with `did:` and carry a non-empty, whitespace-free
/// method-specific identifier. Both `did:web:…` and `did:key:…` are accepted.
pub fn validate_did(s: &str) -> Result<()> {
    ensure!(s.starts_with("did:"), "did must start with \"did:\": {s:?}");
    let rest = &s[4..];
    ensure!(!rest.is_empty(), "did must have a method: {s:?}");
    ensure!(
        !rest.chars().any(char::is_whitespace),
        "did must not contain whitespace: {s:?}"
    );
    let mut parts = rest.splitn(2, ':');
    let method = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("");
    ensure!(
        !method.is_empty() && !id.is_empty(),
        "did must be \"did:<method>:<id>\": {s:?}"
    );
    Ok(())
}

/// `format: "datetime"` — atproto ISO-8601 UTC, millisecond precision, `Z`.
///
/// The canonical shape is exactly `YYYY-MM-DDTHH:MM:SS.mmmZ`, and the fields
/// are checked semantically (real month/day bounds incl. leap years, 24-hour
/// clock), not just for separator positions.
pub fn validate_datetime(s: &str) -> Result<()> {
    ensure!(s.ends_with('Z'), "datetime must end with 'Z' (UTC): {s:?}");
    let pre = &s[..s.len() - 1];
    let bytes = pre.as_bytes();
    ensure!(
        bytes.len() == 23,
        "datetime must be YYYY-MM-DDTHH:MM:SS.mmm (millisecond precision): {s:?}"
    );
    ensure!(
        bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes[10] == b'T'
            && bytes[13] == b':'
            && bytes[16] == b':'
            && bytes[19] == b'.',
        "datetime must be ISO-8601 (YYYY-MM-DDTHH:MM:SS.mmm): {s:?}"
    );
    ensure!(
        bytes
            .iter()
            .enumerate()
            .all(|(i, b)| matches!(i, 4 | 7 | 10 | 13 | 16 | 19) || b.is_ascii_digit()),
        "datetime must be all digits outside its separators: {s:?}"
    );
    // Every field is all-digit and <= 4 chars, so parse cannot fail — but stay
    // panic-free anyway (u32::MAX falls out of every range below).
    let num = |lo: usize, hi: usize| pre[lo..hi].parse::<u32>().unwrap_or(u32::MAX);
    let (year, month, day) = (num(0, 4), num(5, 7), num(8, 10));
    let (hour, minute, second) = (num(11, 13), num(14, 16), num(17, 19));
    ensure!(
        (1..=12).contains(&month),
        "datetime month out of range: {s:?}"
    );
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 31,
    };
    ensure!(
        (1..=days_in_month).contains(&day),
        "datetime day out of range: {s:?}"
    );
    ensure!(
        hour <= 23 && minute <= 59 && second <= 59,
        "datetime time out of range: {s:?}"
    );
    Ok(())
}

/// A required free-form string that must not be empty (`field` names it for the
/// error message).
pub fn validate_nonempty(s: &str, field: &str) -> Result<()> {
    ensure!(!s.is_empty(), "{field} must not be empty");
    Ok(())
}

// ── ListItemRecord: group/subject/createdAt, identical across every list lexicon ──

/// An atproto *listitem* record: `{ group: at-uri, subject: did, createdAt:
/// datetime }`. The owner's claim that `subject` belongs to `group`; bidirectional
/// consent (the subject's own record separately consenting) is a fact about
/// the wider system, not this record — see [`check_bidirectional_consent`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListItemRecord {
    /// at-uri of the owning list (group) record.
    pub group: String,
    /// The DID of the member being claimed.
    pub subject: String,
    /// ISO-8601 UTC datetime.
    pub created_at: String,
}

impl ListItemRecord {
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
        Ok(Self {
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

    pub fn to_value(&self) -> DagCbor {
        DagCbor::str_map([
            ("createdAt", DagCbor::Text(self.created_at.clone())),
            ("group", DagCbor::Text(self.group.clone())),
            ("subject", DagCbor::Text(self.subject.clone())),
        ])
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    /// `nsid` attributes error messages to the calling lexicon.
    pub fn from_dag_cbor(bytes: &[u8], nsid: &str) -> Result<Self> {
        Self::from_value(&DagCbor::decode(bytes)?, nsid)
    }

    pub fn from_value(value: &DagCbor, nsid: &str) -> Result<Self> {
        let map = value.as_map()?;
        let group = map_get_str(value, "group", nsid)?.to_owned();
        let subject = map_get_str(value, "subject", nsid)?.to_owned();
        let created_at = map_get_str(value, "createdAt", nsid)?.to_owned();
        for (k, _v) in map {
            match k.as_str()? {
                "group" | "subject" | "createdAt" => {}
                other => bail!("{nsid}: unknown field {other:?} (lexicon is 3 fields)"),
            }
        }
        Self::new(group, subject, created_at)
    }
}

/// Generic bidirectional-consent check: the owner's claim (a fetched
/// [`ListItemRecord`]) names `subject` as a member of `group_uri` AND the
/// member's own consent set (an already-resolved list of group at-uris from
/// whatever member-side lexicon — caller's responsibility) includes `group_uri`.
///
/// Authz-prefiltering (whether the *caller* may even query this group) is a
/// separate, additional check owned by the caller — this function only
/// establishes the membership fact, not the right to act on it.
pub fn check_bidirectional_consent(
    owner_claim: &ListItemRecord,
    group_uri: &str,
    member_did: &str,
    member_consented_groups: &[String],
) -> bool {
    owner_claim.group == group_uri
        && owner_claim.subject == member_did
        && member_consented_groups.iter().any(|g| g == group_uri)
}

// ── ListRecord<E>: name/ownerDid/purpose?/createdAt + lexicon-specific extra ──

/// A list record's lexicon-specific extra fields beyond the common
/// `name`/`ownerDid`/`purpose`/`createdAt` shape (e.g. `keysetId` for event
/// groups; none — `()` — for placement groups).
pub trait ListExtra: Clone + std::fmt::Debug + Eq + Sized {
    /// Field names this payload owns, for the unknown-field rejection. Must not
    /// overlap `name`/`ownerDid`/`purpose`/`createdAt`.
    fn field_names() -> &'static [&'static str];
    /// Append this payload's k/v pairs into the (not-yet-sorted) field list.
    fn encode_into(&self, pairs: &mut Vec<(&'static str, DagCbor)>);
    /// Parse this payload's fields out of an already-decoded map. The common
    /// fields and the unknown-field check are handled by [`ListRecord::from_value`].
    fn decode_from(value: &DagCbor, nsid: &str) -> Result<Self>;
    /// Validate an already-constructed payload. Called from both
    /// [`ListRecord::new`] (construction) and [`ListRecord::from_value`] (via
    /// `decode_from`, which should itself validate) — so a record built directly
    /// in code and one round-tripped through DAG-CBOR enforce the identical
    /// invariants. Default: no extra invariants.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// No extra fields (e.g. `ai.hyprstream.placement.group`).
impl ListExtra for () {
    fn field_names() -> &'static [&'static str] {
        &[]
    }
    fn encode_into(&self, _pairs: &mut Vec<(&'static str, DagCbor)>) {}
    fn decode_from(_value: &DagCbor, _nsid: &str) -> Result<Self> {
        Ok(())
    }
}

/// An atproto *list* record: `{ name, ownerDid, purpose?, createdAt }` plus
/// lexicon-specific extra fields `E`. Mirrors `app.bsky.graph.list`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ListRecord<E: ListExtra> {
    /// Human-readable group name.
    pub name: String,
    /// The DID that owns the group.
    pub owner_did: String,
    /// Optional free-form description.
    pub purpose: Option<String>,
    /// ISO-8601 UTC datetime.
    pub created_at: String,
    /// Lexicon-specific extra fields.
    pub extra: E,
}

impl<E: ListExtra> ListRecord<E> {
    /// Validate and construct a record.
    pub fn new(
        name: impl Into<String>,
        owner_did: impl Into<String>,
        purpose: Option<String>,
        created_at: impl Into<String>,
        extra: E,
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
        extra.validate()?;
        Ok(Self {
            name,
            owner_did,
            purpose,
            created_at,
            extra,
        })
    }

    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    pub fn to_value(&self) -> DagCbor {
        let mut pairs: Vec<(&str, DagCbor)> = vec![
            ("name", DagCbor::Text(self.name.clone())),
            ("ownerDid", DagCbor::Text(self.owner_did.clone())),
            ("createdAt", DagCbor::Text(self.created_at.clone())),
        ];
        if let Some(purpose) = &self.purpose {
            pairs.push(("purpose", DagCbor::Text(purpose.clone())));
        }
        self.extra.encode_into(&mut pairs);
        DagCbor::str_map(pairs)
    }

    /// Decode canonical DAG-CBOR bytes into a record, validating all fields.
    /// `nsid` attributes error messages to the calling lexicon.
    pub fn from_dag_cbor(bytes: &[u8], nsid: &str) -> Result<Self> {
        Self::from_value(&DagCbor::decode(bytes)?, nsid)
    }

    pub fn from_value(value: &DagCbor, nsid: &str) -> Result<Self> {
        let map = value.as_map()?;
        let name = map_get_str(value, "name", nsid)?.to_owned();
        let owner_did = map_get_str(value, "ownerDid", nsid)?.to_owned();
        let purpose = map_get_opt_str(value, "purpose")?.map(str::to_owned);
        let created_at = map_get_str(value, "createdAt", nsid)?.to_owned();
        let extra = E::decode_from(value, nsid)?;
        let known_extra = E::field_names();
        for (k, _v) in map {
            let key = k.as_str()?;
            match key {
                "name" | "ownerDid" | "purpose" | "createdAt" => {}
                other if known_extra.contains(&other) => {}
                other => bail!("{nsid}: unknown field {other:?}"),
            }
        }
        Self::new(name, owner_did, purpose, created_at, extra)
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

    const NSID: &str = "ai.hyprstream.test.group";
    const ITEM_NSID: &str = "ai.hyprstream.test.groupItem";

    fn sample_list() -> ListRecord<()> {
        ListRecord::new(
            "east-coast-gpus",
            "did:web:alice.example.com",
            Some("GPU nodes in us-east".into()),
            "2026-06-23T12:34:56.789Z",
            (),
        )
        .expect("valid sample")
    }

    fn sample_item() -> ListItemRecord {
        ListItemRecord::new(
            "at://did:web:alice.example.com/ai.hyprstream.test.group/3kxy",
            "did:web:node1.example.com",
            "2026-06-23T12:34:56.789Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn list_round_trip_same_cid_no_extra() {
        let r = sample_list();
        let bytes = r.to_dag_cbor();
        let back = ListRecord::<()>::from_dag_cbor(&bytes, NSID).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
    }

    #[test]
    fn list_optional_purpose_omitted() {
        let mut r = sample_list();
        r.purpose = None;
        let v = r.to_value();
        assert!(v.get("purpose").is_none(), "absent purpose must be omitted");
        let back = ListRecord::<()>::from_dag_cbor(&r.to_dag_cbor(), NSID).expect("round-trip");
        assert_eq!(r, back);
    }

    #[test]
    fn list_fields_canonical_order_no_extra() {
        let r = sample_list();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        assert_eq!(keys, vec!["createdAt", "name", "ownerDid", "purpose"]);
    }

    #[test]
    fn list_rejects_extra_fields() {
        let mut extra = sample_list().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("bogus".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(ListRecord::<()>::from_value(&extra, NSID).is_err());
    }

    #[test]
    fn list_validates_formats() {
        assert!(ListRecord::new("", "did:web:x", None, "2026-06-23T12:34:56.789Z", ()).is_err());
        assert!(ListRecord::new("g", "not-a-did", None, "2026-06-23T12:34:56.789Z", ()).is_err());
        assert!(ListRecord::new(
            "g",
            "did:web:x",
            Some("".into()),
            "2026-06-23T12:34:56.789Z",
            ()
        )
        .is_err());
        assert!(ListRecord::new("g", "did:web:x", None, "2026", ()).is_err());
    }

    #[test]
    fn datetime_is_validated_semantically() {
        // Valid millisecond-precision UTC datetimes are accepted.
        for ok in [
            "2026-06-23T12:34:56.789Z",
            "2024-02-29T00:00:00.000Z", // leap day in a leap year
            "2026-12-31T23:59:59.999Z",
        ] {
            assert!(validate_datetime(ok).is_ok(), "{ok:?} must be accepted");
        }
        for bad in [
            "2026-99-99T99:99:99.999Z",  // nonsense month/day/time
            "2026-00-10T12:00:00.000Z",  // month 0
            "2026-13-10T12:00:00.000Z",  // month 13
            "2026-04-31T12:00:00.000Z",  // April has 30 days
            "2026-02-29T12:00:00.000Z",  // not a leap year
            "2026-06-23T24:00:00.000Z",  // hour 24
            "2026-06-23T12:60:00.000Z",  // minute 60
            "2026-06-23T12:34:60.000Z",  // second 60
            "2026-06-23T12:34:56.Z",     // missing fractional digits
            "2026-06-23T12:34:56.78Z",   // not millisecond precision
            "2026-06-23T12:34:56.7890Z", // too many fractional digits
            "2026-06-23T12:34:56.789",   // missing Z
            "aaaa-aa-aaTaa:aa:aa.aaaZ",  // separators only, no digits
        ] {
            assert!(validate_datetime(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn item_round_trip_same_cid() {
        let r = sample_item();
        let bytes = r.to_dag_cbor();
        let back = ListItemRecord::from_dag_cbor(&bytes, ITEM_NSID).expect("round-trip");
        assert_eq!(r, back);
        assert_eq!(r.cid(), back.cid());
    }

    #[test]
    fn item_fields_canonical_order() {
        let r = sample_item();
        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        assert_eq!(keys, vec!["createdAt", "group", "subject"]);
    }

    #[test]
    fn item_rejects_extra_fields() {
        let mut extra = sample_item().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("bogus".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(ListItemRecord::from_value(&extra, ITEM_NSID).is_err());
    }

    #[test]
    fn item_validates_formats() {
        assert!(ListItemRecord::new("nope", "did:web:n", "2026-06-23T12:34:56.789Z").is_err());
        assert!(ListItemRecord::new(
            "at://did:web:x/ai.hyprstream.test.group/3kxy",
            "not-a-did",
            "2026-06-23T12:34:56.789Z"
        )
        .is_err());
    }

    #[test]
    fn bidirectional_consent_requires_both_sides() {
        let item = sample_item();
        let group_uri = item.group.clone();
        let member = item.subject.clone();

        assert!(!check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            &[]
        ));
        assert!(!check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            &["at://did:web:other/ai.hyprstream.test.group/zzzz".to_owned()],
        ));
        assert!(check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            std::slice::from_ref(&group_uri),
        ));
    }

    // A two-field extra payload, to exercise the generic path end-to-end.
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestExtra {
        keyset_id: String,
    }
    impl ListExtra for TestExtra {
        fn field_names() -> &'static [&'static str] {
            &["keysetId"]
        }
        fn encode_into(&self, pairs: &mut Vec<(&'static str, DagCbor)>) {
            pairs.push(("keysetId", DagCbor::Text(self.keyset_id.clone())));
        }
        fn decode_from(value: &DagCbor, nsid: &str) -> Result<Self> {
            Ok(Self {
                keyset_id: map_get_str(value, "keysetId", nsid)?.to_owned(),
            })
        }
    }

    #[test]
    fn list_with_extra_round_trips_and_orders_canonically() {
        let r = ListRecord::new(
            "qwen3-serving",
            "did:web:nodeA.example.com",
            None,
            "2026-06-30T12:00:00.000Z",
            TestExtra {
                keyset_id: "ks-1".into(),
            },
        )
        .expect("valid");
        let back = ListRecord::<TestExtra>::from_dag_cbor(&r.to_dag_cbor(), NSID).expect("decode");
        assert_eq!(r, back);

        let v = DagCbor::decode(&r.to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // lexicographic byte order across base + extra fields.
        assert_eq!(keys, vec!["createdAt", "keysetId", "name", "ownerDid"]);
    }

    #[test]
    fn list_with_extra_rejects_unknown_field_outside_both_sets() {
        let r = ListRecord::new(
            "g",
            "did:web:x",
            None,
            "2026-06-30T12:00:00.000Z",
            TestExtra {
                keyset_id: "ks-1".into(),
            },
        )
        .expect("valid");
        let mut v = r.to_value();
        if let DagCbor::Map(ref mut m) = v {
            m.push((DagCbor::Text("bogus".into()), DagCbor::Unsigned(1)));
            m.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(ListRecord::<TestExtra>::from_value(&v, NSID).is_err());
    }
}
