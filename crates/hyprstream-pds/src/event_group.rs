//! `ai.hyprstream.event.group` / `ai.hyprstream.event.groupItem` — DRAFT lexicon
//! records for EventService managed-list groups (epic #600, EV2 #602).
//!
//! **DRAFT / RFC — not signed off.** This lexicon is deliberately modeled on the
//! signed-off `ai.hyprstream.placement.group` / `ai.hyprstream.placement.groupItem`
//! pattern from #524 (atproto **list**, owner-curated, with a separate listitem per
//! member; bidirectional consent = owner lists the member **and** the member's own
//! record consents). It is a **parallel, pattern-matching lexicon**, not a literal
//! reuse of `ai.hyprstream.placement.group` — that lexicon is already sign-off-frozen
//! (unknown fields are rejected at decode time, see `ai.hyprstream.model`'s
//! `record.rs`), so adding an EventService-specific field to it would require
//! reopening #524's sign-off. Whether EventService groups should literally BE
//! placement groups (one record, two consumers) or stay a parallel lexicon is an
//! **open question for the human reviewer** — see the PR description.
//!
//! # Key material is NOT in the record
//!
//! `keysetId` is a stable **opaque reference** ("this group's key material is
//! generation X"), never the key bytes. The actual `K_group` symmetric key is
//! generated fresh per rotation (exactly as `SecureEventPublisher::register_prefix`
//! does today) and delivered **per-member** via DH-wrap-on-join
//! (`derive_wrap_key`/`wrap_group_key`) — never committed to the public,
//! firehose-synced atproto repo. This matches v2's "durable facts = atproto
//! records, volatile/secret facts = ephemeral RPC" split from #524's sign-off, and
//! keeps the relay/firehose blind to `K_group` (epic #600's threat model).

use anyhow::{bail, ensure, Result};

use crate::cid::Cid;
use crate::dag_cbor::DagCbor;

/// NSID for the group (list) record.
pub const GROUP_COLLECTION_NSID: &str = "ai.hyprstream.event.group";
/// NSID for the groupItem (listitem) record.
pub const GROUP_ITEM_COLLECTION_NSID: &str = "ai.hyprstream.event.groupItem";

/// `ai.hyprstream.event.group` — the list record, lives in the owner's repo.
///
/// Confirmed-shape (DRAFT, pending sign-off):
/// ```json
/// {
///   "lexicon": 1,
///   "id": "ai.hyprstream.event.group",
///   "defs": { "main": { "type": "record", "key": "tid", "record": {
///     "type": "object",
///     "required": ["name", "ownerDid", "keysetId", "createdAt"],
///     "properties": {
///       "name":      { "type": "string" },
///       "ownerDid":  { "type": "string", "format": "did" },
///       "purpose":   { "type": "string" },
///       "keysetId":  { "type": "string" },
///       "createdAt": { "type": "string", "format": "datetime" }
///     }
/// }}}
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventGroupRecord {
    /// Human-readable group name.
    pub name: String,
    /// `did:...` of the owning repo (the group's authority).
    pub owner_did: String,
    /// Optional free-text purpose/description.
    pub purpose: Option<String>,
    /// Opaque reference to the current key-material generation. NOT a secret —
    /// see module docs. A member resolves `(group_uri, keyset_id, epoch)` to the
    /// actual `K_group` bytes via the DH-wrap-on-join path, never from this field.
    pub keyset_id: String,
    /// ISO-8601 UTC datetime (`format: "datetime"`).
    pub created_at: String,
}

impl EventGroupRecord {
    pub fn new(
        name: impl Into<String>,
        owner_did: impl Into<String>,
        purpose: Option<String>,
        keyset_id: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        let name = name.into();
        let owner_did = owner_did.into();
        let keyset_id = keyset_id.into();
        let created_at = created_at.into();
        ensure!(!name.is_empty(), "group name must be non-empty");
        validate_did(&owner_did)?;
        ensure!(!keyset_id.is_empty(), "keysetId must be non-empty");
        validate_datetime(&created_at)?;
        Ok(Self {
            name,
            owner_did,
            purpose,
            keyset_id,
            created_at,
        })
    }

    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn cid(&self) -> Cid {
        Cid::from_dag_cbor(&self.to_dag_cbor())
    }

    pub fn to_value(&self) -> DagCbor {
        let mut fields = vec![
            ("name", DagCbor::Text(self.name.clone())),
            ("ownerDid", DagCbor::Text(self.owner_did.clone())),
            ("keysetId", DagCbor::Text(self.keyset_id.clone())),
            ("createdAt", DagCbor::Text(self.created_at.clone())),
        ];
        if let Some(purpose) = &self.purpose {
            fields.push(("purpose", DagCbor::Text(purpose.clone())));
        }
        DagCbor::str_map(fields)
    }

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&DagCbor::decode(bytes)?)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let map = value.as_map()?;
        let name = map_get_str(value, "name")?.to_owned();
        let owner_did = map_get_str(value, "ownerDid")?.to_owned();
        let keyset_id = map_get_str(value, "keysetId")?.to_owned();
        let created_at = map_get_str(value, "createdAt")?.to_owned();
        let purpose = value
            .get("purpose")
            .map(|v| v.as_str().map(str::to_owned))
            .transpose()?;
        for (k, _v) in map {
            match k.as_str()? {
                "name" | "ownerDid" | "purpose" | "keysetId" | "createdAt" => {}
                other => bail!(
                    "ai.hyprstream.event.group: unknown field {other:?} (lexicon is 5 fields)"
                ),
            }
        }
        ensure!(!name.is_empty(), "group name must be non-empty");
        validate_did(&owner_did)?;
        ensure!(!keyset_id.is_empty(), "keysetId must be non-empty");
        validate_datetime(&created_at)?;
        Ok(Self {
            name,
            owner_did,
            purpose,
            keyset_id,
            created_at,
        })
    }
}

/// `ai.hyprstream.event.groupItem` — the owner-side membership claim (listitem),
/// lives in the **owner's** repo. Mirrors `app.bsky.graph.listitem` / the
/// `ai.hyprstream.placement.groupItem` pattern.
///
/// Membership is **bidirectional**: this record is the owner's claim that
/// `subject` belongs to `group`; it is NOT sufficient on its own — the member's
/// own identity record must independently consent (carry `group` in its own
/// `groups`-style field). Which lexicon carries that member-side consent for
/// EventService members is an **open question** (see module docs / PR body):
/// candidates are reusing `ai.hyprstream.placement.node.groups` for members that
/// are also placement nodes, or a new minimal `ai.hyprstream.event.member` record
/// for members that are not. This module implements the bidirectional-consent
/// *check* generically (`MembershipCheck`, below) over two already-fetched
/// claims, deliberately not committing to the member-side lexicon.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventGroupItemRecord {
    /// at-uri of the `ai.hyprstream.event.group` this item belongs to.
    pub group: String,
    /// `did:...` of the member being claimed.
    pub subject: String,
    pub created_at: String,
}

impl EventGroupItemRecord {
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

    pub fn to_dag_cbor(&self) -> Vec<u8> {
        self.to_value().encode()
    }

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

    pub fn from_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_value(&DagCbor::decode(bytes)?)
    }

    pub fn from_value(value: &DagCbor) -> Result<Self> {
        let map = value.as_map()?;
        let group = map_get_str(value, "group")?.to_owned();
        let subject = map_get_str(value, "subject")?.to_owned();
        let created_at = map_get_str(value, "createdAt")?.to_owned();
        for (k, _v) in map {
            match k.as_str()? {
                "group" | "subject" | "createdAt" => {}
                other => bail!(
                    "ai.hyprstream.event.groupItem: unknown field {other:?} (lexicon is 3 fields)"
                ),
            }
        }
        validate_at_uri(&group)?;
        validate_did(&subject)?;
        validate_datetime(&created_at)?;
        Ok(Self {
            group,
            subject,
            created_at,
        })
    }
}

/// Generic bidirectional-consent check: owner claims `subject` is a member
/// (via a fetched [`EventGroupItemRecord`]) AND the member's own consent set
/// (an already-resolved list of group at-uris from whatever member-side
/// lexicon — caller's responsibility, see module docs) includes `group`.
///
/// Authz-prefiltering (whether the *caller* may even query this group) is a
/// separate, additional check owned by EV4/DiscoveryService — this function
/// only establishes the membership fact, not the right to act on it.
pub fn check_bidirectional_consent(
    owner_claim: &EventGroupItemRecord,
    group_uri: &str,
    member_did: &str,
    member_consented_groups: &[String],
) -> bool {
    owner_claim.group == group_uri
        && owner_claim.subject == member_did
        && member_consented_groups.iter().any(|g| g == group_uri)
}

// ── field validators (mirrors record.rs's pattern; duplicated rather than
// refactored to keep this draft's diff focused — a shared validator module is
// a natural follow-up if this lexicon is adopted) ──────────────────────────

fn map_get_str<'a>(value: &'a DagCbor, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing required field {key:?}"))?
        .as_str()
}

fn validate_at_uri(s: &str) -> Result<()> {
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

fn validate_did(s: &str) -> Result<()> {
    ensure!(s.starts_with("did:"), "did must start with \"did:\": {s:?}");
    ensure!(
        !s[4..].is_empty() && !s.chars().any(char::is_whitespace),
        "did must have a non-empty, whitespace-free method-specific-id: {s:?}"
    );
    Ok(())
}

fn validate_datetime(s: &str) -> Result<()> {
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
    ensure!(
        bytes.get(19) == Some(&b'.'),
        "datetime must have millisecond precision (.mmm): {s:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;

    fn sample_group() -> EventGroupRecord {
        EventGroupRecord::new(
            "qwen3-serving",
            "did:web:nodeA.example.com",
            Some("serve:model:qwen3 subscribers".to_owned()),
            "ks-2026-06-30-01",
            "2026-06-30T12:00:00.000Z",
        )
        .expect("valid sample")
    }

    fn sample_item() -> EventGroupItemRecord {
        EventGroupItemRecord::new(
            "at://did:web:nodeA.example.com/ai.hyprstream.event.group/3k2x",
            "did:web:memberB.example.com",
            "2026-06-30T12:01:00.000Z",
        )
        .expect("valid sample")
    }

    #[test]
    fn group_deterministic_across_rebuild() {
        let r1 = sample_group();
        let r2 = EventGroupRecord::new(
            r1.name.clone(),
            r1.owner_did.clone(),
            r1.purpose.clone(),
            r1.keyset_id.clone(),
            r1.created_at.clone(),
        )
        .unwrap();
        assert_eq!(r1.to_dag_cbor(), r2.to_dag_cbor());
        assert_eq!(r1.cid(), r2.cid());
    }

    #[test]
    fn group_round_trips_through_dag_cbor() {
        let r = sample_group();
        let bytes = r.to_dag_cbor();
        let decoded = EventGroupRecord::from_dag_cbor(&bytes).expect("decode");
        assert_eq!(r, decoded);
    }

    #[test]
    fn group_round_trips_without_optional_purpose() {
        let r = EventGroupRecord::new(
            "no-purpose-group",
            "did:web:nodeA.example.com",
            None,
            "ks-1",
            "2026-06-30T12:00:00.000Z",
        )
        .unwrap();
        let decoded = EventGroupRecord::from_dag_cbor(&r.to_dag_cbor()).expect("decode");
        assert_eq!(r, decoded);
        assert!(decoded.purpose.is_none());
    }

    #[test]
    fn group_rejects_extra_fields() {
        let mut extra = sample_group().to_value();
        if let DagCbor::Map(ref mut v) = extra {
            v.push((DagCbor::Text("extra".into()), DagCbor::Unsigned(1)));
            v.sort_by(|a, b| a.0.as_str().unwrap_or("").cmp(b.0.as_str().unwrap_or("")));
        }
        assert!(EventGroupRecord::from_value(&extra).is_err());
    }

    #[test]
    fn group_rejects_bad_did_and_keyset() {
        assert!(
            EventGroupRecord::new("g", "not-a-did", None, "ks-1", "2026-06-30T12:00:00.000Z")
                .is_err()
        );
        assert!(
            EventGroupRecord::new("g", "did:web:x", None, "", "2026-06-30T12:00:00.000Z").is_err()
        );
    }

    #[test]
    fn group_item_round_trips() {
        let item = sample_item();
        let decoded = EventGroupItemRecord::from_dag_cbor(&item.to_dag_cbor()).expect("decode");
        assert_eq!(item, decoded);
    }

    #[test]
    fn group_item_rejects_bad_at_uri_and_did() {
        assert!(EventGroupItemRecord::new(
            "not-an-at-uri",
            "did:web:x",
            "2026-06-30T12:00:00.000Z"
        )
        .is_err());
        assert!(EventGroupItemRecord::new(
            "at://did:web:x/ai.hyprstream.event.group/abc",
            "not-a-did",
            "2026-06-30T12:00:00.000Z"
        )
        .is_err());
    }

    #[test]
    fn bidirectional_consent_requires_both_sides() {
        let item = sample_item();
        let group_uri = item.group.clone();
        let member = item.subject.clone();

        // Owner claims member, member has NOT consented -> false.
        assert!(!check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            &[]
        ));

        // Owner claims member, member consents to a DIFFERENT group -> false.
        assert!(!check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            &["at://did:web:other/ai.hyprstream.event.group/zzzz".to_owned()],
        ));

        // Both sides agree -> true.
        assert!(check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            std::slice::from_ref(&group_uri),
        ));

        // Member consents but owner claim is for a different group -> false
        // (caller passed the wrong group_uri / owner claim mismatch).
        assert!(!check_bidirectional_consent(
            &item,
            "at://did:web:nodeA.example.com/ai.hyprstream.event.group/different",
            &member,
            &[group_uri],
        ));
    }
}
