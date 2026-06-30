//! `ai.hyprstream.event.group` / `ai.hyprstream.event.groupItem` — lexicon
//! records for EventService managed-list groups (epic #600, EV2 #602).
//!
//! Built on the generic [`crate::list_record`] module: a **parallel,
//! pattern-matching lexicon**, deliberately modeled on the signed-off
//! `ai.hyprstream.placement.group`/`groupItem` pattern from #524 (atproto
//! **list**, owner-curated, with a separate listitem per member; bidirectional
//! consent = owner lists the member **and** the member's own record consents) —
//! but NOT a literal reuse of `ai.hyprstream.placement.group`, since that
//! lexicon is sign-off-frozen (unknown fields rejected at decode) and event
//! groups need an extra `keysetId` field placement groups don't have.
//!
//! **Resolved design decision** (human review, #602): keep these as separate
//! NSIDs/collections — this preserves cheap collection-based firehose/AppView
//! filtering and keeps placement-membership and event-key-membership as
//! genuinely distinct axes — while sharing the list/listitem/consent
//! *mechanics* via [`crate::list_record`] so neither lexicon hand-duplicates
//! validation, encoding, or the consent check.
//!
//! # Key material is NOT in the record
//!
//! `keysetId` is a stable **opaque reference** ("this group's key material is
//! generation X"), never the key bytes. The actual `K_group` symmetric key is
//! generated fresh per rotation (exactly as `SecureEventPublisher::register_prefix`
//! does today) and delivered **per-member** via DH-wrap-on-join
//! (`derive_wrap_key`/`wrap_group_key`) — never committed to the public,
//! firehose-synced atproto repo. This matches #524's "durable facts = atproto
//! records, volatile/secret facts = ephemeral RPC" split, and keeps the
//! relay/firehose blind to `K_group` (epic #600's threat model).
//!
//! # Member-side consent lexicon — still open
//!
//! Membership is **bidirectional**: [`EventGroupItemRecord`] is the owner's
//! claim that a member belongs to the group; it is NOT sufficient alone — the
//! member's own identity record must independently consent (carry the group's
//! at-uri in its own `groups`-style field). Which lexicon carries that
//! member-side consent for EventService members remains an **open question**:
//! candidates are reusing `ai.hyprstream.placement.node.groups` for members
//! that are also placement nodes, or a new minimal `ai.hyprstream.event.member`
//! record for members that are not. [`check_bidirectional_consent`] implements
//! the check generically over two already-fetched claims, deliberately not
//! committing to the member-side lexicon.

use anyhow::{ensure, Result};

use crate::dag_cbor::DagCbor;
use crate::list_record::{self, ListExtra, ListItemRecord, ListRecord};

/// NSID for the group (list) record.
pub const GROUP_COLLECTION_NSID: &str = "ai.hyprstream.event.group";
/// NSID for the groupItem (listitem) record.
pub const GROUP_ITEM_COLLECTION_NSID: &str = "ai.hyprstream.event.groupItem";

/// `ai.hyprstream.event.group`'s extra field beyond the common list shape.
///
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
pub struct EventGroupExtra {
    /// Opaque reference to the current key-material generation. NOT a secret —
    /// see module docs. A member resolves `(group_uri, keyset_id, epoch)` to the
    /// actual `K_group` bytes via the DH-wrap-on-join path, never from this field.
    pub keyset_id: String,
}

impl ListExtra for EventGroupExtra {
    fn field_names() -> &'static [&'static str] {
        &["keysetId"]
    }

    fn encode_into(&self, pairs: &mut Vec<(&'static str, DagCbor)>) {
        pairs.push(("keysetId", DagCbor::Text(self.keyset_id.clone())));
    }

    fn decode_from(value: &DagCbor, nsid: &str) -> Result<Self> {
        let keyset_id = list_record::map_get_str(value, "keysetId", nsid)?.to_owned();
        Ok(Self { keyset_id })
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.keyset_id.is_empty(), "keysetId must be non-empty");
        Ok(())
    }
}

/// `ai.hyprstream.event.group` — the list record, lives in the owner's repo.
pub type EventGroupRecord = ListRecord<EventGroupExtra>;

/// `ai.hyprstream.event.groupItem` — the owner-side membership claim
/// (listitem), lives in the owner's repo. Identical shape to every other
/// listitem lexicon in this crate — see [`ListItemRecord`].
pub type EventGroupItemRecord = ListItemRecord;

/// Generic bidirectional-consent check — see [`list_record::check_bidirectional_consent`].
pub use list_record::check_bidirectional_consent;

impl EventGroupRecord {
    /// Convenience constructor matching this lexicon's field order (`keyset_id`
    /// as a plain argument rather than wrapping it in [`EventGroupExtra`] by hand).
    pub fn new_event_group(
        name: impl Into<String>,
        owner_did: impl Into<String>,
        purpose: Option<String>,
        keyset_id: impl Into<String>,
        created_at: impl Into<String>,
    ) -> Result<Self> {
        ListRecord::new(
            name,
            owner_did,
            purpose,
            created_at,
            EventGroupExtra {
                keyset_id: keyset_id.into(),
            },
        )
    }
}

impl EventGroupItemRecord {
    /// Decode using this lexicon's NSID for error attribution.
    pub fn from_event_dag_cbor(bytes: &[u8]) -> Result<Self> {
        Self::from_dag_cbor(bytes, GROUP_ITEM_COLLECTION_NSID)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;

    fn sample_group() -> EventGroupRecord {
        EventGroupRecord::new_event_group(
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
        let r2 = EventGroupRecord::new_event_group(
            r1.name.clone(),
            r1.owner_did.clone(),
            r1.purpose.clone(),
            r1.extra.keyset_id.clone(),
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
        let decoded =
            EventGroupRecord::from_dag_cbor(&bytes, GROUP_COLLECTION_NSID).expect("decode");
        assert_eq!(r, decoded);
    }

    #[test]
    fn group_round_trips_without_optional_purpose() {
        let r = EventGroupRecord::new_event_group(
            "no-purpose-group",
            "did:web:nodeA.example.com",
            None,
            "ks-1",
            "2026-06-30T12:00:00.000Z",
        )
        .unwrap();
        let decoded = EventGroupRecord::from_dag_cbor(&r.to_dag_cbor(), GROUP_COLLECTION_NSID)
            .expect("decode");
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
        assert!(EventGroupRecord::from_value(&extra, GROUP_COLLECTION_NSID).is_err());
    }

    #[test]
    fn group_rejects_bad_did_and_keyset() {
        assert!(EventGroupRecord::new_event_group(
            "g",
            "not-a-did",
            None,
            "ks-1",
            "2026-06-30T12:00:00.000Z"
        )
        .is_err());
        assert!(EventGroupRecord::new_event_group(
            "g",
            "did:web:x",
            None,
            "",
            "2026-06-30T12:00:00.000Z"
        )
        .is_err());
    }

    #[test]
    fn group_field_order_matches_lexicon_canonical_cbor() {
        let v = DagCbor::decode(&sample_group().to_dag_cbor()).expect("decode");
        let map = v.as_map().expect("map");
        let keys: Vec<&str> = map.iter().map(|(k, _)| k.as_str().expect("str")).collect();
        // lexicographic byte order across base + extra fields.
        assert_eq!(
            keys,
            vec!["createdAt", "keysetId", "name", "ownerDid", "purpose"]
        );
    }

    #[test]
    fn group_item_round_trips() {
        let item = sample_item();
        let decoded =
            EventGroupItemRecord::from_event_dag_cbor(&item.to_dag_cbor()).expect("decode");
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
            &["at://did:web:other/ai.hyprstream.event.group/zzzz".to_owned()],
        ));
        assert!(check_bidirectional_consent(
            &item,
            &group_uri,
            &member,
            std::slice::from_ref(&group_uri),
        ));
        assert!(!check_bidirectional_consent(
            &item,
            "at://did:web:nodeA.example.com/ai.hyprstream.event.group/different",
            &member,
            &[group_uri],
        ));
    }
}
