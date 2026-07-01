//! Canonical at-record subject + keyed group-scoped routing leaf (EV3, epic #600).
//!
//! Replaces the unkeyed 64-bit `oid_hash`/`publication_broadcast_path` (in
//! [`crate::moq_event`]) with a construction that is:
//! 1. **Canonical** — a record addressed three ways (handle vs DID, NSID case,
//!    trailing fragment) hashes to exactly one value.
//! 2. **Keyed** — the routing leaf is a PRF of `(K_group, subject, epoch)`, so
//!    knowing the at-uri is NOT enough to derive the topic (confidentiality of
//!    the subject against the relay). `K_group` is the PRF key.
//! 3. **Epoch-rotated** — rotating the group key (EV2's `GroupKeyRegistry`)
//!    rotates every leaf, giving cross-time unlinkability.
//!
//! The leaf is a MoQ-path-safe (hex) segment; the semantic topic name moves
//! UNDER the payload AEAD (see [`crate::crypto::event_crypto`]), with the AAD
//! binding the keyed leaf + epoch so cross-topic/cross-epoch ciphertext replay
//! is rejected.
//!
//! Lexicon: events address records by their atproto collection NSID. The formal
//! `ai.hyprstream.event.*` lexicon is a separate follow-up; this module is
//! NSID-agnostic (takes the NSID as a validated string).

use crate::crypto::keyed_mac;

/// Separator-free canonical encoding prefix — every field is length-prefixed so
/// two distinct subjects cannot collide on concatenation (the `build_*_aad`
/// discipline used elsewhere in the crate).
const SUBJECT_DOMAIN: &str = "hyprstream event-subject v1";

/// A canonical at-record subject: the publishing/owning DID, the collection
/// NSID, and the record key (rkey).
///
/// All three are normalized on construction:
/// - `did` must already be a DID (`did:` prefix); handle→DID resolution is the
///   caller's job (see [`resolve_handle_to_did`], a stub until the live
///   `getRecord` path lands).
/// - `nsid` is lowercased (NSIDs are ASCII reverse-DNS, case-normalized).
/// - `rkey` is taken as-is (atproto record-key syntax).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CanonicalSubject {
    pub did: String,
    pub nsid: String,
    pub rkey: String,
}

impl CanonicalSubject {
    /// Construct a canonical subject, normalizing the NSID and validating the
    /// DID + rkey are non-empty. `did` must be a DID (not a handle) — resolve
    /// handles first via [`resolve_handle_to_did`].
    pub fn new(
        did: impl Into<String>,
        nsid: impl Into<String>,
        rkey: impl Into<String>,
    ) -> Result<Self, String> {
        let did = did.into();
        let nsid = nsid.into().to_ascii_lowercase();
        let rkey = rkey.into();
        if !did.starts_with("did:") {
            return Err(format!("subject did must be a DID (did:...), got {did:?}"));
        }
        if did.chars().any(char::is_whitespace) {
            return Err("subject did must not contain whitespace".to_owned());
        }
        if nsid.is_empty() || nsid.chars().any(char::is_whitespace) {
            return Err(format!(
                "subject nsid must be non-empty/no-whitespace, got {nsid:?}"
            ));
        }
        if rkey.is_empty() || rkey.chars().any(char::is_whitespace) {
            return Err(format!(
                "subject rkey must be non-empty/no-whitespace, got {rkey:?}"
            ));
        }
        Ok(Self { did, nsid, rkey })
    }

    /// Length-prefixed canonical bytes: `domain || u32_le(len(did)) || did ||
    /// u32_le(len(nsid)) || nsid || u32_le(len(rkey)) || rkey`. Deterministic;
    /// two distinct subjects cannot produce the same bytes.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let did = self.did.as_bytes();
        let nsid = self.nsid.as_bytes();
        let rkey = self.rkey.as_bytes();
        let mut out =
            Vec::with_capacity(SUBJECT_DOMAIN.len() + 4 * 3 + did.len() + nsid.len() + rkey.len());
        out.extend_from_slice(SUBJECT_DOMAIN.as_bytes());
        for field in [did, nsid, rkey] {
            out.extend_from_slice(&(field.len() as u32).to_le_bytes());
            out.extend_from_slice(field);
        }
        out
    }
}

/// Resolve a bare handle (e.g. `alice.example.com`) to a DID.
///
/// **STUB** — atproto handles are mutable and resolve via DID discovery; the
/// live `getRecord`/DID-resolution path is not wired here yet. Callers that
/// already hold a DID should pass it directly to [`CanonicalSubject::new`]; this
/// function exists so the handle→DID seam is explicit and fail-closed (a real
/// handle is rejected, not silently treated as a DID).
pub fn resolve_handle_to_did(_handle: &str) -> Result<String, String> {
    // TODO(atproto getRecord wiring): implement handle→DID resolution.
    Err(
        "handle→DID resolution not wired — pass a DID (did:...) to CanonicalSubject::new"
            .to_owned(),
    )
}

/// The keyed routing leaf for a subject under a group key + epoch.
///
/// `topic = hex(keyed_mac(K_group, canonical_subject_bytes ‖ epoch_le))` — a
/// 256-bit BLAKE3 keyed hash (HMAC-SHA256 under FIPS), hex-encoded to a 64-char
/// MoQ-path-safe segment. This is the opaque routing leaf that replaces
/// `oid_hash`; the semantic `(did, nsid, rkey)` stays under the payload AEAD.
///
/// Knowing the at-uri is NOT enough to derive this leaf — `K_group` is required
/// (relay-opaque, unguessable). A different group key or epoch produces a
/// different leaf (cross-group/cross-epoch unlinkability).
pub fn topic_leaf(k_group: &[u8; 32], subject: &CanonicalSubject, epoch: u64) -> String {
    let mut data = subject.to_canonical_bytes();
    data.extend_from_slice(&epoch.to_le_bytes());
    let mac = keyed_mac(k_group, &data);
    hex::encode(mac)
}

/// Full broadcast path: `{prefix}/{topic_leaf}`. Defaults to the publications
/// prefix used by the per-OID firehose tier; pass a group-scoped prefix for the
/// confidential profile.
pub fn keyed_broadcast_path(k_group: &[u8; 32], subject: &CanonicalSubject, epoch: u64) -> String {
    keyed_broadcast_path_with(
        crate::moq_event::PUBLICATIONS_PREFIX,
        k_group,
        subject,
        epoch,
    )
}

/// `keyed_broadcast_path` with an explicit prefix (e.g. a group-scoped path for
/// the confidential profile, vs the public firehose prefix).
pub fn keyed_broadcast_path_with(
    prefix: &str,
    k_group: &[u8; 32],
    subject: &CanonicalSubject,
    epoch: u64,
) -> String {
    format!("{prefix}/{}", topic_leaf(k_group, subject, epoch))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sub(did: &str, nsid: &str, rkey: &str) -> CanonicalSubject {
        CanonicalSubject::new(did, nsid, rkey).unwrap()
    }

    #[test]
    fn canonical_bytes_are_deterministic() {
        let a = sub("did:web:node.example", "ai.hyprstream.model", "v1");
        let b = sub("did:web:node.example", "ai.hyprstream.model", "v1");
        assert_eq!(a.to_canonical_bytes(), b.to_canonical_bytes());
    }

    #[test]
    fn nsid_is_lowercased() {
        let a = sub("did:web:n", "AI.HyprStream.Model", "v1");
        let b = sub("did:web:n", "ai.hyprstream.model", "v1");
        assert_eq!(a.to_canonical_bytes(), b.to_canonical_bytes());
    }

    #[test]
    fn distinct_subjects_have_distinct_bytes() {
        let a = sub("did:web:n", "ai.hyprstream.model", "v1");
        let b = sub("did:web:n", "ai.hyprstream.model", "v2");
        assert_ne!(a.to_canonical_bytes(), b.to_canonical_bytes());
    }

    #[test]
    fn length_prefixing_prevents_concat_collision() {
        // Without length prefixes, ("ab","c") and ("a","bc") could collide.
        let a = sub("did:web:ab", "c", "r");
        let b = sub("did:web:a", "bc", "r");
        assert_ne!(a.to_canonical_bytes(), b.to_canonical_bytes());
    }

    #[test]
    fn rejects_handle_and_garbage() {
        assert!(CanonicalSubject::new("alice.example.com", "x", "r").is_err());
        assert!(CanonicalSubject::new("did:web:n", "", "r").is_err());
        assert!(CanonicalSubject::new("did:web:n", "x", "").is_err());
        assert!(CanonicalSubject::new("did:web:n", "x y", "r").is_err());
    }

    #[test]
    fn resolve_handle_is_fail_closed_stub() {
        assert!(resolve_handle_to_did("alice.example.com").is_err());
    }

    #[test]
    fn topic_leaf_is_keyed_prf() {
        let s = sub("did:web:n", "ai.hyprstream.model", "v1");
        let k1 = [1u8; 32];
        let k2 = [2u8; 32];
        // Same subject+key+epoch → same leaf (deterministic PRF).
        assert_eq!(topic_leaf(&k1, &s, 0), topic_leaf(&k1, &s, 0));
        // Different key → different leaf.
        assert_ne!(topic_leaf(&k1, &s, 0), topic_leaf(&k2, &s, 0));
        // Different epoch → different leaf (cross-time unlinkability).
        assert_ne!(topic_leaf(&k1, &s, 0), topic_leaf(&k1, &s, 1));
        // Different subject → different leaf.
        let s2 = sub("did:web:n", "ai.hyprstream.model", "v2");
        assert_ne!(topic_leaf(&k1, &s, 0), topic_leaf(&k1, &s2, 0));
    }

    #[test]
    fn topic_leaf_is_hex_path_safe_256bit() {
        let s = sub("did:web:n", "ai.hyprstream.model", "v1");
        let leaf = topic_leaf(&[9u8; 32], &s, 7);
        // 32 bytes → 64 hex chars; only [0-9a-f].
        assert_eq!(leaf.len(), 64);
        assert!(leaf
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn keyed_broadcast_path_shape() {
        let s = sub("did:web:n", "ai.hyprstream.model", "v1");
        let p = keyed_broadcast_path(&[1u8; 32], &s, 0);
        assert!(p.starts_with("local/events/publications/"));
        assert_eq!(p.split('/').next_back().unwrap().len(), 64);
        let p2 = keyed_broadcast_path_with("local/events/g", &[1u8; 32], &s, 0);
        assert!(p2.starts_with("local/events/g/"));
    }
}
