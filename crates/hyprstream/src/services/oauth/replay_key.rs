//! Fixed-size keys for OAuth replay barriers.
//!
//! A replay barrier keeps one map entry and one heap node for each live key.
//! On the supported 64-bit targets, a `[u8; 32]` key plus `Entry<()>` is 56
//! bytes and a heap node is 56 bytes. `HashMap` needs at most eight buckets
//! for seven live entries, so the map contributes at most 64 bytes per live
//! key before allocation growth; together that is 120 bytes. Capacity planning
//! rounds this to 128 bytes per live key before allocator slack. Unlike a
//! `String`, every retained part of this key is fixed-size.
//!
//! Replay identifiers are controlled by remote callers. Keeping their raw
//! strings in a bounded entry-count cache would still leave retained memory
//! unbounded, and `TtlCache` stores each key in both its map and expiry heap.
//! A BLAKE3-256 digest makes retained key storage exactly 32 bytes per copy.

/// BLAKE3-256 replay-barrier key.
pub type ReplayKey = [u8; 32];

/// Conservative per-live-entry planning size for a no-refresh replay barrier.
///
/// See this module's documentation for the 64-bit layout derivation. The
/// cache's geometric allocations make its exact resident size a fixed function
/// of its entry cap, rather than an exact multiple of this number.
pub const REPLAY_BARRIER_ENTRY_BYTES: usize = 128;

fn digest(domain: &[u8], components: &[&[u8]]) -> ReplayKey {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"hyprstream:replay-key:v1\0");
    hasher.update(&(domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    for component in components {
        hasher.update(&(component.len() as u64).to_be_bytes());
        hasher.update(component);
    }
    *hasher.finalize().as_bytes()
}

/// Digest a DPoP proof JTI in its own replay domain.
pub fn dpop_jti(jti: &str) -> ReplayKey {
    digest(b"dpop", &[jti.as_bytes()])
}

/// Digest a one-use mount ticket JTI in its own replay domain.
pub fn mount_ticket_jti(jti: &str) -> ReplayKey {
    digest(b"mount-ticket", &[jti.as_bytes()])
}

/// Digest a length-framed client-ID/JTI tuple in the client-assertion domain.
pub fn client_assertion_jti(client_id: &str, jti: &str) -> ReplayKey {
    digest(b"client-assertion", &[client_id.as_bytes(), jti.as_bytes()])
}

/// Digest a length-framed issuer/JTI tuple in the ATProto assertion domain.
pub fn atproto_service_assertion_jti(issuer: &str, jti: &str) -> ReplayKey {
    digest(b"atproto-service-assertion", &[issuer.as_bytes(), jti.as_bytes()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_fixed_size_domain_separated_and_tuple_framed() {
        assert_eq!(std::mem::size_of::<ReplayKey>(), 32);
        assert_eq!(REPLAY_BARRIER_ENTRY_BYTES, 128);
        assert_ne!(dpop_jti("mount-ticket:x"), mount_ticket_jti("x"));
        assert_ne!(dpop_jti("same"), mount_ticket_jti("same"));
        assert_ne!(
            client_assertion_jti("a", "b\x1fc"),
            client_assertion_jti("a\x1fb", "c")
        );
        assert_ne!(
            atproto_service_assertion_jti("a", "b\x1fc"),
            atproto_service_assertion_jti("a\x1fb", "c")
        );
    }
}
