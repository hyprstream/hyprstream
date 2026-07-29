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

fn digest(parts: &[&[u8]]) -> ReplayKey {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

/// Digest the exact raw key previously used for a DPoP proof JTI.
pub fn dpop_jti(jti: &str) -> ReplayKey {
    digest(&[jti.as_bytes()])
}

/// Digest the exact raw key previously used for a one-use mount ticket.
pub fn mount_ticket_jti(jti: &str) -> ReplayKey {
    digest(&[b"mount-ticket:", jti.as_bytes()])
}

/// Digest the exact `{client_id}\x1f{jti}` client-assertion key.
pub fn client_assertion_jti(client_id: &str, jti: &str) -> ReplayKey {
    digest(&[client_id.as_bytes(), b"\x1f", jti.as_bytes()])
}

/// Digest the exact `{issuer}\x1f{jti}` ATProto service-assertion key.
pub fn atproto_service_assertion_jti(issuer: &str, jti: &str) -> ReplayKey {
    digest(&[issuer.as_bytes(), b"\x1f", jti.as_bytes()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_the_previous_raw_key_encoding() {
        assert_eq!(std::mem::size_of::<ReplayKey>(), 32);
        assert_eq!(REPLAY_BARRIER_ENTRY_BYTES, 128);
        assert_eq!(dpop_jti("jti"), *blake3::hash(b"jti").as_bytes());
        assert_eq!(
            mount_ticket_jti("jti"),
            *blake3::hash(b"mount-ticket:jti").as_bytes()
        );
        assert_eq!(
            client_assertion_jti("client", "jti"),
            *blake3::hash(b"client\x1fjti").as_bytes()
        );
        assert_eq!(
            atproto_service_assertion_jti("did:plc:issuer", "jti"),
            *blake3::hash(b"did:plc:issuer\x1fjti").as_bytes()
        );
    }
}
