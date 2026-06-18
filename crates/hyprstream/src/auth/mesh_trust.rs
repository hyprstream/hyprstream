//! Admin-anchored mesh post-quantum trust store construction (#157, Option A).
//!
//! Builds the process-global kid-anchored [`KeyedPqTrustStore`] eagerly from the
//! operator-configured `mesh_peers` (see [`crate::config::MeshPeerConfig`]). The
//! store is admin-anchored and immutable after construction: only ML-DSA-65 keys
//! an operator configured **out-of-band** are trusted, matching the
//! `KeyedPqTrustStore` contract ("Entries MUST be established out-of-band").
//!
//! Each peer entry carries two inline `Multikey` (`publicKeyMultibase`) strings,
//! copied from that peer's published DID document:
//!   - `#mesh`    → Ed25519 mesh signer key (multicodec `ed25519-pub`, `0xed01`)
//!   - `#mesh-pq` → ML-DSA-65 mesh verifying key (multicodec `ml-dsa-65-pub`, `0x1211`)
//!
//! The Ed25519 key is the kid anchor; the ML-DSA-65 key is the trusted PQ key
//! bound to it. Empty/absent `mesh_peers` yields an empty store — unchanged
//! behavior (Hybrid fails closed for unknown peers).

use hyprstream_rpc::envelope::KeyedPqTrustStore;

use crate::config::OAuthConfig;

/// Multicodec `ed25519-pub` unsigned-varint prefix (`0xed01` → bytes `0xed 0x01`).
const MULTICODEC_ED25519_PUB: [u8; 2] = [0xed, 0x01];
/// Multicodec `ml-dsa-65-pub` unsigned-varint prefix (`0x1211` → bytes `0x91 0x24`).
const MULTICODEC_ML_DSA_65_PUB: [u8; 2] = [0x91, 0x24];

/// Decode a `Multikey` `publicKeyMultibase` string into the raw key bytes,
/// verifying the multibase prefix (`z`, base58btc) and the expected multicodec
/// header. Returns the payload with the multicodec prefix stripped.
pub fn decode_multikey(multibase: &str, expected_codec: &[u8; 2]) -> anyhow::Result<Vec<u8>> {
    let body = multibase
        .strip_prefix('z')
        .ok_or_else(|| anyhow::anyhow!("Multikey must use base58btc multibase ('z') prefix"))?;
    let decoded = bs58::decode(body)
        .into_vec()
        .map_err(|e| anyhow::anyhow!("invalid base58btc Multikey: {e}"))?;
    if decoded.len() < 2 || &decoded[..2] != expected_codec {
        anyhow::bail!(
            "unexpected multicodec prefix (expected {expected_codec:02x?}, got {:02x?})",
            decoded.get(..2).unwrap_or(&decoded)
        );
    }
    Ok(decoded[2..].to_vec())
}

/// Build the kid-anchored ML-DSA-65 trust store from the admin-configured
/// `mesh_peers` (#157, Option A — eager, admin-anchored, immutable).
///
/// An invalid entry is logged and skipped (fail-safe: a malformed peer key must
/// not silently trust the wrong identity). An empty/absent `mesh_peers` yields
/// an empty store (non-breaking default).
pub fn build_mesh_pq_trust_store(oauth: &OAuthConfig) -> KeyedPqTrustStore {
    let mut store = KeyedPqTrustStore::new();
    for (label, peer) in &oauth.mesh_peers {
        let ed_bytes = match decode_multikey(&peer.ed25519_multibase, &MULTICODEC_ED25519_PUB) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid ed25519_multibase, skipping: {e}");
                continue;
            }
        };
        let ed_pubkey: [u8; 32] = match ed_bytes.as_slice().try_into() {
            Ok(a) => a,
            Err(_) => {
                tracing::error!(
                    "mesh_peer '{label}': ed25519 key is {} bytes (expected 32), skipping",
                    ed_bytes.len()
                );
                continue;
            }
        };
        let pq_bytes = match decode_multikey(&peer.mldsa65_multibase, &MULTICODEC_ML_DSA_65_PUB) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid mldsa65_multibase, skipping: {e}");
                continue;
            }
        };
        let pq_vk = match hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pq_bytes) {
            Ok(vk) => vk,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid ML-DSA-65 verifying key, skipping: {e}");
                continue;
            }
        };
        store.bind(ed_pubkey, &pq_vk);
        tracing::info!("mesh_peer '{label}': anchored ML-DSA-65 key for Ed25519 signer identity");
    }
    store
}

/// Encode raw key bytes as a `Multikey` `publicKeyMultibase` string (base58btc,
/// multicodec-prefixed). Inverse of [`decode_multikey`]; used by tests and by
/// operators generating peer entries.
pub fn encode_multikey(raw: &[u8], codec: &[u8; 2]) -> String {
    let mut payload = Vec::with_capacity(2 + raw.len());
    payload.extend_from_slice(codec);
    payload.extend_from_slice(raw);
    format!("z{}", bs58::encode(payload).into_string())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::MeshPeerConfig;
    use ed25519_dalek::SigningKey;
    use hyprstream_rpc::crypto::pq;
    use rand::rngs::OsRng;

    fn oauth_with_peers(peers: Vec<(&str, MeshPeerConfig)>) -> OAuthConfig {
        let mut oauth = OAuthConfig::default();
        for (k, v) in peers {
            oauth.mesh_peers.insert(k.to_owned(), v);
        }
        oauth
    }

    #[test]
    fn empty_mesh_peers_yields_empty_store() {
        let oauth = OAuthConfig::default();
        let store = build_mesh_pq_trust_store(&oauth);
        assert!(store.is_empty(), "absent mesh_peers must produce an empty store");
    }

    #[test]
    fn mesh_peer_entry_verifies_peer_signature() {
        // A peer's mesh identity = an Ed25519 signer key + its derived ML-DSA-65
        // mesh key. Encode both as Multikey strings (as published in the peer's
        // DID doc), build the store, and confirm it anchors the peer's ML-DSA key
        // keyed by the peer's Ed25519 signer identity.
        let peer_ed = SigningKey::generate(&mut OsRng);
        let peer_pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&peer_ed);
        let peer_pq_vk_bytes = pq::ml_dsa_sk_to_vk_bytes(&peer_pq_sk);

        let ed_mb = encode_multikey(&peer_ed.verifying_key().to_bytes(), &MULTICODEC_ED25519_PUB);
        let pq_mb = encode_multikey(&peer_pq_vk_bytes, &MULTICODEC_ML_DSA_65_PUB);

        let oauth = oauth_with_peers(vec![(
            "peer-a",
            MeshPeerConfig { ed25519_multibase: ed_mb, mldsa65_multibase: pq_mb },
        )]);
        let store = build_mesh_pq_trust_store(&oauth);
        assert_eq!(store.len(), 1, "one valid peer must produce one binding");

        // The store resolves the peer's Ed25519 signer identity to its ML-DSA vk.
        use hyprstream_rpc::envelope::PqTrustStore;
        let resolved = store
            .ml_dsa_key_for(&peer_ed.verifying_key().to_bytes())
            .expect("peer's ML-DSA key must be anchored");

        // A signature from the peer's ML-DSA key verifies under the anchored key.
        let msg = b"mesh peer attestation payload";
        let sig = pq::ml_dsa_sign(&peer_pq_sk, msg);
        pq::ml_dsa_verify(&resolved, msg, &sig).expect("peer signature must verify");

        // An unknown signer identity is not anchored.
        assert!(store.ml_dsa_key_for(&[0u8; 32]).is_none());
    }

    #[test]
    fn malformed_peer_entry_is_skipped_not_trusted() {
        // A bad ed25519 multibase must be skipped (fail-safe), leaving the store
        // empty rather than trusting a wrong/garbage identity.
        let good_pq = encode_multikey(&vec![0u8; 1952], &MULTICODEC_ML_DSA_65_PUB);
        let oauth = oauth_with_peers(vec![(
            "bad-peer",
            MeshPeerConfig {
                ed25519_multibase: "not-multibase".to_owned(),
                mldsa65_multibase: good_pq,
            },
        )]);
        let store = build_mesh_pq_trust_store(&oauth);
        assert!(store.is_empty(), "malformed entry must be skipped, not trusted");
    }

    #[test]
    fn decode_multikey_rejects_wrong_codec() {
        // An ed25519 Multikey decoded as ml-dsa must be rejected on the codec.
        let ed_mb = encode_multikey(&[7u8; 32], &MULTICODEC_ED25519_PUB);
        assert!(decode_multikey(&ed_mb, &MULTICODEC_ML_DSA_65_PUB).is_err());
        // Round-trips with the correct codec.
        let raw = decode_multikey(&ed_mb, &MULTICODEC_ED25519_PUB).unwrap();
        assert_eq!(raw, [7u8; 32]);
    }
}
