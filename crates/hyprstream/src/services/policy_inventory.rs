//! Complete, read-only public-key inventory for Federate Policy admission.
//!
//! The older request-proof manifest helper intentionally skips unavailable
//! sources. That is suitable for its legacy overlap check, but not for a
//! durable Federate session: an omitted foreign key would make a new proof
//! signer appear distinct when it is not. This loader must fail instead.
//! No production factory installs it until the scoped DB/profile and signed
//! account authority are supplied together.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_rpc::crypto::pq::{ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes, ml_dsa_vk_from_bytes};
use hyprstream_rpc::did_key::{decode_multikey, MULTICODEC_ED25519_PUB, MULTICODEC_ML_DSA_65_PUB};
use hyprstream_session_store::primary::{CollisionInventory, InventorySource};
use zeroize::Zeroize as _;

use crate::auth::identity_store;
use crate::config::OAuthConfig;

/// Build one fingerprint of *all* configured foreign protocol keys. The
/// service names come from the trusted service-factory inventory, not from a
/// browser or from whichever records happened to parse successfully.
#[allow(dead_code)] // Used when the Federate Policy runtime factory is installed.
pub(in crate::services) fn complete_collision_inventory(
    oauth: &OAuthConfig,
    credentials_dir: &Path,
    node_key: &SigningKey,
    required_services: &[&str],
) -> Result<CollisionInventory> {
    ensure!(
        !required_services.is_empty(),
        "required service inventory is empty"
    );
    let unique_services: BTreeSet<_> = required_services.iter().copied().collect();
    ensure!(
        unique_services.len() == required_services.len()
            && unique_services.iter().all(|name| !name.is_empty()),
        "required service inventory is invalid"
    );

    let mut required = Vec::new();
    let mut sources = Vec::new();
    let mut add = |id: String, keys: Vec<Vec<u8>>| {
        required.push(id.clone());
        sources.push(InventorySource { id, keys });
    };

    // The CA JWT key is purpose-derived from the already loaded node root.
    // Refuse a stale public credential file rather than inventorying a key
    // different from the one Policy actually uses to sign.
    let ca_jwt = hyprstream_rpc::node_identity::derive_purpose_key(node_key, "hyprstream-jwt-v1");
    let ca_ed = identity_store::load_ca_verifying_key(credentials_dir)
        .context("Federate collision inventory CA Ed key unavailable")?;
    let ca_pq = identity_store::load_ca_ml_dsa_verifying_key(credentials_dir)
        .context("Federate collision inventory CA PQ key unavailable")?;
    let expected_ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca_jwt);
    ensure!(
        ca_ed == ca_jwt.verifying_key()
            && ml_dsa_vk_bytes(&ca_pq) == ml_dsa_sk_to_vk_bytes(&expected_ca_pq),
        "Federate collision inventory CA keys do not match the active authority"
    );
    add(
        "ca:jwt".into(),
        vec![ca_ed.as_bytes().to_vec(), ml_dsa_vk_bytes(&ca_pq)],
    );

    // Check the OS-owned node seed without generating or changing it. The
    // in-memory key is the authority; the file is a required consistency
    // source, not a fallback to another identity.
    let mut node_seed = identity_store::read_secret(credentials_dir, "signing-key")
        .context("read node identity for Federate collision inventory")?
        .context("node identity missing from Federate collision inventory")?;
    let node_matches = node_seed.len() == 32 && node_seed.as_slice() == node_key.to_bytes();
    node_seed.zeroize();
    ensure!(node_matches, "node identity differs from active authority");
    let node_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(node_key);
    add(
        "node:envelope".into(),
        vec![
            node_key.verifying_key().as_bytes().to_vec(),
            ml_dsa_sk_to_vk_bytes(&node_pq),
        ],
    );

    // The existing parser returns an empty map when this file is absent.
    // For Policy admission that would silently omit every service identity.
    ensure!(
        identity_store::read_secret(credentials_dir, "bootstrap-pubkeys")
            .context("read Federate bootstrap public-key source")?
            .is_some(),
        "Federate bootstrap public-key source is missing"
    );
    let bootstrap = identity_store::load_bootstrap_pubkeys_hybrid(credentials_dir)
        .context("parse Federate bootstrap public-key source")?;
    ensure!(
        !bootstrap.is_empty(),
        "Federate bootstrap public-key source is empty"
    );
    // The low-level loader deliberately still accepts legacy classical-only
    // records, but every configured record enters this inventory, so one stale
    // Ed25519-only entry would omit a service's PQ half and let a fresh proof
    // key paired with the PQ key that service derives from its seed pass
    // `permits`. Reject the whole file instead.
    identity_store::ensure_bootstrap_pubkeys_hybrid(&bootstrap)
        .context("Federate collision inventory bootstrap source is not fully hybrid")?;
    for service in required_services {
        ensure!(
            bootstrap.contains_key(*service),
            "required Federate bootstrap service {service} is missing"
        );
    }
    for (name, entry) in bootstrap {
        let mut keys = vec![entry.ed25519.as_bytes().to_vec()];
        if let Some(pq) = entry.ml_dsa_65 {
            keys.push(ml_dsa_vk_bytes(&pq));
        }
        add(format!("bootstrap:{name}"), keys);
    }

    // Every configured mesh peer contributes both halves. A codec prefix
    // alone accepts cross-algorithm payloads, so each multikey payload is
    // also validated against its declared algorithm. A malformed peer is an
    // unavailable required source, not an empty source.
    for (name, peer) in &oauth.mesh_peers {
        ensure!(!name.is_empty(), "mesh peer has an empty inventory ID");
        let ed = decode_multikey(&peer.ed25519_multibase, &MULTICODEC_ED25519_PUB)
            .with_context(|| format!("mesh peer {name} Ed key is invalid"))?;
        let ed_bytes: [u8; 32] = ed.as_slice().try_into().map_err(|_| {
            anyhow!(
                "mesh peer {name} Ed key payload is {} bytes (expected 32)",
                ed.len()
            )
        })?;
        VerifyingKey::from_bytes(&ed_bytes).with_context(|| {
            format!("mesh peer {name} Ed key is not a valid Ed25519 verifying key")
        })?;
        let pq = decode_multikey(&peer.mldsa65_multibase, &MULTICODEC_ML_DSA_65_PUB)
            .with_context(|| format!("mesh peer {name} PQ key is invalid"))?;
        ml_dsa_vk_from_bytes(&pq).with_context(|| {
            format!("mesh peer {name} PQ key is not a valid ML-DSA-65 verifying key")
        })?;
        add(format!("mesh:{name}"), vec![ed, pq]);
    }

    CollisionInventory::from_sources(&required, sources)
        .context("Federate foreign-key collision inventory is incomplete")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::auth::identity_store::BootstrapPubkey;
    use std::collections::HashMap;

    fn fixture(dir: &Path) -> SigningKey {
        let node = SigningKey::from_bytes(&[0x49; 32]);
        let ca = hyprstream_rpc::node_identity::derive_purpose_key(&node, "hyprstream-jwt-v1");
        identity_store::write_secret(dir, "signing-key", &node.to_bytes()).unwrap();
        identity_store::write_ca_verifying_key(dir, &ca.verifying_key()).unwrap();
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let ca_pq = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        identity_store::write_ca_ml_dsa_verifying_key(dir, &ca_pq).unwrap();
        let service = SigningKey::from_bytes(&[0x53; 32]);
        identity_store::write_bootstrap_pubkeys_hybrid(
            dir,
            &HashMap::from([(
                "policy".into(),
                BootstrapPubkey::for_service_key(&service).unwrap(),
            )]),
        )
        .unwrap();
        node
    }

    #[test]
    fn complete_inventory_rejects_foreign_keys_and_missing_sources() {
        let dir = tempfile::tempdir().unwrap();
        let node = fixture(dir.path());
        let mut oauth = OAuthConfig::default();
        let inventory =
            complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).unwrap();
        let fresh_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(
            &SigningKey::from_bytes(&[0x7b; 32]),
        );
        let fresh_pq = ml_dsa_sk_to_vk_bytes(&fresh_pq);
        assert!(inventory.permits(&[0x7a; 32], &fresh_pq));
        assert!(!inventory.permits(node.verifying_key().as_bytes(), &fresh_pq));
        assert!(!inventory.permits(
            &[0x7a; 32],
            &ml_dsa_sk_to_vk_bytes(&hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&node),)
        ));
        assert!(!inventory.permits(
            SigningKey::from_bytes(&[0x53; 32])
                .verifying_key()
                .as_bytes(),
            &fresh_pq,
        ));

        let peer = SigningKey::from_bytes(&[0x65; 32]);
        let peer_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&peer);
        let peer_pq = ml_dsa_sk_to_vk_bytes(&peer_pq);
        oauth.mesh_peers.insert(
            "trusted-peer".into(),
            crate::config::MeshPeerConfig {
                ed25519_multibase: crate::auth::mesh_trust::encode_multikey(
                    peer.verifying_key().as_bytes(),
                    &MULTICODEC_ED25519_PUB,
                ),
                mldsa65_multibase: crate::auth::mesh_trust::encode_multikey(
                    &peer_pq,
                    &MULTICODEC_ML_DSA_65_PUB,
                ),
            },
        );
        let with_peer =
            complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).unwrap();
        assert_ne!(inventory.id(), with_peer.id());
        assert!(!with_peer.permits(peer.verifying_key().as_bytes(), &fresh_pq));
        assert!(!with_peer.permits(&[0x7a; 32], &peer_pq));

        assert!(
            complete_collision_inventory(&oauth, dir.path(), &node, &["policy", "oauth"]).is_err()
        );
        std::fs::remove_file(dir.path().join("bootstrap-pubkeys")).unwrap();
        assert!(complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).is_err());
    }

    #[test]
    fn malformed_or_mismatched_authority_never_becomes_an_empty_source() {
        let dir = tempfile::tempdir().unwrap();
        let node = fixture(dir.path());
        let mut oauth = OAuthConfig::default();
        oauth.mesh_peers.insert(
            "broken".into(),
            crate::config::MeshPeerConfig {
                ed25519_multibase: "not-a-multikey".into(),
                mldsa65_multibase: "not-a-multikey".into(),
            },
        );
        assert!(complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).is_err());
        oauth.mesh_peers.clear();

        identity_store::write_secret(dir.path(), "ca-pubkey", &[0x01; 32]).unwrap();
        assert!(complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).is_err());
        identity_store::write_ca_verifying_key(
            dir.path(),
            &hyprstream_rpc::node_identity::derive_purpose_key(&node, "hyprstream-jwt-v1")
                .verifying_key(),
        )
        .unwrap();

        identity_store::write_secret(dir.path(), "signing-key", &[0x02; 32]).unwrap();
        assert!(complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).is_err());
    }

    #[test]
    fn classical_bootstrap_entry_is_rejected_whether_required_or_extra() {
        let dir = tempfile::tempdir().unwrap();
        let node = fixture(dir.path());
        let oauth = OAuthConfig::default();
        assert!(complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).is_ok());

        // The required service provisioned by a pre-hybrid run: Ed25519-only.
        let stale = SigningKey::from_bytes(&[0x11; 32]);
        identity_store::write_bootstrap_pubkeys_hybrid(
            dir.path(),
            &HashMap::from([(
                "policy".into(),
                BootstrapPubkey::classical(stale.verifying_key()),
            )]),
        )
        .unwrap();
        let error = complete_collision_inventory(&oauth, dir.path(), &node, &["policy"])
            .err()
            .expect("classical-only required bootstrap service must be rejected");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("Ed25519-only") && rendered.contains("policy"),
            "error must name the stale entry, got: {rendered}"
        );

        // The loader promises to inventory every configured record, so an
        // extra non-required classical entry denies too, with everything else
        // (CA, node, required hybrid entry) still valid.
        let service = SigningKey::from_bytes(&[0x53; 32]);
        identity_store::write_bootstrap_pubkeys_hybrid(
            dir.path(),
            &HashMap::from([
                (
                    "policy".into(),
                    BootstrapPubkey::for_service_key(&service).unwrap(),
                ),
                (
                    "legacy".into(),
                    BootstrapPubkey::classical(stale.verifying_key()),
                ),
            ]),
        )
        .unwrap();
        let error = complete_collision_inventory(&oauth, dir.path(), &node, &["policy"])
            .err()
            .expect("extra classical-only bootstrap entry must be rejected");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("Ed25519-only") && rendered.contains("legacy"),
            "error must name the stale entry, got: {rendered}"
        );
    }

    #[test]
    fn mesh_peer_multikeys_are_validated_by_declared_algorithm() {
        let dir = tempfile::tempdir().unwrap();
        let node = fixture(dir.path());
        let mut oauth = OAuthConfig::default();
        let peer = SigningKey::from_bytes(&[0x65; 32]);
        let peer_ed = peer.verifying_key().as_bytes().to_vec();
        let peer_pq =
            ml_dsa_sk_to_vk_bytes(&hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&peer));
        let ed_multikey =
            crate::auth::mesh_trust::encode_multikey(&peer_ed, &MULTICODEC_ED25519_PUB);
        let pq_multikey =
            crate::auth::mesh_trust::encode_multikey(&peer_pq, &MULTICODEC_ML_DSA_65_PUB);

        // Ed codec prefix carrying an ML-DSA-65-length payload; PQ half valid.
        oauth.mesh_peers.insert(
            "ed-wrong-length".into(),
            crate::config::MeshPeerConfig {
                ed25519_multibase: crate::auth::mesh_trust::encode_multikey(
                    &peer_pq,
                    &MULTICODEC_ED25519_PUB,
                ),
                mldsa65_multibase: pq_multikey.clone(),
            },
        );
        let error = complete_collision_inventory(&oauth, dir.path(), &node, &["policy"])
            .err()
            .expect("Ed multikey with a 1952-byte payload must be rejected");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("ed-wrong-length") && rendered.contains("expected 32"),
            "error must name the peer and the length mismatch, got: {rendered}"
        );
        oauth.mesh_peers.clear();

        // ML-DSA codec prefix carrying a 32-byte payload; Ed half valid.
        oauth.mesh_peers.insert(
            "pq-wrong-length".into(),
            crate::config::MeshPeerConfig {
                ed25519_multibase: ed_multikey.clone(),
                mldsa65_multibase: crate::auth::mesh_trust::encode_multikey(
                    &peer_ed,
                    &MULTICODEC_ML_DSA_65_PUB,
                ),
            },
        );
        let error = complete_collision_inventory(&oauth, dir.path(), &node, &["policy"])
            .err()
            .expect("ML-DSA multikey with a 32-byte payload must be rejected");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("pq-wrong-length") && rendered.contains("ML-DSA-65"),
            "error must name the peer and the algorithm, got: {rendered}"
        );
        oauth.mesh_peers.clear();

        // Exactly 32 bytes but not a valid Ed25519 point encoding; PQ half
        // still valid, isolating the encoding check as the cause.
        oauth.mesh_peers.insert(
            "ed-not-a-point".into(),
            crate::config::MeshPeerConfig {
                ed25519_multibase: crate::auth::mesh_trust::encode_multikey(
                    &[0x02; 32],
                    &MULTICODEC_ED25519_PUB,
                ),
                mldsa65_multibase: pq_multikey.clone(),
            },
        );
        let error = complete_collision_inventory(&oauth, dir.path(), &node, &["policy"])
            .err()
            .expect("32-byte non-point Ed encoding must be rejected");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("ed-not-a-point") && rendered.contains("valid Ed25519 verifying key"),
            "error must name the peer and the encoding failure, got: {rendered}"
        );
        oauth.mesh_peers.clear();

        // The same keys under correct encodings load, so the denials above are
        // caused by the malformed payload, not by the peer itself.
        oauth.mesh_peers.insert(
            "valid".into(),
            crate::config::MeshPeerConfig {
                ed25519_multibase: ed_multikey,
                mldsa65_multibase: pq_multikey,
            },
        );
        assert!(complete_collision_inventory(&oauth, dir.path(), &node, &["policy"]).is_ok());
    }
}
