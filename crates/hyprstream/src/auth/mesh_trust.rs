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
use hyprstream_rpc::Subject;

// The multikey codec + multicodec constants are the single canonical home in
// `hyprstream-crypto` (re-exported via `hyprstream_rpc::did_key`); the previous
// local copies here were deduplicated in #916.
use hyprstream_rpc::did_key::{decode_multikey, MULTICODEC_ED25519_PUB, MULTICODEC_ML_DSA_65_PUB};

use crate::config::OAuthConfig;

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

/// Anchor the ML-DSA-65 keys carried by the node's own `bootstrap-pubkeys`
/// entries into `store`, keyed by the Ed25519 signer identity each entry
/// already anchors in the classical trust store.
///
/// The bootstrap file is OS-owned, written out-of-band by the provisioning
/// wizard, so it satisfies the [`KeyedPqTrustStore`] contract exactly as
/// `mesh_peers` does. Classical-only entries contribute nothing and clear
/// nothing: anchoring is monotonic, so a legacy file left in place after a
/// hybrid enrollment cannot downgrade an already-anchored service.
///
/// A missing or unreadable file yields zero bindings — the pre-hybrid
/// behavior, unchanged.
///
/// Returns the number of identities anchored from the file.
pub fn seed_bootstrap_pq_bindings(
    store: &mut KeyedPqTrustStore,
    credentials_dir: &std::path::Path,
) -> usize {
    let entries = match crate::auth::identity_store::load_bootstrap_pubkeys_hybrid(credentials_dir)
    {
        Ok(entries) => entries,
        Err(e) => {
            tracing::error!(
                "bootstrap-pubkeys could not be read for PQ anchoring, no bindings seeded \
                 (re-run the provisioning wizard if hybrid services are expected): {e}"
            );
            return 0;
        }
    };

    let mut anchored = 0usize;
    for (name, entry) in &entries {
        let ed_pubkey = entry.ed25519.to_bytes();
        if store.register(ed_pubkey, entry.ml_dsa_65.as_ref()) && entry.is_hybrid() {
            anchored += 1;
            tracing::info!("bootstrap service '{name}': anchored ML-DSA-65 key for its Ed25519 signer identity");
        }
    }
    anchored
}

/// Build the per-host mesh identity roster from the admin-configured
/// `mesh_peers` (#328): each peer's Ed25519 signer pubkey → its per-host
/// authorization subject (`service:inference:host-<label>`).
///
/// Reuses the SAME `mesh_peers` enrollment record as
/// [`build_mesh_pq_trust_store`] (no new roster type): the Ed25519 key is the
/// envelope signer identity, and the operator-assigned label is the host id.
/// An invalid Ed25519 multibase is logged and skipped (fail-safe — never trust
/// a malformed identity). An empty/absent `mesh_peers` yields an empty roster.
///
/// This is the source for fail-closed per-host identity resolution: a networked
/// peer whose key is NOT in this roster resolves to `anonymous`, never the
/// `"system"` god principal (#328).
pub fn build_mesh_identity_roster(oauth: &OAuthConfig) -> Vec<([u8; 32], Subject)> {
    let mut roster = Vec::new();
    for (label, peer) in &oauth.mesh_peers {
        let ed_bytes = match decode_multikey(&peer.ed25519_multibase, &MULTICODEC_ED25519_PUB) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid ed25519_multibase, skipping identity roster entry: {e}");
                continue;
            }
        };
        let ed_pubkey: [u8; 32] = match ed_bytes.as_slice().try_into() {
            Ok(a) => a,
            Err(_) => {
                tracing::error!(
                    "mesh_peer '{label}': ed25519 key is {} bytes (expected 32), skipping identity roster entry",
                    ed_bytes.len()
                );
                continue;
            }
        };
        let subject = hyprstream_rpc::node_identity::mesh_host_subject(label);
        tracing::info!(
            "mesh_peer '{label}': enrolled as per-host subject {:?}",
            subject.name()
        );
        roster.push((ed_pubkey, subject));
    }
    roster
}

/// Build the v16 proof enrollment resolver from the same admin-anchored
/// `mesh_peers` roster that seeds the PQ trust store and the identity roster.
///
/// Each configured peer is enrolled as **one** logical signer using the
/// weakly-non-separable Ed25519 + ML-DSA-65 suite, pinning that peer's exact
/// published component keys in suite order. The peer's Ed25519 public key is
/// the key ID for both components: kids are opaque in the profile, and a
/// deployment's kid convention is enrollment data. The two components share
/// one logical signer group and count as one approval.
///
/// Deliberately partial, and fail-closed where it is:
///
/// - **Approvers.** `mesh_peers` describes signers, not approval roles, so no
///   approver is enrolled here. A method whose generated policy requires
///   approvals therefore denies until a manifest supplies them — it is never
///   satisfied by a primary signer standing in for an approver.
/// - **Service response signers.** Likewise absent, so response-proof
///   verification denies rather than trusting any enrolled key.
/// - **Enrollment expiry.** Admin-anchored entries are established out-of-band
///   and do not themselves expire, matching the existing trust-store
///   convention. The *credential's* expiry still bounds every proof: dispatch
///   independently requires `proof.exp` not to exceed the verified
///   credential's `exp`.
///
/// An invalid entry is logged and skipped: a malformed peer key must not
/// silently enrol the wrong identity.
pub fn build_mesh_enrollment_resolver(
    oauth: &OAuthConfig,
) -> hyprstream_rpc::proof::enrollment::InMemoryEnrollmentResolver {
    use hyprstream_rpc::proof::enrollment::{
        ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole, SignerSuiteRecord,
    };

    let mut resolver = InMemoryEnrollmentResolver::new();
    for (label, peer) in &oauth.mesh_peers {
        let ed_bytes = match decode_multikey(&peer.ed25519_multibase, &MULTICODEC_ED25519_PUB) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid ed25519_multibase, not enrolled: {e}");
                continue;
            }
        };
        let ed_pubkey: [u8; 32] = match ed_bytes.as_slice().try_into() {
            Ok(a) => a,
            Err(_) => {
                tracing::error!(
                    "mesh_peer '{label}': ed25519 key is {} bytes (expected 32), not enrolled",
                    ed_bytes.len()
                );
                continue;
            }
        };
        let ed_vk = match ed25519_dalek::VerifyingKey::from_bytes(&ed_pubkey) {
            Ok(vk) => vk,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid ed25519 verifying key, not enrolled: {e}");
                continue;
            }
        };
        let pq_bytes = match decode_multikey(&peer.mldsa65_multibase, &MULTICODEC_ML_DSA_65_PUB) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("mesh_peer '{label}': invalid mldsa65_multibase, not enrolled: {e}");
                continue;
            }
        };
        let pq_vk = match hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pq_bytes) {
            Ok(vk) => vk,
            Err(e) => {
                tracing::error!(
                    "mesh_peer '{label}': invalid ML-DSA-65 verifying key, not enrolled: {e}"
                );
                continue;
            }
        };

        let subject = hyprstream_rpc::node_identity::mesh_host_subject(label);
        let principal = subject
            .name()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| label.clone());
        let record = SignerSuiteRecord {
            principal,
            suite_id: hyprstream_rpc::proof::SUITE_HYBRID.to_owned(),
            components: vec![
                EnrolledComponent::new(ed_pubkey.to_vec(), ComponentKey::Ed25519(ed_vk)),
                EnrolledComponent::new(ed_pubkey.to_vec(), ComponentKey::MlDsa65(Box::new(pq_vk))),
            ],
            epoch: 0,
            role: SignerRole::Primary,
            approver_role: None,
            not_after: u64::MAX,
            revoked: false,
        };
        match resolver.enrol_primary(&ed_vk, record) {
            Ok(()) => tracing::info!(
                "mesh_peer '{label}': enrolled as a hybrid primary proof signer"
            ),
            Err(e) => tracing::error!("mesh_peer '{label}': enrollment rejected: {e}"),
        }
    }
    resolver
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

    // -- v16 proof enrollment ------------------------------------------------

    fn peer_entry() -> (SigningKey, MeshPeerConfig) {
        let ed = SigningKey::generate(&mut OsRng);
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ed);
        let cfg = MeshPeerConfig {
            ed25519_multibase: encode_multikey(
                &ed.verifying_key().to_bytes(),
                &MULTICODEC_ED25519_PUB,
            ),
            mldsa65_multibase: encode_multikey(
                &pq::ml_dsa_sk_to_vk_bytes(&pq_sk),
                &MULTICODEC_ML_DSA_65_PUB,
            ),
        };
        (ed, cfg)
    }

    /// A configured peer becomes one hybrid primary signer whose exact
    /// published component keys are pinned in suite order.
    #[test]
    fn a_mesh_peer_is_enrolled_as_one_hybrid_primary_signer() {
        use hyprstream_rpc::proof::enrollment::EnrollmentResolver;

        let (ed, cfg) = peer_entry();
        let oauth = oauth_with_peers(vec![("peer-a", cfg)]);
        let resolver = build_mesh_enrollment_resolver(&oauth);

        let record = resolver
            .resolve_primary(&ed.verifying_key())
            .expect("the configured peer must resolve");
        assert_eq!(record.suite_id, hyprstream_rpc::proof::SUITE_HYBRID);
        assert_eq!(record.components.len(), 2);
        assert_eq!(record.components[0].alg, hyprstream_rpc::proof::ALG_ED25519);
        assert_eq!(record.components[1].alg, hyprstream_rpc::proof::ALG_ML_DSA_65);
        assert!(record.pins_ed25519(&ed.verifying_key()));
    }

    /// An unconfigured key resolves to nothing, and no approver or service
    /// signer is invented — a method requiring either denies.
    #[test]
    fn mesh_enrollment_invents_no_signer_it_was_not_given() {
        use hyprstream_rpc::proof::enrollment::EnrollmentResolver;

        let (ed, cfg) = peer_entry();
        let stranger = SigningKey::generate(&mut OsRng);
        let oauth = oauth_with_peers(vec![("peer-a", cfg)]);
        let resolver = build_mesh_enrollment_resolver(&oauth);

        assert!(resolver.resolve_primary(&stranger.verifying_key()).is_none());
        assert!(resolver
            .resolve_approver(&ed.verifying_key().to_bytes())
            .is_none());
        assert!(resolver.resolve_service("registry.svc.hyprstream.test").is_none());
    }

    /// A malformed peer is skipped rather than enrolled under wrong material.
    #[test]
    fn a_malformed_peer_is_not_enrolled() {
        use hyprstream_rpc::proof::enrollment::EnrollmentResolver;

        let (ed, mut cfg) = peer_entry();
        cfg.mldsa65_multibase = "znot-a-multikey".to_owned();
        let oauth = oauth_with_peers(vec![("peer-a", cfg)]);
        let resolver = build_mesh_enrollment_resolver(&oauth);
        assert!(resolver.resolve_primary(&ed.verifying_key()).is_none());
    }

    #[test]
    fn empty_mesh_peers_enrol_nothing() {
        use hyprstream_rpc::proof::enrollment::EnrollmentResolver;

        let resolver = build_mesh_enrollment_resolver(&OAuthConfig::default());
        let stranger = SigningKey::generate(&mut OsRng);
        assert!(resolver.resolve_primary(&stranger.verifying_key()).is_none());
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
    fn identity_roster_maps_peers_to_per_host_subjects() {
        // #328: each enrolled peer's Ed25519 key maps to its own granular
        // per-host subject (service:inference:host-<label>), reusing the SAME
        // mesh_peers enrollment record.
        let peer_ed = SigningKey::generate(&mut OsRng);
        let peer_pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&peer_ed);
        let ed_mb = encode_multikey(&peer_ed.verifying_key().to_bytes(), &MULTICODEC_ED25519_PUB);
        let pq_mb = encode_multikey(
            &pq::ml_dsa_sk_to_vk_bytes(&peer_pq_sk),
            &MULTICODEC_ML_DSA_65_PUB,
        );
        let oauth = oauth_with_peers(vec![(
            "gpu-node-3",
            MeshPeerConfig { ed25519_multibase: ed_mb, mldsa65_multibase: pq_mb },
        )]);

        let roster = build_mesh_identity_roster(&oauth);
        assert_eq!(roster.len(), 1);
        let (pubkey, subject) = &roster[0];
        assert_eq!(*pubkey, peer_ed.verifying_key().to_bytes());
        assert_eq!(subject.name(), Some("service:inference:host-gpu-node-3"));
        // Crucially, NOT "system".
        assert_ne!(subject.name(), Some("system"));
    }

    #[test]
    fn identity_roster_skips_malformed_entries() {
        let good_pq = encode_multikey(&vec![0u8; 1952], &MULTICODEC_ML_DSA_65_PUB);
        let oauth = oauth_with_peers(vec![(
            "bad-peer",
            MeshPeerConfig {
                ed25519_multibase: "not-multibase".to_owned(),
                mldsa65_multibase: good_pq,
            },
        )]);
        let roster = build_mesh_identity_roster(&oauth);
        assert!(roster.is_empty(), "malformed entry must be skipped, not enrolled");
    }

    #[test]
    fn empty_mesh_peers_yields_empty_identity_roster() {
        let roster = build_mesh_identity_roster(&OAuthConfig::default());
        assert!(roster.is_empty());
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

    // ─── bootstrap-pubkeys PQ seeding ────────────────────────────────────────

    fn write_entries(
        dir: &std::path::Path,
        entries: Vec<(&str, crate::auth::identity_store::BootstrapPubkey)>,
    ) {
        let map: std::collections::HashMap<String, _> =
            entries.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
        crate::auth::identity_store::write_bootstrap_pubkeys_hybrid(dir, &map)
            .expect("write bootstrap-pubkeys");
    }

    #[test]
    fn hybrid_bootstrap_file_anchors_its_services() {
        use crate::auth::identity_store::BootstrapPubkey;
        use hyprstream_rpc::envelope::PqTrustStore;

        let dir = tempfile::TempDir::new().unwrap();
        let ed = SigningKey::generate(&mut OsRng);
        let (_pq_sk, pq_vk) = pq::ml_dsa_generate_keypair();
        write_entries(
            dir.path(),
            vec![("policy", BootstrapPubkey::hybrid(ed.verifying_key(), pq_vk))],
        );

        let mut store = KeyedPqTrustStore::new();
        let anchored = seed_bootstrap_pq_bindings(&mut store, dir.path());
        assert_eq!(anchored, 1, "the hybrid entry must be anchored");
        assert!(store
            .ml_dsa_key_for(&ed.verifying_key().to_bytes())
            .is_some());
    }

    #[test]
    fn classical_bootstrap_file_anchors_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let ed = SigningKey::generate(&mut OsRng);
        write_entries(
            dir.path(),
            vec![(
                "policy",
                crate::auth::identity_store::BootstrapPubkey::classical(ed.verifying_key()),
            )],
        );

        let mut store = KeyedPqTrustStore::new();
        let anchored = seed_bootstrap_pq_bindings(&mut store, dir.path());
        assert_eq!(anchored, 0, "a classical-only file must anchor nothing");
        assert!(store.is_empty(), "a classical-only file must change nothing");
    }

    #[test]
    fn absent_bootstrap_file_anchors_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut store = KeyedPqTrustStore::new();
        assert_eq!(seed_bootstrap_pq_bindings(&mut store, dir.path()), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn a_later_classical_file_cannot_unanchor_a_service() {
        use crate::auth::identity_store::BootstrapPubkey;

        let dir = tempfile::TempDir::new().unwrap();
        let ed = SigningKey::generate(&mut OsRng);
        let (_pq_sk, pq_vk) = pq::ml_dsa_generate_keypair();

        let mut store = KeyedPqTrustStore::new();
        write_entries(
            dir.path(),
            vec![("policy", BootstrapPubkey::hybrid(ed.verifying_key(), pq_vk))],
        );
        assert_eq!(seed_bootstrap_pq_bindings(&mut store, dir.path()), 1);

        // Re-seeding from a file that lost its PQ half must leave the anchor.
        write_entries(
            dir.path(),
            vec![("policy", BootstrapPubkey::classical(ed.verifying_key()))],
        );
        seed_bootstrap_pq_bindings(&mut store, dir.path());
        assert!(
            store.is_anchored(&ed.verifying_key().to_bytes()),
            "a classical re-registration must not clear an established anchor"
        );
    }
}
