//! Node identity provider — native key management with HKDF purpose derivation.
//!
//! Wraps a root Ed25519 signing key and derives purpose-keyed identities via HKDF.
//! The root key comes from `identity_store.rs` (file-based) or OS keyring.
//!
//! HKDF derivation: `HKDF-SHA256(ikm=root_seed, info=purpose_bytes)` → 32-byte Ed25519 seed.
//! Same purpose always produces the same keypair from the same root.
//!
//! Derived pubkeys are tracked automatically — `resolve()` recognizes both
//! the root node pubkey and any purpose-derived pubkeys opened via `identity_open()`.

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use hkdf::Hkdf;
use sha2::Sha256;
use std::collections::HashMap;
use parking_lot::RwLock;
use zeroize::Zeroize;

use crate::identity::{IdentityProvider, SigningIdentity};
use crate::Subject;

/// Domain-separated HKDF salt for node identity derivation.
/// Prevents cross-protocol confusion if the same seed material is used elsewhere.
const NODE_IDENTITY_SALT: &[u8] = b"hyprstream-node-identity-hkdf-v1";

/// Build the per-host mesh-peer subject for an enrolled host label (#328).
///
/// Extends the existing `service:<name>` convention (see
/// `EnvelopeContext::from_callback_service`) with a per-host third segment so
/// every mesh peer authenticates as its OWN granular principal
/// (`service:inference:host-<label>`) rather than collapsing to the omnipotent
/// `"system"` principal. The label is the operator-assigned `mesh_peers` key.
pub fn mesh_host_subject(label: &str) -> Subject {
    Subject::new(format!("service:inference:host-{label}"))
}

/// Derive a purpose-keyed Ed25519 signing key from a root key.
///
/// Standalone function for use in sync contexts (e.g., JWT signing)
/// where the full async `IdentityProvider` isn't needed.
#[allow(clippy::expect_used)] // HKDF-SHA256 expand to 32 bytes cannot fail
pub fn derive_purpose_key(root_key: &SigningKey, purpose: &str) -> SigningKey {
    assert!(!purpose.is_empty(), "purpose must be non-empty");
    assert!(purpose.len() <= 255, "purpose must be at most 255 bytes");
    let hk = Hkdf::<Sha256>::new(Some(NODE_IDENTITY_SALT), &root_key.to_bytes());
    let mut okm = [0u8; 32];
    hk.expand(purpose.as_bytes(), &mut okm)
        .expect("HKDF-SHA256 expand to 32 bytes cannot fail");
    let key = SigningKey::from_bytes(&okm);
    okm.zeroize();
    key
}

/// HKDF purpose label for the mesh ML-DSA-65 signing key (#157).
///
/// Distinct from any Ed25519 / JWT purpose label so the post-quantum mesh key
/// satisfies the PQUIP key-separation restriction (a derived key is bound to
/// exactly one cryptographic purpose). The `-v1` suffix allows a future
/// rotation by bumping the version without changing the root node key.
pub const MESH_MLDSA_PURPOSE: &str = "hyprstream-mesh-mldsa-v1";

/// Deterministically derive the node's mesh ML-DSA-65 signing key from its
/// persisted Ed25519 root/signing key (#157).
///
/// Uses [`derive_purpose_key`] (HKDF-SHA256 over the Ed25519 seed) with the
/// dedicated [`MESH_MLDSA_PURPOSE`] label, then re-keys that 32-byte output as
/// an ML-DSA-65 seed via [`crate::crypto::pq::ml_dsa_sk_from_seed`]. This is
/// stable across restarts (no new key file) and key-separated from both the
/// Ed25519 identity and the JWT ML-DSA keyset.
///
/// The matching public key (`signing_key.verifying_key()`) is what peers must
/// anchor in their `KeyedPqTrustStore`, keyed by the Ed25519 signer identity.
pub fn derive_mesh_mldsa_key(ed25519_key: &SigningKey) -> crate::crypto::pq::MlDsaSigningKey {
    let derived = derive_purpose_key(ed25519_key, MESH_MLDSA_PURPOSE);
    let mut seed = derived.to_bytes();
    let sk = crate::crypto::pq::ml_dsa_sk_from_seed(&seed);
    seed.zeroize();
    sk
}

/// A purpose-derived Ed25519 signing identity.
/// Inner SigningKey is zeroized on drop via ed25519-dalek's `zeroize` feature.
struct DerivedIdentity {
    signing_key: SigningKey,
    pubkey: [u8; 32],
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl SigningIdentity for DerivedIdentity {
    fn pubkey(&self) -> [u8; 32] {
        self.pubkey
    }

    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]> {
        use ed25519_dalek::Signer as _;
        Ok(self.signing_key.sign(canonical_bytes).to_bytes())
    }
}

/// Native identity provider backed by a root Ed25519 key.
///
/// Derives purpose-keyed identities via HKDF. Maps every recognized pubkey to a
/// specific authorization [`Subject`] (#328): the node's own root/derived keys
/// map to `"system"` (genuine in-process authority), while enrolled mesh-peer
/// keys map to their per-host subject (`service:inference:host-<label>`). An
/// unrecognized pubkey resolves to [`Subject::anonymous`] — deny-by-default,
/// never the `"system"` god principal.
pub struct NodeIdentityProvider {
    root_seed: [u8; 32],
    #[allow(dead_code)]
    node_pubkey: [u8; 32],
    /// Recognized pubkey → authorization subject.
    ///
    /// The node's own root and purpose-derived keys are inserted with
    /// `Subject::new("system")`; mesh peers are inserted via
    /// [`NodeIdentityProvider::register_peer`] with their per-host subject.
    known_pubkeys: RwLock<HashMap<[u8; 32], Subject>>,
}

impl Drop for NodeIdentityProvider {
    fn drop(&mut self) {
        self.root_seed.zeroize();
    }
}

impl NodeIdentityProvider {
    /// Create from a root signing key.
    pub fn new(root_key: &SigningKey) -> Self {
        let node_pubkey = root_key.verifying_key().to_bytes();
        let mut known = HashMap::new();
        // The node's own root key is genuine in-process authority → "system".
        known.insert(node_pubkey, Subject::new("system"));
        Self {
            root_seed: root_key.to_bytes(),
            node_pubkey,
            known_pubkeys: RwLock::new(known),
        }
    }

    /// Derive a purpose-keyed Ed25519 signing key via HKDF.
    fn derive(&self, purpose: &str) -> Result<SigningKey> {
        if purpose.is_empty() {
            anyhow::bail!("purpose must be non-empty");
        }
        if purpose.len() > 255 {
            anyhow::bail!("purpose must be at most 255 bytes");
        }
        let hk = Hkdf::<Sha256>::new(Some(NODE_IDENTITY_SALT), &self.root_seed);
        let mut okm = [0u8; 32];
        #[allow(clippy::expect_used)] // HKDF-SHA256 expand to 32 bytes cannot fail
        hk.expand(purpose.as_bytes(), &mut okm)
            .expect("HKDF-SHA256 expand to 32 bytes cannot fail");
        let key = SigningKey::from_bytes(&okm);
        okm.zeroize();
        Ok(key)
    }

    /// Pre-register a purpose so its derived pubkey is recognized by `resolve()`.
    ///
    /// The derived key belongs to THIS node, so it maps to the in-process
    /// `"system"` authority. Called automatically by `identity_open()`, but can
    /// also be called at startup to register purposes used internally.
    pub fn register_purpose(&self, purpose: &str) -> Result<[u8; 32]> {
        let key = self.derive(purpose)?;
        let pubkey = key.verifying_key().to_bytes();
        self.known_pubkeys.write().insert(pubkey, Subject::new("system"));
        Ok(pubkey)
    }

    /// Enroll a networked mesh peer's Ed25519 signer key → per-host subject (#328).
    ///
    /// The peer authenticates as its OWN granular principal
    /// (`service:inference:host-<label>`), NEVER as `"system"`. The roster is
    /// populated from the operator-configured `mesh_peers` enrollment record
    /// (see `hyprstream::auth::mesh_trust`), so a peer whose key is not enrolled
    /// resolves to [`Subject::anonymous`] (fail-closed, deny-by-default).
    ///
    /// A peer key MUST NOT shadow the node's own `"system"` keys: if the pubkey
    /// is already registered (e.g. the node's root/derived key), the existing
    /// binding is kept and the peer entry is rejected to prevent a misconfigured
    /// roster from downgrading in-process authority.
    pub fn register_peer(&self, pubkey: [u8; 32], subject: Subject) {
        let mut map = self.known_pubkeys.write();
        match map.entry(pubkey) {
            std::collections::hash_map::Entry::Occupied(existing) => {
                // `tracing` is a native-only dependency for this crate (see Cargo.toml:
                // it lives under `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]`),
                // so the macro must be cfg-gated to keep the wasm32 build compiling (#468).
                #[cfg(not(target_arch = "wasm32"))]
                tracing::warn!(
                    existing = ?existing.get().name(),
                    rejected = ?subject.name(),
                    "node_identity: mesh peer key collides with an existing binding; keeping existing, rejecting peer enrollment"
                );
                #[cfg(target_arch = "wasm32")]
                let _ = &existing; // silence unused warning on wasm where tracing is absent
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(subject);
            }
        }
    }

    /// Check if a pubkey is recognized by this node (root, derived, or enrolled peer).
    pub fn is_known(&self, pubkey: &[u8; 32]) -> bool {
        self.known_pubkeys.read().contains_key(pubkey)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl IdentityProvider for NodeIdentityProvider {
    async fn identity_open(&self, purpose: &str) -> Result<Box<dyn SigningIdentity>> {
        let signing_key = self.derive(purpose)?;
        let pubkey = signing_key.verifying_key().to_bytes();
        // Track this derived pubkey so resolve() recognizes it as in-process "system".
        self.known_pubkeys.write().insert(pubkey, Subject::new("system"));
        Ok(Box::new(DerivedIdentity { signing_key, pubkey }))
    }

    fn resolve(&self, pubkey: &[u8; 32]) -> Subject {
        // Return the per-key subject: "system" for the node's own root/derived
        // keys, a per-host subject for enrolled mesh peers, anonymous otherwise
        // (#328 — no "any known key → system" collapse, deny-by-default).
        self.known_pubkeys
            .read()
            .get(pubkey)
            .cloned()
            .unwrap_or_else(Subject::anonymous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn same_purpose_same_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let id1 = provider.identity_open("test-purpose-v1").await.unwrap();
        let id2 = provider.identity_open("test-purpose-v1").await.unwrap();
        assert_eq!(id1.pubkey(), id2.pubkey());
    }

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn different_purpose_different_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let id1 = provider.identity_open("purpose-a").await.unwrap();
        let id2 = provider.identity_open("purpose-b").await.unwrap();
        assert_ne!(id1.pubkey(), id2.pubkey());
    }

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn sign_and_verify() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let id = provider.identity_open("test-sign-v1").await.unwrap();
        let msg = b"hello world";
        let sig = id.sign(msg).await.unwrap();

        let vk = ed25519_dalek::VerifyingKey::from_bytes(&id.pubkey()).unwrap();
        let signature = ed25519_dalek::Signature::from_bytes(&sig);
        use ed25519_dalek::Verifier;
        assert!(vk.verify(msg, &signature).is_ok());
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn resolve_node_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);
        let node_pub = root.verifying_key().to_bytes();

        assert_eq!(provider.resolve(&node_pub).name(), Some("system"));
        assert_eq!(provider.resolve(&[0u8; 32]).name(), None);
    }

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn resolve_derived_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        // Before opening, derived pubkey is unknown
        let pubkey = provider.register_purpose("test-v1").unwrap();
        assert_eq!(provider.resolve(&pubkey).name(), Some("system"));

        // identity_open also registers
        let id = provider.identity_open("another-v1").await.unwrap();
        assert_eq!(provider.resolve(&id.pubkey()).name(), Some("system"));

        // Unknown key still anonymous
        assert_eq!(provider.resolve(&[0u8; 32]).name(), None);
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn enrolled_peer_resolves_to_per_host_subject() {
        // #328: an enrolled mesh peer must resolve to its OWN granular per-host
        // subject, never to "system".
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let peer = SigningKey::generate(&mut rand::rngs::OsRng);
        let peer_pub = peer.verifying_key().to_bytes();
        provider.register_peer(peer_pub, mesh_host_subject("gpu-node-7"));

        assert_eq!(
            provider.resolve(&peer_pub).name(),
            Some("service:inference:host-gpu-node-7"),
        );
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn unenrolled_peer_resolves_to_anonymous_not_system() {
        // #328 regression: the old "any known key → system god principal"
        // behavior is gone. A networked peer key that is NOT in the roster must
        // resolve to anonymous, never "system".
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let stranger = SigningKey::generate(&mut rand::rngs::OsRng);
        let stranger_pub = stranger.verifying_key().to_bytes();

        let resolved = provider.resolve(&stranger_pub);
        assert!(resolved.is_anonymous(), "unenrolled peer must be anonymous");
        assert_ne!(resolved.name(), Some("system"));
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn node_root_key_still_resolves_to_system() {
        // #328 no-regression: the node's own root key (genuine in-process
        // authority) must keep resolving to "system".
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let node_pub = root.verifying_key().to_bytes();
        let provider = NodeIdentityProvider::new(&root);
        assert_eq!(provider.resolve(&node_pub).name(), Some("system"));
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn peer_enrollment_cannot_shadow_system_keys() {
        // #328: a misconfigured roster must not be able to downgrade the node's
        // own "system" authority by re-binding the node's root key.
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let node_pub = root.verifying_key().to_bytes();
        let provider = NodeIdentityProvider::new(&root);

        provider.register_peer(node_pub, mesh_host_subject("evil"));
        assert_eq!(
            provider.resolve(&node_pub).name(),
            Some("system"),
            "existing system binding must win over a colliding peer enrollment"
        );
    }

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn empty_purpose_rejected() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);
        assert!(provider.identity_open("").await.is_err());
    }

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn long_purpose_rejected() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);
        let long = "x".repeat(256);
        assert!(provider.identity_open(&long).await.is_err());
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn mesh_mldsa_key_is_deterministic() {
        // #157: deriving the mesh ML-DSA key from the same node Ed25519 key must
        // be stable across calls (so it survives restarts without a key file).
        let root = SigningKey::from_bytes(&[3u8; 32]);
        let sk1 = derive_mesh_mldsa_key(&root);
        let sk2 = derive_mesh_mldsa_key(&root);
        assert_eq!(
            crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&sk1),
            crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&sk2),
            "same node seed must derive the same mesh ML-DSA key"
        );
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn mesh_mldsa_key_differs_per_node() {
        let root_a = SigningKey::generate(&mut rand::rngs::OsRng);
        let root_b = SigningKey::generate(&mut rand::rngs::OsRng);
        let vk_a = crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(&root_a));
        let vk_b = crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(&root_b));
        assert_ne!(vk_a, vk_b, "different node seeds must derive different mesh keys");
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn mesh_mldsa_key_separated_from_ed25519() {
        // PQUIP key separation: the derived ML-DSA seed must not equal the
        // Ed25519 node seed (it goes through a dedicated purpose label).
        let root = SigningKey::from_bytes(&[9u8; 32]);
        let mesh_sk = derive_mesh_mldsa_key(&root);
        let mesh_seed = crate::crypto::pq::ml_dsa_sk_to_seed(&mesh_sk);
        assert_ne!(
            mesh_seed,
            root.to_bytes(),
            "mesh ML-DSA seed must be key-separated from the Ed25519 node seed"
        );
        // And it equals the explicit two-step derivation (purpose key -> seed).
        let expected_seed = derive_purpose_key(&root, MESH_MLDSA_PURPOSE).to_bytes();
        assert_eq!(mesh_seed, expected_seed, "derivation must match the purpose-key path");
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn salt_produces_different_keys_than_none() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        // With salt (current implementation)
        let salted = derive_purpose_key(&root, "test-purpose");
        // Without salt (old implementation)
        let hk = Hkdf::<Sha256>::new(None, &root.to_bytes());
        let mut unsalted_okm = [0u8; 32];
        hk.expand(b"test-purpose", &mut unsalted_okm).unwrap();
        let unsalted = SigningKey::from_bytes(&unsalted_okm);
        unsalted_okm.fill(0);

        assert_ne!(
            salted.verifying_key().to_bytes(),
            unsalted.verifying_key().to_bytes(),
            "Salted and unsalted HKDF must produce different keys"
        );
    }
}
