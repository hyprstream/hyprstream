//! Node identity provider — native key management with HKDF purpose derivation.
//!
//! Wraps a root Ed25519 signing key and derives purpose-keyed identities via HKDF.
//! The root key comes from `credentials.rs` (file-based) or OS keyring.
//!
//! HKDF derivation: `HKDF-SHA256(ikm=root_seed, info=purpose_bytes)` → 32-byte Ed25519 seed.
//! Same purpose always produces the same keypair from the same root.

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use hkdf::Hkdf;
use sha2::Sha256;

use crate::identity::{IdentityProvider, SigningIdentity};
use crate::Subject;

/// A purpose-derived Ed25519 signing identity.
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
/// Derives purpose-keyed identities via HKDF. The root key is loaded
/// from disk or OS keyring by the caller — this struct doesn't do I/O.
pub struct NodeIdentityProvider {
    root_seed: [u8; 32],
    node_pubkey: [u8; 32],
}

impl NodeIdentityProvider {
    /// Create from a root signing key.
    pub fn new(root_key: &SigningKey) -> Self {
        let node_pubkey = root_key.verifying_key().to_bytes();
        Self {
            root_seed: root_key.to_bytes(),
            node_pubkey,
        }
    }

    /// Derive a purpose-keyed Ed25519 signing key via HKDF.
    fn derive(&self, purpose: &str) -> SigningKey {
        let hk = Hkdf::<Sha256>::new(None, &self.root_seed);
        let mut okm = [0u8; 32];
        hk.expand(purpose.as_bytes(), &mut okm)
            .expect("HKDF-SHA256 expand to 32 bytes cannot fail");
        let key = SigningKey::from_bytes(&okm);
        // Zeroize the intermediate key material
        okm.fill(0);
        key
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl IdentityProvider for NodeIdentityProvider {
    async fn identity_open(&self, purpose: &str) -> Result<Box<dyn SigningIdentity>> {
        let signing_key = self.derive(purpose);
        let pubkey = signing_key.verifying_key().to_bytes();
        Ok(Box::new(DerivedIdentity { signing_key, pubkey }))
    }

    fn resolve(&self, pubkey: &[u8; 32]) -> Subject {
        if pubkey == &self.node_pubkey {
            Subject::new("system")
        } else {
            Subject::anonymous()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[tokio::test]
    async fn same_purpose_same_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let id1 = provider.identity_open("test-purpose-v1").await.unwrap();
        let id2 = provider.identity_open("test-purpose-v1").await.unwrap();
        assert_eq!(id1.pubkey(), id2.pubkey());
    }

    #[tokio::test]
    async fn different_purpose_different_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);

        let id1 = provider.identity_open("purpose-a").await.unwrap();
        let id2 = provider.identity_open("purpose-b").await.unwrap();
        assert_ne!(id1.pubkey(), id2.pubkey());
    }

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

    #[test]
    fn resolve_node_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let provider = NodeIdentityProvider::new(&root);
        let node_pub = root.verifying_key().to_bytes();

        assert_eq!(provider.resolve(&node_pub).name(), Some("system"));
        assert_eq!(provider.resolve(&[0u8; 32]).name(), None); // anonymous
    }
}
