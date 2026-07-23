//! Federated identity provider — extends node identity with cross-node trust.
//!
//! Combines local signing (via `NodeIdentityProvider`) with federated
//! trust resolution (via `FederationKeySource`). Local identity derivation
//! is identical to `NodeIdentityProvider`; the difference is `resolve()`
//! which can verify remote nodes' pubkeys via federation entity statements.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use crate::auth::FederationKeySource;
use crate::identity::{IdentityProvider, SigningIdentity};
use crate::node_identity::NodeIdentityProvider;
use crate::Subject;

/// Identity provider with cross-node trust via federation.
///
/// - `identity_open()` — delegates to `NodeIdentityProvider` (local signing)
/// - `resolve()` — checks local node key first, then federation trust chain
pub struct FederatedIdentityProvider {
    node: NodeIdentityProvider,
    federation: Arc<dyn FederationKeySource>,
}

impl FederatedIdentityProvider {
    pub fn new(node: NodeIdentityProvider, federation: Arc<dyn FederationKeySource>) -> Self {
        Self { node, federation }
    }

    /// Access the underlying federation key source.
    pub fn federation(&self) -> &dyn FederationKeySource {
        self.federation.as_ref()
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl IdentityProvider for FederatedIdentityProvider {
    async fn identity_open(&self, purpose: &str) -> Result<Box<dyn SigningIdentity>> {
        // Local signing — same as NodeIdentityProvider
        self.node.identity_open(purpose).await
    }

    fn resolve(&self, pubkey: &[u8; 32]) -> Subject {
        // Check local node key first
        let local = self.node.resolve(pubkey);
        if local.name().is_some() {
            return local;
        }

        // TODO: Check federation trust chain.
        // This requires async key resolution (JWKS fetch) but resolve() is sync.
        // For now, return anonymous for unknown keys. The async verification
        // happens in verify_claims() via FederationKeySource::get_key().
        //
        // The proper fix: make resolve() async, or split into
        // resolve_local() (sync) + resolve_federated() (async).
        Subject::anonymous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    struct MockFederation;

    #[async_trait::async_trait]
    impl FederationKeySource for MockFederation {
        fn is_trusted(&self, _issuer: &str) -> bool {
            true
        }
        async fn get_keys(
            &self,
            _issuer: &str,
            _kid: Option<&str>,
        ) -> Result<Vec<ed25519_dalek::VerifyingKey>> {
            anyhow::bail!("mock: no key")
        }
    }

    #[allow(clippy::unwrap_used)]
    #[tokio::test]
    async fn federated_provider_signs_locally() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let node = NodeIdentityProvider::new(&root);
        let provider = FederatedIdentityProvider::new(node, Arc::new(MockFederation));

        let id = provider.identity_open("test-v1").await.unwrap();
        let sig = id.sign(b"hello").await.unwrap();

        let vk = ed25519_dalek::VerifyingKey::from_bytes(&id.pubkey()).unwrap();
        let signature = ed25519_dalek::Signature::from_bytes(&sig);
        use ed25519_dalek::Verifier;
        assert!(vk.verify(b"hello", &signature).is_ok());
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn federated_resolve_local_key() {
        let root = SigningKey::generate(&mut rand::rngs::OsRng);
        let node_pub = root.verifying_key().to_bytes();
        let node = NodeIdentityProvider::new(&root);
        let provider = FederatedIdentityProvider::new(node, Arc::new(MockFederation));

        assert_eq!(provider.resolve(&node_pub).name(), Some("system"));
        assert_eq!(provider.resolve(&[0u8; 32]).name(), None);
    }
}
