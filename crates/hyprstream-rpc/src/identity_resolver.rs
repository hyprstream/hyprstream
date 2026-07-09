//! Method-dispatched [`IdentityResolver`] — the real `Did → IdentityKeys` binding (#579).
//!
//! This is the single, non-mirror implementation of the canonical
//! [`crate::identity::IdentityResolver`] contract. It replaces the fixture /
//! test-double resolvers that stood in for it while the #549 perimeter gateway
//! and #137 admission gate were built against the interface.
//!
//! # Division of labor
//!
//! The resolver owns **verification + assurance derivation**; it does *not* own
//! object labelling, the `KeyedPqTrustStore.bind` mesh anchor (#894), or any
//! perimeter policy — those consumers are generic over the trait and unchanged.
//!
//! # Method dispatch
//!
//! [`resolve_identity_keys`](IdentityResolver::resolve_identity_keys) branches on
//! the DID method:
//!
//! - **`did:web`** — resolve the DID document and extract its verification
//!   methods. An Ed25519 VM (`#mesh` / `#iroh`) is the classical anchor; a
//!   verified ML-DSA-65 VM (`#mesh-pq`, multicodec `0x1211`) is the PQ anchor.
//!   Both present ⇒ [`Assurance::PqHybrid`] (a did:web we operate); Ed25519-only ⇒
//!   [`Assurance::Classical`] (a third-party did:web). No Ed25519 VM ⇒ nothing to
//!   bind ⇒ `Err` (fail-closed).
//! - **`did:key`** — a single Ed25519 key decoded from the DID string itself; no
//!   fetch. A single classical key can never be hybrid, so the honest ceiling is
//!   [`Assurance::Classical`].
//! - **`did:at9p`** — reserved for the capsule GATE pipeline (#879/#880 D2). This
//!   resolver does **not** verify capsules; the arm returns `Err` (fail-closed)
//!   with a clear TODO(#894) marker until that epic fills it.
//! - **any other method** — `Err` (fail-closed).
//!
//! # Fail-closed
//!
//! Per the [`IdentityResolver`] contract, every inability to establish key
//! material — an unknown method, a fetch failure, a malformed document, a
//! did:web with no Ed25519 VM — returns `Err`. A resolver **never** returns a
//! default-`Unverified` `Ok`.
//!
//! # Sync trait over an async fetch
//!
//! The contract is synchronous and enrollment happens **off the auth path**
//! (once, at session establishment), so did:web document retrieval is abstracted
//! behind a synchronous [`DidDocumentProvider`]. This keeps the decode +
//! assurance-derivation core pure and testable with an injected fixture (no live
//! network), mirroring the injected-fetcher pattern in [`crate::did_web`]. The
//! async→sync bridge to the live [`crate::did_web::HttpDidDocFetcher`] is a
//! wiring concern for the accept-path integration (there is no production
//! construction site of the perimeter gateway today).

use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::auth::mac::Assurance;
use crate::did_web::{
    did_key_to_ed25519, verification_method_ed25519_keys, verification_method_ml_dsa_65_keys,
};
use crate::identity::{Did, IdentityKeys, IdentityResolver};

/// A synchronous DID-document provider for the did:web arm.
///
/// Abstracted so the resolver's decode/assurance core is testable with an
/// injected fixture (no network). Enrollment is off the auth path, so a blocking
/// implementation is acceptable; the trait keeps that policy out of the resolver.
pub trait DidDocumentProvider: Send + Sync {
    /// Fetch and parse the DID document for `did` (already known to be `did:web`).
    fn document(&self, did: &str) -> Result<Value>;
}

/// The real, method-dispatched [`IdentityResolver`] (#579).
///
/// Generic over a [`DidDocumentProvider`] so the did:web fetch is injectable
/// (fixture in tests, blocking HTTPS fetcher in production wiring).
pub struct MethodDispatchResolver<P: DidDocumentProvider> {
    docs: P,
}

impl<P: DidDocumentProvider> MethodDispatchResolver<P> {
    /// Construct a resolver over a DID-document provider.
    pub fn new(docs: P) -> Self {
        Self { docs }
    }

    /// The did:web arm: resolve the document and derive key material + assurance.
    ///
    /// - Ed25519 VM present + ML-DSA-65 VM present ⇒ `PqHybrid`.
    /// - Ed25519 VM present, no ML-DSA-65 VM ⇒ `Classical`.
    /// - No Ed25519 VM ⇒ `Err` (fail-closed: nothing to anchor).
    fn resolve_did_web(&self, did: &Did) -> Result<IdentityKeys> {
        let doc = self
            .docs
            .document(did.as_str())
            .map_err(|e| anyhow!("did:web {did} document did not resolve: {e}"))?;

        // First Ed25519 VM is the classical anchor (the `#mesh` / `#iroh` key).
        // A did:web with no Ed25519 VM cannot be bound (the PQ trust store is
        // keyed BY the Ed25519 identity), so fail closed.
        let ed25519 = verification_method_ed25519_keys(&doc).into_iter().next().ok_or_else(|| {
            anyhow!("did:web {did}: no Ed25519 verificationMethod to anchor — fail-closed")
        })?;

        // A verified ML-DSA-65 VM (`#mesh-pq`) upgrades the edge to PqHybrid; its
        // absence is the ordinary classical case, not an error.
        let ml_dsa_65 = verification_method_ml_dsa_65_keys(&doc).into_iter().next();

        let assurance =
            if ml_dsa_65.is_some() { Assurance::PqHybrid } else { Assurance::Classical };

        Ok(IdentityKeys { ed25519: Some(ed25519), ml_dsa_65, assurance })
    }

    /// The did:key arm: a single self-certifying Ed25519 key, no fetch.
    ///
    /// A `did:key` encodes exactly one key; a single classical key can never be
    /// hybrid, so assurance is `Classical` (the honest ceiling).
    fn resolve_did_key(&self, did: &Did) -> Result<IdentityKeys> {
        let ed25519 = did_key_to_ed25519(did.as_str())
            .map_err(|e| anyhow!("did:key {did} is not a valid Ed25519 identity: {e}"))?;
        Ok(IdentityKeys {
            ed25519: Some(ed25519),
            ml_dsa_65: None,
            assurance: Assurance::Classical,
        })
    }
}

impl<P: DidDocumentProvider> IdentityResolver for MethodDispatchResolver<P> {
    fn resolve_identity_keys(&self, did: &Did) -> Result<IdentityKeys> {
        if did.is_did_web() {
            self.resolve_did_web(did)
        } else if did.is_did_key() {
            self.resolve_did_key(did)
        } else if did.is_did_at9p() {
            // Reserved for the capsule GATE pipeline (#879/#880 D2). This
            // resolver does not verify capsules; fail closed until #894 fills it.
            Err(anyhow!(
                "did:at9p {did}: capsule verification is not implemented here — reserved for #894 (fail-closed)"
            ))
        } else {
            // Unknown DID method — fail closed (never a default-Unverified Ok).
            Err(anyhow!("unsupported DID method for {did}: no resolver arm — fail-closed"))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::pq::{ml_dsa_sk_to_vk_bytes, ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use crate::did_key::{ed25519_to_did_key, MULTICODEC_ED25519_PUB, MULTICODEC_ML_DSA_65_PUB};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use serde_json::json;

    fn rand_ed25519() -> [u8; 32] {
        SigningKey::generate(&mut OsRng).verifying_key().to_bytes()
    }

    fn encode_multikey(raw: &[u8], codec: [u8; 2]) -> String {
        let mut payload = Vec::with_capacity(2 + raw.len());
        payload.extend_from_slice(&codec);
        payload.extend_from_slice(raw);
        format!("z{}", bs58::encode(payload).into_string())
    }

    /// A DID document with an Ed25519 `#mesh` VM and, optionally, an ML-DSA-65
    /// `#mesh-pq` VM.
    fn did_doc(ed: &[u8; 32], ml_dsa_vk_bytes: Option<&[u8]>) -> Value {
        let mut vms = vec![json!({
            "id": "did:web:peer.example#mesh",
            "type": "Multikey",
            "controller": "did:web:peer.example",
            "publicKeyMultibase": encode_multikey(ed, MULTICODEC_ED25519_PUB),
        })];
        if let Some(pq) = ml_dsa_vk_bytes {
            vms.push(json!({
                "id": "did:web:peer.example#mesh-pq",
                "type": "Multikey",
                "controller": "did:web:peer.example",
                "publicKeyMultibase": encode_multikey(pq, MULTICODEC_ML_DSA_65_PUB),
            }));
        }
        json!({
            "@context": ["https://www.w3.org/ns/did/v1"],
            "id": "did:web:peer.example",
            "verificationMethod": vms,
            "service": [],
        })
    }

    /// Fixture provider returning a fixed document.
    struct FixtureDocs(Value);
    impl DidDocumentProvider for FixtureDocs {
        fn document(&self, _did: &str) -> Result<Value> {
            Ok(self.0.clone())
        }
    }

    /// Provider that always fails — exercises the fail-closed fetch path.
    struct FailingDocs;
    impl DidDocumentProvider for FailingDocs {
        fn document(&self, did: &str) -> Result<Value> {
            Err(anyhow!("simulated fetch failure for {did}"))
        }
    }

    /// A provider that MUST NOT be called (did:key / did:at9p never fetch).
    struct NeverDocs;
    impl DidDocumentProvider for NeverDocs {
        fn document(&self, did: &str) -> Result<Value> {
            panic!("resolver must not fetch a document for {did}");
        }
    }

    const DID_WEB: &str = "did:web:peer.example";

    #[test]
    fn did_web_ed25519_only_is_classical() {
        let ed = rand_ed25519();
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, None)));
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("ed25519-only did:web resolves");
        assert_eq!(keys.assurance, Assurance::Classical);
        assert_eq!(keys.ed25519, Some(ed));
        assert!(keys.ml_dsa_65.is_none());
    }

    #[test]
    fn did_web_with_ml_dsa_is_pqhybrid() {
        let ed = rand_ed25519();
        let (_sk, vk) = ml_dsa_generate_keypair();
        let pq_bytes = ml_dsa_vk_bytes(&vk);
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, Some(&pq_bytes))));
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("hybrid did:web resolves");
        assert_eq!(keys.assurance, Assurance::PqHybrid);
        assert_eq!(keys.ed25519, Some(ed));
        // The resolved PQ key is exactly the published one.
        let resolved = keys.ml_dsa_65.expect("PQ anchor present");
        assert_eq!(ml_dsa_vk_bytes(&resolved), pq_bytes);
    }

    #[test]
    fn did_web_derives_pq_key_from_signing_key_roundtrip() {
        // A mesh-derived ML-DSA key (as node_identity produces) round-trips
        // through the VM extractor into a usable verifying key.
        let ed = rand_ed25519();
        let peer_ed = SigningKey::generate(&mut OsRng);
        let peer_pq_sk = crate::node_identity::derive_mesh_mldsa_key(&peer_ed);
        let pq_bytes = ml_dsa_sk_to_vk_bytes(&peer_pq_sk);
        let resolver = MethodDispatchResolver::new(FixtureDocs(did_doc(&ed, Some(&pq_bytes))));
        let keys = resolver
            .resolve_identity_keys(&Did::new(DID_WEB.to_owned()))
            .expect("hybrid did:web resolves");
        assert_eq!(keys.assurance, Assurance::PqHybrid);
        assert_eq!(ml_dsa_vk_bytes(&keys.ml_dsa_65.unwrap()), pq_bytes);
    }

    #[test]
    fn did_web_no_ed25519_vm_fails_closed() {
        // A doc with only an ML-DSA VM (no Ed25519 anchor) cannot be bound.
        let (_sk, vk) = ml_dsa_generate_keypair();
        let doc = json!({
            "id": DID_WEB,
            "verificationMethod": [{
                "id": "did:web:peer.example#mesh-pq",
                "publicKeyMultibase": encode_multikey(&ml_dsa_vk_bytes(&vk), MULTICODEC_ML_DSA_65_PUB),
            }],
        });
        let resolver = MethodDispatchResolver::new(FixtureDocs(doc));
        assert!(resolver.resolve_identity_keys(&Did::new(DID_WEB.to_owned())).is_err());
    }

    #[test]
    fn did_web_empty_doc_fails_closed() {
        let doc = json!({ "id": DID_WEB, "verificationMethod": [] });
        let resolver = MethodDispatchResolver::new(FixtureDocs(doc));
        assert!(resolver.resolve_identity_keys(&Did::new(DID_WEB.to_owned())).is_err());
    }

    #[test]
    fn did_web_fetch_failure_fails_closed() {
        let resolver = MethodDispatchResolver::new(FailingDocs);
        assert!(resolver.resolve_identity_keys(&Did::new(DID_WEB.to_owned())).is_err());
    }

    #[test]
    fn did_key_is_classical_without_fetch() {
        let ed = rand_ed25519();
        let did = ed25519_to_did_key(&ed);
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let keys = resolver
            .resolve_identity_keys(&Did::new(did))
            .expect("did:key resolves without fetch");
        assert_eq!(keys.assurance, Assurance::Classical);
        assert_eq!(keys.ed25519, Some(ed));
        assert!(keys.ml_dsa_65.is_none());
    }

    #[test]
    fn did_key_malformed_fails_closed() {
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let did = Did::new("did:key:zNotAValidEd25519Multikey".to_owned());
        assert!(resolver.resolve_identity_keys(&did).is_err());
    }

    #[test]
    fn did_at9p_arm_is_reserved_and_fails_closed() {
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let did = Did::new("did:at9p:cid512abcdef".to_owned());
        // `IdentityKeys` is not `Debug` (the ML-DSA key type isn't), so match on
        // the result rather than `unwrap_err`.
        match resolver.resolve_identity_keys(&did) {
            Ok(_) => panic!("did:at9p must fail closed (reserved for #894)"),
            Err(err) => {
                assert!(err.to_string().contains("#894"), "reserved arm should point at #894: {err}");
            }
        }
    }

    #[test]
    fn unknown_method_fails_closed() {
        let resolver = MethodDispatchResolver::new(NeverDocs);
        let did = Did::new("did:plc:unsupported".to_owned());
        assert!(resolver.resolve_identity_keys(&did).is_err());
    }
}
