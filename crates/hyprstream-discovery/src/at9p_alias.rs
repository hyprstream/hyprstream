//! `did:web` / `did:key` → `did:at9p` aliasing resolver (#896 / D4; design
//! #879 Q4, #905 §2/§6).
//!
//! This is the **I/O half** of the D4 aliasing bridge. It composes the
//! untrusted [`CapsuleSource`] + GATE pipeline (D1 / #893) with the pure
//! authoritative-direction rule (`hyprstream_pds::at9p_alias`) to resolve a
//! classical DID to its authoritative `did:at9p` identity when — and only when
//! — the two are mutually `alsoKnownAs`-attested. Sibling to
//! [`crate::at9p_resolver::At9pResolver`]: where that turns a `did:at9p` into
//! dialable reach, this turns a classical DID + its at9p claim into the
//! authoritative at9p *identity* (and its assurance).
//!
//! # Fail-closed posture
//!
//! An authoritative identity is emitted **only** when:
//!
//! 1. the capsule named by the classical leg fetches and passes the full GATE
//!    pipeline (canon → hash → sig), and
//! 2. both legs name each other (mutual attestation).
//!
//! Any GATE failure, a classical leg pointing at a different capsule, or a
//! capsule that does not reciprocate the classical DID returns `Err` — never a
//! partial or one-way trust upgrade. See `hyprstream_pds::at9p_alias` for the
//! authoritative-direction rule and the two-leg strength table.

use std::sync::Arc;

use anyhow::Result;

use hyprstream_pds::at9p_alias::{resolve_authoritative_alias, AuthoritativeIdentity};
use hyprstream_pds::at9p_gate::verify_did_at9p;
use hyprstream_rpc::identity::Did;

use crate::at9p_resolver::CapsuleSource;

/// `did:web` / `did:key` → `did:at9p` aliasing resolver.
///
/// Wraps an untrusted [`CapsuleSource`] (the mainline locator is the production
/// implementation) and resolves a classical DID to its authoritative at9p
/// identity via mutual `alsoKnownAs` attestation + the GATE pipeline.
pub struct At9pAliasResolver {
    source: Arc<dyn CapsuleSource>,
}

impl At9pAliasResolver {
    /// Build an aliasing resolver over an untrusted capsule source.
    pub fn new(source: Arc<dyn CapsuleSource>) -> Self {
        Self { source }
    }

    /// Resolve `classical_did` to its authoritative `did:at9p` identity.
    ///
    /// `classical_aka_at9p` is the **classical leg's** claim — the `did:at9p`
    /// identifier the classical DID document names in its own `alsoKnownAs`
    /// (classical-signed, best-effort). This fetches the capsule for that at9p
    /// DID from the (untrusted) source, runs the full GATE pipeline over the
    /// bytes, and applies the mutual-attestation / authoritative-direction rule.
    ///
    /// Fails closed at the first GATE failure or any one-way / unattested
    /// claim. On success the at9p identity is authoritative at `PqHybrid`
    /// assurance (the capsule leg walked).
    pub async fn resolve_authoritative(
        &self,
        classical_did: &Did,
        classical_aka_at9p: &Did,
    ) -> Result<AuthoritativeIdentity> {
        let did_str = classical_aka_at9p.as_str();
        // The locator derives zero authority — GATE is what makes the bytes
        // trustworthy, not the source.
        let bytes = self.source.fetch_capsule(did_str).await?;
        // GATE — canon → hash → sig. The only place capsule authority is granted.
        let verified = verify_did_at9p(did_str, &bytes)?;
        // Authoritative-direction rule — mutual attestation or fail closed.
        resolve_authoritative_alias(classical_did, classical_aka_at9p, &verified)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::at9p_resolver::CapsuleSource;

    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
    };
    use hyprstream_pds::at9p_sign::sign_capsule;
    use hyprstream_rpc::auth::mac::Assurance;

    /// Serve a fixed capsule's bytes for any requested did.
    struct FixedSource {
        bytes: Vec<u8>,
    }

    #[async_trait::async_trait]
    impl CapsuleSource for FixedSource {
        async fn fetch_capsule(&self, _did: &str) -> Result<Vec<u8>> {
            Ok(self.bytes.clone())
        }
    }

    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer(tag: u8) -> Signer {
        let mut seed = [0u8; 32];
        seed[0] = tag;
        seed[31] = tag.wrapping_add(7);
        let ed_sk = SigningKey::from_bytes(&seed);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let keypair = HybridKeyPair::new(
            ed_sk.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .unwrap();
        Signer {
            ed_sk,
            pq_sk,
            keypair,
        }
    }

    /// A signed capsule that attests `aliases`, its bytes, and its did.
    fn signed_with_aliases(aliases: Vec<String>) -> (Vec<u8>, String) {
        let s = signer(5);
        let endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://node5").unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![s.keypair.clone()], vec![service]).unwrap();
        if !aliases.is_empty() {
            body.also_known_as = Some(aliases);
        }
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        let bytes = capsule.to_dag_cbor();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        (bytes, did)
    }

    #[tokio::test]
    async fn resolver_mutual_attestation_yields_authoritative_at9p() {
        let classical = Did::new("did:web:node.example".to_owned());
        let (bytes, did) = signed_with_aliases(vec![classical.as_str().to_owned()]);
        let at9p = Did::new(did);
        let resolver = At9pAliasResolver::new(Arc::new(FixedSource { bytes }));
        let res = resolver
            .resolve_authoritative(&classical, &at9p)
            .await
            .unwrap();
        assert_eq!(res.at9p_did, at9p);
        assert_eq!(res.classical_did, classical);
        assert_eq!(res.assurance, Assurance::PqHybrid);
    }

    #[tokio::test]
    async fn resolver_one_way_capsule_claim_fails_closed() {
        // Capsule attests alice, but we reach it claiming to be bob.
        let (bytes, did) = signed_with_aliases(vec!["did:web:alice.example".to_owned()]);
        let at9p = Did::new(did);
        let bob = Did::new("did:web:bob.example".to_owned());
        let resolver = At9pAliasResolver::new(Arc::new(FixedSource { bytes }));
        let err = resolver
            .resolve_authoritative(&bob, &at9p)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("does not name"), "{err}");
    }

    #[tokio::test]
    async fn resolver_classical_leg_pointing_elsewhere_fails_closed() {
        // Capsule genuinely attests the classical DID, but the classical leg
        // names a DIFFERENT valid at9p identity than the one the source serves.
        let classical = Did::new("did:web:node.example".to_owned());
        let (bytes, _did) = signed_with_aliases(vec![classical.as_str().to_owned()]);
        // A second, genuinely-valid capsule → a real distinct did:at9p cid512.
        let (_other_bytes, other_did) = signed_with_aliases(Vec::new());
        let wrong_at9p = Did::new(other_did);
        let resolver = At9pAliasResolver::new(Arc::new(FixedSource { bytes }));
        let err = resolver
            .resolve_authoritative(&classical, &wrong_at9p)
            .await
            .unwrap_err();
        // GATE hash-gate rejects (the served bytes hash to a different cid) before
        // the aliasing rule ever runs — the classical leg's claim is not believed.
        let msg = err.to_string();
        assert!(
            msg.contains("hash-gate") || msg.contains("does not name the verified capsule"),
            "expected GATE or aliasing rejection, got: {msg}"
        );
    }
}
