//! `hyprstream-pds` implementation of `hyprstream_rpc::identity_resolver::At9pCapsuleResolver`
//! — the bridge from the A4 GATE pipeline (#884) to the `did:at9p` admission and
//! resolver arms (D2/#894).
//!
//! `hyprstream-rpc` owns the admission gate and the method-dispatched identity
//! resolver, but the GATE (`canon → hash → sig`) and the capsule schema live here
//! in `hyprstream-pds` (which depends on rpc, so the dependency does not reverse).
//! The rpc crate therefore abstracts capsule verification behind the
//! [`At9pCapsuleResolver`] trait; this module is its real implementation, a thin
//! wrapper over [`crate::at9p_gate::verify_did_at9p`] that re-expresses the
//! capsule's content-verified subject keys in the rpc-local key types.
//!
//! # Provenance
//!
//! A [`VerifiedAt9pKeys`] returned here is only constructable after the full GATE
//! pipeline passed over canonical bytes whose BLAKE3-512 hash equals the claimed
//! `cid512` and whose composite signature (pinned Hybrid) verified against the
//! capsule's own primary subject key. The Ed25519/ML-DSA-65 pair is therefore
//! **content-verified**, never config-supplied — the D2/#894 provenance boundary.

use anyhow::{anyhow, Result};

use hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes;
use hyprstream_rpc::identity_resolver::{At9pCapsuleResolver, VerifiedAt9pKeys};

use crate::at9p::ED25519_PUBLIC_KEY_LEN;
use crate::at9p_gate::verify_did_at9p;

/// The production [`At9pCapsuleResolver`] — the A4 GATE pipeline over a fetched
/// or peer-presented capsule.
///
/// Construct with [`At9pGateResolver::new`]; the locator-backed `resolve` path is
/// Track C (#890) and returns `Err` until a locator is wired, but
/// [`verify_bytes`](At9pCapsuleResolver::verify_bytes) — the admission path the
/// peer presents bytes over — is fully functional today.
#[derive(Default)]
pub struct At9pGateResolver;

impl At9pGateResolver {
    /// Construct the GATE-backed capsule resolver.
    pub fn new() -> Self {
        Self
    }
}

impl At9pCapsuleResolver for At9pGateResolver {
    fn verify_bytes(&self, did: &str, capsule_bytes: &[u8]) -> Result<VerifiedAt9pKeys> {
        // The A4 GATE: canon → hash(H512 == cid512) → sig (pinned Hybrid). Only a
        // capsule that passes all three gates reaches the Ok, so the keys below
        // are content-verified binding material.
        let verified = verify_did_at9p(did, capsule_bytes)
            .map_err(|e| anyhow!("did:at9p {did} GATE rejected: {e}"))?;
        let capsule = verified.capsule();

        // The primary subject key is the hybrid genesis identity. Its Ed25519
        // half is not a NodeId/channel key, and GATE is not live possession.
        let subject = capsule
            .body
            .subject_keys
            .first()
            .ok_or_else(|| anyhow!("did:at9p {did} capsule carries no subject key"))?;

        let ed25519: [u8; 32] =
            subject.ed25519_pub.as_slice().try_into().map_err(|_| {
                anyhow!("ed25519 subject key is not {ED25519_PUBLIC_KEY_LEN} bytes")
            })?;

        let ml_dsa_65 = ml_dsa_vk_from_bytes(&subject.mldsa65_pub)?;

        Ok(VerifiedAt9pKeys::new_gate_verified(ed25519, ml_dsa_65))
    }

    fn resolve(&self, did: &str) -> Result<VerifiedAt9pKeys> {
        // The locator fetch (mainline DHT, Track C / #890) is not wired yet. Until
        // it is, the resolver path fails closed — admission (verify_bytes, with
        // peer-presented bytes) is the live D2 path.
        Err(anyhow!(
            "did:at9p {did}: mainline locator fetch is not wired (#890) — resolve fails closed"
        ))
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use crate::at9p::{
        Capsule, CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
        ED25519_SIGNATURE_LEN,
    };
    use crate::at9p_gate::DID_AT9P_PREFIX;
    use crate::at9p_sign::sign_capsule;

    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_rpc::admission::{
        admit_key_against_did, AdmissionTrustSurface, At9pAdmission, VerifiedApplicationSigner,
    };
    use hyprstream_rpc::envelope::{KeyedPqTrustStore, PqTrustStore};

    use anyhow::Result as AnyResult;
    use async_trait::async_trait;
    use serde_json::Value;

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

    fn body_for(s: &Signer, tag: u8) -> CapsuleBody {
        let endpoint = ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{tag}")).unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        CapsuleBody::new(vec![s.keypair.clone()], vec![service]).unwrap()
    }

    /// A fully self-signed capsule, its canonical bytes, and its `did:at9p:<cid>`.
    fn signed(tag: u8) -> (Capsule, Vec<u8>, String) {
        let s = signer(tag);
        let body = body_for(&s, tag);
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("{DID_AT9P_PREFIX}{}", capsule.cid512().unwrap());
        (capsule, bytes, did)
    }

    /// A did:web resolver that MUST NOT be called for the did:at9p admission arm.
    struct NeverResolve;
    #[async_trait]
    impl hyprstream_rpc::admission::DidDocResolve for NeverResolve {
        async fn resolve_doc(&self, did: &str) -> AnyResult<Value> {
            panic!("did:at9p admission must not resolve a DID document (called for {did})");
        }
    }

    #[test]
    fn gate_resolver_returns_verified_subject_keys() {
        let (capsule, bytes, did) = signed(1);
        let primary = capsule.body.subject_keys[0].clone();
        let verified = At9pGateResolver::new()
            .verify_bytes(&did, &bytes)
            .expect("GATE passes");
        assert_eq!(verified.ed25519().as_slice(), primary.ed25519_pub);
        assert_eq!(ml_dsa_vk_bytes(verified.ml_dsa_65()), primary.mldsa65_pub);
    }

    #[test]
    fn gate_resolver_rejects_bad_signature() {
        // Corrupt the Ed25519 signature (keep it schema-valid) and claim the
        // tampered capsule's own recomputed cid so canon/hash pass — only sig-gate
        // can reject.
        let (mut capsule, _bytes, _did) = signed(2);
        capsule.signatures.ed25519_signature = vec![0u8; ED25519_SIGNATURE_LEN];
        let bytes = capsule.to_dag_cbor().unwrap();
        let cid = Capsule::from_dag_cbor(&bytes).unwrap().cid512().unwrap();
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        assert!(At9pGateResolver::new().verify_bytes(&did, &bytes).is_err());
    }

    #[test]
    fn gate_resolver_rejects_wrong_cid() {
        // A genuine capsule claimed under a different (well-formed) cid fails
        // hash-gate.
        let (_c1, bytes1, did1) = signed(3);
        let did_other = signed(4).2;
        assert_ne!(did1, did_other);
        assert!(At9pGateResolver::new()
            .verify_bytes(&did_other, &bytes1)
            .is_err());
    }

    #[tokio::test]
    async fn admission_binds_hybrid_keys_from_gate_verified_capsule() {
        // End-to-end D2: a real self-signed capsule → GATE passes → the verified
        // ed25519→ml_dsa_65 pair is bound into KeyedPqTrustStore. The binding
        // provably comes from the capsule (the store was empty; the bound PQ key
        // equals the capsule's), not from any out-of-band config.
        let (capsule, bytes, did) = signed(5);
        let ed = capsule.body.subject_keys[0].ed25519_pub.clone();
        let primary_pq = capsule.body.subject_keys[0].mldsa65_pub.clone();
        let ed_arr: [u8; 32] = ed.as_slice().try_into().unwrap();

        let gate = At9pGateResolver::new();
        let mut store = KeyedPqTrustStore::new();
        assert!(store.is_empty());

        let admitted = admit_key_against_did(
            &NeverResolve,
            "https://peer.example",
            VerifiedApplicationSigner::pq_hybrid(ed_arr),
            AdmissionTrustSurface::Native,
            &did,
            None,
            Some(At9pAdmission {
                capsule_bytes: &bytes,
                gate: &gate,
                pq_store: &mut store,
            }),
        )
        .await
        .expect("verified capsule admits and binds");

        assert_eq!(admitted.did.as_deref(), Some(did.as_str()));
        assert_eq!(admitted.key, ed_arr);
        assert_eq!(store.len(), 1);
        let bound = store
            .ml_dsa_key_for(&ed_arr)
            .expect("hybrid binding installed");
        assert_eq!(ml_dsa_vk_bytes(&bound), primary_pq);
    }

    #[tokio::test]
    async fn admission_rejects_gate_failure_and_binds_nothing() {
        // Tampered signature → GATE rejects → admission fails closed and NOTHING
        // is bound into the trust store.
        let (mut capsule, _bytes, _did) = signed(6);
        capsule.signatures.ed25519_signature = vec![0u8; ED25519_SIGNATURE_LEN];
        let bytes = capsule.to_dag_cbor().unwrap();
        let cid = Capsule::from_dag_cbor(&bytes).unwrap().cid512().unwrap();
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        let ed: [u8; 32] = capsule.body.subject_keys[0]
            .ed25519_pub
            .as_slice()
            .try_into()
            .unwrap();

        let gate = At9pGateResolver::new();
        let mut store = KeyedPqTrustStore::new();
        let res = admit_key_against_did(
            &NeverResolve,
            "https://peer.example",
            VerifiedApplicationSigner::pq_hybrid(ed),
            AdmissionTrustSurface::Native,
            &did,
            None,
            Some(At9pAdmission {
                capsule_bytes: &bytes,
                gate: &gate,
                pq_store: &mut store,
            }),
        )
        .await;
        assert!(res.is_err(), "GATE failure must reject");
        assert!(store.is_empty(), "a rejected capsule must bind nothing");
    }

    #[test]
    fn resolve_fails_closed_until_locator_wired() {
        // The resolver path (locator fetch, #890) is not wired; it must fail
        // closed rather than silently degrade.
        let (_c, _bytes, did) = signed(7);
        assert!(At9pGateResolver::new().resolve(&did).is_err());
    }
}
