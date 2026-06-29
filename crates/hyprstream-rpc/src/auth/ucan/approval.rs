//! The signed **`(UCAN, bundle_hash)` approval binding** (S5 / #571).
//!
//! This is the containment backstop of the whole compiler design (#547, design
//! §14): an approved UCAN is cryptographically tied to the **hash of the bundle
//! it compiles to**, so a compiler bug can never produce a bundle that exceeds
//! what was reviewed. The reviewer approves a specific `(UCAN, lattice
//! generation, bundle_hash)` triple; the policy loader (S4
//! `compiled::PolicyLoader`) then accepts ONLY a compiled policy whose recomputed
//! hash matches an approval for its generation.
//!
//! ## What this binds, and why those three things
//!
//! - **`ucan_cid`** — the content hash (BLAKE3 of canonical CBOR) of the
//!   approved UCAN. Ties the approval to the *exact* delegation that was
//!   reviewed; re-issuing or mutating the UCAN changes the CID.
//! - **`generation`** — the [`LatticeVersion`] generation the bundle is valid
//!   for. Binding it prevents replaying an approval under a different lattice
//!   version whose compartment bits mean something else (the same desync guard
//!   S4 enforces; design §14 crypto/policy-agility).
//! - **`bundle_hash`** — the `[u8; 32]` BLAKE3 hash of the compiled bundle. This
//!   is **exactly** the value S4's `CompiledPolicy::policy_hash()` produces and
//!   `compiled::PolicyApproval::approved_hash` consumes, so an
//!   [`ApprovalBinding`] minted here drops straight into the S4 loader once
//!   bundle emission (milestone 2) is wired.
//!
//! ## Signing: hybrid COSE, NEVER Ed25519-only
//!
//! The binding is signed with the project's hybrid post-quantum COSE composite
//! (EdDSA + ML-DSA-65) via [`crate::crypto::cose_sign::sign_composite`] and
//! verified with `require_pq = true`. A classical-only signature is rejected
//! (fail-closed) — an approval is a long-lived authority artifact and is exactly
//! the kind of confidentiality/integrity-critical object the hybrid-PQC plan
//! requires to resist harvest-now-decrypt-later forgery.

use crate::crypto::cose_sign::{sign_composite, verify_composite};
use crate::crypto::pq::{MlDsaSigningKey, MlDsaVerifyingKey};
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::token::Ucan;

/// Domain-separation AAD for approval-binding signatures. Distinct from the UCAN
/// payload AAD (`hs-mac-ucan-payload-v1`), the compiled-policy AAD
/// (`hs-mac-compiled-policy-v1`), and envelope AADs, so an approval signature can
/// never be confused with — or replayed as — any other signed object.
pub const APPROVAL_AAD: &[u8] = b"hs-mac-ucan-approval-v1";

/// BLAKE3 content id of a UCAN: hash of its canonical CBOR encoding. The same
/// codec the rest of the MAC stack uses, so the CID is stable across processes.
pub fn ucan_cid(ucan: &Ucan) -> Result<[u8; 32], ApprovalError> {
    let bytes = ucan
        .to_cbor()
        .map_err(|e| ApprovalError::Encode(e.to_string()))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// The reviewed binding: the approved UCAN's CID, the lattice generation it is
/// valid for, and the hash of the bundle it compiles to. These are the bytes the
/// hybrid-COSE signature covers (see [`ApprovalBinding::signing_payload`]).
///
/// `bundle_hash` is intentionally a raw `[u8; 32]` — the exact type S4's
/// `CompiledPolicy::policy_hash()` returns — so milestone-2 bundle emission feeds
/// its output straight in with no conversion, and a verified binding yields an
/// S4 `compiled::PolicyApproval { generation, approved_hash: bundle_hash }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalBinding {
    /// BLAKE3 CID of the approved UCAN ([`ucan_cid`]).
    pub ucan_cid: [u8; 32],
    /// The lattice/policy generation this bundle is valid for
    /// ([`LatticeVersion::generation`]). Bound so the approval cannot be replayed
    /// under a different compartment vocabulary.
    pub generation: u64,
    /// BLAKE3 hash of the compiled bundle (== S4 `CompiledPolicy::policy_hash()`).
    pub bundle_hash: [u8; 32],
}

impl ApprovalBinding {
    /// Construct a binding from an approved UCAN, the generation it targets, and
    /// the hash of the bundle it compiled to.
    pub fn new(ucan: &Ucan, generation: u64, bundle_hash: [u8; 32]) -> Result<Self, ApprovalError> {
        Ok(Self {
            ucan_cid: ucan_cid(ucan)?,
            generation,
            bundle_hash,
        })
    }

    /// The exact, domain-separated bytes the signature covers:
    /// `b"hs-mac-ucan-approval" || ucan_cid || generation_be || bundle_hash`.
    /// Deterministic and length-prefixed by fixed field widths, so the layout is
    /// unambiguous.
    pub fn signing_payload(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(19 + 32 + 8 + 32);
        v.extend_from_slice(b"hs-mac-ucan-approval"); // intra-payload tag (belt-and-braces with AAD)
        v.extend_from_slice(&self.ucan_cid);
        v.extend_from_slice(&self.generation.to_be_bytes());
        v.extend_from_slice(&self.bundle_hash);
        v
    }
}

/// A signed approval: the binding plus a detached hybrid-COSE composite
/// signature over its [`ApprovalBinding::signing_payload`]. This is the artifact
/// a reviewer/authority emits and the policy loader consumes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedApproval {
    /// The reviewed binding (re-derivable + re-hashable at the verifier).
    pub binding: ApprovalBinding,
    /// Detached hybrid-COSE composite signature (EdDSA + ML-DSA-65).
    pub signature: Vec<u8>,
}

impl SignedApproval {
    /// Sign an approval binding with the authority's HYBRID keypair. `pq_sk`
    /// MUST be `Some` — an approval is never Ed25519-only (design §14). Passing
    /// the classical key without a PQ key is rejected fail-closed so a misuse
    /// can't silently downgrade the suite.
    pub fn sign(
        binding: ApprovalBinding,
        ed_sk: &SigningKey,
        pq_sk: &MlDsaSigningKey,
    ) -> Result<Self, ApprovalError> {
        let payload = binding.signing_payload();
        let signature = sign_composite(ed_sk, Some(pq_sk), &payload, APPROVAL_AAD)
            .map_err(|e| ApprovalError::Sign(e.to_string()))?;
        Ok(Self { binding, signature })
    }

    /// Verify the approval under the HYBRID policy (`require_pq = true`): both the
    /// EdDSA and the anchored ML-DSA-65 layers must verify against the authority's
    /// keys, over the binding's signing payload. Fail-closed: a stripped PQ layer,
    /// a wrong key, or a tampered binding all reject.
    ///
    /// On success returns the verified [`ApprovalBinding`] (so the caller does not
    /// re-trust `self.binding` without having checked the signature).
    pub fn verify(
        &self,
        ed_vk: &VerifyingKey,
        pq_vk: &MlDsaVerifyingKey,
    ) -> Result<&ApprovalBinding, ApprovalError> {
        let payload = self.binding.signing_payload();
        verify_composite(
            &self.signature,
            ed_vk,
            Some(pq_vk),
            &payload,
            APPROVAL_AAD,
            true,
        )
        .map_err(|e| ApprovalError::Verify(e.to_string()))?;
        Ok(&self.binding)
    }

    /// Verify the approval AND confirm it binds the given UCAN, generation, and
    /// bundle hash. This is the check the policy loader runs: it has the compiled
    /// bundle's `(generation, bundle_hash)` and the approved UCAN in hand and must
    /// confirm the signed approval covers exactly them. Fail-closed on any
    /// mismatch.
    pub fn verify_binds(
        &self,
        ed_vk: &VerifyingKey,
        pq_vk: &MlDsaVerifyingKey,
        ucan: &Ucan,
        generation: u64,
        bundle_hash: &[u8; 32],
    ) -> Result<(), ApprovalError> {
        let binding = self.verify(ed_vk, pq_vk)?;
        let cid = ucan_cid(ucan)?;
        if binding.ucan_cid != cid {
            return Err(ApprovalError::UcanMismatch);
        }
        if binding.generation != generation {
            return Err(ApprovalError::GenerationMismatch {
                approved: binding.generation,
                bundle: generation,
            });
        }
        if &binding.bundle_hash != bundle_hash {
            return Err(ApprovalError::BundleHashMismatch);
        }
        Ok(())
    }
}

/// Errors from approval construction / signing / verification. All fail-closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalError {
    /// UCAN encode failure while computing the CID.
    Encode(String),
    /// Hybrid-COSE signing failed.
    Sign(String),
    /// Hybrid-COSE verification failed (bad signature, stripped PQ layer, etc.).
    Verify(String),
    /// The signed approval does not bind the presented UCAN.
    UcanMismatch,
    /// The approval's generation does not match the bundle's generation.
    GenerationMismatch { approved: u64, bundle: u64 },
    /// The approval's bundle hash does not match the presented bundle.
    BundleHashMismatch,
}

impl fmt::Display for ApprovalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApprovalError::Encode(e) => write!(f, "approval UCAN encode failed: {e}"),
            ApprovalError::Sign(e) => write!(f, "approval signing failed: {e}"),
            ApprovalError::Verify(e) => write!(f, "approval verification failed: {e}"),
            ApprovalError::UcanMismatch => {
                write!(f, "signed approval does not bind the presented UCAN")
            }
            ApprovalError::GenerationMismatch { approved, bundle } => write!(
                f,
                "approval generation {approved} != bundle generation {bundle}"
            ),
            ApprovalError::BundleHashMismatch => {
                write!(
                    f,
                    "approval bundle hash does not match the presented bundle"
                )
            }
        }
    }
}

impl std::error::Error for ApprovalError {}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::token::test_support::*;
    use super::*;

    /// A fresh authority hybrid keypair + a sample approved UCAN.
    fn setup() -> (TestIdentity, Ucan, u64, [u8; 32]) {
        let authority = TestIdentity::generate();
        let issuer = TestIdentity::generate();
        let ucan = signed_ucan(
            &issuer,
            &issuer.did(),
            vec![cap("mac://model/*", "infer")],
            vec![],
        );
        let generation = 7u64;
        // Stand-in bundle hash (milestone 2 supplies the real CompiledPolicy::policy_hash()).
        let bundle_hash = *blake3::hash(b"compiled-bundle-bytes").as_bytes();
        (authority, ucan, generation, bundle_hash)
    }

    #[test]
    fn ucan_cid_is_stable_and_content_sensitive() {
        let issuer = TestIdentity::generate();
        let a = signed_ucan(&issuer, &issuer.did(), vec![cap("mac://x", "y")], vec![]);
        assert_eq!(ucan_cid(&a).unwrap(), ucan_cid(&a).unwrap());
        let b = signed_ucan(&issuer, &issuer.did(), vec![cap("mac://x", "z")], vec![]);
        assert_ne!(ucan_cid(&a).unwrap(), ucan_cid(&b).unwrap());
    }

    #[test]
    fn sign_then_verify_roundtrips() {
        let (auth, ucan, generation, bundle_hash) = setup();
        let binding = ApprovalBinding::new(&ucan, generation, bundle_hash).unwrap();
        let signed = SignedApproval::sign(binding.clone(), &auth.ed_sk, &auth.pq_sk).unwrap();
        let verified = signed.verify(&auth.ed_vk, &auth.pq_vk).unwrap();
        assert_eq!(verified, &binding);
    }

    #[test]
    fn verify_rejects_wrong_authority_key() {
        let (auth, ucan, generation, bundle_hash) = setup();
        let other = TestIdentity::generate();
        let binding = ApprovalBinding::new(&ucan, generation, bundle_hash).unwrap();
        let signed = SignedApproval::sign(binding, &auth.ed_sk, &auth.pq_sk).unwrap();
        // Wrong PQ key.
        assert!(matches!(
            signed.verify(&auth.ed_vk, &other.pq_vk),
            Err(ApprovalError::Verify(_))
        ));
        // Wrong Ed key.
        assert!(matches!(
            signed.verify(&other.ed_vk, &auth.pq_vk),
            Err(ApprovalError::Verify(_))
        ));
    }

    #[test]
    fn verify_rejects_tampered_binding() {
        let (auth, ucan, generation, bundle_hash) = setup();
        let binding = ApprovalBinding::new(&ucan, generation, bundle_hash).unwrap();
        let mut signed = SignedApproval::sign(binding, &auth.ed_sk, &auth.pq_sk).unwrap();
        // Flip the bundle hash after signing.
        signed.binding.bundle_hash[0] ^= 0xFF;
        assert!(matches!(
            signed.verify(&auth.ed_vk, &auth.pq_vk),
            Err(ApprovalError::Verify(_))
        ));
    }

    #[test]
    fn verify_binds_confirms_exact_triple() {
        let (auth, ucan, generation, bundle_hash) = setup();
        let binding = ApprovalBinding::new(&ucan, generation, bundle_hash).unwrap();
        let signed = SignedApproval::sign(binding, &auth.ed_sk, &auth.pq_sk).unwrap();

        // Correct triple binds.
        assert!(signed
            .verify_binds(&auth.ed_vk, &auth.pq_vk, &ucan, generation, &bundle_hash)
            .is_ok());

        // Wrong generation.
        assert!(matches!(
            signed.verify_binds(
                &auth.ed_vk,
                &auth.pq_vk,
                &ucan,
                generation + 1,
                &bundle_hash
            ),
            Err(ApprovalError::GenerationMismatch { .. })
        ));

        // Wrong bundle hash.
        let wrong = *blake3::hash(b"different-bundle").as_bytes();
        assert!(matches!(
            signed.verify_binds(&auth.ed_vk, &auth.pq_vk, &ucan, generation, &wrong),
            Err(ApprovalError::BundleHashMismatch)
        ));

        // Wrong UCAN.
        let other_issuer = TestIdentity::generate();
        let other_ucan = signed_ucan(
            &other_issuer,
            &other_issuer.did(),
            vec![cap("mac://q", "r")],
            vec![],
        );
        assert!(matches!(
            signed.verify_binds(
                &auth.ed_vk,
                &auth.pq_vk,
                &other_ucan,
                generation,
                &bundle_hash
            ),
            Err(ApprovalError::UcanMismatch)
        ));
    }

    #[test]
    fn signing_payload_binds_all_three_fields() {
        // Distinct (cid, generation, bundle_hash) → distinct signing payloads.
        let issuer = TestIdentity::generate();
        let ucan = signed_ucan(&issuer, &issuer.did(), vec![cap("mac://x", "y")], vec![]);
        let b1 = ApprovalBinding::new(&ucan, 1, [0u8; 32]).unwrap();
        let mut b2 = b1.clone();
        b2.generation = 2;
        let mut b3 = b1.clone();
        b3.bundle_hash = [9u8; 32];
        assert_ne!(b1.signing_payload(), b2.signing_payload());
        assert_ne!(b1.signing_payload(), b3.signing_payload());
    }
}
