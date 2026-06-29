//! Compiled-policy distribution + signing (S4 / #570).
//!
//! The PolicyService (PDP authority) compiles policy into a [`CompiledPolicy`] (the TE
//! matrix plus the lattice generation it is valid for), serializes it canonically, hashes
//! it, and signs `(generation, policy_hash)` with the hybrid COSE composite (EdDSA +
//! ML-DSA-65; design §13a, §14). Nodes (PEPs) load the signed artifact and the
//! [`PolicyLoader`] **rejects any policy whose signature does not verify** and (per design
//! §14 / S5 containment backstop) whose hash does not match an approval.
//!
//! This module is the *transport/verification* boundary; the per-op hot path (`te` + `avc`)
//! never touches it. Signature verification happens **once at load**, never per op (design §7,
//! §13a "never verify-per-op").
//!
//! ## What S5 owns vs what this owns
//! - S5: the UCAN→TE compiler that *produces* the [`TeMatrix`] and the approval signature over
//!   `(canonical_UCAN_CID, bundle_hash)`.
//! - S4 (here): wrapping the matrix as a versioned, hashed, signed distributable; and the
//!   loader that rejects anything unsigned/mismatched before it can feed an evaluator.
//!
//! The actual COSE composite sign/verify lives in `hyprstream_rpc::crypto::cose_sign`; this
//! module abstracts it behind [`PolicySigner`] / [`PolicyVerifier`] so the hot-path crate has
//! no crypto dependency and so tests can use a trivial signer. The production wiring (using
//! `MlDsaSigningKeyStore` + Ed25519) is in [`cose`].

use crate::mac::lattice::{Lattice, LatticeCodecError};
use crate::mac::te::TeMatrix;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A compiled, distributable policy artifact: the TE matrix, **the S1 lattice it was
/// compiled against (embedded verbatim)**, and the generation that binds them. `generation`
/// ties the matrix to a specific lattice generation (S1) AND is the AVC's cache-invalidation
/// key — bumping it invalidates every cached decision.
///
/// ## S1 reconciliation (#570): embed the lattice, bind the version (desync-proof)
///
/// The signed policy carries the lattice's **canonical CBOR bytes** ([`Lattice::to_bytes`] —
/// the same codec the COSE signing path uses). Because those bytes are the closed
/// compartment vocabulary in bit-index order, every process that loads this policy
/// reconstructs the **identical** name↔bit map: same bytes in → identical bits out, so a
/// [`SecurityLabel`](crate::mac::lattice::SecurityLabel)'s compartment bits mean the same
/// thing in the compiler, the PDP, and every distributed AVC. The loader additionally
/// enforces `lattice.version().generation() == generation` — a compiled matrix is valid
/// ONLY against the lattice generation that minted its bit assignments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPolicy {
    /// Monotonic policy generation. Equals the embedded lattice's
    /// [`LatticeVersion::generation`](crate::mac::lattice::LatticeVersion::generation)
    /// (enforced at construction and re-checked on load).
    pub generation: u64,
    /// The compiled TE allow/escalate matrix (produced by S5's compiler).
    pub matrix: TeMatrix,
    /// The S1 lattice policy, serialized to its canonical CBOR byte form. Embedded so the
    /// PDP and every distributed AVC reconstruct an identical name↔bit vocabulary (no
    /// cross-process compartment-bit desync). Reconstruct via [`CompiledPolicy::lattice`].
    pub lattice_bytes: Vec<u8>,
}

impl CompiledPolicy {
    /// Compile a policy against an S1 [`Lattice`]. The generation is taken from the lattice
    /// version — they are definitionally equal (#570): the lattice that minted the
    /// compartment bits IS the policy generation. The lattice is embedded verbatim (its
    /// canonical CBOR bytes) so it travels inside the signature.
    pub fn new(matrix: TeMatrix, lattice: &Lattice) -> Self {
        Self {
            generation: lattice.version().generation(),
            matrix,
            lattice_bytes: lattice.to_bytes(),
        }
    }

    /// Reconstruct the embedded S1 [`Lattice`] from its canonical bytes. Fail-closed: a
    /// malformed / tampered vocabulary (duplicate names, over-width) is rejected rather than
    /// silently yielding a different lattice (the desync-proof property).
    pub fn lattice(&self) -> Result<Lattice, LatticeCodecError> {
        Lattice::from_bytes(&self.lattice_bytes)
    }

    /// Canonical bytes for hashing/signing. Deterministic: same policy → identical bytes
    /// (design §13a determinism). We sort the rule sets into a stable vec form first so the
    /// `HashSet` iteration order never leaks into the hash.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PolicyDistError> {
        // Re-serialize via a canonical, order-stable representation.
        let canonical = CanonicalPolicy::from(self);
        serde_json::to_vec(&canonical).map_err(|e| PolicyDistError::Encode(e.to_string()))
    }

    /// BLAKE3 content hash of the canonical bytes — the `policy_hash` the approval binds to.
    pub fn policy_hash(&self) -> Result<[u8; 32], PolicyDistError> {
        let bytes = self.canonical_bytes()?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

/// Order-stable view of a [`CompiledPolicy`] for canonical encoding. The `TeMatrix` stores
/// rules in `HashSet`s (O(1) hot-path lookup); for hashing we project to sorted vecs so the
/// hash is deterministic regardless of set iteration order. The embedded `lattice_bytes` are
/// already canonical (S1's deterministic CBOR), so the policy hash binds the exact lattice
/// vocabulary too — a tampered compartment map changes the hash and fails the signature.
#[derive(Serialize, Deserialize)]
struct CanonicalPolicy {
    generation: u64,
    allow: Vec<crate::mac::te::TeRule>,
    escalate: Vec<crate::mac::te::TeRule>,
    lattice_bytes: Vec<u8>,
}

impl From<&CompiledPolicy> for CanonicalPolicy {
    fn from(p: &CompiledPolicy) -> Self {
        // Reconstruct sorted rule vecs from the matrix. We can't read the private sets
        // directly, so we expose a sorted projection via TeMatrix below.
        let (mut allow, mut escalate) = p.matrix.sorted_rules();
        allow.sort();
        escalate.sort();
        Self {
            generation: p.generation,
            allow,
            escalate,
            lattice_bytes: p.lattice_bytes.clone(),
        }
    }
}

/// A signed, distributable policy: the canonical policy bytes + a detached composite
/// signature over `(generation || policy_hash)`. This is the wire artifact PolicyService
/// publishes and PEPs fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedPolicy {
    /// Canonical bytes of the [`CompiledPolicy`] (re-hashable at the loader; the loader does
    /// NOT trust a transmitted hash — it recomputes).
    pub policy_bytes: Vec<u8>,
    /// Detached COSE composite signature over the signing input (see [`signing_input`]).
    pub signature: Vec<u8>,
    /// Generation, duplicated out-of-band for quick routing; authoritative value is inside
    /// `policy_bytes` and is what gets signed.
    pub generation: u64,
}

/// The exact bytes that get signed: domain-separated `(generation, policy_hash)`. Binding the
/// generation prevents a rollback/replay of an older signed matrix under a new generation.
pub fn signing_input(generation: u64, policy_hash: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + 8 + 32);
    v.extend_from_slice(b"hs-mac-te"); // domain separation tag
    v.extend_from_slice(&generation.to_be_bytes());
    v.extend_from_slice(policy_hash);
    v
}

/// Abstraction over the authority's signing key (production = hybrid EdDSA+ML-DSA-65 COSE
/// composite). Implemented in [`cose`] over the existing crypto stack.
pub trait PolicySigner {
    /// Sign the signing-input bytes, returning a detached signature.
    fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, PolicyDistError>;
}

/// Abstraction over signature verification at the loader (production = composite verify with
/// `require_pq = true`, fail-closed).
pub trait PolicyVerifier {
    /// Verify `signature` over `signing_input`. Returns `Ok(())` on success, error otherwise.
    fn verify(&self, signing_input: &[u8], signature: &[u8]) -> Result<(), PolicyDistError>;
}

/// PolicyService side: compile → hash → sign → distributable artifact.
pub fn sign_policy<S: PolicySigner>(
    policy: &CompiledPolicy,
    signer: &S,
) -> Result<SignedPolicy, PolicyDistError> {
    let policy_bytes = policy.canonical_bytes()?;
    let hash = policy.policy_hash()?;
    let input = signing_input(policy.generation, &hash);
    let signature = signer.sign(&input)?;
    Ok(SignedPolicy {
        policy_bytes,
        signature,
        generation: policy.generation,
    })
}

/// An approval record (design §14 containment backstop): the human-approved binding of a
/// policy generation to its expected hash. The loader accepts ONLY a policy whose recomputed
/// hash matches an approval for its generation. Even an undetected compiler bug then can't
/// smuggle authority past review. S5 produces these (signed `(UCAN_CID, bundle_hash)`); here
/// we model the hash-match the loader enforces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyApproval {
    pub generation: u64,
    pub approved_hash: [u8; 32],
}

impl PolicyApproval {
    /// **S5 → S4 keystone (#571 M2, additive).** Convert a *verified* UCAN approval
    /// binding into the [`PolicyApproval`] this loader requires. This is the one
    /// wire that lets the S5 compiler's output reach the S4 loader: the reviewer
    /// signs a UCAN approval whose `bundle_hash` is exactly
    /// [`CompiledPolicy::policy_hash`]; once the signature is verified
    /// (`SignedApproval::verify` / `verify_binds`, hybrid COSE) the resulting
    /// [`ApprovalBinding`](hyprstream_rpc::auth::ucan::approval::ApprovalBinding)'s
    /// `(generation, bundle_hash)` ARE the `(generation, approved_hash)` the loader
    /// matches against — with the loader RECOMPUTING the hash itself (it never
    /// trusts a transmitted hash; see [`PolicyLoader::load`]).
    ///
    /// Pass only a binding you have already verified — borrowing the
    /// `&ApprovalBinding` returned by `SignedApproval::verify`/`verify_binds` makes
    /// that the natural call shape. No `did:key`/crypto happens here: this is a pure
    /// field projection of an already-trusted binding.
    pub fn from_verified(binding: &hyprstream_rpc::auth::ucan::approval::ApprovalBinding) -> Self {
        Self {
            generation: binding.generation,
            approved_hash: binding.bundle_hash,
        }
    }
}

/// PEP side: verify signature + (optionally) match against an approval, then yield the
/// [`CompiledPolicy`] ready to feed a [`crate::mac::te::LatticeTeEvaluator`]. Fail-closed:
/// any failure returns an error and NO policy is loaded (the PEP keeps denying).
pub struct PolicyLoader<V: PolicyVerifier> {
    verifier: V,
    /// Approvals indexed by generation. If empty, approval-matching is skipped (sig-only);
    /// production SHOULD populate this (S5).
    approvals: Vec<PolicyApproval>,
}

impl<V: PolicyVerifier> PolicyLoader<V> {
    pub fn new(verifier: V) -> Self {
        Self {
            verifier,
            approvals: Vec::new(),
        }
    }

    /// Register an approval the loader will require a hash-match against.
    pub fn with_approval(mut self, approval: PolicyApproval) -> Self {
        self.approvals.push(approval);
        self
    }

    /// Verify + load. Steps (all fail-closed):
    /// 1. Decode the canonical policy bytes.
    /// 2. Recompute the hash from the bytes (never trust a transmitted hash).
    /// 3. Verify the composite signature over `(generation, hash)`.
    /// 4. If approvals are registered, require a matching `(generation, hash)` approval.
    pub fn load(&self, signed: &SignedPolicy) -> Result<CompiledPolicy, PolicyDistError> {
        // 1. Decode.
        let canonical: CanonicalPolicy = serde_json::from_slice(&signed.policy_bytes)
            .map_err(|e| PolicyDistError::Decode(e.to_string()))?;
        let matrix = TeMatrix::new(
            canonical.allow.iter().copied().collect(),
            canonical.escalate.iter().copied().collect(),
        );
        // Reassemble preserving the embedded lattice bytes verbatim (do NOT re-derive the
        // generation here — we must round-trip the exact transmitted bytes to recompute the
        // same hash; the generation/version binding is checked explicitly below).
        let policy = CompiledPolicy {
            generation: canonical.generation,
            matrix,
            lattice_bytes: canonical.lattice_bytes,
        };

        // The out-of-band generation must match the signed-in generation (no mismatch routing).
        if policy.generation != signed.generation {
            return Err(PolicyDistError::GenerationMismatch {
                outer: signed.generation,
                inner: policy.generation,
            });
        }

        // S1 reconciliation (#570): the embedded lattice must reconstruct (fail-closed on a
        // tampered/over-width vocabulary) AND its version must equal the policy generation —
        // a matrix is only valid against the lattice generation that minted its bits.
        let lattice = policy
            .lattice()
            .map_err(|e| PolicyDistError::Decode(e.to_string()))?;
        let lattice_gen = lattice.version().generation();
        if lattice_gen != policy.generation {
            return Err(PolicyDistError::LatticeGenerationMismatch {
                policy: policy.generation,
                lattice: lattice_gen,
            });
        }

        // 2. Recompute hash from the decoded policy's OWN canonical bytes. If the transmitted
        //    bytes were canonical, this round-trips; if they were tampered into a non-canonical
        //    form, the recomputed hash diverges and signature/approval checks fail closed.
        let hash = policy.policy_hash()?;
        let input = signing_input(policy.generation, &hash);

        // 3. Signature verify (ONCE, at load — never per op).
        self.verifier.verify(&input, &signed.signature)?;

        // 4. Approval hash-match (containment backstop).
        if !self.approvals.is_empty() {
            let matched = self
                .approvals
                .iter()
                .any(|a| a.generation == policy.generation && a.approved_hash == hash);
            if !matched {
                return Err(PolicyDistError::NoMatchingApproval {
                    generation: policy.generation,
                });
            }
        }

        Ok(policy)
    }
}

/// Errors in compiled-policy distribution. All are load-time; the hot path never sees them.
#[derive(Error, Debug)]
pub enum PolicyDistError {
    #[error("policy encode failed: {0}")]
    Encode(String),
    #[error("policy decode failed: {0}")]
    Decode(String),
    #[error("signature verification failed: {0}")]
    BadSignature(String),
    #[error("no approval matches policy generation {generation} (hash mismatch — possible compiler tampering)")]
    NoMatchingApproval { generation: u64 },
    #[error("generation mismatch: outer={outer} inner={inner}")]
    GenerationMismatch { outer: u64, inner: u64 },
    #[error(
        "embedded lattice version {lattice} != policy generation {policy} (S1↔S4 bit-desync guard)"
    )]
    LatticeGenerationMismatch { policy: u64, lattice: u64 },
}

// ---------------------------------------------------------------------------------------
// Production crypto wiring over the existing hybrid COSE composite stack.
// Kept behind these adapters so `te`/`avc` stay crypto-free and tests can use a stub signer.
// ---------------------------------------------------------------------------------------

/// Production signer/verifier adapters over `hyprstream_rpc::crypto::cose_sign`.
pub mod cose {
    use super::{PolicyDistError, PolicySigner, PolicyVerifier};
    use ed25519_dalek::{SigningKey, VerifyingKey};
    use hyprstream_rpc::crypto::cose_sign::{sign_composite, verify_composite};
    use hyprstream_rpc::crypto::pq::{MlDsaSigningKey, MlDsaVerifyingKey};

    /// Domain-separation AAD for compiled-policy signatures (distinct from envelope/token
    /// AADs so a policy signature can never be confused for another message type).
    const POLICY_AAD: &[u8] = b"hs-mac-compiled-policy-v1";

    /// Hybrid (EdDSA + ML-DSA-65) signer for compiled policy. `pq` is `None` only when the
    /// crypto policy is Classical; production MUST be `Some` (design §14: not Ed25519-only).
    pub struct HybridPolicySigner<'a> {
        pub ed_sk: &'a SigningKey,
        pub pq_sk: Option<&'a MlDsaSigningKey>,
    }

    impl PolicySigner for HybridPolicySigner<'_> {
        fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, PolicyDistError> {
            sign_composite(self.ed_sk, self.pq_sk, signing_input, POLICY_AAD)
                .map_err(|e| PolicyDistError::Encode(e.to_string()))
        }
    }

    /// Hybrid verifier. `require_pq = true` enforces the ML-DSA outer layer (fail-closed if a
    /// classical-only signature is presented under a hybrid policy — design §13a).
    pub struct HybridPolicyVerifier<'a> {
        pub ed_vk: &'a VerifyingKey,
        pub pq_vk: Option<&'a MlDsaVerifyingKey>,
        pub require_pq: bool,
    }

    impl PolicyVerifier for HybridPolicyVerifier<'_> {
        fn verify(&self, signing_input: &[u8], signature: &[u8]) -> Result<(), PolicyDistError> {
            verify_composite(
                signature,
                self.ed_vk,
                self.pq_vk,
                signing_input,
                POLICY_AAD,
                self.require_pq,
            )
            .map(|_verified| ())
            .map_err(|e| PolicyDistError::BadSignature(e.to_string()))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mac::lattice::{Compartment, LatticeVersion};
    use crate::mac::te::{Action, ObjectType, SubjectType, TeRule};
    use std::collections::HashSet;

    fn rule(s: u32, o: u32, a: u32) -> TeRule {
        TeRule {
            subject_type: SubjectType(s),
            object_type: ObjectType(o),
            action: Action(a),
        }
    }

    /// A lattice at the given generation/version (the bit-vocabulary the policy binds).
    fn lattice(gen: u32) -> Lattice {
        Lattice::new(
            LatticeVersion(gen),
            [Compartment::new("pii"), Compartment::new("finance")],
        )
    }

    fn policy(gen: u64) -> CompiledPolicy {
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        allow.insert(rule(1, 2, 1));
        // Generation is taken from the lattice version (they are definitionally equal).
        CompiledPolicy::new(TeMatrix::from_allow(allow), &lattice(gen as u32))
    }

    /// A trivial HMAC-free stub signer for unit tests: signature = blake3(key || input).
    struct StubSigner {
        key: [u8; 32],
    }
    impl PolicySigner for StubSigner {
        fn sign(&self, input: &[u8]) -> Result<Vec<u8>, PolicyDistError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            Ok(h.finalize().as_bytes().to_vec())
        }
    }
    struct StubVerifier {
        key: [u8; 32],
    }
    impl PolicyVerifier for StubVerifier {
        fn verify(&self, input: &[u8], sig: &[u8]) -> Result<(), PolicyDistError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            if h.finalize().as_bytes().as_slice() == sig {
                Ok(())
            } else {
                Err(PolicyDistError::BadSignature("stub mismatch".into()))
            }
        }
    }

    #[test]
    fn canonical_hash_is_deterministic_across_set_order() {
        // Two matrices with the same rules inserted in different order must hash equal.
        let mut a = HashSet::new();
        a.insert(rule(1, 1, 1));
        a.insert(rule(2, 3, 4));
        let mut b = HashSet::new();
        b.insert(rule(2, 3, 4));
        b.insert(rule(1, 1, 1));
        let pa = CompiledPolicy::new(TeMatrix::from_allow(a), &lattice(5));
        let pb = CompiledPolicy::new(TeMatrix::from_allow(b), &lattice(5));
        assert_eq!(pa.policy_hash().unwrap(), pb.policy_hash().unwrap());
    }

    #[test]
    fn embedded_lattice_roundtrips_and_binds_generation() {
        // #570 desync-proof: the policy carries the lattice; reconstruction yields the same
        // bit vocabulary, and the version equals the generation.
        let p = policy(4);
        assert_eq!(p.generation, 4);
        let l = p.lattice().unwrap();
        assert_eq!(l.version().generation(), 4);
        assert_eq!(l.bit_of(&Compartment::new("finance")), Some(1));
    }

    #[test]
    fn loader_rejects_lattice_generation_mismatch() {
        // Hand-craft a policy whose embedded lattice version disagrees with the generation:
        // the loader must reject it (bit-desync guard), even with a valid signature.
        let key = [7u8; 32];
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        let mut p = CompiledPolicy::new(TeMatrix::from_allow(allow), &lattice(2));
        // Force a mismatch: claim generation 9 but the embedded lattice is version 2.
        p.generation = 9;
        let signed = sign_policy(&p, &StubSigner { key }).unwrap();
        let loader = PolicyLoader::new(StubVerifier { key });
        assert!(matches!(
            loader.load(&signed),
            Err(PolicyDistError::LatticeGenerationMismatch {
                policy: 9,
                lattice: 2
            })
        ));
    }

    #[test]
    fn sign_then_load_roundtrips() {
        let key = [7u8; 32];
        let signed = sign_policy(&policy(3), &StubSigner { key }).unwrap();
        let loader = PolicyLoader::new(StubVerifier { key });
        let loaded = loader.load(&signed).unwrap();
        assert_eq!(loaded.generation, 3);
        assert_eq!(loaded.matrix.allow_len(), 2);
    }

    #[test]
    fn tampered_policy_bytes_rejected() {
        let key = [7u8; 32];
        let mut signed = sign_policy(&policy(3), &StubSigner { key }).unwrap();
        // Flip a byte in the policy body — recomputed hash diverges → signature fails.
        signed.policy_bytes[0] ^= 0xFF;
        let loader = PolicyLoader::new(StubVerifier { key });
        assert!(
            loader.load(&signed).is_err(),
            "tampered policy must be rejected"
        );
    }

    #[test]
    fn wrong_key_signature_rejected() {
        let signed = sign_policy(&policy(3), &StubSigner { key: [1u8; 32] }).unwrap();
        let loader = PolicyLoader::new(StubVerifier { key: [2u8; 32] });
        assert!(matches!(
            loader.load(&signed),
            Err(PolicyDistError::BadSignature(_))
        ));
    }

    #[test]
    fn approval_hash_match_enforced() {
        let key = [7u8; 32];
        let p = policy(9);
        let hash = p.policy_hash().unwrap();
        let signed = sign_policy(&p, &StubSigner { key }).unwrap();

        // Loader with the CORRECT approval loads.
        let good = PolicyLoader::new(StubVerifier { key }).with_approval(PolicyApproval {
            generation: 9,
            approved_hash: hash,
        });
        assert!(good.load(&signed).is_ok());

        // Loader with a WRONG approval hash rejects, even though the signature verifies.
        let bad = PolicyLoader::new(StubVerifier { key }).with_approval(PolicyApproval {
            generation: 9,
            approved_hash: [0u8; 32],
        });
        assert!(matches!(
            bad.load(&signed),
            Err(PolicyDistError::NoMatchingApproval { generation: 9 })
        ));
    }

    #[test]
    fn generation_mismatch_rejected() {
        let key = [7u8; 32];
        let mut signed = sign_policy(&policy(3), &StubSigner { key }).unwrap();
        signed.generation = 4; // outer disagrees with the signed-in inner generation
        let loader = PolicyLoader::new(StubVerifier { key });
        assert!(matches!(
            loader.load(&signed),
            Err(PolicyDistError::GenerationMismatch { outer: 4, inner: 3 })
        ));
    }
}
