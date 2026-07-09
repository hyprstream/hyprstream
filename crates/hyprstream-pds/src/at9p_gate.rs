//! GATE verification pipeline for `did:at9p` genesis capsules (#884, design
//! #879 §5.6).
//!
//! Given the claimed self-certifying identifier `did:at9p:<cid512>` and the raw
//! bytes fetched for it from some untrusted locator, this module decides — as a
//! **pure, I/O-free, synchronous** function — whether those bytes are the
//! genuine capsule that identifier names. It composes three gates, in a fixed
//! order:
//!
//! 1. **canon-gate** (R3 / A16) — decode the bytes with the strict deterministic
//!    DAG-CBOR codec and require `reserialize(decode(bytes)) == bytes`. Non-
//!    canonical encodings are *rejected*, never normalized, so no two byte
//!    strings can present as "the same" capsule. [`Capsule::from_dag_cbor`]
//!    already enforces this round-trip plus every schema gate.
//! 2. **hash-gate** (A1 / A16) — recompute the BLAKE3-512 CID512 over the
//!    canonical bytes and require it to equal the claimed `cid512`. This is the
//!    self-certification binding: **the identity *is* the hash of the capsule**,
//!    so a capsule whose bytes do not hash to the claimed identifier is rejected
//!    even when its own self-signature is perfectly valid (A1: forging a capsule
//!    for a known cid512 is a BLAKE3-512 second-preimage, infeasible).
//! 3. **sig-gate** (§4.4 / R4 / A19) — composite-verify via [`verify_capsule`],
//!    which pins Hybrid (EdDSA **and** ML-DSA-65) independent of node
//!    `CryptoPolicy` and checks the per-record COSE context string.
//!
//! # Why this order (cheapest-first, and each gate depends on the prior)
//!
//! - **canon before hash.** The hash is only meaningful over *canonical* bytes:
//!   without the canonical check first, an attacker could serve a non-canonical
//!   re-encoding of a legitimate capsule whose hash differs, or (worse) rely on
//!   decoder laxity. canon-gate is also the cheapest — one decode + one
//!   re-encode + a `memcmp`.
//! - **hash before sig.** The composite verification (ML-DSA-65) is by far the
//!   most expensive step; the hash comparison is a single BLAKE3 pass. Rejecting
//!   an identity mismatch before touching the signature avoids spending a PQ
//!   verify on bytes that cannot possibly be the requested capsule. More
//!   importantly, hash-gate establishes *which identity* these bytes claim to
//!   be; sig-gate then confirms the capsule's own keys signed it. A valid self-
//!   signature over the *wrong* cid must still be rejected (see the tests), and
//!   ordering hash before sig makes that structural.
//!
//! The pipeline returns a [`VerifiedCapsule`] — a witness that all three gates
//! passed, constructible only here — or a [`GateError`] naming the gate that
//! rejected the input.

use std::fmt;

use hyprstream_rpc::cid::{decode_cid, encode_cid, Codec, HashAlgo};

use crate::at9p::{at9p_capsule_cid512, Capsule, H512_LEN};
use crate::at9p_sign::verify_capsule;

/// The `did:at9p:` method prefix. A `did:at9p` identifier is exactly this prefix
/// followed by the base32 CIDv1 (`cid512`) of the genesis capsule.
pub const DID_AT9P_PREFIX: &str = "did:at9p:";

/// Which gate of the [`verify_genesis_capsule`] pipeline rejected the input.
///
/// The variants are ordered as the pipeline evaluates them; each names the gate
/// so a caller (or audit record) can distinguish "these bytes are malformed"
/// from "these bytes are not the capsule you asked for" from "these bytes are
/// the right shape and identity but are not authentically signed".
#[derive(Debug)]
pub enum GateError {
    /// The claimed identifier was not a well-formed at9p `cid512`
    /// (`did:at9p:<cid512>`): wrong multibase, wrong multicodec, or not a
    /// BLAKE3-512 multihash. Checked before any capsule bytes are examined so
    /// the hash-gate always compares against a real cid512.
    ClaimedId(anyhow::Error),
    /// canon-gate (R3/A16): the bytes did not decode as a canonical,
    /// schema-valid capsule — either the DAG-CBOR was malformed, non-canonical
    /// (`reserialize(decode(bytes)) != bytes`), or violated a capsule schema
    /// invariant.
    Canon(anyhow::Error),
    /// hash-gate (A1/A16): `BLAKE3-512(canonical bytes)` did not equal the
    /// claimed `cid512`. The bytes are a well-formed capsule, but not the one
    /// this identity names.
    Hash {
        /// The identifier the caller asked to verify (canonicalized).
        claimed: String,
        /// The identifier the supplied bytes actually hash to.
        computed: String,
    },
    /// sig-gate (§4.4/R4/A19): composite verification failed under pinned
    /// Hybrid — a missing/invalid EdDSA or ML-DSA-65 layer, a context-string
    /// mismatch, or a signature that does not verify against the capsule's own
    /// primary subject key.
    Sig(anyhow::Error),
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClaimedId(e) => write!(f, "claimed-id gate: malformed did:at9p cid512: {e}"),
            Self::Canon(e) => write!(f, "canon-gate: {e}"),
            Self::Hash { claimed, computed } => write!(
                f,
                "hash-gate: capsule bytes hash to {computed}, not the claimed {claimed}"
            ),
            Self::Sig(e) => write!(f, "sig-gate: {e}"),
        }
    }
}

impl std::error::Error for GateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ClaimedId(e) | Self::Canon(e) | Self::Sig(e) => Some(e.as_ref()),
            Self::Hash { .. } => None,
        }
    }
}

/// A capsule that has passed the full canon → hash → sig GATE pipeline for a
/// specific claimed `cid512`.
///
/// The only constructor is [`verify_genesis_capsule`], so holding a
/// `VerifiedCapsule` is proof that: the bytes were canonical (R3), they hash to
/// [`Self::cid512`] (self-certification), and the capsule's own primary subject
/// key composite-signed them under pinned Hybrid (§4.4). Downstream authority
/// (keys, endpoints, label *hints*, delegations) may be read from
/// [`Self::capsule`] without re-checking the binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedCapsule {
    cid512: String,
    capsule: Capsule,
}

impl VerifiedCapsule {
    /// The verified `cid512` — the BLAKE3-512 self-certifying identifier these
    /// bytes hash to (canonical form).
    pub fn cid512(&self) -> &str {
        &self.cid512
    }

    /// The full `did:at9p:<cid512>` identifier for the verified capsule.
    pub fn did(&self) -> String {
        format!("{DID_AT9P_PREFIX}{}", self.cid512)
    }

    /// The verified capsule.
    pub fn capsule(&self) -> &Capsule {
        &self.capsule
    }

    /// Consume the witness, yielding the verified capsule.
    pub fn into_capsule(self) -> Capsule {
        self.capsule
    }
}

/// Verify that `capsule_bytes` are the genuine genesis capsule named by the
/// claimed `did:at9p:<did_body>` identifier, where `did_body` is the bare
/// `cid512` (no `did:at9p:` prefix).
///
/// Pure and I/O-free: no network, no filesystem, no async, no clock. The caller
/// is responsible for having *fetched* the bytes for this identifier from
/// whatever (untrusted) locator; this function decides whether to *accept*
/// them.
///
/// Returns a [`VerifiedCapsule`] witness, or a [`GateError`] naming the gate
/// that rejected the input. Gates run in order canon → hash → sig; a later gate
/// is never reached if an earlier one fails.
pub fn verify_genesis_capsule(
    claimed_cid512: &str,
    capsule_bytes: &[u8],
) -> Result<VerifiedCapsule, GateError> {
    // Pre-gate: the claimed identifier must be a well-formed at9p cid512 before
    // we can meaningfully compare a computed hash against it. This is cheap and
    // never touches the (attacker-controlled) capsule bytes.
    let claimed = canonical_cid512(claimed_cid512).map_err(GateError::ClaimedId)?;

    // GATE 1 — canon (R3 / A16): decode canonically or reject. `from_dag_cbor`
    // requires `decode(bytes).encode() == bytes` and runs every schema gate,
    // including PR #937's strict decoder hardening (depth limit, string-only map
    // keys). A16 is defeated here: no non-canonical re-encoding survives.
    let capsule = Capsule::from_dag_cbor(capsule_bytes).map_err(GateError::Canon)?;

    // GATE 2 — hash (A1 / A16): recompute the self-certifying cid over the
    // canonical bytes and bind it to the claimed identity. canon-gate proved the
    // input is canonical, so hashing the input bytes is exactly hashing the
    // canonical form. Forging a capsule for a fixed cid512 is a BLAKE3-512
    // second-preimage (A1) — infeasible — so a match means these are *the* bytes
    // the identity commits to.
    let computed = at9p_capsule_cid512(capsule_bytes).map_err(GateError::Canon)?;
    if computed != claimed {
        return Err(GateError::Hash { claimed, computed });
    }

    // GATE 3 — sig (§4.4 / R4 / A19): composite verify, Hybrid pinned, per-record
    // COSE context checked. A valid self-signature over the *wrong* identity was
    // already rejected at hash-gate; this confirms the correctly-identified bytes
    // are also authentically signed by the capsule's own primary subject key.
    verify_capsule(&capsule).map_err(GateError::Sig)?;

    Ok(VerifiedCapsule {
        cid512: computed,
        capsule,
    })
}

/// Verify a full `did:at9p:<cid512>` identifier against fetched capsule bytes.
///
/// Convenience wrapper over [`verify_genesis_capsule`] that strips the
/// `did:at9p:` method prefix. Same purity guarantees.
pub fn verify_did_at9p(did: &str, capsule_bytes: &[u8]) -> Result<VerifiedCapsule, GateError> {
    let cid512 = did.strip_prefix(DID_AT9P_PREFIX).ok_or_else(|| {
        GateError::ClaimedId(anyhow::anyhow!(
            "identifier is not a did:at9p DID: {did:?}"
        ))
    })?;
    verify_genesis_capsule(cid512, capsule_bytes)
}

/// Validate that `s` is a well-formed at9p `cid512` and return its canonical
/// string form.
///
/// Enforces the at9p-capsule multicodec and a BLAKE3-512 (64-byte) multihash,
/// then re-encodes from the decoded digest so the returned value is the exact
/// canonical CIDv1 string [`at9p_capsule_cid512`] produces. Both sides of the
/// hash-gate comparison are therefore canonical, so a byte-exact `==` is a
/// faithful digest comparison (no case/representation skew).
fn canonical_cid512(s: &str) -> anyhow::Result<String> {
    let cid = decode_cid(s)?;
    anyhow::ensure!(
        cid.codec == Codec::At9pCapsule,
        "cid512 must use the at9p-capsule multicodec"
    );
    anyhow::ensure!(
        cid.multihash.algo == HashAlgo::Blake3 && cid.multihash.digest.len() == H512_LEN,
        "cid512 must be a BLAKE3-512 (64-byte) multihash"
    );
    encode_cid(Codec::At9pCapsule, HashAlgo::Blake3, &cid.multihash.digest)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]
mod tests {
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use proptest::prelude::*;

    use super::*;
    use crate::at9p::{
        Capsule, CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
        ED25519_SIGNATURE_LEN,
    };
    use crate::at9p_sign::sign_capsule;
    use crate::dag_cbor::DagCbor;

    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer(tag: u8) -> Signer {
        // Deterministic-ish per-tag ed key so distinct tags give distinct DIDs;
        // ML-DSA key is random (fine — the capsule commits to whatever it holds).
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

    /// A fully self-signed capsule, its canonical bytes, and its true cid512.
    fn signed(tag: u8) -> (Capsule, Vec<u8>, String) {
        let s = signer(tag);
        let body = body_for(&s, tag);
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        let bytes = capsule.to_dag_cbor();
        let cid = capsule.cid512().unwrap();
        (capsule, bytes, cid)
    }

    #[test]
    fn accepts_a_valid_capsule() {
        let (capsule, bytes, cid) = signed(1);
        let verified = verify_genesis_capsule(&cid, &bytes).expect("valid capsule must pass");
        assert_eq!(verified.cid512(), cid);
        assert_eq!(verified.did(), format!("{DID_AT9P_PREFIX}{cid}"));
        assert_eq!(verified.capsule(), &capsule);
    }

    #[test]
    fn accepts_via_full_did() {
        let (_capsule, bytes, cid) = signed(2);
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        let verified = verify_did_at9p(&did, &bytes).expect("valid did:at9p must pass");
        assert_eq!(verified.did(), did);
    }

    #[test]
    fn rejects_non_did_at9p_identifier() {
        let (_capsule, bytes, cid) = signed(3);
        let err = verify_did_at9p(&format!("did:web:{cid}"), &bytes).unwrap_err();
        assert!(matches!(err, GateError::ClaimedId(_)), "got {err}");
    }

    #[test]
    fn rejects_malformed_claimed_cid() {
        let (_capsule, bytes, _cid) = signed(4);
        let err = verify_genesis_capsule("not-a-cid", &bytes).unwrap_err();
        assert!(matches!(err, GateError::ClaimedId(_)), "got {err}");
    }

    /// A1 / A16: hash-binding. A capsule whose bytes do not hash to the claimed
    /// cid512 is rejected at hash-gate even though the bytes are perfectly
    /// canonical AND carry a valid self-signature. The identity IS the hash.
    #[test]
    fn valid_self_signed_capsule_under_wrong_cid_rejected() {
        let (_c1, bytes1, cid1) = signed(10);
        let (_c2, _bytes2, cid2) = signed(11);
        assert_ne!(cid1, cid2);
        // bytes1 is genuinely self-signed and canonical, but we claim it is the
        // identity cid2. hash-gate must reject before sig-gate ever runs.
        let err = verify_genesis_capsule(&cid2, &bytes1).unwrap_err();
        match err {
            GateError::Hash { claimed, computed } => {
                assert_eq!(claimed, cid2);
                assert_eq!(computed, cid1);
            }
            other => panic!("expected hash-gate rejection, got {other}"),
        }
    }

    /// Wrong CID (a different well-formed at9p cid512) fails hash-gate.
    #[test]
    fn wrong_cid_fails_hash_gate() {
        let (_capsule, bytes, cid) = signed(20);
        // Flip one digit region of the cid to get a different (but still
        // structurally valid) at9p cid512.
        let other = signed(21).2;
        assert_ne!(cid, other);
        let err = verify_genesis_capsule(&other, &bytes).unwrap_err();
        assert!(matches!(err, GateError::Hash { .. }), "got {err}");
    }

    /// A16 / R3: tampering the bytes fails at canon-gate (non-canonical / schema)
    /// or hash-gate (canonical but different hash) — never silently accepted.
    #[test]
    fn tampered_bytes_fail_canon_or_hash() {
        let (_capsule, bytes, cid) = signed(30);
        for i in 0..bytes.len() {
            let mut t = bytes.clone();
            t[i] ^= 0x01;
            if t == bytes {
                continue;
            }
            match verify_genesis_capsule(&cid, &t) {
                Ok(_) => panic!("tampered byte {i} was accepted"),
                Err(GateError::Canon(_)) | Err(GateError::Hash { .. }) => {}
                Err(other) => panic!("tampered byte {i}: unexpected {other}"),
            }
        }
    }

    /// Valid canonical bytes with the correct (recomputed) cid512 but a broken
    /// signature reach and fail sig-gate. Because the cid commits to the whole
    /// capsule (signatures included), tampering the signature changes the cid;
    /// we therefore claim the tampered capsule's *own* recomputed cid so canon
    /// and hash pass and only sig fails.
    #[test]
    fn valid_canon_bad_sig_fails_sig_gate() {
        let (capsule, _bytes, _cid) = signed(40);
        // Corrupt the ed25519 signature but keep it schema-valid (same length).
        let mut broken = capsule.clone();
        broken.signatures.ed25519_signature = vec![0u8; ED25519_SIGNATURE_LEN];
        let bytes = broken.to_dag_cbor();
        // The tampered capsule is still canonical & schema-valid.
        let reparsed = Capsule::from_dag_cbor(&bytes).expect("still canonical");
        let cid = reparsed.cid512().unwrap();
        let err = verify_genesis_capsule(&cid, &bytes).unwrap_err();
        assert!(matches!(err, GateError::Sig(_)), "got {err}");
    }

    /// Non-canonical bytes (reversed top-level map) fail canon-gate — even when
    /// they decode to a structurally identical capsule.
    #[test]
    fn non_canonical_bytes_fail_canon_gate() {
        let (_capsule, bytes, cid) = signed(50);
        let decoded = DagCbor::decode(&bytes).unwrap();
        let DagCbor::Map(mut pairs) = decoded else {
            panic!("capsule encodes as a map");
        };
        pairs.reverse();
        let non_canonical = DagCbor::Map(pairs).encode();
        assert_ne!(non_canonical, bytes);
        let err = verify_genesis_capsule(&cid, &non_canonical).unwrap_err();
        assert!(matches!(err, GateError::Canon(_)), "got {err}");
    }

    /// Garbage bytes fail canon-gate.
    #[test]
    fn garbage_bytes_fail_canon_gate() {
        let (_capsule, _bytes, cid) = signed(60);
        let err = verify_genesis_capsule(&cid, b"\xff\xff not cbor at all").unwrap_err();
        assert!(matches!(err, GateError::Canon(_)), "got {err}");
    }

    proptest! {
        /// A1: every genuinely self-signed capsule verifies against its own
        /// cid512, and against NO other capsule's cid512 (hash-binding /
        /// dedup-domain: distinct bytes ⇒ distinct BLAKE3-512 cid).
        #[test]
        fn prop_roundtrip_accepts_self_rejects_others(a in 0u8..40, b in 0u8..40) {
            let (_ca, bytes_a, cid_a) = signed(a);
            let (_cb, _bytes_b, cid_b) = signed(b);

            // Accepts its own identity.
            prop_assert!(verify_genesis_capsule(&cid_a, &bytes_a).is_ok());

            if cid_a != cid_b {
                // Rejects a different identity at hash-gate.
                let err = verify_genesis_capsule(&cid_b, &bytes_a).unwrap_err();
                let is_hash = matches!(err, GateError::Hash { .. });
                prop_assert!(is_hash);
            }
        }

        /// A16 / R3: single-bit flips are never silently accepted.
        #[test]
        fn prop_bitflip_never_accepted(tag in 0u8..40, idx in any::<prop::sample::Index>()) {
            let (_c, bytes, cid) = signed(tag);
            let i = idx.index(bytes.len());
            let mut t = bytes.clone();
            t[i] ^= 0x01;
            prop_assume!(t != bytes);
            prop_assert!(verify_genesis_capsule(&cid, &t).is_err());
        }
    }
}
