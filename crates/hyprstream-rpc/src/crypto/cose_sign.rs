//! Nested-COSE **Strong-Non-Separable (SNS)** composite signing for the
//! hyprstream `SignedEnvelope` (EdDSA + ML-DSA-65).
//!
//! M3 (#152) migrates envelope authentication to a **nested COSE composite**.
//! A prior design used a `COSE_Sign` with two *independent* signatures over the
//! same payload (concatenation / Weak-Non-Separable, WNS). A security review
//! found that scheme **fail-open + strippable**: an attacker could remove the
//! ML-DSA-65 signature and the remaining EdDSA-only object still verified under
//! a verifier that did not actively require the PQ entry. This module replaces
//! it with a **Strong-Non-Separable (SNS)** nesting:
//!
//! ```text
//! Classical:
//!   inner  = EdDSA   COSE_Sign1 over  payload                       (detached)
//!
//! Hybrid (SNS):
//!   inner  = EdDSA   COSE_Sign1 over  payload                       (detached)
//!   outer  = ML-DSA  COSE_Sign1 over  payload ‖ inner_eddsa_sig     (detached)
//! ```
//!
//! Both layers bind the same `external_aad` (the CBOR `[envelope_schema_id,
//! inner_type_id]` schema binding, see [`crate::crypto::cose_sign1::build_external_aad`]).
//!
//! # Why this is SNS
//!
//! - The OUTER ML-DSA-65 signature covers `payload ‖ inner_eddsa_signature`, so
//!   the inner signature is *part of the outer's signed message*. Tampering with
//!   or removing the inner EdDSA signature invalidates the outer.
//! - Under Hybrid policy the verifier REQUIRES the outer ML-DSA-65 layer and an
//!   anchored PQ key. **Stripping the outer fails the policy** (there is no
//!   valid outer to verify), so a downgrade-to-classical attack is rejected.
//! - The PQ key is kid-anchored: it is resolved from a [`PqTrustStore`] keyed by
//!   the EdDSA signer identity, never self-asserted in the COSE object. This
//!   closes the prior self-certification weakness.
//!
//! # Wire shape (`SignedEnvelope.cose`)
//!
//! A definite-length CBOR array of two entries:
//!
//! ```text
//! [ inner_cose_sign1_bytes : bstr,
//!   outer_cose_sign1_bytes : bstr / null ]   ; null in Classical mode
//! ```
//!
//! The signed payloads are *detached* (never embedded). Each COSE_Sign1 carries
//! its signer's `kid` + `alg` in its protected header.

use anyhow::{anyhow, bail, Context, Result};
use ciborium::value::Value as CborValue;
use coset::{
    iana::{self, EnumI64},
    CborSerializable, CoseSign1, CoseSign1Builder, HeaderBuilder,
};

use crate::crypto::pq::{ml_dsa_sign, ml_dsa_verify, MlDsaSigningKey, MlDsaVerifyingKey};

/// IANA COSE algorithm id for ML-DSA-65 (FIPS 204, draft-ietf-cose-dilithium).
pub const ALG_ML_DSA_65: i64 = iana::Algorithm::ML_DSA_65 as i64;

/// EdDSA kid convention: raw 32-byte Ed25519 verifying key.
fn eddsa_kid(vk: &ed25519_dalek::VerifyingKey) -> Vec<u8> {
    vk.to_bytes().to_vec()
}

/// ML-DSA-65 kid convention: raw verifying key bytes (1952 bytes).
fn ml_dsa_kid(vk: &MlDsaVerifyingKey) -> Vec<u8> {
    crate::crypto::pq::ml_dsa_vk_bytes(vk)
}

/// Build the inner EdDSA `COSE_Sign1` over the detached `payload`.
fn build_inner_eddsa(
    ed_sk: &ed25519_dalek::SigningKey,
    payload: &[u8],
    external_aad: &[u8],
) -> Result<CoseSign1> {
    use ed25519_dalek::Signer as _;
    let protected = HeaderBuilder::new()
        .algorithm(iana::Algorithm::EdDSA)
        .key_id(eddsa_kid(&ed_sk.verifying_key()))
        .build();
    Ok(CoseSign1Builder::new()
        .protected(protected)
        .create_detached_signature(payload, external_aad, |tbs| {
            ed_sk.sign(tbs).to_bytes().to_vec()
        })
        .build())
}

/// Build the outer ML-DSA-65 `COSE_Sign1` over the detached
/// `payload ‖ inner_eddsa_signature` (the SNS binding).
fn build_outer_mldsa(
    pq_sk: &MlDsaSigningKey,
    outer_payload: &[u8],
    external_aad: &[u8],
) -> Result<CoseSign1> {
    use ml_dsa::Keypair;
    let pq_vk = pq_sk.verifying_key().clone();
    let protected = HeaderBuilder::new()
        .algorithm(iana::Algorithm::ML_DSA_65)
        .key_id(ml_dsa_kid(&pq_vk))
        .build();
    Ok(CoseSign1Builder::new()
        .protected(protected)
        .create_detached_signature(outer_payload, external_aad, |tbs| ml_dsa_sign(pq_sk, tbs))
        .build())
}

/// Encode the nested composite as a 2-element CBOR array
/// `[inner_bstr, outer_bstr | null]`.
fn encode_composite(inner: Vec<u8>, outer: Option<Vec<u8>>) -> Result<Vec<u8>> {
    let arr = CborValue::Array(vec![
        CborValue::Bytes(inner),
        match outer {
            Some(o) => CborValue::Bytes(o),
            None => CborValue::Null,
        },
    ]);
    let mut buf = Vec::new();
    ciborium::ser::into_writer(&arr, &mut buf)
        .map_err(|e| anyhow!("failed to encode nested COSE composite: {e}"))?;
    Ok(buf)
}

/// Decode the nested composite back into `(inner_bytes, Option<outer_bytes>)`.
fn decode_composite(bytes: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let value: CborValue = ciborium::de::from_reader(bytes)
        .map_err(|e| anyhow!("malformed nested COSE composite: {e}"))?;
    let arr = match value {
        CborValue::Array(a) => a,
        _ => bail!("nested COSE composite must be a CBOR array"),
    };
    if arr.len() != 2 {
        bail!("nested COSE composite must have exactly 2 elements, got {}", arr.len());
    }
    let inner = match &arr[0] {
        CborValue::Bytes(b) => b.clone(),
        _ => bail!("nested COSE composite inner entry must be a byte string"),
    };
    let outer = match &arr[1] {
        CborValue::Null => None,
        CborValue::Bytes(b) => Some(b.clone()),
        _ => bail!("nested COSE composite outer entry must be a byte string or null"),
    };
    Ok((inner, outer))
}

/// The raw inner EdDSA signature bytes (64) extracted from an inner COSE_Sign1.
fn inner_signature_bytes(inner: &CoseSign1) -> &[u8] {
    &inner.signature
}

/// Compute the outer payload for the SNS binding: `payload ‖ inner_eddsa_sig`.
fn outer_payload(payload: &[u8], inner_sig: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(payload.len() + inner_sig.len());
    v.extend_from_slice(payload);
    v.extend_from_slice(inner_sig);
    v
}

/// Sign `payload` (detached) producing the CBOR-encoded nested COSE composite.
///
/// - Classical (`pq_sk = None`): single inner EdDSA COSE_Sign1, outer = null.
/// - Hybrid (`pq_sk = Some`): inner EdDSA over `payload`, outer ML-DSA-65 over
///   `payload ‖ inner_eddsa_signature` (SNS).
pub fn sign_composite(
    ed_sk: &ed25519_dalek::SigningKey,
    pq_sk: Option<&MlDsaSigningKey>,
    payload: &[u8],
    external_aad: &[u8],
) -> Result<Vec<u8>> {
    let inner = build_inner_eddsa(ed_sk, payload, external_aad)?;
    let inner_sig = inner_signature_bytes(&inner).to_vec();
    let inner_bytes = inner
        .to_vec()
        .map_err(|e| anyhow!("failed to serialize inner EdDSA COSE_Sign1: {e}"))?;

    let outer_bytes = match pq_sk {
        Some(pq) => {
            let outer_pl = outer_payload(payload, &inner_sig);
            let outer = build_outer_mldsa(pq, &outer_pl, external_aad)?;
            Some(
                outer
                    .to_vec()
                    .map_err(|e| anyhow!("failed to serialize outer ML-DSA COSE_Sign1: {e}"))?,
            )
        }
        None => None,
    };

    encode_composite(inner_bytes, outer_bytes)
}

// ────────────────────────────────────────────────────────────────────────────
// Out-of-band (async / WASM) signing API for the nested SNS composite.
//
// Abstract signers (e.g. the WASM `JsSigner`) cannot run inside a synchronous
// signing closure, so they compute each layer's to-be-signed bytes, sign them
// out-of-band, and assemble the composite here. The SNS nesting REQUIRES the
// outer ML-DSA layer to sign `payload ‖ inner_eddsa_signature`, so callers MUST
// sign the inner first, take its signature, then sign the outer over
// [`outer_tbs`].
// ────────────────────────────────────────────────────────────────────────────

/// Build an unsigned inner EdDSA `COSE_Sign1` (protected header only).
fn unsigned_inner(ed_kid: Vec<u8>) -> CoseSign1 {
    CoseSign1Builder::new()
        .protected(
            HeaderBuilder::new()
                .algorithm(iana::Algorithm::EdDSA)
                .key_id(ed_kid)
                .build(),
        )
        .build()
}

/// Build an unsigned outer ML-DSA-65 `COSE_Sign1` (protected header only).
fn unsigned_outer(pq_kid: Vec<u8>) -> CoseSign1 {
    CoseSign1Builder::new()
        .protected(
            HeaderBuilder::new()
                .algorithm(iana::Algorithm::ML_DSA_65)
                .key_id(pq_kid)
                .build(),
        )
        .build()
}

/// Compute the inner EdDSA layer's detached to-be-signed bytes.
pub fn inner_tbs(ed_kid: Vec<u8>, payload: &[u8], external_aad: &[u8]) -> Vec<u8> {
    unsigned_inner(ed_kid).tbs_detached_data(payload, external_aad)
}

/// Compute the outer ML-DSA-65 layer's detached to-be-signed bytes over
/// `payload ‖ inner_eddsa_signature` (the SNS binding).
pub fn outer_tbs(
    pq_kid: Vec<u8>,
    payload: &[u8],
    inner_eddsa_sig: &[u8],
    external_aad: &[u8],
) -> Vec<u8> {
    let outer_pl = outer_payload(payload, inner_eddsa_sig);
    unsigned_outer(pq_kid).tbs_detached_data(&outer_pl, external_aad)
}

/// Assemble the nested SNS composite from out-of-band signatures.
///
/// - `ed = (ed_kid, ed_signature)` — required; `ed_signature` is the raw 64-byte
///   Ed25519 signature over [`inner_tbs`].
/// - `pq = Some((pq_kid, pq_signature))` — Hybrid; `pq_signature` is the ML-DSA-65
///   signature over [`outer_tbs`] (which was computed using `ed_signature`).
pub fn assemble_composite_nested(
    ed: (Vec<u8>, Vec<u8>),
    pq: Option<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<u8>> {
    let (ed_kid, ed_sig) = ed;
    let mut inner = unsigned_inner(ed_kid);
    inner.signature = ed_sig;
    let inner_bytes = inner
        .to_vec()
        .map_err(|e| anyhow!("serialize inner EdDSA COSE_Sign1: {e}"))?;

    let outer_bytes = match pq {
        Some((pq_kid, pq_sig)) => {
            let mut outer = unsigned_outer(pq_kid);
            outer.signature = pq_sig;
            Some(
                outer
                    .to_vec()
                    .map_err(|e| anyhow!("serialize outer ML-DSA COSE_Sign1: {e}"))?,
            )
        }
        None => None,
    };

    encode_composite(inner_bytes, outer_bytes)
}

/// Test-only: decode a composite into `(inner_bytes, Option<outer_bytes>)`.
///
/// Exposed so integration tests in other modules (e.g. `envelope.rs`) can
/// simulate an attacker stripping the outer ML-DSA layer.
#[doc(hidden)]
pub fn decode_composite_for_test(bytes: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    decode_composite(bytes)
}

/// Test-only: re-encode a composite from `(inner_bytes, Option<outer_bytes>)`.
#[doc(hidden)]
pub fn encode_composite_for_test(inner: Vec<u8>, outer: Option<Vec<u8>>) -> Result<Vec<u8>> {
    encode_composite(inner, outer)
}

/// Outcome of composite verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeVerified {
    /// Inner EdDSA was verified.
    pub eddsa: bool,
    /// Outer ML-DSA-65 (SNS) was verified.
    pub ml_dsa: bool,
}

/// Verify a CBOR-encoded nested COSE composite against a detached `payload`.
///
/// # kid-anchoring
///
/// `expected_ed_vk` is the EdDSA verifying key resolved by the caller. When
/// `expected_pq_vk` is `Some`, it is the trust-anchored ML-DSA-65 key — the
/// outer COSE entry's kid AND signature must verify against it, closing the
/// self-certification gap.
///
/// # Policy
///
/// - `require_pq = true` (Hybrid verifier): the OUTER ML-DSA-65 layer MUST be
///   present and verify against the anchored key over `payload ‖ inner_sig`, AND
///   the inner EdDSA layer MUST verify over `payload`. If `expected_pq_vk` is
///   `None`, verification fails (no anchor — fail-closed). **Stripping the outer
///   layer (outer = null) is rejected.**
/// - `require_pq = false` (Classical verifier): only the inner EdDSA layer is
///   required and verified; any outer ML-DSA layer is IGNORED (skip-unknown
///   interop — a Hybrid-signed envelope still verifies via its inner EdDSA).
pub fn verify_composite(
    cose_composite_bytes: &[u8],
    expected_ed_vk: &ed25519_dalek::VerifyingKey,
    expected_pq_vk: Option<&MlDsaVerifyingKey>,
    payload: &[u8],
    external_aad: &[u8],
    require_pq: bool,
) -> Result<CompositeVerified> {
    let (inner_bytes, outer_bytes) = decode_composite(cose_composite_bytes)?;

    // --- Inner EdDSA layer (always required) ---
    let inner = CoseSign1::from_slice(&inner_bytes)
        .map_err(|e| anyhow!("malformed inner EdDSA COSE_Sign1: {e}"))?;

    let inner_alg = header_alg(&inner)?;
    if inner_alg != iana::Algorithm::EdDSA.to_i64() {
        bail!("inner COSE_Sign1 must be EdDSA, got alg={inner_alg}");
    }
    let inner_kid = &inner.protected.header.key_id;
    if !inner_kid.is_empty() && inner_kid.as_slice() != expected_ed_vk.to_bytes() {
        bail!("inner EdDSA COSE_Sign1 kid does not match anchored EdDSA key");
    }
    inner
        .verify_detached_signature(payload, external_aad, |sig, tbs| {
            ed_verify(expected_ed_vk, tbs, sig)
        })
        .context("inner EdDSA signature verification failed")?;
    let eddsa_ok = true;

    // Capture the inner signature; it is the SNS binding for the outer layer.
    let inner_sig = inner_signature_bytes(&inner).to_vec();

    // --- Outer ML-DSA-65 layer (SNS) ---
    let mut ml_dsa_ok = false;

    match outer_bytes {
        Some(ref ob) if require_pq || expected_pq_vk.is_some() => {
            // Anchored PQ key is mandatory to verify the outer layer.
            let Some(pq_vk) = expected_pq_vk else {
                if require_pq {
                    bail!("Hybrid policy requires an anchored ML-DSA-65 key, none provided");
                }
                // Classical verifier without an anchor: ignore the outer layer.
                return Ok(CompositeVerified { eddsa: eddsa_ok, ml_dsa: false });
            };
            let outer = CoseSign1::from_slice(ob)
                .map_err(|e| anyhow!("malformed outer ML-DSA COSE_Sign1: {e}"))?;
            let outer_alg = header_alg(&outer)?;
            if outer_alg != ALG_ML_DSA_65 {
                bail!("outer COSE_Sign1 must be ML-DSA-65, got alg={outer_alg}");
            }
            let outer_kid = &outer.protected.header.key_id;
            if !outer_kid.is_empty() && outer_kid.as_slice() != ml_dsa_kid(pq_vk).as_slice() {
                bail!("outer ML-DSA-65 COSE_Sign1 kid does not match anchored PQ key (self-cert rejected)");
            }
            let outer_pl = outer_payload(payload, &inner_sig);
            outer
                .verify_detached_signature(&outer_pl, external_aad, |sig, tbs| {
                    ml_dsa_verify(pq_vk, tbs, sig)
                })
                .context("outer ML-DSA-65 signature verification failed")?;
            ml_dsa_ok = true;
        }
        Some(_) => {
            // require_pq is false and no anchor: ignore outer (skip-unknown).
        }
        None => {
            // No outer layer present (Classical-signed).
            if require_pq {
                bail!("Hybrid policy requires the outer ML-DSA-65 layer, but it is absent (stripped or classical-only)");
            }
        }
    }

    if require_pq && !ml_dsa_ok {
        bail!("Hybrid policy: outer ML-DSA-65 layer missing or unverified");
    }

    Ok(CompositeVerified { eddsa: eddsa_ok, ml_dsa: ml_dsa_ok })
}

/// Extract the IANA alg integer from a COSE_Sign1 protected header.
fn header_alg(cose: &CoseSign1) -> Result<i64> {
    match cose.protected.header.alg.as_ref() {
        Some(coset::Algorithm::Assigned(a)) => Ok(a.to_i64()),
        Some(coset::Algorithm::PrivateUse(v)) => Ok(*v),
        Some(coset::Algorithm::Text(s)) => bail!("unsupported text algorithm: {s}"),
        None => bail!("COSE_Sign1 missing alg in protected header"),
    }
}

fn ed_verify(vk: &ed25519_dalek::VerifyingKey, msg: &[u8], sig: &[u8]) -> Result<()> {
    if sig.len() != 64 {
        bail!("Ed25519 signature must be 64 bytes, got {}", sig.len());
    }
    let mut arr = [0u8; 64];
    arr.copy_from_slice(sig);
    let signature = ed25519_dalek::Signature::from_bytes(&arr);
    vk.verify_strict(msg, &signature)
        .map_err(|e| anyhow!("Ed25519 verification failed: {e}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::crypto::cose_sign1::build_external_aad;
    use crate::crypto::pq::ml_dsa_generate_keypair;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    const SCHEMA_ID: u64 = 0xcf5e_016c_5d4a_af92;
    const INNER_TYPE_ID: u64 = 0xdead_beef_cafe_babe;

    fn aad() -> Vec<u8> {
        build_external_aad(SCHEMA_ID, INNER_TYPE_ID)
    }

    #[test]
    fn classical_sign_verify() {
        let ed = SigningKey::generate(&mut OsRng);
        let payload = b"classical payload";
        let cose = sign_composite(&ed, None, payload, &aad()).unwrap();
        let res = verify_composite(&cose, &ed.verifying_key(), None, payload, &aad(), false).unwrap();
        assert!(res.eddsa);
        assert!(!res.ml_dsa);
    }

    #[test]
    fn hybrid_sign_verify_both() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let payload = b"hybrid payload";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        let res = verify_composite(&cose, &ed.verifying_key(), Some(&pq_vk), payload, &aad(), true).unwrap();
        assert!(res.eddsa);
        assert!(res.ml_dsa);
    }

    #[test]
    fn hybrid_signed_verified_by_classical_via_eddsa() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let payload = b"cross payload";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        // Classical verifier: no PQ key, require_pq=false → inner EdDSA only.
        let res = verify_composite(&cose, &ed.verifying_key(), None, payload, &aad(), false).unwrap();
        assert!(res.eddsa);
        assert!(!res.ml_dsa);
    }

    #[test]
    fn classical_signed_rejected_by_hybrid_policy() {
        let ed = SigningKey::generate(&mut OsRng);
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let payload = b"classical only";
        let cose = sign_composite(&ed, None, payload, &aad()).unwrap();
        // Hybrid verifier requires the outer layer → absent → rejected.
        let res = verify_composite(&cose, &ed.verifying_key(), Some(&pq_vk), payload, &aad(), true);
        assert!(res.is_err(), "hybrid policy must reject classical-only (no outer layer)");
    }

    #[test]
    fn self_cert_wrong_pq_key_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let (_other_sk, other_vk) = ml_dsa_generate_keypair();
        let payload = b"anchored";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        let res = verify_composite(&cose, &ed.verifying_key(), Some(&other_vk), payload, &aad(), true);
        assert!(res.is_err(), "PQ key not matching kid-anchored key must be rejected");
    }

    #[test]
    fn tampered_payload_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let cose = sign_composite(&ed, Some(&pq_sk), b"orig", &aad()).unwrap();
        let res = verify_composite(&cose, &ed.verifying_key(), Some(&pq_vk), b"tampered", &aad(), true);
        assert!(res.is_err());
    }

    #[test]
    fn wrong_aad_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let cose = sign_composite(&ed, None, b"p", &aad()).unwrap();
        let wrong = build_external_aad(SCHEMA_ID, INNER_TYPE_ID ^ 1);
        let res = verify_composite(&cose, &ed.verifying_key(), None, b"p", &wrong, false);
        assert!(res.is_err());
    }

    /// SNS: stripping the outer ML-DSA layer (set to null) must be rejected
    /// under Hybrid policy. This is the attack the review found accepted before.
    #[test]
    fn strip_outer_mldsa_rejected_under_hybrid() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let payload = b"strip me";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();

        // Strip: rebuild the composite with the outer set to null.
        let (inner, _outer) = decode_composite(&cose).unwrap();
        let stripped = encode_composite(inner, None).unwrap();

        let res = verify_composite(&stripped, &ed.verifying_key(), Some(&pq_vk), payload, &aad(), true);
        assert!(res.is_err(), "stripping the outer ML-DSA layer must fail Hybrid policy");
    }

    /// SNS: tampering with the inner EdDSA signature invalidates the outer,
    /// because the outer signs `payload ‖ inner_sig`.
    #[test]
    fn tamper_inner_eddsa_invalidates_outer() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let payload = b"sns binding";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();

        // Re-sign the inner with a DIFFERENT EdDSA key but keep the original
        // outer. The inner kid won't match the anchored ed_vk, and even if it
        // did, the outer was bound to the original inner_sig.
        let (_inner, outer) = decode_composite(&cose).unwrap();
        let other_ed = SigningKey::generate(&mut OsRng);
        let forged_inner = build_inner_eddsa(&other_ed, payload, &aad()).unwrap();
        let forged_inner_bytes = forged_inner.to_vec().unwrap();
        let tampered = encode_composite(forged_inner_bytes, outer).unwrap();

        // Verify under the ORIGINAL ed_vk + anchored pq_vk: inner kid mismatch
        // OR outer-binding mismatch → reject.
        let res = verify_composite(&tampered, &ed.verifying_key(), Some(&pq_vk), payload, &aad(), true);
        assert!(res.is_err(), "tampering the inner EdDSA must invalidate the composite");
    }

    /// Hybrid policy with no anchored PQ key must fail closed.
    #[test]
    fn hybrid_no_anchor_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let payload = b"no anchor";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        let res = verify_composite(&cose, &ed.verifying_key(), None, payload, &aad(), true);
        assert!(res.is_err(), "Hybrid policy requires an anchored PQ key");
    }
}
