//! Nested-COSE hybrid composite signing for the hyprstream `SignedEnvelope`
//! (EdDSA + ML-DSA-65).
//!
//! ## Where this sits in the PQUIP hybrid-signature taxonomy
//!
//! In the taxonomy of `draft-ietf-pquip-hybrid-signature-spectrums` §1.3.4 this
//! construction is still **Weak Non-Separable (WNS)**, *not* Strong Non-Separable
//! (SNS). SNS requires *fusion* — the two algorithms' signatures combined into a
//! single object that cannot be verified by either component alone. We do not do
//! that: we emit two distinct COSE_Sign1 signatures and bind them by nesting
//! (the outer signs `payload ‖ inner_sig`).
//!
//! As of #278 the inner EdDSA layer additionally binds a hybrid-composite alg-id
//! into its signed input when produced in Hybrid mode (see
//! [`HYBRID_COMPOSITE_ALG_ID`] / [`build_hybrid_external_aad`]). This delivers
//! *signature-level* non-separability for the inner component: an inner EdDSA
//! signature lifted out of a Hybrid composite (its outer ML-DSA layer stripped)
//! is NO LONGER a valid standalone Classical signature, because its `Sig_structure`
//! was bound to the hybrid-alg-id that a Classical verifier does not reconstruct.
//! A genuine *Classical*-mode inner (no hybrid-alg-id) still verifies under a
//! Classical verifier — that intended interop is preserved. We keep the WNS label
//! because the two signatures remain separable *objects* (no fusion) and a
//! standalone outer ML-DSA layer is still independently meaningful; #278 closes
//! the inner-component separability gap, not the structural one.
//!
//! **Downgrade resistance is now enforced by BOTH crypto and policy:**
//!   1. signature-level non-separability (#278): a stripped-inner attack fails the
//!      inner EdDSA *signature* check (AAD mismatch), not merely a policy gate.
//!   2. `require_pq` (fail-closed Hybrid policy): a Hybrid verifier rejects any
//!      envelope lacking a valid outer ML-DSA-65 layer, so an attacker who strips
//!      the outer layer is *also* rejected by policy.
//!   3. kid-anchoring: the PQ key is resolved from a [`PqTrustStore`] keyed by
//!      the EdDSA signer identity, never self-asserted in the COSE object —
//!      closing the self-certification gap a naïve two-signature scheme has.
//!
//! ## What the nesting *does* buy
//!
//! M3 (#152) replaced an earlier `COSE_Sign` with two *independent* signatures
//! over the same payload (a fail-open, trivially strippable scheme). The nesting
//! below is a real improvement over that: because the outer ML-DSA-65 signature
//! covers `payload ‖ inner_eddsa_signature`, the inner signature is part of the
//! outer's signed message — so once a verifier has *committed to checking the
//! outer* (Hybrid policy), it cannot be fed a mismatched inner/outer pair. The
//! nesting binds the two layers together; the *requirement* to check the outer
//! at all comes from policy.
//!
//! ```text
//! Classical:                                          external_aad
//!   inner  = EdDSA  COSE_Sign1 over  payload          base_aad
//!
//! Hybrid (#278 hybrid-alg-id bound; policy-enforced): external_aad
//!   inner  = EdDSA  COSE_Sign1 over  payload          [base_aad, hybridAlgID]
//!   outer  = ML-DSA COSE_Sign1 over  payload‖inner_sig [base_aad, hybridAlgID]
//! ```
//!
//! Both layers bind the base `external_aad` (the CBOR `[envelope_schema_id,
//! inner_type_id]` schema binding, see [`crate::crypto::cose_sign1::build_external_aad`]),
//! and — in **Hybrid** mode only — *additionally* bind a hybrid-composite
//! algorithm identifier (see [`HYBRID_COMPOSITE_ALG_ID`]) into both layers'
//! `external_aad` via [`build_hybrid_external_aad`].
//!
//! ## Signature-level non-separability of the inner EdDSA (#278)
//!
//! Per `draft-ietf-pquip-hybrid-signature-spectrums` (§ Nested Construction /
//! Non-Separability), signature-level non-separability requires binding a
//! `hybridAlgID` into each component's signed input: `sig = Sign(hybridAlgID ‖ m)`.
//! We achieve this by extending the COSE `external_aad` (which RFC 9052 §4.4
//! folds into every layer's `Sig_structure`) with the composite alg-id whenever
//! signing in Hybrid mode.
//!
//! Consequence: the inner EdDSA `COSE_Sign1` to-be-signed bytes produced in
//! Hybrid mode are now **byte-distinct** from those produced in Classical mode,
//! so a stripped inner EdDSA signature lifted out of a Hybrid composite is *not*
//! a valid standalone Classical signature — it fails verification at the
//! *signature* level (not merely the `require_pq` policy check). In **Classical**
//! mode the `external_aad` is left exactly as before (no hybrid-alg-id), so
//! classical-only signing/verification is byte-for-byte unchanged.
//!
//! ## Backwards-compatibility / transition window
//!
//! This changes *what is signed* under Hybrid mode: a Hybrid signer at this
//! revision binds the hybrid-alg-id, so its inner EdDSA layer will NOT verify
//! against a pre-#278 Classical verifier that reconstructs the bare base AAD,
//! and vice-versa. Per WNS the classical `sig`/`cnf` fields on the envelope are
//! retained for the raw-EdDSA advertisement / JWT-cnf paths and are unaffected.
//! During a mixed-version deployment, peers MUST agree on the same policy
//! (Classical↔Classical or Hybrid↔Hybrid); a Hybrid producer talking to a
//! Classical-only consumer was already rejected by the `require_pq` policy, so
//! no *new* interop surface is broken — only the Hybrid inner AAD changed.
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

/// Stable composite/hybrid algorithm identifier for the EdDSA + ML-DSA-65
/// component pair signed by this nested construction.
///
/// Bound into the COSE `external_aad` of BOTH layers in Hybrid mode so that the
/// inner EdDSA `Sig_structure` becomes byte-distinct from a Classical-mode one
/// (signature-level non-separability, see this module's docs and #278).
///
/// Value: the ASCII bytes of the registered composite-signature name
/// `id-MLDSA65-Ed25519` from `draft-ietf-lamps-pq-composite-sigs` (the LAMPS
/// composite that pairs ML-DSA-65 with Ed25519). We bind a deterministic,
/// human-auditable label rather than an integer OID-arc to keep the AAD
/// self-describing on the wire and avoid depending on a not-yet-final IANA COSE
/// code-point allocation. It is a fixed constant: changing it is a wire-breaking
/// change that both signer and verifier must adopt together.
pub const HYBRID_COMPOSITE_ALG_ID: &[u8] = b"id-MLDSA65-Ed25519";

/// Extend a base `external_aad` with the [`HYBRID_COMPOSITE_ALG_ID`] binding.
///
/// Produces the deterministic CBOR encoding of the 2-element array
/// `[base_aad : bstr, HYBRID_COMPOSITE_ALG_ID : bstr]`. Wrapping the (already
/// CBOR) base AAD as a byte string inside a fresh 2-bstr array is unambiguous:
/// the Classical AAD is the bare base bytes (a CBOR `[int,int]` array, major
/// type 4), whereas the Hybrid AAD is a CBOR `[bstr,bstr]` array (also major
/// type 4 but with bstr elements), so the two TBS inputs can never collide.
///
/// Used ONLY in Hybrid mode; Classical mode signs/verifies against the bare
/// `base_aad` so classical-only behaviour is byte-for-byte unchanged.
pub fn build_hybrid_external_aad(base_aad: &[u8]) -> Vec<u8> {
    let value = CborValue::Array(vec![
        CborValue::Bytes(base_aad.to_vec()),
        CborValue::Bytes(HYBRID_COMPOSITE_ALG_ID.to_vec()),
    ]);
    let mut buf = Vec::with_capacity(base_aad.len() + HYBRID_COMPOSITE_ALG_ID.len() + 8);
    // Infallible for this fixed [bstr,bstr] shape into an unbounded Vec. Fail
    // LOUD rather than silently degrading the hybrid-alg-id binding to an empty
    // (classical-equivalent) AAD on BOTH sign and verify — a silent empty return
    // would erode the #278 non-separability property invisibly (review finding).
    #[allow(clippy::expect_used)]
    ciborium::ser::into_writer(&value, &mut buf)
        .expect("CBOR serialization of the hybrid external_aad is infallible");
    buf
}

/// Select the `external_aad` to bind into each layer for the given mode.
///
/// - `hybrid = true`  → bind the hybrid-composite alg-id ([`build_hybrid_external_aad`]).
/// - `hybrid = false` → bare `base_aad` (Classical; unchanged).
fn layer_aad(base_aad: &[u8], hybrid: bool) -> Vec<u8> {
    if hybrid {
        build_hybrid_external_aad(base_aad)
    } else {
        base_aad.to_vec()
    }
}

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
/// `payload ‖ inner_eddsa_signature` (the inner→outer binding).
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
        bail!(
            "nested COSE composite must have exactly 2 elements, got {}",
            arr.len()
        );
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

/// Compute the outer payload for the inner→outer binding: `payload ‖ inner_eddsa_sig`.
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
///   `payload ‖ inner_eddsa_signature` (the inner→outer binding).
pub fn sign_composite(
    ed_sk: &ed25519_dalek::SigningKey,
    pq_sk: Option<&MlDsaSigningKey>,
    payload: &[u8],
    external_aad: &[u8],
) -> Result<Vec<u8>> {
    // Hybrid iff an ML-DSA-65 key is present. In Hybrid mode the hybrid-composite
    // alg-id is bound into BOTH layers' external_aad (#278); in Classical mode the
    // bare base AAD is used so classical signatures are byte-identical to before.
    let hybrid = pq_sk.is_some();
    let aad = layer_aad(external_aad, hybrid);

    let inner = build_inner_eddsa(ed_sk, payload, &aad)?;
    let inner_sig = inner_signature_bytes(&inner).to_vec();
    let inner_bytes = inner
        .to_vec()
        .map_err(|e| anyhow!("failed to serialize inner EdDSA COSE_Sign1: {e}"))?;

    let outer_bytes = match pq_sk {
        Some(pq) => {
            let outer_pl = outer_payload(payload, &inner_sig);
            let outer = build_outer_mldsa(pq, &outer_pl, &aad)?;
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
// Out-of-band (async / WASM) signing API for the nested WNS composite.
//
// Abstract signers (e.g. the WASM `JsSigner`) cannot run inside a synchronous
// signing closure, so they compute each layer's to-be-signed bytes, sign them
// out-of-band, and assemble the composite here. The nesting REQUIRES the
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
///
/// `hybrid` MUST match the composite mode the caller will assemble: when `true`
/// the hybrid-composite alg-id is bound into the `external_aad` (#278), making
/// the inner TBS byte-distinct from a Classical one. Pass `false` for a
/// Classical-only composite.
pub fn inner_tbs(ed_kid: Vec<u8>, payload: &[u8], external_aad: &[u8], hybrid: bool) -> Vec<u8> {
    let aad = layer_aad(external_aad, hybrid);
    unsigned_inner(ed_kid).tbs_detached_data(payload, &aad)
}

/// Compute the outer ML-DSA-65 layer's detached to-be-signed bytes over
/// `payload ‖ inner_eddsa_signature` (the inner→outer binding).
///
/// The outer layer only exists in Hybrid mode, so the hybrid-composite alg-id is
/// always bound into its `external_aad` (#278) — symmetric with [`inner_tbs`]
/// called with `hybrid = true`.
pub fn outer_tbs(
    pq_kid: Vec<u8>,
    payload: &[u8],
    inner_eddsa_sig: &[u8],
    external_aad: &[u8],
) -> Vec<u8> {
    let aad = build_hybrid_external_aad(external_aad);
    let outer_pl = outer_payload(payload, inner_eddsa_sig);
    unsigned_outer(pq_kid).tbs_detached_data(&outer_pl, &aad)
}

/// Assemble the nested WNS composite from out-of-band signatures.
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
    /// Outer ML-DSA-65 layer was verified.
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
///
/// # Hybrid-alg-id binding (#278)
///
/// The inner/outer layers in a Hybrid composite bind the hybrid-composite alg-id
/// into their `external_aad` (see [`build_hybrid_external_aad`]). The verifier
/// reconstructs the inner AAD from the composite's *self-describing* shape: if an
/// outer ML-DSA layer is present, the inner was Hybrid-signed → reconstruct the
/// hybrid inner AAD; if the outer is absent (Classical or stripped), reconstruct
/// the bare base AAD. Consequently a Hybrid-signed inner whose outer layer has
/// been STRIPPED (outer = null) fails the inner EdDSA *signature* check (AAD
/// mismatch) — signature-level non-separability — independent of `require_pq`.
pub fn verify_composite(
    cose_composite_bytes: &[u8],
    expected_ed_vk: &ed25519_dalek::VerifyingKey,
    expected_pq_vk: Option<&MlDsaVerifyingKey>,
    payload: &[u8],
    external_aad: &[u8],
    require_pq: bool,
) -> Result<CompositeVerified> {
    let (inner_bytes, outer_bytes) = decode_composite(cose_composite_bytes)?;

    // The composite is self-describing: an outer ML-DSA layer means the inner was
    // produced in Hybrid mode (hybrid-alg-id bound into its AAD). Reconstruct the
    // inner AAD accordingly. A stripped Hybrid composite (outer=null) therefore
    // reconstructs the *bare* AAD and FAILS the inner signature check below.
    let inner_hybrid = outer_bytes.is_some();
    let inner_aad = layer_aad(external_aad, inner_hybrid);

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
        .verify_detached_signature(payload, &inner_aad, |sig, tbs| {
            ed_verify(expected_ed_vk, tbs, sig)
        })
        .context("inner EdDSA signature verification failed")?;
    let eddsa_ok = true;

    // Capture the inner signature; it is bound into the outer layer's signed message.
    let inner_sig = inner_signature_bytes(&inner).to_vec();

    // --- Outer ML-DSA-65 layer (inner→outer binding) ---
    let mut ml_dsa_ok = false;

    match outer_bytes {
        Some(ref ob) if require_pq || expected_pq_vk.is_some() => {
            // Anchored PQ key is mandatory to verify the outer layer.
            let Some(pq_vk) = expected_pq_vk else {
                if require_pq {
                    bail!("Hybrid policy requires an anchored ML-DSA-65 key, none provided");
                }
                // Classical verifier without an anchor: ignore the outer layer.
                return Ok(CompositeVerified {
                    eddsa: eddsa_ok,
                    ml_dsa: false,
                });
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
            // The outer layer only exists in Hybrid mode, so it always binds the
            // hybrid-composite alg-id into its external_aad (#278).
            let outer_aad = build_hybrid_external_aad(external_aad);
            outer
                .verify_detached_signature(&outer_pl, &outer_aad, |sig, tbs| {
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

    Ok(CompositeVerified {
        eddsa: eddsa_ok,
        ml_dsa: ml_dsa_ok,
    })
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
        let res =
            verify_composite(&cose, &ed.verifying_key(), None, payload, &aad(), false).unwrap();
        assert!(res.eddsa);
        assert!(!res.ml_dsa);
    }

    #[test]
    fn hybrid_sign_verify_both() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let payload = b"hybrid payload";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        let res = verify_composite(
            &cose,
            &ed.verifying_key(),
            Some(&pq_vk),
            payload,
            &aad(),
            true,
        )
        .unwrap();
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
        let res =
            verify_composite(&cose, &ed.verifying_key(), None, payload, &aad(), false).unwrap();
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
        let res = verify_composite(
            &cose,
            &ed.verifying_key(),
            Some(&pq_vk),
            payload,
            &aad(),
            true,
        );
        assert!(
            res.is_err(),
            "hybrid policy must reject classical-only (no outer layer)"
        );
    }

    #[test]
    fn self_cert_wrong_pq_key_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let (_other_sk, other_vk) = ml_dsa_generate_keypair();
        let payload = b"anchored";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        let res = verify_composite(
            &cose,
            &ed.verifying_key(),
            Some(&other_vk),
            payload,
            &aad(),
            true,
        );
        assert!(
            res.is_err(),
            "PQ key not matching kid-anchored key must be rejected"
        );
    }

    #[test]
    fn tampered_payload_rejected() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let cose = sign_composite(&ed, Some(&pq_sk), b"orig", &aad()).unwrap();
        let res = verify_composite(
            &cose,
            &ed.verifying_key(),
            Some(&pq_vk),
            b"tampered",
            &aad(),
            true,
        );
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

    /// Policy-enforced: stripping the outer ML-DSA layer (set to null) must be rejected
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

        let res = verify_composite(
            &stripped,
            &ed.verifying_key(),
            Some(&pq_vk),
            payload,
            &aad(),
            true,
        );
        assert!(
            res.is_err(),
            "stripping the outer ML-DSA layer must fail Hybrid policy"
        );
    }

    /// inner→outer binding: tampering with the inner EdDSA signature invalidates the outer,
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
        let res = verify_composite(
            &tampered,
            &ed.verifying_key(),
            Some(&pq_vk),
            payload,
            &aad(),
            true,
        );
        assert!(
            res.is_err(),
            "tampering the inner EdDSA must invalidate the composite"
        );
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

    // ── #278: hybrid-composite alg-id binding (signature-level non-separability) ──

    /// The hybrid AAD must differ from the bare base AAD and embed the alg-id.
    #[test]
    fn hybrid_external_aad_binds_alg_id_and_differs() {
        let base = aad();
        let hybrid = build_hybrid_external_aad(&base);
        assert_ne!(
            hybrid, base,
            "hybrid AAD must differ from the bare base AAD"
        );
        // Determinism.
        assert_eq!(hybrid, build_hybrid_external_aad(&base));
        // The composite alg-id bytes appear in the encoded hybrid AAD.
        let needle = HYBRID_COMPOSITE_ALG_ID;
        assert!(
            hybrid.windows(needle.len()).any(|w| w == needle),
            "hybrid AAD must embed HYBRID_COMPOSITE_ALG_ID"
        );
    }

    /// The inner EdDSA TBS / signature bytes in Hybrid mode must be byte-distinct
    /// from Classical mode over the same payload + base AAD.
    #[test]
    fn inner_eddsa_bytes_hybrid_distinct_from_classical() {
        let ed = SigningKey::generate(&mut OsRng);
        let payload = b"distinctness";
        let kid = ed.verifying_key().to_bytes().to_vec();

        // TBS bytes differ by mode.
        let tbs_classical = inner_tbs(kid.clone(), payload, &aad(), false);
        let tbs_hybrid = inner_tbs(kid, payload, &aad(), true);
        assert_ne!(
            tbs_classical, tbs_hybrid,
            "inner EdDSA TBS in Hybrid mode must differ from Classical mode"
        );

        // Signature bytes differ by mode (Ed25519 is deterministic, so equal TBS
        // would yield equal sigs; distinct TBS yields distinct sigs).
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let classical = sign_composite(&ed, None, payload, &aad()).unwrap();
        let hybrid = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();
        let (inner_c, _) = decode_composite(&classical).unwrap();
        let (inner_h, _) = decode_composite(&hybrid).unwrap();
        let sig_c = CoseSign1::from_slice(&inner_c).unwrap().signature;
        let sig_h = CoseSign1::from_slice(&inner_h).unwrap().signature;
        assert_ne!(
            sig_c, sig_h,
            "inner EdDSA signature bytes must differ between Hybrid and Classical"
        );
    }

    /// SIGNATURE-LEVEL non-separability: a Hybrid-signed composite with the outer
    /// ML-DSA layer stripped must FAIL standalone-Classical verification at the
    /// *signature* level (not merely the require_pq policy check).
    #[test]
    fn stripped_hybrid_inner_fails_classical_signature_verification() {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq_sk, _pq_vk) = ml_dsa_generate_keypair();
        let payload = b"strip and downgrade";
        let cose = sign_composite(&ed, Some(&pq_sk), payload, &aad()).unwrap();

        // Strip the outer layer → outer = null.
        let (inner, _outer) = decode_composite(&cose).unwrap();
        let stripped = encode_composite(inner, None).unwrap();

        // Classical verifier (require_pq=false, no PQ anchor): would have ACCEPTED
        // the inner EdDSA before #278. Now the inner was bound to the hybrid-alg-id,
        // so reconstructing the bare base AAD fails the signature check.
        let res = verify_composite(&stripped, &ed.verifying_key(), None, payload, &aad(), false);
        assert!(
            res.is_err(),
            "stripped Hybrid inner must fail standalone Classical signature verification"
        );
        let msg = format!("{:#}", res.unwrap_err());
        assert!(
            msg.contains("inner EdDSA signature verification failed"),
            "failure must be at the signature level, got: {msg}"
        );
    }

    /// Regression guard: a genuine Classical-signed inner still verifies under a
    /// Classical verifier (no hybrid-alg-id involved on either side).
    #[test]
    fn classical_roundtrip_unchanged_after_278() {
        let ed = SigningKey::generate(&mut OsRng);
        let payload = b"classical unchanged";
        let cose = sign_composite(&ed, None, payload, &aad()).unwrap();
        let res =
            verify_composite(&cose, &ed.verifying_key(), None, payload, &aad(), false).unwrap();
        assert!(res.eddsa);
        assert!(!res.ml_dsa);
    }

    /// A genuine Classical inner is NOT mistaken for a Hybrid one: verifying a
    /// Classical composite with the hybrid AAD reconstruction would fail, but the
    /// verifier keys the inner AAD off outer-presence, so the normal Classical
    /// path succeeds (covered above) while a hybrid-AAD reconstruction is rejected.
    #[test]
    fn classical_inner_not_accepted_under_hybrid_aad() {
        let ed = SigningKey::generate(&mut OsRng);
        let payload = b"no false hybrid";
        let cose = sign_composite(&ed, None, payload, &aad()).unwrap();
        // Decode the classical inner and re-verify it directly against the HYBRID
        // AAD to prove the two TBS inputs are genuinely separated.
        let (inner, _) = decode_composite(&cose).unwrap();
        let inner_cose = CoseSign1::from_slice(&inner).unwrap();
        let hybrid_aad = build_hybrid_external_aad(&aad());
        let res = inner_cose.verify_detached_signature(payload, &hybrid_aad, |sig, tbs| {
            ed_verify(&ed.verifying_key(), tbs, sig)
        });
        assert!(
            res.is_err(),
            "classical inner must not verify under the hybrid AAD"
        );
    }
}
