//! Policy-gated composite signing and verification for `did:at9p` records
//! (capsules and update-records), per design #879 §4.4 (Hybrid-pinning) and
//! review rules R4/R8 (findings A18/A19).
//!
//! # GATE 2: Hybrid is PINNED, not configurable
//!
//! `CryptoPolicy::{Hybrid, Classical}` is per-node runtime configuration for
//! the general RPC plane. at9p records do **not** honor it: capsule,
//! update-record, and manifest signatures **always** require BOTH composite
//! components (EdDSA + ML-DSA-65) to verify, regardless of the node's policy.
//! A `Classical`-configured node still rejects an Ed25519-only-valid capsule
//! (§4.4, A19/R4) — otherwise an attacker holding only a stolen classical key
//! could forge identities for every such node.
//!
//! The pin is built into this module's API surface so it cannot be bypassed:
//!
//! - **Signing** takes a `&MlDsaSigningKey` by value, never an `Option`. There
//!   is no code path that calls the underlying
//!   [`hyprstream_crypto::cose_sign::sign_composite`] with `pq_sk = None`
//!   (review rule R4 — `sign_composite` does *not* self-enforce policy; its
//!   callers do). The one call site is [`sign_record`], and it always passes
//!   `Some(pq_sk)`.
//! - **Verification** always calls
//!   [`hyprstream_crypto::cose_sign::verify_composite`] with `require_pq = true`
//!   and a non-`None` anchored ML-DSA-65 key. A missing outer layer (stripped
//!   or classical-only) fails closed.
//!
//! # Per-record-type COSE context strings (R8, A18)
//!
//! Every at9p record type binds its context string
//! (`at9p-capsule/1`, `at9p-update/1`) into the COSE protected header, which is
//! folded into BOTH signature layers as `external_aad` (RFC 9052 §4.4). The
//! context is **checked** on verify against the record type being verified, so
//! a signature minted for one record type (or an RPC envelope) cannot be
//! replayed as another — cross-context signature reuse (A18) is defeated.

use anyhow::{ensure, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};

use hyprstream_crypto::cose_sign::{
    assemble_composite_nested, sign_composite, split_composite, verify_composite,
};
use hyprstream_crypto::pq::{
    ml_dsa_sk_to_vk_bytes, ml_dsa_vk_bytes, ml_dsa_vk_from_bytes, MlDsaSigningKey,
    MlDsaVerifyingKey,
};

use crate::at9p::{
    Capsule, CapsuleBody, CoseCompositeSignature, HybridKeyPair, UpdateRecord,
    CAPSULE_SIGNATURE_CONTEXT, ED25519_PUBLIC_KEY_LEN, UPDATE_SIGNATURE_CONTEXT,
};
use crate::dag_cbor::DagCbor;

/// Composite/hybrid algorithm identifier bound into every at9p protected
/// header. Matches the LAMPS composite name the underlying nested COSE
/// construction binds (`draft-ietf-lamps-pq-composite-sigs`,
/// `id-MLDSA65-Ed25519`). Pinning it here documents on the wire that at9p
/// records are hybrid-only.
const AT9P_COMPOSITE_ALG_ID: &str = "id-MLDSA65-Ed25519";

/// Build the deterministic COSE protected-header bytes that bind the
/// composite alg-id and the record-type `context` string.
///
/// These bytes are both stored in `CoseCompositeSignature.protected` and used
/// as the `external_aad` for the composite signature, so the context is folded
/// into both signature layers. Encoded with the crate's canonical DAG-CBOR
/// codec (sorted keys, minimal ints) so it is byte-stable across signer and
/// verifier.
fn context_protected_header(context: &str) -> Vec<u8> {
    DagCbor::str_map([
        ("alg", DagCbor::Text(AT9P_COMPOSITE_ALG_ID.to_owned())),
        ("context", DagCbor::Text(context.to_owned())),
    ])
    .encode()
}

/// Derive the concrete verifying keys from a declared hybrid keypair.
fn verifying_keys(keypair: &HybridKeyPair) -> Result<(VerifyingKey, MlDsaVerifyingKey)> {
    keypair.validate()?;
    let ed_bytes: [u8; ED25519_PUBLIC_KEY_LEN] = keypair
        .ed25519_pub
        .as_slice()
        .try_into()
        .context("ed25519 subject key must be 32 bytes")?;
    let ed_vk = VerifyingKey::from_bytes(&ed_bytes)
        .context("declared ed25519 subject key is not a valid point")?;
    let pq_vk = ml_dsa_vk_from_bytes(&keypair.mldsa65_pub)
        .context("declared ML-DSA-65 subject key is invalid")?;
    Ok((ed_vk, pq_vk))
}

/// Sign `payload` under the pinned-Hybrid composite path, binding `context`.
///
/// R4: the sole `sign_composite` call site — always `Some(pq_sk)`, never the
/// classical-only production.
fn sign_record(
    payload: &[u8],
    context: &str,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<CoseCompositeSignature> {
    let protected = context_protected_header(context);
    // GATE 2 / R4: Hybrid pinned — pq_sk is mandatory, so the composite always
    // carries the outer ML-DSA-65 layer. `sign_composite` is never invoked with
    // `pq_sk = None` from at9p code.
    let composite = sign_composite(ed_sk, Some(pq_sk), payload, &protected)
        .context("at9p composite signing failed")?;

    let (ed_sig, pq_sig) = split_composite(&composite)?;
    let pq_sig = pq_sig
        .ok_or_else(|| anyhow::anyhow!("at9p composite signing produced no ML-DSA-65 layer"))?;

    CoseCompositeSignature::new(context, protected, ed_sig, pq_sig)
}

/// Verify a decomposed at9p composite signature under PINNED Hybrid policy.
///
/// Enforces (independent of node `CryptoPolicy`):
/// 1. the stored context matches `expected_context` (R8/A18);
/// 2. the stored protected header matches the canonical binding for that
///    context (defense in depth);
/// 3. BOTH the inner EdDSA and outer ML-DSA-65 layers verify against the
///    anchored keys over `payload` (`require_pq = true`).
fn verify_record(
    payload: &[u8],
    sig: &CoseCompositeSignature,
    expected_context: &str,
    ed_vk: &VerifyingKey,
    pq_vk: &MlDsaVerifyingKey,
) -> Result<()> {
    // R8/A18: the context string is CHECKED, not decorative.
    ensure!(
        sig.context == expected_context,
        "at9p signature context mismatch: expected {expected_context:?}, got {:?}",
        sig.context
    );

    // The external_aad is derived from the EXPECTED record type, not from the
    // attacker-controlled stored bytes: a signature minted under a different
    // context reconstructs a different aad and fails the crypto check below,
    // even if its `context`/`protected` fields were rewritten.
    let expected_protected = context_protected_header(expected_context);
    ensure!(
        sig.protected == expected_protected,
        "at9p signature protected header does not match the canonical binding for {expected_context:?}"
    );

    // Reassemble the nested composite from the decomposed raw signatures. The
    // kids are the anchored public keys; the protected headers are rebuilt
    // deterministically, so the inner→outer binding round-trips exactly.
    let ed_kid = ed_vk.to_bytes().to_vec();
    let pq_kid = ml_dsa_vk_bytes(pq_vk);
    let composite = assemble_composite_nested(
        (ed_kid, sig.ed25519_signature.clone()),
        Some((pq_kid, sig.mldsa65_signature.clone())),
    )
    .context("failed to reassemble at9p composite for verification")?;

    // GATE 2: Hybrid PINNED — require_pq = true, always. A stripped or
    // classical-only composite fails closed regardless of node policy.
    let verified = verify_composite(
        &composite,
        ed_vk,
        Some(pq_vk),
        payload,
        &expected_protected,
        /* require_pq = */ true,
    )
    .context("at9p composite verification failed")?;

    // Belt-and-suspenders: verify_composite already fails closed when either
    // layer is missing under require_pq, but assert the invariant explicitly.
    ensure!(
        verified.eddsa && verified.ml_dsa,
        "at9p verification did not confirm both composite layers (eddsa={}, ml_dsa={})",
        verified.eddsa,
        verified.ml_dsa
    );
    Ok(())
}

/// Sign a capsule body with its primary subject key, producing a fully-signed
/// [`Capsule`]. Self-certifying: the signing keys MUST match
/// `body.subject_keys[0]`.
pub fn sign_capsule(
    body: CapsuleBody,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<Capsule> {
    body.validate()?;
    let primary = body
        .subject_keys
        .first()
        .ok_or_else(|| anyhow::anyhow!("capsule has no subject keys"))?;
    ensure!(
        primary.ed25519_pub == ed_sk.verifying_key().to_bytes(),
        "signing ed25519 key does not match the capsule primary subject key"
    );
    ensure!(
        primary.mldsa65_pub == ml_dsa_sk_to_vk_bytes(pq_sk),
        "signing ML-DSA-65 key does not match the capsule primary subject key"
    );

    let payload = body.to_dag_cbor();
    let signatures = sign_record(&payload, CAPSULE_SIGNATURE_CONTEXT, ed_sk, pq_sk)?;
    Capsule::new(body, signatures)
}

/// Verify a genesis/self-certifying [`Capsule`]: the composite signature must
/// verify (pinned Hybrid) against the capsule's own primary subject key.
pub fn verify_capsule(capsule: &Capsule) -> Result<()> {
    capsule.body.validate()?;
    let primary = capsule
        .body
        .subject_keys
        .first()
        .ok_or_else(|| anyhow::anyhow!("capsule has no subject keys"))?;
    let (ed_vk, pq_vk) = verifying_keys(primary)?;
    verify_record(
        &capsule.body.to_dag_cbor(),
        &capsule.signatures,
        CAPSULE_SIGNATURE_CONTEXT,
        &ed_vk,
        &pq_vk,
    )
}

/// Sign an update-record's content with the authorizing hybrid keypair,
/// producing a fully-signed [`UpdateRecord`].
///
/// The signature covers everything except the `signatures` field. Which key is
/// *authorized* to sign a rotation (the pre-rotation subject key) is the
/// rotation-chain gate's concern (#884); this function only mints the
/// pinned-Hybrid, context-bound signature over the supplied content.
#[allow(clippy::too_many_arguments)]
pub fn sign_update_record(
    subject_cid512: String,
    epoch: u64,
    prev_record_digest: [u8; crate::at9p::H512_LEN],
    new_capsule_body: CapsuleBody,
    expires_at: String,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<UpdateRecord> {
    new_capsule_body.validate()?;
    // Placeholder signature so we can build the record and compute its
    // canonical signable bytes; it is replaced by the real signature below.
    let placeholder = CoseCompositeSignature::new(
        UPDATE_SIGNATURE_CONTEXT,
        context_protected_header(UPDATE_SIGNATURE_CONTEXT),
        vec![0u8; crate::at9p::ED25519_SIGNATURE_LEN],
        vec![0u8; crate::at9p::ML_DSA65_SIGNATURE_LEN],
    )?;
    let mut record = UpdateRecord {
        subject_cid512,
        epoch,
        prev_record_digest,
        new_capsule_body,
        expires_at,
        signatures: placeholder,
    };
    let payload = record.signable_bytes();
    record.signatures = sign_record(&payload, UPDATE_SIGNATURE_CONTEXT, ed_sk, pq_sk)?;
    // Round-trip through the canonical codec to enforce every schema gate.
    UpdateRecord::from_dag_cbor(&record.to_dag_cbor())
}

/// Verify an [`UpdateRecord`] under PINNED Hybrid policy against the authorizing
/// `signer` keypair. The caller (rotation-chain validator, #884) decides which
/// key is authorized; this function enforces the crypto + context binding.
pub fn verify_update_record(record: &UpdateRecord, signer: &HybridKeyPair) -> Result<()> {
    let (ed_vk, pq_vk) = verifying_keys(signer)?;
    verify_record(
        &record.signable_bytes(),
        &record.signatures,
        UPDATE_SIGNATURE_CONTEXT,
        &ed_vk,
        &pq_vk,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::at9p::{
        ServiceEndpoint, ServiceEntry, ServiceType, Transport, ML_DSA65_SIGNATURE_LEN,
    };
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::ml_dsa_generate_keypair;
    use rand::rngs::OsRng;

    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer() -> Signer {
        let ed_sk = SigningKey::generate(&mut OsRng);
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

    fn capsule_body(keypair: HybridKeyPair) -> CapsuleBody {
        let endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://node0").unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        CapsuleBody::new(vec![keypair], vec![service]).unwrap()
    }

    fn signed_capsule() -> (Signer, Capsule) {
        let s = signer();
        let body = capsule_body(s.keypair.clone());
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        (s, capsule)
    }

    fn signed_update(s: &Signer) -> UpdateRecord {
        let body = capsule_body(s.keypair.clone());
        let subject_cid512 = Capsule::new(
            body.clone(),
            // any valid-shaped signature just to derive a cid for the field
            sign_record(
                &body.to_dag_cbor(),
                CAPSULE_SIGNATURE_CONTEXT,
                &s.ed_sk,
                &s.pq_sk,
            )
            .unwrap(),
        )
        .unwrap()
        .cid512()
        .unwrap();
        sign_update_record(
            subject_cid512,
            7,
            [9u8; crate::at9p::H512_LEN],
            body,
            "2026-07-09T00:00:00Z".to_owned(),
            &s.ed_sk,
            &s.pq_sk,
        )
        .unwrap()
    }

    #[test]
    fn capsule_sign_verify_round_trip() {
        let (_s, capsule) = signed_capsule();
        verify_capsule(&capsule).expect("capsule must verify");
        // Survives a canonical encode/decode round-trip.
        let bytes = capsule.to_dag_cbor();
        let decoded = Capsule::from_dag_cbor(&bytes).unwrap();
        verify_capsule(&decoded).expect("decoded capsule must verify");
    }

    #[test]
    fn update_record_sign_verify_round_trip() {
        let s = signer();
        let record = signed_update(&s);
        verify_update_record(&record, &s.keypair).expect("update must verify");
        let bytes = record.to_dag_cbor();
        let decoded = UpdateRecord::from_dag_cbor(&bytes).unwrap();
        verify_update_record(&decoded, &s.keypair).expect("decoded update must verify");
    }

    /// A18/R8: a capsule signature replayed as an update signature (and vice
    /// versa) is rejected — the context binding differs.
    #[test]
    fn cross_type_replay_rejected() {
        let (s, capsule) = signed_capsule();
        // Take the capsule's valid composite and try to pass it off as an
        // update signature over the same body bytes.
        let mut forged = capsule.signatures.clone();
        // Rewrite the fields to claim it is an update signature; the crypto aad
        // for the update context won't match, so it must still fail.
        forged.context = UPDATE_SIGNATURE_CONTEXT.to_owned();
        forged.protected = context_protected_header(UPDATE_SIGNATURE_CONTEXT);
        let (ed_vk, pq_vk) = verifying_keys(&s.keypair).unwrap();
        let res = verify_record(
            &capsule.body.to_dag_cbor(),
            &forged,
            UPDATE_SIGNATURE_CONTEXT,
            &ed_vk,
            &pq_vk,
        );
        assert!(res.is_err(), "cross-context signature reuse must be rejected");

        // And verifying the genuine capsule signature while EXPECTING the update
        // context is rejected too.
        let res2 = verify_record(
            &capsule.body.to_dag_cbor(),
            &capsule.signatures,
            UPDATE_SIGNATURE_CONTEXT,
            &ed_vk,
            &pq_vk,
        );
        assert!(res2.is_err(), "wrong expected context must be rejected");
    }

    /// A19/R4: a classical-only (Ed25519-only) capsule is rejected even under a
    /// node configured `Classical`. There is no node-policy input to
    /// `verify_capsule` at all — Hybrid is pinned — so we prove it by crafting a
    /// classical-only composite and confirming it cannot verify.
    #[test]
    fn classical_only_signature_rejected_under_classical_policy() {
        let s = signer();
        let body = capsule_body(s.keypair.clone());
        let payload = body.to_dag_cbor();
        let protected = context_protected_header(CAPSULE_SIGNATURE_CONTEXT);

        // Directly emit a CLASSICAL-only composite (pq_sk = None) — the exact
        // thing R4 forbids in at9p code, done here only to prove verification
        // rejects it. Extract the EdDSA sig; there is no ML-DSA layer, so we
        // cannot even construct a schema-valid CoseCompositeSignature (which
        // requires a 3309-byte ML-DSA sig). Substitute a bogus ML-DSA sig of the
        // right LENGTH to get past the schema and prove the CRYPTO gate rejects.
        let classical = sign_composite(&s.ed_sk, None, &payload, &protected).unwrap();
        let (ed_sig, pq_sig) = split_composite(&classical).unwrap();
        assert!(pq_sig.is_none(), "classical composite has no ML-DSA layer");

        let forged = CoseCompositeSignature::new(
            CAPSULE_SIGNATURE_CONTEXT,
            protected,
            ed_sig,
            vec![0u8; ML_DSA65_SIGNATURE_LEN],
        )
        .unwrap();
        let capsule = Capsule::new(body, forged).unwrap();

        // verify_capsule pins Hybrid: the bogus ML-DSA layer cannot verify.
        let res = verify_capsule(&capsule);
        assert!(
            res.is_err(),
            "classical-only / missing-PQ capsule must be rejected under pinned Hybrid"
        );
    }

    /// Tampering the signed body invalidates the signature.
    #[test]
    fn tampered_payload_rejected() {
        let (s, capsule) = signed_capsule();
        let (ed_vk, pq_vk) = verifying_keys(&s.keypair).unwrap();
        let mut tampered = capsule.body.to_dag_cbor();
        // Flip a byte in the payload.
        let last = tampered.len() - 1;
        tampered[last] ^= 0x01;
        let res = verify_record(
            &tampered,
            &capsule.signatures,
            CAPSULE_SIGNATURE_CONTEXT,
            &ed_vk,
            &pq_vk,
        );
        assert!(res.is_err(), "tampered payload must fail verification");
    }

    /// A signature verifies only against the key it was minted with; a
    /// different subject key is rejected.
    #[test]
    fn wrong_key_rejected() {
        let (_s, capsule) = signed_capsule();
        let other = signer();
        let res = verify_update_record(
            &UpdateRecord {
                subject_cid512: capsule.cid512().unwrap(),
                epoch: 0,
                prev_record_digest: [0u8; crate::at9p::H512_LEN],
                new_capsule_body: capsule.body.clone(),
                expires_at: "2026-07-09T00:00:00Z".to_owned(),
                signatures: capsule.signatures.clone(),
            },
            &other.keypair,
        );
        assert!(res.is_err(), "verification under the wrong key must fail");
    }

    /// The API forbids classical-only signing by construction: `sign_capsule`
    /// requires a `&MlDsaSigningKey`, so a valid capsule always carries a
    /// verifiable ML-DSA-65 layer. This asserts the produced signature has a
    /// real (non-zero, correct-length) PQ component that verifies.
    #[test]
    fn signing_always_hybrid() {
        let (_s, capsule) = signed_capsule();
        assert_eq!(
            capsule.signatures.mldsa65_signature.len(),
            ML_DSA65_SIGNATURE_LEN
        );
        assert!(
            capsule.signatures.mldsa65_signature.iter().any(|&b| b != 0),
            "ML-DSA-65 signature must be a real signature, not a placeholder"
        );
        verify_capsule(&capsule).unwrap();
    }
}
