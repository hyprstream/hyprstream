//! COSE_Sign1 (RFC 9052) detached-payload signing for the hyprstream
//! `SignedEnvelope`.
//!
//! Stage C of Phase 0.5 migrates envelope authentication from a bespoke
//! `Ed25519.sign(serialize(envelope))` scheme to standards-based
//! COSE_Sign1 with a detached payload and a domain-separating
//! `external_aad`.
//!
//! # Wire shape
//!
//! Each `SignedEnvelope.signatures` entry is the CBOR encoding of a
//! `COSE_Sign1` structure (see RFC 9052 §4.2). The signed payload is the
//! Cap'n Proto-canonical bytes of the `RequestEnvelope`, *detached* — it
//! is not embedded in the COSE structure itself. The signer's identity
//! and algorithm are conveyed via the protected header (`kid`, `alg`).
//!
//! # external_aad
//!
//! To prevent schema-confusion (signing one schema's bytes and replaying
//! them as another schema), `external_aad` is computed as the CBOR
//! encoding of `[envelope_schema_id, inner_type_id]` — two Cap'n Proto
//! file/type IDs as `u64`s. Sender and receiver must agree on these
//! IDs; the verifier passes them in, and a mismatch causes verification
//! to fail with `InvalidSignature`.

use anyhow::{anyhow, bail, Context, Result};
use coset::{
    cbor::Value as CborValue,
    iana::{self, EnumI64},
    CborSerializable, CoseSign1, CoseSign1Builder, HeaderBuilder, Label,
};

/// Build the COSE `external_aad` for envelope signing.
///
/// Encoded as CBOR `[envelope_schema_id, inner_type_id]` — both `u64`.
/// Receivers MUST pass the same values they expect to verify; otherwise
/// signature verification fails.
pub fn build_external_aad(envelope_schema_id: u64, inner_type_id: u64) -> Vec<u8> {
    let value = CborValue::Array(vec![
        CborValue::Integer(envelope_schema_id.into()),
        CborValue::Integer(inner_type_id.into()),
    ]);
    let mut buf = Vec::with_capacity(24);
    // CBOR encoding of a fixed two-integer array writes into an
    // unbounded Vec; the writer is infallible for this shape, but we
    // still drop any error rather than panic.
    if ciborium::ser::into_writer(&value, &mut buf).is_err() {
        return Vec::new();
    }
    buf
}

/// Trait abstracting over a signing key.
///
/// Implemented for `ed25519_dalek::SigningKey` below; ML-DSA-65 / PQ
/// support can be added later by implementing this trait + extending
/// `alg_for_signer`.
pub trait CoseSigner {
    /// IANA COSE algorithm identifier (e.g. `iana::Algorithm::EdDSA`).
    fn alg(&self) -> iana::Algorithm;

    /// Stable key identifier used for the COSE `kid` protected header.
    ///
    /// Convention: for Ed25519 signers this is the 32-byte raw verifying
    /// key. Verifiers look the signer up by this `kid`.
    fn kid(&self) -> Vec<u8>;

    /// Sign arbitrary bytes (the COSE `Sig_structure` value).
    fn sign(&self, msg: &[u8]) -> Vec<u8>;
}

/// Trait abstracting over a verifying key.
pub trait CoseVerifier {
    fn alg(&self) -> iana::Algorithm;
    fn verify(&self, msg: &[u8], sig: &[u8]) -> Result<()>;
}

impl CoseSigner for ed25519_dalek::SigningKey {
    fn alg(&self) -> iana::Algorithm {
        iana::Algorithm::EdDSA
    }
    fn kid(&self) -> Vec<u8> {
        self.verifying_key().to_bytes().to_vec()
    }
    fn sign(&self, msg: &[u8]) -> Vec<u8> {
        use ed25519_dalek::Signer;
        Signer::sign(self, msg).to_bytes().to_vec()
    }
}

impl CoseVerifier for ed25519_dalek::VerifyingKey {
    fn alg(&self) -> iana::Algorithm {
        iana::Algorithm::EdDSA
    }
    fn verify(&self, msg: &[u8], sig: &[u8]) -> Result<()> {
        if sig.len() != 64 {
            bail!("Ed25519 signature must be 64 bytes, got {}", sig.len());
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(sig);
        let signature = ed25519_dalek::Signature::from_bytes(&arr);
        self.verify_strict(msg, &signature)
            .map_err(|e| anyhow!("Ed25519 verification failed: {e}"))
    }
}

/// Sign `payload` (detached) with `signing_key`, producing CBOR-encoded
/// `COSE_Sign1` bytes.
///
/// The protected header contains the algorithm + `kid`. `external_aad`
/// is folded into the `Sig_structure` per RFC 9052 §4.4 but is not
/// transmitted in the COSE object — receivers must reconstruct it.
pub fn sign_detached<S: CoseSigner>(
    signing_key: &S,
    payload: &[u8],
    external_aad: &[u8],
) -> Result<Vec<u8>> {
    let protected = HeaderBuilder::new()
        .algorithm(signing_key.alg())
        .key_id(signing_key.kid())
        .build();

    let cose = CoseSign1Builder::new()
        .protected(protected)
        .create_detached_signature(payload, external_aad, |bytes| signing_key.sign(bytes))
        .build();

    cose.to_vec()
        .map_err(|e| anyhow!("failed to serialize COSE_Sign1: {e}"))
}

/// Verify a CBOR-encoded `COSE_Sign1` against the detached `payload`
/// and `external_aad`.
///
/// On success returns `(kid_b64, alg_label)` where `kid_b64` is the
/// signer key id (base64url, no padding) and `alg_label` is the
/// well-known short name (e.g. `"EdDSA"`). Returns `Err` for malformed
/// COSE, alg mismatch, missing `kid`, or signature failure.
pub fn verify_detached<V: CoseVerifier>(
    cose_sign1_bytes: &[u8],
    verifying_key: &V,
    payload: &[u8],
    external_aad: &[u8],
) -> Result<(String, String)> {
    let cose = CoseSign1::from_slice(cose_sign1_bytes)
        .map_err(|e| anyhow!("malformed COSE_Sign1: {e}"))?;

    // Extract alg from protected header
    let alg = cose
        .protected
        .header
        .alg
        .as_ref()
        .ok_or_else(|| anyhow!("COSE_Sign1 missing alg in protected header"))?;
    let alg_int = match alg {
        coset::Algorithm::Assigned(a) => a.to_i64(),
        coset::Algorithm::PrivateUse(v) => *v,
        coset::Algorithm::Text(s) => bail!("unsupported text algorithm: {s}"),
    };
    let expected_alg_int = verifying_key.alg().to_i64();
    if alg_int != expected_alg_int {
        bail!("COSE_Sign1 alg mismatch: header={alg_int} expected={expected_alg_int}");
    }
    let alg_label = format!("{:?}", verifying_key.alg());

    // Extract kid from protected header
    let kid_bytes = &cose.protected.header.key_id;
    if kid_bytes.is_empty() {
        bail!("COSE_Sign1 missing kid in protected header");
    }
    use base64::Engine;
    let kid_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(kid_bytes);

    // Verify detached signature
    cose.verify_detached_signature(payload, external_aad, |sig, data| {
        verifying_key.verify(data, sig)
    })
    .context("COSE_Sign1 detached signature verification failed")?;

    Ok((kid_b64, alg_label))
}

/// Inspect a COSE_Sign1 without verifying it, returning `(kid_bytes, alg_int)`.
///
/// Used by receivers to dispatch a verifier from a key registry.
pub fn peek_kid_alg(cose_sign1_bytes: &[u8]) -> Result<(Vec<u8>, i64)> {
    let cose = CoseSign1::from_slice(cose_sign1_bytes)
        .map_err(|e| anyhow!("malformed COSE_Sign1: {e}"))?;
    let alg = cose
        .protected
        .header
        .alg
        .as_ref()
        .ok_or_else(|| anyhow!("COSE_Sign1 missing alg"))?;
    let alg_int = match alg {
        coset::Algorithm::Assigned(a) => a.to_i64(),
        coset::Algorithm::PrivateUse(v) => *v,
        coset::Algorithm::Text(_) => bail!("text algorithms unsupported"),
    };
    let kid = cose.protected.header.key_id.clone();
    if kid.is_empty() {
        bail!("COSE_Sign1 missing kid");
    }
    Ok((kid, alg_int))
}

// Silence dead-code on the Label re-export until we exercise unprotected headers.
#[allow(dead_code)]
fn _label_marker(_: Label) {}

// ────────────────────────────────────────────────────────────────────────────
// Ed25519-specific concrete API for the envelope-signing call sites.
//
// `envelope.rs` was wired to a concrete-typed API (parse/sign_ed25519_detached/
// verify_ed25519_detached) before this module's generic trait-based API
// existed. The wrappers below preserve that call-site shape so the envelope
// migration can land without churning unrelated code.
// ────────────────────────────────────────────────────────────────────────────

/// Parsed COSE_Sign1 with kid + alg extracted from the protected header.
pub struct ParsedCoseSign1 {
    /// The decoded COSE_Sign1 structure (signature bytes accessible via `.signature`).
    pub sign1: CoseSign1,
    /// Signer key id from the protected header.
    pub kid: Vec<u8>,
    /// IANA COSE algorithm integer (e.g. `-8` for EdDSA).
    pub alg: i64,
}

/// Parse a CBOR-encoded COSE_Sign1 blob. Does not verify the signature.
pub fn parse(blob: &[u8]) -> Result<ParsedCoseSign1> {
    let sign1 = CoseSign1::from_slice(blob).map_err(|e| anyhow!("malformed COSE_Sign1: {e}"))?;
    let alg = sign1
        .protected
        .header
        .alg
        .as_ref()
        .ok_or_else(|| anyhow!("COSE_Sign1 missing alg"))?;
    let alg_int = match alg {
        coset::Algorithm::Assigned(a) => a.to_i64(),
        coset::Algorithm::PrivateUse(v) => *v,
        coset::Algorithm::Text(_) => bail!("text algorithms unsupported"),
    };
    let kid = sign1.protected.header.key_id.clone();
    if kid.is_empty() {
        bail!("COSE_Sign1 missing kid");
    }
    Ok(ParsedCoseSign1 {
        sign1,
        kid,
        alg: alg_int,
    })
}

/// Concrete Ed25519 detached-signature wrapper for envelope.rs.
///
/// Builds the `external_aad` from `(envelope_schema_id, inner_type_id)` and
/// produces a CBOR-encoded COSE_Sign1 over `payload`.
pub fn sign_ed25519_detached(
    signing_key: &ed25519_dalek::SigningKey,
    envelope_schema_id: u64,
    inner_type_id: u64,
    payload: &[u8],
) -> Result<Vec<u8>> {
    let aad = build_external_aad(envelope_schema_id, inner_type_id);
    sign_detached(signing_key, payload, &aad)
}

/// Concrete Ed25519 verification wrapper. Reconstructs `external_aad` from
/// the given schema/inner IDs and verifies the parsed COSE_Sign1's detached
/// signature against `pubkey` + `payload`.
pub fn verify_ed25519_detached(
    parsed: &ParsedCoseSign1,
    pubkey: &ed25519_dalek::VerifyingKey,
    envelope_schema_id: u64,
    inner_type_id: u64,
    payload: &[u8],
) -> Result<()> {
    // Confirm the parsed alg matches EdDSA before doing the signature check.
    let expected = iana::Algorithm::EdDSA.to_i64();
    if parsed.alg != expected {
        bail!(
            "COSE_Sign1 alg mismatch: parsed={} expected EdDSA={}",
            parsed.alg,
            expected
        );
    }
    let aad = build_external_aad(envelope_schema_id, inner_type_id);
    parsed
        .sign1
        .verify_detached_signature(payload, &aad, |sig, data| {
            <ed25519_dalek::VerifyingKey as CoseVerifier>::verify(pubkey, data, sig)
        })
        .context("COSE_Sign1 Ed25519 detached verification failed")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    const SCHEMA_ID: u64 = 0xcf5e_016c_5d4a_af92;
    const INNER_TYPE_ID: u64 = 0xdead_beef_cafe_babe;

    fn fresh_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    #[test]
    fn external_aad_is_deterministic_cbor_array() {
        let a = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);
        let b = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);
        assert_eq!(a, b, "external_aad must be deterministic");
        // CBOR array(2) header is 0x82
        assert_eq!(a[0], 0x82, "must encode as CBOR array of length 2");
    }

    #[test]
    fn external_aad_distinguishes_inner_type() {
        let a = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);
        let b = build_external_aad(SCHEMA_ID, INNER_TYPE_ID ^ 1);
        assert_ne!(a, b);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let sk = fresh_key();
        let vk = sk.verifying_key();
        let payload = b"hello cose";
        let aad = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);

        let cose_bytes = sign_detached(&sk, payload, &aad).expect("sign");
        let (kid, alg) = verify_detached(&cose_bytes, &vk, payload, &aad).expect("verify");

        assert_eq!(alg, "EdDSA");
        // kid is base64url(no-pad) of the raw 32-byte verifying key
        use base64::Engine;
        let expected_kid = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(vk.to_bytes());
        assert_eq!(kid, expected_kid);
    }

    #[test]
    fn tampered_payload_rejected() {
        let sk = fresh_key();
        let vk = sk.verifying_key();
        let payload = b"original payload";
        let aad = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);

        let cose_bytes = sign_detached(&sk, payload, &aad).expect("sign");
        let tampered = b"tampered payload";
        let err = verify_detached(&cose_bytes, &vk, tampered, &aad);
        assert!(err.is_err(), "tampered payload must fail verification");
    }

    #[test]
    fn tampered_external_aad_rejected() {
        let sk = fresh_key();
        let vk = sk.verifying_key();
        let payload = b"payload";
        let aad = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);
        let cose_bytes = sign_detached(&sk, payload, &aad).expect("sign");

        let wrong_aad = build_external_aad(SCHEMA_ID, INNER_TYPE_ID ^ 1);
        let err = verify_detached(&cose_bytes, &vk, payload, &wrong_aad);
        assert!(err.is_err(), "schema-confusion (wrong aad) must fail");
    }

    #[test]
    fn wrong_key_rejected() {
        let sk = fresh_key();
        let other = fresh_key().verifying_key();
        let payload = b"payload";
        let aad = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);

        let cose_bytes = sign_detached(&sk, payload, &aad).expect("sign");
        let err = verify_detached(&cose_bytes, &other, payload, &aad);
        assert!(err.is_err(), "wrong verifying key must fail");
    }

    #[test]
    fn peek_kid_alg_works() {
        let sk = fresh_key();
        let aad = build_external_aad(SCHEMA_ID, INNER_TYPE_ID);
        let cose_bytes = sign_detached(&sk, b"x", &aad).expect("sign");
        let (kid, alg_int) = peek_kid_alg(&cose_bytes).expect("peek");
        assert_eq!(kid, sk.verifying_key().to_bytes().to_vec());
        assert_eq!(alg_int, iana::Algorithm::EdDSA.to_i64());
    }
}
