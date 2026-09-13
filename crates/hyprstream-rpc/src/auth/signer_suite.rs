//! Signer-suite thumbprint — the ONE shared helper and encoding for the v16
//! `cnf.hs_signer_suite` confirmation method (frozen WS-A credential-profile
//! §1.1/§5). Both the mint side (WS-B `ExchangeDelegated`) and the verify side
//! (WS-C proof resolution) compute the identical function over the identical
//! bytes — there is no second hash encoding.
//!
//! Content thumbprint: `SHA-256(` RFC 8949 **deterministic-CBOR** of
//! `[suite_id, [ordered raw component public keys]]` `)`. It uniformly covers a
//! classical (one-key) and a hybrid (two-key) primary signer group. Component
//! key order is the suite-plan order — NEVER a kid/group_id/label/position.
//!
//! This is a direct port of A's frozen `gen_proof_vectors.py::signer_suite_thumbprint`
//! (+ its `enc`); the golden vectors in the tests are cross-checked against A's
//! own tooling at `af08825528627069b3cfbdd763948c4a24689cb5`.

use sha2::{Digest, Sha256};

use super::det_cbor::{det_cbor, DetCborValue};

/// Frozen suite ID for a classical (single Ed25519) primary signer group.
pub const SUITE_CLASSICAL_ED25519: &str = "hs-cose-sign-ed25519-v1";

/// Frozen suite ID for a hybrid (Ed25519 + ML-DSA-65) primary signer group.
/// Component order is `[ed25519, ml_dsa_65]`.
pub const SUITE_HYBRID_ED25519_MLDSA65: &str = "hs-cose-sign-ed25519-mldsa65-wns-v1";

/// The v16 `cnf.hs_signer_suite` content thumbprint: `SHA-256(det-CBOR([suite_id,
/// [ordered raw component public keys]]))`, using the ONE shared
/// [`det_cbor`] encoder (no second det-CBOR implementation).
///
/// `ordered_component_pubkeys` are the RAW public key bytes in suite-plan order
/// (Ed25519 = 32 bytes; ML-DSA-65 = 1952 bytes). The result is the 32-byte
/// digest; callers base64url-encode (no padding) it for the JWT `cnf` member.
#[must_use]
pub fn signer_suite_thumbprint(suite_id: &str, ordered_component_pubkeys: &[&[u8]]) -> [u8; 32] {
    let value = DetCborValue::Array(vec![
        DetCborValue::Text(suite_id),
        DetCborValue::Array(
            ordered_component_pubkeys
                .iter()
                .map(|pk| DetCborValue::Bytes(pk))
                .collect(),
        ),
    ]);
    Sha256::digest(det_cbor(&value)).into()
}

/// The v16 `cnf.hs_signer_suite` value (base64url, unpadded) for a primary
/// signer group given its raw component public keys: classical
/// (`SUITE_CLASSICAL_ED25519`, Ed25519 only) when `ml_dsa_65` is `None`, or
/// hybrid (`SUITE_HYBRID_ED25519_MLDSA65`, Ed25519 + ML-DSA-65 in that frozen
/// order) when present. The single place mint paths turn key material into the
/// stamped confirmation value.
#[must_use]
pub fn service_signer_suite_b64(ed25519: &[u8; 32], ml_dsa_65: Option<&[u8]>) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    let thumbprint = match ml_dsa_65 {
        Some(pq) => signer_suite_thumbprint(SUITE_HYBRID_ED25519_MLDSA65, &[ed25519, pq]),
        None => signer_suite_thumbprint(SUITE_CLASSICAL_ED25519, &[ed25519]),
    };
    URL_SAFE_NO_PAD.encode(thumbprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

    /// Golden vectors computed by A's own frozen `gen_proof_vectors.py`
    /// (`enc`/`signer_suite_thumbprint`) at `af0882552`, over the fixed inputs
    /// ed = 32×0x01 and mldsa = 1952×0x02. If these ever change, the mint (B)
    /// and verify (C) sides have diverged from the frozen A encoding.
    #[test]
    fn classical_thumbprint_matches_a_golden_vector() {
        let ed = [0x01u8; 32];
        let tp = signer_suite_thumbprint(SUITE_CLASSICAL_ED25519, &[&ed]);
        assert_eq!(
            URL_SAFE_NO_PAD.encode(tp),
            "VKnD78bkwV7dWjvbXrRwAYKWUmcIRjWFY7UwnEkOj_w"
        );
    }

    #[test]
    fn hybrid_thumbprint_matches_a_golden_vector() {
        let ed = [0x01u8; 32];
        let mldsa = [0x02u8; 1952];
        let tp = signer_suite_thumbprint(SUITE_HYBRID_ED25519_MLDSA65, &[&ed, &mldsa]);
        assert_eq!(
            URL_SAFE_NO_PAD.encode(tp),
            "n47qkIMqPn-EuRo0YmrWkGFyyCY0UmVhKE19Bxbtt0U"
        );
    }

    /// Component order is load-bearing: swapping the two hybrid keys yields a
    /// different thumbprint (the encoding never sorts or canonicalizes keys).
    #[test]
    fn component_order_is_significant() {
        let ed = [0x01u8; 32];
        let mldsa = [0x02u8; 1952];
        let ordered = signer_suite_thumbprint(SUITE_HYBRID_ED25519_MLDSA65, &[&ed, &mldsa]);
        let swapped = signer_suite_thumbprint(SUITE_HYBRID_ED25519_MLDSA65, &[&mldsa, &ed]);
        assert_ne!(ordered, swapped);
    }
}
