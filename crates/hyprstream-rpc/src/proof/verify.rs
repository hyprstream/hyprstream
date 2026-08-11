//! COSE signature verification for proof CWTs.
//!
//! After the structural parser validates headers, claims, and plan topology,
//! this module reconstructs the COSE `Sig_structure` and verifies each
//! component signature against the resolved key material:
//!
//! - **Unattributed proofs**: keys from the self-asserted `COSE_KeySet` in
//!   the body protected header (`hs_unattributed_key_set`).
//! - **Authenticated proofs**: keys from the credential `cnf`-resolved
//!   signer-suite record (provided by the caller).
//!
//! Uses `coset`'s built-in `CoseSign1::verify_detached_signature` and
//! `CoseSign::verify_detached_signature` for Sig_structure construction.

use anyhow::{anyhow, bail, Result};
use coset::CborSerializable;

use ed25519_dalek::Verifier;

use super::{
    parser::{CoseStructure, ParsedProof},
    ProofDisposition,
    ALG_ED25519,
};

/// Verify all component signatures on a parsed proof.
///
/// For unattributed proofs, the public keys are extracted from the proof's
/// body-protected `hs_unattributed_key_set`. For authenticated proofs, the
/// caller must provide the credential `cnf`-bound Ed25519 verifying key.
pub fn verify_proof_signatures(proof: &ParsedProof, cnf_ed25519_key: Option<&ed25519_dalek::VerifyingKey>) -> Result<()> {
    match proof.disposition {
        ProofDisposition::Unattributed => verify_unattributed(proof),
        ProofDisposition::Authenticated => {
            let key = cnf_ed25519_key.ok_or_else(|| {
                anyhow!("authenticated proof requires cnf Ed25519 verifying key")
            })?;
            verify_authenticated(proof, key)
        }
    }
}

/// Verify an unattributed proof's signatures against the self-asserted
/// COSE_KeySet in the body protected header.
fn verify_unattributed(proof: &ParsedProof) -> Result<()> {
    // Re-decode the COSE object to get coset types.
    let cbor_bytes = proof_to_cose_bytes(proof)?;

    // Extract Ed25519 public keys from the key set.
    let keys = extract_ed25519_keys_from_protected(&proof.protected_bytes)?;

    if keys.is_empty() {
        bail!("unattributed proof: no Ed25519 keys in key set");
    }

    match proof.structure {
        CoseStructure::Sign1 => {
            let sign1 = coset::CoseSign1::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign1 decode: {e}"))?;
            // External AAD is zero-length per the frozen profile.
            // The payload is embedded in the COSE object, so use
            // verify_signature (not verify_detached_signature).
            sign1
                .verify_signature(&[], |sig, data| {
                    verify_ed25519(sig, data, &keys[0].1)
                })
                .map_err(|e| anyhow!("Ed25519 signature verification failed: {e}"))
        }
        CoseStructure::Sign => {
            let sign = coset::CoseSign::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign decode: {e}"))?;
            // Verify each signature entry using the alg from our own parser.
            for (i, parsed_sig) in proof.signatures.iter().enumerate() {
                if parsed_sig.alg == ALG_ED25519 {
                    let key = find_key_by_kid(&keys, &parsed_sig.kid)
                        .ok_or_else(|| anyhow!("signature {i}: no matching Ed25519 key for kid"))?;
                    sign
                        .verify_signature(i, &[], |s, d| {
                            verify_ed25519(s, d, key)
                        })
                        .map_err(|e| anyhow!("signature {i} Ed25519 verification: {e}"))?;
                }
            }
            Ok(())
        }
    }
}

/// Verify an authenticated proof's Ed25519 component against the
/// credential cnf-bound key.
fn verify_authenticated(proof: &ParsedProof, cnf_key: &ed25519_dalek::VerifyingKey) -> Result<()> {
    let cbor_bytes = proof_to_cose_bytes(proof)?;

    match proof.structure {
        CoseStructure::Sign1 => {
            let sign1 = coset::CoseSign1::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign1 decode: {e}"))?;
            sign1
                .verify_signature(&[], |sig, data| {
                    verify_ed25519(sig, data, cnf_key)
                })
                .map_err(|e| anyhow!("cnf Ed25519 verification failed: {e}"))
        }
        CoseStructure::Sign => {
            let sign = coset::CoseSign::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign decode: {e}"))?;
            for (i, parsed_sig) in proof.signatures.iter().enumerate() {
                if parsed_sig.alg == ALG_ED25519 {
                    sign
                        .verify_signature(i, &[], |s, d| {
                            verify_ed25519(s, d, cnf_key)
                        })
                        .map_err(|e| anyhow!("cnf Ed25519 verification (sig {i}): {e}"))?;
                }
            }
            Ok(())
        }
    }
}

/// Reconstruct the raw COSE bytes from the parsed proof fields.
/// The ParsedProof stores the decoded components; we need to re-encode
/// the COSE object to verify with coset.
fn proof_to_cose_bytes(proof: &ParsedProof) -> Result<Vec<u8>> {
    // The proof was originally parsed from raw bytes. For verification, we
    // need to reconstruct or retain those bytes. Since we stored
    // protected_bytes and payload_bytes, we can reconstruct the COSE array.
    use ciborium::value::Value as CborValue;

    let unprotected = CborValue::Map(vec![]);
    let protected_bstr = CborValue::Bytes(proof.protected_bytes.clone());
    let payload_bstr = CborValue::Bytes(proof.payload_bytes.clone());

    let cose_value = match proof.structure {
        CoseStructure::Sign1 => {
            let sig_bstr = CborValue::Bytes(proof.signatures[0].signature.clone());
            CborValue::Array(vec![
                protected_bstr,
                unprotected,
                payload_bstr,
                sig_bstr,
            ])
        }
        CoseStructure::Sign => {
            let mut sig_entries = Vec::new();
            for sig in &proof.signatures {
                let sig_protected_bstr = CborValue::Bytes(sig.protected_bytes.clone());
                sig_entries.push(CborValue::Array(vec![
                    sig_protected_bstr,
                    CborValue::Map(vec![]),
                    CborValue::Bytes(sig.signature.clone()),
                ]));
            }
            CborValue::Array(vec![
                protected_bstr,
                unprotected,
                payload_bstr,
                CborValue::Array(sig_entries),
            ])
        }
    };

    let mut buf = Vec::new();
    ciborium::ser::into_writer(&cose_value, &mut buf)
        .map_err(|e| anyhow!("COSE re-encode: {e}"))?;
    Ok(buf)
}

/// Extract Ed25519 public keys from the body-protected header's
/// `hs_unattributed_key_set`.
fn extract_ed25519_keys_from_protected(protected_bytes: &[u8]) -> Result<Vec<(Vec<u8>, ed25519_dalek::VerifyingKey)>> {
    let protected: ciborium::Value =
        ciborium::de::from_reader(&mut std::io::Cursor::new(protected_bytes))
            .map_err(|e| anyhow!("protected header decode: {e}"))?;

    let map = match &protected {
        ciborium::Value::Map(m) => m,
        _ => bail!("protected header not a map"),
    };

    // Find hs_unattributed_key_set (-70103)
    let key_set = map
        .iter()
        .find(|(k, _)| {
            matches!(k,
                ciborium::Value::Integer(i)
                if i128::from(*i) == super::HEADER_HS_UNATTRIBUTED_KEY_SET as i128
            )
        })
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow!("no hs_unattributed_key_set in protected header"))?;

    let key_arr = match key_set {
        ciborium::Value::Array(a) => a,
        _ => bail!("key set not an array"),
    };

    let mut keys = Vec::new();
    for key_val in key_arr {
        let key_map = match key_val {
            ciborium::Value::Map(m) => m,
            _ => continue,
        };

        // Check kty == 1 (OKP)
        let kty = key_map.iter().find(|(k, _)| {
            matches!(k, ciborium::Value::Integer(i) if i128::from(*i) == 1)
        }).and_then(|(_, v)| match v {
            ciborium::Value::Integer(i) => Some(i128::from(*i)),
            _ => None,
        });

        if kty != Some(1) {
            continue; // Skip non-OKP keys (ML-DSA-65 handled separately)
        }

        // Extract kid (label 2)
        let kid = key_map.iter().find(|(k, _)| {
            matches!(k, ciborium::Value::Integer(i) if i128::from(*i) == 2)
        }).and_then(|(_, v)| match v {
            ciborium::Value::Bytes(b) => Some(b.clone()),
            _ => None,
        }).unwrap_or_default();

        // Extract x (label -2) — the 32-byte Ed25519 public key
        let x = key_map.iter().find(|(k, _)| {
            matches!(k, ciborium::Value::Integer(i) if i128::from(*i) == -2)
        }).and_then(|(_, v)| match v {
            ciborium::Value::Bytes(b) => Some(b.clone()),
            _ => None,
        }).ok_or_else(|| anyhow!("OKP key missing x coordinate"))?;

        if x.len() != 32 {
            bail!("Ed25519 x coordinate must be 32 bytes, got {}", x.len());
        }

        let mut pk_bytes = [0u8; 32];
        pk_bytes.copy_from_slice(&x);
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes)
            .map_err(|e| anyhow!("invalid Ed25519 key: {e}"))?;

        keys.push((kid, verifying_key));
    }

    Ok(keys)
}

/// Find a key by kid from the list of (kid, key) pairs.
fn find_key_by_kid<'a>(
    keys: &'a [(Vec<u8>, ed25519_dalek::VerifyingKey)],
    kid: &[u8],
) -> Option<&'a ed25519_dalek::VerifyingKey> {
    keys.iter().find(|(k, _)| k == kid).map(|(_, vk)| vk)
}

/// Verify an Ed25519 signature.
fn verify_ed25519(
    sig: &[u8],
    data: &[u8],
    key: &ed25519_dalek::VerifyingKey,
) -> Result<(), &'static str> {
    if sig.len() != 64 {
        return Err("signature must be 64 bytes");
    }
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(sig);
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    key.verify(data, &sig).map_err(|_| "Ed25519 verification failed")
}

#[cfg(test)]
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use super::super::parser::ParsedProof;

    #[test]
    fn test_p1_unattributed_signature_verifies() {
        let vectors = super::super::tests::load_positive_vectors();
        let p1 = vectors.iter().find(|(id, _)| id == "P-1").unwrap();
        let proof = ParsedProof::parse(&p1.1).expect("P-1 should parse");
        assert_eq!(proof.disposition, super::super::ProofDisposition::Unattributed);
        let result = verify_unattributed(&proof);
        assert!(result.is_ok(), "P-1 Ed25519 signature should verify: {:?}", result.err());
    }
}
