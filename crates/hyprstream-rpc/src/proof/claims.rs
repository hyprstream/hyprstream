//! Proof CWT claims-set types and bounded deterministic-CBOR decode.
//!
//! The claims set is a closed profile of RFC 8392 CWT: registered claims
//! (`aud`, `exp`, `iat`, `cti`, `Nonce`) carry registered semantics, and four
//! private-use claims (−70001…−70004) carry the Hyprstream-specific bindings.
//! The claim set is closed: an unknown claim key denies. Every listed claim is
//! always present; an inapplicable value is CBOR `null`.

use anyhow::{bail, Result};
use ciborium::value::Value as CborValue;

#[allow(unused_imports)]
use super::{
    CredentialHash, RequestId, CLAIM_CAPNP_BODY_BYTES, CLAIM_CAPNP_SCHEMA_ID,
    CLAIM_CREDENTIAL_HASH, CLAIM_RESPONSE_BINDING, CREDENTIAL_HASH_SIZE, CWT_CLAIM_AUD,
    CWT_CLAIM_CTI, CWT_CLAIM_EXP, CWT_CLAIM_IAT, CWT_CLAIM_NONCE, MAX_AUD_BYTES, MAX_BODY_BYTES,
    MAX_CHALLENGE_BYTES, MIN_CHALLENGE_BYTES, REQUEST_ID_SIZE,
};
use crate::proof::response::ResponseBinding;

/// Decoded proof-CWT claims set.
#[derive(Debug, Clone)]
pub struct ProofClaims {
    /// Canonical service domain (CWT `aud`).
    pub aud: String,
    /// Proof expiry, integer Unix seconds (CWT `exp`).
    pub exp: u64,
    /// Proof issuance, integer Unix seconds (CWT `iat`).
    pub iat: u64,
    /// 128-bit request identifier (CWT `cti`).
    pub request_id: RequestId,
    /// Server challenge (CWT `Nonce`, RFC 9711). Present exactly when the
    /// profile requires a challenge.
    pub nonce: Option<Vec<u8>>,
    /// SHA-256 of the exact credential bytes, or `None` when no credential is
    /// presented (encoded as CBOR `null`).
    pub credential_hash: Option<CredentialHash>,
    /// Cap'n Proto 64-bit root type ID of the signed body.
    pub capnp_schema_id: u64,
    /// Exact Cap'n Proto body bytes.
    pub capnp_request_bytes: Vec<u8>,
    /// Response binding map, or `None` when unbound (encoded as CBOR `null`).
    pub response_binding: Option<ResponseBinding>,
}

impl ProofClaims {
    /// Decode a claims payload from a CBOR byte string.
    ///
    /// Enforces:
    /// - RFC 8949 core deterministic encoding (no indefinite lengths, no
    ///   non-minimal integers, no tags, no floating-point values, sorted unique
    ///   map keys).
    /// - Closed claim set: unknown claim keys deny.
    /// - Every listed claim is present (absent claims deny).
    /// - Proof-v1 size caps on `aud`, body bytes, challenge.
    pub fn decode(payload: &[u8]) -> Result<Self> {
        // Raw-byte deterministic audit BEFORE ciborium deserialization
        // (ciborium resolves indefinite lengths and non-minimal integers
        // transparently, destroying the evidence).
        super::cbor_audit::audit_deterministic(payload)?;

        let value: CborValue = ciborium::de::from_reader(&mut std::io::Cursor::new(payload))
            .map_err(|e| anyhow::anyhow!("proof claims: CBOR decode failed: {e}"))?;

        let map = match &value {
            CborValue::Map(m) => m,
            _ => bail!("proof claims: expected CBOR map"),
        };

        // Collect known keys; deny unknown keys.
        let mut aud = None;
        let mut exp = None;
        let mut iat = None;
        let mut cti = None;
        let mut nonce = None;
        let mut credential_hash = None;
        let mut capnp_schema_id = None;
        let mut capnp_request_bytes = None;
        let mut response_binding = None;

        for (key, val) in map.iter() {
            let ik = match key {
                CborValue::Integer(i) => i128::from(*i),
                _ => bail!("proof claims: non-integer claim key"),
            };
            match ik {
                x if x == CWT_CLAIM_AUD as i128 => {
                    aud = Some(decode_text(val, "aud", 1, MAX_AUD_BYTES)?);
                }
                x if x == CWT_CLAIM_EXP as i128 => {
                    exp = Some(decode_uint(val, "exp")?);
                }
                x if x == CWT_CLAIM_IAT as i128 => {
                    iat = Some(decode_uint(val, "iat")?);
                }
                x if x == CWT_CLAIM_CTI as i128 => {
                    cti = Some(decode_bstr_fixed(val, "cti", REQUEST_ID_SIZE)?);
                }
                x if x == CWT_CLAIM_NONCE as i128 => {
                    nonce = Some(decode_bstr_range(
                        val,
                        "Nonce",
                        MIN_CHALLENGE_BYTES,
                        MAX_CHALLENGE_BYTES,
                    )?);
                }
                x if x == CLAIM_CREDENTIAL_HASH as i128 => {
                    credential_hash = Some(decode_credential_hash(val)?);
                }
                x if x == CLAIM_CAPNP_SCHEMA_ID as i128 => {
                    capnp_schema_id = Some(decode_uint(val, "capnp_schema_id")?);
                }
                x if x == CLAIM_CAPNP_BODY_BYTES as i128 => {
                    let b = decode_bstr(val, "capnp_request_bytes")?;
                    if b.len() > MAX_BODY_BYTES {
                        bail!(
                            "proof claims: capnp_request_bytes exceeds {} bytes",
                            MAX_BODY_BYTES
                        );
                    }
                    capnp_request_bytes = Some(b);
                }
                x if x == CLAIM_RESPONSE_BINDING as i128 => {
                    response_binding = Some(ResponseBinding::decode(val)?);
                }
                _ => bail!("proof claims: unknown claim key {}", ik),
            }
        }

        // Every listed claim is always present (Nonce is the exception).
        let aud = aud.ok_or_else(|| anyhow::anyhow!("proof claims: missing aud"))?;
        let exp = exp.ok_or_else(|| anyhow::anyhow!("proof claims: missing exp"))?;
        let iat = iat.ok_or_else(|| anyhow::anyhow!("proof claims: missing iat"))?;
        let request_id = cti.ok_or_else(|| anyhow::anyhow!("proof claims: missing cti"))?;
        let credential_hash = credential_hash
            .ok_or_else(|| anyhow::anyhow!("proof claims: missing credential_hash"))?;
        let capnp_schema_id = capnp_schema_id
            .ok_or_else(|| anyhow::anyhow!("proof claims: missing capnp_schema_id"))?;
        let capnp_request_bytes = capnp_request_bytes
            .ok_or_else(|| anyhow::anyhow!("proof claims: missing capnp_request_bytes"))?;
        let response_binding = response_binding
            .ok_or_else(|| anyhow::anyhow!("proof claims: missing response_binding"))?;

        Ok(Self {
            aud,
            exp,
            iat,
            request_id,
            nonce,
            credential_hash,
            capnp_schema_id,
            capnp_request_bytes,
            response_binding,
        })
    }
}

/// Decode a `credential_hash` claim: either `null` (no credential) or a
/// 32-byte bstr.
fn decode_credential_hash(val: &CborValue) -> Result<Option<CredentialHash>> {
    match val {
        CborValue::Null => Ok(None),
        CborValue::Bytes(b) => {
            if b.len() != CREDENTIAL_HASH_SIZE {
                bail!(
                    "proof claims: credential_hash must be {} bytes, got {}",
                    CREDENTIAL_HASH_SIZE,
                    b.len()
                );
            }
            let mut arr = [0u8; CREDENTIAL_HASH_SIZE];
            arr.copy_from_slice(b);
            Ok(Some(arr))
        }
        _ => bail!("proof claims: credential_hash must be bstr or null"),
    }
}

// ---------------------------------------------------------------------------
// Typed CBOR extractors — deterministic encoding is enforced at the byte
// level by `cbor_audit::audit_deterministic` before ciborium deserialization.
// ---------------------------------------------------------------------------

fn decode_text(v: &CborValue, name: &str, min_bytes: usize, max_bytes: usize) -> Result<String> {
    match v {
        CborValue::Text(s) => {
            if s.len() < min_bytes {
                bail!("proof claims: {name} must be at least {min_bytes} bytes");
            }
            if s.len() > max_bytes {
                bail!("proof claims: {name} exceeds {max_bytes} bytes");
            }
            Ok(s.clone())
        }
        _ => bail!("proof claims: {name} must be text"),
    }
}

fn decode_uint(v: &CborValue, name: &str) -> Result<u64> {
    match v {
        CborValue::Integer(i) => {
            if i128::from(*i) < 0 {
                bail!("proof claims: {name} must be unsigned");
            }
            Ok(i128::from(*i) as u64)
        }
        _ => bail!("proof claims: {name} must be uint"),
    }
}

fn decode_bstr(v: &CborValue, name: &str) -> Result<Vec<u8>> {
    match v {
        CborValue::Bytes(b) => Ok(b.clone()),
        _ => bail!("proof claims: {name} must be bstr"),
    }
}

fn decode_bstr_fixed<const N: usize>(v: &CborValue, name: &str, size: usize) -> Result<[u8; N]> {
    let b = decode_bstr(v, name)?;
    // We need to verify b.len() == size, but N is the array size.
    // For our use, N == size always (REQUEST_ID_SIZE etc.).
    if b.len() != size {
        bail!("proof claims: {name} must be {size} bytes, got {}", b.len());
    }
    // N is the compile-time size; at runtime we checked against `size`.
    // Since N == size for all call sites, this is safe.
    let mut arr = [0u8; N];
    arr.copy_from_slice(&b[..N]);
    Ok(arr)
}

fn decode_bstr_range(v: &CborValue, name: &str, min: usize, max: usize) -> Result<Vec<u8>> {
    let b = decode_bstr(v, name)?;
    if b.len() < min || b.len() > max {
        bail!(
            "proof claims: {name} must be {min}..{max} bytes, got {}",
            b.len()
        );
    }
    Ok(b)
}
