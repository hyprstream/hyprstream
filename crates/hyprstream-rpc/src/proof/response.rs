//! Response binding types — the client's commitment to how the response must
//! be produced.
//!
//! In a request proof the map is the client's commitment. It is `null` when
//! the response is neither encrypted nor streamed. In a response proof it is
//! the realized binding and MUST equal the request's map where both are
//! present.
//!
//! Gate-2 amendments 3 + 4 make the two dimensions orthogonal and the map
//! four-field: `{1: root_type_id, 2: response_kind, 3: protection_mode,
//! 4: kem_recipient-or-null}`, with `response_kind = {1: unary, 2: stream_setup}`
//! and `protection_mode = {1: cleartext, 2: encrypted}`. The KEM recipient is
//! non-null iff `protection_mode == encrypted`, independent of `response_kind`.

use anyhow::{bail, Result};
use ciborium::value::Value as CborValue;

use super::ALG_ML_KEM_768;

/// How the response is delivered (Gate-2 amendment 4, axis 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseKind {
    /// A single unary response.
    Unary,
    /// A stream is set up.
    StreamSetup,
}

impl ResponseKind {
    fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(Self::Unary),
            2 => Ok(Self::StreamSetup),
            _ => bail!("response_kind: invalid value {v}"),
        }
    }
}

/// Whether the response is encrypted (Gate-2 amendment 4, axis 2). Orthogonal
/// to [`ResponseKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionMode {
    /// Response bytes are cleartext.
    Cleartext,
    /// Response bytes are encrypted to the carried KEM recipient.
    Encrypted,
}

impl ProtectionMode {
    fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(Self::Cleartext),
            2 => Ok(Self::Encrypted),
            _ => bail!("protection_mode: invalid value {v}"),
        }
    }
}

/// ML-KEM-768 recipient material for an encrypted response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KemRecipient {
    /// COSE KEM algorithm identifier — exactly the interim private-use
    /// `hs-kem-ml-kem-768-v1` value ([`ALG_ML_KEM_768`], −70200).
    pub alg: i64,
    /// ML-KEM-768 encapsulation key (1184 bytes).
    pub encapsulation_key: Vec<u8>,
    /// Key ID of the recipient key (1..64 bytes).
    pub kid: Vec<u8>,
}

/// Response binding map carried in the `response_binding` claim (−70004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseBinding {
    /// Cap'n Proto 64-bit type ID of the response root type (map key 1).
    pub root_type_id: u64,
    /// How the response is delivered (map key 2).
    pub response_kind: ResponseKind,
    /// Whether the response is encrypted (map key 3).
    pub protection_mode: ProtectionMode,
    /// KEM recipient material (map key 4), non-null iff
    /// `protection_mode == Encrypted`.
    pub kem_recipient: Option<KemRecipient>,
}

impl ResponseBinding {
    /// Decode from a CBOR value: either `null` (no binding) or the response
    /// binding map.
    pub fn decode(v: &CborValue) -> Result<Option<Self>> {
        match v {
            CborValue::Null => Ok(None),
            _ => Ok(Some(Self::decode_map(v)?)),
        }
    }

    fn decode_map(v: &CborValue) -> Result<Self> {
        let map = match v {
            CborValue::Map(m) => m,
            _ => bail!("response_binding: expected map or null"),
        };

        let mut root_type_id = None;
        let mut response_kind = None;
        let mut protection_mode = None;
        let mut kem_recipient = None;

        for (key, val) in map.iter() {
            let ik = match key {
                CborValue::Integer(i) => i128::from(*i),
                _ => bail!("response_binding: non-integer key"),
            };
            match ik {
                1 => {
                    root_type_id = Some(decode_uint(val, "root_type_id")?);
                }
                2 => {
                    let k = decode_uint(val, "response_kind")?;
                    response_kind = Some(ResponseKind::from_u64(k)?);
                }
                3 => {
                    let p = decode_uint(val, "protection_mode")?;
                    protection_mode = Some(ProtectionMode::from_u64(p)?);
                }
                4 => {
                    kem_recipient = match val {
                        CborValue::Null => None,
                        _ => Some(KemRecipient::decode(val)?),
                    };
                }
                _ => bail!("response_binding: unknown key {ik}"),
            }
        }

        let root_type_id = root_type_id
            .ok_or_else(|| anyhow::anyhow!("response_binding: missing root_type_id"))?;
        let response_kind = response_kind
            .ok_or_else(|| anyhow::anyhow!("response_binding: missing response_kind"))?;
        let protection_mode = protection_mode
            .ok_or_else(|| anyhow::anyhow!("response_binding: missing protection_mode"))?;

        // The KEM recipient is present iff the response is encrypted — the two
        // axes are orthogonal, so this relation depends only on protection_mode.
        match protection_mode {
            ProtectionMode::Encrypted => {
                if kem_recipient.is_none() {
                    bail!("response_binding: encrypted protection_mode requires kem_recipient");
                }
            }
            ProtectionMode::Cleartext => {
                if kem_recipient.is_some() {
                    bail!(
                        "response_binding: cleartext protection_mode must not carry kem_recipient"
                    );
                }
            }
        }

        Ok(Self {
            root_type_id,
            response_kind,
            protection_mode,
            kem_recipient,
        })
    }
}

impl KemRecipient {
    fn decode(v: &CborValue) -> Result<Self> {
        let map = match v {
            CborValue::Map(m) => m,
            _ => bail!("kem_recipient: expected map"),
        };
        let mut alg = None;
        let mut encapsulation_key = None;
        let mut kid = None;

        for (key, val) in map.iter() {
            let ik = match key {
                CborValue::Integer(i) => i128::from(*i),
                _ => bail!("kem_recipient: non-integer key"),
            };
            match ik {
                1 => {
                    alg = Some(decode_int(val, "kem alg")?);
                }
                2 => {
                    let k = decode_bstr(val, "encapsulation_key")?;
                    if k.len() != 1184 {
                        bail!(
                            "kem_recipient: encapsulation_key must be 1184 bytes, got {}",
                            k.len()
                        );
                    }
                    encapsulation_key = Some(k);
                }
                3 => {
                    let k = decode_bstr(val, "kem kid")?;
                    if k.is_empty() || k.len() > 64 {
                        bail!("kem_recipient: kid must be 1..64 bytes");
                    }
                    kid = Some(k);
                }
                _ => bail!("kem_recipient: unknown key {ik}"),
            }
        }

        let alg = alg.ok_or_else(|| anyhow::anyhow!("kem_recipient: missing alg"))?;
        // Gate-2 amendment 5: the algorithm is exactly the project-private
        // ML-KEM-768 value. Any other value denies — a registered COSE value is
        // an explicit incompatible profile revision, not accepted here.
        if alg != ALG_ML_KEM_768 {
            bail!(
                "kem_recipient: alg must be exactly {ALG_ML_KEM_768} (hs-kem-ml-kem-768-v1), got {alg}"
            );
        }
        let encapsulation_key = encapsulation_key
            .ok_or_else(|| anyhow::anyhow!("kem_recipient: missing encapsulation_key"))?;
        let kid = kid.ok_or_else(|| anyhow::anyhow!("kem_recipient: missing kid"))?;

        Ok(Self {
            alg,
            encapsulation_key,
            kid,
        })
    }
}

fn decode_uint(v: &CborValue, name: &str) -> Result<u64> {
    match v {
        CborValue::Integer(i) if i128::from(*i) >= 0 => Ok(i128::from(*i) as u64),
        _ => bail!("{name}: must be non-negative uint"),
    }
}

fn decode_int(v: &CborValue, name: &str) -> Result<i64> {
    match v {
        CborValue::Integer(i) => Ok(i128::from(*i) as i64),
        _ => bail!("{name}: must be integer"),
    }
}

fn decode_bstr(v: &CborValue, name: &str) -> Result<Vec<u8>> {
    match v {
        CborValue::Bytes(b) => Ok(b.clone()),
        _ => bail!("{name}: must be bstr"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn encapsulation_key() -> Vec<u8> {
        vec![0x11u8; 1184]
    }

    fn kem_map(alg: i64) -> CborValue {
        CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(alg.into())),
            (
                CborValue::Integer(2.into()),
                CborValue::Bytes(encapsulation_key()),
            ),
            (
                CborValue::Integer(3.into()),
                CborValue::Bytes(b"kem-1".to_vec()),
            ),
        ])
    }

    /// The amended four-field binding with orthogonal axes decodes, and the KEM
    /// recipient sits at key 4 with the exact −70200 algorithm.
    #[test]
    fn four_field_encrypted_unary_binding_decodes() {
        let v = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(42.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(1.into())), // unary
            (CborValue::Integer(3.into()), CborValue::Integer(2.into())), // encrypted
            (CborValue::Integer(4.into()), kem_map(ALG_ML_KEM_768)),
        ]);
        let b = ResponseBinding::decode(&v).unwrap().unwrap();
        assert_eq!(b.root_type_id, 42);
        assert_eq!(b.response_kind, ResponseKind::Unary);
        assert_eq!(b.protection_mode, ProtectionMode::Encrypted);
        assert_eq!(b.kem_recipient.as_ref().unwrap().alg, ALG_ML_KEM_768);
    }

    /// The two axes are orthogonal: an encrypted stream setup is a valid
    /// combination and still carries a recipient.
    #[test]
    fn encrypted_stream_setup_is_a_valid_orthogonal_combination() {
        let v = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(7.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(2.into())), // stream_setup
            (CborValue::Integer(3.into()), CborValue::Integer(2.into())), // encrypted
            (CborValue::Integer(4.into()), kem_map(ALG_ML_KEM_768)),
        ]);
        let b = ResponseBinding::decode(&v).unwrap().unwrap();
        assert_eq!(b.response_kind, ResponseKind::StreamSetup);
        assert_eq!(b.protection_mode, ProtectionMode::Encrypted);
        assert!(b.kem_recipient.is_some());
    }

    /// Cleartext binding carries no recipient at key 4.
    #[test]
    fn cleartext_unary_binding_forbids_recipient() {
        let ok = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(9.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(1.into())),
            (CborValue::Integer(3.into()), CborValue::Integer(1.into())),
            (CborValue::Integer(4.into()), CborValue::Null),
        ]);
        assert!(ResponseBinding::decode(&ok).unwrap().is_some());

        let bad = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(9.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(1.into())),
            (CborValue::Integer(3.into()), CborValue::Integer(1.into())), // cleartext
            (CborValue::Integer(4.into()), kem_map(ALG_ML_KEM_768)),
        ]);
        assert!(ResponseBinding::decode(&bad).is_err());
    }

    /// Encrypted binding without a recipient denies.
    #[test]
    fn encrypted_binding_requires_recipient() {
        let v = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(9.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(1.into())),
            (CborValue::Integer(3.into()), CborValue::Integer(2.into())), // encrypted
            (CborValue::Integer(4.into()), CborValue::Null),
        ]);
        assert!(ResponseBinding::decode(&v).is_err());
    }

    /// A KEM recipient with any algorithm other than −70200 denies.
    #[test]
    fn kem_recipient_alg_must_be_exactly_ml_kem_768() {
        let v = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(9.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(1.into())),
            (CborValue::Integer(3.into()), CborValue::Integer(2.into())),
            (CborValue::Integer(4.into()), kem_map(-70199)),
        ]);
        assert!(ResponseBinding::decode(&v).is_err());
    }

    /// The pre-amendment three-field encoding (combined mode at key 2, KEM at
    /// key 3, no protection_mode) is rejected by the four-field parser. This is
    /// the exact shape frozen vector P-4 still carries until WS-A re-issues it.
    #[test]
    fn pre_amendment_three_field_binding_is_rejected() {
        let v = CborValue::Map(vec![
            (CborValue::Integer(1.into()), CborValue::Integer(9.into())),
            (CborValue::Integer(2.into()), CborValue::Integer(2.into())), // old combined "unary encrypted"
            (CborValue::Integer(3.into()), kem_map(ALG_ML_KEM_768)), // old kem at key 3
        ]);
        assert!(ResponseBinding::decode(&v).is_err());
    }
}
