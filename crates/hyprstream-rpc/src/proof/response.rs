//! Response binding types — the client's commitment to how the response must
//! be produced.
//!
//! In a request proof the map is the client's commitment. It is `null` when
//! the response is neither encrypted nor streamed. In a response proof it is
//! the realized binding and MUST equal the request's map where both are
//! present.

use anyhow::{bail, Result};
use ciborium::value::Value as CborValue;

/// Response mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMode {
    /// Unary cleartext response.
    UnaryCleartext,
    /// Unary encrypted response.
    UnaryEncrypted,
    /// Stream setup.
    StreamSetup,
}

impl ResponseMode {
    fn from_u64(v: u64) -> Result<Self> {
        match v {
            1 => Ok(Self::UnaryCleartext),
            2 => Ok(Self::UnaryEncrypted),
            3 => Ok(Self::StreamSetup),
            _ => bail!("response_mode: invalid value {v}"),
        }
    }
}

/// ML-KEM-768 recipient material for an encrypted response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KemRecipient {
    /// COSE KEM algorithm identifier (interim private-use −70200).
    pub alg: i64,
    /// ML-KEM-768 encapsulation key (1184 bytes).
    pub encapsulation_key: Vec<u8>,
    /// Key ID of the recipient key (1..64 bytes).
    pub kid: Vec<u8>,
}

/// Response binding map carried in the `response_binding` claim (−70004).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseBinding {
    /// Cap'n Proto 64-bit type ID of the response root type.
    pub response_schema_id: u64,
    /// How the response is produced.
    pub mode: ResponseMode,
    /// KEM recipient material (non-null iff mode is encrypted).
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

        let mut response_schema_id = None;
        let mut mode = None;
        let mut kem_recipient = None;

        for (key, val) in map.iter() {
            let ik = match key {
                CborValue::Integer(i) => i128::from(*i),
                _ => bail!("response_binding: non-integer key"),
            };
            match ik {
                1 => {
                    response_schema_id = Some(decode_uint(val, "response_schema_id")?);
                }
                2 => {
                    let m = decode_uint(val, "response_mode")?;
                    mode = Some(ResponseMode::from_u64(m)?);
                }
                3 => {
                    kem_recipient = Some(KemRecipient::decode(val)?);
                }
                _ => bail!("response_binding: unknown key {ik}"),
            }
        }

        let response_schema_id = response_schema_id
            .ok_or_else(|| anyhow::anyhow!("response_binding: missing response_schema_id"))?;
        let mode = mode.ok_or_else(|| anyhow::anyhow!("response_binding: missing response_mode"))?;

        // Encrypted mode MUST carry KEM recipient; others MUST NOT.
        match mode {
            ResponseMode::UnaryEncrypted => {
                if kem_recipient.is_none() {
                    bail!("response_binding: encrypted mode requires kem_recipient");
                }
            }
            ResponseMode::UnaryCleartext | ResponseMode::StreamSetup => {
                if kem_recipient.is_some() {
                    bail!("response_binding: non-encrypted mode must not carry kem_recipient");
                }
            }
        }

        Ok(Self {
            response_schema_id,
            mode,
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
                        bail!("kem_recipient: encapsulation_key must be 1184 bytes, got {}", k.len());
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
        let encapsulation_key =
            encapsulation_key.ok_or_else(|| anyhow::anyhow!("kem_recipient: missing encapsulation_key"))?;
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
