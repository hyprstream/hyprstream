//! Holder-proof binding for the two untrusted-rendezvous KEM recipients.
//!
//! This narrowly scoped source-slice extension is carried in CWT claim
//! `-70009`; it is required only for deferred Federate admission. Other proof
//! producers may omit it and retain the generic proof-v1 behavior. The two
//! hybrid recipients are committed as SHA-256 of canonical
//! `RecipientPublic::encode()` bytes; legacy `clientDhPublic` is bound as its
//! exact 32-byte value. Absence is represented explicitly as CBOR null so a
//! relay cannot add, remove, or replace any forwarded key.

use anyhow::{Result, bail};
use ciborium::value::Value as CborValue;
use sha2::{Digest as _, Sha256};

use crate::crypto::hybrid_kem::RecipientPublic;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FederateRecipientBinding {
    pub response: Option<[u8; 32]>,
    pub stream: Option<[u8; 32]>,
    pub client_dh_public: Option<[u8; 32]>,
}

impl FederateRecipientBinding {
    pub(crate) fn decode(value: &CborValue) -> Result<Self> {
        let CborValue::Map(fields) = value else {
            bail!("federate recipient binding: expected map")
        };
        if fields.len() != 3 {
            bail!("federate recipient binding: expected exactly three fields")
        }
        let mut response = None;
        let mut stream = None;
        let mut client_dh_public = None;
        let mut saw_response = false;
        let mut saw_stream = false;
        let mut saw_client_dh_public = false;
        for (key, value) in fields {
            match key {
                CborValue::Integer(key) if i128::from(*key) == 1 => {
                    if saw_response {
                        bail!("federate recipient binding: duplicate response key")
                    }
                    saw_response = true;
                    response = decode_commitment(value, "response")?;
                }
                CborValue::Integer(key) if i128::from(*key) == 2 => {
                    if saw_stream {
                        bail!("federate recipient binding: duplicate stream key")
                    }
                    saw_stream = true;
                    stream = decode_commitment(value, "stream")?;
                }
                CborValue::Integer(key) if i128::from(*key) == 3 => {
                    if saw_client_dh_public {
                        bail!("federate recipient binding: duplicate clientDhPublic key")
                    }
                    saw_client_dh_public = true;
                    client_dh_public = decode_public_key(value, "clientDhPublic")?;
                }
                _ => bail!("federate recipient binding: unknown/non-integer key"),
            }
        }
        if !saw_response || !saw_stream || !saw_client_dh_public {
            bail!("federate recipient binding: missing recipient field")
        }
        Ok(Self {
            response,
            stream,
            client_dh_public,
        })
    }

    pub fn from_recipients(
        response: Option<&RecipientPublic>,
        stream: Option<&RecipientPublic>,
        client_dh_public: Option<[u8; 32]>,
    ) -> Result<Self> {
        Ok(Self {
            response: response.map(commitment).transpose()?,
            stream: stream.map(commitment).transpose()?,
            client_dh_public,
        })
    }

    pub fn matches(
        &self,
        response: Option<&RecipientPublic>,
        stream: Option<&RecipientPublic>,
        client_dh_public: Option<[u8; 32]>,
    ) -> bool {
        Self::from_recipients(response, stream, client_dh_public)
            .is_ok_and(|binding| self == &binding)
    }
}

fn decode_public_key(value: &CborValue, name: &str) -> Result<Option<[u8; 32]>> {
    match value {
        CborValue::Null => Ok(None),
        CborValue::Bytes(bytes) if bytes.len() == 32 => {
            let mut public_key = [0; 32];
            public_key.copy_from_slice(bytes);
            Ok(Some(public_key))
        }
        CborValue::Bytes(_) => bail!("federate recipient binding: {name} must be 32 bytes"),
        _ => bail!("federate recipient binding: {name} must be bstr or null"),
    }
}

fn decode_commitment(value: &CborValue, name: &str) -> Result<Option<[u8; 32]>> {
    match value {
        CborValue::Null => Ok(None),
        CborValue::Bytes(bytes) if bytes.len() == 32 => {
            let mut commitment = [0; 32];
            commitment.copy_from_slice(bytes);
            Ok(Some(commitment))
        }
        CborValue::Bytes(_) => {
            bail!("federate recipient binding: {name} commitment must be 32 bytes")
        }
        _ => bail!("federate recipient binding: {name} commitment must be bstr or null"),
    }
}

fn commitment(recipient: &RecipientPublic) -> Result<[u8; 32]> {
    recipient.validate()?;
    Ok(Sha256::digest(recipient.encode()).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hybrid_kem::{SuiteId, generate_recipient};

    #[test]
    fn recipient_binding_rejects_omitted_and_malformed_fields() {
        let key = |value: i64| CborValue::Integer(value.into());
        let omitted_client_dh = CborValue::Map(vec![
            (key(1), CborValue::Null),
            (key(2), CborValue::Null),
        ]);
        assert!(FederateRecipientBinding::decode(&omitted_client_dh).is_err());

        let malformed_client_dh = CborValue::Map(vec![
            (key(1), CborValue::Null),
            (key(2), CborValue::Null),
            (key(3), CborValue::Bytes(vec![0x33; 31])),
        ]);
        assert!(FederateRecipientBinding::decode(&malformed_client_dh).is_err());
    }

    #[test]
    fn recipient_binding_matches_only_the_forwarded_recipient_set() -> Result<()> {
        let response = generate_recipient(SuiteId::HyKemX25519MlKem768)?.public();
        let stream = generate_recipient(SuiteId::HyKemX25519MlKem768)?.public();
        let replacement = generate_recipient(SuiteId::HyKemX25519MlKem768)?.public();
        let client_dh = [0x33; 32];
        let binding = FederateRecipientBinding::from_recipients(
            Some(&response),
            Some(&stream),
            Some(client_dh),
        )?;

        assert!(binding.matches(Some(&response), Some(&stream), Some(client_dh)));
        assert!(!binding.matches(Some(&replacement), Some(&stream), Some(client_dh)));
        assert!(!binding.matches(Some(&response), Some(&replacement), Some(client_dh)));
        assert!(!binding.matches(None, Some(&stream), Some(client_dh)));
        assert!(!binding.matches(Some(&response), None, Some(client_dh)));
        assert!(!binding.matches(Some(&response), Some(&stream), None));
        assert!(!binding.matches(Some(&response), Some(&stream), Some([0x44; 32])));
        Ok(())
    }

    #[test]
    fn recipient_binding_decodes_only_a_closed_three_field_map() -> Result<()> {
        let key = |value: i64| CborValue::Integer(value.into());
        let valid = CborValue::Map(vec![
            (key(1), CborValue::Bytes(vec![0x11; 32])),
            (key(2), CborValue::Null),
            (key(3), CborValue::Bytes(vec![0x33; 32])),
        ]);
        assert_eq!(
            FederateRecipientBinding::decode(&valid)?,
            FederateRecipientBinding {
                response: Some([0x11; 32]),
                stream: None,
                client_dh_public: Some([0x33; 32]),
            }
        );

        let duplicate = CborValue::Map(vec![
            (key(1), CborValue::Null),
            (key(1), CborValue::Null),
            (key(2), CborValue::Null),
            (key(3), CborValue::Null),
        ]);
        assert!(FederateRecipientBinding::decode(&duplicate).is_err());

        let unknown = CborValue::Map(vec![
            (key(1), CborValue::Null),
            (key(2), CborValue::Null),
            (key(4), CborValue::Null),
        ]);
        assert!(FederateRecipientBinding::decode(&unknown).is_err());

        let wrong_type = CborValue::Map(vec![
            (key(1), CborValue::Text("not bytes".into())),
            (key(2), CborValue::Null),
            (key(3), CborValue::Null),
        ]);
        assert!(FederateRecipientBinding::decode(&wrong_type).is_err());
        Ok(())
    }
}
