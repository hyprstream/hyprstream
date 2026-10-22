//! The settlement attestation: a canonical, self-contained, content-addressed
//! record (PAY-ARCH-01 guard-rail).
//!
//! ## Why this exists
//!
//! `settlement.capnp` declares `attestation :Data`. An opaque byte blob is fine
//! for "carry this to the verifier and back", but it is *not* fine for what the
//! attestation is actually for: a future exchange record (#926) must be able to
//! embed this blob **verbatim** and content-address it. That requires three
//! properties an arbitrary `Data` payload does not have:
//!
//! 1. **Canonical.** One logical attestation has exactly one byte
//!    representation, so its CID is a function of its meaning rather than of
//!    whichever encoder produced it. [`SettlementAttestation::decode`] enforces
//!    this by re-encoding what it parsed and rejecting any input that is not
//!    already in canonical form — a non-canonical blob is refused rather than
//!    silently normalized, because normalizing would change the CID a peer
//!    already committed to.
//! 2. **Self-contained.** Everything needed to decide what was settled — issuer,
//!    settlement id, unit, destination, amount, grant — is inside the signed
//!    body. Nothing is passed alongside it. A verifier that has to be told the
//!    settlement id out of band cannot detect a blob presented under the wrong
//!    id, so the id is bound in here instead.
//! 3. **Domain-separated.** The signed bytes begin with a version tag unique to
//!    this record type, so a signature over some other artifact can never be
//!    replayed as a settlement attestation, and a future v2 body can never be
//!    read as a v1 one.
//!
//! ## Normative encoding
//!
//! The body is canonical DAG-CBOR — the same discipline
//! `hyprstream_ledger::AccountId::derive` and `hyprstream-pds` use (definite
//! lengths, minimal-width integers, no maps so no key ordering to get wrong).
//! A single definite-length array, in this fixed order:
//!
//! ```text
//! [
//!   "hs-pay-settlement-attestation-v1",  // domain + version tag
//!   <issuer_did      : tstr>,
//!   <settlement_id   : tstr>,
//!   <unit_issuer_did : tstr>,
//!   <unit_class      : tstr>,
//!   <destination_did : tstr>,
//!   <amount_minor    : bstr(16)>,        // big-endian u128
//!   <grant_cid       : bstr>,            // empty when absent
//!   <settled_at      : uint>,            // unix seconds
//!   <signature       : bstr>             // COSE composite over the body below
//! ]
//! ```
//!
//! The **signing input** is the same array with the trailing `signature`
//! element omitted (a 9-element array), so the signature covers every semantic
//! field and nothing else. Amounts are 16-byte big-endian rather than CBOR
//! integers because the ledger's amounts are `u128`, which CBOR's integer range
//! cannot carry losslessly.
//!
//! ## CID
//!
//! [`SettlementAttestation::cid`] is `blake3(canonical bytes)`, 32 bytes. It is
//! stable across encode/decode round-trips and across processes, so an exchange
//! record can reference the attestation by CID and a peer can re-derive that CID
//! from the embedded bytes with no re-encoding step.

use ciborium::value::Value;
use serde::{Deserialize, Serialize};

use crate::{PayError, UnitRef};

/// Domain-separation and version tag. Changing the body shape means minting a
/// new tag, never reinterpreting this one.
pub const ATTESTATION_V1_TAG: &str = "hs-pay-settlement-attestation-v1";

/// A settlement attestation: the signed, self-contained claim that a specific
/// settlement committed, and therefore that a specific issuance is owed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettlementAttestation {
    /// DID of the party that signed this attestation (the settlement
    /// authority). Verification resolves key material from this, so it is part
    /// of the signed body rather than supplied alongside it.
    pub issuer_did: String,
    /// The settlement row this attests to. Bound into the signed body so an
    /// attestation cannot be presented under a different id.
    pub settlement_id: String,
    /// The credit unit to be issued.
    pub unit: UnitRef,
    /// The pseudonymous destination identity for the issued credits.
    pub destination_did: String,
    /// Amount in minor units.
    pub amount_minor: u128,
    /// The allocation grant this settlement backs, if any.
    pub grant_cid: Option<Vec<u8>>,
    /// When the settlement committed (unix seconds).
    pub settled_at: u64,
    /// COSE composite signature (EdDSA + ML-DSA-65) over [`Self::signing_input`].
    pub signature: Vec<u8>,
}

impl SettlementAttestation {
    /// The exact bytes a signature must cover: the canonical body **without**
    /// the signature element.
    pub fn signing_input(&self) -> Result<Vec<u8>, PayError> {
        encode_canonical(&self.body_fields())
    }

    /// The canonical wire bytes, signature included. This is what travels in
    /// the capnp `attestation :Data` field and what a future exchange record
    /// embeds verbatim.
    pub fn encode(&self) -> Result<Vec<u8>, PayError> {
        let mut fields = self.body_fields();
        fields.push(Value::Bytes(self.signature.clone()));
        encode_canonical(&fields)
    }

    /// Parse canonical wire bytes.
    ///
    /// Rejects any input that is not already canonical. Accepting a
    /// non-canonical encoding and normalizing it would give the same logical
    /// attestation two different CIDs, which defeats content-addressing it.
    pub fn decode(bytes: &[u8]) -> Result<Self, PayError> {
        let value: Value = ciborium::from_reader(bytes)
            .map_err(|e| PayError::AttestationInvalid(format!("attestation is not CBOR: {e}")))?;
        let items = match value {
            Value::Array(items) => items,
            _ => {
                return Err(PayError::AttestationInvalid(
                    "attestation must be a CBOR array".to_owned(),
                ))
            }
        };
        if items.len() != 10 {
            return Err(PayError::AttestationInvalid(format!(
                "attestation must have 10 fields, found {}",
                items.len()
            )));
        }
        if text(&items[0])? != ATTESTATION_V1_TAG {
            return Err(PayError::AttestationInvalid(
                "attestation is not a hs-pay-settlement-attestation-v1 record".to_owned(),
            ));
        }

        let parsed = SettlementAttestation {
            issuer_did: text(&items[1])?.to_owned(),
            settlement_id: text(&items[2])?.to_owned(),
            unit: UnitRef {
                issuer_did: text(&items[3])?.to_owned(),
                resource_class: text(&items[4])?.to_owned(),
            },
            destination_did: text(&items[5])?.to_owned(),
            amount_minor: amount(&items[6])?,
            grant_cid: {
                let g = bytes_of(&items[7])?;
                if g.is_empty() {
                    None
                } else {
                    Some(g.to_vec())
                }
            },
            settled_at: uint(&items[8])?,
            signature: bytes_of(&items[9])?.to_vec(),
        };

        // Canonicality: what we parsed must re-encode to exactly what we were
        // given, byte for byte.
        let reencoded = parsed.encode()?;
        if reencoded != bytes {
            return Err(PayError::AttestationInvalid(
                "attestation bytes are not canonical (re-encoding differs); refusing to \
                 normalize, because that would change the content address"
                    .to_owned(),
            ));
        }
        Ok(parsed)
    }

    /// Content address: `blake3` over the canonical bytes.
    ///
    /// Stable across processes and round-trips, so an exchange record can carry
    /// the attestation and its CID together and any peer can re-derive one from
    /// the other.
    pub fn cid(&self) -> Result<[u8; 32], PayError> {
        Ok(*blake3::hash(&self.encode()?).as_bytes())
    }

    /// The signed fields, in normative order, without the signature.
    fn body_fields(&self) -> Vec<Value> {
        vec![
            Value::Text(ATTESTATION_V1_TAG.to_owned()),
            Value::Text(self.issuer_did.clone()),
            Value::Text(self.settlement_id.clone()),
            Value::Text(self.unit.issuer_did.clone()),
            Value::Text(self.unit.resource_class.clone()),
            Value::Text(self.destination_did.clone()),
            Value::Bytes(self.amount_minor.to_be_bytes().to_vec()),
            Value::Bytes(self.grant_cid.clone().unwrap_or_default()),
            Value::Integer(self.settled_at.into()),
        ]
    }
}

fn encode_canonical(fields: &[Value]) -> Result<Vec<u8>, PayError> {
    let mut buf = Vec::new();
    ciborium::into_writer(&Value::Array(fields.to_vec()), &mut buf)
        .map_err(|e| PayError::Internal(format!("attestation encode failed: {e}")))?;
    Ok(buf)
}

fn text(v: &Value) -> Result<&str, PayError> {
    v.as_text()
        .ok_or_else(|| PayError::AttestationInvalid("expected a text field".to_owned()))
}

fn bytes_of(v: &Value) -> Result<&[u8], PayError> {
    v.as_bytes()
        .map(Vec::as_slice)
        .ok_or_else(|| PayError::AttestationInvalid("expected a byte-string field".to_owned()))
}

fn uint(v: &Value) -> Result<u64, PayError> {
    let i = v
        .as_integer()
        .ok_or_else(|| PayError::AttestationInvalid("expected an integer field".to_owned()))?;
    u64::try_from(i)
        .map_err(|_| PayError::AttestationInvalid("integer field out of range".to_owned()))
}

/// Decode a 16-byte big-endian amount, requiring exactly 16 bytes so a short or
/// long encoding cannot silently denote a different value.
fn amount(v: &Value) -> Result<u128, PayError> {
    let b = bytes_of(v)?;
    if b.len() != 16 {
        return Err(PayError::AttestationInvalid(format!(
            "amount must be exactly 16 bytes, found {}",
            b.len()
        )));
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(b);
    Ok(u128::from_be_bytes(arr))
}

#[cfg(test)]
mod tests {
    // A test asserting a known-good value legitimately unwraps.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn sample() -> SettlementAttestation {
        SettlementAttestation {
            issuer_did: "did:web:settlement.test".to_owned(),
            settlement_id: "stl_01HQ".to_owned(),
            unit: UnitRef {
                issuer_did: "did:web:issuer.test".to_owned(),
                resource_class: "gpu.h100.seconds".to_owned(),
            },
            destination_did: "did:key:zPurchaser".to_owned(),
            amount_minor: 12_345_678_901_234_567_890_123_456_789,
            grant_cid: Some(vec![0xde, 0xad, 0xbe, 0xef]),
            settled_at: 1_753_600_000,
            signature: vec![7u8; 96],
        }
    }

    #[test]
    fn encode_is_deterministic() {
        assert_eq!(sample().encode().unwrap(), sample().encode().unwrap());
    }

    #[test]
    fn roundtrips_through_the_wire_form() {
        let a = sample();
        let bytes = a.encode().unwrap();
        assert_eq!(SettlementAttestation::decode(&bytes).unwrap(), a);
    }

    #[test]
    fn a_large_u128_amount_survives_exactly() {
        let a = sample();
        let decoded = SettlementAttestation::decode(&a.encode().unwrap()).unwrap();
        assert_eq!(
            decoded.amount_minor, a.amount_minor,
            "a u128 amount beyond the CBOR integer range must not be truncated"
        );
    }

    #[test]
    fn the_cid_is_stable_across_a_roundtrip() {
        let a = sample();
        let bytes = a.encode().unwrap();
        let decoded = SettlementAttestation::decode(&bytes).unwrap();
        assert_eq!(
            a.cid().unwrap(),
            decoded.cid().unwrap(),
            "an embedded attestation must content-address identically after a roundtrip"
        );
        // ...and re-encoding must be byte-identical, so an exchange record can
        // embed the bytes verbatim with no re-encoding step.
        assert_eq!(decoded.encode().unwrap(), bytes);
    }

    #[test]
    fn the_cid_changes_when_any_signed_field_changes() {
        let base = sample().cid().unwrap();
        let mut other = sample();
        other.amount_minor += 1;
        assert_ne!(base, other.cid().unwrap());
        let mut other = sample();
        other.destination_did = "did:key:zSomeoneElse".to_owned();
        assert_ne!(base, other.cid().unwrap());
        let mut other = sample();
        other.settlement_id = "stl_other".to_owned();
        assert_ne!(base, other.cid().unwrap());
    }

    #[test]
    fn the_signing_input_excludes_the_signature() {
        let mut a = sample();
        let input = a.signing_input().unwrap();
        a.signature = vec![9u8; 96];
        assert_eq!(
            input,
            a.signing_input().unwrap(),
            "changing the signature must not change what was signed"
        );
        assert_ne!(
            input,
            a.encode().unwrap(),
            "the signing input must not be the full record"
        );
    }

    #[test]
    fn the_signing_input_covers_every_semantic_field() {
        let base = sample().signing_input().unwrap();
        for mutate in [
            (|a: &mut SettlementAttestation| a.issuer_did.push('x'))
                as fn(&mut SettlementAttestation),
            |a| a.settlement_id.push('x'),
            |a| a.unit.issuer_did.push('x'),
            |a| a.unit.resource_class.push('x'),
            |a| a.destination_did.push('x'),
            |a| a.amount_minor += 1,
            |a| a.settled_at += 1,
            |a| a.grant_cid = Some(vec![1, 2, 3]),
        ] {
            let mut a = sample();
            mutate(&mut a);
            assert_ne!(
                base,
                a.signing_input().unwrap(),
                "a semantic field is not covered by the signature"
            );
        }
    }

    #[test]
    fn a_non_canonical_encoding_is_refused() {
        // An indefinite-length array carries the same logical content but
        // different bytes. Accepting it would give one attestation two CIDs.
        let mut noncanonical = vec![0x9fu8]; // indefinite-length array header
        let canonical = sample().encode().unwrap();
        // Skip the definite-length array header (0x8a = array(10)).
        noncanonical.extend_from_slice(&canonical[1..]);
        noncanonical.push(0xff); // break

        let err = SettlementAttestation::decode(&noncanonical).unwrap_err();
        assert!(
            matches!(err, PayError::AttestationInvalid(_)),
            "a non-canonical encoding must be refused, got {err:?}"
        );
    }

    #[test]
    fn a_record_with_the_wrong_domain_tag_is_refused() {
        let mut fields = sample().body_fields();
        fields[0] = Value::Text("hs-pay-something-else-v1".to_owned());
        fields.push(Value::Bytes(vec![7u8; 96]));
        let bytes = encode_canonical(&fields).unwrap();
        assert!(SettlementAttestation::decode(&bytes).is_err());
    }

    #[test]
    fn a_truncated_amount_is_refused() {
        let mut fields = sample().body_fields();
        fields[6] = Value::Bytes(vec![0u8; 8]);
        fields.push(Value::Bytes(vec![7u8; 96]));
        let bytes = encode_canonical(&fields).unwrap();
        let err = SettlementAttestation::decode(&bytes).unwrap_err();
        assert!(matches!(err, PayError::AttestationInvalid(_)));
    }
}
