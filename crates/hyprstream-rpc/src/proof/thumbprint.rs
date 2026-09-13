//! Frozen CDDL §7.1 replay namespace derivation. The preimages contain only
//! arrays, text, byte strings and unsigned integers, encoded deterministically.

use anyhow::{bail, Result};
use ciborium::Value;
use sha2::{Digest, Sha256};

use super::{
    plan::SignaturePlan, ALG_ED25519, HEADER_HS_SIGNATURE_PLAN, HEADER_HS_UNATTRIBUTED_KEY_SET,
    MAX_COSE_OBJECT_BYTES,
};

const AUTHENTICATED_DOMAIN: &str = "hs-rpc-replay-primary-suite-v1";
const KEY_SET_DOMAIN: &str = "hs-rpc-replay-key-set-v1";

type GroupContent = (String, Vec<Vec<u8>>);

// Infallible encoding of the restricted preimage grammar avoids changing the
// enrolled-record API into a fallible serializer. All lengths are definite;
// the argument always uses its shortest encoding (RFC 8949 §4.2.1).
fn head(out: &mut Vec<u8>, major: u8, n: u64) {
    let tag = major << 5;
    match n {
        0..=23 => out.push(tag | n as u8),
        24..=0xff => out.extend_from_slice(&[tag | 24, n as u8]),
        0x100..=0xffff => {
            out.push(tag | 25);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
        0x10000..=0xffffffff => {
            out.push(tag | 26);
            out.extend_from_slice(&(n as u32).to_be_bytes());
        }
        _ => {
            out.push(tag | 27);
            out.extend_from_slice(&n.to_be_bytes());
        }
    }
}

fn text(out: &mut Vec<u8>, value: &str) {
    head(out, 3, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

fn keys(out: &mut Vec<u8>, public_keys: &[Vec<u8>]) {
    head(out, 4, public_keys.len() as u64);
    for key in public_keys {
        head(out, 2, key.len() as u64);
        out.extend_from_slice(key);
    }
}

pub(super) fn authenticated(suite: &str, public_keys: &[Vec<u8>], epoch: u64) -> [u8; 32] {
    Sha256::digest(authenticated_preimage(suite, public_keys, epoch)).into()
}

fn authenticated_preimage(suite: &str, public_keys: &[Vec<u8>], epoch: u64) -> Vec<u8> {
    let mut out = Vec::new();
    head(&mut out, 4, 4);
    text(&mut out, AUTHENTICATED_DOMAIN);
    text(&mut out, suite);
    keys(&mut out, public_keys);
    head(&mut out, 0, epoch);
    out
}

fn key_set_preimage(groups: &[GroupContent]) -> Vec<u8> {
    let mut records = groups
        .iter()
        .map(|(suite, public_keys)| {
            let mut record = Vec::new();
            head(&mut record, 4, 2);
            text(&mut record, suite);
            keys(&mut record, public_keys);
            record
        })
        .collect::<Vec<_>>();
    // Byte-sort canonical records, never group_id/kid/plan positions. Preserve
    // duplicate records: the namespace binds a multiset, not a deduplicated set.
    records.sort();
    let mut out = Vec::new();
    head(&mut out, 4, 2);
    text(&mut out, KEY_SET_DOMAIN);
    head(&mut out, 4, records.len() as u64);
    for record in records {
        out.extend_from_slice(&record);
    }
    out
}

fn key_set(groups: &[GroupContent]) -> [u8; 32] {
    Sha256::digest(key_set_preimage(groups)).into()
}

fn field(value: &Value, label: i64) -> Result<&Value> {
    let Value::Map(map) = value else {
        bail!("thumbprint: expected map");
    };
    map.iter()
        .find_map(|(k, v)| match k {
            Value::Integer(i) if i128::from(*i) == i128::from(label) => Some(v),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("thumbprint: missing label {label}"))
}

fn content_groups(plan: &SignaturePlan, key_set: &Value) -> Result<Vec<GroupContent>> {
    let Value::Array(entries) = key_set else {
        bail!("thumbprint: key set must be array");
    };
    plan.groups.iter().map(|group| {
        let public_keys = group.components.iter().map(|component| {
            let key = entries.iter().find(|key| {
                matches!(field(key, 2), Ok(Value::Bytes(kid)) if *kid == component.kid)
                    && matches!(field(key, 3), Ok(Value::Integer(alg)) if i128::from(*alg) == i128::from(component.alg))
            }).ok_or_else(|| anyhow::anyhow!("thumbprint: missing component key"))?;
            let label = if component.alg == ALG_ED25519 { -2 } else { -1 };
            match field(key, label)? {
                Value::Bytes(bytes) => Ok(bytes.clone()),
                _ => bail!("thumbprint: public key must be bytes"),
            }
        }).collect::<Result<Vec<_>>>()?;
        Ok((group.suite_id.clone(), public_keys))
    }).collect()
}

pub(super) fn unattributed(protected_bytes: &[u8]) -> Result<[u8; 32]> {
    if protected_bytes.len() > MAX_COSE_OBJECT_BYTES {
        bail!("thumbprint: protected header exceeds proof size cap");
    }
    // Public callers do not necessarily come through ParsedProof. Audit before
    // deserialization for bounded nesting, unique canonical map keys and no
    // trailing data, then apply the same plan/key-set acceptance rules.
    super::cbor_audit::audit_deterministic(protected_bytes)?;
    let protected: Value = ciborium::de::from_reader(protected_bytes)?;
    let plan = SignaturePlan::decode(field(&protected, HEADER_HS_SIGNATURE_PLAN)?)?;
    let key_set_value = field(&protected, HEADER_HS_UNATTRIBUTED_KEY_SET)?;
    super::parser::validate_unattributed_key_set(key_set_value, &plan)?;
    Ok(key_set(&content_groups(&plan, key_set_value)?))
}

#[cfg(test)]
mod tests {
    use super::super::{
        admission,
        enrollment::{ComponentKey, EnrolledComponent, SignerRole, SignerSuiteRecord},
        parser::ParsedProof,
        SUITE_CLASSICAL, SUITE_HYBRID,
    };
    use super::*;

    fn vectors() -> Result<serde_json::Value> {
        Ok(serde_json::from_str(include_str!(
            "../../../../docs/standards/v16/vectors/proof-v1-thumbprints.json"
        ))?)
    }
    fn bytes(v: &serde_json::Value, name: &str) -> Result<Vec<u8>> {
        Ok(hex::decode(v[name].as_str().ok_or_else(|| {
            anyhow::anyhow!("missing fixture {name}")
        })?)?)
    }
    fn encoded(v: &Value) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        ciborium::ser::into_writer(v, &mut out)?;
        Ok(out)
    }
    fn protected(wire: &[u8]) -> Result<Vec<u8>> {
        let Value::Array(object) = ciborium::de::from_reader(wire)? else {
            bail!("fixture object");
        };
        let Some(Value::Bytes(header)) = object.first() else {
            bail!("fixture header");
        };
        Ok(header.clone())
    }
    fn field_mut(value: &mut Value, label: i64) -> Result<&mut Value> {
        let Value::Map(map) = value else {
            bail!("fixture map");
        };
        map.iter_mut()
            .find_map(|(key, value)| match key {
                Value::Integer(i) if i128::from(*i) == i128::from(label) => Some(value),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("missing fixture label"))
    }
    fn array_mut(v: &mut Value) -> Result<&mut Vec<Value>> {
        match v {
            Value::Array(a) => Ok(a),
            _ => bail!("fixture array"),
        }
    }

    #[test]
    fn replay_thumbprint_authenticated_matches_frozen_vector() -> Result<()> {
        let fixture = vectors()?;
        let vector = &fixture["authenticated"];
        let public_keys = vector["component_public_keys_hex"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("fixture keys"))?
            .iter()
            .map(|key| {
                Ok(hex::decode(
                    key.as_str().ok_or_else(|| anyhow::anyhow!("fixture hex"))?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let ed: [u8; 32] = public_keys[0].as_slice().try_into()?;
        let record = SignerSuiteRecord {
            principal: "fixture-primary".into(),
            suite_id: SUITE_HYBRID.into(),
            components: vec![
                EnrolledComponent::new(
                    b"ed".to_vec(),
                    ComponentKey::Ed25519(ed25519_dalek::VerifyingKey::from_bytes(&ed)?),
                ),
                EnrolledComponent::new(
                    b"pq".to_vec(),
                    ComponentKey::MlDsa65(Box::new(crate::crypto::pq::ml_dsa_vk_from_bytes(
                        &public_keys[1],
                    )?)),
                ),
            ],
            epoch: 1,
            role: SignerRole::Primary,
            approver_role: None,
            enrollment_policy_id: "fixture".into(),
            not_after: u64::MAX,
            revoked: false,
        };
        assert_eq!(
            authenticated_preimage(&record.suite_id, &public_keys, record.epoch),
            bytes(vector, "preimage_hex")?
        );
        assert_eq!(
            record.replay_thumbprint().as_slice(),
            bytes(vector, "thumbprint_sha256")?
        );
        let mut renamed = record.clone();
        renamed.principal = "other-label".into();
        renamed.components[0].kid = b"other-kid".to_vec();
        assert_eq!(record.replay_thumbprint(), renamed.replay_thumbprint());
        renamed.epoch += 1;
        assert_ne!(record.replay_thumbprint(), renamed.replay_thumbprint());
        let mut reversed = record.clone();
        reversed.components.reverse();
        assert_ne!(record.replay_thumbprint(), reversed.replay_thumbprint());
        Ok(())
    }

    #[test]
    fn replay_thumbprint_public_helpers_match_vector_and_m1_labels() -> Result<()> {
        let fixture = vectors()?;
        let (_, wire) = super::super::tests::load_positive_vectors()
            .into_iter()
            .find(|(id, _)| id == "P-1")
            .ok_or_else(|| anyhow::anyhow!("P-1"))?;
        let original = ParsedProof::parse(&wire)?;
        let expected = bytes(&fixture["unattributed"], "thumbprint_sha256")?;
        assert_eq!(
            original
                .unattributed_replay_thumbprint()
                .ok_or_else(|| anyhow::anyhow!("thumbprint"))?
                .as_slice(),
            expected
        );
        assert_eq!(
            admission::unattributed_thumbprint(&original.protected_bytes)?.as_slice(),
            expected
        );
        // Structural parser/helper tests: relabel headers consistently. Signatures
        // are not re-signed or admitted; this only checks namespace derivation.
        for change_group in [true, false] {
            let mut object: Value = ciborium::de::from_reader(wire.as_slice())?;
            let mut header: Value = ciborium::de::from_reader(original.protected_bytes.as_slice())?;
            if change_group {
                let plan = array_mut(field_mut(&mut header, HEADER_HS_SIGNATURE_PLAN)?)?;
                *field_mut(&mut plan[0], 1)? = Value::Integer(77.into());
                *field_mut(&mut header, super::super::HEADER_HS_LOGICAL_SIGNER_GROUP)? =
                    Value::Integer(77.into());
            } else {
                let kid = Value::Bytes(b"renamed-key".to_vec());
                let plan = array_mut(field_mut(&mut header, HEADER_HS_SIGNATURE_PLAN)?)?;
                let components = array_mut(field_mut(&mut plan[0], 3)?)?;
                *field_mut(&mut components[0], 2)? = kid.clone();
                let entries = array_mut(field_mut(&mut header, HEADER_HS_UNATTRIBUTED_KEY_SET)?)?;
                *field_mut(&mut entries[0], 2)? = kid.clone();
                *field_mut(&mut header, 4)? = kid;
            }
            array_mut(&mut object)?[0] = Value::Bytes(encoded(&header)?);
            let parsed = ParsedProof::parse(&encoded(&object)?)?;
            assert_eq!(
                parsed.unattributed_replay_thumbprint(),
                original.unattributed_replay_thumbprint()
            );
            assert_eq!(
                admission::unattributed_thumbprint(&parsed.protected_bytes)?.as_slice(),
                expected
            );
        }
        let mut header: Value = ciborium::de::from_reader(original.protected_bytes.as_slice())?;
        let entries = array_mut(field_mut(&mut header, HEADER_HS_UNATTRIBUTED_KEY_SET)?)?;
        let Value::Bytes(key) = field_mut(&mut entries[0], -2)? else {
            bail!("fixture key");
        };
        key[0] ^= 1;
        assert_ne!(
            admission::unattributed_thumbprint(&encoded(&header)?)?.as_slice(),
            expected
        );
        assert!(admission::unattributed_thumbprint(&vec![0; MAX_COSE_OBJECT_BYTES + 1]).is_err());
        let mut trailing = original.protected_bytes.clone();
        trailing.push(0);
        assert!(admission::unattributed_thumbprint(&trailing).is_err());
        Ok(())
    }

    #[test]
    fn replay_thumbprint_r1_vector_order_and_multiset_distinctions() -> Result<()> {
        let fixture = vectors()?;
        let mut canonical = Vec::new();
        for name in ["order_ab", "order_ba"] {
            let vector = &fixture["group_order_canonicalization"][name];
            let wire = bytes(vector, "cbor_hex")?;
            let header_bytes = protected(&wire)?;
            let header: Value = ciborium::de::from_reader(header_bytes.as_slice())?;
            let plan = SignaturePlan::decode(field(&header, HEADER_HS_SIGNATURE_PLAN)?)?;
            let groups = content_groups(&plan, field(&header, HEADER_HS_UNATTRIBUTED_KEY_SET)?)?;
            assert_eq!(key_set_preimage(&groups), bytes(vector, "preimage_hex")?);
            assert_eq!(
                key_set(&groups).as_slice(),
                bytes(vector, "thumbprint_sha256")?
            );
            // R1's derivation is exercised without widening the existing
            // public one-group unattributed proof support for this repair.
            assert!(ParsedProof::parse(&wire).is_err());
            assert!(admission::unattributed_thumbprint(&header_bytes).is_err());
            canonical.push(key_set(&groups));
            assert_ne!(key_set(&groups), key_set(&groups[..1]));
            let mut duplicate = groups.clone();
            duplicate.push(groups[0].clone());
            assert_ne!(key_set(&groups), key_set(&duplicate));
            let mut changed = groups.clone();
            changed[0].1[0][0] ^= 1;
            assert_ne!(key_set(&groups), key_set(&changed));
        }
        assert_eq!(canonical[0], canonical[1]);
        let classical = vec![(SUITE_CLASSICAL.to_owned(), vec![vec![1; 32]])];
        let hybrid = vec![(SUITE_HYBRID.to_owned(), vec![vec![1; 32], vec![2; 1952]])];
        assert_ne!(key_set(&classical), key_set(&hybrid));
        let mut reversed = hybrid.clone();
        reversed[0].1.reverse();
        assert_ne!(key_set(&hybrid), key_set(&reversed));
        Ok(())
    }
}
