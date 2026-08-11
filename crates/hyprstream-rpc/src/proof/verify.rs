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

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use coset::CborSerializable;

use ed25519_dalek::Verifier;

use super::{
    enrollment::{EnrollmentResolver, SignerRole, SignerSuiteRecord},
    parser::{CoseStructure, ParsedProof},
    plan::SignerGroup,
    ProofDisposition, ProofKind, ALG_ED25519, ALG_ML_DSA_65,
};

/// What a successful verification establishes about the proof's signers.
#[derive(Debug, Clone)]
pub struct VerifiedProof {
    /// The replay namespace this proof is admitted under (§4.5): the
    /// credential-bound primary signer-suite thumbprint for authenticated
    /// proofs, the plan/key-set thumbprint for unattributed ones.
    pub replay_thumbprint: [u8; 32],
    /// The resolved primary principal, for authenticated proofs only. An
    /// unattributed key set proves only internal proof consistency and is
    /// never an identity.
    pub primary_principal: Option<String>,
}

/// Verify all component signatures on a parsed request proof.
///
/// For unattributed proofs, the public keys come from the proof's
/// body-protected `hs_unattributed_key_set` — self-asserted material that
/// proves only internal consistency. For authenticated proofs, every logical
/// signer group resolves through `resolver` to an enrolled signer-suite record
/// that pins its exact suite, ordered component keys, principal, and epoch;
/// a component verifying under any unpinned key denies.
///
/// A missing credential key or a missing resolver is `Rejected`; neither
/// downgrades to the unattributed branch.
pub fn verify_proof_signatures(
    proof: &ParsedProof,
    cnf_ed25519_key: Option<&ed25519_dalek::VerifyingKey>,
    resolver: Option<&dyn EnrollmentResolver>,
    now: u64,
) -> Result<VerifiedProof> {
    if proof.kind != ProofKind::Request {
        bail!("verify_proof_signatures: only request proofs; use verify_response_proof");
    }
    match proof.disposition {
        ProofDisposition::Unattributed => {
            verify_unattributed(proof)?;
            let thumbprint = proof.unattributed_replay_thumbprint().ok_or_else(|| {
                anyhow!("unattributed proof: cannot compute replay namespace thumbprint")
            })?;
            Ok(VerifiedProof {
                replay_thumbprint: thumbprint,
                primary_principal: None,
            })
        }
        ProofDisposition::Authenticated => {
            let cnf = cnf_ed25519_key.ok_or_else(|| {
                anyhow!("authenticated proof requires a resolved credential cnf key")
            })?;
            let resolver = resolver.ok_or_else(|| {
                anyhow!("authenticated proof requires an enrollment resolver; none installed")
            })?;
            verify_authenticated(proof, cnf, resolver, now)
        }
    }
}

/// Verify a response proof against the enrolled service identity for the
/// service domain the *request* was addressed to, and against the exact
/// request it answers.
///
/// A response signed by a different enrolled service's key — even a validly
/// enrolled one — fails: the caller compares against the one service it
/// addressed, never against "any enrolled service" (§9.4). A response proof
/// can never verify for another request ID.
pub fn verify_response_proof(
    proof: &ParsedProof,
    expected_service_domain: &str,
    expected_request_id: &super::RequestId,
    resolver: &dyn EnrollmentResolver,
    now: u64,
) -> Result<VerifiedProof> {
    if proof.kind != ProofKind::Response {
        bail!("verify_response_proof: proof is not a response proof");
    }
    // The response proof's cti echoes the request_id it answers (N-22).
    if &proof.claims.request_id != expected_request_id {
        bail!(
            "response proof cti {} does not match the request_id {} it must answer",
            hex::encode(proof.claims.request_id),
            hex::encode(expected_request_id)
        );
    }
    if proof.claims.aud != expected_service_domain {
        bail!(
            "response proof aud '{}' is not the addressed service domain '{}'",
            proof.claims.aud,
            expected_service_domain
        );
    }

    let record = resolver.resolve_service(expected_service_domain).ok_or_else(|| {
        anyhow!("no enrolled response signer for service domain '{expected_service_domain}'")
    })?;
    record.check_usable(now, proof.claims.exp, SignerRole::Service)?;

    let groups = &proof.plan.groups;
    if groups.len() != 1 {
        bail!(
            "response proof must carry exactly one signer group, got {}",
            groups.len()
        );
    }
    if !group_matches_record(&groups[0], &record) {
        bail!("response proof plan does not match the enrolled service signer record");
    }

    let mut resolved = HashMap::new();
    resolved.insert(groups[0].group_id, record.clone());
    verify_entries_against_records(proof, &resolved)?;

    Ok(VerifiedProof {
        replay_thumbprint: record.replay_thumbprint(),
        primary_principal: Some(record.principal),
    })
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
            let entry = proof
                .signatures
                .first()
                .ok_or_else(|| anyhow!("COSE_Sign1 has no signature entry"))?;
            if entry.alg != ALG_ED25519 {
                bail!("unattributed COSE_Sign1: alg {} not in profile", entry.alg);
            }
            // Match the key by its exact kid — never positionally.
            let key = find_key_by_kid(&keys, &entry.kid)
                .ok_or_else(|| anyhow!("unattributed proof: no key set entry for the signature kid"))?;
            let sign1 = coset::CoseSign1::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign1 decode: {e}"))?;
            sign1
                .verify_signature(&[], |sig, data| verify_ed25519(sig, data, key))
                .map_err(|e| anyhow!("Ed25519 signature verification failed: {e}"))
        }
        CoseStructure::Sign => {
            let sign = coset::CoseSign::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign decode: {e}"))?;
            // Every required component MUST verify. No component is skipped.
            for (i, parsed_sig) in proof.signatures.iter().enumerate() {
                match parsed_sig.alg {
                    ALG_ED25519 => {
                        let key = find_key_by_kid(&keys, &parsed_sig.kid)
                            .ok_or_else(|| anyhow!("sig {i}: no Ed25519 key for kid"))?;
                        sign
                            .verify_signature(i, &[], |s, d| {
                                verify_ed25519(s, d, key)
                            })
                            .map_err(|e| anyhow!("sig {i} Ed25519 verification: {e}"))?;
                    }
                    ALG_ML_DSA_65 => {
                        let pq_key = extract_mldsa65_key_from_protected(
                            &proof.protected_bytes, &parsed_sig.kid,
                        )?
                        .ok_or_else(|| anyhow!("sig {i}: no ML-DSA-65 key for kid"))?;
                        sign
                            .verify_signature(i, &[], |s, d| {
                                verify_mldsa65(s, d, &pq_key)
                            })
                            .map_err(|e| anyhow!("sig {i} ML-DSA-65 verification: {e}"))?;
                    }
                    other => bail!("unknown algorithm {other} in signature; denying"),
                }
            }
            Ok(())
        }
    }
}

/// Verify an authenticated proof's component signatures against per-entry
/// enrollment records.
///
/// Every logical signer group resolves independently:
///
/// - the **primary** group is the one whose declared suite and ordered
///   `(alg, kid)` components exactly match the record the credential `cnf`
///   key resolves to — exactly one group may match;
/// - every **additional** group resolves through the anchored approver
///   enrollment for its first component's key ID;
/// - resolved groups must name distinct principals and must not share a
///   pinned public key, so one key holder cannot satisfy two logical signers;
/// - every signature entry verifies under the exact key its own group's
///   record pins for its `(alg, kid)` — an unpinned key denies even when it
///   is validly enrolled to the same principal.
fn verify_authenticated(
    proof: &ParsedProof,
    cnf_key: &ed25519_dalek::VerifyingKey,
    resolver: &dyn EnrollmentResolver,
    now: u64,
) -> Result<VerifiedProof> {
    let primary = resolver
        .resolve_primary(cnf_key)
        .ok_or_else(|| anyhow!("no enrolled signer-suite record for the credential cnf key"))?;
    primary.check_usable(now, proof.claims.exp, SignerRole::Primary)?;
    if !primary.pins_ed25519(cnf_key) {
        bail!("resolved primary record does not pin the credential cnf key");
    }

    // Exactly one plan group may claim the credential-bound primary record.
    let primary_groups: Vec<&SignerGroup> = proof
        .plan
        .groups
        .iter()
        .filter(|g| group_matches_record(g, &primary))
        .collect();
    if primary_groups.len() != 1 {
        bail!(
            "expected exactly one plan group matching the cnf-resolved signer suite, found {}",
            primary_groups.len()
        );
    }
    let primary_group_id = primary_groups[0].group_id;

    let mut resolved: HashMap<u64, SignerSuiteRecord> = HashMap::new();
    resolved.insert(primary_group_id, primary.clone());

    for group in &proof.plan.groups {
        if group.group_id == primary_group_id {
            continue;
        }
        let anchor_kid = &group
            .components
            .first()
            .ok_or_else(|| anyhow!("plan group {} has no components", group.group_id))?
            .kid;
        let record = resolver.resolve_approver(anchor_kid).ok_or_else(|| {
            anyhow!(
                "no enrolled approver for group {} anchor kid {}",
                group.group_id,
                hex::encode(anchor_kid)
            )
        })?;
        record.check_usable(now, proof.claims.exp, SignerRole::Approver)?;
        if !group_matches_record(group, &record) {
            bail!(
                "plan group {} does not match its resolved approver enrollment",
                group.group_id
            );
        }
        resolved.insert(group.group_id, record);
    }

    check_distinct_signers(&resolved)?;
    verify_entries_against_records(proof, &resolved)?;

    Ok(VerifiedProof {
        replay_thumbprint: primary.replay_thumbprint(),
        primary_principal: Some(primary.principal),
    })
}

/// A plan group matches an enrollment record only if the declared suite and
/// the ordered `(alg, kid)` component list are exactly equal — no supersets,
/// no reordering, no principal-level equivalence.
fn group_matches_record(group: &SignerGroup, record: &SignerSuiteRecord) -> bool {
    group.suite_id == record.suite_id
        && group.components.len() == record.components.len()
        && group
            .components
            .iter()
            .zip(record.components.iter())
            .all(|(c, e)| c.alg == e.alg && c.kid == e.kid)
}

/// Distinct logical signer groups must resolve to distinct enrolled
/// principals, and no pinned public key may be counted in two groups.
fn check_distinct_signers(resolved: &HashMap<u64, SignerSuiteRecord>) -> Result<()> {
    let mut principals: HashMap<&str, u64> = HashMap::new();
    let mut keys: HashMap<Vec<u8>, u64> = HashMap::new();
    for (group_id, record) in resolved {
        if let Some(other) = principals.insert(record.principal.as_str(), *group_id) {
            bail!(
                "principal '{}' is counted in both group {} and group {}",
                record.principal,
                other,
                group_id
            );
        }
        for component in &record.components {
            if let Some(other) = keys.insert(component.key.encoded(), *group_id) {
                bail!(
                    "one pinned key is counted in both group {} and group {}",
                    other,
                    group_id
                );
            }
        }
    }
    Ok(())
}

/// Verify every signature entry under the exact key pinned by its own
/// group's record, over the COSE `Sig_structure` with a zero-length
/// `external_aad`.
fn verify_entries_against_records(
    proof: &ParsedProof,
    resolved: &HashMap<u64, SignerSuiteRecord>,
) -> Result<()> {
    let cbor_bytes = proof_to_cose_bytes(proof)?;

    match proof.structure {
        CoseStructure::Sign1 => {
            let entry = proof
                .signatures
                .first()
                .ok_or_else(|| anyhow!("COSE_Sign1 has no signature entry"))?;
            let record = resolved
                .get(&entry.group_id)
                .ok_or_else(|| anyhow!("no resolved record for group {}", entry.group_id))?;
            let component = record.component(entry.alg, &entry.kid).ok_or_else(|| {
                anyhow!(
                    "group {} pins no key for (alg={}, kid={})",
                    entry.group_id,
                    entry.alg,
                    hex::encode(&entry.kid)
                )
            })?;
            let sign1 = coset::CoseSign1::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign1 decode: {e}"))?;
            sign1
                .verify_signature(&[], |sig, data| component.key.verify(sig, data))
                .map_err(|e| anyhow!("enrolled key verification failed: {e}"))
        }
        CoseStructure::Sign => {
            let sign = coset::CoseSign::from_slice(&cbor_bytes)
                .map_err(|e| anyhow!("COSE_Sign decode: {e}"))?;
            for (i, entry) in proof.signatures.iter().enumerate() {
                let record = resolved
                    .get(&entry.group_id)
                    .ok_or_else(|| anyhow!("no resolved record for group {}", entry.group_id))?;
                let component = record.component(entry.alg, &entry.kid).ok_or_else(|| {
                    anyhow!(
                        "group {} pins no key for (alg={}, kid={})",
                        entry.group_id,
                        entry.alg,
                        hex::encode(&entry.kid)
                    )
                })?;
                sign.verify_signature(i, &[], |sig, data| component.key.verify(sig, data))
                    .map_err(|e| {
                        anyhow!(
                            "sig {i} (group {}, alg {}) verification failed: {e}",
                            entry.group_id,
                            entry.alg
                        )
                    })?;
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

/// Extract an ML-DSA-65 public key from the body-protected header's
/// COSE_KeySet by matching kid.
fn extract_mldsa65_key_from_protected(
    protected_bytes: &[u8],
    kid: &[u8],
) -> Result<Option<crate::crypto::pq::MlDsaVerifyingKey>> {
    let protected: ciborium::Value =
        ciborium::de::from_reader(&mut std::io::Cursor::new(protected_bytes))
            .map_err(|e| anyhow!("protected header decode: {e}"))?;

    let map = match &protected {
        ciborium::Value::Map(m) => m,
        _ => bail!("protected header not a map"),
    };

    let key_set = map
        .iter()
        .find(|(k, _)| {
            matches!(k,
                ciborium::Value::Integer(i)
                if i128::from(*i) == super::HEADER_HS_UNATTRIBUTED_KEY_SET as i128
            )
        })
        .map(|(_, v)| v);

    let key_set = match key_set {
        Some(v) => v,
        None => return Ok(None),
    };

    let key_arr = match key_set {
        ciborium::Value::Array(a) => a,
        _ => bail!("key set not an array"),
    };

    for key_val in key_arr {
        let key_map = match key_val {
            ciborium::Value::Map(m) => m,
            _ => continue,
        };

        // Check kty == 7 (AKP)
        let kty = key_map.iter().find(|(k, _)| {
            matches!(k, ciborium::Value::Integer(i) if i128::from(*i) == 7)
        }).and_then(|(_, v)| match v {
            ciborium::Value::Integer(i) => Some(i128::from(*i)),
            _ => None,
        });

        if kty != Some(7) {
            continue;
        }

        // Match kid
        let key_kid = key_map.iter().find(|(k, _)| {
            matches!(k, ciborium::Value::Integer(i) if i128::from(*i) == 2)
        }).and_then(|(_, v)| match v {
            ciborium::Value::Bytes(b) => Some(b.clone()),
            _ => None,
        }).unwrap_or_default();

        if key_kid != kid {
            continue;
        }

        // Extract pub (label -1) — 1952-byte ML-DSA-65 public key
        let pub_key = key_map.iter().find(|(k, _)| {
            matches!(k, ciborium::Value::Integer(i) if i128::from(*i) == -1)
        }).and_then(|(_, v)| match v {
            ciborium::Value::Bytes(b) => Some(b.clone()),
            _ => None,
        }).ok_or_else(|| anyhow!("AKP key missing pub field"))?;

        if pub_key.len() != 1952 {
            bail!("ML-DSA-65 pub must be 1952 bytes, got {}", pub_key.len());
        }

        let vk = crate::crypto::pq::ml_dsa_vk_from_bytes(&pub_key)?;
        return Ok(Some(vk));
    }

    Ok(None)
}

/// Verify an ML-DSA-65 signature.
fn verify_mldsa65(
    sig: &[u8],
    data: &[u8],
    key: &crate::crypto::pq::MlDsaVerifyingKey,
) -> Result<(), &'static str> {
    crate::crypto::pq::ml_dsa_verify(key, data, sig).map_err(|_| "ML-DSA-65 verification failed")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::parser::ParsedProof;
    use super::super::tests::{
        classical_enrollment, ed25519_public, hybrid_enrollment, FIXTURE_NOW,
        FIXTURE_REQUEST_ID,
    };
    use super::*;

    fn load_vector(id: &str) -> Vec<u8> {
        let vectors = super::super::tests::load_positive_vectors();
        vectors
            .iter()
            .find(|(v_id, _)| v_id == id)
            .unwrap_or_else(|| panic!("vector {id} must exist"))
            .1
            .clone()
    }

    fn parse(id: &str) -> ParsedProof {
        ParsedProof::parse(&load_vector(id)).unwrap_or_else(|e| panic!("{id} must parse: {e}"))
    }

    fn client_cnf() -> ed25519_dalek::VerifyingKey {
        ed25519_public("client-ed25519-1")
    }

    // -- unattributed ------------------------------------------------------

    #[test]
    fn p1_unattributed_signature_verifies() {
        let proof = parse("P-1");
        assert_eq!(proof.disposition, ProofDisposition::Unattributed);
        assert_eq!(proof.structure, CoseStructure::Sign1);
        let verified = verify_proof_signatures(&proof, None, None, FIXTURE_NOW)
            .expect("P-1 self-asserted Ed25519 signature must verify");
        assert!(
            verified.primary_principal.is_none(),
            "an unattributed key set is never an identity"
        );
        assert_eq!(
            verified.replay_thumbprint,
            proof.unattributed_replay_thumbprint().unwrap()
        );
    }

    #[test]
    fn p1_corrupted_signature_denies() {
        let mut proof = parse("P-1");
        proof.signatures[0].signature[0] ^= 0xFF;
        assert!(verify_proof_signatures(&proof, None, None, FIXTURE_NOW).is_err());
    }

    // -- authenticated: enrolled suites verify -----------------------------

    #[test]
    fn p2_hybrid_verifies_under_its_enrolled_suite() {
        let proof = parse("P-2");
        assert_eq!(proof.disposition, ProofDisposition::Authenticated);
        assert_eq!(proof.structure, CoseStructure::Sign);
        assert_eq!(proof.signatures.len(), 2);
        let resolver = hybrid_enrollment();
        let verified =
            verify_proof_signatures(&proof, Some(&client_cnf()), Some(&resolver), FIXTURE_NOW)
                .expect("P-2 must verify: both Ed25519 and ML-DSA-65 components");
        assert_eq!(verified.primary_principal.as_deref(), Some("client"));
    }

    #[test]
    fn p4_classical_sign1_verifies_under_its_enrolled_suite() {
        let proof = parse("P-4");
        let resolver = classical_enrollment();
        verify_proof_signatures(&proof, Some(&client_cnf()), Some(&resolver), FIXTURE_NOW)
            .expect("P-4 must verify under the enrolled classical suite");
    }

    #[test]
    fn p5_two_logical_groups_verify_against_distinct_enrollments() {
        let proof = parse("P-5");
        assert_eq!(proof.plan.groups.len(), 2);
        let resolver = classical_enrollment();
        let verified =
            verify_proof_signatures(&proof, Some(&client_cnf()), Some(&resolver), FIXTURE_NOW)
                .expect("P-5 must verify: primary group plus one anchored approver");
        assert_eq!(verified.primary_principal.as_deref(), Some("client"));
    }

    /// The replay namespace excludes approver groups: the same primary signer
    /// is admitted under one namespace whether or not approvers are present.
    #[test]
    fn approver_groups_do_not_change_the_replay_namespace() {
        let resolver = classical_enrollment();
        let p4 = verify_proof_signatures(
            &parse("P-4"),
            Some(&client_cnf()),
            Some(&resolver),
            FIXTURE_NOW,
        )
        .unwrap();
        let p5 = verify_proof_signatures(
            &parse("P-5"),
            Some(&client_cnf()),
            Some(&resolver),
            FIXTURE_NOW,
        )
        .unwrap();
        assert_eq!(p4.replay_thumbprint, p5.replay_thumbprint);
    }

    /// The unattributed namespace is disjoint from the authenticated one.
    #[test]
    fn unattributed_and_authenticated_namespaces_are_disjoint() {
        let resolver = classical_enrollment();
        let unattributed = verify_proof_signatures(&parse("P-1"), None, None, FIXTURE_NOW).unwrap();
        let authenticated = verify_proof_signatures(
            &parse("P-4"),
            Some(&client_cnf()),
            Some(&resolver),
            FIXTURE_NOW,
        )
        .unwrap();
        assert_ne!(
            unattributed.replay_thumbprint,
            authenticated.replay_thumbprint
        );
    }

    // -- authenticated: fail-closed ----------------------------------------

    /// A missing enrollment resolver, or a credential with no enrolled
    /// signer-suite record, is Rejected — never downgraded to the
    /// self-asserted branch.
    #[test]
    fn missing_resolver_or_enrollment_denies() {
        let proof = parse("P-4");
        assert!(
            verify_proof_signatures(&proof, Some(&client_cnf()), None, FIXTURE_NOW).is_err(),
            "no resolver installed must deny"
        );
        assert!(
            verify_proof_signatures(
                &proof,
                Some(&client_cnf()),
                Some(&super::super::enrollment::InMemoryEnrollmentResolver::new()),
                FIXTURE_NOW
            )
            .is_err(),
            "an empty enrollment must deny"
        );
    }

    /// A missing credential is Rejected for every authenticated vector.
    #[test]
    fn missing_credential_denies() {
        let resolver = classical_enrollment();
        for id in ["P-2", "P-4", "P-5"] {
            assert!(
                verify_proof_signatures(&parse(id), None, Some(&resolver), FIXTURE_NOW).is_err(),
                "{id} with no credential must deny"
            );
        }
    }

    /// A credential that resolves to no enrolled record denies, even though
    /// the proof's own signatures are internally valid.
    #[test]
    fn unenrolled_credential_denies() {
        let stranger = ed25519_dalek::SigningKey::from_bytes(&[3u8; 32]).verifying_key();
        let resolver = classical_enrollment();
        assert!(
            verify_proof_signatures(&parse("P-4"), Some(&stranger), Some(&resolver), FIXTURE_NOW)
                .is_err()
        );
    }

    /// Component-key separation: a key enrolled for the WNS hybrid suite is
    /// unusable in the standalone suite, and vice versa. Each vector verifies
    /// under exactly the deployment that enrolled its suite.
    #[test]
    fn cross_suite_enrollment_denies() {
        let hybrid = hybrid_enrollment();
        let classical = classical_enrollment();
        for id in ["P-4", "P-5"] {
            assert!(
                verify_proof_signatures(&parse(id), Some(&client_cnf()), Some(&hybrid), FIXTURE_NOW)
                    .is_err(),
                "{id} (standalone suite) must deny under a hybrid enrollment"
            );
        }
        assert!(
            verify_proof_signatures(
                &parse("P-2"),
                Some(&client_cnf()),
                Some(&classical),
                FIXTURE_NOW
            )
            .is_err(),
            "P-2 (hybrid suite) must deny under a standalone enrollment"
        );
    }

    /// A corrupted post-quantum component denies the whole proof: no
    /// component is optional, and a surviving classical signature cannot
    /// carry a hybrid group.
    #[test]
    fn corrupted_ml_dsa_component_denies() {
        let mut proof = parse("P-2");
        let idx = proof
            .signatures
            .iter()
            .position(|s| s.alg == ALG_ML_DSA_65)
            .expect("P-2 has an ML-DSA-65 component");
        proof.signatures[idx].signature[0] ^= 0xFF;
        let resolver = hybrid_enrollment();
        assert!(
            verify_proof_signatures(&proof, Some(&client_cnf()), Some(&resolver), FIXTURE_NOW)
                .is_err(),
            "a corrupted ML-DSA-65 component must deny"
        );
    }

    /// The classical component of a hybrid proof is equally mandatory.
    #[test]
    fn corrupted_ed25519_component_of_a_hybrid_proof_denies() {
        let mut proof = parse("P-2");
        let idx = proof
            .signatures
            .iter()
            .position(|s| s.alg == ALG_ED25519)
            .expect("P-2 has an Ed25519 component");
        proof.signatures[idx].signature[0] ^= 0xFF;
        let resolver = hybrid_enrollment();
        assert!(
            verify_proof_signatures(&proof, Some(&client_cnf()), Some(&resolver), FIXTURE_NOW)
                .is_err()
        );
    }

    /// A revoked enrollment denies; so does one whose validity has passed,
    /// and a proof that would outlive its credential.
    #[test]
    fn revoked_expired_or_overlong_enrollment_denies() {
        let proof = parse("P-4");
        let cnf = client_cnf();

        let mut revoked = classical_enrollment();
        {
            let mut r = revoked.resolve_primary(&cnf).unwrap();
            r.revoked = true;
            revoked = rebuild_primary(r, &cnf);
        }
        assert!(
            verify_proof_signatures(&proof, Some(&cnf), Some(&revoked), FIXTURE_NOW).is_err(),
            "revoked enrollment must deny"
        );

        let resolver = classical_enrollment();
        let past_validity = resolver.resolve_primary(&cnf).unwrap().not_after + 1;
        assert!(
            verify_proof_signatures(&proof, Some(&cnf), Some(&resolver), past_validity).is_err(),
            "expired enrollment must deny"
        );

        let mut short = classical_enrollment();
        {
            let mut r = short.resolve_primary(&cnf).unwrap();
            r.not_after = proof.claims.exp - 1;
            short = rebuild_primary(r, &cnf);
        }
        assert!(
            verify_proof_signatures(&proof, Some(&cnf), Some(&short), FIXTURE_NOW).is_err(),
            "a proof outliving its credential must deny"
        );
    }

    fn rebuild_primary(
        record: super::super::enrollment::SignerSuiteRecord,
        cnf: &ed25519_dalek::VerifyingKey,
    ) -> super::super::enrollment::InMemoryEnrollmentResolver {
        let mut resolver = super::super::enrollment::InMemoryEnrollmentResolver::new();
        resolver.enrol_primary(cnf, record).unwrap();
        resolver
    }

    /// An approver enrollment cannot stand in as the credential-bound primary
    /// signer: one key holder must not satisfy two logical roles.
    #[test]
    fn an_approver_enrollment_cannot_act_as_primary() {
        use super::super::enrollment::{
            ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole,
            SignerSuiteRecord,
        };
        let cnf = client_cnf();
        let approver_shaped = SignerSuiteRecord {
            principal: "client".into(),
            suite_id: super::super::SUITE_CLASSICAL.into(),
            components: vec![EnrolledComponent::new(
                b"client-ed25519-1".to_vec(),
                ComponentKey::Ed25519(cnf),
            )],
            epoch: 1,
            role: SignerRole::Approver,
            not_after: 1_786_000_600,
            revoked: false,
        };
        // The registration surface refuses it outright...
        let mut resolver = InMemoryEnrollmentResolver::new();
        assert!(resolver
            .enrol_primary(&cnf, approver_shaped.clone())
            .is_err());
        // ...and the role check denies even if a resolver returned it anyway.
        assert!(approver_shaped
            .check_usable(FIXTURE_NOW, 1_786_000_030, SignerRole::Primary)
            .is_err());
    }

    /// Two logical signer groups resolving to the same principal deny: a
    /// multi-party approval requires distinct enrolled principals.
    #[test]
    fn two_groups_resolving_to_one_principal_deny() {
        use super::super::enrollment::{
            ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole,
            SignerSuiteRecord,
        };
        let cnf = client_cnf();
        let mut resolver = InMemoryEnrollmentResolver::new();
        resolver
            .enrol_primary(
                &cnf,
                SignerSuiteRecord {
                    principal: "client".into(),
                    suite_id: super::super::SUITE_CLASSICAL.into(),
                    components: vec![EnrolledComponent::new(
                        b"client-ed25519-1".to_vec(),
                        ComponentKey::Ed25519(cnf),
                    )],
                    epoch: 1,
                    role: SignerRole::Primary,
                    not_after: 1_786_000_600,
                    revoked: false,
                },
            )
            .unwrap();
        // The approver group is enrolled to the *same* principal.
        resolver
            .enrol_approver(SignerSuiteRecord {
                principal: "client".into(),
                suite_id: super::super::SUITE_CLASSICAL.into(),
                components: vec![EnrolledComponent::new(
                    b"approver-ed25519-1".to_vec(),
                    ComponentKey::Ed25519(ed25519_public("approver-ed25519-1")),
                )],
                epoch: 1,
                role: SignerRole::Approver,
                not_after: 1_786_000_600,
                revoked: false,
            })
            .unwrap();
        assert!(
            verify_proof_signatures(&parse("P-5"), Some(&cnf), Some(&resolver), FIXTURE_NOW)
                .is_err(),
            "P-5 must deny when both groups resolve to one principal"
        );
    }

    // -- response proofs ---------------------------------------------------

    #[test]
    fn p3_response_proof_verifies_against_the_enrolled_service() {
        let proof = parse("P-3");
        let resolver = classical_enrollment();
        let verified = verify_response_proof(
            &proof,
            "registry.svc.hyprstream.test",
            &FIXTURE_REQUEST_ID,
            &resolver,
            FIXTURE_NOW,
        )
        .expect("P-3 must verify against the enrolled service signer");
        assert_eq!(
            verified.primary_principal.as_deref(),
            Some("registry-service")
        );
    }

    /// A response proof is only trusted for the service domain the request was
    /// addressed to, and is never accepted through the request path.
    #[test]
    fn response_proof_is_scoped_and_never_a_request() {
        let proof = parse("P-3");
        let resolver = classical_enrollment();
        assert!(
            verify_response_proof(
                &proof,
                "other.svc.hyprstream.test",
                &FIXTURE_REQUEST_ID,
                &resolver,
                FIXTURE_NOW
            )
            .is_err(),
            "a response for another service domain must deny"
        );
        assert!(
            verify_proof_signatures(&proof, Some(&client_cnf()), Some(&resolver), FIXTURE_NOW)
                .is_err(),
            "a response proof must never verify through the request path"
        );
        assert!(
            verify_response_proof(
                &parse("P-4"),
                "registry.svc.hyprstream.test",
                &FIXTURE_REQUEST_ID,
                &resolver,
                FIXTURE_NOW
            )
            .is_err(),
            "a request proof must never verify through the response path"
        );
    }

    /// A response signed by a different enrolled service's key denies, even
    /// though that key is validly enrolled elsewhere.
    #[test]
    fn response_signed_by_another_enrolled_service_denies() {
        use super::super::enrollment::{
            ComponentKey, EnrolledComponent, InMemoryEnrollmentResolver, SignerRole,
            SignerSuiteRecord,
        };
        let mut resolver = InMemoryEnrollmentResolver::new();
        // The addressed domain enrols a *different* service key.
        resolver
            .enrol_service(
                "registry.svc.hyprstream.test",
                SignerSuiteRecord {
                    principal: "other-service".into(),
                    suite_id: super::super::SUITE_CLASSICAL.into(),
                    components: vec![EnrolledComponent::new(
                        b"service-ed25519-1".to_vec(),
                        ComponentKey::Ed25519(ed25519_public("approver-ed25519-1")),
                    )],
                    epoch: 1,
                    role: SignerRole::Service,
                    not_after: 1_786_000_600,
                    revoked: false,
                },
            )
            .unwrap();
        assert!(verify_response_proof(
            &parse("P-3"),
            "registry.svc.hyprstream.test",
            &FIXTURE_REQUEST_ID,
            &resolver,
            FIXTURE_NOW
        )
        .is_err());
    }

    #[test]
    fn response_proof_without_service_enrollment_denies() {
        let resolver = super::super::enrollment::InMemoryEnrollmentResolver::new();
        assert!(verify_response_proof(
            &parse("P-3"),
            "registry.svc.hyprstream.test",
            &FIXTURE_REQUEST_ID,
            &resolver,
            FIXTURE_NOW
        )
        .is_err());
    }
}
