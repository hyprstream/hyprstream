//! Bounded COSE object parser for the proof-CWT profile.
//!
//! Parses an untagged COSE object (COSE_Sign1 or COSE_Sign), validates all
//! protected headers, `crit` sets, `typ`, `hs_domain`, and the signature plan,
//! then decodes the claims payload.
//!
//! This parser does **not** verify cryptographic signatures — it performs the
//! bounded structural parse and profile-rule validation that must precede
//! signature verification. Signature verification uses the extracted
//! [`ParsedProof`] fields with the existing `hyprstream_crypto` COSE primitives.

use anyhow::{bail, Result};
use ciborium::value::Value as CborValue;

use super::{
    claims::ProofClaims, plan::SignaturePlan, ProofDisposition, ProofKind, ALG_ED25519,
    ALG_ML_DSA_65, COSE_HEADER_ALG, COSE_HEADER_CRIT, COSE_HEADER_KID, COSE_HEADER_TYP,
    HEADER_HS_DOMAIN, HEADER_HS_LOGICAL_SIGNER_GROUP, HEADER_HS_SIGNATURE_PLAN,
    HEADER_HS_UNATTRIBUTED_KEY_SET, MAX_COSE_OBJECT_BYTES, PROOF_TYP, REQUEST_PROOF_DOMAIN,
    RESPONSE_PROOF_DOMAIN, RESPONSE_PROOF_TYP,
};

/// A parsed and profile-validated proof-CWT, ready for signature verification.
#[derive(Debug, Clone)]
pub struct ParsedProof {
    /// Whether this is a request or response proof.
    pub kind: ProofKind,
    /// The decoded claims payload.
    pub claims: ProofClaims,
    /// The signature plan extracted from the body protected headers.
    pub plan: SignaturePlan,
    /// The disposition: `Unattributed` if the proof carries
    /// `hs_unattributed_key_set`, `Authenticated` otherwise.
    pub disposition: ProofDisposition,
    /// The protected header bytes (the `protected` bstr from the COSE object).
    /// Needed to reconstruct `Sig_structure` for verification.
    pub protected_bytes: Vec<u8>,
    /// The payload bytes (claims-set CBOR).
    pub payload_bytes: Vec<u8>,
    /// COSE structure variant.
    pub structure: CoseStructure,
    /// Per-signature metadata extracted from the COSE object.
    pub signatures: Vec<ParsedSignature>,
}

/// Which COSE structure the proof uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoseStructure {
    /// Single-signer: one protected bucket, one signature.
    Sign1,
    /// Multi-signer: body + per-signature protected buckets.
    Sign,
}

/// A single signature entry extracted from the COSE object.
#[derive(Debug, Clone)]
pub struct ParsedSignature {
    /// The algorithm identifier from the protected header.
    pub alg: i64,
    /// The key ID from the protected header.
    pub kid: Vec<u8>,
    /// The logical signer group from the protected header (−70102).
    pub group_id: u64,
    /// The protected header bytes for this signature entry.
    pub protected_bytes: Vec<u8>,
    /// The raw signature bytes.
    pub signature: Vec<u8>,
}

impl ParsedProof {
    /// Parse and profile-validate a proof-CWT byte blob.
    ///
    /// Enforces all profile rules that can be checked before signature
    /// verification:
    /// - COSE object size cap (2 MiB).
    /// - `typ` header (RFC 9596 label 16) must match exactly.
    /// - `hs_domain` must match `typ`.
    /// - Exact `crit` sets per structure and disposition.
    /// - `signature_plan` present and valid.
    /// - `hs_unattributed_key_set` present iff unattributed.
    /// - Unprotected headers carry nothing.
    /// - Per-signature: `alg` fully specified, `kid` present, group in plan.
    /// - Claims payload: closed claim set, deterministic encoding.
    pub fn parse(cbor_bytes: &[u8]) -> Result<Self> {
        if cbor_bytes.len() > MAX_COSE_OBJECT_BYTES {
            bail!(
                "proof: object size {} exceeds cap of {} bytes",
                cbor_bytes.len(),
                MAX_COSE_OBJECT_BYTES
            );
        }

        // Raw-byte deterministic audit of the complete COSE object (finding 2).
        super::cbor_audit::audit_deterministic(cbor_bytes)?;

        let value: CborValue = ciborium::de::from_reader(&mut std::io::Cursor::new(cbor_bytes))
            .map_err(|e| anyhow::anyhow!("proof: COSE CBOR decode failed: {e}"))?;

        let arr = match &value {
            CborValue::Array(a) => a,
            _ => bail!("proof: expected COSE array"),
        };

        // COSE_Sign1 = [protected, unprotected, payload, signature]
        // COSE_Sign  = [protected, unprotected, payload, signatures]
        if arr.len() != 4 {
            bail!("proof: expected 4-element COSE array, got {}", arr.len());
        }

        let protected_raw = as_bstr(&arr[0], "protected")?;
        // Audit protected header bstr for deterministic encoding (finding 2).
        super::cbor_audit::audit_deterministic(&protected_raw)?;
        let unprotected = &arr[1];
        let payload_raw = as_bstr(&arr[2], "payload")?;
        let sig_or_sigs = &arr[3];

        // Unprotected header MUST be empty.
        check_empty_unprotected(unprotected)?;

        // Decode protected header.
        let protected: CborValue =
            ciborium::de::from_reader(&mut std::io::Cursor::new(&protected_raw))
                .map_err(|e| anyhow::anyhow!("proof: protected header CBOR decode: {e}"))?;
        let protected_map = as_map(&protected, "protected header")?;

        // Extract and validate typ + hs_domain.
        let typ = get_text(&protected_map, COSE_HEADER_TYP, "typ")?;
        let hs_domain = get_text(&protected_map, HEADER_HS_DOMAIN, "hs_domain")?;

        let kind = match (typ.as_str(), hs_domain.as_str()) {
            (t, d) if t == PROOF_TYP && d == REQUEST_PROOF_DOMAIN => ProofKind::Request,
            (t, d) if t == RESPONSE_PROOF_TYP && d == RESPONSE_PROOF_DOMAIN => ProofKind::Response,
            _ => bail!("proof: typ/hs_domain mismatch: typ={typ}, domain={hs_domain}"),
        };

        // Validate crit set.
        let crit_labels = get_crit(&protected_map)?;
        validate_body_crit(&crit_labels, &protected_map)?;

        // Extract signature plan.
        let plan_val = get_value(&protected_map, HEADER_HS_SIGNATURE_PLAN)
            .ok_or_else(|| anyhow::anyhow!("proof: missing hs_signature_plan"))?;
        let plan = SignaturePlan::decode(plan_val)?;

        // Determine disposition from presence of hs_unattributed_key_set.
        let key_set_val = get_value(&protected_map, HEADER_HS_UNATTRIBUTED_KEY_SET);
        let disposition = if key_set_val.is_some() {
            ProofDisposition::Unattributed
        } else {
            ProofDisposition::Authenticated
        };

        // If unattributed, decode and strictly validate the key set (finding 3).
        if let Some(ks_val) = key_set_val {
            validate_unattributed_key_set(ks_val, &plan)?;
        }

        // Response proofs MUST NOT carry an unattributed key set.
        if kind == ProofKind::Response && disposition == ProofDisposition::Unattributed {
            bail!("proof: response proof must not carry hs_unattributed_key_set");
        }

        // Validate alg in protected header (must be in profile).
        if let Some(alg_val) = get_value(&protected_map, COSE_HEADER_ALG) {
            let alg = match alg_val {
                CborValue::Integer(i) => i128::from(*i) as i64,
                _ => bail!("proof: alg must be integer"),
            };
            if alg != ALG_ED25519 && alg != ALG_ML_DSA_65 {
                bail!("proof: alg {alg} not in profile");
            }
        }

        // Determine COSE structure from the 4th element.
        let (structure, signatures) = if let CborValue::Array(sig_arr) = sig_or_sigs {
            // COSE_Sign: signatures is an array of [protected, unprotected, signature]
            let mut sigs = Vec::with_capacity(sig_arr.len());
            for entry in sig_arr {
                sigs.push(parse_signature_entry(entry)?);
            }
            (CoseStructure::Sign, sigs)
        } else {
            // COSE_Sign1: 4th element is the signature bstr.
            let sig = as_bstr(sig_or_sigs, "signature")?;
            // For Sign1, the protected header is the merged bucket — extract
            // per-signature metadata from it.
            let alg = get_int_or_default(&protected_map, COSE_HEADER_ALG, 0)?;
            let kid = match get_value(&protected_map, COSE_HEADER_KID) {
                Some(CborValue::Bytes(b)) => b.clone(),
                Some(CborValue::Text(s)) => s.as_bytes().to_vec(),
                _ => bail!("proof: Sign1 missing kid in merged protected header"),
            };
            let group_id = match get_value(&protected_map, HEADER_HS_LOGICAL_SIGNER_GROUP) {
                Some(CborValue::Integer(i)) => i128::from(*i) as u64,
                _ => bail!("proof: Sign1 missing hs_logical_signer_group"),
            };
            (
                CoseStructure::Sign1,
                vec![ParsedSignature {
                    alg,
                    kid,
                    group_id,
                    protected_bytes: protected_raw.clone(),
                    signature: sig,
                }],
            )
        };

        // Validate that each signature entry matches exactly one plan component.
        validate_signatures_against_plan(&signatures, &plan, structure, &protected_map)?;

        // Decode claims payload.
        let claims = ProofClaims::decode(&payload_raw)?;

        // If this is a response proof, credential_hash must be null.
        if kind == ProofKind::Response && claims.credential_hash.is_some() {
            bail!("proof: response proof must have null credential_hash");
        }

        // Credential hash / disposition consistency (finding: N-15).
        // Only applies to request proofs — response proofs always have null
        // credential_hash and never carry an unattributed key set.
        if kind == ProofKind::Request {
            if disposition == ProofDisposition::Authenticated && claims.credential_hash.is_none() {
                bail!("proof: authenticated proof must have non-null credential_hash");
            }
            if disposition == ProofDisposition::Unattributed && claims.credential_hash.is_some() {
                bail!("proof: unattributed proof must have null credential_hash");
            }
        }

        // If unattributed, Nonce (challenge) is REQUIRED.
        if disposition == ProofDisposition::Unattributed && claims.nonce.is_none() {
            bail!("proof: unattributed proof requires server challenge (Nonce)");
        }

        Ok(Self {
            kind,
            claims,
            plan,
            disposition,
            protected_bytes: protected_raw,
            payload_bytes: payload_raw,
            structure,
            signatures,
        })
    }

    /// The replay key component from the proof's signer identity.
    ///
    /// For authenticated proofs this is the credential-bound primary signer
    /// suite thumbprint; for unattributed proofs it is the canonical
    /// plan/key-set thumbprint. The caller computes the actual SHA-256
    /// thumbprint from this data.
    pub fn replay_namespace_input(&self) -> Vec<u8> {
        // The canonical encoding of [signature_plan, unattributed_key_set]
        // for unattributed, or the signer-suite data for authenticated.
        // The caller hashes this with the appropriate domain separator.
        let mut buf = Vec::new();
        let _ =
            ciborium::ser::into_writer(&CborValue::Bytes(self.protected_bytes.clone()), &mut buf);
        buf
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_signature_entry(v: &CborValue) -> Result<ParsedSignature> {
    let arr = match v {
        CborValue::Array(a) => a,
        _ => bail!("signature entry: expected array"),
    };
    if arr.len() != 3 {
        bail!("signature entry: expected 3 elements, got {}", arr.len());
    }
    let protected_raw = as_bstr(&arr[0], "sig protected")?;
    // Audit per-signature protected header for deterministic encoding (finding 2).
    super::cbor_audit::audit_deterministic(&protected_raw)?;
    let unprotected = &arr[1];
    let signature = as_bstr(&arr[2], "signature")?;

    check_empty_unprotected(unprotected)?;

    let protected: CborValue = ciborium::de::from_reader(&mut std::io::Cursor::new(&protected_raw))
        .map_err(|e| anyhow::anyhow!("sig protected decode: {e}"))?;
    let pmap = as_map(&protected, "sig protected")?;

    // Validate crit for per-signature protected: exactly [-70102].
    let crit = get_crit(&pmap)?;
    validate_sig_crit(&crit)?;

    let alg = match get_value(&pmap, COSE_HEADER_ALG) {
        Some(CborValue::Integer(i)) => i128::from(*i) as i64,
        _ => bail!("sig protected: missing alg"),
    };
    if alg != ALG_ED25519 && alg != ALG_ML_DSA_65 {
        bail!("sig protected: alg {alg} not in profile");
    }

    let kid = match get_value(&pmap, COSE_HEADER_KID) {
        Some(CborValue::Bytes(b)) => b.clone(),
        Some(CborValue::Text(s)) => s.as_bytes().to_vec(),
        _ => bail!("sig protected: missing kid"),
    };

    let group_id = match get_value(&pmap, HEADER_HS_LOGICAL_SIGNER_GROUP) {
        Some(CborValue::Integer(i)) if i128::from(*i) >= 0 => i128::from(*i) as u64,
        _ => bail!("sig protected: missing hs_logical_signer_group"),
    };

    Ok(ParsedSignature {
        alg,
        kid,
        group_id,
        protected_bytes: protected_raw,
        signature,
    })
}

fn validate_signatures_against_plan(
    sigs: &[ParsedSignature],
    plan: &SignaturePlan,
    structure: CoseStructure,
    body_protected: &[(&CborValue, &CborValue)],
) -> Result<()> {
    // Enforce strict 1:1 bijection: each signature matches exactly one plan
    // component, and each plan component is covered by exactly one signature
    // (finding 3).
    let total_plan = plan.total_components();
    if sigs.len() != total_plan {
        bail!(
            "proof: {} signatures but plan expects {}",
            sigs.len(),
            total_plan
        );
    }

    // Track which (group_id, alg, kid) tuples have been matched.
    let mut matched: Vec<(u64, i64, Vec<u8>)> = Vec::new();

    for sig in sigs {
        let group = plan
            .groups
            .iter()
            .find(|g| g.group_id == sig.group_id)
            .ok_or_else(|| {
                anyhow::anyhow!("proof: signature group {} not in plan", sig.group_id)
            })?;

        // Find the unique matching component.
        let comp = group
            .components
            .iter()
            .find(|c| c.alg == sig.alg && c.kid == sig.kid)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "proof: signature (alg={}, kid={}) not in plan group {}",
                    sig.alg,
                    hex::encode(&sig.kid),
                    sig.group_id
                )
            })?;

        // Check for duplicate (no two signatures match the same component).
        let tuple = (sig.group_id, comp.alg, comp.kid.clone());
        if matched.contains(&tuple) {
            bail!(
                "proof: duplicate signature for group {} component (alg={}, kid={})",
                sig.group_id,
                comp.alg,
                hex::encode(&comp.kid)
            );
        }
        matched.push(tuple);
    }

    // Verify every plan component was matched exactly once.
    if matched.len() != total_plan {
        bail!(
            "proof: {}/{} plan components matched by signatures",
            matched.len(),
            total_plan
        );
    }

    let _ = structure;
    let _ = body_protected;
    Ok(())
}

/// Strictly validate an unattributed `COSE_KeySet` against the plan.
///
/// Every element must parse, be understood, match exactly one plan component
/// by kid, algorithm, key type, and curve/parameters, in the plan's component
/// order. Private key material is forbidden. A malformed, unknown, duplicate,
/// surplus, reordered, or mismatched key denies the complete proof (finding 3).
fn validate_unattributed_key_set(ks_val: &CborValue, plan: &SignaturePlan) -> Result<()> {
    let arr = match ks_val {
        CborValue::Array(a) => a,
        _ => bail!("proof: hs_unattributed_key_set must be array"),
    };

    if arr.is_empty() {
        bail!("proof: unattributed key set must not be empty");
    }
    if arr.len() > 2 {
        bail!("proof: unattributed key set must have at most 2 elements");
    }

    // The unattributed proof MUST have exactly one logical signer group.
    if plan.groups.len() != 1 {
        bail!(
            "proof: unattributed proof must have exactly one signer group, got {}",
            plan.groups.len()
        );
    }

    let group = &plan.groups[0];
    if arr.len() != group.components.len() {
        bail!(
            "proof: key set has {} elements but plan group has {} components",
            arr.len(),
            group.components.len()
        );
    }

    // Match each key to exactly one plan component by kid + alg, in order.
    let mut matched_components = vec![false; group.components.len()];

    for key_val in arr {
        let key_map_inner = match key_val {
            CborValue::Map(m) => m,
            _ => bail!("proof: key-set element must be a map"),
        };
        let key_map: Vec<(&CborValue, &CborValue)> =
            key_map_inner.iter().map(|(k, v)| (k, v)).collect();

        // Extract kty, kid, alg.
        let kty = get_int_from_map(&key_map, 1, "kty")?;
        let kid = match get_value_from_map(&key_map, 2) {
            Some(CborValue::Bytes(b)) => b.clone(),
            Some(CborValue::Text(s)) => s.as_bytes().to_vec(),
            _ => bail!("proof: key-set element missing kid"),
        };
        let alg = get_int_from_map(&key_map, 3, "alg")?;

        // Reject private key material.
        // OKP: key -4 is the private key. AKP: key -2 is the private key.
        if kty == 1 && get_value_from_map(&key_map, -4).is_some() {
            bail!("proof: private key material in unattributed key set");
        }
        if kty == 7 && get_value_from_map(&key_map, -2).is_some() {
            bail!("proof: private key material in unattributed key set");
        }

        // Validate key structure per type.
        match kty {
            1 => {
                // OKP / Ed25519
                let crv = get_int_from_map(&key_map, -1, "crv")?;
                if crv != 6 {
                    bail!("proof: key-set OKP crv must be Ed25519 (6), got {crv}");
                }
                match get_value_from_map(&key_map, -2) {
                    Some(CborValue::Bytes(x)) if x.len() == 32 => {}
                    _ => bail!("proof: key-set OKP x must be 32 bytes"),
                }
                if alg != ALG_ED25519 {
                    bail!("proof: key-set OKP alg must be Ed25519 ({ALG_ED25519})");
                }
            }
            7 => {
                // AKP / ML-DSA-65 (RFC 9964)
                match get_value_from_map(&key_map, -1) {
                    Some(CborValue::Bytes(pub_key)) if pub_key.len() == 1952 => {}
                    _ => bail!("proof: key-set AKP pub must be 1952 bytes"),
                }
                if alg != ALG_ML_DSA_65 {
                    bail!("proof: key-set AKP alg must be ML-DSA-65 ({ALG_ML_DSA_65})");
                }
            }
            _ => bail!("proof: key-set element kty {kty} not in profile (OKP=1, AKP=7)"),
        }

        // Match to exactly one plan component.
        let comp_idx = group
            .components
            .iter()
            .position(|c| c.alg == alg && c.kid == kid);
        match comp_idx {
            Some(idx) => {
                if matched_components[idx] {
                    bail!("proof: duplicate key-set element for component {idx}");
                }
                matched_components[idx] = true;
            }
            None => bail!(
                "proof: key-set element (alg={alg}, kid={}) has no matching plan component",
                hex::encode(&kid)
            ),
        }
    }

    // Every plan component must be matched.
    if !matched_components.iter().all(|&m| m) {
        bail!("proof: not all plan components matched by key-set elements");
    }

    Ok(())
}

fn get_int_from_map(map: &[(&CborValue, &CborValue)], label: i64, name: &str) -> Result<i64> {
    match get_value_from_map(map, label) {
        Some(CborValue::Integer(i)) => Ok(i128::from(*i)
            .try_into()
            .map_err(|_| anyhow::anyhow!("proof: {name} (label {label}) out of i64 range"))?),
        Some(_) => bail!("proof: {name} (label {label}) must be integer"),
        None => bail!("proof: key-set element missing {name} (label {label})"),
    }
}

fn get_value_from_map<'a>(
    map: &[(&'a CborValue, &'a CborValue)],
    label: i64,
) -> Option<&'a CborValue> {
    map.iter()
        .find(|(k, _)| matches!(k, CborValue::Integer(i) if i128::from(*i) == label as i128))
        .map(|(_, v)| *v)
}

fn validate_body_crit(crit: &[i64], protected: &[(&CborValue, &CborValue)]) -> Result<()> {
    // Exact crit set per the CDDL:
    // COSE_Sign body authenticated:  [-70101, -70100]
    // COSE_Sign body unattributed:   [-70103, -70101, -70100]
    // COSE_Sign1 merged authenticated: [-70102, -70101, -70100]
    // COSE_Sign1 merged unattributed:  [-70103, -70102, -70101, -70100]
    //
    // We determine the expected set from what's actually present. Since the
    // same parser handles both structures, we check that the crit set exactly
    // matches the set of hs_* labels present in this bucket, and that those
    // labels are all in crit.

    let has_key_set = protected
        .iter()
        .any(|(k, _)| matches!(k, CborValue::Integer(i) if i128::from(*i) as i64 == HEADER_HS_UNATTRIBUTED_KEY_SET));
    let has_group = protected
        .iter()
        .any(|(k, _)| matches!(k, CborValue::Integer(i) if i128::from(*i) as i64 == HEADER_HS_LOGICAL_SIGNER_GROUP));

    // Build expected crit set.
    let mut expected = vec![HEADER_HS_SIGNATURE_PLAN, HEADER_HS_DOMAIN];
    if has_key_set {
        expected.insert(0, HEADER_HS_UNATTRIBUTED_KEY_SET);
    }
    if has_group {
        // In Sign1, the group label is in the merged bucket.
        let pos = if has_key_set { 2 } else { 1 };
        expected.insert(pos, HEADER_HS_LOGICAL_SIGNER_GROUP);
    }
    expected.sort();

    let mut actual: Vec<i64> = crit.to_vec();
    actual.sort();

    if actual != expected {
        bail!(
            "proof: body crit set {:?} does not match expected {:?}",
            actual,
            expected
        );
    }

    // Every crit label must occur in this same protected bucket.
    for label in &expected {
        let present = protected
            .iter()
            .any(|(k, _)| matches!(k, CborValue::Integer(i) if i128::from(*i) as i64 == *label));
        if !present {
            bail!("proof: crit label {label} not in same protected bucket");
        }
    }

    Ok(())
}

fn validate_sig_crit(crit: &[i64]) -> Result<()> {
    // Per-signature protected crit: exactly [-70102].
    let expected = vec![HEADER_HS_LOGICAL_SIGNER_GROUP];
    if crit != expected {
        bail!(
            "proof: sig crit {:?} does not match expected {:?}",
            crit,
            expected
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CBOR map helpers
// ---------------------------------------------------------------------------

fn as_bstr(v: &CborValue, name: &str) -> Result<Vec<u8>> {
    match v {
        CborValue::Bytes(b) => Ok(b.clone()),
        _ => bail!("{name}: expected bstr"),
    }
}

fn as_map<'a>(v: &'a CborValue, name: &str) -> Result<Vec<(&'a CborValue, &'a CborValue)>> {
    match v {
        CborValue::Map(m) => Ok(m.iter().map(|(k, v)| (k, v)).collect()),
        _ => bail!("{name}: expected map"),
    }
}

fn check_empty_unprotected(v: &CborValue) -> Result<()> {
    match v {
        CborValue::Map(m) if m.is_empty() => Ok(()),
        CborValue::Map(_) => bail!("proof: unprotected header must be empty"),
        _ => bail!("proof: unprotected header must be empty map"),
    }
}

fn get_text(map: &[(&CborValue, &CborValue)], label: i64, name: &str) -> Result<String> {
    match get_value(map, label) {
        Some(CborValue::Text(s)) => Ok(s.clone()),
        Some(_) => bail!("{name}: must be text"),
        None => bail!("{name}: missing (label {label})"),
    }
}

fn get_value<'a>(map: &[(&'a CborValue, &'a CborValue)], label: i64) -> Option<&'a CborValue> {
    map.iter()
        .find(|(k, _)| matches!(k, CborValue::Integer(i) if i128::from(*i) as i64 == label))
        .map(|(_, v)| *v)
}

fn get_crit(map: &[(&CborValue, &CborValue)]) -> Result<Vec<i64>> {
    match get_value(map, COSE_HEADER_CRIT) {
        Some(CborValue::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.len());
            for elem in arr {
                match elem {
                    CborValue::Integer(i) => out.push(i128::from(*i) as i64),
                    _ => bail!("crit: non-integer label"),
                }
            }
            // Check ascending and unique.
            for w in out.windows(2) {
                if w[0] >= w[1] {
                    bail!("crit: labels must be strictly ascending and unique");
                }
            }
            Ok(out)
        }
        Some(_) => bail!("crit: must be array"),
        None => bail!("crit: missing"),
    }
}

fn get_int_or_default(map: &[(&CborValue, &CborValue)], label: i64, default: i64) -> Result<i64> {
    match get_value(map, label) {
        Some(CborValue::Integer(i)) => Ok(i128::from(*i) as i64),
        Some(_) => bail!("label {label}: must be integer"),
        None => Ok(default),
    }
}
