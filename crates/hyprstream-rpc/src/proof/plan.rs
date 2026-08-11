//! Canonical signature-plan types — the signed description of every logical
//! signer group and its components.
//!
//! The plan is carried in the body protected headers as `hs_signature_plan`
//! (−70101). It is covered by every component's `Sig_structure`, so a
//! surviving component still attests its suite/group membership.

use anyhow::{bail, Result};
use ciborium::value::Value as CborValue;

use super::{
    ALG_ED25519, ALG_ML_DSA_65, MAX_COMPONENTS_PER_GROUP, MAX_KID_BYTES,
    MAX_SIGNER_GROUPS, MAX_SIGNATURE_ENTRIES, MAX_SUITE_ID_BYTES, SUITE_CLASSICAL,
    SUITE_HYBRID,
};

/// A single component within a signer group (one COSE algorithm + key ID).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureComponent {
    /// Fully-specified COSE algorithm (Ed25519 −19 or ML-DSA-65 −49).
    pub alg: i64,
    /// Key identifier (1..64 bytes).
    pub kid: Vec<u8>,
}

/// One logical signer group — one or two components authenticating one
/// principal. For a hybrid suite the group has two components (Ed25519 +
/// ML-DSA-65) that count as one logical signer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignerGroup {
    /// Ascending group ID, unique within the plan.
    pub group_id: u64,
    /// Versioned suite identifier (e.g. `hs-cose-sign-ed25519-v1`).
    pub suite_id: String,
    /// 1..2 signature components in the suite-declared order.
    pub components: Vec<SignatureComponent>,
}

/// The complete set of logical signer groups that actually sign. Sorted by
/// ascending `group_id`; IDs unique; every `(alg, kid)` pair unique across
/// the complete plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignaturePlan {
    pub groups: Vec<SignerGroup>,
}

impl SignaturePlan {
    /// Decode a signature plan from a CBOR value (the `hs_signature_plan`
    /// header value).
    pub fn decode(v: &CborValue) -> Result<Self> {
        let arr = match v {
            CborValue::Array(a) => a,
            _ => bail!("signature_plan: expected array"),
        };
        if arr.is_empty() {
            bail!("signature_plan: empty plan");
        }
        if arr.len() > MAX_SIGNER_GROUPS {
            bail!(
                "signature_plan: {} groups exceeds cap of {}",
                arr.len(),
                MAX_SIGNER_GROUPS
            );
        }

        let mut groups = Vec::with_capacity(arr.len());
        for elem in arr {
            groups.push(SignerGroup::decode(elem)?);
        }

        // Validate: ascending unique group IDs.
        for w in groups.windows(2) {
            if w[0].group_id >= w[1].group_id {
                bail!("signature_plan: group IDs not strictly ascending or not unique");
            }
        }

        // Validate: total signature entries ≤ 16, unique (alg, kid) pairs.
        let mut total = 0usize;
        let mut seen_pairs = std::collections::HashSet::new();
        for g in &groups {
            total += g.components.len();
            for c in &g.components {
                let pair = (c.alg, c.kid.clone());
                if !seen_pairs.insert(pair) {
                    bail!("signature_plan: duplicate (alg, kid) pair across plan");
                }
            }
        }
        if total > MAX_SIGNATURE_ENTRIES {
            bail!(
                "signature_plan: {total} signature entries exceeds cap of {MAX_SIGNATURE_ENTRIES}"
            );
        }

        // Validate each group's suite.
        for g in &groups {
            validate_suite(g)?;
        }

        Ok(Self { groups })
    }

    /// Total number of signature entries expected.
    pub fn total_components(&self) -> usize {
        self.groups.iter().map(|g| g.components.len()).sum()
    }
}

impl SignerGroup {
    fn decode(v: &CborValue) -> Result<Self> {
        let map = match v {
            CborValue::Map(m) => m,
            _ => bail!("signer_group: expected map"),
        };

        let mut group_id = None;
        let mut suite_id = None;
        let mut components = None;

        for (key, val) in map.iter() {
            let ik = match key {
                CborValue::Integer(i) => i128::from(*i),
                _ => bail!("signer_group: non-integer key"),
            };
            match ik {
                1 => {
                    group_id = Some(decode_uint(val, "group_id")?);
                }
                2 => {
                    suite_id = Some(decode_text(val, "suite_id", MAX_SUITE_ID_BYTES)?);
                }
                3 => {
                    components = Some(decode_components(val)?);
                }
                _ => bail!("signer_group: unknown key {ik}"),
            }
        }

        let group_id = group_id.ok_or_else(|| anyhow::anyhow!("signer_group: missing group_id"))?;
        let suite_id = suite_id.ok_or_else(|| anyhow::anyhow!("signer_group: missing suite_id"))?;
        let components =
            components.ok_or_else(|| anyhow::anyhow!("signer_group: missing components"))?;

        Ok(Self {
            group_id,
            suite_id,
            components,
        })
    }
}

fn decode_components(v: &CborValue) -> Result<Vec<SignatureComponent>> {
    let arr = match v {
        CborValue::Array(a) => a,
        _ => bail!("components: expected array"),
    };
    if arr.is_empty() {
        bail!("components: empty");
    }
    if arr.len() > MAX_COMPONENTS_PER_GROUP {
        bail!(
            "components: {} exceeds cap of {}",
            arr.len(),
            MAX_COMPONENTS_PER_GROUP
        );
    }

    let mut out = Vec::with_capacity(arr.len());
    for elem in arr {
        let map = match elem {
            CborValue::Map(m) => m,
            _ => bail!("signature_component: expected map"),
        };
        let mut alg = None;
        let mut kid = None;
        for (key, val) in map.iter() {
            let ik = match key {
                CborValue::Integer(i) => i128::from(*i),
                _ => bail!("signature_component: non-integer key"),
            };
            match ik {
                1 => {
                    alg = Some(decode_int(val, "alg")?);
                }
                2 => {
                    let k = decode_bstr(val, "kid")?;
                    if k.is_empty() || k.len() > MAX_KID_BYTES {
                        bail!("kid: must be 1..{MAX_KID_BYTES} bytes, got {}", k.len());
                    }
                    kid = Some(k);
                }
                _ => bail!("signature_component: unknown key {ik}"),
            }
        }
        let alg = alg.ok_or_else(|| anyhow::anyhow!("signature_component: missing alg"))?;
        if alg != ALG_ED25519 && alg != ALG_ML_DSA_65 {
            bail!("signature_component: alg {alg} not in profile (only {ALG_ED25519} and {ALG_ML_DSA_65})");
        }
        let kid = kid.ok_or_else(|| anyhow::anyhow!("signature_component: missing kid"))?;
        out.push(SignatureComponent { alg, kid });
    }
    Ok(out)
}

/// Validate a signer group against the versioned suite registry.
fn validate_suite(g: &SignerGroup) -> Result<()> {
    match g.suite_id.as_str() {
        s if s == SUITE_CLASSICAL => {
            if g.components.len() != 1 {
                bail!(
                    "suite {SUITE_CLASSICAL}: expected exactly 1 component, got {}",
                    g.components.len()
                );
            }
            if g.components[0].alg != ALG_ED25519 {
                bail!(
                    "suite {SUITE_CLASSICAL}: component alg must be {ALG_ED25519}, got {}",
                    g.components[0].alg
                );
            }
        }
        s if s == SUITE_HYBRID => {
            if g.components.len() != 2 {
                bail!(
                    "suite {SUITE_HYBRID}: expected exactly 2 components, got {}",
                    g.components.len()
                );
            }
            // Component order: Ed25519 (-19) first, then ML-DSA-65 (-49).
            if g.components[0].alg != ALG_ED25519 {
                bail!("suite {SUITE_HYBRID}: first component must be Ed25519 ({ALG_ED25519})");
            }
            if g.components[1].alg != ALG_ML_DSA_65 {
                bail!("suite {SUITE_HYBRID}: second component must be ML-DSA-65 ({ALG_ML_DSA_65})");
            }
        }
        other => bail!("validate_suite: unknown suite_id '{other}'"),
    }
    Ok(())
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

fn decode_text(v: &CborValue, name: &str, max: usize) -> Result<String> {
    match v {
        CborValue::Text(s) => {
            if s.len() > max {
                bail!("{name}: exceeds {max} bytes");
            }
            Ok(s.clone())
        }
        _ => bail!("{name}: must be text"),
    }
}

fn decode_bstr(v: &CborValue, name: &str) -> Result<Vec<u8>> {
    match v {
        CborValue::Bytes(b) => Ok(b.clone()),
        _ => bail!("{name}: must be bstr"),
    }
}
