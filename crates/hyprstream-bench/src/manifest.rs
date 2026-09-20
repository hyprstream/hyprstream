//! The frozen item manifest: per-item blake3 hashes, family designations,
//! generation config, and license/disclosure pointers.
//!
//! **Freeze rule (plan v1.6): the manifest is frozen before any P1.3
//! synthesis run.** P1.3 consumes this artifact — the family firewall is "a
//! consumed artifact, not a promise": gate families are never synthesized
//! into training data, and no training item may hash-collide with (or
//! duplicate) a manifest item.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::family::{Designation, Family};
use crate::gen::{generate_all, BenchConfig};
use crate::item::Item;

/// Release id of the frozen benchmark.
pub const RELEASE: &str = "vob-1.0";

/// One manifest row per item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestItem {
    /// Item id.
    pub id: String,
    /// blake3 hex digest of the item's canonical bytes.
    pub blake3: String,
    /// Family id.
    pub family: String,
    /// Stratum wire form (`clean` / `nearmiss` / `perm-N`).
    pub stratum: String,
    /// Permutation group id.
    pub group: String,
    /// Question kind (`noul` / `choice` / `score`).
    pub kind: String,
    /// Correct label index in canonical label order.
    pub truth: usize,
}

/// Family designation row — the family-level holdout firewall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyRow {
    /// Family id.
    pub id: String,
    /// `open` (train-usable) or `gate` (never synthesized; zero-shot gate).
    pub designation: String,
}

/// The frozen manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Release id ([`RELEASE`]).
    pub release: String,
    /// Harness crate + version that generated this manifest.
    pub harness: String,
    /// Pinned generation config.
    pub config: ManifestConfig,
    /// Family designations (the firewall).
    pub families: Vec<FamilyRow>,
    /// License pointers: harness Apache-2.0, items CC-BY-4.0.
    pub license: ManifestLicense,
    /// Path of the pre-committed disclosure text + its blake3 digest.
    pub disclosure: ManifestDisclosure,
    /// Total item count.
    pub item_count: usize,
    /// Per-item rows, in generation order.
    pub items: Vec<ManifestItem>,
}

/// Pinned generation config inside the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestConfig {
    /// Base items per (family, base stratum, kind).
    pub seeds_per_stratum: u32,
    /// Seed stream base (hex).
    pub seed_base: String,
}

/// License pointers inside the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestLicense {
    /// Harness license (Apache-2.0).
    pub harness: String,
    /// Generated-items license (CC-BY-4.0).
    pub items: String,
}

/// Disclosure pointer inside the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestDisclosure {
    /// Repo-relative path of the disclosure text.
    pub path: String,
    /// blake3 hex digest of the disclosure file at freeze time.
    pub blake3: String,
}

/// Errors from manifest verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// Manifest release id mismatch.
    ReleaseMismatch { expected: String, found: String },
    /// Item count mismatch.
    CountMismatch { expected: usize, found: usize },
    /// Item at `index` diverged (first divergence reported).
    ItemMismatch { index: usize, id: String },
    /// A family designation changed (firewall violation).
    DesignationChanged { family: String },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReleaseMismatch { expected, found } => {
                write!(f, "release mismatch: expected {expected}, found {found}")
            }
            Self::CountMismatch { expected, found } => {
                write!(f, "item count mismatch: expected {expected}, found {found}")
            }
            Self::ItemMismatch { index, id } => {
                write!(f, "item #{index} ({id}) diverged from the frozen manifest")
            }
            Self::DesignationChanged { family } => {
                write!(f, "family {family} designation changed — firewall violation")
            }
        }
    }
}

impl std::error::Error for VerifyError {}

/// Build the manifest from a fresh generation run.
pub fn build(config: &BenchConfig, items: &[Item], disclosure_blake3: String) -> Manifest {
    Manifest {
        release: RELEASE.to_owned(),
        harness: format!("hyprstream-bench {}", env!("CARGO_PKG_VERSION")),
        config: ManifestConfig {
            seeds_per_stratum: config.seeds_per_stratum,
            seed_base: format!("0x{:x}", config.seed_base),
        },
        families: Family::ALL
            .iter()
            .map(|family| FamilyRow {
                id: family.as_str().to_owned(),
                designation: family.designation().as_str().to_owned(),
            })
            .collect(),
        license: ManifestLicense {
            harness: "Apache-2.0".to_owned(),
            items: "CC-BY-4.0".to_owned(),
        },
        disclosure: ManifestDisclosure {
            path: "crates/hyprstream-bench/DISCLOSURE.md".to_owned(),
            blake3: disclosure_blake3,
        },
        item_count: items.len(),
        items: items
            .iter()
            .map(|item| ManifestItem {
                id: item.id.clone(),
                blake3: item.hash(),
                family: item.family.as_str().to_owned(),
                stratum: item.stratum.as_str(),
                group: item.group.clone(),
                kind: item.question.kind.as_str().to_owned(),
                truth: item.truth,
            })
            .collect(),
    }
}

impl Manifest {
    /// The firewalled gate families (P1.4 leg-e measurement surface).
    pub fn gate_families(&self) -> Vec<Family> {
        self.families
            .iter()
            .filter(|row| row.designation == Designation::Gate.as_str())
            .filter_map(|row| Family::from_id(&row.id))
            .collect()
    }

    /// Whether `hash` belongs to a frozen item (P1.3 item-level firewall).
    pub fn contains_hash(&self, hash: &str) -> bool {
        self.items.iter().any(|item| item.blake3 == hash)
    }

    /// Deterministic pretty JSON (2-space, trailing newline).
    pub fn to_json(&self) -> String {
        match serde_json::to_string_pretty(self) {
            Ok(mut out) => {
                out.push('\n');
                out
            }
            // Manifest derives Serialize over JSON-safe fields only.
            Err(_) => unreachable!("manifest serialization is infallible"),
        }
    }

    /// Parse a manifest from JSON.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Re-generate from `config` and verify every row against this manifest.
    pub fn verify(&self, config: &BenchConfig) -> Result<(), VerifyError> {
        if self.release != RELEASE {
            return Err(VerifyError::ReleaseMismatch {
                expected: RELEASE.to_owned(),
                found: self.release.clone(),
            });
        }
        let fresh = generate_all(config);
        if fresh.len() != self.item_count {
            return Err(VerifyError::CountMismatch {
                expected: self.item_count,
                found: fresh.len(),
            });
        }
        for (index, (row, item)) in self.items.iter().zip(fresh.iter()).enumerate() {
            if row.id != item.id || row.blake3 != item.hash() {
                return Err(VerifyError::ItemMismatch {
                    index,
                    id: row.id.clone(),
                });
            }
        }
        for row in &self.families {
            if let Some(family) = Family::from_id(&row.id) {
                if row.designation != family.designation().as_str() {
                    return Err(VerifyError::DesignationChanged {
                        family: row.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// blake3 hex digest of a file's bytes (used to pin DISCLOSURE.md).
pub fn blake3_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for byte in blake3::hash(bytes).as_bytes() {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::gen::generate_all;

    #[test]
    fn manifest_roundtrip_json() {
        let config = BenchConfig {
            seeds_per_stratum: 2,
            seed_base: 0x06,
        };
        let items = generate_all(&config);
        let manifest = build(&config, &items, "0".repeat(64));
        let parsed = Manifest::from_json(&manifest.to_json()).unwrap();
        assert_eq!(manifest, parsed);
    }

    #[test]
    fn verify_detects_regeneration_and_tamper() {
        let config = BenchConfig {
            seeds_per_stratum: 2,
            seed_base: 0x06,
        };
        let items = generate_all(&config);
        let manifest = build(&config, &items, "0".repeat(64));
        manifest.verify(&config).unwrap();

        let mut tampered = manifest.clone();
        tampered.items[0].blake3 = "f".repeat(64);
        assert!(matches!(
            tampered.verify(&config),
            Err(VerifyError::ItemMismatch { index: 0, .. })
        ));
    }

    #[test]
    fn gate_families_match_designation() {
        let config = BenchConfig::default();
        let items = generate_all(&config);
        let manifest = build(&config, &items, "0".repeat(64));
        assert_eq!(
            manifest.gate_families(),
            vec![Family::Temporal, Family::Syllogism]
        );
    }
}
