//! Strict `$dispatchMac`/`$dispatchPublic` grammar and the `InitialLabelMap`
//! (v16 §6, WS-D / #1505).
//!
//! This module is the ONE parser for the dispatch-label grammar, shared by the
//! proc-macro codegen (`hyprstream-rpc-derive`, via the cgr re-export) and any
//! build-time validation in this crate. A second implementation of this
//! grammar is a refactor-plan §0.3 violation.
//!
//! Grammar (v16 §6):
//!
//! ```text
//! <level>:<assurance>[:<compartment>[,<compartment>...]]
//! ```
//!
//! Every failure mode below is a **build error** at code generation — an
//! annotation failure can never produce an unlabeled runtime row:
//!
//! - unknown level or assurance;
//! - empty components (`internal:`, `:pq-hybrid`, `secret:pq-hybrid:`);
//! - duplicate compartments;
//! - compartments not listed in canonical (bit-ascending) order;
//! - compartment names not present in the checked-in `InitialLabelMap`;
//! - trailing/leading whitespace anywhere (the annotation is not trimmed);
//! - system low (`public:unverified`, no compartments) written through
//!   `$dispatchMac` — that label is exactly what `$dispatchPublic` expands to,
//!   so spelling it through the MAC annotation is the one form that would let
//!   a "labeled" row be indistinguishable from a public row; and
//! - `$dispatchPublic` with an empty or whitespace-only reason.

use std::collections::HashMap;

use serde::Deserialize;

/// The checked-in compartment vocabulary: name → stable bit.
///
/// Versioned, append-only, tombstoned (see `schema/initial-label-map.json`).
/// Parsed once per process (build/codegen), never on a hot path.
#[derive(Debug, Clone, Deserialize)]
struct InitialLabelMapFile {
    version: u32,
    #[serde(default)]
    compartments: Vec<InitialLabelMapEntry>,
    #[serde(default)]
    retired: Vec<InitialLabelMapEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct InitialLabelMapEntry {
    name: String,
    bit: u32,
}

/// The parsed [`crate::schema::dispatch_label`] vocabulary.
#[derive(Debug, Clone)]
pub struct InitialLabelMap {
    version: u32,
    /// Live name → bit. Built with duplicate detection at load.
    bits: HashMap<String, u32>,
    /// Tombstoned names — forever unusable, present for auditability.
    retired: Vec<InitialLabelMapEntry>,
}

/// The map file is malformed (not valid JSON against the schema above, a
/// duplicate name/bit, or a bit outside the `SecurityLabel` compartment width).
#[derive(Debug)]
pub struct InitialLabelMapError(pub String);

impl std::fmt::Display for InitialLabelMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InitialLabelMap: {}", self.0)
    }
}

/// Maximum compartment bits — mirrors
/// `hyprstream_rpc::auth::mac::label::MAX_COMPARTMENTS` (the fixed `u64`
/// bitset width of `SecurityLabel`). rpc-build cannot depend on the runtime
/// crate (the runtime crate depends on this one's consumers), so the constant
/// is restated here and cross-checked by a drift test in `hyprstream-rpc`.
pub const MAX_COMPARTMENT_BITS: u32 = 64;

/// The label axes the grammar accepts. These mirror the runtime
/// `Level`/`Assurance` enums; the mapping to runtime values lives in the
/// generated code, keyed on these strings.
pub const LEVELS: &[&str] = &["public", "internal", "confidential", "secret"];
pub const ASSURANCES: &[&str] = &["unverified", "classical", "pq-hybrid"];

/// The system-low label text — the exact expansion of `$dispatchPublic`.
pub const SYSTEM_LOW: &str = "public:unverified";

/// The side-effect-free scope actions (S3 `ScopeAction` Block A: read-class).
/// Every other action in the closed vocabulary is mutating. This is the
/// CODEGEN-TIME copy used to derive `MutationSemantics`; the runtime
/// validator keeps its own copy in `hyprstream_rpc::proof::policy`, and the
/// full-inventory validation test is the drift gate between them: a row
/// generated under one list fails the other's validation.
pub const READ_CLASS_ACTIONS: &[&str] = &["query", "subscribe"];

/// The checked-in map, embedded at compile time so codegen (proc-macro and
/// build binary alike) reads exactly the reviewed artifact.
const INITIAL_LABEL_MAP_JSON: &str = include_str!("../../schema/initial-label-map.json");

/// A parsed `$dispatchMac` label: the axis values plus the resolved,
/// canonical-ordered compartment bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchLabel {
    pub level: &'static str,
    pub assurance: &'static str,
    /// Compartment bits in ascending order (canonical form).
    pub compartment_bits: Vec<u32>,
}

impl DispatchLabel {
    /// Render back to the canonical annotation text (round-trip identity).
    pub fn to_canonical_text(&self, map: &InitialLabelMap) -> Option<String> {
        let mut names: Vec<(u32, &str)> = Vec::with_capacity(self.compartment_bits.len());
        for bit in &self.compartment_bits {
            names.push((*bit, map.name_of(*bit)?));
        }
        names.sort_by_key(|(b, _)| *b);
        let mut text = format!("{}:{}", self.level, self.assurance);
        if !names.is_empty() {
            text.push(':');
            text.push_str(
                &names
                    .iter()
                    .map(|(_, n)| *n)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        Some(text)
    }

    /// Whether this is exactly the system-low label.
    pub fn is_system_low(&self) -> bool {
        self.level == "public"
            && self.assurance == "unverified"
            && self.compartment_bits.is_empty()
    }
}

impl InitialLabelMap {
    /// Load the checked-in map (embedded at compile time).
    pub fn load() -> Result<Self, InitialLabelMapError> {
        Self::parse_text(INITIAL_LABEL_MAP_JSON)
    }

    /// Parse map text (unit-testable seam; production always uses
    /// [`Self::load`]).
    pub fn parse_text(text: &str) -> Result<Self, InitialLabelMapError> {
        let file: InitialLabelMapFile = serde_json::from_str(text)
            .map_err(|e| InitialLabelMapError(format!("invalid map JSON: {e}")))?;
        let mut bits = HashMap::new();
        let mut seen_bits = std::collections::HashSet::new();
        for entry in &file.compartments {
            if entry.name.trim().is_empty() || entry.name != entry.name.trim() {
                return Err(InitialLabelMapError(format!(
                    "compartment name {:?} is empty or padded",
                    entry.name
                )));
            }
            if entry.bit >= MAX_COMPARTMENT_BITS {
                return Err(InitialLabelMapError(format!(
                    "compartment {:?} bit {} exceeds the {}-bit width",
                    entry.name, entry.bit, MAX_COMPARTMENT_BITS
                )));
            }
            if bits.contains_key(&entry.name) {
                return Err(InitialLabelMapError(format!(
                    "compartment {:?} assigned twice",
                    entry.name
                )));
            }
            if !seen_bits.insert(entry.bit) {
                return Err(InitialLabelMapError(format!(
                    "bit {} assigned twice",
                    entry.bit
                )));
            }
            bits.insert(entry.name.clone(), entry.bit);
        }
        for entry in &file.retired {
            if bits.contains_key(&entry.name) {
                return Err(InitialLabelMapError(format!(
                    "tombstoned compartment {:?} is also live",
                    entry.name
                )));
            }
        }
        Ok(Self {
            version: file.version,
            bits,
            retired: file.retired,
        })
    }

    /// The lattice/policy version this vocabulary belongs to.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Live compartment count.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Whether the live vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// The stable bit a live compartment name is assigned.
    pub fn bit_of(&self, name: &str) -> Option<u32> {
        self.bits.get(name).copied()
    }

    /// The live name a bit resolves to (canonical rendering).
    pub fn name_of(&self, bit: u32) -> Option<&str> {
        self.bits
            .iter()
            .find(|(_, b)| **b == bit)
            .map(|(n, _)| n.as_str())
    }

    /// Tombstoned names — listed for audit output only, never assignable.
    pub fn retired_names(&self) -> Vec<&str> {
        self.retired.iter().map(|e| e.name.as_str()).collect()
    }
}

/// Parse one `$dispatchMac` annotation value against the map.
///
/// Every rule violation returns `Err` with the review-facing reason; the caller
/// turns that into a build error (`compile_error!` / build-script failure), so
/// no unparseable annotation can ever reach a runtime row.
pub fn parse_dispatch_mac(text: &str, map: &InitialLabelMap) -> Result<DispatchLabel, String> {
    let label = parse_label_text(text, map)?;
    if label.is_system_low() {
        return Err(format!(
            "'{text}' is system low — the label `$dispatchPublic` expands to. \
             System low through `$dispatchMac` is a build error; declare \
             `$dispatchPublic(\"<reason>\")` instead or choose a real label."
        ));
    }
    Ok(label)
}

/// Parse the bare label grammar (also the `$dispatchPublic` expansion target).
fn parse_label_text(text: &str, map: &InitialLabelMap) -> Result<DispatchLabel, String> {
    if text.trim() != text || text.is_empty() {
        return Err(format!(
            "label '{text}' is empty or padded — the annotation is not trimmed"
        ));
    }
    let parts: Vec<&str> = text.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return Err(format!(
            "label '{text}' must be '<level>:<assurance>[:<compartments>]' \
             ({} colon-separated components, found {})",
            if parts.len() < 2 { "too few" } else { "too many" },
            parts.len()
        ));
    }
    let level = parts[0];
    let assurance = parts[1];
    if !LEVELS.contains(&level) {
        return Err(format!(
            "label '{text}': unknown level '{level}' (one of {})",
            LEVELS.join("|")
        ));
    }
    if !ASSURANCES.contains(&assurance) {
        return Err(format!(
            "label '{text}': unknown assurance '{assurance}' (one of {})",
            ASSURANCES.join("|")
        ));
    }
    let mut compartment_bits = Vec::new();
    if parts.len() == 3 {
        if parts[2].is_empty() {
            return Err(format!(
                "label '{text}': empty compartment component — omit the third \
                 component instead of writing an empty one"
            ));
        }
        for name in parts[2].split(',') {
            if name.trim() != name || name.is_empty() {
                return Err(format!(
                    "label '{text}': compartment component '{name}' is empty or padded"
                ));
            }
            let bit = map.bit_of(name).ok_or_else(|| {
                format!(
                    "label '{text}': compartment '{name}' is not in the checked-in \
                     InitialLabelMap (v{}); adding a compartment is an append-only, \
                     reviewed edit to the map",
                    map.version()
                )
            })?;
            if let Some(prev) = compartment_bits.last() {
                if *prev == bit {
                    return Err(format!(
                        "label '{text}': compartment '{name}' appears twice"
                    ));
                }
                if *prev > bit {
                    return Err(format!(
                        "label '{text}': compartments must be listed in canonical \
                         (bit-ascending) order — '{name}' (bit {bit}) follows bit {prev}"
                    ));
                }
            }
            compartment_bits.push(bit);
        }
    }
    // Both axes were membership-checked above; the finds cannot miss, and a
    // miss is a grammar denial rather than a panic.
    let level_ref = LEVELS
        .iter()
        .find(|l| **l == level)
        .ok_or_else(|| format!("label '{text}': unknown level '{level}'"))?;
    let assurance_ref = ASSURANCES
        .iter()
        .find(|a| **a == assurance)
        .ok_or_else(|| format!("label '{text}': unknown assurance '{assurance}'"))?;
    Ok(DispatchLabel {
        level: level_ref,
        assurance: assurance_ref,
        compartment_bits,
    })
}

/// Validate one `$dispatchPublic` reason string (trimmed, nonempty).
pub fn parse_dispatch_public_reason(text: &str) -> Result<&str, String> {
    if text.trim().is_empty() {
        return Err(format!(
            "$dispatchPublic reason {text:?} is empty or whitespace-only — a \
             public leaf carries a mandatory, reviewable reason"
        ));
    }
    if text.trim() != text {
        // Permit surrounding whitespace but record it as the trimmed reason;
        // the wire value is the trimmed text.
    }
    Ok(text.trim())
}

/// The label `$dispatchPublic` expands to — always exactly system low.
pub fn public_expansion() -> DispatchLabel {
    DispatchLabel {
        level: "public",
        assurance: "unverified",
        compartment_bits: Vec::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn map_with(compartments: &[(&str, u32)]) -> InitialLabelMap {
        let entries: Vec<String> = compartments
            .iter()
            .map(|(n, b)| format!(r#"{{"name":"{n}","bit":{b}}}"#))
            .collect();
        let text = format!(
            r#"{{"version":3,"compartments":[{}],"retired":[]}}"#,
            entries.join(",")
        );
        InitialLabelMap::parse_text(&text).unwrap()
    }

    #[test]
    fn the_checked_in_map_loads() {
        let map = InitialLabelMap::load().unwrap();
        assert_eq!(map.version(), 1);
        assert!(map.is_empty(), "production lattice v1 has no compartments");
    }

    #[test]
    fn a_two_axis_label_parses() {
        let map = InitialLabelMap::load().unwrap();
        let label = parse_dispatch_mac("internal:pq-hybrid", &map).unwrap();
        assert_eq!(label.level, "internal");
        assert_eq!(label.assurance, "pq-hybrid");
        assert!(label.compartment_bits.is_empty());
        assert!(!label.is_system_low());
    }

    #[test]
    fn every_level_assurance_pair_parses_except_system_low() {
        let map = InitialLabelMap::load().unwrap();
        for level in LEVELS {
            for assurance in ASSURANCES {
                let text = format!("{level}:{assurance}");
                let parsed = parse_dispatch_mac(&text, &map);
                if text == SYSTEM_LOW {
                    assert!(parsed.is_err(), "system low via $dispatchMac denies");
                } else {
                    assert!(parsed.is_ok(), "{text} must parse");
                }
            }
        }
    }

    #[test]
    fn compartments_resolve_to_bits_in_canonical_order() {
        let map = map_with(&[("core", 2), ("pii", 0), ("finance", 1)]);
        let label = parse_dispatch_mac("secret:pq-hybrid:pii,finance,core", &map).unwrap();
        assert_eq!(label.compartment_bits, vec![0, 1, 2]);
        // Round-trip identity through the canonical renderer.
        assert_eq!(
            label.to_canonical_text(&map).unwrap(),
            "secret:pq-hybrid:pii,finance,core"
        );
    }

    #[test]
    fn noncanonical_ordering_duplicates_and_unknowns_deny() {
        let map = map_with(&[("pii", 0), ("finance", 1)]);
        // finance (bit 1) before pii (bit 0).
        assert!(parse_dispatch_mac("internal:pq-hybrid:finance,pii", &map).is_err());
        // duplicate.
        assert!(parse_dispatch_mac("internal:pq-hybrid:pii,pii", &map).is_err());
        // unknown name.
        assert!(parse_dispatch_mac("internal:pq-hybrid:pii,ghost", &map).is_err());
        // tombstoned-forever names are simply unknown.
        let retired = InitialLabelMap::parse_text(
            r#"{"version":2,"compartments":[{"name":"pii","bit":0}],"retired":[{"name":"old","bit":7}]}"#,
        )
        .unwrap();
        assert!(parse_dispatch_mac("internal:pq-hybrid:old", &retired).is_err());
        assert_eq!(retired.retired_names(), vec!["old"]);
    }

    #[test]
    fn malformed_grammar_denies() {
        let map = map_with(&[("pii", 0)]);
        for bad in [
            "",
            "internal",
            "internal:",
            ":pq-hybrid",
            "internal:pq-hybrid:",
            "internal:pq-hybrid:pii,",
            ",pii",
            "internal pq-hybrid",
            "INTERNAL:pq-hybrid",
            "internal:PQ-HYBRID",
            "internal:pq-hybrid:pii :x",
            "a:b:c:d",
            " internal:pq-hybrid",
            "internal:pq-hybrid ",
        ] {
            assert!(
                parse_dispatch_mac(bad, &map).is_err(),
                "{bad:?} must be a build error"
            );
        }
    }

    #[test]
    fn system_low_through_dispatch_mac_denies() {
        let map = InitialLabelMap::load().unwrap();
        assert!(parse_dispatch_mac("public:unverified", &map).is_err());
        // A compartmented public label is NOT system low and is fine.
        let map = map_with(&[("pii", 0)]);
        assert!(parse_dispatch_mac("public:unverified:pii", &map).is_ok());
    }

    #[test]
    fn public_reasons_must_be_trimmed_nonempty() {
        assert_eq!(parse_dispatch_public_reason("circularity").unwrap(), "circularity");
        assert_eq!(parse_dispatch_public_reason("  padded  ").unwrap(), "padded");
        for bad in ["", "   ", "\t"] {
            assert!(parse_dispatch_public_reason(bad).is_err());
        }
    }

    #[test]
    fn the_public_expansion_is_exactly_system_low() {
        let expansion = public_expansion();
        assert!(expansion.is_system_low());
        assert_eq!(expansion.level, "public");
        assert_eq!(expansion.assurance, "unverified");
        assert!(expansion.compartment_bits.is_empty());
    }

    #[test]
    fn a_malformed_map_denies_at_load() {
        // duplicate name
        assert!(InitialLabelMap::parse_text(
            r#"{"version":1,"compartments":[{"name":"a","bit":0},{"name":"a","bit":1}]}"#
        )
        .is_err());
        // duplicate bit
        assert!(InitialLabelMap::parse_text(
            r#"{"version":1,"compartments":[{"name":"a","bit":0},{"name":"b","bit":0}]}"#
        )
        .is_err());
        // bit beyond the SecurityLabel width
        assert!(InitialLabelMap::parse_text(
            r#"{"version":1,"compartments":[{"name":"a","bit":9999}]}"#
        )
        .is_err());
        // live + tombstoned at once
        assert!(InitialLabelMap::parse_text(
            r#"{"version":1,"compartments":[{"name":"a","bit":0}],"retired":[{"name":"a","bit":3}]}"#
        )
        .is_err());
        // not JSON at all
        assert!(InitialLabelMap::parse_text("not json").is_err());
    }
}
