//! Question families, holdout designations, and difficulty strata.
//!
//! The family taxonomy is the unit of the **family-level holdout firewall**
//! (plan v1.6): families designated [`Designation::Gate`] are *never
//! synthesized into training data* — P1.3's synthesis run consumes the frozen
//! manifest and must exclude them, and the P1.4 leg-(e) zero-shot transfer
//! gate measures the model on exactly these families. Designation is decided
//! NOW, at benchmark freeze, not after seeing results.

use std::fmt;

/// A question family: a cluster of procedurally generated, mechanically
/// verifiable question shapes sharing one truth procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// Arithmetic word problems (add/subtract/multi-step counts, unit prices).
    Arith,
    /// JSON well-formedness, field extraction, and repair identification.
    JsonCheck,
    /// Unit conversion (length, mass, temperature) with numeric outcomes.
    UnitConv,
    /// Date arithmetic and event ordering. **Held out (gate).**
    Temporal,
    /// Set-logic syllogisms (All/Some/None chains). **Held out (gate).**
    Syllogism,
}

/// Whether a family may be synthesized into training data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Designation {
    /// Usable for training synthesis and evaluation.
    Open,
    /// **Firewalled**: never synthesized into training data; reserved for the
    /// zero-shot transfer gate (P1.4 leg e) and public reporting.
    Gate,
}

impl Designation {
    /// Wire form (`"open"` / `"gate"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Gate => "gate",
        }
    }
}

impl fmt::Display for Designation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Family {
    /// Every family, in canonical (manifest) order.
    pub const ALL: [Family; 5] = [
        Family::Arith,
        Family::JsonCheck,
        Family::UnitConv,
        Family::Temporal,
        Family::Syllogism,
    ];

    /// Stable family id used in item ids and the manifest.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Arith => "arith",
            Self::JsonCheck => "jsoncheck",
            Self::UnitConv => "unitconv",
            Self::Temporal => "temporal",
            Self::Syllogism => "syllogism",
        }
    }

    /// Parse a family id.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "arith" => Some(Self::Arith),
            "jsoncheck" => Some(Self::JsonCheck),
            "unitconv" => Some(Self::UnitConv),
            "temporal" => Some(Self::Temporal),
            "syllogism" => Some(Self::Syllogism),
            _ => None,
        }
    }

    /// The firewall designation. **Frozen at the vob-1.0 manifest freeze**;
    /// P1.3 treats the manifest, not this code, as the consumed artifact.
    pub fn designation(self) -> Designation {
        match self {
            Self::Arith | Self::JsonCheck | Self::UnitConv => Designation::Open,
            Self::Temporal | Self::Syllogism => Designation::Gate,
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The difficulty/robustness stratum an item belongs to. Reported per stratum
/// by the P0.2 measurement protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stratum {
    /// Base form of the item.
    Clean,
    /// Adversarial near-miss form: near-miss arithmetic (±1, ×10 slips,
    /// transposed digits as distractors/claims), near-miss JSON (trailing
    /// commas, unquoted keys, single quotes), close distractors.
    NearMiss,
    /// Cyclic-permutation stratum (CircularEval precedent, S6b1): the same
    /// choice item with options rotated by `rotation` positions; truth rotates
    /// with the options. Applied to **choice** items only — score levels are
    /// ordered (index = level is the semantics), so permuting them would
    /// change the question, not the presentation. Noul has no options.
    Permuted {
        /// Cyclic rotation applied, in `1..cardinality`.
        rotation: u32,
    },
}

impl Stratum {
    /// Wire form (`clean`, `nearmiss`, `perm-2`, ...).
    pub fn as_str(&self) -> String {
        match self {
            Self::Clean => "clean".to_owned(),
            Self::NearMiss => "nearmiss".to_owned(),
            Self::Permuted { rotation } => format!("perm-{rotation}"),
        }
    }

    /// Whether this stratum is a base form (not a permutation expansion).
    pub fn is_base(&self) -> bool {
        !matches!(self, Self::Permuted { .. })
    }
}

impl fmt::Display for Stratum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn designations_roundtrip_and_gate_set_is_frozen() {
        // The gate set is the P1.4 leg-(e) measurement surface; changing it
        // after freeze invalidates the firewall. Lock it here.
        let gates: Vec<Family> = Family::ALL
            .iter()
            .copied()
            .filter(|family| family.designation() == Designation::Gate)
            .collect();
        assert_eq!(gates, vec![Family::Temporal, Family::Syllogism]);
    }

    #[test]
    fn family_ids_roundtrip() {
        for family in Family::ALL {
            assert_eq!(Family::from_id(family.as_str()), Some(family));
        }
    }
}
