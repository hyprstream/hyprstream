//! Synthetic workflow families.
//!
//! Synthesis families are disjoint from the benchmark's held-out gate
//! families by construction (and the [`crate::firewall`] re-checks that at
//! runtime against the consumed manifest): these are the *seen* families the
//! generalist trains on; `temporal` and `syllogism` stay firewalled for the
//! zero-shot transfer gate.

use std::fmt;

/// A synthetic workflow family: a cluster of templated (state, question)
/// shapes with paraphrase diversity. Answers come from the teacher ensemble,
/// never from a truth procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthFamily {
    /// Support-ticket triage: urgency noul, team-routing choice, severity score.
    Triage,
    /// Policy compliance: violation noul, policy-area choice, risk score.
    Compliance,
    /// Record extraction/verification: field-presence noul, document-type
    /// choice, completeness score.
    Extraction,
    /// Spend/leave approvals: limit noul, disposition choice, priority score.
    Approvals,
}

impl SynthFamily {
    /// Every synthetic family, in canonical order.
    pub const ALL: [SynthFamily; 4] = [
        SynthFamily::Triage,
        SynthFamily::Compliance,
        SynthFamily::Extraction,
        SynthFamily::Approvals,
    ];

    /// Stable family id used in item ids and corpus rows.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Triage => "triage",
            Self::Compliance => "compliance",
            Self::Extraction => "extraction",
            Self::Approvals => "approvals",
        }
    }

    /// Parse a family id.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "triage" => Some(Self::Triage),
            "compliance" => Some(Self::Compliance),
            "extraction" => Some(Self::Extraction),
            "approvals" => Some(Self::Approvals),
            _ => None,
        }
    }

    /// Canonical index (seed-mixing tag).
    pub(crate) fn tag(self) -> u64 {
        Self::ALL
            .iter()
            .position(|family| *family == self)
            .map_or(0, |position| position as u64)
    }
}

impl fmt::Display for SynthFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_ids_roundtrip() {
        for family in SynthFamily::ALL {
            assert_eq!(SynthFamily::from_id(family.as_str()), Some(family));
        }
    }

    #[test]
    fn synthesis_families_are_disjoint_from_benchmark_gate_families() {
        // The family firewall's first line of defense is structural: no
        // synthesis family may share an id with a benchmark family at all,
        // gate or open. The runtime check against the consumed manifest is
        // the second line (firewall.rs).
        for family in SynthFamily::ALL {
            assert!(hyprstream_bench::Family::from_id(family.as_str()).is_none());
        }
    }
}
