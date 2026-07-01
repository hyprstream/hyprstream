//! **Genesis labeling** — the initial-SID / `file_contexts` equivalent.
//!
//! Design §1 invariant 2 + the S1 acceptance criteria: *every* existing object
//! and service must carry a label **before** enforcement turns on. Without
//! genesis, enabling MAC denies everything (an unlabeled object = deny). With a
//! *permissive* genesis (label-everything-public), MAC is a no-op. So genesis
//! is the security-critical bootstrap: it must be **explicit, total, and
//! audited** — every static node mapped to a declared label, with NO catch-all
//! that silently lifts coverage.
//!
//! This module provides the genesis *mechanism* (a deterministic map from node
//! path → label, with a completeness check). The genesis *content* (the actual
//! path→label table) comes from the schema `$scope`/TE annotations (S3, #569)
//! and site policy; S1 supplies the seam + the fail-closed completeness gate.
//!
//! Key property: [`GenesisMap::resolve`] returns `None` for any path not
//! explicitly assigned. There is deliberately **no default label** — an
//! unmapped node is unlabeled, hence denied. The operator closes the gap by
//! adding an explicit assignment, never by widening a default.

use super::label::SecurityLabel;
use super::lattice::{Lattice, LabelError};
use super::manifest::StaticNodeLabel;
use std::collections::BTreeMap;

/// The result of checking genesis completeness over a known set of nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenesisReport {
    /// Nodes that have an explicit, well-formed label assignment.
    pub labeled: Vec<String>,
    /// Nodes with NO assignment — these are denied at runtime. Enforcement
    /// MUST NOT be enabled while this is non-empty (or each entry is an
    /// intentional, audited deny).
    pub unlabeled: Vec<String>,
    /// Assignments whose label is ill-formed under the active lattice (e.g.
    /// references an unknown compartment). Fail-closed: treated as gaps.
    pub ill_formed: Vec<(String, LabelError)>,
}

impl GenesisReport {
    /// Genesis is **complete** iff every known node is labeled with a
    /// well-formed label (no gaps, no ill-formed assignments). Enforcement may
    /// only be enabled when this is true. This is the gate that prevents both
    /// "deny everything" (gaps) and "trust a typo'd label".
    pub fn is_complete(&self) -> bool {
        self.unlabeled.is_empty() && self.ill_formed.is_empty()
    }
}

/// A deterministic, explicit map from a static node path to its genesis label.
///
/// Built once at startup from the schema annotations + site policy. There is no
/// wildcard fallthrough to a default label — coverage is by explicit assignment
/// only. (A site MAY add an explicit `"*"`-style rule in its *source* policy,
/// but it must materialize to concrete per-node assignments here so the
/// completeness check sees real coverage, not an implicit catch-all.)
#[derive(Debug, Clone, Default)]
pub struct GenesisMap {
    assignments: BTreeMap<String, SecurityLabel>,
}

impl GenesisMap {
    pub fn new() -> Self {
        GenesisMap::default()
    }

    /// Assign a genesis label to a node path. Returns `self` for chaining.
    pub fn assign(mut self, path: impl Into<String>, label: SecurityLabel) -> Self {
        self.assignments.insert(path.into(), label);
        self
    }

    /// The label for a node path, if explicitly assigned. `None` ⇒ unlabeled
    /// ⇒ deny. **No default.**
    pub fn resolve(&self, path: &str) -> Option<&SecurityLabel> {
        self.assignments.get(path)
    }

    /// Resolve a node path to a [`StaticNodeLabel`], validating it against the
    /// active lattice. An unassigned path or an ill-formed label both yield an
    /// *unlabeled* node (fail-closed) — the distinction is surfaced only by
    /// [`GenesisMap::report`] for operator visibility.
    pub fn resolve_node(&self, path: &str, lattice: &Lattice) -> StaticNodeLabel {
        match self.assignments.get(path) {
            Some(label) if lattice.validate(label).is_ok() => {
                StaticNodeLabel::labeled(*label)
            }
            _ => StaticNodeLabel::unlabeled(),
        }
    }

    /// Check genesis completeness over the full set of nodes that *must* be
    /// labeled (the schema's static-node inventory). Partitions every node into
    /// labeled / unlabeled / ill-formed. This is the audited bootstrap gate.
    pub fn report<'a>(
        &self,
        all_nodes: impl IntoIterator<Item = &'a str>,
        lattice: &Lattice,
    ) -> GenesisReport {
        let mut labeled = Vec::new();
        let mut unlabeled = Vec::new();
        let mut ill_formed = Vec::new();

        for node in all_nodes {
            match self.assignments.get(node) {
                None => unlabeled.push(node.to_owned()),
                Some(label) => match lattice.validate(label) {
                    Ok(()) => labeled.push(node.to_owned()),
                    Err(e) => ill_formed.push((node.to_owned(), e)),
                },
            }
        }
        GenesisReport {
            labeled,
            unlabeled,
            ill_formed,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::label::{Assurance, Compartment, CompartmentSet, Level};
    use super::super::lattice::LatticeVersion;
    use super::super::manifest::LabeledObject;
    use super::*;

    fn comp(s: &str) -> Compartment {
        Compartment::new(s)
    }

    fn lattice() -> Lattice {
        // 2-compartment vocabulary → bits 0 ("pii") and 1 ("health").
        Lattice::new(LatticeVersion(1), [comp("pii"), comp("health")])
    }

    fn map() -> GenesisMap {
        let l = lattice();
        GenesisMap::new()
            .assign(
                "/health",
                SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY),
            )
            .assign(
                "/ctl/policy",
                l.label(Level::Secret, Assurance::PqHybrid, [comp("pii")]).unwrap(),
            )
    }

    #[test]
    fn resolve_returns_assigned_label_only() {
        let m = map();
        assert!(m.resolve("/health").is_some());
        assert!(m.resolve("/unknown").is_none()); // no default → deny
    }

    #[test]
    fn resolve_node_unassigned_is_unlabeled() {
        let node = map().resolve_node("/nope", &lattice());
        assert!(node.security_label().is_none());
    }

    #[test]
    fn resolve_node_assigned_is_labeled() {
        let node = map().resolve_node("/health", &lattice());
        assert!(node.security_label().is_some());
    }

    #[test]
    fn report_detects_gaps() {
        let report = map().report(["/health", "/ctl/policy", "/streams", "/metrics"], &lattice());
        assert_eq!(report.labeled.len(), 2);
        // `report` preserves input order of `all_nodes`.
        assert_eq!(report.unlabeled, vec!["/streams".to_owned(), "/metrics".to_owned()]);
        assert!(!report.is_complete());
    }

    #[test]
    fn report_complete_when_all_labeled() {
        let report = map().report(["/health", "/ctl/policy"], &lattice());
        assert!(report.is_complete());
        assert!(report.unlabeled.is_empty());
        assert!(report.ill_formed.is_empty());
    }

    #[test]
    fn report_flags_ill_formed_label_as_not_complete() {
        // a label naming an unknown compartment is ill-formed → fail-closed.
        // Bit 9 is not registered in the 2-compartment vocabulary → ill-formed.
        let m = GenesisMap::new().assign(
            "/bad",
            SecurityLabel::from_bits(Level::Public, Assurance::Classical, [9]),
        );
        let report = m.report(["/bad"], &lattice());
        assert!(report.labeled.is_empty());
        assert_eq!(report.ill_formed.len(), 1);
        assert!(!report.is_complete());
    }

    #[test]
    fn ill_formed_label_resolves_as_unlabeled() {
        // Bit 9 is not registered in the 2-compartment vocabulary → ill-formed.
        let m = GenesisMap::new().assign(
            "/bad",
            SecurityLabel::from_bits(Level::Public, Assurance::Classical, [9]),
        );
        // resolve_node fails closed: ill-formed → unlabeled → deny.
        assert!(m.resolve_node("/bad", &lattice()).security_label().is_none());
    }
}
