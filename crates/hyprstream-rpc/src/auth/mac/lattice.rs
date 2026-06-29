//! The MAC **lattice** as policy — defined once, versioned, pinned.
//!
//! Two things are deliberately separated here:
//!
//! 1. **The order algebra** (dominance ⊒, join ⊔) is *fixed* and lives on
//!    [`SecurityLabel`] itself. It is the product order of `Level × Assurance ×
//!    CompartmentSet`. It never depends on site policy, so the per-op monitor
//!    (S2) can compute dominance with zero policy lookups — content truth only.
//!
//! 2. **The well-formedness policy** (which compartments exist + their stable
//!    bit assignment, the lattice *version*) lives in [`Lattice`]. This is the
//!    "defined once as policy" object (design §3). It is consulted at
//!    *label-creation / enrollment / genesis* time to (a) **intern** compartment
//!    *names* into the bit indices a [`SecurityLabel`] carries, and (b) reject
//!    typo'd / unknown compartments — NOT on the hot path. Closing the
//!    compartment vocabulary is what makes labels auditable and keeps the TCB
//!    small.
//!
//! Why split them? Because the dominance check must be uncircumventable and
//! trivially verifiable. If dominance consulted policy, a policy bug could
//! grant access. By keeping the order intrinsic and only validating
//! *membership* against policy, a policy bug can at worst reject a legitimate
//! label (fail-closed), never widen access.
//!
//! ## S1↔S4 reconciliation: the name↔bit interner lives here (#547)
//!
//! Because [`SecurityLabel`] carries a packed [`CompartmentSet`] bitset (so S4's
//! AVC can key on it directly), the *one* place that maps a compartment **name**
//! (`"pii"`) ↔ a **bit index** is this policy object. Genesis / enrollment
//! resolve names → bits via [`Lattice::intern`]; nothing on the per-op hot path
//! ever touches a string.

use super::label::{Assurance, Compartment, CompartmentSet, Level, SecurityLabel, MAX_COMPARTMENTS};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Monotonic lattice policy version. Crypto-agility / policy-agility rule
/// (design §14): the suite/lattice is *pinned and versioned*; migration is a
/// deliberate version bump, never an in-band negotiation. Labels and the
/// `(UCAN, bundle_hash)` approval (S5) bind this version so a label minted
/// under v1 can't be silently reinterpreted under v2.
///
/// **S1↔S4 reconciliation:** this is the SAME monotonic counter that S4's
/// `compiled::CompiledPolicy.generation` carries — a compiled TE matrix is valid
/// only against the lattice generation that minted its bit assignments. See
/// [`LatticeVersion::generation`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct LatticeVersion(pub u32);

impl LatticeVersion {
    /// The lattice version as the policy **generation** (S4's `CompiledPolicy`
    /// and AVC tag this same value). They are definitionally equal: a bump of
    /// the compartment vocabulary IS a new policy generation.
    #[inline]
    pub const fn generation(self) -> u64 {
        self.0 as u64
    }
}

impl fmt::Display for LatticeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Reasons a label fails the well-formedness policy. All are fail-closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelError {
    /// A compartment name in the label is not in the policy's known set.
    UnknownCompartment(Compartment),
    /// A compartment *bit* in the label has no registered name under this policy
    /// (an out-of-vocabulary or stale bit — fail-closed).
    UnknownCompartmentBit(u32),
    /// The lattice version on the label does not match the active policy.
    VersionMismatch {
        expected: LatticeVersion,
        found: LatticeVersion,
    },
}

impl fmt::Display for LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LabelError::UnknownCompartment(c) => {
                write!(f, "unknown compartment in label: {c}")
            }
            LabelError::UnknownCompartmentBit(b) => {
                write!(f, "unknown compartment bit in label: c{b}")
            }
            LabelError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "lattice version mismatch: expected {expected}, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for LabelError {}

/// The active lattice policy: the *closed* compartment vocabulary (name↔bit) and
/// the version. Levels and assurances are fixed by their enums (TCB), so only the
/// open dimension — compartments — needs a registry.
///
/// Built once at startup from the schema `$scope` annotations + site policy
/// (S3, #569) and treated as immutable thereafter (a change is a version bump).
/// Bit assignments are stable within a version (S4's compiled TE matrix is keyed
/// to them via the shared generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice {
    version: LatticeVersion,
    /// Compartment name → stable bit index. The bit index is what a
    /// [`SecurityLabel`] actually carries. A label bit with no entry here (or a
    /// name not present) is ill-formed → rejected at creation.
    name_to_bit: BTreeMap<Compartment, u32>,
    /// Reverse map for rendering / audit. Kept in sync with `name_to_bit`.
    bit_to_name: BTreeMap<u32, Compartment>,
}

impl Lattice {
    /// Construct a lattice policy from a compartment vocabulary, assigning each
    /// name a stable bit index in iteration order (0, 1, 2, …). Duplicate names
    /// collapse to the first index. Names beyond [`MAX_COMPARTMENTS`] are
    /// rejected by truncation (they simply never get a bit — fail-closed).
    pub fn new(
        version: LatticeVersion,
        known_compartments: impl IntoIterator<Item = Compartment>,
    ) -> Self {
        let mut name_to_bit = BTreeMap::new();
        let mut bit_to_name = BTreeMap::new();
        let mut next: u32 = 0;
        for c in known_compartments {
            if name_to_bit.contains_key(&c) {
                continue;
            }
            if next >= MAX_COMPARTMENTS {
                // Vocabulary exceeds the bitset width: refuse to assign (the
                // name stays unknown → labels using it fail validation). This is
                // a policy-authoring error surfaced fail-closed.
                break;
            }
            name_to_bit.insert(c.clone(), next);
            bit_to_name.insert(next, c);
            next += 1;
        }
        Lattice {
            version,
            name_to_bit,
            bit_to_name,
        }
    }

    pub fn version(&self) -> LatticeVersion {
        self.version
    }

    /// Number of compartments in the vocabulary.
    pub fn compartment_count(&self) -> u32 {
        self.name_to_bit.len() as u32
    }

    /// The bit index assigned to a compartment name, if known.
    pub fn bit_of(&self, c: &Compartment) -> Option<u32> {
        self.name_to_bit.get(c).copied()
    }

    /// The compartment name for a bit index, if registered.
    pub fn name_of(&self, bit: u32) -> Option<&Compartment> {
        self.bit_to_name.get(&bit)
    }

    /// Is `c` a recognized compartment under this policy?
    pub fn knows(&self, c: &Compartment) -> bool {
        self.name_to_bit.contains_key(c)
    }

    /// **Intern** a set of compartment names into a [`CompartmentSet`] bitset.
    ///
    /// This is the genesis/enrollment seam that turns the human-facing vocabulary
    /// into the packed representation a [`SecurityLabel`] carries. An unknown name
    /// is an error (fail-closed) — it never silently becomes "no compartment".
    pub fn intern<'a>(
        &self,
        names: impl IntoIterator<Item = &'a Compartment>,
    ) -> Result<CompartmentSet, LabelError> {
        let mut set = CompartmentSet::EMPTY;
        for name in names {
            match self.name_to_bit.get(name) {
                Some(&bit) => set = set.union(CompartmentSet::single(bit)),
                None => return Err(LabelError::UnknownCompartment(name.clone())),
            }
        }
        Ok(set)
    }

    /// Build a [`SecurityLabel`] from names, interning compartments through this
    /// policy. The ergonomic genesis constructor: `(level, assurance, names) →
    /// label`. Fails closed on any unknown compartment name.
    pub fn label(
        &self,
        level: Level,
        assurance: Assurance,
        names: impl IntoIterator<Item = Compartment>,
    ) -> Result<SecurityLabel, LabelError> {
        let owned: Vec<Compartment> = names.into_iter().collect();
        let compartments = self.intern(owned.iter())?;
        Ok(SecurityLabel::new(level, assurance, compartments))
    }

    /// Validate that a label is **well-formed** under this policy: every
    /// compartment *bit* it carries maps to a registered name. Use at genesis /
    /// enrollment / seal time before a label is allowed to exist. Does NOT touch
    /// dominance.
    pub fn validate(&self, label: &SecurityLabel) -> Result<(), LabelError> {
        for bit in label.compartments.iter() {
            if !self.bit_to_name.contains_key(&bit) {
                return Err(LabelError::UnknownCompartmentBit(bit));
            }
        }
        Ok(())
    }

    /// The lattice top ⊤ under this policy = max level, max assurance, *all*
    /// known compartments. The only clearance that dominates every well-formed
    /// label. (Useful for genesis-labeling the policy authority itself, and for
    /// tests.)
    pub fn top(&self) -> SecurityLabel {
        let all = self
            .bit_to_name
            .keys()
            .copied()
            .fold(CompartmentSet::EMPTY, |acc, b| {
                acc.union(CompartmentSet::single(b))
            });
        SecurityLabel::new(Level::TOP, Assurance::TOP, all)
    }

    /// The lattice bottom ⊥ = `(Public, Unverified, {})`. Same as
    /// [`SecurityLabel::bottom`]; provided here for symmetry with [`top`].
    ///
    /// [`top`]: Lattice::top
    pub fn bottom(&self) -> SecurityLabel {
        SecurityLabel::bottom()
    }

    /// Dominance, re-exposed at the policy object for callers that hold a
    /// `Lattice`. Identical to [`SecurityLabel::dominates`] — the policy does
    /// NOT change the order; this is purely an ergonomic forwarder. Debug
    /// builds assert both labels are well-formed (a mis-labeled object reaching
    /// the monitor is a genesis bug).
    #[must_use]
    pub fn dominates(&self, subject: &SecurityLabel, object: &SecurityLabel) -> bool {
        debug_assert!(self.validate(subject).is_ok(), "subject label ill-formed");
        debug_assert!(self.validate(object).is_ok(), "object label ill-formed");
        subject.dominates(object)
    }

    /// IFC join, re-exposed at the policy object. Identical algebra to
    /// [`SecurityLabel::join`]; the result is always well-formed if both inputs
    /// are (union of known compartments is known).
    #[must_use]
    pub fn join(&self, a: &SecurityLabel, b: &SecurityLabel) -> SecurityLabel {
        a.join(b)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn comp(s: &str) -> Compartment {
        Compartment::new(s)
    }

    fn policy() -> Lattice {
        Lattice::new(
            LatticeVersion(1),
            [comp("pii"), comp("finance"), comp("tenant:acme")],
        )
    }

    #[test]
    fn intern_assigns_stable_bits() {
        let p = policy();
        assert_eq!(p.bit_of(&comp("pii")), Some(0));
        assert_eq!(p.bit_of(&comp("finance")), Some(1));
        assert_eq!(p.bit_of(&comp("tenant:acme")), Some(2));
        assert_eq!(p.bit_of(&comp("nope")), None);
    }

    #[test]
    fn label_via_policy_interns_names() {
        let p = policy();
        let l = p
            .label(
                Level::Confidential,
                Assurance::Classical,
                [comp("pii"), comp("finance")],
            )
            .unwrap();
        assert!(l.compartments.contains(0));
        assert!(l.compartments.contains(1));
        assert!(p.validate(&l).is_ok());
    }

    #[test]
    fn label_via_policy_rejects_unknown_name() {
        let p = policy();
        assert_eq!(
            p.label(Level::Public, Assurance::Classical, [comp("typo")]),
            Err(LabelError::UnknownCompartment(comp("typo")))
        );
    }

    #[test]
    fn validate_rejects_unregistered_bit() {
        let p = policy();
        // Bit 9 has no registered name in a 3-compartment vocabulary.
        let l = SecurityLabel::from_bits(Level::Public, Assurance::Classical, [9]);
        assert_eq!(p.validate(&l), Err(LabelError::UnknownCompartmentBit(9)));
    }

    #[test]
    fn empty_compartments_always_wellformed() {
        let l = SecurityLabel::from_bits(Level::Secret, Assurance::PqHybrid, []);
        assert!(policy().validate(&l).is_ok());
    }

    #[test]
    fn top_dominates_every_wellformed_label() {
        let p = policy();
        let top = p.top();
        let arbitrary = p
            .label(
                Level::Secret,
                Assurance::PqHybrid,
                [comp("pii"), comp("finance"), comp("tenant:acme")],
            )
            .unwrap();
        assert!(p.dominates(&top, &arbitrary));
    }

    #[test]
    fn bottom_dominated_by_everything() {
        let p = policy();
        let bottom = p.bottom();
        let any = p
            .label(Level::Internal, Assurance::Classical, [comp("pii")])
            .unwrap();
        assert!(any.dominates(&bottom));
        // bottom only dominates bottom.
        assert!(!bottom.dominates(&any));
        assert!(bottom.dominates(&bottom));
    }

    #[test]
    fn lattice_forwarders_match_intrinsic_algebra() {
        let p = policy();
        let a = p
            .label(Level::Internal, Assurance::Classical, [comp("pii")])
            .unwrap();
        let b = p
            .label(Level::Secret, Assurance::PqHybrid, [comp("finance")])
            .unwrap();
        assert_eq!(p.dominates(&a, &b), a.dominates(&b));
        assert_eq!(p.join(&a, &b), a.join(&b));
    }

    #[test]
    fn version_display_and_generation() {
        assert_eq!(LatticeVersion(3).to_string(), "v3");
        // S1↔S4: LatticeVersion ≡ CompiledPolicy.generation.
        assert_eq!(LatticeVersion(3).generation(), 3u64);
    }
}
