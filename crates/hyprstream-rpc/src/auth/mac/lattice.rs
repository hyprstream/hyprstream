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

/// Reasons a serialized [`Lattice`] is rejected on decode. All fail-closed: a
/// malformed wire vocabulary never silently becomes a smaller/different lattice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeDecodeError {
    /// The wire vocabulary lists the same compartment name twice. Bit
    /// assignment must be unambiguous, so this is rejected rather than
    /// silently collapsed (which would shift every later bit).
    DuplicateCompartment(Compartment),
    /// The wire vocabulary exceeds the bitset width. Truncating would drop
    /// compartments and desync bits from the issuer, so reject instead.
    TooManyCompartments { got: usize, max: u32 },
}

impl fmt::Display for LatticeDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeDecodeError::DuplicateCompartment(c) => {
                write!(f, "duplicate compartment in lattice vocabulary: {c}")
            }
            LatticeDecodeError::TooManyCompartments { got, max } => {
                write!(f, "lattice vocabulary has {got} compartments, max is {max}")
            }
        }
    }
}

impl std::error::Error for LatticeDecodeError {}

/// Failure decoding a [`Lattice`] from its canonical byte form: either the CBOR
/// itself was malformed or the decoded vocabulary was rejected (see
/// [`LatticeDecodeError`]). Carries a rendered message; the distinction is not
/// load-bearing for callers, which fail closed either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeCodecError(String);

impl fmt::Display for LatticeCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lattice decode failed: {}", self.0)
    }
}

impl std::error::Error for LatticeCodecError {}

/// The active lattice policy: the *closed* compartment vocabulary (name↔bit) and
/// the version. Levels and assurances are fixed by their enums (TCB), so only the
/// open dimension — compartments — needs a registry.
///
/// Built once at startup from the schema `$scope` annotations + site policy
/// (S3, #569) and treated as immutable thereafter (a change is a version bump).
/// Bit assignments are stable within a version (S4's compiled TE matrix is keyed
/// to them via the shared generation).
///
/// ## Canonical serialization (S1↔S4, #570)
///
/// A signed `CompiledPolicy` must carry the lattice so the PDP and every
/// distributed AVC reconstruct the **identical** name↔bit map — otherwise
/// [`CompartmentSet`] bits desync from genesis across processes. The wire form
/// is deliberately minimal: just the [`LatticeVersion`] and the compartment
/// names **in bit-index order**. Decoding re-interns those names in that exact
/// order via [`Lattice::new`], so bit assignments are *derived*, never trusted
/// from the wire — a tampered reverse map cannot exist. Use [`Lattice::to_bytes`]
/// / [`Lattice::from_bytes`] for the deterministic (CBOR) byte form, or the
/// `Serialize`/`Deserialize` impls to embed in a larger signed structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(into = "LatticeWire", try_from = "LatticeWire")]
pub struct Lattice {
    version: LatticeVersion,
    /// Compartment name → stable bit index. The bit index is what a
    /// [`SecurityLabel`] actually carries. A label bit with no entry here (or a
    /// name not present) is ill-formed → rejected at creation.
    name_to_bit: BTreeMap<Compartment, u32>,
    /// Compartments in bit-index order: `by_bit[i]` is the name interned at bit
    /// `i`. This IS the canonical construction order and the serialized form.
    by_bit: Vec<Compartment>,
}

/// The canonical, deterministic wire form of a [`Lattice`]: version + the
/// compartment vocabulary in bit-index order. The name↔bit map is reconstructed
/// (not transmitted), making issuer/consumer bit desync structurally impossible.
#[derive(Serialize, Deserialize)]
struct LatticeWire {
    version: LatticeVersion,
    compartments: Vec<Compartment>,
}

impl From<Lattice> for LatticeWire {
    fn from(l: Lattice) -> Self {
        LatticeWire {
            version: l.version,
            compartments: l.by_bit,
        }
    }
}

impl TryFrom<LatticeWire> for Lattice {
    type Error = LatticeDecodeError;

    /// Reconstruct by interning in wire order. Rejects (fail-closed) a
    /// vocabulary that [`Lattice::new`] would have silently altered — duplicate
    /// names or an over-width list — since either would desync bits from the
    /// issuer rather than reproduce them.
    fn try_from(w: LatticeWire) -> Result<Self, Self::Error> {
        if w.compartments.len() > MAX_COMPARTMENTS as usize {
            return Err(LatticeDecodeError::TooManyCompartments {
                got: w.compartments.len(),
                max: MAX_COMPARTMENTS,
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for c in &w.compartments {
            if !seen.insert(c) {
                return Err(LatticeDecodeError::DuplicateCompartment(c.clone()));
            }
        }
        Ok(Lattice::new(w.version, w.compartments))
    }
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
        let mut by_bit: Vec<Compartment> = Vec::new();
        for c in known_compartments {
            if name_to_bit.contains_key(&c) {
                continue;
            }
            if by_bit.len() >= MAX_COMPARTMENTS as usize {
                // Vocabulary exceeds the bitset width: refuse to assign (the
                // name stays unknown → labels using it fail validation). This is
                // a policy-authoring error surfaced fail-closed.
                break;
            }
            name_to_bit.insert(c.clone(), by_bit.len() as u32);
            by_bit.push(c);
        }
        Lattice {
            version,
            name_to_bit,
            by_bit,
        }
    }

    pub fn version(&self) -> LatticeVersion {
        self.version
    }

    /// Number of compartments in the vocabulary.
    pub fn compartment_count(&self) -> u32 {
        self.by_bit.len() as u32
    }

    /// The compartment vocabulary in **canonical construction (bit-index)
    /// order**: `compartment_names()[i]` is the name interned at bit `i`. This
    /// is the order [`Lattice::new`] assigned and the order the wire form
    /// preserves — S4's PDP can iterate it to align its TE matrix columns with
    /// the exact bits a [`SecurityLabel`] carries.
    pub fn compartment_names(&self) -> &[Compartment] {
        &self.by_bit
    }

    /// Canonical, deterministic byte form (CBOR) — the same codec the COSE
    /// signing path uses — so a signed `CompiledPolicy` can carry the lattice
    /// verbatim. Round-trips through [`Lattice::from_bytes`] preserving every
    /// name↔bit assignment exactly. Serialization of an in-memory lattice is
    /// infallible.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        #[allow(clippy::expect_used)]
        ciborium::ser::into_writer(self, &mut buf)
            .expect("Lattice CBOR serialization into a Vec is infallible");
        buf
    }

    /// Reconstruct a lattice from its canonical byte form. Fails closed on a
    /// malformed encoding or a vocabulary [`TryFrom`] would have to alter
    /// (duplicate names / over-width) — never silently yields a different
    /// lattice than the issuer signed.
    ///
    /// [`TryFrom`]: Lattice::try_from
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LatticeCodecError> {
        ciborium::de::from_reader(bytes).map_err(|e| LatticeCodecError(e.to_string()))
    }

    /// The bit index assigned to a compartment name, if known.
    pub fn bit_of(&self, c: &Compartment) -> Option<u32> {
        self.name_to_bit.get(c).copied()
    }

    /// The compartment name for a bit index, if registered.
    pub fn name_of(&self, bit: u32) -> Option<&Compartment> {
        self.by_bit.get(bit as usize)
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
            if (bit as usize) >= self.by_bit.len() {
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
        let all = (0..self.by_bit.len() as u32).fold(CompartmentSet::EMPTY, |acc, b| {
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

    #[test]
    fn compartment_names_are_in_construction_order() {
        let p = policy();
        let names: Vec<&str> = p.compartment_names().iter().map(Compartment::as_str).collect();
        // Exactly the order passed to `new`, i.e. bit order 0,1,2.
        assert_eq!(names, ["pii", "finance", "tenant:acme"]);
        // And it agrees with the name↔bit map for every entry.
        for (bit, name) in p.compartment_names().iter().enumerate() {
            assert_eq!(p.bit_of(name), Some(bit as u32));
            assert_eq!(p.name_of(bit as u32), Some(name));
        }
    }

    #[test]
    fn to_from_bytes_roundtrips_bits_exactly() {
        let p = policy();
        let restored = Lattice::from_bytes(&p.to_bytes()).unwrap();
        // Same version, same vocabulary, same bit assignment for every name.
        assert_eq!(restored.version(), p.version());
        assert_eq!(restored.compartment_names(), p.compartment_names());
        for name in p.compartment_names() {
            assert_eq!(restored.bit_of(name), p.bit_of(name));
        }
        // A label minted under `p` validates identically under the restored
        // lattice — the whole point: no cross-process bit desync.
        let l = p
            .label(Level::Secret, Assurance::PqHybrid, [comp("tenant:acme")])
            .unwrap();
        assert!(restored.validate(&l).is_ok());
        assert_eq!(restored.bit_of(&comp("tenant:acme")), Some(2));
    }

    #[test]
    fn byte_form_is_deterministic() {
        // Two independently-built identical lattices encode byte-identically —
        // required so a signature over the bytes is stable across processes.
        assert_eq!(policy().to_bytes(), policy().to_bytes());
    }

    #[test]
    fn serde_roundtrips_through_canonical_wire() {
        let p = policy();
        // Exercise the Serialize/Deserialize impls (the embed-in-CompiledPolicy
        // path) via CBOR, distinct from the to_bytes helper.
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&p, &mut buf).unwrap();
        let restored: Lattice = ciborium::de::from_reader(&buf[..]).unwrap();
        assert_eq!(restored.compartment_names(), p.compartment_names());
        assert_eq!(restored.version(), p.version());
    }

    #[test]
    fn decode_rejects_duplicate_compartment() {
        let wire = LatticeWire {
            version: LatticeVersion(1),
            compartments: vec![comp("pii"), comp("finance"), comp("pii")],
        };
        assert_eq!(
            Lattice::try_from(wire).unwrap_err(),
            LatticeDecodeError::DuplicateCompartment(comp("pii"))
        );
    }

    #[test]
    fn decode_rejects_over_width_vocabulary() {
        let too_many: Vec<Compartment> = (0..=MAX_COMPARTMENTS)
            .map(|i| comp(&format!("c{i}")))
            .collect();
        let got = too_many.len();
        let wire = LatticeWire {
            version: LatticeVersion(1),
            compartments: too_many,
        };
        assert_eq!(
            Lattice::try_from(wire).unwrap_err(),
            LatticeDecodeError::TooManyCompartments {
                got,
                max: MAX_COMPARTMENTS
            }
        );
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(Lattice::from_bytes(&[0xde, 0xad, 0xbe, 0xef]).is_err());
    }
}
