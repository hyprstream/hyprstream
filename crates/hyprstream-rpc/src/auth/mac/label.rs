//! The security **label** — the content-bound classification carried by every
//! object, and the verified **clearance** carried by every subject.
//!
//! This is S1 (#567), the foundation of the native-MAC authz model (epic #547).
//! It is **TCB**: keep it small and verifiable. The whole lattice is three
//! orthogonal axes, each a *small total/partial order*, combined into one
//! product lattice (see [`super::lattice`]).
//!
//! Design invariants (epic #547 §1, design doc §1):
//! - **No unlabeled state.** There is no `Default` that means "public/allow".
//!   The absence of a label is represented by `Option::None` at call sites and
//!   is treated as **denied** by the monitor — never by a permissive default
//!   here. (We deliberately do NOT derive `Default` for [`SecurityLabel`].)
//! - **Labels are content-truth.** A label is meant to be bound to content (a
//!   manifest hash) so it cannot be relabeled without rehashing. This module
//!   defines the *type*; the binding seam lives in [`super::manifest`].
//! - **Assurance is a lattice axis, not a flag** (#548). A classical-DID peer
//!   is bounded LOW; a PQC-bound peer may reach HIGH. The dominance check then
//!   *automatically* prevents PQC-required data from flowing to a classical
//!   peer — no separate enforcement path.
//!
//! ## Representation (S1↔S4 reconciliation, epic #547)
//!
//! [`SecurityLabel`] is a **fixed-size, `Copy + Ord + Hash` value**:
//! ```text
//!     level: Level (u8)  ×  assurance: Assurance (u8)  ×  compartments: CompartmentSet (u64 bitset)
//! ```
//! Compartments are a **fixed bitset**, not a heap set. Each compartment name
//! (`"pii"`, `"tenant:acme"`, …) is interned to a bit index by the policy
//! ([`super::lattice::Lattice`]); the label carries only the bitset. This is what
//! lets S4's Access Vector Cache (#570) key on a [`SecurityLabel`] *directly* —
//! no interning layer, no allocation, sub-µs hashing on the hot path. The
//! string↔bit mapping lives in the `Lattice` policy (the one place the
//! compartment vocabulary is closed and audited).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Number of distinct compartments representable in a label.
///
/// A fixed bitset width keeps [`SecurityLabel`] `Copy` and pointer-free so it can
/// be an AVC key (S4, #570). 64 is ample headroom for the closed, audited
/// compartment vocabulary; widening it is a deliberate (versioned) TCB change,
/// the same class of change as adding a [`Level`].
pub const MAX_COMPARTMENTS: u32 = 64;

/// Multi-Level-Security sensitivity level — a **total order** (the MLS axis).
///
/// Small and fixed on purpose (TCB). Higher discriminant ⇒ more sensitive.
/// Ordering is derived from declaration order, so `Public < Internal <
/// Confidential < Secret`. This is the classic Bell–LaPadula level.
///
/// Adding a level is a deliberate policy change (a new lattice version), not a
/// runtime/config knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Level {
    /// World-readable. The *floor* of the level axis. NOTE: floor ≠ unlabeled.
    /// An object explicitly labeled `Public` is labeled; an object with no
    /// label at all is denied (see module docs).
    Public = 0,
    /// Internal / org-only.
    Internal = 1,
    /// Confidential.
    Confidential = 2,
    /// Secret — the top of the level axis.
    Secret = 3,
}

impl Level {
    /// The bottom of the level axis (used as the join identity element).
    pub const BOTTOM: Level = Level::Public;
    /// The top of the level axis.
    pub const TOP: Level = Level::Secret;
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Level::Public => "public",
            Level::Internal => "internal",
            Level::Confidential => "confidential",
            Level::Secret => "secret",
        };
        f.write_str(s)
    }
}

/// Crypto-**assurance** axis (#548) — a **total order** derived from *verified*
/// key material, never claimed.
///
/// This is the dimension that lets a classical (ATProto/p256/secp256k1)
/// federation perimeter coexist with the mandatory PQC interior without a hole:
/// PQC-required data is labeled at an assurance only a PQC-bound principal
/// dominates, and the ordinary dominance check does the rest.
///
/// Derivation (a property of the *verified* identity — see
/// [`super::context::SecurityContext::derive_assurance`]):
/// - `Unverified` — assurance could not be established (fail-closed; the floor).
/// - `Classical` — authenticated by a classical DID only (Ed25519/p256/
///   secp256k1); no bound PQC anchor.
/// - `PqHybrid`  — has a verified hybrid-PQC anchor (Ed25519 ↔ ML-DSA-65 bound
///   via `register_pq_trust`), i.e. the [`crate::crypto::CryptoPolicy::Hybrid`]
///   composite verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum Assurance {
    /// Assurance could not be verified. Floor of the axis. A *subject* at
    /// `Unverified` dominates nothing above it; an *object* requiring more than
    /// `Unverified` is therefore unreachable by such a subject — fail-closed.
    Unverified = 0,
    /// Classical DID, no post-quantum binding (the federation edge).
    Classical = 1,
    /// Verified hybrid post-quantum anchor (EdDSA + ML-DSA-65).
    PqHybrid = 2,
}

impl Assurance {
    /// Floor of the assurance axis (fail-closed for unknown provenance).
    pub const BOTTOM: Assurance = Assurance::Unverified;
    /// Top of the assurance axis and identity for its `min` join.
    pub const TOP: Assurance = Assurance::PqHybrid;
}

impl fmt::Display for Assurance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Assurance::Unverified => "unverified",
            Assurance::Classical => "classical",
            Assurance::PqHybrid => "pq-hybrid",
        };
        f.write_str(s)
    }
}

/// A need-to-know **compartment** *name* (the categories axis vocabulary).
///
/// Compartments form a powerset lattice under ⊆. This type is the *human-facing
/// name* (e.g. `"tenant:acme"`, `"pii"`, `"model:qwen-7b"`); inside a
/// [`SecurityLabel`] a compartment is stored as a **bit** in a [`CompartmentSet`]
/// bitset. The name↔bit mapping is owned by the policy
/// ([`super::lattice::Lattice`], built from the schema `$scope` annotations —
/// S3, #569). This type does not constrain the vocabulary.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Compartment(pub String);

impl Compartment {
    pub fn new(s: impl Into<String>) -> Self {
        Compartment(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Compartment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Compartment({:?})", self.0)
    }
}

impl fmt::Display for Compartment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A **fixed bitset** of compartment bits — the powerset-lattice axis, packed so
/// [`SecurityLabel`] stays `Copy + Ord + Hash`.
///
/// Bit `i` set ⟺ the label is in the compartment interned at index `i` by the
/// policy. Subset/union are bitwise `AND`/`OR`. The empty set (`0`) is the
/// compartment-axis bottom. This is the representation that lets S4's AVC (#570)
/// key on a label with no interning indirection.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompartmentSet(pub u64);

impl CompartmentSet {
    /// The empty compartment set — the axis bottom.
    pub const EMPTY: CompartmentSet = CompartmentSet(0);

    /// A set containing exactly the single bit `index` (`0 ≤ index < `
    /// [`MAX_COMPARTMENTS`]). Out-of-range indices are ignored (empty) —
    /// fail-closed: an un-representable compartment grants nothing.
    #[inline]
    pub const fn single(index: u32) -> Self {
        if index >= MAX_COMPARTMENTS {
            CompartmentSet(0)
        } else {
            CompartmentSet(1u64 << index)
        }
    }

    /// Is this the empty set?
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `self ⊆ other` — is every compartment of `self` also in `other`?
    #[inline]
    pub const fn is_subset(self, other: CompartmentSet) -> bool {
        self.0 & other.0 == self.0
    }

    /// Set **union** (the IFC join on this axis).
    #[inline]
    pub const fn union(self, other: CompartmentSet) -> CompartmentSet {
        CompartmentSet(self.0 | other.0)
    }

    /// Set **intersection** (the lattice *meet* on this axis) — the dual of
    /// [`Self::union`]. Used to compose the need-to-know of two principals
    /// (#681 delegated meet): the result is cleared into a compartment only if
    /// *both* inputs are, so a delegated call never widens need-to-know past
    /// either principal.
    #[inline]
    pub const fn intersect(self, other: CompartmentSet) -> CompartmentSet {
        CompartmentSet(self.0 & other.0)
    }

    /// Does this set contain compartment bit `index`?
    #[inline]
    pub const fn contains(self, index: u32) -> bool {
        index < MAX_COMPARTMENTS && (self.0 & (1u64 << index)) != 0
    }

    /// Iterate the set bit indices, ascending. (Off the hot path: diagnostics /
    /// rendering / policy validation.)
    pub fn iter(self) -> impl Iterator<Item = u32> {
        (0..MAX_COMPARTMENTS).filter(move |&i| self.contains(i))
    }

    /// Number of compartments in the set.
    #[inline]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }
}

impl FromIterator<u32> for CompartmentSet {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        iter.into_iter().fold(CompartmentSet::EMPTY, |acc, i| {
            acc.union(CompartmentSet::single(i))
        })
    }
}

impl fmt::Debug for CompartmentSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompartmentSet({:#x})", self.0)
    }
}

/// A complete security label: a point in the product lattice
/// `Level × Assurance × CompartmentSet`.
///
/// The same type is used for an **object's** classification and a **subject's**
/// clearance (a clearance is just a label the subject is allowed to read up
/// to). Dominance and join are defined on the inherent
/// [`SecurityLabel::can_access`] / [`SecurityLabel::join`] helpers below — note
/// those helpers encode the *fixed* product-lattice algebra and are independent
/// of any site policy (the policy in [`super::lattice::Lattice`] only constrains
/// which labels are *well-formed/known*, never how they compare).
///
/// **`Copy + Ord + Hash`** (compartments are a packed [`CompartmentSet`] bitset,
/// not a heap set) so S4's AVC (#570) can key on a label directly — and so two
/// equal labels serialize identically, which content-binding (manifest hash) and
/// the `(UCAN, bundle_hash)` approval (S5) require.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SecurityLabel {
    /// MLS sensitivity level (total order).
    pub level: Level,
    /// Crypto-assurance required to handle this label (total order, #548).
    pub assurance: Assurance,
    /// Need-to-know compartments as a fixed bitset (powerset lattice under ⊆).
    #[serde(default)]
    pub compartments: CompartmentSet,
}

impl fmt::Debug for SecurityLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Render in the SELinux-ish `level:assurance:{bits}` form for logs/audit.
        write!(f, "{self}")
    }
}

impl fmt::Display for SecurityLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Note: the label carries compartment *bits*, not names — name resolution
        // needs the policy. We render bit indices here; a name-aware renderer
        // lives on `Lattice`.
        write!(f, "{}:{}", self.level, self.assurance)?;
        if !self.compartments.is_empty() {
            let joined = self
                .compartments
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            write!(f, ":{{c{joined}}}")?;
        }
        Ok(())
    }
}

/// The **credential-clearance wire projection** of a [`SecurityLabel`] (v16
/// §11, operator Gate-2 value 11). A credential's `clearance` claim is the
/// semantic two-element array `[level, [compartments]]` and NOTHING else — the
/// assurance axis is **structurally absent from the wire** and is derived only
/// from verified key material at credential admission (see
/// [`super::SecurityContext::new`]). The authority-asserted clearance claim
/// therefore carries THIS type, never a full [`SecurityLabel`], so a synthetic
/// assurance can never ride in on the wire.
///
/// Wire grammar (identical for the JWT JSON value and the CWT CBOR value):
/// - element 0 `level`: a uint with the frozen mapping `0=Public, 1=Internal,
///   2=Confidential, 3=Secret`;
/// - element 1 `compartments`: an array of compartment **bit indices**, each a
///   uint `0..=63`, strictly ascending and unique (empty allowed) — the
///   credential projection of the versioned compartment vocabulary, never a
///   bitmask integer and never names;
/// - the outer array has **exactly two** elements.
///
/// Unknown levels, out-of-range/duplicate/descending compartments, a bitmask
/// integer, names, extra elements, or any assurance field all fail
/// deserialization (fail-closed).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CredentialClearance {
    /// MLS sensitivity level (element 0 of the wire array).
    pub level: Level,
    /// Need-to-know compartments (bitset; the wire form is the ascending index
    /// list of element 1).
    pub compartments: CompartmentSet,
}

impl CredentialClearance {
    /// The wire projection of a full label: level + compartments, dropping the
    /// crypto-derived assurance axis (which is never issuer-asserted).
    pub fn from_label(label: SecurityLabel) -> Self {
        Self {
            level: label.level,
            compartments: label.compartments,
        }
    }
}

impl From<SecurityLabel> for CredentialClearance {
    fn from(label: SecurityLabel) -> Self {
        Self::from_label(label)
    }
}

impl fmt::Debug for CredentialClearance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Assurance is deliberately absent — this is the wire projection.
        write!(f, "{}", self.level)?;
        if !self.compartments.is_empty() {
            let joined = self
                .compartments
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            write!(f, ":{{c{joined}}}")?;
        }
        Ok(())
    }
}

impl serde::Serialize for CredentialClearance {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq as _;
        let indices: Vec<u8> = self.compartments.iter().map(|i| i as u8).collect();
        let mut seq = serializer.serialize_seq(Some(2))?;
        seq.serialize_element(&(self.level as u8))?;
        seq.serialize_element(&indices)?;
        seq.end()
    }
}

impl<'de> serde::Deserialize<'de> for CredentialClearance {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WireVisitor;
        impl<'de> serde::de::Visitor<'de> for WireVisitor {
            type Value = CredentialClearance;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "a two-element credential-clearance array [level, [compartment indices]]",
                )
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<CredentialClearance, A::Error> {
                use serde::de::Error as _;
                let level_raw: u8 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::custom("clearance: missing level element"))?;
                let level = match level_raw {
                    0 => Level::Public,
                    1 => Level::Internal,
                    2 => Level::Confidential,
                    3 => Level::Secret,
                    other => {
                        return Err(A::Error::custom(format!(
                            "clearance: unknown level {other} (0..=3)"
                        )));
                    }
                };
                let indices: Vec<u8> = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::custom("clearance: missing compartments array"))?;
                // Exactly two elements: any third element (e.g. an appended
                // assurance) is a malformed, assurance-bearing wire value.
                if seq.next_element::<serde::de::IgnoredAny>()?.is_some() {
                    return Err(A::Error::custom(
                        "clearance: array must have exactly two elements",
                    ));
                }
                let mut bits: u64 = 0;
                let mut prev: Option<u8> = None;
                for idx in indices {
                    if u32::from(idx) >= MAX_COMPARTMENTS {
                        return Err(A::Error::custom(format!(
                            "clearance: compartment index {idx} out of range 0..={}",
                            MAX_COMPARTMENTS - 1
                        )));
                    }
                    if prev.is_some_and(|p| idx <= p) {
                        return Err(A::Error::custom(
                            "clearance: compartment indices must be strictly ascending and unique",
                        ));
                    }
                    prev = Some(idx);
                    bits |= 1u64 << idx;
                }
                Ok(CredentialClearance {
                    level,
                    compartments: CompartmentSet(bits),
                })
            }
        }
        deserializer.deserialize_seq(WireVisitor)
    }
}

impl SecurityLabel {
    /// Construct a label from its axes. There is intentionally **no `Default`** —
    /// every label must be chosen explicitly so "unlabeled" can never silently
    /// mean "public".
    pub fn new(level: Level, assurance: Assurance, compartments: CompartmentSet) -> Self {
        SecurityLabel {
            level,
            assurance,
            compartments,
        }
    }

    /// Construct a label from compartment **bit indices** (e.g. resolved from
    /// names by the policy). Out-of-range indices are dropped (fail-closed).
    pub fn from_bits(
        level: Level,
        assurance: Assurance,
        bits: impl IntoIterator<Item = u32>,
    ) -> Self {
        SecurityLabel {
            level,
            assurance,
            compartments: bits.into_iter().collect(),
        }
    }

    /// The fail-closed label `(Public, Unverified, {})`.
    ///
    /// This is deliberately distinct from the identity for provenance joins:
    /// the Biba assurance axis joins with `min`, whose identity is `PqHybrid`.
    /// In particular, an empty provenance set must remain unverified rather
    /// than acquire the join identity's top assurance.
    pub const fn bottom() -> Self {
        SecurityLabel {
            level: Level::BOTTOM,
            assurance: Assurance::BOTTOM,
            compartments: CompartmentSet::EMPTY,
        }
    }

    /// Per-axis identity for a non-empty provenance join.
    const fn join_identity() -> Self {
        SecurityLabel {
            level: Level::BOTTOM,
            assurance: Assurance::TOP,
            compartments: CompartmentSet::EMPTY,
        }
    }

    /// Whether this label is the lattice bottom.
    pub fn is_bottom(&self) -> bool {
        self.level == Level::BOTTOM
            && self.assurance == Assurance::BOTTOM
            && self.compartments.is_empty()
    }

    /// May a subject with this clearance access an object labelled `object`?
    ///
    /// This is the **Bell–LaPadula dominance** relation — clearance `self`
    /// *dominates* the object label (`self ⊒ object`) iff, on *every* axis, the
    /// subject is at least the object:
    /// - `self.level     >= object.level`            (no read-up)
    /// - `self.assurance >= object.assurance`        (#548 crypto-assurance axis)
    /// - `object.compartments ⊆ self.compartments`   (cleared into all categories)
    ///
    /// The assurance field is currently overloaded: here it is an authentication-
    /// strength gate, while provenance joining treats it as Biba integrity. Keep
    /// this gate unchanged pending the field-separation/proof follow-up in #1232.
    ///
    /// The per-op MAC floor (design §3, §10): the *fixed* product-lattice order,
    /// content truth, never overridable by a token/UCAN/grant.
    #[inline]
    #[must_use]
    pub fn can_access(&self, object: &SecurityLabel) -> bool {
        self.level >= object.level
            && self.assurance >= object.assurance
            && object.compartments.is_subset(self.compartments)
    }

    /// Bell–LaPadula *-property (confinement / no-write-down): may a subject
    /// with clearance `self` **write** to an object labelled `object`?
    ///
    /// The write-direction dual of [`can_access`]. A subject may write only to
    /// objects at least as restrictive as its own clearance — preventing a
    /// high-level subject from leaking secrets to a lower-level object:
    /// - `object.level >= self.level`              (no-write-down)
    /// - `self.assurance >= object.assurance`       (crypto floor is NOT flipped —
    ///   you cannot write to a higher-assurance object than your key proves)
    /// - `self.compartments.is_subset(object.compartments)` (no compartment
    ///   write-down: the target must be at least as compartmented as the writer)
    ///
    /// The assurance axis intentionally does **not** flip: it is a
    /// crypto-authentication floor on both read and write, not a confidentiality
    /// axis. This is conservative — it tightens rather than relaxes the gate.
    /// See the assurance-axis overload note on [`can_access`] and #1232.
    ///
    /// [`can_access`]: SecurityLabel::can_access
    #[inline]
    #[must_use]
    pub fn can_write_to(&self, object: &SecurityLabel) -> bool {
        object.level >= self.level
            && self.assurance >= object.assurance
            && self.compartments.is_subset(object.compartments)
    }

    /// IFC provenance join over confidentiality, integrity, and compartments.
    ///
    /// `a ⊔ b` = `(max(level), min(assurance), union(compartments))`: secrecy
    /// rises to the most restrictive input, while provenance integrity degrades
    /// to its weakest link. Associative, commutative, and idempotent.
    #[inline]
    #[must_use]
    pub fn join(&self, other: &SecurityLabel) -> SecurityLabel {
        SecurityLabel {
            level: if self.level >= other.level {
                self.level
            } else {
                other.level
            },
            assurance: if self.assurance <= other.assurance {
                self.assurance
            } else {
                other.assurance
            },
            compartments: self.compartments.union(other.compartments),
        }
    }

    /// Fold the IFC join over a set of input labels (provenance-DAG join).
    ///
    /// Non-empty inputs fold from the per-axis identity `(Public, PqHybrid, {})`.
    /// Empty provenance instead returns the fail-closed `bottom()` label, never
    /// the top-assurance identity.
    #[must_use]
    pub fn join_all<'a>(labels: impl IntoIterator<Item = &'a SecurityLabel>) -> SecurityLabel {
        let mut labels = labels.into_iter();
        let Some(first) = labels.next() else {
            return SecurityLabel::bottom();
        };

        labels.fold(SecurityLabel::join_identity().join(first), |acc, label| {
            acc.join(label)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn label(l: Level, a: Assurance, bits: &[u32]) -> SecurityLabel {
        SecurityLabel::from_bits(l, a, bits.iter().copied())
    }

    #[test]
    fn level_is_totally_ordered() {
        assert!(Level::Public < Level::Internal);
        assert!(Level::Internal < Level::Confidential);
        assert!(Level::Confidential < Level::Secret);
        assert_eq!(Level::BOTTOM, Level::Public);
        assert_eq!(Level::TOP, Level::Secret);
    }

    #[test]
    fn assurance_is_totally_ordered_unverified_floor() {
        assert!(Assurance::Unverified < Assurance::Classical);
        assert!(Assurance::Classical < Assurance::PqHybrid);
        assert_eq!(Assurance::BOTTOM, Assurance::Unverified);
    }

    #[test]
    fn compartment_set_subset_and_union() {
        let a = CompartmentSet::single(0).union(CompartmentSet::single(2));
        let b = CompartmentSet::single(0);
        assert!(b.is_subset(a));
        assert!(!a.is_subset(b));
        assert_eq!(a.union(b), a);
        assert_eq!(a.len(), 2);
        assert!(a.contains(0));
        assert!(a.contains(2));
        assert!(!a.contains(1));
    }

    #[test]
    fn label_is_copy_and_hashable() {
        // The reconciliation requirement: a label is a cheap Copy+Hash+Ord AVC key.
        fn assert_avc_key<T: Copy + std::hash::Hash + Ord>() {}
        assert_avc_key::<SecurityLabel>();
        let x = label(Level::Secret, Assurance::PqHybrid, &[1, 3]);
        let y = x; // Copy, not move
        assert_eq!(x, y);
    }

    #[test]
    fn dominance_reflexive() {
        let x = label(Level::Confidential, Assurance::Classical, &[0, 1]);
        assert!(x.can_access(&x));
    }

    #[test]
    fn dominance_level_read_down() {
        let secret = label(Level::Secret, Assurance::PqHybrid, &[]);
        let public = label(Level::Public, Assurance::PqHybrid, &[]);
        assert!(secret.can_access(&public));
        assert!(!public.can_access(&secret));
    }

    #[test]
    fn dominance_requires_compartment_superset() {
        let cleared = label(Level::Secret, Assurance::PqHybrid, &[0, 1]); // pii, tenant:acme
        let needs_pii = label(Level::Public, Assurance::Unverified, &[0]);
        let needs_other = label(Level::Public, Assurance::Unverified, &[5]); // finance
        assert!(cleared.can_access(&needs_pii));
        assert!(!cleared.can_access(&needs_other)); // not cleared into "finance"
    }

    #[test]
    fn assurance_classical_cannot_dominate_pqc_label() {
        // #548 acceptance: a classical-DID peer cannot dominate a PQC-required label.
        let classical_subject = label(Level::Secret, Assurance::Classical, &[0]);
        let pqc_object = label(Level::Public, Assurance::PqHybrid, &[]);
        assert!(!classical_subject.can_access(&pqc_object));

        // a PQC-bound peer can (level/compartments permitting).
        let pqc_subject = label(Level::Secret, Assurance::PqHybrid, &[0]);
        assert!(pqc_subject.can_access(&pqc_object));
    }

    #[test]
    fn unverified_subject_dominates_nothing_above_floor() {
        let unverified = label(Level::Secret, Assurance::Unverified, &[0]);
        let needs_classical = label(Level::Public, Assurance::Classical, &[]);
        assert!(!unverified.can_access(&needs_classical));
    }

    #[test]
    fn write_direction_no_write_down() {
        // A Secret subject may write to a Secret object (same level).
        let secret = label(Level::Secret, Assurance::Classical, &[]);
        assert!(secret.can_write_to(&secret));

        // A Secret subject may NOT write to a Public object (no-write-down).
        let public = label(Level::Public, Assurance::Classical, &[]);
        assert!(!secret.can_write_to(&public));

        // A Public subject MAY write to a Secret object (writing up is allowed).
        assert!(public.can_write_to(&secret));
    }

    #[test]
    fn write_direction_compartments_no_write_out() {
        // A subject in {0,1} may write to an object in {0,1} or a superset.
        let subj = label(Level::Secret, Assurance::Classical, &[0, 1]);
        let same = label(Level::Secret, Assurance::Classical, &[0, 1]);
        let superset = label(Level::Secret, Assurance::Classical, &[0, 1, 2]);
        assert!(subj.can_write_to(&same));
        assert!(subj.can_write_to(&superset));

        // Writing to a subset (e.g. {0}) leaks compartment-1 data — denied.
        let subset = label(Level::Secret, Assurance::Classical, &[0]);
        assert!(!subj.can_write_to(&subset));
    }

    #[test]
    fn write_direction_assurance_floor_not_flipped() {
        // Assurance is a crypto floor on both axes: a Classical subject cannot
        // write to a PqHybrid-required object even though the *-property level
        // direction is satisfied (Public→Public or Public→Secret is "write up").
        let classical_subj = label(Level::Public, Assurance::Classical, &[]);
        let pqc_object = label(Level::Secret, Assurance::PqHybrid, &[]);
        assert!(!classical_subj.can_write_to(&pqc_object));

        // A PqHybrid subject writing to a Classical object at ≥ level: allowed.
        let pqc_subj = label(Level::Public, Assurance::PqHybrid, &[]);
        let classical_obj = label(Level::Secret, Assurance::Classical, &[]);
        assert!(pqc_subj.can_write_to(&classical_obj));
    }

    #[test]
    fn join_takes_confidentiality_max_integrity_min_and_union() {
        let a = label(Level::Internal, Assurance::Classical, &[0]); // pii
        let b = label(Level::Secret, Assurance::PqHybrid, &[5]); // finance
        let j = a.join(&b);
        assert_eq!(j.level, Level::Secret);
        assert_eq!(j.assurance, Assurance::Classical);
        assert!(j.compartments.contains(0));
        assert!(j.compartments.contains(5));
    }

    #[test]
    fn join_preserves_each_axis_bound() {
        let a = label(Level::Internal, Assurance::Classical, &[0]);
        let b = label(Level::Confidential, Assurance::Unverified, &[5]);
        let j = a.join(&b);
        assert!(j.level >= a.level && j.level >= b.level);
        assert!(j.assurance <= a.assurance && j.assurance <= b.assurance);
        assert!(a.compartments.is_subset(j.compartments));
        assert!(b.compartments.is_subset(j.compartments));
    }

    #[test]
    fn join_laws() {
        let a = label(Level::Internal, Assurance::Classical, &[0]);
        let b = label(Level::Secret, Assurance::Unverified, &[1]);
        let c = label(Level::Public, Assurance::PqHybrid, &[2]);
        // commutative
        assert_eq!(a.join(&b), b.join(&a));
        // associative
        assert_eq!(a.join(&b).join(&c), a.join(&b.join(&c)));
        // idempotent
        assert_eq!(a.join(&a), a);
        // per-axis identity
        assert_eq!(a.join(&SecurityLabel::join_identity()), a);
    }

    #[test]
    fn join_all_secret_input_propagates() {
        // Confidentiality rises, integrity falls, and compartments accumulate.
        let inputs = vec![
            label(Level::Public, Assurance::Classical, &[0]),
            label(Level::Secret, Assurance::PqHybrid, &[1]),
            label(Level::Internal, Assurance::Classical, &[2]),
        ];
        let j = SecurityLabel::join_all(&inputs);
        assert_eq!(j.level, Level::Secret);
        assert_eq!(j.assurance, Assurance::Classical);
        assert_eq!(j.compartments.len(), 3);
    }

    #[test]
    fn join_all_mixed_classical_and_pq_hybrid_yields_classical() {
        let inputs = vec![
            label(Level::Public, Assurance::PqHybrid, &[]),
            label(Level::Public, Assurance::Classical, &[]),
        ];

        assert_eq!(
            SecurityLabel::join_all(&inputs).assurance,
            Assurance::Classical
        );
    }

    #[test]
    fn join_all_empty_is_fail_closed_unverified() {
        let none: Vec<&SecurityLabel> = vec![];
        let joined = SecurityLabel::join_all(none);
        assert_eq!(joined.assurance, Assurance::Unverified);
        assert!(joined.is_bottom());
        assert_ne!(joined, SecurityLabel::join_identity());
    }

    #[test]
    fn serde_roundtrip_is_canonical() {
        let l = label(Level::Secret, Assurance::PqHybrid, &[1, 0]);
        let json = serde_json::to_string(&l).unwrap();
        let back: SecurityLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(l, back);
        // canonical: re-serializing yields identical bytes (content-binding needs this)
        assert_eq!(json, serde_json::to_string(&back).unwrap());
    }

    // ── CredentialClearance two-axis wire grammar (v16 §11, H2) ─────────────

    /// The canonical fixture `[2,[5,7]]` is Confidential with compartments
    /// {5,7}; serialization is exactly that two-element array and round-trips,
    /// and every frozen level maps.
    #[test]
    fn credential_clearance_wire_is_canonical_two_axis() {
        let value = CredentialClearance::from_label(label(Level::Confidential, Assurance::Classical, &[5, 7]));
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(json, "[2,[5,7]]", "exact two-axis wire form");
        let back: CredentialClearance = serde_json::from_str(&json).unwrap();
        assert_eq!(value, back);

        let parsed: CredentialClearance = serde_json::from_str("[2,[5,7]]").unwrap();
        assert_eq!(parsed.level, Level::Confidential);
        assert!(parsed.compartments.contains(5) && parsed.compartments.contains(7));
        assert_eq!(parsed.compartments.len(), 2);

        // Empty compartments are allowed.
        let empty: CredentialClearance = serde_json::from_str("[0,[]]").unwrap();
        assert_eq!(empty.level, Level::Public);
        assert!(empty.compartments.is_empty());

        // The frozen level domain 0..=3 maps exactly.
        for (raw, lvl) in [
            (0, Level::Public),
            (1, Level::Internal),
            (2, Level::Confidential),
            (3, Level::Secret),
        ] {
            let v: CredentialClearance =
                serde_json::from_str(&format!("[{raw},[]]")).unwrap();
            assert_eq!(v.level, lvl);
        }
    }

    /// Every malformed / noncanonical / assurance-bearing wire value denies.
    #[test]
    fn credential_clearance_wire_rejects_malformed() {
        let reject = |s: &str| {
            assert!(
                serde_json::from_str::<CredentialClearance>(s).is_err(),
                "must reject {s}"
            );
        };
        reject("[4,[]]"); // unknown level (domain 0..=3)
        reject("[1,[64]]"); // compartment index out of range (0..=63)
        reject("[1,[5,5]]"); // duplicate index
        reject("[1,[7,5]]"); // descending / non-ascending order
        reject("[1,[],1]"); // three elements (an appended assurance axis)
        reject("[1]"); // one element (missing compartments)
        reject("[]"); // empty array
        reject("[1,160]"); // compartments as a bitmask integer, not an index array
        reject(r#"[1,["pii"]]"#); // compartment NAMES instead of indices
        reject(r#"["confidential",[]]"#); // level as a name string
        // The struct/object form (assurance-bearing) never parses as clearance.
        reject(r#"{"level":1,"assurance":"classical","compartments":0}"#);
    }

    /// The wire projection is assurance-free: two labels differing ONLY in the
    /// assurance axis project to the identical wire value and bytes.
    #[test]
    fn credential_clearance_wire_drops_assurance() {
        let classical =
            CredentialClearance::from_label(label(Level::Internal, Assurance::Classical, &[3]));
        let hybrid =
            CredentialClearance::from_label(label(Level::Internal, Assurance::PqHybrid, &[3]));
        assert_eq!(classical, hybrid, "assurance is not part of the wire identity");
        let json = serde_json::to_string(&hybrid).unwrap();
        assert_eq!(json, serde_json::to_string(&classical).unwrap());
        assert_eq!(json, "[1,[3]]");
        assert!(
            !json.contains("assurance")
                && !json.contains("classical")
                && !json.contains("pq"),
            "no assurance token on the wire: {json}"
        );
    }
}
