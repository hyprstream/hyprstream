//! Lattice interface — the **assumed S1 contract** (ticket #567/S1 owns the impl).
//!
//! S4 (this ticket, #570) develops the TE evaluator + AVC against this interface so the
//! two streams can proceed in parallel. EVERYTHING in this file marked `S1-ASSUMPTION`
//! is a requirement the S1 lattice implementation must satisfy for reconciliation. If S1
//! lands a different shape, only this file changes — `te.rs` / `avc.rs` depend solely on
//! the traits/types declared here.
//!
//! ## The contract S1 must satisfy
//!
//! 1. A [`SecurityLabel`] is a small, cheap-to-clone, content-bound identifier for a point
//!    in the lattice. It is `Copy` (a compact id), `Ord`+`Hash` (so it can key the AVC and
//!    the compiled TE matrix), and totally serializable.
//! 2. The lattice is a *partial* order with a [`Lattice::dominates`] (`⊒`) test and a
//!    [`Lattice::join`] (least-upper-bound, for IFC of derived layers). Both are TOTAL —
//!    defined for every pair of labels the lattice knows — and never panic.
//! 3. There is a designated **bottom** (public / least-classified) and the lattice can
//!    report whether a raw label id is *known* ([`Lattice::is_known`]). An unknown label is
//!    treated as denied by the floor (design §1 invariant 2: unlabeled = denied).
//!
//! The MAC floor (per-op `subject.ctx ⊒ object.label` + IFC join) is computed against this
//! trait and is **independent of any grant/token/UCAN** (design §3, §10).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Compact, content-bound identifier for a point in the security lattice.
///
/// S1-ASSUMPTION: a label is representable as a `u32` interned id (one per distinct
/// MLS-level×compartment-set or type-lattice point). The actual semantic content (level +
/// compartments) lives in the S1 lattice; the evaluator/AVC only ever compare *ids* on the
/// hot path — comparing structured labels per op would blow the latency budget.
///
/// `0` is reserved for [`SecurityLabel::BOTTOM`] (public / least). The interner is owned by
/// S1; ids are stable for the lifetime of a compiled policy generation (see
/// `compiled::CompiledPolicy::generation`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SecurityLabel(pub u32);

impl SecurityLabel {
    /// The bottom of the lattice (public / least-classified). Dominated by everything.
    pub const BOTTOM: SecurityLabel = SecurityLabel(0);

    /// Raw interned id.
    #[inline]
    pub const fn id(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SecurityLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// The lattice engine S1 provides: dominance (`⊒`) + IFC join (least-upper-bound).
///
/// S1-ASSUMPTION: implementations are pure, total, deterministic, and `Send + Sync`. No
/// method allocates on the hot path beyond what's stated, and none panics for any input —
/// unknown ids fail closed (see [`Lattice::dominates`]).
pub trait Lattice: Send + Sync {
    /// `clearance ⊒ label` — does the subject clearance dominate (is at least as high as)
    /// the object label? This is the MAC floor's core comparison (design §3 per-op
    /// dominance, §13 "exact lattice comparison").
    ///
    /// S1-ASSUMPTION: TOTAL. For two *known* labels returns the exact partial-order
    /// result. If EITHER label is unknown to this lattice generation, MUST return `false`
    /// (fail closed — unlabeled = denied, design §1 invariant 2). Never panics.
    fn dominates(&self, clearance: SecurityLabel, label: SecurityLabel) -> bool;

    /// IFC join — least upper bound of two labels (design §3: a derived layer's label =
    /// join(inputs); aggregation can't launder classification).
    ///
    /// S1-ASSUMPTION: TOTAL and well-defined for every known pair (the lattice has a top,
    /// or joins are closed). If either input is unknown, MUST return the lattice **top**
    /// (the most restrictive label) so an unknown input can only *raise* classification,
    /// never lower it. Returned label is always known.
    fn join(&self, a: SecurityLabel, b: SecurityLabel) -> SecurityLabel;

    /// Whether `label` is a known point in *this* lattice generation. Used by the floor to
    /// reject unlabeled subjects/objects before any grant logic runs.
    ///
    /// S1-ASSUMPTION: `is_known(SecurityLabel::BOTTOM)` is always `true`.
    fn is_known(&self, label: SecurityLabel) -> bool;

    /// The lattice top (most restrictive / highest). Used as the fail-closed join result.
    ///
    /// S1-ASSUMPTION: dominated-by-nothing-below; `dominates(top, x) == true` for all known
    /// `x`.
    fn top(&self) -> SecurityLabel;
}

/// IFC convenience: join a slice of input labels into the derived label (design §3 — sealed
/// at rollup). Folds with [`Lattice::join`]; empty input yields BOTTOM. This is the function
/// S6/seal-time labeling calls; it lives here because it is pure lattice algebra.
pub fn ifc_join<L: Lattice + ?Sized>(lattice: &L, inputs: &[SecurityLabel]) -> SecurityLabel {
    inputs
        .iter()
        .copied()
        .fold(SecurityLabel::BOTTOM, |acc, x| lattice.join(acc, x))
}

// ---------------------------------------------------------------------------------------
// A minimal in-tree lattice so S4 is testable/runnable BEFORE S1 lands. This is a STUB:
// a flat MLS chain (BOTTOM=0 ⊑ 1 ⊑ 2 ⊑ ...) with no compartments. S1 replaces it. The TE
// evaluator and AVC never reference this type — only the `Lattice` trait — so swapping in
// S1's implementation is a one-line change at construction sites.
// ---------------------------------------------------------------------------------------

/// Stub linear-order lattice for S4-local testing. **Replace with S1.**
///
/// Levels `0..=max` form a total chain; `dominates(a,b) = a >= b`; `join(a,b) = max(a,b)`.
/// No compartments, no real type lattice. Marked clearly so it is not mistaken for the real
/// engine.
#[derive(Debug, Clone)]
pub struct StubLinearLattice {
    max_level: u32,
}

impl StubLinearLattice {
    /// Levels `0..=max_level` are known.
    pub fn new(max_level: u32) -> Self {
        Self { max_level }
    }
}

impl Lattice for StubLinearLattice {
    fn dominates(&self, clearance: SecurityLabel, label: SecurityLabel) -> bool {
        if !self.is_known(clearance) || !self.is_known(label) {
            return false; // fail closed
        }
        clearance.0 >= label.0
    }

    fn join(&self, a: SecurityLabel, b: SecurityLabel) -> SecurityLabel {
        if !self.is_known(a) || !self.is_known(b) {
            return self.top(); // fail closed — unknown can only raise
        }
        SecurityLabel(a.0.max(b.0))
    }

    fn is_known(&self, label: SecurityLabel) -> bool {
        label.0 <= self.max_level
    }

    fn top(&self) -> SecurityLabel {
        SecurityLabel(self.max_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_dominance_is_chain() {
        let l = StubLinearLattice::new(3);
        assert!(l.dominates(SecurityLabel(2), SecurityLabel(1)));
        assert!(l.dominates(SecurityLabel(1), SecurityLabel(1)));
        assert!(!l.dominates(SecurityLabel(1), SecurityLabel(2)));
    }

    #[test]
    fn unknown_label_fails_closed() {
        let l = StubLinearLattice::new(3);
        // 9 is unknown -> dominates must be false either way (fail closed)
        assert!(!l.dominates(SecurityLabel(9), SecurityLabel(1)));
        assert!(!l.dominates(SecurityLabel(3), SecurityLabel(9)));
        // join with unknown returns top
        assert_eq!(l.join(SecurityLabel(9), SecurityLabel(1)), l.top());
    }

    #[test]
    fn ifc_join_raises_to_max() {
        let l = StubLinearLattice::new(3);
        let labels = [SecurityLabel(1), SecurityLabel(3), SecurityLabel(2)];
        assert_eq!(ifc_join(&l, &labels), SecurityLabel(3));
        assert_eq!(ifc_join(&l, &[]), SecurityLabel::BOTTOM);
    }
}
