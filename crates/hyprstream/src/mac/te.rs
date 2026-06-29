//! Type-Enforcement evaluator — the **Policy Decision Point** hot-path core (S4 / #570).
//!
//! A TOTAL, default-deny function `(subject_ctx, object_label, action) → Decision` over the
//! compiled TE matrix AND the independent MAC lattice floor. This is intentionally tiny and
//! verifiable — it is the data-plane evaluator that runs per op. It is **NOT** Casbin's
//! general string matcher (that stays control-plane; see [`crate::mac`] module docs).
//!
//! ## Two independent checks, conjoined (design §3, §10)
//!
//! ```text
//! PERMIT(subject, object, action) ⟺
//!       te_matrix.permits(subject.type, object.type, action)   ← compiled TE (authority gate)
//!     ∧ lattice.dominates(subject.clearance, object.label)     ← MAC floor (content-truth, IFC)
//! ```
//!
//! The lattice floor is **independent and uncircumventable** — no TE entry, grant, or token
//! can relax it (design §1 invariant 3). Default is DENY: an unknown subject type, object
//! type, action, or unlabeled object → DENY. There is no permissive mode.
//!
//! ## Relationship to the AVC and to Casbin
//!
//! - The [`AvcEvaluator`](crate::mac::avc) caches the *output* of this function so the PEP
//!   (S2) never re-runs it per op once warm. This evaluator is what fills the cache on a
//!   miss, and is itself cheap enough to be the miss path.
//! - Casbin authors the policy; S5's compiler lowers it into the [`TeMatrix`] entries this
//!   evaluator reads. Casbin's `Enforcer::enforce` is NEVER on this path.

use crate::mac::lattice::{Lattice, SecurityLabel};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Interned id of a TE *subject type* (SELinux "domain"). One per subject security context
/// class in the compiled policy. Stable within a policy generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubjectType(pub u32);

/// Interned id of a TE *object type*. One per object class in the compiled policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObjectType(pub u32);

/// Interned id of an *action* / op (9p walk/open/read/write/create/remove + RPC ops).
/// One per distinct op in the compiled policy. Maps from the UCAN `cmd` at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Action(pub u32);

/// The decision a PDP returns. Three-valued per design §8 — `Escalate` is a
/// *deny-on-the-data-plane* that the control plane may turn into a ceiling-amendment flow,
/// but for the per-op monitor it is NOT a permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    /// Access permitted: TE matrix grants it AND the lattice floor holds.
    Permit,
    /// Access denied (default; TE miss, floor failure, or explicit deny).
    Deny,
    /// Statically denied here, but eligible for an escalation-tier ceiling amendment
    /// (design §8). The monitor treats this as DENY for the op; the control plane may act.
    Escalate,
}

impl Decision {
    /// Is this a permit? The only value the PEP may treat as "allow".
    #[inline]
    pub fn is_permit(self) -> bool {
        matches!(self, Decision::Permit)
    }
}

/// The subject's security context as the evaluator sees it: a TE type (domain) plus the
/// lattice clearance. Built by the PEP from the Ed25519-verified token/claims (design §2
/// "subject ctx from Ed25519 claims"). Both fields are interned ids — cheap to pass per op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubjectCtx {
    /// TE subject type (SELinux domain).
    pub subject_type: SubjectType,
    /// Lattice clearance — checked against the object label by the MAC floor.
    pub clearance: SecurityLabel,
}

/// The object's security context: a TE object type plus its content-bound label. The label
/// comes ONLY from the content-addressed manifest (design §3, §14 "labels ONLY from
/// manifests"), never from a token. The PEP resolves it before calling the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectCtx {
    /// TE object type.
    pub object_type: ObjectType,
    /// Content-bound label from the manifest.
    pub label: SecurityLabel,
}

/// A single permitted TE triple `(subject_type, object_type, action)`. The compiled matrix
/// is the *allow-set* of these triples (default-deny: absence = deny). Produced by S5's
/// UCAN→TE compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TeRule {
    pub subject_type: SubjectType,
    pub object_type: ObjectType,
    pub action: Action,
}

/// The compiled type-enforcement matrix: a flat allow-set of [`TeRule`]s. Default-deny —
/// any triple not present is denied. This is the data-plane artifact loaded from signed
/// compiled policy (see [`crate::mac::compiled`]). Kept as a `HashSet` so the hot-path
/// lookup is O(1); it is also the unit S5 produces and `compiled` signs.
///
/// Optionally carries an *escalation-set*: triples that, while not permitted, are eligible
/// for a ceiling-amendment (design §8). A triple in neither set is a hard `Deny`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TeMatrix {
    allow: HashSet<TeRule>,
    escalate: HashSet<TeRule>,
}

impl TeMatrix {
    /// Build from explicit allow/escalate rule sets (the shape S5's compiler emits).
    pub fn new(allow: HashSet<TeRule>, escalate: HashSet<TeRule>) -> Self {
        Self { allow, escalate }
    }

    /// Build an allow-only matrix (no escalation band).
    pub fn from_allow(allow: impl IntoIterator<Item = TeRule>) -> Self {
        Self {
            allow: allow.into_iter().collect(),
            escalate: HashSet::new(),
        }
    }

    /// Number of allow rules (for diagnostics / matrix hashing sanity).
    pub fn allow_len(&self) -> usize {
        self.allow.len()
    }

    /// Sorted-vec projection of the allow/escalate sets, for canonical (order-stable)
    /// encoding/hashing in [`crate::mac::compiled`]. Returns `(allow, escalate)`; the caller
    /// sorts. The hot path never calls this.
    pub fn sorted_rules(&self) -> (Vec<TeRule>, Vec<TeRule>) {
        (
            self.allow.iter().copied().collect(),
            self.escalate.iter().copied().collect(),
        )
    }

    /// Raw TE decision for a triple, *without* the lattice floor. Total + default-deny.
    /// Pure on the matrix contents — this is the part S5 produces policy for.
    #[inline]
    pub fn te_decision(&self, rule: TeRule) -> Decision {
        if self.allow.contains(&rule) {
            Decision::Permit
        } else if self.escalate.contains(&rule) {
            Decision::Escalate
        } else {
            Decision::Deny
        }
    }
}

/// The Policy Decision Point evaluator interface — **what S2 (the PEP) calls per op**.
///
/// Implementations are TOTAL and default-deny: every well-formed call returns a [`Decision`]
/// and never panics. This is the clean contract the AVC wraps and S5/S6 produce policy for.
pub trait TeEvaluator: Send + Sync {
    /// Evaluate one operation. TOTAL, default-deny. Conjoins the compiled TE decision with
    /// the independent lattice floor: a `Permit` requires BOTH; the floor can only deny.
    fn evaluate(&self, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Decision;

    /// The policy generation this evaluator answers for. The AVC tags cached entries with
    /// it so a policy reload invalidates stale decisions (no per-op staleness check beyond
    /// a generation-counter compare).
    fn generation(&self) -> u64;
}

/// Concrete PDP: a compiled [`TeMatrix`] + a [`Lattice`]. Small, total, allocation-free on
/// the hot path. This is the verifiable TCB core.
pub struct LatticeTeEvaluator<L: Lattice> {
    matrix: TeMatrix,
    lattice: L,
    generation: u64,
}

impl<L: Lattice> LatticeTeEvaluator<L> {
    /// Construct from a compiled matrix, a lattice, and the policy generation id.
    pub fn new(matrix: TeMatrix, lattice: L, generation: u64) -> Self {
        Self {
            matrix,
            lattice,
            generation,
        }
    }

    /// Borrow the matrix (diagnostics / re-signing).
    pub fn matrix(&self) -> &TeMatrix {
        &self.matrix
    }
}

impl<L: Lattice> TeEvaluator for LatticeTeEvaluator<L> {
    #[inline]
    fn evaluate(&self, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Decision {
        // 1. MAC floor FIRST and independently — content-bound, uncircumventable.
        //    Fail-closed: unknown/unlabeled or non-dominating clearance → DENY, regardless
        //    of what the TE matrix says (design §1 inv.3, §10).
        if !self.lattice.dominates(subject.clearance, object.label) {
            return Decision::Deny;
        }
        // 2. Compiled TE decision (the authority gate). Default-deny by construction.
        self.matrix.te_decision(TeRule {
            subject_type: subject.subject_type,
            object_type: object.object_type,
            action,
        })
    }

    #[inline]
    fn generation(&self) -> u64 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mac::lattice::StubLinearLattice;

    fn rule(s: u32, o: u32, a: u32) -> TeRule {
        TeRule {
            subject_type: SubjectType(s),
            object_type: ObjectType(o),
            action: Action(a),
        }
    }

    fn subj(t: u32, clr: u32) -> SubjectCtx {
        SubjectCtx {
            subject_type: SubjectType(t),
            clearance: SecurityLabel(clr),
        }
    }

    fn obj(t: u32, lbl: u32) -> ObjectCtx {
        ObjectCtx {
            object_type: ObjectType(t),
            label: SecurityLabel(lbl),
        }
    }

    fn eval() -> LatticeTeEvaluator<StubLinearLattice> {
        // Allow (subj=1, obj=1, act=1). Escalate (subj=1, obj=1, act=2).
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        let mut escalate = HashSet::new();
        escalate.insert(rule(1, 1, 2));
        LatticeTeEvaluator::new(
            TeMatrix::new(allow, escalate),
            StubLinearLattice::new(3),
            7,
        )
    }

    #[test]
    fn permit_requires_te_and_floor() {
        let e = eval();
        // TE allows AND clearance(2) ⊒ label(1): permit.
        assert_eq!(e.evaluate(subj(1, 2), obj(1, 1), Action(1)), Decision::Permit);
    }

    #[test]
    fn floor_denies_even_when_te_allows() {
        let e = eval();
        // TE allows but clearance(0) does NOT dominate label(1): floor denies.
        assert_eq!(e.evaluate(subj(1, 0), obj(1, 1), Action(1)), Decision::Deny);
    }

    #[test]
    fn default_deny_on_te_miss() {
        let e = eval();
        // Floor holds (2 ⊒ 1) but no TE rule for action 9: deny.
        assert_eq!(e.evaluate(subj(1, 2), obj(1, 1), Action(9)), Decision::Deny);
        // Unknown subject type: deny.
        assert_eq!(e.evaluate(subj(5, 2), obj(1, 1), Action(1)), Decision::Deny);
    }

    #[test]
    fn escalate_is_not_a_permit_but_floor_still_first() {
        let e = eval();
        // Floor holds, TE says escalate.
        assert_eq!(e.evaluate(subj(1, 2), obj(1, 1), Action(2)), Decision::Escalate);
        assert!(!Decision::Escalate.is_permit());
        // But if the floor fails, escalate downgrades to a hard deny (floor is independent).
        assert_eq!(e.evaluate(subj(1, 0), obj(1, 1), Action(2)), Decision::Deny);
    }

    #[test]
    fn unlabeled_object_fails_closed() {
        let e = eval();
        // Object label 99 is unknown to the lattice (max 3): floor denies (design §1 inv.2).
        assert_eq!(e.evaluate(subj(1, 3), obj(1, 99), Action(1)), Decision::Deny);
    }
}
