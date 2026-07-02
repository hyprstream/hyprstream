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
//!     ∧ subject.clearance ⊒ object.label                       ← MAC floor (content-truth, IFC)
//! ```
//!
//! The lattice floor is the **intrinsic** dominance on S1's [`SecurityLabel`]
//! ([`SecurityLabel::can_access`]) — it takes no policy argument and no `Lattice` lookup, so it
//! is uncircumventable: no TE entry, grant, or token can relax it (design §1 invariant 3).
//! Default is DENY: an unknown subject type, object type, action, or a non-dominating /
//! unlabeled clearance → DENY. There is no permissive mode.
//!
//! ## S1 reconciliation (#567 landed)
//!
//! - The floor is S1's intrinsic [`SecurityLabel::can_access`], not a `Lattice::can_access`
//!   trait method. The evaluator still holds the S1 [`Lattice`] policy object, but only for
//!   *well-formedness* concerns (column alignment / validation / version binding), never for
//!   the per-op order.
//! - The subject's `clearance` carried in [`SubjectCtx`] is the clearance label the PEP
//!   already produced from a verified [`SecurityContext`] (assurance clamped to the verified
//!   key material — #548). The PEP, not this evaluator, applies that clamp.
//! - "Unlabeled ⇒ deny" lives at the PEP boundary: a subject/object with no S1 label is
//!   `Option::None` and never reaches `evaluate`. There is deliberately no `Default`
//!   [`SecurityLabel`].
//!
//! ## Relationship to the AVC and to Casbin
//!
//! - The AVC ([`crate::mac::avc`]) caches the *output* of this function so the PEP (S2) never
//!   re-runs it per op once warm. This evaluator is what fills the cache on a miss, and is
//!   itself cheap enough to be the miss path.
//! - Casbin authors the policy; S5's compiler lowers it into the [`TeMatrix`] entries this
//!   evaluator reads. Casbin's `Enforcer::enforce` is NEVER on this path.

use crate::mac::lattice::{Compartment, Lattice, SecurityLabel};
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

/// Interned id of an *action* / op. One per distinct op in the compiled policy. Maps from
/// S3's action vocabulary at compile time — see [`Action::from_scope_action`] for the
/// canonical [`ScopeAction`]→id assignment so a compiled matrix and a PEP agree on ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Action(pub u32);

/// S3's canonical action vocabulary (#569), mirrored from the Cap'n Proto `ScopeAction`
/// enum in `hyprstream-rpc/schema/annotations.capnp` (`$mcpScope(...)`). Keeping the
/// discriminants 1:1 with that schema enum is what lets S5's UCAN→TE compiler and the PEP
/// intern the same `cmd`/`$scope` to the same [`Action`] id without a side table.
///
/// Note (minimal handling, per #570): `Subscribe`/`Publish` exist in the `ScopeAction`
/// schema enum but have **no dedicated `auth::Operation` Rust variant** today. They are
/// represented here so streaming ops are interned to a *stable* id; the
/// control-plane `Operation` parity for them lands with the streaming-scope work. The
/// remaining variants line up 1:1 with `auth::Operation` (`Query`/`Write`/`Manage`/
/// `Infer`/`Train`/`Serve`/`Context`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u32)]
pub enum ScopeAction {
    /// `query @0` ⇄ `Operation::Query`.
    Query = 0,
    /// `write @1` ⇄ `Operation::Write`.
    Write = 1,
    /// `manage @2` ⇄ `Operation::Manage`.
    Manage = 2,
    /// `infer @3` ⇄ `Operation::Infer`.
    Infer = 3,
    /// `train @4` ⇄ `Operation::Train`.
    Train = 4,
    /// `serve @5` ⇄ `Operation::Serve`.
    Serve = 5,
    /// `context @6` ⇄ `Operation::Context`.
    Context = 6,
    /// `subscribe @7` — streaming read; no `Operation` parity yet (handled minimally).
    Subscribe = 7,
    /// `publish @8` — streaming write; no `Operation` parity yet (handled minimally).
    Publish = 8,
}

impl ScopeAction {
    /// Every canonical action, in schema-discriminant order. The closed action
    /// vocabulary: a wildcard ability expands over exactly this set (#676).
    pub const ALL: [ScopeAction; 9] = [
        ScopeAction::Query,
        ScopeAction::Write,
        ScopeAction::Manage,
        ScopeAction::Infer,
        ScopeAction::Train,
        ScopeAction::Serve,
        ScopeAction::Context,
        ScopeAction::Subscribe,
        ScopeAction::Publish,
    ];

    /// The canonical verb string, exactly as named in the Cap'n Proto
    /// `ScopeAction` schema enum (`annotations.capnp`). The single string↔id
    /// assignment shared by the S3 scope parser, the S5 compiler's
    /// `PermissionMap`, and the PEP — no side tables.
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            ScopeAction::Query => "query",
            ScopeAction::Write => "write",
            ScopeAction::Manage => "manage",
            ScopeAction::Infer => "infer",
            ScopeAction::Train => "train",
            ScopeAction::Serve => "serve",
            ScopeAction::Context => "context",
            ScopeAction::Subscribe => "subscribe",
            ScopeAction::Publish => "publish",
        }
    }

    /// Parse a canonical verb string. `None` for anything else — an
    /// unrecognized verb grants nothing (fail-closed), never a guess.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.as_str() == s)
    }

    /// Recover the enum from an interned [`Action`] id. `None` for an id
    /// outside the schema enum — such a rule has no recognized meaning and is
    /// treated as an escalation by `check_no_escalation` (fail-closed).
    pub const fn from_action(action: Action) -> Option<Self> {
        match action.0 {
            0 => Some(ScopeAction::Query),
            1 => Some(ScopeAction::Write),
            2 => Some(ScopeAction::Manage),
            3 => Some(ScopeAction::Infer),
            4 => Some(ScopeAction::Train),
            5 => Some(ScopeAction::Serve),
            6 => Some(ScopeAction::Context),
            7 => Some(ScopeAction::Subscribe),
            8 => Some(ScopeAction::Publish),
            _ => None,
        }
    }
}

impl Action {
    /// Intern a canonical S3 [`ScopeAction`] to its stable [`Action`] id. The id IS the
    /// schema discriminant, so the assignment is fixed across the compiler and every PEP
    /// (no per-process numbering drift).
    #[inline]
    pub const fn from_scope_action(a: ScopeAction) -> Self {
        Action(a as u32)
    }
}

impl From<ScopeAction> for Action {
    #[inline]
    fn from(a: ScopeAction) -> Self {
        Action::from_scope_action(a)
    }
}

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
/// "subject ctx from Ed25519 claims") via S1's [`SecurityContext`] — the clearance carried
/// here is `SecurityContext::clearance()`, with assurance already clamped to the verified
/// key material (#548). The TE type is an interned id; the clearance is S1's `Copy`
/// structured label — both cheap to pass per op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubjectCtx {
    /// TE subject type (SELinux domain).
    pub subject_type: SubjectType,
    /// Lattice clearance (S1 structured label) — checked against the object label by the
    /// MAC floor via intrinsic dominance.
    pub clearance: SecurityLabel,
}

/// The object's security context: a TE object type plus its content-bound label. The label
/// comes ONLY from the content-addressed manifest (design §3, §14 "labels ONLY from
/// manifests"), never from a token. The PEP resolves it (S1 `LabeledObject::security_label`,
/// `None` ⇒ deny before this point) before calling the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectCtx {
    /// TE object type.
    pub object_type: ObjectType,
    /// Content-bound label (S1 structured label) from the manifest.
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
    /// a generation-counter compare). Equals the embedded lattice's
    /// [`LatticeVersion::generation`].
    fn generation(&self) -> u64;
}

/// Concrete PDP: a compiled [`TeMatrix`] + the S1 [`Lattice`] policy object. Small, total,
/// allocation-free on the hot path. This is the verifiable TCB core.
///
/// The [`Lattice`] is held for *well-formedness* and *column alignment*, NOT for the per-op
/// order: the floor uses S1's intrinsic [`SecurityLabel::can_access`]. The PDP's `generation`
/// is bound to `lattice.version().generation()` at construction — the desync-proof binding
/// (#570): the same lattice bytes that minted a label's compartment bits also tag the
/// compiled policy that evaluates it.
pub struct LatticeTeEvaluator {
    matrix: TeMatrix,
    lattice: Lattice,
    generation: u64,
}

impl LatticeTeEvaluator {
    /// Construct from a compiled matrix and the S1 lattice policy. The policy generation is
    /// taken from the lattice version — they are definitionally equal (#570 reconciliation):
    /// a compiled TE matrix is only valid against the lattice generation that minted its
    /// compartment bit assignments.
    pub fn new(matrix: TeMatrix, lattice: Lattice) -> Self {
        let generation = lattice.version().generation();
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

    /// Borrow the lattice policy (column alignment / validation / re-serialization into the
    /// signed `CompiledPolicy`).
    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// The compartment vocabulary in **canonical bit-index order** (`names()[i]` is the
    /// compartment interned at bit `i`). A TE-matrix builder / compiler aligns its
    /// compartment columns to label bits by iterating this — NEVER by hardcoding compartment
    /// names — so a column index always equals the [`CompartmentSet`](crate::mac::lattice::CompartmentSet)
    /// bit a [`SecurityLabel`] carries.
    pub fn compartment_columns(&self) -> &[Compartment] {
        self.lattice.compartment_names()
    }
}

impl TeEvaluator for LatticeTeEvaluator {
    #[inline]
    fn evaluate(&self, subject: SubjectCtx, object: ObjectCtx, action: Action) -> Decision {
        // 1. MAC floor FIRST and independently — S1 intrinsic dominance, content-bound and
        //    uncircumventable. Fail-closed: a non-dominating clearance → DENY, regardless of
        //    what the TE matrix says (design §1 inv.3, §10). Unlabeled subject/object never
        //    reach here (the PEP denies on `None` before constructing the contexts).
        if !subject.clearance.can_access(&object.label) {
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mac::lattice::{Assurance, Compartment, LatticeVersion, Level};

    fn rule(s: u32, o: u32, a: u32) -> TeRule {
        TeRule {
            subject_type: SubjectType(s),
            object_type: ObjectType(o),
            action: Action(a),
        }
    }

    fn lattice() -> Lattice {
        Lattice::new(
            LatticeVersion(7),
            [Compartment::new("pii"), Compartment::new("finance")],
        )
    }

    /// A clearance label at (level, assurance, compartment names).
    fn clearance(level: Level, assurance: Assurance, names: &[&str]) -> SecurityLabel {
        lattice()
            .label(level, assurance, names.iter().map(|n| Compartment::new(*n)))
            .unwrap()
    }

    fn subj(t: u32, clr: SecurityLabel) -> SubjectCtx {
        SubjectCtx {
            subject_type: SubjectType(t),
            clearance: clr,
        }
    }

    fn obj(t: u32, lbl: SecurityLabel) -> ObjectCtx {
        ObjectCtx {
            object_type: ObjectType(t),
            label: lbl,
        }
    }

    fn eval() -> LatticeTeEvaluator {
        // Allow (subj=1, obj=1, act=1). Escalate (subj=1, obj=1, act=2).
        let mut allow = HashSet::new();
        allow.insert(rule(1, 1, 1));
        let mut escalate = HashSet::new();
        escalate.insert(rule(1, 1, 2));
        LatticeTeEvaluator::new(TeMatrix::new(allow, escalate), lattice())
    }

    #[test]
    fn generation_bound_to_lattice_version() {
        // #570 reconciliation: generation == LatticeVersion::generation.
        assert_eq!(eval().generation(), 7);
    }

    #[test]
    fn permit_requires_te_and_floor() {
        let e = eval();
        let high = clearance(Level::Secret, Assurance::PqHybrid, &["pii"]);
        let low = clearance(Level::Confidential, Assurance::Classical, &[]);
        // TE allows AND clearance ⊒ label: permit.
        assert_eq!(
            e.evaluate(subj(1, high), obj(1, low), Action(1)),
            Decision::Permit
        );
    }

    #[test]
    fn floor_denies_even_when_te_allows() {
        let e = eval();
        // TE allows but a Public clearance does NOT dominate a Secret label: floor denies.
        let clr = clearance(Level::Public, Assurance::Classical, &[]);
        let lbl = clearance(Level::Secret, Assurance::Classical, &[]);
        assert_eq!(
            e.evaluate(subj(1, clr), obj(1, lbl), Action(1)),
            Decision::Deny
        );
    }

    #[test]
    fn floor_denies_on_missing_compartment() {
        let e = eval();
        // High level/assurance but NOT cleared into the object's "finance" compartment.
        let clr = clearance(Level::Secret, Assurance::PqHybrid, &["pii"]);
        let lbl = clearance(Level::Public, Assurance::Unverified, &["finance"]);
        assert_eq!(
            e.evaluate(subj(1, clr), obj(1, lbl), Action(1)),
            Decision::Deny
        );
    }

    #[test]
    fn assurance_floor_denies_classical_on_pqc_object() {
        // #548 via the SAME dominance check: a classical clearance can't read a pq-hybrid label.
        let e = eval();
        let clr = clearance(Level::Secret, Assurance::Classical, &["pii"]);
        let lbl = clearance(Level::Public, Assurance::PqHybrid, &[]);
        assert_eq!(
            e.evaluate(subj(1, clr), obj(1, lbl), Action(1)),
            Decision::Deny
        );
    }

    #[test]
    fn default_deny_on_te_miss() {
        let e = eval();
        let high = clearance(Level::Secret, Assurance::PqHybrid, &["pii"]);
        let low = clearance(Level::Public, Assurance::Classical, &[]);
        // Floor holds but no TE rule for action 9: deny.
        assert_eq!(
            e.evaluate(subj(1, high), obj(1, low), Action(9)),
            Decision::Deny
        );
        // Unknown subject type: deny.
        assert_eq!(
            e.evaluate(subj(5, high), obj(1, low), Action(1)),
            Decision::Deny
        );
    }

    #[test]
    fn escalate_is_not_a_permit_but_floor_still_first() {
        let e = eval();
        let high = clearance(Level::Secret, Assurance::PqHybrid, &["pii"]);
        let low = clearance(Level::Public, Assurance::Classical, &[]);
        // Floor holds, TE says escalate.
        assert_eq!(
            e.evaluate(subj(1, high), obj(1, low), Action(2)),
            Decision::Escalate
        );
        assert!(!Decision::Escalate.is_permit());
        // But if the floor fails, escalate downgrades to a hard deny (floor is independent).
        let pub_clr = clearance(Level::Public, Assurance::Classical, &[]);
        let sec_lbl = clearance(Level::Secret, Assurance::Classical, &[]);
        assert_eq!(
            e.evaluate(subj(1, pub_clr), obj(1, sec_lbl), Action(2)),
            Decision::Deny
        );
    }

    #[test]
    fn scope_action_interns_to_schema_discriminant() {
        // Stable 1:1 with the capnp ScopeAction enum so compiler and PEP agree on ids.
        assert_eq!(Action::from(ScopeAction::Query), Action(0));
        assert_eq!(Action::from(ScopeAction::Context), Action(6));
        assert_eq!(Action::from(ScopeAction::Subscribe), Action(7));
        assert_eq!(Action::from(ScopeAction::Publish), Action(8));
    }

    #[test]
    fn compartment_columns_align_to_bits() {
        let e = eval();
        let cols = e.compartment_columns();
        // Column index == the bit a SecurityLabel carries for that compartment name.
        assert_eq!(cols[0], Compartment::new("pii"));
        assert_eq!(cols[1], Compartment::new("finance"));
        assert_eq!(e.lattice().bit_of(&Compartment::new("finance")), Some(1));
    }
}
