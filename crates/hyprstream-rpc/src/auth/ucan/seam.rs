//! **Milestone-2 seams** — clean interfaces deliberately left unimplemented in
//! S5 milestone 1 (#571). Each is gated on other work; defining the trait now
//! lets the milestone-1 core (token / chain / approval) compile, test, and ship
//! independently, and pins the contract so the deferred pieces wire in last with
//! no churn to the validated attenuation/approval logic.
//!
//! Nothing here is on any hot path. These are control-plane, compile-time
//! interfaces.

use super::capability::{Ability, Resource};

// ---------------------------------------------------------------------------
// SEAM 1 — action-vocabulary mapping (UCAN cmd ↔ ScopeAction / Operation)
// TODO(#571 milestone 2 / post-#582): wire in last, once S3 (#582) merges.
// ---------------------------------------------------------------------------

/// Maps the UCAN vocabulary (a structural [`Ability`] verb + [`Resource`]) onto
/// S3's concrete enforcement vocabulary — the `ScopeAction` / `Operation` action
/// ids and TE object types the compiled `TeMatrix` keys on.
///
/// **DEFERRED — do NOT implement in milestone 1.** S3 (#582) is mid-fix on the
/// `$scope` / `annotations.capnp` work that owns the canonical `ScopeAction`
/// enum (and will add `Operation::Subscribe` / `Operation::Publish`). The
/// milestone-1 attenuation core is intentionally vocabulary-independent so it
/// never depends on this; the mapping is the *last* thing to wire so it can pull
/// the final, merged `ScopeAction` discriminants without colliding with #582.
///
/// When implemented (milestone 2), the concrete `Action`/`ObjectType` ids it
/// returns MUST match `hyprstream::mac::te::Action::from_scope_action` /
/// `ScopeAction` so the compiler and every PEP intern the same verb to the same
/// id with no side table. The associated types are left abstract here so this
/// crate (the lower one) does not yet need to name the `hyprstream`-crate TE
/// types.
pub trait ActionVocabulary {
    /// The interned action-id type (milestone 2 = `hyprstream::mac::te::Action`).
    type Action;
    /// The interned object-type id type (milestone 2 = `hyprstream::mac::te::ObjectType`).
    type ObjectType;

    /// Map a UCAN ability verb to its enforcement action id, if the verb is in
    /// the closed vocabulary. `None` ⇒ unknown verb ⇒ the compiler MUST reject
    /// (fail-closed — never default to a permissive action).
    fn action_of(&self, ability: &Ability) -> Option<Self::Action>;

    /// Map a UCAN resource pointer to its TE object type, if recognized. `None`
    /// ⇒ unknown resource class ⇒ reject.
    fn object_type_of(&self, resource: &Resource) -> Option<Self::ObjectType>;
}

// ---------------------------------------------------------------------------
// SEAM 2 — Casbin/TE bundle emission
// TODO(#571 milestone 2): scaffold only; produces the hyprstream::mac TeMatrix.
// ---------------------------------------------------------------------------

/// Lowers a validated UCAN ceiling into the compiled enforcement bundle (the
/// Casbin/TE artifact S4 distributes). This is what actually *produces* the
/// `bundle_hash` the [`super::approval::ApprovalBinding`] signs.
///
/// **SCAFFOLD ONLY in milestone 1.** The concrete emitter lives in milestone 2
/// because it depends on (a) the action vocabulary ([`ActionVocabulary`], gated
/// on #582) and (b) the `hyprstream`-crate `TeMatrix` / `CompiledPolicy` types
/// (S4, `crates/hyprstream/src/mac/compiled.rs`). Crate layering note: the
/// emitter most likely lives in the `hyprstream` crate (which already depends on
/// this one and owns `TeMatrix`), consuming the milestone-1 validated `Ucan` +
/// `ActionVocabulary` from here. The contract is fixed now:
///
/// 1. input: a UCAN whose chain ALREADY passed [`super::chain::validate`];
/// 2. output: a compiled bundle whose authority is provably ⊆ the UCAN's
///    capabilities (the faithfulness obligation — see [`FaithfulnessCheck`]);
/// 3. the caller hashes the bundle (BLAKE3, == `CompiledPolicy::policy_hash()`)
///    and binds it via [`super::approval::SignedApproval::sign`].
pub trait BundleEmitter {
    /// The compiled bundle type (milestone 2 = a `hyprstream::mac::CompiledPolicy`
    /// or its `TeMatrix` + lattice generation).
    type Bundle;
    /// Emission error type.
    type Error;

    /// Compile a validated UCAN ceiling into a bundle. Implementors MUST be
    /// deterministic (same UCAN + same lattice generation ⇒ byte-identical
    /// bundle ⇒ identical hash) so the approval binding is reproducible.
    fn emit(&self, validated_ucan: &super::token::Ucan) -> Result<Self::Bundle, Self::Error>;
}

// ---------------------------------------------------------------------------
// SEAM 3 — faithfulness framework
// TODO(#571 milestone 2): reference interpreter + differential property tests
//                         + SMT equivalence for the structural fragment.
// ---------------------------------------------------------------------------

/// The faithfulness obligation: the emitted bundle grants *exactly* the
/// authority the approved UCAN does — no more (soundness: the bundle is a
/// ceiling) and, for the auditable fragment, no less (completeness).
///
/// **SCAFFOLD ONLY in milestone 1.** Milestone 2 implements this as three
/// mutually-reinforcing checks, all off the hot path:
/// 1. a **reference interpreter** that decides `(subject, object, action)`
///    directly from the UCAN capability set (the spec of intent);
/// 2. **differential property tests** comparing the reference interpreter to the
///    emitted bundle's `TeEvaluator` over randomized requests (must agree on
///    every input);
/// 3. an **SMT equivalence proof** for the structural (vocabulary-closed)
///    fragment, discharging the "for all requests" quantifier the property
///    tests only sample.
///
/// The contract: `check` returns `Ok(())` iff the bundle is faithful to the
/// UCAN, else a counterexample. A compiler that cannot discharge this MUST NOT
/// emit an approval (fail-closed: an unfaithful bundle is never signed).
pub trait FaithfulnessCheck {
    /// The bundle type under check (milestone 2 = the emitted TE bundle).
    type Bundle;
    /// A counterexample / failure describing where bundle and UCAN disagree.
    type Counterexample;

    /// Verify the emitted `bundle` faithfully realizes `ucan`'s ceiling.
    fn check(
        &self,
        ucan: &super::token::Ucan,
        bundle: &Self::Bundle,
    ) -> Result<(), Self::Counterexample>;
}
