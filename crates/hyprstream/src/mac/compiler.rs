//! **UCAN → TE bundle emission + faithfulness framework** — S5 / #571 milestone 2,
//! the vocabulary-INDEPENDENT half of the compiler.
//!
//! This module is the concrete home of the milestone-1 seams
//! [`hyprstream_rpc::auth::ucan::BundleEmitter`] and
//! [`hyprstream_rpc::auth::ucan::FaithfulnessCheck`]. It lives in the `hyprstream`
//! crate (not `hyprstream-rpc`) because it needs BOTH the M1 UCAN model (the lower
//! crate) AND S4's [`TeMatrix`] / [`CompiledPolicy`] (this crate) — exactly the
//! crate-layering the seam docs predicted. The seam *traits* stay abstract in
//! `hyprstream-rpc`; the concrete lowering is here.
//!
//! ## What is in scope (vocab-INDEPENDENT) and what is deferred
//!
//! The compiler turns a structurally-validated UCAN ceiling
//! (`ucan.capabilities()`, already proven ⊆ every ancestor and live at `now` by
//! [`hyprstream_rpc::auth::ucan::chain::validate`]) into a [`TeMatrix`] allow-set.
//! Doing that *concretely* requires knowing what a UCAN
//! [`Ability`](hyprstream_rpc::auth::ucan::capability::Ability) /
//! [`Resource`](hyprstream_rpc::auth::ucan::capability::Resource) MEANS to the
//! enforcement layer — i.e. which [`SubjectType`] / [`ObjectType`] / [`Action`] ids
//! it lowers to. That mapping is S3's `ActionVocabulary`
//! ([`hyprstream_rpc::auth::ucan::ActionVocabulary`]), gated on #582, and is the
//! ONE piece this milestone defers.
//!
//! To stay vocab-independent while still being a *complete, testable* emitter, the
//! lowering is **parameterized over an injected [`CapabilityVocab`]** — a pure,
//! deterministic function `Capability → Set<TeRule>`. The structural emitter and
//! the reference interpreter consume the *same* injected map, so faithfulness is
//! proven over the structural fragment for ANY vocabulary the eventual
//! `ActionVocabulary` supplies. When #582 lands (name-stable; only ordinals
//! regroup), the concrete `ActionVocabulary` adapts into a `CapabilityVocab` with
//! NO change to the emitter or the proofs here.
//!
//! ## The faithfulness checks (design §14, S5 milestone-2 deliverable)
//!
//! 1. **Reference interpreter** ([`ReferenceInterpreter`]) — decides a `(resource,
//!    ability, caveats)` request DIRECTLY from the validated UCAN capability set
//!    via the M1 attenuation predicate
//!    ([`Capability::authorizes`](hyprstream_rpc::auth::ucan::capability::Capability::authorizes)).
//!    The spec oracle; it never looks at the compiled bundle.
//! 2. **SET-equality faithfulness** ([`Faithfulness`]) — proves the emitted
//!    allow-set EQUALS the vocab-image of the ceiling, over ALL rules (not a
//!    sample): soundness = no-widening, completeness = no-drop.
//! 3. **Differential property tests** — generate a request space and assert the
//!    compiled [`TeMatrix`] evaluation AGREES with the reference interpreter on
//!    every request (see the `tests` module / `proptest`, 512 cases).
//! 4. **SMT structural-equivalence** ([`smt`]) — scaffolded interface; no usable
//!    z3/SMT crate is vendored, so this is explicitly DEFERRED (the differential
//!    tests + the set-equality proof are the must-have faithfulness guarantees).
//!
//! ## Determinism + no-widening (the two load-bearing emitter obligations)
//!
//! - **Deterministic**: same UCAN + same `CapabilityVocab` + same lattice
//!   generation ⇒ a byte-identical [`CompiledPolicy`] ⇒ identical
//!   [`CompiledPolicy::policy_hash`]. We collect rules into a `BTreeSet`
//!   (order-stable) and emit through S4's canonical encoder.
//! - **No-widening (⊆)**: every [`TeRule`] in the emitted matrix is the image,
//!   under the injected vocab, of SOME capability in the validated ceiling. The
//!   emitter NEVER invents a rule. [`Faithfulness::check`] re-derives the image and
//!   confirms set-equality.
//!
//! ## Fail-closed signing
//!
//! [`compile_and_bind`] is the ONE entry a producer calls: it emits, runs
//! [`Faithfulness::check`], and ONLY if the check passes builds the
//! [`hyprstream_rpc::auth::ucan::approval::ApprovalBinding`] and signs it. An
//! unfaithful bundle is NEVER signed (design §14 containment).

use std::collections::{BTreeSet, HashSet};

use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::ucan::approval::{ApprovalBinding, ApprovalError, SignedApproval};
use hyprstream_rpc::auth::ucan::capability::{Ability, Capability, Caveats, Resource};
use hyprstream_rpc::auth::ucan::seam::{BundleEmitter, FaithfulnessCheck};
use hyprstream_rpc::auth::ucan::token::Ucan;
use hyprstream_rpc::crypto::pq::MlDsaSigningKey;

use crate::mac::compiled::{CompiledPolicy, PolicyDistError};
use crate::mac::lattice::Lattice;
use crate::mac::te::{Decision, TeMatrix, TeRule};

/// A pure, deterministic mapping from a UCAN [`Capability`] to the set of
/// enforcement [`TeRule`]s it authorizes — the **vocabulary seam, injected**.
///
/// This is the vocab-independent stand-in for the deferred
/// [`hyprstream_rpc::auth::ucan::ActionVocabulary`] (#582). The concrete
/// `ActionVocabulary` will be adapted into one of these (it maps a single
/// `Ability`/`Resource` to one `Action`/`ObjectType`); for milestone 2 the compiler
/// and its faithfulness checks are generic over ANY implementation, so the proofs
/// hold for whatever vocabulary lands.
///
/// ## Contract (fail-closed)
///
/// - `rules_for` MUST be a pure function of its input (no interior mutability, no
///   clock/RNG): determinism of the emitted bundle depends on it.
/// - An UNRECOGNIZED capability (unknown verb / resource class) MUST yield an
///   EMPTY set — the compiler then grants nothing for it. It MUST NOT yield a
///   permissive/wildcard rule (that would widen). This mirrors the
///   `ActionVocabulary` "`action_of` `None` ⇒ reject" intent in the additive
///   direction: an un-mappable grant contributes no authority. When the real
///   `ActionVocabulary` lands, its `None` ⇒ reject maps directly onto
///   "contributes empty set", so no permissive default can ever leak.
/// - The emitted [`TeRule`]s carry the fully-resolved
///   [`SubjectType`]/[`ObjectType`]/[`Action`] ids; the injected map decides the
///   subject domain a delegation is enforced under (the real `ActionVocabulary`
///   adapter resolves it from the audience/context). Keeping it inside `rules_for`
///   keeps the seam to a single method.
pub trait CapabilityVocab {
    /// The enforcement triples a single capability authorizes. Empty ⇒ the
    /// capability contributes no authority (fail-closed). MUST be deterministic.
    fn rules_for(&self, cap: &Capability) -> BTreeSet<TeRule>;
}

/// An emitted bundle: the compiled, signable artifact plus the order-stable allow
/// set the emitter built (for cheap subset checks).
///
/// Not `PartialEq`/`Eq`: the embedded S4 [`CompiledPolicy`] is not comparable (its
/// canonical identity is [`CompiledPolicy::policy_hash`], not field equality).
/// Compare bundles by their `allow` set or `bundle_hash()` instead.
#[derive(Debug, Clone)]
pub struct EmittedBundle {
    /// The compiled, signable artifact (matrix + embedded lattice + generation).
    pub policy: CompiledPolicy,
    /// The order-stable set of allow rules actually emitted. Mirrors
    /// `policy.matrix` but as the deterministic `BTreeSet` the emitter built.
    pub allow: BTreeSet<TeRule>,
}

impl EmittedBundle {
    /// The bundle hash the [`ApprovalBinding`] binds — exactly S4's
    /// [`CompiledPolicy::policy_hash`]. There is NO second hash implementation:
    /// this forwards to S4's own method, so the emit-side hash and the loader-side
    /// hash are byte-identical (no cross-crate domain drift).
    pub fn bundle_hash(&self) -> Result<[u8; 32], PolicyDistError> {
        self.policy.policy_hash()
    }
}

/// Errors from bundle emission. All fail-closed.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// The lattice generation used for emission could not be reconciled with the
    /// compiled policy (defensive — `CompiledPolicy::new` ties them, so this is
    /// unreachable, but we never `unwrap` in TCB code).
    #[error("lattice/policy generation reconciliation failed")]
    Generation,
}

/// The concrete UCAN → TE [`BundleEmitter`]. Lowers a *validated* UCAN ceiling
/// into a [`CompiledPolicy`] over the supplied lattice, using the injected
/// [`CapabilityVocab`]. Deterministic and subset-bounded by construction.
pub struct TeBundleEmitter<'a, V: CapabilityVocab> {
    lattice: &'a Lattice,
    vocab: &'a V,
}

impl<'a, V: CapabilityVocab> TeBundleEmitter<'a, V> {
    /// Construct against the S1 lattice (whose version becomes the bundle
    /// generation) and the injected vocabulary.
    pub fn new(lattice: &'a Lattice, vocab: &'a V) -> Self {
        Self { lattice, vocab }
    }

    /// Lower a UCAN ceiling — `caps` is `ucan.capabilities()`, already proven ⊆
    /// every ancestor by `chain::validate`. Collects the image of every capability
    /// under the vocab into an order-stable allow set; no rule is emitted that is
    /// not justified by some capability.
    fn lower(&self, caps: &[Capability]) -> EmittedBundle {
        // Deterministic accumulation: BTreeSet → stable iteration & dedup.
        let mut allow: BTreeSet<TeRule> = BTreeSet::new();
        for cap in caps {
            // The image of THIS capability. Union into the allow set; an unmapped
            // capability contributes nothing (fail-closed, never widens).
            for rule in self.vocab.rules_for(cap) {
                allow.insert(rule);
            }
        }
        // Hand S4 a HashSet (its hot-path representation); the canonical encoder
        // re-sorts, so determinism is preserved through the hash.
        let matrix = TeMatrix::from_allow(allow.iter().copied().collect::<HashSet<_>>());
        let policy = CompiledPolicy::new(matrix, self.lattice);
        EmittedBundle { policy, allow }
    }
}

impl<V: CapabilityVocab> BundleEmitter for TeBundleEmitter<'_, V> {
    type Bundle = EmittedBundle;
    type Error = EmitError;

    fn emit(&self, validated_ucan: &Ucan) -> Result<Self::Bundle, Self::Error> {
        let bundle = self.lower(validated_ucan.capabilities());
        // Defensive generation reconciliation (see EmitError::Generation docs).
        if bundle.policy.generation != self.lattice.version().generation() {
            return Err(EmitError::Generation);
        }
        Ok(bundle)
    }
}

// ---------------------------------------------------------------------------
// Reference interpreter — the faithfulness SPEC ORACLE.
// ---------------------------------------------------------------------------

/// A request the reference interpreter / differential tests decide: a concrete
/// `(resource, ability, caveats)` access, exactly the shape a delegate would
/// present and the shape `Capability::authorizes` judges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessRequest {
    /// The resource being accessed.
    pub resource: Resource,
    /// The ability/verb requested.
    pub ability: Ability,
    /// Caveats the requester satisfies (its restriction context).
    pub caveats: Caveats,
}

impl AccessRequest {
    /// As a [`Capability`] — the request IS a claimed capability the ceiling must
    /// authorize.
    fn as_capability(&self) -> Capability {
        Capability::with_caveats(
            self.resource.clone(),
            self.ability.clone(),
            self.caveats.clone(),
        )
    }
}

/// Interprets a validated UCAN ceiling DIRECTLY, independent of any compiled
/// bundle. The *intent* against which the compiled [`TeMatrix`] is differentially
/// checked. It NEVER consults a `TeMatrix` — it is the spec.
///
/// Decision rule (the M1 ceiling semantics, restated as allow/deny over requests):
/// a request is ALLOWED iff (a) the UCAN is temporally valid at the decision clock
/// AND (b) some capability in the validated ceiling authorizes it on all three
/// axes (resource, ability, caveats). Otherwise DENY. Default-deny, fail-closed.
///
/// ## Temporal parity with the M1 chain validator
///
/// M1's `chain::validate(ucan, verifier, now)` enforces **absolute** expiry /
/// `not_before` on every link via
/// [`Ucan::is_valid_at`](hyprstream_rpc::auth::ucan::token::Ucan::is_valid_at)
/// before a ceiling is ever produced. For the interpreter to be a faithful oracle
/// of a decision at time `now`, it applies the SAME absolute-temporal gate via
/// [`decide_at`](ReferenceInterpreter::decide_at). Build with
/// [`from_validated`](ReferenceInterpreter::from_validated) to capture the leaf's
/// window; [`from_ceiling`](ReferenceInterpreter::from_ceiling) is time-unbounded
/// for tests isolating the structural fragment.
pub struct ReferenceInterpreter {
    /// The validated ceiling capabilities (`ucan.capabilities()`).
    ceiling: Vec<Capability>,
    /// Earliest valid unix second of the leaf (`not_before`), if bounded.
    not_before: Option<u64>,
    /// Latest valid unix second of the leaf (`expiration`), if bounded.
    expiration: Option<u64>,
}

impl ReferenceInterpreter {
    /// Build from a UCAN whose chain ALREADY passed `chain::validate`. The leaf's
    /// own capabilities ARE the effective ceiling (most attenuated, proven ⊆ every
    /// ancestor). Captures the leaf's validity window so [`decide_at`] applies the
    /// same absolute-temporal gate the chain validator did.
    ///
    /// [`decide_at`]: ReferenceInterpreter::decide_at
    pub fn from_validated(ucan: &Ucan) -> Self {
        Self {
            ceiling: ucan.capabilities().to_vec(),
            not_before: ucan.payload.not_before,
            expiration: ucan.payload.expiration,
        }
    }

    /// Build directly from a capability ceiling, time-unbounded (used by property
    /// tests that isolate the structural fragment).
    pub fn from_ceiling(ceiling: Vec<Capability>) -> Self {
        Self {
            ceiling,
            not_before: None,
            expiration: None,
        }
    }

    /// Is the ceiling temporally valid at `now`? Mirrors `Ucan::is_valid_at`.
    #[must_use]
    fn is_valid_at(&self, now: u64) -> bool {
        if let Some(nbf) = self.not_before {
            if now < nbf {
                return false;
            }
        }
        if let Some(exp) = self.expiration {
            if now > exp {
                return false;
            }
        }
        true
    }

    /// The spec decision at clock `now`: ALLOW iff the ceiling is temporally valid
    /// at `now` AND authorizes the request. Default-deny, fail-closed.
    #[must_use]
    pub fn decide_at(&self, req: &AccessRequest, now: u64) -> Decision {
        if !self.is_valid_at(now) {
            return Decision::Deny;
        }
        self.decide(req)
    }

    /// Structural-fragment decision, ignoring time (the temporal gate is assumed
    /// already passed). Equivalent to `decide_at` at a `now` inside the window.
    /// Used by the differential tests isolating the authority gate.
    #[must_use]
    pub fn decide(&self, req: &AccessRequest) -> Decision {
        let claimed = req.as_capability();
        if self.ceiling.iter().any(|h| h.authorizes(&claimed)) {
            Decision::Permit
        } else {
            Decision::Deny
        }
    }
}

// ---------------------------------------------------------------------------
// FaithfulnessCheck — structural soundness + no-widening over the emitted bundle.
// ---------------------------------------------------------------------------

/// A point where the emitted bundle and the UCAN ceiling disagree — a
/// counterexample. Either the bundle invented authority the ceiling does not grant
/// (a WIDENING — the unforgivable direction), or, for the auditable structural
/// fragment, the bundle dropped authority the ceiling does grant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Counterexample {
    /// The emitted matrix contains a rule no ceiling capability justifies under
    /// the vocab — the bundle WIDENS. Fail-closed: such a bundle MUST NOT be
    /// signed.
    Widened {
        /// The offending emitted rule with no ceiling provenance.
        rule: TeRule,
    },
    /// A ceiling capability maps (under the vocab) to a rule the emitted matrix
    /// omits — the bundle is incomplete on the structural fragment.
    Dropped {
        /// A rule the ceiling justifies but the bundle omitted.
        rule: TeRule,
    },
}

/// The faithfulness checker: confirms an [`EmittedBundle`] grants EXACTLY the
/// structural authority the validated UCAN ceiling does, under the same injected
/// [`CapabilityVocab`] the emitter used.
///
/// - **Soundness (no-widening, the security-critical direction)**: every rule in
///   the emitted matrix must be in the vocab-image of the ceiling. A widened rule
///   is a [`Counterexample::Widened`].
/// - **Completeness (structural fragment)**: every rule the vocab-image of the
///   ceiling contains must be in the emitted matrix. A dropped rule is a
///   [`Counterexample::Dropped`].
///
/// Together these make the emitted matrix EQUAL to the vocab-image of the ceiling —
/// the strongest structural faithfulness statement available without the
/// caveat-value algebra (deferred), over ALL rules (not a sample). The differential
/// property tests check the *request-level* agreement this set-equality implies.
pub struct Faithfulness<'a, V: CapabilityVocab> {
    vocab: &'a V,
}

impl<'a, V: CapabilityVocab> Faithfulness<'a, V> {
    /// Build against the same vocab the emitter used.
    pub fn new(vocab: &'a V) -> Self {
        Self { vocab }
    }

    /// The vocab-image of a ceiling: the union of `rules_for` over every
    /// capability. This is the set the emitted matrix MUST equal.
    fn ceiling_image(&self, ceiling: &[Capability]) -> BTreeSet<TeRule> {
        let mut image = BTreeSet::new();
        for cap in ceiling {
            for rule in self.vocab.rules_for(cap) {
                image.insert(rule);
            }
        }
        image
    }
}

impl<V: CapabilityVocab> FaithfulnessCheck for Faithfulness<'_, V> {
    type Bundle = EmittedBundle;
    type Counterexample = Counterexample;

    fn check(&self, ucan: &Ucan, bundle: &Self::Bundle) -> Result<(), Self::Counterexample> {
        let image = self.ceiling_image(ucan.capabilities());
        // Soundness: no emitted rule outside the ceiling image (no widening).
        for rule in &bundle.allow {
            if !image.contains(rule) {
                return Err(Counterexample::Widened { rule: *rule });
            }
        }
        // Completeness (structural fragment): no ceiling-image rule omitted.
        for rule in &image {
            if !bundle.allow.contains(rule) {
                return Err(Counterexample::Dropped { rule: *rule });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fail-closed signing gate: compile + faithfulness + bind + sign, atomically.
// ---------------------------------------------------------------------------

/// Errors from the [`compile_and_bind`] fail-closed signing path.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// Emission failed.
    #[error("bundle emission failed: {0}")]
    Emit(#[from] EmitError),
    /// The emitted bundle is NOT faithful to the UCAN ceiling — it widened or
    /// dropped authority. NO approval is signed (fail-closed, design §14).
    #[error("faithfulness check failed: {0:?}")]
    Unfaithful(Counterexample),
    /// Hashing the bundle for the binding failed.
    #[error("bundle hash failed: {0}")]
    Hash(#[from] PolicyDistError),
    /// Building / signing the approval binding failed.
    #[error("approval binding failed: {0}")]
    Approval(#[from] ApprovalError),
}

/// THE producer entry point — emit a bundle from a validated UCAN ceiling, prove
/// it faithful, and ONLY THEN bind + sign the approval. Fail-closed: if the
/// [`Faithfulness`] check returns a [`Counterexample`], this returns
/// [`CompileError::Unfaithful`] and **no signature is produced** — an unfaithful
/// bundle is never signed (design §14 containment backstop).
///
/// Steps:
/// 1. `emitter.emit(ucan)` → [`EmittedBundle`].
/// 2. `Faithfulness::check(ucan, &bundle)` — MUST be `Ok`, else fail-closed.
/// 3. `ApprovalBinding::new(ucan, generation, bundle.bundle_hash())` — binds the
///    UCAN CID, the lattice generation, and the recomputed S4 `policy_hash()`.
/// 4. `SignedApproval::sign(binding, ed_sk, pq_sk)` — hybrid COSE (never
///    Ed25519-only).
///
/// Returns the emitted bundle (for distribution) and the signed approval (for the
/// loader). The caller is expected to have run `chain::validate(ucan, verifier,
/// now)` already (the contract on `BundleEmitter::emit`).
///
/// The `caveats`-value algebra is deferred, so faithfulness is over the structural
/// fragment; the SMT proof ([`smt`]) that would discharge the "for all requests"
/// quantifier is scaffolded/deferred — `SolverUnavailable` is NOT a hard fail here
/// because the [`Faithfulness::check`] set-equality is itself a proof over ALL
/// rules, which is the actual soundness gate on the signature.
pub fn compile_and_bind<V: CapabilityVocab>(
    ucan: &Ucan,
    lattice: &Lattice,
    vocab: &V,
    ed_sk: &SigningKey,
    pq_sk: &MlDsaSigningKey,
) -> Result<(EmittedBundle, SignedApproval), CompileError> {
    let emitter = TeBundleEmitter::new(lattice, vocab);
    let bundle = emitter.emit(ucan)?;

    // FAIL-CLOSED GATE: an unfaithful bundle is never signed.
    if let Err(ce) = Faithfulness::new(vocab).check(ucan, &bundle) {
        return Err(CompileError::Unfaithful(ce));
    }

    let bundle_hash = bundle.bundle_hash()?;
    let binding = ApprovalBinding::new(ucan, bundle.policy.generation, bundle_hash)?;
    let signed = SignedApproval::sign(binding, ed_sk, pq_sk)?;
    Ok((bundle, signed))
}

// ---------------------------------------------------------------------------
// SMT structural-equivalence — SCAFFOLD ONLY (deferred: no solver vendored).
// ---------------------------------------------------------------------------

/// **DEFERRED sub-piece.** SMT-backed proof of emitter ≡ reference-interpreter over
/// the structural (vocabulary-closed, caveat-value-blind) fragment — the proof that
/// would discharge the "for all requests" quantifier the differential property
/// tests only sample.
///
/// No usable z3/SMT crate is vendored in this workspace (only transitive
/// `proptest`/`quickcheck`, no `z3`/`rsmt2`), so this is intentionally a SCAFFOLD:
/// the interface is pinned and the encoding is documented, but the milestone is NOT
/// blocked on it — the differential `proptest` suite + the set-equality
/// [`Faithfulness::check`] (which proves over ALL rules) are the faithfulness
/// guarantees.
pub mod smt {
    use super::{Counterexample, EmittedBundle};
    use hyprstream_rpc::auth::ucan::token::Ucan;

    /// Outcome of an SMT structural-equivalence attempt.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum SmtResult {
        /// Proven equivalent over the structural fragment.
        Equivalent,
        /// A model witnessing disagreement (resource/ability containment).
        Counterexample(Counterexample),
        /// No solver available — the proof is deferred (NOT a failure; callers
        /// fall back to the differential property tests + set-equality check).
        SolverUnavailable,
    }

    /// The SMT equivalence prover seam. When a solver lands, an implementor encodes
    /// the structural fragment:
    /// - resource/ability containment as the M1 `covers` relation over a finite
    ///   symbol alphabet (string-prefix / namespace containment is decidable over
    ///   the closed vocabulary);
    /// - the emitted [`TeMatrix`](crate::mac::te::TeMatrix) allow-set as a boolean
    ///   function of `(subject, object, action)`;
    /// - asserts `emitter_output(req) ⇔ reference_interpreter(req)` is valid (UNSAT
    ///   of the negation), ignoring opaque caveat VALUES (the deferred algebra).
    ///
    /// The default impl returns [`SmtResult::SolverUnavailable`] — the honest state
    /// today.
    pub trait StructuralEquivalence {
        /// Attempt to prove the bundle structurally equivalent to the UCAN ceiling.
        fn prove(&self, ucan: &Ucan, bundle: &EmittedBundle) -> SmtResult;
    }

    /// The no-op prover used until a solver is vendored. Always defers.
    pub struct DeferredProver;

    impl StructuralEquivalence for DeferredProver {
        fn prove(&self, _ucan: &Ucan, _bundle: &EmittedBundle) -> SmtResult {
            SmtResult::SolverUnavailable
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mac::compiled::{
        sign_policy, PolicyApproval, PolicyLoader, PolicySigner, PolicyVerifier,
    };
    use crate::mac::lattice::{Compartment, LatticeVersion};
    use crate::mac::te::{Action as TeAction, ObjectType, SubjectType};
    use hyprstream_rpc::auth::ucan::capability::{CaveatValue, Caveats};
    use hyprstream_rpc::auth::ucan::token::{Did, UcanPayload};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    // ---- A concrete, deterministic test vocabulary --------------------------
    //
    // Stands in for the deferred S3 ActionVocabulary over a CLOSED alphabet:
    // resources `mac://model/<name>` (+ wildcard `mac://model/*`) → ObjectType;
    // abilities `model/read` → Action(0), `model/write` → Action(1). A wildcard
    // resource FANS OUT to the full closed set of concrete objects (so the emitter
    // enumerates, never emits a wildcard rule). Unknown verbs/resources → empty
    // (fail-closed).

    const OBJECTS: &[&str] = &["qwen", "llama", "mistral"];

    fn object_id(name: &str) -> Option<ObjectType> {
        OBJECTS
            .iter()
            .position(|o| *o == name)
            .map(|i| ObjectType(i as u32))
    }

    fn action_id(verb: &str) -> Option<TeAction> {
        match verb {
            "model/read" => Some(TeAction(0)),
            "model/write" => Some(TeAction(1)),
            _ => None,
        }
    }

    struct TestVocab {
        subject: SubjectType,
    }

    impl CapabilityVocab for TestVocab {
        fn rules_for(&self, cap: &Capability) -> BTreeSet<TeRule> {
            let mut out = BTreeSet::new();
            // Resolve the action verb; unknown → no rules (fail-closed).
            let Some(action) = action_id(cap.ability.as_str()) else {
                return out;
            };
            let res = cap.resource.as_str();
            // Resolve the resource into one or more concrete objects. A wildcard
            // fans out across the closed alphabet (emitter must enumerate).
            let concrete: Vec<&str> = if let Some(rest) = res.strip_prefix("mac://model/") {
                if rest == "*" {
                    OBJECTS.to_vec()
                } else if OBJECTS.contains(&rest) {
                    vec![rest]
                } else {
                    vec![]
                }
            } else {
                vec![]
            };
            for name in concrete {
                if let Some(object_type) = object_id(name) {
                    out.insert(TeRule {
                        subject_type: self.subject,
                        object_type,
                        action,
                    });
                }
            }
            out
        }
    }

    fn vocab() -> TestVocab {
        TestVocab {
            subject: SubjectType(1),
        }
    }

    fn lattice(gen: u32) -> Lattice {
        Lattice::new(
            LatticeVersion(gen),
            [Compartment::new("pii"), Compartment::new("finance")],
        )
    }

    fn cap(res: &str, ab: &str) -> Capability {
        Capability::new(Resource::new(res), Ability::new(ab))
    }

    /// Build a self-issued (root) UCAN holding `caps`. Uses a real hybrid identity
    /// so signing in `compile_and_bind` tests works; the emitter/faithfulness only
    /// read `capabilities()`, but the signing-gate tests need a valid key.
    struct Identity {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        pq_vk: hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
        did: Did,
    }

    fn identity() -> Identity {
        use ed25519_dalek::SigningKey as Sk;
        use hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair;
        use rand::rngs::OsRng;
        let ed_sk = Sk::generate(&mut OsRng);
        let did = Did::from_ed25519(&ed_sk.verifying_key().to_bytes());
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        Identity {
            ed_sk,
            pq_sk,
            pq_vk,
            did,
        }
    }

    fn ucan_with(caps: Vec<Capability>) -> Ucan {
        let did = Did::from_ed25519(&[0u8; 32]);
        Ucan {
            payload: UcanPayload {
                issuer: did.clone(),
                audience: did,
                capabilities: caps,
                not_before: None,
                expiration: Some(9_999_999_999),
                nonce: vec![],
            },
            proofs: vec![],
            signature: vec![],
        }
    }

    /// A root UCAN issued by a real identity (for the signing-gate tests).
    fn ucan_signed_by(id: &Identity, caps: Vec<Capability>) -> Ucan {
        use hyprstream_rpc::crypto::cose_sign::sign_composite;
        let payload = UcanPayload {
            issuer: id.did.clone(),
            audience: id.did.clone(),
            capabilities: caps,
            not_before: None,
            expiration: Some(9_999_999_999),
            nonce: vec![],
        };
        let bytes = payload.signing_bytes().unwrap();
        let signature = sign_composite(
            &id.ed_sk,
            Some(&id.pq_sk),
            &bytes,
            hyprstream_rpc::auth::ucan::APPROVAL_AAD, // any AAD; not re-verified here
        )
        .unwrap();
        Ucan {
            payload,
            proofs: vec![],
            signature,
        }
    }

    // ---- Emitter: no-widening (⊆ ceiling) -----------------------------------

    #[test]
    fn emit_is_subset_of_ceiling_no_widening() {
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let bundle = emitter.emit(&u).unwrap();
        assert_eq!(Faithfulness::new(&v).check(&u, &bundle), Ok(()));
        assert_eq!(
            bundle.allow,
            BTreeSet::from([TeRule {
                subject_type: SubjectType(1),
                object_type: ObjectType(0),
                action: TeAction(0),
            }])
        );
    }

    #[test]
    fn faithfulness_detects_injected_widening() {
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let mut bundle = emitter.emit(&u).unwrap();
        let rogue = TeRule {
            subject_type: SubjectType(1),
            object_type: ObjectType(1),
            action: TeAction(1),
        };
        bundle.allow.insert(rogue);
        assert_eq!(
            Faithfulness::new(&v).check(&u, &bundle),
            Err(Counterexample::Widened { rule: rogue })
        );
    }

    #[test]
    fn faithfulness_detects_injected_drop() {
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let mut bundle = emitter.emit(&u).unwrap();
        bundle.allow.clear();
        assert!(matches!(
            Faithfulness::new(&v).check(&u, &bundle),
            Err(Counterexample::Dropped { .. })
        ));
    }

    #[test]
    fn wildcard_ceiling_fans_out_no_wildcard_rule() {
        // A `mac://model/*` ceiling enumerates every concrete object — the emitted
        // matrix has interned ids, never a wildcard, so it cannot widen to a future
        // object the ceiling never reviewed.
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/*", "model/read")]);
        let bundle = emitter.emit(&u).unwrap();
        assert_eq!(bundle.allow.len(), OBJECTS.len());
        assert_eq!(Faithfulness::new(&v).check(&u, &bundle), Ok(()));
    }

    // ---- TE-soundness test (a): default-deny on unmapped / un-imaged ---------

    #[test]
    fn te_soundness_unmapped_capability_contributes_nothing() {
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        // Unknown verb AND unknown resource both contribute zero rules.
        let u = ucan_with(vec![
            cap("mac://model/qwen", "admin/superuser"),
            cap("mac://other/x", "model/read"),
        ]);
        let bundle = emitter.emit(&u).unwrap();
        assert!(bundle.allow.is_empty(), "unmapped grants emit no rules");
        assert_eq!(Faithfulness::new(&v).check(&u, &bundle), Ok(()));
    }

    #[test]
    fn te_soundness_default_deny_on_unmapped_request() {
        // A request whose (resource, ability) maps to no rule hits default-deny in
        // the matrix — even with a permissive ceiling.
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/*", "model/read")]);
        let bundle = emitter.emit(&u).unwrap();
        // Unmapped verb → no image rule → Deny.
        let req = AccessRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("admin/x"),
            caveats: Caveats::empty(),
        };
        assert_eq!(
            matrix_decides(&v, &bundle.policy.matrix, &req),
            Decision::Deny
        );
        // Unmapped resource → no image rule → Deny.
        let req2 = AccessRequest {
            resource: Resource::new("mac://other/z"),
            ability: Ability::new("model/read"),
            caveats: Caveats::empty(),
        };
        assert_eq!(
            matrix_decides(&v, &bundle.policy.matrix, &req2),
            Decision::Deny
        );
    }

    // ---- TE-soundness test (c): well-formedness (no malformed/contradiction) -

    #[test]
    fn te_soundness_model_well_formed_no_contradiction() {
        // The emitter only ever produces ALLOW rules (escalate is empty), so a
        // triple can never be simultaneously allow AND escalate (no contradiction),
        // and default-deny is structurally preserved (absence ⇒ Deny). Every
        // emitted rule is in the ceiling image (no malformed/invented id).
        let v = vocab();
        let l = lattice(3);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![
            cap("mac://model/qwen", "model/read"),
            cap("mac://model/*", "model/write"),
        ]);
        let bundle = emitter.emit(&u).unwrap();
        let (allow, escalate) = bundle.policy.matrix.sorted_rules();
        // No escalate band emitted by the compiler.
        assert!(
            escalate.is_empty(),
            "emitter must not produce an escalate band"
        );
        // allow ∩ escalate = ∅ (trivially, escalate empty — assert explicitly).
        let allow_set: BTreeSet<_> = allow.iter().copied().collect();
        let esc_set: BTreeSet<_> = escalate.iter().copied().collect();
        assert!(
            allow_set.is_disjoint(&esc_set),
            "allow ∩ escalate must be ∅"
        );
        // Every emitted rule is in the ceiling image (no rule outside the ceiling).
        let image = {
            let mut s = BTreeSet::new();
            for c in u.capabilities() {
                s.extend(v.rules_for(c));
            }
            s
        };
        for r in &allow_set {
            assert!(
                image.contains(r),
                "no emitted rule may lie outside the ceiling image"
            );
        }
    }

    // ---- Emitter: determinism (byte-identical bundle ⇒ identical hash) ------

    #[test]
    fn emit_is_deterministic_byte_identical_hash() {
        let v = vocab();
        let l = lattice(5);
        let emitter = TeBundleEmitter::new(&l, &v);
        // Same caps, DIFFERENT insertion order — same hash.
        let u1 = ucan_with(vec![
            cap("mac://model/qwen", "model/read"),
            cap("mac://model/llama", "model/write"),
        ]);
        let u2 = ucan_with(vec![
            cap("mac://model/llama", "model/write"),
            cap("mac://model/qwen", "model/read"),
        ]);
        let h1 = emitter.emit(&u1).unwrap().bundle_hash().unwrap();
        let h2 = emitter.emit(&u2).unwrap().bundle_hash().unwrap();
        assert_eq!(h1, h2, "emission must be order-independent / deterministic");
    }

    #[test]
    fn emit_hash_matches_compiled_policy_hash() {
        // The bundle_hash the approval binds IS CompiledPolicy::policy_hash().
        let v = vocab();
        let l = lattice(7);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let bundle = emitter.emit(&u).unwrap();
        assert_eq!(
            bundle.bundle_hash().unwrap(),
            bundle.policy.policy_hash().unwrap()
        );
    }

    #[test]
    fn emit_generation_bound_to_lattice() {
        let v = vocab();
        let l = lattice(42);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let bundle = emitter.emit(&u).unwrap();
        assert_eq!(bundle.policy.generation, 42);
    }

    // ---- Reference interpreter ---------------------------------------------

    #[test]
    fn reference_interpreter_decides_via_attenuation() {
        let interp = ReferenceInterpreter::from_ceiling(vec![cap("mac://model/*", "model/read")]);
        assert_eq!(
            interp.decide(&AccessRequest {
                resource: Resource::new("mac://model/qwen"),
                ability: Ability::new("model/read"),
                caveats: Caveats::empty(),
            }),
            Decision::Permit
        );
        assert_eq!(
            interp.decide(&AccessRequest {
                resource: Resource::new("mac://model/qwen"),
                ability: Ability::new("model/write"),
                caveats: Caveats::empty(),
            }),
            Decision::Deny
        );
    }

    #[test]
    fn reference_interpreter_respects_caveats() {
        let mut cv = BTreeMap::new();
        cv.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let restricted = Capability::with_caveats(
            Resource::new("mac://model/*"),
            Ability::new("model/read"),
            Caveats(cv.clone()),
        );
        let interp = ReferenceInterpreter::from_ceiling(vec![restricted]);
        // Request WITHOUT the caveat is NOT covered (would widen) → deny.
        assert_eq!(
            interp.decide(&AccessRequest {
                resource: Resource::new("mac://model/qwen"),
                ability: Ability::new("model/read"),
                caveats: Caveats::empty(),
            }),
            Decision::Deny
        );
        // Request satisfying the caveat → permit.
        assert_eq!(
            interp.decide(&AccessRequest {
                resource: Resource::new("mac://model/qwen"),
                ability: Ability::new("model/read"),
                caveats: Caveats(cv),
            }),
            Decision::Permit
        );
    }

    #[test]
    fn reference_interpreter_temporal_gate_parity() {
        // Parity with chain::validate(ucan, verifier, now): an expired /
        // not-yet-valid ceiling denies even a structurally-covered request.
        let mut u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        u.payload.not_before = Some(100);
        u.payload.expiration = Some(200);
        let interp = ReferenceInterpreter::from_validated(&u);
        let req = AccessRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("model/read"),
            caveats: Caveats::empty(),
        };
        assert_eq!(interp.decide_at(&req, 150), Decision::Permit);
        assert_eq!(interp.decide_at(&req, 99), Decision::Deny);
        assert_eq!(interp.decide_at(&req, 201), Decision::Deny);
        // Inclusive boundaries (mirrors Ucan::is_valid_at).
        assert_eq!(interp.decide_at(&req, 100), Decision::Permit);
        assert_eq!(interp.decide_at(&req, 200), Decision::Permit);
    }

    // ---- Fail-closed compile_and_bind --------------------------------------

    #[test]
    fn compile_and_bind_signs_only_faithful_bundle() {
        let v = vocab();
        let l = lattice(11);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        // Faithful by construction → signs.
        let (bundle, signed) = compile_and_bind(&u, &l, &v, &id.ed_sk, &id.pq_sk).unwrap();
        // The signed approval verifies and binds the exact (UCAN, generation, hash).
        signed
            .verify_binds(
                &id.ed_sk.verifying_key(),
                &id.pq_vk,
                &u,
                bundle.policy.generation,
                &bundle.bundle_hash().unwrap(),
            )
            .unwrap();
    }

    #[test]
    fn compile_and_bind_refuses_unfaithful_bundle() {
        // The fail-closed gate keys on EXACTLY `Faithfulness::check`. With a single
        // self-consistent vocab the emitter and the check agree by construction (an
        // emitter that lowers via the same vocab the check re-derives from is always
        // faithful) — which is the CORRECT behavior. The security property is: the
        // gate's PREDICATE rejects any bundle whose allow-set is not the ceiling
        // image, so a widened bundle is never signed. We prove the predicate that
        // `compile_and_bind` branches on: an injected widening yields
        // `Counterexample::Widened`, i.e. `compile_and_bind` would return
        // `CompileError::Unfaithful` and produce NO SignedApproval.
        let v = vocab();
        let l = lattice(3);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let mut bundle = TeBundleEmitter::new(&l, &v).emit(&u).unwrap();
        // Inject a rule no ceiling capability justifies (a widening / buggy emitter).
        bundle.allow.insert(TeRule {
            subject_type: SubjectType(7),
            object_type: ObjectType(7),
            action: TeAction(7),
        });
        match Faithfulness::new(&v).check(&u, &bundle) {
            Err(Counterexample::Widened { .. }) => {
                // Map through the SAME branch compile_and_bind takes, to show the
                // gate yields CompileError::Unfaithful (no signature path reached).
                let ce = Faithfulness::new(&v).check(&u, &bundle).unwrap_err();
                let gate: Result<(), CompileError> = Err(CompileError::Unfaithful(ce));
                assert!(matches!(gate, Err(CompileError::Unfaithful(_))));
            }
            other => panic!("expected Widened counterexample, got {other:?}"),
        }
    }

    // ---- from_verified keystone + tamper test (S5 → S4 wiring) --------------

    /// Stub signer/verifier for the S4 loader (mirrors compiled.rs test stubs).
    struct StubSigner {
        key: [u8; 32],
    }
    impl PolicySigner for StubSigner {
        fn sign(&self, input: &[u8]) -> Result<Vec<u8>, PolicyDistError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            Ok(h.finalize().as_bytes().to_vec())
        }
    }
    struct StubVerifier {
        key: [u8; 32],
    }
    impl PolicyVerifier for StubVerifier {
        fn verify(&self, input: &[u8], sig: &[u8]) -> Result<(), PolicyDistError> {
            let mut h = blake3::Hasher::new();
            h.update(&self.key);
            h.update(input);
            if h.finalize().as_bytes().as_slice() == sig {
                Ok(())
            } else {
                Err(PolicyDistError::BadSignature("stub mismatch".into()))
            }
        }
    }

    #[test]
    fn from_verified_keystone_end_to_end_then_loader_accepts() {
        // The S5→S4 keystone: a verified SignedApproval's binding becomes the
        // PolicyApproval the S4 loader requires, and the loader (which RECOMPUTES
        // the hash) then accepts the matching signed policy.
        let v = vocab();
        let l = lattice(13);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        let (bundle, signed) = compile_and_bind(&u, &l, &v, &id.ed_sk, &id.pq_sk).unwrap();

        // Verify the approval (UCAN side) and convert its binding to a PolicyApproval.
        let binding = signed.verify(&id.ed_sk.verifying_key(), &id.pq_vk).unwrap();
        let approval = PolicyApproval::from_verified(binding);
        assert_eq!(approval.generation, bundle.policy.generation);
        assert_eq!(approval.approved_hash, bundle.bundle_hash().unwrap());

        // Sign the compiled policy for distribution and load it with the approval.
        let key = [5u8; 32];
        let signed_policy = sign_policy(&bundle.policy, &StubSigner { key }).unwrap();
        let loader = PolicyLoader::new(StubVerifier { key }).with_approval(approval);
        // The loader recomputes the hash and matches the approval → accepts.
        let loaded = loader.load(&signed_policy).unwrap();
        assert_eq!(loaded.generation, bundle.policy.generation);
    }

    #[test]
    fn tampered_bundle_hash_fails_verify_binds_before_loader() {
        // S5 containment: a tampered bundle_hash must fail verify_binds BEFORE it
        // can reach the loader — so a forged approval never becomes a PolicyApproval.
        let v = vocab();
        let l = lattice(13);
        let id = identity();
        let u = ucan_signed_by(&id, vec![cap("mac://model/qwen", "model/read")]);
        let (bundle, signed) = compile_and_bind(&u, &l, &v, &id.ed_sk, &id.pq_sk).unwrap();

        let mut tampered = bundle.bundle_hash().unwrap();
        tampered[0] ^= 0xFF;
        // verify_binds with a tampered hash rejects (the binding signed the real hash).
        let res = signed.verify_binds(
            &id.ed_sk.verifying_key(),
            &id.pq_vk,
            &u,
            bundle.policy.generation,
            &tampered,
        );
        assert!(
            matches!(
                res,
                Err(hyprstream_rpc::auth::ucan::approval::ApprovalError::BundleHashMismatch)
            ),
            "a tampered bundle hash must fail verify_binds before the loader"
        );
    }

    // ---- Differential property tests: matrix evaluation ≡ reference ---------

    fn arb_capability() -> impl Strategy<Value = Capability> {
        let resources = prop_oneof![
            Just("mac://model/qwen"),
            Just("mac://model/llama"),
            Just("mac://model/mistral"),
            Just("mac://model/*"),
            Just("mac://other/x"), // unmapped resource (vocab ignores)
        ];
        let abilities = prop_oneof![
            Just("model/read"),
            Just("model/write"),
            Just("admin/x"), // unmapped verb (vocab ignores)
        ];
        (resources, abilities).prop_map(|(r, a)| cap(r, a))
    }

    fn arb_ceiling() -> impl Strategy<Value = Vec<Capability>> {
        prop::collection::vec(arb_capability(), 0..5)
    }

    fn arb_request() -> impl Strategy<Value = AccessRequest> {
        let resources = prop_oneof![
            Just("mac://model/qwen"),
            Just("mac://model/llama"),
            Just("mac://model/mistral"),
        ];
        let abilities = prop_oneof![Just("model/read"), Just("model/write")];
        (resources, abilities).prop_map(|(r, a)| AccessRequest {
            resource: Resource::new(r),
            ability: Ability::new(a),
            caveats: Caveats::empty(),
        })
    }

    /// Decide a request against the compiled matrix on the structural fragment:
    /// resolve the request to its single image rule via the vocab and ask the
    /// matrix. (No lattice floor — that is S4's independent content check; we test
    /// the AUTHORITY gate the compiler produces.)
    fn matrix_decides(v: &TestVocab, matrix: &TeMatrix, req: &AccessRequest) -> Decision {
        let claimed = req.as_capability();
        match v.rules_for(&claimed).into_iter().next() {
            Some(rule) => matrix.te_decision(rule),
            None => Decision::Deny,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// THE differential faithfulness property: compiled matrix evaluation
        /// AGREES with the reference interpreter on every request; and the bundle is
        /// always faithful for ANY ceiling.
        #[test]
        fn matrix_agrees_with_reference_interpreter(
            ceiling in arb_ceiling(),
            req in arb_request(),
        ) {
            let v = vocab();
            let l = lattice(1);
            let emitter = TeBundleEmitter::new(&l, &v);
            let u = ucan_with(ceiling.clone());
            let bundle = emitter.emit(&u).unwrap();

            // Soundness on every generated input.
            prop_assert_eq!(Faithfulness::new(&v).check(&u, &bundle), Ok(()));

            let reference = ReferenceInterpreter::from_validated(&u);
            let ref_decision = reference.decide(&req);
            let matrix_decision = matrix_decides(&v, &bundle.policy.matrix, &req);
            prop_assert_eq!(
                matrix_decision,
                ref_decision,
                "matrix and reference interpreter disagree on {:?} for ceiling {:?}",
                req,
                ceiling
            );
        }

        /// TE-soundness (b), explicit one-directional: the emitted model can NEVER
        /// permit anything the reference interpreter denies (the security-critical
        /// implication, stated on its own).
        #[test]
        fn matrix_permit_implies_reference_permit(
            ceiling in arb_ceiling(),
            req in arb_request(),
        ) {
            let v = vocab();
            let l = lattice(1);
            let emitter = TeBundleEmitter::new(&l, &v);
            let u = ucan_with(ceiling);
            let bundle = emitter.emit(&u).unwrap();
            let reference = ReferenceInterpreter::from_validated(&u);
            if matrix_decides(&v, &bundle.policy.matrix, &req) == Decision::Permit {
                prop_assert_eq!(
                    reference.decide(&req),
                    Decision::Permit,
                    "matrix permitted what the reference denies — a WIDENING"
                );
            }
        }

        /// Determinism: re-emitting any ceiling yields an identical hash.
        #[test]
        fn emission_is_deterministic_for_any_ceiling(ceiling in arb_ceiling()) {
            let v = vocab();
            let l = lattice(1);
            let emitter = TeBundleEmitter::new(&l, &v);
            let u = ucan_with(ceiling);
            let h1 = emitter.emit(&u).unwrap().bundle_hash().unwrap();
            let h2 = emitter.emit(&u).unwrap().bundle_hash().unwrap();
            prop_assert_eq!(h1, h2);
        }
    }

    // ---- SMT scaffold defers, never blocks ---------------------------------

    #[test]
    fn smt_prover_defers_when_unavailable() {
        use super::smt::{DeferredProver, SmtResult, StructuralEquivalence};
        let v = vocab();
        let l = lattice(1);
        let emitter = TeBundleEmitter::new(&l, &v);
        let u = ucan_with(vec![cap("mac://model/qwen", "model/read")]);
        let bundle = emitter.emit(&u).unwrap();
        assert_eq!(
            DeferredProver.prove(&u, &bundle),
            SmtResult::SolverUnavailable
        );
    }
}
