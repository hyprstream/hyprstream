//! Mandatory Access Control — the **Policy Decision Point** (PDP) for the 9P/VFS surface.
//!
//! This is ticket **S4 (#570)** of the native-MAC security epic (#547): the data-plane
//! authorization engine. It implements, against an assumed S1 lattice interface:
//!
//! - [`lattice`] — the **assumed S1 contract** (`Lattice` trait + `SecurityLabel`). S1 (#567)
//!   owns the real implementation; everything here is built against the trait so the two
//!   streams reconcile by S1 satisfying the documented `S1-ASSUMPTION` notes. Ships a clearly
//!   marked `StubLinearLattice` so S4 is testable before S1 lands.
//! - [`te`] — the **type-enforcement evaluator**: a TOTAL, default-deny
//!   `(subject_ctx, object_label, action) → Decision` over the compiled TE matrix AND the
//!   independent lattice floor. Small + verifiable. This is the PDP core.
//! - [`avc`] — the **Access Vector Cache**: the fast local decision cache the PEP (S2) calls
//!   per op. Sub-µs hits (hash lookup, no Casbin, no signature verify, no lattice walk on a
//!   hit). Defines how an OAuth access token maps to a cached decision (token = distributed
//!   AVC entry).
//! - [`compiled`] — **compiled-policy distribution + signing**: PolicyService hashes + signs
//!   the TE matrix (hybrid EdDSA + ML-DSA-65 COSE composite); the loader rejects any
//!   unsigned/mismatched/unapproved policy. Verification happens ONCE at load, never per op.
//!
//! ## The control-plane / data-plane split (be explicit)
//!
//! ```text
//! CONTROL PLANE (authoring / compilation store)      DATA PLANE (per-op enforcement)
//! ─────────────────────────────────────────────      ───────────────────────────────────
//! UCAN                  the grant/delegation source
//! Casbin Enforcer       policy STORE + AUTHORING      ── NEVER on the per-op hot path ──
//!   (auth::policy_manager::PolicyManager)
//! S5 UCAN→TE compiler   lowers Casbin/UCAN → matrix──► TeMatrix         (compiled object code)
//! PolicyService         compiles + SIGNS policy ─────► compiled::SignedPolicy
//!                                                      │ loader verifies ONCE
//!                                                      ▼
//!                                                      te::LatticeTeEvaluator  (the PDP)
//!                                                      avc::CachingAvc          (the cache)
//!                                                      ▲ PEP (S2) calls per op
//! ```
//!
//! **Casbin stays control-plane.** The existing `casbin::Enforcer` in
//! [`crate::auth::policy_manager::PolicyManager`] remains the policy store and authoring
//! surface (RBAC/ABAC string matching, templates, `add_policy`/`apply_template`). Its
//! `enforce()` string-matcher is **never** invoked per op — that would blow the latency
//! budget and enlarge the TCB. Instead, S5's compiler lowers the authored policy into the
//! compact [`te::TeMatrix`] (interned-id allow-set, O(1) lookup), which this PDP evaluates.
//! The mandatory lattice floor is **independent of Casbin entirely** (design §5a: "the
//! lattice is NOT encoded in Casbin").
//!
//! ## TCB note
//!
//! The per-op TCB is intentionally tiny: a hash lookup ([`avc`]), and on a miss a set lookup
//! plus one lattice `dominates` call ([`te`]). All heavy/bug-prone logic (Casbin matching,
//! UCAN chain validation, compilation, signature verification) is concentrated off the hot
//! path — in PolicyService and the [`compiled`] loader, the one audited place (design §2).

pub mod avc;
pub mod compiled;
pub mod lattice;
pub mod te;

// Re-export the per-op contract surface S2 (PEP) and S5/S6 (policy producers) consume.
pub use avc::{Avc, AvcKey, CachingAvc, TokenScope};
pub use compiled::{
    sign_policy, CompiledPolicy, PolicyApproval, PolicyDistError, PolicyLoader, PolicySigner,
    PolicyVerifier, SignedPolicy,
};
pub use lattice::{ifc_join, Lattice, SecurityLabel};
pub use te::{
    Action, Decision, LatticeTeEvaluator, ObjectCtx, ObjectType, SubjectCtx, SubjectType,
    TeEvaluator, TeMatrix, TeRule,
};
