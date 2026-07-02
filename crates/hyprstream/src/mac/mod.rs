//! Mandatory Access Control — the **Policy Decision Point** (PDP) for the 9P/VFS surface.
//!
//! This is ticket **S4 (#570)** of the native-MAC security epic (#547): the data-plane
//! authorization engine. It implements, against an assumed S1 lattice interface:
//!
//! - [`lattice`] — a thin **re-export of S1's canonical types** (#567): the structured
//!   `SecurityLabel` (Level × Assurance × CompartmentSet, `Copy + Ord + Hash`, no `Default`),
//!   the `Lattice` *policy object* (closed compartment name↔bit vocabulary, versioned), and
//!   `SecurityContext`. Dominance/join are INTRINSIC to the label, not trait methods — the
//!   PDP floor calls them directly. (The original S4 stub lattice was deleted at
//!   reconciliation; see the `lattice` module docs.)
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
//! plus one intrinsic `SecurityLabel::can_access` call ([`te`]). All heavy/bug-prone logic
//! (Casbin matching,
//! UCAN chain validation, compilation, signature verification) is concentrated off the hot
//! path — in PolicyService and the [`compiled`] loader, the one audited place (design §2).

pub mod audit;
pub mod avc;
pub mod compiled;
pub mod compiler;
// S6 (#572): runtime grant path — UCAN grant-request → access/refresh tokens.
// Pure fail-closed core: grant validation + ceiling-subset + MAC clearance +
// sender-binding, consuming S1/S5/the compiler rather than re-implementing.
pub mod exchange;
pub mod lattice;
pub mod te;

// Re-export the per-op contract surface S2 (PEP) and S5/S6 (policy producers) consume.
pub use avc::{Avc, AvcKey, CachingAvc, TokenScope};
// S7 (#573): tamper-evident audit of every authorization decision + OTel fan-out.
pub use audit::{
    audit_signing_input, AuditError, AuditRecord, AuditSigner, AuditSink, AuditVerifier,
    AuditedAvc, DecisionReason, NullAuditSink, WalAuditStore,
};
pub use compiled::{
    sign_policy, CompiledPolicy, PolicyApproval, PolicyDistError, PolicyLoader, PolicySigner,
    PolicyVerifier, SignedPolicy,
};
// Lattice surface is now S1's canonical types (#567), re-exported through `lattice` so the
// PDP has one import path. `ifc_join` is the local forwarder over the intrinsic join.
pub use lattice::{
    ifc_join, Assurance, Compartment, CompartmentSet, LabelError, Lattice, LatticeCodecError,
    LatticeDecodeError, LatticeVersion, Level, SecurityContext, SecurityLabel,
    SubjectContextClaims, VerifiedKeyMaterial, MAX_COMPARTMENTS,
};
pub use te::{
    Action, Decision, LatticeTeEvaluator, ObjectCtx, ObjectType, ScopeAction, SubjectCtx,
    SubjectType, TeEvaluator, TeMatrix, TeRule,
};
// S5 (#571): the UCAN→TE policy compiler — compile a validated grant into a
// CompiledPolicy, verify it grants no privilege beyond the grant, and sign it
// (fail-closed). Lives here (not `hyprstream-rpc`) because it needs both the UCAN
// model and S4's `TeMatrix`/`CompiledPolicy`.
pub use compiler::{
    authorize, authorize_at, check_no_escalation, compile, compile_policy, missing_permission,
    AccessRequest, CompileError, PermissionMap, PrivilegeEscalation,
};

/// Construct the [`UcanVerifier`] the HTTP grant path uses to validate a
/// presented UCAN's signatures.
///
/// This is the S6↔trust-store seam: the verifier resolves each UCAN issuer's
/// anchored ML-DSA-65 verifying key (the same `register_pq_trust` binding the
/// rest of the TCB uses). Until that binding is wired through the OAuth state,
/// this returns `None`, which makes the HTTP grant path deny every request
/// (`evaluate_grant` never runs against an unverified chain — fail-closed).
///
/// The core `evaluate_grant` is exercised through its own tests with a real
/// verifier, so the security-critical logic is covered independently of this
/// wiring landing.
///
/// TODO(#572-verifier): wire the trust store's anchored PQ keys into a concrete
///   `UcanVerifier` and return it from this function.
pub fn exchange_ucan_verifier(
    _state: &crate::services::oauth::state::OAuthState,
) -> Option<Box<dyn hyprstream_rpc::auth::ucan::token::UcanVerifier + Send + Sync>> {
    // Fail-closed: no verifier is wired ⇒ the HTTP grant path denies. The
    // single authority path; no fallback.
    None
}
