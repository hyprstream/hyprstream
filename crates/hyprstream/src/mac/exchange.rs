//! **MAC S6 — runtime grant path: UCAN grant-request → access/refresh tokens.**
//! Ticket #572 of the native-MAC epic (#547).
//!
//! This is the **runtime** companion to S5's compile-time UCAN→Casbin/TE compiler.
//! S5 compiles an *approved* UCAN into a ceiling (a `CompiledPolicy` +
//! `SignedApproval` binding) — that is the standing authorization-to-REQUEST.
//! S6 is the path a principal takes at runtime to turn a **subset-UCAN grant**
//! into **ephemeral, sender-bound OAuth tokens** (RFC 8693 token-exchange shape).
//!
//! ## The #547 MAC model this honours
//!
//! - **UCAN authors; Casbin/TE enforces.** The grant presented here is a UCAN
//!   delegation chain; we never mint tokens wider than that grant.
//! - **ZSP (Zero Standing Privilege).** The ceiling is standing
//!   authorization-to-REQUEST, NOT access. Access is **ephemeral** (short-ttl),
//!   **per-task** (not per-op), and **re-evaluated on refresh**. There is no
//!   long-lived bearer that bypasses the grant.
//! - **Ceiling = the grant itself.** The subset-UCAN a principal presents is a
//!   strict attenuated subset of an approved ceiling. The grant's root
//!   capabilities ARE the ceiling for this request; `authorize(grant, request)`
//!   is the ceiling-subset check.
//! - **Fail-closed / single path.** Every failure mode — bad chain, over-ceiling
//!   request, insufficient MAC clearance, missing sender-binding — denies. There
//!   is **no fallback path**: a fallback is a downgrade vector. Authority being
//!   unreachable is a denial, not a reason to mint a narrower token.
//! - **Escalation ≠ auto-widen.** A request beyond the grant's ceiling is a
//!   denial pending a **ceiling amendment** (the tiered decision — static Casbin
//!   / agentic / admin-UDF — which is authority-owned and MAC-bounded). S6 does
//!   NOT auto-escalate; it routes to `Decision::Escalate` and returns
//!   `insufficient_scope`. The amendment flow is deferred (TODO locations below).
//!
//! ## What this module owns (and what it does NOT)
//!
//! **Owns (the fail-closed core):**
//! - Validate the presented UCAN grant chain (delegates to S5's
//!   [`hyprstream_rpc::auth::ucan::chain::validate`] — signatures + attenuation +
//!   temporal validity at the trusted `now`).
//! - **Ceiling-subset check**: the access being requested now must be authorized
//!   by the grant's capabilities ([`mac::compiler::authorize`]).
//! - **MAC clearance gate**: the principal's clearance (from verified key
//!   material / S1 `SubjectContextClaims`) must dominate the request's label.
//!   Insufficient clearance ⇒ deny (the S1 floor, reused — not reinvented).
//! - Compose the subset of capabilities the token will actually carry
//!   (never the whole grant — ZSP: mint the least subset that covers the
//!   requested access).
//!
//! **Does NOT own (consumed from siblings — no parallel implementation):**
//! - UCAN chain validation, attenuation, signature verification → S5
//!   (`hyprstream_rpc::auth::ucan`).
//! - The grant-vs-request authorization relation → S5's compiler
//!   ([`mac::compiler::authorize`]).
//! - Security labels, clearance, dominance → S1 (`hyprstream_rpc::auth::mac`).
//! - DPoP sender-binding (#514/#515) → `services::oauth::dpop`.
//! - JWT signing → `services::oauth` (via `hyprstream_rpc::auth::jwt`).
//! - Hybrid-PQC signing of the minted tokens → **S8 (#574) landed**: the
//!   `mint_grant_token` path now signs via the hybrid composite JWT
//!   (`encode_composite_ml_dsa_65_ed25519`) when an ML-DSA-65 key is
//!   provisioned, matching the hybrid signature on the UCAN/approval it
//!   consumes. Classical Ed25519 is the explicit policy-selected fallback.
//!
//! ## Scope of this PR (core, sound, tested) vs. deferred
//!
//! **In scope:** the pure fail-closed core — grant validation, ceiling-subset
//! check, MAC clearance gate, sender-binding requirement, and the
//! `GrantDecision` it produces. Plus full negative-test coverage (over-ceiling,
//! bad chain, insufficient clearance, missing DPoP). The HTTP grant wiring and
//! token minting live in `services::oauth::token_exchange` and call into
//! [`evaluate_grant`] here.
//!
//! **Deferred (documented TODOs, fail-closed until landed):**
//! - Full escalation tiers (static Casbin / agentic / admin-UDF amendment) —
//!   see [`GrantDecision::Escalate`] and the `TODO(#572-escalation)` markers.
//! - Refresh-token rotation with re-evaluation plumbing end-to-end (the
//!   `evaluate_grant` call on refresh is wired; rotation storage is the OAuth
//!   layer's job) — `TODO(#572-refresh-rotation)`.
//! - Revocation propagation into the grant path (jti blocklist check at grant
//!   time) — `TODO(#572-revocation)`.

use hyprstream_rpc::auth::mac::{
    SecurityContext, SecurityLabel, SubjectContextClaims, VerifiedKeyMaterial,
};
use hyprstream_rpc::auth::ucan::capability::{Ability, Capability, Caveats, Resource};
use hyprstream_rpc::auth::ucan::chain::{self, ChainError};
use hyprstream_rpc::auth::ucan::token::{Ucan, UcanVerifier};

use crate::mac::compiler::{authorize_at, AccessRequest};
use crate::mac::te::Decision;

/// The token-type URI identifying a UCAN grant as the RFC 8693 `subject_token`.
///
/// A presenter supplies a CBOR-encoded UCAN as `subject_token` with this
/// `subject_token_type` at the token-exchange endpoint. Distinct from the OIDC
/// `id_token` / `access_token` / `jwt` types so the grant path cannot be
/// confused with the existing exchange flows.
pub const UCAN_GRANT_TOKEN_TYPE: &str = "urn:hyprstream:token-type:ucan-grant";

/// Hard ceiling on the access-token TTL minted from a UCAN grant. ZSP: grant
/// tokens are deliberately short-lived so the re-evaluation-on-refresh property
/// has teeth. The OAuth layer clamps its configured `token_ttl` down to this.
pub const MAX_ACCESS_TOKEN_TTL_SECS: u32 = 900; // 15 minutes

/// Why a grant request was denied. Every variant is fail-closed; there is no
/// "best-effort" or "narrow-and-continue" outcome. The cardinal rule of #547:
/// authority-unreachable ⇒ deny, never downgrade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantError {
    /// The UCAN chain failed S5 validation (bad signature, broken linkage,
    /// widening, window escape, expired/not-yet-valid, or over-depth). Wrapped
    /// verbatim from S5 so the failure mode is auditable.
    Chain(ChainError),
    /// The request exceeds the grant's ceiling — an escalation attempt. This is
    /// routed to the (deferred) escalation tier; until that lands it is a
    /// denial. **S6 never auto-escalates.** The offending requested capability
    /// (the one no grant capability authorizes) is included for diagnostics.
    OverCeiling { requested: Capability },
    /// The principal's MAC clearance does not dominate the request's label — the
    /// S1 floor denies. This is independent of the ceiling: a principal may hold
    /// a valid grant for an object whose label they are not cleared for. Both
    /// gates must pass.
    InsufficientClearance,
    /// No sender-binding (DPoP proof) was supplied. ZSP: the minted token MUST
    /// be sender-bound — a bearer token minted from a grant would re-introduce
    /// standing access, defeating the whole model.
    MissingSenderBinding,
    /// The subject carried no derivable clearance at all (unlabeled subject).
    /// Per S1, an unlabeled subject has no context and the monitor denies —
    /// there is no default clearance.
    UnlabeledSubject,
    /// The grant carries no capabilities at all, so there is no ceiling to
    /// request a subset of. An empty grant mints nothing.
    EmptyGrant,
    /// **B2 (#674):** the request would otherwise `Permit`, but the decision
    /// could not be durably audited (the audit sink's `record` failed, or its
    /// signer cannot sign under the enforced crypto policy). Mirrors S7's
    /// `AuditedAvc` fail-closed rule at the grant path: a decision that cannot
    /// be audited is never allowed through. See [`audited_evaluate_grant`].
    AuditUnavailable,
}

impl std::fmt::Display for GrantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrantError::Chain(e) => write!(f, "UCAN grant chain invalid: {e}"),
            GrantError::OverCeiling { requested } => write!(
                f,
                "grant request exceeds ceiling (escalation denied pending amendment): {requested}"
            ),
            GrantError::InsufficientClearance => {
                write!(
                    f,
                    "subject MAC clearance does not dominate the request label"
                )
            }
            GrantError::MissingSenderBinding => {
                write!(
                    f,
                    "UCAN grant token-exchange requires a sender-binding (DPoP) proof"
                )
            }
            GrantError::UnlabeledSubject => {
                write!(f, "subject carries no MAC clearance (unlabeled ⇒ deny)")
            }
            GrantError::EmptyGrant => write!(f, "grant carries no capabilities (empty ceiling)"),
            GrantError::AuditUnavailable => write!(
                f,
                "grant decision could not be durably audited (fail-closed)"
            ),
        }
    }
}

impl std::error::Error for GrantError {}

/// The outcome of evaluating a grant request: the fail-closed "mint" decision.
///
/// On `Permit`, [`GrantedAccess`] holds the **minimum subset** of the grant the
/// token will encode (ZSP: never mint the whole grant; mint the least authority
/// that covers the request). On `Escalate`, the request is over-ceiling and
/// awaits a (deferred) amendment; the caller returns `insufficient_scope`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantDecision {
    /// The request is a valid subset of a valid grant, the subject is cleared,
    /// and a sender-binding is present. The contained [`GrantedAccess`] is what
    /// the token minting layer encodes.
    Permit(GrantedAccess),
    /// The request exceeds the grant ceiling. Routed to the escalation tier
    /// (TODO(#572-escalation)); until the tiered-decision lands this is a
    /// denial. `requested` is the over-ceiling capability for diagnostics.
    Escalate { requested: Capability },
}

/// The minimum subset of the grant a minted token will carry — the ZSP output.
///
/// This is the single capability (resource + ability + caveats) that the
/// presenter requested, re-checked against the grant. It is intentionally a
/// *subset* (≤ the grant), not the grant verbatim: the minted token is the
/// least authority covering the request, so a refresh re-evaluation can only
/// re-grant what was originally requested, never grow toward the ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantedAccess {
    /// The capability the minted token encodes: the requested access, which is
    /// guaranteed (by the ceiling-subset check) to be authorized by the grant.
    pub capability: Capability,
    /// The audience the token is minted for (RFC 8707 resource indicator), if
    /// any. Carried through to the JWT `aud` claim.
    pub audience: Option<String>,
}

/// A resolved grant request: the principal's desired access, parsed off the
/// token-exchange `resource`/`scope` form fields into the S5 capability shape so
/// the compiler's `authorize` can evaluate it.
#[derive(Debug, Clone)]
pub struct GrantRequest {
    /// The resource being requested (`mac://model/qwen-7b` style). Required.
    pub resource: Resource,
    /// The ability being requested (`infer`, `model/read`, …). Required.
    pub ability: Ability,
    /// Caveats the presenter asks to be bound by (e.g. tenant). May be empty;
    /// the resulting token is at least this restricted.
    pub caveats: Caveats,
    /// Optional audience (RFC 8707 resource indicator).
    pub audience: Option<String>,
    /// The MAC label of the object the request targets — used for the S1
    /// clearance floor. `None` ⇒ unlabeled object ⇒ deny (the S1 rule). The
    /// caller (OAuth layer) obtains this from the manifest/TE vocabulary; S6
    /// does not invent it.
    pub object_label: Option<SecurityLabel>,
}

/// Evaluate a UCAN grant request: the single fail-closed S6 path.
///
/// Inputs:
/// - `grant`: the presented subset-UCAN (a delegation chain whose root
///   capabilities are the ceiling for this request).
/// - `verifier`: the S5 `UcanVerifier` (trust store resolving each issuer's
///   anchored ML-DSA-65 key). Signatures are verified under the hybrid policy.
/// - `now`: the single trusted clock (unix seconds), threaded unchanged into
///   every chain link by S5 — never read from the token.
/// - `request`: what the principal is asking to do now (the subset).
/// - `subject`: the principal's verified security context (S1 clearance +
///   assurance, clamped to verified key material). `None` ⇒ unlabeled ⇒ deny.
/// - `sender_bound`: whether a DPoP proof was supplied and verified by the
///   caller. ZSP requires sender-binding; `false` ⇒ deny.
///
/// Gates, in order (all fail-closed; the first failure wins):
/// 1. **Sender-binding required** (ZSP) — no bearer tokens from a grant.
/// 2. **Non-empty grant** — an empty grant mints nothing.
/// 3. **Labeled subject** — S1: unlabeled subject ⇒ deny (no default clearance).
/// 4. **Labeled object** — S1: unlabeled object ⇒ deny.
/// 5. **MAC clearance floor** — `subject.clearance ⊒ object_label`.
/// 6. **UCAN chain valid** — S5 `validate` (signatures + attenuation + temporal).
/// 7. **Ceiling-subset** — `authorize_at(grant.capabilities(), grant.window,
///    request, now)` permits. `Deny` ⇒ over-ceiling ⇒ `Escalate` (not auto-mint).
///
/// Order rationale: the cheap structural/label gates run before the
/// cryptographic chain walk, so a malformed or uncleared request is rejected
/// without spending the signature-verification budget. The chain is validated
/// before the ceiling check so an invalid chain can never reach `authorize`.
pub fn evaluate_grant<V: UcanVerifier + ?Sized>(
    grant: &Ucan,
    verifier: &V,
    now: u64,
    request: &GrantRequest,
    subject: Option<&SecurityContext>,
    sender_bound: bool,
) -> Result<GrantDecision, GrantError> {
    // (1) ZSP: sender-binding is mandatory. A bearer token minted from a grant
    // re-introduces standing access — the exact thing ZSP removes. The caller
    // verifies the DPoP proof; we only require that it did.
    if !sender_bound {
        return Err(GrantError::MissingSenderBinding);
    }

    // (2) An empty grant is no ceiling at all. (set_attenuates treats an empty
    // `held` as covering only an empty `claimed`; we surface it explicitly so
    // the error is unambiguous.)
    if grant.capabilities().is_empty() {
        return Err(GrantError::EmptyGrant);
    }

    // (3)+(4) S1 floor: unlabeled subject or unlabeled object ⇒ deny. The MAC
    // model has no default clearance and no default label; absence is denial.
    let subject = subject.ok_or(GrantError::UnlabeledSubject)?;
    let object_label = request
        .object_label
        .ok_or(GrantError::InsufficientClearance)?;

    // (5) MAC clearance dominance — the S1 floor, reused not reinvented.
    // `SecurityContext::can_access` is the intrinsic lattice dominance
    // (level ≥, assurance ≥, compartments ⊇). This gate is INDEPENDENT of the
    // ceiling: a principal may hold a valid grant for an object whose label they
    // are not cleared for (e.g. PQC-required object, classical-only subject).
    // Both must pass.
    if !subject.can_access(&object_label) {
        return Err(GrantError::InsufficientClearance);
    }

    // (6) S5 full chain validation: structure → hybrid signatures →
    // attenuation/linkage → temporal validity at `now`. Fail-closed on any
    // malformation, widening, broken linkage, window escape, expiry, or
    // over-depth. This is the load-bearing authority check; nothing below runs
    // against an unvalidated chain.
    if let Err(e) = chain::validate(grant, verifier, now) {
        return Err(GrantError::Chain(e));
    }

    // (7) Ceiling-subset: is the requested access authorized by the grant's
    // capabilities at `now`? This is S5's compiler `authorize_at`, the same
    // relation `check_no_escalation` uses — not a parallel implementation.
    let req = AccessRequest {
        resource: request.resource.clone(),
        ability: request.ability.clone(),
        caveats: request.caveats.clone(),
    };
    let decision = authorize_at(
        grant.capabilities(),
        grant.payload.not_before,
        grant.payload.expiration,
        &req,
        now,
    );
    match decision {
        Decision::Permit => {
            // ZSP: mint the LEAST authority covering the request — the requested
            // capability itself, which `authorize` just confirmed the grant
            // covers. Never mint the whole grant; a refresh can only re-grant
            // this subset.
            let capability = Capability::with_caveats(
                request.resource.clone(),
                request.ability.clone(),
                request.caveats.clone(),
            );
            Ok(GrantDecision::Permit(GrantedAccess {
                capability,
                audience: request.audience.clone(),
            }))
        }
        // Over-ceiling. This is the escalation tier ingress. S6 does NOT
        // auto-escalate: a request beyond the ceiling is denied pending a
        // ceiling amendment via the (deferred) tiered decision.
        // TODO(#572-escalation): route to the tiered decision (static Casbin /
        //   agentic / admin-UDF — authority-owned, MAC-bounded). Until that
        //   lands, over-ceiling is a hard denial via the caller's
        //   `insufficient_scope` response.
        Decision::Deny | Decision::Escalate => Err(GrantError::OverCeiling {
            requested: Capability::with_caveats(
                request.resource.clone(),
                request.ability.clone(),
                request.caveats.clone(),
            ),
        }),
    }
}

/// Re-evaluate a grant on token refresh: the ZSP "access is re-evaluated on
/// refresh" property.
///
/// Refresh is NOT a free re-mint. The presenter must re-present the grant (the
/// ceiling may have been amended or revoked since the access token was minted),
/// and the same [`evaluate_grant`] gates run again. This is a thin forwarder
/// today — the name exists so the refresh path has an obvious, audited call site
/// distinct from the initial mint, and so the deferred rotation logic
/// (TODO(#572-refresh-rotation)) has a home.
///
/// SECURITY: the refresh MUST supply a fresh DPoP proof (`sender_bound` from a
/// new proof, not carried over) and the same `now`-threading rules apply.
pub fn evaluate_refresh<V: UcanVerifier + ?Sized>(
    grant: &Ucan,
    verifier: &V,
    now: u64,
    request: &GrantRequest,
    subject: Option<&SecurityContext>,
    sender_bound: bool,
) -> Result<GrantDecision, GrantError> {
    // Identical gates to the initial mint. ZSP: no path diverges here — a
    // refresh that weakens any gate is a refresh that denies.
    evaluate_grant(grant, verifier, now, request, subject, sender_bound)
}

/// As [`evaluate_grant`], but records a tamper-evident [`AuditRecord`] of the
/// outcome through `sink` before returning — S7's complete-mediation guarantee
/// extended to the grant path (B2, #674). This is the mint-path entry point
/// production code should call; the plain [`evaluate_grant`] is the
/// unaudited core the gate tests exercise directly.
///
/// Every gate outcome is audited, permit and deny alike (`DecisionReason`'s
/// `Grant*` variants — see [`crate::mac::audit::DecisionReason`]). On a would-be
/// `Permit`, the audit write happens **before** the token layer is told to
/// mint: if it fails, the permit is downgraded to
/// [`GrantError::AuditUnavailable`] rather than handed back — mirroring
/// [`crate::mac::audit::AuditedAvc`]'s fail-closed rule that a decision which
/// cannot be audited is never allowed through.
#[allow(clippy::too_many_arguments)]
pub fn audited_evaluate_grant<V: UcanVerifier + ?Sized>(
    grant: &Ucan,
    verifier: &V,
    now: u64,
    request: &GrantRequest,
    subject: Option<&SecurityContext>,
    on_behalf_of: Option<&SecurityContext>,
    sender_bound: bool,
    sink: &dyn crate::mac::audit::AuditSink,
) -> Result<GrantDecision, GrantError> {
    let result = evaluate_grant(grant, verifier, now, request, subject, sender_bound);
    audit_grant_outcome(subject, on_behalf_of, request, &result, sink)
}

/// As [`evaluate_refresh`], with the same audit-then-decide wrapping as
/// [`audited_evaluate_grant`]. The refresh path's audit records are
/// indistinguishable in shape from the mint path's (both are grant-path
/// decisions); the caller distinguishes them by context if needed.
#[allow(clippy::too_many_arguments)]
pub fn audited_evaluate_refresh<V: UcanVerifier + ?Sized>(
    grant: &Ucan,
    verifier: &V,
    now: u64,
    request: &GrantRequest,
    subject: Option<&SecurityContext>,
    on_behalf_of: Option<&SecurityContext>,
    sender_bound: bool,
    sink: &dyn crate::mac::audit::AuditSink,
) -> Result<GrantDecision, GrantError> {
    let result = evaluate_refresh(grant, verifier, now, request, subject, sender_bound);
    audit_grant_outcome(subject, on_behalf_of, request, &result, sink)
}

/// Shared audit-then-decide core for [`audited_evaluate_grant`] /
/// [`audited_evaluate_refresh`]. Builds the [`AuditRecord`] from whatever
/// context is available (subject may be `None` on an unlabeled-subject
/// denial; the request's ability may not parse as a canonical
/// [`crate::mac::te::ScopeAction`] — both are still recorded, using the
/// sentinel ids documented on [`crate::mac::te::GRANT_PATH_SUBJECT`]).
fn audit_grant_outcome(
    subject: Option<&SecurityContext>,
    on_behalf_of: Option<&SecurityContext>,
    request: &GrantRequest,
    result: &Result<GrantDecision, GrantError>,
    sink: &dyn crate::mac::audit::AuditSink,
) -> Result<GrantDecision, GrantError> {
    use crate::mac::audit::{AuditRecord, DecisionReason, DelegationPrincipal};
    use crate::mac::te::{Action, ScopeAction, ACTION_UNRECOGNIZED, GRANT_PATH_OBJECT, GRANT_PATH_SUBJECT};
    use hyprstream_rpc::auth::mac::SecurityLabel;

    let (decision, reason) = match result {
        Ok(GrantDecision::Permit(_)) => (Decision::Permit, DecisionReason::Permit),
        Ok(GrantDecision::Escalate { .. }) => (Decision::Escalate, DecisionReason::GrantOverCeiling),
        Err(GrantError::Chain(_)) => (Decision::Deny, DecisionReason::GrantChainInvalid),
        Err(GrantError::OverCeiling { .. }) => (Decision::Deny, DecisionReason::GrantOverCeiling),
        Err(GrantError::InsufficientClearance) => (Decision::Deny, DecisionReason::GrantFloorDeny),
        Err(GrantError::MissingSenderBinding) => {
            (Decision::Deny, DecisionReason::GrantMissingSenderBinding)
        }
        Err(GrantError::UnlabeledSubject) => (Decision::Deny, DecisionReason::GrantUnlabeledSubject),
        Err(GrantError::EmptyGrant) => (Decision::Deny, DecisionReason::GrantEmptyGrant),
        // Constructed below, never as the pre-audit decision — but exhaustive
        // match requires a reachable-if-mis-called arm. Records as a deny.
        Err(GrantError::AuditUnavailable) => (Decision::Deny, DecisionReason::GrantAuditFailClosed),
    };

    let action = ScopeAction::parse(request.ability.as_str())
        .map(Action::from)
        .unwrap_or(ACTION_UNRECOGNIZED);
    let record = AuditRecord {
        seq: 0,          // the sink (WalAuditStore) assigns the real seq.
        prev_hash: [0u8; 32], // the sink assigns the real chain link.
        ts_unix_nanos: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        decision,
        generation: 0, // grant decisions are not TE-matrix-generation-versioned.
        policy_hash: None,
        subject_type: GRANT_PATH_SUBJECT,
        subject_clearance: subject.map(|s| *s.clearance()).unwrap_or_else(SecurityLabel::bottom),
        // #680/#681: on a delegated grant the delegator (authority source) is
        // recorded here; `subject_clearance` above is the effective met context
        // the gates evaluated. Same GRANT_PATH_SUBJECT sentinel type — the
        // clearance is the distinguishing field on the grant path.
        on_behalf_of: on_behalf_of.map(|d| DelegationPrincipal {
            subject_type: GRANT_PATH_SUBJECT,
            subject_clearance: *d.clearance(),
        }),
        object_type: GRANT_PATH_OBJECT,
        object_label: request.object_label.unwrap_or_else(SecurityLabel::bottom),
        action,
        reason,
    };

    match sink.record(&record) {
        Ok(()) => result.clone(),
        Err(_) => {
            // Fail-closed: a would-be Permit that cannot be durably audited is
            // downgraded. A deny that cannot be audited is still enforced as
            // decided (the deny is not weakened by a broken audit sink) but the
            // failure is surfaced so it is observable.
            if matches!(result, Ok(GrantDecision::Permit(_))) {
                let deny_record = AuditRecord {
                    reason: DecisionReason::GrantAuditFailClosed,
                    decision: Decision::Deny,
                    ..record
                };
                let _ = sink.record(&deny_record);
                Err(GrantError::AuditUnavailable)
            } else {
                tracing::error!("MAC grant-path audit write failed on a Deny decision (deny still enforced)");
                result.clone()
            }
        }
    }
}

/// Resolve the S1 `SecurityContext` for a UCAN grant's audience from a
/// `SubjectContextClaims` implementation.
///
/// This is the seam between "the UCAN names an audience DID" and "the S1
/// clearance floor needs a verified subject context". The caller resolves the
/// audience DID to whatever claims object it has (envelope claims, registered
/// subject, …) implementing `SubjectContextClaims`, plus the assurance derived
/// from the verified key material. Returns `None` for an unlabeled subject —
/// which [`evaluate_grant`] maps to [`GrantError::UnlabeledSubject`].
///
/// S6 does not invent clearance: it only forwards what the claims layer
/// already provides (S1 contract). The concrete `clearance` field on `Claims`
/// lands with S8 (#574); until then implementors test against a stub claims
/// object, exactly as S1/S2 do.
pub fn subject_context(
    claims: &dyn SubjectContextClaims,
    key_material: VerifiedKeyMaterial,
) -> Option<SecurityContext> {
    claims.security_context(key_material)
}

// A small helper kept private so the public surface stays minimal: the
// reference decision used by tests to cross-check the compiler path. Not part
// of the contract.
#[cfg(test)]
fn reference_authorize(grant: &[Capability], request: &AccessRequest) -> Decision {
    crate::mac::compiler::authorize(grant, request)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    //! S6 negative tests + the happy path. No test asserts permissive behaviour:
    //! every case either confirms a correct minimal mint or confirms a denial.
    //! The denial cases are the security load — over-ceiling, bad chain,
    //! insufficient clearance, missing DPoP.
    use super::*;
    use crate::mac::te::Decision;
    use ed25519_dalek::{SigningKey, VerifyingKey};
    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityLabel, VerifiedKeyMaterial,
    };
    use hyprstream_rpc::auth::ucan::capability::{Ability, Capability, Resource};
    use hyprstream_rpc::auth::ucan::token::{
        Did, Ucan, UcanError, UcanPayload, UcanVerifier, UCAN_AAD,
    };
    use hyprstream_rpc::crypto::cose_sign::{sign_composite, verify_composite};
    use hyprstream_rpc::crypto::pq::{ml_dsa_generate_keypair, MlDsaSigningKey, MlDsaVerifyingKey};
    use std::collections::HashMap;

    // UCAN payload-signature AAD is the public `hyprstream_rpc::auth::ucan::token::UCAN_AAD`
    // constant (S8 promoted it so production verifiers and tests share the exact bytes).

    /// Trusted `now` inside every test UCAN's validity window
    /// (`not_before: None`, `expiration: 9_999_999_999`).
    const NOW: u64 = 1_000;

    /// A hybrid keypair (Ed25519 + ML-DSA-65) for one DID, mirroring S5's
    /// `TestIdentity` but built here on public APIs so S6 tests do not depend on
    /// S5's `pub(super)` test_support.
    struct Identity {
        ed_sk: SigningKey,
        ed_vk: VerifyingKey,
        pq_sk: MlDsaSigningKey,
        pq_vk: MlDsaVerifyingKey,
    }

    impl Identity {
        fn generate() -> Self {
            use rand::rngs::OsRng;
            let ed_sk = SigningKey::generate(&mut OsRng);
            let ed_vk = ed_sk.verifying_key();
            let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
            Self {
                ed_sk,
                ed_vk,
                pq_sk,
                pq_vk,
            }
        }

        fn did(&self) -> Did {
            Did::from_ed25519(&self.ed_vk.to_bytes())
        }
    }

    /// A trust store keyed by DID string → anchored ML-DSA-65 verifying key,
    /// implementing `UcanVerifier` over the real hybrid composite verify. Mirrors
    /// S5's `TestTrustStore`.
    struct TrustStore {
        pq_by_did: HashMap<String, MlDsaVerifyingKey>,
    }

    impl TrustStore {
        fn new() -> Self {
            Self {
                pq_by_did: HashMap::new(),
            }
        }
        fn anchor(&mut self, id: &Identity) {
            self.pq_by_did
                .insert(id.did().into_string(), id.pq_vk.clone());
        }
    }

    impl UcanVerifier for TrustStore {
        fn verify(
            &self,
            issuer: &Did,
            ed_key: &[u8; 32],
            payload: &[u8],
            signature: &[u8],
        ) -> Result<(), UcanError> {
            let ed_vk = VerifyingKey::from_bytes(ed_key)
                .map_err(|e| UcanError::BadSignature(e.to_string()))?;
            let pq_vk = self.pq_by_did.get(issuer.as_str());
            verify_composite(signature, &ed_vk, pq_vk, payload, UCAN_AAD, true)
                .map(|_| ())
                .map_err(|e| UcanError::BadSignature(e.to_string()))
        }
    }

    /// Build and hybrid-sign a UCAN from `issuer` to `audience` with `caps` and
    /// `proofs`. Mirrors S5's `signed_ucan` helper.
    fn signed_ucan(
        issuer: &Identity,
        audience: &Did,
        caps: Vec<Capability>,
        proofs: Vec<Ucan>,
    ) -> Ucan {
        let payload = UcanPayload {
            issuer: issuer.did(),
            audience: audience.clone(),
            capabilities: caps,
            not_before: None,
            expiration: Some(9_999_999_999),
            nonce: vec![1, 2, 3],
        };
        let bytes = payload.signing_bytes().unwrap();
        let signature =
            sign_composite(&issuer.ed_sk, Some(&issuer.pq_sk), &bytes, UCAN_AAD).unwrap();
        Ucan {
            payload,
            proofs,
            signature,
        }
    }

    fn cap(resource: &str, ability: &str) -> Capability {
        Capability::new(Resource::new(resource), Ability::new(ability))
    }

    /// A compartment bitset from bit indices.
    fn comps(bits: &[u32]) -> CompartmentSet {
        bits.iter().copied().collect()
    }

    /// A subject cleared to (Secret, PqHybrid, {0,1}) — dominates most test
    /// object labels. Built with PqHybrid key material so assurance is not the
    /// binding constraint.
    fn cleared_subject() -> SecurityContext {
        SecurityContext::new(Level::Secret, comps(&[0, 1]), VerifiedKeyMaterial::PqHybrid)
    }

    fn req(resource: &str, ability: &str, label: SecurityLabel) -> GrantRequest {
        GrantRequest {
            resource: Resource::new(resource),
            ability: Ability::new(ability),
            caveats: Caveats::default(),
            audience: None,
            object_label: Some(label),
        }
    }

    /// A single-capability self-issued root UCAN granting `cap`, hybrid-signed,
    /// plus a trust store anchoring its PQ key.
    fn root_grant(cap: Capability) -> (Ucan, TrustStore) {
        let id = Identity::generate();
        let u = signed_ucan(&id, &id.did(), vec![cap], vec![]);
        let mut s = TrustStore::new();
        s.anchor(&id);
        (u, s)
    }

    // ── Happy path: subset within ceiling, cleared, sender-bound ────────────

    #[test]
    fn subset_within_ceiling_permits_minimal_mint() {
        // Grant: mac://model/* / model/*. Request: mac://model/qwen / model/read
        // — a strict subset. Subject cleared, DPoP present, object at a label
        // the subject dominates.
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);

        let decision = match evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true, // sender-bound
        ) {
            Ok(d) => d,
            Err(e) => {
                panic!("a strict subset of a valid grant, cleared + bound, must permit: {e:?}")
            }
        };

        let granted = match decision {
            GrantDecision::Permit(g) => g,
            other => panic!("expected Permit, got {other:?}"),
        };
        // ZSP: the minted capability is the REQUESTED subset, not the grant.
        assert_eq!(
            granted.capability.resource,
            Resource::new("mac://model/qwen")
        );
        assert_eq!(granted.capability.ability, Ability::new("model/read"));
        // Cross-check: the reference compiler decision agrees (no parallel impl).
        assert_eq!(
            reference_authorize(
                grant.capabilities(),
                &AccessRequest {
                    resource: request.resource.clone(),
                    ability: request.ability.clone(),
                    caveats: request.caveats.clone(),
                }
            ),
            Decision::Permit
        );
    }

    // ── Fail-closed: over-ceiling (escalation) ──────────────────────────────

    #[test]
    fn over_ceiling_request_is_denied_not_escalated() {
        // Grant: mac://model/qwen only. Request: mac://model/* (wider resource)
        // — widening past the ceiling. Must deny; must NOT auto-mint the
        // requested wider capability.
        let (grant, store) = root_grant(cap("mac://model/qwen", "model/read"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/*", "model/read", object_label);

        let outcome = evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true,
        );
        match outcome {
            Err(GrantError::OverCeiling { .. }) => {}
            other => panic!("expected OverCeiling denial, got {other:?}"),
        }
    }

    #[test]
    fn over_ceiling_on_ability_is_denied() {
        // Grant: model/read. Request: model/write — wider ability. Deny.
        let (grant, store) = root_grant(cap("mac://model/qwen", "model/read"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/write", object_label);

        assert!(matches!(
            evaluate_grant(
                &grant,
                &store,
                NOW,
                &request,
                Some(&cleared_subject()),
                true
            ),
            Err(GrantError::OverCeiling { .. })
        ));
    }

    // ── Fail-closed: bad chain ──────────────────────────────────────────────

    #[test]
    fn broken_signature_is_denied() {
        // Tamper the grant's signature → S5 chain validation fails → S6 denies.
        let (mut grant, store) = root_grant(cap("mac://model/*", "model/*"));
        grant.signature[0] ^= 0xFF; // break the hybrid signature
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);

        match evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true,
        ) {
            Err(GrantError::Chain(_)) => {}
            other => panic!("expected Chain denial on bad signature, got {other:?}"),
        }
    }

    #[test]
    fn widening_chain_is_denied() {
        // A 2-link chain where the delegate widens past its delegator — S5
        // rejects the chain; S6 must surface that as a Chain denial (NOT a
        // ceiling check, because the chain never validated).
        let root = Identity::generate();
        let alice = Identity::generate();
        let root_ucan = signed_ucan(
            &root,
            &alice.did(),
            vec![cap("mac://model/*", "model/read")],
            vec![],
        );
        // alice widens: claims mac://other/* past root's mac://model/*.
        let alice_ucan = signed_ucan(
            &alice,
            &alice.did(),
            vec![cap("mac://other/*", "model/*")],
            vec![root_ucan],
        );
        let mut store = TrustStore::new();
        store.anchor(&root);
        store.anchor(&alice);

        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://other/x", "model/read", object_label);

        match evaluate_grant(
            &alice_ucan,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true,
        ) {
            Err(GrantError::Chain(ChainError::Widening { .. })) => {}
            other => panic!("expected Chain::Widening denial, got {other:?}"),
        }
    }

    #[test]
    fn expired_grant_is_denied() {
        // A grant that has expired at now=2000. S5's absolute-time gate denies;
        // S6 surfaces it as a Chain::Expired denial.
        let id = Identity::generate();
        let payload = UcanPayload {
            issuer: id.did(),
            audience: id.did(),
            capabilities: vec![cap("mac://model/*", "model/*")],
            not_before: None,
            expiration: Some(500), // expired well before now=2000
            nonce: vec![],
        };
        let bytes = payload.signing_bytes().unwrap();
        let signature = sign_composite(&id.ed_sk, Some(&id.pq_sk), &bytes, UCAN_AAD).unwrap();
        let grant = Ucan {
            payload,
            proofs: vec![],
            signature,
        };
        let mut store = TrustStore::new();
        store.anchor(&id);

        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);

        match evaluate_grant(
            &grant,
            &store,
            2000,
            &request,
            Some(&cleared_subject()),
            true,
        ) {
            Err(GrantError::Chain(ChainError::Expired { .. })) => {}
            other => panic!("expected Chain::Expired denial, got {other:?}"),
        }
    }

    // ── Fail-closed: insufficient MAC clearance ─────────────────────────────

    #[test]
    fn insufficient_clearance_is_denied() {
        // Grant valid for the request, but the object label requires a
        // compartment (bit 5) the subject is NOT cleared into. The S1 floor
        // denies independently of the ceiling.
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        // Object at {Confidential, PqHybrid, {5}} — subject only has {0,1}.
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[5]));
        let request = req("mac://model/qwen", "model/read", object_label);

        match evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true,
        ) {
            Err(GrantError::InsufficientClearance) => {}
            other => panic!("expected InsufficientClearance denial, got {other:?}"),
        }
    }

    #[test]
    fn classical_subject_denied_on_pq_object() {
        // #548 end-to-end at the grant path: an object labeled PqHybrid is
        // unreachable by a Classical-assurance subject via the SAME dominance
        // check — no separate path.
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        let object_label =
            SecurityLabel::new(Level::Public, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let request = req("mac://model/qwen", "model/read", object_label);
        // Classical subject (cleared high in level, but Classical assurance).
        let classical = SecurityContext::new(
            Level::Secret,
            comps(&[0, 1]),
            VerifiedKeyMaterial::Classical,
        );

        assert!(matches!(
            evaluate_grant(&grant, &store, NOW, &request, Some(&classical), true),
            Err(GrantError::InsufficientClearance)
        ));
    }

    // ── Fail-closed: missing sender-binding (ZSP) ───────────────────────────

    #[test]
    fn missing_sender_binding_is_denied() {
        // A request that would otherwise permit, but with no DPoP proof. ZSP:
        // the minted token MUST be sender-bound. Deny before any other gate.
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);

        match evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            false, // no sender-binding
        ) {
            Err(GrantError::MissingSenderBinding) => {}
            other => panic!("expected MissingSenderBinding denial, got {other:?}"),
        }
    }

    // ── Fail-closed: unlabeled subject / object, empty grant ────────────────

    #[test]
    fn unlabeled_subject_is_denied() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        let object_label =
            SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
        let request = req("mac://model/qwen", "model/read", object_label);

        // No subject context at all.
        assert!(matches!(
            evaluate_grant(&grant, &store, NOW, &request, None, true),
            Err(GrantError::UnlabeledSubject)
        ));
    }

    #[test]
    fn unlabeled_object_is_denied() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        // No object label supplied ⇒ the S1 floor cannot be evaluated ⇒ deny.
        let request = GrantRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("model/read"),
            caveats: Caveats::default(),
            audience: None,
            object_label: None,
        };

        assert!(matches!(
            evaluate_grant(
                &grant,
                &store,
                NOW,
                &request,
                Some(&cleared_subject()),
                true
            ),
            Err(GrantError::InsufficientClearance)
        ));
    }

    #[test]
    fn empty_grant_is_denied() {
        // A self-issued root with NO capabilities — an empty ceiling.
        let id = Identity::generate();
        let empty = signed_ucan(&id, &id.did(), vec![], vec![]);
        let mut store = TrustStore::new();
        store.anchor(&id);

        let object_label =
            SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
        let request = req("mac://x", "y", object_label);

        assert!(matches!(
            evaluate_grant(
                &empty,
                &store,
                NOW,
                &request,
                Some(&cleared_subject()),
                true
            ),
            Err(GrantError::EmptyGrant)
        ));
    }

    // ── Caveat subset honoured (more-restrictive request is still a subset) ──

    #[test]
    fn caveat_added_in_request_still_subset() {
        // Grant: mac://model/* / * with tenant=acme caveat. Request: same but
        // ALSO adds a max=5 caveat (more restricted) ⇒ still attenuates.
        use hyprstream_rpc::auth::ucan::capability::CaveatValue;
        use std::collections::BTreeMap;
        let mut g_caveats = BTreeMap::new();
        g_caveats.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let cap_with_caveat = Capability::with_caveats(
            Resource::new("mac://model/*"),
            Ability::new("model/*"),
            Caveats(g_caveats),
        );
        let (grant, store) = root_grant(cap_with_caveat);

        let mut r_caveats = BTreeMap::new();
        r_caveats.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        r_caveats.insert("max".to_owned(), CaveatValue::Int(5));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = GrantRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("model/read"),
            caveats: Caveats(r_caveats),
            audience: None,
            object_label: Some(object_label),
        };

        assert!(matches!(
            evaluate_grant(
                &grant,
                &store,
                NOW,
                &request,
                Some(&cleared_subject()),
                true
            ),
            Ok(GrantDecision::Permit(_))
        ));
    }

    #[test]
    fn caveat_dropped_in_request_is_over_ceiling() {
        // Grant has tenant=acme; request drops it ⇒ widening on caveats ⇒ deny.
        use hyprstream_rpc::auth::ucan::capability::CaveatValue;
        use std::collections::BTreeMap;
        let mut g_caveats = BTreeMap::new();
        g_caveats.insert("tenant".to_owned(), CaveatValue::Text("acme".to_owned()));
        let cap_with_caveat = Capability::with_caveats(
            Resource::new("mac://model/*"),
            Ability::new("model/*"),
            Caveats(g_caveats),
        );
        let (grant, store) = root_grant(cap_with_caveat);

        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        // Request has NO caveats — drops tenant ⇒ over-ceiling.
        let request = req("mac://model/qwen", "model/read", object_label);

        assert!(matches!(
            evaluate_grant(
                &grant,
                &store,
                NOW,
                &request,
                Some(&cleared_subject()),
                true
            ),
            Err(GrantError::OverCeiling { .. })
        ));
    }

    // ── Refresh is the same gate (ZSP: no weaker path) ─────────────────────

    #[test]
    fn refresh_runs_the_same_gates_as_mint() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/*"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);

        let mint = evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true,
        );
        let refresh = evaluate_refresh(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            true,
        );
        assert_eq!(format!("{mint:?}"), format!("{refresh:?}"));

        // And refresh denies on a missing binding just like mint.
        assert!(matches!(
            evaluate_refresh(
                &grant,
                &store,
                NOW,
                &request,
                Some(&cleared_subject()),
                false
            ),
            Err(GrantError::MissingSenderBinding)
        ));
    }

    // ── B2 (#674): the audited grant path ───────────────────────────────────

    /// An in-memory sink that records every [`crate::mac::audit::AuditRecord`]
    /// it's given, or can be told to fail the next `record` call (to exercise
    /// the fail-closed-on-audit-failure path without a real `WalAuditStore`).
    struct SpySink {
        records: parking_lot::Mutex<Vec<crate::mac::audit::AuditRecord>>,
        fail_next: std::sync::atomic::AtomicBool,
    }
    impl SpySink {
        fn new() -> Self {
            Self {
                records: parking_lot::Mutex::new(Vec::new()),
                fail_next: std::sync::atomic::AtomicBool::new(false),
            }
        }
        fn fail_next_write(&self) {
            self.fail_next.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        fn records(&self) -> Vec<crate::mac::audit::AuditRecord> {
            self.records.lock().clone()
        }
    }
    impl crate::mac::audit::AuditSink for SpySink {
        fn record(
            &self,
            record: &crate::mac::audit::AuditRecord,
        ) -> Result<(), crate::mac::audit::AuditError> {
            if self.fail_next.swap(false, std::sync::atomic::Ordering::SeqCst) {
                return Err(crate::mac::audit::AuditError::Io("injected failure".into()));
            }
            self.records.lock().push(record.clone());
            Ok(())
        }
    }

    /// A permit is audited (exactly one `Permit` record, with the grant-path
    /// sentinel ids) and the token still mints — auditing must not itself
    /// deny a decision the gates approved.
    #[test]
    fn audited_permit_is_recorded_and_still_mints() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/read"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);
        let sink = SpySink::new();

        let decision = audited_evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            None, // no delegator — single-principal (self-issued) grant
            true,
            &sink,
        );
        assert!(
            matches!(decision, Ok(GrantDecision::Permit(_))),
            "an honest permit must still mint under audit: {decision:?}"
        );

        let recs = sink.records();
        assert_eq!(recs.len(), 1, "exactly one record per decision");
        assert_eq!(recs[0].decision, crate::mac::te::Decision::Permit);
        assert_eq!(recs[0].reason, crate::mac::audit::DecisionReason::Permit);
        assert_eq!(recs[0].subject_type, crate::mac::te::GRANT_PATH_SUBJECT);
        assert_eq!(recs[0].object_type, crate::mac::te::GRANT_PATH_OBJECT);
        // Single-principal grant ⇒ no delegator recorded (#680/#681).
        assert!(
            recs[0].on_behalf_of.is_none(),
            "a self-issued grant records no delegator"
        );
    }

    /// #680/#681: a delegated grant records the DELEGATOR on the audit record's
    /// `on_behalf_of`, distinct from the effective (met) `subject_clearance` the
    /// gates evaluated — so the audit trail names both principals of a delegated
    /// decision (the user-attribution #445 asks for), not just the actor.
    #[test]
    fn delegated_grant_records_delegator_as_on_behalf_of() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/read"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);
        let sink = SpySink::new();

        // The delegator (authority source): a DISTINCT clearance from the met
        // subject (`cleared_subject` is Secret/{0,1}), so the record must not
        // conflate the two.
        let delegator = SecurityContext::new(Level::Secret, comps(&[0, 1, 2]), VerifiedKeyMaterial::PqHybrid);

        let decision = audited_evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()), // met context the gates evaluate
            Some(&delegator),         // delegator, for two-principal audit
            true,
            &sink,
        );
        assert!(matches!(decision, Ok(GrantDecision::Permit(_))));

        let recs = sink.records();
        assert_eq!(recs.len(), 1);
        // A delegated grant records its delegator.
        let obo = recs[0].on_behalf_of.unwrap();
        assert_eq!(
            obo.subject_clearance,
            *delegator.clearance(),
            "on_behalf_of carries the delegator's own clearance"
        );
        assert_ne!(
            obo.subject_clearance, recs[0].subject_clearance,
            "delegator (on_behalf_of) is distinct from the met subject clearance"
        );
        assert_eq!(obo.subject_type, crate::mac::te::GRANT_PATH_SUBJECT);
    }

    /// The core #674 obligation: a would-be Permit that CANNOT be durably
    /// audited must be downgraded to a deny (`AuditUnavailable`), never handed
    /// back as a Permit — mirroring `AuditedAvc`'s fail-closed rule at the TE
    /// path. Also asserts the resulting deny record IS still attempted/audited.
    #[test]
    fn permit_downgrades_to_deny_when_audit_write_fails() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/read"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);
        let sink = SpySink::new();
        sink.fail_next_write(); // the Permit's own audit write will fail

        let decision = audited_evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            None, // no delegator — single-principal (self-issued) grant
            true,
            &sink,
        );
        assert!(
            matches!(decision, Err(GrantError::AuditUnavailable)),
            "an unauditable permit must be downgraded, got {decision:?}"
        );
        // The fallback deny record was itself recorded (best-effort, per the
        // AuditedAvc precedent) — the second `record` call, which SpySink
        // allows since only the first was told to fail.
        let recs = sink.records();
        assert_eq!(recs.len(), 1, "the downgraded deny is recorded");
        assert_eq!(recs[0].decision, crate::mac::te::Decision::Deny);
        assert_eq!(
            recs[0].reason,
            crate::mac::audit::DecisionReason::GrantAuditFailClosed
        );
    }

    /// A denial that cannot be audited is still enforced as a deny (auditing
    /// failure never WEAKENS a decision — only a would-be Permit is affected).
    #[test]
    fn deny_stays_deny_when_audit_write_fails() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/read"));
        // Missing object label ⇒ InsufficientClearance (gate 4), independent of
        // the audit sink.
        let request = GrantRequest {
            resource: Resource::new("mac://model/qwen"),
            ability: Ability::new("model/read"),
            caveats: Caveats::default(),
            audience: None,
            object_label: None,
        };
        let sink = SpySink::new();
        sink.fail_next_write();

        let decision = audited_evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            None, // no delegator — single-principal (self-issued) grant
            true,
            &sink,
        );
        assert!(
            matches!(decision, Err(GrantError::InsufficientClearance)),
            "a deny must be enforced as decided even if its audit write fails: {decision:?}"
        );
        // The (failed) attempt leaves no record — asserting we don't silently
        // fabricate one; the failure was logged instead (see tracing::error!).
        assert!(sink.records().is_empty());
    }

    /// An unrecognized ability (doesn't parse as a canonical `ScopeAction`)
    /// still produces a well-formed audit record — using the sentinel
    /// `ACTION_UNRECOGNIZED` id rather than dropping the record. Diagnosable,
    /// not silently lost.
    #[test]
    fn unparseable_ability_still_produces_a_diagnosable_record() {
        let (grant, store) = root_grant(cap("mac://model/*", "totally-not-a-verb"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "totally-not-a-verb", object_label);
        let sink = SpySink::new();

        // The grant will actually Permit (the capability's ability string
        // matches verbatim), but the ability is not a canonical ScopeAction —
        // exercising the ACTION_UNRECOGNIZED fallback on a real decision.
        let _ = audited_evaluate_grant(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            None, // no delegator — single-principal (self-issued) grant
            true,
            &sink,
        );
        let recs = sink.records();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].action, crate::mac::te::ACTION_UNRECOGNIZED);
    }

    /// `audited_evaluate_refresh` audits identically to
    /// `audited_evaluate_grant` (same gate chain, same record shape) — the
    /// refresh path is not a quieter/less-audited sibling.
    #[test]
    fn audited_refresh_is_recorded_the_same_way() {
        let (grant, store) = root_grant(cap("mac://model/*", "model/read"));
        let object_label =
            SecurityLabel::new(Level::Confidential, Assurance::PqHybrid, comps(&[0]));
        let request = req("mac://model/qwen", "model/read", object_label);
        let sink = SpySink::new();

        let decision = audited_evaluate_refresh(
            &grant,
            &store,
            NOW,
            &request,
            Some(&cleared_subject()),
            None, // no delegator — single-principal (self-issued) grant
            true,
            &sink,
        );
        assert!(matches!(decision, Ok(GrantDecision::Permit(_))));
        assert_eq!(sink.records().len(), 1);
    }
}
