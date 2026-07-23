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
// S4 (#570): the daemon-boot compile → sign → verify-load → install path that
// finally populates `COMPILED_POLICY`.
pub mod bootload;
pub mod compiled;
pub mod compiler;
// S6 (#572): runtime grant path — UCAN grant-request → access/refresh tokens.
// Pure fail-closed core: grant validation + ceiling-subset + MAC clearance +
// sender-binding, consuming S1/S5/the compiler rather than re-implementing.
pub mod exchange;
// S1 activation (#567): production genesis CONTENT + enumerator + composite
// ObjectLabelResolver + boot-time coverage gate consumed by the active 9P PEP.
pub mod genesis;
pub mod lattice;
// #676: the production S3-scope ↔ S5-TE-rule vocabulary (injective + exact;
// wildcards expand at compile time over a closed registry).
pub mod pep;
pub mod permission_map;
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
// S1 activation (#567): production genesis content/enumerator/resolver/gate.
pub use genesis::{
    floor_label, genesis_lattice, CompositeObjectLabelResolver, GenesisGate, ManifestLabelSource,
    NamespaceEnumerator, NoManifests, SitePolicy,
};
pub use pep::NinePAccessDecider;
// S4 (#570): the boot path that installs the verified `CompiledPolicy` at daemon
// startup (dormant — makes the PDP inputs real without enabling enforcement).
pub use bootload::{compile_sign_load_install, install_baseline_boot_policy, BootPolicyError};

/// Construct the [`UcanVerifier`] the HTTP grant path uses to validate a
/// presented UCAN's signatures.
///
/// This is the S6↔trust-store seam. **S8 (#574) wires it to the process-global
/// `PqTrustStore`** (the same `register_pq_trust` binding the rest of the TCB
/// uses, installed via `envelope::install_verify_config`). The verifier resolves
/// each UCAN issuer's anchored ML-DSA-65 verifying key from that store and
/// verifies the hybrid composite with `require_pq = true` (the Hybrid policy).
///
/// Returns `None` (→ the HTTP grant path denies every request, fail-closed) when
/// no PQ trust store has been installed — a node that has not configured hybrid
/// verification cannot validate hybrid-signed UCANs, and `evaluate_grant` never
/// runs against an unverified chain.
///
/// The core `evaluate_grant` is exercised through its own tests with a real
/// verifier, so the security-critical logic is covered independently of this
/// wiring.
pub fn exchange_ucan_verifier(
    _state: &crate::services::oauth::state::OAuthState,
) -> Option<Box<dyn hyprstream_rpc::auth::ucan::token::UcanVerifier + Send + Sync>> {
    // The verifier needs the kid-anchored PQ store (the same `register_pq_trust`
    // binding the envelope verifier uses). If none is installed, fail-closed.
    let pq_store = hyprstream_rpc::envelope::global_pq_store()?;
    Some(Box::new(GlobalPqUcanVerifier { pq_store }))
}

/// A `UcanVerifier` backed by the process-global `PqTrustStore` (the same store
/// the envelope verifier uses). Resolves each UCAN issuer's anchored ML-DSA-65
/// key and verifies the hybrid composite under the Hybrid policy (`require_pq`).
///
/// This is the S8 activation: the HTTP grant path now validates hybrid UCAN
/// signatures against the node's trust store rather than denying by default.
struct GlobalPqUcanVerifier {
    pq_store: std::sync::Arc<dyn hyprstream_rpc::envelope::PqTrustStore>,
}

impl hyprstream_rpc::auth::ucan::token::UcanVerifier for GlobalPqUcanVerifier {
    fn verify(
        &self,
        issuer: &hyprstream_rpc::auth::ucan::token::Did,
        ed_key: &[u8; 32],
        payload: &[u8],
        signature: &[u8],
    ) -> Result<(), hyprstream_rpc::auth::ucan::token::UcanError> {
        use hyprstream_rpc::auth::ucan::token::UcanError;
        let ed_vk = ed25519_dalek::VerifyingKey::from_bytes(ed_key)
            .map_err(|e| UcanError::BadSignature(e.to_string()))?;
        // kid-anchoring: resolve the issuer's trusted ML-DSA-65 key from the
        // global PQ store, keyed by the Ed25519 identity (the cnf-equivalent).
        // The `issuer` DID encodes the same Ed25519 key; the store is keyed by
        // the raw bytes (the canonical anchor), so we resolve via `ed_key`.
        // No anchor ⇒ fail-closed under Hybrid (require_pq).
        let _ = issuer; // available for future per-DID anchoring policy.
        let pq_vk = self.pq_store.ml_dsa_key_for(ed_key);
        hyprstream_rpc::crypto::cose_sign::verify_composite(
            signature,
            &ed_vk,
            pq_vk.as_ref(),
            payload,
            hyprstream_rpc::auth::ucan::token::UCAN_AAD,
            /* require_pq */ true,
        )
        .map(|_| ())
        .map_err(|e| UcanError::BadSignature(e.to_string()))
    }
}

/// Process-global installed policy the [`exchange_enrollment_resolver`] seam reads.
/// Mirrors `hyprstream_rpc::envelope::global_pq_store`'s pattern: a node loads and
/// verifies a [`SignedPolicy`] via [`PolicyLoader`] at startup and installs the
/// result once here; a node that hasn't leaves this `None` and the seam below
/// fails closed.
static COMPILED_POLICY: std::sync::OnceLock<std::sync::Arc<CompiledPolicy>> =
    std::sync::OnceLock::new();

/// Install the node's verified [`CompiledPolicy`] for the
/// [`exchange_enrollment_resolver`] seam. **Write-once per process** (backed by a
/// `OnceLock`): returns `true` if THIS call installed `policy`, `false` if a
/// policy was already installed and this call was a no-op. The caller MUST have
/// already verified the policy via [`PolicyLoader`] — this does not re-verify.
///
/// The `bool` return makes the write-once contract honest: because the slot is a
/// `OnceLock`, the **first** installed policy pins the seam for the process
/// lifetime. Today the only caller is the boot baseline
/// ([`bootload::install_baseline_boot_policy`]), so this is exactly-once; a
/// future config-driven real policy CANNOT swap the baseline out through this API
/// (it would get `false` and leave the empty baseline in place). Making the seam
/// swap-capable — e.g. an `ArcSwap<CompiledPolicy>` keyed by policy generation so
/// a newer, verified generation replaces an older one atomically — is the
/// follow-up; until then callers must treat `false` as "the global seam still
/// holds the earlier policy", never as success.
// `pub(crate)` — this seam installs `policy` WITHOUT re-verifying it (the caller
// MUST have already run it through `PolicyLoader`). Keeping it crate-private
// prevents an out-of-crate caller from bypassing the verify-once-at-load path and
// pinning an unsigned / classical-only policy into the process-global seam. The
// only caller is `bootload`, which self-verifies through the full sign→load path.
#[must_use]
pub(crate) fn install_compiled_policy(policy: std::sync::Arc<CompiledPolicy>) -> bool {
    COMPILED_POLICY.set(policy).is_ok()
}

/// The installed [`CompiledPolicy`], if any. `None` on a node that hasn't loaded
/// one — callers MUST fail closed (see [`exchange_enrollment_resolver`]), never
/// substitute a permissive default.
pub fn compiled_policy() -> Option<std::sync::Arc<CompiledPolicy>> {
    COMPILED_POLICY.get().cloned()
}

/// The `SubjectContextResolver` for a delegated actor, backed by the signed
/// policy's enrollment table (`CompiledPolicy::clearance_for`, #698 PR A).
///
/// **#698 Decision D** (operator-ratified, 2026-07-03): a delegated actor proves
/// only DPoP possession of a classical ephemeral key at the grant path — UCAN
/// chain validation (`GlobalPqUcanVerifier`, `require_pq: true`) proves *issuers*,
/// never the audience — so this resolver clamps assurance to
/// `VerifiedKeyMaterial::Classical` unconditionally, regardless of the enrolled
/// clearance's own level/compartments. That is the truthful label for what the
/// crypto actually proves; assigning anything higher would let an unverified
/// actor claim PqHybrid assurance, defeating #548. An object labeled with a
/// PqHybrid assurance requirement is therefore structurally unreachable through
/// this resolver — by construction, not by a missing rule (see
/// `docs/mac-architecture.md`). Raising a specific enrolled actor above Classical
/// is the enrollment-key-registration follow-up (#718), not this resolver.
///
/// An unenrolled DID resolves to `None` — the existing fail-closed contract
/// (`SubjectContextResolver::resolve`'s `None` ⇒ deny) is unchanged.
pub struct EnrollmentSubjectContextResolver {
    policy: std::sync::Arc<CompiledPolicy>,
}

impl EnrollmentSubjectContextResolver {
    /// Construct a resolver over a verified, installed [`CompiledPolicy`].
    pub fn new(policy: std::sync::Arc<CompiledPolicy>) -> Self {
        Self { policy }
    }
}

impl crate::services::oauth::token_exchange::SubjectContextResolver
    for EnrollmentSubjectContextResolver
{
    fn resolve(&self, audience_did: &str) -> Option<SecurityContext> {
        let clearance = self.policy.clearance_for(audience_did)?;
        // Decision D: floor at Classical no matter what the enrollment table's
        // clearance component says — the assurance axis is what's cryptographically
        // proven here, not what's authorized.
        Some(SecurityContext::from_clearance(
            clearance,
            VerifiedKeyMaterial::Classical,
        ))
    }
}

/// Construct the `SubjectContextResolver` the HTTP grant path resolves delegated
/// actors' MAC context through (#698 Decision D).
///
/// Returns [`EnrollmentSubjectContextResolver`] when a compiled policy has been
/// installed via [`install_compiled_policy`]; falls back to
/// [`crate::services::oauth::token_exchange::DenyUnlabeledResolver`] (fail-closed)
/// on a node that has not loaded one — mirrors [`exchange_ucan_verifier`]'s
/// Option-to-fail-closed pattern.
pub fn exchange_enrollment_resolver(
) -> Box<dyn crate::services::oauth::token_exchange::SubjectContextResolver> {
    match compiled_policy() {
        Some(policy) => Box::new(EnrollmentSubjectContextResolver::new(policy)),
        None => Box::new(crate::services::oauth::token_exchange::DenyUnlabeledResolver),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod enrollment_resolver_tests {
    use super::*;
    use crate::services::oauth::token_exchange::SubjectContextResolver as _;
    use std::collections::BTreeMap;

    fn test_lattice() -> Lattice {
        Lattice::new(LatticeVersion(1), [])
    }

    fn policy_with(did: &str, clearance: SecurityLabel) -> std::sync::Arc<CompiledPolicy> {
        let lattice = test_lattice();
        let mut policy = CompiledPolicy::new(TeMatrix::default(), &lattice);
        policy = policy.with_enrollment(BTreeMap::from([(did.to_owned(), clearance)]));
        std::sync::Arc::new(policy)
    }

    #[test]
    fn enrolled_did_resolves_at_classical_regardless_of_table_assurance() {
        // Enrollment table asserts PqHybrid for this DID; the resolver must
        // still floor the resulting SecurityContext at Classical (#698 Decision
        // D) — the table's own assurance component is not authoritative here.
        let clearance =
            SecurityLabel::new(Level::Secret, Assurance::PqHybrid, CompartmentSet::EMPTY);
        let resolver =
            EnrollmentSubjectContextResolver::new(policy_with("did:key:actor", clearance));

        // enrolled DID must resolve
        let ctx = resolver.resolve("did:key:actor").unwrap();

        assert_eq!(ctx.clearance().assurance, Assurance::Classical);
        assert_eq!(ctx.clearance().level, Level::Secret);
    }

    #[test]
    fn unenrolled_did_denies() {
        let resolver = EnrollmentSubjectContextResolver::new(policy_with(
            "did:key:actor",
            SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY),
        ));

        assert!(resolver.resolve("did:key:someone-else").is_none());
    }
}
