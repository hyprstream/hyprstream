//! CAS-native MAC Policy Enforcement Point (#1270, epic #1267 T3).
//!
//! The CAS data plane — substrate ingest, `CasMount` read, registry streaming
//! continuation, and xet HTTP — had **no native-MAC PEP** and no trusted label
//! at seal (T2 audit `.fleet-coord/mac-pep-audit.md`, rows "CAS substrate
//! ingest" / "CAS substrate read" / "`CasMount` object/xorb open/read/stat" /
//! "Registry CAS read continuation" / "Xet HTTP CAS read"). Every read was
//! either unmediated or gated only by `BootstrapCasAuthorizer` ("any
//! authenticated caller"), with labels caller-asserted or absent.
//!
//! This module provides the CAS-plane PEP that all CAS read paths consume. It
//! uses the canonical RPC MAC contract's [`MacDecision`], [`MacDenyReason`],
//! and [`RpcObjectLabelResolver`] seam while deliberately not implementing
//! [`MacDispatchPep`](hyprstream_rpc::auth::mac::MacDispatchPep), whose
//! `EnvelopeContext` input is RPC-plane-specific.
//!
//! - **Trusted content-bound label at seal** ([`domain_label`] /
//!   [`seal_label`]): derived from the [`DedupDomain`]'s structural properties
//!   (trust boundary → level axis), **not** caller-asserted. The domain is the
//!   content's provenance — it encodes *where the content lives* (local vs
//!   shared-remote) and *which tenant it belongs to* (compartment). The staging
//!   path's joined hint label is a D1 hint trusted only from our own ingest
//!   service; the domain label is authoritative and the effective seal label is
//!   the restrict-only `join` of the two.
//!
//! - **Clearance-input seam** ([`CasClearanceSource`]): consumes authority-bound
//!   subject + `verified_tenant` inputs. Until a production CAS adapter is
//!   installed, [`DenyAllClearanceSource`] returns `None` for every subject —
//!   **fail-closed**, never a permissive default. There is no permissive mode
//!   (per #547).
//!
//! - **Per-op enforcement** ([`CasPep`]): resolves the subject clearance via
//!   the seam, checks `clearance.can_access(object_label)`, and records every
//!   decision through the existing tamper-evident MAC audit sink. Missing
//!   subject, missing clearance, missing label, or audit failure → deny.
//!
//! - **`CasMount` integration** ([`MacCasAuthorizer`]): implements
//!   [`CasMountAuthorizer`](`crate::storage::cas::mount::CasMountAuthorizer`)
//!   so that the existing per-op authorization hook in `CasMount::open`/
//!   `read`/`stat` delegates to the MAC PEP without adding a new enforcement
//!   surface.
//!
//! ## What this module does NOT do
//!
//! - It does not author, derive, or enforce MAC policy from the `Compartment`
//!   string in [`DedupDomain`] — that requires lattice internment which lives
//!   in the policy layer. Compartment isolation is already structural (distinct
//!   physical storage roots per domain); the label's level axis is the
//!   load-bearing enforcement here. Enriching the seal label with compartment
//!   bits via the lattice is a follow-up gated on threading the lattice to the
//!   CAS layer.
//!
//! - It does not fake a clearance for any subject. If the clearance source
//!   returns `None`, the installed PEP denies. The #698 clearance primitives
//!   have landed, but a plane-specific adapter must still be installed before
//!   a CAS subject can pass.

use std::sync::Arc;

use anyhow::Context as _;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::mac::{
    MacDecision, MacDenyReason, RpcObjectLabelResolver, SecurityContext, SecurityLabel,
};
use hyprstream_vfs::Subject;

use crate::mac::audit::{AuditRecord, AuditSink, DecisionReason};
use crate::mac::te::{Decision, ObjectType, SubjectType};
use crate::storage::cas::{DedupDomain, TrustBoundary};

/// Reserved audit type ids for CAS PEP decisions (below the grant-path
/// `u32::MAX` sentinels, above the 9P sentinels).
const CAS_SUBJECT_TYPE: SubjectType = SubjectType(u32::MAX - 3);
const CAS_OBJECT_TYPE: ObjectType = ObjectType(u32::MAX - 3);
const CAS_SERVICE_DOMAIN: &str = "cas";

// ────────────────────────────────────────────────────────────────────────────
// Trusted content-bound label derivation
// ────────────────────────────────────────────────────────────────────────────

/// Derive a **trusted, content-bound** [`SecurityLabel`] from a [`DedupDomain`].
///
/// The label is derived from the domain's structural properties — the trust
/// boundary determines the level axis, and the assurance floors at `Classical`
/// (the substrate cannot prove PQ-hybrid for stored content). This is **not**
/// caller-asserted: the domain is the content's physical provenance, set by the
/// storage layer at domain construction, not by the ingest caller.
///
/// Mapping:
/// - [`TrustBoundary::Local`] → [`Level::Internal`](hyprstream_rpc::auth::mac::Level::Internal):
///   node-local content is at least org-internal.
/// - [`TrustBoundary::SharedRemote`] → [`Level::Confidential`](hyprstream_rpc::auth::mac::Level::Confidential):
///   federated/remote content carries higher sensitivity.
///
/// Compartment bits are **not** populated here: the lattice internment lives in
/// the policy layer, not the storage layer. Compartment isolation is already
/// structural — each domain maps to a distinct physical storage root
/// ([`DedupDomain::relative_path`]) — so cross-compartment content leakage is
/// physically impossible at the substrate level. The MAC label's level axis
/// provides the independent clearance enforcement on top.
///
/// Enriching the seal label with interned compartment bits is a follow-up gated
/// on threading the [`Lattice`](hyprstream_rpc::auth::mac::Lattice) to the CAS
/// layer.
pub fn domain_label(domain: &DedupDomain) -> SecurityLabel {
    use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
    let level = match domain.trust_boundary {
        TrustBoundary::Local => Level::Internal,
        TrustBoundary::SharedRemote => Level::Confidential,
    };
    SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
}

/// Compute the **effective seal label**: `join(domain_label(domain), hint)`.
///
/// The domain label is authoritative (content-bound). The `hint` is a D1 label
/// from our own staging path — trusted as a *restrict-only* input, never
/// permissive. `join` (BLP least-upper-bound) can only make the effective label
/// *more* restrictive than either input, never less.
///
/// Callers:
/// - [`CasSubstrate::put`](crate::storage::CasSubstrate::put) calls this when a
///   staging hint is supplied, stamping the result on the
///   [`BlobManifest`](crate::storage::cas::BlobManifest) carrier field.
/// - The seal path in `CasMount::seal_slot` passes the staging-joined label as
///   the hint.
pub fn seal_label(domain: &DedupDomain, hint: Option<SecurityLabel>) -> SecurityLabel {
    let base = domain_label(domain);
    match hint {
        Some(h) => base.join(&h),
        None => base,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Canonical object-label resolver seam
// ────────────────────────────────────────────────────────────────────────────

/// Adapt a trusted CAS carrier label to the canonical object-label resolver.
///
/// CAS supplies the concrete label from either a sealed manifest or the
/// structural [`DedupDomain`] floor. The RPC-only method discriminator is
/// always `None` on this plane.
#[derive(Debug, Clone, Copy)]
pub struct CasObjectLabelResolver {
    label: Option<SecurityLabel>,
}

impl CasObjectLabelResolver {
    /// Resolve a known trusted label.
    pub const fn new(label: Option<SecurityLabel>) -> Self {
        Self { label }
    }

    /// Resolve the structural floor for a CAS dedup domain.
    pub fn from_domain(domain: &DedupDomain) -> Self {
        Self::new(Some(domain_label(domain)))
    }
}

impl RpcObjectLabelResolver for CasObjectLabelResolver {
    fn resolve(&self, service_domain: &str, method: Option<u16>) -> Option<SecurityLabel> {
        if service_domain == CAS_SERVICE_DOMAIN && method.is_none() {
            self.label
        } else {
            None
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Clearance-input seam
// ────────────────────────────────────────────────────────────────────────────

/// Resolve a verified subject identity to a MAC [`SecurityContext`] for CAS
/// access.
///
/// The CAS PEP calls
/// [`CasClearanceSource::clearance_for`] with the verified subject and tenant
/// binding (extracted from `EnvelopeContext` or an equivalent verified HTTP
/// identity) and receives a `SecurityContext` — **never** a caller-supplied
/// clearance.
///
/// `None` ⇒ **unresolvable** ⇒ the PEP MUST deny. Implementors must NOT
/// manufacture a permissive default.
///
/// The #698 clearance primitives are available, but until a CAS-specific
/// adapter is installed [`DenyAllClearanceSource`] returns `None` for every
/// subject.
pub trait CasClearanceSource: Send + Sync {
    /// Resolve `subject_id` to its MAC clearance, or `None` if unresolvable.
    fn clearance_for(
        &self,
        subject_id: &str,
        verified_tenant: Option<&str>,
    ) -> Option<SecurityContext>;
}

/// Fail-closed clearance source: denies every subject.
///
/// Every CAS read through this explicitly installed source denies. This is the
/// safe default while no authority-backed CAS adapter is configured.
pub struct DenyAllClearanceSource;

impl CasClearanceSource for DenyAllClearanceSource {
    fn clearance_for(
        &self,
        _subject_id: &str,
        _verified_tenant: Option<&str>,
    ) -> Option<SecurityContext> {
        None
    }
}

/// Production CAS clearance source backed only by contexts derived from
/// verified `Claims × VerifiedKeyMaterial` at an RPC/attach boundary.
#[derive(Debug, Default, Clone, Copy)]
pub struct VerifiedClaimsCasClearanceSource;

impl CasClearanceSource for VerifiedClaimsCasClearanceSource {
    fn clearance_for(
        &self,
        subject_id: &str,
        verified_tenant: Option<&str>,
    ) -> Option<SecurityContext> {
        hyprstream_rpc::auth::mac::subject_context(
            &Subject::new(subject_id.to_owned()),
            verified_tenant,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CAS PEP — per-op enforcement
// ────────────────────────────────────────────────────────────────────────────

/// The CAS MAC Policy Enforcement Point.
///
/// Holds the clearance-input seam ([`CasClearanceSource`]) and the audit sink.
/// Every CAS read path calls [`CasPep::check_read`] before serving bytes:
///
/// 1. Resolve the subject clearance via the seam → `None` = deny (fail-closed).
/// 2. Check `clearance.can_access(object_label)`.
/// 3. Audit every decision (a decision that cannot be durably audited is
///    downgraded to `Deny`, same fail-closed contract as
///    [`NinePAccessDecider`](crate::mac::pep::NinePAccessDecider)).
pub struct CasPep {
    clearance_source: Arc<dyn CasClearanceSource>,
    sink: Arc<dyn AuditSink>,
}

impl CasPep {
    /// Construct a CAS PEP with the given clearance source and audit sink.
    pub fn new(clearance_source: Arc<dyn CasClearanceSource>, sink: Arc<dyn AuditSink>) -> Self {
        Self {
            clearance_source,
            sink,
        }
    }

    /// Construct a fail-closed CAS PEP using [`DenyAllClearanceSource`] and the
    /// null audit sink. Every read denies. Use this at production constructors
    /// where no production CAS clearance adapter is configured.
    pub fn fail_closed() -> Self {
        Self::new(
            Arc::new(DenyAllClearanceSource),
            Arc::new(crate::mac::audit::NullAuditSink),
        )
    }

    /// Check read access for a verified subject against a trusted object-label
    /// resolver.
    ///
    /// Returns the canonical [`MacDecision`]. An installed CAS PEP is
    /// fail-closed for:
    /// - missing clearance (subject not resolvable via the seam)
    /// - missing object label (resolver returns `None`, per #547)
    /// - audit failure (downgraded to deny)
    #[must_use]
    pub fn check_read(
        &self,
        subject_id: &str,
        verified_tenant: Option<&str>,
        resolver: &dyn RpcObjectLabelResolver,
    ) -> MacDecision {
        self.check(
            subject_id,
            verified_tenant,
            resolver.resolve(CAS_SERVICE_DOMAIN, None),
        )
    }

    /// Internal: resolve clearance, check dominance, audit.
    fn check(
        &self,
        subject_id: &str,
        verified_tenant: Option<&str>,
        label: Option<SecurityLabel>,
    ) -> MacDecision {
        let Some(ctx) = self
            .clearance_source
            .clearance_for(subject_id, verified_tenant)
        else {
            return self.audit(
                subject_id,
                verified_tenant,
                None,
                label,
                MacDecision::Deny(MacDenyReason::NoClearance),
            );
        };

        let Some(label) = label else {
            return self.audit(
                subject_id,
                verified_tenant,
                Some(&ctx),
                None,
                MacDecision::Deny(MacDenyReason::UnlabeledObject),
            );
        };

        let decision = if ctx.can_access(&label) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        };
        self.audit(
            subject_id,
            verified_tenant,
            Some(&ctx),
            Some(label),
            decision,
        )
    }

    fn audit(
        &self,
        subject_id: &str,
        verified_tenant: Option<&str>,
        ctx: Option<&SecurityContext>,
        label: Option<SecurityLabel>,
        decision: MacDecision,
    ) -> MacDecision {
        use std::time::{SystemTime, UNIX_EPOCH};

        let (audit_decision, reason) = match decision {
            MacDecision::Permit => (Decision::Permit, DecisionReason::Permit),
            MacDecision::Deny(MacDenyReason::NoClearance) => {
                (Decision::Deny, DecisionReason::NoClearance)
            }
            MacDecision::Deny(MacDenyReason::UnlabeledObject) => {
                (Decision::Deny, DecisionReason::UnlabeledObject)
            }
            MacDecision::Deny(
                MacDenyReason::FloorDeny
                | MacDenyReason::NoPepInstalled
                | MacDenyReason::StaleAuthority,
            ) => (Decision::Deny, DecisionReason::FloorDeny),
        };
        let policy = crate::mac::compiled_policy();
        let generation = policy.as_ref().map_or(0, |p| p.generation);
        let policy_hash = policy.as_ref().and_then(|p| p.policy_hash().ok());
        let record = AuditRecord {
            seq: 0,
            prev_hash: [0; 32],
            ts_unix_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
            decision: audit_decision,
            generation,
            policy_hash,
            subject_type: CAS_SUBJECT_TYPE,
            subject_clearance: ctx
                .map(|c| *c.clearance())
                .unwrap_or_else(SecurityLabel::bottom),
            on_behalf_of: None,
            object_type: CAS_OBJECT_TYPE,
            object_label: label.unwrap_or_else(SecurityLabel::bottom),
            action: crate::mac::te::Action::from_scope_action(crate::mac::te::ScopeAction::Query),
            reason,
            subject_id: Some(subject_id.to_owned()),
            object_id: Some(match verified_tenant {
                Some(tenant) => format!("{tenant}/{CAS_SERVICE_DOMAIN}"),
                None => CAS_SERVICE_DOMAIN.to_owned(),
            }),
        };

        match self.sink.record(&record) {
            Ok(()) => decision,
            Err(error) => {
                let deny_record = AuditRecord {
                    decision: Decision::Deny,
                    reason: DecisionReason::AuditFailClosed,
                    ..record
                };
                let _ = self.sink.record(&deny_record);
                tracing::error!(
                    target: "hyprstream.mac.cas_pep",
                    %error,
                    reason = DecisionReason::AuditFailClosed.as_str(),
                    "CAS PEP decision could not be durably audited; enforcing deny"
                );
                // The canonical contract has no audit-specific deny variant.
                // Keep the detailed cause in the audit vocabulary and expose a
                // conservative floor denial to shared consumers.
                MacDecision::Deny(MacDenyReason::FloorDeny)
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CasMount integration
// ────────────────────────────────────────────────────────────────────────────

/// A [`CasMountAuthorizer`] backed by the CAS MAC PEP.
///
/// This replaces [`BootstrapCasAuthorizer`] at production constructors. On each
/// `open`/`read`/`stat` it derives the content-bound label from the request's
/// [`DedupDomain`] and checks the subject clearance via the PEP.
///
/// The label is derived from the domain (structural, trusted), NOT from any
/// caller-supplied value. This is the trusted content-bound label at the point
/// of use.
pub struct MacCasAuthorizer {
    pep: Arc<CasPep>,
}

impl MacCasAuthorizer {
    /// Construct a MAC CAS authorizer wrapping the given PEP.
    pub fn new(pep: Arc<CasPep>) -> Self {
        Self { pep }
    }

    /// Construct a fail-closed MAC CAS authorizer (every read denies).
    /// Use this at production constructors where no authority-backed CAS
    /// clearance adapter is configured.
    pub fn fail_closed() -> Self {
        Self::new(Arc::new(CasPep::fail_closed()))
    }
}

impl crate::storage::cas::CasMountAuthorizer for MacCasAuthorizer {
    fn authorize(
        &self,
        caller: &Subject,
        request: crate::storage::cas::CasMountAuthzRequest<'_>,
    ) -> Result<(), hyprstream_vfs::MountError> {
        let resolver = CasObjectLabelResolver::from_domain(request.domain);
        let decision = self
            .pep
            .check_read(&caller.to_string(), request.verified_tenant, &resolver);
        if decision.is_permit() {
            Ok(())
        } else {
            Err(hyprstream_vfs::MountError::PermissionDenied(format!(
                "CAS MAC {} {:?} {} denied for {}: {decision:?} (#1270)",
                request.operation, request.kind, request.address, caller,
            )))
        }
    }
}

/// Assemble the production CAS PEP from verified subject contexts and a
/// tamper-evident signed WAL.  Failure to obtain either signing key aborts the
/// production constructor; it never substitutes the test-only null sink.
pub async fn production_cas_pep(
    signing_key: SigningKey,
    oauth: &crate::config::OAuthConfig,
    audit_stream: &str,
) -> anyhow::Result<Arc<CasPep>> {
    anyhow::ensure!(
        !audit_stream.is_empty()
            && audit_stream
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_'),
        "invalid CAS MAC audit stream name"
    );
    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
    let ml_dsa_store = crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, oauth);
    let signer = crate::mac::audit::cose::OwnedCoseAuditSigner::new(
        Arc::new(signing_key),
        ml_dsa_store.active_key().await,
        hyprstream_rpc::envelope::mandatory_envelope_policy(),
    );
    anyhow::ensure!(
        signer.can_sign(),
        "CAS MAC PEP audit signer unavailable under mandatory Hybrid policy"
    );
    let audit_store = crate::mac::audit::WalAuditStore::open(
        secrets_dir.join("mac-audit").join(audit_stream),
        signer,
    )
    .context("open CAS MAC audit store")?;
    Ok(Arc::new(CasPep::new(
        Arc::new(VerifiedClaimsCasClearanceSource),
        Arc::new(audit_store),
    )))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use parking_lot::Mutex;

    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityContext, SecurityLabel, VerifiedKeyMaterial,
    };

    use super::*;
    use crate::mac::audit::{AuditError, AuditRecord};
    use crate::storage::cas::{CasMountAuthorizer, CasMountAuthzRequest, CasMountObjectKind};
    use crate::storage::cas::{DedupDomain, TrustBoundary};

    // ── domain_label ────────────────────────────────────────────────────────

    #[test]
    fn local_domain_label_is_internal() {
        let label = domain_label(&DedupDomain::local_default());
        assert_eq!(label.level, Level::Internal);
        assert_eq!(label.assurance, Assurance::Classical);
        assert!(label.compartments.is_empty());
    }

    #[test]
    fn shared_remote_domain_label_is_confidential() {
        let domain = DedupDomain {
            trust_boundary: TrustBoundary::SharedRemote,
            ..DedupDomain::local_default()
        };
        let label = domain_label(&domain);
        assert_eq!(label.level, Level::Confidential);
    }

    // ── seal_label ──────────────────────────────────────────────────────────

    #[test]
    fn seal_label_without_hint_is_domain_label() {
        let domain = DedupDomain::local_default();
        assert_eq!(seal_label(&domain, None), domain_label(&domain));
    }

    #[test]
    fn seal_label_join_is_restrict_only() {
        // Domain label = Internal; hint = Secret → join = Secret (more restrictive).
        let domain = DedupDomain::local_default();
        let hint = SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY);
        let effective = seal_label(&domain, Some(hint));
        assert_eq!(effective.level, Level::Secret);
    }

    #[test]
    fn seal_label_hint_cannot_lower_below_domain() {
        // Domain label = Confidential (shared-remote); hint = Public → join stays
        // Confidential (join never lowers).
        let domain = DedupDomain {
            trust_boundary: TrustBoundary::SharedRemote,
            ..DedupDomain::local_default()
        };
        let hint = SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
        let effective = seal_label(&domain, Some(hint));
        assert_eq!(
            effective.level,
            Level::Confidential,
            "seal label must never be less restrictive than the domain floor"
        );
    }

    #[test]
    fn cas_resolver_uses_canonical_service_and_non_browser_method() {
        let resolver = CasObjectLabelResolver::from_domain(&DedupDomain::local_default());
        assert!(resolver.resolve(CAS_SERVICE_DOMAIN, None).is_some());
        assert!(resolver.resolve("registry", None).is_none());
        assert!(resolver.resolve(CAS_SERVICE_DOMAIN, Some(1)).is_none());
    }

    // ── DenyAllClearanceSource ──────────────────────────────────────────────

    #[test]
    fn deny_all_clearance_source_returns_none() {
        let src = DenyAllClearanceSource;
        assert!(src.clearance_for("anyone", Some("tenant-a")).is_none());
        assert!(src.clearance_for("admin", None).is_none());
    }

    // ── CasPep fail-closed ──────────────────────────────────────────────────

    #[test]
    fn fail_closed_pep_denies_every_read() {
        let pep = CasPep::fail_closed();
        let label = domain_label(&DedupDomain::local_default());
        let labeled = CasObjectLabelResolver::new(Some(label));
        assert_eq!(
            pep.check_read("any-subject", Some("tenant-a"), &labeled),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );
        // Unlabeled object also denies.
        let unlabeled = CasObjectLabelResolver::new(None);
        assert_eq!(
            pep.check_read("any-subject", Some("tenant-a"), &unlabeled),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );
    }

    // ── CasPep with a real clearance source ─────────────────────────────────

    struct FixtureClearance {
        subject: String,
        ctx: SecurityContext,
    }

    impl CasClearanceSource for FixtureClearance {
        fn clearance_for(
            &self,
            subject_id: &str,
            _verified_tenant: Option<&str>,
        ) -> Option<SecurityContext> {
            if subject_id == self.subject {
                Some(self.ctx.clone())
            } else {
                None
            }
        }
    }

    fn ctx(level: Level) -> SecurityContext {
        SecurityContext::from_clearance(
            SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY),
            VerifiedKeyMaterial::Classical,
        )
    }

    struct TenantClearance;

    impl CasClearanceSource for TenantClearance {
        fn clearance_for(
            &self,
            subject_id: &str,
            verified_tenant: Option<&str>,
        ) -> Option<SecurityContext> {
            (subject_id == "alice" && verified_tenant == Some("tenant-a"))
                .then(|| ctx(Level::Secret))
        }
    }

    fn check(pep: &CasPep, subject_id: &str, label: Option<SecurityLabel>) -> MacDecision {
        let resolver = CasObjectLabelResolver::new(label);
        pep.check_read(subject_id, Some("tenant-a"), &resolver)
    }

    #[derive(Default)]
    struct SpySink {
        records: Mutex<Vec<AuditRecord>>,
    }

    impl AuditSink for SpySink {
        fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
            self.records.lock().push(record.clone());
            Ok(())
        }
    }

    #[test]
    fn pep_permits_when_clearance_dominates_label() {
        let sink = Arc::new(SpySink::default());
        let pep = CasPep::new(
            Arc::new(FixtureClearance {
                subject: "secret-user".to_owned(),
                ctx: ctx(Level::Secret),
            }),
            sink.clone(),
        );
        let label = domain_label(&DedupDomain::local_default()); // Internal
        assert_eq!(check(&pep, "secret-user", Some(label)), MacDecision::Permit);

        let records = sink.records.lock();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].decision, Decision::Permit);
    }

    #[test]
    fn pep_threads_verified_tenant_to_clearance_source() {
        let pep = CasPep::new(Arc::new(TenantClearance), Arc::new(SpySink::default()));
        let resolver = CasObjectLabelResolver::from_domain(&DedupDomain::local_default());
        assert_eq!(
            pep.check_read("alice", Some("tenant-a"), &resolver),
            MacDecision::Permit
        );
        assert_eq!(
            pep.check_read("alice", None, &resolver),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );
    }

    #[test]
    fn pep_denies_when_clearance_insufficient() {
        let sink = Arc::new(SpySink::default());
        let pep = CasPep::new(
            Arc::new(FixtureClearance {
                subject: "public-user".to_owned(),
                ctx: ctx(Level::Public),
            }),
            sink.clone(),
        );
        let label = domain_label(&DedupDomain::local_default()); // Internal
        assert_eq!(
            check(&pep, "public-user", Some(label)),
            MacDecision::Deny(MacDenyReason::FloorDeny)
        );

        let records = sink.records.lock();
        assert_eq!(records[0].decision, Decision::Deny);
        assert_eq!(records[0].reason, DecisionReason::FloorDeny);
        assert_eq!(records[0].subject_id.as_deref(), Some("public-user"));
        assert_eq!(records[0].object_id.as_deref(), Some("tenant-a/cas"));
    }

    #[test]
    fn pep_denies_unresolvable_subject() {
        let sink = Arc::new(SpySink::default());
        let pep = CasPep::new(
            Arc::new(FixtureClearance {
                subject: "known".to_owned(),
                ctx: ctx(Level::Secret),
            }),
            sink.clone(),
        );
        let label = domain_label(&DedupDomain::local_default());
        assert_eq!(
            check(&pep, "unknown", Some(label)),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );

        let records = sink.records.lock();
        assert_eq!(records[0].reason, DecisionReason::NoClearance);
    }

    #[test]
    fn pep_denies_unlabeled_object_even_with_clearance() {
        let sink = Arc::new(SpySink::default());
        let pep = CasPep::new(
            Arc::new(FixtureClearance {
                subject: "admin".to_owned(),
                ctx: ctx(Level::Secret),
            }),
            sink.clone(),
        );
        assert_eq!(
            check(&pep, "admin", None),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );

        let records = sink.records.lock();
        assert_eq!(records[0].reason, DecisionReason::UnlabeledObject);
    }

    // ── MacCasAuthorizer ────────────────────────────────────────────────────

    #[test]
    fn mac_cas_authorizer_denies_with_fail_closed_pep() {
        let authz = MacCasAuthorizer::fail_closed();
        let domain = DedupDomain::local_default();
        let req = CasMountAuthzRequest {
            kind: CasMountObjectKind::Xorb,
            address: "deadbeef",
            domain: &domain,
            verified_tenant: Some("tenant-a"),
            operation: "read",
            requested_label: None,
        };
        let subject = Subject::new("user");
        assert!(authz.authorize(&subject, req).is_err());
    }

    #[test]
    fn mac_cas_authorizer_permits_with_dominating_clearance() {
        let sink = Arc::new(SpySink::default());
        let pep = Arc::new(CasPep::new(
            Arc::new(FixtureClearance {
                subject: "secret-user".to_owned(),
                ctx: ctx(Level::Secret),
            }),
            sink,
        ));
        let authz = MacCasAuthorizer::new(pep);
        let domain = DedupDomain::local_default();
        let req = CasMountAuthzRequest {
            kind: CasMountObjectKind::Xorb,
            address: "deadbeef",
            domain: &domain,
            verified_tenant: Some("tenant-a"),
            operation: "read",
            requested_label: None,
        };
        let subject = Subject::new("secret-user");
        assert!(authz.authorize(&subject, req).is_ok());
    }

    #[test]
    fn mac_cas_authorizer_uses_verified_http_tenant_for_clearance() {
        let pep = Arc::new(CasPep::new(
            Arc::new(TenantClearance),
            Arc::new(SpySink::default()),
        ));
        let authz = MacCasAuthorizer::new(pep);
        let domain = DedupDomain::local_default();
        let subject = Subject::new("alice");

        let tenant_a = CasMountAuthzRequest {
            kind: CasMountObjectKind::Xorb,
            address: "deadbeef",
            domain: &domain,
            verified_tenant: Some("tenant-a"),
            operation: "read",
            requested_label: None,
        };
        assert!(authz.authorize(&subject, tenant_a).is_ok());

        let tenant_b = CasMountAuthzRequest {
            kind: CasMountObjectKind::Xorb,
            address: "deadbeef",
            domain: &domain,
            verified_tenant: Some("tenant-b"),
            operation: "read",
            requested_label: None,
        };
        assert!(authz.authorize(&subject, tenant_b).is_err());
    }

    #[test]
    fn mac_cas_authorizer_uses_domain_for_label() {
        // SharedRemote domain → Confidential. A Secret clearance dominates; a
        // Public clearance does not.
        let sink = Arc::new(SpySink::default());
        let pep = Arc::new(CasPep::new(
            Arc::new(FixtureClearance {
                subject: "public-user".to_owned(),
                ctx: ctx(Level::Public),
            }),
            sink,
        ));
        let authz = MacCasAuthorizer::new(pep);
        let domain = DedupDomain {
            trust_boundary: TrustBoundary::SharedRemote,
            ..DedupDomain::local_default()
        };
        let req = CasMountAuthzRequest {
            kind: CasMountObjectKind::Object,
            address: "some-addr",
            domain: &domain,
            verified_tenant: Some("tenant-a"),
            operation: "open",
            requested_label: None,
        };
        let subject = Subject::new("public-user");
        assert!(
            authz.authorize(&subject, req).is_err(),
            "Public clearance must not read Confidential (shared-remote) content"
        );
    }
}
