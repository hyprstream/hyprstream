//! Coverage-gated MAC activation control and verified-subject context cache.
//!
//! The reference monitor is always present.  The operator control only selects
//! which subject context it receives:
//! - [`MacActivationMode::FloorOnly`] uses the anonymous floor;
//! - [`MacActivationMode::IdentityAware`] uses a context derived from verified
//!   `Claims × VerifiedKeyMaterial`.
//!
//! Widening is refused unless the supplied genesis report is complete.  No
//! startup path calls [`MacActivationControl::widen_identity_aware`]
//! automatically; narrowing is always available.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use parking_lot::RwLock;

use super::{
    Assurance, CompartmentSet, GenesisReport, Level, SecurityContext, SecurityLabel,
    VerifiedKeyMaterial,
};
use crate::envelope::Subject;
use crate::service::EnvelopeContext;

const FLOOR_ONLY: u8 = 0;
const IDENTITY_AWARE: u8 = 1;

/// The two allowed production enforcement states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacActivationMode {
    /// Mandatory monitor with the anonymous-floor subject context.
    FloorOnly,
    /// Mandatory monitor with verified identity-aware subject contexts.
    IdentityAware,
}

/// Why an operator-requested widening was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacActivationError {
    pub unlabeled: usize,
    pub ill_formed: usize,
    pub missing_gates: Vec<&'static str>,
    pub blocked_transports: Vec<&'static str>,
}

impl std::fmt::Display for MacActivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MAC identity-aware widening refused: readiness evidence incomplete \
             (unlabeled={}, ill_formed={}, missing_gates={:?}, blocked_transports={:?})",
            self.unlabeled, self.ill_formed, self.missing_gates, self.blocked_transports
        )
    }
}

impl std::error::Error for MacActivationError {}

/// Operator-supplied evidence for the epic's G1-G7 decision gate.
///
/// These are attestations, not automatically inferred health signals. The
/// operator control plane must assemble them from the signed activation
/// evidence described in the runbook. G3 is proven by this control's
/// reversible narrow operation itself.
pub struct MacActivationEvidence<'a> {
    pub genesis: &'a GenesisReport,
    pub mediation_integrity_g2: bool,
    pub denial_handling_g4: bool,
    pub observability_g5: bool,
    pub runbook_signoff_g6: bool,
    pub revocation_reload_g7: bool,
}

impl MacActivationEvidence<'_> {
    fn missing_gates(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if !self.genesis.is_complete() {
            missing.push("G1");
        }
        if !self.mediation_integrity_g2 {
            missing.push("G2");
        }
        if !self.denial_handling_g4 {
            missing.push("G4");
        }
        if !self.observability_g5 {
            missing.push("G5");
        }
        if !self.runbook_signoff_g6 {
            missing.push("G6");
        }
        if !self.revocation_reload_g7 {
            missing.push("G7");
        }
        missing
    }
}

/// Process-wide widen/narrow control.  It never removes a PEP.
#[derive(Debug)]
pub struct MacActivationControl {
    mode: AtomicU8,
    unverified_attach_transports: RwLock<BTreeSet<&'static str>>,
}

impl Default for MacActivationControl {
    fn default() -> Self {
        Self {
            mode: AtomicU8::new(FLOOR_ONLY),
            unverified_attach_transports: RwLock::new(BTreeSet::new()),
        }
    }
}

impl MacActivationControl {
    /// Current subject-context selection.
    #[must_use]
    pub fn mode(&self) -> MacActivationMode {
        if self.mode.load(Ordering::Acquire) == IDENTITY_AWARE {
            MacActivationMode::IdentityAware
        } else {
            MacActivationMode::FloorOnly
        }
    }

    /// Explicit operator widening.  Coverage must be complete at the instant of
    /// the request; merely constructing or logging a report never flips state.
    pub fn widen_identity_aware(
        &self,
        evidence: &MacActivationEvidence<'_>,
    ) -> Result<(), MacActivationError> {
        let blocked_transports = self.blocked_transports();
        let mut missing_gates = evidence.missing_gates();
        if !blocked_transports.is_empty() && !missing_gates.contains(&"G2") {
            missing_gates.push("G2");
        }
        if !missing_gates.is_empty() {
            return Err(MacActivationError {
                unlabeled: evidence.genesis.unlabeled.len(),
                ill_formed: evidence.genesis.ill_formed.len(),
                missing_gates,
                blocked_transports,
            });
        }
        // A widening starts a new cache generation. No verified subject
        // context from an earlier floor-only/reload epoch can survive it.
        flush_verified_subject_cache_generation();
        self.mode.store(IDENTITY_AWARE, Ordering::Release);
        Ok(())
    }

    /// Kill-switch: narrow subject context back to the anonymous floor while
    /// leaving every monitor installed and authoritative.
    pub fn narrow_to_floor(&self) {
        self.mode.store(FLOOR_ONLY, Ordering::Release);
        flush_verified_subject_cache_generation();
    }

    /// Select the context a PEP must evaluate in the current mode.
    #[must_use]
    pub fn select_context(&self, verified: Option<SecurityContext>) -> Option<SecurityContext> {
        match self.mode() {
            MacActivationMode::FloorOnly => Some(anonymous_floor()),
            MacActivationMode::IdentityAware => verified,
        }
    }

    /// Permanently block identity-aware widening in this process while a live
    /// 9P transport still has no verified attach-credential carrier.
    ///
    /// Registration also narrows an already-widened process immediately, so a
    /// late worker constructor cannot turn UDS/vsock attaches into a deny-all
    /// availability outage.
    pub fn block_unverified_attach_transport(&self, transport: &'static str) {
        self.unverified_attach_transports.write().insert(transport);
        self.narrow_to_floor();
    }

    /// Runtime transport blockers that make G2 structurally incomplete.
    #[must_use]
    pub fn blocked_transports(&self) -> Vec<&'static str> {
        self.unverified_attach_transports
            .read()
            .iter()
            .copied()
            .collect()
    }
}

/// The process-global activation control.  It starts floor-only.
#[must_use]
pub fn global_mac_activation_control() -> &'static MacActivationControl {
    static CONTROL: OnceLock<MacActivationControl> = OnceLock::new();
    CONTROL.get_or_init(MacActivationControl::default)
}

/// Register a live production 9P transport that still uses a deny-only
/// authenticator. Once registered, this process cannot widen identity-aware
/// MAC until that constructor is replaced with a verified credential carrier
/// in a fresh process.
pub fn block_identity_widening_for_unverified_attach_transport(transport: &'static str) {
    global_mac_activation_control().block_unverified_attach_transport(transport);
}

/// Canonical anonymous-floor context used by every PEP during narrowing.
#[must_use]
pub fn anonymous_floor() -> SecurityContext {
    SecurityContext::from_clearance(
        SecurityLabel::new(Level::Public, Assurance::Unverified, CompartmentSet::EMPTY),
        VerifiedKeyMaterial::Unverified,
    )
}

#[derive(Clone)]
struct VerifiedSubjectEntry {
    context: SecurityContext,
    tenant: Option<String>,
    expires_at: i64,
    /// Issuer-scoped credential ID `(iss, jti/cti)` this entry was derived
    /// from. Typed so eviction targets the exact credential without
    /// cross-issuer or JWT/CWT ambiguity.
    credential_id: Option<crate::auth::CredentialId>,
    generation: u64,
}

#[derive(Default)]
struct VerifiedSubjectCache {
    generation: u64,
    subjects: HashMap<String, VerifiedSubjectEntry>,
}

fn verified_subjects() -> &'static RwLock<VerifiedSubjectCache> {
    static SUBJECTS: OnceLock<RwLock<VerifiedSubjectCache>> = OnceLock::new();
    SUBJECTS.get_or_init(|| RwLock::new(VerifiedSubjectCache::default()))
}

/// Revoke every cached subject context derived from the given credential.
///
/// The shared credential-revocation store calls this hook on every
/// revocation, so VFS/CAS/MoQ lookups cannot continue using authority
/// cached before the blocklist update. The typed `CredentialId` ensures
/// only entries derived from the exact `(iss, jti/cti)` pair are evicted
/// — no cross-issuer or JWT/CWT ambiguity.
pub fn revoke_verified_subject_credential(id: &crate::auth::CredentialId) -> usize {
    let mut cache = verified_subjects().write();
    let before = cache.subjects.len();
    cache
        .subjects
        .retain(|_, entry| entry.credential_id.as_ref() != Some(id));
    before.saturating_sub(cache.subjects.len())
}

/// Advance the verified-subject cache generation and remove all prior entries.
///
/// Policy/resolver reload and revocation control planes call this G7 hook.
/// Widening and narrowing also rotate the generation automatically.
pub fn flush_verified_subject_cache_generation() -> u64 {
    let mut cache = verified_subjects().write();
    cache.generation = cache.generation.saturating_add(1);
    cache.subjects.clear();
    cache.generation
}

/// Cache the context of a request whose envelope and Claims have already been
/// verified.  This is the bridge for in-process VFS/CAS/MoQ APIs that carry a
/// verified [`Subject`] but not the full [`EnvelopeContext`].
///
/// The cache is never an authority source: insertion requires the same
/// `Claims × VerifiedKeyMaterial` derivation used by the RPC PEP, entries expire
/// with the signed Claims, and lookup still passes through the activation
/// control.
pub fn remember_verified_subject(ctx: &EnvelopeContext) {
    let Some(claims) = ctx.claims() else {
        return;
    };
    let subject = ctx.subject();
    remember_verified_claims(
        &subject,
        claims,
        ctx.verified_key_material(),
        ctx.verified_tenant(),
    );
}

/// Cache an already-verified Claims binding for a lower-level PEP boundary
/// that does not carry an [`EnvelopeContext`] (notably unified 9P attach).
///
/// The caller must invoke this only after signature, expiry, local-issuer,
/// tenant, and sender-binding verification. The two-input derivation is
/// repeated here so a Claims value alone can never create a subject context.
pub fn remember_verified_claims(
    subject: &Subject,
    claims: &crate::auth::Claims,
    key_material: VerifiedKeyMaterial,
    verified_tenant: Option<&str>,
) {
    use super::SubjectContextClaims as _;

    let Some(name) = subject.name() else {
        return;
    };
    if claims.exp <= chrono::Utc::now().timestamp() {
        return;
    }
    if claims.sub != name {
        return;
    }
    let Some(context) = claims.security_context(key_material) else {
        return;
    };
    // Derive the JWT credential ID from `(iss, jti)`. CWT credential paths
    // use [`remember_verified_claims_with_credential`] with an explicit CWT
    // CredentialId so `cti` bytes are never stringified.
    let credential_id = claims
        .jti
        .as_ref()
        .map(|jti| crate::auth::CredentialId::jwt(&claims.iss, jti));
    insert_verified_subject_entry(
        name,
        context,
        verified_tenant,
        claims.exp,
        credential_id,
    );
}

/// Cache a verified-Claims binding with an explicit [`CredentialId`],
/// for credential encodings whose identifier is not derivable from
/// `Claims::jti` (notably CWT `cti` byte strings).
///
/// This is the **only** insertion path for CWT credentials. It repeats
/// every invariant check that [`remember_verified_claims`] enforces:
/// `sub` must match `subject.name()`, `exp` must be in the future,
/// `security_context` must be derivable from `Claims × VerifiedKeyMaterial`,
/// and the credential ID's issuer must match `claims.iss`.
/// The cache entry is keyed by the caller-supplied `credential_id`, so
/// eviction targets the exact `(iss, jti/cti)` pair.
///
/// `credential_id.issuer` MUST equal `claims.iss`. This prevents a caller
/// from attaching a credential ID from a different issuer to a cache
/// entry, which would break issuer-scoped eviction.
pub fn remember_verified_claims_with_credential(
    subject: &Subject,
    claims: &crate::auth::Claims,
    key_material: VerifiedKeyMaterial,
    verified_tenant: Option<&str>,
    credential_id: crate::auth::CredentialId,
) {
    use super::SubjectContextClaims as _;

    let Some(name) = subject.name() else {
        return;
    };
    if claims.exp <= chrono::Utc::now().timestamp() {
        return;
    }
    if claims.sub != name {
        return;
    }
    let Some(context) = claims.security_context(key_material) else {
        return;
    };
    // The credential ID's issuer must match the claims' issuer, so
    // issuer-scoped eviction cannot be subverted by attaching a
    // foreign-issuer credential ID.
    if !credential_id.is_valid() || credential_id.issuer != claims.iss {
        return;
    }
    insert_verified_subject_entry(
        name,
        context,
        verified_tenant,
        claims.exp,
        Some(credential_id),
    );
}

fn insert_verified_subject_entry(
    name: &str,
    context: SecurityContext,
    verified_tenant: Option<&str>,
    expires_at: i64,
    credential_id: Option<crate::auth::CredentialId>,
) {
    let mut cache = verified_subjects().write();
    let generation = cache.generation;
    cache.subjects.insert(
        name.to_owned(),
        VerifiedSubjectEntry {
            context,
            tenant: verified_tenant.map(str::to_owned),
            expires_at,
            credential_id,
            generation,
        },
    );
}

/// Resolve a VFS/CAS/MoQ subject through the verified-Claims cache and current
/// activation mode.  Tenant mismatch is a hard miss.
///
/// Revocation revalidation: a cache hit whose entry carries a credential ID
/// is revalidated against the process-global credential-revocation store on
/// EVERY read — in non-policy processes that is one authority RPC per hit,
/// which is deliberate strict observation, mirroring the per-request check in
/// `verify_claims`. A revoked credential, an absent store, or an unreachable
/// authority (the store's own fail-closed `true`) all evict the entry and
/// deny. Entries without a credential ID keep the cache-only behavior.
pub async fn subject_context(
    subject: &Subject,
    verified_tenant: Option<&str>,
) -> Option<SecurityContext> {
    subject_context_with(
        crate::auth::global_credential_revocation_store().map(std::convert::AsRef::as_ref),
        subject,
        verified_tenant,
    )
    .await
}

/// [`subject_context`] against an explicit revocation store instead of the
/// process-global handle. `None` behaves exactly like an unpublished global
/// store: credential-bearing hits fail closed.
pub async fn subject_context_with(
    store: Option<&dyn crate::auth::CredentialRevocationStore>,
    subject: &Subject,
    verified_tenant: Option<&str>,
) -> Option<SecurityContext> {
    let verified = match subject.name() {
        Some(name) => {
            // Sync screening pass (unchanged rules): generation, expiry,
            // tenant. A miss here never reaches the store.
            let entry = {
                let now = chrono::Utc::now().timestamp();
                let mut cache = verified_subjects().write();
                match cache.subjects.get(name) {
                    Some(entry) => {
                        let entry = entry.clone();
                        if entry.generation != cache.generation || entry.expires_at <= now {
                            cache.subjects.remove(name);
                            None
                        } else if let Some(expected) = verified_tenant {
                            if entry.tenant.as_deref() != Some(expected) {
                                None
                            } else {
                                Some(entry)
                            }
                        } else {
                            Some(entry)
                        }
                    }
                    None => None,
                }
            };
            match entry {
                Some(entry) => {
                    if let Some(ref credential_id) = entry.credential_id {
                        let revoked = match store {
                            Some(store) => store.is_revoked(credential_id).await,
                            // No authority handle published — fail closed.
                            None => true,
                        };
                        if revoked {
                            revoke_verified_subject_credential(credential_id);
                            None
                        } else {
                            Some(entry.context)
                        }
                    } else {
                        Some(entry.context)
                    }
                }
                None => None,
            }
        }
        None => None,
    };
    global_mac_activation_control().select_context(verified)
}

/// Test lock for tests that touch the global verified-subjects cache.
/// All tests across modules that insert/evict/flush cache entries MUST
/// acquire this lock to prevent parallel test interference.
#[cfg(test)]
pub(crate) static CACHE_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn report(complete: bool) -> GenesisReport {
        GenesisReport {
            labeled: vec!["/".to_owned()],
            unlabeled: if complete {
                Vec::new()
            } else {
                vec!["/gap".to_owned()]
            },
            ill_formed: Vec::new(),
        }
    }

    #[test]
    fn widening_requires_complete_coverage_and_narrowing_is_always_available() {
        let _guard = CACHE_TEST_LOCK.lock();
        let control = MacActivationControl::default();
        let incomplete = report(false);
        let mut evidence = MacActivationEvidence {
            genesis: &incomplete,
            mediation_integrity_g2: true,
            denial_handling_g4: true,
            observability_g5: true,
            runbook_signoff_g6: true,
            revocation_reload_g7: true,
        };
        assert!(control.widen_identity_aware(&evidence).is_err());
        assert_eq!(control.mode(), MacActivationMode::FloorOnly);
        let complete = report(true);
        evidence.genesis = &complete;
        assert!(control.widen_identity_aware(&evidence).is_ok());
        assert_eq!(control.mode(), MacActivationMode::IdentityAware);
        control.narrow_to_floor();
        assert_eq!(control.mode(), MacActivationMode::FloorOnly);
    }

    #[test]
    fn unverified_attach_transport_structurally_blocks_g2_widening() {
        let _guard = CACHE_TEST_LOCK.lock();
        let control = MacActivationControl::default();
        control.block_unverified_attach_transport("worker-uds-vsock");

        let complete = report(true);
        let evidence = MacActivationEvidence {
            genesis: &complete,
            mediation_integrity_g2: true,
            denial_handling_g4: true,
            observability_g5: true,
            runbook_signoff_g6: true,
            revocation_reload_g7: true,
        };
        let Err(error) = control.widen_identity_aware(&evidence) else {
            panic!("unverified worker transport must block identity-aware widening");
        };
        assert_eq!(control.mode(), MacActivationMode::FloorOnly);
        assert!(error.missing_gates.contains(&"G2"));
        assert_eq!(error.blocked_transports, vec!["worker-uds-vsock"]);
    }

    #[test]
    fn jti_revocation_and_generation_rotation_evict_cached_subjects() {
        let _guard = CACHE_TEST_LOCK.lock();
        flush_verified_subject_cache_generation();

        let now = chrono::Utc::now().timestamp();
        let mut claims =
            crate::auth::Claims::new("did:web:alice".to_owned(), now, now + 300).with_clearance(
                SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY),
            );
        claims.iss = "https://local.example".to_owned();
        claims.jti = Some("revocable-jti".to_owned());
        let subject = Subject::new("did:web:alice");
        remember_verified_claims(
            &subject,
            &claims,
            VerifiedKeyMaterial::Classical,
            Some("tenant-a"),
        );

        assert_eq!(verified_subjects().read().subjects.len(), 1);
        let other_id = crate::auth::CredentialId::jwt("https://local.example", "other-jti");
        let revocable_id = crate::auth::CredentialId::jwt("https://local.example", "revocable-jti");
        let cross_issuer_id = crate::auth::CredentialId::jwt("https://other.example", "revocable-jti");
        assert_eq!(revoke_verified_subject_credential(&other_id), 0);
        assert_eq!(revoke_verified_subject_credential(&cross_issuer_id), 0,
            "same jti from a different issuer must NOT evict");
        assert_eq!(revoke_verified_subject_credential(&revocable_id), 1);
        assert!(verified_subjects().read().subjects.is_empty());

        remember_verified_claims(
            &subject,
            &claims,
            VerifiedKeyMaterial::Classical,
            Some("tenant-a"),
        );
        let old_generation = verified_subjects().read().generation;
        let new_generation = flush_verified_subject_cache_generation();
        assert!(new_generation >= old_generation);
        assert!(verified_subjects().read().subjects.is_empty());
    }
}
