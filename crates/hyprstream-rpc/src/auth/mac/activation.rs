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

use std::collections::HashMap;
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
}

impl std::fmt::Display for MacActivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MAC identity-aware widening refused: readiness evidence incomplete \
             (unlabeled={}, ill_formed={}, missing_gates={:?})",
            self.unlabeled, self.ill_formed, self.missing_gates
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
}

impl Default for MacActivationControl {
    fn default() -> Self {
        Self {
            mode: AtomicU8::new(FLOOR_ONLY),
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
        let missing_gates = evidence.missing_gates();
        if !missing_gates.is_empty() {
            return Err(MacActivationError {
                unlabeled: evidence.genesis.unlabeled.len(),
                ill_formed: evidence.genesis.ill_formed.len(),
                missing_gates,
            });
        }
        self.mode.store(IDENTITY_AWARE, Ordering::Release);
        Ok(())
    }

    /// Kill-switch: narrow subject context back to the anonymous floor while
    /// leaving every monitor installed and authoritative.
    pub fn narrow_to_floor(&self) {
        self.mode.store(FLOOR_ONLY, Ordering::Release);
    }

    /// Select the context a PEP must evaluate in the current mode.
    #[must_use]
    pub fn select_context(&self, verified: Option<SecurityContext>) -> Option<SecurityContext> {
        match self.mode() {
            MacActivationMode::FloorOnly => Some(anonymous_floor()),
            MacActivationMode::IdentityAware => verified,
        }
    }
}

/// The process-global activation control.  It starts floor-only.
#[must_use]
pub fn global_mac_activation_control() -> &'static MacActivationControl {
    static CONTROL: OnceLock<MacActivationControl> = OnceLock::new();
    CONTROL.get_or_init(MacActivationControl::default)
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
}

fn verified_subjects() -> &'static RwLock<HashMap<String, VerifiedSubjectEntry>> {
    static SUBJECTS: OnceLock<RwLock<HashMap<String, VerifiedSubjectEntry>>> = OnceLock::new();
    SUBJECTS.get_or_init(|| RwLock::new(HashMap::new()))
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
    verified_subjects().write().insert(
        name.to_owned(),
        VerifiedSubjectEntry {
            context,
            tenant: verified_tenant.map(str::to_owned),
            expires_at: claims.exp,
        },
    );
}

/// Resolve a VFS/CAS/MoQ subject through the verified-Claims cache and current
/// activation mode.  Tenant mismatch is a hard miss.
#[must_use]
pub fn subject_context(
    subject: &Subject,
    verified_tenant: Option<&str>,
) -> Option<SecurityContext> {
    let verified = subject.name().and_then(|name| {
        let now = chrono::Utc::now().timestamp();
        let mut subjects = verified_subjects().write();
        let entry = subjects.get(name)?.clone();
        if entry.expires_at <= now {
            subjects.remove(name);
            return None;
        }
        if let Some(expected) = verified_tenant {
            if entry.tenant.as_deref() != Some(expected) {
                return None;
            }
        }
        Some(entry.context)
    });
    global_mac_activation_control().select_context(verified)
}

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
}
