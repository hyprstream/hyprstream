//! Per-entry enrollment resolution for authenticated proofs.
//!
//! An authenticated proof's signer keys are never self-certified by a header.
//! Each logical signer group resolves to exactly one **signer-suite record**
//! that pins, by construction:
//!
//! - the exact suite ID the group must declare,
//! - the exact ordered public component keys (classical and post-quantum),
//! - the principal,
//! - the enrollment epoch, and
//! - the role the record may take in a proof.
//!
//! A component signature verifying under any key not pinned in that record
//! denies — even a key validly enrolled to the same principal (design §4.4).
//! The credential `cnf` thumbprint selects the primary record; additional
//! authorization groups resolve through anchored approver enrollments; a
//! response proof's signer resolves from the service enrollment for the
//! request's signed service domain (§9.4).
//!
//! This module owns the *seam*, not the manifest: the production resolver is
//! the enrollment-manifest consumer (workstream B). Nothing resolves by
//! default — an absent resolver denies, it never falls back to a permissive
//! or self-asserted key source.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use super::{ALG_ED25519, ALG_ML_DSA_65};

/// Domain separator for the authenticated-proof replay namespace thumbprint.
const AUTHENTICATED_THUMBPRINT_DOMAIN: &[u8] = b"hs-proof-authenticated-replay-v1";

// ---------------------------------------------------------------------------
// Pinned component keys
// ---------------------------------------------------------------------------

/// A public component key pinned by an enrollment record.
#[derive(Debug, Clone)]
pub enum ComponentKey {
    Ed25519(ed25519_dalek::VerifyingKey),
    MlDsa65(Box<crate::crypto::pq::MlDsaVerifyingKey>),
}

impl ComponentKey {
    /// The fully-specified COSE algorithm this key material signs with.
    pub fn alg(&self) -> i64 {
        match self {
            Self::Ed25519(_) => ALG_ED25519,
            Self::MlDsa65(_) => ALG_ML_DSA_65,
        }
    }

    /// The canonical public-key encoding, used for pin comparison and for the
    /// replay-namespace thumbprint.
    pub fn encoded(&self) -> Vec<u8> {
        match self {
            Self::Ed25519(k) => k.to_bytes().to_vec(),
            Self::MlDsa65(k) => crate::crypto::pq::ml_dsa_vk_bytes(k),
        }
    }

    /// Verify `sig` over `data` under this exact pinned key.
    pub fn verify(&self, sig: &[u8], data: &[u8]) -> Result<()> {
        match self {
            Self::Ed25519(k) => {
                use ed25519_dalek::Verifier;
                let bytes: [u8; 64] = sig
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("Ed25519 signature must be 64 bytes"))?;
                k.verify(data, &ed25519_dalek::Signature::from_bytes(&bytes))
                    .map_err(|_| anyhow::anyhow!("Ed25519 verification failed"))
            }
            Self::MlDsa65(k) => crate::crypto::pq::ml_dsa_verify(k, data, sig),
        }
    }
}

/// One pinned component of an enrolled signer suite: exact algorithm, exact
/// key ID, exact public key.
#[derive(Debug, Clone)]
pub struct EnrolledComponent {
    pub alg: i64,
    pub kid: Vec<u8>,
    pub key: ComponentKey,
}

impl EnrolledComponent {
    pub fn new(kid: impl Into<Vec<u8>>, key: ComponentKey) -> Self {
        Self {
            alg: key.alg(),
            kid: kid.into(),
            key,
        }
    }
}

/// The role an enrolled record may take. A record is usable in exactly the
/// role it was enrolled for: an approver enrollment can never stand in as the
/// credential-bound primary signer, and neither can stand in as the enrolled
/// service response signer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignerRole {
    /// The credential `cnf`-bound primary request signer.
    Primary,
    /// An anchored approver in an additional authorization group.
    Approver,
    /// The enrolled response signer for one service domain.
    Service,
}

/// One enrolled signer-suite record — the exact chain a `cnf` thumbprint,
/// approver anchor, or service domain resolves to.
#[derive(Debug, Clone)]
pub struct SignerSuiteRecord {
    /// The enrolled principal. Distinct logical signer groups in one proof
    /// MUST resolve to distinct principals.
    pub principal: String,
    /// The exact suite ID the signed plan must declare for this group.
    pub suite_id: String,
    /// The pinned component keys, in the suite-declared order.
    pub components: Vec<EnrolledComponent>,
    /// Enrollment epoch — part of the authenticated replay namespace, so a
    /// re-enrollment cannot reuse a retired namespace.
    pub epoch: u64,
    /// The role this record is enrolled for.
    pub role: SignerRole,
    /// The named approver role this enrollment holds, for role-specific
    /// threshold rules. Meaningful only for [`SignerRole::Approver`]; the
    /// name is enrollment data, never taken from the request.
    pub approver_role: Option<String>,
    /// The enrollment policy this record was issued under (§4.4).
    ///
    /// Component-key separation is normative: a key enrolled for the WNS
    /// hybrid suite MUST NOT simultaneously be enrolled for a standalone
    /// suite, a different hybrid, another protocol or domain separator, or a
    /// different logical signer group. An exception requires a separate
    /// operator-approved cryptographic analysis and an explicit policy
    /// identifier — this field carries it, so an overlap can never be
    /// silently inherited from a key's other role.
    pub enrollment_policy_id: String,
    /// Credential/session expiry. A proof whose `exp` exceeds this denies.
    pub not_after: u64,
    /// Whether the enrollment has been revoked.
    pub revoked: bool,
}

impl SignerSuiteRecord {
    /// Check the record is usable right now for a proof expiring at
    /// `proof_exp`. Revocation, enrollment expiry, and a proof outliving its
    /// credential are all `Rejected`, never downgraded.
    pub fn check_usable(&self, now: u64, proof_exp: u64, role: SignerRole) -> Result<()> {
        if self.revoked {
            bail!(
                "enrollment for principal '{}' is revoked",
                self.principal
            );
        }
        if self.role != role {
            bail!(
                "enrollment for principal '{}' is enrolled as {:?}, presented as {:?}",
                self.principal,
                self.role,
                role
            );
        }
        if now >= self.not_after {
            bail!(
                "enrollment for principal '{}' expired at {}",
                self.principal,
                self.not_after
            );
        }
        if proof_exp > self.not_after {
            bail!(
                "proof exp {} exceeds credential/session expiry {} for principal '{}'",
                proof_exp,
                self.not_after,
                self.principal
            );
        }
        Ok(())
    }

    /// Whether this record pins the given Ed25519 key as one of its exact
    /// component keys.
    pub fn pins_ed25519(&self, key: &ed25519_dalek::VerifyingKey) -> bool {
        self.components.iter().any(|c| match &c.key {
            ComponentKey::Ed25519(k) => k.to_bytes() == key.to_bytes(),
            ComponentKey::MlDsa65(_) => false,
        })
    }

    /// Resolve the pinned key for an exact `(alg, kid)` pair.
    pub fn component(&self, alg: i64, kid: &[u8]) -> Option<&EnrolledComponent> {
        self.components
            .iter()
            .find(|c| c.alg == alg && c.kid == kid)
    }

    /// The authenticated-proof replay namespace thumbprint: SHA-256 over the
    /// canonical deterministic encoding of the exact suite ID, ordered public
    /// component keys, and enrollment epoch, under the authenticated-proof
    /// domain separator (§4.5).
    ///
    /// Approver groups are deliberately excluded, so every allowed approver
    /// subset for the same primary-signed request stays in one namespace.
    pub fn replay_thumbprint(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(AUTHENTICATED_THUMBPRINT_DOMAIN);
        hasher.update((self.suite_id.len() as u64).to_be_bytes());
        hasher.update(self.suite_id.as_bytes());
        hasher.update((self.components.len() as u64).to_be_bytes());
        for c in &self.components {
            let encoded = c.key.encoded();
            hasher.update(c.alg.to_be_bytes());
            hasher.update((encoded.len() as u64).to_be_bytes());
            hasher.update(&encoded);
        }
        hasher.update(self.epoch.to_be_bytes());
        hasher.finalize().into()
    }
}

// ---------------------------------------------------------------------------
// Resolver seam
// ---------------------------------------------------------------------------

/// Resolves the enrolled signer-suite records an authenticated proof's groups
/// must verify against.
///
/// Every method returns `None` for anything it does not have an exact,
/// unambiguous enrollment for; the caller denies. There is no permissive
/// default implementation and no self-certified fallback.
pub trait EnrollmentResolver: Send + Sync {
    /// Resolve the primary signer-suite record pinned by a credential `cnf`
    /// Ed25519 key.
    fn resolve_primary(&self, cnf: &ed25519_dalek::VerifyingKey) -> Option<SignerSuiteRecord>;

    /// Resolve an anchored approver record by the key ID of the group's first
    /// component.
    fn resolve_approver(&self, kid: &[u8]) -> Option<SignerSuiteRecord>;

    /// Resolve the enrolled response signer for one canonical service domain.
    fn resolve_service(&self, service_domain: &str) -> Option<SignerSuiteRecord>;
}

/// An in-memory resolver over an explicit enrollment table.
///
/// This is the shape a manifest consumer produces; it holds no defaults and
/// resolves only what was explicitly enrolled.
#[derive(Default)]
pub struct InMemoryEnrollmentResolver {
    /// Primary records keyed by the credential `cnf` Ed25519 public key.
    primary: HashMap<[u8; 32], SignerSuiteRecord>,
    /// Approver records keyed by their first component's key ID.
    approver: HashMap<Vec<u8>, SignerSuiteRecord>,
    /// Service response signers keyed by canonical service domain.
    service: HashMap<String, SignerSuiteRecord>,
}

impl InMemoryEnrollmentResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enrol a primary record, anchored by the credential `cnf` key.
    ///
    /// The `cnf` key MUST be one of the record's pinned components: the
    /// credential binds the proof by pinning the exact component keys, not by
    /// naming a principal.
    pub fn enrol_primary(
        &mut self,
        cnf: &ed25519_dalek::VerifyingKey,
        record: SignerSuiteRecord,
    ) -> Result<()> {
        if record.role != SignerRole::Primary {
            bail!("enrol_primary: record role is {:?}", record.role);
        }
        if !record.pins_ed25519(cnf) {
            bail!("enrol_primary: cnf key is not a pinned component of the record");
        }
        self.primary.insert(cnf.to_bytes(), record);
        Ok(())
    }

    /// Enrol an anchored approver record, keyed by its first component's kid.
    pub fn enrol_approver(&mut self, record: SignerSuiteRecord) -> Result<()> {
        if record.role != SignerRole::Approver {
            bail!("enrol_approver: record role is {:?}", record.role);
        }
        let kid = record
            .components
            .first()
            .ok_or_else(|| anyhow::anyhow!("enrol_approver: record has no components"))?
            .kid
            .clone();
        self.approver.insert(kid, record);
        Ok(())
    }

    /// Enrol the response signer for one canonical service domain.
    pub fn enrol_service(&mut self, service_domain: &str, record: SignerSuiteRecord) -> Result<()> {
        if record.role != SignerRole::Service {
            bail!("enrol_service: record role is {:?}", record.role);
        }
        self.service.insert(service_domain.to_owned(), record);
        Ok(())
    }
}

impl EnrollmentResolver for InMemoryEnrollmentResolver {
    fn resolve_primary(&self, cnf: &ed25519_dalek::VerifyingKey) -> Option<SignerSuiteRecord> {
        self.primary.get(&cnf.to_bytes()).cloned()
    }

    fn resolve_approver(&self, kid: &[u8]) -> Option<SignerSuiteRecord> {
        self.approver.get(kid).cloned()
    }

    fn resolve_service(&self, service_domain: &str) -> Option<SignerSuiteRecord> {
        self.service.get(service_domain).cloned()
    }
}

// ---------------------------------------------------------------------------
// Process-global registration
// ---------------------------------------------------------------------------

static ENROLLMENT_RESOLVER: OnceLock<Box<dyn EnrollmentResolver>> = OnceLock::new();

/// Install the process enrollment resolver. There is no auto-install: an
/// authenticated proof presented with no resolver denies.
pub fn set_global_enrollment_resolver(
    resolver: Box<dyn EnrollmentResolver>,
) -> std::result::Result<(), Box<dyn EnrollmentResolver>> {
    ENROLLMENT_RESOLVER.set(resolver)
}

pub fn global_enrollment_resolver() -> Option<&'static dyn EnrollmentResolver> {
    ENROLLMENT_RESOLVER.get().map(|r| &**r)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn ed(seed: u8) -> ed25519_dalek::VerifyingKey {
        ed25519_dalek::SigningKey::from_bytes(&[seed; 32]).verifying_key()
    }

    fn record(role: SignerRole, key: ed25519_dalek::VerifyingKey) -> SignerSuiteRecord {
        SignerSuiteRecord {
            principal: "p".into(),
            suite_id: super::super::SUITE_CLASSICAL.into(),
            components: vec![EnrolledComponent::new(
                b"kid-1".to_vec(),
                ComponentKey::Ed25519(key),
            )],
            epoch: 1,
            role,
            approver_role: None,
            enrollment_policy_id: "test-enrollment-v1".to_owned(),
            not_after: 2_000,
            revoked: false,
        }
    }

    #[test]
    fn revoked_enrollment_is_rejected() {
        let mut r = record(SignerRole::Primary, ed(1));
        r.revoked = true;
        assert!(r.check_usable(1_000, 1_010, SignerRole::Primary).is_err());
    }

    #[test]
    fn expired_enrollment_is_rejected() {
        let r = record(SignerRole::Primary, ed(1));
        assert!(r.check_usable(2_000, 2_010, SignerRole::Primary).is_err());
    }

    #[test]
    fn proof_outliving_the_credential_is_rejected() {
        let r = record(SignerRole::Primary, ed(1));
        assert!(r.check_usable(1_000, 2_001, SignerRole::Primary).is_err());
        assert!(r.check_usable(1_000, 2_000, SignerRole::Primary).is_ok());
    }

    #[test]
    fn a_record_cannot_be_used_in_another_role() {
        let approver = record(SignerRole::Approver, ed(1));
        assert!(approver
            .check_usable(1_000, 1_010, SignerRole::Primary)
            .is_err());
        let service = record(SignerRole::Service, ed(1));
        assert!(service
            .check_usable(1_000, 1_010, SignerRole::Approver)
            .is_err());
    }

    #[test]
    fn primary_enrollment_requires_the_cnf_key_to_be_pinned() {
        let mut resolver = InMemoryEnrollmentResolver::new();
        let r = record(SignerRole::Primary, ed(1));
        assert!(resolver.enrol_primary(&ed(2), r.clone()).is_err());
        assert!(resolver.enrol_primary(&ed(1), r).is_ok());
    }

    #[test]
    fn a_role_mismatch_cannot_be_enrolled() {
        let mut resolver = InMemoryEnrollmentResolver::new();
        assert!(resolver
            .enrol_primary(&ed(1), record(SignerRole::Approver, ed(1)))
            .is_err());
        assert!(resolver
            .enrol_approver(record(SignerRole::Primary, ed(1)))
            .is_err());
        assert!(resolver
            .enrol_service("svc", record(SignerRole::Primary, ed(1)))
            .is_err());
    }

    /// The replay namespace must change when the enrollment epoch rotates,
    /// otherwise a re-enrolled signer inherits a retired namespace's history.
    #[test]
    fn replay_thumbprint_binds_suite_keys_and_epoch() {
        let base = record(SignerRole::Primary, ed(1));
        let mut rotated = base.clone();
        rotated.epoch = 2;
        let mut other_key = base.clone();
        other_key.components[0].key = ComponentKey::Ed25519(ed(9));
        let mut other_suite = base.clone();
        other_suite.suite_id = super::super::SUITE_HYBRID.into();

        assert_ne!(base.replay_thumbprint(), rotated.replay_thumbprint());
        assert_ne!(base.replay_thumbprint(), other_key.replay_thumbprint());
        assert_ne!(base.replay_thumbprint(), other_suite.replay_thumbprint());

        // The principal is not part of the namespace; the pinned keys are.
        let mut renamed = base.clone();
        renamed.principal = "someone-else".into();
        assert_eq!(base.replay_thumbprint(), renamed.replay_thumbprint());
    }

    /// Approver groups are excluded from the primary namespace by
    /// construction: the thumbprint is computed from the primary record only.
    #[test]
    fn approver_membership_does_not_change_the_primary_namespace() {
        let primary = record(SignerRole::Primary, ed(1));
        let before = primary.replay_thumbprint();
        let _approver = record(SignerRole::Approver, ed(2));
        assert_eq!(before, primary.replay_thumbprint());
    }
}
