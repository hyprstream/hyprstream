//! H3b.1: disabled request-local proof verification, NOT dispatch admission.
//!
//! No production constructor for CredentialHandle or installed provider exists.
//! A future bridge must verify the original host JWT (including signed generation,
//! direct holder provenance and jti), bind the handle to its connection, and
//! authenticate the bounded Policy RPC. Neither a subject nor a relay credential
//! can construct a handle here. Fixtures deliberately bypass that unfinished bridge.
//!
//! Success establishes signatures and byte commitments only. Generated method
//! policy, MAC, durable replay, account/grant authority and streaming currentness
//! still precede any handler effect. This module is not called by dispatch or
//! cached activation and cannot mint an admission permit.

use std::{sync::Arc, time::Duration};

use ed25519_dalek::VerifyingKey;
use hyprstream_rpc::{
    auth::signer_suite::signer_suite_thumbprint,
    crypto::pq::ml_dsa_vk_from_bytes,
    proof::{
        enrollment::{
            ComponentKey, EnrolledComponent, EnrollmentResolver, SignerRole, SignerSuiteRecord,
        },
        parser::ParsedProof,
        verify::{verify_proof_signatures, VerifiedProof},
        ProofDisposition, ProofKind,
    },
};
use hyprstream_rpc_std::policy_client::SessionPrimary;
use hyprstream_session_store::{primary::ExpectedPrimary, CollisionInventory, PROFILE, SUITE};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use super::federate_source::{session_primary_kids, CLIENT, HOST};

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum Error {
    #[error("dynamic proof consumer disabled")]
    Disabled,
    #[error("dynamic proof denied")]
    Denied,
    #[error("dynamic proof authority unavailable")]
    Unavailable,
}
type Result<T> = std::result::Result<T, Error>;

/// Immutable verified credential facts, never a positive authority snapshot.
/// No Deserialize, subject-keyed registry, Clone or production constructor.
struct CredentialHandle {
    expected: ExpectedPrimary,
    credential_id: String,
    credential_hash: [u8; 32],
}

/// FUTURE trusted adapter contract, currently implemented by fixtures only.
/// Authenticate Policy with static service keys and an explicit capability;
/// recheck this exact credential's revocation on EVERY call. Never cache success.
/// Return only the complete active H3a response; no browser-selected DB locator.
#[async_trait::async_trait]
trait CurrentPrimary: Send + Sync {
    async fn resolve_current(&self, credential: &CredentialHandle) -> Result<SessionPrimary>;
}

/// Input bytes are the original holder's proof/credential, not the relay's.
struct Request<'a> {
    proof: &'a [u8],
    credential: &'a [u8],
    service: &'a str,
    schema_id: u64,
    body: &'a [u8],
}

#[derive(Default)]
struct Consumer {
    provider: Option<Arc<dyn CurrentPrimary>>,
}

struct LocalPolicy<'a> {
    inventory: &'a CollisionInventory,
    static_enrollments: &'a dyn EnrollmentResolver,
    in_flight: &'a Semaphore,
}

impl Consumer {
    async fn verify(
        &self,
        handle: &CredentialHandle,
        request: &Request<'_>,
        policy: &LocalPolicy<'_>,
        clock: impl Fn() -> u64,
    ) -> Result<VerifiedProof> {
        let provider = self.provider.as_ref().ok_or(Error::Disabled)?;
        let now = clock();
        let e = &handle.expected;
        if e.issuer != HOST
            || e.audience != HOST
            || e.client != CLIENT
            || e.profile != PROFILE
            || !bounded(&e.sid, 128)
            || !bounded(&e.subject, 256)
            || !bounded(&e.tenant, 256)
            || !bounded(&handle.credential_id, 256)
            || e.expires_at <= 0
            || e.expires_at as u64 <= now
            || e.expires_at as u64 > now.saturating_add(300)
            || e.scopes.is_empty()
            || e.scopes.len() > 64
            || e.scopes
                .iter()
                .any(|s| !bounded(s, 256) || !s.bytes().all(|b| (0x21..=0x7e).contains(&b)))
            || e.scopes.windows(2).any(|s| s[0] >= s[1])
            || request.credential.is_empty()
            || request.credential.len() > 16 * 1024
            || <[u8; 32]>::from(Sha256::digest(request.credential)) != handle.credential_hash
        {
            return Err(Error::Denied);
        }
        let proof = ParsedProof::parse(request.proof).map_err(|_| Error::Denied)?;
        if proof.kind != ProofKind::Request
            || proof.disposition != ProofDisposition::Authenticated
            || proof.claims.aud != request.service
            || proof.claims.capnp_schema_id != request.schema_id
            || proof.claims.capnp_body_bytes != request.body
            || proof.claims.credential_hash != Some(handle.credential_hash)
            || proof.claims.exp > e.expires_at as u64
        {
            return Err(Error::Denied);
        }
        // No queue, positive cache or fallback; this includes each reuse of handle.
        let _permit = policy
            .in_flight
            .try_acquire()
            .map_err(|_| Error::Unavailable)?;
        let active = tokio::time::timeout(Duration::from_secs(2), provider.resolve_current(handle))
            .await
            .map_err(|_| Error::Unavailable)??;
        // Read the trusted verifier clock again after the await. Adding rounded
        // elapsed seconds to the initial timestamp can accept an expired proof.
        let now = clock();
        if e.expires_at as u64 <= now || active.created_at > now as i64 {
            return Err(Error::Denied);
        }
        let resolver = local_resolver(e, &active, policy)?;
        let holder = VerifyingKey::from_bytes(&e.ed_public).map_err(|_| Error::Denied)?;
        verify_proof_signatures(&proof, Some(&holder), Some(&resolver), now)
            .map_err(|_| Error::Denied)
    }
}

fn bounded(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max
}

/// Lives on one verification stack only. Static primary lookup is NEVER used.
struct RequestResolver<'a> {
    primary: SignerSuiteRecord,
    static_enrollments: &'a dyn EnrollmentResolver,
}

impl EnrollmentResolver for RequestResolver<'_> {
    fn resolve_primary(&self, key: &VerifyingKey) -> Option<SignerSuiteRecord> {
        self.primary.pins_ed25519(key).then(|| self.primary.clone())
    }
    fn resolve_approver(&self, kid: &[u8]) -> Option<SignerSuiteRecord> {
        self.static_enrollments.resolve_approver(kid)
    }
    fn resolve_service(&self, service: &str) -> Option<SignerSuiteRecord> {
        self.static_enrollments.resolve_service(service)
    }
}

fn local_resolver<'a>(
    e: &ExpectedPrimary,
    a: &SessionPrimary,
    policy: &LocalPolicy<'a>,
) -> Result<RequestResolver<'a>> {
    if a.host != e.issuer
        || a.profile != e.profile
        || a.suite != SUITE
        || a.sid != e.sid
        || a.subject != e.subject
        || a.tenant != e.tenant
        || a.client != e.client
        || a.resource != e.audience
        || a.scopes != e.scopes
        || a.ed_public != e.ed_public
        || a.generation != e.generation
        || a.expires_at < e.expires_at
        || a.created_at < 0
        || a.created_at >= e.expires_at
        || a.proof_epoch == 0
        || a.proof_epoch > i64::MAX as u64
        || !bounded(&a.account_id, 256)
        || !bounded(&a.grant_revision, 256)
        || a.collision_inventory_id != policy.inventory.id()
        || !policy.inventory.permits(&e.ed_public, &a.pq_public)
        || signer_suite_thumbprint(SUITE, &[&a.ed_public, &a.pq_public]) != e.suite_thumbprint
    {
        return Err(Error::Denied);
    }
    let ed = VerifyingKey::from_bytes(&e.ed_public).map_err(|_| Error::Denied)?;
    let pq = ml_dsa_vk_from_bytes(&a.pq_public).map_err(|_| Error::Denied)?;
    let kids = session_primary_kids(&e.ed_public, &a.pq_public);
    Ok(RequestResolver {
        primary: SignerSuiteRecord {
            principal: e.subject.clone(),
            suite_id: SUITE.into(),
            components: vec![
                EnrolledComponent::new(kids[0], ComponentKey::Ed25519(ed)),
                EnrolledComponent::new(kids[1], ComponentKey::MlDsa65(Box::new(pq))),
            ],
            epoch: a.proof_epoch,
            role: SignerRole::Primary,
            approver_role: None,
            enrollment_policy_id: PROFILE.into(),
            not_after: e.expires_at as u64,
            revoked: false, // Only the fresh active authority response reaches here.
        },
        static_enrollments: policy.static_enrollments,
    })
}

#[cfg(test)]
mod tests;
