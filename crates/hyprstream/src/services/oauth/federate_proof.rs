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
    auth::{Claims, Scope, parse_protected_header, signer_suite::signer_suite_thumbprint},
    crypto::pq::ml_dsa_vk_from_bytes,
    proof::{
        enrollment::{
            ComponentKey, EnrolledComponent, EnrollmentResolver, SignerRole, SignerSuiteRecord,
        },
        parser::ParsedProof,
        verify::{verify_proof_signatures, VerifiedProof},
        ProofDisposition, ProofKind,
    },
    service::EnvelopeContext,
};
use hyprstream_rpc_std::policy_client::{AdmitFederateRequest, PolicyClient, ResolveSessionPrimary, SessionPrimary};
use hyprstream_session_store::{primary::{CollisionInventory, ExpectedPrimary}, PROFILE, SUITE};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use super::federate_source::{session_primary_kids, CLIENT, HOST};

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Error {
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

impl CredentialHandle {
    /// The RPC verifier must have signature-verified the direct reserved host
    /// JWT and deferred identity publication. This creates only a credential
    /// snapshot; current primary, original-holder proof, durable replay and
    /// fresh action authority must all succeed before dispatch.
    fn from_verified_context(ctx: &EnvelopeContext) -> Result<Self> {
        let (claims, token) = ctx.deferred_federate_credential().ok_or(Error::Denied)?;
        let header = parse_protected_header(token).map_err(|_| Error::Denied)?;
        if header.typ != "at+jwt" {
            return Err(Error::Denied);
        }
        Self::from_claims_snapshot(claims, token)
    }

    /// Fixture-accessible parser of already signed facts. Never call on an
    /// unverified JWT; only `from_verified_context` is a production entry.
    fn from_claims_snapshot(claims: &Claims, token: &str) -> Result<Self> {
        let cnf = claims.cnf.as_ref().ok_or(Error::Denied)?;
        let jwk = cnf.jwk.as_ref().ok_or(Error::Denied)?;
        let ed = claims.cnf_key_bytes().ok_or(Error::Denied)?;
        let suite = cnf.hs_signer_suite.as_deref().ok_or(Error::Denied)?;
        let suite_bytes = URL_SAFE_NO_PAD.decode(suite).map_err(|_| Error::Denied)?;
        let suite_thumbprint: [u8; 32] = suite_bytes.try_into().map_err(|_| Error::Denied)?;
        let generation = claims
            .session_authority_generation()
            .map_err(|_| Error::Denied)?
            .ok_or(Error::Denied)?;
        let sid = claims.sid.as_deref().ok_or(Error::Denied)?;
        let subject = claims.sub.as_str();
        let tenant = claims.tenant.as_deref().ok_or(Error::Denied)?;
        let jti = claims.jti.as_deref().ok_or(Error::Denied)?;
        let scope = claims.scope.as_deref().ok_or(Error::Denied)?;
        let scopes: Vec<String> = scope.split(' ').map(str::to_owned).collect();
        if claims.iss != HOST
            || claims.aud.as_deref() != Some(HOST)
            || claims.client_id.as_deref() != Some(CLIENT)
            || claims.workload_session_id.is_some()
            || claims.act.is_some()
            || claims.cap.is_some()
            || cnf.jkt.is_some()
            || jwk.kty != "OKP"
            || jwk.crv != "Ed25519"
            || URL_SAFE_NO_PAD.encode(ed) != jwk.x
            || URL_SAFE_NO_PAD.encode(suite_thumbprint) != suite
            || !bounded(sid, 128)
            || !bounded(subject, 256)
            || subject == "anonymous"
            || subject.starts_with("service:")
            || !bounded(tenant, 256)
            || tenant == "*"
            || !bounded(jti, 256)
            || generation == [0; 32]
            || claims.iat < 0
            || claims.exp <= claims.iat
            || claims.exp - claims.iat > 300
            || scopes.is_empty()
            || scopes.len() > 64
            || scopes.windows(2).any(|window| window[0] >= window[1])
            || scopes.iter().any(|text| {
                !bounded(text, 256)
                    || !text.bytes().all(|b| (0x21..=0x7e).contains(&b))
                    || match Scope::parse(text) {
                        Ok(parsed) => {
                            parsed.action.contains('*')
                                || parsed.resource.contains('*')
                                || parsed.identifier.contains('*')
                        }
                        Err(_) => true,
                    }
            })
            || token.len() > 16 * 1024
        {
            return Err(Error::Denied);
        }
        Ok(Self {
            expected: ExpectedPrimary {
                issuer: claims.iss.clone(),
                profile: PROFILE.into(),
                sid: sid.into(),
                subject: subject.into(),
                tenant: tenant.into(),
                client: CLIENT.into(),
                audience: HOST.into(),
                scopes,
                ed_public: ed,
                suite_thumbprint,
                generation,
                expires_at: claims.exp,
            },
            credential_id: jti.into(),
            credential_hash: Sha256::digest(token.as_bytes()).into(),
        })
    }
}

/// FUTURE trusted adapter contract, currently implemented by fixtures only.
/// Authenticate Policy with static service keys and an explicit capability;
/// recheck this exact credential's revocation on EVERY call. Never cache success.
/// Return only the complete active H3a response; no browser-selected DB locator.
#[async_trait::async_trait]
trait CurrentPrimary: Send + Sync {
    async fn resolve_current(&self, credential: &CredentialHandle) -> Result<SessionPrimary>;
}

/// An authenticated service client, never the browser bearer. The trusted
/// service factory must construct `PolicyClient` with its own enrolled signer
/// and capability; this type has no public/runtime constructor or fallback.
#[allow(dead_code)] // Provider installation remains disabled until dispatch admission is complete.
struct PolicyPrimaryProvider {
    client: PolicyClient,
}

impl PolicyPrimaryProvider {
    fn request(credential: &CredentialHandle) -> ResolveSessionPrimary {
        let e = &credential.expected;
        ResolveSessionPrimary {
            issuer: e.issuer.clone(),
            profile: e.profile.clone(),
            sid: e.sid.clone(),
            subject: e.subject.clone(),
            tenant: e.tenant.clone(),
            client: e.client.clone(),
            audience: e.audience.clone(),
            scopes: e.scopes.clone(),
            ed_public: e.ed_public.to_vec(),
            suite_thumbprint: e.suite_thumbprint.to_vec(),
            generation: e.generation.to_vec(),
            expires_at: e.expires_at,
        }
    }
}

#[async_trait::async_trait]
impl CurrentPrimary for PolicyPrimaryProvider {
    async fn resolve_current(&self, credential: &CredentialHandle) -> Result<SessionPrimary> {
        self.client
            .resolve_session_primary(&Self::request(credential))
            .await
            .map_err(|_| Error::Unavailable)
    }
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

/// Disabled serving adapter. A future factory must provide an authenticated
/// service Policy client, complete local inventory and static approver roster;
/// no Model/Registry constructor installs one in this source slice.
#[allow(dead_code)]
pub(crate) struct DispatchAdapter {
    client: PolicyClient,
    inventory: CollisionInventory,
    static_enrollments: Arc<dyn EnrollmentResolver>,
    in_flight: Semaphore,
}

impl DispatchAdapter {
    pub(crate) async fn admit(
        &self,
        ctx: &EnvelopeContext,
        proof: &ParsedProof,
        service: &str,
        schema_id: u64,
        body: &[u8],
        resource: &str,
        operation: &str,
    ) -> Result<(VerifiedProof, [u8; 32])> {
        let handle = CredentialHandle::from_verified_context(ctx)?;
        let raw = ctx.request_proof_cwt().ok_or(Error::Denied)?;
        let consumer = Consumer {
            provider: Some(Arc::new(PolicyPrimaryProvider { client: self.client.clone() })),
        };
        let local = LocalPolicy {
            inventory: &self.inventory,
            static_enrollments: self.static_enrollments.as_ref(),
            in_flight: &self.in_flight,
        };
        let request = Request { proof: raw, credential: ctx.deferred_federate_credential().ok_or(Error::Denied)?.1.as_bytes(), service, schema_id, body };
        let (verified, active) = consumer.verify_with_primary(
            &handle, &request, &local,
            || chrono::Utc::now().timestamp().max(0) as u64,
        ).await?;
        if verified.primary_principal.as_deref() != Some(handle.expected.subject.as_str())
            || verified.primary_suite != SUITE
        {
            return Err(Error::Denied);
        }
        let e = &handle.expected;
        let data = AdmitFederateRequest {
            issuer: e.issuer.clone(),
            profile: e.profile.clone(),
            sid: e.sid.clone(),
            subject: e.subject.clone(),
            tenant: e.tenant.clone(),
            client: e.client.clone(),
            audience: e.audience.clone(),
            scopes: e.scopes.clone(),
            ed_public: e.ed_public.to_vec(),
            suite_thumbprint: e.suite_thumbprint.to_vec(),
            generation: e.generation.to_vec(),
            collision_inventory_id: active.collision_inventory_id,
            expires_at: e.expires_at,
            proof_epoch: active.proof_epoch,
            verified_namespace: verified.replay_thumbprint.to_vec(),
            request_id: proof.claims.request_id.to_vec(),
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        };
        let admitted = self.client.admit_federate_request(&data).await
            .map_err(|_| Error::Unavailable)?;
        if !admitted { return Err(Error::Denied); }
        Ok((verified, e.ed_public))
    }
}

impl Consumer {
    async fn verify(
        &self,
        handle: &CredentialHandle,
        request: &Request<'_>,
        policy: &LocalPolicy<'_>,
        clock: impl Fn() -> u64,
    ) -> Result<VerifiedProof> {
        self.verify_with_primary(handle, request, policy, clock)
            .await
            .map(|(verified, _)| verified)
    }

    async fn verify_with_primary(
        &self,
        handle: &CredentialHandle,
        request: &Request<'_>,
        policy: &LocalPolicy<'_>,
        clock: impl Fn() -> u64,
    ) -> Result<(VerifiedProof, SessionPrimary)> {
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
        let verified = verify_proof_signatures(&proof, Some(&holder), Some(&resolver), now)
            .map_err(|_| Error::Denied)?;
        Ok((verified, active))
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
