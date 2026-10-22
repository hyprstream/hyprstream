//! Request-local Federate proof verification and Policy admission.
//!
//! The serving factory installs this adapter with authenticated Policy transport
//! and static proof enrollment; no session-store URL is required. The adapter
//! accepts only a host JWT already signature-verified by the RPC layer, gets
//! the hybrid proof key from that signed credential, asks Policy for a fresh
//! grant decision, then returns replay admission data for dispatch to consume
//! only after the generated method-policy and mandatory MAC checks. No
//! subject-keyed positive cache or browser-selected DB locator is used.

use std::{sync::Arc, time::Duration};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::VerifyingKey;
use hyprstream_rpc::{
    auth::{parse_protected_header, signer_suite::signer_suite_thumbprint, Claims, Scope},
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
use hyprstream_rpc_std::policy_client::{AdmitFederateRequest, PolicyClient, SessionPrimary};
use hyprstream_session_store::{primary::ExpectedPrimary, PROFILE, SUITE};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

#[cfg(test)]
use super::federate_source::HOST;
use super::federate_source::{session_primary_kids, FederateHost, CLIENT};

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

/// Exercise the real signed-credential/holder verifier at the dispatch boundary.
/// The caller's fixture supplies action admission/replay separately; this does
/// not stand in for the production Policy decision.
#[cfg(test)]
pub(crate) async fn verify_dispatch_fixture(
    ctx: &EnvelopeContext,
    service: &str,
    schema_id: u64,
    body: &[u8],
    host: &FederateHost,
) -> Result<(VerifiedProof, [u8; 32])> {
    let handle = CredentialHandle::from_verified_context(ctx, host)?;
    let enrollments = hyprstream_rpc::proof::enrollment::InMemoryEnrollmentResolver::new();
    let semaphore = Semaphore::new(1);
    let local = LocalPolicy {
        static_enrollments: &enrollments,
        in_flight: &semaphore,
    };
    let consumer = Consumer {
        provider: Some(Arc::new(SignedCredentialPrimaryProvider)),
    };
    let request = Request {
        proof: ctx.request_proof_cwt().ok_or(Error::Denied)?,
        credential: ctx
            .deferred_federate_credential()
            .ok_or(Error::Denied)?
            .1
            .as_bytes(),
        service,
        schema_id,
        body,
    };
    let (verified, _) = consumer
        .verify_with_primary(&handle, &request, &local, || {
            chrono::Utc::now().timestamp().max(0) as u64
        })
        .await?;
    Ok((verified, handle.expected.ed_public))
}

/// Immutable verified credential facts, never a positive authority snapshot.
/// No Deserialize, subject-keyed registry, Clone or production constructor.
struct CredentialHandle {
    host: FederateHost,
    expected: ExpectedPrimary,
    credential_id: String,
    credential_hash: [u8; 32],
    pq_public: Vec<u8>,
    issued_at: i64,
}

impl CredentialHandle {
    /// The RPC verifier must have signature-verified the direct reserved host
    /// JWT and deferred identity publication. This creates only a credential
    /// snapshot; original-holder proof, shared replay and fresh Policy action
    /// authority must all succeed before dispatch.
    fn from_verified_context(ctx: &EnvelopeContext, host: &FederateHost) -> Result<Self> {
        let (claims, token) = ctx.deferred_federate_credential().ok_or(Error::Denied)?;
        let header = parse_protected_header(token).map_err(|_| Error::Denied)?;
        if header.typ != "at+jwt" {
            return Err(Error::Denied);
        }
        Self::from_claims_for_host(claims, token, host)
    }

    /// The dispatcher calls this only after the holder proof has been
    /// committed to the request context. The claims and token remain the
    /// exact facts verified by the envelope verifier; no caller-supplied
    /// identity is accepted here.
    fn from_committed_context(ctx: &EnvelopeContext, host: &FederateHost) -> Result<Self> {
        if !ctx.is_federate_admitted() {
            return Err(Error::Denied);
        }
        let claims = ctx.claims().ok_or(Error::Denied)?;
        let token = ctx.jwt_token().ok_or(Error::Denied)?;
        let header = parse_protected_header(token).map_err(|_| Error::Denied)?;
        if header.typ != "at+jwt" {
            return Err(Error::Denied);
        }
        Self::from_claims_for_host(claims, token, host)
    }

    /// Fixture-accessible parser of already signed facts. Never call on an
    /// unverified JWT; only `from_verified_context` is a production entry.
    #[cfg(test)]
    fn from_claims_snapshot(claims: &Claims, token: &str) -> Result<Self> {
        Self::from_claims_for_host(
            claims,
            token,
            &FederateHost::new(HOST).map_err(|_| Error::Denied)?,
        )
    }
    fn from_claims_for_host(claims: &Claims, token: &str, host: &FederateHost) -> Result<Self> {
        let cnf = claims.cnf.as_ref().ok_or(Error::Denied)?;
        let jwk = cnf.jwk.as_ref().ok_or(Error::Denied)?;
        let ed = claims.cnf_key_bytes().ok_or(Error::Denied)?;
        let suite = claims.hs_signer_suite_v1.as_deref().ok_or(Error::Denied)?;
        let suite_bytes = URL_SAFE_NO_PAD.decode(suite).map_err(|_| Error::Denied)?;
        let suite_thumbprint: [u8; 32] = suite_bytes.try_into().map_err(|_| Error::Denied)?;
        let pq_encoded = claims.hs_signer_pq_key_v1.as_deref().ok_or(Error::Denied)?;
        let pq_public = URL_SAFE_NO_PAD
            .decode(pq_encoded)
            .map_err(|_| Error::Denied)?;
        let source_issuer = claims.hs_source_issuer_v1.as_deref().ok_or(Error::Denied)?;
        let source_subject = claims
            .hs_source_subject_v1
            .as_deref()
            .ok_or(Error::Denied)?;
        let generation = claims
            .federate_signer_inventory()
            .map_err(|_| Error::Denied)?
            .ok_or(Error::Denied)?;
        let sid = claims.sid.as_deref().ok_or(Error::Denied)?;
        let subject = claims.sub.as_str();
        let tenant = claims.tenant.as_deref().ok_or(Error::Denied)?;
        let jti = claims.jti.as_deref().ok_or(Error::Denied)?;
        let scope = claims.scope.as_deref().ok_or(Error::Denied)?;
        let scopes: Vec<String> = scope.split(' ').map(str::to_owned).collect();
        if claims.iss != host.as_str()
            || claims.aud.as_deref() != Some(host.as_str())
            || claims.client_id.as_deref() != Some(CLIENT)
            || claims.workload_session_id.is_some()
            || claims.act.is_some()
            || claims.cap.is_some()
            || cnf.jkt.is_some()
            || cnf.hs_signer_suite.is_some()
            // Mixed-marker defense (W1.3/W1.4): the www consumer accepts
            // ONLY the exact www class. Any TUI-consumer marker on a
            // www-class credential is rejected here — www and TUI marker
            // sets may never coexist on one accepted credential.
            || claims.has_federate_tui_profile_marker()
            || claims.hs_profile.as_deref() != Some(PROFILE)
            || jwk.kty != "OKP"
            || jwk.crv != "Ed25519"
            || URL_SAFE_NO_PAD.encode(ed) != jwk.x
            || URL_SAFE_NO_PAD.encode(suite_thumbprint) != suite
            || claims
                .session_authority_generation()
                .map_err(|_| Error::Denied)?
                .is_some()
            || pq_public.len() != 1952
            || URL_SAFE_NO_PAD.encode(&pq_public) != pq_encoded
            || ml_dsa_vk_from_bytes(&pq_public).is_err()
            || signer_suite_thumbprint(SUITE, &[&ed, &pq_public]) != suite_thumbprint
            || source_issuer != super::federate_source::ISSUER
            || !bounded(source_subject, 256)
            || !bounded(sid, 128)
            || !bounded(subject, 256)
            || subject == "anonymous"
            || subject.starts_with("service:")
            || !crate::auth::valid_tenant_component(tenant)
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
            host: host.clone(),
            expected: ExpectedPrimary {
                issuer: claims.iss.clone(),
                profile: PROFILE.into(),
                sid: sid.into(),
                subject: subject.into(),
                tenant: tenant.into(),
                client: CLIENT.into(),
                audience: host.as_str().into(),
                scopes,
                ed_public: ed,
                suite_thumbprint,
                generation,
                expires_at: claims.exp,
            },
            credential_id: jti.into(),
            credential_hash: Sha256::digest(token.as_bytes()).into(),
            pq_public,
            issued_at: claims.iat,
        })
    }
}

/// Verified facts of a TUI-consumer Federate credential. Constructed only
/// from an already signature-verified local JWT; never a client DTO.
pub(crate) struct TuiCredentialFacts {
    pub(crate) subject: String,
    pub(crate) tenant: String,
    pub(crate) sid: String,
    pub(crate) scopes: Vec<String>,
    pub(crate) ed_public: [u8; 32],
    pub(crate) pq_public: Vec<u8>,
}

/// Consumer check for the TUI client class (W1.3): the class-aware
/// counterpart of the www `CredentialHandle` profile checks. Called only
/// with claims the RPC layer already signature-verified through NORMAL
/// JWT verification — the TUI credential never enters the deferred
/// federate branch, so there is no `deferred_federate` handle for it.
///
/// Rejects, fail-closed:
/// * a client other than the exact configured TUI client (including the
///   www staging client — cross-class substitution);
/// * any www `federate-session-v1` marker (mixed markers), or a missing /
///   incomplete TUI marker set (stripped markers);
/// * any scope whose resource is not `tui` (registry|model consumer
///   authority can never ride a TUI credential);
/// * a missing/malformed `cnf.jwk` Ed25519 holder binding or PQ key half.
pub(crate) fn verify_tui_credential_claims(
    claims: &Claims,
    host: &FederateHost,
    tui_client: &str,
) -> Result<TuiCredentialFacts> {
    use super::federate_source;
    let cnf = claims.cnf.as_ref().ok_or(Error::Denied)?;
    let jwk = cnf.jwk.as_ref().ok_or(Error::Denied)?;
    let ed = claims.cnf_key_bytes().ok_or(Error::Denied)?;
    let pq_encoded = claims
        .hs_tui_signer_pq_key_v1
        .as_deref()
        .ok_or(Error::Denied)?;
    let pq_public = URL_SAFE_NO_PAD
        .decode(pq_encoded)
        .map_err(|_| Error::Denied)?;
    let sid = claims.sid.as_deref().ok_or(Error::Denied)?;
    let subject = claims.sub.as_str();
    let tenant = claims.tenant.as_deref().ok_or(Error::Denied)?;
    let jti = claims.jti.as_deref().ok_or(Error::Denied)?;
    let scope = claims.scope.as_deref().ok_or(Error::Denied)?;
    let scopes: Vec<String> = scope.split(' ').map(str::to_owned).collect();
    if claims.iss != host.as_str()
        || claims.aud.as_deref() != Some(host.as_str())
        || claims.client_id.as_deref() != Some(tui_client)
        || claims.workload_session_id.is_some()
        || claims.act.is_some()
        || claims.cap.is_some()
        || cnf.jkt.is_some()
        || cnf.hs_signer_suite.is_some()
        // The complete TUI marker set, exactly this class's profile string.
        || claims.hs_tui_profile.as_deref() != Some(federate_source::TUI_PROFILE)
        || claims.hs_tui_source_issuer_v1.as_deref() != Some(federate_source::ISSUER)
        || claims.hs_tui_source_subject_v1.as_deref().is_none()
        // Mixed markers: no www federate-session marker may coexist.
        || claims.has_federate_profile_marker()
        // Wrong proof keys / malformed holder binding.
        || jwk.kty != "OKP"
        || jwk.crv != "Ed25519"
        || URL_SAFE_NO_PAD.encode(ed) != jwk.x
        || pq_public.len() != 1952
        || URL_SAFE_NO_PAD.encode(&pq_public) != pq_encoded
        || ml_dsa_vk_from_bytes(&pq_public).is_err()
        || claims
            .session_authority_generation()
            .map_err(|_| Error::Denied)?
            .is_some()
        || !bounded(&claims.hs_tui_source_subject_v1.clone().unwrap_or_default(), 256)
        || !bounded(sid, 128)
        || !bounded(subject, 256)
        || subject == "anonymous"
        || subject.starts_with("service:")
        || !crate::auth::valid_tenant_component(tenant)
        || !bounded(jti, 256)
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
                    // Intended-vs-forbidden services (W1.4): only the `tui`
                    // service resource is consumer authority for this class.
                    Ok(parsed) => parsed.resource != "tui" || parsed.identifier.is_empty(),
                    Err(_) => true,
                }
        })
    {
        return Err(Error::Denied);
    }
    Ok(TuiCredentialFacts {
        subject: subject.into(),
        tenant: tenant.into(),
        sid: sid.into(),
        scopes,
        ed_public: ed,
        pq_public,
    })
}

/// Resolve proof keys from the already verified, host-signed credential. This
/// is proof material only; Policy still authorizes every protected operation.
#[async_trait::async_trait]
trait CurrentPrimary: Send + Sync {
    async fn resolve_current(&self, credential: &CredentialHandle) -> Result<SessionPrimary>;
}

struct SignedCredentialPrimaryProvider;

#[async_trait::async_trait]
impl CurrentPrimary for SignedCredentialPrimaryProvider {
    async fn resolve_current(&self, credential: &CredentialHandle) -> Result<SessionPrimary> {
        let e = &credential.expected;
        Ok(SessionPrimary {
            host: e.issuer.clone(),
            profile: e.profile.clone(),
            suite: SUITE.into(),
            sid: e.sid.clone(),
            account_id: federate_account_id(&e.issuer, &e.subject),
            subject: e.subject.clone(),
            tenant: e.tenant.clone(),
            client: e.client.clone(),
            resource: e.audience.clone(),
            scopes: e.scopes.clone(),
            grant_revision: credential.credential_id.clone(),
            ed_public: e.ed_public.to_vec(),
            pq_public: credential.pq_public.clone(),
            generation: e.generation.to_vec(),
            created_at: credential.issued_at,
            expires_at: e.expires_at,
            proof_epoch: 1,
            collision_inventory_id: e.generation.to_vec(),
        })
    }
}

fn federate_account_id(host: &str, did: &str) -> String {
    let name = format!("hyprstream-federate-account-v1\0{host}\0{did}");
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, name.as_bytes()).to_string()
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
    static_enrollments: &'a dyn EnrollmentResolver,
    in_flight: &'a Semaphore,
}

/// Serving adapter installed only when the authenticated Policy client and
/// static proof enrollment are available. Policy validates the complete signer
/// inventory at login and checks its signed digest at every action; serving
/// services do not need the node root key or a session database.
pub(crate) struct DispatchAdapter {
    host: FederateHost,
    client: PolicyClient,
    static_enrollments: Arc<dyn EnrollmentResolver>,
    in_flight: Semaphore,
}

struct GlobalEnrollmentResolver(&'static dyn EnrollmentResolver);

impl EnrollmentResolver for GlobalEnrollmentResolver {
    fn resolve_primary(&self, key: &VerifyingKey) -> Option<SignerSuiteRecord> {
        self.0.resolve_primary(key)
    }

    fn resolve_approver(&self, kid: &[u8]) -> Option<SignerSuiteRecord> {
        self.0.resolve_approver(kid)
    }

    fn resolve_service(&self, service: &str) -> Option<SignerSuiteRecord> {
        self.0.resolve_service(service)
    }
}

impl DispatchAdapter {
    pub(crate) fn from_global_enrollment(
        client: PolicyClient,
        host: FederateHost,
    ) -> anyhow::Result<Self> {
        let resolver = hyprstream_rpc::proof::enrollment::global_enrollment_resolver()
            .ok_or_else(|| anyhow::anyhow!("Federate proof enrollment is unavailable"))?;
        Ok(Self {
            host,
            client,
            static_enrollments: Arc::new(GlobalEnrollmentResolver(resolver)),
            in_flight: Semaphore::new(32),
        })
    }

    pub(crate) async fn admit(
        &self,
        ctx: &EnvelopeContext,
        proof: &ParsedProof,
        service: &str,
        schema_id: u64,
        body: &[u8],
    ) -> Result<(
        VerifiedProof,
        [u8; 32],
        hyprstream_rpc::proof::admission::ProofReplayKey,
        u64,
    )> {
        let handle = CredentialHandle::from_verified_context(ctx, &self.host)?;
        let raw = ctx.request_proof_cwt().ok_or(Error::Denied)?;
        let consumer = Consumer {
            provider: Some(Arc::new(SignedCredentialPrimaryProvider)),
        };
        let local = LocalPolicy {
            static_enrollments: self.static_enrollments.as_ref(),
            in_flight: &self.in_flight,
        };
        let request = Request {
            proof: raw,
            credential: ctx
                .deferred_federate_credential()
                .ok_or(Error::Denied)?
                .1
                .as_bytes(),
            service,
            schema_id,
            body,
        };
        let verified = consumer
            .verify(&handle, &request, &local, || {
                chrono::Utc::now().timestamp().max(0) as u64
            })
            .await?;
        if verified.primary_principal.as_deref() != Some(handle.expected.subject.as_str())
            || verified.primary_suite != SUITE
        {
            return Err(Error::Denied);
        }
        let e = &handle.expected;
        let replay_key = hyprstream_rpc::proof::admission::ProofReplayKey {
            signer_thumbprint: verified.replay_thumbprint,
            request_id: proof.claims.request_id,
        };
        let replay_expiry = proof.claims.exp.min(e.expires_at as u64);
        Ok((verified, e.ed_public, replay_key, replay_expiry))
    }

    /// Run the fresh Policy check and its durable replay consume only after
    /// the dispatcher has passed generated method-policy and MAC checks.
    pub(crate) async fn commit_request_use(
        &self,
        ctx: &EnvelopeContext,
        resource: &str,
        operation: &str,
        replay_key: &hyprstream_rpc::proof::admission::ProofReplayKey,
        replay_expiry: u64,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp().max(0) as u64;
        if replay_expiry <= now || !ctx.admitted_federate_operation(resource, operation) {
            return Err(Error::Denied);
        }
        let handle = CredentialHandle::from_committed_context(ctx, &self.host)?;
        let active = SignedCredentialPrimaryProvider
            .resolve_current(&handle)
            .await?;
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
            verified_namespace: replay_key.signer_thumbprint.to_vec(),
            request_id: replay_key.request_id.to_vec(),
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        };
        let admitted = self
            .client
            .admit_federate_request(&data)
            .await
            .map_err(|_| Error::Unavailable)?;
        if admitted {
            Ok(())
        } else {
            Err(Error::Denied)
        }
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
        if e.issuer != handle.host.as_str()
            || e.audience != handle.host.as_str()
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
        let proof = ParsedProof::parse_deferred_federate_request(request.proof)
            .map_err(|_| Error::Denied)?;
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
        let resolver = local_resolver(e, &active, &handle.pq_public, policy)?;
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
    credential_pq_public: &[u8],
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
        || a.pq_public != credential_pq_public
        || a.generation != e.generation
        || a.expires_at < e.expires_at
        || a.created_at < 0
        || a.created_at >= e.expires_at
        || a.proof_epoch == 0
        || a.proof_epoch > i64::MAX as u64
        || !bounded(&a.account_id, 256)
        || !bounded(&a.grant_revision, 256)
        // This value comes from authenticated Policy after it verified the
        // active key against its complete local inventory. Serving services
        // must not reconstruct that inventory from their own (service) key.
        || a.collision_inventory_id.len() != 32
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
