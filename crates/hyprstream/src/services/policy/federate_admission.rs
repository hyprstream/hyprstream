//! Disabled Policy-owned admission core. No RPC route, factory or token signer.
//! OAuth's future H4 adapter must supply typed verified possession evidence.
//! All serving requests/tool calls still require the separate H2 authority-at-use
//! gate: admission deliberately does not lock account/PDS/Casbin through commit.
#![allow(dead_code)] // Source-only slice; no runtime installer is exposed.

use super::{EnvelopeContext, PolicyManager};
use crate::auth::{
    postgres_store::PolicyAccountReader, service_enrollment::ServiceEnrollmentManifest,
};
use anyhow::{ensure, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_rpc::{
    auth::{signer_suite::signer_suite_thumbprint, Scope},
    proof::enrollment::authenticated_replay_namespace,
};
use hyprstream_rpc_std::policy_client::AdmitFederateRequest;
use hyprstream_session_store::{Admission, PendingSession, PrimaryRecord, Session, Source, Store};
use rand::RngCore as _;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};
use tokio::sync::{Mutex, Semaphore};

/// A Federate DID must have a small, explicit set of concrete tenant
/// memberships. This bounds the number of per-scope Policy checks and keeps
/// unrelated tenant rules out of login admission work.
const MAX_FEDERATE_TENANT_MEMBERSHIPS: usize = 128;

/// Fixed, authority-owned ceilings; not browser input or dynamic client metadata.
struct Profile {
    issuer: String,
    host: String,
    client: String,
    resource: String,
    client_scopes: BTreeSet<String>,
    resource_scopes: BTreeSet<String>,
}

/// Immutable result of Policy's prepare read. No public field/serde constructor.
#[derive(Clone, PartialEq, Eq)]
struct Decision {
    account_id: String,
    subject: String,
    tenant: String,
    scopes: Vec<String>,
    revision: String,
    generation: [u8; 32],
    collision_inventory_id: [u8; 32],
}

/// Returned only to the authenticated OAuth service. The handle itself stays
/// in OAuth's server-side pending challenge, never in its browser response.
pub(super) struct PreparedDecision {
    pub(super) account_id: String,
    pub(super) subject: String,
    pub(super) tenant: String,
    pub(super) requested: Vec<String>,
    pub(super) granted: Vec<String>,
    pub(super) revision: String,
    pub(super) handle: [u8; 32],
}

struct PendingDecision {
    source: Source,
    requested: Vec<String>,
    decision: Decision,
    ed_public: [u8; 32],
    pq_public: Vec<u8>,
    challenge_id: [u8; 32],
    created_at: i64,
    expires_at: i64,
}

/// Internal upstream evidence, not a browser DTO or verification boolean.
/// There is intentionally no production constructor until H4 implements BOTH
/// signatures over the exact challenge and strict source-token validation.
struct PossessionEvidence {
    source: Source,
    ed_public: [u8; 32],
    pq_public: Vec<u8>,
    sid: String,
    requested: Vec<String>,
    challenge: Decision,
    created_at: i64,
    expires_at: i64,
    session_expires_at: i64,
}

struct Authorities {
    users: PolicyAccountReader,
    policy: Arc<PolicyManager>,
    enrollment: Arc<ServiceEnrollmentManifest>,
    profile: Profile,
    /// Server-owned serving tuple from a complete trusted inventory/profile
    /// loader. No production constructor is installed by this source slice.
    serving_generation: [u8; 32],
    local_collision_inventory_id: [u8; 32],
}

/// Default denies before any authority or DB access. Future wiring must admit a
/// writer-endpoint UserStore, fresh signed PDS mount and sole serving Policy.
#[derive(Default)]
pub(super) struct AdmissionService {
    authority: Option<Authorities>,
    capacity: Option<Semaphore>,
    session_pool: Option<deadpool_postgres::Pool>,
    pending: Mutex<BTreeMap<[u8; 32], PendingDecision>>,
}

/// Source-only Policy authority for a single protected Registry/Model request.
/// No runtime factory installs it. The same configured `AdmissionService`
/// supplies the sole account/tenant/grant owner; serving processes hold only
/// authenticated Policy clients, never database credentials.
#[allow(dead_code)]
pub(super) struct RequestUseReader {
    admission: Arc<AdmissionService>,
    pool: deadpool_postgres::Pool,
    capacity: Semaphore,
    clock_skew_secs: i64,
}

impl RequestUseReader {
    pub(super) async fn admit(
        &self,
        ctx: &EnvelopeContext,
        data: &AdmitFederateRequest,
    ) -> Result<()> {
        let authority = self
            .admission
            .authority
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Federate authority disabled"))?;
        let caller = serving_caller(ctx, &authority.enrollment)?;
        ensure!(
            (caller == "registry" && data.resource.starts_with("registry:"))
                || (caller == "model" && data.resource.starts_with("model:")),
            "serving capability does not cover resource"
        );
        ensure!(
            self.clock_skew_secs >= 0 && self.clock_skew_secs <= 300,
            "invalid replay retention configuration"
        );
        ensure!(
            !data.sid.is_empty()
                && data.sid.len() <= 128
                && !data.subject.is_empty()
                && data.subject.len() <= 256
                && !data.tenant.is_empty()
                && data.tenant.len() <= 256
                && data.tenant != "*"
                && data.expires_at > chrono::Utc::now().timestamp(),
            "invalid or expired verified credential tuple"
        );
        canonical_requested(&data.scopes)?;
        let generation: [u8; 32] = data.generation.as_slice().try_into()?;
        let inventory: [u8; 32] = data.collision_inventory_id.as_slice().try_into()?;
        let namespace: [u8; 32] = data.verified_namespace.as_slice().try_into()?;
        let request_id: [u8; 16] = data.request_id.as_slice().try_into()?;
        let ed: [u8; 32] = data.ed_public.as_slice().try_into()?;
        let suite_thumbprint: [u8; 32] = data.suite_thumbprint.as_slice().try_into()?;
        ensure!(
            data.issuer == authority.profile.host
                && data.profile == hyprstream_session_store::PROFILE
                && data.client == authority.profile.client
                && data.audience == authority.profile.resource
                && generation == authority.serving_generation
                && inventory == authority.local_collision_inventory_id
                && data.proof_epoch > 0
                && data.proof_epoch <= i64::MAX as u64,
            "serving profile mismatch"
        );
        let _permit = self.capacity.try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), async {
            let mut client = self.pool.get().await?;
            let primary = Store::lookup_primary(
                &mut client,
                &authority.profile.host,
                &data.sid,
                &generation,
                &inventory,
            )
            .await?
            .ok_or_else(|| anyhow::anyhow!("session primary inactive"))?;
            let session = primary.session();
            ensure!(
                session.host == data.issuer
                    && session.sid == data.sid
                    && session.subject == data.subject
                    && session.tenant == data.tenant
                    && session.client_id == data.client
                    && session.resource == data.audience
                    && session.scopes == data.scopes
                    && session.ed_public == ed
                    && session.generation == generation
                    && session.collision_inventory_id == inventory
                    && session.proof_epoch == data.proof_epoch as i64
                    && session.expires_at >= data.expires_at,
                "verified credential/session mismatch"
            );
            let ordered = [session.ed_public.to_vec(), session.pq_public.clone()];
            ensure!(
                signer_suite_thumbprint(
                    hyprstream_session_store::SUITE,
                    &[&session.ed_public, &session.pq_public]
                ) == suite_thumbprint
                    && authenticated_replay_namespace(
                        hyprstream_session_store::SUITE,
                        &ordered,
                        session.proof_epoch as u64,
                    ) == namespace,
                "verified holder namespace mismatch"
            );
            let (source_issuer, source_subject) = primary.source_identity();
            ensure!(
                primary.source_atproto_did() == session.subject,
                "session subject does not match verified ATProto DID"
            );
            authority
                .authorize_use(
                    session,
                    source_issuer,
                    source_subject,
                    &data.resource,
                    &data.operation,
                    chrono::Utc::now().timestamp(),
                )
                .await?;
            // One shared writer transaction rechecks active session/profile
            // under row locks and atomically consumes the proof request ID.
            Store::consume_request_replay(
                &mut client,
                session,
                &namespace,
                &request_id,
                self.clock_skew_secs,
            )
            .await?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }
}

fn serving_caller(
    ctx: &EnvelopeContext,
    manifest: &ServiceEnrollmentManifest,
) -> Result<&'static str> {
    let subject = ctx.subject();
    let name = subject
        .name()
        .ok_or_else(|| anyhow::anyhow!("serving caller missing"))?;
    let caller = match name {
        "service:registry" => "registry",
        "service:model" => "model",
        _ => anyhow::bail!("unapproved serving caller"),
    };
    let claims = ctx
        .claims()
        .ok_or_else(|| anyhow::anyhow!("serving credential missing"))?;
    ensure!(
        ctx.is_authenticated()
            && !subject.is_federated()
            && claims.sub == name
            && claims.act.is_none()
            && claims.sid.is_none()
            && claims.workload_session_id.is_none(),
        "direct enrolled serving credential required"
    );
    let entry = manifest
        .services
        .get(caller)
        .ok_or_else(|| anyhow::anyhow!("serving enrollment missing"))?;
    let enrolled = URL_SAFE_NO_PAD.decode(&entry.ed25519_pubkey)?;
    ensure!(enrolled.as_slice() == ctx.cnf, "serving holder mismatch");
    Ok(caller)
}

fn oauth_caller(ctx: &EnvelopeContext, manifest: &ServiceEnrollmentManifest) -> Result<()> {
    let subject = ctx.subject();
    ensure!(
        ctx.is_authenticated()
            && ctx.claims().is_some()
            && !subject.is_federated()
            && subject.name() == Some("service:oauth"),
        "enrolled OAuth caller required"
    );
    let claims = ctx
        .claims()
        .ok_or_else(|| anyhow::anyhow!("service claims missing"))?;
    ensure!(
        claims.sub == "service:oauth" && claims.act.is_none() && claims.sid.is_none(),
        "direct OAuth service credential required"
    );
    let entry = manifest
        .services
        .get("oauth")
        .ok_or_else(|| anyhow::anyhow!("OAuth enrollment missing"))?;
    let enrolled = URL_SAFE_NO_PAD.decode(&entry.ed25519_pubkey)?;
    ensure!(enrolled.as_slice() == ctx.cnf, "OAuth holder mismatch");
    Ok(())
}

fn canonical_requested(requested: &[String]) -> Result<()> {
    ensure!(
        !requested.is_empty() && requested.len() <= 64,
        "invalid scope count"
    );
    ensure!(
        requested.windows(2).all(|w| w[0] < w[1]),
        "noncanonical scopes"
    );
    for text in requested {
        ensure!(
            text.len() <= 256 && text.bytes().all(|b| (0x21..=0x7e).contains(&b)),
            "invalid scope"
        );
        let scope = Scope::parse(text)?;
        ensure!(
            [&scope.action, &scope.resource, &scope.identifier]
                .iter()
                .all(|s| !s.is_empty() && !s.contains('*')),
            "concrete scopes required"
        );
    }
    Ok(())
}

fn candidate_tenants(
    did: &str,
    policies: &[Vec<String>],
    domain_groups: &[Vec<String>],
) -> Result<BTreeSet<String>> {
    let mut tenants = BTreeSet::new();
    // Only concrete grants or memberships attached directly to this verified
    // DID seed candidates. Role-based policies remain supported: the DID's
    // g2 membership supplies the domain, and Casbin resolves its role chain
    // during the final check. Unrelated users' tenant rules are never scanned
    // with check_with_domain.
    for (rules, subject_index, tenant_index) in [(policies, 0, 1), (domain_groups, 0, 2)] {
        for rule in rules {
            if rule.get(subject_index).is_some_and(|subject| subject == did) {
                if let Some(tenant) = rule
                    .get(tenant_index)
                    .filter(|tenant| !tenant.is_empty() && *tenant != "*")
                {
                    tenants.insert(tenant.clone());
                    ensure!(
                        tenants.len() <= MAX_FEDERATE_TENANT_MEMBERSHIPS,
                        "Federate DID exceeds the concrete tenant-membership limit"
                    );
                }
            }
        }
    }
    Ok(tenants)
}

impl Authorities {
    async fn decision(&self, source: &Source, requested: &[String]) -> Result<Decision> {
        ensure!(source.issuer == self.profile.issuer, "issuer mismatch");
        canonical_requested(requested)?;
        let username = self
            .users
            .get_external_identity_user(&source.issuer, &source.subject)
            .await?
            .ok_or_else(|| anyhow::anyhow!("account not admitted"))?;
        ensure!(
            !username.is_empty() && !username.starts_with("service:") && username != "anonymous",
            "local account subject required"
        );
        let profile = self
            .users
            .get_profile(&username)
            .await?
            .ok_or_else(|| anyhow::anyhow!("account not admitted"))?;
        ensure!(profile.active == Some(true), "inactive account");
        let account_id = profile
            .sub
            .ok_or_else(|| anyhow::anyhow!("account UUID missing"))?;
        uuid::Uuid::parse_str(&account_id)?;
        let did = &source.atproto_did;
        let policies = self.policy.get_policy().await;
        let domain_groups = self.policy.get_domain_grouping_policy().await;
        let tenants = candidate_tenants(did, &policies, &domain_groups)?;
        let mut matching = Vec::new();
        for tenant in tenants {
            let mut scopes = Vec::new();
            for text in requested {
                if !self.profile.client_scopes.contains(text)
                    || !self.profile.resource_scopes.contains(text)
                {
                    continue;
                }
                let scope = Scope::parse(text)?;
                if self
                    .policy
                    .check_with_domain(did, &tenant, &scope.policy_resource(), &scope.action)
                    .await
                {
                    scopes.push(text.clone());
                }
            }
            if !scopes.is_empty() {
                matching.push((tenant, scopes));
            }
        }
        ensure!(
            matching.len() == 1,
            "ATProto DID has no unique tenant grant for requested scopes"
        );
        let (tenant, scopes) = matching
            .pop()
            .ok_or_else(|| anyhow::anyhow!("unique DID tenant grant disappeared"))?;
        // Revision of this observed decision, not a global mutation epoch. A
        // change after these reads is accepted; use-time authorization is vital.
        let bytes = serde_json::to_vec(&(
            "federate-policy-decision-v1",
            &self.profile.issuer,
            &self.profile.host,
            &self.profile.client,
            &self.profile.resource,
            &self.profile.client_scopes,
            &self.profile.resource_scopes,
            &source.subject,
            &account_id,
            did,
            &tenant,
            requested,
            &scopes,
        ))?;
        let revision = blake3::hash(&bytes).to_hex().to_string();
        Ok(Decision {
            account_id,
            subject: did.clone(),
            tenant,
            scopes,
            revision,
            generation: self.serving_generation,
            collision_inventory_id: self.local_collision_inventory_id,
        })
    }

    /// Re-evaluate one protected operation from an exact, active session record
    /// and its durable source binding. `session` and `source` must come from the
    /// authenticated full-record lookup after the request's host JWT, holder
    /// proof, and v16 replay admission have been verified. This source slice has
    /// no constructor/route that can create that evidence; callers must not
    /// deserialize either structure from a client request.
    async fn authorize_use(
        &self,
        session: &Session,
        source_issuer: &str,
        source_subject: &str,
        resource: &str,
        operation: &str,
        now: i64,
    ) -> Result<()> {
        ensure!(
            source_issuer == self.profile.issuer,
            "source issuer mismatch"
        );
        ensure!(
            session.host == self.profile.host
                && session.client_id == self.profile.client
                && session.resource == self.profile.resource,
            "session profile binding mismatch"
        );
        ensure!(
            session.generation == self.serving_generation
                && session.collision_inventory_id == self.local_collision_inventory_id
                && session.proof_epoch > 0,
            "session authority is stale or unavailable"
        );
        ensure!(session.expires_at > now, "session expired");
        ensure!(
            !resource.is_empty()
                && resource.len() <= 256
                && resource.bytes().all(|b| (0x21..=0x7e).contains(&b))
                && !resource.contains('*')
                && !operation.is_empty()
                && operation.len() <= 64
                && operation
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_'),
            "invalid protected operation"
        );
        canonical_requested(&session.scopes)?;
        ensure!(
            session.scopes.iter().all(|scope| {
                self.profile.client_scopes.contains(scope)
                    && self.profile.resource_scopes.contains(scope)
            }),
            "session exceeds the fixed profile ceiling"
        );
        let scoped = session.scopes.iter().any(|text| {
            Scope::parse(text)
                .is_ok_and(|scope| scope.action == operation && scope.policy_resource() == resource)
        });
        ensure!(scoped, "operation is outside the admitted session scope");

        // These reads are intentionally fresh at every call boundary. Never
        // replace them with a request/session authorization cache.
        let username = self
            .users
            .get_external_identity_user(source_issuer, source_subject)
            .await?
            .ok_or_else(|| anyhow::anyhow!("source identity is no longer admitted"))?;
        let profile = self
            .users
            .get_profile(&username)
            .await?
            .ok_or_else(|| anyhow::anyhow!("account is no longer admitted"))?;
        ensure!(profile.active == Some(true), "account is inactive");
        let account_id = profile
            .sub
            .ok_or_else(|| anyhow::anyhow!("account UUID missing"))?;
        ensure!(account_id == session.account_id, "account binding changed");
        let did = &session.subject;
        ensure!(
            (did.starts_with("did:plc:") || did.starts_with("did:web:")) && did.len() <= 255,
            "session subject is not an ATProto DID"
        );
        ensure!(
            self.policy
                .check_with_domain(did, &session.tenant, resource, operation)
                .await,
            "current DID tenant grant denied"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;

impl AdmissionService {
    /// Cross-process OAuth prepare. A Policy restart drops all unredeemed
    /// handles; the OAuth process likewise drops its pending challenges. A
    /// restart is fail-closed, never a reason to reconstruct browser authority.
    pub(super) async fn prepare_rpc(
        &self,
        ctx: &EnvelopeContext,
        source: Source,
        requested: Vec<String>,
        challenge_id: [u8; 32],
        created_at: i64,
        ed_public: [u8; 32],
        pq_public: Vec<u8>,
    ) -> Result<PreparedDecision> {
        self.authority(ctx)?;
        ensure!(
            self.session_pool.is_some(),
            "Federate session writer disabled"
        );
        let now = chrono::Utc::now().timestamp();
        ensure!(challenge_id != [0; 32], "invalid challenge ID");
        ensure!(pq_public.len() == 1952, "invalid PQ public key");
        ensure!(
            created_at >= now - 30 && created_at <= now + 5,
            "stale challenge"
        );
        ensure!(
            source.issued_at >= 0 && source.issued_at <= now + 30,
            "invalid source time"
        );
        ensure!(
            source.expires_at > now && source.expires_at <= source.issued_at + 300,
            "expired source"
        );
        let expires_at = created_at
            .checked_add(60)
            .ok_or_else(|| anyhow::anyhow!("invalid challenge time"))?
            .min(source.expires_at)
            .min(source.issued_at + 120);
        ensure!(expires_at > now, "challenge already expired");
        let decision = self.prepare(ctx, &source, &requested).await?;
        let mut pending = self.pending.lock().await;
        pending.retain(|_, entry| entry.expires_at > now);
        ensure!(pending.len() < 128, "Federate challenge capacity exhausted");
        let mut handle = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut handle);
        ensure!(
            handle != [0; 32] && !pending.contains_key(&handle),
            "handle collision"
        );
        let response = PreparedDecision {
            account_id: decision.account_id.clone(),
            subject: decision.subject.clone(),
            tenant: decision.tenant.clone(),
            requested: requested.clone(),
            granted: decision.scopes.clone(),
            revision: decision.revision.clone(),
            handle,
        };
        pending.insert(
            handle,
            PendingDecision {
                source,
                requested,
                decision,
                ed_public,
                pq_public,
                challenge_id,
                created_at,
                expires_at,
            },
        );
        Ok(response)
    }

    /// Consume exactly one server-owned challenge before doing any database
    /// work. Failure (including ambiguous commit) does not make it reusable.
    pub(super) async fn commit_rpc(
        &self,
        ctx: &EnvelopeContext,
        handle: [u8; 32],
        source: Source,
        challenge_id: [u8; 32],
        created_at: i64,
        expires_at: i64,
        ed_public: [u8; 32],
        pq_public: Vec<u8>,
    ) -> Result<Session> {
        self.authority(ctx)?;
        let pool = self
            .session_pool
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Federate session writer disabled"))?;
        let pending = self
            .pending
            .lock()
            .await
            .remove(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown or consumed challenge"))?;
        let now = chrono::Utc::now().timestamp();
        ensure!(
            now < pending.expires_at
                && source == pending.source
                && challenge_id == pending.challenge_id
                && created_at == pending.created_at
                && expires_at == pending.expires_at
                && ed_public == pending.ed_public
                && pq_public == pending.pq_public,
            "challenge/source binding changed"
        );
        ensure!(pq_public.len() == 1952, "invalid PQ public key");
        let mut client = tokio::time::timeout(Duration::from_secs(2), pool.get()).await??;
        self.redeem(
            ctx,
            &mut client,
            PossessionEvidence {
                source,
                ed_public,
                pq_public,
                sid: uuid::Uuid::new_v4().to_string(),
                requested: pending.requested,
                challenge: pending.decision,
                created_at,
                expires_at,
                session_expires_at: pending.source.expires_at.min(now + 300),
            },
        )
        .await
    }

    fn authority(&self, ctx: &EnvelopeContext) -> Result<&Authorities> {
        let authority = self
            .authority
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Federate admission disabled"))?;
        oauth_caller(ctx, &authority.enrollment)?;
        Ok(authority)
    }

    async fn prepare(
        &self,
        ctx: &EnvelopeContext,
        source: &Source,
        requested: &[String],
    ) -> Result<Decision> {
        let a = self.authority(ctx)?;
        let _permit = self
            .capacity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), a.decision(source, requested)).await?
    }

    async fn redeem(
        &self,
        ctx: &EnvelopeContext,
        client: &mut tokio_postgres::Client,
        evidence: PossessionEvidence,
    ) -> Result<Session> {
        let a = self.authority(ctx)?;
        let _permit = self
            .capacity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), async {
            let current = a.decision(&evidence.source, &evidence.requested).await?;
            ensure!(current == evidence.challenge, "challenge decision changed");
            let session = PendingSession {
                host: a.profile.host.clone(),
                sid: evidence.sid,
                account_id: current.account_id,
                subject: current.subject,
                tenant: current.tenant,
                client_id: a.profile.client.clone(),
                resource: a.profile.resource.clone(),
                scopes: current.scopes,
                grant_revision: current.revision,
                ed_public: evidence.ed_public,
                pq_public: evidence.pq_public,
                generation: current.generation,
                collision_inventory_id: current.collision_inventory_id,
                expires_at: evidence.session_expires_at,
            };
            // No cross-store lock: a mutation here may leave a stale session.
            // No token signer or protected handler is reachable from this core.
            Ok(Store::admit(
                client,
                &Admission {
                    session,
                    grant_expires_at: evidence.source.expires_at,
                    source: evidence.source,
                    local_collision_inventory_id: a.local_collision_inventory_id,
                    challenge_created_at: evidence.created_at,
                    challenge_expires_at: evidence.expires_at,
                },
            )
            .await?)
        })
        .await?
    }

    async fn lookup(
        &self,
        ctx: &EnvelopeContext,
        client: &tokio_postgres::Client,
        sid: &str,
        generation: &[u8; 32],
    ) -> Result<Option<Session>> {
        let a = self.authority(ctx)?;
        let _permit = self
            .capacity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        ensure!(!sid.is_empty() && sid.len() <= 128, "invalid sid");
        ensure!(
            generation == &a.serving_generation,
            "stale serving generation"
        );
        Ok(tokio::time::timeout(
            Duration::from_secs(2),
            Store::lookup(
                client,
                &a.profile.host,
                sid,
                generation,
                &a.local_collision_inventory_id,
            ),
        )
        .await??)
    }

    /// Fresh authority-at-use check. It intentionally returns no reusable
    /// permit/credential; the caller invokes it again for every later request
    /// or distinct tool call. The protected response stream belongs to the
    /// already-authorized request and does not poll this method.
    async fn authorize_use(
        &self,
        ctx: &EnvelopeContext,
        client: &mut tokio_postgres::Client,
        primary: &PrimaryRecord,
        resource: &str,
        operation: &str,
    ) -> Result<()> {
        let authority = self.authority(ctx)?;
        let _permit = self
            .capacity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), async {
            // A PrimaryRecord is a request-local proof/source binding, not a
            // reusable authorization snapshot. Refresh current DB state at
            // every protected request or distinct tool-call boundary so a
            // revoke/profile disable/generation rotation takes effect there.
            let current = Store::lookup_primary(
                client,
                &authority.profile.host,
                &primary.session().sid,
                &authority.serving_generation,
                &authority.local_collision_inventory_id,
            )
            .await?
            .ok_or_else(|| anyhow::anyhow!("session is no longer active"))?;
            ensure!(
                current.session() == primary.session()
                    && current.source_identity() == primary.source_identity()
                    && current.source_atproto_did() == primary.source_atproto_did(),
                "session primary binding changed"
            );
            ensure!(
                current.source_atproto_did() == current.session().subject,
                "session subject does not match verified ATProto DID"
            );
            authority
                .authorize_use(
                    current.session(),
                    current.source_identity().0,
                    current.source_identity().1,
                    resource,
                    operation,
                    chrono::Utc::now().timestamp(),
                )
                .await
        })
        .await??;
        Ok(())
    }

    async fn revoke(
        &self,
        ctx: &EnvelopeContext,
        client: &tokio_postgres::Client,
        sid: &str,
        generation: &[u8; 32],
    ) -> Result<bool> {
        let a = self.authority(ctx)?;
        let _permit = self
            .capacity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        ensure!(!sid.is_empty() && sid.len() <= 128, "invalid sid");
        Ok(tokio::time::timeout(
            Duration::from_secs(2),
            Store::revoke(client, &a.profile.host, sid, generation),
        )
        .await??)
    }
}
