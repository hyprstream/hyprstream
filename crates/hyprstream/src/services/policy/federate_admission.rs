//! Policy-owned admission core for the Federate session flow.
//! The production factory installs it only with explicit TLS PostgreSQL config,
//! the enrolled service manifest, complete local key inventory, and one enabled
//! matching profile. OAuth verifies source-token possession; Policy rechecks
//! current grants at admission and every protected request/tool-call boundary.
//! Admission deliberately does not lock account/PDS/Casbin through commit.
#![allow(dead_code)] // The staging profile is intentionally not a general-purpose API.

use super::{EnvelopeContext, PolicyManager};
use crate::auth::service_enrollment::ServiceEnrollmentManifest;
use anyhow::{ensure, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_rpc::{
    auth::{signer_suite::signer_suite_thumbprint, Scope},
    proof::{
        admission::{
            global_proof_replay_store, CredentialReplayKey, ProofAdmissionResult, ProofReplayStore,
            ReplayDomainGuarantee,
        },
        enrollment::authenticated_replay_namespace,
    },
};
use hyprstream_rpc_std::policy_client::AdmitFederateRequest;
use hyprstream_session_store::{
    primary::CollisionInventory, Admission, PendingSession, PrimaryRecord, Session, Source, Store,
};
use rand::RngCore as _;
use sha2::{Digest, Sha256};
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
    /// The verified client class selector this challenge was prepared under.
    /// The commit must name the same class or the one-use handle is denied.
    client: String,
    /// The session-establishment flow OAuth verified this source under
    /// (W1b). The commit must repeat it, and device-flow consumption uses
    /// the nonce-free replay identity only (D2).
    device_flow: bool,
    source: Source,
    requested: Vec<String>,
    decision: Decision,
    ed_public: [u8; 32],
    pq_public: Vec<u8>,
    challenge_id: [u8; 32],
    created_at: i64,
    expires_at: i64,
}

trait CredentialReplayAdmission: Send + Sync {
    fn admit_source_credential(
        &self,
        key: &CredentialReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult;
}

impl<T: ProofReplayStore> CredentialReplayAdmission for T {
    fn admit_source_credential(
        &self,
        key: &CredentialReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        self.check_and_insert_credential(key, expires_at)
    }
}

/// Production Policy uses the same shared admission backend as RPC request
/// proofs. It refuses single-process memory because credential consumption
/// must survive verifier restarts and coordinate concurrent instances.
struct GlobalCredentialReplay;

impl CredentialReplayAdmission for GlobalCredentialReplay {
    fn admit_source_credential(
        &self,
        key: &CredentialReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        let Some(store) = global_proof_replay_store() else {
            return ProofAdmissionResult::Failed;
        };
        if store.domain_guarantee() != ReplayDomainGuarantee::LinearizableSharedStore {
            return ProofAdmissionResult::Failed;
        }
        store.check_and_insert_credential(key, expires_at)
    }
}

/// Internal upstream evidence, not a browser DTO or verification boolean.
/// OAuth's FederateIssuer constructs it only after strict source-token
/// validation and both signatures over the exact challenge have verified.
struct PossessionEvidence {
    /// The verified client class this redemption belongs to (W1.2).
    client: String,
    /// The verified session-establishment flow (W1b); selects the
    /// nonce-free replay identity for device sources (D2).
    device_flow: bool,
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
    policy: Arc<PolicyManager>,
    enrollment: Arc<ServiceEnrollmentManifest>,
    /// The www staging client's fixed profile. Preserved verbatim (W1.2).
    profile: Profile,
    /// The configured TUI client's fixed profile, from the trusted-issuer
    /// `federate_tui_client` class. `None` = staging-only: the TUI class
    /// default-denies at admission (W1.1 invariant).
    tui_profile: Option<Profile>,
    /// Server-owned serving tuple from a complete trusted inventory/profile
    /// loader.
    serving_generation: [u8; 32],
    local_collision_inventory_id: [u8; 32],
    /// Public keys remain in Policy so issuance can reject cross-protocol
    /// proof-key reuse without persisting session rows or sharing root keys.
    collision_inventory: CollisionInventory,
}

impl Authorities {
    /// Resolve the per-client grant/profile table entry for an authenticated
    /// client selector. Only the hardcoded www pin and the configured TUI
    /// class resolve; anything else — including the TUI class when
    /// unconfigured — fails closed.
    fn profile_for(&self, client_id: &str) -> Result<&Profile> {
        if client_id == self.profile.client {
            return Ok(&self.profile);
        }
        match &self.tui_profile {
            Some(profile) if profile.client == client_id => Ok(profile),
            _ => Err(anyhow::anyhow!("unconfigured Federate client class")),
        }
    }
}

/// Default denies before any authority or DB access. The verified ATProto DID
/// is the human principal. OIDC `(issuer, sub)` remains login provenance; no
/// local UserStore row is required to establish the Hyprstream identity.
#[derive(Default)]
pub(super) struct AdmissionService {
    authority: Option<Authorities>,
    capacity: Option<Semaphore>,
    session_pool: Option<deadpool_postgres::Pool>,
    source_replay: Option<Arc<dyn CredentialReplayAdmission>>,
    pending: Mutex<BTreeMap<[u8; 32], PendingDecision>>,
}

/// Policy authority for a single protected Registry/Model request.
/// The production factory installs it with the same configured `AdmissionService`
/// supplies the sole account/tenant/grant owner; serving processes hold only
/// authenticated Policy clients, never database credentials.
#[allow(dead_code)]
pub(super) struct RequestUseReader {
    admission: Arc<AdmissionService>,
    pool: Option<deadpool_postgres::Pool>,
    capacity: Semaphore,
    clock_skew_secs: i64,
}

impl RequestUseReader {
    pub(super) fn new(
        admission: Arc<AdmissionService>,
        pool: Option<deadpool_postgres::Pool>,
    ) -> Self {
        Self {
            admission,
            pool,
            capacity: Semaphore::new(32),
            clock_skew_secs: 30,
        }
    }

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
        // The deferred consumer path is www-only BY DESIGN: the request-use
        // reader serves Registry/Model dispatch of `federate-session-v1`
        // credentials. A TUI-class session (`federate-tui-v1`) can never
        // satisfy this binding — its credential dispatches through normal
        // JWT verification and carries no registry|model consumer
        // authority (W1.3), so it is denied here even if presented.
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
        if self.pool.is_none() {
            let did = data.subject.as_str();
            let now = chrono::Utc::now().timestamp();
            ensure!(
                (did.starts_with("did:plc:") || did.starts_with("did:web:"))
                    && did.len() <= 255
                    && data.expires_at <= now.saturating_add(300)
                    && data.proof_epoch == 1,
                "invalid stateless Federate credential"
            );
            ensure!(
                data.scopes.iter().all(|scope| {
                    authority.profile.client_scopes.contains(scope)
                        && authority.profile.resource_scopes.contains(scope)
                }),
                "credential exceeds the Federate scope ceiling"
            );
            ensure!(
                data.scopes.iter().any(|text| {
                    Scope::parse(text).is_ok_and(|scope| {
                        scope.action == data.operation && scope.policy_resource() == data.resource
                    })
                }),
                "operation is outside the signed scope"
            );
            let _permit = self.capacity.try_acquire()?;
            ensure!(
                authority
                    .policy
                    .check_with_domain(did, &data.tenant, &data.resource, &data.operation)
                    .await,
                "current DID tenant grant denied"
            );
            return Ok(());
        }
        let _permit = self.capacity.try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), async {
            let pool = self
                .pool
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Federate session pool unavailable"))?;
            let mut client = pool.get().await?;
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
            let source_issuer = primary.source_identity().0;
            ensure!(
                primary.source_atproto_did() == session.subject,
                "session subject does not match verified ATProto DID"
            );
            authority
                .authorize_use(
                    session,
                    source_issuer,
                    primary.source_atproto_did(),
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

/// Fixed, authority-owned TUI scope ceiling: the TUI-consumer authority of
/// the second client class. Deliberately disjoint from the www staging
/// ceiling — a TUI-class session can never be granted registry|model
/// consumer authority, and a www session can never be granted TUI
/// session authority.
fn tui_scopes() -> BTreeSet<String> {
    BTreeSet::from([
        "connect:tui:session".to_owned(),
        "query:tui:session".to_owned(),
        "write:tui:session".to_owned(),
    ])
}

impl AdmissionService {
    /// Construct the exact configured-host profile from host-owned configuration.
    /// The policy factory must separately qualify the TLS pool and active
    /// profile generation before installing this authority. The www staging
    /// client and its four scope ceilings are preserved verbatim; the
    /// optional configured TUI class gets its own fixed table entry.
    pub(super) fn configured(
        policy: Arc<PolicyManager>,
        enrollment: Arc<ServiceEnrollmentManifest>,
        collision_inventory: CollisionInventory,
        session_pool: Option<deadpool_postgres::Pool>,
        host: crate::services::oauth::federate_source::FederateHost,
        tui_client: Option<&str>,
    ) -> Self {
        let inventory_id = collision_inventory.id();
        let scopes = BTreeSet::from([
            "infer:model:qwen2.5-0.5b-instruct:main".to_owned(),
            "query:model:Status".to_owned(),
            "query:registry:qwen2.5-0.5b-instruct".to_owned(),
            "write:model:Load".to_owned(),
        ]);
        let tui_profile = tui_client.map(|client| Profile {
            issuer: crate::services::oauth::federate_source::ISSUER.into(),
            host: host.as_str().into(),
            client: client.into(),
            resource: host.as_str().into(),
            client_scopes: tui_scopes(),
            resource_scopes: tui_scopes(),
        });
        Self {
            authority: Some(Authorities {
                policy,
                enrollment,
                profile: Profile {
                    issuer: crate::services::oauth::federate_source::ISSUER.into(),
                    host: host.as_str().into(),
                    client: crate::services::oauth::federate_source::CLIENT.into(),
                    resource: host.as_str().into(),
                    client_scopes: scopes.clone(),
                    resource_scopes: scopes,
                },
                tui_profile,
                serving_generation: inventory_id,
                local_collision_inventory_id: inventory_id,
                collision_inventory,
            }),
            capacity: Some(Semaphore::new(32)),
            session_pool,
            source_replay: Some(Arc::new(GlobalCredentialReplay)),
            pending: Mutex::default(),
        }
    }
}

/// D2 replay identity: derived ONLY from immutable authenticated fields —
/// (issuer, client class) for the namespace and (jti, token_hash) for the
/// credential identity. The commitment nonce (and any login_random/code/key
/// material) is deliberately NOT part of the key: exactly one consumption of
/// a signed source token must succeed regardless of client-supplied
/// commitment inputs, and the device-flow variant (W1b) does not pin the
/// nonce claim at all. The commitment/possession check remains a separate
/// gate.
/// The pre-D2 replay identity (nonce included). Retained ONLY for the
/// upgrade transition: a source token consumed by a pre-D2 instance is
/// recorded under this key, and a still-valid token must not be consumed
/// again by a post-D2 instance (or vice versa) in a mixed fleet.
fn legacy_source_credential_replay_key(source: &Source, client_id: &str) -> CredentialReplayKey {
    let current = source_credential_replay_key(source, client_id);
    let mut digest = Sha256::new();
    digest.update(b"hyprstream-source-credential-identity-v1 ");
    for field in [
        source.jti.as_bytes(),
        source.nonce.as_bytes(),
        &source.token_hash,
    ] {
        digest.update((field.len() as u64).to_be_bytes());
        digest.update(field);
    }
    CredentialReplayKey {
        issuer_client_namespace: current.issuer_client_namespace,
        credential_identity: digest.finalize().into(),
    }
}

fn source_credential_replay_key(source: &Source, client_id: &str) -> CredentialReplayKey {
    fn digest(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(domain);
        for field in fields {
            digest.update((field.len() as u64).to_be_bytes());
            digest.update(field);
        }
        digest.finalize().into()
    }

    CredentialReplayKey {
        issuer_client_namespace: digest(
            b"hyprstream-source-credential-namespace-v1\0",
            &[source.issuer.as_bytes(), client_id.as_bytes()],
        ),
        credential_identity: digest(
            b"hyprstream-source-credential-identity-v1\0",
            &[source.jti.as_bytes(), &source.token_hash],
        ),
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
            if rule
                .get(subject_index)
                .is_some_and(|subject| subject == did)
            {
                if let Some(tenant) = rule
                    .get(tenant_index)
                    .filter(|tenant| !tenant.is_empty() && *tenant != "*")
                {
                    // D4: a malformed Policy tenant rejects the decision
                    // outright — the same canonical syntax the HTTP
                    // authorization-domain middleware enforces.
                    ensure!(
                        crate::auth::valid_tenant_component(tenant),
                        "malformed Policy tenant in a federate grant"
                    );
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
    /// The per-client grant/profile decision. `client_id` is the VERIFIED
    /// client class selector relayed by OAuth (never a browser-chosen
    /// value): it selects the table entry whose issuer/host/resource and
    /// scope ceilings govern this admission.
    async fn decision(
        &self,
        client_id: &str,
        source: &Source,
        requested: &[String],
    ) -> Result<Decision> {
        let profile = self.profile_for(client_id)?;
        ensure!(source.issuer == profile.issuer, "issuer mismatch");
        canonical_requested(requested)?;
        let did = &source.atproto_did;
        ensure!(
            (did.starts_with("did:plc:") || did.starts_with("did:web:")) && did.len() <= 255,
            "invalid verified ATProto DID"
        );
        // Internal join identifier only; the verified DID remains the
        // authorization principal and no local account is created or read.
        let account_id = federate_account_id(&profile.host, did);
        let policies = self.policy.get_policy().await;
        let domain_groups = self.policy.get_domain_grouping_policy().await;
        let tenants = candidate_tenants(did, &policies, &domain_groups)?;
        let mut matching = Vec::new();
        for tenant in tenants {
            let mut scopes = Vec::new();
            for text in requested {
                if !profile.client_scopes.contains(text) || !profile.resource_scopes.contains(text)
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
            &profile.issuer,
            &profile.host,
            &profile.client,
            &profile.resource,
            &profile.client_scopes,
            &profile.resource_scopes,
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
    /// and its verified source provenance. `session` and `source` must come from the
    /// authenticated full-record lookup after the request's host JWT, holder
    /// proof, and v16 replay admission have been verified. This source slice has
    /// no constructor/route that can create that evidence; callers must not
    /// deserialize either structure from a client request.
    async fn authorize_use(
        &self,
        session: &Session,
        source_issuer: &str,
        source_atproto_did: &str,
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
        let did = &session.subject;
        ensure!(
            (did.starts_with("did:plc:") || did.starts_with("did:web:")) && did.len() <= 255,
            "session subject is not an ATProto DID"
        );
        ensure!(
            source_atproto_did == did,
            "source DID does not match session principal"
        );
        ensure!(
            session.account_id == federate_account_id(&self.profile.host, did),
            "session account binding changed"
        );
        // Current Policy grants are authoritative for both membership and
        // suspension. These reads are fresh at each protected operation.
        ensure!(
            self.policy
                .check_with_domain(did, &session.tenant, resource, operation)
                .await,
            "current DID tenant grant denied"
        );
        Ok(())
    }
}

fn federate_account_id(host: &str, did: &str) -> String {
    let name = format!("hyprstream-federate-account-v1\0{host}\0{did}");
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, name.as_bytes()).to_string()
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
        client_id: &str,
        source: Source,
        device_flow: bool,
        requested: Vec<String>,
        challenge_id: [u8; 32],
        created_at: i64,
        ed_public: [u8; 32],
        pq_public: Vec<u8>,
    ) -> Result<PreparedDecision> {
        let authority = self.authority(ctx)?;
        // Fail closed on the client selector BEFORE any time, replay or
        // capacity work: only a configured table entry resolves. When the
        // TUI class is unconfigured this is the admission-side default-deny.
        authority.profile_for(client_id)?;
        let now = chrono::Utc::now().timestamp();
        ensure!(challenge_id != [0; 32], "invalid challenge ID");
        ensure!(
            authority
                .collision_inventory
                .permits(&ed_public, &pq_public),
            "Federate proof key collides with a trusted protocol key or has an invalid PQ length"
        );
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
        let decision = self.prepare(ctx, client_id, &source, &requested).await?;
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
                client: client_id.to_owned(),
                device_flow,
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
        client_id: &str,
        source: Source,
        device_flow: bool,
        challenge_id: [u8; 32],
        created_at: i64,
        expires_at: i64,
        ed_public: [u8; 32],
        pq_public: Vec<u8>,
    ) -> Result<Session> {
        let authority = self.authority(ctx)?;
        let pending = self
            .pending
            .lock()
            .await
            .remove(&handle)
            .ok_or_else(|| anyhow::anyhow!("unknown or consumed challenge"))?;
        // The committing client class must resolve to a configured table
        // entry AND equal the class the pending decision was prepared under.
        // The handle is already removed above, so any mismatch consumes it.
        authority.profile_for(client_id)?;
        let now = chrono::Utc::now().timestamp();
        ensure!(
            now < pending.expires_at
                && client_id == pending.client
                && device_flow == pending.device_flow
                && source == pending.source
                && challenge_id == pending.challenge_id
                && created_at == pending.created_at
                && expires_at == pending.expires_at
                && ed_public == pending.ed_public
                && pq_public == pending.pq_public,
            "challenge/source binding changed"
        );
        ensure!(pq_public.len() == 1952, "invalid PQ public key");
        ensure!(
            authority
                .collision_inventory
                .permits(&ed_public, &pq_public),
            "Federate proof key collides with a trusted protocol key"
        );
        if let Some(pool) = &self.session_pool {
            let mut client = tokio::time::timeout(Duration::from_secs(2), pool.get()).await??;
            return self
                .redeem(
                    ctx,
                    &mut client,
                    PossessionEvidence {
                        client: client_id.to_owned(),
                        device_flow: pending.device_flow,
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
                .await;
        }
        let profile = authority.profile_for(client_id)?;
        let current = tokio::time::timeout(
            Duration::from_secs(2),
            authority.decision(client_id, &source, &pending.requested),
        )
        .await??;
        ensure!(current == pending.decision, "challenge decision changed");
        let session_expires_at = pending.source.expires_at.min(now + 300);
        ensure!(session_expires_at > now, "source credential expired");
        self.consume_source_credential(&source, &profile.client, pending.device_flow)?;
        Ok(Session {
            host: profile.host.clone(),
            sid: uuid::Uuid::new_v4().to_string(),
            account_id: current.account_id,
            subject: current.subject,
            tenant: current.tenant,
            client_id: profile.client.clone(),
            resource: profile.resource.clone(),
            scopes: current.scopes,
            grant_revision: current.revision,
            ed_public,
            pq_public,
            generation: current.generation,
            collision_inventory_id: current.collision_inventory_id,
            proof_epoch: 1,
            expires_at: session_expires_at,
        })
    }

    fn authority(&self, ctx: &EnvelopeContext) -> Result<&Authorities> {
        let authority = self
            .authority
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Federate admission disabled"))?;
        oauth_caller(ctx, &authority.enrollment)?;
        Ok(authority)
    }

    fn consume_source_credential(
        &self,
        source: &Source,
        client_id: &str,
        device_flow: bool,
    ) -> Result<()> {
        let expires_at = u64::try_from(source.expires_at)?;
        let now = u64::try_from(chrono::Utc::now().timestamp())?;
        ensure!(expires_at > now, "source credential expired");
        let replay = self
            .source_replay
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("source credential replay admission disabled"))?;
        // D2 replay identity: exactly one consumption of a signed source
        // token, independent of client-supplied commitment inputs.
        let key = source_credential_replay_key(source, client_id);
        // Code flow (D2 transition): consume under BOTH the legacy
        // (nonce-bearing) and the current (nonce-free) identity. Its nonce
        // is an issuer-signed claim, so the legacy key is token-stable and
        // a mixed-version fleet cannot consume one token twice in either
        // direction. Legacy first: it serializes mixed-version consumers.
        // A failure between the two writes only ever leaves the token
        // recorded under the legacy key of the SAME nonce, which any retry
        // of the same token hits (Replayed).
        //
        // Device flow (W1b): consume under the CURRENT identity ONLY. The
        // device commitment nonce is client-controlled (login_random,
        // device_code, keys), never issuer-signed, so a legacy write would
        // let a partially-failed consumption (legacy recorded, current
        // write failed) be retried under a DIFFERENT device commitment —
        // a fresh legacy key and an absent current key — and commit twice.
        // Single-key consumption removes the window entirely: the write
        // either lands (every retry is Replayed) or fails (nothing
        // recorded; the retry re-attempts the same key). No pre-W1b
        // instance can hold a device-flow record (the flow did not exist),
        // so no legacy transition applies. The code-flow legacy read can
        // be dropped once every pre-D2 token has expired (source tokens
        // live at most 300s past issue).
        let legacy = (!device_flow).then(|| legacy_source_credential_replay_key(source, client_id));
        let candidates: Vec<&CredentialReplayKey> = match &legacy {
            Some(legacy) => vec![legacy, &key],
            None => vec![&key],
        };
        for candidate in candidates {
            match replay.admit_source_credential(candidate, expires_at) {
                ProofAdmissionResult::Admitted => {}
                ProofAdmissionResult::Replayed => {
                    return Err(anyhow::anyhow!("source credential already consumed"));
                }
                ProofAdmissionResult::Failed => {
                    return Err(anyhow::anyhow!(
                        "source credential replay admission unavailable"
                    ));
                }
            }
        }
        Ok(())
    }

    async fn prepare(
        &self,
        ctx: &EnvelopeContext,
        client_id: &str,
        source: &Source,
        requested: &[String],
    ) -> Result<Decision> {
        let a = self.authority(ctx)?;
        let _permit = self
            .capacity
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        tokio::time::timeout(
            Duration::from_secs(2),
            a.decision(client_id, source, requested),
        )
        .await?
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
            let profile = a.profile_for(&evidence.client)?;
            let current = a
                .decision(&evidence.client, &evidence.source, &evidence.requested)
                .await?;
            ensure!(current == evidence.challenge, "challenge decision changed");
            self.consume_source_credential(
                &evidence.source,
                &profile.client,
                evidence.device_flow,
            )?;
            let session = PendingSession {
                host: profile.host.clone(),
                sid: evidence.sid,
                account_id: current.account_id,
                subject: current.subject,
                tenant: current.tenant,
                client_id: profile.client.clone(),
                resource: profile.resource.clone(),
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
                    current.source_atproto_did(),
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
