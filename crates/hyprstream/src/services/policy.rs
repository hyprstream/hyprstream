//! Policy service for authorization checks over ZMQ
//!
//! Wraps PolicyManager and exposes it as a RequestService.
//! Handlers are async and use `.await` directly (compatible with single-threaded runtime).

use async_trait::async_trait;
use crate::auth::PolicyManager;
use crate::auth::policy_templates;
use crate::services::{EnvelopeContext, RequestService};
use crate::services::generated::policy_client::{
    ErrorInfo, PolicyHandler, PolicyResponseVariant, TokenInfo, ScopeList,
    PolicyCheck, IssueToken, IssueTokenProfile,
    ApplyTemplate, ApplyDraft, RollbackPolicy, GetHistory, GetDiff,
    PolicyInfo, PolicyRule, Grouping,
    PolicyHistory, PolicyHistoryEntry, DraftStatus,
    AddGrouping, RemoveGrouping, SetBranchVisibility,
    RegisterEventPrefix, SubscribeEventPrefix, GetPendingSubscribers, DepositWrappedKeys,
    EventPrefixAccess, PendingSubscribers,
    ResolveServiceKey, RegisterServiceKey, ServiceKeyCandidate, ServiceKeyResponse,
    RefreshServiceTokenRequest, ExchangeWit, ExchangeDelegated,
    RevokeCredential, CheckCredentialRevocation,
    RegisterSession, RevokeSession, CheckSession,
    dispatch_policy, serialize_response,
};
use anyhow::{anyhow, Result};
use git2db::{Git2DB, RepoId};
use hyprstream_pds::repo_authority::is_path_form_did_web;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::transport::TransportConfig;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

/// Evaluate a policy check on behalf of an already-verified upstream caller.
///
/// The compatibility `PolicyCheck.subject/domain` fields are never identity
/// evidence. A service that mediates a user request relays the original bearer
/// in its signed envelope so PolicyService can independently verify both the
/// user and the hosted-account tenant.
pub(crate) async fn check_with_verified_bearer(
    client: &crate::services::PolicyClient,
    request: &PolicyCheck,
    bearer: Option<&str>,
    upstream_subject: &Subject,
) -> Result<bool> {
    match bearer {
        Some(token) => client
            .clone()
            .with_delegated_bearer(token.to_owned())
            .check(request)
            .await,
        None => {
            anyhow::ensure!(
                !upstream_subject.is_federated()
                    && upstream_subject
                        .name()
                        .is_some_and(|name| { name == "system" || name.starts_with("service:") }),
                "service-mediated user policy check requires a verified upstream bearer"
            );
            client.check(request).await
        }
    }
}

// ============================================================================
// PolicyService (server-side)
// ============================================================================

/// Policy service that wraps PolicyManager.
/// Receives policy check requests over ZMQ and delegates to PolicyManager.
/// Per-prefix event state (blind key relay).
///
/// PolicyService stores opaque wrapped key blobs — it never sees plaintext
/// group keys or wrap keys. The publisher wraps directly against subscriber
/// pubkeys via DH.
struct EventPrefixState {
    owner: String,
    publisher_pubkey: [u8; 32],
    schema: String,
    /// Subscriber ephemeral pubkeys, keyed by Blake3 hash of pubkey.
    subscriber_pubkeys: HashMap<[u8; 32], [u8; 32]>,
    /// Opaque wrapped key blobs deposited by the publisher, keyed by subscriber pubkey hash.
    wrapped_keys: HashMap<[u8; 32], Vec<u8>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct EventPrefixKey { tenant: String, prefix: String }
impl EventPrefixKey {
    fn new(tenant: String, prefix: &str) -> Self { Self { tenant, prefix: prefix.to_owned() } }
}
fn cross_tenant_prefix_shadow(a: &EventPrefixKey, b: &EventPrefixKey) -> bool {
    a.tenant != b.tenant && a.prefix != b.prefix
        && (a.prefix.starts_with(&b.prefix) || b.prefix.starts_with(&a.prefix))
}
#[derive(Debug, Eq, PartialEq)]
enum EventPrefixRegistrationError { OwnedByAnotherSubject, CrossTenantShadow }
fn validate_event_prefix_registration(
    prefixes: &HashMap<EventPrefixKey, EventPrefixState>, key: &EventPrefixKey, owner: &str,
) -> Result<(), EventPrefixRegistrationError> {
    if prefixes.get(key).is_some_and(|state| state.owner != owner) {
        return Err(EventPrefixRegistrationError::OwnedByAnotherSubject);
    }
    if prefixes.keys().any(|existing| cross_tenant_prefix_shadow(existing, key)) {
        return Err(EventPrefixRegistrationError::CrossTenantShadow);
    }
    Ok(())
}

/// Default revocation-publication horizon: covers the hard 30-day
/// service-JWT renewal clamp with margin. Production construction derives
/// the horizon from the configured issuance maxima instead — see
/// `PolicyService::with_revocation_max_ttl_secs`.
const DEFAULT_REVOCATION_MAX_TTL_SECS: i64 = 45 * 24 * 3600;

pub struct PolicyService {
    // Business logic
    policy_manager: Arc<PolicyManager>,
    signing_key: Arc<SigningKey>,
    /// Purpose-derived key for JWT token signing (isolated from envelope signing)
    jwt_signing_key: SigningKey,
    token_config: crate::config::TokenConfig,
    /// Supported scopes computed once at construction from ServiceFactory inventory
    supported_scopes: Vec<String>,
    /// Shared git2db registry for git operations on .registry repo
    git2db: Arc<RwLock<Git2DB>>,
    /// RepoId of the .registry self-tracked entry
    registry_repo_id: RepoId,
    /// Default audience for issued tokens (OAuth issuer URL, shared instance identifier).
    /// Used when IssueToken.audience is empty, ensuring all tokens get an `aud` claim.
    default_audience: Option<String>,
    /// JWT key source for verifying JWTs (local and federated).
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    // Infrastructure (for Spawnable)
    transport: TransportConfig,
    /// Event prefix state for secure event transport (Phase 7).
    /// PolicyService is a blind relay — stores opaque wrapped blobs, never plaintext keys.
    event_prefixes: RwLock<HashMap<EventPrefixKey, EventPrefixState>>,
    /// ES256 (P-256) key rotation store for DPoP/atproto interop.
    es256_key_store: Option<Arc<crate::auth::Es256SigningKeyStore>>,
    /// ML-DSA-65 key rotation store for PQ-hybrid composite token issuance.
    ml_dsa_key_store: Option<Arc<crate::auth::MlDsaSigningKeyStore>>,
    /// Authority-owned enrollment lookup for MAC-labeled credentials.
    token_clearance_resolver: Arc<
        dyn Fn(&str) -> Option<hyprstream_rpc::auth::mac::SecurityLabel> + Send + Sync,
    >,
    /// Maximum retention horizon the revocation authority accepts for a
    /// published entry (`expires_at - now`). Derived at construction from the
    /// configured issuance maxima so no issuable credential can outlive its
    /// revocability — see `with_revocation_max_ttl_secs`.
    revocation_max_ttl_secs: i64,
    /// Fail-closed authority for derived `AsOriginator` delegation edges
    /// (`exchangeDelegated`). `None` (uninstalled) ⇒ every delegation DENIES;
    /// WS-E installs the reviewed `DispatchCallManifest` implementation. See
    /// [`DelegationEdgeAuthorizer`].
    delegation_edge_authorizer: Option<Arc<dyn DelegationEdgeAuthorizer>>,
    /// Optional injected service-enrollment manifest override. `None` uses the
    /// process-global manifest (`global_service_enrollment`). Set in isolated
    /// fixtures so a test can supply its own enrollment without touching the
    /// set-once process global. See [`Self::enrollment`].
    enrollment_manifest: Option<Arc<crate::auth::service_enrollment::ServiceEnrollmentManifest>>,
    /// Fail-closed authoritative primary-enrollment resolver for user/classical
    /// source credentials (frozen A §5/T1). `None` (uninstalled) ⇒ a user
    /// primary confirmation DENIES; WS-C installs the real resolver. See
    /// [`PrimaryEnrollmentResolver`].
    primary_enrollment_resolver: Option<Arc<dyn PrimaryEnrollmentResolver>>,
}

impl PolicyService {
    /// Create a new policy service with infrastructure
    pub fn new(
        policy_manager: Arc<PolicyManager>,
        signing_key: Arc<SigningKey>,
        token_config: crate::config::TokenConfig,
        git2db: Arc<RwLock<Git2DB>>,
        transport: TransportConfig,
    ) -> Self {
        let registry_repo_id = RepoId::from_uuid(git2db::registry::registry_self_uuid());
        let jwt_signing_key = hyprstream_rpc::node_identity::derive_purpose_key(&signing_key, "hyprstream-jwt-v1");
        Self {
            policy_manager,
            signing_key,
            jwt_signing_key,
            token_config,
            supported_scopes: compute_supported_scopes(),
            git2db,
            registry_repo_id,
            default_audience: None,
            jwt_key_source: None,
            transport,
            event_prefixes: RwLock::new(HashMap::new()),
            es256_key_store: None,
            ml_dsa_key_store: None,
            token_clearance_resolver: Arc::new(|subject| {
                let policy = crate::mac::compiled_policy()?;
                policy.clearance_for(subject)
            }),
            revocation_max_ttl_secs: DEFAULT_REVOCATION_MAX_TTL_SECS,
            delegation_edge_authorizer: None,
            enrollment_manifest: None,
            primary_enrollment_resolver: None,
        }
    }

    /// Install the fail-closed authoritative primary-enrollment resolver (WS-C's
    /// enrollment store) used to confirm a user/classical source credential's
    /// `cnf`. Without it, user primaries deny.
    pub fn with_primary_enrollment_resolver(
        mut self,
        resolver: Arc<dyn PrimaryEnrollmentResolver>,
    ) -> Self {
        self.primary_enrollment_resolver = Some(resolver);
        self
    }

    /// Install the fail-closed delegation-edge authorizer (WS-E's reviewed
    /// `DispatchCallManifest`). Without it, `exchangeDelegated` denies every
    /// edge — the end-state default, never a bypass.
    pub fn with_delegation_edge_authorizer(
        mut self,
        authorizer: Arc<dyn DelegationEdgeAuthorizer>,
    ) -> Self {
        self.delegation_edge_authorizer = Some(authorizer);
        self
    }

    /// Override the service-enrollment manifest for an isolated fixture (avoids
    /// the set-once process global). **Test-only** — production is single
    /// authority (the process global); there is no production manifest-injection
    /// path, so this setter is compiled only under `cfg(test)`.
    #[cfg(test)]
    pub fn with_enrollment_manifest(
        mut self,
        manifest: Arc<crate::auth::service_enrollment::ServiceEnrollmentManifest>,
    ) -> Self {
        self.enrollment_manifest = Some(manifest);
        self
    }

    /// The effective service-enrollment manifest: the injected override if set,
    /// else the process-global manifest.
    fn enrollment(
        &self,
    ) -> Option<Arc<crate::auth::service_enrollment::ServiceEnrollmentManifest>> {
        self.enrollment_manifest.clone().or_else(|| {
            crate::auth::service_enrollment::global_service_enrollment().cloned()
        })
    }

    /// Set the revocation-publication horizon. The service factory computes
    /// this as max(every configured issuance maximum) plus a clock-skew
    /// margin, so an operator raising token TTLs automatically raises the
    /// authority's retention horizon — issuance and revocation horizons are
    /// bound by construction rather than by a shared default.
    pub fn with_revocation_max_ttl_secs(mut self, secs: i64) -> Self {
        self.revocation_max_ttl_secs = secs;
        self
    }

    /// Set the default audience for issued tokens (typically the OAuth issuer URL).
    pub fn with_default_audience(mut self, audience: String) -> Self {
        self.default_audience = Some(audience);
        self
    }

    /// Override enrollment lookup for an isolated service fixture.
    pub fn with_token_clearance_resolver(
        mut self,
        resolver: Arc<
            dyn Fn(&str) -> Option<hyprstream_rpc::auth::mac::SecurityLabel> + Send + Sync,
        >,
    ) -> Self {
        self.token_clearance_resolver = resolver;
        self
    }

    /// Sign a token with the mandatory hybrid suite (Fu3/#677).
    ///
    /// The composite PQ signature (EdDSA + ML-DSA-65) is used under a Hybrid
    /// policy. If no ML-DSA-65 signing key is provisioned, this returns `Err`
    /// rather than silently minting a classical-only token — mirroring
    /// [`crate::mac::audit::CoseAuditSigner`],
    /// which the S7 audit path already gates the same way. Previously this seam
    /// picked composite-vs-classical by *keystore state*, so a Hybrid node with
    /// an empty/rotating ML-DSA store quietly downgraded minted tokens.
    async fn sign_token(
        &self,
        claims: &hyprstream_rpc::auth::Claims,
        is_service: bool,
    ) -> Result<String> {
        // Sign against the SAME configured local authority the source
        // verification path resolves keys from: the policy service's own
        // `jwt_key_source` composite ledger. In production the factory always
        // installs a `ClusterKeySource` (whose ledger defaults to the
        // process-global authority, so behaviour is identical); an isolated
        // fixture supplies its own via `with_composite_key_set`, keeping
        // verification and mint on ONE exact issuer-scoped ledger. Fail closed
        // if no key source is configured — never fall back to the global
        // authority, which would reopen cross-issuer ambiguity.
        let key_source = self
            .jwt_key_source
            .as_ref()
            .ok_or_else(|| anyhow!("no configured key source for token signing"))?;
        let snapshot = key_source
            .composite_key_set()
            .mint_snapshot()
            .map_err(|error| anyhow!("composite authority unavailable: {error}"))?;
        let signing = snapshot
            .active_signing_pair(hyprstream_rpc::auth::CompositePairRole::Policy)
            .and_then(hyprstream_rpc::auth::CompositeKeyPair::signing_keys);
        let Some((ml_key, ed_key)) = signing else {
            warn!("no authorized active Policy composite pair; refusing to mint");
            return Err(anyhow!("hybrid token signing pair not provisioned"));
        };
        Ok(if is_service {
            crate::auth::jwt::encode_composite_service_jwt(
                claims, &ml_key, &ed_key,
            )
        } else {
            crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
                claims, &ml_key, &ed_key,
            )
        })
    }

    /// Set the JWT key source for verifying JWTs (local and federated).
    pub fn with_jwt_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    ) -> Self {
        self.jwt_key_source = Some(src);
        self
    }

    /// Resolve the workload-session disposition for a service-credential renewal
    /// (v16 §3.3), as a pure function of the canonical session registry and the
    /// authoritative family policy. The renewal handler applies the result to
    /// the renewed claims; extracting it keeps the security-critical
    /// check-BEFORE-narrowing ordering testable without the signing/persistence
    /// boundary.
    ///
    /// - A carried session (`old_wsid = Some`) is checked for revocation FIRST,
    ///   independent of `family_policy` — a revoked/expired/unresolvable session
    ///   DENIES, so removing the family policy can never launder a revoked family
    ///   into an unsessioned renewal. A live session is re-stamped only while the
    ///   family remains enrolled; a narrowed (policy-removed) family drops it.
    /// - No carried session (`old_wsid = None`) creates one only for a family
    ///   EXPLICITLY enrolled (`family_policy == Some(true)`); a legacy-default or
    ///   disabled family carries none.
    async fn resolve_renewal_workload_session(
        &self,
        issuer: &str,
        subject: &str,
        tenant: &str,
        now: i64,
        family_policy: Option<bool>,
        old_wsid: Option<&str>,
    ) -> RenewalWorkloadSession {
        // No manifest at all: legacy continuity (an unenrolled deployment keeps
        // renewing). An enrolled family with `workload_session = false` is a
        // deliberate policy, NOT legacy — `family_policy` distinguishes them.
        let family_allowed = family_policy.unwrap_or(true);
        match (old_wsid, family_allowed) {
            (Some(wsid), allowed) => {
                // A carried workload session must be ACTIVE before renewal,
                // regardless of current enrollment policy — the revocation
                // check runs BEFORE the narrowing branch below.
                let session_key =
                    hyprstream_rpc::auth::SessionKey::workload(issuer.to_owned(), wsid.to_owned());
                let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
                    return RenewalWorkloadSession::Deny {
                        code: "UNAVAILABLE",
                        message: "session registry is not initialized".to_owned(),
                    };
                };
                if registry.is_revoked(&session_key).await {
                    return RenewalWorkloadSession::Deny {
                        code: "SESSION_REVOKED",
                        message: "workload session is revoked or expired; re-bootstrap the service credential".to_owned(),
                    };
                }
                // Only a still-enrolled family carries the session forward;
                // `!allowed` is deliberate narrowing of a LIVE family — the
                // session ID does not survive this renewal (the session itself
                // expires naturally; no credential carries it forward).
                RenewalWorkloadSession::Stamp(allowed.then(|| wsid.to_owned()))
            }
            (None, true) => {
                // First online renewal of an ENROLLED family creates the
                // workload session with the canonical registry; a legacy-default
                // family (`family_policy == None`) manufactures no session.
                if family_policy != Some(true) {
                    return RenewalWorkloadSession::Stamp(None);
                }
                use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
                use rand::RngCore;
                let mut id_bytes = [0u8; 32];
                rand::rngs::OsRng.fill_bytes(&mut id_bytes);
                let wsid = URL_SAFE_NO_PAD.encode(id_bytes);
                let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
                    return RenewalWorkloadSession::Deny {
                        code: "UNAVAILABLE",
                        message: "session registry is not initialized".to_owned(),
                    };
                };
                let session_state = hyprstream_rpc::auth::SessionState {
                    subject: subject.to_owned(),
                    tenant: tenant.to_owned(),
                    kind: hyprstream_rpc::auth::SessionKind::Workload,
                    created_at: now,
                    expires_at: now + self.revocation_max_ttl_secs,
                    status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                    clearance_epoch: 0,
                };
                if let Err(e) = registry
                    .register_session(
                        hyprstream_rpc::auth::SessionKey::workload(issuer.to_owned(), wsid.clone()),
                        session_state,
                    )
                    .await
                {
                    return RenewalWorkloadSession::Deny {
                        code: "UNAVAILABLE",
                        message: format!("workload session registration failed: {e}"),
                    };
                }
                RenewalWorkloadSession::Stamp(Some(wsid))
            }
            (None, false) => RenewalWorkloadSession::Stamp(None),
        }
    }

    /// Attach the ES256 (P-256) key rotation store.
    pub fn with_es256_key_store(mut self, store: Arc<crate::auth::Es256SigningKeyStore>) -> Self {
        self.es256_key_store = Some(store);
        self
    }

    /// Attach the ML-DSA-65 key rotation store for composite token issuance.
    pub fn with_ml_dsa_key_store(mut self, store: Arc<crate::auth::MlDsaSigningKeyStore>) -> Self {
        self.ml_dsa_key_store = Some(store);
        self
    }

    /// Stage policies/ and commit with the given message via git2db.
    /// Returns the commit OID as a hex string.
    async fn stage_and_commit_policies(&self, message: &str) -> Result<String> {
        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;

        // Stage the policies/ directory
        handle.staging().add_all().await
            .map_err(|e| anyhow!("Failed to stage policy files: {}", e))?;

        // Commit
        let oid = handle.commit(message).await
            .map_err(|e| anyhow!("Failed to commit policy: {}", e))?;

        Ok(oid.to_string())
    }

    /// Select the Casbin domain for PolicyService itself.
    ///
    /// Tenant-bearing callers stay in their authority-verified tenant. The
    /// policy/OAuth authority is also a global, cross-tenant subsystem: a
    /// tenantless, authenticated `service:*` principal (or the PolicyService
    /// authority key itself during bootstrap) therefore remains in the global
    /// policy domain. Ordinary tenantless identities still fail closed.
    fn request_domain(&self, ctx: &EnvelopeContext) -> Result<String> {
        if ctx.verified_tenant().is_some() {
            return ctx.domain();
        }

        let subject = ctx.subject();
        let is_global_service = subject
            .name()
            .is_some_and(|name| name.starts_with("service:"))
            && !subject.is_federated();
        let is_policy_authority =
            ctx.cnf == self.signing_key.verifying_key().to_bytes();
        anyhow::ensure!(
            is_global_service || is_policy_authority,
            "authorization denied: no verified tenant domain"
        );
        Ok("*".to_owned())
    }
}

/// Collect all supported scopes from compile-time schema metadata
/// via the ServiceFactory inventory. No hardcoded service imports needed.
///
/// Scopes use flat format `action:service:*` — coarse-grained per OAuth convention.
/// Fine-grained authorization is handled by Casbin resource patterns.
fn compute_supported_scopes() -> Vec<String> {
    use hyprstream_service::list_factories;

    let mut scopes = std::collections::BTreeSet::new();

    for factory in list_factories() {
        if let Some(metadata_fn) = factory.metadata {
            let (service_name, methods) = metadata_fn();
            for method in methods {
                // S3 (#547): scope is mandatory at build time, so `method.scope` is
                // non-empty for every enforced method. An empty scope here can only be a
                // `$scopeExempt` method (e.g. the authz check itself) — it requires no
                // grant, so it contributes no advertised scope. No silent "query" default.
                if method.scope.is_empty() {
                    continue;
                }
                scopes.insert(format!("{}:{}:*", method.scope, service_name));
            }
        }
    }

    scopes.into_iter().collect()
}

// ============================================================================
// PolicyHandler implementation (generated trait)
// ============================================================================

/// Validate an event prefix string.
///
/// Rejects empty strings, path traversal (`..`), and Casbin metacharacters (`*`, `#`).
/// Only allows alphanumeric, `.`, `-`, and `_`.
fn validate_event_prefix(prefix: &str) -> Result<(), String> {
    if prefix.is_empty() {
        return Err("prefix must not be empty".to_owned());
    }
    if prefix.len() > 128 {
        return Err("prefix exceeds 128 characters".to_owned());
    }
    if prefix.contains("..") {
        return Err("prefix must not contain '..'".to_owned());
    }
    if !prefix
        .chars()
        .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_')
    {
        return Err(
            "prefix may only contain alphanumeric, '.', '-', '_' characters".to_owned(),
        );
    }
    Ok(())
}

/// Build the rotation-safe wire projection of a service's published key set.
///
/// The scalar fields are a transition projection for one-key deployments.
/// They are empty during overlap, where a positional singleton is unsafe.
fn published_service_key_response(
    trust: &hyprstream_service::TrustStore,
    service_name: &str,
) -> Result<ServiceKeyResponse> {
    let keys = trust.published_keys_for_scope(service_name);
    if keys.is_empty() {
        anyhow::bail!("service key '{service_name}' not registered");
    }
    let singleton = (keys.len() == 1).then(|| &keys[0]);
    Ok(ServiceKeyResponse {
        verifying_key: singleton.map(|entry| entry.verifying_key.to_bytes().to_vec()).unwrap_or_default(),
        service_jwt: singleton.and_then(|entry| entry.attestation.jwt.clone()),
        keys: keys.into_iter().map(|entry| ServiceKeyCandidate {
            key_id: entry.key_id,
            verifying_key: entry.verifying_key.to_bytes().to_vec(),
            service_jwt: entry.attestation.jwt,
            not_after: entry.attestation.expires_at,
        }).collect(),
    })
}

/// Confirmation material is mandatory: an absent or malformed `cnf.jwk` must
/// never turn a valid CA token into authority for arbitrary key material.
/// Claims for a renewed service JWT.
///
/// `aud` is stamped alongside `iss` — the same shape the provisioning path
/// mints — because strict composite audience validation rejects an aud-less
/// token on every dispatch, and the on-disk reuse predicate treats a token
/// that does not bind the local issuer URL in both claims as stale.
fn renewed_service_claims(
    subject: String,
    now: i64,
    expires_at: i64,
    issuer: &str,
    tenant: String,
    cnf_key: &[u8; 32],
) -> hyprstream_rpc::auth::Claims {
    let mut claims = hyprstream_rpc::auth::Claims::new(subject, now, expires_at)
        .with_tenant(tenant)
        .with_cnf_jwk(cnf_key);
    if !issuer.is_empty() {
        claims = claims
            .with_issuer(issuer.to_owned())
            .with_audience(Some(issuer.to_owned()));
    }
    claims
}

/// Verify a service JWT presented to `registerServiceKey`.
///
/// Hybrid (`ML-DSA-65-Ed25519`) tokens resolve their composite kid through
/// the same key-source path the dispatch plane uses, with the CA's own
/// derived pair as the authoritative fallback (PolicyService IS the CA, so it
/// can reconstruct the exact pair the hybrid service-JWT mint signs with); an
/// unknown kid fails closed. Classical EdDSA tokens verify against the CA key
/// only when `policy` permits classical — matching the dispatch alg gate.
fn verify_service_registration_jwt(
    service_jwt: &str,
    key_source: Option<&dyn hyprstream_rpc::auth::JwtKeySource>,
    ca_jwt_key: &SigningKey,
    policy: hyprstream_rpc::crypto::CryptoPolicy,
) -> Result<hyprstream_rpc::auth::Claims> {
    let is_composite = hyprstream_rpc::auth::jwt::header_alg(service_jwt)
        .ok()
        .flatten()
        .is_some_and(|alg| alg == "ML-DSA-65-Ed25519");
    if is_composite {
        let dispatch =
            hyprstream_rpc::auth::jwt::parse_composite_dispatch(service_jwt, &["wit+jwt"])
                .map_err(|e| anyhow!("Invalid service JWT: {e}"))?;
        let pair = key_source
            .and_then(|ks| ks.composite_pair(dispatch.kid()))
            .or_else(|| {
                let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(
                    &hyprstream_rpc::node_identity::derive_mesh_mldsa_key(ca_jwt_key),
                );
                let ed_vk = ca_jwt_key.verifying_key();
                let kid = hyprstream_rpc::auth::composite_kid(&pq_vk, &ed_vk);
                (kid == dispatch.kid()).then(|| {
                    hyprstream_rpc::auth::CompositeKeyPair::verifying(
                        kid,
                        pq_vk,
                        ed_vk,
                        hyprstream_rpc::auth::CompositePairRole::Policy,
                        hyprstream_rpc::auth::CompositePairState::Active,
                        0,
                        i64::MAX,
                    )
                })
            })
            .ok_or_else(|| anyhow!("Invalid service JWT: unknown composite kid"))?;
        // Signing-domain separation: service certification is the Policy
        // domain. The OAuth-role pair legitimately signs browser and workload
        // WITs (`wit+jwt` too), so accepting its kid here would let anyone
        // holding an OAuth-plane token mint an installable service identity
        // with an arbitrary `cnf`. Only Policy-role pairs — the ledger's
        // Policy slot and the derived CA pair, which is constructed with the
        // Policy role — may certify service keys.
        anyhow::ensure!(
            pair.role() == hyprstream_rpc::auth::CompositePairRole::Policy,
            "Invalid service JWT: composite pair role is not authorized for service certification"
        );
        hyprstream_rpc::auth::jwt::decode_composite(
            service_jwt,
            pair.ml_dsa(),
            pair.ed25519(),
            None,
            &dispatch,
        )
        .map_err(|e| anyhow!("Invalid service JWT: {e}"))
    } else {
        anyhow::ensure!(
            !policy.uses_pq(),
            "Invalid service JWT: Hybrid crypto policy requires a post-quantum alg; \
             classical-only token rejected"
        );
        hyprstream_rpc::auth::jwt::decode_with_key(
            service_jwt,
            &ca_jwt_key.verifying_key(),
            None,
        )
        .map_err(|e| anyhow!("Invalid service JWT: {e}"))
    }
}

fn validate_service_key_registration(
    claims: &hyprstream_rpc::auth::Claims,
    service_name: &str,
    verifying_key: &[u8; 32],
) -> Result<()> {
    let expected_sub = format!("service:{service_name}");
    if claims.sub != expected_sub {
        anyhow::bail!("JWT subject '{}' does not match service name '{}'", claims.sub, service_name);
    }
    let cnf_bytes = claims.cnf_key_bytes()
        .ok_or_else(|| anyhow!("service JWT must contain a well-formed cnf.jwk"))?;
    if cnf_bytes != *verifying_key {
        anyhow::bail!("JWT cnf.jwk does not match provided verifying key");
    }
    Ok(())
}

/// The credential profile of a verified source credential (v16 typ contract).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceProfile {
    /// User access token (`at+jwt`).
    AtJwt,
    /// Service credential (`wit+jwt`).
    WitJwt,
}

/// A verified source credential and the provenance the derived mint requires:
/// the profile (typ) and the crypto assurance derived from the ACTUAL verified
/// algorithm (never fabricated).
struct VerifiedSourceCredential {
    claims: hyprstream_rpc::auth::Claims,
    profile: SourceProfile,
    key_material: hyprstream_rpc::auth::mac::VerifiedKeyMaterial,
}

/// Verify a presented source credential for the delegated-exchange authority
/// path (RFC 8693 §4 on-behalf-of). **JWT-only:** this Text seam carries a JWT
/// `at+jwt`/`wit+jwt`; a CWT source is out of scope and is not handled here (it
/// would be a separate typed seam). Both the hybrid (composite) and classical
/// (Ed25519) credential profiles the frozen A profile permits are accepted; the
/// derived Classical/PqHybrid assurance is taken from the ACTUAL verified
/// algorithm and the clearance meet/target dominance decides authority.
///
/// Trust rule (frozen A profile: trust the issuer key, NOT an internal
/// typ→role mapping — production signs both `at+jwt` and `wit+jwt` with the
/// Policy role): the exact `kid` must resolve to an issuer-authorized composite
/// pair in the trusted key source / ledger and that pair must be within its
/// validity window; for the classical path the verifying key(s) resolve
/// strictly from the trusted key source keyed by `expected_issuer` and the
/// exact `kid`. An untrusted issuer or an unknown/wrong `kid`/key denies. The
/// JOSE `typ` selects the profile exactly. Issuer-claim equality, revocation,
/// session, tenant, subject↔profile coherence, and clearance are enforced by
/// the caller.
async fn verify_presented_credential(
    token: &str,
    key_source: Option<&dyn hyprstream_rpc::auth::JwtKeySource>,
    expected_issuer: &str,
) -> Result<VerifiedSourceCredential> {
    use hyprstream_rpc::auth::mac::VerifiedKeyMaterial;

    let header = hyprstream_rpc::auth::jwt::parse_protected_header(token)
        .map_err(|e| anyhow!("invalid source credential header: {e}"))?;
    let profile = match header.typ.as_str() {
        "at+jwt" => SourceProfile::AtJwt,
        "wit+jwt" => SourceProfile::WitJwt,
        other => anyhow::bail!(
            "invalid source credential typ '{other}' (expected at+jwt or wit+jwt)"
        ),
    };

    match header.alg.as_str() {
        "ML-DSA-65-Ed25519" => {
            // Re-parse for dispatch with the EXACT typ, so the signed header typ
            // must equal what gated the profile above.
            let dispatch = hyprstream_rpc::auth::jwt::parse_composite_dispatch(
                token,
                &[header.typ.as_str()],
            )
            .map_err(|e| anyhow!("invalid source credential: {e}"))?;
            // Resolve the exact kid to an issuer-AUTHORIZED composite pair via
            // the configured key source. `ClusterKeySource::composite_pair`
            // resolves this node's exact-pair ledger AND its out-of-ledger
            // CA-derived composite pair (which the bootstrap/service-WIT path
            // signs with, and which is NOT in the global ledger — a global-only
            // lookup would wrongly reject a legitimate local service WIT).
            // `FederatedKeySource::composite_pair` delegates only to its local
            // source, so it never exposes a foreign issuer's pair. The key
            // source is bound to `expected_issuer`, so a resolved pair is
            // authorized for it; a kid outside that authority denies.
            let ks = key_source.ok_or_else(|| {
                anyhow!("composite source credential requires a trusted key source")
            })?;
            anyhow::ensure!(
                ks.is_trusted(expected_issuer),
                "source credential issuer is not trusted by this authority"
            );
            let pair = ks.composite_pair(dispatch.kid()).ok_or_else(|| {
                anyhow!("invalid source credential: composite kid is not an issuer-authorized pair")
            })?;
            // Pair lifecycle: the signing pair must be within its validity
            // window AND in a verify-valid state. Active and Drain are both
            // verify-valid (a Drain pair is rotating out, but tokens it signed
            // remain valid until expiry); the explicit match fails closed if the
            // ledger ever grows a non-verifying state.
            let now = chrono::Utc::now().timestamp();
            anyhow::ensure!(
                pair.not_before() <= now && now < pair.expires_at(),
                "source credential signing pair is outside its validity window"
            );
            match pair.state() {
                hyprstream_rpc::auth::CompositePairState::Active
                | hyprstream_rpc::auth::CompositePairState::Drain => {}
            }
            let claims = hyprstream_rpc::auth::jwt::decode_composite(
                token,
                pair.ml_dsa(),
                pair.ed25519(),
                None,
                &dispatch,
            )
            .map_err(|e| anyhow!("invalid source credential: {e}"))?;
            Ok(VerifiedSourceCredential {
                claims,
                profile,
                key_material: VerifiedKeyMaterial::PqHybrid,
            })
        }
        "EdDSA" => {
            // Classical profile: resolve the verifying key(s) STRICTLY from the
            // trusted key source keyed by the expected issuer and the exact kid.
            let ks = key_source.ok_or_else(|| {
                anyhow!("classical source credential requires a trusted key source")
            })?;
            anyhow::ensure!(
                ks.is_trusted(expected_issuer),
                "source credential issuer is not trusted by this authority"
            );
            // `kid` is a strict selector (present and non-empty per the parsed
            // header); an unknown/wrong kid resolves to no key and denies.
            let candidates = ks
                .get_keys(expected_issuer, Some(&header.kid))
                .await
                .map_err(|e| anyhow!("no verifying key for source credential: {e}"))?;
            anyhow::ensure!(
                !candidates.is_empty(),
                "no verifying key resolved for the source credential kid"
            );
            let claims = hyprstream_rpc::auth::jwt::decode_with_any_key(token, &candidates, None)
                .map_err(|e| anyhow!("invalid source credential: {e}"))?;
            Ok(VerifiedSourceCredential {
                claims,
                profile,
                key_material: VerifiedKeyMaterial::Classical,
            })
        }
        other => anyhow::bail!("unsupported source credential alg '{other}'"),
    }
}

/// Parse a canonical `ability@resource` capability token into the reviewed
/// typed [`hyprstream_rpc::auth::ucan::Capability`]. `None` for a malformed
/// token (empty ability/resource, or no `@`).
fn parse_capability(token: &str) -> Option<hyprstream_rpc::auth::ucan::Capability> {
    use hyprstream_rpc::auth::ucan::{Ability, Capability, Resource};
    let (ability, resource) = token.split_once('@')?;
    if ability.is_empty() || resource.is_empty() {
        return None;
    }
    Some(Capability::new(Resource::new(resource), Ability::new(ability)))
}

/// Attenuate an OAuth scope axis for a delegated hop. v16 §8.1 requires an
/// EXPLICIT attenuation at every hop: a scope-bearing source never silently
/// inherits its full scope into the derived credential. Equality is allowed
/// only when the full subset is explicitly requested. A source with no scope
/// cannot grant any — a non-empty request against it denies.
fn attenuate_delegated_scope(
    source_scope: Option<&str>,
    requested: Option<&str>,
) -> std::result::Result<Option<String>, String> {
    let source_set: std::collections::BTreeSet<&str> =
        source_scope.unwrap_or("").split_whitespace().collect();
    let requested_scopes: Vec<&str> = requested.unwrap_or("").split_whitespace().collect();
    if source_set.is_empty() {
        if !requested_scopes.is_empty() {
            return Err(
                "source credential carries no scope; requested scope cannot be granted".to_owned(),
            );
        }
        return Ok(None);
    }
    if requested_scopes.is_empty() {
        return Err(
            "scope-bearing source requires an explicit requested scope subset (no silent inheritance)"
                .to_owned(),
        );
    }
    for scope in &requested_scopes {
        if !source_set.contains(scope) {
            return Err(format!(
                "requested scope '{scope}' is not held by the source credential (broadening denied)"
            ));
        }
    }
    Ok(Some(requested_scopes.join(" ")))
}

/// Attenuate the MAC/UCAN capability axis for a delegated hop using the reviewed
/// [`hyprstream_rpc::auth::ucan::set_attenuates`] relation — never a second,
/// ad-hoc capability algebra. v16 §8.1 requires explicit attenuation: a
/// cap-bearing source never silently inherits its full `cap`; a source with no
/// `cap` cannot grant any (a non-empty request denies, output stays `None`).
/// Malformed tokens and any capability the source does not `cover` are rejected.
fn attenuate_delegated_capabilities(
    source_cap: Option<&str>,
    requested: Option<&str>,
) -> std::result::Result<Option<String>, String> {
    let parse_set = |s: &str| -> std::result::Result<
        Vec<hyprstream_rpc::auth::ucan::Capability>,
        String,
    > {
        s.split_whitespace()
            .map(|tok| {
                parse_capability(tok).ok_or_else(|| {
                    format!("malformed capability '{tok}' (expected ability@resource)")
                })
            })
            .collect()
    };
    let held = parse_set(source_cap.unwrap_or(""))?;
    let requested_str = requested.unwrap_or("").trim();
    if held.is_empty() {
        if !requested_str.is_empty() {
            return Err(
                "source credential carries no capability; requested capability cannot be granted"
                    .to_owned(),
            );
        }
        return Ok(None);
    }
    if requested_str.is_empty() {
        return Err(
            "capability-bearing source requires an explicit requested capability subset (no silent inheritance)"
                .to_owned(),
        );
    }
    let claimed = parse_set(requested_str)?;
    if !hyprstream_rpc::auth::ucan::set_attenuates(&held, &claimed) {
        return Err(
            "requested capability broadens the source authority (not covered by the source cap)"
                .to_owned(),
        );
    }
    // Stamp the requested canonical `ability@resource` form.
    Ok(Some(
        claimed
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    ))
}

/// The authenticated delegation edge presented to a [`DelegationEdgeAuthorizer`].
/// Every field is AUTHORITY-DERIVED, never caller-asserted: the actor is the
/// verified RPC envelope subject/tenant, the source audience is the verified
/// source credential's `aud`, and the target is the validated requested
/// audience. It carries no clearance, scope, or key — only the identity edge.
pub struct DelegationEdge<'a> {
    /// Bare terminal-actor service name (e.g. `"mcp"`), from the verified envelope.
    pub actor_service: &'a str,
    /// The terminal actor's verified tenant/Casbin domain.
    pub actor_tenant: &'a str,
    /// The verified source credential's `aud` — the resource the source was
    /// minted FOR. The authorizer proves this names the CURRENT terminal actor
    /// (the confused-deputy control): a source credential minted for service A
    /// cannot be rebound to service B, even with B's valid enrollment/scope.
    pub source_audience: Option<&'a str>,
    /// The verified originator subject (the source credential's `sub`).
    pub originator: &'a str,
    /// The requested derived-call target audience (validated non-empty).
    pub target_audience: &'a str,
    /// The generated target method id, when the typed dispatch call carries one
    /// (WS-E populates it from the reviewed inventory). `None` until then.
    pub target_method_id: Option<&'a str>,
}

/// Fail-closed authority deciding whether a derived `AsOriginator` delegation
/// edge is a declared/allowed call edge (v16 §8.1). WS-E installs the generated,
/// reviewed `DispatchCallManifest` implementation before E-ready / F activation;
/// **an uninstalled authorizer DENIES every edge** — the exchange cannot mint.
///
/// It proves BOTH that the source credential was valid for the current terminal
/// actor (source audience ↔ actor binding — NOT the outbound
/// `allowed_audiences` ceiling, which is a distinct control) AND that the
/// requested target is a declared edge. It is never a caller boolean and never
/// defaults to allow: a generic `manage`-scoped service credential is not, by
/// itself, the confused-deputy control.
pub trait DelegationEdgeAuthorizer: Send + Sync {
    /// `true` iff this exact edge is an authorized derived call.
    fn authorize(&self, edge: &DelegationEdge<'_>) -> bool;
}

/// A principal's ACTIVE primary signer-suite record (frozen A §5): the EXPLICIT
/// suite ID owned by the authoritative record plus its ordered raw component
/// public keys, in suite-plan order. B recomputes the confirmation thumbprint
/// over exactly `[suite_id, ordered_component_keys]` — it never infers the suite
/// ID from the key count, so a wrong-suite enrollment B would accept but C would
/// reject is caught here. Component 0 is the Ed25519 key (the `cnf.jwk` key).
pub struct PrimaryGroup {
    /// The record's explicit suite ID (validated against the frozen registry).
    pub suite_id: String,
    /// Ordered raw component public keys (Ed25519 = 32 bytes; ML-DSA-65 = 1952).
    pub ordered_component_keys: Vec<Vec<u8>>,
}

/// Fail-closed resolver of a principal's AUTHORITATIVE, off-wire PRIMARY signer
/// enrollment (frozen A §5/T1: principal/tenant/role/keys/epoch — never the wire
/// `cnf`). A **user/classical** source credential's `cnf`/`hs_signer_suite` is
/// confirmed against THIS resolver, never self-derived from the same wire key it
/// claims to bind. **Uninstalled ⇒ every user primary denies** (WS-C installs
/// the real resolver over its enrollment store). Service primaries resolve
/// against the enrollment manifest instead, so this seam governs only the
/// user/classical case.
pub trait PrimaryEnrollmentResolver: Send + Sync {
    /// The ACTIVE primary signer group of `principal` in `tenant`, or `None`
    /// when unknown, inactive, or tenant-mismatched (all deny).
    fn primary_group(&self, principal: &str, tenant: &str) -> Option<PrimaryGroup>;
}

/// Resolve the authoritative v16 `cnf.hs_signer_suite` thumbprint (base64url,
/// unpadded) for an enrolled SERVICE from the enrollment manifest, binding it to
/// the verified envelope signer key. The suite is chosen by the exact enrolled
/// key material — classical (Ed25519 only) or hybrid (Ed25519 + ML-DSA-65) — and
/// never fabricated. Fail closed on an absent manifest/enrollment, a signer key
/// that does not equal the enrolled Ed25519 key, or a malformed PQ half.
fn enrolled_service_signer_suite(
    manifest: Option<&crate::auth::service_enrollment::ServiceEnrollmentManifest>,
    service_name: &str,
    verified_ed_key: &[u8; 32],
) -> Result<String> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    let manifest =
        manifest.ok_or_else(|| anyhow!("service-enrollment manifest is not installed"))?;
    let entry = manifest
        .services
        .get(service_name)
        .ok_or_else(|| anyhow!("service '{service_name}' has no enrollment entry"))?;
    let enrolled_ed: [u8; 32] = URL_SAFE_NO_PAD
        .decode(&entry.ed25519_pubkey)
        .ok()
        .and_then(|b| <[u8; 32]>::try_from(b).ok())
        .ok_or_else(|| anyhow!("service '{service_name}' has a malformed enrolled ed25519 key"))?;
    anyhow::ensure!(
        &enrolled_ed == verified_ed_key,
        "verified signer key does not match the enrolled ed25519 key for '{service_name}'"
    );
    let pq = match &entry.ml_dsa_pubkey {
        Some(pq_b64) => Some(
            URL_SAFE_NO_PAD
                .decode(pq_b64)
                .ok()
                .filter(|b| b.len() == 1952)
                .ok_or_else(|| anyhow!("service '{service_name}' has a malformed ml_dsa key"))?,
        ),
        None => None,
    };
    Ok(hyprstream_rpc::auth::service_signer_suite_b64(
        &enrolled_ed,
        pq.as_deref(),
    ))
}

/// Outcome of the workload-session narrowing decision on a service-credential
/// renewal (v16 §3.3). Extracted from the renewal handler so the check-BEFORE-
/// narrowing invariant is unit-testable WITHOUT provisioning the signing /
/// disk-persistence boundary: the decision is a pure function of the session
/// registry state and the authoritative family policy.
#[derive(Debug)]
enum RenewalWorkloadSession {
    /// Deny the renewal with this error code + message (fail-closed: a revoked,
    /// expired, or unresolvable session never renews).
    Deny {
        code: &'static str,
        message: String,
    },
    /// The `workload_session_id` to stamp into the renewed credential, or `None`
    /// to OMIT it — either a deliberate family narrowing (the policy was removed
    /// from a live family) or a standalone service that carries no session.
    Stamp(Option<String>),
}

/// A v16 dispatch credential MUST carry a `cnf.hs_signer_suite` that equals the
/// AUTHORITATIVE signer suite of its PRIMARY signer group — signature
/// verification alone never checks the confirmation claim (frozen WS-A §5/T1),
/// and the suite is NEVER self-derived from the same wire `cnf.jwk` it claims to
/// bind.
///
/// **Primary selection (multi-hop §8.1):** for a source that carries an `act`
/// chain, `sub` stays the originator but the `cnf` belongs to the OUTERMOST
/// (current) terminal actor — so the primary principal is `act.sub` when an
/// `act` is present, else `sub`. A correct first-hop delegated token is thus a
/// valid second-hop source (its `cnf` resolves to its terminal actor).
///
/// **Resolution:** a SERVICE primary resolves against the enrollment manifest
/// (key-bound); a USER/classical primary resolves against the fail-closed
/// authoritative [`PrimaryEnrollmentResolver`] (never the wire key) — an
/// uninstalled resolver or an unknown/mismatched primary denies. The presented
/// `cnf.jwk` key MUST equal the resolved primary's Ed25519 component, and the
/// `hs_signer_suite` MUST equal the suite over the resolved ordered keys.
fn validate_credential_hs_suite(
    manifest: Option<&crate::auth::service_enrollment::ServiceEnrollmentManifest>,
    primary_resolver: Option<&dyn PrimaryEnrollmentResolver>,
    claims: &hyprstream_rpc::auth::Claims,
) -> Result<()> {
    let cnf = claims
        .cnf
        .as_ref()
        .ok_or_else(|| anyhow!("credential has no cnf confirmation"))?;
    let present = cnf
        .hs_signer_suite
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("credential cnf lacks hs_signer_suite (not a v16 credential)"))?;
    let cnf_ed = claims
        .cnf_key_bytes()
        .ok_or_else(|| anyhow!("credential cnf lacks a well-formed ed25519 primary key"))?;

    // §8.1: the primary is the OUTERMOST/current terminal actor when delegated.
    let primary_principal = claims
        .act
        .as_ref()
        .map_or(claims.sub.as_str(), |a| a.sub.as_str());

    let expected = if let Some(svc) = primary_principal.strip_prefix("service:") {
        // Service primary: the enrollment manifest is authoritative; the helper
        // key-binds the presented Ed key to the enrolled Ed key and computes the
        // suite over the enrolled ordered keys.
        enrolled_service_signer_suite(manifest, svc, &cnf_ed)?
    } else {
        // User/classical primary: the AUTHORITATIVE off-wire resolver is the
        // only source of truth — never the wire `cnf.jwk`.
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let resolver = primary_resolver.ok_or_else(|| {
            anyhow!("primary-enrollment resolver is not installed; user primary denied")
        })?;
        let tenant = claims.tenant.as_deref().unwrap_or_default();
        let group = resolver
            .primary_group(primary_principal, tenant)
            .ok_or_else(|| anyhow!("no active primary enrollment for the credential principal"))?;
        // Validate the record's EXPLICIT suite ID + component shape against the
        // frozen registry — never infer the suite from the key count, so a
        // wrong-suite enrollment cannot be blessed here and rejected by C.
        let keys = &group.ordered_component_keys;
        let shape_ok = if group.suite_id == hyprstream_rpc::auth::SUITE_CLASSICAL_ED25519 {
            keys.len() == 1 && keys[0].len() == 32
        } else if group.suite_id == hyprstream_rpc::auth::SUITE_HYBRID_ED25519_MLDSA65 {
            keys.len() == 2 && keys[0].len() == 32 && keys[1].len() == 1952
        } else {
            false
        };
        anyhow::ensure!(
            shape_ok,
            "primary enrollment suite id/shape is not a frozen-registry signer suite"
        );
        // The presented `cnf.jwk` MUST equal the enrolled Ed25519 component 0.
        anyhow::ensure!(
            keys[0].as_slice() == cnf_ed,
            "credential cnf.jwk does not equal the enrolled primary Ed25519 key"
        );
        // Recompute the thumbprint over the EXACT record value.
        let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
        URL_SAFE_NO_PAD.encode(hyprstream_rpc::auth::signer_suite_thumbprint(
            &group.suite_id,
            &refs,
        ))
    };
    anyhow::ensure!(
        present == expected,
        "credential cnf.hs_signer_suite does not equal its authoritative primary signer suite"
    );
    Ok(())
}

#[async_trait::async_trait(?Send)]
impl PolicyHandler for PolicyService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject();
        let domain = self.request_domain(ctx)?;
        let allowed = self.policy_manager.check_with_domain(
            &subject.to_string(),
            &domain,
            resource,
            operation,
        ).await;
        if allowed {
            Ok(())
        } else {
            anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
        }
    }

    async fn handle_check(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &PolicyCheck,
    ) -> Result<PolicyResponseVariant> {
        let subject = ctx.subject();
        if !ctx.is_authenticated() {
            warn!(
                requested_subject = %data.subject,
                requested_domain = %data.domain,
                resource = %data.resource,
                operation = %data.operation,
                "policy check denied: unauthenticated envelope context"
            );
            ctx.audit_authz(&data.resource, &data.operation, false);
            return Ok(PolicyResponseVariant::CheckResult(false));
        }
        let domain = match self.request_domain(ctx) {
            Ok(domain) => domain,
            Err(error) => {
                warn!(
                    caller = %subject,
                    requested_subject = %data.subject,
                    requested_domain = %data.domain,
                    resource = %data.resource,
                    operation = %data.operation,
                    %error,
                    "policy check denied: no verified authorization domain"
                );
                ctx.audit_authz(&data.resource, &data.operation, false);
                return Ok(PolicyResponseVariant::CheckResult(false));
            }
        };

        trace!(
            "Policy check: verified_subject={}, verified_domain={}, resource={}, operation={}",
            subject, domain, data.resource, data.operation
        );

        // Subject and domain are derived exclusively from the verified envelope.
        // The request fields remain on the wire for compatibility, but cannot
        // select another identity or tenant. Pass the operation string directly
        // so dot-namespaced actions are forwarded verbatim to Casbin.
        let allowed = self.policy_manager.check_with_domain(
            &subject.to_string(),
            &domain,
            &data.resource,
            &data.operation,
        ).await;

        if allowed {
            debug!(caller = %subject, %domain, "policy check allowed");
        } else {
            warn!(
                caller = %subject,
                %domain,
                resource = %data.resource,
                operation = %data.operation,
                "policy check denied by policy"
            );
        }
        ctx.audit_authz(&data.resource, &data.operation, allowed);
        Ok(PolicyResponseVariant::CheckResult(allowed))
    }

    async fn handle_issue_token(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &IssueToken,
    ) -> Result<PolicyResponseVariant> {
        trace!("Issuing JWT token");

        // Determine subject: explicit subject (if provided and authorized) or envelope identity.
        // JWT sub must contain a bare username (e.g. "randy", "birdetta") — the identity
        // system adds the namespace prefix ("token:randy") when the JWT is decoded.
        // For service tokens: sub = "service:{name}", e.g. "service:model".
        let caller_domain = self.request_domain(ctx)?;
        let requested_tenant = data
            .tenant
            .as_ref()
            .filter(|tenant| !tenant.is_empty())
            .cloned();
        if requested_tenant.as_deref() == Some("*") {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "Wildcard is not a valid token tenant".to_owned(),
                code: "INVALID_TENANT".to_owned(),
                details: String::new(),
            }));
        }
        let target_domain = requested_tenant.unwrap_or_else(|| caller_domain.clone());

        let requested_subject = data.subject.as_ref().filter(|subject| !subject.is_empty());
        if requested_subject.is_some() || target_domain != caller_domain {
            // Explicit-subject and cross-tenant issuance are authorized in the
            // TARGET tenant. This permits deliberate delegation only when the
            // caller has policy:IssueToken/manage there.
            let caller = ctx.subject().to_string();
            let allowed = self.policy_manager.check_with_domain(
                &caller,
                &target_domain,
                "policy:IssueToken",
                "manage",
            ).await;
            if !allowed {
                let target_subject = requested_subject
                    .map_or_else(|| ctx.user().to_owned(), Clone::clone);
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!(
                        "Subject '{}' is not authorized to issue tokens on behalf of '{}' in tenant '{}'",
                        caller, target_subject, target_domain
                    ),
                    code: "UNAUTHORIZED_SUBJECT".to_owned(),
                    details: "Requires 'manage' permission on 'policy:IssueToken' in the target tenant".to_owned(),
                }));
            }
        }
        let subject = if let Some(subj) = requested_subject {
            subj.clone()
        } else {
            // Use bare username from the envelope identity.
            ctx.user().to_owned()
        };
        let is_service_token = subject.starts_with("service:");

        // `IssueToken` is the shared signing boundary. Its zero/default wire
        // value is deliberately `InteractiveSession`, so a caller that omits
        // the profile cannot silently mint an unsessioned user credential.
        // The only sid-less user profiles are the separately typed RFC 8693
        // and RFC 7523 exchanges; service credentials have their own profile
        // and are never allowed to carry an interactive sid.
        match data.issuance_profile {
            IssueTokenProfile::InteractiveSession => {
                if is_service_token {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "service credentials must use the service issuance profile".to_owned(),
                        code: "INVALID_ISSUANCE_PROFILE".to_owned(),
                        details: String::new(),
                    }));
                }
                if data.session_id.as_deref().is_none_or(str::is_empty) {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "interactive user/OIDC issuance requires a session id".to_owned(),
                        code: "MISSING_SESSION".to_owned(),
                        details: String::new(),
                    }));
                }
            }
            IssueTokenProfile::Rfc8693 | IssueTokenProfile::Rfc7523 => {
                if is_service_token {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "service credentials must use the service issuance profile".to_owned(),
                        code: "INVALID_ISSUANCE_PROFILE".to_owned(),
                        details: String::new(),
                    }));
                }
                if data.session_id.is_some() {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "non-interactive RFC issuance cannot carry an OIDC session id".to_owned(),
                        code: "INVALID_SESSION".to_owned(),
                        details: String::new(),
                    }));
                }
            }
            IssueTokenProfile::Service => {
                if !is_service_token || data.session_id.is_some() {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "service issuance requires a service subject and no OIDC session id".to_owned(),
                        code: "INVALID_ISSUANCE_PROFILE".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        }

        // RFC 9068 §2.2.1 `client_id` (v16 credential profile): the user
        // `at+jwt` profiles (interactive / RFC 8693 / RFC 7523) MUST carry a
        // non-empty `client_id`; the service profile mints a `wit+jwt` and MUST
        // NOT carry one. Stamped into the signed claims below so the emitted
        // credential is profile-compliant by construction.
        let client_id = data
            .client_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        match data.issuance_profile {
            IssueTokenProfile::InteractiveSession
            | IssueTokenProfile::Rfc8693
            | IssueTokenProfile::Rfc7523 => {
                if client_id.is_none() {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "at+jwt issuance requires a non-empty client_id (RFC 9068 §2.2.1)"
                            .to_owned(),
                        code: "MISSING_CLIENT_ID".to_owned(),
                        details: String::new(),
                    }));
                }
            }
            IssueTokenProfile::Service => {
                if client_id.is_some() {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "service (wit+jwt) issuance cannot carry a client_id".to_owned(),
                        code: "INVALID_CLIENT_ID".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        }

        let mut subject_clearance = if data.require_clearance {
            (self.token_clearance_resolver)(&subject).map(|clearance| {
                let context = hyprstream_rpc::auth::mac::SecurityContext::from_clearance(
                    clearance,
                    hyprstream_rpc::auth::mac::VerifiedKeyMaterial::Classical,
                );
                *context.clearance()
            })
        } else {
            None
        };
        if data.require_clearance && subject_clearance.is_none() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Subject '{subject}' has no verified enrollment clearance"),
                code: "UNLABELED_SUBJECT".to_owned(),
                details: "MAC-enforcing token issuance fails closed for unenrolled subjects".to_owned(),
            }));
        }

        // #1159 freeze: this is the shared signing boundary for every caller
        // using PolicyClient::issue_token, including RFC 8693 token exchange
        // and RFC 7523 JWT bearer. Check the resolved concrete subject, not
        // merely an optional request field, so an envelope-derived subject
        // cannot bypass the freeze either.
        if is_path_form_did_web(&subject) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "path-form did:web account subjects are frozen; host-form account minting is not available yet (#1159)".to_owned(),
                code: "FROZEN_PATH_FORM_SUBJECT".to_owned(),
                details: String::new(),
            }));
        }

        // Validate TTL — service tokens get a longer default (7 days)
        let default_ttl = if is_service_token {
            data.ttl.filter(|&t| t != 0).unwrap_or(604800) // 7 days for service tokens
        } else {
            data.ttl.filter(|&t| t != 0).unwrap_or(self.token_config.default_ttl_seconds)
        };
        let requested_ttl = default_ttl;

        const MIN_TTL_SECONDS: u32 = 60;
        if requested_ttl < MIN_TTL_SECONDS {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("TTL too short: {} < {} seconds minimum", requested_ttl, MIN_TTL_SECONDS),
                code: "TTL_TOO_SHORT".to_owned(),
                details: String::new(),
            }));
        }

        if requested_ttl > self.token_config.max_ttl_seconds {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("TTL exceeds maximum: {} > {}", requested_ttl, self.token_config.max_ttl_seconds),
                code: "TTL_EXCEEDED".to_owned(),
                details: String::new(),
            }));
        }

        // Create and sign JWT with audience (RFC 8707) and the OAuth grant's
        // scope ceiling. Casbin still supplies subject policy; verifiers must
        // intersect it with this signed per-grant authority (#1146 T2.1).
        let now = chrono::Utc::now().timestamp();
        let audience = data.audience.as_ref().filter(|s| !s.is_empty()).cloned()
            .or_else(|| self.default_audience.clone());

        // ServiceEnrollmentManifest (v16 §11): for service subjects the
        // manifest is the authoritative target-clearance and audience source.
        // Clearance is stamped from the manifest for every enrolled service
        // (issuance cannot gain, and renewal cannot preserve, authority the
        // manifest does not grant); a declared audience list is enforced
        // fail-closed.
        if let Some(service_name) = subject.strip_prefix("service:") {
            if let Some(manifest) = crate::auth::service_enrollment::global_service_enrollment()
            {
                let Some(clearance) = manifest.clearance_for_service(service_name) else {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: format!(
                            "service '{service_name}' has no enrollment entry"
                        ),
                        code: "UNENROLLED_SERVICE".to_owned(),
                        details: String::new(),
                    }));
                };
                subject_clearance = Some(clearance);
                // A declared audience list requires the effective audience to
                // be present AND a member — an enrolled service cannot mint
                // without an exact allowed audience.
                if !manifest.allows_audience(service_name, audience.as_deref()) {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: format!(
                            "audience '{}' is not enrolled for service '{service_name}'",
                            audience.as_deref().unwrap_or("<none>")
                        ),
                        code: "AUDIENCE_NOT_ENROLLED".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        }

        let granted_scope = data.requested_scopes.as_ref().map(|scopes| {
            scopes.iter()
                .map(String::as_str)
                .filter(|scope| !scope.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        });

        // OAuth delegates for a service, so its envelope signer is OAuth's key,
        // not the service assertion signer. Bind `cnf` to the explicitly passed
        // assertion key. A registered service may attest a new sibling key.
        let service_key_bytes: Option<[u8; 32]> = if is_service_token {
            let svc_name = &subject["service:".len()..];
            let trust = hyprstream_service::global_trust_store();
            use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
            let invalid_assertion_key = |message: &str| {
                PolicyResponseVariant::Error(ErrorInfo {
                    message: message.to_owned(),
                    code: "INVALID_ASSERTION_KEY".to_owned(),
                    details: String::new(),
                })
            };
            let Some(encoded) = data.user_pub_key.as_deref() else {
                return Ok(invalid_assertion_key(
                    "service token issuance requires the assertion-verified Ed25519 public key",
                ));
            };
            let decoded = match URL_SAFE_NO_PAD.decode(encoded) {
                Ok(decoded) => decoded,
                Err(_) => return Ok(invalid_assertion_key(
                    "service token assertion key is not base64url",
                )),
            };
            let requested_bytes: [u8; 32] = match decoded.try_into() {
                Ok(bytes) => bytes,
                Err(_) => return Ok(invalid_assertion_key(
                    "service token assertion key must be 32 bytes",
                )),
            };
            let requested = match VerifyingKey::from_bytes(&requested_bytes) {
                Ok(requested) => requested,
                Err(_) => return Ok(invalid_assertion_key(
                    "service token assertion key is not a valid Ed25519 verifying key",
                )),
            };
            if trust.is_authorized(&requested, svc_name) {
                Some(requested_bytes)
            } else {
                let caller = match VerifyingKey::from_bytes(&ctx.cnf) {
                    Ok(caller) => caller,
                    Err(_) => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: "service key rotation caller has an invalid Ed25519 verifying key".to_owned(),
                        code: "UNAUTHORIZED_SERVICE_KEY".to_owned(),
                        details: String::new(),
                    })),
                };
                let expected_subject = format!("service:{svc_name}");
                if ctx.subject().name() != Some(expected_subject.as_str()) || !trust.is_authorized(&caller, svc_name) {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: format!(
                            "unregistered service key for '{svc_name}' may only be attested by a registered sibling"
                        ),
                        code: "UNAUTHORIZED_SERVICE_KEY".to_owned(),
                        details: String::new(),
                    }));
                }
                Some(requested_bytes)
            }
        } else {
            use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
            data.user_pub_key.as_deref().and_then(|s| {
                URL_SAFE_NO_PAD.decode(s).ok()?.try_into().ok()
            })
        };

        // OAuth may override the issuer for a profile-specific token (atproto
        // requires an origin). Other callers retain the configured default.
        let issuer = data.issuer.as_ref()
            .filter(|issuer| !issuer.is_empty())
            .cloned()
            .or_else(|| self.default_audience.clone())
            .unwrap_or_default();
        let mut claims = hyprstream_rpc::auth::Claims::new(
            subject.clone(),
            now,
            now + requested_ttl as i64,
        ).with_issuer(issuer.clone())
         .with_audience(audience)
         .with_scope(granted_scope);
        if target_domain != "*" {
            claims = claims.with_tenant(target_domain.clone());
        }
        if let Some(clearance) = subject_clearance {
            claims = claims.with_clearance(clearance);
        }
        // RFC 9068 §2.2.1: stamp the OAuth `client_id` on the user `at+jwt`
        // credential (validated non-empty above for those profiles; the service
        // profile carries none).
        if let Some(cid) = client_id {
            claims = claims.with_client_id(cid);
        }
        // Session binding (v16 §3.3): the OIDC `sid` profile is enforced at
        // the authority boundary. Service credentials NEVER carry an
        // interactive session (workload IDs enter only through the enrolled
        // renewal family, not this field). A user session ID is stamped only
        // when the canonical registry holds an ACTIVE record bound to this
        // exact subject and tenant — unknown, revoked, expired, cross-subject,
        // or cross-tenant sessions are rejected (no session fixation).
        if let Some(sid) = data.session_id.as_deref().filter(|s| !s.is_empty()) {
            if sid.len() > 1024 {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "session id exceeds 1024 bytes".to_owned(),
                    code: "INVALID_ARGUMENT".to_owned(),
                    details: String::new(),
                }));
            }
            if subject.starts_with("service:") {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "service credentials cannot carry an OIDC session id".to_owned(),
                    code: "INVALID_ARGUMENT".to_owned(),
                    details: String::new(),
                }));
            }
            let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "session registry is not initialized".to_owned(),
                    code: "UNAVAILABLE".to_owned(),
                    details: String::new(),
                }));
            };
            let session_key = hyprstream_rpc::auth::SessionKey::oidc(&issuer, sid);
            let bindable = match registry.session_state(&session_key).await {
                Some(state) => {
                    state.status == hyprstream_rpc::auth::ActiveOrRevoked::Active
                        && state.expires_at > now
                        && state.subject == subject
                        && state.tenant == target_domain
                }
                None => false,
            };
            if !bindable {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "session is unknown, inactive, expired, or bound to a different subject/tenant".to_owned(),
                    code: "INVALID_SESSION".to_owned(),
                    details: String::new(),
                }));
            }
            claims = claims.with_sid(sid);
        }

        // DPoP jkt takes priority over userPubKey (RFC 9449 § 6).
        if let Some(ref jkt) = data.dpop_jkt {
            claims = claims.with_cnf_jkt_thumbprint(jkt.clone());
        } else if let Some(key_bytes) = service_key_bytes {
            claims = claims.with_cnf_jwk(&key_bytes);
        }

        // v16 confirmation (frozen A §5), by STRUCTURAL classification:
        //  - DISPATCH-CAPABLE credentials — a service `wit+jwt`, or a user
        //    `at+jwt` carrying a proof-of-possession `cnf.jwk` — MUST bind their
        //    AUTHORITATIVE Primary signer group's `cnf.hs_signer_suite`, or FAIL
        //    CLOSED (never mint a non-conformant dispatch credential C rejects).
        //  - NON-DISPATCH OAuth tokens — a no-cnf bearer, or a DPoP `cnf.jkt`-
        //    only token — are standards-valid access tokens with no proof key to
        //    bind a signer suite: issuance stays valid and no suite is stamped.
        //    Delegation SOURCE validation and C proof admission independently
        //    reject any hs-less credential presented for MAC dispatch.
        let issue_err = |code: &str, message: String| {
            PolicyResponseVariant::Error(ErrorInfo {
                message,
                code: code.to_owned(),
                details: String::new(),
            })
        };
        if is_service_token {
            // Service (`wit+jwt`): the enrolled suite (manifest-authoritative,
            // key-bound to the assertion key resolved above).
            let svc = &subject["service:".len()..];
            let Some(key) = service_key_bytes else {
                return Ok(issue_err(
                    "SIGNER_SUITE_UNAVAILABLE",
                    format!("service '{svc}' issuance has no assertion key to bind a signer suite"),
                ));
            };
            match enrolled_service_signer_suite(self.enrollment().as_deref(), svc, &key) {
                Ok(suite) => claims = claims.with_cnf_hs_signer_suite(suite),
                Err(e) => {
                    return Ok(issue_err(
                        "SIGNER_SUITE_UNAVAILABLE",
                        format!("v16 signer suite for service '{svc}': {e}"),
                    ))
                }
            }
        } else if data.dpop_jkt.is_none() {
            // A user `at+jwt` carrying a proof-of-possession `cnf.jwk` is
            // DISPATCH-CAPABLE — resolve the AUTHORITATIVE Primary record for the
            // subject/tenant, require the verified cnf key to equal its Ed
            // component, and stamp the record's suite; no resolver / no active
            // record / key mismatch all FAIL CLOSED (frozen A §5). A no-cnf
            // bearer (no `service_key_bytes`) is non-dispatch and stamps none.
            if let Some(key) = service_key_bytes {
            let Some(resolver) = self.primary_enrollment_resolver.as_deref() else {
                return Ok(issue_err(
                    "PRIMARY_RESOLVER_UNAVAILABLE",
                    "primary-enrollment resolver is not installed; user issuance is fail-closed \
                     until the authoritative primary enrollment is available"
                        .to_owned(),
                ));
            };
            let Some(group) = resolver.primary_group(&subject, &target_domain) else {
                return Ok(issue_err(
                    "PRIMARY_UNRESOLVABLE",
                    format!("no active authoritative primary enrollment for '{subject}'"),
                ));
            };
            // Frozen-registry suite/shape + exact key binding.
            use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
            let keys = &group.ordered_component_keys;
            let shape_ok = if group.suite_id == hyprstream_rpc::auth::SUITE_CLASSICAL_ED25519 {
                keys.len() == 1 && keys[0].len() == 32
            } else if group.suite_id == hyprstream_rpc::auth::SUITE_HYBRID_ED25519_MLDSA65 {
                keys.len() == 2 && keys[0].len() == 32 && keys[1].len() == 1952
            } else {
                false
            };
            if !shape_ok || keys[0].as_slice() != key {
                return Ok(issue_err(
                    "PRIMARY_UNRESOLVABLE",
                    "user cnf key does not match the authoritative primary enrollment, or the \
                     enrolled suite/shape is not a frozen-registry signer suite"
                        .to_owned(),
                ));
            }
            let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
            claims = claims.with_cnf_hs_signer_suite(
                URL_SAFE_NO_PAD.encode(hyprstream_rpc::auth::signer_suite_thumbprint(
                    &group.suite_id,
                    &refs,
                )),
            );
            }
            // else: no-cnf bearer → non-dispatch OAuth token; no suite stamped.
        }
        // else: DPoP `cnf.jkt`-only → non-dispatch OAuth token; no suite stamped.

        let token = match self.sign_token(&claims, is_service_token).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Failed to issue token".to_owned(),
                    code: "SIGNING_NOT_CONFIGURED".to_owned(),
                    details: e.to_string(),
                }));
            }
        };

        Ok(PolicyResponseVariant::IssueTokenResult(TokenInfo {
            token,
            expires_at: claims.exp,
        }))
    }

    async fn handle_list_scopes(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        Ok(PolicyResponseVariant::ListScopesResult(ScopeList {
            scopes: self.supported_scopes.clone(),
        }))
    }

    async fn handle_get_policy(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        trace!("Getting current policy");

        let policies = self.policy_manager.get_policy().await;
        let groupings = self.policy_manager.get_grouping_policy().await;

        let rules: Vec<PolicyRule> = policies
            .into_iter()
            .map(|p| PolicyRule {
                subject: p.first().cloned().unwrap_or_default(),
                domain: p.get(1).cloned().unwrap_or_default(),
                resource: p.get(2).cloned().unwrap_or_default(),
                action: p.get(3).cloned().unwrap_or_default(),
                effect: p.get(4).cloned().unwrap_or_default(),
            })
            .collect();

        let grouping_list: Vec<Grouping> = groupings
            .into_iter()
            .map(|g| Grouping {
                user: g.first().cloned().unwrap_or_default(),
                role: g.get(1).cloned().unwrap_or_default(),
            })
            .collect();

        Ok(PolicyResponseVariant::GetPolicyResult(PolicyInfo {
            rules,
            groupings: grouping_list,
        }))
    }

    async fn handle_apply_template(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ApplyTemplate,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let allowed = self.policy_manager.check_with_domain(
            &caller, &domain, "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        info!("Applying policy template: {}", data.name);

        // Validate template exists
        let template = match policy_templates::get_template(&data.name) {
            Some(t) => t,
            None => {
                let available: Vec<&str> = policy_templates::get_templates()
                    .iter()
                    .map(|t| t.name)
                    .collect();
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!(
                        "Unknown template: '{}'. Available: {}",
                        data.name,
                        available.join(", ")
                    ),
                    code: "UNKNOWN_TEMPLATE".to_owned(),
                    details: String::new(),
                }));
            }
        };

        // Apply template rules via the Casbin enforcer.
        // Base rules are always present (injected at init/reload), so templates
        // only add their own rules on top. The enforcer's save_policy() persists
        // everything (base + template) to disk via the FileAdapter.
        if let Err(e) = self.policy_manager.apply_template(template).await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Failed to apply template: {}", e),
                code: "TEMPLATE_APPLY_FAILED".to_owned(),
                details: String::new(),
            }));
        }

        // Stage and commit
        let commit_msg = format!("policy: apply {} template", data.name);
        match self.stage_and_commit_policies(&commit_msg).await {
            Ok(_) => {
                info!("Template '{}' applied and committed", data.name);
                Ok(PolicyResponseVariant::ApplyTemplateResult(commit_msg))
            }
            Err(e) => {
                // Policy is already reloaded and valid, just commit failed
                warn!("Template applied but commit failed: {}", e);
                Ok(PolicyResponseVariant::ApplyTemplateResult(
                    format!("policy: apply {} template (commit failed: {})", data.name, e)
                ))
            }
        }
    }

    async fn handle_apply_draft(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ApplyDraft,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let allowed = self.policy_manager.check_with_domain(
            &caller, &domain, "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        info!("Applying draft policy changes");

        // Validate current disk state
        if let Err(e) = self.policy_manager.reload().await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Policy validation failed: {}", e),
                code: "VALIDATION_FAILED".to_owned(),
                details: "Fix errors in policy.csv before applying.".to_owned(),
            }));
        }

        // Generate commit message
        let commit_msg = data.message.as_ref().filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            format!("policy: update access control rules ({timestamp})")
        });

        // Stage and commit
        match self.stage_and_commit_policies(&commit_msg).await {
            Ok(_) => {
                info!("Draft policy applied: {}", commit_msg);
                Ok(PolicyResponseVariant::ApplyDraftResult(commit_msg))
            }
            Err(e) => {
                Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Commit failed: {}", e),
                    code: "COMMIT_FAILED".to_owned(),
                    details: String::new(),
                }))
            }
        }
    }

    async fn handle_rollback(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RollbackPolicy,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let allowed = self.policy_manager.check_with_domain(
            &caller, &domain, "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        info!("Rolling back policy to: {}", data.git_ref);

        // Validate git_ref to prevent shell injection or path traversal.
        // Accept: 40-hex SHA, short SHA (7+ hex chars), or simple branch/tag names.
        let git_ref = data.git_ref.trim().to_owned();
        {
            let valid = !git_ref.is_empty()
                && git_ref.len() <= 256
                && git_ref.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.'));
            if !valid {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Invalid git ref '{}': only [a-zA-Z0-9._/-] allowed", git_ref),
                    code: "INVALID_GIT_REF".into(),
                    details: String::new(),
                }));
            }
        }

        // Use git2 escape hatch to checkout policies/ from the target ref
        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        // Resolve ref and checkout policies/ from it
        let git_ref_clone = git_ref.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let obj = repo.revparse_single(&git_ref_clone)
                .map_err(|e| anyhow!("Invalid git ref '{}': {}", git_ref_clone, e))?;
            let commit = obj.peel_to_commit()
                .map_err(|e| anyhow!("Ref '{}' does not point to a commit: {}", git_ref_clone, e))?;
            let tree = commit.tree()
                .map_err(|e| anyhow!("Failed to get tree: {}", e))?;

            // Find the policies/ subtree
            let policies_entry = tree.get_path(std::path::Path::new("policies"))
                .map_err(|e| anyhow!("No policies/ directory in {}: {}", git_ref_clone, e))?;
            let policies_tree = repo.find_tree(policies_entry.id())
                .map_err(|e| anyhow!("Failed to read policies tree: {}", e))?;

            // Checkout the policies tree to the workdir
            let mut checkout_opts = git2::build::CheckoutBuilder::new();
            checkout_opts.force();
            checkout_opts.path("policies");
            repo.checkout_tree(policies_tree.as_object(), Some(&mut checkout_opts))
                .map_err(|e| anyhow!("Failed to checkout policies/ from {}: {}", git_ref_clone, e))?;

            Ok(())
        }).await
            .map_err(|e| anyhow!("Checkout task failed: {}", e))??;

        // Validate the restored policy
        if let Err(e) = self.policy_manager.reload().await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Rollback validation failed: {}", e),
                code: "VALIDATION_FAILED".to_owned(),
                details: "The target version contains invalid policy.".to_owned(),
            }));
        }

        // Stage and commit the rollback
        let commit_msg = format!("policy: rollback to {}", git_ref);
        match self.stage_and_commit_policies(&commit_msg).await {
            Ok(_) => {
                info!("Policy rolled back to {}", git_ref);
                Ok(PolicyResponseVariant::RollbackResult(commit_msg))
            }
            Err(e) => {
                Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Rollback commit failed: {}", e),
                    code: "COMMIT_FAILED".to_owned(),
                    details: String::new(),
                }))
            }
        }
    }

    async fn handle_get_history(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetHistory,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let allowed = self.policy_manager.check_with_domain(
            &caller, &domain, "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        let count = if data.count == 0 { 10 } else { data.count as usize };

        trace!("Getting policy history (count={})", count);

        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        let entries = tokio::task::spawn_blocking(move || -> Result<Vec<PolicyHistoryEntry>> {
            let mut entries = Vec::new();
            let mut revwalk = repo.revwalk()
                .map_err(|e| anyhow!("Failed to create revwalk: {}", e))?;
            revwalk.push_head()
                .map_err(|e| anyhow!("Failed to push HEAD: {}", e))?;

            for oid_result in revwalk {
                if entries.len() >= count {
                    break;
                }

                let oid = oid_result.map_err(|e| anyhow!("Revwalk error: {}", e))?;
                let commit = repo.find_commit(oid)
                    .map_err(|e| anyhow!("Failed to find commit: {}", e))?;

                // Check if this commit touches policies/
                let dominated = if let Ok(parent) = commit.parent(0) {
                    let commit_tree = commit.tree().ok();
                    let parent_tree = parent.tree().ok();
                    if let (Some(ct), Some(pt)) = (commit_tree, parent_tree) {
                        let diff = repo.diff_tree_to_tree(Some(&pt), Some(&ct), None).ok();
                        diff.is_some_and(|d| {
                            d.deltas().any(|delta| {
                                let path = delta.new_file().path()
                                    .or_else(|| delta.old_file().path());
                                path.is_some_and(|p| p.starts_with("policies"))
                            })
                        })
                    } else {
                        false
                    }
                } else {
                    // Root commit — check if it has policies/
                    commit.tree().ok()
                        .and_then(|t| t.get_path(std::path::Path::new("policies")).ok())
                        .is_some()
                };

                if dominated {
                    let time = commit.time();
                    let date = chrono::DateTime::from_timestamp(time.seconds(), 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "unknown".to_owned());

                    entries.push(PolicyHistoryEntry {
                        hash: oid.to_string()[..8].to_owned(),
                        message: commit.message().unwrap_or("").trim().to_owned(),
                        date,
                    });
                }
            }

            Ok(entries)
        }).await
            .map_err(|e| anyhow!("History task failed: {}", e))??;

        Ok(PolicyResponseVariant::GetHistoryResult(PolicyHistory { entries }))
    }

    async fn handle_get_diff(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetDiff,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let allowed = self.policy_manager.check_with_domain(
            &caller, &domain, "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        let git_ref = data.git_ref.as_ref().filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| "HEAD".to_owned());

        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        let output = tokio::task::spawn_blocking(move || -> Result<String> {
            let obj = repo.revparse_single(&git_ref)
                .map_err(|e| anyhow!("Invalid git ref '{}': {}", git_ref, e))?;
            let tree = obj.peel_to_tree()
                .map_err(|e| anyhow!("Could not peel {} to tree: {}", git_ref, e))?;

            let mut diff_opts = git2::DiffOptions::new();
            diff_opts.pathspec("policies/");

            let diff = repo.diff_tree_to_workdir_with_index(
                Some(&tree),
                Some(&mut diff_opts),
            ).map_err(|e| anyhow!("Failed to compute diff: {}", e))?;

            let mut result = String::new();
            diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
                let origin = line.origin();
                if origin == '+' || origin == '-' || origin == ' ' {
                    result.push(origin);
                }
                if let Ok(s) = std::str::from_utf8(line.content()) {
                    result.push_str(s);
                }
                true
            })?;

            Ok(result)
        }).await
            .map_err(|e| anyhow!("Diff task failed: {}", e))??;

        Ok(PolicyResponseVariant::GetDiffResult(output))
    }

    async fn handle_get_draft_status(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let allowed = self.policy_manager.check_with_domain(
            &caller, &domain, "policy:*", "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage policy", caller),
                code: "UNAUTHORIZED".into(),
                details: String::new(),
            }));
        }

        let reg = self.git2db.read().await;
        let handle = reg.repo(&self.registry_repo_id)?;
        let repo = handle.open_repo()
            .map_err(|e| anyhow!("Failed to open repository: {}", e))?;

        let (has_changes, summary) = tokio::task::spawn_blocking(move || -> Result<(bool, String)> {
            let mut opts = git2::StatusOptions::new();
            opts.pathspec("policies/");
            opts.include_untracked(true);

            let statuses = repo.statuses(Some(&mut opts))
                .map_err(|e| anyhow!("Failed to get status: {}", e))?;

            let count = statuses.len();
            let summary = if count == 0 {
                "no changes".to_owned()
            } else {
                format!("{} file(s) changed", count)
            };

            Ok((count > 0, summary))
        }).await
            .map_err(|e| anyhow!("Status task failed: {}", e))??;

        Ok(PolicyResponseVariant::GetDraftStatusResult(DraftStatus {
            has_changes,
            summary,
        }))
    }

    async fn handle_add_grouping(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &AddGrouping,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;

        // Fine-grained permission check: caller must have ttt.writeback on policy:roles
        let allowed = self.policy_manager.check_with_domain(
            &caller,
            &domain,
            "policy:roles",
            "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "Subject '{}' is not authorized to manage role assignments",
                    caller
                ),
                code: "UNAUTHORIZED".to_owned(),
                details: "Requires 'ttt.writeback' permission on 'policy:roles'".to_owned(),
            }));
        }

        // Validate inputs
        if data.user.is_empty() || data.role.is_empty() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "user and role must be non-empty".to_owned(),
                code: "INVALID_INPUT".to_owned(),
                details: String::new(),
            }));
        }

        // Elevated roles can only be assigned by policy service.
        const ELEVATED_ROLES: &[&str] = &["ttt.privileged", "operator"];
        let caller_subject = ctx.subject().to_string();
        if ELEVATED_ROLES.contains(&data.role.as_str())
            && caller_subject != "service:policy"
        {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "Role '{}' is an elevated role and can only be assigned by policy service",
                    data.role
                ),
                code: "UNAUTHORIZED".to_owned(),
                details: "Elevated roles require service:policy identity".to_owned(),
            }));
        }

        // Callers cannot assign roles to themselves
        if data.user == caller {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "Cannot assign roles to yourself".to_owned(),
                code: "SELF_ASSIGNMENT".to_owned(),
                details: "Callers may not assign roles to themselves".to_owned(),
            }));
        }

        // Apply the role assignment
        self.policy_manager
            .add_role_for_user_in_domain(&data.user, &data.role, &domain)
            .await
            .map_err(|e| anyhow!("Failed to add role: {}", e))?;

        // Persist in-memory Casbin state to disk before staging
        self.policy_manager.save().await
            .map_err(|e| anyhow!("Failed to save policy after role grant: {}", e))?;

        // Commit to git
        let commit_msg = format!(
            "policy: grant role {} to {} in {} [by {}]",
            data.role, data.user, domain, caller
        );
        let sha = match self.stage_and_commit_policies(&commit_msg).await {
            Ok(sha) => sha,
            Err(e) => {
                warn!("Role granted but commit failed: {}", e);
                format!("(commit failed: {})", e)
            }
        };

        info!(
            "Granted role '{}' to '{}' in domain '{}' (caller={})",
            data.role, data.user, domain, caller
        );
        Ok(PolicyResponseVariant::AddGroupingResult(sha))
    }

    async fn handle_remove_grouping(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RemoveGrouping,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;

        // Fine-grained permission check: caller must have ttt.writeback on policy:roles
        let allowed = self.policy_manager.check_with_domain(
            &caller,
            &domain,
            "policy:roles",
            "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "Subject '{}' is not authorized to manage role assignments",
                    caller
                ),
                code: "UNAUTHORIZED".to_owned(),
                details: "Requires 'ttt.writeback' permission on 'policy:roles'".to_owned(),
            }));
        }

        // Validate inputs
        if data.user.is_empty() || data.role.is_empty() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "user and role must be non-empty".to_owned(),
                code: "INVALID_INPUT".to_owned(),
                details: String::new(),
            }));
        }

        // Remove the role assignment
        self.policy_manager
            .remove_role_for_user_in_domain(&data.user, &data.role, &domain)
            .await
            .map_err(|e| anyhow!("Failed to remove role: {}", e))?;

        // Persist in-memory Casbin state to disk before staging
        self.policy_manager.save().await
            .map_err(|e| anyhow!("Failed to save policy after role revoke: {}", e))?;

        // Commit to git
        let commit_msg = format!(
            "policy: revoke role {} from {} in {} [by {}]",
            data.role, data.user, domain, caller
        );
        let sha = match self.stage_and_commit_policies(&commit_msg).await {
            Ok(sha) => sha,
            Err(e) => {
                warn!("Role revoked but commit failed: {}", e);
                format!("(commit failed: {})", e)
            }
        };

        info!(
            "Revoked role '{}' from '{}' in domain '{}' (caller={})",
            data.role, data.user, domain, caller
        );
        Ok(PolicyResponseVariant::RemoveGroupingResult(sha))
    }

    async fn handle_set_branch_visibility(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &SetBranchVisibility,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();
        let domain = ctx.domain()?;
        let resource = format!("model:{}:{}", data.model_name, data.branch_name);

        // Require manage (ttt.writeback) on the model resource
        let allowed = self.policy_manager.check_with_domain(
            &caller,
            &domain,
            &resource,
            "ttt.writeback",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Unauthorized: {} cannot manage {}", caller, resource),
                code: "UNAUTHORIZED".to_owned(),
                details: String::new(),
            }));
        }

        if data.public {
            // Make public: add wildcard infer+query rules
            let _ = self.policy_manager.add_policy_with_domain(
                "*", "*", &resource, "infer.generate", "allow").await;
            let _ = self.policy_manager.add_policy_with_domain(
                "*", "*", &resource, "query.status", "allow").await;
        } else {
            // Make private: remove wildcard rules
            let _ = self.policy_manager.remove_policy_with_domain(
                "*", "*", &resource, "infer.generate", "allow").await;
            let _ = self.policy_manager.remove_policy_with_domain(
                "*", "*", &resource, "query.status", "allow").await;
        }

        // Persist in-memory Casbin state to disk before staging
        self.policy_manager.save().await
            .map_err(|e| anyhow!("Failed to save policy after visibility change: {}", e))?;

        let vis_str = if data.public { "public" } else { "private" };
        let msg = format!(
            "policy: set {}/{} visibility={} [by {}]",
            data.model_name, data.branch_name, vis_str, caller
        );
        let sha = match self.stage_and_commit_policies(&msg).await {
            Ok(sha) => sha,
            Err(e) => {
                warn!("Visibility set but commit failed: {}", e);
                format!("(commit failed: {})", e)
            }
        };

        info!(
            "Set branch {}/{} to {} (caller={})",
            data.model_name, data.branch_name, vis_str, caller
        );
        Ok(PolicyResponseVariant::SetBranchVisibilityResult(sha))
    }

    /// Publisher registers a topic prefix. No group key stored here — publisher holds it.
    async fn handle_register_event_prefix(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterEventPrefix,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope (prevents Casbin metacharacter injection)
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: publish:events:{prefix}.*
        let scope = format!("publish:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "register").await?;
        let key = EventPrefixKey::new(ctx.domain()?, &data.prefix);
        let owner = ctx.subject().to_string();

        let mut pubkey = [0u8; 32];
        if data.publisher_ephemeral_pubkey.len() != 32 {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "publisher pubkey must be 32 bytes".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        pubkey.copy_from_slice(&data.publisher_ephemeral_pubkey);

        let mut prefixes = self.event_prefixes.write().await;
        match validate_event_prefix_registration(&prefixes, &key, &owner) {
            Ok(()) => {}
            Err(EventPrefixRegistrationError::OwnedByAnotherSubject) => return Ok(PolicyResponseVariant::Error(ErrorInfo { message: format!("prefix '{}' is already registered by another subject in this tenant", data.prefix), code: "ALREADY_EXISTS".to_owned(), details: String::new() })),
            Err(EventPrefixRegistrationError::CrossTenantShadow) => return Ok(PolicyResponseVariant::Error(ErrorInfo { message: format!("prefix '{}' conflicts with a cross-tenant prefix", data.prefix), code: "CONFLICT".to_owned(), details: String::new() })),
        }
        prefixes.insert(key, EventPrefixState {
            owner,
            publisher_pubkey: pubkey,
            schema: data.schema.clone(),
            subscriber_pubkeys: HashMap::new(),
            wrapped_keys: HashMap::new(),
        });

        tracing::info!(prefix = %data.prefix, "Registered event prefix");
        Ok(PolicyResponseVariant::RegisterEventPrefixResult)
    }

    /// Subscriber requests access. Checks scope, stores subscriber pubkey,
    /// returns publisher pubkey + any pre-wrapped key blob.
    async fn handle_subscribe_event_prefix(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &SubscribeEventPrefix,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: subscribe:events:{prefix}.*
        let scope = format!("subscribe:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "subscribe").await?;
        let key = EventPrefixKey::new(ctx.domain()?, &data.prefix);

        let mut sub_pubkey = [0u8; 32];
        if data.subscriber_ephemeral_pubkey.len() != 32 {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "subscriber pubkey must be 32 bytes".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        sub_pubkey.copy_from_slice(&data.subscriber_ephemeral_pubkey);

        let sub_hash = blake3::hash(&sub_pubkey);
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(sub_hash.as_bytes());

        let mut prefixes = self.event_prefixes.write().await;
        let state = match prefixes.get_mut(&key) {
            Some(s) => s,
            None => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("prefix '{}' not registered", data.prefix),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        };

        state.subscriber_pubkeys.insert(hash_bytes, sub_pubkey);

        // If publisher has already wrapped a key for this subscriber, return it.
        let wrapped = state.wrapped_keys.get(&hash_bytes).cloned().unwrap_or_default();

        Ok(PolicyResponseVariant::SubscribeEventPrefixResult(EventPrefixAccess {
            publisher_ephemeral_pubkey: state.publisher_pubkey.to_vec(),
            wrapped_group_key: wrapped,
            schema: state.schema.clone(),
        }))
    }

    /// Publisher fetches new subscriber pubkeys that need wrapping.
    async fn handle_get_pending_subscribers(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &GetPendingSubscribers,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: publish:events:{prefix}.*
        let scope = format!("publish:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "get_subscribers").await?;
        let key = EventPrefixKey::new(ctx.domain()?, &data.prefix);

        let prefixes = self.event_prefixes.read().await;
        let state = match prefixes.get(&key) {
            Some(s) => s,
            None => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("prefix '{}' not registered", data.prefix),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        };

        // Return pubkeys that don't have wrapped keys yet.
        let pending: Vec<Vec<u8>> = state.subscriber_pubkeys.iter()
            .filter(|(hash, _)| !state.wrapped_keys.contains_key(*hash))
            .map(|(_, pubkey)| pubkey.to_vec())
            .collect();

        Ok(PolicyResponseVariant::GetPendingSubscribersResult(PendingSubscribers {
            pubkeys: pending,
        }))
    }

    /// Publisher deposits wrapped group key blobs (opaque to PolicyService).
    async fn handle_deposit_wrapped_keys(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &DepositWrappedKeys,
    ) -> Result<PolicyResponseVariant> {
        // Validate prefix BEFORE constructing scope
        if let Err(e) = validate_event_prefix(&data.prefix) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: e,
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }

        // Authorization: publish:events:{prefix}.*
        let scope = format!("publish:events:{}.*", data.prefix);
        self.authorize(ctx, &scope, "deposit_keys").await?;
        let key = EventPrefixKey::new(ctx.domain()?, &data.prefix);

        let mut prefixes = self.event_prefixes.write().await;
        let state = match prefixes.get_mut(&key) {
            Some(s) => s,
            None => return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("prefix '{}' not registered", data.prefix),
                code: "NOT_FOUND".to_owned(),
                details: String::new(),
            })),
        };

        let mut deposited = 0u32;
        for entry in &data.entries {
            if entry.sub_pubkey_hash.len() != 32 {
                warn!(
                    prefix = %data.prefix,
                    hash_len = entry.sub_pubkey_hash.len(),
                    "Rejecting malformed wrapped key entry: sub_pubkey_hash must be 32 bytes"
                );
                continue;
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&entry.sub_pubkey_hash);
            state.wrapped_keys.insert(hash, entry.wrapped_blob.clone());
            deposited += 1;
        }

        tracing::debug!(
            prefix = %data.prefix,
            deposited,
            submitted = data.entries.len(),
            "Deposited wrapped keys"
        );
        Ok(PolicyResponseVariant::DepositWrappedKeysResult)
    }

    async fn handle_resolve_service_key(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ResolveServiceKey,
    ) -> Result<PolicyResponseVariant> {
        // #441: authoritative resolution — return the REGISTERED key or ERROR.
        // We never derive an "expected" key from the root CA as a fallback: a
        // consumer must never receive a key the signer didn't actually register,
        // because a guessed key produces a silent mis-verify ("Response signed by
        // unexpected key") three layers away at the envelope check. Registered-or-
        // error converts that into a clear, early failure.
        let response = published_service_key_response(
            hyprstream_service::global_trust_store(),
            &data.service_name,
        )?;
        debug!(key_count = response.keys.len(), "Resolved service key set for '{}'", data.service_name);
        Ok(PolicyResponseVariant::ResolveServiceKeyResult(response))
    }

    async fn handle_register_service_key(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterServiceKey,
    ) -> Result<PolicyResponseVariant> {
        let caller = ctx.subject().to_string();

        // Verify the caller is who they claim to be.
        // The service JWT must be signed by the CA (our jwt_signing_key) and
        // its subject must match "service:{serviceName}".
        let claims = verify_service_registration_jwt(
            &data.service_jwt,
            self.jwt_key_source.as_deref(),
            &self.jwt_signing_key,
            hyprstream_rpc::envelope::global_verify_policy(),
        )?;

        // Verify the provided verifying key matches the JWT's cnf.jwk claim.
        let vk_bytes: [u8; 32] = data.verifying_key.as_slice().try_into()
            .map_err(|_| anyhow!("verifying_key must be 32 bytes"))?;
        let vk = VerifyingKey::from_bytes(&vk_bytes)
            .map_err(|e| anyhow!("Invalid Ed25519 verifying key: {e}"))?;

        validate_service_key_registration(&claims, &data.service_name, &vk_bytes)?;

        // Store in trust store (key-centric: the key IS the identity)
        {
            let trust = hyprstream_service::global_trust_store();
            trust.insert(vk, hyprstream_service::Attestation {
                scopes: std::iter::once(data.service_name.clone()).collect(),
                subject: None,
                jwt: Some(data.service_jwt.clone()),
                expires_at: claims.exp,
                attested_by: Some(self.signing_key.verifying_key().to_bytes()),
            });
        }

        info!(service = %data.service_name, caller = %caller, "Registered service verifying key");

        Ok(PolicyResponseVariant::RegisterServiceKeyResult)
    }

    async fn handle_refresh_service_token(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RefreshServiceTokenRequest,
    ) -> Result<PolicyResponseVariant> {
        const MAX_TTL: i64 = 2_592_000; // 30 days
        const MIN_TTL: i64 = 3_600;     // 1 hour

        let subject = match ctx.subject().name() {
            Some(s) if s.starts_with("service:") => s.to_owned(),
            Some(s) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("Subject '{s}' is not a service identity; only services may self-renew"),
                    code: "NOT_A_SERVICE".to_owned(),
                    details: String::new(),
                }));
            }
            None => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Anonymous callers cannot refresh service tokens".to_owned(),
                    code: "ANONYMOUS".to_owned(),
                    details: String::new(),
                }));
            }
        };

        let ttl = data.ttl_seconds.clamp(MIN_TTL, MAX_TTL);
        let svc_name = &subject["service:".len()..];

        let now = chrono::Utc::now().timestamp();
        let expires_at = now + ttl;

        // Bind the renewed JWT to the verified caller key, not an arbitrary
        // sibling which happens to be published during overlap.
        let trust = hyprstream_service::global_trust_store();
        let vk = VerifyingKey::from_bytes(&ctx.cnf)
            .map_err(|_| anyhow!("refresh caller has an invalid Ed25519 verifying key"))?;
        if !trust.is_authorized(&vk, svc_name) {
            anyhow::bail!("service key '{svc_name}' is not registered for the renewing caller; refusing to fabricate cnf.jwk");
        }

        let issuer = self.default_audience.clone().unwrap_or_default();
        let tenant = ctx.domain()?;
        let mut claims =
            renewed_service_claims(subject.clone(), now, expires_at, &issuer, tenant.clone(), &ctx.cnf);

        // ServiceEnrollmentManifest (v16 §11): renewal re-derives clearance
        // from the manifest — authority removed from enrollment never
        // survives a renewal, and renewal never gains authority. Resolve the
        // manifest ONCE through the injected-or-global authority (`enrollment`)
        // so clearance, signer-suite, and the workload-family policy below all
        // read the SAME authoritative source (and an isolated test can inject a
        // `workload_session=false` family without mutating process globals).
        let enrollment_manifest = self.enrollment();
        if let Some(manifest) = enrollment_manifest.as_ref() {
            let Some(clearance) = manifest.clearance_for_service(svc_name) else {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: format!("service '{svc_name}' has no enrollment entry"),
                    code: "UNENROLLED_SERVICE".to_owned(),
                    details: String::new(),
                }));
            };
            claims = claims.with_clearance(clearance);
            // v16 confirmation: the renewed service WIT stamps the SAME
            // authoritative signer suite (enrolled Ed + optional ML-DSA), so a
            // renewed credential stays a v16 dispatch credential (never drops to
            // legacy `cnf.jwk`-only). Fail closed if it cannot be resolved.
            match enrolled_service_signer_suite(Some(&**manifest), svc_name, &ctx.cnf) {
                Ok(suite) => claims = claims.with_cnf_hs_signer_suite(suite),
                Err(e) => {
                    return Ok(PolicyResponseVariant::Error(ErrorInfo {
                        message: format!("v16 signer suite unavailable for '{svc_name}': {e}"),
                        code: "SIGNER_SUITE_UNAVAILABLE".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        }

        // Workload credential family (v16 §3.3): only an enrolled workload
        // family carries `workload_session_id`. The disposition — deny, stamp,
        // or omit — is resolved by `resolve_renewal_workload_session` (pure over
        // the canonical session registry + the authoritative family policy),
        // which enforces the check-BEFORE-narrowing ordering: a revoked/expired
        // carried session denies regardless of whether the family policy would
        // otherwise narrow it away.
        let old_wsid = ctx
            .claims()
            .and_then(|c| c.workload_session_id.as_deref())
            .map(str::to_owned);
        let family_policy = enrollment_manifest
            .as_ref()
            .map(|m| m.workload_session_policy(svc_name));
        match self
            .resolve_renewal_workload_session(
                &issuer,
                &subject,
                &tenant,
                now,
                family_policy,
                old_wsid.as_deref(),
            )
            .await
        {
            RenewalWorkloadSession::Deny { code, message } => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message,
                    code: code.to_owned(),
                    details: String::new(),
                }));
            }
            RenewalWorkloadSession::Stamp(Some(wsid)) => {
                claims = claims.with_workload_session_id(wsid);
            }
            RenewalWorkloadSession::Stamp(None) => {}
        }

        let token = match self.sign_token(&claims, true).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Failed to refresh service token".to_owned(),
                    code: "SIGNING_NOT_CONFIGURED".to_owned(),
                    details: e.to_string(),
                }));
            }
        };

        // Persist renewed JWT to disk so it survives a server restart
        let credentials_dir = crate::auth::identity_store::credentials_dir()?;
        let secrets_profile = crate::auth::identity_store::SecretsProfile::from_env()?;
        if let Err(e) = crate::auth::identity_store::write_service_jwt_for_profile(
            &credentials_dir,
            svc_name,
            secrets_profile,
            &token,
        ) {
            warn!(
                service = svc_name,
                "Failed to persist renewed JWT to disk: {e}"
            );
        }

        info!(service = svc_name, expires_at, "Renewed service JWT");
        Ok(PolicyResponseVariant::RefreshServiceTokenResult(TokenInfo {
            token,
            expires_at,
        }))
    }
    async fn handle_exchange_wit(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ExchangeWit,
    ) -> Result<PolicyResponseVariant> {
        // Identity is read from the already-verified envelope WIT — no credential submission.
        let sub = match ctx.subject().name() {
            Some(s) => s.to_owned(),
            None => {
                return Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: "Anonymous callers cannot exchange WIT for access token".to_owned(),
                    code: "ANONYMOUS".to_owned(),
                    details: String::new(),
                }));
            }
        };
        let domain = ctx.domain()?;

        // ExchangeWit signs directly rather than through `handle_issue_token`.
        // Keep its authenticated-envelope subject under the same account-DID
        // freeze before it can reach the direct signing boundary.
        if is_path_form_did_web(&sub) {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "path-form did:web account subjects are frozen; host-form account minting is not available yet (#1159)".to_owned(),
                code: "FROZEN_PATH_FORM_SUBJECT".to_owned(),
                details: String::new(),
            }));
        }

        // cnf.jwk from the verified WIT — carried through into the issued at+jwt.
        let cnf_key_bytes = ctx.claims().and_then(hyprstream_rpc::auth::Claims::cnf_key_bytes);
        if cnf_key_bytes.is_none() {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "Caller WIT missing cnf.jwk — key binding required for ExchangeWit".to_owned(),
                code: "NO_CNF_JWK".to_owned(),
                details: String::new(),
            }));
        }

        // Casbin: caller must have 'exchange' on 'policy:exchange-wit'.
        let allowed = self.policy_manager.check_with_domain(
            &sub,
            &domain,
            "policy:exchange-wit",
            "exchange",
        ).await;
        if !allowed {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!("Subject '{sub}' is not authorized to exchange WIT for access token"),
                code: "UNAUTHORIZED".to_owned(),
                details: "Requires 'exchange' permission on 'policy:exchange-wit'".to_owned(),
            }));
        }

        let _ = data;
        // v16 disposition (frozen A): this path minted an `at+jwt` with NO
        // RFC 9068 `client_id`, NO authoritative interactive/non-interactive
        // session classification, and NO authoritative primary-suite
        // `cnf.hs_signer_suite` — a non-conformant v16 dispatch credential that
        // every current admission boundary (svc.rs client_id gate, C's §5
        // confirmation) would reject downstream. Rather than mint an invalid
        // shape and rely on a later reject, this dormant path now fails CLOSED:
        // a conformant user access token is minted only through the typed OAuth
        // issuance profiles (`handle_issue_token`), never a WIT→at+jwt bridge
        // that cannot supply an OAuth client identity. Retype through those
        // authoritative profiles if this bridge is ever revived.
        Ok(PolicyResponseVariant::Error(ErrorInfo {
            message: "WIT→at+jwt exchange cannot produce a v16-conformant access token \
                      (no OAuth client_id / session classification / authoritative \
                      signer-suite confirmation); use the typed OAuth issuance profiles"
                .to_owned(),
            code: "V16_PROFILE_UNAVAILABLE".to_owned(),
            details: String::new(),
        }))
    }

    /// RFC 8693 §4 on-behalf-of delegated mint (v16 §8.1 `AsOriginator`).
    ///
    /// The authenticated RPC caller IS the terminal actor: actor subject, cnf,
    /// and tenant are derived from the verified policy envelope (never from a
    /// request field), and originator/scope/capability/session/clearance are
    /// derived from the authority-verified source credential. The authority
    /// mints a NEW delegated credential — fresh `jti`, originator `sub`, nested
    /// terminal `act`, terminal-actor `cnf`, fail-closed `meet(originator,
    /// terminal actor)` clearance (the source's own clearance is already the
    /// prior-hop meet), explicitly-attenuated scope AND capability, a
    /// manifest-bound audience, and conditional `sid`/`workload_session_id`
    /// retention — never a bearer relay. The source credential is reusable.
    async fn handle_exchange_delegated(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &ExchangeDelegated,
    ) -> Result<PolicyResponseVariant> {
        use hyprstream_rpc::auth::mac::{SecurityContext, SubjectContextClaims as _};

        let deny = |code: &str, message: &str| {
            PolicyResponseVariant::Error(ErrorInfo {
                message: message.to_owned(),
                code: code.to_owned(),
                details: String::new(),
            })
        };
        // The effective enrollment manifest (injected override or process
        // global) — the authoritative source of the actor's signer suite,
        // outbound audience ceiling, and workload-family policy.
        let manifest = self.enrollment();
        let manifest = manifest.as_deref();

        // ── 1. Terminal actor = the AUTHENTICATED RPC caller (never a field) ──
        let Some(actor_sub) = ctx.subject().name().map(str::to_owned) else {
            return Ok(deny(
                "ANONYMOUS",
                "delegated exchange requires an authenticated terminal actor",
            ));
        };
        // AsOriginator derived dispatch is service→service: only a service
        // identity holds standing authority to act on another principal's
        // behalf. A caller that cannot become a terminal actor is rejected.
        let Some(actor_svc) = actor_sub.strip_prefix("service:") else {
            return Ok(deny(
                "NOT_A_TERMINAL_ACTOR",
                "only a service identity can become a delegated terminal actor",
            ));
        };
        let actor_domain = ctx.domain()?;
        // The terminal actor's cnf = the verified Ed25519 key that signed THIS
        // request. Binding the new credential to it forces the downstream
        // envelope signer to equal the terminal actor (#680 confused-deputy).
        let actor_cnf = ctx.cnf;
        if actor_cnf == [0u8; 32] {
            return Ok(deny(
                "NO_TERMINAL_CNF",
                "terminal actor request carried no verified signing key",
            ));
        }
        // The terminal actor's OWN verified context: authority-asserted
        // clearance clamped to its verified crypto assurance. Never wire-supplied.
        let Some(actor_ctx) = ctx.security_context() else {
            return Ok(deny(
                "UNLABELED_ACTOR",
                "terminal actor has no resolvable clearance",
            ));
        };

        // ── 2. This authority's configured issuer is the trust anchor for the
        // source credential (key possession alone is not issuer trust). It is
        // resolved BEFORE verification so the classical key lookup is keyed by
        // the trusted issuer; the requested `aud` is a separate validated
        // target and never sets `iss`. ──
        let expected_issuer = match self.default_audience.as_deref() {
            Some(i) if !i.is_empty() => i.to_owned(),
            _ => {
                return Ok(deny(
                    "ISSUER_NOT_CONFIGURED",
                    "delegated exchange requires a configured authorization-server issuer",
                ))
            }
        };

        // ── 3. Verify the presented source credential (originator authority) ──
        let VerifiedSourceCredential {
            claims: source,
            profile,
            key_material: source_key_material,
        } = match verify_presented_credential(
            &data.source_credential,
            self.jwt_key_source.as_deref(),
            &expected_issuer,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                return Ok(deny("INVALID_SOURCE", &format!("source credential rejected: {e}")))
            }
        };
        let now = chrono::Utc::now().timestamp();
        if source.exp <= now {
            return Ok(deny("SOURCE_EXPIRED", "source credential has expired"));
        }
        // Issuer-claim exact-equality (again) after decode: the verified `iss`
        // MUST equal the trusted issuer the keys were resolved under.
        if source.iss != expected_issuer {
            return Ok(deny(
                "UNTRUSTED_ISSUER",
                "source credential issuer is not this authority's trusted issuer",
            ));
        }
        let originator = source.sub.clone();
        if originator.is_empty() {
            return Ok(deny("INVALID_SOURCE", "source credential has no subject"));
        }
        if is_path_form_did_web(&originator) {
            return Ok(deny(
                "FROZEN_PATH_FORM_SUBJECT",
                "path-form did:web account subjects are frozen (#1159)",
            ));
        }
        // Profile ↔ subject coherence: at+jwt is a user credential, wit+jwt a
        // service one. A typ that disagrees with the subject spelling is rejected.
        let originator_is_service = originator.starts_with("service:");
        match profile {
            SourceProfile::AtJwt if originator_is_service => {
                return Ok(deny(
                    "PROFILE_MISMATCH",
                    "at+jwt source credential carries a service subject",
                ))
            }
            SourceProfile::WitJwt if !originator_is_service => {
                return Ok(deny(
                    "PROFILE_MISMATCH",
                    "wit+jwt source credential carries a non-service subject",
                ))
            }
            _ => {}
        }

        // Frozen-A profile coherence on the SOURCE (rejected before mint):
        //  - a user `at+jwt` MUST carry a non-empty client_id AND a non-empty
        //    OIDC `sid`, and MUST NOT carry a `workload_session_id`;
        //  - a service `wit+jwt` MUST NOT carry `sid` or `client_id`; a
        //    `workload_session_id` is permitted ONLY for a service the
        //    enrollment manifest marks as an authoritative workload family.
        let source_client_id = source
            .client_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let source_sid = source.sid.as_deref().filter(|s| !s.is_empty());
        let source_wsid = source.workload_session_id.as_deref().filter(|s| !s.is_empty());
        match profile {
            SourceProfile::AtJwt => {
                // Malformations of a user at+jwt (frozen A): client_id is always
                // required; a user credential never carries a workload id.
                if source_client_id.is_none() {
                    return Ok(deny(
                        "MALFORMED_SOURCE_PROFILE",
                        "user at+jwt source requires a non-empty client_id",
                    ));
                }
                if source_wsid.is_some() {
                    return Ok(deny(
                        "MALFORMED_SOURCE_PROFILE",
                        "user at+jwt source must not carry a workload_session_id",
                    ));
                }
                // JOSE typ alone cannot distinguish an interactive session
                // at+jwt (sid REQUIRED) from a separately-typed non-interactive
                // RFC 8693/7523 at+jwt (sid ABSENT), and the credential carries
                // no off-wire discriminator for that split. Rather than guess or
                // invent a claim, the supported unambiguous delegation-source
                // subset is the interactive session credential: an at+jwt source
                // MUST present a non-empty OIDC sid (validated active below). A
                // sid-less at+jwt is a valid credential but an UNSUPPORTED
                // delegation source here (fail closed), not a malformed one.
                if source_sid.is_none() {
                    return Ok(deny(
                        "UNSUPPORTED_SOURCE",
                        "a delegatable at+jwt source must be an interactive session credential (active sid); a non-interactive/sid-less at+jwt is not a supported delegation source",
                    ));
                }
            }
            SourceProfile::WitJwt => {
                if source_sid.is_some() {
                    return Ok(deny(
                        "MALFORMED_SOURCE_PROFILE",
                        "service wit+jwt source must not carry an OIDC sid",
                    ));
                }
                if source_client_id.is_some() {
                    return Ok(deny(
                        "MALFORMED_SOURCE_PROFILE",
                        "service wit+jwt source must not carry a client_id",
                    ));
                }
                if source_wsid.is_some() {
                    let svc = &originator["service:".len()..];
                    let enrolled_family = manifest
                        .map(|m| m.workload_session_policy(svc))
                        .unwrap_or(false);
                    if !enrolled_family {
                        return Ok(deny(
                            "MALFORMED_SOURCE_PROFILE",
                            "service source carries a workload_session_id but is not an enrolled workload family",
                        ));
                    }
                }
            }
        }

        // The source MUST be a v16 credential: its `cnf.hs_signer_suite` present
        // and equal to its own subject's authoritative signer suite. Signature
        // verification alone does not check the confirmation claim.
        if let Err(e) = validate_credential_hs_suite(
            manifest,
            self.primary_enrollment_resolver.as_deref(),
            &source,
        ) {
            return Ok(deny(
                "INVALID_SOURCE_CNF",
                &format!("source credential confirmation rejected: {e}"),
            ));
        }

        // Issuer-scoped source credentials must be revocable and not revoked.
        let Some(jti) = source.jti.as_deref().filter(|s| !s.is_empty()) else {
            return Ok(deny(
                "SOURCE_NOT_REVOCABLE",
                "source credential carries no jti (not revocable)",
            ));
        };
        let id = hyprstream_rpc::auth::CredentialId::jwt(expected_issuer.clone(), jti.to_owned());
        let revoked = match hyprstream_rpc::auth::global_credential_revocation_store() {
            Some(store) => store.is_revoked(&id).await,
            None => true, // fail closed: liveness cannot be proven
        };
        if revoked {
            return Ok(deny(
                "SOURCE_REVOKED",
                "source credential is revoked or its liveness cannot be proven",
            ));
        }

        // ── 4. Coherence: originator and terminal actor MUST share a tenant ──
        // Tenant comes from the verified envelope (the actor's); the source's
        // tenant must equal it — no cross-tenant delegation.
        match source.tenant.as_deref() {
            Some(t) if t == actor_domain => {}
            _ => {
                return Ok(deny(
                    "TENANT_MISMATCH",
                    "source credential tenant does not match the terminal actor's verified tenant",
                ))
            }
        }

        // ── 5. Existing delegation chain must be bounded and fully labeled ──
        // (well-formedness only — the source's top-level clearance is already
        // the prior-hop meet, so intermediate act clearances are NOT re-folded).
        const MAX_DELEGATION_HOPS: usize = 8;
        {
            let mut hop = source.act.as_ref();
            let mut depth = 0usize;
            while let Some(actor) = hop {
                if actor.clearance.is_none() {
                    return Ok(deny(
                        "MALFORMED_ACTOR_CHAIN",
                        "an actor in the source delegation chain carries no clearance",
                    ));
                }
                depth += 1;
                if depth > MAX_DELEGATION_HOPS {
                    return Ok(deny(
                        "MALFORMED_ACTOR_CHAIN",
                        "source delegation chain exceeds the maximum hop depth",
                    ));
                }
                hop = actor.act.as_deref();
            }
        }

        // ── 6. Clearance = fail-closed meet(originator, EVERY intermediate
        // actor, terminal actor). We do NOT assume the source's top-level
        // clearance already folds its prior act-chain hops — that induction is
        // unsafe for a foreign/older accepted source — so every intermediate
        // `act.clearance` is folded in explicitly (the meet only ever lowers the
        // level and shrinks compartments, so folding is always safe). The
        // originator's assurance is the ACTUAL verified source algorithm; the
        // effective assurance is the terminal signer's (the final fold). ──
        let Some(mut met) = source.security_context(source_key_material) else {
            return Ok(deny(
                "UNLABELED_ORIGINATOR",
                "source credential has no resolvable clearance",
            ));
        };
        {
            let mut hop = source.act.as_ref();
            while let Some(actor) = hop {
                let Some(cc) = actor.clearance else {
                    // Already rejected by the well-formedness walk above, but
                    // fail closed here too rather than silently skipping a hop.
                    return Ok(deny(
                        "UNLABELED_ORIGINATOR",
                        "a source delegation-chain actor is unlabeled",
                    ));
                };
                // Assurance here is transient (overwritten by the terminal fold);
                // only the level/compartments of this hop enter the meet.
                let hop_ctx = SecurityContext::new(
                    cc.level,
                    cc.compartments,
                    hyprstream_rpc::auth::mac::VerifiedKeyMaterial::Classical,
                );
                met = SecurityContext::delegated_meet(&met, &hop_ctx);
                hop = actor.act.as_deref();
            }
        }
        // Terminal actor folds last, so the effective assurance is its verified
        // signer's crypto floor.
        let met = SecurityContext::delegated_meet(&met, &actor_ctx);

        // ── 7. Explicit attenuation of BOTH authority axes (v16 §8.1) ──
        let granted_scope = match attenuate_delegated_scope(
            source.scope.as_deref(),
            data.requested_scopes.as_deref(),
        ) {
            Ok(s) => s,
            Err(msg) => return Ok(deny("SCOPE_ATTENUATION", &msg)),
        };
        let granted_cap = match attenuate_delegated_capabilities(
            source.cap.as_deref(),
            data.requested_capabilities.as_deref(),
        ) {
            Ok(c) => c,
            Err(msg) => return Ok(deny("CAPABILITY_ATTENUATION", &msg)),
        };

        // ── 8. Audience: REQUIRED, non-empty, and bound to the reviewed
        // derived-call contract — the terminal actor's enrollment-manifest
        // allowed audiences, or (no manifest) the source credential's own
        // audience. Never an arbitrary string; never defaulted to the issuer. ──
        let Some(audience) = data
            .audience
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return Ok(deny(
                "AUDIENCE_REQUIRED",
                "delegated exchange requires an explicit non-empty audience",
            ));
        };
        // Outbound target ceiling (manifest `allowed_audiences`) — a DISTINCT
        // control from the edge authority: it bounds what a service may mint
        // for, never proof that the source named this actor. Kept as
        // defense-in-depth for an enrolled actor.
        if let Some(m) = manifest {
            if !m.allows_audience(actor_svc, Some(audience)) {
                return Ok(deny(
                    "AUDIENCE_NOT_ENROLLED",
                    "requested audience exceeds the terminal actor's outbound audience ceiling",
                ));
            }
        }
        // The derived-call target method is REQUIRED and non-empty: the
        // authorizer enforces the exact reviewed DispatchCallManifest method
        // edge, and no authorizer may treat an absent method as a wildcard.
        let Some(target_method_id) = data
            .target_method_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return Ok(deny(
                "TARGET_METHOD_REQUIRED",
                "delegated exchange requires a non-empty target method id",
            ));
        };
        // Authoritative confused-deputy control: an edge authorizer must prove
        // BOTH that the source credential was valid FOR this terminal actor
        // (source `aud` ↔ actor) AND that the requested (target audience, target
        // method) is a declared call edge. An UNINSTALLED authorizer denies — a
        // generic manage-scoped service credential is never, by itself, the
        // delegation control.
        let Some(authorizer) = self.delegation_edge_authorizer.as_deref() else {
            return Ok(deny(
                "EDGE_AUTHORIZER_UNAVAILABLE",
                "delegation-edge authorizer is not installed; derived dispatch denied",
            ));
        };
        let edge = DelegationEdge {
            actor_service: actor_svc,
            actor_tenant: &actor_domain,
            source_audience: source.aud.as_deref(),
            originator: &originator,
            target_audience: audience,
            target_method_id: Some(target_method_id),
        };
        if !authorizer.authorize(&edge) {
            return Ok(deny(
                "EDGE_DENIED",
                "the delegation edge is not an authorized derived call (source audience / actor / target / method)",
            ));
        }

        // ── 9. Lifetime: never outlives the source, the terminal actor's
        // authority, a retained session, or the configured issuance maximum ──
        let mut expires_at = source.exp;
        if let Some(actor_claims) = ctx.claims() {
            expires_at = expires_at.min(actor_claims.exp);
        }
        let ttl_override = data
            .ttl
            .map(i64::from)
            .unwrap_or(self.token_config.default_ttl_seconds as i64);
        expires_at = expires_at.min(now + ttl_override);
        expires_at = expires_at.min(now + self.token_config.max_ttl_seconds as i64);

        // ── 10. Session retention. A source's OIDC `sid` and workload
        // `workload_session_id` are DISJOINT namespaces (never cross-mapped);
        // each, when present, must be ACTIVE and exact before the mint, else the
        // whole exchange denies. A credential carrying both is malformed. ──
        if source.sid.is_some() && source.workload_session_id.is_some() {
            return Ok(deny(
                "MALFORMED_SESSION",
                "source credential carries both an OIDC sid and a workload session id",
            ));
        }
        let mut retained_sid: Option<String> = None;
        let mut retained_wsid: Option<String> = None;
        if let Some(sid) = source.sid.as_deref().filter(|s| !s.is_empty()) {
            let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
                return Ok(deny("UNAVAILABLE", "session registry is not initialized"));
            };
            let key = hyprstream_rpc::auth::SessionKey::oidc(&expected_issuer, sid);
            let state = registry.session_state(&key).await;
            let bindable = matches!(&state, Some(s)
                if s.status == hyprstream_rpc::auth::ActiveOrRevoked::Active
                    && s.expires_at > now
                    && s.subject == originator
                    && s.tenant == actor_domain);
            if !bindable {
                return Ok(deny(
                    "SESSION_INVALID",
                    "source OIDC session is revoked, expired, unknown, or mismatched",
                ));
            }
            if let Some(s) = state {
                expires_at = expires_at.min(s.expires_at);
            }
            retained_sid = Some(sid.to_owned());
        }
        if let Some(wsid) = source.workload_session_id.as_deref().filter(|s| !s.is_empty()) {
            let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
                return Ok(deny("UNAVAILABLE", "session registry is not initialized"));
            };
            let key = hyprstream_rpc::auth::SessionKey::workload(&expected_issuer, wsid);
            let state = registry.session_state(&key).await;
            let bindable = matches!(&state, Some(s)
                if s.status == hyprstream_rpc::auth::ActiveOrRevoked::Active
                    && s.expires_at > now
                    && s.subject == originator
                    && s.tenant == actor_domain);
            if !bindable {
                return Ok(deny(
                    "SESSION_INVALID",
                    "source workload session is revoked, expired, unknown, or mismatched",
                ));
            }
            if let Some(s) = state {
                expires_at = expires_at.min(s.expires_at);
            }
            retained_wsid = Some(wsid.to_owned());
        }

        // The derived credential must retain a usable lifetime after every clamp.
        const MIN_TTL_SECONDS: i64 = 60;
        if expires_at - now < MIN_TTL_SECONDS {
            return Ok(deny(
                "SOURCE_TOO_SHORT",
                "source/actor/session lifetime leaves no usable delegated TTL",
            ));
        }

        // ── 11. RFC 9068 client_id provenance. A user-originator delegated
        // credential is an `at+jwt` and inherits the originator's OAuth
        // client_id (validated present by the profile-coherence gate above); a
        // service-originator mints a `wit+jwt` and carries NO client_id (a
        // service source's client_id was already rejected). Never stamp a
        // client_id onto a service output. ──
        let client_id = if originator_is_service {
            None
        } else {
            source_client_id
        };

        // ── 12. Resolve the TERMINAL ACTOR's authoritative v16 signer suite from
        // its enrollment (classical or hybrid by exact enrolled key material),
        // key-bound to the verified envelope signer. The delegated credential's
        // `cnf` binds the terminal actor (it signs the downstream envelope), so
        // its confirmation is the actor's suite — a bare Ed25519 `cnf.jwk`
        // cannot represent a hybrid actor. Fail closed if unresolvable. ──
        let actor_suite = match enrolled_service_signer_suite(manifest, actor_svc, &actor_cnf) {
            Ok(s) => s,
            Err(e) => {
                return Ok(deny(
                    "ACTOR_SUITE_UNRESOLVABLE",
                    &format!("terminal actor v16 signer suite could not be resolved: {e}"),
                ))
            }
        };

        // ── 13. Build and sign the fresh delegated credential ──
        let met_label = *met.clearance();
        let terminal_actor = hyprstream_rpc::auth::ActClaim {
            sub: actor_sub.clone(),
            clearance: Some(hyprstream_rpc::auth::mac::CredentialClearance::from_label(
                *actor_ctx.clearance(),
            )),
            // Nest the source's existing delegation chain beneath the new
            // terminal actor (RFC 8693 §4.1 outermost-is-current ordering).
            act: source.act.clone().map(Box::new),
        };
        let mut claims = hyprstream_rpc::auth::Claims::new(originator.clone(), now, expires_at)
            .with_issuer(expected_issuer)
            .with_tenant(actor_domain)
            .with_audience(Some(audience.to_owned()))
            .with_scope(granted_scope)
            // Stamp the already-met clearance so even a reader that ignores the
            // `act` chain sees the lowest (meet) authority; a downstream
            // delegated_meet re-take is idempotent (meet ≤ every input).
            .with_clearance(met_label)
            .with_act(terminal_actor)
            // v16 confirmation = the terminal actor's authoritative signer suite;
            // the legacy `cnf.jwk` (actor Ed key) is preserved alongside it.
            .with_cnf_jwk(&actor_cnf)
            .with_cnf_hs_signer_suite(actor_suite);
        if let Some(cap) = granted_cap {
            claims = claims.with_cap(cap);
        }
        if let Some(cid) = client_id {
            claims = claims.with_client_id(cid);
        }
        if let Some(sid) = retained_sid {
            claims = claims.with_sid(sid);
        }
        if let Some(wsid) = retained_wsid {
            claims = claims.with_workload_session_id(wsid);
        }

        // `sign_token` injects a fresh `jti` (the encoder mints one when absent),
        // so the delegated credential's id is always distinct from the source.
        let token = match self.sign_token(&claims, originator_is_service).await {
            Ok(t) => t,
            Err(e) => {
                return Ok(deny(
                    "SIGNING_NOT_CONFIGURED",
                    &format!("failed to mint delegated credential: {e}"),
                ))
            }
        };

        info!(
            originator = %originator,
            actor = %actor_sub,
            expires_at = claims.exp,
            "ExchangeDelegated: minted delegated credential"
        );
        Ok(PolicyResponseVariant::ExchangeDelegatedResult(TokenInfo {
            token,
            expires_at: claims.exp,
        }))
    }

    /// Publish a credential revocation into the canonical store this service
    /// owns. Fail-closed: an uninitialized store or a failed durable write is
    /// reported as an error, never as a silent success.
    async fn handle_revoke_credential(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RevokeCredential,
    ) -> Result<PolicyResponseVariant> {
        let Some(id) = crate::services::revocation::credential_id_from_ref(&data.credential)
        else {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "malformed credential id (empty issuer or value)".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        };
        // Server-side publication bounds. The retention horizon is derived
        // from the configured issuance maxima at construction (default 45
        // days covers the hard 30-day service-JWT clamp with margin), so no
        // issuable credential can outlive its revocability. A
        // caller-controlled far-future exp would otherwise make an entry
        // effectively permanent (never GC'd, never dropped on load), and
        // unbounded issuer/value sizes would fill the authority's durable
        // log. Reject, never clamp: a clamped exp would silently un-revoke a
        // still-live token after reload.
        const MAX_ID_FIELD_BYTES: usize = 1024;
        let now = chrono::Utc::now().timestamp();
        if data.expires_at <= 0 || data.expires_at > now + self.revocation_max_ttl_secs {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "expires_at out of bounds (must be 0 < exp <= now + {}s)",
                    self.revocation_max_ttl_secs
                ),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        let value_len = match &id.value {
            hyprstream_rpc::auth::CredentialValue::Jwt(jti) => jti.len(),
            hyprstream_rpc::auth::CredentialValue::Cwt(cti) => cti.len(),
        };
        if id.issuer.len() > MAX_ID_FIELD_BYTES || value_len > MAX_ID_FIELD_BYTES {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "credential id field exceeds 1024 bytes".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        let Some(store) = hyprstream_rpc::auth::global_credential_revocation_store() else {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "revocation authority store is not initialized".to_owned(),
                code: "UNAVAILABLE".to_owned(),
                details: String::new(),
            }));
        };
        if let Err(e) = store.revoke_credential(id.clone(), data.expires_at).await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "revocation publication was not durably accepted".to_owned(),
                code: "UNAVAILABLE".to_owned(),
                details: e.to_string(),
            }));
        }
        info!(credential = %id, "Credential revocation published");
        Ok(PolicyResponseVariant::RevokeCredentialResult)
    }

    /// Answer a revocation check from the canonical store. Fail-closed twice
    /// over: a malformed credential ID reports revoked, and an uninitialized
    /// store reports revoked — a credential whose status cannot be checked is
    /// never reported live.
    async fn handle_check_credential_revocation(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &CheckCredentialRevocation,
    ) -> Result<PolicyResponseVariant> {
        let revoked = match crate::services::revocation::credential_id_from_ref(&data.credential) {
            Some(id) => match hyprstream_rpc::auth::global_credential_revocation_store() {
                Some(store) => store.is_revoked(&id).await,
                None => true,
            },
            None => true,
        };
        Ok(PolicyResponseVariant::CheckCredentialRevocationResult(revoked))
    }

    /// Register a session with the canonical registry this service owns.
    /// Fail-closed: a malformed record, an out-of-horizon expiry, an
    /// uninitialized registry, or a failed durable write is an error, never
    /// a silent success. Issuance callers own the session lifecycle; the
    /// registry never mints session IDs.
    async fn handle_register_session(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RegisterSession,
    ) -> Result<PolicyResponseVariant> {
        const MAX_FIELD_BYTES: usize = 1024;
        let invalid = |message: &str| {
            Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: message.to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }))
        };
        let Some(key) = crate::services::revocation::session_key_from_ref(&data.session) else {
            return invalid("malformed session key (empty issuer or identifier)");
        };
        let now = chrono::Utc::now().timestamp();
        if data.subject.is_empty()
            || data.tenant.is_empty()
            || data.subject.len() > MAX_FIELD_BYTES
            || data.tenant.len() > MAX_FIELD_BYTES
        {
            return invalid("session subject/tenant empty or over 1024 bytes");
        }
        if data.expires_at <= 0 || data.expires_at > now + self.revocation_max_ttl_secs {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: format!(
                    "expires_at out of bounds (must be 0 < exp <= now + {}s)",
                    self.revocation_max_ttl_secs
                ),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        }
        let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "session registry is not initialized".to_owned(),
                code: "UNAVAILABLE".to_owned(),
                details: String::new(),
            }));
        };
        let kind = match &key.id {
            hyprstream_rpc::auth::SessionIdentifier::OidcSid(_) => {
                hyprstream_rpc::auth::SessionKind::Interactive
            }
            hyprstream_rpc::auth::SessionIdentifier::WorkloadSessionId(_) => {
                hyprstream_rpc::auth::SessionKind::Workload
            }
        };
        let state = hyprstream_rpc::auth::SessionState {
            subject: data.subject.clone(),
            tenant: data.tenant.clone(),
            kind,
            created_at: now,
            expires_at: data.expires_at,
            status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
            clearance_epoch: data.clearance_epoch,
        };
        match registry.register_session(key, state).await {
            Ok(()) => Ok(PolicyResponseVariant::RegisterSessionResult),
            Err(e) => {
                let code = match &e {
                    hyprstream_rpc::auth::SessionRegisterError::Exists(_) => "ALREADY_EXISTS",
                    hyprstream_rpc::auth::SessionRegisterError::PublicationFailed(_) => {
                        "UNAVAILABLE"
                    }
                    _ => "INVALID_ARGUMENT",
                };
                Ok(PolicyResponseVariant::Error(ErrorInfo {
                    message: e.to_string(),
                    code: code.to_owned(),
                    details: String::new(),
                }))
            }
        }
    }

    /// Revoke a session: every credential carrying it is then rejected.
    /// Fail-closed: an uninitialized registry or a failed durable write is an
    /// error, never a silent success.
    async fn handle_revoke_session(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &RevokeSession,
    ) -> Result<PolicyResponseVariant> {
        let Some(key) = crate::services::revocation::session_key_from_ref(&data.session) else {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "malformed session key (empty issuer or identifier)".to_owned(),
                code: "INVALID_ARGUMENT".to_owned(),
                details: String::new(),
            }));
        };
        let Some(registry) = hyprstream_rpc::auth::global_session_registry() else {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "session registry is not initialized".to_owned(),
                code: "UNAVAILABLE".to_owned(),
                details: String::new(),
            }));
        };
        if let Err(e) = registry.revoke_session(&key).await {
            return Ok(PolicyResponseVariant::Error(ErrorInfo {
                message: "session revocation was not durably accepted".to_owned(),
                code: "UNAVAILABLE".to_owned(),
                details: e.to_string(),
            }));
        }
        Ok(PolicyResponseVariant::RevokeSessionResult)
    }

    /// Answer a session check from the canonical registry. True means ACTIVE
    /// and known; revoked, expired, unknown, malformed, or an uninitialized
    /// registry all read false — a session whose state cannot be checked is
    /// never reported active.
    async fn handle_check_session(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &CheckSession,
    ) -> Result<PolicyResponseVariant> {
        let active = match crate::services::revocation::session_key_from_ref(&data.session) {
            Some(key) => match hyprstream_rpc::auth::global_session_registry() {
                Some(registry) => !registry.is_revoked(&key).await,
                None => false,
            },
            None => false,
        };
        Ok(PolicyResponseVariant::CheckSessionResult(active))
    }
}

#[async_trait(?Send)]
impl RequestService for PolicyService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
        trace!(
            "Policy request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_policy(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "policy"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        (*self.signing_key).clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.default_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.jwt_key_source.clone()
    }

    fn resolve_key_subject(&self, signer_pubkey: &[u8; 32]) -> Option<hyprstream_rpc::envelope::Subject> {
        hyprstream_service::global_trust_store().resolve_subject(signer_pubkey)
    }

    // credential_revocation_store() uses the default trait impl, which
    // returns the process-global store. No override needed.

    fn cache_key_binding(
        &self,
        verifying_key: ed25519_dalek::VerifyingKey,
        subject: &str,
        jwt: &str,
        expires_at: i64,
    ) {
        hyprstream_service::global_trust_store().insert(verifying_key, hyprstream_service::Attestation {
            scopes: std::collections::HashSet::new(),
            subject: Some(subject.to_owned()),
            jwt: Some(jwt.to_owned()),
            expires_at,
            attested_by: Some(self.signing_key.verifying_key().to_bytes()),
        });
    }

    fn accept_delegated_bearer(&self, signer_pubkey: &[u8; 32]) -> bool {
        let Some(actor) =
            hyprstream_service::global_trust_store().resolve_subject(signer_pubkey)
        else {
            return false;
        };
        let Some(actor_name) = actor.name() else {
            return false;
        };
        if actor.is_federated() || !actor_name.starts_with("service:") {
            return false;
        }

        policy_templates::SERVICE_BASE_POLICIES.iter().any(|rule| {
            (rule.subject == actor_name || rule.subject == "service:*")
                && rule.domain == "*"
                && matches!(rule.resource, "policy:*" | "policy:PolicyCheck")
                && matches!(rule.action, "*" | "check")
                && rule.effect == "allow"
        })
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = PolicyResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// Policy file watcher (hot-reload)
// ============================================================================

/// Watch policy.csv for changes and reload PolicyManager automatically.
///
/// Watches the parent directory (not the file directly) to handle atomic
/// rename patterns used by editors like vim and emacs.
pub(crate) async fn watch_policy_file(
    policy_manager: Arc<PolicyManager>,
    policy_csv: std::path::PathBuf,
) {
    use notify::{Event, EventKind, RecursiveMode, Watcher};
    use tracing::{info, warn};

    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(16);

    let csv_path = policy_csv.clone();
    let mut watcher = match notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            match event.kind {
                EventKind::Modify(_) | EventKind::Create(_)
                    if event.paths.iter().any(|p| p.ends_with("policy.csv")) =>
                {
                    let _ = tx.blocking_send(());
                }
                _ => {}
            }
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            warn!("Failed to create policy file watcher: {}", e);
            return;
        }
    };

    // Watch parent directory to catch atomic renames
    let watch_dir = match policy_csv.parent() {
        Some(dir) => dir,
        None => {
            warn!("policy.csv has no parent directory, cannot watch");
            return;
        }
    };

    if let Err(e) = watcher.watch(watch_dir, RecursiveMode::NonRecursive) {
        warn!("Failed to watch {}: {}", watch_dir.display(), e);
        return;
    }

    info!("Watching {} for policy changes", csv_path.display());

    loop {
        // Wait for first event
        if rx.recv().await.is_none() {
            break; // Channel closed
        }

        // Debounce: wait 500ms then drain remaining events
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        while rx.try_recv().is_ok() {}

        // Reload policy
        match policy_manager.reload().await {
            Ok(()) => info!("Policy reloaded from disk"),
            Err(e) => warn!("Failed to reload policy: {}", e),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};

    /// Mint the exact hybrid service JWT a fresh bootstrap produces for
    /// `service:{name}`: composite-signed by the CA pair derived from
    /// `ca_jwt_key`.
    fn bootstrap_hybrid_service_jwt(ca_jwt_key: &SigningKey, name: &str) -> String {
        let now = chrono::Utc::now().timestamp();
        let claims =
            hyprstream_rpc::auth::Claims::new(format!("service:{name}"), now, now + 3600);
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(ca_jwt_key);
        let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq);
        hyprstream_rpc::auth::jwt::encode_service_jwt_hybrid(&claims, ca_jwt_key, &ca_pq, &ca_pq_vk)
    }

    /// A freshly bootstrapped hybrid service JWT passes the registration
    /// verification under a Hybrid policy — through the dispatch-style
    /// key-source resolution when the source carries the CA composite pair,
    /// and through the CA's own derived pair when no key source is wired.
    #[test]
    fn registration_accepts_hybrid_service_jwt_under_hybrid_policy() {
        let ca = SigningKey::from_bytes(&[0x2A; 32]);
        let jwt = bootstrap_hybrid_service_jwt(&ca, "discovery");

        // Key-source path: the same composite resolution the dispatch plane uses.
        let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
        let key_source = hyprstream_rpc::auth::ClusterKeySource::new(
            ca.verifying_key(),
            "http://localhost:9080".to_owned(),
        )
        .with_ca_composite_key(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq));
        let claims = verify_service_registration_jwt(
            &jwt,
            Some(&key_source),
            &ca,
            hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
        )
        .unwrap();
        assert_eq!(claims.sub, "service:discovery");

        // CA-fallback path: no key source wired, PolicyService derives the pair.
        let claims = verify_service_registration_jwt(
            &jwt,
            None,
            &ca,
            hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
        )
        .unwrap();
        assert_eq!(claims.sub, "service:discovery");
    }

    /// A classical EdDSA service JWT is rejected at registration under a
    /// Hybrid policy (matching the dispatch alg gate), and accepted only
    /// under Classical.
    #[test]
    fn registration_rejects_classical_service_jwt_under_hybrid_policy() {
        let ca = SigningKey::from_bytes(&[0x2B; 32]);
        let now = chrono::Utc::now().timestamp();
        let claims =
            hyprstream_rpc::auth::Claims::new("service:discovery".to_owned(), now, now + 3600);
        let jwt = hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, &ca);

        assert!(verify_service_registration_jwt(
            &jwt,
            None,
            &ca,
            hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
        )
        .is_err());
        assert!(verify_service_registration_jwt(
            &jwt,
            None,
            &ca,
            hyprstream_rpc::crypto::CryptoPolicy::Classical,
        )
        .is_ok());
    }

    /// A renewed service JWT binds the issuer URL in BOTH `iss` and `aud`
    /// (matching the provisioning mint, which strict composite audience
    /// validation requires); with no issuer configured, neither is stamped.
    #[test]
    fn renewed_service_claims_bind_issuer_and_audience() {
        let cnf = [7u8; 32];
        let claims = renewed_service_claims(
            "service:model".to_owned(),
            100,
            200,
            "http://localhost:9080",
            "tenant-a.example".to_owned(),
            &cnf,
        );
        assert_eq!(claims.iss, "http://localhost:9080");
        assert_eq!(claims.aud.as_deref(), Some("http://localhost:9080"));
        assert_eq!(claims.sub, "service:model");

        let bare = renewed_service_claims(
            "service:model".to_owned(),
            100,
            200,
            "",
            "tenant-a.example".to_owned(),
            &cnf,
        );
        assert!(bare.iss.is_empty());
        assert!(bare.aud.is_none());
    }

    /// Signing-domain separation: a `wit+jwt` composite-signed by the ACTIVE
    /// OAUTH-role pair — the pair that legitimately signs browser/workload
    /// WITs — must be rejected by the registration verification even though
    /// its kid resolves, while the same token signed by a Policy-role pair
    /// is accepted. Otherwise a compromised OAuth signing key could mint an
    /// installable service identity with an arbitrary `cnf`.
    #[test]
    fn registration_rejects_oauth_role_pair_and_accepts_policy_role_pair() {
        use hyprstream_rpc::auth::{
            CompositeKeyPair, CompositeKeySet, CompositePairRole, CompositePairState,
        };
        use std::sync::Arc;

        let ca = SigningKey::from_bytes(&[0x2E; 32]);
        let now = chrono::Utc::now().timestamp();
        let claims =
            hyprstream_rpc::auth::Claims::new("service:discovery".to_owned(), now, now + 3600);

        let mut role_pairs = Vec::new();
        for (seed, role) in [
            (0x2Fu8, CompositePairRole::OAuth),
            (0x30u8, CompositePairRole::Policy),
        ] {
            let ed = SigningKey::from_bytes(&[seed; 32]);
            let (pq, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
            let jwt = crate::auth::jwt::encode_composite_service_jwt(&claims, &pq, &ed);
            let kid = crate::auth::jwt::composite_kid(&pq_vk, &ed.verifying_key());
            role_pairs.push((
                jwt,
                CompositeKeyPair::signing(
                    kid,
                    Arc::new(pq),
                    Arc::new(ed),
                    role,
                    CompositePairState::Active,
                    0,
                    i64::MAX,
                ),
            ));
        }

        let key_set = Arc::new(CompositeKeySet::default());
        key_set
            .publish(
                1,
                "role-separation".to_owned(),
                role_pairs.iter().map(|(_, pair)| pair.clone()).collect(),
            )
            .unwrap();
        let key_source = hyprstream_rpc::auth::ClusterKeySource::new(
            ca.verifying_key(),
            "http://localhost:9080".to_owned(),
        )
        .with_composite_key_set(key_set);

        let (oauth_jwt, _) = &role_pairs[0];
        assert!(
            verify_service_registration_jwt(
                oauth_jwt,
                Some(&key_source),
                &ca,
                hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            )
            .is_err(),
            "an OAuth-role pair must not certify a service identity"
        );

        let (policy_jwt, _) = &role_pairs[1];
        let verified = verify_service_registration_jwt(
            policy_jwt,
            Some(&key_source),
            &ca,
            hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
        )
        .unwrap();
        assert_eq!(verified.sub, "service:discovery");
    }

    /// A composite service JWT signed by a DIFFERENT pair (unknown kid)
    /// fails closed — neither the key source nor the CA fallback resolves it.
    #[test]
    fn registration_rejects_unknown_composite_kid() {
        let ca = SigningKey::from_bytes(&[0x2C; 32]);
        let other = SigningKey::from_bytes(&[0x2D; 32]);
        let jwt = bootstrap_hybrid_service_jwt(&other, "discovery");

        let result = verify_service_registration_jwt(
            &jwt,
            None,
            &ca,
            hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
        );
        assert!(
            result.is_err(),
            "a composite kid the CA did not sign for must fail closed"
        );
    }

    async fn test_service_with_manager(
        manager: Arc<PolicyManager>,
    ) -> (PolicyService, tempfile::TempDir) {
        let root = tempfile::tempdir().expect("test: create policy git directory");
        let git2db = Arc::new(RwLock::new(
            Git2DB::open(root.path()).await.expect("test: open policy git database"),
        ));
        let service = PolicyService::new(
            manager,
            Arc::new(SigningKey::from_bytes(&[0x51; 32])),
            crate::config::TokenConfig::default(),
            git2db,
            TransportConfig::inproc("policy-path-form-subject-test"),
        );
        (service, root)
    }

    async fn test_service() -> (PolicyService, tempfile::TempDir) {
        let manager = Arc::new(PolicyManager::permissive().await.expect("test: policy manager"));
        test_service_with_manager(manager).await
    }

    fn issue(subject: &str) -> IssueToken {
        IssueToken {
            requested_scopes: Some(vec!["read".to_owned()]),
            ttl: Some(60),
            audience: None,
            subject: Some(subject.to_owned()),
            user_pub_key: None,
            dpop_jkt: None,
            issuer: None,
            tenant: None,
            require_clearance: false,
            session_id: None,
            issuance_profile: IssueTokenProfile::Rfc8693,
            // The user at+jwt profiles require a non-empty RFC 9068 client_id.
            client_id: Some("hyprstream-oauth-client-1".to_owned()),
        }
    }

    /// A classical single-Ed authoritative Primary resolver for `subject` bound
    /// to `ed_key` — the isolated fixture equivalent of WS-C's enrollment store,
    /// so v16 user-issuance can stamp `cnf.hs_signer_suite`.
    fn v16_primary_resolver(subject: &str, ed_key: [u8; 32]) -> Arc<dyn PrimaryEnrollmentResolver> {
        struct R {
            subject: String,
            key: [u8; 32],
        }
        impl PrimaryEnrollmentResolver for R {
            fn primary_group(&self, principal: &str, _tenant: &str) -> Option<PrimaryGroup> {
                (principal == self.subject).then(|| PrimaryGroup {
                    suite_id: hyprstream_rpc::auth::SUITE_CLASSICAL_ED25519.to_owned(),
                    ordered_component_keys: vec![self.key.to_vec()],
                })
            }
        }
        Arc::new(R {
            subject: subject.to_owned(),
            key: ed_key,
        })
    }

    /// A single-service classical enrollment manifest for `service` bound to
    /// `ed_key`, so v16 service-issuance can stamp the enrolled signer suite.
    fn v16_service_manifest(
        service: &str,
        ed_key: [u8; 32],
    ) -> Arc<crate::auth::service_enrollment::ServiceEnrollmentManifest> {
        use crate::auth::service_enrollment::{
            ServiceEnrollment, ServiceEnrollmentManifest, SERVICE_ENROLLMENT_VERSION,
        };
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel};
        let mut services = std::collections::BTreeMap::new();
        services.insert(
            service.to_owned(),
            ServiceEnrollment {
                ed25519_pubkey: URL_SAFE_NO_PAD.encode(ed_key),
                ml_dsa_pubkey: None,
                clearance: SecurityLabel::new(
                    Level::Internal,
                    Assurance::Classical,
                    CompartmentSet::EMPTY,
                ),
                allowed_audiences: None,
                workload_session: false,
            },
        );
        Arc::new(ServiceEnrollmentManifest {
            version: SERVICE_ENROLLMENT_VERSION,
            services,
        })
    }

    #[tokio::test]
    async fn request_domain_keeps_only_global_authorities_outside_tenant_domains() {
        let (service, _root) = test_service().await;
        let service_signer = SigningKey::from_bytes(&[0x50; 32]).verifying_key();

        let oauth = EnvelopeContext::for_test_authenticated_subject(
            Subject::new("service:oauth"),
            service_signer,
        );
        assert_eq!(
            service.request_domain(&oauth).expect("global OAuth domain"),
            "*"
        );

        let tenant_user = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("alice"),
            "tenant-a",
            service_signer,
        );
        assert_eq!(
            service
                .request_domain(&tenant_user)
                .expect("verified tenant domain"),
            "tenant-a"
        );

        let tenantless_user =
            EnvelopeContext::for_test_authenticated_subject(Subject::new("alice"), service_signer);
        assert!(service.request_domain(&tenantless_user).is_err());

        let federated_service = EnvelopeContext::for_test_authenticated_subject(
            Subject::federated("https://peer.example", "service:oauth"),
            service_signer,
        );
        assert!(
            service.request_domain(&federated_service).is_err(),
            "a federated subject cannot claim local global-service authority by name"
        );

        let wildcard_service = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("service:oauth"),
            "*",
            service_signer,
        );
        assert!(service.request_domain(&wildcard_service).is_err());

        let policy_authority = EnvelopeContext::for_test_authenticated_subject(
            Subject::anonymous(),
            service.signing_key.verifying_key(),
        );
        assert_eq!(
            service
                .request_domain(&policy_authority)
                .expect("PolicyService bootstrap authority domain"),
            "*"
        );
    }

    #[tokio::test]
    async fn unauthenticated_policy_check_is_denied() {
        let manager = Arc::new(
            PolicyManager::new_in_memory()
                .await
                .expect("test: policy manager"),
        );
        manager
            .add_policy_with_domain(
                "alice",
                "tenant-a",
                "model:allowed",
                "infer.generate",
                "allow",
            )
            .await
            .expect("test: policy grant");
        let (service, _root) = test_service_with_manager(manager).await;
        let ctx = EnvelopeContext::for_test_authenticated_subject(
            Subject::anonymous(),
            SigningKey::from_bytes(&[0x53; 32]).verifying_key(),
        );
        let request = PolicyCheck {
            subject: "alice".to_owned(),
            domain: "tenant-a".to_owned(),
            resource: "model:allowed".to_owned(),
            operation: "infer.generate".to_owned(),
        };

        let response = service
            .handle_check(&ctx, 1, &request)
            .await
            .expect("denial is a policy response");
        assert!(matches!(
            response,
            PolicyResponseVariant::CheckResult(false)
        ));
    }

    #[tokio::test]
    async fn policy_check_uses_verified_context_not_requested_identity() {
        let manager = Arc::new(
            PolicyManager::new_in_memory()
                .await
                .expect("test: policy manager"),
        );
        manager
            .add_policy_with_domain(
                "alice",
                "tenant-a",
                "model:own",
                "infer.generate",
                "allow",
            )
            .await
            .expect("test: own-tenant grant");
        manager
            .add_policy_with_domain(
                "victim",
                "tenant-b",
                "model:foreign",
                "infer.generate",
                "allow",
            )
            .await
            .expect("test: foreign-tenant grant");
        let (service, _root) = test_service_with_manager(manager).await;
        let ctx = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("alice"),
            "tenant-a",
            SigningKey::from_bytes(&[0x54; 32]).verifying_key(),
        );

        let forged_probe = PolicyCheck {
            subject: "victim".to_owned(),
            domain: "tenant-b".to_owned(),
            resource: "model:foreign".to_owned(),
            operation: "infer.generate".to_owned(),
        };
        let denied = service
            .handle_check(&ctx, 1, &forged_probe)
            .await
            .expect("denial is a policy response");
        assert!(matches!(
            denied,
            PolicyResponseVariant::CheckResult(false)
        ));

        let stale_wire_identity = PolicyCheck {
            subject: "victim".to_owned(),
            domain: "tenant-b".to_owned(),
            resource: "model:own".to_owned(),
            operation: "infer.generate".to_owned(),
        };
        let allowed = service
            .handle_check(&ctx, 2, &stale_wire_identity)
            .await
            .expect("allow is a policy response");
        assert!(matches!(
            allowed,
            PolicyResponseVariant::CheckResult(true)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn service_mediated_check_uses_verified_user_not_broad_deputy() {
        use hyprstream_service::{InprocManager, ServiceManager};

        crate::mac::install_explicit_test_dispatch_pep();
        let _ = hyprstream_rpc::envelope::install_verify_config(
            hyprstream_rpc::envelope::EnvelopeVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        let _ = hyprstream_rpc::envelope::install_response_verify_config(
            hyprstream_rpc::envelope::ResponseVerifyConfig {
                policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
                pq_store: None,
            },
        );
        // Independently runnable: RPC-path token verification fails closed
        // without a process-global revocation store. Publish an isolated empty
        // store when none exists (guarded, matching the convention in
        // middleware/oauth::revocation tests) so this security test does not
        // depend on a sibling publishing the store first.
        if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
            let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
            ));
        }

        let manager = Arc::new(
            PolicyManager::new_in_memory()
                .await
                .expect("test: policy manager"),
        );
        manager
            .add_policy_with_domain(
                "alice",
                "did:web:tenant-a.example",
                "model:allowed",
                "infer.generate",
                "allow",
            )
            .await
            .expect("test: tenant user grant");
        manager
            .add_policy_with_domain(
                "service:registry",
                "*",
                "model:deputy-only",
                "infer.generate",
                "allow",
            )
            .await
            .expect("test: broad deputy grant");
        let allowed_federation =
            crate::auth::federation_registration_resource("https://allowed.example")
                .expect("test: federation resource");
        let blocked_federation =
            crate::auth::federation_registration_resource("https://blocked.example")
                .expect("test: federation resource");
        manager
            .add_policy_with_domain(
                "*",
                "*",
                &allowed_federation,
                "check",
                "allow",
            )
            .await
            .expect("test: per-origin federation grant");

        let endpoint = format!("policy-delegation-{}", rand::random::<u64>());
        let policy_key = SigningKey::from_bytes(&[0x70; 32]);
        let actor_key = SigningKey::from_bytes(&[0x71; 32]);
        let actor_verifying_key = actor_key.verifying_key();
        let user_key = SigningKey::from_bytes(&[0x72; 32]);
        let other_user_key = SigningKey::from_bytes(&[0x73; 32]);
        let issuer = "https://policy-delegation.test";
        let key_source = hyprstream_rpc::auth::ClusterKeySource::new(
            policy_key.verifying_key(),
            issuer.to_owned(),
        );
        let root = tempfile::tempdir().expect("test: policy git directory");
        let git2db = Arc::new(RwLock::new(
            Git2DB::open(root.path())
                .await
                .expect("test: open policy git database"),
        ));
        let service = PolicyService::new(
            manager,
            Arc::new(policy_key.clone()),
            crate::config::TokenConfig::default(),
            git2db,
            TransportConfig::inproc(&endpoint),
        )
        .with_jwt_key_source(Arc::new(key_source));
        let services = InprocManager::new();
        services
            .spawn(Box::new(service))
            .await
            .expect("test: spawn policy service");

        let now = chrono::Utc::now().timestamp();
        hyprstream_service::global_trust_store().insert(
            actor_verifying_key,
            hyprstream_service::Attestation {
                scopes: ["registry".to_owned()].into_iter().collect(),
                subject: None,
                jwt: None,
                expires_at: now + 300,
                attested_by: Some(policy_key.verifying_key().to_bytes()),
            },
        );
        let user_token = hyprstream_rpc::auth::jwt::encode(
            &hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 300)
                .with_issuer(issuer.to_owned())
                .with_tenant("did:web:tenant-a.example".to_owned())
                .with_cnf_jwk(user_key.verifying_key().as_bytes())
                .with_client_id("hyprstream-oauth-client-1"),
            &policy_key,
        );
        let other_tenant_token = hyprstream_rpc::auth::jwt::encode(
            &hyprstream_rpc::auth::Claims::new("bob".to_owned(), now, now + 300)
                .with_issuer(issuer.to_owned())
                .with_tenant("did:web:tenant-b.example".to_owned())
                .with_cnf_jwk(other_user_key.verifying_key().as_bytes())
                .with_client_id("hyprstream-oauth-client-1"),
            &policy_key,
        );
        let client = crate::services::PolicyClient::for_local_endpoint_bootstrap(
            &format!("inproc://{endpoint}"),
            actor_key,
            policy_key.verifying_key(),
            None,
        )
        .expect("test: policy client");

        let allowed_origin = client
            .check(&PolicyCheck {
                subject: String::new(),
                domain: String::new(),
                resource: allowed_federation,
                operation: "check".to_owned(),
            })
            .await
            .expect("test: allowlisted federation decision");
        assert!(allowed_origin, "the allowlisted origin must be admitted");

        let blocked_origin = client
            .check(&PolicyCheck {
                subject: String::new(),
                domain: String::new(),
                resource: blocked_federation,
                operation: "check".to_owned(),
            })
            .await
            .expect("test: non-allowlisted federation decision");
        assert!(
            !blocked_origin,
            "the decision must vary by origin, not the deputy service"
        );

        let allowed = client
            .clone()
            .with_delegated_bearer(user_token.clone())
            .check(&PolicyCheck {
                subject: "victim".to_owned(),
                domain: "did:web:tenant-b.example".to_owned(),
                resource: "model:allowed".to_owned(),
                operation: "infer.generate".to_owned(),
            })
            .await
            .expect("test: tenant user decision");
        assert!(allowed, "verified delegated user grant must be effective");

        let cross_tenant = client
            .clone()
            .with_delegated_bearer(other_tenant_token)
            .check(&PolicyCheck {
                subject: "alice".to_owned(),
                domain: "did:web:tenant-a.example".to_owned(),
                resource: "model:allowed".to_owned(),
                operation: "infer.generate".to_owned(),
            })
            .await
            .expect("test: cross-tenant decision");
        assert!(
            !cross_tenant,
            "tenant-b's verified bearer must not inherit tenant-a's grant"
        );

        let denied = client
            .with_delegated_bearer(user_token)
            .check(&PolicyCheck {
                subject: "service:registry".to_owned(),
                domain: "*".to_owned(),
                resource: "model:deputy-only".to_owned(),
                operation: "infer.generate".to_owned(),
            })
            .await
            .expect("test: broad deputy decision");
        assert!(
            !denied,
            "the deputy's broad global grant must not replace the delegated user"
        );
        hyprstream_service::global_trust_store().remove(&actor_verifying_key);
    }

    #[tokio::test]
    async fn tenantless_user_policy_check_is_denied() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::for_test_authenticated_subject(
            Subject::new("alice"),
            SigningKey::from_bytes(&[0x55; 32]).verifying_key(),
        );
        let request = PolicyCheck {
            subject: "alice".to_owned(),
            domain: "*".to_owned(),
            resource: "model:any".to_owned(),
            operation: "infer.generate".to_owned(),
        };

        let response = service
            .handle_check(&ctx, 1, &request)
            .await
            .expect("denial is a policy response");
        assert!(matches!(
            response,
            PolicyResponseVariant::CheckResult(false)
        ));
    }

    #[tokio::test]
    async fn cross_domain_token_mint_is_denied_without_target_tenant_grant() {
        let manager = Arc::new(
            PolicyManager::new_in_memory()
                .await
                .expect("test: policy manager"),
        );
        manager
            .add_policy_with_domain(
                "service:test-caller",
                "tenant-a",
                "policy:IssueToken",
                "manage",
                "allow",
            )
            .await
            .expect("test: caller-domain mint grant");
        let (service, _root) = test_service_with_manager(manager).await;
        let signer = SigningKey::from_bytes(&[0x52; 32]).verifying_key();
        let ctx = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("service:test-caller"),
            "tenant-a",
            signer,
        );
        let mut request = issue("victim");
        request.tenant = Some("tenant-b".to_owned());

        let response = service
            .handle_issue_token(&ctx, 1, &request)
            .await
            .expect("cross-domain denial is a policy response");

        match response {
            PolicyResponseVariant::Error(error) => {
                assert_eq!(error.code, "UNAUTHORIZED_SUBJECT");
                assert!(error.message.contains("tenant-b"));
            }
            other => panic!("cross-domain token mint unexpectedly proceeded: {other:?}"),
        }
    }

    fn attestation(expires_at: i64, jwt: &str) -> hyprstream_service::Attestation {
        hyprstream_service::Attestation {
            scopes: std::iter::once("model".to_owned()).collect(),
            subject: None,
            jwt: Some(jwt.to_owned()),
            expires_at,
            attested_by: None,
        }
    }

    #[tokio::test]
    async fn issue_token_rejects_path_form_subject_at_shared_signing_boundary() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        let response = service
            .handle_issue_token(&ctx, 1, &issue("did:web:accounts.example:users:alice"))
            .await
            .expect("path-form rejection is a policy response");

        match response {
            PolicyResponseVariant::Error(error) => {
                assert_eq!(error.code, "FROZEN_PATH_FORM_SUBJECT");
                assert!(error.message.contains("path-form did:web"));
            }
            other => panic!("path-form subject reached token signing: {other:?}"),
        }
    }

    #[tokio::test]
    async fn issue_token_path_form_guard_allows_legitimate_subject_families() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        for subject in [
            "alice",
            "did:web:alice.example",
            "did:web:localhost%3A6791",
            "did:key:z6MkpTHR8VNsBxYAAHWutMGeQ4hz2FV6B14xd9CZpkmS5i5o",
            "service:model",
        ] {
            let response = service.handle_issue_token(&ctx, 1, &issue(subject)).await;
            assert!(
                !matches!(
                    response,
                    Ok(PolicyResponseVariant::Error(ErrorInfo { ref code, .. }))
                        if code == "FROZEN_PATH_FORM_SUBJECT"
                ),
                "legitimate subject {subject:?} was rejected by the path-form guard",
            );
        }
    }

    #[tokio::test]
    async fn interactive_issue_token_rejects_missing_session_id() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");
        let mut request = issue("alice");
        request.issuance_profile = IssueTokenProfile::InteractiveSession;
        request.session_id = None;

        let response = service
            .handle_issue_token(&ctx, 1, &request)
            .await
            .expect("missing-session denial is a policy response");
        assert!(matches!(
            response,
            PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "MISSING_SESSION"
        ));
    }

    #[tokio::test]
    async fn deliberate_noninteractive_rfc_profiles_mint_without_session_id() {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let (service, _root) = test_service().await;
        let user_ed = SigningKey::from_bytes(&[0x8a; 32]);
        let service = service.with_primary_enrollment_resolver(v16_primary_resolver(
            "alice",
            user_ed.verifying_key().to_bytes(),
        ));
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        for profile in [IssueTokenProfile::Rfc8693, IssueTokenProfile::Rfc7523] {
            let mut request = issue("alice");
            request.issuance_profile = profile;
            request.session_id = None;
            request.user_pub_key = Some(URL_SAFE_NO_PAD.encode(user_ed.verifying_key().to_bytes()));
            let response = service
                .handle_issue_token(&ctx, 1, &request)
                .await
                .expect("non-interactive issuance is a policy response");
            // The RFC profiles require no session id: reaching the mint proves
            // the profile gate accepted without one. test_service provisions no
            // composite signing authority, so a cleared gate surfaces as
            // SIGNING_NOT_CONFIGURED rather than a session/profile rejection.
            match response {
                PolicyResponseVariant::IssueTokenResult(_) => {}
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. })
                    if code == "SIGNING_NOT_CONFIGURED" => {}
                other => panic!("RFC profile must pass the gate without a session: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn service_subject_rejects_noninteractive_user_profile() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");
        let mut request = issue("service:model");
        request.issuance_profile = IssueTokenProfile::Rfc8693;

        let response = service
            .handle_issue_token(&ctx, 1, &request)
            .await
            .expect("profile-mismatch denial is a policy response");
        assert!(matches!(
            response,
            PolicyResponseVariant::Error(ErrorInfo { ref code, .. })
                if code == "INVALID_ISSUANCE_PROFILE"
        ));
    }

    /// Process-global in-memory session registry for the issuance-profile
    /// tests (guarded/idempotent). `handle_issue_token` binds a supplied `sid`
    /// against the PROCESS-GLOBAL registry, so the interactive contracts are
    /// proven against that exact handle. `register_session`/`revoke_session`
    /// are trait methods, so registering through the returned `dyn` handle
    /// works regardless of which sibling test published the global first (both
    /// publish an InMemory registry and use distinct keys).
    fn interactive_test_session_registry(
    ) -> &'static std::sync::Arc<dyn hyprstream_rpc::auth::SessionRegistry> {
        if hyprstream_rpc::auth::global_session_registry().is_none() {
            let _ = hyprstream_rpc::auth::set_global_session_registry(std::sync::Arc::new(
                hyprstream_rpc::auth::InMemorySessionRegistry::new(),
            ));
        }
        hyprstream_rpc::auth::global_session_registry()
            .expect("session registry is published above")
    }

    fn active_oidc_session(
        subject: &str,
        tenant: &str,
        now: i64,
    ) -> hyprstream_rpc::auth::SessionState {
        hyprstream_rpc::auth::SessionState {
            subject: subject.to_owned(),
            tenant: tenant.to_owned(),
            kind: hyprstream_rpc::auth::SessionKind::Interactive,
            created_at: now,
            expires_at: now + 3600,
            status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
            clearance_epoch: 0,
        }
    }

    /// Contract 1 (empty sid): the interactive profile treats an empty-string
    /// `sid` exactly like a missing one — both deny before minting.
    #[tokio::test]
    async fn interactive_issue_token_rejects_empty_session_id() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");
        let mut request = issue("alice");
        request.issuance_profile = IssueTokenProfile::InteractiveSession;
        request.session_id = Some(String::new());
        let response = service
            .handle_issue_token(&ctx, 1, &request)
            .await
            .expect("empty-session denial is a policy response");
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "MISSING_SESSION"
            ),
            "an empty sid must deny like a missing one: {response:?}"
        );
    }

    /// Contract 3: an interactive credential whose `sid` names a registered,
    /// ACTIVE session bound to this exact subject and tenant mints, and the
    /// registered `sid` is stamped into the signed claims.
    #[tokio::test]
    async fn interactive_issue_token_with_registered_session_succeeds() {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let (service, _root) = test_service().await;
        let user_ed = SigningKey::from_bytes(&[0x89; 32]);
        let service = service.with_primary_enrollment_resolver(v16_primary_resolver(
            "alice",
            user_ed.verifying_key().to_bytes(),
        ));
        let reg = interactive_test_session_registry();
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");
        let issuer = "https://issuer.interactive-ok.test";
        let tenant = "tenant-interactive-ok";
        let sid = "sid-interactive-ok";
        let now = chrono::Utc::now().timestamp();
        reg.register_session(
            hyprstream_rpc::auth::SessionKey::oidc(issuer, sid),
            active_oidc_session("alice", tenant, now),
        )
        .await
        .expect("register active session");

        let mut request = issue("alice");
        request.issuance_profile = IssueTokenProfile::InteractiveSession;
        request.issuer = Some(issuer.to_owned());
        request.tenant = Some(tenant.to_owned());
        request.session_id = Some(sid.to_owned());
        request.user_pub_key = Some(URL_SAFE_NO_PAD.encode(user_ed.verifying_key().to_bytes()));
        let response = service
            .handle_issue_token(&ctx, 1, &request)
            .await
            .expect("interactive issuance is a policy response");
        match response {
            PolicyResponseVariant::IssueTokenResult(info) => {
                let claims = hyprstream_rpc::auth::decode_unverified(&info.token)
                    .expect("issued token decodes");
                assert_eq!(
                    claims.sid.as_deref(),
                    Some(sid),
                    "the registered sid is stamped into the minted claims"
                );
            }
            // test_service provisions no global composite signing authority, so
            // a mint that clears the session gate fails at the signing boundary
            // instead. Reaching it proves the correctly-registered session was
            // ACCEPTED (the contract here); the sid-stamping assertion above is
            // exercised whenever the full-suite run provisions signing.
            PolicyResponseVariant::Error(ErrorInfo { ref code, .. })
                if code == "SIGNING_NOT_CONFIGURED" => {}
            other => panic!("a correctly-registered interactive session was rejected: {other:?}"),
        }
    }

    /// Contract 2: an interactive `sid` that is unknown, revoked, or bound to a
    /// different subject/tenant is rejected — the registry is the authority and
    /// there is no session fixation.
    #[tokio::test]
    async fn interactive_issue_token_denies_unknown_revoked_or_mismatched_session() {
        let (service, _root) = test_service().await;
        let reg = interactive_test_session_registry();
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");
        let issuer = "https://issuer.interactive-deny.test";
        let tenant = "tenant-interactive-deny";
        let now = chrono::Utc::now().timestamp();

        let request_for = |sid: &str| {
            let mut r = issue("alice");
            r.issuance_profile = IssueTokenProfile::InteractiveSession;
            r.issuer = Some(issuer.to_owned());
            r.tenant = Some(tenant.to_owned());
            r.session_id = Some(sid.to_owned());
            r
        };
        let assert_denied = |response: PolicyResponseVariant, label: &str| {
            assert!(
                matches!(
                    response,
                    PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "INVALID_SESSION"
                ),
                "{label} must deny with INVALID_SESSION: {response:?}"
            );
        };

        // Unknown: never registered.
        assert_denied(
            service
                .handle_issue_token(&ctx, 1, &request_for("never-registered"))
                .await
                .unwrap(),
            "an unknown sid",
        );

        // Revoked: registered then revoked.
        let revoked = "sid-revoked";
        reg.register_session(
            hyprstream_rpc::auth::SessionKey::oidc(issuer, revoked),
            active_oidc_session("alice", tenant, now),
        )
        .await
        .unwrap();
        reg.revoke_session(&hyprstream_rpc::auth::SessionKey::oidc(issuer, revoked))
            .await
            .unwrap();
        assert_denied(
            service
                .handle_issue_token(&ctx, 2, &request_for(revoked))
                .await
                .unwrap(),
            "a revoked sid",
        );

        // Mismatched subject: session bound to a different user.
        let cross_subject = "sid-cross-subject";
        reg.register_session(
            hyprstream_rpc::auth::SessionKey::oidc(issuer, cross_subject),
            active_oidc_session("bob", tenant, now),
        )
        .await
        .unwrap();
        assert_denied(
            service
                .handle_issue_token(&ctx, 3, &request_for(cross_subject))
                .await
                .unwrap(),
            "a cross-subject sid",
        );

        // Mismatched tenant: session bound to a different tenant.
        let cross_tenant = "sid-cross-tenant";
        reg.register_session(
            hyprstream_rpc::auth::SessionKey::oidc(issuer, cross_tenant),
            active_oidc_session("alice", "some-other-tenant", now),
        )
        .await
        .unwrap();
        assert_denied(
            service
                .handle_issue_token(&ctx, 4, &request_for(cross_tenant))
                .await
                .unwrap(),
            "a cross-tenant sid",
        );
    }

    /// Contract 4: a service subject can never acquire a user session — neither
    /// under the service profile carrying a `sid`, nor by relabeling itself
    /// interactive. Both deny at the profile gate before any registry lookup.
    #[tokio::test]
    async fn service_subject_with_user_session_is_denied() {
        let (service, _root) = test_service().await;
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        let mut service_profile = issue("service:model");
        service_profile.issuance_profile = IssueTokenProfile::Service;
        service_profile.session_id = Some("sid-illegal".to_owned());
        let response = service
            .handle_issue_token(&ctx, 1, &service_profile)
            .await
            .unwrap();
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "INVALID_ISSUANCE_PROFILE"
            ),
            "service profile + sid must deny: {response:?}"
        );

        let mut relabeled = issue("service:model");
        relabeled.issuance_profile = IssueTokenProfile::InteractiveSession;
        relabeled.session_id = Some("sid-illegal".to_owned());
        let response = service.handle_issue_token(&ctx, 2, &relabeled).await.unwrap();
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "INVALID_ISSUANCE_PROFILE"
            ),
            "a service subject relabeled interactive must deny: {response:?}"
        );
    }

    /// Contract 5 (negative): the non-interactive RFC profiles cannot smuggle a
    /// session, and a non-interactive exchange cannot relabel itself
    /// interactive to bypass session verification — a fabricated (unregistered)
    /// sid denies, and an absent sid denies for want of a registered session.
    #[tokio::test]
    async fn noninteractive_profiles_cannot_smuggle_or_relabel_a_session() {
        let (service, _root) = test_service().await;
        // Ensure the registry is initialized so the relabel case reaches
        // INVALID_SESSION rather than an UNAVAILABLE registry error.
        let _ = interactive_test_session_registry();
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        for profile in [IssueTokenProfile::Rfc8693, IssueTokenProfile::Rfc7523] {
            let mut smuggle = issue("alice");
            smuggle.issuance_profile = profile;
            smuggle.session_id = Some("smuggled-sid".to_owned());
            let response = service.handle_issue_token(&ctx, 1, &smuggle).await.unwrap();
            assert!(
                matches!(
                    response,
                    PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "INVALID_SESSION"
                ),
                "an RFC profile carrying a sid must deny: {response:?}"
            );
        }

        // Relabel to interactive with a fabricated, unregistered sid → denied.
        let mut fabricated = issue("alice");
        fabricated.issuance_profile = IssueTokenProfile::InteractiveSession;
        fabricated.issuer = Some("https://issuer.relabel.test".to_owned());
        fabricated.session_id = Some("fabricated-unregistered".to_owned());
        let response = service
            .handle_issue_token(&ctx, 2, &fabricated)
            .await
            .unwrap();
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "INVALID_SESSION"
            ),
            "relabel with a fabricated sid must deny: {response:?}"
        );

        // Relabel to interactive with no sid → denied for want of a session.
        let mut no_sid = issue("alice");
        no_sid.issuance_profile = IssueTokenProfile::InteractiveSession;
        no_sid.session_id = None;
        let response = service.handle_issue_token(&ctx, 3, &no_sid).await.unwrap();
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "MISSING_SESSION"
            ),
            "relabel without a sid must deny: {response:?}"
        );
    }

    /// Gate 2 (v16 §3.3): a carried `workload_session_id` must be ACTIVE before
    /// a service credential may renew. A revoked family session DENIES the
    /// renewal — the revocation is checked BEFORE the enrollment-policy branch,
    /// so removing the workload-family policy (family narrowing) can never let
    /// a revoked family renew into an unsessioned credential.
    #[tokio::test]
    async fn revoked_workload_session_denies_service_renewal() {
        let (service, _root) = test_service().await;
        let service = service.with_default_audience("https://issuer.gate2.test".to_owned());
        let issuer = "https://issuer.gate2.test";
        let reg = interactive_test_session_registry();
        let now = chrono::Utc::now().timestamp();
        let wsid = "wl-gate2-revoked";

        // Register then revoke the workload family session.
        let workload_key = hyprstream_rpc::auth::SessionKey::workload(issuer, wsid);
        reg.register_session(
            workload_key.clone(),
            hyprstream_rpc::auth::SessionState {
                subject: "service:model".to_owned(),
                tenant: "tenant-a".to_owned(),
                kind: hyprstream_rpc::auth::SessionKind::Workload,
                created_at: now,
                expires_at: now + 3600,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await
        .unwrap();
        reg.revoke_session(&workload_key).await.unwrap();

        // The renewing caller's key must be trust-registered for the service.
        let caller = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        let trust = hyprstream_service::global_trust_store();
        trust.insert(
            caller,
            hyprstream_service::Attestation {
                scopes: std::iter::once("model".to_owned()).collect(),
                subject: None,
                jwt: Some("gate2-renew".to_owned()),
                expires_at: now + 300,
                attested_by: None,
            },
        );

        // The presented credential carries the now-revoked workload session.
        let carried =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 3600)
                .with_workload_session_id(wsid);
        let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new("service:model"),
            "tenant-a",
            caller,
            carried,
        );

        let response = service
            .handle_refresh_service_token(
                &ctx,
                1,
                &RefreshServiceTokenRequest { ttl_seconds: 3600 },
            )
            .await
            .expect("refresh is a policy response");
        trust.remove(&caller);
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "SESSION_REVOKED"
            ),
            "a carried revoked workload session must deny renewal: {response:?}"
        );
    }

    /// Gate 2 check-BEFORE-narrowing (v16 §3.3): the sibling of the test above,
    /// closing the case that one could not reach. Here an authoritative manifest
    /// enrolls the family with `workload_session = false` (policy REMOVED), so
    /// `family_allowed` is `false` and renewal takes the deliberate-narrowing
    /// branch `(Some(wsid), false)` — the session ID is dropped, not re-stamped.
    /// A revoked family session must STILL deny the renewal: the revocation is
    /// checked before the narrowing branch, so removing the family policy can
    /// never launder a revoked family into an unsessioned credential. The prior
    /// test only reaches `(Some(wsid), true)` because, with no manifest,
    /// `family_allowed` defaults to `true`.
    #[tokio::test]
    async fn revoked_session_with_removed_family_policy_still_denies_renewal() {
        // The caller/enrolled key is shared: the renewal path resolves the
        // signer suite against the injected manifest (the block above the
        // workload branch), which requires the verified `cnf` key to equal the
        // enrolled ed25519 key. Match them so renewal reaches the workload gate.
        let caller_sk = SigningKey::from_bytes(&[0x2b; 32]);
        let caller = caller_sk.verifying_key();

        let (service, _root) = test_service().await;
        let service = service
            .with_default_audience("https://issuer.gate2.test".to_owned())
            // Authoritative family policy REMOVED: `v16_service_manifest` enrolls
            // "model" with `workload_session = false`. Injected (not global) so
            // the process-global manifest is untouched — the renewal path now
            // reads this same injected authority for clearance, signer-suite,
            // AND the workload-family policy.
            .with_enrollment_manifest(v16_service_manifest("model", caller.to_bytes()));
        let issuer = "https://issuer.gate2.test";
        let reg = interactive_test_session_registry();
        let now = chrono::Utc::now().timestamp();
        let wsid = "wl-gate2-narrowed-revoked";

        // Register then revoke the workload family session.
        let workload_key = hyprstream_rpc::auth::SessionKey::workload(issuer, wsid);
        reg.register_session(
            workload_key.clone(),
            hyprstream_rpc::auth::SessionState {
                subject: "service:model".to_owned(),
                tenant: "tenant-a".to_owned(),
                kind: hyprstream_rpc::auth::SessionKind::Workload,
                created_at: now,
                expires_at: now + 3600,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await
        .unwrap();
        reg.revoke_session(&workload_key).await.unwrap();

        // The renewing caller's key must be trust-registered for the service.
        let trust = hyprstream_service::global_trust_store();
        trust.insert(
            caller,
            hyprstream_service::Attestation {
                scopes: std::iter::once("model".to_owned()).collect(),
                subject: None,
                jwt: Some("gate2-narrow-renew".to_owned()),
                expires_at: now + 300,
                attested_by: None,
            },
        );

        // The presented credential carries the now-revoked workload session.
        let carried =
            hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 3600)
                .with_workload_session_id(wsid);
        let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new("service:model"),
            "tenant-a",
            caller,
            carried,
        );

        let response = service
            .handle_refresh_service_token(
                &ctx,
                1,
                &RefreshServiceTokenRequest { ttl_seconds: 3600 },
            )
            .await
            .expect("refresh is a policy response");
        trust.remove(&caller);
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "SESSION_REVOKED"
            ),
            "removing the family policy must not launder a revoked session into an \
             unsessioned renewal — revocation is checked before narrowing: {response:?}"
        );
    }

    /// Testability guard for the two tests above: proves the renewal path reads
    /// the INJECTED (`self.enrollment()`) manifest, not `global_service_enrollment()`
    /// alone. Without this routing, an injected `workload_session = false` family
    /// could never take effect in an isolated test (the process-global manifest
    /// is unset), so `family_allowed` would silently default to `true` and the
    /// narrowing branch above would be unreachable — a false green.
    ///
    /// The observable: an injected manifest enrolls "model" with an ed25519 key
    /// that DIFFERS from the renewing caller's verified `cnf` key. The renewal
    /// signer-suite block (which reads the same injected binding) then denies
    /// with `SIGNER_SUITE_UNAVAILABLE`. The pre-routing code read only the unset
    /// global manifest, skipped that block entirely, and would have reached the
    /// signing boundary (`SIGNING_NOT_CONFIGURED`) instead — so this exact code
    /// distinguishes "injected manifest consulted" from "not consulted".
    #[tokio::test]
    async fn service_renewal_consults_injected_enrollment_manifest() {
        let caller = SigningKey::from_bytes(&[0x2b; 32]).verifying_key();
        // Enroll "model" under a DIFFERENT key than the caller's verified cnf.
        let enrolled_other = SigningKey::from_bytes(&[0x5c; 32]).verifying_key();

        let (service, _root) = test_service().await;
        let service = service
            .with_default_audience("https://issuer.gate2.test".to_owned())
            .with_enrollment_manifest(v16_service_manifest("model", enrolled_other.to_bytes()));
        let now = chrono::Utc::now().timestamp();

        let trust = hyprstream_service::global_trust_store();
        trust.insert(
            caller,
            hyprstream_service::Attestation {
                scopes: std::iter::once("model".to_owned()).collect(),
                subject: None,
                jwt: Some("gate2-inject-renew".to_owned()),
                expires_at: now + 300,
                attested_by: None,
            },
        );

        let ctx = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("service:model"),
            "tenant-a",
            caller,
        );
        let response = service
            .handle_refresh_service_token(
                &ctx,
                1,
                &RefreshServiceTokenRequest { ttl_seconds: 3600 },
            )
            .await
            .expect("refresh is a policy response");
        trust.remove(&caller);
        assert!(
            matches!(
                response,
                PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "SIGNER_SUITE_UNAVAILABLE"
            ),
            "renewal must consult the injected manifest and reject a caller key that \
             disagrees with the enrolled key: {response:?}"
        );
    }

    /// The DEFINITIVE causal coverage the two handler-level tests above cannot
    /// observe (`test_service` provisions no signing authority, so a cleared
    /// renewal stops at `SIGNING_NOT_CONFIGURED` before any token is minted).
    /// This exercises `resolve_renewal_workload_session` directly and inspects
    /// the exact disposition, proving the `(Some(wsid), false)` narrowing branch
    /// genuinely OMITS the session — and that the revocation check runs BEFORE
    /// narrowing, with revoked/expired/unknown all failing closed.
    #[tokio::test]
    async fn renewal_workload_decision_narrows_and_fails_closed() {
        let (service, _root) = test_service().await;
        let reg = interactive_test_session_registry();
        let now = chrono::Utc::now().timestamp();
        // A per-test issuer keeps these session keys off any other test's keys
        // in the shared global registry.
        let issuer = "https://issuer.narrow-decision.test";
        let active_state = |exp: i64| hyprstream_rpc::auth::SessionState {
            subject: "service:model".to_owned(),
            tenant: "tenant-a".to_owned(),
            kind: hyprstream_rpc::auth::SessionKind::Workload,
            created_at: now,
            expires_at: exp,
            status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
            clearance_epoch: 0,
        };
        let decide = |fp: Option<bool>, wsid: Option<&'static str>| {
            service.resolve_renewal_workload_session(issuer, "service:model", "tenant-a", now, fp, wsid)
        };

        // Register one ACTIVE session used by the two contrasting live cases.
        let active = "wl-active-narrow";
        reg.register_session(
            hyprstream_rpc::auth::SessionKey::workload(issuer, active),
            active_state(now + 3600),
        )
        .await
        .unwrap();

        // (1) OMISSION CONTROL — ACTIVE session, family policy REMOVED
        // (`Some(false)`): the LIVE session is dropped from the renewed
        // credential. This is the positive proof that `family_allowed = false`
        // actually narrows (not a vacuous always-None).
        assert!(
            matches!(decide(Some(false), Some(active)).await, RenewalWorkloadSession::Stamp(None)),
            "active session + removed family policy must OMIT the workload_session_id"
        );

        // (2) CONTRAST — the SAME active session, family still ENROLLED
        // (`Some(true)`): the session is re-stamped. Proves the false branch in
        // (1) is a real divergence, not a constant.
        assert!(
            matches!(
                decide(Some(true), Some(active)).await,
                RenewalWorkloadSession::Stamp(Some(ref w)) if w == active
            ),
            "active session + enrolled family must RE-STAMP the workload_session_id"
        );

        // (3) CHECK-BEFORE-NARROWING — a REVOKED session denies for EVERY family
        // policy, including the narrowing `Some(false)`: removing the policy can
        // never launder a revoked family into an unsessioned renewal.
        let revoked = "wl-revoked-narrow";
        let rk = hyprstream_rpc::auth::SessionKey::workload(issuer, revoked);
        reg.register_session(rk.clone(), active_state(now + 3600)).await.unwrap();
        reg.revoke_session(&rk).await.unwrap();
        for fp in [Some(false), Some(true), None] {
            assert!(
                matches!(
                    decide(fp, Some(revoked)).await,
                    RenewalWorkloadSession::Deny { code, .. } if code == "SESSION_REVOKED"
                ),
                "a revoked carried session must deny renewal for family policy {fp:?}"
            );
        }

        // (4) FAIL-CLOSED, unknown — a session that was never registered denies.
        assert!(
            matches!(
                decide(Some(true), Some("wl-never-registered-narrow")).await,
                RenewalWorkloadSession::Deny { code, .. } if code == "SESSION_REVOKED"
            ),
            "an unknown carried session must fail closed"
        );

        // (5) FAIL-CLOSED, expired — an expired-but-Active session denies.
        let expired = "wl-expired-narrow";
        reg.register_session(
            hyprstream_rpc::auth::SessionKey::workload(issuer, expired),
            active_state(now - 1),
        )
        .await
        .unwrap();
        assert!(
            matches!(
                decide(Some(true), Some(expired)).await,
                RenewalWorkloadSession::Deny { code, .. } if code == "SESSION_REVOKED"
            ),
            "an expired carried session must fail closed"
        );

        // (6) No carried session: a disabled (`Some(false)`) or legacy (`None`)
        // family manufactures none.
        assert!(
            matches!(decide(Some(false), None).await, RenewalWorkloadSession::Stamp(None)),
            "no session + disabled family stays unsessioned"
        );
        assert!(
            matches!(decide(None, None).await, RenewalWorkloadSession::Stamp(None)),
            "no session + legacy family stays unsessioned"
        );

        // (7) No carried session, family ENROLLED (`Some(true)`): a fresh session
        // is created, registered ACTIVE, and stamped.
        let created = decide(Some(true), None).await;
        let RenewalWorkloadSession::Stamp(Some(new_wsid)) = created else {
            panic!("an enrolled family with no carried session must create one: {created:?}");
        };
        assert!(
            !reg
                .is_revoked(&hyprstream_rpc::auth::SessionKey::workload(issuer, &new_wsid))
                .await,
            "the freshly created workload session must be registered ACTIVE"
        );
    }

    /// RFC 9068 §2.2.1 issuance (v16 credential profile): a user `at+jwt`
    /// profile without a non-empty `client_id` is denied; a service profile
    /// carrying a `client_id` is denied; and a supplied `client_id` is stamped
    /// into the minted `at+jwt` claims (the emitter positive).
    #[tokio::test]
    async fn issuance_enforces_and_stamps_client_id() {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let (service, _root) = test_service().await;
        // v16: a user at+jwt binds its authoritative Primary suite. Install the
        // resolver + a matching cnf key so the positive reaches the signing
        // boundary rather than the fail-closed Primary gate.
        let user_ed = SigningKey::from_bytes(&[0x88; 32]);
        let service = service
            .with_primary_enrollment_resolver(v16_primary_resolver(
                "alice",
                user_ed.verifying_key().to_bytes(),
            ));
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        // User profile (RFC 8693) with a MISSING client_id → denied.
        let mut missing = issue("alice");
        missing.issuance_profile = IssueTokenProfile::Rfc8693;
        missing.client_id = None;
        let resp = service.handle_issue_token(&ctx, 1, &missing).await.unwrap();
        assert!(
            matches!(resp, PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "MISSING_CLIENT_ID"),
            "missing client_id: {resp:?}"
        );

        // Empty/whitespace client_id → denied.
        let mut empty = issue("alice");
        empty.issuance_profile = IssueTokenProfile::Rfc8693;
        empty.client_id = Some("   ".to_owned());
        let resp = service.handle_issue_token(&ctx, 2, &empty).await.unwrap();
        assert!(
            matches!(resp, PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "MISSING_CLIENT_ID"),
            "empty client_id: {resp:?}"
        );

        // Service (`wit+jwt`) profile carrying a client_id → denied.
        let mut svc_cid = issue("service:model");
        svc_cid.issuance_profile = IssueTokenProfile::Service;
        svc_cid.client_id = Some("hyprstream-oauth-client-1".to_owned());
        let resp = service.handle_issue_token(&ctx, 3, &svc_cid).await.unwrap();
        assert!(
            matches!(resp, PolicyResponseVariant::Error(ErrorInfo { ref code, .. }) if code == "INVALID_CLIENT_ID"),
            "service + client_id: {resp:?}"
        );

        // Emitter positive: a supplied client_id is stamped into the minted
        // at+jwt (test_service may stop at the signing boundary; assert the
        // stamped value only when a token is actually produced).
        let mut ok = issue("alice");
        ok.issuance_profile = IssueTokenProfile::Rfc8693;
        ok.client_id = Some("hyprstream-oauth-client-1".to_owned());
        ok.user_pub_key = Some(URL_SAFE_NO_PAD.encode(user_ed.verifying_key().to_bytes()));
        match service.handle_issue_token(&ctx, 4, &ok).await.unwrap() {
            PolicyResponseVariant::IssueTokenResult(info) => {
                let claims = hyprstream_rpc::auth::decode_unverified(&info.token)
                    .expect("issued token decodes");
                assert_eq!(
                    claims.client_id.as_deref(),
                    Some("hyprstream-oauth-client-1"),
                    "the supplied client_id is stamped into the minted claims"
                );
            }
            // test_service configures no signing pair, so the flow stops at the
            // signing boundary AFTER passing the v16 Primary/client_id gates.
            PolicyResponseVariant::Error(ErrorInfo { ref code, .. })
                if code == "SIGNING_NOT_CONFIGURED" => {}
            other => panic!("unexpected issuance response: {other:?}"),
        }
    }

    /// v16 fail-closed: a DISPATCH-CAPABLE user `at+jwt` (recoverable cnf.jwk)
    /// whose authoritative Primary cannot be resolved is denied before signing —
    /// no resolver installed, or a resolver whose key does not match the cnf.
    #[tokio::test]
    async fn dispatch_capable_user_issuance_without_primary_fails_closed() {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let user_ed = SigningKey::from_bytes(&[0x91; 32]);
        let user_pk = URL_SAFE_NO_PAD.encode(user_ed.verifying_key().to_bytes());
        let ctx = EnvelopeContext::from_callback_service(1, "test-caller");

        // No resolver installed → fail closed.
        let (service, _root) = test_service().await;
        let mut req = issue("alice");
        req.issuance_profile = IssueTokenProfile::Rfc8693;
        req.user_pub_key = Some(user_pk.clone());
        let resp = service.handle_issue_token(&ctx, 1, &req).await.unwrap();
        assert_eq!(
            err_code_of(&resp),
            Some("PRIMARY_RESOLVER_UNAVAILABLE"),
            "no resolver must fail closed: {resp:?}"
        );

        // Resolver returns a DIFFERENT key than the presented cnf → fail closed.
        let (service, _root) = test_service().await;
        let other = SigningKey::from_bytes(&[0x92; 32]);
        let service = service.with_primary_enrollment_resolver(v16_primary_resolver(
            "alice",
            other.verifying_key().to_bytes(),
        ));
        let mut req = issue("alice");
        req.issuance_profile = IssueTokenProfile::Rfc8693;
        req.user_pub_key = Some(user_pk);
        let resp = service.handle_issue_token(&ctx, 2, &req).await.unwrap();
        assert_eq!(
            err_code_of(&resp),
            Some("PRIMARY_UNRESOLVABLE"),
            "cnf/primary key mismatch must fail closed: {resp:?}"
        );
    }

    fn err_code_of(resp: &PolicyResponseVariant) -> Option<&str> {
        match resp {
            PolicyResponseVariant::Error(ErrorInfo { code, .. }) => Some(code.as_str()),
            _ => None,
        }
    }

    #[test]
    fn resolve_service_key_publishes_every_overlap_candidate() {
        let trust = hyprstream_service::TrustStore::new();
        let retired = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        let lead = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        let now = chrono::Utc::now().timestamp();
        trust.insert(retired, attestation(now + 60, "retired-attestation"));
        trust.insert(lead, attestation(0, "lead-attestation"));

        let response = match published_service_key_response(&trust, "model") {
            Ok(response) => response,
            Err(error) => panic!("key set: {error}"),
        };
        assert!(response.verifying_key.is_empty(), "no singleton fallback");
        assert!(response.service_jwt.is_none(), "no singleton attestation fallback");
        assert_eq!(response.keys.len(), 2, "overlap keys must both be published");
        assert!(response.keys.iter().any(|entry| entry.verifying_key == retired.to_bytes()));
        assert!(response.keys.iter().any(|entry| entry.verifying_key == lead.to_bytes()));
        assert!(response.keys.iter().all(|entry| entry.key_id.starts_with("ed25519:")));
    }

    #[test]
    fn registration_requires_matching_present_confirmation_key() {
        let key = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new("service:model".to_owned(), now, now + 60);
        assert!(validate_service_key_registration(&claims, "model", key.as_bytes()).is_err());

        let bound = claims.with_cnf_jwk(key.as_bytes());
        assert!(validate_service_key_registration(&bound, "model", key.as_bytes()).is_ok());

        let sibling = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        assert!(validate_service_key_registration(&bound, "model", sibling.as_bytes()).is_err());
    }

    #[test]
    fn one_key_response_keeps_legacy_projection_during_rollout() {
        let trust = hyprstream_service::TrustStore::new();
        let key = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        trust.insert(key, attestation(chrono::Utc::now().timestamp() + 60, "certificate"));

        let response = match published_service_key_response(&trust, "model") {
            Ok(response) => response,
            Err(error) => panic!("one-key response: {error}"),
        };
        assert_eq!(response.keys.len(), 1);
        assert_eq!(response.verifying_key, key.to_bytes());
        assert_eq!(response.service_jwt.as_deref(), Some("certificate"));
    }

    #[tokio::test]
    async fn issue_token_returns_structured_errors_for_invalid_service_keys() {
        let (service, _root) = test_service().await;
        let mut ctx = EnvelopeContext::from_callback_service(1, "rotation-error-test");
        ctx.cnf = SigningKey::generate(&mut rand::rngs::OsRng)
            .verifying_key()
            .to_bytes();

        let mut request = issue("service:rotation-error-test");
        request.issuance_profile = IssueTokenProfile::Service;
        request.client_id = None;
        request.user_pub_key = Some("not-base64url!".to_owned());
        let malformed = service
            .handle_issue_token(&ctx, 1, &request)
            .await
            .expect("malformed assertion key is a policy response");
        assert!(matches!(
            malformed,
            PolicyResponseVariant::Error(ErrorInfo { ref code, .. })
                if code == "INVALID_ASSERTION_KEY"
        ));

        let requested = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        request.user_pub_key = Some(URL_SAFE_NO_PAD.encode(requested.as_bytes()));
        let unauthorized = service
            .handle_issue_token(&ctx, 2, &request)
            .await
            .expect("unregistered sibling is a policy response");
        assert!(matches!(
            unauthorized,
            PolicyResponseVariant::Error(ErrorInfo { ref code, .. })
                if code == "UNAUTHORIZED_SERVICE_KEY"
        ));
    }

    #[tokio::test]
    async fn registered_sibling_reaches_service_token_signing_boundary() {
        let (service, _root) = test_service().await;
        let caller = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        let requested = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();
        // v16: the service credential binds its enrolled signer suite; enroll
        // the service to the attested assertion key so the suite resolves.
        let service = service.with_enrollment_manifest(v16_service_manifest(
            "rotation-sibling-test",
            requested.to_bytes(),
        ));
        let trust = hyprstream_service::global_trust_store();
        trust.insert(caller, hyprstream_service::Attestation {
            scopes: std::iter::once("rotation-sibling-test".to_owned()).collect(),
            subject: None,
            jwt: Some("registered-sibling".to_owned()),
            expires_at: chrono::Utc::now().timestamp() + 60,
            attested_by: None,
        });

        let mut ctx = EnvelopeContext::from_callback_service(1, "rotation-sibling-test");
        ctx.cnf = caller.to_bytes();
        let mut request = issue("service:rotation-sibling-test");
        request.issuance_profile = IssueTokenProfile::Service;
        request.client_id = None;
        request.user_pub_key = Some(URL_SAFE_NO_PAD.encode(requested.as_bytes()));
        let response = service.handle_issue_token(&ctx, 1, &request).await;
        trust.remove(&caller);

        match response.expect("registered sibling reaches the signing boundary") {
            PolicyResponseVariant::IssueTokenResult(info) => {
                let claims = hyprstream_rpc::auth::decode_unverified(&info.token)
                    .expect("issued service token decodes");
                assert_eq!(claims.cnf_key_bytes(), Some(requested.to_bytes()));
            }
            PolicyResponseVariant::Error(ErrorInfo { code, .. })
                if code == "SIGNING_NOT_CONFIGURED" => {}
            other => panic!("registered sibling was rejected before signing: {other:?}"),
        }
    }
}

#[cfg(test)]
mod event_prefix_registry_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn state(owner: &str, key: u8) -> EventPrefixState {
        EventPrefixState { owner: owner.to_owned(), publisher_pubkey: [key; 32], schema: String::new(), subscriber_pubkeys: HashMap::new(), wrapped_keys: HashMap::new() }
    }

    #[test]
    fn tenant_scoping_refuses_shadowing_and_takeover() {
        let a = EventPrefixKey::new("tenant-a".to_owned(), "orders");
        let b = EventPrefixKey::new("tenant-b".to_owned(), "orders");
        let shadow = EventPrefixKey::new("tenant-b".to_owned(), "orders.created");
        let mut prefixes = HashMap::new();
        validate_event_prefix_registration(&prefixes, &a, "subject-a").unwrap();
        prefixes.insert(a.clone(), state("subject-a", 0x0A));
        validate_event_prefix_registration(&prefixes, &b, "subject-b").unwrap();
        prefixes.insert(b.clone(), state("subject-b", 0x0B));
        assert_eq!(prefixes[&a].publisher_pubkey, [0x0A; 32]);
        assert_eq!(prefixes[&b].publisher_pubkey, [0x0B; 32]);
        assert_eq!(validate_event_prefix_registration(&prefixes, &shadow, "subject-b"), Err(EventPrefixRegistrationError::CrossTenantShadow));
        assert_eq!(validate_event_prefix_registration(&prefixes, &a, "subject-b"), Err(EventPrefixRegistrationError::OwnedByAnotherSubject));
    }
}

// ── ExchangeDelegated (AsOriginator delegated mint) causal proofs ──
//
// Fully parallel-safe: source verification AND the output mint both resolve
// keys from the PolicyService's OWN injected `ClusterKeySource` composite
// ledger (never the process global), the enrollment manifest is injected via
// the cfg(test) setter, and the primary-enrollment resolver + edge authorizer
// are per-service mocks. No process-global state is published or mutated.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod exchange_delegated_tests {
        use super::*;
        use hyprstream_rpc::auth::{
            ClusterKeySource, CompositeKeyPair, CompositeKeySet, CompositePairRole,
            CompositePairState,
        };
        use hyprstream_rpc::auth::mac::{
            Assurance, CompartmentSet, CredentialClearance, Level, SecurityLabel,
        };
        use crate::auth::service_enrollment::{
            ServiceEnrollment, ServiceEnrollmentManifest, SERVICE_ENROLLMENT_VERSION,
        };
        use std::collections::BTreeMap;
        use std::sync::Arc;

        const ISSUER: &str = "https://deleg.test";
        const TENANT: &str = "did:web:acme.example";
        const CLIENT_ID: &str = "hyprstream-oauth-client-1";

        struct Fixture {
            service: PolicyService,
            ml_dsa: Arc<hyprstream_rpc::crypto::pq::MlDsaSigningKey>,
            ed: Arc<SigningKey>,
            actor_ed: SigningKey,
            user_ed: SigningKey,
            // Owned so per-path RAII removes exactly this fixture's scratch tree.
            _root: tempfile::TempDir,
        }

        struct MockAuthorizer {
            allow: bool,
            require_source_aud: Option<String>,
            require_method: Option<String>,
        }
        impl DelegationEdgeAuthorizer for MockAuthorizer {
            fn authorize(&self, edge: &DelegationEdge<'_>) -> bool {
                self.allow
                    && self
                        .require_source_aud
                        .as_deref()
                        .is_none_or(|w| edge.source_audience == Some(w))
                    && self
                        .require_method
                        .as_deref()
                        .is_none_or(|w| edge.target_method_id == Some(w))
                    && edge.target_method_id.is_some()
            }
        }

        /// Authoritative primary resolver mock: returns the configured Ed key for
        /// `subject`, or `None` for anyone else (unknown ⇒ deny).
        struct MockPrimaryResolver {
            subject: String,
            ed25519: [u8; 32],
        }
        impl PrimaryEnrollmentResolver for MockPrimaryResolver {
            fn primary_group(&self, principal: &str, _tenant: &str) -> Option<PrimaryGroup> {
                (principal == self.subject).then(|| PrimaryGroup {
                    suite_id: hyprstream_rpc::auth::SUITE_CLASSICAL_ED25519.to_owned(),
                    ordered_component_keys: vec![self.ed25519.to_vec()],
                })
            }
        }

        fn label() -> SecurityLabel {
            SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY)
        }

        async fn fixture(
            authorizer: Option<Arc<dyn DelegationEdgeAuthorizer>>,
            resolver: Option<Arc<dyn PrimaryEnrollmentResolver>>,
        ) -> Fixture {
            use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
            let ca = SigningKey::from_bytes(&[0x40; 32]);
            let ed = Arc::new(SigningKey::from_bytes(&[0x41; 32]));
            let (ml_dsa_sk, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
            let ml_dsa = Arc::new(ml_dsa_sk);
            let kid = hyprstream_rpc::auth::composite_kid(&ml_dsa_vk, &ed.verifying_key());
            let pair = CompositeKeyPair::signing(
                kid,
                ml_dsa.clone(),
                ed.clone(),
                CompositePairRole::Policy,
                CompositePairState::Active,
                0,
                i64::MAX,
            );
            // The service's OWN isolated authority: source verification AND the
            // output mint both resolve from this exact ledger.
            let root = tempfile::tempdir().unwrap();
            let key_set = Arc::new(CompositeKeySet::default());
            key_set.publish(1, "deleg-fixture".to_owned(), vec![pair]).unwrap();
            // Configure the isolated ledger as a disk-backed minting authority
            // (per-fixture, never the process global) so `sign_token` can mint
            // the output on the SAME ledger it verifies sources against.
            let auth = root.path().join("composite-authority");
            std::fs::create_dir_all(&auth).unwrap();
            let marker =
                serde_json::json!({ "version": 1, "component_digest": "deleg-fixture" });
            let ledger_path = auth.join("ledger.json");
            let committed_path = auth.join("committed.json");
            std::fs::write(&ledger_path, serde_json::to_vec(&marker).unwrap()).unwrap();
            std::fs::write(&committed_path, serde_json::to_vec(&marker).unwrap()).unwrap();
            key_set.configure_authority(
                ledger_path,
                committed_path,
                auth.join("ledger-prefix"),
                auth.join("ledger.lock"),
            );
            let key_source = ClusterKeySource::new(ca.verifying_key(), ISSUER.to_owned())
                .with_composite_key_set(key_set);

            let actor_ed = SigningKey::from_bytes(&[0x42; 32]);
            let user_ed = SigningKey::from_bytes(&[0x43; 32]);

            let mut services = BTreeMap::new();
            services.insert(
                "mcp".to_owned(),
                ServiceEnrollment {
                    ed25519_pubkey: URL_SAFE_NO_PAD.encode(actor_ed.verifying_key().to_bytes()),
                    ml_dsa_pubkey: None,
                    clearance: label(),
                    allowed_audiences: Some(vec!["res-b".to_owned()]),
                    workload_session: false,
                },
            );
            let manifest = Arc::new(ServiceEnrollmentManifest {
                version: SERVICE_ENROLLMENT_VERSION,
                services,
            });

            let git2db = Arc::new(RwLock::new(Git2DB::open(root.path()).await.unwrap()));
            let manager = Arc::new(PolicyManager::permissive().await.unwrap());
            let mut service = PolicyService::new(
                manager,
                Arc::new(ca),
                crate::config::TokenConfig::default(),
                git2db,
                TransportConfig::inproc("deleg-test"),
            )
            .with_default_audience(ISSUER.to_owned())
            .with_jwt_key_source(Arc::new(key_source))
            .with_enrollment_manifest(manifest);
            if let Some(a) = authorizer {
                service = service.with_delegation_edge_authorizer(a);
            }
            if let Some(r) = resolver {
                service = service.with_primary_enrollment_resolver(r);
            }
            Fixture { service, ml_dsa, ed, actor_ed, user_ed, _root: root }
        }

        fn allow_authorizer() -> Arc<dyn DelegationEdgeAuthorizer> {
            Arc::new(MockAuthorizer { allow: true, require_source_aud: None, require_method: None })
        }

        fn user_resolver(user_ed: &SigningKey) -> Arc<dyn PrimaryEnrollmentResolver> {
            Arc::new(MockPrimaryResolver {
                subject: "alice".to_owned(),
                ed25519: user_ed.verifying_key().to_bytes(),
            })
        }

        fn sign_source(fx: &Fixture, claims: &hyprstream_rpc::auth::Claims) -> String {
            crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(claims, &fx.ml_dsa, &fx.ed)
        }

        /// A valid interactive user `at+jwt` source. Its `cnf.hs_signer_suite` is
        /// the suite over the AUTHORITATIVE primary key (the resolver's key), not
        /// a value the handler re-derives from the wire.
        fn user_source_claims(fx: &Fixture, now: i64) -> hyprstream_rpc::auth::Claims {
            let hs = hyprstream_rpc::auth::service_signer_suite_b64(
                &fx.user_ed.verifying_key().to_bytes(),
                None,
            );
            hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
                .with_issuer(ISSUER.to_owned())
                .with_tenant(TENANT.to_owned())
                .with_sid("sess-1")
                .with_client_id(CLIENT_ID)
                .with_scope(Some("read write".to_owned()))
                .with_clearance(label())
                .with_audience(Some("res-a".to_owned()))
                .with_cnf_jwk(&fx.user_ed.verifying_key().to_bytes())
                .with_cnf_hs_signer_suite(hs)
        }

        fn actor_ctx(fx: &Fixture, now: i64) -> EnvelopeContext {
            let actor_claims =
                hyprstream_rpc::auth::Claims::new("service:mcp".to_owned(), now, now + 3600)
                    .with_clearance(label());
            EnvelopeContext::for_test_authenticated_subject_with_claims(
                Subject::new("service:mcp"),
                TENANT,
                fx.actor_ed.verifying_key(),
                actor_claims,
            )
        }

        fn session_registry() -> &'static Arc<dyn hyprstream_rpc::auth::SessionRegistry> {
            if hyprstream_rpc::auth::global_session_registry().is_none() {
                let _ = hyprstream_rpc::auth::set_global_session_registry(Arc::new(
                    hyprstream_rpc::auth::InMemorySessionRegistry::new(),
                ));
            }
            hyprstream_rpc::auth::global_session_registry().expect("session registry published")
        }

        async fn register_active_sid(now: i64) {
            let reg = session_registry();
            let key = hyprstream_rpc::auth::SessionKey::oidc(ISSUER, "sess-1");
            let _ = reg
                .register_session(
                    key,
                    hyprstream_rpc::auth::SessionState {
                        subject: "alice".to_owned(),
                        tenant: TENANT.to_owned(),
                        kind: hyprstream_rpc::auth::SessionKind::Interactive,
                        created_at: now,
                        expires_at: now + 3600,
                        status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                        clearance_epoch: 0,
                    },
                )
                .await;
        }

        fn ensure_revocation_store() {
            if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
                let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                    hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
                ));
            }
        }

        fn request(source: String) -> ExchangeDelegated {
            ExchangeDelegated {
                source_credential: source,
                requested_scopes: Some("read".to_owned()),
                requested_capabilities: None,
                audience: Some("res-b".to_owned()),
                target_method_id: Some("model.Infer".to_owned()),
                ttl: Some(300),
            }
        }

        fn err_code(resp: &PolicyResponseVariant) -> Option<&str> {
            match resp {
                PolicyResponseVariant::Error(ErrorInfo { code, .. }) => Some(code.as_str()),
                _ => None,
            }
        }

        /// Happy path: a valid interactive user source + enrolled actor + allowing
        /// edge + authoritative primary resolver mints a delegated `at+jwt` whose
        /// `jti` is fresh (≠ source), `sub` is the originator, `act` nests the
        /// terminal actor, `cnf.hs_signer_suite` equals the ACTOR's authoritative
        /// suite (jwk preserved), clearance is the meet, scope is attenuated,
        /// client_id is inherited, and the active sid is retained.
        #[tokio::test]
        async fn delegated_mint_happy_path_stamps_expected_output() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), None).await;
            let resolver = user_resolver(&fx.user_ed);
            let fx = Fixture {
                service: fx.service.with_primary_enrollment_resolver(resolver),
                ..fx
            };

            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let source_jti = hyprstream_rpc::auth::decode_unverified(&source).unwrap().jti;
            let ctx = actor_ctx(&fx, now);
            let resp = fx
                .service
                .handle_exchange_delegated(&ctx, 1, &request(source))
                .await
                .unwrap();

            let token = match resp {
                PolicyResponseVariant::ExchangeDelegatedResult(info) => info.token,
                other => panic!("expected a minted delegated token, got {other:?}"),
            };
            let minted = hyprstream_rpc::auth::decode_unverified(&token).unwrap();
            assert_eq!(minted.sub, "alice", "originator sub preserved");
            assert_ne!(minted.jti, source_jti, "fresh jti distinct from the source");
            assert!(minted.jti.is_some());
            let act = minted.act.expect("act chain present");
            assert_eq!(act.sub, "service:mcp", "terminal actor nested in act");
            let cnf = minted.cnf.expect("cnf present");
            let expected_actor_hs = hyprstream_rpc::auth::service_signer_suite_b64(
                &fx.actor_ed.verifying_key().to_bytes(),
                None,
            );
            assert_eq!(
                cnf.hs_signer_suite.as_deref(),
                Some(expected_actor_hs.as_str()),
                "cnf binds the terminal actor's authoritative signer suite"
            );
            assert!(cnf.jwk.is_some(), "legacy cnf.jwk preserved");
            assert_eq!(
                minted.clearance,
                Some(CredentialClearance::from_label(label())),
                "clearance is the meet"
            );
            assert_eq!(minted.scope.as_deref(), Some("read"), "scope attenuated to subset");
            assert_eq!(minted.client_id.as_deref(), Some(CLIENT_ID), "client_id inherited");
            assert_eq!(minted.sid.as_deref(), Some("sess-1"), "active sid retained");
            assert_eq!(minted.iss, ISSUER, "minted under the authority issuer");
        }

        /// A first-hop delegated token (its `cnf` bound to the first terminal
        /// actor `service:worker`) is a valid SECOND-hop source: its `cnf`
        /// resolves against `act.sub` (the terminal actor enrollment), not `sub`.
        #[tokio::test]
        async fn two_hop_delegated_source_resolves_cnf_against_terminal_actor() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;

            // The first-hop terminal actor is `service:worker`, enrolled with the
            // fixture's actor Ed key (reuse `mcp`'s key is fine — we add worker).
            // Build a source: sub = alice (originator), act = service:mcp (the
            // enrolled terminal actor), cnf bound to service:mcp's enrolled key.
            let actor_hs = hyprstream_rpc::auth::service_signer_suite_b64(
                &fx.actor_ed.verifying_key().to_bytes(),
                None,
            );
            let src_claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
                .with_issuer(ISSUER.to_owned())
                .with_tenant(TENANT.to_owned())
                .with_sid("sess-1")
                .with_client_id(CLIENT_ID)
                .with_scope(Some("read".to_owned()))
                .with_clearance(label())
                .with_audience(Some("res-a".to_owned()))
                .with_act(hyprstream_rpc::auth::ActClaim {
                    sub: "service:mcp".to_owned(),
                    clearance: Some(CredentialClearance::from_label(label())),
                    act: None,
                })
                .with_cnf_jwk(&fx.actor_ed.verifying_key().to_bytes())
                .with_cnf_hs_signer_suite(actor_hs);
            let source = sign_source(&fx, &src_claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            // The source cnf validates against the terminal actor enrollment, so
            // the delegated mint proceeds to a token (not INVALID_SOURCE_CNF).
            assert!(
                matches!(resp, PolicyResponseVariant::ExchangeDelegatedResult(_)),
                "two-hop source cnf must resolve against act.sub, got {resp:?}"
            );
        }

        /// A user source whose primary is UNKNOWN to the resolver denies.
        #[tokio::test]
        async fn user_primary_unknown_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            // Resolver only knows "someone-else", not "alice".
            let resolver: Arc<dyn PrimaryEnrollmentResolver> = Arc::new(MockPrimaryResolver {
                subject: "someone-else".to_owned(),
                ed25519: [0u8; 32],
            });
            let fx = fixture(Some(allow_authorizer()), Some(resolver)).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("INVALID_SOURCE_CNF"));
        }

        /// A user source whose presented `cnf.jwk` mismatches the resolved primary
        /// Ed key denies (the wire key never self-confirms).
        #[tokio::test]
        async fn user_primary_key_mismatch_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            // Resolver returns a DIFFERENT key than the source's cnf.jwk.
            let resolver: Arc<dyn PrimaryEnrollmentResolver> = Arc::new(MockPrimaryResolver {
                subject: "alice".to_owned(),
                ed25519: SigningKey::from_bytes(&[0x55; 32]).verifying_key().to_bytes(),
            });
            let fx = fixture(Some(allow_authorizer()), Some(resolver)).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("INVALID_SOURCE_CNF"));
        }

        /// An uninstalled primary resolver denies a user source (fail closed).
        #[tokio::test]
        async fn uninstalled_primary_resolver_denies_user_source() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), None).await; // no resolver
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("INVALID_SOURCE_CNF"));
        }

        /// An uninstalled edge authorizer denies every delegation (fail closed).
        #[tokio::test]
        async fn uninstalled_edge_authorizer_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(None, Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("EDGE_AUTHORIZER_UNAVAILABLE"));
        }

        /// A cross-service source (source `aud` names another resource) is denied
        /// by the edge authorizer despite the actor's valid enrollment.
        #[tokio::test]
        async fn cross_service_source_audience_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let authz: Arc<dyn DelegationEdgeAuthorizer> = Arc::new(MockAuthorizer {
                allow: true,
                require_source_aud: Some("res-for-A".to_owned()),
                require_method: None,
            });
            let fx = fixture(Some(authz), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("EDGE_DENIED"));
        }

        /// A missing/empty target method id denies (no wildcard).
        #[tokio::test]
        async fn missing_target_method_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let mut req = request(source);
            req.target_method_id = Some("   ".to_owned());
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &req).await.unwrap();
            assert_eq!(err_code(&resp), Some("TARGET_METHOD_REQUIRED"));
        }

        /// Requested scope that broadens the source scope is denied.
        #[tokio::test]
        async fn scope_broadening_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let mut req = request(source);
            req.requested_scopes = Some("read admin".to_owned());
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &req).await.unwrap();
            assert_eq!(err_code(&resp), Some("SCOPE_ATTENUATION"));
        }

        /// A non-service caller cannot be a terminal actor.
        #[tokio::test]
        async fn non_service_actor_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let user_claims =
                hyprstream_rpc::auth::Claims::new("bob".to_owned(), now, now + 3600).with_clearance(label());
            let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
                Subject::new("bob"), TENANT, fx.actor_ed.verifying_key(), user_claims,
            );
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("NOT_A_TERMINAL_ACTOR"));
        }

        /// A source `at+jwt` with no sid (ambiguous non-interactive) is unsupported.
        #[tokio::test]
        async fn user_source_without_sid_is_unsupported() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.sid = None;
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("UNSUPPORTED_SOURCE"));
        }

        /// A revoked source credential is denied.
        #[tokio::test]
        async fn revoked_source_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let jti = hyprstream_rpc::auth::decode_unverified(&source).unwrap().jti.unwrap();
            let store = hyprstream_rpc::auth::global_credential_revocation_store().unwrap();
            store
                .revoke_credential(hyprstream_rpc::auth::CredentialId::jwt(ISSUER, jti), now + 3600)
                .await
                .unwrap();
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("SOURCE_REVOKED"));
        }

        /// A source tenant that differs from the actor's verified tenant denies.
        #[tokio::test]
        async fn tenant_mismatch_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.tenant = Some("did:web:other.example".to_owned());
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("TENANT_MISMATCH"));
        }

        /// A source signed by a FOREIGN composite pair (not in the authority's
        /// ledger) is denied — key possession is not issuer trust.
        #[tokio::test]
        async fn foreign_pair_source_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let foreign_ed = Arc::new(SigningKey::from_bytes(&[0x77; 32]));
            let (foreign_ml, _) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
            let foreign_ml = Arc::new(foreign_ml);
            let source = crate::auth::jwt::encode_composite_ml_dsa_65_ed25519(
                &user_source_claims(&fx, now), &foreign_ml, &foreign_ed,
            );
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("INVALID_SOURCE"));
        }

        /// An edge whose target method is not the exact reviewed method denies.
        #[tokio::test]
        async fn wrong_target_method_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let authz: Arc<dyn DelegationEdgeAuthorizer> = Arc::new(MockAuthorizer {
                allow: true,
                require_source_aud: None,
                require_method: Some("only.Allowed".to_owned()),
            });
            let fx = fixture(Some(authz), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let ctx = actor_ctx(&fx, now);
            // request() uses "model.Infer" ≠ "only.Allowed".
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("EDGE_DENIED"));
        }

        /// A capability that broadens the source `cap` is denied (monotonic
        /// attenuation on the capability axis).
        #[tokio::test]
        async fn capability_broadening_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let claims = user_source_claims(&fx, now).with_cap("read@mac://model/x".to_owned());
            let source = sign_source(&fx, &claims);
            let mut req = request(source);
            req.requested_capabilities = Some("write@mac://model/x".to_owned()); // not covered by read
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &req).await.unwrap();
            assert_eq!(err_code(&resp), Some("CAPABILITY_ATTENUATION"));
        }

        /// An expired source credential is denied. The signed-credential decoder
        /// enforces `exp` first (fail-closed at verification), so the stable deny
        /// is `INVALID_SOURCE` — the decoder never admits an expired token to the
        /// later `SOURCE_EXPIRED` re-check.
        #[tokio::test]
        async fn expired_source_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.exp = now - 10; // already expired
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("INVALID_SOURCE"));
        }

        /// A source whose `iss` is not this authority's configured issuer denies
        /// (key possession is not issuer trust).
        #[tokio::test]
        async fn untrusted_issuer_source_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.iss = "https://evil.example".to_owned();
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("UNTRUSTED_ISSUER"));
        }

        /// A terminal actor whose verified signer key does NOT match its enrolled
        /// key cannot resolve a v16 signer suite for the output (fail closed).
        #[tokio::test]
        async fn actor_cnf_not_enrolled_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            // Actor authenticates with a key that is NOT the enrolled `mcp` key.
            let wrong = SigningKey::from_bytes(&[0x99; 32]);
            let actor_claims =
                hyprstream_rpc::auth::Claims::new("service:mcp".to_owned(), now, now + 3600).with_clearance(label());
            let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
                Subject::new("service:mcp"), TENANT, wrong.verifying_key(), actor_claims,
            );
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("ACTOR_SUITE_UNRESOLVABLE"));
        }

        /// An unlabeled terminal actor (no clearance) denies.
        #[tokio::test]
        async fn unlabeled_actor_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let source = sign_source(&fx, &user_source_claims(&fx, now));
            let actor_claims =
                hyprstream_rpc::auth::Claims::new("service:mcp".to_owned(), now, now + 3600); // no clearance
            let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
                Subject::new("service:mcp"), TENANT, fx.actor_ed.verifying_key(), actor_claims,
            );
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("UNLABELED_ACTOR"));
        }

        /// A source OIDC session that is REVOKED denies (unique sid, no shared
        /// session-registry contamination).
        #[tokio::test]
        async fn revoked_session_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            let sid = "sess-revoked-unique-1";
            let reg = session_registry();
            let key = hyprstream_rpc::auth::SessionKey::oidc(ISSUER, sid);
            reg.register_session(
                key.clone(),
                hyprstream_rpc::auth::SessionState {
                    subject: "alice".to_owned(),
                    tenant: TENANT.to_owned(),
                    kind: hyprstream_rpc::auth::SessionKind::Interactive,
                    created_at: now,
                    expires_at: now + 3600,
                    status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                    clearance_epoch: 0,
                },
            )
            .await
            .unwrap();
            reg.revoke_session(&key).await.unwrap();
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.sid = Some(sid.to_owned());
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("SESSION_INVALID"));
        }

        /// A source OIDC session bound to a DIFFERENT subject denies (mismatch).
        #[tokio::test]
        async fn mismatched_session_subject_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            let sid = "sess-mismatch-unique-1";
            let reg = session_registry();
            let key = hyprstream_rpc::auth::SessionKey::oidc(ISSUER, sid);
            reg.register_session(
                key,
                hyprstream_rpc::auth::SessionState {
                    subject: "carol".to_owned(), // NOT alice
                    tenant: TENANT.to_owned(),
                    kind: hyprstream_rpc::auth::SessionKind::Interactive,
                    created_at: now,
                    expires_at: now + 3600,
                    status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                    clearance_epoch: 0,
                },
            )
            .await
            .unwrap();
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.sid = Some(sid.to_owned());
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("SESSION_INVALID"));
        }

        /// A source whose sid is UNKNOWN to the registry denies.
        #[tokio::test]
        async fn unknown_session_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            let _ = session_registry(); // ensure the registry exists (empty for this sid)
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.sid = Some("sess-never-registered-unique-1".to_owned());
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("SESSION_INVALID"));
        }

        /// A source whose sid is registered but EXPIRED denies.
        #[tokio::test]
        async fn expired_session_denies() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            let sid = "sess-expired-unique-1";
            let reg = session_registry();
            reg.register_session(
                hyprstream_rpc::auth::SessionKey::oidc(ISSUER, sid),
                hyprstream_rpc::auth::SessionState {
                    subject: "alice".to_owned(),
                    tenant: TENANT.to_owned(),
                    kind: hyprstream_rpc::auth::SessionKind::Interactive,
                    created_at: now - 7200,
                    expires_at: now - 3600, // already expired
                    status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                    clearance_epoch: 0,
                },
            )
            .await
            .unwrap();
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let mut claims = user_source_claims(&fx, now);
            claims.sid = Some(sid.to_owned());
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("SESSION_INVALID"));
        }

        /// A low-clearance INTERMEDIATE actor in the source's delegation chain
        /// lowers the effective meet: the minted credential's clearance is the
        /// intersection/min across originator, every actor, and the terminal
        /// actor — never above any input.
        #[tokio::test]
        async fn low_clearance_intermediate_lowers_the_meet() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            let public = SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
            let actor_hs = hyprstream_rpc::auth::service_signer_suite_b64(
                &fx.actor_ed.verifying_key().to_bytes(), None,
            );
            // Source: originator alice (Internal), act = terminal service:mcp
            // (Internal) nesting an intermediate service:low (Public). The cnf
            // resolves against the OUTERMOST act (service:mcp).
            let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
                .with_issuer(ISSUER.to_owned())
                .with_tenant(TENANT.to_owned())
                .with_sid("sess-1")
                .with_client_id(CLIENT_ID)
                .with_scope(Some("read".to_owned()))
                .with_clearance(label())
                .with_audience(Some("res-a".to_owned()))
                .with_act(hyprstream_rpc::auth::ActClaim {
                    sub: "service:mcp".to_owned(),
                    clearance: Some(CredentialClearance::from_label(label())),
                    act: Some(Box::new(hyprstream_rpc::auth::ActClaim {
                        sub: "service:low".to_owned(),
                        clearance: Some(CredentialClearance::from_label(public)),
                        act: None,
                    })),
                })
                .with_cnf_jwk(&fx.actor_ed.verifying_key().to_bytes())
                .with_cnf_hs_signer_suite(actor_hs);
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let token = match fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap() {
                PolicyResponseVariant::ExchangeDelegatedResult(info) => info.token,
                other => panic!("expected a minted token, got {other:?}"),
            };
            let minted = hyprstream_rpc::auth::decode_unverified(&token).unwrap();
            assert_eq!(
                minted.clearance,
                Some(CredentialClearance::from_label(public)),
                "the low-clearance intermediate must lower the effective meet to Public"
            );
        }

        /// A NON-DISPATCH OAuth token (valid signature/profile but NO
        /// `cnf.hs_signer_suite`) presented as a delegation SOURCE is denied —
        /// issuance may accept such a bearer, but delegation admission rejects it
        /// for the missing authoritative confirmation.
        #[tokio::test]
        async fn non_dispatch_source_without_hs_is_denied() {
            ensure_revocation_store();
            let now = chrono::Utc::now().timestamp();
            register_active_sid(now).await;
            let fx = fixture(Some(allow_authorizer()), Some(user_resolver(&SigningKey::from_bytes(&[0x43; 32])))).await;
            // The same valid interactive user source, but WITHOUT the v16 cnf
            // signer-suite confirmation (a bearer/non-dispatch shape).
            let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 3600)
                .with_issuer(ISSUER.to_owned())
                .with_tenant(TENANT.to_owned())
                .with_sid("sess-1")
                .with_client_id(CLIENT_ID)
                .with_scope(Some("read write".to_owned()))
                .with_clearance(label())
                .with_audience(Some("res-a".to_owned()))
                .with_cnf_jwk(&fx.user_ed.verifying_key().to_bytes());
            let source = sign_source(&fx, &claims);
            let ctx = actor_ctx(&fx, now);
            let resp = fx.service.handle_exchange_delegated(&ctx, 1, &request(source)).await.unwrap();
            assert_eq!(err_code(&resp), Some("INVALID_SOURCE_CNF"));
        }
}
