//! Policy-authority credential revocation and session registry: one canonical
//! store of each kind per deployment.
//!
//! The policy service process owns the single canonical credential-revocation
//! store and session registry, each backed by a durable local file
//! ([`hyprstream_rpc::auth::FileBackedCredentialRevocationStore`] and
//! [`hyprstream_rpc::auth::FileBackedSessionRegistry`]). Every other service
//! process publishes [`PolicyAuthorityRevocationStore`] and
//! [`PolicyAuthoritySessionRegistry`] as its process-global handles: checks
//! and publications cross the RPC bus to the policy service. Fail-closed
//! everywhere — at startup (an unreachable or unhealthy authority aborts the
//! process), at check (any RPC error or timeout reports the credential as
//! revoked / the session as not active), and at publication (a failed
//! publication is an error, never a silent no-op).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use hyprstream_rpc::auth::{
    CredentialId, CredentialRevocationStore, CredentialValue, RevocationPublishError,
    SessionKey, SessionRegisterError, SessionRegistry, SessionRevokeError, SessionState,
};
use hyprstream_rpc::crypto::SigningKey;
use hyprstream_service::ServiceContext;

use crate::services::PolicyClient;
use crate::services::generated::policy_client::{
    CheckCredentialRevocation, CheckSession, CredentialIdRef, CredentialIdRefContent,
    RegisterSession, RevokeCredential, RevokeSession, SessionKeyRef, SessionKeyRefContent,
};

/// Timeout for a single revocation check RPC. The underlying RPC client has a
/// fixed 30s call timeout; a revocation check sits on the verification hot
/// path, so it gets a much shorter budget.
const CHECK_TIMEOUT: Duration = Duration::from_secs(5);

/// Startup probe policy: attempts, and the delay between them. The policy
/// service is a startup dependency, but its listener may not be accepting yet
/// when a dependent process boots — retry instead of racing.
const PROBE_ATTEMPTS: u32 = 10;
const PROBE_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Map a wire [`CredentialIdRef`] to a validated [`CredentialId`], preserving
/// the JWT/CWT typed namespace. `None` = malformed (empty issuer or empty
/// value) — callers fail closed.
pub(crate) fn credential_id_from_ref(reference: &CredentialIdRef) -> Option<CredentialId> {
    let id = match &reference.content {
        CredentialIdRefContent::JwtJti(jti) => {
            CredentialId::jwt(reference.issuer.clone(), jti.clone())
        }
        CredentialIdRefContent::CwtCti(cti) => {
            CredentialId::cwt(reference.issuer.clone(), cti.clone())
        }
    };
    id.is_valid().then_some(id)
}

/// Map a [`CredentialId`] to its wire form, preserving the JWT/CWT typed
/// namespace.
fn credential_id_to_ref(id: &CredentialId) -> CredentialIdRef {
    let content = match &id.value {
        CredentialValue::Jwt(jti) => CredentialIdRefContent::JwtJti(jti.clone()),
        CredentialValue::Cwt(cti) => CredentialIdRefContent::CwtCti(cti.clone()),
    };
    CredentialIdRef {
        issuer: id.issuer.clone(),
        content,
    }
}

/// Map a wire [`SessionKeyRef`] to a validated [`SessionKey`], preserving
/// the OIDC/workload typed namespace. `None` = malformed (empty issuer or
/// identifier) — callers fail closed.
pub(crate) fn session_key_from_ref(reference: &SessionKeyRef) -> Option<SessionKey> {
    let key = match &reference.content {
        SessionKeyRefContent::OidcSid(sid) => {
            SessionKey::oidc(reference.issuer.clone(), sid.clone())
        }
        SessionKeyRefContent::WorkloadSessionId(id) => {
            SessionKey::workload(reference.issuer.clone(), id.clone())
        }
    };
    key.is_valid().then_some(key)
}

/// Map a [`SessionKey`] to its wire form, preserving the OIDC/workload typed
/// namespace.
fn session_key_to_ref(key: &SessionKey) -> SessionKeyRef {
    let content = match &key.id {
        hyprstream_rpc::auth::SessionIdentifier::OidcSid(sid) => {
            SessionKeyRefContent::OidcSid(sid.clone())
        }
        hyprstream_rpc::auth::SessionIdentifier::WorkloadSessionId(id) => {
            SessionKeyRefContent::WorkloadSessionId(id.clone())
        }
    };
    SessionKeyRef {
        issuer: key.issuer.clone(),
        content,
    }
}

/// Session registry that delegates to the policy service — the one canonical
/// session authority — over the RPC bus. Published as the process-global
/// registry by [`init_process_authority_stores`] in every process that does
/// not host the policy service.
pub struct PolicyAuthoritySessionRegistry {
    client: PolicyClient,
}

impl PolicyAuthoritySessionRegistry {
    /// Wrap a policy client as a session registry.
    pub fn new(client: PolicyClient) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl SessionRegistry for PolicyAuthoritySessionRegistry {
    /// The remote surface answers active/revoked checks only; a full state
    /// record is never fabricated. Consumers must use `is_revoked`.
    async fn session_state(&self, _key: &SessionKey) -> Option<SessionState> {
        None
    }

    async fn register_session(
        &self,
        key: SessionKey,
        state: SessionState,
    ) -> Result<(), SessionRegisterError> {
        let request = RegisterSession {
            session: session_key_to_ref(&key),
            subject: state.subject,
            tenant: state.tenant,
            expires_at: state.expires_at,
            clearance_epoch: state.clearance_epoch,
        };
        self.client
            .register_session(&request)
            .await
            .map_err(|e| {
                let message = e.to_string();
                if message.contains("already exists") {
                    hyprstream_rpc::auth::SessionExists.into()
                } else if message.contains("durably") || message.contains("not initialized") {
                    hyprstream_rpc::auth::SessionPublicationFailed.into()
                } else {
                    hyprstream_rpc::auth::InvalidSessionRecord.into()
                }
            })
    }

    async fn revoke_session(&self, key: &SessionKey) -> Result<(), SessionRevokeError> {
        let request = RevokeSession {
            session: session_key_to_ref(key),
        };
        // Publication BEFORE eviction: only after the authority has durably
        // accepted the revocation is the local cache generation flushed.
        self.client
            .revoke_session(&request)
            .await
            .map_err(|e| SessionRevokeError::new(format!("authority rejected publication: {e}")))?;
        hyprstream_rpc::auth::mac::flush_verified_subject_cache_generation();
        Ok(())
    }

    async fn is_revoked(&self, key: &SessionKey) -> bool {
        let request = CheckSession {
            session: session_key_to_ref(key),
        };
        match tokio::time::timeout(CHECK_TIMEOUT, self.client.check_session(&request)).await {
            // true = active and known; anything else is not active.
            Ok(Ok(active)) => !active,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "session check RPC failed — failing closed");
                true
            }
            Err(_) => {
                tracing::warn!(
                    timeout_ms = CHECK_TIMEOUT.as_millis() as u64,
                    "session check RPC timed out — failing closed"
                );
                true
            }
        }
    }
}

/// Credential-revocation store that delegates to the policy service — the one
/// canonical revocation authority — over the RPC bus. Published as the
/// process-global store by [`init_process_authority_stores`] in
/// every process that does not host the policy service.
pub struct PolicyAuthorityRevocationStore {
    client: PolicyClient,
}

impl PolicyAuthorityRevocationStore {
    /// Wrap a policy client as a revocation store.
    pub fn new(client: PolicyClient) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl CredentialRevocationStore for PolicyAuthorityRevocationStore {
    async fn is_revoked_checked(&self, id: &CredentialId) -> bool {
        let request = CheckCredentialRevocation {
            credential: credential_id_to_ref(id),
        };
        match tokio::time::timeout(CHECK_TIMEOUT, self.client.check_credential_revocation(&request))
            .await
        {
            Ok(Ok(revoked)) => revoked,
            Ok(Err(e)) => {
                tracing::warn!(credential = %id, error = %e,
                    "revocation check RPC failed — failing closed");
                true
            }
            Err(_) => {
                tracing::warn!(credential = %id, timeout_ms = CHECK_TIMEOUT.as_millis() as u64,
                    "revocation check RPC timed out — failing closed");
                true
            }
        }
    }

    async fn revoke_credential(
        &self,
        id: CredentialId,
        expires_at: i64,
    ) -> Result<(), RevocationPublishError> {
        let request = RevokeCredential {
            credential: credential_id_to_ref(&id),
            expires_at,
        };
        // Publication BEFORE eviction: only after the authority has durably
        // accepted the revocation are cached handles derived from the
        // credential evicted. On failure nothing is evicted.
        self.client
            .revoke_credential(&request)
            .await
            .map_err(|e| {
                RevocationPublishError::new(format!("authority rejected publication: {e}"))
            })?;
        hyprstream_rpc::auth::mac::revoke_verified_subject_credential(&id);
        Ok(())
    }
}

/// Initialize the process-global authority stores at startup: the
/// credential-revocation store AND the session registry.
///
/// When this process hosts the policy service (`hosts_policy`), it owns both
/// canonical stores as durable files under the deployment data dir
/// (`credential-revocations.jsonl`, `sessions.jsonl`). Every other process
/// builds a [`PolicyClient`] through the production resolver and PROBES the
/// authority (a freshly generated random credential ID, expected not revoked;
/// and a random session key, expected not active) before publishing the RPC
/// client stores, which share the one client.
///
/// Fail-closed: any error — an unreadable/corrupt durable file, or an
/// authority unreachable after [`PROBE_ATTEMPTS`] attempts — is propagated
/// and MUST abort startup. A process that cannot check revocations or
/// sessions must not serve credential verification.
pub async fn init_process_authority_stores(
    ctx: &ServiceContext,
    signing_key: &SigningKey,
    hosts_policy: bool,
) -> anyhow::Result<()> {
    if hosts_policy {
        let data_dir = ctx.deployment_data_dir()?;
        let credentials_path = data_dir.join("credential-revocations.jsonl");
        let store = hyprstream_rpc::auth::FileBackedCredentialRevocationStore::open(
            &credentials_path,
        )
        .with_context(|| {
            format!("open credential-revocation store {}", credentials_path.display())
        })?;
        hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(store))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        tracing::info!(path = %credentials_path.display(), "Credential revocation authority store initialized (durable)");

        let sessions_path = data_dir.join("sessions.jsonl");
        let sessions =
            hyprstream_rpc::auth::FileBackedSessionRegistry::open(&sessions_path)
                .with_context(|| {
                    format!("open session registry {}", sessions_path.display())
                })?;
        hyprstream_rpc::auth::set_global_session_registry(Arc::new(sessions))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        tracing::info!(path = %sessions_path.display(), "Session authority registry initialized (durable)");
        return Ok(());
    }

    let client = PolicyClient::from_resolver(
        signing_key.clone(),
        crate::services::factories::service_token(signing_key),
    )?;
    let probe = CredentialId::jwt(
        "https://revocation-probe.invalid",
        format!("probe-{:032x}", rand::random::<u128>()),
    );
    let session_probe = SessionKey::oidc(
        "https://revocation-probe.invalid",
        format!("probe-{:032x}", rand::random::<u128>()),
    );
    let mut last_error = String::new();
    for attempt in 1..=PROBE_ATTEMPTS {
        let request = CheckCredentialRevocation {
            credential: credential_id_to_ref(&probe),
        };
        let session_request = CheckSession {
            session: session_key_to_ref(&session_probe),
        };
        let revocation_probe =
            tokio::time::timeout(CHECK_TIMEOUT, client.check_credential_revocation(&request)).await;
        let session_probe_result =
            tokio::time::timeout(CHECK_TIMEOUT, client.check_session(&session_request)).await;
        match (revocation_probe, session_probe_result) {
            (Ok(Ok(false)), Ok(Ok(false))) => {
                hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                    PolicyAuthorityRevocationStore::new(client.clone()),
                ))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                hyprstream_rpc::auth::set_global_session_registry(Arc::new(
                    PolicyAuthoritySessionRegistry::new(client),
                ))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                tracing::info!(
                    "Revocation/session authority reachable; RPC client stores published"
                );
                return Ok(());
            }
            // A never-issued random ID reading as revoked (or a random
            // session reading active) means the authority itself is
            // unhealthy (its own store missing → fail-closed answers).
            (Ok(Ok(true)), _) => {
                last_error =
                    "authority reported a never-issued probe credential as revoked".to_owned();
            }
            (_, Ok(Ok(true))) => {
                last_error =
                    "authority reported a never-registered probe session as active".to_owned();
            }
            (Ok(Err(e)), _) | (_, Ok(Err(e))) => last_error = e.to_string(),
            (Err(_), _) | (_, Err(_)) => {
                last_error = format!("probe timed out after {}s", CHECK_TIMEOUT.as_secs());
            }
        }
        tracing::warn!(attempt, error = %last_error, "authority probe failed; retrying");
        if attempt < PROBE_ATTEMPTS {
            tokio::time::sleep(PROBE_RETRY_DELAY).await;
        }
    }
    anyhow::bail!(
        "revocation/session authority unreachable after {PROBE_ATTEMPTS} attempts: {last_error}"
    )
}
