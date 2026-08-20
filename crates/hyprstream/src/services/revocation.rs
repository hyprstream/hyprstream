//! Policy-authority credential revocation: one canonical store per deployment.
//!
//! The policy service process owns the single canonical credential-revocation
//! store, backed by a durable local file
//! ([`hyprstream_rpc::auth::FileBackedCredentialRevocationStore`]). Every other
//! service process publishes [`PolicyAuthorityRevocationStore`] as its
//! process-global store: revocation checks and publications cross the RPC bus
//! to the policy service. Fail-closed everywhere — at startup (an unreachable
//! or unhealthy authority aborts the process), at check (any RPC error or
//! timeout reports the credential as revoked), and at publication (a failed
//! publication is an error, never a silent no-op).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use hyprstream_rpc::auth::{
    CredentialId, CredentialRevocationStore, CredentialValue, RevocationPublishError,
};
use hyprstream_rpc::crypto::SigningKey;
use hyprstream_service::ServiceContext;

use crate::services::PolicyClient;
use crate::services::generated::policy_client::{
    CheckCredentialRevocation, CredentialIdRef, CredentialIdRefContent, RevokeCredential,
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

/// Credential-revocation store that delegates to the policy service — the one
/// canonical revocation authority — over the RPC bus. Published as the
/// process-global store by [`init_process_credential_revocation_store`] in
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

/// Initialize the process-global credential-revocation store at startup.
///
/// When this process hosts the policy service (`hosts_policy`), it owns the
/// canonical store: a [`hyprstream_rpc::auth::FileBackedCredentialRevocationStore`]
/// at `credential-revocations.jsonl` under the deployment data dir. Every
/// other process builds a [`PolicyClient`] through the production resolver
/// and PROBES the authority with a freshly generated random credential ID
/// (expected answer: not revoked) before publishing the RPC client store.
///
/// Fail-closed: any error — an unreadable/corrupt durable file, or an
/// authority unreachable after [`PROBE_ATTEMPTS`] attempts — is propagated
/// and MUST abort startup. A process that cannot check revocations must not
/// serve JTI-bearing-token verification.
pub async fn init_process_credential_revocation_store(
    ctx: &ServiceContext,
    signing_key: &SigningKey,
    hosts_policy: bool,
) -> anyhow::Result<()> {
    if hosts_policy {
        let path = ctx
            .deployment_data_dir()?
            .join("credential-revocations.jsonl");
        let store = hyprstream_rpc::auth::FileBackedCredentialRevocationStore::open(&path)
            .with_context(|| {
                format!("open credential-revocation store {}", path.display())
            })?;
        hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(store))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        tracing::info!(path = %path.display(), "Credential revocation authority store initialized (durable)");
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
    let mut last_error = String::new();
    for attempt in 1..=PROBE_ATTEMPTS {
        let request = CheckCredentialRevocation {
            credential: credential_id_to_ref(&probe),
        };
        match tokio::time::timeout(CHECK_TIMEOUT, client.check_credential_revocation(&request))
            .await
        {
            Ok(Ok(false)) => {
                hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                    PolicyAuthorityRevocationStore::new(client),
                ))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                tracing::info!(
                    "Credential revocation authority reachable; RPC client store published"
                );
                return Ok(());
            }
            // A never-issued random ID reading as revoked means the authority
            // itself is unhealthy (its own store missing → fail-closed true).
            Ok(Ok(true)) => {
                last_error =
                    "authority reported a never-issued probe credential as revoked".to_owned();
            }
            Ok(Err(e)) => last_error = e.to_string(),
            Err(_) => {
                last_error = format!("probe timed out after {}s", CHECK_TIMEOUT.as_secs());
            }
        }
        tracing::warn!(attempt, error = %last_error, "revocation authority probe failed; retrying");
        if attempt < PROBE_ATTEMPTS {
            tokio::time::sleep(PROBE_RETRY_DELAY).await;
        }
    }
    anyhow::bail!(
        "revocation authority unreachable after {PROBE_ATTEMPTS} attempts: {last_error}"
    )
}
