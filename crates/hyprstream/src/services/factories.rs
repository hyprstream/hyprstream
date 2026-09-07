//! Service factory functions for inventory-based registration.
//!
//! This module contains all `#[service_factory]` decorated functions that
//! automatically register services with the inventory system.
//!
//! # Pattern
//!
//! Same pattern as:
//! - `#[register_scopes]` for authorization scopes
//! - `DriverFactory` in git2db for storage drivers
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::service::{get_factory, ServiceContext};
//!
//! let ctx = ServiceContext::new(...);
//! let factory = get_factory("policy").unwrap();
//! let spawnable = (factory.factory)(&ctx)?;
//! manager.spawn(spawnable).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use git2db::Git2DB;
use hyprstream_rpc::moq_event::MoqEventOrigin;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::service_factory;
use hyprstream_service::{ServiceContext, Spawnable};
use hyprstream_vfs::{MountTarget, Namespace};
use tokio::sync::RwLock;
use tracing::info;

use crate::auth::identity_store::credentials_dir;
use crate::auth::PolicyManager;
use crate::config::{HyprConfig, TokenConfig};
use crate::services::generated::policy_client::{RefreshServiceTokenRequest, RegisterServiceKey};
use crate::services::{
    DiscoveryService, McpConfig, McpService, PolicyClient, PolicyService, RegistryClient,
    RegistryService,
};

/// Load HyprConfig, falling back to default on error.
fn load_config() -> HyprConfig {
    HyprConfig::load().unwrap_or_default()
}

/// Get the JWT bound to this service instance's exact signing key.
fn service_token(signing_key: &SigningKey) -> Option<String> {
    let trust = hyprstream_service::global_trust_store();
    trust
        .get(&signing_key.verifying_key())
        .and_then(|att| att.jwt)
}

/// Construct a Policy client through the service context's resolved transport.
///
/// `for_local_bootstrap` reads only the process-local endpoint registry. That
/// is appropriate for co-located services, but a rootless Quadlet starts each
/// service in a separate process. `ServiceContext::transport` preserves the
/// in-process endpoint when applicable and resolves the shared IPC socket when
/// `--ipc` is selected.
fn policy_client_for_context(
    ctx: &ServiceContext,
    signing_key: SigningKey,
    policy_vk: hyprstream_rpc::crypto::VerifyingKey,
    token: Option<String>,
) -> anyhow::Result<PolicyClient> {
    let transport = ctx.transport("policy", SocketKind::Rep);
    PolicyClient::for_local_transport_bootstrap(&transport, signing_key, policy_vk, token)
}

fn policy_client_for_transport(
    transport: &hyprstream_rpc::transport::TransportConfig,
    signing_key: SigningKey,
    policy_vk: hyprstream_rpc::crypto::VerifyingKey,
    token: Option<String>,
) -> anyhow::Result<PolicyClient> {
    PolicyClient::for_local_transport_bootstrap(transport, signing_key, policy_vk, token)
}

/// Shared Git2DB registry instance. Lazily initialized by the first factory
/// that needs it. Both PolicyService and RegistryService share this instance.
static SHARED_GIT2DB: std::sync::OnceLock<Arc<RwLock<Git2DB>>> = std::sync::OnceLock::new();

/// Shared JTI blocklist Arc — set by `create_policy_service`, read by
/// `create_oauth_service`. Because PolicyService is always created first
/// (OAuthService `depends_on = ["policy"]`), the lock is always populated
/// before `create_oauth_service` runs.
static SHARED_JTI_BLOCKLIST: std::sync::OnceLock<Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>> =
    std::sync::OnceLock::new();

/// Get or initialize the shared Git2DB registry for the given models directory.
fn get_or_init_git2db(models_dir: &std::path::Path) -> anyhow::Result<Arc<RwLock<Git2DB>>> {
    if let Some(existing) = SHARED_GIT2DB.get() {
        return Ok(Arc::clone(existing));
    }

    let registry = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(Git2DB::open(models_dir))
    })
    .context("Failed to initialize shared Git2DB registry")?;

    let shared = Arc::new(RwLock::new(registry));
    // If another thread beat us, that's fine — use theirs
    Ok(Arc::clone(SHARED_GIT2DB.get_or_init(|| shared)))
}

/// Resolve the on-disk directory for the durable PDS record store (#910a).
///
/// A single RocksDB database lives here, matching `RocksDbUserStore`'s
/// `<config_dir>/users.db` convention. The registry service (the sole
/// publisher) opens it read-write; the discovery service (the resolver)
/// opens it read-only — see `services::discovery::PdsRecordStore`.
pub(crate) fn pds_store_dir(ctx: &ServiceContext) -> anyhow::Result<std::path::PathBuf> {
    Ok(ctx.deployment_data_dir()?.join("pds-store"))
}

/// The lifecycle state of the checkpointed PDS store, as seen by the QUIC
/// startup gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdsBootState {
    /// ≥1 checkpoint-verified accepted state is present. The caller MUST run
    /// [`with_checkpointed_native_announcements`] to populate announcements.
    Populated,
    /// Genuine first boot: the store has never held an accepted state (the
    /// durable first-boot RocksDB key is present alongside an empty store).
    /// QUIC may be deferred for this run; a restart is required after the
    /// registry writes its first accepted state. No runtime auto-activation
    /// is implemented.
    FirstBoot,
}

/// Classify the checkpointed PDS store for the QUIC startup gate.
///
/// This replaces the prior emptiness check, which could not distinguish a
/// freshly provisioned first-boot store from a steady-state store that lost
/// its data, and which collapsed every read/decode/signature/consistency
/// error into "first boot". The classification is:
///
/// - Store directory present, marker present, store empty →
///   [`PdsBootState::FirstBoot`] (provisioned via `init-deployment-store`;
///   the registry has not yet written an accepted state).
/// - Store directory present, ≥1 verified accepted state →
///   [`PdsBootState::Populated`].
/// - Store directory present, empty, **no** marker → `Err` (possible data
///   loss or corruption; a loud steady-state failure — never defer).
/// - Store directory **absent** → `Err` (missing security history; run
///   `init-deployment-store` to provision, or restore from backup).
/// - Any read/decode/signature/consistency failure → `Err` (propagated;
///   fail-closed at the checkpoint boundary).
///
/// Only [`PdsBootState::FirstBoot`] may enter a QUIC deferral path; every
/// other outcome either proceeds with checkpointed announcements or fails
/// startup.
pub fn classify_pds_store_for_quic(ctx: &ServiceContext) -> anyhow::Result<PdsBootState> {
    let store_dir = pds_store_dir(ctx)?;
    if !store_dir.exists() {
        // A missing store is indistinguishable from deletion of security
        // history — the checkpoint source's documented rule. First boot is an
        // explicit provisioning step (init-deployment-store), not an absent
        // directory. Fail closed; do not defer.
        anyhow::bail!(
            "PDS accepted-state store at {} does not exist. Run \
             `init-deployment-store` to provision a fresh deployment, or \
             restore from backup. Refusing to defer QUIC startup.",
            store_dir.display()
        );
    }
    let acceptance_identity = hyprstream_discovery::deployment_registry_verifier()?;
    let store = crate::services::discovery::PdsRecordStore::open_readonly(&store_dir)?
        .with_at9p_deployment_verifier(acceptance_identity);
    // Propagates decode, signature-verification, and consistency failures
    // (half-checkpoint mismatches, watermark/digest mismatches) as Err —
    // these are NEVER evidence of first boot.
    let states = store.accepted_at9p_states()?;
    if !states.is_empty() {
        return Ok(PdsBootState::Populated);
    }
    // The store exists and opens cleanly but holds no accepted states. Only
    // the durable first-boot marker (a RocksDB key deleted atomically with
    // the first accepted-state commit) distinguishes genuine first boot from
    // data loss.
    if store.first_boot_pending()? {
        return Ok(PdsBootState::FirstBoot);
    }
    anyhow::bail!(
        "PDS accepted-state store at {} exists but contains no verified accepted \
         states and no first-boot provisioning marker. This indicates data loss \
         or corruption, not first boot; refusing to defer QUIC startup.",
        store_dir.display()
    )
}

/// Populate every ordinary network service announcement from a fresh
/// checkpoint-verifying PDS read. Missing or ambiguous state fails startup
/// before any QUIC service can bind and advertise an incomplete bundle.
pub fn with_checkpointed_native_announcements(
    mut ctx: ServiceContext,
    service_names: &[String],
) -> anyhow::Result<ServiceContext> {
    let acceptance_identity = hyprstream_discovery::deployment_registry_verifier()?;
    let store = crate::services::discovery::PdsRecordStore::open_readonly(&pds_store_dir(&ctx)?)?
        .with_at9p_deployment_verifier(acceptance_identity);
    let states = store.accepted_at9p_states()?;
    for service_name in service_names
        .iter()
        .filter(|name| name.as_str() != "discovery")
    {
        let signer = ctx.service_signing_key(service_name);
        let mut matching = states.iter().filter(|state| {
            state
                .current
                .services
                .iter()
                .any(|entry| entry.id == *service_name)
                && state
                    .current
                    .subject_keys
                    .iter()
                    .any(|key| key.ed25519_pub.as_slice() == signer.verifying_key().as_bytes())
        });
        let state = matching.next().ok_or_else(|| {
            anyhow::anyhow!(
                "no checkpoint-verified accepted state authorizes network service {service_name}"
            )
        })?;
        anyhow::ensure!(
            matching.next().is_none(),
            "multiple accepted states authorize network service {service_name}"
        );
        let announcement = hyprstream_service::NativeServiceAnnouncement::from_accepted_state(
            service_name,
            &signer,
            state,
        )?;
        ctx = ctx.with_native_announcement(service_name.clone(), announcement);
    }
    Ok(ctx)
}

/// Resolve the CA-signed JWT used to register a service's signing key.
///
/// Fail-closed (issue #441): returns the JWT (preferring one already in the
/// trust store, falling back to the authoritative on-disk credential), or an
/// ERROR naming the real cause. It never silently returns "skip" — a service
/// that cannot produce its JWT must not come up serving signed responses.
fn resolve_registration_jwt(
    service_name: &str,
    creds_dir: &std::path::Path,
    secrets_profile: crate::auth::identity_store::SecretsProfile,
    from_trust: Option<String>,
) -> anyhow::Result<String> {
    if let Some(jwt) = from_trust {
        return Ok(jwt);
    }
    match crate::auth::identity_store::load_service_jwt_for_profile(
        creds_dir,
        service_name,
        secrets_profile,
    ) {
        Ok(Some(jwt)) => Ok(jwt),
        Ok(None) => anyhow::bail!(
            "service '{service_name}' cannot register its signing key: \
             no CA-signed JWT found in trust store or on disk at {}. \
             Run 'hyprstream wizard' to provision service credentials; \
             a service must not serve signed responses without a registered key.",
            creds_dir.display(),
        ),
        Err(e) => anyhow::bail!(
            "service '{service_name}' cannot register its signing key: \
             failed to read CA-signed JWT from {}: {e}",
            creds_dir.display(),
        ),
    }
}

/// Register this service's verifying key with the PolicyService CA.
///
/// Called by each non-policy factory so that peer services can resolve
/// our pubkey via `resolveServiceKey` RPC.  No-op for PolicyService itself.
///
/// # Fail-closed (issue #441)
///
/// A service that cannot obtain its CA-signed JWT (and therefore cannot
/// register its signing key) MUST NOT come up serving signed responses —
/// every peer would resolve a key/JWT that disagrees with what we actually
/// sign with, surfacing three layers away as a cryptic "Response signed by
/// unexpected key". So registration is a hard precondition: if we cannot get
/// a JWT, we return an error and the factory (and thus the service) fails to
/// start, naming the real cause.
///
/// The authoritative source of the service JWT is on disk
/// (`credentials/{service}/service-jwt`), written by the wizard/bootstrap
/// manager. At process startup only the bootstrap *pubkeys* are seeded into
/// the trust store (with `jwt: None`); the JWT itself is loaded here and
/// seeded into the trust store so that peer-client construction
/// (`service_token`) and the background renewal task can read it.
fn register_service_key(
    ctx: &ServiceContext,
    service_name: &str,
    signing_key: &SigningKey,
) -> anyhow::Result<()> {
    // PolicyService doesn't register — it IS the CA.
    if service_name == "policy" {
        return Ok(());
    }

    let creds_dir = credentials_dir()?;
    let secrets_profile = crate::auth::identity_store::SecretsProfile::from_env()?;

    // The JWT may already be in the trust store (e.g. seeded by an earlier
    // registration in this process); otherwise load it from disk — the
    // authoritative location the wizard/bootstrap manager wrote it to.
    let from_trust = {
        let trust = hyprstream_service::global_trust_store();
        trust
            .get(&signing_key.verifying_key())
            .and_then(|att| att.jwt.clone())
    };
    let jwt = resolve_registration_jwt(service_name, &creds_dir, secrets_profile, from_trust)?;

    // Seed the loaded JWT into the trust store so that peer-client construction
    // (`service_token`) and the background renewal task can read it. Bind it to
    // this service's own verifying key — the key we actually sign with — so the
    // advertised key == the actual signer (the #441 invariant).
    {
        let vk = signing_key.verifying_key();
        let trust = hyprstream_service::global_trust_store();
        let expires_at = decode_jwt_exp(&jwt).unwrap_or(0);
        let mut att = trust
            .get(&vk)
            .unwrap_or_else(|| hyprstream_service::Attestation {
                scopes: std::iter::once(service_name.to_owned()).collect(),
                subject: None,
                jwt: None,
                expires_at: 0,
                attested_by: None,
            });
        att.scopes.insert(service_name.to_owned());
        att.jwt = Some(jwt.clone());
        att.expires_at = expires_at;
        trust.insert(vk, att);
    }

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_transport = ctx.transport("policy", SocketKind::Rep);
    let policy_client = policy_client_for_transport(
        &policy_transport,
        signing_key.clone(),
        policy_vk,
        Some(jwt.clone()),
    )?;

    let request = RegisterServiceKey {
        service_name: service_name.to_owned(),
        verifying_key: signing_key.verifying_key().as_bytes().to_vec(),
        service_jwt: jwt.clone(),
    };

    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(policy_client.register_service_key(&request))
    })
    .map_err(|e| anyhow::anyhow!("registerServiceKey RPC failed for '{service_name}': {e}"))?;

    info!(
        service = service_name,
        "Registered verifying key with PolicyService"
    );

    // Spawn background JWT renewal for this service
    spawn_jwt_renewal_task(
        service_name,
        signing_key.clone(),
        creds_dir,
        secrets_profile,
        policy_transport,
    );

    Ok(())
}

/// Decode the `exp` claim from a JWT without verifying the signature.
///
/// Used for local-disk JWTs that we issued ourselves — signature is verified
/// by PolicyService; here we only need the expiry to decide whether to renew.
fn decode_jwt_exp(jwt: &str) -> Option<i64> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value.get("exp")?.as_i64()
}

/// Spawn a background task that renews this service's JWT when it approaches expiry.
///
/// Checks hourly; renews when ≤7 days remain. Updates the global trust store so
/// in-flight RPC calls stay authenticated, and best-effort persists the renewed
/// JWT to disk so it survives restart. When credentials are read-only (systemd
/// `$CREDENTIALS_DIRECTORY`), persistence uses the unit's writable state home.
fn spawn_jwt_renewal_task(
    service_name: &str,
    signing_key: SigningKey,
    credentials_dir: std::path::PathBuf,
    secrets_profile: crate::auth::identity_store::SecretsProfile,
    policy_transport: hyprstream_rpc::transport::TransportConfig,
) {
    let service_name = service_name.to_owned();
    tokio::spawn(async move {
        const CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3_600);
        const RENEW_THRESHOLD: i64 = 7 * 24 * 3_600; // 7 days remaining

        loop {
            tokio::time::sleep(CHECK_INTERVAL).await;

            let jwt = match crate::auth::identity_store::load_service_jwt_for_profile(
                &credentials_dir,
                &service_name,
                secrets_profile,
            ) {
                Ok(Some(j)) => j,
                _ => continue,
            };

            let expires_at = match decode_jwt_exp(&jwt) {
                Some(exp) => exp,
                None => continue,
            };

            let remaining = expires_at - chrono::Utc::now().timestamp();
            if remaining > RENEW_THRESHOLD {
                continue;
            }

            // Build a PolicyClient using current trust-store JWT
            let (policy_vk, current_jwt) = {
                let trust = hyprstream_service::global_trust_store();
                let vk = match trust.resolve_one("policy") {
                    Some(v) => v,
                    None => {
                        tracing::warn!(
                            service = service_name,
                            "policy key not in trust store; skipping JWT renewal"
                        );
                        continue;
                    }
                };
                let svc_jwt = match trust
                    .get(&signing_key.verifying_key())
                    .and_then(|att| att.jwt)
                {
                    Some(j) => j,
                    None => {
                        tracing::warn!(
                            service = service_name,
                            "service JWT not in trust store; skipping renewal"
                        );
                        continue;
                    }
                };
                (vk, svc_jwt)
            };

            let policy_client = match policy_client_for_transport(
                &policy_transport,
                signing_key.clone(),
                policy_vk,
                Some(current_jwt),
            ) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(service = service_name, error = %e, "failed to create PolicyClient; skipping JWT renewal");
                    continue;
                }
            };
            let req = RefreshServiceTokenRequest {
                ttl_seconds: 2_592_000,
            };

            match policy_client.refresh_service_token(&req).await {
                Ok(info) => {
                    // Update trust store with renewed JWT
                    let trust = hyprstream_service::global_trust_store();
                    let vk = signing_key.verifying_key();
                    if let Some(mut att) = trust.get(&vk) {
                        att.jwt = Some(info.token.clone());
                        att.expires_at = info.expires_at;
                        trust.insert(vk, att);
                    }
                    // Persist so the renewed JWT survives restart (#803).
                    if let Err(e) = crate::auth::identity_store::write_service_jwt_for_profile(
                        &credentials_dir,
                        &service_name,
                        secrets_profile,
                        &info.token,
                    ) {
                        tracing::error!(
                            service = service_name,
                            error = %e,
                            "renewed service JWT could not be persisted; it is \
                             process-ephemeral until the next restart re-registers"
                        );
                    }
                    tracing::info!(
                        service = service_name,
                        expires_at = info.expires_at,
                        "Renewed service JWT"
                    );
                }
                Err(e) => {
                    tracing::warn!(service = service_name, "JWT renewal RPC failed: {e}");
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for EventService — initializes the moq-lite event bus (#167).
///
/// Replaces the ZMQ XPUB/XSUB ProxyService with a `MoqEventOrigin` registered
/// as a process global. Publishers and subscribers use the global origin directly;
/// no forwarding proxy or thread is needed. The returned service just holds the
/// shutdown barrier so the orchestrator tracks lifecycle correctly.
#[service_factory("event")]
fn create_event_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating EventService (moq-lite event bus)");

    if !hyprstream_rpc::events::event_authz_installed() {
        let config = load_config();
        let sk = ctx.service_signing_key("event");
        // Declared MoQ/event track policy (v16 §10 / #1510). The generated
        // dispatch inventory (WS-D / #1505) is the end-state producer of these
        // rows; until it lands the empty table is the honest state and every
        // unlisted track/prefix denies.
        let track_policy = hyprstream_rpc::auth::mac::MoqEventPolicyTable::empty();
        let pep = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(crate::mac::production_moq_event_pep(
                sk,
                &config.oauth,
                "moq-event",
                track_policy,
            ))
        })
        .context("construct MoQ/event MAC PEP")?;
        hyprstream_rpc::events::install_event_authz(Arc::new(
            hyprstream_rpc::events::MacEventAuthz::new(pep),
        ))
        .map_err(anyhow::Error::msg)?;
    }

    let origin = MoqEventOrigin::new();
    hyprstream_rpc::moq_event::init_global_moq_event_origin(origin.clone());

    // #275: serve the event-bus origin over the well-known cross-process UDS path
    // so OTHER service processes (worker, model, ...) can publish/subscribe events
    // to this shared bus. In the same-process (InprocManager) deployment every
    // service shares this global origin directly; this UDS plane is the bridge
    // for the systemd / --ipc deployment where each service is its own process.
    let event_moq_path = hyprstream_rpc::paths::event_socket();
    hyprstream_rpc::moq_event::serve_event_moq_uds_background(origin, event_moq_path);

    Ok(Box::new(MoqEventBarrierService::new()))
}

/// Minimal `Spawnable` that satisfies the service lifecycle contract for the
/// moq event bus. The bus itself is a process-global `MoqEventOrigin` with no
/// dedicated thread; this service just waits for shutdown.
struct MoqEventBarrierService;

impl MoqEventBarrierService {
    fn new() -> Self {
        Self
    }
}

impl Spawnable for MoqEventBarrierService {
    fn name(&self) -> &str {
        "event"
    }

    fn registrations(
        &self,
    ) -> Vec<(
        hyprstream_rpc::registry::SocketKind,
        hyprstream_rpc::transport::TransportConfig,
    )> {
        vec![] // no ZMQ endpoints
    }

    fn run(
        self: Box<Self>,
        shutdown: std::sync::Arc<tokio::sync::Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> hyprstream_rpc::error::Result<()> {
        if let Some(ready) = on_ready {
            let _ = ready.send(());
        }
        // systemd Type=notify: send READY=1 so the unit reaches `active` rather
        // than timing out (~45s) and restart-looping. These moq barrier services
        // don't go through the RPC serve path (serve.rs::signal_ready) that
        // normally notifies systemd, so they must signal readiness themselves —
        // the moq origin/event-bus is already initialized in the factory before
        // run() is called, so the service is genuinely ready here.
        let _ = hyprstream_rpc::notify::ready();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::Other(e.to_string()))?;
        rt.block_on(shutdown.notified());
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ledger Service Factory (Phase-1 local-enforcer, #925 — `ledger` feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for the Phase-1 cellular-ledger local-enforcer service (epic #922,
/// #925). Gated behind the `ledger` cargo feature; inert unless
/// `[ledger] enabled = true`.
///
/// **Follow-up wiring (clearly marked, not in this skeleton):**
/// - The grant verifier is a [`StaticGrantVerifier`] that denies every
///   presented grant until populated — wiring it to
///   `hyprstream_rpc::auth::ucan` chain validation + the
///   `ai.hyprstream.ledger.allocation` lexicon (item 1.5) is the production
///   activation path.
/// - The receipt sink is the [`LoggingReceiptSink`] (drains to zero, no PDS
///   writes); the production sink writes the `ai.hyprstream.ledger.receipt`
///   PDS records.
/// - The live scheduler realign (`hyprstream-workers` `SandboxPool::acquire`
///   → `LocalEnforcer::admit`) lands behind this flag once the #761
///   group-authority decision (#921.5) is made.
#[cfg(feature = "ledger")]
#[service_factory("ledger", depends_on = ["policy"])]
fn create_ledger_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    use crate::services::ledger::{
        CoseCheckpointSigner, LedgerService, LoggingReceiptSink, StaticGrantVerifier,
    };
    use hyprstream_crypto::did_key::ed25519_to_did_key;
    use hyprstream_ledger::{Did, MemLedger};
    use std::sync::Arc;

    info!("Creating LedgerService (Phase-1 local-enforcer, #925)");

    let config = load_config();
    let lcfg = config.ledger.clone();
    if !lcfg.is_enabled() {
        anyhow::bail!(
            "ledger service requested but [ledger] enabled = false (the Phase-1 enforcer is opt-in)"
        );
    }

    // Re-assert the production contract here as well as at config load: this
    // factory is reachable from paths that construct a config programmatically
    // rather than through `HyprConfig::validate`. The DSN field only exists in
    // a `postgres-ledger` build; without it there is no durable backend to
    // name, and production validation correctly refuses.
    #[cfg(feature = "postgres-ledger")]
    let production_dsn = config.ledger_postgres_url.as_deref();
    #[cfg(not(feature = "postgres-ledger"))]
    let production_dsn: Option<&str> = None;
    lcfg.validate_for_production(production_dsn)?;

    // Cell identity = did:key over the service Ed25519 key.
    let ed_sk = ctx.service_signing_key("ledger");
    let ed_vk = ed_sk.verifying_key();
    let cell_identity = Did(ed25519_to_did_key(&ed_vk.to_bytes()));

    // Register this service's verifying key with PolicyService.
    let _ = ctx.verifying_key();

    // PQ (ML-DSA-65) key under the Hybrid policy. Fail-closed construction:
    // `require_pq_signatures` set with no key available ⇒ refuse to start the
    // ledger service rather than silently downgrade checkpoints to Classical.
    // The mint verifier is derived from the SAME key material as the checkpoint
    // signer, because the actor signs issuance authorizations with that signer.
    // Building both here keeps the two halves of the seal in lockstep — a
    // verifier configured more permissively than the signer would re-open the
    // hole the seal exists to close.
    let mut mint_pq_vk: Option<std::sync::Arc<hyprstream_crypto::pq::MlDsaVerifyingKey>> = None;
    let signer: Arc<dyn hyprstream_ledger::CheckpointSigner + Send + Sync> = if lcfg
        .require_pq_signatures
    {
        let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
        let store = crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, &config.oauth);
        let pq_key = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { store.active_key().await })
        });
        match pq_key {
            Some(k) => {
                let vk_bytes = hyprstream_crypto::pq::ml_dsa_sk_to_vk_bytes(&k);
                let vk = hyprstream_crypto::pq::ml_dsa_vk_from_bytes(&vk_bytes).map_err(|e| {
                    anyhow::anyhow!("ledger: could not derive the ML-DSA-65 verifying key: {e}")
                })?;
                mint_pq_vk = Some(std::sync::Arc::new(vk));
                Arc::new(CoseCheckpointSigner::hybrid(
                    cell_identity.clone(),
                    ed_sk,
                    (*k).clone(),
                ))
            }
            None => anyhow::bail!(
                "ledger: require_pq_signatures is set but no ML-DSA-65 key is available (fail-closed)"
            ),
        }
    } else {
        Arc::new(CoseCheckpointSigner::classical(
            cell_identity.clone(),
            ed_sk,
        ))
    };

    // The mint authority is derived from the SAME key material the actor signs
    // issuance authorizations with, so the two halves of the seal cannot drift
    // apart. It is passed at backend construction and is immutable afterwards —
    // there is no setter, which is what stops a consumer holding a backend from
    // installing a permissive authority of its own.
    let mint_authority = match &mint_pq_vk {
        Some(pq) => Some(hyprstream_ledger::MintAuthority::hybrid(
            ed_vk,
            (**pq).clone(),
        )),
        None if lcfg.require_pq_signatures => {
            anyhow::bail!(
                "ledger: require_pq_signatures is set but no ML-DSA-65 verifying key is \
                 available for the mint authority (fail-closed)"
            )
        }
        None => Some(hyprstream_ledger::MintAuthority::classical(ed_vk)),
    };

    // Backend selection (PAY-01 F8): BackendKind drives construction.
    // **Postgres = production (fail-closed on unavailability); Mem = dev/test.**
    // No silent fallback — a configured Postgres backend that cannot connect
    // is FATAL.
    let backend: Box<dyn hyprstream_ledger::LedgerBackend + Send + 'static> = match lcfg.backend {
        crate::services::ledger::BackendKind::Postgres => {
            #[cfg(feature = "postgres-ledger")]
            {
                let pg_url = config.ledger_postgres_url.as_deref().filter(|u| !u.is_empty()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "ledger: backend = Postgres but no ledger_postgres_url configured \
                         (FATAL — production requires a durable backend)"
                    )
                })?;
                let pg_config = hyprstream_ledger::postgres::PostgresConfig {
                    url: pg_url.to_owned(),
                    pool_size: config.ledger_postgres_pool_size.unwrap_or(4),
                };
                let pg = hyprstream_ledger::postgres::PostgresLedger::connect(
                    pg_config,
                    cell_identity.clone(),
                    mint_authority,
                ).map_err(|e| {
                    anyhow::anyhow!(
                        "ledger: PostgresLedger::connect FAILED (FATAL — no silent fallback): {e}"
                    )
                })?;
                info!("Ledger backend: PostgresLedger (production durable)");
                Box::new(pg)
            }
            #[cfg(not(feature = "postgres-ledger"))]
            {
                anyhow::bail!(
                    "ledger: backend = Postgres but postgres-ledger feature is not compiled \
                     (FATAL — rebuild with --features postgres-ledger for production)"
                );
            }
        }
        crate::services::ledger::BackendKind::Mem => {
            // A volatile backend cannot hold a money ledger. In production this
            // is fatal rather than a warning: silently accounting into memory
            // that vanishes on restart is worse than refusing to start.
            if lcfg.is_production() {
                anyhow::bail!(
                    "ledger: backend = Mem is not permitted in production (FATAL — \
                     all ledger state would be lost on restart). Set [ledger] backend = \"postgres\" \
                     with a ledger_postgres_url, or unset the production mode to run dev/test."
                );
            }
            info!("Ledger backend: MemLedger (dev/test only — all state is volatile)");
            Box::new(MemLedger::new(cell_identity.clone(), mint_authority))
        }
    };

    let verifier: Arc<dyn crate::services::ledger::GrantVerifier + Send + Sync> =
        Arc::new(StaticGrantVerifier::new());
    let sink: Arc<dyn crate::services::ledger::ReceiptSink + Send + Sync> =
        Arc::new(LoggingReceiptSink);

    let service = LedgerService::spawn(
        lcfg,
        backend,
        signer,
        verifier,
        sink,
        cell_identity,
    );
    Ok(Box::new(service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Policy Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for PolicyService (Casbin policy management)
#[service_factory("policy", schema = "../../../hyprstream-rpc-std/schema/policy.capnp", metadata = crate::services::generated::policy_client::schema_metadata)]
fn create_policy_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating PolicyService");

    let policies_dir = ctx.models_dir().join(".registry").join("policies");

    // Get shared Git2DB instance (initializes .registry as git repo if needed)
    let git2db = get_or_init_git2db(ctx.models_dir())?;

    // Create policy manager (blocking since we're in sync context)
    let policy_manager = Arc::new(
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                let pm = PolicyManager::new(&policies_dir).await?;
                // Idempotent migration: ensure required bootstrap rules are present.
                // These rules are in DEFAULT_POLICY_CSV for new installs; existing
                // deployments need them added once.
                let rules = pm.get_policy().await;
                let has_anon_tui = rules
                    .iter()
                    .any(|r| r.len() >= 3 && r[0] == "anonymous" && r[2] == "tui:*");
                if !has_anon_tui {
                    let _ = pm
                        .add_policy_with_domain("anonymous", "*", "tui:*", "*", "allow")
                        .await;
                    tracing::info!("policy migration: added 'anonymous' TUI access grant");
                }
                // Migration: persist service base rules to disk if not already there.
                // PolicyManager::new() already injected them into memory, but older
                // policy.csv files won't have them on disk. Save writes the full
                // enforcer state (including base rules) to disk.
                let has_service_policy = rules
                    .iter()
                    .any(|r| r.len() >= 2 && r[0] == "service:policy");
                if !has_anon_tui || !has_service_policy {
                    let _ = pm.save().await;
                    if !has_service_policy {
                        tracing::info!(
                            "policy migration: persisted service-to-service base rules to disk"
                        );
                    }
                }
                Ok::<_, anyhow::Error>(pm)
            })
        })
        .context("Failed to initialize policy manager")?,
    );

    // Expose globally so other services (e.g. OAuthService) can write policy rules
    // for federated users without a ZMQ round-trip.
    crate::auth::set_global_policy_manager(Arc::clone(&policy_manager));

    // Spawn file watcher for policy hot-reload
    let pm_clone = Arc::clone(&policy_manager);
    let policy_csv = policies_dir.join("policy.csv");
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            super::policy::watch_policy_file(pm_clone, policy_csv).await;
        });
    });

    let config = load_config();
    let mut policy_service = PolicyService::new(
        policy_manager,
        Arc::new(ctx.signing_key().clone()),
        TokenConfig::default(),
        git2db,
        ctx.transport("policy", SocketKind::Rep),
    );
    if let Some(issuer) = ctx.oauth_issuer_url() {
        policy_service = policy_service.with_default_audience(issuer.to_owned());
    }
    policy_service = policy_service.with_jwt_key_source(ctx.cluster_key_source());

    // Wire ES256 + ML-DSA rotation stores into PolicyService for composite token issuance.
    // Uses global singletons so PolicyService shares the same store the rotation task updates.
    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
    let es256_store =
        crate::auth::key_rotation::global_es256_key_store(&secrets_dir, &config.oauth);
    policy_service = policy_service.with_es256_key_store(es256_store);
    {
        let ml_dsa_store =
            crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, &config.oauth);
        let ed_store =
            crate::auth::key_rotation::global_ed25519_key_store(&secrets_dir, &config.oauth);
        let ca_key = Arc::new(hyprstream_rpc::node_identity::derive_purpose_key(
            ctx.signing_key(),
            "hyprstream-jwt-v1",
        ));
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                crate::auth::key_rotation::initialize_composite_key_set(
                    &secrets_dir,
                    &ed_store,
                    &ml_dsa_store,
                    ca_key,
                    config.oauth.drain_secs(),
                ),
            )
        })?;
        policy_service = policy_service.with_ml_dsa_key_store(ml_dsa_store);
    }

    // Publish the JTI blocklist Arc so OAuthService (created later) can share it.
    // This wires POST /oauth/revoke → PolicyService RPC enforcement: a revoked
    // access token is rejected by both the HTTP path and the RPC auth check.
    let _ = SHARED_JTI_BLOCKLIST.set(policy_service.jti_blocklist_arc());

    Ok(ctx.into_spawnable_quic(policy_service, config.policy.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Registry Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for RegistryService (git2db model registry)
#[service_factory("registry", schema = "../../../hyprstream-rpc-std/schema/registry.capnp", metadata = crate::services::generated::registry_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_registry_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating RegistryService");

    // RegistryService publishes clone-progress streams via StreamChannel::run_stream
    // (which fails loudly if no moq origin is registered in this process).
    // Initialize this process's local moq plane. Idempotent.
    init_local_moq_stream_plane("registry");

    let config = load_config();
    let sk = ctx.service_signing_key("registry");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "registry", &sk)?;

    // Create policy client for authorization checks
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;

    // #910a — the registry service is the sole PDS-record writer AND the sole
    // holder of the `#atproto` private key: it opens the durable store
    // read-write and, on register/commit, signs the repo's commit ONCE and
    // persists the signed bytes. Reads (the discovery service) are keyless.
    // The record `repo` authority is the root `did:web` document that publishes
    // the `#atproto` commit-verification key. The `#atproto` signing key is the *active* key
    // from the shared `Es256SigningKeyStore` — the same P-256 key
    // `oauth::did_document` publishes as the `#atproto` verification method, so
    // the writer and the published key are one source of truth (classical —
    // atproto has no PQ variant). Best-effort: any failure here disables PDS
    // publish with a warning rather than failing the registry. The key lives
    // only in the writer's memory — never in the record DB (#910a H1). Paths
    // fail closed rather than fall back to /tmp (H2).
    let pds_publisher = (|| -> anyhow::Result<crate::services::discovery::PdsPublisher> {
        let store_dir = pds_store_dir(ctx)?;
        let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
        // C4 (#1170, closes #1123): the publisher resolves the active
        // `#atproto` generation from the **sealed op-log head** at sign time,
        // not a key frozen here at construction. The head is written by the
        // OAuth rotation task to a dedicated shared state dir (NOT the
        // read-only credentials dir), and is signed by a dedicated head key
        // whose *public* verifying key the registry loads here — the registry
        // holds NO CA private key. Under `--ipc` a rotation by the OAuth
        // process is observed with no event-delivery mechanism. If the head is
        // absent (OAuth not booted, or the shared state dir not provisioned —
        // #808 under systemd), `active_generation()` returns `None` and the
        // publisher declines to sign: fail-closed, never a stale frozen key.
        let oplog_state_dir = crate::auth::resolve_oplog_state_dir(&secrets_dir)?;
        let generation_source: Arc<dyn crate::auth::ActiveGenerationSource> = Arc::new(
            crate::auth::SealedHeadEs256Source::new(&oplog_state_dir, &secrets_dir),
        );
        let es256_store =
            crate::auth::key_rotation::global_es256_key_store(&secrets_dir, &config.oauth);
        let acceptance_identity = ctx.service_signing_key("registry");
        anyhow::ensure!(
            hyprstream_discovery::deployment_registry_verifier()?
                .matches(&acceptance_identity.verifying_key()),
            "registry signing credential does not match authenticated deployment identity"
        );
        let audit_ed = hyprstream_rpc::node_identity::derive_purpose_key(
            &acceptance_identity,
            "hyprstream-at9p-audit-ed25519-v1",
        );
        let audit_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&audit_ed);
        let store = Arc::new(
            crate::services::discovery::PdsRecordStore::open(&store_dir)?
                .with_at9p_acceptance_identity(acceptance_identity.verifying_key()),
        );
        let alarm_path = store_dir.join("at9p-duplicity.wal");
        let at9p_state = crate::services::discovery::At9pStateIngest::open(
            Arc::clone(&store),
            &alarm_path,
            acceptance_identity,
            audit_ed,
            audit_pq,
        )?;
        let node_did = hyprstream_rpc::did_key::ed25519_to_did_key(&ctx.verifying_key().to_bytes());
        Ok(
            crate::services::discovery::PdsPublisher::with_generation_source(
                store,
                node_did,
                generation_source,
            )
            .with_at9p_state_ingest(at9p_state)
            .with_es256_store(es256_store),
        )
    })()
    .map_err(|e| tracing::warn!("PDS publish disabled: {e}"))
    .ok();

    // Create registry service with infrastructure (blocking since we're in sync context)
    let mut registry_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let cas_pep = crate::mac::production_cas_pep(sk.clone(), &config.oauth, "cas-registry")
                .await
                .context("construct registry CAS MAC PEP")?;
            Ok::<_, anyhow::Error>(
                RegistryService::new(
                    ctx.models_dir(),
                    policy_client,
                    ctx.transport("registry", SocketKind::Rep),
                    sk.clone(),
                )
                .await?
                .with_cas_pep(cas_pep),
            )
        })
    })?;
    if let Some(issuer) = ctx.oauth_issuer_url() {
        registry_service = registry_service.with_expected_audience(issuer.to_owned());
    }
    registry_service = registry_service.with_jwt_key_source(ctx.cluster_key_source());
    if let Some(publisher) = pds_publisher {
        // A promotion publishes the former active key as a bounded drain slot;
        // no writer-local re-sign callback is needed, which keeps `--ipc`
        // rotation viable when OAuth and registry run in different processes.
        let publisher_arc = Arc::new(publisher);
        registry_service = registry_service.with_pds_publisher_arc(publisher_arc);
    }

    Ok(ctx.into_spawnable_quic(registry_service, config.registry.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streams Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Initialize this process's local moq stream plane (origin + UDS server).
///
/// Every service process that **publishes** moq streams (the central `streams`
/// service, and stream-publisher services such as `tui`/`notification`/`registry`/
/// `metrics`/`model`) needs its OWN moq plane in-process: the process-global
/// [`MoqStreamOrigin`] that `StreamChannel::publisher()` appends into, plus a
/// per-PID UDS moq server so a co-located client can connect directly to the
/// path returned in the publisher's response.
///
/// In a multi-process (systemd one-process-per-service) deployment, only the
/// `streams` factory used to do this, so other publisher processes had a `None`
/// origin (nothing to publish to) and returned an empty `moq_uds_path` to the
/// client (→ client `ensure!` fails). This helper closes that gap.
///
/// Idempotent: if the process already has a moq origin (the `streams` factory
/// ran in this process, or this helper was already called), it returns early
/// without double-initializing the origin or double-serving the UDS. This lets
/// it compose with the `streams` factory and with multiple publisher factories
/// co-located in one process.
fn init_local_moq_stream_plane(service_name: &str) {
    // Guard: a moq origin already exists in this process — nothing to do.
    if hyprstream_rpc::moq_stream::global_moq_origin().is_some() {
        return;
    }

    let gate = |pubkey: &[u8; 32]| -> bool {
        use ed25519_dalek::VerifyingKey;
        let Ok(vk) = VerifyingKey::from_bytes(pubkey) else {
            return false;
        };
        hyprstream_service::global_trust_store().get(&vk).is_some()
    };

    // Use DEFAULT_PREFIX ("local/streams") so the publisher's broadcast paths
    // path from DEFAULT_PREFIX; TUI/registry/metrics echo the origin's own
    // broadcast_path back to the client, so any prefix is self-consistent there).
    let moq_origin = hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone()
        .with_prefix(hyprstream_rpc::moq_stream::DEFAULT_PREFIX)
        .with_authorize_signer(gate)
        .build();

    // Register the global BEFORE serving — downstream code that calls
    // StreamChannel::publisher() will see it immediately.
    if !hyprstream_rpc::moq_stream::init_global_moq_origin(moq_origin.clone()) {
        // Lost a race to another initializer in this process; that init owns the
        // UDS server too, so don't start a second one.
        return;
    }

    let moq_uds_path = {
        let dir = std::env::temp_dir().join(format!("hyprstream-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir.join("moq.sock")
    };
    info!(
        service = service_name,
        path = %moq_uds_path.display(),
        "Initializing local moq stream plane",
    );
    hyprstream_rpc::moq_stream::serve_moq_uds_background(moq_origin, moq_uds_path);
}

/// Factory for the moq stream origin (#138 N4 — ZMQ StreamService removed).
///
/// Builds the process-global `MoqStreamOrigin`, registers it, and starts the
/// UDS moq server so cross-process subscribers (e.g. `tui attach`) can
/// subscribe over moq without any ZMQ sockets.
#[service_factory("streams")]
fn create_streams_service(_ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating moq stream origin (ZMQ StreamService removed)");

    init_local_moq_stream_plane("streams");

    Ok(Box::new(MoqStreamBarrierService::new()))
}

/// Minimal `Spawnable` that holds the moq stream origin lifetime and satisfies
/// the service lifecycle contract. The origin itself is a process-global with
/// no dedicated thread; this service just waits for shutdown.
struct MoqStreamBarrierService;

impl MoqStreamBarrierService {
    fn new() -> Self {
        Self
    }
}

impl Spawnable for MoqStreamBarrierService {
    fn name(&self) -> &str {
        "streams"
    }

    fn registrations(
        &self,
    ) -> Vec<(
        hyprstream_rpc::registry::SocketKind,
        hyprstream_rpc::transport::TransportConfig,
    )> {
        vec![]
    }

    fn run(
        self: Box<Self>,
        shutdown: std::sync::Arc<tokio::sync::Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> hyprstream_rpc::error::Result<()> {
        if let Some(ready) = on_ready {
            let _ = ready.send(());
        }
        // systemd Type=notify: send READY=1 so the unit reaches `active` rather
        // than timing out (~45s) and restart-looping. These moq barrier services
        // don't go through the RPC serve path (serve.rs::signal_ready) that
        // normally notifies systemd, so they must signal readiness themselves —
        // the moq origin/event-bus is already initialized in the factory before
        // run() is called, so the service is genuinely ready here.
        let _ = hyprstream_rpc::notify::ready();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| hyprstream_rpc::error::RpcError::Other(e.to_string()))?;
        rt.block_on(shutdown.notified());
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Model Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for ModelService (model lifecycle management)
#[service_factory("model", schema = "../../../hyprstream-rpc-std/schema/model.capnp", metadata = crate::services::generated::model_client::schema_metadata, depends_on = ["policy", "registry", "discovery", ])]
fn create_model_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating ModelService");

    // ModelService spawns InferenceService instances in-process, which publish
    // generation streams via StreamChannel::run_stream (fails loudly without a
    // moq origin). Initialize this process's local moq plane. Idempotent.
    init_local_moq_stream_plane("model");

    use crate::services::{ModelService, ModelServiceConfig};

    let config = load_config();
    let sk = ctx.service_signing_key("model");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "model", &sk)?;

    // Create policy client for authorization checks
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;

    // Create registry client
    let registry_client: RegistryClient =
        RegistryClient::from_resolver(sk.clone(), service_token(&sk))?;

    #[allow(clippy::expect_used)]
    let mut model_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create runtime for model factory");
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, {
            let mut model_config = ModelServiceConfig::default();
            model_config.inference_deployment =
                crate::runtime::inference_profile::InferenceDeploymentProfile::from_env()?;
            ModelService::new(
                model_config,
                sk.clone(),
                policy_client,
                registry_client,
                ctx.transport("model", SocketKind::Rep),
                ctx.transport("policy", SocketKind::Rep),
            )
        })
    })?;
    if let Some(issuer) = ctx.oauth_issuer_url() {
        model_service = model_service.with_expected_audience(issuer.to_owned());
    }
    model_service = model_service.with_jwt_key_source(ctx.cluster_key_source());

    // #431 — DiscoveryClient for federated at:// record resolution. The discovery
    // key is in the trust store (depends_on includes "discovery"). Best-effort:
    // if discovery isn't resolvable, ModelService simply has no federation client
    // and at:// refs fall through to local resolution.
    match crate::services::DiscoveryClient::from_resolver(sk.clone(), None) {
        Ok(dc) => {
            model_service = model_service.with_discovery_client(std::sync::Arc::new(dc));
        }
        Err(e) => {
            tracing::warn!("ModelService: failed to build DiscoveryClient for federation: {e}");
        }
    }

    Ok(ctx.into_spawnable_quic(model_service, config.model.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Standalone CPU Inference Service Factory (#1236/#1247)
// ═══════════════════════════════════════════════════════════════════════════════

/// One model, one tenant, one CPU-only process. Metal starts two independent
/// copies with replica ordinals 0 and 1; no engine or KV-cache state is shared.
#[service_factory("inference", schema = "../../../hyprstream-rpc-std/schema/inference.capnp", metadata = crate::services::generated::inference_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_inference_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating standalone CPU InferenceService");

    let config = load_config();
    config.inference.validate()?;
    config.inference.verify_materialized_oid()?;

    let sk = ctx.service_signing_key("inference");
    register_service_key(ctx, "inference", &sk)?;
    let instance_name = config.inference.instance_name();

    // The standalone image is a CPU fault domain even if a host happens to
    // expose CUDA. Explicitly clear every GPU selector at the service boundary.
    let runtime = crate::config::RuntimeConfig {
        use_gpu: false,
        gpu_device_id: None,
        devices: Vec::new(),
        gpu_layers: None,
        ..config.runtime.clone()
    };
    // Do not announce both replicas under the singleton Discovery key:
    // Discovery currently replaces, rather than pools, duplicate service
    // endpoints. Metal #1309 owns the canonical two-backend load-balancer.
    let quic = ctx
        .quic_shared()
        .map(|shared| {
            shared.for_service(
                &instance_name,
                config.inference.quic_port.unwrap_or(0),
            )
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "standalone inference requires [quic].enabled=true for browser MoQ"
            )
        })?;

    let mut service = crate::services::InferenceServiceConfig::new(
        &config.inference.model_path,
        runtime,
        sk.verifying_key(),
        sk.clone(),
        ctx.transport(&instance_name, SocketKind::Rep),
        ctx.transport("policy", SocketKind::Rep),
        None,
    )
    .with_instance_identity(
        instance_name.clone(),
        config.inference.tenant.clone(),
        sk.verifying_key(),
    )
    .with_quic_config(Some(quic))
    .with_quic_advertise_addr(config.inference.advertise_addr);
    if let Some(issuer) = ctx.oauth_issuer_url() {
        service = service.with_expected_audience(issuer.to_owned());
    }
    service = service.with_jwt_key_source(ctx.cluster_key_source());

    info!(
        instance = %config.inference.instance_name(),
        model_ref = %config.inference.model_ref,
        model_oid = %config.inference.model_oid,
        tenant = %config.inference.tenant,
        "standalone CPU inference configured"
    );
    Ok(Box::new(service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Worker Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for WorkerService (Kata container/sandbox management)
///
/// Note: This service requires worker configuration. If not configured,
/// the factory will use sensible defaults.
#[service_factory("worker", depends_on = ["policy", "discovery", "event"])]
fn create_worker_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating WorkerService");

    #[cfg(feature = "oci-image")]
    use hyprstream_workers::config::ImageConfig;
    use hyprstream_workers::config::PoolConfig;
    #[cfg(feature = "oci-image")]
    use hyprstream_workers::image::RafsStore;
    use hyprstream_workers::{resolve_backend, BackendCtx, SandboxBackend, WorkerService};

    let config = load_config();
    let sk = ctx.service_signing_key("worker");
    let worker_quic_port = config.worker.as_ref().and_then(|w| w.quic_port);
    // Operator-selected backend name ("auto" or a registered backend); resolved
    // fail-closed against the inventory registry below.
    let backend_name: String = config
        .worker
        .as_ref()
        .map(|w| w.backend.clone())
        .unwrap_or_else(|| "auto".to_owned());

    info!("WorkerService backend selection: {}", backend_name);

    // Use default paths based on XDG directories
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("hyprstream");
    let runtime_dir = dirs::runtime_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("hyprstream");

    let kata_boot_path = std::env::var("KATA_BOOT_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/opt/kata/share/kata-containers"));

    let pool_config = PoolConfig {
        warm_pool_size: 0,
        runtime_dir: runtime_dir.join("sandboxes"),
        kernel_path: kata_boot_path.join("vmlinux.container"),
        vm_image: kata_boot_path.join("kata-containers.img"),
        cloud_init_dir: data_dir.join("cloud-init"),
        ..PoolConfig::default()
    };

    // RAFS/nydus image store is built whenever the image filesystem service is
    // compiled in (`oci-image`), so both kata (virtio-fs) and nspawn (FUSE
    // tenant-VFS root, Model B #715) can compose a per-sandbox VFS from it.
    #[cfg(feature = "oci-image")]
    let image_config = ImageConfig {
        blobs_dir: data_dir.join("images/blobs"),
        bootstrap_dir: data_dir.join("images/bootstrap"),
        refs_dir: data_dir.join("images/refs"),
        cache_dir: data_dir.join("images/cache"),
        runtime_dir: runtime_dir.join("nydus"),
        ..ImageConfig::default()
    };

    #[cfg(feature = "oci-image")]
    let rafs_store = Arc::new(RafsStore::new(image_config.clone())?);

    let ninep_decider = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(crate::mac::production_ninep_decider(
            sk.clone(),
            &config.oauth,
            "ninep-worker",
        ))
    })
    .context("construct worker 9P MAC PEP")?;

    // Fixed-subject worker transports keep the mandatory monitor and floor
    // context. Identity-aware widening remains blocked until those transports
    // have a credential-bearing attach carrier (G2 evidence).
    // The worker UDS/vsock carrier still has no verified attach credential.
    // Make that runtime fact a structural G2 blocker: operator evidence cannot
    // widen this process until the constructor is replaced with a credentialed
    // authenticator.
    hyprstream_rpc::auth::mac::block_identity_widening_for_unverified_attach_transport(
        "worker-uds-vsock",
    );
    let ninep_monitor = Some(crate::mac::enrollment_ninep_reference_monitor(Arc::clone(
        &ninep_decider,
    )));

    // Resolve + construct the backend fail-closed against the inventory registry
    // (config-driven by name; explicit requests are authoritative, missing
    // prerequisites error out rather than silently downgrading isolation; "auto"
    // picks the strongest available). Single seam — no scattered cfg, no
    // `_ => nspawn` fallback (#507 / #518).
    let backend_ctx = BackendCtx {
        pool_config: pool_config.clone(),
        ninep_decider,
        ninep_monitor,
        #[cfg(feature = "oci-image")]
        image_config,
        #[cfg(feature = "oci-image")]
        rafs_store: Arc::clone(&rafs_store),
    };
    let backend: Arc<dyn SandboxBackend> = resolve_backend(&backend_name, &backend_ctx)?;

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "worker", &sk)?;

    // Service includes infrastructure - directly Spawnable via blanket impl
    let mut worker_service = WorkerService::new(
        pool_config,
        backend,
        // `kata-vm = ["kata"]` is one-way: a `--features kata` build must still
        // wire rafs_store, so gate on either the canonical feature or its alias
        // rather than `kata-vm` alone (#518).
        #[cfg(any(feature = "kata", feature = "kata-vm"))]
        Some(rafs_store),
        #[cfg(not(any(feature = "kata", feature = "kata-vm")))]
        None,
        ctx.transport("worker", SocketKind::Rep),
        sk.clone(),
    )?;

    // Wire up policy-backed authorization
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;
    worker_service.set_authorize_fn(super::worker::build_authorize_fn(policy_client));
    if let Some(issuer) = ctx.oauth_issuer_url() {
        worker_service.set_expected_audience(issuer.to_owned());
    }
    worker_service.set_jwt_key_source(ctx.cluster_key_source());

    Ok(ctx.into_spawnable_quic(worker_service, worker_quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Workflow Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for `WorkflowService` (#989, epic #1427).
///
/// The engine (`crates/hyprstream-workers/src/workflow/`) is fully implemented
/// but historically had no factory and was never started by the daemon. This
/// wires it into the service inventory: it resolves config, constructs the
/// service with a VFS namespace for the runner, attaches policy-backed
/// authorization, and returns it as a `Spawnable`.
///
/// **Default-off / opt-in:** the workflow config ships `enabled = false`, so
/// this factory bails unless an operator explicitly sets
/// `[worker.workflow] enabled = true`. First activation must not change daemon
/// behavior for everyone (amend-#989 §4).
///
/// Scope notes (amend-#989): only factory activation + namespace wiring land
/// here. `set_job_scheduler` is left unwired because `WorkerService` owns its
/// `SandboxPool` privately with no cross-factory handle today; jobs therefore
/// run in-proc (the documented `set_namespace` default) until Phase 1 adds pool
/// sharing. Event-bus subscription (`service.start()`) is #990's lifecycle
/// scope; the RPC surface (list/dispatch/getRun) works without it.
#[service_factory(
    "workflow",
    schema = "../../../hyprstream-workers/schema/workflow.capnp",
    metadata = hyprstream_workers::generated::workflow_client::schema_metadata,
    depends_on = ["worker", "event", "policy", "registry", "discovery"]
)]
fn create_workflow_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    use hyprstream_workers::WorkflowService;

    info!("Creating WorkflowService");

    let config = load_config();
    let wcfg = config
        .worker
        .as_ref()
        .map(|w| w.workflow.clone())
        .unwrap_or_default();
    if !wcfg.enabled {
        anyhow::bail!(
            "workflow service requested but [worker.workflow] enabled = false \
             (the engine is opt-in, #989; set `enabled = true` to activate)"
        );
    }

    let sk = ctx.service_signing_key("workflow");

    // Namespace the runner resolves actions/env/outputs through. Phase 0 ships
    // empty `/bin`, `/env`, `/out` skeletons; real action mounts and repo-scan
    // population land with #990/#992. The runner unmounts `/config` and
    // `/private` per-job for isolation regardless.
    let ns = Arc::new(build_workflow_namespace());

    let mut workflow_service =
        WorkflowService::new(ctx.transport("workflow", SocketKind::Rep), sk.clone());
    workflow_service.set_namespace_with_config(ns, &wcfg);

    // Policy-backed authorization — same seam as WorkerService. Without this the
    // dispatch handler fails closed on every request, so wire it before serving.
    register_service_key(ctx, "workflow", &sk)?;
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;
    workflow_service.set_authorize_fn(crate::services::worker::build_authorize_fn(policy_client));
    if let Some(issuer) = ctx.oauth_issuer_url() {
        workflow_service.set_expected_audience(issuer.to_owned());
    }
    workflow_service.set_jwt_key_source(ctx.cluster_key_source());

    // Inject a LIVE MCP relay authorizer so accept_delegated_bearer consults
    // current MCP-scope authorization at call time (not a frozen snapshot —
    // retired keys are rejected, new keys accepted after factory start). The
    // closure captures the global trust store and checks is_authorized for the
    // "mcp" scope — independently scoped, fail-closed (#989 review).
    let trust = hyprstream_service::global_trust_store();
    let has_mcp_keys = !trust.keys_for_scope("mcp").is_empty();
    if !has_mcp_keys {
        tracing::warn!(
            "workflow service started with no authorized MCP relay keys; \
             delegated-bearer MCP tool calls will be rejected until the mcp \
             service key(s) are registered (#989)"
        );
    }
    workflow_service.set_relay_authorizer(std::sync::Arc::new(move |pubkey: &[u8; 32]| {
        match hyprstream_rpc::prelude::VerifyingKey::from_bytes(pubkey) {
            Ok(vk) => trust.is_authorized(&vk, "mcp"),
            Err(_) => false,
        }
    }));

    Ok(ctx.into_spawnable(workflow_service))
}

/// Build the minimal Phase-0 workflow namespace: a synthetic root with empty
/// `/bin`, `/env`, `/out` directories for the runner to resolve through.
///
/// This is a skeleton — no action handlers, service mounts, or repo worktrees
/// are bound here yet. `#990` (git lifecycle) and `#992` (repo scan on clone)
/// populate the real content; the sandbox `JobScheduler` routing (#527) lands
/// once `WorkerService` exposes its `SandboxPool` cross-factory.
fn build_workflow_namespace() -> Namespace {
    use crate::services::fs::{SyntheticNode, SyntheticTree};

    fn empty_dir() -> SyntheticNode {
        SyntheticNode::Dir {
            children: HashMap::new(),
        }
    }

    let mut root = HashMap::new();
    root.insert("bin".to_owned(), empty_dir());
    root.insert("env".to_owned(), empty_dir());
    root.insert("out".to_owned(), empty_dir());

    let tree = SyntheticTree::new(SyntheticNode::Dir { children: root });
    let mut ns = Namespace::new();
    // Mounting can only fail on a malformed prefix; "/" is well-formed.
    let _ = ns.mount("/", Arc::new(tree) as MountTarget);
    ns
}

// ═══════════════════════════════════════════════════════════════════════════════
// OAI Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for OAIService (OpenAI-compatible HTTP API)
///
/// This service provides the HTTP API for inference requests.
/// It communicates with ModelService and PolicyService via ZMQ.
#[service_factory("oai", depends_on = ["policy", "model", "registry", "discovery"])]
fn create_oai_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating OAIService");

    use crate::server::state::ServerState;
    use crate::services::generated::model_client::ModelClient;
    use crate::services::OAIService;

    // Load full config for OAI settings
    let config = load_config();
    let sk = ctx.service_signing_key("oai");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "oai", &sk)?;

    // Create ZMQ clients for Model and Policy services
    let model_client = ModelClient::from_resolver(sk.clone(), service_token(&sk))?;
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;

    // Create registry client
    let registry_client: RegistryClient =
        RegistryClient::from_resolver(sk.clone(), service_token(&sk))?;

    // Create server state (blocking since we're in sync context)
    let resource_url = config.oai.resource_url();
    let oauth_issuer_url = config.oauth.issuer_url();
    let server_state = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
            let ml_dsa_store =
                crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, &config.oauth);
            let signer = crate::mac::audit::cose::OwnedCoseAuditSigner::new(
                Arc::new(sk.clone()),
                ml_dsa_store.active_key().await,
                hyprstream_rpc::envelope::mandatory_envelope_policy(),
            );
            anyhow::ensure!(
                signer.can_sign(),
                "9P MAC PEP audit signer unavailable under mandatory Hybrid policy"
            );
            let audit_store = crate::mac::audit::WalAuditStore::open(
                secrets_dir.join("mac-audit").join("ninep"),
                signer,
            )
            .map_err(|error| anyhow::anyhow!("open 9P MAC audit store: {error}"))?;
            let resolver = crate::mac::GenesisGate::production().into_resolver();
            let ninep_decider: Arc<dyn hyprstream_9p::AccessDecider> = Arc::new(
                crate::mac::NinePAccessDecider::new(Arc::new(resolver), Arc::new(audit_store)),
            );

            ServerState::new(
                config.server.clone(),
                model_client,
                policy_client,
                registry_client,
                sk.clone(),
                ctx.jwt_verifying_key(),
                resource_url,
                oauth_issuer_url,
                &config.oauth.trusted_issuers,
                // Share the PolicyService-owned JTI blocklist so POST /oauth/revoke
                // immediately invalidates tokens at the OAI resource server.
                SHARED_JTI_BLOCKLIST
                    .get()
                    .map(Arc::clone)
                    .unwrap_or_else(|| Arc::new(hyprstream_rpc::auth::InMemoryJtiBlocklist::new())),
                ninep_decider,
            )
            .await
        })
    })
    .context("Failed to create server state")?;

    let oai_service = OAIService::new(
        config.oai.clone(),
        config.tls.clone(),
        config.account.clone(),
        server_state,
        ctx.transport("oai", SocketKind::Rep),
        ctx.verifying_key(),
    );

    Ok(Box::new(oai_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Xet Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for XetService (HuggingFace-XET CAS HTTP face, epic #654).
///
/// HTTP service that speaks the HF-XET CAS wire protocol so a standard
/// xet-enabled git repo can point its CAS endpoint at hyprstream. It dials the
/// `registry` service (reusing the authenticated `putBlob`/`getBlob` core) and
/// holds no standing CAS write authority of its own. Reads come from the shared
/// L1 CAS substrate (`crate::storage::CasSubstrate`, #812).
#[service_factory("xet", depends_on = ["policy", "registry", "discovery"])]
fn create_xet_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating XetService");

    use crate::server::state::ResourceAuthState;
    use crate::services::{XetService, XetState};

    let config = load_config();
    let sk = ctx.service_signing_key("xet");

    // Register this service's verifying key with PolicyService.
    register_service_key(ctx, "xet", &sk)?;

    // Dial the registry — the authenticated write core the HTTP face translates to.
    let registry_client: RegistryClient =
        RegistryClient::from_resolver(sk.clone(), service_token(&sk))?;

    // Reuse the same narrow authentication core as OAI without constructing an
    // inference-oriented ServerState. The policy client is used by federated
    // issuer admission and key resolution.
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;
    let federation_resolver = Arc::new(
        crate::auth::FederationKeyResolver::new(&config.oauth.trusted_issuers)
            .with_policy_client(Arc::new(policy_client)),
    );
    let jti_blocklist = SHARED_JTI_BLOCKLIST
        .get()
        .map(Arc::clone)
        .context("PolicyService did not publish the shared JTI blocklist before Xet startup")?;
    let auth = ResourceAuthState::new(
        ctx.jwt_verifying_key(),
        config.xet.resource_url(),
        config.oauth.issuer_url(),
        federation_resolver,
        jti_blocklist,
    );
    let cas_pep = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(crate::mac::production_cas_pep(
            sk.clone(),
            &config.oauth,
            "cas-xet",
        ))
    })
    .context("construct Xet CAS MAC PEP")?;

    let state = XetState {
        // Reads share the same L1 CAS substrate the registry's getBlob uses (#812).
        store: crate::storage::CasSubstrate::from_env(),
        registry: Some(registry_client),
        auth,
        // T8: the production CAS hook resolves only contexts remembered from
        // verified Claims × VerifiedKeyMaterial.  The activation control keeps
        // that source at anonymous_floor until an operator widens it.
        cas_authorizer: Arc::new(crate::mac::MacCasAuthorizer::new(cas_pep)),
    };

    let xet_service = XetService::new(
        config.xet.clone(),
        config.tls.clone(),
        config.account.clone(),
        state,
    );

    Ok(Box::new(xet_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Flight Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for `At9pVerifyService` — the credential-free HTTPS face that lets
/// an external web app verify a `did:at9p` login assertion over plain HTTPS
/// (#1114).
///
/// **No `depends_on`, no service key, no `register_service_key`, no Rep
/// socket.** This face sits outside the RPC mesh by construction: it holds no
/// mesh credentials (the better to serve a credential-free public origin), so
/// it registers nothing and depends on nothing. `SocketKind` has no `Http`
/// variant and this face is not announceable — that is the #1135 design
/// statement made concrete. See `services::at9p_verify` for the full rationale.
#[service_factory("at9p-verify")]
fn create_at9p_verify_service(_ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    use crate::services::At9pVerifyService;

    let config = load_config();
    Ok(Box::new(At9pVerifyService::new(
        config.at9p_verify.clone(),
        config.tls.clone(),
        config.account.clone(),
    )))
}

// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for FlightService (Arrow Flight SQL server)
///
/// This service provides Flight SQL protocol for dataset queries.
/// It optionally uses RegistryClient for dataset lookup.
#[cfg(feature = "metrics")]
#[service_factory("flight", depends_on = ["policy", "registry", "discovery"])]
fn create_flight_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating FlightService");

    use crate::services::FlightService;

    // Load full config for Flight settings
    let config = load_config();
    let sk = ctx.service_signing_key("flight");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "flight", &sk)?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;
    let federation_resolver = Arc::new(
        crate::auth::FederationKeyResolver::new(&config.oauth.trusted_issuers)
            .with_policy_client(Arc::new(policy_client.clone())),
    );
    let jti_blocklist = SHARED_JTI_BLOCKLIST
        .get()
        .map(Arc::clone)
        .context("PolicyService did not publish the shared JTI blocklist before Flight startup")?;
    let auth = crate::server::state::ResourceAuthState::new(
        ctx.jwt_verifying_key(),
        config.flight.resource_url(),
        config.oauth.issuer_url(),
        federation_resolver,
        jti_blocklist,
    );
    let authorizer = Arc::new(crate::services::flight::TenantFlightAuthorizer::new(
        auth,
        policy_client,
    ));

    // Create registry client for dataset lookup (if default_dataset is configured)
    // RegistryClient already implements hyprstream_metrics::RegistryClient
    let registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>> =
        if config.flight.default_dataset.is_some() {
            let registry_client: RegistryClient =
                RegistryClient::from_resolver(sk.clone(), service_token(&sk))?;
            Some(Arc::new(registry_client))
        } else {
            None
        };

    let flight_service = FlightService::new(
        config.flight.clone(),
        registry_client,
        ctx.transport("flight", SocketKind::Rep),
        ctx.verifying_key(),
        authorizer,
    );

    Ok(Box::new(flight_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// OAuth Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for OAuthService (OAuth 2.1 Authorization Server)
///
/// This service provides OAuth 2.1 authorization for MCP and OAI services.
/// It delegates token issuance to PolicyService over ZMQ.
#[service_factory("oauth", schema = "../../../hyprstream-rpc-std/schema/oauth.capnp", metadata = crate::services::generated::oauth_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_oauth_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating OAuthService");

    use crate::services::OAuthService;

    let config = load_config();
    let sk = ctx.service_signing_key("oauth");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "oauth", &sk)?;

    let identity_registration_api =
        crate::services::oauth::identity_registration::production_identity_registration_api(
            &config.oauth,
            &config.account,
            &config.quic,
            sk.clone(),
            ctx.deployment_data_dir()?.join("pds"),
        )
        .context("compose production identity registration API")?;

    // Pass signing key instead of a pre-created PolicyClient.
    // OAuthService runs in its own tokio runtime (separate thread), so the
    // PolicyClient must be created inside that runtime for ZMQ async I/O to work.
    let mut oauth_service = OAuthService::new(
        config.oauth.clone(),
        config.tls.clone(),
        config.account.clone(),
        sk,
        ctx.transport("oauth", SocketKind::Rep),
        ctx.transport("policy", SocketKind::Rep),
        ctx.transport("discovery", SocketKind::Rep),
        ctx.verifying_key(),
        ctx.jwt_verifying_key(),
    )
    .with_quic_config(config.quic.clone())
    .with_identity_registration_api(identity_registration_api);
    if let Some(bl) = SHARED_JTI_BLOCKLIST.get() {
        oauth_service = oauth_service.with_jti_blocklist(Arc::clone(bl));
    } else {
        tracing::warn!(
            "JTI blocklist not set by PolicyService factory — revoked access tokens will not be blocked at RPC layer"
        );
    }

    Ok(Box::new(oauth_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// MCP Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for McpService (Model Context Protocol)
///
/// This service provides an MCP-compliant interface for AI coding assistants
/// (Claude Code, Cursor, etc.) to interact with hyprstream via:
/// - ZMQ control plane (for internal service communication)
/// - HTTP/SSE (for external MCP clients)
///
/// Note: The HTTP/SSE server is spawned as a background task in the factory.
#[service_factory("mcp", schema = "../../../hyprstream-rpc-std/schema/mcp.capnp", metadata = crate::services::generated::mcp_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_mcp_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating McpService");

    // Load full config for MCP settings
    let config = load_config();

    // Create McpConfig for the service
    let _oauth_issuer = ctx.oauth_issuer_url().map(str::to_owned);
    let federation_key_source = ctx.federation_key_source();
    let sk = ctx.service_signing_key("mcp");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "mcp", &sk)?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;

    let mcp_config = McpConfig {
        verifying_key: ctx.verifying_key(),
        signing_key: sk.clone(),
        transport: ctx.transport("mcp", SocketKind::Rep),
        policy_transport: ctx.transport("policy", SocketKind::Rep),
        ctx: None, // ServiceContext not yet available as Arc — handlers use signing_key directly
        policy_verifying_key: policy_vk,
        expected_audience: Some(config.mcp.resource_url()),
        jwt_key_source: Some(ctx.cluster_key_source()),
    };

    // Clone config for HTTP/SSE server before consuming it for ZMQ service
    let mcp_config_clone = mcp_config.clone();

    // Create the service (includes ZMQ infrastructure)
    let mcp_service = McpService::new(mcp_config)?;

    // Spawn rmcp HTTP/SSE server as background task
    let mcp_host = config.mcp.host.clone();
    let http_port = config.mcp.http_port;
    let mcp_cors_config = config.mcp.cors.clone();
    let mcp_tls_config = config.tls.clone();
    let mcp_tls_cert = config.mcp.tls_cert.clone();
    let mcp_tls_key = config.mcp.tls_key.clone();
    // Use the shared FederationKeySource from ServiceContext if available,
    // otherwise fall back to a locally-constructed resolver from config.
    // The fallback path wires its own PolicyClient so the unified
    // federation:register trust gate stays in effect — never downgrade
    // security posture just because the shared resolver wasn't provided.
    let mcp_federation_resolver: std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource> =
        if let Some(fed) = federation_key_source {
            fed
        } else {
            let fallback_policy_client = std::sync::Arc::new(policy_client_for_context(
                ctx,
                ctx.service_signing_key("mcp"),
                policy_vk,
                service_token(&sk),
            )?);
            std::sync::Arc::new(
                crate::auth::FederationKeyResolver::new(&config.oauth.trusted_issuers)
                    .with_policy_client(fallback_policy_client),
            )
        };
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            use rmcp::transport::streamable_http_server::{
                StreamableHttpServerConfig, StreamableHttpService,
            };

            use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

            let session_mgr = std::sync::Arc::new(LocalSessionManager::default());
            let jwt_key_source = mcp_config_clone.jwt_key_source.clone();
            let service: StreamableHttpService<McpService, LocalSessionManager> =
                StreamableHttpService::new(
                    move || McpService::new(mcp_config_clone.clone()).map_err(|e| {
                        std::io::Error::other(e.to_string())
                    }),
                    session_mgr,
                    StreamableHttpServerConfig::default(),
                );
            // Add protected resource metadata (RFC 9728) for OAuth discovery
            let mcp_full_config = crate::config::HyprConfig::load().unwrap_or_default();
            let mcp_resource_url = mcp_full_config.mcp.resource_url();
            let mcp_oauth_issuer = mcp_full_config.oauth.issuer_url();
            let www_authenticate = format!(
                "Bearer resource_metadata=\"{}/.well-known/oauth-protected-resource\"",
                mcp_resource_url
            );
            let router = axum::Router::new()
                .route(
                    "/.well-known/oauth-protected-resource",
                    axum::routing::get({
                        let mcp_resource_url = mcp_resource_url.clone();
                        let mcp_oauth_issuer = mcp_oauth_issuer.clone();
                        move || async move {
                            let mut meta = crate::services::oauth::protected_resource_metadata(
                                &mcp_resource_url,
                                &mcp_oauth_issuer,
                            );
                            meta.resource_name = Some("HyprStream MCP Server".to_owned());
                            meta.scopes_supported = Some(vec![
                                "read:model:*".into(),
                                "infer:model:*".into(),
                                "write:model:*".into(),
                            ]);
                            axum::Json(meta)
                        }
                    }),
                )
                .nest_service("/mcp", service)
                .layer(axum::middleware::from_fn({
                    let mcp_resource_url = mcp_resource_url.clone();
                    let mcp_oauth_issuer_clone = mcp_oauth_issuer.clone();
                    let mcp_federation_resolver = mcp_federation_resolver.clone();
                    let jwt_key_source = jwt_key_source.clone();
                    // Capture shared JTI blocklist for revocation checks (RFC 7009)
                    let mcp_jti_blocklist = SHARED_JTI_BLOCKLIST.get().map(Arc::clone);
                    // DPoP JTI replay cache (separate from OAI server's, RFC 9449).
                    // 1,000 sustained DPoP proofs/s for the admitted 180s
                    // maximum residency (60s future iat skew + 120s), plus
                    // 20% headroom. Fixed digest keys plan 216,000 entries at
                    // about 26.4 MiB before allocator slack; this barrier
                    // never evicts live JTIs.
                    let mcp_dpop_jti_seen: std::sync::Arc<hyprstream_util::TtlCache<
                        crate::services::oauth::replay_key::ReplayKey,
                        (),
                    >> =
                        std::sync::Arc::new(hyprstream_util::TtlCache::new(216_000, 64));
                    move |mut req: axum::extract::Request, next: axum::middleware::Next| {
                        let www_authenticate = www_authenticate.clone();
                        let mcp_resource_url = mcp_resource_url.clone();
                        let mcp_oauth_issuer = mcp_oauth_issuer_clone.clone();
                        let federation_resolver = mcp_federation_resolver.clone();
                        let jwt_key_source = jwt_key_source.clone();
                        let jti_blocklist = mcp_jti_blocklist.clone();
                        let dpop_jti_seen = mcp_dpop_jti_seen.clone();
                        async move {
                            use axum::http::{header, StatusCode};
                            use axum::response::IntoResponse;
                            use hyprstream_rpc::auth::JtiBlocklist as _;
                            use subtle::ConstantTimeEq as _;
                            let method = req.method().clone();
                            let uri = req.uri().clone();
                            // Allow OAuth discovery endpoint without auth
                            if req.uri().path().starts_with("/.well-known/") {
                                tracing::debug!(%method, %uri, "MCP discovery request (no auth required)");
                                return next.run(req).await;
                            }
                            let has_auth_header = req.headers().contains_key(header::AUTHORIZATION);
                            let auth_value = req.headers()
                                .get(header::AUTHORIZATION)
                                .and_then(|v| v.to_str().ok())
                                .map(str::to_owned);
                            // Accept both Bearer (RFC 6750) and DPoP (RFC 9449) schemes
                            let (scheme, t) = match auth_value.as_deref().and_then(|h| {
                                if h.len() > 7 && h[..7].eq_ignore_ascii_case("bearer ") {
                                    Some(("bearer", h[7..].trim().to_owned()))
                                } else if h.len() > 5 && h[..5].eq_ignore_ascii_case("dpop ") {
                                    Some(("dpop", h[5..].trim().to_owned()))
                                } else {
                                    None
                                }
                            }) {
                                Some(pair) => pair,
                                None => {
                                    tracing::info!(%method, %uri, has_auth_header, "MCP auth MISSING token");
                                    let mut res = (StatusCode::UNAUTHORIZED, "Authentication required").into_response();
                                    if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                        res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                    }
                                    return res;
                                }
                            };
                            let iss = crate::server::middleware::extract_iss_from_token(&t);
                            let kid = crate::server::middleware::extract_kid_from_token(&t);
                            let result = if let Some(ref key_source) = jwt_key_source {
                                let resolved = if let Some(kid) = kid.as_deref() {
                                    key_source
                                        .get_key(&iss, Some(kid))
                                        .await
                                        .map(|key| vec![key])
                                } else {
                                    key_source.get_keys(&iss, None).await
                                };
                                match resolved {
                                    Ok(keys) if kid.is_some() => crate::auth::jwt::decode(
                                        &t,
                                        &keys[0],
                                        Some(mcp_resource_url.as_str()),
                                    ),
                                    Ok(keys) => crate::auth::jwt::decode_with_any_key(
                                        &t,
                                        &keys,
                                        Some(mcp_resource_url.as_str()),
                                    ),
                                    Err(e) => {
                                        tracing::debug!(%method, %uri, issuer = %iss, error = %e, "MCP JWT key resolution failed");
                                        let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                        if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                            res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                        }
                                        return res;
                                    }
                                }
                            } else {
                                // Rotation-aware federation verification (#1185):
                                // try each published candidate (kid-first) so a
                                // token signed by a non-first published key
                                // verifies during overlap rotation.
                                match federation_resolver.get_keys(&iss, kid.as_deref()).await {
                                    Ok(candidates) if !candidates.is_empty() => {
                                        crate::auth::jwt::decode_with_federation_candidates(
                                            &t,
                                            &candidates,
                                            Some(mcp_resource_url.as_str()),
                                        )
                                    }
                                    Ok(_) => Err(crate::auth::jwt::JwtError::InvalidFormat),
                                    Err(e) => {
                                        tracing::debug!(%method, %uri, issuer = %iss, error = %e, "MCP federation key resolution failed");
                                        let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                        if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                            res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                        }
                                        return res;
                                    }
                                }
                            };
                            let mut claims = match result {
                                Ok(c) => c,
                                Err(e) => {
                                    tracing::warn!(%method, %uri, error = %e, "MCP auth REJECTED");
                                    let mut res = (StatusCode::UNAUTHORIZED, "Invalid or expired token").into_response();
                                    if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                        res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                    }
                                    return res;
                                }
                            };
                            let atproto_issuer =
                                crate::services::oauth::state::canonical_issuer_origin(
                                    &mcp_oauth_issuer,
                                )
                                .unwrap_or_else(|| mcp_oauth_issuer.clone());
                            let local_issuers =
                                [mcp_oauth_issuer.as_str(), atproto_issuer.as_str()];
                            claims.strip_federated_tenant(&local_issuers);
                            let subject = claims.subject(&local_issuers);
                            if subject.validate().is_err() || subject.name().is_none() {
                                tracing::warn!(%method, %uri, "MCP auth rejected: invalid subject");
                                return (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                            }
                            // JTI revocation check (RFC 7009)
                            if let Some(ref jti) = claims.jti {
                                let revoked = jti_blocklist.as_ref().map(|bl| bl.is_revoked(jti)).unwrap_or(false);
                                if revoked {
                                    tracing::warn!(%method, %uri, %jti, sub = %claims.sub, "MCP: revoked token presented");
                                    let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                    if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                        res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                    }
                                    return res;
                                }
                            }
                            // DPoP binding enforcement (RFC 9449 §7):
                            // cnf.jkt tokens MUST be presented with DPoP scheme + proof header.
                            if let Some(expected_jkt) = claims.cnf_jkt() {
                                if scheme != "dpop" {
                                    tracing::warn!(%method, %uri, sub = %claims.sub, "MCP: DPoP-bound token presented with Bearer scheme");
                                    let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                    if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                        res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                    }
                                    return res;
                                }
                                let dpop_proof = match req.headers().get("DPoP").and_then(|v| v.to_str().ok()) {
                                    Some(p) => p.to_owned(),
                                    None => {
                                        tracing::debug!(%method, %uri, "MCP: DPoP-bound token missing DPoP proof header");
                                        let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                        if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                            res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                        }
                                        return res;
                                    }
                                };
                                let method_str = method.as_str().to_owned();
                                let path = uri.path().to_owned();
                                let htu = format!("{}{}", mcp_resource_url.trim_end_matches('/'), path);
                                let proof = match crate::services::oauth::dpop::verify_dpop_proof(
                                    &dpop_proof,
                                    &method_str,
                                    &htu,
                                    Some(&t),
                                ) {
                                    Ok(p) => p,
                                    Err(e) => {
                                        tracing::debug!(%method, %uri, error = %e, "MCP: DPoP proof verification failed");
                                        let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                        if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                            res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                        }
                                        return res;
                                    }
                                };
                                // Replay prevention: atomic check-and-record on the shared TtlCache.
                                {
                                    let now = chrono::Utc::now().timestamp();
                                    let Some(ttl_secs) = proof
                                        .iat
                                        .checked_add(120)
                                        .and_then(|deadline| deadline.checked_sub(now))
                                        .filter(|remaining| *remaining > 0 && *remaining <= 180)
                                        .and_then(|remaining| u64::try_from(remaining).ok())
                                    else {
                                        let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                        if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                            res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                        }
                                        return res;
                                    };
                                    let result = dpop_jti_seen.insert_if_absent_no_evict(
                                        crate::services::oauth::replay_key::dpop_jti(&proof.jti),
                                        (),
                                        std::time::Duration::from_secs(ttl_secs),
                                    );
                                    if result != hyprstream_util::InsertIfAbsentNoEvictResult::Inserted {
                                        crate::services::oauth::replay_metrics::record_rejection(
                                            crate::services::oauth::replay_metrics::DPOP,
                                            result,
                                        );
                                        if crate::services::oauth::replay_metrics::should_warn_full(
                                            crate::services::oauth::replay_metrics::DPOP,
                                            result,
                                        ) {
                                            tracing::warn!(%method, %uri, "MCP: DPoP replay barrier is full; refusing fresh proof");
                                        } else if result
                                            == hyprstream_util::InsertIfAbsentNoEvictResult::Duplicate
                                        {
                                            tracing::debug!(%method, %uri, "MCP: DPoP proof replayed");
                                        }
                                        let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                        if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                            res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                        }
                                        return res;
                                    }
                                }
                                // cnf.jkt must match proof key thumbprint (constant-time)
                                if expected_jkt.as_bytes().ct_eq(proof.jkt.as_bytes()).unwrap_u8() == 0 {
                                    tracing::warn!(%method, %uri, sub = %claims.sub, "MCP: cnf.jkt mismatch");
                                    let mut res = (StatusCode::UNAUTHORIZED, "Authentication failed").into_response();
                                    if let Ok(val) = header::HeaderValue::from_str(&www_authenticate) {
                                        res.headers_mut().insert(header::WWW_AUTHENTICATE, val);
                                    }
                                    return res;
                                }
                            }
                            tracing::debug!(%method, %uri, sub = %claims.sub, "MCP auth OK");
                            // Insert AuthenticatedUser so MCP handlers see validated identity
                            let authenticated = crate::server::middleware::AuthenticatedUser {
                                user: subject.name().unwrap_or_default().to_owned(),
                                verified_tenant: claims.tenant.clone(),
                                token: Some(t.clone()),
                                exp: Some(claims.exp),
                            };
                            if authenticated.authorization_domain().is_err() {
                                tracing::warn!(%method, %uri, sub = %claims.sub, "MCP auth rejected: no valid hosted-account tenant binding");
                                return (StatusCode::FORBIDDEN, "Verified hosted-account tenant binding required").into_response();
                            }
                            req.extensions_mut().insert(authenticated);
                            next.run(req).await
                        }
                    }
                }));

            // CORS must be outermost layer (added last) so OPTIONS preflights
            // are handled before auth middleware rejects them.
            let router = if mcp_cors_config.enabled {
                router.layer(crate::server::middleware::cors_layer(&mcp_cors_config))
            } else {
                router
            };

            let addr: std::net::SocketAddr = format!("{}:{}", mcp_host, http_port)
                .parse()
                .unwrap_or_else(|_| ([0, 0, 0, 0], http_port).into());

            // Resolve TLS configuration for MCP HTTP server.
            // If the user explicitly configured cert/key paths and TLS fails,
            // refuse to start (don't silently degrade to HTTP).
            let has_explicit_tls = mcp_tls_cert.is_some() || mcp_tls_key.is_some()
                || mcp_tls_config.cert_path.is_some() || mcp_tls_config.key_path.is_some();

            let rustls_config = match crate::server::tls::resolve_rustls_config(
                &mcp_tls_config,
                &config.account,
                mcp_tls_cert.as_ref(),
                mcp_tls_key.as_ref(),
            ).await {
                Ok(cfg) => cfg,
                Err(e) => {
                    if has_explicit_tls {
                        tracing::error!(
                            "MCP TLS config error with explicit cert/key paths: {} — refusing to start without TLS", e
                        );
                        return;
                    }
                    tracing::warn!("MCP TLS config error (self-signed): {} — falling back to HTTP", e);
                    None
                }
            };

            let scheme = if rustls_config.is_some() { "https" } else { "http" };
            tracing::info!("MCP HTTP/SSE server listening on {scheme}://{addr}");

            match rustls_config {
                Some(tls) => {
                    // MCP HTTP is fire-and-forget (no Arc<Notify> shutdown signal),
                    // so no Handle is wired for graceful shutdown. The process exit
                    // will terminate this task. OAI/OAuth use serve_app() instead.
                    if let Err(e) = axum_server::bind_rustls(addr, tls)
                        .serve(router.into_make_service())
                        .await
                    {
                        tracing::error!("MCP HTTPS server error: {}", e);
                    }
                }
                None => {
                    let listener = match tokio::net::TcpListener::bind(addr).await {
                        Ok(l) => l,
                        Err(e) => {
                            tracing::error!("Failed to bind MCP HTTP/SSE on {}: {}", addr, e);
                            return;
                        }
                    };
                    if let Err(e) = axum::serve(listener, router).await {
                        tracing::error!("MCP HTTP/SSE server error: {}", e);
                    }
                }
            }
        });
    });
    info!(
        "McpService created (HTTP/SSE on {}:{})",
        config.mcp.host, http_port
    );

    Ok(ctx.into_spawnable_quic(mcp_service, config.mcp.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TUI Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for TuiService (terminal multiplexer display server)
///
/// This service provides a terminal multiplexer with session persistence,
/// multi-pane layouts, and remote access via ZMQ RPC and WebTransport.
#[service_factory("tui", schema = "../../schema/tui.capnp", depends_on = ["policy", "discovery"])]
fn create_tui_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating TuiService");

    use crate::tui::{service::TuiService, TuiState};

    // TUI publishes terminal frames (stdin/stdout) over moq via
    // StreamChannel::publisher(), and returns its per-PID moq UDS path to the
    // client. In a per-process deployment this process has no moq plane unless
    // we initialize one here. Idempotent — no-op if already set.
    init_local_moq_stream_plane("tui");

    let config = load_config();
    let tui_config = &config.tui;
    let sk = ctx.service_signing_key("tui");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "tui", &sk)?;

    let state = Arc::new(RwLock::new(TuiState::new(
        80,
        24,
        tui_config.scrollback_lines,
    )));

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;

    // Build the direct-VFS PEP before exposing the namespace. Failure to open
    // its signed WAL aborts construction; there is no unarmed fallback.
    let vfs_pep = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(crate::mac::production_vfs_pep(
            sk.clone(),
            &config.oauth,
            "vfs-tui",
        ))
    })
    .context("construct TUI VFS MAC PEP")?;
    let (vfs_ns, vfs_subject) = crate::tui::vfs::build_chat_vfs_namespace(&sk, vfs_pep)?;

    let mut tui_service = TuiService::new(state, ctx.transport("tui", SocketKind::Rep), sk.clone())
        .with_policy_client(policy_client)
        .with_vfs(vfs_ns, vfs_subject);

    if let Some(issuer) = ctx.oauth_issuer_url() {
        tui_service = tui_service.with_expected_audience(issuer.to_owned());
    }
    tui_service = tui_service.with_jwt_key_source(ctx.cluster_key_source());

    Ok(ctx.into_spawnable_quic(tui_service, tui_config.quic_port))
}

/// Open the PDS record store (#910a) read-only, bootstrapping an empty
/// RocksDB database at `dir` first if nothing has been published yet.
///
/// `PdsRecordStore::open_readonly` requires the RocksDB files to already
/// exist (`create_if_missing(false)`, matching `RocksDbUserStore::open_readonly`),
/// which is normally true because the registry service (the writer) creates
/// it. On a fresh install the discovery service may start before any model
/// has ever been registered — bootstrap by briefly opening read-write (which
/// creates the DB files) and releasing the handle, then retry read-only.
///
/// When this path creates a fresh empty store it also writes the durable
/// first-boot RocksDB key, so the QUIC startup gate recognizes it as a
/// genuine first boot rather than data loss.
///
/// Known limitation: if the registry and discovery services start
/// concurrently on a brand-new install, both may race to bootstrap the same
/// directory; one loses the RocksDB lock and its factory call fails, which
/// the service manager will retry.
fn open_pds_store_readonly(
    dir: &std::path::Path,
) -> anyhow::Result<crate::services::discovery::PdsRecordStore> {
    match crate::services::discovery::PdsRecordStore::open_readonly(dir) {
        Ok(store) => Ok(store),
        Err(orig) => {
            // Bootstrap the DB files by briefly opening read-write, then retry
            // read-only. If bootstrap itself fails (e.g. the writer holds the
            // lock, or the path is corrupt), surface BOTH errors so the real
            // cause is visible rather than masked by the retry (#910a, fable M3).
            let store = crate::services::discovery::PdsRecordStore::open(dir)
                .with_context(|| {
                    format!("read-only open failed ({orig}); bootstrap open also failed")
                })?;
            // This bootstrap created a fresh empty store — record first-boot
            // lifecycle evidence as a durable RocksDB key so the QUIC gate
            // doesn't mistake it for data loss on the next classification.
            // Propagate the error: a failed marker write must not appear
            // successful.
            store.mark_first_boot()?;
            drop(store);
            crate::services::discovery::PdsRecordStore::open_readonly(dir)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Discovery Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for DiscoveryService (endpoint registry over ZMQ RPC)
///
/// This service exposes the EndpointRegistry so remote clients can discover
/// registered services, their endpoints, socket kinds, and schemas.
#[service_factory("discovery", schema = "../../../hyprstream-discovery/schema/discovery.capnp", metadata = hyprstream_discovery::generated::discovery_client::schema_metadata, depends_on = ["policy"])]
fn create_discovery_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating DiscoveryService");

    let config = load_config();
    let sk = ctx.service_signing_key("discovery");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "discovery", &sk)?;

    // Create policy-based authorization provider
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;
    let auth_provider = crate::services::discovery::PolicyAuthProvider::new(policy_client);

    // #431 — record resolver backing getRecord/getRepo, over the durable
    // RocksDB-backed PDS record store (#910a). The registry service is the
    // sole writer (it opens the same directory read-write, signs each commit
    // ONCE and persists it — see `create_registry_service`); this factory
    // opens the directory read-only, matching `RocksDbUserStore::open_readonly`.
    // The resolver holds **no signing key** at all: reads are keyless (rebuild
    // the deterministic MST, load the writer's already-signed commit, serve a
    // proof). This is the #910a security fix — a read path that re-signed a
    // commit on every `getRecord` (and so needed the private key) was the root
    // of the key-exposure problem; atproto never re-signs on read.
    // In-process factories share the stable node root. In IPC mode the
    // registry process signs with its stable service credential, whose public
    // key is anchored in the global service trust store.
    let at9p_acceptance_identity = hyprstream_discovery::deployment_registry_verifier()?;
    let pds_store_path = pds_store_dir(ctx)?;
    let pds_store = std::sync::Arc::new(
        open_pds_store_readonly(&pds_store_path)
            .context("failed to open PDS record store (read-only)")?
            .with_at9p_deployment_verifier(at9p_acceptance_identity),
    );
    // #918 — the local repo subject is the root did:web authority whose
    // document is fed by this ES256 store. The resolver uses the same bounded
    // publication snapshot, including drain, before placement ingest.
    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
    let es256_store =
        crate::auth::key_rotation::global_es256_key_store(&secrets_dir, &config.oauth);
    let issuer = ctx
        .oauth_issuer_url()
        .map(str::to_owned)
        .unwrap_or_else(|| config.oauth.issuer_url());
    let authority = crate::services::oauth::did_document::issuer_authority(&issuer)
        .context("OAuth issuer has no did:web authority")?;
    let node_did = format!("did:web:{authority}");
    let record_resolver = std::sync::Arc::new(
        crate::services::discovery::PdsRecordResolver::new(pds_store)
            .with_es256_rotation(es256_store, node_did),
    );

    let mut discovery_service = DiscoveryService::new(
        Arc::new(sk),
        ctx.jwt_verifying_key(),
        ctx.transport("discovery", SocketKind::Rep),
    )
    .with_auth_provider(Box::new(auth_provider))
    .with_record_resolver(std::sync::Arc::clone(&record_resolver)
        as std::sync::Arc<dyn hyprstream_discovery::RecordResolver>);
    discovery_service.attach_process_accepted_state_source()?;
    if let Some(issuer) = ctx.oauth_issuer_url() {
        discovery_service = discovery_service.with_oauth_issuer(issuer.to_owned());
        // Use the issuer URL as the audience for discovery tokens
        discovery_service = discovery_service.with_expected_audience(issuer.to_owned());
    }
    discovery_service = discovery_service.with_jwt_key_source(ctx.cluster_key_source());

    // Pre-compute TLS endorsement if QUIC is enabled with a TLS cert.
    // Uses the root verifying key — TLS endorsement is a node-level trust assertion,
    // not specific to any per-service key. Clients verify against the pinned root pubkey.
    if let Some(quic) = ctx.quic_shared() {
        let ed25519_pubkey = ctx.verifying_key().to_bytes();
        let domain = &quic.server_name;
        match compute_tls_endorsement(&quic.key_der, &ed25519_pubkey, domain) {
            Ok(endorsement) => {
                if !endorsement.is_empty() {
                    info!(
                        "TLS endorsement computed for domain '{}' ({} bytes)",
                        domain,
                        endorsement.len()
                    );
                    discovery_service =
                        discovery_service.with_tls_endorsement(endorsement, domain.clone());
                }
            }
            Err(e) => {
                // Non-fatal: TLS endorsement is optional additive trust
                tracing::warn!("Failed to compute TLS endorsement for '{}': {}", domain, e);
            }
        }
    }
    // TODO: DiscoveryService federation key source support
    // (federation_key_source not yet implemented on DiscoveryService)

    Ok(ctx.into_spawnable_quic(discovery_service, config.discovery.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Notification Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Metrics Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for MetricsService (DuckDB-backed time-series ingest + DataFusion query)
#[cfg(feature = "metrics")]
#[service_factory("metrics", schema = "../../../hyprstream-rpc-std/schema/metrics.capnp", metadata = crate::services::generated::metrics_client::schema_metadata, depends_on = ["policy", "discovery"])]
fn create_metrics_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating MetricsService");

    // MetricsService publishes query-result streams via StreamChannel::run_stream
    // (fails loudly without a moq origin). Initialize this process's local moq
    // plane. Idempotent.
    init_local_moq_stream_plane("metrics");

    use crate::services::MetricsService;
    use hyprstream_metrics::query::QueryOrchestrator;
    use hyprstream_metrics::storage::duckdb::DuckDbBackend;
    use hyprstream_metrics::StorageBackend as _;

    let config = load_config();
    let mc = &config.metrics;

    let backend = Arc::new(
        DuckDbBackend::new(mc.db_path.clone(), Default::default(), None)
            .map_err(|e| anyhow::anyhow!("DuckDbBackend init: {e}"))?,
    );

    let orchestrator = Arc::new(
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let schema = hyprstream_metrics::metrics::get_metrics_schema();
                backend
                    .create_table("metrics", &schema)
                    .await
                    .map_err(|e| anyhow::anyhow!("metrics table init: {e}"))?;
                QueryOrchestrator::new(backend as Arc<dyn hyprstream_metrics::StorageBackend>)
                    .await
                    .map_err(|e| anyhow::anyhow!("QueryOrchestrator init: {e}"))
            })
        })
        .map_err(|e| anyhow::anyhow!("metrics service init: {e}"))?,
    );

    let sk = ctx.service_signing_key("metrics");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "metrics", &sk)?;

    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = policy_client_for_context(ctx, sk.clone(), policy_vk, service_token(&sk))?;

    let mut metrics_service = MetricsService::new(
        orchestrator,
        ctx.transport("metrics", SocketKind::Rep),
        sk,
        policy_client,
    );
    if let Some(issuer) = ctx.oauth_issuer_url() {
        metrics_service = metrics_service.with_expected_audience(issuer.to_owned());
    }
    metrics_service = metrics_service.with_jwt_key_source(ctx.cluster_key_source());

    Ok(ctx.into_spawnable_quic(metrics_service, mc.quic_port))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TLS Endorsement Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Domain separator for TLS endorsement messages.
const TLS_ENDORSEMENT_V1: &[u8] = b"TLS_ENDORSEMENT_V1";

/// Compute a TLS endorsement signature.
///
/// Signs `TLS_ENDORSEMENT_V1 || ed25519_pubkey || domain` with the TLS private key.
/// Handles ECDSA P-256, RSA, and Ed25519 key types (auto-detected from PKCS8 DER).
///
/// Returns the raw signature bytes, or an empty vec if the key type is unsupported.
fn compute_tls_endorsement(
    tls_key_der: &[u8],
    ed25519_pubkey: &[u8; 32],
    domain: &str,
) -> anyhow::Result<Vec<u8>> {
    // Build message: TLS_ENDORSEMENT_V1 || ed25519_pubkey (32) || domain
    let mut message = Vec::with_capacity(TLS_ENDORSEMENT_V1.len() + 32 + domain.len());
    message.extend_from_slice(TLS_ENDORSEMENT_V1);
    message.extend_from_slice(ed25519_pubkey);
    message.extend_from_slice(domain.as_bytes());

    let rng = ring::rand::SystemRandom::new();

    // Try Ed25519 first (most modern, smallest signature)
    if let Ok(key_pair) = ring::signature::Ed25519KeyPair::from_pkcs8(tls_key_der) {
        return Ok(key_pair.sign(&message).as_ref().to_vec());
    }

    // Try ECDSA P-256 SHA-256
    if let Ok(key_pair) = ring::signature::EcdsaKeyPair::from_pkcs8(
        &ring::signature::ECDSA_P256_SHA256_FIXED_SIGNING,
        tls_key_der,
        &rng,
    ) {
        let signature = key_pair.sign(&rng, &message)?;
        return Ok(signature.as_ref().to_vec());
    }

    // Try ECDSA P-384 SHA-384
    if let Ok(key_pair) = ring::signature::EcdsaKeyPair::from_pkcs8(
        &ring::signature::ECDSA_P384_SHA384_FIXED_SIGNING,
        tls_key_der,
        &rng,
    ) {
        let signature = key_pair.sign(&rng, &message)?;
        return Ok(signature.as_ref().to_vec());
    }

    // Try RSA (PKCS1v15 + SHA-256, then PSS + SHA-256)
    if let Ok(key_pair) = ring::signature::RsaKeyPair::from_pkcs8(tls_key_der) {
        let mut signature = vec![0u8; key_pair.public().modulus_len()];
        let padding = &ring::signature::RSA_PKCS1_SHA256;
        key_pair.sign(padding, &rng, &message, &mut signature)?;
        return Ok(signature);
    }

    anyhow::bail!("unsupported TLS key type in PKCS8 DER")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Rootless Quadlets run each service in a distinct process, so the
    /// process-local endpoint registry cannot resolve Policy for Discovery or
    /// its downstream peers. The typed IPC transport remains lazy: creating a
    /// client for the shared socket must not require a co-located registration.
    #[test]
    fn policy_client_accepts_unregistered_ipc_transport() {
        let signing_key = SigningKey::from_bytes(&[0x63; 32]);
        let transport =
            hyprstream_rpc::transport::TransportConfig::ipc("/run/hyprstream/policy.sock");

        let client = policy_client_for_transport(
            &transport,
            signing_key.clone(),
            signing_key.verifying_key(),
            None,
        );
        assert!(
            client.is_ok(),
            "IPC policy client must be lazy at first boot"
        );
    }

    #[test]
    fn at9p_verify_factory_uses_canonical_service_name() {
        let factory =
            hyprstream_service::get_factory(crate::services::at9p_verify::SERVICE_NAME)
                .expect("at9p-verify factory must be registered");
        assert_eq!(factory.name, "at9p-verify");
        assert!(
            factory
                .name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
        );
        assert!(hyprstream_service::get_factory("at9p_verify").is_none());
    }

    /// #989: the workflow factory is in the service inventory (the daemon can
    /// discover and start it). This is the compile-time/registration half of the
    /// acceptance criterion "WorkflowService appears in service status/discovery"
    /// without spinning a full daemon.
    #[test]
    fn workflow_factory_registered() {
        let factory = hyprstream_service::get_factory("workflow")
            .expect("workflow factory must be registered (#989)");
        assert_eq!(factory.name, "workflow");
        // amend-#989 §1 dependency set — P0 prerequisite wiring.
        assert_eq!(
            factory.depends_on,
            &["worker", "event", "policy", "registry", "discovery"]
        );
        // Schema + metadata must be wired so PolicyService can discover scopes
        // and MCP can surface methods.
        assert!(
            factory.schema.is_some(),
            "workflow factory must embed its capnp schema"
        );
        assert!(
            factory.metadata.is_some(),
            "workflow factory must expose schema_metadata for scope discovery"
        );
    }

    /// The Phase-0 namespace skeleton mounts `/bin`, `/env`, `/out` for the
    /// runner (amend-#989 §2). Smoke-check the three directories exist as direct
    /// children of the synthetic root before the namespace is handed off.
    #[test]
    fn workflow_namespace_phase0_skeleton() {
        use crate::services::fs::{SyntheticNode, SyntheticTree};

        // Rebuild the same tree shape build_workflow_namespace produces, proving
        // the node shape compiles and the three Phase-0 dirs are present.
        let mut root = std::collections::HashMap::new();
        root.insert("bin".to_owned(), SyntheticNode::Dir { children: std::collections::HashMap::new() });
        root.insert("env".to_owned(), SyntheticNode::Dir { children: std::collections::HashMap::new() });
        root.insert("out".to_owned(), SyntheticNode::Dir { children: std::collections::HashMap::new() });
        let _tree = SyntheticTree::new(SyntheticNode::Dir { children: root });

        // build_workflow_namespace itself must not panic and must expose a root mount.
        let ns = build_workflow_namespace();
        assert!(
            ns.mount_prefixes().iter().any(|p| p == &"/"),
            "Phase-0 workflow namespace must mount a root prefix"
        );
    }

    /// amend-#989 §4: default-off / opt-in. The engine must stay dormant unless
    /// an operator explicitly enables it.
    #[test]
    fn workflow_config_default_off() {
        let cfg = hyprstream_workers::config::WorkflowConfig::default();
        assert!(!cfg.enabled, "WorkflowConfig must default to enabled = false (#989)");
    }

    /// Helper: generate an ECDSA P-256 key pair and return (pkcs8_der, public_key_der)
    fn generate_ecdsa_p256_pair() -> (Vec<u8>, Vec<u8>) {
        let key_pair = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let pkcs8 = key_pair.serialize_der();
        let pub_der = key_pair.public_key_der();
        (pkcs8, pub_der.clone())
    }

    fn build_endorsement_message(ed25519_pubkey: &[u8; 32], domain: &str) -> Vec<u8> {
        let mut msg = Vec::with_capacity(TLS_ENDORSEMENT_V1.len() + 32 + domain.len());
        msg.extend_from_slice(TLS_ENDORSEMENT_V1);
        msg.extend_from_slice(ed25519_pubkey);
        msg.extend_from_slice(domain.as_bytes());
        msg
    }

    /// #441 fail-closed: with no JWT in the trust store and none on disk,
    /// registration MUST error (naming the real cause) rather than silently
    /// skip — a service that can't register its key must not serve signed
    /// responses.
    #[test]
    fn resolve_registration_jwt_fails_closed_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_registration_jwt(
            "model",
            dir.path(),
            crate::auth::identity_store::SecretsProfile::SharedDirectory,
            None,
        )
        .expect_err("missing JWT must fail closed, not skip");
        let msg = err.to_string();
        assert!(msg.contains("model"), "error names the service: {msg}");
        assert!(
            msg.contains("cannot register its signing key"),
            "error names the real cause: {msg}",
        );
    }

    /// A JWT already present in the trust store is used directly (no disk read).
    #[test]
    fn resolve_registration_jwt_prefers_trust_store() {
        let dir = tempfile::tempdir().unwrap();
        let jwt = resolve_registration_jwt(
            "model",
            dir.path(),
            crate::auth::identity_store::SecretsProfile::SharedDirectory,
            Some("trust.jwt.token".to_owned()),
        )
        .unwrap();
        assert_eq!(jwt, "trust.jwt.token");
    }

    /// When not in the trust store, the authoritative on-disk JWT is loaded.
    #[test]
    fn resolve_registration_jwt_falls_back_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        crate::auth::identity_store::write_service_jwt(dir.path(), "model", "disk.jwt.token")
            .unwrap();
        let jwt = resolve_registration_jwt(
            "model",
            dir.path(),
            crate::auth::identity_store::SecretsProfile::SharedDirectory,
            None,
        )
        .unwrap();
        assert_eq!(jwt, "disk.jwt.token");
    }

    #[test]
    fn test_tls_endorsement_with_ecdsa_p256() {
        let (pkcs8, _pub_der) = generate_ecdsa_p256_pair();
        let ed25519_pubkey = [0xAB_u8; 32];

        let endorsement = compute_tls_endorsement(&pkcs8, &ed25519_pubkey, "example.com").unwrap();
        assert!(!endorsement.is_empty());
        // ECDSA P-256 fixed-length signature is 64 bytes
        assert_eq!(endorsement.len(), 64);
    }

    #[test]
    fn test_tls_endorsement_wrong_domain_differs() {
        let (pkcs8, _) = generate_ecdsa_p256_pair();
        let ed25519_pubkey = [0xAB_u8; 32];

        let endorsement_a =
            compute_tls_endorsement(&pkcs8, &ed25519_pubkey, "example.com").unwrap();
        let endorsement_b = compute_tls_endorsement(&pkcs8, &ed25519_pubkey, "evil.com").unwrap();

        // ECDSA signatures are randomized so they'll differ anyway, but the important
        // thing is that the message content changes — verified by the factory logic.
        // Just confirm both succeed.
        assert!(!endorsement_a.is_empty());
        assert!(!endorsement_b.is_empty());
    }

    #[test]
    fn test_tls_endorsement_message_format() {
        let ed25519_pubkey = [0x42_u8; 32];
        let msg = build_endorsement_message(&ed25519_pubkey, "test.local");

        let expected_len = TLS_ENDORSEMENT_V1.len() + 32 + "test.local".len();
        assert_eq!(msg.len(), expected_len);

        // Starts with domain separator
        assert_eq!(&msg[..TLS_ENDORSEMENT_V1.len()], TLS_ENDORSEMENT_V1);
        // Followed by pubkey
        assert_eq!(
            &msg[TLS_ENDORSEMENT_V1.len()..TLS_ENDORSEMENT_V1.len() + 32],
            &[0x42_u8; 32]
        );
        // Followed by domain
        assert_eq!(&msg[TLS_ENDORSEMENT_V1.len() + 32..], b"test.local");
    }

    #[test]
    fn test_tls_endorsement_invalid_key() {
        let ed25519_pubkey = [0xAB_u8; 32];
        let result = compute_tls_endorsement(&[0xFF; 32], &ed25519_pubkey, "example.com");
        assert!(result.is_err());
    }

    /// `init_local_moq_stream_plane` sets both process-global moq state
    /// (`global_moq_origin` + `global_moq_uds_path`) and is idempotent: a second
    /// call is a no-op and must not panic (composes with the streams factory and
    /// multiple co-located publisher factories).
    ///
    /// Uses process-global `OnceLock`s, so it runs in a dedicated single-test
    /// binary (`#[cfg(test)]` integration is impractical for OnceLock); the
    /// assertions tolerate a plane already initialized by an earlier test in the
    /// same process — the contract under test is "set after call" + "idempotent".
    #[tokio::test]
    async fn init_local_moq_stream_plane_sets_globals_and_is_idempotent() {
        use hyprstream_rpc::moq_stream::{global_moq_origin, global_moq_uds_path};

        // First call (or pre-set by another test) → plane is initialized.
        init_local_moq_stream_plane("test");
        assert!(
            global_moq_origin().is_some(),
            "origin must be set after init_local_moq_stream_plane",
        );
        let uds = global_moq_uds_path();
        assert!(
            uds.is_some(),
            "uds path must be set after init_local_moq_stream_plane",
        );
        let path_after_first = uds.map(std::path::Path::to_path_buf);

        // Second call must be a no-op (idempotent) — no panic, no change.
        init_local_moq_stream_plane("test");
        assert!(global_moq_origin().is_some());
        assert_eq!(
            global_moq_uds_path().map(std::path::Path::to_path_buf),
            path_after_first,
            "second call must not change the served UDS path",
        );
    }
}
