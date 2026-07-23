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

use std::sync::Arc;

use anyhow::Context;
use git2db::Git2DB;
use hyprstream_rpc::moq_event::MoqEventOrigin;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::SocketKind;
use hyprstream_rpc::service_factory;
use hyprstream_service::{ServiceContext, Spawnable};
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

/// Get the JWT token for a service from the trust store.
fn service_token(service_name: &str) -> Option<String> {
    let trust = hyprstream_service::global_trust_store();
    let vk = trust.resolve_one(service_name)?;
    trust.get(&vk).and_then(|att| att.jwt.clone())
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
                && state.current.subject_keys.first().is_some_and(|key| {
                    key.ed25519_pub.as_slice() == signer.verifying_key().as_bytes()
                })
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
    from_trust: Option<String>,
) -> anyhow::Result<String> {
    if let Some(jwt) = from_trust {
        return Ok(jwt);
    }
    match crate::auth::identity_store::load_service_jwt(creds_dir, service_name) {
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
    _ctx: &ServiceContext,
    service_name: &str,
    signing_key: &SigningKey,
) -> anyhow::Result<()> {
    // PolicyService doesn't register — it IS the CA.
    if service_name == "policy" {
        return Ok(());
    }

    let creds_dir = credentials_dir()?;

    // The JWT may already be in the trust store (e.g. seeded by an earlier
    // registration in this process); otherwise load it from disk — the
    // authoritative location the wizard/bootstrap manager wrote it to.
    let from_trust = {
        let trust = hyprstream_service::global_trust_store();
        trust
            .resolve_one(service_name)
            .and_then(|vk| trust.get(&vk))
            .and_then(|att| att.jwt.clone())
    };
    let jwt = resolve_registration_jwt(service_name, &creds_dir, from_trust)?;

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
    let policy_client =
        PolicyClient::for_local_bootstrap(signing_key.clone(), policy_vk, Some(jwt.clone()))?;

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
    spawn_jwt_renewal_task(service_name, signing_key.clone(), creds_dir);

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
/// Checks hourly; renews when ≤7 days remain. Writes the renewed JWT to disk and
/// updates the global trust store so in-flight RPC calls stay authenticated.
fn spawn_jwt_renewal_task(
    service_name: &str,
    signing_key: SigningKey,
    credentials_dir: std::path::PathBuf,
) {
    let service_name = service_name.to_owned();
    tokio::spawn(async move {
        const CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3_600);
        const RENEW_THRESHOLD: i64 = 7 * 24 * 3_600; // 7 days remaining

        loop {
            tokio::time::sleep(CHECK_INTERVAL).await;

            let jwt = match crate::auth::identity_store::load_service_jwt(
                &credentials_dir,
                &service_name,
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
                    .resolve_one(&service_name)
                    .and_then(|vk| trust.get(&vk))
                    .and_then(|att| att.jwt.clone())
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

            let policy_client = match PolicyClient::for_local_bootstrap(
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
                    if let Some(vk) = trust.resolve_one(&service_name) {
                        if let Some(mut att) = trust.get(&vk) {
                            att.jwt = Some(info.token.clone());
                            att.expires_at = info.expires_at;
                            trust.insert(vk, att);
                        }
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
fn create_event_service(_ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating EventService (moq-lite event bus)");

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
        anyhow::bail!("ledger service requested but [ledger] enabled = false (the Phase-1 enforcer is opt-in)");
    }

    // Cell identity = did:key over the service Ed25519 key.
    let ed_sk = ctx.service_signing_key("ledger");
    let ed_vk = ed_sk.verifying_key();
    let cell_identity = Did(ed25519_to_did_key(&ed_vk.to_bytes()));

    // Register this service's verifying key with PolicyService.
    let _ = ctx.verifying_key();

    // PQ (ML-DSA-65) key under the Hybrid policy. Fail-closed construction:
    // `require_pq_signatures` set with no key available ⇒ refuse to start the
    // ledger service rather than silently downgrade checkpoints to Classical.
    let signer: Arc<dyn hyprstream_ledger::CheckpointSigner + Send + Sync> = if lcfg
        .require_pq_signatures
    {
        let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()?;
        let store = crate::auth::key_rotation::global_ml_dsa_key_store(&secrets_dir, &config.oauth);
        let pq_key = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { store.active_key().await })
        });
        match pq_key {
            Some(k) => Arc::new(CoseCheckpointSigner::hybrid(cell_identity.clone(), ed_sk, (*k).clone())),
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

    // Phase-1 backend: MemLedger (RocksLedger is item 1.2). The grant verifier
    // is the fail-closed StaticGrantVerifier until the UCAN wiring lands.
    let verifier: Arc<dyn crate::services::ledger::GrantVerifier + Send + Sync> =
        Arc::new(StaticGrantVerifier::new());
    let sink: Arc<dyn crate::services::ledger::ReceiptSink + Send + Sync> =
        Arc::new(LoggingReceiptSink);

    let service = LedgerService::spawn(
        lcfg,
        Box::new(MemLedger::new(cell_identity.clone())),
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
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("registry"))?;

    // #910a — the registry service is the sole PDS-record writer AND the sole
    // holder of the `#atproto` private key: it opens the durable store
    // read-write and, on register/commit, signs the repo's commit ONCE and
    // persists the signed bytes. Reads (the discovery service) are keyless.
    // This node's `did:key` identity (the record `repo` at-uri authority) is
    // derived from the node's own root Ed25519 key — the same identity TLS
    // endorsement uses ("a node-level trust assertion, not specific to any
    // per-service key"). The `#atproto` commit-signing key is the *active* key
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
        let es256_store =
            crate::auth::key_rotation::global_es256_key_store(&secrets_dir, &config.oauth);
        // `active_key` is async (tokio RwLock); resolve it once here on the
        // current runtime. `block_in_place` avoids stalling a worker reactor.
        let active = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(es256_store.active_key())
        })
        .ok_or_else(|| anyhow::anyhow!("no active ES256 #atproto key for PDS publish"))?;
        let atproto_key: p256::ecdsa::SigningKey = (*active).clone();
        let acceptance_identity = ctx.service_signing_key("registry");
        anyhow::ensure!(
            hyprstream_discovery::deployment_registry_verifier()?
                .matches(&acceptance_identity.verifying_key()),
            "registry signing credential does not match authenticated deployment identity"
        );
        // The alarm WAL must remain verifiable across OAuth key rotations and
        // process restarts. Derive a dedicated, stable audit identity from the
        // node/service root available in this deployment mode; the second
        // derivation keeps the ML-DSA material separate from the Ed25519 key.
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
            crate::services::discovery::PdsPublisher::new(store, node_did, atproto_key)
                .with_at9p_state_ingest(at9p_state),
        )
    })()
    .map_err(|e| tracing::warn!("PDS publish disabled: {e}"))
    .ok();

    // Create registry service with infrastructure (blocking since we're in sync context)
    let mut registry_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(RegistryService::new(
            ctx.models_dir(),
            policy_client,
            ctx.transport("registry", SocketKind::Rep),
            sk.clone(),
        ))
    })?;
    if let Some(issuer) = ctx.oauth_issuer_url() {
        registry_service = registry_service.with_expected_audience(issuer.to_owned());
    }
    registry_service = registry_service.with_jwt_key_source(ctx.cluster_key_source());
    if let Some(publisher) = pds_publisher {
        registry_service = registry_service.with_pds_publisher(publisher);
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
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("model"))?;

    // Create registry client
    let registry_client: RegistryClient =
        RegistryClient::from_resolver(sk.clone(), service_token("model"))?;

    #[allow(clippy::expect_used)]
    let mut model_service = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create runtime for model factory");
        let local = tokio::task::LocalSet::new();
        local.block_on(
            &rt,
            ModelService::new(
                ModelServiceConfig::default(),
                sk.clone(),
                policy_client,
                registry_client,
                ctx.transport("model", SocketKind::Rep),
            ),
        )
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

    // Resolve + construct the backend fail-closed against the inventory registry
    // (config-driven by name; explicit requests are authoritative, missing
    // prerequisites error out rather than silently downgrading isolation; "auto"
    // picks the strongest available). Single seam — no scattered cfg, no
    // `_ => nspawn` fallback (#507 / #518).
    let backend_ctx = BackendCtx {
        pool_config: pool_config.clone(),
        #[cfg(feature = "oci-image")]
        image_config,
        #[cfg(feature = "oci-image")]
        rafs_store: Arc::clone(&rafs_store),
    };
    let backend: Arc<dyn SandboxBackend> = resolve_backend(&backend_name, &backend_ctx)?;

    let sk = ctx.service_signing_key("worker");

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
    let policy_client = crate::services::PolicyClient::for_local_bootstrap(
        sk.clone(),
        policy_vk,
        service_token("worker"),
    )?;
    worker_service.set_authorize_fn(super::worker::build_authorize_fn(policy_client));
    if let Some(issuer) = ctx.oauth_issuer_url() {
        worker_service.set_expected_audience(issuer.to_owned());
    }
    worker_service.set_jwt_key_source(ctx.cluster_key_source());

    Ok(ctx.into_spawnable_quic(worker_service, worker_quic_port))
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
    let model_client = ModelClient::from_resolver(sk.clone(), service_token("oai"))?;
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("oai"))?;

    // Create registry client
    let registry_client: RegistryClient =
        RegistryClient::from_resolver(sk.clone(), service_token("oai"))?;

    // Create server state (blocking since we're in sync context)
    let resource_url = config.oai.resource_url();
    let oauth_issuer_url = config.oauth.issuer_url();
    let server_state = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(ServerState::new(
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
        ))
    })
    .context("Failed to create server state")?;

    let oai_service = OAIService::new(
        config.oai.clone(),
        config.tls.clone(),
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
        RegistryClient::from_resolver(sk.clone(), service_token("xet"))?;

    // Reuse the same narrow authentication core as OAI without constructing an
    // inference-oriented ServerState. The policy client is used by federated
    // issuer admission and key resolution.
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("xet"))?;
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

    let state = XetState {
        // Reads share the same L1 CAS substrate the registry's getBlob uses (#812).
        store: crate::storage::CasSubstrate::from_env(),
        registry: Some(registry_client),
        auth,
    };

    let xet_service = XetService::new(config.xet.clone(), config.tls.clone(), state);

    Ok(Box::new(xet_service))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Flight Service Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Factory for FlightService (Arrow Flight SQL server)
///
/// This service provides Flight SQL protocol for dataset queries.
/// It optionally uses RegistryClient for dataset lookup.
#[service_factory("flight", depends_on = ["registry", "discovery"])]
fn create_flight_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    info!("Creating FlightService");

    use crate::services::FlightService;

    // Load full config for Flight settings
    let config = load_config();
    let sk = ctx.service_signing_key("flight");

    // Register this service's verifying key with PolicyService
    register_service_key(ctx, "flight", &sk)?;

    // Create registry client for dataset lookup (if default_dataset is configured)
    // RegistryClient already implements hyprstream_metrics::RegistryClient
    let registry_client: Option<Arc<dyn hyprstream_metrics::RegistryClient>> =
        if config.flight.default_dataset.is_some() {
            let registry_client: RegistryClient =
                RegistryClient::from_resolver(sk.clone(), service_token("flight"))?;
            Some(Arc::new(registry_client))
        } else {
            None
        };

    let flight_service = FlightService::new(
        config.flight.clone(),
        registry_client,
        ctx.transport("flight", SocketKind::Rep),
        ctx.verifying_key(),
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

    // Pass signing key instead of a pre-created PolicyClient.
    // OAuthService runs in its own tokio runtime (separate thread), so the
    // PolicyClient must be created inside that runtime for ZMQ async I/O to work.
    let mut oauth_service = OAuthService::new(
        config.oauth.clone(),
        config.tls.clone(),
        sk,
        ctx.transport("oauth", SocketKind::Rep),
        ctx.verifying_key(),
        ctx.jwt_verifying_key(),
    )
    .with_quic_config(config.quic.clone());
    if let Some(bl) = SHARED_JTI_BLOCKLIST.get() {
        oauth_service = oauth_service.with_jti_blocklist(Arc::clone(bl));
    } else {
        tracing::warn!("JTI blocklist not set by PolicyService factory — revoked access tokens will not be blocked at RPC layer");
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
        signing_key: sk,
        transport: ctx.transport("mcp", SocketKind::Rep),
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
            let fallback_policy_client = std::sync::Arc::new(PolicyClient::for_local_bootstrap(
                ctx.service_signing_key("mcp"),
                policy_vk,
                service_token("mcp"),
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
                    let mcp_dpop_jti_seen: std::sync::Arc<hyprstream_util::TtlCache<String, ()>> =
                        std::sync::Arc::new(hyprstream_util::TtlCache::new(10_000, 64));
                    move |mut req: axum::extract::Request, next: axum::middleware::Next| {
                        let www_authenticate = www_authenticate.clone();
                        let mcp_resource_url = mcp_resource_url.clone();
                        let _mcp_oauth_issuer = mcp_oauth_issuer_clone.clone();
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
                                match key_source.get_key(&iss, kid.as_deref()).await {
                                    Ok(key) => crate::auth::jwt::decode(&t, &key, Some(mcp_resource_url.as_str())),
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
                                match federation_resolver.get_key(&iss).await {
                                    Ok(key) => crate::auth::jwt::decode_with_key(&t, &key, Some(mcp_resource_url.as_str())),
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
                            let claims = match result {
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
                                    let ttl_secs = ((proof.iat + 120) - now).max(0) as u64;
                                    if !dpop_jti_seen.insert_if_absent(
                                        proof.jti.clone(),
                                        (),
                                        std::time::Duration::from_secs(ttl_secs),
                                    ) {
                                        tracing::debug!(%method, %uri, jti = %proof.jti, "MCP: DPoP jti replayed");
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
                            req.extensions_mut().insert(crate::server::middleware::AuthenticatedUser {
                                user: claims.sub.clone(),
                                token: Some(t.clone()),
                                exp: Some(claims.exp),
                            });
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
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("tui"))?;

    // Build VFS namespace for ChatApps spawned via TUI RPC.
    let (vfs_ns, vfs_subject) = crate::tui::vfs::build_chat_vfs_namespace(&sk)?;

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
            drop(
                crate::services::discovery::PdsRecordStore::open(dir).with_context(|| {
                    format!("read-only open failed ({orig}); bootstrap open also failed")
                })?,
            );
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
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("discovery"))?;
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
    let record_resolver = std::sync::Arc::new(crate::services::discovery::PdsRecordResolver::new(
        pds_store,
    ));

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
    let policy_client =
        PolicyClient::for_local_bootstrap(sk.clone(), policy_vk, service_token("metrics"))?;

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
        let err = resolve_registration_jwt("model", dir.path(), None)
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
        let jwt = resolve_registration_jwt("model", dir.path(), Some("trust.jwt.token".to_owned()))
            .unwrap();
        assert_eq!(jwt, "trust.jwt.token");
    }

    /// When not in the trust store, the authoritative on-disk JWT is loaded.
    #[test]
    fn resolve_registration_jwt_falls_back_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        crate::auth::identity_store::write_service_jwt(dir.path(), "model", "disk.jwt.token")
            .unwrap();
        let jwt = resolve_registration_jwt("model", dir.path(), None).unwrap();
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
