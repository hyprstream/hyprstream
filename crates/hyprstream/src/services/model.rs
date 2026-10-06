//! Model service for managing InferenceService instances over ZMQ
//!
//! This service manages the lifecycle of InferenceService instances.
//! It handles model loading, unloading, and routes inference requests
//! to the appropriate InferenceService based on model reference.
//!
//! # Architecture
//!
//! ```text
//! REST API / CLI
//!       │
//!       │ ModelClient (async ZMQ I/O)
//!       ▼
//! ModelService (multi-threaded runtime)
//!       │
//!       ├── LRU cache of loaded models
//!       ├── Spawns InferenceService per model
//!       └── Routes requests to InferenceService
//!             │
//!             │ InferenceClient (async ZMQ I/O)
//!             ▼
//!       InferenceService (dedicated thread per model)
//! ```
//!
//! # Endpoint
//!
//! Uses `registry().endpoint("model", SocketKind::Rep)` for the REP endpoint.
//! Default fallback: `inproc://hyprstream/model`

use async_trait::async_trait;
// GenerationRequest import removed — was only used by deleted ModelZmqClient
use hyprstream_rpc_std::model_client::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::runtime::inference_profile::{
    InferenceCompute, InferenceDeploymentProfile, InferenceInstanceId,
    InferenceIsolationProfile,
};
use crate::services::{
    EnvelopeContext,
    PolicyClient,
};
use hyprstream_rpc_std::inference_client::InferenceClient;
use hyprstream_rpc_std::registry_client::RegistryClient;
use hyprstream_rpc_std::registry_client::{StageFilesRequest, CommitWithAuthorRequest};
use crate::services::WorktreeClientExt;
use hyprstream_rpc_std::policy_client::PolicyCheck;
use crate::storage::ModelRef;
use anyhow::{anyhow, Context, Result};
use hyprstream_rpc::latch::{Terminal, TerminalStore};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::events::EventPublisher;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::{EndpointType, TransportConfig};
use hyprstream_rpc::stream_info::TransportConfig as WireTransportConfig;
use lru::LruCache;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};
#[cfg(test)]
use zeroize::Zeroizing;

/// Default endpoint for the model service
pub const MODEL_ENDPOINT: &str = "inproc://hyprstream/model";

// ============================================================================
// Latched load terminal (EV7/#649) — host-side retained load outcome
// ============================================================================

/// Terminal payload for a model load attempt (EV7/#649). Latched host-side by
/// [`ModelService`] when a load completes — [`Loaded`](Self::Loaded) on success,
/// [`LoadFailed`](Self::LoadFailed) otherwise — so a late `load --wait` (or a
/// future P9 `/model/<ref>/loaded` file) is served the retained value instead
/// of missing the live `model.lifecycle` edge it subscribed to too late.
///
/// Rides the same plaintext `model` EventService source as the live edge (EV5
/// /#605; encrypted-default flip deferred to #555), so retaining it host-side
/// releases no plaintext the live edge would not.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelLoadTerminal {
    /// Load succeeded; `endpoint` is the inference endpoint string.
    Loaded { endpoint: String },
    /// Load failed; `error` is the failure message.
    LoadFailed { error: String },
}

/// Per-load-attempt key into the model-service [`TerminalStore`] (EV7/#649).
///
/// Combines the model ref with a monotonic per-process epoch (see
/// [`ModelServiceInner::load_epoch`]) so a reload allocates a fresh key and
/// latches a NEW terminal — monotonic write-once then holds *per epoch*, the
/// EV7 "reload = new latch, write-once within an attempt" contract. Without
/// per-epoch keying the first `Loaded` latch would shadow every reload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelLoadKey {
    /// Authority-verified tenant that owns this load attempt.
    pub tenant: String,
    /// The model reference string (e.g. "qwen3-small:main").
    pub model_ref: String,
    /// Monotonic load-attempt number for this process.
    pub epoch: u64,
}

// ============================================================================
// ModelService (server-side)
// ============================================================================


/// Information about a loaded model
pub struct LoadedModel {
    /// Tenant/model/replica identity of this inference instance.
    pub instance: InferenceInstanceId,
    /// Model reference string (e.g., "qwen3-small:main")
    pub model_ref: String,
    /// Exact Git commit whose owned base inputs were used, if pinned.
    pub pinned_commit: Option<git2::Oid>,
    /// Local transport for this model's InferenceService (#320). This remains
    /// the `Inproc` arm registered by the spawner and is never advertised as a
    /// remotely dialable reach.
    pub transport: TransportConfig,
    /// Advertised network reach for remote callers. Local dispatch continues
    /// to use `transport` as the in-process fast path.
    pub network_transport: TransportConfig,
    /// Handle to stop the InferenceService
    pub service_handle: hyprstream_service::SpawnedService,
    /// Client for communicating with the InferenceService (built from
    /// `transport` via `dial()` — the co-located fast path).
    pub client: InferenceClient,
    /// Worker-generated 256-bit incarnation of THIS running pinned worker,
    /// reported through the private readiness handoff for this spawn attempt.
    /// Fresh per run: pre-restart work orders fail audience verification at
    /// the replacement worker (Sol bounded incarnation plan).
    pub incarnation: String,
    /// Exact versioned internal-work audience of this incarnation
    /// (`iw1/{deterministic service name}/{incarnation}`). Minting uses THIS
    /// binding — never a recomputed deterministic name.
    pub work_audience: String,
    /// Spawn-attempt generation this binding came from. A delayed result from
    /// an older attempt cannot overwrite a newer binding (generation checked
    /// at handoff validation).
    pub load_generation: u64,
    /// #322 leaf cell-router state for this model. Holds the session→owner
    /// affinity map (heartbeat-lease renewal, KV-cache stickiness) and the
    /// per-node load/health counters used by HRW placement. In v1 this is a
    /// single co-located node (the router fast-paths to `client`); the
    /// `load_state` set grows to multiple nodes when cross-host replicas are
    /// resolved via the Resolver.
    pub router: crate::services::router::CellRouter,
    /// Replica-set snapshot for the router (one entry per known inference
    /// server serving this model's OID). Updated as the Resolver yields new
    /// reaches; consumed by HRW placement.
    pub load_state: Vec<crate::services::router::InferenceServerInfo>,
    /// When the model was loaded
    pub loaded_at: Instant,
    /// When the model was last used
    pub last_used: Instant,
    /// Online training (TTT) configuration (if enabled)
    pub ttt_config: Option<crate::training::ttt::TTTConfig>,
    /// Generation parameter defaults from model's generation_config.json
    pub generation_defaults: crate::config::SamplingParams,
}

/// Model service configuration
pub struct ModelServiceConfig {
    /// Maximum number of models to keep loaded
    pub max_models: usize,
    /// Maximum context length for KV cache allocation
    pub max_context: Option<u32>,
    /// KV cache quantization type
    pub kv_quant: KVQuantType,
    /// Isolation, tenancy, compute, and resource contract for spawned engines.
    pub inference_deployment: InferenceDeploymentProfile,
    /// Optional exact-ref admission for the bounded staging deployment.
    pub staging_model_pin: Option<StagingModelPin>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagingModelPin {
    pub model_ref: String,
    pub commit: git2::Oid,
}

impl StagingModelPin {
    pub fn new(model_ref: impl Into<String>, commit: git2::Oid) -> Result<Self> {
        let model_ref = model_ref.into();
        let parsed = ModelRef::parse(&model_ref)
            .with_context(|| format!("invalid bounded staging modelRef {model_ref:?}"))?;
        anyhow::ensure!(
            matches!(parsed.git_ref, crate::storage::GitRef::Branch(_))
                && parsed.to_string() == model_ref,
            "bounded staging modelRef must be a canonical local branch ref"
        );
        Ok(Self { model_ref, commit })
    }

    /// `staging` selects the code-reviewed A0 identity. Optional explicit values
    /// must match it; incomplete/invalid profiles fail closed at startup.
    pub fn from_env() -> Result<Option<Self>> {
        const PROFILE: &str = "HYPRSTREAM_MODEL_ADMISSION_PROFILE";
        const REF: &str = "HYPRSTREAM_STAGING_MODEL_REF";
        const OID: &str = "HYPRSTREAM_STAGING_MODEL_OID";
        let profile = match std::env::var(PROFILE) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "default".to_owned(),
            Err(error) => return Err(error.into()),
        };
        let read_optional = |name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(anyhow!("read {name}: {error}")),
        };
        let model_ref = read_optional(REF)?;
        let oid = read_optional(OID)?;
        Self::from_settings(&profile, model_ref.as_deref(), oid.as_deref())
    }

    fn from_settings(profile: &str, model_ref: Option<&str>, oid: Option<&str>) -> Result<Option<Self>> {
        const PROFILE: &str = "HYPRSTREAM_MODEL_ADMISSION_PROFILE";
        const REF: &str = "HYPRSTREAM_STAGING_MODEL_REF";
        const OID: &str = "HYPRSTREAM_STAGING_MODEL_OID";
        const REVIEWED_REF: &str = "qwen2.5-0.5b-instruct:main";
        const REVIEWED_OID: &str = "18c562db6830c2ef6636b8eaf42fb479272052f7";
        if profile == "staging" {
            match (model_ref, oid) {
                (None, None) => return Self::new(REVIEWED_REF, git2::Oid::from_str(REVIEWED_OID)?).map(Some),
                (Some(model_ref), Some(oid)) => anyhow::ensure!(model_ref == REVIEWED_REF && oid == REVIEWED_OID,
                    "staging override must match the complete reviewed model pin"),
                _ => anyhow::bail!("{REF} and {OID} must either both be absent or both match the reviewed pin"),
            }
            return Self::new(REVIEWED_REF, git2::Oid::from_str(REVIEWED_OID)?).map(Some);
        }
        anyhow::ensure!(profile == "default", "{PROFILE} must be 'default' or 'staging'");
        match (model_ref, oid) {
            (None, None) => Ok(None),
            (Some(model_ref), Some(oid)) => {
                anyhow::ensure!(oid.len() == 40 && oid.bytes().all(|byte| byte.is_ascii_hexdigit()),
                    "{OID} must be a full 40-character hexadecimal Git OID");
                Self::new(model_ref, git2::Oid::from_str(oid)?).map(Some)
            }
            _ => anyhow::bail!("{REF} and {OID} must be configured together"),
        }
    }

    fn admits(&self, model_ref: &str) -> bool { self.model_ref == model_ref }
}

impl Default for ModelServiceConfig {
    fn default() -> Self {
        Self {
            max_models: 5,
            max_context: None,
            kv_quant: KVQuantType::None,
            inference_deployment: InferenceDeploymentProfile::default(),
            staging_model_pin: None,
        }
    }
}

/// Inner state for ModelService, behind Arc for continuation capture.
pub struct ModelServiceInner {
    // Business logic
    /// LRU cache of loaded models
    loaded_models: RwLock<LruCache<InferenceInstanceId, LoadedModel>>,
    /// Models currently being loaded (accepted but not yet in LRU cache)
    pending_loads: parking_lot::Mutex<HashMap<InferenceInstanceId, u64>>,
    /// Models whose cache entry has been removed and whose worker teardown is
    /// still in progress. This survives cancellation of the caller awaiting
    /// `unload_model`, so a replacement load cannot race a draining worker.
    unloading_models: Mutex<HashSet<InferenceInstanceId>>,
    /// Serializes admission of a new load with the transition that reserves an
    /// existing worker for teardown. The guard covers the unloading check,
    /// cache observation, and pending-load insertion as one decision.
    load_unload_gate: Mutex<()>,
    /// Serializes lifecycle event delivery. It is deliberately separate from
    /// lifecycle admission: a successful unload acquires this order before
    /// releasing its slot, preventing a later loaded event from overtaking the
    /// listener-visible unloaded completion without holding admission locks
    /// across network/event delivery.
    lifecycle_event_publish_gate: Mutex<()>,
    /// Service configuration
    config: ModelServiceConfig,
    /// Ed25519 signing key for creating InferenceClients
    signing_key: SigningKey,
    /// Policy client for authorization checks in InferenceService
    policy_client: PolicyClient,
    /// No runtime installer in this slice; reserved Federate JWTs still deny.
    #[cfg(feature = "postgres")]
    federate_dispatch: Option<Arc<crate::services::oauth::federate_proof::DispatchAdapter>>,
    /// Event publisher for model lifecycle events (replaces NotificationService, EV5/#605).
    /// Plaintext (Public) path; the encrypted-default flip is deferred to #555.
    event_publisher: EventPublisher,
    /// Registry client for resolving model paths
    registry: RegistryClient,
    // Infrastructure (for Spawnable)
    transport: TransportConfig,
    policy_transport: TransportConfig,
    /// Expected JWT audience for token validation (RFC 8707).
    expected_audience: Option<String>,
    /// Unified JWT key source for verifying JWTs (local and federated).
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    /// Discovery client for federated record resolution (#431). None = no
    /// federation; `resolve_model_ref`'s at:// branch then falls through to
    /// local resolution.
    discovery_client: Option<Arc<hyprstream_rpc_std::discovery_client::DiscoveryClient>>,
    /// Persistent 9P synthetic trees, partitioned by authority-verified tenant.
    fs_trees: dashmap::DashMap<String, Arc<crate::services::fs::SyntheticTree>>,
    /// Retained terminal for each model load attempt (EV7/#649) — the
    /// host-side "file holds the truth" retain. A late `load --wait` reads the
    /// retained [`ModelLoadTerminal`] from here instead of missing the live
    /// `model.lifecycle` edge. Distinct from the live EventService edge (which
    /// never holds the retained value) and from the `loaded_models` cache
    /// (which tracks residency, not the per-attempt terminal outcome).
    load_terminals: TerminalStore<ModelLoadKey, ModelLoadTerminal>,
    /// Monotonic per-process epoch counter; each genuine load attempt
    /// allocates the next value, keying a fresh [`TerminalStore`] entry so a
    /// reload latches under a new key (reload ⇒ new terminal, EV7/#649).
    load_epoch: AtomicU64,
    /// Monotonic spawn-attempt generation for pinned worker incarnation
    /// handoffs. Each `InferenceService` spawn attempt allocates the next
    /// value; the worker echoes it in its private readiness handoff and a
    /// delayed/stale handoff from an older attempt is discarded.
    incarnation_generation: AtomicU64,
    /// The sole tenant admitted by an in-process deployment. A second tenant
    /// must never share the FFI engine fault radius.
    in_process_tenant: std::sync::OnceLock<String>,
    /// QUIC/MoQ reach populated by the outer service spawner after bind.
    ///
    /// Dynamically loaded inference engines share this handle so their stream
    /// tokens advertise the ModelService's browser-addressable `/moq` relay.
    producer_reach_config: hyprstream_rpc::moq_stream::ProducerReachConfigHandle,
    /// Service-scoped MoQ origin populated by the outer service spawner.
    moq_origin: hyprstream_rpc::moq_stream::MoqStreamOriginHandle,
    /// Test-only, in-memory capture of the most recently minted direct work
    /// order. This deliberately does not exist in production builds: a work
    /// order must never become an observable RPC value, log field, file, or
    /// environment value merely to support restart acceptance coverage.
    #[cfg(test)]
    test_work_order_capture: Mutex<Option<Zeroizing<String>>>,
    /// Test-only ordering witness: successful unload clears admission before
    /// making its `model.unloaded` event visible.
    #[cfg(test)]
    before_unloaded_event: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

/// Model service that manages InferenceService lifecycle.
///
/// Wraps `ModelServiceInner` in `Arc` so continuations can capture a cheap
/// clone. All field access is transparent via `Deref`.
///
/// Load requests are handled asynchronously: the request loop returns an
/// immediate "accepted" response and spawns the actual model loading as a
/// `Continuation` (via `spawn_local`), keeping the service responsive for
/// list, health, info, and other requests during long GPU weight transfers.
pub struct ModelService {
    inner: Arc<ModelServiceInner>,
}

struct SpawnPublicationGuard(Option<hyprstream_service::SpawnedService>);

impl SpawnPublicationGuard {
    fn new(handle: hyprstream_service::SpawnedService) -> Self { Self(Some(handle)) }
    fn publish(mut self) -> Result<hyprstream_service::SpawnedService> {
        self.0.take().ok_or_else(|| anyhow!("spawn publication guard was already disarmed"))
    }
}

impl Drop for SpawnPublicationGuard {
    fn drop(&mut self) {
        if let Some(mut handle) = self.0.take() {
            match tokio::runtime::Handle::try_current() {
                Ok(runtime) => { runtime.spawn(async move {
                    if let Err(error) = handle.stop().await {
                        tracing::error!(%error, "failed to reclaim unpublished inference worker");
                    }
                }); }
                Err(error) => tracing::error!(%error, "cannot reclaim unpublished inference worker outside Tokio runtime"),
            }
        }
    }
}

/// Result of the gated admission used by the non-blocking load RPC path.
///
/// Only `Accepted` owns a continuation and a pending reservation. An
/// already-loaded or already-pending request is a read/deduplication fast path
/// and must not create a second load attempt.
enum InterceptedLoadAdmission {
    Loaded { reach: Vec<WireTransportConfig> },
    Pending,
    Accepted(InterceptedLoadReservation),
}

/// Owns an intercepted load's pending marker from admission through either
/// continuation completion or cancellation. The marker uses a tiny synchronous
/// mutex deliberately: `Drop` must be able to roll it back if serialization or
/// the continuation handoff is cancelled before the future is ever polled.
struct InterceptedLoadReservation {
    inner: Arc<ModelServiceInner>,
    instance: InferenceInstanceId,
    attempt: u64,
    active: bool,
}

impl InterceptedLoadReservation {
    fn new(inner: Arc<ModelServiceInner>, instance: InferenceInstanceId, attempt: u64) -> Self {
        Self { inner, instance, attempt, active: true }
    }

    fn release(&mut self) {
        if self.active {
            let mut pending = self.inner.pending_loads.lock();
            if pending.get(&self.instance) == Some(&self.attempt) {
                pending.remove(&self.instance);
            }
            self.active = false;
        }
    }
}

impl Drop for InterceptedLoadReservation {
    fn drop(&mut self) {
        self.release();
    }
}

fn admit_in_process_tenant(
    admitted_tenant: &std::sync::OnceLock<String>,
    verified_tenant: &str,
) -> Result<()> {
    if let Some(admitted) = admitted_tenant.get() {
        anyhow::ensure!(
            admitted == verified_tenant,
            "in-process inference is bound to tenant {admitted:?}; refusing cross-tenant engine sharing"
        );
        return Ok(());
    }

    if admitted_tenant.set(verified_tenant.to_owned()).is_err() {
        let admitted = admitted_tenant
            .get()
            .ok_or_else(|| anyhow!("in-process tenant admission raced without a winner"))?;
        anyhow::ensure!(
            admitted == verified_tenant,
            "in-process inference is bound to tenant {admitted:?}; refusing cross-tenant engine sharing"
        );
    }
    Ok(())
}

impl Clone for ModelService {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl std::ops::Deref for ModelService {
    type Target = ModelServiceInner;
    fn deref(&self) -> &Self::Target { &self.inner }
}

/// Prefix-dispatch arm for the `modelRef` grammar (#395).
///
/// `modelRef :Text` stays `Text` in the capnp schema; the Rust-side grammar is:
///
/// ```text
/// modelRef ::= "at://" <at-uri>   # federated (resolve via atproto NAME → git OID)
///            | "did:" <did>       # federated, bare-DID form
///            | <name> [":" <gitref>]  # local ModelRef (unchanged, backward-compatible)
/// ```
///
/// `at://` and `did:` are federated; everything else is the legacy local
/// [`ModelRef::parse`] path. This enum is split out from
/// [`ModelService::resolve_model_ref`] so the grammar is unit-testable without
/// constructing a full [`ModelService`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelRefDispatch<'a> {
    /// `at://…` or `did:…` — federated resolution (atproto record store, #392).
    Federated {
        /// The matched scheme prefix (`"at://"` or `"did:"`).
        scheme: &'static str,
        /// The ModelRef string with the scheme prefix stripped.
        rest: &'a str,
    },
    /// No federated prefix — fall through to the local [`ModelRef::parse`].
    Local,
}

/// Classify a `modelRef` string into its grammar arm (#395 prefix dispatch).
///
/// This is the pure prefix-detection core of [`ModelService::resolve_model_ref`];
/// it performs no I/O and no validation, so it can be unit-tested in isolation.
pub fn model_ref_dispatch(s: &str) -> ModelRefDispatch<'_> {
    if let Some(rest) = s.strip_prefix("at://") {
        ModelRefDispatch::Federated { scheme: "at://", rest }
    } else if let Some(rest) = s.strip_prefix("did:") {
        ModelRefDispatch::Federated { scheme: "did:", rest }
    } else {
        ModelRefDispatch::Local
    }
}

/// Outcome of [`ModelService::route_decision`] — which arm of the replica
/// set HRW/least-loaded selected, kept separate from `InferenceClient`
/// construction so the pure ranking decision is unit-testable without a
/// live [`LoadedModel`] (no `InferenceClient`/`SpawnedService` required).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteDecision {
    /// The router selected the co-located replica (`load_state`'s
    /// co-located entry) — the in-process fast path, no dial, no mesh
    /// policy fires (same trust domain).
    CoLocated,
    /// The router selected a DIFFERENT (non-co-located) replica from the
    /// resolved candidate set. #282 has not wired a cross-host dial yet,
    /// so callers still fall back to the co-located client. Placement metadata
    /// supplies neither network reach nor authority.
    Remote(crate::services::router::ReplicaId),
    /// No healthy candidate at all (every entry excluded as down/stalled,
    /// or the replica set is empty).
    NoHealthyCandidate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplicaLocality {
    CoLocated,
    Remote,
}

struct RoutedReplica<T> {
    value: T,
    replica_id: crate::services::router::ReplicaId,
    locality: ReplicaLocality,
}

/// Apply affinity-preserving selection and bounded dial-failure reselection.
///
/// A remote failure is marked down before the next placement. The co-located
/// value is returned only when the router explicitly selects that replica; an
/// exhausted/empty set is an error, never an implicit local fallback.
async fn route_and_dial_replica<T, F, Fut>(
    router: &mut crate::services::router::CellRouter,
    load_state: &[crate::services::router::InferenceServerInfo],
    co_located: crate::services::router::ReplicaId,
    session_id: &str,
    local_value: T,
    mut dial_remote: F,
) -> Result<RoutedReplica<T>>
where
    F: FnMut(crate::services::router::InferenceServerInfo) -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut local_value = Some(local_value);
    let mut failures = Vec::new();
    for _ in 0..load_state.len() {
        let now = Instant::now();
        match ModelService::route_decision(router, load_state, co_located, session_id, now) {
            RouteDecision::CoLocated => {
                let value = local_value
                    .take()
                    .ok_or_else(|| anyhow!("co-located inference client was already consumed"))?;
                return Ok(RoutedReplica {
                    value,
                    replica_id: co_located,
                    locality: ReplicaLocality::CoLocated,
                });
            }
            RouteDecision::Remote(replica_id) => {
                let candidate = load_state
                    .iter()
                    .find(|candidate| candidate.replica_id == replica_id)
                    .cloned()
                    .ok_or_else(|| anyhow!("router selected an absent replica"))?;
                match dial_remote(candidate).await {
                    Ok(value) => {
                        return Ok(RoutedReplica {
                            value,
                            replica_id,
                            locality: ReplicaLocality::Remote,
                        });
                    }
                    Err(error) => {
                        router.report_dial_fail(replica_id, now);
                        failures.push(format!("{}: {error}", hex::encode(replica_id.as_bytes())));
                    }
                }
            }
            RouteDecision::NoHealthyCandidate => break,
        }
    }
    if failures.is_empty() {
        anyhow::bail!("no healthy inference replica candidates");
    }
    anyhow::bail!(
        "inference replica candidates exhausted after dial/authorization failures: {}",
        failures.join("; ")
    )
}

impl ModelService {
    /// Create a new model service with infrastructure
    pub async fn new(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: RegistryClient,
        transport: TransportConfig,
        policy_transport: TransportConfig,
    ) -> Result<Self> {
        config.inference_deployment.validate()?;
        // SAFETY: 5 is a valid non-zero value
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        // EV5/#605: model lifecycle events now ride the unified EventService
        // (plaintext Public path); the parallel NotificationService is removed.
        let event_publisher = EventPublisher::new("model")?;

        Ok(Self { inner: Arc::new(ModelServiceInner {
            loaded_models: RwLock::new(LruCache::new(cache_size)),
            pending_loads: parking_lot::Mutex::new(HashMap::new()),
            unloading_models: Mutex::new(HashSet::new()),
            load_unload_gate: Mutex::new(()),
            lifecycle_event_publish_gate: Mutex::new(()),
            config,
            signing_key,
            policy_client,
            #[cfg(feature = "postgres")]
            federate_dispatch: None,
            event_publisher,
            registry,
            transport,
            policy_transport,
            expected_audience: None,
            jwt_key_source: None,
            discovery_client: None,
            fs_trees: dashmap::DashMap::new(),
            load_terminals: TerminalStore::new(),
            load_epoch: AtomicU64::new(0),
            incarnation_generation: AtomicU64::new(0),
            in_process_tenant: std::sync::OnceLock::new(),
            producer_reach_config: std::sync::Arc::new(parking_lot::RwLock::new(
                hyprstream_rpc::moq_stream::ProducerReachConfig::default(),
            )),
            moq_origin: std::sync::Arc::new(parking_lot::RwLock::new(None)),
            #[cfg(test)]
            test_work_order_capture: Mutex::new(None),
            #[cfg(test)]
            before_unloaded_event: Mutex::new(None),
        })})
    }

    /// Retain a direct work order only for an in-crate test. The production
    /// Model→Inference boundary still attaches the bearer directly to the
    /// generated client and never exposes it to a caller.
    #[cfg(test)]
    async fn capture_test_work_order(&self, work_order: &str) {
        let mut capture = self.test_work_order_capture.lock().await;
        *capture = Some(Zeroizing::new(work_order.to_owned()));
    }

    /// Consume the in-memory test capture. Keeping the value move-only avoids
    /// a durable test transcript and makes each assertion explicitly own the
    /// short-lived secret it needs to replay against the replacement worker.
    #[cfg(test)]
    async fn take_test_work_order(&self) -> Option<Zeroizing<String>> {
        self.test_work_order_capture.lock().await.take()
    }

    /// Set the expected JWT audience for token validation.
    ///
    /// # Panics
    /// Panics if called after the service has been cloned (Arc refcount > 1).
    /// Must be called during construction, before the service is shared.
    #[allow(clippy::expect_used)]
    pub fn with_expected_audience(mut self, audience: String) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_expected_audience must be called before service is shared")
            .expected_audience = Some(audience);
        self
    }

    /// Set the unified JWT key source for verifying JWTs.
    ///
    /// # Panics
    /// Panics if called after the service has been cloned (Arc refcount > 1).
    #[allow(clippy::expect_used)]
    pub fn with_jwt_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    ) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_jwt_key_source must be called before service is shared")
            .jwt_key_source = Some(src);
        self
    }

    /// Set the DiscoveryClient for federated `at://` record resolution (#431).
    ///
    /// # Panics
    /// Panics if called after the service has been cloned (Arc refcount > 1).
    #[allow(clippy::expect_used)]
    pub fn with_discovery_client(
        mut self,
        client: Arc<hyprstream_rpc_std::discovery_client::DiscoveryClient>,
    ) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_discovery_client must be called before service is shared")
            .discovery_client = Some(client);
        self
    }

    /// Derive the deterministic InferenceService transport for a model ref (#320).
    ///
    /// Resolved via the registry (the local [`hyprstream_rpc::Resolver`] backend):
    /// the `Inproc` arm for a co-located service. The spawner registers it in the
    /// in-process dial registry and the router dials it via `dial()`.
    /// Allocate the next monotonic load epoch (EV7/#649). Each genuine load
    /// attempt (one that proceeds past the already-loaded / already-pending
    /// fast paths) draws a fresh epoch so its terminal latches under a new
    /// [`ModelLoadKey`] — reload ⇒ new terminal, monotonic write-once per epoch.
    fn allocate_load_epoch(&self) -> u64 {
        self.load_epoch.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Latch the retained terminal for a completed load attempt (EV7/#649) —
    /// the host-side retain a late `load --wait` reads from. Monotonic
    /// write-once per (`model_ref`, `epoch`): the first terminal for an attempt
    /// wins; a real attempt always latches under a fresh epoch, so the guard is
    /// belt-and-suspenders. Split out so the latch decision is unit-testable
    /// independent of the full load path.
    fn latch_load_terminal(
        &self,
        instance: &InferenceInstanceId,
        epoch: u64,
        result: &Result<String>,
    ) {
        let terminal = match result {
            Ok(endpoint) => Terminal {
                value: ModelLoadTerminal::Loaded { endpoint: endpoint.clone() },
                latched_by: "model-service".to_owned(),
            },
            Err(e) => Terminal {
                value: ModelLoadTerminal::LoadFailed { error: e.to_string() },
                latched_by: "model-service".to_owned(),
            },
        };
        let key = ModelLoadKey {
            tenant: instance.tenant().to_owned(),
            model_ref: instance.model_ref().to_owned(),
            epoch,
        };
        let _ = self.load_terminals.latch(key, terminal);
    }

    /// Atomically admit the intercepted (non-blocking) load path with unload's
    /// reservation and cache transition. `Accepted` reserves `pending_loads`
    /// before the response is sent; an unload that arrives afterwards rejects
    /// against that reservation rather than invalidating an accepted request.
    async fn reserve_intercepted_load(
        &self,
        instance: &InferenceInstanceId,
    ) -> Result<InterceptedLoadAdmission> {
        self.validate_staging_model_checkout(instance).await?;
        self.admit_instance(instance)?;
        let _load_unload_gate = self.load_unload_gate.lock().await;
        self.ensure_not_unloading(instance).await?;

        {
            let mut cache = self.loaded_models.write().await;
            if let Some(model) = cache.get_mut(instance) {
                model.last_used = Instant::now();
                return Ok(InterceptedLoadAdmission::Loaded {
                    reach: Self::model_reach(&model.network_transport),
                });
            }
        }

        let mut pending = self.pending_loads.lock();
        if pending.contains_key(instance) {
            return Ok(InterceptedLoadAdmission::Pending);
        }

        let attempt = self.allocate_load_epoch();
        pending.insert(instance.clone(), attempt);
        Ok(InterceptedLoadAdmission::Accepted(InterceptedLoadReservation::new(
            Arc::clone(&self.inner),
            instance.clone(),
            attempt,
        )))
    }

    fn validate_model_load_ref(&self, model_ref: &str) -> Result<()> {
        if let Some(pin) = &self.config.staging_model_pin {
            anyhow::ensure!(pin.admits(model_ref), "Model.load is restricted to the configured staging modelRef");
        }
        Ok(())
    }

    async fn validate_staging_model_checkout(&self, instance: &InferenceInstanceId) -> Result<()> {
        let Some(pin) = &self.config.staging_model_pin else { return Ok(()); };
        self.validate_model_load_ref(instance.model_ref())?;
        let model_ref = ModelRef::parse(instance.model_ref())?;
        let tracked = self.registry.get_by_name(model_ref.name()).await
            .map_err(|e| anyhow!("staging model '{}' not found in registry: {e}", model_ref.name()))?;
        let repo_client = self.registry.repo(&tracked.id);
        let worktrees = repo_client.list_worktrees().await?;
        let branch = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(branch) => branch,
            _ => anyhow::bail!("staging modelRef must select its configured local branch"),
        };
        anyhow::ensure!(worktrees.iter().any(|worktree| worktree.branch_name == branch.as_str()),
            "staging worktree for {}:{} not found", model_ref.name(), branch);
        let path = crate::storage::StoragePaths::new()?.worktree_path(model_ref.name(), branch)?;
        crate::storage::pinned_model::verify_selected_checkout(&path, pin.commit)
            .with_context(|| format!("admit staging worktree for {}", pin.model_ref))
    }

    /// Execute one previously admitted intercepted load. Its pending reservation
    /// was created under `load_unload_gate`; unload rejects against that marker
    /// until this continuation has finished and released it.
    async fn run_intercepted_load(
        &self,
        instance: &InferenceInstanceId,
        max_context: Option<u32>,
        kv_quant: Option<KVQuantType>,
        mut reservation: InterceptedLoadReservation,
    ) -> Result<String> {
        let epoch = reservation.attempt;
        let result = self.load_model_inner(instance, max_context, kv_quant).await;
        reservation.release();
        self.latch_load_terminal(instance, epoch, &result);
        result
    }

    /// Build an instance key only from the authority-verified tenant binding.
    fn inference_instance(
        ctx: &EnvelopeContext,
        model_ref_str: &str,
    ) -> Result<InferenceInstanceId> {
        let object_label = crate::services::inference::inference_object_label();
        crate::services::inference::enforce_inference_mac(ctx, &object_label)?;
        InferenceInstanceId::new(&ctx.domain()?, model_ref_str, 0)
    }

    /// Enforce the configured fault-radius contract before touching model state.
    fn admit_instance(&self, instance: &InferenceInstanceId) -> Result<()> {
        self.config.inference_deployment.validate()?;
        match &self.config.inference_deployment.isolation {
            InferenceIsolationProfile::InProcess => {
                admit_in_process_tenant(&self.in_process_tenant, instance.tenant())
            }
            InferenceIsolationProfile::PerTenantSubprocess
            | InferenceIsolationProfile::PerTenantMicroVmTask { .. } => {
                anyhow::bail!(
                    "inference isolation profile {:?} requires the external tenant launcher; \
                     refusing to downgrade to an in-process engine",
                    self.config.inference_deployment.isolation
                )
            }
        }
    }

    fn inference_transport(instance: &InferenceInstanceId) -> TransportConfig {
        registry().endpoint(&instance.service_name(), SocketKind::Rep)
    }

    /// The deterministic local InferenceService endpoint string for diagnostics.
    ///
    /// This is never advertised. Loaded status and lifecycle events use the
    /// separately bound Iroh network reach.
    fn inference_endpoint(instance: &InferenceInstanceId) -> String {
        Self::inference_transport(instance).endpoint_string()
    }

    /// The network-routable reach list a remote caller would use to dial this
    /// model's InferenceService (#320). An `Inproc` transport is never
    /// advertised; a networked (Quic/Iroh) reach maps through the single
    /// dial-to-wire reach codec.
    fn model_reach(transport: &TransportConfig) -> Vec<WireTransportConfig> {
        hyprstream_rpc::moq_stream::dial_transport_to_wire(transport)
            .into_iter()
            .collect()
    }

    /// Pure HRW/least-loaded + session-affinity ranking decision (#523 P3):
    /// which candidate in `load_state` the router picked for `session_id`,
    /// classified against the co-located node. Contains no I/O and no
    /// `InferenceClient` — reused directly by unit tests that construct a
    /// synthetic multi-entry `load_state` without needing a live model.
    fn route_decision(
        router: &mut crate::services::router::CellRouter,
        load_state: &[crate::services::router::InferenceServerInfo],
        co_located: crate::services::router::ReplicaId,
        session_id: &str,
        now: Instant,
    ) -> RouteDecision {
        match router.place(session_id, load_state, now) {
            Some(placement) if placement.replica_id == co_located => RouteDecision::CoLocated,
            Some(placement) => RouteDecision::Remote(placement.replica_id),
            None => RouteDecision::NoHealthyCandidate,
        }
    }

    /// Router seam (#322 leaf cell-router over the #320 single-select base,
    /// unified with #523 P3): pick ONE InferenceService for a request from a
    /// loaded model's replica set, applying capacity-weighted HRW +
    /// least-loaded + session affinity (via [`Self::route_decision`] /
    /// `CellRouter::place`) over the P1 candidate set resolved into
    /// `load_state`.
    ///
    /// A co-located selection returns the in-process client directly. A remote
    /// selection treats its transport only as a selector into Discovery's
    /// checkpoint-current, identity-bound candidate set. Resolution,
    /// authorization, readiness, or dial failure marks that replica down and
    /// re-runs placement; exhaustion fails closed.
    ///
    /// `session_id` is the placement key (defaults to "default" when the caller
    /// has no session — keeps HRW stable for non-session-scoped requests like
    /// `apply_chat_template`).
    /// Locality-aware selection: the caller needs [`RoutedReplica`] to apply
    /// the internal work order ONLY on the co-located arm (K3 finding M1).
    async fn select_inference_routed(
        &self,
        model: &mut LoadedModel,
        session_id: &str,
        bearer: Option<&str>,
    ) -> Result<RoutedReplica<InferenceClient>> {
        let co_located = Self::co_located_replica_id(model);
        let resolution_service_name = model.instance.service_name();
        let signing_key = self.signing_key.clone();
        let bearer = bearer.map(ToOwned::to_owned);
        let routed = route_and_dial_replica(
            &mut model.router,
            &model.load_state,
            co_located,
            session_id,
            model.client.clone(),
            move |candidate| {
                let resolution_service_name = resolution_service_name.clone();
                let signing_key = signing_key.clone();
                let bearer = bearer.clone();
                async move {
                    anyhow::ensure!(
                        matches!(
                            &candidate.transport.endpoint,
                            EndpointType::Quic { .. } | EndpointType::Iroh { .. }
                        ),
                        "remote inference replica advertised a non-network transport"
                    );
                    // Cross-host replicas are OTHER services, not this Model's
                    // subprocessor: they still see the caller's credential as
                    // a relayed delegated bearer under their own admission
                    // contract. No bearer ⇒ no remote relay (fail-closed).
                    let relay_bearer = bearer
                        .clone()
                        .ok_or_else(|| anyhow!(
                            "remote inference replica selection requires a relayable caller bearer"
                        ))?;
                    let rpc = hyprstream_discovery::production_inference_rpc_client_at_transport(
                        &resolution_service_name,
                        &candidate.transport,
                        signing_key,
                        None,
                    )?;
                    let client = InferenceClient::new(rpc)
                        .with_delegated_bearer(relay_bearer);
                    anyhow::ensure!(
                        client.is_ready().await?,
                        "remote inference replica is not ready"
                    );
                    Ok(client)
                }
            },
        )
        .await?;
        info!(
            replica_id = %hex::encode(routed.replica_id.as_bytes()),
            locality = ?routed.locality,
            session_id,
            tenant = model.instance.tenant(),
            model_ref = model.instance.model_ref(),
            "selected inference replica"
        );
        Ok(routed)
    }

    /// Opaque placement identifier for the co-located InferenceService.
    fn co_located_replica_id(model: &LoadedModel) -> crate::services::router::ReplicaId {
        // `load_state[0]` is the co-located entry (seeded first at load). All
        // future multi-replica entries carry their own opaque placement ID.
        model
            .load_state
            .first()
            .map(|info| info.replica_id)
            .unwrap_or(crate::services::router::ReplicaId::from_bytes([0u8; 32]))
    }

    /// Resolve a model identifier string to a [`ModelRef`].
    ///
    /// Accepts three prefixes (#395 prefix-dispatch grammar, no capnp/wire change
    /// — `modelRef` stays `Text` in the schema):
    ///
    /// ```text
    /// modelRef ::= "at://" <at-uri>   # federated (resolve via atproto NAME → git OID)
    ///            | "did:" <did>       # federated, bare-DID form
    ///            | <name> [":" <gitref>]  # local ModelRef (unchanged)
    /// ```
    ///
    /// The federated branch (`at://`, `did:`) is resolved via the atproto record
    /// store from #392 (PDS record → git OID). Until that store is wired up the
    /// federated branch logs the attempt and falls through to the local
    /// [`ModelRef::parse`], preserving backward compatibility — every existing
    /// `name:ref` caller keeps working unchanged.
    ///
    /// Local ModelRefs (no `at://` / `did:` prefix) are parsed by
    /// [`ModelRef::parse`] exactly as before.
    async fn resolve_model_ref(&self, model_ref_str: &str) -> Result<ModelRef> {
        match model_ref_dispatch(model_ref_str) {
            ModelRefDispatch::Federated { scheme, rest } => {
                // Federated resolution (#431): at:// → DiscoveryService.getRecord →
                // CAR proof → verify offline → extract currentOid → local ModelRef.
                // Only the full `at://<did>/<collection>/<rkey>` form resolves here;
                // anything else (bare `did:`, partial at://) falls through to local.
                if scheme == "at://" {
                    match self.resolve_federated_at_uri(model_ref_str).await {
                        Ok(Some(model_ref)) => return Ok(model_ref),
                        Ok(None) => {
                            // Attempted-but-unresolvable federated ref. FAIL CLOSED
                            // with a clear error rather than falling through to
                            // ModelRef::parse — which would mis-split `at://did:..`
                            // on the first ':' into model="at", git_ref="//did:..",
                            // a garbage local ref that fails later with a misleading
                            // "not found". `Ok(None)` means either no DiscoveryClient
                            // is configured (federation not enabled) or the string is
                            // not a full at://<did>/<collection>/<rkey>.
                            if self.inner.discovery_client.is_none() {
                                anyhow::bail!(
                                    "federation not enabled: cannot resolve federated ref '{model_ref_str}' \
                                     (no DiscoveryClient configured on this node)"
                                );
                            }
                            anyhow::bail!(
                                "federated ref not resolvable: '{model_ref_str}' is not a full \
                                 at://<did>/<collection>/<rkey>"
                            );
                        }
                        Err(e) => {
                            // Attempted and failed (record denied/missing/proof invalid):
                            // surface it; never mask with a local parse.
                            return Err(e);
                        }
                    }
                }
                // `did:` (bare-DID) is reserved for federated resolution but has no
                // record-fetch path yet — fail closed, do NOT reinterpret as local.
                anyhow::bail!(
                    "federated ModelRef scheme '{scheme}' for '{rest}' is not resolvable on this node \
                     (bare did: resolution not implemented; use a full at:// record ref)"
                )
            }
            ModelRefDispatch::Local => ModelRef::parse(model_ref_str),
        }
    }

    /// Resolve a full `at://<did>/<collection>/<rkey>` to a local [`ModelRef`]
    /// via DiscoveryService.getRecord (#431).
    ///
    /// Steps: call `getRecord` over the (configured) DiscoveryClient → parse the
    /// returned CARv1 proof → locate the record block → decode it → extract its
    /// `currentOid` (a git-raw CID) → decode the git OID → form a local ModelRef.
    ///
    /// Returns `Ok(None)` when no DiscoveryClient is configured or the ref is not
    /// a full at-uri (caller falls through to local resolution). Returns `Err`
    /// when resolution was attempted but failed (record denied/missing/invalid).
    ///
    /// INTEGRITY NOTE: the CAR proof's ES256 commit signature is verified offline
    /// against the account's published `#atproto` P-256 key via
    /// `hyprstream_pds::car::verify_record_proof`. Resolving that published key
    /// requires fetching the remote DID document (`#atproto` verification method),
    /// for which there is not yet an in-crate resolver — so today we verify the
    /// CAR's *structural* integrity (it parses, the commit is its root, the record
    /// block is present and decodes, and the record CID is the one the proof
    /// addresses) and extract the OID. Full signature verification against the
    /// fetched DID key is a follow-up (DID-document resolution); the untrusted-
    /// relay posture is preserved because the eventual key check is the same
    /// `verify_record_proof` the e7 harness already exercises.
    async fn resolve_federated_at_uri(&self, at_uri: &str) -> Result<Option<ModelRef>> {
        let Some(dc) = self.inner.discovery_client.as_ref() else {
            return Ok(None);
        };

        // Parse at://<did>/<collection>/<rkey>. The DID may contain ':' but no
        // '/', so the post-prefix remainder splits cleanly into three parts.
        let rest = match at_uri.strip_prefix("at://") {
            Some(r) => r,
            None => return Ok(None),
        };
        let mut parts = rest.splitn(3, '/');
        let (did, collection, rkey) = match (parts.next(), parts.next(), parts.next()) {
            (Some(d), Some(c), Some(r)) if !d.is_empty() && !c.is_empty() && !r.is_empty() => {
                (d.to_owned(), c.to_owned(), r.to_owned())
            }
            // Not a full at-uri — fall through to local resolution.
            _ => return Ok(None),
        };

        let car_resp = dc
            .get_record(&hyprstream_rpc_std::discovery_client::GetRecordRequest {
                uri: at_uri.to_owned(),
                did: did.clone(),
                collection: collection.clone(),
                rkey: rkey.clone(),
            })
            .await
            .map_err(|e| anyhow!("DiscoveryService.getRecord failed for {at_uri}: {e}"))?;

        // Parse the CARv1 proof and pull out the record block.
        let (roots, blocks) = hyprstream_pds::car::parse_car_v1(&car_resp.car)
            .map_err(|e| anyhow!("getRecord returned an unparseable CAR for {at_uri}: {e}"))?;
        let commit_cid = roots
            .first()
            .copied()
            .ok_or_else(|| anyhow!("getRecord CAR for {at_uri} has no root commit CID"))?;

        // Decode the commit (the proof's root) — confirms the relay returned a
        // structurally valid signed commit. (Signature verification against the
        // DID's published #atproto key is the follow-up noted above.)
        let _commit = blocks
            .iter()
            .find(|(c, _)| *c == commit_cid)
            .map(|(_, b)| hyprstream_pds::commit::Commit::from_dag_cbor(b))
            .transpose()
            .map_err(|e| anyhow!("getRecord CAR commit block did not decode: {e}"))?
            .ok_or_else(|| anyhow!("getRecord CAR for {at_uri} is missing its commit block"))?;

        // Find the record block: the only block that decodes as a ModelRecord.
        // (The CAR also contains the commit + MST nodes, which are not records.)
        let record = blocks
            .iter()
            .filter(|(c, _)| *c != commit_cid)
            .find_map(|(_, b)| hyprstream_pds::record::ModelRecord::from_dag_cbor(b).ok())
            .ok_or_else(|| anyhow!("getRecord CAR for {at_uri} contained no ai.hyprstream.model record"))?;

        // currentOid is a git-raw CID string → decode to the git OID hex.
        let cid = hyprstream_rpc::cid::decode_cid(&record.current_oid)
            .map_err(|e| anyhow!("record currentOid is not a valid CID: {e}"))?;
        let oid_hex = cid
            .multihash
            .digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        info!(
            %at_uri,
            did = %did,
            current_oid = %record.current_oid,
            git_oid = %oid_hex,
            "resolved federated at-uri via DiscoveryService.getRecord (#431)"
        );

        // Form a local ModelRef pinned to the resolved git OID. The name is the
        // DID-derived repo handle; the git ref is the resolved commit OID.
        let local_ref = format!("{did}:{oid_hex}");
        match ModelRef::parse(&local_ref) {
            Ok(model_ref) => Ok(Some(model_ref)),
            // If the resolved (name:oid) shape isn't a valid local ModelRef yet
            // (the local registry mapping for federated repos is a follow-up),
            // surface a clear error rather than a misleading local fallthrough.
            Err(e) => Err(anyhow!(
                "resolved {at_uri} → git OID {oid_hex}, but local ModelRef::parse rejected {local_ref:?}: {e}"
            )),
        }
    }

    /// Load a model by reference with optional per-model config, returns the inference endpoint
    async fn load_model(
        &self,
        instance: &InferenceInstanceId,
        max_context: Option<u32>,
        kv_quant: Option<KVQuantType>,
    ) -> Result<String> {
        self.validate_staging_model_checkout(instance).await?;
        self.admit_instance(instance)?;
        let model_ref_str = instance.model_ref();
        // A load's admission decision is atomic with unload's reservation and
        // cache removal. Without this guard, a load could observe "not
        // unloading", then begin after the old worker has been removed.
        let load_unload_gate = self.load_unload_gate.lock().await;
        self.ensure_not_unloading(instance).await?;
        // Check if already loaded
        {
            let mut cache = self.loaded_models.write().await;
            if let Some(model) = cache.get_mut(instance) {
                model.last_used = Instant::now();
                debug!("Model {} already loaded", model_ref_str);
                return Ok(model.network_transport.endpoint_string());
            }
        }

        // Atomically check-and-insert into pending_loads (prevents duplicate GPU loads
        // when multiple requests arrive during the ~40s load window).
        // HashMap::insert is guarded by contains_key so a direct attempt owns
        // the epoch stored beside its pending marker.
        let epoch;
        {
            let mut pending = self.pending_loads.lock();
            if pending.contains_key(instance) {
                anyhow::bail!(
                    "Model {} is already being loaded — please retry shortly",
                    model_ref_str
                );
            }
            epoch = self.allocate_load_epoch();
            pending.insert(instance.clone(), epoch);
        }
        drop(load_unload_gate);

        // EV7/#649: the gated pending insertion allocated this genuine load's
        // fresh epoch, so its terminal latches under a new key on completion.
        let result = self.load_model_inner(instance, max_context, kv_quant).await;

        // Always remove from pending, whether load succeeded or failed
        self.pending_loads.lock().remove(instance);

        // EV7/#649: latch the retained terminal (Loaded / LoadFailed) for this
        // load attempt — the host-side retain a late `load --wait` reads.
        self.latch_load_terminal(instance, epoch, &result);

        if result.is_ok() {
            info!("Model {} loaded successfully", model_ref_str);
        }
        result
    }

    /// Inner model loading logic, called by load_model() which manages pending_loads.
    async fn load_model_inner(
        &self,
        instance: &InferenceInstanceId,
        max_context: Option<u32>,
        kv_quant: Option<KVQuantType>,
    ) -> Result<String> {
        let model_ref_str = instance.model_ref();
        self.validate_model_load_ref(model_ref_str)?;
        // Parse/resolve model reference
        let model_ref = self.resolve_model_ref(model_ref_str).await?;

        // Get model path from registry
        #[cfg(test)]
        crate::services::restart_diag::phase("registry.get_by_name.enter");
        let tracked = self.registry.get_by_name(model_ref.name()).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", model_ref.name(), e))?;
        #[cfg(test)]
        crate::services::restart_diag::phase("registry.get_by_name.done");
        let repo_client = self.registry.repo(&tracked.id);

        let branch_name = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            crate::storage::GitRef::Commit(_) => repo_client.get_head().await
                .context("resolve selected worktree branch for pinned commit")?,
            crate::storage::GitRef::Tag(_) | crate::storage::GitRef::Revspec(_) => {
                anyhow::bail!("tag and revspec model loads are unsupported; use a full commit OID")
            }
            crate::storage::GitRef::DefaultBranch => {
                repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned())
            }
        };
        #[cfg(test)]
        crate::services::restart_diag::phase("registry.list_worktrees.enter");
        let worktrees = repo_client.list_worktrees().await?;
        #[cfg(test)]
        crate::services::restart_diag::phase("registry.list_worktrees.done");
        if !worktrees.iter().any(|wt| wt.branch_name == branch_name) {
            return Err(anyhow!("worktree for {}:{} not found", model_ref.name(), branch_name));
        }
        // Derive worktree path locally
        let storage_paths = crate::storage::StoragePaths::new()?;
        let model_path = storage_paths.worktree_path(model_ref.name(), &branch_name)?;

        if !model_path.exists() {
            return Err(anyhow!(
                "Model worktree not found for {}. Please clone the model first.",
                model_ref_str
            ));
        }

        let pinned_artifact = if let Some(pin) = &self.config.staging_model_pin {
            anyhow::ensure!(model_ref_str == pin.model_ref, "staging modelRef changed after admission");
            Some(crate::storage::pinned_model::acquire_pinned_model(&model_path, pin.commit).await?)
        } else if let crate::storage::GitRef::Commit(oid) = &model_ref.git_ref {
            Some(crate::storage::pinned_model::acquire_pinned_model(&model_path, *oid).await?)
        } else {
            None
        };
        let model_input_path = pinned_artifact.as_ref()
            .map_or(model_path.as_path(), |artifact| artifact.root());

        let endpoint = Self::inference_endpoint(instance);

        info!("Loading model {} at endpoint {}", model_ref_str, endpoint);

        // Create runtime config - use per-model config if provided, otherwise service defaults
        let runtime_config = RuntimeConfig {
            max_context: max_context.or(self.config.max_context),
            kv_quant_type: kv_quant.unwrap_or(self.config.kv_quant),
            use_gpu: self.config.inference_deployment.compute != InferenceCompute::Cpu,
            gpu_device_id: if self.config.inference_deployment.compute == InferenceCompute::Cpu {
                None
            } else {
                RuntimeConfig::default().gpu_device_id
            },
            devices: if self.config.inference_deployment.compute == InferenceCompute::Cpu {
                Vec::new()
            } else {
                RuntimeConfig::default().devices
            },
            ..Default::default()
        };

        // Obtain FsOps from the registry for path-contained adapter I/O
        let fs: Option<hyprstream_rpc_std::registry_client::WorktreeClient> = Some(repo_client.worktree(&branch_name));

        // Start InferenceService for this model via standard Spawnable infrastructure
        let spawner = hyprstream_service::ServiceSpawner::threaded();

        // Resolve the InferenceService transport via the typed `TransportConfig`
        // (#320). For a co-located service this is the `Inproc` arm; the spawner
        // registers it in the in-process dial registry (`register_inproc`) and
        // the same typed transport builds the client below — no string parsing on
        // the inference client path. The service also binds an Iroh endpoint
        // before readiness and publishes that separate network reach below.
        let transport = Self::inference_transport(instance);
        let mut service_config = crate::services::InferenceServiceConfig::new(
            &model_path,
            runtime_config,
            self.signing_key.verifying_key(),
            self.signing_key.clone(),
            transport.clone(),
            self.policy_transport.clone(),
            fs,
        );
        if let Some(ref artifact) = pinned_artifact {
            service_config = service_config.with_pinned_artifact(Arc::clone(artifact));
        }
        let mut service_config = service_config.with_instance_identity(
            instance.service_name(),
            instance.tenant().to_owned(),
            self.signing_key.verifying_key(),
        )
        .with_model_ref(instance.model_ref().to_owned())
        .with_stream_plane(
            Arc::clone(&self.producer_reach_config),
            Arc::clone(&self.moq_origin),
        );
        if let Some(ref aud) = self.expected_audience {
            service_config = service_config.with_expected_audience(aud.clone());
        }
        if let Some(ref src) = self.jwt_key_source {
            service_config = service_config.with_jwt_key_source(src.clone());
        }
        // Bounded worker incarnation binding (Sol plan): this spawn attempt
        // allocates a private handoff channel and the next generation. The
        // worker generates its own 256-bit incarnation inside its service
        // thread and reports `(generation, instance, controller, audience)`
        // through the handoff after its engine is initialized; Model accepts
        // the handoff only for THIS attempt's expected values. A worker that
        // dies before reporting (or a missing/failed handoff) fails the load.
        let generation = self
            .inner
            .incarnation_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let (incarnation_tx, incarnation_rx) = tokio::sync::oneshot::channel();
        service_config = service_config.with_incarnation_handoff(incarnation_tx, generation);

        let network_reach = service_config.network_reach_handle();
        #[cfg(test)]
        crate::services::restart_diag::phase("inference.spawn.enter");
        let service_handle = spawner.spawn(service_config).await
            .map_err(|e| anyhow!("Failed to spawn inference service: {}", e))?;
        let spawn_guard = SpawnPublicationGuard::new(service_handle);
        #[cfg(test)]
        crate::services::restart_diag::phase("inference.spawn.done");
        let network_transport = network_reach
            .read()
            .clone()
            .ok_or_else(|| anyhow!("InferenceService became ready without network reach"))?;
        let endpoint = network_transport.endpoint_string();

        // Await the worker's private incarnation handoff. Missing (worker died
        // before init), timed out, or mismatched (instance/controller/generation)
        // all FAIL THE LOAD closed — no binding, no minting from this attempt.
        let expected_controller = self.signing_key.verifying_key().to_bytes();
        const HANDOFF_TIMEOUT: std::time::Duration =
            std::time::Duration::from_secs(crate::services::inference::INCARNATION_HANDOFF_TIMEOUT_SECS);
        #[cfg(test)]
        crate::services::restart_diag::phase("incarnation.handoff.enter");
        let incarnation_binding = match tokio::time::timeout(HANDOFF_TIMEOUT, incarnation_rx).await {
            Err(_) => anyhow::bail!(
                "pinned worker incarnation handoff timed out after {}s without a ready worker",
                crate::services::inference::INCARNATION_HANDOFF_TIMEOUT_SECS
            ),
            Ok(Err(_)) => {
                anyhow::bail!("pinned worker died before reporting its incarnation handoff")
            }
            Ok(Ok(handoff)) => handoff,
        };
        #[cfg(test)]
        crate::services::restart_diag::phase("incarnation.handoff.done");
        anyhow::ensure!(
            incarnation_binding.generation == generation,
            "stale incarnation handoff: worker reported generation {} for attempt {generation}",
            incarnation_binding.generation
        );
        anyhow::ensure!(
            incarnation_binding.instance_service_name == instance.service_name(),
            "incarnation handoff names the wrong deterministic instance"
        );
        anyhow::ensure!(
            incarnation_binding.controller_pubkey == expected_controller,
            "incarnation handoff names the wrong pinned controller"
        );

        // Create client for this service from the typed transport (#320).
        // Inference services share the model service's signing key, so use our
        // own verifying key directly — no PolicyService lookup needed.
        let client = InferenceClient::for_local_transport_bootstrap(
            &transport,
            self.signing_key.clone(),
            self.signing_key.verifying_key(),
            None,
        )?;

        // Load TTT config from model's config.json (if TTT is enabled)
        let ttt_config = crate::runtime::model_config::ModelConfig::load_training_config(model_input_path)
            .and_then(|tc| {
                if tc.is_enabled() && tc.mode == crate::config::TrainingMode::TestTimeTraining {
                    Some(crate::training::ttt::TTTConfig {
                        learning_rate: tc.ttt.learning_rate,
                        gradient_steps: tc.ttt.gradient_steps,
                        max_grad_norm: tc.ttt.max_grad_norm,
                        min_input_length: tc.ttt.min_input_length,
                        max_ttt_context: tc.ttt.max_ttt_context,
                        enabled: true,
                        ..crate::training::ttt::TTTConfig::default()
                    })
                } else {
                    None
                }
            });

        // Load generation parameter defaults from model's generation_config.json
        let generation_defaults = crate::config::SamplingParams::from_model_path(model_input_path)
            .await
            .unwrap_or_default();

        // Check if we need to evict
        {
            let mut cache = self.loaded_models.write().await;
            if cache.len() >= self.config.max_models {
                if let Some((evicted_ref, mut evicted)) = cache.pop_lru() {
                    info!(
                        "Evicting model {} to load {}",
                        evicted_ref.model_ref(),
                        model_ref_str
                    );
                    // Stop the evicted service in background (fire-and-forget)
                    #[allow(clippy::let_underscore_future)]
                    let _ = tokio::spawn(async move {
                        let _ = evicted.service_handle.stop().await;
                    });
                }
            }

            // Add to cache
            let service_handle = spawn_guard.publish()?;
            cache.put(
                instance.clone(),
                LoadedModel {
                    instance: instance.clone(),
                    model_ref: model_ref_str.to_owned(),
                    pinned_commit: pinned_artifact.as_ref().map(|artifact| artifact.commit()),
                    transport: transport.clone(),
                    network_transport,
                    service_handle,
                    client,
                    // Worker incarnation binding for THIS run (Sol bounded
                    // incarnation plan): minting uses exactly this audience.
                    incarnation: incarnation_binding.incarnation,
                    work_audience: incarnation_binding.audience,
                    load_generation: incarnation_binding.generation,
                    // #322 leaf cell-router. v1: single co-located replica. Its
                    // placement ID is purpose-separated from every transport,
                    // application-signing, and subject identity. The
                    // router fast-paths to `client` when HRW picks the
                    // co-located node; the replica set grows when cross-host
                    // reaches are resolved via the Resolver.
                    router: crate::services::router::CellRouter::default(),
                    load_state: vec![crate::services::router::InferenceServerInfo {
                        replica_id: crate::services::router::ReplicaId::from_bytes(
                            hyprstream_rpc::node_identity::derive_purpose_key(
                                &self.signing_key,
                                "hyprstream-inference-replica-placement-v1",
                            )
                            .verifying_key()
                            .to_bytes(),
                        ),
                        transport: transport.clone(),
                        gpu_memory_free: 0,
                        active_sessions: 0,
                        last_heartbeat: Instant::now(),
                    }],
                    loaded_at: Instant::now(),
                    last_used: Instant::now(),
                    ttt_config,
                    generation_defaults,
                },
            );
        }

        if let Some(ref artifact) = pinned_artifact {
            info!(model = %model_ref_str, commit = %artifact.commit(), files = artifact.file_count(),
                "loaded model from sealed exact-tree artifact");
        }

        Ok(endpoint)
    }

    async fn ensure_not_unloading(&self, instance: &InferenceInstanceId) -> Result<()> {
        anyhow::ensure!(
            !self.unloading_models.lock().await.contains(instance),
            "Model {} is unloading — please retry after teardown completes",
            instance.model_ref(),
        );
        Ok(())
    }

    /// Unload a model.
    ///
    /// The teardown is owned by an internal task before this method awaits its
    /// result. Dropping an RPC continuation therefore cannot leave an absent
    /// cache entry while the old worker's thread still drains in the
    /// background: the instance remains in `unloading_models` until a
    /// successful shutdown/join. An ambiguous stop error retains that
    /// reservation fail-closed.
    async fn unload_model(&self, instance: &InferenceInstanceId) -> Result<()> {
        let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();
        let inner = Arc::clone(&self.inner);
        let instance = instance.clone();

        tokio::spawn(async move {
            let result = Self::unload_model_owned(Arc::clone(&inner), &instance).await;
            if let Err(Err(error)) = completion_tx.send(result) {
                warn!(
                    model = %instance.model_ref(),
                    error = %error,
                    "model unload caller cancelled; owned teardown completed with an error"
                );
            }
        });

        completion_rx
            .await
            .map_err(|_| anyhow!("model unload task exited before reporting completion"))?
    }

    async fn unload_model_owned(
        inner: Arc<ModelServiceInner>,
        instance: &InferenceInstanceId,
    ) -> Result<()> {
        let model_ref_str = instance.model_ref().to_owned();
        let (model, pending_load) = {
            let load_unload_gate = inner.load_unload_gate.lock().await;
            let mut unloading = inner.unloading_models.lock().await;
            let inserted = unloading.insert(instance.clone());
            drop(unloading);
            anyhow::ensure!(
                inserted,
                "Model {} is already unloading",
                model_ref_str,
            );
            let model = {
                let mut cache = inner.loaded_models.write().await;
                cache.pop_entry(instance).map(|(_, model)| model)
            };
            let pending_load = if model.is_none() {
                inner.pending_loads.lock().contains_key(instance)
            } else {
                false
            };
            drop(load_unload_gate);
            (model, pending_load)
        };

        let Some(mut model) = model else {
            if pending_load {
                inner.unloading_models.lock().await.remove(instance);
                anyhow::bail!("Model {} is loading", model_ref_str);
            }
            inner.unloading_models.lock().await.remove(instance);
            anyhow::bail!("Model {} is not loaded", model_ref_str);
        };

        info!("Unloading model {}", model_ref_str);
        if let Err(error) = model.service_handle.stop().await {
            // `SpawnedService::stop` can report a join-task failure after the
            // handle has moved to `spawn_blocking`; that does not establish
            // that the worker thread ended. Keep the reservation fail-closed
            // rather than admitting a replacement alongside an ambiguous old
            // worker. A process-level recovery can clear this poisoned state.
            warn!(
                model = %model_ref_str,
                error = %error,
                "model unload stop failed; retaining unload reservation"
            );
            return Err(anyhow!(
                "failed to stop model {} before unload: {error}",
                model_ref_str
            ));
        }
        // Acquire delivery order before releasing admission. A successful join
        // proves the worker is gone, so a listener may reload immediately; its
        // later loaded event must wait behind this unloaded completion, but no
        // lifecycle admission lock is held across the publish await.
        let _event_publish_gate = inner.lifecycle_event_publish_gate.lock().await;
        inner.unloading_models.lock().await.remove(instance);
        #[cfg(test)]
        if let Some(before_publish) = inner.before_unloaded_event.lock().await.take() {
            let _ = before_publish.send(());
        }
        let model_name = model_ref_str.split(':').next().unwrap_or(&model_ref_str);
        let scope = format!("serve:model:{}", model_name);
        let event = crate::events::EventEnvelope::new(
            crate::events::EventSource::Model,
            scope.clone(),
            crate::events::EventPayload::ModelUnloaded {
                model_ref: model_ref_str.clone(),
            },
        );
        if let Ok(payload) = serde_json::to_vec(&event) {
            let _ = inner.event_publisher.publish("lifecycle", "unloaded", &payload).await;
        }
        Ok(())
    }

    /// Convert a TTTConfig to a generated OnlineTrainingConfig wire type.
    fn ttt_config_to_wire(cfg: &crate::training::ttt::TTTConfig) -> GenOnlineTrainingConfig {
        GenOnlineTrainingConfig {
            enabled: cfg.enabled,
            learning_rate: cfg.learning_rate,
            gradient_steps: cfg.gradient_steps,
            max_grad_norm: cfg.max_grad_norm,
            min_input_length: cfg.min_input_length,
            max_ttt_context: cfg.max_ttt_context,
        }
    }

    /// Convert SamplingParams to a generated GenerationDefaults wire type.
    fn sampling_params_to_wire(params: &crate::config::SamplingParams) -> GenGenerationDefaults {
        GenGenerationDefaults {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k.map(|v| v as u32),
            max_tokens: params.max_tokens.map(|v| v as u32),
            repeat_penalty: params.repeat_penalty,
            stop_tokens: params.stop_tokens.clone().unwrap_or_default(),
            do_sample: params.do_sample,
        }
    }

    /// Return status entries for all known models (loaded, loading, and unloading).
    /// Absence from this list means unloaded.
    async fn model_status_all(&self, verified_tenant: &str) -> Vec<GenModelStatusEntry> {
        // Take the same gate as load admission and unload reservation, then
        // snapshot every lifecycle store. A reservation is authoritative even
        // during the small inner-load interval where the cache and pending set
        // are both populated.
        let _load_unload_gate = self.load_unload_gate.lock().await;
        let unloading: HashSet<_> = self
            .unloading_models
            .lock()
            .await
            .iter()
            .filter(|instance| instance.tenant() == verified_tenant)
            .cloned()
            .collect();
        let (mut entries, loaded): (Vec<_>, HashSet<_>) = {
            let cache = self.loaded_models.read().await;
            (
                cache
                    .iter()
                    .filter(|(instance, _)| {
                        instance.tenant() == verified_tenant && !unloading.contains(*instance)
                    })
                    .map(|(_, model)| GenModelStatusEntry {
                        model_ref: model.model_ref.clone(),
                        status: "loaded".to_owned(),
                        reach: Self::model_reach(&model.network_transport),
                        loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                        last_used: model.last_used.elapsed().as_millis() as i64,
                        online_training_config: model.ttt_config.as_ref()
                            .map(Self::ttt_config_to_wire)
                            .unwrap_or_default(),
                        generation_defaults: Self::sampling_params_to_wire(&model.generation_defaults),
                    })
                    .collect(),
                cache
                    .iter()
                    .filter(|(instance, _)| {
                        instance.tenant() == verified_tenant && !unloading.contains(*instance)
                    })
                    .map(|(instance, _)| instance.clone())
                    .collect(),
            )
        };
        for instance in &unloading {
            entries.push(GenModelStatusEntry {
                model_ref: instance.model_ref().to_owned(),
                status: "unloading".to_owned(),
                reach: Vec::new(),
                loaded_at: 0,
                last_used: 0,
                online_training_config: GenOnlineTrainingConfig::default(),
                generation_defaults: GenGenerationDefaults::default(),
            });
        }
        let pending: HashSet<_> = {
            self.pending_loads
                .lock()
                .keys()
                .filter(|instance| instance.tenant() == verified_tenant)
                .cloned()
                .collect()
        };
        for instance in &pending {
            if !loaded.contains(instance) && !unloading.contains(instance) {
                entries.push(GenModelStatusEntry {
                    model_ref: instance.model_ref().to_owned(),
                    status: "loading".to_owned(),
                    reach: Vec::new(),
                    loaded_at: 0,
                    last_used: 0,
                    online_training_config: GenOnlineTrainingConfig::default(),
                    generation_defaults: GenGenerationDefaults::default(),
                });
            }
        }
        entries
    }

    /// Return status entry for a specific model ref (0 or 1 element).
    async fn model_status_single(&self, instance: &InferenceInstanceId) -> Vec<GenModelStatusEntry> {
        let _load_unload_gate = self.load_unload_gate.lock().await;
        if self.unloading_models.lock().await.contains(instance) {
            return vec![GenModelStatusEntry {
                model_ref: instance.model_ref().to_owned(),
                status: "unloading".to_owned(),
                reach: Vec::new(),
                loaded_at: 0,
                last_used: 0,
                online_training_config: GenOnlineTrainingConfig::default(),
                generation_defaults: GenGenerationDefaults::default(),
            }];
        }
        {
            let cache = self.loaded_models.read().await;
            if let Some(model) = cache.peek(instance) {
                return vec![GenModelStatusEntry {
                    model_ref: instance.model_ref().to_owned(),
                    status: "loaded".to_owned(),
                    reach: Self::model_reach(&model.network_transport),
                    loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                    last_used: model.last_used.elapsed().as_millis() as i64,
                    online_training_config: model.ttt_config.as_ref()
                        .map(Self::ttt_config_to_wire)
                        .unwrap_or_default(),
                    generation_defaults: Self::sampling_params_to_wire(&model.generation_defaults),
                }];
            }
        }
        let pending = self.pending_loads.lock();
        if pending.contains_key(instance) {
            vec![GenModelStatusEntry {
                model_ref: instance.model_ref().to_owned(),
                status: "loading".to_owned(),
                reach: Vec::new(),
                loaded_at: 0,
                last_used: 0,
                online_training_config: GenOnlineTrainingConfig::default(),
                generation_defaults: GenGenerationDefaults::default(),
            }]
        } else {
            vec![]
        }
    }

    /// Get model status
    async fn model_status(&self, instance: &InferenceInstanceId) -> ModelStatusResponse {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(instance) {
            ModelStatusResponse {
                loaded: true,
                reach: Self::model_reach(&model.network_transport),
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                ..Default::default()
            }
        } else {
            ModelStatusResponse { loaded: false, ..Default::default() }
        }
    }
    /// Mint the per-request internal execution work order for one
    /// already-authorized Model→Inference operation (user-approved D5
    /// single-service boundary).
    ///
    /// The caller credential does NOT cross this hop. The order is signed by
    /// Model, audience-pinned to the one allocated instance, bound to the
    /// verified tenant/model and the exact downstream dispatch coordinate, and
    /// carries the caller's verified claims snapshot so TTT state, stream
    /// ownership, and accounting keep attributing to the ORIGINAL caller.
    fn mint_internal_work_token(
        &self,
        instance: &InferenceInstanceId,
        ctx: &EnvelopeContext,
        scope: &hyprstream_rpc::auth::internal_work::InternalWorkScope,
        live_work_audience: &str,
    ) -> Result<String> {
        use hyprstream_rpc::auth::internal_work::{
            encode_internal_work, InternalWorkClaims, INTERNAL_WORK_ISSUER, MAX_LIFETIME_SECS,
        };
        let caller = ctx.claims().ok_or_else(|| {
            anyhow!("internal work requires a verified caller identity at Model ingress")
        })?;
        let subject = ctx.subject();
        anyhow::ensure!(
            !subject.is_anonymous(),
            "internal work requires an authenticated original subject"
        );
        let now = chrono::Utc::now().timestamp();
        let exp = caller.exp.min(now + MAX_LIFETIME_SECS);
        anyhow::ensure!(
            exp > now,
            "caller credential is expired; refusing to mint internal work"
        );
        // The caller snapshot's subject is canonicalized to the RESOLVED
        // subject string so the snapshot and the work-order subject are
        // provably identical (including federated callers, whose raw token
        // `sub` differs from the resolved Casbin label) — K3 finding m1.
        let subject_string = subject.to_string();
        let mut caller_snapshot = caller.clone();
        caller_snapshot.sub = subject_string.clone();
        let claims = InternalWorkClaims {
            iss: INTERNAL_WORK_ISSUER.to_owned(),
            sub: subject_string,
            // The EXACT audience of the currently live worker incarnation
            // (Sol bounded incarnation plan) — never a recomputed
            // deterministic name, which would survive a worker restart.
            aud: live_work_audience.to_owned(),
            tenant: instance.tenant().to_owned(),
            model: instance.model_ref().to_owned(),
            resource: scope.resource.clone(),
            operation: scope.operation.clone(),
            iat: now,
            exp,
            jti: hex::encode(hyprstream_rpc::envelope::generate_nonce()),
            cnf: InternalWorkClaims::controller_cnf(&self.signing_key.verifying_key()),
            owner_did: ctx.authenticated_pairwise_did().map(|d| d.as_str().to_owned()),
            caller: caller_snapshot,
        };
        Ok(encode_internal_work(&claims, &self.signing_key))
    }

    /// The authorize coordinate the generated Inference dispatch enforces.
    ///
    /// Generated dispatch builds the per-method authorize resource as
    /// `format!("{}:{}", service, variant_pascal)` where `variant_pascal` is
    /// the schema variant name through `to_pascal_case` (codegen/handler.rs).
    /// Text-payload variants instead embed the PAYLOAD value
    /// (`{service}:{payload}`) — see the `loadLora` call site.
    ///
    /// SCOPE NOTE (K3 follow-up): this helper uppercases the first character
    /// ONLY, which matches `to_pascal_case` exactly for the CURRENT camelCase
    /// schema leaves (no `_`/`-` separators — `hasLora` → `HasLora`). It is
    /// NOT a general `to_pascal_case`: the generator also removes `_`/`-`
    /// separators and capitalizes the following character
    /// (hyprstream-rpc-build/src/util.rs). If a future leaf carries
    /// separators, use the generator's own `to_pascal_case` spelling here —
    /// the worker's exact-match PEP denies any other spelling.
    pub(crate) fn inference_dispatch_resource(variant: &str) -> String {
        let mut chars = variant.chars();
        match chars.next() {
            Some(first) => {
                format!("inference:{}{}", first.to_ascii_uppercase(), chars.as_str())
            }
            None => "inference:".to_owned(),
        }
    }

    /// Mint-side scope for one forwarded struct/Void Inference operation:
    /// the exact dispatch coordinate for `variant` plus its scope operation.
    fn inference_scope(
        variant: &str,
        operation: &str,
    ) -> hyprstream_rpc::auth::internal_work::InternalWorkScope {
        hyprstream_rpc::auth::internal_work::InternalWorkScope::new(
            Self::inference_dispatch_resource(variant),
            operation,
        )
    }

    async fn get_inference_client(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        work: hyprstream_rpc::auth::internal_work::InternalWorkScope,
    ) -> Result<InferenceClient> {
        let instance = Self::inference_instance(ctx, model_ref_str)?;
        let _endpoint = self.load_model(&instance, None, None).await?;
        let mut cache = self.loaded_models.write().await;
        let model = cache
            .get_mut(&instance)
            .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
        // Bounded worker incarnation binding (Sol plan): a cached entry whose
        // worker DIED is fail-closed — evict and deny until a fresh run
        // reports ready with a fresh incarnation. Never mint against a dead
        // worker, even if its (old) receipt would still be time-valid.
        if !model.service_handle.is_running() {
            cache.pop(&instance);
            anyhow::bail!(
                "pinned worker for model {model_ref_str} is not running; \
                 the model must be loaded again before internal work"
            );
        }
        model.last_used = Instant::now();
        // The live binding for THIS running incarnation. Minting uses exactly
        // this audience; the worker denies anything else.
        let live_work_audience = model.work_audience.clone();
        let live_generation = model.load_generation;
        // #322 placement key. The envelope does not yet carry an explicit
        // session_id; use the authenticated subject as a stable per-caller key
        // (keeps HRW affinity effective for repeat requests from the same
        // caller). When a real session_id is plumbed through `EnvelopeContext`,
        // swap it in here — the router body is session_id-keyed, not
        // subject-keyed, by design.
        let placement_key = ctx.subject().to_string();
        // The co-located pinned instance executes Model-origin internal work —
        // the caller credential is never relayed. A remote replica is a
        // different controller's service: it may receive an end-user bearer as
        // delegated authority, but never a holder-bound service credential.
        // For service callers the remote arm therefore fails its no-bearer
        // gate and the router retries a healthy co-located candidate; when no
        // such candidate exists, the request denies rather than transferring a
        // credential whose cnf belongs to the original service signer.
        let relay_bearer = Self::remote_relay_bearer(ctx).map(ToOwned::to_owned);
        let routed = self
            .select_inference_routed(model, &placement_key, relay_bearer.as_deref())
            .await?;
        let work_token =
            self.mint_internal_work_token(&instance, ctx, &work, &live_work_audience)?;
        #[cfg(test)]
        self.capture_test_work_order(&work_token).await;
        tracing::debug!(
            generation = live_generation,
            "minted internal work order against live worker incarnation"
        );
        Ok(Self::attach_work_order(routed, work_token))
    }

    /// Attach the internal work order to the routed client BY LOCALITY
    /// (K3 finding M1): the co-located subprocessor gets the work order as
    /// its DIRECT bearer; a remote-selected replica keeps the
    /// delegated-relay client the remote arm already built. Overlaying both
    /// on one client is a hard wire error (an envelope cannot carry a direct
    /// JWT and a delegated bearer), and a remote instance would deny a
    /// foreign-controller work order anyway — fail-closed on both sides.
    fn attach_work_order(
        routed: RoutedReplica<InferenceClient>,
        work_token: String,
    ) -> InferenceClient {
        match routed.locality {
            ReplicaLocality::CoLocated => routed.value.with_bearer(work_token),
            ReplicaLocality::Remote => routed.value,
        }
    }

    fn remote_relay_bearer(ctx: &EnvelopeContext) -> Option<&str> {
        // The reserved Federate credential is holder-bound at this ingress.
        // Never send it to a remote replica as a transferable bearer; only a
        // co-located worker receives Model's audience-pinned internal order.
        if ctx.is_federate_admitted() {
            return None;
        }
        Self::remote_relay_bearer_for_claims(ctx.claims(), ctx.jwt_token())
    }

    fn remote_relay_bearer_for_claims<'a>(
        claims: Option<&hyprstream_rpc::auth::Claims>,
        bearer: Option<&'a str>,
    ) -> Option<&'a str> {
        if claims.is_some_and(|claims| claims.sub.starts_with("service:")) {
            None
        } else {
            bearer
        }
    }

    /// Load a LoRA adapter from a file
    async fn load_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext, path: &str) -> Result<()> {
        // Text-payload variant (K3 finding B2): the generated dispatch
        // resource embeds the PAYLOAD (`inference:{path}`), so the work order
        // binds the exact adapter path rather than the method name.
        let client = self.get_inference_client(model_ref_str, ctx,
            hyprstream_rpc::auth::internal_work::InternalWorkScope::new(
                format!("inference:{path}"),
                "write",
            ),
        ).await?;
        client.load_lora(path).await
    }

    /// Unload the current LoRA adapter
    async fn unload_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("unloadLora", "write"),
        ).await?;
        client.unload_lora().await
    }

    /// Check if a LoRA adapter is loaded
    async fn has_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<bool> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("hasLora", "query"),
        ).await?;
        client.has_lora().await
    }

    // Training loop control - forward to InferenceService via ZMQ
    async fn writeback_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("tttWriteback", "train"),
        ).await?;
        client.ttt_writeback().await
    }

    async fn evict_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("tttEvict", "train"),
        ).await?;
        client.ttt_evict().await
    }

    async fn zero_delta(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("tttZero", "manage"),
        ).await?;
        client.ttt_zero().await
    }

    async fn get_delta_status_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<hyprstream_rpc_std::inference_client::DeltaStatusResult> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("getDeltaStatus", "query"),
        ).await?;
        client.get_delta_status().await
    }

    async fn snapshot_delta_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<hyprstream_rpc_std::inference_client::SnapshotDeltaResult> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("snapshotDelta", "write"),
        ).await?;
        client.snapshot_delta().await
    }

    async fn export_peft_adapter_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        data: &ExportPeftRequest,
    ) -> Result<ExportPeftResult> {
        let client = self.get_inference_client(model_ref_str, ctx,
            Self::inference_scope("exportPeftAdapter", "write"),
        ).await?;
        client.export_peft_adapter(data).await
    }

}

// ═══════════════════════════════════════════════════════════════════════════════
// ModelHandler Implementation — generated dispatch for top-level + typed scope traits
// ═══════════════════════════════════════════════════════════════════════════════

use hyprstream_rpc_std::model_client::{
    ModelResponseVariant,
    LoadedModelResponse, ErrorInfo, ModelHealthStatus,
    StatusRequest,
    ModelStatusEntry as GenModelStatusEntry, OnlineTrainingConfig as GenOnlineTrainingConfig,
    GenerationDefaults as GenGenerationDefaults,
    // Top-level request types
    LoadModelRequest, UnloadModelRequest,
    // TTT types (names follow inference.capnp via using-import)
    LoraConfig, TrainStepRequest, TrainStepResult,
    DeltaStatusResult,
    SaveAdaptationRequest, SaveAdaptationResult,
    SnapshotDeltaResult, ExportPeftRequest, ExportPeftResult,
    WriteTttConfigRequest,
    // Adapter types
    AdapterInfo, MergeLoraRequest,
    // Infer types (GenerationRequest follows inference.capnp name)
    GenerationRequest, ChatTemplateRequest, ModelStatusResponse,
    EmbedImagesRequest, EmbedImagesResponse,
};
use crate::services::generated::model_client::{
    AdapterHandler, InferHandler, ModelHandler, TttHandler, dispatch_model, serialize_response,
};
// AdaptationStrategy is now from inference_client (canonical source via using-import).
// model_client types reference it directly — no conversion needed.


#[async_trait::async_trait(?Send)]
impl TttHandler for ModelService {
    async fn handle_init(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &LoraConfig,
    ) -> Result<()> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("createLora", "write"),
        ).await?;
        client.create_lora(data).await
    }

    async fn handle_train(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<TrainStepResult> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("trainStep", "train"),
        ).await?;
        client.train_step(data).await
    }

    async fn handle_train_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<(hyprstream_rpc_std::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("trainStepStream", "train"),
        ).await?;
        let ephemeral_pubkey = ctx.ephemeral_pubkey()
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;
        let stream_info = client.train_step_stream(data, ephemeral_pubkey).await?;
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_writeback(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.writeback_adaptation(model_ref, ctx).await
    }

    async fn handle_evict(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.evict_adaptation(model_ref, ctx).await
    }

    async fn handle_zero(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.zero_delta(model_ref, ctx).await
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<DeltaStatusResult> {
        self.get_delta_status_forward(model_ref, ctx).await
    }

    async fn handle_save(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &SaveAdaptationRequest,
    ) -> Result<SaveAdaptationResult> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("saveAdaptation", "write"),
        ).await?;
        client.save_adaptation(data).await
    }

    async fn handle_snapshot(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<SnapshotDeltaResult> {
        self.snapshot_delta_forward(model_ref, ctx).await
    }

    async fn handle_export(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ExportPeftRequest,
    ) -> Result<ExportPeftResult> {
        self.export_peft_adapter_forward(model_ref, ctx, data).await
    }

    async fn handle_write_ttt_config(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &WriteTttConfigRequest,
    ) -> Result<()> {
        // 1. Parse/resolve model ref and resolve worktree path
        let parsed = self.resolve_model_ref(model_ref).await?;
        let tracked = self.registry.get_by_name(parsed.name()).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", parsed.name(), e))?;
        let repo_client = self.registry.repo(&tracked.id);

        let branch_name = match &parsed.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };

        let storage_paths = crate::storage::StoragePaths::new()?;
        let model_path = storage_paths.worktree_path(parsed.name(), &branch_name)?;

        if !model_path.exists() {
            return Err(anyhow!("Model worktree not found for {}", model_ref));
        }

        // 2. Build HyprstreamTrainingConfig from request
        let ttt_config = crate::config::TTTTrainingConfig {
            learning_rate: if data.learning_rate > 0.0 { data.learning_rate } else { 3e-4 },
            gradient_steps: if data.gradient_steps > 0 { data.gradient_steps } else { 3 },
            max_grad_norm: if data.max_grad_norm > 0.0 { data.max_grad_norm } else { 1.0 },
            min_input_length: if data.min_input_length > 0 { data.min_input_length } else { 32 },
            max_ttt_context: if data.max_ttt_context > 0 { data.max_ttt_context } else { 512 },
            rank_oracle: None,
            gradient_gating: None,
        };

        let training_config = crate::config::HyprstreamTrainingConfig {
            mode: crate::config::TrainingMode::TestTimeTraining,
            ttt: ttt_config,
            lora_rank: if data.lora_rank > 0 { data.lora_rank as usize } else { crate::config::default_lora_rank() },
            lora_alpha: if data.lora_alpha > 0.0 { Some(data.lora_alpha) } else { None },
            target_modules: if data.target_modules.is_empty() {
                crate::config::default_target_modules()
            } else {
                data.target_modules.clone()
            },
            ..Default::default()
        };

        // 3. Write config.json
        crate::runtime::model_config::ModelConfig::save_training_config(&model_path, &training_config)?;

        // 4. Stage and commit via worktree-scoped API
        let wt = repo_client.worktree(&branch_name);
        wt.stage_files(&StageFilesRequest {
            files: vec!["config.json".to_owned()],
        }).await?;
        wt.commit_with_author(&CommitWithAuthorRequest {
            message: "Update hyprstream_training config via RPC".to_owned(),
            author_name: "hyprstream".to_owned(),
            author_email: "noreply@hyprstream.dev".to_owned(),
        }).await?;

        info!("TTT config written for {}", model_ref);

        // 5. Auto-reload if requested and model is loaded
        if data.auto_reload {
            let instance = Self::inference_instance(ctx, model_ref)?;
            let is_loaded = {
                let cache = self.loaded_models.read().await;
                cache.contains(&instance)
            };
            if is_loaded {
                info!("Auto-reloading {} after TTT config change", model_ref);
                self.unload_model(&instance).await?;
                self.load_model(&instance, None, None).await?;
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl AdapterHandler for ModelService {
    async fn handle_load(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, value: &str,
    ) -> Result<()> {
        self.load_lora(model_ref, ctx, value).await
    }

    async fn handle_unload(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<()> {
        self.unload_lora(model_ref, ctx).await
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<bool> {
        self.has_lora(model_ref, ctx).await
    }

    async fn handle_inspect(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, value: &str,
    ) -> Result<AdapterInfo> {
        // Resolve model_ref to a worktree client (does NOT require model loaded in memory)
        let parsed = self.resolve_model_ref(model_ref).await?;
        let tracked = self.registry.get_by_name(parsed.name()).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", parsed.name(), e))?;
        let repo_client = self.registry.repo(&tracked.id);
        let branch_name = match &parsed.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };
        let fs = repo_client.worktree(&branch_name);

        // Read adapter_config.json from the adapter directory
        let config_path = format!("{}/adapter_config.json", value);
        let config_bytes = fs.read_file_chunked(&config_path).await
            .map_err(|e| anyhow!("Failed to read {}: {}", config_path, e))?;
        let config_json: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| anyhow!("Failed to parse adapter_config.json: {}", e))?;

        // Verify adapter_model.safetensors exists
        let model_path = format!("{}/adapter_model.safetensors", value);
        let stat = fs.stat_path(&model_path).await
            .map_err(|e| anyhow!("Failed to stat {}: {}", model_path, e))?;
        if !stat.exists {
            anyhow::bail!("adapter_model.safetensors not found in {}", value);
        }

        // Extract PEFT fields from config
        let rank = config_json.get("r")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;
        let lora_alpha = config_json.get("lora_alpha")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;
        let target_modules = config_json.get("target_modules")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let base_model = config_json.get("base_model_name_or_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();

        // Extract directory name from path
        let name = value.rsplit('/').next().unwrap_or(value).to_owned();

        Ok(AdapterInfo {
            name,
            path: value.to_owned(),
            rank,
            lora_alpha,
            target_modules,
            base_model,
        })
    }

    async fn handle_merge(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &MergeLoraRequest,
    ) -> Result<()> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("mergeLora", "write"),
        ).await?;
        client.merge_lora(data).await
    }
}

#[async_trait::async_trait(?Send)]
impl InferHandler for ModelService {
    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &GenerationRequest,
    ) -> Result<(hyprstream_rpc_std::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("generateStream", "infer"),
        ).await?;
        let ephemeral_pubkey = ctx.ephemeral_pubkey()
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;
        let stream_info = client.generate_stream(data, ephemeral_pubkey).await?;
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_apply_chat_template(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ChatTemplateRequest,
    ) -> Result<String> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("applyChatTemplate", "query"),
        ).await?;
        client.apply_chat_template(data).await
    }

    async fn handle_embed(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &EmbedImagesRequest,
    ) -> Result<EmbedImagesResponse> {
        let client = self.get_inference_client(model_ref, ctx,
            Self::inference_scope("embed", "infer"),
        ).await?;
        client.embed(data).await
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<ModelStatusResponse> {
        let instance = Self::inference_instance(ctx, model_ref)?;
        Ok(self.model_status(&instance).await)
    }
}

#[async_trait::async_trait(?Send)]
impl ModelHandler for ModelService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject();
        let domain = ctx.domain()?;
        let object_label = crate::services::inference::inference_object_label();
        crate::services::inference::enforce_inference_mac(ctx, &object_label)?;
        let request = PolicyCheck {
            subject: subject.to_string(),
            domain,
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        };
        if ctx.is_federate_admitted() {
            anyhow::ensure!(ctx.admitted_federate_operation(resource, operation),
                "Federate request-local operation mismatch");
            // The H2 request-use RPC has already performed the fresh Policy
            // check for this exact original holder, resource and operation,
            // and consumed the shared replay key. Generic Policy.check would
            // authorize this service's identity, not the original caller.
            return Ok(());
        }
        let allowed = crate::services::policy::check_with_holder_evidence(
            &self.policy_client,
            &request,
            ctx.jwt_token(),
            &ctx.subject(),
            ctx.original_holder_evidence(),
        ).await.unwrap_or_else(|e| {
            warn!("Policy check failed for {} on {}: {} - denying access", subject, resource, e);
            false
        });
        if allowed {
            Ok(())
        } else {
            anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
        }
    }

    async fn handle_load(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &LoadModelRequest,
    ) -> Result<ModelResponseVariant> {
        let max_ctx = data.max_context.filter(|&n| n != 0);
        let kv_q = data.kv_quant.filter(|q| *q != KVQuantType::None);
        let model_ref = &data.model_ref;
        let instance = Self::inference_instance(ctx, model_ref)?;
        match self.load_model(&instance, max_ctx, kv_q).await {
            Ok(_endpoint) => {
                let cache = self.loaded_models.read().await;
                let model = cache
                    .peek(&instance)
                    .ok_or_else(|| anyhow!("model missing after successful load"))?;
                Ok(ModelResponseVariant::LoadResult(LoadedModelResponse {
                    model_ref: model_ref.to_owned(),
                    reach: Self::model_reach(&model.network_transport),
                }))
            }
            Err(e) => Ok(ModelResponseVariant::Error(ErrorInfo {
                message: format!("Failed to load model: {e}"),
                code: "LOAD_FAILED".into(),
                details: String::new(),
            })),
        }
    }

    async fn handle_unload(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &UnloadModelRequest,
    ) -> Result<ModelResponseVariant> {
        let model_ref = &data.model_ref;
        let instance = Self::inference_instance(ctx, model_ref)?;
        match self.unload_model(&instance).await {
            Ok(()) => Ok(ModelResponseVariant::UnloadResult),
            Err(e) => Ok(ModelResponseVariant::Error(ErrorInfo {
                message: format!("Failed to unload model: {e}"),
                code: "UNLOAD_FAILED".into(),
                details: String::new(),
            })),
        }
    }

    async fn handle_status(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        data: &StatusRequest,
    ) -> Result<ModelResponseVariant> {
        let tenant = ctx.domain()?;
        let entries = if data.model_ref.is_empty() {
            self.model_status_all(&tenant).await
        } else {
            let instance = InferenceInstanceId::new(&tenant, &data.model_ref, 0)?;
            self.model_status_single(&instance).await
        };
        Ok(ModelResponseVariant::StatusResult(entries))
    }

    async fn handle_health_check(
        &self, ctx: &EnvelopeContext, _request_id: u64,
    ) -> Result<ModelResponseVariant> {
        let tenant = ctx.domain()?;
        let cache = self.loaded_models.read().await;
        let loaded_count = cache
            .iter()
            .filter(|(instance, _)| instance.tenant() == tenant.as_str())
            .count() as u32;
        let max_models = self.config.max_models as u32;
        drop(cache);
        Ok(ModelResponseVariant::HealthCheckResult(ModelHealthStatus {
            status: "healthy".into(),
            loaded_model_count: loaded_count,
            max_models,
            total_memory_bytes: 0,
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9P Filesystem Handler (FsHandler trait)
// ═══════════════════════════════════════════════════════════════════════════════

use hyprstream_rpc_std::model_client::{
    NpWalk, NpOpen, NpRead, NpWrite, NpClunk, NpStatReq, NpCreate, NpRemove,
    RWalk, ROpen, RRead, RWrite, RStat,
    Qid as GenQid, NpStat as GenNpStat,
};
use crate::services::generated::model_client::FsHandler;
use crate::services::fs::{SyntheticTree, SyntheticNode, SyntheticQid};
use hyprstream_vfs::DirEntry;

impl ModelService {
    /// Get the persistent tenant-partitioned synthetic 9P tree.
    fn fs_tree(&self, verified_tenant: &str) -> Arc<SyntheticTree> {
        if let Some(tree) = self.inner.fs_trees.get(verified_tenant) {
            return Arc::clone(tree.value());
        }
        let tree = Arc::new(self.build_fs_tree(verified_tenant.to_owned()));
        let entry = self
            .inner
            .fs_trees
            .entry(verified_tenant.to_owned())
            .or_insert(tree);
        Arc::clone(entry.value())
    }

    /// Build a synthetic 9P tree from current model service state.
    fn build_fs_tree(&self, verified_tenant: String) -> SyntheticTree {
        let inner_list = Arc::clone(&self.inner);
        let inner_resolve = Arc::clone(&self.inner);
        let list_tenant = verified_tenant.clone();
        let resolve_tenant = verified_tenant;

        SyntheticTree::new(SyntheticNode::DynamicDir {
            list: Box::new(move || {
                let cache = inner_list.loaded_models.blocking_read();
                let pending = inner_list.pending_loads.lock();
                let mut entries: Vec<DirEntry> = cache
                    .iter()
                    .filter(|(instance, _)| instance.tenant() == list_tenant)
                    .map(|(instance, _)| DirEntry {
                        name: instance.service_name(),
                        is_dir: true,
                        size: 0,
                        stat: None,
                    })
                    .collect();
                for instance in pending
                    .keys()
                    .filter(|instance| instance.tenant() == list_tenant)
                {
                    if !cache.contains(instance) {
                        entries.push(DirEntry {
                            name: instance.service_name(),
                            is_dir: true,
                            size: 0,
                            stat: None,
                        });
                    }
                }
                entries
            }),
            resolve: Box::new(move |ref_name| {
                let cache = inner_resolve.loaded_models.blocking_read();
                let model = cache
                    .iter()
                    .find(|(instance, _)| {
                        instance.tenant() == resolve_tenant
                            && instance.service_name() == ref_name
                    })
                    .map(|(_, model)| model);
                let is_loaded = model.is_some();
                let defaults_json = if let Some(model) = model {
                    serde_json::to_vec_pretty(&model.generation_defaults).unwrap_or_default()
                } else {
                    b"{}".to_vec()
                };
                let status_str = if is_loaded { "loaded" } else { "unloaded" };

                let mut children = std::collections::HashMap::new();
                let status_owned = status_str.to_owned();
                children.insert("status".to_owned(), SyntheticNode::ReadFile(
                    Box::new(move || format!("{status_owned}\n").into_bytes()),
                ));
                children.insert("defaults".to_owned(), SyntheticNode::ReadFile(
                    Box::new(move || defaults_json.clone()),
                ));
                children.insert("ctl".to_owned(), SyntheticNode::CtlFile {
                    handler: Box::new(|data, _subject| {
                        let cmd = String::from_utf8_lossy(data).trim().to_owned();
                        Ok(format!("ctl: {cmd}\n").into_bytes())
                    }),
                });
                Some(SyntheticNode::Dir { children })
            }),
        })
    }

    fn qid_to_gen(qid: &SyntheticQid) -> GenQid {
        GenQid { qtype: qid.qtype, version: qid.version, path: qid.path }
    }
}

#[async_trait::async_trait(?Send)]
impl FsHandler for ModelService {
    async fn handle_walk(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpWalk,
    ) -> Result<RWalk> {
        let tree = self.fs_tree(&ctx.domain()?);
        let owner = ctx.subject().to_string();
        let (_fid, qid) = tree.walk(&data.wnames, &owner, Some(data.newfid))
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RWalk { qid: Self::qid_to_gen(&qid) })
    }

    async fn handle_open(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpOpen,
    ) -> Result<ROpen> {
        let tree = self.fs_tree(&ctx.domain()?);
        let owner = ctx.subject().to_string();
        // Re-walk to get the fid, then open.
        let (qid, iounit) = tree.open(data.fid, data.mode, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(ROpen { qid: Self::qid_to_gen(&qid), iounit })
    }

    async fn handle_read(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpRead,
    ) -> Result<RRead> {
        let tree = self.fs_tree(&ctx.domain()?);
        let owner = ctx.subject().to_string();
        let bytes = tree.read(data.fid, data.offset, data.count, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RRead { data: bytes })
    }

    async fn handle_write(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpWrite,
    ) -> Result<RWrite> {
        let tree = self.fs_tree(&ctx.domain()?);
        let subject = ctx.subject();
        let owner = subject.to_string();
        let count = tree.write(data.fid, data.offset, &data.data, &owner, &subject)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RWrite { count })
    }

    async fn handle_clunk(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpClunk,
    ) -> Result<()> {
        let tree = self.fs_tree(&ctx.domain()?);
        let owner = ctx.subject().to_string();
        tree.clunk(data.fid, &owner);
        Ok(())
    }

    async fn handle_stat(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpStatReq,
    ) -> Result<RStat> {
        let tree = self.fs_tree(&ctx.domain()?);
        let owner = ctx.subject().to_string();
        let (qid, name) = tree.stat(data.fid, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RStat {
            stat: GenNpStat {
                qid: Self::qid_to_gen(&qid),
                mode: if qid.qtype & 0x80 != 0 { 0o040755 } else { 0o100644 },
                atime: 0,
                mtime: 0,
                length: 0,
                name,
                uid: String::new(),
                gid: String::new(),
                muid: String::new(),
            },
        })
    }

    async fn handle_create(&self, _ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, _data: &NpCreate,
    ) -> Result<ROpen> {
        anyhow::bail!("create not supported on model fs")
    }

    async fn handle_remove(&self, _ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, _data: &NpRemove,
    ) -> Result<()> {
        anyhow::bail!("remove not supported on model fs")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Load request interception — parse capnp to detect load before dispatch
// ═══════════════════════════════════════════════════════════════════════════════

/// Parsed load request data extracted from Cap'n Proto payload.
struct ParsedLoadRequest {
    model_ref: String,
    max_context: u32,
    kv_quant: hyprstream_rpc_std::model_capnp::KVQuantType,
}

impl ParsedLoadRequest {
    fn to_load_params(&self) -> (Option<u32>, Option<KVQuantType>) {
        use hyprstream_rpc_std::model_capnp::KVQuantType as CKV;
        let max_ctx = match self.max_context {
            0 => None,
            n => Some(n),
        };
        let kv_q = match self.kv_quant {
            CKV::Int8 => Some(KVQuantType::Int8),
            CKV::Nf4 => Some(KVQuantType::Nf4),
            CKV::Fp4 => Some(KVQuantType::Fp4),
            CKV::None => None,
        };
        (max_ctx, kv_q)
    }
}

impl ModelService {
    /// Try to parse a load request from the ONE decoded request body.
    /// Returns `None` for all other request variants (list, unload, health, scoped, etc.).
    ///
    /// Reads from the already-decoded message (v16 §5.2) — pointer traversal,
    /// not a second decode of the signed bytes.
    fn try_parse_load_request(
        body: &hyprstream_rpc::service::DecodedRequestBody,
    ) -> Option<(u64, ParsedLoadRequest)> {
        use hyprstream_rpc_std::model_capnp::model_request;
        use hyprstream_rpc_std::model_capnp::KVQuantType as CKV;
        use hyprstream_rpc::optional_capnp::option_uint32;
        use hyprstream_rpc_std::model_capnp::option_k_v_quant_type;
        let req = body.root::<model_request::Reader>().ok()?;
        let request_id = req.get_id();
        match req.which().ok()? {
            model_request::Which::Load(data) => {
                let data = data.ok()?;
                let model_ref = data.get_model_ref().ok()?.to_str().ok()?.to_owned();
                let max_context = match data.get_max_context().ok()?.which().ok()? {
                    option_uint32::Which::None(()) => 0u32,
                    option_uint32::Which::Some(v) => v,
                };
                let kv_quant = match data.get_kv_quant().ok()?.which().ok()? {
                    option_k_v_quant_type::Which::None(()) => CKV::None,
                    option_k_v_quant_type::Which::Some(v) => v.unwrap_or(CKV::None),
                };
                Some((request_id, ParsedLoadRequest { model_ref, max_context, kv_quant }))
            }
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RequestService Implementation — delegates to generated dispatch_model
// ═══════════════════════════════════════════════════════════════════════════════

/// Only the generated `infer.generateStream` leaf can use this request-local
/// Federate admission. The resource is the exact generated authorize value.
#[cfg(feature = "postgres")]
fn federate_model_operation(
    body: &hyprstream_rpc::service::DecodedRequestBody,
) -> Result<(String, String)> {
    use hyprstream_rpc_std::model_capnp::{infer_request, model_request};
    let request = body.root::<model_request::Reader>()?;
    let infer = match request.which()? {
        model_request::Which::Infer(inner) => inner?,
        _ => anyhow::bail!("unsupported Federate Model method"),
    };
    anyhow::ensure!(matches!(infer.which()?, infer_request::Which::GenerateStream(_)),
        "unsupported Federate inference method");
    let model_ref = infer.get_model_ref()?.to_str()?;
    anyhow::ensure!(!model_ref.is_empty(), "empty Federate model reference");
    Ok((format!("model:{model_ref}"), "infer".to_owned()))
}

#[async_trait(?Send)]
impl crate::services::RequestService for ModelService {
    fn accept_deferred_federate_credential(&self) -> bool {
        #[cfg(feature = "postgres")]
        { self.federate_dispatch.is_some() }
        #[cfg(not(feature = "postgres"))]
        { false }
    }

    async fn admit_deferred_federate_request(
        &self,
        ctx: &EnvelopeContext,
        body: &hyprstream_rpc::service::DecodedRequestBody,
        proof: &hyprstream_rpc::proof::parser::ParsedProof,
    ) -> anyhow::Result<(hyprstream_rpc::proof::verify::VerifiedProof, [u8; 32], String, String)> {
        #[cfg(feature = "postgres")]
        {
            let adapter = self.federate_dispatch.as_ref()
                .ok_or_else(|| anyhow!("Federate dispatch is disabled"))?;
            let (resource, operation) = federate_model_operation(body)?;
            let (verified, holder) = adapter.admit(
                ctx, proof, "model", 0xe7339d5d26ab3076, body.bytes(),
                &resource, &operation,
            ).await?;
            Ok((verified, holder, resource, operation))
        }
        #[cfg(not(feature = "postgres"))]
        { let _ = (ctx, body, proof); anyhow::bail!("Federate dispatch is disabled") }
    }

    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> anyhow::Result<hyprstream_rpc::service::DecodedRequestBody> {
        // The ONE bounded decode (v16 §5.2): the generated decoder derives
        // the full method leaf and returns the decoded message that policy,
        // MAC, the load fast-path, and dispatch below all consume.
        crate::services::generated::model_client::decode_model_request_body(signed_body)
    }

    async fn handle_request(&self, ctx: &EnvelopeContext, body: &hyprstream_rpc::service::DecodedRequestBody) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
        debug!(
            "Model request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );

        // Intercept load requests to avoid blocking the request loop.
        // Model loading can take 60s+ (weight transfer to GPU), which would
        // block all other model service requests (list, health, info, etc.).
        // Instead, return an immediate "accepted" response and do the actual
        // load in a Continuation (spawned via spawn_local after the REP is sent).
        if let Some((request_id, load_data)) = Self::try_parse_load_request(body) {
            // The intercepted fast path bypasses generated dispatch, so it
            // must perform the same operation authorization and audit before
            // parsing pin/ref details or consulting Registry/Git state.
            let audit_resource = "model:Load";
            let authorization =
                <Self as ModelHandler>::authorize(self, ctx, audit_resource, "write").await;
            ctx.audit_authz(audit_resource, "write", authorization.is_ok());
            authorization?;

            self.validate_model_load_ref(&load_data.model_ref)?;
            let instance = Self::inference_instance(ctx, &load_data.model_ref)?;
            self.validate_staging_model_checkout(&instance).await?;

            // This interception runs before generated dispatch, so derive the
            // instance only from the already-verified tenant in the envelope.
            let model_ref = load_data.model_ref.clone();
            let reservation = match self.reserve_intercepted_load(&instance).await? {
                InterceptedLoadAdmission::Loaded { reach } => {
                    let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                        LoadedModelResponse { model_ref, reach },
                    ))?;
                    return Ok((response, None));
                }
                InterceptedLoadAdmission::Pending => {
                    debug!("Model {} already being loaded, deduplicating", model_ref);
                    let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                        LoadedModelResponse { model_ref, reach: Vec::new() },
                    ))?;
                    return Ok((response, None));
                }
                InterceptedLoadAdmission::Accepted(reservation) => reservation,
            };

            // The accepted response is sent only after the shared gate has
            // reserved `pending_loads`. Transfer its attempt-scoped RAII guard
            // into the continuation *before* response signing/serialization:
            // if that fallible response path returns early, dropping this
            // continuation synchronously releases only this attempt's marker.
            info!("Load request accepted for {} (async)", model_ref);
            let response_model_ref = model_ref.clone();
            let service = self.clone(); // Arc clone — cheap, 'static
            let (load_max_context, load_kv_quant) = load_data.to_load_params();
            let continuation: crate::services::Continuation = Box::pin(async move {
                let model_name = model_ref.split(':').next().unwrap_or(&model_ref);
                let scope = format!("serve:model:{}", model_name);
                match service
                    .run_intercepted_load(
                        &instance,
                        load_max_context,
                        load_kv_quant,
                        reservation,
                    )
                    .await
                {
                    Ok(endpoint) => {
                        info!("Model {} loaded successfully at {}", model_ref, endpoint);
                        let event = crate::events::EventEnvelope::new(
                            crate::events::EventSource::Model,
                            scope.clone(),
                            crate::events::EventPayload::ModelLoaded {
                                model_ref: model_ref.clone(),
                                endpoint,
                            },
                        );
                        if let Ok(payload) = serde_json::to_vec(&event) {
                            let _event_publish_gate = service.lifecycle_event_publish_gate.lock().await;
                            let _ = service.event_publisher.publish("lifecycle", "loaded", &payload).await;
                            debug!("Published model.loaded event");
                        }
                    }
                    Err(e) => {
                        warn!("Model {} failed to load: {}", model_ref, e);
                        let event = crate::events::EventEnvelope::new(
                            crate::events::EventSource::Model,
                            scope.clone(),
                            crate::events::EventPayload::ModelFailed {
                                model_ref: model_ref.clone(),
                                error: e.to_string(),
                            },
                        );
                        if let Ok(payload) = serde_json::to_vec(&event) {
                            let _event_publish_gate = service.lifecycle_event_publish_gate.lock().await;
                            let _ = service.event_publisher.publish("lifecycle", "failed", &payload).await;
                        }
                    }
                }
            });

            let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                LoadedModelResponse { model_ref: response_model_ref, reach: Vec::new() },
            ))?;

            return Ok((response, Some(continuation)));
        }

        dispatch_model(self, ctx, body).await
    }

    fn name(&self) -> &str {
        "model"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn jwt_key_source(&self) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        self.inner.jwt_key_source.clone()
    }

    fn producer_reach_config_handle(
        &self,
    ) -> Option<hyprstream_rpc::moq_stream::ProducerReachConfigHandle> {
        Some(Arc::clone(&self.producer_reach_config))
    }

    fn moq_origin_handle(
        &self,
    ) -> Option<hyprstream_rpc::moq_stream::MoqStreamOriginHandle> {
        Some(Arc::clone(&self.moq_origin))
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = ModelResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// Helper types
// ============================================================================

// ModelStatusEntry, OnlineTrainingConfigInfo, ModelStatusInfo deleted — use generated types directly:
// - GenModelStatusEntry = generated::model_client::ModelStatusEntry
// - GenOnlineTrainingConfig = generated::model_client::OnlineTrainingConfig
// - ModelStatusResponse = generated::model_client::ModelStatusResponse (infer-scoped)

// ModelZmqClient removed — use generated ModelClient directly.
// Convenience methods (load/unload/status/infer_stream) are now inlined at call sites.



#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "postgres")]
    #[test]
    fn federate_model_route_is_exact_generate_stream_and_model_ref() -> Result<()> {
        let bytes = hyprstream_rpc::serialize_message(|message| {
            let mut request = message.init_root::<hyprstream_rpc_std::model_capnp::model_request::Builder>();
            request.set_id(1);
            let mut infer = request.init_infer();
            infer.set_model_ref("qwen2.5-0.5b-instruct:main");
            infer.init_generate_stream();
        })?;
        let body = crate::services::generated::model_client::decode_model_request_body(&bytes)?;
        assert_eq!(federate_model_operation(&body)?,
            ("model:qwen2.5-0.5b-instruct:main".into(), "infer".into()));

        let bytes = hyprstream_rpc::serialize_message(|message| {
            let mut request = message.init_root::<hyprstream_rpc_std::model_capnp::model_request::Builder>();
            request.set_id(2);
            let mut infer = request.init_infer();
            infer.set_model_ref("qwen2.5-0.5b-instruct:main");
            infer.set_status(());
        })?;
        let body = crate::services::generated::model_client::decode_model_request_body(&bytes)?;
        assert!(federate_model_operation(&body).is_err());

        let bytes = hyprstream_rpc::serialize_message(|message| {
            let mut request = message.init_root::<hyprstream_rpc_std::model_capnp::model_request::Builder>();
            request.set_id(3);
            request.init_load().set_model_ref("qwen2.5-0.5b-instruct:main");
        })?;
        let body = crate::services::generated::model_client::decode_model_request_body(&bytes)?;
        assert!(federate_model_operation(&body).is_err());
        Ok(())
    }

    #[test]
    fn test_config_defaults() {
        let config = ModelServiceConfig::default();
        assert_eq!(config.max_models, 5);
        assert_eq!(config.max_context, None);
        assert_eq!(config.kv_quant, KVQuantType::None);
        assert_eq!(
            config.inference_deployment.isolation,
            InferenceIsolationProfile::InProcess
        );
    }

    #[test]
    fn staging_pin_rejects_wrong_name_and_ref() -> Result<()> {
        let oid = git2::Oid::from_str("18c562db6830c2ef6636b8eaf42fb479272052f7")?;
        let pin = StagingModelPin::new("qwen2.5-0.5b-instruct:main", oid)?;
        assert!(pin.admits("qwen2.5-0.5b-instruct:main"));
        for wrong in ["other-model:main", "qwen2.5-0.5b-instruct:other", "qwen2.5-0.5b-instruct"] {
            assert!(!pin.admits(wrong));
        }
        assert!(StagingModelPin::new("qwen2.5-0.5b-instruct:refs/heads/main", oid).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn denied_model_load_is_audited_before_pin_or_registry_io() -> Result<()> {
        use tracing::field::{Field, Visit};
        use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

        #[derive(Clone, Default, Debug)]
        struct AuditObservation {
            target: String,
            resource: String,
            action: String,
            decision: String,
        }

        #[derive(Default)]
        struct AuditVisitor(AuditObservation);

        impl Visit for AuditVisitor {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.record_str(field, format!("{value:?}").trim_matches('"'));
            }

            fn record_str(&mut self, field: &Field, value: &str) {
                match field.name() {
                    "resource" => self.0.resource = value.to_owned(),
                    "action" => self.0.action = value.to_owned(),
                    "decision" => self.0.decision = value.to_owned(),
                    _ => {}
                }
            }
        }

        #[derive(Clone)]
        struct AuditLayer(Arc<parking_lot::Mutex<Vec<AuditObservation>>>);

        impl<S: tracing::Subscriber> Layer<S> for AuditLayer {
            fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
                if event.metadata().target() != "audit" {
                    return;
                }
                let mut visitor = AuditVisitor::default();
                event.record(&mut visitor);
                visitor.0.target = event.metadata().target().to_owned();
                self.0.lock().push(visitor.0);
            }
        }

        let registry_calls = Arc::new(BoundaryDialState::default());
        let policy_calls = Arc::new(BoundaryDialState::default());
        let registry_rpc = Arc::new(BoundaryRpcClient::new(
            TransportConfig::inproc("pin-admission-registry"),
            Arc::clone(&registry_calls),
        ));
        let policy_rpc = Arc::new(BoundaryRpcClient::new(
            TransportConfig::inproc("pin-admission-policy"),
            Arc::clone(&policy_calls),
        ));
        let mut config = ModelServiceConfig::default();
        config.staging_model_pin = Some(StagingModelPin::new(
            "qwen2.5-0.5b-instruct:main",
            git2::Oid::from_str("18c562db6830c2ef6636b8eaf42fb479272052f7")?,
        )?);
        let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
            hyprstream_rpc::moq_event::MoqEventOrigin::new(),
        );
        let service = ModelService::new(
            config,
            SigningKey::from_bytes(&[0x45; 32]),
            PolicyClient::new(policy_rpc),
            RegistryClient::new(registry_rpc),
            TransportConfig::inproc("pin-admission-model"),
            TransportConfig::inproc("pin-admission-policy-transport"),
        )
        .await?;

        let context = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            hyprstream_rpc::envelope::Subject::new("unauthorized-caller"),
            "pin-admission-tenant",
            SigningKey::from_bytes(&[0x46; 32]).verifying_key(),
        );
        let records = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::Registry::default()
            .with(AuditLayer(Arc::clone(&records)));
        tracing::subscriber::with_default(subscriber, || -> Result<()> {
            for model_ref in ["other-model:main", "qwen2.5-0.5b-instruct:main"] {
                let bytes = hyprstream_rpc::serialize_message(|message| {
                    let mut request = message
                        .init_root::<hyprstream_rpc_std::model_capnp::model_request::Builder>();
                    request.set_id(91);
                    request.init_load().set_model_ref(model_ref);
                })?;
                let body =
                    crate::services::generated::model_client::decode_model_request_body(&bytes)?;
                let error = match
                    futures::executor::block_on(service.handle_request(&context, &body))
                {
                    Err(error) => error,
                    Ok(_) => anyhow::bail!(
                        "missing MAC clearance must deny the load before pin/ref status handling"
                    ),
                };
                anyhow::ensure!(
                    error
                        .to_string()
                        .contains("authorization denied: inference caller has no verified MAC clearance"),
                    "denial must be authorization-specific, not a pin or registry error: {error}"
                );
                anyhow::ensure!(
                    !error.to_string().contains("configured staging modelRef"),
                    "unauthorized callers must not receive pin-specific diagnostics"
                );
            }
            Ok(())
        })?;

        let observations = records.lock().clone();
        anyhow::ensure!(
            observations.len() == 2,
            "each denied load must emit one authz audit event: {observations:?}"
        );
        for observation in observations {
            anyhow::ensure!(
                observation.target == "audit",
                "authorization denial must use the audit target"
            );
            anyhow::ensure!(
                observation.resource == "model:Load",
                "unexpected audited resource: {observation:?}"
            );
            anyhow::ensure!(
                observation.action == "write",
                "unexpected audited action: {observation:?}"
            );
            anyhow::ensure!(
                observation.decision == "deny",
                "denied caller must be audited as denied: {observation:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn staging_pin_settings_fail_closed_on_missing_or_partial_override() -> Result<()> {
        let pin = StagingModelPin::from_settings("staging", None, None)?
            .ok_or_else(|| anyhow!("staging profile must have reviewed pin"))?;
        assert!(pin.admits("qwen2.5-0.5b-instruct:main"));
        assert!(StagingModelPin::from_settings("staging", Some("qwen2.5-0.5b-instruct:main"), None).is_err());
        assert!(StagingModelPin::from_settings("staging", None, Some("18c562db6830c2ef6636b8eaf42fb479272052f7")).is_err());
        assert!(StagingModelPin::from_settings("staging", Some("other:main"), Some("18c562db6830c2ef6636b8eaf42fb479272052f7")).is_err());
        assert!(StagingModelPin::from_settings("default", None, None)?.is_none());
        assert!(StagingModelPin::from_settings("default", Some("x:main"), None).is_err());
        assert!(StagingModelPin::from_settings("default", Some("x:main"), Some("deadbeef")).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn unpublished_spawn_guard_stops_worker_after_forced_failure() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let stopped = temp.path().join("worker-stopped");
        let witness = stopped.clone();
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = std::thread::spawn(move || {
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return; };
            runtime.block_on(worker_shutdown.notified());
            let _ = std::fs::write(witness, b"stopped");
        });
        let handle = hyprstream_service::SpawnedService::thread(
            "model-pin-guard-test".to_owned(), Some(worker), shutdown, None,
        );
        let result: Result<()> = async {
            let _guard = SpawnPublicationGuard::new(handle);
            anyhow::bail!("forced post-spawn pre-publication failure")
        }.await;
        assert!(result.is_err());
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !stopped.exists() { tokio::time::sleep(std::time::Duration::from_millis(10)).await; }
        }).await?;
        assert_eq!(std::fs::read(stopped)?, b"stopped");
        Ok(())
    }

    #[test]
    fn inference_instance_denies_unlabeled_tenant_before_load() {
        let signer = SigningKey::from_bytes(&[44u8; 32]).verifying_key();
        let ctx = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            hyprstream_rpc::envelope::Subject::new("alice"),
            "did-hosted-account",
            signer,
        );
        assert!(
            ModelService::inference_instance(&ctx, "tiny-llama:main").is_err(),
            "an authority-bound tenant without MAC clearance must not reach engine loading"
        );
    }

    // ========================================================================
    // #649 — latched load terminal (EV7 host-side retain)
    //
    // `ModelLoadTerminal` + `ModelLoadKey` latch directly into a
    // `TerminalStore`; these tests prove the per-load-attempt / monotonic-
    // write-once contract without constructing a full `ModelService` (which
    // needs a registry + transport). The wiring (`allocate_load_epoch` +
    // `latch_load_terminal` called from `load_model`) is thin glue over these
    // primitives.
    // ========================================================================

    fn loaded_term(endpoint: &str) -> Terminal<ModelLoadTerminal> {
        Terminal {
            value: ModelLoadTerminal::Loaded { endpoint: endpoint.to_owned() },
            latched_by: "model-service".to_owned(),
        }
    }

    fn failed_term(err: &str) -> Terminal<ModelLoadTerminal> {
        Terminal {
            value: ModelLoadTerminal::LoadFailed { error: err.to_owned() },
            latched_by: "model-service".to_owned(),
        }
    }

    // Read back a latched terminal without `.unwrap()` (clippy denies it here).
    fn latched_value(
        store: &TerminalStore<ModelLoadKey, ModelLoadTerminal>,
        key: &ModelLoadKey,
    ) -> ModelLoadTerminal {
        match store.get(key) {
            Some(t) => t.value,
            None => panic!("expected a terminal latched for {key:?}"),
        }
    }

    /// Monotonic write-once within a single load attempt (single epoch): a
    /// second latch on the same key is rejected and the first terminal wins.
    #[test]
    fn load_terminal_is_monotonic_write_once_per_epoch() {
        let store: TerminalStore<ModelLoadKey, ModelLoadTerminal> = TerminalStore::new();
        let key = ModelLoadKey {
            tenant: "tenant-a".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 1,
        };
        assert!(store.latch(key.clone(), loaded_term("ep1")));
        // A late failure for the SAME attempt cannot overwrite the real Loaded.
        assert!(!store.latch(key.clone(), failed_term("race")));
        assert_eq!(
            latched_value(&store, &key),
            ModelLoadTerminal::Loaded { endpoint: "ep1".to_owned() }
        );
    }

    /// Per-load-attempt keying: a reload allocates a fresh epoch and latches a
    /// NEW terminal — the EV7 "reload = new latch" contract. Without per-epoch
    /// keying the first Loaded would shadow every subsequent reload.
    #[test]
    fn reload_latches_new_terminal_under_new_epoch() {
        let store: TerminalStore<ModelLoadKey, ModelLoadTerminal> = TerminalStore::new();
        // First load attempt (epoch 1): Loaded.
        let k1 = ModelLoadKey {
            tenant: "tenant-a".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 1,
        };
        assert!(store.latch(k1.clone(), loaded_term("ep1")));
        // Reload (epoch 2): a distinct key, so it latches a fresh terminal.
        let k2 = ModelLoadKey {
            tenant: "tenant-a".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 2,
        };
        assert!(store.latch(k2.clone(), loaded_term("ep2")));
        // Both attempts' terminals are retained (monotonic per-epoch).
        assert_eq!(
            latched_value(&store, &k1),
            ModelLoadTerminal::Loaded { endpoint: "ep1".to_owned() }
        );
        assert_eq!(
            latched_value(&store, &k2),
            ModelLoadTerminal::Loaded { endpoint: "ep2".to_owned() }
        );
    }

    /// A failed attempt latches `LoadFailed` under its epoch; a later
    /// successful reload (new epoch) latches `Loaded`. The failure is retained,
    /// not overwritten — each attempt has its own authoritative terminal.
    #[test]
    fn failed_then_reload_latches_distinct_terminals() {
        let store: TerminalStore<ModelLoadKey, ModelLoadTerminal> = TerminalStore::new();
        let kf = ModelLoadKey {
            tenant: "tenant-a".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 1,
        };
        assert!(store.latch(kf.clone(), failed_term("oom")));
        let ks = ModelLoadKey {
            tenant: "tenant-a".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 2,
        };
        assert!(store.latch(ks.clone(), loaded_term("ep2")));
        assert!(matches!(latched_value(&store, &kf), ModelLoadTerminal::LoadFailed { .. }));
        assert!(matches!(latched_value(&store, &ks), ModelLoadTerminal::Loaded { .. }));
    }

    #[test]
    fn load_terminals_are_tenant_scoped() {
        let store: TerminalStore<ModelLoadKey, ModelLoadTerminal> = TerminalStore::new();
        let tenant_a = ModelLoadKey {
            tenant: "tenant-a".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 1,
        };
        let tenant_b = ModelLoadKey {
            tenant: "tenant-b".to_owned(),
            model_ref: "qwen3:main".to_owned(),
            epoch: 1,
        };
        assert!(store.latch(tenant_a.clone(), loaded_term("a")));
        assert!(store.latch(tenant_b.clone(), loaded_term("b")));
        assert_ne!(
            latched_value(&store, &tenant_a),
            latched_value(&store, &tenant_b)
        );
    }

    #[test]
    fn in_process_profile_never_admits_two_tenants() {
        let admitted = std::sync::OnceLock::new();
        assert!(admit_in_process_tenant(&admitted, "tenant-a").is_ok());
        assert!(admit_in_process_tenant(&admitted, "tenant-a").is_ok());
        assert!(admit_in_process_tenant(&admitted, "tenant-b").is_err());
    }

    /// #320: the co-located InferenceService resolves (via the local Resolver
    /// registry) to an `Inproc` transport — the same arm the spawner registers in
    /// the in-process dial registry and the router single-selects.
    #[test]
    fn inference_transport_is_inproc_co_located() {
        // Default registry mode is Inproc; idempotent if another test inited it.
        hyprstream_rpc::registry::init(hyprstream_rpc::registry::EndpointMode::Inproc, None);
        let instance = InferenceInstanceId::new("tenant-a", "qwen3-small:main", 0)
            .unwrap_or_else(|e| panic!("valid instance: {e}"));
        let t = ModelService::inference_transport(&instance);
        match &t.endpoint {
            hyprstream_rpc::transport::EndpointType::Inproc { endpoint } => {
                assert!(
                    endpoint.contains("inference-"),
                    "opaque deterministic per-tenant/model inproc name, got {endpoint}"
                );
            }
            other => panic!("expected Inproc co-located transport, got {other:?}"),
        }
    }

    /// #320: a co-located (`Inproc`) service publishes an EMPTY wire reach list —
    /// same-host endpoints are never advertised; the co-located caller uses the
    /// in-process fast path. (This is what the router's single-select consumes:
    /// empty list ⇒ co-located fast path.)
    #[test]
    fn model_reach_co_located_is_empty() {
        let inproc = TransportConfig::inproc("hyprstream/inference-x");
        assert!(
            ModelService::model_reach(&inproc).is_empty(),
            "co-located Inproc reach must not be wire-advertised"
        );
    }

    /// #320: a networked (Iroh) inference reach maps to exactly ONE wire arm —
    /// the seam the router single-selects an Iroh reach from when no co-located
    /// fast path is present. The carrier target (`nodeId`) is preserved only for
    /// routing; it does not grant an application role.
    #[test]
    fn model_reach_networked_iroh_single_select() {
        let node_id = [0xEEu8; 32];
        let iroh = TransportConfig::iroh(node_id, Vec::new(), Some("https://relay.example".to_owned()));
        let reach = ModelService::model_reach(&iroh);
        assert_eq!(reach.len(), 1, "single-select: exactly one reach per service");
        match &reach[0] {
            WireTransportConfig::Iroh(i) => assert_eq!(i.node_id, node_id),
            other => panic!("expected wire Iroh reach, got {other:?}"),
        }
    }

    // ---- #395 ModelRef prefix-dispatch grammar ----

    #[test]
    fn model_ref_dispatch_at_uri_is_federated() {
        // at:// → federated arm, scheme captured, prefix stripped from `rest`.
        assert_eq!(
            model_ref_dispatch("at://did:plc:x123/hyprstream.models/qwen3/v1"),
            ModelRefDispatch::Federated {
                scheme: "at://",
                rest: "did:plc:x123/hyprstream.models/qwen3/v1",
            }
        );
    }

    #[test]
    fn model_ref_dispatch_bare_did_is_federated() {
        // did: → federated arm (bare-DID form). Note `did:` (no `//`) is matched
        // literally, and `at://` is NOT — `did:plc:...` must not be confused with
        // an at-uri.
        assert_eq!(
            model_ref_dispatch("did:plc:abcdef"),
            ModelRefDispatch::Federated {
                scheme: "did:",
                rest: "plc:abcdef",
            }
        );
        // did:web variant too.
        assert_eq!(
            model_ref_dispatch("did:web:hyprstream.example.com"),
            ModelRefDispatch::Federated {
                scheme: "did:",
                rest: "web:hyprstream.example.com",
            }
        );
    }

    #[test]
    fn model_ref_dispatch_local_name_is_local() {
        // Bare name, no federated prefix → local ModelRef arm.
        assert_eq!(model_ref_dispatch("qwen3"), ModelRefDispatch::Local);
        // name:ref — still local (the colon does not make it federated).
        assert_eq!(model_ref_dispatch("qwen3:main"), ModelRefDispatch::Local);
        assert_eq!(
            model_ref_dispatch("Qwen3-0.6B:tags/v1.0"),
            ModelRefDispatch::Local
        );
        // HuggingFace-style repo id (slash, no colon).
        assert_eq!(
            model_ref_dispatch("org/model-name"),
            ModelRefDispatch::Local
        );
    }

    #[test]
    fn model_ref_dispatch_uuid_is_local() {
        // UUID (backwards-compat) has no federated prefix → local arm.
        assert_eq!(
            model_ref_dispatch("550e8400-e29b-41d4-a716-446655440000"),
            ModelRefDispatch::Local
        );
    }

    #[test]
    fn model_ref_dispatch_local_arms_parse_unchanged() {
        // Every `Local` dispatch must parse via ModelRef::parse exactly as it did
        // pre-#395 — backward compatibility contract.
        for s in &["qwen3", "qwen3:main", "Qwen3-0.6B:tags/v1.0"] {
            assert_eq!(model_ref_dispatch(s), ModelRefDispatch::Local);
            assert!(
                ModelRef::parse(s).is_ok(),
                "local ModelRef '{s}' must still parse after #395"
            );
        }
    }

    // ========================================================================
    // #523 P3 — route_decision (HRW/least-loaded + session affinity ranking)
    //
    // `route_decision` is pure (no `InferenceClient`/`SpawnedService`), so
    // these tests build a `CellRouter` + synthetic `load_state` directly,
    // proving the ranking substrate without needing a live `LoadedModel`.
    // The `CellRouter`/`InferenceServerInfo` ranking machinery itself is
    // already exhaustively covered in `services::router::tests`; these tests
    // cover `route_decision`'s CoLocated/Remote/NoHealthyCandidate
    // classification specifically, which is new in this PR.
    // ========================================================================

    use crate::services::router::{CellRouter, InferenceServerInfo};

    fn server(id: u8, mem_free: u64, active: u64) -> InferenceServerInfo {
        InferenceServerInfo {
            replica_id: crate::services::router::ReplicaId::from_bytes([id; 32]),
            transport: TransportConfig::inproc("test"),
            gpu_memory_free: mem_free,
            active_sessions: active,
            last_heartbeat: Instant::now(),
        }
    }

    const SELECTOR_BEARER: &str = "fixture-delegated-bearer";
    static SELECTOR_BOUNDARY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    static SELECTOR_DIAL_STATE: std::sync::OnceLock<Arc<BoundaryDialState>> =
        std::sync::OnceLock::new();
    static SELECTOR_FIXTURE: std::sync::OnceLock<hyprstream_discovery::ProductionInferenceFixture> =
        std::sync::OnceLock::new();

    #[test]
    fn holder_bound_service_credentials_are_not_relayed_to_remote_replicas() {
        let service = hyprstream_rpc::auth::Claims::new(
            "service:registry".to_owned(),
            0,
            9_999_999_999,
        );
        let user = hyprstream_rpc::auth::Claims::new("alice".to_owned(), 0, 9_999_999_999);

        assert_eq!(
            ModelService::remote_relay_bearer_for_claims(Some(&service), Some("service-jwt")),
            None,
            "a service credential is holder-bound and must not be delegated to another controller"
        );
        assert_eq!(
            ModelService::remote_relay_bearer_for_claims(Some(&user), Some("user-jwt")),
            Some("user-jwt"),
            "an authenticated end-user bearer remains relayable"
        );
        assert_eq!(
            ModelService::remote_relay_bearer_for_claims(None, None),
            None,
            "an unauthenticated caller cannot select a remote relay path"
        );
    }

    #[derive(Clone, Debug)]
    struct ReadinessObservation {
        transport: TransportConfig,
        service: String,
        method_discriminator: u16,
        delegated_bearer: Option<String>,
        jwt: Option<String>,
    }

    #[derive(Default)]
    struct BoundaryDialBehavior {
        failures_remaining: usize,
        fail_all: bool,
    }

    #[derive(Default)]
    struct BoundaryDialState {
        dials: parking_lot::Mutex<Vec<TransportConfig>>,
        readiness: parking_lot::Mutex<Vec<ReadinessObservation>>,
        expected_bearer: parking_lot::Mutex<Option<String>>,
        expected_jwt: parking_lot::Mutex<Option<String>>,
        behavior: parking_lot::Mutex<BoundaryDialBehavior>,
    }

    impl BoundaryDialState {
        fn reset(&self, expected_bearer: Option<&str>) {
            self.dials.lock().clear();
            self.readiness.lock().clear();
            *self.expected_bearer.lock() = expected_bearer.map(ToOwned::to_owned);
            *self.behavior.lock() = BoundaryDialBehavior::default();
        }

        fn fail_next(&self, count: usize) {
            self.behavior.lock().failures_remaining = count;
        }

        fn fail_all(&self) {
            self.behavior.lock().fail_all = true;
        }
    }

    struct BoundaryRpcClient {
        transport: TransportConfig,
        state: Arc<BoundaryDialState>,
        request_id: std::sync::atomic::AtomicU64,
    }

    impl BoundaryRpcClient {
        fn new(transport: TransportConfig, state: Arc<BoundaryDialState>) -> Self {
            Self {
                transport,
                state,
                request_id: std::sync::atomic::AtomicU64::new(1),
            }
        }

        fn unsupported() -> Result<Vec<u8>> {
            anyhow::bail!("selector fixture received an unexpected RPC method")
        }
    }

    #[async_trait]
    impl hyprstream_rpc::RpcClient for BoundaryRpcClient {
        async fn call(&self, _payload: Vec<u8>) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_for_service(
            &self,
            _service_domain: &str,
            _payload: Vec<u8>,
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_for_service_with_method(
            &self,
            _service_domain: &str,
            _method_discriminator: u16,
            _payload: Vec<u8>,
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_with_options(
            &self,
            _payload: Vec<u8>,
            _options: hyprstream_rpc::CallOptions,
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_with_options_for_service(
            &self,
            _service_domain: &str,
            _payload: Vec<u8>,
            _options: hyprstream_rpc::CallOptions,
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_with_options_for_service_with_method(
            &self,
            service_domain: &str,
            method_discriminator: u16,
            payload: Vec<u8>,
            options: hyprstream_rpc::CallOptions,
        ) -> Result<Vec<u8>> {
            let reader = capnp::serialize::read_message(
                &mut std::io::Cursor::new(&payload),
                capnp::message::ReaderOptions::new(),
            )?;
            let request = reader.get_root::<hyprstream_rpc_std::inference_capnp::inference_request::Reader>()?;
            anyhow::ensure!(
                matches!(
                    request.which()?,
                    hyprstream_rpc_std::inference_capnp::inference_request::Which::IsReady(())
                ),
                "selector fixture received a non-readiness request"
            );
            let expected_bearer = self.state.expected_bearer.lock().clone();
            let expected_jwt = self.state.expected_jwt.lock().clone();
            self.state.readiness.lock().push(ReadinessObservation {
                transport: self.transport.clone(),
                service: service_domain.to_owned(),
                method_discriminator,
                delegated_bearer: options.delegated_bearer.clone(),
                jwt: options.jwt.clone(),
            });
            anyhow::ensure!(
                service_domain == InferenceClient::SERVICE_NAME,
                "readiness used the wrong generated service authority"
            );
            anyhow::ensure!(
                method_discriminator == 2,
                "readiness used the wrong schema method discriminator"
            );
            anyhow::ensure!(
                options.jwt == expected_jwt
                    && options.delegated_bearer == expected_bearer,
                "readiness did not carry the expected bearer shape"
            );
            let should_fail = {
                let mut behavior = self.state.behavior.lock();
                if behavior.fail_all {
                    true
                } else if behavior.failures_remaining > 0 {
                    behavior.failures_remaining -= 1;
                    true
                } else {
                    false
                }
            };
            if should_fail {
                anyhow::bail!("fixture readiness failure");
            }
            crate::services::generated::inference_client::serialize_response(
                request.get_id(),
                &hyprstream_rpc_std::inference_client::InferenceResponseVariant::IsReadyResult(
                    true,
                ),
            )
        }

        async fn call_streaming(
            &self,
            _payload: Vec<u8>,
            _ephemeral_pubkey: [u8; 32],
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_streaming_for_service(
            &self,
            _service_domain: &str,
            _payload: Vec<u8>,
            _ephemeral_pubkey: [u8; 32],
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn call_streaming_for_service_with_method(
            &self,
            _service_domain: &str,
            _method_discriminator: u16,
            _payload: Vec<u8>,
            _ephemeral_pubkey: [u8; 32],
        ) -> Result<Vec<u8>> {
            Self::unsupported()
        }

        async fn open_stream(
            &self,
            _payload: Vec<u8>,
        ) -> Result<Box<dyn hyprstream_rpc::stream_consumer::StreamHandle>> {
            anyhow::bail!("selector fixture received an unexpected streaming RPC")
        }

        async fn open_stream_from_info(
            &self,
            _stream_info: hyprstream_rpc::stream_info::StreamInfo,
            _client_secret: [u8; 32],
            _client_pubkey: [u8; 32],
        ) -> Result<Box<dyn hyprstream_rpc::stream_consumer::StreamHandle>> {
            anyhow::bail!("selector fixture received an unexpected stream-open RPC")
        }

        fn next_id(&self) -> u64 {
            self.request_id.fetch_add(1, Ordering::Relaxed)
        }
    }

    fn selector_instance() -> InferenceInstanceId {
        InferenceInstanceId::new("fixture-tenant", "fixture-model:main", 0)
            .unwrap_or_else(|error| panic!("fixture inference identity failed: {error}"))
    }

    fn remote_transport(id: u8) -> TransportConfig {
        TransportConfig::iroh([id; 32], Vec::new(), None)
    }

    fn selector_fixture() -> (
        &'static hyprstream_discovery::ProductionInferenceFixture,
        &'static Arc<BoundaryDialState>,
    ) {
        let state = SELECTOR_DIAL_STATE.get_or_init(|| Arc::new(BoundaryDialState::default()));
        let fixture = SELECTOR_FIXTURE.get_or_init(|| {
            let dial_state = Arc::clone(state);
            let dial = Arc::new(move |transport: &TransportConfig| {
                dial_state.dials.lock().push(transport.clone());
                Ok(Arc::new(BoundaryRpcClient::new(
                    transport.clone(),
                    Arc::clone(&dial_state),
                )) as Arc<dyn hyprstream_rpc::RpcClient>)
            });
            hyprstream_discovery::install_production_inference_fixture(
                &selector_instance().service_name(),
                &[remote_transport(0x01)],
                dial,
            )
            .unwrap_or_else(|error| panic!("install selector fixture failed: {error}"))
        });
        (fixture, state)
    }

    async fn selector_model_service() -> ModelService {
        let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
            hyprstream_rpc::moq_event::MoqEventOrigin::new(),
        );
        let infrastructure_state = Arc::new(BoundaryDialState::default());
        let infrastructure_rpc = Arc::new(BoundaryRpcClient::new(
            TransportConfig::inproc("selector-infrastructure"),
            infrastructure_state,
        )) as Arc<dyn hyprstream_rpc::RpcClient>;
        ModelService::new(
            ModelServiceConfig::default(),
            SigningKey::from_bytes(&[0x41; 32]),
            PolicyClient::new(Arc::clone(&infrastructure_rpc)),
            RegistryClient::new(infrastructure_rpc),
            TransportConfig::inproc("selector-model-service"),
            TransportConfig::inproc("policy"),
        )
        .await
        .unwrap_or_else(|error| panic!("construct model-free selector service failed: {error}"))
    }

    fn selector_loaded_model(
        load_state: Vec<InferenceServerInfo>,
        local_state: Arc<BoundaryDialState>,
    ) -> LoadedModel {
        let local_transport = TransportConfig::inproc("selector-local-inference");
        let network_transport = load_state
            .iter()
            .find(|candidate| {
                matches!(
                    &candidate.transport.endpoint,
                    EndpointType::Quic { .. } | EndpointType::Iroh { .. }
                )
            })
            .map_or_else(
                || local_transport.clone(),
                |candidate| candidate.transport.clone(),
            );
        LoadedModel {
            instance: selector_instance(),
            model_ref: "fixture-model:main".to_owned(),
            pinned_commit: None,
            incarnation: "fixture-incarnation-0000".to_owned(),
            work_audience: crate::services::inference::internal_work_audience(
                &selector_instance().service_name(),
                "fixture-incarnation-0000",
            ),
            load_generation: 1,
            transport: local_transport.clone(),
            network_transport,
            service_handle: hyprstream_service::SpawnedService::dummy(),
            client: InferenceClient::new(Arc::new(BoundaryRpcClient::new(
                local_transport,
                local_state,
            ))),
            router: CellRouter::default(),
            load_state,
            loaded_at: Instant::now(),
            last_used: Instant::now(),
            ttt_config: None,
            generation_defaults: crate::config::SamplingParams::default(),
        }
    }

    fn server_at(id: u8, transport: TransportConfig) -> InferenceServerInfo {
        InferenceServerInfo {
            replica_id: crate::services::router::ReplicaId::from_bytes([id; 32]),
            transport,
            gpu_memory_free: 1024,
            active_sessions: 0,
            last_heartbeat: Instant::now(),
        }
    }

    fn exclude_local(model: &mut LoadedModel) -> crate::services::router::ReplicaId {
        let local = ModelService::co_located_replica_id(model);
        model.router.report_dial_fail(local, Instant::now());
        local
    }

    #[tokio::test]
    async fn selector_boundary_fixed_authority_exact_reach_and_delegated_readiness() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        let advertised = remote_transport(0x11);
        fixture
            .reset(std::slice::from_ref(&advertised))
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        let local_state = Arc::new(BoundaryDialState::default());
        let local = server_at(0x10, TransportConfig::inproc("selector-local"));
        let remote = server_at(0x20, advertised.clone());
        let mut model = selector_loaded_model(vec![local, remote], local_state);
        exclude_local(&mut model);
        let service = selector_model_service().await;

        service
            .select_inference_routed(&mut model, "selector-positive", Some(SELECTOR_BEARER))
            .await
            .unwrap_or_else(|error| panic!("production selector rejected exact reach: {error}"));

        let dials = dial_state.dials.lock().clone();
        assert_eq!(dials, vec![advertised.clone()]);
        let readiness = dial_state.readiness.lock().clone();
        assert_eq!(readiness.len(), 1);
        assert_eq!(readiness[0].transport, advertised);
        assert_eq!(readiness[0].service, "inference");
        assert_eq!(readiness[0].method_discriminator, 2);
        assert_eq!(
            readiness[0].delegated_bearer.as_deref(),
            Some(SELECTOR_BEARER)
        );
        assert!(readiness[0].jwt.is_none());
        assert_ne!(selector_instance().service_name(), "inference");
    }

    #[tokio::test]
    async fn selector_boundary_rejects_unadvertised_reach_before_bearer_disclosure() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        fixture
            .reset(&[remote_transport(0x21)])
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        let local_state = Arc::new(BoundaryDialState::default());
        let selected_id = crate::services::router::ReplicaId::from_bytes([0x22; 32]);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x22, remote_transport(0x22)),
            ],
            local_state,
        );
        exclude_local(&mut model);
        let error = selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-unadvertised", Some(SELECTOR_BEARER))
            .await
            .err()
            .unwrap_or_else(|| panic!("unadvertised reach was accepted"));
        assert!(error
            .to_string()
            .contains("not a current authorized candidate"));
        assert!(dial_state.dials.lock().is_empty());
        assert!(dial_state.readiness.lock().is_empty());
        assert!(model
            .router
            .is_excluded(selected_id, Instant::now())
            .is_some());
    }

    #[tokio::test]
    async fn selector_boundary_rejects_stale_reach_before_bearer_disclosure() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        let advertised = remote_transport(0x31);
        fixture
            .reset(std::slice::from_ref(&advertised))
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        fixture.mark_stale();
        dial_state.reset(Some(SELECTOR_BEARER));
        let selected_id = crate::services::router::ReplicaId::from_bytes([0x23; 32]);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x23, advertised),
            ],
            Arc::new(BoundaryDialState::default()),
        );
        exclude_local(&mut model);
        assert!(selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-stale", Some(SELECTOR_BEARER))
            .await
            .is_err());
        assert!(dial_state.dials.lock().is_empty());
        assert!(dial_state.readiness.lock().is_empty());
        assert!(model
            .router
            .is_excluded(selected_id, Instant::now())
            .is_some());
    }

    #[tokio::test]
    async fn selector_boundary_rejects_cross_authority_retry_set_before_bearer_disclosure() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        let selected = remote_transport(0x41);
        fixture
            .reset(std::slice::from_ref(&selected))
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        fixture
            .add_foreign_authority(&remote_transport(0x42))
            .unwrap_or_else(|error| panic!("add foreign fixture authority failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        let selected_id = crate::services::router::ReplicaId::from_bytes([0x24; 32]);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x24, selected),
            ],
            Arc::new(BoundaryDialState::default()),
        );
        exclude_local(&mut model);
        let error = selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-authority", Some(SELECTOR_BEARER))
            .await
            .err()
            .unwrap_or_else(|| panic!("cross-authority retry set was accepted"));
        assert!(error
            .to_string()
            .contains("ambiguous validated service authority"));
        assert!(dial_state.dials.lock().is_empty());
        assert!(dial_state.readiness.lock().is_empty());
        assert!(model
            .router
            .is_excluded(selected_id, Instant::now())
            .is_some());
    }

    #[tokio::test]
    async fn selector_boundary_rejects_non_network_transport_before_bearer_disclosure() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        fixture
            .reset(&[remote_transport(0x51)])
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        let selected_id = crate::services::router::ReplicaId::from_bytes([0x25; 32]);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x25, TransportConfig::ipc("/run/hyprstream/attacker.sock")),
            ],
            Arc::new(BoundaryDialState::default()),
        );
        exclude_local(&mut model);
        let error = selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-transport", Some(SELECTOR_BEARER))
            .await
            .err()
            .unwrap_or_else(|| panic!("non-network remote transport was accepted"));
        assert!(error.to_string().contains("non-network transport"));
        assert!(dial_state.dials.lock().is_empty());
        assert!(dial_state.readiness.lock().is_empty());
        assert!(model
            .router
            .is_excluded(selected_id, Instant::now())
            .is_some());
    }

    #[tokio::test]
    async fn selector_boundary_failure_marks_exact_selected_replica_and_reselects_once() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        let first = remote_transport(0x61);
        let second = remote_transport(0x62);
        fixture
            .reset(&[first.clone(), second.clone()])
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        dial_state.fail_next(1);
        let first_id = crate::services::router::ReplicaId::from_bytes([0x26; 32]);
        let second_id = crate::services::router::ReplicaId::from_bytes([0x27; 32]);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x26, first.clone()),
                server_at(0x27, second.clone()),
            ],
            Arc::new(BoundaryDialState::default()),
        );
        exclude_local(&mut model);
        selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-reselect", Some(SELECTOR_BEARER))
            .await
            .unwrap_or_else(|error| panic!("bounded reselection did not recover: {error}"));
        let dials = dial_state.dials.lock().clone();
        assert_eq!(dials.len(), 2);
        assert_ne!(dials[0], dials[1]);
        let (failed_id, surviving_id) = if dials[0] == first {
            (first_id, second_id)
        } else {
            assert_eq!(dials[0], second);
            (second_id, first_id)
        };
        assert!(model
            .router
            .is_excluded(failed_id, Instant::now())
            .is_some());
        assert!(model
            .router
            .is_excluded(surviving_id, Instant::now())
            .is_none());
        assert_eq!(dial_state.readiness.lock().len(), 2);
    }

    #[tokio::test]
    async fn selector_boundary_explicit_colocation_returns_local_client_without_remote_dial() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        fixture
            .reset(&[remote_transport(0x71)])
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        let local_state = Arc::new(BoundaryDialState::default());
        local_state.reset(None);
        let mut model = selector_loaded_model(
            vec![server_at(0x10, TransportConfig::inproc("selector-local"))],
            Arc::clone(&local_state),
        );
        let client = selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-local", Some(SELECTOR_BEARER))
            .await
            .unwrap_or_else(|error| panic!("explicit co-location failed: {error}"))
            .value;
        assert!(dial_state.dials.lock().is_empty());
        assert!(dial_state.readiness.lock().is_empty());
        assert!(client
            .is_ready()
            .await
            .unwrap_or_else(|error| panic!("returned local client failed: {error}")));
        assert_eq!(local_state.readiness.lock().len(), 1);
    }

    #[tokio::test]
    async fn selector_boundary_exhaustion_fails_without_local_fallback() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        let first = remote_transport(0x81);
        let second = remote_transport(0x82);
        fixture
            .reset(&[first.clone(), second.clone()])
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        dial_state.fail_all();
        let first_id = crate::services::router::ReplicaId::from_bytes([0x28; 32]);
        let second_id = crate::services::router::ReplicaId::from_bytes([0x29; 32]);
        let local_state = Arc::new(BoundaryDialState::default());
        local_state.reset(None);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x28, first),
                server_at(0x29, second),
            ],
            Arc::clone(&local_state),
        );
        exclude_local(&mut model);
        let error = selector_model_service()
            .await
            .select_inference_routed(&mut model, "selector-exhausted", Some(SELECTOR_BEARER))
            .await
            .err()
            .unwrap_or_else(|| panic!("exhausted remote candidates fell back locally"));
        assert!(error.to_string().contains("candidates exhausted"));
        assert_eq!(dial_state.dials.lock().len(), 2);
        assert_eq!(dial_state.readiness.lock().len(), 2);
        assert!(local_state.readiness.lock().is_empty());
        assert!(model.router.is_excluded(first_id, Instant::now()).is_some());
        assert!(model
            .router
            .is_excluded(second_id, Instant::now())
            .is_some());
    }

    /// Degenerate/solo case (acceptance criterion): a single-entry replica set
    /// (today's actual `load_state` shape at load time) always classifies as
    /// `CoLocated` — no regression from pre-P3 behavior.
    #[test]
    fn route_decision_single_replica_is_co_located() {
        let co_located = crate::services::router::ReplicaId::from_bytes([0x10; 32]);
        let load_state = vec![server(0x10, 1024, 0)];
        let mut router = CellRouter::default();
        let decision = ModelService::route_decision(
            &mut router,
            &load_state,
            co_located,
            "sess-A",
            Instant::now(),
        );
        assert_eq!(decision, RouteDecision::CoLocated);
    }

    /// Multi-replica (synthetic `load_state`, per the PR notes on the P1
    /// candidate-population gap): HRW must be able to select a NON-co-located
    /// entry, classified as `Remote`, distributing distinct sessions across
    /// more than one node — the acceptance criterion's "requests distribute
    /// by HRW/least-loaded over live candidates" proven at this layer.
    #[test]
    fn route_decision_multi_replica_distributes_and_detects_remote() {
        let co_located = crate::services::router::ReplicaId::from_bytes([0x10; 32]);
        let load_state = vec![
            server(0x10, 16 * 1024 * 1024 * 1024, 0), // co-located
            server(0x20, 16 * 1024 * 1024 * 1024, 0), // remote
            server(0x30, 16 * 1024 * 1024 * 1024, 0), // remote
        ];
        let mut saw_co_located = false;
        let mut saw_remote = false;
        for i in 0..150u32 {
            // Fresh router per session avoids affinity stickiness masking the
            // underlying HRW spread this test wants to observe.
            let mut router = CellRouter::default();
            match ModelService::route_decision(
                &mut router,
                &load_state,
                co_located,
                &format!("sess-{i}"),
                Instant::now(),
            ) {
                RouteDecision::CoLocated => saw_co_located = true,
                RouteDecision::Remote(_) => saw_remote = true,
                RouteDecision::NoHealthyCandidate => panic!("healthy replica set must place"),
            }
        }
        assert!(saw_co_located, "some sessions must land on the co-located replica");
        assert!(saw_remote, "some sessions must land on a remote replica (not always co-located)");
    }

    /// Session affinity holds: the same session_id must classify identically
    /// across repeated calls on the SAME router instance (KV-cache stickiness),
    /// whether it lands co-located or remote.
    #[test]
    fn route_decision_session_affinity_holds() {
        let co_located = crate::services::router::ReplicaId::from_bytes([0x10; 32]);
        let load_state = vec![
            server(0x10, 16 * 1024 * 1024 * 1024, 0),
            server(0x20, 16 * 1024 * 1024 * 1024, 0),
            server(0x30, 16 * 1024 * 1024 * 1024, 0),
        ];
        let mut router = CellRouter::default();
        let now = Instant::now();
        let first = ModelService::route_decision(&mut router, &load_state, co_located, "sess-sticky", now);
        for _ in 0..5 {
            let again = ModelService::route_decision(&mut router, &load_state, co_located, "sess-sticky", now);
            assert_eq!(first, again, "same session must stick to the same replica within its lease");
        }
    }

    /// Fail-over on replica loss: once the affinity-bound node is marked down
    /// (dial-fail), the NEXT `route_decision` for the same session must
    /// reassign to a different node rather than returning the excluded one.
    #[test]
    fn route_decision_fails_over_on_replica_loss() {
        let co_located = crate::services::router::ReplicaId::from_bytes([0x10; 32]);
        let load_state = vec![
            server(0x10, 16 * 1024 * 1024 * 1024, 0),
            server(0x20, 16 * 1024 * 1024 * 1024, 0),
            server(0x30, 16 * 1024 * 1024 * 1024, 0),
        ];
        let mut router = CellRouter::default();
        let now = Instant::now();
        let first = ModelService::route_decision(&mut router, &load_state, co_located, "sess-B", now);
        let first_node = match first {
            RouteDecision::CoLocated => co_located,
            RouteDecision::Remote(n) => n,
            RouteDecision::NoHealthyCandidate => panic!("healthy replica set must place"),
        };
        router.report_dial_fail(first_node, now);
        let second = ModelService::route_decision(&mut router, &load_state, co_located, "sess-B", now);
        let second_node = match second {
            RouteDecision::CoLocated => co_located,
            RouteDecision::Remote(n) => n,
            RouteDecision::NoHealthyCandidate => panic!("two healthy replicas remain after one failure"),
        };
        assert_ne!(second_node, first_node, "must fail over off the down replica");
    }

    /// Empty/all-excluded replica set classifies as `NoHealthyCandidate` (the
    /// caller's fallback-to-co-located-client path), never a panic.
    #[test]
    fn route_decision_empty_replica_set_is_no_healthy_candidate() {
        let co_located = crate::services::router::ReplicaId::from_bytes([0x10; 32]);
        let load_state: Vec<InferenceServerInfo> = vec![];
        let mut router = CellRouter::default();
        let decision = ModelService::route_decision(
            &mut router,
            &load_state,
            co_located,
            "sess-A",
            Instant::now(),
        );
        assert_eq!(decision, RouteDecision::NoHealthyCandidate);
    }

    #[tokio::test]
    async fn selector_boundary_colocated_client_carries_internal_work_bearer_not_relay() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let local_state = Arc::new(BoundaryDialState::default());
        // The co-located instance expects the minted work order as the DIRECT
        // jwt — never as a delegated relay of a caller credential.
        let work_token = "iw-minted-work-order-fixture".to_owned();
        *local_state.expected_jwt.lock() = Some(work_token.clone());
        local_state.reset(None);
        let mut model = selector_loaded_model(
            vec![server_at(0x10, TransportConfig::inproc("selector-local"))],
            Arc::clone(&local_state),
        );
        let service = selector_model_service().await;
        let routed = service
            .select_inference_routed(&mut model, "selector-internal", None)
            .await
            .unwrap_or_else(|error| panic!("co-location without a relay bearer failed: {error}"));
        // The co-located arm carries the work order as its DIRECT bearer via
        // the same locality gate get_inference_client uses (K3 finding M1).
        let bearer_client =
            ModelService::attach_work_order(routed, work_token.clone());
        assert!(
            bearer_client.is_ready().await
                .unwrap_or_else(|error| panic!("returned local client failed: {error}")),
            "readiness with the internal work bearer must succeed"
        );
        let readiness = local_state.readiness.lock().clone();
        assert_eq!(readiness.len(), 1);
        assert_eq!(readiness[0].jwt.as_deref(), Some(work_token.as_str()));
        assert!(
            readiness[0].delegated_bearer.is_none(),
            "the internal leg must never relay a caller credential as delegated bearer"
        );
    }

    /// K3 finding M1 regression: the REMOTE arm must never carry the internal
    /// work order. The routed client keeps its delegated-relay shape; the
    /// locality gate must not overlay a direct JWT (an envelope carrying both
    /// is a hard wire error, and a remote instance would deny a
    /// foreign-controller work order anyway).
    #[tokio::test]
    async fn selector_boundary_remote_arm_never_overlays_internal_work_order() {
        let _guard = SELECTOR_BOUNDARY_LOCK.lock().await;
        let (fixture, dial_state) = selector_fixture();
        let advertised = remote_transport(0x91);
        fixture
            .reset(std::slice::from_ref(&advertised))
            .unwrap_or_else(|error| panic!("reset selector fixture failed: {error}"));
        dial_state.reset(Some(SELECTOR_BEARER));
        // The remote relay must present the delegated bearer and NO direct
        // jwt on its first forwarded call.
        *dial_state.expected_jwt.lock() = None;
        let local_state = Arc::new(BoundaryDialState::default());
        local_state.reset(None);
        let mut model = selector_loaded_model(
            vec![
                server_at(0x10, TransportConfig::inproc("selector-local")),
                server_at(0x91, advertised.clone()),
            ],
            Arc::clone(&local_state),
        );
        exclude_local(&mut model);
        let service = selector_model_service().await;
        let routed = service
            .select_inference_routed(&mut model, "selector-remote-internal", Some(SELECTOR_BEARER))
            .await
            .unwrap_or_else(|error| panic!("remote selection failed: {error}"));
        // The same locality gate get_inference_client applies: Remote ⇒ no
        // work-order overlay, the delegated relay shape is untouched.
        let client = ModelService::attach_work_order(routed, "iw-foreign-controller-order".to_owned());
        assert!(
            client.is_ready().await
                .unwrap_or_else(|error| panic!("remote client failed: {error}")),
            "remote relay readiness must succeed with the delegated bearer only"
        );
        // Two observations: the remote arm's selection readiness probe AND the
        // post-attach forwarded call. BOTH must carry the delegated relay and
        // never the work order.
        let readiness = dial_state.readiness.lock().clone();
        assert_eq!(readiness.len(), 2);
        for observation in &readiness {
            assert_eq!(
                observation.delegated_bearer.as_deref(),
                Some(SELECTOR_BEARER),
                "remote relay must carry the delegated bearer"
            );
            assert!(
                observation.jwt.is_none(),
                "the remote arm must not carry the internal work order as a direct jwt"
            );
        }
    }


    // ========================================================================
    // User-approved single-service Model→Inference boundary (D5): mint side.
    //
    // Proves the work order Model mints per forwarded operation is bound to
    // the pinned instance, the verified tenant/model, the exact downstream
    // dispatch coordinate, the caller's own expiry, and Model's key — and
    // that it verifies against the PINNED controller key the spawned
    // instance enforces.
    // ========================================================================

    use base64::Engine as _;
    use hyprstream_rpc::auth::internal_work::{
        verify_internal_work, InternalWorkScope, MAX_LIFETIME_SECS,
    };

    const BOUNDARY_CALLER_KEY: [u8; 32] = [0xCB; 32];

    /// The exact live worker audience a mint must bind (Sol incarnation plan):
    /// the versioned audience of the fixture instance's test incarnation.
    fn audience_fixture() -> String {
        crate::services::inference::internal_work_audience(
            &selector_instance().service_name(),
            "test-incarnation-0000",
        )
    }

    // ── Sol bounded incarnation plan: live binding + dead-worker fail-close ─

    fn running_worker() -> hyprstream_service::SpawnedService {
        // A live parked thread keeps the handle "running" for the guard; the
        // parked thread exits when the test process ends (no join needed).
        let handle = std::thread::spawn(|| {
            loop {
                std::thread::park_timeout(std::time::Duration::from_secs(3600));
            }
        });
        hyprstream_service::SpawnedService::thread(
            "s1-alive-worker".to_owned(),
            Some(handle),
            Arc::new(tokio::sync::Notify::new()),
            None,
        )
    }

    fn dead_worker() -> hyprstream_service::SpawnedService {
        hyprstream_service::SpawnedService::thread(
            "s1-dead-worker".to_owned(),
            None,
            Arc::new(tokio::sync::Notify::new()),
            None,
        )
    }

    fn blocking_unload_worker(
        armed: std::sync::mpsc::Sender<()>,
        shutdown_observed: tokio::sync::oneshot::Sender<()>,
        release: Arc<tokio::sync::Notify>,
    ) -> hyprstream_service::SpawnedService {
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let worker_shutdown = Arc::clone(&shutdown);
        let handle = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap_or_else(|error| panic!("build blocking-unload runtime: {error}"));
            runtime.block_on(async move {
                let notified = worker_shutdown.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                armed
                    .send(())
                    .unwrap_or_else(|error| panic!("signal blocking worker readiness: {error}"));
                notified.await;
                let _ = shutdown_observed.send(());
                release.notified().await;
            });
        });
        hyprstream_service::SpawnedService::thread(
            "s1-blocking-unload-worker".to_owned(),
            Some(handle),
            shutdown,
            None,
        )
    }

    fn panicking_unload_worker() -> hyprstream_service::SpawnedService {
        let handle = std::thread::spawn(|| {
            panic!("test worker panics before join")
        });
        hyprstream_service::SpawnedService::thread(
            "s1-panicking-unload-worker".to_owned(),
            Some(handle),
            Arc::new(tokio::sync::Notify::new()),
            None,
        )
    }

    #[tokio::test]
    async fn cancelled_unload_keeps_instance_reserved_until_thread_join() {
        let service = selector_model_service().await;
        let instance = selector_instance();
        let local_state = Arc::new(BoundaryDialState::default());
        let (armed_tx, armed_rx) = std::sync::mpsc::channel();
        let (shutdown_observed_tx, shutdown_observed_rx) = tokio::sync::oneshot::channel();
        let release = Arc::new(tokio::sync::Notify::new());
        let mut model = selector_loaded_model(Vec::new(), local_state);
        model.service_handle = blocking_unload_worker(
            armed_tx,
            shutdown_observed_tx,
            Arc::clone(&release),
        );
        service.loaded_models.write().await.put(instance.clone(), model);
        // This is the real inner-load/outer-load handoff overlap: a worker is
        // visible in the cache while the load reservation still exists. The
        // unloading worker below blocks its join so both status APIs must
        // prefer the reservation for the whole drain interval.
        service.pending_loads.lock().insert(instance.clone(), 1);

        tokio::task::spawn_blocking(move || {
            armed_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .map_err(|error| anyhow!("blocking worker did not arm: {error}"))
        })
        .await
        .unwrap_or_else(|error| panic!("wait for blocking worker readiness task: {error}"))
        .unwrap_or_else(|error| panic!("wait for blocking worker readiness: {error}"));

        let unload_service = service.clone();
        let unload_instance = instance.clone();
        let caller = tokio::spawn(async move {
            unload_service.unload_model(&unload_instance).await
        });

        tokio::time::timeout(std::time::Duration::from_secs(5), shutdown_observed_rx)
            .await
            .unwrap_or_else(|_| panic!("unload did not notify the blocking worker"))
            .unwrap_or_else(|error| panic!("blocking worker dropped shutdown observation: {error}"));

        caller.abort();
        assert!(
            matches!(caller.await, Err(error) if error.is_cancelled()),
            "the caller task must be cancelled while owned teardown continues"
        );
        assert!(
            service.unloading_models.lock().await.contains(&instance),
            "cancellation must retain the unloading reservation until join completes"
        );
        assert!(
            !service.loaded_models.read().await.contains(&instance),
            "the cache must not expose a worker after its shutdown begins"
        );
        let (all_status, single_status) = tokio::join!(
            service.model_status_all(instance.tenant()),
            service.model_status_single(&instance),
        );
        assert_eq!(
            all_status
                .iter()
                .filter(|entry| entry.model_ref == instance.model_ref())
                .map(|entry| entry.status.as_str())
                .collect::<Vec<_>>(),
            vec!["unloading"],
            "all-status must retain an unloading worker after its caller is cancelled"
        );
        assert_eq!(
            single_status
                .iter()
                .map(|entry| entry.status.as_str())
                .collect::<Vec<_>>(),
            vec!["unloading"],
            "single-status must not report the draining worker as absent"
        );
        let error = match service.load_model(&instance, None, None).await {
            Err(error) => error,
            Ok(_) => panic!("a replacement load must not race the draining worker"),
        };
        assert!(
            error.to_string().contains("is unloading"),
            "replacement denial must disclose only lifecycle state: {error:#}"
        );

        // In production the outer load completion clears this marker. Keep the
        // worker drain blocked but model that completion before asserting final
        // status absence below.
        service.pending_loads.lock().remove(&instance);
        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if !service.unloading_models.lock().await.contains(&instance) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("owned teardown did not clear its reservation after join"));
        let (all_status, single_status) = tokio::join!(
            service.model_status_all(instance.tenant()),
            service.model_status_single(&instance),
        );
        assert!(
            all_status.is_empty() && single_status.is_empty(),
            "status absence is permitted only after teardown releases its reservation"
        );
    }

    #[tokio::test]
    async fn unloading_reservation_precedes_loaded_and_pending_status_snapshot() {
        let service = selector_model_service().await;
        let instance = selector_instance();

        // `load_model_inner` publishes the cache entry before outer
        // `load_model` clears this reservation. If unload reserves and pops in
        // that interval, both sets are deliberately present: teardown is the
        // authoritative lifecycle state, never a second loading attempt.
        service.loaded_models.write().await.put(
            instance.clone(),
            selector_loaded_model(Vec::new(), Arc::new(BoundaryDialState::default())),
        );
        service.pending_loads.lock().insert(instance.clone(), 1);
        service.unloading_models.lock().await.insert(instance.clone());

        let (all_status, single_status) = tokio::join!(
            service.model_status_all(instance.tenant()),
            service.model_status_single(&instance),
        );
        assert_eq!(
            all_status
                .iter()
                .filter(|entry| entry.model_ref == instance.model_ref())
                .map(|entry| entry.status.as_str())
                .collect::<Vec<_>>(),
            vec!["unloading"],
            "list status must not let pending_loads conceal an active teardown"
        );
        assert_eq!(
            single_status
                .iter()
                .map(|entry| entry.status.as_str())
                .collect::<Vec<_>>(),
            vec!["unloading"],
            "single status must not let pending_loads conceal an active teardown"
        );
    }

    #[tokio::test]
    async fn intercepted_load_reservation_rejects_unload_before_continuation_runs() {
        let service = selector_model_service().await;
        let instance = selector_instance();

        // Force the former check-to-response gap: handler admission reserves a
        // pending load, then unload attempts to win before the continuation is
        // polled. Both transitions use `load_unload_gate`, so the unload must
        // fail closed and leave the accepted continuation as sole owner.
        let reservation = match service.reserve_intercepted_load(&instance).await {
            Ok(InterceptedLoadAdmission::Accepted(reservation)) => reservation,
            Ok(_) => panic!("fixture load must receive a new accepted reservation"),
            Err(error) => panic!("fixture load admission failed: {error:#}"),
        };
        let unload_error = match service.unload_model(&instance).await {
            Err(error) => error,
            Ok(()) => panic!("unload must not cancel an accepted pending load"),
        };
        assert!(
            unload_error.to_string().contains("is loading"),
            "unload must reject against the accepted pending reservation: {unload_error:#}"
        );
        assert!(
            service.pending_loads.lock().contains_key(&instance),
            "the continuation must retain ownership of the accepted reservation"
        );
        assert!(
            !service.unloading_models.lock().await.contains(&instance),
            "a rejected unload must not leave a false teardown reservation"
        );

        // The selector fixture fails normal model resolution immediately; this
        // drives the continuation's ordinary failure completion without
        // starting a worker and proves it releases its own reservation.
        assert!(
            service
                .run_intercepted_load(&instance, None, None, reservation)
                .await
                .is_err(),
            "fixture resolution must reach the continuation failure path"
        );
        assert!(
            !service.pending_loads.lock().contains_key(&instance),
            "continuation completion must release its accepted reservation"
        );
    }

    #[tokio::test]
    async fn dropped_intercepted_load_reservation_rolls_back_pending_marker() {
        let service = selector_model_service().await;
        let instance = selector_instance();

        // This is the response-serialization / unpolled-continuation failure
        // path: admission succeeded, the continuation owns its guard, but the
        // response path drops that continuation before its first poll. Drop
        // must synchronously remove the marker so neither status nor a later
        // admission sees a permanent false load.
        let reservation = match service.reserve_intercepted_load(&instance).await {
            Ok(InterceptedLoadAdmission::Accepted(reservation)) => reservation,
            Ok(_) => panic!("fixture load must receive a new accepted reservation"),
            Err(error) => panic!("fixture load admission failed: {error:#}"),
        };
        assert!(service.pending_loads.lock().contains_key(&instance));
        let unpolled: crate::services::Continuation = Box::pin(async move {
            let _reservation = reservation;
            std::future::pending::<()>().await;
        });
        drop(unpolled);
        assert!(
            !service.pending_loads.lock().contains_key(&instance),
            "dropping an unpolled accepted continuation must roll back pending"
        );

        // A late Drop from an earlier handoff must not erase a newer attempt
        // for the same instance. This is the attempt identity contract, not a
        // best-effort instance-wide cleanup.
        let stale = InterceptedLoadReservation::new(Arc::clone(&service.inner), instance.clone(), 7);
        service.pending_loads.lock().insert(instance.clone(), 8);
        drop(stale);
        assert_eq!(
            service.pending_loads.lock().get(&instance),
            Some(&8),
            "a stale guard must release only its own attempt marker"
        );
        service.pending_loads.lock().remove(&instance);
        let retry = service.reserve_intercepted_load(&instance).await;
        assert!(
            matches!(retry, Ok(InterceptedLoadAdmission::Accepted(_))),
            "a later load must not deduplicate against a dropped reservation"
        );
    }

    #[tokio::test]
    async fn successful_unload_releases_admission_before_unloaded_event() {
        let service = selector_model_service().await;
        let instance = selector_instance();
        service.loaded_models.write().await.put(
            instance.clone(),
            selector_loaded_model(Vec::new(), Arc::new(BoundaryDialState::default())),
        );
        let (before_publish_tx, before_publish_rx) = tokio::sync::oneshot::channel();
        *service.before_unloaded_event.lock().await = Some(before_publish_tx);

        let unload_service = service.clone();
        let unload_instance = instance.clone();
        let unload = tokio::spawn(async move { unload_service.unload_model(&unload_instance).await });
        before_publish_rx
            .await
            .unwrap_or_else(|error| panic!("unload did not reach event boundary: {error}"));

        assert!(
            !service.unloading_models.lock().await.contains(&instance),
            "successful join must release admission before model.unloaded is published"
        );
        let reload = service.reserve_intercepted_load(&instance).await;
        assert!(
            matches!(reload, Ok(InterceptedLoadAdmission::Accepted(_))),
            "an event-driven reload must be admitted after a successful join"
        );
        unload
            .await
            .unwrap_or_else(|error| panic!("unload task panicked: {error}"))
            .unwrap_or_else(|error| panic!("dummy worker unload failed: {error:#}"));
    }

    #[tokio::test]
    async fn unload_stop_error_retains_instance_reservation() {
        let service = selector_model_service().await;
        let instance = selector_instance();
        let local_state = Arc::new(BoundaryDialState::default());
        let mut model = selector_loaded_model(Vec::new(), local_state);
        model.service_handle = panicking_unload_worker();
        service.loaded_models.write().await.put(instance.clone(), model);

        let error = match service.unload_model(&instance).await {
            Err(error) => error,
            Ok(()) => panic!("a thread join panic must make unload fail"),
        };
        assert!(
            error.to_string().contains("failed to stop model"),
            "unload must return the stop failure: {error:#}"
        );
        assert!(
            service.unloading_models.lock().await.contains(&instance),
            "an ambiguous stop failure must retain the replacement-load reservation"
        );
        assert!(
            !service.loaded_models.read().await.contains(&instance),
            "a failed stop must not expose the popped worker as live"
        );
        let error = match service.load_model(&instance, None, None).await {
            Err(error) => error,
            Ok(_) => panic!("a failed stop must not admit a replacement worker"),
        };
        assert!(
            error.to_string().contains("is unloading"),
            "replacement denial must remain a lifecycle-only message: {error:#}"
        );
    }

    #[tokio::test]
    async fn dead_worker_cache_entry_cannot_mint_and_is_evicted() {
        let service = selector_model_service().await;
        let instance = selector_instance();
        let mut cache = service.loaded_models.write().await;
        cache.put(
            instance.clone(),
            LoadedModel {
                instance: instance.clone(),
                model_ref: instance.model_ref().to_owned(),
                pinned_commit: None,
                transport: TransportConfig::inproc("dead-worker-test"),
                network_transport: TransportConfig::inproc("dead-worker-test"),
                service_handle: dead_worker(),
                client: InferenceClient::new(Arc::new(BoundaryRpcClient::new(
                    TransportConfig::inproc("dead-worker-test"),
                    Arc::new(BoundaryDialState::default()),
                ))),
                incarnation: "dead-incarnation".to_owned(),
                work_audience: "iw1/dead/dead".to_owned(),
                load_generation: 1,
                router: CellRouter::default(),
                load_state: Vec::new(),
                loaded_at: Instant::now(),
                last_used: Instant::now(),
                ttt_config: None,
                generation_defaults: crate::config::SamplingParams::default(),
            },
        );
        drop(cache);

        let ctx = boundary_caller_ctx(chrono::Utc::now().timestamp() + 3600);
        let scope = InternalWorkScope::new("inference:GenerateStream", "infer");
        let error = match service
            .get_inference_client(instance.model_ref(), &ctx, scope)
            .await
        {
            Ok(_) => panic!("a dead worker cache entry must not mint work orders"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("not running"),
            "dead-worker denial must name the liveness guard: {error:#}"
        );
        // The dead entry was evicted: the cache no longer holds it.
        assert!(!service.loaded_models.read().await.contains(&instance));
    }

    #[test]
    fn running_worker_liveness_guard_passes() {
        // Positive control for the liveness guard itself: an entry whose
        // service handle reports running passes the guard.
        let worker = running_worker();
        assert!(worker.is_running(), "live thread fixture must report running");
        let dead = dead_worker();
        assert!(!dead.is_running(), "dead fixture must report not-running");
        drop(worker);
    }

    #[test]
    fn incarnation_handoff_validation_rejects_stale_or_mismatched() {
        // The model-side handoff validation ensures the reported handoff
        // matches THIS spawn attempt exactly. Exercised through the same
        // ensure-conditions the load path applies (generation/instance/
        // controller), via direct value checks mirroring model.rs load logic.
        let instance = selector_instance();
        let expected_generation: u64 = 3;
        let expected_controller = SigningKey::from_bytes(&[0x41; 32]).verifying_key().to_bytes();

        let handoff = crate::services::inference::IncarnationHandoff {
            instance_service_name: instance.service_name(),
            controller_pubkey: expected_controller,
            generation: expected_generation,
            incarnation: "aa11".to_owned(),
            audience: "iw1/x/aa11".to_owned(),
        };
        assert_eq!(handoff.generation, expected_generation);
        assert_eq!(handoff.instance_service_name, instance.service_name());
        assert_eq!(handoff.controller_pubkey, expected_controller);

        let stale = crate::services::inference::IncarnationHandoff {
            generation: expected_generation - 1,
            ..handoff.clone()
        };
        assert_ne!(stale.generation, expected_generation, "stale handoff");
    }

    fn boundary_caller_ctx(caller_exp: i64) -> EnvelopeContext {
        let caller_key = SigningKey::from_bytes(&BOUNDARY_CALLER_KEY);
        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, caller_exp)
            .with_tenant("fixture-tenant".to_owned())
            .with_cnf_jwk(caller_key.verifying_key().as_bytes())
            // Model's ingress MAC gate (inference_instance →
            // enforce_inference_mac) runs inside get_inference_client before
            // the worker binding is consulted; the caller must carry the same
            // clearance the real ingress path requires.
            .with_clearance(crate::services::inference::inference_object_label());
        EnvelopeContext::for_test_authenticated_subject_with_claims(
            hyprstream_rpc::envelope::Subject::new("alice"),
            "fixture-tenant",
            caller_key.verifying_key(),
            claims,
        )
    }

    fn restart_acceptance_caller_ctx(
        subject: &str,
        caller_key: &SigningKey,
        tenant: &str,
    ) -> EnvelopeContext {
        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new(subject.to_owned(), now, now + 300)
            .with_tenant(tenant.to_owned())
            .with_cnf_jwk(caller_key.verifying_key().as_bytes())
            .with_clearance(crate::services::inference::inference_object_label());
        EnvelopeContext::for_test_authenticated_subject_with_claims(
            hyprstream_rpc::envelope::Subject::new(subject),
            tenant,
            caller_key.verifying_key(),
            claims,
        )
    }

    fn resolve_gitdir_reference(
        reference_file: &std::path::Path,
        gitfile_format: bool,
    ) -> Result<std::path::PathBuf> {
        let contents = std::fs::read_to_string(reference_file)?;
        let reference = if gitfile_format {
            contents
                .strip_prefix("gitdir: ")
                .ok_or_else(|| anyhow!("fixture Git file is malformed"))?
        } else {
            contents.as_str()
        }
        .trim();
        anyhow::ensure!(
            !reference.is_empty() && !reference.contains(['\n', '\r']),
            "fixture Git link is malformed"
        );
        let path = std::path::PathBuf::from(reference);
        let resolved = if path.is_absolute() {
            path
        } else {
            reference_file
                .parent()
                .ok_or_else(|| anyhow!("fixture Git link has no parent"))?
                .join(path)
        };
        Ok(resolved.canonicalize()?)
    }

    fn validate_restart_fixture_worktree_link(
        repo: &std::path::Path,
        worktree: &std::path::Path,
    ) -> Result<()> {
        let admin_worktree = repo.join("worktrees").join("main");
        let expected_gitfile = worktree.join(".git");
        anyhow::ensure!(expected_gitfile.is_file(), "fixture main worktree has no Git file");
        anyhow::ensure!(
            resolve_gitdir_reference(&admin_worktree.join("gitdir"), false)?
                == expected_gitfile.canonicalize()?,
            "fixture repository main worktree does not name the isolated StoragePaths worktree"
        );
        anyhow::ensure!(
            resolve_gitdir_reference(&expected_gitfile, true)? == admin_worktree.canonicalize()?,
            "isolated StoragePaths worktree does not link back to fixture repository main"
        );
        Ok(())
    }

    /// A guest-provisioned, local-only fixture for the real restart acceptance
    /// test.  The repository must be bare and its linked `main` worktree must
    /// already be the XDG-isolated path that `StoragePaths` resolves.  The test
    /// never clones, downloads, or copies model weights.
    struct RestartAcceptanceFixture {
        model: ModelService,
        model_key: SigningKey,
        model_ref: String,
        instance: InferenceInstanceId,
        _auth: RestartAuthServices,
    }

    const RESTART_FIXTURE_ISSUER: &str = "http://127.0.0.1:6791";

    /// Shared by the guest acceptance and the focused dispatch regression.
    /// Own the spawned services for as long as either test uses their clients.
    struct RestartAuthServices {
        model_key: SigningKey,
        policy_key: SigningKey,
        registry_key: SigningKey,
        model_token: String,
        policy_endpoint: String,
        registry_endpoint: String,
        policy_transport: TransportConfig,
        jwt_source: Arc<hyprstream_rpc::auth::ClusterKeySource>,
        _registry_handle: hyprstream_service::SpawnedService,
        _policy_handle: hyprstream_service::SpawnedService,
        _registry_base: tempfile::TempDir,
        _policy_base: tempfile::TempDir,
        _credentials: tempfile::TempDir,
    }

    fn restart_acceptance_inproc_endpoint(kind: &str, tag: &str) -> (TransportConfig, String) {
        let transport = TransportConfig::inproc(format!("restart-acceptance-{kind}-{tag}"));
        let uri = transport.endpoint_string();
        (transport, uri)
    }

    fn install_restart_fixture_auth(keys: &[SigningKey]) -> Result<()> {
        use hyprstream_rpc::crypto::CryptoPolicy;
        use hyprstream_rpc::envelope::{EnvelopeVerifyConfig, KeyedPqTrustStore, ResponseVerifyConfig};

        let _ = hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
            hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(10_000),
        ));
        if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
            let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
                hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
            ));
        }
        let mut pq_store = KeyedPqTrustStore::new();
        for key in keys {
            let pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(key);
            let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
                &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
            )?;
            pq_store.bind(key.verifying_key().to_bytes(), &pq_vk);
        }
        let pq_store = Arc::new(pq_store);
        hyprstream_rpc::envelope::install_verify_config(EnvelopeVerifyConfig {
            policy: CryptoPolicy::Hybrid,
            pq_store: Some(pq_store.clone()),
        })?;
        hyprstream_rpc::envelope::install_response_verify_config(ResponseVerifyConfig {
            policy: CryptoPolicy::Hybrid,
            pq_store: Some(pq_store),
        })?;
        crate::mac::install_production_rpc_dispatch_pep()?;
        anyhow::ensure!(
            hyprstream_rpc::auth::mac::global_mac_dispatch_pep().is_some(),
            "restart fixture has no production dispatch PEP"
        );
        Ok(())
    }

    fn restart_fixture_service_jwt(
        credentials: &tempfile::TempDir,
        name: &str,
        ca: &SigningKey,
        key: &SigningKey,
    ) -> Result<String> {
        let bootstrap = crate::auth::identity_store::BootstrapPubkey::for_service_key(key)?;
        crate::auth::service_jwt::issue_or_load_service_jwt(
            credentials.path(),
            name,
            ca,
            &bootstrap,
            RESTART_FIXTURE_ISSUER,
            chrono::Utc::now().timestamp(),
            Some(&crate::mac::dispatch_labels::BOOTSTRAP_SERVICE_CLEARANCE),
        )
    }

    impl RestartAuthServices {
        async fn new(repo: &std::path::Path, model_name: &str) -> Result<Self> {
            use hyprstream_service::ServiceManager as _;

            let tag = hex::encode(hyprstream_rpc::envelope::generate_nonce());
            let policy_key = SigningKey::from_bytes(&[0x91; 32]);
            let registry_key = SigningKey::from_bytes(&[0x92; 32]);
            let model_key = SigningKey::from_bytes(&[0x93; 32]);
            // The fourth key is anchored only for the wrong-holder regression:
            // its denial must be a credential binding failure, not a missing PQ anchor.
            let wrong_holder = SigningKey::from_bytes(&[0x94; 32]);
            install_restart_fixture_auth(&[
                policy_key.clone(), registry_key.clone(), model_key.clone(), wrong_holder,
            ])?;
            let ca = hyprstream_rpc::node_identity::derive_purpose_key(&policy_key, "hyprstream-jwt-v1");
            let ca_pq = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&ca);
            let jwt_source = Arc::new(
                hyprstream_rpc::auth::ClusterKeySource::new(ca.verifying_key(), RESTART_FIXTURE_ISSUER.to_owned())
                    .with_ca_composite_key(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&ca_pq)),
            );
            let credentials = tempfile::TempDir::new()?;
            let model_token = restart_fixture_service_jwt(&credentials, "model", &ca, &model_key)?;
            let registry_token = restart_fixture_service_jwt(&credentials, "registry", &ca, &registry_key)?;
            for (key, name) in [(&policy_key, "policy"), (&registry_key, "registry")] {
                hyprstream_service::global_trust_store().insert(
                    key.verifying_key(),
                    hyprstream_service::Attestation {
                        scopes: std::iter::once(name.to_owned()).collect(),
                        subject: Some(format!("service:{name}")),
                        jwt: None,
                        expires_at: 0,
                        attested_by: None,
                    },
                );
            }

            let policy_base = tempfile::TempDir::new()?;
            let policy_manager = Arc::new(crate::auth::PolicyManager::permissive().await?);
            let (policy_transport, policy_endpoint) = restart_acceptance_inproc_endpoint("policy", &tag);
            let policy_service = crate::services::PolicyService::new(
                policy_manager,
                Arc::new(policy_key.clone()),
                crate::config::TokenConfig::default(),
                Arc::new(RwLock::new(git2db::Git2DB::open(policy_base.path()).await?)),
                policy_transport.clone(),
            ).with_jwt_key_source(jwt_source.clone());
            crate::services::restart_diag::phase("policy.spawn.enter");
            let manager = hyprstream_service::InprocManager::new();
            let policy_handle = manager.spawn(Box::new(policy_service)).await?;
            crate::services::restart_diag::phase("policy.spawn.done");

            let registry_base = tempfile::TempDir::new()?;
            {
                let mut registry_store = git2db::Git2DB::open(registry_base.path()).await?;
                registry_store.register(git2db::RepoId::new())
                    .name(model_name)
                    .worktree_path(repo)
                    .url(String::new())
                    .exec().await?;
            }
            let (registry_transport, registry_endpoint) = restart_acceptance_inproc_endpoint("registry", &tag);
            let registry_policy = PolicyClient::for_local_endpoint_bootstrap(
                &policy_endpoint, registry_key.clone(), policy_key.verifying_key(), Some(registry_token),
            )?;
            let registry_service = crate::services::RegistryService::new(
                registry_base.path(), registry_policy, registry_transport, registry_key.clone(),
            ).await?
                .with_expected_audience(RESTART_FIXTURE_ISSUER.to_owned())
                .with_jwt_key_source(jwt_source.clone());
            crate::services::restart_diag::phase("registry.spawn.enter");
            let registry_handle = manager.spawn(Box::new(registry_service)).await?;
            crate::services::restart_diag::phase("registry.spawn.done");
            Ok(Self {
                model_key, policy_key, registry_key, model_token,
                policy_endpoint, registry_endpoint, policy_transport, jwt_source,
                _registry_handle: registry_handle, _policy_handle: policy_handle,
                _registry_base: registry_base, _policy_base: policy_base, _credentials: credentials,
            })
        }

        fn model_registry_client(&self) -> Result<RegistryClient> {
            RegistryClient::for_local_endpoint_bootstrap(
                &self.registry_endpoint,
                self.model_key.clone(),
                self.registry_key.verifying_key(),
                Some(self.model_token.clone()),
            )
        }

        fn model_policy_client(&self) -> Result<PolicyClient> {
            PolicyClient::for_local_endpoint_bootstrap(
                &self.policy_endpoint,
                self.model_key.clone(),
                self.policy_key.verifying_key(),
                Some(self.model_token.clone()),
            )
        }
    }

    #[test]
    fn restart_acceptance_inproc_endpoints_match_client_dial_names() {
        for kind in ["policy", "registry"] {
            let (server, uri) = restart_acceptance_inproc_endpoint(kind, "fixture");
            assert_eq!(
                server.endpoint,
                TransportConfig::from_endpoint(&uri).endpoint,
                "{kind} server registration and client dial must use the same name"
            );
            assert_eq!(uri, format!("inproc://restart-acceptance-{kind}-fixture"));
        }
    }

    struct ClassicalRestartSigner(SigningKey);

    #[async_trait::async_trait]
    impl hyprstream_rpc::transport_traits::Signer for ClassicalRestartSigner {
        fn pubkey(&self) -> [u8; 32] {
            self.0.verifying_key().to_bytes()
        }

        async fn sign(&self, bytes: &[u8]) -> Result<[u8; 64]> {
            use ed25519_dalek::Signer as _;
            Ok(self.0.sign(bytes).to_bytes())
        }
    }

    fn assert_restart_dispatch_denied<T>(result: Result<T>, case: &str) -> Result<()> {
        let error = result.err().ok_or_else(|| anyhow!("{case} reached the Registry handler"))?;
        anyhow::ensure!(
            error.to_string().contains(hyprstream_rpc::service::dispatch::DISPATCH_DENIED),
            "{case} failed outside the uniform dispatch boundary: {error}"
        );
        Ok(())
    }

    /// Run by exact filter in its own test process: request/response verify
    /// configs are first-write globals, while the dispatch PEP is swappable.
    #[tokio::test]
    #[ignore = "requires first-write verify-config globals; run with --exact --test-threads=1"]
    async fn restart_fixture_authenticated_hybrid_get_by_name_requires_model_holder() -> Result<()> {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::transport::in_memory::InMemoryTransport;

        let repo = tempfile::TempDir::new()?;
        let model_name = "restart-auth-regression";
        let auth = RestartAuthServices::new(repo.path(), model_name).await?;
        let model = auth.model_registry_client()?;
        let policy_check = hyprstream_rpc_std::policy_client::PolicyCheck {
            subject: "service:model".to_owned(),
            domain: "*".to_owned(),
            resource: "registry:*".to_owned(),
            operation: "query".to_owned(),
        };
        anyhow::ensure!(auth.model_policy_client()?.check(&policy_check).await?,
            "model holder must reach real Policy dispatch");
        let found = model.get_by_name(model_name).await?;
        anyhow::ensure!(found.name == model_name,
            "the authenticated model lookup returned the wrong repository");
        anyhow::ensure!(model.list().await?.iter().any(|entry| entry.name == model_name),
            "Registry-to-Policy mediation lost the authenticated model caller");

        let client = |key: SigningKey, token: Option<String>| {
            RegistryClient::for_local_endpoint_bootstrap(
                &auth.registry_endpoint, key, auth.registry_key.verifying_key(), token,
            )
        };
        let missing = client(auth.model_key.clone(), None)?.get_by_name(model_name).await;
        assert_restart_dispatch_denied(missing, "tokenless model lookup")?;
        let wrong_holder = client(SigningKey::from_bytes(&[0x94; 32]), Some(auth.model_token.clone()))?
            .get_by_name(model_name).await;
        assert_restart_dispatch_denied(wrong_holder, "wrong holder with model token")?;

        let processor = hyprstream_rpc::dial::lookup_inproc(
            auth.registry_endpoint.strip_prefix("inproc://")
                .ok_or_else(|| anyhow!("fixture Registry endpoint is not in-process"))?,
        ).ok_or_else(|| anyhow!("fixture Registry handle was not retained"))?;
        let classical = RpcClientImpl::new(
            ClassicalRestartSigner(auth.model_key.clone()),
            InMemoryTransport::new(processor),
            Some(auth.registry_key.verifying_key()),
        ).with_default_jwt(auth.model_token.clone());
        let classical = RegistryClient::new(Arc::new(classical));
        let classical_error = classical.get_by_name(model_name).await
            .err().ok_or_else(|| anyhow!("classical signer reached Registry"))?;
        anyhow::ensure!(classical_error.to_string().contains("mandatory Hybrid suite requires an ML-DSA-65 signer key"),
            "classical signer failed for an unrelated reason: {classical_error}");

        // This signer has its own valid model certificate, but no PQ anchor in
        // the process verifier. Credential authority cannot fill that gap.
        let unanchored_key = SigningKey::from_bytes(&[0x95; 32]);
        let ca = hyprstream_rpc::node_identity::derive_purpose_key(&auth.policy_key, "hyprstream-jwt-v1");
        let unanchored_credentials = tempfile::TempDir::new()?;
        let unanchored_token = restart_fixture_service_jwt(&unanchored_credentials, "model", &ca, &unanchored_key)?;
        let unanchored = client(unanchored_key, Some(unanchored_token))?
            .get_by_name(model_name).await;
        let unanchored_error = unanchored.err()
            .ok_or_else(|| anyhow!("unanchored PQ signer reached Registry"))?;
        anyhow::ensure!(unanchored_error.to_string() == "registry envelope admission failed",
            "unanchored PQ signer failed outside envelope admission: {unanchored_error:#}");
        anyhow::ensure!(unanchored_error.chain().any(|cause| matches!(
            cause.downcast_ref::<hyprstream_rpc::EnvelopeError>(),
            Some(hyprstream_rpc::EnvelopeError::PqSignatureInvalid(reason))
                if reason == "mandatory Hybrid suite requires an anchored ML-DSA-65 signer key"
        )), "unanchored PQ signer failed for an unrelated admission reason: {unanchored_error:#}");
        Ok(())
    }

    impl RestartAcceptanceFixture {
        async fn from_guest_env() -> Result<Self> {
            let repo = std::path::PathBuf::from(std::env::var(
                "HYPRSTREAM_RESTART_ACCEPTANCE_MODEL_REPO",
            ).map_err(|_| anyhow!(
                "HYPRSTREAM_RESTART_ACCEPTANCE_MODEL_REPO must name the sealed local bare model repository"
            ))?);
            anyhow::ensure!(repo.is_dir(), "restart acceptance model repository is not a directory");
            let model_name = std::env::var("HYPRSTREAM_RESTART_ACCEPTANCE_MODEL_NAME")
                .unwrap_or_else(|_| "restart-acceptance-fixture".to_owned());
            let instance_name = std::env::var("HYPRSTREAM_INSTANCE").map_err(|_| anyhow!(
                "HYPRSTREAM_INSTANCE must be set to an isolated synthetic-guest namespace"
            ))?;
            anyhow::ensure!(
                instance_name.starts_with("restart-acceptance-"),
                "HYPRSTREAM_INSTANCE must use the restart-acceptance-* isolated namespace"
            );

            let storage = crate::storage::StoragePaths::new()?;
            let worktree = storage.worktree_path(&model_name, "main")?;
            anyhow::ensure!(worktree.is_dir(), "fixture main worktree is absent");
            anyhow::ensure!(
                worktree.join("config.json").is_file()
                    && worktree.join("tokenizer.json").is_file()
                    && std::fs::read_dir(&worktree)?.flatten().any(|entry| {
                        entry.file_name().to_string_lossy().ends_with(".safetensors")
                    }),
                "fixture must provide config.json, tokenizer.json, and safetensors weights"
            );
            validate_restart_fixture_worktree_link(&repo, &worktree)?;

            let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
                hyprstream_rpc::moq_event::MoqEventOrigin::new(),
            );
            hyprstream_rpc::registry::init(
                hyprstream_rpc::registry::EndpointMode::Inproc,
                None,
            );
            anyhow::ensure!(
                hyprstream_rpc::registry::global().mode()
                    == hyprstream_rpc::registry::EndpointMode::Inproc,
                "restart fixture endpoint registry must use Inproc mode"
            );
            let auth = RestartAuthServices::new(&repo, &model_name).await?;
            let mut config = ModelServiceConfig::default();
            config.inference_deployment.compute = InferenceCompute::Cpu;
            let model = ModelService::new(
                config,
                auth.model_key.clone(),
                auth.model_policy_client()?,
                auth.model_registry_client()?,
                TransportConfig::inproc(format!("restart-acceptance-model-{}", hex::encode(hyprstream_rpc::envelope::generate_nonce()))),
                auth.policy_transport.clone(),
            )
            .await?
            .with_expected_audience(RESTART_FIXTURE_ISSUER.to_owned())
            .with_jwt_key_source(auth.jwt_source.clone());
            let model_ref = format!("{model_name}:main");
            let instance = InferenceInstanceId::new("restart-acceptance-tenant", &model_ref, 0)?;
            Ok(Self {
                model,
                model_key: auth.model_key.clone(),
                model_ref,
                instance,
                _auth: auth,
            })
        }
    }

    #[test]
    fn restart_fixture_gitdir_backlink_requires_expected_checkout() -> Result<()> {
        fn git(args: &[&str]) -> Result<()> {
            let status = std::process::Command::new("git")
                .args(args)
                .status()
                .map_err(|error| anyhow!("temporary Git fixture command did not start: {error}"))?;
            anyhow::ensure!(
                status.success(),
                "temporary Git fixture command failed: git {args:?}"
            );
            Ok(())
        }

        let temp = tempfile::tempdir()?;
        let source = temp.path().join("source");
        let repo = temp.path().join("fixture.git");
        let expected_checkout = temp.path().join("isolated/models/worktrees/main");
        let other_checkout = temp.path().join("other-worktree");
        let source_arg = source
            .to_str()
            .ok_or_else(|| anyhow!("temporary source path is not UTF-8"))?;
        let repo_arg = repo
            .to_str()
            .ok_or_else(|| anyhow!("temporary bare repository path is not UTF-8"))?;
        let expected_arg = expected_checkout
            .to_str()
            .ok_or_else(|| anyhow!("temporary expected checkout path is not UTF-8"))?;
        let other_arg = other_checkout
            .to_str()
            .ok_or_else(|| anyhow!("temporary mismatched checkout path is not UTF-8"))?;

        git(&["init", "-b", "main", source_arg])?;
        git(&["-C", source_arg, "config", "user.name", "restart-fixture-test"])?;
        git(&[
            "-C",
            source_arg,
            "config",
            "user.email",
            "restart-fixture-test@example.invalid",
        ])?;
        std::fs::write(source.join("fixture.txt"), "fixture\n")?;
        git(&["-C", source_arg, "add", "fixture.txt"])?;
        git(&["-C", source_arg, "commit", "-m", "temporary fixture"])?;
        git(&["clone", "--bare", source_arg, repo_arg])?;
        std::fs::create_dir_all(
            expected_checkout
                .parent()
                .ok_or_else(|| anyhow!("temporary expected checkout has no parent"))?,
        )?;
        git(&["--git-dir", repo_arg, "worktree", "add", expected_arg, "main"])?;
        git(&[
            "--git-dir",
            repo_arg,
            "worktree",
            "add",
            "-b",
            "other",
            other_arg,
            "main",
        ])?;

        validate_restart_fixture_worktree_link(&repo, &expected_checkout)?;
        let mismatch = match validate_restart_fixture_worktree_link(&repo, &other_checkout) {
            Ok(()) => anyhow::bail!("a different linked checkout must be rejected"),
            Err(error) => error,
        };
        anyhow::ensure!(
            mismatch
                .to_string()
                .contains("does not name the isolated StoragePaths worktree"),
            "mismatched checkout error must identify the administrative backlink"
        );
        Ok(())
    }

    fn assert_restart_order_holder(
        order: &str,
        controller: &SigningKey,
        audience: &str,
        subject: &str,
        caller_key: &SigningKey,
    ) -> Result<()> {
        let claims = verify_internal_work(
            order,
            &controller.verifying_key(),
            audience,
            chrono::Utc::now().timestamp(),
        )?;
        anyhow::ensure!(claims.sub == subject && claims.caller.sub == subject);
        anyhow::ensure!(claims.tenant == "restart-acceptance-tenant");
        anyhow::ensure!(claims.resource == "inference:HasLora" && claims.operation == "query");
        anyhow::ensure!(
            claims.owner_did.as_deref()
                == Some(hyprstream_rpc::identity::Did::from_ed25519(
                    &caller_key.verifying_key().to_bytes(),
                ).as_str()),
            "work order owner DID must remain bound to the verified caller"
        );
        Ok(())
    }

    /// Test-only observer for the one server-side denial class that identifies
    /// an old worker order at its replacement.  It retains only a boolean: no
    /// bearer, subject, audience, request ID, or formatted event is kept.
    #[derive(Clone)]
    struct RestartAudienceDenialProbe {
        seen: Arc<std::sync::atomic::AtomicBool>,
    }

    impl RestartAudienceDenialProbe {
        fn install() -> Result<Arc<std::sync::atomic::AtomicBool>> {
            use tracing_subscriber::prelude::*;

            let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let subscriber = tracing_subscriber::registry().with(Self {
                seen: Arc::clone(&seen),
            });
            // The prescribed invocation is one ignored test in its own
            // process (`--exact --test-threads=1`). A global dispatcher is
            // necessary because the real worker dispatches on its own service
            // task; a thread-local subscriber would not observe that task.
            tracing::subscriber::set_global_default(subscriber).map_err(|_| {
                anyhow!(
                    "restart acceptance requires an isolated test process without a preinstalled tracing subscriber"
                )
            })?;
            Ok(seen)
        }
    }

    struct RestartAudienceDenialVisitor<'a> {
        seen: &'a std::sync::atomic::AtomicBool,
    }

    impl RestartAudienceDenialVisitor<'_> {
        fn record_message(&self, field: &tracing::field::Field, message: &str) {
            if field.name() == "message"
                && message.contains("claims verification failed")
                && message.contains("audience")
            {
                self.seen.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    impl tracing::field::Visit for RestartAudienceDenialVisitor<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.record_message(field, &format!("{value:?}"));
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.record_message(field, value);
        }
    }

    impl<S> tracing_subscriber::Layer<S> for RestartAudienceDenialProbe
    where
        S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _context: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if event.metadata().target() == "hyprstream_rpc::service::dispatch" {
                event.record(&mut RestartAudienceDenialVisitor { seen: &self.seen });
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "requires the sealed local CPU-model repository and isolated XDG guest fixture"]
    async fn real_cpu_worker_restart_rejects_old_order_and_preserves_holders() -> Result<()> {
        crate::services::restart_diag::enable();
        crate::services::restart_diag::phase("test.enter");
        let audience_denial = RestartAudienceDenialProbe::install()?;
        crate::services::restart_diag::phase("fixture.enter");
        let fixture = tokio::time::timeout(
            std::time::Duration::from_secs(180),
            RestartAcceptanceFixture::from_guest_env(),
        )
        .await
        .map_err(|_| anyhow!("RESTART_DIAG_PHASE_TIMEOUT=fixture"))??;
        crate::services::restart_diag::phase("fixture.done");
        let caller_a_key = SigningKey::from_bytes(&[0xA1; 32]);
        let caller_b_key = SigningKey::from_bytes(&[0xB1; 32]);
        let caller_a = restart_acceptance_caller_ctx(
            "restart-caller-a",
            &caller_a_key,
            "restart-acceptance-tenant",
        );
        let caller_b = restart_acceptance_caller_ctx(
            "restart-caller-b",
            &caller_b_key,
            "restart-acceptance-tenant",
        );
        let scope = ModelService::inference_scope("hasLora", "query");

        // A is a real CPU Model load, then the normal Model forwarding path.
        crate::services::restart_diag::phase("worker_a.load.enter");
        tokio::time::timeout(
            std::time::Duration::from_secs(600),
            fixture.model.load_model(&fixture.instance, None, None),
        )
        .await
        .map_err(|_| anyhow!("RESTART_DIAG_PHASE_TIMEOUT=worker_a.load"))??;
        crate::services::restart_diag::phase("worker_a.load.done");
        let (audience_a, incarnation_a) = {
            let cache = fixture.model.loaded_models.read().await;
            let loaded = cache.peek(&fixture.instance)
                .ok_or_else(|| anyhow!("worker A missing after successful load"))?;
            (loaded.work_audience.clone(), loaded.incarnation.clone())
        };
        let a_client = fixture.model.get_inference_client(&fixture.model_ref, &caller_a, scope.clone()).await?;
        let mut old_order = fixture.model.take_test_work_order().await
            .ok_or_else(|| anyhow!("test-only capture missed worker A work order"))?;
        assert_restart_order_holder(&old_order, &fixture.model_key, &audience_a, "restart-caller-a", &caller_a_key)?;
        crate::services::restart_diag::phase("worker_a.call.enter");
        let _ = a_client.has_lora().await?;
        crate::services::restart_diag::phase("worker_a.call.done");

        // Model's actual unload owns the cached SpawnedService stop/join. A
        // subsequent load creates B through the same production init closure.
        crate::services::restart_diag::phase("worker_a.unload.enter");
        fixture.model.unload_model(&fixture.instance).await?;
        crate::services::restart_diag::phase("worker_a.unload.done");
        anyhow::ensure!(
            !fixture.model.loaded_models.read().await.contains(&fixture.instance),
            "unload must remove worker A before B is created"
        );
        crate::services::restart_diag::phase("worker_b.load.enter");
        tokio::time::timeout(
            std::time::Duration::from_secs(600),
            fixture.model.load_model(&fixture.instance, None, None),
        )
        .await
        .map_err(|_| anyhow!("RESTART_DIAG_PHASE_TIMEOUT=worker_b.load"))??;
        crate::services::restart_diag::phase("worker_b.load.done");
        let (audience_b, incarnation_b, worker_b) = {
            let cache = fixture.model.loaded_models.read().await;
            let loaded = cache.peek(&fixture.instance)
                .ok_or_else(|| anyhow!("worker B missing after successful reload"))?;
            (loaded.work_audience.clone(), loaded.incarnation.clone(), loaded.client.clone())
        };
        anyhow::ensure!(audience_a != audience_b && incarnation_a != incarnation_b,
            "reload must install a fresh worker-generated incarnation");

        // This is the causal assertion: B receives A's real captured bearer
        // through generated dispatch. Its client response stays uniformly
        // opaque, while the in-process test probe records only the server-side
        // audience-denial class before the `hasLora` handler can succeed.
        crate::services::restart_diag::phase("old_order.denial.enter");
        let replay_error = match worker_b
            .with_bearer(std::mem::take(&mut *old_order))
            .has_lora()
            .await
        {
            Ok(_) => anyhow::bail!("worker B accepted worker A's order"),
            Err(error) => error,
        };
        anyhow::ensure!(
            replay_error.to_string() == hyprstream_rpc::service::dispatch::DISPATCH_DENIED,
            "old order must receive the uniform dispatch denial"
        );
        anyhow::ensure!(
            audience_denial.load(std::sync::atomic::Ordering::SeqCst),
            "replacement worker must classify the old order as an audience mismatch"
        );
        crate::services::restart_diag::phase("old_order.denial.done");

        // Fresh A work succeeds at B and carries the same original holder.
        let b_client_for_a = fixture.model.get_inference_client(&fixture.model_ref, &caller_a, scope.clone()).await?;
        let fresh_a_order = fixture.model.take_test_work_order().await
            .ok_or_else(|| anyhow!("test-only capture missed worker B fresh order"))?;
        assert_restart_order_holder(&fresh_a_order, &fixture.model_key, &audience_b, "restart-caller-a", &caller_a_key)?;
        crate::services::restart_diag::phase("fresh_a.call.enter");
        let _ = b_client_for_a.has_lora().await?;
        crate::services::restart_diag::phase("fresh_a.call.done");

        // Caller B gets a different holder-bound order; it cannot inherit A's
        // subject, caller snapshot, or pairwise owner DID.
        let b_client_for_b = fixture.model.get_inference_client(&fixture.model_ref, &caller_b, scope).await?;
        let fresh_b_order = fixture.model.take_test_work_order().await
            .ok_or_else(|| anyhow!("test-only capture missed caller B order"))?;
        assert_restart_order_holder(&fresh_b_order, &fixture.model_key, &audience_b, "restart-caller-b", &caller_b_key)?;
        anyhow::ensure!(fresh_a_order != fresh_b_order, "distinct caller requests require distinct work orders");
        crate::services::restart_diag::phase("fresh_b.call.enter");
        let _ = b_client_for_b.has_lora().await?;
        crate::services::restart_diag::phase("fresh_b.call.done");
        crate::services::restart_diag::phase("holders.done");

        crate::services::restart_diag::phase("worker_b.unload.enter");
        fixture.model.unload_model(&fixture.instance).await?;
        crate::services::restart_diag::phase("worker_b.unload.done");
        Ok(())
    }

    #[tokio::test]
    async fn internal_work_mint_binds_instance_tenant_model_operation_and_caller() {
        let service = selector_model_service().await;
        let instance = selector_instance();
        let scope = ModelService::inference_scope("generateStream", "infer");
        let caller_exp = chrono::Utc::now().timestamp() + 3600;
        let ctx = boundary_caller_ctx(caller_exp);
        let token = service
                                .mint_internal_work_token(
                        &instance,
                        &ctx,
                        &scope,
                        &audience_fixture(),
                    )
            .unwrap_or_else(|e| panic!("mint failed: {e}"));

        let model_key = SigningKey::from_bytes(&[0x41; 32]);
        let now = chrono::Utc::now().timestamp();
        let claims = verify_internal_work(
            &token,
            &model_key.verifying_key(),
            &audience_fixture(),
            now,
        )
        .unwrap_or_else(|e| panic!("minted order must verify against the pinned controller: {e}"));
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.tenant, "fixture-tenant");
        assert_eq!(claims.model, "fixture-model:main");
        assert_eq!(claims.resource, "inference:GenerateStream");
        assert_eq!(claims.operation, "infer");
        assert!(
            claims.exp <= caller_exp && claims.exp <= now + MAX_LIFETIME_SECS,
            "exp must be capped by the caller credential and the hard bound"
        );
        assert!(
            claims.exp > now,
            "a live caller credential must mint a live work order"
        );
        assert!(!claims.jti.is_empty());
        let expected_x = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(model_key.verifying_key().as_bytes());
        assert_eq!(
            claims.cnf.jwk.as_ref().map(|j| j.x.as_str()),
            Some(expected_x.as_str()),
            "cnf must bind the pinned controller key"
        );
        // The original caller's pairwise DID (from the verified caller envelope
        // key) rides for ledger attribution.
        let caller_key = SigningKey::from_bytes(&BOUNDARY_CALLER_KEY);
        assert_eq!(
            claims.owner_did.as_deref(),
            Some(
                hyprstream_rpc::identity::Did::from_ed25519(&caller_key.verifying_key().to_bytes())
                    .as_str()
            )
        );
        // The caller snapshot carries the verified claims WITHOUT the raw
        // bearer string (serde skip).
        assert_eq!(claims.caller.sub, "alice");
        assert!(claims.caller.token.is_none());
    }

    #[tokio::test]
    async fn internal_work_mint_requires_verified_caller_identity_and_live_expiry() {
        let service = selector_model_service().await;
        let instance = selector_instance();
        // Text-payload variant grammar: the payload IS the coordinate.
        let scope = InternalWorkScope::new("inference:probe/adapters/k3", "write");

        // No verified caller claims (e.g. a bearer-less service caller): never
        // minted — the deny outcome matches the pre-boundary behavior.
        let bare = EnvelopeContext::for_test_authenticated_subject(
            hyprstream_rpc::envelope::Subject::anonymous(),
            SigningKey::from_bytes(&BOUNDARY_CALLER_KEY).verifying_key(),
        );
        assert!(
            service.mint_internal_work_token(&instance, &bare, &scope, &audience_fixture()).is_err(),
            "internal work without a verified caller identity must not mint"
        );

        // Expired caller credential: never minted.
        let expired = boundary_caller_ctx(chrono::Utc::now().timestamp() - 10);
        assert!(
            service.mint_internal_work_token(&instance, &expired, &scope, &audience_fixture()).is_err(),
            "internal work must not outlive an expired caller credential"
        );
    }

}
