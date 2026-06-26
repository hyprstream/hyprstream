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
use crate::runtime::KVQuantType;
use crate::runtime::RuntimeConfig;
use crate::services::{
    EnvelopeContext,
    NotificationClient, NotificationPublisher, PolicyClient,
};
use crate::services::generated::inference_client::InferenceClient;
use crate::services::RegistryClient;
use crate::services::generated::registry_client::{StageFilesRequest, CommitWithAuthorRequest};
use crate::services::generated::policy_client::PolicyCheck;
use crate::storage::ModelRef;
use anyhow::{anyhow, Result};
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::stream_info::TransportConfig as WireTransportConfig;
use lru::LruCache;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Default endpoint for the model service
pub const MODEL_ENDPOINT: &str = "inproc://hyprstream/model";

// ============================================================================
// ModelService (server-side)
// ============================================================================


/// Information about a loaded model
pub struct LoadedModel {
    /// Model reference string (e.g., "qwen3-small:main")
    pub model_ref: String,
    /// Resolved transport for this model's InferenceService (#320). For a
    /// co-located service this is the `Inproc` arm registered in the in-process
    /// dial registry by the spawner; the cross-host/Iroh reach (when an
    /// inference service has a network endpoint) is resolved via the Resolver.
    /// The router single-selects this to talk to the inference service.
    pub transport: TransportConfig,
    /// Handle to stop the InferenceService
    pub service_handle: hyprstream_service::SpawnedService,
    /// Client for communicating with the InferenceService (built from
    /// `transport` via `dial()` — the co-located fast path).
    pub client: InferenceClient,
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
}

impl Default for ModelServiceConfig {
    fn default() -> Self {
        Self {
            max_models: 5,
            max_context: None,
            kv_quant: KVQuantType::None,
        }
    }
}

/// Inner state for ModelService, behind Arc for continuation capture.
pub struct ModelServiceInner {
    // Business logic
    /// LRU cache of loaded models
    loaded_models: RwLock<LruCache<String, LoadedModel>>,
    /// Models currently being loaded (accepted but not yet in LRU cache)
    pending_loads: Mutex<HashSet<String>>,
    /// Service configuration
    config: ModelServiceConfig,
    /// Ed25519 signing key for creating InferenceClients
    signing_key: SigningKey,
    /// Policy client for authorization checks in InferenceService
    policy_client: PolicyClient,
    /// Notification publisher for model lifecycle events
    notification_publisher: NotificationPublisher,
    /// Registry client for resolving model paths
    registry: RegistryClient,
    // Infrastructure (for Spawnable)
    transport: TransportConfig,
    /// Expected JWT audience for token validation (RFC 8707).
    expected_audience: Option<String>,
    /// Unified JWT key source for verifying JWTs (local and federated).
    jwt_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::JwtKeySource>>,
    /// Discovery client for federated record resolution (#431). None = no
    /// federation; `resolve_model_ref`'s at:// branch then falls through to
    /// local resolution.
    discovery_client: Option<Arc<hyprstream_discovery::DiscoveryClient>>,
    /// Persistent 9P synthetic tree for the fs scope (lazily initialized).
    fs_tree: std::sync::OnceLock<crate::services::fs::SyntheticTree>,
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

impl ModelService {
    /// Create a new model service with infrastructure
    pub async fn new(
        config: ModelServiceConfig,
        signing_key: SigningKey,
        policy_client: PolicyClient,
        registry: RegistryClient,
        transport: TransportConfig,
    ) -> Result<Self> {
        // SAFETY: 5 is a valid non-zero value
        const DEFAULT_CACHE_SIZE: NonZeroUsize = match NonZeroUsize::new(5) {
            Some(n) => n,
            None => unreachable!(),
        };
        let cache_size = NonZeroUsize::new(config.max_models).unwrap_or(DEFAULT_CACHE_SIZE);

        let key_resp = policy_client.resolve_service_key(
            &crate::services::generated::policy_client::ResolveServiceKey {
                service_name: "notification".to_owned(),
            },
        ).await?;
        let notif_vk = hyprstream_rpc::crypto::VerifyingKey::from_bytes(
            key_resp.verifying_key.as_slice().try_into()
                .map_err(|_| anyhow!("Invalid verifying key length"))?,
        ).map_err(|e| anyhow!("Invalid Ed25519 key: {e}"))?;
        let notif_client = NotificationClient::for_service(
            signing_key.clone(),
            notif_vk,
            None,
        )?;
        let notification_publisher = NotificationPublisher::new(notif_client, signing_key.clone());

        Ok(Self { inner: Arc::new(ModelServiceInner {
            loaded_models: RwLock::new(LruCache::new(cache_size)),
            pending_loads: Mutex::new(HashSet::new()),
            config,
            signing_key,
            policy_client,
            notification_publisher,
            registry,
            transport,
            expected_audience: None,
            jwt_key_source: None,
            discovery_client: None,
            fs_tree: std::sync::OnceLock::new(),
        })})
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
        client: Arc<hyprstream_discovery::DiscoveryClient>,
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
    fn inference_transport(model_ref_str: &str) -> TransportConfig {
        let safe_name = model_ref_str.replace([':', '/', '\\'], "-");
        let service_name = format!("inference-{safe_name}");
        registry().endpoint(&service_name, SocketKind::Rep)
    }

    /// The deterministic InferenceService endpoint *string* for a model ref.
    ///
    /// Retained only for human-facing display and the JSON `model.loaded` event
    /// (`EventPayload::ModelLoaded`); the routable reach is the typed
    /// [`Self::inference_transport`] / the capnp `reach` list (#320).
    fn inference_endpoint(model_ref_str: &str) -> String {
        Self::inference_transport(model_ref_str).endpoint_string()
    }

    /// The network-routable reach list a remote caller would use to dial this
    /// model's InferenceService (#320). For a co-located service the transport is
    /// the `Inproc` arm, which is NEVER advertised on the wire (a remote caller
    /// can't dial it; a co-located caller uses the in-process fast path), so this
    /// returns an EMPTY list. A networked (Quic/Iroh) inference reach maps to the
    /// corresponding wire arm via the single dial→wire reach codec.
    fn model_reach(transport: &TransportConfig) -> Vec<WireTransportConfig> {
        hyprstream_rpc::moq_stream::dial_transport_to_wire(transport)
            .into_iter()
            .collect()
    }

    /// Router seam (#322 leaf cell-router over the #320 single-select base):
    /// pick ONE InferenceService for a request from a loaded model's replica
    /// set, applying capacity-weighted HRW + session affinity.
    ///
    /// Today a model still maps 1:1 to one co-located InferenceService, so the
    /// router's `load_state` has a single entry and HRW trivially picks it; the
    /// fast path returns the in-process `model.client` directly (no dial). When
    /// the replica set grows (cross-host reaches resolved via the Resolver per
    /// the federated-model-addressing spike), this is where HRW single-selects a
    /// networked (Iroh) reach and dials it. The router body operates on the
    /// resolved replica set; OID/reach resolution is the entry's job.
    ///
    /// `session_id` is the placement key (defaults to "default" when the caller
    /// has no session — keeps HRW stable for non-session-scoped requests like
    /// `apply_chat_template`).
    fn select_inference_server(
        model: &mut LoadedModel,
        session_id: &str,
    ) -> InferenceClient {
        let now = Instant::now();
        if let Some(placement) = model.router.place(session_id, &model.load_state, now) {
            // In v1 the co-located node is always in the replica set (seeded at
            // load). If HRW picks it, use the in-process fast path — no dial.
            if placement.node_id == Self::co_located_node_id(model) {
                return model.client.clone();
            }
            // Networked selection: a future branch dials the chosen Iroh reach
            // (after the #319/#328 mesh-authz gate). Not wired in v1 — fall back
            // to the co-located client so requests still succeed while the
            // federation entry is being built. See the AUTHZ SEAM note below.
            debug!(
                "router placed session {session_id} on remote node {:?} but no \
                 networked dial path is wired yet — using co-located fallback",
                placement.node_id
            );
        }
        model.client.clone()

        // AUTHZ SEAM (#319/#328 mesh policy) — GAP, intentionally not wired here:
        // a co-located `Inproc` selection is an in-process call within the same
        // trust domain, so no mesh policy fires. When this seam grows a branch
        // that DIALS a networked (Iroh) reach to a remote InferenceService, that
        // dial MUST first be gated by the #319 mesh check on the target host
        // identity — `mesh.rpc` / `infer.stage` for the `service:inference:host-<id>`
        // subject in the tenant domain (the `mesh-host` policy template in
        // `auth::policy_templates`, deny-by-default, never wildcard). The vocab +
        // per-host identity (`node_identity::inference_host_subject`) exist; the
        // enforcement call belongs on that future networked branch, not on the
        // co-located fast path. (Per-host identity:
        // `hyprstream_rpc::node_identity::mesh_host_subject`.) Cross-host
        // inference spawn/discovery is deferred to #282.
    }

    /// Identity of the co-located InferenceService = our own Ed25519 verifying
    /// key. Used by [`Self::select_inference_server`] to recognise the HRW
    /// winner as the in-process fast path.
    fn co_located_node_id(model: &LoadedModel) -> crate::services::router::NodeId {
        // `load_state[0]` is the co-located entry (seeded first at load). All
        // entries share the model's signing identity in v1; in a multi-replica
        // future each entry carries its own node_id.
        model
            .load_state
            .first()
            .map(|info| info.node_id)
            .unwrap_or([0u8; 32])
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
            .get_record(&hyprstream_discovery::GetRecordRequest {
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
    async fn load_model(&self, model_ref_str: &str, max_context: Option<u32>, kv_quant: Option<KVQuantType>) -> Result<String> {
        // Check if already loaded
        {
            let mut cache = self.loaded_models.write().await;
            if let Some(model) = cache.get_mut(model_ref_str) {
                model.last_used = Instant::now();
                debug!("Model {} already loaded", model_ref_str);
                return Ok(Self::inference_endpoint(model_ref_str));
            }
        }

        // Atomically check-and-insert into pending_loads (prevents duplicate GPU loads
        // when multiple requests arrive during the ~40s load window).
        // HashSet::insert returns false if the value was already present.
        {
            let mut pending = self.pending_loads.lock().await;
            if !pending.insert(model_ref_str.to_owned()) {
                anyhow::bail!(
                    "Model {} is already being loaded — please retry shortly",
                    model_ref_str
                );
            }
        }

        let result = self.load_model_inner(model_ref_str, max_context, kv_quant).await;

        // Always remove from pending, whether load succeeded or failed
        self.pending_loads.lock().await.remove(model_ref_str);

        if result.is_ok() {
            info!("Model {} loaded successfully", model_ref_str);
        }
        result
    }

    /// Inner model loading logic, called by load_model() which manages pending_loads.
    async fn load_model_inner(&self, model_ref_str: &str, max_context: Option<u32>, kv_quant: Option<KVQuantType>) -> Result<String> {
        // Parse/resolve model reference
        let model_ref = self.resolve_model_ref(model_ref_str).await?;

        // Get model path from registry
        let tracked = self.registry.get_by_name(model_ref.name()).await
            .map_err(|e| anyhow!("Model '{}' not found in registry: {}", model_ref.name(), e))?;
        let repo_client = self.registry.repo(&tracked.id);

        let branch_name = match &model_ref.git_ref {
            crate::storage::GitRef::Branch(name) => name.clone(),
            _ => repo_client.get_head().await.unwrap_or_else(|_| "main".to_owned()),
        };
        let worktrees = repo_client.list_worktrees().await?;
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

        let endpoint = Self::inference_endpoint(model_ref_str);

        info!("Loading model {} at endpoint {}", model_ref_str, endpoint);

        // Create runtime config - use per-model config if provided, otherwise service defaults
        let runtime_config = RuntimeConfig {
            max_context: max_context.or(self.config.max_context),
            kv_quant_type: kv_quant.unwrap_or(self.config.kv_quant),
            ..Default::default()
        };

        // Obtain FsOps from the registry for path-contained adapter I/O
        let fs: Option<crate::services::WorktreeClient> = Some(repo_client.worktree(&branch_name));

        // Start InferenceService for this model via standard Spawnable infrastructure
        let spawner = hyprstream_service::ServiceSpawner::threaded();

        // Resolve the InferenceService transport via the typed `TransportConfig`
        // (#320). For a co-located service this is the `Inproc` arm; the spawner
        // registers it in the in-process dial registry (`register_inproc`) and
        // the same typed transport builds the client below — no string parsing on
        // the inference client path. (The cross-host/Iroh reach is resolved via
        // the Resolver when an inference service has a network endpoint; pkarr
        // auto-discovery stays deferred to #282.)
        let transport = Self::inference_transport(model_ref_str);
        let mut service_config = crate::services::InferenceServiceConfig::new(
            &model_path,
            runtime_config,
            self.signing_key.verifying_key(),
            self.signing_key.clone(),
            transport.clone(),
            fs,
        );
        if let Some(ref aud) = self.expected_audience {
            service_config = service_config.with_expected_audience(aud.clone());
        }
        if let Some(ref src) = self.jwt_key_source {
            service_config = service_config.with_jwt_key_source(src.clone());
        }
        let service_handle = spawner.spawn(service_config).await
            .map_err(|e| anyhow!("Failed to spawn inference service: {}", e))?;

        // Create client for this service from the typed transport (#320).
        // Inference services share the model service's signing key, so use our
        // own verifying key directly — no PolicyService lookup needed.
        let client = InferenceClient::for_transport(
            &transport,
            self.signing_key.clone(),
            self.signing_key.verifying_key(),
            None,
        )?;

        // Load TTT config from model's config.json (if TTT is enabled)
        let ttt_config = crate::runtime::model_config::ModelConfig::load_training_config(&model_path)
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
        let generation_defaults = crate::config::SamplingParams::from_model_path(&model_path)
            .await
            .unwrap_or_default();

        // Check if we need to evict
        {
            let mut cache = self.loaded_models.write().await;
            if cache.len() >= self.config.max_models {
                if let Some((evicted_ref, mut evicted)) = cache.pop_lru() {
                    info!("Evicting model {} to load {}", evicted_ref, model_ref_str);
                    // Stop the evicted service in background (fire-and-forget)
                    #[allow(clippy::let_underscore_future)]
                    let _ = tokio::spawn(async move {
                        let _ = evicted.service_handle.stop().await;
                    });
                }
            }

            // Add to cache
            cache.put(
                model_ref_str.to_owned(),
                LoadedModel {
                    model_ref: model_ref_str.to_owned(),
                    transport: transport.clone(),
                    service_handle,
                    client,
                    // #322 leaf cell-router. v1: single co-located replica (the
                    // service's own verifying key as the node identity). The
                    // router fast-paths to `client` when HRW picks the
                    // co-located node; the replica set grows when cross-host
                    // reaches are resolved via the Resolver.
                    router: crate::services::router::CellRouter::default(),
                    load_state: vec![crate::services::router::InferenceServerInfo {
                        node_id: self.signing_key.verifying_key().to_bytes(),
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

        Ok(endpoint)
    }

    /// Unload a model
    async fn unload_model(&self, model_ref_str: &str) -> Result<()> {
        let mut cache = self.loaded_models.write().await;
        if let Some((_, mut model)) = cache.pop_entry(model_ref_str) {
            info!("Unloading model {}", model_ref_str);
            let _ = model.service_handle.stop().await;
            let model_name = model_ref_str.split(':').next().unwrap_or(model_ref_str);
            let scope = format!("serve:model:{}", model_name);
            let event = crate::events::EventEnvelope::new(
                crate::events::EventSource::Model,
                scope.clone(),
                crate::events::EventPayload::ModelUnloaded {
                    model_ref: model_ref_str.to_owned(),
                },
            );
            if let Ok(payload) = serde_json::to_vec(&event) {
                let _ = self.notification_publisher.publish(&scope, &payload).await;
            }
            Ok(())
        } else {
            Err(anyhow!("Model {} is not loaded", model_ref_str))
        }
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

    /// Return status entries for all known models (loaded + loading).
    /// Absence from this list means unloaded.
    async fn model_status_all(&self) -> Vec<GenModelStatusEntry> {
        let cache = self.loaded_models.read().await;
        let pending = self.pending_loads.lock().await;
        let mut entries: Vec<GenModelStatusEntry> = cache
            .iter()
            .map(|(_, model)| GenModelStatusEntry {
                model_ref: model.model_ref.clone(),
                status: "loaded".to_owned(),
                reach: Self::model_reach(&model.transport),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                generation_defaults: Self::sampling_params_to_wire(&model.generation_defaults),
            })
            .collect();
        for model_ref in pending.iter() {
            if !cache.contains(model_ref) {
                entries.push(GenModelStatusEntry {
                    model_ref: model_ref.clone(),
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
    async fn model_status_single(&self, model_ref_str: &str) -> Vec<GenModelStatusEntry> {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(model_ref_str) {
            return vec![GenModelStatusEntry {
                model_ref: model_ref_str.to_owned(),
                status: "loaded".to_owned(),
                reach: Self::model_reach(&model.transport),
                loaded_at: model.loaded_at.elapsed().as_millis() as i64,
                last_used: model.last_used.elapsed().as_millis() as i64,
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                generation_defaults: Self::sampling_params_to_wire(&model.generation_defaults),
            }];
        }
        let pending = self.pending_loads.lock().await;
        if pending.contains(model_ref_str) {
            vec![GenModelStatusEntry {
                model_ref: model_ref_str.to_owned(),
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
    async fn model_status(&self, model_ref_str: &str) -> ModelStatusResponse {
        let cache = self.loaded_models.read().await;
        if let Some(model) = cache.peek(model_ref_str) {
            ModelStatusResponse {
                loaded: true,
                reach: Self::model_reach(&model.transport),
                online_training_config: model.ttt_config.as_ref()
                    .map(Self::ttt_config_to_wire)
                    .unwrap_or_default(),
                ..Default::default()
            }
        } else {
            ModelStatusResponse { loaded: false, ..Default::default() }
        }
    }
    async fn get_inference_client(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<InferenceClient> {
        let _endpoint = self.load_model(model_ref_str, None, None).await?;
        let mut cache = self.loaded_models.write().await;
        let model = cache
            .get_mut(model_ref_str)
            .ok_or_else(|| anyhow!("Model {} not found after loading", model_ref_str))?;
        model.last_used = Instant::now();
        // #322 placement key. The envelope does not yet carry an explicit
        // session_id; use the authenticated subject as a stable per-caller key
        // (keeps HRW affinity effective for repeat requests from the same
        // caller). When a real session_id is plumbed through `EnvelopeContext`,
        // swap it in here — the router body is session_id-keyed, not
        // subject-keyed, by design.
        let placement_key = ctx.subject().to_string();
        let client = Self::select_inference_server(model, &placement_key);
        // TODO: Forward user JWT to worker via per-call builder (delegated_bearer or
        // request().jwt(token).call(payload)) once inference methods support CallOptions.
        // The previous with_jwt() call was mutating shared state — unsafe with pooling.
        let _ = ctx.jwt_token();
        Ok(client)
    }

    /// Load a LoRA adapter from a file
    async fn load_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext, path: &str) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.load_lora(path).await
    }

    /// Unload the current LoRA adapter
    async fn unload_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.unload_lora().await
    }

    /// Check if a LoRA adapter is loaded
    async fn has_lora(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<bool> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.has_lora().await
    }

    // Training loop control - forward to InferenceService via ZMQ
    async fn writeback_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.ttt_writeback().await
    }

    async fn evict_adaptation(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.ttt_evict().await
    }

    async fn zero_delta(&self, model_ref_str: &str, ctx: &EnvelopeContext) -> Result<()> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.ttt_zero().await
    }

    async fn get_delta_status_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<crate::services::generated::inference_client::DeltaStatusResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.get_delta_status().await
    }

    async fn snapshot_delta_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
    ) -> Result<crate::services::generated::inference_client::SnapshotDeltaResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.snapshot_delta().await
    }

    async fn export_peft_adapter_forward(
        &self,
        model_ref_str: &str,
        ctx: &EnvelopeContext,
        data: &ExportPeftRequest,
    ) -> Result<ExportPeftResult> {
        let client = self.get_inference_client(model_ref_str, ctx).await?;
        client.export_peft_adapter(data).await
    }

}

// ═══════════════════════════════════════════════════════════════════════════════
// ModelHandler Implementation — generated dispatch for top-level + typed scope traits
// ═══════════════════════════════════════════════════════════════════════════════

use crate::services::generated::model_client::{
    ModelHandler, TttHandler, AdapterHandler, InferHandler,
    dispatch_model, serialize_response, ModelResponseVariant,
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
// AdaptationStrategy is now from inference_client (canonical source via using-import).
// model_client types reference it directly — no conversion needed.


#[async_trait::async_trait(?Send)]
impl TttHandler for ModelService {
    async fn handle_init(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &LoraConfig,
    ) -> Result<()> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.create_lora(data).await
    }

    async fn handle_train(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<TrainStepResult> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.train_step(data).await
    }

    async fn handle_train_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &TrainStepRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let client = self.get_inference_client(model_ref, ctx).await?;
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
        let client = self.get_inference_client(model_ref, ctx).await?;
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
        &self, _ctx: &EnvelopeContext, _request_id: u64,
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
            let is_loaded = {
                let cache = self.loaded_models.read().await;
                cache.contains(model_ref)
            };
            if is_loaded {
                info!("Auto-reloading {} after TTT config change", model_ref);
                self.unload_model(model_ref).await?;
                self.load_model(model_ref, None, None).await?;
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
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.merge_lora(data).await
    }
}

#[async_trait::async_trait(?Send)]
impl InferHandler for ModelService {
    async fn handle_generate_stream(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &GenerationRequest,
    ) -> Result<(crate::services::generated::model_client::StreamInfo, hyprstream_rpc::service::Continuation)> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        let ephemeral_pubkey = ctx.ephemeral_pubkey()
            .ok_or_else(|| anyhow!("Streaming requires client ephemeral pubkey for E2E authentication"))?;
        let stream_info = client.generate_stream(data, ephemeral_pubkey).await?;
        Ok((stream_info, Box::pin(async {})))
    }

    async fn handle_apply_chat_template(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &ChatTemplateRequest,
    ) -> Result<String> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.apply_chat_template(data).await
    }

    async fn handle_embed(
        &self, ctx: &EnvelopeContext, _request_id: u64,
        model_ref: &str, data: &EmbedImagesRequest,
    ) -> Result<EmbedImagesResponse> {
        let client = self.get_inference_client(model_ref, ctx).await?;
        client.embed(data).await
    }

    async fn handle_status(
        &self, _ctx: &EnvelopeContext, _request_id: u64, model_ref: &str,
    ) -> Result<ModelStatusResponse> {
        Ok(self.model_status(model_ref).await)
    }
}

#[async_trait::async_trait(?Send)]
impl ModelHandler for ModelService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        let subject = ctx.subject();
        let allowed = self.policy_client.check(&PolicyCheck { subject: subject.to_string(), domain: "*".to_owned(), resource: resource.to_owned(), operation: operation.to_owned() }).await.unwrap_or_else(|e| {
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
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &LoadModelRequest,
    ) -> Result<ModelResponseVariant> {
        let max_ctx = data.max_context.filter(|&n| n != 0);
        let kv_q = data.kv_quant.filter(|q| *q != KVQuantType::None);
        let model_ref = &data.model_ref;
        match self.load_model(model_ref, max_ctx, kv_q).await {
            Ok(_endpoint) => Ok(ModelResponseVariant::LoadResult(LoadedModelResponse {
                model_ref: model_ref.to_owned(),
                reach: Self::model_reach(&Self::inference_transport(model_ref)),
            })),
            Err(e) => Ok(ModelResponseVariant::Error(ErrorInfo {
                message: format!("Failed to load model: {e}"),
                code: "LOAD_FAILED".into(),
                details: String::new(),
            })),
        }
    }

    async fn handle_unload(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &UnloadModelRequest,
    ) -> Result<ModelResponseVariant> {
        let model_ref = &data.model_ref;
        match self.unload_model(model_ref).await {
            Ok(()) => Ok(ModelResponseVariant::UnloadResult),
            Err(e) => Ok(ModelResponseVariant::Error(ErrorInfo {
                message: format!("Failed to unload model: {e}"),
                code: "UNLOAD_FAILED".into(),
                details: String::new(),
            })),
        }
    }

    async fn handle_status(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
        data: &StatusRequest,
    ) -> Result<ModelResponseVariant> {
        let entries = if data.model_ref.is_empty() {
            self.model_status_all().await
        } else {
            self.model_status_single(&data.model_ref).await
        };
        Ok(ModelResponseVariant::StatusResult(entries))
    }

    async fn handle_health_check(
        &self, _ctx: &EnvelopeContext, _request_id: u64,
    ) -> Result<ModelResponseVariant> {
        let cache = self.loaded_models.read().await;
                let loaded_count = cache.len() as u32;
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

use crate::services::generated::model_client::{
    NpWalk, NpOpen, NpRead, NpWrite, NpClunk, NpStatReq, NpCreate, NpRemove,
    RWalk, ROpen, RRead, RWrite, RStat,
    Qid as GenQid, NpStat as GenNpStat,
    FsHandler,
};
use crate::services::fs::{SyntheticTree, SyntheticNode, SyntheticQid};
use hyprstream_vfs::DirEntry;

impl ModelService {
    /// Get the persistent synthetic 9P tree (lazily initialized).
    fn fs_tree(&self) -> &SyntheticTree {
        self.inner.fs_tree.get_or_init(|| self.build_fs_tree())
    }

    /// Build a synthetic 9P tree from current model service state.
    fn build_fs_tree(&self) -> SyntheticTree {
        let inner_list = Arc::clone(&self.inner);
        let inner_resolve = Arc::clone(&self.inner);

        SyntheticTree::new(SyntheticNode::DynamicDir {
            list: Box::new(move || {
                let cache = inner_list.loaded_models.blocking_read();
                let pending = inner_list.pending_loads.blocking_lock();
                let mut entries: Vec<DirEntry> = cache
                    .iter()
                    .map(|(name, _)| DirEntry { name: name.clone(), is_dir: true, size: 0, stat: None })
                    .collect();
                for name in pending.iter() {
                    if !cache.contains(name) {
                        entries.push(DirEntry { name: name.clone(), is_dir: true, size: 0, stat: None });
                    }
                }
                entries
            }),
            resolve: Box::new(move |ref_name| {
                let cache = inner_resolve.loaded_models.blocking_read();
                let is_loaded = cache.contains(ref_name);
                let defaults_json = if let Some(model) = cache.peek(ref_name) {
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
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        let (_fid, qid) = tree.walk(&data.wnames, &owner, Some(data.newfid))
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RWalk { qid: Self::qid_to_gen(&qid) })
    }

    async fn handle_open(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpOpen,
    ) -> Result<ROpen> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        // Re-walk to get the fid, then open.
        let (qid, iounit) = tree.open(data.fid, data.mode, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(ROpen { qid: Self::qid_to_gen(&qid), iounit })
    }

    async fn handle_read(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpRead,
    ) -> Result<RRead> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        let bytes = tree.read(data.fid, data.offset, data.count, &owner)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RRead { data: bytes })
    }

    async fn handle_write(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpWrite,
    ) -> Result<RWrite> {
        let tree = self.fs_tree();
        let subject = ctx.subject();
        let owner = subject.to_string();
        let count = tree.write(data.fid, data.offset, &data.data, &owner, &subject)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(RWrite { count })
    }

    async fn handle_clunk(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpClunk,
    ) -> Result<()> {
        let tree = self.fs_tree();
        let owner = ctx.subject().to_string();
        tree.clunk(data.fid, &owner);
        Ok(())
    }

    async fn handle_stat(&self, ctx: &EnvelopeContext, _request_id: u64,
        _model_ref: &str, data: &NpStatReq,
    ) -> Result<RStat> {
        let tree = self.fs_tree();
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
    kv_quant: crate::model_capnp::KVQuantType,
}

impl ParsedLoadRequest {
    fn to_load_params(&self) -> (Option<u32>, Option<KVQuantType>) {
        use crate::model_capnp::KVQuantType as CKV;
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
    /// Try to parse a load request from the raw Cap'n Proto payload.
    /// Returns `None` for all other request variants (list, unload, health, scoped, etc.).
    fn try_parse_load_request(payload: &[u8]) -> Option<(u64, ParsedLoadRequest)> {
        use crate::model_capnp::model_request;
        use crate::model_capnp::KVQuantType as CKV;
        use crate::optional_capnp::option_uint32;
        use crate::model_capnp::option_k_v_quant_type;
        let reader = capnp::serialize::read_message(
            &mut std::io::Cursor::new(payload),
            capnp::message::ReaderOptions::new(),
        ).ok()?;
        let req = reader.get_root::<model_request::Reader>().ok()?;
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

#[async_trait(?Send)]
impl crate::services::RequestService for ModelService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
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
        if let Some((request_id, load_data)) = Self::try_parse_load_request(payload) {
            // Fast path: if already loaded or already loading, return immediately
            {
                let mut cache = self.loaded_models.write().await;
                if let Some(model) = cache.get_mut(&load_data.model_ref) {
                    model.last_used = Instant::now();
                    let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                        LoadedModelResponse {
                            model_ref: load_data.model_ref.clone(),
                            reach: Self::model_reach(&model.transport),
                        },
                    ))?;
                    return Ok((response, None));
                }
            }
            // If already being loaded by another request, return the reach
            // without spawning a duplicate continuation
            {
                let pending = self.pending_loads.lock().await;
                if pending.contains(&load_data.model_ref) {
                    debug!("Model {} already being loaded, deduplicating", load_data.model_ref);
                    let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                        LoadedModelResponse {
                            model_ref: load_data.model_ref.clone(),
                            reach: Self::model_reach(&Self::inference_transport(&load_data.model_ref)),
                        },
                    ))?;
                    return Ok((response, None));
                }
            }

            // Slow path: return "accepted" immediately, load in continuation
            let model_ref = load_data.model_ref.clone();
            info!("Load request accepted for {} (async)", model_ref);

            let response = serialize_response(request_id, &ModelResponseVariant::LoadResult(
                LoadedModelResponse {
                    model_ref: model_ref.clone(),
                    reach: Self::model_reach(&Self::inference_transport(&model_ref)),
                },
            ))?;

            let service = self.clone(); // Arc clone — cheap, 'static
            let (load_max_context, load_kv_quant) = load_data.to_load_params();
            let continuation: crate::services::Continuation = Box::pin(async move {
                let model_name = model_ref.split(':').next().unwrap_or(&model_ref);
                let scope = format!("serve:model:{}", model_name);
                match service.load_model(&model_ref, load_max_context, load_kv_quant).await {
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
                            let n = service.notification_publisher.publish(&scope, &payload).await
                                .unwrap_or(0);
                            debug!("Published model.loaded to {} subscriber(s)", n);
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
                            let _ = service.notification_publisher.publish(&scope, &payload).await;
                        }
                    }
                }
            });

            return Ok((response, Some(continuation)));
        }

        dispatch_model(self, ctx, payload).await
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

    #[test]
    fn test_config_defaults() {
        let config = ModelServiceConfig::default();
        assert_eq!(config.max_models, 5);
        assert_eq!(config.max_context, None);
        assert_eq!(config.kv_quant, KVQuantType::None);
    }

    /// #320: the co-located InferenceService resolves (via the local Resolver
    /// registry) to an `Inproc` transport — the same arm the spawner registers in
    /// the in-process dial registry and the router single-selects.
    #[test]
    fn inference_transport_is_inproc_co_located() {
        // Default registry mode is Inproc; idempotent if another test inited it.
        hyprstream_rpc::registry::init(hyprstream_rpc::registry::EndpointMode::Inproc, None);
        let t = ModelService::inference_transport("qwen3-small:main");
        match &t.endpoint {
            hyprstream_rpc::transport::EndpointType::Inproc { endpoint } => {
                assert!(
                    endpoint.contains("inference-qwen3-small-main"),
                    "deterministic per-model inproc name, got {endpoint}"
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
    /// fast path is present. Identity (nodeId) is preserved (real identity bind).
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
}
