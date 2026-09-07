//! Volatile Discovery state behind a backend-neutral contract.
//!
//! The values in this store are never identity or policy authority. Callers
//! validate signed artifacts before writes and re-check accepted-current state
//! before use. The shared backend only makes short-lived reach and liveness
//! observations coherent across Discovery replicas.

use anyhow::{bail, Result};
use async_trait::async_trait;
use hyprstream_rpc::identity::Did;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiscoveryStateBackend {
    #[default]
    Memory,
    Valkey,
    Tiered,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryStateConfig {
    pub announcement_capacity: usize,
    pub liveness_capacity: usize,
    pub artifact_capacity: usize,
}

impl Default for MemoryStateConfig {
    fn default() -> Self {
        Self {
            announcement_capacity: 16_384,
            liveness_capacity: 16_384,
            artifact_capacity: 4_096,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ValkeyStateConfig {
    pub url: String,
    pub key_prefix: String,
    pub pool_size: usize,
    pub announcement_capacity: usize,
    pub liveness_capacity: usize,
    pub artifact_capacity: usize,
    /// Upper bound for a shared-state command. Required-HA callers receive an
    /// error after this interval instead of waiting indefinitely or serving L1.
    pub command_timeout_ms: u64,
}

impl Default for ValkeyStateConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            key_prefix: "hs".to_owned(),
            pool_size: 8,
            announcement_capacity: 65_536,
            liveness_capacity: 65_536,
            artifact_capacity: 16_384,
            command_timeout_ms: 2_000,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TieredStateConfig {
    /// Maximum time an L1 value may be reused after its L2 revision was
    /// observed. Every L1 hit still verifies the cheap L2 revision key.
    pub l1_max_ttl_ms: u64,
}

impl Default for TieredStateConfig {
    fn default() -> Self {
        Self {
            l1_max_ttl_ms: 1_000,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DiscoveryStateConfig {
    pub backend: DiscoveryStateBackend,
    /// Declares that more than one active Discovery replica may serve calls.
    /// Such a deployment must select a shared backend.
    pub active_active: bool,
    pub memory: MemoryStateConfig,
    pub valkey: ValkeyStateConfig,
    pub tiered: TieredStateConfig,
}

/// Constructed backend handle accepted by [`crate::DiscoveryService`].
#[derive(Clone)]
pub struct DiscoveryState(Arc<dyn DiscoveryStateStore>);

impl DiscoveryState {
    pub async fn connect(config: &DiscoveryStateConfig) -> Result<Self> {
        anyhow::ensure!(
            config.memory.announcement_capacity > 0
                && config.memory.liveness_capacity > 0
                && config.memory.artifact_capacity > 0,
            "Discovery memory capacities must be positive"
        );
        if config.active_active && config.backend == DiscoveryStateBackend::Memory {
            bail!("discovery.state.active_active requires valkey or tiered backend");
        }
        if config.backend == DiscoveryStateBackend::Tiered {
            anyhow::ensure!(
                config.tiered.l1_max_ttl_ms > 0,
                "Discovery tiered l1_max_ttl_ms must be positive"
            );
        }
        let memory = || {
            Arc::new(MemoryStateStore::new(
                config.memory.announcement_capacity,
                config.memory.liveness_capacity,
                config.memory.artifact_capacity,
            ))
        };

        match config.backend {
            DiscoveryStateBackend::Memory => Ok(Self(memory())),
            DiscoveryStateBackend::Valkey | DiscoveryStateBackend::Tiered => {
                #[cfg(target_arch = "wasm32")]
                {
                    bail!("Valkey-backed Discovery state is unavailable on wasm32")
                }
                #[cfg(all(not(target_arch = "wasm32"), not(feature = "valkey")))]
                {
                    bail!("Valkey-backed Discovery state requires the valkey feature")
                }
                #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
                {
                    let valkey = Arc::new(ValkeyStateStore::connect(&config.valkey).await?);
                    if config.backend == DiscoveryStateBackend::Valkey {
                        Ok(Self(valkey))
                    } else {
                        Ok(Self(Arc::new(TieredStateStore::new(
                            memory(),
                            valkey,
                            config.tiered.l1_max_ttl_ms,
                        ))))
                    }
                }
            }
        }
    }

    pub(crate) fn into_inner(self) -> Arc<dyn DiscoveryStateStore> {
        self.0
    }
}

/// Endpoint data stored per announced entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AnnouncedEndpoint {
    pub(crate) socket_kind: String,
    pub(crate) endpoint: String,
    pub(crate) service_jwt: String,
    pub(crate) service_did: Did,
    pub(crate) capabilities: BTreeSet<String>,
    pub(crate) accepted_state_digest: Vec<u8>,
    pub(crate) accepted_state_epoch: u64,
    pub(crate) response_key_id: String,
    pub(crate) request_kem_key_id: String,
    pub(crate) request_kem_recipient: Vec<u8>,
    /// Signed/application expiry carried by the announcement.
    pub(crate) expires_at_unix_ms: i64,
    pub(crate) source_signer: [u8; 32],
    /// Effective cache lifetime: no later than heartbeat, signed artifact, and
    /// accepted-current-state expiry.
    pub(crate) live_until_unix_ms: i64,
}

impl AnnouncedEndpoint {
    pub(crate) fn is_live_at(&self, now_unix_ms: i64) -> bool {
        now_unix_ms < self.live_until_unix_ms && now_unix_ms < self.expires_at_unix_ms
    }

    fn order(&self) -> (u64, i64) {
        (self.accepted_state_epoch, self.expires_at_unix_ms)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct LiveAllocatable {
    pub(crate) allocatable: Vec<(String, String)>,
    pub(crate) load_fraction: f32,
    /// Server receipt time, never the reporting node's clock.
    pub(crate) last_seen: i64,
    pub(crate) live_until_unix_ms: i64,
}

impl LiveAllocatable {
    fn is_live_at(&self, now_unix_ms: i64) -> bool {
        now_unix_ms < self.live_until_unix_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CachedEntityStatement {
    pub(crate) jwt: String,
    pub(crate) fetched_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CachedEnvelopeKeyset {
    pub(crate) cose_keyset_cbor: Vec<u8>,
    pub(crate) fetched_at: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PutResult {
    Stored,
    IgnoredOlder,
}

/// Typed query surface needed by Discovery handlers and resolvers.
///
/// Native futures are `Send`; browser/embedded WASM uses the repository's
/// established `?Send` convention. Implementations remain `Send + Sync` so a
/// native service can share one store across handlers.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub(crate) trait DiscoveryStateStore: Send + Sync {
    async fn put_announcement(
        &self,
        service_name: &str,
        endpoint: AnnouncedEndpoint,
    ) -> Result<PutResult>;
    async fn announcements_for(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Result<Vec<AnnouncedEndpoint>>;
    async fn all_announcements(
        &self,
        now_unix_ms: i64,
    ) -> Result<Vec<(String, Vec<AnnouncedEndpoint>)>>;

    async fn put_liveness(&self, node: &Did, value: LiveAllocatable) -> Result<PutResult>;
    #[cfg(test)]
    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>>;
    /// Bounded enumeration of live nodes from the authoritative backend.
    async fn all_liveness(&self, now_unix_ms: i64) -> Result<Vec<(Did, LiveAllocatable)>>;

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()>;
    async fn entity_statement(&self, issuer: &str) -> Result<Option<CachedEntityStatement>>;
    async fn known_issuers(&self) -> Result<Vec<String>>;
    async fn known_issuer_count(&self) -> Result<usize>;

    async fn put_envelope_keyset(
        &self,
        service_did: &str,
        value: CachedEnvelopeKeyset,
    ) -> Result<()>;
    async fn envelope_keyset(&self, service_did: &str) -> Result<Option<CachedEnvelopeKeyset>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AnnouncementKey {
    service_name: String,
    socket_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExpiringKey {
    Announcement(AnnouncementKey),
    Liveness(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpiryEntry {
    expires_at_unix_ms: i64,
    version: u64,
    key: ExpiringKey,
}

impl Ord for ExpiryEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.expires_at_unix_ms
            .cmp(&other.expires_at_unix_ms)
            .then_with(|| self.version.cmp(&other.version))
            .then_with(|| self.key.cmp(&other.key))
    }
}

impl PartialOrd for ExpiryEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct Versioned<T> {
    value: T,
    version: u64,
}

struct MemoryInner {
    announcements: HashMap<AnnouncementKey, Versioned<AnnouncedEndpoint>>,
    service_index: HashMap<String, BTreeSet<String>>,
    liveness: HashMap<String, Versioned<LiveAllocatable>>,
    expiry: BinaryHeap<Reverse<ExpiryEntry>>,
    entity_statements: HashMap<String, CachedEntityStatement>,
    envelope_keysets: HashMap<String, CachedEnvelopeKeyset>,
    next_version: u64,
}

/// Bounded in-process backend used by single-replica and WASM deployments.
pub(crate) struct MemoryStateStore {
    inner: Mutex<MemoryInner>,
    announcement_capacity: usize,
    liveness_capacity: usize,
    artifact_capacity: usize,
}

impl MemoryStateStore {
    pub(crate) fn new(
        announcement_capacity: usize,
        liveness_capacity: usize,
        artifact_capacity: usize,
    ) -> Self {
        Self {
            inner: Mutex::new(MemoryInner {
                announcements: HashMap::new(),
                service_index: HashMap::new(),
                liveness: HashMap::new(),
                expiry: BinaryHeap::new(),
                entity_statements: HashMap::new(),
                envelope_keysets: HashMap::new(),
                next_version: 1,
            }),
            announcement_capacity,
            liveness_capacity,
            artifact_capacity,
        }
    }

    pub(crate) fn production_default() -> Arc<dyn DiscoveryStateStore> {
        Arc::new(Self::new(16_384, 16_384, 4_096))
    }

    fn next_version(inner: &mut MemoryInner) -> u64 {
        let version = inner.next_version;
        inner.next_version = inner.next_version.saturating_add(1);
        version
    }

    // Replacement/declined L1 fills must not accumulate an unbounded heap of
    // stale expiry versions while the actual maps remain capacity-bounded.
    fn compact_expiry(inner: &mut MemoryInner) {
        let limit = 2 * (inner.announcements.len() + inner.liveness.len());
        if inner.expiry.len() > limit {
            let MemoryInner {
                announcements,
                liveness,
                expiry,
                ..
            } = inner;
            expiry.retain(|Reverse(entry)| match &entry.key {
                ExpiringKey::Announcement(key) => announcements
                    .get(key)
                    .is_some_and(|value| value.version == entry.version),
                ExpiringKey::Liveness(node) => liveness
                    .get(node)
                    .is_some_and(|value| value.version == entry.version),
            });
        }
    }

    fn reap(inner: &mut MemoryInner, now_unix_ms: i64) {
        while let Some(Reverse(head)) = inner.expiry.peek() {
            if head.expires_at_unix_ms > now_unix_ms {
                break;
            }
            let Some(Reverse(expired)) = inner.expiry.pop() else {
                break;
            };
            match &expired.key {
                ExpiringKey::Announcement(key) => {
                    let remove = inner
                        .announcements
                        .get(key)
                        .is_some_and(|entry| entry.version == expired.version);
                    if remove {
                        inner.announcements.remove(key);
                        if let Some(kinds) = inner.service_index.get_mut(&key.service_name) {
                            kinds.remove(&key.socket_kind);
                            if kinds.is_empty() {
                                inner.service_index.remove(&key.service_name);
                            }
                        }
                    }
                }
                ExpiringKey::Liveness(node) => {
                    let remove = inner
                        .liveness
                        .get(node)
                        .is_some_and(|entry| entry.version == expired.version);
                    if remove {
                        inner.liveness.remove(node);
                    }
                }
            }
        }
    }

    pub(crate) fn put_announcement_sync(
        &self,
        service_name: &str,
        endpoint: AnnouncedEndpoint,
    ) -> Result<PutResult> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, unix_millis_now());
        let key = AnnouncementKey {
            service_name: service_name.to_owned(),
            socket_kind: endpoint.socket_kind.clone(),
        };
        if let Some(existing) = inner.announcements.get(&key) {
            if existing.value.order() > endpoint.order() {
                return Ok(PutResult::IgnoredOlder);
            }
        } else if inner.announcements.len() >= self.announcement_capacity {
            bail!("Discovery memory announcement capacity exhausted");
        }
        let version = Self::next_version(&mut inner);
        inner
            .service_index
            .entry(service_name.to_owned())
            .or_default()
            .insert(endpoint.socket_kind.clone());
        inner.expiry.push(Reverse(ExpiryEntry {
            expires_at_unix_ms: endpoint.live_until_unix_ms,
            version,
            key: ExpiringKey::Announcement(key.clone()),
        }));
        inner.announcements.insert(
            key,
            Versioned {
                value: endpoint,
                version,
            },
        );
        Self::compact_expiry(&mut inner);
        Ok(PutResult::Stored)
    }

    pub(crate) fn announcements_for_sync(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Vec<AnnouncedEndpoint> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now_unix_ms);
        inner
            .service_index
            .get(service_name)
            .into_iter()
            .flatten()
            .filter_map(|socket_kind| {
                inner
                    .announcements
                    .get(&AnnouncementKey {
                        service_name: service_name.to_owned(),
                        socket_kind: socket_kind.clone(),
                    })
                    .map(|entry| entry.value.clone())
            })
            .filter(|entry| entry.is_live_at(now_unix_ms))
            .collect()
    }

    fn all_announcements_sync(&self, now_unix_ms: i64) -> Vec<(String, Vec<AnnouncedEndpoint>)> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now_unix_ms);
        inner
            .service_index
            .iter()
            .filter_map(|(service_name, socket_kinds)| {
                let endpoints: Vec<_> = socket_kinds
                    .iter()
                    .filter_map(|socket_kind| {
                        inner
                            .announcements
                            .get(&AnnouncementKey {
                                service_name: service_name.clone(),
                                socket_kind: socket_kind.clone(),
                            })
                            .map(|entry| entry.value.clone())
                    })
                    .filter(|entry| entry.is_live_at(now_unix_ms))
                    .collect();
                (!endpoints.is_empty()).then(|| (service_name.clone(), endpoints))
            })
            .collect()
    }

    #[cfg(any(feature = "valkey", test, feature = "test-fixtures"))]
    pub(crate) fn clear_announcements_sync(&self, service_name: &str) {
        let mut inner = self.inner.lock();
        let Some(socket_kinds) = inner.service_index.remove(service_name) else {
            return;
        };
        for socket_kind in socket_kinds {
            inner.announcements.remove(&AnnouncementKey {
                service_name: service_name.to_owned(),
                socket_kind,
            });
        }
        Self::compact_expiry(&mut inner);
    }

    #[cfg(feature = "valkey")]
    fn clear_liveness_sync(&self, node: &Did) {
        let mut inner = self.inner.lock();
        inner.liveness.remove(node.as_str());
        Self::compact_expiry(&mut inner);
    }

    #[cfg(feature = "valkey")]
    fn clear_entity_statement_sync(&self, issuer: &str) {
        self.inner.lock().entity_statements.remove(issuer);
    }

    #[cfg(feature = "valkey")]
    fn clear_envelope_keyset_sync(&self, service_did: &str) {
        self.inner.lock().envelope_keysets.remove(service_did);
    }
}

impl Default for MemoryStateStore {
    fn default() -> Self {
        Self::new(16_384, 16_384, 4_096)
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl DiscoveryStateStore for MemoryStateStore {
    async fn put_announcement(
        &self,
        service_name: &str,
        endpoint: AnnouncedEndpoint,
    ) -> Result<PutResult> {
        self.put_announcement_sync(service_name, endpoint)
    }

    async fn announcements_for(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Result<Vec<AnnouncedEndpoint>> {
        Ok(self.announcements_for_sync(service_name, now_unix_ms))
    }

    async fn all_announcements(
        &self,
        now_unix_ms: i64,
    ) -> Result<Vec<(String, Vec<AnnouncedEndpoint>)>> {
        Ok(self.all_announcements_sync(now_unix_ms))
    }

    async fn put_liveness(&self, node: &Did, value: LiveAllocatable) -> Result<PutResult> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, unix_millis_now());
        let node = node.as_str().to_owned();
        if let Some(existing) = inner.liveness.get(&node) {
            // Both fields derive from receipt time. A newer receipt may
            // shorten lifetime; a newer lifetime also supersedes legacy
            // client-clock skew without waiting for the old value to expire.
            if existing.value.last_seen > value.last_seen
                && existing.value.live_until_unix_ms >= value.live_until_unix_ms
            {
                return Ok(PutResult::IgnoredOlder);
            }
        } else if inner.liveness.len() >= self.liveness_capacity {
            bail!("Discovery memory liveness capacity exhausted");
        }
        let version = Self::next_version(&mut inner);
        inner.expiry.push(Reverse(ExpiryEntry {
            expires_at_unix_ms: value.live_until_unix_ms,
            version,
            key: ExpiringKey::Liveness(node.clone()),
        }));
        inner.liveness.insert(node, Versioned { value, version });
        Self::compact_expiry(&mut inner);
        Ok(PutResult::Stored)
    }

    #[cfg(test)]
    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now_unix_ms);
        Ok(inner
            .liveness
            .get(node.as_str())
            .map(|entry| entry.value.clone())
            .filter(|entry| entry.is_live_at(now_unix_ms)))
    }

    async fn all_liveness(&self, now_unix_ms: i64) -> Result<Vec<(Did, LiveAllocatable)>> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now_unix_ms);
        Ok(inner
            .liveness
            .iter()
            .filter(|(_, entry)| entry.value.is_live_at(now_unix_ms))
            .map(|(node, entry)| (Did::new(node.clone()), entry.value.clone()))
            .collect())
    }

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()> {
        let mut inner = self.inner.lock();
        if !inner.entity_statements.contains_key(issuer)
            && inner.entity_statements.len() >= self.artifact_capacity
        {
            // These are inert artifacts, verified again at use. Retain the
            // newest fetched entries without interpreting JWT expiry as trust.
            if let Some(oldest) = inner
                .entity_statements
                .iter()
                .min_by_key(|(name, value)| (value.fetched_at, *name))
                .map(|(name, _)| name.clone())
            {
                inner.entity_statements.remove(&oldest);
            }
        }
        inner.entity_statements.insert(issuer.to_owned(), value);
        Ok(())
    }

    async fn entity_statement(&self, issuer: &str) -> Result<Option<CachedEntityStatement>> {
        Ok(self.inner.lock().entity_statements.get(issuer).cloned())
    }

    async fn known_issuers(&self) -> Result<Vec<String>> {
        Ok(self
            .inner
            .lock()
            .entity_statements
            .keys()
            .cloned()
            .collect())
    }

    async fn known_issuer_count(&self) -> Result<usize> {
        Ok(self.inner.lock().entity_statements.len())
    }

    async fn put_envelope_keyset(
        &self,
        service_did: &str,
        value: CachedEnvelopeKeyset,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        if !inner.envelope_keysets.contains_key(service_did)
            && inner.envelope_keysets.len() >= self.artifact_capacity
        {
            bail!("Discovery memory federation artifact capacity exhausted");
        }
        inner.envelope_keysets.insert(service_did.to_owned(), value);
        Ok(())
    }

    async fn envelope_keyset(&self, service_did: &str) -> Result<Option<CachedEnvelopeKeyset>> {
        Ok(self.inner.lock().envelope_keysets.get(service_did).cloned())
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
#[derive(Serialize, Deserialize)]
struct StoredLiveness {
    node: Did,
    #[serde(default)]
    shared_receipt: bool,
    #[serde(flatten)]
    value: LiveAllocatable,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
#[derive(Clone)]
struct ValkeyStateStore {
    pool: fred::prelude::RedisPool,
    // The last store owner closes this channel, stopping the driver runtime.
    _driver: Arc<tokio::sync::oneshot::Sender<()>>,
    prefix: String,
    announcement_capacity: usize,
    liveness_capacity: usize,
    artifact_capacity: usize,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
impl ValkeyStateStore {
    async fn connect(config: &ValkeyStateConfig) -> Result<Self> {
        use anyhow::Context as _;
        use fred::prelude::*;

        anyhow::ensure!(
            config.pool_size > 0,
            "Discovery Valkey pool_size must be positive"
        );
        anyhow::ensure!(
            !config.url.trim().is_empty(),
            "Discovery Valkey URL must be configured explicitly"
        );
        anyhow::ensure!(
            config.command_timeout_ms > 0,
            "Discovery Valkey command_timeout_ms must be positive"
        );
        anyhow::ensure!(
            config.announcement_capacity > 0
                && config.liveness_capacity > 0
                && config.artifact_capacity > 0,
            "Discovery Valkey capacities must be positive"
        );
        // Fred's URL parser constructs a rustls config. Initialize the existing
        // external TLS policy first, including in standalone state-store use.
        hyprstream_rpc::transport::pq_provider::install_pq_crypto_provider()?;
        let redis = RedisConfig::from_url(&config.url).context("invalid Discovery Valkey URL")?;
        let mut builder = Builder::from_config(redis);
        builder.with_performance_config(|performance| {
            performance.default_command_timeout =
                std::time::Duration::from_millis(config.command_timeout_ms);
        });
        // fred spawns drivers on the current runtime. Give them a runtime
        // owned by this store, independent of a temporary factory/bootstrap
        // runtime and of the caller's executor. Closing the last owner also
        // shuts down the runtime; cancelled/failed connects cannot leak it.
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let pool_size = config.pool_size;
        let timeout = std::time::Duration::from_millis(config.command_timeout_ms);
        std::thread::Builder::new()
            .name("discovery-valkey".to_owned())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = ready_tx.send(Err(anyhow::Error::from(error)));
                        return;
                    }
                };
                runtime.block_on(async move {
                    let connected: Result<RedisPool> = async {
                        let pool = builder.build_pool(pool_size)?;
                        pool.connect();
                        tokio::time::timeout(timeout, pool.wait_for_connect()).await??;
                        let _: String = pool.ping().await?;
                        Ok(pool)
                    }
                    .await;
                    match connected {
                        Ok(pool) => {
                            if ready_tx.send(Ok(pool.clone())).is_ok() {
                                let _ = shutdown_rx.await;
                                let _ = tokio::time::timeout(timeout, pool.quit()).await;
                            }
                        }
                        Err(error) => {
                            let _ = ready_tx.send(Err(error));
                        }
                    }
                });
            })
            .context("start Discovery Valkey driver")?;
        let pool = ready_rx
            .await
            .context("Discovery Valkey driver stopped during startup")??;
        Ok(Self {
            pool,
            _driver: Arc::new(shutdown_tx),
            // One hash tag keeps every Lua transaction in one cluster slot.
            prefix: format!(
                "{}:{{discovery-state}}",
                config.key_prefix.trim_end_matches(':')
            ),
            announcement_capacity: config.announcement_capacity,
            liveness_capacity: config.liveness_capacity,
            artifact_capacity: config.artifact_capacity,
        })
    }

    fn digest(value: &str) -> String {
        blake3::hash(value.as_bytes()).to_hex().to_string()
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}:{suffix}", self.prefix)
    }

    fn service_id(service_name: &str) -> String {
        Self::digest(service_name)
    }

    fn announcement_key(&self, service_name: &str, socket_kind: &str) -> String {
        format!(
            "{}:announcement:{}:{}",
            self.prefix,
            Self::service_id(service_name),
            Self::digest(socket_kind)
        )
    }

    fn announcement_index(&self, service_name: &str) -> String {
        format!(
            "{}:announcement-index:{}",
            self.prefix,
            Self::service_id(service_name)
        )
    }

    async fn revision(&self, key: String) -> Result<u64> {
        use fred::prelude::*;
        Ok(self.pool.get::<Option<u64>, _>(key).await?.unwrap_or(0))
    }

    // Persistent family-wide generations bound revision storage independently
    // of identity churn. Never expire/reset these counters: recreating a scope
    // must not reuse a revision still attached to another replica's L1 value.
    async fn announcement_revision(&self) -> Result<u64> {
        self.revision(self.key("announcement-global-revision"))
            .await
    }

    #[cfg(test)]
    async fn liveness_revision(&self) -> Result<u64> {
        self.revision(self.key("liveness-global-revision")).await
    }

    // Reap values and ALL secondary metadata in the same transaction as the
    // capacity check/read. The expiry index contains at most the live capacity
    // plus the last expired cohort; no historical service-name scan is needed.
    // All derived keys retain the deployment's cluster hash tag.
    const REAP_ANNOUNCEMENTS: &str = r#"
local function reap(expiry, services, names, revision, now)
  local expired = redis.call('ZRANGEBYSCORE', expiry, '-inf', now)
  if #expired > 0 then redis.call('INCR', revision) end
  for _, key in ipairs(expired) do
    local service = string.match(key, ':announcement:([^:]+):[^:]+$')
    local index = string.gsub(key, ':announcement:[^:]+:[^:]+$', ':announcement-index:' .. service)
    redis.call('DEL', key)
    redis.call('SREM', index, key)
    redis.call('ZREM', expiry, key)
    if redis.call('SCARD', index) == 0 then
      redis.call('SREM', services, service)
      redis.call('HDEL', names, service)
    end
  end
end
"#;

    async fn entity_revision(&self) -> Result<u64> {
        // Survives eviction/reinsertion: no per-issuer tombstones and no ABA
        // when another replica still has an evicted artifact in L1.
        self.revision(self.key("entity-global-revision")).await
    }

    async fn envelope_revision(&self, service_did: &str) -> Result<u64> {
        self.revision(format!(
            "{}:envelope-revision:{}",
            self.prefix,
            Self::digest(service_did)
        ))
        .await
    }

    async fn announcements_for_inner(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Result<Vec<AnnouncedEndpoint>> {
        use fred::prelude::*;

        // Cleanup must share the transaction with reads: a later writer may
        // refresh the same key, so GET followed by a separate DEL/SREM can
        // delete its newer value or remove its live index membership.
        const LIST: &str = r#"
reap(KEYS[4], KEYS[2], KEYS[3], KEYS[5], ARGV[1])
local live = {}
for _, key in ipairs(redis.call('SMEMBERS', KEYS[1])) do
  local encoded = redis.call('GET', key)
  if encoded then
    local ok, value = pcall(cjson.decode, encoded)
    if not ok then return redis.error_reply('corrupt Discovery announcement') end
    if tonumber(value.live_until_unix_ms) > tonumber(ARGV[1]) and tonumber(value.expires_at_unix_ms) > tonumber(ARGV[1]) then
      table.insert(live, encoded)
    else
      redis.call('INCR', KEYS[5])
      redis.call('DEL', key)
      redis.call('SREM', KEYS[1], key)
      redis.call('ZREM', KEYS[4], key)
    end
  else
    redis.call('SREM', KEYS[1], key)
    redis.call('ZREM', KEYS[4], key)
  end
end
if redis.call('SCARD', KEYS[1]) == 0 then
  redis.call('SREM', KEYS[2], ARGV[2])
  redis.call('HDEL', KEYS[3], ARGV[2])
end
return live
"#;
        let encoded: Vec<String> = self
            .pool
            .eval(
                format!("{}{LIST}", Self::REAP_ANNOUNCEMENTS),
                vec![
                    self.announcement_index(service_name),
                    self.key("services"),
                    self.key("service-names"),
                    self.key("announcement-expiry"),
                    self.key("announcement-global-revision"),
                ],
                vec![now_unix_ms.to_string(), Self::service_id(service_name)],
            )
            .await?;
        encoded
            .into_iter()
            .map(|value| serde_json::from_str(&value).map_err(Into::into))
            .collect()
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl DiscoveryStateStore for ValkeyStateStore {
    async fn put_announcement(
        &self,
        service_name: &str,
        endpoint: AnnouncedEndpoint,
    ) -> Result<PutResult> {
        use fred::prelude::*;

        const PUT: &str = r#"
reap(KEYS[6], KEYS[3], KEYS[4], KEYS[5], ARGV[7])
local current = redis.call('GET', KEYS[1])
if not current and redis.call('ZCARD', KEYS[6]) >= tonumber(ARGV[8]) then
  return redis.error_reply('Discovery Valkey announcement capacity exhausted')
end
if current then
  local ok, decoded = pcall(cjson.decode, current)
  if not ok then return redis.error_reply('corrupt Discovery announcement') end
  local old_epoch = tonumber(decoded.accepted_state_epoch) or 0
  local old_exp = tonumber(decoded.expires_at_unix_ms) or 0
  local new_epoch = tonumber(ARGV[2])
  local new_exp = tonumber(ARGV[3])
  if old_epoch > new_epoch or (old_epoch == new_epoch and old_exp > new_exp) then
    return 0
  end
end
redis.call('SET', KEYS[1], ARGV[1], 'PXAT', ARGV[4])
redis.call('SADD', KEYS[2], KEYS[1])
redis.call('SADD', KEYS[3], ARGV[5])
redis.call('HSET', KEYS[4], ARGV[5], ARGV[6])
redis.call('INCR', KEYS[5])
redis.call('ZADD', KEYS[6], ARGV[4], KEYS[1])
return 1
"#;
        let service_id = Self::service_id(service_name);
        let encoded = serde_json::to_string(&endpoint)?;
        let stored: i64 = self
            .pool
            .eval(
                format!("{}{PUT}", Self::REAP_ANNOUNCEMENTS),
                vec![
                    self.announcement_key(service_name, &endpoint.socket_kind),
                    self.announcement_index(service_name),
                    self.key("services"),
                    self.key("service-names"),
                    self.key("announcement-global-revision"),
                    self.key("announcement-expiry"),
                ],
                vec![
                    encoded,
                    endpoint.accepted_state_epoch.to_string(),
                    endpoint.expires_at_unix_ms.to_string(),
                    endpoint
                        .live_until_unix_ms
                        .min(endpoint.expires_at_unix_ms)
                        .to_string(),
                    service_id,
                    service_name.to_owned(),
                    unix_millis_now().to_string(),
                    self.announcement_capacity.to_string(),
                ],
            )
            .await?;
        Ok(if stored == 1 {
            PutResult::Stored
        } else {
            PutResult::IgnoredOlder
        })
    }

    async fn announcements_for(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Result<Vec<AnnouncedEndpoint>> {
        self.announcements_for_inner(service_name, now_unix_ms)
            .await
    }

    async fn all_announcements(
        &self,
        now_unix_ms: i64,
    ) -> Result<Vec<(String, Vec<AnnouncedEndpoint>)>> {
        use fred::prelude::*;

        // One transaction returns names and values from the capacity-bounded
        // expiry index. Reap and refresh still serialize, and a listing never
        // labels an L1 snapshot or performs round trips per service.
        const LIST: &str = r#"
reap(KEYS[1], KEYS[2], KEYS[3], KEYS[4], ARGV[1])
local values = {}
for _, key in ipairs(redis.call('ZRANGE', KEYS[1], 0, -1)) do
  local value = redis.call('GET', key)
  if value then
    local service = string.match(key, ':announcement:([^:]+):[^:]+$')
    local name = redis.call('HGET', KEYS[3], service)
    if not name then return redis.error_reply('missing Discovery service name') end
    table.insert(values, {name, value})
  end
end
return values
"#;
        let encoded: Vec<(String, String)> = self
            .pool
            .eval(
                format!("{}{LIST}", Self::REAP_ANNOUNCEMENTS),
                vec![
                    self.key("announcement-expiry"),
                    self.key("services"),
                    self.key("service-names"),
                    self.key("announcement-global-revision"),
                ],
                vec![now_unix_ms.to_string()],
            )
            .await?;
        let mut all: HashMap<String, Vec<AnnouncedEndpoint>> = HashMap::new();
        for (name, value) in encoded {
            let endpoint: AnnouncedEndpoint = serde_json::from_str(&value)?;
            if endpoint.is_live_at(now_unix_ms) {
                all.entry(name).or_default().push(endpoint);
            }
        }
        Ok(all.into_iter().collect())
    }

    async fn put_liveness(&self, node: &Did, value: LiveAllocatable) -> Result<PutResult> {
        use fred::prelude::*;

        const PUT: &str = r#"
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local current = redis.call('GET', KEYS[1])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
if not current and redis.call('ZCARD', KEYS[3]) >= tonumber(ARGV[3]) then
  return redis.error_reply('Discovery Valkey liveness capacity exhausted')
end
local expiry = now + tonumber(ARGV[2])
-- Preserve JSON arrays/numeric payload exactly: Lua cjson turns [] into {}.
local encoded = string.sub(ARGV[1], 1, -2) .. ',"last_seen":' .. string.format('%.0f', now)
  .. ',"live_until_unix_ms":' .. string.format('%.0f', expiry) .. '}'
redis.call('SET', KEYS[1], encoded, 'PXAT', expiry)
redis.call('INCR', KEYS[2])
redis.call('ZADD', KEYS[3], expiry, KEYS[1])
return 1
"#;
        let node_id = Self::digest(node.as_str());
        let stored: i64 = self
            .pool
            .eval(
                PUT,
                vec![
                    format!("{}:liveness:{node_id}", self.prefix),
                    self.key("liveness-global-revision"),
                    self.key("liveness-expiry"),
                ],
                vec![
                    serde_json::to_string(&serde_json::json!({
                        "node": node, "allocatable": value.allocatable,
                        "load_fraction": value.load_fraction, "shared_receipt": true,
                    }))?,
                    value.live_until_unix_ms.saturating_sub(value.last_seen).clamp(1, 45_000).to_string(),
                    self.liveness_capacity.to_string(),
                ],
            )
            .await?;
        Ok(if stored == 1 {
            PutResult::Stored
        } else {
            PutResult::IgnoredOlder
        })
    }

    #[cfg(test)]
    async fn liveness(&self, node: &Did, _now_unix_ms: i64) -> Result<Option<LiveAllocatable>> {
        use fred::prelude::*;
        let encoded: Option<String> = self
            .pool
            .get(format!(
                "{}:liveness:{}",
                self.prefix,
                Self::digest(node.as_str())
            ))
            .await?;
        Ok(encoded
            .map(|encoded| serde_json::from_str::<StoredLiveness>(&encoded))
            .transpose()?.filter(|record| record.shared_receipt).map(|record| record.value))
    }

    async fn all_liveness(&self, _now_unix_ms: i64) -> Result<Vec<(Did, LiveAllocatable)>> {
        use fred::prelude::*;
        // The expiry index is capacity-bounded on every write. Enumerate it
        // atomically with the values so concurrent expiry/replacement cannot
        // leave a process-local candidate seed out of sync with shared state.
        const LIST: &str = r#"
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local values = {}
for _, key in ipairs(redis.call('ZRANGE', KEYS[1], 0, -1)) do
  local value = redis.call('GET', key)
  if value then
    local record = cjson.decode(value)
    if record.shared_receipt == true then
      table.insert(values, value)
    else
      -- Legacy replica-clock rows have no trustworthy shared lifetime. Retire
      -- only this volatile row; the next admitted heartbeat recreates it.
      redis.call('DEL', key)
      redis.call('ZREM', KEYS[1], key)
      redis.call('INCR', KEYS[2])
    end
  end
end
return values
"#;
        let encoded: Vec<String> = self
            .pool
            .eval(
                LIST,
                vec![self.key("liveness-expiry"), self.key("liveness-global-revision")],
                Vec::<String>::new(),
            )
            .await?;
        let mut live = Vec::with_capacity(encoded.len());
        for value in encoded {
            let record: StoredLiveness = serde_json::from_str(&value)?;
            live.push((record.node, record.value));
        }
        Ok(live)
    }

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()> {
        use fred::prelude::*;
        const PUT: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 and redis.call('SCARD', KEYS[2]) >= tonumber(ARGV[4]) then
  local oldest = redis.call('ZRANGE', KEYS[5], 0, 0)[1]
  if not oldest then return redis.error_reply('Discovery issuer cache index is inconsistent') end
  local oldkey = string.gsub(KEYS[1], ':entity:[^:]+$', ':entity:' .. oldest)
  redis.call('DEL', oldkey)
  redis.call('SREM', KEYS[2], oldest)
  redis.call('HDEL', KEYS[3], oldest)
  redis.call('ZREM', KEYS[5], oldest)
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[2], ARGV[2])
redis.call('HSET', KEYS[3], ARGV[2], ARGV[3])
redis.call('INCR', KEYS[4])
redis.call('ZADD', KEYS[5], ARGV[5], ARGV[2])
return 1
"#;
        let id = Self::digest(issuer);
        let _: i64 = self
            .pool
            .eval(
                PUT,
                vec![
                    format!("{}:entity:{id}", self.prefix),
                    self.key("issuers"),
                    self.key("issuer-names"),
                    self.key("entity-global-revision"),
                    self.key("issuer-fetched"),
                ],
                vec![
                    serde_json::to_string(&value)?,
                    id,
                    issuer.to_owned(),
                    self.artifact_capacity.to_string(),
                    value.fetched_at.to_string(),
                ],
            )
            .await?;
        Ok(())
    }

    async fn entity_statement(&self, issuer: &str) -> Result<Option<CachedEntityStatement>> {
        use fred::prelude::*;
        let encoded: Option<String> = self
            .pool
            .get(format!("{}:entity:{}", self.prefix, Self::digest(issuer)))
            .await?;
        encoded
            .map(|encoded| serde_json::from_str(&encoded).map_err(Into::into))
            .transpose()
    }

    async fn known_issuers(&self) -> Result<Vec<String>> {
        use fred::prelude::*;
        Ok(self.pool.hvals(self.key("issuer-names")).await?)
    }

    async fn known_issuer_count(&self) -> Result<usize> {
        use fred::prelude::*;
        Ok(self.pool.scard(self.key("issuers")).await?)
    }

    async fn put_envelope_keyset(
        &self,
        service_did: &str,
        value: CachedEnvelopeKeyset,
    ) -> Result<()> {
        use fred::prelude::*;
        const PUT: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 and redis.call('SCARD', KEYS[3]) >= tonumber(ARGV[2]) then
  return redis.error_reply('Discovery Valkey federation artifact capacity exhausted')
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('INCR', KEYS[2])
redis.call('SADD', KEYS[3], KEYS[1])
return 1
"#;
        let id = Self::digest(service_did);
        let _: i64 = self
            .pool
            .eval(
                PUT,
                vec![
                    format!("{}:envelope:{id}", self.prefix),
                    format!("{}:envelope-revision:{id}", self.prefix),
                    self.key("envelopes"),
                ],
                vec![
                    serde_json::to_string(&value)?,
                    self.artifact_capacity.to_string(),
                ],
            )
            .await?;
        Ok(())
    }

    async fn envelope_keyset(&self, service_did: &str) -> Result<Option<CachedEnvelopeKeyset>> {
        use fred::prelude::*;
        let encoded: Option<String> = self
            .pool
            .get(format!(
                "{}:envelope:{}",
                self.prefix,
                Self::digest(service_did)
            ))
            .await?;
        encoded
            .map(|encoded| serde_json::from_str(&encoded).map_err(Into::into))
            .transpose()
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
#[derive(Clone, Copy)]
struct ObservedRevision {
    revision: u64,
    valid_until_unix_ms: i64,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
struct TieredStateStore {
    memory: Arc<MemoryStateStore>,
    valkey: Arc<ValkeyStateStore>,
    l1_max_ttl_ms: i64,
    observed: Mutex<HashMap<String, ObservedRevision>>,
    // Serialize cache contents/revision labels within a scope, while unrelated
    // network reads and heartbeats can use the pool concurrently. Fixed storage
    // avoids accumulating a mutex for every historical identity.
    operations: [tokio::sync::Mutex<()>; 64],
}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
impl TieredStateStore {
    fn new(
        memory: Arc<MemoryStateStore>,
        valkey: Arc<ValkeyStateStore>,
        l1_max_ttl_ms: u64,
    ) -> Self {
        Self {
            memory,
            valkey,
            l1_max_ttl_ms: i64::try_from(l1_max_ttl_ms).unwrap_or(i64::MAX),
            observed: Mutex::new(HashMap::new()),
            operations: std::array::from_fn(|_| tokio::sync::Mutex::new(())),
        }
    }

    fn scope(kind: &str, value: &str) -> String {
        format!("{kind}:{value}")
    }

    fn operation(&self, kind: &str, value: &str) -> &tokio::sync::Mutex<()> {
        let digest = blake3::hash(Self::scope(kind, value).as_bytes());
        &self.operations[usize::from(digest.as_bytes()[0]) % self.operations.len()]
    }

    fn is_observed(&self, scope: &str, revision: u64, now_unix_ms: i64) -> bool {
        self.observed.lock().get(scope).is_some_and(|observed| {
            observed.revision == revision && now_unix_ms < observed.valid_until_unix_ms
        })
    }

    fn observe(&self, scope: String, revision: u64, now_unix_ms: i64) {
        let mut observed = self.observed.lock();
        observed.retain(|_, value| now_unix_ms < value.valid_until_unix_ms);
        let capacity = self
            .memory
            .announcement_capacity
            .saturating_add(self.memory.liveness_capacity)
            .saturating_add(self.memory.artifact_capacity.saturating_mul(2));
        if observed.len() >= capacity {
            observed.clear();
        }
        observed.insert(
            scope,
            ObservedRevision {
                revision,
                valid_until_unix_ms: now_unix_ms.saturating_add(self.l1_max_ttl_ms),
            },
        );
    }

    fn l1_announcement(
        &self,
        mut endpoint: AnnouncedEndpoint,
        now_unix_ms: i64,
    ) -> AnnouncedEndpoint {
        endpoint.live_until_unix_ms = endpoint
            .live_until_unix_ms
            .min(now_unix_ms.saturating_add(self.l1_max_ttl_ms));
        endpoint
    }

}

#[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
#[async_trait]
impl DiscoveryStateStore for TieredStateStore {
    async fn put_announcement(
        &self,
        service_name: &str,
        endpoint: AnnouncedEndpoint,
    ) -> Result<PutResult> {
        let _operation = self.operation("announcement", service_name).lock().await;
        // Invalidate before the command, including on ambiguous write errors
        // or cancellation. Never tag a caller's value with a later revision.
        self.observed
            .lock()
            .remove(&Self::scope("announcement", service_name));
        self.memory.clear_announcements_sync(service_name);
        self.valkey.put_announcement(service_name, endpoint).await
    }

    async fn announcements_for(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Result<Vec<AnnouncedEndpoint>> {
        let _operation = self.operation("announcement", service_name).lock().await;
        let scope = Self::scope("announcement", service_name);
        let revision = self.valkey.announcement_revision().await?;
        if self.is_observed(&scope, revision, now_unix_ms) {
            return self
                .memory
                .announcements_for(service_name, now_unix_ms)
                .await;
        }
        let values = self
            .valkey
            .announcements_for(service_name, now_unix_ms)
            .await?;
        self.observed.lock().remove(&scope);
        self.memory.clear_announcements_sync(service_name);
        // L1 admission is optional; only a complete nonempty result is marked
        // observed. Oversized scopes are served from L2 without partial hits.
        for value in &values {
            if self
                .memory
                .put_announcement(
                    service_name,
                    self.l1_announcement(value.clone(), now_unix_ms),
                )
                .await
                .is_err()
            {
                self.memory.clear_announcements_sync(service_name);
                return Ok(values);
            }
        }
        if !values.is_empty() {
            self.observe(scope, revision, now_unix_ms);
        }
        Ok(values)
    }

    async fn all_announcements(
        &self,
        now_unix_ms: i64,
    ) -> Result<Vec<(String, Vec<AnnouncedEndpoint>)>> {
        // A bounded L1 is not a complete replica of L2. Listings always read
        // authoritative indexes and cannot invalidate point-cache snapshots.
        self.valkey.all_announcements(now_unix_ms).await
    }

    async fn put_liveness(&self, node: &Did, value: LiveAllocatable) -> Result<PutResult> {
        let _operation = self.operation("liveness", node.as_str()).lock().await;
        self.observed
            .lock()
            .remove(&Self::scope("liveness", node.as_str()));
        self.memory.clear_liveness_sync(node);
        self.valkey.put_liveness(node, value).await
    }

    #[cfg(test)]
    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>> {
        let _operation = self.operation("liveness", node.as_str()).lock().await;
        // One authoritative GET is cheaper than revision + value on a miss,
        // and leaves expiry entirely on the shared clock, even with host skew.
        self.valkey.liveness(node, now_unix_ms).await
    }

    async fn all_liveness(&self, now_unix_ms: i64) -> Result<Vec<(Did, LiveAllocatable)>> {
        self.valkey.all_liveness(now_unix_ms).await
    }

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()> {
        let _operation = self.operation("entity", issuer).lock().await;
        self.observed.lock().remove(&Self::scope("entity", issuer));
        self.memory.clear_entity_statement_sync(issuer);
        self.valkey.put_entity_statement(issuer, value).await
    }

    async fn entity_statement(&self, issuer: &str) -> Result<Option<CachedEntityStatement>> {
        let _operation = self.operation("entity", issuer).lock().await;
        let now = unix_millis_now();
        let scope = Self::scope("entity", issuer);
        let revision = self.valkey.entity_revision().await?;
        if self.is_observed(&scope, revision, now) {
            // A fill of a different issuer can evict this L1 entry without a
            // shared write. Missing L1 data is a miss, never cached absence.
            if let Some(value) = self.memory.entity_statement(issuer).await? {
                return Ok(Some(value));
            }
        }
        let value = self.valkey.entity_statement(issuer).await?;
        self.observed.lock().remove(&scope);
        self.memory.clear_entity_statement_sync(issuer);
        if let Some(value) = &value {
            if self
                .memory
                .put_entity_statement(issuer, value.clone())
                .await
                .is_ok()
            {
                self.observe(scope, revision, now);
            }
        }
        Ok(value)
    }

    async fn known_issuers(&self) -> Result<Vec<String>> {
        self.valkey.known_issuers().await
    }

    async fn known_issuer_count(&self) -> Result<usize> {
        self.valkey.known_issuer_count().await
    }

    async fn put_envelope_keyset(
        &self,
        service_did: &str,
        value: CachedEnvelopeKeyset,
    ) -> Result<()> {
        let _operation = self.operation("envelope", service_did).lock().await;
        self.observed
            .lock()
            .remove(&Self::scope("envelope", service_did));
        self.memory.clear_envelope_keyset_sync(service_did);
        self.valkey.put_envelope_keyset(service_did, value).await
    }

    async fn envelope_keyset(&self, service_did: &str) -> Result<Option<CachedEnvelopeKeyset>> {
        let _operation = self.operation("envelope", service_did).lock().await;
        let now = unix_millis_now();
        let scope = Self::scope("envelope", service_did);
        let revision = self.valkey.envelope_revision(service_did).await?;
        if self.is_observed(&scope, revision, now) {
            return self.memory.envelope_keyset(service_did).await;
        }
        let value = self.valkey.envelope_keyset(service_did).await?;
        self.observed.lock().remove(&scope);
        self.memory.clear_envelope_keyset_sync(service_did);
        if let Some(value) = &value {
            if self
                .memory
                .put_envelope_keyset(service_did, value.clone())
                .await
                .is_ok()
            {
                self.observe(scope, revision, now);
            }
        }
        Ok(value)
    }
}

pub(crate) fn unix_millis_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::indexing_slicing, clippy::unwrap_used)]

    use super::*;

    fn endpoint(kind: &str, epoch: u64, signed_expiry: i64, live_until: i64) -> AnnouncedEndpoint {
        AnnouncedEndpoint {
            socket_kind: kind.to_owned(),
            endpoint: format!("iroh://{kind}"),
            service_jwt: "jwt".to_owned(),
            service_did: Did::new("did:at9p:test".to_owned()),
            capabilities: BTreeSet::from(["discovery".to_owned()]),
            accepted_state_digest: vec![7; 64],
            accepted_state_epoch: epoch,
            response_key_id: "did:at9p:test#response".to_owned(),
            request_kem_key_id: "did:at9p:test#kem".to_owned(),
            request_kem_recipient: vec![1, 2, 3],
            expires_at_unix_ms: signed_expiry,
            source_signer: [9; 32],
            live_until_unix_ms: live_until,
        }
    }

    async fn assert_announcement_backend_contract(store: &dyn DiscoveryStateStore) {
        let now = unix_millis_now();
        store
            .put_announcement("model", endpoint("iroh", 1, now + 10_000, now + 50))
            .await
            .unwrap();
        assert_eq!(
            store.announcements_for("model", now).await.unwrap().len(),
            1
        );
        store
            .put_announcement("model", endpoint("iroh", 2, now + 10_000, now + 50))
            .await
            .unwrap();
        assert_eq!(
            store.announcements_for("model", now).await.unwrap()[0].accepted_state_epoch,
            2
        );

        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        assert!(store
            .announcements_for("model", unix_millis_now())
            .await
            .unwrap()
            .is_empty());

        let now = unix_millis_now();
        store
            .put_announcement("policy", endpoint("iroh", 1, now + 10_000, now + 5_000))
            .await
            .unwrap();
        let error = store
            .put_announcement("other", endpoint("iroh", 1, now + 10_000, now + 5_000))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("capacity exhausted"));
    }

    #[tokio::test]
    async fn memory_satisfies_announcement_backend_contract() {
        assert_announcement_backend_contract(&MemoryStateStore::new(1, 1, 1)).await;
    }

    async fn assert_receipt_ordered_liveness(store: &dyn DiscoveryStateStore) {
        let node = Did::new("did:web:clock-rollback.example".to_owned());
        let now = unix_millis_now();
        let mut value = LiveAllocatable {
            allocatable: vec![("cpu".to_owned(), "1".to_owned())],
            load_fraction: 0.9,
            // Legacy client-clock record must not poison subsequent receipts.
            last_seen: now + 86_400_000,
            live_until_unix_ms: now + 100,
        };
        store.put_liveness(&node, value.clone()).await.unwrap();
        value.last_seen = now;
        value.load_fraction = 0.2;
        value.allocatable[0].1 = "8".to_owned();
        value.live_until_unix_ms = now + 500;
        assert_eq!(
            store.put_liveness(&node, value.clone()).await.unwrap(),
            PutResult::Stored
        );
        assert_eq!(
            store.liveness(&node, now + 200).await.unwrap(),
            Some(value.clone())
        );
        let mut delayed = value.clone();
        delayed.last_seen = now - 1;
        delayed.live_until_unix_ms = now + 300;
        delayed.load_fraction = 1.0;
        assert_eq!(
            store.put_liveness(&node, delayed).await.unwrap(),
            PutResult::IgnoredOlder
        );
        assert_eq!(store.liveness(&node, now + 200).await.unwrap(), Some(value));
        assert!(store.liveness(&node, now + 501).await.unwrap().is_none());

        // Liveness refreshes do not renew a signed announcement's authority.
        store
            .put_announcement("clock", endpoint("iroh", 1, now + 50, now + 500))
            .await
            .unwrap();
        assert!(store
            .announcements_for("clock", now + 51)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn memory_heartbeat_receipt_survives_clock_rollback() {
        assert_receipt_ordered_liveness(&MemoryStateStore::new(8, 8, 8)).await;
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_heartbeat_receipt_survives_clock_rollback() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        use fred::prelude::*;
        let config = valkey_config(url, "rollback");
        let shared = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let other = ValkeyStateStore::connect(&config).await.unwrap();
        let tiered = TieredStateStore::new(Arc::new(MemoryStateStore::new(8, 8, 8)), shared.clone(), 1000);
        let node = Did::new("did:web:skew.example".to_owned());
        let now = unix_millis_now();
        let mut value = LiveAllocatable {
            allocatable: vec![], load_fraction: 0.9,
            last_seen: now + 86_400_000, live_until_unix_ms: now + 86_400_400,
        };
        tiered.put_liveness(&node, value.clone()).await.unwrap();
        let first = tiered.liveness(&node, i64::MAX).await.unwrap().unwrap();
        assert!((first.last_seen - now).abs() < 5000);
        value.last_seen = now - 86_400_000;
        value.live_until_unix_ms = value.last_seen + 400;
        value.load_fraction = 0.2;
        value.allocatable = vec![("cpu".to_owned(), "8".to_owned())];
        assert_eq!(other.put_liveness(&node, value).await.unwrap(), PutResult::Stored);
        let latest = tiered.liveness(&node, i64::MAX).await.unwrap().unwrap();
        assert!(latest.last_seen >= first.last_seen);
        assert_eq!(latest.load_fraction, 0.2);
        assert_eq!(latest.allocatable[0].1, "8");
        assert_eq!(latest.live_until_unix_ms - latest.last_seen, 400);
        assert_eq!(tiered.all_liveness(i64::MAX).await.unwrap().len(), 1);
        let ttl: i64 = shared.pool.pttl(format!("{}:liveness:{}", shared.prefix, ValkeyStateStore::digest(node.as_str()))).await.unwrap();
        assert!((1..=400).contains(&ttl), "expiry uses shared receipt, not ahead host time");
        tokio::time::sleep(std::time::Duration::from_millis(450)).await;
        assert!(tiered.liveness(&node, i64::MIN).await.unwrap().is_none());
        assert!(tiered.all_liveness(i64::MIN).await.unwrap().is_empty());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn rediss_uses_tls_and_rejects_plaintext_valkey() {
        use fred::prelude::RedisConfig;
        hyprstream_rpc::transport::install_pq_crypto_provider().unwrap();
        assert!(RedisConfig::from_url("rediss://localhost:6379")
            .unwrap()
            .tls
            .is_some());
        assert!(RedisConfig::from_url("redis://localhost:6379")
            .unwrap()
            .tls
            .is_none());
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let mut config = valkey_config(url.clone(), "tls");
        let plaintext = ValkeyStateStore::connect(&config).await.unwrap();
        assert!(plaintext
            .all_announcements(unix_millis_now())
            .await
            .unwrap()
            .is_empty());
        config.url = url.replacen("redis://", "rediss://", 1);
        let start = std::time::Instant::now();
        assert!(
            ValkeyStateStore::connect(&config).await.is_err(),
            "TLS must not downgrade to plaintext"
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(3));
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_batched_listing_preserves_names_values_and_cleanup() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let mut config = valkey_config(url, "batched-list");
        config.announcement_capacity = 512;
        let writer = ValkeyStateStore::connect(&config).await.unwrap();
        let reader = TieredStateStore::new(
            Arc::new(MemoryStateStore::new(1, 1, 1)),
            Arc::new(ValkeyStateStore::connect(&config).await.unwrap()),
            10_000,
        );
        let now = unix_millis_now();
        for i in 0..256 {
            let name = format!("service:{i}:λ");
            for kind in ["iroh", "quic"] {
                writer
                    .put_announcement(&name, endpoint(kind, 1, now + 60_000, now + 60_000))
                    .await
                    .unwrap();
            }
        }
        assert_eq!(
            reader
                .announcements_for("service:0:λ", now)
                .await
                .unwrap()
                .len(),
            2
        );
        let entries = reader.all_announcements(now).await.unwrap();
        assert_eq!(entries.len(), 256);
        assert!(entries
            .iter()
            .all(|(name, values)| name.starts_with("service:") && values.len() == 2));
        writer
            .put_announcement(
                "service:0:λ",
                endpoint("iroh", 2, now + 90_000, now + 90_000),
            )
            .await
            .unwrap();
        let remaining = reader.all_announcements(now + 60_001).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].0, "service:0:λ");
        assert_eq!(remaining[0].1.len(), 1);
        assert_eq!(remaining[0].1[0].accepted_state_epoch, 2);
        // Reaping restores live capacity and cannot populate/poison point L1.
        writer
            .put_announcement("new", endpoint("iroh", 1, now + 90_000, now + 90_000))
            .await
            .unwrap();
        assert_eq!(
            reader
                .announcements_for("service:0:λ", now + 60_001)
                .await
                .unwrap()[0]
                .accepted_state_epoch,
            2
        );
        assert_eq!(
            reader.all_announcements(now + 60_001).await.unwrap().len(),
            2
        );
    }

    #[tokio::test]
    async fn memory_replaces_by_key_without_fleet_scan() {
        let store = MemoryStateStore::new(8, 8, 8);
        let now = unix_millis_now();
        store
            .put_announcement("model", endpoint("iroh", 1, now + 10_000, now + 1_000))
            .await
            .unwrap();
        store
            .put_announcement("model", endpoint("iroh", 2, now + 20_000, now + 2_000))
            .await
            .unwrap();
        let entries = store.announcements_for("model", now).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].accepted_state_epoch, 2);
    }

    #[tokio::test]
    async fn memory_rejects_delayed_older_announcement() {
        let store = MemoryStateStore::new(8, 8, 8);
        let now = unix_millis_now();
        store
            .put_announcement("model", endpoint("iroh", 3, now + 20_000, now + 2_000))
            .await
            .unwrap();
        assert_eq!(
            store
                .put_announcement("model", endpoint("iroh", 2, now + 30_000, now + 3_000))
                .await
                .unwrap(),
            PutResult::IgnoredOlder
        );
        assert_eq!(
            store.announcements_for("model", now).await.unwrap()[0].accepted_state_epoch,
            3
        );
    }

    #[tokio::test]
    async fn memory_expiry_removes_secondary_index_entry() {
        let store = MemoryStateStore::new(8, 8, 8);
        let now = unix_millis_now();
        store
            .put_announcement("model", endpoint("iroh", 1, now + 10_000, now + 1))
            .await
            .unwrap();
        assert!(store.all_announcements(now + 2).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn memory_capacity_rejects_instead_of_evicting_live_reach() {
        let store = MemoryStateStore::new(1, 1, 1);
        let now = unix_millis_now();
        store
            .put_announcement("model", endpoint("iroh", 1, now + 10_000, now + 1_000))
            .await
            .unwrap();
        let error = store
            .put_announcement("policy", endpoint("iroh", 1, now + 10_000, now + 1_000))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("capacity exhausted"));
    }

    #[tokio::test]
    async fn active_active_rejects_isolated_memory() {
        let config = DiscoveryStateConfig {
            active_active: true,
            ..DiscoveryStateConfig::default()
        };
        let error = match DiscoveryState::connect(&config).await {
            Ok(_) => panic!("active-active memory configuration was admitted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("requires valkey or tiered"));
    }

    #[tokio::test]
    async fn rejects_zero_memory_capacity_and_tiered_freshness() {
        let mut memory = DiscoveryStateConfig::default();
        memory.memory.announcement_capacity = 0;
        let error = match DiscoveryState::connect(&memory).await {
            Ok(_) => panic!("zero-capacity memory configuration was admitted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("capacities must be positive"));

        let mut tiered = DiscoveryStateConfig {
            backend: DiscoveryStateBackend::Tiered,
            ..DiscoveryStateConfig::default()
        };
        tiered.tiered.l1_max_ttl_ms = 0;
        let error = match DiscoveryState::connect(&tiered).await {
            Ok(_) => panic!("zero-freshness tiered configuration was admitted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("l1_max_ttl_ms must be positive"));
    }

    async fn assert_issuer_eviction_contract(store: &dyn DiscoveryStateStore) {
        for (issuer, fetched_at) in [
            ("old", 1),
            ("refresh", 2),
            ("keep", 3),
            ("refresh", 4),
            ("new", 5),
        ] {
            store
                .put_entity_statement(
                    issuer,
                    CachedEntityStatement {
                        jwt: format!("inert:{issuer}:{fetched_at}"),
                        fetched_at,
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(store.known_issuer_count().await.unwrap(), 3);
        let mut names = store.known_issuers().await.unwrap();
        names.sort();
        assert_eq!(names, ["keep", "new", "refresh"]);
        assert!(store.entity_statement("old").await.unwrap().is_none());
        assert_eq!(
            store
                .entity_statement("refresh")
                .await
                .unwrap()
                .unwrap()
                .fetched_at,
            4
        );
        store
            .put_entity_statement(
                "old",
                CachedEntityStatement {
                    jwt: "reinserted".to_owned(),
                    fetched_at: 6,
                },
            )
            .await
            .unwrap();
        assert!(store.entity_statement("keep").await.unwrap().is_none());
        assert_eq!(
            store.entity_statement("old").await.unwrap().unwrap().jwt,
            "reinserted"
        );
    }

    #[tokio::test]
    async fn memory_issuer_cache_evicts_oldest_and_accepts_reinsertion() {
        assert_issuer_eviction_contract(&MemoryStateStore::new(1, 1, 3)).await;
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_issuer_cache_churn_bounds_metadata_and_invalidates_other_l1() {
        use fred::prelude::*;
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let mut config = valkey_config(url, "issuer-churn");
        config.artifact_capacity = 3;
        let writer = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        assert_issuer_eviction_contract(writer.as_ref()).await;
        let reader = TieredStateStore::new(
            Arc::new(MemoryStateStore::new(1, 1, 1)),
            Arc::new(ValkeyStateStore::connect(&config).await.unwrap()),
            60_000,
        );
        // A smaller L1 must refetch entries evicted by other fills even if L2
        // has not changed. Previously an observed-but-missing entry returned None.
        for issuer in ["old", "refresh", "old"] {
            assert!(reader.entity_statement(issuer).await.unwrap().is_some());
        }
        let generation = writer.entity_revision().await.unwrap();
        for i in 0..96 {
            writer
                .put_entity_statement(
                    &format!("issuer-{i}"),
                    CachedEntityStatement {
                        jwt: format!("inert-{i}"),
                        fetched_at: 100 + i,
                    },
                )
                .await
                .unwrap();
            assert_eq!(writer.known_issuer_count().await.unwrap(), 3);
        }
        assert!(reader.entity_statement("old").await.unwrap().is_none());
        let counts: Vec<usize> = writer.pool.eval(
            "return {redis.call('SCARD', KEYS[1]), redis.call('HLEN', KEYS[2]), redis.call('ZCARD', KEYS[3]), #redis.call('KEYS', ARGV[1]), #redis.call('KEYS', ARGV[2])}",
            vec![writer.key("issuers"), writer.key("issuer-names"), writer.key("issuer-fetched")],
            vec![writer.key("*"), writer.key("*revision*")],
        ).await.unwrap();
        assert_eq!(counts, [3, 3, 3, 7, 1]);
        assert!(writer.entity_revision().await.unwrap() > generation);
        writer
            .put_entity_statement(
                "old",
                CachedEntityStatement {
                    jwt: "new-generation".to_owned(),
                    fetched_at: 1000,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            reader.entity_statement("old").await.unwrap().unwrap().jwt,
            "new-generation"
        );
        // Retain a positive old snapshot across eviction AND reinsertion (ABA).
        for i in 0..3 {
            writer
                .put_entity_statement(
                    &format!("replacement-{i}"),
                    CachedEntityStatement {
                        jwt: "other".to_owned(),
                        fetched_at: 2000 + i,
                    },
                )
                .await
                .unwrap();
        }
        writer
            .put_entity_statement(
                "old",
                CachedEntityStatement {
                    jwt: "after-aba".to_owned(),
                    fetched_at: 3000,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            reader.entity_statement("old").await.unwrap().unwrap().jwt,
            "after-aba"
        );
        assert!(reader.observed.lock().len() <= 4);
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_issuer_listing_and_count_return_complete_bounded_cache() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let mut config = valkey_config(url, "issuer-list-count");
        config.artifact_capacity = 128;
        let writer = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let reader = TieredStateStore::new(
            Arc::new(MemoryStateStore::new(1, 1, 1)),
            writer.clone(),
            60_000,
        );
        let mut expected = Vec::new();
        for i in 0..256 {
            let name = format!("https://issuer:{i}:λ.example");
            writer
                .put_entity_statement(
                    &name,
                    CachedEntityStatement {
                        jwt: "inert".to_owned(),
                        fetched_at: i,
                    },
                )
                .await
                .unwrap();
            if i >= 128 {
                expected.push(name);
            }
        }
        expected.sort();
        for _ in 0..2 {
            let mut actual = reader.known_issuers().await.unwrap();
            actual.sort();
            assert_eq!(actual, expected);
            assert_eq!(reader.known_issuer_count().await.unwrap(), 128);
        }
        assert!(reader.memory.inner.lock().entity_statements.is_empty());
        assert!(reader.observed.lock().is_empty());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    fn valkey_config(url: String, suffix: &str) -> ValkeyStateConfig {
        ValkeyStateConfig {
            url,
            key_prefix: format!(
                "hs-test-{}-{}-{suffix}",
                std::process::id(),
                unix_millis_now()
            ),
            pool_size: 2,
            announcement_capacity: 64,
            liveness_capacity: 64,
            artifact_capacity: 64,
            command_timeout_ms: 250,
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_satisfies_announcement_backend_contract() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let mut config = valkey_config(url, "contract");
        config.announcement_capacity = 1;
        let store = ValkeyStateStore::connect(&config).await.unwrap();
        assert_announcement_backend_contract(&store).await;
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_two_replicas_share_all_discovery_state() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let config = valkey_config(url, "shared");
        let replica_a = ValkeyStateStore::connect(&config).await.unwrap();
        let replica_b = ValkeyStateStore::connect(&config).await.unwrap();
        let now = unix_millis_now();

        replica_a
            .put_announcement("model", endpoint("iroh", 1, now + 30_000, now + 20_000))
            .await
            .unwrap();
        assert_eq!(
            replica_b
                .announcements_for("model", now)
                .await
                .unwrap()
                .len(),
            1
        );

        replica_b
            .put_liveness(
                &Did::new("did:web:node.example".to_owned()),
                LiveAllocatable {
                    allocatable: vec![("cpu".to_owned(), "8".to_owned())],
                    load_fraction: 0.25,
                    last_seen: now,
                    live_until_unix_ms: now + 20_000,
                },
            )
            .await
            .unwrap();
        assert!(replica_a
            .liveness(&Did::new("did:web:node.example".to_owned()), now)
            .await
            .unwrap()
            .is_some());

        replica_a
            .put_entity_statement(
                "https://issuer.example",
                CachedEntityStatement {
                    jwt: "statement".to_owned(),
                    fetched_at: now / 1_000,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            replica_b
                .entity_statement("https://issuer.example")
                .await
                .unwrap()
                .unwrap()
                .jwt,
            "statement"
        );

        replica_b
            .put_envelope_keyset(
                "did:web:service.example",
                CachedEnvelopeKeyset {
                    cose_keyset_cbor: vec![1, 2, 3],
                    fetched_at: now / 1_000,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            replica_a
                .envelope_keyset("did:web:service.example")
                .await
                .unwrap()
                .unwrap()
                .cose_keyset_cbor,
            vec![1, 2, 3]
        );

        // A newly connected replica sees the same state after process-local
        // state is discarded.
        let restarted = ValkeyStateStore::connect(&config).await.unwrap();
        assert_eq!(
            restarted
                .announcements_for("model", now)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_rejects_delayed_old_update_and_expires_server_side() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let config = valkey_config(url, "ordering");
        let store = ValkeyStateStore::connect(&config).await.unwrap();
        let now = unix_millis_now();
        store
            .put_announcement("model", endpoint("iroh", 4, now + 30_000, now + 20_000))
            .await
            .unwrap();
        assert_eq!(
            store
                .put_announcement("model", endpoint("iroh", 3, now + 40_000, now + 20_000))
                .await
                .unwrap(),
            PutResult::IgnoredOlder
        );
        assert_eq!(
            store.announcements_for("model", now).await.unwrap()[0].accepted_state_epoch,
            4
        );

        store
            .put_announcement("short", endpoint("iroh", 1, now + 10_000, now + 50))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        assert!(store
            .announcements_for("short", unix_millis_now())
            .await
            .unwrap()
            .is_empty());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn tiered_revision_check_invalidates_other_replica_l1() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let config = valkey_config(url, "tiered");
        let shared_a = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let shared_b = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let tier_a =
            TieredStateStore::new(Arc::new(MemoryStateStore::new(8, 8, 8)), shared_a, 10_000);
        let tier_b =
            TieredStateStore::new(Arc::new(MemoryStateStore::new(8, 8, 8)), shared_b, 10_000);
        let now = unix_millis_now();
        tier_a
            .put_announcement("model", endpoint("iroh", 1, now + 30_000, now + 20_000))
            .await
            .unwrap();
        assert_eq!(
            tier_b.announcements_for("model", now).await.unwrap()[0].accepted_state_epoch,
            1
        );
        tier_a
            .put_announcement("model", endpoint("iroh", 2, now + 40_000, now + 20_000))
            .await
            .unwrap();
        assert_eq!(
            tier_b.announcements_for("model", now).await.unwrap()[0].accepted_state_epoch,
            2,
            "L2 revision change must invalidate replica B's populated L1"
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn tiered_does_not_resurrect_expired_l2_data() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let config = valkey_config(url, "tiered-expiry");
        let shared = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let tiered =
            TieredStateStore::new(Arc::new(MemoryStateStore::new(8, 8, 8)), shared, 10_000);
        let now = unix_millis_now();
        tiered
            .put_announcement("model", endpoint("iroh", 1, now + 10_000, now + 50))
            .await
            .unwrap();
        assert_eq!(
            tiered.announcements_for("model", now).await.unwrap().len(),
            1
        );

        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        assert!(tiered
            .announcements_for("model", unix_millis_now())
            .await
            .unwrap()
            .is_empty());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn tiered_fails_closed_after_l2_client_shutdown() {
        use fred::prelude::ClientLike as _;

        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let config = valkey_config(url, "tiered-outage");
        let shared = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let tiered = TieredStateStore::new(
            Arc::new(MemoryStateStore::new(8, 8, 8)),
            Arc::clone(&shared),
            10_000,
        );
        let now = unix_millis_now();
        tiered
            .put_announcement("model", endpoint("iroh", 1, now + 30_000, now + 20_000))
            .await
            .unwrap();
        assert_eq!(
            tiered.announcements_for("model", now).await.unwrap().len(),
            1
        );

        shared.pool.quit().await.unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            tiered.announcements_for("model", unix_millis_now()),
        )
        .await
        .expect("configured command timeout must bound an L2 outage");
        assert!(
            result.is_err(),
            "tiered state must not return its populated L1"
        );
    }
    #[tokio::test]
    async fn memory_liveness_enumeration_excludes_expiry_and_remains_bounded() {
        let store = MemoryStateStore::new(1, 1, 1);
        let node = Did::new("did:web:live.example".to_owned());
        let now = unix_millis_now();
        for i in 0..100 {
            store
                .put_liveness(
                    &node,
                    LiveAllocatable {
                        allocatable: vec![],
                        load_fraction: 0.1,
                        last_seen: now + i,
                        live_until_unix_ms: now + 30_000,
                    },
                )
                .await
                .unwrap();
        }
        assert_eq!(store.all_liveness(now).await.unwrap()[0].0, node);
        assert!(store.inner.lock().expiry.len() <= 2);
        assert!(store.all_liveness(now + 30_000).await.unwrap().is_empty());
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_driver_survives_temporary_bootstrap_runtime() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        for backend in [DiscoveryStateBackend::Valkey, DiscoveryStateBackend::Tiered] {
            let config = DiscoveryStateConfig {
                backend,
                active_active: true,
                valkey: valkey_config(url.clone(), &format!("bootstrap-{backend:?}")),
                ..DiscoveryStateConfig::default()
            };
            // Match the synchronous production factory, including exiting its
            // thread and dropping its current-thread runtime before any use.
            let state = std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(DiscoveryState::connect(&config))
                    .unwrap()
            })
            .join()
            .unwrap();
            let store = state.into_inner();
            let now = unix_millis_now();
            store
                .put_announcement("model", endpoint("iroh", 1, now + 30_000, now + 20_000))
                .await
                .unwrap();
            assert_eq!(
                store.announcements_for("model", now).await.unwrap().len(),
                1
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            assert_eq!(
                store
                    .all_announcements(unix_millis_now())
                    .await
                    .unwrap()
                    .len(),
                1
            );
        }
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn tiered_scoped_operations_allow_unrelated_liveness_and_heartbeats_to_progress() {
        use futures::{stream, StreamExt};
        use std::time::Duration;
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let shared = Arc::new(
            ValkeyStateStore::connect(&valkey_config(url, "striped-concurrency"))
                .await
                .unwrap(),
        );
        let tier = TieredStateStore::new(
            Arc::new(MemoryStateStore::new(64, 64, 64)),
            shared.clone(),
            60_000,
        );
        let now = unix_millis_now();
        // Pick sixteen distinct stripes so the regression is deterministic,
        // including when the hash happens to collide for adjacent node names.
        let mut nodes: Vec<Did> = Vec::new();
        for i in 0..1024 {
            let node = Did::new(format!("did:web:concurrent-{i}.example"));
            if nodes.iter().all(|other| {
                !std::ptr::eq(
                    tier.operation("liveness", other.as_str()),
                    tier.operation("liveness", node.as_str()),
                )
            }) {
                nodes.push(node);
            }
            if nodes.len() == 16 {
                break;
            }
        }
        assert_eq!(nodes.len(), 16);
        for node in &nodes {
            shared
                .put_liveness(
                    node,
                    LiveAllocatable {
                        allocatable: vec![],
                        load_fraction: 0.1,
                        last_seen: now,
                        live_until_unix_ms: now + 60_000,
                    },
                )
                .await
                .unwrap();
        }
        // Hold one scope exactly as a delayed read/write does across its L2
        // awaits. A store-wide mutex prevents every other operation below from
        // finishing; striped ordering lets all fifteen real Valkey reads pass.
        let stalled = tier.operation("liveness", nodes[0].as_str()).lock().await;
        let mut reads = stream::iter(
            nodes
                .iter()
                .map(|node| {
                    let tier = &tier;
                    async move { (node, tier.liveness(node, now).await) }
                }),
        )
        .buffer_unordered(16);
        tokio::time::timeout(Duration::from_secs(2), async {
            for _ in 0..15 {
                let (node, result) = reads.next().await.unwrap();
                assert_ne!(node, &nodes[0]);
                assert_eq!(result.unwrap().unwrap().load_fraction, 0.1);
            }
        })
        .await
        .expect("unrelated tiered liveness reads serialized behind a stalled scope");
        assert!(
            tokio::time::timeout(Duration::from_millis(20), reads.next())
                .await
                .is_err()
        );

        let heartbeat = LiveAllocatable {
            allocatable: vec![],
            load_fraction: 0.8,
            last_seen: now + 1,
            live_until_unix_ms: now + 60_001,
        };
        tokio::time::timeout(
            Duration::from_secs(2),
            tier.put_liveness(&nodes[1], heartbeat.clone()),
        )
        .await
        .expect("unrelated heartbeat serialized behind a stalled scope")
        .unwrap();
        // Same-scope writes still wait. Cancelling a queued operation must not
        // corrupt the cache or leave the stripe permanently locked.
        assert!(tokio::time::timeout(
            Duration::from_millis(20),
            tier.put_liveness(&nodes[0], heartbeat.clone())
        )
        .await
        .is_err());
        drop(reads);
        drop(stalled);
        assert_eq!(
            tier.liveness(&nodes[0], now)
                .await
                .unwrap()
                .unwrap()
                .load_fraction,
            0.1
        );
        tier.put_liveness(&nodes[0], heartbeat).await.unwrap();
        assert_eq!(
            tier.liveness(&nodes[0], now)
                .await
                .unwrap()
                .unwrap()
                .load_fraction,
            0.8
        );
        assert_eq!(
            tier.liveness(&nodes[1], now)
                .await
                .unwrap()
                .unwrap()
                .load_fraction,
            0.8
        );
        assert_eq!(tier.operations.len(), 64);
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn tiered_capacity_never_limits_authoritative_results_or_writes() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let shared = Arc::new(
            ValkeyStateStore::connect(&valkey_config(url, "capacity"))
                .await
                .unwrap(),
        );
        let memory = Arc::new(MemoryStateStore::new(1, 1, 1));
        let tier = TieredStateStore::new(memory.clone(), shared, 10_000);
        let now = unix_millis_now();
        for name in ["a", "b", "c"] {
            tier.put_announcement(name, endpoint("iroh", 1, now + 30_000, now + 20_000))
                .await
                .unwrap();
            assert_eq!(tier.announcements_for(name, now).await.unwrap().len(), 1);
            let node = Did::new(format!("did:web:{name}.example"));
            tier.put_liveness(
                &node,
                LiveAllocatable {
                    allocatable: vec![],
                    load_fraction: 0.1,
                    last_seen: now,
                    live_until_unix_ms: now + 20_000,
                },
            )
            .await
            .unwrap();
            assert!(tier.liveness(&node, now).await.unwrap().is_some());
            tier.put_entity_statement(
                name,
                CachedEntityStatement {
                    jwt: name.to_owned(),
                    fetched_at: now,
                },
            )
            .await
            .unwrap();
            assert!(tier.entity_statement(name).await.unwrap().is_some());
            tier.put_envelope_keyset(
                name,
                CachedEnvelopeKeyset {
                    cose_keyset_cbor: vec![1],
                    fetched_at: now,
                },
            )
            .await
            .unwrap();
            assert!(tier.envelope_keyset(name).await.unwrap().is_some());
        }
        for _ in 0..2 {
            assert_eq!(tier.all_announcements(now).await.unwrap().len(), 3);
            assert_eq!(tier.all_liveness(now).await.unwrap().len(), 3);
            assert_eq!(tier.known_issuers().await.unwrap().len(), 3);
        }
        // A single service can also be larger than L1. Never mark a partial
        // result as observed, including after repeated declined cache fills.
        tier.put_announcement("a", endpoint("quic", 1, now + 30_000, now + 20_000))
            .await
            .unwrap();
        for i in 0..50 {
            assert_eq!(tier.announcements_for("a", now).await.unwrap().len(), 2);
            let absent = format!("missing-{i}");
            assert!(tier
                .announcements_for(&absent, now)
                .await
                .unwrap()
                .is_empty());
            assert!(tier.entity_statement(&absent).await.unwrap().is_none());
            assert!(tier.envelope_keyset(&absent).await.unwrap().is_none());
            assert!(tier
                .liveness(&Did::new(absent), now)
                .await
                .unwrap()
                .is_none());
        }
        let cache = memory.inner.lock();
        assert!(cache.announcements.len() <= 1);
        assert!(cache.liveness.len() <= 1);
        assert!(cache.entity_statements.len() <= 1);
        assert!(cache.envelope_keysets.len() <= 1);
        assert!(cache.expiry.len() <= 4);
        assert!(tier.observed.lock().len() <= 4);
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_expired_identity_churn_bounds_metadata_and_listing_work() {
        use fred::prelude::*;

        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let mut config = valkey_config(url, "metadata-churn");
        config.announcement_capacity = 4;
        config.liveness_capacity = 4;
        let store = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let tier = TieredStateStore::new(
            Arc::new(MemoryStateStore::new(4, 4, 4)),
            store.clone(),
            30_000,
        );
        let now = unix_millis_now();
        let anchor = Did::new("did:web:anchor.example".to_owned());
        store
            .put_announcement("anchor", endpoint("iroh", 1, now + 30_000, now + 30_000))
            .await
            .unwrap();
        store
            .put_liveness(
                &anchor,
                LiveAllocatable {
                    allocatable: vec![],
                    load_fraction: 0.1,
                    last_seen: now,
                    live_until_unix_ms: now + 30_000,
                },
            )
            .await
            .unwrap();
        tier.announcements_for("anchor", now).await.unwrap();
        tier.liveness(&anchor, now).await.unwrap();
        let announcement_generation = store.announcement_revision().await.unwrap();
        let liveness_generation = store.liveness_revision().await.unwrap();

        // No historical service is ever individually read. Each cohort expires
        // server-side before the next one; 36 distinct identities exceed the
        // live capacity nine times while the anchor's L1 snapshot stays warm.
        for cohort in 0..12 {
            let now = unix_millis_now();
            for slot in 0..3 {
                let name = format!("churn-{cohort}-{slot}");
                store
                    .put_announcement(&name, endpoint("iroh", 1, now + 100, now + 100))
                    .await
                    .unwrap();
                store
                    .put_liveness(
                        &Did::new(format!("did:web:{name}.example")),
                        LiveAllocatable {
                            allocatable: vec![],
                            load_fraction: 0.2,
                            last_seen: now,
                            live_until_unix_ms: now + 100,
                        },
                    )
                    .await
                    .unwrap();
            }
            // Inspect the backing namespace BEFORE any listing can clean it.
            // KEYS is test-only, confined to this test's unique prefix.
            let counts: Vec<usize> = store.pool.eval(
                "return {redis.call('SCARD', KEYS[1]), redis.call('HLEN', KEYS[2]), #redis.call('KEYS', ARGV[1]), #redis.call('KEYS', ARGV[2]), #redis.call('KEYS', ARGV[3])}",
                vec![store.key("services"), store.key("service-names")],
                vec![store.key("*"), store.key("announcement-index:*"), store.key("*revision*")],
            ).await.unwrap();
            // services is exactly the worklist consumed by all_announcements.
            assert!(
                counts[0] <= 4,
                "global listing work grew with history: {counts:?}"
            );
            assert!(counts[1] <= 4, "service names leaked: {counts:?}");
            assert!(counts[2] <= 3 * 4 + 6, "backend keys leaked: {counts:?}");
            assert!(counts[3] <= 4, "per-service indexes leaked: {counts:?}");
            assert_eq!(counts[4], 2, "only two global generations may persist");
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
        let now = unix_millis_now();
        let all = store.all_announcements(now).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "anchor");
        assert_eq!(store.all_liveness(now).await.unwrap().len(), 1);
        let count: usize = store.pool.scard(store.key("services")).await.unwrap();
        assert_eq!(count, 1, "expired cohort must leave no listing work");
        assert!(store.announcement_revision().await.unwrap() > announcement_generation);
        assert!(store.liveness_revision().await.unwrap() > liveness_generation);

        // Replace the still-cached anchor with a short-lived value on L2, then
        // expire and recreate the SAME scopes. Revisions must not reset/ABA.
        store
            .put_announcement("anchor", endpoint("iroh", 2, now + 100, now + 100))
            .await
            .unwrap();
        store
            .put_liveness(
                &anchor,
                LiveAllocatable {
                    allocatable: vec![],
                    load_fraction: 0.5,
                    last_seen: now,
                    live_until_unix_ms: now + 100,
                },
            )
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let now = unix_millis_now();
        assert!(store.all_announcements(now).await.unwrap().is_empty());
        assert!(store.all_liveness(now).await.unwrap().is_empty());
        let keys: Vec<String> = store
            .pool
            .eval(
                "return redis.call('KEYS', ARGV[1])",
                Vec::<String>::new(),
                vec![store.key("*")],
            )
            .await
            .unwrap();
        assert_eq!(
            keys.len(),
            2,
            "only persistent generations survive expiry: {keys:?}"
        );
        let expired_generation = store.announcement_revision().await.unwrap();
        let expired_liveness_generation = store.liveness_revision().await.unwrap();
        store
            .put_announcement("anchor", endpoint("iroh", 3, now + 30_000, now + 30_000))
            .await
            .unwrap();
        store
            .put_liveness(
                &anchor,
                LiveAllocatable {
                    allocatable: vec![],
                    load_fraction: 0.9,
                    last_seen: now,
                    live_until_unix_ms: now + 30_000,
                },
            )
            .await
            .unwrap();
        assert!(store.announcement_revision().await.unwrap() > expired_generation);
        assert!(store.liveness_revision().await.unwrap() > expired_liveness_generation);
        assert_eq!(
            tier.announcements_for("anchor", now).await.unwrap()[0].accepted_state_epoch,
            3
        );
        assert_eq!(
            tier.liveness(&anchor, now)
                .await
                .unwrap()
                .unwrap()
                .load_fraction,
            0.9
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn tiered_writes_invalidate_instead_of_tagging_caller_values() {
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let config = valkey_config(url, "write-race");
        let shared = Arc::new(ValkeyStateStore::connect(&config).await.unwrap());
        let other = ValkeyStateStore::connect(&config).await.unwrap();
        let tier = TieredStateStore::new(Arc::new(MemoryStateStore::new(8, 8, 8)), shared, 10_000);
        let now = unix_millis_now();
        let node = Did::new("did:web:node.example".to_owned());
        // Two writers contend on all value families. An older announcement or
        // heartbeat may be fenced; unordered artifacts follow completion order.
        for epoch in 1..=8 {
            let a = endpoint("iroh", epoch, now + 30_000, now + 20_000);
            let b = endpoint("iroh", epoch + 1, now + 30_000, now + 20_000);
            let (left, right) = tokio::join!(
                tier.put_announcement("model", a),
                other.put_announcement("model", b)
            );
            left.unwrap();
            right.unwrap();
            assert!(!tier.observed.lock().contains_key("announcement:model"));
            assert_eq!(
                tier.announcements_for("model", now).await.unwrap()[0].accepted_state_epoch,
                epoch + 1
            );
            let a = LiveAllocatable {
                allocatable: vec![],
                load_fraction: 0.1,
                last_seen: now + epoch as i64,
                live_until_unix_ms: now + 20_000,
            };
            let mut b = a.clone();
            b.last_seen += 1;
            b.load_fraction = 0.9;
            let (left, right) = tokio::join!(
                tier.put_liveness(&node, a),
                other.put_liveness(&node, b.clone())
            );
            left.unwrap();
            right.unwrap();
            assert!(!tier
                .observed
                .lock()
                .contains_key(&TieredStateStore::scope("liveness", node.as_str())));
            assert_eq!(
                tier.liveness(&node, now)
                    .await
                    .unwrap()
                    .unwrap()
                    .load_fraction,
                other.liveness(&node, now).await.unwrap().unwrap().load_fraction
            );
            let a = CachedEntityStatement {
                jwt: "a".to_owned(),
                fetched_at: now,
            };
            let b = CachedEntityStatement {
                jwt: "b".to_owned(),
                fetched_at: now,
            };
            let (left, right) = tokio::join!(
                tier.put_entity_statement("issuer", a),
                other.put_entity_statement("issuer", b)
            );
            left.unwrap();
            right.unwrap();
            assert!(!tier.observed.lock().contains_key("entity:issuer"));
            assert_eq!(
                tier.entity_statement("issuer").await.unwrap(),
                other.entity_statement("issuer").await.unwrap()
            );
            let a = CachedEnvelopeKeyset {
                cose_keyset_cbor: vec![1],
                fetched_at: now,
            };
            let b = CachedEnvelopeKeyset {
                cose_keyset_cbor: vec![2],
                fetched_at: now,
            };
            let (left, right) = tokio::join!(
                tier.put_envelope_keyset("service", a),
                other.put_envelope_keyset("service", b)
            );
            left.unwrap();
            right.unwrap();
            assert!(!tier.observed.lock().contains_key("envelope:service"));
            assert_eq!(
                tier.envelope_keyset("service").await.unwrap(),
                other.envelope_keyset("service").await.unwrap()
            );
        }
    }
    #[cfg(all(not(target_arch = "wasm32"), feature = "valkey"))]
    #[tokio::test]
    async fn valkey_legacy_liveness_encoding_fails_closed_until_heartbeat() {
        use fred::prelude::*;
        let Ok(url) = std::env::var("HYPRSTREAM_TEST_VALKEY_URL") else {
            return;
        };
        let store = ValkeyStateStore::connect(&valkey_config(url, "legacy-liveness"))
            .await
            .unwrap();
        let node = Did::new("did:web:legacy-node.example".to_owned());
        let now = unix_millis_now();
        let value = LiveAllocatable {
            allocatable: vec![],
            load_fraction: 0.1,
            last_seen: now,
            live_until_unix_ms: now + 30_000,
        };
        store.put_liveness(&node, value.clone()).await.unwrap();
        // Reproduce the pre-repair wire value, which had no enumerable DID.
        let _: String = store
            .pool
            .eval(
                "return redis.call('SET', KEYS[1], ARGV[1], 'KEEPTTL')",
                vec![format!(
                    "{}:liveness:{}",
                    store.prefix,
                    ValkeyStateStore::digest(node.as_str())
                )],
                vec![serde_json::to_string(&value).unwrap()],
            )
            .await
            .unwrap();
        assert!(store
            .all_liveness(now)
            .await
            .unwrap()
            .is_empty());
        store.put_liveness(&node, value).await.unwrap();
        assert_eq!(store.all_liveness(now).await.unwrap()[0].0, node);
    }
}
