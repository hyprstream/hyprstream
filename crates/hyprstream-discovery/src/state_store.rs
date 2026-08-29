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
            url: "redis://127.0.0.1:6379".to_owned(),
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
    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>>;

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()>;
    async fn entity_statement(&self, issuer: &str) -> Result<Option<CachedEntityStatement>>;
    async fn known_issuers(&self) -> Result<Vec<String>>;

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
    }

    #[cfg(feature = "valkey")]
    fn clear_all_announcements_sync(&self) {
        let mut inner = self.inner.lock();
        inner.announcements.clear();
        inner.service_index.clear();
    }

    #[cfg(feature = "valkey")]
    fn clear_liveness_sync(&self, node: &Did) {
        self.inner.lock().liveness.remove(node.as_str());
    }

    #[cfg(feature = "valkey")]
    fn clear_entity_statement_sync(&self, issuer: &str) {
        self.inner.lock().entity_statements.remove(issuer);
    }

    #[cfg(feature = "valkey")]
    fn clear_envelope_keyset_sync(&self, service_did: &str) {
        self.inner.lock().envelope_keysets.remove(service_did);
    }

    #[cfg(feature = "valkey")]
    fn clear_entity_statements_sync(&self) {
        self.inner.lock().entity_statements.clear();
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
            if existing.value.last_seen > value.last_seen {
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
        Ok(PutResult::Stored)
    }

    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>> {
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now_unix_ms);
        Ok(inner
            .liveness
            .get(node.as_str())
            .map(|entry| entry.value.clone())
            .filter(|entry| entry.is_live_at(now_unix_ms)))
    }

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()> {
        let mut inner = self.inner.lock();
        if !inner.entity_statements.contains_key(issuer)
            && inner.entity_statements.len() >= self.artifact_capacity
        {
            bail!("Discovery memory federation artifact capacity exhausted");
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
#[derive(Clone)]
struct ValkeyStateStore {
    pool: fred::prelude::RedisPool,
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
            config.command_timeout_ms > 0,
            "Discovery Valkey command_timeout_ms must be positive"
        );
        anyhow::ensure!(
            config.announcement_capacity > 0
                && config.liveness_capacity > 0
                && config.artifact_capacity > 0,
            "Discovery Valkey capacities must be positive"
        );
        let redis = RedisConfig::from_url(&config.url).context("invalid Discovery Valkey URL")?;
        let mut builder = Builder::from_config(redis);
        builder.with_performance_config(|performance| {
            performance.default_command_timeout =
                std::time::Duration::from_millis(config.command_timeout_ms);
        });
        let pool = builder.build_pool(config.pool_size)?;
        pool.connect();
        pool.wait_for_connect().await?;
        let _: String = pool.ping().await?;
        Ok(Self {
            pool,
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

    fn announcement_revision_key(&self, service_name: &str) -> String {
        format!(
            "{}:announcement-revision:{}",
            self.prefix,
            Self::service_id(service_name)
        )
    }

    async fn revision(&self, key: String) -> Result<u64> {
        use fred::prelude::*;
        Ok(self.pool.get::<Option<u64>, _>(key).await?.unwrap_or(0))
    }

    async fn announcement_revision(&self, service_name: &str) -> Result<u64> {
        self.revision(self.announcement_revision_key(service_name))
            .await
    }

    async fn announcement_global_revision(&self) -> Result<u64> {
        self.revision(self.key("announcement-global-revision"))
            .await
    }

    async fn liveness_revision(&self, node: &Did) -> Result<u64> {
        self.revision(format!(
            "{}:liveness-revision:{}",
            self.prefix,
            Self::digest(node.as_str())
        ))
        .await
    }

    async fn entity_revision(&self, issuer: &str) -> Result<u64> {
        self.revision(format!(
            "{}:entity-revision:{}",
            self.prefix,
            Self::digest(issuer)
        ))
        .await
    }

    async fn entity_global_revision(&self) -> Result<u64> {
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

        let index = self.announcement_index(service_name);
        let record_keys: Vec<String> = self.pool.smembers(&index).await?;
        let mut live = Vec::with_capacity(record_keys.len());
        for record_key in record_keys {
            let encoded: Option<String> = self.pool.get(&record_key).await?;
            match encoded {
                Some(encoded) => {
                    let endpoint: AnnouncedEndpoint = serde_json::from_str(&encoded)?;
                    if endpoint.is_live_at(now_unix_ms) {
                        live.push(endpoint);
                    } else {
                        let _: i64 = self.pool.del(&record_key).await?;
                        let _: i64 = self.pool.srem(&index, &record_key).await?;
                    }
                }
                None => {
                    let _: i64 = self.pool.srem(&index, &record_key).await?;
                }
            }
        }
        Ok(live)
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
local current = redis.call('GET', KEYS[1])
redis.call('ZREMRANGEBYSCORE', KEYS[7], '-inf', ARGV[7])
if not current and redis.call('ZCARD', KEYS[7]) >= tonumber(ARGV[8]) then
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
redis.call('INCR', KEYS[6])
redis.call('ZADD', KEYS[7], ARGV[4], KEYS[1])
return 1
"#;
        let service_id = Self::service_id(service_name);
        let encoded = serde_json::to_string(&endpoint)?;
        let stored: i64 = self
            .pool
            .eval(
                PUT,
                vec![
                    self.announcement_key(service_name, &endpoint.socket_kind),
                    self.announcement_index(service_name),
                    self.key("services"),
                    self.key("service-names"),
                    self.announcement_revision_key(service_name),
                    self.key("announcement-global-revision"),
                    self.key("announcement-expiry"),
                ],
                vec![
                    encoded,
                    endpoint.accepted_state_epoch.to_string(),
                    endpoint.expires_at_unix_ms.to_string(),
                    endpoint.live_until_unix_ms.to_string(),
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

        let service_ids: Vec<String> = self.pool.smembers(self.key("services")).await?;
        let mut all = Vec::with_capacity(service_ids.len());
        for service_id in service_ids {
            let service_name: Option<String> = self
                .pool
                .hget(self.key("service-names"), &service_id)
                .await?;
            let Some(service_name) = service_name else {
                let _: i64 = self.pool.srem(self.key("services"), &service_id).await?;
                continue;
            };
            let entries = self
                .announcements_for_inner(&service_name, now_unix_ms)
                .await?;
            if entries.is_empty() {
                let _: i64 = self.pool.srem(self.key("services"), &service_id).await?;
                let _: i64 = self
                    .pool
                    .hdel(self.key("service-names"), &service_id)
                    .await?;
            } else {
                all.push((service_name, entries));
            }
        }
        Ok(all)
    }

    async fn put_liveness(&self, node: &Did, value: LiveAllocatable) -> Result<PutResult> {
        use fred::prelude::*;

        const PUT: &str = r#"
local current = redis.call('GET', KEYS[1])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[4])
if not current and redis.call('ZCARD', KEYS[3]) >= tonumber(ARGV[5]) then
  return redis.error_reply('Discovery Valkey liveness capacity exhausted')
end
if current then
  local ok, decoded = pcall(cjson.decode, current)
  if not ok then return redis.error_reply('corrupt Discovery liveness') end
  if (tonumber(decoded.last_seen) or 0) > tonumber(ARGV[2]) then return 0 end
end
redis.call('SET', KEYS[1], ARGV[1], 'PXAT', ARGV[3])
redis.call('INCR', KEYS[2])
redis.call('ZADD', KEYS[3], ARGV[3], KEYS[1])
return 1
"#;
        let node_id = Self::digest(node.as_str());
        let stored: i64 = self
            .pool
            .eval(
                PUT,
                vec![
                    format!("{}:liveness:{node_id}", self.prefix),
                    format!("{}:liveness-revision:{node_id}", self.prefix),
                    self.key("liveness-expiry"),
                ],
                vec![
                    serde_json::to_string(&value)?,
                    value.last_seen.to_string(),
                    value.live_until_unix_ms.to_string(),
                    unix_millis_now().to_string(),
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

    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>> {
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
            .map(|encoded| serde_json::from_str(&encoded))
            .transpose()?
            .filter(|value: &LiveAllocatable| value.is_live_at(now_unix_ms)))
    }

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()> {
        use fred::prelude::*;
        const PUT: &str = r#"
if redis.call('EXISTS', KEYS[1]) == 0 and redis.call('SCARD', KEYS[2]) >= tonumber(ARGV[4]) then
  return redis.error_reply('Discovery Valkey federation artifact capacity exhausted')
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('SADD', KEYS[2], ARGV[2])
redis.call('HSET', KEYS[3], ARGV[2], ARGV[3])
redis.call('INCR', KEYS[4])
redis.call('INCR', KEYS[5])
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
                    format!("{}:entity-revision:{id}", self.prefix),
                    self.key("entity-global-revision"),
                ],
                vec![
                    serde_json::to_string(&value)?,
                    id,
                    issuer.to_owned(),
                    self.artifact_capacity.to_string(),
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
        let ids: Vec<String> = self.pool.smembers(self.key("issuers")).await?;
        let mut issuers = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(issuer) = self
                .pool
                .hget::<Option<String>, _, _>(self.key("issuer-names"), &id)
                .await?
            {
                issuers.push(issuer);
            }
        }
        Ok(issuers)
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
        }
    }

    fn scope(kind: &str, value: &str) -> String {
        format!("{kind}:{value}")
    }

    fn is_observed(&self, scope: &str, revision: u64, now_unix_ms: i64) -> bool {
        self.observed.lock().get(scope).is_some_and(|observed| {
            observed.revision == revision && now_unix_ms < observed.valid_until_unix_ms
        })
    }

    fn observe(&self, scope: String, revision: u64, now_unix_ms: i64) {
        self.observed.lock().insert(
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

    fn l1_liveness(&self, mut value: LiveAllocatable, now_unix_ms: i64) -> LiveAllocatable {
        value.live_until_unix_ms = value
            .live_until_unix_ms
            .min(now_unix_ms.saturating_add(self.l1_max_ttl_ms));
        value
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
        let result = self
            .valkey
            .put_announcement(service_name, endpoint.clone())
            .await?;
        if result == PutResult::Stored {
            let now = unix_millis_now();
            self.memory
                .put_announcement(service_name, self.l1_announcement(endpoint, now))
                .await?;
            let revision = self.valkey.announcement_revision(service_name).await?;
            self.observe(Self::scope("announcement", service_name), revision, now);
        }
        Ok(result)
    }

    async fn announcements_for(
        &self,
        service_name: &str,
        now_unix_ms: i64,
    ) -> Result<Vec<AnnouncedEndpoint>> {
        let scope = Self::scope("announcement", service_name);
        let revision = self.valkey.announcement_revision(service_name).await?;
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
        self.memory.clear_announcements_sync(service_name);
        for value in &values {
            self.memory
                .put_announcement(
                    service_name,
                    self.l1_announcement(value.clone(), now_unix_ms),
                )
                .await?;
        }
        self.observe(scope, revision, now_unix_ms);
        Ok(values)
    }

    async fn all_announcements(
        &self,
        now_unix_ms: i64,
    ) -> Result<Vec<(String, Vec<AnnouncedEndpoint>)>> {
        let scope = "announcement-global".to_owned();
        let revision = self.valkey.announcement_global_revision().await?;
        if self.is_observed(&scope, revision, now_unix_ms) {
            return self.memory.all_announcements(now_unix_ms).await;
        }
        let values = self.valkey.all_announcements(now_unix_ms).await?;
        self.memory.clear_all_announcements_sync();
        for (service_name, endpoints) in &values {
            for endpoint in endpoints {
                self.memory
                    .put_announcement(
                        service_name,
                        self.l1_announcement(endpoint.clone(), now_unix_ms),
                    )
                    .await?;
            }
        }
        self.observe(scope, revision, now_unix_ms);
        Ok(values)
    }

    async fn put_liveness(&self, node: &Did, value: LiveAllocatable) -> Result<PutResult> {
        let result = self.valkey.put_liveness(node, value.clone()).await?;
        if result == PutResult::Stored {
            let now = unix_millis_now();
            self.memory
                .put_liveness(node, self.l1_liveness(value, now))
                .await?;
            let revision = self.valkey.liveness_revision(node).await?;
            self.observe(Self::scope("liveness", node.as_str()), revision, now);
        }
        Ok(result)
    }

    async fn liveness(&self, node: &Did, now_unix_ms: i64) -> Result<Option<LiveAllocatable>> {
        let scope = Self::scope("liveness", node.as_str());
        let revision = self.valkey.liveness_revision(node).await?;
        if self.is_observed(&scope, revision, now_unix_ms) {
            return self.memory.liveness(node, now_unix_ms).await;
        }
        let value = self.valkey.liveness(node, now_unix_ms).await?;
        self.memory.clear_liveness_sync(node);
        if let Some(value) = &value {
            self.memory
                .put_liveness(node, self.l1_liveness(value.clone(), now_unix_ms))
                .await?;
        }
        self.observe(scope, revision, now_unix_ms);
        Ok(value)
    }

    async fn put_entity_statement(&self, issuer: &str, value: CachedEntityStatement) -> Result<()> {
        self.valkey
            .put_entity_statement(issuer, value.clone())
            .await?;
        self.memory.put_entity_statement(issuer, value).await?;
        let revision = self.valkey.entity_revision(issuer).await?;
        self.observe(Self::scope("entity", issuer), revision, unix_millis_now());
        Ok(())
    }

    async fn entity_statement(&self, issuer: &str) -> Result<Option<CachedEntityStatement>> {
        let now = unix_millis_now();
        let scope = Self::scope("entity", issuer);
        let revision = self.valkey.entity_revision(issuer).await?;
        if self.is_observed(&scope, revision, now) {
            return self.memory.entity_statement(issuer).await;
        }
        let value = self.valkey.entity_statement(issuer).await?;
        self.memory.clear_entity_statement_sync(issuer);
        if let Some(value) = &value {
            self.memory
                .put_entity_statement(issuer, value.clone())
                .await?;
        }
        self.observe(scope, revision, now);
        Ok(value)
    }

    async fn known_issuers(&self) -> Result<Vec<String>> {
        let now = unix_millis_now();
        let scope = "entity-global".to_owned();
        let revision = self.valkey.entity_global_revision().await?;
        if self.is_observed(&scope, revision, now) {
            return self.memory.known_issuers().await;
        }
        let issuers = self.valkey.known_issuers().await?;
        self.memory.clear_entity_statements_sync();
        for issuer in &issuers {
            if let Some(value) = self.valkey.entity_statement(issuer).await? {
                self.memory.put_entity_statement(issuer, value).await?;
            }
        }
        self.observe(scope, revision, now);
        Ok(issuers)
    }

    async fn put_envelope_keyset(
        &self,
        service_did: &str,
        value: CachedEnvelopeKeyset,
    ) -> Result<()> {
        self.valkey
            .put_envelope_keyset(service_did, value.clone())
            .await?;
        self.memory.put_envelope_keyset(service_did, value).await?;
        let revision = self.valkey.envelope_revision(service_did).await?;
        self.observe(
            Self::scope("envelope", service_did),
            revision,
            unix_millis_now(),
        );
        Ok(())
    }

    async fn envelope_keyset(&self, service_did: &str) -> Result<Option<CachedEnvelopeKeyset>> {
        let now = unix_millis_now();
        let scope = Self::scope("envelope", service_did);
        let revision = self.valkey.envelope_revision(service_did).await?;
        if self.is_observed(&scope, revision, now) {
            return self.memory.envelope_keyset(service_did).await;
        }
        let value = self.valkey.envelope_keyset(service_did).await?;
        self.memory.clear_envelope_keyset_sync(service_did);
        if let Some(value) = &value {
            self.memory
                .put_envelope_keyset(service_did, value.clone())
                .await?;
        }
        self.observe(scope, revision, now);
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
}
