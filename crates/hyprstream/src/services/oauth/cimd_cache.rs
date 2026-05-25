//! Client ID Metadata Document cache.
//!
//! Spec: draft-ietf-oauth-client-id-metadata-document-00 §6.2 — servers
//! SHOULD cache metadata respecting HTTP cache headers. MCP 2025-11-25
//! authorization spec (§ Client ID Metadata Documents → Authorization
//! Servers) lifts this requirement verbatim.
//!
//! Design:
//!
//! * `entries: HashMap<client_id_url, CachedClient>` — O(1) lookup.
//! * `heap: BinaryHeap<Reverse<HeapEntry>>` — min-heap of `(expires_at,
//!   client_id, version)` for O(log n) ordered eviction.
//! * **Lazy deletion** via per-client version counter: on re-insert we
//!   push a new heap node but do NOT walk the heap to remove the old one.
//!   The reaper discards heap nodes whose version no longer matches the
//!   HashMap entry's version.
//! * **Inline reap** at every cache access, bounded to `reap_budget` pops
//!   per call to keep tail latency predictable. No background task.
//! * **TTL bounds** clamp HTTP `Cache-Control: max-age` / `Expires` to
//!   `[min_ttl, max_ttl]`. Default `[5min, 24h]`.
//! * **Capacity bound**: when full, evict the next-to-expire entry.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

use super::state::RegisteredClient;

/// Cache configuration. Defaults documented per field.
#[derive(Debug, Clone)]
pub struct CimdCacheConfig {
    /// Minimum TTL applied to any cached entry, regardless of upstream
    /// cache headers. Defends against thundering-herd refetch when an
    /// upstream returns `max-age=0`.
    pub min_ttl: Duration,
    /// Maximum TTL applied to any cached entry. Bounds the staleness of
    /// CIMD metadata when an upstream returns aggressive Cache-Control.
    pub max_ttl: Duration,
    /// Maximum number of cached entries. Defends against DoS amplification
    /// where an attacker forces the AS to register unique client_id URLs.
    pub max_entries: usize,
    /// Maximum number of heap pops per inline reap. Bounds tail latency
    /// at cache access while still amortizing cleanup.
    pub reap_budget: usize,
}

impl Default for CimdCacheConfig {
    fn default() -> Self {
        Self {
            min_ttl: Duration::from_secs(300),     // 5 min
            max_ttl: Duration::from_secs(86_400),  // 24 h
            max_entries: 10_000,
            reap_budget: 16,
        }
    }
}

/// A cached client with expiry + version tag for lazy heap deletion.
#[derive(Debug, Clone)]
struct CachedClient {
    client: RegisteredClient,
    expires_at: Instant,
    /// Monotonic version per client_id. Bumped on every insert. The
    /// reaper compares this to the heap node's `version` to detect stale
    /// nodes left behind by re-inserts.
    version: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry {
    expires_at: Instant,
    client_id: String,
    version: u64,
}

// Min-heap by expires_at: smaller expires_at sorts first when wrapped in
// Reverse. PartialOrd/Ord only consider expires_at; ties broken by string
// comparison for determinism.
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.expires_at
            .cmp(&other.expires_at)
            .then_with(|| self.client_id.cmp(&other.client_id))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// CIMD metadata cache. Thread-safe; cheap to share via `Arc`.
pub struct CimdCache {
    entries: RwLock<HashMap<String, CachedClient>>,
    heap: Mutex<BinaryHeap<Reverse<HeapEntry>>>,
    config: CimdCacheConfig,
    next_version: AtomicU64,
}

impl CimdCache {
    pub fn new(config: CimdCacheConfig) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            heap: Mutex::new(BinaryHeap::new()),
            config,
            next_version: AtomicU64::new(1),
        }
    }

    /// Compute a clamped TTL from an optional upstream `max-age` value.
    /// `None` (no cache header) yields `min_ttl`.
    pub fn clamp_ttl(&self, http_max_age: Option<Duration>) -> Duration {
        let ttl = http_max_age.unwrap_or(self.config.min_ttl);
        ttl.clamp(self.config.min_ttl, self.config.max_ttl)
    }

    /// Look up a cached CIMD client by its `client_id` URL.
    ///
    /// Performs a bounded inline reap before the lookup. Returns `None`
    /// if absent or expired.
    pub async fn get(&self, client_id: &str) -> Option<RegisteredClient> {
        self.reap(Instant::now()).await;
        let entries = self.entries.read().await;
        let cached = entries.get(client_id)?;
        if Instant::now() >= cached.expires_at {
            None
        } else {
            Some(cached.client.clone())
        }
    }

    /// Insert (or replace) a cached CIMD client with the given TTL.
    ///
    /// If the cache is at capacity, the next-to-expire entry is evicted
    /// first. The new entry is added to both the HashMap and the heap;
    /// any prior heap node for this client_id becomes stale and will be
    /// reaped lazily.
    pub async fn insert(&self, client: RegisteredClient, ttl: Duration) {
        let now = Instant::now();
        self.reap(now).await;

        let expires_at = now + ttl;
        let client_id = client.client_id.clone();
        let version = self.next_version.fetch_add(1, Ordering::Relaxed);

        // Evict if at capacity AND this would be a new key.
        {
            let entries = self.entries.read().await;
            let at_capacity = entries.len() >= self.config.max_entries
                && !entries.contains_key(&client_id);
            drop(entries);
            if at_capacity {
                self.evict_one().await;
            }
        }

        let cached = CachedClient { client, expires_at, version };
        self.entries.write().await.insert(client_id.clone(), cached);
        self.heap.lock().await.push(Reverse(HeapEntry { expires_at, client_id, version }));
    }

    /// Reap expired entries from the heap, bounded by `reap_budget`.
    /// Stale heap nodes (version mismatch) are discarded without touching
    /// the HashMap.
    async fn reap(&self, now: Instant) {
        let mut heap = self.heap.lock().await;
        for _ in 0..self.config.reap_budget {
            let Some(Reverse(top)) = heap.peek() else { break };
            if top.expires_at > now {
                break;
            }
            let popped = heap.pop().unwrap_or_else(|| unreachable!()).0;
            // Acquire write lock briefly to remove if version still matches.
            let mut entries = self.entries.write().await;
            if let Some(cached) = entries.get(&popped.client_id) {
                if cached.version == popped.version && cached.expires_at <= now {
                    entries.remove(&popped.client_id);
                }
                // else: stale heap node (entry was refreshed) — drop silently.
            }
        }
    }

    /// Evict the single next-to-expire entry. Called when the cache is at
    /// capacity and a new key arrives.
    async fn evict_one(&self) {
        let mut heap = self.heap.lock().await;
        while let Some(Reverse(top)) = heap.pop() {
            let mut entries = self.entries.write().await;
            if let Some(cached) = entries.get(&top.client_id) {
                if cached.version == top.version {
                    entries.remove(&top.client_id);
                    return;
                }
                // Stale heap node — keep popping.
            }
        }
    }

    /// Current cache size (for metrics / tests).
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// True when no entries are cached. Mostly for tests.
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_client(id: &str) -> RegisteredClient {
        RegisteredClient {
            client_id: id.to_owned(),
            redirect_uris: vec!["http://localhost/cb".to_owned()],
            client_name: None,
            client_uri: None,
            logo_uri: None,
            grant_types: vec![],
            response_types: vec![],
            token_endpoint_auth_method: None,
            jwks: None,
            jwks_uri: None,
            is_cimd: true,
            registered_at: Instant::now(),
        }
    }

    #[tokio::test]
    async fn insert_and_get_roundtrip() {
        let cache = CimdCache::new(CimdCacheConfig::default());
        cache.insert(make_client("https://a.test/c"), Duration::from_secs(60)).await;
        let got = cache.get("https://a.test/c").await;
        assert!(got.is_some());
        assert_eq!(got.unwrap().client_id, "https://a.test/c");
    }

    #[tokio::test]
    async fn expired_entry_returns_none() {
        let cache = CimdCache::new(CimdCacheConfig::default());
        cache.insert(make_client("https://a.test/c"), Duration::from_millis(10)).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(cache.get("https://a.test/c").await.is_none());
    }

    #[tokio::test]
    async fn ttl_is_clamped_to_min_and_max() {
        let cache = CimdCache::new(CimdCacheConfig {
            min_ttl: Duration::from_secs(60),
            max_ttl: Duration::from_secs(3600),
            ..CimdCacheConfig::default()
        });
        // None → min_ttl
        assert_eq!(cache.clamp_ttl(None), Duration::from_secs(60));
        // Below min → min
        assert_eq!(cache.clamp_ttl(Some(Duration::from_secs(10))), Duration::from_secs(60));
        // Within range → as-is
        assert_eq!(cache.clamp_ttl(Some(Duration::from_secs(600))), Duration::from_secs(600));
        // Above max → max
        assert_eq!(cache.clamp_ttl(Some(Duration::from_secs(99_999))), Duration::from_secs(3600));
    }

    #[tokio::test]
    async fn reinsert_does_not_leak_heap_after_reap() {
        let cache = CimdCache::new(CimdCacheConfig {
            reap_budget: 32,
            ..CimdCacheConfig::default()
        });
        // Insert 5 versions of the same key, each with a short TTL.
        for _ in 0..5 {
            cache.insert(make_client("https://a.test/c"), Duration::from_millis(10)).await;
        }
        // After expiry + reap, only one map entry exists (or zero), and
        // the heap should not grow without bound: subsequent inserts
        // should still resolve.
        tokio::time::sleep(Duration::from_millis(50)).await;
        cache.insert(make_client("https://a.test/c"), Duration::from_secs(60)).await;
        assert_eq!(cache.len().await, 1);
        assert!(cache.get("https://a.test/c").await.is_some());
    }

    #[tokio::test]
    async fn evict_one_terminates_under_stale_heap_pressure() {
        // Repeatedly re-insert the same key with bumped versions to
        // accumulate stale heap nodes, then trigger evict_one by
        // overfilling. Must terminate (not infinite-loop) and must
        // actually evict the right entry.
        let cache = CimdCache::new(CimdCacheConfig {
            max_entries: 2,
            reap_budget: 4, // small budget keeps reap from clearing the stale nodes
            min_ttl: Duration::from_secs(60),
            max_ttl: Duration::from_secs(3600),
        });

        // 50 stale heap nodes for client A, all "live" entry version.
        for _ in 0..50 {
            cache.insert(make_client("https://a.test/c"), Duration::from_secs(120)).await;
        }
        // One entry for client B at capacity.
        cache.insert(make_client("https://b.test/c"), Duration::from_secs(60)).await;
        assert_eq!(cache.len().await, 2, "A + B at capacity");

        // Insert C — triggers evict_one which must walk past stale A
        // heap nodes to find a fresh victim and terminate.
        cache.insert(make_client("https://c.test/c"), Duration::from_secs(180)).await;
        assert_eq!(cache.len().await, 2, "still 2 after eviction");
        // B had the shortest TTL among current map entries, so should be evicted.
        assert!(
            cache.get("https://b.test/c").await.is_none(),
            "B (shortest TTL) should have been the victim"
        );
        assert!(cache.get("https://a.test/c").await.is_some(), "A still present");
        assert!(cache.get("https://c.test/c").await.is_some(), "C just inserted");
    }

    #[tokio::test]
    async fn capacity_bound_evicts_next_to_expire() {
        let cache = CimdCache::new(CimdCacheConfig {
            max_entries: 2,
            ..CimdCacheConfig::default()
        });
        cache.insert(make_client("https://a.test/c1"), Duration::from_secs(10)).await;
        cache.insert(make_client("https://b.test/c2"), Duration::from_secs(60)).await;
        // Capacity reached. Inserting a 3rd evicts the next-to-expire (c1).
        cache.insert(make_client("https://c.test/c3"), Duration::from_secs(30)).await;
        assert_eq!(cache.len().await, 2);
        assert!(cache.get("https://a.test/c1").await.is_none());
        assert!(cache.get("https://b.test/c2").await.is_some());
        assert!(cache.get("https://c.test/c3").await.is_some());
    }
}
