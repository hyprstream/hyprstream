//! Client ID Metadata Document cache.
//!
//! Spec: draft-ietf-oauth-client-id-metadata-document-00 §6.2 — servers
//! SHOULD cache metadata respecting HTTP cache headers. MCP 2025-11-25
//! authorization spec (§ Client ID Metadata Documents → Authorization
//! Servers) lifts this requirement verbatim.
//!
//! Design:
//!
//! This cache is a thin OAuth-specific wrapper over the shared
//! [`TtlCache`](hyprstream_util::TtlCache) substrate (extracted in
//! #524). `TtlCache` provides the eviction machinery — per-entry TTL,
//! lazy version-tagged deletion on re-insert, bounded inline reaping and
//! a capacity bound (next-to-expire victim). This wrapper adds:
//!
//! * **TTL clamping**: HTTP `Cache-Control: max-age` / `Expires` is
//!   clamped to `[min_ttl, max_ttl]` (default `[5min, 24h]`) before the
//!   entry is inserted. Defends against thundering-herd refetch (`max-age=0`)
//!   and unbounded staleness.
//! * **`RegisteredClient` keying** by `client_id` URL.
//!
//! The public API remains `async` for call-site compatibility; the
//! underlying `TtlCache` is synchronous (operations are short, in-memory
//! and never span an `.await`).

use std::time::Duration;

use hyprstream_util::TtlCache;

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
            min_ttl: Duration::from_secs(300),    // 5 min
            max_ttl: Duration::from_secs(86_400), // 24 h
            max_entries: 10_000,
            reap_budget: 16,
        }
    }
}

/// CIMD metadata cache. Thread-safe; cheap to share via `Arc`.
pub struct CimdCache {
    cache: TtlCache<String, RegisteredClient>,
    config: CimdCacheConfig,
}

impl CimdCache {
    #[must_use]
    pub fn new(config: CimdCacheConfig) -> Self {
        let cache = TtlCache::new(config.max_entries, config.reap_budget);
        Self { cache, config }
    }

    /// Compute a clamped TTL from an optional upstream `max-age` value.
    /// `None` (no cache header) yields `min_ttl`.
    #[must_use]
    pub fn clamp_ttl(&self, http_max_age: Option<Duration>) -> Duration {
        let ttl = http_max_age.unwrap_or(self.config.min_ttl);
        ttl.clamp(self.config.min_ttl, self.config.max_ttl)
    }

    /// Look up a cached CIMD client by its `client_id` URL.
    ///
    /// Performs a bounded inline reap before the lookup. Returns `None`
    /// if absent or expired.
    pub async fn get(&self, client_id: &str) -> Option<RegisteredClient> {
        self.cache.get(client_id)
    }

    /// Insert (or replace) a cached CIMD client with the given TTL.
    ///
    /// If the cache is at capacity, the next-to-expire entry is evicted
    /// first. Re-inserting an existing `client_id` refreshes it (bumps
    /// the version tag); the prior heap node is reaped lazily.
    pub async fn insert(&self, client: RegisteredClient, ttl: Duration) {
        let client_id = client.client_id.clone();
        self.cache.insert(client_id, client, ttl);
    }

    /// Current cache size (for metrics / tests).
    pub async fn len(&self) -> usize {
        self.cache.len()
    }

    /// True when no entries are cached. Mostly for tests.
    pub async fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::time::Instant;

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
            hyprstream_node_did: None,
            scope: None,
            dpop_bound_access_tokens: None,
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
