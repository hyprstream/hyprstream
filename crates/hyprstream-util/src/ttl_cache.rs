//! Generic TTL cache with lazy, version-tagged eviction.
//!
//! `TtlCache<K, V>` is the shared eviction substrate used across
//! hyprstream. It was extracted from the OAuth Client ID Metadata
//! Document cache (`hyprstream::services::oauth::cimd_cache`) so that
//! the same tested implementation can back both that cache and the
//! discovery placement directory (see epic #523 / #524) — one eviction
//! impl, no `moka` dependency.
//!
//! # Design
//!
//! * `entries: HashMap<K, Entry<V>>` — O(1) lookup.
//! * `heap: BinaryHeap<Reverse<HeapEntry<K>>>` — min-heap of
//!   `(expires_at, key, version)` for O(log n) ordered eviction.
//! * **Lazy deletion** via a per-key version counter: on re-insert we
//!   push a new heap node but do NOT walk the heap to remove the old
//!   one. The reaper discards heap nodes whose version no longer
//!   matches the map entry's version.
//! * **Heartbeat refresh = re-insert.** Re-inserting a key bumps its
//!   version tag and extends its expiry; the prior heap node becomes
//!   stale and is reaped lazily.
//! * **Inline reap** at every access, bounded to `reap_budget` pops per
//!   call to keep tail latency predictable. No background task.
//! * **Capacity bound**: when full, evict the next-to-expire entry.
//!
//! # Concurrency
//!
//! Thread-safe and cheap to share via `Arc`. All mutable state lives
//! behind a single `parking_lot::Mutex`; operations are short, in-memory
//! and lock-free of any `.await`, so the synchronous lock never spans an
//! I/O point even when called from async code.

use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// A cached value with its expiry instant and version tag.
#[derive(Debug, Clone)]
struct Entry<V> {
    value: V,
    expires_at: Instant,
    /// Monotonic version per key, bumped on every insert. The reaper
    /// compares this to a heap node's `version` to detect stale nodes
    /// left behind by re-inserts.
    version: u64,
}

/// Heap node ordered by `expires_at` (ties broken by key for
/// determinism). Wrapped in `Reverse` to make the `BinaryHeap` a
/// min-heap over `expires_at`.
#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry<K> {
    expires_at: Instant,
    key: K,
    version: u64,
}

impl<K: Ord> Ord for HeapEntry<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.expires_at
            .cmp(&other.expires_at)
            .then_with(|| self.key.cmp(&other.key))
    }
}
impl<K: Ord> PartialOrd for HeapEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct Inner<K, V> {
    entries: HashMap<K, Entry<V>>,
    heap: BinaryHeap<Reverse<HeapEntry<K>>>,
}

/// Generic per-entry-TTL cache with lazy version-tagged eviction and a
/// capacity bound. See the module docs for the design.
pub struct TtlCache<K, V> {
    inner: Mutex<Inner<K, V>>,
    /// Maximum number of live entries. When full, an insert of a new key
    /// first evicts the next-to-expire entry.
    max_entries: usize,
    /// Maximum heap pops per inline reap; bounds tail latency while
    /// amortizing cleanup.
    reap_budget: usize,
    next_version: AtomicU64,
}

impl<K, V> TtlCache<K, V>
where
    K: Clone + Eq + Hash + Ord,
{
    /// Create a cache bounded to `max_entries`, reaping at most
    /// `reap_budget` expired heap nodes per access.
    #[must_use]
    pub fn new(max_entries: usize, reap_budget: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                heap: BinaryHeap::new(),
            }),
            max_entries,
            reap_budget,
            next_version: AtomicU64::new(1),
        }
    }

    /// Insert (or replace) `key` with `value`, expiring after `ttl`.
    ///
    /// Re-inserting an existing key refreshes it: a new version is
    /// assigned and the prior heap node becomes stale (reaped lazily).
    /// If the cache is at capacity and `key` is new, the next-to-expire
    /// entry is evicted first.
    pub fn insert(&self, key: K, value: V, ttl: Duration) {
        let now = Instant::now();
        let expires_at = now + ttl;
        let version = self.next_version.fetch_add(1, Ordering::Relaxed);

        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now, self.reap_budget);

        if inner.entries.len() >= self.max_entries && !inner.entries.contains_key(&key) {
            Self::evict_one(&mut inner);
        }

        inner.entries.insert(
            key.clone(),
            Entry {
                value,
                expires_at,
                version,
            },
        );
        inner.heap.push(Reverse(HeapEntry {
            expires_at,
            key,
            version,
        }));
    }

    /// Atomically insert `key` only if no live (unexpired) entry exists
    /// for it, returning `true` if the key was newly inserted.
    ///
    /// Returns `false` when a live entry already exists — i.e. this key
    /// is a duplicate/replay within its validity window. The check and
    /// the insert happen under a single lock, so concurrent callers for
    /// the same key cannot both observe "absent" (the race that a
    /// `get`-then-`insert` sequence would open). Expired entries are
    /// reaped inline first, so an expired entry does not block insertion.
    ///
    /// This is the primitive for replay/dedup caches (e.g. DPoP `jti`
    /// replay prevention, RFC 9449 §11.1): a caller proceeds only when
    /// this returns `true`, and rejects on `false`.
    pub fn insert_if_absent(&self, key: K, value: V, ttl: Duration) -> bool {
        let now = Instant::now();
        let expires_at = now + ttl;
        let version = self.next_version.fetch_add(1, Ordering::Relaxed);

        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now, self.reap_budget);

        if inner.entries.contains_key(&key) {
            return false;
        }
        if inner.entries.len() >= self.max_entries {
            Self::evict_one(&mut inner);
        }
        inner.entries.insert(
            key.clone(),
            Entry {
                value,
                expires_at,
                version,
            },
        );
        inner.heap.push(Reverse(HeapEntry {
            expires_at,
            key,
            version,
        }));
        true
    }

    /// Look up `key`, returning a clone of the value if present and not
    /// yet expired. Performs a bounded inline reap first.
    ///
    /// Accepts any borrowed form of the key (e.g. `&str` for a
    /// `String`-keyed cache) to avoid allocating on lookup.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
        V: Clone,
    {
        let now = Instant::now();
        let mut inner = self.inner.lock();
        Self::reap(&mut inner, now, self.reap_budget);
        let entry = inner.entries.get(key)?;
        if now >= entry.expires_at {
            None
        } else {
            Some(entry.value.clone())
        }
    }

    /// Number of live (un-reaped) entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }

    /// True when no entries are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().entries.is_empty()
    }

    /// Reap up to `budget` expired entries from the front of the heap.
    /// Stale heap nodes (version mismatch) are discarded without
    /// touching the map.
    fn reap(inner: &mut Inner<K, V>, now: Instant, budget: usize) {
        for _ in 0..budget {
            let expired = match inner.heap.peek() {
                Some(Reverse(top)) => top.expires_at <= now,
                None => break,
            };
            if !expired {
                break;
            }
            let Some(Reverse(popped)) = inner.heap.pop() else {
                break;
            };
            if let Some(entry) = inner.entries.get(&popped.key) {
                if entry.version == popped.version && entry.expires_at <= now {
                    inner.entries.remove(&popped.key);
                }
                // else: stale heap node (entry was refreshed) — drop silently.
            }
        }
    }

    /// Evict the single next-to-expire entry. Walks past stale heap
    /// nodes (version mismatch) until it finds a live victim, then
    /// terminates.
    fn evict_one(inner: &mut Inner<K, V>) {
        while let Some(Reverse(top)) = inner.heap.pop() {
            if let Some(entry) = inner.entries.get(&top.key) {
                if entry.version == top.version {
                    inner.entries.remove(&top.key);
                    return;
                }
                // Stale heap node — keep popping.
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_roundtrip() {
        let cache: TtlCache<String, u32> = TtlCache::new(10_000, 16);
        cache.insert("a".to_owned(), 1, Duration::from_secs(60));
        assert_eq!(cache.get("a"), Some(1));
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn borrowed_key_lookup_avoids_allocation() {
        // String-keyed cache, looked up by &str.
        let cache: TtlCache<String, u32> = TtlCache::new(10, 16);
        cache.insert("https://a.test/c".to_owned(), 7, Duration::from_secs(60));
        let key: &str = "https://a.test/c";
        assert_eq!(cache.get(key), Some(7));
    }

    #[test]
    fn expired_entry_returns_none() {
        let cache: TtlCache<String, u32> = TtlCache::new(10, 16);
        cache.insert("a".to_owned(), 1, Duration::from_millis(5));
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(cache.get("a"), None);
    }

    #[test]
    fn reinsert_refreshes_and_does_not_leak_heap() {
        let cache: TtlCache<String, u32> = TtlCache::new(10_000, 32);
        // Five short-TTL versions of the same key.
        for v in 0..5 {
            cache.insert("a".to_owned(), v, Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(20));
        // A long-TTL re-insert; the reap during insert clears stale nodes.
        cache.insert("a".to_owned(), 99, Duration::from_secs(60));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get("a"), Some(99));
    }

    #[test]
    fn refresh_extends_expiry() {
        let cache: TtlCache<String, u32> = TtlCache::new(10, 16);
        cache.insert("a".to_owned(), 1, Duration::from_millis(30));
        std::thread::sleep(Duration::from_millis(15));
        // Heartbeat: re-insert before expiry extends the lifetime.
        cache.insert("a".to_owned(), 2, Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(30));
        // Original TTL has elapsed, but the refresh keeps it alive.
        assert_eq!(cache.get("a"), Some(2));
    }

    #[test]
    fn capacity_bound_evicts_next_to_expire() {
        let cache: TtlCache<String, u32> = TtlCache::new(2, 16);
        cache.insert("a".to_owned(), 1, Duration::from_secs(10));
        cache.insert("b".to_owned(), 2, Duration::from_secs(60));
        // Capacity reached; inserting a 3rd evicts the next-to-expire (a).
        cache.insert("c".to_owned(), 3, Duration::from_secs(30));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.get("b"), Some(2));
        assert_eq!(cache.get("c"), Some(3));
    }

    #[test]
    fn evict_one_terminates_under_stale_heap_pressure() {
        // Small reap budget keeps reap from clearing the stale nodes, so
        // evict_one must walk past them to find a live victim.
        let cache: TtlCache<String, u32> = TtlCache::new(2, 4);
        for v in 0..50 {
            cache.insert("a".to_owned(), v, Duration::from_secs(120));
        }
        cache.insert("b".to_owned(), 1, Duration::from_secs(60));
        assert_eq!(cache.len(), 2, "A + B at capacity");

        // Insert C — triggers evict_one past 49 stale A nodes.
        cache.insert("c".to_owned(), 2, Duration::from_secs(180));
        assert_eq!(cache.len(), 2, "still 2 after eviction");
        assert_eq!(cache.get("b"), None, "B (shortest TTL) is the victim");
        assert!(cache.get("a").is_some(), "A still present");
        assert_eq!(cache.get("c"), Some(2), "C just inserted");
    }

    #[test]
    fn lazy_delete_keeps_live_entry_after_stale_node_reaped() {
        // Insert short-lived v0, then refresh to long-lived v1 before
        // expiry. The v0 heap node is now stale; reaping it must NOT
        // evict the live v1 entry.
        let cache: TtlCache<String, u32> = TtlCache::new(10, 16);
        cache.insert("a".to_owned(), 0, Duration::from_millis(5));
        cache.insert("a".to_owned(), 1, Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(20));
        // get triggers a reap that pops the stale (expired) v0 node.
        assert_eq!(cache.get("a"), Some(1));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn insert_if_absent_rejects_duplicate_within_window() {
        let cache: TtlCache<String, ()> = TtlCache::new(10, 16);
        assert!(cache.insert_if_absent("jti-1".to_owned(), (), Duration::from_secs(60)));
        assert!(!cache.insert_if_absent("jti-1".to_owned(), (), Duration::from_secs(60)));
        assert!(cache.insert_if_absent("jti-2".to_owned(), (), Duration::from_secs(60)));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn insert_if_absent_accepts_after_expiry() {
        let cache: TtlCache<String, ()> = TtlCache::new(10, 16);
        assert!(cache.insert_if_absent("jti".to_owned(), (), Duration::from_millis(5)));
        std::thread::sleep(Duration::from_millis(20));
        assert!(cache.insert_if_absent("jti".to_owned(), (), Duration::from_secs(60)));
    }
}
