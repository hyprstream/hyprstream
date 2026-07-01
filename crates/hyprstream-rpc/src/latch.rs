//! Latched completion / retained terminal state (EV7, epic #600).
//!
//! First-class fan-out completion: a resource's terminal value (task exit,
//! model-load result, fd close/EOF, …) is retained host-side in a
//! [`TerminalStore`] (the authoritative value, plaintext-at-owner — NOT the
//! content-blind relay, NOT [`crate::stream_info::StreamOpt::Retention`]), and
//! served to late watchers immediately while early watchers block on the
//! EventService edge.
//!
//! [`read_then_subscribe`] is the one helper that subsumes `load --wait` and
//! the P9 `/task/<id>/exit` pattern under one shape: serve the retained value
//! if present, else subscribe to the live edge and block until a terminal
//! event arrives.
//!
//! # Await contract
//!
//! [`read_then_subscribe`] is a real awaitable `Future` — it `.await`s
//! [`crate::events::EventSubscriber::recv`], which registers a waker with the
//! runtime. It is NOT correct-only-under-poll-spin. A consumer that drives VFS
//! futures via a noop-waker spin loop (e.g. the TUI `chat_app.rs` `poll_local`
//! driver, being replaced with a real async driver in the workers/P9-task lane,
//! epic #608) would busy-spin on a blocking read here — consumers MUST drive
//! this with a real async runtime.
//!
//! # Lane split (epic #600 vs epic #608)
//!
//! This module is the **generic EventService-side primitive**. The concrete
//! per-resource Mount-served files (`/task/<id>/{exit,status,fd}`) and the
//! per-type terminal payloads (`Exited(code)|Killed(sig)|Failed(reason)`,
//! `Loaded(handle)|LoadFailed`, fd close/EOF) are owned by the workers/P9-task
//! lane (epic #608): they instantiate [`TerminalState`]-style payloads, own a
//! [`TerminalStore`], and call [`read_then_subscribe`]. This crate provides
//! only the generic envelope + store + helper + the await/liveness contracts.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parking_lot::RwLock;

use crate::events::EventSubscriber;

/// A resource identifier for a latched terminal value (e.g. a task id, a
/// model load-attempt id). Cheap to clone/hash/compare.
pub trait ResourceKey: Eq + Hash + Clone + Send + Sync + std::fmt::Debug {}
impl<T: Eq + Hash + Clone + Send + Sync + std::fmt::Debug> ResourceKey for T {}

/// Marker for a per-type terminal payload. Implementations live in the consumer
/// (epic #608): task-exit, model-load, fd-close, … They carry their own value
/// type; this trait only pins the "I am a terminal payload" contract so the
/// store/helper stay generic.
pub trait TerminalState: Clone + Send + Sync {}
impl<T: Clone + Send + Sync> TerminalState for T {}

/// A latched terminal value — monotonic write-once. The owner of a resource
/// latches exactly one of these when the resource terminates.
///
/// Generic over the per-type payload `V`. A successful latch IS the terminal
/// edge; a second latch attempt on an already-terminal resource is rejected
/// ([`TerminalStore::latch`] returns `false`), so the value can never be
/// overwritten — matching the Plan9-style "terminal is monotonic write-once"
/// contract.
#[derive(Clone, Debug)]
pub struct Terminal<V: TerminalState> {
    /// The per-type terminal payload.
    pub value: V,
    /// Identity that latched this terminal (the resource owner / publisher, or
    /// the reaper for an abnormal-death fallback). Auditable.
    pub latched_by: String,
}

/// Monotonic-write-once retained-terminal-state store — the authoritative,
/// host-side, plaintext-at-owner value for each resource.
///
/// This is the "file holds the truth" half of Plan9-style completion: a late
/// watcher (one that subscribes AFTER termination) is served from here
/// immediately, without replaying the event stream. Distinct from
/// [`crate::stream_info::StreamOpt::Retention`] (a best-effort content-blind
/// transport replay buffer) and from the relay (which never holds plaintext).
#[derive(Default)]
pub struct TerminalStore<K: ResourceKey, V: TerminalState> {
    entries: Arc<RwLock<HashMap<K, Terminal<V>>>>,
}

impl<K: ResourceKey, V: TerminalState> TerminalStore<K, V> {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Latch a terminal value for `key`. Monotonic write-once: returns `true`
    /// if this call latched (the key was not already terminal), `false` if the
    /// key was already terminal (the existing value is NEVER overwritten).
    pub fn latch(&self, key: K, terminal: Terminal<V>) -> bool {
        let mut entries = self.entries.write();
        if entries.contains_key(&key) {
            return false;
        }
        entries.insert(key, terminal);
        true
    }

    /// Return the retained terminal for `key`, if any. Serves a late watcher
    /// (subject to [`TerminalAuthz`] at the call site — see
    /// [`read_then_subscribe`]).
    pub fn get(&self, key: &K) -> Option<Terminal<V>> {
        self.entries.read().get(key).cloned()
    }

    /// Whether `key` is already terminal.
    pub fn is_latched(&self, key: &K) -> bool {
        self.entries.read().contains_key(key)
    }

    /// Latch a fallback terminal for every `key` that is not already terminal.
    /// Called by the resource owner's death-detection (reaper / liveness
    /// timeout) so a fan-out [`read_then_subscribe`] waiter on a resource whose
    /// owner died never blocks forever — the EV6-liveness obligation. Returns
    /// the keys that were reaped (newly latched with `fallback`).
    ///
    /// Already-latched keys are left untouched — a real terminal always beats a
    /// synthetic `Failed`/`EOF`. The caller (not this store) owns owner→key
    /// bookkeeping and death-detection; this just applies the fallback to the
    /// supplied keys.
    pub fn reap_unlatched<'k, I, F>(&self, keys: I, fallback: F) -> Vec<K>
    where
        I: IntoIterator<Item = &'k K>,
        K: 'k,
        F: Fn(&K) -> Terminal<V>,
    {
        let mut reaped = Vec::new();
        let mut entries = self.entries.write();
        for key in keys {
            if entries.contains_key(key) {
                continue;
            }
            let terminal = fallback(key);
            entries.insert(key.clone(), terminal);
            reaped.push(key.clone());
        }
        reaped
    }
}

impl<K: ResourceKey, V: TerminalState> Clone for TerminalStore<K, V> {
    /// Cheap clone — the store is `Arc`-shared internally (same authoritative
    /// state visible to all clones), so a watcher and the owner can hold their
    /// own handles to the same store.
    fn clone(&self) -> Self {
        Self {
            entries: Arc::clone(&self.entries),
        }
    }
}

/// Authorization hook for a late read of a retained terminal value. The
/// late-watcher file read must enforce the same MAC as the subscribe; until
/// EV4 (#604) wires real MAC this is **fail-closed** — [`DenyAllTerminalAuthz`]
/// denies every late read, so a late watcher cannot observe a retained value
/// yet. An EARLY watcher (one that subscribes before the edge) is unaffected:
/// it receives the live edge through the normal subscribe path (whose authz is
/// separate).
//
// TODO(EV4 #604): replace DenyAllTerminalAuthz with a MAC-backed impl that
// enforces subject.ctx ⊒ object.label on the retained-value read.
pub trait TerminalAuthz<K: ResourceKey> {
    /// Authorize `watcher` reading the retained terminal for `key`.
    fn authorize_read(&self, key: &K, watcher: &str) -> bool;
}

/// Fail-closed default [`TerminalAuthz`] until EV4 (#604). Denies all late
/// reads — a watcher must subscribe before the edge to receive the terminal
/// value live. Correct ZSP default; not a security gap (no plaintext is
/// released to a late watcher the MAC hasn't authorized).
pub struct DenyAllTerminalAuthz;
impl<K: ResourceKey> TerminalAuthz<K> for DenyAllTerminalAuthz {
    fn authorize_read(&self, _key: &K, _watcher: &str) -> bool {
        false
    }
}

/// Latched-completion read: serve the retained terminal for `key` if already
/// latched (and `authz` permits the late read), else subscribe to the live
/// EventService edge via `subscriber` and block until an event that `classify`
/// identifies as terminal, then latch + return it.
///
/// Subsumes `load --wait` and P9 `/task/<id>/exit` under one pattern. The
/// caller must have already configured `subscriber` with the relevant
/// subscription pattern(s) before the first `.await`.
///
/// **Await contract:** this is a real `Future` (it `.await`s
/// [`EventSubscriber::recv`]). Consumers MUST drive it with a real async
/// runtime — a noop-waker poll-spin driver busy-loops (see module docs).
///
/// No lock is held across `.await`: the fast-path store read drops its guard
/// before the `subscriber.recv().await` loop.
pub async fn read_then_subscribe<K, V, F>(
    store: &TerminalStore<K, V>,
    subscriber: &mut EventSubscriber,
    authz: &impl TerminalAuthz<K>,
    watcher: &str,
    key: K,
    classify: F,
) -> Result<Terminal<V>>
where
    K: ResourceKey,
    V: TerminalState,
    F: Fn(&str, &[u8]) -> Option<V>,
{
    // Fast path — late watcher: the owner already latched the terminal. Serve
    // from the retain-store (subject to MAC authz on the read).
    if let Some(existing) = store.get(&key) {
        if authz.authorize_read(&key, watcher) {
            return Ok(existing);
        }
        return Err(anyhow!(
            "late read of terminal {key:?} denied (EV4 authz stub deny-all)"
        ));
    }

    // Slow path — early watcher: subscribe to the live edge until a terminal
    // event arrives. The store guard from `get` is already dropped; nothing is
    // held across the `.await`.
    loop {
        let (topic, payload) = subscriber.recv().await?;
        if let Some(value) = classify(&topic, &payload) {
            let terminal = Terminal {
                value,
                latched_by: watcher.to_owned(),
            };
            // Best-effort latch: the owner (or a racing watcher) may have
            // latched first — first latch wins, the value is the same. This
            // fan-out latch is what lets a *subsequent* late watcher hit the
            // fast path instead of re-subscribing.
            let _ = store.latch(key.clone(), terminal.clone());
            return Ok(terminal);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // Trivial example terminal payload (the real per-type payloads — task-exit,
    // model-load, fd-close — live in the epic #608 consumer crate).
    #[derive(Clone, Debug, PartialEq)]
    struct ExampleExit(u8);

    fn term(code: u8, by: &str) -> Terminal<ExampleExit> {
        Terminal {
            value: ExampleExit(code),
            latched_by: by.to_owned(),
        }
    }

    #[test]
    fn latch_is_monotonic_write_once() {
        let store: TerminalStore<String, ExampleExit> = TerminalStore::new();
        assert!(store.latch("t1".to_owned(), term(0, "owner")));
        // Second latch is rejected — the existing terminal is never overwritten.
        assert!(!store.latch("t1".to_owned(), term(1, "owner")));
        // The retained value is the FIRST (monotonic).
        assert_eq!(store.get(&"t1".to_owned()).unwrap().value.0, 0);
    }

    #[test]
    fn is_latched_and_get_reflect_state() {
        let store: TerminalStore<u64, ExampleExit> = TerminalStore::new();
        assert!(!store.is_latched(&7));
        assert!(store.latch(7, term(0, "owner")));
        assert!(store.is_latched(&7));
        assert_eq!(store.get(&7).unwrap().latched_by, "owner");
        assert!(store.get(&8).is_none());
    }

    #[test]
    fn reap_unlatched_latches_fallback_only_for_unlatched() {
        let store: TerminalStore<u32, ExampleExit> = TerminalStore::new();
        // One resource already terminated with a real exit (code 0).
        assert!(store.latch(1, term(0, "owner")));
        // Owner dies with resources {1,2,3} unlatched-or-not.
        let keys = [1u32, 2, 3];
        let reaped = store.reap_unlatched(&keys, |k| Terminal {
            // Synthetic Failed-equivalent (code 255 here) for the example.
            value: ExampleExit(255),
            latched_by: format!("reaper@{k}"),
        });
        // Only the previously-unlatched keys were reaped.
        assert_eq!(reaped, vec![2, 3]);
        // The already-terminal key kept its REAL terminal (monotonic).
        assert_eq!(store.get(&1).unwrap().value.0, 0);
        // Reaped keys got the fallback.
        assert_eq!(store.get(&2).unwrap().value.0, 255);
        assert_eq!(store.get(&3).unwrap().latched_by, "reaper@3");
    }

    #[test]
    fn deny_all_authz_blocks_late_read() {
        // The fast path of read_then_subscribe cannot be exercised without a
        // live EventSubscriber (tokio + moq), but the authz decision itself is
        // synchronous — verify the fail-closed default directly.
        let store: TerminalStore<String, ExampleExit> = TerminalStore::new();
        store.latch("t1".to_owned(), term(0, "owner"));
        let authz = DenyAllTerminalAuthz;
        assert!(!authz.authorize_read(&"t1".to_owned(), "watcher"));
    }

    #[test]
    fn store_clone_shares_authoritative_state() {
        // The owner and a watcher hold separate handles to the SAME store.
        let owner_store: TerminalStore<u64, ExampleExit> = TerminalStore::new();
        let watcher_store = owner_store.clone();
        assert!(owner_store.latch(42, term(7, "owner")));
        // Watcher sees the owner's latch through its own handle.
        assert!(watcher_store.is_latched(&42));
        assert_eq!(watcher_store.get(&42).unwrap().value.0, 7);
    }

    // A full `read_then_subscribe` end-to-end test needs a running moq EventBus
    // + tokio runtime (out of scope for a lib unit test here); the fast-path
    // authz + monotonic-latch + reap contracts above are the unit-testable
    // invariants. The slow-path loop is a thin `subscriber.recv().await` whose
    // await-contract is guaranteed by EventSubscriber::recv being a real Future.
}
