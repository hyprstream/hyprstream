//! Replay admission store — partitioned by disposition, fail-closed on
//! capacity.
//!
//! The store is the one place an accepted proof is recorded so exact replay
//! denies. It is partitioned at minimum by disposition class (unattributed vs
//! authenticated) so saturation of the unattributed partition cannot delay or
//! deny authenticated admission. Within a partition, capacity pressure fails
//! closed — an unexpired accepted record is never evicted.
//!
//! This in-memory implementation follows the same `parking_lot::RwLock` +
//! `HashMap` pattern as `InMemoryJtiBlocklist` and `InMemoryNonceCache`. A
//! Valkey/Redis backing store can implement the same trait on the #1256
//! substrate for domain-wide linearizable admission.

use std::collections::HashMap;

use super::{ProofDisposition, RequestId};

/// The replay key — one per accepted proof. Composed of the signer identity
/// thumbprint and the request ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReplayKey {
    /// SHA-256 thumbprint of the canonical encoding of the signer identity
    /// data (credential-bound primary signer-suite thumbprint for
    /// authenticated proofs; plan/key-set thumbprint for unattributed proofs).
    pub signer_thumbprint: [u8; 32],
    /// The proof's 128-bit request ID (CWT `cti`).
    pub request_id: RequestId,
}

/// The replay admission domain trait.
///
/// `check_and_insert` atomically admits a fresh key or rejects a replay. On a
/// hit (replay) it returns `AdmissionResult::Replayed`. On capacity pressure
/// it returns `AdmissionResult::Failed` (fail-closed) — it never evicts an
/// unexpired accepted record.
pub trait ReplayStore: Send + Sync {
    /// Atomically check and insert a replay key.
    ///
    /// `expires_at` is the Unix-seconds timestamp at which the entry can be
    /// garbage-collected (the proof's `exp`, further bounded by
    /// `challenge_accept_until` for unattributed proofs).
    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ReplayKey,
        expires_at: u64,
    ) -> AdmissionResult;
}

/// The outcome of a replay-admission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionResult {
    /// The key was fresh and has been admitted.
    Admitted,
    /// The key was a replay of an already-admitted proof.
    Replayed,
    /// The store is at capacity and cannot admit without evicting an
    /// unexpired record. Fail-closed.
    Failed,
}

/// In-memory replay store with per-partition capacity and fail-closed
/// behavior.
///
/// Uses `parking_lot::RwLock` on non-wasm targets and `std::sync::RwLock` on
/// wasm32. Entries are garbage-collected opportunistically on insert; an
/// unexpired entry is never evicted.
pub struct InMemoryReplayStore {
    #[cfg(not(target_arch = "wasm32"))]
    authenticated: parking_lot::Mutex<HashMap<ReplayKey, u64>>,
    #[cfg(target_arch = "wasm32")]
    authenticated: std::sync::Mutex<HashMap<ReplayKey, u64>>,
    #[cfg(not(target_arch = "wasm32"))]
    unattributed: parking_lot::Mutex<HashMap<ReplayKey, u64>>,
    #[cfg(target_arch = "wasm32")]
    unattributed: std::sync::Mutex<HashMap<ReplayKey, u64>>,
    /// Maximum entries per partition before fail-closed.
    max_per_partition: usize,
}

impl Default for InMemoryReplayStore {
    fn default() -> Self {
        Self::new(100_000)
    }
}

impl InMemoryReplayStore {
    pub fn new(max_per_partition: usize) -> Self {
        Self {
            authenticated: Default::default(),
            unattributed: Default::default(),
            max_per_partition,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lock(&self, partition: ProofDisposition) -> parking_lot::MutexGuard<'_, HashMap<ReplayKey, u64>> {
        match partition {
            ProofDisposition::Authenticated => self.authenticated.lock(),
            ProofDisposition::Unattributed => self.unattributed.lock(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn lock(
        &self,
        partition: ProofDisposition,
    ) -> std::sync::MutexGuard<'_, HashMap<ReplayKey, u64>> {
        let guard = match partition {
            ProofDisposition::Authenticated => self.authenticated.lock(),
            ProofDisposition::Unattributed => self.unattributed.lock(),
        };
        guard.expect("replay store lock poisoned")
    }
}

impl ReplayStore for InMemoryReplayStore {
    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ReplayKey,
        expires_at: u64,
    ) -> AdmissionResult {
        let mut map = self.lock(partition);

        // Replay check.
        if map.contains_key(key) {
            return AdmissionResult::Replayed;
        }

        // Opportunistic GC: remove expired entries.
        let now = current_unix_seconds();
        map.retain(|_, exp| *exp > now);

        // Capacity check: fail-closed if at limit.
        if map.len() >= self.max_per_partition {
            return AdmissionResult::Failed;
        }

        // Admit.
        map.insert(key.clone(), expires_at);
        AdmissionResult::Admitted
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn current_unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn current_unix_seconds() -> u64 {
    // On wasm32, use the JS performance API or return 0 (tests inject timestamps).
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(n: u8) -> ReplayKey {
        ReplayKey {
            signer_thumbprint: [n; 32],
            request_id: [n; 16],
        }
    }

    #[test]
    fn fresh_proof_admits() {
        let store = InMemoryReplayStore::default();
        let key = make_key(1);
        assert_eq!(
            store.check_and_insert(ProofDisposition::Authenticated, &key, u64::MAX),
            AdmissionResult::Admitted
        );
    }

    #[test]
    fn replay_denies() {
        let store = InMemoryReplayStore::default();
        let key = make_key(2);
        store.check_and_insert(ProofDisposition::Authenticated, &key, u64::MAX);
        assert_eq!(
            store.check_and_insert(ProofDisposition::Authenticated, &key, u64::MAX),
            AdmissionResult::Replayed
        );
    }

    #[test]
    fn partitions_are_independent() {
        let store = InMemoryReplayStore::default();
        let key = make_key(3);
        store.check_and_insert(ProofDisposition::Authenticated, &key, u64::MAX);
        // Same key in a different partition should also admit.
        assert_eq!(
            store.check_and_insert(ProofDisposition::Unattributed, &key, u64::MAX),
            AdmissionResult::Admitted
        );
    }

    #[test]
    fn capacity_fails_closed() {
        let store = InMemoryReplayStore::new(2);
        let k1 = make_key(1);
        let k2 = make_key(2);
        let k3 = make_key(3);
        store.check_and_insert(ProofDisposition::Unattributed, &k1, u64::MAX);
        store.check_and_insert(ProofDisposition::Unattributed, &k2, u64::MAX);
        // At capacity — k3 must fail closed, not evict k1/k2.
        assert_eq!(
            store.check_and_insert(ProofDisposition::Unattributed, &k3, u64::MAX),
            AdmissionResult::Failed
        );
        // k1 and k2 are still admitted (not evicted).
        assert_eq!(
            store.check_and_insert(ProofDisposition::Unattributed, &k1, u64::MAX),
            AdmissionResult::Replayed
        );
    }
}
