//! Proof replay admission and challenge manager wiring.
//!
//! "Admitted once" means once per **replay admission domain** — the complete
//! set of verifier instances that can admit requests for one service domain —
//! never once per node (§4.6). A deployment satisfies that with exactly one
//! of three shapes, and every store must say which one it is:
//!
//! 1. a single verifier instance;
//! 2. a shared store with linearizable check-and-insert; or
//! 3. per-node stores with **mandatory** replay-namespace-affine routing,
//!    where failover to a node holding no history for that namespace denies
//!    rather than admits.
//!
//! There is exactly one store family here. Request-proof replay admission and
//! one-shot credential consumption are two operations on the same trait, not
//! two stores: a deployment cannot satisfy one and silently miss the other.
//! One-shot consumption additionally requires a domain-wide linearizable
//! consume — key-affine routing is not sufficient for it, because the
//! credential, not the proof signer set, is the admission key and may be
//! presented at any node.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use super::{ProofDisposition, RequestId};

/// A capacity-bounded set of admitted keys with **expiry-ordered** reclamation.
///
/// Each key holds a signed expiry. A secondary `BTreeMap` orders keys by
/// `(expires_at, seq)`, so garbage collection pops only the records that have
/// actually expired — from the front, in expiry order — and stops at the first
/// unexpired one. Reclamation therefore touches O(expired) records, never the
/// whole partition, so a non-replay admission far below capacity is O(log n),
/// not an O(n) full scan on every request. The invariants Opus F-D must
/// preserve are unchanged: an unexpired accepted record is never evicted, and a
/// full partition fails closed rather than making room by eviction.
struct ExpiryMap<K: std::hash::Hash + Eq + Clone> {
    /// key -> (expires_at, seq)
    entries: HashMap<K, (u64, u64)>,
    /// (expires_at, seq) -> key, for expiry-ordered reclamation.
    by_expiry: BTreeMap<(u64, u64), K>,
    /// Monotonic tie-breaker so equal expiries never collide in `by_expiry`.
    next_seq: u64,
}

impl<K: std::hash::Hash + Eq + Clone> Default for ExpiryMap<K> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            by_expiry: BTreeMap::new(),
            next_seq: 0,
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone> ExpiryMap<K> {
    /// Reclaim only the records whose signed expiry is at or before `now`,
    /// in expiry order, stopping at the first unexpired record.
    // A `while let` on `first_key_value()` would hold an immutable borrow of
    // `by_expiry` across the `remove()` below (the scrutinee temporary lives to
    // the end of the loop body), so the peek is a separate statement that copies
    // the key and releases the borrow first.
    #[allow(clippy::while_let_loop)]
    fn gc(&mut self, now: u64) {
        loop {
            // Peek the earliest-expiring record and copy its ordering key, so
            // the immutable borrow ends before the mutation below.
            let front = match self.by_expiry.first_key_value() {
                Some((&front, _)) => front,
                None => break,
            };
            if front.0 > now {
                break; // earliest-expiring record is still live; nothing more to reclaim.
            }
            if let Some(key) = self.by_expiry.remove(&front) {
                // Only drop the primary entry if it still points at this record;
                // a re-inserted key would carry a newer (exp, seq).
                if self.entries.get(&key) == Some(&front) {
                    self.entries.remove(&key);
                }
            }
        }
    }

    /// Admit `key` under fail-closed capacity, or report a replay.
    fn admit(&mut self, key: &K, expires_at: u64, max: usize, now: u64) -> ProofAdmissionResult {
        if self.entries.contains_key(key) {
            return ProofAdmissionResult::Replayed;
        }
        // Reclaim expired records first — bounded to the expired set, never a
        // full-partition scan.
        self.gc(now);
        if self.entries.len() >= max {
            return ProofAdmissionResult::Failed;
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.entries.insert(key.clone(), (expires_at, seq));
        self.by_expiry.insert((expires_at, seq), key.clone());
        ProofAdmissionResult::Admitted
    }
}

// ---------------------------------------------------------------------------
// Replay admission store
// ---------------------------------------------------------------------------

/// The replay admission key: (signer namespace thumbprint, request_id).
///
/// The thumbprint is the credential-bound primary signer-suite thumbprint for
/// authenticated proofs and the (plan, key set) thumbprint for unattributed
/// ones. It is produced by verification, never taken from the wire.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProofReplayKey {
    pub signer_thumbprint: [u8; 32],
    pub request_id: RequestId,
}

/// A one-shot credential's admission identity: issuer plus the credential's
/// own identifier bytes (JWT `jti` text or CWT `cti` bytes, kept in their own
/// namespace — two issuers producing the same value never collide).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OneShotCredentialId {
    pub issuer: String,
    pub value: Vec<u8>,
}

/// The outcome of a replay admission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofAdmissionResult {
    /// Admitted exactly once in this domain.
    Admitted,
    /// This exact key was already admitted.
    Replayed,
    /// The store could not guarantee admission — capacity, backend
    /// unavailability, or a guarantee the deployment does not provide. Always
    /// a denial; never an admission.
    Failed,
}

/// How a deployment satisfies domain-wide "admitted once" (§4.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDomainGuarantee {
    /// Exactly one verifier instance admits for this service domain.
    SingleVerifierInstance,
    /// A shared store providing linearizable check-and-insert across every
    /// verifier in the domain.
    LinearizableSharedStore,
    /// Per-node stores with mandatory replay-namespace-affine routing. Every
    /// admission must first confirm this node owns the key's namespace.
    NamespaceAffineRouting,
}

/// Proof replay admission store.
///
/// Implementations are the substrate (in-process for a single verifier, or a
/// shared linearizable backend); this trait is the only admission surface.
pub trait ProofReplayStore: Send + Sync {
    /// Which domain-wide guarantee this deployment provides. Dispatch refuses
    /// to admit under a guarantee the store cannot honour for the key at hand.
    fn domain_guarantee(&self) -> ReplayDomainGuarantee;

    /// Whether this node owns the replay namespace for `signer_thumbprint`.
    ///
    /// Only consulted under [`ReplayDomainGuarantee::NamespaceAffineRouting`],
    /// where a node that does not own the namespace holds none of its history
    /// and MUST deny rather than admit. The other guarantees own every
    /// namespace by construction.
    fn owns_namespace(&self, signer_thumbprint: &[u8; 32]) -> bool {
        let _ = signer_thumbprint;
        !matches!(
            self.domain_guarantee(),
            ReplayDomainGuarantee::NamespaceAffineRouting
        )
    }

    /// Atomically admit a fresh key or reject a replay, within the partition
    /// for `partition`. Capacity pressure MUST NOT evict an unexpired accepted
    /// record: the partition fails closed instead.
    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ProofReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult;

    /// Consume a one-shot credential ID exactly once in the domain.
    ///
    /// This is the single admission action for a one-shot credential: it does
    /// not additionally create a proof replay entry. It requires a domain-wide
    /// linearizable consume; a store whose guarantee is namespace-affine
    /// routing MUST fail rather than consume locally.
    fn consume_one_shot_credential(
        &self,
        id: &OneShotCredentialId,
        expires_at: u64,
    ) -> ProofAdmissionResult;
}

/// In-memory replay store for a deployment with exactly one verifier instance.
///
/// Partitioned by disposition, so saturation of the unattributed partition can
/// neither delay nor deny authenticated admission. Capacity fails closed: an
/// unexpired accepted record is never evicted to make room.
///
/// This shape is sound **only** as
/// [`ReplayDomainGuarantee::SingleVerifierInstance`]. A multi-instance
/// deployment needs a shared linearizable backend implementing the same trait;
/// there is deliberately no in-memory constructor that claims a guarantee this
/// type cannot provide.
pub struct InMemoryProofReplayStore {
    authenticated: parking_lot::Mutex<ExpiryMap<ProofReplayKey>>,
    unattributed: parking_lot::Mutex<ExpiryMap<ProofReplayKey>>,
    one_shot: parking_lot::Mutex<ExpiryMap<OneShotCredentialId>>,
    max_per_partition: usize,
}

impl InMemoryProofReplayStore {
    /// Construct the single-verifier-instance store. The name is the
    /// deployment assertion: installing this in a multi-instance domain breaks
    /// the domain-wide guarantee.
    pub fn single_verifier_instance(max_per_partition: usize) -> Self {
        Self {
            authenticated: Default::default(),
            unattributed: Default::default(),
            one_shot: Default::default(),
            max_per_partition,
        }
    }

    fn lock(
        &self,
        partition: ProofDisposition,
    ) -> parking_lot::MutexGuard<'_, ExpiryMap<ProofReplayKey>> {
        match partition {
            ProofDisposition::Authenticated => self.authenticated.lock(),
            ProofDisposition::Unattributed => self.unattributed.lock(),
        }
    }
}

impl ProofReplayStore for InMemoryProofReplayStore {
    fn domain_guarantee(&self) -> ReplayDomainGuarantee {
        ReplayDomainGuarantee::SingleVerifierInstance
    }

    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ProofReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        let now = current_unix_seconds();
        let max = self.max_per_partition;
        self.lock(partition).admit(key, expires_at, max, now)
    }

    fn consume_one_shot_credential(
        &self,
        id: &OneShotCredentialId,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        let now = current_unix_seconds();
        let max = self.max_per_partition;
        self.one_shot.lock().admit(id, expires_at, max, now)
    }
}

// ---------------------------------------------------------------------------
// Single-verifier sole-membership lease (§4.6 topology enforcement)
// ---------------------------------------------------------------------------

/// An exclusive, process-lifetime lease proving this process is the **sole**
/// verifier for a replay admission domain (§4.6, shape 1).
///
/// The `SingleVerifierInstance` guarantee is only sound if exactly one process
/// admits for the domain. An operator env string alone cannot establish that —
/// two replicas can each set it and silently recreate a per-process replay
/// split. This lease converts the claim into an OS-enforced fact: it holds an
/// **exclusive, non-blocking** advisory lock (`flock(LOCK_EX|LOCK_NB)`) on an
/// operator-configured lease path for the whole process lifetime. A second
/// process attempting to acquire the same lease fails, so its proof admission
/// stays closed rather than admitting under a guarantee it cannot honour.
///
/// The lease path's storage defines the domain the exclusion spans: a
/// same-host domain uses any local path; a multi-host domain MUST place the
/// lease on storage shared and lock-coherent across every verifier (an
/// operator responsibility the mechanism cannot infer, but which — unlike a
/// bare env string — it does enforce wherever the lock is actually coherent).
#[cfg(not(target_arch = "wasm32"))]
pub struct SingleVerifierLease {
    // The open file whose exclusive flock is released when this handle drops.
    // Kept process-global so the lock outlives startup.
    _file: std::fs::File,
    path: std::path::PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl SingleVerifierLease {
    /// Acquire the exclusive sole-verifier lease, or fail closed.
    ///
    /// Returns `Err` when another process already holds the lease (the domain
    /// already has a verifier) or the path cannot be opened/locked. The caller
    /// MUST NOT install a `SingleVerifierInstance` store without holding this.
    pub fn acquire(path: &std::path::Path) -> Result<Self> {
        use nix::fcntl::{flock, FlockArg};
        use std::os::fd::AsRawFd;

        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| {
                anyhow::anyhow!("cannot open single-verifier lease {}: {e}", path.display())
            })?;
        // Non-blocking exclusive lock: a second holder fails immediately rather
        // than queueing, so a misconfigured second replica denies at once.
        flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock).map_err(|e| {
            anyhow::anyhow!(
                "another process already holds the single-verifier lease {} \
                 (this domain already has a verifier): {e}",
                path.display()
            )
        })?;
        Ok(Self {
            _file: file,
            path: path.to_path_buf(),
        })
    }

    /// The lease path (for diagnostics).
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

// The held lease is parked here so the exclusive lock lives for the whole
// process, not just past startup. Dropping it would release the lock and let a
// second verifier acquire it.
#[cfg(not(target_arch = "wasm32"))]
static SINGLE_VERIFIER_LEASE: OnceLock<SingleVerifierLease> = OnceLock::new();

/// Park an acquired sole-verifier lease for the process lifetime.
#[cfg(not(target_arch = "wasm32"))]
pub fn hold_single_verifier_lease(
    lease: SingleVerifierLease,
) -> std::result::Result<(), SingleVerifierLease> {
    SINGLE_VERIFIER_LEASE.set(lease)
}

// Process-global registration. There is no auto-install: an absent store is a
// denial, never a locally-defaulted admission.
static PROOF_REPLAY_STORE: OnceLock<Box<dyn ProofReplayStore>> = OnceLock::new();

pub fn set_global_proof_replay_store(
    store: Box<dyn ProofReplayStore>,
) -> std::result::Result<(), Box<dyn ProofReplayStore>> {
    PROOF_REPLAY_STORE.set(store)
}

pub fn global_proof_replay_store() -> Option<&'static dyn ProofReplayStore> {
    PROOF_REPLAY_STORE.get().map(|s| &**s)
}

/// Admit one request proof, enforcing the domain guarantee before the store
/// is consulted.
///
/// Under namespace-affine routing a node that does not own the key's namespace
/// holds none of its history, so it denies instead of admitting locally.
pub fn admit_request_proof(
    store: &dyn ProofReplayStore,
    partition: ProofDisposition,
    key: &ProofReplayKey,
    expires_at: u64,
) -> ProofAdmissionResult {
    if !store.owns_namespace(&key.signer_thumbprint) {
        return ProofAdmissionResult::Failed;
    }
    store.check_and_insert(partition, key, expires_at)
}

/// Consume one one-shot credential ID, enforcing the domain guarantee first.
///
/// Namespace-affine routing is keyed by the proof signer set, but a one-shot
/// credential can be presented at any node, so affinity cannot make its
/// consumption domain-wide. Such a deployment fails closed here rather than
/// consuming locally and admitting the same credential twice.
pub fn consume_one_shot_credential(
    store: &dyn ProofReplayStore,
    id: &OneShotCredentialId,
    expires_at: u64,
) -> ProofAdmissionResult {
    if store.domain_guarantee() == ReplayDomainGuarantee::NamespaceAffineRouting {
        return ProofAdmissionResult::Failed;
    }
    store.consume_one_shot_credential(id, expires_at)
}

// ---------------------------------------------------------------------------
// Process-global challenge manager
// ---------------------------------------------------------------------------

static CHALLENGE_MANAGER: OnceLock<super::challenge::ChallengeManager> = OnceLock::new();

/// Install a challenge manager for the process.
pub fn set_global_challenge_manager(
    mgr: super::challenge::ChallengeManager,
) -> std::result::Result<(), super::challenge::ChallengeManager> {
    CHALLENGE_MANAGER.set(mgr)
}

pub fn global_challenge_manager() -> Option<&'static super::challenge::ChallengeManager> {
    CHALLENGE_MANAGER.get()
}

// ---------------------------------------------------------------------------
// Helper: compute unattributed replay thumbprint from proof
// ---------------------------------------------------------------------------

/// Compute the SHA-256 thumbprint of the canonical (plan, key_set) tuple
/// for unattributed proof replay keying.
pub fn unattributed_thumbprint(protected_bytes: &[u8]) -> Result<[u8; 32]> {
    let protected: ciborium::Value =
        ciborium::de::from_reader(&mut std::io::Cursor::new(protected_bytes))
            .map_err(|e| anyhow::anyhow!("protected header decode: {e}"))?;

    let map = match &protected {
        ciborium::Value::Map(m) => m,
        _ => bail!("protected header not a map"),
    };

    let plan_val = map
        .iter()
        .find(|(k, _)| {
            matches!(k,
                ciborium::Value::Integer(i)
                if i128::from(*i) == super::HEADER_HS_SIGNATURE_PLAN as i128
            )
        })
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow::anyhow!("no hs_signature_plan"))?;

    let key_set_val = map
        .iter()
        .find(|(k, _)| {
            matches!(k,
                ciborium::Value::Integer(i)
                if i128::from(*i) == super::HEADER_HS_UNATTRIBUTED_KEY_SET as i128
            )
        })
        .map(|(_, v)| v)
        .ok_or_else(|| anyhow::anyhow!("no hs_unattributed_key_set"))?;

    let tuple = ciborium::Value::Array(vec![plan_val.clone(), key_set_val.clone()]);
    let mut tuple_bytes = Vec::new();
    ciborium::ser::into_writer(&tuple, &mut tuple_bytes)
        .map_err(|e| anyhow::anyhow!("tuple encode: {e}"))?;

    let mut hasher = Sha256::new();
    hasher.update(b"hs-proof-key-set-replay-v1");
    hasher.update(&tuple_bytes);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    Ok(out)
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
    0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn key(thumb: u8, id: u8) -> ProofReplayKey {
        ProofReplayKey {
            signer_thumbprint: [thumb; 32],
            request_id: [id; 16],
        }
    }

    fn far_future() -> u64 {
        current_unix_seconds() + 3_600
    }

    /// The expiry-ordered reclamation admits a new record once an old one has
    /// expired, reclaims only expired records (stops at the first live one),
    /// and never evicts an unexpired record to make room.
    #[test]
    fn expiry_map_reclaims_only_expired_records_in_order() {
        let mut m: ExpiryMap<u64> = ExpiryMap::default();
        let max = 2;
        // Two records: one expires at t=100, one is long-lived (t=1000).
        assert_eq!(m.admit(&1, 100, max, 50), ProofAdmissionResult::Admitted);
        assert_eq!(m.admit(&2, 1000, max, 50), ProofAdmissionResult::Admitted);
        // At capacity: a third at t<100 fails closed (nothing expired yet), and
        // the live records are untouched.
        assert_eq!(m.admit(&3, 200, max, 60), ProofAdmissionResult::Failed);
        assert!(m.entries.contains_key(&1));
        assert!(m.entries.contains_key(&2));
        // After key 1 expires, its slot is reclaimed and key 3 is admitted;
        // key 2 (still live) is NOT evicted.
        assert_eq!(m.admit(&3, 200, max, 150), ProofAdmissionResult::Admitted);
        assert!(!m.entries.contains_key(&1)); // reclaimed
        assert!(m.entries.contains_key(&2)); // unexpired, retained
        assert!(m.entries.contains_key(&3));
        // gc touched only the one expired record, leaving the live front intact.
        assert_eq!(m.entries.len(), 2);
    }

    #[test]
    fn a_fresh_key_is_admitted_once_and_never_twice() {
        let store = InMemoryProofReplayStore::single_verifier_instance(16);
        let k = key(1, 1);
        assert_eq!(
            admit_request_proof(&store, ProofDisposition::Authenticated, &k, far_future()),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            admit_request_proof(&store, ProofDisposition::Authenticated, &k, far_future()),
            ProofAdmissionResult::Replayed
        );
    }

    /// The same request_id under a different signer namespace is a different
    /// request, so the thumbprint must be part of the key.
    #[test]
    fn the_signer_namespace_is_part_of_the_key() {
        let store = InMemoryProofReplayStore::single_verifier_instance(16);
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(1, 7),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(2, 7),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
    }

    /// Partitions are separate: the same key in the other partition is a
    /// distinct record, and a saturated unattributed partition can neither
    /// delay nor deny authenticated admission.
    #[test]
    fn partitions_are_independent_and_saturation_does_not_cross() {
        let store = InMemoryProofReplayStore::single_verifier_instance(2);
        let k = key(1, 1);
        assert_eq!(
            admit_request_proof(&store, ProofDisposition::Unattributed, &k, far_future()),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            admit_request_proof(&store, ProofDisposition::Authenticated, &k, far_future()),
            ProofAdmissionResult::Admitted
        );

        // Saturate the unattributed partition.
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Unattributed,
                &key(1, 2),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Unattributed,
                &key(1, 3),
                far_future()
            ),
            ProofAdmissionResult::Failed
        );

        // The authenticated partition is unaffected.
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(1, 3),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
    }

    /// Capacity pressure fails closed. It never evicts an unexpired accepted
    /// record to make room, so an admitted proof can never be re-admitted.
    #[test]
    fn capacity_fails_closed_without_evicting_unexpired_records() {
        let store = InMemoryProofReplayStore::single_verifier_instance(1);
        let admitted = key(1, 1);
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &admitted,
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(1, 2),
                far_future()
            ),
            ProofAdmissionResult::Failed
        );
        // The first record survived the pressure.
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &admitted,
                far_future()
            ),
            ProofAdmissionResult::Replayed
        );
    }

    /// Records are collectable once their signed expiry has passed; because
    /// verification enforces the identical deadline, nothing admissible is
    /// ever collected.
    #[test]
    fn expired_records_are_collected() {
        let store = InMemoryProofReplayStore::single_verifier_instance(1);
        let stale = key(1, 1);
        // Expiry already in the past.
        assert_eq!(
            store.check_and_insert(ProofDisposition::Authenticated, &stale, 1),
            ProofAdmissionResult::Admitted
        );
        // A new key can be admitted: the stale record is collected, not evicted.
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(1, 2),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
    }

    #[test]
    fn a_one_shot_credential_is_consumed_exactly_once() {
        let store = InMemoryProofReplayStore::single_verifier_instance(16);
        let id = OneShotCredentialId {
            issuer: "https://issuer.example".into(),
            value: b"one-shot-1".to_vec(),
        };
        assert_eq!(
            consume_one_shot_credential(&store, &id, far_future()),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            consume_one_shot_credential(&store, &id, far_future()),
            ProofAdmissionResult::Replayed
        );
    }

    /// A consumed one-shot credential ID is retained until its expiry, so a
    /// second presentation before expiry is a replay even if a fresh proof
    /// carries it. Retention is keyed by the `expires_at` the caller supplies
    /// (dispatch supplies the CREDENTIAL lifetime, not the proof's).
    #[test]
    fn a_consumed_one_shot_credential_is_retained_until_expiry() {
        let store = InMemoryProofReplayStore::single_verifier_instance(16);
        let id = OneShotCredentialId {
            issuer: "https://issuer.example".into(),
            value: b"txn-1".to_vec(),
        };
        // Consumed with a far-future (credential-lifetime) expiry.
        assert_eq!(
            consume_one_shot_credential(&store, &id, far_future()),
            ProofAdmissionResult::Admitted
        );
        // A second presentation — even under a different, later proof — is a
        // replay while the credential is still valid.
        assert_eq!(
            consume_one_shot_credential(&store, &id, far_future()),
            ProofAdmissionResult::Replayed
        );
    }

    /// Credential IDs are issuer-scoped: two issuers producing the same value
    /// never consume each other's credential.
    #[test]
    fn one_shot_credential_ids_are_issuer_scoped() {
        let store = InMemoryProofReplayStore::single_verifier_instance(16);
        let a = OneShotCredentialId {
            issuer: "https://a.example".into(),
            value: b"same".to_vec(),
        };
        let b = OneShotCredentialId {
            issuer: "https://b.example".into(),
            value: b"same".to_vec(),
        };
        assert_eq!(
            consume_one_shot_credential(&store, &a, far_future()),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            consume_one_shot_credential(&store, &b, far_future()),
            ProofAdmissionResult::Admitted
        );
    }

    /// One-shot consumption does not create a second proof replay entry: it is
    /// the single admission action for that credential.
    #[test]
    fn one_shot_consumption_is_a_separate_namespace_from_proof_replay() {
        let store = InMemoryProofReplayStore::single_verifier_instance(16);
        let id = OneShotCredentialId {
            issuer: "https://issuer.example".into(),
            value: vec![1u8; 16],
        };
        assert_eq!(
            consume_one_shot_credential(&store, &id, far_future()),
            ProofAdmissionResult::Admitted
        );
        // A proof whose request_id happens to equal the credential value is
        // still admissible: the two namespaces do not alias.
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(0, 1),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
    }

    // -- domain guarantees --------------------------------------------------

    /// A namespace-affine deployment that routes a request to a node holding
    /// no history for that namespace MUST deny, not admit.
    struct AffineStore {
        owned: [u8; 32],
        inner: InMemoryProofReplayStore,
    }

    impl ProofReplayStore for AffineStore {
        fn domain_guarantee(&self) -> ReplayDomainGuarantee {
            ReplayDomainGuarantee::NamespaceAffineRouting
        }
        fn owns_namespace(&self, signer_thumbprint: &[u8; 32]) -> bool {
            *signer_thumbprint == self.owned
        }
        fn check_and_insert(
            &self,
            partition: ProofDisposition,
            key: &ProofReplayKey,
            expires_at: u64,
        ) -> ProofAdmissionResult {
            self.inner.check_and_insert(partition, key, expires_at)
        }
        fn consume_one_shot_credential(
            &self,
            id: &OneShotCredentialId,
            expires_at: u64,
        ) -> ProofAdmissionResult {
            self.inner.consume_one_shot_credential(id, expires_at)
        }
    }

    #[test]
    fn affine_routing_denies_a_namespace_this_node_does_not_own() {
        let store = AffineStore {
            owned: [1u8; 32],
            inner: InMemoryProofReplayStore::single_verifier_instance(16),
        };
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(1, 1),
                far_future()
            ),
            ProofAdmissionResult::Admitted,
            "the owned namespace is admitted"
        );
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(2, 1),
                far_future()
            ),
            ProofAdmissionResult::Failed,
            "failover to a node without this namespace's history must deny"
        );
    }

    /// Key-affine routing cannot make one-shot consumption domain-wide, so an
    /// affine deployment fails closed rather than consuming locally.
    #[test]
    fn affine_routing_cannot_consume_a_one_shot_credential() {
        let store = AffineStore {
            owned: [1u8; 32],
            inner: InMemoryProofReplayStore::single_verifier_instance(16),
        };
        let id = OneShotCredentialId {
            issuer: "https://issuer.example".into(),
            value: b"one-shot".to_vec(),
        };
        assert_eq!(
            consume_one_shot_credential(&store, &id, far_future()),
            ProofAdmissionResult::Failed
        );
    }

    /// A store whose backend is unavailable reports Failed, and Failed is
    /// always a denial — it is never treated as "not seen before".
    #[test]
    fn an_unavailable_backend_fails_closed() {
        struct DeadBackend;
        impl ProofReplayStore for DeadBackend {
            fn domain_guarantee(&self) -> ReplayDomainGuarantee {
                ReplayDomainGuarantee::LinearizableSharedStore
            }
            fn check_and_insert(
                &self,
                _p: ProofDisposition,
                _k: &ProofReplayKey,
                _e: u64,
            ) -> ProofAdmissionResult {
                ProofAdmissionResult::Failed
            }
            fn consume_one_shot_credential(
                &self,
                _id: &OneShotCredentialId,
                _e: u64,
            ) -> ProofAdmissionResult {
                ProofAdmissionResult::Failed
            }
        }
        let store = DeadBackend;
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(1, 1),
                far_future()
            ),
            ProofAdmissionResult::Failed
        );
        assert_eq!(
            consume_one_shot_credential(
                &store,
                &OneShotCredentialId {
                    issuer: "i".into(),
                    value: vec![1],
                },
                far_future()
            ),
            ProofAdmissionResult::Failed
        );
    }

    // -- single-verifier lease ---------------------------------------------

    /// The lease is exclusive: a second acquisition on the same path — the
    /// "two replicas both set single-verifier-instance" case — fails, so the
    /// second verifier cannot install a per-process store under a domain-wide
    /// guarantee it does not hold.
    #[test]
    fn the_single_verifier_lease_is_exclusive() {
        let path = std::env::temp_dir().join(format!(
            "hyprstream-replay-lease-{}.lock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let held = SingleVerifierLease::acquire(&path).expect("first acquisition succeeds");
        assert_eq!(held.path(), path.as_path());

        // A second process (here, a second acquisition of the same path while
        // the first lock is still held) must fail — this is exactly the
        // misconfigured-second-replica case.
        let second = SingleVerifierLease::acquire(&path);
        assert!(
            second.is_err(),
            "a second verifier must not be able to hold the same lease"
        );

        // Once the first lease drops, the lock is released and the path is
        // acquirable again (a genuine single-verifier restart).
        drop(held);
        assert!(
            SingleVerifierLease::acquire(&path).is_ok(),
            "the lease is reacquirable after release"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// A linearizable shared store owns every namespace by construction.
    #[test]
    fn a_linearizable_store_owns_every_namespace() {
        struct Shared(InMemoryProofReplayStore);
        impl ProofReplayStore for Shared {
            fn domain_guarantee(&self) -> ReplayDomainGuarantee {
                ReplayDomainGuarantee::LinearizableSharedStore
            }
            fn check_and_insert(
                &self,
                p: ProofDisposition,
                k: &ProofReplayKey,
                e: u64,
            ) -> ProofAdmissionResult {
                self.0.check_and_insert(p, k, e)
            }
            fn consume_one_shot_credential(
                &self,
                id: &OneShotCredentialId,
                e: u64,
            ) -> ProofAdmissionResult {
                self.0.consume_one_shot_credential(id, e)
            }
        }
        let store = Shared(InMemoryProofReplayStore::single_verifier_instance(16));
        assert!(store.owns_namespace(&[9u8; 32]));
        assert_eq!(
            admit_request_proof(
                &store,
                ProofDisposition::Authenticated,
                &key(9, 1),
                far_future()
            ),
            ProofAdmissionResult::Admitted
        );
    }
}
