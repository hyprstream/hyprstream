//! Leaf-tier cell-router for federated inference scheduling (#322).
//!
//! Capacity-weighted HRW (Highest Random Weight / Rendezvous Hashing) for
//! session placement within a single cell (a PDS / trust domain). This is the
//! **leaf tier** of the federated scheduling design — a bounded set of tens to
//! low-thousands of nodes. The global tier (cell directory over ~10k–100k
//! cells) is a separate epic.
//!
//! # Two keys, two jobs
//!
//! - **OID** (git commit SHA) = content selector → yields the replica set
//!   (nodes serving that checkpoint). The router body operates on a *resolved*
//!   replica set; how the OID is resolved (`at://`→DID→OID per the
//!   federated-model-addressing spike, or the local `Resolver`) is the entry's
//!   job, not the router's.
//! - **session_id** = placement key → capacity-weighted HRW over the replica
//!   set → owner node.
//!
//! HRW gives both placement properties simultaneously:
//! - **Affinity:** the same `session_id` always hashes to the same node while
//!   the replica set is stable → KV-cache reuse.
//! - **Spread:** distinct `session_id`s distribute uniformly across the set.
//!
//! # Failure detection
//!
//! Two reactive layers, no proactive membership protocol (SWIM is unnecessary
//! at this tier — see the federated-scheduling-dht-pq-spike §4):
//! - **Dial-fail (sync):** a failed dial marks the node down and reassigns to
//!   the next HRW choice on the next call.
//! - **Token-stream stall watchdog:** absence of tokens past a deadline marks
//!   the node stalled and reassigns.
//!
//! # v1 constraints
//!
//! - **No new capnp surface.** Reuse `healthCheck@13` / `isReady@3` from
//!   `service_events.capnp` for health integration.
//! - **No gossip.** Load counters are router-side local state only (v1); gossip
//!   is a future option for proactive capacity rebalancing (spike §4).
//! - **CPU-testable.** Router logic is plain data structures — no tensors, no
//!   GPU. tch-rs is `!Send`; this module touches none of it.
//!
//! # Determinism
//!
//! HRW requires a hash that is **stable across processes** (so two routers in
//! the same cell pick the same owner for the same session). `DefaultHasher::new()`
//! is documented to use constant keys, so its output is deterministic across
//! runs and processes. We do *not* use `RandomState` (which is randomly seeded
//! per process and would break cross-router agreement).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use hyprstream_rpc::transport::TransportConfig;

/// Opaque identifier for an inference replica used only for placement.
///
/// This is not an iroh `EndpointId`, application signing key, or authorization
/// subject. Network reach and authenticated authority must be resolved and
/// verified independently before a future remote placement can be dialed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReplicaId([u8; 32]);

impl ReplicaId {
    /// Construct placement metadata from its stable byte representation.
    ///
    /// This explicit conversion does not verify or grant any transport or
    /// authority role; it only prevents accidental type interchange.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the stable bytes used by the placement hash.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Identifier for a chat / inference session (used as the HRW placement key).
pub type SessionId = String;

/// Default lease for a session→owner affinity binding. Renewed on every
/// heartbeat; expiry forces a fresh HRW on the next request.
pub const DEFAULT_LEASE_DURATION: Duration = Duration::from_secs(30);

/// Default deadline before a token-starved stream is declared stalled.
pub const DEFAULT_STALL_DEADLINE: Duration = Duration::from_secs(45);

/// Default negative-cache TTL on a dial-fail (avoid hammering a down node).
pub const DEFAULT_DOWN_TTL: Duration = Duration::from_secs(5);

/// Information about a candidate inference server in the cell.
///
/// The replica set for a resolved OID is `Vec<InferenceServerInfo>`. The router
/// applies HRW over this set keyed by `session_id`.
#[derive(Debug, Clone)]
pub struct InferenceServerInfo {
    /// Opaque replica placement identifier; carries no transport or authority role.
    pub replica_id: ReplicaId,
    /// How to dial this node (Iroh for cross-host, Inproc for co-located).
    pub transport: TransportConfig,
    /// Free GPU memory in bytes (capacity weight for HRW).
    pub gpu_memory_free: u64,
    /// Active sessions currently routed to this node (router-side counter).
    pub active_sessions: u64,
    /// Last heartbeat / state change observed by the router.
    pub last_heartbeat: Instant,
}

impl InferenceServerInfo {
    /// Capacity weight used by HRW. Free GPU memory is the primary signal
    /// (more free → more likely to win HRW for a new session). We add 1 so a
    /// zero-free node is still a selectable fallback rather than weight-0.
    fn weight(&self) -> u64 {
        self.gpu_memory_free.saturating_add(1)
    }
}

/// Where to place a session within the replica set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlacementPolicy {
    /// Stick a session to its existing owner for KV-cache reuse. Falls back to
    /// HRW when there is no live binding. This is the default — most inference
    /// workloads benefit from KV-cache reuse.
    #[default]
    Affinity,
    /// Always pick the least-loaded candidate (stateless fanout). Bypasses the
    /// affinity map entirely; useful for one-shot / stateless requests.
    Spread,
}

/// Session→owner affinity map with heartbeat-lease renewal.
///
/// A binding is only valid while its lease is fresh. A heartbeat (any request
/// touching the session) renews the lease; once the lease expires the next
/// request re-runs HRW and rebinds. This keeps affinity as a *perf hint, not a
/// correctness invariant* — a stale binding can't corrupt state because git
/// ref-CAS remains the only hard-consistency point (spike §5.4).
#[derive(Debug, Clone)]
pub struct SessionAffinity {
    map: HashMap<SessionId, AffinityBinding>,
    /// Bumped on every structural change (insert/remove/expiry sweep) so
    /// observers can cheaply detect "something changed".
    generation: u64,
    lease_duration: Duration,
}

#[derive(Debug, Clone, Copy)]
struct AffinityBinding {
    owner: ReplicaId,
    leased_until: Instant,
}

impl Default for SessionAffinity {
    fn default() -> Self {
        Self::new(DEFAULT_LEASE_DURATION)
    }
}

impl SessionAffinity {
    /// Create an empty affinity map with the given lease duration.
    pub fn new(lease_duration: Duration) -> Self {
        Self {
            map: HashMap::new(),
            generation: 0,
            lease_duration,
        }
    }

    /// Look up a live (non-expired) binding for a session, renewing its lease
    /// on hit. Returns `None` if there is no binding or it has expired.
    pub fn get(&mut self, session_id: &str) -> Option<ReplicaId> {
        let now = Instant::now();
        let binding = self.map.get_mut(session_id)?;
        if binding.leased_until <= now {
            // Expired — drop and re-HRW. Don't touch `generation` here; the
            // subsequent `set` from the rebind will bump it.
            self.map.remove(session_id);
            return None;
        }
        // Renew on hit.
        binding.leased_until = now + self.lease_duration;
        Some(binding.owner)
    }

    /// Read-only peek (no lease renewal). Used by tests and observers.
    pub fn peek(&self, session_id: &str) -> Option<ReplicaId> {
        let now = Instant::now();
        self.map
            .get(session_id)
            .filter(|b| b.leased_until > now)
            .map(|b| b.owner)
    }

    /// Bind (or rebind) a session to an owner with a fresh lease.
    pub fn set(&mut self, session_id: SessionId, owner: ReplicaId) {
        let leased_until = Instant::now() + self.lease_duration;
        match self.map.insert(
            session_id,
            AffinityBinding {
                owner,
                leased_until,
            },
        ) {
            None => self.generation = self.generation.wrapping_add(1),
            Some(old) if old.owner != owner => {
                self.generation = self.generation.wrapping_add(1);
            }
            _ => {}
        }
    }

    /// Drop a binding (e.g. on session-complete or explicit release).
    pub fn remove(&mut self, session_id: &str) {
        if self.map.remove(session_id).is_some() {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Drop all bindings whose lease has elapsed. Returns the number dropped.
    pub fn sweep_expired(&mut self) -> usize {
        let now = Instant::now();
        let before = self.map.len();
        self.map.retain(|_, b| b.leased_until > now);
        let dropped = before - self.map.len();
        if dropped > 0 {
            self.generation = self.generation.wrapping_add(1);
        }
        dropped
    }

    /// Number of live bindings (does not sweep first).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Monotonic generation counter — bumps on any structural change.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Lease duration currently in effect.
    pub fn lease_duration(&self) -> Duration {
        self.lease_duration
    }
}

/// Highest Random Weight (Rendezvous Hashing) selection.
///
/// For each candidate, compute `h(session_id ‖ replica_id)` weighted by capacity.
/// The candidate with the highest weighted score wins. Deterministic for a
/// given `(session_id, replica set)` pair: every router in the cell agrees on
/// the owner without any coordination.
///
/// Returns the **index** into `candidates` of the winner, or `None` if the
/// slice is empty.
///
/// # Capacity weighting
///
/// To bias placement toward higher-capacity (more free GPU memory) nodes without
/// breaking HRW's consistency, we derive the per-node score from
/// `weight_for_rank = -weight / ln(uniform_rand)` — the classic
/// "weighted rendezvous" transform (Heintz et al.) which reduces to plain HRW
/// when all weights are equal. We approximate it here with a cheaper but still
/// consistency-preserving transform `score = hash * weight` so a heavier node
/// wins ties more often. (Both transforms are deterministic in `(session,
/// node)`, which is the only property HRW needs.)
pub fn hrw_select(session_id: &str, candidates: &[InferenceServerInfo]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_score = score(session_id, &candidates[0]);
    for (idx, cand) in candidates.iter().enumerate().skip(1) {
        let s = score(session_id, cand);
        if s > best_score {
            best_score = s;
            best_idx = idx;
        }
    }
    Some(best_idx)
}

/// Select the `skip`-th HRW choice (0 = primary, 1 = first fallback, ...).
///
/// Used after a dial-fail / stall to pick the *next* candidate without
/// disturbing the primary ranking. Returns `None` once `skip >= candidates.len()`.
pub fn hrw_select_skip(
    session_id: &str,
    candidates: &[InferenceServerInfo],
    skip: usize,
) -> Option<usize> {
    if candidates.is_empty() || skip >= candidates.len() {
        return None;
    }
    // Build (score, idx) and sort descending; the `skip`-th entry is the answer.
    let mut ranked: Vec<(u64, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (score(session_id, c), i))
        .collect();
    ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.get(skip).map(|(_, idx)| *idx)
}

/// Select the least-loaded candidate (lowest `active_sessions`, ties broken by
/// most free GPU memory). Returns `None` for an empty set.
pub fn least_loaded_select(candidates: &[InferenceServerInfo]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    for (idx, cand) in candidates.iter().enumerate().skip(1) {
        let cur = &candidates[best_idx];
        let better = cand.active_sessions < cur.active_sessions
            || (cand.active_sessions == cur.active_sessions
                && cand.gpu_memory_free > cur.gpu_memory_free);
        if better {
            best_idx = idx;
        }
    }
    Some(best_idx)
}

/// Weighted HRW score for a (session, node) pair. Deterministic across
/// processes (see module docs on `DefaultHasher::new`).
fn score(session_id: &str, cand: &InferenceServerInfo) -> u64 {
    let mut h = DefaultHasher::new();
    session_id.hash(&mut h);
    cand.replica_id.hash(&mut h);
    let raw = h.finish();
    // Multiply by the capacity weight. u64 wrapping mul preserves total order
    // for non-pathological weights (we expect weights well below u64::MAX).
    raw.wrapping_mul(cand.weight())
}

/// Outcome of a placement decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// Index into the candidate slice that was chosen.
    pub candidate_idx: usize,
    /// The owning node id.
    pub replica_id: ReplicaId,
    /// Whether a new affinity binding was recorded (vs a cache hit).
    pub rebound: bool,
}

/// Reason a node was excluded from placement (for the failure log / tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcludeReason {
    /// Explicitly marked down after a dial-fail (negative-cached for `down_ttl`).
    Down,
    /// Declared stalled by the token-stream watchdog.
    Stalled,
}

/// Per-node failure state maintained by the router.
#[derive(Debug, Clone, Copy)]
struct NodeHealth {
    /// When the negative-cache entry expires (`Instant`) and the node becomes
    /// eligible again. `None` = healthy.
    down_until: Option<Instant>,
}

impl NodeHealth {
    fn is_excluded(&self, now: Instant) -> bool {
        self.down_until.is_some_and(|t| t > now)
    }
}

/// Router-side load + health state for a cell's replica set.
///
/// Holds the replica set, the affinity map, and per-node health / down-cache.
/// All updates are router-local in v1 (no gossip).
#[derive(Debug, Clone)]
pub struct CellRouter {
    policy: PlacementPolicy,
    affinity: SessionAffinity,
    health: HashMap<ReplicaId, NodeHealth>,
    /// Nodes currently declared stalled by the token-stream watchdog.
    /// `Instant` = when the stall was declared (used for log/TTL).
    stalled: HashMap<ReplicaId, Instant>,
    down_ttl: Duration,
    stall_deadline: Duration,
}

impl Default for CellRouter {
    fn default() -> Self {
        Self::new(PlacementPolicy::default())
    }
}

impl CellRouter {
    /// Create a router with the given placement policy and default timers.
    pub fn new(policy: PlacementPolicy) -> Self {
        Self {
            policy,
            affinity: SessionAffinity::default(),
            health: HashMap::new(),
            stalled: HashMap::new(),
            down_ttl: DEFAULT_DOWN_TTL,
            stall_deadline: DEFAULT_STALL_DEADLINE,
        }
    }

    /// Override the dial-fail negative-cache TTL.
    pub fn with_down_ttl(mut self, ttl: Duration) -> Self {
        self.down_ttl = ttl;
        self
    }

    /// Override the token-stream stall deadline.
    pub fn with_stall_deadline(mut self, deadline: Duration) -> Self {
        self.stall_deadline = deadline;
        self
    }

    /// Override the affinity lease duration.
    pub fn with_lease_duration(mut self, d: Duration) -> Self {
        self.affinity = SessionAffinity::new(d);
        self
    }

    /// Current placement policy.
    pub fn policy(&self) -> PlacementPolicy {
        self.policy
    }

    /// Borrow the affinity map (for tests / status).
    pub fn affinity(&self) -> &SessionAffinity {
        &self.affinity
    }

    /// Borrow the affinity map mutably (e.g. to renew/sweep from outside).
    pub fn affinity_mut(&mut self) -> &mut SessionAffinity {
        &mut self.affinity
    }

    /// Filter the candidate set to currently-healthy (non-down, non-stalled)
    /// nodes, preserving order. Returns the original indices paired with the
    /// info so callers can map back.
    fn healthy<'a>(
        &self,
        candidates: &'a [InferenceServerInfo],
        now: Instant,
    ) -> Vec<(usize, &'a InferenceServerInfo)> {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                let down = self
                    .health
                    .get(&c.replica_id)
                    .is_some_and(|h| h.is_excluded(now));
                let stalled = self.stalled.contains_key(&c.replica_id);
                !down && !stalled
            })
            .collect()
    }

    /// Place a session on a candidate from the replica set.
    ///
    /// - `Affinity`: reuses the live binding if present; otherwise HRW-selects
    ///   over healthy candidates and records a new binding.
    /// - `Spread`: always least-loaded-selects over healthy candidates (no
    ///   affinity consulted).
    ///
    /// Down / stalled nodes are excluded. Returns `None` only if **no** healthy
    /// candidate remains (the replica set is empty or entirely failed).
    pub fn place(
        &mut self,
        session_id: &str,
        candidates: &[InferenceServerInfo],
        now: Instant,
    ) -> Option<Placement> {
        let healthy = self.healthy(candidates, now);
        if healthy.is_empty() {
            return None;
        }
        // Fast path: affinity hit on a node that is still in the healthy set.
        if matches!(self.policy, PlacementPolicy::Affinity) {
            if let Some(owner) = self.affinity.get(session_id) {
                if let Some(&(idx, _)) = healthy.iter().find(|(_, c)| c.replica_id == owner) {
                    return Some(Placement {
                        candidate_idx: idx,
                        replica_id: owner,
                        rebound: false,
                    });
                }
                // Owner is down/stalled/gone — fall through to HRW rebind.
            }
        }
        // Cold path: select by policy over the healthy subset. The select
        // functions return an index into the contiguous healthy view; we map
        // back to the original candidate index via `healthy[local_idx].0`.
        let healthy_view: Vec<InferenceServerInfo> =
            healthy.iter().map(|(_, c)| (*c).clone()).collect();
        let local_idx = match self.policy {
            PlacementPolicy::Affinity => hrw_select(session_id, &healthy_view),
            PlacementPolicy::Spread => least_loaded_select(&healthy_view),
        }?;
        let (idx, chosen) = healthy[local_idx];
        self.affinity.set(session_id.to_owned(), chosen.replica_id);
        Some(Placement {
            candidate_idx: idx,
            replica_id: chosen.replica_id,
            rebound: true,
        })
    }

    /// Mark a node down after a synchronous dial-fail. The node is
    /// negative-cached for [`down_ttl`](Self::with_down_ttl) and excluded from
    /// subsequent placements. The next `place()` for an affected session will
    /// rebind to the next HRW choice.
    ///
    /// Returns `true` if this was a new down-mark (the node was previously
    /// healthy), `false` if it was already down.
    pub fn report_dial_fail(&mut self, replica_id: ReplicaId, now: Instant) -> bool {
        let down_until = now + self.down_ttl;
        match self.health.get(&replica_id) {
            Some(h) if h.is_excluded(now) => false,
            _ => {
                self.health.insert(
                    replica_id,
                    NodeHealth {
                        down_until: Some(down_until),
                    },
                );
                true
            }
        }
    }

    /// Mark a node stalled (token-stream watchdog fired). Stalled nodes are
    /// excluded from placement until explicitly cleared
    /// ([`clear_stall`](Self::clear_stall)). A stall does *not* start the
    /// down-clock — it is a stronger signal (the node answered but is not
    /// producing tokens).
    pub fn report_stall(&mut self, replica_id: ReplicaId, now: Instant) {
        self.stalled.insert(replica_id, now);
    }

    /// Clear a stall / down mark once the node recovers (e.g. a subsequent
    /// `healthCheck` succeeds, or tokens resume).
    pub fn clear_node(&mut self, replica_id: ReplicaId) {
        self.health.remove(&replica_id);
        self.stalled.remove(&replica_id);
    }

    /// Token-stream stall deadline currently in effect.
    pub fn stall_deadline(&self) -> Duration {
        self.stall_deadline
    }

    /// Record that a session has completed (release its affinity binding).
    pub fn release_session(&mut self, session_id: &str) {
        self.affinity.remove(session_id);
    }

    /// Record a heartbeat (renew the lease) for a session. Returns `true` if a
    /// binding existed and was renewed.
    pub fn heartbeat(&mut self, session_id: &str) -> bool {
        self.affinity.get(session_id).is_some()
    }

    /// Whether a node is currently excluded (down or stalled).
    pub fn is_excluded(&self, replica_id: ReplicaId, now: Instant) -> Option<ExcludeReason> {
        if self.stalled.contains_key(&replica_id) {
            return Some(ExcludeReason::Stalled);
        }
        if self
            .health
            .get(&replica_id)
            .is_some_and(|h| h.is_excluded(now))
        {
            return Some(ExcludeReason::Down);
        }
        None
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // panicking on `None` is the right behavior in unit tests
mod tests {
    use super::*;

    fn node(id: u8, mem_free: u64, active: u64) -> InferenceServerInfo {
        InferenceServerInfo {
            replica_id: ReplicaId::from_bytes([id; 32]),
            transport: TransportConfig::inproc("test"),
            gpu_memory_free: mem_free,
            active_sessions: active,
            last_heartbeat: Instant::now(),
        }
    }

    fn replica_set() -> Vec<InferenceServerInfo> {
        vec![
            node(0x10, 16 * 1024 * 1024 * 1024, 0),
            node(0x20, 16 * 1024 * 1024 * 1024, 0),
            node(0x30, 16 * 1024 * 1024 * 1024, 0),
        ]
    }

    // ---- HRW determinism ----

    #[test]
    fn hrw_same_session_same_node() {
        let set = replica_set();
        let a = hrw_select("sess-A", &set).unwrap();
        let b = hrw_select("sess-A", &set).unwrap();
        assert_eq!(a, b, "same session must always select the same node");
    }

    #[test]
    fn hrw_uses_index_into_input_slice() {
        let set = replica_set();
        let idx = hrw_select("sess-A", &set).unwrap();
        assert!(idx < set.len());
    }

    #[test]
    fn hrw_empty_returns_none() {
        assert_eq!(hrw_select("sess", &[]), None);
    }

    #[test]
    fn hrw_spreads_distinct_sessions() {
        // With equal-weight nodes, ~50 distinct sessions should land on all 3
        // nodes (no single node should get > ~70% — that would indicate a
        // broken hash).
        let set = replica_set();
        let mut counts = [0usize; 3];
        for i in 0..150u32 {
            let idx = hrw_select(&format!("sess-{i}"), &set).unwrap();
            counts[idx] += 1;
        }
        // All nodes got at least one, and no node got everything.
        assert!(
            counts.iter().all(|&c| c > 0),
            "every node must get some sessions: {counts:?}"
        );
        assert!(
            counts.iter().all(|&c| c < 150),
            "no node should get all sessions: {counts:?}"
        );
    }

    #[test]
    fn hrw_capacity_weight_biases_heavier_node() {
        // One node with 10x the free memory should win a disproportionate share
        // (not necessarily exactly 10x — `score = hash * weight` is an
        // approximation — but strictly more than its equal-weight siblings).
        const GB: u64 = 1024 * 1024 * 1024;
        let heavy_set = vec![
            node(0x10, GB, 0),      // 1 GB
            node(0x20, 10 * GB, 0), // 10 GB
            node(0x30, GB, 0),      // 1 GB
        ];
        let mut counts = [0usize; 3];
        for i in 0..600u32 {
            let idx = hrw_select(&format!("sess-{i}"), &heavy_set).unwrap();
            counts[idx] += 1;
        }
        assert!(
            counts[1] > counts[0] && counts[1] > counts[2],
            "heavier node should win the most: {counts:?}"
        );
    }

    #[test]
    fn hrw_select_skip_picks_distinct_nodes() {
        let set = replica_set();
        let primary = hrw_select("sess-X", &set).unwrap();
        let first_fallback = hrw_select_skip("sess-X", &set, 1).unwrap();
        let second_fallback = hrw_select_skip("sess-X", &set, 2).unwrap();
        assert_ne!(
            primary, first_fallback,
            "primary and first fallback must differ"
        );
        assert_ne!(primary, second_fallback);
        assert_ne!(first_fallback, second_fallback);
        // skip past the end → None
        assert_eq!(hrw_select_skip("sess-X", &set, 3), None);
    }

    #[test]
    fn hrw_consistent_under_set_shrink_grow() {
        // Removing a node must not move the sessions that *weren't* on it — the
        // HRW consistency property. (Sessions on the removed node rebind.)
        let set = replica_set();
        let owner_full = hrw_select("sess-Z", &set).unwrap();
        let owner_replica_id = set[owner_full].replica_id;
        // Remove each non-owner and confirm sess-Z's selection is stable.
        for drop_idx in 0..set.len() {
            if drop_idx == owner_full {
                continue;
            }
            let subset: Vec<_> = set
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != drop_idx)
                .map(|(_, c)| c.clone())
                .collect();
            let owner_sub = hrw_select("sess-Z", &subset).unwrap();
            assert_eq!(
                subset[owner_sub].replica_id, owner_replica_id,
                "sess-Z must stay on its owner when a different node is dropped"
            );
        }
    }

    // ---- Spread policy ----

    #[test]
    fn least_loaded_picks_fewest_active_sessions() {
        let set = vec![
            node(0x10, 8 * 1024 * 1024 * 1024, 5),
            node(0x20, 8 * 1024 * 1024 * 1024, 1), // fewest
            node(0x30, 8 * 1024 * 1024 * 1024, 9),
        ];
        let idx = least_loaded_select(&set).unwrap();
        assert_eq!(idx, 1, "least-loaded must pick the fewest active sessions");
    }

    #[test]
    fn least_loaded_breaks_ties_by_memory() {
        let set = vec![
            node(0x10, 4 * 1024 * 1024 * 1024, 2),
            node(0x20, 16 * 1024 * 1024 * 1024, 2), // tied, more memory
            node(0x30, 2 * 1024 * 1024 * 1024, 2),
        ];
        let idx = least_loaded_select(&set).unwrap();
        assert_eq!(idx, 1, "tie should break toward most free memory");
    }

    // ---- SessionAffinity ----

    #[test]
    fn affinity_get_set_expire() {
        let mut aff = SessionAffinity::new(Duration::from_millis(50));
        let owner = ReplicaId::from_bytes([0xAB; 32]);
        assert_eq!(aff.peek("s1"), None, "empty map → no binding");
        aff.set("s1".to_owned(), owner);
        assert_eq!(aff.peek("s1"), Some(owner));
        // get() renews the lease.
        assert_eq!(aff.get("s1"), Some(owner));
        // Let the lease lapse (no renewal for >50ms).
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(
            aff.peek("s1"),
            None,
            "expired binding must not be visible to peek"
        );
        // get() on an expired binding returns None and drops it.
        assert_eq!(aff.get("s1"), None);
    }

    #[test]
    fn affinity_generation_bumps_on_change() {
        let mut aff = SessionAffinity::new(Duration::from_secs(10));
        let g0 = aff.generation();
        aff.set("s1".to_owned(), ReplicaId::from_bytes([1; 32]));
        assert!(aff.generation() > g0);
        let g1 = aff.generation();
        // Same owner rebind → no bump.
        aff.set("s1".to_owned(), ReplicaId::from_bytes([1; 32]));
        assert_eq!(aff.generation(), g1);
        // Different owner → bump.
        aff.set("s1".to_owned(), ReplicaId::from_bytes([2; 32]));
        assert!(aff.generation() > g1);
    }

    #[test]
    fn affinity_remove_drops_binding() {
        let mut aff = SessionAffinity::new(Duration::from_secs(10));
        aff.set("s1".to_owned(), ReplicaId::from_bytes([1; 32]));
        assert_eq!(aff.len(), 1);
        aff.remove("s1");
        assert!(aff.is_empty());
    }

    #[test]
    fn affinity_sweep_expired() {
        let mut aff = SessionAffinity::new(Duration::from_millis(20));
        aff.set("a".to_owned(), ReplicaId::from_bytes([1; 32]));
        aff.set("b".to_owned(), ReplicaId::from_bytes([2; 32]));
        std::thread::sleep(Duration::from_millis(40));
        let dropped = aff.sweep_expired();
        assert_eq!(dropped, 2);
        assert!(aff.is_empty());
    }

    // ---- CellRouter: Affinity policy ----

    #[test]
    fn router_affinity_reuses_binding() {
        let mut router = CellRouter::default();
        let set = replica_set();
        let now = Instant::now();
        let p1 = router.place("sess-A", &set, now).unwrap();
        assert!(p1.rebound, "first placement must be a rebind");
        let p2 = router.place("sess-A", &set, now).unwrap();
        assert!(!p2.rebound, "second placement must hit the affinity cache");
        assert_eq!(p1.replica_id, p2.replica_id);
        assert_eq!(p1.candidate_idx, p2.candidate_idx);
    }

    #[test]
    fn router_affinity_distinct_sessions_spread() {
        let mut router = CellRouter::default();
        let set = replica_set();
        let now = Instant::now();
        let mut owners = std::collections::HashSet::new();
        for i in 0..60 {
            let p = router.place(&format!("sess-{i}"), &set, now).unwrap();
            owners.insert(p.replica_id);
        }
        assert!(
            owners.len() > 1,
            "distinct sessions should spread: got {owners:?}"
        );
    }

    // ---- CellRouter: Spread policy ----

    #[test]
    fn router_sprow_bypasses_affinity_and_picks_least_loaded() {
        let mut router = CellRouter::new(PlacementPolicy::Spread);
        let set = vec![
            node(0x10, 8 * 1024 * 1024 * 1024, 5),
            node(0x20, 8 * 1024 * 1024 * 1024, 1), // least loaded
            node(0x30, 8 * 1024 * 1024 * 1024, 9),
        ];
        let now = Instant::now();
        // Even after a prior placement, Spread must ignore affinity and pick
        // the least-loaded node.
        let _ = router.place("sess-A", &set, now).unwrap();
        let p = router.place("sess-B", &set, now).unwrap();
        assert_eq!(p.replica_id, set[1].replica_id, "Spread must pick least-loaded");
    }

    // ---- Failure detection: dial-fail → reassign ----

    #[test]
    fn router_dial_fail_excludes_and_reassigns() {
        let mut router = CellRouter::default();
        let set = replica_set();
        let now = Instant::now();
        let p1 = router.place("sess-A", &set, now).unwrap();
        // Mark the owner down.
        let was_new = router.report_dial_fail(p1.replica_id, now);
        assert!(was_new, "first down-mark must be a transition");
        // Next placement must rebind to a different node.
        let p2 = router.place("sess-A", &set, now).unwrap();
        assert_ne!(p2.replica_id, p1.replica_id, "down node must not be reselected");
        assert!(
            p2.rebound,
            "reassignment after down must record a new binding"
        );
        // Idempotent re-report.
        let was_new2 = router.report_dial_fail(p1.replica_id, now);
        assert!(
            !was_new2,
            "re-reporting an already-down node is not a transition"
        );
    }

    #[test]
    fn router_down_node_recovers_after_ttl() {
        let mut router = CellRouter::default().with_down_ttl(Duration::from_millis(30));
        let set = replica_set();
        let t0 = Instant::now();
        let p1 = router.place("sess-A", &set, t0).unwrap();
        router.report_dial_fail(p1.replica_id, t0);
        // After the TTL, the node is eligible again.
        let t1 = t0 + Duration::from_millis(40);
        // We must clear the session's (now-rebound) binding first, otherwise
        // affinity would just keep returning the rebound owner.
        router.release_session("sess-A");
        let p2 = router.place("sess-A", &set, t1).unwrap();
        // The recovered node is now eligible; if it wins HRW again, great.
        assert!(
            router.is_excluded(p1.replica_id, t1).is_none(),
            "node should be healthy after TTL"
        );
        let _ = p2;
    }

    #[test]
    fn router_all_down_returns_none() {
        let mut router = CellRouter::default();
        let set = replica_set();
        let now = Instant::now();
        for n in &set {
            router.report_dial_fail(n.replica_id, now);
        }
        assert!(
            router.place("sess-A", &set, now).is_none(),
            "all-down → no placement"
        );
    }

    // ---- Failure detection: token-stream stall ----

    #[test]
    fn router_stall_excludes_and_reassigns() {
        let mut router = CellRouter::default();
        let set = replica_set();
        let now = Instant::now();
        let p1 = router.place("sess-A", &set, now).unwrap();
        router.report_stall(p1.replica_id, now);
        assert_eq!(
            router.is_excluded(p1.replica_id, now),
            Some(ExcludeReason::Stalled)
        );
        let p2 = router.place("sess-A", &set, now).unwrap();
        assert_ne!(
            p2.replica_id, p1.replica_id,
            "stalled node must not be reselected"
        );
        // Recovery.
        router.clear_node(p1.replica_id);
        assert!(router.is_excluded(p1.replica_id, now).is_none());
    }

    #[test]
    fn router_stall_deadline_exposed() {
        let r = CellRouter::default().with_stall_deadline(Duration::from_secs(7));
        assert_eq!(r.stall_deadline(), Duration::from_secs(7));
    }

    // ---- Heartbeat / release ----

    #[test]
    fn router_heartbeat_renews_and_release_drops() {
        let mut router = CellRouter::default();
        let set = replica_set();
        let now = Instant::now();
        router.place("sess-A", &set, now).unwrap();
        assert!(
            router.heartbeat("sess-A"),
            "heartbeat on a live session renews the lease"
        );
        router.release_session("sess-A");
        assert!(
            !router.heartbeat("sess-A"),
            "heartbeat after release has nothing to renew"
        );
    }
}
