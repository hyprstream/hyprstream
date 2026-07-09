//! Admission control for `SandboxPool::acquire` (#525, P2 of epic #523).
//!
//! Replaces the old "check `active.len()`, then either pop/create" shape
//! (immediate `PoolExhausted`, no queue, no identity, no resource dimension)
//! with a real decision engine:
//!
//! - **Fail-closed identity**: no verified [`Subject`] → reject before doing
//!   any bookkeeping. There is no "anonymous but allowed" path.
//! - **Reserve-then-place, never check-then-place** (the #489/#519 TOCTOU
//!   lesson): every admission decision — capacity, per-Subject quota,
//!   per-group quota, and cpu/memory/GPU capacity — is checked *and*
//!   committed atomically under one lock in [`try_admit_locked`]. There is no
//!   window between "checked OK" and "reserved" where a second caller can
//!   slip in and overshoot.
//! - **Bounded wait-queue with backpressure**: a request that cannot be
//!   admitted immediately because of *capacity* (never quota — see below)
//!   waits, bounded by [`AdmissionConfig::queue_capacity`] and
//!   [`AdmissionConfig::queue_timeout_secs`]. Over the bound, or past the
//!   timeout, it fails clearly rather than blocking forever or silently
//!   degrading.
//!
//! ## 🔒 NEEDS REVIEWER SIGN-OFF
//!
//! This module makes several judgment calls the issue explicitly flags as
//! needing sign-off rather than being final:
//!
//! 1. **Quota model shape**: a simple in-memory per-Subject/per-group counter
//!    with an operator-configured numeric ceiling ([`AdmissionConfig`]). This
//!    is *not* Casbin-policy-driven — there is no policy lookup, no dynamic
//!    per-tenant configuration source, just a static number in `PoolConfig`.
//!    Full Casbin-backed quota configuration (e.g. deriving `max_per_subject`
//!    from a Casbin role/policy rather than a flat config value) is a
//!    follow-up.
//! 2. **"Group" derivation (B′, ratified in #921 decision 5)**: the group is
//!    derived from the **verified [`Subject`], never trusted from the
//!    annotation**. The `hyprstream.io/group` annotation is only a *selector*
//!    among groups the Subject provably belongs to; a
//!    [`GroupSelectorValidator`] resolves it against the authoritative
//!    membership source (the bidirectional-consent placement group,
//!    `ai.hyprstream.placement.{group,groupItem}` / `PlacementIndex::is_member`).
//!    A selector naming a group the Subject is not a member of is **rejected**;
//!    an absent selector means **no group partition** (never a quota bypass).
//!    The default [`DenyUnknownGroupValidator`] is fail-closed: it denies every
//!    non-empty selector until a membership-backed validator is wired, so an
//!    unverified group can never partition capacity. See point 3 for what the
//!    counter itself means.
//! 3. **Wait-queue fairness**: the queue is a single bounded **FIFO** — new
//!    arrivals never barge past a non-empty queue, and each release hands
//!    capacity to the queue *head* (rather than waking all waiters to
//!    thundering-herd re-race). This eliminates the starvation the blocking
//!    review flagged (a queued waiter losing every re-race under sustained
//!    arrival). The tradeoff is **head-of-line blocking**: a large demand at
//!    the head can block a smaller demand behind it even when capacity for the
//!    smaller one exists. That is the conservative choice; a per-Subject/
//!    per-group fair sub-queue + weighted round-robin is the documented (not
//!    implemented) follow-up. Quota- and infeasibility-rejected requests never
//!    queue (they fail fast).
//! 4. **GPU/resource vocabulary**: cpu/memory/GPU demand is read from
//!    `config.linux.resources` (existing CRI-shaped fields) plus a new,
//!    separate annotation `hyprstream.io/gpu-request` (a plain integer count,
//!    not a GPU *class*) — independent of the existing boolean
//!    `hyprstream.io/gpu` passthrough annotation consumed by `oci_backend`/
//!    `nspawn`. The two are intentionally not unified in this change; see the
//!    module docs on `pool.rs` for the full rationale.
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify};
use tokio::time::Instant;

use hyprstream_discovery::scheduling::{self, Predicate, RejectionReason, ResourceRequest};
use hyprstream_vfs::Subject;

use crate::error::{Result, WorkerError};

use super::client::{KeyValue, PodSandboxConfig};

/// Annotation key: per-request GPU count demand (a plain non-negative
/// integer string, e.g. `"1"`). Independent of `oci_backend`'s existing
/// boolean `hyprstream.io/gpu` passthrough flag — see module docs, point 4.
pub const ANN_GPU_REQUEST: &str = "hyprstream.io/gpu-request";

/// Annotation key: the group *selector* for this request (B′, #921 decision
/// 5). This is **not** a trusted group claim — it only names which of the
/// groups the verified [`Subject`] *provably belongs to* this request should
/// be partitioned under. A [`GroupSelectorValidator`] resolves it against the
/// authoritative membership source; a selector the Subject is not a member of
/// is rejected. See module docs, point 2.
pub const ANN_GROUP: &str = "hyprstream.io/group";

fn annotation<'a>(annotations: &'a [KeyValue], key: &str) -> Option<&'a str> {
    annotations
        .iter()
        .find(|kv| kv.key == key)
        .map(|kv| kv.value.as_str())
}

/// Resource demand derived from a [`PodSandboxConfig`] for admission purposes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Demand {
    pub cpu_millis: u64,
    pub memory_bytes: u64,
    pub gpu: usize,
}

/// Derive the resource `Demand` this request represents.
///
/// CRI convention: `0` means "unspecified" for `cpu_period`/`cpu_quota`/
/// `memory_limit_in_bytes`, so an all-zero (`Default`) `LinuxContainerResources`
/// — what every pre-#525 caller passes — derives a zero `Demand`, and a zero
/// `Demand` never trips a capacity check. This is what keeps the solo/no-op
/// case regression-free (acceptance criterion: "no regression for the
/// existing solo case").
pub fn derive_demand(config: &PodSandboxConfig) -> Demand {
    let res = &config.linux.resources;
    let cpu_millis = if res.cpu_period > 0 && res.cpu_quota > 0 {
        ((res.cpu_quota as f64 / res.cpu_period as f64) * 1000.0).round() as u64
    } else {
        0
    };
    let memory_bytes = res.memory_limit_in_bytes.max(0) as u64;
    let gpu = annotation(&config.annotations, ANN_GPU_REQUEST)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    Demand {
        cpu_millis,
        memory_bytes,
        gpu,
    }
}

/// Derive the group *selector* for a request, if any (the raw
/// `hyprstream.io/group` annotation). This is an **untrusted** selector, not
/// an authoritative group: it must be validated against the verified Subject's
/// membership by a [`GroupSelectorValidator`] before it can partition capacity
/// (B′ — see module docs, point 2).
pub fn derive_group_selector(config: &PodSandboxConfig) -> Option<String> {
    annotation(&config.annotations, ANN_GROUP).map(str::to_owned)
}

/// Per-Subject / per-group admission quotas + wait-queue bounds + declared
/// local schedulable capacity (#525 P2).
///
/// 🔒 See the module-level "NEEDS REVIEWER SIGN-OFF" doc for the judgment
/// calls baked into this shape.
///
/// All limits default to **unconstrained** so that a `PoolConfig` with no
/// explicit `admission` section behaves exactly as before this change (no
/// regression for the existing solo/no-quota case): only an operator who
/// opts in by lowering these gets admission-control behavior beyond the
/// pre-existing `max_sandboxes` capacity check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdmissionConfig {
    /// Maximum concurrently-active sandboxes for a single Subject.
    /// `None` (default) = unconstrained.
    ///
    /// Serialized as *absent* when unlimited (rather than a sentinel like
    /// `usize::MAX`, which is not representable as a TOML i64 and would break
    /// `Config::save()`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_per_subject: Option<usize>,
    /// Maximum concurrently-active sandboxes for a single group (see
    /// [`derive_group_selector`]). `None` (default) = unconstrained.
    ///
    /// This counter is **node-local capacity partitioning** — a fairness /
    /// organization mechanism, *not* a trust or billing boundary. The
    /// authoritative quota is the ledger/capability path (#922/#925: verify
    /// presented grant → debit cell ledger → emit receipt), which this module
    /// deliberately does **not** implement. See module docs, point 2.
    ///
    /// Serialized as *absent* when unlimited (see [`Self::max_per_subject`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_per_group: Option<usize>,
    /// Bound on the number of callers allowed to wait in the admission queue
    /// at once. A request that would exceed this is rejected immediately
    /// (`QueueFull`) rather than queued.
    pub queue_capacity: usize,
    /// How long a queued request waits for capacity before giving up
    /// (`QueueTimeout`).
    pub queue_timeout_secs: u64,
    /// Total schedulable CPU (millicores) for this node's local capacity
    /// overlay. `None` (default) = the CPU dimension is not enforced.
    pub cpu_millis_total: Option<u64>,
    /// Total schedulable memory (bytes) for this node's local capacity
    /// overlay. `None` (default) = the memory dimension is not enforced.
    pub memory_bytes_total: Option<u64>,
    /// Total GPU count available for scheduling. Default `0`: a request that
    /// asks for a GPU (`Demand::gpu > 0`) is fail-closed rejected unless an
    /// operator explicitly declares GPU capacity here. Requests that don't
    /// ask for a GPU (`Demand::gpu == 0`, the default) are never affected by
    /// this value.
    pub gpu_total: usize,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            max_per_subject: None,
            max_per_group: None,
            queue_capacity: 64,
            queue_timeout_secs: 30,
            cpu_millis_total: None,
            memory_bytes_total: None,
            gpu_total: 0,
        }
    }
}

/// What a committed reservation consumed, so [`AdmissionTracker::release`] /
/// [`AdmissionTracker::rollback`] can give it back precisely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRecord {
    subject: String,
    group: Option<String>,
    demand: Demand,
}

/// A local resource-capacity candidate for the #628 scheduling substrate's
/// `filter → rank → select` pipeline (see module docs on GPU/resource
/// vocabulary). In the single-node/solo case this is the *only* candidate;
/// the extension point for cross-node placement is `rank_by` in
/// [`fit_over_local_candidate`] — feeding `queryCandidates` results in as additional
/// candidates here (with a real least-loaded/best-fit `rank_by`) is P2's
/// documented (not implemented) cross-node extension.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalCandidate {
    name: &'static str,
    cpu_millis_free: u64,
    memory_bytes_free: u64,
    gpu_free: usize,
}

/// Build the fit predicates for `demand` over a [`LocalCandidate`]. Only
/// dimensions actually requested (or actually enforced by config) produce a
/// predicate — an all-zero `Demand` (the default/no-op case) yields no
/// predicates, so it always "fits."
fn fit_predicates(demand: Demand) -> Vec<Predicate<LocalCandidate>> {
    let mut preds: Vec<Predicate<LocalCandidate>> = Vec::new();
    if demand.cpu_millis > 0 {
        preds.push(Box::new(move |c: &LocalCandidate| {
            let req = ResourceRequest::new("cpu", format!("{}m", demand.cpu_millis));
            if req.satisfied_by(&format!("{}m", c.cpu_millis_free)) {
                None
            } else {
                Some(RejectionReason(format!(
                    "insufficient cpu: need {}m, {}m free",
                    demand.cpu_millis, c.cpu_millis_free
                )))
            }
        }));
    }
    if demand.memory_bytes > 0 {
        preds.push(Box::new(move |c: &LocalCandidate| {
            let req = ResourceRequest::new("memory", demand.memory_bytes.to_string());
            if req.satisfied_by(&c.memory_bytes_free.to_string()) {
                None
            } else {
                Some(RejectionReason(format!(
                    "insufficient memory: need {} bytes, {} bytes free",
                    demand.memory_bytes, c.memory_bytes_free
                )))
            }
        }));
    }
    if demand.gpu > 0 {
        preds.push(Box::new(move |c: &LocalCandidate| {
            let req = ResourceRequest::new("gpu", demand.gpu.to_string());
            if req.satisfied_by(&c.gpu_free.to_string()) {
                None
            } else {
                Some(RejectionReason(format!(
                    "insufficient gpu: need {}, {} free",
                    demand.gpu, c.gpu_free
                )))
            }
        }));
    }
    preds
}

/// Run the #628 scheduling substrate's `filter → rank → select` over the
/// (today: single) local candidate for `demand`. Returns `Ok(())` if it fits,
/// `Err(reason)` with the joined rejection reasons otherwise.
fn fit_over_local_candidate(
    cpu_millis_free: u64,
    memory_bytes_free: u64,
    gpu_free: usize,
    demand: Demand,
) -> std::result::Result<(), String> {
    let candidate = LocalCandidate {
        name: "self",
        cpu_millis_free,
        memory_bytes_free,
        gpu_free,
    };
    let preds = fit_predicates(demand);
    // Single candidate today, so ranking is a no-op (`Ordering::Equal`); the
    // rank_by closure is the seam a cross-node extension plugs a real
    // least-loaded/best-fit comparator into.
    let report = scheduling::select(&[candidate], &preds, |_, _| std::cmp::Ordering::Equal, 1);
    if report.selected.is_some() {
        Ok(())
    } else {
        let reasons: Vec<String> = report
            .candidates
            .iter()
            .filter_map(|c| c.rejection_reason.clone())
            .collect();
        Err(reasons.join("; "))
    }
}

#[derive(Debug, Default)]
struct AdmissionState {
    active_total: usize,
    active_by_subject: HashMap<String, usize>,
    active_by_group: HashMap<String, usize>,
    cpu_millis_reserved: u64,
    memory_bytes_reserved: u64,
    gpu_reserved: usize,
    /// Bounded **FIFO** wait-queue. A release hands capacity to the *head*
    /// (front) only; a new arrival never barges past a non-empty queue. See
    /// module docs, point 3, for the head-of-line-blocking tradeoff and why
    /// this is not yet a per-key fair sub-queue.
    queue: VecDeque<Arc<Notify>>,
}

/// Why [`try_admit_locked`] refused to admit. Three failure classes with
/// different queueing semantics:
/// - `Infeasible`: demand exceeds the node's *configured totals* — no release
///   can ever satisfy it. Never queues; immediate hard reject.
/// - `Quota`: the Subject/group is at its per-key ceiling. Never queues;
///   retrying won't help until that Subject/group frees up on its own.
/// - `Capacity`: transient — a release elsewhere can free room. May queue.
enum AdmitReject {
    Infeasible(String),
    Quota(String),
    Capacity(String),
}

fn check_feasible(
    demand: Demand,
    cfg: &AdmissionConfig,
    max_sandboxes: usize,
) -> std::result::Result<(), String> {
    if max_sandboxes == 0 {
        return Err(
            "pool declares zero schedulable capacity (max_sandboxes = 0)".to_owned(),
        );
    }
    if let Some(total) = cfg.cpu_millis_total {
        if demand.cpu_millis > total {
            return Err(format!(
                "cpu demand {}m exceeds node total {total}m",
                demand.cpu_millis
            ));
        }
    }
    if let Some(total) = cfg.memory_bytes_total {
        if demand.memory_bytes > total {
            return Err(format!(
                "memory demand {} bytes exceeds node total {total} bytes",
                demand.memory_bytes
            ));
        }
    }
    if demand.gpu > cfg.gpu_total {
        return Err(format!(
            "gpu demand {} exceeds node total {}",
            demand.gpu, cfg.gpu_total
        ));
    }

    Ok(())
}

/// Check admission for `(subject_key, group_key, demand)` and, if it fits,
/// commit the reservation — atomically, under the caller's lock. This is the
/// entire "reserve-then-place" critical section; there is no gap between
/// "checked OK" and "reserved" (the #489/#519 TOCTOU lesson).
#[allow(clippy::too_many_arguments)]
fn try_admit_locked(
    state: &mut AdmissionState,
    subject_key: &str,
    group_key: Option<&str>,
    demand: Demand,
    cfg: &AdmissionConfig,
    max_sandboxes: usize,
) -> std::result::Result<(), AdmitReject> {
    // ── Feasibility first: demand exceeding the configured TOTALS (not just
    //    the currently-free amount) can never be satisfied by any release, so
    //    it must reject immediately rather than burn the whole queue_timeout.
    //    Checked independently of `*_reserved`. ──
    check_feasible(demand, cfg, max_sandboxes).map_err(AdmitReject::Infeasible)?;

    // ── Quota checks: fail-closed, never queue ──
    if let Some(max) = cfg.max_per_subject {
        let subj_count = *state.active_by_subject.get(subject_key).unwrap_or(&0);
        if subj_count >= max {
            return Err(AdmitReject::Quota(format!(
                "subject '{subject_key}' at quota ({subj_count}/{max})"
            )));
        }
    }
    if let Some(g) = group_key {
        if let Some(max) = cfg.max_per_group {
            let gcount = *state.active_by_group.get(g).unwrap_or(&0);
            if gcount >= max {
                return Err(AdmitReject::Quota(format!(
                    "group '{g}' at quota ({gcount}/{max})"
                )));
            }
        }
    }

    // ── Capacity checks: may queue ──
    if state.active_total >= max_sandboxes {
        return Err(AdmitReject::Capacity(format!(
            "pool at capacity ({}/{max_sandboxes})",
            state.active_total
        )));
    }
    let cpu_free = cfg
        .cpu_millis_total
        .map(|t| t.saturating_sub(state.cpu_millis_reserved))
        .unwrap_or(u64::MAX);
    let memory_free = cfg
        .memory_bytes_total
        .map(|t| t.saturating_sub(state.memory_bytes_reserved))
        .unwrap_or(u64::MAX);
    let gpu_free = cfg.gpu_total.saturating_sub(state.gpu_reserved);
    if let Err(reason) = fit_over_local_candidate(cpu_free, memory_free, gpu_free, demand) {
        return Err(AdmitReject::Capacity(reason));
    }

    // ── Admit: commit the reservation ──
    state.active_total += 1;
    *state
        .active_by_subject
        .entry(subject_key.to_owned())
        .or_insert(0) += 1;
    if let Some(g) = group_key {
        *state.active_by_group.entry(g.to_owned()).or_insert(0) += 1;
    }
    state.cpu_millis_reserved += demand.cpu_millis;
    state.memory_bytes_reserved += demand.memory_bytes;
    state.gpu_reserved += demand.gpu;
    Ok(())
}

/// Wake the current FIFO head (if any) so it retries admission. `Notify`
/// stores one permit if the head isn't yet awaiting, so this never loses a
/// wakeup. Idempotent-ish: a spurious extra wake just makes the head re-check
/// under the lock.
fn wake_head(state: &AdmissionState) {
    if let Some(head) = state.queue.front() {
        head.notify_one();
    }
}

fn release_locked(state: &mut AdmissionState, record: &ReservationRecord) {
    state.active_total = state.active_total.saturating_sub(1);
    if let Some(c) = state.active_by_subject.get_mut(&record.subject) {
        *c = c.saturating_sub(1);
        if *c == 0 {
            state.active_by_subject.remove(&record.subject);
        }
    }
    if let Some(g) = &record.group {
        if let Some(c) = state.active_by_group.get_mut(g) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                state.active_by_group.remove(g);
            }
        }
    }
    state.cpu_millis_reserved = state
        .cpu_millis_reserved
        .saturating_sub(record.demand.cpu_millis);
    state.memory_bytes_reserved = state
        .memory_bytes_reserved
        .saturating_sub(record.demand.memory_bytes);
    state.gpu_reserved = state.gpu_reserved.saturating_sub(record.demand.gpu);
}

/// Resolves an untrusted group *selector* against the verified [`Subject`]'s
/// membership (B′, ratified in #921 decision 5).
///
/// The group is **never** trusted from the annotation. The annotation is only
/// a selector naming which of the groups the Subject *provably belongs to* a
/// request should be partitioned under. This trait is the seam that resolves
/// that selector against the authoritative membership source — the
/// bidirectional-consent placement group
/// (`ai.hyprstream.placement.{group,groupItem}` / `PlacementIndex::is_member`,
/// in `hyprstream-pds`/`hyprstream-discovery`).
///
/// # Fail-closed
///
/// The default impl ([`DenyUnknownGroupValidator`]) denies **every** non-empty
/// selector, so a pool constructed without an explicit membership-backed
/// validator can never partition capacity by an unverified group. Wire a
/// real membership-source-backed impl at construction (where the discovery /
/// PDS client is available) to enable group partitioning.
///
/// Returning `Ok(None)` for an absent selector means **no group partition** —
/// never a quota bypass.
#[async_trait]
pub trait GroupSelectorValidator: Send + Sync + std::fmt::Debug {
    /// Resolve `selector` for `subject`:
    /// - `Ok(None)` — no selector present → no group partition.
    /// - `Ok(Some(group))` — selector validated: `subject` is a consented
    ///   member of `group`, which becomes the authoritative partition key.
    /// - `Err(_)` — selector present but `subject` is not a member → reject.
    async fn validate(&self, subject: &Subject, selector: Option<&str>)
        -> Result<Option<String>>;
}

/// Fail-closed default [`GroupSelectorValidator`]: denies every non-empty
/// group selector (no membership source is wired), permits the absent-selector
/// (no-partition) case. This is the production default until a real
/// membership-backed validator is wired — an unverified group can never
/// partition capacity.
#[derive(Debug, Default)]
pub struct DenyUnknownGroupValidator;

#[async_trait]
impl GroupSelectorValidator for DenyUnknownGroupValidator {
    async fn validate(
        &self,
        _subject: &Subject,
        selector: Option<&str>,
    ) -> Result<Option<String>> {
        match selector {
            None => Ok(None),
            Some(sel) => Err(WorkerError::AdmissionDenied {
                reason: format!(
                    "group selector '{sel}' cannot be verified: no membership source wired (fail-closed, B′)"
                ),
            }),
        }
    }
}

/// Reference [`GroupSelectorValidator`] backed by a static in-memory
/// membership set (group → member Subject keys). Models the authoritative
/// bidirectional-consent membership fact without the PDS/discovery plumbing;
/// a production deployment wires an `is_member`-backed impl instead.
#[derive(Debug, Default)]
pub struct StaticGroupMembership {
    members: HashMap<String, HashSet<String>>,
}

impl StaticGroupMembership {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `subject` is a consented member of `group`.
    pub fn with_member(mut self, group: &str, subject: &str) -> Self {
        self.members
            .entry(group.to_owned())
            .or_default()
            .insert(subject.to_owned());
        self
    }
}

#[async_trait]
impl GroupSelectorValidator for StaticGroupMembership {
    async fn validate(
        &self,
        subject: &Subject,
        selector: Option<&str>,
    ) -> Result<Option<String>> {
        match selector {
            None => Ok(None),
            Some(sel) => {
                let subj = subject.name().unwrap_or("");
                if self.members.get(sel).is_some_and(|m| m.contains(subj)) {
                    Ok(Some(sel.to_owned()))
                } else {
                    Err(WorkerError::AdmissionDenied {
                        reason: format!(
                            "subject '{subj}' is not a consented member of group '{sel}'"
                        ),
                    })
                }
            }
        }
    }
}

/// Admission decision engine backing `SandboxPool::acquire` (#525 P2).
#[derive(Debug)]
pub struct AdmissionTracker {
    state: Mutex<AdmissionState>,
    config: AdmissionConfig,
    max_sandboxes: usize,
    group_validator: Arc<dyn GroupSelectorValidator>,
}

impl AdmissionTracker {
    /// Construct with the fail-closed [`DenyUnknownGroupValidator`] — the
    /// production default until a membership-backed validator is wired
    /// (any group selector is rejected; the no-selector case is unaffected).
    pub fn new(config: AdmissionConfig, max_sandboxes: usize) -> Self {
        Self::with_group_validator(
            config,
            max_sandboxes,
            Arc::new(DenyUnknownGroupValidator),
        )
    }

    /// Construct with an explicit [`GroupSelectorValidator`] (e.g. one backed
    /// by a real membership source, or a test double).
    pub fn with_group_validator(
        config: AdmissionConfig,
        max_sandboxes: usize,
        group_validator: Arc<dyn GroupSelectorValidator>,
    ) -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            config,
            max_sandboxes,
            group_validator,
        }
    }

    /// Reserve capacity for `(subject, group_selector, demand)`, fail-closed on
    /// unauthenticated callers, queueing (bounded, **FIFO**) when only
    /// *capacity* (not quota/infeasibility) blocks admission.
    ///
    /// `group_selector` is the untrusted `hyprstream.io/group` annotation; it
    /// is resolved against the verified Subject's membership by the configured
    /// [`GroupSelectorValidator`] (B′). A selector the Subject is not a member
    /// of is rejected; an absent selector means no group partition.
    ///
    /// FIFO (F4): a new arrival never barges past a non-empty queue, and each
    /// release hands capacity to the queue head. See module docs, point 3, for
    /// the head-of-line-blocking tradeoff.
    pub async fn reserve(
        &self,
        subject: &Subject,
        group_selector: Option<&str>,
        demand: Demand,
    ) -> Result<ReservationRecord> {
        // Fail-closed identity: no verified Subject → reject before any
        // bookkeeping, queueing, or lock acquisition whatsoever.
        if subject.is_anonymous() {
            return Err(WorkerError::Unauthorized {
                subject: "anonymous".to_owned(),
                operation: "acquire".to_owned(),
                resource: "sandbox-pool".to_owned(),
            });
        }
        let subject_key = subject.name().unwrap_or("").to_owned();

        // B′: derive the authoritative group from the verified Subject, never
        // trust the annotation. Rejects a selector the Subject is not a member
        // of; `None` means no group partition.
        let group: Option<String> = self
            .group_validator
            .validate(subject, group_selector)
            .await?;
        let group_key = group.as_deref();

        let deadline = Instant::now() + Duration::from_secs(self.config.queue_timeout_secs);
        // Our place in the FIFO queue, once we join it. `None` until we first
        // fail an admission attempt on *capacity* (or find the queue non-empty
        // ahead of us).
        let mut ticket: Option<Arc<Notify>> = None;

        loop {
            {
                let mut state = self.state.lock().await;

                // A caller may attempt admission only if nobody is ahead of it
                // in the FIFO: it holds the head ticket, or it holds no ticket
                // and the queue is empty (no barging past waiters — F4).
                let may_attempt = match &ticket {
                    Some(t) => state.queue.front().is_some_and(|h| Arc::ptr_eq(h, t)),
                    None => state.queue.is_empty(),
                };

                if may_attempt {
                    match try_admit_locked(
                        &mut state,
                        &subject_key,
                        group_key,
                        demand,
                        &self.config,
                        self.max_sandboxes,
                    ) {
                        Ok(()) => {
                            if ticket.is_some() {
                                state.queue.pop_front();
                            }
                            wake_head(&state);
                            return Ok(ReservationRecord {
                                subject: subject_key,
                                group,
                                demand,
                            });
                        }
                        Err(AdmitReject::Infeasible(reason)) => {
                            if ticket.is_some() {
                                state.queue.pop_front();
                                wake_head(&state);
                            }
                            return Err(WorkerError::AdmissionInfeasible { reason });
                        }
                        Err(AdmitReject::Quota(reason)) => {
                            if ticket.is_some() {
                                state.queue.pop_front();
                                wake_head(&state);
                            }
                            return Err(WorkerError::AdmissionDenied { reason });
                        }
                        Err(AdmitReject::Capacity(reason)) => {
                            if ticket.is_none() {
                                if state.queue.len() >= self.config.queue_capacity {
                                    return Err(WorkerError::QueueFull {
                                        waiting: state.queue.len(),
                                        bound: self.config.queue_capacity,
                                    });
                                }
                                tracing::debug!(subject = %subject_key, reason = %reason, "acquire: queueing for capacity (FIFO)");
                                let t = Arc::new(Notify::new());
                                state.queue.push_back(t.clone());
                                ticket = Some(t);
                            }
                            // else: we are the head and capacity still isn't
                            // free — stay queued and wait again.
                        }
                    }
                } else if ticket.is_none() {
                    // Someone is ahead of us: join the back of the queue
                    // (bounded) and wait our turn rather than barging. Demand
                    // infeasible against configured totals never queues, even
                    // behind existing waiters.
                    if let Err(reason) = check_feasible(demand, &self.config, self.max_sandboxes) {
                        return Err(WorkerError::AdmissionInfeasible { reason });
                    }
                    if state.queue.len() >= self.config.queue_capacity {
                        return Err(WorkerError::QueueFull {
                            waiting: state.queue.len(),
                            bound: self.config.queue_capacity,
                        });
                    }
                    let t = Arc::new(Notify::new());
                    state.queue.push_back(t.clone());
                    ticket = Some(t);
                }
            }

            // Wait to be handed capacity (or to reach the head), bounded by the
            // deadline. Every path that falls through to here has set `ticket`;
            // the `None` arm is unreachable, so re-loop rather than panic.
            let t = match &ticket {
                Some(t) => t.clone(),
                None => continue,
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            let timed_out = remaining.is_zero()
                || tokio::time::timeout(remaining, t.notified()).await.is_err();
            if timed_out {
                let mut state = self.state.lock().await;
                let was_head = state.queue.front().is_some_and(|h| Arc::ptr_eq(h, &t));
                state.queue.retain(|x| !Arc::ptr_eq(x, &t));
                // If the head gave up, hand its turn to the next waiter.
                if was_head {
                    wake_head(&state);
                }
                return Err(WorkerError::QueueTimeout {
                    timeout_secs: self.config.queue_timeout_secs,
                });
            }
            // Woken: loop back and retry admission under the lock.
        }
    }

    /// Give back a committed reservation (a sandbox using it was released or
    /// destroyed) and hand the freed capacity to the FIFO queue head (F4 — no
    /// thundering-herd re-race).
    pub async fn release(&self, record: &ReservationRecord) {
        let mut state = self.state.lock().await;
        release_locked(&mut state, record);
        wake_head(&state);
    }

    /// Give back a reservation that was committed but never resulted in a
    /// usable sandbox (e.g. the backend failed to create/reconfigure one
    /// after admission succeeded). Identical to [`Self::release`] — kept as
    /// a distinctly-named entry point for readability at call sites.
    pub async fn rollback(&self, record: &ReservationRecord) {
        self.release(record).await;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn demand(cpu_millis: u64, memory_bytes: u64, gpu: usize) -> Demand {
        Demand {
            cpu_millis,
            memory_bytes,
            gpu,
        }
    }

    #[tokio::test]
    async fn anonymous_subject_rejected_fail_closed() {
        let tracker = AdmissionTracker::new(AdmissionConfig::default(), 10);
        let err = tracker
            .reserve(&Subject::anonymous(), None, demand(0, 0, 0))
            .await
            .unwrap_err();
        assert!(
            matches!(err, WorkerError::Unauthorized { .. }),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn zero_demand_never_trips_resource_capacity() {
        // No cpu/mem/gpu totals configured (the default) → a zero-demand
        // request (the CRI "unspecified" convention) always fits, matching
        // pre-#525 behavior exactly.
        let tracker = AdmissionTracker::new(AdmissionConfig::default(), 10);
        let r = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await;
        assert!(r.is_ok());
    }

    /// Membership validator with alice + bob both consented members of
    /// team-a — models the authoritative bidirectional-consent membership
    /// fact so the group tests exercise quota, not the fail-closed default.
    fn team_a_membership() -> Arc<dyn GroupSelectorValidator> {
        Arc::new(
            StaticGroupMembership::new()
                .with_member("team-a", "alice")
                .with_member("team-a", "bob"),
        )
    }

    #[tokio::test]
    async fn per_subject_quota_rejects_not_queues() {
        let cfg = AdmissionConfig {
            max_per_subject: Some(1),
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        let alice = Subject::new("alice");
        let r1 = tracker.reserve(&alice, None, demand(0, 0, 0)).await;
        assert!(r1.is_ok());
        let r2 = tracker.reserve(&alice, None, demand(0, 0, 0)).await;
        assert!(
            matches!(r2, Err(WorkerError::AdmissionDenied { .. })),
            "got: {r2:?}"
        );
    }

    #[tokio::test]
    async fn per_group_quota_rejects_across_subjects() {
        let cfg = AdmissionConfig {
            max_per_group: Some(1),
            ..Default::default()
        };
        let tracker = AdmissionTracker::with_group_validator(cfg, 10, team_a_membership());
        let r1 = tracker
            .reserve(&Subject::new("alice"), Some("team-a"), demand(0, 0, 0))
            .await;
        assert!(r1.is_ok());
        let r2 = tracker
            .reserve(&Subject::new("bob"), Some("team-a"), demand(0, 0, 0))
            .await;
        assert!(
            matches!(r2, Err(WorkerError::AdmissionDenied { .. })),
            "got: {r2:?}"
        );
    }

    // ── B′: group is derived from the verified Subject, never trusted from
    //    the annotation (F6) ────────────────────────────────────────────────

    #[tokio::test]
    async fn group_selector_from_non_member_is_rejected() {
        // Only alice is a member of team-a.
        let validator = Arc::new(StaticGroupMembership::new().with_member("team-a", "alice"));
        let tracker =
            AdmissionTracker::with_group_validator(AdmissionConfig::default(), 10, validator);
        // A member's selector validates and partitions.
        let r_alice = tracker
            .reserve(&Subject::new("alice"), Some("team-a"), demand(0, 0, 0))
            .await;
        assert!(r_alice.is_ok(), "member's selector must validate: {r_alice:?}");
        // A non-member asserting team-a is rejected — no cross-tenant quota
        // consumption.
        let r_bob = tracker
            .reserve(&Subject::new("bob"), Some("team-a"), demand(0, 0, 0))
            .await;
        assert!(
            matches!(r_bob, Err(WorkerError::AdmissionDenied { .. })),
            "non-member selector must be rejected: {r_bob:?}"
        );
    }

    #[tokio::test]
    async fn deny_unknown_group_validator_rejects_any_selector_but_allows_none() {
        // Default tracker uses the fail-closed DenyUnknownGroupValidator.
        let tracker = AdmissionTracker::new(AdmissionConfig::default(), 10);
        // No selector → no group partition, admitted.
        let r_none = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await;
        assert!(r_none.is_ok(), "absent selector must not partition: {r_none:?}");
        // Any selector is rejected fail-closed (no membership source wired).
        let r_sel = tracker
            .reserve(&Subject::new("alice"), Some("team-a"), demand(0, 0, 0))
            .await;
        assert!(
            matches!(r_sel, Err(WorkerError::AdmissionDenied { .. })),
            "fail-closed: unverifiable selector must be rejected: {r_sel:?}"
        );
    }

    #[tokio::test]
    async fn gpu_capacity_enforced_only_when_requested() {
        let cfg = AdmissionConfig {
            gpu_total: 1,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        // No GPU asked for → unaffected even though gpu_total is small.
        let r = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await;
        assert!(r.is_ok());
        // One GPU asked for and available → admitted.
        let r1 = tracker
            .reserve(&Subject::new("bob"), None, demand(0, 0, 1))
            .await;
        assert!(r1.is_ok());
        // Second GPU request with none free → queues then times out (queue
        // bound left at default; timeout shortened for the test).
    }

    #[tokio::test]
    async fn gpu_over_capacity_queues_then_times_out() {
        let cfg = AdmissionConfig {
            gpu_total: 1,
            queue_timeout_secs: 0,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        let r1 = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 1))
            .await;
        assert!(r1.is_ok());
        let r2 = tracker
            .reserve(&Subject::new("bob"), None, demand(0, 0, 1))
            .await;
        assert!(
            matches!(r2, Err(WorkerError::QueueTimeout { .. })),
            "got: {r2:?}"
        );
    }

    // ── F1: demand exceeding configured TOTALS rejects immediately
    //    (infeasible), never burns the queue timeout ─────────────────────────

    #[tokio::test]
    async fn gpu_demand_exceeding_total_rejects_immediately_infeasible() {
        // gpu_total defaults to 0. A GPU request can never be satisfied, so it
        // must reject *immediately* (Infeasible), not queue for 30s.
        let cfg = AdmissionConfig {
            // Long timeout so a wrong "queue" answer would be obvious/slow.
            queue_timeout_secs: 30,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        let start = Instant::now();
        let r = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 1))
            .await;
        assert!(
            matches!(r, Err(WorkerError::AdmissionInfeasible { .. })),
            "got: {r:?}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "infeasible demand must reject immediately, not wait out the timeout"
        );
    }

    #[tokio::test]
    async fn memory_demand_exceeding_total_is_infeasible() {
        let cfg = AdmissionConfig {
            memory_bytes_total: Some(256),
            queue_timeout_secs: 30,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        let start = Instant::now();
        let r = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 512, 0))
            .await;
        assert!(
            matches!(r, Err(WorkerError::AdmissionInfeasible { .. })),
            "got: {r:?}"
        );
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn cpu_demand_exceeding_total_is_infeasible() {
        let cfg = AdmissionConfig {
            cpu_millis_total: Some(1000),
            queue_timeout_secs: 30,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        let r = tracker
            .reserve(&Subject::new("alice"), None, demand(2000, 0, 0))
            .await;
        assert!(
            matches!(r, Err(WorkerError::AdmissionInfeasible { .. })),
            "got: {r:?}"
        );
    }

    #[tokio::test]
    async fn demand_equal_to_total_is_feasible_not_infeasible() {
        // Boundary: demand == total fits (busy, not infeasible). First takes
        // the GPU; a second equal request queues on *capacity* (would time
        // out), proving it wasn't classified infeasible.
        let cfg = AdmissionConfig {
            gpu_total: 1,
            queue_timeout_secs: 0,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
        let r1 = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 1))
            .await;
        assert!(r1.is_ok(), "demand == total must fit: {r1:?}");
        let r2 = tracker
            .reserve(&Subject::new("bob"), None, demand(0, 0, 1))
            .await;
        assert!(
            matches!(r2, Err(WorkerError::QueueTimeout { .. })),
            "second equal request is busy (capacity), not infeasible: {r2:?}"
        );
    }

    #[tokio::test]
    async fn infeasible_request_behind_waiter_rejects_without_taking_queue_slot() {
        let cfg = AdmissionConfig {
            gpu_total: 0,
            queue_capacity: 2,
            queue_timeout_secs: 5,
            ..Default::default()
        };
        let tracker = Arc::new(AdmissionTracker::new(cfg, 1));
        let held = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await
            .expect("first request should occupy the only sandbox slot");

        let t_bob = tracker.clone();
        let bob = tokio::spawn(async move {
            t_bob
                .reserve(&Subject::new("bob"), None, demand(0, 0, 0))
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        let start = Instant::now();
        let carol = tracker
            .reserve(&Subject::new("carol"), None, demand(0, 0, 1))
            .await;
        assert!(
            matches!(carol, Err(WorkerError::AdmissionInfeasible { .. })),
            "infeasible demand behind a waiter must fail immediately: {carol:?}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "infeasible demand must not wait behind the existing queue"
        );

        let t_dave = tracker.clone();
        let dave = tokio::spawn(async move {
            t_dave
                .reserve(&Subject::new("dave"), None, demand(0, 0, 0))
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !dave.is_finished(),
            "feasible follow-up should fit in the queue slot not consumed by the infeasible request"
        );

        tracker.release(&held).await;
        let bob_res = tokio::time::timeout(Duration::from_secs(2), bob)
            .await
            .expect("bob should be admitted")
            .expect("task ok")
            .expect("bob reservation ok");
        tracker.release(&bob_res).await;
        let dave_res = tokio::time::timeout(Duration::from_secs(2), dave)
            .await
            .expect("dave should be admitted")
            .expect("task ok");
        assert!(dave_res.is_ok(), "dave should have remained queued: {dave_res:?}");
    }

    // ── F4: FIFO — arrivals never barge past queued waiters ────────────────

    #[tokio::test]
    async fn fifo_earlier_waiter_is_not_starved_by_later_arrivals() {
        // Capacity 1. alice holds it; bob queues first, carol queues after.
        // Freeing one slot must admit bob (the earlier waiter), never carol.
        let cfg = AdmissionConfig {
            queue_timeout_secs: 5,
            ..Default::default()
        };
        let tracker = Arc::new(AdmissionTracker::new(cfg, 1));
        let held = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await
            .unwrap();

        let t_bob = tracker.clone();
        let bob = tokio::spawn(async move {
            t_bob
                .reserve(&Subject::new("bob"), None, demand(0, 0, 0))
                .await
        });
        // Ensure bob is queued at the head before carol arrives.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let t_carol = tracker.clone();
        let carol = tokio::spawn(async move {
            t_carol
                .reserve(&Subject::new("carol"), None, demand(0, 0, 0))
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Free exactly one slot.
        tracker.release(&held).await;

        // bob (earlier) must be the one admitted.
        let bob_res = tokio::time::timeout(Duration::from_secs(2), bob)
            .await
            .expect("bob should be woken")
            .expect("task ok");
        assert!(bob_res.is_ok(), "earlier waiter bob must be admitted: {bob_res:?}");

        // carol must still be waiting (only one slot freed, bob took it — no
        // barging).
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !carol.is_finished(),
            "later arrival carol must not barge ahead of / with bob"
        );

        // Clean up: release bob's slot so carol can finish.
        tracker.release(&bob_res.unwrap()).await;
        let carol_res = tokio::time::timeout(Duration::from_secs(2), carol)
            .await
            .expect("carol should now be woken")
            .expect("task ok");
        assert!(carol_res.is_ok(), "carol admitted once capacity frees: {carol_res:?}");
    }

    #[tokio::test]
    async fn queue_full_rejects_immediately() {
        let cfg = AdmissionConfig {
            queue_capacity: 0,
            queue_timeout_secs: 5,
            ..Default::default()
        };
        // Capacity 1, already taken; queue_capacity 0 means the very next
        // over-capacity request must reject immediately, not wait 5s.
        let tracker = AdmissionTracker::new(cfg, 1);
        let r1 = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await;
        assert!(r1.is_ok());
        let start = Instant::now();
        let r2 = tracker
            .reserve(&Subject::new("bob"), None, demand(0, 0, 0))
            .await;
        assert!(
            matches!(r2, Err(WorkerError::QueueFull { .. })),
            "got: {r2:?}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "must reject immediately, not wait out the timeout"
        );
    }

    #[tokio::test]
    async fn release_wakes_a_queued_waiter() {
        let cfg = AdmissionConfig {
            queue_timeout_secs: 5,
            ..Default::default()
        };
        let tracker = Arc::new(AdmissionTracker::new(cfg, 1));
        let r1 = tracker
            .reserve(&Subject::new("alice"), None, demand(0, 0, 0))
            .await
            .unwrap();

        let t2 = tracker.clone();
        let waiter = tokio::spawn(async move {
            t2.reserve(&Subject::new("bob"), None, demand(0, 0, 0))
                .await
        });

        // Give the waiter task a moment to actually enter the queue.
        tokio::time::sleep(Duration::from_millis(50)).await;
        tracker.release(&r1).await;

        let r2 = tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("waiter should be woken well within 2s")
            .expect("task should not panic");
        assert!(
            r2.is_ok(),
            "queued waiter should be admitted once capacity frees: {r2:?}"
        );
    }

    #[tokio::test]
    async fn derive_demand_is_zero_for_default_config() {
        let config = PodSandboxConfig::default();
        assert_eq!(derive_demand(&config), Demand::default());
        assert_eq!(derive_group_selector(&config), None);
    }

    #[tokio::test]
    async fn derive_demand_reads_cpu_memory_and_gpu_annotation() {
        let mut config = PodSandboxConfig::default();
        config.linux.resources.cpu_period = 100_000;
        config.linux.resources.cpu_quota = 50_000;
        config.linux.resources.memory_limit_in_bytes = 1024;
        config.annotations.push(KeyValue {
            key: ANN_GPU_REQUEST.to_owned(),
            value: "2".to_owned(),
        });
        config.annotations.push(KeyValue {
            key: ANN_GROUP.to_owned(),
            value: "team-a".to_owned(),
        });

        let d = derive_demand(&config);
        assert_eq!(d.cpu_millis, 500);
        assert_eq!(d.memory_bytes, 1024);
        assert_eq!(d.gpu, 2);
        assert_eq!(derive_group_selector(&config).as_deref(), Some("team-a"));
    }
}
