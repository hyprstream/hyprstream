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
//! 2. **"Group" identity**: there is no first-class "group" claim on
//!    [`hyprstream_rpc::auth::Claims`] today. `derive_group` reads it from a
//!    plain annotation (`hyprstream.io/group`) on the request rather than
//!    inventing a new capnp field or claim — deliberately, to avoid a
//!    wire-format change here. This is a placeholder pending a real
//!    tenant/group model.
//! 3. **Wait-queue fairness**: the queue is a single bounded FIFO-ish
//!    structure (all waiters are woken on every release and re-race for the
//!    lock), *not* the per-Subject/per-group fair sub-queue + weighted-
//!    round-robin the issue's design note describes. Quota-rejected requests
//!    never queue (they fail fast), which limits one noisy Subject's ability
//!    to starve others out of the *quota* axis, but the *capacity* wait queue
//!    itself has no per-key fairness weighting. Flagged as a scope-down, not
//!    a silent omission.
//! 4. **GPU/resource vocabulary**: cpu/memory/GPU demand is read from
//!    `config.linux.resources` (existing CRI-shaped fields) plus a new,
//!    separate annotation `hyprstream.io/gpu-request` (a plain integer count,
//!    not a GPU *class*) — independent of the existing boolean
//!    `hyprstream.io/gpu` passthrough annotation consumed by `oci_backend`/
//!    `nspawn`. The two are intentionally not unified in this change; see the
//!    module docs on `pool.rs` for the full rationale.
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

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

/// Annotation key: the admission "group" this request belongs to (#525
/// placeholder for a real tenant/group claim — see module docs, point 2).
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

/// Derive the admission "group" key for a request, if any (see module docs,
/// point 2).
pub fn derive_group(config: &PodSandboxConfig) -> Option<String> {
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
    /// Default: unconstrained (`usize::MAX`).
    pub max_per_subject: usize,
    /// Maximum concurrently-active sandboxes for a single group (see
    /// [`derive_group`]). Default: unconstrained (`usize::MAX`).
    pub max_per_group: usize,
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
            max_per_subject: usize::MAX,
            max_per_group: usize::MAX,
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
/// [`fit_report`] — feeding `queryCandidates` results in as additional
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
    /// Bounded wait-queue. On any release, every waiter here is woken and
    /// re-races to retry admission under the lock (see module docs, sign-off
    /// point 3, on why this is not a per-key fair sub-queue).
    queue: VecDeque<Arc<Notify>>,
}

/// Why [`try_admit_locked`] refused to admit — distinguishes *quota*
/// (never queues; retrying won't help until the Subject/group frees up on
/// its own) from *capacity* (may queue; a release elsewhere can free room).
enum AdmitReject {
    Quota(String),
    Capacity(String),
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
    // ── Quota checks first: fail-closed, never queue ──
    let subj_count = *state.active_by_subject.get(subject_key).unwrap_or(&0);
    if subj_count >= cfg.max_per_subject {
        return Err(AdmitReject::Quota(format!(
            "subject '{subject_key}' at quota ({subj_count}/{})",
            cfg.max_per_subject
        )));
    }
    if let Some(g) = group_key {
        let gcount = *state.active_by_group.get(g).unwrap_or(&0);
        if gcount >= cfg.max_per_group {
            return Err(AdmitReject::Quota(format!(
                "group '{g}' at quota ({gcount}/{})",
                cfg.max_per_group
            )));
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

/// Admission decision engine backing `SandboxPool::acquire` (#525 P2).
#[derive(Debug)]
pub struct AdmissionTracker {
    state: Mutex<AdmissionState>,
    config: AdmissionConfig,
    max_sandboxes: usize,
}

impl AdmissionTracker {
    pub fn new(config: AdmissionConfig, max_sandboxes: usize) -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            config,
            max_sandboxes,
        }
    }

    /// Reserve capacity for `(subject, group, demand)`, fail-closed on
    /// unauthenticated callers, queueing (bounded) when only *capacity* (not
    /// quota) blocks admission.
    pub async fn reserve(
        &self,
        subject: &Subject,
        group: Option<&str>,
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
        let deadline = Instant::now() + Duration::from_secs(self.config.queue_timeout_secs);

        loop {
            let ticket = {
                let mut state = self.state.lock().await;
                match try_admit_locked(
                    &mut state,
                    &subject_key,
                    group,
                    demand,
                    &self.config,
                    self.max_sandboxes,
                ) {
                    Ok(()) => {
                        return Ok(ReservationRecord {
                            subject: subject_key,
                            group: group.map(str::to_owned),
                            demand,
                        });
                    }
                    Err(AdmitReject::Quota(reason)) => {
                        return Err(WorkerError::AdmissionDenied { reason });
                    }
                    Err(AdmitReject::Capacity(reason)) => {
                        if state.queue.len() >= self.config.queue_capacity {
                            return Err(WorkerError::QueueFull {
                                waiting: state.queue.len(),
                                bound: self.config.queue_capacity,
                            });
                        }
                        tracing::debug!(subject = %subject_key, reason = %reason, "acquire: queueing for capacity");
                        let ticket = Arc::new(Notify::new());
                        state.queue.push_back(ticket.clone());
                        ticket
                    }
                }
            };

            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let mut state = self.state.lock().await;
                state.queue.retain(|t| !Arc::ptr_eq(t, &ticket));
                return Err(WorkerError::QueueTimeout {
                    timeout_secs: self.config.queue_timeout_secs,
                });
            }
            if tokio::time::timeout(remaining, ticket.notified())
                .await
                .is_err()
            {
                let mut state = self.state.lock().await;
                state.queue.retain(|t| !Arc::ptr_eq(t, &ticket));
                return Err(WorkerError::QueueTimeout {
                    timeout_secs: self.config.queue_timeout_secs,
                });
            }
            // Woken: loop back and retry admission under the lock.
        }
    }

    /// Give back a committed reservation (a sandbox using it was released or
    /// destroyed) and wake queued waiters to retry.
    pub async fn release(&self, record: &ReservationRecord) {
        let queue = {
            let mut state = self.state.lock().await;
            release_locked(&mut state, record);
            std::mem::take(&mut state.queue)
        };
        for ticket in queue {
            ticket.notify_one();
        }
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

    #[tokio::test]
    async fn per_subject_quota_rejects_not_queues() {
        let cfg = AdmissionConfig {
            max_per_subject: 1,
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
            max_per_group: 1,
            ..Default::default()
        };
        let tracker = AdmissionTracker::new(cfg, 10);
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
        assert_eq!(derive_group(&config), None);
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
        assert_eq!(derive_group(&config).as_deref(), Some("team-a"));
    }
}
