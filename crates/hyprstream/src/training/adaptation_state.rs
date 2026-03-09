use std::collections::HashMap;
use std::time::Instant;
use anyhow::{anyhow, Result};
use tch::Tensor;

/// Outcome of resolving an adaptation through the state machine.
#[derive(Debug)]
pub enum ResolveOutcome {
    WrittenBack,
    /// The adaptation was evicted (rolled back). The baseline snapshot is returned
    /// so the caller can restore delta weights without retaining ownership beforehand.
    Evicted {
        snapshot: HashMap<String, Tensor>,
        muon: HashMap<String, Tensor>,
        eff_ranks: HashMap<String, usize>,
    },
    StoredPending,
    Skipped { reason: String },
}

/// Strategy for handling a completed adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationStrategy {
    /// Write back if recommendation positive, evict if negative.
    AutoWriteback,
    /// Always evict — eval/benchmark mode.
    AutoEvict,
    /// Store as pending — client calls writeback() or evict() later.
    Speculative,
    /// Write back if loss_improvement exceeds threshold, else evict.
    WritebackIfAbove { threshold: f32 },
}

/// Guard status — proof that invariant checks were run before resolution.
/// Returned by the guard middleware; required input to resolve().
#[derive(Debug, Clone)]
pub struct GuardStatus {
    /// True if the pending adaptation has expired (past timeout_ms).
    pub expired: bool,
    /// True if accumulated_steps >= max_accumulated_steps.
    pub at_capacity: bool,
    /// LoRA generation counter at guard-check time.
    pub lora_generation: u64,
}

/// Info returned after a successful writeback (for stats accounting by the caller).
#[derive(Debug, Clone)]
pub struct WritebackInfo {
    pub steps_performed: usize,
    pub loss_improvement: f32,
}

/// Snapshot returned after eviction (for restoring delta state by the caller).
#[derive(Debug)]
pub struct EvictedSnapshot {
    pub pre_snapshot: HashMap<String, Tensor>,
    pub pre_muon: HashMap<String, Tensor>,
    pub pre_eff_ranks: HashMap<String, usize>,
}

/// The lifecycle state of a tenant delta's pending adaptation.
///
/// Stored inside TenantDelta. Replaces the external
/// `pending_adaptations: Mutex<HashMap<Subject, PendingAdaptation>>`.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum DeltaAdaptationState {
    /// No pending work. Delta holds its last committed state.
    Idle,
    /// Adaptation applied in-place. Snapshot held for potential rollback.
    Pending {
        pre_snapshot: HashMap<String, Tensor>,
        pre_muon: HashMap<String, Tensor>,
        pre_eff_ranks: HashMap<String, usize>,
        ttt_result: super::ttt::TTTResult,
        created_at: Instant,
        timeout_ms: u64,
    },
}

impl DeltaAdaptationState {
    pub fn new() -> Self {
        Self::Idle
    }

    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }

    pub fn is_expired(&self) -> bool {
        match self {
            Self::Pending { created_at, timeout_ms, .. } => {
                created_at.elapsed().as_millis() as u64 > *timeout_ms
            }
            Self::Idle => false,
        }
    }

    /// Resolve a new adaptation result according to the given strategy.
    ///
    /// This is the single entry point for all commit/rollback/pending decisions.
    /// Called by both the adapt-then-generate and pure-training paths.
    ///
    /// For the `Evicted` outcome, the baseline snapshot (pre-first-adaptation state)
    /// is returned inside `ResolveOutcome::Evicted` so the caller can restore weights
    /// without needing to retain ownership of the inputs before this call.
    pub fn resolve(
        &mut self,
        strategy: AdaptationStrategy,
        guard: &GuardStatus,
        result: &super::ttt::TTTResult,
        pre_snapshot: HashMap<String, Tensor>,
        pre_muon: HashMap<String, Tensor>,
        pre_eff_ranks: HashMap<String, usize>,
    ) -> ResolveOutcome {
        // 1. Capacity check
        if guard.at_capacity && !result.skipped {
            return ResolveOutcome::Skipped {
                reason: "Delta at capacity. Save, export, or reset to continue.".into(),
            };
        }

        // 2. Expired pending: auto-evict silently (guard detected, we clear it)
        // NOTE: We transition the *state* to Idle but do NOT restore the delta's
        // LoRA weights here. The expired adaptation's weight changes remain in the
        // live delta. Weight restoration requires &mut TenantDelta, which will be
        // threaded through resolve() in the service-layer integration (Task 3/4).
        // Callers that receive ResolveOutcome::Evicted are responsible for calling
        // delta.load_state_dict(&snapshot) with whatever snapshot is appropriate.
        if guard.expired && self.is_pending() {
            *self = Self::Idle;
        }

        // 3. Skipped result: no state change
        if result.skipped {
            return ResolveOutcome::Skipped {
                reason: result.skip_reason.clone().unwrap_or_default(),
            };
        }

        // 4. Stacked adaptation: promote old baseline.
        // If an old Pending exists, extract its stored baseline (pre-first-adaptation
        // state) so that rollback always unwinds to before any pending work.
        // Otherwise use the caller's snapshot directly (moved in, no copy needed).
        let (base_snapshot, base_muon, base_eff_ranks) = if let Self::Pending {
            pre_snapshot: old_snap,
            pre_muon: old_muon,
            pre_eff_ranks: old_ranks,
            ..
        } = std::mem::replace(self, Self::Idle) {
            // Old pending exists: promote its baseline (pre-first-adaptation state)
            (old_snap, old_muon, old_ranks)
        } else {
            (pre_snapshot, pre_muon, pre_eff_ranks)
        };

        // 5. Apply strategy
        match strategy {
            AdaptationStrategy::AutoWriteback if result.recommendation => {
                *self = Self::Idle;
                ResolveOutcome::WrittenBack
            }
            AdaptationStrategy::AutoWriteback | AdaptationStrategy::AutoEvict => {
                *self = Self::Idle;
                ResolveOutcome::Evicted {
                    snapshot: base_snapshot,
                    muon: base_muon,
                    eff_ranks: base_eff_ranks,
                }
            }
            AdaptationStrategy::Speculative => {
                *self = Self::Pending {
                    pre_snapshot: base_snapshot,
                    pre_muon: base_muon,
                    pre_eff_ranks: base_eff_ranks,
                    ttt_result: result.clone(),
                    created_at: Instant::now(),
                    timeout_ms: 60_000,
                };
                ResolveOutcome::StoredPending
            }
            AdaptationStrategy::WritebackIfAbove { threshold } => {
                *self = Self::Idle;
                if result.loss_improvement > threshold {
                    ResolveOutcome::WrittenBack
                } else {
                    ResolveOutcome::Evicted {
                        snapshot: base_snapshot,
                        muon: base_muon,
                        eff_ranks: base_eff_ranks,
                    }
                }
            }
        }
    }

    /// Write back (accept) a pending adaptation. Returns stats info for the caller
    /// to apply to the delta's counters.
    pub fn writeback_stats(&mut self) -> Result<WritebackInfo> {
        match std::mem::replace(self, Self::Idle) {
            Self::Pending { ttt_result, .. } => {
                Ok(WritebackInfo {
                    steps_performed: ttt_result.steps_performed,
                    loss_improvement: ttt_result.loss_improvement,
                })
            }
            Self::Idle => {
                Err(anyhow!("No pending adaptation to write back"))
            }
        }
    }

    /// Evict (discard) a pending adaptation. Returns the snapshot for the caller
    /// to restore onto the delta.
    pub fn evict(&mut self) -> Result<EvictedSnapshot> {
        match std::mem::replace(self, Self::Idle) {
            Self::Pending { pre_snapshot, pre_muon, pre_eff_ranks, .. } => {
                Ok(EvictedSnapshot { pre_snapshot, pre_muon, pre_eff_ranks })
            }
            Self::Idle => {
                Err(anyhow!("No pending adaptation to evict"))
            }
        }
    }
}

impl Default for DeltaAdaptationState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_guard(expired: bool, at_capacity: bool) -> GuardStatus {
        GuardStatus { expired, at_capacity, lora_generation: 0 }
    }

    fn make_fake_snapshot() -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        m.insert("test".into(), Tensor::zeros([2, 2], (tch::Kind::Float, tch::Device::Cpu)));
        m
    }

    fn make_fake_result(recommendation: bool, loss_improvement: f32, steps: usize) -> super::super::ttt::TTTResult {
        let mut r = super::super::ttt::TTTResult::skipped("test");
        r.recommendation = recommendation;
        r.loss_improvement = loss_improvement;
        r.steps_performed = steps;
        r.skipped = false;
        r.skip_reason = None;
        r
    }

    #[test]
    fn test_new_state_is_idle() {
        let state = DeltaAdaptationState::new();
        assert!(!state.is_pending());
        assert!(!state.is_expired());
    }

    #[test]
    fn test_resolve_auto_writeback_positive() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);
        let outcome = state.resolve(
            AdaptationStrategy::AutoWriteback,
            &guard,
            &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::WrittenBack));
        assert!(!state.is_pending());
    }

    #[test]
    fn test_resolve_auto_writeback_negative_evicts() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(false, -0.01, 3);
        let outcome = state.resolve(
            AdaptationStrategy::AutoWriteback,
            &guard,
            &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::Evicted { .. }));
        assert!(!state.is_pending());
    }

    #[test]
    fn test_resolve_speculative_stores_pending() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);
        let outcome = state.resolve(
            AdaptationStrategy::Speculative,
            &guard,
            &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::StoredPending));
        assert!(state.is_pending());
    }

    #[test]
    fn test_resolve_at_capacity_skips() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, true);
        let result = make_fake_result(true, 0.05, 3);
        let outcome = state.resolve(
            AdaptationStrategy::AutoWriteback,
            &guard,
            &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::Skipped { .. }));
    }

    #[test]
    fn test_resolve_expired_auto_evicts_before_new() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);
        let snap1 = make_fake_snapshot();
        state.resolve(
            AdaptationStrategy::Speculative, &guard, &result,
            snap1, HashMap::new(), HashMap::new(),
        );
        assert!(state.is_pending());

        let guard_expired = make_guard(true, false);
        let result2 = make_fake_result(true, 0.03, 2);
        let outcome = state.resolve(
            AdaptationStrategy::AutoWriteback, &guard_expired, &result2,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::WrittenBack));
    }

    #[test]
    fn test_stacked_pending_promotes_baseline() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);

        let snap1 = make_fake_snapshot();
        let snap1_keys: Vec<String> = snap1.keys().cloned().collect();
        state.resolve(
            AdaptationStrategy::Speculative, &guard, &result,
            snap1, HashMap::new(), HashMap::new(),
        );

        let snap2 = {
            let mut m = HashMap::new();
            m.insert("different_key".into(), Tensor::ones([2, 2], (tch::Kind::Float, tch::Device::Cpu)));
            m
        };
        let result2 = make_fake_result(true, 0.03, 2);
        state.resolve(
            AdaptationStrategy::Speculative, &guard, &result2,
            snap2, HashMap::new(), HashMap::new(),
        );

        if let DeltaAdaptationState::Pending { ref pre_snapshot, .. } = state {
            assert!(pre_snapshot.contains_key(&snap1_keys[0]),
                "Stacked pending should promote the FIRST snapshot as baseline");
        } else {
            panic!("Expected Pending state");
        }
    }

    #[test]
    fn test_writeback_from_idle_errors() {
        let mut state = DeltaAdaptationState::new();
        assert!(state.writeback_stats().is_err());
    }

    #[test]
    fn test_writeback_returns_stats() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);
        state.resolve(
            AdaptationStrategy::Speculative, &guard, &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        let info = state.writeback_stats().unwrap();
        assert_eq!(info.steps_performed, 3);
        assert!((info.loss_improvement - 0.05).abs() < 1e-6);
        assert!(!state.is_pending());
    }

    #[test]
    fn test_evict_from_idle_errors() {
        let mut state = DeltaAdaptationState::new();
        assert!(state.evict().is_err());
    }

    #[test]
    fn test_evict_returns_snapshot() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);
        state.resolve(
            AdaptationStrategy::Speculative, &guard, &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        let evicted = state.evict().unwrap();
        assert!(!evicted.pre_snapshot.is_empty());
        assert!(!state.is_pending());
    }

    #[test]
    fn test_auto_evict_always_evicts() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = make_fake_result(true, 0.05, 3);
        let outcome = state.resolve(
            AdaptationStrategy::AutoEvict, &guard, &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::Evicted { .. }));
    }

    #[test]
    fn test_writeback_if_above_threshold() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);

        let result = make_fake_result(true, 0.01, 3);
        let outcome = state.resolve(
            AdaptationStrategy::WritebackIfAbove { threshold: 0.02 },
            &guard, &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::Evicted { .. }));

        let result2 = make_fake_result(true, 0.05, 3);
        let outcome2 = state.resolve(
            AdaptationStrategy::WritebackIfAbove { threshold: 0.02 },
            &guard, &result2,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome2, ResolveOutcome::WrittenBack));
    }

    #[test]
    fn test_skipped_result_does_nothing() {
        let mut state = DeltaAdaptationState::new();
        let guard = make_guard(false, false);
        let result = crate::training::ttt::TTTResult::skipped("too short");
        let outcome = state.resolve(
            AdaptationStrategy::AutoWriteback, &guard, &result,
            make_fake_snapshot(), HashMap::new(), HashMap::new(),
        );
        assert!(matches!(outcome, ResolveOutcome::Skipped { .. }));
        assert!(!state.is_pending());
    }
}
