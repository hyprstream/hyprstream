//! Test-Time Training (TTT) Module
//!
//! Implements research-valid test-time training that adapts the model
//! to the input distribution BEFORE generation using next-token prediction
//! loss on the input context itself.
//!
//! # Research Foundation
//!
//! Based on TTT-E2E (arxiv.org/abs/2512.23675) and TLM approaches:
//! - Train on INPUT context (not model outputs)
//! - Use next-token prediction loss (not confidence heuristics)
//! - Adapt BEFORE generation (not after)
//!
//! # Key Difference from SelfSupervisedTrainer
//!
//! | Aspect | SelfSupervised (old) | TTT (this) |
//! |--------|---------------------|------------|
//! | When | After generation | Before generation |
//! | Data | Model outputs | User inputs |
//! | Loss | Confidence heuristic | NTP cross-entropy |
//! | Weights | Permanent LoRA | Scratch (temporary) |

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};
use tch::{Device, Tensor};

use crate::runtime::TorchEngine;
use super::tenant_delta::TenantDelta;

/// Configuration for Test-Time Training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTTConfig {
    /// Learning rate for TTT adaptation
    /// Higher than fine-tuning since we do few steps
    /// Recommended: 1e-4 to 5e-4
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Number of gradient steps per input
    /// More steps = better adaptation but higher latency
    /// Recommended: 1-5
    #[serde(default = "default_gradient_steps")]
    pub gradient_steps: usize,

    /// Maximum gradient norm for clipping
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,

    /// Minimum input length (tokens) to trigger TTT
    /// Short inputs don't benefit from adaptation
    #[serde(default = "default_min_input_length")]
    pub min_input_length: usize,

    /// Maximum input length to process for TTT
    /// Truncates very long inputs to control latency
    #[serde(default = "default_max_ttt_context")]
    pub max_ttt_context: usize,

    /// Whether TTT is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    // Adaptive TTT thresholds (perplexity gating)
    /// Perplexity threshold below which adaptation is skipped (input already well-modeled)
    #[serde(default = "default_tau_skip")]
    pub tau_skip: f32,

    /// Perplexity threshold for light adaptation (1 step)
    #[serde(default = "default_tau_light")]
    pub tau_light: f32,

    /// Perplexity threshold for heavy adaptation (3 steps)
    #[serde(default = "default_tau_heavy")]
    pub tau_heavy: f32,

    /// Minimum loss improvement to consider adaptation beneficial (confidence gate)
    #[serde(default = "default_delta_min")]
    pub delta_min: f32,

    /// Whether to use adaptive step counts based on perplexity
    #[serde(default = "default_adaptive_steps")]
    pub adaptive_steps: bool,

    /// Maximum wall-clock time for a single TTT adaptation pass (milliseconds).
    /// When using ZMQ RPC (default timeout 30s), the default leaves headroom for
    /// serialization, network transfer, and response processing.
    #[serde(default = "default_max_adaptation_ms")]
    pub max_adaptation_ms: u64,

    /// Hard ceiling on gradient steps per request, regardless of client override.
    /// Prevents unbounded compute from malicious or misconfigured clients.
    #[serde(default = "default_max_gradient_steps")]
    pub max_gradient_steps: usize,

    /// Auto-rollback timeout for pending adaptations (milliseconds).
    /// After a non-auto-commit training, the server holds the pre-adaptation
    /// snapshot for this long before auto-rolling back. Increase for interactive
    /// workflows (MCP, human-in-the-loop); decrease for programmatic API clients.
    #[serde(default = "default_pending_rollback_ms")]
    pub pending_rollback_ms: u64,
}

fn default_learning_rate() -> f64 {
    3e-4
}
fn default_gradient_steps() -> usize {
    3
}
fn default_max_grad_norm() -> f64 {
    1.0
}
fn default_min_input_length() -> usize {
    32
}
fn default_max_ttt_context() -> usize {
    512
}
fn default_enabled() -> bool {
    true
}
fn default_tau_skip() -> f32 {
    5.0
}
fn default_tau_light() -> f32 {
    15.0
}
fn default_tau_heavy() -> f32 {
    50.0
}
fn default_delta_min() -> f32 {
    0.01
}
fn default_adaptive_steps() -> bool {
    true
}
fn default_max_adaptation_ms() -> u64 {
    20_000
}
fn default_max_gradient_steps() -> usize {
    50
}
fn default_pending_rollback_ms() -> u64 {
    60_000
}

/// Per-layer gradient gating configuration.
///
/// After the first gradient step, identifies low-signal layers via gradient
/// norms and sets `requires_grad_(false)` on their parameters for subsequent
/// steps. This saves backward-pass FLOPs and prevents Muon momentum
/// from drifting on frozen layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientGatingConfig {
    /// Enable per-layer gradient gating (default: true)
    #[serde(default = "default_grad_gating_enabled")]
    pub enabled: bool,
    /// Minimum gradient L2 norm to keep updating a layer (default: 1e-5)
    #[serde(default = "default_min_grad_norm")]
    pub min_grad_norm: f64,
    /// Number of initial adaptation steps before gating activates (default: 1)
    #[serde(default = "default_grad_warmup")]
    pub warmup_steps: usize,
}

fn default_grad_gating_enabled() -> bool {
    true
}
fn default_min_grad_norm() -> f64 {
    1e-5
}
fn default_grad_warmup() -> usize {
    1
}

impl Default for GradientGatingConfig {
    fn default() -> Self {
        Self {
            enabled: default_grad_gating_enabled(),
            min_grad_norm: default_min_grad_norm(),
            warmup_steps: default_grad_warmup(),
        }
    }
}

/// Compute per-layer gradient norms from VarStore.
///
/// Returns map of "layer_idx.module_name" -> L2 norm of combined A+B gradients.
fn compute_per_layer_grad_norms(
    vs: &tch::nn::VarStore,
) -> HashMap<String, f64> {
    let mut norms: HashMap<String, f64> = HashMap::new();
    for (name, var) in vs.variables() {
        if !var.grad().defined() {
            continue;
        }
        if let Some(key) = parse_varstore_key_to_delta_key(&name) {
            let grad_sq = var.grad().norm().double_value(&[]).powi(2);
            *norms.entry(key).or_insert(0.0) += grad_sq;
        }
    }
    // Take sqrt of accumulated squared norms
    for v in norms.values_mut() {
        *v = v.sqrt();
    }
    norms
}

/// Identify layers to gate (freeze) based on gradient norms.
fn identify_gated_layers(norms: &HashMap<String, f64>, threshold: f64) -> Vec<String> {
    norms
        .iter()
        .filter(|(_, &norm)| norm <= threshold)
        .map(|(key, _)| key.clone())
        .collect()
}

/// Parse VarStore key "layer_{idx}.{module}.lora_{a|b}" to delta key "idx.module"
fn parse_varstore_key_to_delta_key(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 3 {
        let layer_part = parts[0]; // "layer_0"
        let module = parts[1]; // "q_proj"
        if let Some(idx_str) = layer_part.strip_prefix("layer_") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                return Some(format!("{idx}.{module}"));
            }
        }
    }
    None
}

/// Configuration for runtime rank adaptation oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankOracleConfig {
    /// How many TTT adaptations between rank evaluations (default: 10)
    #[serde(default = "default_adaptation_interval")]
    pub adaptation_interval: usize,
    /// Whether to auto-adapt effective ranks (default: false — just log recommendations)
    #[serde(default)]
    pub auto_adapt: bool,
    /// Available rank levels for reallocation
    #[serde(default = "default_rank_levels")]
    pub rank_levels: Vec<usize>,
    /// Utilization below this triggers rank decrease (default: 0.25)
    #[serde(default = "default_low_util")]
    pub low_utilization_threshold: f64,
    /// Utilization above this triggers rank increase (default: 0.85)
    #[serde(default = "default_high_util")]
    pub high_utilization_threshold: f64,
}

fn default_adaptation_interval() -> usize {
    10
}
fn default_rank_levels() -> Vec<usize> {
    vec![1, 2, 4, 8]
}
fn default_low_util() -> f64 {
    0.25
}
fn default_high_util() -> f64 {
    0.85
}

impl Default for RankOracleConfig {
    fn default() -> Self {
        Self {
            adaptation_interval: default_adaptation_interval(),
            auto_adapt: false,
            rank_levels: default_rank_levels(),
            low_utilization_threshold: default_low_util(),
            high_utilization_threshold: default_high_util(),
        }
    }
}

/// Per-tenant rank adaptation oracle.
///
/// Wraps `RankUtilizationTracker` and produces rank adaptation signals.
/// Stored per-tenant (inside `TenantDelta`, not in `TestTimeTrainer`)
/// because utilization history is tenant-specific.
pub struct RankOracle {
    pub(crate) config: RankOracleConfig,
    tracker: crate::runtime::ttn_profile::RankUtilizationTracker,
    observation_count: usize,
}

impl RankOracle {
    pub fn new(config: RankOracleConfig) -> Self {
        let window = config.adaptation_interval * 2;
        Self {
            tracker: crate::runtime::ttn_profile::RankUtilizationTracker::new(window),
            config,
            observation_count: 0,
        }
    }

    pub fn observe(&mut self, utilizations: &HashMap<String, f64>) {
        for (key, &val) in utilizations {
            self.tracker.record(key, val);
        }
        self.observation_count += 1;
    }

    #[allow(clippy::manual_is_multiple_of)]
    pub fn should_evaluate(&self) -> bool {
        self.observation_count > 0 && self.observation_count % self.config.adaptation_interval == 0
    }

    pub fn recommend(
        &self,
    ) -> HashMap<String, crate::runtime::ttn_profile::RankSignal> {
        self.tracker.rank_adaptation_signals(
            self.config.low_utilization_threshold,
            self.config.high_utilization_threshold,
        )
    }
}

impl Default for TTTConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            gradient_steps: default_gradient_steps(),
            max_grad_norm: default_max_grad_norm(),
            min_input_length: default_min_input_length(),
            max_ttt_context: default_max_ttt_context(),
            enabled: default_enabled(),
            tau_skip: default_tau_skip(),
            tau_light: default_tau_light(),
            tau_heavy: default_tau_heavy(),
            delta_min: default_delta_min(),
            adaptive_steps: default_adaptive_steps(),
            max_adaptation_ms: default_max_adaptation_ms(),
            max_gradient_steps: default_max_gradient_steps(),
            pending_rollback_ms: default_pending_rollback_ms(),
        }
    }
}

/// Result of TTT adaptation
#[derive(Debug, Clone, serde::Serialize)]
pub struct TTTResult {
    /// Average loss across gradient steps
    pub avg_loss: f32,

    /// Loss improvement (initial - final)
    pub loss_improvement: f32,

    /// Number of gradient steps actually performed
    pub steps_performed: usize,

    /// Time spent on TTT (milliseconds)
    pub adaptation_time_ms: u64,

    /// Whether adaptation was skipped (input too short, etc.)
    pub skipped: bool,

    /// Reason for skipping (if skipped)
    pub skip_reason: Option<String>,

    // Advanced metrics (for ML practitioners)
    /// Average gradient norm across steps
    pub avg_grad_norm: f32,

    /// Maximum gradient norm observed
    pub max_grad_norm: f32,

    /// Whether gradients were clipped
    pub gradient_clipped: bool,

    /// Number of tokens actually used for adaptation (after truncation)
    pub tokens_used: usize,

    /// Total tokens provided in input
    pub tokens_provided: usize,

    /// Whether input was truncated (exceeded max_ttt_context)
    pub was_truncated: bool,

    // Tenant-aware TTT fields
    /// Initial perplexity before adaptation (exp(initial_loss))
    pub initial_perplexity: f32,

    /// Final perplexity after adaptation (exp(final_loss))
    pub final_perplexity: f32,

    /// Server's recommendation: true = commit, false = rollback
    pub recommendation: bool,

    /// Number of steps determined by perplexity gating
    pub gated_steps: usize,

    /// Whether adaptation is pending client commit/rollback
    pub pending: bool,

    /// Wall-clock time used for adaptation (milliseconds)
    pub time_used_ms: u64,

    /// Time budget that was enforced (milliseconds).
    /// Client can derive `timed_out = time_used_ms >= time_budget_ms`.
    pub time_budget_ms: u64,

    /// Per-key rank utilization from this adaptation (empty if skipped)
    pub rank_utilization: HashMap<String, f64>,

    /// Layers gated (gradients frozen) during this adaptation
    pub gated_layers: Vec<String>,

    /// Rank adaptation signals from this evaluation (empty if not evaluated)
    pub rank_signals: HashMap<String, crate::runtime::ttn_profile::RankSignal>,
}

impl TTTResult {
    /// Create a result indicating adaptation was skipped
    pub fn skipped(reason: &str) -> Self {
        Self {
            avg_loss: 0.0,
            loss_improvement: 0.0,
            steps_performed: 0,
            adaptation_time_ms: 0,
            skipped: true,
            skip_reason: Some(reason.to_owned()),
            avg_grad_norm: 0.0,
            max_grad_norm: 0.0,
            gradient_clipped: false,
            tokens_used: 0,
            tokens_provided: 0,
            was_truncated: false,
            initial_perplexity: 0.0,
            final_perplexity: 0.0,
            recommendation: false,
            gated_steps: 0,
            pending: false,
            time_used_ms: 0,
            time_budget_ms: 0,
            rank_utilization: HashMap::new(),
            gated_layers: Vec::new(),
            rank_signals: HashMap::new(),
        }
    }
}

/// Per-request TTT overrides from the client
#[derive(Debug, Clone, Default)]
pub struct TTTOverrides {
    /// Override: enable/disable TTT for this request
    pub enabled: Option<bool>,
    /// Override: number of gradient steps (0 = skip)
    pub gradient_steps: Option<u32>,
    /// Override: learning rate
    pub learning_rate: Option<f32>,
    /// If true, server auto-commits based on its recommendation
    pub auto_commit: bool,
    /// Override: maximum wall-clock time for adaptation (milliseconds)
    pub max_adaptation_ms: Option<u64>,
}

/// Context for TTT adaptation (for future cross-model verification)
#[derive(Debug, Clone)]
pub struct TTTContext {
    /// Original input tokens
    pub input_tokens: Vec<u32>,

    /// Loss values at each TTT step
    pub loss_history: Vec<f32>,

    /// Timestamp of adaptation
    pub adapted_at: Instant,
}

/// Test-Time Trainer
///
/// Adapts the model to input context BEFORE generation using next-token
/// prediction loss on the input itself. This is the core TTT algorithm.
///
/// # Thread Safety
///
/// Each adaptation run is isolated - scratch weights are created per-request
/// and do not persist between calls.
///
/// # Example
///
/// ```ignore
/// let trainer = TestTimeTrainer::new(TTTConfig::default(), Device::Cuda(0));
///
/// // Adapt to input BEFORE generation
/// let result = trainer.adapt(&engine, &input_tokens).await?;
///
/// // Now generate with adapted model
/// let response = engine.generate(request)?;
/// ```
pub struct TestTimeTrainer {
    pub config: TTTConfig,
    pub gradient_gating: GradientGatingConfig,
    device: Device,
}

impl TestTimeTrainer {
    /// Create a new TTT trainer with the given configuration
    pub fn new(config: TTTConfig, device: Device) -> Self {
        Self { config, gradient_gating: GradientGatingConfig::default(), device }
    }

    /// Create a new TTT trainer with gradient gating configuration
    pub fn with_gradient_gating(mut self, gating: GradientGatingConfig) -> Self {
        self.gradient_gating = gating;
        self
    }

    /// Check if TTT is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration
    pub fn config(&self) -> &TTTConfig {
        &self.config
    }

    /// Perform a single gradient step on a tenant delta using Muon
    ///
    /// Steps: backward → compute grad_norm → clip → muon_step → zero_grad
    pub fn ttt_step(
        &self,
        loss: &Tensor,
        delta: &mut TenantDelta,
        learning_rate: Option<f64>,
    ) -> Result<(f64, bool)> {
        // Override learning rate for this step
        if let Some(lr) = learning_rate {
            delta.muon_config.lr = lr;
        }

        // Backward pass
        loss.backward();

        // Compute gradient norm for metrics
        let grad_norm = {
            let mut total_norm_sq = 0.0f64;
            let variables = delta.vs.trainable_variables();

            for var in &variables {
                if var.grad().defined() {
                    let grad = var.grad();
                    total_norm_sq += grad.norm().double_value(&[]).powi(2);
                }
            }

            total_norm_sq.sqrt()
        };

        let clipped = grad_norm > self.config.max_grad_norm;
        if clipped {
            // Manual gradient clipping (replaces optimizer.clip_grad_norm)
            let clip_coef = self.config.max_grad_norm / (grad_norm + 1e-6);
            let _guard = tch::no_grad_guard();
            for var in delta.vs.trainable_variables() {
                if var.grad().defined() {
                    let clipped = &var.grad() * clip_coef;
                    var.grad().copy_(&clipped);
                }
            }
        }

        // Muon step for all trainable parameters + zero gradients
        self.optimizer_step(delta);

        Ok((grad_norm, clipped))
    }

    /// Perform Muon optimization step on all trainable parameters and zero gradients.
    fn optimizer_step(&self, delta: &mut TenantDelta) {
        use super::muon::muon_step;
        let _guard = tch::no_grad_guard();
        let variables = delta.vs.variables();
        let config = delta.muon_config.clone();
        for (name, var) in &variables {
            if var.requires_grad() && var.grad().defined() {
                let state = delta
                    .muon_states
                    .entry(name.clone())
                    .or_default();
                muon_step(var, state, &config);
            }
        }
        // Zero gradients
        for var in variables.values() {
            if var.grad().defined() {
                let _ = var.grad().zero_();
            }
        }
    }

    /// Determine number of gradient steps based on perplexity gating
    fn gate_steps(&self, perplexity: f32, overrides: &TTTOverrides) -> usize {
        // Client override takes precedence
        if let Some(steps) = overrides.gradient_steps {
            return steps as usize;
        }

        if !self.config.adaptive_steps {
            return self.config.gradient_steps;
        }

        if perplexity < self.config.tau_skip {
            0 // Input already well-modeled
        } else if perplexity < self.config.tau_light {
            1 // Light adaptation
        } else if perplexity < self.config.tau_heavy {
            3 // Moderate adaptation
        } else {
            5 // Heavy adaptation
        }
    }

    /// Compute server's commit recommendation based on confidence gate
    fn compute_recommendation(
        &self,
        loss_improvement: f32,
        _gradient_clipped: bool,
        initial_ppl: f32,
        final_ppl: f32,
    ) -> bool {
        // Gradient clipping is informational, not a veto — clipping is normal
        // during early adaptation steps, especially with freshly initialized deltas.
        loss_improvement > self.config.delta_min && final_ppl < initial_ppl
    }

    /// Adapt a tenant delta to input context using perplexity-gated TTT
    ///
    /// This is the tenant-aware entry point for TTT. It:
    /// 1. Snapshots delta state (for rollback)
    /// 2. Computes initial NTP loss → perplexity
    /// 3. Gates step count based on perplexity
    /// 4. Runs SGD gradient loop on the tenant delta
    /// 5. Returns metrics (does NOT auto-commit — client decides)
    ///
    /// # Arguments
    /// * `engine` - TorchEngine with loaded model
    /// * `delta` - Tenant's LoRA delta (mutable for SGD updates)
    /// * `input_tokens` - Tokenized input context
    /// * `overrides` - Per-request TTT overrides from client
    ///
    /// # Returns
    /// (TTTResult, pre_snapshot) where pre_snapshot is the state before adaptation
    pub fn adapt_tenant(
        &self,
        engine: &TorchEngine,
        delta: &mut TenantDelta,
        input_tokens: &[u32],
        overrides: &TTTOverrides,
    ) -> Result<(TTTResult, HashMap<String, Tensor>)> {
        let start = Instant::now();

        // Check if enabled (respect override)
        let enabled = overrides.enabled.unwrap_or(self.config.enabled);
        if !enabled {
            return Ok((TTTResult::skipped("TTT disabled"), HashMap::new()));
        }

        // Check minimum length
        if input_tokens.len() < self.config.min_input_length {
            return Ok((
                TTTResult::skipped(&format!(
                    "Input too short: {} < {} tokens",
                    input_tokens.len(),
                    self.config.min_input_length
                )),
                HashMap::new(),
            ));
        }

        // Snapshot delta state before adaptation (for rollback)
        let pre_snapshot = delta.extract_state_dict();

        // Snapshot SSM states before TTT loop (C4 fix: prevents training data from
        // accumulating into the inference recurrent state for Qwen3.5 GDN layers).
        // No-op for non-Qwen3.5 models.
        let ssm_snapshot = engine.snapshot_ssm_states();

        // Track whether input was truncated
        let tokens_provided = input_tokens.len();
        let was_truncated = tokens_provided > self.config.max_ttt_context;

        // Truncate if too long (use last N tokens for recency)
        let tokens: Vec<u32> = if was_truncated {
            input_tokens[input_tokens.len() - self.config.max_ttt_context..].to_vec()
        } else {
            input_tokens.to_vec()
        };
        let tokens_used = tokens.len();

        // Compute initial NTP loss
        let initial_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, &tokens)?;
        let initial_loss: f32 = initial_loss_tensor.double_value(&[]) as f32;
        let initial_perplexity = initial_loss.exp();

        // Gate step count based on perplexity, then clamp to max
        let gated_steps = self.gate_steps(initial_perplexity, overrides)
            .min(self.config.max_gradient_steps);

        let time_budget_ms = overrides.max_adaptation_ms
            .unwrap_or(self.config.max_adaptation_ms);

        if gated_steps == 0 {
            // Restore SSM states — initial NTP loss forward pass may have mutated them.
            engine.restore_ssm_states(ssm_snapshot);
            return Ok((
                TTTResult {
                    avg_loss: initial_loss,
                    loss_improvement: 0.0,
                    steps_performed: 0,
                    adaptation_time_ms: start.elapsed().as_millis() as u64,
                    skipped: true,
                    skip_reason: Some(format!(
                        "Perplexity {:.1} below skip threshold {:.1}",
                        initial_perplexity, self.config.tau_skip
                    )),
                    avg_grad_norm: 0.0,
                    max_grad_norm: 0.0,
                    gradient_clipped: false,
                    tokens_used,
                    tokens_provided,
                    was_truncated,
                    initial_perplexity,
                    final_perplexity: initial_perplexity,
                    recommendation: false,
                    gated_steps: 0,
                    pending: false,
                    time_used_ms: start.elapsed().as_millis() as u64,
                    time_budget_ms,
                    rank_utilization: HashMap::new(),
                    gated_layers: Vec::new(),
                    rank_signals: HashMap::new(),
                },
                pre_snapshot,
            ));
        }

        // Determine learning rate (respect override)
        let lr = overrides
            .learning_rate
            .map(|r| r as f64)
            .unwrap_or(delta.learning_rate);

        let budget = Duration::from_millis(time_budget_ms);

        // Snapshot Muon momentum buffers for rollback (matches weight snapshot above)
        let muon_snapshot = super::muon::snapshot_muon_states(&delta.muon_states);

        // Run gradient loop with catch_unwind for panic safety.
        // On panic, delta is restored from pre_snapshot.
        let loop_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.adapt_tenant_inner(engine, delta, &tokens, gated_steps, lr, &start, budget, initial_loss_tensor)
        }));

        match loop_result {
            Ok(Ok(inner)) => {
                let (actual_steps, losses, grad_norms, any_clipped, final_loss, final_perplexity, rank_utils, gated_layer_keys) = inner;

                let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;
                let loss_improvement = initial_loss - final_loss;

                let avg_grad_norm = if !grad_norms.is_empty() {
                    grad_norms.iter().sum::<f32>() / grad_norms.len() as f32
                } else {
                    0.0
                };
                let max_grad_norm_val = grad_norms.iter().copied().fold(0.0f32, f32::max);

                let recommendation = self.compute_recommendation(
                    loss_improvement,
                    any_clipped,
                    initial_perplexity,
                    final_perplexity,
                );

                // Track accumulated steps (adapt_tenant was missing this)
                delta.accumulated_steps += actual_steps as u64;
                delta.request_count += 1;
                let n = delta.request_count as f64;
                delta.avg_loss_improvement =
                    delta.avg_loss_improvement * ((n - 1.0) / n) + loss_improvement as f64 / n;

                let time_used_ms = start.elapsed().as_millis() as u64;

                // Per-tenant rank oracle: observe utilization and optionally adapt
                let rank_signals = if let Some(ref mut oracle) = delta.rank_oracle {
                    oracle.observe(&rank_utils);
                    if oracle.should_evaluate() {
                        let signals = oracle.recommend();
                        if oracle.config.auto_adapt {
                            use crate::runtime::ttn_profile::RankSignal;
                            for (key, signal) in &signals {
                                if let Some(current) = delta.effective_rank(key) {
                                    match signal {
                                        RankSignal::Decrease => {
                                            let new_rank = (current / 2).max(1);
                                            delta.set_effective_rank(key, new_rank);
                                            tracing::info!(key = %key, from = current, to = new_rank, "Rank oracle: decreased");
                                        }
                                        RankSignal::Increase => {
                                            let new_rank = current * 2;
                                            delta.set_effective_rank(key, new_rank);
                                            let actual = delta.effective_rank(key).unwrap_or(new_rank);
                                            tracing::info!(key = %key, from = current, to = actual, "Rank oracle: increased");
                                        }
                                        RankSignal::Hold => {}
                                    }
                                }
                            }
                        }
                        signals
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                };

                // Restore SSM states so TTT training data does not pollute inference recurrent state.
                engine.restore_ssm_states(ssm_snapshot);

                Ok((
                    TTTResult {
                        avg_loss,
                        loss_improvement,
                        steps_performed: actual_steps,
                        adaptation_time_ms: time_used_ms,
                        skipped: false,
                        skip_reason: None,
                        avg_grad_norm,
                        max_grad_norm: max_grad_norm_val,
                        gradient_clipped: any_clipped,
                        tokens_used,
                        tokens_provided,
                        was_truncated,
                        initial_perplexity,
                        final_perplexity,
                        recommendation,
                        gated_steps,
                        pending: true, // Awaiting client commit/rollback
                        time_used_ms,
                        time_budget_ms,
                        rank_utilization: rank_utils,
                        gated_layers: gated_layer_keys,
                        rank_signals,
                    },
                    pre_snapshot,
                ))
            }
            Ok(Err(e)) => {
                // Gradient loop returned an error — restore delta, momentum, and SSM states
                tracing::warn!("TTT adapt_tenant gradient loop error, restoring delta: {}", e);
                let _ = delta.load_state_dict(&pre_snapshot);
                super::muon::restore_muon_states(&mut delta.muon_states, &muon_snapshot);
                delta.zero_grad();
                engine.restore_ssm_states(ssm_snapshot);
                Err(e)
            }
            Err(panic_info) => {
                // Panic during gradient loop — restore delta, momentum, and SSM states.
                // A panic mid-backward/step leaves stale gradient accumulation and
                // corrupted Muon momentum buffers.
                tracing::error!("TTT adapt_tenant panicked during gradient loop, restoring delta");
                let _ = delta.load_state_dict(&pre_snapshot);
                super::muon::restore_muon_states(&mut delta.muon_states, &muon_snapshot);
                delta.zero_grad();
                engine.restore_ssm_states(ssm_snapshot);
                let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                    (*s).to_owned()
                } else {
                    "unknown panic".to_owned()
                };
                Err(anyhow::anyhow!("TTT adaptation panicked: {}", msg))
            }
        }
    }

    /// Inner gradient loop for adapt_tenant (extracted for catch_unwind).
    /// Receives the pre-computed `initial_loss_tensor` from the caller to avoid
    /// a redundant forward pass.
    #[allow(clippy::type_complexity)]
    fn adapt_tenant_inner(
        &self,
        engine: &TorchEngine,
        delta: &mut TenantDelta,
        tokens: &[u32],
        gated_steps: usize,
        lr: f64,
        start: &Instant,
        budget: Duration,
        initial_loss_tensor: Tensor,
    ) -> Result<(usize, Vec<f32>, Vec<f32>, bool, f32, f32, HashMap<String, f64>, Vec<String>)> {
        let initial_loss = initial_loss_tensor.double_value(&[]) as f32;
        let mut losses = vec![initial_loss];
        let mut grad_norms = Vec::with_capacity(gated_steps);
        let mut any_clipped = false;

        // Diagnostic: log autograd state before first backward pass
        if delta.accumulated_steps == 0 {
            let vars = delta.vs.trainable_variables();
            let any_requires_grad = vars.iter().any(tch::Tensor::requires_grad);
            tracing::info!(
                "[TTT] Pre-backward check: loss.requires_grad={}, vars_require_grad={}, num_vars={}",
                initial_loss_tensor.requires_grad(), any_requires_grad, vars.len()
            );
        }

        // First step (always runs at least one)
        let (gn, clipped) = self.ttt_step(&initial_loss_tensor, delta, Some(lr))?;
        grad_norms.push(gn as f32);
        if clipped {
            any_clipped = true;
        }
        let mut actual_steps = 1;

        // Diagnostic: log gradient state after first backward pass
        if delta.accumulated_steps == 0 {
            let variables = delta.vs.trainable_variables();
            let defined_count = variables.iter().filter(|v| v.grad().defined()).count();
            tracing::info!(
                "[TTT] Post-backward: {}/{} vars have defined grad, grad_norm={:.6}",
                defined_count, variables.len(), gn
            );
        }

        // Per-layer gradient gating: after warmup, freeze low-signal layers
        // to save backward FLOPs and prevent momentum drift
        let gated_layer_keys = if self.gradient_gating.enabled && gated_steps > self.gradient_gating.warmup_steps {
            let grad_norms_map = compute_per_layer_grad_norms(&delta.vs);
            let gated = identify_gated_layers(&grad_norms_map, self.gradient_gating.min_grad_norm);
            if !gated.is_empty() {
                let _guard = tch::no_grad_guard();
                for (name, var) in delta.vs.variables() {
                    if let Some(key) = parse_varstore_key_to_delta_key(&name) {
                        if gated.contains(&key) {
                            let _ = var.set_requires_grad(false);
                        }
                    }
                }
                tracing::debug!(
                    gated_count = gated.len(),
                    "Gradient gating: froze low-signal layers"
                );
            }
            gated
        } else {
            Vec::new()
        };

        // Remaining steps with time budget check BEFORE each step
        for _ in 1..gated_steps {
            if start.elapsed() >= budget {
                tracing::warn!(
                    "TTT: time budget exhausted after {}/{} steps ({:.0}ms >= {}ms)",
                    actual_steps, gated_steps,
                    start.elapsed().as_millis(), budget.as_millis()
                );
                break;
            }

            let loss = self.compute_ntp_loss_with_delta(engine, delta, tokens)?;
            losses.push(loss.double_value(&[]) as f32);

            let (gn, clipped) = self.ttt_step(&loss, delta, Some(lr))?;
            grad_norms.push(gn as f32);
            if clipped {
                any_clipped = true;
            }
            actual_steps += 1;

            // Delta rank utilization monitoring every 10 cumulative steps (Phase 3e).
            // Uses delta.accumulated_steps (cross-call total) so monitoring fires even
            // for short TTT runs (gated_steps < 10). SVD on [rank×rank] is ~μs.
            if (delta.accumulated_steps + actual_steps as u64).is_multiple_of(10) {
                for (key, a) in &delta.lora_a {
                    if let Some(b) = delta.lora_b.get(key) {
                        // Narrow to effective rank so utilization reflects active subspace only
                        let eff_rank = delta.effective_ranks.get(key).copied()
                            .unwrap_or_else(|| a.size()[0] as usize) as i64;
                        let a_eff = a.narrow(0, 0, eff_rank);
                        let b_eff = b.narrow(1, 0, eff_rank);
                        let util = crate::runtime::ttn_profile::delta_rank_utilization(&a_eff, &b_eff);
                        tracing::debug!(key = %key, utilization = %util, "Delta rank utilization");
                    }
                }
            }
        }

        // Compute final loss for perplexity
        let final_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, tokens)?;
        let final_loss = final_loss_tensor.double_value(&[]) as f32;
        let final_perplexity = final_loss.exp();

        // Collect final rank utilization for all keys (narrowed to effective rank)
        let rank_utils: HashMap<String, f64> = {
            let _guard = tch::no_grad_guard();
            delta
                .lora_a
                .iter()
                .filter_map(|(key, a)| {
                    delta.lora_b.get(key).map(|b| {
                        let eff_rank = delta.effective_ranks.get(key).copied()
                            .unwrap_or_else(|| a.size()[0] as usize) as i64;
                        let a_eff = a.narrow(0, 0, eff_rank);
                        let b_eff = b.narrow(1, 0, eff_rank);
                        (key.clone(), crate::runtime::ttn_profile::delta_rank_utilization(&a_eff, &b_eff))
                    })
                })
                .collect()
        };

        // Restore requires_grad on all vars (undo gradient gating for next TTT call)
        if !gated_layer_keys.is_empty() {
            let _guard = tch::no_grad_guard();
            for (_, var) in delta.vs.variables() {
                if !var.requires_grad() {
                    let _ = var.set_requires_grad(true);
                }
            }
        }

        Ok((actual_steps, losses, grad_norms, any_clipped, final_loss, final_perplexity, rank_utils, gated_layer_keys))
    }

    /// Compute NTP loss with a tenant delta applied
    ///
    /// Runs a forward pass that injects the delta's A/B matrices after q_proj/v_proj
    /// projections inside each attention layer. This creates a differentiable path
    /// from the loss back to the delta parameters, enabling gradient-based training.
    fn compute_ntp_loss_with_delta(
        &self,
        engine: &TorchEngine,
        delta: &TenantDelta,
        tokens: &[u32],
    ) -> Result<Tensor> {
        // Defensive: ensure GradMode is enabled for training regardless of leaked state.
        // If a previous inference panic inside tch::no_grad leaked disabled GradMode,
        // this ensures gradients still flow through delta's A/B matrices.
        tch::with_grad(|| {
            let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();

            let input_ids = Tensor::from_slice(&tokens_i64)
                .to_device(self.device)
                .unsqueeze(0); // [1, seq_len]

            // Forward pass with delta injection — gradients flow through delta's A/B matrices
            let logits = engine.forward_with_delta(&input_ids, delta)?;

            let seq_len = tokens.len() as i64;
            let vocab_size = logits.size()[2];

            // Shift for next-token prediction
            let pred_logits = logits
                .narrow(1, 0, seq_len - 1)
                .reshape([-1, vocab_size]);

            let target_ids = input_ids.narrow(1, 1, seq_len - 1).reshape([-1]);

            let loss = pred_logits.cross_entropy_loss::<Tensor>(
                &target_ids,
                None,
                tch::Reduction::Mean,
                -100,
                0.0,
            );

            Ok(loss)
        })
    }

    /// Run pure training steps on a tenant delta without generation
    ///
    /// Used by the `trainStep` API endpoint for explicit training.
    /// Steps are clamped to `max_gradient_steps` and time-budgeted.
    pub fn train_step(
        &self,
        engine: &TorchEngine,
        delta: &mut TenantDelta,
        input_tokens: &[u32],
        gradient_steps: usize,
        learning_rate: Option<f64>,
        max_adaptation_ms: Option<u64>,
    ) -> Result<TTTResult> {
        let start = Instant::now();

        let tokens_provided = input_tokens.len();
        let was_truncated = tokens_provided > self.config.max_ttt_context;

        let tokens: Vec<u32> = if was_truncated {
            input_tokens[input_tokens.len() - self.config.max_ttt_context..].to_vec()
        } else {
            input_tokens.to_vec()
        };
        let tokens_used = tokens.len();

        if tokens_used < 2 {
            return Ok(TTTResult::skipped("Input too short for NTP loss"));
        }

        // Clamp steps to hard ceiling
        let clamped_steps = gradient_steps.min(self.config.max_gradient_steps);
        let time_budget_ms = max_adaptation_ms
            .unwrap_or(self.config.max_adaptation_ms);
        let budget = Duration::from_millis(time_budget_ms);

        let lr = learning_rate.unwrap_or(delta.learning_rate);

        // Snapshot for panic recovery
        let pre_snapshot = delta.extract_state_dict();
        let muon_snapshot = super::muon::snapshot_muon_states(&delta.muon_states);

        // Snapshot SSM states before training loop (prevents training data from polluting inference state).
        let ssm_snapshot = engine.snapshot_ssm_states();

        let loop_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.train_step_inner(engine, delta, &tokens, clamped_steps, lr, &start, budget)
        }));

        match loop_result {
            Ok(Ok(inner)) => {
                let (actual_steps, losses, grad_norms, any_clipped, final_loss, final_perplexity, initial_loss, initial_perplexity) = inner;

                let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;
                let loss_improvement = initial_loss - final_loss;
                let avg_grad_norm = if !grad_norms.is_empty() {
                    grad_norms.iter().sum::<f32>() / grad_norms.len() as f32
                } else {
                    0.0
                };
                let max_grad_norm_val = grad_norms.iter().copied().fold(0.0f32, f32::max);

                let recommendation = self.compute_recommendation(
                    loss_improvement,
                    any_clipped,
                    initial_perplexity,
                    final_perplexity,
                );

                // Update delta accumulation stats
                delta.accumulated_steps += actual_steps as u64;
                delta.request_count += 1;
                let n = delta.request_count as f64;
                delta.avg_loss_improvement =
                    delta.avg_loss_improvement * ((n - 1.0) / n) + loss_improvement as f64 / n;

                let time_used_ms = start.elapsed().as_millis() as u64;

                // Collect rank utilization (narrowed to effective rank)
                let rank_utils: HashMap<String, f64> = {
                    let _guard = tch::no_grad_guard();
                    delta
                        .lora_a
                        .iter()
                        .filter_map(|(key, a)| {
                            delta.lora_b.get(key).map(|b| {
                                let eff_rank = delta.effective_ranks.get(key).copied()
                                    .unwrap_or_else(|| a.size()[0] as usize) as i64;
                                let a_eff = a.narrow(0, 0, eff_rank);
                                let b_eff = b.narrow(1, 0, eff_rank);
                                (key.clone(), crate::runtime::ttn_profile::delta_rank_utilization(&a_eff, &b_eff))
                            })
                        })
                        .collect()
                };

                // Feed rank oracle (same logic as adapt_tenant)
                let rank_signals = if let Some(ref mut oracle) = delta.rank_oracle {
                    oracle.observe(&rank_utils);
                    if oracle.should_evaluate() {
                        let signals = oracle.recommend();
                        if oracle.config.auto_adapt {
                            use crate::runtime::ttn_profile::RankSignal;
                            for (key, signal) in &signals {
                                if let Some(current) = delta.effective_rank(key) {
                                    match signal {
                                        RankSignal::Decrease => {
                                            let new_rank = (current / 2).max(1);
                                            delta.set_effective_rank(key, new_rank);
                                            tracing::info!(key = %key, from = current, to = new_rank, "Rank oracle: decreased");
                                        }
                                        RankSignal::Increase => {
                                            let new_rank = current * 2;
                                            delta.set_effective_rank(key, new_rank);
                                            let actual = delta.effective_rank(key).unwrap_or(new_rank);
                                            tracing::info!(key = %key, from = current, to = actual, "Rank oracle: increased");
                                        }
                                        RankSignal::Hold => {}
                                    }
                                }
                            }
                        }
                        signals
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                };

                // Restore SSM states so training data does not pollute inference recurrent state.
                engine.restore_ssm_states(ssm_snapshot);

                Ok(TTTResult {
                    avg_loss,
                    loss_improvement,
                    steps_performed: actual_steps,
                    adaptation_time_ms: time_used_ms,
                    skipped: false,
                    skip_reason: None,
                    avg_grad_norm,
                    max_grad_norm: max_grad_norm_val,
                    gradient_clipped: any_clipped,
                    tokens_used,
                    tokens_provided,
                    was_truncated,
                    initial_perplexity,
                    final_perplexity,
                    recommendation,
                    gated_steps: clamped_steps,
                    pending: false, // trainStep commits immediately based on auto_commit
                    time_used_ms,
                    time_budget_ms,
                    rank_utilization: rank_utils,
                    gated_layers: Vec::new(),
                    rank_signals,
                })
            }
            Ok(Err(e)) => {
                tracing::warn!("TTT train_step gradient loop error, restoring delta: {}", e);
                let _ = delta.load_state_dict(&pre_snapshot);
                super::muon::restore_muon_states(&mut delta.muon_states, &muon_snapshot);
                delta.zero_grad();
                engine.restore_ssm_states(ssm_snapshot);
                Err(e)
            }
            Err(panic_info) => {
                tracing::error!("TTT train_step panicked during gradient loop, restoring delta");
                let _ = delta.load_state_dict(&pre_snapshot);
                super::muon::restore_muon_states(&mut delta.muon_states, &muon_snapshot);
                delta.zero_grad();
                engine.restore_ssm_states(ssm_snapshot);
                let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                    (*s).to_owned()
                } else {
                    "unknown panic".to_owned()
                };
                Err(anyhow::anyhow!("TTT training panicked: {}", msg))
            }
        }
    }

    /// Inner gradient loop for train_step (extracted for catch_unwind)
    #[allow(clippy::type_complexity)]
    fn train_step_inner(
        &self,
        engine: &TorchEngine,
        delta: &mut TenantDelta,
        tokens: &[u32],
        clamped_steps: usize,
        lr: f64,
        start: &Instant,
        budget: Duration,
    ) -> Result<(usize, Vec<f32>, Vec<f32>, bool, f32, f32, f32, f32)> {
        // Compute initial loss
        let initial_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, tokens)?;
        let initial_loss = initial_loss_tensor.double_value(&[]) as f32;
        let initial_perplexity = initial_loss.exp();

        let mut losses = vec![initial_loss];
        let mut grad_norms = Vec::with_capacity(clamped_steps);
        let mut any_clipped = false;

        // First step (always runs)
        let (gn, clipped) = self.ttt_step(&initial_loss_tensor, delta, Some(lr))?;
        grad_norms.push(gn as f32);
        if clipped {
            any_clipped = true;
        }
        let mut actual_steps = 1;

        // Remaining steps with time budget check
        for _ in 1..clamped_steps {
            if start.elapsed() >= budget {
                tracing::warn!(
                    "TTT train_step: time budget exhausted after {}/{} steps ({:.0}ms >= {}ms)",
                    actual_steps, clamped_steps,
                    start.elapsed().as_millis(), budget.as_millis()
                );
                break;
            }

            let loss = self.compute_ntp_loss_with_delta(engine, delta, tokens)?;
            losses.push(loss.double_value(&[]) as f32);
            let (gn, clipped) = self.ttt_step(&loss, delta, Some(lr))?;
            grad_norms.push(gn as f32);
            if clipped {
                any_clipped = true;
            }
            actual_steps += 1;
        }

        // Final loss
        let final_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, tokens)?;
        let final_loss = final_loss_tensor.double_value(&[]) as f32;
        let final_perplexity = final_loss.exp();

        Ok((actual_steps, losses, grad_norms, any_clipped, final_loss, final_perplexity, initial_loss, initial_perplexity))
    }
}

// Thread safety - TestTimeTrainer is Send + Sync because it only holds
// config data and device info, not mutable state
unsafe impl Send for TestTimeTrainer {}
unsafe impl Sync for TestTimeTrainer {}

/// Trait for TTT verifiers (future extension)
///
/// This trait allows plugging in different verification strategies:
/// - Cross-model verification (use another model to verify)
/// - Cycle consistency (for multimodal)
/// - Execution verification (for code)
#[async_trait::async_trait]
pub trait TTTVerifier: Send + Sync {
    /// Verify that TTT adaptation is beneficial
    ///
    /// Returns true if adaptation should be accepted
    async fn verify(&self, context: &TTTContext, engine: &TorchEngine) -> Result<bool>;
}

/// No-op verifier that always accepts adaptation
pub struct NoOpVerifier;

#[async_trait::async_trait]
impl TTTVerifier for NoOpVerifier {
    async fn verify(&self, _context: &TTTContext, _engine: &TorchEngine) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;

    #[test]
    fn test_ttt_config_default() {
        let config = TTTConfig::default();
        assert!((config.learning_rate - 3e-4).abs() < 1e-10);
        assert_eq!(config.gradient_steps, 3);
        assert_eq!(config.min_input_length, 32);
        assert_eq!(config.max_ttt_context, 512);
        assert!(config.enabled);
        assert_eq!(config.max_adaptation_ms, 20_000);
        assert_eq!(config.max_gradient_steps, 50);
    }

    #[test]
    fn test_ttt_result_skipped() {
        let result = TTTResult::skipped("test reason");
        assert!(result.skipped);
        assert_eq!(result.skip_reason, Some("test reason".to_owned()));
        assert_eq!(result.steps_performed, 0);
    }

    #[test]
    fn test_ttt_config_serde() {
        let json = r#"{
            "learning_rate": 0.001,
            "gradient_steps": 5,
            "enabled": false
        }"#;

        let config: TTTConfig = serde_json::from_str(json).unwrap();
        assert!((config.learning_rate - 0.001).abs() < 1e-10);
        assert_eq!(config.gradient_steps, 5);
        assert!(!config.enabled);
        // Defaults should apply for missing fields
        assert_eq!(config.min_input_length, 32);
    }

    #[test]
    fn test_gradient_gating_config_defaults() {
        let config = GradientGatingConfig::default();
        assert!(config.enabled);
        assert!(config.min_grad_norm > 0.0);
        assert!(config.warmup_steps > 0);
    }

    #[test]
    fn test_identify_gated_layers() {
        let mut norms = HashMap::new();
        norms.insert("0.q_proj".to_owned(), 1e-3); // above threshold
        norms.insert("0.v_proj".to_owned(), 1e-8); // below threshold
        norms.insert("1.q_proj".to_owned(), 5e-6); // at threshold
        let gated = identify_gated_layers(&norms, 1e-5);
        assert!(!gated.contains(&"0.q_proj".to_owned())); // not gated (high norm)
        assert!(gated.contains(&"0.v_proj".to_owned())); // gated (low norm)
        assert!(gated.contains(&"1.q_proj".to_owned())); // gated (at/below threshold)
    }

    #[test]
    fn test_gradient_gating_with_real_backward() {
        use tch::{Kind, nn::VarStore};
        let vs = VarStore::new(Device::Cpu);
        let root = vs.root();
        let a = root
            .sub("layer_0")
            .sub("q_proj")
            .kaiming_uniform("lora_a", &[4, 8]);
        let b = root
            .sub("layer_0")
            .sub("q_proj")
            .zeros("lora_b", &[8, 4]);
        let x = Tensor::randn([2, 8], (Kind::Float, Device::Cpu));

        // Forward + backward with requires_grad=true
        let out = x.matmul(&a.tr()).matmul(&b.tr());
        let loss = out.sum(Kind::Float);
        loss.backward();
        assert!(a.grad().defined());

        // Zero grads, disable requires_grad, forward + backward
        for (_, v) in vs.variables() {
            if v.grad().defined() {
                let _ = v.grad().zero_();
            }
        }
        let _ = a.set_requires_grad(false);
        let out2 = x.matmul(&a.tr()).matmul(&b.tr());
        let loss2 = out2.sum(Kind::Float);
        loss2.backward();
        // a should have NO gradient (frozen)
        assert!(
            !a.grad().defined() || a.grad().abs().sum(Kind::Double).double_value(&[]) == 0.0
        );
        // b should still have gradient
        assert!(b.grad().defined());

        // Re-enable
        let _ = a.set_requires_grad(true);
    }

    #[test]
    fn test_parse_varstore_key() {
        assert_eq!(
            parse_varstore_key_to_delta_key("layer_0.q_proj.lora_a"),
            Some("0.q_proj".to_owned())
        );
        assert_eq!(
            parse_varstore_key_to_delta_key("layer_23.v_proj.lora_b"),
            Some("23.v_proj".to_owned())
        );
        assert_eq!(parse_varstore_key_to_delta_key("invalid_key"), None);
    }

    #[test]
    fn test_rank_oracle_config_defaults() {
        let config = RankOracleConfig::default();
        assert!(config.adaptation_interval > 0);
        assert!(!config.auto_adapt); // conservative default
        assert!(!config.rank_levels.is_empty());
    }

    #[test]
    fn test_rank_oracle_produces_signals() {
        use crate::runtime::ttn_profile::RankSignal;
        let mut oracle = RankOracle::new(RankOracleConfig::default());
        // Simulate: feed low utilization for 10 steps
        for _ in 0..10 {
            let mut utils = HashMap::new();
            utils.insert("0.q_proj".to_owned(), 0.1);
            utils.insert("0.v_proj".to_owned(), 0.9);
            oracle.observe(&utils);
        }
        let signals = oracle.recommend();
        assert_eq!(signals.get("0.q_proj"), Some(&RankSignal::Decrease));
        assert_eq!(signals.get("0.v_proj"), Some(&RankSignal::Increase));
    }

    #[test]
    fn test_rank_oracle_should_evaluate() {
        let mut oracle = RankOracle::new(RankOracleConfig {
            adaptation_interval: 3,
            ..Default::default()
        });
        let utils: HashMap<String, f64> = HashMap::new();
        oracle.observe(&utils); // 1
        assert!(!oracle.should_evaluate());
        oracle.observe(&utils); // 2
        assert!(!oracle.should_evaluate());
        oracle.observe(&utils); // 3
        assert!(oracle.should_evaluate());
    }
}
