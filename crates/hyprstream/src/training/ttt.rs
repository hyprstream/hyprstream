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
use std::time::Instant;
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
    device: Device,
}

impl TestTimeTrainer {
    /// Create a new TTT trainer with the given configuration
    pub fn new(config: TTTConfig, device: Device) -> Self {
        Self { config, device }
    }

    /// Check if TTT is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration
    pub fn config(&self) -> &TTTConfig {
        &self.config
    }

    /// Perform a single gradient step on a tenant delta using AdamW
    ///
    /// Steps: backward → compute grad_norm → clip_grad_norm → optimizer.step → zero_grad
    pub fn ttt_step(
        &self,
        loss: &Tensor,
        delta: &mut TenantDelta,
        learning_rate: Option<f64>,
    ) -> Result<(f64, bool)> {
        if let Some(lr) = learning_rate {
            delta.optimizer.set_lr(lr);
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
            delta.optimizer.clip_grad_norm(self.config.max_grad_norm);
        }

        // AdamW step (handles weight decay internally)
        delta.optimizer.step();
        delta.optimizer.zero_grad();

        Ok((grad_norm, clipped))
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

        // Gate step count based on perplexity
        let gated_steps = self.gate_steps(initial_perplexity, overrides);

        if gated_steps == 0 {
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
                },
                pre_snapshot,
            ));
        }

        // Determine learning rate (respect override)
        let lr = overrides
            .learning_rate
            .map(|r| r as f64)
            .unwrap_or(delta.learning_rate);

        // SGD gradient loop
        let mut losses = vec![initial_loss];
        let mut grad_norms = Vec::with_capacity(gated_steps);
        let mut any_clipped = false;

        // First step uses the already-computed loss
        let (gn, clipped) = self.ttt_step(&initial_loss_tensor, delta, Some(lr))?;
        grad_norms.push(gn as f32);
        if clipped {
            any_clipped = true;
        }

        // Remaining steps
        for _ in 1..gated_steps {
            let loss = self.compute_ntp_loss_with_delta(engine, delta, &tokens)?;
            let loss_value = loss.double_value(&[]) as f32;
            losses.push(loss_value);

            let (gn, clipped) = self.ttt_step(&loss, delta, Some(lr))?;
            grad_norms.push(gn as f32);
            if clipped {
                any_clipped = true;
            }
        }

        // Compute final loss for perplexity
        let final_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, &tokens)?;
        let final_loss: f32 = final_loss_tensor.double_value(&[]) as f32;
        let final_perplexity = final_loss.exp();

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

        Ok((
            TTTResult {
                avg_loss,
                loss_improvement,
                steps_performed: gated_steps,
                adaptation_time_ms: start.elapsed().as_millis() as u64,
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
            },
            pre_snapshot,
        ))
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
    }

    /// Run pure training steps on a tenant delta without generation
    ///
    /// Used by the `trainStep` API endpoint for explicit training.
    pub fn train_step(
        &self,
        engine: &TorchEngine,
        delta: &mut TenantDelta,
        input_tokens: &[u32],
        gradient_steps: usize,
        learning_rate: Option<f64>,
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

        let lr = learning_rate.unwrap_or(delta.learning_rate);

        // Compute initial loss
        let initial_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, &tokens)?;
        let initial_loss = initial_loss_tensor.double_value(&[]) as f32;
        let initial_perplexity = initial_loss.exp();

        let mut losses = vec![initial_loss];
        let mut grad_norms = Vec::with_capacity(gradient_steps);
        let mut any_clipped = false;

        // First step
        let (gn, clipped) = self.ttt_step(&initial_loss_tensor, delta, Some(lr))?;
        grad_norms.push(gn as f32);
        if clipped {
            any_clipped = true;
        }

        // Remaining steps
        for _ in 1..gradient_steps {
            let loss = self.compute_ntp_loss_with_delta(engine, delta, &tokens)?;
            losses.push(loss.double_value(&[]) as f32);
            let (gn, clipped) = self.ttt_step(&loss, delta, Some(lr))?;
            grad_norms.push(gn as f32);
            if clipped {
                any_clipped = true;
            }
        }

        // Final loss
        let final_loss_tensor = self.compute_ntp_loss_with_delta(engine, delta, &tokens)?;
        let final_loss = final_loss_tensor.double_value(&[]) as f32;
        let final_perplexity = final_loss.exp();

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
        delta.accumulated_steps += gradient_steps as u64;
        delta.request_count += 1;
        // Running average of loss improvement
        let n = delta.request_count as f64;
        delta.avg_loss_improvement =
            delta.avg_loss_improvement * ((n - 1.0) / n) + loss_improvement as f64 / n;

        Ok(TTTResult {
            avg_loss,
            loss_improvement,
            steps_performed: gradient_steps,
            adaptation_time_ms: start.elapsed().as_millis() as u64,
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
            gated_steps: gradient_steps,
            pending: false, // trainStep commits immediately based on auto_commit
        })
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
}
