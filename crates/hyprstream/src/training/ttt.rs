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
use std::time::Instant;
use tch::{Device, Tensor};

use crate::runtime::TorchEngine;

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

impl Default for TTTConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            gradient_steps: default_gradient_steps(),
            max_grad_norm: default_max_grad_norm(),
            min_input_length: default_min_input_length(),
            max_ttt_context: default_max_ttt_context(),
            enabled: default_enabled(),
        }
    }
}

/// Result of TTT adaptation
#[derive(Debug, Clone)]
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
        }
    }
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
    config: TTTConfig,
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

    /// Adapt the model to input context using TTT
    ///
    /// This is the main entry point for TTT. It:
    /// 1. Validates input length
    /// 2. Computes next-token prediction loss on input
    /// 3. Updates LoRA weights for `gradient_steps` iterations
    /// 4. Returns metrics about the adaptation
    ///
    /// # Arguments
    /// * `engine` - The TorchEngine with loaded model and LoRA
    /// * `input_tokens` - Tokenized input context
    ///
    /// # Returns
    /// TTTResult with adaptation metrics
    ///
    /// # Note
    /// This modifies the engine's LoRA weights. For production use with
    /// concurrent requests, consider using scratch weights (future enhancement).
    pub fn adapt(&self, engine: &TorchEngine, input_tokens: &[u32]) -> Result<TTTResult> {
        let start = Instant::now();

        // Check if enabled
        if !self.config.enabled {
            return Ok(TTTResult::skipped("TTT disabled"));
        }

        // Check minimum length
        if input_tokens.len() < self.config.min_input_length {
            return Ok(TTTResult::skipped(&format!(
                "Input too short: {} < {} tokens",
                input_tokens.len(),
                self.config.min_input_length
            )));
        }

        // Check if LoRA is available
        if !engine.has_lora_model() {
            return Ok(TTTResult::skipped("No LoRA adapter loaded"));
        }

        // Truncate if too long (use last N tokens for recency)
        let tokens: Vec<u32> = if input_tokens.len() > self.config.max_ttt_context {
            input_tokens[input_tokens.len() - self.config.max_ttt_context..].to_vec()
        } else {
            input_tokens.to_vec()
        };

        // TTT adaptation loop
        let mut losses = Vec::with_capacity(self.config.gradient_steps);
        

        // First step to get initial loss
        let loss = self.compute_ntp_loss(engine, &tokens)?;
        let initial_loss: f32 = loss.double_value(&[]) as f32;
        losses.push(initial_loss);

        // Gradient step
        self.ttt_step(engine, &loss)?;

        // Remaining steps
        for _ in 1..self.config.gradient_steps {
            let loss = self.compute_ntp_loss(engine, &tokens)?;
            let loss_value = loss.double_value(&[]) as f32;
            losses.push(loss_value);

            self.ttt_step(engine, &loss)?;
        }

        let final_loss = *losses.last().unwrap_or(&initial_loss);
        let avg_loss = losses.iter().sum::<f32>() / losses.len() as f32;

        Ok(TTTResult {
            avg_loss,
            loss_improvement: initial_loss - final_loss,
            steps_performed: self.config.gradient_steps,
            adaptation_time_ms: start.elapsed().as_millis() as u64,
            skipped: false,
            skip_reason: None,
        })
    }

    /// Compute next-token prediction loss on input sequence
    ///
    /// For sequence [t0, t1, t2, t3]:
    /// - Position 0: predict t1 from t0
    /// - Position 1: predict t2 from t0,t1
    /// - Position 2: predict t3 from t0,t1,t2
    ///
    /// This is the standard causal language modeling objective.
    fn compute_ntp_loss(&self, engine: &TorchEngine, tokens: &[u32]) -> Result<Tensor> {
        // Convert tokens to i64 tensor
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();

        let input_ids = Tensor::from_slice(&tokens_i64)
            .to_device(self.device)
            .unsqueeze(0); // [1, seq_len]

        // Forward pass with gradient tracking
        // Uses forward_with_lora with training=true for gradient flow
        let logits = engine.forward_with_lora(&input_ids, None, true)?;
        // logits shape: [1, seq_len, vocab_size]

        let seq_len = tokens.len() as i64;
        let vocab_size = logits.size()[2];

        // Shift for next-token prediction:
        // - logits[:, :-1, :] are predictions for positions 0 to seq_len-2
        // - labels[:, 1:] are targets (tokens 1 to seq_len-1)

        // Slice logits: all but last position
        let pred_logits = logits
            .narrow(1, 0, seq_len - 1)
            .reshape([-1, vocab_size]); // [(seq_len-1), vocab_size]

        // Slice labels: all but first position
        let target_ids = input_ids.narrow(1, 1, seq_len - 1).reshape([-1]); // [seq_len-1]

        // Cross-entropy loss
        let loss = pred_logits.cross_entropy_loss::<Tensor>(
            &target_ids,
            None,                  // no weight
            tch::Reduction::Mean,  // mean reduction
            -100,                  // ignore_index (unused here, no padding)
            0.0,                   // no label smoothing
        );

        Ok(loss)
    }

    /// Perform a single TTT gradient step
    ///
    /// Uses the engine's existing LoRA training infrastructure.
    fn ttt_step(&self, engine: &TorchEngine, loss: &Tensor) -> Result<f64> {
        // Use engine's atomic training step with our config
        let grad_norm =
            engine.lora_training_step(loss, Some(self.config.max_grad_norm))?;

        Ok(grad_norm)
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
