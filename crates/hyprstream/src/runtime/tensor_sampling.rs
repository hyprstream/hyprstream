//! Device-agnostic tensor sampling for token generation
//!
//! This module implements sampling algorithms directly on PyTorch tensors,
//! working efficiently on both CPU and GPU devices. Despite the historical
//! "GPU" naming, this sampler is fully device-agnostic and respects the
//! device parameter provided during construction.

use anyhow::Result;
use std::collections::HashSet;
use tch::{Device, Tensor};

/// Device-agnostic token sampler operating on tensors
///
/// Performs all sampling operations (temperature scaling, top-k filtering,
/// top-p nucleus sampling) directly on tensors, allowing the PyTorch backend
/// to optimize for the target device (CPU, CUDA, ROCm, Metal, etc.).
#[derive(Clone)]
pub struct TensorSampler {
    device: Device,  // Not dead code - used throughout the implementation
}

impl TensorSampler {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Set the random seed for deterministic sampling
    /// This enables reproducible token generation for debugging
    pub fn set_seed(seed: u64) {
        // Use PyTorch's manual_seed via tch-rs
        tch::manual_seed(seed as i64);
        tracing::info!("Set PyTorch random seed to {}", seed);
    }

    /// Sample next token directly from logits tensor
    pub fn sample_token(
        &self,
        logits_tensor: &Tensor,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<usize> {
        let logits = self.squeeze_logits(logits_tensor);
        let penalized_logits =
            self.apply_repetition_penalty(logits, repeat_penalty, previous_tokens)?;
        self.sample_from_logits(penalized_logits, temperature, top_p, top_k)
    }

    /// Sample a token with reduced repeat penalty for single-character tokens.
    ///
    /// Exempt tokens (digits, punctuation) receive `sqrt(penalty)` instead of
    /// the full penalty, preventing digit suppression in year generation while
    /// still discouraging runaway single-character repetition.
    pub fn sample_token_with_penalty_exemptions(
        &self,
        logits_tensor: &Tensor,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        previous_tokens: &[i64],
        exempt_tokens: &HashSet<i64>,
    ) -> Result<usize> {
        let logits = self.squeeze_logits(logits_tensor);
        let penalized_logits = self.apply_repetition_penalty_tiered(
            logits, repeat_penalty, previous_tokens, exempt_tokens,
        )?;
        self.sample_from_logits(penalized_logits, temperature, top_p, top_k)
    }

    // ── Shared pipeline helpers ──────────────────────────────────────────

    /// Extract last-token logits from a potentially batched tensor, yielding a 1-D vector.
    fn squeeze_logits(&self, logits_tensor: &Tensor) -> Tensor {
        let logits = if logits_tensor.dim() > 1 {
            let shape = logits_tensor.size();
            if shape.len() == 3 {
                // [batch, seq_len, vocab_size] → last position
                logits_tensor.select(1, shape[1] - 1).squeeze_dim(0)
            } else if shape.len() == 2 {
                logits_tensor.squeeze_dim(0)
            } else {
                logits_tensor.shallow_clone()
            }
        } else {
            logits_tensor.shallow_clone()
        }
        .to_device(self.device);

        if logits.dim() != 1 {
            tracing::warn!(
                "Logits not 1D after squeeze: shape={:?}",
                logits.size()
            );
        }
        logits
    }

    /// Run the sampling pipeline (greedy / top-k / temperature / softmax / top-p / multinomial)
    /// on already-penalized logits.
    fn sample_from_logits(
        &self,
        penalized_logits: Tensor,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    ) -> Result<usize> {
        // PERF: Greedy path - bypass softmax/multinomial when temperature is very low
        if temperature <= 0.01 || temperature.is_nan() || temperature.is_infinite() {
            let vocab_size = penalized_logits.size()[0] as usize;
            let (_, max_idx) = penalized_logits.max_dim(0, false);
            let token_id = max_idx.int64_value(&[]) as usize;
            if token_id >= vocab_size {
                tracing::error!("Greedy: token_id {} >= vocab_size {}", token_id, vocab_size);
                return Ok(0);
            }
            return Ok(token_id);
        }

        // Top-k filtering (more efficient before softmax)
        let filtered_logits = if let Some(k) = top_k {
            self.apply_top_k_to_logits(&penalized_logits, k)?
        } else {
            penalized_logits
        };

        // Temperature scaling
        let scaled_logits = if (temperature - 1.0).abs() < 1e-6 {
            filtered_logits
        } else {
            filtered_logits / (temperature as f64)
        };

        // Softmax → probabilities (always FP32 for numerical stability)
        let probs = self.softmax_stable(&scaled_logits)?;

        // Top-p (nucleus) sampling
        let final_probs = if top_p < 1.0 {
            self.apply_top_p(&probs, top_p)?
        } else {
            probs
        };

        self.multinomial_sample(&final_probs)
    }

    // ── Repetition penalty ──────────────────────────────────────────────

    /// Apply uniform repetition penalty to all tokens in the recent window.
    ///
    /// PERF: Takes ownership of logits to enable smart copy — only copies if
    /// the tensor is a view.
    fn apply_repetition_penalty(
        &self,
        logits: Tensor,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<Tensor> {
        if repeat_penalty <= 0.0 || (repeat_penalty - 1.0).abs() < 1e-6 || previous_tokens.is_empty() {
            return Ok(logits);
        }

        let vocab_size = logits.size()[0] as usize;
        let logits_kind = logits.kind();

        // Deduplicate tokens (penalty is applied once per token, not per occurrence)
        let mut seen = HashSet::new();
        let unique_ids: Vec<usize> = previous_tokens
            .iter()
            .rev()
            .filter_map(|&tid| {
                let uid = tid as usize;
                (tid >= 0 && uid < vocab_size && seen.insert(uid)).then_some(uid)
            })
            .collect();

        if unique_ids.is_empty() {
            return Ok(logits);
        }

        let mut result = logits.contiguous();
        self.apply_penalty_to_indices(&mut result, &unique_ids, repeat_penalty, logits_kind)?;
        Ok(result)
    }

    /// Apply tiered repetition penalty: full penalty for normal tokens,
    /// `sqrt(penalty)` for exempt tokens (single-character tokens like digits).
    ///
    /// This prevents digit suppression in year generation ("1917" → "209")
    /// while still discouraging runaway digit repetition ("1822222…").
    fn apply_repetition_penalty_tiered(
        &self,
        logits: Tensor,
        repeat_penalty: f32,
        previous_tokens: &[i64],
        exempt_tokens: &HashSet<i64>,
    ) -> Result<Tensor> {
        if repeat_penalty <= 0.0 || (repeat_penalty - 1.0).abs() < 1e-6 || previous_tokens.is_empty() {
            return Ok(logits);
        }

        let vocab_size = logits.size()[0] as usize;
        let logits_kind = logits.kind();

        // Partition tokens into full-penalty and reduced-penalty groups
        let mut full_penalty_ids = Vec::new();
        let mut reduced_penalty_ids = Vec::new();
        let mut seen = HashSet::new();

        for &token_id in previous_tokens.iter().rev() {
            let uid = token_id as usize;
            if token_id >= 0 && uid < vocab_size && seen.insert(uid) {
                if exempt_tokens.contains(&token_id) {
                    reduced_penalty_ids.push(uid);
                } else {
                    full_penalty_ids.push(uid);
                }
            }
        }

        if full_penalty_ids.is_empty() && reduced_penalty_ids.is_empty() {
            return Ok(logits);
        }

        let mut result = logits.contiguous();

        if !full_penalty_ids.is_empty() {
            self.apply_penalty_to_indices(&mut result, &full_penalty_ids, repeat_penalty, logits_kind)?;
        }

        if !reduced_penalty_ids.is_empty() {
            // sqrt(penalty) gives a much gentler nudge: e.g. 1.1 → 1.049
            let reduced = repeat_penalty.sqrt();
            self.apply_penalty_to_indices(&mut result, &reduced_penalty_ids, reduced, logits_kind)?;
        }

        Ok(result)
    }

    /// Apply a scalar penalty to specific token indices in a logits tensor.
    ///
    /// For positive logits the value is divided by `penalty`;
    /// for negative logits the value is multiplied by `penalty`.
    fn apply_penalty_to_indices(
        &self,
        result: &mut Tensor,
        token_ids: &[usize],
        penalty: f32,
        logits_kind: tch::Kind,
    ) -> Result<()> {
        let indices_vec: Vec<i64> = token_ids.iter().map(|&id| id as i64).collect();
        let indices = Tensor::from_slice(&indices_vec).to_device(self.device);
        let current_logits = result.index_select(0, &indices);

        let penalty_tensor = Tensor::from_slice(&[penalty])
            .to_device(self.device)
            .to_kind(logits_kind);
        let penalties = Tensor::ones([indices_vec.len() as i64], (logits_kind, self.device))
            * penalty_tensor;

        let penalized_positive = &current_logits / &penalties;
        let penalized_negative = &current_logits * &penalties;

        let positive_mask = current_logits.gt(0.0).to_kind(logits_kind);
        let negative_mask = Tensor::ones_like(&positive_mask) - &positive_mask;

        let new_values =
            &penalized_positive * &positive_mask + &penalized_negative * &negative_mask;

        let _ = result.index_put_(&[Some(indices)], &new_values, false);
        Ok(())
    }

    // ── Softmax / top-k / top-p / multinomial ───────────────────────────

    /// Softmax in FP32 for numerical stability.
    ///
    /// BF16 softmax over ~150K vocabulary causes significant precision loss
    /// (many small probabilities underflow to zero). This matches
    /// HuggingFace/vLLM standard practice.
    fn softmax_stable(&self, logits: &Tensor) -> Result<Tensor> {
        let logits_fp32 = if logits.kind() != tch::Kind::Float {
            logits.to_kind(tch::Kind::Float)
        } else {
            logits.shallow_clone()
        };
        // Keep FP32 for accurate multinomial sampling
        Ok(logits_fp32.softmax(-1, tch::Kind::Float))
    }

    /// Apply top-k filtering to logits (before softmax)
    /// More efficient than filtering probabilities after softmax
    fn apply_top_k_to_logits(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = logits.size()[0] as usize;

        if k >= vocab_size {
            return Ok(logits.shallow_clone());
        }

        // Get top-k values and indices
        let (top_values, top_indices) = logits.topk(k as i64, -1, true, true);

        // Create filtered logits (all -inf except top-k)
        let mut filtered_logits = Tensor::full(
            [vocab_size as i64],
            f64::NEG_INFINITY,
            (logits.kind(), self.device),
        );

        // Set top-k logits to their original values
        let _ = filtered_logits.index_put_(&[Some(top_indices)], &top_values, false);

        Ok(filtered_logits)
    }

    /// Apply top-p (nucleus) sampling
    /// Keeps tokens where cumulative probability BEFORE adding them is < top_p
    fn apply_top_p(&self, probs: &Tensor, top_p: f32) -> Result<Tensor> {
        if top_p <= 0.0 || top_p >= 1.0 {
            return Ok(probs.shallow_clone());
        }

        // Sort probabilities in descending order
        let (sorted_probs, sorted_indices) = probs.sort(-1, true);

        // Compute cumulative sum
        let cumsum = sorted_probs.cumsum(-1, sorted_probs.kind());

        // Standard nucleus sampling: keep token i if cumsum[i-1] < top_p
        // This means we include tokens BEFORE cumsum exceeds top_p
        // Shift cumsum right by 1 position (first position gets 0)
        let cumsum_size = cumsum.size()[0];

        let cumsum_shifted = Tensor::cat(
            &[
                Tensor::zeros([1], (sorted_probs.kind(), self.device)),
                cumsum.narrow(-1, 0, cumsum_size - 1),
            ],
            -1,
        );

        // Keep tokens where cumsum before adding them is < top_p
        let mask = cumsum_shifted.lt(top_p as f64);

        // Apply mask
        let filtered_sorted = &sorted_probs * &mask;

        // Scatter back to original order
        let mut filtered_probs = Tensor::zeros_like(probs);
        let _ = filtered_probs.scatter_(-1, &sorted_indices, &filtered_sorted);

        // Renormalize - top-p filtering always keeps at least one token (highest prob)
        // so sum > 0, avoiding the need for GPU-CPU sync to check
        let sum = filtered_probs.sum(filtered_probs.kind());
        Ok(&filtered_probs / &sum)
    }

    /// Sample from multinomial distribution on GPU
    /// Optimized to minimize GPU-CPU synchronization
    fn multinomial_sample(&self, probs: &Tensor) -> Result<usize> {
        let vocab_size = probs.size()[0] as usize;

        // Use PyTorch's multinomial sampling (GPU accelerated)
        // This is the primary path - avoid validation overhead in hot path
        let sample = probs.multinomial(1, false); // Sample 1 token
        let token_id = sample.int64_value(&[0]) as usize; // Single GPU sync point

        // Validate the sampled token is within vocabulary bounds
        if token_id >= vocab_size {
            // Rare error path - only log and do extra work here
            tracing::error!(
                "OUT OF BOUNDS! token_id={} >= vocab_size={}",
                token_id,
                vocab_size
            );
            // Fall back to argmax
            let (_, max_idx) = probs.max_dim(0, false);
            let safe_token_id = max_idx.int64_value(&[]) as usize;
            if safe_token_id < vocab_size {
                Ok(safe_token_id)
            } else {
                Ok(0) // Ultimate fallback to first token
            }
        } else {
            Ok(token_id)
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    fn create_test_logits() -> Tensor {
        // Create deterministic test logits
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0])
    }

    #[test]
    fn test_temperature_scaling() -> Result<()> {
        let sampler = TensorSampler::new(Device::Cpu);
        let logits = create_test_logits();

        // Very low temperature should approach greedy (highest logit wins)
        let token = sampler.sample_token(
            &logits,
            0.01,  // Very low temp
            1.0,
            None,
            1.0,
            &[],
        )?;

        // Should select token 4 (highest logit = 5.0)
        assert_eq!(token, 4);
        Ok(())
    }

    #[test]
    fn test_top_k_masking() -> Result<()> {
        let sampler = TensorSampler::new(Device::Cpu);
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 2.5, 1.5]);

        let filtered = sampler.apply_top_k_to_logits(&logits, 2)?;
        let filtered_vec: Vec<f32> = Vec::try_from(filtered)?;

        // Only top 2 logits should be kept (rest should be -inf)
        // Index 2 (3.0) and 3 (2.5) are highest
        assert!(filtered_vec[2].is_finite() && filtered_vec[2] > 0.0); // 3.0 (highest)
        assert!(filtered_vec[3].is_finite() && filtered_vec[3] > 0.0); // 2.5 (second)
        assert!(filtered_vec[0].is_infinite() && filtered_vec[0] < 0.0); // -inf
        assert!(filtered_vec[1].is_infinite() && filtered_vec[1] < 0.0); // -inf
        assert!(filtered_vec[4].is_infinite() && filtered_vec[4] < 0.0); // -inf
        Ok(())
    }


    #[test]
    fn test_repetition_penalty() -> Result<()> {
        let sampler = TensorSampler::new(Device::Cpu);
        let logits = create_test_logits();

        // With high repetition penalty on token 4
        let token = sampler.sample_token(
            &logits,
            0.01,  // Very low temp for deterministic
            1.0,
            None,
            10.0,  // High penalty
            &[4],  // Previous token was 4
        )?;

        // Should NOT select token 4 despite it having highest logit
        assert_ne!(token, 4, "Repetition penalty not applied");
        Ok(())
    }

    #[test]
    fn test_repetition_penalty_values() -> Result<()> {
        // Verify that penalty actually modifies logit values
        let sampler = TensorSampler::new(Device::Cpu);
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);

        // Save original values before penalty (shallow_clone shares storage, so save first)
        let original_vec: Vec<f32> = Vec::try_from(&logits)?;

        // Apply penalty to token 4 (highest logit = 5.0)
        let penalized = sampler.apply_repetition_penalty(
            logits,  // Pass ownership directly
            2.0,  // 2x penalty
            &[4, 4, 4],  // Token 4 appeared 3 times (but penalty is uniform, not exponential)
        )?;

        let penalized_vec: Vec<f32> = Vec::try_from(penalized)?;

        // Token 4 should be penalized with UNIFORM penalty (frequency ignored)
        // Original: 5.0, Penalized: 5.0 / 2.0 = 2.5
        // (NOT exponential: would be 5.0 / 2^3 = 0.625)
        assert!(penalized_vec[4] < original_vec[4],
            "Token 4 logit should be reduced: {} vs {}",
            penalized_vec[4], original_vec[4]);
        assert!((penalized_vec[4] - 2.5).abs() < 0.01,
            "Token 4 should be ~2.5 (uniform penalty), got {}", penalized_vec[4]);

        // Other tokens should be unchanged
        for i in 0..4 {
            assert_eq!(penalized_vec[i], original_vec[i],
                "Token {} should be unchanged", i);
        }
        Ok(())
    }

    #[test]
    fn test_device_consistency() {
        // Test that sampler respects the device parameter
        let cpu_sampler = TensorSampler::new(Device::Cpu);
        assert_eq!(cpu_sampler.device, Device::Cpu);

        // Operations should stay on CPU
        let logits = Tensor::randn([100], (Kind::Float, Device::Cpu));
        let _ = cpu_sampler.sample_token(&logits, 1.0, 1.0, None, 1.0, &[]);
        // No panic = success (tensor device mismatch would panic)
    }
}
