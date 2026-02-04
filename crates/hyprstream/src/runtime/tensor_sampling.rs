//! Device-agnostic tensor sampling for token generation
//!
//! This module implements sampling algorithms directly on PyTorch tensors,
//! working efficiently on both CPU and GPU devices. Despite the historical
//! "GPU" naming, this sampler is fully device-agnostic and respects the
//! device parameter provided during construction.

use anyhow::Result;
use std::collections::HashMap;
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
        logits_tensor: &Tensor, // [1, vocab_size] or [batch, seq_len, vocab_size] tensor on GPU
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<usize> {
        // Ensure logits are on the correct device and squeezed to 1D
        // Preserve native precision (BF16/FP16) for GPU performance
        // For multimodal models, logits might be [batch, seq_len, vocab_size]
        // We need to take the last token's logits: [-1, :] -> [vocab_size]
        let logits = if logits_tensor.dim() > 1 {
            // Get the shape
            let shape = logits_tensor.size();
            if shape.len() == 3 {
                // [batch, seq_len, vocab_size] -> take last position's logits
                logits_tensor.select(1, shape[1] - 1).squeeze_dim(0) // [vocab_size]
            } else if shape.len() == 2 {
                // [batch, vocab_size] or [seq_len, vocab_size] -> squeeze first dim
                logits_tensor.squeeze_dim(0) // [vocab_size]
            } else {
                logits_tensor.shallow_clone()
            }
        } else {
            logits_tensor.shallow_clone()
        }
        .to_device(self.device);

        // Validate logits shape (no GPU sync - just checks dimensions)
        if logits.dim() != 1 {
            tracing::warn!(
                "⚠️ Logits not 1D after squeeze: shape={:?}",
                logits.size()
            );
        }

        // NOTE: Debug stats removed to eliminate GPU-CPU sync overhead
        // Previous code extracted min/max/mean which forced 3 sync points per token

        // Step 1: Apply repetition penalty to logits
        // PERF: Pass ownership - enables smart copy in apply_repetition_penalty
        let penalized_logits =
            self.apply_repetition_penalty(logits, repeat_penalty, previous_tokens)?;

        // PERF: Greedy path - bypass softmax/multinomial when temperature is very low
        // This eliminates expensive GPU sync in multinomial sampling for deterministic generation
        if temperature <= 0.01 || temperature.is_nan() || temperature.is_infinite() {
            // Greedy decoding: just take argmax of logits (still has GPU sync, but faster than multinomial)
            let vocab_size = penalized_logits.size()[0] as usize;
            let (_, max_idx) = penalized_logits.max_dim(0, false);
            let token_id = max_idx.int64_value(&[]) as usize;

            if token_id >= vocab_size {
                tracing::error!("Greedy: token_id {} >= vocab_size {}", token_id, vocab_size);
                return Ok(0);
            }
            return Ok(token_id);
        }

        // Step 2: Apply top-k filtering to logits (more efficient than after softmax)
        let filtered_logits = if let Some(k) = top_k {
            self.apply_top_k_to_logits(&penalized_logits, k)?
        } else {
            penalized_logits
        };

        // Step 3: Apply temperature scaling
        // Use scalar division directly - tch-rs handles precision correctly with f64
        let scaled_logits = if (temperature - 1.0).abs() < 1e-6 {
            // Temperature is effectively 1.0, no scaling needed
            filtered_logits
        } else {
            // Scalar division avoids tensor allocation per token
            filtered_logits / (temperature as f64)
        };

        // Step 4: Convert to probabilities with numerical stability
        let probs = self.softmax_stable(&scaled_logits)?;

        // Step 5: Apply top-p (nucleus) sampling
        let final_probs = if top_p < 1.0 {
            self.apply_top_p(&probs, top_p)?
        } else {
            probs
        };

        // Step 6: Sample from distribution
        self.multinomial_sample(&final_probs)
    }

    /// Apply repetition penalty (fully GPU-accelerated, no CPU transfers)
    /// PERF: Takes ownership of logits to enable smart copy - only copies if tensor is a view
    fn apply_repetition_penalty(
        &self,
        logits: Tensor,  // Take ownership for in-place modification
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<Tensor> {
        // PERF: Early return if no penalty needed - no copy at all
        if (repeat_penalty - 1.0).abs() < 1e-6 || previous_tokens.is_empty() {
            return Ok(logits);  // Return owned tensor, no copy
        }

        let vocab_size = logits.size()[0] as usize;
        let logits_kind = logits.kind();

        // Build frequency map on CPU (small data, cheap operation)
        let mut token_counts = HashMap::new();
        for &token_id in previous_tokens.iter().rev() {
            if token_id >= 0 && (token_id as usize) < vocab_size {
                *token_counts.entry(token_id as usize).or_insert(0) += 1;
            }
        }

        if token_counts.is_empty() {
            return Ok(logits);  // Return owned tensor, no copy
        }

        // PERF: Smart copy - contiguous() is a no-op if tensor already owns its memory,
        // only copies if logits is a view (from select/squeeze). This replaces the
        // unconditional copy() which always copied 600KB.
        let mut result = logits.contiguous();

        // Prepare data for GPU operations
        let token_ids: Vec<i64> = token_counts.keys().map(|&id| id as i64).collect();
        let counts: Vec<f32> = token_counts.values().map(|&c| c as f32).collect();

        // === ALL OPERATIONS BELOW HAPPEN ON GPU ===

        // 1. Create index and count tensors on device
        let indices = Tensor::from_slice(&token_ids).to_device(self.device);
        let penalty_counts = Tensor::from_slice(&counts)
            .to_kind(logits_kind)  // Match logits precision (BF16)
            .to_device(self.device);

        // 2. Extract current logit values for penalized tokens (GPU gather)
        let current_logits = result.index_select(0, &indices);

        // 3. Apply penalty once per token (not exponential based on count)
        // Standard implementation: penalty is applied uniformly to any token that appeared,
        // regardless of how many times it appeared in the context window.
        // This prevents catastrophic penalty accumulation for frequently-used tokens (like digits).
        // FIX: Create penalty tensor with same precision as logits to prevent precision loss
        let penalty_tensor = Tensor::from_slice(&[repeat_penalty])
            .to_device(self.device)
            .to_kind(logits_kind);  // Match logits precision exactly
        let penalties = Tensor::ones_like(&penalty_counts) * penalty_tensor;

        // 4. Apply conditional penalty using GPU masking
        // If logit > 0: divide by penalty, else: multiply by penalty
        let penalized_positive = &current_logits / &penalties;
        let penalized_negative = &current_logits * &penalties;

        // Vectorized conditional select using mask multiplication
        // positive_mask: 1.0 where current_logits > 0, else 0.0
        let positive_mask = current_logits.gt(0.0).to_kind(logits_kind);
        let negative_mask = &Tensor::ones_like(&positive_mask) - &positive_mask;

        let new_values = &penalized_positive * &positive_mask + &penalized_negative * &negative_mask;

        // 5. Update in-place (safe because we own result via contiguous())
        let _ = result.index_put_(&[Some(indices)], &new_values, false);

        Ok(result)
    }

    /// Softmax computation using PyTorch's optimized fused kernel
    fn softmax_stable(&self, logits: &Tensor) -> Result<Tensor> {
        // Use PyTorch's native softmax - it's a single fused kernel that handles
        // numerical stability internally (subtracts max before exp)
        Ok(logits.softmax(-1, logits.kind()))
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
        if top_p >= 1.0 {
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
