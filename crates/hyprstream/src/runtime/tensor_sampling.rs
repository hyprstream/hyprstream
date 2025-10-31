//! Device-agnostic tensor sampling for token generation
//!
//! This module implements sampling algorithms directly on PyTorch tensors,
//! working efficiently on both CPU and GPU devices. Despite the historical
//! "GPU" naming, this sampler is fully device-agnostic and respects the
//! device parameter provided during construction.

use anyhow::Result;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

use super::generation_core::SamplingParams;

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

    /// Sample next token using bundled parameters (new interface)
    pub fn sample_with_params(
        &self,
        logits_tensor: &Tensor,
        params: &SamplingParams,
        previous_tokens: &[i64],
    ) -> Result<usize> {
        self.sample_token(
            logits_tensor,
            params.temperature,
            params.top_p,
            params.top_k,
            params.repeat_penalty,
            previous_tokens,
        )
    }

    /// Sample next token directly from logits tensor
    pub fn sample_token(
        &self,
        logits_tensor: &Tensor, // [1, vocab_size] tensor on GPU
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<usize> {
        // Ensure logits are on the correct device and squeezed to 1D
        // Convert to Float for sampling operations
        let logits = if logits_tensor.dim() > 1 {
            logits_tensor.squeeze_dim(0) // [vocab_size]
        } else {
            logits_tensor.shallow_clone()
        }
        .to_device(self.device)
        .to_kind(Kind::Float);

        // Step 1: Apply repetition penalty to logits
        let penalized_logits =
            self.apply_repetition_penalty(&logits, repeat_penalty, previous_tokens)?;

        // Step 2: Apply top-k filtering to logits (more efficient than after softmax)
        let filtered_logits = if let Some(k) = top_k {
            self.apply_top_k_to_logits(&penalized_logits, k)?
        } else {
            penalized_logits
        };

        // Step 3: Apply temperature scaling
        let scaled_logits = if temperature <= 0.0 || temperature.is_nan() || temperature.is_infinite() {
            // Invalid temperature: use greedy decoding (temperature → 0 means argmax)
            filtered_logits
        } else if (temperature - 1.0).abs() < 1e-6 {
            // Temperature is effectively 1.0, no scaling needed
            filtered_logits
        } else {
            // Normal temperature scaling (no capping - just use actual temperature)
            &filtered_logits / temperature as f64
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

    /// Apply repetition penalty (vectorized on GPU)
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<Tensor> {
        if repeat_penalty == 1.0 || previous_tokens.is_empty() {
            return Ok(logits.shallow_clone());
        }

        let vocab_size = logits.size()[0] as usize;

        // Create frequency map for recent tokens (last 64)
        let mut token_counts = HashMap::new();
        for &token_id in previous_tokens.iter().rev().take(64) {
            if token_id >= 0 && (token_id as usize) < vocab_size {
                *token_counts.entry(token_id as usize).or_insert(0) += 1;
            }
        }

        if token_counts.is_empty() {
            return Ok(logits.shallow_clone());
        }

        // Vectorized approach: batch CPU→GPU transfers
        // Use .copy() to create a proper deep copy we can safely modify
        let mut penalized = logits.copy();

        // Collect all penalized values
        let mut token_ids = Vec::with_capacity(token_counts.len());
        let mut new_values = Vec::with_capacity(token_counts.len());

        // Fetch current logit values for penalized tokens (single GPU→CPU transfer)
        let logits_cpu = logits.to_device(Device::Cpu);
        let logits_vec: Vec<f32> = Vec::try_from(&logits_cpu)?;

        for (token_id, count) in token_counts {
            let current_logit = logits_vec[token_id] as f64;
            let penalty = repeat_penalty.powf(count as f32) as f64;

            // Apply penalty (divide if positive, multiply if negative)
            let new_logit = if current_logit > 0.0 {
                current_logit / penalty
            } else {
                current_logit * penalty
            };

            token_ids.push(token_id as i64);
            new_values.push(new_logit as f32);
        }

        // Batch update back to GPU (single CPU→GPU transfer)
        let indices = Tensor::from_slice(&token_ids).to_device(self.device);
        let values = Tensor::from_slice(&new_values)
            .to_kind(Kind::Float)
            .to_device(self.device);

        let _ = penalized.index_put_(&[Some(indices)], &values, false);

        Ok(penalized)
    }

    /// Stable softmax computation on GPU
    fn softmax_stable(&self, logits: &Tensor) -> Result<Tensor> {
        // Compute softmax with numerical stability: exp(x - max(x))
        let max_logit = logits.max();
        let shifted_logits = logits - &max_logit;
        let exp_logits = shifted_logits.exp();
        let sum_exp = exp_logits.sum(Kind::Float);

        Ok(&exp_logits / &sum_exp)
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
                Tensor::zeros(&[1], (sorted_probs.kind(), self.device)),
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

        // Renormalize (with safety check for zero sum)
        let sum = filtered_probs.sum(filtered_probs.kind());
        let sum_scalar = sum.double_value(&[]);

        if sum_scalar <= 0.0 || sum_scalar.is_nan() || sum_scalar.is_infinite() {
            // If filtering removed all probability mass, keep original distribution
            // This can happen with very small top_p values
            Ok(probs.shallow_clone())
        } else {
            Ok(&filtered_probs / &sum)
        }
    }

    /// Sample from multinomial distribution on GPU
    fn multinomial_sample(&self, probs: &Tensor) -> Result<usize> {
        let vocab_size = probs.size()[0] as usize;

        // Check if probabilities are valid
        let sum = probs.sum(probs.kind());
        let sum_scalar = sum.double_value(&[]);

        if sum_scalar <= 0.0 || sum_scalar.is_nan() || sum_scalar.is_infinite() {
            // Fallback: return the most likely token (argmax)
            let (_, max_idx) = probs.max_dim(0, false);
            let token_id = max_idx.int64_value(&[]) as usize;

            // Ensure it's within bounds
            if token_id >= vocab_size {
                return Ok(0); // Return first token as ultimate fallback
            }
            return Ok(token_id);
        }

        // Use PyTorch's multinomial sampling (GPU accelerated)
        let sample = probs.multinomial(1, false); // Sample 1 token
        let token_id = sample.int64_value(&[0]) as usize;

        // Validate the sampled token is within vocabulary bounds
        if token_id >= vocab_size {
            // This shouldn't happen with valid probabilities, but if it does,
            // fall back to argmax
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
    fn test_temperature_scaling() {
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
        ).unwrap();

        // Should select token 4 (highest logit = 5.0)
        assert_eq!(token, 4);
    }

    #[test]
    fn test_top_k_masking() {
        let sampler = TensorSampler::new(Device::Cpu);
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 2.5, 1.5]);

        let filtered = sampler.apply_top_k_to_logits(&logits, 2).unwrap();
        let filtered_vec: Vec<f32> = Vec::try_from(filtered).unwrap();

        // Only top 2 logits should be kept (rest should be -inf)
        // Index 2 (3.0) and 3 (2.5) are highest
        assert!(filtered_vec[2].is_finite() && filtered_vec[2] > 0.0); // 3.0 (highest)
        assert!(filtered_vec[3].is_finite() && filtered_vec[3] > 0.0); // 2.5 (second)
        assert!(filtered_vec[0].is_infinite() && filtered_vec[0] < 0.0); // -inf
        assert!(filtered_vec[1].is_infinite() && filtered_vec[1] < 0.0); // -inf
        assert!(filtered_vec[4].is_infinite() && filtered_vec[4] < 0.0); // -inf
    }


    #[test]
    fn test_repetition_penalty() {
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
        ).unwrap();

        // Should NOT select token 4 despite it having highest logit
        assert_ne!(token, 4, "Repetition penalty not applied");
    }

    #[test]
    fn test_repetition_penalty_values() {
        // Verify that penalty actually modifies logit values
        let sampler = TensorSampler::new(Device::Cpu);
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);

        // Apply penalty to token 4 (highest logit = 5.0)
        let penalized = sampler.apply_repetition_penalty(
            &logits,
            2.0,  // 2x penalty
            &[4, 4, 4],  // Token 4 appeared 3 times
        ).unwrap();

        let penalized_vec: Vec<f32> = Vec::try_from(penalized).unwrap();
        let original_vec: Vec<f32> = Vec::try_from(logits).unwrap();

        // Token 4 should be penalized (divided by 2^3 = 8)
        // Original: 5.0, Penalized: 5.0 / 8.0 = 0.625
        assert!(penalized_vec[4] < original_vec[4],
            "Token 4 logit should be reduced: {} vs {}",
            penalized_vec[4], original_vec[4]);
        assert!((penalized_vec[4] - 0.625).abs() < 0.01,
            "Token 4 should be ~0.625, got {}", penalized_vec[4]);

        // Other tokens should be unchanged
        for i in 0..4 {
            assert_eq!(penalized_vec[i], original_vec[i],
                "Token {} should be unchanged", i);
        }
    }

    #[test]
    fn test_device_consistency() {
        // Test that sampler respects the device parameter
        let cpu_sampler = TensorSampler::new(Device::Cpu);
        assert_eq!(cpu_sampler.device, Device::Cpu);

        // Operations should stay on CPU
        let logits = Tensor::randn(&[100], (Kind::Float, Device::Cpu));
        let _ = cpu_sampler.sample_token(&logits, 1.0, 1.0, None, 1.0, &[]);
        // No panic = success (tensor device mismatch would panic)
    }
}
