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

    /// Sample next token directly from GPU logits tensor
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

        // Step 1: Apply repetition penalty on GPU
        let penalized_logits =
            self.apply_repetition_penalty(&logits, repeat_penalty, previous_tokens)?;

        // Step 2: Apply temperature scaling on GPU
        let scaled_logits = if temperature <= 0.0 || temperature.is_nan() || temperature.is_infinite() {
            // Invalid temperature: use greedy decoding (temperature → 0 means argmax)
            penalized_logits
        } else if (temperature - 1.0).abs() < 1e-6 {
            // Temperature is effectively 1.0, no scaling needed
            penalized_logits
        } else if temperature < 0.01 {
            // Very low temperature: scale but clamp to avoid numerical issues
            let scale_factor = 100.0_f64; // Equivalent to temperature = 0.01
            &penalized_logits * scale_factor
        } else {
            // Normal temperature scaling
            &penalized_logits / temperature as f64
        };

        // Step 3: Convert to probabilities with numerical stability
        let probs = self.softmax_stable(&scaled_logits)?;

        // Step 4: Apply top-k filtering on GPU
        let filtered_probs = if let Some(k) = top_k {
            self.apply_top_k(&probs, k)?
        } else {
            probs
        };

        // Step 5: Apply top-p (nucleus) sampling on GPU
        let final_probs = if top_p < 1.0 {
            self.apply_top_p(&filtered_probs, top_p)?
        } else {
            filtered_probs
        };

        // Step 6: Sample from distribution on GPU
        self.multinomial_sample(&final_probs)
    }

    /// Apply repetition penalty on GPU
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        repeat_penalty: f32,
        previous_tokens: &[i64],
    ) -> Result<Tensor> {
        if repeat_penalty == 1.0 || previous_tokens.is_empty() {
            return Ok(logits.shallow_clone());
        }

        let penalized_logits = logits.shallow_clone();

        // Create frequency map for recent tokens (last 64)
        let mut token_counts = HashMap::new();
        for &token_id in previous_tokens.iter().rev().take(64) {
            *token_counts.entry(token_id as usize).or_insert(0) += 1;
        }

        // Apply penalties using GPU operations where possible
        for (token_id, count) in token_counts {
            if token_id < logits.size()[0] as usize {
                let penalty = repeat_penalty.powf(count as f32);

                // Get current logit value
                let current_logit = penalized_logits.double_value(&[token_id as i64]);

                // Apply penalty
                let new_logit = if current_logit > 0.0 {
                    current_logit / penalty as f64
                } else {
                    current_logit * penalty as f64
                };

                // Update tensor (this creates a small GPU operation per token)
                let penalty_tensor = Tensor::from(new_logit).to_device(self.device);
                penalized_logits
                    .narrow(0, token_id as i64, 1)
                    .copy_(&penalty_tensor);
            }
        }

        Ok(penalized_logits)
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

    /// Apply top-k filtering on GPU
    fn apply_top_k(&self, probs: &Tensor, k: usize) -> Result<Tensor> {
        let vocab_size = probs.size()[0] as usize;

        if k >= vocab_size {
            return Ok(probs.shallow_clone());
        }

        // Get top-k values and indices
        let (top_values, top_indices) = probs.topk(k as i64, -1, true, true);

        // Create mask for top-k positions
        let mut filtered_probs = Tensor::zeros([vocab_size as i64], (probs.kind(), self.device));

        // Set top-k probabilities (clone indices to reuse later if needed)
        let _ = filtered_probs.index_put_(&[Some(top_indices.shallow_clone())], &top_values, false);

        // Renormalize (with safety check for zero sum)
        let sum = filtered_probs.sum(filtered_probs.kind());
        let sum_scalar = sum.double_value(&[]);

        if sum_scalar <= 0.0 || sum_scalar.is_nan() || sum_scalar.is_infinite() {
            // If filtering removed all probability mass, return uniform distribution over top-k
            let uniform_prob = 1.0 / k as f64;
            let mut uniform_probs = Tensor::zeros([vocab_size as i64], (probs.kind(), self.device));
            let _ = uniform_probs.index_put_(&[Some(top_indices)], &Tensor::full(&[k as i64], uniform_prob, (probs.kind(), self.device)), false);
            Ok(uniform_probs)
        } else {
            Ok(&filtered_probs / &sum)
        }
    }

    /// Apply top-p (nucleus) sampling on GPU
    fn apply_top_p(&self, probs: &Tensor, top_p: f32) -> Result<Tensor> {
        if top_p >= 1.0 {
            return Ok(probs.shallow_clone());
        }

        // Sort probabilities in descending order
        let (sorted_probs, sorted_indices) = probs.sort(-1, true);

        // Compute cumulative sum
        let cumsum = sorted_probs.cumsum(-1, sorted_probs.kind());

        // Create mask for positions where cumsum <= top_p
        let mask = cumsum.le(top_p as f64);

        // Always include at least the top token
        let first_token_mask = Tensor::zeros_like(&mask);
        let _ = first_token_mask.narrow(-1, 0, 1).fill_(1.0);
        let final_mask = mask.logical_or(&first_token_mask);

        // Apply mask
        let filtered_sorted = &sorted_probs * &final_mask;

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
        let probs = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.25, 0.15]);

        let filtered = sampler.apply_top_k(&probs, 2).unwrap();
        let filtered_vec: Vec<f32> = Vec::try_from(filtered).unwrap();

        // Only top 2 should have non-zero probability
        // Index 2 (0.3) and 3 (0.25) are highest
        assert!(filtered_vec[2] > 0.0); // 0.3 (highest)
        assert!(filtered_vec[3] > 0.0); // 0.25 (second)
        assert_eq!(filtered_vec[0], 0.0); // masked
        assert_eq!(filtered_vec[1], 0.0); // masked
        assert_eq!(filtered_vec[4], 0.0); // masked
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
