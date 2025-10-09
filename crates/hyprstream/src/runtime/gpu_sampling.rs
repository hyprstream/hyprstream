//! GPU-based sampling to eliminate CPU transfer bottlenecks
//!
//! This module implements sampling algorithms directly on GPU tensors,
//! avoiding expensive GPU→CPU transfers of large logits tensors.

use anyhow::Result;
use tch::{Device, Tensor, Kind};
use std::collections::HashMap;

/// GPU-based token sampler
#[derive(Clone)]
pub struct GpuSampler {
    device: Device,
}

impl GpuSampler {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Sample next token directly from GPU logits tensor
    pub fn sample_token(
        &self,
        logits_tensor: &Tensor,          // [1, vocab_size] tensor on GPU
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
        }.to_device(self.device).to_kind(Kind::Float);

        // Step 1: Apply repetition penalty on GPU
        let penalized_logits = self.apply_repetition_penalty(&logits, repeat_penalty, previous_tokens)?;

        // Step 2: Apply temperature scaling on GPU
        let scaled_logits = if temperature > 0.0 && temperature != 1.0 {
            &penalized_logits / temperature as f64
        } else {
            penalized_logits
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
                let _ = penalized_logits.narrow(0, token_id as i64, 1).copy_(&penalty_tensor);
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
        let mut filtered_probs = Tensor::zeros(&[vocab_size as i64], (probs.kind(), self.device));
        
        // Set top-k probabilities
        let _ = filtered_probs.index_put_(&[Some(top_indices)], &top_values, false);
        
        // Renormalize
        let sum = filtered_probs.sum(filtered_probs.kind());
        Ok(&filtered_probs / &sum)
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
        filtered_probs.scatter_(-1, &sorted_indices, &filtered_sorted);
        
        // Renormalize
        let sum = filtered_probs.sum(filtered_probs.kind());
        Ok(&filtered_probs / &sum)
    }

    /// Sample from multinomial distribution on GPU
    fn multinomial_sample(&self, probs: &Tensor) -> Result<usize> {
        // Use PyTorch's multinomial sampling (GPU accelerated)
        let sample = probs.multinomial(1, false); // Sample 1 token
        let token_id = sample.int64_value(&[0]) as usize;
        
        Ok(token_id)
    }
}