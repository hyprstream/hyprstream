//! Simple LoRA adapter merging
//! 
//! Provides basic merging strategies for combining multiple LoRA adapters.

use anyhow::{Result, anyhow};
use tch::{Tensor, Device};
use std::collections::HashMap;

use crate::lora::torch_adapter::PyTorchLoRA;

/// Simple merging strategies
#[derive(Clone, Debug)]
pub enum MergeStrategy {
    /// Average all adapter weights equally
    Average,
    
    /// Weighted average with specified weights (must sum to 1.0)
    WeightedAverage { weights: Vec<f64> },
}

/// Handles merging of LoRA adapters
pub struct LoRAMerger;

impl LoRAMerger {
    /// Merge multiple LoRA adapters into a single adapter
    pub fn merge_adapters(
        adapters: Vec<PyTorchLoRA>,
        strategy: MergeStrategy,
        device: Device,
    ) -> Result<PyTorchLoRA> {
        if adapters.is_empty() {
            return Err(anyhow!("Cannot merge zero adapters"));
        }
        
        // Validate all adapters have same config
        let first_config = adapters[0].config();
        for adapter in adapters.iter().skip(1) {
            if adapter.config().rank != first_config.rank {
                return Err(anyhow!("All adapters must have the same rank"));
            }
            if adapter.config().target_modules != first_config.target_modules {
                return Err(anyhow!("All adapters must target the same modules"));
            }
        }
        
        // Get weights based on strategy
        let merge_weights = match strategy {
            MergeStrategy::Average => {
                vec![1.0 / adapters.len() as f64; adapters.len()]
            }
            MergeStrategy::WeightedAverage { ref weights } => {
                if weights.len() != adapters.len() {
                    return Err(anyhow!(
                        "Number of weights ({}) must match number of adapters ({})",
                        weights.len(),
                        adapters.len()
                    ));
                }
                let sum: f64 = weights.iter().sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return Err(anyhow!("Weights must sum to 1.0, got {}", sum));
                }
                weights.clone()
            }
        };
        
        // Merge the weights
        let mut merged_weights = HashMap::new();
        
        for module in &first_config.target_modules {
            // Collect all weights for this module
            let mut lora_a_tensors = Vec::new();
            let mut lora_b_tensors = Vec::new();
            
            for adapter in &adapters {
                let weights = adapter.get_module_weights(module)?;
                lora_a_tensors.push(weights.0);
                lora_b_tensors.push(weights.1);
            }
            
            // Compute weighted average
            let merged_a = Self::weighted_average(&lora_a_tensors, &merge_weights, device)?;
            let merged_b = Self::weighted_average(&lora_b_tensors, &merge_weights, device)?;
            
            merged_weights.insert(module.clone(), (merged_a, merged_b));
        }
        
        // Create new adapter with merged weights
        let mut merged_adapter = PyTorchLoRA::new(first_config.clone(), device)?;
        merged_adapter.load_weights(merged_weights)?;
        
        Ok(merged_adapter)
    }
    
    /// Compute weighted average of tensors
    fn weighted_average(
        tensors: &[Tensor],
        weights: &[f64],
        device: Device,
    ) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(anyhow!("Cannot average zero tensors"));
        }
        
        let mut result = Tensor::zeros_like(&tensors[0]).to_device(device);
        
        for (tensor, &weight) in tensors.iter().zip(weights.iter()) {
            result = result + tensor.to_device(device) * weight;
        }
        
        Ok(result)
    }
    
    /// Merge LoRA weights into base model weights
    pub fn merge_into_base(
        base_weight: &Tensor,
        lora_a: &Tensor,
        lora_b: &Tensor,
        scaling: f64,
    ) -> Result<Tensor> {
        // LoRA formula: W' = W + (B @ A) * scaling
        let lora_weight = lora_b.matmul(lora_a) * scaling;
        Ok(base_weight + lora_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_average_weights() {
        let device = Device::Cpu;
        let t1 = Tensor::ones(&[2, 2], (tch::Kind::Float, device));
        let t2 = Tensor::ones(&[2, 2], (tch::Kind::Float, device)) * 3.0;
        
        let avg = LoRAMerger::weighted_average(
            &[t1, t2],
            &[0.5, 0.5],
            device
        ).unwrap();
        
        // Average of 1 and 3 is 2
        assert!((avg.mean(tch::Kind::Float).double_value(&[]) - 2.0).abs() < 1e-6);
    }
}