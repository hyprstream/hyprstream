//! Common utilities and shared operations for LoRA implementations

use anyhow::{Result, anyhow};
use tch::{Tensor, Device, Kind};

/// Common matrix multiplication with scaling for LoRA
/// Computes: output = input @ A @ B * (alpha / rank)
pub fn lora_forward(
    input: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    alpha: f32,
    rank: usize,
    dropout: Option<f32>,
    training: bool,
) -> Result<Tensor> {
    // Validate dimensions
    let input_shape = input.size();
    let a_shape = lora_a.size();
    let b_shape = lora_b.size();
    
    if input_shape.len() < 2 {
        return Err(anyhow!("Input must be at least 2D"));
    }
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(anyhow!("LoRA matrices must be 2D"));
    }
    
    // Apply dropout if training
    let input_dropped = if training && dropout.is_some() && dropout.unwrap() > 0.0 {
        input.dropout(dropout.unwrap() as f64, training)
    } else {
        input.shallow_clone()
    };
    
    // Compute: input @ A @ B
    let intermediate = input_dropped.matmul(lora_a);
    let output = intermediate.matmul(lora_b);
    
    // Apply scaling: alpha / rank
    let scaling = alpha / rank as f32;
    Ok(output * scaling as f64)
}


/// Validate LoRA configuration parameters
pub fn validate_lora_config(
    rank: usize,
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    if rank == 0 || rank > in_features.min(out_features) {
        return Err(anyhow!(
            "Invalid rank {}: must be between 1 and min({}, {})",
            rank, in_features, out_features
        ));
    }
    
    if in_features == 0 || out_features == 0 {
        return Err(anyhow!("Feature dimensions must be positive"));
    }
    
    Ok(())
}

/// Common initialization strategies for LoRA weights
pub enum InitStrategy {
    Zeros,
    Random { std: f32 },
    Xavier,
    Kaiming { fan_in: usize },
}

pub fn initialize_lora_weights(
    shape: &[i64],
    strategy: InitStrategy,
    device: Device,
) -> Result<Tensor> {
    match strategy {
        InitStrategy::Zeros => {
            Ok(Tensor::zeros(shape, (Kind::Float, device)))
        }
        InitStrategy::Random { std } => {
            Ok(Tensor::randn(shape, (Kind::Float, device)) * std as f64)
        }
        InitStrategy::Xavier => {
            let fan_in = shape[0] as f32;
            let fan_out = shape[1] as f32;
            let std = (2.0 / (fan_in + fan_out)).sqrt();
            Ok(Tensor::randn(shape, (Kind::Float, device)) * std as f64)
        }
        InitStrategy::Kaiming { fan_in } => {
            let std = (2.0 / fan_in as f32).sqrt();
            Ok(Tensor::randn(shape, (Kind::Float, device)) * std as f64)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    
    #[test]
    fn test_validate_config() {
        // Valid config
        assert!(validate_lora_config(16, 768, 768).is_ok());
        
        // Invalid rank
        assert!(validate_lora_config(0, 768, 768).is_err());
        assert!(validate_lora_config(1000, 768, 768).is_err());
        
        // Invalid dimensions
        assert!(validate_lora_config(16, 0, 768).is_err());
        assert!(validate_lora_config(16, 768, 0).is_err());
    }
}