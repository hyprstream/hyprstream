//! Configuration structures for LoRA training

use serde::{Deserialize, Serialize};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Weight decay for AdamW
    pub weight_decay: f64,
    
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Number of training epochs
    pub epochs: usize,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    
    /// Warmup steps for learning rate scheduler
    pub warmup_steps: usize,
    
    /// Use mixed precision training
    pub mixed_precision: bool,
    
    /// Checkpoint save frequency (steps)
    pub save_steps: usize,
    
    /// Logging frequency (steps)
    pub logging_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            batch_size: 4,
            epochs: 3,
            gradient_accumulation_steps: 4,
            warmup_steps: 100,
            mixed_precision: true,
            save_steps: 500,
            logging_steps: 10,
        }
    }
}

/// Simple quantization options (optional feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization bits (4 or 8)
    pub bits: u8,
    
    /// Quantization type
    pub quant_type: QuantizationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    /// NormalFloat4 from QLoRA paper
    NF4,
    /// Standard int8 quantization
    Int8,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            quant_type: QuantizationType::Int8,
        }
    }
}