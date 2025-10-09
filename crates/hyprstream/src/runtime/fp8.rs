//! FP8 (E4M3/E5M2) support for efficient training and inference
//!
//! This module provides FP8 quantization and mixed-precision training
//! based on the NVIDIA Transformer Engine approach used by DeepL.

use anyhow::{Result, anyhow};
use tch::{Kind as DType, Device, Tensor};
use std::sync::Arc;
use parking_lot::RwLock;

/// FP8 formats as defined by IEEE and used in ML
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FP8Format {
    /// E4M3: 1 sign, 4 exponent, 3 mantissa bits
    /// Range: ±448, More precision, less range
    /// Best for: Forward pass, weights
    E4M3,
    
    /// E5M2: 1 sign, 5 exponent, 2 mantissa bits  
    /// Range: ±57,344, Less precision, more range
    /// Best for: Gradients, backward pass
    /// NOTE: Not yet supported in candle-core
    E5M2,
}

impl FP8Format {
    /// Get the maximum representable value
    pub fn max_value(&self) -> f32 {
        match self {
            FP8Format::E4M3 => 448.0,
            FP8Format::E5M2 => 57344.0,
        }
    }
    
    /// Get the minimum positive normal value
    pub fn min_positive(&self) -> f32 {
        match self {
            FP8Format::E4M3 => 2.0_f32.powi(-9),  // ~0.00195
            FP8Format::E5M2 => 2.0_f32.powi(-14), // ~0.00006
        }
    }
    
    /// Convert to candle DType (when available)
    pub fn to_dtype(&self) -> Result<DType> {
        match self {
            FP8Format::E4M3 => {
                // E4M3 not yet available as a DType in candle 0.9.1
                // Will be added in future versions
                Err(anyhow!("F8E4M3 not yet available as DType in candle-core 0.9.1. Use BF16 instead."))
            }
            FP8Format::E5M2 => {
                // E5M2 not yet supported in candle
                Err(anyhow!("F8E5M2 not yet supported in candle-core. Use BF16 instead."))
            }
        }
    }
    
    /// Check if format is supported by current candle version
    pub fn is_supported(&self) -> bool {
        // FP8 dtypes not yet available in candle 0.9.1
        // Will be enabled when candle adds F8E4M3 and F8E5M2
        false
    }
}

/// Dynamic scaling for FP8 quantization
/// Based on NVIDIA Transformer Engine approach
#[derive(Debug, Clone)]
pub struct FP8Scaler {
    /// Scale factor for forward pass (E4M3)
    forward_scale: Arc<RwLock<f32>>,
    
    /// Scale factor for backward pass (E5M2)
    backward_scale: Arc<RwLock<f32>>,
    
    /// History of gradient magnitudes for adaptive scaling
    grad_history: Arc<RwLock<Vec<f32>>>,
    
    /// Update frequency for scale factors
    update_frequency: usize,
    
    /// Counter for updates
    update_counter: Arc<RwLock<usize>>,
    
    /// Margin factor to prevent overflow (typically 0.9)
    margin: f32,
}

impl FP8Scaler {
    /// Create new FP8 scaler with default settings
    pub fn new() -> Self {
        Self {
            forward_scale: Arc::new(RwLock::new(1.0)),
            backward_scale: Arc::new(RwLock::new(1.0)),
            grad_history: Arc::new(RwLock::new(Vec::with_capacity(100))),
            update_frequency: 100,
            update_counter: Arc::new(RwLock::new(0)),
            margin: 0.9,
        }
    }
    
    /// Quantize tensor to FP8 with scaling
    pub fn quantize(&self, tensor: &Tensor, format: FP8Format) -> Result<(Tensor, f32)> {
        // Get appropriate scale based on format
        let _scale = match format {
            FP8Format::E4M3 => *self.forward_scale.read(),
            FP8Format::E5M2 => *self.backward_scale.read(),
        };
        
        // Find max absolute value in tensor
        let max_val = tensor.abs().max().double_value(&[]) as f32;
        
        // Calculate optimal scale to fit in FP8 range
        let target_max = format.max_value() * self.margin;
        let computed_scale = if max_val > 0.0 {
            target_max / max_val
        } else {
            1.0
        };
        
        // Apply scaling
        let scale_tensor = Tensor::from_slice(&[computed_scale]).to(tensor.device());
        let scaled = tensor * &scale_tensor;
        
        // Convert to FP8 (if supported)
        let quantized = if format.is_supported() {
            scaled.to_dtype(format.to_dtype()?, false, false)
        } else {
            // Fallback to BF16 for unsupported formats
            tracing::warn!("FP8 format {:?} not supported, using BF16", format);
            scaled.to_dtype(DType::Half, false, false)
        };
        
        Ok((quantized, computed_scale))
    }
    
    /// Dequantize FP8 tensor back to higher precision
    pub fn dequantize(&self, tensor: &Tensor, scale: f32, target_dtype: DType) -> Result<Tensor> {
        // Convert to target dtype and rescale
        let dequantized = tensor.to_dtype(target_dtype, false, false);
        let inv_scale = Tensor::from_slice(&[1.0 / scale]).to(tensor.device());
        Ok(&dequantized * &inv_scale)
    }
    
    /// Update scaling factors based on observed values
    pub fn update_scales(&self, forward_max: f32, backward_max: f32) {
        let mut counter = self.update_counter.write();
        *counter += 1;
        
        // Update gradient history
        {
            let mut history = self.grad_history.write();
            history.push(backward_max);
            
            // Keep only recent history
            if history.len() > 1000 {
                history.drain(0..500);
            }
        }
        
        // Update scales periodically
        if *counter % self.update_frequency == 0 {
            // Forward scale update (more stable)
            {
                let mut scale = self.forward_scale.write();
                let target_max = FP8Format::E4M3.max_value() * self.margin;
                let new_scale = target_max / forward_max.max(1e-6);
                
                // Smooth update to prevent oscillation
                *scale = *scale * 0.7 + new_scale * 0.3;
            }
            
            // Backward scale update (based on history)
            {
                let history = self.grad_history.read();
                if !history.is_empty() {
                    // Use 95th percentile to be robust to outliers
                    let mut sorted = history.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let percentile_95 = sorted[sorted.len() * 95 / 100];
                    
                    let mut scale = self.backward_scale.write();
                    let target_max = FP8Format::E5M2.max_value() * self.margin;
                    let new_scale = target_max / percentile_95.max(1e-6);
                    
                    // Smooth update
                    *scale = *scale * 0.8 + new_scale * 0.2;
                }
            }
        }
    }
}

/// Mixed precision configuration for FP8 training
#[derive(Debug, Clone)]
pub struct FP8Config {
    /// Use E4M3 for forward pass
    pub forward_format: FP8Format,
    
    /// Use E5M2 for backward pass (when available)
    pub backward_format: FP8Format,
    
    /// Master weight dtype (BF16 or FP32)
    pub master_dtype: DType,
    
    /// LoRA adapter dtype (always BF16 or higher)
    pub lora_dtype: DType,
    
    /// Enable dynamic loss scaling
    pub dynamic_scaling: bool,
    
    /// Scaler for quantization
    pub scaler: Arc<FP8Scaler>,
}

impl Default for FP8Config {
    fn default() -> Self {
        Self {
            forward_format: FP8Format::E4M3,
            backward_format: FP8Format::E5M2, // Will fallback to BF16
            master_dtype: DType::Half,
            lora_dtype: DType::Half, // LoRA always needs precision
            dynamic_scaling: true,
            scaler: Arc::new(FP8Scaler::new()),
        }
    }
}

impl FP8Config {
    /// Create config optimized for inference
    pub fn inference() -> Self {
        Self {
            forward_format: FP8Format::E4M3,
            backward_format: FP8Format::E4M3, // No gradients in inference
            master_dtype: DType::Half,
            lora_dtype: DType::Half,
            dynamic_scaling: false, // No need for dynamic scaling
            scaler: Arc::new(FP8Scaler::new()),
        }
    }
    
    /// Create config optimized for training
    pub fn training() -> Self {
        Self {
            forward_format: FP8Format::E4M3,
            backward_format: FP8Format::E5M2,
            master_dtype: DType::Float, // Higher precision for optimizer
            lora_dtype: DType::Half,
            dynamic_scaling: true,
            scaler: Arc::new(FP8Scaler::new()),
        }
    }
    
    /// Check if configuration is fully supported
    pub fn is_fully_supported(&self) -> bool {
        self.forward_format.is_supported() && self.backward_format.is_supported()
    }
    
    /// Get fallback configuration if FP8 not fully supported
    pub fn get_fallback(&self) -> Self {
        if self.is_fully_supported() {
            self.clone()
        } else {
            // Fallback to BF16 for unsupported formats
            Self {
                forward_format: FP8Format::E4M3, // Keep E4M3 if supported
                backward_format: FP8Format::E4M3, // Use E4M3 instead of E5M2
                master_dtype: self.master_dtype,
                lora_dtype: self.lora_dtype,
                dynamic_scaling: self.dynamic_scaling,
                scaler: self.scaler.clone(),
            }
        }
    }
}

/// FP8 mixed precision engine wrapper
pub struct FP8Engine {
    config: FP8Config,
    #[allow(dead_code)]
    device: Device,

    /// Base model weights in FP8
    base_weights_fp8: Option<Vec<Tensor>>,
    
    /// Master weights in higher precision
    master_weights: Option<Vec<Tensor>>,
    
    /// LoRA weights (always BF16+)
    lora_weights: Option<Vec<Tensor>>,
}

impl FP8Engine {
    /// Create new FP8 engine
    pub fn new(device: Device, config: FP8Config) -> Self {
        let config = if config.is_fully_supported() {
            config
        } else {
            tracing::warn!("FP8 not fully supported, using fallback configuration");
            config.get_fallback()
        };
        
        Self {
            config,
            device,
            base_weights_fp8: None,
            master_weights: None,
            lora_weights: None,
        }
    }
    
    /// Convert model weights to FP8
    pub fn quantize_model(&mut self, weights: Vec<Tensor>) -> Result<()> {
        let mut fp8_weights = Vec::new();
        let mut master_weights = Vec::new();
        
        for weight in weights {
            // Keep master copy
            let master = weight.to_dtype(self.config.master_dtype, false, false);
            master_weights.push(master.shallow_clone());
            
            // Quantize to FP8
            let (fp8_weight, _scale) = self.config.scaler.quantize(&weight, self.config.forward_format)?;
            fp8_weights.push(fp8_weight);
        }
        
        self.base_weights_fp8 = Some(fp8_weights);
        self.master_weights = Some(master_weights);
        
        Ok(())
    }
    
    /// Forward pass with FP8 weights
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weights = self.base_weights_fp8.as_ref()
            .ok_or_else(|| anyhow!("Model not quantized"))?;
        
        // Placeholder for actual forward pass
        // In practice, this would call into model-specific logic
        
        // For now, just show the pattern
        let mut output = input.shallow_clone();
        
        for weight in weights {
            // Dequantize weight for computation (or use FP8 kernels if available)
            // Since FP8 is not yet available, weights are already in BF16/F16
            let weight_bf16 = weight.shallow_clone();
            
            // Apply weight (simplified)
            output = output.matmul(&weight_bf16);
        }
        
        // Apply LoRA if present
        if let Some(lora_weights) = &self.lora_weights {
            let lora_output = self.apply_lora(&input, lora_weights)?;
            output = &output + &lora_output;
        }
        
        Ok(output)
    }
    
    /// Apply LoRA adapter (always in higher precision)
    fn apply_lora(&self, input: &Tensor, lora_weights: &[Tensor]) -> Result<Tensor> {
        // LoRA computation stays in BF16/FP32
        let mut output = Tensor::zeros_like(input);
        
        for (i, _weight) in lora_weights.iter().enumerate() {
            if i % 2 == 0 && i + 1 < lora_weights.len() {
                // LoRA: output = input @ A @ B
                let a = &lora_weights[i];
                let b = &lora_weights[i + 1];
                
                let intermediate = input.matmul(a);
                let lora_out = intermediate.matmul(b);
                output = &output + &lora_out;
            }
        }
        
        Ok(output)
    }
    
    /// Backward pass with mixed precision
    pub fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        // Quantize gradients to E5M2 (when available)
        let (grad_fp8, scale) = if self.config.backward_format.is_supported() {
            self.config.scaler.quantize(grad_output, self.config.backward_format)?
        } else {
            // Fallback to BF16
            (grad_output.to_dtype(DType::Half, false, false), 1.0)
        };
        
        // Update scaler statistics
        let grad_max = grad_output.abs().max().double_value(&[]) as f32;
        let weight_max = self.master_weights.as_ref()
            .and_then(|w| w.first())
            .map(|w| w.abs().max().double_value(&[]) as f32)
            .unwrap_or(1.0);
        
        self.config.scaler.update_scales(weight_max, grad_max);
        
        // Compute gradients (placeholder)
        let grad_input = self.config.scaler.dequantize(&grad_fp8, scale, self.config.master_dtype)?;
        
        Ok(grad_input)
    }
}

/// Check if current hardware supports FP8
pub fn supports_fp8(device: &Device) -> bool {
    match device {
        Device::Cuda(_cuda_device) => {
            // FP8 requires compute capability 9.0+ (H100)
            // This is a simplified check - real implementation would query device
            false // Conservative default
        }
        _ => false, // Only CUDA supports FP8 currently
    }
}

/// Get optimal FP8 configuration for hardware
pub fn optimal_fp8_config(device: &Device) -> FP8Config {
    if supports_fp8(device) {
        FP8Config::training()
    } else {
        // Fallback configuration
        FP8Config {
            forward_format: FP8Format::E4M3, // Use E4M3 if available
            backward_format: FP8Format::E4M3, // Can't use E5M2 yet
            master_dtype: DType::Half,
            lora_dtype: DType::Half,
            dynamic_scaling: true,
            scaler: Arc::new(FP8Scaler::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fp8_format_properties() {
        assert_eq!(FP8Format::E4M3.max_value(), 448.0);
        assert_eq!(FP8Format::E5M2.max_value(), 57344.0);
        
        assert!(!FP8Format::E4M3.is_supported()); // Not yet in candle 0.9.1
        assert!(!FP8Format::E5M2.is_supported()); // Not yet in candle
    }
    
    #[test]
    fn test_fp8_config_fallback() {
        let config = FP8Config::training();
        assert!(!config.is_fully_supported()); // E5M2 not supported
        
        let fallback = config.get_fallback();
        assert_eq!(fallback.backward_format, FP8Format::E4M3); // Falls back to E4M3
    }
}