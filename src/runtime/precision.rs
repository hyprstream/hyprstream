//! Precision management for BF16 and future FP8 support
//! 
//! This module handles dtype selection, conversion, and optimization
//! for different hardware targets and use cases.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use std::fmt;
use super::fp8::{FP8Format, FP8Config, FP8Scaler};

/// Precision modes for different use cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionMode {
    /// BF16 for everything (recommended default)
    BF16Standard,
    /// FP16 fallback for older hardware
    FP16Fallback,
    /// FP32 for debugging/research
    FP32Full,
    /// Mixed precision with FP8
    FP8Mixed {
        forward: FP8Format,
        backward: FP8Format,
        master: DType,
    },
}

/// Precision configuration for models and training
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    /// Base model weights precision
    pub base_model_dtype: DType,
    /// LoRA adapter weights precision (always BF16 or better)
    pub lora_adapter_dtype: DType,
    /// Gradient computation precision
    pub gradient_dtype: DType,
    /// Optimizer state precision (usually FP32)
    pub optimizer_dtype: DType,
    /// Compute/accumulation precision
    pub compute_dtype: DType,
    /// Scaling manager for FP8 (when needed)
    pub scaling_manager: Option<ScalingManager>,
}

impl PrecisionConfig {
    /// Create config for a specific precision mode
    pub fn from_mode(mode: PrecisionMode, device: &Device) -> Result<Self> {
        match mode {
            PrecisionMode::BF16Standard => {
                if !Self::supports_bf16(device) {
                    return Err(anyhow!("BF16 not supported on this device"));
                }
                Ok(Self {
                    base_model_dtype: DType::BF16,
                    lora_adapter_dtype: DType::BF16,
                    gradient_dtype: DType::BF16,
                    optimizer_dtype: DType::F32,
                    compute_dtype: DType::BF16,
                    scaling_manager: None,
                })
            }
            PrecisionMode::FP16Fallback => Ok(Self {
                base_model_dtype: DType::F16,
                lora_adapter_dtype: DType::F16,
                gradient_dtype: DType::F16,
                optimizer_dtype: DType::F32,
                compute_dtype: DType::F16,
                scaling_manager: None,
            }),
            PrecisionMode::FP32Full => Ok(Self {
                base_model_dtype: DType::F32,
                lora_adapter_dtype: DType::F32,
                gradient_dtype: DType::F32,
                optimizer_dtype: DType::F32,
                compute_dtype: DType::F32,
                scaling_manager: None,
            }),
            PrecisionMode::FP8Mixed { forward, backward, master } => {
                // Check if FP8 is actually supported
                let fp8_config = FP8Config::training();
                if !fp8_config.is_fully_supported() {
                    tracing::warn!("FP8 not fully supported, using fallback configuration");
                }
                
                Ok(Self {
                    base_model_dtype: forward.to_dtype().unwrap_or(DType::BF16),
                    lora_adapter_dtype: master, // LoRA stays in higher precision!
                    gradient_dtype: backward.to_dtype().unwrap_or(DType::BF16),
                    optimizer_dtype: DType::F32,
                    compute_dtype: master,
                    scaling_manager: Some(ScalingManager::new()),
                })
            }
        }
    }

    /// Auto-detect best precision for hardware
    pub fn auto_detect(device: &Device) -> Self {
        // Check if FP8 is supported (requires H100+)
        let fp8_config = super::fp8::optimal_fp8_config(device);
        
        if fp8_config.is_fully_supported() {
            // Use FP8 if fully supported
            Self::from_mode(
                PrecisionMode::FP8Mixed {
                    forward: FP8Format::E4M3,
                    backward: FP8Format::E5M2,
                    master: DType::BF16,
                },
                device,
            ).unwrap_or_else(|_| Self::bf16_default())
        } else if FP8Format::E4M3.is_supported() {
            // Use partial FP8 (E4M3 only)
            Self::from_mode(
                PrecisionMode::FP8Mixed {
                    forward: FP8Format::E4M3,
                    backward: FP8Format::E4M3, // Use E4M3 for both
                    master: DType::BF16,
                },
                device,
            ).unwrap_or_else(|_| Self::bf16_default())
        } else if Self::supports_bf16(device) {
            Self::bf16_default()
        } else {
            Self::fp16_fallback()
        }
    }

    /// BF16 default configuration
    pub fn bf16_default() -> Self {
        Self {
            base_model_dtype: DType::BF16,
            lora_adapter_dtype: DType::BF16,
            gradient_dtype: DType::BF16,
            optimizer_dtype: DType::F32,
            compute_dtype: DType::BF16,
            scaling_manager: None,
        }
    }

    /// FP16 fallback configuration
    pub fn fp16_fallback() -> Self {
        Self {
            base_model_dtype: DType::F16,
            lora_adapter_dtype: DType::F16,
            gradient_dtype: DType::F16,
            optimizer_dtype: DType::F32,
            compute_dtype: DType::F16,
            scaling_manager: None,
        }
    }

    /// Check BF16 support
    fn supports_bf16(device: &Device) -> bool {
        match device {
            Device::Cpu => true, // Most modern CPUs support BF16
            Device::Cuda(_) => {
                // Check compute capability
                // BF16 requires compute capability 8.0+ (A100, RTX 30xx+)
                // This is a simplified check - real implementation would query device
                true // Assume modern GPU for now
            }
            Device::Metal(_) => true, // M1/M2/M3 support BF16
        }
    }

    /// Check FP8 support
    fn supports_fp8(device: &Device) -> bool {
        super::fp8::supports_fp8(device)
    }

    /// Convert tensor to target precision
    pub fn convert_tensor(&self, tensor: &Tensor, target: TensorTarget) -> Result<Tensor> {
        let target_dtype = match target {
            TensorTarget::BaseModel => self.base_model_dtype,
            TensorTarget::LoRAAdapter => self.lora_adapter_dtype,
            TensorTarget::Gradient => self.gradient_dtype,
            TensorTarget::Optimizer => self.optimizer_dtype,
            TensorTarget::Compute => self.compute_dtype,
        };

        if tensor.dtype() == target_dtype {
            Ok(tensor.clone())
        } else {
            // Apply scaling if needed (for FP8)
            if let Some(ref scaler) = self.scaling_manager {
                scaler.scale_and_convert(tensor, target_dtype)
            } else {
                Ok(tensor.to_dtype(target_dtype)?)
            }
        }
    }
}

/// Target tensor type for precision conversion
#[derive(Debug, Clone, Copy)]
pub enum TensorTarget {
    BaseModel,
    LoRAAdapter,
    Gradient,
    Optimizer,
    Compute,
}

/// Scaling manager for FP8 conversions
#[derive(Debug, Clone)]
pub struct ScalingManager {
    fp8_scaler: std::sync::Arc<FP8Scaler>,
}

impl ScalingManager {
    pub fn new() -> Self {
        Self {
            fp8_scaler: std::sync::Arc::new(FP8Scaler::new()),
        }
    }

    /// Scale and convert tensor to target dtype
    pub fn scale_and_convert(&self, tensor: &Tensor, target_dtype: DType) -> Result<Tensor> {
        // For now, just convert without scaling
        // Real FP8 implementation would:
        // 1. Find max absolute value in tensor
        // 2. Calculate scale factor
        // 3. Apply scaling
        // 4. Convert to FP8
        // 5. Store scale factor for backward pass
        
        Ok(tensor.to_dtype(target_dtype)?)
    }

    /// Update scaling based on gradient statistics
    pub fn update_scaling(&self, forward_max: f32, backward_max: f32) {
        self.fp8_scaler.update_scales(forward_max, backward_max);
    }
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self::bf16_default()
    }
}

impl fmt::Display for PrecisionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecisionMode::BF16Standard => write!(f, "BF16"),
            PrecisionMode::FP16Fallback => write!(f, "FP16"),
            PrecisionMode::FP32Full => write!(f, "FP32"),
            PrecisionMode::FP8Mixed { .. } => write!(f, "FP8-Mixed"),
        }
    }
}

/// Hardware capability detection
pub struct HardwareCapabilities {
    pub supports_bf16: bool,
    pub supports_fp8: bool,
    pub supports_int8: bool,
    pub memory_gb: f32,
    pub compute_capability: Option<(u32, u32)>, // (major, minor) for CUDA
}

impl HardwareCapabilities {
    pub fn detect(device: &Device) -> Self {
        match device {
            Device::Cuda(cuda_device) => {
                // Simplified - real implementation would query CUDA
                Self {
                    supports_bf16: true,
                    supports_fp8: false, // H100+ only
                    supports_int8: true,
                    memory_gb: 24.0, // Placeholder
                    compute_capability: Some((8, 6)), // Example: RTX 3090
                }
            }
            Device::Metal(_) => Self {
                supports_bf16: true,
                supports_fp8: false,
                supports_int8: false,
                memory_gb: 32.0, // Placeholder
                compute_capability: None,
            },
            Device::Cpu => Self {
                supports_bf16: true,
                supports_fp8: false,
                supports_int8: true,
                memory_gb: 64.0, // System RAM placeholder
                compute_capability: None,
            },
        }
    }

    pub fn recommended_precision(&self) -> PrecisionMode {
        if self.supports_fp8 {
            PrecisionMode::FP8Mixed {
                forward: FP8Format::E4M3,
                backward: if FP8Format::E5M2.is_supported() {
                    FP8Format::E5M2
                } else {
                    FP8Format::E4M3 // Fallback to E4M3 for both
                },
                master: DType::BF16,
            }
        } else if self.supports_bf16 {
            PrecisionMode::BF16Standard
        } else {
            PrecisionMode::FP16Fallback
        }
    }
}