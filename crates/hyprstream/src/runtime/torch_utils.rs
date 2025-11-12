//! Utilities for safe PyTorch operations with proper error handling
//!
//! ## Panic Handling and FFI Boundaries
//!
//! PyTorch's C++ backend (libtorch) throws exceptions on OOM and other errors.
//! The tch-rs bindings translate these C++ exceptions into Rust panics during
//! operations like tensor allocation (`Tensor::zeros`) and device transfers
//! (`tensor.to_device`).
//!
//! This module provides wrappers that use `catch_unwind` to catch these panics
//! and convert them to proper `Result` errors with actionable error messages.
//!
//! **Important**: This only works because tch-rs performs the C++ exception → Rust
//! panic translation. Raw libtorch panics from C++ cannot be caught by `catch_unwind`.
//!
//! ## Thread Safety
//!
//! These utilities are safe to call from multiple threads, but be aware that
//! catching panics during tensor operations may leave PyTorch's CUDA/ROCm
//! allocator in an inconsistent state. After catching an OOM panic, it's
//! recommended to avoid further allocations on the same device, or restart
//! the application to reset allocator state.

use anyhow::{anyhow, Result};
use std::panic::catch_unwind;
use tch::{Device, Tensor};
use tracing::{error, warn};

/// Safely execute a tensor operation that might panic (e.g., OOM)
///
/// PyTorch operations like `Tensor::zeros()` and `to_device()` can panic
/// when GPU memory is exhausted. This wrapper catches such panics and
/// converts them to proper Result errors.
///
/// # Example
/// ```no_run
/// use tch::{Device, Tensor};
/// let result = safe_tensor_op(|| {
///     Tensor::zeros([1024, 1024], (tch::Kind::BFloat16, Device::Cuda(0)))
/// });
/// ```
pub fn safe_tensor_op<F, T>(operation: F) -> Result<T>
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    match catch_unwind(operation) {
        Ok(result) => Ok(result),
        Err(panic_info) => {
            // Try to extract panic message
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic during tensor operation".to_string()
            };

            error!("PyTorch operation panicked: {}", panic_msg);

            // Check if it's an OOM error
            if panic_msg.contains("out of memory") || panic_msg.contains("OOM") {
                Err(anyhow!("GPU OOM: {}", panic_msg))
            } else {
                Err(anyhow!("Tensor operation failed: {}", panic_msg))
            }
        }
    }
}

/// Safely transfer a tensor to a device (CPU to GPU)
///
/// This is a common operation that can OOM, so we provide a dedicated wrapper
/// with clear error messages.
pub fn safe_to_device(tensor: &Tensor, device: Device) -> Result<Tensor> {
    safe_tensor_op(|| tensor.to_device(device))
        .map_err(|e| anyhow!("Failed to_device({:?}): {} | shape: {:?}", device, e, tensor.size()))
}

/// Safely allocate a tensor (zeros, ones, randn, etc.)
///
/// Wraps tensor allocation operations that might OOM.
pub fn safe_zeros(shape: &[i64], kind_device: (tch::Kind, Device)) -> Result<Tensor> {
    let (kind, device) = kind_device;
    let size_mb = estimate_tensor_size_mb(shape, kind);
    safe_tensor_op(|| Tensor::zeros(shape, (kind, device)))
        .map_err(|e| anyhow!("Failed to allocate tensor on {:?}: {} | shape: {:?} ({:.1} MB)", device, e, shape, size_mb))
}

/// Estimate tensor size in megabytes
pub fn estimate_tensor_size_mb(shape: &[i64], dtype: tch::Kind) -> f64 {
    let num_elements: i64 = shape.iter().product();
    let bytes_per_element = match dtype {
        tch::Kind::Float => 4,
        tch::Kind::Double => 8,
        tch::Kind::Half | tch::Kind::BFloat16 => 2,
        tch::Kind::Int8 | tch::Kind::Uint8 => 1,
        tch::Kind::Int16 => 2,
        tch::Kind::Int => 4,
        tch::Kind::Int64 => 8,
        _ => 4, // Default to 4 bytes
    };

    (num_elements * bytes_per_element) as f64 / (1024.0 * 1024.0)
}

/// Check available GPU memory before attempting an operation
///
/// Returns (free_bytes, total_bytes) for the specified device.
/// Only works for CUDA/ROCm devices.
pub fn check_gpu_memory(device: Device) -> Result<(usize, usize)> {
    match device {
        Device::Cuda(device_idx) => {
            // Try to get CUDA memory info
            // Note: This requires tch-rs to expose CUDA memory APIs
            // For now, we'll try to call it via catch_unwind in case it panics
            match catch_unwind(|| {
                // Try to allocate a small tensor to test memory
                let test = Tensor::zeros([1], (tch::Kind::Float, device));
                drop(test);
            }) {
                Ok(_) => {
                    warn!("GPU memory check not fully implemented - using workaround");
                    // Return dummy values - this API needs tch-rs support
                    Ok((0, 0))
                }
                Err(_) => {
                    Err(anyhow!("GPU device {} is not accessible", device_idx))
                }
            }
        }
        Device::Cpu => {
            // CPU memory check is different and less critical
            Ok((0, 0))
        }
        _ => Ok((0, 0)),
    }
}

/// Pre-flight check before loading a model
///
/// Attempts to detect if there's enough GPU memory before starting
/// to load a model. This is a best-effort check.
pub fn preflight_gpu_check(device: Device, estimated_model_size_mb: f64) -> Result<()> {
    if !device.is_cuda() {
        return Ok(()); // Only check GPU devices
    }

    // Try to get memory info
    match check_gpu_memory(device) {
        Ok((free, total)) if total > 0 => {
            let free_mb = free as f64 / (1024.0 * 1024.0);
            if free_mb < estimated_model_size_mb {
                return Err(anyhow!(
                    "Insufficient GPU memory: {:.2} MB free, but model requires ~{:.2} MB\n\
                     \n\
                     Free GPU memory before retrying:\n\
                     - Kill other GPU processes\n\
                     - Restart hyprstream server\n\
                     - Use a smaller model",
                    free_mb,
                    estimated_model_size_mb
                ));
            }
        }
        _ => {
            // Memory check not available, proceed with caution
            warn!(
                "Could not check GPU memory before loading model. \
                 Estimated size: {:.2} MB",
                estimated_model_size_mb
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tensor_size() {
        // BFloat16 tensor: [1, 2048, 32, 128]
        let shape = vec![1, 2048, 32, 128];
        let size_mb = estimate_tensor_size_mb(&shape, tch::Kind::BFloat16);

        // Expected: 1 * 2048 * 32 * 128 * 2 bytes = 16,777,216 bytes = 16 MB
        assert!((size_mb - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_safe_tensor_op_cpu() {
        let result = safe_tensor_op(|| {
            Tensor::zeros([10, 10], (tch::Kind::Float, Device::Cpu))
        });
        assert!(result.is_ok());
    }
}
