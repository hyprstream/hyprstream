//! # bitsandbytes-sys
//!
//! Rust FFI bindings for the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
//! quantization library.
//!
//! This crate provides low-level bindings to bitsandbytes for:
//! - 8-bit blockwise quantization (LLM.int8())
//! - 4-bit quantization (NF4, FP4)
//! - Quantized matrix multiplication
//!
//! ## Backends
//!
//! The library supports multiple backends:
//! - `cuda` - NVIDIA GPUs via CUDA
//! - `hip` - AMD GPUs via ROCm/HIP
//! - `cpu` - CPU fallback
//!
//! ## Example
//!
//! ```rust,ignore
//! use bitsandbytes_sys::{QuantState, quantize_blockwise_fp32};
//!
//! let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//! let (quantized, state) = quantize_blockwise_fp32(&input, 64)?;
//! let dequantized = dequantize_blockwise_fp32(&quantized, &state)?;
//! ```

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::c_int;
#[allow(unused_imports)]
use std::ptr;

// ============================================================================
// Raw FFI Bindings
// ============================================================================

// Re-export bindings module
mod bindings;
pub use bindings::*;

// ============================================================================
// Safe Rust Wrappers
// ============================================================================

/// Error type for bitsandbytes operations
#[derive(Debug, Clone)]
pub enum BnbError {
    /// Library not found or not loaded
    LibraryNotFound,
    /// Invalid input parameters
    InvalidInput(String),
    /// Quantization failed
    QuantizationFailed(String),
    /// Memory allocation failed
    AllocationFailed,
    /// GPU not available
    NoGpu,
}

impl std::fmt::Display for BnbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BnbError::LibraryNotFound => write!(f, "bitsandbytes library not found"),
            BnbError::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            BnbError::QuantizationFailed(msg) => write!(f, "quantization failed: {}", msg),
            BnbError::AllocationFailed => write!(f, "memory allocation failed"),
            BnbError::NoGpu => write!(f, "GPU not available"),
        }
    }
}

impl std::error::Error for BnbError {}

pub type Result<T> = std::result::Result<T, BnbError>;

/// Quantization type for 4-bit quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// Normalized Float 4-bit (recommended for LLM weight quantization)
    Nf4,
    /// Float Point 4-bit
    Fp4,
}

/// Quantization state storing metadata for dequantization
#[derive(Debug, Clone)]
pub struct QuantState {
    /// Per-block absolute maximum values
    pub absmax: Vec<f32>,
    /// Quantization codebook (256 values for 8-bit, 16 for 4-bit)
    pub code: Vec<f32>,
    /// Block size used for quantization
    pub blocksize: usize,
    /// Number of elements (before quantization)
    pub n_elements: usize,
    /// Quantization type (for 4-bit)
    pub quant_type: Option<QuantType>,
    /// Whether this is 4-bit (true) or 8-bit (false) quantization
    pub is_4bit: bool,
}

impl QuantState {
    /// Create a new quantization state for 8-bit blockwise quantization
    pub fn new_8bit(n_elements: usize, blocksize: usize) -> Self {
        let n_blocks = (n_elements + blocksize - 1) / blocksize;

        // Initialize the dynamic quantization codebook (linear mapping)
        let mut code = vec![0.0f32; 256];
        for i in 0..256 {
            code[i] = (i as f32 - 127.0) / 127.0;
        }

        Self {
            absmax: vec![0.0f32; n_blocks],
            code,
            blocksize,
            n_elements,
            quant_type: None,
            is_4bit: false,
        }
    }

    /// Create a new quantization state for 4-bit quantization
    pub fn new_4bit(n_elements: usize, blocksize: usize, quant_type: QuantType) -> Self {
        let n_blocks = (n_elements + blocksize - 1) / blocksize;

        // Initialize the 4-bit codebook based on type
        let code = match quant_type {
            QuantType::Nf4 => {
                // NF4 codebook values (from bitsandbytes paper)
                vec![
                    -1.0,
                    -0.6961928009986877,
                    -0.5250730514526367,
                    -0.39491748809814453,
                    -0.28444138169288635,
                    -0.18477343022823334,
                    -0.09105003625154495,
                    0.0,
                    0.07958029955625534,
                    0.16093020141124725,
                    0.24611230945587158,
                    0.33791524171829224,
                    0.44070982933044434,
                    0.5626170039176941,
                    0.7229568362236023,
                    1.0,
                ]
            }
            QuantType::Fp4 => {
                // FP4 codebook values
                vec![
                    0.0,
                    0.0625,
                    0.125,
                    0.1875,
                    0.25,
                    0.3125,
                    0.375,
                    0.4375,
                    0.5,
                    0.5625,
                    0.625,
                    0.6875,
                    0.75,
                    0.8125,
                    0.875,
                    1.0,
                ]
            }
        };

        Self {
            absmax: vec![0.0f32; n_blocks],
            code,
            blocksize,
            n_elements,
            quant_type: Some(quant_type),
            is_4bit: true,
        }
    }

    /// Get the number of blocks
    pub fn n_blocks(&self) -> usize {
        self.absmax.len()
    }

    /// Get the size of quantized data in bytes
    pub fn quantized_size(&self) -> usize {
        if self.is_4bit {
            // 4-bit: 2 values per byte
            (self.n_elements + 1) / 2
        } else {
            // 8-bit: 1 value per byte
            self.n_elements
        }
    }
}

// ============================================================================
// High-Level Quantization Functions
// ============================================================================

/// Quantize f32 data using 8-bit blockwise quantization
///
/// # Arguments
/// * `input` - Input f32 data
/// * `blocksize` - Block size for quantization (typically 4096)
///
/// # Returns
/// Tuple of (quantized data, quantization state)
#[cfg(not(bnb_stub))]
pub fn quantize_blockwise_fp32(input: &[f32], blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    if input.is_empty() {
        return Err(BnbError::InvalidInput("input is empty".to_string()));
    }

    let n = input.len();
    let mut state = QuantState::new_8bit(n, blocksize);
    let mut output = vec![0u8; n];

    unsafe {
        cquantize_blockwise_fp32(
            state.code.as_mut_ptr(),
            input.as_ptr() as *mut f32,
            state.absmax.as_mut_ptr(),
            output.as_mut_ptr(),
            blocksize as c_int,
            n as c_int,
        );
    }

    Ok((output, state))
}

/// Dequantize 8-bit blockwise quantized data back to f32
#[cfg(not(bnb_stub))]
pub fn dequantize_blockwise_fp32(input: &[u8], state: &QuantState) -> Result<Vec<f32>> {
    if state.is_4bit {
        return Err(BnbError::InvalidInput(
            "use dequantize_4bit_fp32 for 4-bit data".to_string(),
        ));
    }

    let n = state.n_elements;
    let mut output = vec![0.0f32; n];

    unsafe {
        cdequantize_blockwise_fp32(
            state.code.as_ptr() as *mut f32,
            input.as_ptr() as *mut u8,
            state.absmax.as_ptr() as *mut f32,
            output.as_mut_ptr(),
            state.blocksize as c_int,
            n as c_int,
            ptr::null_mut(), // default stream
        );
    }

    Ok(output)
}

/// Quantize f32 data using 4-bit NF4 quantization
///
/// NF4 (Normalized Float 4-bit) is optimized for normally distributed weights.
///
/// # Arguments
/// * `input` - Input f32 data
/// * `blocksize` - Block size for quantization (typically 64)
///
/// # Returns
/// Tuple of (quantized data, quantization state)
#[cfg(not(bnb_stub))]
pub fn quantize_4bit_nf4_fp32(input: &[f32], blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    if input.is_empty() {
        return Err(BnbError::InvalidInput("input is empty".to_string()));
    }

    let n = input.len();
    let mut state = QuantState::new_4bit(n, blocksize, QuantType::Nf4);

    // Output is packed: 2 values per byte
    let output_size = (n + 1) / 2;
    let mut output = vec![0u8; output_size];

    unsafe {
        cquantize_blockwise_fp32_nf4(
            state.code.as_mut_ptr(),
            input.as_ptr() as *mut f32,
            state.absmax.as_mut_ptr(),
            output.as_mut_ptr(),
            blocksize as c_int,
            n as c_int,
        );
    }

    Ok((output, state))
}

/// Dequantize 4-bit NF4 quantized data back to f32
#[cfg(not(bnb_stub))]
pub fn dequantize_4bit_nf4_fp32(input: &[u8], state: &QuantState) -> Result<Vec<f32>> {
    if !state.is_4bit || state.quant_type != Some(QuantType::Nf4) {
        return Err(BnbError::InvalidInput(
            "state does not match NF4 quantization".to_string(),
        ));
    }

    let n = state.n_elements;
    let mut output = vec![0.0f32; n];

    unsafe {
        cdequantize_blockwise_fp32_nf4(
            state.code.as_ptr() as *mut f32,
            input.as_ptr() as *mut u8,
            state.absmax.as_ptr() as *mut f32,
            output.as_mut_ptr(),
            state.blocksize as c_int,
            n as c_int,
            ptr::null_mut(), // default stream
        );
    }

    Ok(output)
}

/// Quantize f32 data using 4-bit FP4 quantization
///
/// FP4 (Float Point 4-bit) uses a linear quantization scheme.
///
/// # Arguments
/// * `input` - Input f32 data
/// * `blocksize` - Block size for quantization (typically 64)
///
/// # Returns
/// Tuple of (quantized data, quantization state)
#[cfg(not(bnb_stub))]
pub fn quantize_4bit_fp4_fp32(input: &[f32], blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    if input.is_empty() {
        return Err(BnbError::InvalidInput("input is empty".to_string()));
    }

    let n = input.len();
    let mut state = QuantState::new_4bit(n, blocksize, QuantType::Fp4);

    // Output is packed: 2 values per byte
    let output_size = (n + 1) / 2;
    let mut output = vec![0u8; output_size];

    unsafe {
        cquantize_blockwise_fp32_fp4(
            state.code.as_mut_ptr(),
            input.as_ptr() as *mut f32,
            state.absmax.as_mut_ptr(),
            output.as_mut_ptr(),
            blocksize as c_int,
            n as c_int,
        );
    }

    Ok((output, state))
}

/// Dequantize 4-bit FP4 quantized data back to f32
#[cfg(not(bnb_stub))]
pub fn dequantize_4bit_fp4_fp32(input: &[u8], state: &QuantState) -> Result<Vec<f32>> {
    if !state.is_4bit || state.quant_type != Some(QuantType::Fp4) {
        return Err(BnbError::InvalidInput(
            "state does not match FP4 quantization".to_string(),
        ));
    }

    let n = state.n_elements;
    let mut output = vec![0.0f32; n];

    unsafe {
        cdequantize_blockwise_fp32_fp4(
            state.code.as_ptr() as *mut f32,
            input.as_ptr() as *mut u8,
            state.absmax.as_ptr() as *mut f32,
            output.as_mut_ptr(),
            state.blocksize as c_int,
            n as c_int,
            ptr::null_mut(), // default stream
        );
    }

    Ok(output)
}

/// Generic 4-bit quantization function that dispatches to NF4 or FP4
#[cfg(not(bnb_stub))]
pub fn quantize_4bit_fp32(
    input: &[f32],
    blocksize: usize,
    quant_type: QuantType,
) -> Result<(Vec<u8>, QuantState)> {
    match quant_type {
        QuantType::Nf4 => quantize_4bit_nf4_fp32(input, blocksize),
        QuantType::Fp4 => quantize_4bit_fp4_fp32(input, blocksize),
    }
}

/// Generic 4-bit dequantization function that dispatches based on state
#[cfg(not(bnb_stub))]
pub fn dequantize_4bit_fp32(input: &[u8], state: &QuantState) -> Result<Vec<f32>> {
    match state.quant_type {
        Some(QuantType::Nf4) => dequantize_4bit_nf4_fp32(input, state),
        Some(QuantType::Fp4) => dequantize_4bit_fp4_fp32(input, state),
        None => Err(BnbError::InvalidInput(
            "quant_type not set in state".to_string(),
        )),
    }
}

// ============================================================================
// CPU Fallback Functions
// ============================================================================

/// Quantize f32 data using 8-bit blockwise quantization (CPU implementation)
#[cfg(not(bnb_stub))]
pub fn quantize_blockwise_cpu_fp32(input: &[f32], blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    if input.is_empty() {
        return Err(BnbError::InvalidInput("input is empty".to_string()));
    }

    let n = input.len();
    let mut state = QuantState::new_8bit(n, blocksize);
    let mut output = vec![0u8; n];

    unsafe {
        cquantize_blockwise_cpu_fp32(
            state.code.as_mut_ptr(),
            input.as_ptr() as *mut f32,
            state.absmax.as_mut_ptr(),
            output.as_mut_ptr(),
            blocksize as std::ffi::c_longlong,
            n as std::ffi::c_longlong,
        );
    }

    Ok((output, state))
}

/// Dequantize 8-bit blockwise quantized data back to f32 (CPU implementation)
#[cfg(not(bnb_stub))]
pub fn dequantize_blockwise_cpu_fp32(input: &[u8], state: &QuantState) -> Result<Vec<f32>> {
    if state.is_4bit {
        return Err(BnbError::InvalidInput(
            "use dequantize_4bit_cpu_fp32 for 4-bit data".to_string(),
        ));
    }

    let n = state.n_elements;
    let mut output = vec![0.0f32; n];

    unsafe {
        cdequantize_blockwise_cpu_fp32(
            state.code.as_ptr() as *mut f32,
            input.as_ptr() as *mut u8,
            state.absmax.as_ptr() as *mut f32,
            output.as_mut_ptr(),
            state.blocksize as c_int,
            n as c_int,
        );
    }

    Ok(output)
}

// ============================================================================
// Stub implementations (when library not found)
// ============================================================================

#[cfg(bnb_stub)]
pub fn quantize_blockwise_fp32(_input: &[f32], _blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn dequantize_blockwise_fp32(_input: &[u8], _state: &QuantState) -> Result<Vec<f32>> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn quantize_4bit_nf4_fp32(_input: &[f32], _blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn dequantize_4bit_nf4_fp32(_input: &[u8], _state: &QuantState) -> Result<Vec<f32>> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn quantize_4bit_fp4_fp32(_input: &[f32], _blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn dequantize_4bit_fp4_fp32(_input: &[u8], _state: &QuantState) -> Result<Vec<f32>> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn quantize_4bit_fp32(
    _input: &[f32],
    _blocksize: usize,
    _quant_type: QuantType,
) -> Result<(Vec<u8>, QuantState)> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn dequantize_4bit_fp32(_input: &[u8], _state: &QuantState) -> Result<Vec<f32>> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn quantize_blockwise_cpu_fp32(_input: &[f32], _blocksize: usize) -> Result<(Vec<u8>, QuantState)> {
    Err(BnbError::LibraryNotFound)
}

#[cfg(bnb_stub)]
pub fn dequantize_blockwise_cpu_fp32(_input: &[u8], _state: &QuantState) -> Result<Vec<f32>> {
    Err(BnbError::LibraryNotFound)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if bitsandbytes library is available
pub fn is_available() -> bool {
    #[cfg(bnb_stub)]
    {
        false
    }
    #[cfg(not(bnb_stub))]
    {
        true
    }
}

/// Get the current backend type
#[allow(unreachable_code)]
pub fn get_backend() -> &'static str {
    #[cfg(bnb_cuda)]
    return "cuda";
    #[cfg(bnb_hip)]
    return "hip";
    #[cfg(bnb_cpu)]
    return "cpu";
    #[cfg(bnb_stub)]
    return "stub";
    #[cfg(not(any(bnb_cuda, bnb_hip, bnb_cpu, bnb_stub)))]
    return "unknown";
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_state_new_8bit() {
        let state = QuantState::new_8bit(1024, 64);
        assert_eq!(state.blocksize, 64);
        assert_eq!(state.n_elements, 1024);
        assert_eq!(state.n_blocks(), 16);
        assert_eq!(state.quantized_size(), 1024);
        assert!(!state.is_4bit);
    }

    #[test]
    fn test_quant_state_new_4bit() {
        let state = QuantState::new_4bit(1024, 64, QuantType::Nf4);
        assert_eq!(state.blocksize, 64);
        assert_eq!(state.n_elements, 1024);
        assert_eq!(state.n_blocks(), 16);
        assert_eq!(state.quantized_size(), 512); // 4-bit: half the bytes
        assert!(state.is_4bit);
        assert_eq!(state.quant_type, Some(QuantType::Nf4));
    }

    #[test]
    fn test_backend_detection() {
        let backend = get_backend();
        println!("Detected backend: {}", backend);
        // Should be one of the valid backends
        assert!(["cuda", "hip", "cpu", "stub", "unknown"].contains(&backend));
    }

    #[test]
    fn test_is_available() {
        let available = is_available();
        println!("bitsandbytes available: {}", available);
        // Just check it doesn't panic
    }
}
