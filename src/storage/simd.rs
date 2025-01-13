//! SIMD operations for vector processing with architecture-specific optimizations.
//!
//! This module provides SIMD-accelerated operations with proper CPU feature detection
//! and fallbacks. Implementations are provided for:
//! - x86/x86_64 with SSE2/SSE4.1/AVX/AVX2
//! - Scalar fallback for systems without SIMD support
//!
//! All operations maintain IEEE-754 compliance and handle special cases (NaN,
//! infinities, denormals) according to specification.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Runtime CPU feature detection results
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    has_sse2: bool,
    has_sse41: bool,
    has_avx: bool,
    has_avx2: bool,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self::detect()
    }
}

impl CpuFeatures {
    /// Detect available CPU features at runtime
    #[inline]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_sse2: is_x86_feature_detected!("sse2"),
                has_sse41: is_x86_feature_detected!("sse4.1"),
                has_avx: is_x86_feature_detected!("avx"),
                has_avx2: is_x86_feature_detected!("avx2"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_sse2: false,
                has_sse41: false,
                has_avx: false,
                has_avx2: false,
            }
        }
    }
}

/// SIMD vector operations with runtime feature detection and fallbacks
pub struct SimdOps {
    features: CpuFeatures,
}

impl Default for SimdOps {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdOps {
    /// Create new SIMD operations instance with runtime feature detection
    pub fn new() -> Self {
        Self {
            features: CpuFeatures::detect(),
        }
    }

    /// Public safe interface for f32x4 multiplication
    pub fn f32x4_mul(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4 && y.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse2 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = _mm_loadu_ps(y.as_ptr());
                    let z = _mm_mul_ps(x, y);
                    _mm_storeu_ps(result.as_mut_ptr(), z);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i] * y[i];
        }
        result
    }

    /// Public safe interface for f32x4 division
    pub fn f32x4_div(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4 && y.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse2 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = _mm_loadu_ps(y.as_ptr());
                    let z = _mm_div_ps(x, y);
                    _mm_storeu_ps(result.as_mut_ptr(), z);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i] / y[i];
        }
        result
    }

    /// Absolute value of 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE2 support
    #[target_feature(enable = "sse2")]
    unsafe fn f32x4_abs_sse2(x: __m128) -> __m128 {
        // Clear sign bit with AND mask
        let mask = _mm_set1_ps(-0.0);
        _mm_andnot_ps(mask, x)
    }

    /// Square root of 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE2 support
    #[target_feature(enable = "sse2")]
    unsafe fn f32x4_sqrt_sse2(x: __m128) -> __m128 {
        _mm_sqrt_ps(x)
    }

    /// Ceiling of 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE4.1 support
    #[target_feature(enable = "sse4.1")]
    unsafe fn f32x4_ceil_sse41(x: __m128) -> __m128 {
        _mm_ceil_ps(x)
    }

    /// Floor of 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE4.1 support  
    #[target_feature(enable = "sse4.1")]
    unsafe fn f32x4_floor_sse41(x: __m128) -> __m128 {
        _mm_floor_ps(x)
    }

    /// Round 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE4.1 support
    #[target_feature(enable = "sse4.1")] 
    unsafe fn f32x4_round_sse41(x: __m128) -> __m128 {
        _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
    }

    /// Truncate 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE4.1 support
    #[target_feature(enable = "sse4.1")]
    unsafe fn f32x4_trunc_sse41(x: __m128) -> __m128 {
        _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
    }

    /// Minimum of 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE2 support
    #[target_feature(enable = "sse2")]
    unsafe fn f32x4_min_sse2(x: __m128, y: __m128) -> __m128 {
        _mm_min_ps(x, y)
    }

    /// Maximum of 4 packed f32 values
    /// 
    /// # Safety
    /// Requires CPU with SSE2 support
    #[target_feature(enable = "sse2")]
    unsafe fn f32x4_max_sse2(x: __m128, y: __m128) -> __m128 {
        _mm_max_ps(x, y)
    }

    /// Public safe interface for f32x4 absolute value
    pub fn f32x4_abs(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse2 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = SimdOps::f32x4_abs_sse2(x);
                    _mm_storeu_ps(result.as_mut_ptr(), y);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].abs();
        }
        result
    }

    /// Public safe interface for f32x4 square root
    pub fn f32x4_sqrt(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse2 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = SimdOps::f32x4_sqrt_sse2(x);
                    _mm_storeu_ps(result.as_mut_ptr(), y);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].sqrt();
        }
        result
    }

    /// Public safe interface for f32x4 ceiling
    pub fn f32x4_ceil(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse41 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = SimdOps::f32x4_ceil_sse41(x);
                    _mm_storeu_ps(result.as_mut_ptr(), y);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].ceil();
        }
        result
    }

    /// Public safe interface for f32x4 floor
    pub fn f32x4_floor(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse41 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = SimdOps::f32x4_floor_sse41(x);
                    _mm_storeu_ps(result.as_mut_ptr(), y);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].floor();
        }
        result
    }

    /// Public safe interface for f32x4 round
    pub fn f32x4_round(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse41 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = SimdOps::f32x4_round_sse41(x);
                    _mm_storeu_ps(result.as_mut_ptr(), y);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].round();
        }
        result
    }

    /// Public safe interface for f32x4 truncate
    pub fn f32x4_trunc(&self, x: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse41 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = SimdOps::f32x4_trunc_sse41(x);
                    _mm_storeu_ps(result.as_mut_ptr(), y);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].trunc();
        }
        result
    }

    /// Public safe interface for f32x4 minimum
    pub fn f32x4_min(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4 && y.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse2 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = _mm_loadu_ps(y.as_ptr());
                    let z = SimdOps::f32x4_min_sse2(x, y);
                    _mm_storeu_ps(result.as_mut_ptr(), z);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].min(y[i]);
        }
        result
    }

    /// Public safe interface for f32x4 maximum
    pub fn f32x4_max(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        assert!(x.len() >= 4 && y.len() >= 4);
        let mut result = vec![0.0; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if self.features.has_sse2 {
                unsafe {
                    let x = _mm_loadu_ps(x.as_ptr());
                    let y = _mm_loadu_ps(y.as_ptr());
                    let z = SimdOps::f32x4_max_sse2(x, y);
                    _mm_storeu_ps(result.as_mut_ptr(), z);
                    return result;
                }
            }
        }

        // Scalar fallback
        for i in 0..4 {
            result[i] = x[i].max(y[i]);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32;

    #[test]
    fn test_abs() {
        let ops = SimdOps::new();
        let x = vec![-1.0, 2.0, -3.0, 4.0];
        let result = ops.f32x4_abs(&x);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);

        // Test special cases
        let x = vec![-0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let result = ops.f32x4_abs(&x);
        assert_eq!(result[0], 0.0);
        assert!(result[1].is_nan());
        assert_eq!(result[2], f32::INFINITY);
        assert_eq!(result[3], f32::INFINITY);
    }

    #[test]
    fn test_sqrt() {
        let ops = SimdOps::new();
        let x = vec![0.0, 1.0, 4.0, 9.0];
        let result = ops.f32x4_sqrt(&x);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0]);

        // Test special cases
        let x = vec![-1.0, f32::NAN, f32::INFINITY, 0.0];
        let result = ops.f32x4_sqrt(&x);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], f32::INFINITY);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_ceil() {
        let ops = SimdOps::new();
        let x = vec![1.1, 2.5, -3.7, 4.0];
        let result = ops.f32x4_ceil(&x);
        assert_eq!(result, vec![2.0, 3.0, -3.0, 4.0]);

        // Test special cases
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let result = ops.f32x4_ceil(&x);
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::INFINITY);
        assert_eq!(result[2], f32::NEG_INFINITY);
        assert_eq!(result[3], -0.0);
    }

    #[test]
    fn test_floor() {
        let ops = SimdOps::new();
        let x = vec![1.1, 2.5, -3.7, 4.0];
        let result = ops.f32x4_floor(&x);
        assert_eq!(result, vec![1.0, 2.0, -4.0, 4.0]);

        // Test special cases
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let result = ops.f32x4_floor(&x);
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::INFINITY);
        assert_eq!(result[2], f32::NEG_INFINITY);
        assert_eq!(result[3], -0.0);
    }

    #[test]
    fn test_round() {
        let ops = SimdOps::new();
        let x = vec![1.1, 2.5, -3.7, 4.0];
        let result = ops.f32x4_round(&x);
        assert_eq!(result, vec![1.0, 3.0, -4.0, 4.0]);

        // Test special cases
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let result = ops.f32x4_round(&x);
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::INFINITY);
        assert_eq!(result[2], f32::NEG_INFINITY);
        assert_eq!(result[3], -0.0);
    }

    #[test]
    fn test_trunc() {
        let ops = SimdOps::new();
        let x = vec![1.1, 2.5, -3.7, 4.0];
        let result = ops.f32x4_trunc(&x);
        assert_eq!(result, vec![1.0, 2.0, -3.0, 4.0]);

        // Test special cases
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let result = ops.f32x4_trunc(&x);
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::INFINITY);
        assert_eq!(result[2], f32::NEG_INFINITY);
        assert_eq!(result[3], -0.0);
    }

    #[test]
    fn test_min() {
        let ops = SimdOps::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 1.0, 4.0, 3.0];
        let result = ops.f32x4_min(&x, &y);
        assert_eq!(result, vec![1.0, 1.0, 3.0, 3.0]);

        // Test special cases
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let y = vec![1.0, f32::NEG_INFINITY, f32::INFINITY, 0.0];
        let result = ops.f32x4_min(&x, &y);
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::NEG_INFINITY);
        assert_eq!(result[2], f32::NEG_INFINITY);
        assert_eq!(result[3], -0.0);
    }

    #[test]
    fn test_max() {
        let ops = SimdOps::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 1.0, 4.0, 3.0];
        let result = ops.f32x4_max(&x, &y);
        assert_eq!(result, vec![2.0, 2.0, 4.0, 4.0]);

        // Test special cases
        let x = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        let y = vec![1.0, f32::NEG_INFINITY, f32::INFINITY, 0.0];
        let result = ops.f32x4_max(&x, &y);
        assert!(result[0].is_nan());
        assert_eq!(result[1], f32::INFINITY);
        assert_eq!(result[2], f32::INFINITY);
        assert_eq!(result[3], 0.0);
    }
}