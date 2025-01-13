use std::arch::x86_64::*;
use crate::test_utils::{check_simd_alignment, validate_simd_operations};

#[cfg(target_arch = "x86_64")]
mod simd_tests {
    use super::*;

    // Test SIMD alignment requirements
    #[test]
    fn test_vector_alignment() {
        let aligned_data: Vec<f32> = vec![1.0; 32];
        assert!(check_simd_alignment(&aligned_data), "Data should be properly aligned for SIMD");
        
        // Test with different sizes
        let data_256: Vec<f32> = vec![1.0; 256];
        assert!(check_simd_alignment(&data_256), "256-element vector should be aligned");
    }

    // Test basic vector operations
    #[test]
    fn test_vector_operations() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test on unsupported hardware");
            return;
        }

        unsafe {
            // Prepare aligned test data
            let mut a = vec![1.0f32; 8];
            let mut b = vec![2.0f32; 8];
            let mut result = vec![0.0f32; 8];

            // Perform SIMD addition
            let va = _mm256_loadu_ps(a.as_ptr());
            let vb = _mm256_loadu_ps(b.as_ptr());
            let vresult = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr(), vresult);

            // Verify results
            for val in result.iter() {
                assert_eq!(*val, 3.0, "SIMD addition should yield 3.0");
            }
        }
    }

    // Test batch processing
    #[test]
    fn test_batch_processing() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test on unsupported hardware");
            return;
        }

        unsafe {
            let size = 1024;
            let mut data = vec![1.0f32; size];
            let mut result = vec![0.0f32; size];
            
            // Process in batches of 8 (AVX2 register size for f32)
            for i in (0..size).step_by(8) {
                let v = _mm256_loadu_ps(data[i..].as_ptr());
                let doubled = _mm256_add_ps(v, v);  // Multiply by 2
                _mm256_storeu_ps(&mut result[i] as *mut f32, doubled);
            }

            // Verify results
            for val in result.iter() {
                assert_eq!(*val, 2.0, "Batch processing should double values");
            }
        }
    }

    // Test different data types
    #[test]
    fn test_data_types() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test on unsupported hardware");
            return;
        }

        unsafe {
            // Test i32
            let mut int_data = vec![1i32; 8];
            let int_result = {
                let v = _mm256_loadu_si256(int_data.as_ptr() as *const __m256i);
                let doubled = _mm256_add_epi32(v, v);
                let mut result = vec![0i32; 8];
                _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, doubled);
                result
            };
            for val in int_result.iter() {
                assert_eq!(*val, 2, "Integer SIMD operation should double values");
            }

            // Test f64
            let mut double_data = vec![1.0f64; 4];
            let double_result = {
                let v = _mm256_loadu_pd(double_data.as_ptr());
                let doubled = _mm256_add_pd(v, v);
                let mut result = vec![0.0f64; 4];
                _mm256_storeu_pd(result.as_mut_ptr(), doubled);
                result
            };
            for val in double_result.iter() {
                assert_eq!(*val, 2.0, "Double SIMD operation should double values");
            }
        }
    }

    // Test SIMD validation helper
    #[test]
    fn test_simd_validation() {
        let data: Vec<f32> = vec![1.0; 32];
        assert!(validate_simd_operations(&data), "SIMD validation should pass for aligned data");
    }
}
