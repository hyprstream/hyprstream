//! FFI bindings for bitsandbytes
//!
//! These bindings match the actual exported symbols from libbitsandbytes.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_int, c_longlong, c_void};

/// Stream handle type (CUDA/HIP stream pointer)
pub type bnb_stream_t = *mut c_void;

extern "C" {
    // ========================================================================
    // 8-bit Blockwise Quantization
    // ========================================================================

    pub fn cquantize_blockwise_fp32(
        code: *mut f32,
        A: *mut f32,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cquantize_blockwise_fp16(
        code: *mut f32,
        A: *mut c_void,  // half*
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cquantize_blockwise_bf16(
        code: *mut f32,
        A: *mut c_void,  // bfloat16*
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cdequantize_blockwise_fp32(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut f32,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequantize_blockwise_fp16(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut c_void,  // half*
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequantize_blockwise_bf16(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut c_void,  // bfloat16*
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    // ========================================================================
    // 4-bit NF4 Quantization
    // ========================================================================

    pub fn cquantize_blockwise_fp32_nf4(
        code: *mut f32,
        A: *mut f32,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cquantize_blockwise_fp16_nf4(
        code: *mut f32,
        A: *mut c_void,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cquantize_blockwise_bf16_nf4(
        code: *mut f32,
        A: *mut c_void,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cdequantize_blockwise_fp32_nf4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut f32,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequantize_blockwise_fp16_nf4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut c_void,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequantize_blockwise_bf16_nf4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut c_void,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    // ========================================================================
    // 4-bit FP4 Quantization
    // ========================================================================

    pub fn cquantize_blockwise_fp32_fp4(
        code: *mut f32,
        A: *mut f32,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cquantize_blockwise_fp16_fp4(
        code: *mut f32,
        A: *mut c_void,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cquantize_blockwise_bf16_fp4(
        code: *mut f32,
        A: *mut c_void,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cdequantize_blockwise_fp32_fp4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut f32,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequantize_blockwise_fp16_fp4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut c_void,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequantize_blockwise_bf16_fp4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut c_void,
        blocksize: c_int,
        n: c_int,
        stream: bnb_stream_t,
    );

    // ========================================================================
    // Basic Quantization (non-blockwise)
    // ========================================================================

    pub fn cquantize(
        code: *mut f32,
        A: *mut f32,
        out: *mut u8,
        n: c_int,
    );

    pub fn cdequantize(
        code: *mut f32,
        A: *mut u8,
        out: *mut f32,
        n: c_int,
        stream: bnb_stream_t,
    );

    // ========================================================================
    // Int8 Vector Quantization (LLM.int8())
    // ========================================================================

    pub fn cint8_vector_quant(
        A: *mut c_void,  // half*
        out: *mut i8,
        row_stats: *mut f32,
        threshold: f32,
        rows: c_int,
        cols: c_int,
        stream: bnb_stream_t,
    );

    pub fn cdequant_mm_int32_fp16(
        A: *mut c_int,
        row_stats: *mut f32,
        col_stats: *mut f32,
        out: *mut c_void,  // half*
        bias: *mut c_void,  // half*
        num_rows: c_int,
        num_cols: c_int,
        stream: bnb_stream_t,
    );

    // ========================================================================
    // 4-bit Matrix Multiplication
    // ========================================================================

    pub fn cgemm_4bit_inference(
        m: c_int, n: c_int, k: c_int,
        A: *mut c_void,  // half*
        B: *mut u8,
        absmax: *mut f32,
        datatype: *mut c_void,
        out: *mut c_void,  // half*
        lda: c_int, ldb: c_int, ldc: c_int,
        blocksize: c_int,
    );

    pub fn cgemm_4bit_inference_naive_fp16(
        m: c_int, n: c_int, k: c_int,
        A: *mut c_void,  // half*
        B: *mut u8,
        absmax: *mut f32,
        code: *mut f32,
        out: *mut c_void,  // half*
        blocksize: c_int,
    );

    pub fn cgemm_4bit_inference_naive_fp32(
        m: c_int, n: c_int, k: c_int,
        A: *mut f32,
        B: *mut u8,
        absmax: *mut f32,
        code: *mut f32,
        out: *mut f32,
        blocksize: c_int,
    );

    pub fn cgemm_4bit_inference_naive_bf16(
        m: c_int, n: c_int, k: c_int,
        A: *mut c_void,  // bfloat16*
        B: *mut u8,
        absmax: *mut f32,
        code: *mut f32,
        out: *mut c_void,  // bfloat16*
        blocksize: c_int,
    );

    // ========================================================================
    // CPU Quantization (fallback)
    // ========================================================================

    pub fn cquantize_blockwise_cpu_fp32(
        code: *mut f32,
        A: *mut f32,
        absmax: *mut f32,
        out: *mut u8,
        blocksize: c_longlong,
        n: c_longlong,
    );

    pub fn cdequantize_blockwise_cpu_fp32(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut f32,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cdequantize_blockwise_cpu_fp32_nf4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut f32,
        blocksize: c_int,
        n: c_int,
    );

    pub fn cdequantize_blockwise_cpu_fp32_fp4(
        code: *mut f32,
        A: *mut u8,
        absmax: *mut f32,
        out: *mut f32,
        blocksize: c_int,
        n: c_int,
    );

    // ========================================================================
    // Utility Functions
    // ========================================================================

    pub fn cget_managed_ptr(num_bytes: usize) -> *mut c_void;
    pub fn cprefetch(ptr: *mut c_void, bytes: usize, device: c_int);
}
