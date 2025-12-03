/**
 * bitsandbytes-sys wrapper header
 *
 * C-compatible declarations for bitsandbytes quantization library.
 * Function names use 'c' prefix matching the actual exported symbols.
 */

#ifndef BITSANDBYTES_SYS_WRAPPER_H
#define BITSANDBYTES_SYS_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stream handle (opaque pointer to CUDA/HIP stream)
 */
typedef void* bnb_stream_t;

/* ============================================================================
 * 8-bit Blockwise Quantization
 * ============================================================================ */

/**
 * Quantize float32 tensor to 8-bit using blockwise quantization
 *
 * @param code       Quantization codebook (256 float values)
 * @param A          Input tensor (float32)
 * @param absmax     Output: per-block absolute maximum values
 * @param out        Output: quantized 8-bit values
 * @param blocksize  Block size (typically 4096)
 * @param n          Total number of elements
 */
void cquantize_blockwise_fp32(
    float* code,
    float* A,
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cquantize_blockwise_fp16(
    float* code,
    void* A,  /* half* */
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cquantize_blockwise_bf16(
    float* code,
    void* A,  /* bfloat16* */
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

/**
 * Dequantize 8-bit values back to float32
 */
void cdequantize_blockwise_fp32(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n,
    bnb_stream_t stream
);

void cdequantize_blockwise_fp16(
    float* code,
    unsigned char* A,
    float* absmax,
    void* out,  /* half* */
    int blocksize,
    int n,
    bnb_stream_t stream
);

void cdequantize_blockwise_bf16(
    float* code,
    unsigned char* A,
    float* absmax,
    void* out,  /* bfloat16* */
    int blocksize,
    int n,
    bnb_stream_t stream
);

/* ============================================================================
 * 4-bit NF4 Quantization (Normalized Float 4-bit)
 * ============================================================================ */

void cquantize_blockwise_fp32_nf4(
    float* code,
    float* A,
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cquantize_blockwise_fp16_nf4(
    float* code,
    void* A,  /* half* */
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cquantize_blockwise_bf16_nf4(
    float* code,
    void* A,  /* bfloat16* */
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cdequantize_blockwise_fp32_nf4(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n,
    bnb_stream_t stream
);

void cdequantize_blockwise_fp16_nf4(
    float* code,
    unsigned char* A,
    float* absmax,
    void* out,  /* half* */
    int blocksize,
    int n,
    bnb_stream_t stream
);

void cdequantize_blockwise_bf16_nf4(
    float* code,
    unsigned char* A,
    float* absmax,
    void* out,  /* bfloat16* */
    int blocksize,
    int n,
    bnb_stream_t stream
);

/* ============================================================================
 * 4-bit FP4 Quantization (Float Point 4-bit)
 * ============================================================================ */

void cquantize_blockwise_fp32_fp4(
    float* code,
    float* A,
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cquantize_blockwise_fp16_fp4(
    float* code,
    void* A,  /* half* */
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cquantize_blockwise_bf16_fp4(
    float* code,
    void* A,  /* bfloat16* */
    float* absmax,
    unsigned char* out,
    int blocksize,
    int n
);

void cdequantize_blockwise_fp32_fp4(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n,
    bnb_stream_t stream
);

void cdequantize_blockwise_fp16_fp4(
    float* code,
    unsigned char* A,
    float* absmax,
    void* out,  /* half* */
    int blocksize,
    int n,
    bnb_stream_t stream
);

void cdequantize_blockwise_bf16_fp4(
    float* code,
    unsigned char* A,
    float* absmax,
    void* out,  /* bfloat16* */
    int blocksize,
    int n,
    bnb_stream_t stream
);

/* ============================================================================
 * Basic Quantization (non-blockwise)
 * ============================================================================ */

void cquantize(float* code, float* A, unsigned char* out, int n);
void cdequantize(float* code, unsigned char* A, float* out, int n, bnb_stream_t stream);

/* ============================================================================
 * Int8 Vector Quantization (LLM.int8())
 * ============================================================================ */

void cint8_vector_quant(
    void* A,  /* half* */
    signed char* out,
    float* row_stats,
    float threshold,
    int rows,
    int cols,
    bnb_stream_t stream
);

void cdequant_mm_int32_fp16(
    int* A,
    float* row_stats,
    float* col_stats,
    void* out,  /* half* */
    void* bias,  /* half* */
    int num_rows,
    int num_cols,
    bnb_stream_t stream
);

/* ============================================================================
 * 4-bit Matrix Multiplication
 * ============================================================================ */

void cgemm_4bit_inference(
    int m, int n, int k,
    void* A,  /* half* */
    unsigned char* B,
    float* absmax,
    void* datatype,
    void* out,  /* half* */
    int lda, int ldb, int ldc,
    int blocksize
);

void cgemm_4bit_inference_naive_fp16(
    int m, int n, int k,
    void* A,  /* half* */
    unsigned char* B,
    float* absmax,
    float* code,
    void* out,  /* half* */
    int blocksize
);

void cgemm_4bit_inference_naive_fp32(
    int m, int n, int k,
    float* A,
    unsigned char* B,
    float* absmax,
    float* code,
    float* out,
    int blocksize
);

void cgemm_4bit_inference_naive_bf16(
    int m, int n, int k,
    void* A,  /* bfloat16* */
    unsigned char* B,
    float* absmax,
    float* code,
    void* out,  /* bfloat16* */
    int blocksize
);

/* ============================================================================
 * CPU Quantization (fallback)
 * ============================================================================ */

void cquantize_blockwise_cpu_fp32(
    float* code,
    float* A,
    float* absmax,
    unsigned char* out,
    long long blocksize,
    long long n
);

void cdequantize_blockwise_cpu_fp32(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n
);

void cdequantize_blockwise_cpu_fp32_nf4(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n
);

void cdequantize_blockwise_cpu_fp32_fp4(
    float* code,
    unsigned char* A,
    float* absmax,
    float* out,
    int blocksize,
    int n
);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get managed CUDA/HIP memory pointer
 */
void* cget_managed_ptr(size_t num_bytes);

/**
 * Prefetch memory to device
 */
void cprefetch(void* ptr, size_t bytes, int device);

#ifdef __cplusplus
}
#endif

#endif /* BITSANDBYTES_SYS_WRAPPER_H */
