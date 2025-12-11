# bitsandbytes-sys

Rust FFI bindings for the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) quantization library.

## Features

- **8-bit blockwise quantization** (LLM.int8())
- **4-bit quantization** (NF4, FP4)
- **Quantized matrix multiplication** (`cgemm_4bit_inference`)
- Support for **CUDA**, **ROCm/HIP**, and **CPU** backends

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bitsandbytes-sys = { path = "../bitsandbytes-sys" }
```

### Prerequisites

You need the bitsandbytes library installed. Install via pip:

```bash
pip install bitsandbytes
```

Or build from source for ROCm:

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes
cd bitsandbytes
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a;gfx942;gfx1100" -S . -B build
cmake --build build
```

Set the library path:

```bash
export BITSANDBYTES_LIB_PATH=/path/to/libbitsandbytes_rocm70.so
export LD_LIBRARY_PATH=/path/to/library/dir:$LD_LIBRARY_PATH
```

## Usage

```rust
use bitsandbytes_sys::{
    quantize_blockwise_fp32, dequantize_blockwise_fp32,
    quantize_4bit_fp32, dequantize_4bit_fp32,
    quantize_blockwise_cpu_fp32, dequantize_blockwise_cpu_fp32,
    QuantType, is_available, get_backend,
};

// Check availability
println!("Available: {}, Backend: {}", is_available(), get_backend());

// 8-bit quantization (GPU)
let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, /* ... */];
let (quantized, state) = quantize_blockwise_fp32(&input, 64)?;
let dequantized = dequantize_blockwise_fp32(&quantized, &state)?;

// 8-bit quantization (CPU fallback)
let (quantized, state) = quantize_blockwise_cpu_fp32(&input, 64)?;
let dequantized = dequantize_blockwise_cpu_fp32(&quantized, &state)?;

// 4-bit NF4 quantization
let (quantized, state) = quantize_4bit_fp32(&input, 64, QuantType::Nf4)?;
let dequantized = dequantize_4bit_fp32(&quantized, &state)?;

// 4-bit FP4 quantization
let (quantized, state) = quantize_4bit_fp32(&input, 64, QuantType::Fp4)?;
let dequantized = dequantize_4bit_fp32(&quantized, &state)?;
```

## Backend Selection

The backend is auto-detected based on environment variables:

- `ROCM_PATH` or `HIP_PATH` → HIP backend
- `CUDA_HOME` or `CUDA_PATH` → CUDA backend
- Otherwise → CPU backend

Override with Cargo features:

```bash
cargo build --features cuda
cargo build --features hip
cargo build --features cpu
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BITSANDBYTES_LIB_PATH` | Direct path to libbitsandbytes.so |
| `BITSANDBYTES_BUILD_DIR` | Build directory to search |
| `ROCM_PATH` | ROCm installation path |
| `HIP_PATH` | HIP installation path |
| `CUDA_HOME` | CUDA installation path |

## Available Functions

### High-Level Safe Wrappers

| Function | Description |
|----------|-------------|
| `quantize_blockwise_fp32` | 8-bit blockwise quantization (GPU) |
| `dequantize_blockwise_fp32` | 8-bit blockwise dequantization (GPU) |
| `quantize_4bit_fp32` | Generic 4-bit quantization (NF4/FP4) |
| `dequantize_4bit_fp32` | Generic 4-bit dequantization |
| `quantize_4bit_nf4_fp32` | NF4-specific 4-bit quantization |
| `quantize_4bit_fp4_fp32` | FP4-specific 4-bit quantization |
| `quantize_blockwise_cpu_fp32` | 8-bit blockwise quantization (CPU) |
| `dequantize_blockwise_cpu_fp32` | 8-bit blockwise dequantization (CPU) |

### Low-Level FFI (re-exported from bindings)

All C functions from bitsandbytes are available with their original `c` prefix:
- `cquantize_blockwise_fp32`, `cdequantize_blockwise_fp32`
- `cquantize_blockwise_fp32_nf4`, `cdequantize_blockwise_fp32_nf4`
- `cquantize_blockwise_fp32_fp4`, `cdequantize_blockwise_fp32_fp4`
- `cgemm_4bit_inference`, `cgemm_4bit_inference_naive_*`
- `cint8_vector_quant`, `cdequant_mm_int32_fp16`
- And more (see `src/bindings.rs`)

## Memory Savings

| Format | Bits | Memory Reduction | Use Case |
|--------|------|------------------|----------|
| INT8 | 8 | 50% vs FP16 | LLM.int8() inference |
| NF4 | 4 | 75% vs FP16 | QLoRA, weight quantization |
| FP4 | 4 | 75% vs FP16 | General quantization |

## Running Examples

```bash
# Test the CPU quantization functions
BITSANDBYTES_LIB_PATH=/path/to/libbitsandbytes_rocm70.so \
LD_LIBRARY_PATH=/path/to/library/dir:$LD_LIBRARY_PATH \
cargo run -p bitsandbytes-sys --example test_quantize
```

## License

MIT License
