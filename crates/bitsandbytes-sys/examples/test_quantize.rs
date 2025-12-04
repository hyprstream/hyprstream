//! Test the bitsandbytes quantization functions

use bitsandbytes_sys::{
    dequantize_blockwise_cpu_fp32, get_backend, is_available, quantize_blockwise_cpu_fp32,
    QuantState, QuantType,
};

fn main() {
    println!("=== bitsandbytes-sys Test ===\n");

    // Check availability
    println!("Library available: {}", is_available());
    println!("Backend: {}", get_backend());
    println!();

    // Test data: create some float values
    let n = 4096;
    let blocksize = 64;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 100.0).collect();

    println!("Input data: {} elements", input.len());
    println!("First 10 values: {:?}", &input[..10]);
    println!("Block size: {}", blocksize);
    println!();

    // Test 8-bit quantization using CPU functions
    println!("=== 8-bit Blockwise Quantization (CPU) ===");
    match quantize_blockwise_cpu_fp32(&input, blocksize) {
        Ok((quantized, state)) => {
            println!("Quantization successful!");
            println!("Quantized data size: {} bytes", quantized.len());
            println!("Number of blocks: {}", state.n_blocks());
            println!("First 10 absmax values: {:?}", &state.absmax[..10.min(state.absmax.len())]);
            println!("First 10 quantized values: {:?}", &quantized[..10]);
            println!();

            // Dequantize
            match dequantize_blockwise_cpu_fp32(&quantized, &state) {
                Ok(dequantized) => {
                    println!("Dequantization successful!");
                    println!("Dequantized data size: {} elements", dequantized.len());
                    println!("First 10 dequantized values: {:?}", &dequantized[..10]);

                    // Calculate reconstruction error
                    let mse: f32 = input
                        .iter()
                        .zip(dequantized.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        / input.len() as f32;
                    let rmse = mse.sqrt();
                    println!("RMSE: {:.6}", rmse);

                    let max_error: f32 = input
                        .iter()
                        .zip(dequantized.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    println!("Max error: {:.6}", max_error);
                }
                Err(e) => {
                    println!("Dequantization failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Quantization failed: {}", e);
        }
    }

    println!();

    // Test QuantState creation
    println!("=== QuantState Tests ===");

    let state_8bit = QuantState::new_8bit(1024, 64);
    println!(
        "8-bit state: {} blocks, {} bytes quantized, codebook size: {}",
        state_8bit.n_blocks(),
        state_8bit.quantized_size(),
        state_8bit.code.len()
    );

    let state_nf4 = QuantState::new_4bit(1024, 64, QuantType::Nf4);
    println!(
        "NF4 state: {} blocks, {} bytes quantized, codebook size: {}",
        state_nf4.n_blocks(),
        state_nf4.quantized_size(),
        state_nf4.code.len()
    );
    println!("NF4 codebook: {:?}", state_nf4.code);

    let state_fp4 = QuantState::new_4bit(1024, 64, QuantType::Fp4);
    println!(
        "FP4 state: {} blocks, {} bytes quantized, codebook size: {}",
        state_fp4.n_blocks(),
        state_fp4.quantized_size(),
        state_fp4.code.len()
    );
    println!("FP4 codebook: {:?}", state_fp4.code);

    println!("\n=== Test Complete ===");
}
