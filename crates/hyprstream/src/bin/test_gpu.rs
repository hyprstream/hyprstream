// Test binary intentionally prints diagnostic output
#![allow(clippy::print_stdout, clippy::print_stderr)]

use tch::Device;

fn main() {
    println!("ROCm GPU Detection Test");
    println!("=======================");

    // Set environment for ROCm — use system ROCm path, let HSA select the correct arch natively
    std::env::set_var("ROCM_PATH", "/opt/rocm");
    std::env::set_var("HIP_VISIBLE_DEVICES", "0");
    // HSA_OVERRIDE_GFX_VERSION intentionally not set: let ROCm detect the actual GPU arch.
    // Setting it to a fixed value (e.g. 9.0.0 for gfx90a/MI210) breaks non-MI210 hardware
    // such as gfx1151 (Strix Halo) by loading the wrong ISA kernels.

    println!("Environment set for ROCm (auto-detecting GPU arch)");

    // Check device
    let device = Device::cuda_if_available();

    match device {
        Device::Cpu => {
            println!("Result: CPU mode");
            println!("\nDebug: Trying to force GPU...");

            // Try to force GPU allocation
            match std::panic::catch_unwind(|| {
                let _tensor = tch::Tensor::zeros([1], (tch::Kind::Float, Device::Cuda(0)));
                println!("Force GPU: Success!");
            }) {
                Ok(_) => {}
                Err(_) => println!("Force GPU: Failed - GPU not accessible"),
            }
        }
        Device::Cuda(n) => {
            println!("Result: ✅ GPU {n} detected!");
        }
        _ => {
            println!("Result: Other device type detected");
        }
    }
}
