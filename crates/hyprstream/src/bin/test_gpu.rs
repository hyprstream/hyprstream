// Test binary intentionally prints diagnostic output
#![allow(clippy::print_stdout, clippy::print_stderr)]

use tch::Device;

fn main() {
    println!("ROCm GPU Detection Test");
    println!("=======================");

    // Set environment for ROCm — use /opt/rocm (standard install path) and let
    // the runtime auto-detect the GPU ISA. HSA_OVERRIDE_GFX_VERSION=9.0.0 was
    // removed: it forced MI210/gfx90a kernels on all GPUs, breaking gfx1151
    // (Strix Halo / Radeon 8060S) by loading the wrong ISA and silently
    // falling back to CPU (#228).
    std::env::set_var("ROCM_PATH", "/opt/rocm");
    std::env::set_var("HIP_VISIBLE_DEVICES", "0");

    println!("Environment set for ROCm (auto-detect GPU architecture)");

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
