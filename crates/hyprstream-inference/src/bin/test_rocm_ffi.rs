// Direct FFI test bypassing tch-rs wrappers
// Test binary intentionally prints diagnostic output
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::os::raw::c_int;

#[link(name = "torch")]
extern "C" {
    // Direct binding to PyTorch C++ API
    #[link_name = "_ZN5torch4cuda12is_availableEv"]
    fn torch_cuda_is_available() -> bool;

    #[link_name = "_ZN5torch4cuda12device_countEv"]
    fn torch_cuda_device_count() -> c_int;
}

fn main() {
    println!("ROCm Direct FFI Test");
    println!("====================");

    // Set ROCm environment — use /opt/rocm (standard install path); let runtime
    // auto-detect PYTORCH_ROCM_ARCH. The hardcoded gfx90a (MI210) caused silent
    // CPU fallback on gfx1151 (Strix Halo / Radeon 8060S) (#228).
    std::env::set_var("ROCM_PATH", "/opt/rocm");
    std::env::set_var("HIP_VISIBLE_DEVICES", "0");

    unsafe {
        let available = torch_cuda_is_available();
        let count = torch_cuda_device_count();

        println!("GPU Available (direct FFI): {available}");
        println!("Device Count (direct FFI): {count}");

        if available {
            println!("✅ ROCm GPU detected via direct FFI!");
        } else {
            println!("❌ No GPU detected");
        }
    }

    // Now compare with tch-rs
    println!("\nComparing with tch-rs:");
    let device = tch::Device::cuda_if_available();
    match device {
        tch::Device::Cpu => println!("tch-rs: CPU (not detecting GPU)"),
        tch::Device::Cuda(n) => println!("tch-rs: GPU {n} detected"),
        _ => println!("tch-rs: Other device"),
    }
}
