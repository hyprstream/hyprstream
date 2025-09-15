// Direct FFI test bypassing tch-rs wrappers
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
    
    // Set ROCm environment
    std::env::set_var("ROCM_PATH", "/usr");
    std::env::set_var("HIP_VISIBLE_DEVICES", "0");
    std::env::set_var("PYTORCH_ROCM_ARCH", "gfx90a");
    
    unsafe {
        let available = torch_cuda_is_available();
        let count = torch_cuda_device_count();
        
        println!("GPU Available (direct FFI): {}", available);
        println!("Device Count (direct FFI): {}", count);
        
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
        tch::Device::Cuda(n) => println!("tch-rs: GPU {} detected", n),
        _ => println!("tch-rs: Other device"),
    }
}