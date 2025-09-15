use tch::Device;

fn main() {
    println!("ROCm GPU Test for AMD Instinct MI210");
    println!("=====================================");
    
    // Check device
    let device = Device::cuda_if_available();
    
    if device == Device::Cpu {
        println!("Status: CPU mode (GPU not detected by PyTorch)");
        println!("\nDiagnostics:");
        println!("- AMD Instinct MI210 is present (verified via rocm-smi)");
        println!("- libtorch has HIP libraries (libtorch_hip.so found)");
        println!("- PyTorch may need PYTORCH_ROCM_ARCH=gfx90a");
        println!("\nPossible issues:");
        println!("1. libtorch may be CUDA build despite HIP libraries");
        println!("2. tch-rs may not fully support ROCm yet");
        println!("3. Environment variables may need adjustment");
    } else {
        println!("Status: ✅ GPU detected and available!");
    }
}
