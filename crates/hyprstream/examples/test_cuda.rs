// Test program to check if tch-rs can see CUDA
use tch::{Device, Tensor};

fn main() {
    println!("=== tch-rs CUDA Test for Hyprstream ===\n");

    // Check if CUDA is available
    let cuda_available = tch::Cuda::is_available();
    println!("CUDA available: {}", cuda_available);

    if cuda_available {
        let device_count = tch::Cuda::device_count();
        println!("CUDA device count: {}", device_count);

        // Print cuDNN availability
        println!("cuDNN available: {}", tch::Cuda::cudnn_is_available());

        // Test each device
        for i in 0..device_count {
            println!("\n--- Device {} ---", i);
        }

        // Create tensors and perform operations
        println!("\n--- Testing CUDA Operations ---");

        // Create a tensor on CPU
        let cpu_tensor = Tensor::rand([5, 3], (tch::Kind::Float, Device::Cpu));
        println!("CPU Tensor:\n{}", cpu_tensor);

        // Move to GPU
        let gpu_tensor = cpu_tensor.to(Device::Cuda(0));
        println!("\nMoved to GPU (device 0)");

        // Perform computation on GPU
        let result = &gpu_tensor * 2.0;
        println!("\nGPU Tensor * 2:\n{}", result);

        // Test matrix multiplication
        let a = Tensor::rand([100, 100], (tch::Kind::Float, Device::Cuda(0)));
        let b = Tensor::rand([100, 100], (tch::Kind::Float, Device::Cuda(0)));
        let c = a.matmul(&b);
        println!("\nMatrix multiplication (100x100) successful!");
        println!("Result shape: {:?}", c.size());

        // Test multi-GPU if available
        if device_count > 1 {
            println!("\n--- Testing Multi-GPU ---");
            let gpu1_tensor = Tensor::rand([3, 3], (tch::Kind::Float, Device::Cuda(1)));
            println!("Created tensor on GPU 1:\n{}", gpu1_tensor);
        }

        println!("\n✓ All CUDA tests passed!");
    } else {
        println!("\n✗ CUDA is not available");
        println!("Make sure:");
        println!("1. NVIDIA drivers are installed");
        println!("2. CUDA toolkit is installed");
        println!("3. LIBTORCH with CUDA support is installed");
    }

    // Print environment info
    println!("\n--- Environment Variables ---");
    if let Ok(val) = std::env::var("LIBTORCH") {
        println!("LIBTORCH={}", val);
    } else {
        println!("LIBTORCH: not set");
    }
    if let Ok(val) = std::env::var("LD_LIBRARY_PATH") {
        println!("LD_LIBRARY_PATH={}", val);
    } else {
        println!("LD_LIBRARY_PATH: not set");
    }
    if let Ok(val) = std::env::var("CUDA_HOME") {
        println!("CUDA_HOME={}", val);
    } else {
        println!("CUDA_HOME: not set");
    }
}
