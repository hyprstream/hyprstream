use tch::Device;

fn main() {
    println!("ROCm GPU Detection Test");
    println!("=======================");

    // Set environment for ROCm
    std::env::set_var("ROCM_PATH", "/usr");
    std::env::set_var("HIP_VISIBLE_DEVICES", "0");
    std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "9.0.0");

    println!("Environment set for ROCm MI210");

    // Check device
    let device = Device::cuda_if_available();

    match device {
        Device::Cpu => {
            println!("Result: CPU mode");
            println!("\nDebug: Trying to force GPU...");

            // Try to force GPU allocation
            match std::panic::catch_unwind(|| {
                let _tensor = tch::Tensor::zeros(&[1], (tch::Kind::Float, Device::Cuda(0)));
                println!("Force GPU: Success!");
            }) {
                Ok(_) => {}
                Err(_) => println!("Force GPU: Failed - GPU not accessible"),
            }
        }
        Device::Cuda(n) => {
            println!("Result: ✅ GPU {} detected!", n);
        }
        _ => {
            println!("Result: Other device type detected");
        }
    }
}
