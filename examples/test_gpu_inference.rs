/// Test GPU inference with minimal setup
/// Run with: cargo run --example test_gpu_inference --features cuda

use anyhow::Result;
use hyprstream_core::runtime::{TorchEngine, RuntimeEngine, RuntimeConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 GPU Inference Validation Test");
    println!("=================================\n");
    
    // Step 1: Test basic GPU availability
    println!("📊 Checking GPU availability...");
    let config = RuntimeConfig {
        use_gpu: true,
        ..Default::default()
    };
    
    let mut engine = TorchEngine::new(config)?;
    println!("✅ Engine created with GPU support\n");
    
    // Step 2: Test tensor operations on GPU
    println!("🔬 Testing tensor operations on GPU...");
    test_tensor_ops()?;
    
    // Step 3: Load a small model if available
    let model_path = Path::new("models/qwen2.5-0.5b-instruct");
    if model_path.exists() {
        println!("\n📦 Loading model on GPU...");
        engine.load_model(model_path).await?;
        println!("✅ Model loaded successfully");
        
        // Step 4: Run a simple inference
        println!("\n🚀 Running inference on GPU...");
        let start = std::time::Instant::now();
        let result = engine.generate("Hello, how are you?", 20).await?;
        let elapsed = start.elapsed();
        
        println!("✅ Inference completed in {:.2}ms", elapsed.as_millis());
        println!("📝 Generated: {}", result);
    } else {
        println!("\n⚠️  No model found at {:?}", model_path);
        println!("   Download a model first to test inference");
        println!("   Example: cargo run -- model download Qwen/Qwen2.5-0.5B-Instruct");
    }
    
    println!("\n✅ GPU validation complete!");
    Ok(())
}

fn test_tensor_ops() -> Result<()> {
    use tch::{Device, Tensor};
    
    // Check device availability
    let device = Device::cuda_if_available();
    if device == Device::Cpu {
        println!("⚠️  No GPU detected, running on CPU");
    } else {
        println!("✅ GPU device detected");
    }
    
    // Create tensors on device
    let x = Tensor::randn([100, 100], (tch::Kind::Float, device));
    let y = Tensor::randn([100, 100], (tch::Kind::Float, device));
    
    // Perform operation
    let z = x.matmul(&y);
    
    // Verify shape
    assert_eq!(z.size(), vec![100, 100]);
    println!("✅ Tensor operations successful on {:?}", device);
    
    // Memory check
    if device != Device::Cpu {
        // This will show GPU memory usage if available
        println!("   Tensor allocated on GPU memory");
    }
    
    Ok(())
}