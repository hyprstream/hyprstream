/// Test the refactored weight loading through ModelFactory
use anyhow::Result;
use hyprstream_core::runtime::{TorchEngine, RuntimeConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing refactored weight loading...\n");
    
    // Create engine with CPU config
    let config = RuntimeConfig {
        use_gpu: false,
        ..Default::default()
    };
    
    let mut engine = TorchEngine::new(config)?;
    println!("✅ Engine created");
    
    // Test model path - you'll need to adjust this to a real model path
    let model_path = Path::new("/private/birdetta/.local/share/hyprstream/models/hf/Qwen_Qwen3-4B-Instruct-2507/model-00001-of-00003.safetensors");
    
    if !model_path.exists() {
        println!("❌ Model path does not exist: {}", model_path.display());
        println!("Please update the path in the test to point to a real model");
        return Ok(());
    }
    
    println!("📦 Loading model from: {}", model_path.display());
    
    // Load the model - this should now use ModelFactory internally
    match engine.load_model(model_path).await {
        Ok(_) => {
            println!("✅ Model loaded successfully!");
            println!("🎉 Weight loading refactor is working correctly");
            
            // Test inference
            println!("\n🧪 Testing inference...");
            let result = engine.generate("Hello, world!", 10).await?;
            println!("Generated: {}", result);
        }
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}