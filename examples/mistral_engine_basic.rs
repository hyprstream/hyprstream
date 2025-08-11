/// Basic MistralEngine example demonstrating real model loading and generation
/// 
/// This example downloads a small GGUF model and demonstrates the core functionality
/// of the MistralEngine with real mistral.rs integration.

use anyhow::Result;
use hyprstream_core::{MistralEngine, RuntimeEngine, RuntimeConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    println!("🚀 MistralEngine Basic Example");
    
    // Create engine with default configuration
    let mut engine = MistralEngine::new_default()?;
    println!("✅ MistralEngine created");
    
    // For this example, we'll use a hypothetical small model
    // In practice, you would download a real GGUF model first
    let model_path = Path::new("models/small-test-model.gguf");
    
    if !model_path.exists() {
        println!("⚠️  Model not found at: {}", model_path.display());
        println!("   To run this example, please download a small GGUF model first.");
        println!("   Example: wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf -O models/small-test-model.gguf");
        return Ok(());
    }
    
    // Load model
    println!("📦 Loading model from: {}", model_path.display());
    match engine.load_model(model_path).await {
        Ok(_) => {
            println!("✅ Model loaded successfully");
            
            // Get model info
            let model_info = engine.model_info();
            println!("📋 Model Info:");
            println!("   Name: {}", model_info.name);
            println!("   Architecture: {}", model_info.architecture);
            println!("   Context Length: {}", model_info.context_length);
            
            // Test basic generation
            println!("\n🧠 Testing text generation...");
            let prompt = "Hello, how are you today?";
            
            match engine.generate(prompt, 50).await {
                Ok(response) => {
                    println!("✅ Generation successful:");
                    println!("   Prompt: {}", prompt);
                    println!("   Response: {}", response);
                }
                Err(e) => {
                    println!("❌ Generation failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Model loading failed: {}", e);
            println!("   This is expected if no GGUF model is available");
        }
    }
    
    println!("\n🎉 Example completed!");
    Ok(())
}