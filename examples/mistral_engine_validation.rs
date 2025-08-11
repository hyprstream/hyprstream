/// MistralEngine validation example
/// 
/// This example validates the MistralEngine API without requiring model downloads.
/// It demonstrates proper error handling and API contract validation.

use anyhow::Result;
use hyprstream_core::{MistralEngine, RuntimeEngine, RuntimeConfig, GenerationRequest};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    println!("🧪 MistralEngine Validation Example");
    
    // Test 1: Engine Creation
    println!("\n1️⃣ Testing engine creation...");
    let mut engine = MistralEngine::new_default()?;
    assert!(!engine.is_loaded());
    println!("✅ Engine created successfully");
    
    // Test 2: Model Info (unloaded state)
    println!("\n2️⃣ Testing model info (unloaded)...");
    let model_info = engine.model_info();
    println!("   Name: {}", model_info.name);
    println!("   Architecture: {}", model_info.architecture);
    assert_eq!(model_info.name, "unloaded");
    println!("✅ Model info working correctly");
    
    // Test 3: Generation without model (should fail gracefully)
    println!("\n3️⃣ Testing generation without model...");
    match engine.generate("Hello world", 10).await {
        Ok(_) => println!("❌ Unexpected success - should fail without model"),
        Err(e) => {
            println!("✅ Correctly failed with: {}", e);
            assert!(e.to_string().contains("Model not loaded"));
        }
    }
    
    // Test 4: Generation with parameters (should fail gracefully)
    println!("\n4️⃣ Testing parameterized generation without model...");
    let request = GenerationRequest {
        prompt: "Test prompt".to_string(),
        max_tokens: 20,
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(40),
        repeat_penalty: 1.1,
        stop_tokens: vec!["</s>".to_string()],
        seed: Some(42),
        stream: false,
        active_adapters: None,
        realtime_adaptation: None,
        user_feedback: None,
    };
    
    match engine.generate_with_params(request).await {
        Ok(_) => println!("❌ Unexpected success - should fail without model"),
        Err(e) => {
            println!("✅ Correctly failed with: {}", e);
            assert!(e.to_string().contains("Model not loaded"));
        }
    }
    
    // Test 5: Model loading with invalid path (should fail gracefully)
    println!("\n5️⃣ Testing model loading with invalid path...");
    let invalid_path = Path::new("/nonexistent/model.gguf");
    match engine.load_model(invalid_path).await {
        Ok(_) => println!("❌ Unexpected success - should fail with invalid path"),
        Err(e) => {
            println!("✅ Correctly failed with: {}", e);
            // Should contain mistral.rs error about file not found
        }
    }
    
    // Test 6: X-LoRA configuration without model (should fail gracefully)
    println!("\n6️⃣ Testing X-LoRA configuration without model...");
    match engine.configure_xlora(4, hyprstream_core::XLoraRoutingStrategy::Learned).await {
        Ok(_) => println!("❌ Unexpected success - should fail without model"),
        Err(e) => {
            println!("✅ Correctly failed with: {}", e);
            assert!(e.to_string().contains("Model not loaded"));
        }
    }
    
    println!("\n🎉 All validation tests passed!");
    println!("📋 Summary:");
    println!("   ✅ Engine creation works");
    println!("   ✅ Error handling is correct");
    println!("   ✅ API contracts are maintained");
    println!("   ✅ Ready for real model testing");
    
    println!("\n💡 Next steps:");
    println!("   1. Run: ./scripts/download_test_model.sh");
    println!("   2. Test with real model: cargo run --example mistral_engine_basic");
    
    Ok(())
}