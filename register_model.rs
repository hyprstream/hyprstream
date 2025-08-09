use std::path::Path;
use hyprstream_core::cli::commands::download::register_existing_model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("/private/birdetta/.local/share/hyprstream/models/Qwen2-1.5B-Instruct-GGUF_qwen2-1_5b-instruct-fp16.gguf");
    let model_uri = "Qwen/Qwen2-1.5B-Instruct-GGUF:fp16";
    
    println!("🔧 Registering existing model...");
    println!("📁 Path: {}", model_path.display());
    println!("🔗 URI: {}", model_uri);
    
    match register_existing_model(model_path, model_uri, None).await {
        Ok(()) => {
            println!("✅ Model registered successfully!");
        }
        Err(e) => {
            println!("❌ Failed to register model: {}", e);
        }
    }
    
    Ok(())
}