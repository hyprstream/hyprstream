//! HyprStream server binary for testing the PyTorch engine

use anyhow::Result;
use hyprstream_core::{HyprStreamEngine, RuntimeConfig};
use tokio::time::Instant;
use tracing::{info, error, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting HyprStream PyTorch server");
    
    // Test engine creation (will fail without model, which is expected)
    test_engine_lifecycle().await?;
    
    Ok(())
}

async fn test_engine_lifecycle() -> Result<()> {
    info!("Testing engine lifecycle...");
    
    // This should fail gracefully if no model is available
    let config = RuntimeConfig::default();
    let result = HyprStreamEngine::new(config);
    
    match result {
        Ok(_engine) => {
            info!("✅ Engine created successfully");
            info!("🎯 TorchEngine ready for model loading");
        }
        Err(e) => {
            warn!("⚠️  Engine creation failed (expected without model): {}", e);
            info!("To test with a real model:");
            info!("  1. Convert a model: python scripts/convert_model.py");
            info!("  2. Set LIBTORCH=/opt/libtorch");
            info!("  3. Run: RUST_LOG=info cargo run --bin hyprstream-server");
        }
    }
    
    Ok(())
}

