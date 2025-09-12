//! Basic inference engine test without VDB features

use hyprstream_core::inference::{
    InferenceEngine, InferenceConfig, InferenceInput, FusedAdapterWeights, FusionMetadata
};
use hyprstream_core::inference::model_loader::ModelLoader;
use std::collections::HashMap;
use tokio;

#[tokio::test]
async fn test_basic_inference_engine() -> anyhow::Result<()> {
    // Create inference engine with default config
    let config = InferenceConfig::default();
    let engine = InferenceEngine::new(config)?;
    
    // Create dummy model loader 
    let model_loader = ModelLoader::new(std::path::Path::new("./")).await?;
    
    // Create empty fused weights for testing
    let fused_weights = FusedAdapterWeights {
        weights: HashMap::new(),
        fusion_metadata: FusionMetadata {
            num_adapters: 0,
            total_sparse_weights: 0,
            fusion_strategy: "test".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        },
    };
    
    // Create test input
    let input = InferenceInput {
        prompt: Some("Hello world".to_string()),
        input_ids: None,
        max_tokens: 10,
        temperature: 1.0,
        top_p: 1.0,
        stream: false,
    };
    
    // Run inference (should fall back to mock implementation)
    let result = engine.infer(&model_loader, &fused_weights, input).await?;
    
    // Verify we got a response
    assert!(!result.text.is_empty());
    assert!(result.tokens_generated > 0);
    println!("✅ Inference test passed: {}", result.text);
    
    Ok(())
}

#[tokio::test]
async fn test_inference_engine_warmup() -> anyhow::Result<()> {
    let config = InferenceConfig::default();
    let engine = InferenceEngine::new(config)?;
    
    // Test warmup
    engine.warmup().await?;
    
    // Check statistics were updated
    let stats = engine.get_stats().await;
    assert!(stats.total_inferences > 0);
    println!("✅ Warmup test passed: {} inferences", stats.total_inferences);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_usage() -> anyhow::Result<()> {
    let config = InferenceConfig::default();
    let engine = InferenceEngine::new(config)?;
    
    let memory_usage = engine.get_memory_usage().await;
    println!("Memory usage: {:?}", memory_usage);
    
    // Basic validation
    assert!(memory_usage.cpu_threads_active > 0);
    assert!(memory_usage.kv_cache_limit_mb > 0);
    
    println!("✅ Memory usage test passed");
    Ok(())
}