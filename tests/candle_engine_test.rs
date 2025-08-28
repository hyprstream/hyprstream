//! Integration tests for Candle engine

use hyprstream_core::runtime::{CandleEngine, RuntimeEngine, RuntimeConfig, GenerationRequest};
use std::path::PathBuf;
use tempfile::TempDir;

/// Create test runtime config
fn create_test_config() -> RuntimeConfig {
    RuntimeConfig {
        use_gpu: false, // CPU for testing
        cpu_threads: Some(2),
        context_length: 512,
        batch_size: 1,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
    }
}

#[tokio::test]
async fn test_engine_creation() {
    let config = create_test_config();
    let engine = CandleEngine::new(config);
    assert!(engine.is_ok(), "Failed to create Candle engine");
    
    let engine = engine.unwrap();
    assert!(!engine.is_loaded(), "Engine should not have a model loaded initially");
}

#[tokio::test]
async fn test_model_info_unloaded() {
    let config = create_test_config();
    let engine = CandleEngine::new(config).unwrap();
    
    let info = engine.model_info();
    assert_eq!(info.name, "unloaded");
    assert_eq!(info.parameters, 0);
    assert_eq!(info.context_length, 0);
}

#[tokio::test]
async fn test_generate_without_model() {
    let config = create_test_config();
    let engine = CandleEngine::new(config).unwrap();
    
    // Should work with placeholder implementation
    let result = engine.generate("Hello", 10).await;
    assert!(result.is_ok(), "Generate should work even without model (placeholder)");
}

#[tokio::test]
async fn test_generation_request() {
    let config = create_test_config();
    let engine = CandleEngine::new(config).unwrap();
    
    let request = GenerationRequest {
        prompt: "Test prompt".to_string(),
        max_tokens: 50,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.1,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_tokens: vec![],
        seed: None,
        stream: false,
        active_adapters: None,
        realtime_adaptation: None,
        user_feedback: None,
    };
    
    let result = engine.generate_with_params(request).await;
    assert!(result.is_ok(), "Failed to generate with params");
    
    let result = result.unwrap();
    assert!(!result.text.is_empty(), "Generated text should not be empty");
}

#[tokio::test]
async fn test_tokenization_fallback() {
    let config = create_test_config();
    let engine = CandleEngine::new(config).unwrap();
    
    // Test with ASCII text
    let text = "Hello World!";
    let result = engine.generate(text, 5).await;
    assert!(result.is_ok(), "Failed to generate with fallback tokenization");
}

#[tokio::test]
async fn test_enable_realtime_adaptation() {
    use hyprstream_core::runtime::AdaptationMode;
    
    let config = create_test_config();
    let mut engine = CandleEngine::new(config).unwrap();
    
    let mode = AdaptationMode::Disabled;
    let result = engine.enable_realtime_adaptation(mode).await;
    assert!(result.is_ok(), "Failed to enable realtime adaptation");
}

#[tokio::test]
async fn test_update_adapter_realtime() {
    use hyprstream_core::adapters::lora_checkpoints::LoRAWeightsData;
    
    let config = create_test_config();
    let mut engine = CandleEngine::new(config).unwrap();
    
    let weights = LoRAWeightsData {
        a_weights: std::collections::HashMap::new(),
        b_weights: std::collections::HashMap::new(),
        alpha: 16.0,
        r: 8,
    };
    
    let result = engine.update_adapter_realtime("test_adapter", &weights).await;
    assert!(result.is_ok(), "Failed to update adapter");
}

#[tokio::test]
async fn test_model_loading_placeholder() {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.safetensors");
    
    // Create a dummy file (actual SafeTensors loading would fail, but that's ok for this test)
    std::fs::write(&model_path, b"dummy model data").unwrap();
    
    let config = create_test_config();
    let mut engine = CandleEngine::new(config).unwrap();
    
    // This will fail with the dummy file, but we're testing the interface
    let result = engine.load_model(&model_path).await;
    assert!(result.is_err(), "Should fail with dummy model file");
}

#[tokio::test]
async fn test_concurrent_generation() {
    let config = create_test_config();
    let engine = std::sync::Arc::new(CandleEngine::new(config).unwrap());
    
    let mut handles = Vec::new();
    
    // Spawn multiple concurrent generation tasks
    for i in 0..3 {
        let engine_clone = engine.clone();
        let prompt = format!("Test prompt {}", i);
        
        let handle = tokio::spawn(async move {
            let result = engine_clone.generate(&prompt, 10).await;
            assert!(result.is_ok(), "Task {} failed to generate", i);
            result.unwrap()
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        results.push(result);
    }
    
    // Verify all tasks produced output
    assert_eq!(results.len(), 3);
    for result in results {
        assert!(!result.is_empty());
    }
}

#[tokio::test]
async fn test_generation_with_context_limit() {
    let config = create_test_config();
    let engine = CandleEngine::new(config).unwrap();
    
    // Generate with a very long prompt to test context windowing
    let long_prompt = "Lorem ipsum ".repeat(500); // Very long prompt
    
    let result = engine.generate(&long_prompt, 20).await;
    assert!(result.is_ok(), "Should handle long prompts with windowing");
}

#[tokio::test]
async fn test_eos_detection() {
    let config = create_test_config();
    let engine = CandleEngine::new(config).unwrap();
    
    // The placeholder implementation should stop at EOS
    let result = engine.generate("Test", 1000).await;
    assert!(result.is_ok(), "Generation should complete");
    
    // Due to our simple implementation, it should stop before max_tokens
    let text = result.unwrap();
    assert!(text.len() < 10000, "Should stop before generating too much");
}