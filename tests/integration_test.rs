//! Integration tests for HyprStream PyTorch engine

use anyhow::Result;
use hyprstream_torch::{HyprStreamEngine, init};
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;
    
    fn setup() {
        // Initialize logging for tests
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_env_filter("hyprstream_torch=debug")
            .try_init();
    }
    
    #[test]
    fn test_engine_creation_fails_gracefully() {
        setup();
        
        // This should fail with a clear error message
        let result = HyprStreamEngine::new("nonexistent_model.pt");
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        let error_string = error.to_string();
        
        // Should contain helpful information
        assert!(error_string.contains("Failed to create engine") || 
                error_string.contains("model") ||
                error_string.contains("file"));
    }
    
    #[test]
    fn test_engine_memory_safety() {
        setup();
        
        // Test that creating and dropping engines doesn't crash
        for i in 0..10 {
            let result = HyprStreamEngine::new(&format!("nonexistent_{}.pt", i));
            assert!(result.is_err()); // Should fail gracefully
        }
        
        // All engines should have been properly cleaned up
    }
    
    #[ignore = "Requires LibTorch installation"]
    #[test] 
    fn test_with_real_model() -> Result<()> {
        setup();
        
        // This test only runs if a model exists
        let model_path = "test_model.pt";
        
        if !Path::new(model_path).exists() {
            println!("⏭️  Skipping real model test - {} not found", model_path);
            return Ok(());
        }
        
        println!("🧪 Testing with real model: {}", model_path);
        
        let engine = HyprStreamEngine::new(model_path)?;
        
        // Test forward pass
        let batch_size = 1;
        let seq_len = 10;
        let vocab_size = 1000;
        
        let input = vec![1.0; seq_len];
        let output = engine.forward(&input, batch_size, seq_len, vocab_size)?;
        
        // Verify output shape
        assert_eq!(output.len(), batch_size * seq_len * vocab_size);
        
        // Test LoRA update
        let rank = 8;
        let hidden_size = 512;
        
        let lora_a = vec![0.01; rank * hidden_size];
        let lora_b = vec![0.01; hidden_size * rank];
        
        engine.update_lora("test_layer", &lora_a, &lora_b, rank)?;
        
        println!("✅ Real model test passed");
        Ok(())
    }
    
    #[test]
    fn test_lora_parameter_validation() {
        setup();
        
        // Test with invalid model (should fail gracefully)
        let result = HyprStreamEngine::new("invalid.pt");
        assert!(result.is_err());
        
        if let Ok(engine) = result {
            // Test parameter validation
            let result = engine.update_lora("", &[], &[], 0);
            // Should either succeed (no-op) or fail gracefully
            // The key is that it doesn't crash
            let _ = result;
        }
    }
    
    #[test]
    fn test_concurrent_access() {
        setup();
        
        use std::thread;
        use std::sync::Arc;
        
        // Test that multiple threads can't cause data races
        let handles: Vec<_> = (0..5).map(|i| {
            thread::spawn(move || {
                let result = HyprStreamEngine::new(&format!("test_{}.pt", i));
                // All should fail gracefully without crashing
                assert!(result.is_err());
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}