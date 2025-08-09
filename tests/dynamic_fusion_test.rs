//! Test dynamic LoRA fusion implementation

use hyprstream::inference::lora_fusion::{LoRAFusion, FusionStrategy};
use hyprstream::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig};
use std::collections::HashMap;

#[tokio::test]
async fn test_dynamic_attention_fusion() -> anyhow::Result<()> {
    println!("🧪 Testing dynamic LoRA fusion with attention mechanism");
    
    // Create fusion engine with attention strategy
    let mut fusion_engine = LoRAFusion::new(FusionStrategy::AttentionFusion);
    
    // Create test adapters
    let config1 = SparseLoRAConfig {
        rank: 8,
        sparsity: 0.95,
        alpha: 16.0,
        ..Default::default()
    };
    
    let config2 = SparseLoRAConfig {
        rank: 8,
        sparsity: 0.98,
        alpha: 32.0,
        ..Default::default()
    };
    
    let adapter1 = SparseLoRAAdapter::new(config1);
    let adapter2 = SparseLoRAAdapter::new(config2);
    
    adapter1.initialize_random().await;
    adapter2.initialize_random().await;
    
    // Prepare adapters for fusion
    let adapters = vec![
        ("domain_1".to_string(), adapter1),
        ("domain_2".to_string(), adapter2),
    ];
    
    // Initial weights (these should be replaced by dynamic weights)
    let mut weights = HashMap::new();
    weights.insert("domain_1".to_string(), 0.5);
    weights.insert("domain_2".to_string(), 0.5);
    
    // Perform dynamic fusion
    let fused_result = fusion_engine.fuse_adapters(adapters, &weights)?;
    
    // Verify fusion results
    assert_eq!(fused_result.fusion_metadata.num_adapters, 2);
    assert_eq!(fused_result.fusion_metadata.fusion_strategy, "attention_fusion");
    assert!(fused_result.fusion_metadata.total_sparse_weights > 0);
    
    println!("✅ Dynamic attention fusion completed successfully");
    println!("   - Adapters fused: {}", fused_result.fusion_metadata.num_adapters);
    println!("   - Total sparse weights: {}", fused_result.fusion_metadata.total_sparse_weights);
    println!("   - Strategy: {}", fused_result.fusion_metadata.fusion_strategy);
    
    // Test fusion statistics
    let stats = fusion_engine.get_stats();
    assert_eq!(stats.total_fusions, 1);
    assert!(stats.avg_fusion_time_ms > 0.0);
    
    println!("   - Fusion time: {:.2}ms", stats.avg_fusion_time_ms);
    println!("   - Average adapters per fusion: {:.1}", stats.avg_adapters_per_fusion);
    
    Ok(())
}

#[tokio::test]
async fn test_layer_wise_attention() -> anyhow::Result<()> {
    println!("🧪 Testing layer-wise attention patterns");
    
    // Create fusion engine with layer attention
    let mut fusion_engine = LoRAFusion::with_layer_attention(FusionStrategy::AttentionFusion, 12);
    
    // Test layer attention updates
    let performance_scores = vec![0.8, 0.9, 0.7, 0.6, 0.85, 0.75, 0.9, 0.8];
    fusion_engine.update_layer_attention("layer_0", &performance_scores);
    
    // Verify layer attention weights
    let layer_weights = fusion_engine.get_layer_attention("layer_0");
    assert!(layer_weights.is_some());
    
    let weights = layer_weights.unwrap();
    assert_eq!(weights.len(), 8);
    
    // Check that weights sum to 1.0 (normalized)
    let sum: f32 = weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    
    println!("✅ Layer-wise attention patterns working correctly");
    println!("   - Layer weights: {:?}", weights);
    
    // Test layer-specific fusion weights
    let mut base_weights = HashMap::new();
    base_weights.insert("adapter_1".to_string(), 0.6);
    base_weights.insert("adapter_2".to_string(), 0.4);
    
    let adjusted_weights = fusion_engine.compute_layer_fusion_weights("layer_0", &base_weights);
    
    // Check that adjusted weights are still normalized
    let adjusted_sum: f32 = adjusted_weights.values().sum();
    assert!((adjusted_sum - 1.0).abs() < 1e-6);
    
    println!("   - Adjusted weights: {:?}", adjusted_weights);
    
    Ok(())
}

#[tokio::test]
async fn test_sparse_mixture_fusion() -> anyhow::Result<()> {
    println!("🧪 Testing sparse mixture fusion (top-k)");
    
    // Create fusion engine with sparse mixture strategy
    let mut fusion_engine = LoRAFusion::new(FusionStrategy::SparseMixture { top_k: 2 });
    
    // Create multiple test adapters
    let mut adapters = Vec::new();
    let mut weights = HashMap::new();
    
    for i in 0..4 {
        let config = SparseLoRAConfig {
            rank: 4,
            sparsity: 0.95,
            alpha: (i + 1) as f32 * 8.0, // Different alphas
            ..Default::default()
        };
        
        let adapter = SparseLoRAAdapter::new(config);
        adapter.initialize_random().await;
        
        let adapter_id = format!("adapter_{}", i);
        adapters.push((adapter_id.clone(), adapter));
        weights.insert(adapter_id, (i + 1) as f32 * 0.1); // Different weights
    }
    
    // Perform sparse mixture fusion
    let fused_result = fusion_engine.fuse_adapters(adapters, &weights)?;
    
    // Verify only top-2 adapters were selected
    assert!(fused_result.fusion_metadata.num_adapters <= 2);
    assert!(fused_result.fusion_metadata.fusion_strategy.starts_with("sparse_mixture"));
    
    println!("✅ Sparse mixture fusion completed successfully");
    println!("   - Selected adapters: {}", fused_result.fusion_metadata.num_adapters);
    println!("   - Strategy: {}", fused_result.fusion_metadata.fusion_strategy);
    
    Ok(())
}

#[test]
fn test_fusion_strategy_optimization() {
    println!("🧪 Testing fusion strategy optimization");
    
    let mut fusion_engine = LoRAFusion::new(FusionStrategy::WeightedAverage);
    
    // Simulate performance metrics
    let mut performance_metrics = HashMap::new();
    performance_metrics.insert("weighted_average".to_string(), 0.7);
    performance_metrics.insert("attention_fusion".to_string(), 0.9);
    performance_metrics.insert("sequential".to_string(), 0.6);
    performance_metrics.insert("sparse_mixture_k2".to_string(), 0.8);
    
    // Optimize strategy based on performance
    let best_strategy = fusion_engine.optimize_strategy(&performance_metrics);
    
    // Should select attention_fusion (highest score: 0.9)
    match best_strategy {
        FusionStrategy::AttentionFusion => {
            println!("✅ Strategy optimization working correctly");
            println!("   - Best strategy: AttentionFusion");
        }
        _ => panic!("Expected AttentionFusion strategy to be selected"),
    }
    
    // Verify engine strategy was updated
    match fusion_engine.strategy() {
        FusionStrategy::AttentionFusion => println!("   - Engine strategy updated correctly"),
        _ => panic!("Engine strategy not updated"),
    }
}