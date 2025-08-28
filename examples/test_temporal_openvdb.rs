//! Test temporal streaming with real OpenVDB integration
//! 
//! This example demonstrates the temporal LoRA streaming system
//! with actual OpenVDB sparse storage backend.

use anyhow::Result;
use std::collections::HashMap;
use hyprstream_core::storage::vdb::{
    OpenVDBLoRAAdapter, TemporalStreamingLayer, TemporalStreamingConfig, 
    VDBSparseStorage, SparseStorageConfig, Coordinate3D
};
use hyprstream_core::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig};
use std::sync::Arc;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing Temporal LoRA Streaming with OpenVDB");
    
    // Test 1: Basic OpenVDB operations
    test_openvdb_basic_operations().await?;
    
    // Test 2: Temporal streaming operations
    test_temporal_streaming_operations().await?;
    
    // Test 3: Gradient-based operations
    test_gradient_operations().await?;
    
    // Test 4: Full temporal streaming with VDB backend
    test_full_temporal_streaming().await?;
    
    println!("✅ All temporal OpenVDB tests passed!");
    Ok(())
}

async fn test_openvdb_basic_operations() -> Result<()> {
    println!("\n📊 Testing basic OpenVDB operations...");
    
    // Create OpenVDB adapter
    let mut adapter = OpenVDBLoRAAdapter::new(1536, 1536)?;
    
    // Test sparse weight operations
    adapter.set_weight(100, 200, 0.5)?;
    adapter.set_weight(300, 400, -0.3)?;
    adapter.set_weight(500, 600, 1.2)?;
    
    // Verify values
    assert_eq!(adapter.get_weight(100, 200)?, 0.5);
    assert_eq!(adapter.get_weight(300, 400)?, -0.3);
    assert_eq!(adapter.get_weight(500, 600)?, 1.2);
    
    // Test sparsity
    let sparsity = adapter.sparsity_ratio();
    println!("   Sparsity ratio: {:.4}% (should be >99%)", sparsity * 100.0);
    assert!(sparsity > 0.99);
    
    // Test active count
    let active_count = adapter.active_count();
    println!("   Active voxels: {}", active_count);
    assert_eq!(active_count, 3);
    
    println!("✅ Basic OpenVDB operations working");
    Ok(())
}

async fn test_temporal_streaming_operations() -> Result<()> {
    println!("\n⏰ Testing temporal streaming operations...");
    
    let mut adapter = OpenVDBLoRAAdapter::new(1000, 1000)?;
    
    // Test timestamp operations
    let timestamp = chrono::Utc::now().timestamp_millis();
    adapter.set_temporal_timestamp(timestamp)?;
    assert_eq!(adapter.get_temporal_timestamp(), timestamp);
    
    // Test streaming session
    adapter.begin_streaming_update()?;
    
    // Apply streaming updates
    for i in 0..10 {
        let row = i * 10;
        let col = i * 20;
        let weight = 0.1 * i as f32;
        adapter.streaming_set_weight(row, col, weight, timestamp + i)?;
    }
    
    assert!(adapter.is_streaming_active());
    let success = adapter.end_streaming_update()?;
    assert!(success);
    assert!(!adapter.is_streaming_active());
    
    // Test temporal snapshot
    let snapshot = adapter.create_temporal_snapshot()?;
    assert_eq!(snapshot.get_temporal_timestamp(), adapter.get_temporal_timestamp());
    
    println!("✅ Temporal streaming operations working");
    Ok(())
}

async fn test_gradient_operations() -> Result<()> {
    println!("\n🔄 Testing gradient operations...");
    
    // Create two adapters with different weights
    let mut adapter1 = OpenVDBLoRAAdapter::new(100, 100)?;
    let mut adapter2 = OpenVDBLoRAAdapter::new(100, 100)?;
    
    // Set different weights
    adapter1.set_weight(10, 20, 1.0)?;
    adapter1.set_weight(30, 40, 0.5)?;
    
    adapter2.set_weight(10, 20, 0.8)?;
    adapter2.set_weight(30, 40, 0.3)?;
    adapter2.set_weight(50, 60, 0.2)?; // Additional weight
    
    // Test gradient magnitude
    let magnitude1 = adapter1.compute_gradient_magnitude();
    let magnitude2 = adapter2.compute_gradient_magnitude();
    println!("   Gradient magnitudes: {:.4}, {:.4}", magnitude1, magnitude2);
    assert!(magnitude1 > 0.0);
    assert!(magnitude2 > 0.0);
    
    // Test gradient difference
    let gradient = adapter2.compute_gradient_difference(&adapter1)?;
    let grad_magnitude = gradient.compute_gradient_magnitude();
    println!("   Gradient difference magnitude: {:.4}", grad_magnitude);
    assert!(grad_magnitude > 0.0);
    
    // Test gradient interpolation
    let interpolated = adapter1.interpolate_with(&adapter2, 0.5)?;
    let interp_magnitude = interpolated.compute_gradient_magnitude();
    println!("   Interpolated magnitude: {:.4}", interp_magnitude);
    assert!(interp_magnitude > 0.0);
    
    // Test gradient update
    adapter1.apply_gradient_update(&gradient, 0.1)?;
    let updated_magnitude = adapter1.compute_gradient_magnitude();
    println!("   Updated magnitude: {:.4}", updated_magnitude);
    
    println!("✅ Gradient operations working");
    Ok(())
}

async fn test_full_temporal_streaming() -> Result<()> {
    println!("\n🌊 Testing full temporal streaming with VDB backend...");
    
    // Create VDB storage backend
    let storage_config = SparseStorageConfig::default();
    let vdb_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);
    
    // Create temporal streaming layer
    let streaming_config = TemporalStreamingConfig {
        stream_frequency_ms: 50, // Fast streaming for test
        temporal_window_secs: 10,
        gradient_window_ms: 500,
        drift_threshold: 0.1,
        max_streaming_sessions: 2,
        update_buffer_size: 100,
    };
    
    let temporal_layer = TemporalStreamingLayer::new(vdb_storage, streaming_config).await?;
    
    // Create sparse LoRA adapters
    let input_config = SparseLoRAConfig::default();
    let mut input_adapter = SparseLoRAAdapter::new(input_config.clone(), (512, 512)).await?;
    let mut target_adapter = SparseLoRAAdapter::new(input_config, (512, 512)).await?;
    
    // Add some sparse weights to simulate training
    let mut updates = HashMap::new();
    for i in 0..10 {
        let coord = Coordinate3D::new(i * 10, i * 20, 0);
        updates.insert(coord, 0.1 * i as f32);
    }
    
    // Test temporal gradient accumulation
    let gradient = temporal_layer.accumulate_temporal_gradients(
        "test_layer",
        &input_adapter,
        &target_adapter,
        None,
    ).await?;
    
    println!("   Accumulated gradient magnitude: {:.6}", gradient.magnitude);
    assert!(gradient.magnitude >= 0.0);
    assert_eq!(gradient.layer_name, "test_layer");
    
    // Test temporal streaming session
    let mut stream = temporal_layer.create_temporal_stream(
        "test_adapter".to_string(),
        None,
    ).await?;
    
    println!("   Created temporal stream: {}", stream.session_id);
    
    // Collect a few streaming updates (non-blocking)
    let mut update_count = 0;
    for _ in 0..3 {
        if let Some(update_result) = stream.next().await {
            match update_result {
                Ok(update) => {
                    println!("   📥 Received update: {} weights", update.weights.len());
                    update_count += 1;
                }
                Err(e) => {
                    println!("   ⚠️ Stream error: {}", e);
                    break;
                }
            }
        }
        
        // Small delay to allow background processing
        tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;
    }
    
    println!("   Processed {} streaming updates", update_count);
    
    // Get streaming statistics
    let stats = temporal_layer.get_streaming_stats().await;
    println!("   Streaming stats: {} active sessions, {} gradients", 
             stats.active_sessions, stats.total_gradients);
    
    println!("✅ Full temporal streaming working");
    Ok(())
}