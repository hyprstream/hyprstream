//! Simple demonstration of integrated conversation routing
//!
//! Shows how the ConversationRouter integrates with the main runtime system

use anyhow::Result;
use std::sync::Arc;
use hyprstream_core::{
    ConversationRouter, ModelPool, RoutingConfig,
    VDBSparseStorage, SparseStorageConfig,
    create_conversation_router,
};
use hyprstream_core::storage::vdb::{TemporalStreamingLayer, TemporalStreamingConfig};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔧 Testing Conversation Router Integration");
    
    // Create VDB storage backend
    let storage_config = SparseStorageConfig::default();
    let vdb_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);
    
    // Create temporal streaming layer
    let streaming_config = TemporalStreamingConfig::default();
    let temporal_streaming = Arc::new(
        TemporalStreamingLayer::new(vdb_storage, streaming_config).await?
    );
    
    // Create model pool
    let model_pool = Arc::new(ModelPool::new().await?);
    
    // Initialize base model
    model_pool.spawn_model_variant(
        "test-model-base".to_string(),
        Vec::new(),
    ).await?;
    
    println!("✅ Created model pool with base model");
    
    // Create conversation router using the integrated function
    let routing_config = RoutingConfig::default();
    let conversation_router = create_conversation_router(
        Arc::clone(&model_pool),
        Arc::clone(&temporal_streaming),
        Some(routing_config),
    ).await?;
    
    println!("✅ Created conversation router successfully");
    
    // Test basic conversation functionality  
    let session_id = conversation_router.start_conversation(
        "test_user".to_string(),
        "test-model-base".to_string(),
    ).await?;
    
    println!("✅ Started conversation session: {}", session_id);
    
    // Test generation with adaptation
    let response = conversation_router.generate_with_adaptation(
        &session_id,
        "Hello! This is a test message.".to_string(),
    ).await?;
    
    println!("✅ Generated response from model: {}", response.model_used);
    println!("   Response: {}", response.turn.assistant_response[..50.min(response.turn.assistant_response.len())].to_string());
    println!("   Adaptation applied: {}", response.adaptation_applied);
    
    // Test feedback application
    conversation_router.apply_feedback(
        &session_id,
        &response.turn.turn_id,
        0.8, // Good feedback
    ).await?;
    
    println!("✅ Applied user feedback successfully");
    
    // Get pool statistics
    let pool_stats = model_pool.get_pool_stats().await;
    println!("📊 Model Pool Stats:");
    println!("   - Active instances: {}", pool_stats.active_instances);
    println!("   - Max capacity: {}", pool_stats.max_capacity);
    
    println!("\n🎉 Conversation Router integration test completed successfully!");
    println!("   ✓ VDB storage backend connected");
    println!("   ✓ Temporal streaming layer active");  
    println!("   ✓ Model pool managing instances");
    println!("   ✓ Conversation routing functional");
    println!("   ✓ Real-time adaptation ready");
    
    Ok(())
}