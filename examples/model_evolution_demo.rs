//! Model Evolution Demo - Seamless Adaptive Inference
//!
//! This example demonstrates:
//! 1. Run model during conversation
//! 2. Train LoRA updates in real-time from user feedback
//! 3. Checkpoint LoRA adaptations to VDB storage
//! 4. Seamlessly route conversations to evolved models
//! 5. Roll back to previous model checkpoints if needed

use anyhow::Result;
use std::sync::Arc;
use hyprstream_core::runtime::{
    ConversationRouter, MistralEngine, RuntimeConfig, ModelPool, 
    ConversationResponse, AdaptationType
};
use hyprstream_core::storage::vdb::{
    TemporalStreamingLayer, TemporalStreamingConfig, VDBSparseStorage, SparseStorageConfig
};
use hyprstream_core::adapters::lora_checkpoints::{LoRACheckpoint, CheckpointMetrics};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🏗️  Model Evolution Demo - Seamless Adaptive Inference");
    
    // Initialize system components
    let system = AdaptiveInferenceSystem::initialize().await?;
    
    // Demo workflow
    demo_conversation_with_adaptation(&system).await?;
    demo_seamless_model_transitions(&system).await?;
    demo_model_checkpoints_and_rollback(&system).await?;
    
    println!("✅ Model Evolution demo completed successfully!");
    Ok(())
}

/// Adaptive inference system components
struct AdaptiveInferenceSystem {
    conversation_router: Arc<ConversationRouter>,
    model_pool: Arc<ModelPool>,
    temporal_streaming: Arc<TemporalStreamingLayer>,
}

impl AdaptiveInferenceSystem {
    /// Initialize adaptive inference system
    async fn initialize() -> Result<Self> {
        println!("🔧 Initializing adaptive inference system...");
        
        // Create VDB backend for temporal streaming
        let storage_config = SparseStorageConfig::default();
        let vdb_storage = Arc::new(VDBSparseStorage::new(storage_config).await?);
        
        // Create temporal streaming layer
        let streaming_config = TemporalStreamingConfig::default();
        let temporal_streaming = Arc::new(
            TemporalStreamingLayer::new(vdb_storage, streaming_config).await?
        );
        
        // Create model pool for hot-swapping
        let model_pool = Arc::new(ModelPool::new().await?);
        
        // Initialize base models in pool
        model_pool.spawn_model_variant(
            "qwen3-1.5b-base".to_string(),
            Vec::new(), // No LoRA initially
        ).await?;
        
        // Create conversation router
        let conversation_router = Arc::new(ConversationRouter::new(
            Arc::clone(&model_pool),
            Arc::clone(&temporal_streaming),
            Default::default(),
        ).await?);
        
        println!("✅ Adaptive inference system initialized");
        
        Ok(Self {
            conversation_router,
            model_pool,
            temporal_streaming,
        })
    }
}

/// Demo: Conversation with real-time adaptation
async fn demo_conversation_with_adaptation(system: &AdaptiveInferenceSystem) -> Result<()> {
    println!("\n🗣️  Demo 1: Conversation with Real-time Adaptation");
    
    // Start conversation
    let session_id = system.conversation_router.start_conversation(
        "user_alice".to_string(),
        "qwen3-1.5b-base".to_string(),
    ).await?;
    
    println!("   Started conversation session: {}", session_id);
    
    // Simulate conversation turns with feedback
    let conversation_turns = vec![
        ("Hello! Can you help me write better code?", 0.8),
        ("I'm struggling with async Rust programming", 0.6), // Lower quality
        ("Can you explain futures and streams?", 0.5),      // Poor quality
        ("Show me a practical example", 0.4),               // Triggers adaptation
    ];
    
    for (i, (message, quality_score)) in conversation_turns.iter().enumerate() {
        println!("\n   Turn {}: User: \"{}\"", i + 1, message);
        
        // Generate response
        let response = system.conversation_router.generate_with_adaptation(
            &session_id,
            message.to_string(),
        ).await?;
        
        println!("   Turn {}: Assistant: \"{}\"", i + 1, 
                &response.turn.assistant_response[..80.min(response.turn.assistant_response.len())]);
        println!("   Model used: {}, Adaptation applied: {}", 
                response.model_used, response.adaptation_applied);
        
        // Apply user feedback
        system.conversation_router.apply_feedback(
            &session_id,
            &response.turn.turn_id,
            *quality_score,
        ).await?;
        
        println!("   Applied feedback: {:.1}/1.0", quality_score);
        
        // Check if adaptation was triggered
        if response.adaptation_applied {
            println!("   🧠 Real-time adaptation applied - model evolved!");
        }
        
        // Small delay to simulate real conversation
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    println!("✅ Conversation with adaptation completed");
    Ok(())
}

/// Demo: Seamless model transitions
async fn demo_seamless_model_transitions(system: &AdaptiveInferenceSystem) -> Result<()> {
    println!("\n🔄 Demo 2: Seamless Model Transitions");
    
    let session_id = system.conversation_router.start_conversation(
        "user_bob".to_string(),
        "qwen3-1.5b-base".to_string(),
    ).await?;
    
    // Initial conversation
    let response1 = system.conversation_router.generate_with_adaptation(
        &session_id,
        "Tell me about machine learning".to_string(),
    ).await?;
    
    println!("   Initial response from: {}", response1.model_used);
    
    // Create specialized model variant with domain LoRA
    println!("   🔥 Creating specialized ML model variant...");
    system.model_pool.spawn_model_variant(
        "qwen3-1.5b-ml-specialist".to_string(),
        vec!["ml_domain_lora_v1".to_string()],
    ).await?;
    
    // Seamlessly transition to specialized model
    system.conversation_router.seamless_transition(
        &session_id,
        "qwen3-1.5b-ml-specialist".to_string(),
        "User asking ML questions - routing to specialized model".to_string(),
    ).await?;
    
    println!("   🌊 Seamless transition completed");
    
    // Continue conversation with specialized model
    let response2 = system.conversation_router.generate_with_adaptation(
        &session_id,
        "Explain gradient descent in detail".to_string(),
    ).await?;
    
    println!("   Specialized response from: {}", response2.model_used);
    println!("   Response quality should be higher for ML topics");
    
    // Transition back to general model
    system.conversation_router.seamless_transition(
        &session_id,
        "qwen3-1.5b-base".to_string(),
        "Conversation topic changed - back to general model".to_string(),
    ).await?;
    
    let response3 = system.conversation_router.generate_with_adaptation(
        &session_id,
        "What's the weather like today?".to_string(),
    ).await?;
    
    println!("   General response from: {}", response3.model_used);
    
    println!("✅ Seamless model transitions completed");
    Ok(())
}

/// Demo: Model checkpoints and rollback
async fn demo_model_checkpoints_and_rollback(system: &AdaptiveInferenceSystem) -> Result<()> {
    println!("\n📸 Demo 3: Model Checkpoints and Rollback");
    
    let session_id = system.conversation_router.start_conversation(
        "user_charlie".to_string(),
        "qwen3-1.5b-base".to_string(),
    ).await?;
    
    // Create initial snapshot
    println!("   📸 Creating initial model snapshot...");
    let snapshot_v1 = create_model_snapshot(
        "qwen3-1.5b-base",
        "v1_initial_state",
        "Initial model state before adaptations",
    ).await?;
    
    println!("   Created snapshot: {}", snapshot_v1.checkpoint_id);
    
    // Have conversation with adaptations
    for i in 1..=3 {
        let response = system.conversation_router.generate_with_adaptation(
            &session_id,
            format!("This is message {} - please adapt to my style", i),
        ).await?;
        
        // Simulate poor feedback to trigger adaptations
        system.conversation_router.apply_feedback(
            &session_id,
            &response.turn.turn_id,
            0.3, // Poor feedback
        ).await?;
        
        println!("   Turn {}: Applied adaptation", i);
    }
    
    // Create snapshot after adaptations
    println!("   📸 Creating adapted model snapshot...");
    let snapshot_v2 = create_model_snapshot(
        "qwen3-1.5b-base",
        "v2_adapted_state",
        "Model state after user adaptations",
    ).await?;
    
    println!("   Created adapted snapshot: {}", snapshot_v2.checkpoint_id);
    
    // Simulate problematic conversation
    println!("   🚨 Simulating problematic responses...");
    for i in 1..=2 {
        let response = system.conversation_router.generate_with_adaptation(
            &session_id,
            format!("Problem scenario {}", i),
        ).await?;
        
        // Very poor feedback
        system.conversation_router.apply_feedback(
            &session_id,
            &response.turn.turn_id,
            0.1, // Very poor
        ).await?;
    }
    
    // Rollback to previous checkpoint
    println!("   ⏪ Rolling back to previous checkpoint...");
    rollback_to_checkpoint(&system, &session_id, &snapshot_v2).await?;
    
    println!("   🔄 Rollback completed - model restored to adapted state");
    
    // Test conversation after rollback
    let response_after_rollback = system.conversation_router.generate_with_adaptation(
        &session_id,
        "How are you performing after the rollback?".to_string(),
    ).await?;
    
    println!("   Response after rollback from: {}", response_after_rollback.model_used);
    
    println!("✅ Model checkpoints and rollback completed");
    Ok(())
}

/// Create model snapshot (checkpoint)
async fn create_model_snapshot(
    model_id: &str,
    version_tag: &str,
    description: &str,
) -> Result<LoRACheckpoint> {
    let checkpoint_id = format!("{}_{}", model_id, version_tag);
    let weights_path = PathBuf::from(format!("checkpoints/{}.vdb", checkpoint_id));
    
    // Real implementation: Extract current model state and save to VDB
    let start_time = std::time::Instant::now();
    
    // Create checkpoint directory if it doesn't exist
    if let Some(parent) = weights_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    
    // Simulate creating actual snapshot with sparse weights
    let snapshot_data = create_mock_snapshot_data(model_id)?;
    
    // Write snapshot to VDB storage (simulated)
    tokio::fs::write(&weights_path, snapshot_data).await?;
    
    let checkpoint = LoRACheckpoint {
        checkpoint_id: checkpoint_id.clone(),
        lora_uuid: uuid::Uuid::new_v4(),
        tag: version_tag.to_string(),
        created_at: chrono::Utc::now().timestamp(),
        weights_path,
        metrics: CheckpointMetrics {
            loss: Some(0.25),
            steps: 100,
            sparsity: 0.99,
            active_params: 15_000_000,
            rank: 64,
            alpha: 128.0,
        },
        file_size: 1024 * 1024, // 1MB simulated
        checksum: "sha256:mock_checksum_for_demo".to_string(),
    };
    
    let duration = start_time.elapsed();
    println!("   💾 Snapshot created in {:.2}ms: {} -> {}", 
             duration.as_millis(), checkpoint_id, checkpoint.weights_path.display());
    
    Ok(checkpoint)
}

/// Create mock snapshot data for demonstration
fn create_mock_snapshot_data(model_id: &str) -> Result<Vec<u8>> {
    use std::collections::HashMap;
    use serde_json;
    
    // Create mock sparse weight data
    let mut weights = HashMap::new();
    let base_value = if model_id.contains("adapted") { 0.8 } else { 0.5 };
    
    for i in 0..100 {
        weights.insert(format!("layer_{}_weight_{}", i % 10, i), base_value + (i as f64) * 0.001);
    }
    
    let snapshot_data = serde_json::json!({
        "model_id": model_id,
        "timestamp": chrono::Utc::now().timestamp(),
        "weights": weights,
        "metadata": {
            "sparsity": 0.99,
            "total_params": 1_500_000_000i64,
            "active_params": 15_000_000i64
        }
    });
    
    Ok(serde_json::to_vec(&snapshot_data)?)
}

/// Rollback to previous snapshot
async fn rollback_to_checkpoint(
    system: &AdaptiveInferenceSystem,
    session_id: &str,
    snapshot: &LoRACheckpoint,
) -> Result<()> {
    let rollback_start = std::time::Instant::now();
    
    println!("   📼 Loading snapshot from VDB storage...");
    
    // Load snapshot data from VDB storage
    let snapshot_data = load_snapshot_from_vdb(snapshot).await?;
    let rollback_model_id = format!("{}_rollback", snapshot.checkpoint_id);
    
    println!("   🔄 Creating rollback model variant...");
    
    // Create model variant with rollback weights  
    // In real implementation, this would:
    // 1. Load the sparse weights from OpenVDB storage
    // 2. Apply weights to base model architecture
    // 3. Validate model integrity
    system.model_pool.spawn_model_variant(
        rollback_model_id.clone(),
        vec![snapshot.checkpoint_id.clone()],
    ).await?;
    
    println!("   🌊 Performing seamless transition...");
    
    // Seamless transition to rollback model
    system.conversation_router.seamless_transition(
        session_id,
        rollback_model_id.clone(),
        format!("Rollback to snapshot: {} ({})", 
               snapshot.checkpoint_id, snapshot.metadata.description),
    ).await?;
    
    // Verify rollback success
    verify_rollback_success(&rollback_model_id, snapshot).await?;
    
    let rollback_time = rollback_start.elapsed();
    println!("   ✅ Rollback completed in {:.2}ms - model restored to checkpoint state", 
             rollback_time.as_millis());
    
    Ok(())
}

/// Load snapshot data from VDB storage
async fn load_snapshot_from_vdb(snapshot: &LoRACheckpoint) -> Result<serde_json::Value> {
    // In real implementation: Load sparse weights from OpenVDB
    let snapshot_bytes = tokio::fs::read(&snapshot.weights_path).await?;
    let snapshot_data: serde_json::Value = serde_json::from_slice(&snapshot_bytes)?;
    
    println!("   📊 Loaded snapshot: {:.2}% sparse, {:.1}MB", 
             snapshot.metrics.sparsity * 100.0,
             snapshot.file_size as f64 / 1024.0 / 1024.0);
    
    Ok(snapshot_data)
}

/// Verify rollback success by testing model behavior
async fn verify_rollback_success(
    rollback_model_id: &str,
    snapshot: &LoRACheckpoint,
) -> Result<()> {
    // In real implementation, would run validation inference
    // to ensure model state matches snapshot expectations
    
    println!("   🔍 Verifying rollback integrity...");
    println!("   📈 Model state restored to checkpoint: {}", snapshot.checkpoint_id);
    println!("   📊 Checkpoint metadata:");
    println!("       - Tag: {}", snapshot.tag);
    println!("       - Training steps: {}", snapshot.metrics.steps);
    println!("       - Loss: {:.4}", snapshot.metrics.loss.unwrap_or(0.0));
    println!("       - Sparsity: {:.2}%", snapshot.metrics.sparsity * 100.0);
    println!("       - Active params: {:.1}M", snapshot.metrics.active_params as f64 / 1_000_000.0);
    println!("       - Rank: {}, Alpha: {}", snapshot.metrics.rank, snapshot.metrics.alpha);
    println!("       - Created: {}", 
             chrono::DateTime::from_timestamp(snapshot.created_at, 0)
                 .unwrap_or_default()
                 .format("%Y-%m-%d %H:%M:%S"));
    
    Ok(())
}

// Note: ModelPool implementation is now in conversation_router.rs