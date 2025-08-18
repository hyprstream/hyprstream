//! Unit tests for VDB storage functionality

use hyprstream_core::storage::vdb::{
    VDBSparseStorage, SparseStorageConfig, SparseWeightUpdate, Coordinate3D,
};
use hyprstream_core::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig, InitMethod};
use std::collections::HashMap;
use tempfile::TempDir;

/// Helper function to create test config
fn create_test_config() -> SparseStorageConfig {
    let temp_dir = TempDir::new().unwrap();
    SparseStorageConfig {
        storage_path: temp_dir.path().to_path_buf(),
        neural_compression: false, // Disable for testing
        hardware_acceleration: false,
        cache_size_mb: 100,
        compaction_interval_secs: 60,
        streaming_updates: true,
        update_batch_size: 10,
    }
}

/// Helper function to create test adapter
fn create_test_adapter(id: &str) -> SparseLoRAAdapter {
    let config = SparseLoRAConfig {
        in_features: 256,
        out_features: 256,
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec!["test_module".to_string()],
        sparsity: 0.95,
        sparsity_threshold: 0.01,
        learning_rate: 0.001,
        bias: false,
        enable_gradient_checkpointing: false,
        init_method: InitMethod::Random,
        mixed_precision: false,
    };
    SparseLoRAAdapter::new(config)
}

#[tokio::test]
async fn test_vdb_storage_creation() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await;
    assert!(storage.is_ok(), "Failed to create VDB storage");
}

#[tokio::test]
async fn test_store_and_load_adapter() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Create and store adapter
    let adapter_id = "test_adapter_1";
    let adapter = create_test_adapter(adapter_id);
    
    let store_result = storage.store_adapter(adapter_id, &adapter).await;
    assert!(store_result.is_ok(), "Failed to store adapter");
    
    // Load adapter back
    let load_config = SparseLoRAConfig {
        in_features: 256,
        out_features: 256,
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec!["test_module".to_string()],
        sparsity: 0.95,
        sparsity_threshold: 0.01,
        learning_rate: 0.001,
        bias: false,
        enable_gradient_checkpointing: false,
        init_method: InitMethod::Random,
        mixed_precision: false,
    };
    
    let loaded = storage.load_adapter(adapter_id, load_config).await;
    assert!(loaded.is_ok(), "Failed to load adapter");
}

#[tokio::test]
async fn test_sparse_weight_update() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Store initial adapter
    let adapter_id = "test_adapter_update";
    let adapter = create_test_adapter(adapter_id);
    storage.store_adapter(adapter_id, &adapter).await.unwrap();
    
    // Create weight update
    let mut updates = HashMap::new();
    updates.insert(Coordinate3D::new(10, 20, 0), 0.5);
    updates.insert(Coordinate3D::new(15, 25, 0), -0.3);
    
    let weight_update = SparseWeightUpdate {
        adapter_id: adapter_id.to_string(),
        updates,
        timestamp: chrono::Utc::now(),
        gradient_norm: Some(0.1),
        learning_rate: 0.001,
    };
    
    // Apply update
    let update_result = storage.apply_sparse_update(weight_update).await;
    assert!(update_result.is_ok(), "Failed to apply sparse update");
}

#[tokio::test]
async fn test_get_adapter_stats() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Store adapter
    let adapter_id = "test_adapter_stats";
    let adapter = create_test_adapter(adapter_id);
    storage.store_adapter(adapter_id, &adapter).await.unwrap();
    
    // Get stats
    let stats = storage.get_adapter_stats(adapter_id).await;
    assert!(stats.is_ok(), "Failed to get adapter stats");
    
    let stats = stats.unwrap();
    assert_eq!(stats.adapter_id, adapter_id);
    assert!(stats.total_weights > 0);
    assert!(stats.sparse_weights > 0);
    assert!(stats.sparsity_ratio > 0.9); // Should be ~95% sparse
}

#[tokio::test]
async fn test_list_adapters() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Store multiple adapters
    for i in 0..3 {
        let adapter_id = format!("test_adapter_list_{}", i);
        let adapter = create_test_adapter(&adapter_id);
        storage.store_adapter(&adapter_id, &adapter).await.unwrap();
    }
    
    // List adapters
    let adapters = storage.list_adapters().await;
    assert!(adapters.is_ok(), "Failed to list adapters");
    
    let adapter_list = adapters.unwrap();
    assert_eq!(adapter_list.len(), 3, "Should have 3 adapters");
}

#[tokio::test]
async fn test_delete_adapter() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Store and delete adapter
    let adapter_id = "test_adapter_delete";
    let adapter = create_test_adapter(adapter_id);
    storage.store_adapter(adapter_id, &adapter).await.unwrap();
    
    let delete_result = storage.delete_adapter(adapter_id).await;
    assert!(delete_result.is_ok(), "Failed to delete adapter");
    
    // Verify it's gone
    let load_config = SparseLoRAConfig {
        in_features: 256,
        out_features: 256,
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec!["test_module".to_string()],
        sparsity: 0.95,
        sparsity_threshold: 0.01,
        learning_rate: 0.001,
        bias: false,
        enable_gradient_checkpointing: false,
        init_method: InitMethod::Random,
        mixed_precision: false,
    };
    
    let loaded = storage.load_adapter(adapter_id, load_config).await;
    assert!(loaded.is_err(), "Adapter should not exist after deletion");
}

#[tokio::test]
async fn test_batch_weight_updates() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Store adapter
    let adapter_id = "test_adapter_batch";
    let adapter = create_test_adapter(adapter_id);
    storage.store_adapter(adapter_id, &adapter).await.unwrap();
    
    // Create multiple updates
    let mut batch_updates = Vec::new();
    for i in 0..5 {
        let mut updates = HashMap::new();
        updates.insert(Coordinate3D::new(i * 10, i * 20, 0), 0.1 * i as f32);
        
        let update = SparseWeightUpdate {
            adapter_id: adapter_id.to_string(),
            updates,
            timestamp: chrono::Utc::now(),
            gradient_norm: Some(0.1),
            learning_rate: 0.001,
        };
        batch_updates.push(update);
    }
    
    // Apply batch
    for update in batch_updates {
        let result = storage.apply_sparse_update(update).await;
        assert!(result.is_ok(), "Failed to apply batch update");
    }
}

#[tokio::test]
async fn test_compression_stats() {
    let mut config = create_test_config();
    config.neural_compression = true; // Enable compression
    
    let storage = VDBSparseStorage::new(config).await.unwrap();
    
    // Store adapter with compression
    let adapter_id = "test_adapter_compression";
    let adapter = create_test_adapter(adapter_id);
    storage.store_adapter(adapter_id, &adapter).await.unwrap();
    
    // Get compression stats
    let stats = storage.get_compression_stats().await;
    assert!(stats.is_ok(), "Failed to get compression stats");
    
    let stats = stats.unwrap();
    assert!(stats.original_size > 0);
    // Compression ratio should be > 1 if compression is working
    assert!(stats.compression_ratio >= 1.0);
}

#[tokio::test]
async fn test_concurrent_access() {
    let config = create_test_config();
    let storage = VDBSparseStorage::new(config).await.unwrap();
    let storage = std::sync::Arc::new(storage);
    
    // Create adapter
    let adapter_id = "test_adapter_concurrent";
    let adapter = create_test_adapter(adapter_id);
    storage.store_adapter(adapter_id, &adapter).await.unwrap();
    
    // Spawn multiple concurrent readers
    let mut handles = Vec::new();
    for i in 0..5 {
        let storage_clone = storage.clone();
        let adapter_id = adapter_id.to_string();
        
        let handle = tokio::spawn(async move {
            let load_config = SparseLoRAConfig {
                in_features: 256,
                out_features: 256,
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
                target_modules: vec!["test_module".to_string()],
                sparsity: 0.95,
                sparsity_threshold: 0.01,
                learning_rate: 0.001,
                bias: false,
                enable_gradient_checkpointing: false,
                init_method: InitMethod::Random,
                mixed_precision: false,
            };
            
            let result = storage_clone.load_adapter(&adapter_id, load_config).await;
            assert!(result.is_ok(), "Thread {} failed to load adapter", i);
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    for handle in handles {
        handle.await.unwrap();
    }
}