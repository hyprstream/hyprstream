use std::sync::Arc;
use tokio::sync::RwLock;

use crate::async_test;
use hypr::storage::vector::{VectorBackend, VectorEmbedding, VectorOperations};
use tonic::Status;

// Test fixtures
fn create_test_embeddings(count: usize, dim: usize) -> Vec<VectorEmbedding> {
    (0..count)
        .map(|i| VectorEmbedding {
            id: format!("test_id_{}", i),
            embedding: vec![1.0; dim],
            metadata: None,
        })
        .collect()
}

#[tokio::test]
async fn test_vector_backend_type_prefix() {
    let backend = VectorBackend::new(1000, 3600);
    let embeddings = create_test_embeddings(1, 128);
    
    // Verify type prefix handling
    let result = backend.store_embeddings(embeddings).await;
    assert!(result.is_ok(), "Vector storage should accept valid type prefixes");
}

#[tokio::test]
async fn test_vector_operations_trait() {
    let backend = VectorBackend::new(1000, 3600);
    
    // Test store operation
    let embeddings = create_test_embeddings(5, 128);
    let store_result = backend.store_embeddings(embeddings.clone()).await;
    assert!(store_result.is_ok(), "Store operation should succeed");

    // Test query operation
    let query_vec = vec![1.0; 128];
    let query_result = backend.query_similar_vectors(query_vec, 3, 0.5).await;
    assert!(query_result.is_ok(), "Query operation should succeed");
}

#[tokio::test]
async fn test_vector_error_handling() {
    let backend = VectorBackend::new(1000, 3600);

    // Test empty embeddings
    let empty_embeddings = Vec::new();
    let result = backend.store_embeddings(empty_embeddings).await;
    assert!(result.is_ok(), "Empty embeddings should be handled gracefully");

    // Test inconsistent dimensions
    let mut bad_embeddings = create_test_embeddings(2, 128);
    bad_embeddings[1].embedding = vec![1.0; 64]; // Different dimension
    let result = backend.store_embeddings(bad_embeddings).await;
    assert!(result.is_err(), "Inconsistent dimensions should return error");
}

#[tokio::test]
async fn test_vector_type_system() {
    let backend = VectorBackend::new(1000, 3600);
    
    // Test type constraints
    let embeddings = create_test_embeddings(1, 128);
    let store_result = backend.store_embeddings(embeddings).await;
    assert!(store_result.is_ok(), "Type system should accept valid embeddings");

    // Test invalid query vector
    let invalid_query: Vec<f32> = vec![];
    let query_result = backend.query_similar_vectors(invalid_query, 5, 0.5).await;
    assert!(query_result.is_err(), "Empty query vector should return error");
}

#[tokio::test]
async fn test_concurrent_vector_operations() {
    let backend = Arc::new(VectorBackend::new(1000, 3600));
    let mut handles = vec![];

    // Spawn multiple concurrent operations
    for i in 0..5 {
        let backend_clone = backend.clone();
        let handle = tokio::spawn(async move {
            let embeddings = create_test_embeddings(2, 128);
            backend_clone.store_embeddings(embeddings).await
        });
        handles.push(handle);
    }

    // Verify all operations complete successfully
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Concurrent operations should succeed");
    }
}
