use hyprstream::storage::vector::{VectorBackend, VectorEmbedding};
use std::collections::HashMap;
use tonic::Status;

#[tokio::test]
async fn test_store_and_query_embeddings() {
    let backend = VectorBackend::new(1000, 3600);
    
    // Create test embeddings
    let embeddings = vec![
        VectorEmbedding {
            id: "1".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 1000,
            metadata: HashMap::new(),
        },
        VectorEmbedding {
            id: "2".to_string(),
            embedding: vec![0.0, 1.0, 0.0, 0.0],
            timestamp: 1001,
            metadata: HashMap::new(),
        },
    ];

    // Store embeddings
    backend.store_embeddings(embeddings).await.unwrap();

    // Query similar vectors
    let query = vec![1.0, 0.1, 0.0, 0.0];
    let results = backend.query_similar_vectors(query, 2, 0.5).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "1");
    assert!(results[0].1 > 0.9);
}

#[tokio::test]
async fn test_cache_behavior() {
    let backend = VectorBackend::new(2, 3600); // Cache size of 2
    
    // Create 3 embeddings
    let embeddings = vec![
        VectorEmbedding {
            id: "1".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 1000,
            metadata: HashMap::new(),
        },
        VectorEmbedding {
            id: "2".to_string(),
            embedding: vec![0.0, 1.0, 0.0, 0.0],
            timestamp: 1001,
            metadata: HashMap::new(),
        },
        VectorEmbedding {
            id: "3".to_string(),
            embedding: vec![0.0, 0.0, 1.0, 0.0],
            timestamp: 1002,
            metadata: HashMap::new(),
        },
    ];

    // Store embeddings (should evict oldest)
    backend.store_embeddings(embeddings).await.unwrap();

    // Query - should only find newer vectors
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = backend.query_similar_vectors(query, 3, 0.0).await.unwrap();
    
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(e, _)| e.id != "1"));
}

#[tokio::test]
async fn test_error_conditions() {
    let backend = VectorBackend::new(1000, 3600);
    
    // Test empty query vector
    let result = backend.query_similar_vectors(vec![], 1, 0.5).await;
    assert!(matches!(result, Err(Status { .. })));

    // Test inconsistent dimensions
    let embeddings = vec![
        VectorEmbedding {
            id: "1".to_string(),
            embedding: vec![1.0, 0.0],
            timestamp: 1000,
            metadata: HashMap::new(),
        },
        VectorEmbedding {
            id: "2".to_string(),
            embedding: vec![0.0, 1.0, 0.0],
            timestamp: 1001,
            metadata: HashMap::new(),
        },
    ];
    
    let result = backend.store_embeddings(embeddings).await;
    assert!(matches!(result, Err(Status { .. })));
}

#[tokio::test]
async fn test_simd_accuracy() {
    let backend = VectorBackend::new(1000, 3600);
    
    // Create test embedding with known cosine similarity
    let embeddings = vec![
        VectorEmbedding {
            id: "1".to_string(),
            embedding: vec![1.0, 1.0, 1.0, 1.0],
            timestamp: 1000,
            metadata: HashMap::new(),
        },
    ];

    backend.store_embeddings(embeddings).await.unwrap();

    // Query with vector that should have 0.5 similarity
    let query = vec![0.5, 0.5, 0.5, 0.5];
    let results = backend.query_similar_vectors(query, 1, 0.0).await.unwrap();

    assert_eq!(results.len(), 1);
    assert!((results[0].1 - 0.5).abs() < 1e-6);
}
