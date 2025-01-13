use std::sync::Arc;
use tokio::sync::RwLock;
use crate::storage::simd::SimdOps;
use tonic::Status;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use std::collections::HashMap;

use crate::config::Credentials;
use crate::metrics::MetricRecord;
use crate::storage::table_manager::{HyprTableManager, HyprAggregationView};
use crate::storage::HyprStorageBackend;
use crate::aggregation::{AggregateFunction, GroupBy, AggregateResult};

/// Vector embedding with metadata, optimized for SIMD operations.
///
/// This structure stores vector embeddings in a format that enables efficient
/// SIMD processing on different architectures. The implementation provides:
///
/// - Zero-copy operations where possible through Arc and reference sharing
/// - Architecture-specific SIMD optimizations (SSE/AVX on x86/x86_64, NEON on ARM)
/// - Automatic handling of unaligned data
/// - Efficient memory layout for vector operations
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
///
/// let embedding = HyprVectorEmbedding {
///     id: "vec1".to_string(),
///     embedding: vec![0.1, 0.2, 0.3, 0.4],
///     timestamp: 1234567890,
///     metadata: HashMap::new(),
/// };
///
/// // Vector will be automatically normalized using SIMD
/// let backend = HyprVectorBackend::new(1000, 3600);
/// backend.store_embeddings(vec![embedding]).await.unwrap();
/// ```
///
/// # SIMD Implementation Details
///
/// The implementation uses different SIMD approaches based on architecture:
///
/// - x86/x86_64: Uses `safe_arch` for SSE/AVX operations
/// - Other architectures: Uses `packed_simd` for portable SIMD
///
/// Operations are optimized to:
/// - Process vectors in 4-float chunks where possible
/// - Handle remaining elements separately
/// - Maintain proper alignment for optimal performance
/// - Provide consistent results across architectures
#[derive(Debug, Clone)]
pub struct HyprVectorEmbedding {
    /// Unique identifier for the embedding
    pub id: String,
    /// The vector embedding data, stored in a format optimized for SIMD operations.
    /// The length should ideally be a multiple of 4 for optimal performance.
    pub embedding: Vec<f32>,
    /// Timestamp when the embedding was created
    pub timestamp: i64,
    /// Additional metadata about the embedding
    pub metadata: HashMap<String, String>,
}

/// Cache entry for vector embeddings
#[derive(Debug, Clone)]
struct CacheEntry {
    embedding: HyprVectorEmbedding,
    last_accessed: std::time::SystemTime,
}

/// Vector storage backend implementation
#[derive(Clone)]
pub struct HyprVectorBackend {
    /// Table manager for schema and view management
    table_manager: HyprTableManager,
    /// Cache for frequently accessed embeddings
    embedding_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Cache eviction threshold (in seconds)
    cache_ttl: u64,
}

impl HyprVectorBackend {
    pub fn new(max_cache_size: usize, cache_ttl: u64) -> Self {
        Self {
            table_manager: HyprTableManager::new(),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size,
            cache_ttl,
        }
    }

    /// Store vector embeddings in the backend using SIMD optimizations
    pub async fn store_embeddings(&self, embeddings: Vec<HyprVectorEmbedding>) -> Result<(), Status> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // Validate embedding dimensions are consistent
        let dim = embeddings[0].embedding.len();
        if !embeddings.iter().all(|e| e.embedding.len() == dim) {
            return Err(Status::invalid_argument("Inconsistent embedding dimensions"));
        }

        let simd = SimdOps::new();
        let normalized = embeddings.into_iter().map(|mut e| {
            // Calculate norm using SIMD
            let mut sum = 0.0f32;
            for chunk in e.embedding.chunks_exact(4) {
                let squared = simd.f32x4_mul(chunk, chunk);
                sum += squared.iter().sum::<f32>();
            }
            for &x in e.embedding.chunks_exact(4).remainder() {
                sum += x * x;
            }
            let norm = sum.sqrt();

            // Normalize using SIMD
            for chunk in e.embedding.chunks_exact_mut(4) {
                let normalized = simd.f32x4_div(chunk, &[norm; 4]);
                chunk.copy_from_slice(&normalized);
            }
            for x in e.embedding.chunks_exact_mut(4).into_remainder() {
                *x /= norm;
            }

            e
        }).collect::<Vec<_>>();

        self.cache_embeddings(normalized).await
    }

    /// Query similar vectors using cosine similarity with SIMD acceleration
    pub async fn query_similar_vectors(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(HyprVectorEmbedding, f32)>, Status> {
        if query_vector.is_empty() {
            return Err(Status::invalid_argument("Empty query vector"));
        }

        let simd = SimdOps::new();

        // Normalize query vector using SIMD
        let mut norm = 0.0f32;
        for chunk in query_vector.chunks_exact(4) {
            let squared = simd.f32x4_mul(chunk, chunk);
            norm += squared.iter().sum::<f32>();
        }
        for &x in query_vector.chunks_exact(4).remainder() {
            norm += x * x;
        }
        norm = norm.sqrt();

        let query_vector: Vec<f32> = query_vector.iter().map(|&x| x / norm).collect();

        // Calculate similarities using SIMD
        let cache = self.embedding_cache.read().await;
        let mut similarities: Vec<(String, f32)> = cache.iter().map(|(id, entry)| {
            let mut similarity = 0.0f32;
            let chunks = query_vector.chunks_exact(4).zip(entry.embedding.embedding.chunks_exact(4));
            
            for (q, e) in chunks {
                let product = simd.f32x4_mul(q, e);
                similarity += product.iter().sum::<f32>();
            }

            // Handle remaining elements
            let remainder = query_vector.chunks_exact(4).remainder();
            for (q, e) in remainder.iter().zip(entry.embedding.embedding.chunks_exact(4).remainder()) {
                similarity += q * e;
            }

            (id.clone(), similarity)
        }).collect();

        // Sort by similarity and take top k
        similarities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let results = similarities.into_iter()
            .take(k)
            .filter(|(_, sim)| *sim >= min_similarity)
            .map(|(id, sim)| {
                let entry = cache.get(&id).unwrap();
                (entry.embedding.clone(), sim)
            })
            .collect();

        Ok(results)
    }

    /// Cache embeddings for faster access
    async fn cache_embeddings(&self, embeddings: Vec<HyprVectorEmbedding>) -> Result<(), Status> {
        let mut cache = self.embedding_cache.write().await;
        
        // Add new embeddings to cache
        for embedding in embeddings {
            if cache.len() >= self.max_cache_size {
                // Evict oldest entries if cache is full
                self.evict_old_entries(&mut cache).await?;
            }
            
            cache.insert(embedding.id.clone(), CacheEntry {
                embedding,
                last_accessed: std::time::SystemTime::now(),
            });
        }
        
        Ok(())
    }

    /// Get cached embeddings by ID
    async fn get_cached_embeddings(&self, ids: &[String]) -> Result<Vec<HyprVectorEmbedding>, Status> {
        let mut cache = self.embedding_cache.write().await;
        let now = std::time::SystemTime::now();
        
        let mut results = Vec::new();
        for id in ids {
            if let Some(entry) = cache.get_mut(id) {
                entry.last_accessed = now;
                results.push(entry.embedding.clone());
            }
        }
        
        Ok(results)
    }

    /// Evict old entries from cache
    async fn evict_old_entries(&self, cache: &mut HashMap<String, CacheEntry>) -> Result<(), Status> {
        let now = std::time::SystemTime::now();
        let ttl = std::time::Duration::from_secs(self.cache_ttl);
        
        cache.retain(|_, entry| {
            entry.last_accessed + ttl > now
        });
        
        Ok(())
    }
}

#[async_trait]
impl HyprStorageBackend for HyprVectorBackend {
    async fn init(&self) -> Result<(), Status> {
        // Initialize storage tables and indexes
        Ok(())
    }

    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Store metrics data
        Ok(())
    }

    async fn query_metrics(&self, from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Query metrics data
        Ok(vec![])
    }

    async fn prepare_sql(&self, _query: &str) -> Result<Vec<u8>, Status> {
        Err(Status::unimplemented("SQL queries not supported for vector backend"))
    }

    async fn query_sql(&self, _statement_handle: &[u8]) -> Result<Vec<MetricRecord>, Status> {
        Err(Status::unimplemented("SQL queries not supported for vector backend"))
    }

    async fn aggregate_metrics(
        &self,
        _function: AggregateFunction,
        _group_by: &GroupBy,
        _from_timestamp: i64,
        _to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        Err(Status::unimplemented("Aggregation not supported for vector backend"))
    }

    fn new_with_options(
        _connection_string: &str,
        options: &HashMap<String, String>,
        _credentials: Option<&Credentials>,
    ) -> Result<Self, Status>
    where
        Self: Sized,
    {
        let max_cache_size = options
            .get("max_cache_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        
        let cache_ttl = options
            .get("cache_ttl")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);
        
        Ok(Self::new(max_cache_size, cache_ttl))
    }

    async fn create_table(&self, table_name: &str, schema: &Schema) -> Result<(), Status> {
        self.table_manager.create_table(table_name.to_string(), schema.clone()).await
    }

    // Removed fallback implementations since SimdOps handles that internally

    async fn insert_into_table(&self, _table_name: &str, _batch: RecordBatch) -> Result<(), Status> {
        Err(Status::unimplemented("Table operations not supported for vector backend"))
    }

    async fn query_table(&self, _table_name: &str, _projection: Option<Vec<String>>) -> Result<RecordBatch, Status> {
        Err(Status::unimplemented("Table operations not supported for vector backend"))
    }

    async fn create_aggregation_view(&self, _view: &HyprAggregationView) -> Result<(), Status> {
        Err(Status::unimplemented("Views not supported for vector backend"))
    }

    async fn query_aggregation_view(&self, _view_name: &str) -> Result<RecordBatch, Status> {
        Err(Status::unimplemented("Views not supported for vector backend"))
    }

    async fn drop_table(&self, table_name: &str) -> Result<(), Status> {
        self.table_manager.drop_table(table_name).await
    }

    async fn drop_aggregation_view(&self, _view_name: &str) -> Result<(), Status> {
        Err(Status::unimplemented("Views not supported for vector backend"))
    }

    fn table_manager(&self) -> &HyprTableManager {
        &self.table_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_vector_operations() {
        let backend = HyprVectorBackend::new(1000, 3600);
        
        // Test vector storage and normalization
        let embeddings = vec![
            HyprVectorEmbedding {
                id: "vec1".to_string(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                timestamp: 1,
                metadata: HashMap::new(),
            },
            HyprVectorEmbedding {
                id: "vec2".to_string(),
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                timestamp: 2,
                metadata: HashMap::new(),
            },
        ];

        backend.store_embeddings(embeddings).await.unwrap();

        // Test similarity search
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let results = backend.query_similar_vectors(query, 2, 0.0).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].1 > results[1].1); // First result should be more similar
    }

    #[tokio::test]
    async fn test_edge_cases() {
        let backend = HyprVectorBackend::new(1000, 3600);

        // Test empty embeddings
        assert!(backend.store_embeddings(vec![]).await.is_ok());

        // Test empty query
        assert!(backend.query_similar_vectors(vec![], 10, 0.5).await.is_err());

        // Test inconsistent dimensions
        let embeddings = vec![
            HyprVectorEmbedding {
                id: "vec1".to_string(),
                embedding: vec![1.0, 0.0],
                timestamp: 1,
                metadata: HashMap::new(),
            },
            HyprVectorEmbedding {
                id: "vec2".to_string(),
                embedding: vec![0.0, 1.0, 0.0], // Different dimension
                timestamp: 2,
                metadata: HashMap::new(),
            },
        ];
        assert!(backend.store_embeddings(embeddings).await.is_err());
    }

    #[tokio::test]
    async fn test_simd_performance() {
        let backend = HyprVectorBackend::new(1000, 3600);
        let dim = 1024; // Typical embedding dimension
        let num_vectors = 1000;

        // Create test embeddings
        let embeddings: Vec<HyprVectorEmbedding> = (0..num_vectors)
            .map(|i| HyprVectorEmbedding {
                id: format!("vec{}", i),
                embedding: (0..dim).map(|x| x as f32).collect(),
                timestamp: i as i64,
                metadata: HashMap::new(),
            })
            .collect();

        // Measure store performance
        let start = Instant::now();
        backend.store_embeddings(embeddings).await.unwrap();
        let store_time = start.elapsed();

        // Measure query performance
        let query = vec![1.0; dim];
        let start = Instant::now();
        let results = backend.query_similar_vectors(query, 10, 0.5).await.unwrap();
        let query_time = start.elapsed();

        println!("Vector Operations Performance:");
        println!("  Store {} vectors ({} dims): {:?}", num_vectors, dim, store_time);
        println!("  Query top 10 similar vectors: {:?}", query_time);
        println!("  Results found: {}", results.len());
    }

    #[tokio::test]
    async fn test_numerical_stability() {
        let backend = HyprVectorBackend::new(1000, 3600);

        // Test with very small values
        let embeddings = vec![
            HyprVectorEmbedding {
                id: "small".to_string(),
                embedding: vec![1e-30, 1e-30, 1e-30, 1e-30],
                timestamp: 1,
                metadata: HashMap::new(),
            },
        ];
        assert!(backend.store_embeddings(embeddings).await.is_ok());

        // Test with very large values
        let embeddings = vec![
            HyprVectorEmbedding {
                id: "large".to_string(),
                embedding: vec![1e30, 1e30, 1e30, 1e30],
                timestamp: 2,
                metadata: HashMap::new(),
            },
        ];
        assert!(backend.store_embeddings(embeddings).await.is_ok());

        // Test with mixed values
        let query = vec![1e-15, 1e15, 1e-15, 1e15];
        let results = backend.query_similar_vectors(query, 2, 0.0).await.unwrap();
        assert!(!results.is_empty());
        assert!(!results.iter().any(|(_, sim)| sim.is_nan()));
    }
}
