//! Context storage for RAG/CAG functionality.
//!
//! This module provides embedding storage with metadata for retrieval-augmented
//! and cache-augmented generation. It uses the existing StorageBackend trait
//! for ADBC-agnostic database operations.
//!
//! # Data Model
//!
//! - **Embeddings + metadata only**: Text is stored elsewhere
//! - **Vector similarity**: Uses existing VectorizedOperator for cosine similarity
//! - **Session-based retrieval**: For cache-augmented generation (CAG)
//! - **Quality filtering**: Retrieve only high-quality context

use super::StorageBackend;
use duckdb::arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    FixedSizeListArray, FixedSizeListBuilder, Float32Builder,
};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tonic::Status;

/// Default embedding dimension (can be configured per model)
pub const DEFAULT_EMBEDDING_DIM: i32 = 768;

/// Context embedding schema - embeddings + metadata, no text
pub fn context_schema(embedding_dim: i32) -> Schema {
    Schema::new(vec![
        // Identity
        Field::new("id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, true),
        // Embedding vector as fixed-size list of floats
        // Note: inner field nullable=true to match FixedSizeListBuilder default
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim,
            ),
            false,
        ),
        // Metadata for filtering
        Field::new("model_id", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("quality_score", DataType::Float64, true),
        Field::new("token_count", DataType::Int64, true),
    ])
}

/// A single context record for storage
#[derive(Debug, Clone)]
pub struct ContextRecord {
    pub id: String,
    pub conversation_id: String,
    pub session_id: Option<String>,
    pub embedding: Vec<f32>,
    pub model_id: String,
    pub timestamp: i64,
    pub quality_score: Option<f64>,
    pub token_count: Option<i64>,
}

impl ContextRecord {
    /// Convert to Arrow RecordBatch for storage
    pub fn to_record_batch(&self) -> Result<RecordBatch, Status> {
        let embedding_dim = self.embedding.len() as i32;
        let schema = Arc::new(context_schema(embedding_dim));

        // Build the fixed-size list array for embedding
        let mut embedding_builder = FixedSizeListBuilder::new(Float32Builder::new(), embedding_dim);
        let values = embedding_builder.values();
        for &val in &self.embedding {
            values.append_value(val);
        }
        embedding_builder.append(true);
        let embedding_array = embedding_builder.finish();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![self.id.as_str()])) as ArrayRef,
                Arc::new(StringArray::from(vec![self.conversation_id.as_str()])) as ArrayRef,
                Arc::new(StringArray::from(vec![self.session_id.as_deref()])) as ArrayRef,
                Arc::new(embedding_array) as ArrayRef,
                Arc::new(StringArray::from(vec![self.model_id.as_str()])) as ArrayRef,
                Arc::new(Int64Array::from(vec![self.timestamp])) as ArrayRef,
                Arc::new(Float64Array::from(vec![self.quality_score])) as ArrayRef,
                Arc::new(Int64Array::from(vec![self.token_count])) as ArrayRef,
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        Ok(batch)
    }

    /// Create multiple records as a single batch (more efficient for bulk inserts)
    pub fn batch_to_record_batch(records: &[ContextRecord]) -> Result<RecordBatch, Status> {
        if records.is_empty() {
            return Err(Status::invalid_argument("Cannot create batch from empty records"));
        }

        let embedding_dim = records[0].embedding.len() as i32;
        let schema = Arc::new(context_schema(embedding_dim));

        // Build arrays
        let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
        let conversation_ids: Vec<&str> = records.iter().map(|r| r.conversation_id.as_str()).collect();
        let session_ids: Vec<Option<&str>> = records.iter().map(|r| r.session_id.as_deref()).collect();
        let model_ids: Vec<&str> = records.iter().map(|r| r.model_id.as_str()).collect();
        let timestamps: Vec<i64> = records.iter().map(|r| r.timestamp).collect();
        let quality_scores: Vec<Option<f64>> = records.iter().map(|r| r.quality_score).collect();
        let token_counts: Vec<Option<i64>> = records.iter().map(|r| r.token_count).collect();

        // Build fixed-size list array for embeddings
        let mut embedding_builder = FixedSizeListBuilder::new(Float32Builder::new(), embedding_dim);
        for record in records {
            let values = embedding_builder.values();
            for &val in &record.embedding {
                values.append_value(val);
            }
            embedding_builder.append(true);
        }
        let embedding_array = embedding_builder.finish();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(conversation_ids)) as ArrayRef,
                Arc::new(StringArray::from(session_ids)) as ArrayRef,
                Arc::new(embedding_array) as ArrayRef,
                Arc::new(StringArray::from(model_ids)) as ArrayRef,
                Arc::new(Int64Array::from(timestamps)) as ArrayRef,
                Arc::new(Float64Array::from(quality_scores)) as ArrayRef,
                Arc::new(Int64Array::from(token_counts)) as ArrayRef,
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?;

        Ok(batch)
    }
}

/// Result of a similarity search
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub conversation_id: String,
    pub score: f32,
    pub quality_score: Option<f64>,
}

/// Context store for embedding storage and retrieval
pub struct ContextStore<B: StorageBackend> {
    backend: Arc<B>,
    table_name: String,
    embedding_dim: i32,
}

impl<B: StorageBackend> ContextStore<B> {
    /// Create a new context store
    pub fn new(backend: Arc<B>, table_name: &str, embedding_dim: i32) -> Self {
        Self {
            backend,
            table_name: table_name.to_string(),
            embedding_dim,
        }
    }

    /// Create with default embedding dimension (768)
    pub fn with_defaults(backend: Arc<B>, table_name: &str) -> Self {
        Self::new(backend, table_name, DEFAULT_EMBEDDING_DIM)
    }

    /// Initialize the context table
    pub async fn init(&self) -> Result<(), Status> {
        let schema = context_schema(self.embedding_dim);
        self.backend.create_table(&self.table_name, &schema).await
    }

    /// Store a single context record
    pub async fn store(&self, record: ContextRecord) -> Result<(), Status> {
        let batch = record.to_record_batch()?;
        self.backend.insert_into_table(&self.table_name, batch).await
    }

    /// Store multiple context records in a batch
    pub async fn store_batch(&self, records: &[ContextRecord]) -> Result<(), Status> {
        if records.is_empty() {
            return Ok(());
        }
        let batch = ContextRecord::batch_to_record_batch(records)?;
        self.backend.insert_into_table(&self.table_name, batch).await
    }

    /// Get recent context by session (CAG - no semantic search)
    pub async fn recent_by_session(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<String>, Status> {
        // Escape single quotes to prevent SQL injection
        let escaped_session_id = session_id.replace('\'', "''");
        let sql = format!(
            "SELECT conversation_id FROM {} WHERE session_id = '{}' ORDER BY timestamp DESC LIMIT {}",
            self.table_name, escaped_session_id, limit
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;

        // Extract conversation_ids from result
        let mut conversation_ids = Vec::new();
        if batch.num_rows() > 0 {
            if let Some(col) = batch.column_by_name("conversation_id") {
                if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            conversation_ids.push(arr.value(i).to_string());
                        }
                    }
                }
            }
        }

        Ok(conversation_ids)
    }

    /// Get recent context by model with quality filter
    pub async fn recent_by_model(
        &self,
        model_id: &str,
        min_quality: Option<f64>,
        limit: usize,
    ) -> Result<Vec<String>, Status> {
        // Escape single quotes to prevent SQL injection
        let escaped_model_id = model_id.replace('\'', "''");
        let quality_filter = min_quality
            .map(|q| format!(" AND quality_score >= {}", q))
            .unwrap_or_default();

        let sql = format!(
            "SELECT conversation_id FROM {} WHERE model_id = '{}'{} ORDER BY timestamp DESC LIMIT {}",
            self.table_name, escaped_model_id, quality_filter, limit
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;

        // Extract conversation_ids from result
        let mut conversation_ids = Vec::new();
        if batch.num_rows() > 0 {
            if let Some(col) = batch.column_by_name("conversation_id") {
                if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        if !arr.is_null(i) {
                            conversation_ids.push(arr.value(i).to_string());
                        }
                    }
                }
            }
        }

        Ok(conversation_ids)
    }

    /// Similarity search using cosine similarity
    ///
    /// This loads all embeddings and computes similarity in-memory.
    /// For large datasets, consider using an external vector database via ADBC.
    pub async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_quality: Option<f64>,
    ) -> Result<Vec<SearchResult>, Status> {
        // Build query with optional quality filter
        let quality_filter = min_quality
            .map(|q| format!(" WHERE quality_score >= {}", q))
            .unwrap_or_default();

        let sql = format!(
            "SELECT id, conversation_id, embedding, quality_score FROM {}{}",
            self.table_name, quality_filter
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let batch = self.backend.query_sql(&handle).await?;

        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }

        // Extract columns
        let id_col = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("Missing id column"))?;

        let conv_id_col = batch
            .column_by_name("conversation_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("Missing conversation_id column"))?;

        let embedding_col = batch
            .column_by_name("embedding")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or_else(|| Status::internal("Missing embedding column"))?;

        let quality_col = batch
            .column_by_name("quality_score")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

        // Compute similarities
        let query_norm = (query_embedding.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let mut results: Vec<SearchResult> = Vec::with_capacity(batch.num_rows());

        for i in 0..batch.num_rows() {
            // Get embedding vector from fixed-size list
            let embedding_values = embedding_col.value(i);
            let embedding = embedding_values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Status::internal("Invalid embedding data type"))?;

            // Compute cosine similarity
            let mut dot_product = 0.0f32;
            let mut stored_norm_sq = 0.0f32;
            for j in 0..embedding.len() {
                let stored_val = embedding.value(j);
                let query_val = query_embedding.get(j).copied().unwrap_or(0.0);
                dot_product += stored_val * query_val;
                stored_norm_sq += stored_val * stored_val;
            }
            let stored_norm = stored_norm_sq.sqrt();
            let score = if query_norm > 0.0 && stored_norm > 0.0 {
                dot_product / (query_norm * stored_norm)
            } else {
                0.0
            };

            results.push(SearchResult {
                id: id_col.value(i).to_string(),
                conversation_id: conv_id_col.value(i).to_string(),
                score,
                quality_score: quality_col.and_then(|c| {
                    if c.is_null(i) {
                        None
                    } else {
                        Some(c.value(i))
                    }
                }),
            });
        }

        // Sort by score descending and take top-k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Delete old context entries
    pub async fn cleanup_before(&self, timestamp: i64) -> Result<usize, Status> {
        let sql = format!(
            "DELETE FROM {} WHERE timestamp < {}",
            self.table_name, timestamp
        );
        let handle = self.backend.prepare_sql(&sql).await?;
        let result = self.backend.query_sql(&handle).await?;

        // Return number of deleted rows (if available)
        Ok(result.num_rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_schema() {
        let schema = context_schema(768);
        assert_eq!(schema.fields().len(), 8);
        assert!(schema.field_with_name("embedding").is_ok());
        assert!(schema.field_with_name("conversation_id").is_ok());
    }

    #[test]
    fn test_context_record_to_batch() {
        let record = ContextRecord {
            id: "test-id".to_string(),
            conversation_id: "conv-123".to_string(),
            session_id: Some("sess-456".to_string()),
            embedding: vec![0.1; 768],
            model_id: "qwen3-small".to_string(),
            timestamp: 1234567890,
            quality_score: Some(0.85),
            token_count: Some(100),
        };

        let batch = record.to_record_batch().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 8);
    }

    #[test]
    fn test_batch_records() {
        let records = vec![
            ContextRecord {
                id: "id-1".to_string(),
                conversation_id: "conv-1".to_string(),
                session_id: None,
                embedding: vec![0.1; 768],
                model_id: "model-1".to_string(),
                timestamp: 1000,
                quality_score: Some(0.9),
                token_count: Some(50),
            },
            ContextRecord {
                id: "id-2".to_string(),
                conversation_id: "conv-2".to_string(),
                session_id: Some("sess-1".to_string()),
                embedding: vec![0.2; 768],
                model_id: "model-1".to_string(),
                timestamp: 2000,
                quality_score: None,
                token_count: None,
            },
        ];

        let batch = ContextRecord::batch_to_record_batch(&records).unwrap();
        assert_eq!(batch.num_rows(), 2);
    }
}
