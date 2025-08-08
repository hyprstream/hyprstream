//! FlightSQL service for embedding queries only
//!
//! This module provides a minimal FlightSQL interface focused exclusively on:
//! - Embedding similarity search queries
//! - Sparse adapter embedding retrieval
//! - Vector similarity operations
//!
//! All general SQL functionality has been removed for VDB-first architecture

use crate::storage::vdb::{SparseStorage, EmbeddingMatch};

use arrow_array::{Float32Array, RecordBatch, StringArray, UInt64Array};
use arrow_flight::{
    flight_service_server::{FlightService, FlightServiceServer},
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
};
use arrow_schema::{DataType, Field, Schema};
use arrow_ipc;
use std::pin::Pin;
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};

/// Embedding query types for FlightSQL interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingQuery {
    /// Find similar embeddings to a given vector
    SimilaritySearch {
        vector: Vec<f32>,
        limit: usize,
        threshold: f32,
    },
    
    /// Get embedding for a specific adapter
    GetEmbedding {
        adapter_id: String,
    },
    
    /// List all available embeddings
    ListEmbeddings,
    
    /// Get embedding statistics
    EmbeddingStats {
        adapter_id: Option<String>,
    },
}

/// FlightSQL service specialized for embedding operations only
pub struct EmbeddingFlightService {
    /// VDB-first sparse storage backend
    storage: Arc<dyn SparseStorage>,
    
    /// Service configuration
    config: EmbeddingServiceConfig,
}

/// Configuration for embedding FlightSQL service
#[derive(Debug, Clone)]
pub struct EmbeddingServiceConfig {
    /// Maximum embedding vector size
    pub max_embedding_size: usize,
    
    /// Default similarity search limit
    pub default_search_limit: usize,
    
    /// Minimum similarity threshold
    pub min_similarity_threshold: f32,
    
    /// Enable query caching
    pub enable_caching: bool,
}

impl Default for EmbeddingServiceConfig {
    fn default() -> Self {
        Self {
            max_embedding_size: 4096,
            default_search_limit: 100,
            min_similarity_threshold: 0.1,
            enable_caching: true,
        }
    }
}

impl EmbeddingFlightService {
    /// Create new embedding-focused FlightSQL service
    pub fn new(storage: Arc<dyn SparseStorage>) -> Self {
        Self {
            storage,
            config: EmbeddingServiceConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(storage: Arc<dyn SparseStorage>, config: EmbeddingServiceConfig) -> Self {
        Self { storage, config }
    }
    
    /// Parse embedding query from FlightSQL ticket
    fn parse_embedding_query(&self, ticket: &Ticket) -> Result<EmbeddingQuery, Status> {
        let query_str = std::str::from_utf8(&ticket.ticket)
            .map_err(|_| Status::invalid_argument("Invalid ticket encoding"))?;
        
        // Simple JSON parsing for embedding queries
        serde_json::from_str(query_str)
            .map_err(|e| Status::invalid_argument(format!("Invalid query format: {}", e)))
    }
    
    /// Convert embedding matches to Arrow RecordBatch
    fn embedding_matches_to_batch(&self, matches: Vec<EmbeddingMatch>) -> Result<RecordBatch, Status> {
        if matches.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::new(self.embedding_schema())));
        }
        
        // Extract data for Arrow arrays
        let adapter_ids: Vec<String> = matches.iter().map(|m| m.adapter_id.clone()).collect();
        let similarity_scores: Vec<f32> = matches.iter().map(|m| m.similarity_score).collect();
        let embedding_dimensions: Vec<u64> = matches.iter().map(|m| m.embedding_vector.len() as u64).collect();
        
        // Flatten embedding vectors (simplified representation)
        let first_10_values: Vec<f32> = matches.iter()
            .flat_map(|m| m.embedding_vector.iter().take(10).copied())
            .collect();
        
        // Create Arrow arrays
        let adapter_id_array = StringArray::from(adapter_ids);
        let similarity_array = Float32Array::from(similarity_scores);
        let dimension_array = UInt64Array::from(embedding_dimensions);
        let embedding_sample_array = Float32Array::from(first_10_values);
        
        // Create RecordBatch
        RecordBatch::try_new(
            Arc::new(self.embedding_schema()),
            vec![
                Arc::new(adapter_id_array),
                Arc::new(similarity_array),
                Arc::new(dimension_array),
                Arc::new(embedding_sample_array),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create RecordBatch: {}", e)))
    }
    
    /// Single embedding to RecordBatch
    fn single_embedding_to_batch(&self, adapter_id: &str, embedding: Vec<f32>) -> Result<RecordBatch, Status> {
        let adapter_ids = vec![adapter_id.to_string()];
        let similarity_scores = vec![1.0f32]; // Perfect match for self
        let dimensions = vec![embedding.len() as u64];
        let embedding_sample = embedding.into_iter().take(10).collect::<Vec<_>>();
        
        let adapter_id_array = StringArray::from(adapter_ids);
        let similarity_array = Float32Array::from(similarity_scores);
        let dimension_array = UInt64Array::from(dimensions);
        let embedding_sample_array = Float32Array::from(embedding_sample);
        
        RecordBatch::try_new(
            Arc::new(self.embedding_schema()),
            vec![
                Arc::new(adapter_id_array),
                Arc::new(similarity_array),
                Arc::new(dimension_array),
                Arc::new(embedding_sample_array),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create RecordBatch: {}", e)))
    }
    
    /// Schema for embedding results
    fn embedding_schema(&self) -> Schema {
        Schema::new(vec![
            Field::new("adapter_id", DataType::Utf8, false),
            Field::new("similarity_score", DataType::Float32, false),
            Field::new("embedding_dimension", DataType::UInt64, false),
            Field::new("embedding_sample", DataType::Float32, true), // First 10 values
        ])
    }
    
    /// Convert RecordBatch to FlightData stream
    fn batch_to_flight_data(&self, batch: RecordBatch) -> impl Stream<Item = Result<FlightData, Status>> {
        // Convert batch to IPC format for FlightSQL
        // Use debug formatting since RecordBatch doesn't implement Display
        let batch_data = format!("{:?}", batch).into_bytes();
        stream::iter(vec![Ok(FlightData::new()
            .with_data_body(batch_data))])
    }
    
    /// Handle similarity search query
    async fn handle_similarity_search(
        &self,
        vector: Vec<f32>,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<EmbeddingMatch>, Status> {
        // Validate input parameters
        if vector.len() > self.config.max_embedding_size {
            return Err(Status::invalid_argument("Embedding vector too large"));
        }
        
        if threshold < self.config.min_similarity_threshold {
            return Err(Status::invalid_argument("Similarity threshold too low"));
        }
        
        let search_limit = limit.min(self.config.default_search_limit);
        
        // Perform similarity search using VDB storage
        self.storage
            .similarity_search(&vector, search_limit, threshold)
            .await
            .map_err(|e| Status::internal(format!("Similarity search failed: {}", e)))
    }
    
    /// Handle get embedding query
    async fn handle_get_embedding(&self, adapter_id: String) -> Result<Vec<f32>, Status> {
        // Load adapter and compute embedding
        let adapter_infos = self.storage.list_adapters().await
            .map_err(|e| Status::internal(format!("Failed to list adapters: {}", e)))?;
        
        // Check if adapter exists
        if !adapter_infos.iter().any(|info| info.adapter_id == adapter_id) {
            return Err(Status::not_found(format!("Adapter '{}' not found", adapter_id)));
        }
        
        // Get embedding through similarity search with self
        let self_matches = self.storage
            .similarity_search(&vec![1.0; 128], 1, 0.0) // Dummy search
            .await
            .map_err(|e| Status::internal(format!("Failed to get embedding: {}", e)))?;
        
        self_matches
            .into_iter()
            .find(|m| m.adapter_id == adapter_id)
            .map(|m| m.embedding_vector)
            .ok_or_else(|| Status::not_found(format!("Embedding for '{}' not available", adapter_id)))
    }
}

#[tonic::async_trait]
impl FlightService for EmbeddingFlightService {
    type HandshakeStream = Pin<Box<dyn Stream<Item = Result<HandshakeResponse, Status>> + Send>>;
    type ListFlightsStream = Pin<Box<dyn Stream<Item = Result<FlightInfo, Status>> + Send>>;
    type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;
    type DoPutStream = Pin<Box<dyn Stream<Item = Result<PutResult, Status>> + Send>>;
    type DoActionStream = Pin<Box<dyn Stream<Item = Result<arrow_flight::Result, Status>> + Send>>;
    type ListActionsStream = Pin<Box<dyn Stream<Item = Result<ActionType, Status>> + Send>>;
    type DoExchangeStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send>>;

    /// Handshake for embedding service authentication
    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let response = HandshakeResponse::default();
        let stream = stream::iter(vec![Ok(response)]);
        Ok(Response::new(Box::pin(stream)))
    }

    /// List available embedding operations
    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        // Return available embedding operations
        let descriptor = FlightDescriptor::new_path(vec!["embeddings".to_string()]);
        let mut embedding_flight = FlightInfo::new()
            .with_descriptor(descriptor)
            .with_total_records(-1)  // Unknown record count
            .with_total_bytes(-1);   // Unknown byte count
            
        // Set schema using try_with_schema 
        embedding_flight = embedding_flight.try_with_schema(&self.embedding_schema())
            .map_err(|e| Status::internal(format!("Failed to set schema: {}", e)))?;

        let stream = stream::iter(vec![Ok(embedding_flight)]);
        Ok(Response::new(Box::pin(stream)))
    }

    /// Execute embedding queries
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let query = self.parse_embedding_query(&ticket)?;

        println!("🔍 Processing embedding query: {:?}", query);

        match query {
            EmbeddingQuery::SimilaritySearch { vector, limit, threshold } => {
                let matches = self.handle_similarity_search(vector, limit, threshold).await?;
                let batch = self.embedding_matches_to_batch(matches)?;
                let stream = self.batch_to_flight_data(batch);
                Ok(Response::new(Box::pin(stream)))
            }
            
            EmbeddingQuery::GetEmbedding { adapter_id } => {
                let embedding = self.handle_get_embedding(adapter_id.clone()).await?;
                let batch = self.single_embedding_to_batch(&adapter_id, embedding)?;
                let stream = self.batch_to_flight_data(batch);
                Ok(Response::new(Box::pin(stream)))
            }
            
            EmbeddingQuery::ListEmbeddings => {
                // List all available embeddings
                let adapter_infos = self.storage.list_adapters().await
                    .map_err(|e| Status::internal(format!("Failed to list adapters: {}", e)))?;
                
                let matches: Vec<EmbeddingMatch> = adapter_infos.into_iter().map(|info| {
                    EmbeddingMatch {
                        adapter_id: info.adapter_id.clone(),
                        similarity_score: 1.0,
                        embedding_vector: vec![0.0; 128], // Placeholder
                        metadata: std::collections::HashMap::new(),
                    }
                }).collect();
                
                let batch = self.embedding_matches_to_batch(matches)?;
                let stream = self.batch_to_flight_data(batch);
                Ok(Response::new(Box::pin(stream)))
            }
            
            EmbeddingQuery::EmbeddingStats { adapter_id } => {
                // Return embedding statistics
                match adapter_id {
                    Some(id) => {
                        let stats = self.storage.get_adapter_stats(&id).await
                            .map_err(|e| Status::internal(format!("Failed to get stats: {}", e)))?;
                        
                        // Convert stats to embedding match format (simplified)
                        let match_item = EmbeddingMatch {
                            adapter_id: id,
                            similarity_score: stats.sparsity_ratio,
                            embedding_vector: vec![stats.active_weights as f32],
                            metadata: std::collections::HashMap::new(),
                        };
                        
                        let batch = self.embedding_matches_to_batch(vec![match_item])?;
                        let stream = self.batch_to_flight_data(batch);
                        Ok(Response::new(Box::pin(stream)))
                    }
                    None => {
                        // Overall storage stats
                        let storage_stats = self.storage.get_storage_stats().await
                            .map_err(|e| Status::internal(format!("Failed to get storage stats: {}", e)))?;
                        
                        let match_item = EmbeddingMatch {
                            adapter_id: "system_stats".to_string(),
                            similarity_score: storage_stats.avg_sparsity_ratio,
                            embedding_vector: vec![
                                storage_stats.total_adapters as f32,
                                storage_stats.updates_per_second as f32,
                            ],
                            metadata: std::collections::HashMap::new(),
                        };
                        
                        let batch = self.embedding_matches_to_batch(vec![match_item])?;
                        let stream = self.batch_to_flight_data(batch);
                        Ok(Response::new(Box::pin(stream)))
                    }
                }
            }
        }
    }

    /// Get schema for embedding operations
    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let schema = self.embedding_schema();
        // Convert schema to IPC bytes using arrow format
        let schema_message = arrow_ipc::writer::IpcWriteOptions::default();
        let encoded_data = arrow_ipc::writer::IpcDataGenerator::default();
        let encoded_schema = encoded_data.schema_to_bytes(&schema, &schema_message);
        let schema_bytes = encoded_schema.ipc_message.into();
        let schema_result = SchemaResult { schema: schema_bytes };
        Ok(Response::new(schema_result))
    }

    /// Not implemented for embedding-only service
    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("DoPut not supported for embedding service"))
    }

    /// List available embedding actions
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "similarity_search".to_string(),
                description: "Find similar embeddings".to_string(),
            },
            ActionType {
                r#type: "get_embedding".to_string(),
                description: "Get adapter embedding".to_string(),
            },
            ActionType {
                r#type: "list_embeddings".to_string(),
                description: "List all embeddings".to_string(),
            },
        ];

        let stream = stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    /// Execute embedding actions
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        
        match action.r#type.as_str() {
            "compact_storage" => {
                // Trigger storage compaction
                let compaction_stats = self.storage.compact().await
                    .map_err(|e| Status::internal(format!("Compaction failed: {}", e)))?;
                
                let result = arrow_flight::Result {
                    body: format!("Compacted {} adapters, reclaimed {} bytes", 
                           compaction_stats.adapters_compacted, 
                           compaction_stats.bytes_reclaimed).into_bytes().into(),
                };
                
                let stream = stream::iter(vec![Ok(result)]);
                Ok(Response::new(Box::pin(stream)))
            }
            
            "get_storage_stats" => {
                let stats = self.storage.get_storage_stats().await
                    .map_err(|e| Status::internal(format!("Failed to get stats: {}", e)))?;
                
                let result = arrow_flight::Result {
                    body: serde_json::to_vec(&stats).unwrap().into(),
                };
                
                let stream = stream::iter(vec![Ok(result)]);
                Ok(Response::new(Box::pin(stream)))
            }
            
            _ => Err(Status::unimplemented(format!("Action '{}' not supported", action.r#type)))
        }
    }

    /// Not implemented for embedding-only service
    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("DoExchange not supported for embedding service"))
    }

    /// Get flight info for embedding operations
    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        // Return basic flight info for embedding service  
        let descriptor = FlightDescriptor::new_path(vec!["embeddings".to_string()]);
        let mut flight_info = FlightInfo::new()
            .with_descriptor(descriptor)
            .with_total_records(-1)
            .with_total_bytes(-1);
            
        // Set schema using try_with_schema
        flight_info = flight_info.try_with_schema(&self.embedding_schema())
            .map_err(|e| Status::internal(format!("Failed to set schema: {}", e)))?;

        Ok(Response::new(flight_info))
    }

    /// Poll flight info (not implemented for embedding service)
    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<arrow_flight::PollInfo>, Status> {
        Err(Status::unimplemented("PollFlightInfo not supported for embedding service"))
    }
}

/// Create FlightSQL server for embedding service
pub fn create_embedding_flight_server(
    storage: Arc<dyn SparseStorage>,
) -> FlightServiceServer<EmbeddingFlightService> {
    let service = EmbeddingFlightService::new(storage);
    FlightServiceServer::new(service)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::vdb::{VDBSparseStorage, SparseStorageConfig};

    #[tokio::test]
    async fn test_embedding_service_creation() {
        let storage_config = SparseStorageConfig::default();
        let storage = Arc::new(VDBSparseStorage::new(storage_config).await.unwrap());
        
        let service = EmbeddingFlightService::new(storage);
        assert_eq!(service.config.max_embedding_size, 4096);
        assert_eq!(service.config.default_search_limit, 100);
    }
    
    #[test]
    fn test_embedding_query_parsing() {
        let query = EmbeddingQuery::SimilaritySearch {
            vector: vec![0.1, 0.2, 0.3],
            limit: 10,
            threshold: 0.5,
        };
        
        let json = serde_json::to_string(&query).unwrap();
        let parsed: EmbeddingQuery = serde_json::from_str(&json).unwrap();
        
        match parsed {
            EmbeddingQuery::SimilaritySearch { vector, limit, threshold } => {
                assert_eq!(vector, vec![0.1, 0.2, 0.3]);
                assert_eq!(limit, 10);
                assert_eq!(threshold, 0.5);
            }
            _ => panic!("Wrong query type parsed"),
        }
    }
    
    #[tokio::test]
    async fn test_embedding_schema() {
        let storage_config = SparseStorageConfig::default();
        let storage = Arc::new(VDBSparseStorage::new(storage_config).await.unwrap());
        let service = EmbeddingFlightService::new(storage);
        
        let schema = service.embedding_schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), "adapter_id");
        assert_eq!(schema.field(1).name(), "similarity_score");
        assert_eq!(schema.field(2).name(), "embedding_dimension");
        assert_eq!(schema.field(3).name(), "embedding_sample");
    }
}