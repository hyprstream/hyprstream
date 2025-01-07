//! Vector-native model storage implementation.
//!
//! This module provides the core functionality for storing and managing foundational models
//! using Apache Arrow arrays. It enables:
//! - Zero-copy access to model weights
//! - Efficient GPU transfer
//! - Versioning and partial loading
//! - Memory-mapped storage
//!
//! The implementation focuses on high performance and efficient memory usage while
//! maintaining compatibility with various model architectures.

use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;
use serde;

/// Represents a layer in a neural network model
#[derive(Debug, Clone)]
pub struct ModelLayer {
    /// Name of the layer
    pub name: String,
    /// Type of the layer (e.g., "linear", "conv2d")
    pub layer_type: String,
    /// Shape of the layer's weights
    pub shape: Vec<usize>,
    /// Layer weights as Arrow arrays
    pub weights: Vec<ArrayRef>,
    /// Layer-specific parameters
    pub parameters: HashMap<String, String>,
}

/// Model version information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelVersion {
    /// Version identifier
    pub version: String,
    /// Timestamp of creation
    pub created_at: i64,
    /// Description of changes
    pub description: String,
    /// Parent version (if any)
    pub parent_version: Option<String>,
}

/// Complete model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub model_id: String,
    /// Model name
    pub name: String,
    /// Model architecture type
    pub architecture: String,
    /// Current version
    pub version: ModelVersion,
    /// Model-specific parameters
    pub parameters: HashMap<String, String>,
}

/// Represents a complete foundational model
#[derive(Debug)]
pub struct Model {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model layers
    pub layers: Vec<ModelLayer>,
}

/// Trait for model storage operations
#[async_trait::async_trait]
pub trait ModelStorage: Send + Sync {
    /// Initialize the storage system
    async fn init(&self) -> Result<(), Status>;

    /// Store a model
    async fn store_model(&self, model: &Model) -> Result<(), Status>;

    /// Load a complete model
    async fn load_model(&self, model_id: &str, version: Option<&str>) -> Result<Model, Status>;

    /// Load specific layers of a model
    async fn load_layers(
        &self,
        model_id: &str,
        layer_names: &[String],
        version: Option<&str>,
    ) -> Result<Vec<ModelLayer>, Status>;

    /// List available models
    async fn list_models(&self) -> Result<Vec<ModelMetadata>, Status>;

    /// List versions of a model
    async fn list_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>, Status>;

    /// Delete a model version
    async fn delete_version(&self, model_id: &str, version: &str) -> Result<(), Status>;
}

impl ModelLayer {
    /// Creates a new model layer
    pub fn new(
        name: String,
        layer_type: String,
        shape: Vec<usize>,
        weights: Vec<ArrayRef>,
        parameters: HashMap<String, String>,
    ) -> Self {
        Self {
            name,
            layer_type,
            shape,
            weights,
            parameters,
        }
    }

    /// Converts layer weights to Arrow record batch
    pub fn to_record_batch(&self) -> Result<RecordBatch, Status> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("layer_type", DataType::Utf8, false),
            Field::new("shape", DataType::Binary, false),
            Field::new("weights", DataType::Binary, false),
            Field::new("parameters", DataType::Binary, false),
        ]));

        // Serialize shape and parameters
        let shape_bytes = bincode::serialize(&self.shape)
            .map_err(|e| Status::internal(format!("Failed to serialize shape: {}", e)))?;
        let params_bytes = bincode::serialize(&self.parameters)
            .map_err(|e| Status::internal(format!("Failed to serialize parameters: {}", e)))?;

        // Serialize weights to bytes
        let mut weights_bytes = Vec::new();
        for weight in &self.weights {
            if let Some(float_array) = weight.as_any().downcast_ref::<Float32Array>() {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        float_array.values().as_ptr() as *const u8,
                        float_array.values().len() * 4,
                    )
                };
                weights_bytes.extend_from_slice(bytes);
            } else {
                return Err(Status::internal("Unsupported weight array type"));
            }
        }

        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::StringArray::from(vec![self.name.as_str()])),
                Arc::new(arrow_array::StringArray::from(vec![self.layer_type.as_str()])),
                Arc::new(arrow_array::BinaryArray::from(vec![shape_bytes.as_slice()])),
                Arc::new(arrow_array::BinaryArray::from(vec![weights_bytes.as_slice()])),
                Arc::new(arrow_array::BinaryArray::from(vec![params_bytes.as_slice()])),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create record batch: {}", e)))?)
    }

    /// Creates a layer from Arrow record batch
    pub fn from_record_batch(batch: &RecordBatch) -> Result<Self, Status> {
        let name = batch
            .column_by_name("name")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid name column"))?
            .value(0)
            .to_string();

        let layer_type = batch
            .column_by_name("layer_type")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid layer_type column"))?
            .value(0)
            .to_string();

        let shape_bytes = batch
            .column_by_name("shape")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid shape column"))?
            .value(0);

        let shape: Vec<usize> = bincode::deserialize(shape_bytes)
            .map_err(|e| Status::internal(format!("Failed to deserialize shape: {}", e)))?;

        let weights_bytes = batch
            .column_by_name("weights")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid weights column"))?
            .value(0);

        let params_bytes = batch
            .column_by_name("parameters")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid parameters column"))?
            .value(0);

        let parameters: HashMap<String, String> = bincode::deserialize(params_bytes)
            .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

        // Convert weights bytes to Arrow arrays
        let weights_slice = unsafe { std::slice::from_raw_parts(
            weights_bytes.as_ptr() as *const f32,
            weights_bytes.len() / 4,
        )};
        let weights = vec![Arc::new(Float32Array::from(weights_slice.to_vec())) as ArrayRef];

        Ok(Self::new(name, layer_type, shape, weights, parameters))
    }
}

impl Model {
    /// Creates a new model
    pub fn new(metadata: ModelMetadata, layers: Vec<ModelLayer>) -> Self {
        Self { metadata, layers }
    }

    /// Converts model to Arrow record batches
    pub fn to_record_batches(&self) -> Result<Vec<RecordBatch>, Status> {
        let mut batches = Vec::new();

        // Add metadata batch
        let metadata_schema = Arc::new(Schema::new(vec![
            Field::new("model_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("architecture", DataType::Utf8, false),
            Field::new("version", DataType::Binary, false),
            Field::new("parameters", DataType::Binary, false),
        ]));

        let version_bytes = bincode::serialize(&self.metadata.version)
            .map_err(|e| Status::internal(format!("Failed to serialize version: {}", e)))?;
        let params_bytes = bincode::serialize(&self.metadata.parameters)
            .map_err(|e| Status::internal(format!("Failed to serialize parameters: {}", e)))?;

        batches.push(RecordBatch::try_new(
            metadata_schema,
            vec![
                Arc::new(arrow_array::StringArray::from(vec![self.metadata.model_id.as_str()])),
                Arc::new(arrow_array::StringArray::from(vec![self.metadata.name.as_str()])),
                Arc::new(arrow_array::StringArray::from(vec![self.metadata.architecture.as_str()])),
                Arc::new(arrow_array::BinaryArray::from(vec![version_bytes.as_slice()])),
                Arc::new(arrow_array::BinaryArray::from(vec![params_bytes.as_slice()])),
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to create metadata batch: {}", e)))?);

        // Add layer batches
        for layer in &self.layers {
            batches.push(layer.to_record_batch()?);
        }

        Ok(batches)
    }

    /// Creates a model from Arrow record batches
    pub fn from_record_batches(batches: &[RecordBatch]) -> Result<Self, Status> {
        if batches.is_empty() {
            return Err(Status::internal("No record batches provided"));
        }

        // Parse metadata from first batch
        let metadata_batch = &batches[0];
        let model_id = metadata_batch
            .column_by_name("model_id")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid model_id column"))?
            .value(0)
            .to_string();

        let name = metadata_batch
            .column_by_name("name")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid name column"))?
            .value(0)
            .to_string();

        let architecture = metadata_batch
            .column_by_name("architecture")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::StringArray>())
            .ok_or_else(|| Status::internal("Invalid architecture column"))?
            .value(0)
            .to_string();

        let version_bytes = metadata_batch
            .column_by_name("version")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid version column"))?
            .value(0);

        let version: ModelVersion = bincode::deserialize(version_bytes)
            .map_err(|e| Status::internal(format!("Failed to deserialize version: {}", e)))?;

        let params_bytes = metadata_batch
            .column_by_name("parameters")
            .and_then(|col| col.as_any().downcast_ref::<arrow_array::BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid parameters column"))?
            .value(0);

        let parameters: HashMap<String, String> = bincode::deserialize(params_bytes)
            .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

        let metadata = ModelMetadata {
            model_id,
            name,
            architecture,
            version,
            parameters,
        };

        // Parse layers from remaining batches
        let mut layers = Vec::new();
        for batch in &batches[1..] {
            layers.push(ModelLayer::from_record_batch(batch)?);
        }

        Ok(Self::new(metadata, layers))
    }
} 