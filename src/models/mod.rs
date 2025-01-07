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

use arrow_array::{Array, ArrayRef, Float32Array, StringArray, Int64Array, BinaryArray, RecordBatch};
use arrow_schema::{Schema, Field, DataType};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;
use serde::{Serialize, Deserialize};
use bincode;

pub mod storage;

/// Represents a layer in a neural network model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLayer {
    /// Name of the layer
    pub name: String,
    /// Type of the layer (e.g., "linear", "conv2d")
    pub layer_type: String,
    /// Shape of the layer's weights
    pub shape: Vec<usize>,
    /// Layer weights as Arrow arrays
    #[serde(skip)]
    pub weights: Vec<ArrayRef>,
    /// Layer-specific parameters
    pub parameters: HashMap<String, String>,
}

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Serialize, Deserialize)]
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

    /// Validates layer weights against shape
    pub fn validate(&self) -> Result<(), Status> {
        // Validate that weights match the declared shape
        let total_elements: usize = self.shape.iter().product();
        for weight in &self.weights {
            if let Some(float_array) = weight.as_any().downcast_ref::<Float32Array>() {
                if float_array.len() != total_elements {
                    return Err(Status::invalid_argument(format!(
                        "Weight array length {} does not match shape product {}",
                        float_array.len(),
                        total_elements
                    )));
                }
            } else {
                return Err(Status::invalid_argument("Unsupported weight array type"));
            }
        }
        Ok(())
    }
}

impl Model {
    /// Creates a new model
    pub fn new(metadata: ModelMetadata, layers: Vec<ModelLayer>) -> Self {
        Self { metadata, layers }
    }

    /// Validates the model structure
    pub fn validate(&self) -> Result<(), Status> {
        // Validate each layer
        for layer in &self.layers {
            layer.validate()?;
        }
        Ok(())
    }

    /// Converts the model into Arrow RecordBatches
    pub fn to_record_batches(&self) -> Result<Vec<RecordBatch>, Status> {
        let mut batches = Vec::new();

        // Create metadata batch
        let metadata_schema = Arc::new(Schema::new(vec![
            Field::new("model_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("architecture", DataType::Utf8, false),
            Field::new("version", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("description", DataType::Utf8, false),
            Field::new("parent_version", DataType::Utf8, true),
        ]));

        // Fix array construction for optional parent_version
        let parent_version = match &self.metadata.version.parent_version {
            Some(v) => Some(v.as_str()),
            None => None,
        };

        let metadata_arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from_iter_values([&self.metadata.model_id])),
            Arc::new(StringArray::from_iter_values([&self.metadata.name])),
            Arc::new(StringArray::from_iter_values([&self.metadata.architecture])),
            Arc::new(StringArray::from_iter_values([&self.metadata.version.version])),
            Arc::new(Int64Array::from_iter_values([self.metadata.version.created_at])),
            Arc::new(StringArray::from_iter_values([&self.metadata.version.description])),
            Arc::new(StringArray::from_iter(std::iter::once(parent_version))),
        ];

        let metadata_batch = RecordBatch::try_new(metadata_schema.clone(), metadata_arrays)
            .map_err(|e| Status::internal(format!("Failed to create metadata batch: {}", e)))?;
        batches.push(metadata_batch);

        // Create layer batches
        let layer_schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("layer_type", DataType::Utf8, false),
            Field::new("shape", DataType::Binary, false),
            Field::new("weights", DataType::Binary, false),
            Field::new("parameters", DataType::Binary, false),
        ]));

        for layer in &self.layers {
            // Serialize shape
            let shape_bytes = bincode::serialize(&layer.shape)
                .map_err(|e| Status::internal(format!("Failed to serialize shape: {}", e)))?;

            // Serialize parameters
            let params_bytes = bincode::serialize(&layer.parameters)
                .map_err(|e| Status::internal(format!("Failed to serialize parameters: {}", e)))?;

            // Serialize weights
            let mut weights_bytes = Vec::new();
            for weight in &layer.weights {
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

            let layer_arrays: Vec<ArrayRef> = vec![
                Arc::new(StringArray::from_iter_values([&layer.name])),
                Arc::new(StringArray::from_iter_values([&layer.layer_type])),
                Arc::new(BinaryArray::from_iter_values([shape_bytes.as_slice()])),
                Arc::new(BinaryArray::from_iter_values([weights_bytes.as_slice()])),
                Arc::new(BinaryArray::from_iter_values([params_bytes.as_slice()])),
            ];

            let layer_batch = RecordBatch::try_new(layer_schema.clone(), layer_arrays)
                .map_err(|e| Status::internal(format!("Failed to create layer batch: {}", e)))?;
            batches.push(layer_batch);
        }

        Ok(batches)
    }

    pub fn estimated_size(&self) -> u64 {
        // Base size for metadata
        let mut size = std::mem::size_of::<Self>() as u64;
        
        // Add size of layers
        for layer in &self.layers {
            // Layer metadata
            size += std::mem::size_of::<ModelLayer>() as u64;
            
            // Layer weights
            size += layer.weights.iter()
                .map(|arr| arr.len() * std::mem::size_of::<f32>())
                .sum::<usize>() as u64;
            
            // Layer parameters
            size += layer.parameters.iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>() as u64;
        }
        
        size
    }
} 