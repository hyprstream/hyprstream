use super::{Model, ModelLayer, ModelMetadata, ModelVersion, ModelStorage};
use crate::storage::{
    HyprStorageBackend, HyprStorageBackendType
};
use arrow_array::{
    Array, ArrayRef, RecordBatch, StringArray, Int64Array, BinaryArray,
};
use arrow_schema::{Schema, Field, DataType};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;
use async_trait::async_trait;
use bincode;

/// Get schema for model metadata table
fn get_model_metadata_schema() -> Schema {
    Schema::new(vec![
        Field::new("model_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("architecture", DataType::Utf8, false),
        Field::new("version", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("parent_version", DataType::Utf8, true),
        Field::new("parameters", DataType::Binary, false),
    ])
}

/// Get schema for model layer table
fn get_model_layer_schema() -> Schema {
    Schema::new(vec![
        Field::new("model_id", DataType::Utf8, false),
        Field::new("version", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("layer_type", DataType::Utf8, false),
        Field::new("shape", DataType::Binary, false),
        Field::new("weights", DataType::Binary, false),
        Field::new("parameters", DataType::Binary, false),
    ])
}

pub struct TimeSeriesModelStorage {
    backend: Arc<HyprStorageBackendType>,
}

impl TimeSeriesModelStorage {
    pub fn new(backend: Arc<HyprStorageBackendType>) -> Self {
        Self { backend }
    }

    async fn init_tables(&self) -> Result<(), Status> {
        // Create tables for model storage
        self.backend.as_ref().as_ref().create_table("model_metadata", &get_model_metadata_schema()).await?;
        self.backend.as_ref().as_ref().create_table("model_layers", &get_model_layer_schema()).await?;
        Ok(())
    }

    /// Extract an optional string value from a StringArray column
    /// Maintains zero-copy until the final conversion to owned String
    fn get_optional_string(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<String>, Status> {
        let value = batch.column_by_name(column)
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal(format!("Invalid {} column", column)))?
            .value(row);

        if value.is_empty() {
            Ok(None)
        } else {
            Ok(Some(value.to_string()))
        }
    }

    /// Extract a required string value from a StringArray column
    fn get_required_string(batch: &RecordBatch, column: &str, row: usize) -> Result<String, Status> {
        Ok(batch.column_by_name(column)
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal(format!("Invalid {} column", column)))?
            .value(row)
            .to_string())
    }
}

#[async_trait]
impl ModelStorage for TimeSeriesModelStorage {
    async fn init(&self) -> Result<(), Status> {
        self.init_tables().await
    }

    async fn store_model(&self, model: &Model) -> Result<(), Status> {
        // Convert model to record batches
        let batches = model.to_record_batches()?;
        
        // Store metadata and layers
        for batch in batches {
            self.backend.insert_into_table("model_layers", batch).await?;
        }
        
        Ok(())
    }

    async fn load_model(&self, model_id: &str, version: Option<&str>) -> Result<Model, Status> {
        // Build query conditions
        let version_condition = version.map_or(String::new(), |v| format!(" AND version = '{}'", v));
        
        // Query metadata
        let metadata_batch = self.backend.query_table(
            "model_metadata",
            Some(vec![format!("* WHERE model_id = '{}'{}", model_id, version_condition)]),
        ).await?;

        // Query layers
        let layers_batch = self.backend.query_table(
            "model_layers",
            Some(vec![format!("* WHERE model_id = '{}'{}", model_id, version_condition)]),
        ).await?;

        // Convert record batches back to Model
        Self::record_batches_to_model(metadata_batch, layers_batch)
    }

    async fn load_layers(
        &self,
        model_id: &str,
        layer_names: &[String],
        version: Option<&str>,
    ) -> Result<Vec<ModelLayer>, Status> {
        // Build query conditions
        let version_condition = version.map_or(String::new(), |v| format!(" AND version = '{}'", v));
        let names_condition = if !layer_names.is_empty() {
            format!(" AND name IN ({})", layer_names.iter()
                .map(|n| format!("'{}'", n))
                .collect::<Vec<_>>()
                .join(","))
        } else {
            String::new()
        };

        // Query specific layers
        let layers_batch = self.backend.query_table(
            "model_layers",
            Some(vec![format!("* WHERE model_id = '{}'{}{}", 
                model_id, version_condition, names_condition)]),
        ).await?;

        // Convert record batch to ModelLayer instances
        Self::record_batch_to_layers(layers_batch)
    }

    async fn list_models(&self) -> Result<Vec<ModelMetadata>, Status> {
        // Query distinct models from metadata
        let metadata_batch = self.backend.query_table(
            "model_metadata",
            Some(vec!["DISTINCT model_id, name, architecture, version, created_at, description, parent_version, parameters".to_string()]),
        ).await?;

        // Convert record batch to ModelMetadata instances
        Self::record_batch_to_metadata_list(metadata_batch)
    }

    async fn list_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>, Status> {
        // Query versions for specific model
        let versions_batch = self.backend.query_table(
            "model_metadata",
            Some(vec![format!("version, created_at, description, parent_version WHERE model_id = '{}'", model_id)]),
        ).await?;

        // Convert record batch to ModelVersion instances
        Self::record_batch_to_versions(versions_batch)
    }

    async fn delete_version(&self, model_id: &str, version: &str) -> Result<(), Status> {
        // Delete metadata
        self.backend.query_table(
            "model_metadata",
            Some(vec![format!("DELETE WHERE model_id = '{}' AND version = '{}'", model_id, version)]),
        ).await?;

        // Delete layers
        self.backend.query_table(
            "model_layers",
            Some(vec![format!("DELETE WHERE model_id = '{}' AND version = '{}'", model_id, version)]),
        ).await?;

        Ok(())
    }
}

// Helper methods for TimeSeriesModelStorage
impl TimeSeriesModelStorage {
    fn record_batches_to_model(metadata_batch: RecordBatch, layers_batch: RecordBatch) -> Result<Model, Status> {
        // First convert metadata
        let metadata = Self::record_batch_to_metadata(&metadata_batch)?;
        
        // Then convert layers
        let layers = Self::record_batch_to_layers(layers_batch)?;

        Ok(Model::new(metadata, layers))
    }

    fn record_batch_to_metadata(batch: &RecordBatch) -> Result<ModelMetadata, Status> {
        if batch.num_rows() == 0 {
            return Err(Status::not_found("Model metadata not found"));
        }
        Self::record_batch_to_metadata_row(batch, 0)
    }

    fn record_batch_to_layers(batch: RecordBatch) -> Result<Vec<ModelLayer>, Status> {
        let mut layers = Vec::with_capacity(batch.num_rows());

        let names = batch.column_by_name("name")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("Invalid name column"))?;

        let layer_types = batch.column_by_name("layer_type")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Status::internal("Invalid layer_type column"))?;

        let shapes = batch.column_by_name("shape")
            .and_then(|col| col.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid shape column"))?;

        let weights = batch.column_by_name("weights")
            .and_then(|col| col.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid weights column"))?;

        let params = batch.column_by_name("parameters")
            .and_then(|col| col.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid parameters column"))?;

        for row_idx in 0..batch.num_rows() {
            // Use references instead of cloning strings where possible
            let name = names.value(row_idx);
            let layer_type = layer_types.value(row_idx);

            let shape: Vec<usize> = bincode::deserialize(shapes.value(row_idx))
                .map_err(|e| Status::internal(format!("Failed to deserialize shape: {}", e)))?;

            let parameters: HashMap<String, String> = bincode::deserialize(params.value(row_idx))
                .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

            // Get weight data with zero copy
            let weight_data = weights.value(row_idx);
            
            // Create Float32Array directly from the raw bytes
            // This avoids an extra allocation by using the existing memory
            let float_data = unsafe {
                std::slice::from_raw_parts(
                    weight_data.as_ptr() as *const f32,
                    weight_data.len() / std::mem::size_of::<f32>()
                )
            };
            
            // Create Float32Array using arrow's builder for better performance
            let mut builder = arrow::array::Float32Builder::with_capacity(float_data.len());
            builder.append_slice(float_data);
            let weights = vec![Arc::new(builder.finish()) as ArrayRef];

            layers.push(ModelLayer {
                name: name.to_string(),
                layer_type: layer_type.to_string(),
                shape,
                weights,
                parameters,
            });
        }

        Ok(layers)
    }

    fn record_batch_to_metadata_list(batch: RecordBatch) -> Result<Vec<ModelMetadata>, Status> {
        let mut metadata_list = Vec::with_capacity(batch.num_rows());
        
        for row_idx in 0..batch.num_rows() {
            let metadata = Self::record_batch_to_metadata_row(&batch, row_idx)?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    fn record_batch_to_metadata_row(batch: &RecordBatch, row_idx: usize) -> Result<ModelMetadata, Status> {
        let model_id = Self::get_required_string(batch, "model_id", row_idx)?;
        let name = Self::get_required_string(batch, "name", row_idx)?;
        let architecture = Self::get_required_string(batch, "architecture", row_idx)?;
        let version_str = Self::get_required_string(batch, "version", row_idx)?;
        let description = Self::get_required_string(batch, "description", row_idx)?;
        let parent_version = Self::get_optional_string(batch, "parent_version", row_idx)?;

        let created_at = batch.column_by_name("created_at")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| Status::internal("Invalid created_at column"))?
            .value(row_idx);

        let params_bytes = batch.column_by_name("parameters")
            .and_then(|col| col.as_any().downcast_ref::<BinaryArray>())
            .ok_or_else(|| Status::internal("Invalid parameters column"))?
            .value(row_idx);

        let parameters: HashMap<String, String> = bincode::deserialize(params_bytes)
            .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

        Ok(ModelMetadata {
            model_id,
            name,
            architecture,
            version: ModelVersion {
                version: version_str,
                created_at,
                description,
                parent_version,
            },
            parameters,
        })
    }

    fn record_batch_to_versions(batch: RecordBatch) -> Result<Vec<ModelVersion>, Status> {
        let mut versions = Vec::with_capacity(batch.num_rows());

        for row_idx in 0..batch.num_rows() {
            let version = ModelVersion {
                version: Self::get_required_string(&batch, "version", row_idx)?,
                created_at: batch.column_by_name("created_at")
                    .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
                    .ok_or_else(|| Status::internal("Invalid created_at column"))?
                    .value(row_idx),
                description: Self::get_required_string(&batch, "description", row_idx)?,
                parent_version: Self::get_optional_string(&batch, "parent_version", row_idx)?,
            };
            versions.push(version);
        }

        Ok(versions)
    }
} 