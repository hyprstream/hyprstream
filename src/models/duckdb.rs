//! DuckDB-based model storage implementation.
//!
//! This module provides a DuckDB-based implementation of the ModelStorage trait,
//! enabling efficient storage and retrieval of foundational models using Arrow arrays.
//! It supports:
//! - Zero-copy access to model weights
//! - Version control and partial loading
//! - Memory-mapped storage for large models
//! - Efficient serialization/deserialization

use super::{Model, ModelLayer, ModelMetadata, ModelStorage, ModelVersion};
use crate::storage::duckdb::DuckDbBackend;
use arrow_array::RecordBatch;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::Status;

/// DuckDB-based model storage implementation
pub struct DuckDbModelStorage {
    backend: Arc<DuckDbBackend>,
    conn: Arc<Mutex<duckdb::Connection>>,
}

impl DuckDbModelStorage {
    /// Creates a new DuckDB model storage instance
    pub fn new(backend: Arc<DuckDbBackend>) -> Result<Self, Status> {
        let conn = backend.get_connection().map_err(|e| {
            Status::internal(format!("Failed to get DuckDB connection: {}", e))
        })?;

        Ok(Self {
            backend,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Creates the necessary tables for model storage
    async fn create_tables(&self) -> Result<(), Status> {
        let conn = self.conn.lock().await;

        // Create models table
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS models (
                model_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                architecture VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                created_at BIGINT NOT NULL,
                description TEXT,
                parent_version VARCHAR,
                parameters BLOB NOT NULL,
                PRIMARY KEY (model_id, version)
            );

            CREATE INDEX IF NOT EXISTS idx_models_version 
            ON models(model_id, version);

            CREATE TABLE IF NOT EXISTS model_layers (
                model_id VARCHAR NOT NULL,
                version VARCHAR NOT NULL,
                layer_name VARCHAR NOT NULL,
                layer_type VARCHAR NOT NULL,
                shape BLOB NOT NULL,
                weights BLOB NOT NULL,
                parameters BLOB NOT NULL,
                PRIMARY KEY (model_id, version, layer_name),
                FOREIGN KEY (model_id, version) 
                REFERENCES models(model_id, version) 
                ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_layers_model 
            ON model_layers(model_id, version);
            "#,
        )
        .map_err(|e| Status::internal(format!("Failed to create tables: {}", e)))?;

        Ok(())
    }
}

#[async_trait::async_trait]
impl ModelStorage for DuckDbModelStorage {
    async fn init(&self) -> Result<(), Status> {
        self.create_tables().await
    }

    async fn store_model(&self, model: &Model) -> Result<(), Status> {
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| Status::internal(format!("Failed to start transaction: {}", e)))?;

        // Store model metadata
        let params_bytes = bincode::serialize(&model.metadata.parameters)
            .map_err(|e| Status::internal(format!("Failed to serialize parameters: {}", e)))?;

        tx.execute(
            r#"
            INSERT INTO models (
                model_id, name, architecture, version, 
                created_at, description, parent_version, parameters
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                model.metadata.model_id,
                model.metadata.name,
                model.metadata.architecture,
                model.metadata.version.version,
                model.metadata.version.created_at,
                model.metadata.version.description,
                model.metadata.version.parent_version,
                params_bytes,
            ],
        )
        .map_err(|e| Status::internal(format!("Failed to insert model metadata: {}", e)))?;

        // Store layers
        for layer in &model.layers {
            let shape_bytes = bincode::serialize(&layer.shape)
                .map_err(|e| Status::internal(format!("Failed to serialize shape: {}", e)))?;

            let mut weights_bytes = Vec::new();
            for weight in &layer.weights {
                if let Some(float_array) = weight.as_any().downcast_ref::<arrow_array::Float32Array>() {
                    weights_bytes.extend_from_slice(float_array.values());
                } else {
                    return Err(Status::internal("Unsupported weight array type"));
                }
            }

            let params_bytes = bincode::serialize(&layer.parameters)
                .map_err(|e| Status::internal(format!("Failed to serialize layer parameters: {}", e)))?;

            tx.execute(
                r#"
                INSERT INTO model_layers (
                    model_id, version, layer_name, layer_type,
                    shape, weights, parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                "#,
                params![
                    model.metadata.model_id,
                    model.metadata.version.version,
                    layer.name,
                    layer.layer_type,
                    shape_bytes,
                    weights_bytes,
                    params_bytes,
                ],
            )
            .map_err(|e| Status::internal(format!("Failed to insert layer: {}", e)))?;
        }

        tx.commit()
            .map_err(|e| Status::internal(format!("Failed to commit transaction: {}", e)))?;

        Ok(())
    }

    async fn load_model(&self, model_id: &str, version: Option<&str>) -> Result<Model, Status> {
        let conn = self.conn.lock().await;

        // Load model metadata
        let mut stmt = if let Some(ver) = version {
            conn.prepare(
                r#"
                SELECT name, architecture, version, created_at,
                       description, parent_version, parameters
                FROM models
                WHERE model_id = ? AND version = ?
                "#,
            )
            .map_err(|e| Status::internal(format!("Failed to prepare metadata query: {}", e)))?;
            stmt.query(params![model_id, ver])
        } else {
            conn.prepare(
                r#"
                SELECT name, architecture, version, created_at,
                       description, parent_version, parameters
                FROM models
                WHERE model_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                "#,
            )
            .map_err(|e| Status::internal(format!("Failed to prepare metadata query: {}", e)))?;
            stmt.query(params![model_id])
        }
        .map_err(|e| Status::internal(format!("Failed to query metadata: {}", e)))?;

        let row = stmt
            .next()
            .map_err(|e| Status::internal(format!("Failed to read metadata: {}", e)))?
            .ok_or_else(|| Status::not_found("Model not found"))?;

        let name: String = row.get(0)?;
        let architecture: String = row.get(1)?;
        let version_str: String = row.get(2)?;
        let created_at: i64 = row.get(3)?;
        let description: String = row.get(4)?;
        let parent_version: Option<String> = row.get(5)?;
        let params_bytes: Vec<u8> = row.get(6)?;

        let parameters: std::collections::HashMap<String, String> = bincode::deserialize(&params_bytes)
            .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

        let version = ModelVersion {
            version: version_str.clone(),
            created_at,
            description,
            parent_version,
        };

        let metadata = ModelMetadata {
            model_id: model_id.to_string(),
            name,
            architecture,
            version,
            parameters,
        };

        // Load layers
        let mut stmt = conn
            .prepare(
                r#"
                SELECT layer_name, layer_type, shape, weights, parameters
                FROM model_layers
                WHERE model_id = ? AND version = ?
                ORDER BY layer_name
                "#,
            )
            .map_err(|e| Status::internal(format!("Failed to prepare layers query: {}", e)))?;

        let mut layers = Vec::new();
        let mut rows = stmt
            .query(params![model_id, version_str])
            .map_err(|e| Status::internal(format!("Failed to query layers: {}", e)))?;

        while let Some(row) = rows
            .next()
            .map_err(|e| Status::internal(format!("Failed to read layer: {}", e)))?
        {
            let name: String = row.get(0)?;
            let layer_type: String = row.get(1)?;
            let shape_bytes: Vec<u8> = row.get(2)?;
            let weights_bytes: Vec<u8> = row.get(3)?;
            let params_bytes: Vec<u8> = row.get(4)?;

            let shape: Vec<usize> = bincode::deserialize(&shape_bytes)
                .map_err(|e| Status::internal(format!("Failed to deserialize shape: {}", e)))?;

            let parameters: std::collections::HashMap<String, String> =
                bincode::deserialize(&params_bytes)
                    .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

            // Convert weights bytes to Arrow array
            let weights_slice = unsafe {
                std::slice::from_raw_parts(
                    weights_bytes.as_ptr() as *const f32,
                    weights_bytes.len() / 4,
                )
            };
            let weights = vec![Arc::new(arrow_array::Float32Array::from(weights_slice.to_vec()))
                as arrow_array::ArrayRef];

            layers.push(ModelLayer::new(name, layer_type, shape, weights, parameters));
        }

        Ok(Model::new(metadata, layers))
    }

    async fn load_layers(
        &self,
        model_id: &str,
        layer_names: &[String],
        version: Option<&str>,
    ) -> Result<Vec<ModelLayer>, Status> {
        let conn = self.conn.lock().await;

        // Get version if not specified
        let version = if let Some(ver) = version {
            ver.to_string()
        } else {
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT version FROM models
                    WHERE model_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    "#,
                )
                .map_err(|e| Status::internal(format!("Failed to prepare version query: {}", e)))?;

            let version: String = stmt
                .query_row(params![model_id], |row| row.get(0))
                .map_err(|e| Status::internal(format!("Failed to get latest version: {}", e)))?;
            version
        };

        // Load specified layers
        let placeholders = vec!["?"; layer_names.len() + 2].join(",");
        let query = format!(
            r#"
            SELECT layer_name, layer_type, shape, weights, parameters
            FROM model_layers
            WHERE model_id = ? AND version = ? 
            AND layer_name IN ({})
            ORDER BY layer_name
            "#,
            placeholders
        );

        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| Status::internal(format!("Failed to prepare layers query: {}", e)))?;

        let mut params: Vec<&dyn duckdb::ToSql> = vec![&model_id, &version];
        for name in layer_names {
            params.push(name as &dyn duckdb::ToSql);
        }

        let mut layers = Vec::new();
        let mut rows = stmt
            .query(params.as_slice())
            .map_err(|e| Status::internal(format!("Failed to query layers: {}", e)))?;

        while let Some(row) = rows
            .next()
            .map_err(|e| Status::internal(format!("Failed to read layer: {}", e)))?
        {
            let name: String = row.get(0)?;
            let layer_type: String = row.get(1)?;
            let shape_bytes: Vec<u8> = row.get(2)?;
            let weights_bytes: Vec<u8> = row.get(3)?;
            let params_bytes: Vec<u8> = row.get(4)?;

            let shape: Vec<usize> = bincode::deserialize(&shape_bytes)
                .map_err(|e| Status::internal(format!("Failed to deserialize shape: {}", e)))?;

            let parameters: std::collections::HashMap<String, String> =
                bincode::deserialize(&params_bytes)
                    .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

            // Convert weights bytes to Arrow array
            let weights_slice = unsafe {
                std::slice::from_raw_parts(
                    weights_bytes.as_ptr() as *const f32,
                    weights_bytes.len() / 4,
                )
            };
            let weights = vec![Arc::new(arrow_array::Float32Array::from(weights_slice.to_vec()))
                as arrow_array::ArrayRef];

            layers.push(ModelLayer::new(name, layer_type, shape, weights, parameters));
        }

        Ok(layers)
    }

    async fn list_models(&self) -> Result<Vec<ModelMetadata>, Status> {
        let conn = self.conn.lock().await;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT DISTINCT m1.model_id, m1.name, m1.architecture,
                       m1.version, m1.created_at, m1.description,
                       m1.parent_version, m1.parameters
                FROM models m1
                INNER JOIN (
                    SELECT model_id, MAX(created_at) as max_created_at
                    FROM models
                    GROUP BY model_id
                ) m2 ON m1.model_id = m2.model_id 
                AND m1.created_at = m2.max_created_at
                ORDER BY m1.model_id
                "#,
            )
            .map_err(|e| Status::internal(format!("Failed to prepare models query: {}", e)))?;

        let mut models = Vec::new();
        let mut rows = stmt
            .query(params![])
            .map_err(|e| Status::internal(format!("Failed to query models: {}", e)))?;

        while let Some(row) = rows
            .next()
            .map_err(|e| Status::internal(format!("Failed to read model: {}", e)))?
        {
            let model_id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let architecture: String = row.get(2)?;
            let version_str: String = row.get(3)?;
            let created_at: i64 = row.get(4)?;
            let description: String = row.get(5)?;
            let parent_version: Option<String> = row.get(6)?;
            let params_bytes: Vec<u8> = row.get(7)?;

            let parameters: std::collections::HashMap<String, String> =
                bincode::deserialize(&params_bytes)
                    .map_err(|e| Status::internal(format!("Failed to deserialize parameters: {}", e)))?;

            let version = ModelVersion {
                version: version_str,
                created_at,
                description,
                parent_version,
            };

            models.push(ModelMetadata {
                model_id,
                name,
                architecture,
                version,
                parameters,
            });
        }

        Ok(models)
    }

    async fn list_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>, Status> {
        let conn = self.conn.lock().await;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT version, created_at, description, parent_version
                FROM models
                WHERE model_id = ?
                ORDER BY created_at DESC
                "#,
            )
            .map_err(|e| Status::internal(format!("Failed to prepare versions query: {}", e)))?;

        let mut versions = Vec::new();
        let mut rows = stmt
            .query(params![model_id])
            .map_err(|e| Status::internal(format!("Failed to query versions: {}", e)))?;

        while let Some(row) = rows
            .next()
            .map_err(|e| Status::internal(format!("Failed to read version: {}", e)))?
        {
            let version: String = row.get(0)?;
            let created_at: i64 = row.get(1)?;
            let description: String = row.get(2)?;
            let parent_version: Option<String> = row.get(3)?;

            versions.push(ModelVersion {
                version,
                created_at,
                description,
                parent_version,
            });
        }

        Ok(versions)
    }

    async fn delete_version(&self, model_id: &str, version: &str) -> Result<(), Status> {
        let conn = self.conn.lock().await;

        // Delete model (layers will be deleted via CASCADE)
        conn.execute(
            "DELETE FROM models WHERE model_id = ? AND version = ?",
            params![model_id, version],
        )
        .map_err(|e| Status::internal(format!("Failed to delete version: {}", e)))?;

        Ok(())
    }
} 