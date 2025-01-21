use arrow::{
    array::{Array, Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use hyprstream_core::{
    storage::{
        duckdb::DuckDbBackend, StorageBackend, StorageBackendType,
        view::ViewDefinition,
    },
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use tonic::Status;

#[tokio::test]
async fn test_cache_operations() -> Result<(), Status> {
    // Create a temporary directory for the database
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    // Create backend (initialization happens synchronously in new())
    let backend = Arc::new(StorageBackendType::DuckDb(DuckDbBackend::new_with_options(
        db_path.to_str().unwrap(),
        &HashMap::new(),
        None,
    )?));

    // Initialize backend
    backend.init().await?;

    // Create test table
    let table_name = "test_metrics";
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]));

    backend.create_table(table_name, &schema).await?;

    // Create test data
    let values: Arc<dyn Array> = Arc::new(Float64Array::from(vec![1.0]));
    let timestamps: Arc<dyn Array> = Arc::new(Int64Array::from(vec![1000]));

    let batch = RecordBatch::try_new(schema.clone(), vec![values, timestamps]).unwrap();

    // Insert test data
    backend.insert_into_table(table_name, batch).await?;

    // Query data
    let result = backend.query_table(table_name, None).await?;
    assert_eq!(result.num_columns(), 2);

    // Create view
    let view_name = format!("view_{}", table_name);
    let view_def = ViewDefinition::new(
        table_name.to_string(),
        vec!["value".to_string(), "timestamp".to_string()],
        vec![],
        None,
        None,
        Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
            Field::new("timestamp", DataType::Int64, false),
        ])),
    );
    backend.as_ref().create_view(&view_name, view_def).await?;

    // Query view
    let result = backend.as_ref().get_view(&view_name).await?;
    assert_eq!(result.definition.schema.fields().len(), 2);

    Ok(())
}
