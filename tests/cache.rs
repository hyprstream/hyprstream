use arrow::{
    array::{Array, Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use futures::StreamExt;
use hyprstream_core::{
    aggregation::{AggregateFunction, GroupBy, TimeWindow},
    storage::{
        duckdb::DuckDbBackend,
        StorageBackend,
        StorageBackendType,
        table_manager::AggregationView,
    },
};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tempfile::tempdir;
use tonic::Status;

#[tokio::test]
async fn test_cache_operations() -> Result<(), Status> {
    // Create a temporary directory for the database
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    // Create backend
    let backend = Arc::new(StorageBackendType::DuckDb(
        DuckDbBackend::new_with_options(
            db_path.to_str().unwrap(),
            &HashMap::new(),
            None,
        )?
    ));

    // Initialize backend
    backend.init().await?;

    // Create test table
    let table_name = "test_metrics";
    let schema = Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);

    backend.create_table(table_name, &schema).await?;

    // Create test data
    let values: Arc<dyn Array> = Arc::new(Float64Array::from(vec![1.0]));
    let timestamps: Arc<dyn Array> = Arc::new(Int64Array::from(vec![1000]));

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![values, timestamps],
    ).unwrap();

    // Insert test data
    backend.insert_into_table(table_name, batch).await?;

    // Query data
    let result = backend.query_table(table_name, None).await?;
    assert_eq!(result.num_columns(), 2);

    // Create aggregation view
    let view = AggregationView {
        source_table: table_name.to_string(),
        function: AggregateFunction::Avg,
        group_by: GroupBy {
            columns: vec![],
            time_column: Some("timestamp".to_string()),
        },
        window: TimeWindow::Fixed(Duration::from_secs(3600)), // 1 hour
        aggregate_columns: vec!["value".to_string()],
    };

    let view_name = format!("agg_view_{}", table_name);
    backend.create_aggregation_view(&view).await?;

    // Query aggregation view
    let result = backend.query_aggregation_view(&view_name).await?;
    assert_eq!(result.num_columns(), 2);

    Ok(())
}
