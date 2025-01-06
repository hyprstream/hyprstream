use hyprstream_core::{
    storage::{StorageBackend, DuckDbBackend},
    aggregation::{TimeWindow, BatchAggregation, AggregateFunction, GroupBy, AggregationView},
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use std::collections::HashMap;

async fn create_test_backend() -> Box<dyn StorageBackend> {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let backend = DuckDbBackend::new(
        db_path.to_str().unwrap().to_string(),
        HashMap::new(),
        None,
    ).unwrap();
    Box::new(backend)
}

#[tokio::test]
async fn test_backend_creation() {
    let backend = create_test_backend().await;
    assert!(backend.is_ok());
}

#[tokio::test]
async fn test_backend_insert_and_query() {
    let backend = create_test_backend().await;
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);
    
    // Create table
    let table_name = "test_metrics";
    let schema_arc = Arc::new(schema.clone());
    backend.create_table(table_name, &schema_arc).await.unwrap();
    
    // Insert test data
    let batch = create_test_batch(&schema);
    backend.insert(table_name, batch).await.unwrap();
    
    // Query data
    let result = backend.query_sql("SELECT * FROM test_metrics").await.unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[tokio::test]
async fn test_backend_aggregation() {
    let backend = create_test_backend().await;
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);
    
    // Create source table
    let table_name = "test_metrics";
    let schema_arc = Arc::new(schema.clone());
    backend.create_table(table_name, &schema_arc).await.unwrap();
    
    // Insert test data
    let batch = create_test_batch(&schema);
    backend.insert(table_name, batch).await.unwrap();
    
    // Create aggregation
    let group_by = GroupBy {
        columns: vec!["metric".to_string()],
        time_column: Some("timestamp".to_string()),
    };
    
    let window = TimeWindow::Fixed(Duration::from_secs(60));
    let agg = BatchAggregation::new(
        Arc::new(schema),
        "value".to_string(),
        group_by,
        Some(window),
    );
    
    // Create aggregation view
    let view_name = "test_agg_view";
    let view = AggregationView {
        name: view_name.to_string(),
        source_table: table_name.to_string(),
        aggregation: agg,
    };
    backend.create_aggregation_view(&view).await.unwrap();
    
    // Query aggregation
    let result = backend.query_sql("SELECT * FROM test_agg_view").await.unwrap();
    assert_eq!(result.num_rows(), 1);
}

fn create_test_batch(schema: &Schema) -> arrow_array::RecordBatch {
    use arrow_array::{StringArray, Float64Array, Int64Array};
    
    let metric_array = StringArray::from(vec!["test_metric"]);
    let value_array = Float64Array::from(vec![42.0]);
    let timestamp_array = Int64Array::from(vec![1000]);
    
    arrow_array::RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(metric_array),
            Arc::new(value_array),
            Arc::new(timestamp_array),
        ],
    ).unwrap()
} 