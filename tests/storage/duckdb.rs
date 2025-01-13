use hyprstream_core::storage::duckdb::DuckDBBackend;
use hyprstream_core::storage::StorageBackend;
use hyprstream_core::aggregation::{BatchAggregation, GroupBy, TimeWindow};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;

async fn create_test_backend() -> DuckDBBackend {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    DuckDBBackend::new(db_path.to_str().unwrap()).await.unwrap()
}

#[tokio::test]
async fn test_create_table() {
    let backend = create_test_backend().await;
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);
    
    let result = backend.create_table("test_metrics", Arc::new(schema)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_insert_and_query() {
    let backend = create_test_backend().await;
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);
    
    // Create table
    backend.create_table("test_metrics", Arc::new(schema.clone())).await.unwrap();
    
    // Insert test data
    let batch = create_test_batch(&schema);
    backend.insert("test_metrics", batch).await.unwrap();
    
    // Query data
    let result = backend.query("SELECT * FROM test_metrics").await.unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[tokio::test]
async fn test_aggregation_view() {
    let backend = create_test_backend().await;
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);
    
    // Create source table
    backend.create_table("test_metrics", Arc::new(schema.clone())).await.unwrap();
    
    // Create aggregation view
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
    
    let result = backend.create_aggregation_view(
        "test_agg_view",
        "test_metrics",
        &agg,
    ).await;
    assert!(result.is_ok());
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