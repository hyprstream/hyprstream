use hyprstream_core::aggregation::{BatchAggregation, GroupBy, TimeWindow};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::time::Duration;

fn create_test_schema() -> Schema {
    Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("host", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ])
}

#[test]
fn test_batch_aggregation_no_window() {
    let schema = create_test_schema();
    let group_by = GroupBy {
        columns: vec!["metric".to_string(), "host".to_string()],
        time_column: None,
    };
    
    let agg = BatchAggregation::new(
        Arc::new(schema),
        "value".to_string(),
        group_by,
        None,
    );
    
    let sql = agg.build_query("test_table");
    assert!(sql.contains("SELECT"));
    assert!(sql.contains("FROM test_table"));
    assert!(sql.contains("GROUP BY metric, host"));
    assert!(!sql.contains("window_start"));
}

#[test]
fn test_batch_aggregation_fixed_window() {
    let schema = create_test_schema();
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
    
    let sql = agg.build_query("test_table");
    assert!(sql.contains("SELECT"));
    assert!(sql.contains("FROM test_table"));
    assert!(sql.contains("GROUP BY metric"));
    assert!(sql.contains("window_start"));
    assert!(sql.contains("timestamp / 60"));
}

#[test]
fn test_batch_aggregation_sliding_window() {
    let schema = create_test_schema();
    let group_by = GroupBy {
        columns: vec!["metric".to_string(), "host".to_string()],
        time_column: Some("timestamp".to_string()),
    };
    
    let window = TimeWindow::Sliding {
        window: Duration::from_secs(300),
        slide: Duration::from_secs(60),
    };
    let agg = BatchAggregation::new(
        Arc::new(schema),
        "value".to_string(),
        group_by,
        Some(window),
    );
    
    let sql = agg.build_query("test_table");
    assert!(sql.contains("SELECT"));
    assert!(sql.contains("FROM test_table"));
    assert!(sql.contains("GROUP BY metric, host"));
    assert!(sql.contains("window_start"));
    assert!(sql.contains("timestamp / 60"));
    assert!(sql.contains("300")); // Window size
}

#[test]
fn test_batch_aggregation_schema_validation() {
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        // Missing 'value' column
    ]);
    
    let group_by = GroupBy {
        columns: vec!["metric".to_string()],
        time_column: None,
    };
    
    let result = BatchAggregation::new(
        Arc::new(schema),
        "value".to_string(), // Column that doesn't exist
        group_by,
        None,
    );
    
    assert!(result.build_query("test_table").is_empty());
} 