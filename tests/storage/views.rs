use arrow_schema::{DataType, Field, Schema};
use hyprstream_core::aggregation::{AggregateFunction, GroupBy, TimeWindow};
use hyprstream_core::storage::view::{AggregationSpec, ViewDefinition};
use hyprstream_core::storage::{DuckDbBackend, StorageBackend};
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_view_creation_and_query() -> Result<(), Box<dyn std::error::Error>> {
    // Create test backend
    let backend = DuckDbBackend::new_in_memory()?;
    backend.init().await?;

    // Create source table
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]));
    backend.create_table("test_source", &source_schema).await?;

    // Create view definition
    let view_def = ViewDefinition::new(
        "test_source".to_string(),
        vec!["metric".to_string()],
        vec![AggregationSpec {
            column: "value".to_string(),
            function: AggregateFunction::Avg,
        }],
        Some(GroupBy {
            columns: vec!["metric".to_string()],
            time_column: Some("timestamp".to_string()),
        }),
        Some(TimeWindow::Fixed(Duration::from_secs(60))),
        Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("avg_value", DataType::Float64, false),
        ])),
    );

    // Create view
    backend.create_view("test_view", view_def.clone()).await?;

    // Verify view exists
    let views = backend.list_views().await?;
    assert!(views.contains(&"test_view".to_string()));

    // Get view metadata
    let metadata = backend.get_view("test_view").await?;
    assert_eq!(metadata.definition.source_table, "test_source");
    assert_eq!(metadata.definition.columns, vec!["metric"]);

    // Drop view
    backend.drop_view("test_view").await?;

    // Verify view was dropped
    let views = backend.list_views().await?;
    assert!(!views.contains(&"test_view".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_view_dependencies() -> Result<(), Box<dyn std::error::Error>> {
    let backend = DuckDbBackend::new_in_memory()?;
    backend.init().await?;

    // Create source tables
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]));
    backend.create_table("source_a", &source_schema).await?;
    backend.create_table("source_b", &source_schema).await?;

    // Create view definition with dependencies
    let view_def = ViewDefinition::new(
        "source_a".to_string(),
        vec!["metric".to_string()],
        vec![AggregationSpec {
            column: "value".to_string(),
            function: AggregateFunction::Sum,
        }],
        Some(GroupBy {
            columns: vec!["metric".to_string()],
            time_column: None,
        }),
        None,
        Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("sum_value", DataType::Float64, false),
        ])),
    );

    // Create view
    backend.create_view("test_view", view_def).await?;

    // Get view metadata and verify dependencies
    let metadata = backend.get_view("test_view").await?;
    assert!(metadata.definition.dependencies.contains("source_a"));
    assert!(!metadata.definition.dependencies.contains("source_b"));

    Ok(())
}

#[tokio::test]
async fn test_view_sql_generation() -> Result<(), Box<dyn std::error::Error>> {
    // Create view definition
    let view_def = ViewDefinition::new(
        "metrics".to_string(),
        vec!["metric".to_string()],
        vec![
            AggregationSpec {
                column: "value".to_string(),
                function: AggregateFunction::Avg,
            },
            AggregationSpec {
                column: "value".to_string(),
                function: AggregateFunction::Sum,
            },
        ],
        Some(GroupBy {
            columns: vec!["metric".to_string()],
            time_column: Some("timestamp".to_string()),
        }),
        Some(TimeWindow::Fixed(Duration::from_secs(60))),
        Arc::new(Schema::new(vec![
            Field::new("metric", DataType::Utf8, false),
            Field::new("avg_value", DataType::Float64, false),
            Field::new("sum_value", DataType::Float64, false),
        ])),
    );

    let sql = view_def.to_sql();
    
    // Verify SQL contains expected clauses
    assert!(sql.contains("SELECT metric"));
    assert!(sql.contains("AVG(value) as avg_value"));
    assert!(sql.contains("SUM(value) as sum_value"));
    assert!(sql.contains("FROM metrics"));
    assert!(sql.contains("GROUP BY metric"));

    Ok(())
}