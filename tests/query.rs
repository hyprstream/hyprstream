use arrow::array::{Float32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use hyprstream_core::{
    query::{
        DataFusionExecutor, DataFusionPlanner, ExecutorConfig, OptimizationHint, PhysicalOperator,
        Query, QueryExecutor, QueryPlanner, VectorizedOperator,
    },
    storage::{duckdb::DuckDbBackend, StorageBackend},
};
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_simple_query_execution() -> Result<()> {
    // Create a simple schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float32, false),
    ]));

    // Create test data
    let id_array = Int64Array::from(vec![1, 2, 3, 4, 5]);
    let value_array = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(value_array)],
    )?;

    // Create query engine with default configuration
    let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
    let planner = DataFusionPlanner::new(backend.clone()).await?;
    let executor = DataFusionExecutor::new(ExecutorConfig::default());

    // Create a simple query
    let query = Query {
        sql: "SELECT id, value FROM test_table WHERE value > 2.0".to_string(),
        schema_hint: Some(schema.as_ref().clone()),
        hints: vec![OptimizationHint::PreferPredicatePushdown],
    };

    // Execute query
    let results = executor.execute_collect(planner.plan_query(&query).await?).await?;

    // Verify results
    assert!(!results.is_empty());
    let result_batch = &results[0];
    assert_eq!(result_batch.schema().as_ref(), schema.as_ref());
    Ok(())
}

#[tokio::test]
async fn test_vector_operations() -> Result<()> {
    // Create schema for vector data
    let schema = Arc::new(Schema::new(vec![
        Field::new("vec1", DataType::Float32, false),
        Field::new("vec2", DataType::Float32, false),
    ]));

    // Create test vectors
    let vec1 = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let vec2 = Float32Array::from(vec![4.0, 5.0, 6.0]);

    let input_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(vec1), Arc::new(vec2)])?;

    // Create vector operator
    let mut properties = HashMap::new();
    properties.insert("operation".to_string(), "add".to_string());
    properties.insert("input_columns".to_string(), "vec1,vec2".to_string());
    properties.insert("output_column".to_string(), "result".to_string());

    let operator = VectorizedOperator::new(schema.clone(), vec![], properties)?;

    // Execute vector operation
    let results = operator.execute(vec![input_batch])?;

    // Verify results
    assert_eq!(results.len(), 1);
    let result_batch = &results[0];
    assert_eq!(result_batch.num_columns(), 3); // original columns + result
    Ok(())
}

#[tokio::test]
async fn test_query_optimization() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create test table
    let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
    backend.init().await?;
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float32, false),
    ]));
    backend.create_table("test_table", &schema).await?;

    // Insert test data
    let id_array = Int64Array::from(vec![1, 2, 3]);
    let value_array = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(id_array), Arc::new(value_array)])?;
    backend.insert_into_table("test_table", batch).await?;

    // Create query engine with optimization hints
    let planner = DataFusionPlanner::new(backend.clone()).await?;
    let executor = DataFusionExecutor::new(ExecutorConfig::default());

    // Create a query that should benefit from optimizations
    let query = Query {
        sql: "SELECT id, AVG(value) FROM test_table GROUP BY id".to_string(),
        schema_hint: None,
        hints: vec![],
    };

    // Plan and execute query
    let plan = planner.create_logical_plan(&query).await?;
    assert!(plan.schema().fields().len() > 0);
    Ok(())
}

#[tokio::test]
async fn test_executor_configuration() -> Result<()> {
    // Create executor with custom configuration
    let config = ExecutorConfig {
        max_concurrent_tasks: 4,
        batch_size: 1024,
        memory_limit: 512 * 1024 * 1024, // 512MB
    };

    let executor = DataFusionExecutor::new(config);

    // Create a simple schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float32, false),
    ]));

    // Create test data
    let id_array = Int64Array::from(vec![1, 2, 3]);
    let value_array = Float32Array::from(vec![1.0, 2.0, 3.0]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(value_array)],
    )?;

    // Create and execute a simple plan
    let plan = create_test_plan(batch)?;
    let result = executor.execute_collect(plan).await?;
    assert!(!result.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_streaming_execution() -> Result<()> {
    let backend = Arc::new(DuckDbBackend::new_in_memory().unwrap());
    let planner = DataFusionPlanner::new(backend.clone()).await?;
    let executor = DataFusionExecutor::new(ExecutorConfig::default());

    // Create a query that produces multiple batches
    let query = Query {
        sql: "SELECT * FROM test_table".to_string(),
        schema_hint: None,
        hints: vec![OptimizationHint::OptimizeForStreaming],
    };

    // Execute as stream
    let mut stream = executor.execute_stream(planner.plan_query(&query).await?).await?;

    // Should be able to consume stream
    use futures::StreamExt;
    while let Some(result) = stream.next().await {
        assert!(result.is_ok());
    }
    Ok(())
}

// Helper function to create a test physical plan
fn create_test_plan(
    batch: RecordBatch,
) -> Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
    use datafusion::physical_plan::memory::MemoryExec;

    let schema = batch.schema();
    Ok(Arc::new(MemoryExec::try_new(&[vec![batch]], schema, None)?))
}
