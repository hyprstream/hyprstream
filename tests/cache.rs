mod common;

use arrow_adbc::{
    connection::Connection,
    database::Database,
    driver::FlightSqlDriver,
};
use arrow_array::RecordBatch;
use std::sync::Arc;
use tonic::Status;

#[tokio::test]
async fn test_cache_operations() -> Result<(), Status> {
    // Start test server with cached storage
    let TestServer { handle, addr, cache, store } = common::start_test_server(true).await;
    let endpoint = common::get_test_endpoint(addr);

    // Get cache and store backends
    let cache = cache.unwrap();
    let store = store.unwrap();

    // Create test table directly in store
    let table_name = "test_metrics";
    let create_sql = format!(
        "CREATE TABLE {} (
            value DOUBLE PRECISION NOT NULL,
            timestamp BIGINT NOT NULL
        )", table_name
    );
    let stmt_handle = store.prepare_sql(&create_sql).await?;
    store.query_sql(&stmt_handle).await?;

    // Insert test data directly into store
    let insert_sql = format!(
        "INSERT INTO {} VALUES (1.0, 1000)",
        table_name
    );
    let stmt_handle = store.prepare_sql(&insert_sql).await?;
    store.query_sql(&stmt_handle).await?;

    // Connect using ADBC to query through cache
    let driver = FlightSqlDriver::new();
    let database = Database::connect(&driver, &endpoint).unwrap();
    let mut connection = database.connect().unwrap();

    // First query - should be a cache miss and fetch from store
    let query_sql = format!("SELECT * FROM {}", table_name);
    let mut stmt = connection.prepare(&query_sql, None).unwrap();
    let result = stmt.query(None).unwrap();
    let batches: Vec<RecordBatch> = result.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches[0].num_columns(), 2);

    // Verify data is now in cache
    let cache_query = format!("SELECT * FROM {}", table_name);
    let stmt_handle = cache.prepare_sql(&cache_query).await?;
    let cache_result = cache.query_sql(&stmt_handle).await?;
    assert_eq!(cache_result.num_columns(), 2);

    // Insert more data directly into store
    let insert_sql = format!(
        "INSERT INTO {} VALUES (2.0, 2000)",
        table_name
    );
    let stmt_handle = store.prepare_sql(&insert_sql).await?;
    store.query_sql(&stmt_handle).await?;

    // Query again - should get updated data through cache
    let mut stmt = connection.prepare(&query_sql, None).unwrap();
    let result = stmt.query(None).unwrap();
    let batches: Vec<RecordBatch> = result.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches[0].num_rows(), 2); // Should see both rows

    // Clean up
    handle.abort();

    Ok(())
}
