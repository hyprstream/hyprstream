use arrow_flight::sql::client::FlightSqlServiceClient;
use tonic::Request;


use arrow_schema::{DataType, Field, Schema};
use hyprstream_core::aggregation::{GroupBy, TimeWindow};
use hyprstream_core::service::FlightSqlServer;
use hyprstream_core::storage::duckdb::DuckDbBackend;
use hyprstream_core::storage::StorageBackendType;
use hyprstream_core::metrics::storage::BatchAggregation;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tonic::transport::Channel;

trait FlightSqlClientExt {
    async fn query_sql(&mut self, sql: String) -> Result<(), tonic::Status>;
}

impl FlightSqlClientExt for FlightSqlServiceClient<Channel> {
    async fn query_sql(&mut self, sql: String) -> Result<(), tonic::Status> {
        let request = Request::new(arrow_flight::Action {
            r#type: "sql.query".to_string(),
            body: sql.into_bytes().into(),
        });
        let mut stream = self
            .do_action(request)
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to execute query: {}", e)))?;
        while let Some(result) = stream
            .message()
            .await
            .map_err(|e| tonic::Status::internal(format!("Failed to read result: {}", e)))?
        {
            let _ = result;
        }
        Ok(())
    }
}

async fn create_test_service() -> String {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let backend =
        DuckDbBackend::new(db_path.to_str().unwrap().to_string(), HashMap::new(), None).unwrap();

    // Create and bind the listener
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let endpoint = format!("http://127.0.0.1:{}", addr.port());

    let service = FlightSqlServer::new(StorageBackendType::DuckDb(backend));
    let incoming_stream = tokio_stream::wrappers::TcpListenerStream::new(listener);
    
    // Spawn the service
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service.into_service())
            .serve_with_incoming(incoming_stream)
            .await
            .unwrap();
    });

    // Wait for service to start
    tokio::time::sleep(Duration::from_secs(1)).await;

    endpoint
}

#[tokio::test]
async fn test_service_start() {
    let endpoint = create_test_service().await;

    // Try to connect
    let channel = Channel::from_shared(endpoint).unwrap().connect().await;
    assert!(channel.is_ok());
}

#[tokio::test]
async fn test_create_table_and_query() {
    let endpoint = create_test_service().await;

    // Connect client
    let channel = Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightSqlServiceClient::new(channel);

    // Create table
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);

    let create_table_sql = "CREATE TABLE test_metrics (
        metric VARCHAR NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        timestamp BIGINT NOT NULL
    )";

    let result = client.execute(create_table_sql.into(), None).await;
    assert!(result.is_ok());

    // Insert data
    let insert_sql = "INSERT INTO test_metrics VALUES ('test_metric', 42.0, 1000)";
    let result = client.execute(insert_sql.into(), None).await;
    assert!(result.is_ok());

    // Query data
    let query_sql = "SELECT * FROM test_metrics";
    let result = client.query_sql(query_sql.into()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_create_aggregation_view() {
    let endpoint = create_test_service().await;

    // Connect client
    let channel = Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightSqlServiceClient::new(channel);

    // Create source table
    let create_table_sql = "CREATE TABLE test_metrics (
        metric VARCHAR NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        timestamp BIGINT NOT NULL
    )";

    client.execute(create_table_sql.into(), None).await.unwrap();

    // Create aggregation view
    let schema = Schema::new(vec![
        Field::new("metric", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);

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

    let create_view_sql = format!(
        "CREATE VIEW test_agg_view AS {}",
        agg.build_query("test_metrics")
    );

    let result = client.execute(create_view_sql.into(), None).await;
    assert!(result.is_ok());

    // Query view
    let query_sql = "SELECT * FROM test_agg_view";
    let result = client.query_sql(query_sql.into()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_simple_sql_execution() {
    let endpoint = create_test_service().await;
    let channel = Channel::from_shared(endpoint).unwrap().connect().await.unwrap();
    let mut client = FlightSqlServiceClient::new(channel);
    
    let result = client.query_sql("SELECT 1;".into()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_query_planner_integration() {
    let endpoint = create_test_service().await;

    // Connect client
    let channel = Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = FlightSqlServiceClient::new(channel);

    // Create test table with data
    let create_table_sql = "CREATE TABLE metrics (
        name VARCHAR NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        timestamp BIGINT NOT NULL
    )";
    client.execute(create_table_sql.into(), None).await.unwrap();

    // Insert test data
    let insert_sql = "INSERT INTO metrics VALUES
        ('cpu', 80.0, 1000),
        ('memory', 60.0, 1000),
        ('cpu', 85.0, 2000),
        ('memory', 65.0, 2000)";
    client.execute(insert_sql.into(), None).await.unwrap();

    // Test aggregation query that should use vector operations
    let query_sql = "SELECT name, AVG(value) as avg_value
                    FROM metrics
                    GROUP BY name
                    ORDER BY name";
    let result = client.query_sql(query_sql.into()).await;
    assert!(result.is_ok());

    // Test time-based query that should use predicate pushdown
    let query_sql = "SELECT * FROM metrics
                    WHERE timestamp >= 1500
                    ORDER BY timestamp";
    let result = client.query_sql(query_sql.into()).await;
    assert!(result.is_ok());
}
