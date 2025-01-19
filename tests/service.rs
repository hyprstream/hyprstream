use arrow_flight::sql::client::FlightSqlServiceClient;
use tonic::Request;

use futures::StreamExt;
use tonic::Streaming;

use arrow_schema::{DataType, Field, Schema};
use hyprstream_core::aggregation::{GroupBy, TimeWindow};
use hyprstream_core::service::FlightSqlService;
use hyprstream_core::storage::duckdb::DuckDbBackend;
use hyprstream_core::storage::{BatchAggregation, StorageBackendType};
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

async fn create_test_service() -> (FlightSqlService, String) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let backend =
        DuckDbBackend::new(db_path.to_str().unwrap().to_string(), HashMap::new(), None).unwrap();

    // Find an available port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let service = FlightSqlService::new(StorageBackendType::DuckDb(backend));
    let endpoint = format!("http://127.0.0.1:{}", addr.port());

    (service, endpoint)
}

#[tokio::test]
async fn test_service_start() {
    let (service, endpoint) = create_test_service().await;

    // Start service in background
    tokio::spawn(async move {
        service.serve().await.unwrap();
    });

    // Wait for service to start
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Try to connect
    let channel = Channel::from_shared(endpoint).unwrap().connect().await;
    assert!(channel.is_ok());
}

#[tokio::test]
async fn test_create_table_and_query() {
    let (service, endpoint) = create_test_service().await;

    // Start service in background
    tokio::spawn(async move {
        service.serve().await.unwrap();
    });

    // Wait for service to start
    tokio::time::sleep(Duration::from_secs(1)).await;

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
    let (service, endpoint) = create_test_service().await;

    // Start service in background
    tokio::spawn(async move {
        service.serve().await.unwrap();
    });

    // Wait for service to start
    tokio::time::sleep(Duration::from_secs(1)).await;

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
