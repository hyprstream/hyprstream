use arrow::array::{Array, Float32Array};
use arrow_flight::flight_service_client::FlightServiceClient;
use bytes::Bytes;
use futures::StreamExt;
use hyprstream_core::{
    models::{Model, ModelLayer, ModelMetadata, ModelVersion, storage::TimeSeriesModelStorage, ModelStorage},
    service::FlightSqlService,
    storage::{StorageBackend, StorageBackendType, duckdb::DuckDbBackend},
};
use std::{collections::HashMap, sync::Arc, time::SystemTime};
use tempfile::tempdir;

#[tokio::test]
async fn test_model_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary directory for the database
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    // Create storage backend
    let engine_backend = Arc::new(StorageBackendType::DuckDb(
        DuckDbBackend::new_with_options(
            db_path.to_str().unwrap(),
            &HashMap::new(),
            None,
        )?
    ));

    // Initialize backend
    engine_backend.init().await?;

    // Create model storage
    let model_storage = Box::new(TimeSeriesModelStorage::new(engine_backend.clone()));
    model_storage.init().await?;

    // Start test server
    let addr = "127.0.0.1:50052"; // Use different port from other tests
    let service = FlightSqlService::new(engine_backend, model_storage);
    let server = tonic::transport::Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve(addr.parse()?);

    tokio::spawn(server);

    // Wait a bit for server to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create client
    let mut client = FlightServiceClient::connect(format!("http://{}", addr)).await?;

    // Test store model
    let model = create_test_model();
    let request = tonic::Request::new(arrow_flight::Action {
        r#type: "model.store".to_string(),
        body: Bytes::from(serde_json::to_vec(&model)?),
    });

    let mut response = client.do_action(request).await?.into_inner();
    assert!(response.next().await.is_some());

    // Test list models
    let request = tonic::Request::new(arrow_flight::Action {
        r#type: "model.list".to_string(),
        body: Bytes::new(),
    });

    let mut response = client.do_action(request).await?.into_inner();
    let models = response.next().await.unwrap()?;
    assert!(!models.body.is_empty());

    Ok(())
}

fn create_test_model() -> Model {
    let version = ModelVersion {
        version: "v1".to_string(),
        created_at: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64,
        description: "Initial version".to_string(),
        parent_version: None,
    };

    let metadata = ModelMetadata {
        model_id: "test-model".to_string(),
        name: "Test Model".to_string(),
        architecture: "dense".to_string(),
        version,
        parameters: HashMap::from([
            ("input_size".to_string(), "3".to_string()),
            ("output_size".to_string(), "1".to_string()),
            ("activation".to_string(), "relu".to_string()),
        ]),
    };

    let weights: Vec<Arc<dyn Array>> = vec![
        Arc::new(Float32Array::from(vec![1.0_f32, 2.0_f32, 3.0_f32])),
    ];

    let layers = vec![ModelLayer {
        name: "layer1".to_string(),
        layer_type: "dense".to_string(),
        shape: vec![3, 1],
        weights,
        parameters: HashMap::from([
            ("activation".to_string(), "relu".to_string()),
        ]),
    }];

    Model::new(metadata, layers)
}
