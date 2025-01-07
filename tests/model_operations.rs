use hyprstream_core::{
    service::FlightSqlService,
    models::{Model, ModelLayer},
};
use arrow_flight::flight_service_client::FlightServiceClient;
use tonic::transport::Channel;

#[tokio::test]
async fn test_model_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
    // Start test server
    let addr = "127.0.0.1:50051";
    let service = FlightSqlService::new(/* ... */);
    let server = tonic::transport::Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve(addr.parse()?);
    
    tokio::spawn(server);
    
    // Create client
    let mut client = FlightServiceClient::connect(format!("http://{}", addr)).await?;
    
    // Test store model
    let model = create_test_model();
    let request = tonic::Request::new(arrow_flight::Action {
        r#type: "model.store".to_string(),
        body: serde_json::to_vec(&model)?,
    });
    
    let response = client.do_action(request).await?;
    assert!(response.into_inner().next().await.is_some());
    
    // Test list models
    let request = tonic::Request::new(arrow_flight::Action {
        r#type: "model.list".to_string(),
        body: vec![],
    });
    
    let response = client.do_action(request).await?;
    let models = response.into_inner().next().await.unwrap()?;
    assert!(!models.body.is_empty());
    
    Ok(())
}

fn create_test_model() -> Model {
    Model::new(
        "test-model".to_string(),
        "v1".to_string(),
        vec![
            ModelLayer {
                name: "layer1".to_string(),
                weights: vec![1.0, 2.0, 3.0],
            }
        ],
    )
} 