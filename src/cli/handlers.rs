use anyhow::{Result, Context};
use std::sync::Arc;
use crate::{
    config::Settings,
    models::{storage::TimeSeriesModelStorage, ModelStorage},
    storage::{
        StorageBackendType,
        StorageBackend,
        adbc::AdbcBackend,
        duckdb::DuckDbBackend,
    },
    service::FlightSqlService,
};
use tonic::transport::Server;
use tracing_subscriber::{fmt, EnvFilter};
use crate::cli::commands::{Commands, ModelCommands};
use std::path::PathBuf;

pub async fn run_command(command: Commands, config: Option<PathBuf>) -> Result<()> {
    let (engine_backend, model_storage) = init_core(config).await?;

    match command {
        Commands::Server(cmd) => {
            handle_server(cmd.listen, cmd.debug, engine_backend, model_storage).await?;
        }
        Commands::Models { command } => {
            handle_models(command, model_storage).await?;
        }
        Commands::Chat(cmd) => {
            handle_chat(cmd.model_tag, model_storage).await?;
        }
        Commands::Sql(cmd) => {
            handle_sql(cmd.host).await?;
        }
    }

    Ok(())
}

async fn init_core(config: Option<PathBuf>) -> Result<(Arc<StorageBackendType>, Box<dyn ModelStorage>)> {
    let settings = Settings::new(config)?;

    // Initialize tracing
    let subscriber = fmt()
        .with_env_filter(EnvFilter::new(&settings.server.log_level))
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    // Create storage backend
    let engine_backend: Arc<StorageBackendType> = Arc::new(match settings.engine.engine.as_str() {
        "adbc" => StorageBackendType::Adbc(
            AdbcBackend::new_with_options(
                &settings.engine.connection,
                &settings.engine.options,
                settings.engine.credentials.as_ref(),
            ).context("Failed to create ADBC backend")?
        ),
        "duckdb" => StorageBackendType::DuckDb(
            DuckDbBackend::new_with_options(
                &settings.engine.connection,
                &settings.engine.options,
                settings.engine.credentials.as_ref(),
            ).context("Failed to create DuckDB backend")?
        ),
        _ => anyhow::bail!("Unsupported engine type"),
    });

    // Initialize backend
    engine_backend.init().await.context("Failed to initialize storage backend")?;

    // Create model storage
    let model_storage = Box::new(TimeSeriesModelStorage::new(engine_backend.clone()));
    model_storage.init().await.context("Failed to initialize model storage")?;

    Ok((engine_backend, model_storage))
}

async fn handle_server(
    listen: Option<String>,
    debug: bool,
    engine_backend: Arc<StorageBackendType>,
    model_storage: Box<dyn ModelStorage>,
) -> Result<()> {
    let addr = listen.unwrap_or_else(|| "127.0.0.1:50051".to_string()).parse()
        .context("Invalid listen address")?;

    if debug {
        tracing::warn!("Running in debug mode");
    }
    
    tracing::warn!("This is a pre-release alpha for preview purposes only.");
    tracing::info!("Starting server on {}", addr);

    let service = FlightSqlService::new(engine_backend, model_storage);
    
    Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service))
        .serve(addr)
        .await
        .context("Server error")?;

    Ok(())
}

async fn handle_models(cmd: ModelCommands, model_storage: Box<dyn ModelStorage>) -> Result<()> {
    match cmd {
        ModelCommands::List => {
            let models = model_storage.list_models().await?;
            for model in models {
                println!("{}", model);
            }
        }
        ModelCommands::Import { path } => {
            model_storage.import_model(&path).await?;
            println!("Model imported successfully");
        }
        ModelCommands::Delete { model_id } => {
            model_storage.delete_model(&model_id).await?;
            println!("Model deleted successfully");
        }
        ModelCommands::Inspect { model_id } => {
            let info = model_storage.get_model_info(&model_id).await?;
            println!("{:#?}", info);
        }
        ModelCommands::Tag { model_id, tag } => {
            model_storage.tag_model(&model_id, &tag).await?;
            println!("Model tagged successfully");
        }
    }
    Ok(())
}

async fn handle_chat(model_tag: String, model_storage: Box<dyn ModelStorage>) -> Result<()> {
    let model = model_storage.load_model(&model_tag).await?;
    println!("Starting chat with model {}", model_tag);
    // TODO: Implement interactive chat loop
    Ok(())
}

async fn handle_sql(host: Option<String>) -> Result<()> {
    let host = host.unwrap_or_else(|| "127.0.0.1:50051".to_string());
    println!("Connecting to {}", host);
    // TODO: Implement SQL client
    Ok(())
}
