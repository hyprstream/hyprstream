//! Hyprstream binary - ML inference server
//!
//! This binary provides the main entry point for the Hyprstream service.

use clap::Parser;
use config::Config;
use hyprstream_core::cli::commands::Commands;
use hyprstream_core::cli::handlers::{
    handle_server, handle_model_command,
    handle_chat_command
};
use hyprstream_core::cli::{
    DeviceConfig, DevicePreference, RuntimeConfig,
    handle_branch, handle_checkout, handle_status, handle_commit,
    handle_lora_train, handle_serve, handle_infer,
    handle_push, handle_pull, handle_merge,
    handle_list, handle_info, handle_clone
};
use tracing::info;
#[cfg(feature = "otel")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
#[cfg(not(feature = "otel"))]
use tracing_subscriber::EnvFilter;

#[cfg(feature = "otel")]
use tracing_opentelemetry::OpenTelemetryLayer;

#[derive(Parser)]
#[command(
    name = "hyprstream",
    version = env!("CARGO_PKG_VERSION"),
    about = "Real-time adaptive ML inference server with dynamic sparse weight adjustments",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}


/// Execute a command with the appropriate runtime configuration
fn with_runtime<F, Fut>(config: RuntimeConfig, handler: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error>>>,
{
    // Check GPU availability early if required
    if matches!(config.device.preference, DevicePreference::RequireGPU) {
        // Early GPU check - this is a simplified check
        // The actual GPU initialization happens in the engines
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() &&
           std::env::var("HIP_VISIBLE_DEVICES").is_err() &&
           !std::path::Path::new("/usr/local/cuda").exists() &&
           !std::path::Path::new("/opt/rocm").exists() {
            return Err("GPU required but no CUDA or ROCm installation detected. Set CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES, or install CUDA/ROCm.".into());
        }
    }

    // Set environment hints for the runtime config
    match config.device.preference {
        DevicePreference::RequestCPU => {
            std::env::set_var("HYPRSTREAM_FORCE_CPU", "1");
        }
        DevicePreference::RequestGPU | DevicePreference::RequireGPU => {
            std::env::remove_var("HYPRSTREAM_FORCE_CPU");
        }
    }

    // Create appropriate runtime
    if config.multi_threaded {
        // Multi-threaded runtime for GPU workloads and server
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(handler())?;
    } else {
        // Single-threaded runtime for lightweight operations
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(handler())?;
    }
    Ok(())
}

/// Server command handler - requires multi-threaded runtime for GPU
fn handle_server_cmd(config: Config) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> {
    async move { handle_server(config).await }
}

/// Model command handler - GPU requirements depend on action
fn handle_model_cmd(cmd: hyprstream_core::cli::commands::model::ModelCommand, server_url: String) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> {
    async move { handle_model_command(cmd, server_url).await }
}


/// Chat command handler - requires multi-threaded runtime for GPU
fn handle_chat_cmd(cmd: hyprstream_core::cli::commands::chat::ChatCommand, server_url: String) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> {
    async move { handle_chat_command(cmd, server_url).await }
}

#[cfg(feature = "otel")]
/// Telemetry provider type
#[derive(Debug, Clone, Copy)]
enum TelemetryProvider {
    /// Use OTLP exporter for production telemetry
    Otlp,
    /// Use stdout exporter for local debugging
    Stdout,
}

#[cfg(feature = "otel")]
/// Initialize OpenTelemetry with the specified provider
fn init_telemetry(provider: TelemetryProvider) -> Result<(), Box<dyn std::error::Error>> {
    use opentelemetry_sdk::Resource;

    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .unwrap_or_else(|_| "hyprstream".to_string());

    // Create resource with service information
    use opentelemetry::KeyValue;
    let resource = Resource::builder()
        .with_service_name(service_name.clone())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .build();

    // Create the appropriate exporter based on provider type
    

    let tracer_provider = match provider {
        TelemetryProvider::Otlp => {
            use opentelemetry_otlp::{SpanExporter, WithExportConfig};

            let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:4317".to_string());

            info!("Using OTLP exporter for OpenTelemetry at {}", endpoint);

            let exporter = SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .build()
                .map_err(|e| format!("Failed to create OTLP exporter: {}", e))?;

            opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_resource(resource.clone())
                .with_simple_exporter(exporter)
                .build()
        }
        TelemetryProvider::Stdout => {
            info!("Using stdout exporter for OpenTelemetry (debug)");

            opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_resource(resource)
                .with_simple_exporter(opentelemetry_stdout::SpanExporter::default())
                .build()
        }
    };

    // Get tracer
    use opentelemetry::trace::TracerProvider;
    let tracer = tracer_provider.tracer("hyprstream");

    // Setup tracing subscriber with OpenTelemetry layer
    let otel_layer = OpenTelemetryLayer::new(tracer);

    // Set log level based on provider
    let default_log_level = match provider {
        TelemetryProvider::Otlp => "hyprstream=info,opentelemetry=info",
        TelemetryProvider::Stdout => "hyprstream=debug,opentelemetry=debug",
    };

    let filter = EnvFilter::builder()
        .parse_lossy(std::env::var("RUST_LOG")
            .unwrap_or_else(|_| default_log_level.to_string()));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_file(true)
        .with_line_number(true);

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    info!("OpenTelemetry initialized with {:?} provider", provider);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let cli = Cli::parse();

    #[cfg(feature = "otel")]
    {
        // Determine telemetry provider based on command
        let telemetry_provider = match &cli.command {
            Commands::Server(_) => TelemetryProvider::Otlp,  // Server uses OTLP
            _ => TelemetryProvider::Stdout,                  // All CLI commands use stdout
        };

        // Initialize OpenTelemetry based on environment
        let otel_enabled = std::env::var("HYPRSTREAM_OTEL_ENABLE")
            .unwrap_or_else(|_| "false".to_string())  // Default to disabled
            .parse::<bool>()
            .unwrap_or(false);

        if otel_enabled {
            init_telemetry(telemetry_provider)?;
        } else {
            // Fallback to simple logging without OpenTelemetry
            let default_log_level = match telemetry_provider {
                TelemetryProvider::Otlp => "hyprstream=info",
                TelemetryProvider::Stdout => "hyprstream=debug",
            };

            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::builder()
                        .parse_lossy(std::env::var("RUST_LOG")
                            .unwrap_or_else(|_| default_log_level.to_string()))
                )
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .init();
        }
    }

    #[cfg(not(feature = "otel"))]
    {
        // Simple logging without OpenTelemetry support
        let default_log_level = match &cli.command {
            Commands::Server(_) => "hyprstream=info",
            _ => "hyprstream=debug",
        };

        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::builder()
                    .parse_lossy(std::env::var("RUST_LOG")
                        .unwrap_or_else(|_| default_log_level.to_string()))
            )
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .init();
    }
    
    info!("Hyprstream v{} starting up", env!("CARGO_PKG_VERSION"));

    // Handle commands with appropriate runtime configuration
    match cli.command {
        Commands::Server(_cmd) => {
            let config = Config::builder()
                .set_default("host", "127.0.0.1")?
                .set_default("port", "50051")?
                .build()?;

            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true
                },
                || handle_server_cmd(config)
            )?
        }
        Commands::Model(cmd) => {
            let server_url = "http://127.0.0.1:50051".to_string();
            let device_config = cmd.action.device_config();
            let multi_threaded = matches!(device_config.preference, DevicePreference::RequestGPU | DevicePreference::RequireGPU);

            with_runtime(
                RuntimeConfig {
                    device: device_config,
                    multi_threaded
                },
                || handle_model_cmd(cmd, server_url)
            )?
        }
        Commands::Chat(cmd) => {
            let server_url = "http://127.0.0.1:50051".to_string();

            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true
                },
                || handle_chat_cmd(cmd, server_url)
            )?
        }

        // Phase 1: Git-style commands
        Commands::Branch { model, name, from } => {
            // Create ModelStorage for git operations
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_branch(&storage, &model, &name, from).await
            })?;
        }

        Commands::Checkout { model_ref, create_branch, force } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_checkout(&storage, &model_ref, create_branch, force).await
            })?;
        }

        Commands::Status { model, verbose } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_status(&storage, model, verbose).await
            })?;
        }

        Commands::Commit { model, message, all } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_commit(&storage, &model, &message, all).await
            })?;
        }

        Commands::LoraTrain { model, adapter, index, rank, learning_rate,
                              batch_size, epochs, data, interactive, config } => {
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true
                },
                || async {
                    let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                    let storage = hyprstream_core::storage::ModelStorage::create(
                        storage_paths.models_dir()?
                    ).await?;
                    handle_lora_train(&storage, &model, adapter, index, rank, learning_rate,
                                     batch_size, epochs, data, interactive, config).await
                        .map_err(|e| e.into())
                }
            )?;
        }

        Commands::FineTune { model, config } => {
            println!("Fine-tuning not yet implemented for model: {}", model);
            println!("Config: {:?}", config);
        }

        Commands::PreTrain { model, config } => {
            println!("Pre-training not yet implemented for model: {}", model);
            println!("Config: {:?}", config);
        }

        Commands::Serve { model, port, host } => {
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true
                },
                || async { handle_serve(model, port, &host).await.map_err(|e| e.into()) }
            )?;
        }

        Commands::Infer { model, prompt, max_tokens, temperature, top_p, top_k, stream, force_download } => {
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true
                },
                || async {
                    let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                    let storage = hyprstream_core::storage::ModelStorage::create(
                        storage_paths.models_dir()?
                    ).await?;
                    handle_infer(&storage, &model, &prompt, max_tokens, temperature, top_p, top_k, stream, force_download).await
                        .map_err(|e| e.into())
                }
            )?;
        }

        Commands::List { branch, tag, dirty, verbose } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_list(&storage, branch, tag, dirty, verbose).await
            })?;
        }

        Commands::Inspect { model, verbose } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_info(&storage, &model, verbose).await
            })?;
        }

        Commands::Clone { repo_url, name } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_clone(&storage, &repo_url, name).await
            })?;
        }

        Commands::Push { model, remote, branch, set_upstream, force } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_push(&storage, &model, remote, branch, set_upstream, force).await
            })?;
        }

        Commands::Pull { model, remote, branch, rebase } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_pull(&storage, &model, remote, branch, rebase).await
            })?;
        }

        Commands::Merge { model, branch, ff_only, no_ff } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage_paths = hyprstream_core::storage::StoragePaths::new()?;
                let storage = hyprstream_core::storage::ModelStorage::create(
                    storage_paths.models_dir()?
                ).await?;
                handle_merge(&storage, &model, &branch, ff_only, no_ff).await
            })?;
        }
    };

    Ok(())
}