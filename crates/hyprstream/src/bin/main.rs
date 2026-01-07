//! Hyprstream binary - ML inference server
//!
//! This binary provides the main entry point for the Hyprstream service.

// Standard library imports
use anyhow::{Context as _, Result};
use clap::Parser;
use tracing::info;

// Core application imports
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::cli::commands::{Commands, TrainingAction};
use hyprstream_core::cli::handlers::handle_server;
use hyprstream_core::cli::{
    handle_branch, handle_checkout, handle_clone, handle_commit, handle_infer, handle_info,
    handle_list, handle_merge, handle_policy_apply, handle_policy_apply_template,
    handle_policy_check, handle_policy_diff, handle_policy_edit, handle_policy_history,
    handle_policy_list_templates, handle_policy_rollback, handle_policy_show, handle_pull,
    handle_push, handle_remote_add, handle_remote_list, handle_remote_remove, handle_remote_rename,
    handle_remote_set_url, handle_remove, handle_status, handle_token_create,
    handle_training_batch, handle_training_checkpoint, handle_training_infer, handle_training_init,
    handle_worktree_add, handle_worktree_info, handle_worktree_list,
    handle_worktree_remove, load_or_generate_signing_key, AppContext, DeviceConfig, DevicePreference, RuntimeConfig,
    // Worker handlers
    handle_images_df, handle_images_list, handle_images_pull, handle_images_rm,
    handle_worker_exec, handle_worker_list, handle_worker_restart, handle_worker_rm,
    handle_worker_run, handle_worker_start, handle_worker_stats, handle_worker_status,
    handle_worker_stop,
};
use hyprstream_core::config::HyprConfig;
use hyprstream_core::storage::{GitRef, ModelRef};

// Registry and policy services - uses ZMQ-based services from hyprstream_core
use hyprstream_core::services::{
    PolicyService, PolicyZmqClient, RegistryClient, RegistryService, RegistryZmqClient, ServiceHandle,
};
// Worker service for Kata-based workload execution
use hyprstream_workers::runtime::WorkerService;
use hyprstream_workers::{ImageConfig, PoolConfig, start_event_service, EventServiceHandle};
// ZMQ context for service startup
use hyprstream_core::zmq::global_context;
use std::sync::Arc;
// Tracing imports (feature-gated)
#[cfg(feature = "otel")]
use opentelemetry::trace::TracerProvider;
#[cfg(feature = "otel")]
use opentelemetry::KeyValue;
#[cfg(feature = "otel")]
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
#[cfg(feature = "otel")]
use opentelemetry_sdk::Resource;
#[cfg(feature = "otel")]
use tracing_opentelemetry::OpenTelemetryLayer;
#[cfg(not(feature = "otel"))]
use tracing_subscriber::EnvFilter;
#[cfg(feature = "otel")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use tracing_appender::rolling::{RollingFileAppender, Rotation};

#[derive(Parser)]
#[command(
    name = "hyprstream",
    version = env!("CARGO_PKG_VERSION"),
    about = "Real-time adaptive ML inference server with dynamic sparse weight adjustments",
    long_about = None
)]
struct Cli {
    /// Path to configuration file (overrides default config locations)
    #[arg(long, global = true, env = "HYPRSTREAM_CONFIG")]
    config: Option<std::path::PathBuf>,

    /// GPU device ID to use (e.g., 0, 1). Defaults to auto-detect (device 0).
    #[arg(long, global = true, env = "HYPRSTREAM_GPU_DEVICE")]
    gpu_device: Option<usize>,

    #[command(subcommand)]
    command: Commands,
}

/// Load HyprConfig from file path or default locations
fn load_config(path: Option<&std::path::Path>) -> Result<HyprConfig> {
    match path {
        Some(p) => {
            info!("Loading config from: {}", p.display());
            HyprConfig::from_file(p)
                .with_context(|| format!("Failed to load config from {}", p.display()))
        }
        None => {
            info!("Loading config from default locations");
            match HyprConfig::load() {
                Ok(config) => Ok(config),
                Err(e) => {
                    info!("No config file found ({}), using defaults", e);
                    Ok(HyprConfig::default())
                }
            }
        }
    }
}

/// Execute a command with the appropriate runtime configuration
fn with_runtime<F, Fut>(config: RuntimeConfig, handler: F) -> Result<()>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    // Check GPU availability early if required
    if matches!(config.device.preference, DevicePreference::RequireGPU) {
        // Early GPU check - this is a simplified check
        // The actual GPU initialization happens in the engines
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err()
            && std::env::var("HIP_VISIBLE_DEVICES").is_err()
            && !std::path::Path::new("/usr/local/cuda").exists()
            && !std::path::Path::new("/opt/rocm").exists()
        {
            anyhow::bail!("GPU required but no CUDA or ROCm installation detected. Set CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES, or install CUDA/ROCm.");
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
        let rt =
            tokio::runtime::Runtime::new().context("Failed to create multi-threaded runtime")?;
        rt.block_on(handler())
    } else {
        // Single-threaded runtime for lightweight operations
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Failed to create single-threaded runtime")?;
        rt.block_on(handler())
    }
}

/// Server command handler - requires multi-threaded runtime for GPU
async fn handle_server_cmd(
    ctx: AppContext,
    flight_config: Option<hyprstream_core::cli::FlightServerConfig>,
) -> Result<()> {
    handle_server(ctx, flight_config)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))
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
fn init_telemetry(provider: TelemetryProvider) -> Result<()> {
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "hyprstream".to_string());

    // Create resource with service information
    let resource = Resource::builder()
        .with_service_name(service_name.clone())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .build();

    // Create the appropriate exporter based on provider type

    let tracer_provider = match provider {
        TelemetryProvider::Otlp => {
            let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:4317".to_string());

            info!("Using OTLP exporter for OpenTelemetry at {}", endpoint);

            let exporter = SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .build()
                .context("Failed to create OTLP exporter")?;

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
    let tracer = tracer_provider.tracer("hyprstream");

    // Setup tracing subscriber with OpenTelemetry layer
    let otel_layer = OpenTelemetryLayer::new(tracer);

    // Set log level based on provider
    let default_log_level = match provider {
        TelemetryProvider::Otlp => "hyprstream=info,opentelemetry=info",
        TelemetryProvider::Stdout => "hyprstream=warn",
    };

    let filter = EnvFilter::builder()
        .parse_lossy(std::env::var("RUST_LOG").unwrap_or_else(|_| default_log_level.to_string()));

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

fn main() -> Result<()> {
    // Set up panic handler to get better debugging information
    std::panic::set_hook(Box::new(|info| {
        eprintln!("🚨 PANIC occurred:");
        if let Some(loc) = info.location() {
            eprintln!("   Location: {}:{}", loc.file(), loc.line());
        }
        if let Some(msg) = info.payload().downcast_ref::<&str>() {
            eprintln!("   Message: {}", msg);
        }
        eprintln!("\nThis was likely caused by a threading issue or memory corruption.");
        eprintln!("Please check for unsafe RefCell usage or race conditions.");
    }));

    // Parse CLI arguments
    let cli = Cli::parse();

    // Load configuration early
    let config = load_config(cli.config.as_deref())?;

    // Validate configuration
    config
        .validate()
        .context("Configuration validation failed")?;

    // Start registry service ONCE at CLI level
    // This runtime must stay alive to keep ZMQ services running
    let _registry_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to create registry runtime")?;

    // Start ZMQ-based services: PolicyService first, then RegistryService, then optional WorkerService
    // Return both client AND service handle so we can stop it on exit
    // Also return keypair for other services (like InferenceService) to use
    use hyprstream_rpc::{SigningKey, VerifyingKey, RequestIdentity};

    // Clone worker config before moving config into the async block
    let worker_config = config.worker.clone();

    let (registry_client, _service_handle, _worker_handle, _event_handle, signing_key, verifying_key): (Arc<dyn RegistryClient>, ServiceHandle, Option<ServiceHandle>, EventServiceHandle, SigningKey, VerifyingKey) = _registry_runtime
        .block_on(async {
            let models_dir = config.models_dir();

            // Load or generate signing key (persisted to .registry/keys/signing.key)
            let keys_dir = models_dir.join(".registry").join("keys");
            let signing_key = load_or_generate_signing_key(&keys_dir).await?;
            let verifying_key = signing_key.verifying_key();

            // Start EventService FIRST (event bus must be running before services that publish events)
            // Uses XPUB/XSUB proxy pattern for efficient message distribution
            let event_handle = start_event_service(global_context())
                .context("Failed to start event service")?;

            // Initialize policy manager for authorization
            // Note: Policy templates should be applied explicitly using CLI commands:
            //   hyprstream policy apply-template local    # For local full access
            //   hyprstream policy apply-template public-inference  # For public inference
            let policies_dir = models_dir.join(".registry").join("policies");
            let policy_manager = Arc::new(
                PolicyManager::new(&policies_dir)
                    .await
                    .context("Failed to initialize policy manager")?
            );

            // Start PolicyService (runs on multi-threaded runtime where block_in_place works)
            // This service wraps PolicyManager and exposes it over ZMQ
            let _policy_handle = PolicyService::start(policy_manager, verifying_key)
                .await
                .context("Failed to start policy service")?;

            // Create policy client for other services to use
            let policy_client = PolicyZmqClient::new(signing_key.clone(), RequestIdentity::local());

            // Start the registry service with policy client (waits for socket binding)
            let service_handle = RegistryService::start(&models_dir, verifying_key, policy_client)
                .await
                .context("Failed to start registry service")?;

            // Start WorkerService if configured (optional, for Kata-based workload execution)
            let worker_handle = if let Some(ref wc) = worker_config {
                info!("Starting WorkerService for container/sandbox management");
                let handle = WorkerService::start(
                    wc.pool.clone(),
                    wc.images.clone(),
                    global_context(),
                    verifying_key,
                )
                .await
                .context("Failed to start worker service")?;
                Some(handle)
            } else {
                None
            };

            // Create client with signing credentials for service communication
            // Uses local identity (current OS user) for authorization
            let client = Arc::new(RegistryZmqClient::new(
                signing_key.clone(),
                RequestIdentity::local(),
            )) as Arc<dyn RegistryClient>;
            Ok::<_, anyhow::Error>((client, service_handle, worker_handle, event_handle, signing_key, verifying_key))
        })
        .context("Failed to initialize services")?;

    // Create application context with shared registry client
    let ctx = AppContext::with_client(config, registry_client.clone());

    // Keep the guard alive for the entire program lifetime
    // Dropping the guard stops the background logging thread
    let _log_guard: Option<tracing_appender::non_blocking::WorkerGuard>;

    #[cfg(feature = "otel")]
    {
        // Determine telemetry provider based on command
        let telemetry_provider = match &cli.command {
            Commands::Server(_) => TelemetryProvider::Otlp, // Server uses OTLP
            _ => TelemetryProvider::Stdout,                 // All CLI commands use stdout
        };

        // Initialize OpenTelemetry based on environment
        let otel_enabled = std::env::var("HYPRSTREAM_OTEL_ENABLE")
            .unwrap_or_else(|_| "false".to_string()) // Default to disabled
            .parse::<bool>()
            .unwrap_or(false);

        if otel_enabled {
            init_telemetry(telemetry_provider)?;
        } else {
            // Fallback to simple logging without OpenTelemetry
            let default_log_level = match telemetry_provider {
                TelemetryProvider::Otlp => "hyprstream=info",
                TelemetryProvider::Stdout => "hyprstream=warn",
            };

            // Check if file logging is enabled via environment variable
            if let Ok(log_dir) = std::env::var("HYPRSTREAM_LOG_DIR") {
                // Create rolling file appender (daily rotation)
                let file_appender = RollingFileAppender::new(
                    Rotation::DAILY,
                    &log_dir,
                    "hyprstream.log"
                );
                let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
                _log_guard = Some(guard);

                tracing_subscriber::fmt()
                    .with_env_filter(EnvFilter::builder().parse_lossy(
                        std::env::var("RUST_LOG").unwrap_or_else(|_| default_log_level.to_string()),
                    ))
                    .with_target(true)
                    .with_file(true)
                    .with_line_number(true)
                    .with_writer(non_blocking)
                    .init();

                info!("File logging enabled to {}/hyprstream.log", log_dir);
            } else {
                _log_guard = None;
                // Console logging only
                tracing_subscriber::fmt()
                    .with_env_filter(EnvFilter::builder().parse_lossy(
                        std::env::var("RUST_LOG").unwrap_or_else(|_| default_log_level.to_string()),
                    ))
                    .with_target(true)
                    .with_file(true)
                    .with_line_number(true)
                    .init();
            }
        }
    }

    #[cfg(not(feature = "otel"))]
    {
        // Simple logging without OpenTelemetry support
        let default_log_level = match &cli.command {
            Commands::Server(_) => "hyprstream=info",
            _ => "hyprstream=warn",
        };

        // Check if file logging is enabled via environment variable
        if let Ok(log_dir) = std::env::var("HYPRSTREAM_LOG_DIR") {
            // Create rolling file appender (daily rotation)
            let file_appender = RollingFileAppender::new(
                Rotation::DAILY,
                &log_dir,
                "hyprstream.log"
            );
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            _log_guard = Some(guard);

            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::builder().parse_lossy(
                    std::env::var("RUST_LOG").unwrap_or_else(|_| default_log_level.to_string()),
                ))
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .with_writer(non_blocking)
                .init();

            info!("File logging enabled to {}/hyprstream.log", log_dir);
        } else {
            _log_guard = None;
            // Console logging only
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::builder().parse_lossy(
                    std::env::var("RUST_LOG").unwrap_or_else(|_| default_log_level.to_string()),
                ))
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .init();
        }
    }

    info!("Hyprstream v{} starting up", env!("CARGO_PKG_VERSION"));

    // Handle commands with appropriate runtime configuration
    match cli.command {
        Commands::Server(cmd) => {
            // Load base config from files
            let mut config = load_config(cli.config.as_deref())?;

            // Merge CLI server arguments into config using apply_to_builder
            config.server = cmd.server.apply_to_builder(config.server.to_builder())
                .build();

            // Apply GPU device setting from CLI if specified
            if let Some(gpu_device) = cli.gpu_device {
                config.runtime.gpu_device_id = Some(gpu_device);
                info!("Using GPU device {} from CLI", gpu_device);
            }

            // Create Flight SQL server config if dataset is specified
            let flight_config = cmd.dataset.as_ref().map(|dataset| {
                hyprstream_core::cli::FlightServerConfig {
                    dataset: Some(dataset.clone()),
                    port: cmd.flight_port,
                    host: cmd.flight_host.clone(),
                }
            });

            // Create context with merged config and shared registry client
            let ctx = AppContext::with_client(config, registry_client.clone());

            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true,
                },
                || handle_server_cmd(ctx, flight_config),
            )?
        }
        // Phase 1: Git-style commands
        Commands::Branch { model, name, from, policy } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_branch(storage, &model, &name, from, policy).await
                },
            )?;
        }

        Commands::Checkout {
            model,
            git_ref,
            create_branch,
            force,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;

                    // Build ModelRef from separate components
                    let git_ref_parsed = match git_ref {
                        Some(ref r) => GitRef::parse(r)?,
                        None => GitRef::DefaultBranch,
                    };
                    let model_ref = ModelRef::with_ref(model, git_ref_parsed);

                    handle_checkout(storage, &model_ref.to_string(), create_branch, force).await
                },
            )?;
        }

        Commands::Status { model, verbose } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_status(storage, model, verbose).await
                },
            )?;
        }

        Commands::Commit {
            model,
            message,
            all,
            all_untracked,
            amend,
            author,
            author_name,
            author_email,
            allow_empty,
            dry_run,
            verbose,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_commit(
                        storage,
                        &model,
                        &message,
                        all,
                        all_untracked,
                        amend,
                        author,
                        author_name,
                        author_email,
                        allow_empty,
                        dry_run,
                        verbose,
                    )
                    .await
                },
            )?;
        }

        Commands::Training(cmd) => {
            let ctx = ctx.clone();
            let signing_key = signing_key.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    match cmd.action {
                        TrainingAction::Init {
                            model,
                            branch,
                            adapter,
                            index,
                            rank,
                            alpha,
                            mode,
                            learning_rate,
                        } => {
                            handle_training_init(
                                storage,
                                &model,
                                branch,
                                adapter,
                                index,
                                rank,
                                alpha,
                                &mode,
                                learning_rate,
                            )
                            .await
                        }
                        TrainingAction::Infer {
                            model,
                            prompt,
                            image,
                            max_tokens,
                            temperature,
                            top_p,
                            top_k,
                            repeat_penalty,
                            stream,
                            max_context,
                            kv_quant,
                        } => {
                            // Read prompt from stdin if not provided
                            let prompt_text = match prompt {
                                Some(p) => p,
                                None => {
                                    use std::io::Read;
                                    let mut buffer = String::new();
                                    std::io::stdin().read_to_string(&mut buffer)?;
                                    buffer.trim().to_string()
                                }
                            };
                            handle_training_infer(
                                storage,
                                &model,
                                &prompt_text,
                                image,
                                max_tokens,
                                temperature,
                                top_p,
                                top_k,
                                repeat_penalty,
                                stream,
                                max_context,
                                kv_quant,
                                signing_key,
                                verifying_key,
                            )
                            .await
                        }
                        TrainingAction::Batch {
                            model,
                            input,
                            input_dir,
                            pattern,
                            format,
                            max_tokens,
                            chunk_size,
                            skip,
                            limit,
                            progress_interval,
                            checkpoint_interval,
                            test_set,
                            max_context,
                            kv_quant,
                        } => {
                            handle_training_batch(
                                storage,
                                &model,
                                input,
                                input_dir,
                                &pattern,
                                &format,
                                max_tokens,
                                chunk_size,
                                skip,
                                limit,
                                progress_interval,
                                checkpoint_interval,
                                test_set,
                                max_context,
                                kv_quant,
                                signing_key,
                                verifying_key,
                            )
                            .await
                        }
                        TrainingAction::Checkpoint {
                            model,
                            message,
                            push,
                            remote,
                        } => {
                            handle_training_checkpoint(storage, &model, message, push, &remote)
                                .await
                        }
                    }
                },
            )?;
        }

        Commands::Infer {
            model,
            prompt,
            image,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            seed,
            stream,
            force_download,
            max_context,
            kv_quant,
        } => {
            // Read prompt from stdin if not provided via --prompt
            let prompt = match prompt {
                Some(p) => p,
                None => {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin().read_to_string(&mut buffer)?;
                    buffer.trim().to_string()
                }
            };

            let ctx = ctx.clone();
            let signing_key = signing_key.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_infer(
                        storage,
                        &model,
                        &prompt,
                        image,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        repeat_penalty,
                        seed,
                        stream,
                        force_download,
                        max_context,
                        kv_quant,
                        signing_key,
                        verifying_key,
                    )
                    .await
                },
            )?;
        }

        Commands::List { .. } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;

                    // Load PolicyManager for permission-aware capability display
                    let registry_path = storage.get_models_dir().join(".registry");
                    let policies_dir = registry_path.join("policies");
                    let policy_manager = PolicyManager::new(&policies_dir)
                        .await
                        .ok(); // Gracefully fall back to None if policies don't exist

                    handle_list(storage, policy_manager.as_ref()).await
                },
            )?;
        }

        Commands::Inspect {
            model,
            verbose,
            adapters_only,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    match handle_info(storage, &model, verbose, adapters_only).await {
                        Ok(()) => Ok(()),
                        Err(e) => {
                            // Print error but continue to show what we can
                            eprintln!("Warning: Some operations failed: {}", e);
                            Ok(())
                        }
                    }
                },
            )?;
        }

        Commands::Clone {
            repo_url,
            name,
            branch,
            depth,
            full,
            quiet,
            verbose,
            policy,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_clone(
                        storage,
                        &repo_url,
                        name,
                        branch,
                        depth,
                        full,
                        quiet,
                        verbose,
                        policy,
                    ).await
                },
            )?;
        }

        Commands::Push {
            model,
            remote,
            branch,
            set_upstream,
            force,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_push(storage, &model, remote, branch, set_upstream, force).await
                },
            )?;
        }

        Commands::Pull {
            model,
            remote,
            branch,
            rebase,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_pull(storage, &model, remote, branch, rebase).await
                },
            )?;
        }

        Commands::Merge {
            target,
            source,
            ff,
            no_ff,
            ff_only,
            no_commit,
            squash,
            message,
            abort,
            continue_merge,
            quit,
            no_stat,
            quiet,
            verbose,
            strategy,
            strategy_option,
            allow_unrelated_histories,
            no_verify,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    use hyprstream_core::cli::git_handlers::MergeOptions;

                    let storage = ctx.storage().await?;
                    let options = MergeOptions {
                        ff,
                        no_ff,
                        ff_only,
                        no_commit,
                        squash,
                        message: message.clone(),
                        abort,
                        continue_merge,
                        quit,
                        no_stat,
                        quiet,
                        verbose,
                        strategy: strategy.clone(),
                        strategy_option: strategy_option.clone(),
                        allow_unrelated_histories,
                        no_verify,
                    };
                    handle_merge(storage, &target, &source, options).await
                },
            )?;
        }
        Commands::Remove {
            model,
            force,
            registry_only,
            files_only,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_remove(storage, &model, force, registry_only, files_only).await
                },
            )?;
        }

        Commands::Worktree { command } => {
            use hyprstream_core::cli::commands::WorktreeCommand;

            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    match command {
                        WorktreeCommand::Add { model, branch, policy } => {
                            handle_worktree_add(storage, &model, &branch, policy).await
                        }
                        WorktreeCommand::List { model, all } => {
                            handle_worktree_list(storage, &model, all).await
                        }
                        WorktreeCommand::Info { model, branch } => {
                            handle_worktree_info(storage, &model, &branch).await
                        }
                        WorktreeCommand::Remove { model, branch, force } => {
                            handle_worktree_remove(storage, &model, &branch, force).await
                        }
                    }
                },
            )?;
        }

        Commands::Flight(args) => {
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    use hyprstream_flight::client::{
                        format_batches_as_csv, format_batches_as_json, format_batches_as_table,
                        FlightClient,
                    };

                    let mut client = FlightClient::connect(&args.addr)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to connect: {}", e))?;

                    if let Some(sql) = args.query {
                        let batches = client
                            .query(&sql)
                            .await
                            .map_err(|e| anyhow::anyhow!("Query failed: {}", e))?;

                        let output = match args.format.as_str() {
                            "csv" => format_batches_as_csv(&batches),
                            "json" => format_batches_as_json(&batches)
                                .map_err(|e| anyhow::anyhow!("JSON format error: {}", e))?,
                            _ => format_batches_as_table(&batches),
                        };
                        print!("{}", output);
                    } else {
                        // Interactive REPL mode - TODO: implement readline-based REPL
                        println!("Connected to Flight SQL server at {}", args.addr);
                        println!("Interactive mode not yet implemented. Use --query to execute SQL.");
                    }
                    Ok(())
                },
            )?;
        }

        Commands::Policy { command } => {
            use hyprstream_core::cli::commands::{PolicyCommand, TokenCommand};

            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    // Get the policies directory from the registry
                    let storage = ctx.storage().await?;
                    let registry_path = storage.get_models_dir().join(".registry");
                    let policies_dir = registry_path.join("policies");

                    // Initialize policy manager
                    let policy_manager = PolicyManager::new(&policies_dir)
                        .await
                        .map_err(|e| anyhow::anyhow!("Failed to initialize policy manager: {}", e))?;

                    match command {
                        PolicyCommand::Show { raw } => {
                            handle_policy_show(&policy_manager, raw).await
                        }
                        PolicyCommand::History { count, oneline } => {
                            handle_policy_history(&policy_manager, count, oneline).await
                        }
                        PolicyCommand::Edit => {
                            handle_policy_edit(&policy_manager).await
                        }
                        PolicyCommand::Diff { against } => {
                            handle_policy_diff(&policy_manager, against).await
                        }
                        PolicyCommand::Apply { dry_run, message } => {
                            handle_policy_apply(&policy_manager, dry_run, message).await
                        }
                        PolicyCommand::Rollback { git_ref, dry_run } => {
                            handle_policy_rollback(&policy_manager, &git_ref, dry_run).await
                        }
                        PolicyCommand::Check { user, resource, action } => {
                            handle_policy_check(&policy_manager, &user, &resource, &action).await
                        }
                        PolicyCommand::Token { command: token_cmd } => {
                            match token_cmd {
                                TokenCommand::Create { user, name, expires, scope, admin } => {
                                    // Load signing key for JWT token generation
                                    let keys_dir = registry_path.join("keys");
                                    let signing_key = load_or_generate_signing_key(&keys_dir).await?;
                                    handle_token_create(&signing_key, &user, name, &expires, scope, admin).await
                                }
                                TokenCommand::List { user: _ } => {
                                    // JWT tokens are stateless - cannot list
                                    println!("JWT tokens are stateless and cannot be listed.");
                                    println!();
                                    println!("Tokens are validated by signature and expiry time.");
                                    println!("To see who has access, review the policy:");
                                    println!("  hyprstream policy show");
                                    Ok(())
                                }
                                TokenCommand::Revoke { token: _, name: _, revoke_user: _, force: _ } => {
                                    // JWT tokens cannot be revoked without a blocklist
                                    println!("JWT tokens cannot be revoked individually.");
                                    println!();
                                    println!("Options for token invalidation:");
                                    println!("  1. Use short expiry times (e.g., --expires 1d)");
                                    println!("  2. Regenerate the signing key to invalidate all tokens:");
                                    println!("     rm ~/.local/share/hyprstream/.registry/keys/signing.key");
                                    println!("  3. Remove user permissions via policy:");
                                    println!("     hyprstream policy edit");
                                    println!();
                                    println!("A token blocklist feature may be added in the future.");
                                    Ok(())
                                }
                            }
                        }
                        PolicyCommand::ApplyTemplate { template, dry_run } => {
                            handle_policy_apply_template(&policy_manager, &template, dry_run).await
                        }
                        PolicyCommand::ListTemplates => {
                            handle_policy_list_templates().await
                        }
                    }
                },
            )?;
        }

        Commands::Remote { command } => {
            use hyprstream_core::cli::commands::RemoteCommand;

            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    match command {
                        RemoteCommand::Add { model, name, url } => {
                            handle_remote_add(storage, &model, &name, &url).await
                        }
                        RemoteCommand::List { model, verbose } => {
                            handle_remote_list(storage, &model, verbose).await
                        }
                        RemoteCommand::Remove { model, name } => {
                            handle_remote_remove(storage, &model, &name).await
                        }
                        RemoteCommand::SetUrl { model, name, url } => {
                            handle_remote_set_url(storage, &model, &name, &url).await
                        }
                        RemoteCommand::Rename { model, old_name, new_name } => {
                            handle_remote_rename(storage, &model, &old_name, &new_name).await
                        }
                    }
                },
            )?;
        }

        Commands::Worker(cmd) => {
            use hyprstream_core::cli::commands::{WorkerAction, ImageCommand};
            use hyprstream_core::services::WorkerClient;

            let signing_key = signing_key.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    // Start WorkerService for CLI commands
                    // Use minimal config without warm pool (no VMs started immediately)
                    info!("Starting WorkerService for CLI worker commands");

                    // Use user-accessible paths for CLI (not /var/lib which requires root)
                    let data_dir = dirs::data_local_dir()
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                        .join("hyprstream");
                    let runtime_dir = dirs::runtime_dir()
                        .unwrap_or_else(|| std::env::temp_dir())
                        .join("hyprstream");

                    // Kata VM files location from KATA_BOOT_PATH env var
                    let kata_boot_path = std::env::var("KATA_BOOT_PATH")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| std::path::PathBuf::from("/opt/kata/share/kata-containers"));

                    let pool_config = PoolConfig {
                        warm_pool_size: 0, // Don't pre-warm VMs for CLI
                        runtime_dir: runtime_dir.join("sandboxes"),
                        kernel_path: kata_boot_path.join("vmlinux.container"),
                        vm_image: kata_boot_path.join("kata-containers.img"),
                        cloud_init_dir: data_dir.join("cloud-init"),
                        ..PoolConfig::default()
                    };
                    let image_config = ImageConfig {
                        blobs_dir: data_dir.join("images/blobs"),
                        bootstrap_dir: data_dir.join("images/bootstrap"),
                        refs_dir: data_dir.join("images/refs"),
                        cache_dir: data_dir.join("images/cache"),
                        runtime_dir: runtime_dir.join("nydus"),
                        ..ImageConfig::default()
                    };

                    tracing::debug!(
                        data_dir = %data_dir.display(),
                        runtime_dir = %runtime_dir.display(),
                        "Using CLI-friendly paths for worker service"
                    );

                    let _worker_handle = WorkerService::start(
                        pool_config,
                        image_config,
                        global_context(),
                        verifying_key,
                    )
                    .await
                    .context("Failed to start worker service")?;

                    // Create WorkerClient for ZMQ communication with WorkerService
                    let worker_client = WorkerClient::new(signing_key, RequestIdentity::local());

                    match cmd.action {
                        WorkerAction::List { sandbox, containers, sandboxes, state, verbose } => {
                            handle_worker_list(&worker_client, sandbox, containers, sandboxes, state, verbose).await
                        }
                        WorkerAction::Run { image, command, sandbox, name, env, workdir, detach, rm } => {
                            handle_worker_run(&worker_client, &image, command, sandbox, name, env, workdir, detach, rm).await
                        }
                        WorkerAction::Stop { id, timeout, force } => {
                            handle_worker_stop(&worker_client, &id, timeout, force).await
                        }
                        WorkerAction::Start { container_id } => {
                            handle_worker_start(&worker_client, &container_id).await
                        }
                        WorkerAction::Restart { id, timeout } => {
                            handle_worker_restart(&worker_client, &id, timeout).await
                        }
                        WorkerAction::Rm { ids, force, volumes: _ } => {
                            handle_worker_rm(&worker_client, ids, force).await
                        }
                        WorkerAction::Status { id, verbose } => {
                            handle_worker_status(&worker_client, &id, verbose).await
                        }
                        WorkerAction::Stats { ids, stream: _, no_header } => {
                            handle_worker_stats(&worker_client, ids, no_header).await
                        }
                        WorkerAction::Exec { container_id, command, timeout } => {
                            handle_worker_exec(&worker_client, &container_id, command, timeout).await
                        }
                        WorkerAction::Images(img_cmd) => {
                            match img_cmd {
                                ImageCommand::List { verbose, filter: _ } => {
                                    handle_images_list(&worker_client, verbose).await
                                }
                                ImageCommand::Pull { image, username, password } => {
                                    handle_images_pull(&worker_client, &image, username, password).await
                                }
                                ImageCommand::Rm { images, force } => {
                                    handle_images_rm(&worker_client, images, force).await
                                }
                                ImageCommand::Df => {
                                    handle_images_df(&worker_client).await
                                }
                            }
                        }
                    }
                },
            )?;
        }

    };

    // Gracefully stop services before exiting (reverse order of startup)
    _registry_runtime.block_on(async {
        _service_handle.stop().await;
    });

    // Stop EventService (uses blocking stop, not async)
    if let Err(e) = _event_handle.stop() {
        tracing::warn!("Failed to stop event service: {}", e);
    }

    Ok(())
}
