//! Hyprstream binary - ML inference server
//!
//! This binary provides the main entry point for the Hyprstream service.

// Standard library imports
use anyhow::{Context as _, Result};
use clap::Parser;
use tracing::info;

// Core application imports
use hyprstream_core::cli::commands::Commands;
use hyprstream_core::cli::handlers::handle_server;
use hyprstream_core::cli::{
    handle_branch, handle_checkout, handle_clone, handle_commit, handle_infer, handle_info,
    handle_list, handle_lora_train, handle_merge, handle_policy_apply, handle_policy_check,
    handle_policy_diff, handle_policy_edit, handle_policy_history, handle_policy_rollback,
    handle_policy_show, handle_pull, handle_push, handle_remove, handle_status,
    handle_token_create, handle_token_list, handle_token_revoke, handle_worktree_info,
    handle_worktree_list, handle_worktree_remove, AppContext, DeviceConfig, DevicePreference,
    RuntimeConfig,
};
use hyprstream_core::config::HyprConfig;
use hyprstream_core::storage::{GitRef, ModelRef};

// Registry service
use git2db::service::{LocalService, RegistryClient};
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
        TelemetryProvider::Stdout => "hyprstream=debug,opentelemetry=debug",
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
    // This runtime must stay alive to keep LocalService running
    let _registry_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to create registry runtime")?;

    let registry_client: Arc<dyn RegistryClient> = _registry_runtime
        .block_on(async {
            let models_dir = config.models_dir();
            LocalService::start(models_dir).await
        })
        .map(|client| Arc::new(client) as Arc<dyn RegistryClient>)
        .context("Failed to start registry service")?;

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
                TelemetryProvider::Stdout => "hyprstream=debug",
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
            _ => "hyprstream=debug",
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
        Commands::Branch { model, name, from } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_branch(storage, &model, &name, from).await
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

        Commands::LoraTrain {
            model,
            branch,
            adapter,
            index,
            rank,
            learning_rate,
            batch_size,
            epochs,
            config,
            training_mode,
        } => {
            let ctx = ctx.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true,
                },
                || async move {
                    let storage = ctx.storage().await?;
                    handle_lora_train(
                        storage,
                        &model,
                        branch,
                        adapter,
                        index,
                        rank,
                        learning_rate,
                        batch_size,
                        epochs,
                        config,
                        training_mode,
                    )
                    .await
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
                    // Pass None for policy_manager - CLI shows full access by default
                    // TODO: Load PolicyManager from .registry/policies/ for permission-aware list
                    handle_list(storage, None).await
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
                        WorktreeCommand::List { model } => {
                            handle_worktree_list(storage, &model).await
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
            use hyprstream_core::auth::{PolicyManager, TokenManager};
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
                            // Initialize token manager
                            let tokens_path = policies_dir.join("tokens.csv");
                            let mut token_manager = TokenManager::new(&tokens_path)
                                .await
                                .map_err(|e| anyhow::anyhow!("Failed to initialize token manager: {}", e))?;

                            match token_cmd {
                                TokenCommand::Create { user, name, expires, scope, admin } => {
                                    handle_token_create(&mut token_manager, &user, name, &expires, scope, admin).await
                                }
                                TokenCommand::List { user } => {
                                    handle_token_list(&token_manager, user).await
                                }
                                TokenCommand::Revoke { token, name, revoke_user, force } => {
                                    handle_token_revoke(&mut token_manager, token, name, revoke_user, force).await
                                }
                            }
                        }
                    }
                },
            )?;
        }

    };

    Ok(())
}
