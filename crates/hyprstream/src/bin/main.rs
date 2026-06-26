//! Hyprstream binary - ML inference server
//!
//! This binary provides the main entry point for the Hyprstream service.
// CLI binary intentionally prints to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

// Standard library imports
use anyhow::{Context as _, Result};
use clap::{Arg, Args as ClapArgs, Command as ClapCommand, FromArgMatches, Subcommand as ClapSubcommand};
use tracing::info;

// Core application imports
use hyprstream_core::cli::commands::{
    ExecutionMode, FlightArgs, ImageCommand, ServiceAction, TrainingAction, TuiAction, WorkerAction,
};
use hyprstream_core::cli::quick::{QuickCommand, RemoteQuickCommand, WorktreeQuickCommand};
use hyprstream_core::cli::schema_cli;
use hyprstream_core::cli::{
    handle_branch, handle_checkout, handle_clone, handle_infer, handle_info, handle_list,
    handle_load, handle_notify_command, parse_filters, parse_status_filter, handle_policy_apply, handle_policy_apply_template, handle_policy_check,
    handle_policy_diff, handle_policy_edit, handle_policy_history, handle_policy_list_templates,
    handle_policy_rollback, handle_policy_show, handle_policy_role_add, handle_policy_role_remove,
    handle_policy_role_list, handle_pull, handle_remote_add, handle_remote_list,
    handle_remote_remove, handle_remote_rename, handle_remote_set_url, handle_remove,
    handle_sign_challenge, handle_status, handle_token_create, handle_unload, handle_training_batch,
    handle_training_checkpoint, handle_training_infer, handle_training_init, handle_worktree_add,
    handle_worktree_info, handle_worktree_list, handle_worktree_remove,
    handle_user_list, handle_user_register, handle_user_remove,
    handle_user_keys_list, handle_user_keys_import, handle_user_keys_remove,
    load_or_generate_signing_key, AppContext, DeviceConfig, DevicePreference, RuntimeConfig,
    // Worker handlers
    handle_images_df, handle_images_list, handle_images_pull, handle_images_rm,
    handle_worker_exec, handle_worker_list, handle_worker_restart, handle_worker_rm,
    handle_worker_run, handle_worker_start, handle_worker_stats, handle_worker_status,
    handle_worker_terminal, handle_worker_stop,
    // Service handlers
    handle_service_install,
    handle_service_start, handle_service_status,
    handle_service_stop, handle_service_uninstall,
};
use hyprstream_core::cli::commands::{PolicyCommand, RoleCommand, TokenCommand, UserCommand, UserKeysCommand};

#[cfg(feature = "experimental")]
use hyprstream_core::cli::{handle_commit, handle_merge, handle_promote, handle_push, MergeOptions};
use hyprstream_core::config::HyprConfig;
use hyprstream_core::storage::{GitRef, ModelRef};

// Registry and policy services
use hyprstream_core::services::{PolicyClient, RegistryClient};
// Worker service for Kata-based workload execution
use hyprstream_workers::runtime::WorkerService;
use hyprstream_workers::{ImageConfig, PoolConfig};
use std::sync::Arc;
// Unified service manager API
use hyprstream_service::{get_factory, InprocManager, ServiceContext, ServiceManager};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::{SigningKey, VerifyingKey};

fn supports_tui() -> bool {
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
        return false;
    }
    match std::env::var("TERM").as_deref() {
        Ok("dumb") | Ok("") | Err(_) => return false,
        _ => {}
    }
    #[cfg(unix)]
    {
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let ok = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) };
        if ok == 0 {
            return ws.ws_col >= 60 && ws.ws_row >= 20;
        }
    }
    true
}
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

// ─────────────────────────────────────────────────────────────────────────────
// CLI builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build the CLI using clap builder API with hybrid derive-based subcommands.
fn build_cli() -> ClapCommand {
    let mut app = ClapCommand::new("hyprstream")
        .version(env!("BUILD_VERSION"))
        .about("Real-time adaptive ML inference server with dynamic sparse weight adjustments")
        .arg(
            Arg::new("config")
                .long("config")
                .global(true)
                .env("HYPRSTREAM_CONFIG")
                .value_parser(clap::value_parser!(std::path::PathBuf))
                .help("Path to configuration file (overrides default config locations)"),
        )
        .arg(
            Arg::new("gpu_device")
                .long("gpu-device")
                .global(true)
                .env("HYPRSTREAM_GPU_DEVICE")
                .value_parser(clap::value_parser!(usize))
                .help("GPU device ID to use (e.g., 0, 1). Defaults to auto-detect."),
        );

    // Schema-driven: hyprstream tool <service> <method>
    app = app.subcommand(schema_cli::build_tool_command());

    // Quick workflows (derive-based subcommands)
    app = app.subcommand(
        <QuickCommand as ClapSubcommand>::augment_subcommands(
            ClapCommand::new("quick")
                .about("Quick workflows \u{2014} multi-step convenience commands"),
        ),
    );

    // Service management (derive-based subcommands)
    app = app.subcommand(
        <ServiceAction as ClapSubcommand>::augment_subcommands(
            ClapCommand::new("service").about("Service management and lifecycle commands"),
        ),
    );

    // TUI display server (derive-based subcommands)
    app = app.subcommand(
        <TuiAction as ClapSubcommand>::augment_subcommands(
            ClapCommand::new("tui").about("TUI display server \u{2014} terminal multiplexer with session persistence"),
        ),
    );

    // Flight SQL client (derive-based args)
    app = app.subcommand(
        <FlightArgs as ClapArgs>::augment_args(
            ClapCommand::new("flight").about("Flight SQL client to query datasets"),
        ),
    );

    // User management (derive-based subcommands)
    app = app.subcommand(
        <UserCommand as ClapSubcommand>::augment_subcommands(
            ClapCommand::new("user").about("Manage local user credentials"),
        ),
    );

    // Interactive setup wizard
    app = app.subcommand(
        ClapCommand::new("wizard")
            .about("Interactive setup wizard — configure policies, users, and API tokens")
            .arg(
                Arg::new("non_interactive")
                    .long("non-interactive")
                    .short('y')
                    .action(clap::ArgAction::SetTrue)
                    .help("Accept defaults without prompting"),
            )
            .arg(
                Arg::new("start")
                    .long("start")
                    .action(clap::ArgAction::SetTrue)
                    .help("Start services after setup"),
            )
            .arg(
                Arg::new("tui")
                    .long("tui")
                    .action(clap::ArgAction::SetTrue)
                    .help("Force TUI wizard (auto-detected when terminal supports it)"),
            )
            .arg(
                Arg::new("bootstrap_only")
                    .long("bootstrap-only")
                    .action(clap::ArgAction::SetTrue)
                    .help("Run only the trust-root bootstrap (phase 1); skip binary install, policy templates, user creation, token mint, and systemd"),
            )
            .arg(
                Arg::new("enable_federation")
                    .long("enable-federation")
                    .action(clap::ArgAction::SetTrue)
                    .help("Apply the federation-open policy template — accept third-party apps AND remote peer servers from any HTTPS origin (atproto-style federation). Default is disabled; under -y this flag is the only way to enable it."),
            )
            .arg(
                Arg::new("initial_user_role")
                    .long("initial-user-role")
                    .value_name("ROLE")
                    .default_value("admin")
                    .help("Role to assign the local user under --non-interactive (admin|operator|trainer|viewer). Default is admin; use operator/viewer in tests for least-privilege."),
            ),
    );

    // Sign challenge (OAuth device flow and auth code flow)
    app = app.subcommand(
        ClapCommand::new("sign-challenge")
            .about("Sign an Ed25519 challenge for OAuth device or auth code flow")
            .arg(
                Arg::new("user_code")
                    .index(1)
                    .required(false)
                    .help("User code from device flow (e.g., ABCD-EFGH)"),
            )
            .arg(
                Arg::new("nonce")
                    .long("nonce")
                    .required(false)
                    .allow_hyphen_values(true)
                    .help("Nonce from the browser authorization challenge form"),
            )
            .arg(
                Arg::new("code_challenge")
                    .long("code-challenge")
                    .required(false)
                    .allow_hyphen_values(true)
                    .help("PKCE code_challenge from the browser authorization URL"),
            )
            .arg(
                Arg::new("server")
                    .long("server")
                    .required(false)
                    .help("OAuth server URL (default: from config or http://localhost:6791)"),
            ),
    );

    // Self-update
    app = app.subcommand(
        ClapCommand::new("update")
            .about("Check for and install updates")
            .arg(
                Arg::new("cleanup")
                    .long("cleanup")
                    .action(clap::ArgAction::SetTrue)
                    .help("Remove old version worktrees"),
            ),
    );

    app
}

// ─────────────────────────────────────────────────────────────────────────────
// Config & runtime helpers
// ─────────────────────────────────────────────────────────────────────────────

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
    if matches!(config.device.preference, DevicePreference::RequireGPU)
        && std::env::var("CUDA_VISIBLE_DEVICES").is_err()
            && std::env::var("HIP_VISIBLE_DEVICES").is_err()
            && !std::path::Path::new("/usr/local/cuda").exists()
            && !std::path::Path::new("/opt/rocm").exists()
        {
            anyhow::bail!("GPU required but no CUDA or ROCm installation detected. Set CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES, or install CUDA/ROCm.");
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
        let rt =
            tokio::runtime::Runtime::new().context("Failed to create multi-threaded runtime")?;
        rt.block_on(handler())
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("Failed to create single-threaded runtime")?;
        rt.block_on(handler())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry
// ─────────────────────────────────────────────────────────────────────────────

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
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "hyprstream".to_owned());

    let resource = Resource::builder()
        .with_service_name(service_name.clone())
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .build();

    let tracer_provider = match provider {
        TelemetryProvider::Otlp => {
            let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:4317".to_owned());

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

    let tracer = tracer_provider.tracer("hyprstream");
    let otel_layer = OpenTelemetryLayer::new(tracer);

    let default_log_level = match provider {
        TelemetryProvider::Otlp => "hyprstream=info,opentelemetry=info",
        TelemetryProvider::Stdout => "hyprstream=warn",
    };

    let filter = EnvFilter::builder()
        .parse_lossy(std::env::var("RUST_LOG").unwrap_or_else(|_| default_log_level.to_owned()));

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

// ─────────────────────────────────────────────────────────────────────────────
// Quick command handler
// ─────────────────────────────────────────────────────────────────────────────

/// Handle all `quick` subcommands — orchestrated multi-step workflows.
fn handle_quick_command(
    cmd: QuickCommand,
    ctx: AppContext,
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
) -> Result<()> {
    match cmd {
        QuickCommand::Infer {
            model,
            prompt,
            image,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            seed,
            sync,
        } => {
            // Read prompt from stdin if not provided via --prompt
            let prompt = match prompt {
                Some(p) => p,
                None => {
                    use std::io::Read;
                    let mut buffer = String::new();
                    std::io::stdin().read_to_string(&mut buffer)?;
                    buffer.trim().to_owned()
                }
            };

            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_gpu(),
                    multi_threaded: true,
                },
                || async move {
                    handle_infer(
                        &model,
                        &prompt,
                        image,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        repeat_penalty,
                        seed,
                        sync,
                        signing_key,
                    )
                    .await
                },
            )
        }

        QuickCommand::Clone {
            repo_url,
            name,
            branch,
            depth,
            full,
            quiet,
            verbose,
            policy,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_clone(
                    ctx.registry(),
                    &repo_url,
                    name,
                    branch,
                    depth,
                    full,
                    quiet,
                    verbose,
                    policy,
                )
                .await
            },
        ),

        QuickCommand::List { filter, status } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                let keys_dir = ctx.models_dir().join(".registry").join("keys");
                let signing_key = load_or_generate_signing_key(&keys_dir).await?;
                let model_server_vk = resolve_service_vk("model")
                    .ok_or_else(|| anyhow::anyhow!("Cannot resolve model service pubkey. Run 'hyprstream wizard -y' to generate bootstrap credentials."))?;
                let model_client = hyprstream_core::services::generated::model_client::ModelClient::for_service(
                    signing_key,
                    model_server_vk,
                    None,
                )?;
                let filters = parse_filters(&filter)?;
                let status_filter = parse_status_filter(&status)?;
                handle_list(ctx.registry(), model_client, &filters, &status_filter).await
            },
        ),

        QuickCommand::Info {
            model,
            verbose,
            adapters_only,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                match handle_info(ctx.registry(), &model, verbose, adapters_only).await {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        eprintln!("Warning: Some operations failed: {}", e);
                        Ok(())
                    }
                }
            },
        ),

        QuickCommand::Status { model, verbose } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move { handle_status(ctx.registry(), model, verbose).await },
        ),

        QuickCommand::Branch {
            model,
            name,
            from,
            policy,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_branch(ctx.registry(), &model, &name, from, policy).await
            },
        ),

        QuickCommand::Checkout {
            model,
            git_ref,
            create_branch,
            force,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                let git_ref_parsed = match git_ref {
                    Some(ref r) => GitRef::parse(r)?,
                    None => GitRef::DefaultBranch,
                };
                let model_ref = ModelRef::with_ref(model, git_ref_parsed);
                handle_checkout(
                    ctx.registry(),
                    &model_ref.to_string(),
                    create_branch,
                    force,
                )
                .await
            },
        ),

        QuickCommand::Pull {
            model,
            remote,
            branch,
            rebase,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_pull(ctx.registry(), &model, remote, branch, rebase).await
            },
        ),

        QuickCommand::Remove {
            model,
            force,
            registry_only,
            files_only,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_remove(
                    ctx.registry(),
                    &model,
                    force,
                    registry_only,
                    files_only,
                )
                .await
            },
        ),

        QuickCommand::Load {
            model,
            max_context,
            kv_quant,
            wait,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_gpu(),
                multi_threaded: true,
            },
            || async move {
                handle_load(&model, max_context, kv_quant.into(), wait, signing_key).await
            },
        ),

        QuickCommand::Unload { model } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move { handle_unload(&model, signing_key).await },
        ),

        QuickCommand::Notify { command } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_notify_command(command, signing_key).await
            },
        ),

        QuickCommand::Worktree { command } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                match command {
                    WorktreeQuickCommand::Add {
                        model,
                        branch,
                        policy,
                    } => {
                        handle_worktree_add(ctx.registry(), &model, &branch, policy).await
                    }
                    WorktreeQuickCommand::List { model, all } => {
                        handle_worktree_list(ctx.registry(), &model, all).await
                    }
                    WorktreeQuickCommand::Info { model, branch } => {
                        handle_worktree_info(ctx.registry(), &model, &branch).await
                    }
                    WorktreeQuickCommand::Remove {
                        model,
                        branch,
                        force,
                    } => {
                        handle_worktree_remove(ctx.registry(), &model, &branch, force)
                            .await
                    }
                }
            },
        ),

        #[cfg(feature = "experimental")]
        QuickCommand::Commit {
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
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_commit(
                    ctx.registry(),
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
        ),

        #[cfg(feature = "experimental")]
        QuickCommand::Promote {
            model,
            branch,
            author_name,
            author_email,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_promote(
                    ctx.registry(),
                    &model,
                    &branch,
                    author_name,
                    author_email,
                )
                .await
            },
        ),

        #[cfg(feature = "experimental")]
        QuickCommand::Push {
            model,
            remote,
            branch,
            set_upstream,
            force,
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                handle_push(
                    ctx.registry(),
                    &model,
                    remote,
                    branch,
                    set_upstream,
                    force,
                )
                .await
            },
        ),

        #[cfg(feature = "experimental")]
        QuickCommand::Merge {
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
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
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
                handle_merge(ctx.registry(), &target, &source, options).await
            },
        ),

        QuickCommand::Training(training_cmd) => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_gpu(),
                multi_threaded: true,
            },
            || async move {
                let registry = ctx.registry();
                match training_cmd.action {
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
                            registry,
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
                        sync,
                        max_context,
                        kv_quant,
                    } => {
                        let prompt_text = match prompt {
                            Some(p) => p,
                            None => {
                                use std::io::Read;
                                let mut buffer = String::new();
                                std::io::stdin().read_to_string(&mut buffer)?;
                                buffer.trim().to_owned()
                            }
                        };
                        handle_training_infer(
                            registry,
                            &model,
                            &prompt_text,
                            image,
                            max_tokens,
                            temperature,
                            top_p,
                            top_k,
                            repeat_penalty,
                            sync,
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
                            registry,
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
                        handle_training_checkpoint(
                            registry,
                            &model,
                            message,
                            push,
                            &remote,
                        )
                        .await
                    }
                }
            },
        ),

        QuickCommand::Policy { command } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                let registry_path = ctx.models_dir().join(".registry");
                let policies_dir = registry_path.join("policies");
                let keys_dir = registry_path.join("keys");

                // Load signing key for RPC authentication
                let signing_key = load_or_generate_signing_key(&keys_dir).await?;

                // Ensure policies directory exists for the editor path
                tokio::fs::create_dir_all(&policies_dir).await?;
                let policy_csv_path = policies_dir.join("policy.csv");

                match command {
                    PolicyCommand::Show { raw } => {
                        handle_policy_show(&signing_key, raw).await
                    }
                    PolicyCommand::History { count, oneline } => {
                        handle_policy_history(&signing_key, count, oneline).await
                    }
                    PolicyCommand::Edit => handle_policy_edit(&signing_key, &policy_csv_path).await,
                    PolicyCommand::Diff { against } => {
                        handle_policy_diff(&signing_key, against).await
                    }
                    PolicyCommand::Apply { dry_run, message } => {
                        handle_policy_apply(&signing_key, dry_run, message).await
                    }
                    PolicyCommand::Rollback { git_ref, dry_run } => {
                        handle_policy_rollback(&signing_key, &git_ref, dry_run).await
                    }
                    PolicyCommand::Check {
                        user,
                        resource,
                        action,
                    } => {
                        handle_policy_check(&signing_key, &user, &resource, &action).await
                    }
                    PolicyCommand::Token {
                        command: token_cmd,
                    } => match token_cmd {
                        TokenCommand::Create {
                            user,
                            name,
                            expires,
                            scope,
                        } => {
                            handle_token_create(
                                &signing_key,
                                &user,
                                name,
                                &expires,
                                scope,
                            )
                            .await
                        }
                        TokenCommand::List { user: _ } => {
                            println!("JWT tokens are stateless and cannot be listed.");
                            println!();
                            println!("Tokens are validated by signature and expiry time.");
                            println!("To see who has access, review the policy:");
                            println!("  hyprstream quick policy show");
                            Ok(())
                        }
                        TokenCommand::Revoke {
                            token: _,
                            name: _,
                            revoke_user: _,
                            force: _,
                        } => {
                            println!("JWT tokens cannot be revoked individually.");
                            println!();
                            println!("Options for token invalidation:");
                            println!("  1. Use short expiry times (e.g., --expires 1d)");
                            println!(
                                "  2. Regenerate the signing key to invalidate all tokens:"
                            );
                            println!(
                                "     rm ~/.local/share/hyprstream/.registry/keys/signing.key"
                            );
                            println!("  3. Remove user permissions via policy:");
                            println!("     hyprstream quick policy edit");
                            println!();
                            println!(
                                "A token blocklist feature may be added in the future."
                            );
                            Ok(())
                        }
                    },
                    PolicyCommand::ApplyTemplate { template, dry_run } => {
                        handle_policy_apply_template(&signing_key, &template, dry_run).await
                    }
                    PolicyCommand::ListTemplates => handle_policy_list_templates().await,
                    PolicyCommand::Role { command } => match command {
                        RoleCommand::Add { user, role, dry_run } => {
                            handle_policy_role_add(&signing_key, &user, &role, dry_run).await
                        }
                        RoleCommand::Remove { user, role, force } => {
                            handle_policy_role_remove(&signing_key, &user, &role, force).await
                        }
                        RoleCommand::List { user, role } => {
                            handle_policy_role_list(&signing_key, user.as_deref(), role.as_deref()).await
                        }
                    },
                }
            },
        ),

        QuickCommand::Worker { command: action } => {
            let signing_key = signing_key.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    let worker_already_running = ctx
                        .config()
                        .services
                        .startup
                        .contains(&"worker".to_owned());

                    if worker_already_running {
                        info!("WorkerService is managed by systemd, connecting to existing service");
                    } else {
                        info!("Starting WorkerService for CLI worker commands");
                    }

                    let data_dir = dirs::data_local_dir()
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                        .join("hyprstream");
                    let runtime_dir = dirs::runtime_dir()
                        .unwrap_or_else(std::env::temp_dir)
                        .join("hyprstream");

                    let kata_boot_path = std::env::var("KATA_BOOT_PATH")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| {
                            std::path::PathBuf::from("/opt/kata/share/kata-containers")
                        });

                    let pool_config = PoolConfig {
                        warm_pool_size: 0,
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

                    let _worker_handle = if !worker_already_running {
                        use hyprstream_workers::image::RafsStore;
                        use hyprstream_workers::runtime::{SandboxBackend, KataBackend};
                        let rafs_store = Arc::new(RafsStore::new(image_config.clone())?);
                        let backend: Arc<dyn SandboxBackend> = Arc::new(
                            KataBackend::new(image_config, Arc::clone(&rafs_store)),
                        );
                        let worker_transport =
                            TransportConfig::inproc("hyprstream/workers");
                        let mut worker_service = WorkerService::new(
                            pool_config,
                            backend,
                            rafs_store,
                            worker_transport,
                            signing_key.clone(),
                        )?;

                        // Wire up policy-backed authorization
                        let worker_policy_client = PolicyClient::for_service(
                            signing_key.clone(),
                            resolve_service_vk("policy")
                                .ok_or_else(|| anyhow::anyhow!("Cannot resolve policy pubkey. Run wizard."))?,
                            None,
                        )?;
                        worker_service.set_authorize_fn(
                            hyprstream_core::services::build_authorize_fn(worker_policy_client),
                        );

                        // Standalone worker entrypoint serves RPC, so it must
                        // install the verify config too (#160) — otherwise the
                        // fail-closed default rejects all mesh traffic. No config
                        // is in scope here, so the mesh PQ store is empty (#157):
                        // identical to prior behavior (Hybrid fails closed for
                        // unknown peers).
                        install_envelope_verify_config(None);

                        let manager = InprocManager::new();
                        Some(
                            manager
                                .spawn(Box::new(worker_service))
                                .await
                                .context("Failed to start worker service")?,
                        )
                    } else {
                        None
                    };

                    use hyprstream_workers::runtime::WorkerClient;
                    let worker_server_vk = resolve_service_vk("worker")
                        .ok_or_else(|| anyhow::anyhow!("Cannot resolve worker pubkey. Run wizard."))?;
                    let worker_client =
                        WorkerClient::for_service(signing_key, worker_server_vk, None)?;

                    match action {
                        WorkerAction::List {
                            sandbox,
                            container,
                            containers,
                            sandboxes,
                            state,
                            verbose,
                        } => {
                            handle_worker_list(
                                &worker_client,
                                sandbox,
                                container,
                                containers,
                                sandboxes,
                                state,
                                verbose,
                            )
                            .await
                        }
                        WorkerAction::Run {
                            image,
                            command,
                            sandbox,
                            name,
                            env,
                            workdir,
                            detach,
                            rm,
                        } => {
                            handle_worker_run(
                                &worker_client,
                                &image,
                                command,
                                sandbox,
                                name,
                                env,
                                workdir,
                                detach,
                                rm,
                            )
                            .await
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
                        WorkerAction::Rm {
                            ids,
                            force,
                            volumes: _,
                        } => handle_worker_rm(&worker_client, ids, force).await,
                        WorkerAction::Status { id, verbose } => {
                            handle_worker_status(&worker_client, &id, verbose).await
                        }
                        WorkerAction::Stats {
                            ids,
                            stream: _,
                            no_header,
                        } => handle_worker_stats(&worker_client, ids, no_header).await,
                        WorkerAction::Exec {
                            container_id,
                            command,
                            timeout,
                        } => {
                            handle_worker_exec(
                                &worker_client,
                                &container_id,
                                command,
                                timeout,
                            )
                            .await
                        }
                        WorkerAction::Terminal {
                            container_id,
                            detach_keys,
                        } => {
                            handle_worker_terminal(
                                &worker_client,
                                &container_id,
                                &detach_keys,
                            )
                            .await
                        }
                        WorkerAction::Images(img_cmd) => match img_cmd {
                            ImageCommand::List {
                                verbose,
                                filter: _,
                            } => handle_images_list(&worker_client, verbose).await,
                            ImageCommand::Pull {
                                image,
                                username,
                                password,
                            } => {
                                handle_images_pull(
                                    &worker_client,
                                    &image,
                                    username,
                                    password,
                                )
                                .await
                            }
                            ImageCommand::Rm { images, force } => {
                                handle_images_rm(&worker_client, images, force).await
                            }
                            ImageCommand::Df => handle_images_df(&worker_client).await,
                        },
                    }
                },
            )
        }

        QuickCommand::Remote { command } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                match command {
                    RemoteQuickCommand::Add { model, name, url } => {
                        handle_remote_add(ctx.registry(), &model, &name, &url).await
                    }
                    RemoteQuickCommand::List { model, verbose } => {
                        handle_remote_list(ctx.registry(), &model, verbose).await
                    }
                    RemoteQuickCommand::Remove { model, name } => {
                        handle_remote_remove(ctx.registry(), &model, &name).await
                    }
                    RemoteQuickCommand::SetUrl { model, name, url } => {
                        handle_remote_set_url(ctx.registry(), &model, &name, &url).await
                    }
                    RemoteQuickCommand::Rename {
                        model,
                        old_name,
                        new_name,
                    } => {
                        handle_remote_rename(ctx.registry(), &model, &old_name, &new_name)
                            .await
                    }
                }
            },
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI pubkey resolution helper
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve a service's verifying key via the global trust store.
///
/// On first call in CLI mode, seeds the trust store from bootstrap-pubkeys.
/// Used by CLI command handlers that don't have a ServiceContext but need
/// to verify responses from a target service.
///
/// Returns `None` if bootstrap-pubkeys is not available (wizard not run).
/// Resolve the configured `[quic] relay` URI into the wire-reach
/// [`TransportConfig`] a producer advertises as its `Role::Relay` reach (#358).
///
/// Reuses the SAME [`hyprstream_rpc::service_entry`] codec the DID document uses
/// (by constructing the equivalent `QuicTransport` service entry and decoding it),
/// then projecting to wire-reach form via
/// [`hyprstream_rpc::moq_stream::relay_reach_from_decoded`] — so the advertised
/// stream relay and the DID transport address are produced by one codec and never
/// drift. WebPKI (public PDS / federation anchor) is the default; a self-signed
/// mesh relay carries its leaf-cert SHA-256 pins after `#` in the URI fragment.
fn resolve_moq_relay_reach(
    relay_uri: &str,
) -> Result<hyprstream_rpc::stream_info::TransportConfig> {
    use serde_json::json;
    // Optional `#<multibase certHash>[,<multibase certHash>...]` fragment pins a
    // self-signed mesh relay; absent ⇒ WebPKI (public CA-fronted relay).
    let (uri, cert_hashes): (&str, Vec<String>) = match relay_uri.split_once('#') {
        Some((u, frag)) => (u, frag.split(',').map(str::to_owned).collect()),
        None => (relay_uri, Vec::new()),
    };
    let webpki = cert_hashes.is_empty();
    let entry = json!({
        "type": "QuicTransport",
        "serviceEndpoint": {
            "uri": uri,
            "webpki": webpki,
            "certHashes": cert_hashes,
            "accept": ["moql"],
        },
    });
    let decoded = hyprstream_rpc::service_entry::decode_service_entry(&entry)
        .map_err(|e| anyhow::anyhow!("relay URI decode: {e}"))?;
    hyprstream_rpc::moq_stream::relay_reach_from_decoded(&decoded.config)
        .ok_or_else(|| anyhow::anyhow!("relay URI is not a network-routable transport"))
}

fn resolve_service_vk(service_name: &str) -> Option<VerifyingKey> {
    let trust = hyprstream_service::global_trust_store();
    // Fast path: already populated (service startup seeded it)
    if let Some(vk) = trust.resolve_one(service_name) {
        return Some(vk);
    }
    // CLI mode: seed from bootstrap-pubkeys on first use
    let secrets_dir = HyprConfig::resolve_secrets_dir();
    if let Ok(pubkeys) = hyprstream_core::auth::identity_store::load_bootstrap_pubkeys(&secrets_dir) {
        for (name, vk) in &pubkeys {
            trust.insert(
                *vk,
                hyprstream_service::Attestation {
                    scopes: std::iter::once(name.clone()).collect(),
                    subject: None,
                    jwt: None,
                    expires_at: 0,
                    attested_by: None,
                },
            );
        }
    }
    trust.resolve_one(service_name)
}

/// Install the process-global envelope verify configuration (#152, #160).
///
/// MUST be called once at startup on **every** process entrypoint that serves
/// RPC — the inproc daemon AND standalone service entrypoints (e.g. a lone
/// worker). When uninstalled, `global_verify_policy()` fail-closes to Hybrid
/// (#160), so a missing call here breaks mesh verification loudly rather than
/// silently downgrading to EdDSA-only. `install_verify_config` is first-write-
/// wins, so calling it from multiple stages within one process is harmless.
///
/// Policy default is Hybrid (SNS nested COSE); operators mid-rollout, before
/// peer ML-DSA bindings are provisioned, may set
/// `HYPRSTREAM_ENVELOPE_POLICY=classical`. Under Hybrid with no anchored peer
/// key the verifier fails closed (correct, by design).
/// Install the process-global envelope verify configuration.
///
/// When `oauth` is `Some`, the kid-anchored PQ trust store is populated eagerly
/// from `mesh_peers` (#157). When `None` (e.g. the standalone worker entrypoint,
/// which has no config in scope), the store is empty — identical to the prior
/// behavior. Either way the store is immutable after install.
fn install_envelope_verify_config(oauth: Option<&hyprstream_core::config::OAuthConfig>) {
    use hyprstream_rpc::crypto::CryptoPolicy;
    use hyprstream_rpc::envelope::{
        envelope_policy_from_env, install_response_verify_config, install_verify_config,
        EnvelopeVerifyConfig, KeyedPqTrustStore, PqTrustStore, ResponseVerifyConfig,
    };

    // Single source of truth for the rollout escape hatch
    // (`HYPRSTREAM_ENVELOPE_POLICY`), shared by the request AND response sides
    // (#277): `classical` downgrades both directions in lock-step.
    let policy = envelope_policy_from_env();

    // Mesh kid-anchored PQ trust store (#157, Option A): populated eagerly from
    // admin-configured `mesh_peers`, immutable after install. An empty store
    // under Hybrid fails closed for unknown peers (correct, by design).
    let keyed_store = match oauth {
        Some(oauth) => hyprstream_core::auth::mesh_trust::build_mesh_pq_trust_store(oauth),
        None => KeyedPqTrustStore::new(),
    };
    tracing::info!("mesh PQ trust store installed with {} peer binding(s)", keyed_store.len());

    // Per-host mesh identity roster (#328): bind each enrolled peer's Ed25519
    // signer key to its OWN per-host subject (`service:inference:host-<label>`)
    // in the global trust store, so a verified mesh peer resolves to its
    // granular principal — never the `"system"` god principal. Reuses the SAME
    // `mesh_peers` enrollment record as the PQ store above (no new roster type).
    // A networked peer whose key is NOT enrolled resolves to anonymous
    // (deny-by-default, fail-closed).
    if let Some(oauth) = oauth {
        let roster = hyprstream_core::auth::mesh_trust::build_mesh_identity_roster(oauth);
        let trust = hyprstream_service::global_trust_store();
        for (ed_pubkey, subject) in &roster {
            if let Ok(vk) = VerifyingKey::from_bytes(ed_pubkey) {
                trust.insert(
                    vk,
                    hyprstream_service::Attestation {
                        scopes: std::iter::once("inference".to_owned()).collect(),
                        subject: subject.name().map(ToOwned::to_owned),
                        jwt: None,
                        // Admin-anchored, out-of-band enrollment: no expiry (0).
                        expires_at: 0,
                        attested_by: None,
                    },
                );
            }
        }
        tracing::info!("mesh identity roster installed with {} per-host binding(s)", roster.len());
    }
    // Shared Arc: the SAME admin-anchored store backs both the request-side and
    // response-side verify configs — built once, anchored once (#277 reuse of
    // #157). The server's `#mesh-pq` ML-DSA-65 key (keyed by its Ed25519 mesh
    // signer identity) anchors response verification just as it anchors request
    // verification.
    let pq_store: std::sync::Arc<dyn PqTrustStore> = std::sync::Arc::new(keyed_store);

    // Install the RESPONSE-side process-global default (#277), mirroring the
    // request-side install below: fail-closed Hybrid by default with the same
    // anchored store, so native RPC clients consult it when no per-client store
    // was set.
    let _ = install_response_verify_config(ResponseVerifyConfig {
        policy,
        pq_store: Some(pq_store.clone()),
    });

    if install_verify_config(EnvelopeVerifyConfig {
        policy,
        pq_store: Some(pq_store),
    })
    .is_ok()
    {
        match policy {
            CryptoPolicy::Hybrid => tracing::info!(
                "envelope verify policy: HYBRID enforced (SNS nested COSE); \
                 peer ML-DSA bindings required for cross-node traffic"
            ),
            CryptoPolicy::Classical => {
                tracing::error!(
                    "⚠ SECURITY DOWNGRADE: envelope verify policy is CLASSICAL (EdDSA-only). \
                     Post-quantum protection is DISABLED for ALL envelope traffic. \
                     Unset HYPRSTREAM_ENVELOPE_POLICY or set it to 'hybrid' to re-enable PQ enforcement. \
                     This setting must not be used in production."
                );
                eprintln!(
                    "SECURITY WARNING: HYPRSTREAM_ENVELOPE_POLICY=classical disables post-quantum \
                     protection. Set to 'hybrid' or unset for production use."
                );
            }
        }
    }
}

fn main() -> Result<()> {
    // ROCm allocator and BLAS optimizations — must be set before any tch/libtorch init.
    // Expandable segments eliminates ~1,900 hipMalloc/hipFree calls per decode step.
    // hipBLASLt split-K improves M=1 GEMM (single-token decode) throughput.
    if std::env::var("HIP_VISIBLE_DEVICES").is_ok() {
        if std::env::var("PYTORCH_HIP_ALLOC_CONF").is_err() {
            // SAFETY: single-threaded at this point (before any thread spawning)
            unsafe {
                std::env::set_var(
                    "PYTORCH_HIP_ALLOC_CONF",
                    "expandable_segments:True,garbage_collection_threshold:0.8",
                );
            }
        }
        if std::env::var("TORCH_BLAS_PREFER_HIPBLASLT").is_err() {
            unsafe { std::env::set_var("TORCH_BLAS_PREFER_HIPBLASLT", "1"); }
        }
        if std::env::var("HIPBLASLT_FORCE_SPLIT_K").is_err() {
            unsafe { std::env::set_var("HIPBLASLT_FORCE_SPLIT_K", "1"); }
        }
    }

    // CUDA allocator optimizations — must be set before any tch/libtorch init.
    // Expandable segments reduces fragmentation from growing KV cache tensors.
    // Garbage collection threshold triggers proactive reclamation at 80% usage.
    if std::env::var("HIP_VISIBLE_DEVICES").is_err()
        && std::env::var("PYTORCH_CUDA_ALLOC_CONF").is_err()
    {
        // SAFETY: single-threaded at this point (before any thread spawning)
        unsafe {
            std::env::set_var(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True,garbage_collection_threshold:0.8",
            );
        }
    }

    // Set up panic handler to get better debugging information
    std::panic::set_hook(Box::new(|info| {
        eprintln!("\u{1f6a8} PANIC occurred:");
        if let Some(loc) = info.location() {
            eprintln!("   Location: {}:{}", loc.file(), loc.line());
        }
        if let Some(msg) = info.payload().downcast_ref::<&str>() {
            eprintln!("   Message: {}", msg);
        } else if let Some(msg) = info.payload().downcast_ref::<String>() {
            eprintln!("   Message: {}", msg);
        }
        eprintln!("\nThis was likely caused by a threading issue or memory corruption.");
        eprintln!("Please check for unsafe RefCell usage or race conditions.");
    }));

    // Check if we should re-exec into an installed GPU variant.
    // Uses Unix execve() to replace this process with the GPU-optimized binary.
    // The binary path comes from our own data directory (not user input).
    {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("hyprstream");
        if let Some((variant_id, version)) =
            hyprstream_core::cli::update_handlers::check_should_reexec(&data_dir)
        {
            hyprstream_core::cli::update_handlers::re_exec_variant(
                &data_dir,
                &variant_id,
                &version,
            );
            // re_exec_variant never returns
        }
    }

    // Parse CLI arguments using builder API
    let matches = build_cli().get_matches();

    // Extract global args
    let config_path = matches
        .get_one::<std::path::PathBuf>("config")
        .map(std::path::PathBuf::as_path);

    // Load configuration early
    let config = load_config(config_path)?;

    // Validate configuration
    config
        .validate()
        .context("Configuration validation failed")?;

    // Install the per-service streaming-response concurrency cap from config
    // (#186) before any RPC service starts. First-write-wins; ignore if already
    // installed on this process.
    let _ = hyprstream_rpc::streaming::install_max_concurrent_streams_per_service(
        config.server.max_concurrent_streams_per_service,
    );

    // ========== TRACING INITIALIZATION (before service startup) ==========
    let is_service_command = matches.subcommand_name() == Some("service");
    let _log_guard: Option<tracing_appender::non_blocking::WorkerGuard>;

    #[cfg(feature = "otel")]
    {
        let telemetry_provider = if is_service_command {
            TelemetryProvider::Otlp
        } else {
            TelemetryProvider::Stdout
        };

        let otel_enabled = std::env::var("HYPRSTREAM_OTEL_ENABLE")
            .unwrap_or_else(|_| "false".to_owned())
            .parse::<bool>()
            .unwrap_or(false);

        if otel_enabled {
            init_telemetry(telemetry_provider)?;
        } else {
            let default_log_level = match telemetry_provider {
                TelemetryProvider::Otlp => "hyprstream=info",
                TelemetryProvider::Stdout => "hyprstream=warn",
            };

            if let Ok(log_dir) = std::env::var("HYPRSTREAM_LOG_DIR") {
                let file_appender =
                    RollingFileAppender::new(Rotation::DAILY, &log_dir, "hyprstream.log");
                let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
                _log_guard = Some(guard);

                tracing_subscriber::fmt()
                    .with_env_filter(EnvFilter::builder().parse_lossy(
                        std::env::var("RUST_LOG")
                            .unwrap_or_else(|_| default_log_level.to_owned()),
                    ))
                    .with_target(true)
                    .with_file(true)
                    .with_line_number(true)
                    .with_writer(non_blocking)
                    .init();

                info!("File logging enabled to {}/hyprstream.log", log_dir);
            } else {
                _log_guard = None;
                tracing_subscriber::fmt()
                    .with_env_filter(EnvFilter::builder().parse_lossy(
                        std::env::var("RUST_LOG")
                            .unwrap_or_else(|_| default_log_level.to_owned()),
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
        let default_log_level = if is_service_command {
            "hyprstream=info"
        } else {
            "hyprstream=warn"
        };

        if let Ok(log_dir) = std::env::var("HYPRSTREAM_LOG_DIR") {
            let file_appender =
                RollingFileAppender::new(Rotation::DAILY, &log_dir, "hyprstream.log");
            let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
            _log_guard = Some(guard);

            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::builder().parse_lossy(
                    std::env::var("RUST_LOG")
                        .unwrap_or_else(|_| default_log_level.to_owned()),
                ))
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .with_writer(non_blocking)
                .init();

            info!("File logging enabled to {}/hyprstream.log", log_dir);
        } else {
            _log_guard = None;
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::builder().parse_lossy(
                    std::env::var("RUST_LOG")
                        .unwrap_or_else(|_| default_log_level.to_owned()),
                ))
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .init();
        }
    }

    info!("Hyprstream v{} starting up", env!("CARGO_PKG_VERSION"));
    // ========== END TRACING INITIALIZATION ==========

    // ========== EXECUTION MODE DETECTION ==========
    let execution_mode = ExecutionMode::detect();
    info!("Execution mode: {}", execution_mode);

    // ========== ENDPOINT REGISTRY INITIALIZATION ==========
    {
        use hyprstream_rpc::registry::init as init_registry;

        let mode = execution_mode.endpoint_mode();
        let runtime_dir = if execution_mode.uses_ipc() {
            Some(hyprstream_rpc::paths::runtime_dir())
        } else {
            None
        };

        init_registry(mode, runtime_dir);
    }
    // ========== END ENDPOINT REGISTRY INITIALIZATION ==========

    // ── Wizard / first-run early dispatch ───────────────────────────────────
    // The wizard is a bootstrap command: it creates credentials that the
    // registry client init (below) depends on. Handle both `wizard` and the
    // no-subcommand first-run path here so they work on a completely fresh
    // machine with no credentials yet. Neither path needs registry_client/ctx.
    {
        let models_dir = config.models_dir().clone();
        let is_wizard = matches!(matches.subcommand_name(), Some("wizard"));
        let is_first_run = is_wizard
            || (matches.subcommand_name().is_none()
                && hyprstream_core::cli::bootstrap_manager::is_first_run(&models_dir));

        if is_first_run {
            let services = config.services.startup.clone();
            if let Some(("wizard", sub_m)) = matches.subcommand() {
                let tui_mode = sub_m.get_flag("tui");
                let non_interactive = sub_m.get_flag("non_interactive");
                let start_services = sub_m.get_flag("start");
                let bootstrap_only = sub_m.get_flag("bootstrap_only");
                let enable_federation = sub_m.get_flag("enable_federation");
                let initial_user_role = sub_m
                    .get_one::<String>("initial_user_role")
                    .cloned()
                    .unwrap_or_else(|| "admin".to_owned());
                let use_tui = tui_mode || (!non_interactive && !bootstrap_only && supports_tui());
                return with_runtime(
                    RuntimeConfig { device: DeviceConfig::request_cpu(), multi_threaded: true },
                    || async move {
                        if use_tui {
                            hyprstream_core::cli::handle_wizard_tui(&models_dir, &services).await
                        } else {
                            hyprstream_core::cli::handle_wizard(
                                &models_dir, &services, non_interactive, start_services,
                                bootstrap_only, enable_federation, &initial_user_role,
                            ).await
                        }
                    },
                );
            }
            // No subcommand + first run → TUI wizard (with text fallback)
            return with_runtime(
                RuntimeConfig { device: DeviceConfig::request_cpu(), multi_threaded: true },
                || async move {
                    if supports_tui() {
                        hyprstream_core::cli::handle_wizard_tui(&models_dir, &services).await
                    } else {
                        // First-run fallback (no `wizard` subcommand) defaults
                        // federation to off — explicit opt-in only.
                        hyprstream_core::cli::handle_wizard(
                            &models_dir, &services, false, false, false, false, "admin",
                        ).await
                    }
                },
            );
        }
    }

    // Start registry service ONCE at CLI level
    let _registry_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to create registry runtime")?;

    // Start services and create keypair
    let (registry_client, signing_key, verifying_key): (
        RegistryClient,
        SigningKey,
        VerifyingKey,
    ) = _registry_runtime
        .block_on(async {
            let models_dir = config.models_dir();
            let keys_dir = models_dir.join(".registry").join("keys");
            let signing_key = load_or_generate_signing_key(&keys_dir).await?;
            let verifying_key = signing_key.verifying_key();

            // Resolve the registry service's verifying key from bootstrap pubkeys.
            // CLI mode doesn't have a ServiceContext, so we look up the target
            // pubkey directly from the credential store.
            let registry_vk = resolve_service_vk("registry")
                .ok_or_else(|| anyhow::anyhow!("Cannot resolve registry pubkey. Run wizard."))?;

            let client = hyprstream_core::services::RegistryClient::for_service(
                signing_key.clone(),
                registry_vk,
                None,
            )?;

            Ok::<_, anyhow::Error>((client, signing_key, verifying_key))
        })
        .context("Failed to connect to services")?;

    // Create application context with shared registry client
    let config_for_service = config.clone();
    let ctx = AppContext::with_client(config, Clone::clone(&registry_client));

    // ========== TOP-LEVEL COMMAND DISPATCH ==========
    match matches.subcommand() {
        // ── Schema-driven tool commands ──────────────────────────────────
        Some(("tool", sub_m)) => {
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    schema_cli::handle_tool_command(sub_m, signing_key).await
                },
            )?;
        }

        // ── Quick workflows ─────────────────────────────────────────────
        Some(("quick", sub_m)) => {
            let cmd = QuickCommand::from_arg_matches(sub_m)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            handle_quick_command(cmd, ctx, signing_key, verifying_key)?;
        }

        // ── Service management ──────────────────────────────────────────
        Some(("service", sub_m)) => {
            let action = ServiceAction::from_arg_matches(sub_m)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            let services = config_for_service.services.startup.clone();

            match action {
                ServiceAction::Install { services: filter, start, enable, system, systemd_session, verbose } => {
                    let models_dir = config_for_service.models_dir().clone();
                    let target = if system {
                        hyprstream_service::ServiceTarget::System
                    } else if systemd_session {
                        hyprstream_service::ServiceTarget::UserSession
                    } else {
                        hyprstream_service::ServiceTarget::User
                    };
                    with_runtime(
                        RuntimeConfig {
                            device: DeviceConfig::request_cpu(),
                            multi_threaded: true,
                        },
                        || async move {
                            handle_service_install(&models_dir, &services, filter, start, enable, target, verbose).await
                        },
                    )?;
                }

                ServiceAction::Uninstall { services: filter } => {
                    with_runtime(
                        RuntimeConfig {
                            device: DeviceConfig::request_cpu(),
                            multi_threaded: false,
                        },
                        || async move { handle_service_uninstall(&services, filter).await },
                    )?;
                }

                ServiceAction::Start {
                    name,
                    foreground,
                    daemon,
                    ipc,
                    services: multi_services,
                    quic_bind,
                    print_cert_hash,
                    standalone,
                } => {
                    if foreground || standalone {
                        // --foreground requires a service name or --services list;
                        // --standalone uses all configured services.
                        let (name, multi_services) = if standalone {
                            (String::from("standalone"), Some(services.clone()))
                        } else {
                            match (name, &multi_services) {
                                (Some(n), _) => (n, multi_services),
                                (None, Some(_)) => (String::from("multi"), multi_services),
                                (None, None) => return Err(anyhow::anyhow!(
                                    "--foreground requires a service name or --services list"
                                )),
                            }
                        };

                        // Standard foreground service startup
                        let rpc_mode = if ipc {
                            hyprstream_rpc::registry::EndpointMode::Ipc
                        } else {
                            hyprstream_rpc::registry::EndpointMode::Inproc
                        };

                        let config = config_for_service;

                        with_runtime(
                            RuntimeConfig {
                                device: DeviceConfig::request_cpu(),
                                multi_threaded: true,
                            },
                            || async move {
                                let (shutdown_tx, shutdown_rx) =
                                    tokio::sync::oneshot::channel::<()>();

                                tokio::spawn(async move {
                                    let _ = tokio::signal::ctrl_c().await;
                                    info!("Received shutdown signal");
                                    let _ = shutdown_tx.send(());
                                });

                                {
                                    use hyprstream_rpc::registry::init as init_registry;
                                    let runtime_dir = if ipc {
                                        Some(hyprstream_rpc::paths::runtime_dir())
                                    } else {
                                        None
                                    };
                                    init_registry(rpc_mode, runtime_dir);
                                }

                                let models_dir = config.models_dir();
                                let keys_dir = models_dir.join(".registry").join("keys");
                                let signing_key =
                                    load_or_generate_signing_key(&keys_dir).await?;
                                let verifying_key = signing_key.verifying_key();

                                let fed_src: Arc<dyn hyprstream_rpc::auth::FederationKeySource> =
                                    Arc::new(hyprstream_core::auth::FederationKeyResolver::new(
                                        &config.oauth.trusted_issuers,
                                    ));
                                let mut ctx = ServiceContext::new(
                                    signing_key.clone(),
                                    verifying_key,
                                    ipc,
                                    models_dir.clone(),
                                )
                                .with_oauth_issuer(config.oauth.issuer_url())
                                .with_federation_key_source(fed_src);

                                // Wire QUIC shared config from --quic-bind or [quic] config
                                let quic_cfg = if let Some(ref bind_addr) = quic_bind {
                                    let mut qc = hyprstream_core::config::QuicConfig::default();
                                    qc.enabled = true;
                                    qc.bind_addr = bind_addr.clone();
                                    qc
                                } else {
                                    config.quic.clone()
                                };

                                // Determine which services to start
                                let service_names: Vec<String> = if let Some(ref svc_list) = multi_services {
                                    svc_list.clone()
                                } else {
                                    vec![name.clone()]
                                };

                                if ipc {
                                    // Multi-process mode: each service gets its own independent key.
                                    let secrets_dir = std::path::PathBuf::from(
                                        std::env::var("HYPRSTREAM__SECRETS__PATH")
                                            .unwrap_or_else(|_| {
                                                dirs::config_dir()
                                                    .map(|d| d.join("hyprstream").join("credentials"))
                                                    .unwrap_or_default()
                                                    .to_str()
                                                    .unwrap_or("")
                                                    .to_owned()
                                            }),
                                    );

                                    // Load CA verifying key (trust anchor) for JWT verification only.
                                    // Do NOT insert into the trust store as "policy" scope — the trust
                                    // store maps ZMQ signing keys to subjects, and the policy service
                                    // signs ZMQ responses with its own independent keypair (not the CA
                                    // key). The CA key is only used to verify service JWTs (at+jwt).
                                    if let Ok(ca_vk) = hyprstream_core::auth::identity_store::load_ca_verifying_key(&secrets_dir) {
                                        ctx = ctx.with_ca_verifying_key(ca_vk);
                                    }

                                    // Load bootstrap pubkeys (all service pubkeys) into trust store
                                    if let Ok(pubkeys) = hyprstream_core::auth::identity_store::load_bootstrap_pubkeys(&secrets_dir) {
                                        for (svc_name, vk) in &pubkeys {
                                            hyprstream_service::global_trust_store().insert(
                                                *vk,
                                                hyprstream_service::Attestation {
                                                    scopes: std::iter::once(svc_name.clone()).collect(),
                                                    subject: None,
                                                    jwt: None,
                                                    expires_at: 0,
                                                    attested_by: None,
                                                },
                                            );
                                        }
                                    }

                                    // C3 fix: systemd uses flat %d/signing-key, standalone uses subdirectory.
                                    let systemd_mode = std::env::var("HYPRSTREAM__SECRETS__PATH").is_ok();
                                    let own_key = if systemd_mode {
                                        hyprstream_core::auth::identity_store::load_or_generate_node_signing_key(&secrets_dir)?
                                    } else {
                                        hyprstream_core::auth::identity_store::load_or_generate_service_signing_key(&secrets_dir, &name)?
                                    };

                                    if name == "policy" {
                                        // PolicyService: signing_key IS the CA key (already loaded).
                                        ctx = ctx.with_service_key(&name, own_key);
                                    } else {
                                        // Non-policy: swap signing_key to service's own independent key.
                                        // CA key is no longer accessible via ctx.signing_key().
                                        ctx = ctx.swap_signing_key(own_key.clone());
                                        ctx = ctx.with_service_key(&name, own_key);
                                    }
                                } else {
                                    // Single-process mode: load keys from disk (same as IPC).
                                    // Wizard must have run to create credentials.
                                    let secrets_dir = dirs::config_dir()
                                        .map(|d| d.join("hyprstream").join("credentials"))
                                        .ok_or_else(|| anyhow::anyhow!("Cannot determine config directory"))?;

                                    // Load CA verifying key (trust anchor)
                                    let ca_vk = hyprstream_core::auth::identity_store::load_ca_verifying_key(&secrets_dir)
                                        .context("CA key not found — run 'hyprstream wizard' first")?;
                                    // CA key is for JWT verification only — not a ZMQ signing key.
                                    ctx = ctx.with_ca_verifying_key(ca_vk);

                                    // Load bootstrap pubkeys (all service pubkeys) into trust store
                                    let pubkeys = hyprstream_core::auth::identity_store::load_bootstrap_pubkeys(&secrets_dir)
                                        .context("Bootstrap pubkeys not found — run 'hyprstream wizard' first")?;
                                    if pubkeys.is_empty() {
                                        anyhow::bail!("Bootstrap pubkeys file is empty — run 'hyprstream wizard' first");
                                    }
                                    for (svc_name, vk) in &pubkeys {
                                        hyprstream_service::global_trust_store().insert(
                                            *vk,
                                            hyprstream_service::Attestation {
                                                scopes: std::iter::once(svc_name.clone()).collect(),
                                                subject: None,
                                                jwt: None,
                                                expires_at: 0,
                                                attested_by: None,
                                            },
                                        );
                                    }
                                    tracing::debug!(
                                        "Loaded {} bootstrap pubkeys from {}",
                                        pubkeys.len(),
                                        secrets_dir.display()
                                    );

                                    // Load signing keys for each service being started
                                    for svc_name in &service_names {
                                        let svc_key = hyprstream_core::auth::identity_store::load_or_generate_service_signing_key(&secrets_dir, svc_name)
                                            .with_context(|| format!("Failed to load signing key for service '{}'", svc_name))?;
                                        ctx = ctx.with_service_key(svc_name, svc_key);
                                    }
                                }

                                // Wire QUIC shared config (must be after key generation so jwt_verifying_key is set)
                                if quic_cfg.enabled {
                                    let qc = quic_cfg;
                                    let (cert_chain, key_der) = qc.load_tls_materials()
                                        .context("Failed to load QUIC TLS materials")?;

                                    // Print cert hash if requested (hash of leaf cert)
                                    if print_cert_hash {
                                        let hash = hyprstream_rpc::transport::zmtp_quic::cert_hash(&cert_chain[0]);
                                        println!("QUIC/WebTransport cert hash: {}", hash);
                                    }

                                    // #358: resolve the producer-chosen moq relay
                                    // (if configured) into wire-reach form via the
                                    // SAME service_entry codec the DID doc uses, so
                                    // the stream relay address and DID address can't
                                    // drift. Empty = direct-only.
                                    let moq_relay = if qc.relay.trim().is_empty() {
                                        None
                                    } else {
                                        match resolve_moq_relay_reach(&qc.relay) {
                                            Ok(reach) => Some(reach),
                                            Err(e) => {
                                                tracing::warn!(
                                                    relay = %qc.relay,
                                                    "ignoring invalid [quic] relay; continuing direct-only: {e}"
                                                );
                                                None
                                            }
                                        }
                                    };
                                    let shared = hyprstream_service::QuicSharedConfig {
                                        cert_chain,
                                        key_der,
                                        base_ip: qc.socket_addr()?.ip(),
                                        server_name: qc.server_name.clone(),
                                        oauth_issuer_url: Some(config.oauth.issuer_url()),
                                        jwt_verifying_key: Some(ctx.jwt_verifying_key()),
                                        // #282: bind iroh in parallel to quinn when opted in.
                                        iroh_enabled: qc.iroh,
                                        // #358: producer-chosen relay rendezvous (None = direct-only).
                                        moq_relay,
                                    };
                                    ctx = ctx.with_quic(shared);
                                }

                                // Wire JWKS-backed key resolution (kid-based, on-miss-refetch).
                                {
                                    let issuer_url = config.oauth.issuer_url();
                                    let jwks_url = format!("{}/oauth/jwks", issuer_url.trim_end_matches('/'));
                                    let fetcher: hyprstream_rpc::auth::JwksFetcher = std::sync::Arc::new(move |url: String| {
                                        Box::pin(async move {
                                            let resp = reqwest::Client::builder()
                                                .danger_accept_invalid_certs(true)
                                                .build()?
                                                .get(&url)
                                                .send()
                                                .await?
                                                .error_for_status()?;
                                            let json: serde_json::Value = resp.json().await?;
                                            Ok(json)
                                        })
                                    });
                                    ctx.set_jwks_fetcher(fetcher);
                                    tracing::debug!("JWKS-backed key source configured: {}", jwks_url);
                                }

                                // Populate ML-DSA-65 verifying keys for PQ-hybrid JWT verification.
                                {
                                    let secrets_dir = hyprstream_core::config::HyprConfig::resolve_secrets_dir();
                                    let ml_dsa_store = hyprstream_core::auth::key_rotation::global_ml_dsa_key_store(
                                        &secrets_dir,
                                        &config.oauth,
                                    );
                                    let vks: Vec<_> = tokio::task::block_in_place(|| {
                                        let rt = tokio::runtime::Handle::current();
                                        rt.block_on(ml_dsa_store.all_slots_snapshot())
                                    })
                                    .iter()
                                    .map(hyprstream_core::auth::MlDsaKeySlot::verifying_key)
                                    .collect();
                                    let shared_vks = hyprstream_core::auth::key_rotation::global_ml_dsa_verifying_keys();
                                    let _ = shared_vks.write().map(|mut guard| *guard = vks);
                                    ctx.set_ml_dsa_verifying_keys(shared_vks);
                                    tracing::info!("PQ-hybrid: ML-DSA-65 verifying keys loaded for JWT verification");
                                }

                                // M3 (#152): install the process-global envelope
                                // verify configuration that closes the fail-open
                                // at the ZMQ RequestLoop + StreamService verify
                                // sites.
                                //
                                // KEY SEPARATION (PQUIP key-reuse restriction):
                                // the mesh hybrid identity's ML-DSA key MUST be
                                // distinct from the JWT-signing ML-DSA keyset
                                // (`global_ml_dsa_verifying_keys`, loaded above).
                                // We therefore DO NOT seed the mesh PqTrustStore
                                // from the JWT keyset. The mesh store is keyed by
                                // Ed25519 signer identity.
                                //
                                // #157 (Option A — eager, admin-anchored): the
                                // store is populated EAGERLY here, before install,
                                // from the operator-configured `mesh_peers`. It is
                                // immutable after install (no lazy-at-verify
                                // resolution). Empty `mesh_peers` => empty store =>
                                // unchanged behavior.
                                //
                                // Policy: Hybrid is enforced by default. Operators
                                // mid-rollout (before peer ML-DSA bindings are
                                // provisioned) may set
                                // HYPRSTREAM_ENVELOPE_POLICY=classical to keep the
                                // legacy EdDSA-only verifier. Under Hybrid with no
                                // anchored key the verifier FAILS CLOSED.
                                install_envelope_verify_config(Some(&config.oauth));

                                let manager = InprocManager::new();
                                let mut handles = Vec::new();

                                // Compute dependency-aware startup stages.
                                let stages = hyprstream_service::startup_stages(&service_names);

                                // #275: in the systemd / --ipc deployment each service
                                // runs in its OWN process. Only the `event` service's
                                // process initializes the process-global moq event bus
                                // origin. If THIS process does not host the `event`
                                // service, wire a CLIENT-mode event origin connected to
                                // the event service's UDS plane BEFORE any factory runs
                                // (so `EventPublisher::new` in e.g. the worker factory
                                // resolves the global). Idempotent + a no-op when `event`
                                // is co-located.
                                let hosts_event_service = stages
                                    .iter()
                                    .any(|stage| stage.iter().any(|s| s == "event"));
                                if !hosts_event_service {
                                    hyprstream_rpc::moq_event::ensure_event_client_origin(
                                        hyprstream_rpc::paths::event_socket(),
                                    );
                                }

                                for stage in &stages {
                                    for svc_name in stage {
                                        let factory = get_factory(svc_name).ok_or_else(|| {
                                            let available: Vec<_> =
                                                hyprstream_service::list_factories()
                                                    .map(|f| f.name)
                                                    .collect();
                                            anyhow::anyhow!(
                                                "Unknown service: '{}'. Available services: {}",
                                                svc_name,
                                                available.join(", ")
                                            )
                                        })?;

                                        info!(
                                            "Starting {} service in standalone mode (IPC: {})",
                                            svc_name, ipc
                                        );

                                        let spawnable = (factory.factory)(&ctx).context(format!(
                                            "Failed to create {} service",
                                            svc_name
                                        ))?;
                                        let handle = manager.spawn(spawnable).await.context(
                                            format!("Failed to start {} service", svc_name),
                                        )?;
                                        handles.push((svc_name.clone(), handle));
                                    }

                                    // Publish ready events for this stage before starting the next.
                                    for svc_name in stage {
                                        if let Ok(mut publisher) =
                                            hyprstream_workers::EventPublisher::new("system")
                                        {
                                            let _ = publisher
                                                .publish_raw(
                                                    &format!("system.{}.ready", svc_name),
                                                    b"",
                                                )
                                                .await;
                                        }
                                    }
                                }

                                if handles.is_empty() {
                                    return Err(anyhow::anyhow!("No services to start"));
                                }

                                let svc_names: Vec<_> = handles.iter().map(|(n, _)| n.as_str()).collect();
                                info!(
                                    "Services ready: [{}], waiting for shutdown signal",
                                    svc_names.join(", ")
                                );

                                let _ = shutdown_rx.await;

                                // Stop all services
                                for (svc_name, mut handle) in handles {
                                    if let Ok(mut publisher) =
                                        hyprstream_workers::EventPublisher::new("system")
                                    {
                                        let _ = publisher
                                            .publish_raw(
                                                &format!("system.{}.stopping", svc_name),
                                                b"",
                                            )
                                            .await;
                                    }
                                    let _ = handle.stop().await;
                                    info!("{} service stopped", svc_name);
                                }

                                Ok(())
                            },
                        )?;
                    } else {
                        with_runtime(
                            RuntimeConfig {
                                device: DeviceConfig::request_cpu(),
                                multi_threaded: false,
                            },
                            || async move {
                                handle_service_start(&services, name, daemon).await
                            },
                        )?;
                    }
                }

                ServiceAction::Stop { name } => {
                    with_runtime(
                        RuntimeConfig {
                            device: DeviceConfig::request_cpu(),
                            multi_threaded: false,
                        },
                        || async move { handle_service_stop(&services, name).await },
                    )?;
                }

                ServiceAction::Status { verbose } => {
                    with_runtime(
                        RuntimeConfig {
                            device: DeviceConfig::request_cpu(),
                            multi_threaded: false,
                        },
                        || async move { handle_service_status(&services, verbose).await },
                    )?;
                }

                ServiceAction::Repair { verbose } => {
                    let models_dir = config_for_service.models_dir().clone();
                    with_runtime(
                        RuntimeConfig {
                            device: DeviceConfig::request_cpu(),
                            multi_threaded: true,
                        },
                        || async move {
                            handle_service_install(&models_dir, &services, None, false, false, hyprstream_service::ServiceTarget::User, verbose).await
                        },
                    )?;
                }
            }
        }

        // ── User management ─────────────────────────────────────────────
        Some(("user", sub_m)) => {
            let cmd = UserCommand::from_arg_matches(sub_m)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let credentials_dir = ctx.config_dir().join("credentials");
            // User handlers are async, run them in a minimal runtime
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("Failed to create runtime for user command")?;
            rt.block_on(async {
                match cmd {
                    UserCommand::Register { username } => {
                        handle_user_register(&credentials_dir, &username).await?;
                    }
                    UserCommand::List => {
                        handle_user_list(&credentials_dir).await?;
                    }
                    UserCommand::Remove { username, force } => {
                        handle_user_remove(&credentials_dir, &username, force).await?;
                    }
                    UserCommand::Keys { command } => match command {
                        UserKeysCommand::List { username } => {
                            handle_user_keys_list(&credentials_dir, &username).await?;
                        }
                        UserKeysCommand::Import { username, format } => {
                            handle_user_keys_import(&credentials_dir, &username, &format).await?;
                        }
                        UserKeysCommand::Remove { username, fingerprint } => {
                            handle_user_keys_remove(&credentials_dir, &username, &fingerprint).await?;
                        }
                    },
                }
                Ok::<_, anyhow::Error>(())
            })?;
        }

        // ── TUI display server ──────────────────────────────────────────
        Some(("tui", sub_m)) => {
            let action = TuiAction::from_arg_matches(sub_m)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    use hyprstream_core::cli::tui_handlers;

                    // Load signing key for RPC authentication
                    let data_dir = dirs::data_local_dir()
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                        .join("hyprstream");
                    let keys_dir = data_dir.join("keys");
                    let signing_key = load_or_generate_signing_key(&keys_dir).await?;

                    match action {
                        TuiAction::Attach { session } => {
                            let sid = if session == 0 { None } else { Some(session) };
                            tui_handlers::handle_tui_attach(&signing_key, sid).await
                        }
                        TuiAction::New => tui_handlers::handle_tui_new(&signing_key).await,
                        TuiAction::List => tui_handlers::handle_tui_list(&signing_key).await,
                        TuiAction::Detach => tui_handlers::handle_tui_detach().await,
                        TuiAction::Play { cast_file, session, loop_playback } => {
                            tui_handlers::handle_tui_play(&signing_key, &cast_file, session, loop_playback).await
                        }
                        TuiAction::Shell { session } => {
                            let models_dir = config_for_service.models_dir().clone();
                            hyprstream_core::cli::shell_handlers::handle_tui_shell(
                                &signing_key, &models_dir, session,
                            ).await
                        }
                    }
                },
            )?;
        }

        // ── Flight SQL client ───────────────────────────────────────────
        Some(("flight", sub_m)) => {
            let args = FlightArgs::from_arg_matches(sub_m)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    use hyprstream_flight::client::{
                        format_batches_as_csv, format_batches_as_json,
                        format_batches_as_table, FlightClient,
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
                        println!(
                            "Connected to Flight SQL server at {}",
                            args.addr
                        );
                        println!(
                            "Interactive mode not yet implemented. Use --query to execute SQL."
                        );
                    }
                    Ok(())
                },
            )?;
        }

        // ── Sign OAuth challenge ─────────────────────────────────────────
        Some(("sign-challenge", sub_m)) => {
            let user_code = sub_m.get_one::<String>("user_code").cloned();
            let nonce = sub_m.get_one::<String>("nonce").cloned();
            let code_challenge = sub_m.get_one::<String>("code_challenge").cloned();
            let server = sub_m.get_one::<String>("server").cloned();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: false,
                },
                || async move {
                    handle_sign_challenge(user_code, nonce, code_challenge, server).await
                },
            )?;
        }

        // ── Wizard ── handled early (before registry init) above ────────────
        Some(("wizard", _)) => {
            unreachable!("wizard command is dispatched before registry init")
        }

        // ── Self-update ──────────────────────────────────────────────────
        Some(("update", sub_m)) => {
            let cleanup = sub_m.get_flag("cleanup");
            let models_dir = config_for_service.models_dir().clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: true,
                },
                || async move {
                    hyprstream_core::cli::update_handlers::handle_update(&models_dir, cleanup).await
                },
            )?;
        }

        // ── No subcommand → wizard (first run) or ShellClient ──────────
        _ => {
            let models_dir = config_for_service.models_dir().clone();
            if hyprstream_core::cli::bootstrap_manager::is_first_run(&models_dir) {
                let services = config_for_service.services.startup.clone();
                with_runtime(
                    RuntimeConfig { device: DeviceConfig::request_cpu(), multi_threaded: true },
                    || async move {
                        if supports_tui() {
                            hyprstream_core::cli::handle_wizard_tui(&models_dir, &services).await
                        } else {
                            // First-run auto-wizard defaults federation to off.
                            hyprstream_core::cli::handle_wizard(
                                &models_dir, &services, false, false, false, false, "admin",
                            ).await
                        }
                    },
                )?;
            } else {
                with_runtime(
                    RuntimeConfig { device: DeviceConfig::request_cpu(), multi_threaded: true },
                    || {
                        let md = models_dir.clone();
                        async move {
                            let keys_dir = md.join(".registry").join("keys");
                            let sk = load_or_generate_signing_key(&keys_dir).await?;
                            hyprstream_core::cli::shell_handlers::handle_shell_tui(&sk, &md).await
                        }
                    },
                )?;
            }
        }
    };

    Ok(())
}
