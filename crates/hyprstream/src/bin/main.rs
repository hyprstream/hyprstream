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
    handle_user_create, handle_user_list, handle_user_register, handle_user_remove,
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
#[cfg(feature = "oci-image")]
use hyprstream_workers::ImageConfig;
use hyprstream_workers::PoolConfig;
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

    // PDS home attachment (RFC 8628 device authorization)
    app = app.subcommand(
        ClapCommand::new("pds")
            .about("Attach this host to its home personal data server")
            .subcommand(
                ClapCommand::new("join")
                    .visible_alias("attach")
                    .about("Authorize and attach this host to one home PDS")
                    .arg(
                        Arg::new("url")
                            .required(true)
                            .value_name("PDS_URL")
                            .help("HTTPS origin of the home PDS"),
                    )
                    .arg(
                        Arg::new("scope")
                            .long("scope")
                            .value_name("SCOPE")
                            .help("Space-delimited OAuth scopes requested from the PDS"),
                    ),
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
                Arg::new("pds_url")
                    .long("pds-url")
                    .value_name("PDS_URL")
                    .help("Attach this host to a home PDS after wizard setup (same as `pds join`)")
                    .conflicts_with("bootstrap_only")
                    .conflicts_with("tui"),
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
            )
            .arg(
                Arg::new("insecure")
                    .long("insecure")
                    .action(clap::ArgAction::SetTrue)
                    .help("Disable TLS certificate verification (use only against a trusted local dev server). By default the local self-signed dev cert is trusted automatically."),
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

    // Native MAC status/inspection (epic #547).
    app = app.subcommand(
        ClapCommand::new("mac")
            .about("Native MAC (mandatory access control) status and inspection")
            .subcommand(
                ClapCommand::new("genesis")
                    .about("Show the MAC genesis coverage report (activation coverage-gate evidence)"),
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
                let model_client = hyprstream_core::services::generated::model_client::ModelClient::from_resolver(
                    signing_key,
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
                    #[cfg(feature = "oci-image")]
                    let image_config = ImageConfig {
                        blobs_dir: data_dir.join("images/blobs"),
                        bootstrap_dir: data_dir.join("images/bootstrap"),
                        refs_dir: data_dir.join("images/refs"),
                        cache_dir: data_dir.join("images/cache"),
                        runtime_dir: runtime_dir.join("nydus"),
                        ..ImageConfig::default()
                    };

                    let _worker_handle = if !worker_already_running {
                        use hyprstream_workers::runtime::{
                            resolve_backend, BackendCtx, SandboxBackend,
                        };

                        // RAFS image store is built whenever the image filesystem
                        // service is compiled in (`oci-image`), so both kata
                        // (virtio-fs) and nspawn (FUSE tenant-VFS root, Model B
                        // #715) can compose a per-sandbox VFS from it.
                        #[cfg(feature = "oci-image")]
                        let rafs_store = {
                            use hyprstream_workers::image::RafsStore;
                            Arc::new(RafsStore::new(image_config.clone())?)
                        };

                        // Resolve + construct the backend fail-closed against the
                        // inventory registry: "auto" (default) picks the strongest
                        // available backend; an explicit name must be registered
                        // and available, else error. No silent nspawn fallback.
                        let backend_name: String = ctx
                            .config()
                            .worker
                            .as_ref()
                            .map(|w| w.backend.clone())
                            .unwrap_or_else(|| "auto".to_owned());
                        let backend_ctx = BackendCtx {
                            pool_config: pool_config.clone(),
                            #[cfg(feature = "oci-image")]
                            image_config,
                            #[cfg(feature = "oci-image")]
                            rafs_store: Arc::clone(&rafs_store),
                        };
                        let backend: Arc<dyn SandboxBackend> =
                            resolve_backend(&backend_name, &backend_ctx)?;

                        let worker_transport =
                            TransportConfig::inproc("hyprstream/workers");
                        let mut worker_service = WorkerService::new(
                            pool_config,
                            backend,
                            // Wire rafs_store for the canonical `kata` feature as
                            // well as its `kata-vm` compat alias — `kata-vm = ["kata"]`
                            // is one-way, so a `--features kata` build must not fall
                            // through to the `None` arm (#518).
                            #[cfg(any(feature = "kata", feature = "kata-vm"))]
                            Some(rafs_store),
                            #[cfg(not(any(feature = "kata", feature = "kata-vm")))]
                            None,
                            worker_transport,
                            signing_key.clone(),
                        )?;

                        // Wire up policy-backed authorization
                        let worker_policy_client = PolicyClient::for_local_bootstrap(
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
                    let worker_client = if worker_already_running {
                        WorkerClient::from_resolver(signing_key, None)?
                    } else {
                        let destination = signing_key.verifying_key();
                        WorkerClient::for_local_bootstrap(signing_key, destination, None)?
                    };

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
    let Ok(secrets_dir) = HyprConfig::resolve_secrets_dir() else {
        return None;
    };
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

async fn install_process_production_resolver(
    signing_key: &SigningKey,
    config: &HyprConfig,
) -> Result<()> {
    let trust_source = hyprstream_discovery::DeploymentTrustSource::from_anchors(
        config.cluster_at9p_did.as_deref(),
        config.cluster_did_web.as_deref(),
    )?;
    hyprstream_discovery::bootstrap_deployment_process(signing_key.clone(), trust_source).await?;
    hyprstream_rpc::envelope::install_browser_currentness_verifier(
        hyprstream_discovery::production_browser_currentness_verifier()?,
    )
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
/// Policy is pinned to Hybrid (SNS nested COSE). With no anchored peer key the
/// verifier fails closed (correct, by design).
/// Install the process-global envelope verify configuration.
///
/// When `oauth` is `Some`, the kid-anchored PQ trust store is populated eagerly
/// from `mesh_peers` (#157). When `None` (e.g. the standalone worker entrypoint,
/// which has no config in scope), the store is empty — identical to the prior
/// behavior. Either way the store is immutable after install.
fn install_envelope_verify_config(oauth: Option<&hyprstream_core::config::OAuthConfig>) {
    use hyprstream_rpc::envelope::{
        install_response_verify_config, install_verify_config, mandatory_envelope_policy,
        EnvelopeVerifyConfig, KeyedPqTrustStore, PqTrustStore, ResponseVerifyConfig,
    };

    // The application suite is pinned: requests and responses both require
    // Ed25519 + ML-DSA-65, with no runtime downgrade selector.
    let policy = mandatory_envelope_policy();

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
        tracing::info!(
            "envelope verify policy: HYBRID enforced (SNS nested COSE); \
             peer ML-DSA bindings required for cross-node traffic"
        );
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

    // RPC clients are used by ordinary CLI commands (`quick`, `tui`, etc.),
    // not only by service entrypoints. Install both request- and response-side
    // verification defaults before dispatch so every command uses the
    // operator-configured mesh trust store (#1018). The installer is
    // first-write-wins, so the service-specific calls below remain harmless.
    install_envelope_verify_config(Some(&config.oauth));

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
                let pds_url = sub_m.get_one::<String>("pds_url").cloned();
                let initial_user_role = sub_m
                    .get_one::<String>("initial_user_role")
                    .cloned()
                    .unwrap_or_else(|| "admin".to_owned());
                let use_tui = tui_mode || (pds_url.is_none() && !non_interactive && !bootstrap_only && supports_tui());
                let config = config.clone();
                return with_runtime(
                    RuntimeConfig { device: DeviceConfig::request_cpu(), multi_threaded: true },
                    || async move {
                        if use_tui {
                            hyprstream_core::cli::handle_wizard_tui(&models_dir, &services).await
                        } else {
                            hyprstream_core::cli::handle_wizard(
                                &models_dir, &services, non_interactive, start_services,
                                bootstrap_only, enable_federation, &initial_user_role,
                            ).await?;
                            if let Some(pds_url) = pds_url {
                                hyprstream_core::cli::pds_handlers::handle_pds_join(
                                    &config, &pds_url, None,
                                ).await?;
                            }
                            Ok(())
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

    // ── `mac` early dispatch ─────────────────────────────────────────────────
    // `mac genesis` is a read-only, in-memory diagnostic (compile-time service
    // inventory + site policy — no registry client, no credentials). Dispatch it
    // before credential loading and registry-key resolution so it works on a
    // fresh or broken installation, where its coverage evidence is most needed.
    if let Some(("mac", sub_m)) = matches.subcommand() {
        match sub_m.subcommand() {
            Some(("genesis", _)) => {
                // Build the boot-time gate and print the coverage report. This is
                // read-only inspection — it changes no enforcement state.
                let gate = hyprstream_core::mac::GenesisGate::production();
                print!("{}", gate.render_report());
            }
            _ => {
                eprintln!("usage: hyprstream mac genesis");
            }
        }
        return Ok(());
    }

    // ── `pds` early dispatch ─────────────────────────────────────────────────
    // PDS attachment is independent of the local registry: it must work on a
    // newly provisioned host before local services are running.
    if let Some(("pds", sub_m)) = matches.subcommand() {
        match sub_m.subcommand() {
            Some(("join", join_m)) => {
                let pds_url = join_m
                    .get_one::<String>("url")
                    .ok_or_else(|| anyhow::anyhow!("PDS URL is required"))?;
                let scope = join_m.get_one::<String>("scope").map(String::as_str);
                return with_runtime(
                    RuntimeConfig { device: DeviceConfig::request_cpu(), multi_threaded: true },
                    || hyprstream_core::cli::pds_handlers::handle_pds_join(&config, pds_url, scope),
                );
            }
            _ => anyhow::bail!("usage: hyprstream pds join <PDS_URL> [--scope <SCOPE>]"),
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
            let keys_dir = config.models_dir().join(".registry").join("keys");
            let signing_key = load_or_generate_signing_key(&keys_dir).await?;
            let verifying_key = signing_key.verifying_key();

            install_process_production_resolver(&signing_key, &config).await
                .context("Failed to install checkpoint-backed production resolver")?;
            let client = hyprstream_core::services::RegistryClient::from_resolver(
                signing_key.clone(),
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
                    // MAC genesis coverage gate (S1 activation, #567): log the
                    // boot-time coverage report on EVERY `service start` path —
                    // foreground (the mode that actually hosts services, incl.
                    // systemd ExecStart and standalone-spawned children),
                    // standalone, and the systemd/spawner dispatch below.
                    // Emit the coverage-gate evidence consumed by the active
                    // 9P translator PEP (see `mac::genesis`).
                    hyprstream_core::mac::GenesisGate::production().log_report();

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
                                    //
                                    // #759: resolve via the SAME authoritative path
                                    // `HyprConfig::resolve_secrets_dir()` uses elsewhere in this
                                    // function (e.g. the ML-DSA store below) and that
                                    // `bootstrap_manager::do_bootstrap` uses to write credentials —
                                    // NOT a hand-rolled `dirs::config_dir()` recomputation. The two
                                    // used to diverge under `HYPRSTREAM_INSTANCE` or a configured
                                    // `[secrets] path` (the manual version ignored both), so a
                                    // bootstrapped credential could land in a different directory
                                    // than this process reads from — the same "consumer silently
                                    // re-derives instead of reading the authoritative record"
                                    // disease #441 targets, just one directory earlier.
                                    let secrets_dir = hyprstream_core::config::HyprConfig::resolve_secrets_dir()?;
                                    ctx = ctx.with_ca_verifying_key(
                                        hyprstream_core::auth::identity_store::load_ca_verifying_key(
                                            &secrets_dir,
                                        )?,
                                    );

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

                                    // C3 fix: scoped credential providers use flat
                                    // %d/signing-key, while standalone uses a subdirectory.
                                    // #759: "policy" always resolves to the flat root/CA key regardless of
                                    // secrets profile — see `resolve_service_signing_key` for why.
                                    let secrets_profile = if std::env::var(
                                        "HYPRSTREAM__SECRETS__PATH",
                                    )
                                    .is_ok()
                                    {
                                        hyprstream_core::auth::identity_store::SecretsProfile::PerServiceScoped
                                    } else {
                                        hyprstream_core::auth::identity_store::SecretsProfile::SharedDirectory
                                    };
                                    let own_key = hyprstream_core::auth::identity_store::resolve_service_signing_key(
                                        &secrets_dir, &name, secrets_profile,
                                    )?;

                                    if name == "policy" {
                                        // PolicyService: signing_key IS the CA key (already loaded).
                                        ctx = ctx.with_service_key(&name, own_key);
                                    } else {
                                        // Non-policy: swap signing_key to service's own independent key.
                                        // CA key is no longer accessible via ctx.signing_key().
                                        ctx = ctx.swap_signing_key(own_key.clone());
                                        ctx = ctx.with_service_key(&name, own_key.clone());

                                        // In systemd mode the credential dir is flat (%d = secrets_dir).
                                        // Load our own service-jwt from disk and seed the trust store so
                                        // that register_service_key() finds it on first call — otherwise
                                        // the trust store only has pubkeys (jwt: None) from bootstrap-pubkeys
                                        // and registration is silently skipped.
                                        if let Ok(Some(jwt_bytes)) = hyprstream_core::auth::identity_store::read_secret(&secrets_dir, "service-jwt") {
                                            if let Ok(jwt_str) = String::from_utf8(jwt_bytes) {
                                                let exp = hyprstream_core::auth::identity_store::decode_jwt_exp_raw(&jwt_str).unwrap_or(0);
                                                hyprstream_service::global_trust_store().insert(
                                                    own_key.verifying_key(),
                                                    hyprstream_service::Attestation {
                                                        scopes: std::iter::once(name.clone()).collect(),
                                                        subject: None,
                                                        jwt: Some(jwt_str),
                                                        expires_at: exp,
                                                        attested_by: None,
                                                    },
                                                );
                                                tracing::info!(service = %name, "Seeded trust store with own service-jwt from credential dir");
                                            }
                                        }
                                    }
                                } else {
                                    // Single-process mode: load keys from disk (same as IPC).
                                    // Wizard must have run to create credentials.
                                    //
                                    // #759: same authoritative resolver as the `--ipc` branch above —
                                    // see the comment there.
                                    let secrets_dir = hyprstream_core::config::HyprConfig::resolve_secrets_dir()?;
                                    ctx = ctx.with_ca_verifying_key(
                                        hyprstream_core::auth::identity_store::load_ca_verifying_key(
                                            &secrets_dir,
                                        )?,
                                    );

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

                                    // Load signing keys for each service being started, and
                                    // seed own service-jwt into trust store so register_service_key
                                    // can find it on first call (trust store starts with jwt: None).
                                    //
                                    // #759: "policy" must resolve to the flat root/CA key (matching
                                    // `bootstrap_pubkeys["policy"]`), not an independent per-service
                                    // key — see `resolve_service_signing_key` for why. This mirrors the
                                    // same fix applied to the `--ipc` branch above.
                                    for svc_name in &service_names {
                                        let svc_key = hyprstream_core::auth::identity_store::resolve_service_signing_key(
                                            &secrets_dir,
                                            svc_name,
                                            hyprstream_core::auth::identity_store::SecretsProfile::SharedDirectory,
                                        )
                                            .with_context(|| format!("Failed to load signing key for service '{}'", svc_name))?;
                                        ctx = ctx.with_service_key(svc_name, svc_key.clone());

                                        if svc_name != "policy" {
                                            if let Ok(Some(jwt_str)) = hyprstream_core::auth::identity_store::load_service_jwt(&secrets_dir, svc_name) {
                                                let exp = hyprstream_core::auth::identity_store::decode_jwt_exp_raw(&jwt_str).unwrap_or(0);
                                                hyprstream_service::global_trust_store().insert(
                                                    svc_key.verifying_key(),
                                                    hyprstream_service::Attestation {
                                                        scopes: std::iter::once(svc_name.clone()).collect(),
                                                        subject: None,
                                                        jwt: Some(jwt_str),
                                                        expires_at: exp,
                                                        attested_by: None,
                                                    },
                                                );
                                                tracing::info!(service = %svc_name, "Seeded trust store with own service-jwt from credential dir");
                                            }
                                        }
                                    }
                                }

                                if quic_cfg.enabled {
                                    ctx = hyprstream_core::services::factories::with_checkpointed_native_announcements(
                                        ctx,
                                        &service_names,
                                    )?;
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
                                        native_announcement_publisher: Some(std::sync::Arc::new(
                                            |request: hyprstream_service::NativeAnnouncementRequest| {
                                                std::thread::spawn(move || {
                                                    let runtime = match tokio::runtime::Builder::new_current_thread()
                                                        .enable_all()
                                                        .build()
                                                    {
                                                        Ok(runtime) => runtime,
                                                        Err(error) => {
                                                            tracing::warn!("Failed to create announcement runtime: {error}");
                                                            return;
                                                        }
                                                    };
                                                    runtime.block_on(async move {
                                                        let client = match hyprstream_discovery::DiscoveryClient::for_local_bootstrap(
                                                            request.signing_key,
                                                            request.discovery_verifying_key,
                                                            None,
                                                        ) {
                                                            Ok(client) => client,
                                                            Err(error) => {
                                                                tracing::warn!("Failed to build DiscoveryClient: {error}");
                                                                return;
                                                            }
                                                        };
                                                        let announcement = hyprstream_discovery::ServiceAnnouncement {
                                                            service_name: request.service_name,
                                                            socket_kind: "quic".to_owned(),
                                                            endpoint: request.endpoint,
                                                            service_jwt: request.service_jwt,
                                                            service_did: request.service_did,
                                                            capabilities: request.capabilities,
                                                            accepted_state_digest: request.accepted_state_digest,
                                                            accepted_state_epoch: request.accepted_state_epoch,
                                                            response_key_id: request.response_key_id,
                                                            request_kem_key_id: request.request_kem_key_id,
                                                            request_kem_recipient: request.request_kem_recipient,
                                                            expires_at_unix_ms: request.expires_at_unix_ms,
                                                        };
                                                        match client.announce(&announcement).await {
                                                            Ok(_) => tracing::info!("Announced QUIC endpoint to DiscoveryService"),
                                                            Err(error) => tracing::warn!("Failed to announce QUIC endpoint: {error}"),
                                                        }
                                                    });
                                                });
                                            },
                                        )),
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
                                    let secrets_dir = hyprstream_core::config::HyprConfig::resolve_secrets_dir()?;
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
                                    let ed_store = hyprstream_core::auth::key_rotation::global_ed25519_key_store(
                                        &secrets_dir,
                                        &config.oauth,
                                    );
                                    let restore_result = tokio::task::block_in_place(|| {
                                        let rt = tokio::runtime::Handle::current();
                                        rt.block_on(hyprstream_core::auth::key_rotation::restore_composite_verifying_key_set(
                                            &secrets_dir,
                                            &ed_store,
                                            &ml_dsa_store,
                                            ctx.jwt_verifying_key(),
                                        ))
                                    });
                                    if let Err(error) = restore_result {
                                        tracing::warn!("exact composite pair ledger unavailable in verifier process: {error}");
                                    }
                                    tracing::info!("PQ-hybrid: ML-DSA-65 verifying keys loaded for JWT verification");

                                    // MAC S4 policy bootload (#570): compile → sign
                                    // → verify-once-at-load → install the node's
                                    // baseline CompiledPolicy so the MAC PDP inputs
                                    // are real. This is the one missing boot step
                                    // that populates `mac::COMPILED_POLICY`, flipping
                                    // `exchange_enrollment_resolver` from the deny-all
                                    // `DenyUnlabeledResolver` to the real (#698)
                                    // `EnrollmentSubjectContextResolver`.
                                    //
                                    // The 9P PEP is active and records the
                                    // bootloaded policy as audit provenance.
                                    //
                                    // Fail-closed: the baseline is hybrid-signed
                                    // (EdDSA + ML-DSA-65) and verified with
                                    // require_pq=true. With no active ML-DSA signing
                                    // key (Classical mode) nothing is installed —
                                    // never an Ed25519-only baseline (design §14) —
                                    // and the resolver keeps denying.
                                    let active_pq = tokio::task::block_in_place(|| {
                                        let rt = tokio::runtime::Handle::current();
                                        rt.block_on(ml_dsa_store.active_key())
                                    });
                                    match active_pq {
                                        Some(pq_sk) => match hyprstream_core::mac::install_baseline_boot_policy(
                                            &signing_key,
                                            &pq_sk,
                                        ) {
                                            Ok((policy, true)) => tracing::info!(
                                                "MAC S4: baseline compiled policy installed \
                                                 (generation {}, 9P enforcement active)",
                                                policy.generation
                                            ),
                                            // The seam is write-once: a policy was
                                            // already installed this process, so the
                                            // baseline did NOT replace it. Report it
                                            // honestly rather than claiming an install.
                                            Ok((_policy, false)) => tracing::warn!(
                                                "MAC S4: a compiled policy was already installed; \
                                                 baseline NOT installed (write-once seam unchanged)"
                                            ),
                                            Err(e) => tracing::error!(
                                                "MAC S4: baseline policy bootload failed \
                                                 (grant-path resolver stays fail-closed): {e}"
                                            ),
                                        },
                                        None => tracing::warn!(
                                            "MAC S4: no active ML-DSA signing key; baseline policy \
                                             NOT installed (fail-closed — grant-path resolver denies)"
                                        ),
                                    }
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
                                // Policy: Hybrid is mandatory. With no anchored
                                // peer key the verifier FAILS CLOSED.
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
                                        if let Ok(publisher) =
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
                                    if let Ok(publisher) =
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
            // Deliberately resolve from `ctx`: it honors the `--config <path>`
            // CLI override. The operator's config must win here, or `--config`
            // invocations would resolve the user store from XDG defaults and
            // split it.
            let credentials_dir =
                hyprstream_core::auth::identity_store::credentials_dir_for_config(Some(ctx.config()))?;
            // User handlers are async, run them in a minimal runtime
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("Failed to create runtime for user command")?;
            rt.block_on(async {
                match cmd {
                    UserCommand::Create { username, role, generate, key, ssh, no_role } => {
                        handle_user_create(
                            &credentials_dir,
                            &username,
                            &role,
                            no_role,
                            generate,
                            key.as_deref(),
                            ssh.as_deref(),
                        )
                        .await?;
                    }
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
            let insecure = sub_m.get_flag("insecure");
            // Pass the already-loaded config (honors `--config`) so the OAuth
            // issuer URL and the local self-signed cert's secrets dir are
            // resolved from the user's selected configuration, not re-derived
            // from defaults (#450). `config_for_service` is the surviving
            // clone (`config` itself is moved into AppContext above).
            let sign_cfg = config_for_service.clone();
            with_runtime(
                RuntimeConfig {
                    device: DeviceConfig::request_cpu(),
                    multi_threaded: false,
                },
                || async move {
                    handle_sign_challenge(
                        user_code,
                        nonce,
                        code_challenge,
                        server,
                        insecure,
                        Some(&sign_cfg),
                    )
                    .await
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

        // ── Native MAC status/inspection ── handled early (before registry
        // init) above: it is a read-only diagnostic that must work on fresh
        // installations with no credentials.
        Some(("mac", _)) => {
            unreachable!("mac command is dispatched before registry init")
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod resolver_startup_controls {
    #[test]
    fn command_and_service_processes_install_before_consumers() {
        let source = include_str!("main.rs");
        let production = source
            .split("mod resolver_startup_controls {")
            .next()
            .expect("production binary source");
        let startup = production
            .find("install_process_production_resolver(&signing_key, &config).await")
            .expect("trusted startup boundary");
        let install = production[startup..]
            .find("install_process_production_resolver(&signing_key, &config).await")
            .expect("process resolver install");
        let first_generated = production[startup..]
            .find("RegistryClient::from_resolver(")
            .expect("first generated client");
        let command_dispatch = production[startup..]
            .find("match matches.subcommand()")
            .expect("top-level command dispatch");
        assert!(install < first_generated);
        assert!(first_generated < command_dispatch);
        assert_eq!(
            production
                .matches("install_process_production_resolver(&signing_key, &config).await")
                .count(),
            1
        );
        assert!(!production.contains("install_process_production_resolver(&mut ctx)"));

        let announcements = production
            .find("with_checkpointed_native_announcements(")
            .expect("authenticated announcements");
        let first_factory = production[announcements..]
            .find("get_factory(")
            .expect("first inventory factory invocation");
        assert!(first_factory > 0);

        // These entry points all dispatch below the single command bootstrap:
        // quick, TUI, schema/tool, MCP/OpenAI services, shell, training, git,
        // and isolated `service start` IPC consumers.
        for entry in [
            "Some((\"tool\"",
            "Some((\"quick\"",
            "Some((\"tui\"",
            "handle_shell_tui",
            "handle_training_batch",
        ] {
            assert!(production.contains(entry), "missing {entry}");
        }
    }

    #[test]
    fn command_local_spawns_never_use_the_production_resolver() {
        let source = include_str!("main.rs");
        let worker = source
            .find("let worker_client = if worker_already_running")
            .expect("worker client selection");
        let worker_selection = &source[worker..worker + 500];
        assert!(worker_selection.contains("WorkerClient::from_resolver"));
        assert!(worker_selection.contains("WorkerClient::for_local_bootstrap"));

        let training = include_str!("../cli/training_handlers.rs");
        assert_eq!(training.matches("InferenceClient::from_resolver").count(), 0);
        assert_eq!(
            training
                .matches("InferenceClient::for_local_bootstrap")
                .count(),
            2
        );
    }
}
