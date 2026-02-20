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
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::cli::commands::{
    ExecutionMode, FlightArgs, ImageCommand, ServiceAction, TrainingAction, WorkerAction,
};
use hyprstream_core::cli::quick::{QuickCommand, RemoteQuickCommand, WorktreeQuickCommand};
use hyprstream_core::cli::schema_cli;
use hyprstream_core::cli::{
    handle_branch, handle_checkout, handle_clone, handle_infer, handle_info, handle_list,
    handle_load, handle_policy_apply, handle_policy_apply_template, handle_policy_check,
    handle_policy_diff, handle_policy_edit, handle_policy_history, handle_policy_list_templates,
    handle_policy_rollback, handle_policy_show, handle_pull, handle_remote_add, handle_remote_list,
    handle_remote_remove, handle_remote_rename, handle_remote_set_url, handle_remove,
    handle_status, handle_token_create, handle_unload, handle_training_batch,
    handle_training_checkpoint, handle_training_infer, handle_training_init, handle_worktree_add,
    handle_worktree_info, handle_worktree_list, handle_worktree_remove,
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
use hyprstream_core::cli::commands::{PolicyCommand, TokenCommand};

#[cfg(feature = "experimental")]
use hyprstream_core::cli::{handle_commit, handle_merge, handle_push, MergeOptions};
use hyprstream_core::config::HyprConfig;
use hyprstream_core::storage::{GitRef, ModelRef};

// Registry and policy services - uses ZMQ-based services from hyprstream_core
use hyprstream_core::services::{PolicyClient, GenRegistryClient};
// Worker service for Kata-based workload execution
use hyprstream_workers::runtime::WorkerService;
use hyprstream_workers::workflow::WorkflowService;
use hyprstream_workers::{ImageConfig, PoolConfig, SpawnedService};
// ZMQ context for service startup
use hyprstream_core::zmq::global_context;
use std::sync::Arc;
// Unified service manager API
use hyprstream_rpc::service::{get_factory, InprocManager, ServiceContext, ServiceManager};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::{RequestIdentity, SigningKey, VerifyingKey};
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

    // Flight SQL client (derive-based args)
    app = app.subcommand(
        <FlightArgs as ClapArgs>::augment_args(
            ClapCommand::new("flight").about("Flight SQL client to query datasets"),
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

        QuickCommand::List => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move {
                let registry_path = ctx.models_dir().join(".registry");
                let policies_dir = registry_path.join("policies");
                let policy_manager = PolicyManager::new(&policies_dir).await.ok();
                handle_list(ctx.registry(), policy_manager.as_ref()).await
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
        } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_gpu(),
                multi_threaded: true,
            },
            || async move { handle_load(&model, max_context, kv_quant.into(), signing_key).await },
        ),

        QuickCommand::Unload { model } => with_runtime(
            RuntimeConfig {
                device: DeviceConfig::request_cpu(),
                multi_threaded: true,
            },
            || async move { handle_unload(&model, signing_key).await },
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
                        let rafs_store = Arc::new(RafsStore::new(image_config.clone())?);
                        let worker_transport =
                            TransportConfig::inproc("hyprstream/workers");
                        let mut worker_service = WorkerService::new(
                            pool_config,
                            image_config,
                            rafs_store,
                            global_context().clone(),
                            worker_transport,
                            signing_key.clone(),
                        )?;

                        // Wire up policy-backed authorization
                        let worker_policy_client = PolicyClient::new(
                            signing_key.clone(),
                            RequestIdentity::local(),
                        );
                        worker_service.set_authorize_fn(
                            hyprstream_core::services::build_authorize_fn(worker_policy_client),
                        );

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

                    use hyprstream_core::services::WorkerZmqClient;
                    let worker_client =
                        WorkerZmqClient::new(signing_key, RequestIdentity::local());

                    match action {
                        WorkerAction::List {
                            sandbox,
                            containers,
                            sandboxes,
                            state,
                            verbose,
                        } => {
                            handle_worker_list(
                                &worker_client,
                                sandbox,
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
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
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
                        .unwrap_or_else(|_| default_log_level.to_string()),
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
                        .unwrap_or_else(|_| default_log_level.to_string()),
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

    // Start registry service ONCE at CLI level
    let _registry_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to create registry runtime")?;

    // Start ZMQ-based services and create keypair
    let (registry_client, mut _service_handles, _workflow_service, signing_key, verifying_key): (
        GenRegistryClient,
        Vec<SpawnedService>,
        Option<Arc<WorkflowService>>,
        SigningKey,
        VerifyingKey,
    ) = if execution_mode == ExecutionMode::Inproc {
        _registry_runtime
            .block_on(async {
                let models_dir = config.models_dir();
                let keys_dir = models_dir.join(".registry").join("keys");
                let signing_key = load_or_generate_signing_key(&keys_dir).await?;
                let verifying_key = signing_key.verifying_key();

                let ctx = ServiceContext::new(
                    global_context(),
                    signing_key.clone(),
                    verifying_key,
                    false,
                    models_dir.clone(),
                );

                let manager = InprocManager::new();
                let mut handles = Vec::new();

                for service_name in &config.services.startup {
                    if service_name == "workflow" {
                        continue;
                    }

                    let factory = get_factory(service_name).ok_or_else(|| {
                        anyhow::anyhow!("Unknown service in startup config: {}", service_name)
                    })?;

                    info!("Starting {} service via factory", service_name);
                    let spawnable = (factory.factory)(&ctx)
                        .context(format!("Failed to create {} service", service_name))?;
                    let handle = manager
                        .spawn(spawnable)
                        .await
                        .context(format!("Failed to start {} service", service_name))?;
                    handles.push(handle);
                }

                let workflow_service =
                    if config.services.startup.iter().any(|s| s == "workflow") {
                        let mut wf_svc = WorkflowService::new(
                            global_context(),
                            TransportConfig::inproc("hyprstream/workflow"),
                            signing_key.clone(),
                        );
                        // Wire up policy-backed authorization
                        let wf_policy_client = PolicyClient::new(
                            signing_key.clone(),
                            RequestIdentity::local(),
                        );
                        wf_svc.set_authorize_fn(
                            hyprstream_core::services::build_authorize_fn(wf_policy_client),
                        );
                        let ws = Arc::new(wf_svc);
                        ws.clone()
                            .start()
                            .await
                            .context("Failed to start workflow service")?;
                        info!("WorkflowService started, subscribed to worker.* events");
                        Some(ws)
                    } else {
                        None
                    };

                let client = hyprstream_core::services::create_service_client(
                    &hyprstream_rpc::registry::global().endpoint("registry", hyprstream_rpc::registry::SocketKind::Rep).to_zmq_string(),
                    signing_key.clone(),
                    RequestIdentity::local(),
                );

                Ok::<_, anyhow::Error>((
                    client,
                    handles,
                    workflow_service,
                    signing_key,
                    verifying_key,
                ))
            })
            .context("Failed to initialize services")?
    } else {
        _registry_runtime
            .block_on(async {
                let models_dir = config.models_dir();
                let keys_dir = models_dir.join(".registry").join("keys");
                let signing_key = load_or_generate_signing_key(&keys_dir).await?;
                let verifying_key = signing_key.verifying_key();

                let client = hyprstream_core::services::create_service_client(
                    &hyprstream_rpc::registry::global().endpoint("registry", hyprstream_rpc::registry::SocketKind::Rep).to_zmq_string(),
                    signing_key.clone(),
                    RequestIdentity::local(),
                );

                Ok::<_, anyhow::Error>((client, Vec::new(), None, signing_key, verifying_key))
            })
            .context("Failed to connect to services")?
    };

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
                ServiceAction::Install { services: filter, start, verbose } => {
                    let models_dir = config_for_service.models_dir().clone();
                    with_runtime(
                        RuntimeConfig {
                            device: DeviceConfig::request_cpu(),
                            multi_threaded: true,
                        },
                        || async move {
                            handle_service_install(&models_dir, &services, filter, start, verbose).await
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
                    callback,
                } => {
                    if foreground {
                        let name = name.ok_or_else(|| {
                            anyhow::anyhow!("--foreground requires a service name")
                        })?;

                        // Check if this is inference callback mode
                        if let Some(callback_endpoint) = callback {
                            use hyprstream_core::services::InferenceService;

                            let instance_id =
                                if let Some(id) = name.strip_prefix("inference@") {
                                    id.to_owned()
                                } else {
                                    return Err(anyhow::anyhow!(
                                        "--callback requires inference@{{id}} format (got: {})",
                                        name
                                    ));
                                };

                            info!(
                                "Starting InferenceService in callback mode: {} -> {}",
                                instance_id, callback_endpoint
                            );

                            return with_runtime(
                                RuntimeConfig {
                                    device: DeviceConfig::request_cpu(),
                                    multi_threaded: false,
                                },
                                || async move {
                                    let data_dir = dirs::data_local_dir()
                                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                                        .join("hyprstream");
                                    let keys_dir = data_dir.join("keys");
                                    let signing_key =
                                        load_or_generate_signing_key(&keys_dir).await?;
                                    let policy_client = PolicyClient::new(
                                        signing_key.clone(),
                                        hyprstream_rpc::RequestIdentity::local(),
                                    );

                                    let runtime_config =
                                        hyprstream_core::runtime::RuntimeConfig::default();

                                    InferenceService::start_with_callback(
                                        instance_id,
                                        callback_endpoint,
                                        runtime_config,
                                        policy_client,
                                    )
                                    .await
                                },
                            );
                        }

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

                                let ctx = ServiceContext::new(
                                    global_context(),
                                    signing_key.clone(),
                                    verifying_key,
                                    ipc,
                                    models_dir.clone(),
                                );

                                let factory = get_factory(&name).ok_or_else(|| {
                                    let available: Vec<_> =
                                        hyprstream_rpc::service::list_factories()
                                            .map(|f| f.name)
                                            .collect();
                                    anyhow::anyhow!(
                                        "Unknown service: '{}'. Available services: {}",
                                        name,
                                        available.join(", ")
                                    )
                                })?;

                                info!(
                                    "Starting {} service in standalone mode (IPC: {})",
                                    name, ipc
                                );

                                let spawnable = (factory.factory)(&ctx).context(format!(
                                    "Failed to create {} service",
                                    name
                                ))?;
                                let manager = InprocManager::new();
                                let mut handle = manager.spawn(spawnable).await.context(
                                    format!("Failed to start {} service", name),
                                )?;

                                if let Ok(mut publisher) =
                                    hyprstream_workers::EventPublisher::new(
                                        &global_context(),
                                        "system",
                                    )
                                {
                                    let _ = publisher
                                        .publish_raw(
                                            &format!("system.{}.ready", name),
                                            b"",
                                        )
                                        .await;
                                }

                                info!(
                                    "{} service ready, waiting for shutdown signal",
                                    name
                                );

                                let _ = shutdown_rx.await;

                                if let Ok(mut publisher) =
                                    hyprstream_workers::EventPublisher::new(
                                        &global_context(),
                                        "system",
                                    )
                                {
                                    let _ = publisher
                                        .publish_raw(
                                            &format!("system.{}.stopping", name),
                                            b"",
                                        )
                                        .await;
                                }

                                let _ = handle.stop().await;
                                info!("{} service stopped", name);

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
                            handle_service_install(&models_dir, &services, None, false, verbose).await
                        },
                    )?;
                }
            }
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

        // ── No subcommand / help ────────────────────────────────────────
        _ => {
            build_cli().print_help()?;
        }
    };

    // Gracefully stop services before exiting (only for Inproc mode)
    _registry_runtime.block_on(async {
        for mut handle in _service_handles {
            if let Err(e) = handle.stop().await {
                tracing::warn!("Failed to stop service: {}", e);
            }
        }
    });

    Ok(())
}
