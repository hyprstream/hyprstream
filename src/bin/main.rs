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
use hyprstream_core::cli::{handle_pytorch_lora_command, DeviceConfig, DevicePreference, RuntimeConfig};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "hyprstream",
    version = env!("CARGO_PKG_VERSION"),
    about = "Real-time adaptive ML inference server with dynamic sparse weight adjustments",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
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

/// LoRA command handler - GPU requirements depend on action
fn handle_lora_cmd(cmd: hyprstream_core::cli::commands::lora::LoRACommand) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> {
    async move {
        info!("Using PyTorch LoRA implementation");
        handle_pytorch_lora_command(cmd).await.map_err(|e| e.into())
    }
}

/// Chat command handler - requires multi-threaded runtime for GPU
fn handle_chat_cmd(cmd: hyprstream_core::cli::commands::chat::ChatCommand, server_url: String) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> {
    async move { handle_chat_command(cmd, server_url).await }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Initialize logging with reasonable defaults
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .parse_lossy("hyprstream=info")
        )
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();
    
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
        Commands::Lora(cmd) => {
            let device_config = cmd.action.device_config();
            let multi_threaded = matches!(device_config.preference, DevicePreference::RequestGPU | DevicePreference::RequireGPU);

            with_runtime(
                RuntimeConfig {
                    device: device_config,
                    multi_threaded
                },
                || handle_lora_cmd(cmd)
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
    };

    Ok(())
}