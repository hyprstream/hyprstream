//! Hyprstream binary - ML inference server with VDB storage
//!
//! This binary provides the main entry point for the Hyprstream service.

use clap::Parser;
use config::Config;
use hyprstream_core::cli::commands::Commands;
use hyprstream_core::cli::handlers::{
    handle_server, handle_model_command, 
    handle_auth_command, handle_chat_command
};
use hyprstream_core::cli::handle_pytorch_lora_command;
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    
    // Quick GPU check
    let device = tch::Device::cuda_if_available();
    info!("Device: {:?}", if device == tch::Device::Cpu { "CPU" } else { "GPU" });
    
    // Handle commands
    match cli.command {
        Commands::Server(cmd) => {
            // Build config for server
            let config = Config::builder()
                .set_default("host", "127.0.0.1")?
                .set_default("port", "50051")?
                .set_default("storage.path", "./vdb_storage")?
                .set_default("storage.neural_compression", true)?
                .set_default("storage.hardware_acceleration", true)?
                .set_default("storage.cache_size_mb", 2048)?
                .build()?;
                
            handle_server(config).await
        }
        Commands::Model(cmd) => {
            // Model commands need a server URL (default to local)
            let server_url = "http://127.0.0.1:50051".to_string();
            handle_model_command(cmd, server_url).await
        }
        Commands::Lora(cmd) => {
            // Use PyTorch-native LoRA with autograd support
            info!("Using PyTorch LoRA implementation");
            handle_pytorch_lora_command(cmd).await
        }
        Commands::Auth(cmd) => {
            handle_auth_command(cmd).await
        }
        Commands::Chat(cmd) => {
            // Chat commands need a server URL (default to local)
            let server_url = "http://127.0.0.1:50051".to_string();
            handle_chat_command(cmd, server_url).await
        }
    }
}