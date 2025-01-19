use clap::Parser;

pub mod commands;
mod handlers;
pub use commands::config;
pub use commands::Commands;
pub use handlers::run_server;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}
