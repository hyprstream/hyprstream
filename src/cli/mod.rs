use clap::Parser;

pub mod commands;
mod handlers;
pub use handlers::run_server;
pub use commands::Commands;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}