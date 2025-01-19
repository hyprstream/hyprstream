use clap::Parser;
use crate::config::CliArgs;

mod handlers;
pub use handlers::run_server;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Run server in detached mode
    #[arg(short = 'd', long)]
    pub detach: bool,

    #[command(flatten)]
    pub args: CliArgs,
}