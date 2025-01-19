use clap::Args;

#[derive(Args)]
pub struct ServerCommand {
    /// Run server in detached mode
    #[arg(short = 'd', long)]
    pub detach: bool,

    /// Listen address in host:port format
    #[arg(long, value_name = "HOST:PORT")]
    pub listen: Option<String>,
}