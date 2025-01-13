use clap::Args;

#[derive(Args)]
pub struct ServerCommand {
    /// Listen address in host:port format
    #[arg(long, value_name = "HOST:PORT")]
    pub listen: Option<String>,

    /// Run in debug mode
    #[arg(short, long)]
    pub debug: bool,
}
