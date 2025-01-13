use clap::Args;

#[derive(Args)]
pub struct SqlCommand {
    /// Server address in host:port format
    #[arg(long, value_name = "HOST:PORT")]
    pub host: Option<String>,
}
