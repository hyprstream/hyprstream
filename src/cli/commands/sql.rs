use clap::Args;

#[derive(Args)]
#[command(disable_help_flag = true)]
pub struct SqlCommand {
    /// Server host:port address (e.g. localhost:8080)
    #[arg(short = 'h', long = "host", value_name = "HOST:PORT")]
    pub host: Option<String>,

    /// SQL query to execute
    #[arg(short = 'c', required = true, value_name = "QUERY")]
    pub query: String,

    /// Print help information
    #[arg(long = "help", action = clap::ArgAction::Help)]
    pub help: Option<bool>,
}