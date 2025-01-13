use clap::Args;

#[derive(Args)]
pub struct ChatCommand {
    /// Model identifier in model:tag format
    pub model_tag: String,
}
