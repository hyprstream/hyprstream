//! Chat command for inference with models/composed models

use clap::Args;

/// Chat with a model or composed model
#[derive(Args)]
pub struct ChatCommand {
    /// Model or composed model ID to chat with
    pub model_id: String,
    
    /// Enable training mode (inference on base, train on LoRA)
    #[arg(long)]
    pub train: bool,
    
    /// Maximum tokens per response
    #[arg(long, default_value = "500")]
    pub max_tokens: usize,
    
    /// Temperature for sampling
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,
    
    /// Top-p for nucleus sampling
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,
    
    /// Load chat history from file
    #[arg(long)]
    pub history: Option<String>,
    
    /// Save chat history to file
    #[arg(long)]
    pub save_history: Option<String>,
    
    /// Single prompt mode (non-interactive)
    #[arg(long)]
    pub prompt: Option<String>,
}