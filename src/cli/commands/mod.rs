pub mod config;
pub mod server;
pub mod model;
pub mod lora;
pub mod chat;
pub mod git;

pub use server::{CacheConfig, EngineConfig, ServerCommand, ServerConfig};
pub use model::ModelCommand;
pub use lora::LoRACommand;
pub use chat::ChatCommand;
pub use git::{GitCommand, GitAction};

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Commands {
    /// Start the Hyprstream server
    Server(ServerCommand),
    /// Manage models from registries (HuggingFace, etc.)
    Model(ModelCommand),
    /// Manage LoRA adapters and training
    Lora(LoRACommand),
    /// Chat with a model or composed model
    Chat(ChatCommand),

    // Phase 1: Git-style commands at top level
    /// Create a new branch
    Branch {
        /// Model name
        model: String,
        /// Branch name
        name: String,
        /// Create from specific ref
        #[arg(long)]
        from: Option<String>,
    },

    /// Switch branches or checkout specific commit/tag
    Checkout {
        /// Model reference (e.g., "llama3:main")
        model_ref: String,
        /// Create new branch if it doesn't exist
        #[arg(short = 'b')]
        create_branch: bool,
        /// Force checkout
        #[arg(long)]
        force: bool,
    },

    /// Show working tree status
    Status {
        /// Model name
        model: Option<String>,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Record changes to the repository
    Commit {
        /// Model name
        model: String,
        /// Commit message
        #[arg(short, long)]
        message: String,
        /// Stage all tracked files
        #[arg(short = 'a', long)]
        all: bool,
    },

    /// LoRA training (shorthand for 'lora train')
    #[command(name = "lt")]
    LoraTrain {
        /// Model reference
        model: String,
        /// Training configuration file
        #[arg(long)]
        config: Option<String>,
        /// Adapter name
        #[arg(long)]
        adapter: Option<String>,
    },

    /// Fine-tuning
    #[command(name = "ft")]
    FineTune {
        /// Model reference
        model: String,
        /// Training configuration file
        #[arg(long)]
        config: Option<String>,
    },

    /// Pre-training
    #[command(name = "pt")]
    PreTrain {
        /// Model reference
        model: String,
        /// Training configuration file
        #[arg(long)]
        config: Option<String>,
    },

    /// Serve model (shorthand for 'server')
    Serve {
        /// Model reference to pre-load (optional)
        model: Option<String>,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Run inference with a model
    Infer {
        /// Model reference (e.g., "Qwen3-4B", "qwen/qwen-2b", "model:branch")
        model: String,

        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short = 'm', long)]
        max_tokens: Option<usize>,

        /// Temperature for sampling (0.0 = deterministic)
        #[arg(short = 't', long)]
        temperature: Option<f32>,

        /// Top-p (nucleus) sampling
        #[arg(long)]
        top_p: Option<f32>,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<usize>,

        /// Stream output tokens as they're generated
        #[arg(short = 's', long)]
        stream: bool,

        /// Force re-download even if cached
        #[arg(long)]
        force_download: bool,
    },

    /// List available models
    List {
        /// Filter by git branch
        #[arg(long)]
        branch: Option<String>,
        /// Filter by git tag
        #[arg(long)]
        tag: Option<String>,
        /// Show only models with uncommitted changes
        #[arg(long)]
        dirty: bool,
        /// Verbose output with detailed info
        #[arg(short, long)]
        verbose: bool,
    },

    /// Get detailed information about a model
    Info {
        /// Model name or reference
        model: String,
        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Clone a model repository
    Clone {
        /// Git repository URL
        repo_url: String,
        /// Local name for the model
        #[arg(long)]
        name: Option<String>,
    },

    /// Push changes to remote
    Push {
        /// Model name
        model: String,
        /// Remote name (default: origin)
        remote: Option<String>,
        /// Branch to push
        branch: Option<String>,
        /// Set upstream
        #[arg(short = 'u', long)]
        set_upstream: bool,
        /// Force push
        #[arg(long)]
        force: bool,
    },

    /// Pull changes from remote
    Pull {
        /// Model name
        model: String,
        /// Remote name (default: origin)
        remote: Option<String>,
        /// Branch to pull
        branch: Option<String>,
        /// Rebase instead of merge
        #[arg(long)]
        rebase: bool,
    },

    /// Merge branches
    Merge {
        /// Model name
        model: String,
        /// Branch to merge
        branch: String,
        /// Fast-forward only
        #[arg(long)]
        ff_only: bool,
        /// No fast-forward (create merge commit)
        #[arg(long)]
        no_ff: bool,
    },
}
