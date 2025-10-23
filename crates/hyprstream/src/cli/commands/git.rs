//! Git-style model management commands for Phase 1
//!
//! Provides essential git operations for training workflows:
//! - branch: Create new branches for experiments
//! - checkout: Switch between branches/tags/commits
//! - status: Show current branch and changes
//! - commit: Save changes to git history

use clap::{Args, Subcommand};

/// Git-style model management commands
#[derive(Args)]
pub struct GitCommand {
    #[command(subcommand)]
    pub action: GitAction,
}

/// Git management actions
#[derive(Subcommand)]
pub enum GitAction {
    /// Create a new branch
    Branch {
        /// Model name or reference
        model: String,

        /// Name for the new branch
        name: String,

        /// Create from specific ref (default: current HEAD)
        #[arg(long)]
        from: Option<String>,
    },

    /// Switch branches or checkout specific commit/tag
    Checkout {
        /// Model reference (e.g., "llama3:main", "llama3:v2.0")
        model_ref: String,

        /// Create new branch if it doesn't exist
        #[arg(short = 'b')]
        create_branch: bool,

        /// Force checkout, discarding local changes
        #[arg(long)]
        force: bool,
    },

    /// Show working tree status
    Status {
        /// Model name (optional, defaults to all models)
        model: Option<String>,

        /// Show verbose output with file changes
        #[arg(short, long)]
        verbose: bool,

        /// Show in short format
        #[arg(short, long)]
        short: bool,
    },

    /// Record changes to the repository
    Commit {
        /// Model name
        model: String,

        /// Commit message
        #[arg(short, long)]
        message: String,

        /// Automatically stage all tracked files
        #[arg(short = 'a', long)]
        all: bool,

        /// Amend previous commit
        #[arg(long)]
        amend: bool,
    },

    /// Stage files for commit (Phase 1.5)
    Add {
        /// Model name
        model: String,

        /// Files to stage (or "." for all)
        #[arg(required = true)]
        files: Vec<String>,
    },

    /// Basic push support (Phase 1.5)
    Push {
        /// Model reference
        model: String,

        /// Remote name (default: origin)
        remote: Option<String>,

        /// Branch to push
        branch: Option<String>,

        /// Set upstream tracking
        #[arg(short = 'u', long)]
        set_upstream: bool,
    },

    /// Basic pull support (Phase 1.5)
    Pull {
        /// Model reference
        model: String,

        /// Remote name (default: origin)
        remote: Option<String>,

        /// Branch to pull
        branch: Option<String>,

        /// Rebase instead of merge
        #[arg(long)]
        rebase: bool,
    },

    /// Remove a model from the system
    Remove {
        /// Model name to remove (e.g., "Qwen3-4B")
        model: String,

        /// Force removal without confirmation
        #[arg(short, long)]
        force: bool,

        /// Remove only from registry, keep files
        #[arg(long)]
        registry_only: bool,

        /// Remove files only, keep registry entry
        #[arg(long)]
        files_only: bool,
    },
}
