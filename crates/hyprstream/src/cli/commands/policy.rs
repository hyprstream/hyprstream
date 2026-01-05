//! Policy management CLI commands
//!
//! Cisco-inspired transactional UX for managing access control policies.
//!
//! Policy states:
//! - **Running** = Current active (HEAD of .registry/policies/)
//! - **Draft** = Uncommitted changes
//! - **History** = Previous versions (HEAD~n)

use clap::Subcommand;

/// Policy management subcommands
#[derive(Subcommand)]
pub enum PolicyCommand {
    /// Display the running policy (current active configuration)
    Show {
        /// Show policy in raw CSV format instead of formatted table
        #[arg(long)]
        raw: bool,
    },

    /// Show policy commit history
    History {
        /// Number of commits to show
        #[arg(short = 'n', long, default_value = "10")]
        count: usize,
        /// Show one commit per line
        #[arg(long)]
        oneline: bool,
    },

    /// Open policy in $VISUAL/$EDITOR for editing
    Edit,

    /// Show diff between draft (uncommitted changes) and running policy
    Diff {
        /// Compare against specific commit/ref instead of HEAD
        #[arg(long)]
        against: Option<String>,
    },

    /// Commit draft changes to running policy
    Apply {
        /// Preview changes without committing
        #[arg(long)]
        dry_run: bool,
        /// Commit message (auto-generated if not provided)
        #[arg(short = 'm', long)]
        message: Option<String>,
    },

    /// Revert to a previous policy version
    Rollback {
        /// Git ref to rollback to (default: HEAD~1)
        #[arg(default_value = "HEAD~1")]
        git_ref: String,
        /// Preview changes without applying
        #[arg(long)]
        dry_run: bool,
    },

    /// Test if a user has permission for an action on a resource
    Check {
        /// User to check permissions for
        user: String,
        /// Resource to check (e.g., "model:qwen3-small")
        resource: String,
        /// Action to check (infer, train, query, write, serve, manage)
        action: String,
    },

    /// API token management (OpenAI-compatible Bearer authentication)
    Token {
        #[command(subcommand)]
        command: TokenCommand,
    },

    /// Apply a built-in policy template
    ApplyTemplate {
        /// Template name: local, public-inference, public-read
        template: String,
        /// Preview changes without applying
        #[arg(long)]
        dry_run: bool,
    },

    /// List available policy templates
    ListTemplates,
}

/// Token management subcommands
#[derive(Subcommand)]
pub enum TokenCommand {
    /// Create a new API token for a user
    Create {
        /// User the token authenticates as
        user: String,

        /// Human-readable name for the token (e.g., "dev-laptop", "ci-pipeline")
        #[arg(long, short = 'n')]
        name: Option<String>,

        /// Token expiration (e.g., "30d", "90d", "1y", "never")
        #[arg(long, short = 'e', default_value = "1y")]
        expires: String,

        /// Limit token to specific resources (repeatable, e.g., --scope model:qwen3-small)
        #[arg(long, short = 's')]
        scope: Vec<String>,

        /// Create an admin token (hypr_admin_ prefix)
        #[arg(long)]
        admin: bool,
    },

    /// List all API tokens
    List {
        /// Filter by user
        #[arg(long, short = 'u')]
        user: Option<String>,
    },

    /// Revoke an API token
    Revoke {
        /// Token to revoke (full token or prefix)
        token: Option<String>,

        /// Revoke by user and name instead of token
        #[arg(long, short = 'n', requires = "revoke_user")]
        name: Option<String>,

        /// User whose token to revoke (use with --name)
        #[arg(long, short = 'u')]
        revoke_user: Option<String>,

        /// Skip confirmation prompt
        #[arg(long, short = 'f')]
        force: bool,
    },
}
