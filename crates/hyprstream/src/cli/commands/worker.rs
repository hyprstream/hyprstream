//! Worker CLI commands
//!
//! Provides commands for managing sandboxes (Kata VMs), containers, and images:
//! - `worker list` - List sandboxes and containers
//! - `worker run` - Run a container in a sandbox
//! - `worker stop` - Stop a sandbox or container
//! - `worker rm` - Remove a sandbox or container
//! - `worker images` - Image management subcommands

use clap::{Args, Subcommand};

/// Worker management commands
#[derive(Args)]
pub struct WorkerCommand {
    #[command(subcommand)]
    pub action: WorkerAction,
}

/// Worker actions
#[derive(Subcommand)]
pub enum WorkerAction {
    /// List sandboxes and containers
    List {
        /// Filter by sandbox ID
        #[arg(long)]
        sandbox: Option<String>,
        /// Show only containers (not sandboxes)
        #[arg(long)]
        containers: bool,
        /// Show only sandboxes (not containers)
        #[arg(long)]
        sandboxes: bool,
        /// Filter by state (ready, not-ready, running, exited)
        #[arg(long)]
        state: Option<String>,
        /// Show all details
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run a container in a sandbox
    Run {
        /// Image reference (e.g., docker.io/library/alpine:latest)
        image: String,
        /// Command to run (optional, uses image default)
        #[arg(trailing_var_arg = true)]
        command: Vec<String>,
        /// Sandbox ID to use (creates new if not specified)
        #[arg(long)]
        sandbox: Option<String>,
        /// Container name
        #[arg(long)]
        name: Option<String>,
        /// Environment variables (KEY=VALUE)
        #[arg(short, long)]
        env: Vec<String>,
        /// Working directory
        #[arg(short, long)]
        workdir: Option<String>,
        /// Run in detached mode
        #[arg(short, long)]
        detach: bool,
        /// Remove container after exit
        #[arg(long)]
        rm: bool,
    },

    /// Stop a sandbox or container
    Stop {
        /// Sandbox or container ID
        id: String,
        /// Timeout in seconds (default: 30)
        #[arg(short, long, default_value = "30")]
        timeout: i64,
        /// Force stop (SIGKILL)
        #[arg(short, long)]
        force: bool,
    },

    /// Start a stopped container
    Start {
        /// Container ID
        container_id: String,
    },

    /// Restart a sandbox or container
    Restart {
        /// Sandbox or container ID
        id: String,
        /// Timeout for stop phase
        #[arg(short, long, default_value = "30")]
        timeout: i64,
    },

    /// Remove a sandbox or container
    Rm {
        /// Sandbox or container IDs
        ids: Vec<String>,
        /// Force removal (stop if running)
        #[arg(short, long)]
        force: bool,
        /// Remove associated volumes
        #[arg(short, long)]
        volumes: bool,
    },

    /// Get detailed status
    Status {
        /// Sandbox or container ID
        id: String,
        /// Show all details
        #[arg(short, long)]
        verbose: bool,
    },

    /// Get resource usage statistics
    Stats {
        /// Sandbox or container IDs (all if empty)
        ids: Vec<String>,
        /// Continuous output
        #[arg(long)]
        stream: bool,
        /// Don't print header
        #[arg(long)]
        no_header: bool,
    },

    /// Execute command in container
    Exec {
        /// Container ID
        container_id: String,
        /// Command to execute
        #[arg(trailing_var_arg = true)]
        command: Vec<String>,
        /// Timeout in seconds (default: 60)
        #[arg(short, long, default_value = "60")]
        timeout: i64,
    },

    /// Attach to container terminal (tmux-like streaming I/O)
    Terminal {
        /// Container ID
        container_id: String,
        /// Detach key sequence (default: ctrl-])
        #[arg(long, default_value = "ctrl-]")]
        detach_keys: String,
    },

    /// Image management commands
    #[command(subcommand)]
    Images(ImageCommand),
}

/// Image management subcommands
#[derive(Subcommand)]
pub enum ImageCommand {
    /// List cached images
    List {
        /// Show all details
        #[arg(short, long)]
        verbose: bool,
        /// Filter by reference pattern
        #[arg(long)]
        filter: Option<String>,
    },

    /// Pull an image from registry
    Pull {
        /// Image reference (e.g., docker.io/library/alpine:latest)
        image: String,
        /// Registry username
        #[arg(short, long)]
        username: Option<String>,
        /// Registry password
        #[arg(short, long)]
        password: Option<String>,
    },

    /// Remove an image
    Rm {
        /// Image references to remove
        images: Vec<String>,
        /// Force removal even if in use
        #[arg(short, long)]
        force: bool,
    },

    /// Show image filesystem usage
    Df,
}
