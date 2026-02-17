pub mod chat;
pub mod config;
pub mod flight;
pub mod git;
pub mod policy;
pub mod training;
pub mod worker;

pub use flight::FlightArgs;
pub use git::{GitAction, GitCommand};
pub use policy::{PolicyCommand, TokenCommand};
pub use training::{TrainingAction, TrainingCommand};
pub use worker::{ImageCommand, WorkerAction};

use clap::{Subcommand, ValueEnum};

use crate::runtime::kv_quant::KVQuantType;

/// KV cache quantization type for inference
///
/// Quantization reduces GPU memory usage at a slight quality cost:
/// - `none`: Full precision (default)
/// - `int8`: 50% memory savings, minimal quality loss
/// - `nf4`: 75% memory savings, best quality for 4-bit
/// - `fp4`: 75% memory savings, standard 4-bit quantization
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KVQuantArg {
    /// No quantization (full precision FP16/BF16)
    #[default]
    None,
    /// 8-bit integer quantization (~50% memory savings)
    Int8,
    /// 4-bit NormalFloat quantization (~75% memory savings)
    Nf4,
    /// 4-bit FloatingPoint quantization (~75% memory savings)
    Fp4,
}

impl From<KVQuantArg> for KVQuantType {
    fn from(arg: KVQuantArg) -> Self {
        match arg {
            KVQuantArg::None => KVQuantType::None,
            KVQuantArg::Int8 => KVQuantType::Int8,
            KVQuantArg::Nf4 => KVQuantType::Nf4,
            KVQuantArg::Fp4 => KVQuantType::Fp4,
        }
    }
}

/// Overall execution mode for the hyprstream CLI and services
///
/// This determines how services are spawned and managed:
/// - **Inproc**: Single process, all services in-process, inproc:// ZMQ transport (mobile/embedded)
/// - **IpcStandalone**: Multiple processes spawned directly, ipc:// ZMQ transport (default)
/// - **IpcSystemd**: Multiple systemd services with socket activation, ipc:// ZMQ transport
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionMode {
    /// In-process mode with all services in a single process (mobile/embedded builds)
    Inproc,

    /// Standalone mode with forked processes (default for desktop/server/containers)
    /// Services spawned via ProcessSpawner with StandaloneBackend
    #[default]
    IpcStandalone,

    /// Systemd mode with socket-activated services
    /// Services managed by systemd, started on-demand via socket activation
    IpcSystemd,
}

impl ExecutionMode {
    /// Detect best execution mode based on system capabilities
    ///
    /// Returns:
    /// - `IpcSystemd` if systemd is available
    /// - `IpcStandalone` otherwise (default for desktop/server/containers)
    ///
    /// Note: `Inproc` mode is reserved for special builds (mobile/embedded)
    /// and must be explicitly requested via CLI flag (future feature).
    pub fn detect() -> Self {
        #[cfg(feature = "systemd")]
        {
            if hyprstream_rpc::has_systemd() {
                return ExecutionMode::IpcSystemd;
            }
        }
        // Default to standalone IPC mode (services as separate processes)
        ExecutionMode::IpcStandalone
    }

    /// Get the EndpointMode for this execution mode
    pub fn endpoint_mode(&self) -> hyprstream_rpc::registry::EndpointMode {
        match self {
            ExecutionMode::Inproc => hyprstream_rpc::registry::EndpointMode::Inproc,
            ExecutionMode::IpcStandalone | ExecutionMode::IpcSystemd => {
                hyprstream_rpc::registry::EndpointMode::Ipc
            }
        }
    }

    /// Whether this mode uses IPC sockets
    pub fn uses_ipc(&self) -> bool {
        matches!(self, ExecutionMode::IpcStandalone | ExecutionMode::IpcSystemd)
    }

    /// Whether this mode uses systemd
    pub fn uses_systemd(&self) -> bool {
        matches!(self, ExecutionMode::IpcSystemd)
    }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Inproc => write!(f, "inproc"),
            ExecutionMode::IpcStandalone => write!(f, "ipc-standalone"),
            ExecutionMode::IpcSystemd => write!(f, "ipc-systemd"),
        }
    }
}

/// Top-level commands for the hyprstream CLI.
///
/// Schema-driven commands (registry, model, inference, policy) are built
/// at runtime from Cap'n Proto schema metadata — see `schema_cli.rs`.
///
/// This enum only contains commands that require hand-coded dispatch:
/// `quick` (orchestrated workflows), `worker`, `service`, and `flight`.
#[derive(Subcommand)]
pub enum Commands {
    /// Quick workflows — multi-step convenience commands
    ///
    /// These commands involve multiple RPC calls, interactive prompts,
    /// progress bars, or other complex orchestration.
    Quick {
        #[command(subcommand)]
        command: super::quick::QuickCommand,
    },

    /// Flight SQL client to query datasets
    Flight(FlightArgs),

    /// Service management and lifecycle commands
    ///
    /// Install, manage, and control hyprstream services. Supports both systemd
    /// (when available) and standalone process management.
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

/// Service management actions
#[derive(Subcommand)]
pub enum ServiceAction {
    /// Install systemd units and start all services
    ///
    /// Creates systemd user unit files and starts all configured services.
    /// Idempotent - safe to run multiple times.
    Install {
        /// Operate on specific services only (comma-separated)
        #[arg(long, short = 's', value_delimiter = ',')]
        services: Option<Vec<String>>,
    },

    /// Upgrade units with current binary path and restart services
    ///
    /// Useful after updating the hyprstream binary or moving the AppImage.
    /// Updates unit files to point to the current executable location.
    Upgrade {
        /// Operate on specific services only
        #[arg(long, short = 's', value_delimiter = ',')]
        services: Option<Vec<String>>,
    },

    /// Full reset: stop, uninstall, reinstall, and start services
    ///
    /// Use this when services are in a broken state or after major changes.
    Reinstall {
        /// Operate on specific services only
        #[arg(long, short = 's', value_delimiter = ',')]
        services: Option<Vec<String>>,
    },

    /// Stop and remove systemd units
    Uninstall {
        /// Operate on specific services only
        #[arg(long, short = 's', value_delimiter = ',')]
        services: Option<Vec<String>>,
    },

    /// Start services (via systemd or direct spawn)
    ///
    /// Without --foreground: Uses systemd if available, else spawns background process.
    /// With --foreground: Runs service in foreground (used by systemd ExecStart).
    /// With --daemon: Bypasses systemd and spawns process directly.
    Start {
        /// Service name (all services if omitted)
        name: Option<String>,

        /// Run in foreground instead of background
        ///
        /// Used by systemd unit files (ExecStart) to run the service process.
        /// When running manually, keeps the service attached to the terminal.
        #[arg(long, short = 'f', visible_alias = "fg")]
        foreground: bool,

        /// Force daemon mode (bypass systemd, spawn process directly)
        ///
        /// Use this to run services as standalone daemons even when systemd is available.
        #[arg(long, short = 'd')]
        daemon: bool,

        /// Use IPC sockets for distributed mode
        #[arg(long)]
        ipc: bool,

        /// Callback endpoint for inference service callback mode
        #[arg(long)]
        callback: Option<String>,
    },

    /// Stop services
    Stop {
        /// Service name (all services if omitted)
        name: Option<String>,
    },

    /// Show service status
    ///
    /// Displays running services, unit file locations, and execution mode.
    Status {
        /// Show verbose output including unit file contents
        #[arg(long, short = 'v')]
        verbose: bool,
    },
}
