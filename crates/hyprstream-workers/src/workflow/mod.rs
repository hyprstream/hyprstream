//! WorkflowService - GitHub Actions compatible workflow orchestration
//!
//! Discovers workflows from git repositories, subscribes to events, and runs
//! jobs as scheduled workloads (#527): each job's `runs_on:` + `resources:`
//! route through the #525 P2 admission engine (`SandboxPool::acquire`) when
//! the job requests isolation, or stay on the existing in-process VFS/Tcl
//! execution path otherwise (see `workflow::scheduler` for the mapping).
//! `WorkflowRunner` talks to `SandboxPool` directly — there is no
//! `WorkerService` dependency in this crate's workflow engine.
//!
//! # Architecture
//!
//! ```text
//! WorkflowService (RequestService)
//!     │
//!     ├── scan_repo()       → Discover .github/workflows/*.yml
//!     ├── subscribe()       → Register event triggers
//!     ├── dispatch()        → Manual workflow trigger
//!     └── get_run()         → Query run status
//!     │
//!     └── WorkflowRunner
//!           ├── JobScheduler → SandboxPool::acquire (isolated jobs) or
//!           │                  in-proc (`IN_PROC_LABELS`) — #527
//!           └── Executes steps via VFS `/bin/` ctl calls + TclShell
//! ```

pub mod adapter;
pub mod gh_adapter;
mod parser;
mod runner;
pub mod scheduler;
mod service;
mod subscription;
mod triggers;

pub use adapter::SubscriberAdapter;
pub use gh_adapter::GitHubActionsAdapter;
pub use parser::{Job, JobResources, ParseMode, Step, Workflow};
pub use runner::WorkflowRunner;
pub use scheduler::{JobScheduler, Placement};
pub use service::WorkflowService;
pub use subscription::WorkflowSubscription;
pub use triggers::{EventHandler, EventTrigger, HandlerResult};

/// Workflow ID
pub type WorkflowId = String;

/// Run ID
pub type RunId = String;

/// Subscription ID
pub type SubscriptionId = String;
