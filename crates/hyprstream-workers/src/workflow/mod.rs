//! WorkflowService - GitHub Actions compatible workflow orchestration
//!
//! Discovers workflows from git repositories, subscribes to events,
//! and spawns containers via WorkerService.
//!
//! # Architecture
//!
//! ```text
//! WorkflowService (ZmqService)
//!     │
//!     ├── scan_repo()       → Discover .github/workflows/*.yml
//!     ├── subscribe()       → Register event triggers
//!     ├── dispatch()        → Manual workflow trigger
//!     └── get_run()         → Query run status
//!     │
//!     └── WorkflowRunner
//!           ├── Creates PodSandbox via WorkerService
//!           └── Executes steps as containers
//! ```

mod service;
mod client;
mod parser;
mod triggers;
mod subscription;
mod runner;

pub use service::WorkflowService;
pub use client::{WorkflowClient, WorkflowZmq};
pub use parser::{Workflow, Job, Step};
pub use triggers::{EventTrigger, EventHandler, HandlerResult};
pub use subscription::WorkflowSubscription;
pub use runner::WorkflowRunner;

/// Workflow ID
pub type WorkflowId = String;

/// Run ID
pub type RunId = String;

/// Subscription ID
pub type SubscriptionId = String;
