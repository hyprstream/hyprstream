//! hyprstream-workers: Isolated workload execution using Kata Containers
//!
//! This crate provides CRI-aligned (Kubernetes Container Runtime Interface) services
//! for running isolated workloads in Kata Container VMs with GitHub Actions-compatible
//! workflow orchestration.
//!
//! # Architecture
//!
//! Two main services:
//!
//! - **WorkerService**: CRI-aligned RuntimeService + ImageService
//!   - PodSandbox = Kata VM (maps to CRI sandbox concept)
//!   - Container = OCI container within VM
//!   - ImageService backed by Nydus RAFS for chunk-level deduplication
//!
//! - **WorkflowService**: High-level workflow orchestration
//!   - Discovers `.github/workflows/*.yml` from RegistryService repos
//!   - Subscribes to event bus for automatic triggers
//!   - Spawns pods/containers via WorkerService
//!
//! # Event Bus
//!
//! Uses XPUB/XSUB proxy pattern with repo-scoped topics:
//! - `git2db.{repo_id}.push` - Repository push events
//! - `training.{model_id}.completed` - Training completion events
//! - `metrics.{model_id}.breach` - Threshold breach events
//!
//! # Example
//!
//! ```ignore
//! use hyprstream_workers::{WorkerService, RuntimeService, ImageService};
//!
//! // Create worker service
//! let worker = WorkerService::new(config).await?;
//!
//! // Pull an image
//! let image_id = worker.pull_image(&ImageSpec::new("alpine:latest"), None).await?;
//!
//! // Create a pod sandbox (Kata VM)
//! let sandbox_id = worker.run_pod_sandbox(&PodSandboxConfig::default()).await?;
//!
//! // Create and start a container
//! let container_id = worker.create_container(&sandbox_id, &container_config, &sandbox_config).await?;
//! worker.start_container(&container_id).await?;
//! ```

pub mod config;
pub mod error;

pub mod runtime;
pub mod image;
pub mod workflow;
pub mod events;

// Re-export main types
pub use config::{HypervisorType, ImageConfig, PoolConfig, WorkerConfig, WorkflowConfig};
pub use error::WorkerError;

// Re-export service types
pub use runtime::{WorkerService, RuntimeService, RuntimeZmq};
pub use image::{ImageService, ImageZmq, RafsStore};
pub use workflow::{WorkflowService, WorkflowOps, WorkflowZmq};
pub use events::{start_event_service, EventServiceHandle, EventPublisher, EventSubscriber, endpoints};

/// Generated Cap'n Proto code
#[allow(dead_code)]
#[allow(clippy::all)]
pub mod workers_capnp {
    include!(concat!(env!("OUT_DIR"), "/workers_capnp.rs"));
}
