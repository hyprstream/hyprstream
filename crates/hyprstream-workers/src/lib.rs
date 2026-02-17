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
//! - **WorkerService**: CRI-aligned RuntimeClient + ImageClient
//!   - PodSandbox = Kata VM (maps to CRI sandbox concept)
//!   - Container = OCI container within VM
//!   - ImageClient backed by Nydus RAFS for chunk-level deduplication
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
//! use hyprstream_workers::{WorkerService, RuntimeClient, ImageClient};
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

// Re-export paths from hyprstream-rpc
pub use hyprstream_rpc::paths;

pub mod runtime;
pub mod image;
pub mod workflow;
pub mod events;

#[cfg(feature = "dbus")]
pub mod dbus;

// Re-export main types
pub use config::{HypervisorType, ImageConfig, PoolConfig, WorkerConfig, WorkflowConfig};
pub use error::WorkerError;

// Re-export service types
pub use runtime::{WorkerService, RuntimeClient};
pub use image::{ImageClient, RafsStore};
pub use workflow::{WorkflowService, WorkflowClient};
pub use events::{
    // Spawner types (new API)
    ProxyService, ServiceSpawner, SpawnedService,
    // Publisher/Subscriber
    EventPublisher, EventSubscriber, endpoints,
    // Endpoint configuration
    EndpointMode,
    // Event types
    WorkerEvent, ReceivedEvent,
    SandboxStarted, SandboxStopped, ContainerStarted, ContainerStopped,
    serialize_sandbox_started, serialize_sandbox_stopped,
    serialize_container_started, serialize_container_stopped,
    // Inproc endpoint constants
    EVENTS_PUB, EVENTS_SUB,
};

// Re-export annotations_capnp and streaming_capnp from hyprstream-rpc (compiled once, shared by all crates)
pub use hyprstream_rpc::annotations_capnp;
pub use hyprstream_rpc::streaming_capnp;

/// Generated Cap'n Proto code for resource-scoped worker schema
pub mod worker_capnp {
    #![allow(dead_code, clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::match_same_arms, clippy::semicolon_if_nothing_returned)]
    #![allow(clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/worker_capnp.rs"));
}

/// Generated Cap'n Proto code for workflow service schema
pub mod workflow_capnp {
    #![allow(dead_code, clippy::all, clippy::unwrap_used, clippy::expect_used)]
    #![allow(clippy::match_same_arms, clippy::semicolon_if_nothing_returned)]
    #![allow(clippy::doc_markdown, clippy::indexing_slicing)]
    #![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    include!(concat!(env!("OUT_DIR"), "/workflow_capnp.rs"));
}

/// Generated RPC dispatch code from `generate_rpc_service!` proc macro.
///
/// Provides `WorkerHandler`/`dispatch_worker` and `WorkflowHandler`/`dispatch_workflow`.
pub mod generated;
