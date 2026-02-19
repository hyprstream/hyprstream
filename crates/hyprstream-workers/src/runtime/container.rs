//! Container types and lifecycle
//!
//! A Container runs within a PodSandbox (Kata VM).
//! Maps directly to Kubernetes CRI Container concept.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::worker_capnp;

use super::client::KeyValue;
// Use generated ContainerMetadata (matches what's in generated ContainerConfig/ContainerStatus)
use crate::generated::worker_client::ContainerMetadata;

/// Container state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ContainerState {
    /// Container is being created
    ContainerCreated,
    /// Container is running
    ContainerRunning,
    /// Container has exited
    ContainerExited,
    /// Unknown state
    #[default]
    ContainerUnknown,
}

impl From<ContainerState> for worker_capnp::ContainerState {
    fn from(s: ContainerState) -> Self {
        match s {
            ContainerState::ContainerCreated => worker_capnp::ContainerState::ContainerCreated,
            ContainerState::ContainerRunning => worker_capnp::ContainerState::ContainerRunning,
            ContainerState::ContainerExited => worker_capnp::ContainerState::ContainerExited,
            ContainerState::ContainerUnknown => worker_capnp::ContainerState::ContainerUnknown,
        }
    }
}

impl From<worker_capnp::ContainerState> for ContainerState {
    fn from(s: worker_capnp::ContainerState) -> Self {
        match s {
            worker_capnp::ContainerState::ContainerCreated => ContainerState::ContainerCreated,
            worker_capnp::ContainerState::ContainerRunning => ContainerState::ContainerRunning,
            worker_capnp::ContainerState::ContainerExited => ContainerState::ContainerExited,
            worker_capnp::ContainerState::ContainerUnknown => ContainerState::ContainerUnknown,
        }
    }
}

// DTO types (ContainerConfig, ImageSpec, Mount, Device, ContainerStatus, etc.)
// are now imported from generated types via super::client

/// Runtime representation of a container
#[derive(Debug, Clone)]
pub struct Container {
    /// Unique container ID
    pub id: String,

    /// Pod sandbox ID this container belongs to
    pub pod_sandbox_id: String,

    /// Container metadata
    pub metadata: ContainerMetadata,

    /// Current state
    pub state: ContainerState,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Image reference (uses generated ImageSpec type)
    pub image: super::client::ImageSpec,

    /// Labels
    pub labels: Vec<KeyValue>,

    /// Annotations
    pub annotations: Vec<KeyValue>,

    // Internal fields for process management
    /// Container process ID (if running)
    pub(crate) pid: Option<u32>,

    /// Exit code (if exited)
    pub(crate) exit_code: Option<i32>,
}

impl Container {
    /// Create a new container from configuration (uses generated ContainerConfig)
    pub fn new(id: String, pod_sandbox_id: String, config: &super::client::ContainerConfig) -> Self {
        Self {
            id,
            pod_sandbox_id,
            metadata: config.metadata.clone(),
            state: ContainerState::ContainerCreated,
            created_at: Utc::now(),
            image: config.image.clone(),
            labels: config.labels.clone(),
            annotations: config.annotations.clone(),
            pid: None,
            exit_code: None,
        }
    }

    /// Create a container from RPC response info
    ///
    /// Used when parsing list_containers response from WorkerService.
    #[allow(clippy::too_many_arguments)]
    pub fn from_info(
        id: String,
        pod_sandbox_id: String,
        metadata: ContainerMetadata,
        image: super::client::ImageSpec,
        state: ContainerState,
        created_at: DateTime<Utc>,
        labels: Vec<KeyValue>,
        annotations: Vec<KeyValue>,
    ) -> Self {
        Self {
            id,
            pod_sandbox_id,
            metadata,
            state,
            created_at,
            image,
            labels,
            annotations,
            pid: None,
            exit_code: None,
        }
    }

    /// Check if container is running
    pub fn is_running(&self) -> bool {
        self.state == ContainerState::ContainerRunning
    }
}
