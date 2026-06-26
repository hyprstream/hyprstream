//! RuntimeClient trait and ZMQ client
//!
//! Client-side trait for CRI RuntimeService (`runtime.v1`) for future kubelet compatibility.
//! Response types use generated types from the Cap'n Proto schema directly.

use std::collections::HashMap; // Only for StatusResponse.info (local-only, not serialized)

// Re-export generated wire types as the canonical response/stats types.
// These are produced by `generate_rpc_service!("worker")` in crate::generated::worker_client.
pub use crate::generated::worker_client::{
    // Generated client
    WorkerClient,
    // Response types
    VersionInfo, RuntimeStatus, RuntimeCondition,
    PodSandboxStatusResponse, ContainerStatusResponse,
    ExecSyncResult,
    // Stats types
    PodSandboxStats, PodSandboxAttributes,
    LinuxPodSandboxStats,
    ContainerStats, ContainerAttributes,
    CpuUsage, MemoryUsage, NetworkUsage,
    NetworkInterfaceUsage, ProcessUsage,
    FilesystemUsage, FilesystemIdentifier,
    // Common types
    KeyValue, Timestamp,
    // Info types for list responses
    PodSandboxInfo, ContainerInfo,
    // Scoped status types (generated)
    PodSandboxStatus, ContainerStatus,
    PodSandboxNetworkStatus,
    // Generated enum types (clean OCI-aligned names)
    PodSandboxState, ContainerState,
    // Request config types (for client RPC calls)
    PodSandboxConfig, ContainerConfig, ImageSpec,
    PodSandboxMetadata, ContainerMetadata,
    DNSConfig, PortMapping, LinuxPodSandboxConfig,
    LinuxSandboxSecurityContext,
    Mount, Device, LinuxContainerConfig,
    LinuxContainerSecurityContext, Capability,
    AuthConfig, StreamInfo,
    ImageInfo, ImageStatusResult,
    LinuxContainerResources,
    // Filter types (generated, with ToCapnp/FromCapnp)
    PodSandboxFilter, ContainerFilter,
    PodSandboxStatsFilter, ContainerStatsFilter,
};

// ─────────────────────────────────────────────────────────────────────────────
// Local Composite Types (not in schema, used internally)
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime status response (local composite, not in schema).
///
/// Bundles the generated RuntimeStatus with verbose diagnostic info.
/// The `info` HashMap is NOT serialized over the wire — it's only used locally.
#[derive(Debug, Clone)]
pub struct StatusResponse {
    /// Overall runtime status (generated wire type)
    pub status: RuntimeStatus,
    /// Additional info (if verbose) — local only, not serialized
    pub info: HashMap<String, String>,
}
