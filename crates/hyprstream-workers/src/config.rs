//! Configuration types for hyprstream-workers
//!
//! Provides configuration for:
//! - WorkerService (CRI runtime)
//! - SandboxPool (warm VM management)
//! - WorkflowService (orchestration)
//! - RafsStore (image storage)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration for the workers crate
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct WorkerConfig {
    /// Pool configuration for VM management
    pub pool: PoolConfig,

    /// Image storage configuration
    pub images: ImageConfig,

    /// Workflow service configuration
    pub workflow: WorkflowConfig,

    /// Event bus configuration
    pub events: EventConfig,
}


/// Hypervisor type for VM management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HypervisorType {
    /// Cloud Hypervisor (recommended, requires cloud-hypervisor binary)
    CloudHypervisor,
    /// Dragonball (built-in VMM, no external binary required)
    #[cfg(feature = "dragonball")]
    Dragonball,
}

impl Default for HypervisorType {
    fn default() -> Self {
        Self::CloudHypervisor
    }
}

impl std::fmt::Display for HypervisorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CloudHypervisor => write!(f, "cloud-hypervisor"),
            #[cfg(feature = "dragonball")]
            Self::Dragonball => write!(f, "dragonball"),
        }
    }
}

/// Configuration for the sandbox pool (Kata VM management)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PoolConfig {
    /// Hypervisor to use for VM management
    pub hypervisor: HypervisorType,

    /// Path to hypervisor binary (for cloud-hypervisor)
    /// If empty, uses PATH lookup
    pub hypervisor_path: PathBuf,

    /// Maximum number of active sandboxes (VMs)
    pub max_sandboxes: usize,

    /// Number of pre-warmed sandboxes to keep ready
    pub warm_pool_size: usize,

    /// Path to the base VM image
    pub vm_image: PathBuf,

    /// Path to the VM kernel
    pub kernel_path: PathBuf,

    /// Path to cloud-init templates directory
    pub cloud_init_dir: PathBuf,

    /// Runtime directory for sandbox sockets and state
    pub runtime_dir: PathBuf,

    /// VM memory in MB
    pub vm_memory_mb: u64,

    /// VM CPUs
    pub vm_cpus: u32,

    /// Timeout for sandbox creation in seconds
    pub create_timeout_secs: u64,

    /// Timeout for sandbox stop in seconds
    pub stop_timeout_secs: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            hypervisor: HypervisorType::default(),
            hypervisor_path: PathBuf::new(), // Use PATH lookup
            max_sandboxes: 10,
            warm_pool_size: 2,
            vm_image: PathBuf::from("/var/lib/hyprstream/vm/rootfs.img"),
            kernel_path: PathBuf::from("/var/lib/hyprstream/vm/vmlinux"),
            cloud_init_dir: PathBuf::from("/var/lib/hyprstream/cloud-init"),
            runtime_dir: crate::paths::sandboxes_dir(),
            vm_memory_mb: 2048,
            vm_cpus: 2,
            create_timeout_secs: 60,
            stop_timeout_secs: 30,
        }
    }
}

/// Configuration for image storage (Nydus RAFS)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageConfig {
    /// Directory for chunk blobs (content-addressed)
    pub blobs_dir: PathBuf,

    /// Directory for RAFS bootstrap metadata
    pub bootstrap_dir: PathBuf,

    /// Directory for tag references
    pub refs_dir: PathBuf,

    /// Cache directory for nydus-storage blob cache
    pub cache_dir: PathBuf,

    /// Runtime directory for nydus daemon sockets
    pub runtime_dir: PathBuf,

    /// Dragonfly peer address for P2P blob fetching (e.g., "127.0.0.1:65001")
    /// When set, blob fetches route through Dragonfly for P2P distribution
    #[serde(default)]
    pub dragonfly_peer: Option<String>,

    /// Number of FUSE threads for nydus daemon
    pub fuse_threads: usize,

    /// Chunk size in bytes (default 1MB)
    pub chunk_size: usize,

    /// Enable lazy loading (on-demand chunk fetch)
    pub lazy_loading: bool,

    /// Maximum cache size in bytes
    pub cache_size_bytes: u64,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            blobs_dir: PathBuf::from("/var/lib/hyprstream/images/blobs"),
            bootstrap_dir: PathBuf::from("/var/lib/hyprstream/images/bootstrap"),
            refs_dir: PathBuf::from("/var/lib/hyprstream/images/refs"),
            cache_dir: PathBuf::from("/var/lib/hyprstream/images/cache"),
            runtime_dir: crate::paths::nydus_dir(),
            dragonfly_peer: None,
            fuse_threads: 4,
            chunk_size: 1024 * 1024, // 1MB
            lazy_loading: true,
            cache_size_bytes: 10 * 1024 * 1024 * 1024, // 10GB
        }
    }
}

/// Configuration for workflow service
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkflowConfig {
    /// Directory for workflow run logs
    pub logs_dir: PathBuf,

    /// Maximum concurrent workflow runs
    pub max_concurrent_runs: usize,

    /// Default timeout for workflow jobs in seconds
    pub job_timeout_secs: u64,

    /// Default timeout for workflow steps in seconds
    pub step_timeout_secs: u64,

    /// Enable automatic workflow discovery on repo events
    pub auto_discover: bool,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            logs_dir: PathBuf::from("/var/lib/hyprstream/workflows/logs"),
            max_concurrent_runs: 10,
            job_timeout_secs: 3600,  // 1 hour
            step_timeout_secs: 600,  // 10 minutes
            auto_discover: true,
        }
    }
}

/// Configuration for event bus
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EventConfig {
    /// ZMQ high water mark for publishers
    pub publisher_hwm: i32,

    /// ZMQ high water mark for subscribers
    pub subscriber_hwm: i32,

    /// Enable event logging for debugging
    pub log_events: bool,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            publisher_hwm: 1000,
            subscriber_hwm: 1000,
            log_events: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WorkerConfig::default();
        assert_eq!(config.pool.max_sandboxes, 10);
        assert_eq!(config.pool.warm_pool_size, 2);
        assert!(config.images.lazy_loading);
        assert!(config.workflow.auto_discover);
    }

    #[test]
    fn test_config_serialization() {
        let config = WorkerConfig::default();
        let yaml = serde_yaml::to_string(&config).expect("serialize config");
        let parsed: WorkerConfig = serde_yaml::from_str(&yaml).expect("parse config");
        assert_eq!(parsed.pool.max_sandboxes, config.pool.max_sandboxes);
    }

    #[test]
    fn test_hypervisor_type_serialization() {
        // Test default (cloud-hypervisor)
        let config = PoolConfig::default();
        assert_eq!(config.hypervisor, HypervisorType::CloudHypervisor);

        // Test serialization roundtrip
        let yaml = serde_yaml::to_string(&config).expect("serialize");
        assert!(yaml.contains("cloud-hypervisor"), "YAML should contain kebab-case hypervisor");
        let parsed: PoolConfig = serde_yaml::from_str(&yaml).expect("parse");
        assert_eq!(parsed.hypervisor, HypervisorType::CloudHypervisor);

        // Test parsing from string
        let yaml_str = "hypervisor: cloud-hypervisor\nmax_sandboxes: 5";
        let parsed: PoolConfig = serde_yaml::from_str(yaml_str).expect("parse from str");
        assert_eq!(parsed.hypervisor, HypervisorType::CloudHypervisor);
        assert_eq!(parsed.max_sandboxes, 5);
    }

    #[test]
    fn test_hypervisor_type_display() {
        assert_eq!(HypervisorType::CloudHypervisor.to_string(), "cloud-hypervisor");
        #[cfg(feature = "dragonball")]
        assert_eq!(HypervisorType::Dragonball.to_string(), "dragonball");
    }
}
