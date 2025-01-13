//! System resource monitoring functionality.

use std::time::Duration;
use crate::error::Result;

/// Trait for system resource monitoring
#[async_trait::async_trait]
pub trait ResourceMonitor: Send + Sync {
    /// Start monitoring resources
    async fn start(&mut self) -> Result<()>;
    
    /// Stop monitoring resources
    async fn stop(&mut self) -> Result<()>;
    
    /// Get CPU utilization percentage
    async fn get_cpu_utilization(&self) -> Result<f32>;
    
    /// Get memory usage in bytes
    async fn get_memory_usage(&self) -> Result<u64>;
    
    /// Get available memory in bytes
    async fn get_available_memory(&self) -> Result<u64>;
    
    /// Get disk usage statistics
    async fn get_disk_usage(&self) -> Result<DiskStats>;
}

/// Disk usage statistics
#[derive(Debug, Clone)]
pub struct DiskStats {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub read_bytes_per_sec: f64,
    pub write_bytes_per_sec: f64,
}
