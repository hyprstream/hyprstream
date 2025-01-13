//! Performance monitoring and metrics collection infrastructure.

use std::time::Duration;
use tokio::time::Instant;
use crate::error::Result;

/// Trait for collecting and monitoring performance metrics
#[async_trait::async_trait]
pub trait MetricsCollector: Send + Sync {
    /// Start collecting metrics
    async fn start(&mut self) -> Result<()>;
    
    /// Stop collecting metrics
    async fn stop(&mut self) -> Result<()>;
    
    /// Record a timing measurement
    async fn record_timing(&mut self, operation: &str, duration: Duration) -> Result<()>;
    
    /// Record a counter measurement
    async fn record_counter(&mut self, name: &str, value: u64) -> Result<()>;
    
    /// Get current metrics snapshot
    async fn get_metrics(&self) -> Result<Vec<(String, f64)>>;
}

/// Basic performance measurement utility
pub struct PerformanceTimer {
    start: Instant,
    operation: String,
}

impl PerformanceTimer {
    pub fn new(operation: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
