//! GPU performance metrics collection and monitoring.
//! 
//! This module integrates with the GPU acceleration system to collect and track:
//! - Device utilization
//! - Memory usage
//! - Operation throughput
//! - Temperature monitoring
//! - Power consumption (where available)

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::error::Result;
use crate::gpu::GpuContext;
use crate::metrics::MetricRecord;
use std::time::{Duration, Instant};

use burn::prelude::Backend;

/// GPU metrics collector
pub struct GpuMetricsCollector<B: Backend> {
    /// GPU context
    context: Arc<GpuContext<B>>,
    /// Collection interval
    interval: Duration,
    /// Last collection time
    last_collection: RwLock<Instant>,
    /// Historical metrics
    history: RwLock<Vec<GpuMetricSnapshot>>,
    /// Maximum history size
    max_history: usize,
}

/// Point-in-time GPU metrics
#[derive(Debug, Clone)]
pub struct GpuMetricSnapshot {
    /// Timestamp
    pub timestamp: i64,
    /// GPU utilization (0-100)
    pub utilization: f32,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Memory utilization percentage
    pub memory_utilization: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Operations per second
    pub ops_per_second: f64,
    /// Power usage in watts (if available)
    pub power_usage: Option<f32>,
}

impl<B: Backend> GpuMetricsCollector<B> {
    /// Create a new GPU metrics collector
    pub fn new(context: Arc<GpuContext<B>>, interval: Duration, max_history: usize) -> Self {
        Self {
            context,
            interval,
            last_collection: RwLock::new(Instant::now()),
            history: RwLock::new(Vec::with_capacity(max_history)),
            max_history,
        }
    }

    /// Start metrics collection
    pub async fn start(&self) -> Result<()> {
        let collector = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(collector.interval).await;
                if let Err(e) = collector.collect_metrics().await {
                    tracing::error!("GPU metrics collection error: {}", e);
                }
            }
        });
        Ok(())
    }

    /// Collect current metrics
    pub async fn collect_metrics(&self) -> Result<()> {
        // Update GPU metrics
        self.context.update_metrics().await?;
        let metrics = self.context.get_metrics().await;

        // Create snapshot
        let snapshot = GpuMetricSnapshot {
            timestamp: chrono::Utc::now().timestamp(),
            utilization: metrics.utilization,
            available_memory: metrics.available_memory,
            total_memory: metrics.total_memory,
            memory_utilization: 100.0 * (metrics.total_memory - metrics.available_memory) as f32 
                / metrics.total_memory as f32,
            temperature: metrics.temperature,
            ops_per_second: metrics.ops_per_second,
            power_usage: None, // TODO: Add power monitoring
        };

        // Update history
        let mut history = self.history.write().await;
        history.push(snapshot);

        // Trim history if needed
        if history.len() > self.max_history {
            history.drain(0..history.len() - self.max_history);
        }

        // Update last collection time
        *self.last_collection.write().await = Instant::now();

        Ok(())
    }

    /// Get current metrics
    pub async fn get_current_metrics(&self) -> Result<GpuMetricSnapshot> {
        let history = self.history.read().await;
        history.last().cloned().ok_or_else(|| {
            crate::error::Error::Metrics("No GPU metrics available".into())
        })
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self) -> Vec<GpuMetricSnapshot> {
        self.history.read().await.clone()
    }

    /// Convert GPU metrics to standard metric records
    pub async fn to_metric_records(&self) -> Vec<MetricRecord> {
        let mut records = Vec::new();
        let snapshot = match self.get_current_metrics().await {
            Ok(snap) => snap,
            Err(_) => return records,
        };

        // GPU utilization
        records.push(MetricRecord {
            metric_id: "gpu.utilization".into(),
            timestamp: snapshot.timestamp,
            value_running_window_sum: snapshot.utilization as f64,
            value_running_window_avg: snapshot.utilization as f64,
            value_running_window_count: 1,
        });

        // Memory utilization
        records.push(MetricRecord {
            metric_id: "gpu.memory_utilization".into(),
            timestamp: snapshot.timestamp,
            value_running_window_sum: snapshot.memory_utilization as f64,
            value_running_window_avg: snapshot.memory_utilization as f64,
            value_running_window_count: 1,
        });

        // Temperature
        records.push(MetricRecord {
            metric_id: "gpu.temperature".into(),
            timestamp: snapshot.timestamp,
            value_running_window_sum: snapshot.temperature as f64,
            value_running_window_avg: snapshot.temperature as f64,
            value_running_window_count: 1,
        });

        // Operations per second
        records.push(MetricRecord {
            metric_id: "gpu.ops_per_second".into(),
            timestamp: snapshot.timestamp,
            value_running_window_sum: snapshot.ops_per_second,
            value_running_window_avg: snapshot.ops_per_second,
            value_running_window_count: 1,
        });

        records
    }
}

impl<B: Backend> Clone for GpuMetricsCollector<B> {
    fn clone(&self) -> Self {
        Self {
            context: Arc::clone(&self.context),
            interval: self.interval,
            last_collection: RwLock::new(*self.last_collection.blocking_read()),
            history: RwLock::new(self.history.blocking_read().clone()),
            max_history: self.max_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use crate::gpu::GpuConfig;

    #[tokio::test]
    async fn test_gpu_metrics_collection() {
        let config = GpuConfig {
            backend: GpuBackend::Wgpu,
            max_batch_size: 1024,
            memory_limit: None,
            enable_tensor_cores: false,
        };

        let context = Arc::new(GpuContext::<Wgpu>::new(config).unwrap());
        let collector = GpuMetricsCollector::new(
            context,
            Duration::from_secs(1),
            100,
        );

        // Test metrics collection
        collector.collect_metrics().await.unwrap();
        
        // Verify metrics
        let metrics = collector.get_current_metrics().await.unwrap();
        assert!(metrics.utilization >= 0.0 && metrics.utilization <= 100.0);
        assert!(metrics.memory_utilization >= 0.0 && metrics.memory_utilization <= 100.0);
        assert!(metrics.temperature >= 0.0);
        assert!(metrics.ops_per_second >= 0.0);

        // Test conversion to metric records
        let records = collector.to_metric_records().await;
        assert!(!records.is_empty());
        assert!(records.iter().any(|r| r.metric_id == "gpu.utilization"));
        assert!(records.iter().any(|r| r.metric_id == "gpu.memory_utilization"));
    }
}
