use std::sync::Arc;
use arrow_array::{ArrayRef, RecordBatch, Float64Array, Int64Array, StringArray};
use arrow_array::builder::{StringBuilder, Int64Builder, Float64Builder};
use arrow_schema::{Schema, Field, DataType};
use arrow::error::ArrowError;
use crate::error::{Error, Result};
use tokio::sync::RwLock;
use tonic::Status;
use burn::prelude::Backend;
use crate::config::ServiceConfig;
use crate::gpu::{GpuContext, GpuConfig};

pub mod aggregation;
pub mod performance;
pub mod gpu;
pub mod resource;
pub mod cache;

/// A single metric record with running window calculations.
#[derive(Debug, Clone)]
pub struct MetricRecord {
    /// Unique identifier for the metric
    pub metric_id: String,
    /// Unix timestamp in seconds
    pub timestamp: i64,
    /// Running sum within the window
    pub value_running_window_sum: f64,
    /// Running average within the window
    pub value_running_window_avg: f64,
    /// Number of values in the window
    pub value_running_window_count: i16,
}

impl MetricRecord {
    pub fn to_arrow_batch(metrics: &[MetricRecord]) -> Result<RecordBatch> {
        let mut id_builder = StringBuilder::new();
        let mut ts_builder = Int64Builder::new();
        let mut sum_builder = Float64Builder::new();
        let mut avg_builder = Float64Builder::new();
        let mut count_builder = Int64Builder::new();

        for metric in metrics {
            id_builder.append_value(&metric.metric_id);
            ts_builder.append_value(metric.timestamp);
            sum_builder.append_value(metric.value_running_window_sum);
            avg_builder.append_value(metric.value_running_window_avg);
            count_builder.append_value(metric.value_running_window_count.into());
        }

        let schema = Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]);

        Ok(RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(ts_builder.finish()),
                Arc::new(sum_builder.finish()),
                Arc::new(avg_builder.finish()),
                Arc::new(count_builder.finish()),
            ],
        )?)
    }

    pub fn from_arrow_batch(batch: &RecordBatch, index: usize) -> Result<Self> {
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Validation("Invalid metric_id column".into()))?;
        let timestamps = batch.column(1).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| Error::Validation("Invalid timestamp column".into()))?;
        let sums = batch.column(2).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Validation("Invalid value_running_window_sum column".into()))?;
        let avgs = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Validation("Invalid value_running_window_avg column".into()))?;
        let counts = batch.column(4).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| Error::Validation("Invalid value_running_window_count column".into()))?;

        Ok(MetricRecord {
            metric_id: ids.value(index).to_string(),
            timestamp: timestamps.value(index),
            value_running_window_sum: sums.value(index),
            value_running_window_avg: avgs.value(index),
            value_running_window_count: counts.value(index) as i16,
        })
    }
}

use std::time::Duration;
pub use self::gpu::{GpuMetricsCollector, GpuMetricSnapshot};

/// Metrics collection service
pub struct MetricsService<B: Backend> {
    /// GPU metrics collector
    gpu_metrics: Option<Arc<GpuMetricsCollector<B>>>,
    /// Collection interval
    interval: Duration,
    /// Metrics retention period
    retention: Duration,
    /// Stored metrics
    metrics: Arc<RwLock<Vec<MetricRecord>>>,
}

impl<B: Backend> MetricsService<B> {
    /// Create a new metrics service
    pub fn new(config: &ServiceConfig, gpu_context: Option<Arc<GpuContext<B>>>) -> Self {
        let gpu_metrics = gpu_context.map(|ctx| {
            Arc::new(GpuMetricsCollector::new(
                ctx,
                Duration::from_secs(config.metrics.interval),
                (config.metrics.retention / config.metrics.interval) as usize,
            ))
        });

        Self {
            gpu_metrics,
            interval: Duration::from_secs(config.metrics.interval),
            retention: Duration::from_secs(config.metrics.retention),
            metrics: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start metrics collection
    pub async fn start(&self) -> Result<(), Status> {
        // Start GPU metrics collection if available
        if let Some(collector) = &self.gpu_metrics {
            collector.start().await.map_err(|e| Status::internal(e.to_string()))?;
        }

        let metrics = Arc::clone(&self.metrics);
        let interval = self.interval;
        let retention = self.retention;
        let gpu_metrics = self.gpu_metrics.clone();

        // Start collection loop
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;

                // Collect GPU metrics
                if let Some(collector) = &gpu_metrics {
                    if let Ok(gpu_records) = collector.to_metric_records().await {
                        let mut metrics = metrics.write().await;
                        metrics.extend(gpu_records);
                    }
                }

                // Prune old metrics
                let cutoff = chrono::Utc::now().timestamp() - retention.as_secs() as i64;
                let mut metrics = metrics.write().await;
                metrics.retain(|m| m.timestamp >= cutoff);
            }
        });

        Ok(())
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> Vec<MetricRecord> {
        self.metrics.read().await.clone()
    }

    /// Get GPU metrics if available
    pub async fn get_gpu_metrics(&self) -> Option<GpuMetricSnapshot> {
        if let Some(collector) = &self.gpu_metrics {
            collector.get_current_metrics().await.ok()
        } else {
            None
        }
    }
}

impl MetricRecord {
    /// Convert metric record to Arrow record batch
    pub fn to_record_batch(metrics: &[MetricRecord]) -> Result<RecordBatch> {
        let schema = Schema::new(vec![
            Field::new("metric_id", DataType::Utf8, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value_running_window_sum", DataType::Float64, false),
            Field::new("value_running_window_avg", DataType::Float64, false),
            Field::new("value_running_window_count", DataType::Int64, false),
        ]);

        let mut id_builder = StringBuilder::new();
        let mut ts_builder = Int64Builder::new();
        let mut sum_builder = Float64Builder::new();
        let mut avg_builder = Float64Builder::new();
        let mut count_builder = Int64Builder::new();

        for metric in metrics {
            id_builder.append_value(&metric.metric_id);
            ts_builder.append_value(metric.timestamp);
            sum_builder.append_value(metric.value_running_window_sum);
            avg_builder.append_value(metric.value_running_window_avg);
            count_builder.append_value(i64::from(metric.value_running_window_count));
        }

        let arrays: Vec<ArrayRef> = vec![
            Arc::new(id_builder.finish()),
            Arc::new(ts_builder.finish()),
            Arc::new(sum_builder.finish()),
            Arc::new(avg_builder.finish()),
            Arc::new(count_builder.finish()),
        ];

        Ok(RecordBatch::try_new(Arc::new(schema), arrays).map_err(|e| Error::Internal(e.to_string()))?)
    }

    /// Create metric record from Arrow arrays
    pub fn from_arrays(
        ids: &StringArray,
        timestamps: &Int64Array,
        sums: &Float64Array,
        avgs: &Float64Array,
        counts: &Int64Array,
        index: usize,
    ) -> Self {
        Self {
            metric_id: ids.value(index).to_string(),
            timestamp: timestamps.value(index),
            value_running_window_sum: sums.value(index),
            value_running_window_avg: avgs.value(index),
            value_running_window_count: counts.value(index).try_into().unwrap_or(i16::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[tokio::test]
    async fn test_metrics_service() {
        let config = ServiceConfig {
            storage: Default::default(),
            metrics: crate::config::MetricsConfig {
                enabled: true,
                interval: 1,
                retention: 3600,
            },
            gpu: Default::default(),
        };

        // Create service with GPU context
        let gpu_config = GpuConfig {
            backend: crate::gpu::GpuBackend::Wgpu,
            max_batch_size: 1024,
            memory_limit: None,
            enable_tensor_cores: false,
        };
        let gpu_context = GpuContext::<Wgpu>::new(gpu_config).ok().map(Arc::new);
        let service = MetricsService::new(&config, gpu_context);

        // Start service
        service.start().await.unwrap();

        // Wait for metrics collection
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Verify metrics
        let metrics = service.get_metrics().await;
        assert!(!metrics.is_empty());

        // Check GPU metrics if available
        if let Some(gpu_metrics) = service.get_gpu_metrics().await {
            assert!(gpu_metrics.utilization >= 0.0);
            assert!(gpu_metrics.memory_utilization >= 0.0);
        }
    }

    #[test]
    fn test_metric_record_conversion() {
        let metrics = vec![
            MetricRecord {
                metric_id: "test".into(),
                timestamp: 1234567890,
                value_running_window_sum: 100.0,
                value_running_window_avg: 50.0,
                value_running_window_count: 2,
            },
        ];

        let batch = MetricRecord::to_record_batch(&metrics).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 5);

        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let timestamps = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let sums = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        let avgs = batch.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        let counts = batch.column(4).as_any().downcast_ref::<Int64Array>().unwrap();

        let record = MetricRecord::from_arrays(ids, timestamps, sums, avgs, counts, 0);
        assert_eq!(record.metric_id, "test");
        assert_eq!(record.timestamp, 1234567890);
        assert_eq!(record.value_running_window_sum, 100.0);
        assert_eq!(record.value_running_window_avg, 50.0);
        assert_eq!(record.value_running_window_count, 2);
    }
}
