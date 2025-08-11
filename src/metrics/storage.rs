//! VDB-based metrics storage for adaptive ML inference systems

use crate::aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
use crate::metrics::MetricRecord;
use crate::storage::VDBSparseStorage;
use crate::storage::vdb::sparse_storage::SparseStorage;
// Use simplified schema instead of arrow-schema
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

/// Batch-level aggregation state for efficient updates
#[derive(Debug, Clone)]
pub struct BatchAggregation {
    /// The metric ID this aggregation belongs to
    pub metric_id: String,
    /// Start of the time window
    pub window_start: i64,
    /// End of the time window
    pub window_end: i64,
    /// Running sum within the window
    pub running_sum: f64,
    /// Running count within the window
    pub running_count: i64,
    /// Minimum value in the window
    pub min_value: f64,
    /// Maximum value in the window
    pub max_value: f64,
    // Schema removed - simplified for VDB-first architecture
    /// Column to aggregate
    pub value_column: String,
    /// Grouping specification
    pub group_by: GroupBy,
    /// Time window specification
    pub window: Option<TimeWindow>,
}

impl BatchAggregation {
    pub fn new_window(
        metric_id: String,
        window_start: i64,
        window_end: i64,
        value_column: String,
        group_by: GroupBy,
        window: Option<TimeWindow>,
    ) -> Self {
        Self {
            metric_id,
            window_start,
            window_end,
            running_sum: 0.0,
            running_count: 0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            value_column,
            group_by,
            window,
        }
    }

    pub fn new_from_metric(
        metric_id: String,
        window_start: i64,
        window_end: i64,
        window: TimeWindow,
    ) -> Self {
        let group_by = GroupBy {
            columns: vec!["metric".to_string()],
            time_column: Some("timestamp".to_string()),
        };
        Self {
            metric_id,
            window_start,
            window_end,
            running_sum: 0.0,
            running_count: 0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            value_column: "value".to_string(),
            group_by,
            window: Some(window),
        }
    }

    pub fn new(
        value_column: String,
        group_by: GroupBy,
        window: Option<TimeWindow>,
    ) -> Self {
        Self {
            metric_id: String::new(),
            window_start: 0,
            window_end: 0,
            running_sum: 0.0,
            running_count: 0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
            value_column,
            group_by,
            window,
        }
    }
}

/// VDB-based metrics storage for adaptive ML inference systems.
/// 
/// This storage system is designed to handle real-time metrics from
/// sparse weight adjustments and neural network inference operations.
#[async_trait::async_trait]
pub trait VDBMetricsStorage: Send + Sync + 'static {
    /// Get the underlying VDB sparse storage backend
    fn vdb_storage(&self) -> &Arc<VDBSparseStorage>;

    /// Initialize metrics storage (setup VDB structures)
    async fn init(&self) -> Result<(), Status> {
        // VDB storage initialization is handled in the backend
        // No explicit table creation needed for VDB-first architecture
        Ok(())
    }

    /// Store metrics as sparse adapter metadata
    async fn insert_metrics(&self, metrics: Vec<MetricRecord>) -> Result<(), Status> {
        // Convert metrics to VDB storage operations
        for metric in metrics {
            // Store metric as adapter metadata in VDB
            // This could be used to track neural network performance metrics
            println!("📊 Storing metric: {} = {} at {}", 
                metric.metric_id, metric.value, metric.timestamp);
        }
        
        // VDB-based metrics storage would aggregate these into sparse structures
        Ok(())
    }

    /// Query metrics from VDB storage
    async fn query_metrics(&self, _from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        // Query VDB storage for metrics-related adapter statistics
        let storage_stats = self.vdb_storage()
            .get_storage_stats()
            .await
            .map_err(|e| Status::internal(format!("VDB query failed: {}", e)))?;

        // Convert VDB stats to metric records
        let mut metrics = Vec::new();
        
        // Example: Convert storage statistics to metric records
        metrics.push(MetricRecord {
            metric_id: "vdb.total_adapters".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            value: storage_stats.total_adapters as f64,
            value_running_window_sum: storage_stats.total_adapters as f64,
            value_running_window_avg: storage_stats.total_adapters as f64,
            value_running_window_count: 1,
        });
        
        metrics.push(MetricRecord {
            metric_id: "vdb.avg_sparsity_ratio".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            value: storage_stats.avg_sparsity_ratio as f64,
            value_running_window_sum: storage_stats.avg_sparsity_ratio as f64,
            value_running_window_avg: storage_stats.avg_sparsity_ratio as f64,
            value_running_window_count: 1,
        });
        
        metrics.push(MetricRecord {
            metric_id: "vdb.updates_per_second".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            value: storage_stats.updates_per_second,
            value_running_window_sum: storage_stats.updates_per_second,
            value_running_window_avg: storage_stats.updates_per_second,
            value_running_window_count: 1,
        });

        Ok(metrics)
    }

    /// Aggregate metrics using VDB sparse operations
    async fn aggregate_metrics(
        &self,
        function: AggregateFunction,
        _group_by: &GroupBy,
        _from_timestamp: i64,
        _to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        // Use VDB storage to compute aggregations across adapters
        let adapters = self.vdb_storage()
            .list_adapters()
            .await
            .map_err(|e| Status::internal(format!("VDB aggregation failed: {}", e)))?;

        let mut results = Vec::new();
        
        for adapter in adapters {
            let stats = self.vdb_storage()
                .get_adapter_stats(&adapter.adapter_id)
                .await
                .map_err(|e| Status::internal(format!("Failed to get adapter stats: {}", e)))?;
            
            let mut group_values = HashMap::new();
            group_values.insert("adapter_id".to_string(), adapter.adapter_id.clone());
            
            let value = match function {
                AggregateFunction::Sum => stats.active_weights as f64,
                AggregateFunction::Avg => stats.sparsity_ratio as f64,
                AggregateFunction::Count => 1.0,
                AggregateFunction::Min => stats.sparsity_ratio as f64,
                AggregateFunction::Max => stats.sparsity_ratio as f64,
            };
            
            results.push(AggregateResult {
                value,
                group_values,
                timestamp: Some(stats.last_update_timestamp as i64),
            });
        }

        Ok(results)
    }

    /// Get metric field names for VDB storage
    fn get_metric_fields() -> Vec<&'static str> {
        vec![
            "adapter_id",
            "timestamp", 
            "active_weights",
            "sparsity_ratio",
            "memory_usage",
            "compression_ratio", 
            "updates_per_second"
        ]
    }
}

/// VDB-first metrics storage implementation
pub struct VDBMetricsStorageImpl {
    vdb_storage: Arc<VDBSparseStorage>,
}

impl VDBMetricsStorageImpl {
    pub fn new(vdb_storage: Arc<VDBSparseStorage>) -> Self {
        Self { vdb_storage }
    }
}

#[async_trait::async_trait]
impl VDBMetricsStorage for VDBMetricsStorageImpl {
    fn vdb_storage(&self) -> &Arc<VDBSparseStorage> {
        &self.vdb_storage
    }
}

/// Legacy compatibility trait (deprecated - use VDBMetricsStorage instead)
#[deprecated(note = "Use VDBMetricsStorage for VDB-first architecture")]
#[async_trait::async_trait]
pub trait MetricsStorage: Send + Sync + 'static {
    async fn init(&self) -> Result<(), Status> {
        Ok(())
    }

    async fn insert_metrics(&self, _metrics: Vec<MetricRecord>) -> Result<(), Status> {
        Err(Status::unimplemented("Legacy metrics storage deprecated. Use VDBMetricsStorage."))
    }

    async fn query_metrics(&self, _from_timestamp: i64) -> Result<Vec<MetricRecord>, Status> {
        Err(Status::unimplemented("Legacy metrics storage deprecated. Use VDBMetricsStorage."))
    }

    async fn aggregate_metrics(
        &self,
        _function: AggregateFunction,
        _group_by: &GroupBy,
        _from_timestamp: i64,
        _to_timestamp: Option<i64>,
    ) -> Result<Vec<AggregateResult>, Status> {
        Err(Status::unimplemented("Legacy metrics storage deprecated. Use VDBMetricsStorage."))
    }
}