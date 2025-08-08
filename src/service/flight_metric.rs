//! VDB-based metrics FlightSQL service

use crate::storage::VDBSparseStorage;
use crate::storage::vdb::sparse_storage::SparseStorage;
use std::sync::Arc;
use tonic::Status;

/// VDB-first metrics FlightSQL service
pub struct MetricFlightSqlService {
    vdb_storage: Arc<VDBSparseStorage>,
}

impl MetricFlightSqlService {
    pub fn new(vdb_storage: Arc<VDBSparseStorage>) -> Self {
        Self { vdb_storage }
    }
    
    /// Get VDB storage metrics
    pub async fn get_metrics(&self) -> Result<serde_json::Value, Status> {
        let stats = self.vdb_storage
            .get_storage_stats()
            .await
            .map_err(|e| Status::internal(format!("Failed to get VDB stats: {}", e)))?;
        
        let metrics = serde_json::json!({
            "total_adapters": stats.total_adapters,
            "avg_sparsity_ratio": stats.avg_sparsity_ratio,
            "updates_per_second": stats.updates_per_second,
            "total_disk_usage_bytes": stats.total_disk_usage_bytes,
            "total_memory_usage_bytes": stats.total_memory_usage_bytes,
            "cache_hit_ratio": stats.cache_hit_ratio
        });
        
        Ok(metrics)
    }
}