//! Configuration management for hyprstream-metrics library.
//!
//! This module provides configuration structures for storage backends,
//! caching, and aggregation settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Storage backend configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct StorageConfig {
    /// Storage engine type: "duckdb", "adbc", or "cached"
    #[serde(default = "default_engine")]
    pub engine: String,
    /// Connection string for the storage backend
    #[serde(default)]
    pub connection_string: String,
    /// Additional engine-specific options
    #[serde(default)]
    pub options: HashMap<String, String>,
}

fn default_engine() -> String {
    "duckdb".to_string()
}

/// Cache configuration for cached storage backends
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheConfig {
    /// Maximum duration in seconds for cached data
    #[serde(default = "default_max_duration")]
    pub max_duration_secs: u64,
    /// Store engine type for persistent storage
    #[serde(default = "default_engine")]
    pub store_engine: String,
}

fn default_max_duration() -> u64 {
    3600
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_duration_secs: default_max_duration(),
            store_engine: default_engine(),
        }
    }
}

/// Aggregation configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AggregationConfig {
    /// Default time window size in seconds
    #[serde(default = "default_window_size")]
    pub default_window_size_secs: u64,
    /// Maximum number of concurrent aggregations
    #[serde(default = "default_max_aggregations")]
    pub max_concurrent_aggregations: usize,
}

fn default_window_size() -> u64 {
    60
}

fn default_max_aggregations() -> usize {
    100
}

/// Complete library configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct MetricsConfig {
    /// Storage backend configuration
    #[serde(default)]
    pub storage: StorageConfig,
    /// Cache configuration
    #[serde(default)]
    pub cache: CacheConfig,
    /// Aggregation configuration
    #[serde(default)]
    pub aggregation: AggregationConfig,
}
