//! Checkpoint state types and metadata.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata for a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint identifier (git commit hash)
    pub id: String,
    /// Human-readable checkpoint name/tag
    pub name: Option<String>,
    /// Timestamp when checkpoint was created
    pub created_at: DateTime<Utc>,
    /// Tables included in this checkpoint
    pub tables: Vec<TableCheckpoint>,
    /// Total size of checkpoint data in bytes
    pub total_size: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Information about a checkpointed table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCheckpoint {
    /// Table name
    pub name: String,
    /// Number of rows in the table
    pub row_count: u64,
    /// Size of the Parquet file in bytes
    pub file_size: u64,
    /// Path to the Parquet file within the checkpoint
    pub file_path: String,
}

/// A checkpoint instance
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Git commit ID for this checkpoint
    pub commit_id: String,
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(commit_id: String, metadata: CheckpointMetadata) -> Self {
        Self {
            commit_id,
            metadata,
        }
    }

    /// Get the checkpoint ID (git commit hash)
    pub fn id(&self) -> &str {
        &self.commit_id
    }

    /// Get the checkpoint name if set
    pub fn name(&self) -> Option<&str> {
        self.metadata.name.as_deref()
    }

    /// Get when the checkpoint was created
    pub fn created_at(&self) -> DateTime<Utc> {
        self.metadata.created_at
    }

    /// Get the list of tables in this checkpoint
    pub fn tables(&self) -> &[TableCheckpoint] {
        &self.metadata.tables
    }

    /// Get total checkpoint size in bytes
    pub fn total_size(&self) -> u64 {
        self.metadata.total_size
    }
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(tables: Vec<TableCheckpoint>) -> Self {
        let total_size = tables.iter().map(|t| t.file_size).sum();
        Self {
            id: String::new(), // Will be set after commit
            name: None,
            created_at: Utc::now(),
            tables,
            total_size,
            metadata: HashMap::new(),
        }
    }

    /// Set the checkpoint name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl TableCheckpoint {
    /// Create a new table checkpoint entry
    pub fn new(name: String, row_count: u64, file_size: u64, file_path: String) -> Self {
        Self {
            name,
            row_count,
            file_size,
            file_path,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata_creation() {
        let tables = vec![
            TableCheckpoint::new("metrics".to_string(), 1000, 50000, "tables/metrics.parquet".to_string()),
            TableCheckpoint::new("events".to_string(), 500, 25000, "tables/events.parquet".to_string()),
        ];

        let metadata = CheckpointMetadata::new(tables)
            .with_name("checkpoint-1")
            .with_metadata("version", "1.0");

        assert_eq!(metadata.name, Some("checkpoint-1".to_string()));
        assert_eq!(metadata.total_size, 75000);
        assert_eq!(metadata.tables.len(), 2);
        assert_eq!(metadata.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_checkpoint_creation() {
        let tables = vec![TableCheckpoint::new(
            "test".to_string(),
            100,
            5000,
            "tables/test.parquet".to_string(),
        )];

        let metadata = CheckpointMetadata::new(tables);
        let checkpoint = Checkpoint::new("abc123".to_string(), metadata);

        assert_eq!(checkpoint.id(), "abc123");
        assert_eq!(checkpoint.tables().len(), 1);
        assert_eq!(checkpoint.total_size(), 5000);
    }
}
