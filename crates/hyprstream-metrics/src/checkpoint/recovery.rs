//! Recovery manager for restoring from checkpoints on startup.

use crate::checkpoint::{Checkpoint, CheckpointManager};
use crate::storage::duckdb::DuckDbBackend;
use crate::storage::StorageBackend;
use std::sync::Arc;
use tonic::Status;
use tracing::{info, instrument, warn};

/// Status of recovery operation
#[derive(Debug, Clone)]
pub enum RecoveryStatus {
    /// Successfully recovered from a checkpoint
    Recovered {
        /// The checkpoint that was restored
        checkpoint_id: String,
        /// Number of tables restored
        tables_restored: usize,
    },
    /// No checkpoint found to recover from
    NoCheckpointFound,
    /// Recovery was skipped (e.g., fresh database)
    Skipped {
        /// Reason for skipping
        reason: String,
    },
    /// Recovery failed
    Failed {
        /// Error message
        error: String,
    },
}

impl RecoveryStatus {
    /// Check if recovery was successful
    pub fn is_success(&self) -> bool {
        matches!(self, RecoveryStatus::Recovered { .. } | RecoveryStatus::NoCheckpointFound | RecoveryStatus::Skipped { .. })
    }
}

/// Recovery manager for startup checkpoint restoration
pub struct RecoveryManager {
    /// Reference to the checkpoint manager
    checkpoint_mgr: Arc<CheckpointManager>,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new(checkpoint_mgr: Arc<CheckpointManager>) -> Self {
        Self { checkpoint_mgr }
    }

    /// Attempt to recover from the latest checkpoint
    #[instrument(skip(self, storage))]
    pub async fn recover_latest(&self, storage: &DuckDbBackend) -> RecoveryStatus {
        info!("Attempting recovery from latest checkpoint");

        // Get the latest checkpoint
        let latest = self.checkpoint_mgr.get_latest_checkpoint().await;

        match latest {
            Some(checkpoint) => {
                info!(checkpoint_id = %checkpoint.id(), "Found checkpoint to restore");
                self.restore_from(&checkpoint, storage).await
            }
            None => {
                info!("No checkpoint found for recovery");
                RecoveryStatus::NoCheckpointFound
            }
        }
    }

    /// Restore from a specific checkpoint
    #[instrument(skip(self, checkpoint, storage), fields(checkpoint_id = %checkpoint.id()))]
    pub async fn restore_from(
        &self,
        checkpoint: &Checkpoint,
        storage: &DuckDbBackend,
    ) -> RecoveryStatus {
        info!("Restoring from checkpoint");

        match self
            .checkpoint_mgr
            .restore(checkpoint.id(), storage)
            .await
        {
            Ok(()) => {
                let tables_restored = checkpoint.tables().len();
                info!(
                    tables_restored = tables_restored,
                    "Recovery completed successfully"
                );
                RecoveryStatus::Recovered {
                    checkpoint_id: checkpoint.id().to_owned(),
                    tables_restored,
                }
            }
            Err(e) => {
                warn!(error = %e, "Recovery failed");
                RecoveryStatus::Failed {
                    error: e.message().to_owned(),
                }
            }
        }
    }

    /// Recover from a checkpoint by ID
    #[instrument(skip(self, storage))]
    pub async fn recover_by_id(
        &self,
        checkpoint_id: &str,
        storage: &DuckDbBackend,
    ) -> Result<RecoveryStatus, Status> {
        info!(checkpoint_id = %checkpoint_id, "Recovering from specific checkpoint");

        // Find the checkpoint in the manager's list
        let checkpoints = self.checkpoint_mgr.list_checkpoints().await;
        let checkpoint = checkpoints
            .iter()
            .find(|c| c.id() == checkpoint_id)
            .cloned();

        match checkpoint {
            Some(cp) => Ok(self.restore_from(&cp, storage).await),
            None => Err(Status::not_found(format!(
                "Checkpoint {checkpoint_id} not found"
            ))),
        }
    }

    /// Check if recovery is needed
    ///
    /// Returns true if:
    /// - A checkpoint exists
    /// - The database appears to be empty or in an inconsistent state
    #[instrument(skip(self, storage))]
    pub async fn needs_recovery(&self, storage: &DuckDbBackend) -> bool {
        // Check if there are any checkpoints
        let has_checkpoints = self.checkpoint_mgr.get_latest_checkpoint().await.is_some();

        if !has_checkpoints {
            return false;
        }

        // Check if database has tables (simple heuristic)
        match storage.list_tables().await {
            Ok(tables) => {
                // Filter out internal tables
                let user_tables: Vec<_> = tables
                    .into_iter()
                    .filter(|t| !t.starts_with("sqlite_") && t != "view_metadata")
                    .collect();

                // If no user tables, recovery might be needed
                user_tables.is_empty()
            }
            Err(_) => {
                // If we can't list tables, assume recovery is needed
                true
            }
        }
    }

    /// Perform automatic recovery if needed
    #[instrument(skip(self, storage))]
    pub async fn auto_recover(&self, storage: &DuckDbBackend) -> RecoveryStatus {
        if self.needs_recovery(storage).await {
            info!("Automatic recovery triggered");
            self.recover_latest(storage).await
        } else {
            info!("No recovery needed");
            RecoveryStatus::Skipped {
                reason: "Database appears to be in valid state".to_owned(),
            }
        }
    }
}

impl std::fmt::Debug for RecoveryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecoveryManager")
            .field("checkpoint_mgr", &"<CheckpointManager>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_status_success_check() {
        assert!(RecoveryStatus::Recovered {
            checkpoint_id: "abc".to_owned(),
            tables_restored: 5
        }
        .is_success());

        assert!(RecoveryStatus::NoCheckpointFound.is_success());

        assert!(RecoveryStatus::Skipped {
            reason: "test".to_owned()
        }
        .is_success());

        assert!(!RecoveryStatus::Failed {
            error: "error".to_owned()
        }
        .is_success());
    }
}
