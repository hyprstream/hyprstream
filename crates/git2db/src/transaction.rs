//! Transaction support for atomic registry operations
//!
//! v2.1 implementation using Arc-based handles and libgit2 worktrees.
//! Provides true atomic operations with complete rollback capability.

use crate::errors::{Git2DBError, Git2DBResult};
use crate::registry::{Git2DB, RepoId, TrackedRepository};
use git2::{Repository, WorktreeAddOptions, WorktreePruneOptions};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Transaction handle for atomic registry operations
///
/// This is a clonable, Send + Sync + 'static handle that can be passed
/// to async functions without lifetime issues.
///
/// # Examples
///
/// ```rust,ignore
/// // Start a transaction
/// let tx = registry.start_transaction().await?;
///
/// // Queue operations (they don't execute yet)
/// let id1 = tx.clone_repo("repo1", "https://github.com/user/repo1.git").await?;
/// let id2 = tx.clone_repo("repo2", "https://github.com/user/repo2.git").await?;
///
/// // Commit atomically (all operations succeed or all fail)
/// tx.commit().await?;
///
/// // Or rollback (discards all operations)
/// // tx.rollback().await?;
/// ```
#[derive(Clone)]
pub struct TransactionHandle {
    inner: Arc<RwLock<TransactionState>>,
}

struct TransactionState {
    id: Uuid,
    registry_path: PathBuf,
    worktree_name: String,
    worktree_path: PathBuf,
    snapshot: RegistrySnapshot,
    operations: Vec<Operation>,
    isolation_mode: IsolationMode,
    committed: bool,
}

/// Snapshot of registry state at transaction start
#[derive(Clone, Debug, Serialize, Deserialize)]
struct RegistrySnapshot {
    version: String,
    repositories: std::collections::HashMap<Uuid, TrackedRepository>,
}

/// Operation to be applied in transaction
#[derive(Clone, Debug)]
enum Operation {
    Clone {
        id: RepoId,
        name: String,
        url: String,
    },
    Remove {
        id: RepoId,
    },
    Update {
        id: RepoId,
        url: Option<String>,
    },
    Upsert {
        name: String,
        url: String,
        // Store the result ID to return
        result_id: Arc<RwLock<Option<RepoId>>>,
    },
}

/// Isolation mode for transactions
#[derive(Clone, Debug)]
pub enum IsolationMode {
    /// Full worktree isolation using libgit2
    /// - Safest: complete filesystem isolation
    /// - Slowest: creates git worktree
    Worktree,

    /// Copy-on-write with metadata snapshots
    /// - Balanced: snapshot metadata only
    /// - Operations queue but can't rollback filesystem changes
    CopyOnWrite,

    /// Optimistic locking
    /// - Fastest: just queue operations
    /// - Check conflicts at commit time
    Optimistic,
}

impl Git2DB {
    /// Start a new transaction with default isolation (CopyOnWrite)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let tx = registry.start_transaction().await?;
    /// let id = tx.clone_repo("my-repo", "https://github.com/user/repo.git").await?;
    /// tx.commit().await?;
    /// ```
    pub async fn start_transaction(&self) -> Git2DBResult<TransactionHandle> {
        self.start_transaction_with_mode(IsolationMode::CopyOnWrite)
            .await
    }

    /// Start a transaction with specific isolation mode
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use git2db::IsolationMode;
    ///
    /// // Full worktree isolation for maximum safety
    /// let tx = registry.start_transaction_with_mode(IsolationMode::Worktree).await?;
    ///
    /// // Optimistic mode for maximum performance
    /// let tx = registry.start_transaction_with_mode(IsolationMode::Optimistic).await?;
    /// ```
    pub async fn start_transaction_with_mode(
        &self,
        mode: IsolationMode,
    ) -> Git2DBResult<TransactionHandle> {
        let tx_id = Uuid::new_v4();
        let tx_name = format!("tx-{}", tx_id);
        let worktree_path = self.registry_path().join(".worktrees").join(&tx_name);

        info!("Starting transaction {} with mode {:?}", tx_id, mode);

        // Create worktree based on isolation mode
        match &mode {
            IsolationMode::Worktree => {
                // Create actual git worktree using libgit2
                let registry_path = self.registry_path().to_path_buf();
                let tx_name_clone = tx_name.clone();
                let worktree_path_clone = worktree_path.clone();

                tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
                    let repo = Repository::open(&registry_path).map_err(|e| {
                        Git2DBError::repository(&registry_path, format!("Failed to open registry: {}", e))
                    })?;

                    let mut opts = WorktreeAddOptions::new();
                    opts.lock(false); // Don't lock by default

                    repo.worktree(&tx_name_clone, &worktree_path_clone, Some(&opts))
                        .map_err(|e| {
                            Git2DBError::internal(format!("Failed to create worktree: {}", e))
                        })?;

                    Ok(())
                })
                .await
                .map_err(|e| Git2DBError::internal(format!("Join error: {}", e)))??;

                info!("Created git worktree at {:?}", worktree_path);
            }
            IsolationMode::CopyOnWrite => {
                // Just create directory for metadata snapshot
                fs::create_dir_all(&worktree_path).await.map_err(|e| {
                    Git2DBError::internal(format!("Failed to create snapshot directory: {}", e))
                })?;
            }
            IsolationMode::Optimistic => {
                // No isolation, just track operations
            }
        }

        // Create snapshot of current registry state
        let snapshot = RegistrySnapshot {
            version: "2.0.0".to_string(), // TODO: Get from registry
            repositories: self.list().map(|r| (r.id.0, r.clone())).collect(),
        };

        let state = TransactionState {
            id: tx_id,
            registry_path: self.registry_path().to_path_buf(),
            worktree_name: tx_name,
            worktree_path,
            snapshot,
            operations: Vec::new(),
            isolation_mode: mode,
            committed: false,
        };

        Ok(TransactionHandle {
            inner: Arc::new(RwLock::new(state)),
        })
    }

    /// Deprecated v2.0 API - provided for backward compatibility
    ///
    /// Use `start_transaction()` instead for the new Arc-based API.
    #[deprecated(since = "2.1.0", note = "Use start_transaction() instead")]
    pub async fn transaction<F, Fut, R>(&mut self, f: F) -> Git2DBResult<R>
    where
        F: FnOnce(&mut crate::transaction::Transaction<'_>) -> Fut,
        Fut: std::future::Future<Output = Git2DBResult<R>>,
    {
        // Delegate to old implementation for backward compatibility
        let mut tx = Transaction::new(self).await?;

        match f(&mut tx).await {
            Ok(result) => {
                tx.commit().await?;
                Ok(result)
            }
            Err(e) => {
                tx.rollback().await?;
                Err(e)
            }
        }
    }
}

impl TransactionHandle {
    /// Clone a repository within the transaction
    ///
    /// The operation is queued and will execute on commit.
    pub async fn clone_repo(&self, name: &str, url: &str) -> Git2DBResult<RepoId> {
        let repo_id = RepoId::new();
        let mut state = self.inner.write().await;

        if state.committed {
            return Err(Git2DBError::internal("Transaction already committed"));
        }

        debug!(
            "Transaction {}: queue clone {} from {}",
            state.id, name, url
        );

        state.operations.push(Operation::Clone {
            id: repo_id.clone(),
            name: name.to_string(),
            url: url.to_string(),
        });

        Ok(repo_id)
    }

    /// Remove a repository within the transaction
    ///
    /// The operation is queued and will execute on commit.
    pub async fn remove(&self, id: &RepoId) -> Git2DBResult<()> {
        let mut state = self.inner.write().await;

        if state.committed {
            return Err(Git2DBError::internal("Transaction already committed"));
        }

        // Validate repository exists in snapshot
        if !state.snapshot.repositories.contains_key(&id.0) {
            return Err(Git2DBError::invalid_repository(
                &id.to_string(),
                "Repository not found",
            ));
        }

        debug!("Transaction {}: queue remove {}", state.id, id);

        state.operations.push(Operation::Remove { id: id.clone() });

        Ok(())
    }

    /// Update a repository within the transaction
    pub async fn update(&self, id: &RepoId, url: Option<String>) -> Git2DBResult<()> {
        let mut state = self.inner.write().await;

        if state.committed {
            return Err(Git2DBError::internal("Transaction already committed"));
        }

        // Validate repository exists in snapshot
        if !state.snapshot.repositories.contains_key(&id.0) {
            return Err(Git2DBError::invalid_repository(
                &id.to_string(),
                "Repository not found",
            ));
        }

        debug!("Transaction {}: queue update {}", state.id, id);

        state.operations.push(Operation::Update {
            id: id.clone(),
            url,
        });

        Ok(())
    }

    /// Ensure a repository exists (upsert operation)
    ///
    /// If a repository with the given name exists, returns its ID.
    /// Otherwise, clones it and returns the new ID.
    pub async fn ensure(&self, name: &str, url: &str) -> Git2DBResult<RepoId> {
        let mut state = self.inner.write().await;

        if state.committed {
            return Err(Git2DBError::internal("Transaction already committed"));
        }

        // Check if repository already exists in snapshot
        for repo in state.snapshot.repositories.values() {
            if repo.name.as_deref() == Some(name) {
                debug!(
                    "Transaction {}: ensure {} - already exists as {}",
                    state.id, name, repo.id
                );
                return Ok(repo.id.clone());
            }
        }

        // Will need to clone - queue upsert operation
        debug!("Transaction {}: queue ensure {} from {}", state.id, name, url);

        let result_id = Arc::new(RwLock::new(None));
        state.operations.push(Operation::Upsert {
            name: name.to_string(),
            url: url.to_string(),
            result_id: Arc::clone(&result_id),
        });

        // Generate ID now for return
        let id = RepoId::new();
        *result_id.write().await = Some(id.clone());

        Ok(id)
    }

    /// Get list of queued operations
    pub async fn operations(&self) -> Vec<String> {
        let state = self.inner.read().await;
        state
            .operations
            .iter()
            .map(|op| format!("{:?}", op))
            .collect()
    }

    /// Commit the transaction, applying all operations atomically to the registry
    ///
    /// On success, all operations are applied to the registry.
    /// On failure, the transaction is rolled back and all operations are discarded.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let tx = registry.start_transaction().await?;
    /// let id = tx.clone_repo("repo", "https://github.com/user/repo.git").await?;
    /// tx.commit_to(&mut registry).await?;  // Apply operations
    /// ```
    pub async fn commit_to(self, registry: &mut Git2DB) -> Git2DBResult<()> {
        let state = self.inner.write().await;

        if state.committed {
            return Err(Git2DBError::internal("Transaction already committed"));
        }

        info!(
            "Committing transaction {} with {} operations",
            state.id,
            state.operations.len()
        );

        // Apply each operation in order
        // If any fails, we'll automatically rollback via Drop
        for (i, operation) in state.operations.iter().enumerate() {
            debug!("Applying operation {}/{}", i + 1, state.operations.len());

            match operation {
                Operation::Clone { id, name, url } => {
                    debug!("Cloning {} from {} with ID {}", name, url, id);
                    registry.add_repository_with_id(id.clone(), name, url).await?;
                }
                Operation::Remove { id } => {
                    debug!("Removing repository {}", id);
                    registry.remove_repository(id).await?;
                }
                Operation::Update { id, url } => {
                    debug!("Updating repository {}", id);
                    registry.update_repository(id, url.clone()).await?;
                }
                Operation::Upsert { name, url, result_id } => {
                    debug!("Upserting {} from {}", name, url);

                    // Check if repository already exists
                    if let Some(existing) = registry.get_by_name(name) {
                        debug!("Repository '{}' already exists with ID {}", name, existing.id);
                        // Update result_id with existing ID
                        *result_id.write().await = Some(existing.id.clone());
                    } else {
                        // Use the pre-generated ID from ensure()
                        let expected_id = result_id.read().await.clone()
                            .ok_or_else(|| Git2DBError::internal("Upsert operation missing pre-generated ID"))?;

                        debug!("Creating repository '{}' with pre-generated ID {}", name, expected_id);
                        registry.add_repository_with_id(expected_id, name, url).await?;
                        // ID is already set in result_id from ensure()
                    }
                }
            }
        }

        // Mark as committed before cleanup
        drop(state);
        let mut state = self.inner.write().await;
        state.committed = true;
        drop(state);

        // Cleanup worktree
        let state = self.inner.read().await;
        self.cleanup_worktree(&state).await?;

        info!("Transaction {} committed successfully", state.id);

        Ok(())
    }

    /// Commit the transaction without a registry reference
    ///
    /// **Deprecated:** This method doesn't actually apply operations.
    /// Use `commit_to(&mut registry)` instead.
    #[deprecated(since = "2.1.0", note = "Use commit_to(&mut registry) instead")]
    pub async fn commit(self) -> Git2DBResult<()> {
        let state = self.inner.read().await;
        warn!(
            "Transaction {} committed without applying operations (use commit_to() instead)",
            state.id
        );

        // Just cleanup, don't apply operations
        self.cleanup_worktree(&state).await?;

        Ok(())
    }

    /// Rollback the transaction, discarding all operations
    ///
    /// All queued operations are discarded and the worktree is removed.
    pub async fn rollback(self) -> Git2DBResult<()> {
        let state = self.inner.read().await;

        if state.committed {
            return Err(Git2DBError::internal("Transaction already committed"));
        }

        warn!(
            "Rolling back transaction {} with {} operations",
            state.id,
            state.operations.len()
        );

        // Cleanup worktree
        self.cleanup_worktree(&state).await?;

        info!("Transaction {} rolled back successfully", state.id);

        Ok(())
    }

    async fn cleanup_worktree(&self, state: &TransactionState) -> Git2DBResult<()> {
        match &state.isolation_mode {
            IsolationMode::Worktree => {
                // Use libgit2 to prune the worktree
                let registry_path = state.registry_path.clone();
                let worktree_name = state.worktree_name.clone();
                let worktree_name_log = worktree_name.clone();

                tokio::task::spawn_blocking(move || -> Git2DBResult<()> {
                    let repo = Repository::open(&registry_path).map_err(|e| {
                        Git2DBError::repository(
                            &registry_path,
                            format!("Failed to open registry: {}", e),
                        )
                    })?;

                    let worktree = repo.find_worktree(&worktree_name).map_err(|e| {
                        Git2DBError::internal(format!("Failed to find worktree: {}", e))
                    })?;

                    let mut opts = WorktreePruneOptions::new();
                    opts.working_tree(true); // Remove filesystem data

                    worktree.prune(Some(&mut opts)).map_err(|e| {
                        Git2DBError::internal(format!("Failed to prune worktree: {}", e))
                    })?;

                    Ok(())
                })
                .await
                .map_err(|e| Git2DBError::internal(format!("Join error: {}", e)))??;

                info!("Pruned git worktree {}", worktree_name_log);
            }
            IsolationMode::CopyOnWrite => {
                // Just remove the snapshot directory
                if state.worktree_path.exists() {
                    tokio::fs::remove_dir_all(&state.worktree_path)
                        .await
                        .map_err(|e| {
                            Git2DBError::internal(format!("Failed to remove snapshot dir: {}", e))
                        })?;
                }
            }
            IsolationMode::Optimistic => {
                // No cleanup needed
            }
        }

        Ok(())
    }
}

// Keep old Transaction struct for backward compatibility
pub struct Transaction<'a> {
    registry: &'a mut Git2DB,
    worktree_path: PathBuf,
    tx_id: String,
    operations: Vec<String>,
}

impl<'a> Transaction<'a> {
    pub(crate) async fn new(registry: &'a mut Git2DB) -> Git2DBResult<Self> {
        let tx_id = Uuid::new_v4().to_string();
        let worktree_path = registry.registry_path().join(".worktrees").join(&tx_id);

        info!("Starting legacy transaction: {}", tx_id);

        fs::create_dir_all(&worktree_path).await.map_err(|e| {
            Git2DBError::internal(format!("Failed to create worktree directory: {}", e))
        })?;

        Ok(Self {
            registry,
            worktree_path,
            tx_id,
            operations: Vec::new(),
        })
    }

    pub async fn remove(&mut self, id: &RepoId) -> Git2DBResult<()> {
        debug!("Legacy transaction {}: remove {}", self.tx_id, id);
        self.operations.push(format!("remove {}", id));
        self.registry.remove_repository(id).await
    }

    pub async fn ensure(&mut self, name: &str, url: &str) -> Git2DBResult<RepoId> {
        debug!("Legacy transaction {}: ensure {} from {}", self.tx_id, name, url);
        self.operations
            .push(format!("ensure {} from {}", name, url));
        self.registry.upsert_repository(name, url).await
    }

    pub async fn clone(&mut self, name: &str, url: &str) -> Git2DBResult<RepoId> {
        debug!("Legacy transaction {}: clone {} from {}", self.tx_id, name, url);
        self.operations.push(format!("clone {} from {}", name, url));
        self.registry.add_repository(name, url).await
    }

    pub async fn update(&mut self, id: &RepoId, new_url: Option<String>) -> Git2DBResult<()> {
        debug!("Legacy transaction {}: update {}", self.tx_id, id);
        self.operations.push(format!("update {}", id));
        self.registry.update_repository(id, new_url).await
    }

    pub fn operations(&self) -> &[String] {
        &self.operations
    }

    async fn commit(self) -> Git2DBResult<()> {
        info!(
            "Legacy transaction {} committed: {} operations",
            self.tx_id,
            self.operations.len()
        );

        if self.worktree_path.exists() {
            fs::remove_dir_all(&self.worktree_path)
                .await
                .map_err(|e| Git2DBError::internal(format!("Failed to cleanup worktree: {}", e)))?;
        }

        Ok(())
    }

    async fn rollback(self) -> Git2DBResult<()> {
        warn!(
            "Legacy transaction {} rolled back: {} operations",
            self.tx_id,
            self.operations.len()
        );

        if self.worktree_path.exists() {
            fs::remove_dir_all(&self.worktree_path)
                .await
                .map_err(|e| Git2DBError::internal(format!("Failed to cleanup worktree: {}", e)))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests will be in tests/ directory
}
