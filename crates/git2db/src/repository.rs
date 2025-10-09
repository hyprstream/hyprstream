//! Repository management and caching
//!
//! Consolidated from repository_cache.rs and git/manager.rs with improvements

use crate::errors::{Git2DBError, Git2DBResult};
use git2::Repository;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Thread-safe repository cache entry with metadata tracking
///
/// This struct doesn't store the git2::Repository directly (not thread-safe),
/// but provides a cached path and metadata for efficient repository access.
#[derive(Debug, Clone)]
pub struct RepositoryCache {
    path: PathBuf,
    created_at: Instant,
    access_count: Arc<parking_lot::Mutex<u64>>,
}

impl RepositoryCache {
    pub(crate) fn new(_repository: Repository, path: PathBuf) -> Self {
        Self {
            path,
            created_at: Instant::now(),
            access_count: Arc::new(parking_lot::Mutex::new(1)),
        }
    }

    /// Get a fresh repository instance (git2::Repository is not thread-safe)
    pub fn open(&self) -> Git2DBResult<Repository> {
        let mut count = self.access_count.lock();
        *count += 1;

        Repository::open(&self.path).map_err(|e| {
            Git2DBError::repository(&self.path, format!("Failed to open repository: {}", e))
        })
    }

    /// Get read access to the repository (for compatibility)
    pub fn read(&self) -> Git2DBResult<Repository> {
        self.open()
    }

    /// Get write access to the repository (for compatibility)
    pub fn write(&self) -> Git2DBResult<Repository> {
        self.open()
    }

    /// Get the repository path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get creation time
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Get access count
    pub fn access_count(&self) -> u64 {
        *self.access_count.lock()
    }

    /// Check if this handle is expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_accesses: u64,
    pub oldest_created: Instant,
    pub newest_created: Instant,
    pub oldest_accessed: Instant,
    pub newest_accessed: Instant,
}

impl CacheStats {
    /// Get the age of the oldest entry
    pub fn oldest_entry_age(&self) -> Duration {
        self.oldest_created.elapsed()
    }

    /// Get the age of the newest entry
    pub fn newest_entry_age(&self) -> Duration {
        self.newest_created.elapsed()
    }

    /// Get the age of the least recently accessed entry
    pub fn oldest_access_age(&self) -> Duration {
        self.oldest_accessed.elapsed()
    }

    /// Calculate average accesses per entry
    pub fn average_accesses_per_entry(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            self.total_accesses as f64 / self.total_entries as f64
        }
    }
}
