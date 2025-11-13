//! Worktree metadata tracking
//!
//! Provides lightweight metadata tracking for model worktrees.
//! Metadata is stored in `.worktree-meta` files within each worktree directory.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metadata about a worktree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeMetadata {
    /// Branch name
    pub branch: String,

    /// When the worktree was created
    pub created_at: DateTime<Utc>,

    /// Reference the worktree was created from (e.g., "main", "tags/v1.0")
    pub created_from: Option<String>,

    /// Storage driver used (e.g., "overlay2", "vfs")
    pub storage_driver: String,

    /// Backend implementation (e.g., "kernel", "fuse", "plain")
    pub backend: Option<String>,

    /// Estimated space saved in bytes (for CoW drivers)
    pub space_saved_bytes: Option<u64>,

    /// Last time the worktree was accessed
    pub last_accessed: Option<DateTime<Utc>>,

    /// Custom tags/labels
    #[serde(default)]
    pub tags: Vec<String>,
}

impl WorktreeMetadata {
    /// Create new metadata for a worktree
    pub fn new(branch: String, created_from: Option<String>, storage_driver: String) -> Self {
        Self {
            branch,
            created_at: Utc::now(),
            created_from,
            storage_driver,
            backend: None,
            space_saved_bytes: None,
            last_accessed: Some(Utc::now()),
            tags: Vec::new(),
        }
    }

    /// Load metadata from a worktree directory
    pub fn load(worktree_path: &Path) -> Result<Self> {
        let meta_path = worktree_path.join(".worktree-meta");
        if !meta_path.exists() {
            anyhow::bail!("Worktree metadata not found at {:?}", meta_path);
        }

        let contents = std::fs::read_to_string(&meta_path)?;
        let mut meta: WorktreeMetadata = serde_json::from_str(&contents)?;

        // Update last accessed time
        meta.last_accessed = Some(Utc::now());
        meta.save(worktree_path)?;

        Ok(meta)
    }

    /// Save metadata to a worktree directory
    pub fn save(&self, worktree_path: &Path) -> Result<()> {
        let meta_path = worktree_path.join(".worktree-meta");
        let contents = serde_json::to_string_pretty(&self)?;
        std::fs::write(&meta_path, contents)?;
        Ok(())
    }

    /// Try to load metadata, return None if not found
    pub fn try_load(worktree_path: &Path) -> Option<Self> {
        Self::load(worktree_path).ok()
    }

    /// Calculate age of the worktree
    pub fn age(&self) -> chrono::Duration {
        Utc::now().signed_duration_since(self.created_at)
    }

    /// Time since last access
    pub fn time_since_last_access(&self) -> Option<chrono::Duration> {
        self.last_accessed
            .map(|last| Utc::now().signed_duration_since(last))
    }

    /// Format space saved as human-readable string
    pub fn space_saved_human(&self) -> String {
        match self.space_saved_bytes {
            Some(bytes) => format_bytes(bytes),
            None => "unknown".to_string(),
        }
    }

    /// Calculate space efficiency percentage
    pub fn space_efficiency(&self) -> Option<f64> {
        self.space_saved_bytes.map(|saved| {
            // Assume typical model size of ~5GB
            let typical_size = 5_000_000_000u64;
            (saved as f64 / typical_size as f64) * 100.0
        })
    }
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

/// Format duration as human-readable string
pub fn format_duration(duration: chrono::Duration) -> String {
    let seconds = duration.num_seconds();

    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m", seconds / 60)
    } else if seconds < 86400 {
        format!("{}h", seconds / 3600)
    } else {
        format!("{}d", seconds / 86400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(4_294_967_296), "4.0 GB");
    }

    #[test]
    fn test_format_duration() {
        use chrono::Duration;

        assert_eq!(format_duration(Duration::seconds(30)), "30s");
        assert_eq!(format_duration(Duration::seconds(120)), "2m");
        assert_eq!(format_duration(Duration::seconds(7200)), "2h");
        assert_eq!(format_duration(Duration::seconds(172800)), "2d");
    }

    #[test]
    fn test_metadata_creation() {
        let meta = WorktreeMetadata::new(
            "training-chat".to_string(),
            Some("main".to_string()),
            "overlay2".to_string(),
        );

        assert_eq!(meta.branch, "training-chat");
        assert_eq!(meta.created_from, Some("main".to_string()));
        assert_eq!(meta.storage_driver, "overlay2");
        assert!(meta.last_accessed.is_some());
    }
}
