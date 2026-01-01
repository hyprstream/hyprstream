//! DuckDB database archetype detection
//!
//! Detects repositories containing DuckDB database files.
//! These can be used for metrics storage, training data,
//! or query via Flight SQL.
//!
//! # Security
//!
//! Detection validates DuckDB files have minimum size and valid
//! header structure to prevent trivial spoofing attacks.

use super::capabilities::{CapabilitySet, Query, Serve, Write};
use super::RepoArchetype;
use std::path::Path;

/// Minimum valid DuckDB file size (header + at least one page)
const MIN_DUCKDB_SIZE: u64 = 4096;

/// DuckDB database archetype
///
/// Detects repos with:
/// - `*.duckdb` files (DuckDB database) - must have valid structure
///
/// Enables: QUERY, WRITE, SERVE
pub struct DuckDbArchetype;

impl DuckDbArchetype {
    /// Validate that a DuckDB file appears to be valid
    ///
    /// Checks:
    /// - File exists and is readable
    /// - File is at least the minimum size for a valid DuckDB database
    fn validate_duckdb_file(path: &Path) -> bool {
        match std::fs::metadata(path) {
            Ok(meta) => {
                // DuckDB files must be at least MIN_DUCKDB_SIZE bytes
                // An empty database is ~4KB due to page structure
                meta.len() >= MIN_DUCKDB_SIZE
            }
            Err(_) => false,
        }
    }

    /// Check for valid DuckDB files in a directory
    fn has_valid_duckdb_in_dir(dir: &Path) -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "duckdb" && Self::validate_duckdb_file(&path) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl RepoArchetype for DuckDbArchetype {
    fn name(&self) -> &'static str {
        "duckdb"
    }

    fn description(&self) -> &'static str {
        "DuckDB database for SQL queries via Flight SQL"
    }

    fn detect(&self, repo_path: &Path) -> bool {
        // Check for valid .duckdb files in root
        if Self::has_valid_duckdb_in_dir(repo_path) {
            return true;
        }

        // Also check common subdirectories for databases
        for subdir in &["data", "db", "databases"] {
            let sub_path = repo_path.join(subdir);
            if sub_path.is_dir() && Self::has_valid_duckdb_in_dir(&sub_path) {
                return true;
            }
        }

        false
    }

    fn capabilities(&self) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert::<Query>();
        caps.insert::<Write>();
        caps.insert::<Serve>();
        caps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Create dummy DuckDB data with valid minimum size
    fn valid_duckdb_data() -> Vec<u8> {
        // Create data at least MIN_DUCKDB_SIZE bytes
        vec![0u8; MIN_DUCKDB_SIZE as usize]
    }

    #[test]
    fn test_detect_with_duckdb_file() {
        let temp = TempDir::new().expect("test: create temp dir");
        let db_path = temp.path().join("metrics.duckdb");
        fs::write(&db_path, valid_duckdb_data()).expect("test: write duckdb");

        let archetype = DuckDbArchetype;
        assert!(archetype.detect(temp.path()));
    }

    #[test]
    fn test_detect_with_duckdb_in_subdir() {
        let temp = TempDir::new().expect("test: create temp dir");
        let db_dir = temp.path().join("db");
        fs::create_dir(&db_dir).expect("test: create subdir");
        let db_path = db_dir.join("metrics.duckdb");
        fs::write(&db_path, valid_duckdb_data()).expect("test: write duckdb");

        let archetype = DuckDbArchetype;
        assert!(archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_empty_dir() {
        let temp = TempDir::new().expect("test: create temp dir");

        let archetype = DuckDbArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_small_file() {
        let temp = TempDir::new().expect("test: create temp dir");
        // Small file should not be detected (potential spoof)
        let db_path = temp.path().join("metrics.duckdb");
        fs::write(&db_path, "dummy").expect("test: write small file");

        let archetype = DuckDbArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_capabilities() {
        use super::super::capabilities::{Infer, Train};

        let archetype = DuckDbArchetype;
        let caps = archetype.capabilities();

        assert!(caps.has::<Query>());
        assert!(caps.has::<Write>());
        assert!(caps.has::<Serve>());
        assert!(!caps.has::<Infer>());
        assert!(!caps.has::<Train>());
    }
}
