//! HuggingFace Dataset archetype detection
//!
//! Detects repositories containing HuggingFace-compatible datasets.
//! Detection is based on presence of dataset marker files and
//! data formats (parquet, arrow, csv).
//!
//! # Security
//!
//! Detection validates marker files contain valid JSON to prevent
//! trivial spoofing attacks.

use super::capabilities::{CapabilitySet, Query, Serve, Write};
use super::RepoArchetype;
use std::path::Path;

/// HuggingFace Dataset archetype
///
/// Detects repos with:
/// - `dataset_infos.json` (HuggingFace dataset marker) - must be valid JSON
/// - `dataset.json` (hyprstream dataset marker) - must be valid JSON
/// - `*.parquet` files (columnar data) - must have PAR1 magic header
///
/// Enables: QUERY, WRITE, SERVE
pub struct HfDatasetArchetype;

impl HfDatasetArchetype {
    /// Validate that a JSON file contains valid JSON
    fn validate_json_file(path: &Path) -> bool {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                // Must be valid JSON (empty object {} is valid)
                serde_json::from_str::<serde_json::Value>(&content).is_ok()
            }
            Err(_) => false,
        }
    }

    /// Validate that a parquet file has the PAR1 magic header
    fn validate_parquet_file(path: &Path) -> bool {
        match std::fs::read(path) {
            Ok(data) => {
                // Parquet files start with "PAR1" magic bytes
                data.len() >= 4 && &data[..4] == b"PAR1"
            }
            Err(_) => false,
        }
    }

    /// Check for valid parquet files in a directory
    fn has_valid_parquet_in_dir(dir: &Path) -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "parquet" && Self::validate_parquet_file(&path) {
                        return true;
                    }
                    // Arrow files are harder to validate, accept by extension
                    if ext == "arrow" {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl RepoArchetype for HfDatasetArchetype {
    fn name(&self) -> &'static str {
        "hf-dataset"
    }

    fn description(&self) -> &'static str {
        "HuggingFace-compatible dataset with parquet or arrow files"
    }

    fn detect(&self, repo_path: &Path) -> bool {
        // Primary detection: HuggingFace dataset marker (must be valid JSON)
        let dataset_infos = repo_path.join("dataset_infos.json");
        if dataset_infos.exists() && Self::validate_json_file(&dataset_infos) {
            return true;
        }

        // Secondary detection: hyprstream dataset marker (must be valid JSON)
        let dataset_json = repo_path.join("dataset.json");
        if dataset_json.exists() && Self::validate_json_file(&dataset_json) {
            return true;
        }

        // Tertiary detection: valid parquet files in root
        if Self::has_valid_parquet_in_dir(repo_path) {
            return true;
        }

        // Also check common subdirectories
        for subdir in &["data", "train", "test", "validation"] {
            let sub_path = repo_path.join(subdir);
            if sub_path.is_dir() && Self::has_valid_parquet_in_dir(&sub_path) {
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

    /// Create dummy parquet data with valid PAR1 magic header
    fn valid_parquet_data() -> Vec<u8> {
        // PAR1 magic + minimal padding
        let mut data = b"PAR1".to_vec();
        data.extend_from_slice(&[0u8; 100]);
        data
    }

    #[test]
    fn test_detect_with_dataset_infos() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        let marker_path = temp.path().join("dataset_infos.json");
        fs::write(&marker_path, "{}")?;

        let archetype = HfDatasetArchetype;
        assert!(archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_detect_with_dataset_json() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        let marker_path = temp.path().join("dataset.json");
        fs::write(&marker_path, "{}")?;

        let archetype = HfDatasetArchetype;
        assert!(archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_detect_with_parquet() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        let parquet_path = temp.path().join("train.parquet");
        fs::write(&parquet_path, valid_parquet_data())?;

        let archetype = HfDatasetArchetype;
        assert!(archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_detect_with_parquet_in_subdir() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        let data_dir = temp.path().join("data");
        fs::create_dir(&data_dir)?;
        let parquet_path = data_dir.join("train.parquet");
        fs::write(&parquet_path, valid_parquet_data())?;

        let archetype = HfDatasetArchetype;
        assert!(archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_no_detect_empty_dir() -> std::io::Result<()> {
        let temp = TempDir::new()?;

        let archetype = HfDatasetArchetype;
        assert!(!archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_no_detect_invalid_parquet() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        // Parquet file without valid PAR1 header should not match
        let parquet_path = temp.path().join("train.parquet");
        fs::write(&parquet_path, "dummy")?;

        let archetype = HfDatasetArchetype;
        assert!(!archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_no_detect_invalid_json() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        // Invalid JSON should not match
        let marker_path = temp.path().join("dataset_infos.json");
        fs::write(&marker_path, "not valid json")?;

        let archetype = HfDatasetArchetype;
        assert!(!archetype.detect(temp.path()));
        Ok(())
    }

    #[test]
    fn test_capabilities() {
        use super::super::capabilities::Infer;

        let archetype = HfDatasetArchetype;
        let caps = archetype.capabilities();

        assert!(caps.has::<Query>());
        assert!(caps.has::<Write>());
        assert!(caps.has::<Serve>());
        assert!(!caps.has::<Infer>());
    }
}
