//! CAG (Context-Augmented Generation) archetype detection
//!
//! Detects repositories containing a context store for context-augmented generation.
//! Detection is based on presence of `context.json` manifest file.

use super::capabilities::{CapabilitySet, Context};
use super::RepoArchetype;
use std::path::Path;

/// CAG Context archetype
///
/// Detects repos with:
/// - `context.json` (context store manifest)
///
/// The manifest declares the context store configuration:
/// ```json
/// {
///   "version": 1,
///   "store": {
///     "type": "duckdb",
///     "path": "context.duckdb"
///   },
///   "embedding": {
///     "dimension": 384,
///     "model": "sentence-transformers/all-MiniLM-L6-v2"
///   }
/// }
/// ```
///
/// Enables: CONTEXT
///
/// When combined with HfModelArchetype, enables context-augmented inference.
pub struct CagContextArchetype;

impl RepoArchetype for CagContextArchetype {
    fn name(&self) -> &'static str {
        "cag-context"
    }

    fn description(&self) -> &'static str {
        "Context store for context-augmented generation (CAG)"
    }

    fn detect(&self, repo_path: &Path) -> bool {
        // Detection: context.json exists
        let context_manifest = repo_path.join("context.json");
        if context_manifest.exists() {
            // Optionally validate it's valid JSON with expected structure
            if let Ok(content) = std::fs::read_to_string(&context_manifest) {
                // Basic validation: should be valid JSON with version field
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    return json.get("version").is_some();
                }
            }
        }
        false
    }

    fn capabilities(&self) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert::<Context>();
        caps
    }
}

/// Configuration for a CAG context store
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ContextManifest {
    /// Manifest version
    pub version: u32,
    /// Context store configuration
    pub store: StoreConfig,
    /// Embedding configuration
    pub embedding: EmbeddingConfig,
}

/// Context store backend configuration
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct StoreConfig {
    /// Store type: "duckdb"
    #[serde(rename = "type")]
    pub store_type: String,
    /// Path to store file (relative to repo root)
    pub path: String,
}

/// Embedding model configuration
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct EmbeddingConfig {
    /// Embedding dimension
    pub dimension: u32,
    /// Embedding model identifier (e.g., HuggingFace model ID)
    pub model: String,
}

impl ContextManifest {
    /// Load context manifest from a repository path
    pub fn load(repo_path: &Path) -> Result<Self, std::io::Error> {
        let manifest_path = repo_path.join("context.json");
        let content = std::fs::read_to_string(&manifest_path)?;
        serde_json::from_str(&content).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_detect_with_context_json() {
        let temp = TempDir::new().unwrap();
        let manifest_path = temp.path().join("context.json");
        fs::write(
            &manifest_path,
            r#"{
                "version": 1,
                "store": {
                    "type": "duckdb",
                    "path": "context.duckdb"
                },
                "embedding": {
                    "dimension": 384,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }"#,
        )
        .unwrap();

        let archetype = CagContextArchetype;
        assert!(archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_empty_dir() {
        let temp = TempDir::new().unwrap();

        let archetype = CagContextArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_invalid_json() {
        let temp = TempDir::new().unwrap();
        let manifest_path = temp.path().join("context.json");
        fs::write(&manifest_path, "not valid json").unwrap();

        let archetype = CagContextArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_missing_version() {
        let temp = TempDir::new().unwrap();
        let manifest_path = temp.path().join("context.json");
        fs::write(&manifest_path, r#"{"store": {}}"#).unwrap();

        let archetype = CagContextArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_capabilities() {
        use super::super::capabilities::{Infer, Query};

        let archetype = CagContextArchetype;
        let caps = archetype.capabilities();

        assert!(caps.has::<Context>());
        assert!(!caps.has::<Infer>());
        assert!(!caps.has::<Query>());
    }

    #[test]
    fn test_load_manifest() {
        let temp = TempDir::new().unwrap();
        let manifest_path = temp.path().join("context.json");
        fs::write(
            &manifest_path,
            r#"{
                "version": 1,
                "store": {
                    "type": "duckdb",
                    "path": "context.duckdb"
                },
                "embedding": {
                    "dimension": 384,
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }"#,
        )
        .unwrap();

        let manifest = ContextManifest::load(temp.path()).unwrap();
        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.store.store_type, "duckdb");
        assert_eq!(manifest.store.path, "context.duckdb");
        assert_eq!(manifest.embedding.dimension, 384);
    }
}
