//! HuggingFace Model archetype detection
//!
//! Detects repositories containing HuggingFace-compatible models.
//! Detection is based on presence of `config.json` and weight files
//! (`.safetensors`, `.bin`).

use super::capabilities::{CapabilitySet, Infer, Serve, Train};
use super::RepoArchetype;
use std::path::Path;

/// HuggingFace Model archetype
///
/// Detects repos with:
/// - `config.json` containing `model_type` or `architectures` field
/// - Optional `*.safetensors` or `*.bin` (model weights)
/// - Optional `adapters/` directory (LoRA support)
///
/// Note: Detection requires valid model config structure to prevent
/// privilege escalation via empty config.json files.
///
/// Enables: INFER, TRAIN, SERVE
pub struct HfModelArchetype;

impl HfModelArchetype {
    /// Validate that config.json contains required HuggingFace model fields
    ///
    /// A valid HF model config must have either:
    /// - `model_type` field (e.g., "llama", "qwen2")
    /// - `architectures` field (e.g., ["Qwen2ForCausalLM"])
    fn validate_model_config(config_path: &Path) -> bool {
        let content = match std::fs::read_to_string(config_path) {
            Ok(c) => c,
            Err(_) => return false,
        };

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(_) => return false,
        };

        // Require model_type OR architectures field
        let has_model_type = json.get("model_type").is_some();
        let has_architectures = json.get("architectures").is_some();

        has_model_type || has_architectures
    }
}

impl RepoArchetype for HfModelArchetype {
    fn name(&self) -> &'static str {
        "hf-model"
    }

    fn description(&self) -> &'static str {
        "HuggingFace-compatible model with config.json and weights"
    }

    fn detect(&self, repo_path: &Path) -> bool {
        let config_path = repo_path.join("config.json");

        // Primary detection: config.json with valid model structure
        if config_path.exists() && Self::validate_model_config(&config_path) {
            return true;
        }

        // Secondary detection: safetensors files exist
        // (some models may not have config.json in root)
        // Note: This is less secure but maintains backward compatibility
        // for models that ship weights without root config.json
        if let Ok(entries) = std::fs::read_dir(repo_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "safetensors" {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn capabilities(&self) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        caps.insert::<Infer>();
        caps.insert::<Train>();
        caps.insert::<Serve>();
        caps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_detect_with_config_json() {
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.json");
        fs::write(&config_path, r#"{"model_type": "llama"}"#).unwrap();

        let archetype = HfModelArchetype;
        assert!(archetype.detect(temp.path()));
    }

    #[test]
    fn test_detect_with_safetensors() {
        let temp = TempDir::new().unwrap();
        let weights_path = temp.path().join("model.safetensors");
        fs::write(&weights_path, "dummy").unwrap();

        let archetype = HfModelArchetype;
        assert!(archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_empty_dir() {
        let temp = TempDir::new().unwrap();

        let archetype = HfModelArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_capabilities() {
        use super::super::capabilities::Query;

        let archetype = HfModelArchetype;
        let caps = archetype.capabilities();

        assert!(caps.has::<Infer>());
        assert!(caps.has::<Train>());
        assert!(caps.has::<Serve>());
        assert!(!caps.has::<Query>());
    }

    #[test]
    fn test_no_detect_empty_config_json() {
        // Security test: empty config.json should NOT grant HfModel domain
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.json");
        fs::write(&config_path, "{}").unwrap();

        let archetype = HfModelArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_no_detect_invalid_config_json() {
        // Security test: config.json without model_type/architectures should NOT match
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.json");
        fs::write(&config_path, r#"{"some_field": "value"}"#).unwrap();

        let archetype = HfModelArchetype;
        assert!(!archetype.detect(temp.path()));
    }

    #[test]
    fn test_detect_with_architectures_field() {
        // config.json with architectures field should match
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.json");
        fs::write(&config_path, r#"{"architectures": ["Qwen2ForCausalLM"]}"#).unwrap();

        let archetype = HfModelArchetype;
        assert!(archetype.detect(temp.path()));
    }
}
