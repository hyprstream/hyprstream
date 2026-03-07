//! Release registry manifest types for variant-aware distribution.
//!
//! The release registry is a git2db bare repo (`hyprstream-releases.git`) containing
//! per-variant backends with XET-stored binaries. Manifests describe available variants
//! and their requirements.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Root manifest for the release registry (`manifest.toml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseManifest {
    pub release: ReleaseInfo,
    pub variants: HashMap<String, VariantRequirements>,
}

/// Release-level metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseInfo {
    pub version: String,
    pub libtorch_version: String,
    /// ABI compatibility (e.g., "pre-cxx11", "cxx11")
    pub abi: String,
}

/// Host requirements for a variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantRequirements {
    /// What the host must have (e.g., "glibc", "nvidia-driver", "rocm")
    pub host_requires: String,
}

/// Per-variant manifest (`backends/<variant>/manifest.toml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantManifest {
    pub variant: VariantInfo,
    #[serde(default)]
    pub host: Option<HostRequirements>,
    #[serde(default)]
    pub files: Option<FileCategories>,
}

/// Variant identification and sizing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantInfo {
    pub id: String,
    #[serde(default)]
    pub binary_sha256: Option<String>,
    #[serde(default)]
    pub libtorch_sha256: Option<String>,
    #[serde(default)]
    pub total_size_bytes: Option<u64>,
    #[serde(default)]
    pub runtime_size_bytes: Option<u64>,
}

/// Host requirements for GPU variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostRequirements {
    #[serde(default)]
    pub min_driver_version: Option<String>,
    #[serde(default)]
    pub cuda_compat: Option<String>,
    #[serde(default)]
    pub rocm_version: Option<String>,
}

/// File categories for selective download.
///
/// Only `runtime` files are downloaded by default. `dev` and `test` are skipped
/// to save ~500MB per variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCategories {
    /// Files needed at runtime (binary + .so libs). Downloaded by default.
    #[serde(default)]
    pub runtime: Vec<String>,
    /// Development files (headers, cmake configs). Skipped by default.
    #[serde(default)]
    pub dev: Vec<String>,
    /// Test files (test binaries, python bindings). Skipped by default.
    #[serde(default)]
    pub test: Vec<String>,
}

impl ReleaseManifest {
    /// Parse from a TOML string.
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Read from a file path.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_toml(&content)?)
    }

    /// Get variant IDs that are available.
    pub fn variant_ids(&self) -> Vec<&str> {
        self.variants.keys().map(std::string::String::as_str).collect()
    }
}

impl VariantManifest {
    /// Parse from a TOML string.
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Read from a file path.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_toml(&content)?)
    }

    /// Get pathspec prefixes for selective checkout (runtime files only).
    ///
    /// Returns paths suitable for `DriverOpts::checkout_paths`.
    /// Uses the variant ID from the manifest itself to prevent caller mismatch.
    pub fn runtime_pathspecs(&self) -> Vec<String> {
        let variant_id = &self.variant.id;
        let mut paths = vec![format!("backends/{variant_id}/")];

        // If file categories are specified, use the explicit runtime list
        if let Some(ref files) = self.files {
            if !files.runtime.is_empty() {
                paths.clear();
                for f in &files.runtime {
                    paths.push(format!("backends/{variant_id}/{f}"));
                }
            }
        }

        // Always include root manifest
        paths.push("manifest.toml".to_owned());
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_release_manifest() -> Result<(), Box<dyn std::error::Error>> {
        let toml = r#"
[release]
version = "0.1.0"
libtorch_version = "2.10.0"
abi = "pre-cxx11"

[variants]
cpu = { host_requires = "glibc" }
cuda130 = { host_requires = "nvidia-driver" }
rocm71 = { host_requires = "rocm" }
"#;
        let manifest = ReleaseManifest::from_toml(toml)?;
        assert_eq!(manifest.release.version, "0.1.0");
        assert_eq!(manifest.variants.len(), 3);
        assert!(manifest.variants.contains_key("cuda130"));
        Ok(())
    }

    #[test]
    fn test_parse_variant_manifest() -> Result<(), Box<dyn std::error::Error>> {
        let toml = r#"
[variant]
id = "cuda130"
total_size_bytes = 2500000000
runtime_size_bytes = 1800000000

[host]
min_driver_version = "535.0"
cuda_compat = "13.0"

[files]
runtime = [
    "hyprstream",
    "libtorch/lib/libtorch.so",
    "libtorch/lib/libtorch_cpu.so",
    "libtorch/lib/libtorch_cuda.so",
    "libtorch/lib/libc10.so",
]
dev = ["libtorch/include/**"]
test = ["libtorch/lib/*_test*"]
"#;
        let manifest = VariantManifest::from_toml(toml)?;
        assert_eq!(manifest.variant.id, "cuda130");
        let host = manifest.host.as_ref().ok_or("host missing")?;
        assert_eq!(host.min_driver_version, Some("535.0".to_owned()));
        let paths = manifest.runtime_pathspecs();
        assert!(paths.contains(&"manifest.toml".to_owned()));
        assert!(paths.iter().any(|p| p.contains("libtorch")));
        Ok(())
    }

    #[test]
    fn test_runtime_pathspecs_default() -> Result<(), Box<dyn std::error::Error>> {
        let toml = r#"
[variant]
id = "cpu"
"#;
        let manifest = VariantManifest::from_toml(toml)?;
        let paths = manifest.runtime_pathspecs();
        assert_eq!(paths, vec!["backends/cpu/", "manifest.toml"]);
        Ok(())
    }
}
