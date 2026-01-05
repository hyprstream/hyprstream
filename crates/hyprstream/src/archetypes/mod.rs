//! Repository archetype detection and capability system
//!
//! Archetypes provide internal capability detection for repositories.
//! They detect what kind of content a repo contains and what features
//! can be enabled for it.
//!
//! # Design Philosophy
//!
//! - Archetypes are **internal** - not user-facing CLI concepts
//! - Detection uses **convention over configuration** - no manifest files
//! - Archetypes are **composable** - a repo can match multiple
//! - Capabilities **gate features** - inference, query, training, etc.
//!
//! # Capability Display Format
//!
//! Capabilities are displayed as comma-separated IDs (e.g., `infer,train,serve`).
//! Available capabilities:
//! - `infer` = run model inference
//! - `train` = modify model weights
//! - `query` = SELECT from data
//! - `write` = INSERT/UPDATE data
//! - `serve` = expose via API
//! - `manage` = admin operations
//! - `context` = context-augmented generation
//!
//! # Built-in Archetypes
//!
//! | Archetype | Detects | Capabilities |
//! |-----------|---------|--------------|
//! | HfModel | config.json + *.safetensors | INFER, TRAIN, SERVE |
//! | HfDataset | dataset_infos.json, *.parquet | QUERY, WRITE, SERVE |
//! | DuckDb | *.duckdb | QUERY, WRITE, SERVE |
//! | CagContext | context.json | CONTEXT |

mod cag;
pub mod capabilities;
pub mod domain;
mod duckdb;
mod hf_dataset;
mod hf_model;
mod registry;

pub use cag::CagContextArchetype;
pub use duckdb::DuckDbArchetype;
pub use hf_dataset::HfDatasetArchetype;
pub use hf_model::HfModelArchetype;
pub use registry::{global_registry, ArchetypeRegistry, DetectedArchetypes, DetectedDomains};

// Re-export type-safe capability system
pub use capabilities::{
    Capability, CapabilitySet, Context, Infer, Manage, Query, Serve, Train, Write,
};

// Re-export domain types
pub use domain::{DetectionError, DetectionResult, Domain, SimpleDomain};

use std::path::Path;

/// Trait for repository archetype detection
///
/// Implementations detect whether a repository matches a particular
/// archetype based on file patterns (convention over configuration).
pub trait RepoArchetype: Send + Sync {
    /// Unique identifier for this archetype
    fn name(&self) -> &'static str;

    /// Human-readable description
    fn description(&self) -> &'static str;

    /// Detect if a repository path matches this archetype
    ///
    /// Should check for marker files/patterns without loading heavy data.
    fn detect(&self, repo_path: &Path) -> bool;

    /// Capabilities this archetype enables
    fn capabilities(&self) -> CapabilitySet;
}

/// Get all built-in archetypes
pub fn builtin_archetypes() -> Vec<Box<dyn RepoArchetype>> {
    vec![
        Box::new(HfModelArchetype),
        Box::new(HfDatasetArchetype),
        Box::new(DuckDbArchetype),
        Box::new(CagContextArchetype),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_display() {
        // Model capabilities: infer + train
        let mut model = CapabilitySet::new();
        model.insert::<Infer>();
        model.insert::<Train>();
        assert_eq!(format!("{}", model), "infer,train");

        // Data capabilities: query + write
        let mut data = CapabilitySet::new();
        data.insert::<Query>();
        data.insert::<Write>();
        assert_eq!(format!("{}", data), "query,write");

        // Full capabilities (sorted alphabetically)
        let mut full = CapabilitySet::new();
        full.insert::<Infer>();
        full.insert::<Train>();
        full.insert::<Query>();
        full.insert::<Write>();
        full.insert::<Serve>();
        full.insert::<Manage>();
        full.insert::<Context>();
        assert_eq!(format!("{}", full), "context,infer,manage,query,serve,train,write");

        // Empty
        let empty = CapabilitySet::new();
        assert_eq!(format!("{}", empty), "");

        // Context capability alone
        let mut context = CapabilitySet::new();
        context.insert::<Context>();
        assert_eq!(format!("{}", context), "context");
    }

    #[test]
    fn test_capabilities_composable() {
        let mut model = CapabilitySet::new();
        model.insert::<Infer>();
        model.insert::<Train>();
        model.insert::<Serve>();

        let mut data = CapabilitySet::new();
        data.insert::<Query>();
        data.insert::<Write>();
        data.insert::<Serve>();

        // Repos can have multiple capabilities (model + data)
        let combined = model.union(&data);
        assert!(combined.has::<Infer>());
        assert!(combined.has::<Query>());
        assert!(combined.has::<Serve>());
        assert_eq!(format!("{}", combined), "infer,query,serve,train,write");

        // Model with CAG context
        let mut model_with_cag = model.clone();
        model_with_cag.insert::<Context>();
        assert_eq!(format!("{}", model_with_cag), "context,infer,serve,train");
    }
}
