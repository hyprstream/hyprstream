//! Archetype/Domain registry with caching
//!
//! Provides efficient domain detection with caching to avoid
//! repeated filesystem scans.

use super::capabilities::{Capability, CapabilitySet};
use super::{builtin_archetypes, RepoArchetype};
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// New Domain-based types (Phase 4)
// ============================================================================

/// Detected domains and their capabilities
///
/// This is the new type-safe replacement for `DetectedArchetypes`.
/// Uses `CapabilitySet` for type-safe capability checks.
#[derive(Debug, Clone)]
pub struct DetectedDomains {
    /// Names of detected domains (e.g., "HfModel", "DuckDb")
    pub domains: Vec<&'static str>,
    /// Combined capabilities from all detected domains
    pub capabilities: CapabilitySet,
    /// Per-domain capabilities
    pub domain_capabilities: HashMap<&'static str, CapabilitySet>,
    /// When this detection was performed
    pub detected_at: Instant,
}

impl DetectedDomains {
    /// Create an empty detection result
    pub fn empty() -> Self {
        Self {
            domains: Vec::new(),
            capabilities: CapabilitySet::new(),
            domain_capabilities: HashMap::new(),
            detected_at: Instant::now(),
        }
    }

    /// Check if any domains were detected
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }

    /// Type-safe capability check
    ///
    /// # Example
    /// ```ignore
    /// if detected.has::<Infer>() {
    ///     // Can run inference
    /// }
    /// ```
    pub fn has<C: Capability>(&self) -> bool {
        self.capabilities.has::<C>()
    }

    /// String-based capability check (for Casbin integration)
    pub fn has_str(&self, id: &str) -> bool {
        self.capabilities.has_str(id)
    }

    /// Check if capability is available in a specific domain
    pub fn has_in_domain<C: Capability>(&self, domain: &str) -> bool {
        self.domain_capabilities
            .get(domain)
            .map(crate::archetypes::CapabilitySet::has::<C>)
            .unwrap_or(false)
    }

    /// Get domains that provide a specific capability
    pub fn domains_for<C: Capability>(&self) -> Vec<&'static str> {
        self.domain_capabilities
            .iter()
            .filter(|(_, caps)| caps.has::<C>())
            .map(|(domain, _)| *domain)
            .collect()
    }

    /// Get domains that provide a capability by string ID
    pub fn domains_for_str(&self, capability: &str) -> Vec<&'static str> {
        self.domain_capabilities
            .iter()
            .filter(|(_, caps)| caps.has_str(capability))
            .map(|(domain, _)| *domain)
            .collect()
    }

    /// Get domain names as comma-separated string for display
    pub fn domains_display(&self) -> String {
        self.domains.join(",")
    }
}

impl Default for DetectedDomains {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// DetectedArchetypes - uses CapabilitySet directly
// ============================================================================

/// Cached archetype detection result
#[derive(Debug, Clone)]
pub struct DetectedArchetypes {
    /// Names of matched archetypes
    pub archetypes: Vec<&'static str>,
    /// Combined capabilities from all matched archetypes
    pub capabilities: CapabilitySet,
    /// When this detection was performed
    pub detected_at: Instant,
}

impl DetectedArchetypes {
    /// Check if any archetypes were detected
    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Type-safe capability check
    pub fn has<C: Capability>(&self) -> bool {
        self.capabilities.has::<C>()
    }

    /// String-based capability check (for Casbin integration)
    pub fn has_str(&self, id: &str) -> bool {
        self.capabilities.has_str(id)
    }

    /// Get archetype names as comma-separated string for display
    pub fn archetypes_display(&self) -> String {
        self.archetypes.join(",")
    }

    /// Convert to `DetectedDomains` (with per-domain capability info)
    pub fn to_detected_domains(&self) -> DetectedDomains {
        DetectedDomains {
            domains: self.archetypes.clone(),
            capabilities: self.capabilities.clone(),
            domain_capabilities: HashMap::new(), // Would need archetype info to populate
            detected_at: self.detected_at,
        }
    }
}

/// Registry for archetype detection with caching
///
/// Caches detection results per-path with configurable TTL.
pub struct ArchetypeRegistry {
    archetypes: Vec<Box<dyn RepoArchetype>>,
    cache: DashMap<PathBuf, DetectedArchetypes>,
    cache_ttl: Duration,
}

impl Default for ArchetypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchetypeRegistry {
    /// Create a new registry with built-in archetypes
    pub fn new() -> Self {
        Self {
            archetypes: builtin_archetypes(),
            cache: DashMap::new(),
            cache_ttl: Duration::from_secs(60), // 1 minute default
        }
    }

    /// Create a registry with custom TTL
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            archetypes: builtin_archetypes(),
            cache: DashMap::new(),
            cache_ttl: ttl,
        }
    }

    /// Detect archetypes for a repository path
    ///
    /// Returns cached result if available and not expired.
    pub fn detect(&self, repo_path: &Path) -> DetectedArchetypes {
        let canonical = match repo_path.canonicalize() {
            Ok(p) => p,
            Err(_) => repo_path.to_path_buf(),
        };

        // Check cache
        if let Some(cached) = self.cache.get(&canonical) {
            if cached.detected_at.elapsed() < self.cache_ttl {
                return cached.clone();
            }
        }

        // Perform detection
        let mut detected = DetectedArchetypes {
            archetypes: Vec::new(),
            capabilities: CapabilitySet::new(),
            detected_at: Instant::now(),
        };

        for archetype in &self.archetypes {
            if archetype.detect(&canonical) {
                detected.archetypes.push(archetype.name());
                detected.capabilities = detected.capabilities.union(&archetype.capabilities());
            }
        }

        // Cache result
        self.cache.insert(canonical, detected.clone());

        detected
    }

    /// Get capabilities for a repository path
    ///
    /// Shorthand for `detect(path).capabilities`
    pub fn capabilities(&self, repo_path: &Path) -> CapabilitySet {
        self.detect(repo_path).capabilities
    }

    /// Check if a repository has a specific capability (type-safe)
    pub fn has<C: Capability>(&self, repo_path: &Path) -> bool {
        self.capabilities(repo_path).has::<C>()
    }

    /// Invalidate cache for a specific path
    pub fn invalidate(&self, repo_path: &Path) {
        if let Ok(canonical) = repo_path.canonicalize() {
            self.cache.remove(&canonical);
        }
        self.cache.remove(&repo_path.to_path_buf());
    }

    /// Invalidate entire cache
    pub fn invalidate_all(&self) {
        self.cache.clear();
    }

    /// Get list of registered archetype names
    pub fn archetype_names(&self) -> Vec<&'static str> {
        self.archetypes.iter().map(|a| a.name()).collect()
    }
}

/// Global archetype registry instance
static REGISTRY: std::sync::OnceLock<Arc<ArchetypeRegistry>> = std::sync::OnceLock::new();

/// Get the global archetype registry
pub fn global_registry() -> Arc<ArchetypeRegistry> {
    REGISTRY
        .get_or_init(|| Arc::new(ArchetypeRegistry::new()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archetypes::{Context, Infer, Query, Serve, Train};
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_detect_model() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        // Valid model config requires model_type or architectures field
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        assert!(detected.archetypes.contains(&"hf-model"));
        assert!(detected.has::<Infer>());
        Ok(())
    }

    #[test]
    fn test_detect_dataset() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        fs::write(temp.path().join("dataset_infos.json"), "{}")?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        assert!(detected.archetypes.contains(&"hf-dataset"));
        assert!(detected.has::<Query>());
        Ok(())
    }

    /// Create valid-sized DuckDB data (minimum 4KB)
    fn valid_duckdb_data() -> Vec<u8> {
        vec![0u8; 4096]
    }

    #[test]
    fn test_detect_duckdb() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        fs::write(temp.path().join("metrics.duckdb"), valid_duckdb_data())?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        assert!(detected.archetypes.contains(&"duckdb"));
        assert!(detected.has::<Query>());
        Ok(())
    }

    #[test]
    fn test_detect_multiple() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        // Both model and database
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;
        fs::write(temp.path().join("metrics.duckdb"), valid_duckdb_data())?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        assert!(detected.archetypes.contains(&"hf-model"));
        assert!(detected.archetypes.contains(&"duckdb"));
        assert!(detected.has::<Infer>());
        assert!(detected.has::<Query>());
        Ok(())
    }

    #[test]
    fn test_cache_invalidation() -> std::io::Result<()> {
        let temp = TempDir::new()?;

        let registry = ArchetypeRegistry::new();

        // Initially empty
        let detected = registry.detect(temp.path());
        assert!(detected.is_empty());

        // Add a valid model config
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;

        // Still cached as empty
        let detected = registry.detect(temp.path());
        assert!(detected.is_empty());

        // Invalidate and re-detect
        registry.invalidate(temp.path());
        let detected = registry.detect(temp.path());
        assert!(!detected.is_empty());
        assert!(detected.archetypes.contains(&"hf-model"));
        Ok(())
    }

    #[test]
    fn test_detect_cag_context() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        fs::write(
            temp.path().join("context.json"),
            r#"{"version": 1, "store": {"type": "duckdb", "path": "ctx.db"}, "embedding": {"dimension": 384, "model": "test"}}"#,
        )?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        assert!(detected.archetypes.contains(&"cag-context"));
        assert!(detected.has::<Context>());
        Ok(())
    }

    #[test]
    fn test_detect_model_with_cag() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        // Model + CAG context - model requires valid config
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;
        fs::write(
            temp.path().join("context.json"),
            r#"{"version": 1, "store": {"type": "duckdb", "path": "ctx.db"}, "embedding": {"dimension": 384, "model": "test"}}"#,
        )?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        assert!(detected.archetypes.contains(&"hf-model"));
        assert!(detected.archetypes.contains(&"cag-context"));
        assert!(detected.has::<Infer>());
        assert!(detected.has::<Context>());
        // Should show detected archetypes
        assert!(detected.archetypes_display().contains("hf-model"));
        assert!(detected.archetypes_display().contains("cag-context"));
        Ok(())
    }

    // ========================================================================
    // DetectedDomains tests (new type-safe API)
    // ========================================================================

    #[test]
    fn test_detected_domains_type_safe_has() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path());

        // Convert to new type
        let domains = detected.to_detected_domains();

        // Type-safe checks
        assert!(domains.has::<Infer>());
        assert!(domains.has::<Train>());
        assert!(domains.has::<Serve>());
        assert!(!domains.has::<Query>());
        Ok(())
    }

    #[test]
    fn test_detected_domains_string_has() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;

        let registry = ArchetypeRegistry::new();
        let detected = registry.detect(temp.path()).to_detected_domains();

        // String-based checks (for Casbin integration)
        assert!(detected.has_str("infer"));
        assert!(detected.has_str("train"));
        assert!(!detected.has_str("query"));
        assert!(!detected.has_str("unknown"));
        Ok(())
    }

    #[test]
    fn test_detected_domains_display() -> std::io::Result<()> {
        let temp = TempDir::new()?;
        fs::write(temp.path().join("config.json"), r#"{"model_type": "llama"}"#)?;
        fs::write(
            temp.path().join("context.json"),
            r#"{"version": 1, "store": {"type": "duckdb", "path": "ctx.db"}, "embedding": {"dimension": 384, "model": "test"}}"#,
        )?;

        let registry = ArchetypeRegistry::new();
        let domains = registry.detect(temp.path()).to_detected_domains();

        // Display shows domain names
        assert!(domains.domains_display().contains("hf-model"));
        assert!(domains.domains_display().contains("cag-context"));
        Ok(())
    }

    #[test]
    fn test_detected_domains_empty() {
        let domains = DetectedDomains::empty();
        assert!(domains.is_empty());
        assert!(!domains.has::<Infer>());
        assert_eq!(domains.domains_display(), "");
    }
}
