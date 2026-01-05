//! CapabilitySet - Type-safe capability container with dual lookup
//!
//! Provides both compile-time type safety via generics and runtime
//! string-based lookup for Casbin integration.

use super::core::{is_known_capability, Capability};
use std::any::TypeId;
use std::collections::HashSet;
use std::fmt;

/// Error returned when inserting an unknown capability
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownCapability(pub String);

impl fmt::Display for UnknownCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown capability: {}", self.0)
    }
}

impl std::error::Error for UnknownCapability {}

/// A type-safe container for capabilities
///
/// CapabilitySet provides dual lookup:
/// - `has::<Infer>()` - Compile-time verified, catches typos
/// - `has_str("infer")` - Runtime lookup for Casbin integration
///
/// # Example
/// ```ignore
/// use hyprstream::archetypes::capabilities::{CapabilitySet, Infer, Train};
///
/// let mut caps = CapabilitySet::new();
/// caps.insert::<Infer>();
/// caps.insert::<Train>();
///
/// // Type-safe check (compile-time verified)
/// assert!(caps.has::<Infer>());
///
/// // String-based check (for Casbin)
/// assert!(caps.has_str("infer"));
///
/// // Display format (comma-separated IDs)
/// assert_eq!(format!("{}", caps), "infer,train");
/// ```
#[derive(Debug, Clone, Default)]
pub struct CapabilitySet {
    /// String-based lookup (for Casbin integration)
    by_id: HashSet<&'static str>,
    /// Type-based lookup (for compile-time safety)
    by_type: HashSet<TypeId>,
}

impl CapabilitySet {
    /// Create an empty capability set
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from static slice of capability IDs
    ///
    /// Used by domains to define their base capabilities.
    /// Note: Type information is not preserved when creating from strings.
    pub fn from_static(ids: &[&'static str]) -> Self {
        Self {
            by_id: ids.iter().copied().collect(),
            by_type: HashSet::new(), // No type info from strings
        }
    }

    /// Insert a capability (type-safe)
    ///
    /// # Example
    /// ```ignore
    /// caps.insert::<Infer>();
    /// ```
    pub fn insert<C: Capability>(&mut self) {
        self.by_id.insert(C::ID);
        self.by_type.insert(TypeId::of::<C>());
    }

    /// Try to insert a capability by string ID
    ///
    /// Returns error if the capability ID is unknown.
    pub fn try_insert_str(&mut self, id: &'static str) -> Result<(), UnknownCapability> {
        if is_known_capability(id) {
            self.by_id.insert(id);
            Ok(())
        } else {
            Err(UnknownCapability(id.to_string()))
        }
    }

    /// Insert a capability by string ID (unchecked)
    ///
    /// Use `try_insert_str` for validation, or this for performance
    /// when you know the ID is valid.
    pub fn insert_str_unchecked(&mut self, id: &'static str) {
        self.by_id.insert(id);
    }

    /// Check if capability is present (type-safe)
    ///
    /// This is the preferred method as it catches typos at compile time.
    ///
    /// # Example
    /// ```ignore
    /// if caps.has::<Infer>() {
    ///     // Safe to run inference
    /// }
    /// ```
    pub fn has<C: Capability>(&self) -> bool {
        // Try type-based first (more reliable)
        if self.by_type.contains(&TypeId::of::<C>()) {
            return true;
        }
        // Fall back to string-based (for caps loaded from strings)
        self.by_id.contains(C::ID)
    }

    /// Check if capability is present by string ID
    ///
    /// Use for Casbin integration or when capability type is not known
    /// at compile time.
    pub fn has_str(&self, id: &str) -> bool {
        self.by_id.contains(id)
    }

    /// Remove a capability (type-safe)
    pub fn remove<C: Capability>(&mut self) -> bool {
        let removed_type = self.by_type.remove(&TypeId::of::<C>());
        let removed_id = self.by_id.remove(C::ID);
        removed_type || removed_id
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Get number of capabilities
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Iterate over capability IDs
    pub fn iter(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_id.iter().copied()
    }

    /// Get all capability IDs as a Vec
    pub fn to_vec(&self) -> Vec<&'static str> {
        self.by_id.iter().copied().collect()
    }

    /// Union with another set
    pub fn union(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            by_id: self.by_id.union(&other.by_id).copied().collect(),
            by_type: self.by_type.union(&other.by_type).copied().collect(),
        }
    }

    /// Intersection with another set
    pub fn intersection(&self, other: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            by_id: self.by_id.intersection(&other.by_id).copied().collect(),
            by_type: self.by_type.intersection(&other.by_type).copied().collect(),
        }
    }

    /// Get capability IDs as a sorted vector
    pub fn to_ids(&self) -> Vec<&'static str> {
        let mut ids: Vec<_> = self.by_id.iter().copied().collect();
        ids.sort();
        ids
    }
}

impl fmt::Display for CapabilitySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ids().join(","))
    }
}

impl PartialEq for CapabilitySet {
    fn eq(&self, other: &Self) -> bool {
        self.by_id == other.by_id
    }
}

impl Eq for CapabilitySet {}

// ============================================================================
// Builder pattern for common capability sets
// ============================================================================

impl CapabilitySet {
    /// Create a capability set for a model domain (infer, train, serve)
    pub fn model() -> Self {
        use super::core::{Infer, Serve, Train};
        let mut set = Self::new();
        set.insert::<Infer>();
        set.insert::<Train>();
        set.insert::<Serve>();
        set
    }

    /// Create a capability set for a data domain (query, write, serve)
    pub fn data() -> Self {
        use super::core::{Query, Serve, Write};
        let mut set = Self::new();
        set.insert::<Query>();
        set.insert::<Write>();
        set.insert::<Serve>();
        set
    }

    /// Create a capability set for context-augmented generation
    pub fn context() -> Self {
        use super::core::Context;
        let mut set = Self::new();
        set.insert::<Context>();
        set
    }

    /// Create a full capability set (all core capabilities)
    pub fn full() -> Self {
        use super::core::{Context, Infer, Manage, Query, Serve, Train, Write};
        let mut set = Self::new();
        set.insert::<Infer>();
        set.insert::<Train>();
        set.insert::<Query>();
        set.insert::<Write>();
        set.insert::<Serve>();
        set.insert::<Manage>();
        set.insert::<Context>();
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archetypes::capabilities::{Context, Infer, Manage, Query, Serve, Train, Write};

    #[test]
    fn test_insert_and_has() {
        let mut caps = CapabilitySet::new();
        assert!(!caps.has::<Infer>());

        caps.insert::<Infer>();
        assert!(caps.has::<Infer>());
        assert!(!caps.has::<Train>());
    }

    #[test]
    fn test_has_str() {
        let mut caps = CapabilitySet::new();
        caps.insert::<Infer>();

        assert!(caps.has_str("infer"));
        assert!(!caps.has_str("train"));
        assert!(!caps.has_str("unknown"));
    }

    #[test]
    fn test_from_static() {
        let caps = CapabilitySet::from_static(&["infer", "train", "serve"]);

        assert!(caps.has_str("infer"));
        assert!(caps.has_str("train"));
        assert!(caps.has_str("serve"));
        assert!(!caps.has_str("query"));

        // String-based lookup works
        assert!(caps.has::<Infer>());
    }

    #[test]
    fn test_try_insert_str() {
        let mut caps = CapabilitySet::new();

        assert!(caps.try_insert_str("infer").is_ok());
        assert!(caps.has_str("infer"));

        assert!(caps.try_insert_str("unknown").is_err());
    }

    #[test]
    fn test_display() {
        let mut caps = CapabilitySet::new();
        caps.insert::<Infer>();
        caps.insert::<Train>();
        caps.insert::<Serve>();
        caps.insert::<Context>();

        // Display shows comma-separated sorted IDs
        let display = format!("{}", caps);
        assert!(display.contains("infer"));
        assert!(display.contains("train"));
        assert!(display.contains("serve"));
        assert!(display.contains("context"));
    }

    #[test]
    fn test_to_ids() {
        let mut caps = CapabilitySet::new();
        caps.insert::<Infer>();
        caps.insert::<Train>();

        let ids = caps.to_ids();
        assert_eq!(ids, vec!["infer", "train"]);
    }

    #[test]
    fn test_model_builder() {
        let caps = CapabilitySet::model();

        assert!(caps.has::<Infer>());
        assert!(caps.has::<Train>());
        assert!(caps.has::<Serve>());
        assert!(!caps.has::<Query>());
    }

    #[test]
    fn test_data_builder() {
        let caps = CapabilitySet::data();

        assert!(caps.has::<Query>());
        assert!(caps.has::<Write>());
        assert!(caps.has::<Serve>());
        assert!(!caps.has::<Infer>());
    }

    #[test]
    fn test_union() {
        let model = CapabilitySet::model();
        let data = CapabilitySet::data();
        let combined = model.union(&data);

        assert!(combined.has::<Infer>());
        assert!(combined.has::<Train>());
        assert!(combined.has::<Query>());
        assert!(combined.has::<Write>());
        assert!(combined.has::<Serve>());
    }

    #[test]
    fn test_intersection() {
        let model = CapabilitySet::model();
        let data = CapabilitySet::data();
        let common = model.intersection(&data);

        // Only serve is common
        assert!(!common.has::<Infer>());
        assert!(!common.has::<Query>());
        assert!(common.has::<Serve>());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut caps = CapabilitySet::new();
        assert!(caps.is_empty());
        assert_eq!(caps.len(), 0);

        caps.insert::<Infer>();
        assert!(!caps.is_empty());
        assert_eq!(caps.len(), 1);

        caps.insert::<Train>();
        assert_eq!(caps.len(), 2);
    }

    #[test]
    fn test_remove() {
        let mut caps = CapabilitySet::new();
        caps.insert::<Infer>();
        caps.insert::<Train>();

        assert!(caps.remove::<Infer>());
        assert!(!caps.has::<Infer>());
        assert!(caps.has::<Train>());

        // Remove non-existent
        assert!(!caps.remove::<Query>());
    }

    #[test]
    fn test_equality() {
        let mut a = CapabilitySet::new();
        let mut b = CapabilitySet::new();

        a.insert::<Infer>();
        a.insert::<Train>();

        b.insert::<Train>();
        b.insert::<Infer>();

        assert_eq!(a, b);
    }

    #[test]
    fn test_full_builder() {
        let caps = CapabilitySet::full();
        assert_eq!(caps.to_ids().len(), 7);
        assert!(caps.has::<Infer>());
        assert!(caps.has::<Train>());
        assert!(caps.has::<Query>());
        assert!(caps.has::<Write>());
        assert!(caps.has::<Serve>());
        assert!(caps.has::<Manage>());
        assert!(caps.has::<Context>());
    }
}
