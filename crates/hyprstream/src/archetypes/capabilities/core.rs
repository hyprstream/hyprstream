//! Core capability trait and zero-sized type markers
//!
//! This module defines the `Capability` trait using a sealed pattern
//! to ensure type safety. Each capability is represented as a zero-sized
//! type (ZST) with compile-time constants for Casbin integration.

use std::any::TypeId;

/// Sealed trait pattern to prevent external implementations
mod private {
    pub trait Sealed {}
}

/// A capability that a domain can provide
///
/// Capabilities are represented as zero-sized types for compile-time
/// type safety while maintaining runtime string compatibility for Casbin.
///
/// # Example
/// ```ignore
/// use hyprstream::archetypes::capabilities::{Capability, Infer};
///
/// // Compile-time constant access
/// assert_eq!(Infer::ID, "infer");
///
/// // Type-safe capability checking
/// fn requires_inference<C: Capability>() { }
/// requires_inference::<Infer>(); // OK
/// ```
pub trait Capability: private::Sealed + Send + Sync + 'static {
    /// The string identifier used in Casbin policies (e.g., "infer")
    const ID: &'static str;

    /// Single character code for display (e.g., 'i')
    const CODE: char;

    /// Human-readable description
    const DESCRIPTION: &'static str;

    /// Get the TypeId for runtime type checking
    fn type_id() -> TypeId {
        TypeId::of::<Self>()
    }
}

// ============================================================================
// Core Capability ZST Markers
// ============================================================================

/// Inference capability - run model inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Infer;

impl private::Sealed for Infer {}
impl Capability for Infer {
    const ID: &'static str = "infer";
    const CODE: char = 'i';
    const DESCRIPTION: &'static str = "Run model inference";
}

/// Training capability - train/fine-tune models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Train;

impl private::Sealed for Train {}
impl Capability for Train {
    const ID: &'static str = "train";
    const CODE: char = 't';
    const DESCRIPTION: &'static str = "Train or fine-tune models";
}

/// Query capability - read/query data via SQL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Query;

impl private::Sealed for Query {}
impl Capability for Query {
    const ID: &'static str = "query";
    const CODE: char = 'q';
    const DESCRIPTION: &'static str = "Query data via SQL";
}

/// Write capability - insert/update data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Write;

impl private::Sealed for Write {}
impl Capability for Write {
    const ID: &'static str = "write";
    const CODE: char = 'w';
    const DESCRIPTION: &'static str = "Write or update data";
}

/// Serve capability - expose via API
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Serve;

impl private::Sealed for Serve {}
impl Capability for Serve {
    const ID: &'static str = "serve";
    const CODE: char = 's';
    const DESCRIPTION: &'static str = "Serve via API";
}

/// Manage capability - administrative operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Manage;

impl private::Sealed for Manage {}
impl Capability for Manage {
    const ID: &'static str = "manage";
    const CODE: char = 'm';
    const DESCRIPTION: &'static str = "Administrative operations";
}

/// Context capability - context-augmented generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Context;

impl private::Sealed for Context {}
impl Capability for Context {
    const ID: &'static str = "context";
    const CODE: char = 'c';
    const DESCRIPTION: &'static str = "Context-augmented generation";
}

// ============================================================================
// Extended Capability Markers (domain-specific)
// ============================================================================

/// LoRA support capability - model supports LoRA adapters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LoraSupport;

impl private::Sealed for LoraSupport {}
impl Capability for LoraSupport {
    const ID: &'static str = "lora_support";
    const CODE: char = 'l';
    const DESCRIPTION: &'static str = "LoRA adapter support";
}

/// Quantization capability - model supports quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Quantization;

impl private::Sealed for Quantization {}
impl Capability for Quantization {
    const ID: &'static str = "quantization";
    const CODE: char = 'z';
    const DESCRIPTION: &'static str = "Quantization support";
}

/// Multimodal capability - model handles multiple modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Multimodal;

impl private::Sealed for Multimodal {}
impl Capability for Multimodal {
    const ID: &'static str = "multimodal";
    const CODE: char = 'v';
    const DESCRIPTION: &'static str = "Multimodal (text, image, etc.)";
}

// ============================================================================
// Known Capabilities Registry
// ============================================================================

/// All known capability IDs for validation
pub static KNOWN_CAPABILITY_IDS: &[&str] = &[
    Infer::ID,
    Train::ID,
    Query::ID,
    Write::ID,
    Serve::ID,
    Manage::ID,
    Context::ID,
    LoraSupport::ID,
    Quantization::ID,
    Multimodal::ID,
];

/// Check if a capability ID is known
pub fn is_known_capability(id: &str) -> bool {
    KNOWN_CAPABILITY_IDS.contains(&id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_ids() {
        assert_eq!(Infer::ID, "infer");
        assert_eq!(Train::ID, "train");
        assert_eq!(Query::ID, "query");
        assert_eq!(Write::ID, "write");
        assert_eq!(Serve::ID, "serve");
        assert_eq!(Manage::ID, "manage");
        assert_eq!(Context::ID, "context");
    }

    #[test]
    fn test_capability_codes() {
        assert_eq!(Infer::CODE, 'i');
        assert_eq!(Train::CODE, 't');
        assert_eq!(Query::CODE, 'q');
        assert_eq!(Write::CODE, 'w');
        assert_eq!(Serve::CODE, 's');
        assert_eq!(Manage::CODE, 'm');
        assert_eq!(Context::CODE, 'c');
    }

    #[test]
    fn test_type_ids_are_unique() {
        let ids = [
            Infer::type_id(),
            Train::type_id(),
            Query::type_id(),
            Write::type_id(),
            Serve::type_id(),
            Manage::type_id(),
            Context::type_id(),
        ];

        // Check all are unique
        for (i, id1) in ids.iter().enumerate() {
            for (j, id2) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(id1, id2, "TypeIds should be unique");
                }
            }
        }
    }

    #[test]
    fn test_known_capabilities() {
        assert!(is_known_capability("infer"));
        assert!(is_known_capability("train"));
        assert!(is_known_capability("context"));
        assert!(!is_known_capability("unknown"));
        assert!(!is_known_capability(""));
    }

    #[test]
    fn test_zst_size() {
        // Zero-sized types should have size 0
        assert_eq!(std::mem::size_of::<Infer>(), 0);
        assert_eq!(std::mem::size_of::<Train>(), 0);
        assert_eq!(std::mem::size_of::<Context>(), 0);
    }
}
