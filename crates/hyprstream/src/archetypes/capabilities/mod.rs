//! Type-safe capability system
//!
//! This module provides a type-safe capability system that replaces
//! the bitflags-based `Capabilities` type with zero-sized type markers.
//!
//! # Features
//!
//! - **Compile-time safety**: `has::<Infer>()` catches typos at compile time
//! - **Runtime compatibility**: `has_str("infer")` works for Casbin integration
//! - **Extensible**: Add new capabilities without modifying existing code
//! - **Zero-cost**: Capability markers are zero-sized types
//!
//! # Example
//!
//! ```ignore
//! use hyprstream::archetypes::capabilities::{CapabilitySet, Infer, Train, Serve};
//!
//! // Create a capability set for a model
//! let mut caps = CapabilitySet::new();
//! caps.insert::<Infer>();
//! caps.insert::<Train>();
//! caps.insert::<Serve>();
//!
//! // Type-safe checks
//! if caps.has::<Infer>() {
//!     println!("Can run inference");
//! }
//!
//! // String-based checks (for Casbin)
//! if caps.has_str("infer") {
//!     println!("Policy allows inference");
//! }
//!
//! // Display format
//! println!("Capabilities: {}", caps); // "it--s--"
//! ```
//!
//! # Builder Patterns
//!
//! Common capability sets can be created using builders:
//!
//! ```ignore
//! let model_caps = CapabilitySet::model();  // infer, train, serve
//! let data_caps = CapabilitySet::data();    // query, write, serve
//! let context_caps = CapabilitySet::context(); // context
//! ```

mod core;
mod set;

// Re-export core capability trait and markers
pub use core::{
    is_known_capability, Capability, Context, Infer, LoraSupport, Manage, Multimodal,
    Quantization, Query, Serve, Train, Write, KNOWN_CAPABILITY_IDS,
};

// Re-export capability set
pub use set::{CapabilitySet, UnknownCapability};
