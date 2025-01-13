#![cfg_attr(not(target_os = "windows"), doc = include_str!("../README.md"))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod deserialize;
pub mod field;
pub mod serialize;

// The proc macro is implemented in derive_internal, and re-exported by this
// crate. This is because a single crate can not define both a proc macro and a
// macro_rules macro.
#[cfg(feature = "arrow_convert_derive")]
#[doc(hidden)]
pub use arrow_convert_derive::{ArrowDeserialize, ArrowField, ArrowSerialize};

// Test README with doctests
#[cfg_attr(not(target_os = "windows"), doc = include_str!("../README.md"))]
#[cfg(doctest)]
struct ReadmeDoctests;

mod features;
