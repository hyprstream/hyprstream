//! XET large file storage filter integration
//!
//! This module re-exports the git-xet-filter crate for convenience.
//! The filter can also be used standalone by depending on git-xet-filter directly.

#[cfg(feature = "xet-storage")]
pub use git_xet_filter::*;
