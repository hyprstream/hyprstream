//! Cap'n Proto schema parsing.

pub mod cgr_reader;
pub mod types;

pub use cgr_reader::{parse_from_cgr, parse_from_cgr_path};
