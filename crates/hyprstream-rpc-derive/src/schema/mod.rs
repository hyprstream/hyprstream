//! Cap'n Proto schema parsing (CGR-based).

pub mod cgr_reader;
pub mod types;

pub use cgr_reader::parse_from_cgr;
