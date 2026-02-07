//! Cap'n Proto schema parsing.

pub mod parser;
pub mod types;

pub use parser::{parse_capnp_schema, merge_annotations_from_metadata};
