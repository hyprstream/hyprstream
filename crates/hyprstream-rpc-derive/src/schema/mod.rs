//! Cap'n Proto schema parsing.

pub mod cgr_reader;
pub mod parser;
pub mod types;

pub use cgr_reader::parse_from_cgr;
pub use parser::{collect_list_struct_types, merge_annotations_from_metadata, parse_capnp_schema};
