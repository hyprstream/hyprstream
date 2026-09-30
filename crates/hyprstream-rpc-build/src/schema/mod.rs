//! Cap'n Proto schema parsing.

pub mod cgr_reader;
pub mod dispatch_label;
pub mod mutation_policy;
pub mod types;

pub use cgr_reader::{parse_from_cgr, parse_from_cgr_path};
pub use cgr_reader::MAXLEN_ZERO_CONTRACT_ERROR;
