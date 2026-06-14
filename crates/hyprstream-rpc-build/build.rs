//! Compile `StreamContract` (#213/#216) typed readers for the annotation extractor.
//!
//! The struct-valued `$streamPolicy` annotation can't be decoded generically from the
//! CGR — capnp-rust's dynamic reflection needs a `StructSchema` from *generated*
//! introspection (`StructSchema::new(RawBrandedStructSchema)`), with no CGR constructor.
//! So we generate the real `streaming.capnp` types (single source of truth) here and the
//! extractor decodes the annotation value via the typed `stream_contract::Reader`.
//!
//! Schemas live in the sibling `hyprstream-rpc` crate; we compile both `annotations.capnp`
//! (imported by `streaming.capnp`) and `streaming.capnp`.

use std::path::Path;

fn main() {
    let schema_dir = Path::new("../hyprstream-rpc/schema");
    for f in ["annotations.capnp", "streaming.capnp"] {
        println!("cargo:rerun-if-changed=../hyprstream-rpc/schema/{f}");
    }
    if let Err(e) = capnpc::CompilerCommand::new()
        .src_prefix(schema_dir)
        .import_path(schema_dir)
        .file(schema_dir.join("annotations.capnp"))
        .file(schema_dir.join("streaming.capnp"))
        .run()
    {
        panic!("compile streaming.capnp for StreamContract decode (#216): {e}");
    }
}
