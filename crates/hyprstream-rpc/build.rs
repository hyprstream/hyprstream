//! Build script for hyprstream-rpc
//!
//! Compiles Cap'n Proto schemas for RPC envelope types.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");

    let schema_dir = Path::new("schema");

    // Skip if schema directory doesn't exist
    if !schema_dir.exists() {
        return;
    }

    // Compile common schema (identity, envelope)
    let common_schema = schema_dir.join("common.capnp");
    if common_schema.exists() {
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&common_schema)
            .run()
            .expect("failed to compile common.capnp");
    }
}
