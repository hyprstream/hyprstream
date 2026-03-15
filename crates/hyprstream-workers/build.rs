//! Build script for Cap'n Proto schema compilation
//!
//! Compiles worker and workflow schemas with CGR + metadata extraction
//! to support `generate_rpc_service!` proc macro in this crate.

#![allow(clippy::expect_used, clippy::print_stderr)]

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");
    println!("cargo:rerun-if-changed=../hyprstream-rpc/schema/streaming.capnp");

    let schema_dir = Path::new("schema");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");

    hyprstream_rpc_build::compile_schemas(
        schema_dir,
        Path::new(&out_dir),
        &[&rpc_schema_dir],
        &["worker", "workflow"],
    );
}
