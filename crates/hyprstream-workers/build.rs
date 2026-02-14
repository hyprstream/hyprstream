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

    // Note: annotations.capnp lives in hyprstream-rpc/schema/ and is shared via import_path.
    // Local schemas use absolute imports (e.g., /annotations.capnp) to resolve from there.

    // Compile schemas with CGR + metadata extraction for generate_rpc_service!
    for name in ["worker", "workflow"] {
        let path = schema_dir.join(format!("{name}.capnp"));
        if !path.exists() {
            continue;
        }

        let cgr_path = Path::new(&out_dir).join(format!("{name}.cgr"));
        let metadata_path = Path::new(&out_dir).join(format!("{name}_metadata.json"));

        // 1. Compile to Rust AND save raw CodeGeneratorRequest
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .import_path(&rpc_schema_dir)
            .file(&path)
            .raw_code_generator_request_path(&cgr_path)
            .run()
            .unwrap_or_else(|e| panic!("Failed to compile {name}.capnp: {e}"));

        // 2. Parse CGR and extract schema metadata with annotations
        if let Err(e) = hyprstream_rpc_build::parse_schema_and_extract_annotations(&cgr_path, &metadata_path, name) {
            println!("cargo:warning=Failed to parse schema for {name}: {e}");
            println!("cargo:warning=Falling back to text parsing (annotations not available)");
        }
    }
}
