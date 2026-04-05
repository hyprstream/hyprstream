//! Build script for hyprstream-rpc-std
//!
//! Compiles service Cap'n Proto schemas. Imports shared types from
//! hyprstream-rpc/schema/ (common, streaming, annotations, optional).

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");
    println!("cargo:rerun-if-changed=../hyprstream-rpc/schema/");

    let schema_dir = Path::new("schema");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");
    let out_path = Path::new(&out_dir);

    if !schema_dir.exists() {
        return;
    }

    // Import paths: our schema dir + hyprstream-rpc's schema dir (for common, streaming, etc.)
    let import_paths: &[&Path] = &[&rpc_schema_dir, schema_dir];

    let schemas = [
        "inference",
        "model",
        "registry",
        "policy",
        "mcp",
        "metrics",
        "notification",
        "service_events",
        "chat_core",
    ];

    hyprstream_rpc_build::compile_schemas(schema_dir, out_path, import_paths, &schemas);

    // Export OUT_DIR so downstream crates (hyprstream) can find our CGR files
    // via the DEP_HYPRSTREAM_RPC_STD_OUT_DIR env var.
    println!("cargo:out_dir={}", out_dir);
}
