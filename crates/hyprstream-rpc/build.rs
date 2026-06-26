//! Build script for hyprstream-rpc
//!
//! Compiles Cap'n Proto schemas for RPC envelope types.

#![allow(clippy::expect_used)] // build scripts use expect for required env vars

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");

    let schema_dir = Path::new("schema");
    if !schema_dir.exists() {
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");

    hyprstream_rpc_build::compile_schemas(
        schema_dir,
        Path::new(&out_dir),
        &[],
        &["common", "streaming", "events", "annotations", "optional", "nine"],
    );
}
