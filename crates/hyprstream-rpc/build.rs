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

    // Compile streaming schema (StreamBlock, StreamPayload, StreamRegister, etc.)
    let streaming_schema = schema_dir.join("streaming.capnp");
    if streaming_schema.exists() {
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&streaming_schema)
            .run()
            .expect("failed to compile streaming.capnp");
    }

    // Compile events schema (EventEnvelope, WorkerEvent)
    let events_schema = schema_dir.join("events.capnp");
    if events_schema.exists() {
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&events_schema)
            .run()
            .expect("failed to compile events.capnp");
    }
}
