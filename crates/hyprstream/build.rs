//! Build script for Hyprstream

use std::env;
use std::path::Path;

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");

    // Compile Cap'n Proto schemas
    compile_capnp_schemas();

    // If using Python PyTorch or download-libtorch, tch-rs handles libtorch setup
    if env::var("LIBTORCH_USE_PYTORCH").is_ok() || env::var("LIBTORCH").is_err() {
        // tch-rs will handle libtorch setup
        return;
    }

    let libtorch_path = env::var("LIBTORCH").unwrap();

    // Validate libtorch exists
    let libtorch_dir = Path::new(&libtorch_path);
    if !libtorch_dir.exists() {
        panic!("libtorch directory not found at {}", libtorch_path);
    }

    // Configure linking
    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);
    println!("cargo:rustc-env=LIBTORCH_STATIC=0");
    println!("cargo:rustc-env=LIBTORCH_BYPASS_VERSION_CHECK=1");
}

fn compile_capnp_schemas() {
    let schema_dir = Path::new("schema");

    // Skip if schema directory doesn't exist
    if !schema_dir.exists() {
        return;
    }

    // Compile events schema
    let events_schema = schema_dir.join("events.capnp");
    if events_schema.exists() {
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&events_schema)
            .run()
            .expect("failed to compile events.capnp");
    }

    // Compile inference schema
    let inference_schema = schema_dir.join("inference.capnp");
    if inference_schema.exists() {
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&inference_schema)
            .run()
            .expect("failed to compile inference.capnp");
    }

    // Compile registry schema
    let registry_schema = schema_dir.join("registry.capnp");
    if registry_schema.exists() {
        capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&registry_schema)
            .run()
            .expect("failed to compile registry.capnp");
    }
}
