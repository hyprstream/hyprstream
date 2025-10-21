//! Build script for Hyprstream

use std::env;
use std::path::Path;

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");

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