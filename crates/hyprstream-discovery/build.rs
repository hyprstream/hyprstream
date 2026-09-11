//! Build script for the discovery implementation crate.
//!
//! The canonical schema/CGR is owned by `hyprstream-rpc-std`.  Stage its CGR
//! into this crate's output directory for proc-macro expansion; no local
//! Cap'n Proto schema is compiled here.

use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_HYPRSTREAM_RPC_STD_OUT_DIR");
    let dep_out = env::var("DEP_HYPRSTREAM_RPC_STD_OUT_DIR")
        .expect("hyprstream-rpc-std must export its CGR OUT_DIR");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let source = Path::new(&dep_out).join("discovery.cgr");
    let target = Path::new(&out_dir).join("discovery.cgr");
    fs::copy(&source, &target).unwrap_or_else(|e| {
        panic!("failed to stage discovery.cgr from {}: {e}", source.display())
    });
}
