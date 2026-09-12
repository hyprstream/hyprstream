//! Build script for the worker implementation crate.
//!
//! Worker/workflow schemas are canonical in `hyprstream-rpc-std`; this crate
//! intentionally performs no local Cap'n Proto compilation. The build script
//! stages the dependency's exported CGR files into this crate's `OUT_DIR` so
//! proc-macro expansion can consume them (Cargo exposes dependency metadata to
//! build scripts, not directly to proc-macros).

use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEP_HYPRSTREAM_RPC_STD_OUT_DIR");

    let dep_out = match env::var("DEP_HYPRSTREAM_RPC_STD_OUT_DIR") {
        Ok(value) => value,
        Err(_) => panic!("hyprstream-rpc-std must export its CGR OUT_DIR"),
    };
    let out_dir = match env::var("OUT_DIR") {
        Ok(value) => value,
        Err(_) => panic!("OUT_DIR not set"),
    };
    for name in ["worker", "workflow"] {
        let source = Path::new(&dep_out).join(format!("{name}.cgr"));
        let target = Path::new(&out_dir).join(format!("{name}.cgr"));
        fs::copy(&source, &target).unwrap_or_else(|e| {
            panic!("failed to stage {name}.cgr from {}: {e}", source.display())
        });
    }
}
