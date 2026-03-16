fn main() {
    // Compile compositor_ipc.capnp for WASI IPC protocol
    compile_capnp();

    // Emit compile-time libtorch metadata for the wizard's environment detection.
    // These are read by `backend.rs` via `env!()`.

    let version = option_env!("LIBTORCH_VERSION").unwrap_or("2.10.0");
    println!("cargo:rustc-env=LIBTORCH_VERSION={version}");

    // ABI: pre-cxx11 (default for PyTorch pip wheels) or cxx11
    let abi = option_env!("LIBTORCH_ABI").unwrap_or("pre-cxx11");
    println!("cargo:rustc-env=LIBTORCH_ABI={abi}");

    // Variant: cpu, cuda128, cuda130, rocm71
    // Auto-detect from LIBTORCH path if not explicitly set.
    let variant = if let Some(v) = option_env!("LIBTORCH_VARIANT") {
        v.to_string()
    } else if let Some(path) = option_env!("LIBTORCH") {
        detect_variant_from_path(path)
    } else {
        "cpu".to_string()
    };
    println!("cargo:rustc-env=LIBTORCH_VARIANT={variant}");

    // Re-run if these env vars change
    println!("cargo:rerun-if-env-changed=LIBTORCH_VERSION");
    println!("cargo:rerun-if-env-changed=LIBTORCH_ABI");
    println!("cargo:rerun-if-env-changed=LIBTORCH_VARIANT");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
}

fn compile_capnp() {
    println!("cargo:rerun-if-changed=../hyprstream/schema/compositor_ipc.capnp");

    let schema_dir = std::path::Path::new("../hyprstream/schema");
    let capnp_file = schema_dir.join("compositor_ipc.capnp");

    if !capnp_file.exists() {
        return;
    }

    capnpc::CompilerCommand::new()
        .src_prefix(schema_dir)
        .file(&capnp_file)
        .run()
        .expect("Failed to compile compositor_ipc.capnp");
}

fn detect_variant_from_path(path: &str) -> String {
    let lower = path.to_lowercase();
    if lower.contains("cuda") {
        // Try to extract version: cuda13.0 → cuda130, cuda12.8 → cuda128
        if lower.contains("13.0") || lower.contains("130") {
            "cuda130".to_string()
        } else if lower.contains("12.8") || lower.contains("128") {
            "cuda128".to_string()
        } else {
            "cuda130".to_string() // default CUDA
        }
    } else if lower.contains("rocm") {
        "rocm71".to_string()
    } else {
        "cpu".to_string()
    }
}
