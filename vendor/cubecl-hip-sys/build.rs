use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUBECL_ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    let env_vars: Vec<_> = ["CUBECL_ROCM_PATH", "ROCM_PATH", "HIP_PATH"]
        .into_iter()
        .filter_map(|var| env::var(var).ok())
        .collect();

    let rocm_path = env_vars.iter().find(|path| {
        let hip_path = Path::new(path).join("include/hip");
        hip_path.exists()
    });

    if let Some(valid_rocm_path) = rocm_path {
        println!("cargo:rustc-link-lib=dylib=hiprtc");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-search=native={}/lib", valid_rocm_path);
    } else if !env_vars.is_empty() {
        panic!("HIP headers not found in any of the defined CUBECL_ROCM_PATH, ROCM_PATH or HIP_PATH directories.");
    }
}
