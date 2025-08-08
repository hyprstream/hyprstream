//! Build script for NanoVDB integration with CUDA support

use std::env;
use std::path::PathBuf;

fn main() {
    // Only build NanoVDB if not in docs.rs environment
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/openvdb");
    println!("cargo:rerun-if-changed=src/storage/vdb/nanovdb_wrapper.cpp");

    // Check if CUDA is available
    let cuda_available = check_cuda_availability();
    
    // For now, always build CPU-only version to ensure compatibility
    // Full CUDA integration requires more complex toolchain setup
    println!("cargo:warning=Building CPU-only version (CUDA integration in progress)");
    build_cpu_only();
}

fn check_cuda_availability() -> bool {
    // Check for CUDA toolkit
    let nvcc_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    // Also check for CUDA runtime libraries
    let cuda_path_exists = std::env::var("CUDA_PATH").is_ok() || 
                          std::path::Path::new("/usr/local/cuda").exists() ||
                          std::path::Path::new("/opt/cuda").exists();
    
    nvcc_available && cuda_path_exists
}

fn build_with_cuda() {
    let nanovdb_root = PathBuf::from("vendor/openvdb/nanovdb/nanovdb");
    
    // Try to use nvcc if available, otherwise fall back to g++
    let use_nvcc = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(&nanovdb_root)
        .include(nanovdb_root.join("util"))
        .include(nanovdb_root.join("cuda"))
        .file("src/storage/vdb/nanovdb_wrapper.cpp")
        .define("NANOVDB_USE_CUDA", "1");
    
    if use_nvcc {
        // Use nvcc for CUDA compilation
        build
            .compiler("nvcc")
            .flag("-xcuda")
            .flag("--cuda-gpu-arch=sm_70")
            .flag("--cuda-gpu-arch=sm_80")
            .flag("--cuda-gpu-arch=sm_86")
            .define("__CUDACC__", "1");
    } else {
        // Use g++ without CUDA-specific flags
        println!("cargo:warning=Using g++ without CUDA compilation - GPU features disabled");
    }
    
    build.compile("nanovdb_wrapper");

    // Link CUDA runtime if available
    if use_nvcc {
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
        
        // Add CUDA paths
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            // Common CUDA installation paths
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        }
    }
    
    // Generate CUDA bindings
    generate_cuda_bindings();
}

fn build_cpu_only() {
    let nanovdb_parent = PathBuf::from("vendor/openvdb/nanovdb");
    let nanovdb_root = PathBuf::from("vendor/openvdb/nanovdb/nanovdb");
    
    // Build CPU-only version
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&nanovdb_parent)  // Include parent dir for <nanovdb/...> paths
        .include(&nanovdb_root)
        .include(nanovdb_root.join("util"))
        .include(nanovdb_root.join("math"))
        .include(nanovdb_root.join("io"))
        .include(nanovdb_root.join("tools"))
        .file("src/storage/vdb/nanovdb_wrapper_simple.cpp")
        .define("NANOVDB_USE_CUDA", "0")
        .compile("nanovdb_wrapper");
        
    generate_cpu_bindings();
}

fn generate_cuda_bindings() {
    let bindings = bindgen::Builder::default()
        .header("src/storage/vdb/nanovdb_wrapper.h")
        .clang_arg("-Ivendor/openvdb/nanovdb/nanovdb")
        .clang_arg("-Ivendor/openvdb/nanovdb/nanovdb/util") 
        .clang_arg("-Ivendor/openvdb/nanovdb/nanovdb/cuda")
        .clang_arg("-DNANOVDB_USE_CUDA=1")
        .clang_arg("-std=c++17")
        .allowlist_function("nanovdb_.*")
        .allowlist_type("NanoVDB.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("nanovdb_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn generate_cpu_bindings() {
    let bindings = bindgen::Builder::default()
        .header("src/storage/vdb/nanovdb_wrapper_simple.h")
        .allowlist_function("nanovdb_.*")
        .allowlist_type("NanoVDB.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("nanovdb_bindings.rs"))
        .expect("Couldn't write bindings!");
}