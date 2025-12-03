//! Build script for bitsandbytes-sys
//!
//! This script handles:
//! 1. Detecting the bitsandbytes library location
//! 2. Determining the backend (CUDA, HIP, CPU)
//! 3. Generating or using pre-generated FFI bindings
//! 4. Emitting linker configuration

use std::env;
use std::path::PathBuf;

/// Supported backends for bitsandbytes
#[derive(Debug, Clone, Copy, PartialEq)]
enum Backend {
    Cuda,
    Hip,
    Cpu,
}

impl Backend {
    fn from_features() -> Self {
        if cfg!(feature = "cuda") {
            Backend::Cuda
        } else if cfg!(feature = "hip") {
            Backend::Hip
        } else if cfg!(feature = "cpu") {
            Backend::Cpu
        } else {
            // Auto-detect based on environment
            Self::auto_detect()
        }
    }

    fn auto_detect() -> Self {
        // Check for ROCm/HIP first
        if env::var("ROCM_PATH").is_ok() || env::var("HIP_PATH").is_ok() {
            return Backend::Hip;
        }

        // Check for CUDA
        if env::var("CUDA_HOME").is_ok() || env::var("CUDA_PATH").is_ok() {
            return Backend::Cuda;
        }

        // Check common paths
        if PathBuf::from("/opt/rocm").exists() {
            return Backend::Hip;
        }

        if PathBuf::from("/usr/local/cuda").exists() {
            return Backend::Cuda;
        }

        // Default to CPU
        Backend::Cpu
    }

    fn library_suffix(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Hip => "rocm",
            Backend::Cpu => "cpu",
        }
    }
}

/// Find the bitsandbytes library
fn find_bitsandbytes_library(backend: Backend) -> Option<(PathBuf, String)> {
    let suffix = backend.library_suffix();

    // Check environment variable first
    if let Ok(path) = env::var("BITSANDBYTES_LIB_PATH") {
        let lib_path = PathBuf::from(&path);
        if lib_path.exists() {
            // Extract library name from path
            if let Some(name) = lib_path.file_stem() {
                let name = name.to_string_lossy();
                // Remove 'lib' prefix if present
                let lib_name = name.strip_prefix("lib").unwrap_or(&name);
                return Some((lib_path.parent().unwrap().to_path_buf(), lib_name.to_string()));
            }
        }
    }

    // Common search paths
    let search_paths = [
        // Python site-packages (most common installation)
        get_python_site_packages(),
        // System library paths
        Some(PathBuf::from("/usr/local/lib")),
        Some(PathBuf::from("/usr/lib")),
        Some(PathBuf::from("/usr/lib64")),
        // Build directory (if building from source)
        env::var("BITSANDBYTES_BUILD_DIR").ok().map(PathBuf::from),
    ];

    // Library name patterns to search for
    let patterns = [
        format!("libbitsandbytes_{}.so", suffix),
        format!("libbitsandbytes_{}*.so", suffix),
        "libbitsandbytes.so".to_string(),
    ];

    for search_path in search_paths.into_iter().flatten() {
        for pattern in &patterns {
            // Simple glob-like matching
            if let Ok(entries) = std::fs::read_dir(&search_path) {
                for entry in entries.flatten() {
                    let file_name = entry.file_name();
                    let file_name = file_name.to_string_lossy();

                    if pattern.contains('*') {
                        // Simple wildcard matching
                        let prefix = pattern.split('*').next().unwrap();
                        let suffix_part = pattern.split('*').last().unwrap();
                        if file_name.starts_with(prefix) && file_name.ends_with(suffix_part) {
                            let lib_name = file_name
                                .strip_prefix("lib")
                                .unwrap_or(&file_name)
                                .strip_suffix(".so")
                                .unwrap_or(&file_name);
                            return Some((search_path.clone(), lib_name.to_string()));
                        }
                    } else if file_name == *pattern {
                        let lib_name = file_name
                            .strip_prefix("lib")
                            .unwrap_or(&file_name)
                            .strip_suffix(".so")
                            .unwrap_or(&file_name);
                        return Some((search_path.clone(), lib_name.to_string()));
                    }
                }
            }
        }
    }

    None
}

/// Get Python site-packages directory
fn get_python_site_packages() -> Option<PathBuf> {
    // Try to get from python
    let output = std::process::Command::new("python3")
        .args(["-c", "import site; print(site.getsitepackages()[0])"])
        .output()
        .ok()?;

    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout);
        let path = path.trim();
        let bnb_path = PathBuf::from(path).join("bitsandbytes");
        if bnb_path.exists() {
            return Some(bnb_path);
        }
    }

    None
}

/// Generate bindings using bindgen
#[cfg(feature = "bindgen")]
fn generate_bindings(out_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Only generate bindings for our wrapper functions
        .allowlist_function("bnb_.*")
        .allowlist_type("bnb_.*")
        .allowlist_var("BNB_.*")
        // Use core types
        .use_core()
        // Generate Debug trait
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    // Register custom cfg flags for conditional compilation
    println!("cargo::rustc-check-cfg=cfg(bnb_cuda)");
    println!("cargo::rustc-check-cfg=cfg(bnb_hip)");
    println!("cargo::rustc-check-cfg=cfg(bnb_cpu)");
    println!("cargo::rustc-check-cfg=cfg(bnb_stub)");

    println!("cargo:rerun-if-env-changed=BITSANDBYTES_LIB_PATH");
    println!("cargo:rerun-if-env-changed=BITSANDBYTES_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=wrapper.h");

    let backend = Backend::from_features();
    println!("cargo:warning=bitsandbytes-sys: Using {:?} backend", backend);

    // Emit backend configuration
    match backend {
        Backend::Cuda => println!("cargo:rustc-cfg=bnb_cuda"),
        Backend::Hip => println!("cargo:rustc-cfg=bnb_hip"),
        Backend::Cpu => println!("cargo:rustc-cfg=bnb_cpu"),
    }

    // Find the library
    if let Some((lib_dir, lib_name)) = find_bitsandbytes_library(backend) {
        println!("cargo:warning=bitsandbytes-sys: Found library {} in {:?}", lib_name, lib_dir);
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib={}", lib_name);

        // For HIP backend, also link ROCm libraries
        if backend == Backend::Hip {
            if let Ok(rocm_path) = env::var("ROCM_PATH") {
                println!("cargo:rustc-link-search=native={}/lib", rocm_path);
            } else if PathBuf::from("/opt/rocm/lib").exists() {
                println!("cargo:rustc-link-search=native=/opt/rocm/lib");
            }
            println!("cargo:rustc-link-lib=dylib=amdhip64");
        }

        // For CUDA backend, link CUDA runtime
        if backend == Backend::Cuda {
            if let Ok(cuda_path) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
                println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            } else if PathBuf::from("/usr/local/cuda/lib64").exists() {
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            }
            println!("cargo:rustc-link-lib=dylib=cudart");
        }
    } else {
        println!("cargo:warning=bitsandbytes-sys: Library not found, using stub implementation");
        println!("cargo:rustc-cfg=bnb_stub");
    }

    // Generate or copy bindings
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    #[cfg(feature = "bindgen")]
    generate_bindings(&out_path);

    #[cfg(not(feature = "bindgen"))]
    {
        // Copy pre-generated bindings
        let bindings_src = PathBuf::from("src/bindings.rs");
        if bindings_src.exists() {
            std::fs::copy(&bindings_src, out_path.join("bindings.rs"))
                .expect("Failed to copy pre-generated bindings");
        }
    }
}
