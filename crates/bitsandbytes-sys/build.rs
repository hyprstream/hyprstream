//! Build script for bitsandbytes-sys
//!
//! This script handles:
//! 1. Detecting the bitsandbytes library location
//! 2. Determining the backend (CUDA, HIP, CPU)
//! 3. Generating or using pre-generated FFI bindings
//! 4. Emitting linker configuration
//! 5. Optionally cloning and building bitsandbytes from source

use std::env;
use std::path::PathBuf;

#[cfg(feature = "build-from-source")]
use std::path::Path;

#[cfg(feature = "build-from-source")]
const BITSANDBYTES_REPO: &str = "https://github.com/bitsandbytes-foundation/bitsandbytes";
#[cfg(feature = "build-from-source")]
const BITSANDBYTES_VERSION: &str = "0.45.0";

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

    #[cfg(feature = "build-from-source")]
    fn cmake_backend(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Hip => "hip",
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

// ============================================================================
// Build from source functionality
// ============================================================================

/// Clone bitsandbytes from GitHub if not present
#[cfg(feature = "build-from-source")]
fn ensure_bitsandbytes_source() -> Option<PathBuf> {
    // Priority 1: BITSANDBYTES_PATH env var
    if let Ok(path) = env::var("BITSANDBYTES_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            println!(
                "cargo:warning=bitsandbytes-sys: Using existing source at {:?}",
                path
            );
            return Some(path);
        }
    }

    // Priority 2: target-deps/bitsandbytes relative to project root
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir
        .ancestors()
        .find(|p| p.join("Cargo.toml").exists() && p.join("crates").exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest_dir.clone());

    let deps_dir = env::var("DEPS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| project_root.join("target-deps"));
    let clone_dir = deps_dir.join("bitsandbytes");

    if clone_dir.exists() {
        println!(
            "cargo:warning=bitsandbytes-sys: Using cached clone at {:?}",
            clone_dir
        );
        return Some(clone_dir);
    }

    // Priority 3: Clone from GitHub
    println!(
        "cargo:warning=bitsandbytes-sys: Cloning from {}",
        BITSANDBYTES_REPO
    );

    // Create deps directory
    if let Err(e) = std::fs::create_dir_all(&deps_dir) {
        println!("cargo:warning=bitsandbytes-sys: Failed to create deps dir: {}", e);
        return None;
    }

    // Clone using git2
    let repo = match git2::Repository::clone(BITSANDBYTES_REPO, &clone_dir) {
        Ok(repo) => repo,
        Err(e) => {
            println!("cargo:warning=bitsandbytes-sys: Clone failed: {}", e);
            return None;
        }
    };

    // Checkout specific version tag
    if let Err(e) = checkout_version(&repo, BITSANDBYTES_VERSION) {
        println!(
            "cargo:warning=bitsandbytes-sys: Failed to checkout {}: {}",
            BITSANDBYTES_VERSION, e
        );
        // Continue with default branch
    }

    Some(clone_dir)
}

#[cfg(feature = "build-from-source")]
fn checkout_version(repo: &git2::Repository, version: &str) -> Result<(), git2::Error> {
    let tag_name = format!("refs/tags/{}", version);
    let obj = repo.revparse_single(&tag_name)?;
    repo.checkout_tree(&obj, None)?;
    repo.set_head(&tag_name)?;
    Ok(())
}

/// Build bitsandbytes from source using cmake
#[cfg(feature = "build-from-source")]
fn build_bitsandbytes_from_source(backend: Backend) -> Option<(PathBuf, String)> {
    let source_path = ensure_bitsandbytes_source()?;

    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let rocm_arch = env::var("PYTORCH_ROCM_ARCH").unwrap_or_else(|_| "gfx90a".to_string());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("bitsandbytes-build");

    // Create build directory
    if let Err(e) = std::fs::create_dir_all(&build_dir) {
        println!("cargo:warning=bitsandbytes-sys: Failed to create build dir: {}", e);
        return None;
    }

    println!(
        "cargo:warning=bitsandbytes-sys: Building with ROCM_PATH={}, arch={}",
        rocm_path, rocm_arch
    );

    // Run cmake configure
    let status = std::process::Command::new("cmake")
        .current_dir(&build_dir)
        .env("ROCM_PATH", &rocm_path)
        .args([
            format!("-DCOMPUTE_BACKEND={}", backend.cmake_backend()),
            "-DCMAKE_BUILD_TYPE=Release".to_string(),
            format!("-DBNB_ROCM_ARCH={}", rocm_arch),
            source_path.to_string_lossy().to_string(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=bitsandbytes-sys: cmake configure failed with {}", s);
            return None;
        }
        Err(e) => {
            println!("cargo:warning=bitsandbytes-sys: cmake configure error: {}", e);
            return None;
        }
    }

    // Run cmake build
    let status = std::process::Command::new("cmake")
        .current_dir(&build_dir)
        .args(["--build", ".", "--config", "Release", "-j"])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=bitsandbytes-sys: cmake build failed with {}", s);
            return None;
        }
        Err(e) => {
            println!("cargo:warning=bitsandbytes-sys: cmake build error: {}", e);
            return None;
        }
    }

    // Find the built library (output goes to source_path/bitsandbytes/)
    let lib_dir = source_path.join("bitsandbytes");
    find_built_library(&lib_dir, backend).map(|name| (lib_dir, name))
}

#[cfg(feature = "build-from-source")]
fn find_built_library(lib_dir: &Path, backend: Backend) -> Option<String> {
    let suffix = backend.library_suffix();

    if let Ok(entries) = std::fs::read_dir(lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            // Look for libbitsandbytes_rocmXX.so or libbitsandbytes_cudaXX.so
            if name.starts_with(&format!("libbitsandbytes_{}", suffix)) && name.ends_with(".so") {
                let lib_name = name
                    .strip_prefix("lib")
                    .unwrap_or(&name)
                    .strip_suffix(".so")
                    .unwrap_or(&name)
                    .to_string();
                println!(
                    "cargo:warning=bitsandbytes-sys: Found built library: {}",
                    lib_name
                );
                return Some(lib_name);
            }
        }
    }

    println!("cargo:warning=bitsandbytes-sys: Built library not found in {:?}", lib_dir);
    None
}

/// Generate bindings using bindgen
#[cfg(feature = "bindgen")]
fn generate_bindings(out_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Only generate bindings for our wrapper functions
        // Functions use 'c' prefix (cquantize, cdequantize, cgemm, etc.)
        .allowlist_function("c.*")
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

    println!("cargo:rerun-if-env-changed=BITSANDBYTES_LIB_PATH");
    println!("cargo:rerun-if-env-changed=BITSANDBYTES_PATH");
    println!("cargo:rerun-if-env-changed=BITSANDBYTES_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=DEPS_DIR");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PYTORCH_ROCM_ARCH");
    println!("cargo:rerun-if-changed=wrapper.h");

    let backend = Backend::from_features();
    println!("cargo:warning=bitsandbytes-sys: Using {:?} backend", backend);

    // Emit backend configuration
    match backend {
        Backend::Cuda => println!("cargo:rustc-cfg=bnb_cuda"),
        Backend::Hip => println!("cargo:rustc-cfg=bnb_hip"),
        Backend::Cpu => println!("cargo:rustc-cfg=bnb_cpu"),
    }

    // Find the library with priority:
    // 1. BITSANDBYTES_LIB_PATH (explicit pre-built)
    // 2. build-from-source feature
    // 3. Search standard paths
    // 4. Stub fallback
    let lib_info = find_bitsandbytes_library(backend).or_else(|| {
        #[cfg(feature = "build-from-source")]
        {
            build_bitsandbytes_from_source(backend)
        }
        #[cfg(not(feature = "build-from-source"))]
        {
            None
        }
    });

    if let Some((lib_dir, lib_name)) = lib_info {
        println!(
            "cargo:warning=bitsandbytes-sys: Using library {} from {:?}",
            lib_name, lib_dir
        );
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
        panic!(
            "bitsandbytes-sys: Library not found!\n\
            \n\
            Set one of the following environment variables:\n\
            - BITSANDBYTES_LIB_PATH: Path to libbitsandbytes_*.so\n\
            - BITSANDBYTES_PATH: Path to bitsandbytes source (with build-from-source feature)\n\
            \n\
            Or install bitsandbytes via pip: pip install bitsandbytes"
        );
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
