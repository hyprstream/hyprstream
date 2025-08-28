//! Build script for OpenVDB integration with CUDA support

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
    
    // llama.cpp related build configuration
    configure_llama_cpp();

    // Check if CUDA is available
    let _cuda_available = check_cuda_availability();
    
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

// Removed CUDA/NanoVDB build functions - OpenVDB only

fn build_cpu_only() {
    // VDB is always required - try to build with OpenVDB
    match build_openvdb() {
        Ok(_) => {
            println!("cargo:warning=✅ Built with OpenVDB support - VDB features enabled");
        }
        Err(e) => {
            eprintln!("Error: OpenVDB is required for Hyprstream but was not found.");
            eprintln!("Details: {}", e);
            eprintln!("");
            eprintln!("To fix this issue:");
            eprintln!("1. Install OpenVDB development packages:");
            eprintln!("   - Ubuntu/Debian: sudo apt install libopenvdb-dev");
            eprintln!("   - macOS: brew install openvdb");
            eprintln!("   - See OPENVDB_SETUP.md for detailed instructions");
            eprintln!("");
            eprintln!("2. Ensure pkg-config can find OpenVDB:");
            eprintln!("   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH");
            eprintln!("");
            std::process::exit(1);
        }
    }
}

fn build_openvdb() -> Result<(), Box<dyn std::error::Error>> {
    // Try pkg-config first (for custom builds)
    let openvdb_result = pkg_config::Config::new()
        .atleast_version("7.0")
        .probe("openvdb");
        
    let (include_paths, link_paths, libs) = match openvdb_result {
        Ok(openvdb) => {
            // Found via pkg-config
            (openvdb.include_paths, openvdb.link_paths, openvdb.libs)
        }
        Err(_) => {
            // Fallback: try to find system OpenVDB installation (Ubuntu/Debian packages)
            println!("cargo:warning=pkg-config failed, trying system OpenVDB installation...");
            
            let include_path = PathBuf::from("/usr/include");
            let lib_path = PathBuf::from("/usr/lib/x86_64-linux-gnu");
            let cmake_path = lib_path.join("cmake/OpenVDB");
            
            // Verify OpenVDB headers exist
            if !include_path.join("openvdb/openvdb.h").exists() {
                return Err("OpenVDB headers not found at /usr/include/openvdb".into());
            }
            
            // Verify OpenVDB library exists
            if !lib_path.join("libopenvdb.so").exists() {
                return Err("OpenVDB library not found at /usr/lib/x86_64-linux-gnu/libopenvdb.so".into());
            }
            
            // Verify CMake modules exist (indicates proper OpenVDB installation)
            if !cmake_path.join("FindOpenVDB.cmake").exists() {
                return Err("OpenVDB CMake modules not found - incomplete installation".into());
            }
            
            println!("cargo:warning=Found system OpenVDB installation");
            (
                vec![include_path],
                vec![lib_path],
                vec!["openvdb".to_string()]
            )
        }
    };
    
    // Build OpenVDB bridge
    let mut build = cxx_build::bridge("src/storage/vdb/openvdb_bindings.rs");
    
    let current_dir = std::env::current_dir().unwrap();
    let header_path = current_dir.join("src/storage/vdb");
    
    build
        .file("src/storage/vdb/openvdb_bridge.cpp")
        .std("c++17")
        .include(header_path)  // Include directory for our header
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations");
    
    // Add OpenVDB include paths and libraries
    for include_path in &include_paths {
        build.include(include_path);
    }
    
    // Add common boost include paths
    let boost_paths = [
        "/usr/include/boost",
        "/usr/local/include/boost", 
        "/opt/homebrew/include/boost", // macOS Homebrew
    ];
    
    for boost_path in &boost_paths {
        if std::path::Path::new(boost_path).exists() {
            build.include(std::path::Path::new(boost_path).parent().unwrap());
            break;
        }
    }
    
    for link_path in &link_paths {
        println!("cargo:rustc-link-search=native={}", link_path.display());
    }
    
    for lib in &libs {
        println!("cargo:rustc-link-lib={}", lib);
    }
    
    // Common dependencies for OpenVDB (Ubuntu package names)
    println!("cargo:rustc-link-lib=tbb");
    println!("cargo:rustc-link-lib=Imath");  // Half is part of Imath in newer versions
    println!("cargo:rustc-link-lib=blosc");
    
    build.compile("openvdb_bridge");
    
    Ok(())
}


// Removed NanoVDB binding generation - OpenVDB only

/// Configure llama.cpp build environment
fn configure_llama_cpp() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_METAL");
    println!("cargo:rerun-if-env-changed=LLAMA_NATIVE");
    
    // Check for CUDA availability for llama.cpp
    if let Ok(cuda_path) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        
        // Enable CUDA feature for llama-cpp-2
        println!("cargo:rustc-env=LLAMA_CUDA=1");
        println!("cargo:warning=Found CUDA at {}, enabling GPU acceleration", cuda_path);
    } else {
        // Check common CUDA installation paths
        let common_cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda", 
            "/usr/cuda",
        ];
        
        let mut cuda_found = false;
        for path in &common_cuda_paths {
            if PathBuf::from(path).join("lib64").exists() || PathBuf::from(path).join("lib").exists() {
                println!("cargo:rustc-link-search=native={}/lib64", path);
                println!("cargo:rustc-link-search=native={}/lib", path);
                println!("cargo:rustc-env=LLAMA_CUDA=1");
                cuda_found = true;
                println!("cargo:warning=Found CUDA at {}, enabling GPU acceleration", path);
                break;
            }
        }
        
        if !cuda_found {
            println!("cargo:warning=CUDA not found, using CPU-only mode");
        }
    }
    
    // Set additional linker paths for common library locations
    if cfg!(target_os = "linux") {
        // Common system library paths
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/lib64");
        
        // Conda/Mamba library paths (common in ML environments)
        if let Ok(conda_prefix) = env::var("CONDA_PREFIX") {
            println!("cargo:rustc-link-search=native={}/lib", conda_prefix);
        }
        
        if let Ok(mamba_prefix) = env::var("MAMBA_ROOT_PREFIX") {
            println!("cargo:rustc-link-search=native={}/lib", mamba_prefix);
        }
        
        // User-specified library paths from environment
        if let Ok(ld_library_path) = env::var("LD_LIBRARY_PATH") {
            for path in ld_library_path.split(':') {
                if !path.is_empty() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
        }
    }
    
    // macOS specific configurations
    if cfg!(target_os = "macos") {
        // Enable Metal acceleration on Apple Silicon
        if env::var("LLAMA_METAL").is_ok() || cfg!(target_arch = "aarch64") {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo:rustc-env=LLAMA_METAL=1");
            println!("cargo:warning=Enabling Metal acceleration for macOS");
        }
        
        // Homebrew paths
        if let Ok(homebrew_prefix) = env::var("HOMEBREW_PREFIX") {
            println!("cargo:rustc-link-search=native={}/lib", homebrew_prefix);
        }
        
        // Common macOS library paths
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
    }
    
    // Enable optimizations for release builds
    if env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-env=LLAMA_NATIVE=1"); // Enable native CPU optimizations
        println!("cargo:warning=Enabling native CPU optimizations for release build");
    }
    
    // OpenBLAS configuration (fallback for CPU acceleration)
    if env::var("LLAMA_BLAS").is_ok() || env::var("OPENBLAS_PATH").is_ok() {
        if let Ok(openblas_path) = env::var("OPENBLAS_PATH") {
            println!("cargo:rustc-link-search=native={}/lib", openblas_path);
        }
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-env=LLAMA_BLAS=1");
        println!("cargo:warning=Enabling OpenBLAS acceleration");
    }
}