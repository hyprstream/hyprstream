//! Build script for OpenVDB sparse storage integration

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/storage/vdb/openvdb_bridge.cpp");
    println!("cargo:rerun-if-changed=src/storage/vdb/openvdb_bindings.rs");
    
    build_openvdb_bridge();
}

fn build_openvdb_bridge() {
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
    let openvdb_result = pkg_config::Config::new()
        .atleast_version("7.0")
        .probe("openvdb");
        
    let (include_paths, link_paths, libs) = match openvdb_result {
        Ok(openvdb) => (openvdb.include_paths, openvdb.link_paths, openvdb.libs),
        Err(_) => {
            println!("cargo:warning=pkg-config failed, trying system OpenVDB installation...");
            
            let include_path = PathBuf::from("/usr/include");
            let lib_path = PathBuf::from("/usr/lib/x86_64-linux-gnu");
            let cmake_path = lib_path.join("cmake/OpenVDB");
            
            if !include_path.join("openvdb/openvdb.h").exists() {
                return Err("OpenVDB headers not found at /usr/include/openvdb".into());
            }
            
            if !lib_path.join("libopenvdb.so").exists() {
                return Err("OpenVDB library not found at /usr/lib/x86_64-linux-gnu/libopenvdb.so".into());
            }
            
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
    
    let mut build = cxx_build::bridge("src/storage/vdb/openvdb_bindings.rs");
    
    let current_dir = std::env::current_dir().unwrap();
    let header_path = current_dir.join("src/storage/vdb");
    
    build
        .file("src/storage/vdb/openvdb_bridge.cpp")
        .std("c++17")
        .include(header_path)
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations");
    
    for include_path in &include_paths {
        build.include(include_path);
    }
    
    let boost_paths = [
        "/usr/include/boost",
        "/usr/local/include/boost", 
        "/opt/homebrew/include/boost",
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
    
    println!("cargo:rustc-link-lib=tbb");
    println!("cargo:rustc-link-lib=Imath");
    println!("cargo:rustc-link-lib=blosc");
    
    build.compile("openvdb_bridge");
    
    Ok(())
}
