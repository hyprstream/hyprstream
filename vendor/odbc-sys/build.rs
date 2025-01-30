use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_STATIC").is_ok() {
        if cfg!(target_os = "windows") {
            panic!("odbc-sys does not currently support static linking on windows");
        }
        let static_path =
            std::env::var("ODBC_SYS_STATIC_PATH").unwrap_or_else(|_| "/usr/lib".to_string());
        println!("cargo:rerun-if-env-changed=ODBC_SYS_STATIC_PATH");
        println!("cargo:rustc-link-search=native={static_path}");
        println!("cargo:rustc-link-lib=static=odbc");
        println!("cargo:rustc-link-lib=static=ltdl");
        if cfg!(target_os = "macos") {
            // Homebrew's unixodbc uses the system iconv, so we can't do a fully static linking
            // but this way we at least have only dependencies on built-in libraries
            // See also https://github.com/Homebrew/homebrew-core/pull/46145
            println!("cargo:rustc-link-lib=dylib=iconv");
        }
    }

    if cfg!(target_os = "macos") {
        if let Some(homebrew_lib_path) = homebrew_library_path() {
            print_paths(&homebrew_lib_path);
        }

        // if we're on Mac OS X we'll kindly add DYLD_LIBRARY_PATH to rustc's
        // linker search path
        if let Some(dyld_paths) = option_env!("DYLD_LIBRARY_PATH") {
            print_paths(dyld_paths);
        }
        // if we're on Mac OS X we'll kindly add DYLD_FALLBACK_LIBRARY_PATH to rustc's
        // linker search path
        if let Some(dyld_fallback_paths) = option_env!("DYLD_FALLBACK_LIBRARY_PATH") {
            print_paths(dyld_fallback_paths);
        }
    }
}

fn print_paths(paths: &str) {
    for path in paths.split(':').filter(|x| !x.is_empty()) {
        println!("cargo:rustc-link-search=native={path}")
    }
}

fn homebrew_library_path() -> Option<String> {
    let output = Command::new("brew").arg("--prefix").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix =
        String::from_utf8(output.stdout).expect("brew --prefix must yield utf8 encoded response");
    // brew returns also a linebreak (`\n`), we want to get rid of that.
    let prefix = prefix.trim();
    let lib_path = prefix.to_owned() + "/lib";
    Some(lib_path)
}
