fn main() {
    // Link to libgit2 vendored by libgit2-sys
    // This ensures our extern "C" declarations can find the symbols
    println!("cargo:rustc-link-lib=git2");
}
