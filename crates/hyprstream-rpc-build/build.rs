#![allow(clippy::expect_used, clippy::unwrap_used)] // build script: required env vars / capnp invocation

// hyprstream-rpc-build has no generated types of its own for its runtime.
// Schema compilation and annotation extraction happen in the consuming crates'
// build scripts via `hyprstream_rpc_build::compile_schemas`.
//
// EXCEPTION: the `wire_roundtrip` integration test (#807) cross-checks the
// generated `capnp.ts` against the Rust `capnp` crate (the reference wire impl).
// That needs real capnp-rust readers/builders for the fixture schema, compiled
// here into `$OUT_DIR/wire_roundtrip_fixture_capnp.rs` for the test to
// `include!`. capnp is already a workspace-wide build dependency (every crate
// with a `.capnp` schema shells out to it in build.rs), so this adds no new
// toolchain requirement — if `capnp` is absent, every schema-bearing crate's
// build already fails here the same way.
fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let schema = format!("{manifest}/tests/wire_roundtrip_fixture.capnp");

    capnpc::CompilerCommand::new()
        .file(&schema)
        .run()
        .expect("compile tests/wire_roundtrip_fixture.capnp (capnp-rust codegen for #807)");

    println!("cargo::rerun-if-changed={schema}");
}
