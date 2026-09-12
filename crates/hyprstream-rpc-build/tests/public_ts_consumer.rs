//! Consumer-facing TypeScript validation for the canonical RPC catalog.
//!
//! This test intentionally goes through the same two public build steps a
//! third-party package would use: compile the schemas to CGRs, invoke the
//! `hyprstream-ts-codegen` binary, and type-check the resulting sources with
//! TypeScript.  It therefore catches a schema that is present in Rust but not
//! consumable by the published TypeScript surface.
//!
//! Node/TypeScript are optional for the normal Rust test suite.  The
//! TypeScript codegen CI job (`.github/workflows/ts-codegen-roundtrip.yml`)
//! runs this test with `HYPRSTREAM_TS_CONSUMER_REQUIRE=1` so a missing
//! toolchain fails closed there instead of silently skipping the check.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stderr)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use hyprstream_rpc_build::compile_schemas;

const PUBLIC_SCHEMAS: &[&str] = &[
    "inference",
    "model",
    "registry",
    "policy",
    "mcp",
    "metrics",
    "oauth",
    "worker",
    "workflow",
    "discovery",
    "tui",
    "compositor_ipc",
];

fn required() -> bool {
    std::env::var("HYPRSTREAM_TS_CONSUMER_REQUIRE")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn which(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn tsc_command() -> Option<Vec<String>> {
    if which("tsc").is_some() {
        return Some(vec!["tsc".to_owned()]);
    }
    if required() && which("npx").is_some() {
        // Pin the same major as the CI job (`npm install -g typescript@5`):
        // TypeScript 7 removed `moduleResolution=node`, so an unpinned fallback
        // would check against a different compiler contract than the gate.
        return Some(vec![
            "npx".to_owned(),
            "--yes".to_owned(),
            "-p".to_owned(),
            "typescript@5".to_owned(),
            "tsc".to_owned(),
        ]);
    }
    None
}

fn codegen_binary() -> PathBuf {
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_hyprstream-ts-codegen") {
        return PathBuf::from(path);
    }
    let current = std::env::current_exe().expect("integration test executable path");
    current
        .parent()
        .and_then(Path::parent)
        .expect("test executable has target profile parent")
        .join("hyprstream-ts-codegen")
}

#[test]
fn canonical_rpc_schemas_are_consumable_by_typescript() {
    let require = required();
    if which("node").is_none() {
        let message = "skipping canonical TypeScript consumer check: `node` not on PATH";
        if require {
            panic!("{message} (HYPRSTREAM_TS_CONSUMER_REQUIRE=1)");
        }
        eprintln!("{message} (set HYPRSTREAM_TS_CONSUMER_REQUIRE=1 to fail-closed)");
        return;
    }
    let Some(tsc) = tsc_command() else {
        let message = "skipping canonical TypeScript consumer check: `tsc` not found";
        if require {
            panic!("{message} (install TypeScript or make `npx` available)");
        }
        eprintln!("{message} (set HYPRSTREAM_TS_CONSUMER_REQUIRE=1 to fail-closed)");
        return;
    };

    let root = std::env::temp_dir().join(format!(
        "hyprstream_rpc_public_ts_consumer_{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root).expect("remove stale consumer test directory");
    }
    let cgr_dir = root.join("cgr");
    let output_dir = root.join("generated");
    fs::create_dir_all(&cgr_dir).expect("create CGR directory");

    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let schema_dir = manifest.join("../hyprstream-rpc-std/schema");
    let rpc_schema_dir = manifest.join("../hyprstream-rpc/schema");
    compile_schemas(
        &schema_dir,
        &cgr_dir,
        &[rpc_schema_dir.as_path(), schema_dir.as_path()],
        PUBLIC_SCHEMAS,
    );

    let status = Command::new(codegen_binary())
        .args(["--input-dir", cgr_dir.to_str().expect("CGR path is UTF-8")])
        .args([
            "--output-dir",
            output_dir.to_str().expect("output path is UTF-8"),
        ])
        .status()
        .expect("run hyprstream-ts-codegen");
    assert!(
        status.success(),
        "canonical schema TypeScript generation failed"
    );

    assert!(output_dir.join("capnp.ts").is_file());
    assert!(output_dir.join("index.ts").is_file());
    for schema in PUBLIC_SCHEMAS {
        assert!(
            output_dir.join(format!("{schema}.ts")).is_file(),
            "missing generated TypeScript client for {schema}"
        );
    }

    let mut sources: Vec<PathBuf> = fs::read_dir(&output_dir)
        .expect("read generated TypeScript directory")
        .map(|entry| entry.expect("read generated entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ts"))
        .collect();
    sources.sort();
    let mut command = Command::new(&tsc[0]);
    command
        .args(&tsc[1..])
        .args([
            "--noEmit",
            "--module",
            "commonjs",
            "--moduleResolution",
            "node",
            "--target",
            "es2020",
            // The generated streaming-client interfaces intentionally expose
            // `[Symbol.dispose]()` (explicit resource management) alongside
            // `free()`; its typings live in `esnext.disposable`. The es2020
            // target is unchanged — the symbol is a computed property key,
            // supported at runtime since Node 18.18 and all current browsers.
            "--lib",
            "es2020,dom,esnext.disposable",
            "--skipLibCheck",
        ])
        .args(&sources)
        .current_dir(&output_dir);
    let status = command.status().expect("run TypeScript compiler");
    assert!(
        status.success(),
        "generated canonical TypeScript sources do not type-check"
    );

    fs::remove_dir_all(root).expect("remove consumer test directory");
}
