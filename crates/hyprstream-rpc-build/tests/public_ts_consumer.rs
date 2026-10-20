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

// Reuse the EXACT production emitter source, not a mirrored implementation.
// The CLI's `mod ts_codegen;` and this test path compile the same module files.
// Integration tests cannot access a binary-private module through the library.
#[path = "../src/ts_codegen/mod.rs"]
mod canonical_ts_codegen;

fn canonical_data_companion(
    cgr: &Path,
    original_name: &str,
) -> hyprstream_rpc_build::schema::types::ParsedSchema {
    // Parse under the ORIGINAL filename: the CGR reader uses that prefix to
    // select canonical local nodes. Renaming the CGR before parsing is invalid.
    let mut schema =
        hyprstream_rpc_build::schema::cgr_reader::parse_from_cgr_path(cgr, original_name)
            .expect("parse actual canonical service metadata");
    let request = schema
        .request_struct
        .take()
        .expect("canonical request root");
    let response = schema
        .response_struct
        .take()
        .expect("canonical response root");
    schema.structs.push(request);
    schema.structs.push(response);
    let before = serde_json::to_value(&schema.structs).expect("snapshot every canonical StructDef");
    let enums_before =
        serde_json::to_value(&schema.enums).expect("snapshot canonical enum metadata");
    // Only service classification changes; every real root/union/field offset,
    // type, annotation and imported StructDef remains byte-equivalent metadata.
    schema.request_variants.clear();
    schema.response_variants.clear();
    schema.scoped_clients.clear();
    assert_eq!(
        before,
        serde_json::to_value(&schema.structs).expect("projected StructDefs")
    );
    assert_eq!(
        enums_before,
        serde_json::to_value(&schema.enums).expect("projected enum metadata")
    );
    schema
}

const PUBLIC_SCHEMAS: &[&str] = &[
    "inference",
    "model",
    "registry",
    "policy",
    "mcp",
    "metrics",
    "service_events",
    "chat_core",
    "oauth",
    "decision",
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

    // Compile additional real data-only CGRs through the same production CLI,
    // so recursive behavior is tested independently of Clef's Entry names.
    let fixture_schema_dir = root.join("fixture-schemas");
    fs::create_dir_all(&fixture_schema_dir).expect("create recursive schema fixture directory");
    fs::write(
        fixture_schema_dir.join("recursive_codegen_fixture.capnp"),
        include_str!("recursive_codegen_fixture.capnp"),
    )
    .expect("write actual recursive schema fixture");
    compile_schemas(
        &fixture_schema_dir,
        &cgr_dir,
        &[fixture_schema_dir.as_path()],
        &["recursive_codegen_fixture"],
    );

    let canonical_inputs: Vec<_> = ["model", "registry"]
        .into_iter()
        .map(|name| {
            (
                name,
                fs::read(cgr_dir.join(format!("{name}.cgr")))
                    .expect("snapshot original canonical CGR bytes"),
            )
        })
        .collect();

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

    // A fresh process has a fresh randomized HashSet seed. Regenerate the same
    // immutable CGRs and compare every emitted file before fixture augmentation.
    let repeated_output = root.join("generated-repeat");
    let status = Command::new(codegen_binary())
        .arg("--input-dir")
        .arg(&cgr_dir)
        .arg("--output-dir")
        .arg(&repeated_output)
        .status()
        .expect("repeat generator in an independent process");
    assert!(status.success(), "repeated canonical generation failed");
    let emitted = |directory: &Path| {
        let mut files = std::collections::BTreeMap::new();
        for entry in fs::read_dir(directory).expect("list generated SDK") {
            let entry = entry.expect("generated SDK entry");
            assert!(entry
                .file_type()
                .expect("generated SDK file type")
                .is_file());
            files.insert(
                entry.file_name(),
                fs::read(entry.path()).expect("SDK file bytes"),
            );
        }
        files
    };
    assert_eq!(
        emitted(&output_dir),
        emitted(&repeated_output),
        "separate generator processes must emit byte-identical SDKs"
    );

    // Only the rebuilt production binary emits actual ModelClient/RegistryClient
    // and their shared runtime. Test-only response builders reuse those exact
    // emitter sources after canonical extraction, with service classification
    // cleared. No handwritten client, wire encoder, cast or production export
    // change exists solely for this fixture.
    let mut companions = Vec::new();
    for (name, bytes) in &canonical_inputs {
        let path = cgr_dir.join(format!("{name}.cgr"));
        assert_eq!(
            *bytes,
            fs::read(&path).expect("canonical bytes after CLI generation")
        );
        let companion = canonical_data_companion(&path, name);
        companions.push((format!("{name}_wire_fixture"), companion));
        assert_eq!(
            *bytes,
            fs::read(&path).expect("canonical bytes after data projection")
        );
    }
    // Exercise the canonical public emitter entry point in an isolated directory.
    // Copy ONLY the two companion schema modules: runtime/index/actual clients
    // in output_dir remain exclusively the rebuilt production CLI's output.
    let runtime_before =
        fs::read(output_dir.join("capnp.ts")).expect("snapshot actual CLI runtime");
    let index_before = fs::read(output_dir.join("index.ts")).expect("snapshot actual CLI index");
    let companion_dir = root.join("data-companions");
    fs::create_dir_all(&companion_dir).expect("create isolated companion output");
    canonical_ts_codegen::generate_all(
        &canonical_ts_codegen::TypeScriptBackend,
        &companions,
        &companion_dir,
    );
    for (name, _) in &companions {
        fs::copy(
            companion_dir.join(format!("{name}.ts")),
            output_dir.join(format!("{name}.ts")),
        )
        .expect("copy only canonical data-only response companion");
    }
    assert_eq!(
        runtime_before,
        fs::read(output_dir.join("capnp.ts")).expect("actual CLI runtime unchanged")
    );
    assert_eq!(
        index_before,
        fs::read(output_dir.join("index.ts")).expect("actual CLI index unchanged")
    );
    assert!(output_dir.join("capnp.ts").is_file());
    assert!(output_dir.join("index.ts").is_file());
    for schema in PUBLIC_SCHEMAS {
        assert!(
            output_dir.join(format!("{schema}.ts")).is_file(),
            "missing generated TypeScript client for {schema}"
        );
    }

    // Exercise the actual canonical public builders/parsers, not a mirrored
    // option implementation. Absent and present-empty must remain distinct.
    let consumer = output_dir.join("optional_text_list_consumer.ts");
    fs::write(
        &consumer,
        r#"
import { buildQuestionAnswer, parseQuestionAnswer, QuestionAnswer } from './inference';

for (const conformalSet of [undefined, [], ['alpha', 'λ', 'quote"slash\\']] as const) {
  const input: QuestionAnswer = {
    questionId: 'presence',
    value: { variant: 'noul', data: 0.5 },
    conformalSet: conformalSet === undefined ? undefined : [...conformalSet],
  };
  const decoded = parseQuestionAnswer(buildQuestionAnswer(input));
  if (JSON.stringify(decoded.conformalSet) !== JSON.stringify(input.conformalSet)) {
    throw new Error('optional text list lost presence or payload');
  }
  const again = parseQuestionAnswer(buildQuestionAnswer(decoded));
  if (JSON.stringify(again.conformalSet) !== JSON.stringify(input.conformalSet)) {
    throw new Error('optional text list changed on consumer read-modify-write');
  }
}
"#,
    )
    .expect("write canonical optional text-list consumer");

    fs::write(
        output_dir.join("public_decision_consumer.ts"),
        include_str!("public_decision_consumer.ts"),
    )
    .expect("write real generated decision/error public consumer");

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
            "--outDir",
            "compiled",
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
            // Match the website's strict nullable reads and isolated barrel
            // handling; compilation success without these flags missed both.
            "--strict",
            "--isolatedModules",
        ])
        .args(&sources)
        .current_dir(&output_dir);
    let status = command.status().expect("run TypeScript compiler");
    assert!(
        status.success(),
        "generated canonical TypeScript sources do not type-check"
    );

    let status = Command::new("node")
        .arg(output_dir.join("compiled/optional_text_list_consumer.js"))
        .status()
        .expect("run canonical optional text-list consumer");
    assert!(
        status.success(),
        "canonical optional text-list consumer failed"
    );

    let status = Command::new("node")
        .arg(output_dir.join("compiled/public_decision_consumer.js"))
        .status()
        .expect("run actual generated decision/error consumer");
    assert!(
        status.success(),
        "actual generated decision/error public consumer failed"
    );

    fs::remove_dir_all(root).expect("remove consumer test directory");
}
