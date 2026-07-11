//! Cross-implementation round-trip test: Rust `capnp` ⇄ generated `capnp.ts` (#807).
//!
//! The `ts_codegen` string-assertion tests can't see wire-arithmetic bugs — an
//! off-by-one in pointer offsets or a stale-buffer capture after arena growth
//! passes CI at 100% because the *generated source text* still looks right. This
//! test executes the generated TypeScript against the Rust `capnp` reference
//! implementation (the canonical wire encoder/decoder) in **both** directions:
//!
//! - **Rust→TS**: build a message with capnp-rust, serialize, parse the bytes
//!   with the *generated* TS parser, compare to capnp-rust's own read of the
//!   same bytes.
//! - **TS→Rust**: build a message with the *generated* TS builder (including
//!   lists large enough to force arena growth past the ~1 KB slack — the
//!   #725/#772 stale-buffer + post-realloc-offset bug class), parse with
//!   capnp-rust, compare to TS's own self-parse.
//!
//! The fixture schema (`wire_roundtrip_fixture.capnp`) covers every wire shape
//! the runtime supports: all primitive scalars + lists, `Text`/`Data` + their
//! lists, doubly-nested `List(List(Float32/Float64))`, and a union with a
//! Void/scalar/Text/list/group arm. Shapes the codegen deliberately rejects
//! (struct lists, nested `List(List(Text))`) are covered by #725's
//! hard-fail-at-generation invariant — this covers the *supported* half.
//!
//! Node + `tsc` drive the TS. When either is absent the test **skips with a
//! warning** unless `HYPRSTREAM_TS_ROUNDTRIP_REQUIRE=1` is set (the dedicated
//! CI workflow sets it so CI fails-closed; the merge-gate `cargo nextest run`
//! leaves it unset so a nodeless runner skips rather than failing).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::print_stderr
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

/// Deep equality that treats `Number(10)` and `Number(10.0)` as equal.
///
/// serde_json preserves the int-vs-float distinction in `Value::Number`, so a
/// plain `==` flags `10` (TS `JSON.stringify` of `10.0`) vs `10.0` (Rust
/// `json!`) as a mismatch even though they're the same wire value. Every numeric
/// field here is either a small integer (exact as f64) or an f32/f64 (exact
/// widening); 64-bit ints are carried as decimal strings, so `as_f64` comparison
/// is exact for everything that reaches this path.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| values_equal(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| values_equal(v, w)))
        }
        _ => a == b,
    }
}

/// capnp-rust readers/builders for the fixture schema, compiled by `build.rs`
/// into `$OUT_DIR/tests/wire_roundtrip_fixture_capnp.rs`. The file's internal
/// references assume it lives at `crate::wire_roundtrip_fixture_capnp`, so the
/// wrapping module keeps that path. The capnpc-generated source isn't style-
/// clean (trailing unit exprs, duplicated match arms), so the whole module is
/// lint-suppressed — it's machine output, never hand-edited.
mod wire_roundtrip_fixture_capnp {
    #![allow(clippy::all, clippy::pedantic, warnings)]
    include!(concat!(
        env!("OUT_DIR"),
        "/tests/wire_roundtrip_fixture_capnp.rs"
    ));
}

use wire_roundtrip_fixture_capnp::{wire_choice, wire_roundtrip};

/// Resolve the path to the `hyprstream-ts-codegen` binary of this package.
///
/// cargo injects `CARGO_BIN_EXE_<name>` (hyphens → underscores) as a *runtime*
/// env var when the package's integration tests run; if it's absent (e.g. the
/// test binary was copied out of its target dir), fall back to looking next to
/// the test executable at `<target>/<profile>/hyprstream-ts-codegen`.
fn codegen_bin_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_hyprstream_ts_codegen") {
        return PathBuf::from(p);
    }
    let exe = std::env::current_exe().expect("current_exe");
    let deps = exe.parent().expect("test exe has parent (deps dir)");
    let profile = deps.parent().expect("deps dir has parent (profile dir)");
    profile.join("hyprstream-ts-codegen")
}

/// True iff an executable `name` resolves on `$PATH`.
fn which(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Locate the TypeScript compiler.
///
/// Prefers a global `tsc` (the dedicated CI workflow installs it via
/// `npm i -g typescript`). Only when fail-closed is explicitly requested
/// (`HYPRSTREAM_TS_ROUNDTRIP_REQUIRE=1`) does it fall back to `npx -p typescript`
/// — that downloads typescript on demand, which would couple an unrelated
/// `cargo test` run (e.g. the merge gate, which has node but no global tsc) to
/// npm-registry availability. Without the flag, a missing global `tsc` simply
/// skips so the merge gate stays network-free.
fn find_tsc(require: bool) -> Option<Vec<String>> {
    if which("tsc").is_some() {
        return Some(vec!["tsc".to_owned()]);
    }
    if require && which("npx").is_some() {
        return Some(vec![
            "npx".to_owned(),
            "--yes".to_owned(),
            "-p".to_owned(),
            "typescript".to_owned(),
            "tsc".to_owned(),
        ]);
    }
    None
}

/// One-time scaffolding: a temp workdir holding the generated `capnp.ts` +
/// service `.ts` (via the real `hyprstream-ts-codegen` bin), the checked-in node
/// harness, and their `tsc`-compiled `.js` output.
struct Workdir {
    dir: PathBuf,
    node: PathBuf,
    /// True when node/tsc were unavailable (and fail-closed wasn't requested) —
    /// tests no-op instead of failing on a nodeless/network-free runner.
    skipped: bool,
}

impl Workdir {
    /// Sentinel for a skipped run — see [`Workdir::skipped`].
    fn skipped() -> Workdir {
        Workdir {
            dir: PathBuf::new(),
            node: PathBuf::new(),
            skipped: true,
        }
    }

    fn prepare() -> Workdir {
        let require = std::env::var("HYPRSTREAM_TS_ROUNDTRIP_REQUIRE")
            .map(|v| v == "1")
            .unwrap_or(false);

        let node = match which("node") {
            Some(n) => n,
            None => {
                let msg = "skipping ts round-trip: `node` not on PATH";
                if require {
                    panic!("{msg} (HYPRSTREAM_TS_ROUNDTRIP_REQUIRE=1 — install node to run)");
                }
                eprintln!("{msg} (set HYPRSTREAM_TS_ROUNDTRIP_REQUIRE=1 to fail-closed)");
                return Workdir::skipped();
            }
        };
        let tsc = match find_tsc(require) {
            Some(t) => t,
            None => {
                let msg = "skipping ts round-trip: `tsc` (typescript) not found";
                if require {
                    panic!(
                        "{msg} (HYPRSTREAM_TS_ROUNDTRIP_REQUIRE=1 — `npm i -g typescript` to run)"
                    );
                }
                eprintln!("{msg} (set HYPRSTREAM_TS_ROUNDTRIP_REQUIRE=1 to fail-closed)");
                return Workdir::skipped();
            }
        };

        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        // Unique per `prepare()` call — tests run in parallel within one process
        // (same pid), so a pid-only name would collide. The counter guarantees
        // each test gets its own workdir.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let workdir = std::env::temp_dir().join(format!(
            "hyprstream_wire_roundtrip_{}_{seq}",
            std::process::id()
        ));
        if workdir.exists() {
            let _ = fs::remove_dir_all(&workdir);
        }
        fs::create_dir_all(&workdir).expect("create workdir");

        // 1. Compile the fixture schema → raw CGR (the codegen bin's input).
        let fixture_capnp = manifest.join("tests/wire_roundtrip_fixture.capnp");
        let cgr = workdir.join("wire_roundtrip_fixture.cgr");
        capnpc::CompilerCommand::new()
            .src_prefix(manifest.join("tests"))
            .file(&fixture_capnp)
            .raw_code_generator_request_path(&cgr)
            .run()
            .expect("compile wire_roundtrip_fixture.capnp → CGR");

        // 2. Run the real codegen bin (the same path www-cyberdione-ai uses) →
        //    capnp.ts + wire_roundtrip_fixture.ts + index.ts in the workdir.
        //    `CARGO_BIN_EXE_<name>` is a *runtime* env var under cargo 1.97 (not
        //    compile-time `env!`); fall back to locating it next to the test exe.
        let codegen_bin = codegen_bin_path();
        let cgr_dir = workdir.join("cgr");
        fs::create_dir_all(&cgr_dir).expect("create cgr dir");
        fs::copy(&cgr, cgr_dir.join("wire_roundtrip_fixture.cgr")).expect("stage cgr");
        let status = Command::new(&codegen_bin)
            .arg("--input-dir")
            .arg(&cgr_dir)
            .arg("--output-dir")
            .arg(&workdir)
            .status()
            .expect("run hyprstream-ts-codegen");
        assert!(status.success(), "hyprstream-ts-codegen failed");

        // 3. Stage the node harness next to the generated sources.
        let harness_src = manifest.join("tests/wire_roundtrip_harness.cjs");
        fs::copy(&harness_src, workdir.join("wire_roundtrip_harness.cjs")).expect("stage harness");

        // 4. Compile the generated TS to CommonJS JS (no strict, no @types/node
        //    needed — `--lib es2020,dom` covers DataView/Uint8Array/TextEncoder).
        let capnp_ts = workdir.join("capnp.ts");
        let fixture_ts = workdir.join("wire_roundtrip_fixture.ts");
        let tsc_status = Command::new(&tsc[0])
            .args(&tsc[1..])
            .args([
                "--module",
                "commonjs",
                "--target",
                "es2020",
                "--lib",
                "es2020,dom",
                "--skipLibCheck",
            ])
            .arg(&capnp_ts)
            .arg(&fixture_ts)
            .current_dir(&workdir)
            .status()
            .expect("run tsc");
        assert!(
            tsc_status.success(),
            "tsc compilation of generated TS failed"
        );

        Workdir {
            dir: workdir,
            node,
            skipped: false,
        }
    }

    fn is_skipped(&self) -> bool {
        self.skipped
    }

    /// Run the node harness in `mode`; return its parsed stdout JSON.
    fn run(&self, mode: &str, args: &[&str]) -> Value {
        let out = Command::new(&self.node)
            .arg(self.dir.join("wire_roundtrip_harness.cjs"))
            .arg(mode)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("run node {mode}: {e}"));
        assert!(
            out.status.success(),
            "node {mode} failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        serde_json::from_str(stdout.trim())
            .unwrap_or_else(|e| panic!("node {mode} stdout not JSON: {e}; stdout={stdout}"))
    }
}

impl Drop for Workdir {
    fn drop(&mut self) {
        if !self.skipped {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

// ---------------------------------------------------------------------------
// capnp-rust ↔ serde_json canonicalization (the wire oracle)
// ---------------------------------------------------------------------------

/// Independent Rust copy of `wire_roundtrip_harness.cjs::canonicalRoundtrip()`.
///
/// The TS→Rust leg must compare against this oracle, not only TS self-parse vs
/// Rust parse, otherwise a TS builder/parser pair that agrees on the wrong shape
/// could pass.
// The f64 fixture below intentionally uses the decimal expansion of e as a
// non-trivial full-precision value; it is a cross-runtime golden, so keep the
// literal byte-identical rather than swapping in std::f64::consts::E.
#[allow(clippy::approx_constant)]
fn expected_ts_roundtrip_json() -> Value {
    let big_f32s: Vec<Value> = (0..2048).map(|i| json!((i * 2) as f64)).collect();
    let big_embeds: Vec<Value> = (0..4)
        .map(|i| {
            let inner: Vec<Value> = (0..512)
                .map(|j| json!(i as f64 * 1000.0 + j as f64 * 0.5))
                .collect();
            json!(inner)
        })
        .collect();

    json!({
        "flag": true,
        "u8": 200,
        "u16": 60000,
        "u32": 4_000_000_000u64,
        "u64": "4294967297",
        "i8": -5,
        "i16": -1000,
        "i32": -70000,
        "i64": "-9007199254740993",
        "f32": 3.5,
        "f64": 2.718281828459045f64,
        "text": "héllo 🦀 wörld",
        "data": (0u8..=255).collect::<Vec<_>>(),
        "bools": [true, false, true, true, false],
        "u8s": [0, 7, 14, 21, 28, 35],
        "u16s": [10, 20, 30],
        "u32s": [1, 100, 10000],
        "u64s": ["1", "100", "4294967297"],
        "i8s": [-1, -2, 127],
        "i16s": [],
        "i32s": [-70000, 0, 70000],
        "i64s": ["-9007199254740993", "0", "9007199254740991"],
        "f32s": [1.5, 2.5, 3.5],
        "f64s": [1.5, 2.25, 3.125],
        "texts": ["a", "b🦀c", ""],
        "datas": [[1, 2, 3], [], [255, 254]],
        "embeds": [[1.5, 2.5, 3.5], [], [10.0, 20.0]],
        "bigF32s": big_f32s,
        "bigEmbeds": big_embeds,
    })
}

/// Read a serialized WireRoundtrip and render it in the same canonical JSON
/// shape the TS harness emits (bigints → decimal strings, `Data` → number[]).
fn roundtrip_to_json(bytes: &[u8]) -> Value {
    let message =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .expect("capnp read_message");
    let root: wire_roundtrip::Reader = message.get_root().expect("get root");

    // f32 widened to f64 — exact widening, matches the TS reader's f32→f64 read.
    let f32s = |l: capnp::primitive_list::Reader<f32>| {
        let v: Vec<Value> = (0..l.len()).map(|i| json!(l.get(i) as f64)).collect();
        json!(v)
    };
    let f64s = |l: capnp::primitive_list::Reader<f64>| {
        let v: Vec<Value> = (0..l.len()).map(|i| json!(l.get(i))).collect();
        json!(v)
    };
    let big_u64 = |l: capnp::primitive_list::Reader<u64>| {
        let v: Vec<Value> = (0..l.len()).map(|i| json!(l.get(i).to_string())).collect();
        json!(v)
    };
    let big_i64 = |l: capnp::primitive_list::Reader<i64>| {
        let v: Vec<Value> = (0..l.len()).map(|i| json!(l.get(i).to_string())).collect();
        json!(v)
    };

    json!({
        "flag": root.get_flag(),
        "u8": root.get_u8(),
        "u16": root.get_u16(),
        "u32": root.get_u32(),
        "u64": root.get_u64().to_string(),
        "i8": root.get_i8(),
        "i16": root.get_i16(),
        "i32": root.get_i32(),
        "i64": root.get_i64().to_string(),
        "f32": root.get_f32() as f64,
        "f64": root.get_f64(),
        "text": root.get_text().expect("text").to_str().expect("utf8"),
        "data": root.get_data().expect("data").to_vec(),
        "bools": root.get_bools().expect("bools").iter().collect::<Vec<_>>(),
        "u8s": root.get_u8s().expect("u8s").iter().collect::<Vec<_>>(),
        "u16s": root.get_u16s().expect("u16s").iter().collect::<Vec<_>>(),
        "u32s": root.get_u32s().expect("u32s").iter().collect::<Vec<_>>(),
        "u64s": big_u64(root.get_u64s().expect("u64s")),
        "i8s": root.get_i8s().expect("i8s").iter().collect::<Vec<_>>(),
        "i16s": root.get_i16s().expect("i16s").iter().collect::<Vec<_>>(),
        "i32s": root.get_i32s().expect("i32s").iter().collect::<Vec<_>>(),
        "i64s": big_i64(root.get_i64s().expect("i64s")),
        "f32s": f32s(root.get_f32s().expect("f32s")),
        "f64s": f64s(root.get_f64s().expect("f64s")),
        "texts": text_list_json(root.get_texts().expect("texts")),
        "datas": data_list_json(root.get_datas().expect("datas")),
        "embeds": nested_f32_json(root.get_embeds().expect("embeds")),
        "bigF32s": f32s(root.get_big_f32s().expect("bigF32s")),
        "bigEmbeds": nested_f64_json(root.get_big_embeds().expect("bigEmbeds")),
    })
}

fn text_list_json(l: capnp::text_list::Reader) -> Value {
    let v: Vec<Value> = (0..l.len())
        .map(|i| json!(l.get(i).expect("text elem").to_str().expect("utf8")))
        .collect();
    json!(v)
}

fn data_list_json(l: capnp::data_list::Reader) -> Value {
    let v: Vec<Value> = (0..l.len())
        .map(|i| json!(l.get(i).expect("data elem").to_vec()))
        .collect();
    json!(v)
}

fn nested_f32_json(l: capnp::list_list::Reader<capnp::primitive_list::Owned<f32>>) -> Value {
    let v: Vec<Value> = (0..l.len())
        .map(|i| {
            let inner = l.get(i).expect("embed inner");
            let xs: Vec<Value> = (0..inner.len())
                .map(|j| json!(inner.get(j) as f64))
                .collect();
            json!(xs)
        })
        .collect();
    json!(v)
}

fn nested_f64_json(l: capnp::list_list::Reader<capnp::primitive_list::Owned<f64>>) -> Value {
    let v: Vec<Value> = (0..l.len())
        .map(|i| {
            let inner = l.get(i).expect("embed inner");
            let xs: Vec<Value> = (0..inner.len()).map(|j| json!(inner.get(j))).collect();
            json!(xs)
        })
        .collect();
    json!(v)
}

/// Build a WireRoundtrip with capnp-rust and serialize it to the canonical wire bytes.
fn build_rust_roundtrip() -> Vec<u8> {
    let mut msg = capnp::message::Builder::<capnp::message::HeapAllocator>::new_default();
    let mut root: wire_roundtrip::Builder = msg.init_root();

    root.set_flag(true);
    root.set_u8(200);
    root.set_u16(60000);
    root.set_u32(4_000_000_000);
    root.set_u64(4_294_967_297);
    root.set_i8(-5);
    root.set_i16(-1000);
    root.set_i32(-70000);
    root.set_i64(-9_007_199_254_740_993);
    root.set_f32(3.5);
    root.set_f64(1234.56789);
    root.set_text("héllo 🦀 wörld");
    let scalar_data: Vec<u8> = (0u8..=255).collect();
    root.set_data(&scalar_data[..]);

    root.set_bools(&[true, false, true, true, false]).unwrap();
    root.set_u8s(&[0, 7, 14, 21, 28, 35]).unwrap();
    root.set_u16s(&[10, 20, 30]).unwrap();
    root.set_u32s(&[1, 100, 10000]).unwrap();
    root.set_u64s(&[1u64, 100, 4_294_967_297]).unwrap();
    root.set_i8s(&[-1i8, -2, 127]).unwrap();
    root.set_i16s(&[] as &[i16]).unwrap();
    root.set_i32s(&[-70000i32, 0, 70000]).unwrap();
    root.set_i64s(&[-9_007_199_254_740_993i64, 0, 9_007_199_254_740_991])
        .unwrap();
    root.set_f32s(&[1.5f32, 2.5, 3.5]).unwrap();
    root.set_f64s(&[1.5f64, 2.25, 3.125]).unwrap();

    // text_list / data_list via init (SetterInput for the whole list is awkward
    // to build inline; per-element init is unambiguous).
    {
        let mut tl = root.reborrow().init_texts(3);
        tl.set(0, "a");
        tl.set(1, "b🦀c");
        tl.set(2, "");
    }
    {
        let mut dl = root.reborrow().init_datas(3);
        dl.set(0, &[1u8, 2, 3]);
        dl.set(1, &[] as &[u8]);
        dl.set(2, &[255u8, 254]);
    }
    {
        let mut embeds = root.reborrow().init_embeds(3);
        let mut e0 = embeds.reborrow().init(0, 3);
        e0.set(0, 1.5);
        e0.set(1, 2.5);
        e0.set(2, 3.5);
        let _ = embeds.reborrow().init(1, 0); // empty inner
        let mut e2 = embeds.reborrow().init(2, 2);
        e2.set(0, 10.0);
        e2.set(1, 20.0);
    }
    // bigF32s / bigEmbeds intentionally left null → both sides read [].

    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &msg).expect("capnp write_message");
    bytes
}

/// Read a serialized WireChoice and render its `{variant, data}` shape as the
/// TS harness emits it (the `none` Void arm carries `data: undefined`, which
/// JSON drops — so the Rust oracle omits the key for that arm too).
fn choice_to_json(bytes: &[u8]) -> Value {
    let message =
        capnp::serialize::read_message(&mut &bytes[..], capnp::message::ReaderOptions::new())
            .expect("capnp read_message");
    let root: wire_choice::Reader = message.get_root().expect("get root");
    match root.which().expect("which") {
        wire_choice::Which::None(()) => json!({"variant": "none"}),
        wire_choice::Which::Count(v) => json!({"variant": "count", "data": v}),
        wire_choice::Which::Label(t) => {
            json!({"variant": "label", "data": t.expect("label").to_str().expect("utf8")})
        }
        wire_choice::Which::Values(l) => {
            let l = l.expect("values");
            let xs: Vec<Value> = (0..l.len()).map(|i| json!(l.get(i) as f64)).collect();
            json!({"variant": "values", "data": xs})
        }
        wire_choice::Which::Ranged(r) => json!({
            "variant": "ranged",
            "data": {
                "lo": r.reborrow().get_lo() as f64,
                "hi": r.get_hi() as f64,
            }
        }),
    }
}

fn build_rust_choice(arm: &str) -> Vec<u8> {
    let mut msg = capnp::message::Builder::<capnp::message::HeapAllocator>::new_default();
    let mut root: wire_choice::Builder = msg.init_root();
    match arm {
        "none" => root.set_none(()),
        "count" => root.set_count(42),
        "label" => root.set_label("rënamed 🦀"),
        "values" => root.set_values(&[1.5f32, 2.5, 3.5, 4.5]).unwrap(),
        "ranged" => {
            let mut r = root.init_ranged();
            r.set_lo(-1.5);
            r.set_hi(99.5);
        }
        other => panic!("unknown arm {other}"),
    }
    let mut bytes = Vec::new();
    capnp::serialize::write_message(&mut bytes, &msg).expect("capnp write_message");
    bytes
}

fn expected_choice_json(arm: &str) -> Value {
    match arm {
        "none" => json!({"variant": "none"}),
        "count" => json!({"variant": "count", "data": 42}),
        "label" => json!({"variant": "label", "data": "rënamed 🦀"}),
        "values" => json!({"variant": "values", "data": [1.5, 2.5, 3.5, 4.5]}),
        "ranged" => json!({
            "variant": "ranged",
            "data": {
                "lo": -1.5,
                "hi": 99.5,
            }
        }),
        other => panic!("unknown arm {other}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn ts2rust_roundtrip_full_coverage() {
    let wd = Workdir::prepare();
    if wd.is_skipped() {
        return;
    }

    // TS builds (incl. growth-stress lists) → bytes; Rust reads them back.
    let bytes_path = wd.dir.join("ts2rust.bin");
    let ts_self_parse = wd.run("ts2rust", &[bytes_path.to_str().unwrap()]);
    let bytes = fs::read(&bytes_path).expect("read ts2rust bytes");
    let rust_read = roundtrip_to_json(&bytes);
    let expected = expected_ts_roundtrip_json();

    assert!(
        values_equal(&ts_self_parse, &expected),
        "TS→Rust mismatch: generated TS self-parse does not match the canonical \
         payload requested from the builder. \
         ts_self_parse={ts_self_parse:#} expected={expected:#}"
    );
    assert!(
        values_equal(&rust_read, &expected),
        "TS→Rust mismatch: generated TS builder produced bytes that capnp-rust \
         reads differently from the canonical requested payload. \
         rust_read={rust_read:#} expected={expected:#}"
    );

    // The growth cases must actually be present (otherwise the test is silently
    // not exercising the #725/#772 bug class). ~16 KB of nested Float64 forces
    // multiple arena growths past the ~1 KB slack.
    let big_embeds = rust_read
        .get("bigEmbeds")
        .and_then(|v| v.as_array())
        .unwrap();
    assert!(
        big_embeds.len() == 4
            && big_embeds[0]
                .as_array()
                .map(|a| a.len() == 512)
                .unwrap_or(false),
        "growth-stress nested list didn't round-trip at full size: {big_embeds:#?}"
    );
    let big_f32s = rust_read.get("bigF32s").and_then(|v| v.as_array()).unwrap();
    assert_eq!(big_f32s.len(), 2048, "growth-stress flat list truncated");
}

#[test]
fn rust2ts_roundtrip_full_coverage() {
    let wd = Workdir::prepare();
    if wd.is_skipped() {
        return;
    }

    // Rust builds → bytes; generated TS parser reads them back.
    let bytes_path = wd.dir.join("rust2ts.bin");
    let bytes = build_rust_roundtrip();
    fs::write(&bytes_path, &bytes).expect("write rust2ts bytes");
    let ts_parse = wd.run("rust2ts", &[bytes_path.to_str().unwrap()]);
    let rust_read = roundtrip_to_json(&bytes);

    assert!(
        values_equal(&ts_parse, &rust_read),
        "Rust→TS mismatch: capnp-rust-produced bytes parsed differently by the \
         generated TS parser. ts_parse={ts_parse:#} rust_read={rust_read:#}"
    );
}

#[test]
fn choice_roundtrip_both_directions() {
    let wd = Workdir::prepare();
    if wd.is_skipped() {
        return;
    }

    for arm in ["none", "count", "label", "values", "ranged"] {
        // TS→Rust
        let t2r = wd.dir.join(format!("choice_ts2rust_{arm}.bin"));
        let ts_self = wd.run("choice-ts2rust", &[t2r.to_str().unwrap(), arm]);
        let rust_read = choice_to_json(&fs::read(&t2r).expect("read ts2rust choice"));
        let expected = expected_choice_json(arm);
        assert!(
            values_equal(&ts_self, &expected),
            "TS→Rust WireChoice ({arm}) self-parse did not select the requested \
             union arm/data: ts_self={ts_self:#} expected={expected:#}"
        );
        assert!(
            values_equal(&rust_read, &expected),
            "TS→Rust WireChoice ({arm}) bytes did not encode the requested union \
             arm/data: rust_read={rust_read:#} expected={expected:#}"
        );

        // Rust→TS
        let r2t = wd.dir.join(format!("choice_rust2ts_{arm}.bin"));
        let bytes = build_rust_choice(arm);
        fs::write(&r2t, &bytes).expect("write rust2ts choice");
        let ts_parse = wd.run("choice-rust2ts", &[r2t.to_str().unwrap()]);
        let rust_read = choice_to_json(&bytes);
        assert!(
            values_equal(&ts_parse, &rust_read),
            "Rust→TS WireChoice ({arm}) mismatch: ts_parse={ts_parse:#} rust_read={rust_read:#}"
        );
    }
}
