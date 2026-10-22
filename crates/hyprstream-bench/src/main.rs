//! `hyprstream-bench` CLI: generate the frozen release, verify the committed
//! manifest, or list the firewall designations.
//!
//! ```text
//! hyprstream-bench generate <outdir>   # items.jsonl + vob-1.0.manifest.json
//! hyprstream-bench verify [manifest]   # regenerate + compare (default: committed)
//! hyprstream-bench families            # print family designations (the firewall)
//! ```

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::Write as _;
use std::path::Path;

use hyprstream_bench::family::Designation;
use hyprstream_bench::gen::{generate_all, BenchConfig};
use hyprstream_bench::manifest::{blake3_hex, build, Manifest};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("generate") => generate(args.get(2).map(String::as_str)),
        Some("verify") => verify(args.get(2).map(String::as_str)),
        Some("families") => families(),
        _ => {
            eprintln!(
                "usage: hyprstream-bench <generate <outdir>|verify [manifest.json]|families>"
            );
            2
        }
    };
    std::process::exit(code);
}

fn disclosure_digest() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("DISCLOSURE.md");
    match std::fs::read(&path) {
        Ok(bytes) => blake3_hex(&bytes),
        Err(err) => {
            eprintln!("DISCLOSURE.md unreadable at {}: {err}", path.display());
            std::process::exit(1);
        }
    }
}

fn item_json(item: &hyprstream_bench::Item) -> serde_json::Value {
    serde_json::json!({
        "id": item.id,
        "family": item.family.as_str(),
        "stratum": item.stratum.as_str(),
        "group": item.group,
        "seed": format!("0x{:016x}", item.seed),
        "state": item.state.canonical_text(),
        "question": {
            "kind": item.question.kind.as_str(),
            "instructions": item.question.instructions.as_ref().map(hyprstream_decision::Entry::canonical_text),
            "labels": item.question.labels(),
        },
        "truth": item.truth,
        "truth_label": item.truth_label(),
        "blake3": item.hash(),
    })
}

fn generate(outdir: Option<&str>) -> i32 {
    let Some(outdir) = outdir else {
        eprintln!("generate: missing <outdir>");
        return 2;
    };
    let config = BenchConfig::default();
    let items = generate_all(&config);
    let manifest = build(&config, &items, disclosure_digest());
    let dir = Path::new(outdir);
    if let Err(err) = std::fs::create_dir_all(dir) {
        eprintln!("generate: cannot create {outdir}: {err}");
        return 1;
    }
    let items_path = dir.join("items.jsonl");
    let manifest_path = dir.join(format!("{}.manifest.json", manifest.release));
    let mut items_file = match std::fs::File::create(&items_path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("generate: cannot write {}: {err}", items_path.display());
            return 1;
        }
    };
    for item in &items {
        let line = match serde_json::to_string(&item_json(item)) {
            Ok(line) => line,
            Err(_) => unreachable!("item JSON is a plain Value tree"),
        };
        if writeln!(items_file, "{line}").is_err() {
            eprintln!("generate: write failed for {}", items_path.display());
            return 1;
        }
    }
    if let Err(err) = std::fs::write(&manifest_path, manifest.to_json()) {
        eprintln!("generate: cannot write {}: {err}", manifest_path.display());
        return 1;
    }
    println!(
        "generated {} items → {} + {}",
        items.len(),
        items_path.display(),
        manifest_path.display()
    );
    0
}

fn verify(path: Option<&str>) -> i32 {
    let default_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("manifest")
        .join(format!("{}.manifest.json", hyprstream_bench::RELEASE));
    let path = path.map_or(default_path, |p| Path::new(p).to_path_buf());
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("verify: cannot read {}: {err}", path.display());
            return 1;
        }
    };
    let manifest = match Manifest::from_json(&text) {
        Ok(manifest) => manifest,
        Err(err) => {
            eprintln!("verify: manifest JSON invalid: {err}");
            return 1;
        }
    };
    if manifest.disclosure.blake3 != disclosure_digest() {
        eprintln!("verify: DISCLOSURE.md changed since freeze (firewall artifact drift)");
        return 1;
    }
    match manifest.verify(&BenchConfig::default()) {
        Ok(()) => {
            println!(
                "verify: OK — {} items regenerate bit-identically ({} gate families firewalled)",
                manifest.item_count,
                manifest.gate_families().len()
            );
            0
        }
        Err(err) => {
            eprintln!("verify: FAILED — {err}");
            1
        }
    }
}

fn families() -> i32 {
    for row in build(&BenchConfig::default(), &[], disclosure_digest()).families {
        let marker = if row.designation == Designation::Gate.as_str() {
            " (FIREWALLED — never synthesized into training data)"
        } else {
            ""
        };
        println!("{}: {}{}", row.id, row.designation, marker);
    }
    0
}
