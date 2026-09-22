//! `hyprstream-synthesis` CLI: run the synthesis pipeline against the frozen
//! vob-1.1 firewall, or print the consumed pins.
//!
//! ```text
//! hyprstream-synthesis pins
//! hyprstream-synthesis run --roster roster.json [--manifest <path>] \
//!     [--config <config.json>] [--out corpus.jsonl] [--publishable-only]
//! hyprstream-synthesis roster-md --roster roster.json   # DISCLOSURE table
//! ```
//!
//! The CLI instantiates deterministic [`HashTeacher`] stand-ins from the
//! roster (versions pinned `simulated`): it exercises the full pipeline and
//! corpus format. Real teacher adapters implement `teacher::Teacher` in the
//! eval harness (P0.4) and the training loop (P1.4); the roster JSON shape
//! is the integration contract.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

use hyprstream_synthesis::firewall::{
    Firewall, DEFAULT_MANIFEST_PATH, EXPECTED_DISCLOSURE_BLAKE3, EXPECTED_MANIFEST_BLAKE3,
    EXPECTED_RELEASE,
};
use hyprstream_synthesis::labelctl::LabelPolicy;
use hyprstream_synthesis::pipeline::{run, SynthConfig};
use hyprstream_synthesis::teacher::{HashTeacher, Roster, TeacherPin};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("pins") => pins(),
        Some("run") => synthesize(&args[2..]),
        Some("roster-md") => roster_md(&args[2..]),
        _ => {
            eprintln!(
                "usage: hyprstream-synthesis <pins|run --roster roster.json [--manifest path] [--config config.json] [--out corpus.jsonl] [--publishable-only]|roster-md --roster roster.json>"
            );
            2
        }
    };
    std::process::exit(code);
}

fn pins() -> i32 {
    println!("release:            {EXPECTED_RELEASE}");
    println!("manifest blake3:    {EXPECTED_MANIFEST_BLAKE3}");
    println!("disclosure blake3:  {EXPECTED_DISCLOSURE_BLAKE3}");
    println!("default manifest:   {DEFAULT_MANIFEST_PATH}");
    0
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

/// Reject any argument that is not a recognized flag or a consumed flag
/// value. A typo such as `--confg tuned.json` must fail loud, never silently
/// revert to the default configuration — and a misspelled
/// `--publishable-only` must never silently export encumbered rows.
fn reject_unknown_args(
    args: &[String],
    value_flags: &[&str],
    bool_flags: &[&str],
) -> Option<String> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if bool_flags.contains(&arg) {
            index += 1;
        } else if value_flags.contains(&arg) {
            if index + 1 >= args.len() {
                return Some(format!("{arg} requires a value"));
            }
            index += 2;
        } else {
            return Some(format!("unrecognized argument {arg}"));
        }
    }
    None
}

fn load_roster(args: &[String]) -> Option<Roster> {
    let path = flag_value(args, "--roster")?;
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("roster unreadable at {path}: {err}");
            return None;
        }
    };
    match Roster::from_json(&text) {
        Ok(roster) => Some(roster),
        Err(err) => {
            eprintln!("roster unparseable at {path}: {err}");
            None
        }
    }
}

fn roster_md(args: &[String]) -> i32 {
    if let Some(err) = reject_unknown_args(args, &["--roster"], &[]) {
        eprintln!("roster-md: {err}");
        return 2;
    }
    match load_roster(args) {
        Some(roster) => {
            print!("{}", roster.to_markdown_table());
            0
        }
        None => {
            eprintln!("roster-md requires --roster <roster.json>");
            1
        }
    }
}

fn synthesize(args: &[String]) -> i32 {
    if let Some(err) = reject_unknown_args(
        args,
        &["--roster", "--manifest", "--config", "--out"],
        &["--publishable-only"],
    ) {
        eprintln!("run: {err}");
        return 2;
    }
    let Some(roster) = load_roster(args) else {
        eprintln!("run requires --roster <roster.json>");
        return 1;
    };
    let manifest = flag_value(args, "--manifest").unwrap_or(DEFAULT_MANIFEST_PATH);
    let firewall = match Firewall::load(Path::new(manifest)) {
        Ok(firewall) => firewall,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    let config = match flag_value(args, "--config") {
        Some(path) => match std::fs::read_to_string(path)
            .ok()
            .and_then(|text| parse_config(&text))
        {
            Some(config) => config,
            None => {
                eprintln!(
                    "config unreadable or unparseable at {path} (label_policy must be \"balanced\" or \"unlimited\")"
                );
                return 1;
            }
        },
        None => SynthConfig::default(),
    };
    // Simulated stand-in teachers, keyed per roster position. Versions are
    // pinned by the roster file itself.
    let hash_teachers: Vec<HashTeacher> = roster
        .teachers
        .iter()
        .enumerate()
        .map(|(i, pin)| {
            HashTeacher::new(
                TeacherPin {
                    version: format!("simulated:{}", pin.version),
                    ..pin.clone()
                },
                0x5EED + i as u64,
                1.0,
            )
        })
        .collect();
    let teachers: Vec<&dyn hyprstream_synthesis::Teacher> = hash_teachers
        .iter()
        .map(|teacher| teacher as &dyn hyprstream_synthesis::Teacher)
        .collect();
    // No fitted P0.5 corrections exist at dry-run time (the simulated
    // stand-ins are temperature-neutral), so the correction map is empty.
    let corpus = match run(&config, &teachers, &firewall, &HashMap::new()) {
        Ok(corpus) => corpus,
        Err(err) => {
            eprintln!("synthesis failed: {err}");
            return 1;
        }
    };
    let publishable_only = has_flag(args, "--publishable-only");
    let jsonl = corpus.to_jsonl(publishable_only);
    match flag_value(args, "--out") {
        Some(path) => {
            if let Err(err) =
                std::fs::File::create(path).and_then(|mut file| file.write_all(jsonl.as_bytes()))
            {
                eprintln!("cannot write {path}: {err}");
                return 1;
            }
        }
        None => print!("{jsonl}"),
    }
    eprintln!(
        "generated={} accepted={} dedup_dropped={} label_deferred={} publishable={}",
        corpus.stats.generated,
        corpus.stats.accepted,
        corpus.stats.dedup_dropped,
        corpus.stats.label_deferred,
        corpus.publishable().count()
    );
    0
}

fn parse_config(text: &str) -> Option<SynthConfig> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    // The config must be a JSON object; `[]`, `null`, or a bare string would
    // otherwise silently yield the default configuration.
    if !value.is_object() {
        return None;
    }
    // Unknown keys fail loud: a misspelled setting must never silently fall
    // back to its default.
    const KNOWN_KEYS: [&str; 5] = [
        "seed_base",
        "base_items_per_family",
        "paraphrase_variants",
        "label_policy",
        "label_slack",
    ];
    if value
        .as_object()
        .is_some_and(|object| object.keys().any(|key| !KNOWN_KEYS.contains(&key.as_str())))
    {
        return None;
    }
    // Strict presence semantics: an absent key takes the default, but a
    // present value that is not a u64 (false, -1, "16", 1.5, ...) fails loud
    // instead of silently reverting to the default.
    let get_u64 = |key: &str, default: u64| -> Option<u64> {
        match value.get(key) {
            None => Some(default),
            Some(raw) => raw.as_u64(),
        }
    };
    // Checked narrowing: a value that does not fit u32/usize must fail loud,
    // never silently wrap into a different run configuration.
    let get_u32 = |key: &str, default: u32| -> Option<u32> {
        u32::try_from(get_u64(key, u64::from(default))?).ok()
    };
    // label_slack is validated whenever present, even if balancing is off —
    // a malformed value must fail loud regardless of the selected policy.
    let label_slack = get_u64("label_slack", 1)?;
    let policy = match value.get("label_policy") {
        None => LabelPolicy::Unlimited,
        Some(serde_json::Value::String(s)) if s == "balanced" => LabelPolicy::Balanced {
            slack: usize::try_from(label_slack).ok()?,
        },
        Some(serde_json::Value::String(s)) if s == "unlimited" => LabelPolicy::Unlimited,
        // Unknown or non-string policies fail loud — silently dropping label
        // control would skew a training run without any signal.
        Some(_) => return None,
    };
    let base_items_per_family = get_u32("base_items_per_family", 16)?;
    // A zero-sized run writes an empty corpus while reporting success.
    if base_items_per_family == 0 {
        return None;
    }
    Some(SynthConfig {
        seed_base: get_u64("seed_base", 0x51A1)?,
        base_items_per_family,
        paraphrase_variants: get_u32("paraphrase_variants", 2)?,
        label_policy: policy,
    })
}
