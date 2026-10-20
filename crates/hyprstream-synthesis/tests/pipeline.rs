//! Actual CLI fails closed before any synthetic run when admission is missing.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::path::PathBuf;
fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hyprstream-bench/manifest/vob-1.1.manifest.json")
}
#[test]
fn cli_rejects_malformed_config_values() {
    // Regression (review rounds): malformed config must fail loud — unknown
    // or non-string label_policy never silently selects Unlimited, and
    // out-of-range integers never wrap into a different run configuration.
    let bin = env!("CARGO_BIN_EXE_hyprstream-synthesis");
    let dir = std::env::temp_dir();
    let roster_path = dir.join("p13-test-roster.json");
    let config_path = dir.join("p13-test-config.json");
    std::fs::write(
        &roster_path,
        r#"{"teachers":[{"id":"sim","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let run_with = |config: &str| -> std::process::Output {
        std::fs::write(&config_path, config).unwrap();
        std::process::Command::new(bin)
            .args([
                "run",
                "--roster",
                roster_path.to_str().unwrap(),
                "--manifest",
                manifest_path().to_str().unwrap(),
                "--config",
                config_path.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    for bad in [
        r#"[]"#,
        r#"null"#,
        r#""balanced""#,
        r#"42"#,
        r#"{"label_policy":"sometimes","base_items_per_family":1}"#,
        r#"{"label_policy":false,"base_items_per_family":1}"#,
        r#"{"label_policy":42,"base_items_per_family":1}"#,
        r#"{"label_policy":null,"base_items_per_family":1}"#,
        r#"{"base_items_per_family":4294967296}"#,
        r#"{"paraphrase_variants":4294967298}"#,
        r#"{"base_items_per_family":false}"#,
        r#"{"paraphrase_variants":-1}"#,
        r#"{"seed_base":"0x51A1"}"#,
        r#"{"label_policy":"balanced","label_slack":-1}"#,
        r#"{"label_policy":"balanced","label_slack":0}"#,
        r#"{"base_item_per_family":5}"#,
        r#"{"base_items_per_family":0}"#,
        r#"{"unknown_key":1}"#,
        r#"{"label_policy":"unlimited","label_slack":"high"}"#,
        r#"{"label_slack":false}"#,
    ] {
        let out = run_with(bad);
        assert_eq!(
            out.status.code(),
            Some(1),
            "config {bad} must be rejected, stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("config unreadable or unparseable"),
            "invalid config must fail before content-index admission: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    for config in [
        r#"{"label_policy":"unlimited","base_items_per_family":1}"#.to_owned(),
        r#"{"label_policy":"balanced","label_slack":1}"#.to_owned(),
        format!(
            r#"{{"label_policy":"balanced","label_slack":{}}}"#,
            usize::MAX
        ),
    ] {
        let good = run_with(&config);
        assert!(
            !good.status.success()
                && String::from_utf8_lossy(&good.stderr).contains("content index"),
            "stderr: {}",
            String::from_utf8_lossy(&good.stderr)
        );
    }
}

#[test]
fn cli_rejects_unknown_arguments() {
    // Regression (review): argument handling used to search for known flags
    // and ignore the rest, so a typo'd flag silently reverted to defaults —
    // a misspelled --publishable-only would silently export encumbered rows.
    let bin = env!("CARGO_BIN_EXE_hyprstream-synthesis");
    let dir = std::env::temp_dir();
    let roster_path = dir.join("p13-test-roster.json");
    std::fs::write(
        &roster_path,
        r#"{"teachers":[{"id":"sim","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let roster = roster_path.to_str().unwrap();
    let manifest = manifest_path().to_str().unwrap().to_owned();
    for bad_args in [
        vec![
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--confg",
            "tuned.json",
        ],
        vec![
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--publishable",
        ],
        vec!["run", "--roster", roster, "--manifest", &manifest, "stray"],
        vec!["run", "--roster"],
        vec![
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--out",
            "--publishable-only",
        ],
        vec!["roster-md", "--roster", roster, "--verbse"],
    ] {
        let out = std::process::Command::new(bin)
            .args(&bad_args)
            .output()
            .unwrap();
        assert_eq!(
            out.status.code(),
            Some(2),
            "args {bad_args:?} must be rejected, stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // The correctly-spelled flags still pass validation.
    let good = std::process::Command::new(bin)
        .args([
            "run",
            "--roster",
            roster,
            "--manifest",
            &manifest,
            "--publishable-only",
        ])
        .output()
        .unwrap();
    assert!(
        !good.status.success() && String::from_utf8_lossy(&good.stderr).contains("content index"),
        "stderr: {}",
        String::from_utf8_lossy(&good.stderr)
    );
}

#[test]
fn cli_fails_closed_when_disclosure_path_is_unresolvable() {
    // Regression (review): supplying the pinned manifest as a single-
    // component relative path made disclosure_path_for return None, and the
    // loader silently skipped disclosure verification — the exact pinned
    // bytes would load even with DISCLOSURE.md absent or drifted. The
    // unresolvable branch now fails closed with DisclosureIo.
    let bin = env!("CARGO_BIN_EXE_hyprstream-synthesis");
    let dir = std::env::temp_dir().join(format!("p13-fw-bare-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(manifest_path(), dir.join("vob-1.1.manifest.json")).unwrap();
    let roster_path = dir.join("roster.json");
    std::fs::write(
        &roster_path,
        r#"{"teachers":[{"id":"sim","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let out = std::process::Command::new(bin)
        .current_dir(&dir)
        .args([
            "run",
            "--roster",
            roster_path.to_str().unwrap(),
            "--manifest",
            "vob-1.1.manifest.json",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("DISCLOSURE"),
        "stderr must name the disclosure failure: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cli_requires_complete_index_and_never_creates_output_on_rejection() {
    let dir =
        std::env::temp_dir().join(format!("synthesis-index-admission-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let roster = dir.join("roster.json");
    std::fs::write(
        &roster,
        r#"{"teachers":[{"id":"fake","version":"simulated","tos_class":"open-weights"}]}"#,
    )
    .unwrap();
    let index = dir.join("index.json");
    let out = dir.join("must-not-exist.jsonl");
    for contents in [
        None,
        Some("corrupt"),
        Some(r#"{"version":"decision-content/1","items":[]}"#),
    ] {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_hyprstream-synthesis"));
        command
            .arg("run")
            .arg("--roster")
            .arg(&roster)
            .arg("--manifest")
            .arg(manifest_path())
            .arg("--out")
            .arg(&out);
        if let Some(contents) = contents {
            std::fs::write(&index, contents).unwrap();
            command.arg("--content-index").arg(&index);
        }
        let result = command.output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&result.stderr).contains("content index"));
        assert!(!out.exists());
        assert!(result.stdout.is_empty());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
