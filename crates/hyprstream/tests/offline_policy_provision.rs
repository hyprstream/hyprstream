//! Real-binary proof that offline policy provisioning returns before resolver
//! or credential initialization.
#![allow(clippy::expect_used)]

use std::process::Command;

#[test]
fn provisions_with_no_resolver_or_credentials() {
    let root = tempfile::tempdir().expect("isolated process root");
    let models = root.path().join("models");
    let config = root.path().join("config");
    let cache = root.path().join("cache");
    let loras = root.path().join("loras");

    let output = Command::new(env!("CARGO_BIN_EXE_hyprstream"))
        .env("HOME", root.path())
        .env("XDG_CONFIG_HOME", root.path().join("xdg-config"))
        .env("XDG_DATA_HOME", root.path().join("xdg-data"))
        .env("XDG_CACHE_HOME", root.path().join("xdg-cache"))
        .env("XDG_RUNTIME_DIR", root.path().join("missing-runtime"))
        .env("HYPRSTREAM__STORAGE__MODELS_DIR", &models)
        .env("HYPRSTREAM__STORAGE__CONFIG_DIR", &config)
        .env("HYPRSTREAM__STORAGE__CACHE_DIR", &cache)
        .env("HYPRSTREAM__STORAGE__LORAS_DIR", &loras)
        .env(
            "HYPRSTREAM__SECRETS__PATH",
            root.path().join("missing-credentials"),
        )
        .args([
            "service",
            "provision-policy-templates",
            "--template",
            "public-inference",
            "--template",
            "public-read",
        ])
        .output()
        .expect("run hyprstream binary");

    assert!(
        output.status.success(),
        "offline provisioning failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(models.join(".registry/policies/policy.csv").is_file());
    assert!(!root.path().join("missing-credentials").exists());
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("verified 2 policy template(s): public-inference,public-read"));
}
