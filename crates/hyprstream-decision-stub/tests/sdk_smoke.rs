//! Stock-SDK portability proof (P0.7, corrected by S6a §4): the unmodified
//! `typesafe-sdk-python` and `@typesafe-ai/sdk` clients run against the stub facade via
//! `TYPESAFE_BASE_URL`.
//!
//! Gated behind `HYPRSTREAM_SDK_SMOKE=1` because it needs network access (PyPI/npm) and a
//! built stub binary; `scripts/sdk_smoke.sh` is the same check runnable by hand.

use std::process::Command;

#[test]
fn stock_sdks_round_trip_against_stub() {
    if std::env::var("HYPRSTREAM_SDK_SMOKE").as_deref() != Ok("1") {
        // Skipped by default: needs PyPI/npm access. Run scripts/sdk_smoke.sh by hand
        // or set HYPRSTREAM_SDK_SMOKE=1.
        return;
    }
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/sdk_smoke.sh");
    let binary = env!("CARGO_BIN_EXE_hyprstream-decision-stub");
    let status = Command::new("bash")
        .arg(script)
        .arg(binary)
        .status()
        .unwrap_or_else(|error| panic!("spawn sdk_smoke.sh: {error}"));
    assert!(status.success(), "SDK smoke test failed: {status}");
}
