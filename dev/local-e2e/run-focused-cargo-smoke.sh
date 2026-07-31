#!/usr/bin/env bash
# Invoked once inside the fleet BuildQ lease by run-offline-smoke.sh.
set -Eeuo pipefail

umask 077

command -v rg >/dev/null 2>&1 || {
  printf 'focused-cargo-smoke: required tool rg (ripgrep) is not on PATH\n' >&2
  exit 1
}

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cargo_tmp=
cleanup() {
  [[ -z "$cargo_tmp" || ! -e "$cargo_tmp" ]] || rm -r -- "$cargo_tmp"
}
# Cleanup is armed before the first mutation.
trap cleanup EXIT INT TERM
install -d -m 0700 "$harness_dir/runs"
cargo_tmp="$(mktemp -d "$harness_dir/runs/cargo-smoke.XXXXXX")"
chmod 0700 "$cargo_tmp"
export TMPDIR="$cargo_tmp"

build_lib_test_binary() {
  local package=$1
  local target_name=$2
  local executable
  local metadata="$cargo_tmp/$target_name.cargo.jsonl"
  local cargo_status=0
  cargo test --locked -p "$package" --lib --no-run \
    --message-format=json-render-diagnostics >"$metadata" ||
    cargo_status=$?
  if ((cargo_status != 0)); then
    return "$cargo_status"
  fi
  executable="$(
    python3 -c '
import json
import sys

target_name = sys.argv[1]
executable = None
with open(sys.argv[2], encoding="utf-8") as messages:
  for line in messages:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    target = message.get("target", {})
    if (
        message.get("reason") == "compiler-artifact"
        and target.get("name", "").replace("-", "_") == target_name
        and message.get("profile", {}).get("test") is True
        and message.get("executable")
    ):
        executable = message["executable"]
if executable is not None:
    print(executable)
' "$target_name" "$metadata"
  )"
  # No mtime-guessing fallback: CARGO_TARGET_DIR is a shared BuildQ slot, and
  # the newest matching binary under it can belong to a different worktree's
  # concurrent build of the same package. Fail closed instead of laundering
  # a foreign artifact through run_exact as if this checkout produced it.
  [[ -n "$executable" && -x "$executable" ]] || {
    printf 'focused-cargo-smoke: no executable test artifact for %s in this invocation'"'"'s cargo JSON output\n' \
      "$target_name" >&2
    return 1
  }
  printf '%s\n' "$executable"
}

run_exact() {
  local test_binary=$1
  local test_name=$2
  local listed
  listed="$("$test_binary" --list --exact "$test_name")"
  [[ "$(printf '%s\n' "$listed" | rg -c ': test$')" == "1" ]] &&
    printf '%s\n' "$listed" | rg -Fxq "$test_name: test" || {
    printf 'focused-cargo-smoke: expected exactly one named test: %s\n' \
      "$test_name" >&2
    return 1
  }
  "$test_binary" --exact "$test_name" --nocapture
}

# A shared target can contain a fresh fingerprint from an older worktree with
# the same workspace package ID. Refresh this leaf's mtime while holding the
# exclusive lease so Cargo recompiles the current checkout's public interface.
# Applied once, before the first build that depends on hyprstream-util
# (hyprstream-discovery and hyprstream both do; hyprstream-rpc does not), so
# every dependent build in this script sees the refreshed fingerprint rather
# than only the last one.
[[ -f crates/hyprstream-util/src/lib.rs && ! -L crates/hyprstream-util/src/lib.rs ]]
touch crates/hyprstream-util/src/lib.rs

discovery_test="$(build_lib_test_binary hyprstream-discovery hyprstream_discovery)"
run_exact "$discovery_test" \
  service::resolver_tests::deployment_trust_path_resolution_is_explicit_and_split
run_exact "$discovery_test" \
  service::resolver_tests::invalid_present_trust_path_never_falls_back
run_exact "$discovery_test" \
  service::resolver_tests::user_service_paths_reject_writable_files_and_symlinks
run_exact "$discovery_test" \
  service::resolver_tests::user_service_paths_reject_writable_or_symlinked_ancestors
run_exact "$discovery_test" \
  service::resolver_tests::user_service_paths_preserve_complete_enrolled_verification

rpc_test="$(build_lib_test_binary hyprstream-rpc hyprstream_rpc)"
run_exact "$rpc_test" \
  transport::lazy_quinn::tests::lazy_connects_on_first_send_and_caches
run_exact "$rpc_test" \
  transport::lazy_quinn::tests::wrong_cert_pin_does_not_connect

hyprstream_test="$(build_lib_test_binary hyprstream hyprstream_core)"
run_exact "$hyprstream_test" \
  services::oauth::xrpc::tests::service_auth_query_is_strict_and_method_bound
run_exact "$hyprstream_test" \
  services::oauth::browser_session::tests::local_atproto_exchange_cookie_whoami_and_register_end_to_end
run_exact "$hyprstream_test" \
  services::oauth::browser_session::tests::unauthenticated_whoami_is_public_floor
run_exact "$hyprstream_test" \
  services::oauth::browser_session::tests::exchange_requires_dpop_and_consumes_service_assertion_once
run_exact "$hyprstream_test" \
  services::inference::tenant_binding_tests::legacy_in_process_config_preserves_model_based_readiness
run_exact "$hyprstream_test" \
  services::inference::tests::normal_completion_captures_full_stats
