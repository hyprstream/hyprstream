#!/usr/bin/env bash
# Explicit program-leaf coverage, independent of affected-set/default-members.
set -euo pipefail

case "${1:-}" in
  lint)
    cargo clippy --locked --release -p hyprstream-synthesis --all-targets -- -D warnings
    ;;
  test)
    cargo nextest run --locked --cargo-profile release --profile ci \
      -p hyprstream-synthesis --all-targets
    cargo test --locked --release -p hyprstream-synthesis --doc
    ;;
  *)
    echo "usage: synthesis-ci.sh lint|test" >&2
    exit 2
    ;;
esac
