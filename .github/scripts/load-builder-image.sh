#!/usr/bin/env bash
# Load the sole tracked arm64 builder-image pin for the calling GitHub Actions job.
set -euo pipefail

pin_file="${GITHUB_WORKSPACE:-$PWD}/.github/builder-image.env"
[[ -f "$pin_file" ]]
[[ $(wc -l < "$pin_file") -eq 1 ]]

image=$(<"$pin_file")
[[ "$image" =~ ^ghcr\.io/hyprstream/rust-builder-arm64@sha256:[0-9a-f]{64}$ ]]

printf 'BUILDER_IMAGE=%s\n' "$image" >> "$GITHUB_ENV"
