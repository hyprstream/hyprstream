#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

LOCAL_E2E_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
HYPRSTREAM_REPO="$(cd -- "$LOCAL_E2E_DIR/../.." && pwd -P)"
LOCAL_E2E_CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME:?HOME is required}/.config}"
LOCAL_E2E_RUNTIME_PARENT="${XDG_RUNTIME_DIR:?XDG_RUNTIME_DIR is required for systemd --user}"
LOCAL_E2E_CONFIG_DIR="$LOCAL_E2E_CONFIG_HOME/hyprstream/local-e2e"
LOCAL_E2E_STATE_DIR="$LOCAL_E2E_DIR/state"
LOCAL_E2E_RUNTIME_DIR="$LOCAL_E2E_RUNTIME_PARENT/hyprstream"
LOCAL_E2E_GENERATED_DIR="$LOCAL_E2E_DIR/generated"
LOCAL_E2E_PRIVATE_DIR="$LOCAL_E2E_DIR/private"
LOCAL_E2E_BINARY="${HYPRSTREAM_BIN:-$HYPRSTREAM_REPO/target/release/hyprstream}"

local_e2e_log() {
  printf 'local-e2e: %s\n' "$*" >&2
}

local_e2e_die() {
  local_e2e_log "ERROR: $*"
  exit 1
}

local_e2e_require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    local_e2e_die "required command is unavailable: $1"
}

local_e2e_assert_real_dir() {
  local path=$1
  [[ -d "$path" && ! -L "$path" ]] ||
    local_e2e_die "expected a real directory, not a symlink: $path"
}

local_e2e_assert_regular_file() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] ||
    local_e2e_die "expected a regular, non-symlink file: $path"
}

local_e2e_assert_socket() {
  local path=$1
  [[ -S "$path" ]] ||
    local_e2e_die "expected a Unix socket: $path"
}

local_e2e_wait_socket() {
  local path=$1
  local attempts=${2:-60}
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    [[ -S "$path" ]] && return 0
    sleep 1
  done
  local_e2e_die "Unix socket did not become ready: $path"
}

local_e2e_safe_link() {
  local link_path=$1
  local expected_target=$2
  local parent
  parent="$(dirname -- "$link_path")"
  local_e2e_assert_real_dir "$parent"

  if [[ -L "$link_path" ]]; then
    [[ "$(readlink -- "$link_path")" == "$expected_target" ]] ||
      local_e2e_die "refusing unexpected link $link_path -> $(readlink -- "$link_path")"
    return 0
  fi
  [[ ! -e "$link_path" ]] ||
    local_e2e_die "refusing to replace pre-existing non-link: $link_path"
  ln -s -- "$expected_target" "$link_path"
}
