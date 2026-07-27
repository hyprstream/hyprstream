#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

local_e2e_assert_regular_file "$LOCAL_E2E_BINARY"
unit_dir="$LOCAL_E2E_CONFIG_HOME/systemd/user"
install -d -m 0700 "$unit_dir"

escape_sed() {
  printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

for template in "$LOCAL_E2E_DIR"/systemd/*.in; do
  name="$(basename -- "$template" .in)"
  sed \
    -e "s|@@HYPRSTREAM_BIN@@|$(escape_sed "$LOCAL_E2E_BINARY")|g" \
    -e "s|@@CONFIG_DIR@@|$(escape_sed "$LOCAL_E2E_CONFIG_DIR")|g" \
    -e "s|@@RUNTIME_DIR@@|$(escape_sed "$LOCAL_E2E_RUNTIME_DIR")|g" \
    -e "s|@@HARNESS_DIR@@|$(escape_sed "$LOCAL_E2E_DIR")|g" \
    "$template" >"$unit_dir/$name"
  chmod 0600 "$unit_dir/$name"
  ! rg -q '@@[A-Z_]+@@' "$unit_dir/$name" ||
    local_e2e_die "unresolved unit placeholder in $name"
done

local_e2e_log "user units installed under $unit_dir (daemon-reload not run)"
