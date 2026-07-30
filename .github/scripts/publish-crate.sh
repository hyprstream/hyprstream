#!/usr/bin/env bash
# Publish one pre-validated crate idempotently, then prove the local package is
# byte-identical to crates.io. This script is only invoked by publish-crates.yml.
set -euo pipefail

crate="${1:?crate name required}"
version="${2:?version required}"
archive="target/package/${crate}-${version}.crate"
api="https://crates.io/api/v1/crates/${crate}/${version}"
user_agent="hyprstream-release-workflow/1.0 (https://github.com/hyprstream/hyprstream)"

package_ready=false
for attempt in {1..30}; do
  if cargo package --locked -p "$crate"; then
    package_ready=true
    break
  fi
  # A dependent crate can become packageable only after the preceding crate's
  # sparse-index entry propagates. Never bypass Cargo's packaged-source build.
  [[ "$attempt" -lt 30 ]] || break
  sleep 10
done
[[ "$package_ready" == true ]] || {
  echo "::error::failed to package and verify ${crate}@${version}"
  exit 1
}

# Cargo's registry limit is 10 MiB. Keep a local content inventory beside the
# package so the attested evidence shows exactly what was shipped.
size="$(stat --format=%s "$archive")"
[[ "$size" -le 10485760 ]] || {
  echo "::error::${archive} is ${size} bytes (crates.io limit: 10485760)"
  exit 1
}
cargo package --locked -p "$crate" --list > "target/package/${crate}-${version}.files"
local_checksum="$(sha256sum "$archive" | cut -d' ' -f1)"

registry_checksum() {
  curl --fail --silent --show-error --retry 3 \
    -A "$user_agent" "$api" | jq -r '.version.checksum // empty'
}

status="$(curl --silent --output /dev/null --write-out '%{http_code}' -A "$user_agent" "$api")"
if [[ "$status" == 200 ]]; then
  remote_checksum="$(registry_checksum)"
  [[ "$remote_checksum" == "$local_checksum" ]] || {
    echo "::error::${crate}@${version} exists with checksum ${remote_checksum}, expected ${local_checksum}"
    exit 1
  }
  echo "::notice::${crate}@${version} already exists with the expected checksum"
  exit 0
fi
[[ "$status" == 404 ]] || {
  echo "::error::unexpected crates.io status ${status} for ${crate}@${version}"
  exit 1
}

: "${CRATES_IO_TOKEN:?trusted-publisher token required}"
cargo publish --locked --registry crates-io --token "$CRATES_IO_TOKEN" -p "$crate"

for attempt in {1..30}; do
  status="$(curl --silent --output /dev/null --write-out '%{http_code}' -A "$user_agent" "$api")"
  if [[ "$status" == 200 ]]; then
    remote_checksum="$(registry_checksum)"
    [[ "$remote_checksum" == "$local_checksum" ]] || {
      echo "::error::published checksum ${remote_checksum} differs from package ${local_checksum}"
      exit 1
    }
    exit 0
  fi
  sleep 10
done

echo "::error::${crate}@${version} did not become visible on crates.io"
exit 1
