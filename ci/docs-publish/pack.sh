#!/usr/bin/env bash
# Pack a built rustdoc tree into a content-addressed CAR for IPFS handoff.
#
# usage: ci/docs-publish/pack.sh <build-dir>
#
# Packs <build-dir> into ipfs/site.car with the locked ipfs-car@3.1.0 binary
# (installed from the committed lockfile by `npm ci --ignore-scripts` in the
# workflow; the packer is not selectable from the environment), derives the
# root CID locally from those bytes, and writes
# ipfs/{site.car,cid.txt,dnslink-value.txt,manifest.json}.
#
# This script makes NO network call and holds NO credential. It never pins
# and never touches DNS: the promoter (cyberdione/ingest) fetches the CAR,
# independently re-derives the CID from the CAR bytes, pins, and only then
# promotes DNSLink. The manifest written here is a claim, not a proof; the
# promoter treats it as one. Mirrors cyberdione-corp's ci/ipfs-publish*.sh
# (option B) in a single minimal script.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${1:-}"
[[ -n "$BUILD_DIR" && -z "${2:-}" ]] || {
  echo "usage: ci/docs-publish/pack.sh <build-dir>" >&2
  exit 2
}
[[ -d "$BUILD_DIR" ]] || { echo "ERROR: build dir '$BUILD_DIR' does not exist" >&2; exit 1; }
# An empty tree packs to a valid empty-directory CID that downstream would
# happily pin and publish as a blank site. Refuse here instead.
[[ -n "$(find "$BUILD_DIR" -type f -print -quit)" ]] || {
  echo "ERROR: build dir '$BUILD_DIR' contains no files" >&2
  exit 1
}

IPFS_CAR_VERSION="3.1.0"
IPFS_CAR_BIN="$SCRIPT_DIR/node_modules/.bin/ipfs-car"
[[ -x "$IPFS_CAR_BIN" ]] || {
  echo "ERROR: locked ipfs-car binary missing; run: npm ci --ignore-scripts --prefix ci/docs-publish" >&2
  exit 1
}

OUT_DIR="ipfs"
CAR_PATH="$OUT_DIR/site.car"
mkdir -p "$OUT_DIR"

echo "Packing '$BUILD_DIR' with locked ipfs-car@$IPFS_CAR_VERSION ..."
# ipfs-car is deterministic: the same tree yields a byte-identical CAR and the
# same root CID on any machine, which is what lets the promoter re-derive the
# CID and fail closed on any mismatch.
CID="$("$IPFS_CAR_BIN" pack "$BUILD_DIR" --output "$CAR_PATH" | tail -n 1 | tr -d '[:space:]')"

# ipfs-car prints the root CID as its last stdout line. Fail here rather than
# ship a garbage identifier downstream.
case "$CID" in
  baf*) ;;
  *) echo "ERROR: ipfs-car did not return a CIDv1 root (got: '$CID')" >&2; exit 1 ;;
esac

CAR_SHA256="$(sha256sum "$CAR_PATH" | cut -d' ' -f1)"
echo "Root CID:   $CID"
echo "CAR sha256: $CAR_SHA256"

DNSLINK_VALUE="dnslink=/ipfs/$CID"
printf '%s\n' "$CID" > "$OUT_DIR/cid.txt"
printf '%s\n' "$DNSLINK_VALUE" > "$OUT_DIR/dnslink-value.txt"

node "$SCRIPT_DIR/write-manifest.mjs" "$OUT_DIR/manifest.json" \
  "$CID" "$CAR_SHA256" "$IPFS_CAR_VERSION" "$BUILD_DIR" "$DNSLINK_VALUE" \
  "${GITHUB_SHA:-unknown}" "${GITHUB_REF_NAME:-unknown}" \
  "${GITHUB_RUN_ID:-unknown}" "${GITHUB_RUN_ATTEMPT:-unknown}"

echo "Wrote $OUT_DIR/{site.car,cid.txt,dnslink-value.txt,manifest.json}"
