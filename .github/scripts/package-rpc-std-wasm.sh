#!/usr/bin/env bash
# Build and package the public browser SDK exactly once. The resulting tarball is
# the only object handed to a publishing job; publishing never rebuilds it.
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly CRATE_DIR="$ROOT/crates/hyprstream-rpc-std"
readonly WASM_PACK_PACKAGE_NAME="hyprstream-rpc-std"
readonly PACKAGE_NAME="@hyprstream/rpc"
readonly RELEASE_TAG_PREFIX="hyprstream-rpc"
readonly REGISTRY="https://registry.npmjs.org"
readonly CHANNEL="${PACKAGE_CHANNEL:-ci}"
readonly SHA="${GITHUB_SHA:-$(git -C "$ROOT" rev-parse HEAD)}"
readonly SHORT_SHA="${SHA:0:12}"
# Commit count reachable from HEAD. Used for the staging version so that the
# same main commit ALWAYS yields the same staging version — independent of the
# CI run number — letting an operator pre-state and verify the exact identity
# before authorizing a publish. Full history (checkout fetch-depth: 0) is
# required for this to be exact; a shallow clone makes this fail closed.
readonly COMMIT_COUNT="$(git -C "$ROOT" rev-list --count HEAD 2>/dev/null || echo 0)"
if [[ "$COMMIT_COUNT" == "0" && "$CHANNEL" == "staging" ]]; then
  echo "staging version needs full git history (commit count); refusing to publish a floating version" >&2
  exit 1
fi

crate_version="$({
  python3 - "$CRATE_DIR/Cargo.toml" <<'PY'
import pathlib
import sys
import tomllib
print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["package"]["version"])
PY
})"

case "$CHANNEL" in
  production)
    expected_ref="refs/tags/${RELEASE_TAG_PREFIX}-v${crate_version}"
    if [[ "${GITHUB_REF:-}" != "$expected_ref" ]]; then
      echo "production requires exact immutable release tag $expected_ref" >&2
      exit 1
    fi
    package_version="$crate_version"
    ;;
  staging)
    if [[ "${GITHUB_REF:-}" != "refs/heads/main" ]]; then
      echo "staging publication is restricted to refs/heads/main" >&2
      exit 1
    fi
    # Commit-deterministic: 0.1.0-dev.<commit-count>.<short-sha>. The same
    # main commit reproduces this exact version across independent CI runs, so
    # the staging version is not tied to a particular run number.
    package_version="${crate_version}-dev.${COMMIT_COUNT}.${SHORT_SHA}"
    ;;
  ci)
    # Throwaway PR channel: never published. Commit-derived so it does not
    # depend on the CI run number either.
    package_version="${crate_version}-ci.${COMMIT_COUNT}.${SHORT_SHA}"
    ;;
  *)
    echo "unknown PACKAGE_CHANNEL: $CHANNEL" >&2
    exit 1
    ;;
esac

readonly PKG_DIR="$ROOT/target/wasm-pack/hyprstream-rpc-std"
readonly DIST_DIR="$ROOT/dist/hyprstream-rpc-std-wasm"
rm -rf "$PKG_DIR" "$DIST_DIR"
mkdir -p "$PKG_DIR" "$DIST_DIR"

append_rustflag() {
  local flag="$1"
  if [[ " ${RUSTFLAGS:-} " != *" ${flag} "* ]]; then
    RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }${flag}"
  fi
}
append_rustflag '--cfg=web_sys_unstable_apis'
append_rustflag '--cfg=getrandom_backend="wasm_js"'
export RUSTFLAGS

wasm-pack build \
  --target web \
  --release \
  --out-dir "$PKG_DIR" \
  "$CRATE_DIR" \
  --locked

# wasm-pack derives its name from the Rust package. Validate that generated
# identity before rewriting it to the explicitly authorized public npm scope.
node - "$PKG_DIR/package.json" "$package_version" "$WASM_PACK_PACKAGE_NAME" "$PACKAGE_NAME" "$REGISTRY" <<'NODE'
const fs = require('node:fs');
const [manifestPath, version, wasmPackName, packageName, registry] = process.argv.slice(2);
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
if (manifest.name !== wasmPackName) {
  throw new Error(`wasm-pack package name ${manifest.name} != ${wasmPackName}`);
}
manifest.name = packageName;
manifest.version = version;
manifest.license = 'MIT';
manifest.repository = {
  type: 'git',
  url: 'git+https://github.com/hyprstream/hyprstream.git',
  directory: 'crates/hyprstream-rpc-std',
};
manifest.publishConfig = { access: 'public', provenance: true, registry };
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
NODE

# The pack input is a fresh wasm-pack output directory. Reject symlinks and
# unexpected build/source material before npm applies its own package rules.
if find "$PKG_DIR" -type l -print -quit | grep -q .; then
  echo "wasm-pack output must not contain symlinks" >&2
  exit 1
fi
if find "$PKG_DIR" -type f \
    ! -name 'package.json' ! -name 'README.md' ! -name 'LICENSE*' \
    ! -name '*.js' ! -name '*.d.ts' ! -name '*.wasm' ! -name '.gitignore' \
    -print -quit | grep -q .; then
  echo "wasm-pack output contains an unexpected file" >&2
  find "$PKG_DIR" -type f -print >&2
  exit 1
fi
rm -f "$PKG_DIR/.gitignore"

npm pack "$PKG_DIR" --pack-destination "$DIST_DIR" --json \
  > "$DIST_DIR/npm-pack.json"
mapfile -t tarballs < <(find "$DIST_DIR" -maxdepth 1 -type f -name '*.tgz' -print)
if [[ ${#tarballs[@]} -ne 1 ]]; then
  echo "expected exactly one npm tarball, found ${#tarballs[@]}" >&2
  exit 1
fi
readonly TARBALL="${tarballs[0]}"

# Install from only the tarball in a pristine consumer and bundle a browser
# entrypoint. This catches missing package files and broken ESM/package metadata.
consumer="$(mktemp -d)"
trap 'rm -rf "$consumer"' EXIT
(
  cd "$consumer"
  npm init --yes --scope wasm-smoke >/dev/null
)
npm install --prefix "$consumer" --ignore-scripts --package-lock=false "$TARBALL" >/dev/null
cat > "$consumer/index.mjs" <<EOF
import init from '${PACKAGE_NAME}';
if (typeof init !== 'function') throw new Error('missing wasm-pack default initializer');
export { init };
EOF
(
  cd "$consumer"
  npx --yes --package esbuild@0.25.8 esbuild index.mjs \
    --bundle --format=esm --platform=browser --outfile=bundle.js >/dev/null
  test -s bundle.js
  npm ls --depth=0 "$PACKAGE_NAME" >/dev/null
)

node - "$DIST_DIR/npm-pack.json" "$TARBALL" "$DIST_DIR/integrity.json" \
  "$PACKAGE_NAME" "$package_version" "$CHANNEL" "$REGISTRY" "$SHA" <<'NODE'
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const [packJson, tarball, output, name, version, channel, registry, commit] = process.argv.slice(2);
const packed = JSON.parse(fs.readFileSync(packJson, 'utf8'));
if (!Array.isArray(packed) || packed.length !== 1) throw new Error('npm pack did not return one result');
const bytes = fs.readFileSync(tarball);
const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
const sha512 = crypto.createHash('sha512').update(bytes).digest('base64');
const integrity = `sha512-${sha512}`;
if (packed[0].integrity !== integrity) throw new Error('npm integrity differs from locally computed SHA-512');
const evidence = {
  schema: 1,
  package: name,
  version,
  channel,
  registry,
  tarball: path.basename(tarball),
  sha256,
  integrity,
  npmShasum: packed[0].shasum,
  source: {
    repository: 'https://github.com/hyprstream/hyprstream',
    commit,
    ref: process.env.GITHUB_REF || null,
    workflow: process.env.GITHUB_WORKFLOW_REF || null,
    crate: 'crates/hyprstream-rpc-std',
  },
  build: {
    target: 'web',
    profile: 'release',
    rustToolchain: '1.97.0',
    wasmPack: process.env.WASM_PACK_VERSION || '0.13.1',
  },
  files: packed[0].files.map(({path, size}) => ({path, size})),
};
fs.writeFileSync(output, `${JSON.stringify(evidence, null, 2)}\n`);
NODE

# npm-pack.json contains redundant local details; integrity.json is the stable
# downstream evidence contract shipped alongside the immutable tarball.
rm "$DIST_DIR/npm-pack.json"
printf 'built %s@%s (%s)\n' "$PACKAGE_NAME" "$package_version" "$CHANNEL"
cat "$DIST_DIR/integrity.json"
