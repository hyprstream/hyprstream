#!/usr/bin/env bash
# Keep every independent artifact build on the reviewed staging Postgres shape.
set -euo pipefail

readonly FEATURES='otel,gittorrent,xet,credential-pds-postgres,pds-postgres,rocksdb'
readonly PROFILE_SOURCES=(
  Dockerfile
  .github/scripts/graviton-release-lanes.sh
  .github/workflows/graviton-build-validate.yml
  .github/scripts/postgres-image-qualification.sh
)

for source in "${PROFILE_SOURCES[@]}"; do
  rg -Fqx "${FEATURES}" <(rg -o --no-filename 'otel,gittorrent,xet,credential-pds-postgres,pds-postgres,rocksdb' "${source}") || {
    echo "staging Postgres image profile missing from ${source}" >&2
    exit 1
  }
done

# The production artifact must compile with no defaults and cannot select the
# PGlite-bearing credential-pds feature. The negative gate intentionally tests
# a feature-less failure and is not an artifact producer, so it is excluded.
for source in Dockerfile .github/scripts/graviton-release-lanes.sh .github/workflows/graviton-build-validate.yml; do
  rg -F -- '--no-default-features' "${source}" >/dev/null || {
    echo "${source} can build the artifact with default features" >&2
    exit 1
  }
  if rg -P -- '--features(?:\s+|=)[^#\n]*credential-pds(?!-postgres)' "${source}" >/dev/null; then
    echo "${source} selects the PGlite credential-pds feature" >&2
    exit 1
  fi
done

# Keep the qualification harness's redacted failure diagnostics executable;
# this uses a cargo stub and is not PostgreSQL qualification evidence.
bash .github/scripts/test-postgres-image-qualification.sh
