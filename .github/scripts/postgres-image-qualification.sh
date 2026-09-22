#!/usr/bin/env bash
# Run the real-PostgreSQL portion of the staged OCI image profile.
#
# The caller must provide a private URL file for a disposable database.  This
# script deliberately refuses to run without it: the source tests otherwise
# return success after reporting an opt-in skip, which is not qualification.
set -euo pipefail

: "${HYPRSTREAM_POSTGRES_TEST_URL_FILE:?PostgreSQL URL file is required}"
url_file="${HYPRSTREAM_POSTGRES_TEST_URL_FILE}"
[[ -f "${url_file}" && ! -L "${url_file}" && -s "${url_file}" ]] || {
  echo "PostgreSQL qualification requires a nonempty regular URL file" >&2
  exit 1
}

readonly STAGING_POSTGRES_IMAGE_FEATURES='otel,gittorrent,xet,credential-pds-postgres,pds-postgres,rocksdb'

# These test modules each call require_db! and touch the supplied PostgreSQL
# instance. Keep the account-store and PDS record-store coverage separate so a
# module move or an empty filter cannot silently turn this into a green skip.
run_live_module() {
  local filter="$1"
  cargo test -p hyprstream --locked --lib --no-default-features \
    --features "${STAGING_POSTGRES_IMAGE_FEATURES}" -- "${filter}"
}

run_live_module 'auth::postgres_store::tests::'
run_live_module 'services::pds_record_pg::tests::live_'
run_live_module 'services::pds_record_rocksdb::pg_tests::live_'
