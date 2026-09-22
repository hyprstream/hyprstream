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
  local sentinel="$2"
  local log
  local passed
  local cargo_status
  local tee_status
  local -a pipeline_status
  log="$(mktemp "${TMPDIR:-/tmp}/hyprstream-postgres-qualification.XXXXXX.log")"
  # The test modules share one disposable database and each initializes or
  # clears its tables. Run each module serially so their test fixtures cannot
  # race schema creation and turn a valid profile into a false failure.
  # Capture the program statuses explicitly. `if ! pipeline` normalizes the
  # failure to one, which otherwise hides an OOM/signal or a cache-permission
  # failure on the native ARM builder. Neither status can contain the URL.
  set +e
  cargo test -p hyprstream --locked --lib --no-default-features \
      --features "${STAGING_POSTGRES_IMAGE_FEATURES}" -- "${filter}" \
      --test-threads=1 --show-output 2>&1 | tee "${log}"
  pipeline_status=("${PIPESTATUS[@]}")
  cargo_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  set -e
  if (( cargo_status != 0 || tee_status != 0 )); then
    rm -f "${log}"
    printf 'PostgreSQL qualification command failed filter=%s cargo_exit=%s tee_exit=%s\n' \
      "${filter}" "${cargo_status}" "${tee_status}" >&2
    return 1
  fi
  if grep -Fq 'HYPRSTREAM_POSTGRES_TEST_URL_FILE unset' "${log}"; then
    rm -f "${log}"
    echo "PostgreSQL qualification skipped ${filter}" >&2
    return 1
  fi
  if ! grep -Fq "test ${sentinel} ... ok" "${log}"; then
    rm -f "${log}"
    echo "PostgreSQL qualification missed required test ${sentinel}" >&2
    return 1
  fi
  passed="$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*$/\1/p' "${log}" | tail -n 1)"
  rm -f "${log}"
  if [[ ! "${passed}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PostgreSQL qualification ran no tests for ${filter}" >&2
    return 1
  fi
  printf 'postgres-qualification filter=%s passed=%s\n' "${filter}" "${passed}"
}

run_live_module 'auth::postgres_store::tests::' \
  'auth::postgres_store::tests::add_list_remove_pubkey'
run_live_module 'services::pds_record_pg::tests::live_' \
  'services::pds_record_pg::tests::live_put_get_roundtrip_and_absent'
run_live_module 'services::pds_record_rocksdb::pg_tests::live_' \
  'services::pds_record_rocksdb::pg_tests::live_two_handle_persistence_and_visibility'
