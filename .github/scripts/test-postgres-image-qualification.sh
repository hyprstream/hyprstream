#!/usr/bin/env bash
# Exercise the qualification shell harness without pretending a stub is DB
# evidence. Its purpose is to keep failure diagnostics redacted and useful.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_root="$(mktemp -d)"
cleanup() {
  unlink "${test_root}/postgres-url" 2>/dev/null || true
  unlink "${test_root}/cargo" 2>/dev/null || true
  unlink "${test_root}/success.log" 2>/dev/null || true
  unlink "${test_root}/failure.log" 2>/dev/null || true
  rmdir "${test_root}" 2>/dev/null || true
}
trap cleanup EXIT

printf '%s\n' 'postgresql://user:private-password@postgres/example' > "${test_root}/postgres-url"
chmod 0600 "${test_root}/postgres-url"

cat > "${test_root}/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case " $* " in
  *' auth::postgres_store::tests:: '*) sentinel='auth::postgres_store::tests::add_list_remove_pubkey' ;;
  *' services::pds_record_pg::tests::live_ '*) sentinel='services::pds_record_pg::tests::live_put_get_roundtrip_and_absent' ;;
  *' services::pds_record_rocksdb::pg_tests::live_ '*) sentinel='services::pds_record_rocksdb::pg_tests::live_two_handle_persistence_and_visibility' ;;
  *) exit 64 ;;
esac
printf 'test %s ... ok\n' "$sentinel"
printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n'
EOF
chmod 0700 "${test_root}/cargo"

PATH="${test_root}:${PATH}" \
HYPRSTREAM_POSTGRES_TEST_URL_FILE="${test_root}/postgres-url" \
bash "${repo_root}/.github/scripts/postgres-image-qualification.sh" > "${test_root}/success.log"
rg -F 'postgres-qualification filter=' "${test_root}/success.log" -c | grep -qx '3'

cat > "${test_root}/cargo" <<'EOF'
#!/usr/bin/env bash
exit 137
EOF
chmod 0700 "${test_root}/cargo"
if PATH="${test_root}:${PATH}" \
    HYPRSTREAM_POSTGRES_TEST_URL_FILE="${test_root}/postgres-url" \
    bash "${repo_root}/.github/scripts/postgres-image-qualification.sh" \
    > "${test_root}/failure.log" 2>&1; then
  echo 'qualification harness accepted a failed cargo command' >&2
  exit 1
fi
rg -F 'cargo_exit=137 tee_exit=0' "${test_root}/failure.log" >/dev/null
! rg -F 'private-password' "${test_root}/failure.log"
