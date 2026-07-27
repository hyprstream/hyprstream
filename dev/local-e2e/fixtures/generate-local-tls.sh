#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)/lib/common.sh"

for command in openssl install; do
  local_e2e_require_command "$command"
done

tls_dir="$LOCAL_E2E_STATE_DIR/tls"
install -d -m 0700 "$LOCAL_E2E_STATE_DIR" "$tls_dir"
local_e2e_assert_real_dir "$tls_dir"

if [[ ! -f "$tls_dir/local-e2e-ca.key" || ! -f "$tls_dir/local-e2e-ca.pem" ]]; then
  openssl req -x509 -newkey ec \
    -pkeyopt ec_paramgen_curve:P-256 \
    -sha256 -nodes -days 7 \
    -subj '/CN=Hyprstream local E2E development CA' \
    -keyout "$tls_dir/local-e2e-ca.key" \
    -out "$tls_dir/local-e2e-ca.pem"
fi

openssl req -newkey ec \
  -pkeyopt ec_paramgen_curve:P-256 \
  -sha256 -nodes \
  -subj '/CN=pds.accounts.localhost' \
  -keyout "$tls_dir/quic-key.pem" \
  -out "$tls_dir/quic.csr.pem"

openssl x509 -req -sha256 -days 2 \
  -in "$tls_dir/quic.csr.pem" \
  -CA "$tls_dir/local-e2e-ca.pem" \
  -CAkey "$tls_dir/local-e2e-ca.key" \
  -CAcreateserial \
  -extfile "$LOCAL_E2E_DIR/fixtures/openssl-accounts.ext" \
  -out "$tls_dir/quic-leaf.pem"
{
  sed -n '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/p' \
    "$tls_dir/quic-leaf.pem"
  sed -n '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/p' \
    "$tls_dir/local-e2e-ca.pem"
} >"$tls_dir/quic-chain.pem"

chmod 0600 "$tls_dir/local-e2e-ca.key" "$tls_dir/quic-key.pem"
chmod 0644 "$tls_dir/local-e2e-ca.pem" "$tls_dir/quic-chain.pem"
openssl verify -CAfile "$tls_dir/local-e2e-ca.pem" "$tls_dir/quic-leaf.pem"
openssl x509 -in "$tls_dir/quic-leaf.pem" -noout -checkhost pds.accounts.localhost
openssl x509 -in "$tls_dir/quic-leaf.pem" -noout -checkhost alice.accounts.localhost

local_e2e_log "development-only account-zone TLS fixture generated under $tls_dir"
