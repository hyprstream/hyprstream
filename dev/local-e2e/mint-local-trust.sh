#!/usr/bin/env bash
# Mint a throwaway deployment authority, delegate an online registry signer,
# and mint the <=1 hour registry deployment JWT using PR #1371's exact CLI.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

force=0
if [[ ${1:-} == "--force" ]]; then
  force=1
  shift
fi
[[ $# -eq 0 ]] || local_e2e_die "usage: $0 [--force]"

for command in age age-keygen openssl xxd dd install stat; do
  local_e2e_require_command "$command"
done
local_e2e_assert_regular_file "$LOCAL_E2E_BINARY"

trust_dir="$LOCAL_E2E_STATE_DIR/trust"
identity_dir="$LOCAL_E2E_PRIVATE_DIR/age-identities"
bootstrap_dir="$LOCAL_E2E_STATE_DIR/bootstrap-credentials"
registry_seed="$bootstrap_dir/registry/signing-key"
registry_public="$trust_dir/registry-public-key.ed25519"

install -d -m 0700 "$LOCAL_E2E_STATE_DIR" "$trust_dir" \
  "$LOCAL_E2E_PRIVATE_DIR" "$identity_dir"
local_e2e_assert_real_dir "$trust_dir"
local_e2e_assert_real_dir "$identity_dir"
local_e2e_assert_regular_file "$registry_seed"
[[ "$(stat -c %s -- "$registry_seed")" == "32" ]] ||
  local_e2e_die "registry signing-key must be a raw 32-byte Ed25519 seed"

generate_age_identity() {
  local destination=$1
  [[ -e "$destination" ]] && return 0
  age-keygen -o "$destination"
  chmod 0600 "$destination"
}

generate_age_identity "$identity_dir/root-1.agekey"
generate_age_identity "$identity_dir/root-2.agekey"
generate_age_identity "$identity_dir/online.agekey"

root_recipient_1="$(age-keygen -y "$identity_dir/root-1.agekey")"
root_recipient_2="$(age-keygen -y "$identity_dir/root-2.agekey")"
online_recipient="$(age-keygen -y "$identity_dir/online.agekey")"
[[ "$root_recipient_1" != "$root_recipient_2" ]] ||
  local_e2e_die "root recipient ring members must be distinct"

# PKCS#8 OneAsymmetricKey prefix for a raw Ed25519 seed. OpenSSL emits an
# RFC 8410 SubjectPublicKeyInfo whose final 32 bytes are the raw public key.
private_der="$trust_dir/registry-private.pk8.der"
public_der="$trust_dir/registry-public.spki.der"
{
  printf '%s' '302e020100300506032b657004220420'
  xxd -p -c 256 "$registry_seed"
} | xxd -r -p >"$private_der"
openssl pkey -inform DER -in "$private_der" -pubout -outform DER -out "$public_der"
dd if="$public_der" of="$registry_public" bs=1 skip=12 count=32 status=none
chmod 0600 "$private_der" "$public_der" "$registry_public"
[[ "$(stat -c %s -- "$registry_public")" == "32" ]] ||
  local_e2e_die "derived registry public key is not 32 raw bytes"

force_arg=()
((force == 1)) && force_arg=(--force)

"$LOCAL_E2E_BINARY" trust mint-deployment-ca \
  --public-ca "$trust_dir/deployment-ca.hybrid" \
  --authority-key "$trust_dir/deployment-ca.age" \
  --authority-log "$trust_dir/deployment-authority.log.json" \
  --authority-checkpoint "$trust_dir/deployment-authority.head.json" \
  --recipient "$root_recipient_1" \
  --recipient "$root_recipient_2" \
  "${force_arg[@]}"

"$LOCAL_E2E_BINARY" trust delegate-registry-signer \
  --public-ca "$trust_dir/deployment-ca.hybrid" \
  --authority-log "$trust_dir/deployment-authority.log.json" \
  --authority-checkpoint "$trust_dir/deployment-authority.head.json" \
  --authority-key "$trust_dir/deployment-ca.age" \
  --identity "$identity_dir/root-1.agekey" \
  --signer-recipient "$online_recipient" \
  --delegated-key "$trust_dir/registry-delegated-signer.age" \
  --delegation "$trust_dir/registry-signer.delegation.json" \
  --delegation-ttl-seconds 2592000 \
  "${force_arg[@]}"

"$LOCAL_E2E_BINARY" trust mint-registry-jwt \
  --public-ca "$trust_dir/deployment-ca.hybrid" \
  --authority-log "$trust_dir/deployment-authority.log.json" \
  --authority-checkpoint "$trust_dir/deployment-authority.head.json" \
  --identity "$identity_dir/online.agekey" \
  --via-delegated-signer "$trust_dir/registry-delegated-signer.age" \
  --delegation "$trust_dir/registry-signer.delegation.json" \
  --registry-public-key "$registry_public" \
  --ttl-seconds 3600 \
  --jwt "$trust_dir/registry-service.jwt" \
  --contract "$trust_dir/deployment-trust.contract.json" \
  "${force_arg[@]}"

"$LOCAL_E2E_BINARY" trust verify-deployment \
  --public-ca "$trust_dir/deployment-ca.hybrid" \
  --authority-log "$trust_dir/deployment-authority.log.json" \
  --authority-checkpoint "$trust_dir/deployment-authority.head.json" \
  --jwt "$trust_dir/registry-service.jwt" \
  --contract "$trust_dir/deployment-trust.contract.json"

[[ "$(stat -c %s -- "$trust_dir/deployment-ca.hybrid")" == "1984" ]] ||
  local_e2e_die "deployment-ca.hybrid is not exactly 1984 bytes"
[[ "$(awk -F. 'END { print NF }' "$trust_dir/registry-service.jwt")" == "3" ]] ||
  local_e2e_die "registry-service.jwt is not compact JWT form"

local_e2e_log "local deployment trust minted and verified under $trust_dir"
