#!/usr/bin/env bash
# Exercise every automatable local boundary and print the browser-only segment.
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib/common.sh"

[[ ${1:-} == "--execute" && $# -eq 1 ]] ||
  local_e2e_die "usage: $0 --execute"

for command in curl systemctl; do
  local_e2e_require_command "$command"
done

"$LOCAL_E2E_DIR/project-local-runtime.sh" check
systemctl --user is-active --quiet \
  hyprstream-local-e2e-pds.service \
  hyprstream-local-e2e-inference@0.service \
  hyprstream-local-e2e-inference@1.service

curl_base=(
  --silent
  --show-error
  --cacert "$LOCAL_E2E_STATE_DIR/tls/local-e2e-ca.pem"
  --resolve pds.accounts.localhost:6791:127.0.0.1
)
pds=https://pds.accounts.localhost:6791

curl "${curl_base[@]}" --fail \
  "$pds/.well-known/oauth-authorization-server" >/dev/null

# Protected routes must exist. Missing credentials may produce 400/401/403,
# but a 404 proves the relevant PR surface was not composed into the binary.
status="$(curl "${curl_base[@]}" -o /dev/null -w '%{http_code}' \
  "$pds/xrpc/com.atproto.server.getServiceAuth")"
[[ "$status" != "404" ]] ||
  local_e2e_die "#1372 getServiceAuth route is not mounted"

status="$(curl "${curl_base[@]}" -o /dev/null -w '%{http_code}' \
  -X POST "$pds/api/session/exchange")"
[[ "$status" != "404" ]] ||
  local_e2e_die "#1354 /api/session/exchange route is not mounted"

if [[ -n ${LOCAL_E2E_SESSION_COOKIE:-} ]]; then
  curl "${curl_base[@]}" --fail \
    --cookie "hyprstream_session=$LOCAL_E2E_SESSION_COOKIE" \
    "$pds/api/session/whoami" >/dev/null
  local_e2e_log "browser-created hyprstream_session is accepted by whoami"
fi

cat <<'MANUAL'
AUTOMATED LOCAL SURFACES PASS.

The credential-bearing browser sequence remains manual until www !56 lands:

1. Start the www cold-signup client with AS/PDS origin
   https://pds.accounts.localhost:6791 and trust the generated local CA.
2. Submit PAR, follow /oauth/authorize, and POST action=signup.
3. Exchange the resulting DPoP-bound OAuth access token at
   GET /xrpc/com.atproto.server.getServiceAuth with:
     aud=<the PDS host DID>
     lxm=ai.hyprstream.identity.exchangeSession
4. POST its ATProto service-auth JWT plus DPoP proof to
   /api/session/exchange and retain the hyprstream_session cookie.
5. Re-run this script with LOCAL_E2E_SESSION_COOKIE=<cookie-value> to assert
   /api/session/whoami.
6. Exercise the authenticated inference client against the announced replica
   /moq reach and assert application output.

TODO(www !56 / inference client): no landed non-browser driver currently owns
the DPoP key, signup proof, session cookie, MoQ AEAD subscription, and output
assertion. This script refuses to manufacture those credentials or claim an
inference E2E pass from socket existence alone.
MANUAL

exit 2
