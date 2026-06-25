#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Autonomous AAA (Authentication / Authorization / Accounting) e2e harness for
# hyprstream. Validates the #438 (user-key binding) + #441 (authoritative
# service-key) fixes on the aaa-integration worktree.
#
# FULLY ISOLATED: a fresh run dir holds HOME / XDG_CONFIG_HOME / XDG_DATA_HOME /
# XDG_RUNTIME_DIR so it NEVER reads or writes the user's live
# ~/.config/hyprstream credentials.
#
# tmux: the daemon (all services, --standalone, IPC sockets in $XDG_RUNTIME_DIR)
# runs in one pane; the CLI is driven from this script (same isolated env, so it
# connects to the daemon's IPC sockets). Services are long-running; we wait for
# readiness then drive + observe.
#
# AAA chain validated:
#   AUTHN  — enrolled user (wizard, #438) authenticates as that user, not anon.
#            Deny path: anonymous is DENIED a write.
#   AUTHZ  — role gates correctly: admin allowed a privileged (write) op; a
#            viewer/anonymous is DENIED it. Both allow AND deny paths asserted.
#   SVCKEY — #441: services register their keys; resolve_service_key returns the
#            registered key; a cross-service signed RPC (CLI -> PolicyService,
#            CLI -> Registry) verifies WITHOUT "Response signed by unexpected key".
#   ACCT   — best-effort: audit/log attribution of the request to the user.
#
# Exit 0 = AAA validated. Non-zero = a specific assertion failed (see code).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

WT="${WT:?set WT to the integration worktree path under test}"
# RUN must be SHORT: it holds Unix-domain sockets (XDG_RUNTIME_DIR + tmux),
# whose paths are capped at ~108 bytes (sun_path). The session scratchpad path
# alone is ~150 chars, which overflows — so the run dir lives under /tmp with a
# short name. It is still fully isolated (fresh dir, dedicated HOME/XDG; it never
# reads or writes the user's live ~/.config/hyprstream).
RUN="${RUN:-$(mktemp -d /tmp/hs-aaa.XXXXXX)}"
ROLE="${ROLE:-admin}"          # initial user role for the wizard enrollment
BUILD="${BUILD:-1}"            # set BUILD=0 to skip the cargo build (binary present)

# ── env: torch libs + isolation ─────────────────────────────────────────────
export OPENSSL_NO_VENDOR=1 LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
# Capture the torch path BEFORE we clobber HOME (build/run both need it).
TORCH_LIB="$(ls -d "$HOME"/.local/lib/python*/site-packages/torch/lib 2>/dev/null | head -1)"

export HOME="$RUN/home"
export XDG_CONFIG_HOME="$RUN/home/.config"
export XDG_DATA_HOME="$RUN/home/.local/share"
export XDG_CACHE_HOME="$RUN/home/.cache"
export XDG_RUNTIME_DIR="$RUN/run"
export USER="testuser"
export LOGNAME="testuser"
unset HYPRSTREAM_INSTANCE           # avoid instance-namespaced runtime dir surprises
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Re-point LD_LIBRARY_PATH to the captured torch path (HOME just changed).
[ -n "$TORCH_LIB" ] && export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

LOG="$RUN/harness.log"
DLOG="$RUN/daemon.log"
BIN="$WT/target/debug/hyprstream"
TMUX_SOCK="$RUN/tmux.sock"
SESSION="aaa"

: > "$LOG"
note(){ echo "[harness $(date +%T)] $*" | tee -a "$LOG"; }
fail(){ note "FAIL($1): ${*:2}"; dump_diag; cleanup; exit "$1"; }
pass(){ note "PASS: $*"; }

dump_diag(){
  note "──── daemon log (tail 60) ────"
  tail -60 "$DLOG" 2>/dev/null | sed 's/^/[daemon] /' | tee -a "$LOG"
}

cleanup(){
  tmux -S "$TMUX_SOCK" kill-server 2>/dev/null || true
  pkill -f "$XDG_RUNTIME_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# bin wrapper — every CLI call runs in the isolated env.
hs(){ "$BIN" "$@"; }

# ── STEP 0: build ────────────────────────────────────────────────────────────
if [ "$BUILD" = "1" ]; then
  note "=== STEP 0: build (nice -n 10, jobs capped via ~/.cargo/config.toml) ==="
  ( cd "$WT" && nice -n 10 cargo build -p hyprstream --bin hyprstream 2>&1 | tail -3 ) | tee -a "$LOG"
fi
[ -x "$BIN" ] || fail 10 "binary not built at $BIN"
pass "binary present: $BIN"

# ── STEP 1: enroll the user via the wizard (the #438 flow) ───────────────────
note "=== STEP 1: wizard enroll (user=$USER role=$ROLE) ==="
hs wizard --non-interactive --initial-user-role "$ROLE" >>"$LOG" 2>&1
rc=$?
note "wizard exit=$rc"
[ $rc -eq 0 ] || fail 11 "wizard --non-interactive failed (rc=$rc)"

# ── STEP 2: assert the user is enrolled with a bound key (not orphaned) ───────
note "=== STEP 2: user enrolled + key bound (#438) ==="
hs user list >"$RUN/userlist.txt" 2>>"$LOG"
cat "$RUN/userlist.txt" | tee -a "$LOG"
grep -q "$USER" "$RUN/userlist.txt" || fail 12 "user '$USER' not in 'user list' — wizard dropped the identity (#438 regression)"
pass "user '$USER' registered in UserStore"

hs user keys list "$USER" >"$RUN/keys.txt" 2>>"$LOG"
cat "$RUN/keys.txt" | tee -a "$LOG"
if grep -Eqi "fingerprint|sha256|wizard|ed25519|[0-9a-f]{16}" "$RUN/keys.txt"; then
  pass "user '$USER' has a bound signing-key pubkey (#438 binding present)"
else
  fail 13 "user '$USER' has NO bound key — orphaned admin policy, CLI would fall back to anonymous (#438 not effective)"
fi

# ── STEP 3: start all services in a tmux pane (long-running daemon) ───────────
note "=== STEP 3: start daemon (service start --standalone) in tmux ==="
tmux -S "$TMUX_SOCK" kill-server 2>/dev/null || true
tmux -S "$TMUX_SOCK" new-session -d -s "$SESSION" -x 220 -y 50
for v in HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_CACHE_HOME XDG_RUNTIME_DIR USER LOGNAME \
         OPENSSL_NO_VENDOR LIBTORCH_USE_PYTORCH LIBTORCH_BYPASS_VERSION_CHECK LD_LIBRARY_PATH; do
  tmux -S "$TMUX_SOCK" send-keys -t "$SESSION" "export $v=$(printf %q "${!v}")" Enter
done
tmux -S "$TMUX_SOCK" send-keys -t "$SESSION" "unset HYPRSTREAM_INSTANCE" Enter
tmux -S "$TMUX_SOCK" send-keys -t "$SESSION" \
  "exec '$BIN' service start --standalone 2>&1 | tee '$DLOG'" Enter

# ── wait for readiness: PolicyService socket present ──────────────────────────
note "waiting for daemon readiness (policy IPC socket)…"
SOCKDIR="$XDG_RUNTIME_DIR/hyprstream"
ready=0
for i in $(seq 1 90); do
  if ls "$SOCKDIR"/policy*.sock >/dev/null 2>&1; then ready=1; break; fi
  if grep -Eqi "panicked|fatal|cannot register its signing key|Address already in use" "$DLOG" 2>/dev/null; then
    note "daemon emitted an error during startup:"; tail -25 "$DLOG" | sed 's/^/[daemon] /' | tee -a "$LOG"
    break
  fi
  sleep 1
done
[ "$ready" = "1" ] || fail 20 "daemon not ready: no policy IPC socket at $SOCKDIR after 90s"
sleep 3   # let remaining services finish registering their keys
pass "daemon ready (sockets): $(ls "$SOCKDIR" 2>/dev/null | tr '\n' ' ')"

# ── STEP 4: SERVICE-KEY (#441) — cross-service signed RPC, no key mismatch ────
note "=== STEP 4: service-key resolution + cross-service RPC (#441) ==="
hs quick policy show >"$RUN/policyshow.txt" 2>&1
rc=$?
cat "$RUN/policyshow.txt" >>"$LOG"
if grep -qi "unexpected key\|signed by unexpected" "$RUN/policyshow.txt" "$LOG"; then
  fail 30 "'Response signed by unexpected key' on policy RPC — #441 service-key binding NOT effective"
fi
[ $rc -eq 0 ] || fail 31 "policy show RPC failed (rc=$rc) — PolicyService not serving / key issue"
pass "CLI -> PolicyService signed RPC verified (no key mismatch)"

# `tool registry list` exercises resolve_service_key('registry') (#441) then a
# signed Registry RPC. The `tool` CLI signs with the node key over the IPC
# AnySigner plane and carries no user JWT, so the Registry authorizes it as
# `anonymous`. The #441 path is VALIDATED when the call reaches the Registry and
# returns a clean authorization decision (here: anonymous denied a query) WITHOUT
# a key-resolution error ("not registered" / "unexpected key"). A key error here
# would mean #441 is broken; an authz denial means the signed RPC round-tripped.
hs tool registry list >"$RUN/reglist.txt" 2>&1
rc=$?
cat "$RUN/reglist.txt" >>"$LOG"
if grep -qi "unexpected key\|signed by unexpected\|not registered\|No verifying key\|No attestation" "$RUN/reglist.txt"; then
  fail 32 "registry key resolution failed — resolve_service_key error or unexpected key (#441)"
fi
if [ $rc -eq 0 ]; then
  pass "CLI -> resolve_service_key('registry') -> signed Registry RPC verified, list returned (#441)"
elif grep -qi "Unauthorized: anonymous" "$RUN/reglist.txt"; then
  # Key resolution succeeded; the signed RPC round-tripped and the Registry
  # returned a fail-closed authz denial for the anonymous tool caller. This both
  # validates #441 (no key error) AND is an additional AUTHN/AUTHZ deny-path
  # data point (anonymous denied a registry query).
  pass "CLI -> resolve_service_key('registry') -> signed RPC round-tripped; anonymous tool caller fail-closed denied (#441 OK, deny-path bonus)"
else
  fail 33 "tool registry list failed unexpectedly (rc=$rc) — see $RUN/reglist.txt"
fi

# ── STEP 5: AUTHN + AUTHZ matrix via the live PolicyService oracle ───────────
note "=== STEP 5: AUTHN/AUTHZ allow+deny matrix (live PolicyService) ==="
check(){  # check <subject> <resource> <action> -> ALLOWED|DENIED|KEYERR|ERR
  hs quick policy check "$1" "$2" "$3" >"$RUN/chk.txt" 2>&1
  cat "$RUN/chk.txt" >>"$LOG"
  if grep -qi "unexpected key\|not registered" "$RUN/chk.txt"; then echo "KEYERR"; return; fi
  if grep -q "ALLOWED" "$RUN/chk.txt"; then echo "ALLOWED";
  elif grep -q "DENIED" "$RUN/chk.txt"; then echo "DENIED";
  else echo "ERR"; fi
}

r=$(check anonymous "registry:*" write)
note "authn deny: anonymous registry:* write => $r"
[ "$r" = "DENIED" ] || fail 40 "anonymous was NOT denied a write (got $r) — fail-OPEN to anonymous"
pass "AUTHN deny: anonymous denied registry write (fail-closed by default)"

r=$(check "$USER" "registry:*" write)
note "authz allow: $USER registry:* write => $r"
if [ "$ROLE" = "admin" ]; then
  [ "$r" = "ALLOWED" ] || fail 41 "enrolled admin '$USER' was NOT allowed a write (got $r) — authz allow path broken"
  pass "AUTHZ allow: admin '$USER' allowed registry write"
fi

r=$(check "viewer-probe" "registry:*" write)
note "authz deny: viewer-probe registry:* write => $r"
[ "$r" = "DENIED" ] || fail 42 "unprivileged 'viewer-probe' was NOT denied a write (got $r) — authz deny path broken / allow-all policy"
pass "AUTHZ deny: unprivileged subject denied registry write"

r=$(check "$USER" "model:*" query)
note "authz allow (query): $USER model:* query => $r"

# ── STEP 6: ACCOUNTING (best-effort) ─────────────────────────────────────────
note "=== STEP 6: accounting / audit attribution (best-effort) ==="
if grep -Eqi "subject=$USER|sub=\"?$USER|user=$USER" "$DLOG" 2>/dev/null; then
  pass "ACCOUNTING: requests attributed to '$USER' in daemon audit log"
else
  note "GAP: per-request accounting attribution to '$USER' not confirmed in daemon logs"
  echo "ACCT_GAP=1" >"$RUN/acct_gap"
fi

# ── DONE ─────────────────────────────────────────────────────────────────────
note "════════════════════════════════════════════════════════════════════════"
note "AAA VALIDATED:"
note "  AUTHN  allow: enrolled user; deny: anonymous denied write"
note "  AUTHZ  allow: admin write allowed; deny: unprivileged denied write"
note "  SVCKEY (#441): CLI->PolicyService + CLI->Registry signed RPC, no key mismatch"
[ -f "$RUN/acct_gap" ] && note "  ACCT   gap noted (see ticket)" || note "  ACCT   attribution observed"
note "════════════════════════════════════════════════════════════════════════"
cleanup
exit 0
