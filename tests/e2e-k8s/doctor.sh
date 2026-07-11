#!/usr/bin/env bash
# Diagnose host readiness for KIND-in-Podman. Does NOT modify the host — only
# reports and prints the exact sudo fix for each failing check.
# Exit code = number of problems found (0 = ready).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

log "podman: $(podman --version 2>&1 || echo MISSING)"
log "kind:   $(kind version 2>&1 | head -1 || echo MISSING)"
log "kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client 2>&1 | head -1 || echo MISSING)"
echo "---"

problems=0
preflight || problems=$?

echo "---"
if (( problems == 0 )); then
  log "host looks ready for KIND-in-Podman."
else
  warn "${problems} issue(s) found — bring-up will likely fail until fixed (needs sudo)."
fi
exit "$problems"
