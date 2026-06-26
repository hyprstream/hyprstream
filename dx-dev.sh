#!/bin/bash
# dx-dev.sh — Start hyprstream with subsecond hot-patching
#
# Usage: ./dx-dev.sh
#
# Change any .rs file → ~130ms hot-patch → running services updated.
# No AppImage rebuild, no service restart.

set -euo pipefail

TORCH_LIB="$(python3 -c 'import torch; print(torch.__path__[0])')/lib"

exec env \
  OPENSSL_NO_VENDOR=1 \
  LIBTORCH_USE_PYTORCH=1 \
  LIBTORCH_BYPASS_VERSION_CHECK=1 \
  LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
  dx serve --hotpatch --server \
    --package hyprstream --bin hyprstream --open false \
    --args "service start --foreground --services registry,policy,model,streams,worker,mcp,notification,metrics,oauth,event,discovery"
