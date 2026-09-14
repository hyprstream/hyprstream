#!/usr/bin/env bash
# Verify shared-builder/AppImage headers and libraries before Cargo runs.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/appimage/lib.sh"
expected=$(sed -n 's/^ARG LIBTORCH_VERSION=\([0-9][0-9.]*\)$/\1/p' "$root/Dockerfile")
require_libtorch_version "${1:-${LIBTORCH:-/opt/libtorch}}" "$expected"
if [[ "${2:-}" == --cpu-runtime ]]; then
    python3 - "$expected" <<'PYTHON'
import platform
import sys
import torch
assert torch.__version__.split('+')[0] == sys.argv[1], torch.__version__
assert platform.machine() == 'aarch64', platform.machine()
assert torch.version.cuda is None and torch.version.hip is None
assert torch.mm(torch.ones(2, 2), torch.ones(2, 2)).sum().item() == 8
print(f'ARM CPU libtorch runtime verified: {torch.__version__}')
PYTHON
fi
