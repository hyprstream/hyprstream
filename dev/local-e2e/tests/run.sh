#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
install -d -m 0700 "$harness_dir/runs"
task_root="$(mktemp -d "$harness_dir/runs/task-1383.XXXXXX")"
chmod 0700 "$task_root"
trap 'rm -rf -- "$task_root"' EXIT
export TMPDIR="$task_root"
export PYTHONDONTWRITEBYTECODE=1

python3 -m unittest discover -s "$harness_dir/tests" -p 'test_*.py' -v
python3 "$harness_dir/causal_harness.py" owned-run \
  --task-root "$task_root" -- \
  python3 -c '
import json
import os
from pathlib import Path

context = json.loads(Path(os.environ["HYPRSTREAM_CAUSAL_CONTEXT"]).read_text())
assert context["contract_version"] == "owned-run-v1"
assert len(context["units"]) == 8
assert len(set(context["units"].values())) == 8
assert len(set(context["held_loopback_tcp_ports"].values())) == 8
for path in context["xdg"].values():
    assert Path(path).is_dir()
'

if find "$task_root" -mindepth 1 -print -quit | grep -q .; then
  printf 'causal-harness tests: owned-run leaked state\n' >&2
  exit 1
fi
printf 'causal-harness tests: PASS\n'
