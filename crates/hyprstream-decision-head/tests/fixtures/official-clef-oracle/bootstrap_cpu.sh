#!/usr/bin/env bash
set -euo pipefail
ORACLE_TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
uv venv --python 3.11 "${ORACLE_TASK_DIR}/.venv"
uv pip install --python "${ORACLE_TASK_DIR}/.venv/bin/python" --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu'
uv pip install --python "${ORACLE_TASK_DIR}/.venv/bin/python" --no-deps -r "${ORACLE_TASK_DIR}/requirements-cpu.lock"
uv pip check --python "${ORACLE_TASK_DIR}/.venv/bin/python"
