#!/usr/bin/env bash
# Idempotent: Python 3.12 venv in repo root for old glibc (e.g. Ubuntu 18.04) via uv.
#   wget -qO- https://astral.sh/uv/install.sh | sh
#   ./scripts/bootstrap_linux_aux_venv.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
. "$HOME/.local/bin/env"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: wget -qO- https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
uv venv .venv --python 3.12
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install -r requirements.txt
# CPU-only (no NVIDIA driver): avoid huge CUDA stack if PyPI picked GPU wheels; safe to re-run.
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "OK: . $(pwd)/.venv/bin/activate && python tools/_train_env_check.py"
