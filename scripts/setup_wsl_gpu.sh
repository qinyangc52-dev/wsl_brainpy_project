#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

VENV_DIR="${ECMM_VENV_DIR:-$HOME/.venvs/ecmm-brainpy}"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements/remote-gpu-cu13.lock
python -m pip install -e . --no-deps

printf '%s\n' "$VENV_DIR" > .wsl-venv-path

export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/check_gpu.py
