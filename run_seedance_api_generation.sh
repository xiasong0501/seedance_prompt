#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export SEEDANCE_RESOLUTION="${SEEDANCE_RESOLUTION:-480p}"
export SEEDANCE_REFERENCE_MODE_DEFAULT="${SEEDANCE_REFERENCE_MODE_DEFAULT:-asset}"

"$PYTHON_BIN" scripts/generate_seedance_api_script.py
