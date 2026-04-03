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

"$PYTHON_BIN" scripts/run_seedance_api_batch.py "$@"
