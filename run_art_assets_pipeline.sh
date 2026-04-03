#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/art_assets_pipeline.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_art_assets.py" --config "${CONFIG_PATH}"
