#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/video_pipeline.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

# if [ ! -x "${PYTHON_BIN}" ]; then
#   echo "未检测到 ${PYTHON_BIN}"
#   echo "请先运行: ${ROOT_DIR}/scripts/install_video_pipeline.sh"
#   exit 1
# fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_video_pipeline.py" --config "${CONFIG_PATH}"
