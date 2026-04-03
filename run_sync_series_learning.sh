#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/sync_series_learning.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/sync_series_learning_to_genres.py"
INTERACTIVE_LAUNCHER="${ROOT_DIR}/scripts/interactive_pipeline_launcher.py"
MODE="${1:-interactive}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[series-learning-sync] 配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[series-learning-sync] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  config)
    echo "[series-learning-sync] 使用配置文件中的固定剧名运行。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}"
    ;;
  interactive|run)
    echo "[series-learning-sync] 进入交互模式，将从 analysis/ 下选择要积累经验的剧。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}"       --mode sync_series_learning       --config "${CONFIG_PATH}"       --runner "${RUNNER_PATH}"       --run-mode interactive
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_sync_series_learning.sh         # 交互选择 analysis 下已有整剧经验的剧" >&2
    echo "  ./run_sync_series_learning.sh run     # 同上，run 为 interactive 别名" >&2
    echo "  ./run_sync_series_learning.sh config  # 不交互，完全按 config/sync_series_learning.local.json 运行" >&2
    exit 2
    ;;
 esac
