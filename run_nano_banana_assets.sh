#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/nano_banana_assets.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/generate_nano_banana_assets.py"
INTERACTIVE_LAUNCHER="${ROOT_DIR}/scripts/interactive_pipeline_launcher.py"
MODE="${1:-interactive}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[nano-banana] 配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[nano-banana] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  config)
    echo "[nano-banana] 使用配置文件中当前的 dry_run 设置运行。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}"
    ;;
  interactive)
    echo "[nano-banana] 进入交互模式，使用配置文件中的当前 dry_run 设置。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode nano_banana \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode interactive
    ;;
  preview|dry-run)
    echo "[nano-banana] 进入交互预览模式，强制 dry_run=true。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode nano_banana \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode preview
    ;;
  run)
    echo "[nano-banana] 进入交互正式运行模式，强制 dry_run=false。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode nano_banana \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode run
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_nano_banana_assets.sh             # 交互选择剧本和 ep 范围，按配置里的 dry_run 运行" >&2
    echo "  ./run_nano_banana_assets.sh preview     # 交互选择剧本和 ep 范围，只预览不调用模型" >&2
    echo "  ./run_nano_banana_assets.sh run         # 交互选择剧本和 ep 范围，正式运行" >&2
    echo "  ./run_nano_banana_assets.sh config      # 不交互，完全按 config/nano_banana_assets.local.json 运行" >&2
    exit 2
    ;;
esac
