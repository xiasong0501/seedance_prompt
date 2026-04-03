#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/style_transfer_assets.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/generate_style_transferred_assets.py"
MODE="${1:-interactive}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[style-transfer] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  config)
    echo "[style-transfer] 使用配置文件中当前的 dry_run 设置运行。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" --mode config
    ;;
  interactive)
    echo "[style-transfer] 进入交互模式，使用配置文件中的当前 dry_run 设置。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" --mode interactive
    ;;
  preview|dry-run)
    echo "[style-transfer] 进入交互预览模式，强制 dry_run=true。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" --mode preview
    ;;
  run)
    echo "[style-transfer] 进入交互正式运行模式，强制 dry_run=false。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" --mode run
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_style_transfer_assets.sh             # 交互选择剧本、ep 与非真实风格，按配置里的 dry_run 运行" >&2
    echo "  ./run_style_transfer_assets.sh preview     # 只预览重命名与风格转换计划" >&2
    echo "  ./run_style_transfer_assets.sh run         # 正式执行人物风格转换" >&2
    echo "  ./run_style_transfer_assets.sh config      # 完全按 config/style_transfer_assets.local.json 运行" >&2
    exit 2
    ;;
esac
