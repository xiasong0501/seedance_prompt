#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/genre_package_audit.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/audit_genre_package.py"
INTERACTIVE_LAUNCHER="${ROOT_DIR}/scripts/interactive_pipeline_launcher.py"
MODE="${1:-interactive}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[genre-package-audit] 配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[genre-package-audit] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  config)
    echo "[genre-package-audit] 使用配置文件中的固定题材运行。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}"
    ;;
  interactive|run)
    echo "[genre-package-audit] 进入交互模式，将从正式题材目录中选择要体检的题材。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode genre_package_audit \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode interactive
    ;;
  preview|dry-run)
    echo "[genre-package-audit] 进入交互预演模式，强制 dry_run=true。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode genre_package_audit \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode preview
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_genre_package_audit.sh           # 交互选择题材并进行体检" >&2
    echo "  ./run_genre_package_audit.sh preview   # 交互选择题材，只做预检不调用模型" >&2
    echo "  ./run_genre_package_audit.sh config    # 不交互，完全按 config/genre_package_audit.local.json 运行" >&2
    exit 2
    ;;
esac
