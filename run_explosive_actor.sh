#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/explosive_rewrite_pipeline.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/generate_explosive_rewrites.py"
INTERACTIVE_LAUNCHER="${ROOT_DIR}/scripts/interactive_pipeline_launcher.py"
MODE="${1:-interactive}"
COLLECT_METRICS="${2:-}"
EXTRA_ARGS=()

if [[ -n "${COLLECT_METRICS}" ]]; then
  EXTRA_ARGS+=(--collect-metrics "${COLLECT_METRICS}")
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[explosive-actor] 配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[explosive-actor] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  config)
    echo "[explosive-actor] 使用配置文件中当前的 dry_run 设置运行。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
    ;;
  interactive)
    echo "[explosive-actor] 进入交互模式，使用配置文件中的当前 dry_run 设置。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode explosive_rewrite \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode interactive \
      "${EXTRA_ARGS[@]}"
    ;;
  preview|dry-run)
    echo "[explosive-actor] 进入交互预览模式，强制 dry_run=true。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode explosive_rewrite \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode preview \
      "${EXTRA_ARGS[@]}"
    ;;
  run)
    echo "[explosive-actor] 进入交互正式运行模式，强制 dry_run=false。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode explosive_rewrite \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode run \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_explosive_actor.sh             # 交互选择剧本、ep 范围和目标风格，按配置里的 dry_run 运行" >&2
    echo "  ./run_explosive_actor.sh preview     # 交互选择剧本、ep 范围和目标风格，只预览不调用模型" >&2
    echo "  ./run_explosive_actor.sh run         # 交互选择剧本、ep 范围和目标风格，正式改稿" >&2
    echo "  ./run_explosive_actor.sh run true    # 交互正式改稿，并强制开启统计" >&2
    echo "  ./run_explosive_actor.sh run false   # 交互正式改稿，并强制关闭统计" >&2
    echo "  ./run_explosive_actor.sh config      # 不交互，完全按 config/explosive_rewrite_pipeline.local.json 运行" >&2
    echo "  ./run_explosive_actor.sh config true # 不交互，完全按 config 运行，并强制开启统计" >&2
    exit 2
    ;;
esac
