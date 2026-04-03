#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/series_pipeline.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/run_series_pipeline.py"
INTERACTIVE_LAUNCHER="${ROOT_DIR}/scripts/interactive_pipeline_launcher.py"
MODE="${1:-interactive}"
COLLECT_METRICS="${2:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[series-pipeline] 配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[series-pipeline] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

case "${MODE}" in
  config)
    echo "[series-pipeline] 使用配置文件中当前的 dry_run 设置运行。"
    if [[ -n "${COLLECT_METRICS}" ]]; then
      exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" --collect-metrics "${COLLECT_METRICS}"
    fi
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}"
    ;;
  interactive)
    echo "[series-pipeline] 进入交互模式，使用配置文件中的当前 dry_run 设置。"
    if [[ -n "${COLLECT_METRICS}" ]]; then
      exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
        --mode series_pipeline \
        --config "${CONFIG_PATH}" \
        --runner "${RUNNER_PATH}" \
        --run-mode interactive \
        --collect-metrics "${COLLECT_METRICS}"
    fi
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode series_pipeline \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode interactive
    ;;
  preview|dry-run)
    echo "[series-pipeline] 进入交互预览模式，强制 dry_run=true。"
    if [[ -n "${COLLECT_METRICS}" ]]; then
      exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
        --mode series_pipeline \
        --config "${CONFIG_PATH}" \
        --runner "${RUNNER_PATH}" \
        --run-mode preview \
        --collect-metrics "${COLLECT_METRICS}"
    fi
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode series_pipeline \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode preview
    ;;
  run)
    echo "[series-pipeline] 进入交互正式运行模式，强制 dry_run=false。"
    if [[ -n "${COLLECT_METRICS}" ]]; then
      exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
        --mode series_pipeline \
        --config "${CONFIG_PATH}" \
        --runner "${RUNNER_PATH}" \
        --run-mode run \
        --collect-metrics "${COLLECT_METRICS}"
    fi
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode series_pipeline \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode run
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_series_pipeline.sh             # 交互选择视频目录和 ep 范围，按配置里的 dry_run 运行" >&2
    echo "  ./run_series_pipeline.sh preview     # 交互选择视频目录和 ep 范围，只预览不调用模型" >&2
    echo "  ./run_series_pipeline.sh run         # 交互选择视频目录和 ep 范围，正式运行" >&2
    echo "  ./run_series_pipeline.sh config      # 不交互，完全按 config/series_pipeline.local.json 运行" >&2
    echo "  ./run_series_pipeline.sh run true    # 正式运行并强制开启时间/token统计" >&2
    echo "  ./run_series_pipeline.sh run false   # 正式运行并强制关闭时间/token统计" >&2
    exit 2
    ;;
esac
