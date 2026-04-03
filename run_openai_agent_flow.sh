#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/openai_agent_flow.local.json"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUNNER_PATH="${ROOT_DIR}/scripts/run_openai_agent_flow.py"
INTERACTIVE_LAUNCHER="${ROOT_DIR}/scripts/interactive_pipeline_launcher.py"
MODE="${1:-run}"
COLLECT_METRICS="${2:-}"
EXTRA_ARGS=()

if [[ -n "${COLLECT_METRICS}" ]]; then
  EXTRA_ARGS+=(--collect-metrics "${COLLECT_METRICS}")
fi

if [[ ! -e "${CONFIG_PATH}" ]]; then
  echo "[openai-agent-flow] 配置文件不存在：${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[openai-agent-flow] Python 不存在或不可执行：${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER_PATH}" ]]; then
  echo "[openai-agent-flow] 主流程脚本不存在：${RUNNER_PATH}" >&2
  exit 1
fi

if [[ ! -f "${INTERACTIVE_LAUNCHER}" ]]; then
  echo "[openai-agent-flow] 交互启动器不存在：${INTERACTIVE_LAUNCHER}" >&2
  exit 1
fi

read_current_dry_run() {
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

path = Path(r"${CONFIG_PATH}")
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:  # pragma: no cover - shell wrapper fallback
    raise SystemExit(f"[openai-agent-flow] 读取配置失败：{path} ({exc})")
print(str(bool(data.get("runtime", {}).get("dry_run", True))).lower())
PY
}

CURRENT_DRY_RUN="$(read_current_dry_run)"

case "${MODE}" in
  config)
    echo "[openai-agent-flow] 使用配置文件中当前的 dry_run 设置运行（runtime.dry_run=${CURRENT_DRY_RUN}）。"
    exec "${PYTHON_BIN}" "${RUNNER_PATH}" --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
    ;;
  interactive)
    echo "[openai-agent-flow] 进入交互模式，沿用配置文件中的 dry_run 设置（runtime.dry_run=${CURRENT_DRY_RUN}）。"
    echo "[openai-agent-flow] 如果要强制正式运行，请使用：./run_openai_agent_flow.sh run"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode openai_flow \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode interactive \
      "${EXTRA_ARGS[@]}"
    ;;
  preview|dry-run)
    echo "[openai-agent-flow] 进入交互预览模式，强制 dry_run=true。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode openai_flow \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode preview \
      "${EXTRA_ARGS[@]}"
    ;;
  run)
    echo "[openai-agent-flow] 进入交互正式运行模式，强制 dry_run=false。"
    exec "${PYTHON_BIN}" "${INTERACTIVE_LAUNCHER}" \
      --mode openai_flow \
      --config "${CONFIG_PATH}" \
      --runner "${RUNNER_PATH}" \
      --run-mode run \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "用法：" >&2
    echo "  ./run_openai_agent_flow.sh                   # 默认等同于 run：交互选择剧本和 ep 范围，正式运行" >&2
    echo "  ./run_openai_agent_flow.sh preview           # 交互选择剧本和 ep 范围，只预览不调用模型" >&2
    echo "  ./run_openai_agent_flow.sh run               # 交互选择剧本和 ep 范围，正式运行" >&2
    echo "  ./run_openai_agent_flow.sh interactive       # 交互选择剧本和 ep 范围，沿用配置里的 dry_run 运行（当前=${CURRENT_DRY_RUN}）" >&2
    echo "  ./run_openai_agent_flow.sh run true          # 交互正式运行，并强制开启统计" >&2
    echo "  ./run_openai_agent_flow.sh run false         # 交互正式运行，并强制关闭统计" >&2
    echo "  ./run_openai_agent_flow.sh config            # 不交互，完全按 config/openai_agent_flow.local.json 运行" >&2
    echo "  ./run_openai_agent_flow.sh config true       # 不交互，完全按 config 运行，并强制开启统计" >&2
    exit 2
    ;;
esac
