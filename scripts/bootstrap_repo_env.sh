#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements-video-pipeline.txt"
ENV_EXAMPLE="${ROOT_DIR}/.env.example"
ENV_FILE="${ROOT_DIR}/.env"

log() {
  printf '[bootstrap] %s\n' "$1"
}

pick_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  return 1
}

ensure_venv() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    log "检测到现有虚拟环境：${VENV_DIR}"
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    local py_bin
    py_bin="$(pick_python)"
    log "使用 uv 创建虚拟环境"
    uv venv "${VENV_DIR}" --python "${py_bin}" --seed
    return
  fi

  local py_bin
  py_bin="$(pick_python)" || {
    log "未找到 python3.11 / python3 / python"
    exit 1
  }
  log "未检测到 uv，回退到 ${py_bin} -m venv"
  "${py_bin}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
}

install_python_requirements() {
  if command -v uv >/dev/null 2>&1; then
    log "使用 uv 安装 Python 依赖"
    uv pip install --python "${VENV_DIR}/bin/python" -r "${REQ_FILE}"
  else
    log "使用 pip 安装 Python 依赖"
    "${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"
  fi
}

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -e "${dst}" ]]; then
    log "已存在，跳过：${dst}"
    return
  fi
  cp -a "${src}" "${dst}"
  log "已创建：${dst}"
}

init_local_configs() {
  copy_if_missing "${ROOT_DIR}/config/video_pipeline.example.json" "${ROOT_DIR}/config/video_pipeline.local.json"
  copy_if_missing "${ROOT_DIR}/config/series_pipeline.example.json" "${ROOT_DIR}/config/series_pipeline.local.json"
  copy_if_missing "${ROOT_DIR}/config/openai_agent_flow.example.json" "${ROOT_DIR}/config/openai_agent_flow.local.json"
  copy_if_missing "${ROOT_DIR}/config/art_assets_pipeline.example.json" "${ROOT_DIR}/config/art_assets_pipeline.local.json"
  copy_if_missing "${ROOT_DIR}/config/explosive_rewrite_pipeline.example.json" "${ROOT_DIR}/config/explosive_rewrite_pipeline.local.json"
  copy_if_missing "${ROOT_DIR}/config/nano_banana_assets.example.json" "${ROOT_DIR}/config/nano_banana_assets.local.json"
  copy_if_missing "${ROOT_DIR}/config/genre_package_audit.example.json" "${ROOT_DIR}/config/genre_package_audit.local.json"
  copy_if_missing "${ROOT_DIR}/config/sync_series_learning.example.json" "${ROOT_DIR}/config/sync_series_learning.local.json"
  copy_if_missing "${ROOT_DIR}/config/style_transfer_assets.example.json" "${ROOT_DIR}/config/style_transfer_assets.local.json"
}

main() {
  log "开始初始化仓库环境"
  ensure_venv
  init_local_configs
  copy_if_missing "${ENV_EXAMPLE}" "${ENV_FILE}"
  install_python_requirements

  log "初始化完成"
  log "下一步请编辑：${ENV_FILE}"
  log "并按需修改：config/*.local.json"
  log "说明文档见：${ROOT_DIR}/ENVIRONMENT_SETUP.md"
}

main "$@"
