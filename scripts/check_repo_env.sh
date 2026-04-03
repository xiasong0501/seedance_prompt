#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_OK=0

say() {
  printf '%s\n' "${1-}"
}

ok() {
  say "[OK] $1"
}

warn() {
  say "[MISS] $1"
  STATUS_OK=1
}

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    ok "命令存在：$name"
  else
    warn "命令缺失：$name"
  fi
}

check_file() {
  local path="$1"
  if [[ -e "$path" ]]; then
    ok "文件存在：$path"
  else
    warn "文件缺失：$path"
  fi
}

check_env() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    ok "环境变量已设置：$name"
  else
    warn "环境变量未设置：$name"
  fi
}

for env_file in "${ROOT_DIR}/.env" "${ROOT_DIR}/.env.local" "${ROOT_DIR}/config/.env" "${ROOT_DIR}/config/.env.local"; do
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
done

say "== 基础命令 =="
check_cmd bash
check_cmd python3
if command -v python3.11 >/dev/null 2>&1; then
  ok "命令存在：python3.11"
else
  warn "命令缺失：python3.11（可用 python3 替代，但推荐 3.11）"
fi
if command -v uv >/dev/null 2>&1; then
  ok "命令存在：uv"
else
  warn "命令缺失：uv（bootstrap 会回退到 python -m venv，但推荐安装 uv）"
fi

say
say "== 虚拟环境 =="
check_file "${ROOT_DIR}/.venv/bin/python"

say
say "== 配置文件 =="
check_file "${ROOT_DIR}/config/video_pipeline.local.json"
check_file "${ROOT_DIR}/config/series_pipeline.local.json"
check_file "${ROOT_DIR}/config/openai_agent_flow.local.json"
check_file "${ROOT_DIR}/config/art_assets_pipeline.local.json"
check_file "${ROOT_DIR}/config/explosive_rewrite_pipeline.local.json"
check_file "${ROOT_DIR}/config/nano_banana_assets.local.json"
check_file "${ROOT_DIR}/config/genre_package_audit.local.json"
check_file "${ROOT_DIR}/config/sync_series_learning.local.json"
check_file "${ROOT_DIR}/config/style_transfer_assets.local.json"

say
say "== 通用密钥 =="
check_env OPENAI_API_KEY
check_env GEMINI_API_KEY
check_env DASHSCOPE_API_KEY
check_env ARK_API_KEY

say
say "== TOS / Model Gate =="
if [[ -n "${TOS_ACCESS_KEY_ID:-}" && -n "${TOS_SECRET_ACCESS_KEY:-}" ]]; then
  ok "已设置 TOS_ACCESS_KEY_ID / TOS_SECRET_ACCESS_KEY"
elif [[ -n "${TOS_AK:-}" && -n "${TOS_SK:-}" ]]; then
  ok "已设置 TOS_AK / TOS_SK"
else
  warn "未设置 TOS 凭证（TOS_ACCESS_KEY_ID/TOS_SECRET_ACCESS_KEY 或 TOS_AK/TOS_SK）"
fi

if [[ -n "${MODEL_GATE_ACCESS_KEY:-}" && -n "${MODEL_GATE_SECRET_KEY:-}" ]]; then
  ok "已设置 MODEL_GATE_ACCESS_KEY / MODEL_GATE_SECRET_KEY"
else
  warn "未设置 Model Gate 凭证（MODEL_GATE_ACCESS_KEY / MODEL_GATE_SECRET_KEY）"
fi

say
say "== 结论 =="
if [[ "${STATUS_OK}" -eq 0 ]]; then
  ok "环境检查通过"
else
  warn "环境未完全配齐，请先运行 ./scripts/bootstrap_repo_env.sh 并补全 .env / config/*.local.json"
fi

exit "${STATUS_OK}"
