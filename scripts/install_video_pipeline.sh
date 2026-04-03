#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements-video-pipeline.txt"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv 未安装，请先确保 uv 可用。"
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/pip" ]; then
  rm -rf "${VENV_DIR}"
  uv venv "${VENV_DIR}" --python 3.11 --seed
fi

uv pip install --python "${VENV_DIR}/bin/python" -r "${REQ_FILE}"

echo "安装完成。"
echo "虚拟环境: ${VENV_DIR}"
echo "ffmpeg 使用 imageio-ffmpeg 打包的本地二进制，不依赖 sudo/apt。"
echo "下一步：编辑 config/video_pipeline.local.json，然后运行 ./run_video_pipeline.sh"
