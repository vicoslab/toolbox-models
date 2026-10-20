#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/jovanavidenovic/DAM4SAM "$INSTALL_DIR"
cd "$INSTALL_DIR"

git apply "$dir"/*.patch

uv venv --python 3.10
uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
uv pip install gunicorn flask ffmpeg-python huggingface_hub
uv pip install -e /opt/apps/modelargs # use editable install so it works when docker image updates