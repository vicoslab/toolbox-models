#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
git clone --depth 1 https://github.com/jovanavidenovic/DAM4SAM "$TOOLBOX_CACHE/dam4sam"
cd "$TOOLBOX_CACHE/dam4sam"

git apply "$dir"/*.patch

uv venv --python 3.10
uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
uv pip install gunicorn flask /opt/apps/modelargs ffmpeg-python huggingface_hub