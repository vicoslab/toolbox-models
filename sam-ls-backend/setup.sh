#! /usr/bin/env bash
set -euo pipefail

mkdir -p "$TOOLBOX_CACHE/sam-ls-backend"
cd "$TOOLBOX_CACHE/sam-ls-backend"
uv venv --python 3.12
uv pip install /opt/apps/label-studio-ml-backend /opt/apps/modelargs /cache/sam gunicorn einops pycocotools psutil
