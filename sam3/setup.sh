#! /usr/bin/env bash
set -euo pipefail

: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/facebookresearch/sam3.git "$INSTALL_DIR"
cd "$INSTALL_DIR"
curl https://patch-diff.githubusercontent.com/raw/facebookresearch/sam3/pull/403.patch | git apply -

uv venv --python 3.12
uv pip install /opt/apps/label-studio-ml-backend . gunicorn einops pycocotools psutil setuptools==79.0.0
uv pip install -e /opt/apps/modelargs # use editable install so it works when docker image updates
