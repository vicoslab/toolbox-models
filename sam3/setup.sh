#! /usr/bin/env bash
set -euo pipefail

: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/facebookresearch/sam3.git "$INSTALL_DIR"
cd "$INSTALL_DIR"
curl https://patch-diff.githubusercontent.com/raw/facebookresearch/sam3/pull/403.patch | git apply -

UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/.venv" uv sync --python 3.12
