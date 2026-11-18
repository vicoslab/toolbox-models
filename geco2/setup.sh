#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/jerpelhan/GECO2 "$INSTALL_DIR"
cp -r "$INSTALL_DIR/Deformable-DETR/models/ops/" "$INSTALL_DIR/models/"

UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/.venv" uv sync
