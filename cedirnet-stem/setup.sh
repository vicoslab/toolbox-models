#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
: "${INSTALL_DIR:?Model definition requires newer toolbox version}"

git clone --depth 1 --branch master --single-branch https://github.com/vicoslab/CeDiRNet-STEM.git "$INSTALL_DIR"

UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/.venv" uv sync

cd "$INSTALL_DIR"
git apply "$dir/0001-python-311-collections.patch"
