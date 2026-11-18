#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/jovanavidenovic/DAM4SAM "$INSTALL_DIR"

UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/.venv" uv sync

cd "$INSTALL_DIR"
git apply "$dir"/*.patch
