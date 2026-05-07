#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
git clone --depth 1 https://github.com/blaz-r/SuperSimpleNet.git "$TOOLBOX_CACHE/super-simple-net"
cd "$TOOLBOX_CACHE/super-simple-net"

git apply "$dir"/*.patch

uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install pytorch_lightning lightning opencv-python kornia tifffile psutil mlflow /opt/apps/modelargs flask gunicorn /opt/apps/label-studio-ml-backend
